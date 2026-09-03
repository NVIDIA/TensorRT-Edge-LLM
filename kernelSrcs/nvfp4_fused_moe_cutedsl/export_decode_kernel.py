#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AOT export script for the fused decode MoE kernel.

Exports a pointer-based JIT wrapper around ``MoEDecodeKernel`` that
constructs CuTe tensors from raw pointers and runtime ``Int32``
dimensions, making the exported C entry point shape-agnostic for
num_tokens / max_rows / state_E / weight_E / K / N / top_k.  The MMA
N-tile (``mma_tiler_n``) remains compile-time and is selected at build
time via the ``--mma_tiler_n`` flag (n128 or n256 variants).

The decode kernel fuses route/pack + FC1 + activation + quantize +
FC2 + scatter in a single resident-grid launch with a barrier between
phases.  Best for small routed working sets (decode / low token counts).

Usage (from kernelSrcs/):
    python nvfp4_fused_moe_cutedsl/export_decode_kernel.py \\
        --activation swiglu \\
        --mma_tiler_n 128 \\
        --output_dir /tmp/staging \\
        --file_name nvfp4_fused_moe_decode_swiglu_n128 \\
        --function_prefix nvfp4_fused_moe_decode_swiglu_n128

Usage (invoked by build_cutedsl.py — PYTHONPATH set automatically).
"""

import argparse
import os
import sys

import cupy as cp


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


_NVFP4_BLOCK_SIZE = 16

# Dummy problem shape for tracing.  The wrapper accepts runtime Int32
# dimensions (K, N, state_E, weight_E, num_topk, num_tokens, max_rows)
# so these exact values do NOT restrict the caller -- they are only used
# to seed ``cute.compile`` with well-typed placeholder scalars.
_DUMMY_STATE_E = 8
_DUMMY_WEIGHT_E = 8
_DUMMY_M = 64
_DUMMY_K = 2048
_DUMMY_N = 1024
_DUMMY_NUM_TOPK = 8
_DUMMY_MAX_ROWS = 128


def export_decode_moe_variant(args):
    """Export a single fused decode MoE kernel variant."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from cute_dsl_utils import cute_compile_options, get_num_sm, make_ptr
    from moe_decode_kernel import MoEDecodeKernel

    cp.cuda.Device(0).use()

    activation = args.activation
    verbose = getattr(args, "verbose", False)
    is_gated = activation in ("swiglu", "geglu")
    # v1 Nvfp4MoePlugin consumes FP16 hidden states. BF16 is intentionally
    # deferred; when it returns, add --io_dtype {bf16,fp16} here and in the
    # sibling prefill export script.
    io_dtype = cutlass.Float16

    sf_vec_size = 16
    # MMA tile M and N are both compile-time axes (m64/m128 x n128/n256
    # variants); the scheduler only requires M to be a multiple of 64.
    # Shape axes H/I/E/top_k are runtime via the wrapper below.
    mma_tiler_mn = (args.mma_tiler_m, args.mma_tiler_n)

    print(f"Fused decode MoE variant: activation={activation}, "
          f"io_dtype=fp16, mma_tiler_mn={mma_tiler_mn}")

    kernel = MoEDecodeKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        activation=activation,
        input_scales_are_reciprocal=False,
        fast_math=True,
        io_dtype=io_dtype,
    )

    # ---- Build the pointer-based wrapper ----
    class _DecodeMoELaunch:
        """Thin JIT wrapper — accepts pointers + runtime Int32 scalars,
        constructs CuTe tensors internally, delegates to MoEDecodeKernel.

        All shape axes (K, N, state_E, weight_E, num_topk, num_tokens,
        max_rows) are runtime Int32 arguments. The MMA N-tile is
        captured at build time via ``mma_tiler_mn`` (outer closure) and
        is thus compile-time per variant (n128 / n256).
        """

        def __init__(self, kern, is_gated):
            self._kernel = kern
            self._is_gated = is_gated

        @cute.jit
        def __call__(
            self,
            a_input_ptr: cute.Pointer,
            topk_ids_ptr: cute.Pointer,
            topk_weights_ptr: cute.Pointer,
            packed_a_ptr: cute.Pointer,
            sfa_ptr: cute.Pointer,
            packed_a_storage_ptr: cute.Pointer,
            scale_storage_ptr: cute.Pointer,
            barrier_count: cute.Tensor,
            barrier_epoch: cute.Tensor,
            b_w13_ptr: cute.Pointer,
            sfb_w13_ptr: cute.Pointer,
            b_down_ptr: cute.Pointer,
            sfb_down_ptr: cute.Pointer,
            row_counts_ptr: cute.Pointer,
            active_expert_count: cute.Tensor,
            weight_expert_ids_ptr: cute.Pointer,
            global_to_local_expert_ptr: cute.Pointer,
            input_gs_ptr: cute.Pointer,
            alpha_ptr: cute.Pointer,
            down_alpha_ptr: cute.Pointer,
            global_scale_ptr: cute.Pointer,
            scatter_ptr: cute.Pointer,
            token_map_ptr: cute.Pointer,
            token_weights_ptr: cute.Pointer,
            # Runtime scalar dimensions
            num_tokens: cutlass.Int32,
            max_rows: cutlass.Int32,
            state_E: cutlass.Int32,
            weight_E: cutlass.Int32,
            rows_padded: cutlass.Int32,
            K: cutlass.Int32,
            N: cutlass.Int32,
            num_topk: cutlass.Int32,
            max_active_clusters: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            half_k = K // cutlass.Int32(2)
            # cols_pad_k = ceil_up(K / _NVFP4_BLOCK_SIZE, 4) computed as Int32 arith.
            sf_cols = K // cutlass.Int32(_NVFP4_BLOCK_SIZE)
            cols_pad_k = (
                (sf_cols + cutlass.Int32(3)) // cutlass.Int32(4)
            ) * cutlass.Int32(4)
            # w1_n_dim is N for non-gated, 2*N for gated (swiglu/geglu).
            # Initialize before the branch — CuTe DSL treats the Python `if`
            # as dynamic control flow and errors on "name defined only in one
            # arm" (see export_prefill_kernel.py for the same fix).
            w1_n_dim = N
            if self._is_gated:
                w1_n_dim = N * cutlass.Int32(2)

            a_input = cute.make_tensor(
                a_input_ptr,
                layout=cute.make_layout(
                    (num_tokens, K), stride=(K, 1)),
            )
            topk_ids = cute.make_tensor(
                topk_ids_ptr,
                layout=cute.make_layout(
                    (num_tokens * num_topk,), stride=(1,)),
            )
            topk_weights_t = cute.make_tensor(
                topk_weights_ptr,
                layout=cute.make_layout(
                    (num_tokens * num_topk,), stride=(1,)),
            )
            packed_a = cute.make_tensor(
                packed_a_ptr,
                layout=cute.make_layout(
                    (max_rows, K, state_E),
                    stride=(K, 1, max_rows * K),
                ),
            )
            packed_a_storage = cute.make_tensor(
                packed_a_storage_ptr,
                layout=cute.make_layout(
                    (state_E * max_rows * half_k,), stride=(1,)),
            )
            scale_storage = cute.make_tensor(
                scale_storage_ptr,
                layout=cute.make_layout(
                    (state_E * rows_padded * cols_pad_k,), stride=(1,)),
            )
            scatter_output = cute.make_tensor(
                scatter_ptr,
                layout=cute.make_layout(
                    (num_tokens, K), stride=(K, 1)),
            )
            # Weight tensors: pointer + runtime layout. stride_order=(1,0,2)
            # in the legacy ``make_fake_compact_tensor`` corresponds to
            # ``make_ordered_layout(..., order=(1,0,2))``: dim-1 (K) is
            # innermost, dim-0 (N) is next, dim-2 (E) is outermost.
            b_w13 = cute.make_tensor(
                b_w13_ptr,
                layout=cute.make_ordered_layout(
                    (w1_n_dim, K, weight_E), order=(1, 0, 2)),
            )
            b_down = cute.make_tensor(
                b_down_ptr,
                layout=cute.make_ordered_layout(
                    (K, N, weight_E), order=(1, 0, 2)),
            )
            row_counts = cute.make_tensor(
                row_counts_ptr,
                layout=cute.make_layout((state_E,), stride=(1,)),
            )
            weight_expert_ids = cute.make_tensor(
                weight_expert_ids_ptr,
                layout=cute.make_layout((state_E,), stride=(1,)),
            )
            global_to_local_expert = cute.make_tensor(
                global_to_local_expert_ptr,
                layout=cute.make_layout((weight_E,), stride=(1,)),
            )
            input_gs = cute.make_tensor(
                input_gs_ptr,
                layout=cute.make_layout((weight_E,), stride=(1,)),
            )
            alpha = cute.make_tensor(
                alpha_ptr,
                layout=cute.make_layout((weight_E,), stride=(1,)),
            )
            down_alpha = cute.make_tensor(
                down_alpha_ptr,
                layout=cute.make_layout((weight_E,), stride=(1,)),
            )
            global_scale = cute.make_tensor(
                global_scale_ptr,
                layout=cute.make_layout((weight_E,), stride=(1,)),
            )
            token_map = cute.make_tensor(
                token_map_ptr,
                layout=cute.make_layout(
                    (state_E, max_rows), stride=(max_rows, 1)),
            )
            token_weights = cute.make_tensor(
                token_weights_ptr,
                layout=cute.make_layout(
                    (state_E, max_rows), stride=(max_rows, 1)),
            )

            self._kernel(
                a_input,
                topk_ids,
                topk_weights_t,
                packed_a,
                sfa_ptr,
                packed_a_storage,
                scale_storage,
                barrier_count,
                barrier_epoch,
                b_w13,
                sfb_w13_ptr,
                b_down,
                sfb_down_ptr,
                row_counts,
                active_expert_count,
                weight_expert_ids,
                global_to_local_expert,
                input_gs,
                alpha,
                down_alpha,
                global_scale,
                scatter_output,
                token_map,
                token_weights,
                max_active_clusters=max_active_clusters,
                stream=stream,
            )

    launch = _DecodeMoELaunch(kernel, is_gated=is_gated)

    # max_active_clusters is a RUNTIME wrapper argument (cluster_size=1, so it
    # equals the SM count of the GPU the kernel launches on). The deployed
    # caller passes cudaDevAttrMultiProcessorCount at launch time; the trace
    # value below is a placeholder. This removes the build-GPU dependency the
    # old constexpr bake had (and the HardwareInfo/GB10 sm_120a pitfall).
    max_active_clusters = cutlass.Int32(get_num_sm())

    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    a_dtype = io_dtype
    alpha_dtype = cutlass.Float32

    rows_pad_k = _align_up(_DUMMY_MAX_ROWS, 128)

    # Runtime-shaped tensors → pointers
    a_input_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    topk_ids_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    topk_weights_fake = make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4)
    packed_a_fake = make_ptr(ab_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
    scale_storage_fake = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)

    # Scalar / [1] tensors
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    active_expert_count_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)

    # Weight tensors: pointer path (shape-polymorphic). The wrapper builds
    # the cute.Tensor with a runtime ``make_ordered_layout`` so the AOT
    # binary's TMA descriptor is rebuilt at launch from runtime N/K/E.
    b_w13_fake = make_ptr(ab_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = make_ptr(ab_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)

    # Runtime-shaped metadata → pointers
    row_counts_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    weight_expert_ids_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    global_to_local_expert_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    input_gs_fake = make_ptr(alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    alpha_fake = make_ptr(alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    down_alpha_fake = make_ptr(alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    global_scale_fake = make_ptr(alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    scatter_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    token_map_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    token_weights_fake = make_ptr(alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    print("Compiling fused decode MoE kernel via wrapper...")
    compiled = cute.compile(
        launch,
        # Pointers
        a_input_fake,
        topk_ids_fake,
        topk_weights_fake,
        packed_a_fake,
        sfa_fake,
        packed_a_storage_fake,
        scale_storage_fake,
        # Tensors
        barrier_count_fake,
        barrier_epoch_fake,
        b_w13_fake,
        sfb_w13_fake,
        b_down_fake,
        sfb_down_fake,
        # Pointers
        row_counts_fake,
        active_expert_count_fake,
        weight_expert_ids_fake,
        global_to_local_expert_fake,
        input_gs_fake,
        alpha_fake,
        down_alpha_fake,
        global_scale_fake,
        scatter_fake,
        token_map_fake,
        token_weights_fake,
        # Runtime Int32 placeholders
        _DUMMY_M,
        _DUMMY_MAX_ROWS,
        _DUMMY_STATE_E,
        _DUMMY_WEIGHT_E,
        rows_pad_k,
        _DUMMY_K,
        _DUMMY_N,
        _DUMMY_NUM_TOPK,
        # Runtime persistent-grid size
        max_active_clusters,
        stream,
        options=cute_compile_options(),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Exporting to {args.output_dir}/{args.file_name}.[h|o]")
    compiled.export_to_c(
        file_path=args.output_dir,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
    )

    header = os.path.join(args.output_dir, f"{args.file_name}.h")
    obj = os.path.join(args.output_dir, f"{args.file_name}.o")
    ok = True
    for path, label in [(header, "Header"), (obj, "Object")]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            ok = False
        elif os.path.getsize(path) == 0:
            print(f"ERROR: {label} is empty: {path}", file=sys.stderr)
            ok = False
    if not ok:
        sys.exit(1)

    print(f"  header: {os.path.getsize(header)} bytes")
    print(f"  object: {os.path.getsize(obj)} bytes")

    if verbose:
        print("\n--- Generated Header (first 100 lines) ---")
        with open(header) as f:
            for i, line in enumerate(f):
                if i >= 100:
                    print("  ... (truncated)")
                    break
                print(f"  {line.rstrip()}")

    return header, obj


def main():
    parser = argparse.ArgumentParser(
        description="Export fused decode MoE kernel for AOT compilation"
    )
    parser.add_argument(
        "--activation", type=str, default="swiglu",
        choices=["identity", "silu", "swiglu", "gelu", "geglu", "relu2"],
        help="Activation function (default: swiglu)"
    )
    parser.add_argument(
        "--mma_tiler_n", type=int, default=128,
        choices=[128, 256],
        help="Compile-time MMA N-tile size (n128 or n256 variants)"
    )
    parser.add_argument(
        "--mma_tiler_m", type=int, default=128,
        choices=[64, 128],
        help="Compile-time MMA M-tile size. Decode routes few rows per expert "
             "(top_k tokens), so m64 wastes less of the tile; the scheduler "
             "requires a multiple of 64."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write .h and .o files"
    )
    parser.add_argument(
        "--file_name", type=str, required=True,
        help="Base name for output files (without extension)"
    )
    parser.add_argument(
        "--function_prefix", type=str, required=True,
        help="C function name prefix"
    )
    parser.add_argument(
        "--export_only", action="store_true",
        help="Accepted for build_cutedsl.py compatibility (no-op)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print generated header content"
    )
    args = parser.parse_args()

    export_decode_moe_variant(args)
    print("\nFused decode MoE kernel export completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
