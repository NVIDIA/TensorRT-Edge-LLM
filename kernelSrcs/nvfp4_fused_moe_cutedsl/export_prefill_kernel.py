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
"""AOT export script for the fused prefill MoE kernel.

Exports ``_PrefillMoELaunch.__call__`` -- a pointer-based JIT wrapper
around ``MoEPrefillKernel`` that constructs CuTe tensors from raw
pointers and runtime ``Int32`` dimensions.  All shape axes
(K, N, E, num_topk, num_tokens, max_rows) are runtime; only the MMA
N-tile is a compile-time variant axis (n128 / n256), selected via the
``--mma_tiler_n`` flag.

The prefill kernel fuses route/pack + FC1 + activation + quantize +
FC2 + scatter using a global task queue for producer/consumer overlap.
Best for large routed working sets (prefill / high token counts).

Usage (from kernelSrcs/):
    python nvfp4_fused_moe_cutedsl/export_prefill_kernel.py \\
        --activation swiglu \\
        --mma_tiler_n 128 \\
        --output_dir /tmp/staging \\
        --file_name nvfp4_fused_moe_prefill_swiglu_n128 \\
        --function_prefix nvfp4_fused_moe_prefill_swiglu_n128

Usage (invoked by build_cutedsl.py -- PYTHONPATH set automatically).
"""

import argparse
import os
import sys

def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


_NVFP4_BLOCK_SIZE = 16
_LEVEL_TILE_M = 128

# Dummy problem shape for tracing.  All values below are runtime Int32
# arguments to the exported wrapper, so the AOT binary is not
# specialized to them.
_DUMMY_E = 8
_DUMMY_M = 256
_DUMMY_K = 2048
_DUMMY_N = 1024
_DUMMY_NUM_TOPK = 8
_DUMMY_MAX_TASKS = 64
_DUMMY_MAX_PHYS_TILES = 64


def export_prefill_moe_variant(args):
    """Export a single fused prefill MoE kernel variant."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from cutedsl_utils import aot_placeholders
    from moe_prefill_kernel import MoEPrefillKernel

    activation = args.activation
    verbose = getattr(args, "verbose", False)
    is_gated = activation in ("swiglu", "geglu")
    # v1 Nvfp4MoePlugin consumes FP16 hidden states. BF16 is intentionally
    # deferred; when it returns, add --io_dtype {bf16,fp16} here and in the
    # sibling decode export script.
    io_dtype = cutlass.Float16

    sf_vec_size = 16
    # MMA tile N is a compile-time axis (n128 or n256 variants); M is fixed
    # at _LEVEL_TILE_M. All other shape axes (K/N/E/top_k) are runtime.
    mma_tiler_mn = (_LEVEL_TILE_M, args.mma_tiler_n)

    print(f"Fused prefill MoE variant: activation={activation}, "
          f"io_dtype=fp16, mma_tiler_mn={mma_tiler_mn}")

    kernel = MoEPrefillKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        activation=activation,
        input_scales_are_reciprocal=False,
        fast_math=True,
        io_dtype=io_dtype,
    )

    # ---- Wrapper class (adapted from moe_dispatch._PrefillMoELaunch) ----
    class _PrefillMoELaunch:
        """Thin JIT wrapper -- accepts pointers + runtime Int32 scalars.

        All shape axes (K, N, state_E/weight_E, num_topk, num_tokens,
        rows_padded, max_tasks, max_phys_tiles) are runtime Int32. Only
        the MMA N-tile is captured via ``mma_tiler_mn`` (outer closure)
        and is thus compile-time per variant (n128 / n256).
        """

        def __init__(self, kern, is_gated):
            self._kernel = kern
            self._is_gated = is_gated

        @cute.jit
        def __call__(
            self,
            a_ptr: cute.Pointer,
            topk_ids_ptr: cute.Pointer,
            topk_weights_ptr: cute.Pointer,
            packed_a_ptr: cute.Pointer,
            sfa_ptr: cute.Pointer,
            packed_a_storage_ptr: cute.Pointer,
            scale_storage_ptr: cute.Pointer,
            barrier_count: cute.Tensor,
            barrier_epoch: cute.Tensor,
            pair_head: cute.Tensor,
            producers_done_count: cute.Tensor,
            all_work_published: cute.Tensor,
            task_head: cute.Tensor,
            task_tail: cute.Tensor,
            task_ready_ptr: cute.Pointer,
            task_expert_ptr: cute.Pointer,
            task_m_tile_ptr: cute.Pointer,
            task_slice_begin_ptr: cute.Pointer,
            task_slice_count_ptr: cute.Pointer,
            task_valid_rows_ptr: cute.Pointer,
            tile_write_count_ptr: cute.Pointer,
            b_w13_ptr: cute.Pointer,
            sfb_w13_ptr: cute.Pointer,
            b_down_ptr: cute.Pointer,
            sfb_down_ptr: cute.Pointer,
            row_counts_ptr: cute.Pointer,
            expert_write_rows_ptr: cute.Pointer,
            expert_tile_base_ptr: cute.Pointer,
            input_gs_ptr: cute.Pointer,
            alpha_ptr: cute.Pointer,
            down_alpha_ptr: cute.Pointer,
            global_scale_ptr: cute.Pointer,
            scatter_ptr: cute.Pointer,
            token_map_ptr: cute.Pointer,
            token_weights_ptr: cute.Pointer,
            num_tokens: cutlass.Int32,
            max_rows: cutlass.Int32,
            rows_padded: cutlass.Int32,
            max_tasks: cutlass.Int32,
            max_phys_tiles: cutlass.Int32,
            K: cutlass.Int32,
            N: cutlass.Int32,
            weight_E: cutlass.Int32,
            num_topk: cutlass.Int32,
            max_active_clusters: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            half_k = K // cutlass.Int32(2)
            sf_cols = K // cutlass.Int32(_NVFP4_BLOCK_SIZE)
            cols_pad_k = (
                (sf_cols + cutlass.Int32(3)) // cutlass.Int32(4)
            ) * cutlass.Int32(4)
            # Initialize before the branch — CuTe DSL treats the if as dynamic
            # control flow and forbids defining a name only inside one arm.
            w1_n_dim = N
            if self._is_gated:
                w1_n_dim = N * cutlass.Int32(2)

            a_input = cute.make_tensor(
                a_ptr,
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
            scatter_output = cute.make_tensor(
                scatter_ptr,
                layout=cute.make_layout(
                    (num_tokens, K), stride=(K, 1)),
            )
            packed_a = cute.make_tensor(
                packed_a_ptr,
                layout=cute.make_layout(
                    (rows_padded, K, 1),
                    stride=(K, 1, rows_padded * K),
                ),
            )
            packed_a_storage = cute.make_tensor(
                packed_a_storage_ptr,
                layout=cute.make_layout(
                    (rows_padded * half_k,), stride=(1,)),
            )
            scale_storage = cute.make_tensor(
                scale_storage_ptr,
                layout=cute.make_layout(
                    (rows_padded * cols_pad_k,), stride=(1,)),
            )
            token_map = cute.make_tensor(
                token_map_ptr,
                layout=cute.make_layout((rows_padded,), stride=(1,)),
            )
            token_weights_t = cute.make_tensor(
                token_weights_ptr,
                layout=cute.make_layout((rows_padded,), stride=(1,)),
            )
            task_ready = cute.make_tensor(
                task_ready_ptr,
                layout=cute.make_layout((max_tasks,), stride=(1,)),
            )
            task_expert = cute.make_tensor(
                task_expert_ptr,
                layout=cute.make_layout((max_tasks,), stride=(1,)),
            )
            task_m_tile = cute.make_tensor(
                task_m_tile_ptr,
                layout=cute.make_layout((max_tasks,), stride=(1,)),
            )
            task_slice_begin = cute.make_tensor(
                task_slice_begin_ptr,
                layout=cute.make_layout((max_tasks,), stride=(1,)),
            )
            task_slice_count = cute.make_tensor(
                task_slice_count_ptr,
                layout=cute.make_layout((max_tasks,), stride=(1,)),
            )
            task_valid_rows = cute.make_tensor(
                task_valid_rows_ptr,
                layout=cute.make_layout((max_tasks,), stride=(1,)),
            )
            tile_write_count = cute.make_tensor(
                tile_write_count_ptr,
                layout=cute.make_layout((max_phys_tiles,), stride=(1,)),
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
            # Per-expert metadata: pointer + runtime (E,) layout.
            row_counts = cute.make_tensor(
                row_counts_ptr,
                layout=cute.make_layout((weight_E,), stride=(1,)),
            )
            expert_write_rows = cute.make_tensor(
                expert_write_rows_ptr,
                layout=cute.make_layout((weight_E,), stride=(1,)),
            )
            expert_tile_base = cute.make_tensor(
                expert_tile_base_ptr,
                layout=cute.make_layout(
                    (weight_E + cutlass.Int32(1),), stride=(1,)),
            )
            input_global_scale = cute.make_tensor(
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
                pair_head,
                producers_done_count,
                all_work_published,
                task_head,
                task_tail,
                task_ready,
                task_expert,
                task_m_tile,
                task_slice_begin,
                task_slice_count,
                task_valid_rows,
                tile_write_count,
                b_w13,
                sfb_w13_ptr,
                b_down,
                sfb_down_ptr,
                row_counts,
                expert_write_rows,
                expert_tile_base,
                input_global_scale,
                alpha,
                down_alpha,
                global_scale,
                scatter_output,
                token_map,
                token_weights_t,
                max_active_clusters=max_active_clusters,
                stream=stream,
            )

    launch = _PrefillMoELaunch(kernel, is_gated=is_gated)

    # max_active_clusters is a RUNTIME wrapper argument (cluster_size=1, so it
    # equals the SM count of the GPU the kernel launches on). The deployed
    # caller passes cudaDevAttrMultiProcessorCount at launch time; the trace
    # value below is a placeholder. This removes the build-GPU dependency the
    # old constexpr bake had.
    max_active_clusters = aot_placeholders.runtime_int32()

    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    a_dtype = io_dtype
    alpha_dtype = cutlass.Float32

    # Runtime-shaped tensors → pointers
    a_input_fake = aot_placeholders.make_ptr(a_dtype, assumed_align=16)
    topk_ids_fake = aot_placeholders.make_ptr(cutlass.Int32, assumed_align=4)
    topk_weights_fake = aot_placeholders.make_ptr(cutlass.Float32, assumed_align=4)
    packed_a_fake = aot_placeholders.make_ptr(ab_dtype, assumed_align=16)
    sfa_fake = aot_placeholders.make_ptr(sf_dtype, assumed_align=16)
    packed_a_storage_fake = aot_placeholders.make_ptr(cutlass.Uint8, assumed_align=16)
    scale_storage_fake = aot_placeholders.make_ptr(cutlass.Uint8, assumed_align=16)

    # Scalar / [1] tensors
    barrier_count_fake = aot_placeholders.make_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    barrier_epoch_fake = aot_placeholders.make_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    pair_head_fake = aot_placeholders.make_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    producers_done_count_fake = aot_placeholders.make_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    all_work_published_fake = aot_placeholders.make_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    task_head_fake = aot_placeholders.make_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    task_tail_fake = aot_placeholders.make_compact_tensor(cutlass.Int32, (1,), assumed_align=4)

    # Prefill task queue → pointers
    task_ready_fake = aot_placeholders.make_ptr(cutlass.Int32, assumed_align=4)
    task_expert_fake = aot_placeholders.make_ptr(cutlass.Int32, assumed_align=4)
    task_m_tile_fake = aot_placeholders.make_ptr(cutlass.Int32, assumed_align=4)
    task_slice_begin_fake = aot_placeholders.make_ptr(cutlass.Int32, assumed_align=4)
    task_slice_count_fake = aot_placeholders.make_ptr(cutlass.Int32, assumed_align=4)
    task_valid_rows_fake = aot_placeholders.make_ptr(cutlass.Int32, assumed_align=4)
    tile_write_count_fake = aot_placeholders.make_ptr(cutlass.Int32, assumed_align=4)

    # Weight tensors: pointer path (shape-polymorphic). The wrapper builds
    # the cute.Tensor with a runtime ``make_ordered_layout`` so the AOT
    # binary's TMA descriptor is rebuilt at launch from runtime N/K/E.
    b_w13_fake = aot_placeholders.make_ptr(ab_dtype, assumed_align=16)
    sfb_w13_fake = aot_placeholders.make_ptr(sf_dtype, assumed_align=16)
    b_down_fake = aot_placeholders.make_ptr(ab_dtype, assumed_align=16)
    sfb_down_fake = aot_placeholders.make_ptr(sf_dtype, assumed_align=16)

    # Per-expert metadata: pointer path (runtime E). These were
    # previously baked via ``make_fake_compact_tensor(..., (_DUMMY_E,))``
    # which pinned the number of experts at compile time; switching to
    # pointers + runtime ``make_layout((weight_E,))`` inside the JIT
    # removes that restriction.
    row_counts_fake = aot_placeholders.make_ptr(cutlass.Int32, assumed_align=4)
    expert_write_rows_fake = aot_placeholders.make_ptr(cutlass.Int32, assumed_align=4)
    expert_tile_base_fake = aot_placeholders.make_ptr(cutlass.Int32, assumed_align=4)
    input_gs_fake = aot_placeholders.make_ptr(alpha_dtype, assumed_align=16)
    alpha_fake = aot_placeholders.make_ptr(alpha_dtype, assumed_align=16)
    down_alpha_fake = aot_placeholders.make_ptr(alpha_dtype, assumed_align=16)
    global_scale_fake = aot_placeholders.make_ptr(alpha_dtype, assumed_align=16)

    scatter_fake = aot_placeholders.make_ptr(a_dtype, assumed_align=16)
    token_map_fake = aot_placeholders.make_ptr(cutlass.Int32, assumed_align=4)
    token_weights_fake = aot_placeholders.make_ptr(alpha_dtype, assumed_align=16)

    stream = aot_placeholders.make_stream()

    print("Compiling fused prefill MoE kernel via wrapper...")
    compiled = cute.compile(
        launch,
        a_input_fake,
        topk_ids_fake,
        topk_weights_fake,
        packed_a_fake,
        sfa_fake,
        packed_a_storage_fake,
        scale_storage_fake,
        barrier_count_fake,
        barrier_epoch_fake,
        pair_head_fake,
        producers_done_count_fake,
        all_work_published_fake,
        task_head_fake,
        task_tail_fake,
        task_ready_fake,
        task_expert_fake,
        task_m_tile_fake,
        task_slice_begin_fake,
        task_slice_count_fake,
        task_valid_rows_fake,
        tile_write_count_fake,
        b_w13_fake,
        sfb_w13_fake,
        b_down_fake,
        sfb_down_fake,
        row_counts_fake,
        expert_write_rows_fake,
        expert_tile_base_fake,
        input_gs_fake,
        alpha_fake,
        down_alpha_fake,
        global_scale_fake,
        scatter_fake,
        token_map_fake,
        token_weights_fake,
        # Runtime Int32 placeholders (must match the order of the JIT
        # ``__call__`` signature: num_tokens, max_rows, rows_padded,
        # max_tasks, max_phys_tiles, K, N, weight_E, num_topk).
        _DUMMY_M,
        _DUMMY_M,
        _DUMMY_M,
        _DUMMY_MAX_TASKS,
        _DUMMY_MAX_PHYS_TILES,
        _DUMMY_K,
        _DUMMY_N,
        _DUMMY_E,
        _DUMMY_NUM_TOPK,
        # Runtime persistent-grid size
        max_active_clusters,
        stream,
        options=aot_placeholders.compile_options("--opt-level 2"),
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
        description="Export fused prefill MoE kernel for AOT compilation"
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

    export_prefill_moe_variant(args)
    print("\nFused prefill MoE kernel export completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
