#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edge-LLM CuTe DSL SM120/121 dense NVFP4 GEMM with FP16 output.

The kernel keeps the production GEMM contract used by
``CuteDslGemmNvFp4Runner``: packed NVFP4 A/B tensors, SfAtom-layout FP8 scale
factors, an FP32 output scale, persistent tile scheduling, a dedicated TMA
warp, and FP16 output.  The decode-only ``raw_sfa`` specialization accepts
the row-major activation-scale row emitted by TensorRT and stages it directly
into the kernel's shared-memory SfAtom layout.
SM121 executes warp-level ``MmaMXF4NVF4Op`` instructions because tcgen05 and
TMEM are unavailable on this architecture.
"""

import argparse
import os
import sys
import time
from typing import Tuple

_KERNEL_SRCS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _KERNEL_SRCS)
sys.path.insert(0, os.path.join(_KERNEL_SRCS, "nvfp4_fused_moe_cutedsl"))

_saved_argv = list(sys.argv) if __name__ == "__main__" else None
if __name__ == "__main__":
    sys.argv = [sys.argv[0]]

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute

from gemm_blackwell_nvfp4_ws import _create_nvfp4_pointers
from common import export_compiled_kernel, parse_comma_separated_ints
from cutedsl_utils import aot_placeholders
from dense_gemm_sm120 import DenseGemmKernel


class GemmBlackwellGeForceNvFp4(DenseGemmKernel):
    """Pointer-based Edge-LLM AOT adapter for the SM12x dense GEMM."""

    def __init__(self, *, raw_sfa: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.raw_sfa = raw_sfa

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        scale_k = k // 16
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, 1), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(
            sfa_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, (m + 127) // 128, 4, scale_k // 4, 1),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        sfb = cute.make_tensor(
            sfb_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, (n + 127) // 128, 4, scale_k // 4, 1),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        c = cute.make_tensor(
            c_ptr, layout=cute.make_ordered_layout((m, n, 1), order=(1, 0, 2))
        )
        alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((1,)))
        raw_sfa = None
        if cutlass.const_expr(self.raw_sfa):
            raw_sfa = cute.make_tensor(
                cute.recast_ptr(sfa_ptr, dtype=cutlass.Uint8),
                layout=cute.make_ordered_layout(
                    (m, scale_k, 1), order=(1, 0, 2)
                ),
            )
        return self(
            a,
            b,
            sfa,
            sfb,
            c,
            alpha,
            max_active_clusters,
            stream,
            raw_sfa=raw_sfa,
        )


def _decode_fp8_e4m3(values):
    """Decode positive finite E4M3 bytes into FP32 values."""
    exponent = ((values >> 3) & 0xF).astype(cp.int32)
    mantissa = (values & 0x7).astype(cp.float32)
    normal = (1.0 + mantissa / 8.0) * cp.exp2(exponent - 7)
    subnormal = mantissa / 8.0 * (2.0**-6)
    magnitude = cp.where(exponent == 0, subnormal, normal)
    return cp.where((values & 0x80) != 0, -magnitude, magnitude)


def _swizzle_scale_factors(scale_linear, rows: int, k: int):
    """Convert row-major FP8 block scales to the production SfAtom layout."""
    scale_k = k // 16
    row_tiles = (rows + 127) // 128
    k_tiles = (scale_k + 3) // 4
    padded = cp.zeros((row_tiles * 128, k_tiles * 4), dtype=cp.uint8)
    padded[:rows, :scale_k] = scale_linear
    return (
        padded.reshape(row_tiles, 4, 32, k_tiles, 4)
        .transpose(0, 3, 2, 1, 4)
        .copy()
        .reshape(-1)
    )


def _make_reference_operand(rows: int, k: int, rng):
    """Create deterministic packed NVFP4 data and finite E4M3 scales."""
    codes = rng.randint(0, 16, size=(rows, k)).astype(cp.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scale_choices = cp.asarray((0x28, 0x30, 0x38, 0x40), dtype=cp.uint8)
    scale_linear = scale_choices[
        rng.randint(0, len(scale_choices), size=(rows, k // 16))
    ]
    return packed, scale_linear


def _dequantize_reference(packed, scale_linear):
    """Dequantize packed E2M1 values with row-major E4M3 block scales."""
    rows, packed_k = packed.shape
    k = packed_k * 2
    codes = cp.empty((rows, k), dtype=cp.uint8)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    magnitudes = cp.asarray(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=cp.float32
    )
    values = magnitudes[codes & 0x7]
    values = cp.where((codes & 0x8) != 0, -values, values)
    scales = _decode_fp8_e4m3(scale_linear)
    return (
        values.reshape(rows, k // 16, 16) * scales[..., None]
    ).reshape(rows, k)


def run(
    mnk: Tuple[int, int, int],
    mma_tiler_m: int,
    mma_tiler_n: int,
    mma_tiler_k: int,
    export_only: bool,
    output_dir: str,
    file_name: str,
    function_prefix: str,
    warmup: int,
    iterations: int,
    max_active_clusters: int,
    raw_sfa: bool,
    skip_ref_check: bool,
    tolerance: float,
    activation_dq_scale: float,
    weight_dq_scale: float,
):
    m, n, k = mnk
    if raw_sfa and m != 1:
        raise ValueError("--raw_sfa requires M=1")
    if not export_only and cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required")

    if export_only:
        ptrs = {
            "a": aot_placeholders.make_ptr(
                cutlass.Float4E2M1FN, assumed_align=32
            ),
            "b": aot_placeholders.make_ptr(
                cutlass.Float4E2M1FN, assumed_align=32
            ),
            "sfa": aot_placeholders.make_ptr(
                cutlass.Float8E4M3FN, assumed_align=16
            ),
            "sfb": aot_placeholders.make_ptr(
                cutlass.Float8E4M3FN, assumed_align=16
            ),
            "c": aot_placeholders.make_ptr(
                cutlass.Float16, assumed_align=16
            ),
            "alpha": aot_placeholders.make_ptr(
                cutlass.Float32, assumed_align=4
            ),
        }
        backing = None
        stream = aot_placeholders.make_stream()
        compile_clusters = aot_placeholders.runtime_int32()
    else:
        ptrs, backing = _create_nvfp4_pointers(
            m, n, k, sf_vec_size=16, c_dtype=cutlass.Float16
        )
        backing[-1].fill(activation_dq_scale * weight_dq_scale)
        stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
        compile_clusters = cutlass.Int32(1)

    kernel = GemmBlackwellGeForceNvFp4(
        sf_vec_size=16,
        mma_tiler_mn=(mma_tiler_m, mma_tiler_n),
        cluster_shape_mn=(1, 1),
        tile_k=mma_tiler_k,
        raw_sfa=raw_sfa,
    )

    start = time.time()
    compiled = cute.compile(
        kernel.wrapper,
        ptrs["a"],
        ptrs["b"],
        ptrs["sfa"],
        ptrs["sfb"],
        ptrs["c"],
        ptrs["alpha"],
        m,
        n,
        k,
        compile_clusters,
        stream,
        **(
            dict(options=aot_placeholders.compile_options())
            if export_only
            else {}
        ),
    )
    print(f"[{file_name}] Compilation time: {time.time() - start:.4f}s")
    if export_only:
        export_compiled_kernel(
            compiled,
            output_dir=output_dir,
            file_name=file_name,
            function_prefix=function_prefix,
            tag=f"[{file_name}]",
        )
    else:
        if not skip_ref_check:
            rng = cp.random.RandomState(1111)
            a_packed, a_sf_linear = _make_reference_operand(m, k, rng)
            b_packed, b_sf_linear = _make_reference_operand(n, k, rng)
            a_sf_swizzled = _swizzle_scale_factors(a_sf_linear, m, k)
            b_sf_swizzled = _swizzle_scale_factors(b_sf_linear, n, k)

            cp.copyto(backing[0], a_packed)
            cp.copyto(backing[1], b_packed)
            a_sf = a_sf_linear if raw_sfa else a_sf_swizzled
            a_sf_cupy = a_sf.reshape(-1)
            if raw_sfa:
                backing[2].fill(0)
                cp.copyto(backing[2][: a_sf_cupy.size], a_sf_cupy)
            else:
                cp.copyto(backing[2], a_sf_cupy)
            cp.copyto(backing[3], b_sf_swizzled)

            a_dequant = _dequantize_reference(a_packed, a_sf_linear)
            b_dequant = _dequantize_reference(b_packed, b_sf_linear)
            reference = (
                cp.matmul(a_dequant, b_dequant.transpose())
                * (activation_dq_scale * weight_dq_scale)
            ).astype(cp.float16)
            reference_abs_max = float(cp.abs(reference.astype(cp.float32)).max())
            reference_nonzero = int(cp.count_nonzero(reference))
            if reference_abs_max == 0.0 or reference_nonzero == 0:
                raise RuntimeError(
                    "Numerical reference is identically zero; the validation "
                    "inputs or quantization scales are not meaningful"
                )

        device_clusters = cp.cuda.runtime.getDeviceProperties(0)["multiProcessorCount"]
        active_clusters = device_clusters if max_active_clusters == 0 else max_active_clusters
        launch_args = (
            ptrs["a"],
            ptrs["b"],
            ptrs["sfa"],
            ptrs["sfb"],
            ptrs["c"],
            ptrs["alpha"],
            m,
            n,
            k,
            cutlass.Int32(active_clusters),
            stream,
        )
        for _ in range(warmup):
            compiled(*launch_args)
        cp.cuda.get_current_stream().synchronize()
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record()
        for _ in range(iterations):
            compiled(*launch_args)
        end_event.record()
        end_event.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
        if not skip_ref_check:
            actual = backing[4][:m]
            error = cp.abs(actual.astype(cp.float32) - reference.astype(cp.float32))
            max_abs_error = float(error.max())
            mean_abs_error = float(error.mean())
            p99_abs_error = float(cp.percentile(error, 99))
            actual_abs_max = float(cp.abs(actual.astype(cp.float32)).max())
            actual_nonzero = int(cp.count_nonzero(actual))
            print(
                f"Reference max/nonzero: {reference_abs_max:.6f}/{reference_nonzero}; "
                f"actual max/nonzero: {actual_abs_max:.6f}/{actual_nonzero}; "
                f"absolute error max/p99/mean: {max_abs_error:.6f}/"
                f"{p99_abs_error:.6f}/{mean_abs_error:.6f} "
                f"(atol={tolerance}, rtol=0.01)"
            )
            cp.testing.assert_allclose(
                actual,
                reference,
                atol=tolerance,
                rtol=0.01,
            )
        print(f"PASS average_us={elapsed_ms * 1000.0 / iterations:.3f}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnk", type=parse_comma_separated_ints, default=(64, 5120, 4096))
    parser.add_argument("--mma_tiler_m", type=int, choices=[64, 128], default=64)
    parser.add_argument("--mma_tiler_n", type=int, choices=[64, 128], default=64)
    parser.add_argument("--mma_tiler_k", type=int, choices=[128, 256], default=128)
    parser.add_argument("--export_only", action="store_true")
    parser.add_argument("--output_dir", default="./gemm_nvfp4_aot_artifacts")
    parser.add_argument("--file_name", default="gemm_blackwell_geforce_nvfp4_fp16_tm64_tn64")
    parser.add_argument("--function_prefix", default="gemm_blackwell_geforce_nvfp4_fp16_tm64_tn64")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--max_active_clusters",
        type=int,
        default=0,
        help="Persistent CTA limit; zero uses the device SM count",
    )
    parser.add_argument(
        "--raw_sfa",
        action="store_true",
        help="Stage row-major activation scales directly for the M=1 path",
    )
    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        help="Skip the numerical check against a dequantized CuPy reference",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="Absolute tolerance for the numerical reference check",
    )
    parser.add_argument(
        "--activation_dq_scale",
        type=float,
        default=1.0,
        help="TensorRT dequantization scale for dynamic activation block scales",
    )
    parser.add_argument(
        "--weight_dq_scale",
        type=float,
        default=1.0,
        help="TensorRT secondary dequantization scale for weight block scales",
    )
    return parser.parse_known_args(args=argv)[0]


if __name__ == "__main__":
    args = _parse_args(_saved_argv)
    if len(args.mnk) != 3:
        raise ValueError("--mnk must contain exactly 3 values (M,N,K)")
    run(
        mnk=args.mnk,
        mma_tiler_m=args.mma_tiler_m,
        mma_tiler_n=args.mma_tiler_n,
        mma_tiler_k=args.mma_tiler_k,
        export_only=args.export_only,
        output_dir=args.output_dir,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
        warmup=args.warmup,
        iterations=args.iterations,
        max_active_clusters=args.max_active_clusters,
        raw_sfa=args.raw_sfa,
        skip_ref_check=args.skip_ref_check,
        tolerance=args.tolerance,
        activation_dq_scale=args.activation_dq_scale,
        weight_dq_scale=args.weight_dq_scale,
    )
