# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export standalone IQ2_XXS dense kernels for SM110 (Thor) and Blackwell GeForce SM120/SM121."""
import argparse
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutedsl_utils import aot_placeholders
from w2a16_gemm_cutedsl.gemm_blackwell_geforce import DenseGemmKernel
from w2a16_gemm_cutedsl.gemv import IQ2XSGemv


class DenseLaunch:
    """Keep M/N/K live; specialize only tile, packing layout, and split count."""

    def __init__(self, tile_m, tile_n, tile_k, slices, packing):
        self.simt = tile_n == 4
        self.slices = slices
        self.packing = packing
        self.gemv = IQ2XSGemv(tile_m, packing)
        self.gemm = DenseGemmKernel(tile_m, tile_n, tile_k, slices, packing)

    @cute.jit
    def __call__(self, x: cute.Pointer, values: cute.Pointer,
                 bases: cute.Pointer, lut: cute.Pointer, output: cute.Pointer,
                 scratch: cute.Pointer, m: cutlass.Int32,
                 n_tiles: cutlass.Int32, k_blocks: cutlass.Int32,
                 stream: cuda.CUstream):
        n = cutlass.Int64(n_tiles) * self.packing
        k = cutlass.Int64(k_blocks) * 256
        if cutlass.const_expr(self.simt):
            self.gemv(x, cute.recast_ptr(values, dtype=cutlass.Uint16), bases,
                      cute.recast_ptr(lut, dtype=cutlass.Uint16), output, m,
                      stream, n, k)
        else:
            a = cute.make_tensor(x,
                                 cute.make_layout((m, k, 1), stride=(k, 1, 0)))
            b = cute.make_tensor(
                values,
                cute.make_layout(((self.packing, n_tiles), (64, k_blocks), 1),
                                 stride=((64, self.packing * k // 4),
                                         (1, self.packing * 64), 0)))
            scales = cute.make_tensor(bases,
                                      cute.make_layout(n * k_blocks * 2))
            table = cute.make_tensor(lut, cute.make_layout(4096))
            target = output
            if cutlass.const_expr(self.slices > 1):
                target = scratch
            c = cute.make_tensor(
                target,
                cute.make_layout((m, n, self.slices),
                                 stride=(n, 1, cutlass.Int64(m) * n)))
            self.gemm(a, b, table, scales, c, 48, stream)
            if cutlass.const_expr(self.slices > 1):
                self.reduce(scratch, output,
                            cutlass.Int64(m) * n).launch(
                                grid=((cutlass.Int64(m) * n + 255) // 256, 1,
                                      1),
                                block=(256, 1, 1),
                                stream=stream)

    @cute.kernel
    def reduce(self, partials: cute.Pointer, output: cute.Pointer,
               size: cutlass.Int64):
        offset = cutlass.Int64(
            cute.arch.block_idx()[0]) * 256 + cute.arch.thread_idx()[0]
        if offset < size:
            value = cutlass.Float32(0)
            for part in cutlass.range_constexpr(self.slices):
                value += partials[cutlass.Int64(part) * size + offset]
            output[offset] = cutlass.BFloat16(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--file_name', required=True)
    parser.add_argument('--function_prefix', required=True)
    parser.add_argument('--tile_m', type=int, required=True)
    parser.add_argument('--tile_n', type=int, required=True)
    parser.add_argument('--tile_k', type=int, required=True)
    parser.add_argument('--slices',
                        type=int,
                        choices=(1, 2, 4, 8),
                        required=True)
    parser.add_argument('--packing', type=int, choices=(8, 128), required=True)
    args = parser.parse_args()
    valid = ((args.tile_m in (1, 2) and args.tile_n == 4 and args.tile_k == 256
              and args.slices == 1)
             or (args.tile_m, args.tile_n, args.tile_k) in ((16, 64, 128),
                                                            (16, 128, 256))
             or ((args.tile_m, args.tile_n, args.tile_k, args.slices)
                 in ((64, 128, 128, 1), (128, 128, 64, 1)))
             or ((args.tile_m, args.tile_n, args.tile_k, args.slices,
                  args.packing) == (8, 64, 256, 1, 128)))
    if not valid:
        parser.error('Unsupported IQ2_XXS launch configuration')
    launch = DenseLaunch(args.tile_m, args.tile_n, args.tile_k, args.slices,
                         args.packing)
    types = (cutlass.BFloat16, cutlass.Uint8, cutlass.Uint8, cutlass.Uint8,
             cutlass.BFloat16, cutlass.Float32)
    program = cute.compile(
        launch,
        *(aot_placeholders.make_ptr(t) for t in types),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cuda.CUstream(0),
        options=aot_placeholders.compile_options(
            '--opt-level=2' if args.tile_n == 4 else '--opt-level=3'))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    program.export_to_c(file_path=args.output_dir,
                        file_name=args.file_name,
                        function_prefix=args.function_prefix)


if __name__ == '__main__':
    main()
