# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AOT export of the DIGITS W2A16 ReLU2 MoE kernel."""
import argparse
import json
from pathlib import Path
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutedsl_utils import aot_placeholders
from w2a16_moe_cutedsl.moe_kernel import W2A16MoeKernel
from w2a16_moe_cutedsl.umma_kernel import W2A16UmmaKernel


class MoeLaunch:

    def __init__(self, decode, row_tile):
        self.decode = decode
        self.row_tile = row_tile
        self.kernel = W2A16MoeKernel(decode, row_tile)
        self.grid = self.kernel.sms * self.kernel.blocks_per_sm

    @cute.jit
    def __call__(self, x: cute.Pointer, w1: cute.Pointer, w2: cute.Pointer,
                 s1: cute.Pointer, s2: cute.Pointer, unit: cute.Pointer,
                 lut: cute.Pointer, routes: cute.Pointer, experts: cute.Pointer,
                 route_count: cute.Pointer, route_weights: cute.Pointer,
                 activated: cute.Pointer, slots: cute.Pointer,
                 output: cute.Pointer, scratch1: cute.Pointer,
                 scratch2: cute.Pointer, locks: cute.Pointer, m: cutlass.Int32,
                 e: cutlass.Int32, intermediate_size: cutlass.Int32,
                 stream: cuda.CUstream):
        rows = cutlass.Int64(m) * 9
        route_capacity = rows + cutlass.Int64(e) * (self.row_tile - 1)
        a = cute.make_tensor(
            activated,
            cute.make_layout(rows * cutlass.Int64(intermediate_size)))
        slot = cute.make_tensor(slots, cute.make_layout(rows * 1024))
        out = cute.make_tensor(output,
                               cute.make_layout(cutlass.Int64(m) * 1024))
        r = cute.make_tensor(routes, cute.make_layout(route_capacity))
        b = cute.make_tensor(experts, cute.make_layout(route_capacity))
        count = cute.make_tensor(route_count, cute.make_layout(1))
        t1 = cute.make_tensor(scratch1,
                              cute.make_layout(4 * self.kernel.sms * 64 * 256))
        t2 = cute.make_tensor(scratch2,
                              cute.make_layout(4 * self.kernel.sms * 64 * 256))
        lk = cute.make_tensor(locks, cute.make_layout(4 * self.kernel.sms + 2))
        unused_float = cute.make_tensor(unit,
                                        cute.make_layout(cutlass.Int64(e)))
        fc2 = out if cutlass.const_expr(self.decode) else slot
        elements = cutlass.Int64(e) * 1024 * cutlass.Int64(
            intermediate_size) // 16
        self.kernel(x, w1, w2, elements, elements, slot, a, fc2, s1, s2, unit,
                    unit, r, b, count, route_weights, t1, t2, lk, lut, e, m,
                    cutlass.Int32(self.grid), intermediate_size, stream)


def export(args):
    umma = args.mode == 'umma'
    if umma and args.target_sm not in (101, 110):
        raise ValueError('UMMA requires Thor SM101 or SM110')
    if not umma and args.row_tile > 64:
        raise ValueError('Fused AOT row tiles are 8/16/32/48/64')
    launch = (W2A16UmmaKernel(args.row_tile, args.stages, args.specialized,
                              args.groups, args.b_stages, args.pair,
                              args.fused, args.lag, args.group_chunks,
                              args.lut_copies, args.prefetch)
              if umma else
              MoeLaunch(args.mode == 'decode', args.row_tile))
    dtypes = [
        cutlass.BFloat16, cutlass.Int32, cutlass.Int32, cutlass.Int32,
        cutlass.Int32, cutlass.Float32, cutlass.Uint8, cutlass.Int32,
        cutlass.Int32, cutlass.Int32, cutlass.Float32, cutlass.BFloat16,
        cutlass.BFloat16, cutlass.BFloat16, cutlass.Float32, cutlass.Float32,
        cutlass.Int32
    ]
    if umma:
        dtypes = [
            cutlass.BFloat16, cutlass.Uint32, cutlass.Uint32, cutlass.Float16,
            cutlass.Float16, cutlass.Uint32, cutlass.Int32, cutlass.Int32,
            cutlass.Float32, cutlass.BFloat16, cutlass.BFloat16,
            cutlass.BFloat16, cutlass.Int32
        ]
    pointers = [aot_placeholders.make_ptr(dtype) for dtype in dtypes]
    compiled = cute.compile(
        launch,
        *pointers,
        cutlass.Int32(8),
        cutlass.Int32(512),
        cutlass.Int32(2560),
        cuda.CUstream(0),
        options=aot_placeholders.compile_options('--opt-level=2'))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    compiled.export_to_c(file_path=args.output_dir,
                         file_name=args.file_name,
                         function_prefix=args.function_prefix)
    metadata = {'name': args.file_name}
    if umma:
        metadata.update(token_tile=launch.tokens,
                        stages=launch.stages,
                        tile_k=launch.tile_k,
                        specialized=launch.specialized,
                        groups=launch.groups,
                        shared_bytes=launch.shared_bytes)
    else:
        metadata.update(grid=launch.grid,
                        num_sms=launch.kernel.sms,
                        row_tile=launch.row_tile,
                        tile_k=launch.kernel.fc1.tile_k,
                        tile_n=launch.kernel.fc1.tile_n,
                        stages=launch.kernel.fc1.stages,
                        shared_bytes=launch.kernel.shared_words * 4,
                        blocks_per_sm=launch.kernel.blocks_per_sm)
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--file_name', required=True)
    parser.add_argument('--function_prefix', required=True)
    parser.add_argument('--target_sm', type=int)
    parser.add_argument('--mode',
                        choices=('decode', 'prefill', 'umma'),
                        required=True)
    parser.add_argument('--row_tile', type=int, choices=(8, 16, 32, 48, 64, 96, 192), required=True)
    parser.add_argument('--stages', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--specialized', action='store_true')
    parser.add_argument('--groups', type=int, choices=(1, 2), default=1)
    parser.add_argument('--b_stages', type=int, choices=(2, 3, 4, 6))
    parser.add_argument('--pair', action='store_true')
    parser.add_argument('--fused', action='store_true')
    parser.add_argument('--lag', type=int, default=4)
    parser.add_argument('--group_chunks', type=int, default=1)
    parser.add_argument('--prefetch', type=int, choices=(1, 3, 7), default=1)
    parser.add_argument('--lut_copies',
                        type=int,
                        choices=(1, 2, 4, 8, 16),
                        default=1)
    export(parser.parse_args())
