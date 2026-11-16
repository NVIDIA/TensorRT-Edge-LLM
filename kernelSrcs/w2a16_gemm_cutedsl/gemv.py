# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 by the b12x authors.
# SPDX-License-Identifier: Apache-2.0
# Derived from b12x 2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88.

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from w2a16_gemm_cutedsl.intrinsics import (get_ptr_as_int64,
                                           iq2_xxs_descriptor_pair,
                                           ld_global_v4_u32, u32_as_f32,
                                           warp_reduce)


def _fadd(a, b):
    return a + b


@cute.jit
def _flat(pointer: cute.Pointer):
    return cute.make_tensor(pointer, cute.make_layout(
        (Int64(1) << Int64(50), )))


@cute.jit
def _dot_bf16x8(source: cute.Tensor, offset: Int64, w0: Uint32, w1: Uint32,
                w2: Uint32, w3: Uint32, accumulator: Float32):
    x0, x1, x2, x3 = ld_global_v4_u32(get_ptr_as_int64(source, offset))
    result = accumulator
    for wv, xv in ((w0, x0), (w1, x1), (w2, x2), (w3, x3)):
        result += u32_as_f32(wv << Uint32(16)) * u32_as_f32(xv << Uint32(16))
        result += u32_as_f32(wv & Uint32(4294901760)) * u32_as_f32(
            xv & Uint32(4294901760))
    return result


@dsl_user_op
def _decode_eight(descriptor, base, subscale, lut_addr, *, loc=None, ip=None):
    pairs = []
    for i in range(4):
        pairs.append(f"""
            shl.b32 lo, w{i}, 16;
            and.b32 hi, w{i}, 0xffff0000;
            mul.f32 lo, lo, scale;
            mul.f32 hi, hi, scale;
            cvt.rn.satfinite.bf16x2.f32 ${i}, hi, lo;
            shr.u32 pair_signs, signs, {i * 2};
            mul.lo.u32 pair_signs, pair_signs, 0x40008000;
            lop3.b32 ${i}, ${i}, pair_signs, 0x80008000, 0x78;
        """)
    asm = """{
        .reg .b16 dh;
        .reg .u32 grid, signs, parity, w<4>, pair_signs;
        .reg .u64 address;
        .reg .f32 scale, nibble, lo, hi;
        cvt.u16.u32 dh, $5;
        cvt.f32.f16 scale, dh;
        cvt.rn.f32.u32 nibble, $6;
        add.f32 nibble, nibble, 0f3f000000;
        mul.f32 scale, scale, nibble;
        mul.f32 scale, scale, 0f3e800000;
        shr.u32 signs, $4, 9;
        popc.b32 parity, signs;
        mad.lo.u32 signs, parity, 128, signs;
        and.b32 grid, $4, 0x1ff;
        mad.wide.u32 address, grid, 16, $7;
        ld.global.nc.v4.u32 {w0, w1, w2, w3}, [address];
    """ + "\n".join(pairs) + "\n}"
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            Uint32(descriptor).ir_value(loc=loc, ip=ip),
            Uint32(base).ir_value(loc=loc, ip=ip),
            Uint32(subscale).ir_value(loc=loc, ip=ip),
            Int64(lut_addr).ir_value(loc=loc, ip=ip)
        ],
        asm,
        "=r,=r,=r,=r,r,r,r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(4))


class IQ2XSGemv:

    def __init__(self, tile_m, descriptor_tile_n):
        self.tile_m = tile_m
        self.descriptor_tile_n = descriptor_tile_n

    @cute.jit
    def __call__(self, source: cute.Pointer, descriptors: cute.Pointer,
                 metadata: cute.Pointer, lut: cute.Pointer,
                 output: cute.Pointer, rows: Int32, stream: cuda.CUstream,
                 n: Int64, k: Int64):
        self.kernel(_flat(source), _flat(descriptors), _flat(metadata),
                    _flat(lut), _flat(output), rows, n,
                    k).launch(grid=((n + 3) // 4,
                                    (rows + self.tile_m - 1) // self.tile_m,
                                    1),
                              block=(128, 1, 1),
                              stream=stream)

    @cute.kernel
    def kernel(self, source: cute.Tensor, descriptors: cute.Tensor,
               metadata: cute.Tensor, lut: cute.Tensor, output: cute.Tensor,
               rows: Int32, n: Int64, k: Int64):
        thread, _, _ = cute.arch.thread_idx()
        tile_n, tile_m, _ = cute.arch.block_idx()
        lane = Int32(thread) % 32
        column = Int64(tile_n) * 4 + Int64(Int32(thread) // 32)
        first_row = Int64(tile_m) * self.tile_m
        accum = cute.make_rmem_tensor((self.tile_m, ), Float32)
        for r in cutlass.range_constexpr(self.tile_m):
            accum[r] = Float32(0)
        if column < n:
            block = Int64(0)
            meta = column * 2
            record = (column // self.descriptor_tile_n * (k // 256) *
                      (self.descriptor_tile_n * 32) +
                      column % self.descriptor_tile_n * 32 +
                      Int64(lane // 4) * 4)
            while block < k // 256:
                base = Uint32(metadata[meta]) | Uint32(metadata[meta + 1]) << 8
                grids = Uint32(descriptors[record]) | Uint32(
                    descriptors[record + 1]) << 16
                signs_scale = Uint32(
                    descriptors[record +
                                2]) | Uint32(descriptors[record + 3]) << 16
                pair = iq2_xxs_descriptor_pair(grids, signs_scale,
                                               lane % 4 // 2)
                descriptor = pair >> lane % 2 * 16 & Uint32(65535)
                subscale = signs_scale >> 28
                address = get_ptr_as_int64(lut, 0)
                w0, w1, w2, w3 = _decode_eight(descriptor, base, subscale,
                                               address)
                for r in cutlass.range_constexpr(self.tile_m):
                    row = first_row + r
                    if row < Int64(rows):
                        accum[r] = _dot_bf16x8(
                            source, row * k + block * 256 + Int64(lane) * 8,
                            w0, w1, w2, w3, accum[r])
                block += 1
                meta += n * 2
                record += self.descriptor_tile_n * 32
            for r in cutlass.range_constexpr(self.tile_m):
                row = first_row + r
                if row < Int64(rows):
                    total = warp_reduce(accum[r], _fadd)
                    if lane == 0:
                        output[row * n + column] = total.to(BFloat16)
