# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 by the b12x authors.
# SPDX-License-Identifier: Apache-2.0
# Derived from b12x 2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88.

import math
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:
    """Get pointer to element at coordinate in tensor."""
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def ld_global_v4_u32(base_ptr: Int64,
                     *,
                     loc=None,
                     ip=None) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load 128 bits (4 x uint32) from global memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(),
                                     T.i32(),
                                     T.i32(),
                                     T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    v0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    v1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    v2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    v3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)

    return Uint32(v0), Uint32(v1), Uint32(v2), Uint32(v3)


@dsl_user_op
def u32_as_f32(value: Uint32, *, loc=None, ip=None) -> Float32:
    """Bitcast a uint32 to float32 (mov.b32; no numeric conversion)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(value).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        ))


@dsl_user_op
def shared_ptr_to_u32(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """Convert an address-space-3 shared-memory pointer to a u32 address."""
    return Int32(llvm.ptrtoint(T.i32(), ptr.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def cp_async_bulk_g2s_mbar(
    smem_dst_u32: Int32,
    gmem_src_i64: Int64,
    nbytes: Int32,
    mbar_u32: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Copy nbytes from global to shared memory and complete a CTA mbarrier.

    The destination and barrier use u32 shared-memory addresses; the source
    uses an i64 global address. Dense GEMM uses this for base scales and packed
    weights. The caller must initialize the barrier's expected byte count.
    """
    llvm.inline_asm(
        None,
        [
            Int32(smem_dst_u32).ir_value(loc=loc, ip=ip),
            Int64(gmem_src_i64).ir_value(loc=loc, ip=ip),
            Int32(nbytes).ir_value(loc=loc, ip=ip),
            Int32(mbar_u32).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
        " [$0], [$1], $2, [$3];",
        "r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async4_shared_global(smem_addr: Int32,
                            gmem_addr: Int64,
                            *,
                            loc=None,
                            ip=None):
    """16-byte `cp.async.cg.shared.global` copy."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def get_ptr_as_int64(tensor: cute.Tensor,
                     offset,
                     *,
                     loc=None,
                     ip=None) -> Int64:
    """Get the memory address of tensor[offset] as Int64."""
    elem_ptr = tensor.iterator + offset
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int64(ptr_int)


@cute.jit
def warp_reduce(val, op, width: cutlass.Constexpr[int] = 32):
    """Reduce across threads in a warp using butterfly shuffle."""
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
        return val


@cute.jit
def iq2_xxs_descriptor_pair(grids: Uint32, signs_scale: Uint32, half: Int32):
    """Normalize one K16 half in registers, retaining native K32 storage."""
    grids = grids >> (half * Int32(16))
    signs = signs_scale >> (half * Int32(14))
    lo = (grids & Uint32(255)) | ((signs & Uint32(127)) << 9)
    hi = ((grids >> 8) & Uint32(255)) | (((signs >> 7) & Uint32(127)) << 9)
    return lo | (hi << 16)


@dsl_user_op
def iq2_xs_descriptor_pair_to_bf16x2x2(descriptors,
                                       base,
                                       subscale,
                                       lut_addr,
                                       pair_offset,
                                       *,
                                       loc=None,
                                       ip=None):
    """Decode matching pairs from two adjacent descriptors using one FP32 scale."""
    decode = []
    for i in range(2):
        decode.append(f"""
            bfe.u32 descriptor, descriptor_word, {i * 16}, 16;
            shr.u32 signs, descriptor, 9;
            popc.b32 parity, signs;
            mad.lo.u32 signs, parity, 128, signs;
            shr.u32 signs, signs, pair_start;
            and.b32 grid, descriptor, 0x1ff;
            mad.lo.u32 address, grid, 16, table;
            add.u32 address, address, offset;
            ld.shared.u32 word, [address];
            shl.b32 lo, word, 16;
            and.b32 hi, word, 0xffff0000;
            mul.f32 lo, lo, scale;
            mul.f32 hi, hi, scale;
            cvt.rn.satfinite.bf16x2.f32 ${i}, hi, lo;
            mul.lo.u32 sign_pair, signs, 0x40008000;
            lop3.b32 ${i}, ${i}, sign_pair, 0x80008000, 0x78;
        """)
    asm = """{
        .reg .b16 dh;
        .reg .u32 descriptor, descriptor_word, grid, signs, parity, word, sign_pair, offset, address, table, pair_start;
        .reg .f32 scale, nibble, lo, hi;
        mov.u32 descriptor_word, $2;
        mov.u32 pair_start, $6;
        cvt.u16.u32 dh, $3;
        cvt.f32.f16 scale, dh;
        cvt.rn.f32.u32 nibble, $4;
        add.f32 nibble, nibble, 0f3f000000;
        mul.f32 scale, scale, nibble;
        mul.f32 scale, scale, 0f3e800000;
        cvt.u32.u64 table, $5;
        shl.b32 offset, $6, 1;
    """ + "".join(decode) + "\n}"
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            Uint32(descriptors).ir_value(loc=loc, ip=ip),
            Uint32(base).ir_value(loc=loc, ip=ip),
            Uint32(subscale).ir_value(loc=loc, ip=ip),
            Int64(lut_addr).ir_value(loc=loc, ip=ip),
            Int32(pair_offset).ir_value(loc=loc, ip=ip)
        ],
        asm,
        "=r,=r,r,r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(2))
