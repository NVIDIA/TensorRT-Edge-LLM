# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 by the b12x authors.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from b12x 2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88.
# See README.md for the supported IQ2_XXS/BF16 contract.

from typing import Tuple
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def st_global_v4_f32(
    base_ptr: Int64,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
):
    """Store 128 bits (4 x float32) to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def ld_global_v4_f32(
    base_ptr: Int64,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Load 128 bits (4 x float32) from global memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(),
                                     T.f32(),
                                     T.f32(),
                                     T.f32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def st_global_v4_u32(
    base_ptr: Int64,
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    *,
    loc=None,
    ip=None,
):
    """Store 128 bits (4 x uint32) to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
            Uint32(v2).ir_value(loc=loc, ip=ip),
            Uint32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def shared_ptr_to_u32(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """Convert an address-space-3 shared-memory pointer to a u32 address."""
    return Int32(llvm.ptrtoint(T.i32(), ptr.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def ldmatrix_m8n8x4_b16(smem_addr: Int32,
                        *,
                        loc=None,
                        ip=None) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Issue `ldmatrix.sync.aligned.m8n8.x4.shared.b16` from a shared-memory byte address."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(),
                                     T.i32(),
                                     T.i32(),
                                     T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1), Uint32(r2), Uint32(r3)


@dsl_user_op
def ldmatrix_m8n8x2_b16(smem_addr: Int32,
                        *,
                        loc=None,
                        ip=None) -> Tuple[Uint32, Uint32]:
    """Issue `ldmatrix.sync.aligned.m8n8.x2.shared.b16` from a shared-memory byte address."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1)


@dsl_user_op
def ld_shared_v4_u32(smem_addr: Int32,
                     *,
                     loc=None,
                     ip=None) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load 128 bits (4 x uint32) from shared memory. smem_addr is a u32 shared-memory address."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(),
                                     T.i32(),
                                     T.i32(),
                                     T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1), Uint32(r2), Uint32(r3)


@dsl_user_op
def ld_shared_v2_u32(smem_addr: Int32,
                     *,
                     loc=None,
                     ip=None) -> Tuple[Uint32, Uint32]:
    """Load 64 bits (2 x uint32) from shared memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.u32 {$0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1)


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
def cp_async4_shared_global_pred(smem_addr: Int32,
                                 gmem_addr: Int64,
                                 pred: Int32,
                                 *,
                                 loc=None,
                                 ip=None):
    """Predicated 16-byte `cp.async.cg.shared.global` copy."""
    llvm.inline_asm(
        None,
        [
            Int32(pred).ir_value(loc=loc, ip=ip),
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.b32 p, $0, 0; @p cp.async.cg.shared.global [$1], [$2], 16; }",
        "r,r,l",
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


@dsl_user_op
def atomic_add_global_i32(addr: Int64,
                          val: Int32,
                          *,
                          loc=None,
                          ip=None) -> Int32:
    """Global memory int32 atomic add. Returns old value."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(val).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        ))


@dsl_user_op
def red_add_global_release_i32(addr: Int64, val: Int32, *, loc=None, ip=None):
    """No-return global int32 add with a GPU-scope release fence."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "fence.acq_rel.gpu;\nred.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_i32_relaxed(addr: Int32, *, loc=None, ip=None) -> Int32:
    """Load int32 from shared memory at a 32-bit byte address."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.s32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        ))


@dsl_user_op
def ld_shared_u32(addr: Int32, *, loc=None, ip=None) -> Uint32:
    """Load uint32 from shared memory at a 32-bit byte address."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.u32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        ))


@dsl_user_op
def st_shared_i32(addr: Int32, val: Int32, *, loc=None, ip=None):
    """Store int32 to shared memory at a 32-bit byte address."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.s32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_u32(addr: Int32, val: Uint32, *, loc=None, ip=None):
    """Store uint32 to shared memory at a 32-bit byte address."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Uint32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_v4_f32(addr: Int32,
                     *,
                     loc=None,
                     ip=None) -> Tuple[Float32, Float32, Float32, Float32]:
    """Load 128 bits (4 x float32) from shared memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(),
                                     T.f32(),
                                     T.f32(),
                                     T.f32()]),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def st_shared_v4_f32(
    addr: Int32,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
):
    """Store 128 bits (4 x float32) to shared memory."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_bf16_from_f32(addr: Int32, val: Float32, *, loc=None, ip=None):
    """Convert float32 to BF16 and store one 16-bit value to shared memory."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b16 tmp; cvt.rn.bf16.f32 tmp, $1; st.shared.b16 [$0], tmp; }",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_i32(addr: Int64, val: Int32, *, loc=None, ip=None):
    """Store int32 to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def ld_global_acquire_i32(addr: Int64, *, loc=None, ip=None) -> Int32:
    """Load int32 from global memory with acquire semantics."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.acquire.gpu.s32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        ))


@dsl_user_op
def threadfence(*, loc=None, ip=None):
    """Emit membar.gl — threadfence across global memory."""
    llvm.inline_asm(
        None,
        [],
        "membar.gl;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bfloat2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Multiply two BFloat2 values element-wise: (a.x*b.x, a.y*b.y)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip)
            ],
            "mul.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        ))


@dsl_user_op
def broadcast_f32_to_bfloat2(x: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack one float32 value into both lanes of a bf16x2 register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "cvt.rn.satfinite.bf16x2.f32 $0, $1, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        ))


@dsl_user_op
def pack_f32x2_to_bfloat2(x0: Float32,
                          x1: Float32,
                          *,
                          loc=None,
                          ip=None) -> Uint32:
    """Pack 2 float32 values into one bf16x2 register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(x0).ir_value(loc=loc, ip=ip),
                Float32(x1).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.satfinite.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        ))


@dsl_user_op
def bf16_mma_m16n8k16_f32(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA helper for `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(),
                                     T.f32(),
                                     T.f32(),
                                     T.f32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(b0).ir_value(loc=loc, ip=ip),
            Uint32(b1).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def bf16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    b0_0: Uint32,
    b1_0: Uint32,
    b0_1: Uint32,
    b1_1: Uint32,
    a0: Uint32,
    a1: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """BF16 MMA form used by the routed m-block-size-8 path.

    The dequantized RHS fragments feed the hardware A operand, while the
    routed activation fragment loaded with `ldmatrix.x2` feeds hardware B.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(),
                                     T.f32(),
                                     T.f32(),
                                     T.f32()]),
        [
            Uint32(b0_0).ir_value(loc=loc, ip=ip),
            Uint32(b1_0).ir_value(loc=loc, ip=ip),
            Uint32(b0_1).ir_value(loc=loc, ip=ip),
            Uint32(b1_1).ir_value(loc=loc, ip=ip),
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def bfloat2_to_float2_scaled(bf2: Uint32,
                             scale: Float32,
                             *,
                             loc=None,
                             ip=None) -> Tuple[Float32, Float32]:
    """Convert bfloat16x2 to float2 AND multiply by scale."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [
            Uint32(bf2).ir_value(loc=loc, ip=ip),
            Float32(scale).ir_value(loc=loc, ip=ip)
        ],
        """
        {
            .reg .b32 lo, hi;
            .reg .f32 f0, f1;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 f0, lo;
            mov.b32 f1, hi;
            mul.f32 $0, f0, $3;
            mul.f32 $1, f1, $3;
        }
        """,
        "=f,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)

    return Float32(f0), Float32(f1)


@cute.jit
def iq2_xxs_descriptor_pair(grids: Uint32, signs_scale: Uint32, half: Int32):
    """Normalize one K16 half in registers, retaining native K32 storage."""
    grids = grids >> (half * Int32(16))
    signs = signs_scale >> (half * Int32(14))
    lo = (grids & Uint32(255)) | ((signs & Uint32(127)) << 9)
    hi = ((grids >> 8) & Uint32(255)) | (((signs >> 7) & Uint32(127)) << 9)
    return lo | (hi << 16)


def _packed_decode_iq2_xs_to_bfloat2x4(
    q_row0,
    q_row1,
    base_pair,
    subscale_pair,
    execution_lut_addr,
    pair_byte_offset,
    *,
    shared_lut=False,
    selector_lut=False,
    loc=None,
    ip=None,
):
    """Decode one IQ2_XS MMA RHS fragment with one-round BF16 conversion."""

    extracts = (
        "and.b32 d0, $4, 0xffff;",
        "shr.u32 d1, $4, 16;",
        "and.b32 d2, $5, 0xffff;",
        "shr.u32 d3, $5, 16;",
    )
    rows = (0, 0, 1, 1)
    table_stride = 8 if selector_lut else 16
    load_type = "u16" if selector_lut else "u32"
    decode = []
    for index, (extract, row) in enumerate(zip(extracts, rows, strict=True)):
        load = (f"""
            cvt.u32.u64 sa{index}, $8;
            mad.lo.u32 sa{index}, d{index}, {table_stride}, sa{index};
            add.u32 sa{index}, sa{index}, so;
            ld.shared.{load_type} u{index}, [sa{index}];
            """ if shared_lut else f"""
            mad.wide.u32 a{index}, d{index}, {table_stride}, $8;
            add.u64 a{index}, a{index}, po;
            ld.global.nc.{load_type} u{index}, [a{index}];
            """)
        conversion = (f"prmt.b32 ${index}, t{row}0, t{row}1, u{index};"
                      if selector_lut else f"""
            shl.b32 fl{index}, u{index}, 16;
            and.b32 fh{index}, u{index}, 0xffff0000;
            mul.f32 fl{index}, fl{index}, s{row};
            mul.f32 fh{index}, fh{index}, s{row};
            cvt.rn.satfinite.bf16x2.f32 ${index}, fh{index}, fl{index};
            """)
        decode.append(f"""
            {extract}
            shr.u32 signs{index}, d{index}, 9;
            popc.b32 parity{index}, signs{index};
            mad.lo.u32 signs{index}, parity{index}, 128, signs{index};
            shr.u32 signs{index}, signs{index}, $9;
            and.b32 d{index}, d{index}, 0x1ff;
            {load}
            {conversion}
            mul.lo.u32 sign_pair{index}, signs{index}, 0x40008000;
            lop3.b32 ${index}, ${index}, sign_pair{index}, 0x80008000, 0x78;
            """)
    selector_setup = ""
    if selector_lut:
        selector_setup = """
            .reg .f32 m00, m01, m02, m10, m11, m12;
            .reg .b32 t00, t01, t10, t11;
        """
        for row in range(2):
            selector_setup += f"""
            mul.f32 m{row}0, s{row}, 0f41000000;
            mul.f32 m{row}1, s{row}, 0f41c80000;
            mul.f32 m{row}2, s{row}, 0f422c0000;
            cvt.rn.satfinite.bf16x2.f32 t{row}0, m{row}1, m{row}0;
            cvt.rn.satfinite.bf16x2.f32 t{row}1, m{row}2, m{row}2;
            """
    pair_shift = 0 if selector_lut else 1
    asm = (
        """
        {
            .reg .b16 dh0, dh1;
            .reg .b32 d0, d1, d2, d3, n0, n1;
            .reg .u32 u0, u1, u2, u3, xl0, xl1, xl2, xl3,
                      xh0, xh1, xh2, xh3;
            .reg .u32 signs0, signs1, signs2, signs3,
                      parity0, parity1, parity2, parity3,
                      sign_pair0, sign_pair1, sign_pair2, sign_pair3;
            .reg .u64 po, a0, a1, a2, a3;
            .reg .u32 sa0, sa1, sa2, sa3, so;
            .reg .f32 s0, s1, fl0, fl1, fl2, fl3,
                      fh0, fh1, fh2, fh3;
            mov.b32 {dh0, dh1}, $6;
            cvt.f32.f16 s0, dh0;
            cvt.f32.f16 s1, dh1;
            bfe.u32 n0, $7, 0, 4;
            bfe.u32 n1, $7, 8, 4;
            cvt.rn.f32.u32 fl0, n0;
            cvt.rn.f32.u32 fl1, n1;
            add.f32 fl0, fl0, 0f3f000000;
            add.f32 fl1, fl1, 0f3f000000;
            mul.f32 s0, s0, fl0;
            mul.f32 s1, s1, fl1;
            mul.f32 s0, s0, 0f3e800000;
            mul.f32 s1, s1, 0f3e800000;
        """ +
        f"shl.b32 so, $9, {pair_shift};\n            cvt.u64.u32 po, so;\n" +
        selector_setup + "".join(decode) + "\n}")
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(),
                                     T.i32(),
                                     T.i32(),
                                     T.i32()]),
        [
            Uint32(q_row0).ir_value(loc=loc, ip=ip),
            Uint32(q_row1).ir_value(loc=loc, ip=ip),
            Uint32(base_pair).ir_value(loc=loc, ip=ip),
            Uint32(subscale_pair).ir_value(loc=loc, ip=ip),
            Int64(execution_lut_addr).ir_value(loc=loc, ip=ip),
            Int32(pair_byte_offset).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=r,=r,=r,=r,r,r,r,r,l,r",
        has_side_effects=bool(shared_lut),
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip))
        for index in range(4))


@dsl_user_op
def packed_decode_iq2_xs_to_bfloat2x4(
    q_row0,
    q_row1,
    base_pair,
    subscale_pair,
    execution_lut_addr,
    pair_byte_offset,
    *,
    shared_lut=False,
    selector_lut=False,
    loc=None,
    ip=None,
):
    return _packed_decode_iq2_xs_to_bfloat2x4(
        q_row0,
        q_row1,
        base_pair,
        subscale_pair,
        execution_lut_addr,
        pair_byte_offset,
        shared_lut=shared_lut,
        selector_lut=selector_lut,
        loc=loc,
        ip=ip,
    )
