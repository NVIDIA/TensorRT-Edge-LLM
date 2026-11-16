# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PTX primitives for the SM110 W2A16 UMMA kernels."""

import cutlass
from cutlass import cutlass_dsl
from cutlass._mlir.dialects import llvm


@cutlass_dsl.dsl_user_op
def ptx(code, constraints, operands=(), results=(), *, loc=None, ip=None):
    """Emit a typed PTX primitive, preserving memory and instruction ordering."""
    types = {
        'u32': (cutlass_dsl.T.i32, cutlass.Uint32),
        'u64': (cutlass_dsl.T.i64, cutlass.Uint64),
        'f32': (cutlass_dsl.T.f32, cutlass.Float32)
    }
    result_types = [types[name][0]() for name in results]
    result_type = None
    if len(results) == 1:
        result_type = result_types[0]
    elif results:
        result_type = llvm.StructType.get_literal(result_types)
    value = llvm.inline_asm(
        result_type,
        [operand.ir_value(loc=loc, ip=ip) for operand in operands],
        code,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip)
    if len(results) == 1:
        return types[results[0]][1](value)
    return tuple(types[name][1](llvm.extractvalue(
        result_types[index], value, [index], loc=loc, ip=ip))
                 for index, name in enumerate(results))


@cutlass_dsl.dsl_user_op
def decode_record(grids, controls, base, lut, copies, *, loc=None, ip=None):
    """Decode an IQ2_XXS K32 record to sixteen packed BF16 pairs.

    `lut` points at this lane's LUT copy; entry e's low and high selector words sit at
    e * 8 * copies.
    """
    # p8 = 2 * round(scale * (b + 0.5)) equals the reference's scaled product exactly, and
    # p25 and p43 multiply it by 25/8 and 43/8, so all three round as in the reference.
    code = """{
    .reg .b16 dh, unused;
    .reg .f32 scale, factor, p8, p25, p43;
    .reg .b32 palette0, palette1, grid, signs, parity, lo, hi, selector, bits;
    mov.b32 {dh, unused}, $18;
    cvt.f32.f16 scale, dh;
    shr.u32 bits, $17, 28;
    mad.lo.u32 bits, bits, 2, 1;
    cvt.rn.f32.u32 factor, bits;
    mul.f32 p8, scale, factor;
    mul.f32 p25, p8, 0f40480000;
    mul.f32 p43, p8, 0f40ac0000;
    cvt.rn.bf16x2.f32 palette0, p25, p8;
    cvt.rn.bf16x2.f32 palette1, p43, p43;
    """
    for sub in range(4):
        code += f"""
        bfe.u32 grid, $16, {sub*8}, 8;
        bfe.u32 signs, $17, {sub*7}, 7;
        popc.b32 parity, signs;
        and.b32 parity, parity, 1;
        mad.lo.u32 signs, parity, 128, signs;
        mad.lo.u32 bits, grid, {8 * copies}, $19;
        ld.shared.v2.u32 {{lo, hi}}, [bits];
        """
        for pair in range(4):
            # signs * (2^(15 - 2p) + 2^(30 - 2p)) moves sign bits 2p and 2p + 1 to bits 15 and
            # 31; the two shifted copies do not overlap, so no carry reaches either bit.
            code += f"""
            shr.u32 selector, {'lo' if pair < 2 else 'hi'}, {(pair%2)*16};
            prmt.b32 ${sub*4+pair}, palette0, palette1, selector;
            mul.lo.u32 bits, signs, {0x40008000 >> (2 * pair)};
            and.b32 bits, bits, 0x80008000;
            xor.b32 ${sub*4+pair}, ${sub*4+pair}, bits;
            """
    code += '}'
    return ptx(code,
               ','.join(['=&r'] * 16 + ['r'] * 4),
               (cutlass.Uint32(grids), cutlass.Uint32(controls),
                cutlass.Uint32(base), cutlass.Uint32(lut)), ('u32', ) * 16,
               loc=loc,
               ip=ip)


@cutlass_dsl.dsl_user_op
def packed_epilogue(lo, hi, weights, first, *, loc=None, ip=None):
    """Round before ReLU squared or routing multiplication, returning BF16x2."""
    code = '''{
    .reg .b32 operand;
    cvt.rn.bf16x2.f32 $0, $2, $1;
    '''
    if first:
        code += '''
        mov.b32 operand, 0;
        max.bf16x2 $0, $0, operand;
        mul.bf16x2 $0, $0, $0;
        '''
    else:
        code += '''
        ld.shared.u32 operand, [$3];
        mul.bf16x2 $0, $0, operand;
        '''
    code += '}'
    return ptx(
        code,
        '=&r,f,f,r',
        (cutlass.Float32(lo), cutlass.Float32(hi), cutlass.Uint32(weights)),
        ('u32', ),
        loc=loc,
        ip=ip)
