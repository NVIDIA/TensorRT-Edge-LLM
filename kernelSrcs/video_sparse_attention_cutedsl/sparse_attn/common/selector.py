# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CTA-local routing-mask helpers shared by the CuTe architecture backends."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def popc_b32(
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(value).ir_value(loc=loc, ip=ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@cute.jit
def route_is_exact(
    q_block: Int32,
    kv_block: Int32,
    column_mean: Float32,
    threshold: Float32,
    valid: cutlass.Boolean,
) -> cutlass.Boolean:
    distance = q_block - kv_block
    if distance < Int32(0):
        distance = Int32(0) - distance
    return ((column_mean > threshold) or distance <= Int32(1)) and valid
__all__ = ["popc_b32", "route_is_exact"]
