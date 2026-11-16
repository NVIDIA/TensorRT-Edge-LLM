# Copyright (c) 2025, Tri Dao.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Modified by NVIDIA for Edge-LLM: extract only the AOT tensor-alignment helper
# so the copied kernel does not acquire the upstream Torch/Triton dependencies.

"""Tensor alignment helpers used by the AOT-only sparse-attention source."""

import cutlass.cute as cute


def assume_strides_aligned(t):
    """Assume all strides except the last are divisible by 128 bits.

    Python int strides (e.g., stride=0 from GQA expand) are kept as-is
    since they're static and don't need alignment assumptions.
    """
    divby = 128 // t.element_type.width
    strides = tuple(
        s if isinstance(s, int) else cute.assume(s, divby=divby)
        for s in t.stride[:-1]
    )
    return (*strides, t.stride[-1])


def assume_tensor_aligned(t):
    """Rebuild a tensor with 128-bit aligned stride assumptions. Passes through None."""
    if t is None:
        return None
    return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=assume_strides_aligned(t)))
