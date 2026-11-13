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
"""Layers shared by the MiniMax-H3 text encoder and generation transformer."""

from __future__ import annotations

import tensorrt as trt

from ...ops import Linear, Module, RMSNorm
from ...ops import functional as F

#: Projections whose outputs can exceed the FP16 range scale their inputs by a
#: power of two before the FP16 GEMM and undo it on the FP32 result, which
#: changes no rounding.
PROJECTION_INPUT_SCALE = 2.0**-7


def scaled_projection(projection, hidden):
    """Apply a bias-free FP16 projection whose output can exceed FP16."""
    scaled = (hidden.cast(trt.float32) * PROJECTION_INPUT_SCALE).cast(
        trt.float16)
    return projection(scaled).cast(
        trt.float32) * (1.0 / PROJECTION_INPUT_SCALE)


def apply_rotary(hidden, cos, sin, rotary_dim: int):
    """Rotate the leading ``rotary_dim`` channels of ``[1, S, heads, dim]``.

    Split-half RoPE evaluated in FP32; ``cos`` and ``sin`` broadcast as
    ``[1, S, 1, rotary_dim]``.
    """
    head_dim = int(hidden.shape[-1])
    hidden = hidden.cast(trt.float32)
    rotary = hidden.slice_last_dim(0, rotary_dim, 4)
    half = rotary_dim // 2
    rotated = F.concatenate((-rotary.slice_last_dim(half, half, 4),
                             rotary.slice_last_dim(0, half, 4)), 3)
    output = rotary * cos + rotated * sin
    if rotary_dim < head_dim:
        output = F.concatenate(
            (output, hidden.slice_last_dim(rotary_dim, head_dim - rotary_dim,
                                           4)), 3)
    return output.cast(trt.float16)


class Attention(Module):
    """Noncausal self-attention with per-head Q/K RMSNorm.

    Without ``suffix_rows`` every row attends to the whole sequence with
    TensorRT attention. With it, the trailing video rows use block-sparse
    routing and the leading text and audio rows stay dense.
    """

    def __init__(self, ctx, prefix: str, num_heads: int, head_dim: int,
                 eps: float) -> None:
        super().__init__(ctx, prefix)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.to_q = Linear(ctx, self.key("to_q"), tensor_parallel=False)
        self.to_k = Linear(ctx, self.key("to_k"), tensor_parallel=False)
        self.to_v = Linear(ctx, self.key("to_v"), tensor_parallel=False)
        self.to_out = Linear(ctx, self.key("to_out.0"), tensor_parallel=False)
        self.norm_q = RMSNorm(ctx, self.key("norm_q"), eps, rank=4)
        self.norm_k = RMSNorm(ctx, self.key("norm_k"), eps, rank=4)

    def forward(self, hidden, rope=None, suffix_rows=None):
        heads = (0, 0, self.num_heads, self.head_dim)
        query = self.norm_q(self.to_q(hidden).reshape(heads))
        key = self.norm_k(self.to_k(hidden).reshape(heads))
        value = self.to_v(hidden).reshape(heads)
        if rope is not None:
            query = apply_rotary(query, *rope)
            key = apply_rotary(key, *rope)
        if suffix_rows is None:
            attended = F.scaled_dot_product_attention(
                query.transpose((0, 2, 1, 3)),
                key.transpose((0, 2, 1, 3)),
                value.transpose((0, 2, 1, 3)),
                scale=self.head_dim**-0.5).transpose((0, 2, 1, 3))
        else:
            flat = (-1, self.num_heads, self.head_dim)
            attended = F.sparse_attention(query.reshape(flat),
                                          key.reshape(flat),
                                          value.reshape(flat), suffix_rows)
        return scaled_projection(
            self.to_out,
            attended.reshape((1, -1, self.num_heads * self.head_dim)))


class FeedForward(Module):
    """SwiGLU over the fused ``[up | gate]`` projection."""

    def __init__(self, ctx, prefix: str, inner_dim: int) -> None:
        super().__init__(ctx, prefix)
        self.inner_dim = inner_dim
        self.proj = Linear(ctx, self.key("net.0.proj"), tensor_parallel=False)
        self.down = Linear(ctx, self.key("net.2"), tensor_parallel=False)

    def forward(self, hidden):
        fused = self.proj(hidden).cast(trt.float32)
        up = fused.slice_last_dim(0, self.inner_dim, 3)
        gate = fused.slice_last_dim(self.inner_dim, self.inner_dim, 3)
        return scaled_projection(self.down, up * gate.silu())
