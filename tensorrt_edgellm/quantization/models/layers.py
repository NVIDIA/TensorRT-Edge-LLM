# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Shared layers and utilities for standalone quantization models.

Common building blocks used by both EAGLE3 and MTP draft models.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Standard RMSNorm (Llama-style): ``weight * RMSNorm(x)``.

    Weight is initialised to **one**.  Used by EAGLE3 and other standard
    transformer architectures.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


class Gemma4RMSNorm(nn.Module):
    """Gemma4 RMSNorm computed in FP32; ``with_scale=False`` is weightless."""

    def __init__(self, hidden_size, eps=1e-6, with_scale=True):
        super().__init__()
        self.weight = (nn.Parameter(torch.ones(hidden_size))
                       if with_scale else None)
        self.eps = eps

    def forward(self, x):
        y = x.to(torch.float32)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.to(torch.float32)
        return y.to(x.dtype)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(
            nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    """Rotary position embedding."""

    def __init__(self,
                 dim,
                 max_position=2048,
                 base=10000.0,
                 proportional_rotary_factor=None):
        super().__init__()
        if proportional_rotary_factor is None:
            inv_freq = 1.0 / (base**(
                torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        else:
            # Gemma4 proportional RoPE rotates only the leading frequencies
            # (still normalized by the full dim); the rest stay unrotated.
            rope_angles = int(proportional_rotary_factor * dim // 2)
            rotated = 1.0 / (base**(torch.arange(
                0, 2 * rope_angles, 2, dtype=torch.float32) / dim))
            inv_freq = torch.cat((rotated,
                                  torch.zeros(dim // 2 - rope_angles,
                                              dtype=torch.float32)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position = max_position

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [B, L, D] — only used for device/dtype
        # position_ids: [B, L]
        inv_freq = self.inv_freq.float()[None, :, None].to(x.device)
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos).transpose(1, 2)  # [B, L, D//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, L, D]
        cos = emb.cos()[:, None, :, :]  # [B, 1, L, D]
        sin = emb.sin()[:, None, :, :]
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x):
    """Rotate the second half of the last dimension."""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x, n_rep):
    """Repeat KV heads for grouped-query attention."""
    if n_rep == 1:
        return x
    B, H, L, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, L,
                                      D).reshape(B, H * n_rep, L, D)
