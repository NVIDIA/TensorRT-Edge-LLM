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
"""pi0.5 PaliGemma prefix tower (gemma_2b) as a prefill-only K/V producer.

The policy path runs this tower ONCE to produce the per-layer K/V the action
expert attends to; it never decodes and never emits text, so there is no
lm_head and no KV-cache plumbing. Attention is **bidirectional** over the whole
prefix (PaliGemma prefix-LM: openpi ``embed_prefix`` sets ``att_masks`` all
zero), which the KV-cache ``AttentionPlugin`` cannot express -- and that plugin
also fails Myelin compilation in this graph shape (see
``experimental_models/cosmos3``). Both reasons point at the same path: the
TRT-native ``trt::attention_onnx`` / ``trt::rope_onnx`` ops.

The runtime compacts the prefix (no missing-camera or language padding reaches
the graph), so attention needs no mask and ``attention_pos_id`` is ``0..S-1``.

The tower's final norm is intentionally absent: openpi discards the prefix
hidden states and keeps only ``past_key_values``, so the norm is dead weight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class Pi05PrefixConfig:
    """Hyperparameters for the pi0.5 prefix tower."""

    hidden_size: int = 2048
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    intermediate_size: int = 16384
    rms_norm_eps: float = 1e-6
    vocab_size: int = 257152
    rope_theta: float = 10000.0


class GemmaRMSNorm(nn.Module):
    """Gemma RMSNorm: ``normed * (1 + weight)``, reduction in fp32."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        normed = x.float() * torch.rsqrt(var + self.eps)
        return (normed * (1.0 + self.weight.float())).to(dtype)


class Pi05PrefixAttention(nn.Module):
    """Bidirectional self-attention; also emits the K/V the expert consumes."""

    def __init__(self, cfg: Pi05PrefixConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.qk_scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(cfg.hidden_size,
                                self.num_heads * self.head_dim,
                                bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size,
                                self.num_kv_heads * self.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size,
                                self.num_kv_heads * self.head_dim,
                                bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                cfg.hidden_size,
                                bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from ..ops import attention_onnx, rope_onnx

        bsz, seq_len, _ = hidden_states.shape
        io_type = hidden_states.dtype
        compute_type = torch.float16

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads,
                                            self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads,
                                            self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads,
                                            self.head_dim).transpose(1, 2)

        q = rope_onnx(q.to(compute_type), rope_cos, rope_sin,
                      position_ids).to(io_type)
        k = rope_onnx(k.to(compute_type), rope_cos, rope_sin,
                      position_ids).to(io_type)
        q = q * self.qk_scale

        attn_output = attention_onnx(q,
                                     k,
                                     v,
                                     attn_mask=None,
                                     is_causal=False,
                                     scale=1.0)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        out = self.o_proj(attn_output)

        # Post-RoPE K and plain V, seq-major [B, S, H_kv, D] for the expert.
        return out, k.transpose(1, 2), v.transpose(1, 2)


class Pi05MLP(nn.Module):
    """Gemma GeGLU MLP (``hidden_act='gelu_pytorch_tanh'``)."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Pi05PrefixDecoderLayer(nn.Module):

    def __init__(self, cfg: Pi05PrefixConfig) -> None:
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(cfg.hidden_size,
                                                     cfg.rms_norm_eps)
        self.self_attn = Pi05PrefixAttention(cfg)
        self.mlp = Pi05MLP(cfg.hidden_size, cfg.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        attn_out, k, v = self.self_attn(self.input_layernorm(hidden_states),
                                        rope_cos, rope_sin, position_ids)
        hidden_states = residual + attn_out
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states, k, v


class _Pi05PrefixModel(nn.Module):
    """Container matching the checkpoint's ``layers`` nesting."""

    def __init__(self, cfg: Pi05PrefixConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Pi05PrefixDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)
        ])


class Pi05Prefix(nn.Module):
    """Prefill-only prefix tower; emits per-layer (K, V)."""

    def __init__(self, cfg: Pi05PrefixConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _Pi05PrefixModel(cfg)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        attention_pos_id: torch.Tensor,
    ) -> tuple:
        half = self.cfg.head_dim // 2
        rope_cos = rope_rotary_cos_sin[..., :half].reshape(-1, half).to(
            torch.float16)
        rope_sin = rope_rotary_cos_sin[...,
                                       half:].reshape(-1,
                                                      half).to(torch.float16)

        hidden = inputs_embeds
        ks: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []
        for layer in self.model.layers:
            hidden, k, v = layer(hidden, rope_cos, rope_sin, attention_pos_id)
            ks.append(k)
            vs.append(v)
        return tuple(ks) + tuple(vs)

    def get_onnx_export_args(self,
                             device: str) -> Tuple[tuple, list, list, tuple]:
        cfg = self.cfg
        n = cfg.num_hidden_layers
        # Batch 2 keeps the batch axis symbolic (torch.export specializes size-1).
        b, s = 2, 16

        inputs_embeds = torch.zeros(b,
                                    s,
                                    cfg.hidden_size,
                                    device=device,
                                    dtype=torch.float16)
        rope = torch.zeros(b,
                           s,
                           cfg.head_dim,
                           device=device,
                           dtype=torch.float32)
        pos = torch.arange(s, device=device,
                           dtype=torch.int32).unsqueeze(0).expand(b, -1)

        args = (inputs_embeds, rope, pos)
        input_names = [
            "inputs_embeds", "rope_rotary_cos_sin", "attention_pos_id"
        ]
        output_names = ([f"k_layer{i:02d}" for i in range(n)] +
                        [f"v_layer{i:02d}" for i in range(n)])

        batch = torch.export.Dim("batch_size", min=1, max=256)
        prefix_len = torch.export.Dim("prefix_len", min=2, max=32768)
        dynamic_shapes = ({
            0: batch,
            1: prefix_len
        }, {
            0: batch,
            1: prefix_len
        }, {
            0: batch,
            1: prefix_len
        })
        return args, input_names, output_names, dynamic_shapes


def _load_prefix_weights(model: Pi05Prefix, weights: Dict[str, torch.Tensor],
                         dtype: torch.dtype) -> None:
    """Assign prefix-tower weights from the split dict (see ``weights.py``)."""
    state = {}
    for key, tensor in weights.items():
        if key.startswith("norm."):
            continue  # final norm is dead: only K/V leave this graph
        state["model." + key] = (tensor.to(dtype)
                                 if tensor.is_floating_point() else tensor)

    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise KeyError("pi0.5 prefix parameters received no checkpoint "
                       "tensor: " + ", ".join(incompatible.missing_keys[:8]))
    if incompatible.unexpected_keys:
        logger.warning("Unexpected prefix checkpoint keys (first 5): %s",
                       incompatible.unexpected_keys[:5])
    logger.info("Loaded %d pi0.5 prefix tensors", len(state))


def build_pi05_prefix(cfg: Pi05PrefixConfig, weights: Dict[str, torch.Tensor],
                      dtype: torch.dtype) -> Pi05Prefix:
    model = Pi05Prefix(cfg).to(dtype)
    _load_prefix_weights(model, weights, dtype)
    model.eval()
    return model
