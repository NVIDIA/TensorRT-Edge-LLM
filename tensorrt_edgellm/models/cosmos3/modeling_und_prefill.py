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
"""Cosmos3 UND tower as a PREFILL-ONLY per-layer K/V producer for the policy path.

The policy/action path runs the understanding (UND) tower ONCE as a prefill to
produce the per-layer K/V the GEN diffusion expert cross-attends to — it never
does incremental (KV-cache) decode.  The KV-cache ``AttentionPlugin`` (FMHA
prefill + XQA decode) used by the autoregressive text engine cannot be compiled
by TensorRT's Myelin backend when interleaved with the fp16 pointwise chain
(stride_order CHECK failure, both TRT 10.16 and 11).  This module avoids that
entirely: causal self-attention via the SAME prefill ``trt::attention_onnx`` /
``trt::rope_onnx`` ops the GEN graph uses (which build fine), emitting the 28
per-layer K/V as graph outputs in seq-major ``[B, S, H_kv, D]`` — exactly the
layout the GEN graph's ``und_k_layerNN`` / ``und_v_layerNN`` inputs expect.

Policy-variant flags: non-gated squared-ReLU MLP (``down(relu(up(x))**2)``) and
no UND qk-norm.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling_gen import RMSNorm

logger = logging.getLogger(__name__)


class _UndAttn(nn.Module):

    def __init__(self,
                 hidden: int,
                 n_heads: int,
                 n_kv: int,
                 head_dim: int,
                 use_und_k_norm: bool = False,
                 rms_eps: float = 1e-6) -> None:
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim = n_heads, n_kv, head_dim
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)
        self.qk_scale = 1.0 / math.sqrt(head_dim)
        # Per-head RMSNorm applied to the UND K that is exported for the GEN
        # cross-attention (Cosmos3-Edge use_und_k_norm_for_gen=True). The reasoner's
        # own self-attention keeps the raw (un-normed) K. None => share the raw K.
        self.k_norm_und_for_gen = RMSNorm(head_dim,
                                          rms_eps) if use_und_k_norm else None

    def forward(self, h, rope_cos, rope_sin, pos, causal_mask):
        from tensorrt_edgellm.models.ops import attention_onnx, rope_onnx

        b, s, _ = h.shape
        io = h.dtype
        q = self.q_proj(h).view(b, s, self.n_heads,
                                self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        # UND has no qk-norm (qk_norm_und=false). Qwen3 RoPE (unified_3d_mrope cos/sin).
        q = rope_onnx(q.to(torch.float16), rope_cos, rope_sin, pos).to(io)
        k_self = rope_onnx(k.to(torch.float16), rope_cos, rope_sin, pos).to(io)
        q = q * self.qk_scale
        attn = attention_onnx(q,
                              k_self,
                              v,
                              attn_mask=causal_mask,
                              is_causal=False,
                              scale=1.0)
        attn = attn.transpose(1, 2).reshape(b, s, -1)
        out = self.o_proj(attn)
        # K exported for the GEN cross-attention. With use_und_k_norm_for_gen
        # (Cosmos3-Edge), the released model applies RMSNorm(K) BEFORE RoPE
        # (per-head over head_dim) so the gen tower attends to normalized UND K;
        # the reasoner's own self-attention above keeps the raw (un-normed) K.
        if self.k_norm_und_for_gen is not None:
            k_gen = self.k_norm_und_for_gen(k)
            k_gen = rope_onnx(k_gen.to(torch.float16), rope_cos, rope_sin,
                              pos).to(io)
        else:
            k_gen = k_self
        # Per-layer K/V (post-RoPE K, plain V) seq-major [B, S, H_kv, D] for the GEN graph.
        return out, k_gen.transpose(1, 2), v.transpose(1, 2)


class _UndMLP(nn.Module):
    """Non-gated Nemotron-H MLP ``down(act(up(x)))``; ``hidden_act='relu2'``
    is squared ReLU (the Cosmos3-Edge text tower)."""

    def __init__(self, hidden: int, inter: int, hidden_act: str) -> None:
        super().__init__()
        if hidden_act == "relu2":
            self.act_fn = lambda x: F.relu(x).square()
        elif hidden_act == "silu":
            self.act_fn = F.silu
        else:
            raise ValueError(
                f"Unsupported Cosmos3 UND hidden_act: {hidden_act!r}")
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class _UndLayer(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"],
                                                cfg["rms_norm_eps"])
        self.self_attn = _UndAttn(cfg["hidden_size"],
                                  cfg["num_attention_heads"],
                                  cfg["num_key_value_heads"],
                                  cfg["head_dim"],
                                  use_und_k_norm=cfg.get(
                                      "use_und_k_norm_for_gen", False),
                                  rms_eps=cfg["rms_norm_eps"])
        self.mlp = _UndMLP(cfg["hidden_size"], cfg["intermediate_size"],
                           cfg["hidden_act"])

    def forward(self, x, rope_cos, rope_sin, pos, causal_mask):
        attn_out, k, v = self.self_attn(self.input_layernorm(x), rope_cos,
                                        rope_sin, pos, causal_mask)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, k, v


class _UndModel(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_UndLayer(cfg) for _ in range(cfg["num_hidden_layers"])])
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])


class Cosmos3UndPrefill(nn.Module):
    """Prefill-only UND tower; emits 28 per-layer (K, V) + final hidden_states."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.n = cfg["num_hidden_layers"]
        self.head_dim = cfg["head_dim"]
        self.model = _UndModel(cfg)

    def forward(self, inputs_embeds, rope_rotary_cos_sin, attention_pos_id):
        half = self.head_dim // 2
        rope_cos = rope_rotary_cos_sin[..., :half].reshape(-1, half).to(
            torch.float16)
        rope_sin = rope_rotary_cos_sin[...,
                                       half:].reshape(-1,
                                                      half).to(torch.float16)
        # TensorRT 10.16 squares the softmax scale of an is_causal Attention
        # built with a dynamic sequence profile, so pass the causal mask explicitly.
        future = (attention_pos_id[:, None, None, :]
                  > attention_pos_id[:, None, :, None])
        causal_mask = torch.zeros_like(future,
                                       dtype=torch.float16).masked_fill(
                                           future, float("-inf"))
        x = inputs_embeds
        ks: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []
        for layer in self.model.layers:
            x, k, v = layer(x, rope_cos, rope_sin, attention_pos_id,
                            causal_mask)
            ks.append(k)
            vs.append(v)
        hidden = self.model.norm(x)
        return tuple(ks) + tuple(vs) + (hidden, )

    def get_onnx_export_args(self,
                             device: str = "cpu"
                             ) -> Tuple[tuple, list, list, tuple]:
        cfg = self.cfg
        n, hkv, d = self.n, cfg["num_key_value_heads"], self.head_dim
        b, s = 1, 13
        inputs_embeds = torch.zeros(b,
                                    s,
                                    cfg["hidden_size"],
                                    dtype=torch.float16,
                                    device=device)
        rope = torch.zeros(b, s, d, dtype=torch.float32, device=device)
        pos = torch.arange(s, dtype=torch.int32, device=device).unsqueeze(0)
        args = (inputs_embeds, rope, pos)
        input_names = [
            "inputs_embeds", "rope_rotary_cos_sin", "attention_position_ids"
        ]
        output_names = ([f"und_k_layer{i:02d}" for i in range(n)] +
                        [f"und_v_layer{i:02d}"
                         for i in range(n)] + ["hidden_states"])
        batch = torch.export.Dim("batch", min=1, max=256)
        seq = torch.export.Dim("und_len", min=2, max=32768)
        dynamic_shapes = ({
            0: batch,
            1: seq
        }, {
            0: batch,
            1: seq
        }, {
            0: batch,
            1: seq
        })
        return args, input_names, output_names, dynamic_shapes


def build_cosmos3_und_prefill(transformer_cfg: dict,
                              und_weights: Dict[str, torch.Tensor],
                              dtype: torch.dtype):
    """Build the prefill UND tower and load the UND split weights (HF/Qwen naming)."""
    cfg = {
        "hidden_size":
        transformer_cfg["hidden_size"],
        "num_hidden_layers":
        transformer_cfg["num_hidden_layers"],
        "num_attention_heads":
        transformer_cfg["num_attention_heads"],
        "num_key_value_heads":
        transformer_cfg["num_key_value_heads"],
        "head_dim":
        transformer_cfg["head_dim"],
        "intermediate_size":
        transformer_cfg["intermediate_size"],
        # Required: the activation defines the MLP graph ('relu2' = squared
        # ReLU for Cosmos3-Edge); no silent default.
        "hidden_act":
        str(transformer_cfg["hidden_act"]),
        # Required: the runtime builds the text rope cache from this value
        # (Cosmos3-Edge uses 1e8); no silent default.
        "rope_theta":
        float(transformer_cfg["rope_theta"]),
        "rms_norm_eps":
        transformer_cfg.get("rms_norm_eps", 1e-6),
        # Released Cosmos3-Edge normalizes the UND K fed to the gen tower; older
        # (pre-release) checkpoints omit it (weight absent, flag False).
        "use_und_k_norm_for_gen":
        bool(transformer_cfg.get("use_und_k_norm_for_gen", False)),
    }
    model = Cosmos3UndPrefill(cfg).to(dtype)
    state = {}
    for key, t in und_weights.items():
        if key.startswith(("embed_tokens.", "lm_head.")):
            continue  # embed applied on host; lm_head not used in prefill
        state["model." + key] = t.to(dtype) if t.is_floating_point() else t
    missing = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing.missing_keys]
    logger.info(
        "Cosmos3 UND-prefill load: assigned=%d missing=%d unexpected=%d",
        len(state), len(real_missing), len(missing.unexpected_keys))
    if real_missing:
        logger.warning("UND-prefill missing (first 5): %s", real_missing[:5])
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Component contract (consumed by the experimental Cosmos3 builder/runtime)
# ---------------------------------------------------------------------------


def make_und_prefill_config(cfg: dict, max_und_len: int) -> dict:
    """Return the UND-prefill component ``config.json`` payload."""
    hd = cfg["head_dim"]
    hs = cfg["hidden_size"]
    return {
        "component": "und_prefill",
        "onnx_filename": "model.onnx",
        "engine_filename": "und_prefill.engine",
        "optimization_profile": {
            "inputs_embeds": {
                "min": [1, 2, hs],
                "opt": [1, 16, hs],
                "max": [1, max_und_len, hs],
            },
            "rope_rotary_cos_sin": {
                "min": [1, 2, hd],
                "opt": [1, 16, hd],
                "max": [1, max_und_len, hd],
            },
            "attention_position_ids": {
                "min": [1, 2],
                "opt": [1, 16],
                "max": [1, max_und_len],
            },
        },
        "tensor_contract": {
            "inputs": {
                "inputs_embeds": ["batch", "und_len", "hidden_size"],
                "rope_rotary_cos_sin": ["batch", "und_len", "head_dim"],
                "attention_position_ids": ["batch", "und_len"],
            },
            "outputs": {
                "und_k_layerNN":
                ["batch", "und_len", "num_kv_heads", "head_dim"],
                "und_v_layerNN":
                ["batch", "und_len", "num_kv_heads", "head_dim"],
                "hidden_states": ["batch", "und_len", "hidden_size"],
            },
        },
        "builder_config": {
            "max_batch_size": 1,
            "max_und_len": max_und_len
        },
        "num_hidden_layers": cfg["num_hidden_layers"],
        "hidden_size": hs,
        "num_key_value_heads": cfg["num_key_value_heads"],
        "head_dim": hd,
        "rope_theta": cfg["rope_theta"],
    }
