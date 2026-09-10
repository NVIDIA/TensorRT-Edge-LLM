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
"""
From-scratch InternVL3 vision encoder (model_type: ``internvl_chat``).

Architecture (intern_vit_6b style):
    InternVL3Embeddings
        class_embedding [1, 1, H]   +  patch_embedding Conv2d
        + position_embedding [1, N+1, H]  (learnable)
    → 24 × InternVL3VisionLayer
        norm1  →  attn(qkv, proj) × ls1  →  residual
        norm2  →  mlp(fc1, GELU, fc2) × ls2  →  residual
    → pixel_shuffle downsampling (downsample_ratio)
    → mlp1 projector: LayerNorm → Linear → GELU → Linear

Checkpoint weight key prefixes:
    Vision encoder:  ``vision_model.*``
    Projector:       ``mlp1.*``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import config as config_module
from ..linear import make_linear
from ..ops import init_fp8_mha, quantize_qkv_for_fp8_mha, vit_attention_plugin

if TYPE_CHECKING:
    from ...config import ModelConfig

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _make_norm(norm_type: str, hidden_size: int, eps: float) -> nn.Module:
    if norm_type == "rms_norm":
        return _RMSNorm(hidden_size, eps)
    return nn.LayerNorm(hidden_size, eps=eps)


class _RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(input_dtype) * x.to(input_dtype)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class InternVL3Embeddings(nn.Module):
    """CLS token + patch embedding + learnable absolute position embedding.

    Checkpoint keys (under ``vision_model.embeddings.``):
        ``class_embedding``          [1, 1, hidden_size]
        ``patch_embedding.weight``   [hidden_size, num_channels, patch_size, patch_size]
        ``patch_embedding.bias``     [hidden_size]
        ``position_embedding``       [1, num_patches + 1, hidden_size]
    """

    def __init__(self, num_channels: int, hidden_size: int, patch_size: int,
                 image_size: int) -> None:
        super().__init__()
        num_patches = (image_size // patch_size)**2
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.patch_embedding = nn.Conv2d(num_channels,
                                         hidden_size,
                                         kernel_size=patch_size,
                                         stride=patch_size,
                                         bias=True)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.size(0)
        # [B, H, h, w] → [B, N, H]
        patch_embeds = self.patch_embedding(
            pixel_values.to(
                self.patch_embedding.weight.dtype)).flatten(2).transpose(1, 2)
        cls_tokens = self.class_embedding.expand(B, -1, -1)
        x = torch.cat((cls_tokens, patch_embeds), dim=1)
        x = x + self.position_embedding
        return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class InternVL3VisionAttention(nn.Module):
    """Combined-QKV MHA for InternVL3.

    Checkpoint keys (under ``vision_model.encoder.layers.N.attn.``):
        ``qkv.weight``, ``qkv.bias``  [3*hidden, hidden]
        ``proj.weight``, ``proj.bias`` [hidden, hidden]
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 attention_scale: float,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attention_scale = attention_scale
        self.embed_dim = hidden_size
        self.qkv = make_linear(
            model_config,
            hidden_size,
            hidden_size * 3,
            bias=True,
            module_name=f"{name_prefix}.qkv" if name_prefix else "")
        self.proj = make_linear(
            model_config,
            hidden_size,
            hidden_size,
            bias=True,
            module_name=f"{name_prefix}.proj" if name_prefix else "")
        # Fused-qkv attention: init_fp8_mha creates synthetic k_proj/v_proj
        # placeholders for the modelopt scale paths (see its docstring).
        # InternViT-300M d64 / InternViT-6B d128 both have FP8 kernels.
        self.enable_fp8_mha = init_fp8_mha(self, model_config, self.head_dim)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                max_seqlen_carrier: torch.Tensor) -> torch.Tensor:
        B, N, _ = hidden_states.shape
        q, k, v = self.qkv(hidden_states).reshape(
            B * N, 3 * self.num_heads, self.head_dim).split(self.num_heads,
                                                            dim=1)
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
        q, k, v, qkv_scales = quantize_qkv_for_fp8_mha(self, q, k, v)
        attn_output = vit_attention_plugin(
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen_carrier,
            num_heads=self.num_heads,
            head_size=self.head_dim,
            attention_scale=self.attention_scale,
            qkv_scales=qkv_scales,
        )
        out = attn_output.reshape(B, N, self.embed_dim)
        return self.proj(out)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class InternVL3VisionMLP(nn.Module):
    """Two-layer GELU FFN.

    Checkpoint keys: ``mlp.fc1.*``, ``mlp.fc2.*``
    """

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 hidden_act: str,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.fc1 = make_linear(
            model_config,
            hidden_size,
            intermediate_size,
            bias=True,
            module_name=f"{name_prefix}.fc1" if name_prefix else "")
        self.fc2 = make_linear(
            model_config,
            intermediate_size,
            hidden_size,
            bias=True,
            module_name=f"{name_prefix}.fc2" if name_prefix else "")
        act_map = {
            "gelu": F.gelu,
            "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
            "relu": F.relu,
            "silu": F.silu,
        }
        self.act_fn = act_map.get(hidden_act, F.gelu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))


# ---------------------------------------------------------------------------
# Vision Layer
# ---------------------------------------------------------------------------


class InternVL3VisionLayer(nn.Module):
    """Single InternVL3 vision transformer layer with layer-scale (ls1, ls2).

    Checkpoint keys (under ``vision_model.encoder.layers.N.``):
        ``norm1.*``, ``attn.*``, ``ls1``
        ``norm2.*``, ``mlp.*``, ``ls2``
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 intermediate_size: int,
                 hidden_act: str,
                 norm_type: str,
                 layer_norm_eps: float,
                 attention_scale: float,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.norm1 = _make_norm(norm_type, hidden_size, layer_norm_eps)
        self.attn = InternVL3VisionAttention(
            hidden_size,
            num_heads,
            attention_scale,
            model_config,
            name_prefix=f"{name_prefix}.attn" if name_prefix else "")
        self.ls1 = nn.Parameter(torch.ones(hidden_size))
        self.norm2 = _make_norm(norm_type, hidden_size, layer_norm_eps)
        self.mlp = InternVL3VisionMLP(
            hidden_size,
            intermediate_size,
            hidden_act,
            model_config,
            name_prefix=f"{name_prefix}.mlp" if name_prefix else "")
        self.ls2 = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                max_seqlen_carrier: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.ls1 * self.attn(
            self.norm1(hidden_states), cu_seqlens, max_seqlen_carrier)
        hidden_states = hidden_states + self.ls2 * self.mlp(
            self.norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Vision Encoder
# ---------------------------------------------------------------------------


class InternVL3VisionEncoder(nn.Module):
    """Stack of InternVL3VisionLayer.

    Checkpoint keys: ``vision_model.encoder.layers.N.*``
    """

    def __init__(self,
                 num_hidden_layers: int,
                 hidden_size: int,
                 num_heads: int,
                 intermediate_size: int,
                 hidden_act: str,
                 norm_type: str,
                 layer_norm_eps: float,
                 attention_scale: float,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            InternVL3VisionLayer(
                hidden_size,
                num_heads,
                intermediate_size,
                hidden_act,
                norm_type,
                layer_norm_eps,
                attention_scale,
                model_config,
                name_prefix=f"{name_prefix}.layers.{i}" if name_prefix else "")
            for i in range(num_hidden_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, _ = hidden_states.shape
        # Uniform-length ragged descriptors for the ViT attention plugin: every
        # image contributes exactly N tokens, so cu_seqlens is a static-stride
        # ramp and the max-seqlen carrier is shape-only (its length N is what
        # the plugin reads; contents are never touched).
        cu_seqlens = torch.arange(0, (B + 1) * N,
                                  N,
                                  dtype=torch.int32,
                                  device=hidden_states.device)
        max_seqlen_carrier = torch.zeros(N,
                                         dtype=torch.int32,
                                         device=hidden_states.device)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens,
                                  max_seqlen_carrier)
        return hidden_states


# ---------------------------------------------------------------------------
# Pixel shuffle
# ---------------------------------------------------------------------------


def _pixel_shuffle(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Spatial downsampling via pixel shuffle (merge spatial → channel)."""
    B, H, W, C = x.shape
    scale = int(1.0 / scale_factor)
    new_H = H // scale
    new_W = W // scale
    x = x.view(B, new_H, scale, new_W, scale, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, new_H, new_W, scale * scale * C)
    return x


# ---------------------------------------------------------------------------
# Top-level visual model
# ---------------------------------------------------------------------------


class InternVL3VisualModel(nn.Module):
    """From-scratch InternVL3 visual encoder + projector (model_type: internvl_chat).

    Expects the full ``config.json`` dict (not just ``vision_config``), because it
    needs ``llm_config.hidden_size`` and ``downsample_ratio`` from the top level.

    Output: ``[total_patches * tokens_per_patch, llm_hidden_size]``
    """

    def __init__(self, config: dict, model_config: "ModelConfig") -> None:
        super().__init__()
        vc = config["vision_config"]
        # LLM hidden size lives under llm_config for internvl_chat
        llm_cfg = config.get("llm_config", config.get("text_config", {}))
        llm_hidden_size: int = llm_cfg["hidden_size"]
        self.downsample_ratio: float = float(config["downsample_ratio"])

        image_size = vc["image_size"]
        patch_size = vc["patch_size"]
        hidden_size: int = vc["hidden_size"]
        head_dim = hidden_size // vc["num_attention_heads"]
        attention_scale = config_module._get_attention_scaling(
            vc, head_dim, 1.0 / (float(head_dim)**0.5))

        self.embeddings = InternVL3Embeddings(
            num_channels=vc.get("num_channels", 3),
            hidden_size=hidden_size,
            patch_size=patch_size,
            image_size=image_size,
        )
        self.encoder = InternVL3VisionEncoder(
            num_hidden_layers=vc["num_hidden_layers"],
            hidden_size=hidden_size,
            num_heads=vc["num_attention_heads"],
            intermediate_size=vc["intermediate_size"],
            hidden_act=vc.get("hidden_act", "gelu"),
            norm_type=vc.get("norm_type", "layer_norm"),
            layer_norm_eps=vc.get("layer_norm_eps", 1e-6),
            attention_scale=attention_scale,
            model_config=model_config,
            name_prefix="vision_model.encoder",
        )

        scale = int(1.0 / self.downsample_ratio)
        in_dim = hidden_size * scale * scale
        # Keep nn.Sequential so checkpoint keys (mlp1.0.* / mlp1.1.* / mlp1.3.*)
        # still resolve; ``make_linear`` returns ``FP16Linear`` or a quantised
        # subclass and ``_set_tensor`` handles both buffer and parameter writes.
        # Pass the index-qualified name so MIXED_PRECISION ``layer_overrides``
        # entries (``mlp1.1`` / ``mlp1.3``) resolve.
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            make_linear(model_config,
                        in_dim,
                        llm_hidden_size,
                        bias=True,
                        module_name="mlp1.1"),
            nn.GELU(),
            make_linear(model_config,
                        llm_hidden_size,
                        llm_hidden_size,
                        bias=True,
                        module_name="mlp1.3"),
        )
        self._llm_hidden_size = llm_hidden_size
        # Precomputed to avoid int(math.isqrt(N)) on a runtime SymInt in forward,
        # which would specialize the batch dimension during torch.export tracing.
        self._feat_side: int = image_size // patch_size

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [num_patches, C, H, W]

        Returns:
            image_features: [num_patches * tokens_per_patch, llm_hidden_size]
        """
        # Vision encoder: [B, N+1, H]
        x = self.embeddings(pixel_values)
        x = self.encoder(x)

        # Remove CLS token: [B, N, H]
        x = x[:, 1:, :]

        B = x.shape[0]
        C = x.shape[2]
        feat_side = self._feat_side  # precomputed in __init__
        # Reshape to spatial: [B, H, W, C]
        x = x.reshape(B, feat_side, feat_side, C)
        # Pixel shuffle downsampling
        x = _pixel_shuffle(x, self.downsample_ratio)
        # Flatten spatial: [B, N_out, in_dim]
        x = x.reshape(B, -1, x.shape[-1])
        # Project: [B, N_out, llm_hidden_size]
        x = self.mlp1(x)
        # Flatten batch: [B*N_out, llm_hidden_size]
        return x.reshape(-1, self._llm_hidden_size)

    def get_onnx_export_args(self, config: dict, device: str):
        """Return (args, input_names, output_names, dynamic_shapes) for ONNX export."""
        vc = config.get("vision_config", config)
        image_size = vc["image_size"]
        num_channels = vc.get("num_channels", 3)
        # Use batch=2 so the tracer sees a non-unit batch and keeps it symbolic.
        pixel_values = torch.zeros(2,
                                   num_channels,
                                   image_size,
                                   image_size,
                                   dtype=torch.float16,
                                   device=device)
        args = (pixel_values, )
        input_names = ["input"]
        output_names = ["output"]
        N = torch.export.Dim("num_images", min=1)
        dynamic_shapes = {"pixel_values": {0: N}}
        return args, input_names, output_names, dynamic_shapes


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _load_internvl3_weights(model: InternVL3VisualModel,
                            weights: dict) -> None:
    """Load vision_model and mlp1 weights into *model*.

    Checkpoint key → model attribute path:
      ``vision_model.embeddings.*`` / ``vision_model.encoder.*`` →
        attribute path is ``embeddings.*`` / ``encoder.*``
      ``mlp1.*`` is already at the right path
    """
    from ...checkpoint.loader import load_submodule_weights

    def _remap(k: str) -> "str | None":
        if k.startswith("vision_model."):
            return k[len("vision_model."):]
        if k.startswith("mlp1."):
            return k
        return None

    load_submodule_weights(model,
                           weights,
                           _remap,
                           label="InternVL3VisualModel")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_internvl_visual(
        config: dict,
        weights: dict,
        model_config: "ModelConfig",
        dtype: torch.dtype = torch.float16) -> InternVL3VisualModel:
    """Build and return an :class:`InternVL3VisualModel` with loaded weights.

    Args:
        config:       Full parsed ``config.json`` dict (contains
                      ``vision_config``, ``llm_config``, ``downsample_ratio``).
        weights:      Flat ``{key: tensor}`` dict from safetensors.
        model_config: Top-level ``ModelConfig``.  Visual layers dispatch
                      through ``make_linear`` so quantized checkpoints are
                      honoured; an FP16 checkpoint yields ``FP16Linear``.
        dtype:        Target dtype (default ``float16``).
    """
    model = InternVL3VisualModel(config, model_config=model_config)
    model.to(dtype)
    _load_internvl3_weights(model, weights)
    model.eval()
    return model


__all__ = [
    "InternVL3VisualModel",
    "build_internvl_visual",
]
