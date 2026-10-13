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
"""pi0.5 SigLIP-So400m/14 vision tower + PaliGemma multi-modal projector.

One 224x224 view yields 256 tokens projected into the prefix tower's embedding
space. Attention is per-view (each view is its own batch row), bidirectional and
without RoPE, so it runs on the TRT-native ``trt::attention_onnx``.

Two deviations from stock HuggingFace PaliGemma, both taken from openpi's
``transformers_replace``:
  - ``get_image_features`` does **not** divide by ``sqrt(text_hidden_size)``.
  - the SigLIP attention-pooling head is absent (the converted checkpoint
    carries no head weights).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class Pi05VisualConfig:
    """Hyperparameters for the pi0.5 vision tower."""

    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    patch_size: int = 14
    image_size: int = 224
    layer_norm_eps: float = 1e-6
    projection_dim: int = 2048

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_positions(self) -> int:
        return (self.image_size // self.patch_size)**2


class Pi05VisionEmbeddings(nn.Module):
    """Patch convolution plus a learned absolute position embedding."""

    def __init__(self, cfg: Pi05VisualConfig) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv2d(3,
                                         cfg.hidden_size,
                                         kernel_size=cfg.patch_size,
                                         stride=cfg.patch_size,
                                         padding="valid")
        self.position_embedding = nn.Embedding(cfg.num_positions,
                                               cfg.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(pixel_values)
        embeddings = patches.flatten(2).transpose(1, 2)
        return embeddings + self.position_embedding.weight[None, :, :]


class Pi05VisionAttention(nn.Module):
    """Bidirectional per-view self-attention (no RoPE, no mask)."""

    def __init__(self, cfg: Pi05VisualConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.qk_scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from ..ops import attention_onnx

        bsz, seq_len, _ = hidden_states.shape
        io_type = hidden_states.dtype

        def _heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(bsz, seq_len, self.num_heads,
                          self.head_dim).transpose(1, 2)

        q = _heads(self.q_proj(hidden_states))
        k = _heads(self.k_proj(hidden_states))
        v = _heads(self.v_proj(hidden_states))

        # Scaled inside the attention, as openpi does. head_dim is 72 here, so the
        # factor is not a power of two and pre-scaling would round every query element.
        attn_output = attention_onnx(q.to(io_type),
                                     k.to(io_type),
                                     v.to(io_type),
                                     attn_mask=None,
                                     is_causal=False,
                                     scale=self.qk_scale)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.out_proj(attn_output)


class Pi05VisionMLP(nn.Module):

    def __init__(self, cfg: Pi05VisualConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Pi05VisionEncoderLayer(nn.Module):

    def __init__(self, cfg: Pi05VisualConfig) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size,
                                        eps=cfg.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size,
                                        eps=cfg.layer_norm_eps)
        self.self_attn = Pi05VisionAttention(cfg)
        self.mlp = Pi05VisionMLP(cfg)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states))
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class Pi05VisionEncoder(nn.Module):

    def __init__(self, cfg: Pi05VisualConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Pi05VisionEncoderLayer(cfg) for _ in range(cfg.num_hidden_layers)
        ])


class Pi05VisionModel(nn.Module):
    """SigLIP tower without the attention-pooling head."""

    def __init__(self, cfg: Pi05VisualConfig) -> None:
        super().__init__()
        self.embeddings = Pi05VisionEmbeddings(cfg)
        self.encoder = Pi05VisionEncoder(cfg)
        self.post_layernorm = nn.LayerNorm(cfg.hidden_size,
                                           eps=cfg.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        for layer in self.encoder.layers:
            hidden_states = layer(hidden_states)
        return self.post_layernorm(hidden_states)


class _Pi05VisionTower(nn.Module):
    """Container matching the checkpoint's ``vision_tower.vision_model`` path."""

    def __init__(self, cfg: Pi05VisualConfig) -> None:
        super().__init__()
        self.vision_model = Pi05VisionModel(cfg)


class _Pi05MultiModalProjector(nn.Module):

    def __init__(self, cfg: Pi05VisualConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(cfg.hidden_size, cfg.projection_dim, bias=True)


class Pi05Visual(nn.Module):
    """``[N, 3, 224, 224] -> [N, 256, projection_dim]``."""

    def __init__(self, cfg: Pi05VisualConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_tower = _Pi05VisionTower(cfg)
        self.multi_modal_projector = _Pi05MultiModalProjector(cfg)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.vision_tower.vision_model(pixel_values)
        # No 1/sqrt(hidden) scaling here: openpi removed it from the upstream
        # PaliGemma get_image_features.
        return self.multi_modal_projector.linear(features)

    def get_onnx_export_args(self,
                             device: str) -> Tuple[tuple, list, list, dict]:
        cfg = self.cfg
        # Batch 2 keeps the view axis symbolic (torch.export specializes size-1).
        pixel_values = torch.zeros(2,
                                   3,
                                   cfg.image_size,
                                   cfg.image_size,
                                   device=device,
                                   dtype=torch.float16)
        num_views = torch.export.Dim("num_views", min=1, max=256)
        return (pixel_values, ), ["pixel_values"], ["image_features"], {
            "pixel_values": {
                0: num_views
            }
        }


def _load_visual_weights(model: Pi05Visual, weights: Dict[str, torch.Tensor],
                         dtype: torch.dtype) -> None:
    """Assign vision-tower and projector weights from the split dict."""
    state = {
        k: (v.to(dtype) if v.is_floating_point() else v)
        for k, v in weights.items()
    }
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise KeyError("pi0.5 visual parameters received no checkpoint "
                       "tensor: " + ", ".join(incompatible.missing_keys[:8]))
    if incompatible.unexpected_keys:
        logger.warning("Unexpected visual checkpoint keys (first 5): %s",
                       incompatible.unexpected_keys[:5])
    logger.info("Loaded %d pi0.5 visual tensors", len(state))


def build_pi05_visual(cfg: Pi05VisualConfig, weights: Dict[str, torch.Tensor],
                      dtype: torch.dtype) -> Pi05Visual:
    model = Pi05Visual(cfg).to(dtype)
    _load_visual_weights(model, weights, dtype)
    model.eval()
    return model
