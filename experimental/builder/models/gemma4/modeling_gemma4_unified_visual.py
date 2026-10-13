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
"""Gemma4 Unified encoder-free visual embedding graph."""

import math

import numpy as np
import tensorrt as trt

from ...ops import Module, NetworkModule
from ...ops import functional as F


class Gemma4UnifiedLayerNorm(Module):
    """FP32 LayerNorm for the high-dynamic-range packed-patch path."""

    def __init__(self,
                 ctx,
                 prefix: str,
                 hidden_size: int,
                 eps: float = 1e-5) -> None:
        super().__init__(ctx, prefix)
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, hidden):
        weight = self.weights.f32(self.key("weight"))
        bias = (self.weights.f32(self.key("bias")) if self.weights.has(
            self.key("bias")) else np.zeros(self.hidden_size,
                                            dtype=np.float32))
        return F.layer_norm(hidden, weight, bias, self.eps, hidden.rank)


class Gemma4UnifiedUnitRMSNorm(Module):
    """Provider weightless RMSNorm before multimodal projection."""

    def __init__(self, ctx, hidden_size: int, eps: float) -> None:
        super().__init__(ctx)
        self.weight = np.ones(hidden_size, dtype=np.float32)
        self.eps = eps

    def forward(self, hidden):
        return F.rms_norm(hidden, self.weight, self.eps, hidden.rank)


class Gemma4UnifiedF32Linear(Module):
    """Checkpoint-backed FP32 projection."""

    def __init__(self, ctx, prefix: str, bias: bool) -> None:
        super().__init__(ctx, prefix)
        self.has_bias = bias

    def forward(self, hidden):
        bias = (self.weights.f32(self.key("bias")) if self.has_bias else None)
        return F.linear_f32(hidden,
                            self.weights.f32(self.key("weight")),
                            bias,
                            rank=hidden.rank)


class Gemma4UnifiedVisionEmbedder(Module):
    """Project packed RGB patches and add factorized 2-D positions."""

    def __init__(self, ctx, config: dict) -> None:
        super().__init__(ctx, "vision_embedder")
        patch_weight = self.weights.parameter_spec(
            self.key("patch_dense.weight"), np.float32)
        if len(patch_weight.shape) != 2:
            raise ValueError(
                "Gemma4 Unified patch projection must be a matrix")
        patch_dim = int(patch_weight.shape[1])
        patch_area, channels = divmod(patch_dim, 3)
        patch_size = math.isqrt(patch_area)
        if channels or patch_size * patch_size != patch_area:
            raise ValueError(
                "Gemma4 Unified patch projection input must be square RGB "
                f"patches, got width {patch_dim}")
        configured_patch_size = config.get("model_patch_size")
        if (configured_patch_size is not None
                and int(configured_patch_size) != patch_size):
            raise ValueError(
                "Gemma4 Unified model_patch_size disagrees with checkpoint: "
                f"{configured_patch_size} != {patch_size}")
        hidden_size = int(config["mm_embed_dim"])
        output_size = int(config["output_proj_dims"])
        if hidden_size != output_size:
            raise ValueError(
                "Gemma4 Unified mm_embed_dim must equal output_proj_dims")
        self.patch_dim = patch_dim
        self.hidden_size = hidden_size
        self.patch_ln1 = Gemma4UnifiedLayerNorm(ctx, self.key("patch_ln1"),
                                                patch_dim)
        self.patch_dense = Gemma4UnifiedF32Linear(ctx, self.key("patch_dense"),
                                                  True)
        self.patch_ln2 = Gemma4UnifiedLayerNorm(ctx, self.key("patch_ln2"),
                                                hidden_size)
        self.pos_norm = Gemma4UnifiedLayerNorm(ctx, self.key("pos_norm"),
                                               hidden_size)

    def forward(self, pixels, position_ids):
        hidden = self.patch_ln1(pixels.cast(trt.float32))
        hidden = self.patch_ln2(self.patch_dense(hidden))
        positions = position_ids.maximum(np.int64(0))
        axes = positions * np.int64(0) + np.asarray([[0, 1]], dtype=np.int64)
        indices = F.concatenate(
            (positions.unsqueeze(2, 2), axes.unsqueeze(2, 2)), 2)
        table = F.constant(self.weights.f32(self.key("pos_embedding")),
                           "unified_position_embedding")
        positional = F.gather_nd(table, indices)
        valid = np.float32(1.0) - position_ids.equal(np.int64(-1)).cast(
            trt.float32)
        positional = (positional * valid.unsqueeze(2, 2)).sum(1, keepdim=False)
        return self.pos_norm(hidden.cast(trt.float32) + positional)


class Gemma4UnifiedMultimodalEmbedder(Module):
    """Weightless RMSNorm and FP32 projection into text hidden space."""

    def __init__(self, ctx, prefix: str, config: dict) -> None:
        super().__init__(ctx, prefix)
        hidden_size = int(config["output_proj_dims"])
        self.norm = Gemma4UnifiedUnitRMSNorm(
            ctx, hidden_size, float(config.get("rms_norm_eps", 1e-6)))
        self.projection = Gemma4UnifiedF32Linear(
            ctx, self.key("embedding_projection"), False)

    def forward(self, hidden):
        return self.projection(self.norm(hidden))


class Gemma4UnifiedVisualModel(NetworkModule):
    """Packed image patches to language-model embeddings."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle.root)

    def __init__(self, ctx, root: dict) -> None:
        super().__init__(ctx)
        config = root.get("vision_config") or root
        self.vision_embedder = Gemma4UnifiedVisionEmbedder(ctx, config)
        self.embed_vision = Gemma4UnifiedMultimodalEmbedder(
            ctx, "embed_vision", config)

    def input_tensors(self):
        return {
            "pixels":
            self.add_input("input", trt.float16,
                           (-1, self.vision_embedder.patch_dim)),
            "positions":
            self.add_input("pixel_position_ids", trt.int64, (-1, 2)),
        }

    def forward(self, pixels, positions):
        hidden = self.vision_embedder(pixels, positions)
        return {"output": self.embed_vision(hidden).cast(trt.float16)}
