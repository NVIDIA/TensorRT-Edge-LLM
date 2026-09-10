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
"""Encoder-free Gemma 4 Unified visual embedding export model.

Gemma 4 Unified does not have the ViT used by the smaller Gemma 4 models.
The image processor merges each 3x3 group of 16x16 RGB teacher patches into
one flattened 48x48 model patch.  This module maps those packed patches
directly into the language-model hidden space::

    patch [N, 6912]
      -> LayerNorm -> Linear -> LayerNorm
      -> factorized x/y positional embedding -> LayerNorm
      -> weightless RMSNorm -> Linear
      -> embedding [N, 3840]

Checkpoint prefixes are ``model.vision_embedder.*`` and
``model.embed_vision.*`` in the released layout; ``save_pretrained`` output
nests the same tensors differently. Both load, see
``_MODULE_TREE_PREFIX_REWRITES``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..linear import make_linear

if TYPE_CHECKING:
    from ...config import ModelConfig

logger = logging.getLogger(__name__)


class Gemma4UnifiedRMSNorm(nn.Module):
    """Weightless RMSNorm used before each Unified multimodal projection."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return hidden_states.to(input_dtype)


class Gemma4UnifiedMultimodalEmbedder(nn.Module):
    """Weightless RMSNorm followed by projection into the text hidden size."""

    def __init__(self,
                 multimodal_config: dict,
                 text_config: dict,
                 model_config: "ModelConfig",
                 module_name: str,
                 force_fp32_projection: bool = False,
                 multimodal_hidden_size: int | None = None) -> None:
        super().__init__()
        # Callers whose sub-config may have lost ``output_proj_dims`` resolve
        # it themselves and pass it in; see ``Gemma4UnifiedAudioModel``.
        if multimodal_hidden_size is None:
            multimodal_hidden_size = int(multimodal_config["output_proj_dims"])
        text_hidden_size = int(text_config["hidden_size"])
        eps = float(multimodal_config.get("rms_norm_eps", 1e-6))
        self.embedding_pre_projection_norm = Gemma4UnifiedRMSNorm(
            multimodal_hidden_size, eps=eps)
        if force_fp32_projection:
            self.embedding_projection = nn.Linear(multimodal_hidden_size,
                                                  text_hidden_size,
                                                  bias=False)
        else:
            self.embedding_projection = make_linear(
                model_config,
                multimodal_hidden_size,
                text_hidden_size,
                bias=False,
                module_name=module_name,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding_pre_projection_norm(hidden_states)
        return self.embedding_projection(
            hidden_states.to(self.embedding_projection.weight.dtype))


class Gemma4UnifiedVisionEmbedder(nn.Module):
    """Project merged RGB patches and add factorized 2-D position embeddings."""

    def __init__(self, vision_config: dict, model_config: "ModelConfig",
                 patch_dim: int) -> None:
        super().__init__()
        mm_embed_dim = int(vision_config["mm_embed_dim"])
        output_proj_dims = int(vision_config["output_proj_dims"])
        if mm_embed_dim != output_proj_dims:
            raise ValueError(
                "Gemma4 Unified vision requires mm_embed_dim to equal "
                f"output_proj_dims, got {mm_embed_dim} and {output_proj_dims}."
            )

        self.patch_ln1 = nn.LayerNorm(patch_dim)
        self.patch_dense = nn.Linear(patch_dim, mm_embed_dim, bias=True)
        self.patch_ln2 = nn.LayerNorm(mm_embed_dim)
        self.pos_embedding = nn.Parameter(torch.empty(
            int(vision_config["mm_posemb_size"]), 2, mm_embed_dim),
                                          requires_grad=False)
        self.pos_norm = nn.LayerNorm(mm_embed_dim)

    def forward(self, pixel_values: torch.Tensor,
                pixel_position_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_ln1(
            pixel_values.to(self.patch_dense.weight.dtype))
        hidden_states = self.patch_dense(hidden_states)
        hidden_states = self.patch_ln2(hidden_states)

        clamped_positions = pixel_position_ids.clamp(min=0).long()
        valid = (pixel_position_ids
                 >= 0).to(self.pos_embedding.dtype).unsqueeze(-1)
        axes = torch.arange(2, device=pixel_position_ids.device)
        position_embeddings = (self.pos_embedding[clamped_positions, axes] *
                               valid).sum(-2)
        hidden_states = hidden_states + position_embeddings
        return self.pos_norm(hidden_states)


class Gemma4UnifiedVisualModel(nn.Module):
    """Packed-patch Gemma 4 Unified visual graph used for ONNX export."""

    def __init__(self, config: dict, model_config: "ModelConfig",
                 patch_dim: int) -> None:
        super().__init__()
        vision_config = config.get("vision_config", config)
        text_config = config.get("text_config") or {}
        if "hidden_size" not in text_config:
            raise ValueError(
                "Gemma4 Unified visual export requires text_config.hidden_size"
            )
        self.vision_embedder = Gemma4UnifiedVisionEmbedder(
            vision_config, model_config, patch_dim)
        self.embed_vision = Gemma4UnifiedMultimodalEmbedder(
            vision_config,
            text_config,
            model_config,
            module_name="embed_vision.embedding_projection",
            force_fp32_projection=True,
        )

    def forward(self, pixel_values: torch.Tensor,
                pixel_position_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.vision_embedder(pixel_values, pixel_position_ids)
        # Keep the runtime interface FP16 even though the Unified checkpoint's
        # unusually large visual projection weights require FP32 internally.
        return self.embed_vision(hidden_states).to(torch.float16)

    def get_onnx_export_args(self, config: dict, device: str):
        """Return packed-patch I/O and dynamic-shape metadata.

        ONNX bindings:
          - ``input``: ``[num_patches, model_patch_size**2 * 3]`` FP16
          - ``pixel_position_ids``: ``[num_patches, 2]`` INT64
          - ``output``: ``[num_patches, text_hidden_size]`` FP16
        """
        vision_config = config.get("vision_config", config)
        patch_dim = self.vision_embedder.patch_dense.in_features
        num_patches = min(4, int(vision_config.get("num_soft_tokens", 4)))
        pixel_values = torch.zeros(num_patches,
                                   patch_dim,
                                   dtype=torch.float16,
                                   device=device)
        pixel_position_ids = torch.zeros(num_patches,
                                         2,
                                         dtype=torch.int64,
                                         device=device)
        num_patches_dim = torch.export.Dim("num_patches", min=1)
        args = (pixel_values, pixel_position_ids)
        input_names = ["input", "pixel_position_ids"]
        output_names = ["output"]
        dynamic_shapes = {
            "pixel_values": {
                0: num_patches_dim
            },
            "pixel_position_ids": {
                0: num_patches_dim
            },
        }
        return args, input_names, output_names, dynamic_shapes


#: Every quantized checkpoint nests the patch embedder under ``embed_vision``
#: and the projection under ``multimodal_embedder``, where the released layout
#: keeps them flat. First match wins, so the ``multimodal_embedder`` entry must
#: precede the bare ``embed_vision`` ones; released keys match none.
_MODULE_TREE_PREFIX_REWRITES = (
    ("embed_vision.multimodal_embedder.", "embed_vision."),
    ("embed_vision.patch_", "vision_embedder.patch_"),
    ("embed_vision.pos_", "vision_embedder.pos_"),
)


def _load_weights(model: nn.Module, weights: dict) -> None:
    """Strictly load Unified vision/embedder tensors from the root checkpoint.

    Accepts the released and ``save_pretrained`` layouts, see
    ``_MODULE_TREE_PREFIX_REWRITES``.
    """
    from ...checkpoint.loader import load_submodule_weights

    def _remap(key: str) -> "str | None":
        candidate = key[len("model."):] if key.startswith("model.") else key
        for prefix, replacement in _MODULE_TREE_PREFIX_REWRITES:
            if candidate.startswith(prefix):
                candidate = replacement + candidate[len(prefix):]
                break
        if (candidate.startswith("vision_embedder.")
                or candidate.startswith("embed_vision.")):
            return candidate
        return None

    mapped_keys = {
        mapped
        for key in weights if (mapped := _remap(key)) is not None
    }
    expected_keys = set(model.state_dict())
    missing = sorted(expected_keys - mapped_keys)
    unexpected = sorted(mapped_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "Gemma4 Unified visual checkpoint schema mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}")

    load_submodule_weights(
        model,
        weights,
        _remap,
        transform=lambda _key, tensor: tensor.to(torch.float32),
        label="gemma4 unified visual",
        log=logger)


def build_gemma4_unified_visual(
        config: dict,
        weights: dict,
        model_config: "ModelConfig",
        dtype: torch.dtype = torch.float16) -> Gemma4UnifiedVisualModel:
    """Instantiate and load the FP32-internal Unified visual graph.

    ``dtype`` remains part of the common encoder factory API, but this model's
    runtime input and output are always FP16 and its internal math is FP32.
    """
    vision_config = config.get("vision_config", config)
    if "model_patch_size" in vision_config:
        ps = int(vision_config["model_patch_size"])
        patch_dim = ps * ps * 3
    else:
        # Infer patch_dim from the checkpoint weight rather than hard-coding an
        # architectural constant.  patch_dense maps [num_patches, patch_dim] →
        # [num_patches, mm_embed_dim], so weight shape is [mm_embed_dim, patch_dim].
        # Both layouts, for the same reason ``_MODULE_TREE_PREFIX_REWRITES``
        # exists: a quantized checkpoint nests the patch embedder under
        # ``embed_vision``, and loses ``model_patch_size`` to the same
        # round-trip that sends it down this branch.
        _keys = ("model.vision_embedder.patch_dense.weight",
                 "model.embed_vision.patch_dense.weight")
        _key = next((k for k in _keys if k in weights), None)
        if _key is None:
            raise KeyError(
                f"Cannot infer patch_dim: none of {_keys} is present in "
                "weights and 'model_patch_size' is not in vision_config.")
        patch_dim = int(weights[_key].shape[1])
    model = Gemma4UnifiedVisualModel(config,
                                     model_config=model_config,
                                     patch_dim=patch_dim)
    # The checkpoint is trained in BF16 and its visual patch projection has a
    # much larger dynamic range than ordinary transformer weights.  Converting
    # it to FP16 can overflow patch_dense on non-uniform images; the following
    # LayerNorm then turns that inf into an all-NaN visual embedding.  Keep the
    # complete visual path in FP32 and cast only the graph output back to FP16.
    # This retains the existing FP16 TensorRT bindings and does not affect the
    # separate audio model, which continues to use the requested dtype.
    model.to(torch.float32)
    _load_weights(model, weights)
    model.eval()
    return model


__all__ = [
    "Gemma4UnifiedMultimodalEmbedder",
    "Gemma4UnifiedRMSNorm",
    "Gemma4UnifiedVisionEmbedder",
    "Gemma4UnifiedVisualModel",
    "build_gemma4_unified_visual",
]
