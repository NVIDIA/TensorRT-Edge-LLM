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
From-scratch Phi-4 Multimodal vision encoder implementation.

Architecture (matches checkpoint from transformers ≤4.46.1 key names):
    Phi4MMVisionModel  (CLIP-style ViT)
        Phi4MMVisionEmbeddings  (patch_embedding Conv2d + position_embedding)
        Phi4MMVisionEncoder     (27 × Phi4MMVisionEncoderLayer)
            layer_norm1 / self_attn (q/k/v_proj, out_proj) / layer_norm2 / mlp (fc1, fc2)
        post_layernorm          (LayerNorm, not applied to the extracted feature)
    image_token_compression     (AvgPool2d kernel=2 stride=2)
    img_projection              (Sequential: Linear(1152→3072), GELU, Linear(3072→3072))

Checkpoint weight key prefix:
    ``model.embed_tokens_extend.image_embed.*``

Sidecar tensors (separate safetensors file):
    ``glb_GN`` [1, 1, 1152] — global image separator token
    ``sub_GN`` [1, 1, 1, 1152] — sub-image separator token

Forward I/O (ONNX):
    Input:  pixel_values  [N, 3, 448, 448]  float16
    Output: image_embeds  [N * 256, 3072]   float16

The ``feature_layer = -2`` convention means the second-to-last layer hidden
state (layer index 25 out of 27) is extracted before ``post_layernorm``.
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import config as config_module
from ..linear import make_linear
from ..ops import init_fp8_mha, quantize_qkv_for_fp8_mha, vit_attention_plugin

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...config import ModelConfig

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class Phi4MMVisionEmbeddings(nn.Module):
    """Patch + positional embeddings.

    Checkpoint keys (under prefix):
        ``img_processor.embeddings.patch_embedding.*``
        ``img_processor.embeddings.position_embedding.*``
    """

    def __init__(self, hidden_size: int, num_patches: int, num_channels: int,
                 patch_size: int) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,  # "valid"
            bias=True,
        )
        self.position_embedding = nn.Embedding(num_patches, hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [N, 3, 448, 448]
        x = self.patch_embedding(pixel_values)  # [N, H, h, w]
        N = x.shape[0]
        x = x.flatten(2).transpose(1, 2)  # [N, num_patches, H]
        # Fixed position ids (no interpolation needed for 448×448 → 32×32)
        pos_ids = torch.arange(self.num_patches,
                               device=x.device,
                               dtype=torch.long).unsqueeze(0).expand(N, -1)
        x = x + self.position_embedding(pos_ids)
        return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Phi4MMVisionAttention(nn.Module):
    """Standard multi-head self-attention (bidirectional, no RoPE).

    Checkpoint keys (under prefix + ``img_processor.encoder.layers.N.self_attn``):
        q_proj.weight / q_proj.bias
        k_proj.weight / k_proj.bias
        v_proj.weight / v_proj.bias
        out_proj.weight / out_proj.bias
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
        self.hidden_size = hidden_size
        # FP8 head-dim padding (TRT-LLM Gemma4-vision recipe): SigLIP's
        # d=72 rows are not 16-byte aligned for FP8 TMA, so when FP8 MHA is
        # requested the q/k/v projections are widened per-head to d=80 with
        # zero output rows and out_proj absorbs the extra columns with zero
        # weights (exact: QK dots are unchanged by zero columns, V zero
        # columns produce zero O columns that meet zero out_proj weights).
        # The softmax scale stays 1/sqrt(72). The zero rows/columns are
        # injected into the checkpoint tensors by ``_load_phi4mm_weights``.
        fp8_requested = getattr(model_config.quant, "visual_mha_quant",
                                None) == "fp8"
        self.kernel_head_dim = (80 if fp8_requested and self.head_dim == 72
                                else self.head_dim)
        qkv_out = num_heads * self.kernel_head_dim
        self.q_proj = make_linear(
            model_config,
            hidden_size,
            qkv_out,
            bias=True,
            module_name=f"{name_prefix}.q_proj" if name_prefix else "")
        self.k_proj = make_linear(
            model_config,
            hidden_size,
            qkv_out,
            bias=True,
            module_name=f"{name_prefix}.k_proj" if name_prefix else "")
        self.v_proj = make_linear(
            model_config,
            hidden_size,
            qkv_out,
            bias=True,
            module_name=f"{name_prefix}.v_proj" if name_prefix else "")
        self.out_proj = make_linear(
            model_config,
            qkv_out,
            hidden_size,
            bias=True,
            module_name=f"{name_prefix}.out_proj" if name_prefix else "")
        # Real q/k/v projections: scale buffers land on the existing
        # submodules (see init_fp8_mha docstring).
        self.enable_fp8_mha = init_fp8_mha(self,
                                           model_config,
                                           self.kernel_head_dim,
                                           k_proj=self.k_proj,
                                           v_proj=self.v_proj)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                max_seqlen_carrier: torch.Tensor) -> torch.Tensor:
        N, S, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(N * S, self.num_heads,
                                               self.kernel_head_dim)
        k = self.k_proj(hidden_states).reshape(N * S, self.num_heads,
                                               self.kernel_head_dim)
        v = self.v_proj(hidden_states).reshape(N * S, self.num_heads,
                                               self.kernel_head_dim)
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
            head_size=self.kernel_head_dim,
            attention_scale=self.attention_scale,
            qkv_scales=qkv_scales,
        )
        out = attn_output.reshape(N, S, self.num_heads * self.kernel_head_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class Phi4MMVisionMLP(nn.Module):
    """Two-layer FFN with GELU activation.

    Checkpoint keys (under prefix + ``img_processor.encoder.layers.N.mlp``):
        fc1.weight / fc1.bias
        fc2.weight / fc2.bias
    """

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


# ---------------------------------------------------------------------------
# Encoder Layer
# ---------------------------------------------------------------------------


class Phi4MMVisionEncoderLayer(nn.Module):
    """Single ViT encoder block (pre-norm).

    Checkpoint keys (under prefix + ``img_processor.encoder.layers.N``):
        layer_norm1.*, self_attn.*, layer_norm2.*, mlp.*
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 intermediate_size: int,
                 layer_norm_eps: float,
                 attention_scale: float,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn = Phi4MMVisionAttention(
            hidden_size,
            num_heads,
            attention_scale,
            model_config,
            name_prefix=f"{name_prefix}.self_attn" if name_prefix else "")
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = Phi4MMVisionMLP(
            hidden_size,
            intermediate_size,
            model_config,
            name_prefix=f"{name_prefix}.mlp" if name_prefix else "")

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                max_seqlen_carrier: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states), cu_seqlens, max_seqlen_carrier)
        hidden_states = hidden_states + self.mlp(
            self.layer_norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Phi4MMVisionEncoder(nn.Module):
    """Stack of Phi4MMVisionEncoderLayer blocks.

    Checkpoint keys: ``img_processor.encoder.layers.N.*``
    """

    def __init__(self,
                 num_layers: int,
                 hidden_size: int,
                 num_heads: int,
                 intermediate_size: int,
                 layer_norm_eps: float,
                 attention_scale: float,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Phi4MMVisionEncoderLayer(
                hidden_size,
                num_heads,
                intermediate_size,
                layer_norm_eps,
                attention_scale,
                model_config,
                name_prefix=f"{name_prefix}.layers.{i}" if name_prefix else "")
            for i in range(num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor,
                feature_layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (feature_hidden_states, last_hidden_states)."""
        N, S, _ = hidden_states.shape
        # Uniform-length ragged descriptors for the ViT attention plugin: every
        # crop contributes exactly S tokens, so cu_seqlens is a static-stride
        # ramp and the max-seqlen carrier is shape-only (its length S is what
        # the plugin reads; contents are never touched).
        cu_seqlens = torch.arange(0, (N + 1) * S,
                                  S,
                                  dtype=torch.int32,
                                  device=hidden_states.device)
        max_seqlen_carrier = torch.zeros(S,
                                         dtype=torch.int32,
                                         device=hidden_states.device)
        feature: torch.Tensor | None = None
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, cu_seqlens,
                                  max_seqlen_carrier)
            if i == feature_layer_idx:
                feature = hidden_states
        assert feature is not None
        return feature, hidden_states


# ---------------------------------------------------------------------------
# Vision Model (CLIP-style ViT)
# ---------------------------------------------------------------------------


class Phi4MMVisionModel(nn.Module):
    """Full Phi-4 CLIP-style ViT.

    Checkpoint keys: ``img_processor.*``
    """

    def __init__(self,
                 hidden_size: int,
                 num_layers: int,
                 num_heads: int,
                 intermediate_size: int,
                 num_patches: int,
                 num_channels: int,
                 patch_size: int,
                 layer_norm_eps: float,
                 feature_layer: int,
                 attention_scale: float,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.embeddings = Phi4MMVisionEmbeddings(hidden_size, num_patches,
                                                 num_channels, patch_size)
        self.encoder = Phi4MMVisionEncoder(
            num_layers,
            hidden_size,
            num_heads,
            intermediate_size,
            layer_norm_eps,
            attention_scale,
            model_config,
            name_prefix=f"{name_prefix}.encoder" if name_prefix else "")
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        # Absolute index of the feature layer inside the encoder
        # feature_layer = -2 ⟹ second-to-last out of num_layers outputs
        self._feature_idx = num_layers + feature_layer

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Returns the feature hidden state (before post_layernorm)."""
        x = self.embeddings(pixel_values)
        feature, _ = self.encoder(x, self._feature_idx)
        return feature


# ---------------------------------------------------------------------------
# Top-level visual model with projection
# ---------------------------------------------------------------------------


class Phi4MMVisualModel(nn.Module):
    """Complete Phi-4 Multimodal visual encoder (ViT + compression + projection).

    Forward I/O:
        Input:  pixel_values   [N, 3, 448, 448]  float16
        Output: image_embeds   [N * 256, 3072]    float16

    Checkpoint keys (all under ``model.embed_tokens_extend.image_embed.``):
        img_processor.*
        image_token_compression.*   (AvgPool2d — no learned weights)
        img_projection.0.*          (Linear 1152→3072)
        img_projection.2.*          (Linear 3072→3072)
        glb_GN                      [1, 1, 1152]
        sub_GN                      [1, 1, 1, 1152]
    """

    def __init__(self, config: dict, model_config: "ModelConfig") -> None:
        super().__init__()
        hidden_size: int = config["hidden_size"]
        num_layers: int = config["num_hidden_layers"]
        num_heads: int = config["num_attention_heads"]
        intermediate_size: int = config["intermediate_size"]
        image_size: int = config["image_size"]
        patch_size: int = config["patch_size"]
        num_channels: int = config.get("num_channels", 3)
        layer_norm_eps: float = config.get("layer_norm_eps", 1e-5)
        feature_layer: int = config["feature_layer"]
        head_dim = hidden_size // num_heads
        attention_scale = config_module._get_attention_scaling(
            config, head_dim, 1.0 / (float(head_dim)**0.5))

        # LLM projection hidden size — look for the LLM dim in parent config
        proj_hidden_size: int = config["proj_hidden_size"]

        num_patches_per_side = image_size // patch_size
        num_patches = num_patches_per_side**2
        self._num_img_tokens = (num_patches_per_side // 2)**2
        self._hidden_size = hidden_size
        self._proj_hidden_size = proj_hidden_size

        # Phi-4mm checkpoint stores image+audio under
        # ``model.embed_tokens_extend.image_embed.*``.  Use the leading-``model.``
        # stripped form for ``module_name`` so it matches what
        # ``layer_overrides`` / ``excluded`` carry — the parser strips ``model.``
        # from both via ``_normalize_module_name`` (see config.py).  Other VLM
        # families (Qwen3-VL ``visual.*``, InternVL3 ``vision_model.*``) follow
        # the same convention; Phi-4mm previously kept the full ``model.``
        # prefix and silently fell through to the dominant-quant fallback for
        # FP8 checkpoints, producing FP8Linear against FP16 weights.
        img_embed_prefix = "embed_tokens_extend.image_embed"
        self.img_processor = Phi4MMVisionModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_patches=num_patches,
            num_channels=num_channels,
            patch_size=patch_size,
            layer_norm_eps=layer_norm_eps,
            feature_layer=feature_layer,
            attention_scale=attention_scale,
            model_config=model_config,
            name_prefix=f"{img_embed_prefix}.img_processor",
        )
        self.image_token_compression = nn.AvgPool2d(kernel_size=2, stride=2)
        # img_projection.0 = Linear, img_projection.1 = GELU, img_projection.2 = Linear
        self.img_projection = nn.Sequential(
            make_linear(model_config,
                        hidden_size,
                        proj_hidden_size,
                        bias=True,
                        module_name=f"{img_embed_prefix}.img_projection.0"),
            nn.GELU(approximate="tanh"),
            make_linear(model_config,
                        proj_hidden_size,
                        proj_hidden_size,
                        bias=True,
                        module_name=f"{img_embed_prefix}.img_projection.2"),
        )
        # GN separator tokens (buffers — not registered as parameters so
        # load_state_dict can set them from the checkpoint)
        self.glb_GN = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.sub_GN = nn.Parameter(torch.zeros(1, 1, 1, hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract visual features and project to LLM hidden size.

        Args:
            pixel_values: [N, 3, image_size, image_size]

        Returns:
            [N * num_img_tokens, proj_hidden_size]
        """
        N = pixel_values.shape[0]

        # ViT feature extraction — output of feature_layer
        features = self.img_processor(
            pixel_values)  # [N, num_patches, hidden_size]

        # Reshape → NCHW → AvgPool2d → NHWC → flatten tokens
        S = features.shape[1]
        side = int(math.isqrt(S))
        features = features.view(N, side, side,
                                 self._hidden_size)  # [N, side, side, H]
        features = features.permute(0, 3, 1, 2)  # NCHW
        features = self.image_token_compression(features)
        features = features.permute(0, 2, 3, 1)  # NHWC
        features = features.view(N, self._num_img_tokens,
                                 self._hidden_size)  # [N, num_img_tokens, H]

        # Project to LLM hidden size
        out = self.img_projection(
            features)  # [N, num_img_tokens, proj_hidden_size]
        return out.view(N * self._num_img_tokens, self._proj_hidden_size)

    def get_onnx_export_args(self, config: dict, device: str):
        """Return (args, input_names, output_names, dynamic_shapes) for ONNX export."""
        image_size = config["image_size"]
        # Use batch=2 to keep the batch dimension symbolic during tracing.
        pixel_values = torch.zeros(2,
                                   3,
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

    def save_onnx_sidecar(self, output_dir: str) -> str:
        """Save Phi-4mm GN projection sidecar tensors.

        Args:
            output_dir: Directory where the sidecar ``.safetensors`` file is written.

        Returns:
            Path to the saved sidecar file.
        """
        import os

        from tensorrt_edgellm._safetensors_io import save_file
        glb = self.glb_GN.view(1, -1).to(torch.float16)
        sub = self.sub_GN.view(1, -1).to(torch.float16)
        with torch.no_grad():
            glb_proj = self.img_projection(glb).squeeze(0).to(
                torch.float16).cpu().contiguous()
            sub_proj = self.img_projection(sub).squeeze(0).to(
                torch.float16).cpu().contiguous()
        sidecar_path = os.path.join(output_dir, "phi4mm_gn_proj.safetensors")
        save_file({"glb_GN": glb_proj, "sub_GN": sub_proj}, sidecar_path)
        return sidecar_path


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _load_phi4mm_weights(
    model: Phi4MMVisualModel,
    weights: dict,
    prefix: str = "model.embed_tokens_extend.image_embed.",
) -> None:
    """Load Phi-4mm visual weights via the shared ``load_submodule_weights``
    pipeline so quantized ``make_linear`` classes get their weights set
    through ``_set_tensor`` (preserves dtype) and ``apply_all_repacking``
    runs at the end.
    """
    from ...checkpoint.loader import load_submodule_weights

    weights = _pad_attention_head_dim(model, weights, prefix)

    def _remap(k: str) -> "str | None":
        if not k.startswith(prefix):
            return None
        return k[len(prefix):]

    load_submodule_weights(model, weights, _remap, label="Phi4MMVisualModel")


def _pad_attention_head_dim(model: Phi4MMVisualModel, weights: dict,
                            prefix: str) -> dict:
    """Zero-pad attention weights from head_dim to kernel_head_dim.

    Active only when :class:`Phi4MMVisionAttention` widened its projections
    for FP8 MHA (SigLIP d=72 -> 80; see the class docstring). q/k/v_proj
    weights/biases gain zero output rows per head; out_proj.weight gains
    zero input columns per head. Zero rows/columns keep the attention
    numerically exact while making the per-head row stride 16B-aligned for
    the FP8 TMA path. Works for both fp16 and pre-quantized FP8 (e4m3 zero
    is exact; per-tensor scales are unaffected by zero padding).
    """
    attn = next(
        (m for m in model.modules() if isinstance(m, Phi4MMVisionAttention)),
        None)
    if attn is None or attn.kernel_head_dim == attn.head_dim:
        return weights
    heads, d, dp = attn.num_heads, attn.head_dim, attn.kernel_head_dim

    qkv_re = re.compile(r"self_attn\.[qkv]_proj\.(weight|bias)$")
    out_re = re.compile(r"self_attn\.out_proj\.weight$")

    padded = dict(weights)
    n_padded = 0
    for key, t in weights.items():
        if not key.startswith(prefix):
            continue
        rel = key[len(prefix):]
        if qkv_re.search(rel):
            if rel.endswith("weight"):  # [heads*d, in] -> [heads*dp, in]
                t3 = t.reshape(heads, d, t.shape[1])
                pad = torch.zeros((heads, dp - d, t.shape[1]), dtype=t.dtype)
                padded[key] = torch.cat([t3, pad],
                                        dim=1).reshape(heads * dp, t.shape[1])
            else:  # bias [heads*d] -> [heads*dp]
                t2 = t.reshape(heads, d)
                pad = torch.zeros((heads, dp - d), dtype=t.dtype)
                padded[key] = torch.cat([t2, pad], dim=1).reshape(heads * dp)
            n_padded += 1
        elif out_re.search(rel):  # [out, heads*d] -> [out, heads*dp]
            t3 = t.reshape(t.shape[0], heads, d)
            pad = torch.zeros((t.shape[0], heads, dp - d), dtype=t.dtype)
            padded[key] = torch.cat([t3, pad],
                                    dim=2).reshape(t.shape[0], heads * dp)
            n_padded += 1
    logger.info(
        "Phi4MM visual: zero-padded %d attention tensors head_dim %d -> %d "
        "for FP8 MHA", n_padded, d, dp)
    return padded


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_phi4mm_visual(
    config: dict,
    weights: dict,
    model_config: "ModelConfig",
    dtype: torch.dtype = torch.float16,
) -> Phi4MMVisualModel:
    """Build and return a :class:`Phi4MMVisualModel` with loaded weights.

    Args:
        config:       Model config dict (reads architecture values with sensible defaults).
        weights:      Flat ``{key: tensor}`` dict from safetensors.
        model_config: Top-level ``ModelConfig`` for quantized Linear dispatch.
        dtype:        Weight dtype (default ``float16``).
    """
    model = Phi4MMVisualModel(config,
                              model_config=model_config).to(dtype=dtype)
    _load_phi4mm_weights(model, weights)
    model.eval()
    return model


__all__ = ["Phi4MMVisualModel", "build_phi4mm_visual"]
