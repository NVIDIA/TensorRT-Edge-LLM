# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Phi-4 multimodal visual model wrapper and export functionality.

This module provides wrapper classes and export functions for Phi-4 visual encoder
to enable ONNX export that is compatible with TensorRT Edge-LLM runtime bindings.

The wrapper aligns I/O names with the expected TensorRT engine bindings:
  - input:          visual input tensor
  - image_attention_mask: per-crop attention mask tensor
  - output:         visual output embeddings (image_embeds)
"""

import types
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file

from ..onnx_export.onnx_utils import export_onnx


class Phi4MMVisionModel(nn.Module):
    """Wrapper for Phi-4 visual encoder.

    Expects the HF model to provide an `image_embed` module at
    `model.embed_tokens_extend.image_embed` that exposes `get_img_features`,
    projection layers and GN buffers similar to DriveOS_LLM_SDK implementation.
    """

    def __init__(self, hf_model: Any) -> None:
        super().__init__()
        # Store handles used during forward
        self.device = hf_model.device
        self.dtype = hf_model.dtype
        # The actual visual module
        self.vision_model = hf_model.model.embed_tokens_extend.image_embed

        # replace the forward function of the embeddings module
        emb = self.vision_model.img_processor.embeddings

        def custom_forward(self, pixel_values, patch_attention_mask):
            batch_size = pixel_values.size(0)
            patch_embeds = self.patch_embedding(pixel_values)
            embeddings = patch_embeds.flatten(2).transpose(1, 2)
            N = self.num_patches_per_side
            position_ids = torch.arange(
                N * N,
                device=self.position_embedding.weight.device,
                dtype=torch.long).repeat(batch_size, 1)
            return embeddings + self.position_embedding(position_ids)

        emb.forward = types.MethodType(custom_forward, emb)

        self.glb_GN = hf_model.model.embed_tokens_extend.image_embed.glb_GN
        self.sub_GN = hf_model.model.embed_tokens_extend.image_embed.sub_GN

    @torch.no_grad()
    def forward(
            self,
            pixel_values: torch.Tensor,  # (total_num_blocks, C, H, W)
    ) -> torch.Tensor:
        """Compute Phi-4 image token embeddings for 448x448 inputs.

        This implementation mirrors the high-definition transform path used in
        the original Phi-4 model but assumes a single high-resolution crop that
        is identical to the global crop (input resolution 448x448).

        Args:
            pixel_values: Tensor of shape (total_num_blocks, 3, 448, 448). The first crop is
                the global view, and the second crop is the identical HD tile.

        Returns:
            Tensor of shape (total_num_blocks * tokens_per_block, hidden_size).
        """

        # [total_num_blocks, 3, 448, 448] -> [total_num_blocks, 256, 1152]
        img_features = self.vision_model.get_img_features(pixel_values)

        # [total_num_blocks, 256, 1152] -> [total_num_blocks, 256, 3072]
        projected_tokens = self.vision_model.img_projection(img_features)

        return projected_tokens.reshape(-1, projected_tokens.shape[-1])


def export_phi4mm_visual(model: Phi4MMVisionModel, output_dir: str,
                         torch_dtype: torch.dtype) -> None:
    """Export Phi-4 visual model to ONNX.

    I/O mapping:
      - input: (num_blocks, 3, H, W)
      - image_attention_mask: (num_blocks, Mh, Mw)
      - output: (num_blocks, hidden_size)
    """

    # dummy input
    total_num_blocks = 7
    C = 3
    H = W = getattr(model.vision_model, "crop_size", 448)

    img_embeds = torch.randn((total_num_blocks, C, H, W),
                             dtype=torch_dtype,
                             device=model.device)

    inputs = (img_embeds, )
    input_names = ["input"]
    output_names = ["output"]

    dynamic_axes = {
        "input": {
            0: "num_blocks",
        },
    }

    export_onnx(model, inputs, output_dir, input_names, output_names,
                dynamic_axes)

    # Save GN projection weights to safetensors file, used by runtime during preprocessing
    vision_model = model.vision_model
    glb = vision_model.glb_GN.reshape(1, -1)
    sub = vision_model.sub_GN.reshape(1, -1)
    glb_proj = vision_model.img_projection(glb).squeeze(0).to(
        torch.float16).cpu().contiguous()
    sub_proj = vision_model.img_projection(sub).squeeze(0).to(
        torch.float16).cpu().contiguous()
    save_file({
        "glb_GN": glb_proj,
        "sub_GN": sub_proj
    }, f"{output_dir}/phi4mm_gn_proj.safetensors")
