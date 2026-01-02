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
Qwen3-VL visual model wrapper and export functionality.

This module provides wrapper classes and export functions for Qwen3-VL visual models,
enabling ONNX export with proper attention mechanism handling.

TODO: Input/output names have been aligned with the old multimodal_export.py for compatibility.
      Future refactoring should consider more descriptive names while maintaining backward compatibility.
"""

import math
from typing import Any, List, Tuple

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
from modelopt.torch.quantization.nn import TensorQuantizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionAttention, Qwen3VLVisionBlock, Qwen3VLVisionModel,
    apply_rotary_pos_emb_vision)

from ..onnx_export.onnx_utils import export_onnx


class Qwen3VLVisionAttentionPatch(Qwen3VLVisionAttention):
    """
    Patched version of Qwen3-VL vision attention for ONNX export.
    Though Qwen3VLVisionAttention uses `cu_seqlens` to isolate attention between different images,
    we still use `attention_mask` to be compatible with TensorRT.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with custom attention implementation.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            position_embeddings: Position embeddings for rotary attention
            
        Returns:
            Attention output
        """
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3,
                                                  self.num_heads,
                                                  -1).permute(1, 0, 2,
                                                              3).unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(
            self.head_dim)
        attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights,
                                                   dim=-1,
                                                   dtype=torch.float32).to(
                                                       v.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class QuantQwen3VLVisionAttentionPatch(Qwen3VLVisionAttentionPatch):
    """
    Quantized MHA version of Qwen3VLVisionAttentionPatch.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._setup()

    def _setup(self) -> None:
        """Initialize quantization components."""
        self.q_bmm_quantizer = TensorQuantizer()
        self.k_bmm_quantizer = TensorQuantizer()
        self.v_bmm_quantizer = TensorQuantizer()
        self.softmax_quantizer = TensorQuantizer()

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantized forward pass with custom attention implementation.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            position_embeddings: Position embeddings for rotary attention
            
        Returns:
            Attention output
        """
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3,
                                                  self.num_heads,
                                                  -1).permute(1, 0, 2,
                                                              3).unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        q = self.q_bmm_quantizer(q)
        k = self.k_bmm_quantizer(k)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(
            self.head_dim)
        attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights,
                                                   dim=-1,
                                                   dtype=torch.float32).to(
                                                       v.dtype)
        attn_weights = self.softmax_quantizer(attn_weights)
        v = self.v_bmm_quantizer(v)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


mtq.register(original_cls=Qwen3VLVisionAttentionPatch,
             quantized_cls=QuantQwen3VLVisionAttentionPatch)


class Qwen3VLVisionBlockPatch(Qwen3VLVisionBlock):
    """
    Patched version of Qwen3VLVisionBlock with custom attention mechanism.
    
    This class replaces the original attention mechanism with a custom implementation
    that is compatible with ONNX export.
    """

    def __init__(self,
                 config: Any,
                 attn_implementation: str = "eager") -> None:
        super().__init__(config)
        self.attn = Qwen3VLVisionAttentionPatch(config)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision block.
        
        Args:
            hidden_states: Input hidden states [seq_len, hidden_size]
            attention_mask: Attention mask [1, seq_len, seq_len]
            position_embeddings: Position embeddings for rotary attention [seq_len, rotary_pos_emb_dim]
        
        Returns:
            torch.Tensor: Output hidden states after processing [seq_len, hidden_size]
        """
        # Apply attention with residual connection
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings)
        # Apply MLP with residual connection
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionModelPatch(Qwen3VLVisionModel):
    """
    Patched version of Qwen3VLVisionModel for ONNX export.
    
    This class provides a wrapper around the original Qwen3-VL vision transformer
    with custom blocks that are compatible with ONNX export.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize the patched vision transformer.
        
        Args:
            config: Model configuration object
        """
        super().__init__(config)
        # Replace all blocks with patched versions
        self.blocks = nn.ModuleList([
            Qwen3VLVisionBlockPatch(config, config._attn_implementation)
            for _ in range(config.depth)
        ])

    def fast_pos_embed_interpolate_optimized(self, grid_thw):
        """
        Optimized version of `fast_pos_embed_interpolate` in Qwen3VLVisionModel.
        The original `fast_pos_embed_interpolate` workflow permutes after embedding,
            which is inefficient and hard to implement in TensorRT.
        We permute the index and weight tensor first during initialization and take them as model inputs.
        """
        grid_ts, grid_hs, grid_ws = \
            grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side -
                                                  1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side -
                                                  1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            # Permute indices and weights first so no need to permute after embedding.
            # From [h, w] to [h // merge_size, w // merge_size, merge_size, merge_size]
            merge_size = self.config.spatial_merge_size
            merged_h, merged_w = h // merge_size, w // merge_size

            indices = [
                (base_h.reshape(merged_h, 1, merge_size, 1) +
                 w_idxs_floor.reshape(1, merged_w, 1, merge_size)).flatten(),
                (base_h.reshape(merged_h, 1, merge_size, 1) +
                 w_idxs_ceil.reshape(1, merged_w, 1, merge_size)).flatten(),
                (base_h_ceil.reshape(merged_h, 1, merge_size, 1) +
                 w_idxs_floor.reshape(1, merged_w, 1, merge_size)).flatten(),
                (base_h_ceil.reshape(merged_h, 1, merge_size, 1) +
                 w_idxs_ceil.reshape(1, merged_w, 1, merge_size)).flatten(),
            ]

            weights = [
                ((1 - dh).reshape(merged_h, 1, merge_size, 1) *
                 (1 - dw).reshape(1, merged_w, 1, merge_size)).flatten(),
                ((1 - dh).reshape(merged_h, 1, merge_size, 1) *
                 dw.reshape(1, merged_w, 1, merge_size)).flatten(),
                (dh.reshape(merged_h, 1, merge_size, 1) *
                 (1 - dw).reshape(1, merged_w, 1, merge_size)).flatten(),
                (dh.reshape(merged_h, 1, merge_size, 1) *
                 dw.reshape(1, merged_w, 1, merge_size)).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list,
                                  dtype=torch.long,
                                  device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(weight_list,
                                     dtype=self.pos_embed.weight.dtype,
                                     device=self.pos_embed.weight.device)
        return idx_tensor, weight_tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_mask: torch.Tensor,
        fast_pos_embed_idx: torch.Tensor,
        fast_pos_embed_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the vision transformer.
        
        Args:
            hidden_states: Input hidden states [seq_len, input_dim]
            rotary_pos_emb: Rotary position embeddings [seq_len, rotary_pos_emb_dim]
            attention_mask: Attention mask [1, seq_len, seq_len]
            fast_pos_embed_idx: Fast position index tensor [4, seq_len]
            fast_pos_embed_weight: Fast position weight tensor [4, seq_len]
        
        Returns:
            `torch.Tensor`: hidden_states.
            list of `torch.Tensor`: deepstack_feature_lists.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.pos_embed(
            fast_pos_embed_idx) * fast_pos_embed_weight[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[
            2] + pos_embeds[3]
        # No need to permute after embedding.
        hidden_states = hidden_states + patch_pos_embeds

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)](
                        hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists


def export_qwen3_vl_visual(
    model: Qwen3VLVisionModelPatch,
    output_dir: str,
    torch_dtype: torch.dtype,
) -> None:
    """
    Export Qwen3-VL visual model to ONNX format.
    
    This function takes a patched Qwen3-VL visual model, prepares dummy inputs 
    for ONNX export, and saves the model in ONNX format.
    
    Args:
        model: Patched Qwen3-VL vision transformer model
        output_dir: Directory to save the exported ONNX model
        torch_dtype: PyTorch data type for the model
    """

    # Prepare dummy inputs for ONNX export
    hw = 16  # Height * width for the input
    in_chans = model.config.in_channels
    temporal_patch_size = model.config.temporal_patch_size
    patch_size = model.config.patch_size
    rotary_pos_emb_dim = model.config.hidden_size // model.config.num_heads // 2

    # Create input tensors with appropriate shapes and dtypes
    pixel_values = torch.randn(
        (hw, in_chans * temporal_patch_size * patch_size * patch_size),
        dtype=torch_dtype,
        device=model.device)
    rotary_pos_emb = torch.randn(
        (hw, rotary_pos_emb_dim),
        dtype=torch.float32,  # Keep as float32 for rotary embeddings
        device=model.device)
    attention_mask = torch.randn((1, hw, hw),
                                 dtype=torch_dtype,
                                 device=model.device)
    fast_pos_embed_idx = torch.arange(hw,
                                      dtype=torch.int64,
                                      device=model.device).unsqueeze(0).repeat(
                                          4, 1)
    fast_pos_embed_weight = torch.randn((4, hw),
                                        dtype=torch_dtype,
                                        device=model.device)

    inputs = (pixel_values, rotary_pos_emb, attention_mask, fast_pos_embed_idx,
              fast_pos_embed_weight)

    input_names = [
        "input", "rotary_pos_emb", "attention_mask", "fast_pos_embed_idx",
        "fast_pos_embed_weight"
    ]
    # In Qwen3-VL 2B, 4B and 8B, there are 3 deepstack features
    output_names = [
        "output", "deepstack_features.0", "deepstack_features.1",
        "deepstack_features.2"
    ]

    # Define dynamic axes for variable input sizes
    dynamic_axes = {
        # Model inputs
        'input': {
            0: 'hw'
        },
        'rotary_pos_emb': {
            0: 'hw'
        },
        'attention_mask': {
            1: 'hw',
            2: 'hw'
        },
        'fast_pos_embed_idx': {
            1: 'hw'
        },
        'fast_pos_embed_weight': {
            1: 'hw'
        },
        # Model outputs
        'output': {
            0: 'image_token_len'
        },
        'deepstack_features.0': {
            0: 'image_token_len'
        },
        'deepstack_features.1': {
            0: 'image_token_len'
        },
        'deepstack_features.2': {
            0: 'image_token_len'
        },
    }

    export_onnx(model, inputs, output_dir, input_names, output_names,
                dynamic_axes)
