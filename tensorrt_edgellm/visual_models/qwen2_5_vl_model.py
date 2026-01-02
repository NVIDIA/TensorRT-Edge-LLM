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
Qwen2.5-VL visual model wrapper and export functionality.

This module provides wrapper classes and export functions for Qwen2.5-VL visual models,
enabling ONNX export with proper attention mechanism handling and FP16 overflow fixes.

TODO: Input/output names have been aligned with the old multimodal_export.py for compatibility.
      Future refactoring should consider more descriptive names while maintaining backward compatibility.
"""

import math
from typing import Any

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
from modelopt.torch.quantization.nn import TensorQuantizer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLMLP,
    Qwen2_5_VLPatchMerger, Qwen2_5_VLVisionAttention, Qwen2_5_VLVisionBlock,
    apply_rotary_pos_emb_vision)

from ..onnx_export.onnx_utils import export_onnx


class Qwen2_5_VLVisionAttentionPatch(Qwen2_5_VLVisionAttention):
    """
    Patched version of Qwen2.5-VL vision attention for ONNX export.
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


class QuantQwen2_5_VLVisionAttention(Qwen2_5_VLVisionAttentionPatch):
    """
    Quantized MHA version of Qwen2_5_VLVisionAttentionPatch.
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


mtq.register(original_cls=Qwen2_5_VLVisionAttentionPatch,
             quantized_cls=QuantQwen2_5_VLVisionAttention)


class Qwen2_5_VLMLPPatch(Qwen2_5_VLMLP):
    """
    Patched version of Qwen2_5_VLMLP to cast Down Proj to FP32 to avoid FP16 overflow.
    
    This class addresses numerical stability issues in FP16 by casting the down projection
    layer to FP32 during computation.
    """

    def __init__(self, config: Any, bias: bool = False) -> None:
        """
        Initialize the patched MLP.
        
        Args:
            config: Model configuration object
            bias: Whether to use bias in the linear layers
        """
        super().__init__(config, bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FP32 casting for numerical stability.
        
        Args:
            hidden_state: Input hidden states
        
        Returns:
            torch.Tensor: Output after MLP processing
        """
        # Apply gate and up projections
        hidden_state = self.act_fn(
            self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        # Cast to FP32 for numerical stability
        hidden_state = hidden_state.to(torch.float32)
        # Cast down projection weights and bias to FP32
        self.down_proj.weight.data = self.down_proj.weight.data.to(
            torch.float32)
        self.down_proj.bias.data = self.down_proj.bias.data.to(torch.float32)
        return self.down_proj(hidden_state)


class Qwen2_5_VLVisionBlockPatchWAR(Qwen2_5_VLVisionBlock):
    """
    Workaround patch for Qwen2.5-VL 3B FP16 overflow issues.
    
    This class provides a workaround for FP16 overflow issues that occur specifically
    in the Qwen2.5-VL 3B model by using FP32 casting in critical operations.
    """

    def __init__(self,
                 config: Any,
                 attn_implementation: str = "eager") -> None:
        """
        Initialize the workaround vision block.
        
        Args:
            config: Model configuration object
            attn_implementation: Attention implementation type
        """
        super().__init__(config)
        self.attn = Qwen2_5_VLVisionAttentionPatch(config)
        self.mlp = Qwen2_5_VLMLPPatch(config, bias=True)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FP32 casting for overflow prevention.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            position_embeddings: Position embeddings
        
        Returns:
            torch.Tensor: Output hidden states
        """
        # Apply attention with residual connection
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings)
        # Apply MLP with FP32 casting and residual connection
        hidden_states = hidden_states.to(torch.float32) + self.mlp(
            self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VLVisionBlockPatch(Qwen2_5_VLVisionBlock):
    """
    Standard patched version of Qwen2_5_VLVisionBlock with custom attention mechanism.
    
    This class replaces the original attention mechanism with a custom implementation
    that is compatible with ONNX export.
    """

    def __init__(self,
                 config: Any,
                 attn_implementation: str = "eager") -> None:
        """
        Initialize the patched vision block.
        
        Args:
            config: Model configuration object
            attn_implementation: Attention implementation type
        """
        super().__init__(config)
        self.attn = Qwen2_5_VLVisionAttentionPatch(config)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision block.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask for the attention mechanism
            position_embeddings: Position embeddings for rotary attention
        
        Returns:
            torch.Tensor: Output hidden states after processing
        """
        # Apply attention with residual connection
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings)
        # Apply MLP with residual connection
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VLPatchMergerWAR(Qwen2_5_VLPatchMerger):
    "WAR for Qwen2.5-VL 3B FP16 overflow"

    def __init__(self,
                 dim: int,
                 context_dim: int,
                 spatial_merge_size: int = 2) -> None:
        super().__init__(dim, context_dim, spatial_merge_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).to(torch.float16).view(-1, self.hidden_size))
        return x


class Qwen2_5_VisionTransformerPretrainedModelPatch(
        Qwen2_5_VisionTransformerPretrainedModel):
    """
    Patched version of Qwen2.5_VisionTransformerPretrainedModel for ONNX export.
    
    This class provides a wrapper around the original Qwen2.5-VL vision transformer
    with custom blocks that are compatible with ONNX export and includes workarounds
    for FP16 overflow issues in the 3B model.
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
            Qwen2_5_VLVisionBlockPatch(config, config._attn_implementation)
            for _ in range(config.depth)
        ])

        # Qwen2.5-VL 3B VIT has overflow issue with FP16 and only happens in /blocks.31/mlp/down_proj
        # Apply Patch to cast /blocks.31/mlp/down_proj to FP32 to avoid this issue.
        if config.out_hidden_size == 2048:
            self.blocks[-1] = Qwen2_5_VLVisionBlockPatchWAR(
                config, config._attn_implementation)
            self.merger = Qwen2_5_VLPatchMergerWAR(
                dim=config.out_hidden_size,
                context_dim=config.hidden_size,
                spatial_merge_size=config.spatial_merge_size,
            )

    def forward(self, hidden_states: torch.Tensor,
                rotary_pos_emb: torch.Tensor, attention_mask: torch.Tensor,
                window_attention_mask: torch.Tensor,
                window_index: torch.Tensor,
                reverse_window_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision transformer with window attention support.
        
        Args:
            hidden_states: Input hidden states
            rotary_pos_emb: Rotary position embeddings
            attention_mask: Attention mask
            window_attention_mask: Window attention mask
            window_index: Window index for attention
            reverse_window_index: Reverse window index for attention
        
        Returns:
            torch.Tensor: Output embeddings after processing through all blocks
        """
        hidden_states = self.patch_embed(hidden_states)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                attention_mask_now = attention_mask
            else:
                attention_mask_now = window_attention_mask
            hidden_states = blk(hidden_states,
                                attention_mask=attention_mask_now,
                                position_embeddings=position_embeddings)

        hidden_states = self.merger(hidden_states)
        hidden_states = hidden_states[reverse_window_index, :]

        return hidden_states


def export_qwen2_5_vl_visual(
    model: Qwen2_5_VisionTransformerPretrainedModelPatch,
    output_dir: str,
    torch_dtype: torch.dtype,
) -> None:
    """
    Export Qwen2.5-VL visual model to ONNX format.
    
    This function takes a patched Qwen2.5-VL visual model, prepares dummy inputs 
    for ONNX export, and saves the model in ONNX format.
    
    Args:
        model: Patched Qwen2.5-VL vision transformer model
        output_dir: Directory to save the exported ONNX model
        torch_dtype: PyTorch data type for the model
    """

    # Prepare dummy input sizes (will be replaced by dynamic axes)
    grid_t = 1
    grid_h = 8
    grid_w = 16
    hw = grid_t * grid_h * grid_w
    in_chans = model.config.in_chans
    temporal_patch_size = model.config.temporal_patch_size
    patch_size = model.config.patch_size
    rotary_pos_emb_dim = model.config.hidden_size // model.config.num_heads // 2

    # Create input tensors with appropriate shapes and dtypes
    input_tensor = torch.randn(
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
    window_attention_mask = torch.randn((1, hw, hw),
                                        dtype=torch_dtype,
                                        device=model.device)

    # Create window index tensors
    window_index = torch.arange(hw // 4,
                                dtype=torch.int64,
                                device=model.device)
    window_index = window_index.reshape(grid_t, grid_h // 8, 4, grid_w // 8, 4)
    window_index = window_index.permute(0, 1, 3, 2, 4).reshape(-1)
    # TensorRT TopK max K = 3840. Compute reverse index outside to support longer image tokens.
    reverse_window_index = torch.argsort(window_index)

    # Define dynamic axes for variable input sizes
    dynamic_axes = {
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
        'window_attention_mask': {
            1: 'hw',
            2: 'hw'
        },
        'window_index': {
            0: 'hw//4'
        },
        'reverse_window_index': {
            0: 'hw//4'
        },
        "output": {
            0: 'image_token_len'
        },
    }

    inputs = (input_tensor, rotary_pos_emb, attention_mask,
              window_attention_mask, window_index, reverse_window_index)
    input_names = [
        "input", "rotary_pos_emb", "attention_mask", "window_attention_mask",
        "window_index", "reverse_window_index"
    ]
    output_names = ["output"]

    export_onnx(model, inputs, output_dir, input_names, output_names,
                dynamic_axes)
