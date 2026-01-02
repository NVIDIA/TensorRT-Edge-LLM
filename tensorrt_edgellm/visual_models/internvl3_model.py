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
InternVL3 visual model wrapper and export functionality.

This module provides wrapper classes and export functions for InternVL3 visual models,
enabling ONNX export with proper vision tower and multi-modal projector handling.

TODO: Input/output names have been aligned with the old multimodal_export.py for compatibility.
      Future refactoring should consider more descriptive names while maintaining backward compatibility.
"""

from typing import Any, Optional, Tuple

import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.quantization.nn import TensorQuantizer
from transformers.models.internvl.modeling_internvl import \
    InternVLVisionAttention

from ..onnx_export.onnx_utils import export_onnx


class QuantInternVLVisionAttention(InternVLVisionAttention):
    """
    Quantized MHA version of QwenVisionAttention.
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with quantization support.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask
            output_attentions: Whether to output attention weights
            **kwargs: Additional arguments
            
        Returns:
            Tuple of output and attention weights
        """
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.reshape(batch_size, seq_len,
                                            self.num_heads,
                                            self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(batch_size, seq_len, self.num_heads,
                                        self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)

        attn_output, attn_weights = self.quant_eager_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scale,
            is_causal=False,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        output = self.projection_layer(attn_output)
        output = self.projection_dropout(output)

        outputs = (output, attn_weights) if output_attentions else (output,
                                                                    None)
        return outputs

    def quant_eager_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantized attention forward pass.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attention_mask: Optional attention mask
            scaling: Attention scaling factor
            dropout: Dropout probability
            **kwargs: Additional arguments
            
        Returns:
            Tuple of attention output and weights
        """
        query = self.q_bmm_quantizer(query)
        key = self.k_bmm_quantizer(key)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # No upcasting of the attention weights to float32 in this implementation
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = torch.nn.functional.dropout(attn_weights,
                                                   p=dropout,
                                                   training=self.training)

        attn_weights = self.softmax_quantizer(attn_weights)
        value = self.v_bmm_quantizer(value)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights


mtq.register(original_cls=InternVLVisionAttention,
             quantized_cls=QuantInternVLVisionAttention)


class InternVLVisionModel(torch.nn.Module):
    """
    Wrapper for InternVL3 vision model with ONNX export support.
    """

    def __init__(self, hf_model: Any) -> None:
        """
        Initialize the vision model wrapper.
        
        Args:
            hf_model: HuggingFace model instance
        """
        super().__init__()
        self.config = hf_model.config
        self.model = hf_model.model
        self.vision_feature_layer = hf_model.config.vision_feature_layer
        self.vision_feature_select_strategy = hf_model.config.vision_feature_select_strategy
        self.device = hf_model.device
        self.dtype = hf_model.dtype

    def to(self, device: torch.device) -> 'InternVLVisionModel':
        """Move the model to the specified device."""
        super().to(device)
        self.device = device
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision model.
        
        Args:
            pixel_values: Input pixel values
            
        Returns:
            Image features
        """
        image_features = self.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=self.vision_feature_layer,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
        )
        # Reshape to (-1, feature_dim)
        return image_features.reshape(-1, image_features.shape[-1])


def export_internvl3_visual(
    model: InternVLVisionModel,
    output_dir: str,
    torch_dtype: torch.dtype,
) -> None:
    """
    Export InternVL3 visual model to ONNX format.
    
    This function takes an InternVL3 vision model wrapper, prepares dummy inputs 
    for ONNX export, and saves the model in ONNX format.
    
    Args:
        model: InternVL3 vision model wrapper
        output_dir: Directory to save the exported ONNX model
        torch_dtype: PyTorch data type for the model
    """

    # dummy input
    num_patches = 1
    input = torch.randn((num_patches, model.config.vision_config.num_channels,
                         model.config.vision_config.image_size[0],
                         model.config.vision_config.image_size[1]),
                        dtype=torch_dtype,
                        device=model.device)
    inputs = (input, )
    dynamic_axes = {
        'input': {
            0: 'num_blocks'
        },
    }

    # Define input and output names for ONNX
    input_names = ["input"]
    output_names = ["output"]

    export_onnx(model, inputs, output_dir, input_names, output_names,
                dynamic_axes)
