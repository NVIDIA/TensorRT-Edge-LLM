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
EAGLE3 Draft Model Implementation

This module provides the EAGLE3 draft model implementation for efficient
accelerated generation. The draft model is used in speculative decoding
to predict multiple tokens ahead with enhanced architecture.

The module contains:
- Eagle3DraftModel: EAGLE3 draft model class with decoder layers and normalization
"""

import json
import os
from typing import Any, List, Optional, Tuple

import modelopt.torch.opt as mto
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import (LlamaRMSNorm,
                                                      LlamaRotaryEmbedding)

from ..layers.gather_nd import custom_gather_nd
from ..layers.layers import EdgeLLMDecoderLayer, PromptTuningEmbedding


class Eagle3DraftModel(nn.Module):
    """
    EAGLE3 Draft Model for speculative decoding.
    
    This model implements the draft component of EAGLE3, which predicts
    multiple tokens ahead to accelerate generation. It features an enhanced
    architecture with proper normalization and fusion layers.
    
    Attributes:
        config: Model configuration object
        padding_idx: Padding token index
        vocab_size: Size of the vocabulary
        embed_tokens: Token embedding layer
        draft_vocab_size: Size of the draft vocabulary (may differ from vocab_size)
        lm_head: Language model head for token prediction
        target_hidden_size: Target hidden size for fusion
        hidden_size: Model hidden size
        fc: Fusion layer for combining different hidden states
        layers: List of decoder layers
        norm: RMS normalization layer
    """

    def __init__(
        self,
        config: Any,
        use_prompt_tuning: bool = False,
    ) -> None:
        """
        Initialize the EAGLE3 draft model.
        
        Args:
            config: Model configuration object containing model parameters
            use_prompt_tuning: Whether to enable prompt tuning support
        """
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_prompt_tuning = use_prompt_tuning
        self.draft_vocab_size = getattr(config, "draft_vocab_size",
                                        config.vocab_size)
        self.register_buffer(
            "d2t", torch.empty(self.draft_vocab_size, dtype=torch.int32))

        # Handle target hidden size for fusion
        self.target_hidden_size = config.target_hidden_size if hasattr(
            config, "target_hidden_size") else config.hidden_size
        self.hidden_size = config.hidden_size

        # Fusion layer for combining hidden states
        bias = getattr(config, "bias", False)
        if hasattr(config, "target_hidden_size"):
            self.fc = nn.Linear(config.target_hidden_size * 3,
                                self.hidden_size,
                                bias=bias)
        else:
            self.fc = nn.Linear(config.hidden_size * 3,
                                self.hidden_size,
                                bias=bias)

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         padding_idx=config.pad_token_id)

        # Decoder layers using our custom EdgeLLMDecoderLayer with config
        self.layers = nn.ModuleList([
            EdgeLLMDecoderLayer(config, index, eagle3_draft=True)
            for index in range(config.num_hidden_layers)
        ])

        # RMS normalization layer
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.log_softmax = nn.LogSoftmax(dim=-1)

        # Language model head for token prediction
        self.lm_head = nn.Linear(config.hidden_size,
                                 self.draft_vocab_size,
                                 bias=False)

        # This logic is adapted from the transformers implementation for Qwen2.5-VL
        # See:https://github.com/huggingface/transformers/blob/v4.55.2/src/transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py#L262
        if config.rope_scaling is not None and "type" in config.rope_scaling:
            if config.rope_scaling["type"] == "mrope":
                config.rope_scaling["type"] = "default"
            config.rope_scaling["rope_type"] = config.rope_scaling["type"]
        # Set default rope theta to 10000 if not specified
        # See: https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py#L111
        if config.rope_scaling is None and not hasattr(config, "rope_theta"):
            print(
                "Warning: rope_theta is not specified, setting default rope_theta to 10000 for EAGLE3 draft model"
            )
            config.rope_theta = 10000.0
        # We use the LlamaRotaryEmbedding for both Qwen2.5-VL and Llama because our quantization process only deals with text inputs.
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    @property
    def device(self):
        """Get the device of the model's parameters."""
        return next(self.parameters()).device

    def forward(
        self,
        past_key_values: List[torch.FloatTensor],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        last_token_ids: torch.Tensor,
        hidden_states_from_base: torch.Tensor,
        hidden_states_from_draft: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kvcache_start_index: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the EAGLE3 draft model.
        
        Args:
            past_key_values: Past key-value cache for efficient decoding
                           List of tensors, each with shape (batch_size, 2, num_kv_heads, max_position_embeddings, head_dim)
            rope_rotary_cos_sin: RoPE rotary embeddings, shape (batch_size, seq_len, rotary_dim)
            context_lengths: Context length tensor indicating current position in cache, shape (batch_size,)
            last_token_ids: Indices of the last tokens to extract, shape (batch_size,)
            hidden_states_from_base: Hidden states from the base model, shape (batch_size, seq_len, target_hidden_size * 3)
            hidden_states_from_draft: Hidden states from previous draft predictions, shape (batch_size, seq_len, hidden_size)
            position_ids: Position IDs for positional encoding, shape (batch_size, seq_len)
            attention_mask: Attention mask for the decoder layers, shape (batch_size, seq_len, seq_len + past_len)
            kvcache_start_index: Start index of KV cache of shape (batch_size), optional
            input_ids: Input token IDs of shape (batch_size, seq_len), optional (used for standard models and prompt tuning)
            image_embeds: Image embeddings tensor of shape (image_token_len, hidden_size), optional (used with prompt tuning)
            inputs_embeds: Input embeddings tensor of shape (batch_size, seq_len, hidden_size), optional (legacy support)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]: (logits, hidden_states, present_key_values)
                - logits: Predicted token logits, shape (batch_size, num_tokens, draft_vocab_size)
                - hidden_states: Final hidden states, shape (batch_size, num_tokens, hidden_size)
                - present_key_values: Updated key-value cache, tuple of tensors
        """

        # Handle input embeddings
        if inputs_embeds is None:
            if self.use_prompt_tuning:
                # For prompt tuning models, use prompt_tuning_embedding
                inputs_embeds = PromptTuningEmbedding(self.embed_tokens)(
                    input_ids, image_embeds)
            else:
                # For standard models, use embed_tokens
                inputs_embeds = self.embed_tokens(input_ids)

        # Fuse hidden states and combine with draft hidden states
        hidden_states = self.fc(hidden_states_from_base)

        # TODO: WAR for INT4 ONNX export
        hidden_states_from_draft = hidden_states_from_draft.to(torch.float16)
        hidden_states = hidden_states.to(torch.float16)
        hidden_states = hidden_states_from_draft + hidden_states

        present_key_values = ()

        # Process through decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            # Get the past_key_value for this specific layer
            past_key_value = past_key_values[idx] if isinstance(
                past_key_values, (list, tuple)) else past_key_values

            hidden_states, present_key_value = decoder_layer(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                rope_rotary_cos_sin=rope_rotary_cos_sin,
                context_lengths=context_lengths,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kvcache_start_index=kvcache_start_index,
            )
            present_key_values += (present_key_value, )

        # Extract last token hidden states using custom_gather_nd to support batch dimensions
        hidden_states = custom_gather_nd(hidden_states, last_token_ids, 1)
        hidden_states_normed = self.norm(hidden_states)
        logits = self.lm_head(hidden_states_normed)
        logits = logits.to(torch.float32)
        logits = self.log_softmax(logits)

        return logits, hidden_states, present_key_values

    def quant_forward(
        self,
        hidden_states,
        hidden_states_from_draft,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """
        Forward pass for quantization of the EAGLE3 draft model.
        
        Args:
            hidden_states: Concatenated hidden states from three selected layers of the base model of shape (batch_size, seq_len, hidden_size * 3)
            hidden_states_from_draft: Hidden states from previous draft predictions of shape (batch_size, seq_len, hidden_size)
            input_ids: Input token IDs of shape (batch_size, seq_len), optional
            inputs_embeds: Input embeddings tensor of shape (batch_size, seq_len, hidden_size), optional (legacy support)
        """
        assert (input_ids is not None) ^ (inputs_embeds is not None), \
            "Exactly one of input_ids or input_embeds must be specified."

        # Handle input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Fuse hidden states and combine with draft hidden states
        hidden_states = self.fc(hidden_states)
        hidden_states = hidden_states_from_draft + hidden_states

        position_ids = torch.arange(0,
                                    input_ids.shape[1],
                                    dtype=input_ids.dtype,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer.quant_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                inputs_embeds=inputs_embeds,
            )

        hidden_states = hidden_states[:, -1]
        hidden_states_normed = self.norm(hidden_states)
        logits = self.lm_head(hidden_states_normed)

        return logits

    @classmethod
    def from_pretrained(
        cls,
        draft_model_dir: str,
        base_model_dir: Optional[str] = None,
        use_prompt_tuning: bool = False,
        device: str = "cuda",
    ) -> "Eagle3DraftModel":
        """
        Load a pre-trained EAGLE3 draft model.
        
        Args:
            draft_model_dir: Path to the draft model directory
            base_model: Base model to copy weights from if needed
            use_prompt_tuning: Whether to enable prompt tuning support
            device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)

        Returns:
            Eagle3DraftModel: Loaded EAGLE3 draft model instance
            
        Raises:
            FileNotFoundError: If model files cannot be found
        """

        # Load configuration
        config = AutoConfig.from_pretrained(draft_model_dir)

        if use_prompt_tuning:
            if hasattr(config, 'text_config'):
                config = config.text_config

        pytorch_bin_path = os.path.join(draft_model_dir, "pytorch_model.bin")
        safetensors_path = os.path.join(draft_model_dir, "model.safetensors")
        # TODO: Compatible with other formats of quantized weights
        quantized_model_path = os.path.join(draft_model_dir,
                                            "modelopt_quantized_model.pth")
        assert os.path.exists(quantized_model_path) or os.path.exists(
            pytorch_bin_path
        ) or os.path.exists(
            safetensors_path
        ), f"Model file not found at {pytorch_bin_path} or {safetensors_path} or {quantized_model_path}"

        model = cls(config, use_prompt_tuning=use_prompt_tuning)

        if os.path.exists(quantized_model_path):
            # Load quantized model from modelopt
            mto.restore(model, quantized_model_path)
        else:
            # Load model from pytorch_model.bin or model.safetensors
            if os.path.exists(pytorch_bin_path):
                print(f"Loading model from {pytorch_bin_path}")
                draft_state_dict = torch.load(pytorch_bin_path,
                                              weights_only=True,
                                              map_location=device)
            else:
                print(f"Loading model from {safetensors_path}")
                draft_state_dict = load_file(safetensors_path, device=device)
            # Handle EAGLE3 specific key mapping
            processed_state_dict = {}
            for key, value in draft_state_dict.items():
                if 'd2t' in key:
                    processed_state_dict[key] = draft_state_dict[key]
                elif 'midlayer' in key:
                    new_key = key.replace('midlayer', 'layers.0')
                    processed_state_dict[new_key] = value
                elif 't2d' in key:
                    continue
                else:
                    processed_state_dict[key] = value

            # Use weights from base model if missing
            def load_embedding_weights(processed_state_dict, base_model_dir,
                                       device):
                from ..model_utils import load_tensor_by_candidate_keys
                if "embed_tokens.weight" not in processed_state_dict:
                    assert base_model_dir is not None, "Base model directory is required to load embedding weights"
                    key_candidates = [
                        "embed_tokens.weight", "model.embed_tokens.weight",
                        "model.language_model.embed_tokens.weight",
                        "language_model.model.embed_tokens.weight"
                    ]
                    embed_tokens_weight = load_tensor_by_candidate_keys(
                        base_model_dir, key_candidates, device)
                    if embed_tokens_weight is not None:
                        processed_state_dict[
                            "embed_tokens.weight"] = embed_tokens_weight
                    else:
                        raise ValueError(
                            "embed_tokens.weight not found in base or draft model"
                        )

            load_embedding_weights(processed_state_dict, base_model_dir,
                                   device)
            model.load_state_dict(processed_state_dict, strict=False)

        return model

    def save_pretrained(self, output_dir: str):
        """
        Save a model to a directory.
        
        Args:
            output_dir: Directory to save the model
        """

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        mto.save(self, os.path.join(output_dir,
                                    "modelopt_quantized_model.pth"))

        # Save config
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"Model saved to {output_dir}")
