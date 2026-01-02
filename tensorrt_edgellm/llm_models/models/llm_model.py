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
LLM Model Implementation for Causal Language Modeling

This module provides the main LLM model implementation for efficient
accelerated generation. The model supports standard models and EAGLE3
variants with unified architecture.

The module contains:
- EdgeLLMModel: Main LLM model class with decoder layers and normalization
- EdgeLLMModelForCausalLM: Wrapper for causal language modeling tasks
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn

from ..layers.gather_nd import custom_gather_nd
from ..layers.layers import (EdgeLLMDecoderLayer, PromptTuningEmbedding,
                             Qwen3VLDeepStackProcess)
from ..layers.reduced_lm_head import reduce_lm_head


class EdgeLLMModel(nn.Module):
    """
    EdgeLLM Model for causal language modeling.
    
    This model implements the main component for language modeling, supporting
    standard models and EAGLE3 variants. It processes input through
    decoder layers with proper normalization and can output hidden states
    for EAGLE variants.
    
    Attributes:
        config: Model configuration object
        padding_idx: Padding token index
        vocab_size: Size of the vocabulary
        layers: List of decoder layers
        norm: RMS normalization layer
        embed_tokens: Token embedding layer
        rotary_emb: Rotary embedding layer
        is_eagle_base: Whether this is an EAGLE3 base model
    """

    def __init__(self,
                 hf_model: nn.Module,
                 is_eagle_base: bool = False,
                 use_prompt_tuning: bool = False) -> None:
        """
        Initialize the EdgeLLM model.
        
        Args:
            hf_model: The original model (LlamaForCausalLM, Qwen2ForCausalLM, etc.)
            is_eagle_base: Whether this is an EAGLE3 base model
            use_prompt_tuning: Whether to enable prompt tuning support
        """
        super().__init__()

        # Copy all the basic attributes
        self.config = hf_model.config
        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size
        self.is_eagle_base = is_eagle_base
        self.use_prompt_tuning = use_prompt_tuning

        # Keep all the original components
        self.torch_dtype = hf_model.dtype
        self.embed_tokens = hf_model.embed_tokens.to(self.torch_dtype)
        self.norm = hf_model.norm.to(self.torch_dtype)

        # Replace decoder layers with our custom ones
        self.layers = nn.ModuleList([
            EdgeLLMDecoderLayer(hf_layer, self.torch_dtype, eagle3_draft=False)
            for hf_layer in hf_model.layers
        ])

        # Set max_position_embeddings on attention modules from the model's config
        for layer in self.layers:
            layer.self_attn.max_position_embeddings = self.config.max_position_embeddings

    @property
    def device(self):
        """Get the device of the model's parameters."""
        return next(self.parameters()).device

    def forward(
        self,
        past_key_values: Tuple[torch.FloatTensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Optional[Tuple[
            torch.Tensor, ...]]]:
        """
        Forward pass of the EdgeLLM model.
        
        Args:
            past_key_values: Past key-value cache for efficient decoding
                           List of tensors, each with shape (batch_size, 2, num_kv_heads, max_position_embeddings, head_dim)
            rope_rotary_cos_sin: RoPE rotary embeddings, shape (batch_size, seq_len, rotary_dim)
            context_lengths: Context length tensor indicating current position in cache, shape (batch_size,)
            kvcache_start_index: Start index of KV cache of shape (kv_cache_start_batch_size, 1), required
            position_ids: Position IDs for positional encoding, shape (batch_size, seq_len), optional
            attention_mask: Attention mask for the decoder layers, shape (batch_size, seq_len, seq_len + past_len), optional
            input_ids: Input token IDs of shape (batch_size, seq_len), optional (used for standard models and prompt tuning)
            image_embeds: Image embeddings tensor of shape (image_token_len, hidden_size), optional (used with prompt tuning)
            inputs_embeds: Input embeddings tensor of shape (batch_size, seq_len, hidden_size), optional (legacy support)
            deepstack_visual_embeds: List of deepstack visual embeddings tensors, each with shape (visual_seqlen, hidden_size), optional (used with deepstack processing)
            output_hidden_states: Whether to output hidden states from all layers
            
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Optional[Tuple[torch.Tensor, ...]]]: (hidden_states, present_key_values, all_hidden_states)
                - hidden_states: Final hidden states, shape (batch_size, seq_len, hidden_size)
                - present_key_values: Updated key-value cache, tuple of tensors
                - all_hidden_states: Hidden states from all layers (if output_hidden_states=True), tuple of tensors
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

        hidden_states = inputs_embeds
        present_key_values = ()
        all_hidden_states = () if output_hidden_states else None

        # Process through decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            # Get the past_key_value for this specific layer
            past_key_value = past_key_values[idx] if isinstance(
                past_key_values, (list, tuple)) else past_key_values

            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            hidden_states, present_key_value = decoder_layer(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                rope_rotary_cos_sin=rope_rotary_cos_sin,
                context_lengths=context_lengths,
                kvcache_start_index=kvcache_start_index,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            present_key_values += (present_key_value, )

            if deepstack_visual_embeds is not None and idx in range(
                    len(deepstack_visual_embeds)):
                assert self.config.model_type == "qwen3_vl_text", "Qwen3VLTextModel is required for deepstack processing"
                hidden_states = Qwen3VLDeepStackProcess(
                    self.embed_tokens.num_embeddings)(
                        input_ids,
                        hidden_states,
                        deepstack_visual_embeds[idx],
                    )

        # Apply final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        return hidden_states, present_key_values, all_hidden_states


class EdgeLLMModelForCausalLM(nn.Module):
    """
    EdgeLLM Model for Causal Language Modeling.
    
    This wrapper provides a consistent interface for different types of language
    models, including standard models and EAGLE variants. It handles model
    structure differences and provides uniform forward pass behavior.
    
    Attributes:
        model: The underlying EdgeLLM model
        lm_head: Language model head for token prediction
        config: Model configuration object
        is_eagle_base: Whether this is an EAGLE3 base model
    """

    def __init__(self,
                 hf_model: nn.Module,
                 is_eagle_base: bool = False,
                 use_prompt_tuning: bool = False,
                 reduced_vocab_size: Optional[int] = None,
                 vocab_map: Optional[torch.Tensor] = None) -> None:
        """
        Initialize the EdgeLLM model for causal LM.
        
        Args:
            hf_model: The original model (LlamaForCausalLM, Qwen2ForCausalLM, etc.)
            is_eagle_base: Whether this is an EAGLE3 base model
            use_prompt_tuning: Whether to enable prompt tuning support
            reduced_vocab_size: Size of the reduced vocabulary (optional)
            vocab_map: Tensor of shape (reduced_vocab_size,) with int32 indices for vocabulary reduction (optional)
        """
        super().__init__()

        if use_prompt_tuning:
            if hasattr(hf_model, 'language_model'):
                language_model = hf_model.language_model
                self.config = hf_model.config.text_config
            else:
                # Phi4MM uses the model.model attribute instead of language_model
                language_model = hf_model.model
                self.config = hf_model.config
            if hasattr(hf_model.config, "quantization_config"):
                self.config.quantization_config = hf_model.config.quantization_config
        else:
            language_model = hf_model.model
            self.config = hf_model.config
        self.torch_dtype = hf_model.dtype

        # Create EdgeLLMModel with the original model
        self.model = EdgeLLMModel(language_model, is_eagle_base,
                                  use_prompt_tuning)

        # Handle lm_head with optional vocabulary reduction
        if reduced_vocab_size is not None and vocab_map is not None:
            # Reduce the vocabulary size of lm_head
            print(
                f"Reducing vocabulary size from {hf_model.lm_head.out_features} "
                f"to {reduced_vocab_size}")
            assert vocab_map.shape[
                0] == reduced_vocab_size, f"vocab_map size {vocab_map.shape[0]} does not match reduced_vocab_size {reduced_vocab_size}"
            self.lm_head = reduce_lm_head(hf_model.lm_head, reduced_vocab_size,
                                          vocab_map)
        else:
            # Keep the original lm_head
            self.lm_head = hf_model.lm_head

        self.is_eagle_base = is_eagle_base

    @property
    def device(self):
        """Get the device of the model's parameters."""
        return next(self.parameters()).device

    def forward(
        self,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        last_token_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kvcache_start_index: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], Tuple[
            torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            past_key_values: Past key-value cache for efficient decoding, tuple of tensors
                           Each tensor has shape (batch_size, 2, num_kv_heads, max_position_embeddings, head_dim)
            rope_rotary_cos_sin: RoPE rotary embeddings, shape (batch_size, seq_len, rotary_dim)
            context_lengths: Context length tensor indicating current position in cache, shape (batch_size,)
            last_token_ids: Indices of the last tokens to extract, shape (batch_size,)
            kvcache_start_index: Start index of KV cache of shape (batch_size), optional
            position_ids: Position IDs for positional encoding, shape (batch_size, seq_len), optional
            attention_mask: Attention mask, shape (batch_size, seq_len, seq_len + past_len), optional
            input_ids: Input token IDs of shape (batch_size, seq_len), optional (used for standard models and prompt tuning)
            image_embeds: Image embeddings tensor of shape (image_token_len, hidden_size), optional (used with prompt tuning)
            inputs_embeds: Input embeddings tensor of shape (batch_size, seq_len, hidden_size), optional (legacy support)
            deepstack_visual_embeds: List of deepstack visual embeddings tensors, each with shape (visual_seqlen, hidden_size), optional (used with deepstack processing)

        Returns:
            Union[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]]: Model outputs
                - For standard models: (logits, past_key_values)
                - For EAGLE3 base: (logits, past_key_values, hidden_states)
        """
        # Determine output configuration based on model type
        output_hidden_states = self.is_eagle_base

        # Forward pass through the model
        hidden_states, present_key_values, all_hidden_states = self.model(
            past_key_values=past_key_values,
            rope_rotary_cos_sin=rope_rotary_cos_sin,
            context_lengths=context_lengths,
            kvcache_start_index=kvcache_start_index,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            input_ids=input_ids,
            image_embeds=image_embeds,
            deepstack_visual_embeds=deepstack_visual_embeds,
            inputs_embeds=inputs_embeds,
        )

        # Extract last token hidden states and compute logits
        # Use custom_gather_nd for all models to support batch dimensions
        last_hidden_state_gathered = custom_gather_nd(hidden_states,
                                                      last_token_ids, 1)

        logits = self.lm_head(last_hidden_state_gathered)
        logits = logits.to(torch.float32)

        # Handle different model types
        if self.is_eagle_base:
            # EAGLE3 base model: return concatenated hidden states from specific layers
            idx = [
                2, ((len(all_hidden_states) - 1) // 2),
                len(all_hidden_states) - 4
            ]
            hidden_states_0 = all_hidden_states[idx[0]]
            hidden_states_1 = all_hidden_states[idx[1]]
            hidden_states_2 = all_hidden_states[idx[2]]
            hidden_states = torch.cat(
                [hidden_states_0, hidden_states_1, hidden_states_2],
                dim=-1).to(self.torch_dtype)

            return logits, hidden_states, tuple(present_key_values)

        # Standard model: return logits and past key values
        return logits, tuple(present_key_values)
