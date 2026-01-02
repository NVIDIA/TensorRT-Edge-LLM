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
Vocabulary reduction utilities for TensorRT Edge-LLM.

This module provides functionality to reduce vocabulary based on token frequency.
Supports two algorithms:
1. Frequency-based approach - analyzes token frequency in input text
2. Input-aware approach - algorithm from https://arxiv.org/html/2508.15229v1
"""

from collections import Counter
from typing import Optional, Set

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer


def get_vocab_size(config: AutoConfig) -> int:
    """
    Extract vocabulary size from config, handling nested text_config for VL models.
    
    Args:
        config: HuggingFace AutoConfig instance
        
    Returns:
        Vocabulary size
        
    Raises:
        AttributeError: If vocab_size not found in config or config.text_config
    """
    # Try direct vocab_size attribute first (most models)
    if hasattr(config, 'vocab_size'):
        return config.vocab_size

    # For vision-language models (e.g., Qwen3VL), check text_config
    if hasattr(config, 'text_config') and hasattr(config.text_config,
                                                  'vocab_size'):
        return config.text_config.vocab_size

    raise AttributeError(
        f"Could not find vocab_size in config. Config type: {type(config).__name__}. "
        f"Expected config.vocab_size or config.text_config.vocab_size")


def extract_d2t_required_tokens(d2t_tensor: torch.Tensor,
                                vocab_size: int) -> Set[int]:
    """
    Extract all base token IDs referenced in EAGLE d2t mapping.
    
    Args:
        d2t_tensor: EAGLE d2t tensor where base_token = reduced_token + d2t[reduced_token]
        vocab_size: Maximum vocabulary size for validation
        
    Returns:
        Set of token IDs that must be included in reduced vocabulary
    """
    required_tokens = set()
    print(f"Processing d2t tensor with {len(d2t_tensor)} entries...")

    for reduced_token_id in range(len(d2t_tensor)):
        offset = d2t_tensor[reduced_token_id].item()
        base_token_id = reduced_token_id + offset
        if 0 <= base_token_id < vocab_size:
            required_tokens.add(base_token_id)

    print(f"Extracted {len(required_tokens)} required tokens from d2t mapping")
    return required_tokens


def get_special_tokens(tokenizer: AutoTokenizer) -> Set[int]:
    """
    Get all special token IDs from tokenizer.
    
    Args:
        tokenizer: HuggingFace AutoTokenizer instance
        
    Returns:
        Set of special token IDs (EOS, BOS, PAD, UNK)
        
    Raises:
        ValueError: If tokenizer has no EOS or PAD token
    """
    special_tokens = set()

    # EOS token is required
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id
        if eos_token_id is None:
            raise ValueError(
                "Tokenizer must have eos_token_id or pad_token_id")
    special_tokens.add(eos_token_id)

    # Add other special tokens if present
    if tokenizer.bos_token_id is not None:
        special_tokens.add(tokenizer.bos_token_id)
    if tokenizer.pad_token_id is not None:
        special_tokens.add(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        special_tokens.add(tokenizer.unk_token_id)

    return special_tokens


def input_frequency_filter(dataset: Dataset, tokenizer: AutoTokenizer,
                           target_size: int,
                           exclude_tokens: Set[int]) -> Set[int]:
    """
    Select tokens based on frequency in input articles.
    
    Args:
        dataset: CNN/DailyMail dataset with 'article' field
        tokenizer: HuggingFace AutoTokenizer instance
        target_size: Number of tokens to select
        exclude_tokens: Tokens to exclude from selection (already selected)
        
    Returns:
        Set of selected token IDs
    """
    print(
        f"Analyzing token frequencies in dataset with {len(dataset)} samples..."
    )
    token_counter = Counter()

    for sample in tqdm(dataset, desc="Tokenizing and counting tokens"):
        article = sample.get('article', '')
        if article:
            tokens = tokenizer.encode(article, add_special_tokens=False)
            token_counter.update(tokens)

    print(f"Found {len(token_counter)} unique tokens in dataset")

    # Select most frequent tokens not in exclude set
    selected = set()
    for token_id, _ in token_counter.most_common():
        if token_id not in exclude_tokens:
            selected.add(token_id)
            if len(selected) >= target_size:
                break

    # Ensure we have exactly target_size tokens
    if len(selected) < target_size:
        raise ValueError(
            f"Not enough unique tokens available. Requested {target_size}, "
            f"but only found {len(selected)} unique tokens in dataset.")

    return selected


def input_aware_filter(dataset: Dataset, tokenizer: AutoTokenizer,
                       config: AutoConfig, target_size: int,
                       exclude_tokens: Set[int]) -> Set[int]:
    """
    Input-aware vocabulary reduction algorithm for vocabulary reduction (summarization task).
    
    Implements: static vocabulary + input-aware filtering + tolerance filtering
    
    Args:
        dataset: Dataset with 'article' and 'highlights' fields (CNN/DailyMail format)
        tokenizer: HuggingFace AutoTokenizer instance
        config: HuggingFace AutoConfig instance
        target_size: Number of tokens to select
        exclude_tokens: Tokens to exclude from selection
        
    Returns:
        Set of selected token IDs
    """
    tolerance_k = 5  # Hardcoded tolerance parameter

    print(f"Input-aware vocabulary reduction algorithm for summarization task")
    print(f"Analyzing dataset with {len(dataset)} samples...")

    # Step 1: Analyze output summaries and input documents
    print(f"[Step 1] Building static vocabulary from output summaries...")
    output_counter = Counter()
    input_tokens = set()

    for sample in tqdm(dataset, desc="Analyzing summaries and documents"):
        summary = sample.get('highlights', '')
        if summary:
            output_counter.update(
                tokenizer.encode(summary, add_special_tokens=False))

        document = sample.get('article', '')
        if document:
            input_tokens.update(
                tokenizer.encode(document, add_special_tokens=False))

    print(f"  - {len(output_counter)} unique tokens in summaries")
    print(f"  - {len(input_tokens)} unique tokens in documents")

    # Step 2: Input-aware filtering
    print(f"[Step 2] Applying input-aware filtering...")
    input_aware = {tid for tid in output_counter if tid in input_tokens}
    print(f"  - {len(input_aware)} tokens pass input-aware filter")

    # Step 3: Select core static vocabulary
    print(f"[Step 3] Selecting most frequent task-specific tokens...")
    tolerance_budget = int(target_size * 0.1)
    core_budget = target_size - tolerance_budget

    core_vocab = set()
    # Prioritize input-aware tokens, then fall back to pure frequency
    for token_id, _ in output_counter.most_common():
        if token_id not in exclude_tokens and len(core_vocab) < core_budget:
            if token_id in input_aware or len(core_vocab) >= len(input_aware):
                core_vocab.add(token_id)

    print(f"  - Selected {len(core_vocab)} core task-specific tokens")

    # Step 4: Tolerance filtering (add neighboring tokens)
    print(f"[Step 4] Applying tolerance filtering (k={tolerance_k})...")
    tolerance_tokens = set()

    vocab_size = get_vocab_size(config)
    for token_id in core_vocab:
        for offset in range(-tolerance_k, tolerance_k + 1):
            neighbor_id = token_id + offset
            if (0 <= neighbor_id < vocab_size and neighbor_id not in core_vocab
                    and neighbor_id not in exclude_tokens):
                tolerance_tokens.add(neighbor_id)
                if len(tolerance_tokens) >= tolerance_budget:
                    break
        if len(tolerance_tokens) >= tolerance_budget:
            break

    print(f"  - Added {len(tolerance_tokens)} tolerance tokens")

    # Ensure we have exactly target_size tokens
    final_selected = core_vocab | tolerance_tokens
    if len(final_selected) != target_size:
        raise ValueError(
            f"Filter returned {len(final_selected)} tokens but expected exactly {target_size}. "
            f"Core vocab: {len(core_vocab)}, tolerance: {len(tolerance_tokens)}"
        )

    return final_selected


def reduce_vocab_size(tokenizer: AutoTokenizer,
                      config: AutoConfig,
                      dataset: Dataset,
                      reduced_vocab_size: int,
                      d2t_tensor: Optional[torch.Tensor] = None,
                      method: str = 'frequency') -> torch.Tensor:
    """
    Reduce vocabulary based on selected method.
    
    Args:
        tokenizer: HuggingFace AutoTokenizer instance
        config: HuggingFace AutoConfig instance
        dataset: Dataset to analyze for token frequency
        reduced_vocab_size: Target vocabulary size (must be < config.vocab_size)
        d2t_tensor: Optional EAGLE d2t tensor for required tokens
        method: Vocabulary reduction method ('frequency' or 'input_aware')
        
    Returns:
        vocab_map: torch.Tensor of shape (reduced_vocab_size,) mapping 
                   reduced token IDs to original token IDs (int32)
            
    Raises:
        ValueError: If reduced_vocab_size >= config.vocab_size
        ValueError: If method is not 'frequency' or 'input_aware'
    """
    vocab_size = get_vocab_size(config)
    if reduced_vocab_size >= vocab_size:
        raise ValueError(
            f"reduced_vocab_size ({reduced_vocab_size}) must be less than "
            f"vocab_size ({vocab_size})")

    if method not in ['frequency', 'input_aware']:
        raise ValueError(
            f"method must be 'frequency' or 'input_aware', got '{method}'")

    # Collect required tokens
    required = get_special_tokens(tokenizer)

    if d2t_tensor is not None:
        if reduced_vocab_size <= len(d2t_tensor):
            raise ValueError(
                f"reduced_vocab_size ({reduced_vocab_size}) must be greater than "
                f"d2t_tensor size ({len(d2t_tensor)})")
        required.update(extract_d2t_required_tokens(d2t_tensor, vocab_size))

    # Calculate remaining slots for method-specific selection
    remaining_slots = reduced_vocab_size - len(required)
    if remaining_slots < 0:
        raise ValueError(
            f"Required tokens ({len(required)}) exceeds reduced_vocab_size ({reduced_vocab_size})"
        )

    # Select additional tokens using the chosen method
    if method == 'frequency':
        additional = input_frequency_filter(dataset, tokenizer,
                                            remaining_slots, required)
    else:  # input_aware
        additional = input_aware_filter(dataset, tokenizer, config,
                                        remaining_slots, required)

    # Build final vocabulary
    final_tokens = required | additional

    # Ensure we have exactly the target size
    if len(final_tokens) != reduced_vocab_size:
        raise ValueError(
            f"Final vocabulary size ({len(final_tokens)}) does not match target "
            f"({reduced_vocab_size}). Required: {len(required)}, Additional: {len(additional)}"
        )

    print(f"Final vocabulary composition ({method}):")
    print(f"  - Required tokens (d2t + special): {len(required)}")
    print(f"  - Method-selected tokens: {len(additional)}")
    print(f"  - Total vocabulary size: {len(final_tokens)}")

    return torch.tensor(sorted(final_tokens), dtype=torch.int32)
