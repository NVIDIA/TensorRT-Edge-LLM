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
Reduced LM Head Implementation

This module provides functionality to reduce the output vocabulary size of
a language model head by selecting a subset of tokens based on a vocabulary map.
"""

import torch
from modelopt.torch.quantization.nn import QuantLinear
from modelopt.torch.quantization.utils import is_quantized_linear
from torch import nn


def reduce_lm_head(lm_head: nn.Linear, reduced_vocab_size: int,
                   vocab_map: torch.Tensor) -> nn.Linear:
    """
    Reduce the vocabulary size of an LM head by selecting tokens based on vocab_map.
    
    Supports both regular nn.Linear and modelopt QuantLinear layers. When reducing
    a QuantLinear, the quantizers (input_quantizer, weight_quantizer, output_quantizer)
    are preserved in the reduced layer.
    
    Args:
        lm_head: Original language model head (Linear or QuantLinear layer)
        reduced_vocab_size: Size of the reduced vocabulary
        vocab_map: Tensor of shape (reduced_vocab_size,) with int32 indices
                   mapping reduced vocab indices to original vocab indices
    
    Returns:
        nn.Linear: New Linear layer with reduced output features (same type as input)
    
    Example:
        >>> lm_head = nn.Linear(4096, 32000)  # hidden_size=4096, vocab_size=32000
        >>> vocab_map = torch.tensor([0, 1, 2, 100, 200], dtype=torch.int32)
        >>> reduced_lm_head = reduce_lm_head(lm_head, 5, vocab_map)
        >>> reduced_lm_head.out_features  # 5
    """
    if vocab_map.shape[0] != reduced_vocab_size:
        raise ValueError(f"vocab_map size {vocab_map.shape[0]} does not match "
                         f"reduced_vocab_size {reduced_vocab_size}")

    # Ensure vocab_map is on the same device as lm_head
    device = lm_head.weight.device
    vocab_map = vocab_map.to(device)

    # Select the corresponding rows from the weight matrix
    # lm_head.weight shape: (vocab_size, hidden_size)
    # We want to select rows corresponding to indices in vocab_map
    reduced_weight = lm_head.weight[
        vocab_map]  # shape: (reduced_vocab_size, hidden_size)

    # Create new Linear layer with reduced output features (no bias for LM heads)
    if is_quantized_linear(lm_head):
        # For QuantLinear, we need to:
        # 1. Create a new QuantLinear with the reduced dimensions
        # 2. Copy the reduced weight
        # 3. Share the quantizers from the original layer (they don't depend on vocab size)

        # Create new QuantLinear instance - this will set up new quantizers via _setup()
        new_lm_head = QuantLinear(lm_head.in_features,
                                  reduced_vocab_size,
                                  bias=False,
                                  device=device,
                                  dtype=lm_head.weight.dtype)

        # Copy the selected weights
        new_lm_head.weight.data = reduced_weight.clone()

        # Replace the auto-created quantizers with the original ones
        # The quantizers are shared (not copied) since they don't depend on vocab size
        # They quantize activations and weights, not vocabulary
        if hasattr(lm_head, 'input_quantizer'):
            # Remove the auto-created quantizer and use the original
            if hasattr(new_lm_head, 'input_quantizer'):
                delattr(new_lm_head, 'input_quantizer')
            new_lm_head.input_quantizer = lm_head.input_quantizer

        if hasattr(lm_head, 'output_quantizer'):
            # Remove the auto-created quantizer and use the original
            if hasattr(new_lm_head, 'output_quantizer'):
                delattr(new_lm_head, 'output_quantizer')
            new_lm_head.output_quantizer = lm_head.output_quantizer

        if hasattr(lm_head, 'weight_quantizer'):
            # Remove the auto-created quantizer and use the original
            if hasattr(new_lm_head, 'weight_quantizer'):
                delattr(new_lm_head, 'weight_quantizer')
            new_lm_head.weight_quantizer = lm_head.weight_quantizer

        print(
            f"Created reduced QuantLinear lm_head from {lm_head.out_features} to {reduced_vocab_size}"
        )
    else:
        # Regular nn.Linear
        new_lm_head = nn.Linear(lm_head.in_features,
                                reduced_vocab_size,
                                bias=False,
                                device=device,
                                dtype=lm_head.weight.dtype)
        # Copy the selected weights
        new_lm_head.weight.data = reduced_weight.clone()

        print(
            f"Created reduced nn.Linear lm_head from {lm_head.out_features} to {reduced_vocab_size}"
        )

    return new_lm_head
