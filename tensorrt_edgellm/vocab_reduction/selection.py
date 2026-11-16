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
"""Reference vocabulary-map generation for reduced-vocabulary export."""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional, Set

import torch


def get_vocab_size(config: Any) -> int:
    """Extract vocabulary size from a text or multimodal HF config."""
    if hasattr(config, "vocab_size"):
        return int(config.vocab_size)
    if hasattr(config, "text_config") and hasattr(config.text_config,
                                                  "vocab_size"):
        return int(config.text_config.vocab_size)
    raise AttributeError(
        f"Could not find vocab_size in {type(config).__name__}. Expected "
        "config.vocab_size or config.text_config.vocab_size.")


def _resolve_d2t_target_ids(d2t_tensor: torch.Tensor,
                            vocab_size: int) -> torch.Tensor:
    """Resolve EAGLE d2t as either offsets or direct target token IDs."""
    raw = d2t_tensor.detach().to(device="cpu", dtype=torch.int64).flatten()
    draft_ids = torch.arange(raw.numel(), dtype=torch.int64)
    direct_valid = bool(torch.all((raw >= 0) & (raw < vocab_size)).item())
    offset_targets = draft_ids + raw
    offset_valid = bool(
        torch.all((offset_targets >= 0)
                  & (offset_targets < vocab_size)).item())
    if not direct_valid and not offset_valid:
        raise ValueError(
            "EAGLE d2t tensor is neither a valid direct map nor offset map")

    use_offset_map = False
    if offset_valid and not direct_valid:
        use_offset_map = True
    elif offset_valid and direct_valid:
        direct_unique = int(torch.unique(raw).numel())
        offset_unique = int(torch.unique(offset_targets).numel())
        use_offset_map = offset_unique > direct_unique

    print("Interpreting EAGLE d2t as "
          f"{'offset' if use_offset_map else 'direct'} table")
    return offset_targets if use_offset_map else raw


def extract_d2t_required_tokens(d2t_tensor: torch.Tensor,
                                vocab_size: int) -> Set[int]:
    """Return base-model token IDs referenced by an EAGLE d2t tensor."""
    print(f"Processing d2t tensor with {len(d2t_tensor)} entries...")
    target_ids = _resolve_d2t_target_ids(d2t_tensor, vocab_size)
    required_tokens = {
        int(token_id)
        for token_id in target_ids.tolist() if 0 <= int(token_id) < vocab_size
    }
    print(f"Extracted {len(required_tokens)} required tokens from d2t mapping")
    return required_tokens


def _eos_ids_from(value: Any) -> Set[int]:
    """Normalize an eos_token_id config value (int or list) to a set of ids."""
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple)):
        return {int(item) for item in value}
    return set()


def get_special_tokens(tokenizer: Any, config: Any = None) -> Set[int]:
    """Return special token IDs that must stay available at runtime.

    The full EOS stop set matters: the runtime unions ``eos_token_id`` from
    config.json and generation_config.json (models like HunYuan declare extra
    stop tokens there), and guided decoding needs those ids present in the
    reduced vocabulary to stop on them.
    """
    special_tokens = set()

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id
        if eos_token_id is None:
            raise ValueError(
                "Tokenizer must have eos_token_id or pad_token_id")
    special_tokens.add(int(eos_token_id))

    if config is not None:
        special_tokens |= _eos_ids_from(getattr(config, "eos_token_id", None))

    if tokenizer.bos_token_id is not None:
        special_tokens.add(int(tokenizer.bos_token_id))
    if tokenizer.pad_token_id is not None:
        special_tokens.add(int(tokenizer.pad_token_id))
    if tokenizer.unk_token_id is not None:
        special_tokens.add(int(tokenizer.unk_token_id))

    return special_tokens


def input_frequency_filter(dataset: Any, tokenizer: Any, target_size: int,
                           exclude_tokens: Set[int]) -> Set[int]:
    """Select tokens by frequency in the dataset input text."""
    from tqdm import tqdm

    print(
        f"Analyzing token frequencies in dataset with {len(dataset)} samples..."
    )
    token_counter = Counter()

    for sample in tqdm(dataset, desc="Tokenizing and counting tokens"):
        article = sample.get("article", "")
        if article:
            token_counter.update(
                tokenizer.encode(article, add_special_tokens=False))

    print(f"Found {len(token_counter)} unique tokens in dataset")

    selected = set()
    for token_id, _ in token_counter.most_common():
        if token_id not in exclude_tokens:
            selected.add(int(token_id))
            if len(selected) >= target_size:
                break

    if len(selected) < target_size:
        raise ValueError(
            f"Not enough unique tokens available. Requested {target_size}, "
            f"but only found {len(selected)} unique tokens in dataset.")

    return selected


def input_aware_filter(dataset: Any, tokenizer: Any, config: Any,
                       target_size: int, exclude_tokens: Set[int]) -> Set[int]:
    """Select tokens with the input-aware summarization heuristic."""
    from tqdm import tqdm

    tolerance_k = 5

    print("Input-aware vocabulary reduction algorithm for summarization task")
    print(f"Analyzing dataset with {len(dataset)} samples...")

    print("[Step 1] Building static vocabulary from output summaries...")
    output_counter = Counter()
    input_counter = Counter()

    for sample in tqdm(dataset, desc="Analyzing summaries and documents"):
        summary = sample.get("highlights", "")
        if summary:
            output_counter.update(
                tokenizer.encode(summary, add_special_tokens=False))

        document = sample.get("article", "")
        if document:
            input_counter.update(
                tokenizer.encode(document, add_special_tokens=False))

    input_tokens = set(input_counter)
    print(f"  - {len(output_counter)} unique tokens in summaries")
    print(f"  - {len(input_tokens)} unique tokens in documents")

    print("[Step 2] Applying input-aware filtering...")
    input_aware = {tid for tid in output_counter if tid in input_tokens}
    print(f"  - {len(input_aware)} tokens pass input-aware filter")

    print("[Step 3] Selecting most frequent task-specific tokens...")
    tolerance_budget = int(target_size * 0.1)
    core_budget = target_size - tolerance_budget

    core_vocab = set()
    for token_id, _ in output_counter.most_common():
        if len(core_vocab) >= core_budget:
            break
        if token_id not in exclude_tokens and token_id in input_aware:
            core_vocab.add(int(token_id))
    for token_id, _ in output_counter.most_common():
        if len(core_vocab) >= core_budget:
            break
        if token_id not in exclude_tokens:
            core_vocab.add(int(token_id))
    for token_id, _ in input_counter.most_common():
        if len(core_vocab) >= core_budget:
            break
        if token_id not in exclude_tokens:
            core_vocab.add(int(token_id))

    print(f"  - Selected {len(core_vocab)} core task-specific tokens")

    print(f"[Step 4] Applying tolerance filtering (k={tolerance_k})...")
    tolerance_tokens = set()

    vocab_size = get_vocab_size(config)
    for token_id in core_vocab:
        for offset in range(-tolerance_k, tolerance_k + 1):
            neighbor_id = token_id + offset
            if (0 <= neighbor_id < vocab_size and neighbor_id not in core_vocab
                    and neighbor_id not in exclude_tokens):
                tolerance_tokens.add(int(neighbor_id))
                if len(tolerance_tokens) >= tolerance_budget:
                    break
        if len(tolerance_tokens) >= tolerance_budget:
            break

    print(f"  - Added {len(tolerance_tokens)} tolerance tokens")

    final_selected = core_vocab | tolerance_tokens
    for counter in (output_counter, input_counter):
        if len(final_selected) >= target_size:
            break
        for token_id, _ in counter.most_common():
            if len(final_selected) >= target_size:
                break
            if token_id not in exclude_tokens:
                final_selected.add(int(token_id))
    for token_id in range(vocab_size):
        if len(final_selected) >= target_size:
            break
        if token_id not in exclude_tokens:
            final_selected.add(int(token_id))

    if len(final_selected) > target_size:
        final_selected = set(sorted(final_selected)[:target_size])
    if len(final_selected) < target_size:
        raise ValueError(
            f"Filter returned {len(final_selected)} tokens but expected "
            f"exactly {target_size}. Core vocab: {len(core_vocab)}, "
            f"tolerance: {len(tolerance_tokens)}")

    return final_selected


def reduce_vocab_size(
        tokenizer: Any,
        config: Any,
        dataset: Any,
        reduced_vocab_size: int,
        d2t_tensor: Optional[torch.Tensor] = None,
        method: str = "frequency",
        extra_required_tokens: Optional[Set[int]] = None) -> torch.Tensor:
    """Create a reduced-vocabulary map from calibration data.

    ``extra_required_tokens`` forces additional ids into the map (e.g. the
    generation_config.json EOS stop set, which the runtime unions with the
    model config's EOS).
    """
    vocab_size = get_vocab_size(config)
    if reduced_vocab_size >= vocab_size:
        raise ValueError(
            f"reduced_vocab_size ({reduced_vocab_size}) must be less than "
            f"vocab_size ({vocab_size})")

    if method not in ["frequency", "input_aware"]:
        raise ValueError(
            f"method must be 'frequency' or 'input_aware', got {method!r}")

    required = get_special_tokens(tokenizer, config)
    if extra_required_tokens:
        required.update(
            int(token_id) for token_id in extra_required_tokens
            if 0 <= int(token_id) < vocab_size)

    if d2t_tensor is not None:
        if reduced_vocab_size <= len(d2t_tensor):
            raise ValueError(
                f"reduced_vocab_size ({reduced_vocab_size}) must be greater "
                f"than d2t_tensor size ({len(d2t_tensor)})")
        required.update(extract_d2t_required_tokens(d2t_tensor, vocab_size))

    remaining_slots = reduced_vocab_size - len(required)
    if remaining_slots < 0:
        raise ValueError(
            f"Required tokens ({len(required)}) exceeds reduced_vocab_size "
            f"({reduced_vocab_size})")

    if method == "frequency":
        additional = input_frequency_filter(dataset, tokenizer,
                                            remaining_slots, required)
    else:
        additional = input_aware_filter(dataset, tokenizer, config,
                                        remaining_slots, required)

    final_tokens = required | additional
    if len(final_tokens) != reduced_vocab_size:
        raise ValueError(
            f"Final vocabulary size ({len(final_tokens)}) does not match "
            f"target ({reduced_vocab_size}). Required: {len(required)}, "
            f"Additional: {len(additional)}")

    print(f"Final vocabulary composition ({method}):")
    print(f"  - Required tokens (d2t + special): {len(required)}")
    print(f"  - Method-selected tokens: {len(additional)}")
    print(f"  - Total vocabulary size: {len(final_tokens)}")

    return torch.tensor(sorted(final_tokens), dtype=torch.int32)
