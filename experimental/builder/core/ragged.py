# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Shared configuration and profile shapes for ragged decoder engines."""

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

from . import contracts

_INT32_MAX = (1 << 31) - 1
_MAX_KV_POOL_PAGES = _INT32_MAX // 2
_KV_PAGE_SIZE = 128


def _checked_positive_int32(value: int, name: str) -> int:
    value = int(value)
    if value <= 0 or value > _INT32_MAX:
        raise ValueError(f"{name} must fit a positive int32")
    return value


def checked_physical_tokens(max_num_sequences: int,
                            max_query_length: int) -> int:
    """Return the entry-padded token capacity after checked int32 arithmetic."""
    max_num_sequences = _checked_positive_int32(max_num_sequences,
                                                "max_num_sequences")
    max_query_length = _checked_positive_int32(max_query_length,
                                               "max_query_length")
    if max_num_sequences > _INT32_MAX // max_query_length:
        raise ValueError("ragged physical-token capacity exceeds int32")
    return max_num_sequences * max_query_length


def checked_kv_pool_pages(max_num_sequences: int,
                          max_kv_cache_capacity: int) -> Tuple[int, int]:
    """Return pages per sequence and pool pages using the C++ limits."""
    max_num_sequences = _checked_positive_int32(max_num_sequences,
                                                "max_num_sequences")
    max_kv_cache_capacity = _checked_positive_int32(max_kv_cache_capacity,
                                                    "max_kv_cache_capacity")
    max_aligned_capacity = (_INT32_MAX // _KV_PAGE_SIZE) * _KV_PAGE_SIZE
    if max_kv_cache_capacity > max_aligned_capacity:
        raise ValueError("max_kv_cache_capacity exceeds the int32 page limit")
    pages_per_sequence = ((max_kv_cache_capacity + _KV_PAGE_SIZE - 1) //
                          _KV_PAGE_SIZE)
    if max_num_sequences > _MAX_KV_POOL_PAGES // pages_per_sequence:
        raise ValueError("minimum active KV pages exceed the int32 pool limit")
    return pages_per_sequence, max_num_sequences * pages_per_sequence


def _role_generation_query_length(args) -> int:
    if args.resolved_spec_role == contracts.SpecRole.DRAFT:
        return _checked_positive_int32(args.max_draft_tree_size,
                                       "max_draft_tree_size")
    if args.resolved_spec_role == contracts.SpecRole.BASE:
        return _checked_positive_int32(args.max_verify_tree_size,
                                       "max_verify_tree_size")
    return 1


def diffusion_canvas_length(root: dict,
                            model_dir: str = "",
                            generation_config=None) -> int:
    """Resolve the checked DLLM canvas shared by every build artifact."""
    if "canvas_length" in root:
        value = root["canvas_length"]
    else:
        if generation_config is None:
            generation_config = {}
            path = os.path.join(model_dir, "generation_config.json")
            if model_dir and os.path.isfile(path):
                with open(path) as config_file:
                    generation_config = json.load(config_file)
        value = generation_config.get("canvas_length", 256)
    return _checked_positive_int32(value, "canvas_length")


def builder_config_fields(_cfg, _args) -> Dict[str, object]:
    """Return the serialized ragged fields shared with the C++ builder."""
    return {"ragged_backend": "entry_padded_compatibility"}


Shape = Tuple[int, ...]
ShapeRange = Tuple[Shape, Shape, Shape]


@dataclass(frozen=True)
class DecoderProfileRange:
    """Dynamic extents for one homogeneous ragged decoder profile."""

    num_sequences: Tuple[int, int, int]
    physical_tokens: Tuple[int, int, int]
    query_offsets: Tuple[int, int, int]
    logits_rows: Tuple[int, int, int]

    def token(self, width: int = 0) -> ShapeRange:
        return tuple((tokens, width) if width else (tokens, )
                     for tokens in self.physical_tokens)

    def sequence(self) -> ShapeRange:
        return tuple((sequences, ) for sequences in self.num_sequences)

    def offsets(self) -> ShapeRange:
        return tuple((offsets, ) for offsets in self.query_offsets)


def decoder_profile_ranges(
    args,
    max_generation_query_length: int = 0
) -> Tuple[DecoderProfileRange, DecoderProfileRange]:
    """Return prefill and generation ranges for a decoder engine role."""
    max_sequences = _checked_positive_int32(args.max_batch_size,
                                            "max_batch_size")
    max_query_length = _checked_positive_int32(args.max_input_len,
                                               "max_input_len")
    max_prefill_tokens = checked_physical_tokens(max_sequences,
                                                 max_query_length)
    opt_prefill_tokens = checked_physical_tokens(max_sequences,
                                                 max(1, max_query_length // 2))
    spec_engine = args.resolved_spec_role != contracts.SpecRole.NONE
    prefill_logits = (1, opt_prefill_tokens,
                      max_prefill_tokens) if spec_engine else (1,
                                                               max_sequences,
                                                               max_sequences)
    prefill = DecoderProfileRange(
        (1, max_sequences, max_sequences),
        (1, opt_prefill_tokens, max_prefill_tokens),
        (2, max_sequences + 1, max_sequences + 1),
        prefill_logits,
    )

    if max_generation_query_length:
        max_generation_query = max_generation_query_length
    else:
        max_generation_query = _role_generation_query_length(args)
    max_generation_tokens = checked_physical_tokens(max_sequences,
                                                    max_generation_query)
    logits_rows = (max_generation_tokens if args.resolved_spec_role
                   != contracts.SpecRole.NONE else max_sequences)
    generation = DecoderProfileRange(
        (1, max_sequences, max_sequences),
        (1, max_generation_tokens, max_generation_tokens),
        (2, max_sequences + 1, max_sequences + 1),
        (1, logits_rows, logits_rows),
    )
    return prefill, generation


def phase_shapes(optimum: int) -> ShapeRange:
    """Profile the phase marker by its shape-encoded ABI extent."""
    return (1, ), (optimum, ), (8, )


def fixed_shape(shape: Shape) -> ShapeRange:
    """Return a fixed-capacity profile range."""
    return shape, shape, shape
