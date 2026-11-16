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
"""Canonical DSpark checkpoint metadata resolution."""


def resolve_dspark_config(draft: dict,
                          sliding_window_size: int | None = None) -> dict:
    """Resolve modern, top-level, and legacy DSpark metadata."""
    nested = draft.get("dspark_config") or {}
    legacy = draft.get("dflash_config") or {}

    def value(name, default):
        return nested.get(name, draft.get(name, legacy.get(name, default)))

    causal_head = nested.get(
        "causal_head",
        nested.get(
            "causal",
            draft.get(
                "causal_head",
                draft.get("dflash_query_causal",
                          draft.get("causal", legacy.get("causal", False))))))
    use_swa = bool(value("use_swa", draft.get("use_sliding_window", True)))
    if sliding_window_size is None:
        default_window = draft.get("sliding_window",
                                   legacy.get("swa_window_size", -1))
    else:
        default_window = (sliding_window_size if sliding_window_size > 0 else
                          legacy.get("swa_window_size", -1))
    resolved_window = nested.get("swa_window_size",
                                 draft.get("swa_window_size", default_window))
    resolved_window = int(resolved_window) if use_swa else -1
    contiguous_query_swa = nested.get(
        "contiguous_query_swa",
        draft.get("contiguous_query_swa", resolved_window > 0))

    return {
        "target_layer_ids": value("target_layer_ids", []),
        "block_size": value("block_size", 7),
        "mask_token_id": value("mask_token_id", 151669),
        "enable_confidence_head": value("enable_confidence_head", False),
        "confidence_head_with_markov": value("confidence_head_with_markov",
                                             False),
        "markov_head_type": value("markov_head_type", ""),
        "markov_rank": value("markov_rank", 0),
        "causal_head": causal_head,
        "attention_sink_bias": value("attention_sink_bias", False),
        "use_swa": use_swa,
        "sliding_window_size": resolved_window,
        "contiguous_query_swa": contiguous_query_swa,
        "sample_from_anchor": value("sample_from_anchor", True),
    }
