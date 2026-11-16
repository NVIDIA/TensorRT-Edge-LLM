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
"""Attention operations."""

from typing import Optional, Sequence, Tuple

import tensorrt as trt

from ..ragged import RaggedDecoderInputs
from ..tensor import Tensor
from ._operation import network_input, operation, operation_available

KV_PAGE_SIZE = 128

__all__ = [
    "KV_PAGE_SIZE", "attention", "gemma4_attention", "qsa_attention",
    "sparse_attention", "sparse_attention_available", "vit_attention"
]


def attention(
    qkv: Tensor,
    past_kv: Tensor,
    rope_cos_sin: Tensor,
    ragged: RaggedDecoderInputs,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    sliding_window_size: int = -1,
    enable_fp8_kv_cache: bool = False,
    qkv_scales: Sequence[float] = (1.0, 1.0, 1.0),
    q_norm_gamma: Optional[Tensor] = None,
    k_norm_gamma: Optional[Tensor] = None,
    rms_norm_eps: float = 1e-6,
    qk_norm_post_rope: bool = False,
    attention_scale: Optional[float] = None,
    enable_kv_shared: bool = False,
    context_mask_selector: Optional[Tensor] = None,
    vision_block_ids: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    attention_pos_id: Optional[Tensor] = None,
    attention_sinks: Optional[Tensor] = None,
    enable_contiguous_query_swa: bool = False,
    skip_softmax_prefill_enabled: bool = False,
    skip_softmax_decode_enabled: bool = False,
    prefix_key_value: Optional[Tensor] = None,
    prefix_page_table: Optional[Tensor] = None,
    fp8_kv_prefix: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Run paged decoder attention and return output plus present KV.

    ``qkv`` is packed on the last dimension as Q, K, V. A shared-KV layer
    passes Q alone because K/V come from a donated cache. With an external
    prefix, past_kv and ragged.kv_page_table describe private action storage;
    the prefix pool and table are read-only.
    """

    if (prefix_key_value is None) != (prefix_page_table is None):
        raise ValueError(
            "prefix_key_value and prefix_page_table must be supplied together")
    if fp8_kv_prefix and prefix_key_value is None:
        raise ValueError("fp8_kv_prefix requires an external prefix pool")
    if (attention_mask is None) != (attention_pos_id is None):
        raise ValueError(
            "attention_mask and attention_pos_id must be supplied together")
    if (q_norm_gamma is None) != (k_norm_gamma is None):
        raise ValueError(
            "q_norm_gamma and k_norm_gamma must be supplied together")
    enable_qk_norm = q_norm_gamma is not None
    if enable_qk_norm and enable_kv_shared:
        raise ValueError("QK normalization is not supported with shared KV")
    if qk_norm_post_rope and not enable_qk_norm:
        raise ValueError("qk_norm_post_rope requires Q/K norm gammas")
    if vision_block_ids is not None and attention_mask is not None:
        raise ValueError(
            "vision-block attention and tree attention are mutually exclusive")
    attributes = {
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "enable_tree_attention": int(attention_mask is not None),
        "enable_qk_norm": int(enable_qk_norm),
        "enable_kv_shared": int(enable_kv_shared),
        "enable_context_mask_selector": int(context_mask_selector is not None),
        "enable_fp8_kv_cache": int(enable_fp8_kv_cache),
        "enable_vision_block_attention": int(vision_block_ids is not None),
        "enable_attention_sink": int(attention_sinks is not None),
        "enable_contiguous_query_swa": int(enable_contiguous_query_swa),
        "sliding_window_size": sliding_window_size,
        "qkv_scales": qkv_scales,
        "skip_softmax_prefill_input": int(skip_softmax_prefill_enabled),
        "skip_softmax_decode_input": int(skip_softmax_decode_enabled),
    }
    if attention_scale is not None:
        attributes["attention_scale"] = attention_scale
    if enable_qk_norm:
        attributes["rms_norm_eps"] = rms_norm_eps
    if qk_norm_post_rope:
        # QK-norm order: rotate then norm (HunYuan V1). Emitted only when set so
        # norm-then-rotate families (Qwen3) keep their exact field set; the
        # plugin defaults the missing field to 0.
        attributes["qk_norm_post_rope"] = 1
    inputs = [
        qkv, past_kv, ragged.query_lengths, rope_cos_sin, ragged.past_lengths,
        ragged.kv_page_table
    ]
    if enable_qk_norm:
        inputs.extend((q_norm_gamma, k_norm_gamma))
    if context_mask_selector is not None:
        inputs.append(context_mask_selector)
    if attention_mask is not None:
        inputs.extend((attention_mask, attention_pos_id))
    if vision_block_ids is not None:
        inputs.append(vision_block_ids)
    if skip_softmax_prefill_enabled:
        # Scale table shared by every attention op: INT32 flattened
        # (seqlen_bin, S) pairs the runtime uploads from config.json; length 0
        # selects the dense cubin.
        inputs.append(
            network_input("skip_softmax_prefill_scale", trt.int32, (-1, )))
    if skip_softmax_decode_enabled:
        # Decode-step table, same format; the plugin reads it on vanilla decode
        # steps only.
        inputs.append(
            network_input("skip_softmax_decode_scale", trt.int32, (-1, )))
    if attention_sinks is not None:
        inputs.append(attention_sinks)
    inputs.extend(
        (ragged.query_start_offsets, ragged.attention_sequence_lengths,
         ragged.execution_phase_marker, ragged.context_sequence_count_carrier))
    if prefix_key_value is not None:
        attributes["external_kv_prefix"] = 1
        attributes["fp8_kv_prefix"] = int(fp8_kv_prefix)
        inputs.extend((prefix_key_value, prefix_page_table))
    return operation("attention", inputs, output_count=2, **attributes)


def qsa_attention(
    qkv: Tensor,
    index_qk: Tensor,
    past_kv: Tensor,
    rope_cos_sin: Tensor,
    ragged: RaggedDecoderInputs,
    q_norm_gamma: Tensor,
    k_norm_gamma: Tensor,
    indexer_q_norm_gamma: Tensor,
    indexer_k_norm_gamma: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    indexer_n_heads: int,
    indexer_head_dim: int,
    indexer_budget: int,
    indexer_compress_ratio: int,
    attention_scale: float,
    rms_norm_eps: float,
    tree_parent_ids: Optional[Tensor] = None,
    tree_depths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Run Qwen sparse attention and return ``[tokens, heads, head_size]`` plus present KV.

    The paged pool is ``head_size + indexer_head_dim`` wide: the tail columns
    persist the indexer state across steps rather than forming a second cache.

    The gamma convention is asymmetric and not interchangeable: the main pair
    carries the effective multiplier the kernel applies as-is, so a
    zero-centered norm must be folded to ``1 + w`` by the caller, while the
    indexer pair stays raw because its kernel adds the one itself.

    With a tree each node writes its tails (raw index-K, plus the FP32 rope row
    when its position opens a block) into its own pool slot; moving an accepted
    row commits them.
    """
    if (tree_parent_ids is None) != (tree_depths is None):
        raise ValueError(
            "tree parent IDs and depths must be supplied together")
    use_spec_verify_state = tree_parent_ids is not None
    if use_spec_verify_state and ragged.spec_verify_layout is None:
        raise ValueError("Verify state requires spec_verify_layout")
    inputs = [
        qkv, index_qk, past_kv, ragged.query_lengths, rope_cos_sin,
        ragged.past_lengths, ragged.kv_page_table, q_norm_gamma, k_norm_gamma,
        indexer_q_norm_gamma, indexer_k_norm_gamma, ragged.query_start_offsets,
        ragged.attention_sequence_lengths, ragged.execution_phase_marker,
        ragged.context_sequence_count_carrier
    ]
    if use_spec_verify_state:
        inputs.extend(
            (tree_parent_ids, tree_depths, ragged.spec_verify_layout))
    return operation("qsa_attention",
                     inputs,
                     output_count=2,
                     num_q_heads=num_q_heads,
                     num_kv_heads=num_kv_heads,
                     head_size=head_size,
                     indexer_n_heads=indexer_n_heads,
                     indexer_head_dim=indexer_head_dim,
                     indexer_budget=indexer_budget,
                     indexer_compress_ratio=indexer_compress_ratio,
                     attention_scale=attention_scale,
                     rms_norm_eps=rms_norm_eps,
                     use_spec_verify_state=int(use_spec_verify_state))


def gemma4_attention(q_raw: Tensor, k_raw: Tensor, value: Tensor,
                     gamma: Tensor, relative_key: Tensor, valid: Tensor,
                     sequence_length: Tensor, chunk_size: int,
                     left_horizon: int, context_size: int,
                     logit_cap: float) -> Tensor:
    """Run Gemma4 chunked local audio attention."""
    inputs = [q_raw, k_raw, value, gamma, relative_key, valid, sequence_length]
    return operation("gemma4_attention",
                     inputs,
                     chunk_size=chunk_size,
                     left_horizon=left_horizon,
                     context_size=context_size,
                     logit_cap=logit_cap)


def vit_attention(query: Tensor, key: Tensor, value: Tensor,
                  cu_seqlens: Tensor, max_seqlen_carrier: Tensor,
                  num_heads: int, head_size: int) -> Tensor:
    """Run packed vision-transformer attention."""
    return operation("vit_attention",
                     [query, key, value, cu_seqlens, max_seqlen_carrier],
                     num_heads=num_heads,
                     head_size=head_size)


def sparse_attention_available() -> bool:
    """Return whether the loaded plugins provide block-sparse attention."""
    return operation_available("sparse_attention")


def sparse_attention(query: Tensor, key: Tensor, value: Tensor,
                     suffix_rows: Tensor) -> Tensor:
    """Run noncausal block-sparse attention over one packed sequence.

    Query, key, and value are ``[sequence, heads, head_size]``. The rows
    before the last ``suffix_rows.shape[0]`` form a dense prefix: they attend
    exactly and are attended exactly. Suffix query blocks attend exactly to
    their routed key blocks and to the other blocks through block summaries.
    """
    inputs = [tensor.cast(trt.bfloat16) for tensor in (query, key, value)]
    return operation("sparse_attention", inputs + [suffix_rows])
