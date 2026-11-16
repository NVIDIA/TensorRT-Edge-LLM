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
"""Domino draft graph: DFlash V1 backbone plus in-engine candidate pool.

Domino shares the DFlash cached-draft transformer unchanged and adds a
sequential causal correction head executed by the runtime.  The engine
materializes everything the correction rollout needs per round:

    spec_pool_candidate_ids [B, P]         Shared candidate pool: top-P token
                                           ids by base logits amax over all
                                           proposal positions.
    spec_pool_base_logits   [B, BS-1, P]   Base logits gathered per pool token
                                           at each proposal position; row 0 is
                                           the uncorrected first proposal.
    spec_pool_codebook      [B, P, R]      Correction codebook rows
                                           (embed_proj.2) gathered per pool
                                           token.
    spec_proposal_zproj     [B, BS-2, R]   Draft-hidden projection through the
                                           z half of embed_proj.0 for the
                                           corrected positions.

The draft engine consumes the token-major ragged ABI shared with DFlash:
``inputs_embeds`` is ``[B * BS, H]`` and the pool outputs are folded back to
per-sequence shapes inside the graph, which requires every sequence to carry
exactly ``BS`` proposal tokens.  Position 0 is seeded with the last accepted
token; with ``shift_label`` the hidden at position i predicts draft token
i + 1 and the last position is dropped, otherwise position i predicts the
masked token i and position 0 is dropped.  The GRU and the state half of
embed_proj.0 stay out of the engine (runtime sidecar
domino_heads.safetensors).
"""

import itertools
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..default.modeling_default import OnnxSpec
from ..dflash2.modeling_dflash2_draft import _make_flat_wrapper
from ..dflash.modeling_dflash_draft import DFlashDraftModel
from ..linear import make_linear
from ..ops import KV_PAGE_SIZE

__all__ = ["DominoDraftModel"]

_BATCH_SIZE = 2
# TensorRT runs TopK with one CTA per row, so a single top-2048 over the full vocab
# costs about 1 ms per step. Selecting per chunk first keeps the exact same set
# (each chunk contributes at most pool_size members to the global top-k) while the
# chunk stage runs as _POOL_TOPK_CHUNKS parallel rows.
_POOL_TOPK_CHUNKS = 16


def select_candidate_pool(pool_scores: torch.Tensor,
                          pool_size: int) -> torch.Tensor:
    """Return the indices of the top ``pool_size`` scores per row.

    Equivalent to ``torch.topk(pool_scores, pool_size, dim=-1).indices`` as a set;
    the order within the pool is by descending score.
    """
    vocab = pool_scores.shape[-1]
    if vocab < _POOL_TOPK_CHUNKS * pool_size:
        return torch.topk(pool_scores, pool_size, dim=-1).indices
    chunk = (vocab + _POOL_TOPK_CHUNKS - 1) // _POOL_TOPK_CHUNKS
    padded = chunk * _POOL_TOPK_CHUNKS
    if padded != vocab:
        pool_scores = F.pad(pool_scores, (0, padded - vocab),
                            value=float("-inf"))
    chunked = pool_scores.reshape(pool_scores.shape[0], _POOL_TOPK_CHUNKS,
                                  chunk)
    chunk_values, chunk_indices = torch.topk(chunked, pool_size, dim=-1)
    offsets = (torch.arange(_POOL_TOPK_CHUNKS, device=pool_scores.device) *
               chunk).view(1, _POOL_TOPK_CHUNKS, 1)
    chunk_indices = (chunk_indices + offsets).reshape(pool_scores.shape[0], -1)
    chunk_values = chunk_values.reshape(pool_scores.shape[0], -1)
    selected = torch.topk(chunk_values, pool_size, dim=-1).indices
    return torch.gather(chunk_indices, 1, selected)


_CTX_LEN = 2
_MIN_BLOCK_SIZE = 3  # 1 seed + 1 uncorrected + >=1 corrected proposal


class DominoCandidatePool(nn.Module):
    """In-engine candidate-pool parameters (filled by a model.py load hook)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.z_projection = make_linear(config,
                                        config.hidden_size,
                                        config.domino_proj_dim,
                                        bias=False,
                                        module_name="domino_pool.z_projection")
        self.codebook = nn.Parameter(torch.empty(config.vocab_size,
                                                 config.domino_proj_dim,
                                                 dtype=torch.float16),
                                     requires_grad=False)


class DominoDraftModel(DFlashDraftModel):
    """DFlash backbone, target head, and Domino candidate-pool outputs."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.domino_pool = DominoCandidatePool(config)

    def forward_ragged(self, inputs_embeds: torch.Tensor,
                       target_hidden_concat: torch.Tensor,
                       past_key_values: Tuple[torch.Tensor, ...], **metadata):
        hidden_states, present = self._run_ragged_backbone(
            inputs_embeds, target_hidden_concat, past_key_values, **metadata)

        batch = metadata["query_lengths"].shape[0]
        block = hidden_states.shape[0] // batch
        block_hidden = self.norm(hidden_states).to(torch.float16).reshape(
            batch, block, self.config.hidden_size)
        # Both contracts yield block - 1 proposals: shift_label drops the last
        # position (it predicts beyond the verify budget), otherwise position 0
        # is dropped (it only carries the anchor token).
        if self.config.domino_shift_label:
            proposal_hidden = block_hidden[:, :-1, :]
        else:
            proposal_hidden = block_hidden[:, 1:, :]
        logits = self._softcap_logits(
            self.lm_head(proposal_hidden).to(torch.float32))

        # Shared candidate pool over every proposal position, so position 0's
        # argmax is in the pool and the runtime picks the first proposal from
        # row 0 without a separate full-vocabulary argmax.
        pool_scores = logits.amax(dim=1)
        pool_ids = select_candidate_pool(pool_scores,
                                         self.config.domino_pool_size)
        pool_base_logits = torch.gather(
            logits, 2,
            pool_ids.unsqueeze(1).expand(-1, logits.shape[1], -1))
        pool_codebook = F.embedding(pool_ids, self.domino_pool.codebook)
        zproj = self.domino_pool.z_projection(proposal_hidden[:, 1:, :]).to(
            torch.float16)

        outputs = (pool_ids.to(torch.int32), pool_base_logits, pool_codebook,
                   zproj)
        return outputs, present

    def onnx_export_spec(self) -> OnnxSpec:
        config = self.config
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        batch = _BATCH_SIZE
        block = config.dflash_block_size
        delta_len = _CTX_LEN
        physical_tokens = batch * block
        delta_tokens = batch * delta_len
        num_layers = config.num_hidden_layers
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        args = [
            torch.zeros(physical_tokens,
                        config.hidden_size,
                        dtype=torch.float16,
                        device=device),
            torch.zeros(delta_tokens,
                        len(config.dflash_target_layer_ids) *
                        config.hidden_size,
                        dtype=torch.float16,
                        device=device),
        ]
        args.extend(
            torch.zeros(2,
                        1,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        config.head_dim,
                        dtype=torch.float16,
                        device=device) for _ in range(num_layers))
        positions = torch.arange(block, dtype=torch.int32,
                                 device=device).repeat(batch)
        lengths = torch.full((batch, ),
                             block,
                             dtype=torch.int32,
                             device=device)
        args.extend([
            torch.zeros(physical_tokens,
                        rotary_dim,
                        dtype=torch.float32,
                        device=device),
            positions,
            torch.arange(0,
                         physical_tokens + 1,
                         block,
                         dtype=torch.int32,
                         device=device),
            lengths,
            torch.zeros(batch, dtype=torch.int32, device=device),
            lengths.clone(),
            torch.arange(batch, dtype=torch.int32, device=device),
            torch.zeros(4, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            torch.zeros(batch, 2, 1, dtype=torch.int32, device=device),
            torch.zeros(delta_tokens,
                        rotary_dim,
                        dtype=torch.float32,
                        device=device),
            torch.arange(delta_len, dtype=torch.int32,
                         device=device).repeat(batch),
            torch.arange(batch, dtype=torch.int32,
                         device=device).repeat_interleave(delta_len),
            positions.clone(),
            torch.zeros(physical_tokens, (block + 31) // 32,
                        dtype=torch.int32,
                        device=device),
        ])
        input_names = (
            ["inputs_embeds", "dflash_target_hidden_concat"] +
            [f"past_key_values_{i}" for i in range(num_layers)] + [
                "rope_rotary_cos_sin", "positions", "query_start_offsets",
                "query_lengths", "past_lengths", "attention_sequence_lengths",
                "state_indices", "execution_phase_marker",
                "context_sequence_count_carrier", "kv_page_table",
                "dflash_delta_rope_cos_sin", "dflash_delta_positions",
                "dflash_delta_token_to_sequence", "attention_position_ids",
                "packed_attention_mask"
            ])
        output_names = [
            "spec_pool_candidate_ids", "spec_pool_base_logits",
            "spec_pool_codebook", "spec_proposal_zproj"
        ] + [f"present_key_values_{i}" for i in range(num_layers)]
        token_dim = torch.export.Dim("physical_tokens",
                                     min=_MIN_BLOCK_SIZE,
                                     max=8_388_608)
        delta_dim = torch.export.Dim("delta_tokens", min=1, max=8_388_608)
        seq_dim = torch.export.Dim("num_sequences", min=1, max=256)
        pages_dim = torch.export.Dim("num_pages", min=1, max=1048576)
        max_pages_dim = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        phase_dim = torch.export.Dim("execution_phase_extent", min=1, max=8)
        packed_width = torch.export.Dim("packed_mask_width", min=1, max=64)
        dynamic_shapes = [{0: token_dim}, {0: delta_dim}]
        dynamic_shapes += [{1: pages_dim} for _ in range(num_layers)]
        dynamic_shapes += [{
            0: token_dim
        }, {
            0: token_dim
        }, {
            0: seq_dim + 1
        }, {
            0: seq_dim
        }, {
            0: seq_dim
        }, {
            0: seq_dim
        }, {
            0: seq_dim
        }, {
            0: phase_dim
        }, {}, {
            0: seq_dim,
            2: max_pages_dim
        }, {
            0: delta_dim
        }, {
            0: delta_dim
        }, {
            0: delta_dim
        }, {
            0: token_dim
        }, {
            0: token_dim,
            1: packed_width
        }]
        wrapped = _make_flat_wrapper(self, num_layers)
        wrapped.eval()
        return OnnxSpec(wrapped=wrapped,
                        args=tuple(args),
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=dynamic_shapes)
