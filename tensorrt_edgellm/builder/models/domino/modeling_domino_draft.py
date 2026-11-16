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
"""Domino checkpoint-direct draft graph: DFlash V1 backbone plus candidate pool.

The engine emits the same proposal bindings as the ONNX Domino draft:

    spec_pool_candidate_ids [B, P]       top-P token ids by base-logit amax
                                         over all proposal positions
    spec_pool_base_logits   [B, BS-1, P] logits gathered per pool token; row 0
                                         is the uncorrected first proposal
    spec_pool_codebook      [B, P, R]    embed_proj.2 rows per pool token
    spec_proposal_zproj     [B, BS-2, R] draft hidden through the z half of
                                         embed_proj.0 for the corrected
                                         positions

The GRU and the state half of embed_proj.0 ship in the runtime sidecar
written by ``domino.artifacts``. Every sequence carries exactly ``BS``
proposal tokens, so the token-major backbone output folds back to
``[B, BS, H]`` inside the graph.
"""

import numpy as np
import tensorrt as trt

from ...ops import functional as F
from ...ops.ragged import RaggedDecoderInputs
from ..dflash2.modeling_dflash2_draft import _to_block_major
from ..dflash.modeling_dflash_draft import DFlashDraftModel

# TensorRT runs TopK with one CTA per row, so a single top-2048 over the full
# vocabulary costs about 1 ms per step. Selecting per chunk first yields the
# exact same set (each chunk contributes at most pool_size members to the
# global top-k) while the chunk stage runs as parallel rows.
_POOL_TOPK_CHUNKS = 16


def pool_topk_chunks(vocab_size: int, pool_size: int) -> int:
    """Chunk count for the two-level pool TopK; 1 selects the single TopK.

    Chunking requires every chunk to hold at least ``pool_size`` entries and an
    exact split, because padded entries would need -inf scores.
    """
    if (vocab_size % _POOL_TOPK_CHUNKS == 0
            and vocab_size >= _POOL_TOPK_CHUNKS * pool_size):
        return _POOL_TOPK_CHUNKS
    return 1


def select_candidate_pool(pool_scores, vocab_size: int, pool_size: int):
    """Return the ids of the top ``pool_size`` scores per row as ``[B, P]``."""
    chunks = pool_topk_chunks(vocab_size, pool_size)
    if chunks == 1:
        _, pool_ids = F.topk(pool_scores, pool_size, 1)
        return pool_ids
    chunk = vocab_size // chunks
    chunked = pool_scores.reshape((-1, chunks, chunk))
    chunk_values, chunk_indices = F.topk(chunked, pool_size, 2)
    offsets = F.constant(
        (np.arange(chunks, dtype=np.int32) * chunk).reshape(1, chunks, 1),
        "pool_chunk_offsets")
    chunk_indices = (chunk_indices + offsets).reshape((-1, chunks * pool_size))
    chunk_values = chunk_values.reshape((-1, chunks * pool_size))
    _, selected = F.topk(chunk_values, pool_size, 1)
    return F.gather_nd(chunk_indices, selected.unsqueeze(2, 2), 1)


def _drop_block_position(hidden, drop_first: bool, width: int):
    """Drop the first or last block position of a ``[B, block, width]`` tensor."""
    shape = F.shape_of(hidden)
    batch = shape[0:1]
    block = shape[1:2]
    one = F.constant(np.asarray([1], dtype=np.int32), "one")
    start = one if drop_first else F.constant(np.asarray([0], dtype=np.int32),
                                              "zero")
    return F.dynamic_slice(hidden, (0, start, 0), (batch, block - one, width))


def _gather_pool_logits(logits, pool_ids):
    """Gather ``[B, F, V]`` logits at ``[B, P]`` ids into ``[B, F, P]``."""
    # Batched gather along the vocab axis; a transpose-based gather_nd would
    # copy the full-vocabulary FP32 logits twice per step.
    return logits.gather(pool_ids, axis=2, num_elementwise_dims=1)


class DominoDraftModel(DFlashDraftModel):
    """DFlash backbone, target head, and Domino candidate-pool outputs."""

    def __init__(self, ctx, lm_head=None) -> None:
        super().__init__(ctx, lm_head)
        cfg = ctx.cfg
        hidden = cfg.hidden_size
        proj_dim = cfg.domino_proj_dim
        if proj_dim <= 0 or cfg.domino_pool_size <= 0:
            raise ValueError(
                "Domino draft requires the dflash_config.projector_type="
                "'domino' contract")
        # embed_proj.0 is one [R, H + G] matrix; the engine consumes the
        # draft-hidden half, the GRU-state half ships in the runtime sidecar.
        embed_proj = ctx.weights.f16("embed_proj.0.weight")
        expected = (proj_dim, hidden + cfg.domino_gru_hidden_dim)
        if tuple(embed_proj.shape) != expected:
            raise ValueError(
                f"Domino embed_proj.0.weight must have shape {expected}, got "
                f"{tuple(embed_proj.shape)}")
        self.z_projection_weight = np.ascontiguousarray(embed_proj[:, :hidden])
        codebook = ctx.weights.f16("embed_proj.2.weight")
        expected = (cfg.vocab_size, proj_dim)
        if tuple(codebook.shape) != expected:
            raise ValueError(
                f"Domino embed_proj.2.weight must have shape {expected}, got "
                f"{tuple(codebook.shape)}")
        self.codebook = codebook

    def forward(self, **io):
        cfg = self.cfg
        hidden = io["inputs_embeds"]
        delta = self.hidden_norm(self.fc(io["base_hidden"]))
        ragged = RaggedDecoderInputs.from_dict(io)
        present = []
        for index, layer in enumerate(self.layers):
            hidden, cache = layer(hidden, delta, io["past_key_values"][index],
                                  io["rope"], ragged, io["delta_rope"],
                                  io["delta_positions"],
                                  io["delta_token_to_sequence"],
                                  io["attention_mask"], io["attention_pos_id"])
            present.append(cache)

        block_hidden = _to_block_major(self.norm(hidden), ragged.query_lengths,
                                       cfg.hidden_size)
        # Both contracts yield block - 1 proposals: shift_label drops the last
        # position (it predicts beyond the verify budget), otherwise position 0
        # is dropped (it only carries the anchor token).
        proposal_hidden = _drop_block_position(
            block_hidden,
            drop_first=not cfg.domino_shift_label,
            width=cfg.hidden_size)
        logits = self.lm_head(proposal_hidden).cast(trt.float32)
        if cfg.final_logit_softcapping is not None:
            cap = np.float32(cfg.final_logit_softcapping)
            logits = (logits / cap).tanh() * cap

        pool_scores = F.reduce(logits, trt.ReduceOperation.MAX, 1 << 1)
        pool_ids = select_candidate_pool(pool_scores, cfg.vocab_size,
                                         cfg.domino_pool_size)
        pool_codebook = F.embedding_lookup(
            F.constant(self.codebook, "domino_codebook"), pool_ids)
        feedback_hidden = _drop_block_position(proposal_hidden,
                                               drop_first=True,
                                               width=cfg.hidden_size)
        zproj = F.linear_with_weights(feedback_hidden,
                                      self.z_projection_weight,
                                      rank=3).cast(trt.float16)

        outputs = {
            "spec_pool_candidate_ids": pool_ids,
            "spec_pool_base_logits": _gather_pool_logits(logits, pool_ids),
            "spec_pool_codebook": pool_codebook,
            "spec_proposal_zproj": zproj,
        }
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
