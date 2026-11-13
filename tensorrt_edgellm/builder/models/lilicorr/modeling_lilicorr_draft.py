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
"""LiLiCorr draft graph: DFlash2 backbone, top-K candidates, lattice transition scores.

The engine matches the ONNX LiLiCorr draft: the DFlash2 draft inputs plus the
runtime-owned FP16 embedding table ``lilicorr_embedding_table [V, H]``, and
the outputs ``spec_proposal_support_ids [B, S, K]`` INT32 and
``spec_lilicorr_transition_scores [B, S, K, K]`` FP32 followed by the
per-layer KV.
"""

from typing import Dict

import numpy as np
import tensorrt as trt

from tensorrt_edgellm.dflash import (LILICORR_EMBEDDING_TABLE_INPUT,
                                     LILICORR_FC_INPUT_SCALE,
                                     LILICORR_SUBLAYER_OUTPUT_SCALE)

from ...ops import GatedMLP, Linear
from ...ops import functional as F
from ..dflash2.modeling_dflash2_draft import (DFlash2DecoderLayer,
                                              DFlash2DraftModel)
from ..dflash.modeling_dflash_draft import DFlashProposalAttention
from .modeling_lilicorr_head import LiLiCorrHead

_SUBLAYER_INPUT_SCALE = 1.0 / LILICORR_SUBLAYER_OUTPUT_SCALE


def select_anchor_rows(delta, delta_owner, query_lengths):
    """Return ``(anchor [B, H], valid [B, 1] FP32)`` from the last delta row of each sequence.

    Delta rows are sequence-major with padding rows owned by ``-1``. A
    sequence without delta rows gets a zero anchor and ``valid = 0``.
    """
    rows = F.arange(F.shape_of(delta_owner)[0:1], start=1)
    sequences = F.arange(F.shape_of(query_lengths)[0:1])
    owned = delta_owner.unsqueeze(0, 1).equal(sequences.unsqueeze(1, 1))
    last_row = (owned.cast(trt.int32) * rows.unsqueeze(0, 1)).reduce(
        trt.ReduceOperation.MAX, 1 << 1, False) - 1
    gathered = delta.gather(last_row.maximum(0), axis=0)
    # Select, not a multiply: an empty sequence gathers row 0, which may be
    # non-finite.
    empty = last_row.equal(-1).unsqueeze(1, 1)
    zero = F.constant(np.zeros((1, 1), dtype=np.float16), "anchor_zero")
    anchor = F.select(empty, zero, gathered)
    valid = (last_row + 1).minimum(1).cast(trt.float32).unsqueeze(1, 1)
    return anchor, valid


class LiLiCorrTargetProjection(Linear):
    """Checkpoint-precision fc on inputs divided by ``LILICORR_FC_INPUT_SCALE``.

    Returns the FP32 projection with the scale restored.
    """

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix, tensor_parallel=False)

    def forward(self, hidden):
        projected = super().forward(hidden * (1.0 / LILICORR_FC_INPUT_SCALE))
        return projected.cast(trt.float32) * LILICORR_FC_INPUT_SCALE


class LiLiCorrMLP(GatedMLP):
    """SwiGLU whose up_proj runs on inputs divided by the sublayer output scale."""

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states).activation(self.activation)
        up = self.up_proj(hidden_states * _SUBLAYER_INPUT_SCALE)
        return self.down_proj(gate * up)


class LiLiCorrProposalAttention(DFlashProposalAttention):
    """Proposal attention with checkpoint sinks and a down-scaled o_proj input."""

    def __init__(self,
                 ctx,
                 prefix: str,
                 layer_index: int,
                 overflow_guard: bool = False) -> None:
        super().__init__(ctx, prefix, layer_index, overflow_guard)
        self.attention_sinks = None
        if ctx.cfg.attention_sink_bias:
            name = self.key("attention_sink_bias")
            sinks = self.weights.f32(name).reshape(-1)
            if sinks.shape != (ctx.cfg.num_attention_heads, ):
                raise ValueError(f"{name} has shape {sinks.shape}, expected "
                                 f"{(ctx.cfg.num_attention_heads,)}")
            self.attention_sinks = np.ascontiguousarray(sinks, np.float32)

    def forward(self, hidden, hidden_delta, past, rope, ragged, delta_rope,
                delta_positions, delta_token_to_sequence, attention_mask,
                attention_pos_id):
        cfg = self.cfg
        key_delta, value_delta = self.project_delta_kv(hidden_delta)
        updated = F.update_dflash_target_cache(key_delta, value_delta, past,
                                               delta_rope, delta_positions,
                                               delta_token_to_sequence,
                                               ragged.kv_page_table)
        qkv, qkv_scales = self.project_qkv(hidden)
        attention, present = F.attention(
            qkv,
            updated,
            rope,
            ragged,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=self.sliding_window_size,
            enable_fp8_kv_cache=False,
            attention_scale=cfg.attention_scaling,
            qkv_scales=qkv_scales,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
            attention_sinks=(F.constant(self.attention_sinks,
                                        "attention_sinks")
                             if self.attention_sinks is not None else None),
        )
        attention = attention.reshape(
            (0, cfg.num_attention_heads * self.head_dim))
        return self.o_proj(attention * _SUBLAYER_INPUT_SCALE), present


class LiLiCorrDecoderLayer(DFlash2DecoderLayer):
    """DFlash2 layer whose attention and MLP outputs are produced at 1/scale.

    The conv finish adds them to the FP32 residual at the same 1/scale and
    restores the scale on its FP32 output.
    """

    attention_class = LiLiCorrProposalAttention

    def __init__(self, ctx, prefix: str, layer_index: int) -> None:
        super().__init__(ctx, prefix, layer_index)
        self.mlp = LiLiCorrMLP(ctx, self.key("mlp"))
        for projection in (self.attention.o_proj, self.mlp.up_proj,
                           self.mlp.down_proj):
            if self.weights.has(projection.prefix + ".bias"):
                raise ValueError("LiLiCorr input scaling requires bias-free "
                                 "o_proj, up_proj and down_proj")

    def _finish(self, conv, output, post, residual):
        scale = LILICORR_SUBLAYER_OUTPUT_SCALE
        return conv.finish(output, post, residual * (1.0 / scale)) * scale


class LiLiCorrDraftModel(DFlash2DraftModel):
    """DFlash2 draft backbone with the LiLiCorr candidate-lattice head."""

    layer_class = LiLiCorrDecoderLayer

    def _target_projection(self, ctx):
        return LiLiCorrTargetProjection(ctx, "fc")

    def _init_projector(self, ctx) -> None:
        if ctx.weights.has("d2t"):
            raise ValueError(
                "LiLiCorr candidates require the full target vocabulary: the "
                "draft checkpoint ships a d2t draft-to-target map")
        self.head = LiLiCorrHead(ctx, "lilicorr")

    def input_tensors(self) -> Dict[str, object]:
        io = super().input_tensors()
        io["embedding_table"] = self.add_input(
            LILICORR_EMBEDDING_TABLE_INPUT, trt.float16,
            (self.cfg.vocab_size, self.cfg.hidden_size))
        return io

    def _proposal_outputs(self, prediction_hidden, delta, io):
        logits = self.lm_head(prediction_hidden).cast(trt.float32)
        values, candidate_ids = F.topk(logits,
                                       self.cfg.lilicorr_candidate_topk, 2)
        maximum = logits.reduce(trt.ReduceOperation.MAX, 1 << 2, True)
        log_partition = (logits - maximum).exp().sum(
            2, keepdim=True).log() + maximum
        anchor, anchor_valid = select_anchor_rows(
            delta, io["delta_token_to_sequence"], io["query_lengths"])
        scores = self.head(
            F.embedding_lookup(io["embedding_table"], candidate_ids),
            values - log_partition, prediction_hidden, anchor, anchor_valid)
        return {
            "spec_proposal_support_ids": candidate_ids,
            "spec_lilicorr_transition_scores": scores,
        }
