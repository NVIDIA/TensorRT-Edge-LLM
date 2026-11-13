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

Inputs are the DFlash2 draft inputs plus ``lilicorr_embedding_table``, the
runtime's FP16 target input-embedding table ``[V, H]``. Outputs
``spec_proposal_support_ids`` ``[B, S, K]`` INT32 and
``spec_lilicorr_transition_scores`` ``[B, S, K, K]`` FP32 (``[b, s, prev,
next]``) followed by the per-layer KV; the runtime walks the chain.
"""

from dataclasses import replace
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...dflash import (LILICORR_EMBEDDING_TABLE_INPUT, LILICORR_FC_INPUT_SCALE,
                       LILICORR_SUBLAYER_OUTPUT_SCALE)
from ..default.modeling_default import OnnxSpec
from ..dflash2.modeling_dflash2_draft import (DFlash2DecoderLayer,
                                              DFlash2DraftModel,
                                              _make_flat_wrapper)
from ..linear import Q8_0Linear
from .modeling_lilicorr_head import LiLiCorrHead

__all__ = [
    "LILICORR_EMBEDDING_TABLE_INPUT", "LILICORR_FC_INPUT_SCALE",
    "LILICORR_SUBLAYER_OUTPUT_SCALE", "LiLiCorrDecoderLayer",
    "LiLiCorrDraftModel", "select_anchor_rows"
]


def select_anchor_rows(h_delta: torch.Tensor, delta_owner: torch.Tensor,
                       batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(anchor [B, H], valid [B] bool)`` from the last delta row of each sequence.

    Delta rows are sequence-major with padding rows owned by ``-1``. A
    sequence without delta rows gets a zero anchor and ``valid = False``.
    """
    row_index = torch.arange(1,
                             delta_owner.shape[0] + 1,
                             dtype=torch.int32,
                             device=delta_owner.device)
    sequences = torch.arange(batch,
                             dtype=delta_owner.dtype,
                             device=delta_owner.device)
    owned = delta_owner[None, :] == sequences[:, None]
    last_row = (owned.to(torch.int32) * row_index[None, :]).amax(dim=1) - 1
    valid = last_row >= 0
    anchor = h_delta.index_select(0, last_row.clamp_min(0))
    # where, not a multiply: an empty sequence gathers row 0, which may be non-finite.
    return torch.where(valid[:, None], anchor, torch.zeros_like(anchor)), valid


def _scale_first_input(scale: float):

    def hook(_module, args):
        return (args[0] * scale, ) + tuple(args[1:])

    return hook


class LiLiCorrDecoderLayer(DFlash2DecoderLayer):
    """DFlash2 layer whose attention and MLP outputs are produced at 1/scale."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        inverse = 1.0 / LILICORR_SUBLAYER_OUTPUT_SCALE
        for projection in (self.self_attn.o_proj, self.mlp.up_proj,
                           self.mlp.down_proj):
            if getattr(projection, "bias", None) is not None:
                raise ValueError("LiLiCorr input scaling requires bias-free "
                                 "o_proj, up_proj and down_proj")
        for projection in (self.self_attn.o_proj, self.mlp.up_proj):
            projection.register_forward_pre_hook(_scale_first_input(inverse))

    @staticmethod
    def _finish_scaled(conv, scaled_output, coefficients, residual):
        scale = LILICORR_SUBLAYER_OUTPUT_SCALE
        return conv.finish(scaled_output, coefficients, residual *
                           (1.0 / scale)) * scale

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "LiLiCorr draft layers run through forward_ragged only")

    def forward_ragged(self, hidden_states, h_delta, past_key_value,
                       rope_cos_sin, **metadata):
        batch = metadata["query_lengths"].shape[0]
        block = hidden_states.shape[0] // batch
        hidden_size = self.self_attn.hidden_size

        residual = hidden_states.to(torch.float32).reshape(
            batch, block, hidden_size)
        attention_input = self.input_layernorm(residual).to(torch.float16)
        attention_input, attention_post = self.attention_conv.prepare(
            attention_input)
        attention, present_kv = self.self_attn.forward_ragged(
            attention_input.reshape(-1, hidden_size), h_delta, past_key_value,
            rope_cos_sin, **metadata)
        hidden_states = self._finish_scaled(
            self.attention_conv, attention.reshape(batch, block, hidden_size),
            attention_post, residual)

        residual = hidden_states
        mlp_input = self.post_attention_layernorm(residual).to(torch.float16)
        mlp_input, mlp_post = self.mlp_conv.prepare(mlp_input)
        hidden_states = self._finish_scaled(self.mlp_conv, self.mlp(mlp_input),
                                            mlp_post, residual)
        return hidden_states.reshape(-1, hidden_size), present_kv


class LiLiCorrDraftModel(DFlash2DraftModel):
    """DFlash2 draft backbone with the LiLiCorr candidate-lattice head."""

    # Folded head constants (rank features, feature-norm affine) feed FP32 ops.
    match_fp32_consumer_initializers = True

    def _init_projector(self, config) -> None:
        self.layers = nn.ModuleList([
            LiLiCorrDecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        ])
        self.lilicorr = LiLiCorrHead(config)

    def _project_target_hidden(
            self, target_hidden_concat: torch.Tensor) -> torch.Tensor:
        """Run the packed fc on FP16 target hidden and return its FP32 projection."""
        if not self.fc_native_precision:
            return super()._project_target_hidden(target_hidden_concat)
        scaled = target_hidden_concat.to(
            torch.float16) * (1.0 / LILICORR_FC_INPUT_SCALE)
        return self.fc(scaled).to(torch.float32) * LILICORR_FC_INPUT_SCALE

    def forward_ragged(self, inputs_embeds, target_hidden_concat,
                       past_key_values, **metadata):
        embedding_table = metadata.pop(LILICORR_EMBEDDING_TABLE_INPUT)
        prediction_hidden, h_delta, present_key_values = self._ragged_prediction_hidden(
            inputs_embeds, target_hidden_concat, past_key_values, **metadata)
        logits = self.lm_head(prediction_hidden).to(torch.float32)
        values, candidate_ids = torch.topk(logits,
                                           self.config.lilicorr_candidate_topk,
                                           dim=-1)
        log_probs = values - torch.logsumexp(logits, dim=-1, keepdim=True)
        token_embeddings = F.embedding(candidate_ids, embedding_table)
        anchor, anchor_valid = select_anchor_rows(
            h_delta, metadata["delta_token_to_sequence"],
            metadata["query_lengths"].shape[0])
        scores = self.lilicorr(token_embeddings, log_probs, prediction_hidden,
                               anchor, anchor_valid)
        return (candidate_ids.to(torch.int32), scores), present_key_values

    def onnx_export_spec(self) -> OnnxSpec:
        if isinstance(self.lm_head, Q8_0Linear):
            self.lm_head.prepare_for_export()
        self.lilicorr.materialize()
        spec = super().onnx_export_spec()
        config = self.config
        embedding_table = torch.zeros(config.vocab_size,
                                      config.hidden_size,
                                      dtype=torch.float16,
                                      device=spec.args[0].device)
        wrapped = _make_flat_wrapper(self, config.num_hidden_layers,
                                     (LILICORR_EMBEDDING_TABLE_INPUT, ))
        wrapped.eval()
        return replace(
            spec,
            wrapped=wrapped,
            args=spec.args + (embedding_table, ),
            input_names=spec.input_names + [LILICORR_EMBEDDING_TABLE_INPUT],
            dynamic_shapes=spec.dynamic_shapes + [{}],
            output_names=[
                "spec_proposal_support_ids", "spec_lilicorr_transition_scores"
            ] + [
                f"present_key_values_{i}"
                for i in range(self.config.num_hidden_layers)
            ])
