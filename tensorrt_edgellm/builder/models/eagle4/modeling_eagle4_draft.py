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
"""EAGLE-4 checkpoint-direct draft graph.

It mirrors ``tensorrt_edgellm.models.eagle4.Eagle4DraftModel``: the EAGLE3
draft inputs, per-feature ``fc_norm``, an optionally normalized feedback
state, a per-layer sliding window, and either the low-rank candidate head
(``spec_proposal_support_*`` outputs) or the dense ``lm_head`` (``logits``).
"""

import numpy as np
import tensorrt as trt

from tensorrt_edgellm.config import _EAGLE_CANDIDATE_HEAD_KEY_PREFIXES

from ...ops import (GatedMLP, Linear, Module, NetworkModule, RMSNorm,
                    TreeAttention)
from ...ops import functional as F
from ...ops.ragged import RaggedDecoderInputs
from ..eagle3.modeling_eagle3_draft import Eagle3DraftModel

# Checkpoint names of the candidate head factors, keyed by the ONNX draft
# model's module names.
_CANDIDATE_HEAD_WEIGHT = {
    module.rstrip("."): f"{prefix}weight"
    for prefix, module in _EAGLE_CANDIDATE_HEAD_KEY_PREFIXES
}


def _sliding_window_size(cfg, layer_index: int) -> int:
    """Window of one draft layer; without layer labels every layer slides.

    ``cfg.sliding_window_size`` is -1 for a draft with ``use_sliding_window:
    false``, even when a layer is labelled ``sliding_attention``; the ONNX
    exporter applies ``sliding_window`` to such a layer.
    """
    if cfg.sliding_window_size <= 0:
        return -1
    labels = cfg.raw_layer_types
    if not labels:
        return cfg.sliding_window_size
    if (layer_index < len(labels)
            and labels[layer_index] == "sliding_attention"):
        return cfg.sliding_window_size
    return -1


class Eagle4DecoderLayer(Module):
    """Attends over ``[input_layernorm(embeds), hidden_norm(hidden)]``."""

    def __init__(self, ctx, prefix: str, layer_index: int) -> None:
        super().__init__(ctx, prefix)
        eps = ctx.cfg.rms_norm_eps
        self.hidden_norm = RMSNorm(ctx, self.key("hidden_norm"), eps)
        self.input_norm = RMSNorm(ctx, self.key("input_layernorm"), eps)
        self.post_norm = RMSNorm(ctx, self.key("post_attention_layernorm"),
                                 eps)
        self.attention = TreeAttention(
            ctx,
            self.key("self_attn"),
            sliding_window_size=_sliding_window_size(ctx.cfg, layer_index),
            attention_scale=ctx.cfg.attention_scaling)
        self.mlp = GatedMLP(ctx, self.key("mlp"))

    def forward(self, hidden, embeds, past, rope, ragged, attention_mask,
                attention_pos_id):
        attention_input = F.concatenate(
            (self.input_norm(embeds), self.hidden_norm(hidden)), 1)
        attention, present = self.attention(attention_input, past, rope,
                                            ragged, attention_mask,
                                            attention_pos_id)
        hidden = hidden + attention
        return hidden + self.mlp(self.post_norm(hidden)), present


class Eagle4DraftModel(NetworkModule):
    """EAGLE-4 draft model with runtime tree-attention metadata."""

    # EAGLE-4 drafts run in the EAGLE3 runtime mode with the same inputs.
    input_tensors = Eagle3DraftModel.input_tensors

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        cfg = ctx.cfg
        eps = cfg.rms_norm_eps
        self.fc_norm = [
            RMSNorm(ctx, f"fc_norm.{index}", eps)
            for index in range(len(cfg.eagle3_target_layer_ids))
        ] if cfg.eagle_fc_norm else []
        self.fc = Linear(ctx, "fc")
        self.layers = [
            Eagle4DecoderLayer(ctx, f"layers.{index}", index)
            for index in range(cfg.num_hidden_layers)
        ]
        self.norm = RMSNorm(ctx, "norm", eps)
        self.lm_head = (None if cfg.draft_candidate_count > 0 else Linear(
            ctx, "lm_head"))

    def _fuse_target_hidden(self, base_hidden):
        if self.fc_norm:
            width = self.cfg.target_hidden_size
            base_hidden = F.concatenate(
                tuple(
                    norm(base_hidden[..., index * width:(index + 1) * width])
                    for index, norm in enumerate(self.fc_norm)), 1)
        return self.fc(base_hidden)

    def _fp16_weight(self, name: str):
        parameter = self.weights.fp16_parameter(name)
        return F.parameter(parameter.name,
                           parameter.value,
                           "fp16",
                           recipe=parameter.recipe)

    def _head(self, normalized):
        cfg = self.cfg
        if self.lm_head is None:
            support_ids, support_log_probs = F.low_rank_candidate_head(
                normalized,
                self._fp16_weight(
                    _CANDIDATE_HEAD_WEIGHT["candidate_head.down_proj"]),
                self._fp16_weight(
                    _CANDIDATE_HEAD_WEIGHT["candidate_head.up_proj"]),
                self._fp16_weight("lm_head.weight"),
                num_candidates=cfg.draft_candidate_count)
            return {
                "spec_proposal_support_ids": support_ids,
                "spec_proposal_support_log_probs": support_log_probs,
            }
        logits = self.lm_head(normalized).cast(trt.float32)
        if cfg.final_logit_softcapping is not None:
            cap = np.float32(cfg.final_logit_softcapping)
            logits = (logits / cap).tanh() * cap
        return {"logits": logits.log_softmax(1)}

    def forward(self, **io):
        ragged = RaggedDecoderInputs.from_dict(io)
        hidden = self._fuse_target_hidden(
            io["base_hidden"]) + io["draft_hidden"]
        present = []
        for index, layer in enumerate(self.layers):
            hidden, cache = layer(hidden, io["inputs_embeds"],
                                  io["past_key_values"][index], io["rope"],
                                  ragged, io["attention_mask"],
                                  io["attention_pos_id"])
            present.append(cache)
        selected = F.gather_token_rows(hidden, ragged.logits_indices)
        normalized = self.norm(selected)
        outputs = self._head(normalized)
        outputs["hidden_states"] = (normalized if self.cfg.eagle_norm_output
                                    else selected)
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
