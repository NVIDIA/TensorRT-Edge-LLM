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
"""Qwen4-Exp MTP checkpoint-direct draft graph.

The draft carries its own copy of the multi-stream residual: the base hands it
the stack's state from *before* the final mixer, ``hc_count * hidden_size``
wide, and the draft returns the same thing for its next step. The token
embedding enters that bus as a block output written to every stream with unit
weight, which is why the two ``pre_fc`` norms are not symmetric --
``pre_fc_norm_embedding`` normalizes one stream and ``pre_fc_norm_hidden``
normalizes all four at once.
"""

from typing import Dict

import tensorrt as trt

from ...core import config
from ...ops import Linear, NetworkModule, RaggedDecoderInputs, RMSNorm
from ...ops import functional as F
from ...ops.ragged import add_ragged_decoder_inputs
from .modeling_qwen4_exp import Qwen4ExpDecoderLayer, speculative_inputs
from .modeling_qwen4_exp_residual import Qwen4ExpGatedResidual

__all__ = ["Qwen4ExpMtpDraftModel"]


class Qwen4ExpMtpDraftModel(NetworkModule):
    """Qwen4-Exp MTP draft model.

    Module names are the checkpoint's ``mtp.*`` paths rather than the base
    model's: the quantization metadata keys the draft's expert bank as
    ``mtp.layers.0.mlp.experts``, and a bare ``layers.0.mlp.experts`` would
    resolve to the base model's NVFP4 entry instead.
    """

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        cfg = ctx.cfg
        eps = cfg.rms_norm_eps
        self.pre_fc_norm_embedding = RMSNorm(ctx,
                                             "mtp.pre_fc_norm_embedding",
                                             eps,
                                             unit_offset=True)
        self.pre_fc_norm_hidden = RMSNorm(ctx,
                                          "mtp.pre_fc_norm_hidden",
                                          eps,
                                          unit_offset=True)
        self.fc_embedding = Linear(ctx, "mtp.fc_embedding")
        self.fc_hidden = Linear(ctx, "mtp.fc_hidden")
        self.layers = [
            Qwen4ExpDecoderLayer(ctx,
                                 f"mtp.layers.{index}",
                                 config.LAYER_ATTN,
                                 has_ple=False)
            for index in range(cfg.num_hidden_layers)
        ]
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(
            ctx, "mtp.hyper_connection_mixer", use_combine=False)
        self.lm_head = Linear(ctx,
                              ctx.weights.causal_lm_head_prefix("mtp.lm_head"))

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        io = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, cfg.hidden_size)),
            "past_key_values": [
                self.add_input(f"past_key_values_{index}", kv_dtype,
                               (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                                cfg.qsa_pool_head_dim))
                for index in range(cfg.num_hidden_layers)
            ],
            "rope":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, cfg.rotary_dim)),
            # Both feedback bindings carry the multi-stream bus, so they are
            # hc_count times wider than the embedding beside them.
            "base_hidden":
            self.add_input("hidden_states_input", trt.float16,
                           (-1, cfg.hc_hidden_size)),
            "draft_hidden":
            self.add_input("hidden_states_from_draft", trt.float16,
                           (-1, cfg.hc_hidden_size)),
        }
        io.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        io.update(speculative_inputs(self.add_input, cfg.engine_role))
        return io

    def forward(self, **io):
        cfg = self.cfg
        ragged = RaggedDecoderInputs.from_dict(io)
        embeds = self.fc_embedding(
            self.pre_fc_norm_embedding(io["inputs_embeds"]))
        # The runtime zeroes whichever feedback is not in play this step.
        hyper_states = io["base_hidden"] + io["draft_hidden"]
        hyper_states = self.pre_fc_norm_hidden(hyper_states)
        # One fc_hidden shared by every stream, applied stream by stream.
        hyper_states = self.fc_hidden(
            hyper_states.reshape((-1, cfg.hidden_size))).reshape(
                (-1, cfg.hc_hidden_size))
        hyper_states = hyper_states + F.concatenate([embeds] * cfg.hc_count, 1)

        present_kv = []
        for index, layer in enumerate(self.layers):
            hyper_states, present, _ = layer(
                hyper_states,
                ragged,
                past_key_value=io["past_key_values"][index],
                rope=io["rope"],
                tree_parent_ids=io["tree_parent_ids"],
                tree_depths=io["tree_depths"])
            present_kv.append(present[0])

        selected = F.gather_token_rows(hyper_states, ragged.logits_indices)
        mixed = self.hyper_connection_mixer(selected)
        outputs = {
            "logits": F.cast(self.lm_head(mixed), trt.float32).log_softmax(1),
            "hidden_states": selected,
        }
        for index, tensor in enumerate(present_kv):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
