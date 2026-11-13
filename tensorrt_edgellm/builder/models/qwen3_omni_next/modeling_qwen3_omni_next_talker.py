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
"""Checkpoint-direct Qwen3-Omni-Next dense Talker."""

import logging
from typing import Dict, Type

import tensorrt as trt

from ...core import config
from ...ops import (Linear, Module, NetworkModule, RaggedDecoderInputs,
                    RMSNorm, add_ragged_decoder_inputs)
from ...ops import functional as F
from .modeling_qwen3_omni_next_text import Qwen3OmniNextDecoderLayer

LOGGER = logging.getLogger("builder.qwen3_omni_next.talker")

__all__ = [
    "Qwen3OmniNextTalkerDecoderLayer",
    "Qwen3OmniNextTalkerModel",
    "Qwen3OmniNextTalker",
]


class Qwen3OmniNextTalkerDecoderLayer(Qwen3OmniNextDecoderLayer):
    """Talker-owned hybrid decoder layer."""


class Qwen3OmniNextTalkerModel(Module):
    """Autoregressive codec decoder with independent hybrid state streams."""

    layer_class: Type[
        Qwen3OmniNextTalkerDecoderLayer] = Qwen3OmniNextTalkerDecoderLayer

    def __init__(self, ctx, prefix: str = "model") -> None:
        super().__init__(ctx, prefix)
        self.layers = [
            self.layer_class(ctx, self.key(f"layers.{index}"), layer_type,
                             index)
            for index, layer_type in enumerate(ctx.cfg.layer_types)
        ]
        self.norm = RMSNorm(ctx,
                            self.key("norm"),
                            ctx.cfg.rms_norm_eps,
                            unit_offset=True)


class Qwen3OmniNextTalker(NetworkModule):
    """Dense Talker with codec logits and CodePredictor residual output."""

    model_class: Type[Qwen3OmniNextTalkerModel] = Qwen3OmniNextTalkerModel

    @classmethod
    def from_config(cls, ctx):
        if cls is Qwen3OmniNextTalker and ctx.cfg.num_experts > 0:
            from .modeling_qwen3_omni_next_moe_talker import \
                Qwen3OmniNextMoeTalker
            return Qwen3OmniNextMoeTalker(ctx)
        return cls(ctx)

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.model = self.model_class(ctx)
        self.codec_head = Linear(ctx, "codec_head")

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        if cfg.gdn_cfg is None:
            raise ValueError(
                "Qwen3-Omni-Next Talker requires Gated DeltaNet dimensions")
        gdn = cfg.gdn_cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        io = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, cfg.hidden_size)),
            "past_key_values": [
                self.add_input(f"past_key_values_{index}", kv_dtype,
                               (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                                cfg.head_dim))
                for index in range(cfg.num_attn_layers)
            ],
            "rope":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, cfg.rotary_dim)),
            "conv_states": [
                self.add_input(f"conv_state_{index}", trt.float16,
                               (-1, gdn.conv_dim, gdn.conv_kernel))
                for index in range(cfg.num_gdn_layers)
            ],
            "recurrent_states": [
                self.add_input(f"recurrent_state_{index}", trt.float32,
                               (-1, gdn.num_value_heads, gdn.key_head_dim,
                                gdn.value_head_dim))
                for index in range(cfg.num_gdn_layers)
            ],
        }
        io.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        return io

    def forward(self, **io):
        hidden_states = io["inputs_embeds"]
        ragged = RaggedDecoderInputs.from_dict(io)
        present_kv = []
        present_conv = []
        present_recurrent = []
        attention_index = 0
        state_index = 0
        for layer_index, (layer, layer_type) in enumerate(
                zip(self.model.layers, self.cfg.layer_types)):
            LOGGER.debug("building Talker layer %d/%d", layer_index + 1,
                         len(self.model.layers))
            if layer_type == config.LAYER_GDN:
                hidden_states, states = layer(
                    hidden_states,
                    ragged,
                    conv_state=io["conv_states"][state_index],
                    recurrent_state=io["recurrent_states"][state_index])
                present_conv.append(states[0])
                present_recurrent.append(states[1])
                state_index += 1
            else:
                hidden_states, present = layer(
                    hidden_states,
                    ragged,
                    past_key_value=io["past_key_values"][attention_index],
                    rope=io["rope"])
                present_kv.append(present)
                attention_index += 1

        hidden_states = self.model.norm(hidden_states)
        selected = F.gather_token_rows(hidden_states, ragged.logits_indices)
        outputs = {
            "logits": self.codec_head(selected).cast(trt.float32),
            "hidden_states": selected,
        }
        for index, tensor in enumerate(present_kv):
            outputs[f"present_key_values_{index}"] = tensor
        for index, tensor in enumerate(present_conv):
            outputs[f"present_conv_state_{index}"] = tensor
        for index, tensor in enumerate(present_recurrent):
            outputs[f"present_recurrent_state_{index}"] = tensor
        return outputs
