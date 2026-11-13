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
"""Checkpoint-direct Qwen3-Omni-Next CodePredictor."""

from typing import Dict

import tensorrt as trt

from ...ops import (DynamicLinear, FP32GatedMLP, GatedDecoderAttention, Module,
                    NetworkModule, RMSNorm)
from ...ops import functional as F
from ...ops.ragged import RaggedDecoderInputs, add_ragged_decoder_inputs

__all__ = [
    "Qwen3OmniNextCodePredictorAttention",
    "Qwen3OmniNextCodePredictorDecoderLayer",
    "Qwen3OmniNextCodePredictorModel",
    "Qwen3OmniNextCodePredictor",
]


class Qwen3OmniNextCodePredictorMLP(FP32GatedMLP):
    """Qwen3-Omni-Next CodePredictor precision-preserving MLP."""


class Qwen3OmniNextCodePredictorAttention(GatedDecoderAttention):
    """CodePredictor gated full attention."""


class Qwen3OmniNextCodePredictorDecoderLayer(Module):
    """One dense full-attention CodePredictor layer."""

    def __init__(self, ctx, prefix: str, layer_index: int) -> None:
        del layer_index
        super().__init__(ctx, prefix)
        eps = ctx.cfg.rms_norm_eps
        self.input_layernorm = RMSNorm(ctx,
                                       self.key("input_layernorm"),
                                       eps,
                                       unit_offset=True)
        self.post_attention_layernorm = RMSNorm(
            ctx, self.key("post_attention_layernorm"), eps, unit_offset=True)
        self.self_attn = Qwen3OmniNextCodePredictorAttention(
            ctx, self.key("self_attn"))
        self.mlp = Qwen3OmniNextCodePredictorMLP(ctx, self.key("mlp"))

    def forward(self, hidden_states, past_key_value, rope, ragged):
        attention, present = self.self_attn(
            self.input_layernorm(hidden_states), past_key_value, rope, ragged)
        hidden_states = hidden_states + attention
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class Qwen3OmniNextCodePredictorModel(Module):
    """Five-layer dense predictor stack from the provider checkpoint."""

    def __init__(self, ctx, prefix: str = "model") -> None:
        super().__init__(ctx, prefix)
        self.layers = [
            Qwen3OmniNextCodePredictorDecoderLayer(ctx,
                                                   self.key(f"layers.{index}"),
                                                   index)
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = RMSNorm(ctx,
                            self.key("norm"),
                            ctx.cfg.rms_norm_eps,
                            unit_offset=True)

    def forward(self, hidden_states, past_key_values, rope, ragged):
        present = []
        for index, layer in enumerate(self.layers):
            hidden_states, cache = layer(hidden_states, past_key_values[index],
                                         rope, ragged)
            present.append(cache)
        return self.norm(hidden_states), present


class Qwen3OmniNextCodePredictor(NetworkModule):
    """CodePredictor with device-selected stacked codec heads."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.model = Qwen3OmniNextCodePredictorModel(ctx)
        self.dynamic_head = DynamicLinear(ctx, ctx.cfg.hidden_size)

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        num_heads = int(
            cfg.raw_component.get("num_code_groups",
                                  (cfg.raw_root.get("talker_config")
                                   or {}).get("num_code_groups", 16))) - 1
        if num_heads < 1:
            raise ValueError(
                "Qwen3-Omni-Next CodePredictor requires num_code_groups > 1")
        io = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, cfg.hidden_size)),
            "past_key_values": [
                self.add_input(f"past_key_values_{index}", kv_dtype,
                               (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                                cfg.head_dim))
                for index in range(cfg.num_hidden_layers)
            ],
            "rope":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, cfg.rotary_dim)),
            "lm_heads":
            self.add_input("lm_heads", trt.float16,
                           (num_heads, cfg.vocab_size, cfg.hidden_size)),
            "lm_head_idx":
            self.add_input("lm_head_idx", trt.int32, (1, )),
        }
        io.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        return io

    def forward(self, **io):
        ragged = RaggedDecoderInputs.from_dict(io)
        hidden_states, present = self.model(io["inputs_embeds"],
                                            io["past_key_values"], io["rope"],
                                            ragged)
        selected = F.gather_token_rows(hidden_states, ragged.logits_indices)
        head = io["lm_heads"].gather(io["lm_head_idx"].cast(trt.int64),
                                     0).reshape((self.cfg.vocab_size,
                                                 self.cfg.hidden_size))
        outputs = {
            "logits": self.dynamic_head(selected, head).cast(trt.float32),
            "hidden_states": hidden_states,
        }
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
