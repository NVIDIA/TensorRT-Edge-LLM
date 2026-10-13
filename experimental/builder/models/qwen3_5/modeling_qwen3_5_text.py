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
"""Qwen3.5 checkpoint-direct graph."""

import logging
from typing import Dict

import tensorrt as trt

from ...core import config
from ...ops import (GatedDecoderAttention, GatedDeltaNet, GatedMLP, Linear,
                    Module, NetworkModule, RaggedDecoderInputs, RMSNorm,
                    add_ragged_decoder_inputs)
from ...ops import functional as F

LOGGER = logging.getLogger("builder.qwen3_5")


class Qwen3_5Attention(GatedDecoderAttention):
    """Qwen3.5 gated attention extension point."""


class Qwen3_5DecoderLayer(Module):
    """One Qwen3.5 Gated DeltaNet or full-attention block."""

    def __init__(self, ctx, prefix: str, layer_type: str,
                 layer_index: int) -> None:
        super().__init__(ctx, prefix)
        self.layer_type = layer_type
        self.input_norm = RMSNorm(ctx,
                                  self.key("input_layernorm"),
                                  ctx.cfg.rms_norm_eps,
                                  unit_offset=True)
        self.post_norm = RMSNorm(ctx,
                                 self.key("post_attention_layernorm"),
                                 ctx.cfg.rms_norm_eps,
                                 unit_offset=True)
        self.mlp = GatedMLP(ctx, self.key("mlp"))
        if layer_type == config.LAYER_GDN:
            self.mixer = GatedDeltaNet(ctx, self.key("linear_attn"))
        elif layer_type == config.LAYER_ATTN:
            self.mixer = Qwen3_5Attention(ctx, self.key("self_attn"))
        else:
            raise ValueError(f"unsupported Qwen3.5 layer type {layer_type!r}")

    def forward(self,
                hidden_states,
                ragged,
                past_key_value=None,
                rope=None,
                conv_state=None,
                recurrent_state=None,
                attention_mask=None,
                attention_pos_id=None,
                tree_parent_ids=None,
                tree_depths=None,
                valid_tree_counts=None,
                collect_intermediate=False):
        normalized = self.input_norm(hidden_states)
        if self.layer_type == config.LAYER_GDN:
            states = self.mixer(normalized, conv_state, recurrent_state,
                                ragged, tree_parent_ids, tree_depths,
                                collect_intermediate)
            mixed, conv_out, recurrent_out = states[:3]
            present = (conv_out, recurrent_out, *states[3:])
        else:
            mixed, present = self.mixer(normalized, past_key_value, rope,
                                        ragged, attention_mask,
                                        attention_pos_id, tree_parent_ids,
                                        tree_depths, valid_tree_counts)
        hidden_states = hidden_states + mixed
        feed_forward = self.mlp(self.post_norm(hidden_states))
        hidden_states = hidden_states + feed_forward
        return hidden_states, present


class Qwen3_5ForCausalLM(NetworkModule):
    """Top-level Qwen3.5 model with compact attention and recurrent-state indexing."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.qwen35 = True
        self.layers = [
            Qwen3_5DecoderLayer(ctx, f"model.layers.{index}", layer_type,
                                index)
            for index, layer_type in enumerate(ctx.cfg.layer_types)
        ]
        self.norm = RMSNorm(ctx,
                            "model.norm",
                            ctx.cfg.rms_norm_eps,
                            unit_offset=True)
        self.lm_head = Linear(ctx, ctx.weights.causal_lm_head_prefix())

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        recurrent = cfg.gdn_cfg or cfg.mamba_cfg
        if cfg.gdn_cfg is not None:
            recurrent_shape = (-1, recurrent.num_value_heads,
                               recurrent.key_head_dim,
                               recurrent.value_head_dim)
            conv_dim = recurrent.conv_dim
            conv_kernel = recurrent.conv_kernel
            state_count = cfg.num_gdn_layers
        else:
            recurrent_shape = (-1, recurrent.num_heads, recurrent.head_dim,
                               recurrent.ssm_state_size)
            conv_dim = recurrent.conv_dim
            conv_kernel = recurrent.conv_kernel
            state_count = cfg.num_mamba_layers
        result = {
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
                               (-1, conv_dim, conv_kernel))
                for index in range(state_count)
            ],
            "recurrent_states": [
                self.add_input(f"recurrent_state_{index}",
                               trt.float32 if self.qwen35 else trt.float16,
                               recurrent_shape) for index in range(state_count)
            ],
        }
        result.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        if cfg.engine_role == "base":
            result["attention_pos_id"] = self.add_input(
                "attention_position_ids", trt.int32, (-1, ))
            result["attention_mask"] = self.add_input("packed_attention_mask",
                                                      trt.int32, (-1, -1))
            if cfg.dflash_tree_base or cfg.mtp_tree_base:
                result["tree_parent_ids"] = self.add_input(
                    "tree_parent_ids", trt.int32, (-1, ))
                result["tree_depths"] = self.add_input("tree_depths",
                                                       trt.int32, (-1, ))
                result["valid_tree_counts"] = self.add_input(
                    "valid_tree_counts", trt.int32, (-1, ))
            else:
                result["tree_parent_ids"] = None
                result["tree_depths"] = None
                result["valid_tree_counts"] = None
        else:
            result.update({
                "attention_pos_id": None,
                "attention_mask": None,
                "tree_parent_ids": None,
                "tree_depths": None,
                "valid_tree_counts": None,
            })
        return result

    def forward(self, **io):
        outputs = {}
        hidden_states = io["inputs_embeds"]
        ragged = RaggedDecoderInputs.from_dict(io)
        present_kv = []
        present_conv = []
        present_recurrent = []
        intermediate_conv = []
        intermediate_recurrent = []
        all_hidden_states = []
        attention_index = 0
        state_index = 0
        for layer_index, (layer, layer_type) in enumerate(
                zip(self.layers, self.cfg.layer_types)):
            LOGGER.info("building layer %d/%d", layer_index + 1,
                        len(self.layers))
            if layer_type in (config.LAYER_MAMBA, config.LAYER_GDN):
                hidden_states, states = layer(
                    hidden_states,
                    ragged,
                    conv_state=io["conv_states"][state_index],
                    recurrent_state=io["recurrent_states"][state_index],
                    tree_parent_ids=io["tree_parent_ids"],
                    tree_depths=io["tree_depths"],
                    valid_tree_counts=io["valid_tree_counts"],
                    collect_intermediate=self.cfg.engine_role == "base")
                present_conv.append(states[0])
                present_recurrent.append(states[1])
                if len(states) > 2 and states[2] is not None:
                    intermediate_conv.append(states[2])
                if len(states) > 3 and states[3] is not None:
                    intermediate_recurrent.append(states[3])
                state_index += 1
            elif layer_type == config.LAYER_ATTN:
                hidden_states, present = layer(
                    hidden_states,
                    ragged,
                    io["past_key_values"][attention_index],
                    io["rope"],
                    attention_mask=io["attention_mask"],
                    attention_pos_id=io["attention_pos_id"],
                    tree_parent_ids=io["tree_parent_ids"],
                    tree_depths=io["tree_depths"],
                    valid_tree_counts=io["valid_tree_counts"])
                present_kv.append(present)
                attention_index += 1
            else:
                hidden_states, _ = layer(hidden_states, ragged)
            all_hidden_states.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        selected = F.gather_token_rows(hidden_states, ragged.logits_indices)
        logits = F.cast(self.lm_head(selected), trt.float32)
        outputs["logits"] = logits
        if self.cfg.engine_role == "base":
            outputs["hidden_states"] = F.hidden_state_feedback(
                hidden_states, all_hidden_states, self.cfg, allow_eagle3=False)
        for index, tensor in enumerate(present_kv):
            outputs[f"present_key_values_{index}"] = tensor
        for index, tensor in enumerate(present_conv):
            outputs[f"present_conv_state_{index}"] = tensor
        for index, tensor in enumerate(present_recurrent):
            outputs[f"present_recurrent_state_{index}"] = tensor
        for index, tensor in enumerate(intermediate_conv):
            outputs[f"intermediate_conv_state_{index}"] = tensor
        for index, tensor in enumerate(intermediate_recurrent):
            outputs[f"intermediate_recurrent_state_{index}"] = tensor
        return outputs
