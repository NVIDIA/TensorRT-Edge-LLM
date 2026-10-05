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
"""Checkpoint-direct Qwen3-Omni-Next dense Thinker."""

import logging
from typing import Dict, Type

import tensorrt as trt

from ...core import config
from ...ops import (GatedDecoderAttention, GatedDeltaNet, GatedMLP, Linear,
                    Module, NetworkModule, RaggedDecoderInputs, RMSNorm,
                    add_ragged_decoder_inputs)
from ...ops import functional as F

LOGGER = logging.getLogger("builder.qwen3_omni_next.thinker")

__all__ = [
    "Qwen3OmniNextAttention",
    "Qwen3OmniNextDecoderLayer",
    "Qwen3OmniNextThinkerModel",
    "Qwen3OmniNextThinker",
]


class Qwen3OmniNextGatedDeltaNet(GatedDeltaNet):
    """Provider-fused Qwen3-Next linear-attention projections."""

    replicated_controls = False

    def _init_input_projections(self) -> None:
        self.in_proj_qkvz = Linear(self.ctx, self.key("in_proj_qkvz"))
        self.in_proj_ba = Linear(self.ctx, self.key("in_proj_ba"))

    def _project_inputs(self, hidden_states):
        gdn = self.cfg.gdn_cfg
        if gdn.num_value_heads % gdn.num_key_heads:
            raise ValueError(
                "Qwen3-Omni-Next value heads must divide into key heads")
        values_per_key = gdn.num_value_heads // gdn.num_key_heads
        value_group = values_per_key * gdn.value_head_dim
        qkvz_group = 2 * gdn.key_head_dim + 2 * value_group

        qkvz = self.in_proj_qkvz(hidden_states).reshape(
            (0, gdn.num_key_heads, qkvz_group))
        ba = self.in_proj_ba(hidden_states).reshape(
            (0, gdn.num_key_heads, 2 * values_per_key))

        offset = 0
        query = qkvz[:, :, offset:offset + gdn.key_head_dim]
        offset += gdn.key_head_dim
        key = qkvz[:, :, offset:offset + gdn.key_head_dim]
        offset += gdn.key_head_dim
        value = qkvz[:, :, offset:offset + value_group]
        offset += value_group
        gate = qkvz[:, :, offset:offset + value_group]
        beta = ba[:, :, :values_per_key]
        alpha = ba[:, :, values_per_key:]

        mixed = F.concatenate((query.reshape(
            (0, gdn.key_dim)), key.reshape(
                (0, gdn.key_dim)), value.reshape((0, gdn.value_dim))), 1)
        return (mixed, gate.reshape(
            (0, gdn.value_dim)), beta.reshape((0, gdn.num_value_heads)),
                alpha.reshape((0, gdn.num_value_heads)))


class Qwen3OmniNextAttention(GatedDecoderAttention):
    """Next gated full attention with unit-offset Q/K normalization."""


class Qwen3OmniNextDecoderLayer(Module):
    """One dense Gated DeltaNet or gated full-attention Thinker block."""

    mlp_class = GatedMLP

    def __init__(self, ctx, prefix: str, layer_type: str,
                 layer_index: int) -> None:
        del layer_index
        super().__init__(ctx, prefix)
        self.layer_type = layer_type
        self.input_layernorm = RMSNorm(ctx,
                                       self.key("input_layernorm"),
                                       ctx.cfg.rms_norm_eps,
                                       unit_offset=True)
        self.post_attention_layernorm = RMSNorm(
            ctx,
            self.key("post_attention_layernorm"),
            ctx.cfg.rms_norm_eps,
            unit_offset=True)
        self.mlp = self.mlp_class(ctx, self.key("mlp"))
        if layer_type == config.LAYER_GDN:
            self.mixer = Qwen3OmniNextGatedDeltaNet(ctx,
                                                    self.key("linear_attn"))
        elif layer_type == config.LAYER_ATTN:
            self.mixer = Qwen3OmniNextAttention(ctx, self.key("self_attn"))
        else:
            raise ValueError(
                f"unsupported Qwen3-Omni-Next layer type {layer_type!r}")

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
        normalized = self.input_layernorm(hidden_states)
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
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class Qwen3OmniNextThinkerModel(Module):
    """Hybrid decoder stack with model-owned Talker hidden-state selection."""

    layer_class: Type[Qwen3OmniNextDecoderLayer] = Qwen3OmniNextDecoderLayer

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

    def emitted_hidden(self, normed_hidden, all_hidden):
        """Return the exact full-sequence tensor consumed by the Talker."""
        if self.cfg.mtp_base:
            return normed_hidden
        accepted = self.cfg.accept_hidden_layer
        if 1 <= accepted <= len(all_hidden):
            return all_hidden[accepted - 1]
        return normed_hidden


class Qwen3OmniNextThinker(NetworkModule):
    """Dense Next Thinker and its hybrid runtime I/O contract."""

    model_class: Type[Qwen3OmniNextThinkerModel] = Qwen3OmniNextThinkerModel

    @classmethod
    def from_config(cls, ctx):
        if cls is Qwen3OmniNextThinker and ctx.cfg.num_experts > 0:
            from .modeling_qwen3_omni_next_moe_text import \
                Qwen3OmniNextMoeThinker
            return Qwen3OmniNextMoeThinker(ctx)
        return cls(ctx)

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.model = self.model_class(ctx)
        self.lm_head = Linear(ctx, ctx.weights.causal_lm_head_prefix())

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        if cfg.gdn_cfg is None:
            raise ValueError(
                "Qwen3-Omni-Next Thinker requires Gated DeltaNet dimensions")
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
            "deepstack_embeds": [
                self.add_input(f"deepstack_embeds_{index}", trt.float16,
                               (-1, cfg.hidden_size))
                for index in range(cfg.num_deepstack_features)
            ],
        }
        io.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        if cfg.engine_role == "base":
            io["attention_pos_id"] = self.add_input("attention_position_ids",
                                                    trt.int32, (-1, ))
            io["attention_mask"] = self.add_input("packed_attention_mask",
                                                  trt.int32, (-1, -1))
            if cfg.dflash_tree_base or cfg.mtp_tree_base:
                io["tree_parent_ids"] = self.add_input("tree_parent_ids",
                                                       trt.int32, (-1, ))
                io["tree_depths"] = self.add_input("tree_depths", trt.int32,
                                                   (-1, ))
                io["valid_tree_counts"] = self.add_input(
                    "valid_tree_counts", trt.int32, (-1, ))
            else:
                io["tree_parent_ids"] = None
                io["tree_depths"] = None
                io["valid_tree_counts"] = None
        else:
            io.update({
                "attention_pos_id": None,
                "attention_mask": None,
                "tree_parent_ids": None,
                "tree_depths": None,
                "valid_tree_counts": None,
            })
        return io

    def forward(self, **io):
        hidden_states = io["inputs_embeds"]
        ragged = RaggedDecoderInputs.from_dict(io)
        present_kv = []
        present_conv = []
        present_recurrent = []
        intermediate_conv = []
        intermediate_recurrent = []
        all_hidden = []
        attention_index = 0
        state_index = 0

        for layer_index, (layer, layer_type) in enumerate(
                zip(self.model.layers, self.cfg.layer_types)):
            LOGGER.debug("building Thinker layer %d/%d", layer_index + 1,
                         len(self.model.layers))
            if layer_type == config.LAYER_GDN:
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
            else:
                hidden_states, present = layer(
                    hidden_states,
                    ragged,
                    past_key_value=io["past_key_values"][attention_index],
                    rope=io["rope"],
                    attention_mask=io["attention_mask"],
                    attention_pos_id=io["attention_pos_id"],
                    tree_parent_ids=io["tree_parent_ids"],
                    tree_depths=io["tree_depths"],
                    valid_tree_counts=io["valid_tree_counts"])
                present_kv.append(present)
                attention_index += 1
            if layer_index < len(io["deepstack_embeds"]):
                hidden_states = hidden_states + io["deepstack_embeds"][
                    layer_index]
            all_hidden.append(hidden_states)

        normed_hidden = self.model.norm(hidden_states)
        selected = F.gather_token_rows(normed_hidden, ragged.logits_indices)
        outputs = {
            "logits": self.lm_head(selected).cast(trt.float32),
            "hidden_states": self.model.emitted_hidden(normed_hidden,
                                                       all_hidden),
        }
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
