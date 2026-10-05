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
"""Nemotron Omni checkpoint-direct text graph."""

import logging
from typing import Dict

import numpy as np
import tensorrt as trt

from ...core import config
from ...ops import (DecoderAttention, GroupedSigmoidRouter, Linear, Module,
                    NetworkModule, NonGatedNvfp4Experts, RaggedDecoderInputs,
                    RMSNorm, add_ragged_decoder_inputs)
from ...ops import functional as F
from . import weights as weight_conversion

LOGGER = logging.getLogger("builder.nemotron_omni.text")


class NemotronOmniAttention(DecoderAttention):
    """Nemotron Omni text-attention extension point."""


class NemotronOmniMamba2Mixer(Module):
    """Mamba2 projection, stateful convolution, selective scan, and gate."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.in_proj = Linear(ctx, self.key("in_proj"))
        self.out_proj = Linear(ctx, self.key("out_proj"))

    def forward(self,
                hidden_states,
                conv_state,
                recurrent_state,
                ragged,
                tree_parent_ids=None,
                tree_depths=None,
                collect_intermediate=False):
        mamba = self.cfg.mamba_cfg
        d_inner = mamba.intermediate_size
        d_state = mamba.n_groups * mamba.ssm_state_size
        projected = self.in_proj(hidden_states)
        gate = F.slice_last_dim(projected, 0, d_inner, 2)
        conv_input = F.slice_last_dim(projected, d_inner, mamba.conv_dim, 2)
        dt = F.slice_last_dim(projected, d_inner + mamba.conv_dim,
                              mamba.num_heads, 2)
        conv_weight = F.constant(self.weights.f16(self.key("conv1d.weight")),
                                 "conv_weight")
        conv_bias_data = self.weights.opt_f16(self.key("conv1d.bias"))
        if conv_bias_data is None:
            conv_bias_data = np.zeros(mamba.conv_dim, dtype=np.float16)
        conv_bias = F.constant(conv_bias_data, "conv_bias")
        conv_output, conv_state_out, intermediate_conv = F.causal_conv1d(
            conv_input,
            conv_weight,
            conv_bias,
            conv_state,
            ragged,
            mamba.conv_dim,
            mamba.conv_kernel - 1,
            tree_parent_ids=tree_parent_ids,
            tree_depths=tree_depths,
            use_intermediate=collect_intermediate)
        conv_output = conv_output.activation(self.cfg.mamba_hidden_act)
        x = conv_output.slice_last_dim(0, d_inner, 2).reshape(
            (0, mamba.num_heads, mamba.head_dim))
        b = conv_output.slice_last_dim(d_inner, d_state, 2).reshape(
            (0, mamba.n_groups, mamba.ssm_state_size))
        c = conv_output.slice_last_dim(d_inner + d_state, d_state, 2).reshape(
            (0, mamba.n_groups, mamba.ssm_state_size))
        a_log = self.weights.f32(self.key("A_log"))
        a = F.constant(-np.exp(a_log).astype(np.float32), "ssm_A")
        d = F.constant(self.weights.f16(self.key("D")), "ssm_D")
        dt_bias = F.constant(self.weights.f16(self.key("dt_bias")), "dt_bias")
        state_outputs = F.update_ssm_state(
            x,
            a,
            b,
            c,
            d,
            dt,
            dt_bias,
            recurrent_state,
            ragged,
            mamba.head_dim,
            mamba.ssm_state_size,
            mamba.num_heads,
            mamba.n_groups,
            use_intermediate=collect_intermediate)
        output, recurrent_state_out = state_outputs[:2]
        replay = state_outputs[2:] if collect_intermediate else (None, ) * 4
        output = output.reshape((0, d_inner))
        gated = (output * gate.silu()).cast(trt.float32)
        group_size = d_inner // mamba.n_groups
        grouped = gated.reshape((0, mamba.n_groups, group_size))
        normalized = F.rms_norm(grouped,
                                np.ones(group_size, dtype=np.float32),
                                self.cfg.rms_norm_eps,
                                rank=3)
        normalized = normalized.cast(trt.float16).reshape((0, d_inner))
        norm_weight = F.constant(
            self.weights.f16(self.key("norm.weight")).reshape(1, -1),
            "mamba_norm")
        normalized = normalized * norm_weight
        return (self.out_proj(normalized), conv_state_out, recurrent_state_out,
                intermediate_conv, *replay)


class NemotronOmniMLP(Module):
    """Nemotron Omni non-gated feed-forward layer."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.up_proj = Linear(ctx, self.key("up_proj"))
        self.down_proj = Linear(ctx, self.key("down_proj"))

    def forward(self, hidden_states):
        activated = self.up_proj(hidden_states).activation(self.cfg.hidden_act)
        return self.down_proj(activated)


class NemotronOmniMoE(Module):
    """Nemotron Omni grouped NVFP4 MoE plus its shared expert."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.latent_in = (Linear(ctx, self.key("fc1_latent_proj"))
                          if ctx.cfg.moe_latent_size is not None else None)
        self.latent_out = (Linear(ctx, self.key("fc2_latent_proj"))
                           if ctx.cfg.moe_latent_size is not None else None)
        self.gate = GroupedSigmoidRouter(ctx, self.key("gate"))
        self.experts = NonGatedNvfp4Experts(
            ctx, self.key("experts"), weight_conversion.repack_nvfp4_experts)
        self.shared_experts = NemotronOmniMLP(ctx, self.key("shared_experts"))

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        routed = (self.latent_in(hidden_states)
                  if self.latent_in is not None else hidden_states)
        routed = self.experts(routed, router_logits, self.gate.correction,
                              self.gate.correction_key)
        if self.latent_out is not None:
            routed = self.latent_out(routed)
        shared = self.shared_experts(hidden_states)
        return routed + shared


class NemotronOmniBlock(Module):
    """One Nemotron Omni recurrent, attention, dense, or MoE block."""

    def __init__(self, ctx, prefix: str, layer_type: str,
                 layer_index: int) -> None:
        super().__init__(ctx, prefix)
        self.layer_type = layer_type
        self.input_norm = RMSNorm(ctx, self.key("norm"), ctx.cfg.rms_norm_eps)
        if layer_type == config.LAYER_MAMBA:
            self.mixer = NemotronOmniMamba2Mixer(ctx, self.key("mixer"))
        elif layer_type == config.LAYER_ATTN:
            self.mixer = NemotronOmniAttention(ctx, self.key("mixer"))
        elif layer_type == config.LAYER_MLP:
            self.mixer = NemotronOmniMLP(ctx, self.key("mixer"))
        elif layer_type == config.LAYER_MOE:
            self.mixer = NemotronOmniMoE(ctx, self.key("mixer"))
        else:
            raise ValueError(
                f"unsupported Nemotron Omni layer type {layer_type!r}")
        _ = layer_index

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
        if self.layer_type == config.LAYER_MAMBA:
            states = self.mixer(normalized, conv_state, recurrent_state,
                                ragged, tree_parent_ids, tree_depths,
                                collect_intermediate)
            mixed, conv_out, recurrent_out = states[:3]
            present = (conv_out, recurrent_out, *states[3:])
        elif self.layer_type == config.LAYER_ATTN:
            mixed, present = self.mixer(normalized, past_key_value, rope,
                                        ragged, attention_mask,
                                        attention_pos_id, tree_parent_ids,
                                        tree_depths, valid_tree_counts)
        else:
            mixed = self.mixer(normalized)
            present = None
        hidden_states = hidden_states + mixed
        return hidden_states, present


class NemotronOmniCausalLM(NetworkModule):
    """Top-level Nemotron Omni text model with compact cache indexing."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.qwen35 = False
        self.layers = [
            NemotronOmniBlock(ctx, f"backbone.layers.{index}", layer_type,
                              index)
            for index, layer_type in enumerate(ctx.cfg.layer_types)
        ]
        self.norm = RMSNorm(ctx, "backbone.norm_f", ctx.cfg.rms_norm_eps)
        self.lm_head = Linear(ctx, "lm_head")

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
        replay_da_states = []
        replay_u_states = []
        replay_b_states = []
        replay_dt_states = []
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
                    collect_intermediate=self.cfg.engine_role == "base")
                present_conv.append(states[0])
                present_recurrent.append(states[1])
                if len(states) > 2 and states[2] is not None:
                    intermediate_conv.append(states[2])
                if len(states) > 6 and states[3] is not None:
                    replay_da_states.append(states[3])
                    replay_u_states.append(states[4])
                    replay_b_states.append(states[5])
                    replay_dt_states.append(states[6])
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
        for index, tensor in enumerate(replay_da_states):
            outputs[f"replay_da_state_{index}"] = tensor
        for index, tensor in enumerate(replay_u_states):
            outputs[f"replay_u_state_{index}"] = tensor
        for index, tensor in enumerate(replay_b_states):
            outputs[f"replay_b_state_{index}"] = tensor
        for index, tensor in enumerate(replay_dt_states):
            outputs[f"replay_dt_state_{index}"] = tensor
        return outputs
