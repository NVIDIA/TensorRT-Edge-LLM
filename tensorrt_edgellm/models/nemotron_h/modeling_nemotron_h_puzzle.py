# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Heterogeneous Nemotron-H Puzzle decoder used by Nemotron 3.5 Bolt."""

from dataclasses import replace

import torch
import torch.nn as nn

from ...config import LAYER_MOE, ModelConfig
from ..linear import Q8_0Linear
from ..ops import Q8_0_OUT_DTYPES, q8_0_qkv_plugin, w2a16_moe_plugin
from .modeling_nemotron_h import (NemotronHAttentionMixer, NemotronHBackbone,
                                  NemotronHCausalLM, NemotronHDecoderLayer,
                                  NemotronHMoEMLP)


class NemotronHPuzzleMoE(NemotronHMoEMLP):
    """IQ2_XXS routed experts with the dedicated W2A16 plugin."""

    def prepare_for_export(self) -> None:
        if self._export_ready:
            return
        from ...checkpoint.iq2_xxs import lookup_table, pack_weights

        device = self.gate.weight.device

        def packed(proj):
            raw = torch.stack(
                [getattr(expert, proj).weight for expert in self.experts])
            return (torch.from_numpy(t).to(device)
                    for t in pack_weights(raw.cpu().numpy()))

        fc1, s1 = packed("up_proj")
        fc2, s2 = packed("down_proj")
        for name, tensor in (("fc1_qweights", fc1), ("fc1_blocks_scale", s1),
                             ("fc2_qweights", fc2), ("fc2_blocks_scale", s2)):
            self.register_buffer(name, tensor.contiguous())
        self.register_buffer(
            "unit_scales",
            torch.ones(self.n_routed_experts,
                       dtype=torch.float32,
                       device=device))
        self.register_buffer("lookup_table",
                             torch.from_numpy(lookup_table()).to(device))
        self.shared_experts.up_proj.prepare_for_export()
        self.shared_experts.down_proj.prepare_for_export()
        self.experts = nn.ModuleList()
        self._export_ready = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._export_ready:
            raise RuntimeError(
                "Nemotron-H Puzzle MoE requires prepare_for_export")
        flat = hidden_states.reshape(-1, self.hidden_size)
        ids, weights = self.gate(flat)
        # The Q8_0 latent projections read and write BF16 directly, so the
        # BF16 expert kernel needs no casts.
        routed_input = self.fc1_latent_proj(flat, out_dtype=torch.bfloat16)
        routed = w2a16_moe_plugin(routed_input, self.fc1_qweights,
                                  self.fc2_qweights, self.fc1_blocks_scale,
                                  self.fc2_blocks_scale, self.unit_scales,
                                  self.lookup_table, ids.to(torch.int32),
                                  weights.float(), self.n_routed_experts,
                                  self.moe_intermediate_size)
        routed = self.fc2_latent_proj(routed, out_dtype=flat.dtype)
        shared = self._expert_forward(self.shared_experts, flat)
        return (routed + shared).reshape(hidden_states.shape)


class NemotronHPuzzleAttentionMixer(NemotronHAttentionMixer):
    """Attention whose Q8_0 Q/K/V projections run as one packed plugin."""

    def project_qkv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projections = (self.q_proj, self.k_proj, self.v_proj)
        if not all(
                isinstance(p, Q8_0Linear) and p.bias is None
                for p in projections):
            return super().project_qkv(hidden_states)
        # One plugin quantizes the activations once and writes Q, K and V
        # side by side, so no concatenation of plugin outputs is needed.
        return q8_0_qkv_plugin(hidden_states,
                               *(p.packed_weight() for p in projections),
                               Q8_0_OUT_DTYPES[hidden_states.dtype])


class NemotronHPuzzleDecoderLayer(NemotronHDecoderLayer):
    attention_class = NemotronHPuzzleAttentionMixer

    def _make_moe(self, config: ModelConfig, module_prefix: str) -> nn.Module:
        return NemotronHPuzzleMoE(config, module_prefix)


class NemotronHPuzzleBackbone(NemotronHBackbone):

    def _make_layers(self, config: ModelConfig):
        if len(config.block_configs) != len(config.layer_types):
            raise ValueError(
                "Nemotron-H Puzzle block_config count does not match layers")
        layers = []
        for index, layer_type in enumerate(config.layer_types):
            block = config.block_configs[index]
            layer_config = replace(
                config, **block) if layer_type == LAYER_MOE else config
            layers.append(
                NemotronHPuzzleDecoderLayer(layer_config, index, layer_type))
        return layers


class NemotronHPuzzleCausalLM(NemotronHCausalLM):
    backbone_class = NemotronHPuzzleBackbone

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if isinstance(self.lm_head, Q8_0Linear):
            # Logits are FP32; the Q8_0 plugin writes them directly.
            self.lm_head.out_dtype = torch.float32
