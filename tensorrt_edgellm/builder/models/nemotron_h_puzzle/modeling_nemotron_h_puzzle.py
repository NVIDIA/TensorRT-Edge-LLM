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
"""Checkpoint-direct Nemotron-H Puzzle decoder with IQ2_XXS experts."""

from dataclasses import replace

import numpy as np
import tensorrt as trt

from ....checkpoint.iq2_xxs import lookup_table, pack_weights
from ...core import config, quantization
from ...core.weights import ParameterSpec
from ...ops import GroupedSigmoidRouter, Linear, Module
from ...ops import functional as F
from ..nemotron_h.modeling_nemotron_h import (NemotronHAttention,
                                              NemotronHBlock,
                                              NemotronHForCausalLM,
                                              NemotronHMLP)


class NemotronHPuzzleMoE(Module):
    """One heterogeneous MoE layer with packed IQ2_XXS routed experts."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        if ctx.options.trt_native_moe:
            raise ValueError("IQ2_XXS experts require the W2A16 MoE plugin")
        if ctx.cfg.n_group != 1 or ctx.cfg.topk_group != 1:
            raise ValueError("Puzzle direct routing requires one expert group")
        self.latent_in = Linear(ctx, self.key("fc1_latent_proj"))
        self.latent_out = Linear(ctx, self.key("fc2_latent_proj"))
        self.gate = GroupedSigmoidRouter(ctx, self.key("gate"))
        self.shared_experts = NemotronHMLP(ctx, self.key("shared_experts"))

    def _experts(self) -> tuple[dict, dict]:
        """Describe the packed expert tensors consumed by W2A16MoePlugin.

        Externalized experts stay in the checkpoint and are reordered by the
        runtime; otherwise they are reordered here. Lookup tables and unit
        scales are small engine constants.
        """
        cfg = self.cfg
        prefix = self.key("experts")
        buffers = {
            "unit_scales": np.ones(cfg.num_experts, dtype=np.float32),
            "lookup_table": lookup_table(),
        }
        recipes = {}
        for index, projection in (("1", "up_proj"), ("2", "down_proj")):
            names = [
                f"{prefix}.{expert}.{projection}.weight"
                for expert in range(cfg.num_experts)
            ]
            rows, blocks, _ = self.weights.parameter_spec(names[0],
                                                          np.uint8).shape
            # Payload: 16 int32 words per 256-weight block; bases: one FP16
            # per block, two per int32 word.
            words = cfg.num_experts * rows * blocks * 16
            payload, bases = self.weights.parameter_value(
                quantization.QUANT_IQ2_XXS, prefix, lambda:
                (ParameterSpec((words, ), np.dtype(np.int32)),
                 ParameterSpec((words // 32, ), np.dtype(np.int32))),
                lambda: pack_weights(
                    np.stack([self.weights.array(name) for name in names])))
            buffers[f"fc{index}_qweights"] = payload
            buffers[f"fc{index}_blocks_scale"] = bases
            recipes[f"fc{index}_qweights"] = self.weights.checkpoint_binding(
                names, assemble="iq2_xxs_moe_qweight", num_experts=len(names))
            recipes[f"fc{index}_blocks_scale"] = (
                self.weights.checkpoint_binding(names,
                                                assemble="iq2_xxs_moe_bases",
                                                num_experts=len(names)))
        return buffers, recipes

    def forward(self, hidden_states):
        cfg = self.cfg
        scores = self.gate(hidden_states).sigmoid()
        correction = F.constant(self.gate.correction.reshape(1, -1),
                                self.gate.correction_key)
        _, ids = F.topk(scores + correction, cfg.num_experts_per_tok, 1)
        weights = scores.gather(ids, axis=1, num_elementwise_dims=1)
        if cfg.norm_topk_prob:
            weights = weights / (weights.sum(1, keepdim=True) + 1e-20)
        weights = weights * cfg.routed_scaling_factor
        # The Q8_0 latent projections read and write BF16 directly, so the
        # BF16 expert kernel needs no casts.
        routed_input = self.latent_in(hidden_states, out_dtype=trt.bfloat16)
        buffers, recipes = self._experts()
        routed = F.w2a16_moe(routed_input,
                             buffers,
                             recipes,
                             ids.cast(trt.int32),
                             weights.cast(trt.float32),
                             cfg.num_experts,
                             cfg.moe_intermediate_size,
                             weight_prefix=self.key("experts"))
        routed = self.latent_out(routed)
        return routed + self.shared_experts(hidden_states)


class NemotronHPuzzleAttention(NemotronHAttention):
    """Attention whose Q8_0 Q/K/V projections run as one packed plugin."""

    def packed_qkv(self, hidden_states):
        projections = (self.q_proj, self.k_proj, self.v_proj)
        if any(projection.quant_type() != quantization.QUANT_Q8_0
               or projection.has_adapter() for projection in projections):
            return super().packed_qkv(hidden_states)
        # One plugin quantizes the activations once and writes Q, K and V
        # side by side, so no concatenation of plugin outputs is needed.
        return F.q8_0_packed_linear(
            hidden_states, [(projection.prefix, projection.weight_descriptor())
                            for projection in projections], trt.float16)


class NemotronHPuzzleBlock(NemotronHBlock):
    attention_class = NemotronHPuzzleAttention
    moe_class = NemotronHPuzzleMoE


class NemotronHPuzzleForCausalLM(NemotronHForCausalLM):
    """Nemotron 3.5 Bolt direct model with per-layer expert geometry."""

    def make_layers(self, ctx):
        if len(ctx.cfg.block_configs) != len(ctx.cfg.layer_types):
            raise ValueError("Puzzle block config count does not match layers")
        layers = []
        for index, (layer_type, block) in enumerate(
                zip(ctx.cfg.layer_types, ctx.cfg.block_configs)):
            layer_ctx = ctx
            if layer_type == config.LAYER_MOE:
                experts = int(block["n_routed_experts"])
                layer_cfg = replace(
                    ctx.cfg,
                    num_experts=experts,
                    n_routed_experts=experts,
                    num_experts_per_tok=int(block["num_experts_per_tok"]),
                    moe_intermediate_size=int(block["moe_intermediate_size"]))
                layer_ctx = ctx.with_checkpoint(layer_cfg, ctx.weights)
            layers.append(
                NemotronHPuzzleBlock(layer_ctx, f"backbone.layers.{index}",
                                     layer_type))
        return layers
