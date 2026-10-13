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
"""Reusable MoE modules whose checkpoint and routing semantics are identical."""

from typing import Callable, Dict, Tuple

import numpy as np
import tensorrt as trt

from ..core import quantization
from ..core.weights import ParameterSpec
from ..weight_packing import int4 as int4_pack
from ..weight_packing import nvfp4 as nvfp4_pack
from . import functional as F
from .linear import Linear
from .module import BuildContext, Module
from .tensor import Tensor


def prepare_gated_nvfp4_weights(
        ctx: BuildContext, experts: "GatedExperts",
        repack_experts: Callable) -> Dict[str, np.ndarray]:
    """Prepare the common nine-buffer NVFP4 gated-expert op layout."""
    cfg = ctx.cfg
    layout = "concat" if ctx.options.sm12x else "interleave"
    packed = repack_experts(
        experts.load_expert_raw_nvfp4,
        cfg.num_experts,
        cfg.hidden_size,
        cfg.moe_intermediate_size,
        cfg.group_size,
        fc1_layout=layout,
    )
    (fc1_qweights, fc1_blocks_scale, fc1_alpha, fc2_qweights, fc2_blocks_scale,
     fc2_alpha) = packed
    return {
        "fc1_qweights": fc1_qweights,
        "fc1_blocks_scale": fc1_blocks_scale,
        "fc1_alpha": fc1_alpha,
        "fc2_qweights": fc2_qweights,
        "fc2_blocks_scale": fc2_blocks_scale,
        "fc2_alpha": fc2_alpha,
        "input_global_scale": np.ones(cfg.num_experts, np.float32),
        "down_input_scale": np.ones(cfg.num_experts, np.float32),
        "e_score_correction_bias": np.zeros(cfg.num_experts, np.float32),
    }


def prepare_gated_int4_weights(
    ctx: BuildContext,
    load_projection: Callable[[int, str], Tuple[np.ndarray, np.ndarray,
                                                np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Prepare model-provided GPTQ experts for the Marlin plugin layout."""
    cfg = ctx.cfg
    gate_up_weights = []
    gate_up_scales = []
    down_weights = []
    down_scales = []
    for expert_index in range(cfg.num_experts):

        def extract(projection: str):
            qweight, qzeros, scales = load_projection(expert_index, projection)
            return int4_pack.extract_gptq_for_moe(
                qweight, qzeros, scales, cfg.group_size,
                cfg.quant.gptq_zero_point_offset)

        gate_weight, gate_scale = extract("gate_proj")
        up_weight, up_scale = extract("up_proj")
        down_weight, down_scale = extract("down_proj")
        gate_up_weights.append(np.concatenate((gate_weight, up_weight),
                                              axis=0))
        gate_up_scales.append(np.concatenate((gate_scale, up_scale), axis=0))
        down_weights.append(down_weight)
        down_scales.append(down_scale)
    gate_weight, gate_scale = int4_pack.pack_moe_marlin(
        np.stack(gate_up_weights), np.stack(gate_up_scales), cfg.group_size)
    down_weight, down_scale = int4_pack.pack_moe_marlin(
        np.stack(down_weights), np.stack(down_scales), cfg.group_size)
    return {
        "fc_gate_up_qweights": gate_weight,
        "fc_gate_up_scales": gate_scale,
        "fc_down_qweights": down_weight,
        "fc_down_scales": down_scale,
    }


class TopKRouter(Module):
    """FP16 router projection producing FP32 logits."""

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.projection = Linear(ctx, prefix, tensor_parallel=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states.reshape((-1, self.cfg.hidden_size))
        return self.projection(hidden_states, rank=2).cast(trt.float32)


class GatedExperts(Module):
    """Load stacked or individually stored gate/up/down expert weights."""

    def __len__(self) -> int:
        return self.cfg.num_experts

    def load_expert_dense(self, expert_index: int) -> Dict[str, np.ndarray]:
        gate_up_key = self.key("gate_up_proj")
        down_key = self.key("down_proj")
        if self.weights.has(gate_up_key) and self.weights.has(down_key):
            gate_up = self.weights.f32(gate_up_key)[expert_index]
            gate, up = np.split(gate_up, 2, axis=0)
            down = self.weights.f32(down_key)[expert_index]
            return {
                "gate": np.ascontiguousarray(gate),
                "up": np.ascontiguousarray(up),
                "down": np.ascontiguousarray(down),
            }

        prefix = self.key(str(expert_index))
        return {
            "gate": self.weights.expert_dense_f32(f"{prefix}.gate_proj"),
            "up": self.weights.expert_dense_f32(f"{prefix}.up_proj"),
            "down": self.weights.expert_dense_f32(f"{prefix}.down_proj"),
        }

    def load_expert_raw_nvfp4(self, expert_index: int) -> dict:
        """Load one provider-packed gate/up/down expert without conversion."""
        prefix = self.key(str(expert_index))
        return {
            projection:
            self.weights.expert_raw_nvfp4(f"{prefix}.{projection}_proj")
            for projection in ("gate", "up", "down")
        }


class GroupedSigmoidRouter(Module):
    """Float32 router used by grouped sigmoid top-k models."""

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.correction_key = self.key("e_score_correction_bias")
        self.correction = (self.weights.f32(self.correction_key)
                           if self.weights.has(self.correction_key) else
                           np.zeros(self.cfg.num_experts, dtype=np.float32))

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states.reshape(
            (-1, self.cfg.hidden_size)).cast(trt.float32)
        weight = F.constant(self.weights.f32(self.key("weight")),
                            "router_weight")
        return hidden_states.matmul(weight,
                                    rhs_op=trt.MatrixOperation.TRANSPOSE)


class NonGatedNvfp4Experts(Module):
    """NVFP4 non-gated expert bank shared by equivalent Nemotron models."""

    def __init__(self, ctx: BuildContext, prefix: str,
                 repack_experts: Callable) -> None:
        super().__init__(ctx, prefix)
        self.repack_experts = repack_experts
        first_expert = self.key("0.up_proj")
        self.a16 = (self.weights.module_quant_type(first_expert) ==
                    quantization.QUANT_NVFP4_A16)

    def _has_stacked_experts(self) -> bool:
        return (self.weights.has(self.key("up_proj"))
                and self.weights.has(self.key("down_proj")))

    def _load_expert(self, expert_index: int) -> dict:
        prefix = self.key(str(expert_index))
        up = self.weights.expert_raw_nvfp4(prefix + ".up_proj")
        down = self.weights.expert_raw_nvfp4(prefix + ".down_proj")
        return {
            "up_packed": up["packed"],
            "up_sf": up["sf"],
            "up_alpha": up["alpha"],
            "down_packed": down["packed"],
            "down_sf": down["sf"],
            "down_alpha": down["alpha"],
        }

    def _pack_stacked_experts(self, hidden_size: int, hidden_alignment: int):
        cfg = self.cfg
        padded_intermediate = ((cfg.moe_intermediate_size + 127) // 128) * 128
        padded_hidden = ((hidden_size + hidden_alignment - 1) //
                         hidden_alignment) * hidden_alignment
        if padded_hidden % cfg.group_size:
            raise ValueError("padded hidden size must divide the FP4 group")
        stacked_up = self.weights.f32(self.key("up_proj"))
        stacked_down = self.weights.f32(self.key("down_proj"))
        fc1_weight = []
        fc1_scale = []
        fc2_weight = []
        fc2_scale = []
        for expert_index in range(cfg.num_experts):
            up = np.zeros((padded_intermediate, padded_hidden), np.float32)
            down = np.zeros((padded_hidden, padded_intermediate), np.float32)
            up[:cfg.
               moe_intermediate_size, :hidden_size] = stacked_up[expert_index]
            down[:hidden_size, :cfg.
                 moe_intermediate_size] = stacked_down[expert_index]
            packed_up, scale_up = nvfp4_pack.pack_nvfp4_moe_weight(
                up, cfg.group_size)
            packed_down, scale_down = nvfp4_pack.pack_nvfp4_moe_weight(
                down, cfg.group_size)
            fc1_weight.append(packed_up)
            fc1_scale.append(scale_up)
            fc2_weight.append(packed_down)
            fc2_scale.append(scale_down)
        ones = np.ones(cfg.num_experts, np.float32)
        return (np.stack(fc1_weight), np.stack(fc1_scale), ones,
                np.stack(fc2_weight), np.stack(fc2_scale), ones.copy(),
                padded_intermediate, padded_hidden)

    def _checkpoint_records(self) -> dict:
        records = {}
        for expert in range(self.cfg.num_experts):
            prefix = self.key(str(expert))
            records[expert,
                    "up"] = self.weights.nvfp4_checkpoint_names(prefix +
                                                                ".up_proj")
            records[expert,
                    "down"] = self.weights.nvfp4_checkpoint_names(prefix +
                                                                  ".down_proj")
        return records

    def _checkpoint_specs(self, padded_intermediate: int,
                          padded_hidden: int) -> dict:
        num_experts = self.cfg.num_experts
        group_size = self.cfg.group_size
        return {
            "fc1_qweights":
            ParameterSpec(
                (num_experts, padded_intermediate, padded_hidden // 2),
                np.int8),
            "fc1_blocks_scale":
            ParameterSpec((num_experts, (padded_intermediate + 127) // 128,
                           (padded_hidden // group_size + 3) // 4, 32, 4, 4),
                          np.int8),
            "fc1_alpha":
            ParameterSpec((num_experts, ), np.float32),
            "fc2_qweights":
            ParameterSpec(
                (num_experts, padded_hidden, padded_intermediate // 2),
                np.int8),
            "fc2_blocks_scale":
            ParameterSpec(
                (num_experts, (padded_hidden + 127) // 128,
                 (padded_intermediate // group_size + 3) // 4, 32, 4, 4),
                np.int8),
            "fc2_alpha":
            ParameterSpec((num_experts, ), np.float32),
            "input_global_scale":
            ParameterSpec((num_experts, ), np.float32),
            "down_input_scale":
            ParameterSpec((num_experts, ), np.float32),
            "e_score_correction_bias":
            ParameterSpec((num_experts, ), np.float32),
        }

    def _checkpoint_bindings(self, records: dict, correction_key: str) -> dict:
        num_experts = self.cfg.num_experts

        def field(projection: str, index: int):
            return [
                records[expert, projection][index]
                for expert in range(num_experts)
            ]

        def reciprocal(projection: str) -> bool:
            values = {
                records[expert, projection][3]
                for expert in range(num_experts)
            }
            if len(values) != 1:
                raise ValueError(
                    "Nemotron NVFP4 experts use inconsistent alpha formats")
            return values.pop()

        correction = (self.weights.checkpoint_binding([correction_key])
                      if self.weights.has(correction_key) else
                      self.weights.checkpoint_binding(
                          [], "generated", "fill", fill_value=0.0))
        return {
            "fc1_qweights":
            self.weights.checkpoint_binding(field("up", 0),
                                            "nvfp4_qweight",
                                            "nvfp4_expert_qweight",
                                            num_experts=num_experts),
            "fc1_blocks_scale":
            self.weights.checkpoint_binding(field("up", 1),
                                            "nvfp4_scale_linear",
                                            "nvfp4_expert_scale",
                                            num_experts=num_experts),
            "fc1_alpha":
            self.weights.checkpoint_binding(field("up", 2),
                                            "plugin",
                                            "nvfp4_fc1_alpha",
                                            num_experts=num_experts,
                                            reciprocal_alpha=reciprocal("up")),
            "fc2_qweights":
            self.weights.checkpoint_binding(field("down", 0),
                                            "nvfp4_qweight",
                                            "nvfp4_expert_qweight",
                                            num_experts=num_experts),
            "fc2_blocks_scale":
            self.weights.checkpoint_binding(field("down", 1),
                                            "nvfp4_scale_linear",
                                            "nvfp4_expert_scale",
                                            num_experts=num_experts),
            "fc2_alpha":
            self.weights.checkpoint_binding(
                field("down", 2),
                "plugin",
                "nvfp4_fc2_alpha",
                num_experts=num_experts,
                reciprocal_alpha=reciprocal("down")),
            "input_global_scale":
            self.weights.checkpoint_binding([],
                                            "generated",
                                            "fill",
                                            fill_value=1.0),
            "down_input_scale":
            self.weights.checkpoint_binding([],
                                            "generated",
                                            "fill",
                                            fill_value=1.0),
            "e_score_correction_bias":
            correction,
        }

    def forward(self, hidden_states: Tensor, router_logits: Tensor,
                correction: np.ndarray, correction_key: str) -> Tensor:
        cfg = self.cfg
        hidden_size = cfg.moe_latent_size or cfg.hidden_size
        if self.a16:
            if self._has_stacked_experts():
                raise NotImplementedError(
                    "stacked dense A16 expert checkpoints are not supported")
            weights = nvfp4_pack.pack_nvfp4_a16_nongated_experts(
                self._load_expert, cfg.num_experts, hidden_size,
                cfg.moe_intermediate_size)
            weights["e_score_correction_bias"] = correction.astype(np.float32)
            return F.nvfp4_a16_moe(router_logits,
                                   hidden_states,
                                   weights,
                                   cfg.num_experts,
                                   cfg.num_experts_per_tok,
                                   hidden_size,
                                   weights["padded_intermediate"],
                                   F.MoeActivation.RELU2,
                                   cfg.n_group,
                                   cfg.topk_group,
                                   int(cfg.norm_topk_prob),
                                   cfg.routed_scaling_factor,
                                   F.MoeRouting.SIGMOID_GROUP_TOPK,
                                   weight_prefix=self.prefix)

        hidden_alignment = 256 if self.ctx.options.sm12x else 1
        padded_intermediate = ((cfg.moe_intermediate_size + 127) // 128) * 128
        padded_hidden = ((hidden_size + hidden_alignment - 1) //
                         hidden_alignment) * hidden_alignment

        def materialize():
            if self._has_stacked_experts():
                packed = self._pack_stacked_experts(hidden_size,
                                                    hidden_alignment)
            else:
                packed = self.repack_experts(self._load_expert,
                                             cfg.num_experts, hidden_size,
                                             cfg.moe_intermediate_size,
                                             cfg.group_size, hidden_alignment)
            (fc1_weight, fc1_scale, fc1_alpha, fc2_weight, fc2_scale,
             fc2_alpha, _, _) = packed
            return {
                "fc1_qweights": fc1_weight,
                "fc1_blocks_scale": fc1_scale,
                "fc1_alpha": fc1_alpha,
                "fc2_qweights": fc2_weight,
                "fc2_blocks_scale": fc2_scale,
                "fc2_alpha": fc2_alpha,
                "input_global_scale": np.ones(cfg.num_experts, np.float32),
                "down_input_scale": np.ones(cfg.num_experts, np.float32),
                "e_score_correction_bias": correction.astype(np.float32),
            }

        bindings = None
        if not self._has_stacked_experts():
            records = self._checkpoint_records()
            weights = self.weights.parameter_value(
                "nvfp4_moe", self.prefix, lambda: self._checkpoint_specs(
                    padded_intermediate, padded_hidden), materialize)
            bindings = self._checkpoint_bindings(records, correction_key)
        else:
            weights = materialize()

        if padded_hidden != hidden_size:
            hidden_states = F.pad_last_dim(hidden_states,
                                           padded_hidden - hidden_size,
                                           hidden_states.ndim)
        output = F.nvfp4_moe(router_logits,
                             hidden_states,
                             weights,
                             cfg.num_experts,
                             cfg.num_experts_per_tok,
                             padded_hidden,
                             padded_intermediate,
                             4,
                             cfg.n_group,
                             cfg.topk_group,
                             int(cfg.norm_topk_prob),
                             cfg.routed_scaling_factor,
                             1,
                             self.ctx.options.sm12x,
                             weight_prefix=self.prefix,
                             weight_bindings=bindings)
        if padded_hidden != hidden_size:
            output = F.slice_last_dim(output, 0, hidden_size, output.ndim)
        return output
