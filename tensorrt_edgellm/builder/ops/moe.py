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

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import tensorrt as trt

from ..core import quantization, weight_policy
from ..core.numpy_dtypes import fp8_e4m3_bytes_to_f32
from ..core.weights import ParameterSpec
from ..weight_packing import int4 as int4_pack
from ..weight_packing import nvfp4 as nvfp4_pack
from . import functional as F
from .linear import Linear
from .module import BuildContext, Module
from .tensor import Tensor

LOGGER = logging.getLogger("builder.moe")

_FP4_BLOCK = 16


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
    input_global_scale, down_input_scale = experts.input_scales()
    return {
        "fc1_qweights": fc1_qweights,
        "fc1_blocks_scale": fc1_blocks_scale,
        "fc1_alpha": fc1_alpha,
        "fc2_qweights": fc2_qweights,
        "fc2_blocks_scale": fc2_blocks_scale,
        "fc2_alpha": fc2_alpha,
        "input_global_scale": input_global_scale,
        "down_input_scale": down_input_scale,
        "e_score_correction_bias": np.zeros(cfg.num_experts, np.float32),
    }


def _nvfp4_a16_expert_parameters(ctx: BuildContext,
                                 prefix: str,
                                 fc1_projections: Tuple[str, ...],
                                 hidden_size: int,
                                 materialize: Callable,
                                 correction: np.ndarray,
                                 correction_key: str = ""):
    cfg = ctx.cfg
    experts = cfg.num_experts
    intermediate = cfg.moe_intermediate_size
    padded_intermediate = (intermediate + 127) // 128 * 128
    fc1_out = len(fc1_projections) * padded_intermediate

    def specs():
        return {
            "fc1_qweights":
            ParameterSpec((experts, hidden_size // 16, 8 * fc1_out), np.int8),
            "fc1_block_scales":
            ParameterSpec((experts, hidden_size // 16, fc1_out), np.int8),
            "fc1_global_scales":
            ParameterSpec(((2, experts) if len(fc1_projections) == 2 else
                           (experts, )), np.float16),
            "fc2_qweights":
            ParameterSpec(
                (experts, padded_intermediate // 16, 8 * hidden_size),
                np.int8),
            "fc2_block_scales":
            ParameterSpec((experts, padded_intermediate // 16, hidden_size),
                          np.int8),
            "fc2_global_scales":
            ParameterSpec((experts, ), np.float16),
            "e_score_correction_bias":
            ParameterSpec((experts, ), np.float32),
            "padded_intermediate":
            padded_intermediate,
        }

    records = {}
    for expert in range(experts):
        expert_prefix = f"{prefix}.{expert}"
        for projection in (*fc1_projections, "down"):
            records[expert, projection] = ctx.weights.nvfp4_checkpoint_names(
                f"{expert_prefix}.{projection}_proj")

    def fields(projections, field):
        return [
            records[expert, projection][field] for expert in range(experts)
            for projection in projections
        ]

    def reciprocal(projections):
        values = {
            records[expert, projection][3]
            for expert in range(experts)
            for projection in projections
        }
        if len(values) != 1:
            raise ValueError("NVFP4 experts use inconsistent alpha formats")
        return values.pop()

    fc1_common = {
        "num_experts": experts,
        "keys_per_expert": len(fc1_projections),
        "source_n": intermediate,
        "source_k": hidden_size,
        "output_n": padded_intermediate,
        "output_k": hidden_size,
    }
    fc2_common = {
        "num_experts": experts,
        "keys_per_expert": 1,
        "source_n": hidden_size,
        "source_k": intermediate,
        "output_n": hidden_size,
        "output_k": padded_intermediate,
    }
    if correction_key and ctx.weights.has(correction_key):
        correction_binding = ctx.weights.checkpoint_binding(
            [correction_key], assemble="cast_to_fp32")
    else:
        if not np.all(correction == correction[0]):
            raise ValueError("generated expert correction must be uniform")
        correction_binding = ctx.weights.checkpoint_binding([],
                                                            "generated",
                                                            "fill",
                                                            fill_value=float(
                                                                correction[0]))
    bindings = {
        "fc1_qweights":
        ctx.weights.checkpoint_binding(fields(fc1_projections,
                                              0), "nvfp4_packed",
                                       "nvfp4_a16_qweight", **fc1_common),
        "fc1_block_scales":
        ctx.weights.checkpoint_binding(fields(fc1_projections,
                                              1), "nvfp4_scale",
                                       "nvfp4_a16_scale", **fc1_common),
        "fc1_global_scales":
        ctx.weights.checkpoint_binding(
            fields(fc1_projections, 2),
            "plugin",
            "nvfp4_a16_alpha",
            num_experts=experts,
            keys_per_expert=len(fc1_projections),
            reciprocal_alpha=reciprocal(fc1_projections),
            alpha_multiplier=128.0),
        "fc2_qweights":
        ctx.weights.checkpoint_binding(fields(("down", ), 0), "nvfp4_packed",
                                       "nvfp4_a16_qweight", **fc2_common),
        "fc2_block_scales":
        ctx.weights.checkpoint_binding(fields(("down", ), 1), "nvfp4_scale",
                                       "nvfp4_a16_scale", **fc2_common),
        "fc2_global_scales":
        ctx.weights.checkpoint_binding(fields(("down", ), 2),
                                       "plugin",
                                       "nvfp4_a16_alpha",
                                       num_experts=experts,
                                       keys_per_expert=1,
                                       reciprocal_alpha=reciprocal(("down", )),
                                       alpha_multiplier=128.0),
        "e_score_correction_bias":
        correction_binding,
    }
    parameters = ctx.weights.parameter_value(
        weight_policy.EXTERNAL_WEIGHT_NVFP4_A16, prefix, specs, materialize)
    return parameters, bindings


def prepare_gated_nvfp4_a16_weights(ctx: BuildContext,
                                    experts: "GatedExperts",
                                    correction: Optional[np.ndarray] = None,
                                    correction_key: str = ""):
    """Prepare checkpoint-backed gated experts for the W4A16 MoE plugin."""
    if correction is None:
        correction = np.zeros(ctx.cfg.num_experts, np.float32)

    def materialize():
        parameters = nvfp4_pack.pack_nvfp4_a16_gated_experts(
            experts.load_expert_raw_nvfp4, ctx.cfg.num_experts,
            ctx.cfg.hidden_size, ctx.cfg.moe_intermediate_size)
        parameters["e_score_correction_bias"] = np.ascontiguousarray(
            correction, np.float32)
        return parameters

    return _nvfp4_a16_expert_parameters(ctx, experts.prefix, ("gate", "up"),
                                        ctx.cfg.hidden_size, materialize,
                                        correction, correction_key)


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


def gated_w4a16_group_size(hidden_size: int, intermediate_size: int) -> int:
    """Select a group size supported by every expert projection."""
    for group_size in (128, 64, 32, 16):
        if (hidden_size % group_size == 0
                and intermediate_size % group_size == 0):
            return group_size
    raise ValueError("W4A16 MoE requires hidden and intermediate sizes "
                     "divisible by 16")


def prepare_gated_w4a16_weights(
    ctx: BuildContext,
    experts: "GatedExperts",
    *,
    group_size: int | None = None,
    load_expert: Callable[[int], Dict[str, np.ndarray]] | None = None
) -> Dict[str, np.ndarray]:
    """Requantize floating-point gated experts for the Marlin MoE op."""
    cfg = ctx.cfg
    group_size = (gated_w4a16_group_size(cfg.hidden_size,
                                         cfg.moe_intermediate_size)
                  if group_size is None else int(group_size))
    if (group_size <= 0 or cfg.hidden_size % group_size
            or cfg.moe_intermediate_size % group_size):
        raise ValueError(
            f"W4A16 MoE group size {group_size} must divide hidden and "
            "intermediate dimensions")
    load_expert = load_expert or experts.load_expert_dense
    gate_up_weights = []
    gate_up_scales = []
    down_weights = []
    down_scales = []

    def quantize(weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        weight = np.asarray(weight, dtype=np.float32)
        out_features, in_features = weight.shape
        grouped = weight.reshape(out_features, in_features // group_size,
                                 group_size)
        absmax = np.max(np.abs(grouped), axis=2)
        scale = np.where(absmax > 0.0, absmax / 7.0, 1.0)
        quantized = np.clip(np.rint(grouped / scale[..., None]), -7,
                            7).astype(np.int8)
        nibbles = (quantized.reshape(out_features, in_features) + 8).astype(
            np.uint8)
        return np.ascontiguousarray(nibbles), np.ascontiguousarray(
            scale, dtype=np.float16)

    for expert_index in range(cfg.num_experts):
        dense = load_expert(expert_index)
        gate, gate_scale = quantize(dense["gate"])
        up, up_scale = quantize(dense["up"])
        down, down_scale = quantize(dense["down"])
        gate_up_weights.append(np.concatenate((gate, up), axis=0))
        gate_up_scales.append(np.concatenate((gate_scale, up_scale), axis=0))
        down_weights.append(down)
        down_scales.append(down_scale)

    gate_weight, gate_scale = int4_pack.pack_moe_marlin(
        np.stack(gate_up_weights), np.stack(gate_up_scales), group_size)
    down_weight, down_scale = int4_pack.pack_moe_marlin(
        np.stack(down_weights), np.stack(down_scales), group_size)
    return {
        "fc_gate_up_qweights": gate_weight,
        "fc_gate_up_scales": gate_scale,
        "fc_down_qweights": down_weight,
        "fc_down_scales": down_scale,
    }


def prepare_gated_trt_native_fp16_weights(
        ctx: BuildContext, experts: "GatedExperts") -> Dict[str, np.ndarray]:
    """Stack FP16 gated experts as ``[E, H, I]`` / ``[E, I, H]`` for the
    native MoE layer (the checkpoint's ``[out, in]`` matrices transposed)."""
    gate, up, down = [], [], []
    for expert_index in range(ctx.cfg.num_experts):
        dense = experts.load_expert_dense(expert_index)
        gate.append(np.ascontiguousarray(dense["gate"].T, dtype=np.float16))
        up.append(np.ascontiguousarray(dense["up"].T, dtype=np.float16))
        down.append(np.ascontiguousarray(dense["down"].T, dtype=np.float16))
    stacked = {
        "fc_gate_weights": np.stack(gate),
        "fc_up_weights": np.stack(up),
        "fc_down_weights": np.stack(down),
    }
    LOGGER.info(
        "%s: TRT-native MoE (FP16): %d experts stacked for the native layer: "
        "gate %s, up %s, down %s", experts.prefix, ctx.cfg.num_experts,
        stacked["fc_gate_weights"].shape, stacked["fc_up_weights"].shape,
        stacked["fc_down_weights"].shape)
    return stacked


def prepare_gated_trt_native_w4a8_weights(
        ctx: BuildContext, experts: "GatedExperts") -> Dict[str, np.ndarray]:
    """Stack NVFP4 gated experts for the native MoE layer's W4A8 form.

    The FP4 nibbles are reused as stored; each FP8 block scale is decoded to
    fp32 and multiplied by its expert's global scale. ModelOpt stores
    ``input_scale = amax / (6 * 448)`` while FP8 wants ``amax / 448``, hence
    the factor 6. FC1 is shared by all experts, so one FP8 scale covers them:
    the largest calibrated one, which saturates no expert.
    """
    cfg = ctx.cfg
    # Module-level precondition, looked up through the first expert's
    # projection like the MoE blocks do (`experts.key("0.up_proj")`).
    group_size = ctx.weights.module_quant_group_size(experts.key("0.up_proj"))
    if group_size != _FP4_BLOCK:
        raise ValueError(f"{experts.prefix}: TRT-native W4A8 MoE needs FP4 "
                         f"block size {_FP4_BLOCK}, got {group_size}")
    # The guard with teeth is per expert: a mixed-precision checkpoint can
    # leave single experts in another format, and only their stored shapes
    # say so.
    expected = {
        "gate": (cfg.moe_intermediate_size, cfg.hidden_size),
        "up": (cfg.moe_intermediate_size, cfg.hidden_size),
        "down": (cfg.hidden_size, cfg.moe_intermediate_size),
    }
    packed = {name: [] for name in ("gate", "up", "down")}
    scales = {name: [] for name in packed}
    for expert_index in range(cfg.num_experts):
        raw = experts.load_expert_raw_nvfp4(expert_index)
        for name in packed:
            projection = raw[name]
            tensor = f"{experts.prefix}.{expert_index}.{name}_proj"
            out_features, in_features = expected[name]
            nibbles = np.ascontiguousarray(projection["packed"])
            if nibbles.shape != (out_features, in_features // 2):
                raise ValueError(
                    f"{tensor}: packed FP4 weight shape {nibbles.shape} != "
                    f"{(out_features, in_features // 2)}; every expert must "
                    "be NVFP4 for the TRT-native W4A8 arm")
            sf = np.asarray(projection["sf"]).reshape(out_features, -1)
            if sf.shape[-1] * _FP4_BLOCK != in_features:
                raise ValueError(
                    f"{tensor}: FP8 block scale shape {sf.shape} implies block "
                    f"size {in_features // sf.shape[-1]}, need {_FP4_BLOCK}; "
                    "every expert must be NVFP4 with per-16 block scales for "
                    "the TRT-native W4A8 arm")
            if nibbles.dtype == np.int8:
                nibbles = nibbles.view(np.uint8)
            packed[name].append(nibbles.astype(np.uint8, copy=False))
            block_scale = fp8_e4m3_bytes_to_f32(sf)
            scales[name].append(
                (block_scale * np.float32(projection["alpha"])).astype(
                    np.float32))
    fc1_scales, fc2_scales = experts.input_scales()
    fc1_scale = float(fc1_scales.max())
    if not np.allclose(fc1_scales, fc1_scale, rtol=1e-6, atol=0.0):
        LOGGER.warning(
            "%s: per-expert FC1 input scales differ (min %.4g, max %.4g); "
            "the shared FP8 activation scale uses the max", experts.prefix,
            float(fc1_scales.min()), fc1_scale)
    bank = {
        "fc_gate_packed": np.stack(packed["gate"]),
        "fc_gate_block_scale": np.stack(scales["gate"]),
        "fc_up_packed": np.stack(packed["up"]),
        "fc_up_block_scale": np.stack(scales["up"]),
        "fc_down_packed": np.stack(packed["down"]),
        "fc_down_block_scale": np.stack(scales["down"]),
        "fc1_activation_scale": np.array(fc1_scale * 6.0, dtype=np.float16),
        "fc_down_activation_scales": (fc2_scales * 6.0).astype(np.float32),
    }
    LOGGER.info(
        "%s: TRT-native MoE (W4A8): %d NVFP4 experts: gate %s, up %s, down %s "
        "(packed FP4, block scales per %d = e4m3(weight_scale) x "
        "weight_scale_2); act_scale=%.6g, down_act_scale[0:2]=%s",
        experts.prefix, cfg.num_experts, bank["fc_gate_packed"].shape,
        bank["fc_up_packed"].shape, bank["fc_down_packed"].shape, _FP4_BLOCK,
        float(bank["fc1_activation_scale"]),
        bank["fc_down_activation_scales"][:2].tolist())
    return bank


def trt_native_gated_moe(
        ctx: BuildContext,
        experts: "GatedExperts",
        router_logits: Tensor,
        hidden_states: Tensor,
        *,
        norm_topk_prob: int,
        routing_mode: F.MoeRouting = F.MoeRouting.SOFTMAX_TOPK) -> Tensor:
    """Route SiLU-gated experts through TensorRT's native MoE layer.

    Selected by ``--trt-native-moe``. Only FP16 and activation-quantized
    NVFP4 experts have a native form; anything else is an error, never a
    silent plugin fallback. The routing built in the graph is plain softmax
    -> top-k (renormalised when ``norm_topk_prob``), so a checkpoint that
    asks for group top-k, a routed scaling factor or another routing mode is
    rejected instead of being built with routing it did not ask for.
    """
    cfg = ctx.cfg
    unsupported = []
    if routing_mode != F.MoeRouting.SOFTMAX_TOPK:
        unsupported.append(
            f"routing_mode={getattr(routing_mode, 'name', routing_mode)}")
    if cfg.n_group > 1 or cfg.topk_group > 1:
        unsupported.append(
            f"n_group={cfg.n_group}, topk_group={cfg.topk_group}")
    if cfg.routed_scaling_factor != 1.0:
        unsupported.append(
            f"routed_scaling_factor={cfg.routed_scaling_factor}")
    if unsupported:
        raise ValueError(
            "--trt-native-moe builds plain softmax -> top-k routing; "
            f"{experts.prefix} needs {', '.join(unsupported)}, which the "
            "native arm does not implement. Drop the flag to build the "
            "plugin path.")
    quant_type = ctx.weights.module_quant_type(experts.key("0.up_proj"))
    fallback = ctx.weights.uses_w4a16_fallback(quant_type)
    if quant_type == quantization.QUANT_FP16:
        weights = prepare_gated_trt_native_fp16_weights(ctx, experts)
        return F.trt_native_moe_fp16(router_logits,
                                     hidden_states,
                                     weights["fc_gate_weights"],
                                     weights["fc_up_weights"],
                                     weights["fc_down_weights"],
                                     cfg.num_experts,
                                     cfg.num_experts_per_tok,
                                     cfg.hidden_size,
                                     cfg.moe_intermediate_size,
                                     norm_topk_prob=norm_topk_prob,
                                     weight_prefix=experts.prefix)
    if quant_type == quantization.QUANT_NVFP4 and not fallback:
        weights = prepare_gated_trt_native_w4a8_weights(ctx, experts)
        return F.trt_native_moe_w4a8(router_logits,
                                     hidden_states,
                                     weights,
                                     cfg.num_experts,
                                     cfg.num_experts_per_tok,
                                     cfg.hidden_size,
                                     cfg.moe_intermediate_size,
                                     norm_topk_prob=norm_topk_prob,
                                     weight_prefix=experts.prefix)
    described = quant_type
    if fallback:
        described += " executed as W4A16 on this device"
    raise ValueError(
        f"--trt-native-moe requested but the experts are {described}; only "
        "FP16 and NVFP4 experts have a TRT_MoE form (INT4 and NVFP4-A16 "
        "experts do not). Drop the flag to build the plugin path.")


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
            gate_up = self.weights.f32(gate_up_key, index=expert_index)
            gate, up = np.split(gate_up, 2, axis=0)
            down = self.weights.f32(down_key, index=expert_index)
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
        for projection in ("gate", "up", "down"):
            if not self.weights.is_nvfp4(f"{prefix}.{projection}_proj"):
                raise ValueError(
                    f"{prefix}.{projection}_proj is not stored as NVFP4 (no "
                    "weight_scale / weight_scale_2); every expert of "
                    f"{self.prefix} must be NVFP4 for the NVFP4 MoE arms")
        return {
            projection:
            self.weights.expert_raw_nvfp4(f"{prefix}.{projection}_proj")
            for projection in ("gate", "up", "down")
        }

    def input_scales(self) -> Tuple[np.ndarray, np.ndarray]:
        """Collect calibrated gated-FC1 and FC2 activation scales."""
        fc1 = []
        fc2 = []
        for expert_index in range(self.cfg.num_experts):
            prefix = self.key(str(expert_index))
            gate_scale = self.weights.nvfp4_input_scale(prefix + ".gate_proj")
            up_scale = self.weights.nvfp4_input_scale(prefix + ".up_proj")
            if gate_scale != up_scale:
                raise ValueError(
                    f"expert {expert_index} gate/up input scales differ")
            fc1.append(gate_scale)
            fc2.append(self.weights.nvfp4_input_scale(prefix + ".down_proj"))
        return (np.asarray(fc1,
                           dtype=np.float32), np.asarray(fc2,
                                                         dtype=np.float32))


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


class NonGatedFp16Experts(Module):
    """Non-gated ReLU2 experts lowered through the FP16 MoE plugin."""

    def plugin_intermediate_size(self) -> int:
        size = self.cfg.moe_intermediate_size
        return ((size + 127) // 128) * 128

    def _materialize(self) -> Dict[str, np.ndarray]:
        intermediate_size = self.cfg.moe_intermediate_size
        padded_intermediate = self.plugin_intermediate_size()
        up_weights = []
        down_weights = []
        for expert_index in range(self.cfg.num_experts):
            prefix = self.key(str(expert_index))
            up = self.weights.expert_dense_f32(prefix + ".up_proj")
            down = self.weights.expert_dense_f32(prefix + ".down_proj")
            if padded_intermediate != intermediate_size:
                up = np.pad(up, ((0, padded_intermediate - intermediate_size),
                                 (0, 0)))
                down = np.pad(down,
                              ((0, 0),
                               (0, padded_intermediate - intermediate_size)))
            up_weights.append(up.astype(np.float16))
            down_weights.append(down.astype(np.float16))
        return {
            "fc1_weights": np.stack(up_weights),
            "fc2_weights": np.stack(down_weights),
        }

    def forward(self, hidden_states: Tensor, router_logits: Tensor,
                correction: np.ndarray, correction_key: str) -> Tensor:
        cfg = self.cfg
        return F.fp16_moe(
            router_logits,
            hidden_states,
            self._materialize(),
            cfg.num_experts,
            cfg.num_experts_per_tok,
            cfg.moe_latent_size or cfg.hidden_size,
            self.plugin_intermediate_size(),
            weight_prefix=self.prefix,
            weight_bindings={},
            activation_type=F.MoeActivation.RELU2,
            n_group=cfg.n_group,
            topk_group=cfg.topk_group,
            norm_topk_prob=int(cfg.norm_topk_prob),
            routed_scaling_factor=cfg.routed_scaling_factor,
            routing_mode=F.MoeRouting.SIGMOID_GROUP_TOPK,
            e_score_correction_bias=F.constant(correction, correction_key),
        )


class NonGatedNvfp4Experts(Module):
    """NVFP4 non-gated expert bank shared by equivalent Nemotron models."""

    def __init__(self, ctx: BuildContext, prefix: str,
                 repack_experts: Callable) -> None:
        super().__init__(ctx, prefix)
        self.repack_experts = repack_experts
        first_expert = self.key("0.up_proj")
        quant_type = self.weights.module_quant_type(first_expert)
        self.a16 = (quant_type == quantization.QUANT_NVFP4_A16
                    or self.weights.uses_w4a16_fallback(quant_type))

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
            "up_input_scale": up["input_scale"],
            "down_packed": down["packed"],
            "down_sf": down["sf"],
            "down_alpha": down["alpha"],
            "down_input_scale": down["input_scale"],
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
            self.weights.nvfp4_moe_input_scale_binding(self.prefix,
                                                       num_experts,
                                                       ("up_proj", )),
            "down_input_scale":
            self.weights.nvfp4_moe_input_scale_binding(self.prefix,
                                                       num_experts,
                                                       ("down_proj", )),
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

            def materialize_a16():
                parameters = nvfp4_pack.pack_nvfp4_a16_nongated_experts(
                    self._load_expert, cfg.num_experts, hidden_size,
                    cfg.moe_intermediate_size)
                parameters["e_score_correction_bias"] = correction.astype(
                    np.float32)
                return parameters

            weights, bindings = _nvfp4_a16_expert_parameters(
                self.ctx, self.prefix, ("up", ), hidden_size, materialize_a16,
                correction, correction_key)
            # Nvfp4A16MoePlugin::getOutputShapes hard-codes a 3D [B,S,H] contract
            # (mirrors input dims 0/1 verbatim); F.nvfp4_a16_moe promotes/demotes
            # the ragged builder's 2D [tokens, H] hidden_states around the plugin
            # call internally.
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
                                   weight_prefix=self.prefix,
                                   weight_bindings=bindings)

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
            if self._has_stacked_experts():
                input_global_scale = np.ones(cfg.num_experts, np.float32)
                down_input_scale = input_global_scale.copy()
            else:
                input_global_scale = np.asarray([
                    self.weights.nvfp4_input_scale(self.key(f"{i}.up_proj"))
                    for i in range(cfg.num_experts)
                ],
                                                dtype=np.float32)
                down_input_scale = np.asarray([
                    self.weights.nvfp4_input_scale(self.key(f"{i}.down_proj"))
                    for i in range(cfg.num_experts)
                ],
                                              dtype=np.float32)
            return {
                "fc1_qweights": fc1_weight,
                "fc1_blocks_scale": fc1_scale,
                "fc1_alpha": fc1_alpha,
                "fc2_qweights": fc2_weight,
                "fc2_blocks_scale": fc2_scale,
                "fc2_alpha": fc2_alpha,
                "input_global_scale": input_global_scale,
                "down_input_scale": down_input_scale,
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
        # Nvfp4MoePlugin::getOutputShapes hard-codes a 3D [B,S,H] contract (mirrors
        # input dims 0/1 verbatim); F.nvfp4_moe promotes/demotes the ragged
        # builder's 2D [tokens, H] hidden_states around the plugin call internally.
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
