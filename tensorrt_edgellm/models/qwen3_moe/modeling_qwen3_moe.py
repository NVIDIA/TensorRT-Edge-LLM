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
"""
Qwen3 MoE causal LM modeling for checkpoint export.

Architecture
------------
Standard decoder transformer with sparse MoE layers interleaved at a
frequency controlled by ``decoder_sparse_step`` (and optionally overridden
per-layer by ``mlp_only_layers``).  Attention is identical to Qwen3 dense
(GQA with per-head QK-norm, optional sliding window).

This module is a standalone reimplementation — it shares only leaf
building blocks (``RMSNorm``, ``Attention``, ``MLP``, ``OnnxSpec``,
``_make_flat_wrapper``) with the default modeling, following the same
decoupling convention used by ``nemotron_h``.

MoE layer design
----------------
The entire Qwen3SparseMoeBlock — router (Linear + Softmax + TopK) and all
expert GEMMs (gate/up/down projections with SiLU gating) plus weighted
combine — is captured as a single custom MoE op. GPTQ/AWQ exports keep using
``trt_edgellm::Int4MoePlugin``. NVFP4 exports use the CuTeDSL
``trt_edgellm::Nvfp4MoePlugin`` contract.

Per-expert weights are loaded into an ``nn.ModuleList`` for checkpoint
compatibility, then stacked into 3-D tensors ``[E, ...]`` during the
post-load repacking pass.

Checkpoint key structure (GPTQ per-expert layout)
--------------------------------------------------
model.embed_tokens.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.post_attention_layernorm.weight
model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,...}
model.layers.{i}.self_attn.{q,k}_norm.weight
model.layers.{i}.mlp.gate_proj.weight          (dense-MLP layers)
model.layers.{i}.mlp.up_proj.weight            (dense-MLP layers)
model.layers.{i}.mlp.down_proj.weight          (dense-MLP layers)
model.layers.{i}.mlp.gate.weight               (MoE layers — router)
model.layers.{i}.mlp.experts.{j}.gate_proj.*   (MoE layers — per-expert)
model.layers.{i}.mlp.experts.{j}.up_proj.*     (MoE layers — per-expert)
model.layers.{i}.mlp.experts.{j}.down_proj.*   (MoE layers — per-expert)
model.norm.weight
lm_head.weight
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

from ...config import (QUANT_FP16, QUANT_INT4_AWQ_MODELOPT, QUANT_INT4_GPTQ,
                       QUANT_NVFP4, QUANT_NVFP4_A16, ModelConfig,
                       module_quant_group_size, module_quant_type)
from ..default.modeling_default import (MLP, Attention, CausalLM, OnnxSpec,
                                        RMSNorm,
                                        _concat_hidden_in_provider_order)
from ..linear import FP16Linear, make_linear
from ..ops import (TRT_DATATYPE_FP4, TRT_DATATYPE_FP8,
                   TRT_MOE_QUANT_DYNAMIC_DBLQ, TRT_MOE_QUANT_STATIC,
                   fp8_dequantize, fp8_quantize, fp16_moe_plugin,
                   int4_moe_plugin, is_trt_native_moe_enabled,
                   is_trt_native_moe_w4a4_enabled, nvfp4_a16_moe_plugin,
                   nvfp4_act_qdq_moe, nvfp4_dequantize_moe,
                   nvfp4_dequantize_moe_with_global_scale, nvfp4_moe_plugin,
                   nvfp4_moe_plugin_geforce, trt_moe_fp16, trt_moe_w4a4,
                   trt_moe_w4a8, use_geforce_nvfp4_moe)

logger = logging.getLogger(__name__)

# SiLU activation type expected by the C++ Int4MoePlugin.
_INT4_ACTIVATION_SILU = 0

# Plugin slot 0 — softmax + flat top-k + renormalize routing kernel
# (``Nvfp4MoeRoutingMode::kSOFTMAX_TOPK``); Qwen3 does not use the
# sigmoid-group-topk path.
_NVFP4_ROUTING_MODE_SOFTMAX_TOPK = 0

# SwiGLU activation_type for the ``Nvfp4MoePlugin`` plugin.
_NVFP4_ACTIVATION_SWIGLU = 2
_NVFP4_MOE_BACKEND_AUTO = 0
_NVFP4_MOE_IO_DTYPE_FP16 = 1
_NVFP4_MOE_MAX_ROUTED_ROWS_AUTO = 0
_NVFP4_MOE_N_GROUP_FLAT = 1
_NVFP4_MOE_TOPK_GROUP_FLAT = 1

# ONNX export dummy-input dims.
__all__ = [
    "Qwen3MoERouter",
    "Qwen3MoEExperts",
    "Qwen3SparseMoeBlock",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoeTransformer",
    "Qwen3MoeCausalLM",
]

# ---------------------------------------------------------------------------
# MoE layer predicate
# ---------------------------------------------------------------------------


def _is_moe_layer(config: ModelConfig, layer_idx: int) -> bool:
    """Return True if *layer_idx* should use a Qwen3SparseMoeBlock.

    Mirrors HuggingFace Qwen3MoeDecoderLayer logic:
    - Any layer listed in ``mlp_only_layers`` is always dense.
    - Otherwise, layer is MoE when ``(layer_idx + 1) % decoder_sparse_step == 0``
      and ``num_experts > 0``.
    """
    if layer_idx in config.mlp_only_layers:
        return False
    return config.num_experts > 0 and (layer_idx +
                                       1) % config.decoder_sparse_step == 0


# ---------------------------------------------------------------------------
# MoE router
# ---------------------------------------------------------------------------


class Qwen3MoERouter(nn.Module):
    """Router weight holder.

    Checkpoint key: ``mlp.gate.weight`` [num_experts, hidden_size].
    The actual routing computation (linear + softmax + topk) is fused
    inside the ``trt_edgellm::Int4MoePlugin`` custom op.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.num_experts,
                        config.hidden_size,
                        dtype=torch.float16),
            requires_grad=False,
        )


# ---------------------------------------------------------------------------
# MoE experts (weight holder)
# ---------------------------------------------------------------------------


class Qwen3MoEExperts(nn.Module):
    """Per-expert MLP modules stored as nn.ModuleList.

    Supports all checkpoint formats including per-expert quantized layouts
    (GPTQ, AWQ) where each expert has independent quantized weight tensors.

    Checkpoint keys (under ``mlp.experts``):
        {i}.gate_proj.*  -- i-th expert gate projection
        {i}.up_proj.*    -- i-th expert up projection
        {i}.down_proj.*  -- i-th expert down projection

    Non-quantized (FP16) checkpoints that use stacked 3-D parameters
    (``gate_up_proj`` / ``down_proj``) are handled by weight-loading
    remapping in the checkpoint loader, not here.
    """

    def __init__(self, config: ModelConfig, module_name: str) -> None:
        super().__init__()
        hidden = config.hidden_size
        inter = config.moe_intermediate_size
        experts = []
        for _ in range(config.num_experts):
            expert = nn.Module()
            expert.gate_proj = make_linear(config,
                                           hidden,
                                           inter,
                                           module_name=module_name)
            expert.up_proj = make_linear(config,
                                         hidden,
                                         inter,
                                         module_name=module_name)
            expert.down_proj = make_linear(config,
                                           inter,
                                           hidden,
                                           module_name=module_name)
            experts.append(expert)
        self._experts = nn.ModuleList(experts)

    def __getitem__(self, idx: int) -> nn.Module:
        return self._experts[idx]

    def __len__(self) -> int:
        return len(self._experts)

    def __iter__(self):
        return iter(self._experts)


def _stack_nvfp4_moe_input_scales(
        experts: Qwen3MoEExperts) -> "tuple[torch.Tensor, torch.Tensor]":
    """Collect the calibrated FC1 and FC2 activation scales."""
    fc1_scales = []
    fc2_scales = []
    for expert_idx, expert in enumerate(experts):
        gate_scale = expert.gate_proj.input_scale.detach().float().reshape(-1)
        up_scale = expert.up_proj.input_scale.detach().float().reshape(-1)
        down_scale = expert.down_proj.input_scale.detach().float().reshape(-1)
        if any(scale.numel() != 1
               for scale in (gate_scale, up_scale, down_scale)):
            raise ValueError("Nvfp4MoePlugin requires scalar input scales; "
                             f"expert {expert_idx} has non-scalar values")
        if not torch.equal(gate_scale, up_scale):
            raise ValueError("Nvfp4MoePlugin requires matching gate/up input "
                             f"scales for expert {expert_idx}")
        fc1_scales.append(gate_scale[0])
        fc2_scales.append(down_scale[0])
    return torch.stack(fc1_scales), torch.stack(fc2_scales)


# ---------------------------------------------------------------------------
# Sparse MoE block
# ---------------------------------------------------------------------------

# TRT-native MoE: MoEActType {kNONE=0, kSILU=1} (NvInfer.h); quantization_mode
# 0 = unquantized. Only kSILU (gated SwiGLU) is expressible today.
_TRT_MOE_ACT_SILU = 1
_TRT_MOE_QUANT_NONE = 0


class Qwen3SparseMoeBlock(nn.Module):
    """Sparse MoE block exported via a fused TRT MoE plugin.

    In the ONNX graph this produces:
    - One ``MatMul`` for the gate (router) linear -- traced from ``gate_linear``
    - One fused MoE plugin node for softmax + topk + expert GEMMs

    Weight loading: ``gate`` (Qwen3MoERouter) and ``experts`` (Qwen3MoEExperts as
    nn.ModuleList) hold per-expert parameters for checkpoint loading.
    After loading, :meth:`_prepare_moe_weights` (called by the repacking
    pass) extracts GPTQ weights, repacks to Marlin format, fuses gate+up
    projections, and registers the result as buffers on this module.

    Checkpoint sub-keys under ``mlp``:
        gate.*        -> Qwen3MoERouter (weight holder)
        experts.*     -> Qwen3MoEExperts (per-expert modules for loading)
    """

    def __init__(self, config: ModelConfig, module_prefix: str = "") -> None:
        super().__init__()
        experts_module_name = (f"{module_prefix}.experts"
                               if module_prefix else "experts")
        expert_quant_type = module_quant_type(experts_module_name, config)
        supported_quant_types = (
            QUANT_FP16,
            QUANT_INT4_AWQ_MODELOPT,
            QUANT_INT4_GPTQ,
            QUANT_NVFP4,
            QUANT_NVFP4_A16,
        )
        if expert_quant_type not in supported_quant_types:
            raise ValueError(
                "Qwen3 MoE has no W8A16 or FP8 expert implementation; "
                "supported expert execution is FP16, INT4 W4A16, NVFP4 "
                "W4A4, or NVFP4 W4A16")
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.zero_point_offset = config.quant.gptq_zero_point_offset
        self.group_size = module_quant_group_size(experts_module_name, config)
        self._use_nvfp4_moe = expert_quant_type == QUANT_NVFP4
        self._use_nvfp4_a16_moe = expert_quant_type == QUANT_NVFP4_A16
        self._use_fp16_moe = expert_quant_type == QUANT_FP16
        # --trt-native-moe: experts through TRT_MoE instead of a plugin. FP16
        # experts take the FP16 form, NVFP4 experts the W4A8 form (or W4A4,
        # quantization_mode 2, with --trt-native-moe-w4a4); INT4 and
        # NVFP4-A16 experts have no TRT_MoE form and are rejected.
        self._use_native_moe = is_trt_native_moe_enabled()
        self._use_native_moe_w4a4 = is_trt_native_moe_w4a4_enabled()
        if self._use_native_moe_w4a4 and not self._use_nvfp4_moe:
            raise ValueError(
                "--trt-native-moe-w4a4 requested but the experts are "
                f"{expert_quant_type!r}; the W4A4 TRT_MoE form needs ModelOpt "
                "NVFP4 experts with quantized activations (FP16 experts have "
                "only the unquantized form, and --no-quantize-activations "
                "turns NVFP4 into NVFP4-A16, which has no TRT_MoE form). Drop "
                "--trt-native-moe-w4a4 to export the W4A8 form, or keep the "
                "activations quantized.")
        if self._use_native_moe and not (self._use_fp16_moe
                                         or self._use_nvfp4_moe):
            raise ValueError(
                "--trt-native-moe requested but the experts are "
                f"{expert_quant_type!r}; only FP16 and NVFP4 experts have a "
                "TRT_MoE form (INT4 and NVFP4-A16 experts do not). Drop the "
                "flag to export the plugin path.")
        # All paths compute the same SwiGLU expert FFN; the integer is just
        # each plugin's own enum for it. Int4MoePlugin names the elementwise
        # activation ("SiLU"=0), Nvfp4MoePlugin/Fp16MoePlugin/Nvfp4A16MoePlugin
        # name the fused pattern ("SwiGLU"=2). See the C++ plugin headers.
        if self._use_nvfp4_moe or self._use_nvfp4_a16_moe or self._use_fp16_moe:
            self.activation_type = _NVFP4_ACTIVATION_SWIGLU
        else:
            self.activation_type = _INT4_ACTIVATION_SILU
        self._padded_moe_intermediate_size = (
            ((self.moe_intermediate_size + 127) // 128) *
            128 if self._use_nvfp4_a16_moe else self.moe_intermediate_size)
        # Plugin attributes consumed by ``Nvfp4MoePlugin``.
        self.backend = _NVFP4_MOE_BACKEND_AUTO
        self.io_dtype = _NVFP4_MOE_IO_DTYPE_FP16
        self.max_routed_rows = _NVFP4_MOE_MAX_ROUTED_ROWS_AUTO
        self.gate = Qwen3MoERouter(config)
        self.experts = Qwen3MoEExperts(config, experts_module_name)
        if self._use_nvfp4_a16_moe:
            # Routed experts are stacked into Nvfp4A16MoePlugin; keep the
            # dense Marlin repack from consuming their raw buffers.
            for expert in self.experts:
                expert.gate_proj._skip_dense_a16_repack = True
                expert.up_proj._skip_dense_a16_repack = True
                expert.down_proj._skip_dense_a16_repack = True

        # Qwen3-Omni MoE Talker (and any HF MoE variant that ships a per-layer
        # shared expert) adds a parallel SwiGLU MLP with a learned scalar gate.
        # Layout:
        #   y = routed_moe(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)
        # where shared_expert is a standard gate/up/down SwiGLU MLP of width
        # ``moe_shared_expert_intermediate_size`` and ``shared_expert_gate`` is
        # a single-output Linear.  ``make_linear`` picks NVFP4Linear when the
        # checkpoint is NVFP4 so the existing dense-NVFP4 inference path
        # (DequantizeLinear + MatMul) handles it -- no MoE plugin involvement.
        shared_inter = config.moe_shared_expert_intermediate_size
        self._has_shared_expert = shared_inter > 0
        if self._has_shared_expert:
            # Quant lookup keys are layer-prefixed ("layers.0.mlp.shared_expert.*").
            prefix = module_prefix + "." if module_prefix else ""
            self.shared_expert = nn.Module()
            self.shared_expert.gate_proj = make_linear(
                config,
                self.hidden_size,
                shared_inter,
                module_name=f"{prefix}shared_expert.gate_proj")
            self.shared_expert.up_proj = make_linear(
                config,
                self.hidden_size,
                shared_inter,
                module_name=f"{prefix}shared_expert.up_proj")
            self.shared_expert.down_proj = make_linear(
                config,
                shared_inter,
                self.hidden_size,
                module_name=f"{prefix}shared_expert.down_proj")
            self.shared_expert_gate = nn.Linear(self.hidden_size,
                                                1,
                                                bias=False,
                                                dtype=torch.float16)

    def _prepare_moe_weights(self) -> None:
        """Extract expert weights and prepare plugin buffers.

        Called by :func:`~checkpoint.repacking._stack_moe_experts` BEFORE
        regular GPTQ repacking.  After extraction, per-expert ``qweight``
        buffers are set to ``None`` so ``_repack_gptq_weights`` skips them.
        """
        if self._use_native_moe:
            self._prepare_native_moe_weights()
            return
        if self._use_fp16_moe:
            self._prepare_fp16_moe_weights()
            return
        if self._use_nvfp4_a16_moe:
            self._prepare_nvfp4_a16_moe_weights()
            return
        if self._use_nvfp4_moe:
            self._prepare_nvfp4_moe_weights()
            return

        from ...checkpoint.repacking import (_extract_awq_for_marlin,
                                             _extract_gptq_for_marlin,
                                             pack_int4_awq_marlin)
        from ..linear import ModelOptAWQPrepackedLinear

        # Detect ckpt format by inspecting the first expert. Both AWQ and GPTQ
        # produce ([N,K] int16 nibbles, [N,G] fp16 scales) — same Marlin pack
        # downstream. AWQ folds ``pre_quant_scale`` into the weight; GPTQ
        # applies zero_point_offset.
        first_proj = self.experts[0].gate_proj
        is_awq_moe = isinstance(first_proj, ModelOptAWQPrepackedLinear)

        def _extract(proj):
            if is_awq_moe:
                return _extract_awq_for_marlin(proj, self.group_size)
            return _extract_gptq_for_marlin(proj, self.group_size,
                                            self.zero_point_offset)

        # Promote Qwen3MoERouter weight -> nn.Linear for standard MatMul trace.
        self.gate_linear = nn.Linear(self.hidden_size,
                                     self.num_experts,
                                     bias=False,
                                     dtype=torch.float16)
        self.gate_linear.weight.data = self.gate.weight.data

        # Extract per-expert quantized weights -> [N, K] int16 + [N, groups] fp16.
        gate_up_weights_list = []
        gate_up_scales_list = []
        down_weights_list = []
        down_scales_list = []

        for expert in self.experts:
            gw, gs = _extract(expert.gate_proj)
            uw, us = _extract(expert.up_proj)
            gate_up_weights_list.append(torch.cat([gw, uw], dim=0))
            gate_up_scales_list.append(torch.cat([gs, us], dim=0))

            dw, ds = _extract(expert.down_proj)
            down_weights_list.append(dw)
            down_scales_list.append(ds)

        # Stack [E, N, K] and Marlin-pack.
        gate_up_w = torch.stack(gate_up_weights_list, dim=0)
        gate_up_s = torch.stack(gate_up_scales_list, dim=0)
        down_w = torch.stack(down_weights_list, dim=0)
        down_s = torch.stack(down_scales_list, dim=0)

        gu_marlin_w, gu_marlin_s = pack_int4_awq_marlin(
            gate_up_w, gate_up_s, self.group_size)
        dn_marlin_w, dn_marlin_s = pack_int4_awq_marlin(
            down_w, down_s, self.group_size)

        # Store as int8 (Marlin int32 viewed as int8).
        self.register_buffer("fc_gate_up_qweights",
                             gu_marlin_w.view(torch.int8).contiguous())
        self.register_buffer("fc_gate_up_scales", gu_marlin_s.contiguous())
        self.register_buffer("fc_down_qweights",
                             dn_marlin_w.view(torch.int8).contiguous())
        self.register_buffer("fc_down_scales", dn_marlin_s.contiguous())

        logger.info(
            "Marlin-packed %d experts: gate_up_qw %s, down_qw %s",
            self.num_experts,
            list(self.fc_gate_up_qweights.shape),
            list(self.fc_down_qweights.shape),
        )

        # Discard per-expert modules — weights are now in the stacked Marlin
        # buffers above.  This also prevents _repack_gptq_weights from seeing
        # the (now-consumed) per-expert qweight buffers.
        self.experts = nn.ModuleList()

    def _prepare_native_moe_weights(self) -> None:
        """Stack the experts into TRT-native ``TRT_MoE`` buffers.

        FP16/BF16 experts: the HF weights transposed (NvInfer.h
        ``IMoELayer::setGatedWeights``): ``fc_gate``/``fc_up`` are
        ``[E, H, I]``, ``fc_down`` is ``[E, I, H]``. The plugin's
        ``_prepare_fp16_moe_weights`` cannot be reused: that layout fuses
        gate+up into one FC1, is N-major, interleaves up/gate in 64-row groups
        and pads I to 128.

        NVFP4 experts (W4A8): the checkpoint's FP4 codes are re-stacked as
        stored. The plugin repack keeps the FP4 bytes and E4M3 block scales
        verbatim too and applies ``weight_scale_2`` as a per-expert fp32
        alpha; ``TRT_MoE`` has no alpha and ``TRT_BlockDequantize`` takes only
        an fp16 scale for FP4 weights, so here the E4M3 per-16 block scale is
        decoded and folded with ``weight_scale_2`` into one fp16 scale (see
        :func:`~tensorrt_edgellm.models.ops.nvfp4_dequantize_moe`).
        Activations: the plugin quantizes them to NVFP4 per token in the
        kernel (W4A4); this arm uses static per-tensor FP8 Q/DQ (W4A8) from
        ModelOpt's calibrated ``<proj>.input_scale = amax / (6 * 448)`` (FP8
        E4M3 wants ``amax / 448``, hence the ``* 6``). The FC1 input is shared
        by all experts, so the LARGEST calibrated scale is used (a smaller one
        would saturate the experts with a larger amax); FC2 keeps a per-expert
        scale.

        NVFP4 experts with ``--trt-native-moe-w4a4`` (W4A4, ``TRT_MoE``
        quantization_mode 2): the FP4 codes are re-stacked the same way but
        the scales stay separate, the checkpoint's E4M3 block scales verbatim
        (an fp8 buffer) plus ``weight_scale_2`` as a per-expert fp16 global
        (see :meth:`_prepare_native_moe_w4a4_weights`). Activations are
        quantized dynamically to NVFP4 (blocks of 16, FP8 block scales) like
        the plugin does, with ModelOpt's ``input_scale`` used VERBATIM as the
        second-level scale: the max over the experts for the shared FC1 input
        and per expert for the FC2 input. No ``* 6`` there; that factor
        belongs to the FP8 form.
        """
        self.gate_linear = nn.Linear(self.hidden_size,
                                     self.num_experts,
                                     bias=False,
                                     dtype=torch.float16)
        self.gate_linear.weight.data = self.gate.weight.data
        device = self.gate.weight.device

        if self._use_fp16_moe:

            def _stack_t(proj_name):
                # nn.Linear weight is [out, in]; the native op wants [in, out].
                tensors = [
                    getattr(expert, proj_name).weight.data.t()
                    for expert in self.experts
                ]
                return torch.stack(tensors,
                                   dim=0).to(device=device,
                                             dtype=torch.float16).contiguous()

            self.register_buffer("fc_gate_weights", _stack_t("gate_proj"))
            self.register_buffer("fc_up_weights", _stack_t("up_proj"))
            self.register_buffer("fc_down_weights", _stack_t("down_proj"))
            logger.info(
                "TRT_MoE FP16: %d experts stacked (transposed layout): "
                "gate %s, up %s, down %s", self.num_experts,
                list(self.fc_gate_weights.shape),
                list(self.fc_up_weights.shape),
                list(self.fc_down_weights.shape))
        elif self._use_native_moe_w4a4:
            self._prepare_native_moe_w4a4_weights(device)
        else:
            from ...checkpoint.repacking import stack_nvfp4_moe_projection

            # ``nvfp4_<proj>_weight``: packed FP4 codes [E, out, in // 2]
            # (int8); ``nvfp4_<proj>_block_scale``: fp16 [E, out, in // 16],
            # already multiplied by the per-expert ``weight_scale_2``.
            for proj in ("gate", "up", "down"):
                weight, block_scale = stack_nvfp4_moe_projection(
                    self.experts, f"{proj}_proj")
                self.register_buffer(f"nvfp4_{proj}_weight",
                                     weight.to(device).contiguous())
                self.register_buffer(f"nvfp4_{proj}_block_scale",
                                     block_scale.to(device).contiguous())

            # Calibrated activation scales -> FP8 (amax/448 = input_scale*6).
            input_gs, down_gs = _stack_nvfp4_moe_input_scales(self.experts)
            act_scale = input_gs.max()
            if not torch.allclose(
                    input_gs, act_scale.expand_as(input_gs), rtol=1e-6,
                    atol=0.0):
                logger.warning(
                    "TRT-native W4A8 MoE: per-expert FC1 input scales differ "
                    "(min %.4g, max %.4g); using the max so the shared FP8 "
                    "activation quantization saturates no expert",
                    float(input_gs.min()), float(act_scale))
            self.register_buffer("native_act_scale",
                                 (act_scale * 6.0).to(device=device,
                                                      dtype=torch.float16))
            self.register_buffer("native_down_act_scale", (down_gs * 6.0).to(
                device=device, dtype=torch.float32).contiguous())

            # Biases are all-or-none and the activation scale must sit at
            # input 9, so pass explicit zeros (a numeric no-op).
            inter = self.moe_intermediate_size
            zeros = lambda n: torch.zeros(
                self.num_experts, n, dtype=torch.float16, device=device)
            self.register_buffer("native_gate_bias", zeros(inter))
            self.register_buffer("native_up_bias", zeros(inter))
            self.register_buffer("native_down_bias", zeros(self.hidden_size))

            logger.info(
                "TRT_MoE W4A8: %d NVFP4 experts; act_scale=%.6g, "
                "down_act_scale[0:2]=%s", self.num_experts,
                float(self.native_act_scale),
                self.native_down_act_scale[:2].tolist())
        self.experts = nn.ModuleList()

    def _prepare_native_moe_w4a4_weights(self, device: torch.device) -> None:
        """Stack NVFP4 experts for the W4A4 ``TRT_MoE`` form (quantization_mode 2).

        Weights keep separate scales: ``nvfp4_<proj>_weight`` (FP4 codes),
        ``nvfp4_<proj>_block_scale_fp8`` (the E4M3 block scales verbatim) and
        ``nvfp4_<proj>_global_scale`` (fp16 ``weight_scale_2`` per expert),
        see :func:`~tensorrt_edgellm.checkpoint.repacking.stack_nvfp4_moe_projection_with_global_scale`.
        The activation scales are ModelOpt's ``input_scale`` verbatim (no
        ``* 6``, which is the FP8 convention of the W4A8 form): the max over
        the experts for the shared FC1 input, rounded to an fp16-representable
        value so the fp32 DynQ scale and the fp16 DQ scale agree, and the
        per-expert FC2 scales as ``TRT_MoE`` input 9. The buffer names keep
        the ``nvfp4_<proj>_weight`` suffix the FP4 reinterpretation keys on
        and put "scale" into every scale buffer, which the export's dtype
        passes skip and keep.
        """
        from ...checkpoint.repacking import \
            stack_nvfp4_moe_projection_with_global_scale

        for proj in ("gate", "up", "down"):
            weight, block_scale, global_scale = (
                stack_nvfp4_moe_projection_with_global_scale(
                    self.experts, f"{proj}_proj"))
            self.register_buffer(f"nvfp4_{proj}_weight",
                                 weight.to(device).contiguous())
            self.register_buffer(f"nvfp4_{proj}_block_scale_fp8",
                                 block_scale.to(device).contiguous())
            self.register_buffer(f"nvfp4_{proj}_global_scale",
                                 global_scale.to(device).contiguous())

        input_gs, down_gs = _stack_nvfp4_moe_input_scales(self.experts)
        for name, scales in (("FC1", input_gs), ("FC2", down_gs)):
            if not bool(torch.isfinite(scales).all() and (scales > 0).all()):
                raise ValueError(
                    f"TRT-native W4A4 MoE: {name} input_scale must be finite "
                    f"and positive for every expert, got {scales.tolist()}")
        act_dblq = input_gs.max()
        if not torch.allclose(
                input_gs, act_dblq.expand_as(input_gs), rtol=1e-6, atol=0.0):
            logger.warning(
                "TRT-native W4A4 MoE: per-expert FC1 input scales differ "
                "(min %.4g, max %.4g); using the max so the shared NVFP4 "
                "block scales saturate for no expert", float(input_gs.min()),
                float(act_dblq))
        # fp32 copy for TRT_FP4DynamicQuantize, fp16 copy (a Cast in the
        # graph) for the block-scale DQ: one fp16-representable value.
        act_dblq = act_dblq.to(torch.float16).to(torch.float32)
        self.register_buffer("native_act_dblq_scale",
                             act_dblq.to(device=device).contiguous())
        self.register_buffer(
            "native_down_act_scale",
            down_gs.to(device=device, dtype=torch.float32).contiguous())

        # Biases are all-or-none and the activation scale must sit at
        # input 9, so pass explicit zeros (a numeric no-op).
        inter = self.moe_intermediate_size
        zeros = lambda n: torch.zeros(
            self.num_experts, n, dtype=torch.float16, device=device)
        self.register_buffer("native_gate_bias", zeros(inter))
        self.register_buffer("native_up_bias", zeros(inter))
        self.register_buffer("native_down_bias", zeros(self.hidden_size))

        logger.info(
            "TRT_MoE W4A4 (quantization_mode 2): %d NVFP4 experts; "
            "act_dblq_scale=%.6g, down_act_scale[0:2]=%s", self.num_experts,
            float(self.native_act_dblq_scale),
            self.native_down_act_scale[:2].tolist())

    def _native_moe_forward(self, hidden_states: torch.Tensor,
                            hidden_flat: torch.Tensor,
                            router_logits: torch.Tensor) -> torch.Tensor:
        """TRT-native (OOTB) MoE: routing in the graph, ``TRT_MoE`` for the experts.

        ``TRT_MoE`` takes PRE-COMPUTED routing (the plugins consume raw
        router_logits and route internally), so softmax + top-k + renormalisation
        happen here. Qwen3-30B-A3B ships ``norm_topk_prob=True`` and the plugin arm
        hardcodes ``norm_topk_prob=1``, so the renormalisation is load-bearing.

        IMoELayer needs rank-3 ``[B, S, *]`` tensors while the token-major
        export contract hands the block ``[T, H]`` (T = sum of the batch's
        query lengths at prefill, the active batch at decode; the same
        ``[T, H]`` the plugins take). The layer is defined per token with no
        cross-token reduction, so its batch/sequence split is layout only: wrap
        all T tokens as one sequence ``[1, T, H]`` / ``[1, T, K]`` and reshape
        the output back to the caller's shape. Holds for any batch size;
        validated at B=4 ragged (tests/test_cases/ragged_batch.json).
        """
        hidden_dim = hidden_flat.shape[-1]
        num_tokens = hidden_flat.shape[0]
        probs = torch.softmax(router_logits, dim=-1)
        sel_scores, sel_idx = torch.topk(probs, self.top_k, dim=-1)
        sel_scores = sel_scores / sel_scores.sum(dim=-1, keepdim=True)
        selected_experts = sel_idx.to(torch.int32).reshape(
            1, num_tokens, self.top_k)
        # scores stay FP32 on purpose -- FP16 breaks Myelin codegen.
        scores = sel_scores.to(torch.float32).reshape(1, num_tokens,
                                                      self.top_k)
        hidden3 = hidden_flat.reshape(1, num_tokens, hidden_dim)
        if self._use_native_moe_w4a4:
            # W4A4 (quantization_mode 2): NVFP4 dynamic Q/DQ of the hidden
            # states, NVFP4 weight DQ with per-expert global scales
            # ([E, out, in] -> [E, in, out] for TRT_MoE's layout), per-expert
            # fp32 FC2 dblq at input 9.
            hidden3 = nvfp4_act_qdq_moe(hidden3, self.native_act_dblq_scale)
            group = 16  # NVFP4 group size
            gate_w = nvfp4_dequantize_moe_with_global_scale(
                self.nvfp4_gate_weight, self.nvfp4_gate_block_scale_fp8,
                self.nvfp4_gate_global_scale, group).transpose(1, 2)
            up_w = nvfp4_dequantize_moe_with_global_scale(
                self.nvfp4_up_weight, self.nvfp4_up_block_scale_fp8,
                self.nvfp4_up_global_scale, group).transpose(1, 2)
            down_w = nvfp4_dequantize_moe_with_global_scale(
                self.nvfp4_down_weight, self.nvfp4_down_block_scale_fp8,
                self.nvfp4_down_global_scale, group).transpose(1, 2)
            routed = trt_moe_w4a4(
                hidden3,
                selected_experts,
                scores,
                gate_w,
                up_w,
                down_w,
                self.native_gate_bias,
                self.native_up_bias,
                self.native_down_bias,
                self.native_down_act_scale,
                _TRT_MOE_ACT_SILU,
                TRT_MOE_QUANT_DYNAMIC_DBLQ,
                TRT_DATATYPE_FP4,
                TRT_DATATYPE_FP8,  # dyn_q_output_scale_dtype: FP8 block scales
                1.0,  # swiglu_alpha
                0.0,  # swiglu_beta (no swiglu_limit: see trt::trt_moe_fp16)
            )
        elif self._use_nvfp4_moe:
            # FP8 activation Q/DQ with the calibrated per-tensor scale, NVFP4
            # weight DQ ([E, out, in] -> [E, in, out] for TRT_MoE's layout).
            hidden3 = fp8_dequantize(
                fp8_quantize(hidden3, self.native_act_scale),
                self.native_act_scale)
            group = 16  # NVFP4 group size
            gate_w = nvfp4_dequantize_moe(self.nvfp4_gate_weight,
                                          self.nvfp4_gate_block_scale,
                                          group).transpose(1, 2)
            up_w = nvfp4_dequantize_moe(self.nvfp4_up_weight,
                                        self.nvfp4_up_block_scale,
                                        group).transpose(1, 2)
            down_w = nvfp4_dequantize_moe(self.nvfp4_down_weight,
                                          self.nvfp4_down_block_scale,
                                          group).transpose(1, 2)
            routed = trt_moe_w4a8(
                hidden3,
                selected_experts,
                scores,
                gate_w,
                up_w,
                down_w,
                self.native_gate_bias,
                self.native_up_bias,
                self.native_down_bias,
                self.native_down_act_scale,
                _TRT_MOE_ACT_SILU,
                TRT_MOE_QUANT_STATIC,
                TRT_DATATYPE_FP8,
                1.0,  # swiglu_alpha
                0.0,  # swiglu_beta (no swiglu_limit: see trt::trt_moe_fp16)
            )
        else:
            routed = trt_moe_fp16(
                hidden3,
                selected_experts,
                scores,
                self.fc_gate_weights,
                self.fc_up_weights,
                self.fc_down_weights,
                _TRT_MOE_ACT_SILU,
                _TRT_MOE_QUANT_NONE,
                1.0,  # swiglu_alpha
                0.0,  # swiglu_beta (no swiglu_limit: see trt::trt_moe_fp16)
            )
        routed = routed.reshape(hidden_states.shape)
        if self._has_shared_expert:
            routed = routed + self._shared_expert_forward(hidden_states)
        return routed

    def _prepare_fp16_moe_weights(self) -> None:
        """Stack unquantized experts into ``Fp16MoePlugin`` FP16 buffers.

        FC1 uses the same 64-row up/gate interleave as the SM100/101/103/110
        ``Nvfp4MoePlugin`` split kernel (see
        kernelSrcs/f16_moe_cutedsl/README.md, "Fixed plugin contract"):
        ``[up0:64, gate0:64, up64:128, gate64:128, ...]`` along FC1_N.
        """
        self.gate_linear = nn.Linear(self.hidden_size,
                                     self.num_experts,
                                     bias=False,
                                     dtype=torch.float16)
        self.gate_linear.weight.data = self.gate.weight.data

        inter = self.moe_intermediate_size
        chunk_rows = 64
        if inter % chunk_rows != 0:
            raise ValueError(
                f"moe_inter_size ({inter}) must be a multiple of {chunk_rows} "
                "for the Fp16MoePlugin SwiGLU FC1 layout")
        n_chunks = inter // chunk_rows

        fc1_list = []
        fc2_list = []
        for expert in self.experts:
            gate_w = expert.gate_proj.weight.data.to(torch.float16)
            up_w = expert.up_proj.weight.data.to(torch.float16)
            down_w = expert.down_proj.weight.data.to(torch.float16)
            up_chunks = up_w.reshape(n_chunks, chunk_rows, self.hidden_size)
            gate_chunks = gate_w.reshape(n_chunks, chunk_rows,
                                         self.hidden_size)
            fc1_list.append(
                torch.stack([up_chunks, gate_chunks],
                            dim=1).reshape(2 * inter, self.hidden_size))
            fc2_list.append(down_w)

        device = self.gate.weight.device
        self.register_buffer(
            "fc1_weights",
            torch.stack(fc1_list, dim=0).to(device).contiguous())
        self.register_buffer(
            "fc2_weights",
            torch.stack(fc2_list, dim=0).to(device).contiguous())

        logger.info(
            "Fp16MoePlugin-packed %d Qwen3 experts (64-row interleave): "
            "fc1 %s, fc2 %s",
            self.num_experts,
            list(self.fc1_weights.shape),
            list(self.fc2_weights.shape),
        )

        self.experts = nn.ModuleList()

    def _prepare_nvfp4_moe_weights(self) -> None:
        """NVFP4 expert repack for ``Nvfp4MoePlugin``."""
        self._prepare_nvfp4_moe_weights_impl()

    def _prepare_nvfp4_moe_weights_impl(self) -> None:
        """Repack Qwen3 NVFP4 experts for the active NVFP4 MoE plugin.

        Passes the ModelOpt FP4 weights and FP8 block scales through verbatim
        (``weight_scale_2`` becomes the per-expert alpha) and emits the
        CuTeDSL 6D MMA scale layout the kernel expects. FC1 SwiGLU layout
        is selected from :func:`use_geforce_nvfp4_moe`:

        * SM100/101/103/110 ``Nvfp4MoePlugin`` -- 64-row up/gate interleave (default).
        * SM12x ``NvFP4MoEPluginGeforce`` -- plain ``[up_all, gate_all]``
          concat along the M axis.
        """
        from ...checkpoint.repacking import (
            NVFP4_MOE_INTERLEAVE_SIZE_ALIGNMENT,
            NVFP4_MOE_INTERMEDIATE_SIZE_ALIGNMENT,
            repack_nvfp4_gated_moe_experts)

        self.gate_linear = nn.Linear(self.hidden_size,
                                     self.num_experts,
                                     bias=False,
                                     dtype=torch.float16)
        self.gate_linear.weight.data = self.gate.weight.data

        input_global_scale, down_input_scale = \
            _stack_nvfp4_moe_input_scales(self.experts)

        use_geforce_plugin = use_geforce_nvfp4_moe()
        fc1_layout = "concat" if use_geforce_plugin else "interleave"
        moe_inter_size_alignment = (NVFP4_MOE_INTERMEDIATE_SIZE_ALIGNMENT
                                    if use_geforce_plugin else
                                    NVFP4_MOE_INTERLEAVE_SIZE_ALIGNMENT)
        (fc1_qweights, fc1_blocks_scale, fc1_alpha, fc2_qweights,
         fc2_blocks_scale, fc2_alpha) = repack_nvfp4_gated_moe_experts(
             self.experts,
             self.hidden_size,
             self.moe_intermediate_size,
             self.group_size,
             fc1_layout=fc1_layout,
             moe_inter_size_alignment=moe_inter_size_alignment)
        self._padded_moe_intermediate_size = int(fc2_qweights.shape[-1]) * 2

        device = self.gate.weight.device
        self.register_buffer("fc1_qweights",
                             fc1_qweights.to(device).contiguous())
        self.register_buffer("fc1_blocks_scale",
                             fc1_blocks_scale.to(device).contiguous())
        self.register_buffer("fc2_qweights",
                             fc2_qweights.to(device).contiguous())
        self.register_buffer("fc2_blocks_scale",
                             fc2_blocks_scale.to(device).contiguous())
        # Per-expert FP32 weight_scale_2, applied as alpha in the kernel
        # epilogue; the FP4 weights and FP8 block scales above are the
        # checkpoint bytes. Activation scales come from checkpoint calibration.
        self.register_buffer("fc1_alpha", fc1_alpha.to(device).contiguous())
        self.register_buffer("fc2_alpha", fc2_alpha.to(device).contiguous())
        self.register_buffer("input_global_scale",
                             input_global_scale.to(device).contiguous())
        self.register_buffer("down_input_scale",
                             down_input_scale.to(device).contiguous())
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.num_experts, dtype=torch.float32, device=device))

        logger.info(
            "%s (CuTeDSL)-packed %d Qwen3 experts (fc1_layout=%s): "
            "fc1_qw %s, fc2_qw %s",
            "NvFP4MoEPluginGeforce"
            if use_geforce_nvfp4_moe() else "Nvfp4MoePlugin",
            self.num_experts,
            fc1_layout,
            list(self.fc1_qweights.shape),
            list(self.fc2_qweights.shape),
        )

        self.experts = nn.ModuleList()

    def _prepare_nvfp4_a16_moe_weights(self) -> None:
        """Stack routed-expert NVFP4 (W4A16) weights for ``Nvfp4A16MoePlugin``."""
        from ...checkpoint.repacking import (
            nvfp4_a16_from_dense, repack_nvfp4_a16_marlin_gated_moe_experts)

        self.gate_linear = nn.Linear(self.hidden_size,
                                     self.num_experts,
                                     bias=False,
                                     dtype=torch.float16)
        self.gate_linear.weight.data = self.gate.weight.data

        def _source(proj: nn.Module):
            buffers = proj._buffers
            if buffers["weight"].dtype in (torch.float16, torch.bfloat16):
                # Block-scaled FP8 experts arrive dequantized (see
                # loader._fold_fp8_block_scales); W4A16 needs NVFP4 codes.
                return nvfp4_a16_from_dense(buffers["weight"], self.group_size)
            return (buffers["weight"], buffers["weight_scale"],
                    buffers["weight_scale_2"])

        gate_p, gate_s, gate_g = [], [], []
        up_p, up_s, up_g = [], [], []
        down_p, down_s, down_g = [], [], []
        for expert in self.experts:
            gate = _source(expert.gate_proj)
            gate_p.append(gate[0])
            gate_s.append(gate[1])
            gate_g.append(gate[2])

            up = _source(expert.up_proj)
            up_p.append(up[0])
            up_s.append(up[1])
            up_g.append(up[2])

            down = _source(expert.down_proj)
            down_p.append(down[0])
            down_s.append(down[1])
            down_g.append(down[2])
        (fc1_qweights, fc1_block_scales, fc1_global, fc2_qweights,
         fc2_block_scales,
         fc2_global) = (repack_nvfp4_a16_marlin_gated_moe_experts(
             gate_p, gate_s, gate_g, up_p, up_s, up_g, down_p, down_s, down_g,
             self._padded_moe_intermediate_size))

        device = self.gate.weight.device
        self.register_buffer("fc1_qweights",
                             fc1_qweights.to(device).contiguous())
        self.register_buffer("fc1_block_scales",
                             fc1_block_scales.to(device).contiguous())
        self.register_buffer("fc1_global_scales",
                             fc1_global.to(device).contiguous())
        self.register_buffer("fc2_qweights",
                             fc2_qweights.to(device).contiguous())
        self.register_buffer("fc2_block_scales",
                             fc2_block_scales.to(device).contiguous())
        self.register_buffer("fc2_global_scales",
                             fc2_global.to(device).contiguous())
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.num_experts, dtype=torch.float32, device=device))

        logger.info(
            "Nvfp4A16MoePlugin-packed %d Qwen3 SwiGLU experts: fc1_qw %s, fc2_qw %s",
            self.num_experts,
            list(self.fc1_qweights.shape),
            list(self.fc2_qweights.shape),
        )
        self.experts = nn.ModuleList()

    def _shared_expert_forward(self,
                               hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the per-layer shared expert MLP with sigmoid gating.

        Matches HF Qwen3-Omni MoE Talker:
            gate_logits = shared_expert_gate(x)                 # [..., 1]
            gated = sigmoid(gate_logits) * shared_expert(x)
            return gated
        ``shared_expert`` is a SwiGLU MLP (gate_proj / up_proj / down_proj)
        whose constituent linears live on the standard dense NVFP4 path.
        """
        gate = self.shared_expert.gate_proj(hidden_states)
        up = self.shared_expert.up_proj(hidden_states)
        intermediate = torch.nn.functional.silu(gate) * up
        shared_out = self.shared_expert.down_proj(intermediate)
        gate_logits = self.shared_expert_gate(hidden_states)
        return torch.sigmoid(gate_logits) * shared_out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_dim = hidden_states.shape[-1]
        hidden_flat = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate_linear(hidden_flat).float()
        if self._use_native_moe:
            return self._native_moe_forward(hidden_states, hidden_flat,
                                            router_logits)
        if self._use_fp16_moe:
            routed = fp16_moe_plugin(
                router_logits,
                hidden_states,
                self.fc1_weights,
                self.fc2_weights,
                self.num_experts,
                self.top_k,
                self.hidden_size,
                self.moe_intermediate_size,
                self.activation_type,
                1,
                self.max_routed_rows,
            )
            if self._has_shared_expert:
                routed = routed + self._shared_expert_forward(hidden_states)
            return routed
        if self._use_nvfp4_a16_moe:
            routed = nvfp4_a16_moe_plugin(
                router_logits,
                hidden_states,
                self.fc1_qweights,
                self.fc1_block_scales,
                self.fc1_global_scales,
                self.fc2_qweights,
                self.fc2_block_scales,
                self.fc2_global_scales,
                self.e_score_correction_bias,
                self.num_experts,
                self.top_k,
                self.hidden_size,
                self._padded_moe_intermediate_size,
                self.activation_type,
                _NVFP4_MOE_N_GROUP_FLAT,
                _NVFP4_MOE_TOPK_GROUP_FLAT,
                1,
                1.0,
                _NVFP4_ROUTING_MODE_SOFTMAX_TOPK,
                self.max_routed_rows,
            )
            if self._has_shared_expert:
                routed = routed + self._shared_expert_forward(hidden_states)
            return routed
        if self._use_nvfp4_moe:
            moe_op = (nvfp4_moe_plugin_geforce
                      if use_geforce_nvfp4_moe() else nvfp4_moe_plugin)
            routed = moe_op(
                router_logits,
                hidden_states,
                self.fc1_qweights,
                self.fc1_blocks_scale,
                self.fc1_alpha,
                self.fc2_qweights,
                self.fc2_blocks_scale,
                self.fc2_alpha,
                self.input_global_scale,
                self.down_input_scale,
                self.e_score_correction_bias,
                self.num_experts,
                self.top_k,
                self.hidden_size,
                self._padded_moe_intermediate_size,
                self.activation_type,
                _NVFP4_MOE_N_GROUP_FLAT,
                _NVFP4_MOE_TOPK_GROUP_FLAT,
                1,
                1.0,
                _NVFP4_ROUTING_MODE_SOFTMAX_TOPK,
                self.backend,
                self.io_dtype,
                self.max_routed_rows,
            )
            if self._has_shared_expert:
                routed = routed + self._shared_expert_forward(hidden_states)
            return routed
        routed = int4_moe_plugin(
            router_logits,
            hidden_states,
            self.fc_gate_up_qweights,
            self.fc_gate_up_scales,
            self.fc_down_qweights,
            self.fc_down_scales,
            self.num_experts,
            self.top_k,
            self.hidden_size,
            self.moe_intermediate_size,
            self.activation_type,
            self.group_size,
        )
        if self._has_shared_expert:
            routed = routed + self._shared_expert_forward(hidden_states)
        return routed


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Qwen3MoeDecoderLayer(nn.Module):
    """Single Qwen3 MoE decoder layer.

    Submodule names match checkpoint keys:
        self_attn, mlp, input_layernorm, post_attention_layernorm

    ``mlp`` is a :class:`Qwen3SparseMoeBlock` for MoE-designated layers
    (per :func:`_is_moe_layer`) and a dense :class:`MLP` otherwise.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Attention(config, layer_idx=layer_idx)
        if _is_moe_layer(config, layer_idx):
            self.mlp = Qwen3SparseMoeBlock(
                config, module_prefix=f"layers.{layer_idx}.mlp")
        else:
            self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        skip_softmax_prefill_scale: "torch.Tensor | None" = None,
        skip_softmax_decode_scale: "torch.Tensor | None" = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            past_key_value,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
            skip_softmax_prefill_scale=skip_softmax_prefill_scale,
            skip_softmax_decode_scale=skip_softmax_decode_scale,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value

    def forward_ragged(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        residual = hidden_states
        attn_output, present_key_value = self.self_attn.forward_ragged(
            self.input_layernorm(hidden_states), **kwargs)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class Qwen3MoeTransformer(nn.Module):
    """Full Qwen3 MoE decoder stack.

    Stored as ``model`` inside :class:`Qwen3MoeCausalLM` so parameter keys
    carry the ``model.`` prefix matching safetensors checkpoint keys.

    Submodules: ``embed_tokens``, ``layers``, ``norm``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # Decoder layer index whose OUTPUT is exposed via
        # ``last_pre_norm_hidden_states`` for the Qwen3-Omni Talker.
        # ``accept_hidden_layer`` follows HF's ``hidden_states[k]`` convention:
        # k=N means "after N decoder layers", so we stash hidden_states right
        # after ``layer_index == accept_hidden_layer - 1``.  Negative value
        # falls back to the legacy "last layer" behaviour.
        self.accept_hidden_layer: int = int(
            getattr(config, "accept_hidden_layer", -1))
        self.last_pre_norm_hidden_states: "torch.Tensor | None" = None
        # Tensor exposed to CausalLM wrapper when ``emit_hidden_states``
        # is enabled. Selection (set in ``forward``):
        #   * accept_hidden_layer >= 1 → pre-norm output of that decoder layer
        #     (Thinker → Talker; HF reads layer_k pre-norm).
        #   * accept_hidden_layer < 1  → post-final-norm output (= self.norm
        #     applied to layer-N output). Matches HF's Talker → CodePredictor
        #     past_hidden semantics (modeling_qwen3_omni_moe.py:3176, where
        #     ``hidden_states[0][-1]`` resolves to the post-norm tensor).
        self.emitted_hidden_states: "torch.Tensor | None" = None
        self.target_hidden_concat: "torch.Tensor | None" = None
        self.all_hidden_states: "Tuple[torch.Tensor, ...] | None" = None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        deepstack_embeds: Tuple[torch.Tensor, ...] = (),
        skip_softmax_prefill_scale: "torch.Tensor | None" = None,
        skip_softmax_decode_scale: "torch.Tensor | None" = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        hidden_states = inputs_embeds
        present_key_values_list: List[torch.Tensor] = []

        # ``accept_hidden_layer == k`` (k >= 1) means the runtime expects the
        # output of decoder layer (k-1) i.e. "the residual after k layers",
        # matching HF Qwen3-Omni's ``hidden_states[k]`` indexing convention.
        # Capture that tensor below; negative or 0 falls back to "last layer".
        target_layer = self.accept_hidden_layer
        captured: "torch.Tensor | None" = None

        for layer_index, layer in enumerate(self.layers):
            hidden_states, next_key_value = layer(
                hidden_states,
                past_key_values[layer_index],
                rope_rotary_cos_sin,
                context_lengths,
                kvcache_start_index,
                kv_page_table,
                skip_softmax_prefill_scale=skip_softmax_prefill_scale,
                skip_softmax_decode_scale=skip_softmax_decode_scale,
            )
            present_key_values_list.append(next_key_value)

            # Multimodal deepstack visual embedding (post-layer, first N layers).
            # Mirrors modeling_default.Transformer.forward; Qwen3-Omni MoE
            # Thinker needs deepstack injection for image understanding.
            if layer_index < len(deepstack_embeds):
                hidden_states = hidden_states + deepstack_embeds[layer_index]

            if target_layer >= 1 and layer_index == target_layer - 1:
                captured = hidden_states

        # Expose the captured layer-output for the Qwen3-Omni Thinker -> Talker
        # next-stage consumer. When ``target_layer`` is unset / negative, fall back to the
        # legacy behaviour of stashing the last decoder layer's output.  Plain
        # text generation ignores this attribute regardless.
        self.last_pre_norm_hidden_states = (captured if captured is not None
                                            else hidden_states)

        normed = self.norm(hidden_states)

        # Emitted tensor (consumed by Qwen3MoeCausalLM.forward when
        # emit_hidden_states=True).  See attribute docstring on
        # emitted_hidden_states in ``__init__`` for the selection rule.
        # Defensive bounds check: if accept_hidden_layer was set but exceeds
        # the actual layer count (e.g. Talker checkpoints inheriting Thinker's
        # value of 24 while having only 20 layers), ``captured`` stays None
        # and we fall back to the post-norm output rather than silently
        # emitting the last layer's pre-norm.
        if target_layer >= 1 and captured is not None:
            self.emitted_hidden_states = self.last_pre_norm_hidden_states
        else:
            self.emitted_hidden_states = normed

        return normed, tuple(present_key_values_list)

    def forward_ragged(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        deepstack_embeds: Tuple[torch.Tensor, ...] = (),
        output_hidden_states: bool = False,
        target_layer_ids: "List[int] | None" = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        hidden_states = inputs_embeds
        present_key_values = []
        target_layer = self.accept_hidden_layer
        captured: "torch.Tensor | None" = None
        all_hidden_states = []
        target_hidden_by_layer: dict[int, torch.Tensor] = {}
        target_layer_set = set(target_layer_ids or [])
        for layer_index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states, present_kv = layer.forward_ragged(
                hidden_states,
                past_key_value=past_key_values[layer_index],
                **kwargs)
            present_key_values.append(present_kv)
            if layer_index in target_layer_set:
                target_hidden_by_layer[layer_index] = hidden_states
            if layer_index < len(deepstack_embeds):
                hidden_states = hidden_states + deepstack_embeds[layer_index]
            if target_layer >= 1 and layer_index == target_layer - 1:
                captured = hidden_states

        self.last_pre_norm_hidden_states = (captured if captured is not None
                                            else hidden_states)
        self.target_hidden_concat = _concat_hidden_in_provider_order(
            target_hidden_by_layer, target_layer_ids)
        normed = self.norm(hidden_states)
        self.emitted_hidden_states = (self.last_pre_norm_hidden_states
                                      if target_layer >= 1
                                      and captured is not None else normed)
        if output_hidden_states:
            all_hidden_states.append(normed)
        self.all_hidden_states = (tuple(all_hidden_states)
                                  if output_hidden_states else None)
        return normed, tuple(present_key_values)


# ---------------------------------------------------------------------------
# CausalLM
# ---------------------------------------------------------------------------


class Qwen3MoeCausalLM(nn.Module):
    """Qwen3 MoE causal LM: Transformer + lm_head.

    The inner transformer is stored as attribute ``model`` so parameter keys
    carry the ``model.`` prefix matching checkpoint key prefixes.

    Subclasses can set the class attribute ``emit_hidden_states = True`` to
    add a second ``hidden_states`` ONNX output carrying the full-sequence
    tensor emitted on the additional ONNX output.  Selection depends on ``config.accept_hidden_layer``:

    * ``accept_hidden_layer >= 1``: pre-norm output of that decoder layer.
      Used by Qwen3-Omni MoE Thinker → Talker (HF picks
      ``outputs.hidden_states[k]`` with k>=1 = layer-(k-1) output, pre-norm).

    * ``accept_hidden_layer < 1`` (legacy default): post-final-norm output
      (= ``self.norm(last_layer_output)``).  Used by Qwen3-Omni MoE
      Talker → CodePredictor (HF picks
      ``outputs.hidden_states[0][-1]`` which resolves to the post-norm tensor
      in ``modeling_qwen3_omni_moe.py:3176``).
    """

    emit_hidden_states: bool = False

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3MoeTransformer(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    def tie_weights(self) -> None:
        """Clone embed_tokens.weight into lm_head.weight when tie_word_embeddings=True."""
        if not self.config.tie_word_embeddings:
            return
        if not isinstance(self.lm_head, FP16Linear):
            return
        embed_weight = self.model.embed_tokens.weight
        self.lm_head.weight = nn.Parameter(embed_weight.detach().clone(),
                                           requires_grad=False)

    def onnx_export_spec(self) -> OnnxSpec:
        """Return the unified token-major ONNX export contract."""
        tree_attention = (self.config.eagle_base
                          or getattr(self.config, "dflash_base", False)
                          or getattr(self.config, "jetspec_base", False)
                          or getattr(self.config, "dspark_base", False))
        return CausalLM._token_major_onnx_export_spec(
            self, tree_attention=tree_attention)

    def forward_ragged(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        positions: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        attention_sequence_lengths: torch.Tensor,
        state_indices: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        kv_page_table: torch.Tensor,
        logits_indices: torch.Tensor,
        deepstack_embeds: Tuple[torch.Tensor, ...] = (),
        skip_softmax_prefill_scale: "torch.Tensor | None" = None,
        skip_softmax_decode_scale: "torch.Tensor | None" = None,
        attention_position_ids: "torch.Tensor | None" = None,
        packed_attention_mask: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        valid_tree_counts: "torch.Tensor | None" = None,
    ) -> Tuple:
        hidden_states, present_key_values = self.model.forward_ragged(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin=rope_rotary_cos_sin,
            positions=positions,
            query_start_offsets=query_start_offsets,
            query_lengths=query_lengths,
            past_lengths=past_lengths,
            attention_sequence_lengths=attention_sequence_lengths,
            state_indices=state_indices,
            execution_phase_marker=execution_phase_marker,
            context_sequence_count_carrier=context_sequence_count_carrier,
            kv_page_table=kv_page_table,
            deepstack_embeds=deepstack_embeds,
            skip_softmax_prefill_scale=skip_softmax_prefill_scale,
            skip_softmax_decode_scale=skip_softmax_decode_scale,
            attention_position_ids=attention_position_ids,
            packed_attention_mask=packed_attention_mask,
            tree_parent_ids=tree_parent_ids,
            tree_depths=tree_depths,
            valid_tree_counts=valid_tree_counts,
            output_hidden_states=self.config.eagle_base,
            target_layer_ids=CausalLM._target_hidden_layer_ids(self))
        selected_hidden_states = torch.index_select(hidden_states, 0,
                                                    logits_indices)
        logits = self.lm_head(selected_hidden_states).to(torch.float32)
        logits = logits.reshape(-1, logits.shape[-1])
        emitted_hidden = CausalLM._ragged_emitted_hidden(self)
        return logits, emitted_hidden, present_key_values

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        last_token_ids: torch.Tensor,
        deepstack_embeds: Tuple[torch.Tensor, ...] = (),
        skip_softmax_prefill_scale: "torch.Tensor | None" = None,
        skip_softmax_decode_scale: "torch.Tensor | None" = None,
    ) -> Tuple:
        hidden_states, present_key_values = self.model(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
            deepstack_embeds,
            skip_softmax_prefill_scale=skip_softmax_prefill_scale,
            skip_softmax_decode_scale=skip_softmax_decode_scale,
        )
        # Select hidden states for specified token positions before lm_head.
        selected_hidden_states = torch.ops.trt.gather_nd(
            hidden_states, last_token_ids)

        logits = self.lm_head(selected_hidden_states).to(torch.float32)

        if self.emit_hidden_states:
            # Full-sequence hidden_states emitted to the next stage.
            #   * Thinker→Talker (accept_hidden_layer>=1): pre-norm layer-k output
            #   * Talker→CodePredictor (accept_hidden_layer<1): post-final-norm
            # Selection logic lives in ``Qwen3MoeTransformer.forward``.
            return logits, self.model.emitted_hidden_states, \
                present_key_values

        return logits, present_key_values
