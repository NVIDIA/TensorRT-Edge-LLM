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

import itertools
import logging
from typing import List, Tuple

import torch
import torch.nn as nn

from ...config import QUANT_NVFP4, ModelConfig
from ..default.modeling_default import (MLP, Attention, OnnxSpec, RMSNorm,
                                        _make_flat_wrapper)
from ..linear import FP16Linear, make_linear
from ..ops import (KV_PAGE_SIZE, int4_moe_plugin, nvfp4_moe_plugin,
                   nvfp4_moe_plugin_geforce, use_geforce_nvfp4_moe)

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
_BATCH_SIZE = 1
_SEQ_LEN = 1
_PAST_LEN = 1
_MAX_POS = 4096

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

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        inter = config.moe_intermediate_size
        experts = []
        for _ in range(config.num_experts):
            expert = nn.Module()
            expert.gate_proj = make_linear(config, hidden, inter)
            expert.up_proj = make_linear(config, hidden, inter)
            expert.down_proj = make_linear(config, inter, hidden)
            experts.append(expert)
        self._experts = nn.ModuleList(experts)

    def __getitem__(self, idx: int) -> nn.Module:
        return self._experts[idx]

    def __len__(self) -> int:
        return len(self._experts)

    def __iter__(self):
        return iter(self._experts)


# ---------------------------------------------------------------------------
# Sparse MoE block
# ---------------------------------------------------------------------------


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

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.group_size = config.quant.group_size
        self.zero_point_offset = config.quant.gptq_zero_point_offset
        self._use_nvfp4_moe = config.quant.quant_type == QUANT_NVFP4
        # ``activation_type`` integer is plugin-specific — ints carry
        # different meanings between Int4MoePlugin (SiLU=0) and
        # Nvfp4MoePlugin (SwiGLU=2). The per-path constants above
        # document each mapping; see the corresponding C++ plugin headers
        # for the source of truth.
        if not self._use_nvfp4_moe:
            self.activation_type = _INT4_ACTIVATION_SILU
        else:
            self.activation_type = _NVFP4_ACTIVATION_SWIGLU
        # Plugin attributes consumed by ``Nvfp4MoePlugin``.
        self.backend = _NVFP4_MOE_BACKEND_AUTO
        self.io_dtype = _NVFP4_MOE_IO_DTYPE_FP16
        self.max_routed_rows = _NVFP4_MOE_MAX_ROUTED_ROWS_AUTO
        self.gate = Qwen3MoERouter(config)
        self.experts = Qwen3MoEExperts(config)

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
            self.shared_expert = nn.Module()
            self.shared_expert.gate_proj = make_linear(
                config,
                self.hidden_size,
                shared_inter,
                module_name="shared_expert.gate_proj")
            self.shared_expert.up_proj = make_linear(
                config,
                self.hidden_size,
                shared_inter,
                module_name="shared_expert.up_proj")
            self.shared_expert.down_proj = make_linear(
                config,
                shared_inter,
                self.hidden_size,
                module_name="shared_expert.down_proj")
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
        if self._use_nvfp4_moe:
            self._prepare_nvfp4_moe_weights()
            return

        from ...checkpoint.repacking import (_extract_gptq_for_marlin,
                                             pack_int4_awq_marlin)

        # Promote Qwen3MoERouter weight -> nn.Linear for standard MatMul trace.
        self.gate_linear = nn.Linear(self.hidden_size,
                                     self.num_experts,
                                     bias=False,
                                     dtype=torch.float16)
        self.gate_linear.weight.data = self.gate.weight.data

        # Extract per-expert GPTQ weights -> [N, K] int16 + [N, groups] fp16.
        gate_up_weights_list = []
        gate_up_scales_list = []
        down_weights_list = []
        down_scales_list = []

        for expert in self.experts:
            gw, gs = _extract_gptq_for_marlin(expert.gate_proj,
                                              self.group_size,
                                              self.zero_point_offset)
            uw, us = _extract_gptq_for_marlin(expert.up_proj, self.group_size,
                                              self.zero_point_offset)
            gate_up_weights_list.append(torch.cat([gw, uw], dim=0))
            gate_up_scales_list.append(torch.cat([gs, us], dim=0))

            dw, ds = _extract_gptq_for_marlin(expert.down_proj,
                                              self.group_size,
                                              self.zero_point_offset)
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

    def _prepare_nvfp4_moe_weights(self) -> None:
        """NVFP4 expert repack for ``Nvfp4MoePlugin``."""
        self._prepare_nvfp4_moe_weights_impl()

    def _prepare_nvfp4_moe_weights_impl(self) -> None:
        """Repack Qwen3 NVFP4 experts for the active NVFP4 MoE plugin.

        Decodes ModelOpt NVFP4 to dense, rounds through BF16, and emits the
        CuTeDSL 6D MMA scale layout the kernel expects. FC1 SwiGLU layout
        is selected from :func:`use_geforce_nvfp4_moe`:

        * SM100/101/110 ``Nvfp4MoePlugin`` -- 64-row up/gate interleave (default).
        * SM12x ``NvFP4MoEPluginGeforce`` -- plain ``[up_all, gate_all]``
          concat along the M axis.
        """
        from ...checkpoint.repacking import repack_nvfp4_qwen3_moe_experts

        self.gate_linear = nn.Linear(self.hidden_size,
                                     self.num_experts,
                                     bias=False,
                                     dtype=torch.float16)
        self.gate_linear.weight.data = self.gate.weight.data

        fc1_layout = "concat" if use_geforce_nvfp4_moe() else "interleave"
        fc1_qweights, fc1_blocks_scale, fc2_qweights, fc2_blocks_scale = (
            repack_nvfp4_qwen3_moe_experts(self.experts,
                                           self.hidden_size,
                                           self.moe_intermediate_size,
                                           self.group_size,
                                           fc1_layout=fc1_layout))

        device = self.gate.weight.device
        self.register_buffer("fc1_qweights",
                             fc1_qweights.to(device).contiguous())
        self.register_buffer("fc1_blocks_scale",
                             fc1_blocks_scale.to(device).contiguous())
        self.register_buffer("fc2_qweights",
                             fc2_qweights.to(device).contiguous())
        self.register_buffer("fc2_blocks_scale",
                             fc2_blocks_scale.to(device).contiguous())
        # FP32 alpha + activation scale buffers: one per expert. Populated
        # by the calibration pipeline; default to ones so an uncalibrated
        # checkpoint still produces numerically reasonable outputs.
        self.register_buffer(
            "fc1_alpha",
            torch.ones(self.num_experts, dtype=torch.float32, device=device))
        self.register_buffer(
            "fc2_alpha",
            torch.ones(self.num_experts, dtype=torch.float32, device=device))
        self.register_buffer(
            "input_global_scale",
            torch.ones(self.num_experts, dtype=torch.float32, device=device))
        self.register_buffer(
            "down_input_scale",
            torch.ones(self.num_experts, dtype=torch.float32, device=device))
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
        batch, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate_linear(hidden_flat).float()
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
                self.moe_intermediate_size,
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
            self.mlp = Qwen3SparseMoeBlock(config)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            past_key_value,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
        )
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

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        deepstack_embeds: Tuple[torch.Tensor, ...] = (),
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
        """Return all model-specific parameters needed for ONNX export."""
        config = self.config
        Na = config.num_hidden_layers
        # Multimodal MoE Thinker variants (e.g. Qwen3-Omni MoE) inject
        # ``num_deepstack_features`` visual embeddings post-layer at the
        # first N layers; match modeling_default.Transformer.
        Nd = config.num_deepstack_features
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        dtype16 = torch.float16
        batch_size, seq_len, past_len, max_pos = (_BATCH_SIZE, _SEQ_LEN,
                                                  _PAST_LEN, _MAX_POS)

        inputs_embeds = torch.zeros(batch_size,
                                    seq_len,
                                    config.hidden_size,
                                    dtype=dtype16,
                                    device=device)
        kv_dtype = (torch.float8_e4m3fn
                    if config.quant.kv_cache_quant == "fp8" else dtype16)
        # Paged KV pool binding: [2, num_pages, KV_PAGE_SIZE, num_kv_heads, head_dim].
        # num_pages is a dummy placeholder for export; the builder sets the real fixed
        # value (see llmBuilder.cpp setupKVCacheProfiles).
        past_key_values_list: List[torch.Tensor] = [
            torch.zeros(2,
                        1,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        config.head_dim,
                        dtype=kv_dtype,
                        device=device) for _ in range(Na)
        ]
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        rope_rotary_cos_sin = torch.zeros(batch_size,
                                          max_pos,
                                          rotary_dim,
                                          dtype=torch.float32,
                                          device=device)
        context_lengths = torch.zeros(batch_size,
                                      dtype=torch.int32,
                                      device=device)
        kvcache_start_index = torch.zeros(batch_size,
                                          dtype=torch.int32,
                                          device=device)
        kv_page_table = torch.zeros(batch_size,
                                    2,
                                    1,
                                    dtype=torch.int32,
                                    device=device)
        last_token_ids = torch.zeros(batch_size,
                                     1,
                                     dtype=torch.int64,
                                     device=device)

        deepstack_embeds_list: List[torch.Tensor] = [
            torch.zeros(batch_size,
                        seq_len,
                        config.hidden_size,
                        dtype=dtype16,
                        device=device) for _ in range(Nd)
        ]

        args = (inputs_embeds, *past_key_values_list, rope_rotary_cos_sin,
                context_lengths, kvcache_start_index, kv_page_table,
                last_token_ids, *deepstack_embeds_list)

        input_names = (
            ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
                "rope_rotary_cos_sin", "context_lengths",
                "kvcache_start_index", "kv_page_table", "last_token_ids"
            ] + [f"deepstack_embeds_{i}" for i in range(Nd)])
        output_names = (["logits"] +
                        [f"present_key_values_{i}" for i in range(Na)])
        if self.emit_hidden_states:
            output_names = (["logits", "hidden_states"] +
                            [f"present_key_values_{i}" for i in range(Na)])

        batch = torch.export.Dim("batch", min=1, max=256)
        seq = torch.export.Dim("seq_len", min=1, max=32768)
        pos = torch.export.Dim("max_pos", min=1, max=32768)
        rope_batch = torch.export.Dim("rope_batch", min=1, max=256)
        kv_batch = torch.export.Dim("kv_batch", min=1, max=256)
        page_batch = torch.export.Dim("page_batch", min=1, max=256)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        num_pages = torch.export.Dim("num_pages", min=1, max=1048576)

        all_shapes: list = [{0: batch, 1: seq}]  # inputs_embeds
        for _ in range(Na):
            all_shapes.append({1:
                               num_pages})  # past_key_values_i (pool-shaped)
        all_shapes.append({0: rope_batch, 1: pos})  # rope_rotary_cos_sin
        all_shapes.append({0: batch})  # context_lengths
        all_shapes.append({0: kv_batch})  # kvcache_start_index
        all_shapes.append({0: page_batch, 2: max_pages})  # kv_page_table
        all_shapes.append({0: batch})  # last_token_ids
        for _ in range(Nd):
            all_shapes.append({0: batch, 1: seq})  # deepstack_embeds_i

        wrapped = _make_flat_wrapper(
            self, Na, Nd, emit_hidden_states=self.emit_hidden_states)
        wrapped.eval()

        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=all_shapes)

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
    ) -> Tuple:
        hidden_states, present_key_values = self.model(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
            deepstack_embeds,
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
