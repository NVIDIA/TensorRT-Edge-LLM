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
NemotronH hybrid causal LM (Mamba2 SSM + MLP + Attention + MoE).

Checkpoint key structure
------------------------
backbone.embeddings.weight                     - token embedding
backbone.layers.{i}.norm.weight                - pre-mixer RMSNorm (all layer types)
backbone.layers.{i}.mixer.*                    - mixer (type depends on layer)
  Attention  : q_proj, k_proj, v_proj, o_proj
  Mamba2 SSM : in_proj, out_proj, conv1d.{weight,bias}, A_log, D, dt_bias, norm.weight
  MLP        : up_proj, down_proj
  MoE        : gate.{weight,e_score_correction_bias},
               experts.{j}.{up_proj,down_proj}, shared_experts.{up_proj,down_proj}
backbone.norm_f.weight                         - final RMSNorm
lm_head.weight                                 - output projection (FP16, tied or standalone)

Layer type pattern is read from ``hybrid_override_pattern`` in config.json:
  'M' -> LAYER_MAMBA   '*' -> LAYER_ATTN   '-' -> LAYER_MLP   'E' -> LAYER_MOE

Forward-pass conventions
------------------------
``NemotronHCausalLM.forward``:

    inputs_embeds        [batch, seq_len, hidden_size]              float16
    past_key_values      tuple of [2, num_pages, page_size, num_kv_heads, head_dim] per attn-layer
    rope_rotary_cos_sin  [batch, max_pos, rotary_dim]               float32
    context_lengths      [batch]                                    int32
    kvcache_start_index  [batch]                                    int32
    kv_page_table        [batch, 2, max_pages_per_seq]              int32
    last_token_ids       [batch, num_tokens]                        int64
    conv_states          tuple of [batch, conv_dim, conv_kernel-1] per mamba-layer
    ssm_states           tuple of [batch, num_heads, head_dim, ssm_state] per mamba-layer
    ──────────────────────────────────────────────────────────────────────
    -> logits             [batch, num_tokens, vocab_size]            float32
    -> present_key_values tuple of updated KV caches per attn-layer
    -> present_conv_states tuple of updated conv states per mamba-layer
    -> present_ssm_states  tuple of updated SSM states per mamba-layer
"""

import itertools
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import (LAYER_ATTN, LAYER_MAMBA, LAYER_MLP, LAYER_MOE,
                       QUANT_NVFP4_A16, MambaConfig, ModelConfig)
from ..default.modeling_default import OnnxSpec
from ..linear import FP16Linear, make_linear
from ..ops import (KV_PAGE_SIZE, attention_plugin, causal_conv1d,
                   causal_conv1d_with_intermediate, nvfp4_a16_moe_plugin,
                   nvfp4_moe_plugin, nvfp4_moe_plugin_geforce,
                   update_ssm_state, update_ssm_state_with_intermediate,
                   use_geforce_nvfp4_moe)

_NVFP4_ACTIVATION_RELU2 = 4
_NVFP4_ROUTING_MODE_SIGMOID_GROUP_TOPK = 1
_NVFP4_MOE_BACKEND_AUTO = 0
_NVFP4_MOE_IO_DTYPE_FP16 = 1
_NVFP4_MOE_MAX_ROUTED_ROWS_AUTO = 0


class RMSNorm(nn.Module):
    """RMSNorm for hybrid Mamba models — primitive decomposed ops.

    Uses explicit Pow+ReduceMean+Rsqrt+Mul ops (same as the default/Qwen
    RMSNorm) so the ONNX graph contains only standard ops that every TRT
    version can parse.  The explicit FP32→FP16 cast at the end creates an
    ONNX partition boundary so TRT splits the ForeignNode cleanly at the
    Mamba plugin inputs (same technique used by ``_gated_rmsnorm`` below).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size,
                                              dtype=torch.float16))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        hidden_states = hidden_states.to(input_dtype)
        return self.weight.to(input_dtype) * hidden_states


# ---------------------------------------------------------------------------
# Mamba-aware flat wrapper (NemotronH-specific)
# ---------------------------------------------------------------------------


def _make_flat_wrapper_mamba(model: nn.Module,
                             Na: int,
                             Nm: int,
                             mtp_base: bool = False,
                             dflash_base: bool = False) -> nn.Module:
    """Build an explicit flat forward wrapper for hybrid Mamba+Attention models.

    Extends the transformer wrapper with ``conv_state_i`` and ``ssm_state_i``
    inputs and ``present_conv_state_i`` / ``present_recurrent_state_i`` outputs.

    Using ``*flat_args`` in ``forward`` triggers a PyTorch 2.10 bug where the
    guard code generated by the non-strict exporter accesses the variadic tuple
    by index during decomposition, causing an ``IndexError``.  An explicit,
    named parameter list avoids this entirely.

    ``Na`` = number of attention layers, ``Nm`` = number of Mamba layers. When
    ``mtp_base`` is set, the wrapper adds the tree-attention / spec-verify
    inputs and returns the extra ``hidden_states`` + per-token intermediate
    conv/recurrent state outputs consumed by the MTP draft runtime.
    """
    param_names: List[str] = (
        ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
            "rope_rotary_cos_sin", "context_lengths", "kvcache_start_index",
            "kv_page_table", "last_token_ids"
        ] + [f"conv_state_{i}"
             for i in range(Nm)] + [f"recurrent_state_{i}" for i in range(Nm)])
    spec_base = mtp_base or dflash_base
    if spec_base:
        param_names += [
            "attention_pos_id", "attention_mask", "spec_verify_phase_marker"
        ]

    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(Na))) if Na else "()"
    conv_tuple = "({},)".format(", ".join(f"conv_state_{i}"
                                          for i in range(Nm))) if Nm else "()"
    ssm_tuple = "({},)".format(", ".join(f"recurrent_state_{i}"
                                         for i in range(Nm))) if Nm else "()"

    if spec_base:
        second = "dflash_hidden_concat" if dflash_base else "hidden_states"
        body = (
            f"    (logits, {second}, present_key_values, "
            f"present_conv_states, present_ssm_states, "
            f"intermediate_conv_states, replay_da_states, replay_u_states, "
            f"replay_b_states) = self._model(\n"
            f"        inputs_embeds, {past_kv_tuple}, rope_rotary_cos_sin, "
            f"context_lengths, kvcache_start_index, kv_page_table, "
            f"last_token_ids,\n"
            f"        {conv_tuple}, {ssm_tuple},\n"
            f"        attention_pos_id=attention_pos_id, "
            f"attention_mask=attention_mask, "
            f"spec_verify_phase_marker=spec_verify_phase_marker)\n"
            f"    return ((logits, {second}) + tuple(present_key_values)\n"
            f"            + tuple(present_conv_states) "
            f"+ tuple(present_ssm_states)\n"
            f"            + tuple(intermediate_conv_states) "
            f"+ tuple(replay_da_states) + tuple(replay_u_states) "
            f"+ tuple(replay_b_states))\n")
    else:
        body = (
            f"    logits, present_key_values, present_conv_states, "
            f"present_ssm_states = self._model(\n"
            f"        inputs_embeds, {past_kv_tuple}, rope_rotary_cos_sin, "
            f"context_lengths, kvcache_start_index, kv_page_table, "
            f"last_token_ids,\n"
            f"        {conv_tuple}, {ssm_tuple})\n"
            f"    return ((logits,) + tuple(present_key_values)\n"
            f"            + tuple(present_conv_states) + tuple(present_ssm_states))\n"
        )

    src = "def _forward(self, {}):\n{}".format(", ".join(param_names), body)
    globs: dict = {}
    exec(src, globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


__all__ = [
    "Conv1dBuffers",
    "MambaMixer",
    "NemotronHMLP",
    "NemotronHMoEMLP",
    "NemotronHTopkRouter",
    "NemotronHAttentionMixer",
    "NemotronHDecoderLayer",
    "NemotronHBackbone",
    "NemotronHCausalLM",
]

# ---------------------------------------------------------------------------
# Conv1dBuffers
# ---------------------------------------------------------------------------


class Conv1dBuffers(nn.Module):
    """Holds conv1d weight and bias as plain buffers (not quantized).

    Named ``conv1d`` inside :class:`MambaMixer` so checkpoint keys like
    ``backbone.layers.N.mixer.conv1d.weight`` resolve correctly.
    """

    def __init__(self, conv_dim: int, conv_kernel: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.zeros(conv_dim, 1, conv_kernel))
        self.register_buffer("bias", torch.zeros(conv_dim))


# ---------------------------------------------------------------------------
# MambaMixer
# ---------------------------------------------------------------------------


class MambaMixer(nn.Module):
    """Mamba2 SSM computation module.

    Named ``mixer`` inside :class:`NemotronHDecoderLayer` to match checkpoint
    key prefix ``backbone.layers.N.mixer.*``.

    Buffers / parameters (names match checkpoint keys exactly):
        in_proj.weight           - input projection (quantised)
        out_proj.weight          - output projection (quantised)
        conv1d.weight            - [conv_dim, 1, conv_kernel]
        conv1d.bias              - [conv_dim]
        A_log                    - [num_heads]  float32
        D                        - [num_heads]  float32
        dt_bias                  - [num_heads]  float32
        norm.weight              - [intermediate_size] (gated RMSNorm)
    """

    # Bound for dt before softplus. FP8 in_proj carries a per-tensor weight
    # scale, so an outlier channel can drive dt past the fp16 range; softplus(dt)
    # then overflows the SSM recurrence. softplus is ~identity in this range, so
    # clamping keeps well-behaved dt exact while capping the pathological ones.
    _DT_CLAMP = 50.0

    def __init__(self, config: ModelConfig, mc: MambaConfig,
                 module_prefix: str) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        d_inner = mc.intermediate_size  # num_heads * head_dim

        # in_proj output: d_inner (gate) + conv_dim + num_heads (dt)
        in_proj_out = d_inner + mc.conv_dim + mc.num_heads
        self.in_proj = make_linear(config,
                                   hidden_size,
                                   in_proj_out,
                                   bias=False,
                                   module_name=f"{module_prefix}.in_proj")
        self.out_proj = make_linear(config,
                                    d_inner,
                                    hidden_size,
                                    bias=False,
                                    module_name=f"{module_prefix}.out_proj")

        self.conv1d = Conv1dBuffers(mc.conv_dim, mc.conv_kernel)

        # A_log must be FP16 so that .to(torch.float32) in forward() produces
        # an explicit Cast node — the Mamba plugin requires ssm_A as FP32.
        # (D and dt_bias are cast to FP16 in forward; register as FP16 directly.)
        self.register_buffer("A_log",
                             torch.zeros(mc.num_heads, dtype=torch.float16))
        self.register_buffer("D", torch.zeros(mc.num_heads,
                                              dtype=torch.float16))
        self.register_buffer("dt_bias",
                             torch.zeros(mc.num_heads, dtype=torch.float16))

        # Gated RMSNorm weight (stored as submodule named "norm")
        self.norm = RMSNorm(d_inner, eps=config.rms_norm_eps)

        self.num_heads = mc.num_heads
        self.head_dim = mc.head_dim
        self.n_groups = mc.n_groups
        self.ssm_state_size = mc.ssm_state_size
        self.conv_dim = mc.conv_dim
        self.conv_kernel = mc.conv_kernel
        self._group_size = (mc.num_heads * mc.head_dim) // mc.n_groups

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        context_lengths: torch.Tensor,
        state_start_index: torch.Tensor,
        collect_intermediate_states: bool = False,
        spec_verify_phase_marker: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        d_inner = self.num_heads * self.head_dim
        d_state = self.n_groups * self.ssm_state_size

        projected_states = self.in_proj(hidden_states)
        # Split: [gate, BC conv path, dt]
        gate, hidden_states_for_conv, dt = projected_states.split(
            [d_inner, self.conv_dim, self.num_heads], dim=-1)
        dt = dt.clamp(-self._DT_CLAMP, self._DT_CLAMP)

        # MTP spec-verify (mtp_base) path emits per-token conv/SSM state snapshots
        # so the runtime can roll recurrent state back to the last accepted token.
        if collect_intermediate_states:
            (hidden_states_for_conv, conv_state_out,
             intermediate_conv_state) = causal_conv1d_with_intermediate(
                 hidden_states_for_conv,
                 self.conv1d.weight,
                 self.conv1d.bias,
                 conv_state,
                 context_lengths,
                 stride=1,
                 padding=self.conv_kernel - 1,
                 dilation=1,
                 groups=self.conv_dim,
                 spec_verify_phase_marker=spec_verify_phase_marker,
             )
        else:
            hidden_states_for_conv, conv_state_out, _ = causal_conv1d(
                hidden_states_for_conv,
                self.conv1d.weight,
                self.conv1d.bias,
                conv_state,
                context_lengths,
                stride=1,
                padding=self.conv_kernel - 1,
                dilation=1,
                groups=self.conv_dim,
            )
        hidden_states_for_conv = F.silu(hidden_states_for_conv)

        ssm_input, ssm_b_flat, ssm_c_flat = hidden_states_for_conv.split(
            [d_inner, d_state, d_state], dim=-1)

        ssm_input_states = ssm_input.view(batch_size, seq_len, self.num_heads,
                                          self.head_dim)
        ssm_b_states = ssm_b_flat.view(batch_size, seq_len, self.n_groups,
                                       self.ssm_state_size)
        ssm_c_states = ssm_c_flat.view(batch_size, seq_len, self.n_groups,
                                       self.ssm_state_size)

        ssm_A = -torch.exp(self.A_log.to(torch.float32))

        if collect_intermediate_states:
            (ssm_output, ssm_state_out, replay_da, replay_u,
             replay_b) = update_ssm_state_with_intermediate(
                 ssm_input_states,
                 ssm_A,
                 ssm_b_states,
                 ssm_c_states,
                 self.D,
                 dt,
                 self.dt_bias,
                 ssm_state,
                 context_lengths,
                 state_start_index,
                 spec_verify_phase_marker,
                 dt_softplus=1,
                 ngroups=self.n_groups,
             )
        else:
            ssm_output, ssm_state_out = update_ssm_state(
                ssm_input_states,
                ssm_A,
                ssm_b_states,
                ssm_c_states,
                self.D,
                dt,
                self.dt_bias,
                ssm_state,
                context_lengths,
                state_start_index,
                dt_softplus=1,
                ngroups=self.n_groups,
            )

        ssm_output = ssm_output.view(batch_size, seq_len, d_inner)
        normed = self._gated_rmsnorm(ssm_output, gate)
        if collect_intermediate_states:
            return (self.out_proj(normed), conv_state_out, ssm_state_out,
                    intermediate_conv_state, replay_da, replay_u, replay_b)
        return self.out_proj(normed), conv_state_out, ssm_state_out

    def _gated_rmsnorm(self, hidden_states: torch.Tensor,
                       gate: torch.Tensor) -> torch.Tensor:
        group_size = self._group_size
        gated = hidden_states * F.silu(gate)
        # Compute RMSNorm in FP32 explicitly, matching what TRT would do
        # internally (it forces ReduceMean to FP32).  The explicit
        # FP32→FP16 Cast at the end creates an ONNX partition boundary
        # so TRT can split the ForeignNode at the plugin inputs.
        gated_f32 = gated.to(torch.float32)
        gated_grouped = gated_f32.view(*gated_f32.shape[:-1], -1, group_size)
        variance = (gated_grouped * gated_grouped).mean(-1, keepdim=True)
        normed = gated_grouped * torch.rsqrt(variance +
                                             self.norm.variance_epsilon)
        normed = normed.view(*hidden_states.shape)
        return normed.to(torch.float16) * self.norm.weight


# ---------------------------------------------------------------------------
# NemotronHMLP  (relu² gated MLP)
# ---------------------------------------------------------------------------


class NemotronHMLP(nn.Module):
    """NemotronH MLP block: up_proj + relu² + down_proj.

    Named ``mixer`` inside :class:`NemotronHDecoderLayer` to match checkpoint
    key prefix ``backbone.layers.N.mixer.*``.
    """

    def __init__(self, config: ModelConfig, module_prefix: str) -> None:
        super().__init__()
        self.up_proj = make_linear(config,
                                   config.hidden_size,
                                   config.intermediate_size,
                                   module_name=f"{module_prefix}.up_proj")
        self.down_proj = make_linear(config,
                                     config.intermediate_size,
                                     config.hidden_size,
                                     module_name=f"{module_prefix}.down_proj")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.up_proj(hidden_states)
        r = F.relu(h)
        return self.down_proj(r * r)


# ---------------------------------------------------------------------------
# NemotronHTopkRouter
# ---------------------------------------------------------------------------


class NemotronHTopkRouter(nn.Module):
    """Sigmoid-based grouped top-k router for MoE layers.

    Submodule names match checkpoint keys:
        weight                  - [n_routed_experts, hidden_size] (FP32)
        e_score_correction_bias - [n_routed_experts] (FP32)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size

        self.weight = nn.Parameter(
            torch.empty(self.n_routed_experts,
                        config.hidden_size,
                        dtype=torch.float16))
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float16))

    def forward(
            self,
            hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits = F.linear(hidden_states, self.weight).float()
        scores = router_logits.sigmoid()

        scores_for_choice = scores + self.e_score_correction_bias.float(
        ).unsqueeze(0)
        group_scores = (scores_for_choice.view(
            -1, self.n_group,
            self.n_routed_experts // self.n_group).topk(2,
                                                        dim=-1)[0].sum(dim=-1))
        group_idx = torch.topk(group_scores,
                               k=self.topk_group,
                               dim=-1,
                               sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (group_mask.unsqueeze(-1).expand(
            -1, self.n_group, self.n_routed_experts // self.n_group).reshape(
                -1, self.n_routed_experts))
        scores_for_choice = scores_for_choice.masked_fill(
            ~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice,
                                  k=self.top_k,
                                  dim=-1,
                                  sorted=False)[1]

        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


# ---------------------------------------------------------------------------
# NemotronHMoEMLP
# ---------------------------------------------------------------------------


class NemotronHMoEMLP(nn.Module):
    """Mixture-of-Experts MLP for NemotronH using the configured NVFP4 MoE plugin.

    Named ``mixer`` inside :class:`NemotronHDecoderLayer` to match checkpoint
    key prefix ``backbone.layers.N.mixer.*``.

    The routed experts dispatch through ``trt_edgellm::Nvfp4MoePlugin``.
    The shared expert runs as a separate FP8/FP16 forward pass added to the
    plugin output.

    Submodule names match checkpoint keys:
        gate                     - NemotronHTopkRouter (weight + bias)
        experts.{j}.up_proj     - per-expert up projection (NVFP4)
        experts.{j}.down_proj   - per-expert down projection (NVFP4)
        shared_experts.up_proj  - shared expert up projection (FP8)
        shared_experts.down_proj - shared expert down projection (FP8)
    """

    def __init__(self, config: ModelConfig, module_prefix: str) -> None:
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.routed_hidden_size = (config.moe_latent_size
                                   if config.moe_latent_size is not None else
                                   config.hidden_size)
        self.moe_intermediate_size = config.moe_intermediate_size
        self.group_size = config.quant.group_size
        self.activation_type = _NVFP4_ACTIVATION_RELU2
        self.backend = _NVFP4_MOE_BACKEND_AUTO
        self.io_dtype = _NVFP4_MOE_IO_DTYPE_FP16
        self.max_routed_rows = _NVFP4_MOE_MAX_ROUTED_ROWS_AUTO
        self._padded_moe_intermediate_size = self.moe_intermediate_size
        # SM12x NvFP4MoEPluginGeforce additionally requires H % 256 == 0
        # (kCuteDslTileK * kStaticAbStage). For checkpoints whose hidden_size
        # does not satisfy that (e.g. Nemotron-Nano H=2688), ``_prepare_for_export_impl``
        # picks ``hidden_size_alignment=256`` so the FC1 K and FC2 M axes get
        # zero-padded; ``forward`` then F.pads hidden_states and slices the
        # plugin output. The SM100/101/110 path keeps the original H.
        self._padded_hidden_size = self.routed_hidden_size
        self.gate = NemotronHTopkRouter(config)

        # Weight-only NVFP4 (W4A16) routes the experts through the Marlin
        # Nvfp4A16MoePlugin instead of the W4A4 CuTeDSL Nvfp4MoePlugin.
        self._is_a16 = config.quant.quant_type == QUANT_NVFP4_A16
        # ReLU2 (non-gated) FC1 padded to a Marlin 128 multiple (1856 -> 1920).
        self._a16_moe_inter_padded = (
            ((self.moe_intermediate_size + 127) // 128) *
            128 if self._is_a16 else self.moe_intermediate_size)

        self.experts = nn.ModuleList([
            self._make_expert(config,
                              config.moe_intermediate_size,
                              f"{module_prefix}.experts.{j}",
                              input_size=self.routed_hidden_size)
            for j in range(config.n_routed_experts)
        ])
        if self._is_a16:
            # Routed experts are stacked into the MoE plugin at export time;
            # keep the per-linear dense repack from consuming their raw buffers.
            for expert in self.experts:
                expert.up_proj._skip_dense_a16_repack = True
                expert.down_proj._skip_dense_a16_repack = True

        self.shared_experts = self._make_expert(
            config,
            config.moe_shared_expert_intermediate_size,
            f"{module_prefix}.shared_experts",
            input_size=config.hidden_size)

        if config.moe_latent_size is not None:
            self.fc1_latent_proj = make_linear(
                config,
                config.hidden_size,
                self.routed_hidden_size,
                module_name=f"{module_prefix}.fc1_latent_proj")
            self.fc2_latent_proj = make_linear(
                config,
                self.routed_hidden_size,
                config.hidden_size,
                module_name=f"{module_prefix}.fc2_latent_proj")
        else:
            self.fc1_latent_proj = nn.Identity()
            self.fc2_latent_proj = nn.Identity()

        self._export_ready = False

    @staticmethod
    def _make_expert(config: ModelConfig, inter_size: int, prefix: str,
                     input_size: int) -> nn.Module:
        expert = nn.Module()
        expert.up_proj = make_linear(config,
                                     input_size,
                                     inter_size,
                                     module_name=f"{prefix}.up_proj")
        expert.down_proj = make_linear(config,
                                       inter_size,
                                       input_size,
                                       module_name=f"{prefix}.down_proj")
        return expert

    @staticmethod
    def _expert_forward(expert: nn.Module,
                        hidden_states: torch.Tensor) -> torch.Tensor:
        h = expert.up_proj(hidden_states)
        r = F.relu(h)
        return expert.down_proj(r * r)

    def prepare_for_export(self) -> None:
        """Pack ModelOpt NVFP4 expert tensors for ``Nvfp4MoePlugin``."""
        self._prepare_for_export_impl()

    def _prepare_for_export_a16(self) -> None:
        """Stack routed-expert NVFP4 (W4A16) weights for ``Nvfp4A16MoePlugin``."""
        from ...checkpoint.repacking import repack_nvfp4_a16_marlin_moe_experts

        def gather(attr):
            return [getattr(e, attr)._buffers["weight"]
                    for e in self.experts], [
                        getattr(e, attr)._buffers["weight_scale"]
                        for e in self.experts
                    ], [
                        getattr(e, attr)._buffers["weight_scale_2"]
                        for e in self.experts
                    ]

        fc1_p, fc1_s, fc1_g = gather("up_proj")
        fc2_p, fc2_s, fc2_g = gather("down_proj")
        (fc1_qweights, fc1_block_scales, fc1_global, fc2_qweights,
         fc2_block_scales, fc2_global) = repack_nvfp4_a16_marlin_moe_experts(
             fc1_p, fc1_s, fc1_g, fc2_p, fc2_s, fc2_g,
             self._a16_moe_inter_padded)
        self._padded_moe_intermediate_size = self._a16_moe_inter_padded
        self._padded_hidden_size = self.routed_hidden_size

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
            "_e_score_correction_bias_fp32",
            self.gate.e_score_correction_bias.data.clone().to(
                torch.float32).to(device))
        # Free the per-expert buffers now that they are stacked.
        for expert in self.experts:
            for proj in ("up_proj", "down_proj"):
                for name in ("weight", "weight_scale", "weight_scale_2"):
                    getattr(expert, proj)._buffers.pop(name, None)
        self._export_ready = True

    def _prepare_for_export_impl(self) -> None:
        """Pack ModelOpt NVFP4 expert tensors for the active NVFP4 MoE plugin."""
        if self._is_a16:
            self._prepare_for_export_a16()
            return
        from ...checkpoint.repacking import repack_nvfp4_moe_experts

        # SM12x NvFP4MoEPluginGeforce requires H % 256 == 0; the SM100/101/110 path only needs
        # the kernel's regular alignment (the repack helper accepts H as-is
        # when ``hidden_size_alignment=1``).
        hidden_size_alignment = 256 if use_geforce_nvfp4_moe() else 1

        (fc1_qweights, fc1_blocks_scale, fc1_alpha, fc2_qweights,
         fc2_blocks_scale, fc2_alpha, padded_inter_size,
         padded_hidden_size) = (repack_nvfp4_moe_experts(
             self.experts,
             self.routed_hidden_size,
             self.moe_intermediate_size,
             self.group_size,
             hidden_size_alignment=hidden_size_alignment,
         ))
        self._padded_moe_intermediate_size = padded_inter_size
        self._padded_hidden_size = padded_hidden_size

        device = self.gate.weight.device
        self.register_buffer("fc1_qweights",
                             fc1_qweights.to(device).contiguous())
        self.register_buffer("fc1_blocks_scale",
                             fc1_blocks_scale.to(device).contiguous())
        self.register_buffer("fc2_qweights",
                             fc2_qweights.to(device).contiguous())
        self.register_buffer("fc2_blocks_scale",
                             fc2_blocks_scale.to(device).contiguous())
        self.register_buffer("fc1_alpha", fc1_alpha.to(device).contiguous())
        self.register_buffer("fc2_alpha", fc2_alpha.to(device).contiguous())
        self.register_buffer(
            "input_global_scale",
            torch.ones(self.n_routed_experts,
                       dtype=torch.float32,
                       device=device))
        self.register_buffer(
            "down_input_scale",
            torch.ones(self.n_routed_experts,
                       dtype=torch.float32,
                       device=device))
        self.register_buffer(
            "_e_score_correction_bias_fp32",
            self.gate.e_score_correction_bias.data.clone().to(
                torch.float32).to(device))
        self._export_ready = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        # Router logits are computed from the original hidden states. When
        # moe_latent_size is set, only the routed expert payload is projected
        # down into latent space.
        router_logits = F.linear(hidden_states.view(-1, self.hidden_size),
                                 self.gate.weight).float()
        routed_hidden_states = self.fc1_latent_proj(hidden_states)

        if self._is_a16:
            # FP16 activations end-to-end: the Marlin MoE kernel has an FP16
            # E2M1 path, so hidden states / global scales / output stay FP16.
            moe_out = nvfp4_a16_moe_plugin(
                router_logits,
                routed_hidden_states,
                self.fc1_qweights,
                self.fc1_block_scales,
                self.fc1_global_scales,
                self.fc2_qweights,
                self.fc2_block_scales,
                self.fc2_global_scales,
                self._e_score_correction_bias_fp32,
                self.n_routed_experts,
                self.num_experts_per_tok,
                self.routed_hidden_size,
                self._a16_moe_inter_padded,
                self.activation_type,
                self.gate.n_group,
                self.gate.topk_group,
                int(bool(self.gate.norm_topk_prob)),
                float(self.gate.routed_scaling_factor),
                _NVFP4_ROUTING_MODE_SIGMOID_GROUP_TOPK,
                self.max_routed_rows,
            )
            moe_out = self.fc2_latent_proj(moe_out)
            return moe_out + self._expert_forward(self.shared_experts,
                                                  hidden_states)

        # SM12x NvFP4MoEPluginGeforce requires the plugin hidden_size to be a
        # multiple of 256. When the checkpoint H does not satisfy that,
        # ``_prepare_for_export_impl`` zero-pads FC1 K / FC2 M and sets
        # ``self._padded_hidden_size``; we F.pad the hidden activations here
        # and slice the plugin output back. relu2(0) = 0 keeps the padded
        # FC1 outputs zero; the FC2 contribution to the padded H slots is
        # therefore zero too. The shared expert path uses the original H.
        plugin_hidden = routed_hidden_states
        if self._padded_hidden_size != self.routed_hidden_size:
            plugin_hidden = F.pad(
                routed_hidden_states,
                (0, self._padded_hidden_size - self.routed_hidden_size))

        # Nemotron-H uses ReLU2 (non-gated) FC1, so the up-only weight tensor
        # has the same row layout under both plugins; only the plugin op name
        # differs between SM100/101/110 ``Nvfp4MoePlugin`` and SM12x
        # ``NvFP4MoEPluginGeforce``.
        moe_op = (nvfp4_moe_plugin_geforce
                  if use_geforce_nvfp4_moe() else nvfp4_moe_plugin)
        moe_out = moe_op(
            router_logits,
            plugin_hidden,
            self.fc1_qweights,
            self.fc1_blocks_scale,
            self.fc1_alpha,
            self.fc2_qweights,
            self.fc2_blocks_scale,
            self.fc2_alpha,
            self.input_global_scale,
            self.down_input_scale,
            self._e_score_correction_bias_fp32,
            self.n_routed_experts,
            self.num_experts_per_tok,
            self._padded_hidden_size,
            self._padded_moe_intermediate_size,
            self.activation_type,
            self.gate.n_group,
            self.gate.topk_group,
            int(bool(self.gate.norm_topk_prob)),
            float(self.gate.routed_scaling_factor),
            _NVFP4_ROUTING_MODE_SIGMOID_GROUP_TOPK,
            self.backend,
            self.io_dtype,
            self.max_routed_rows,
        )
        if self._padded_hidden_size != self.routed_hidden_size:
            moe_out = moe_out[..., :self.routed_hidden_size]

        moe_out = self.fc2_latent_proj(moe_out)
        return moe_out + self._expert_forward(self.shared_experts,
                                              hidden_states)


# ---------------------------------------------------------------------------
# NemotronHAttentionMixer
# ---------------------------------------------------------------------------


class NemotronHAttentionMixer(nn.Module):
    """GQA attention for NemotronH.

    Named ``mixer`` inside :class:`NemotronHDecoderLayer` to match checkpoint
    key prefix ``backbone.layers.N.mixer.*``.

    Submodule names match checkpoint keys:
        q_proj, k_proj, v_proj, o_proj
    """

    def __init__(self,
                 config: ModelConfig,
                 layer_idx: int,
                 module_prefix: str,
                 enable_tree_attention: bool = False) -> None:
        super().__init__()
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = config.head_dim
        hidden_size = config.hidden_size

        self.layer_idx = layer_idx
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_scale = config.attention_scaling
        self.enable_fp8_kv_cache = config.quant.kv_cache_quant == "fp8"
        self.sliding_window_size = -1
        # Tree attention (with attention_mask / attention_pos_id inputs) is used
        # by the MTP draft for speculative decoding; the base model leaves it off.
        self.enable_tree_attention = enable_tree_attention

        self.q_proj = make_linear(config,
                                  hidden_size,
                                  num_attention_heads * head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.q_proj")
        self.k_proj = make_linear(config,
                                  hidden_size,
                                  num_key_value_heads * head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.k_proj")
        self.v_proj = make_linear(config,
                                  hidden_size,
                                  num_key_value_heads * head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.v_proj")
        if self.enable_fp8_kv_cache:
            self.q_proj.register_buffer("q_scale", torch.ones(1))
            self.k_proj.register_buffer("k_scale", torch.ones(1))
            self.v_proj.register_buffer("v_scale", torch.ones(1))

        self.o_proj = make_linear(config,
                                  num_attention_heads * head_dim,
                                  hidden_size,
                                  module_name=f"{module_prefix}.o_proj")

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_pos_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Tree attention is used by the MTP draft (constructor flag) and by the
        # mtp_base variant of the base model (mask + pos_id supplied at call time).
        enable_tree = self.enable_tree_attention or (
            attention_mask is not None and attention_pos_id is not None)
        kwargs: dict = {
            "num_q_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_dim,
            "sliding_window_size": self.sliding_window_size,
            "enable_tree_attention": enable_tree,
            "enable_fp8_kv_cache": self.enable_fp8_kv_cache,
            "attention_scale": self.attention_scale,
            "enable_context_mask_selector": False,
            "enable_vision_block_attention": False,
            "skip_softmax_scale_factor": 0.0,
        }
        if enable_tree:
            kwargs["attention_mask"] = attention_mask
            kwargs["attention_pos_id"] = attention_pos_id
        # Always pass qkv_scales so torch.export includes a valid FLOATS
        # value in the FX graph for the unified ONNX translation.
        kwargs["qkv_scales"] = getattr(self, "_qkv_scales_float",
                                       [1.0, 1.0, 1.0])
        attn_output, present_key_value = attention_plugin(
            torch.cat([query_states, key_states, value_states],
                      dim=-1), past_key_value, context_lengths,
            rope_rotary_cos_sin, kvcache_start_index, kv_page_table, **kwargs)

        attn_output = attn_output.reshape(batch_size, seq_len,
                                          self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_key_value


# ---------------------------------------------------------------------------
# NemotronHDecoderLayer
# ---------------------------------------------------------------------------


class NemotronHDecoderLayer(nn.Module):
    """Single NemotronH decoder layer: pre-norm + mixer.

    Submodule names match checkpoint keys:
        norm    - RMSNorm (pre-mixer)
        mixer   - MambaMixer | NemotronHMLP | NemotronHAttentionMixer
    """

    def __init__(self, config: ModelConfig, layer_idx: int,
                 layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        module_prefix = f"backbone.layers.{layer_idx}.mixer"

        if layer_type == LAYER_MAMBA:
            assert config.mamba_cfg is not None
            self.mixer = MambaMixer(config, config.mamba_cfg, module_prefix)
        elif layer_type == LAYER_MLP:
            self.mixer = NemotronHMLP(config, module_prefix)
        elif layer_type == LAYER_MOE:
            self.mixer = NemotronHMoEMLP(config, module_prefix)
        elif layer_type == LAYER_ATTN:
            self.mixer = NemotronHAttentionMixer(config, layer_idx,
                                                 module_prefix)
        else:
            raise ValueError(f"Unknown layer type: {layer_type!r}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        # Attention-specific (ignored by Mamba/MLP/MoE layers)
        past_key_value: Optional[torch.Tensor] = None,
        rope_rotary_cos_sin: Optional[torch.Tensor] = None,
        context_lengths: Optional[torch.Tensor] = None,
        kvcache_start_index: Optional[torch.Tensor] = None,
        kv_page_table: Optional[torch.Tensor] = None,
        # Mamba-specific (ignored by Attention/MLP/MoE layers)
        conv_state: Optional[torch.Tensor] = None,
        ssm_state: Optional[torch.Tensor] = None,
        # MTP spec-verify (mtp_base) — tree attention + per-token recurrent snapshots.
        collect_intermediate_states: bool = False,
        spec_verify_phase_marker: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_pos_id: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        normed = self.norm(hidden_states)
        if self.layer_type == LAYER_MAMBA:
            if collect_intermediate_states:
                (mixer_out, conv_state_out, ssm_state_out, inter_conv,
                 replay_da, replay_u, replay_b) = self.mixer(
                     normed,
                     conv_state,
                     ssm_state,
                     context_lengths,
                     kvcache_start_index,
                     collect_intermediate_states=True,
                     spec_verify_phase_marker=spec_verify_phase_marker)
                return (residual + mixer_out, conv_state_out, ssm_state_out,
                        inter_conv, replay_da, replay_u, replay_b)
            mixer_out, conv_state_out, ssm_state_out = self.mixer(
                normed, conv_state, ssm_state, context_lengths,
                kvcache_start_index)
            return residual + mixer_out, conv_state_out, ssm_state_out
        elif self.layer_type in (LAYER_MLP, LAYER_MOE):
            return residual + self.mixer(normed)
        else:
            attn_out, present_kv = self.mixer(normed, past_key_value,
                                              rope_rotary_cos_sin,
                                              context_lengths,
                                              kvcache_start_index,
                                              kv_page_table, attention_mask,
                                              attention_pos_id)
            return residual + attn_out, present_kv


# ---------------------------------------------------------------------------
# NemotronHBackbone
# ---------------------------------------------------------------------------


class NemotronHBackbone(nn.Module):
    """NemotronH transformer backbone.

    Submodule names match checkpoint keys:
        embeddings   - token embedding (backbone.embeddings.weight)
        layers       - decoder layer list
        norm_f       - final RMSNorm (backbone.norm_f.weight)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            NemotronHDecoderLayer(config, layer_idx=i, layer_type=lt)
            for i, lt in enumerate(config.layer_types)
        ])
        self.norm_f = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layer_types: List[str] = config.layer_types

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        conv_states: Tuple[torch.Tensor, ...] = (),
        ssm_states: Tuple[torch.Tensor, ...] = (),
        collect_intermediate_states: bool = False,
        spec_verify_phase_marker: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_pos_id: Optional[torch.Tensor] = None,
        dflash_target_layer_ids: Optional[List[int]] = None,
    ):
        hidden_states = inputs_embeds
        present_key_values_list: List[torch.Tensor] = []
        present_conv_states_list: List[torch.Tensor] = []
        present_ssm_states_list: List[torch.Tensor] = []
        intermediate_conv_states_list: List[torch.Tensor] = []
        replay_da_states_list: List[torch.Tensor] = []
        replay_u_states_list: List[torch.Tensor] = []
        replay_b_states_list: List[torch.Tensor] = []
        dflash_hidden_list: List[torch.Tensor] = []
        dflash_target_set = set(dflash_target_layer_ids or [])
        last_layer_idx = len(self.layers) - 1
        attn_idx = 0
        mamba_idx = 0

        for layer_idx, (layer,
                        lt) in enumerate(zip(self.layers, self.layer_types)):
            if lt == LAYER_MAMBA:
                if collect_intermediate_states:
                    (hidden_states, conv_out, ssm_out, inter_conv, replay_da,
                     replay_u, replay_b) = layer(
                         hidden_states,
                         context_lengths=context_lengths,
                         kvcache_start_index=kvcache_start_index,
                         conv_state=conv_states[mamba_idx],
                         ssm_state=ssm_states[mamba_idx],
                         collect_intermediate_states=True,
                         spec_verify_phase_marker=spec_verify_phase_marker,
                     )
                    intermediate_conv_states_list.append(inter_conv)
                    replay_da_states_list.append(replay_da)
                    replay_u_states_list.append(replay_u)
                    replay_b_states_list.append(replay_b)
                else:
                    hidden_states, conv_out, ssm_out = layer(
                        hidden_states,
                        context_lengths=context_lengths,
                        kvcache_start_index=kvcache_start_index,
                        conv_state=conv_states[mamba_idx],
                        ssm_state=ssm_states[mamba_idx],
                    )
                present_conv_states_list.append(conv_out)
                present_ssm_states_list.append(ssm_out)
                mamba_idx += 1
            elif lt in (LAYER_MLP, LAYER_MOE):
                hidden_states = layer(hidden_states)
            else:
                hidden_states, present_kv = layer(
                    hidden_states,
                    past_key_value=past_key_values[attn_idx],
                    rope_rotary_cos_sin=rope_rotary_cos_sin,
                    context_lengths=context_lengths,
                    kvcache_start_index=kvcache_start_index,
                    kv_page_table=kv_page_table,
                    attention_mask=attention_mask,
                    attention_pos_id=attention_pos_id,
                )
                present_key_values_list.append(present_kv)
                attn_idx += 1

            # The final layer's DFlash aux feature is the post-norm_f hidden.
            # Earlier target layers use the raw residual stream.
            if layer_idx in dflash_target_set and layer_idx != last_layer_idx:
                dflash_hidden_list.append(hidden_states)

        normed = self.norm_f(hidden_states)
        if last_layer_idx in dflash_target_set:
            dflash_hidden_list.append(normed)
        dflash_hidden_concat = (torch.cat(dflash_hidden_list, dim=-1)
                                if dflash_hidden_list else None)
        if collect_intermediate_states:
            return (normed, tuple(present_key_values_list),
                    tuple(present_conv_states_list),
                    tuple(present_ssm_states_list),
                    tuple(intermediate_conv_states_list),
                    tuple(replay_da_states_list), tuple(replay_u_states_list),
                    tuple(replay_b_states_list), dflash_hidden_concat)
        return (normed, tuple(present_key_values_list),
                tuple(present_conv_states_list),
                tuple(present_ssm_states_list), dflash_hidden_concat)


# ---------------------------------------------------------------------------
# NemotronHCausalLM
# ---------------------------------------------------------------------------

_BATCH_SIZE = 1
_SEQ_LEN = 1
_PAST_LEN = 1
_MAX_POS = 4096


class NemotronHCausalLM(nn.Module):
    """NemotronH causal LM: backbone + lm_head.

    The inner backbone is stored as attribute ``backbone`` so parameter keys
    carry the ``backbone.`` prefix matching checkpoint key prefixes.
    ``lm_head`` maps directly to ``lm_head.weight``.
    """

    # Dtypes of the Mamba state tensors this model feeds the ONNX graph.
    # These drive (a) the dummy tensor dtypes in ``export_onnx`` and
    # (b) the ``recurrent_state_dtype`` / ``conv_state_dtype`` strings written
    # into ``config.json`` (see checkpoint_utils). They must stay in sync, so
    # the single source of truth is this class attribute, not a separate table.
    # The dtype is dictated by the ``trt_edgellm::update_ssm_state`` plugin
    # schema: ``state`` has type ``T`` where we pick float16.
    RECURRENT_STATE_DTYPE = torch.float16
    CONV_STATE_DTYPE = torch.float16

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = NemotronHBackbone(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    def tie_weights(self) -> None:
        """Clone embeddings.weight into lm_head.weight when tie_word_embeddings=True."""
        if not self.config.tie_word_embeddings:
            return
        if not isinstance(self.lm_head, FP16Linear):
            return
        embed_weight = self.backbone.embeddings.weight
        self.lm_head.weight = nn.Parameter(embed_weight.detach().clone(),
                                           requires_grad=False)

    def onnx_export_spec(self) -> OnnxSpec:
        """Return all model-specific parameters needed for ONNX export."""
        # Pre-process MoE layers: reinterpret FP8 scales as INT8
        for layer in self.backbone.layers:
            if hasattr(layer.mixer, 'prepare_for_export'):
                layer.mixer.prepare_for_export()
        config = self.config
        mc = config.mamba_cfg
        Na = config.num_attn_layers
        Nm = config.num_mamba_layers
        Nd = 0  # NemotronH has no deepstack visual embeddings
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
        past_key_values_list: List[torch.Tensor] = [
            torch.zeros(2,
                        1,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        config.head_dim,
                        dtype=kv_dtype,
                        device=device) for _ in range(Na)
        ]
        rope_rotary_cos_sin = torch.zeros(batch_size,
                                          max_pos,
                                          config.head_dim,
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

        assert mc is not None, "NemotronHCausalLM requires mamba_cfg"
        conv_states: List[torch.Tensor] = [
            torch.zeros(batch_size,
                        mc.conv_dim,
                        mc.conv_kernel,
                        dtype=self.CONV_STATE_DTYPE,
                        device=device) for _ in range(Nm)
        ]
        ssm_states: List[torch.Tensor] = [
            torch.zeros(batch_size,
                        mc.num_heads,
                        mc.head_dim,
                        mc.ssm_state_size,
                        dtype=self.RECURRENT_STATE_DTYPE,
                        device=device) for _ in range(Nm)
        ]

        mtp_base = bool(getattr(config, "mtp_base", False))
        dflash_base = bool(getattr(config, "dflash_base", False))
        spec = mtp_base or dflash_base

        batch = torch.export.Dim("batch", min=1, max=256)
        seq = torch.export.Dim("seq_len", min=1, max=32768)
        pos = torch.export.Dim("max_pos", min=1, max=32768)
        rope_batch = torch.export.Dim("rope_batch", min=1, max=256)
        kv_batch = torch.export.Dim("kv_batch", min=1, max=256)
        page_batch = torch.export.Dim("page_batch", min=1, max=256)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        num_pages = torch.export.Dim("num_pages", min=1, max=1048576)

        # Spec-verify bases widen last_token_ids to select multiple positions.
        if spec:
            last_token_ids = torch.zeros(batch_size,
                                         2,
                                         dtype=torch.int64,
                                         device=device)

        args = (inputs_embeds, *past_key_values_list, rope_rotary_cos_sin,
                context_lengths, kvcache_start_index, kv_page_table,
                last_token_ids, *conv_states, *ssm_states)

        input_names = (
            ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
                "rope_rotary_cos_sin", "context_lengths",
                "kvcache_start_index", "kv_page_table", "last_token_ids"
            ] + [f"conv_state_{i}" for i in range(Nm)] +
            [f"recurrent_state_{i}" for i in range(Nm)])
        output_names = (["logits"] +
                        [f"present_key_values_{i}" for i in range(Na)] +
                        [f"present_conv_state_{i}" for i in range(Nm)] +
                        [f"present_recurrent_state_{i}" for i in range(Nm)])

        all_shapes: list = [{0: batch, 1: seq}]  # inputs_embeds
        for _ in range(Na):
            all_shapes.append({1:
                               num_pages})  # past_key_values_i (pool-shaped)
        all_shapes.append({0: rope_batch, 1: pos})  # rope_rotary_cos_sin
        all_shapes.append({0: batch})  # context_lengths
        all_shapes.append({0: kv_batch})  # kvcache_start_index
        all_shapes.append({0: page_batch, 2: max_pages})  # kv_page_table
        if spec:
            num_selected = torch.export.Dim("num_selected", min=1, max=256)
            all_shapes.append({0: batch, 1: num_selected})  # last_token_ids
        else:
            all_shapes.append({0: batch})  # last_token_ids
        for _ in range(Nm):
            all_shapes.append({0: batch})  # conv_state_i
        for _ in range(Nm):
            all_shapes.append({0: batch})  # recurrent_state_i

        if spec:
            # Tree-attention + spec-verify inputs.
            attn_seq = torch.export.Dim("attn_seq_len", min=1, max=32768)
            mask_kv_len = torch.export.Dim("mask_kv_len", min=1, max=65536)
            attention_pos_id = torch.zeros(batch_size,
                                           seq_len,
                                           dtype=torch.int32,
                                           device=device)
            attention_mask = torch.zeros(batch_size,
                                         seq_len,
                                         seq_len + past_len,
                                         dtype=torch.int32,
                                         device=device)
            spec_verify_phase_marker = torch.zeros(1,
                                                   dtype=torch.int32,
                                                   device=device)
            args = args + (attention_pos_id, attention_mask,
                           spec_verify_phase_marker)
            input_names = input_names + [
                "attention_pos_id", "attention_mask",
                "spec_verify_phase_marker"
            ]
            all_shapes.append({0: batch, 1: attn_seq})  # attention_pos_id
            all_shapes.append({0: batch, 1: attn_seq, 2: mask_kv_len})  # mask
            all_shapes.append({0: torch.export.Dim.AUTO})  # phase marker
            # Extra outputs: full hidden + intermediate conv snapshot + the
            # per-token recurrent replay stash (dA / u / B) for spec-verify.
            output_names = (
                ["logits", "hidden_states"] +
                [f"present_key_values_{i}" for i in range(Na)] +
                [f"present_conv_state_{i}" for i in range(Nm)] +
                [f"present_recurrent_state_{i}" for i in range(Nm)] +
                [f"intermediate_conv_state_{i}" for i in range(Nm)] +
                [f"replay_da_state_{i}" for i in range(Nm)] +
                [f"replay_u_state_{i}" for i in range(Nm)] +
                [f"replay_b_state_{i}" for i in range(Nm)])

        wrapped = _make_flat_wrapper_mamba(self,
                                           Na,
                                           Nm,
                                           mtp_base=mtp_base,
                                           dflash_base=dflash_base)
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
        conv_states: Tuple[torch.Tensor, ...] = (),
        ssm_states: Tuple[torch.Tensor, ...] = (),
        attention_pos_id: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        spec_verify_phase_marker: Optional[torch.Tensor] = None,
    ) -> Tuple:
        mtp_base = bool(getattr(self.config, "mtp_base", False))
        dflash_base = bool(getattr(self.config, "dflash_base", False))
        dflash_target_ids = (self.config.dflash_target_layer_ids
                             if dflash_base else None)
        spec = mtp_base or dflash_base
        if spec:
            (hidden_states, present_key_values, present_conv_states,
             present_ssm_states, intermediate_conv_states, replay_da_states,
             replay_u_states, replay_b_states,
             dflash_hidden_concat) = self.backbone(
                 inputs_embeds,
                 past_key_values,
                 rope_rotary_cos_sin,
                 context_lengths,
                 kvcache_start_index,
                 kv_page_table,
                 conv_states,
                 ssm_states,
                 collect_intermediate_states=True,
                 spec_verify_phase_marker=spec_verify_phase_marker,
                 attention_mask=attention_mask,
                 attention_pos_id=attention_pos_id,
                 dflash_target_layer_ids=dflash_target_ids,
             )
            # Draft consumes the full pre-lm_head hidden; logits use the
            # gathered predicted-token positions.
            selected = torch.ops.trt.gather_nd(hidden_states, last_token_ids)
            logits = self.lm_head(selected).to(torch.float32)
            if dflash_base:
                return (logits, dflash_hidden_concat, present_key_values,
                        present_conv_states, present_ssm_states,
                        intermediate_conv_states, replay_da_states,
                        replay_u_states, replay_b_states)
            return (logits, hidden_states, present_key_values,
                    present_conv_states, present_ssm_states,
                    intermediate_conv_states, replay_da_states,
                    replay_u_states, replay_b_states)

        (hidden_states, present_key_values, present_conv_states,
         present_ssm_states, _dflash_hidden_concat) = self.backbone(
             inputs_embeds,
             past_key_values,
             rope_rotary_cos_sin,
             context_lengths,
             kvcache_start_index,
             kv_page_table,
             conv_states,
             ssm_states,
         )
        # Select hidden states for specified token positions before lm_head.
        hidden_states = torch.ops.trt.gather_nd(hidden_states, last_token_ids)

        logits = self.lm_head(hidden_states).to(torch.float32)
        return logits, present_key_values, present_conv_states, present_ssm_states
