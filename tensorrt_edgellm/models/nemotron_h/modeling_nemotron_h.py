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

Token-major export conventions
------------------------------
``NemotronHCausalLM.forward_ragged``:

    inputs_embeds        [physical_tokens, hidden_size]             float16
    past_key_values      tuple of [2, num_pages, page_size, num_kv_heads, head_dim] per attn-layer
    rope_rotary_cos_sin  [physical_tokens, rotary_dim]              float32
    query metadata       token rows [physical_tokens], sequences [batch]
    state_indices        [batch]                                    int32
    kv_page_table        [batch, 2, max_pages_per_seq]              int32
    logits_indices       [selected_tokens]                          int64
    conv_states          tuple of [resident_rows, conv_dim, conv_kernel] per mamba-layer
    ssm_states           tuple of [resident_rows, num_heads, head_dim, ssm_state] per mamba-layer
    ──────────────────────────────────────────────────────────────────────
    -> logits             [selected_tokens, vocab_size]              float32
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
from ..default.modeling_default import (OnnxSpec,
                                        _concat_hidden_in_provider_order)
from ..linear import FP16Linear, make_linear
from ..ops import (KV_PAGE_SIZE, attention_plugin, causal_conv1d,
                   causal_conv1d_with_intermediate,
                   nvfp4_a16_blackwell_moe_plugin, nvfp4_a16_moe_plugin,
                   nvfp4_moe_plugin, nvfp4_moe_plugin_geforce,
                   update_ssm_state, update_ssm_state_with_intermediate,
                   use_blackwell_nvfp4_a16_moe, use_geforce_nvfp4_moe)

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


def _make_flat_wrapper_mamba_ragged(model: nn.Module,
                                    Na: int,
                                    Nm: int,
                                    tree_attention: bool,
                                    tree_state: bool = True) -> nn.Module:
    param_names = (
        ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
            "rope_rotary_cos_sin", "positions", "query_start_offsets",
            "query_lengths", "past_lengths", "attention_sequence_lengths",
            "state_indices", "execution_phase_marker",
            "context_sequence_count_carrier", "kv_page_table", "logits_indices"
        ] + [f"conv_state_{i}"
             for i in range(Nm)] + [f"recurrent_state_{i}" for i in range(Nm)])
    if tree_attention:
        param_names += ["attention_position_ids", "packed_attention_mask"]
        if tree_state:
            param_names += [
                "tree_parent_ids", "tree_depths", "valid_tree_counts"
            ]
    past_kv = "({},)".format(", ".join(f"past_key_values_{i}"
                                       for i in range(Na))) if Na else "()"
    conv = "({},)".format(", ".join(f"conv_state_{i}"
                                    for i in range(Nm))) if Nm else "()"
    recurrent = "({},)".format(", ".join(f"recurrent_state_{i}"
                                         for i in range(Nm))) if Nm else "()"
    tree_kwargs = (", attention_position_ids=attention_position_ids"
                   ", packed_attention_mask=packed_attention_mask"
                   if tree_attention else "")
    if tree_attention and tree_state:
        tree_kwargs += (", tree_parent_ids=tree_parent_ids"
                        ", tree_depths=tree_depths"
                        ", valid_tree_counts=valid_tree_counts")
    body = (
        f"    outputs = self._model.forward_ragged(\n"
        f"        inputs_embeds, {past_kv}, rope_rotary_cos_sin, positions, "
        f"query_start_offsets, query_lengths, past_lengths, "
        f"attention_sequence_lengths, state_indices, "
        f"execution_phase_marker, context_sequence_count_carrier, "
        f"kv_page_table, logits_indices, {conv}, "
        f"{recurrent}{tree_kwargs})\n"
        f"    (logits, emitted_hidden, present_kv, present_conv, present_ssm, "
        f"inter_conv, replay_da, replay_u, replay_b, replay_dt) = outputs\n"
        f"    flat = (logits,)\n"
        f"    if emitted_hidden is not None:\n"
        f"        flat += (emitted_hidden,)\n"
        f"    return (flat + tuple(present_kv) + tuple(present_conv) + "
        f"tuple(present_ssm) + tuple(inter_conv) + tuple(replay_da) + "
        f"tuple(replay_u) + tuple(replay_b) + tuple(replay_dt))\n")
    globs: dict = {}
    exec("def _forward(self, {}):\n{}".format(", ".join(param_names), body),
         globs)  # noqa: S102

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
        execution_phase_marker: Optional[torch.Tensor] = None,
        tree_parent_ids: Optional[torch.Tensor] = None,
        tree_depths: Optional[torch.Tensor] = None,
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
                 execution_phase_marker=execution_phase_marker,
                 tree_parent_ids=tree_parent_ids,
                 tree_depths=tree_depths,
                 use_ddtree_state=tree_parent_ids is not None,
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
            (ssm_output, ssm_state_out, replay_da, replay_u, replay_b,
             replay_dt) = update_ssm_state_with_intermediate(
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
                 execution_phase_marker,
                 dt_softplus=1,
                 ngroups=self.n_groups,
                 tree_parent_ids=tree_parent_ids,
                 tree_depths=tree_depths,
                 use_ddtree_state=tree_parent_ids is not None,
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
                    intermediate_conv_state, replay_da, replay_u, replay_b,
                    replay_dt)
        return self.out_proj(normed), conv_state_out, ssm_state_out

    def forward_ragged(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        query_lengths: torch.Tensor,
        query_start_offsets: torch.Tensor,
        state_indices: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        collect_intermediate_states: bool = False,
        tree_parent_ids: Optional[torch.Tensor] = None,
        tree_depths: Optional[torch.Tensor] = None,
    ):
        d_inner = self.num_heads * self.head_dim
        d_state = self.n_groups * self.ssm_state_size
        gate, conv_input, dt = self.in_proj(hidden_states).split(
            [d_inner, self.conv_dim, self.num_heads], dim=-1)
        dt = dt.clamp(-self._DT_CLAMP, self._DT_CLAMP)
        if collect_intermediate_states:
            conv_output, conv_state_out, intermediate_conv_state = causal_conv1d_with_intermediate(
                conv_input,
                self.conv1d.weight,
                self.conv1d.bias,
                conv_state,
                query_lengths,
                query_start_offsets,
                state_indices,
                stride=1,
                padding=self.conv_kernel - 1,
                dilation=1,
                groups=self.conv_dim,
                execution_phase_marker=execution_phase_marker,
                context_sequence_count_carrier=context_sequence_count_carrier,
                tree_parent_ids=tree_parent_ids,
                tree_depths=tree_depths,
                use_ddtree_state=tree_parent_ids is not None,
            )
        else:
            conv_output, conv_state_out, _ = causal_conv1d(
                conv_input,
                self.conv1d.weight,
                self.conv1d.bias,
                conv_state,
                query_lengths,
                stride=1,
                padding=self.conv_kernel - 1,
                dilation=1,
                groups=self.conv_dim,
                query_start_offsets=query_start_offsets,
                state_indices=state_indices,
                execution_phase_marker=execution_phase_marker,
                context_sequence_count_carrier=context_sequence_count_carrier,
            )
        ssm_input, ssm_b, ssm_c = F.silu(conv_output).split(
            [d_inner, d_state, d_state], dim=-1)
        ssm_input = ssm_input.view(-1, self.num_heads, self.head_dim)
        ssm_b = ssm_b.view(-1, self.n_groups, self.ssm_state_size)
        ssm_c = ssm_c.view(-1, self.n_groups, self.ssm_state_size)
        ssm_a = -torch.exp(self.A_log.to(torch.float32))
        common_args = (ssm_input, ssm_a, ssm_b, ssm_c, self.D, dt,
                       self.dt_bias, ssm_state, query_lengths,
                       query_start_offsets, state_indices,
                       execution_phase_marker, context_sequence_count_carrier)
        if collect_intermediate_states:
            (ssm_output, ssm_state_out, replay_da, replay_u, replay_b,
             replay_dt) = (update_ssm_state_with_intermediate(
                 *common_args,
                 dt_softplus=1,
                 ngroups=self.n_groups,
                 tree_parent_ids=tree_parent_ids,
                 tree_depths=tree_depths,
                 use_ddtree_state=tree_parent_ids is not None))
        else:
            ssm_output, ssm_state_out = update_ssm_state(*common_args,
                                                         dt_softplus=1,
                                                         ngroups=self.n_groups)
        normed = self._gated_rmsnorm(ssm_output.view(-1, d_inner), gate)
        if collect_intermediate_states:
            return (self.out_proj(normed), conv_state_out, ssm_state_out,
                    intermediate_conv_state, replay_da, replay_u, replay_b,
                    replay_dt)
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

        # W4A16 experts go through an A16 MoE plugin: Marlin by default,
        # Nvfp4A16BlackwellMoePlugin for an explicit SM110 export target.
        self._is_a16 = config.quant.quant_type == QUANT_NVFP4_A16
        # Latched in ``_prepare_for_export_a16``, re-checked in ``forward``.
        self._use_blackwell_a16_moe: Optional[bool] = None
        # ReLU2 (non-gated) FC1 padded to a 128 multiple (1856 -> 1920);
        # Marlin takes the padded size, Blackwell the logical one.
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
        """Stack routed-expert NVFP4 (W4A16) weights for the A16 MoE plugin.

        An explicit SM110 export target stacks into ``BLACKWELL_MOE_N128_K64_V1``
        for ``Nvfp4A16BlackwellMoePlugin`` (FC1 N padding lives inside the
        layout; per-expert global scales stay FP32). Every other target keeps
        the Marlin ``Nvfp4A16MoePlugin`` stack (FC1 N and FC2 K padded to
        ``_a16_moe_inter_padded``; FP16 pre-scaled global scales). The buffer
        names are shared; ``forward`` re-checks the latched route.
        """
        from ...checkpoint.repacking import (
            repack_nvfp4_a16_blackwell_moe_experts,
            repack_nvfp4_a16_marlin_moe_experts)

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
        self._use_blackwell_a16_moe = use_blackwell_nvfp4_a16_moe()
        if self._use_blackwell_a16_moe:
            (fc1_qweights, fc1_block_scales, fc1_global, fc2_qweights,
             fc2_block_scales,
             fc2_global) = repack_nvfp4_a16_blackwell_moe_experts(
                 fc1_p, fc1_s, fc1_g, fc2_p, fc2_s, fc2_g)
        else:
            (fc1_qweights, fc1_block_scales, fc1_global, fc2_qweights,
             fc2_block_scales,
             fc2_global) = repack_nvfp4_a16_marlin_moe_experts(
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
        # Router logits are computed from the original hidden states. When
        # moe_latent_size is set, only the routed expert payload is projected
        # down into latent space.
        router_logits = F.linear(hidden_states.view(-1, self.hidden_size),
                                 self.gate.weight).float()
        routed_hidden_states = self.fc1_latent_proj(hidden_states)

        if self._is_a16:
            use_blackwell = use_blackwell_nvfp4_a16_moe()
            if self._use_blackwell_a16_moe is None:
                raise RuntimeError(
                    "NemotronHMoEMLP NVFP4-A16 forward requires "
                    "prepare_for_export first (no stacked expert buffers)")
            if self._use_blackwell_a16_moe != use_blackwell:
                raise ValueError(
                    "NemotronHMoEMLP NVFP4-A16 plugin route changed between "
                    f"export preparation ({self._use_blackwell_a16_moe}) and "
                    f"forward ({use_blackwell})")
            if use_blackwell:
                # SM110: BLACKWELL_MOE_N128_K64_V1 buffers, FP32 global
                # scales, and the logical intermediate size (the layout pads
                # FC1 N internally). FP16 hidden states and output.
                moe_out = nvfp4_a16_blackwell_moe_plugin(
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
                    self.moe_intermediate_size,
                    self.activation_type,
                    self.gate.n_group,
                    self.gate.topk_group,
                    int(bool(self.gate.norm_topk_prob)),
                    float(self.gate.routed_scaling_factor),
                    _NVFP4_ROUTING_MODE_SIGMOID_GROUP_TOPK,
                    self.max_routed_rows,
                    self.backend,
                )
            else:
                # FP16 activations end-to-end: the Marlin MoE kernel has an
                # FP16 E2M1 path, so hidden states / global scales / output
                # stay FP16.
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

    def forward_ragged(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
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
        attention_position_ids: Optional[torch.Tensor] = None,
        packed_attention_mask: Optional[torch.Tensor] = None,
        tree_parent_ids: Optional[torch.Tensor] = None,
        tree_depths: Optional[torch.Tensor] = None,
        valid_tree_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv = torch.cat([
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states)
        ],
                        dim=-1)
        kwargs: dict = {
            "num_q_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_dim,
            "sliding_window_size": self.sliding_window_size,
            "enable_tree_attention": packed_attention_mask is not None,
            "enable_fp8_kv_cache": self.enable_fp8_kv_cache,
            "attention_scale": self.attention_scale,
            "enable_context_mask_selector": False,
            "enable_vision_block_attention": False,
            "skip_softmax_scale_factor": 0.0,
            "qkv_scales": getattr(self, "_qkv_scales_float", [1.0, 1.0, 1.0]),
            "query_start_offsets": query_start_offsets,
            "attention_sequence_lengths": attention_sequence_lengths,
            "execution_phase_marker": execution_phase_marker,
            "context_sequence_count_carrier": context_sequence_count_carrier,
        }
        if packed_attention_mask is not None:
            kwargs.update(attention_mask=packed_attention_mask,
                          attention_pos_id=attention_position_ids)
        attn_output, present_key_value = attention_plugin(
            qkv, past_key_value, query_lengths, rope_rotary_cos_sin,
            past_lengths, kv_page_table, **kwargs)
        attn_output = attn_output.reshape(hidden_states.shape[0],
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
        execution_phase_marker: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_pos_id: Optional[torch.Tensor] = None,
        tree_parent_ids: Optional[torch.Tensor] = None,
        tree_depths: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        normed = self.norm(hidden_states)
        if self.layer_type == LAYER_MAMBA:
            if collect_intermediate_states:
                (mixer_out, conv_state_out, ssm_state_out, inter_conv,
                 replay_da, replay_u, replay_b, replay_dt) = self.mixer(
                     normed,
                     conv_state,
                     ssm_state,
                     context_lengths,
                     kvcache_start_index,
                     collect_intermediate_states=True,
                     execution_phase_marker=execution_phase_marker,
                     tree_parent_ids=tree_parent_ids,
                     tree_depths=tree_depths)
                return (residual + mixer_out, conv_state_out, ssm_state_out,
                        inter_conv, replay_da, replay_u, replay_b, replay_dt)
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

    def forward_ragged(self, hidden_states: torch.Tensor, **kwargs):
        residual = hidden_states
        normed = self.norm(hidden_states)
        if self.layer_type == LAYER_MAMBA:
            outputs = self.mixer.forward_ragged(
                normed,
                kwargs["conv_state"],
                kwargs["ssm_state"],
                kwargs["query_lengths"],
                kwargs["query_start_offsets"],
                kwargs["state_indices"],
                kwargs["execution_phase_marker"],
                kwargs["context_sequence_count_carrier"],
                collect_intermediate_states=kwargs.get(
                    "collect_intermediate_states", False),
                tree_parent_ids=kwargs.get("tree_parent_ids"),
                tree_depths=kwargs.get("tree_depths"),
            )
            return (residual + outputs[0], ) + outputs[1:]
        if self.layer_type in (LAYER_MLP, LAYER_MOE):
            return residual + self.mixer(normed)
        attn_out, present_kv = self.mixer.forward_ragged(
            normed,
            kwargs["past_key_value"],
            kwargs["rope_rotary_cos_sin"],
            kwargs["positions"],
            kwargs["query_start_offsets"],
            kwargs["query_lengths"],
            kwargs["past_lengths"],
            kwargs["attention_sequence_lengths"],
            kwargs["state_indices"],
            kwargs["execution_phase_marker"],
            kwargs["context_sequence_count_carrier"],
            kwargs["kv_page_table"],
            kwargs.get("attention_position_ids"),
            kwargs.get("packed_attention_mask"),
            kwargs.get("tree_parent_ids"),
            kwargs.get("tree_depths"),
            kwargs.get("valid_tree_counts"),
        )
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
        execution_phase_marker: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_pos_id: Optional[torch.Tensor] = None,
        tree_parent_ids: Optional[torch.Tensor] = None,
        tree_depths: Optional[torch.Tensor] = None,
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
        replay_dt_states_list: List[torch.Tensor] = []
        dflash_hidden_by_layer: dict[int, torch.Tensor] = {}
        dflash_target_set = set(dflash_target_layer_ids or [])
        last_layer_idx = len(self.layers) - 1
        attn_idx = 0
        mamba_idx = 0

        for layer_idx, (layer,
                        lt) in enumerate(zip(self.layers, self.layer_types)):
            if lt == LAYER_MAMBA:
                if collect_intermediate_states:
                    (hidden_states, conv_out, ssm_out, inter_conv, replay_da,
                     replay_u, replay_b, replay_dt) = layer(
                         hidden_states,
                         context_lengths=context_lengths,
                         kvcache_start_index=kvcache_start_index,
                         conv_state=conv_states[mamba_idx],
                         ssm_state=ssm_states[mamba_idx],
                         collect_intermediate_states=True,
                         execution_phase_marker=execution_phase_marker,
                         tree_parent_ids=tree_parent_ids,
                         tree_depths=tree_depths,
                     )
                    intermediate_conv_states_list.append(inter_conv)
                    replay_da_states_list.append(replay_da)
                    replay_u_states_list.append(replay_u)
                    replay_b_states_list.append(replay_b)
                    replay_dt_states_list.append(replay_dt)
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
                dflash_hidden_by_layer[layer_idx] = hidden_states

        normed = self.norm_f(hidden_states)
        if last_layer_idx in dflash_target_set:
            dflash_hidden_by_layer[last_layer_idx] = normed
        dflash_hidden_concat = _concat_hidden_in_provider_order(
            dflash_hidden_by_layer, dflash_target_layer_ids)
        if collect_intermediate_states:
            return (normed, tuple(present_key_values_list),
                    tuple(present_conv_states_list),
                    tuple(present_ssm_states_list),
                    tuple(intermediate_conv_states_list),
                    tuple(replay_da_states_list), tuple(replay_u_states_list),
                    tuple(replay_b_states_list), tuple(replay_dt_states_list),
                    dflash_hidden_concat)
        return (normed, tuple(present_key_values_list),
                tuple(present_conv_states_list),
                tuple(present_ssm_states_list), dflash_hidden_concat)

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
        conv_states: Tuple[torch.Tensor, ...],
        ssm_states: Tuple[torch.Tensor, ...],
        collect_intermediate_states: bool = False,
        attention_position_ids: Optional[torch.Tensor] = None,
        packed_attention_mask: Optional[torch.Tensor] = None,
        tree_parent_ids: Optional[torch.Tensor] = None,
        tree_depths: Optional[torch.Tensor] = None,
        valid_tree_counts: Optional[torch.Tensor] = None,
        dflash_target_layer_ids: Optional[List[int]] = None,
    ):
        hidden_states = inputs_embeds
        present_key_values = []
        present_conv_states = []
        present_ssm_states = []
        intermediate_conv_states = []
        replay_da_states = []
        replay_u_states = []
        replay_b_states = []
        replay_dt_states = []
        dflash_hidden_by_layer: dict[int, torch.Tensor] = {}
        dflash_targets = set(dflash_target_layer_ids or [])
        last_layer_idx = len(self.layers) - 1
        attn_idx = 0
        mamba_idx = 0
        common = dict(
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
            collect_intermediate_states=collect_intermediate_states,
            attention_position_ids=attention_position_ids,
            packed_attention_mask=packed_attention_mask,
            tree_parent_ids=tree_parent_ids,
            tree_depths=tree_depths,
            valid_tree_counts=valid_tree_counts,
        )
        for layer_idx, (layer, layer_type) in enumerate(
                zip(self.layers, self.layer_types)):
            if layer_type == LAYER_MAMBA:
                outputs = layer.forward_ragged(
                    hidden_states,
                    conv_state=conv_states[mamba_idx],
                    ssm_state=ssm_states[mamba_idx],
                    **common)
                hidden_states, conv_out, ssm_out = outputs[:3]
                present_conv_states.append(conv_out)
                present_ssm_states.append(ssm_out)
                if collect_intermediate_states:
                    intermediate_conv_states.append(outputs[3])
                    replay_da_states.append(outputs[4])
                    replay_u_states.append(outputs[5])
                    replay_b_states.append(outputs[6])
                    replay_dt_states.append(outputs[7])
                mamba_idx += 1
            elif layer_type in (LAYER_MLP, LAYER_MOE):
                hidden_states = layer.forward_ragged(hidden_states, **common)
            else:
                hidden_states, present_kv = layer.forward_ragged(
                    hidden_states,
                    past_key_value=past_key_values[attn_idx],
                    **common)
                present_key_values.append(present_kv)
                attn_idx += 1
            if layer_idx in dflash_targets and layer_idx != last_layer_idx:
                dflash_hidden_by_layer[layer_idx] = hidden_states
        normed = self.norm_f(hidden_states)
        if last_layer_idx in dflash_targets:
            dflash_hidden_by_layer[last_layer_idx] = normed
        dflash_hidden_concat = _concat_hidden_in_provider_order(
            dflash_hidden_by_layer, dflash_target_layer_ids)
        return (normed, tuple(present_key_values), tuple(present_conv_states),
                tuple(present_ssm_states), tuple(intermediate_conv_states),
                tuple(replay_da_states), tuple(replay_u_states),
                tuple(replay_b_states), tuple(replay_dt_states),
                dflash_hidden_concat)


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

    def materialize_checkpoint_defaults(self, device: str) -> None:
        """Materialize the optional Q-cache scale omitted by KV-only checkpoints."""
        for layer in self.backbone.layers:
            mixer = layer.mixer
            q_proj = getattr(mixer, "q_proj", None)
            q_scale = getattr(q_proj, "q_scale", None)
            if q_scale is not None and q_scale.device.type == "meta":
                q_proj.q_scale = torch.ones(q_scale.shape,
                                            dtype=q_scale.dtype,
                                            device=device)

    def onnx_export_spec(self) -> OnnxSpec:
        """Return all model-specific parameters needed for ONNX export."""
        # Pre-process MoE layers: reinterpret FP8 scales as INT8
        for layer in self.backbone.layers:
            if hasattr(layer.mixer, 'prepare_for_export'):
                layer.mixer.prepare_for_export()
        return self._token_major_onnx_export_spec()

    def _token_major_onnx_export_spec(self) -> OnnxSpec:
        config = self.config
        mc = config.mamba_cfg
        assert mc is not None
        Na = config.num_attn_layers
        Nm = config.num_mamba_layers
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        num_sequences = 2
        query_length = 2
        physical_tokens = num_sequences * query_length
        resident_rows = 4
        spec = bool(
            getattr(config, "mtp_base", False)
            or getattr(config, "dflash_base", False)
            or getattr(config, "dspark_base", False))
        tree_state = bool(
            getattr(config, "dflash_base", False)
            or getattr(config, "mtp_tree_base", False)
            or getattr(config, "dflash_tree_base", False)
            or getattr(config, "dspark_tree_base", False))
        inputs_embeds = torch.zeros(physical_tokens,
                                    config.hidden_size,
                                    dtype=torch.float16,
                                    device=device)
        kv_dtype = (torch.float8_e4m3fn
                    if config.quant.kv_cache_quant == "fp8" else torch.float16)
        past_key_values = [
            torch.zeros(2,
                        2,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        config.head_dim,
                        dtype=kv_dtype,
                        device=device) for _ in range(Na)
        ]
        rope = torch.zeros(physical_tokens,
                           config.head_dim,
                           dtype=torch.float32,
                           device=device)
        positions = torch.arange(query_length,
                                 dtype=torch.int32,
                                 device=device).repeat(num_sequences)
        query_start_offsets = torch.arange(0,
                                           physical_tokens + 1,
                                           query_length,
                                           dtype=torch.int32,
                                           device=device)
        query_lengths = torch.full((num_sequences, ),
                                   query_length,
                                   dtype=torch.int32,
                                   device=device)
        past_lengths = torch.zeros(num_sequences,
                                   dtype=torch.int32,
                                   device=device)
        attention_sequence_lengths = query_lengths.clone()
        state_indices = torch.arange(num_sequences,
                                     dtype=torch.int32,
                                     device=device)
        execution_phase_marker = torch.zeros(2,
                                             dtype=torch.int32,
                                             device=device)
        context_sequence_count_carrier = torch.zeros(num_sequences,
                                                     dtype=torch.int32,
                                                     device=device)
        kv_page_table = torch.zeros(num_sequences,
                                    2,
                                    2,
                                    dtype=torch.int32,
                                    device=device)
        logits_indices = (torch.tensor(
            [0, 2, 3], dtype=torch.int64, device=device) if spec else
                          query_start_offsets[1:].to(torch.int64) - 1)
        conv_states = [
            torch.zeros(resident_rows,
                        mc.conv_dim,
                        mc.conv_kernel,
                        dtype=self.CONV_STATE_DTYPE,
                        device=device) for _ in range(Nm)
        ]
        ssm_states = [
            torch.zeros(resident_rows,
                        mc.num_heads,
                        mc.head_dim,
                        mc.ssm_state_size,
                        dtype=self.RECURRENT_STATE_DTYPE,
                        device=device) for _ in range(Nm)
        ]
        args = (inputs_embeds, *past_key_values, rope, positions,
                query_start_offsets, query_lengths, past_lengths,
                attention_sequence_lengths, state_indices,
                execution_phase_marker, context_sequence_count_carrier,
                kv_page_table, logits_indices, *conv_states, *ssm_states)
        input_names = (
            ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
                "rope_rotary_cos_sin", "positions", "query_start_offsets",
                "query_lengths", "past_lengths", "attention_sequence_lengths",
                "state_indices", "execution_phase_marker",
                "context_sequence_count_carrier", "kv_page_table",
                "logits_indices"
            ] + [f"conv_state_{i}" for i in range(Nm)] +
            [f"recurrent_state_{i}" for i in range(Nm)])
        output_names = ["logits"]
        if spec:
            output_names.append("hidden_states")
        output_names += ([f"present_key_values_{i}" for i in range(Na)] +
                         [f"present_conv_state_{i}" for i in range(Nm)] +
                         [f"present_recurrent_state_{i}" for i in range(Nm)])
        if spec:
            output_names += (
                [f"intermediate_conv_state_{i}" for i in range(Nm)] +
                [f"replay_da_state_{i}" for i in range(Nm)] +
                [f"replay_u_state_{i}" for i in range(Nm)] +
                [f"replay_b_state_{i}" for i in range(Nm)] +
                [f"replay_dt_state_{i}" for i in range(Nm)])
        tokens = torch.export.Dim("physical_tokens", min=1, max=8_388_608)
        logits_rows = torch.export.Dim("logits_rows", min=1, max=8_388_608)
        sequences = torch.export.Dim("num_sequences", min=1, max=256)
        context_sequences = torch.export.Dim("num_context_sequences",
                                             min=0,
                                             max=256)
        resident = torch.export.Dim("resident_rows", min=1, max=256)
        num_pages = torch.export.Dim("num_pages", min=1, max=1_048_576)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        phase_extent = torch.export.Dim("execution_phase_extent", min=1, max=8)
        all_shapes = [{0: tokens}]
        all_shapes.extend({1: num_pages} for _ in range(Na))
        all_shapes.extend([
            {
                0: tokens
            },
            {
                0: tokens
            },
            {
                0: sequences + 1
            },
            {
                0: sequences
            },
            {
                0: sequences
            },
            {
                0: sequences
            },
            {
                0: sequences
            },
            {
                0: phase_extent
            },
            {
                0: context_sequences
            },
            {
                0: sequences,
                2: max_pages
            },
            {
                0: logits_rows if spec else sequences
            },
        ])
        all_shapes.extend({0: resident} for _ in range(2 * Nm))
        if spec:
            packed_width = torch.export.Dim("packed_mask_width", min=1, max=64)
            attention_position_ids = positions.clone()
            packed_attention_mask = torch.zeros(physical_tokens,
                                                (query_length + 31) // 32,
                                                dtype=torch.int32,
                                                device=device)
            args += (attention_position_ids, packed_attention_mask)
            input_names += ["attention_position_ids", "packed_attention_mask"]
            all_shapes.extend([{0: tokens}, {0: tokens, 1: packed_width}])
            if tree_state:
                tree_parent_ids = torch.full((physical_tokens, ),
                                             -1,
                                             dtype=torch.int32,
                                             device=device)
                tree_depths = torch.zeros(physical_tokens,
                                          dtype=torch.int32,
                                          device=device)
                valid_tree_counts = query_lengths.clone()
                args += (tree_parent_ids, tree_depths, valid_tree_counts)
                input_names += [
                    "tree_parent_ids", "tree_depths", "valid_tree_counts"
                ]
                all_shapes.extend([{0: tokens}, {0: tokens}, {0: sequences}])
        wrapped = _make_flat_wrapper_mamba_ragged(self, Na, Nm, spec,
                                                  tree_state)
        wrapped.eval()
        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=all_shapes)

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
        conv_states: Tuple[torch.Tensor, ...] = (),
        ssm_states: Tuple[torch.Tensor, ...] = (),
        attention_position_ids: Optional[torch.Tensor] = None,
        packed_attention_mask: Optional[torch.Tensor] = None,
        tree_parent_ids: Optional[torch.Tensor] = None,
        tree_depths: Optional[torch.Tensor] = None,
        valid_tree_counts: Optional[torch.Tensor] = None,
    ) -> Tuple:
        mtp_base = bool(getattr(self.config, "mtp_base", False))
        dflash_base = bool(getattr(self.config, "dflash_base", False))
        dspark_base = bool(getattr(self.config, "dspark_base", False))
        target_hidden_base = dflash_base or dspark_base
        spec = mtp_base or target_hidden_base
        target_layer_ids = (
            self.config.dflash_target_layer_ids if dflash_base else
            self.config.dspark_target_layer_ids if dspark_base else None)
        outputs = self.backbone.forward_ragged(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin,
            positions,
            query_start_offsets,
            query_lengths,
            past_lengths,
            attention_sequence_lengths,
            state_indices,
            execution_phase_marker,
            context_sequence_count_carrier,
            kv_page_table,
            conv_states,
            ssm_states,
            collect_intermediate_states=spec,
            attention_position_ids=attention_position_ids,
            packed_attention_mask=packed_attention_mask,
            tree_parent_ids=tree_parent_ids,
            tree_depths=tree_depths,
            valid_tree_counts=valid_tree_counts,
            dflash_target_layer_ids=target_layer_ids)
        (hidden_states, present_kv, present_conv, present_ssm, inter_conv,
         replay_da, replay_u, replay_b, replay_dt, dflash_hidden) = outputs
        selected_hidden_states = torch.index_select(hidden_states, 0,
                                                    logits_indices)
        logits = self.lm_head(selected_hidden_states).to(torch.float32)
        emitted_hidden = dflash_hidden if target_hidden_base else (
            hidden_states if mtp_base else None)
        return (logits, emitted_hidden, present_kv, present_conv, present_ssm,
                inter_conv, replay_da, replay_u, replay_b, replay_dt)

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
        execution_phase_marker: Optional[torch.Tensor] = None,
        tree_parent_ids: Optional[torch.Tensor] = None,
        tree_depths: Optional[torch.Tensor] = None,
    ) -> Tuple:
        mtp_base = bool(getattr(self.config, "mtp_base", False))
        dflash_base = bool(getattr(self.config, "dflash_base", False))
        dspark_base = bool(getattr(self.config, "dspark_base", False))
        target_hidden_base = dflash_base or dspark_base
        dflash_target_ids = (
            self.config.dflash_target_layer_ids if dflash_base else
            self.config.dspark_target_layer_ids if dspark_base else None)
        spec = mtp_base or target_hidden_base
        if spec:
            (hidden_states, present_key_values, present_conv_states,
             present_ssm_states, intermediate_conv_states, replay_da_states,
             replay_u_states, replay_b_states, replay_dt_states,
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
                 execution_phase_marker=execution_phase_marker,
                 attention_mask=attention_mask,
                 attention_pos_id=attention_pos_id,
                 tree_parent_ids=tree_parent_ids,
                 tree_depths=tree_depths,
                 dflash_target_layer_ids=dflash_target_ids,
             )
            # Draft consumes the full pre-lm_head hidden; logits use the
            # gathered predicted-token positions.
            selected = torch.ops.trt.gather_nd(hidden_states, last_token_ids)
            logits = self.lm_head(selected).to(torch.float32)
            if target_hidden_base:
                return (logits, dflash_hidden_concat, present_key_values,
                        present_conv_states, present_ssm_states,
                        intermediate_conv_states, replay_da_states,
                        replay_u_states, replay_b_states, replay_dt_states)
            return (logits, hidden_states, present_key_values,
                    present_conv_states, present_ssm_states,
                    intermediate_conv_states, replay_da_states,
                    replay_u_states, replay_b_states, replay_dt_states)

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
