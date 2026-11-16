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
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import (LAYER_ATTN, LAYER_MAMBA, LAYER_MLP, LAYER_MOE,
                       QUANT_FP16, QUANT_IQ2_XXS, QUANT_NVFP4, QUANT_NVFP4_A16,
                       MambaConfig, ModelConfig)
from ..default.modeling_default import (OnnxSpec,
                                        _concat_hidden_in_provider_order)
from ..linear import FP8Linear, FP16Linear, Q8_0Linear, make_linear
# yapf: disable
from ..ops import (KV_PAGE_SIZE, TRT_DATATYPE_FP4, TRT_DATATYPE_FP8,
                   TRT_MOE_ACT_RELU2, TRT_MOE_QUANT_DYNAMIC_DBLQ,
                   TRT_MOE_QUANT_NONE, TRT_MOE_QUANT_STATIC, attention_plugin,
                   causal_conv1d, causal_conv1d_with_intermediate,
                   fp8_dequantize, fp8_quantize, fp16_moe_plugin_sigmoid,
                   is_trt_native_moe_enabled, is_trt_native_moe_w4a4_enabled,
                   nvfp4_a16_blackwell_moe_plugin, nvfp4_a16_moe_plugin,
                   nvfp4_act_qdq_moe, nvfp4_dequantize_moe,
                   nvfp4_dequantize_moe_with_global_scale, nvfp4_moe_plugin,
                   nvfp4_moe_plugin_geforce, trt_moe_fp16_nongated,
                   trt_moe_w4a4_nongated, trt_moe_w4a8_nongated,
                   update_ssm_state, update_ssm_state_with_intermediate,
                   use_blackwell_nvfp4_a16_moe, use_geforce_nvfp4_moe)

# yapf: enable

logger = logging.getLogger(__name__)

_NVFP4_ACTIVATION_RELU2 = 4
_NVFP4_ROUTING_MODE_SIGMOID_GROUP_TOPK = 1
_NVFP4_MOE_BACKEND_AUTO = 0
_NVFP4_MOE_IO_DTYPE_FP16 = 1
_NVFP4_MOE_MAX_ROUTED_ROWS_AUTO = 0
# NVFP4 block size along the input (K) axis; TRT_BlockDequantize block_shape [1, 1, 16].
_NVFP4_BLOCK_SIZE = 16


def _stack_nvfp4_moe_input_scales(
        experts: nn.ModuleList) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect calibrated non-gated FC1 and FC2 activation scales."""
    fc1_scales = []
    fc2_scales = []
    for expert_idx, expert in enumerate(experts):
        up_scale = expert.up_proj.input_scale.detach().float().reshape(-1)
        down_scale = expert.down_proj.input_scale.detach().float().reshape(-1)
        if up_scale.numel() != 1 or down_scale.numel() != 1:
            raise ValueError("Nvfp4MoePlugin requires scalar input scales; "
                             f"expert {expert_idx} has non-scalar values")
        fc1_scales.append(up_scale[0])
        fc2_scales.append(down_scale[0])
    return torch.stack(fc1_scales), torch.stack(fc2_scales)


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


def _make_flat_wrapper_mamba_ragged(model: nn.Module, Na: int, Nm: int,
                                    tree_attention: bool) -> nn.Module:
    param_names = (
        ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
            "rope_rotary_cos_sin", "positions", "query_start_offsets",
            "query_lengths", "past_lengths", "attention_sequence_lengths",
            "state_indices", "execution_phase_marker",
            "context_sequence_count_carrier", "kv_page_table", "logits_indices"
        ] + [f"conv_state_{i}"
             for i in range(Nm)] + [f"recurrent_state_{i}" for i in range(Nm)])
    if tree_attention:
        param_names += [
            "attention_position_ids", "packed_attention_mask",
            "tree_parent_ids", "tree_depths", "valid_tree_counts",
            "spec_verify_layout"
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
    if tree_attention:
        tree_kwargs += (
            ", tree_parent_ids=tree_parent_ids"
            ", spec_verify_layout=spec_verify_layout, tree_depths=tree_depths"
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

    def prepare_for_export(self) -> None:
        """Materialize FP8 partitions before tracing unsupported constant splits."""
        if isinstance(self.in_proj, FP8Linear):
            sizes = [
                self.num_heads * self.head_dim, self.conv_dim, self.num_heads
            ]
            for name, weight in zip(
                ("gate_weight", "conv_weight", "dt_weight"),
                    self.in_proj.weight.split(sizes, dim=0)):
                self.register_buffer(name,
                                     weight.contiguous(),
                                     persistent=False)

    def _project_inputs(self, hidden_states: torch.Tensor):
        sizes = [self.num_heads * self.head_dim, self.conv_dim, self.num_heads]
        if not isinstance(self.in_proj, FP8Linear):
            return self.in_proj(hidden_states).split(sizes, dim=-1)

        hidden_states = self.in_proj.prepare_input(hidden_states)
        weights = ((self.gate_weight, self.conv_weight,
                    self.dt_weight) if hasattr(self, "gate_weight") else
                   self.in_proj.weight.split(sizes, dim=0))
        gate_weight, conv_weight, dt_weight = (fp8_dequantize(
            weight, self.in_proj.weight_scale) for weight in weights)
        biases = (self.in_proj.bias.to(torch.float16).split(sizes)
                  if self.in_proj.bias is not None else (None, None, None))
        gate = F.linear(hidden_states, gate_weight, biases[0])
        dt = F.linear(hidden_states, dt_weight, biases[2])
        # TRT can retain the fused GEMM's row stride for a split plugin input.
        # A 1x1 projection gives the convolution plugin a dense channel buffer.
        input_size = hidden_states.shape[-1]
        conv = F.conv2d(hidden_states.reshape(-1, input_size, 1, 1),
                        conv_weight.reshape(self.conv_dim, input_size, 1, 1),
                        biases[1])
        return gate, conv.reshape(*hidden_states.shape[:-1], self.conv_dim), dt

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
        spec_verify_layout: "torch.Tensor | None" = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        d_inner = self.num_heads * self.head_dim
        d_state = self.n_groups * self.ssm_state_size

        gate, hidden_states_for_conv, dt = self._project_inputs(hidden_states)
        dt = dt.clamp(-self._DT_CLAMP, self._DT_CLAMP)

        # Verification keeps committed state read-only and emits conv snapshots plus recurrent replay data.
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
                 spec_verify_layout=spec_verify_layout,
                 tree_depths=tree_depths,
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
                 spec_verify_layout=spec_verify_layout,
                 tree_depths=tree_depths,
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
        spec_verify_layout: "torch.Tensor | None" = None,
    ):
        d_inner = self.num_heads * self.head_dim
        d_state = self.n_groups * self.ssm_state_size
        gate, conv_input, dt = self._project_inputs(hidden_states)
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
                spec_verify_layout=spec_verify_layout,
                tree_depths=tree_depths,
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
                 spec_verify_layout=spec_verify_layout,
                 tree_depths=tree_depths,
             ))
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


def trt_native_sigmoid_group_topk_routing(
        router_logits: torch.Tensor, e_score_correction_bias: torch.Tensor,
        top_k: int, n_group: int, topk_group: int, norm_topk_prob: bool,
        routed_scaling_factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid group-top-k routing as graph ops, for the TRT-native MoE arm.

    ``TRT_MoE`` consumes PRECOMPUTED routing (the MoE plugins take the raw
    router logits and route inside the kernel), so the routing of
    :class:`NemotronHTopkRouter` is materialised in the ONNX graph here, same
    math, export-friendly ops: sigmoid scores, the correction bias added for
    the expert CHOICE only, the group mask, top-k, the UNBIASED scores of the
    chosen experts, optional renormalisation and ``routed_scaling_factor``.
    The group stage is a no-op for ``n_group == 1`` (Nemotron-3-Nano) and is
    skipped then. Returns ``(expert ids int32 [T, K], weights fp32 [T, K])``;
    the weights stay fp32 because Myelin rejects fp16 scores.
    """
    scores = torch.sigmoid(router_logits.float())
    # Plain [T, E] + [E] broadcast: an op on the bias buffer would be folded
    # into an anonymous initializer by the exporter.
    scores_for_choice = scores + e_score_correction_bias.float()
    if n_group > 1:
        num_tokens, num_experts = scores.shape
        group_scores = scores_for_choice.view(num_tokens, n_group,
                                              -1).topk(2,
                                                       dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1)[1]
        group_mask = torch.zeros_like(group_scores).scatter(1, group_idx, 1.0)
        score_mask = group_mask.unsqueeze(-1).expand(
            num_tokens, n_group,
            num_experts // n_group).reshape(num_tokens, num_experts)
        scores_for_choice = scores_for_choice.masked_fill(
            score_mask == 0.0, 0.0)
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1)[1]
    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) +
                                       1e-20)
    topk_weights = topk_weights * float(routed_scaling_factor)
    return topk_indices.to(torch.int32), topk_weights


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

    With ``--trt-native-moe`` the routed experts go through TensorRT's native
    ``TRT_MoE`` op instead (``IMoELayer`` non-gated form, kRELU2; FP16, W4A8
    or, with ``--trt-native-moe-w4a4``, W4A4 = quantization_mode 2) and the
    sigmoid group-top-k routing is computed in the graph; the shared expert
    and the latent projections are unchanged.

    Submodule names match checkpoint keys:
        gate                     - NemotronHTopkRouter (weight + bias)
        experts.{j}.up_proj     - per-expert up projection (NVFP4)
        experts.{j}.down_proj   - per-expert down projection (NVFP4)
        shared_experts.up_proj  - shared expert up projection (FP8)
        shared_experts.down_proj - shared expert down projection (FP8)
    """

    def __init__(self, config: ModelConfig, module_prefix: str) -> None:
        super().__init__()
        if config.quant.quant_type not in (QUANT_FP16, QUANT_NVFP4,
                                           QUANT_NVFP4_A16, QUANT_IQ2_XXS):
            raise ValueError(
                "Nemotron-H ONNX MoE supports unquantized (BF16/FP16), NVFP4 "
                "W4A4 or NVFP4 W4A16 experts; W8A16 and FP8 experts are not "
                "implemented")
        # --trt-native-moe: routed experts through TensorRT's native TRT_MoE
        # (IMoELayer::setWeights, the non-gated form, MoEActType::kRELU2)
        # instead of a plugin, routing materialised in the graph. Unquantized
        # experts take the FP16 form, NVFP4 experts the W4A8 form (or W4A4,
        # quantization_mode 2, with --trt-native-moe-w4a4); NVFP4-A16 has no
        # TRT_MoE form.
        self._use_native_moe = is_trt_native_moe_enabled()
        self._use_native_moe_w4a4 = is_trt_native_moe_w4a4_enabled()
        if (self._use_native_moe_w4a4
                and config.quant.quant_type != QUANT_NVFP4):
            raise ValueError(
                "--trt-native-moe-w4a4 requested but the Nemotron-H experts "
                f"are {config.quant.quant_type!r}; the W4A4 TRT_MoE form "
                "needs ModelOpt NVFP4 experts with quantized activations "
                "(unquantized experts have only the FP16 form, and "
                "--no-quantize-activations turns NVFP4 into NVFP4-A16, which "
                "has no TRT_MoE form). Drop --trt-native-moe-w4a4 to export "
                "the W4A8 form, or keep the activations quantized.")
        if self._use_native_moe and config.quant.quant_type not in (
                QUANT_FP16, QUANT_NVFP4):
            raise ValueError(
                "--trt-native-moe requested but the Nemotron-H experts are "
                f"{config.quant.quant_type!r}; only unquantized (BF16/FP16) "
                "and NVFP4 experts have a TRT_MoE form (NVFP4-A16 does not). "
                "Drop the flag to export the MoE plugin path.")
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
        # plugin output. The SM100/101/103/110 path keeps the original H.
        self._padded_hidden_size = self.routed_hidden_size
        self.gate = NemotronHTopkRouter(config)

        # W4A16 experts go through an A16 MoE plugin: Marlin by default,
        # Nvfp4A16BlackwellMoePlugin for an explicit SM110 export target.
        self._is_a16 = config.quant.quant_type == QUANT_NVFP4_A16
        # Unquantized (BF16/FP16) checkpoints have no quantization_config and
        # therefore no weight_scale tensors, so neither NVFP4 repack applies.
        # They route through the dense FP16 grouped-GEMM ``Fp16MoePlugin``
        # (sigmoid group-top-k variant).
        self._is_fp16 = config.quant.quant_type == QUANT_FP16
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
        """Pack the expert tensors for the active MoE arm (plugin or TRT_MoE)."""
        if self._use_native_moe:
            self._prepare_for_export_native()
            return
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

    def _prepare_for_export_fp16(self) -> None:
        """Stack unquantized routed experts for ``Fp16MoePlugin``.

        Nemotron-H experts are ReLU2 (non-gated): only ``up_proj``/``down_proj``
        exist, so ``repack_fp16_moe_experts`` emits ``fc1 = [E, I, H]`` with no
        gate rows and no up/gate interleave. ``I`` is zero-padded to a multiple
        of 128 to satisfy the plugin's ``FC1_N % 128 == 0`` contract; the
        padding is inert because ReLU2(0) = 0 and the matching FC2 columns are
        zero.
        """
        from ...checkpoint.repacking import repack_fp16_moe_experts

        fc1_weights, fc2_weights, padded_inter_size = repack_fp16_moe_experts(
            self.experts,
            self.routed_hidden_size,
            self.moe_intermediate_size,
            self.activation_type,
        )
        self._padded_moe_intermediate_size = padded_inter_size
        self._padded_hidden_size = self.routed_hidden_size

        device = self.gate.weight.device
        self.register_buffer("fc1_weights",
                             fc1_weights.to(device).contiguous())
        self.register_buffer("fc2_weights",
                             fc2_weights.to(device).contiguous())
        self.register_buffer(
            "_e_score_correction_bias_fp32",
            self.gate.e_score_correction_bias.data.clone().to(
                torch.float32).to(device))
        # Drop the per-expert modules (their weights are nn.Parameters, not
        # buffers, so ``_buffers.pop`` would not free them); forward uses the
        # stacked buffers.
        self.experts = nn.ModuleList()
        self._export_ready = True

    def _prepare_for_export_native(self) -> None:
        """Stack the routed experts into TRT-native ``TRT_MoE`` buffers.

        Non-gated ``IMoELayer`` form (``setWeights``): ``fc_up`` is
        ``[E, H, I]`` and ``fc_down`` is ``[E, I, H]``, the HF weights
        transposed, with no intermediate or hidden padding (the 128 / 256
        paddings of the plugin repacks are plugin contracts; H=2688 and
        I=1856 only need to be multiples of the NVFP4 block, 16).

        Unquantized experts are stacked as FP16. NVFP4 experts keep their
        FP4 codes and get one fp16 block scale that folds the E4M3 block scale
        with the per-expert ``weight_scale_2`` (see
        :func:`~tensorrt_edgellm.checkpoint.repacking.stack_nvfp4_moe_projection`);
        the calibrated ModelOpt ``input_scale`` values become the static FP8
        activation scales of the W4A8 form (``amax / 448 = input_scale * 6``).
        The FC1 input is shared by all experts, so the LARGEST scale is used;
        FC2 keeps a per-expert scale. The plugin arm instead quantizes the
        activations per token to NVFP4 in the kernel (W4A4).

        With ``--trt-native-moe-w4a4`` (W4A4, ``TRT_MoE`` quantization_mode
        2) the NVFP4 experts keep separate block and global scales (E4M3 block
        scales verbatim, fp16 per-expert ``weight_scale_2`` global) and the
        activations are quantized dynamically to NVFP4 with ModelOpt's
        ``input_scale`` VERBATIM as the second-level scale: the max over the
        experts for the shared FC1 input, per expert for the FC2 input
        (``TRT_MoE`` input 9). No ``* 6`` there; see
        :meth:`_prepare_for_export_native_w4a4`.
        """
        device = self.gate.weight.device
        if self._is_fp16:

            def _stack_t(proj_name):
                # nn.Linear weight is [out, in]; TRT_MoE wants [in, out].
                tensors = [
                    getattr(expert, proj_name).weight.data.t()
                    for expert in self.experts
                ]
                return torch.stack(tensors,
                                   dim=0).to(device=device,
                                             dtype=torch.float16).contiguous()

            self.register_buffer("fc_up_weights", _stack_t("up_proj"))
            self.register_buffer("fc_down_weights", _stack_t("down_proj"))
            logger.info(
                "TRT_MoE FP16 (non-gated ReLU2): %d experts stacked: up %s, "
                "down %s",
                self.n_routed_experts, list(self.fc_up_weights.shape),
                list(self.fc_down_weights.shape))
        elif self._use_native_moe_w4a4:
            self._prepare_for_export_native_w4a4(device)
        else:
            from ...checkpoint.repacking import stack_nvfp4_moe_projection

            # ``nvfp4_<proj>_weight``: packed FP4 codes [E, out, in // 2]
            # (int8; export.py reinterprets the initializer as FLOAT4E2M1 by
            # this name suffix); ``nvfp4_<proj>_block_scale``: fp16
            # [E, out, in // 16] with ``weight_scale_2`` folded in.
            for proj in ("up", "down"):
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
            logger.info(
                "TRT_MoE W4A8 (non-gated ReLU2): %d NVFP4 experts; "
                "act_scale=%.6g, down_act_scale[0:2]=%s",
                self.n_routed_experts, float(self.native_act_scale),
                self.native_down_act_scale[:2].tolist())

        self.register_buffer(
            "_e_score_correction_bias_fp32",
            self.gate.e_score_correction_bias.data.clone().to(
                torch.float32).to(device))
        # Drop the per-expert modules (their weights are nn.Parameters, not
        # buffers, so ``_buffers.pop`` would not free them); forward uses the
        # stacked buffers.
        self.experts = nn.ModuleList()
        self._export_ready = True

    def _prepare_for_export_native_w4a4(self, device: torch.device) -> None:
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
        passes skip and keep. Non-gated experts take no biases.
        """
        from ...checkpoint.repacking import \
            stack_nvfp4_moe_projection_with_global_scale

        for proj in ("up", "down"):
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
        logger.info(
            "TRT_MoE W4A4 (non-gated ReLU2, quantization_mode 2): %d NVFP4 "
            "experts; act_dblq_scale=%.6g, down_act_scale[0:2]=%s",
            self.n_routed_experts, float(self.native_act_dblq_scale),
            self.native_down_act_scale[:2].tolist())

    def _prepare_for_export_impl(self) -> None:
        """Pack the expert tensors for the active MoE plugin."""
        if self._is_fp16:
            self._prepare_for_export_fp16()
            return
        if self._is_a16:
            self._prepare_for_export_a16()
            return
        from ...checkpoint.repacking import repack_nvfp4_moe_experts

        # SM12x NvFP4MoEPluginGeforce requires H % 256 == 0; the SM100/101/103/110 path only needs
        # the kernel's regular alignment (the repack helper accepts H as-is
        # when ``hidden_size_alignment=1``).
        hidden_size_alignment = 256 if use_geforce_nvfp4_moe() else 1

        input_global_scale, down_input_scale = \
            _stack_nvfp4_moe_input_scales(self.experts)

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
        self.register_buffer("input_global_scale",
                             input_global_scale.to(device).contiguous())
        self.register_buffer("down_input_scale",
                             down_input_scale.to(device).contiguous())
        self.register_buffer(
            "_e_score_correction_bias_fp32",
            self.gate.e_score_correction_bias.data.clone().to(
                torch.float32).to(device))
        self._export_ready = True

    def _native_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """TRT-native (OOTB) MoE: routing in the graph, non-gated ``TRT_MoE`` experts.

        IMoELayer needs rank-3 ``[B, S, *]`` tensors while the token-major
        export hands the block ``[T, H]``; the layer is defined per token
        with no cross-token reduction, so all T tokens are wrapped as one
        sequence ``[1, T, H]`` / ``[1, T, K]`` and the output reshaped back
        (same as the Qwen3 native arm). The shared expert is added on top of
        the routed output exactly as in the plugin paths.
        """
        if not self._export_ready:
            raise RuntimeError(
                "NemotronHMoEMLP TRT-native forward requires "
                "prepare_for_export first (no stacked expert buffers)")
        hidden_flat = hidden_states.reshape(-1, self.hidden_size)
        num_tokens = hidden_flat.shape[0]
        router_logits = F.linear(hidden_flat, self.gate.weight).float()
        selected_experts, scores = trt_native_sigmoid_group_topk_routing(
            router_logits, self._e_score_correction_bias_fp32,
            self.num_experts_per_tok, self.gate.n_group, self.gate.topk_group,
            bool(self.gate.norm_topk_prob),
            float(self.gate.routed_scaling_factor))
        selected_experts = selected_experts.reshape(1, num_tokens,
                                                    self.num_experts_per_tok)
        scores = scores.reshape(1, num_tokens, self.num_experts_per_tok)

        routed_hidden_states = self.fc1_latent_proj(hidden_flat)
        hidden3 = routed_hidden_states.reshape(1, num_tokens,
                                               self.routed_hidden_size)
        if self._is_fp16:
            routed = trt_moe_fp16_nongated(
                hidden3,
                selected_experts,
                scores,
                self.fc_up_weights,
                self.fc_down_weights,
                TRT_MOE_ACT_RELU2,
                TRT_MOE_QUANT_NONE,
            )
        elif self._use_native_moe_w4a4:
            # W4A4 (quantization_mode 2): NVFP4 dynamic Q/DQ of the FC1
            # input, NVFP4 weight DQ with per-expert global scales
            # ([E, out, in] -> [E, in, out] for TRT_MoE's layout), per-expert
            # fp32 FC2 dblq at input 9.
            hidden3 = nvfp4_act_qdq_moe(hidden3, self.native_act_dblq_scale)
            up_w = nvfp4_dequantize_moe_with_global_scale(
                self.nvfp4_up_weight, self.nvfp4_up_block_scale_fp8,
                self.nvfp4_up_global_scale, _NVFP4_BLOCK_SIZE).transpose(1, 2)
            down_w = nvfp4_dequantize_moe_with_global_scale(
                self.nvfp4_down_weight, self.nvfp4_down_block_scale_fp8,
                self.nvfp4_down_global_scale,
                _NVFP4_BLOCK_SIZE).transpose(1, 2)
            routed = trt_moe_w4a4_nongated(
                hidden3,
                selected_experts,
                scores,
                up_w,
                down_w,
                self.native_down_act_scale,
                TRT_MOE_ACT_RELU2,
                TRT_MOE_QUANT_DYNAMIC_DBLQ,
                TRT_DATATYPE_FP4,
                TRT_DATATYPE_FP8,  # dyn_q_output_scale_dtype: FP8 block scales
            )
        else:
            # Static FP8 Q/DQ of the FC1 input with the calibrated scale, NVFP4
            # weight DQ ([E, out, in] -> [E, in, out] for TRT_MoE's layout):
            # up [E, I, H] -> [E, H, I], down [E, H, I] -> [E, I, H].
            hidden3 = fp8_dequantize(
                fp8_quantize(hidden3, self.native_act_scale),
                self.native_act_scale)
            up_w = nvfp4_dequantize_moe(self.nvfp4_up_weight,
                                        self.nvfp4_up_block_scale,
                                        _NVFP4_BLOCK_SIZE).transpose(1, 2)
            down_w = nvfp4_dequantize_moe(self.nvfp4_down_weight,
                                          self.nvfp4_down_block_scale,
                                          _NVFP4_BLOCK_SIZE).transpose(1, 2)
            routed = trt_moe_w4a8_nongated(
                hidden3,
                selected_experts,
                scores,
                up_w,
                down_w,
                self.native_down_act_scale,
                TRT_MOE_ACT_RELU2,
                TRT_MOE_QUANT_STATIC,
                TRT_DATATYPE_FP8,
            )
        routed = routed.reshape(routed_hidden_states.shape)
        moe_out = self.fc2_latent_proj(routed).reshape(hidden_states.shape)
        return moe_out + self._expert_forward(self.shared_experts,
                                              hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_native_moe:
            return self._native_moe_forward(hidden_states)

        # Router logits are computed from the original hidden states. When
        # moe_latent_size is set, only the routed expert payload is projected
        # down into latent space.
        router_logits = F.linear(hidden_states.view(-1, self.hidden_size),
                                 self.gate.weight).float()
        routed_hidden_states = self.fc1_latent_proj(hidden_states)

        if self._is_fp16:
            # Dense FP16 grouped GEMM. Same sigmoid-group-topk routing as the
            # NVFP4 paths (hence the ``_sigmoid`` variant), ReLU2 ungated FC1,
            # and no hidden-size padding: only the intermediate axis is padded.
            moe_out = fp16_moe_plugin_sigmoid(
                router_logits,
                routed_hidden_states,
                self.fc1_weights,
                self.fc2_weights,
                self._e_score_correction_bias_fp32,
                self.n_routed_experts,
                self.num_experts_per_tok,
                self.routed_hidden_size,
                self._padded_moe_intermediate_size,
                self.activation_type,
                self.gate.n_group,
                self.gate.topk_group,
                int(bool(self.gate.norm_topk_prob)),
                float(self.gate.routed_scaling_factor),
                self.max_routed_rows,
            )
            moe_out = self.fc2_latent_proj(moe_out)
            return moe_out + self._expert_forward(self.shared_experts,
                                                  hidden_states)

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
        # differs between SM100/101/103/110 ``Nvfp4MoePlugin`` and SM12x
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

        qkv = self.project_qkv(hidden_states)

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
            "skip_softmax_prefill_input": 0,
        }
        if enable_tree:
            kwargs["attention_mask"] = attention_mask
            kwargs["attention_pos_id"] = attention_pos_id
        # Always pass qkv_scales so torch.export includes a valid FLOATS
        # value in the FX graph for the unified ONNX translation.
        kwargs["qkv_scales"] = getattr(self, "_qkv_scales_float",
                                       [1.0, 1.0, 1.0])
        attn_output, present_key_value = attention_plugin(
            qkv, past_key_value, context_lengths, rope_rotary_cos_sin,
            kvcache_start_index, kv_page_table, **kwargs)

        attn_output = attn_output.reshape(batch_size, seq_len,
                                          self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_key_value

    def project_qkv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return Q, K and V packed along the last axis."""
        return torch.cat([
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states)
        ],
                         dim=-1)

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
        spec_verify_layout: "torch.Tensor | None" = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv = self.project_qkv(hidden_states)
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
            "skip_softmax_prefill_input": 0,
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

    attention_class = NemotronHAttentionMixer

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
            self.mixer = self._make_moe(config, module_prefix)
        elif layer_type == LAYER_ATTN:
            self.mixer = self.attention_class(config, layer_idx, module_prefix)
        else:
            raise ValueError(f"Unknown layer type: {layer_type!r}")

    def _make_moe(self, config: ModelConfig, module_prefix: str) -> nn.Module:
        return NemotronHMoEMLP(config, module_prefix)

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
        # Recurrent verification emits conv snapshots and replay data.
        collect_intermediate_states: bool = False,
        execution_phase_marker: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_pos_id: Optional[torch.Tensor] = None,
        tree_parent_ids: Optional[torch.Tensor] = None,
        tree_depths: Optional[torch.Tensor] = None,
        spec_verify_layout: "torch.Tensor | None" = None,
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
                     spec_verify_layout=spec_verify_layout,
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
                spec_verify_layout=kwargs.get("spec_verify_layout"),
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
        self.layers = nn.ModuleList(self._make_layers(config))
        self.norm_f = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layer_types: List[str] = config.layer_types

    def _make_layers(self, config: ModelConfig) -> List[nn.Module]:
        return [
            NemotronHDecoderLayer(config, layer_idx=i, layer_type=lt)
            for i, lt in enumerate(config.layer_types)
        ]

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
        spec_verify_layout: "torch.Tensor | None" = None,
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
                         spec_verify_layout=spec_verify_layout,
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
        spec_verify_layout: "torch.Tensor | None" = None,
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
            spec_verify_layout=spec_verify_layout,
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

    # The TRT-native MoE arm adds the fp32 routing bias to the fp32 sigmoid
    # scores in the graph. The export's generic dtype-fix pass downgrades fp32
    # initializers to FP16 unless told otherwise; this keeps the ones feeding
    # an element-wise op whose other input is FP32 (the exporter can fold the
    # bias into an anonymous initializer, so a name pattern would not catch
    # it). Without it the Add sees Float and Half, which TensorRT rejects. The
    # plugin arms feed the same buffer to a plugin input the pass keeps FP32.
    match_fp32_elementwise_initializers = True
    backbone_class = NemotronHBackbone

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = self.backbone_class(config)
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
        for module in self.modules():
            if isinstance(module, Q8_0Linear):
                module.prepare_for_export()
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
            args += (torch.empty(2, dtype=torch.int32, device=device), )
            input_names += ["spec_verify_layout"]
            all_shapes.append(
                {0: torch.export.Dim("spec_verify_layout_len", min=0, max=2)})
        wrapped = _make_flat_wrapper_mamba_ragged(self, Na, Nm, spec)
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
        spec_verify_layout: "torch.Tensor | None" = None,
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
            spec_verify_layout=spec_verify_layout,
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
        spec_verify_layout: "torch.Tensor | None" = None,
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
                 spec_verify_layout=spec_verify_layout,
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
