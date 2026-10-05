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
"""Checkpoint-backed Gated DeltaNet module."""

import numpy as np
import tensorrt as trt

from . import functional as F
from .linear import Linear
from .module import Module
from .ragged import RaggedDecoderInputs


class GatedDeltaNet(Module):
    """Linear-attention block composed from projections and recurrent ops."""

    replicated_controls = True

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        gdn = ctx.cfg.gdn_cfg
        self.tp_qkv_segments = (
            gdn.key_dim * ctx.cfg.tp_size,
            gdn.key_dim * ctx.cfg.tp_size,
            gdn.value_dim * ctx.cfg.tp_size,
        )
        self._init_input_projections()
        self.out_proj = Linear(ctx, self.key("out_proj"))

    def _init_input_projections(self) -> None:
        """Create the split projection layout used by Qwen3.5 checkpoints."""
        self.in_proj_qkv = Linear(self.ctx,
                                  self.key("in_proj_qkv"),
                                  tp_output_segments=self.tp_qkv_segments)
        self.in_proj_z = Linear(self.ctx, self.key("in_proj_z"))
        self.in_proj_b = Linear(self.ctx,
                                self.key("in_proj_b"),
                                tensor_parallel=False)
        self.in_proj_a = Linear(self.ctx,
                                self.key("in_proj_a"),
                                tensor_parallel=False)

    def _project_inputs(self, hidden_states):
        return (self.in_proj_qkv(hidden_states), self.in_proj_z(hidden_states),
                self.in_proj_b(hidden_states), self.in_proj_a(hidden_states))

    def _constant_weight(self, suffix: str, shape, dtype, tp_segments=()):
        key = self.key(suffix)
        value = (self.weights.f32(key)
                 if dtype == np.float32 else self.weights.f16(key))
        if tp_segments:
            value = self.weights.shard_segments(value, tp_segments,
                                                self.cfg.tp_size,
                                                self.cfg.tp_rank)
        if value.shape != tuple(shape):
            raise ValueError(
                f"{key} must have shape {tuple(shape)}, got {value.shape}")
        return F.constant(value.astype(dtype, copy=False), suffix)

    def forward(self,
                hidden_states,
                conv_state,
                recurrent_state,
                ragged: RaggedDecoderInputs,
                tree_parent_ids=None,
                tree_depths=None,
                collect_intermediate=False):
        cfg = self.cfg
        gdn = cfg.gdn_cfg
        mixed, gate, beta, alpha = self._project_inputs(hidden_states)
        if cfg.tp_size > 1 and self.replicated_controls:
            # Keep the small alpha and beta projections replicated because a
            # rank-local shard is narrower than the supported NVFP4 GEMM tile.
            start = cfg.tp_rank * gdn.num_value_heads
            stop = start + gdn.num_value_heads
            beta = beta[..., start:stop]
            alpha = alpha[..., start:stop]
        conv_weight = self._constant_weight("conv1d.weight",
                                            (gdn.conv_dim, 1, gdn.conv_kernel),
                                            np.float16, self.tp_qkv_segments)
        conv_bias_data = self.weights.opt_f16(self.key("conv1d.bias"))
        if conv_bias_data is None:
            conv_bias_data = np.zeros(gdn.conv_dim, dtype=np.float16)
        elif cfg.tp_size > 1:
            conv_bias_data = self.weights.shard_segments(
                conv_bias_data, self.tp_qkv_segments, cfg.tp_size, cfg.tp_rank)
        if conv_bias_data.shape != (gdn.conv_dim, ):
            raise ValueError(f"{self.key('conv1d.bias')} must have shape "
                             f"{(gdn.conv_dim,)}, got {conv_bias_data.shape}")
        conv_bias = F.constant(conv_bias_data, "conv_bias")
        mixed, conv_state_out, intermediate_conv = F.causal_conv1d(
            mixed,
            conv_weight,
            conv_bias,
            conv_state,
            ragged,
            gdn.conv_dim,
            gdn.conv_kernel - 1,
            tree_parent_ids=tree_parent_ids,
            tree_depths=tree_depths,
            use_intermediate=collect_intermediate)
        mixed = mixed.activation(cfg.hidden_act)
        query = mixed[..., :gdn.key_dim].reshape(
            (0, gdn.num_key_heads, gdn.key_head_dim))
        key = mixed[..., gdn.key_dim:gdn.key_dim * 2].reshape(
            (0, gdn.num_key_heads, gdn.key_head_dim))
        value = mixed[...,
                      gdn.key_dim * 2:gdn.key_dim * 2 + gdn.value_dim].reshape(
                          (0, gdn.num_value_heads, gdn.value_head_dim))
        full_value_heads = (gdn.num_value_heads * cfg.tp_size, )
        a_log = self._constant_weight("A_log", (gdn.num_value_heads, ),
                                      np.float32, full_value_heads)
        dt_bias = self._constant_weight("dt_bias", (gdn.num_value_heads, ),
                                        np.float16, full_value_heads)
        output, recurrent_state_out, intermediate_recurrent = F.gated_delta_net(
            query,
            key,
            value,
            alpha,
            beta,
            a_log,
            dt_bias,
            recurrent_state,
            ragged,
            gdn.key_head_dim,
            gdn.value_head_dim,
            tree_parent_ids=tree_parent_ids,
            tree_depths=tree_depths,
            use_intermediate=collect_intermediate,
            use_diffusion_state=cfg.engine_role == "dllm")
        gate = gate.reshape((0, gdn.num_value_heads, gdn.value_head_dim))
        norm_weight = self.weights.f16(self.key("norm.weight"))
        if norm_weight.shape != (gdn.value_head_dim, ):
            raise ValueError(
                f"{self.key('norm.weight')} must have shape "
                f"{(gdn.value_head_dim,)}, got {norm_weight.shape}")
        output = F.rms_norm(output, norm_weight, cfg.rms_norm_eps, 3)
        output = (output.cast(trt.float32) *
                  gate.cast(trt.float32).silu()).cast(trt.float16)
        output = output.reshape((0, gdn.value_dim))
        return (self.out_proj(output), conv_state_out, recurrent_state_out,
                intermediate_conv, intermediate_recurrent)
