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
"""TP=1 action expert consuming an immutable LLM pool and private action pages."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from ...config import ActionConfig
from .. import ops
from .modeling_alpamayo_action import (ActionAttention, ActionDecoderLayer,
                                       AlpamayoAction, RMSNorm)


class PagedActionAttention(ActionAttention):
    """Paged non-causal attention for the action expert."""

    def __init__(self, cfg: ActionConfig, layer_idx: int = 0) -> None:
        super().__init__(cfg)
        self.fp8_kv_prefix = cfg.kv_cache_scales is not None
        self.qkv_scales = ([1.0, *cfg.kv_cache_scales[layer_idx]]
                           if self.fp8_kv_prefix else [1.0, 1.0, 1.0])

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        action_key_value: torch.Tensor,
        action_page_table: torch.Tensor,
        query_lengths: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        past_lengths: torch.Tensor,
        kv_page_table: torch.Tensor,
        context_mask_selector: torch.Tensor,
        query_start_offsets: torch.Tensor,
        attention_sequence_lengths: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from ..ops import attention_plugin

        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        qkv = torch.cat(
            [
                q.reshape(bsz * q_len, self.num_heads * self.head_dim),
                k.reshape(bsz * q_len, self.num_kv_heads * self.head_dim),
                v.reshape(bsz * q_len, self.num_kv_heads * self.head_dim),
            ],
            dim=-1,
        )

        attn_output, present_key_value = attention_plugin(
            qkv,
            action_key_value,
            query_lengths,
            rope_rotary_cos_sin,
            past_lengths,
            action_page_table,
            prefix_key_value=past_key_value,
            prefix_page_table=kv_page_table,
            external_kv_prefix=1,
            fp8_kv_prefix=int(self.fp8_kv_prefix),
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            sliding_window_size=-1,
            enable_tree_attention=False,
            enable_fp8_kv_cache=False,
            attention_scale=self.attention_scale,
            enable_context_mask_selector=True,
            enable_vision_block_attention=False,
            skip_softmax_prefill_input=0,
            context_mask_selector=context_mask_selector,
            qkv_scales=self.qkv_scales,
            query_start_offsets=query_start_offsets,
            attention_sequence_lengths=attention_sequence_lengths,
            execution_phase_marker=execution_phase_marker,
            context_sequence_count_carrier=context_sequence_count_carrier,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, present_key_value


class PagedActionExpert(nn.Module):
    """Stack of ActionDecoderLayers + final RMSNorm."""

    def __init__(self, cfg: ActionConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ActionDecoderLayer(cfg, PagedActionAttention(cfg, i))
            for i in range(cfg.num_hidden_layers)
        ])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        query_lengths: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        past_lengths: torch.Tensor,
        kv_page_table: torch.Tensor,
        context_mask_selector: torch.Tensor,
        query_start_offsets: torch.Tensor,
        attention_sequence_lengths: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        action_key_values: Tuple[torch.Tensor, ...],
        action_page_table: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden_states = inputs_embeds
        present_key_values: List[torch.Tensor] = []
        for layer, past_key_value, action_key_value in zip(
                self.layers, past_key_values, action_key_values):
            hidden_states, present_key_value = layer(
                hidden_states,
                past_key_value,
                action_key_value,
                action_page_table,
                query_lengths,
                rope_rotary_cos_sin,
                past_lengths,
                kv_page_table,
                context_mask_selector,
                query_start_offsets,
                attention_sequence_lengths,
                execution_phase_marker,
                context_sequence_count_carrier,
            )
            present_key_values.append(present_key_value)
        hidden_states = self.norm(hidden_states)
        return hidden_states, present_key_values


class PagedAlpamayoAction(AlpamayoAction):
    """Wraps one flow-matching denoising step for ONNX export."""

    def __init__(self, cfg: ActionConfig) -> None:
        super().__init__(cfg, expert=PagedActionExpert(cfg))

    def forward(
        self,
        noise_trajectory: torch.Tensor,
        time_steps_t0: torch.Tensor,
        time_steps_t1: torch.Tensor,
        query_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        attention_sequence_lengths: torch.Tensor,
        query_start_offsets: torch.Tensor,
        kv_page_table: torch.Tensor,
        context_mask_selector: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        action_page_table: torch.Tensor,
        *past_key_values: torch.Tensor,
    ) -> tuple:
        bsz = noise_trajectory.shape[0]

        noise_trajectory = noise_trajectory.to(torch.float16)
        time_steps_t0 = time_steps_t0.to(torch.float16)
        time_steps_t1 = time_steps_t1.to(torch.float16)

        # Expand dt and t_start to (B, 1, 1) matching action_space_dims
        dt = (time_steps_t1 - time_steps_t0).view(-1, 1, 1).expand(bsz, 1, 1)
        t_start = time_steps_t0.view(-1, 1, 1).expand(bsz, 1, 1)

        embeds = self.action_in_proj(noise_trajectory, t_start)

        hidden, present_key_values = self.expert(
            embeds,
            query_lengths,
            rope_rotary_cos_sin.reshape(bsz * self.n_diffusion_tokens,
                                        self.cfg.head_dim),
            past_lengths,
            kv_page_table,
            context_mask_selector,
            query_start_offsets,
            attention_sequence_lengths,
            execution_phase_marker,
            context_sequence_count_carrier,
            tuple(past_key_values[:self.n_layers]),
            tuple(past_key_values[self.n_layers:]),
            action_page_table,
        )

        pred = self.action_out_proj(hidden[:, -self.n_diffusion_tokens:]).view(
            bsz, self.n_diffusion_tokens, 2)
        denoised = (noise_trajectory + dt * pred).to(torch.float32)

        return (denoised, *present_key_values)

    def get_onnx_export_args(
        self,
        max_kv_cache_capacity: int,
        device: str,
    ) -> Tuple[tuple, list, list, dict]:
        """Build dummy inputs, names, and dynamic shapes for ONNX export."""
        cfg = self.cfg
        n_layers = cfg.num_hidden_layers
        n_kv_heads = cfg.num_key_value_heads
        head_dim = cfg.head_dim
        n_diff = cfg.n_diffusion_tokens
        B = 1

        noise_trajectory = torch.randn(B,
                                       n_diff,
                                       2,
                                       device=device,
                                       dtype=torch.float32)
        t0 = torch.tensor([0.0], device=device, dtype=torch.float32)
        t1 = torch.tensor([0.1], device=device, dtype=torch.float32)
        query_lengths = torch.full((B, ),
                                   n_diff,
                                   device=device,
                                   dtype=torch.int32)
        past_lengths = torch.full((B, ),
                                  3027,
                                  device=device,
                                  dtype=torch.int32)
        attention_sequence_lengths = past_lengths + query_lengths
        query_start_offsets = torch.arange(0, (B + 1) * n_diff,
                                           n_diff,
                                           device=device,
                                           dtype=torch.int32)
        max_pages_per_seq = (max_kv_cache_capacity + ops.KV_PAGE_SIZE -
                             1) // ops.KV_PAGE_SIZE
        kv_page_table = torch.zeros(B,
                                    2,
                                    max_pages_per_seq,
                                    device=device,
                                    dtype=torch.int32)
        context_mask_selector = torch.zeros(B,
                                            device=device,
                                            dtype=torch.int32)
        execution_phase_marker = torch.zeros(6,
                                             device=device,
                                             dtype=torch.int32)
        context_sequence_count_carrier = torch.empty(0,
                                                     device=device,
                                                     dtype=torch.int32)
        rope_cos_sin = torch.randn(B,
                                   n_diff,
                                   head_dim,
                                   device=device,
                                   dtype=torch.float32)
        num_pages = B * (
            (max_kv_cache_capacity + ops.KV_PAGE_SIZE - 1) // ops.KV_PAGE_SIZE)
        cache_shape = (2, num_pages, ops.KV_PAGE_SIZE, n_kv_heads, head_dim)
        past_key_values = [
            torch.zeros(cache_shape,
                        device=device,
                        dtype=(torch.float8_e4m3fn if self.cfg.kv_cache_scales
                               is not None else torch.float16))
            for _ in range(n_layers)
        ]

        action_pages_per_seq = (n_diff + ops.KV_PAGE_SIZE -
                                1) // ops.KV_PAGE_SIZE
        action_page_table = torch.zeros(B,
                                        2,
                                        action_pages_per_seq,
                                        device=device,
                                        dtype=torch.int32)
        action_key_values = [
            torch.zeros(2,
                        B * action_pages_per_seq,
                        ops.KV_PAGE_SIZE,
                        n_kv_heads,
                        head_dim,
                        device=device,
                        dtype=torch.float16) for _ in range(n_layers)
        ]

        args = tuple([
            noise_trajectory,
            t0,
            t1,
            query_lengths,
            past_lengths,
            attention_sequence_lengths,
            query_start_offsets,
            kv_page_table,
            context_mask_selector,
            execution_phase_marker,
            context_sequence_count_carrier,
            rope_cos_sin,
            action_page_table,
        ] + past_key_values + action_key_values)

        input_names = ([
            "noise_trajectory",
            "time_steps_t0",
            "time_steps_t1",
            "query_lengths",
            "past_lengths",
            "attention_sequence_lengths",
            "query_start_offsets",
            "kv_page_table",
            "context_mask_selector",
            "execution_phase_marker",
            "context_sequence_count_carrier",
            "rope_rotary_cos_sin",
            "action_page_table",
        ] + [f"past_key_values_{i}" for i in range(n_layers)] +
                       [f"action_key_values_{i}" for i in range(n_layers)])

        output_names = (["denoised_trajectory"] +
                        [f"present_key_values_{i}" for i in range(n_layers)])

        batch = torch.export.Dim("batch_size")
        pages = torch.export.Dim("num_pages", min=1, max=1048576)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        phase_extent = torch.export.Dim("execution_phase_extent", min=1, max=8)
        dynamic_shapes: list = [
            {
                0: batch
            },
            None,  # time_steps_t0
            None,  # time_steps_t1
            {
                0: batch
            },
            {
                0: batch
            },
            {
                0: batch
            },
            {
                0: batch + 1
            },
            {
                0: batch,
                2: max_pages
            },
            {
                0: batch
            },
            {
                0: phase_extent
            },
            {},  # context_sequence_count_carrier
            {
                0: batch
            },
        ]
        dynamic_shapes.append({0: batch})
        action_pages = torch.export.Dim("action_pool_pages",
                                        min=1,
                                        max=1048576)
        dynamic_shapes.append(
            tuple([{
                1: pages
            } for _ in range(n_layers)] + [{
                1: action_pages
            } for _ in range(n_layers)]))

        return args, input_names, output_names, tuple(dynamic_shapes)
