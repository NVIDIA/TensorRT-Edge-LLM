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
"""LFM3 text decoder: gated short convolutions interleaved with attention.

Per layer type (``config.layer_types``):

* ``conv``: gated short convolution. ``in_proj`` yields ``B, C, x``; ``B * x``
  is convolved causally per channel (width ``conv_L_cache``) and gated by
  ``C``. The conv window is carried across steps as ``conv_state_{i}``.
* ``sliding_attention`` / ``full_attention``: GQA with per-head QK RMSNorm,
  a per-head sigmoid output gate, and the layer type's own RoPE (bound as
  ``rope_rotary_cos_sin_sliding`` / ``rope_rotary_cos_sin_full``).

Every layer is followed by a SwiGLU feed-forward block. With
``use_sandwich_norm`` the operator and feed-forward outputs are RMS-normalised
again before their residual add. The final norm is ``model.embedding_norm``.

Checkpoint key correspondence
-----------------------------
``model.layers.N.conv.{in_proj,out_proj,conv}.*``
``model.layers.N.self_attn.{q,k,v,o,gate}_proj.weight``, ``q_norm``, ``k_norm``
``model.layers.N.feed_forward.{w1,w2,w3}.weight``
``model.layers.N.{operator,ffn,post_operator,post_ffn}_norm.weight``
"""

import itertools
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...checkpoint.checkpoint_utils import rotary_dim_for_runtime
from ...config import LAYER_CONV, ModelConfig
from ..default.modeling_default import Attention, CausalLM, OnnxSpec, RMSNorm
from ..linear import TPMode, make_linear
from ..ops import KV_PAGE_SIZE, causal_conv1d

__all__ = ["Lfm3ForCausalLM"]

_NUM_SEQUENCES = 2
_QUERY_LENGTH = 2
_STATE_POOL_ROWS = 4


class Lfm3ConvWeights(nn.Module):
    """Depthwise conv weight ``[conv_dim, 1, kernel]`` and bias as plain buffers.

    LFM3 checkpoints carry no conv bias; the zero bias satisfies the plugin's
    required bias input.
    """

    def __init__(self, conv_dim: int, conv_kernel: int) -> None:
        super().__init__()
        self.register_buffer(
            "weight", torch.zeros(conv_dim,
                                  1,
                                  conv_kernel,
                                  dtype=torch.float16))
        self.register_buffer("bias", torch.zeros(conv_dim,
                                                 dtype=torch.float16))


class Lfm3ShortConv(nn.Module):
    """Gated short convolution (``model.layers.N.conv``)."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        sc = config.short_conv_cfg
        if sc.conv_bias:
            raise NotImplementedError(
                "LFM short convolutions with conv_bias=True are not supported."
            )
        prefix = f"layers.{layer_idx}.conv"
        self.conv_dim = sc.conv_dim
        self.conv_kernel = sc.conv_kernel
        self.in_proj = make_linear(config,
                                   config.hidden_size,
                                   3 * sc.conv_dim,
                                   module_name=f"{prefix}.in_proj")
        self.conv = Lfm3ConvWeights(sc.conv_dim, sc.conv_kernel)
        self.out_proj = make_linear(config,
                                    sc.conv_dim,
                                    config.hidden_size,
                                    module_name=f"{prefix}.out_proj")

    def forward_ragged(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        state_indices: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_gate, c_gate, x = self.in_proj(hidden_states).chunk(3, dim=-1)
        gated = (b_gate * x).reshape(-1, self.conv_dim)
        conv_out, conv_state_out, _ = causal_conv1d(
            gated,
            self.conv.weight,
            self.conv.bias,
            conv_state,
            query_lengths,
            stride=1,
            padding=self.conv_kernel - 1,
            dilation=1,
            groups=self.conv_dim,
            query_start_offsets=query_start_offsets,
            state_indices=state_indices,
            execution_phase_marker=execution_phase_marker,
            context_sequence_count_carrier=context_sequence_count_carrier)
        conv_out = conv_out.reshape(c_gate.shape)
        return self.out_proj(c_gate * conv_out), conv_state_out


class Lfm3Attention(Attention):
    """GQA attention with QK RMSNorm and a per-head sigmoid output gate."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx=layer_idx)
        self.is_sliding = (
            config.layer_types[layer_idx] == "sliding_attention")
        self.sliding_window_size = (config.sliding_window_size
                                    if self.is_sliding else -1)
        self.gate_proj = make_linear(
            config,
            config.hidden_size,
            config.num_attention_heads,
            module_name=f"layers.{layer_idx}.self_attn.gate_proj",
            tp_mode=TPMode.COL)

    def forward_ragged(self, hidden_states: torch.Tensor, **kwargs):
        attn_output, present_key_value = self.attend_ragged(
            hidden_states, **kwargs)
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        attn_output = attn_output.unflatten(-1,
                                            (self.num_heads, self.head_dim))
        attn_output = (attn_output * gate.unsqueeze(-1)).flatten(-2)
        return self.o_proj(attn_output), present_key_value


class Lfm3MLP(nn.Module):
    """SwiGLU feed-forward with LFM key names: ``w2(silu(w1(x)) * w3(x))``."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        prefix = f"layers.{layer_idx}.feed_forward"
        self.w1 = make_linear(config,
                              config.hidden_size,
                              config.intermediate_size,
                              module_name=f"{prefix}.w1",
                              tp_mode=TPMode.COL)
        self.w3 = make_linear(config,
                              config.hidden_size,
                              config.intermediate_size,
                              module_name=f"{prefix}.w3",
                              tp_mode=TPMode.COL)
        self.w2 = make_linear(config,
                              config.intermediate_size,
                              config.hidden_size,
                              module_name=f"{prefix}.w2",
                              tp_mode=TPMode.ROW)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class Lfm3DecoderLayer(nn.Module):

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.is_conv = config.layer_types[layer_idx] == LAYER_CONV
        if self.is_conv:
            self.conv = Lfm3ShortConv(config, layer_idx)
        else:
            self.self_attn = Lfm3Attention(config, layer_idx)
        self.feed_forward = Lfm3MLP(config, layer_idx)
        self.operator_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.use_sandwich_norm = config.use_sandwich_norm
        if self.use_sandwich_norm:
            self.post_operator_norm = RMSNorm(config.hidden_size,
                                              config.rms_norm_eps)
            self.post_ffn_norm = RMSNorm(config.hidden_size,
                                         config.rms_norm_eps)

    @property
    def is_sliding(self) -> bool:
        return not self.is_conv and self.self_attn.is_sliding

    def forward_ragged(
            self, hidden_states: torch.Tensor,
            operator_kwargs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.operator_norm(hidden_states)
        if self.is_conv:
            operator_out, present = self.conv.forward_ragged(
                normed, **operator_kwargs)
        else:
            operator_out, present = self.self_attn.forward_ragged(
                normed, **operator_kwargs)
        if self.use_sandwich_norm:
            operator_out = self.post_operator_norm(operator_out)
        hidden_states = hidden_states + operator_out
        ffn_out = self.feed_forward(self.ffn_norm(hidden_states))
        if self.use_sandwich_norm:
            ffn_out = self.post_ffn_norm(ffn_out)
        return hidden_states + ffn_out, present


class Lfm3Model(nn.Module):
    """Decoder stack stored as ``model`` so keys carry the ``model.`` prefix."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Lfm3DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.embedding_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward_ragged(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin_sliding: torch.Tensor,
        rope_rotary_cos_sin_full: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        attention_sequence_lengths: torch.Tensor,
        state_indices: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        kv_page_table: torch.Tensor,
        conv_states: Tuple[torch.Tensor, ...],
        skip_softmax_scale: "torch.Tensor | None" = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor,
                                                             ...]]:
        hidden_states = inputs_embeds.unsqueeze(1)
        present_key_values: List[torch.Tensor] = []
        present_conv_states: List[torch.Tensor] = []
        for layer in self.layers:
            if layer.is_conv:
                operator_kwargs = {
                    "conv_state":
                    conv_states[len(present_conv_states)],
                    "query_start_offsets":
                    query_start_offsets,
                    "query_lengths":
                    query_lengths,
                    "state_indices":
                    state_indices,
                    "execution_phase_marker":
                    execution_phase_marker,
                    "context_sequence_count_carrier":
                    context_sequence_count_carrier,
                }
            else:
                operator_kwargs = {
                    "past_key_value":
                    past_key_values[len(present_key_values)],
                    "rope_rotary_cos_sin":
                    (rope_rotary_cos_sin_sliding
                     if layer.is_sliding else rope_rotary_cos_sin_full),
                    "query_start_offsets":
                    query_start_offsets,
                    "query_lengths":
                    query_lengths,
                    "past_lengths":
                    past_lengths,
                    "attention_sequence_lengths":
                    attention_sequence_lengths,
                    "execution_phase_marker":
                    execution_phase_marker,
                    "context_sequence_count_carrier":
                    context_sequence_count_carrier,
                    "kv_page_table":
                    kv_page_table,
                    "skip_softmax_scale":
                    skip_softmax_scale,
                }
            hidden_states, present = layer.forward_ragged(
                hidden_states, operator_kwargs)
            if layer.is_conv:
                present_conv_states.append(present)
            else:
                present_key_values.append(present)
        return (self.embedding_norm(hidden_states), tuple(present_key_values),
                tuple(present_conv_states))


def _make_flat_wrapper(model: nn.Module, num_attn: int,
                       num_conv: int) -> nn.Module:
    """Flat positional wrapper whose argument names become ONNX input names."""
    param_names: List[str] = (
        ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(num_attn)] +
        [
            "rope_rotary_cos_sin_sliding", "rope_rotary_cos_sin_full",
            "positions", "query_start_offsets", "query_lengths",
            "past_lengths", "attention_sequence_lengths", "state_indices",
            "execution_phase_marker", "context_sequence_count_carrier",
            "kv_page_table", "logits_indices"
        ] + [f"conv_state_{i}"
             for i in range(num_conv)] + ["skip_softmax_scale"])
    kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(num_attn))) if num_attn else "()"
    conv_tuple = "({},)".format(", ".join(
        f"conv_state_{i}" for i in range(num_conv))) if num_conv else "()"
    body = (
        "    logits, present_key_values, present_conv_states = "
        "self._model.forward_ragged(\n"
        f"        inputs_embeds, {kv_tuple}, rope_rotary_cos_sin_sliding, "
        "rope_rotary_cos_sin_full, query_start_offsets, query_lengths, "
        "past_lengths, attention_sequence_lengths, state_indices, "
        "execution_phase_marker, context_sequence_count_carrier, "
        f"kv_page_table, logits_indices, {conv_tuple}, skip_softmax_scale)\n"
        "    return (logits,) + tuple(present_key_values) + "
        "tuple(present_conv_states)\n")
    src = "def _forward(self, {}):\n{}".format(", ".join(param_names), body)
    globs: dict = {}
    exec(src, globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


class Lfm3ForCausalLM(CausalLM):
    """LFM3 causal LM exported with the hybrid token-major ABI."""

    transformer_cls = Lfm3Model
    # Short-conv layers carry no recurrent state; the dtype is still recorded
    # because the runtime config schema requires it for hybrid engines.
    RECURRENT_STATE_DTYPE = torch.float32
    CONV_STATE_DTYPE = torch.float16

    def __init__(self, config: ModelConfig) -> None:
        if config.short_conv_cfg is None and LAYER_CONV in config.layer_types:
            raise ValueError("LFM3 conv layers require short_conv_cfg.")
        if not config.use_dual_rope:
            raise ValueError(
                "LFM3 requires sliding_attention and full_attention RoPE "
                "parameters.")
        super().__init__(config)

    def forward_ragged(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin_sliding: torch.Tensor,
        rope_rotary_cos_sin_full: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        attention_sequence_lengths: torch.Tensor,
        state_indices: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        kv_page_table: torch.Tensor,
        logits_indices: torch.Tensor,
        conv_states: Tuple[torch.Tensor, ...],
        skip_softmax_scale: "torch.Tensor | None" = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor,
                                                             ...]]:
        hidden_states, present_key_values, present_conv_states = (
            self.model.forward_ragged(
                inputs_embeds, past_key_values, rope_rotary_cos_sin_sliding,
                rope_rotary_cos_sin_full, query_start_offsets, query_lengths,
                past_lengths, attention_sequence_lengths, state_indices,
                execution_phase_marker, context_sequence_count_carrier,
                kv_page_table, conv_states, skip_softmax_scale))
        selected = torch.index_select(
            hidden_states.reshape(-1, hidden_states.shape[-1]), 0,
            logits_indices)
        logits = self.lm_head(selected).to(torch.float32)
        return logits, present_key_values, present_conv_states

    def onnx_export_spec(self) -> OnnxSpec:
        config = self.config
        sc = config.short_conv_cfg
        num_attn = config.num_attn_layers
        num_conv = config.num_conv_layers
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        physical_tokens = _NUM_SEQUENCES * _QUERY_LENGTH
        kv_dtype = (torch.float8_e4m3fn
                    if config.quant.kv_cache_quant == "fp8" else torch.float16)

        def int32(*values: int) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device=device)

        inputs_embeds = torch.zeros(physical_tokens,
                                    config.hidden_size,
                                    dtype=torch.float16,
                                    device=device)
        past_key_values = [
            torch.zeros(2,
                        2,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        config.head_dim,
                        dtype=kv_dtype,
                        device=device) for _ in range(num_attn)
        ]
        rope_sliding = torch.zeros(physical_tokens,
                                   rotary_dim_for_runtime(
                                       config.sliding_rope_config,
                                       config.head_dim, 1.0),
                                   dtype=torch.float32,
                                   device=device)
        rope_full = torch.zeros(physical_tokens,
                                rotary_dim_for_runtime(config.full_rope_config,
                                                       config.head_dim, 1.0),
                                dtype=torch.float32,
                                device=device)
        positions = torch.arange(_QUERY_LENGTH,
                                 dtype=torch.int32,
                                 device=device).repeat(_NUM_SEQUENCES)
        query_start_offsets = torch.arange(0,
                                           physical_tokens + 1,
                                           _QUERY_LENGTH,
                                           dtype=torch.int32,
                                           device=device)
        query_lengths = int32(*([_QUERY_LENGTH] * _NUM_SEQUENCES))
        past_lengths = int32(*([0] * _NUM_SEQUENCES))
        attention_sequence_lengths = query_lengths.clone()
        state_indices = int32(2, 0)
        execution_phase_marker = int32(0, 0)
        context_sequence_count_carrier = int32(*([0] * _NUM_SEQUENCES))
        kv_page_table = torch.zeros(_NUM_SEQUENCES,
                                    2,
                                    2,
                                    dtype=torch.int32,
                                    device=device)
        logits_indices = query_start_offsets[1:].to(torch.int64) - 1
        conv_states = [
            torch.zeros(_STATE_POOL_ROWS,
                        sc.conv_dim,
                        sc.conv_kernel,
                        dtype=self.CONV_STATE_DTYPE,
                        device=device) for _ in range(num_conv)
        ]
        skip_softmax_scale = torch.zeros(2, dtype=torch.int8, device=device)

        tokens = torch.export.Dim("physical_tokens", min=1, max=8_388_608)
        sequences = torch.export.Dim("num_sequences", min=1, max=256)
        context_sequences = torch.export.Dim("num_context_sequences",
                                             min=0,
                                             max=256)
        state_pool = torch.export.Dim("state_pool_rows", min=1, max=256)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        num_pages = torch.export.Dim("num_pages", min=1, max=1048576)
        phase_extent = torch.export.Dim("execution_phase_extent", min=1, max=8)
        skip_dim = torch.export.Dim("skip_softmax_scale_len",
                                    min=0,
                                    max=1048576)

        # (ONNX input name, example tensor, dynamic shape), in wrapper order.
        inputs = [("inputs_embeds", inputs_embeds, {0: tokens})]
        inputs += [(f"past_key_values_{i}", kv, {
            1: num_pages
        }) for i, kv in enumerate(past_key_values)]
        inputs += [
            ("rope_rotary_cos_sin_sliding", rope_sliding, {
                0: tokens
            }),
            ("rope_rotary_cos_sin_full", rope_full, {
                0: tokens
            }),
            ("positions", positions, {
                0: tokens
            }),
            ("query_start_offsets", query_start_offsets, {
                0: sequences + 1
            }),
            ("query_lengths", query_lengths, {
                0: sequences
            }),
            ("past_lengths", past_lengths, {
                0: sequences
            }),
            ("attention_sequence_lengths", attention_sequence_lengths, {
                0: sequences
            }),
            ("state_indices", state_indices, {
                0: sequences
            }),
            ("execution_phase_marker", execution_phase_marker, {
                0: phase_extent
            }),
            ("context_sequence_count_carrier", context_sequence_count_carrier,
             {
                 0: context_sequences
             }),
            ("kv_page_table", kv_page_table, {
                0: sequences,
                2: max_pages
            }),
            ("logits_indices", logits_indices, {
                0: sequences
            }),
        ]
        inputs += [(f"conv_state_{i}", state, {
            0: state_pool
        }) for i, state in enumerate(conv_states)]
        inputs.append(("skip_softmax_scale", skip_softmax_scale, {
            0: skip_dim
        }))
        output_names = (["logits"] +
                        [f"present_key_values_{i}" for i in range(num_attn)] +
                        [f"present_conv_state_{i}" for i in range(num_conv)])

        wrapped = _make_flat_wrapper(self, num_attn, num_conv)
        wrapped.eval()
        return OnnxSpec(wrapped=wrapped,
                        args=tuple(tensor for _, tensor, _ in inputs),
                        input_names=[name for name, _, _ in inputs],
                        output_names=output_names,
                        dynamic_shapes=[shape for _, _, shape in inputs])
