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
Qwen3-TTS CodePredictor — residual codec token predictor.

The CodePredictor is a small 5-layer Qwen3 decoder that generates residual
audio codes to complement the coarse code from the Talker.  It has 15 separate
lm_heads (one per residual codebook layer) and 15 codec embedding tables.

Key differences from a standard CausalLM:
- ``lm_head_weight`` is an ONNX **input** tensor (not a fixed weight) to enable
  dynamic lm_head selection at runtime via 15 CUDA Graphs.
- The forward pass returns both ``logits`` and ``hidden_states`` (for residual
  connection in the multi-token prediction loop).
- ``embed_tokens`` is a ModuleList of 15 codec embeddings (embedding lookup
  happens at runtime, not in the ONNX graph — the model takes ``inputs_embeds``).

Extra weight files extracted alongside the ONNX:
- ``codec_embeddings.safetensors`` — 15 embedding tables [codebookSize, hiddenSize]
- ``lm_heads.safetensors`` — 15 lm_head weights [codebookSize, hiddenSize]
- ``small_to_mtp_projection.safetensors`` — Linear [cpHiddenSize, talkerHiddenSize]

Checkpoint prefix: ``talker.code_predictor.*``
"""

import itertools
import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..default.modeling_default import (_BATCH_SIZE, _MAX_POS, _PAST_LEN,
                                        _SEQ_LEN, CausalLM, OnnxSpec)
from ..linear import TPMode, make_linear
from ..ops import KV_PAGE_SIZE

logger = logging.getLogger(__name__)

__all__ = ["CodePredictorCausalLM"]


class CodePredictorMLP(nn.Module):
    """SwiGLU MLP with an FP32 ``silu(gate) * up`` precision guard.

    CP's intermediate activations reach [-39.5, 72.4] where FP16 spacing
    is 0.0625; an FP16 multiply loses ~6 mantissa bits per layer and
    compounds across the 5-layer stack, shifting codec-EOS prediction
    enough to regress TTS WER by ~20% absolute.  Running ``silu*up`` in
    FP32 and keeping the intermediate FP32 through the down_proj matmul
    (via a Cast on the weight, constant-folded by TRT) preserves
    precision end-to-end.

    The FP8_CP recipe excludes ``down_proj`` from FP8 quantization so
    the down_proj Linear is always FP16Linear whether or not the rest of
    the CP is quantized — this same forward path is safe in both modes.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        prefix = f"layers.{layer_idx}.mlp"
        self.gate_proj = make_linear(config,
                                     config.hidden_size,
                                     config.intermediate_size,
                                     module_name=f"{prefix}.gate_proj",
                                     tp_mode=TPMode.COL)
        self.up_proj = make_linear(config,
                                   config.hidden_size,
                                   config.intermediate_size,
                                   module_name=f"{prefix}.up_proj",
                                   tp_mode=TPMode.COL)
        self.down_proj = make_linear(config,
                                     config.intermediate_size,
                                     config.hidden_size,
                                     module_name=f"{prefix}.down_proj",
                                     tp_mode=TPMode.ROW)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out_dtype = hidden_states.dtype
        gate_output = self.gate_proj(hidden_states).to(torch.float32)
        up_output = self.up_proj(hidden_states).to(torch.float32)
        intermediate = F.silu(gate_output) * up_output
        w32 = self.down_proj.weight.to(torch.float32)
        return F.linear(intermediate, w32).to(out_dtype)


# ---------------------------------------------------------------------------
# CodePredictor flat wrapper for ONNX export
# ---------------------------------------------------------------------------


def _make_code_predictor_flat_wrapper(model: nn.Module, Na: int) -> nn.Module:
    """Build a flat-signature wrapper for CodePredictor ONNX export.

    Unlike the standard CausalLM wrapper, this includes:
    - ``lm_head_weight`` as an input (dynamic lm_head for 15 CUDA graphs)
    - ``hidden_states`` as an output (for residual connection)
    """
    param_names: List[str] = (
        ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
            "rope_rotary_cos_sin", "context_lengths", "kvcache_start_index",
            "kv_page_table", "last_token_ids", "lm_head_weight"
        ])

    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(Na))) if Na else "()"

    body = (
        f"    logits, hidden_states, present_key_values = self._model(\n"
        f"        inputs_embeds, {past_kv_tuple}, rope_rotary_cos_sin, "
        f"context_lengths, kvcache_start_index, kv_page_table, "
        f"last_token_ids, lm_head_weight)\n"
        f"    return (logits, hidden_states) + tuple(present_key_values)\n")

    src = "def _forward(self, {}):\n{}".format(", ".join(param_names), body)
    globs: dict = {}
    exec(src, globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


# ---------------------------------------------------------------------------
# CodePredictorCausalLM
# ---------------------------------------------------------------------------


class CodePredictorCausalLM(CausalLM):
    """CP CausalLM: dynamic ``lm_head_weight`` input (15 heads switched
    per runtime step) + ``hidden_states`` output for the residual loop.
    """

    match_fp32_matmul_initializers = True

    def __init__(self, config) -> None:
        super().__init__(config)
        for layer_idx, layer in enumerate(self.model.layers):
            layer.mlp = CodePredictorMLP(config, layer_idx)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        last_token_ids: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> Tuple:
        hidden_states, present_key_values, _ = self.model(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
        )
        # Select last token hidden states via GatherND
        last_hidden = torch.ops.trt.gather_nd(hidden_states, last_token_ids)
        # Dynamic lm_head: logits = last_hidden @ lm_head_weight.T
        logits = torch.matmul(last_hidden, lm_head_weight.T).to(torch.float32)
        return logits, hidden_states, present_key_values

    def onnx_export_spec(self) -> OnnxSpec:
        """ONNX export spec with lm_head_weight input and hidden_states output."""
        config = self.config
        Na = config.num_hidden_layers
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
        # Paged KV pool binding: [2, num_pages, KV_PAGE_SIZE, num_kv_heads, head_dim].
        # num_pages is a dummy placeholder for export; the builder sets the real fixed
        # value (see llmBuilder.cpp setupKVCacheProfiles).
        past_key_values_list: List[torch.Tensor] = [
            torch.zeros(2,
                        1,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        config.head_dim,
                        dtype=dtype16,
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
        lm_head_weight = torch.zeros(config.vocab_size,
                                     config.hidden_size,
                                     dtype=dtype16,
                                     device=device)

        args = (inputs_embeds, *past_key_values_list, rope_rotary_cos_sin,
                context_lengths, kvcache_start_index, kv_page_table,
                last_token_ids, lm_head_weight)

        input_names = (["inputs_embeds"] +
                       [f"past_key_values_{i}" for i in range(Na)] + [
                           "rope_rotary_cos_sin", "context_lengths",
                           "kvcache_start_index", "kv_page_table",
                           "last_token_ids", "lm_head_weight"
                       ])
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
        all_shapes.append({})  # lm_head_weight (fixed shape)

        wrapped = _make_code_predictor_flat_wrapper(self, Na)
        wrapped.eval()

        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=all_shapes)
