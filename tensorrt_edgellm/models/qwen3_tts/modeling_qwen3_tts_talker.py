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
Qwen3-TTS Talker — speech codec token predictor.

The Talker is a Qwen3-architecture LLM decoder fine-tuned on speech codec
prediction.  It shares the Qwen3 transformer architecture but outputs both
``logits`` and ``hidden_states`` (required by the TTS runtime for the
residual connection to the CodePredictor).

Checkpoint prefix: ``talker.*``

Exported via the standard CausalLM pipeline with ``key_prefix="talker."``
and a minimal ``key_remap`` to rename ``codec_embedding`` → ``embed_tokens``.

Extra weight files (``text_embedding.safetensors``, ``text_projection.safetensors``)
are extracted separately by :func:`tensorrt-edgellm-export._extract_tts_weights`.
"""

import itertools
from typing import List, Tuple

import torch
import torch.nn as nn

from ..default.modeling_default import (_BATCH_SIZE, _MAX_POS, _PAST_LEN,
                                        _SEQ_LEN, CausalLM, OnnxSpec)
from ..ops import KV_PAGE_SIZE

__all__ = ["TalkerCausalLM"]


def _make_talker_flat_wrapper(model: nn.Module, Na: int) -> nn.Module:
    """Build a flat-signature wrapper for Talker ONNX export.

    Like the standard CausalLM wrapper, but the forward returns
    ``(logits, hidden_states) + present_key_values``.
    """
    param_names: List[str] = (
        ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
            "rope_rotary_cos_sin", "context_lengths", "kvcache_start_index",
            "kv_page_table", "last_token_ids"
        ])

    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(Na))) if Na else "()"

    body = (
        f"    logits, hidden_states, present_key_values = self._model(\n"
        f"        inputs_embeds, {past_kv_tuple}, rope_rotary_cos_sin, "
        f"context_lengths, kvcache_start_index, kv_page_table, "
        f"last_token_ids)\n"
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


class TalkerCausalLM(CausalLM):
    """Talker variant of CausalLM that also outputs hidden_states.

    The TTS runtime requires ``hidden_states`` from the talker engine
    for the residual connection to the CodePredictor.
    """

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        last_token_ids: torch.Tensor,
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
        logits = self.lm_head(last_hidden).to(torch.float32)
        # Talker's ``hidden_states`` output is the post-norm final layer
        # output — matches the reference tensorrt_edgellm export, which the
        # CodePredictor's residual path was trained to consume.
        return logits, hidden_states, present_key_values

    def onnx_export_spec(self) -> OnnxSpec:
        """ONNX export spec with hidden_states output."""
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

        args = (inputs_embeds, *past_key_values_list, rope_rotary_cos_sin,
                context_lengths, kvcache_start_index, kv_page_table,
                last_token_ids)

        input_names = (
            ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
                "rope_rotary_cos_sin", "context_lengths",
                "kvcache_start_index", "kv_page_table", "last_token_ids"
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

        wrapped = _make_talker_flat_wrapper(self, Na)
        wrapped.eval()

        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=all_shapes)
