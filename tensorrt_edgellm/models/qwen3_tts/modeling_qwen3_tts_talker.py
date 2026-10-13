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

from typing import Tuple

import torch

from ..default.modeling_default import CausalLM, OnnxSpec

__all__ = ["TalkerCausalLM"]


class TalkerCausalLM(CausalLM):
    """Talker variant of CausalLM that also outputs hidden_states.

    The TTS runtime requires ``hidden_states`` from the talker engine
    for the residual connection to the CodePredictor.
    """

    emit_hidden_states = True

    def _ragged_emitted_hidden(self) -> torch.Tensor:
        hidden_states = self.model.norm(self.model.last_pre_norm_hidden_states)
        return hidden_states.reshape(-1, hidden_states.shape[-1])

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
        return super().onnx_export_spec()
