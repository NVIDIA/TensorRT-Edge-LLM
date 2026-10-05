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
"""Qwen3-Next Omni Talker — Talker CausalLM that also emits hidden_states
for the CodePredictor residual.
"""
from typing import Tuple

import torch

from ..default.modeling_default import OnnxSpec
from ..qwen3_5.modeling_qwen3_5_text import Qwen3_5CausalLM

__all__ = ["Qwen3OmniNextTalkerCausalLM"]


class Qwen3OmniNextTalkerCausalLM(Qwen3_5CausalLM):
    """Talker CausalLM that also surfaces its post-gather hidden_states."""

    emit_hidden_states = True

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
            recurrent_states: Tuple[torch.Tensor, ...] = (),
    ) -> Tuple:
        # Backbone returns a 7-tuple; Talker only needs hidden, kv, conv, rec.
        (hidden, present_kv, present_conv, present_rec, _intermediate_conv,
         _intermediate_rec,
         _dflash_hidden) = self.model(inputs_embeds, past_key_values,
                                      rope_rotary_cos_sin, context_lengths,
                                      kvcache_start_index, kv_page_table,
                                      conv_states, recurrent_states)
        hidden = torch.ops.trt.gather_nd(hidden, last_token_ids)
        self._talker_last_hidden = hidden  # picked up by the ONNX wrapper
        logits = self.lm_head(hidden).to(torch.float32)
        return (logits, present_kv, present_conv, present_rec)

    def onnx_export_spec(self) -> OnnxSpec:
        return super().onnx_export_spec()
