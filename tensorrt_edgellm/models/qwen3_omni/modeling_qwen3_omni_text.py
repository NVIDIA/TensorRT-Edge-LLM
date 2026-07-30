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
Qwen3-Omni LLM backbone (dense Thinker / Talker).

The Qwen3-Omni Thinker is a standard Qwen3 decoder, but its Talker consumes
the pre-norm output of decoder layer ``accept_hidden_layer - 1`` rather than
the post-final-norm output — matching HF's ``outputs.hidden_states[k]``
semantics. The dense Talker reuses the same backbone but emits the
post-final-norm tensor (``accept_hidden_layer < 1``).

To keep ``default.Transformer`` / ``default.CausalLM`` free of Qwen3-Omni-
specific plumbing, the mid-layer hook lives in subclasses here. The Qwen3-MoE
variants apply the same pattern in :mod:`..qwen3_moe.modeling_qwen3_moe`.
"""
from typing import List, Tuple

import torch
from torch import nn

from ..default.modeling_default import CausalLM, Transformer
from ..linear import make_linear

__all__ = ["Qwen3OmniLanguageModel"]


class Qwen3OmniDenseTransformer(Transformer):
    """Dense Qwen3 decoder with a Qwen3-Omni mid-layer hidden-state hook.

    Adds ``accept_hidden_layer`` (HF ``hidden_states[k]`` convention, k>=1)
    to the parent ``Transformer`` and overrides :meth:`forward` to capture
    that layer's pre-norm output. ``emitted_hidden_states`` is the tensor
    consumed by :class:`Qwen3OmniLanguageModel` when ``emit_hidden_states``
    is true:

    * ``accept_hidden_layer >= 1`` (and ≤ ``num_hidden_layers``) → pre-norm
      output of decoder layer ``accept_hidden_layer - 1`` (Thinker → Talker).
    * otherwise → post-final-norm output (Talker → CodePredictor; matches
      HF's ``outputs.hidden_states[0][-1]`` resolving to the post-norm
      tensor in ``modeling_qwen3_omni.py``).
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.accept_hidden_layer: int = int(
            getattr(config, "accept_hidden_layer", -1))
        self.emitted_hidden_states: "torch.Tensor | None" = None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        deepstack_embeds: Tuple[torch.Tensor, ...] = (),
        attention_mask: "torch.Tensor | None" = None,
        attention_pos_id: "torch.Tensor | None" = None,
        output_hidden_states: bool = False,
        dflash_target_layer_ids: "List[int] | None" = None,
    ) -> Tuple[torch.Tensor, Tuple, "Tuple | None"]:
        hidden_states = inputs_embeds
        present_key_values_list: List[torch.Tensor] = []
        all_hidden_states: list = []
        dflash_hidden_list: list = []
        dflash_target_set = set(dflash_target_layer_ids or [])

        target_layer = self.accept_hidden_layer
        captured: "torch.Tensor | None" = None

        for layer_index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states, next_key_value = layer(
                hidden_states,
                past_key_values[layer_index],
                rope_rotary_cos_sin,
                context_lengths,
                kvcache_start_index,
                kv_page_table,
                attention_mask=attention_mask,
                attention_pos_id=attention_pos_id,
            )
            present_key_values_list.append(next_key_value)

            if layer_index in dflash_target_set:
                dflash_hidden_list.append(hidden_states)

            if layer_index < len(deepstack_embeds):
                hidden_states = hidden_states + deepstack_embeds[layer_index]

            if target_layer >= 1 and layer_index == target_layer - 1:
                captured = hidden_states

        self.last_pre_norm_hidden_states = (captured if captured is not None
                                            else hidden_states)
        self.dflash_hidden_concat = (torch.cat(dflash_hidden_list, dim=-1)
                                     if dflash_hidden_list else None)

        normed = self.norm(hidden_states)

        # ``captured is not None`` guards against ``accept_hidden_layer``
        # exceeding the actual layer count (Talker checkpoints sometimes
        # inherit Thinker's value): fall back to post-norm in that case.
        if target_layer >= 1 and captured is not None:
            self.emitted_hidden_states = self.last_pre_norm_hidden_states
        else:
            self.emitted_hidden_states = normed

        if output_hidden_states:
            all_hidden_states.append(normed)

        return (normed, tuple(present_key_values_list),
                tuple(all_hidden_states) if output_hidden_states else None)


class Qwen3OmniLanguageModel(CausalLM):
    """Qwen3-Omni Thinker/Talker CausalLM.

    Replaces the parent's ``self.model`` with
    :class:`Qwen3OmniDenseTransformer` and routes ``emit_hidden_states`` to
    its ``emitted_hidden_states`` so the ONNX export carries the correct
    pre-norm-layer-k / post-norm tensor for the next stage.
    """

    emit_hidden_states = True

    def __init__(self, config) -> None:
        # Bypass ``CausalLM.__init__`` (which would allocate a default
        # ``Transformer`` only to discard it) and assemble the wrapper with
        # the Qwen3-Omni-specific backbone directly.
        nn.Module.__init__(self)
        self.config = config
        self.model = Qwen3OmniDenseTransformer(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        last_token_ids: torch.Tensor,
        deepstack_embeds: Tuple[torch.Tensor, ...] = (),
        attention_mask: "torch.Tensor | None" = None,
        attention_pos_id: "torch.Tensor | None" = None,
    ) -> Tuple:
        hidden_states, present_key_values, _ = self.model(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
            deepstack_embeds,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
        )
        selected_hidden_states = torch.ops.trt.gather_nd(
            hidden_states, last_token_ids)
        logits = self.lm_head(selected_hidden_states).to(torch.float32)
        return logits, self.model.emitted_hidden_states, present_key_values
