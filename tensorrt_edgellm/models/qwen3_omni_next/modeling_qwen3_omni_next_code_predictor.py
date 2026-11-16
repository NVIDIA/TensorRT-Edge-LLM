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
"""Qwen3-Next Omni CodePredictor — dense decoder head.

- ``lm_heads`` + ``lm_head_idx`` are ONNX inputs (head gathered in-graph).
- Forward returns ``hidden_states`` for the residual connection.
"""
import dataclasses
import inspect
from typing import Tuple

import torch
import torch.nn as nn

from ..default.modeling_default import OnnxSpec
from ..qwen3_5.modeling_qwen3_5_text import Qwen3_5CausalLM

__all__ = ["Qwen3OmniNextCodePredictorCausalLM"]


def _wrap_with_dynamic_lm_head(base: nn.Module) -> nn.Module:
    """Wrap *base* so its forward takes ``lm_heads`` + ``lm_head_idx`` inputs
    and returns ``(logits, hidden_states, *present_kv)``.

    ``self._base._model`` is the Qwen3_5CausalLM instance; its forward stashed
    the full pre-last-token hidden_states and the post-gather last_hidden
    during trace for the wrapper to consume here.
    """
    sig = inspect.signature(base.forward)
    names = [
        n for n, p in sig.parameters.items()
        if p.kind is p.POSITIONAL_OR_KEYWORD
    ]
    # Append the stacked heads + device index as trailing positional inputs.
    new_names = names + ["lm_heads", "lm_head_idx"]
    base_call = ", ".join(names)
    src = (f"def _forward(self, {', '.join(new_names)}):\n"
           f"    out = self._base({base_call})\n"
           f"    full_hidden = out[1]\n"
           f"    last_hidden = torch.index_select(full_hidden, 0, "
           f"logits_indices)\n"
           f"    head = lm_heads.index_select(0, "
           f"lm_head_idx.to(torch.long)).squeeze(0)\n"
           f"    logits = torch.matmul(last_hidden, "
           f"head.T).to(torch.float32)\n"
           f"    return (logits, full_hidden) + tuple(out[2:])\n")
    globs: dict = {"torch": torch}
    exec(src, globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, b: nn.Module) -> None:
            super().__init__()
            self._base = b

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(base)


class Qwen3OmniNextCodePredictorCausalLM(Qwen3_5CausalLM):
    """CodePredictor with lm_heads/lm_head_idx inputs + hidden_states output."""

    # Keep down_proj.weight in FP32 so the silu*up intermediate stays FP32
    # through the down-projection matmul. Without this the generic dtype-fix
    # pass would re-cast it to FP16, producing a mixed-dtype MatMul TRT rejects.
    preserve_fp32_initializer_patterns = ("down_proj.weight", )
    match_fp32_matmul_initializers = True
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
        # Backbone returns a 7-tuple; CodePredictor only needs hidden + KV.
        (hidden, present_kv, present_conv, present_rec, _intermediate_conv,
         _intermediate_rec,
         _dflash_hidden) = self.model(inputs_embeds, past_key_values,
                                      rope_rotary_cos_sin, context_lengths,
                                      kvcache_start_index, kv_page_table,
                                      conv_states, recurrent_states)
        self._cp_full_hidden = hidden  # surfaced by the ONNX wrapper
        last_hidden = torch.ops.trt.gather_nd(hidden, last_token_ids)
        self._cp_last_hidden = last_hidden  # used by wrapper for dynamic lm_head
        # The base's static lm_head is unused at runtime (replaced by the
        # gathered head), but we still produce logits here so the parent
        # spec stays type-consistent - ONNX optimisation drops it.
        logits = self.lm_head(last_hidden).to(torch.float32)
        return (logits, present_kv, present_conv, present_rec)

    def onnx_export_spec(self) -> OnnxSpec:
        spec = super().onnx_export_spec()
        # Add lm_heads/lm_head_idx inputs + re-signature the wrapper; replace
        # output list with (logits, hidden_states, *present_key_values).
        # present_conv / present_rec are omitted because Ng=0 for the dense CP.
        config = self.config
        num_heads = config.num_code_groups - 1
        assert num_heads > 0, "num_code_groups missing from CP config"
        device = next(self.parameters()).device
        lm_heads = torch.zeros(num_heads,
                               config.vocab_size,
                               config.hidden_size,
                               dtype=torch.float16,
                               device=device)
        lm_head_idx = torch.zeros(1, dtype=torch.int32, device=device)
        Na = config.num_attn_layers
        output_names = (["logits", "hidden_states"] +
                        [f"present_key_values_{i}" for i in range(Na)])
        return dataclasses.replace(
            spec,
            wrapped=_wrap_with_dynamic_lm_head(spec.wrapped),
            args=spec.args + (lm_heads, lm_head_idx),
            input_names=list(spec.input_names) + ["lm_heads", "lm_head_idx"],
            output_names=output_names,
            dynamic_shapes=list(spec.dynamic_shapes) + [{}, {}],
        )
