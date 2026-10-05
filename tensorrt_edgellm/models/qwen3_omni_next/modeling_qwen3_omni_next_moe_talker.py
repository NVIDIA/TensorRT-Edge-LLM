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
"""Qwen3-Omni Next Talker MoE - sparse-MoE Talker that emits hidden_states.

This is the MoE counterpart of :class:`Qwen3OmniNextTalkerCausalLM`. Structurally
it is the Qwen3.5 MoE hybrid stack (GDN + gated full attention + sparse MoE FFN
with shared expert) configured for the Talker codec vocabulary:

* ``hidden_size = 1280``  (vs dense Talker's 1024)
* ``num_hidden_layers = 20``
* ``num_experts = 128``, ``num_experts_per_tok = 8``,
  ``moe_intermediate_size = 512``
* ``linear_num_value_heads = 32``, ``linear_num_key_heads = 16`` (gqa=2,
  matching the Thinker GDN config)
* ``full_attention_interval = 4`` -> ``layer_types = [lin,lin,lin,full] x 5``
* ``vocab_size = 5120`` (codec vocab, NOT the text vocab)
* ``model_type = 'qwen3_omni_next_talker_text'``

The TTS runtime only needs three extras vs a plain Qwen3.5 MoE causal LM:

1. The pre-lm_head hidden states are emitted as a second ONNX output so the
   C++ Talker -> CodePredictor residual path can pick them up.
2. The HF Talker names its input embedding ``codec_embedding`` (not
   ``embed_tokens``) and its output projection ``codec_head`` (not
   ``lm_head``); the two tensors are independent in the checkpoint. We
   alias ``embed_tokens`` -> ``codec_embedding`` and rebuild ``lm_head``
   with ``module_name='codec_head'`` so the FP16 disable-glob matches.
3. Sidecar tables (``hidden_projection`` Linear, ``speaker_codec_embeddings``)
   are NOT part of the ONNX Talker graph - they live as separate safetensors
   the C++ runtime loads (see scripts/export.py sidecar handling). The model
   class therefore does not own them, mirroring the dense Talker.
"""

from typing import Tuple

import torch

from ..default.modeling_default import OnnxSpec
from ..linear import make_linear
from ..qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeCausalLM

__all__ = ["Qwen3OmniNextMoeTalkerCausalLM"]


class Qwen3OmniNextMoeTalkerCausalLM(Qwen3_5MoeCausalLM):
    """Qwen3-Omni Next MoE Talker: Qwen3.5 MoE backbone over the codec vocab,
    plus the ``hidden_states`` ONNX output the CodePredictor residual needs.

    Like :class:`Qwen3OmniNextTalkerCausalLM`, this subclass overrides
    :meth:`forward` only to splice the post-gather hidden states into a
    side-channel field (``_talker_last_hidden``); the ONNX wrapper installed
    by :meth:`onnx_export_spec` then surfaces it as a positional output.

    The ``hidden_projection`` FP16 Linear (Thinker 2048 -> Talker 1280) and the
    ``speaker_codec_embeddings`` table are sidecar tensors held by the C++
    runtime, not the ONNX graph; they are excluded from the quant globs
    (see ``quantization/qwen3_omni.py`` ``DISABLE_NON_LLM``) and the export
    script writes them as separate safetensors. This class therefore does
    not own them, matching the dense Talker.
    """

    emit_hidden_states = True

    def __init__(self, config) -> None:
        super().__init__(config)
        # HF Talker names the input embedding ``codec_embedding`` rather than
        # ``embed_tokens``. The shared loader keys are remapped onto our
        # ``embed_tokens`` slot, so alias the HF name onto the same tensor.
        self.model.codec_embedding = self.model.embed_tokens

        # HF Talker names the output projection ``codec_head``; this tensor
        # is independent of ``codec_embedding`` (not tied). Recreate the
        # ``lm_head`` Linear with the HF module name so quant-exclusion globs
        # for ``codec_head`` select the FP16 implementation (the codec head
        # is intentionally kept FP16, not INT4).
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="codec_head")
        self.codec_head = self.lm_head

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
        # Qwen3.5 MoE backbone returns the same 7-tuple as the dense variant;
        # MTP/DFlash do not apply to the Talker so we discard the last three.
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
