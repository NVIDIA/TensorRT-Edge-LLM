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
"""Qwen3-Omni Next Thinker MoE text backbone.

The Thinker is a Qwen3.5 hybrid decoder (GatedDeltaNet linear-attention
layers + gated full-attention layers in the ``[lin,lin,lin,full]`` repeating
pattern) where every layer's FFN is a 256-expert sparse MoE block with a
single FP16 shared expert (``shared_expert`` gated by ``shared_expert_gate``).

This file mirrors the Qwen3-Omni MoE Thinker pattern
(:class:`...qwen3_omni.modeling_qwen3_omni_moe_text.Qwen3OmniMoeThinkerCausalLM`)
on top of the Qwen3.5-MoE backbone (:class:`Qwen3_5MoeCausalLM`):

1. Capture the **pre-norm** hidden state at decoder layer
   ``accept_hidden_layer - 1`` for the Talker (HF
   ``outputs.hidden_states[k]`` convention, k >= 1) — the exact same hook
   used by the dense Qwen3-Omni Thinker
   (:class:`...qwen3_omni.modeling_qwen3_omni_text.Qwen3OmniDenseTransformer`).
2. Expose that tensor as a second ``hidden_states`` ONNX output via
   ``emit_hidden_states = True`` so the C++ TTS runtime can forward it to
   the Talker's input projection.
3. Reuse :class:`Qwen3_5MoeBackbone`'s layer stack — ``Qwen3_5SparseMoeBlock``
   already wires the routed-expert NVFP4/INT4 plugin branching and the
   ``shared_expert`` + ``shared_expert_gate`` add.

Dense path (``modeling_qwen3_omni_next_text.py`` /
:class:`Qwen3OmniNextLanguageModel`) is unchanged.
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

# Re-exported so checkpoint utilities matching on ``LAYER_GDN`` still resolve.
from ...config import LAYER_GDN  # noqa: F401  (kept for parity with parent)
from ...config import ModelConfig
from ..default.modeling_default import (OnnxSpec,
                                        _concat_hidden_in_provider_order)
from ..linear import make_linear
from ..qwen3_5.modeling_qwen3_5_text import Qwen3_5RMSNorm, _is_mtp_base_export
from ..qwen3_5_moe.modeling_qwen3_5_moe import (Qwen3_5MoeBackbone,
                                                Qwen3_5MoeCausalLM)

__all__ = ["Qwen3OmniNextMoeLanguageModel"]

logger = logging.getLogger(__name__)


def _emit_accept_hidden(config: ModelConfig) -> bool:
    """True when an ``mtp_base`` graph needs a separate accept-layer output.

    Only when the backbone really captures a distinct mid-stack tensor. With
    ``accept_hidden_layer`` unset or out of range it falls back to the post-norm
    output, and returning that same value twice lets the ONNX exporter collapse
    the pair — which silently shifts every later output name by one.

    The bound mirrors ``Qwen3OmniNextMoeBackbone.forward``, whose capture loop
    zips ``layers`` with ``layer_types``: bounding on ``num_hidden_layers``
    alone would let a config with a shorter ``layer_types`` pass this gate while
    the loop never captures, producing exactly the aliased pair above.
    """
    k = int(getattr(config, "accept_hidden_layer", -1))
    depth = min(int(config.num_hidden_layers), len(config.layer_types))
    return 1 <= k <= depth


# ---------------------------------------------------------------------------
# Backbone with accept_hidden_layer / emitted_hidden_states hook
# ---------------------------------------------------------------------------


class Qwen3OmniNextMoeBackbone(Qwen3_5MoeBackbone):
    """Qwen3.5 hybrid MoE backbone with Qwen3-Omni mid-layer hidden hook.

    Identical to :class:`Qwen3_5MoeBackbone` except :meth:`forward` captures
    the pre-norm output of decoder layer ``accept_hidden_layer - 1`` (HF
    ``hidden_states[k]`` convention) and exposes it as
    ``self.emitted_hidden_states`` when ``accept_hidden_layer >= 1``.

    Selection rule (matches :class:`Qwen3OmniDenseTransformer` and
    :class:`Qwen3MoeTransformer` semantics):

    * ``accept_hidden_layer >= 1`` (and ``<= num_hidden_layers``) → pre-norm
      output of decoder layer ``accept_hidden_layer - 1``
      (Thinker → Talker; HF reads ``outputs.hidden_states[k]``).
    * otherwise → post-final-norm output. Talker → CodePredictor and any
      checkpoint that inherits a stale ``accept_hidden_layer`` value
      exceeding the actual layer count falls back to post-norm.
    """

    def __init__(self, config: ModelConfig) -> None:
        # ``Qwen3_5MoeBackbone.__init__`` uses ``nn.Module.__init__`` directly
        # (it intentionally skips the dense ``Qwen3_5Backbone.__init__`` to
        # rewire the MoE decoder layers). Re-invoke the parent so layers /
        # embed_tokens / norm are built; then add the mid-layer hook state.
        super().__init__(config)
        self.accept_hidden_layer: int = int(
            getattr(config, "accept_hidden_layer", -1))
        self.last_pre_norm_hidden_states: "torch.Tensor | None" = None
        self.emitted_hidden_states: "torch.Tensor | None" = None

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        conv_states: Tuple[torch.Tensor, ...] = (),
        recurrent_states: Tuple[torch.Tensor, ...] = (),
        attention_mask: "torch.Tensor | None" = None,
        attention_pos_id: "torch.Tensor | None" = None,
        execution_phase_marker: "torch.Tensor | None" = None,
        collect_intermediate_states: bool = False,
        dflash_target_layer_ids: "List[int] | None" = None,
    ) -> Tuple[torch.Tensor, Tuple, Tuple, Tuple, Tuple, Tuple, object]:
        # Mirror Qwen3_5Backbone.forward but add the pre-norm capture hook.
        # We cannot simply call ``super().forward`` and reach inside because
        # the capture has to happen between the layer call and the final
        # ``self.norm``.
        hidden_states = inputs_embeds
        present_key_values_list: List[torch.Tensor] = []
        present_conv_states_list: List[torch.Tensor] = []
        present_recurrent_states_list: List[torch.Tensor] = []
        intermediate_conv_states_list: List[torch.Tensor] = []
        intermediate_recurrent_states_list: List[torch.Tensor] = []
        dflash_hidden_by_layer: dict[int, torch.Tensor] = {}
        dflash_target_set = set(dflash_target_layer_ids or [])
        attn_idx = 0
        gdn_idx = 0

        target_layer = self.accept_hidden_layer
        captured: "torch.Tensor | None" = None

        for layer_idx, (layer,
                        lt) in enumerate(zip(self.layers, self.layer_types)):
            if lt == LAYER_GDN:
                (hidden_states, conv_out, rec_out, intermediate_conv_out,
                 intermediate_rec_out) = layer(
                     hidden_states,
                     context_lengths=context_lengths,
                     conv_state=conv_states[gdn_idx],
                     recurrent_state=recurrent_states[gdn_idx],
                     execution_phase_marker=execution_phase_marker,
                     collect_intermediate_states=collect_intermediate_states,
                 )
                present_conv_states_list.append(conv_out)
                present_recurrent_states_list.append(rec_out)
                if collect_intermediate_states:
                    intermediate_conv_states_list.append(intermediate_conv_out)
                    intermediate_recurrent_states_list.append(
                        intermediate_rec_out)
                gdn_idx += 1
            else:
                hidden_states, present_kv = layer(
                    hidden_states,
                    past_key_value=past_key_values[attn_idx],
                    rope_rotary_cos_sin=rope_rotary_cos_sin,
                    context_lengths=context_lengths,
                    kvcache_start_index=kvcache_start_index,
                    kv_page_table=kv_page_table,
                    attention_mask=attention_mask,
                    attention_pos_id=attention_pos_id,
                )
                present_key_values_list.append(present_kv)
                attn_idx += 1

            if layer_idx in dflash_target_set:
                dflash_hidden_by_layer[layer_idx] = hidden_states

            if target_layer >= 1 and layer_idx == target_layer - 1:
                captured = hidden_states

        self.last_pre_norm_hidden_states = (captured if captured is not None
                                            else hidden_states)

        normed_hidden = self.norm(hidden_states)
        dflash_hidden_concat = _concat_hidden_in_provider_order(
            dflash_hidden_by_layer, dflash_target_layer_ids)

        # Defensive bounds check: a Talker checkpoint may inherit a stale
        # ``accept_hidden_layer`` exceeding its real layer count. In that
        # case ``captured`` stays ``None`` and we fall back to post-norm
        # rather than silently emitting the last layer's pre-norm.
        if target_layer >= 1 and captured is not None:
            self.emitted_hidden_states = self.last_pre_norm_hidden_states
        else:
            self.emitted_hidden_states = normed_hidden

        return (normed_hidden, tuple(present_key_values_list),
                tuple(present_conv_states_list),
                tuple(present_recurrent_states_list),
                tuple(intermediate_conv_states_list),
                tuple(intermediate_recurrent_states_list),
                dflash_hidden_concat)

    def _capture_ragged_layer_output(self, layer_idx: int,
                                     hidden_states: torch.Tensor) -> None:
        if self.accept_hidden_layer >= 1 and layer_idx == self.accept_hidden_layer - 1:
            self._ragged_captured_hidden = hidden_states

    def forward_ragged(self, *args, **kwargs) -> Tuple:
        self._ragged_captured_hidden = None
        outputs = super().forward_ragged(*args, **kwargs)
        captured = self._ragged_captured_hidden
        self.last_pre_norm_hidden_states = (captured if captured is not None
                                            else outputs[0])
        self.emitted_hidden_states = (captured
                                      if captured is not None else outputs[0])
        return outputs


# ---------------------------------------------------------------------------
# Flat ONNX wrapper with hidden_states output
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CausalLM with emit_hidden_states wiring
# ---------------------------------------------------------------------------


class Qwen3OmniNextMoeLanguageModel(Qwen3_5MoeCausalLM):
    """Qwen3-Omni Next Thinker MoE causal LM.

    Extends :class:`Qwen3_5MoeCausalLM` with:

    * :class:`Qwen3OmniNextMoeBackbone` as the inner ``self.model`` (adds the
      ``accept_hidden_layer`` pre-norm capture hook).
    * ``emit_hidden_states = True`` — :meth:`forward` returns an extra
      ``hidden_states`` tensor (the captured pre-norm layer-k output, or
      post-norm if ``accept_hidden_layer < 1``).
    * :meth:`onnx_export_spec` adds a ``hidden_states`` output to the ONNX
      graph between ``logits`` and the KV/conv/recurrent present states.

    The sparse-MoE block (:class:`Qwen3_5SparseMoeBlock`) with NVFP4↔INT4
    routed-expert branching plus the FP16 ``shared_expert`` and
    ``shared_expert_gate`` is wired by the parent classes — nothing extra
    to do here.
    """

    emit_hidden_states: bool = True

    def __init__(self, config: ModelConfig) -> None:
        # Bypass ``Qwen3_5MoeCausalLM.__init__`` (which would allocate a
        # default ``Qwen3_5MoeBackbone`` only to discard it) and assemble
        # the wrapper with the Qwen3-Omni-Next-specific backbone directly.
        # Matches the pattern used by ``Qwen3OmniLanguageModel.__init__``.
        nn.Module.__init__(self)
        self.config = config
        self.emit_accept_hidden_states = _emit_accept_hidden(config)
        self.model = Qwen3OmniNextMoeBackbone(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward_ragged(self, *args, **kwargs) -> Tuple:
        outputs = list(super().forward_ragged(*args, **kwargs))
        if _is_mtp_base_export(self.config):
            if self.emit_accept_hidden_states:
                outputs.insert(2, self.model.emitted_hidden_states)
        else:
            outputs[1] = self.model.emitted_hidden_states
        return tuple(outputs)

    def forward(  # type: ignore[override]
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
        attention_pos_id: "torch.Tensor | None" = None,
        attention_mask: "torch.Tensor | None" = None,
        execution_phase_marker: "torch.Tensor | None" = None,
    ) -> Tuple:
        mtp_base = _is_mtp_base_export(self.config)
        (hidden_states, present_key_values, present_conv_states,
         present_recurrent_states, intermediate_conv_states,
         intermediate_recurrent_states, _dflash_hidden_concat) = self.model(
             inputs_embeds,
             past_key_values,
             rope_rotary_cos_sin,
             context_lengths,
             kvcache_start_index,
             kv_page_table,
             conv_states,
             recurrent_states,
             attention_mask=attention_mask,
             attention_pos_id=attention_pos_id,
             execution_phase_marker=execution_phase_marker,
             collect_intermediate_states=mtp_base,
         )
        # Select hidden states for the requested token positions before
        # lm_head — matches the parent's ``gather_nd`` pattern.
        selected_hidden_states = torch.ops.trt.gather_nd(
            hidden_states, last_token_ids)
        logits = self.lm_head(selected_hidden_states).to(torch.float32)

        if mtp_base:
            # Slot 1 is post-norm (the draft's contract) and slot 2 the
            # mid-stack capture (the Talker's). Collapsing them onto
            # ``emitted_hidden_states`` would feed the draft a pre-norm tensor
            # and quietly collapse its acceptance rate.
            if _emit_accept_hidden(self.config):
                return (logits, hidden_states,
                        self.model.emitted_hidden_states, present_key_values,
                        present_conv_states, present_recurrent_states,
                        intermediate_conv_states,
                        intermediate_recurrent_states)
            return (logits, hidden_states, present_key_values,
                    present_conv_states, present_recurrent_states,
                    intermediate_conv_states, intermediate_recurrent_states)

        # Emit the full-sequence captured (pre-norm layer-k) tensor for the
        # downstream Talker stage. The selection logic (pre-norm vs post-norm)
        # lives inside ``Qwen3OmniNextMoeBackbone.forward``.
        return (logits, self.model.emitted_hidden_states, present_key_values,
                present_conv_states, present_recurrent_states)

    # ------------------------------------------------------------------ #
    # ONNX export spec — add ``hidden_states`` output                     #
    # ------------------------------------------------------------------ #

    def onnx_export_spec(self) -> OnnxSpec:  # type: ignore[override]
        """Return the unified hybrid token-major export contract."""
        return Qwen3_5MoeCausalLM._token_major_onnx_export_spec(self)


# Silence "imported but unused" for the RMSNorm pulled in only so downstream
# checkpoint utilities relying on it staying importable from this module
# (parity with the parent module's re-exports) don't break.
_ = Qwen3_5RMSNorm
