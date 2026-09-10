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

import itertools
import logging
from typing import List, Tuple

import torch
import torch.nn as nn

# Re-exported so checkpoint utilities matching on ``LAYER_GDN`` still resolve.
from ...config import LAYER_GDN  # noqa: F401  (kept for parity with parent)
from ...config import ModelConfig
from ..default.modeling_default import OnnxSpec
from ..linear import make_linear
from ..ops import KV_PAGE_SIZE
from ..qwen3_5.modeling_qwen3_5_text import (_BATCH_SIZE, _MAX_POS, _PAST_LEN,
                                             _SEQ_LEN, Qwen3_5RMSNorm,
                                             _is_mtp_base_export)
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
        spec_verify_phase_marker: "torch.Tensor | None" = None,
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
        dflash_hidden_list: List[torch.Tensor] = []
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
                     spec_verify_phase_marker=spec_verify_phase_marker,
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
                dflash_hidden_list.append(hidden_states)

            if target_layer >= 1 and layer_idx == target_layer - 1:
                captured = hidden_states

        self.last_pre_norm_hidden_states = (captured if captured is not None
                                            else hidden_states)

        normed_hidden = self.norm(hidden_states)
        dflash_hidden_concat = (torch.cat(dflash_hidden_list, dim=-1)
                                if dflash_hidden_list else None)

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


# ---------------------------------------------------------------------------
# Flat ONNX wrapper with hidden_states output
# ---------------------------------------------------------------------------


def _make_flat_wrapper_hybrid_with_hidden(
        model: nn.Module,
        Na: int,
        Ng: int,
        mtp_base: bool = False,
        emit_accept_hidden: bool = False) -> nn.Module:
    """Build flat forward wrapper for hybrid MoE Thinker with hidden_states.

    Like :func:`...qwen3_5.modeling_qwen3_5_text._make_flat_wrapper_hybrid`
    but with an extra ``hidden_states`` output slot between ``logits`` and
    the KV present states — matching the Qwen3-Omni MoE Thinker emission
    convention.

    Argument order on the inner model matches
    :meth:`Qwen3OmniNextMoeLanguageModel.forward`.
    """
    param_names: List[str] = (
        ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
            "rope_rotary_cos_sin", "context_lengths", "kvcache_start_index",
            "kv_page_table", "last_token_ids"
        ] + [f"conv_state_{i}"
             for i in range(Ng)] + [f"recurrent_state_{i}" for i in range(Ng)])
    if mtp_base:
        param_names += [
            "attention_pos_id", "attention_mask", "spec_verify_phase_marker"
        ]

    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(Na))) if Na else "()"
    conv_tuple = "({},)".format(", ".join(f"conv_state_{i}"
                                          for i in range(Ng))) if Ng else "()"
    rec_tuple = "({},)".format(", ".join(f"recurrent_state_{i}"
                                         for i in range(Ng))) if Ng else "()"

    if mtp_base:
        # The unpack arity here must equal the inner forward's return arity, and
        # the returned tuple must line up with ``output_names`` element for
        # element — a one-slot drift mislabels every present_* binding.
        accept = ", accept_hidden_states" if emit_accept_hidden else ""
        body = (
            f"    logits, hidden_states{accept}, present_key_values, "
            f"present_conv_states, present_recurrent_states, "
            f"intermediate_conv_states, "
            f"intermediate_recurrent_states = self._model(\n"
            f"        inputs_embeds, {past_kv_tuple}, rope_rotary_cos_sin, "
            f"context_lengths, kvcache_start_index, kv_page_table, "
            f"last_token_ids,\n"
            f"        {conv_tuple}, {rec_tuple},\n"
            f"        attention_pos_id=attention_pos_id, "
            f"attention_mask=attention_mask, "
            f"spec_verify_phase_marker=spec_verify_phase_marker)\n"
            f"    return ((logits, hidden_states{accept})\n"
            f"            + tuple(present_key_values)\n"
            f"            + tuple(present_conv_states)\n"
            f"            + tuple(present_recurrent_states)\n"
            f"            + tuple(intermediate_conv_states)\n"
            f"            + tuple(intermediate_recurrent_states))\n")
    else:
        body = (
            f"    logits, hidden_states, present_key_values, "
            f"present_conv_states, present_recurrent_states = self._model(\n"
            f"        inputs_embeds, {past_kv_tuple}, rope_rotary_cos_sin, "
            f"context_lengths, kvcache_start_index, kv_page_table, "
            f"last_token_ids,\n"
            f"        {conv_tuple}, {rec_tuple})\n"
            f"    return ((logits, hidden_states)\n"
            f"            + tuple(present_key_values)\n"
            f"            + tuple(present_conv_states)\n"
            f"            + tuple(present_recurrent_states))\n")

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
        self.model = Qwen3OmniNextMoeBackbone(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

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
        spec_verify_phase_marker: "torch.Tensor | None" = None,
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
             spec_verify_phase_marker=spec_verify_phase_marker,
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
        config = self.config
        gc = config.gdn_cfg
        Na = config.num_attn_layers
        Ng = config.num_gdn_layers
        assert gc is not None, (
            "Qwen3-Omni Next MoE Thinker requires gdn_cfg (the hybrid "
            "decoder always has at least one GDN layer).")
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
        kv_dtype = (torch.float8_e4m3fn
                    if config.quant.kv_cache_quant == "fp8" else dtype16)
        # Paged KV pool binding: [2, num_pages, KV_PAGE_SIZE, kv_heads, head_dim].
        past_key_values_list: List[torch.Tensor] = [
            torch.zeros(2,
                        1,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        config.head_dim,
                        dtype=kv_dtype,
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

        conv_states: List[torch.Tensor] = [
            torch.zeros(batch_size,
                        gc.conv_dim,
                        gc.conv_kernel,
                        dtype=self.CONV_STATE_DTYPE,
                        device=device) for _ in range(Ng)
        ]
        recurrent_states: List[torch.Tensor] = [
            torch.zeros(batch_size,
                        gc.num_value_heads,
                        gc.key_head_dim,
                        gc.value_head_dim,
                        dtype=self.RECURRENT_STATE_DTYPE,
                        device=device) for _ in range(Ng)
        ]

        args = (inputs_embeds, *past_key_values_list, rope_rotary_cos_sin,
                context_lengths, kvcache_start_index, kv_page_table,
                last_token_ids, *conv_states, *recurrent_states)

        input_names = (
            ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
                "rope_rotary_cos_sin", "context_lengths",
                "kvcache_start_index", "kv_page_table", "last_token_ids"
            ] + [f"conv_state_{i}" for i in range(Ng)] +
            [f"recurrent_state_{i}" for i in range(Ng)])
        output_names = (["logits", "hidden_states"] +
                        [f"present_key_values_{i}" for i in range(Na)] +
                        [f"present_conv_state_{i}" for i in range(Ng)] +
                        [f"present_recurrent_state_{i}" for i in range(Ng)])

        batch = torch.export.Dim("batch", min=1, max=256)
        seq = torch.export.Dim("seq_len", min=1, max=32768)
        pos = torch.export.Dim("max_pos", min=1, max=32768)
        rope_batch = torch.export.Dim("rope_batch", min=1, max=256)
        kv_batch = torch.export.Dim("kv_batch", min=1, max=256)
        page_batch = torch.export.Dim("page_batch", min=1, max=256)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        num_pages = torch.export.Dim("num_pages", min=1, max=1048576)
        mtp_base = _is_mtp_base_export(config)
        # The wrapper's return tuple and output_names must agree, so both read
        # this one value.
        emit_accept = mtp_base and _emit_accept_hidden(config)
        if mtp_base and not emit_accept:
            # Text-only MTP is legitimate, so warn rather than raise; without it
            # the mismatch only surfaces when someone wires up a Talker.
            logger.warning(
                "MTP base export: accept_hidden_layer=%s is unusable for "
                "%d layers, so no '%s' output is emitted. This engine cannot "
                "feed a Qwen3-Omni Talker (text-only speculative decoding).",
                getattr(config, "accept_hidden_layer", None),
                config.num_hidden_layers, "accept_hidden_states")
        num_selected = torch.export.Dim("num_selected", min=1,
                                        max=256) if mtp_base else None

        all_shapes: list = [{0: batch, 1: seq}]  # inputs_embeds
        for _ in range(Na):
            all_shapes.append({1: num_pages})  # past_key_values_i (pool)
        all_shapes.append({0: rope_batch, 1: pos})  # rope_rotary_cos_sin
        all_shapes.append({0: batch})  # context_lengths
        all_shapes.append({0: kv_batch})  # kvcache_start_index
        all_shapes.append({0: page_batch, 2: max_pages})  # kv_page_table
        if mtp_base:
            all_shapes.append({0: batch, 1: num_selected})  # last_token_ids
        else:
            all_shapes.append({0: batch})  # last_token_ids
        for _ in range(Ng):
            all_shapes.append({0: batch})  # conv_state_i
        for _ in range(Ng):
            all_shapes.append({0: batch})  # recurrent_state_i

        if mtp_base:
            attention_pos_id = torch.zeros(batch_size,
                                           seq_len,
                                           dtype=torch.int32,
                                           device=device)
            attention_mask = torch.zeros(batch_size,
                                         seq_len,
                                         seq_len + past_len,
                                         dtype=torch.int32,
                                         device=device)
            spec_verify_phase_marker = torch.zeros(1,
                                                   dtype=torch.int32,
                                                   device=device)
            args = args + (attention_pos_id, attention_mask,
                           spec_verify_phase_marker)
            input_names = input_names + [
                "attention_pos_id", "attention_mask",
                "spec_verify_phase_marker"
            ]
            # ``all_shapes`` below is positional against ``args`` (inputs only),
            # so a new OUTPUT needs no entry there.
            head = ["logits", "hidden_states"]
            if emit_accept:
                head.append("accept_hidden_states")
            output_names = (
                head + [f"present_key_values_{i}" for i in range(Na)] +
                [f"present_conv_state_{i}" for i in range(Ng)] +
                [f"present_recurrent_state_{i}" for i in range(Ng)] +
                [f"intermediate_conv_state_{i}" for i in range(Ng)] +
                [f"intermediate_recurrent_state_{i}" for i in range(Ng)])

            verify_seq = torch.export.Dim("verify_seq_len", min=1, max=32768)
            mask_kv_len = torch.export.Dim("mask_kv_len", min=1, max=65536)
            all_shapes.append({0: batch, 1: verify_seq})  # attention_pos_id
            all_shapes.append({
                0: batch,
                1: verify_seq,
                2: mask_kv_len
            })  # attention_mask
            all_shapes.append({0: torch.export.Dim.AUTO
                               })  # spec_verify_phase_marker

        wrapped = _make_flat_wrapper_hybrid_with_hidden(
            self, Na, Ng, mtp_base=mtp_base, emit_accept_hidden=emit_accept)
        wrapped.eval()

        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=all_shapes)


# Silence "imported but unused" for the RMSNorm pulled in only so downstream
# checkpoint utilities relying on it staying importable from this module
# (parity with the parent module's re-exports) don't break.
_ = Qwen3_5RMSNorm
