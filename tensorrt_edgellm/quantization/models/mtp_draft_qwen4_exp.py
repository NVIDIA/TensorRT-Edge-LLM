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
"""Qwen4-Exp MTP draft for quantization, built from the HF text classes.

Its one decoder layer is a base layer over the hyper-connection residual, so the HF modules
are reused rather than re-implemented.
"""

import copy
from typing import Dict, Optional

import torch
from torch import nn

from .mtp_draft import MtpDraftBase

# ---------------------------------------------------------------------------
# Qwen4-Exp MTP draft (hyper-connection residual, HF text classes)
# ---------------------------------------------------------------------------


def _qwen4_exp_modeling():
    from transformers.models.qwen4_exp import modeling_qwen4_exp
    return modeling_qwen4_exp


def _qwen4_exp_text_model(base_model: nn.Module) -> nn.Module:
    """The HF text backbone owning ``layers`` and ``hyper_connection_mixer``."""
    inner = getattr(base_model, "model", base_model)
    text_model = getattr(inner, "language_model", inner)
    if not hasattr(text_model, "hyper_connection_mixer"):
        raise ValueError("Qwen4-Exp base model has no hyper_connection_mixer")
    return text_model


def _draft_layer_config(config):
    """Config for the draft's single decoder layer: attention, no PLE."""
    attention_types = [
        t for t in config.layer_types if t != "linear_attention"
    ]
    if not attention_types:
        raise ValueError("Qwen4-Exp MTP draft needs an attention layer type "
                         f"in layer_types, got {config.layer_types}")
    layer_config = copy.deepcopy(config)
    layer_config.num_hidden_layers = 1
    layer_config.layer_types = attention_types[:1]
    layer_config.ple_layer_ids = []
    return layer_config


class _ExpertBank(nn.ModuleList):
    """Per-expert ``nn.Linear`` experts behind HF's stacked-expert call contract."""

    def forward(self, hidden_states, top_k_index, top_k_weights):
        routed = torch.zeros_like(hidden_states)
        for expert_id in torch.unique(top_k_index).tolist():
            token_idx, slot = torch.where(top_k_index == expert_id)
            expert_out = self[expert_id](hidden_states[token_idx])
            expert_out = expert_out * top_k_weights[token_idx, slot, None]
            routed.index_add_(0, token_idx, expert_out.to(routed.dtype))
        return routed


class Qwen4ExpMtpDraftModel(MtpDraftBase):
    """Qwen4-Exp MTP draft for quantization, built from the HF text classes.

    The draft consumes the base's last-layer hyper-connection bus
    (``hc_count * hidden_size`` wide, taken before the base's final mixer)
    plus the token embedding written to every stream; its one decoder layer
    is a base layer. Module names match the checkpoint's ``mtp.*`` keys.
    """

    def __init__(self, config):
        super().__init__()
        hf = _qwen4_exp_modeling()
        self.config = config
        hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        eps = config.rms_norm_eps
        self.pre_fc_norm_embedding = hf.Qwen4ExpTextRMSNorm(hidden_size,
                                                            eps=eps)
        self.pre_fc_norm_hidden = hf.Qwen4ExpTextRMSNorm(hidden_size *
                                                         self.hc_count,
                                                         eps=eps)
        self.fc_embedding = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [hf.Qwen4ExpTextDecoderLayer(_draft_layer_config(config), 0)])
        self.hyper_connection_mixer = hf.Qwen4ExpTextGatedResidual(
            config, use_combine=False)
        self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)

        # Shared with base model — set by calibration code
        self.embed_tokens: Optional[nn.Module] = None

        mlp = self.layers[0].mlp
        if isinstance(mlp, hf.Qwen4ExpTextSparseMoeBlock):
            # export_quantized_mtp_state_dict exports QuantLinear modules only,
            # so HF's stacked [E, 2I, H] / [E, H, I] experts become per-expert Linears.
            mlp.experts = _ExpertBank(
                hf.Qwen4ExpTextMLP(
                    config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.num_experts))

    @property
    def is_moe(self) -> bool:
        return isinstance(self.layers[0].mlp,
                          _qwen4_exp_modeling().Qwen4ExpTextSparseMoeBlock)

    def forward(self, input_ids, hidden_bus, position_embeddings,
                attention_mask):
        """Calibration forward on the base's bus for the same positions."""
        assert self.embed_tokens is not None, \
            "embed_tokens must be set (shared from base model) before forward"
        hidden_size = self.config.hidden_size
        embeds = self.fc_embedding(
            self.pre_fc_norm_embedding(self.embed_tokens(input_ids)))
        hyper_states = self.pre_fc_norm_hidden(hidden_bus)
        # One fc_hidden shared by every stream, applied stream by stream.
        hyper_states = self.fc_hidden(hyper_states.reshape(
            -1, hidden_size)).reshape(hidden_bus.shape)
        hyper_states = hyper_states + embeds.repeat(1, 1, self.hc_count)
        hyper_states = self.layers[0](hyper_states,
                                      position_embeddings=position_embeddings,
                                      attention_mask=attention_mask)
        return self.lm_head(self.hyper_connection_mixer(hyper_states))

    def calibration_forward(self, base_model, input_ids):
        """Run the base, then the draft on the base's pre-mixer bus."""
        text_model = _qwen4_exp_text_model(base_model)
        captured: Dict[str, object] = {}
        handles = [
            text_model.layers[-1].register_forward_pre_hook(
                lambda _m, _args, kwargs: captured.update(kwargs),
                with_kwargs=True),
            text_model.hyper_connection_mixer.register_forward_pre_hook(
                lambda _m, args: captured.update(bus=args[0])),
        ]
        try:
            base_model(input_ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return self(input_ids, captured["bus"],
                    captured["position_embeddings"],
                    captured["attention_mask"])
