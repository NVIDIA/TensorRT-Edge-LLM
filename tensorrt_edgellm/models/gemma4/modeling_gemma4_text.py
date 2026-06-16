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
Gemma 4 (``gemma4_text``) causal LM — checkpoint export frontend.

This is the first phase of Gemma 4 support (issue
https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/72).  It teaches the frontend
to *recognize* a Gemma 4 text checkpoint and parse its architecture into
:class:`~tensorrt_edgellm.config.Gemma4Config`, but ONNX export and C++ runtime
support are not implemented yet.

Why a dedicated (currently fail-loud) class instead of the dense
:class:`~tensorrt_edgellm.models.default.modeling_default.CausalLM`?  Gemma 4
departs from the Llama-style dense template in ways that produce silently wrong
results — not just errors — if exported through the default path:

* Per-layer-type attention: local *sliding* layers use ``head_dim`` (256) while
  global *full* layers use ``global_head_dim`` (512).
* Dual RoPE: local layers use a plain RoPE (``theta=10000``); global layers use
  a *proportional* RoPE (``theta=1e6``) over only ``partial_rotary_factor`` of
  the head.
* Per-Layer Embeddings (PLE): a second embedding pathway feeds a residual signal
  into every decoder layer; the C++ runtime would have to compute and thread it
  into the engine alongside ``inputs_embeds``.
* KV sharing: the trailing ``num_kv_shared_layers`` layers reuse a previous
  layer's keys/values.
* Soft-capped final logits, GeGLU (``gelu_pytorch_tanh``) MLP with a double-wide
  variant on the KV-shared layers, and ``(query) scaling = 1.0`` rather than the
  usual ``1/sqrt(head_dim)``.

Several of those live in the C++ runtime / CUDA kernels (per-layer head dim,
dual RoPE, KV sharing, attention scaling, PLE input plumbing) and land in
follow-up phases.  Registering this class makes a Gemma 4 checkpoint fail with
an explicit, actionable message instead of being mis-built as a Llama-style
model with the wrong head dimensions, RoPE, and normalization.
"""

import torch.nn as nn

from ...config import ModelConfig

__all__ = ["Gemma4CausalLM"]

_TRACKING_ISSUE = "https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/72"


def _detected_summary(config: ModelConfig) -> str:
    """One-line summary of the parsed Gemma 4 architecture, for the error text."""
    cfg = config.gemma4_cfg
    if cfg is None:
        return f"model_type={config.model_type!r}, {config.num_hidden_layers} layers"
    return (
        f"{config.num_hidden_layers} layers "
        f"({cfg.num_local_layers} local sliding + {cfg.num_global_layers} global full), "
        f"head_dim local={cfg.local_head_dim}/global={cfg.global_head_dim}, "
        f"RoPE theta local={cfg.sliding_rope_theta:g}/global={cfg.global_rope_theta:g}, "
        f"PLE dim={cfg.hidden_size_per_layer_input}, "
        f"kv_shared_layers={cfg.num_kv_shared_layers}, "
        f"final_logit_softcapping={cfg.final_logit_softcapping}")


def _unsupported_message(config: ModelConfig) -> str:
    """Build the actionable NotImplementedError message for Gemma 4 export."""
    return (
        "Gemma 4 (gemma4_text) export is not implemented yet.\n"
        f"Detected: {_detected_summary(config)}.\n"
        "The frontend can parse this checkpoint, but ONNX export and the C++ "
        "runtime still need: per-layer-type head dimensions, dual/proportional "
        "RoPE, Per-Layer-Embedding (PLE) input plumbing, KV sharing across the "
        "trailing layers, unit query scaling, and final-logit soft-capping.\n"
        f"Track progress at {_TRACKING_ISSUE}.")


class Gemma4CausalLM(nn.Module):
    """Gemma 4 causal LM placeholder that fails loudly until export lands.

    Constructing the model raises :class:`NotImplementedError` with a summary of
    the parsed architecture and the remaining work.  See the module docstring
    for why Gemma 4 cannot reuse the dense export path.

    Args:
        config: Parsed model configuration; expected to carry a populated
            :attr:`~tensorrt_edgellm.config.ModelConfig.gemma4_cfg`.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError(_unsupported_message(config))
