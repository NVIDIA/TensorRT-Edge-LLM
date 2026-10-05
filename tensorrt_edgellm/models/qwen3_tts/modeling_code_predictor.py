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
Qwen3-TTS CodePredictor — residual codec token predictor.

The CodePredictor is a small 5-layer Qwen3 decoder that generates residual
audio codes to complement the coarse code from the Talker.  It has 15 separate
lm_heads (one per residual codebook layer) and 15 codec embedding tables.

Key differences from a standard CausalLM:
- ``lm_heads`` (all heads stacked) and ``lm_head_idx`` are ONNX **inputs**; the
  head is gathered inside the graph, so one CUDA graph serves every step.
- The forward pass returns both ``logits`` and ``hidden_states`` (for residual
  connection in the multi-token prediction loop).
- ``embed_tokens`` is a ModuleList of 15 codec embeddings (embedding lookup
  happens at runtime, not in the ONNX graph — the model takes ``inputs_embeds``).

Extra weight files extracted alongside the ONNX:
- ``codec_embeddings.safetensors`` — 15 embedding tables [codebookSize, hiddenSize]
- ``lm_heads.safetensors`` — 15 lm_head weights [codebookSize, hiddenSize]
- ``small_to_mtp_projection.safetensors`` — Linear [cpHiddenSize, talkerHiddenSize]

Checkpoint prefix: ``talker.code_predictor.*``
"""

import dataclasses
import inspect
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..default.modeling_default import CausalLM, OnnxSpec
from ..linear import FP16Linear, TPMode, make_linear

logger = logging.getLogger(__name__)

__all__ = ["CodePredictorCausalLM"]


class CodePredictorMLP(nn.Module):
    """SwiGLU MLP with an FP32 ``silu(gate) * up`` precision guard.

    CP's intermediate activations reach [-39.5, 72.4] where FP16 spacing
    is 0.0625; an FP16 multiply loses ~6 mantissa bits per layer and
    compounds across the 5-layer stack, shifting codec-EOS prediction
    enough to regress TTS WER by ~20% absolute.  Running ``silu*up`` in
    FP32 and keeping the intermediate FP32 through the down_proj matmul
    (via a Cast on the weight, constant-folded by TRT) preserves
    precision end-to-end.

    Every CP recipe excludes ``down_proj`` (see ``_CP_LINEAR_EXCLUDES``) so
    the down_proj Linear is always FP16Linear whatever the rest of the CP
    runs at — this same forward path is then safe in fp16, fp8 and nvfp4.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        prefix = f"layers.{layer_idx}.mlp"
        self.gate_proj = make_linear(config,
                                     config.hidden_size,
                                     config.intermediate_size,
                                     module_name=f"{prefix}.gate_proj",
                                     tp_mode=TPMode.COL)
        self.up_proj = make_linear(config,
                                   config.hidden_size,
                                   config.intermediate_size,
                                   module_name=f"{prefix}.up_proj",
                                   tp_mode=TPMode.COL)
        self.down_proj = make_linear(config,
                                     config.intermediate_size,
                                     config.hidden_size,
                                     module_name=f"{prefix}.down_proj",
                                     tp_mode=TPMode.ROW)
        # forward() reads down_proj.weight as a plain float tensor; a quantized
        # class keeps a packed buffer there and would cast to garbage silently.
        if not isinstance(self.down_proj, FP16Linear):
            raise ValueError(
                f"CodePredictor {prefix}.down_proj resolved to "
                f"{type(self.down_proj).__name__}; the FP32 matmul needs an "
                "unquantized weight. Keep down_proj excluded from CP quant.")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out_dtype = hidden_states.dtype
        gate_output = self.gate_proj(hidden_states).to(torch.float32)
        up_output = self.up_proj(hidden_states).to(torch.float32)
        intermediate = F.silu(gate_output) * up_output
        w32 = self.down_proj.weight.to(torch.float32)
        return F.linear(intermediate, w32).to(out_dtype)


# ---------------------------------------------------------------------------
# CodePredictor flat wrapper for ONNX export
# ---------------------------------------------------------------------------


def _make_code_predictor_flat_wrapper(base: nn.Module) -> nn.Module:
    """Build a flat-signature wrapper for CodePredictor ONNX export.

    Unlike the standard CausalLM wrapper, this includes:
    - ``lm_heads`` + ``lm_head_idx`` as inputs (head gathered in-graph)
    - ``hidden_states`` as an output (for residual connection)
    """
    signature = inspect.signature(base.forward)
    base_names = [
        name for name, parameter in signature.parameters.items()
        if parameter.kind is parameter.POSITIONAL_OR_KEYWORD
    ]
    names = base_names + ["lm_heads", "lm_head_idx"]
    base_call = ", ".join(base_names)
    src = (
        f"def _forward(self, {', '.join(names)}):\n"
        f"    out = self._base({base_call})\n"
        f"    full_hidden = out[1]\n"
        f"    selected_hidden = torch.index_select(full_hidden, 0, "
        f"logits_indices)\n"
        f"    head = lm_heads.index_select(0, "
        f"lm_head_idx.to(torch.long)).squeeze(0)\n"
        f"    logits = torch.matmul(selected_hidden, head.T).to(torch.float32)\n"
        f"    return (logits, full_hidden) + tuple(out[2:])\n")
    globs: dict = {"torch": torch}
    exec(src, globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, b: nn.Module) -> None:
            super().__init__()
            self._base = b

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(base)


# ---------------------------------------------------------------------------
# CodePredictorCausalLM
# ---------------------------------------------------------------------------


class CodePredictorCausalLM(CausalLM):
    """CP CausalLM: stacked ``lm_heads`` + device ``lm_head_idx`` inputs
    (head gathered in-graph) + ``hidden_states`` output for the residual loop.
    """

    match_fp32_matmul_initializers = True
    emit_hidden_states = True

    def _ragged_emitted_hidden(self) -> torch.Tensor:
        hidden_states = self.model.norm(self.model.last_pre_norm_hidden_states)
        return hidden_states.reshape(-1, hidden_states.shape[-1])

    def __init__(self, config) -> None:
        super().__init__(config)
        for layer_idx, layer in enumerate(self.model.layers):
            layer.mlp = CodePredictorMLP(config, layer_idx)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        last_token_ids: torch.Tensor,
        lm_heads: torch.Tensor,
        lm_head_idx: torch.Tensor,
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
        # Gather the head by device index: logits = last_hidden @ lm_heads[idx].T
        head = lm_heads.index_select(0, lm_head_idx.to(torch.long)).squeeze(0)
        logits = torch.matmul(last_hidden, head.T).to(torch.float32)
        return logits, hidden_states, present_key_values

    def onnx_export_spec(self) -> OnnxSpec:
        """ONNX export spec with lm_heads/lm_head_idx inputs and hidden_states output."""
        spec = super().onnx_export_spec()
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
        return dataclasses.replace(
            spec,
            wrapped=_make_code_predictor_flat_wrapper(spec.wrapped),
            args=spec.args + (lm_heads, lm_head_idx),
            input_names=list(spec.input_names) + ["lm_heads", "lm_head_idx"],
            dynamic_shapes=list(spec.dynamic_shapes) + [{}, {}],
        )
