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
"""
Qwen3.5 MoE hybrid causal LM.

This model combines the Qwen3.5 mixer stack (GatedDeltaNet linear attention
and gated full attention) with a sparse MoE feed-forward block in every layer.
The routed experts reuse the Qwen3 INT4 GPTQ MoE plugin path; Qwen3.5 adds an
FP16 shared expert gated by ``mlp.shared_expert_gate``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import GdnConfig, ModelConfig
from ..linear import make_linear
from ..qwen3_5.modeling_qwen3_5_text import (Qwen3_5Backbone, Qwen3_5CausalLM,
                                             Qwen3_5DecoderLayer,
                                             Qwen3_5RMSNorm)
from ..qwen3_moe.modeling_qwen3_moe import Qwen3SparseMoeBlock

__all__ = ["Qwen3_5MoeCausalLM"]


class Qwen3_5SharedExpert(nn.Module):
    """FP16 shared expert used alongside routed Qwen3.5 MoE experts."""

    def __init__(self, config: ModelConfig, module_prefix: str) -> None:
        super().__init__()
        hidden = config.hidden_size
        inter = (config.moe_shared_expert_intermediate_size
                 or config.moe_intermediate_size)
        self.gate_proj = make_linear(config,
                                     hidden,
                                     inter,
                                     module_name=f"{module_prefix}.gate_proj")
        self.up_proj = make_linear(config,
                                   hidden,
                                   inter,
                                   module_name=f"{module_prefix}.up_proj")
        self.down_proj = make_linear(config,
                                     inter,
                                     hidden,
                                     module_name=f"{module_prefix}.down_proj")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) *
            self.up_proj(hidden_states))


class Qwen3_5SparseMoeBlock(Qwen3SparseMoeBlock):
    """Qwen3.5 sparse MoE block with one FP16 shared expert."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__(config)
        prefix = f"layers.{layer_idx}.mlp"
        self.shared_expert = Qwen3_5SharedExpert(config,
                                                 f"{prefix}.shared_expert")
        self.shared_expert_gate = make_linear(
            config,
            config.hidden_size,
            1,
            module_name=f"{prefix}.shared_expert_gate")



class Qwen3_5MoeDecoderLayer(Qwen3_5DecoderLayer):
    """Qwen3.5 decoder layer with GDN/full-attention mixer and MoE FFN."""

    def __init__(self, config: ModelConfig, gc: GdnConfig, layer_idx: int,
                 layer_type: str) -> None:
        super().__init__(config, gc, layer_idx, layer_type)
        self.mlp = Qwen3_5SparseMoeBlock(config, layer_idx)


class Qwen3_5MoeBackbone(Qwen3_5Backbone):
    """Qwen3.5 hybrid decoder backbone with sparse MoE FFNs."""

    def __init__(self, config: ModelConfig) -> None:
        nn.Module.__init__(self)
        gc = config.gdn_cfg
        assert gc is not None, "Qwen3.5 MoE requires gdn_cfg"
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3_5MoeDecoderLayer(config, gc, layer_idx=i, layer_type=lt)
            for i, lt in enumerate(config.layer_types)
        ])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layer_types = config.layer_types


class Qwen3_5MoeCausalLM(Qwen3_5CausalLM):
    """Qwen3.5 MoE causal LM: hybrid backbone + lm_head."""

    def __init__(self, config: ModelConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.model = Qwen3_5MoeBackbone(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")
