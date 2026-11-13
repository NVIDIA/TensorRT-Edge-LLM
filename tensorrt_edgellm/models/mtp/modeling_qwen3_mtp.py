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
"""MTP draft head for dense Qwen3 checkpoints (AngelSlim/Qwen3-8B-MTP).

The head keeps the Qwen3-Next tensor layout (``mtp.pre_fc_norm_*``, ``mtp.fc``,
one decoder layer, ``mtp.norm``, shared lm_head), so the forward and ONNX
contract come from :class:`Qwen3_5MtpDraftModel`. Only the layer types differ:
a plain Qwen3 decoder layer (QK-norm attention without the output gate) and
Llama-style RMSNorm (``weight * rms(x)``).
"""

import torch.nn as nn

from ...config import ModelConfig
from ..default.modeling_default import DecoderLayer, RMSNorm
from ..linear import make_linear
from ..qwen3_5.modeling_qwen3_5_mtp import Qwen3_5MtpDraftModel

__all__ = ["Qwen3MtpDraftModel"]


class Qwen3MtpDraftModel(Qwen3_5MtpDraftModel):
    """Dense Qwen3 MTP draft on the default decoder layer."""

    def __init__(self, config: ModelConfig) -> None:
        # Build the modules directly instead of calling the Qwen3.5 constructor,
        # which would allocate gated-attention projections only to discard them.
        nn.Module.__init__(self)
        if config.num_hidden_layers != 1:
            raise ValueError("Qwen3 MTP draft currently requires "
                             "num_hidden_layers == 1")
        self.config = config
        hidden_size = config.hidden_size
        self.pre_fc_norm_embedding = RMSNorm(hidden_size, config.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(hidden_size, config.rms_norm_eps)
        self.fc = make_linear(config,
                              hidden_size * 2,
                              hidden_size,
                              bias=False,
                              module_name="fc")
        self.layers = nn.ModuleList([self._make_decoder_layer(config)])
        self.norm = RMSNorm(hidden_size, config.rms_norm_eps)
        self.lm_head = make_linear(config,
                                   hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    def _make_decoder_layer(self, config: ModelConfig) -> nn.Module:
        return DecoderLayer(config, layer_idx=0)
