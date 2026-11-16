#
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
#
"""Dense Qwen3 MTP draft for the checkpoint-direct builder.

Third-party heads such as AngelSlim/Qwen3-8B-MTP use the Qwen3-Next tensor
layout (``mtp.pre_fc_norm_*``, ``mtp.fc``, ``mtp.layers.0``, ``mtp.norm``) over
plain Qwen3 blocks: ungated attention and Llama-style RMSNorm.
"""

from ...ops import Linear, NetworkModule, RMSNorm
from ..qwen3_5.modeling_qwen3_5_mtp import Qwen35MtpDraftModel
from .modeling_qwen3 import Qwen3DecoderLayer


class Qwen3MtpDraftModel(Qwen35MtpDraftModel):
    """Qwen3-Next MTP contract built from dense Qwen3 decoder blocks."""

    def __init__(self, ctx) -> None:
        NetworkModule.__init__(self, ctx)
        eps = ctx.cfg.rms_norm_eps
        self.pre_embed_norm = RMSNorm(ctx, "pre_fc_norm_embedding", eps)
        self.pre_hidden_norm = RMSNorm(ctx, "pre_fc_norm_hidden", eps)
        self.fc = Linear(ctx, "fc")
        self.layers = [
            Qwen3DecoderLayer(ctx, f"layers.{index}", index)
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = RMSNorm(ctx, "norm", eps)
        self.lm_head = Linear(ctx,
                              ctx.weights.causal_lm_head_prefix("mtp.lm_head"))
