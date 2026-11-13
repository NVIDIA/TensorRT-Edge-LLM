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
"""Transformer feed-forward modules built from the unified operation API."""

from typing import Optional

import tensorrt as trt

from .linear import Linear
from .module import BuildContext, Module


class GatedMLP(Module):
    """Transformers-style ``down(act(gate(x)) * up(x))`` feed-forward."""

    def __init__(self,
                 ctx: BuildContext,
                 prefix: str,
                 activation: Optional[str] = None) -> None:
        super().__init__(ctx, prefix)
        self.activation = activation or ctx.cfg.hidden_act
        self.gate_proj = Linear(ctx, self.key("gate_proj"))
        self.up_proj = Linear(ctx, self.key("up_proj"))
        self.down_proj = Linear(ctx, self.key("down_proj"))

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states).activation(self.activation)
        return self.down_proj(gate * self.up_proj(hidden_states))


class FP32GatedMLP(GatedMLP):
    """SwiGLU with an FP32 product and FP32 down-projection accumulation."""

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states).cast(trt.float32).activation(
            self.activation)
        up = self.up_proj(hidden_states).cast(trt.float32)
        return self.down_proj.forward_f32(gate * up).cast(hidden_states.dtype)
