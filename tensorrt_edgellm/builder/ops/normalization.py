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
"""Checkpoint-backed normalization modules shared by model families."""

from typing import Optional

import numpy as np

from . import functional as F
from .module import BuildContext, Module


class RMSNorm(Module):
    """Apply RMSNorm using ``{prefix}.weight`` from the checkpoint."""

    def __init__(self,
                 ctx: BuildContext,
                 prefix: str,
                 eps: Optional[float] = None,
                 rank: Optional[int] = None,
                 *,
                 unit_offset: bool = False) -> None:
        super().__init__(ctx, prefix)
        self.eps = ctx.cfg.rms_norm_eps if eps is None else eps
        self.rank = rank
        self.unit_offset = unit_offset

    def forward(self, hidden_states, rank: Optional[int] = None):
        weight = (self.weights.fp16_parameter(self.key("weight"))
                  if self.ctx.backend == "edgellm" and not self.unit_offset
                  else self.weights.f16(self.key("weight")))
        if self.unit_offset:
            weight = weight + np.float16(1.0)
        return F.rms_norm(
            hidden_states,
            weight,
            self.eps,
            (self.rank if self.rank is not None else hidden_states.ndim)
            if rank is None else rank,
            weight_before_cast=self.unit_offset)


class LayerNorm(Module):
    """Apply LayerNorm using checkpoint weight and optional bias tensors."""

    def __init__(self,
                 ctx: BuildContext,
                 prefix: str,
                 eps: float,
                 rank: Optional[int] = None) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps
        self.rank = rank

    def forward(self, hidden_states, rank: Optional[int] = None):
        resolved_rank = ((self.rank if self.rank is not None else
                          hidden_states.ndim) if rank is None else rank)
        return F.normalization(hidden_states, self.prefix, self.eps,
                               resolved_rank)
