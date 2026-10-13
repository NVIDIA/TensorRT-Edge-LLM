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
"""Checkpoint-backed primitives used by Gemma4 model modules."""

from ...ops import BuildContext, Linear, Module, Tensor
from ...ops import functional as F

__all__ = [
    "BuildContext",
    "Linear",
    "Gemma4RMSNormBase",
    "Module",
]


class Gemma4RMSNormBase(Module):
    """Decomposed RMSNorm; weight at ``{prefix}.weight``."""

    def __init__(self, ctx: BuildContext, prefix: str, eps: float) -> None:
        super().__init__(ctx, prefix)
        self.eps = eps

    def forward(self, x: Tensor, rank: int = 2) -> Tensor:
        weight = self.weights.f32(self.key("weight"))
        return F.rms_norm(x,
                          weight,
                          self.eps,
                          rank=rank,
                          weight_before_cast=True)
