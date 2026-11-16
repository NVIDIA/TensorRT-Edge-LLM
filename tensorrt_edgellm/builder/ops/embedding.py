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
"""Checkpoint-backed embedding module."""

import numpy as np

from ..core.weight_policy import CHECKPOINT_BINDING_ROLE_EMBEDDING
from . import functional as F
from .module import BuildContext, Module


class Embedding(Module):
    """PyTorch-style embedding backed by one static checkpoint parameter."""

    def __init__(self,
                 ctx: BuildContext,
                 prefix: str,
                 *,
                 scale: float = 1.0,
                 runtime_embedding: bool = False) -> None:
        super().__init__(ctx, prefix)
        self.scale = float(scale)
        self.runtime_embedding = runtime_embedding
        self._weight = None

    @property
    def weight(self):
        """Return the scaled table shared by embedding and tied-head use."""
        if self._weight is not None:
            return self._weight
        name = self.key("weight")
        value = self.weights.parameter_value(
            "embedding",
            name,
            lambda: self.weights.parameter_spec(name),
            lambda: np.ascontiguousarray(
                self.weights.f16(name) * np.float16(self.scale)),
        )
        extra = {"embedding_scale": float(np.float16(self.scale))}
        if self.runtime_embedding:
            extra["role"] = CHECKPOINT_BINDING_ROLE_EMBEDDING
        recipe = self.weights.checkpoint_binding([name], "fp16", **extra)
        self._weight = F.parameter(name, value, "embedding", recipe=recipe)
        return self._weight

    def forward(self, input_ids):
        return F.embedding_lookup(self.weight, input_ids)
