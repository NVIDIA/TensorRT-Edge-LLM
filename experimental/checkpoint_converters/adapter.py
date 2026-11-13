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
"""Family-agnostic checkpoint adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .transforms import MappingPlan
from .variant import CheckpointVariant


class CheckpointAdapter(ABC):
    """Abstract interface implemented by checkpoint family adapters."""

    variant: CheckpointVariant

    @abstractmethod
    def plan(self, source_tree: Any) -> MappingPlan:
        """Build a complete, validated mapping plan.

        Args:
            source_tree: Canonical nested source parameter tree.

        Returns:
            Immutable source-to-target mapping plan.
        """

    @abstractmethod
    def build_config(self) -> dict[str, Any]:
        """Build a validated target configuration.

        Returns:
            JSON-compatible Hugging Face configuration.
        """
