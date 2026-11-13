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
"""Abstract Orbax checkpoint restoration boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import CheckpointRestoreError


@dataclass(frozen=True)
class OrbaxCheckpointReader(ABC):
    """Restore and freeze a family checkpoint through an upstream API."""

    revision: str
    family: str

    def restore(self, checkpoint_path: Path,
                restore_concurrent_gb: int) -> Any:
        """Restore one checkpoint and preserve family context on failure."""
        try:
            parameters = self._restore_parameters(checkpoint_path,
                                                  restore_concurrent_gb)
        except Exception as error:
            raise CheckpointRestoreError(
                f"failed to restore {checkpoint_path} with {self.family} "
                f"revision {self.revision}") from error
        if not isinstance(parameters, Mapping):
            raise CheckpointRestoreError(
                f"failed to restore {checkpoint_path} with {self.family} "
                f"revision {self.revision}: expected a mapping root, found "
                f"{type(parameters).__name__}")
        from flax.core import freeze

        return freeze(parameters)

    @abstractmethod
    def _restore_parameters(self, checkpoint_path: Path,
                            restore_concurrent_gb: int) -> Any:
        """Restore the family parameter tree before immutable wrapping."""
