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
"""Gemma4 Orbax checkpoint restoration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...source import OrbaxCheckpointReader
from .schema import GEMMA_REVISION


@dataclass(frozen=True)
class Gemma4OrbaxCheckpointReader(OrbaxCheckpointReader):
    """Restore canonical text-only parameters through the Gemma API."""

    revision: str = GEMMA_REVISION
    family: str = "Gemma4"

    def _restore_parameters(self, checkpoint_path: Path,
                            restore_concurrent_gb: int) -> Any:
        """Delegate supported Orbax layouts to the pinned Gemma loader."""
        from gemma import gm

        return gm.ckpts.load_params(
            checkpoint_path,
            text_only=True,
            restore_concurrent_gb=restore_concurrent_gb,
        )
