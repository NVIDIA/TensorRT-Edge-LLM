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
"""Behavioral contracts for the abstract Orbax reader."""

import sys
from dataclasses import dataclass
from types import ModuleType

import pytest

from experimental.checkpoint_converters.schema import CheckpointRestoreError
from experimental.checkpoint_converters.source import OrbaxCheckpointReader


@dataclass(frozen=True)
class _Reader(OrbaxCheckpointReader):
    result: object | None = None
    error: Exception | None = None

    def _restore_parameters(self, checkpoint_path, restore_concurrent_gb):
        if self.error is not None:
            raise self.error
        return self.result


def test_reader_freezes_family_restore_result(monkeypatch, tmp_path):
    """The template freezes the family result before returning it."""
    params = {"leaf": object()}
    frozen = object()
    flax_core = ModuleType("flax.core")
    flax_core.freeze = lambda value: frozen if value is params else None
    monkeypatch.setitem(sys.modules, "flax.core", flax_core)

    assert _Reader("revision", "family", result=params).restore(tmp_path,
                                                                7) is frozen


def test_reader_wraps_upstream_error_with_family_context(
        monkeypatch, tmp_path):
    """Restore failures preserve their cause and family revision context."""
    upstream = OSError("orbax failed")
    flax_core = ModuleType("flax.core")
    flax_core.freeze = lambda value: value
    monkeypatch.setitem(sys.modules, "flax.core", flax_core)

    with pytest.raises(CheckpointRestoreError,
                       match=rf"{tmp_path}.*family.*revision") as raised:
        _Reader("revision", "family", error=upstream).restore(tmp_path, 1)

    assert raised.value.__cause__ is upstream
