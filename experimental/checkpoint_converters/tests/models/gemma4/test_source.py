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
"""Behavioral contracts for Gemma4 Orbax restoration."""

import sys
from types import ModuleType, SimpleNamespace

from experimental.checkpoint_converters.models.gemma4.source import \
    Gemma4OrbaxCheckpointReader


def test_restore_uses_text_only_gemma_api(monkeypatch, tmp_path):
    """Gemma4 restoration delegates canonicalization to the text-only API."""
    calls = []
    params = object()

    def load_params(path, *, text_only, restore_concurrent_gb):
        calls.append((path, text_only, restore_concurrent_gb))
        return params

    gemma = ModuleType("gemma")
    gemma.gm = SimpleNamespace(ckpts=SimpleNamespace(load_params=load_params))
    monkeypatch.setitem(sys.modules, "gemma", gemma)

    assert Gemma4OrbaxCheckpointReader()._restore_parameters(tmp_path,
                                                             7) is params
    assert calls == [(tmp_path, True, 7)]
