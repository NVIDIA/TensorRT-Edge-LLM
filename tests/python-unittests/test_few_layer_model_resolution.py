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

import os
from types import SimpleNamespace

from defs import config, test_few_layer_validation


def test_model_resolution_checks_shallow_candidates_across_roots_first(
        tmp_path, monkeypatch):
    torch_root = tmp_path / "torch-models"
    data_root = tmp_path / "edge-cache"
    model_dir = data_root / "models" / "NVIDIA-Nemotron"
    torch_root.mkdir()
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").touch()
    (model_dir / "model.safetensors").touch()

    real_listdir = os.listdir

    def reject_expensive_torch_tree_scan(path):
        if os.fspath(path) == os.fspath(torch_root):
            raise AssertionError("searched the first root recursively")
        return real_listdir(path)

    monkeypatch.setattr(config.os, "listdir", reject_expensive_torch_tree_scan)
    env_config = SimpleNamespace(llm_models_dir=str(torch_root),
                                 edgellm_data_dir=str(data_root))

    resolved = test_few_layer_validation._resolve_model_dir(
        env_config, "NVIDIA-Nemotron")

    assert resolved == str(model_dir)
