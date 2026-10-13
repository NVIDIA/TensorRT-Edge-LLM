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

import json
from types import SimpleNamespace

import torch
from safetensors import safe_open

from tensorrt_edgellm.scripts import export as export_script
from tensorrt_edgellm.vocab_reduction import constants


def test_write_draft_vocab_sidecar_schema(tmp_path):
    vocab_map = torch.tensor([17, 3, 101, 42], dtype=torch.int64)
    model = SimpleNamespace(_reduced_vocab_map_for_runtime=vocab_map)

    export_script._write_draft_vocab_sidecar(
        model,
        str(tmp_path),
        full_size=128,
        reduced_size=vocab_map.numel(),
        draft_reduced_vocab_dir="/maps/reduced-vocab",
        log_tag="[Test Draft]",
    )

    map_path = tmp_path / constants.DRAFT_VOCAB_MAP_NAME
    with safe_open(map_path, framework="pt", device="cpu") as sidecar:
        assert list(sidecar.keys()) == ["vocab_map"]
        saved_map = sidecar.get_tensor("vocab_map")

    assert saved_map.dtype == torch.int32
    assert saved_map.numel() == vocab_map.numel()

    with open(tmp_path / constants.DRAFT_VOCAB_INFO_NAME) as info_file:
        info = json.load(info_file)
    assert info == {
        "vocab_size": 128,
        "reduced_vocab_size": vocab_map.numel(),
        "source": "/maps/reduced-vocab",
    }
