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
"""Nemotron-3.5-ASR runtime artifact writing."""

import filecmp
import json
import os
import shutil

from ...core import contracts

_RUNTIME_METADATA = (
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "feature_extractor_config.json",
)


def _root_artifacts_match(bundle, engine_dir: str) -> bool:
    try:
        with open(os.path.join(engine_dir, "config.json"),
                  encoding="utf-8") as config_file:
            if json.load(config_file) != bundle.root:
                return False
    except (OSError, ValueError):
        return False

    for filename in _RUNTIME_METADATA:
        source = os.path.join(bundle.model_dir, filename)
        destination = os.path.join(engine_dir, filename)
        if os.path.isfile(source) != os.path.isfile(destination):
            return False
        if os.path.isfile(source) and not filecmp.cmp(
                source, destination, shallow=False):
            return False
    return True


def _write_root_artifacts(bundle, engine_dir: str) -> None:
    with open(os.path.join(engine_dir, "config.json"), "w",
              encoding="utf-8") as config_file:
        json.dump(bundle.root, config_file, indent=2)
    for filename in _RUNTIME_METADATA:
        source = os.path.join(bundle.model_dir, filename)
        destination = os.path.join(engine_dir, filename)
        if os.path.isfile(source):
            shutil.copy2(source, destination)
        elif os.path.isfile(destination):
            os.remove(destination)


def write_artifacts(bundle, config, args, engine_dir: str) -> None:
    del config
    required = ("tokenizer.json", "processor_config.json")
    missing = [
        filename for filename in required
        if not os.path.isfile(os.path.join(bundle.model_dir, filename))
    ]
    if missing:
        raise FileNotFoundError(
            "Nemotron-3.5-ASR checkpoint is missing required runtime "
            f"artifacts: {', '.join(missing)}")

    output_dir = contracts.component_spec(
        args.resolved_component).output_dir(engine_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(engine_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w",
              encoding="utf-8") as config_file:
        json.dump(bundle.root, config_file, indent=2)

    if not _root_artifacts_match(bundle, engine_dir):
        _write_root_artifacts(bundle, engine_dir)
