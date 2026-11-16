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
"""Qwen3-TTS runtime artifact writing."""

import json
import os

from ...core import contracts
from ...core.artifacts.runtime_artifacts import (write_component_artifacts,
                                                 write_runtime_artifacts)
from . import embeddings, runtime_config, weights


def write_artifacts(bundle, config, args, engine_dir: str) -> None:
    if config is not None:
        write_runtime_artifacts(config,
                                args,
                                engine_dir,
                                weight_conversion=weights,
                                runtime_config_module=runtime_config,
                                embedding_module=embeddings)
        return
    if args.resolved_component in (
            contracts.Component.SPEAKER_ENCODER,
            contracts.Component.SPEECH_TOKENIZER_ENCODER):
        output_dir = contracts.component_spec(
            args.resolved_component).output_dir(engine_dir)
        os.makedirs(output_dir, exist_ok=True)
        name = args.resolved_component.value.replace("-", "_") + "_config.json"
        payload = runtime_config.component_runtime_config(
            bundle, args.resolved_component, args)
        with open(os.path.join(output_dir, name), "w") as config_file:
            json.dump(payload, config_file, indent=2)
        return
    write_component_artifacts(bundle,
                              args,
                              engine_dir,
                              runtime_config_module=runtime_config,
                              embedding_module=embeddings)
