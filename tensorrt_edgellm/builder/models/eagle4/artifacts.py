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
"""EAGLE-4 draft runtime artifact writing."""

from ...core.artifacts.runtime_artifacts import write_runtime_artifacts
from ..eagle3 import weights
from . import runtime_config


def preflight_bundle(bundle, args, engine_dir: str) -> None:
    """Configure the draft before any engine build so its rejections run first."""
    del bundle, engine_dir
    from ...core.builder import load_device_config
    load_device_config(args)


def write_artifacts(bundle, config, args, engine_dir: str) -> None:
    del bundle
    if config is None:
        raise ValueError("EAGLE-4 artifacts require an LLM configuration")
    write_runtime_artifacts(config,
                            args,
                            engine_dir,
                            weight_conversion=weights,
                            runtime_config_module=runtime_config)
