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
"""Runtime artifact writer for Nemotron-H Puzzle."""

from ...core import contracts
from ...core.artifacts.runtime_artifacts import write_runtime_artifacts
from ..nemotron_omni import artifacts as omni_artifacts
from . import weights


def write_artifacts(bundle, config, args, engine_dir: str) -> None:
    if args.resolved_component != contracts.Component.LLM:
        omni_artifacts.write_artifacts(bundle, config, args, engine_dir)
        return
    write_runtime_artifacts(config,
                            args,
                            engine_dir,
                            weight_conversion=weights)
