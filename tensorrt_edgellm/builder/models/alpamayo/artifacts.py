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
"""Alpamayo runtime artifact writing."""

from ...core.artifacts.runtime_artifacts import (write_component_artifacts,
                                                 write_runtime_artifacts)
from . import preprocessing, runtime_config, tokenizer, weights


def write_artifacts(bundle, config, args, engine_dir: str) -> None:
    if config is not None:
        write_runtime_artifacts(config,
                                args,
                                engine_dir,
                                weight_conversion=weights,
                                runtime_config_module=runtime_config,
                                tokenizer_module=tokenizer)
        return
    write_component_artifacts(bundle,
                              args,
                              engine_dir,
                              runtime_config_module=runtime_config,
                              preprocessing_module=preprocessing)
