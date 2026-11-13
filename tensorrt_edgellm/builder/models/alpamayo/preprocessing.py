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
"""Alpamayo preprocessing artifacts."""

import os
import shutil

from ...core import contracts
from .tokenizer import vlm_file


def write_component_assets(bundle, component: contracts.Component, args,
                           output_dir: str) -> None:
    """Copy Alpamayo VLM preprocessing assets for visual runtime."""
    if component != contracts.Component.VISUAL:
        return
    preprocessor = os.path.join(output_dir, "preprocessor_config.json")
    if not os.path.isfile(preprocessor):
        shutil.copy2(
            vlm_file(bundle.root, bundle.model_dir,
                     "preprocessor_config.json"), preprocessor)
