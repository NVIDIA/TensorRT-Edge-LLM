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
"""Import names for optional extras; installation requirements live in pyproject.toml."""

EXPORT_DEPENDENCIES = frozenset({
    "huggingface_hub",
    "numpy",
    "onnx",
    "onnx_graphsurgeon",
    "onnxscript",
    "safetensors",
    "torch",
    "transformers",
})
TOOL_DEPENDENCIES = EXPORT_DEPENDENCIES | {
    "modelopt",
    "datasets",
    "peft",
    "einops",
    "soundfile",
    "librosa",
    "tiktoken",
    "torchvision",
    "backoff",
    "httpx",
    "tqdm",
}
