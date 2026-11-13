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
"""Registry of model-family checkpoint converters."""

from types import MappingProxyType

from ..converter import CheckpointConverter
from ..schema import UnsupportedConversion
from .gemma4.converter import Gemma4CheckpointConverter

_MODEL_FAMILY_CONVERTERS = MappingProxyType({
    "gemma4":
    Gemma4CheckpointConverter(),
})


def get_supported_model_families() -> tuple[str, ...]:
    """Return stable model-family names accepted by the CLI."""
    return tuple(sorted(_MODEL_FAMILY_CONVERTERS))


def get_checkpoint_converter(model_family: str) -> CheckpointConverter:
    """Resolve a registered converter without exposing concrete types."""
    try:
        return _MODEL_FAMILY_CONVERTERS[model_family]
    except KeyError as error:
        supported = ", ".join(get_supported_model_families())
        raise UnsupportedConversion(
            f"unsupported model family {model_family!r}; supported: {supported}"
        ) from error


__all__ = ["get_checkpoint_converter", "get_supported_model_families"]
