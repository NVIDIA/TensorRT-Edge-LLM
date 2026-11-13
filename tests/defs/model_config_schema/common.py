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
"""Shared validation settings and field types for model-test input."""

import re
from pathlib import Path
from typing import Annotated, Union

import pydantic

from ..shared_roots import shared_root_names

NonemptyString = Annotated[
    str, pydantic.StringConstraints(strip_whitespace=True, min_length=1)]


class ConfigModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", strict=True)


def nonblank_string(value: str) -> str:
    if not value.strip():
        raise ValueError("must be a non-empty string")
    return value


def reject_null(value):
    if value is None:
        raise ValueError("omit this field instead of setting it to null")
    return value


# Inference values retain their original spelling in ModelTest.inference.
NonblankString = Annotated[str, pydantic.AfterValidator(nonblank_string)]

PositiveInt = Annotated[int, pydantic.Field(gt=0)]
NonnegativeInt = Annotated[int, pydantic.Field(ge=0)]
Number = Union[pydantic.StrictInt, pydantic.StrictFloat]

SHARED_ROOT_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def checkpoint_reference_shape(reference: str) -> str:
    if SHARED_ROOT_PATTERN.match(reference) or Path(reference).is_absolute():
        return reference
    parts = reference.split("/")
    if len(parts) != 2 or any(part in ("", ".", "..") for part in parts):
        raise ValueError(
            "must be an absolute path, a ${ROOT} reference, or a Hugging Face repository ID"
        )
    return reference


def checkpoint_reference(reference: str) -> str:
    match = SHARED_ROOT_PATTERN.match(reference)
    if match is not None and match.group(1) not in shared_root_names():
        raise ValueError(
            f"unknown shared root {match.group(1)!r}; known roots: {sorted(shared_root_names())}"
        )
    return checkpoint_reference_shape(reference)


CheckpointReference = Annotated[NonemptyString,
                                pydantic.AfterValidator(checkpoint_reference)]
