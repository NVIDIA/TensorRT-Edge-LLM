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
"""Shared identity parsing for final TensorRT Edge-LLM wheels."""

from __future__ import annotations

import dataclasses
import re

PACKAGE_NAME = "tensorrt-edgellm"
WHEEL_DISTRIBUTION = "tensorrt_edgellm"
PUBLIC_PLATFORM_TAGS = {
    "x86_64": "manylinux_2_35_x86_64",
    "aarch64": "manylinux_2_39_aarch64",
}


@dataclasses.dataclass(frozen=True)
class WheelIdentity:
    """Identity fields encoded in one final Edge-LLM wheel filename."""

    filename: str
    version: str
    python_abi: str
    cpu_arch: str
    build_tag: str | None


def public_platform_tag(cpu_arch: str) -> str:
    """Return the reviewed public wheel platform tag for one architecture."""
    try:
        return PUBLIC_PLATFORM_TAGS[cpu_arch]
    except KeyError as error:
        raise RuntimeError(
            f"Unsupported final wheel architecture: {cpu_arch!r}.") from error


def binary_wheel_tag(python_abi: str, cpu_arch: str) -> str:
    """Return the complete public binary tag for a final wheel."""
    if not re.fullmatch(r"cp3\d{2}", python_abi):
        raise RuntimeError(f"Unsupported final wheel ABI: {python_abi!r}.")
    return f"{python_abi}-{python_abi}-{public_platform_tag(cpu_arch)}"


def parse_final_wheel_filename(filename: str,
                               *,
                               allow_build_tag: bool = False) -> WheelIdentity:
    """Parse the distribution, version, ABI, and architecture from a wheel."""
    if not filename.endswith(".whl"):
        raise RuntimeError(f"Artifact is not a wheel: {filename!r}.")
    fields = filename[:-4].split("-")
    if len(fields) == 5:
        distribution, version, python_tag, abi_tag, platform_tag = fields
        build_tag = None
    elif len(fields) == 6:
        distribution, version, build_tag, python_tag, abi_tag, platform_tag = fields
        if not allow_build_tag:
            raise RuntimeError(
                f"Release wheel must not contain a build tag: {filename!r}.")
        if not re.fullmatch(r"[0-9][0-9A-Za-z_]*", build_tag):
            raise RuntimeError(
                f"Wheel has an invalid build tag: {filename!r}.")
    else:
        raise RuntimeError(f"Wheel has an invalid filename: {filename!r}.")
    if distribution != WHEEL_DISTRIBUTION:
        raise RuntimeError(
            f"Unexpected wheel distribution {distribution!r} in {filename!r}.")
    if not version:
        raise RuntimeError(f"Wheel has an empty version: {filename!r}.")
    if not python_tag or python_tag != abi_tag:
        raise RuntimeError(
            f"Wheel has inconsistent Python ABI tags: {filename!r}.")
    architectures = [
        cpu_arch for cpu_arch, expected in PUBLIC_PLATFORM_TAGS.items()
        if platform_tag == expected
    ]
    if len(architectures) != 1:
        raise RuntimeError(
            f"Wheel has an unsupported public platform tag: {filename!r}.")
    cpu_arch = architectures[0]
    return WheelIdentity(filename, version, python_tag, cpu_arch, build_tag)
