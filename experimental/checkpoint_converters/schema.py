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
"""Family-independent checkpoint conversion request and result contracts."""

from __future__ import annotations

import json
import os
import stat
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConversionError(RuntimeError):
    """Base class for failures that abort checkpoint conversion."""


class UnsupportedConversion(ConversionError):
    """Raised when a request falls outside a family conversion contract."""


class RevisionMismatch(ConversionError):
    """Raised when installed source provenance does not match its pin."""


class CheckpointRestoreError(ConversionError):
    """Raised when an upstream checkpoint API cannot restore a source."""


class MappingError(ConversionError):
    """Base class for source-to-target mapping failures."""


class UnmappedSourceTensor(MappingError):
    """Raised when a restored tensor has no declared mapping rule."""


class ShapeMismatch(MappingError):
    """Raised when a source or transformed tensor has an unexpected shape."""


class InvalidMappingPlan(MappingError):
    """Raised when a mapping plan is incomplete or has duplicate targets."""


def require_plain_directory(path: Path, label: str) -> None:
    """Require an existing, non-symlink directory."""
    if not isinstance(path, Path):
        raise ConversionError(
            f"{label} must be a pathlib.Path, found {type(path).__name__}")
    if path.is_symlink() or not path.is_dir():
        raise ConversionError(
            f"{label} must be an existing non-symlink directory: {path}")


def load_json_asset(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read one regular, non-symlink JSON object and retain its exact bytes."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ConversionError(
                    f"JSON asset must be a regular file: {path}")
            data = stream.read()
    except OSError as error:
        raise ConversionError(
            f"JSON asset must be a regular non-symlink file: {path}"
        ) from error
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConversionError(f"invalid JSON asset: {path}") from error
    if not isinstance(value, dict):
        raise ConversionError(f"JSON asset must contain an object: {path}")
    return value, data


@dataclass(frozen=True)
class ConversionRequest(ABC):
    """Common paths and resource bounds for one family conversion."""

    checkpoint_path: Path
    tokenizer_path: Path
    output_path: Path
    restore_concurrent_gb: int = 16
    max_checkpoint_file_bytes: int = 5 * 1024**3

    def validate(self) -> None:
        """Validate common safety constraints and the family contract."""
        require_plain_directory(self.checkpoint_path, "checkpoint path")
        require_plain_directory(self.tokenizer_path, "tokenizer path")
        if (isinstance(self.restore_concurrent_gb, bool)
                or not isinstance(self.restore_concurrent_gb, int)
                or self.restore_concurrent_gb <= 0):
            raise ConversionError("restore_concurrent_gb must be positive")
        if (isinstance(self.max_checkpoint_file_bytes, bool)
                or not isinstance(self.max_checkpoint_file_bytes, int)
                or self.max_checkpoint_file_bytes <= 0):
            raise ConversionError("max_checkpoint_file_bytes must be positive")
        self._validate_output_path()
        self.validate_family()

    def _validate_output_path(self) -> None:
        output = self.output_path
        if not isinstance(output, Path):
            raise ConversionError("output path must be a pathlib.Path, found "
                                  f"{type(output).__name__}")
        if output.is_symlink():
            raise ConversionError(f"output path cannot be a symlink: {output}")
        parent = output.parent
        if parent.is_symlink() or not parent.is_dir():
            raise ConversionError(
                f"output parent must be an existing non-symlink directory: {parent}"
            )
        if output.exists():
            if not output.is_dir():
                raise ConversionError(
                    f"output path must be a directory: {output}")
            if any(output.iterdir()):
                raise ConversionError(
                    f"output directory must be absent or empty: {output}")

    @abstractmethod
    def validate_family(self) -> None:
        """Validate model-family-specific revision, tokenizer, and mode rules."""


@dataclass(frozen=True)
class ConversionReport:
    """Summary of one atomically published checkpoint conversion."""

    output_path: Path
    tensor_count: int
    checkpoint_files: tuple[str, ...]
    total_size: int
