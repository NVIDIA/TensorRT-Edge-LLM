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
"""Internal native payload selection and loading support."""


class NativeRuntimeError(RuntimeError):
    """Base error raised before or while loading an EdgeLLM native payload."""


class NativeDetectionError(NativeRuntimeError):
    """Raised when the current platform cannot be identified safely."""


class NativeSelectionError(NativeRuntimeError):
    """Raised when no unique compatible payload exists."""


class NativeManifestError(NativeSelectionError):
    """Raised when installed payload metadata is absent or invalid."""


class NativeManifestNotFoundError(NativeManifestError):
    """Raised only when the generated native manifest is not installed."""


class NativeLoadError(NativeRuntimeError):
    """Raised when a selected payload cannot be validated or loaded."""


__all__ = [
    "NativeDetectionError",
    "NativeLoadError",
    "NativeManifestError",
    "NativeManifestNotFoundError",
    "NativeRuntimeError",
    "NativeSelectionError",
]
