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
"""Flax-to-HF checkpoint conversion contracts and supported families."""

from .adapter import CheckpointAdapter
from .converter import CheckpointConverter
from .models import get_checkpoint_converter, get_supported_model_families
from .schema import (CheckpointRestoreError, ConversionError, ConversionReport,
                     ConversionRequest, InvalidMappingPlan, MappingError,
                     RevisionMismatch, ShapeMismatch, UnmappedSourceTensor,
                     UnsupportedConversion)
from .source import OrbaxCheckpointReader
from .transforms import (MappingPlan, MappingRule, Transform, TransformKind,
                         apply_transform, convert_fp32_to_bf16, convert_tensor)
from .variant import CheckpointVariant
from .writer import (CheckpointFilePlan, CheckpointFileReport,
                     CheckpointWritePolicy, ConversionMetadata,
                     HFCheckpointWriter)

__all__ = [
    "CheckpointAdapter",
    "CheckpointConverter",
    "CheckpointFilePlan",
    "CheckpointFileReport",
    "CheckpointRestoreError",
    "CheckpointVariant",
    "CheckpointWritePolicy",
    "ConversionError",
    "ConversionReport",
    "ConversionRequest",
    "ConversionMetadata",
    "HFCheckpointWriter",
    "InvalidMappingPlan",
    "MappingError",
    "MappingPlan",
    "MappingRule",
    "OrbaxCheckpointReader",
    "RevisionMismatch",
    "ShapeMismatch",
    "Transform",
    "TransformKind",
    "UnmappedSourceTensor",
    "UnsupportedConversion",
    "apply_transform",
    "convert_fp32_to_bf16",
    "convert_tensor",
    "get_checkpoint_converter",
    "get_supported_model_families",
]
