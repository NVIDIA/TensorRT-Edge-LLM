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
"""Gemma4 checkpoint adapter and supported variants."""

from .converter import Gemma4CheckpointConverter
from .flax_adapter import (Gemma4FlaxAdapter, validate_mapping_plan,
                           validate_rule_contracts, validate_source_coverage,
                           validate_target_coverage)
from .schema import (GEMMA_REVISION, HF_REVISIONS, ROUNDING_POLICY,
                     SOURCE_DTYPE, SUPPORTED_DTYPE, SUPPORTED_VARIANT,
                     SUPPORTED_VARIANTS, TARGET_DTYPE, TOKENIZER_FILES,
                     Gemma4ConversionRequest, check_installed_revision,
                     process_tokenizer_asset, validate_tokenizer_assets)
from .source import Gemma4OrbaxCheckpointReader
from .variant import (A4B_26B_IT_VARIANT, E2B_IT_VARIANT, VARIANTS,
                      AttentionType, Gemma4VariantSpec, KvSharingSpec, MoeSpec,
                      PleSpec, validate_hf_config)
from .writer import Gemma4CheckpointWritePolicy

__all__ = [
    "AttentionType",
    "A4B_26B_IT_VARIANT",
    "E2B_IT_VARIANT",
    "Gemma4CheckpointConverter",
    "Gemma4FlaxAdapter",
    "Gemma4CheckpointWritePolicy",
    "Gemma4ConversionRequest",
    "Gemma4OrbaxCheckpointReader",
    "Gemma4VariantSpec",
    "GEMMA_REVISION",
    "HF_REVISIONS",
    "KvSharingSpec",
    "MoeSpec",
    "PleSpec",
    "ROUNDING_POLICY",
    "SOURCE_DTYPE",
    "SUPPORTED_DTYPE",
    "SUPPORTED_VARIANT",
    "SUPPORTED_VARIANTS",
    "TARGET_DTYPE",
    "TOKENIZER_FILES",
    "VARIANTS",
    "check_installed_revision",
    "process_tokenizer_asset",
    "validate_hf_config",
    "validate_mapping_plan",
    "validate_rule_contracts",
    "validate_source_coverage",
    "validate_target_coverage",
    "validate_tokenizer_assets",
]
