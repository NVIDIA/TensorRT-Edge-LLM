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
"""Gemma4 policy for generic Hugging Face checkpoint publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from ...transforms import MappingRule
from ...writer import CheckpointWritePolicy, ConversionMetadata
from .schema import (GEMMA_REVISION, HF_REVISIONS, ROUNDING_POLICY,
                     SOURCE_DTYPE, SUPPORTED_VARIANT, TARGET_DTYPE,
                     TOKENIZER_FILES, validate_tokenizer_asset)
from .variant import A4B_26B_IT_VARIANT, E2B_IT_VARIANT

_LARGE_PLE_TARGET: Final = (
    "model.language_model.embed_tokens_per_layer.weight", )
_HF_REFERENCE_FAMILIES: Final = {
    E2B_IT_VARIANT.name: "google/gemma-4-e2b-it",
    A4B_26B_IT_VARIANT.name: "google/gemma-4-26B-A4B-it",
}


@dataclass(frozen=True)
class Gemma4CheckpointWritePolicy(CheckpointWritePolicy):
    """Provide Gemma4 tokenizer, provenance, and indivisible PLE policy."""

    variant_name: str = SUPPORTED_VARIANT

    @property
    def required_tokenizer_files(self) -> tuple[str, ...]:
        return TOKENIZER_FILES

    @property
    def generated_checkpoint_files(self) -> dict[str, bytes]:
        return {}

    def validate_tokenizer_asset(self, name: str, value: dict[str,
                                                              Any]) -> None:
        validate_tokenizer_asset(name, value)

    def requires_dedicated_file(self, rule: MappingRule) -> bool:
        return rule.target_names == _LARGE_PLE_TARGET

    @property
    def conversion_metadata(self) -> ConversionMetadata:
        return ConversionMetadata(
            source_family="google-deepmind/gemma",
            source_revision=GEMMA_REVISION,
            hf_reference_family=_HF_REFERENCE_FAMILIES[self.variant_name],
            hf_reference_revision=HF_REVISIONS[self.variant_name],
            source_dtype=SOURCE_DTYPE,
            target_dtype=TARGET_DTYPE,
            rounding_policy=ROUNDING_POLICY,
        )
