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
"""Gemma4 checkpoint conversion orchestration."""

from __future__ import annotations

from argparse import Namespace

from ...converter import CheckpointConverter
from ...schema import ConversionReport, ConversionRequest
from ...writer import HFCheckpointWriter
from .flax_adapter import Gemma4FlaxAdapter
from .schema import Gemma4ConversionRequest
from .source import Gemma4OrbaxCheckpointReader
from .variant import VARIANTS
from .writer import Gemma4CheckpointWritePolicy


class Gemma4CheckpointConverter(CheckpointConverter):
    """Convert one supported pinned Gemma4 Orbax checkpoint."""

    request_type = Gemma4ConversionRequest

    def request_arguments(self, args: Namespace) -> dict[str, object]:
        """Add the selected Gemma4 variant to common request arguments."""
        values = super().request_arguments(args)
        if args.model_variant is not None:
            values["variant_name"] = args.model_variant
        return values

    def convert(self, request: ConversionRequest) -> ConversionReport:
        """Run Gemma4 validation, restore, mapping, and publication."""
        if not isinstance(request, Gemma4ConversionRequest):
            raise TypeError(
                "Gemma4 converter requires Gemma4ConversionRequest")
        request.validate()
        source_tree = Gemma4OrbaxCheckpointReader(
            revision=request.gemma_revision).restore(
                request.checkpoint_path, request.restore_concurrent_gb)
        adapter = Gemma4FlaxAdapter(variant=VARIANTS[request.variant_name])
        plan = adapter.plan(source_tree)
        config = adapter.build_config()
        writer = HFCheckpointWriter(
            output_path=request.output_path,
            max_checkpoint_file_bytes=request.max_checkpoint_file_bytes,
            policy=Gemma4CheckpointWritePolicy(
                variant_name=request.variant_name),
        )
        return writer.write(request.tokenizer_path, source_tree, plan, config)
