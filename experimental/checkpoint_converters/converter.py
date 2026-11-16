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
"""Family-agnostic checkpoint conversion interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import Namespace

from .schema import ConversionReport, ConversionRequest


class CheckpointConverter(ABC):
    """Construct and execute one model-family conversion request."""

    request_type: type[ConversionRequest]

    def request_arguments(self, args: Namespace) -> dict[str, object]:
        """Return family-independent request arguments from the CLI."""
        return {
            "checkpoint_path": args.checkpoint_path,
            "tokenizer_path": args.tokenizer_path,
            "output_path": args.output_path,
            "restore_concurrent_gb": args.restore_concurrent_gb,
            "max_checkpoint_file_bytes": args.max_checkpoint_file_bytes,
        }

    def build_request(self, args: Namespace) -> ConversionRequest:
        """Construct the family request from CLI arguments."""
        return self.request_type(**self.request_arguments(args))

    @abstractmethod
    def convert(self, request: ConversionRequest) -> ConversionReport:
        """Validate, convert, and atomically publish one checkpoint."""
