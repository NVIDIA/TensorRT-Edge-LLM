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
"""Interface between the online driver and a serving protocol."""

import dataclasses
from typing import Any, Mapping, Protocol


@dataclasses.dataclass(frozen=True)
class HttpRequest:
    """An HTTP request produced by a serving protocol adapter."""

    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None = None


class ServingAdapter(Protocol):
    """Translate canonical requests and streamed responses."""

    def readiness_request(self) -> HttpRequest:
        """Build the backend readiness request."""

    def inference_request(self, request: Mapping[str, Any],
                          generation: Mapping[str, Any]) -> HttpRequest:
        """Build one streaming inference request."""

    def parse_stream_line(self, line: str) -> dict[str, Any] | None:
        """Parse one line from the streaming response."""
