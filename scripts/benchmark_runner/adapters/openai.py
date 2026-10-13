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
"""OpenAI Completions protocol adapter for online benchmarks."""

import json
import os
from typing import Any, Mapping

from . import base


class CompletionsAdapter:
    """Translate token-ID requests to OpenAI Completions calls."""

    def __init__(self, config: Mapping[str, Any]):
        endpoint = config.get("endpoint")
        model = config.get("model")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("OpenAI adapter requires an endpoint")
        if not isinstance(model, str) or not model:
            raise ValueError("OpenAI adapter requires a model")
        self._endpoint = self._expand(endpoint).rstrip("/")
        self._model = self._expand(model)
        self._headers = {"Content-Type": "application/json"}
        headers = config.get("headers", {})
        if (not isinstance(headers, dict)
                or any(not isinstance(name, str) or not isinstance(value, str)
                       for name, value in headers.items())):
            raise ValueError("OpenAI adapter headers must be strings")
        self._headers.update({
            name: self._expand(value)
            for name, value in headers.items()
        })
        self._health_path = self._path(config, "health_path", "/health")
        self._completions_path = self._path(config, "completions_path",
                                            "/v1/completions")

    @staticmethod
    def _expand(value: str) -> str:
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            raise ValueError(
                f"OpenAI adapter has an unresolved value: {value}")
        return expanded

    @staticmethod
    def _path(config: Mapping[str, Any], name: str, default: str) -> str:
        value = config.get(name, default)
        if not isinstance(value, str) or not value.startswith("/"):
            raise ValueError(f"OpenAI adapter {name} must start with '/'")
        return value

    def readiness_request(self) -> base.HttpRequest:
        """Build the health request."""
        return base.HttpRequest("GET", self._endpoint + self._health_path,
                                dict(self._headers))

    def inference_request(self, request: Mapping[str, Any],
                          generation: Mapping[str, Any]) -> base.HttpRequest:
        """Build one streaming completion request."""
        body = {
            "model": self._model,
            "prompt": request["token_ids"],
            "max_tokens": request["output_tokens"],
            "temperature": generation["temperature"],
            "top_p": generation["top_p"],
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if generation["top_k"] > 0:
            body["top_k"] = generation["top_k"]
        if generation["ignore_eos"]:
            body["ignore_eos"] = True
        return base.HttpRequest("POST",
                                self._endpoint + self._completions_path,
                                dict(self._headers),
                                json.dumps(body).encode("utf-8"))

    def parse_stream_line(self, line: str) -> dict[str, Any] | None:
        """Parse one server-sent event from a completion response."""
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            return None
        if not stripped.startswith("data:"):
            raise ValueError("Completion stream must use data events")
        payload = stripped[5:].strip()
        if payload == "[DONE]":
            return {"done": True}
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("Completion event must be a JSON object")
        if value.get("error") is not None:
            raise RuntimeError(
                f"Completion server returned an error: {value['error']}")

        result: dict[str, Any] = {}
        choices = value.get("choices", [])
        if isinstance(choices, list):
            text = []
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                fragment = choice.get("text")
                if isinstance(fragment, str) and fragment:
                    text.append(fragment)
                    result["token_observed"] = True
            if text:
                result["text"] = "".join(text)
        usage = value.get("usage")
        if isinstance(usage, dict):
            result["usage"] = usage
        return result or None
