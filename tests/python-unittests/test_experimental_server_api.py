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
"""Tests for the OpenAI-compatible server request fields."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "token_fields,expected",
    [
        ({
            "max_completion_tokens": 17
        }, 17),
        ({
            "max_tokens": 18
        }, 18),
        ({
            "max_completion_tokens": 19,
            "max_tokens": 20
        }, 19),
        ({
            "max_completion_tokens": None,
            "max_tokens": 21
        }, 21),
        ({
            "max_completion_tokens": None
        }, 2048),
        ({
            "max_tokens": None
        }, 2048),
        ({}, 2048),
    ],
)
def test_api_accepts_openai_completion_token_limit(token_fields, expected):
    from fastapi.testclient import TestClient

    from experimental.server.api_server import _create_app

    class FakeRuntime:

        @staticmethod
        def handle_request(_request):
            return SimpleNamespace(output_texts=["ok"],
                                   output_ids=[[1]],
                                   finish_reasons=[],
                                   logprobs=[])

    class FakeLLM:
        _model_id = "test-model"
        has_draft_model = False
        model_dir = "test-model"
        _runtime = FakeRuntime()
        captured_params = None

        def _make_generation_request(self, _messages, params, **_kwargs):
            self.captured_params = params
            return object()

    llm = FakeLLM()
    response = TestClient(_create_app(llm)).post(
        "/v1/chat/completions",
        json={
            "messages": [{
                "role": "user",
                "content": "hello"
            }],
            **token_fields,
        },
    )

    assert response.status_code == 200
    assert llm.captured_params.max_tokens == expected
