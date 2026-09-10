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
"""Tests for OpenAI-compatible logit_bias request validation."""

import math
from types import SimpleNamespace

import pytest

from experimental.server.runtime.engine import (_MAX_LOGIT_BIAS_TOKENS, LLM,
                                                CompletionOutput,
                                                SamplingParams,
                                                _normalize_logit_bias)
from experimental.server.runtime.engine_layout import BundleLayout, EngineType

_INT32_MAX = 2**31 - 1


def _create_app(llm):
    from experimental.server.api.app import create_app
    from experimental.server.runtime.engine_client import EngineClient

    return create_app(EngineClient(llm))


class _FakeServerModel:
    model_id = "test-model"
    model_dir = ""
    bundle_dir = "/nonexistent"
    bundle_layout = BundleLayout(bundle_dir, EngineType.LLM)
    runtime_kind = "chat"
    video_capable = False
    has_draft_model = False


def test_normalize_logit_bias_accepts_integer_like_keys_and_boundaries():
    assert _normalize_logit_bias(None) == {}
    assert _normalize_logit_bias({
        "1": -100,
        2: 0.5,
        "3": 100.0,
        str(_INT32_MAX): 0,
    }) == {
        1: -100.0,
        2: 0.5,
        3: 100.0,
        _INT32_MAX: 0.0,
    }


def test_normalize_logit_bias_rejects_too_many_entries():
    too_many_entries = {str(i): 0.0 for i in range(_MAX_LOGIT_BIAS_TOKENS + 1)}

    with pytest.raises(ValueError, match="max is"):
        _normalize_logit_bias(too_many_entries)


@pytest.mark.parametrize("token_id", [True, 1.25, object(), "not-an-int"])
def test_normalize_logit_bias_rejects_non_integer_token_ids(token_id):
    with pytest.raises(ValueError, match="not an integer"):
        _normalize_logit_bias({token_id: 0.0})


@pytest.mark.parametrize(
    "token_id",
    [-1, "-1", _INT32_MAX + 1, str(_INT32_MAX + 1)])
def test_normalize_logit_bias_rejects_token_ids_outside_nonnegative_int32(
        token_id):
    with pytest.raises(ValueError, match="token ID"):
        _normalize_logit_bias({token_id: 0.0})


@pytest.mark.parametrize("bias", [
    True, "1.0", math.nan, math.inf, -100.1, 100.1,
    pytest.param(10**400, id="overflowing-int")
])
def test_normalize_logit_bias_rejects_invalid_bias_values(bias):
    with pytest.raises(ValueError):
        _normalize_logit_bias({"1": bias})


def test_hlapi_generate_accepts_logit_bias_with_active_spec_decode():

    class FakeAdmission:

        @staticmethod
        def __enter__():
            return None

        @staticmethod
        def __exit__(*args):
            return False

    llm = object.__new__(LLM)
    llm._rt = object()
    llm._runtime = SimpleNamespace(has_draft_model=lambda: True)
    llm._admission = lambda: FakeAdmission()
    llm._ensure_open = lambda: None
    llm._make_generation_request = lambda *args, **kwargs: object()
    llm._complete_prepared_request = lambda *args, **kwargs: SimpleNamespace()

    outputs = llm.generate("hello", SamplingParams(logit_bias={1: 1.0}))

    assert len(outputs) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_api_accepts_logit_bias_with_active_spec_decode(stream):
    TestClient = pytest.importorskip("fastapi.testclient").TestClient

    class FakeLLM(_FakeServerModel):
        has_draft_model = True

        @staticmethod
        def _make_generation_request(*args, **kwargs):
            return object()

        @staticmethod
        def _complete_prepared_request(*args, **kwargs):
            return CompletionOutput()

        @staticmethod
        def generate_stream(*args, **kwargs):
            if False:
                yield None

    response = TestClient(_create_app(FakeLLM())).post(
        "/v1/chat/completions",
        json={
            "messages": [{
                "role": "user",
                "content": "hello"
            }],
            "logit_bias": {
                "1": 1.0
            },
            "stream": stream,
        },
    )

    assert response.status_code == 200


@pytest.mark.parametrize("stream", [False, True])
def test_api_rejects_overflowing_logit_bias(stream):
    TestClient = pytest.importorskip("fastapi.testclient").TestClient

    response = TestClient(_create_app(_FakeServerModel())).post(
        "/v1/chat/completions",
        json={
            "messages": [{
                "role": "user",
                "content": "hello"
            }],
            "logit_bias": {
                "1": 10**400
            },
            "stream": stream,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"].startswith("logit_bias")
