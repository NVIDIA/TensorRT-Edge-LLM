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
"""Tests for guided-decoding request translation and validation in the server."""

import pytest

from experimental.server.runtime.engine import (CompletionOutput,
                                                _normalize_guided_decoding,
                                                _normalize_response_format)
from experimental.server.runtime.engine_layout import BundleLayout, EngineType


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

    @staticmethod
    def _make_generation_request(*args, **kwargs):
        return object()

    @staticmethod
    def _complete_prepared_request(*args, **kwargs):
        return CompletionOutput()


# --------------------------------------------------------------- response_format


@pytest.mark.parametrize("response_format", [None, {"type": "text"}])
def test_response_format_text_leaves_generation_unconstrained(response_format):
    assert _normalize_response_format(response_format) is None


def test_response_format_json_object_maps_to_the_json_object_mode():
    assert _normalize_response_format({"type":
                                       "json_object"}) == ("json_object", "")


def test_response_format_json_schema_is_unwrapped_and_normalized():
    # OpenAI nests the schema one level deeper than the low-level field does, and the
    # dump is key-sorted so two spellings of one schema share a compiled-grammar entry.
    assert _normalize_response_format({
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "b": 1,
                "a": 2
            },
        },
    }) == ("json_schema", '{"a": 2, "b": 1}')


def test_response_format_ignores_strict():
    # Guided decoding is always enforced, so `strict` carries no information; vLLM
    # drops it the same way.
    guide = _normalize_response_format({
        "type": "json_schema",
        "strict": False,
        "json_schema": {
            "schema": {
                "type": "object"
            }
        },
    })
    assert guide == ("json_schema", '{"type": "object"}')


@pytest.mark.parametrize("response_format", [
    {
        "type": "yaml"
    },
    {
        "type": "json_schema"
    },
    {
        "type": "json_schema",
        "json_schema": {}
    },
    "not-an-object",
])
def test_response_format_rejects_malformed_input(response_format):
    with pytest.raises(ValueError):
        _normalize_response_format(response_format)


# --------------------------------------------------------------- guided_decoding


def test_guided_decoding_carries_the_modes_response_format_cannot():
    assert _normalize_guided_decoding({"regex": "a+"}) == ("regex", "a+")
    assert _normalize_guided_decoding({"ebnf": 'root ::= "a"'
                                       }) == ("ebnf", 'root ::= "a"')
    tag = {"type": "structural_tag"}
    assert _normalize_guided_decoding({"structural_tag":
                                       tag})[0] == "structural_tag"


def test_guided_decoding_accepts_a_choice_list():
    # The list is carried to the runtime as a JSON array; order is preserved because it is
    # the payload, not a set.
    assert _normalize_guided_decoding({"choice":
                                       ["yes",
                                        "no"]}) == ("choice", '["yes", "no"]')


def test_guided_decoding_accepts_an_inline_schema_object():
    # Requiring an escaped JSON string here would make the field nearly unusable.
    assert _normalize_guided_decoding({"json_schema": {
        "type": "object"
    }}) == ("json_schema", '{"type": "object"}')


def test_guided_decoding_normalizes_key_order_for_cache_reuse():
    assert (_normalize_guided_decoding({"json_schema": {
        "b": 1,
        "a": 2
    }}) == _normalize_guided_decoding({"json_schema": {
        "a": 2,
        "b": 1
    }}))


@pytest.mark.parametrize("guided_decoding", [
    None,
    {},
    {
        "json_object": False
    },
    {
        "regex": ""
    },
    {
        "ebnf": ""
    },
    {
        "choice": []
    },
])
def test_guided_decoding_treats_unset_and_empty_as_unconstrained(
        guided_decoding):
    # An empty string must not be read as "a grammar accepting nothing", which would
    # compile to a mask that forbids every token.
    assert _normalize_guided_decoding(guided_decoding) is None


def test_guided_decoding_rejects_a_choice_that_is_not_a_list():
    # A bare string would otherwise be forwarded verbatim and rejected much later by the
    # backend as "not valid JSON", which does not describe what the caller got wrong.
    with pytest.raises(ValueError, match="array of strings"):
        _normalize_guided_decoding({"choice": "yes"})


def test_guided_decoding_rejects_more_than_one_mode():
    with pytest.raises(ValueError, match="exactly one"):
        _normalize_guided_decoding({"regex": "a+", "ebnf": 'root ::= "a"'})


def test_guided_decoding_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown field"):
        _normalize_guided_decoding({"json_schemaa": {}})


# --------------------------------------------------------------- HTTP surface


def test_api_rejects_response_format_and_guided_decoding_together():
    TestClient = pytest.importorskip("fastapi.testclient").TestClient

    response = TestClient(_create_app(_FakeServerModel())).post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": "hi"
            }],
            "response_format": {
                "type": "json_object"
            },
            "guided_decoding": {
                "regex": "a+"
            },
        },
    )

    assert response.status_code == 400
    assert "cannot be used together" in response.text


def test_api_reports_an_unsupported_schema_keyword_as_a_400():
    # multipleOf compiles cleanly in XGrammar and is then ignored, so the output would
    # violate the schema while the API claims it cannot. Catching it here keeps the
    # failure a 400 naming the keyword rather than a late, generic request failure.
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
    response = TestClient(_create_app(_FakeServerModel())).post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": "hi"
            }],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "integer",
                        "multipleOf": 5
                    }
                },
            },
        },
    )

    assert response.status_code == 400
    assert "multipleOf" in response.text


def test_api_accepts_a_supported_schema():
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
    response = TestClient(_create_app(_FakeServerModel())).post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": "hi"
            }],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string"
                            }
                        },
                    }
                },
            },
        },
    )

    assert response.status_code == 200
