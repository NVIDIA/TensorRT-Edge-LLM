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
"""Input files, option precedence, and configuration validation."""

import json
import os
import pathlib

import pytest

from experimental.frontend import config
from experimental.server.runtime import engine

from . import helpers


def test_hugging_face_model_id_is_preserved(tmp_path, monkeypatch):
    # Given a model ID without a matching local directory.
    # When settings are resolved, then its ID is preserved for the shared API.
    monkeypatch.chdir(tmp_path)
    model = "Qwen/Qwen3.5-4B"
    result = config.resolve(config.parser().parse_args([model]))
    assert result.model == model


@pytest.mark.parametrize("path_kind", ["relative", "absolute", "home"])
def test_local_model_path_is_resolved(tmp_path, monkeypatch, path_kind):
    # Given a local checkpoint whose name also resembles a Hugging Face ID.
    # When settings are resolved, then its absolute local path takes precedence.
    checkpoint = tmp_path / "Qwen" / "Qwen3.5-4B"
    checkpoint.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    model = {
        "relative": "Qwen/Qwen3.5-4B",
        "absolute": str(checkpoint),
        "home": "~/" + os.path.relpath(checkpoint, pathlib.Path.home()),
    }[path_kind]
    result = config.resolve(config.parser().parse_args([model]))
    assert result.model == str(checkpoint.resolve())


@pytest.mark.parametrize("path_kind", ["relative", "absolute", "home"])
def test_missing_explicit_model_path_is_rejected(tmp_path, monkeypatch,
                                                 path_kind):
    # Given an explicit path to a nonexistent checkpoint.
    # When settings are resolved, then a local-path error prevents a Hub lookup.
    checkpoint = tmp_path / "missing-checkpoint"
    monkeypatch.chdir(tmp_path)
    model = {
        "relative": "./missing-checkpoint",
        "absolute": str(checkpoint),
        "home": "~/" + os.path.relpath(checkpoint, pathlib.Path.home()),
    }[path_kind]
    with pytest.raises(ValueError,
                       match="Checkpoint directory does not exist"):
        config.resolve(config.parser().parse_args([model]))


@pytest.mark.parametrize("messages", [
    {
        "role": "user",
        "content": "not an array"
    },
    [{
        "role": "tool",
        "content": "unsupported role"
    }],
])
def test_template_file_requires_supported_chat_messages(tmp_path, messages):
    # Given an unsupported chat template.
    # When settings are resolved, then validation rejects it.
    path = tmp_path / "template.json"
    path.write_text(json.dumps(messages))
    with pytest.raises(ValueError, match="Template|template"):
        helpers.settings(tmp_path, "--template", str(path))


@pytest.mark.parametrize("option", ["prompt", "system"])
def test_inline_and_file_inputs_are_mutually_exclusive(tmp_path, option):
    # Given inline and file input for one option.
    # When resolved, then neither silently overrides the other.
    path = tmp_path / "input.txt"
    path.write_text("from file")
    with pytest.raises(ValueError, match="Use either"):
        helpers.settings(tmp_path, f"--{option}", "inline", f"--{option}-file",
                         str(path))


def test_config_precedence_and_api_defaults(tmp_path, monkeypatch):
    # Given file, environment, and CLI settings.
    # When resolved, then precedence and API defaults hold.
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"temperature": 0.2, "top_k": 12}))
    monkeypatch.setenv("EDGELLM_TEMPERATURE", "0.4")
    monkeypatch.setenv("EDGELLM_MAX_TOKENS", "32")
    result = helpers.settings(tmp_path, "--config", str(path), "--temperature",
                              "0.6")
    assert result.sampling.temperature == 0.6
    assert result.sampling.top_k == 12
    assert result.sampling.max_tokens == 32
    assert result.sampling.top_p == engine.SamplingParams().top_p
    assert result.runtime == {}


def test_structured_launch_option_precedence_and_build_defaults(
        tmp_path, monkeypatch):
    # Given structured overrides.
    # When resolved, then CLI wins and unspecified build limits keep their
    # defaults.
    path = tmp_path / "launch.json"
    path.write_text(
        json.dumps({
            "stop": ["file"],
            "clear_engine_cache": True,
            "context_cache_config": {
                "enabled": False
            }
        }))
    monkeypatch.setenv("EDGELLM_STOP", '["environment"]')
    monkeypatch.setenv("EDGELLM_CONTEXT_CACHE_CONFIG", '{"enabled":true}')
    result = helpers.settings(tmp_path, "--config", str(path), "--stop", "CLI",
                              "--no-clear-engine-cache", "--tp-size", "1")
    assert result.sampling.stop == ["CLI"]
    assert not result.runtime["clear_engine_cache"]
    assert result.runtime["context_cache_config"].enabled
    build = result.runtime["build_options"]
    assert (build.max_input_len, build.max_batch_size,
            build.max_kv_cache_capacity) == (4096, 1, 8192)


@pytest.mark.parametrize("options", [
    ["--speculative-config", '{"method":"unknown"}'],
    ["--speculative-config", '{"method":"dflash"}'],
    ["--context-cache-config", '{"enabled":false,"max_records":8}'],
    ["--logit-bias", '{"42":101}'],
    ["--guided-decoding", '{"regex":"a","ebnf":"b"}'],
    ["--num-logprobs", "-1"],
])
def test_invalid_launch_options_fail_before_loading(tmp_path, options):
    # Given invalid runtime options.
    # When resolving settings, then validation fails before model loading.
    with pytest.raises(ValueError):
        helpers.settings(tmp_path, *options)


@pytest.mark.parametrize("override", [
    {
        "temperature": "nan"
    },
    {
        "top_p": 2
    },
    {
        "max_tokens": 0
    },
    {
        "top_k": 1.5
    },
    {
        "enable_thinking": "maybe"
    },
    {
        "unknown": 1
    },
])
def test_invalid_settings_fail_before_loading(tmp_path, override):
    # Given invalid settings in a file.
    # When resolved, then validation rejects them before model loading.
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(override))
    with pytest.raises(ValueError):
        helpers.settings(tmp_path, "--config", str(path))


def test_file_inputs_and_constraints(tmp_path):
    # Given input and schema files.
    # When resolved, then their contents define the prompt, system message, and
    # constraint.
    (tmp_path / "prompt.txt").write_text("hello\nworld")
    (tmp_path / "system.txt").write_text("Be brief.")
    (tmp_path /
     "template.json").write_text('[{"role":"system","content":"old"}]')
    (tmp_path / "schema.json").write_text('{"type":"object"}')
    result = helpers.settings(tmp_path, "--prompt-file",
                              str(tmp_path / "prompt.txt"), "--system-file",
                              str(tmp_path / "system.txt"), "--template",
                              str(tmp_path / "template.json"), "--json-schema",
                              str(tmp_path / "schema.json"))
    assert result.prompt == "hello\nworld"
    assert result.messages == [{"role": "system", "content": "Be brief."}]
    assert result.sampling.guided_decoding == ("json_schema",
                                               '{"type": "object"}')
