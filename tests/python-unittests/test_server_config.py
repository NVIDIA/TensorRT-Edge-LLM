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
"""Server launch-contract and parser-registry coverage."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from packaging.requirements import Requirement

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

from experimental.server.config import (ApiConfig, ContextCacheConfig,
                                        ServerConfigError, SpeculativeConfig,
                                        parse_server_config)
from experimental.server.parsing.reasoning import REASONING_PARSERS


def _dependency_names(requirements):
    return {Requirement(value).name.lower() for value in requirements}


def test_server_dependencies_exclude_export_and_native_build_toolchains():
    project_root = Path(__file__).resolve().parents[2]
    with open(project_root / "pyproject.toml", "rb") as file:
        project = tomllib.load(file)["project"]

    base = _dependency_names(project["dependencies"])
    extras = project["optional-dependencies"]
    builder = _dependency_names(extras["builder"])
    server = _dependency_names(extras["server"])
    server_tools = _dependency_names(extras["server-tools"])
    export = _dependency_names(extras["export"])
    tools = _dependency_names(extras["tools"])
    native_build = _dependency_names(extras["native-build"])

    assert {
        "cuda-python",
        "fastapi",
        "uvicorn",
        "huggingface-hub",
        "av",
        "numpy",
        "python-multipart",
    } <= server
    assert {"cuda-python", "numpy"} <= base
    assert not builder
    assert not server & {
        "torch",
        "transformers",
        "onnx",
        "onnxscript",
        "onnx-graphsurgeon",
        "safetensors",
        "pybind11",
    }
    assert server_tools == {"transformers", "jinja2"}
    assert "torch" not in server_tools
    assert native_build == {"pybind11"}
    assert export <= tools


def test_package_import_does_not_load_export_frameworks():
    project_root = Path(__file__).resolve().parents[2]
    code = """
import sys
import tensorrt_edgellm

assert tensorrt_edgellm.__version__
assert "tensorrt_edgellm._export_api" not in sys.modules
assert "torch" not in sys.modules
assert "onnx" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=project_root, check=True)


def test_maps_model_cache_and_build_profile(tmp_path):
    config = parse_server_config([
        "Qwen/Qwen3-0.6B",
        "--cache-dir",
        str(tmp_path),
        "--engine-cache-max-size-gb",
        "12.5",
        "--clear-engine-cache",
        "--max-input-len",
        "2048",
        "--max-kv-cache-capacity",
        "8192",
        "--max-batch-size",
        "4",
        "--enable-context-reuse",
        "--context-cache-max-records",
        "64",
        "--context-cache-recurrent-snapshot-pool-bytes",
        "1048576",
        "--context-cache-partial-kv-snapshot-pool-bytes",
        "2097152",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_xml",
        "--enable-auto-tool-choice",
    ])
    assert config.model.model == "Qwen/Qwen3-0.6B"
    assert config.model.cache_dir == str(tmp_path)
    assert config.model.engine_cache_max_size_gb == 12.5
    assert config.model.clear_engine_cache
    assert config.model.max_input_len == 2048
    assert config.model.max_kv_cache_capacity == 8192
    assert config.model.max_batch_size == 4
    assert config.model.context_cache_config == ContextCacheConfig(
        enabled=True,
        max_records=64,
        recurrent_snapshot_pool_bytes=1048576,
        partial_kv_snapshot_pool_bytes=2097152,
    )
    assert config.api.enable_auto_tool_choice


def test_context_cache_config_rejects_ignored_or_invalid_options():
    with pytest.raises(ServerConfigError, match="tuning requires"):
        ContextCacheConfig(recurrent_snapshot_pool_bytes=1)
    with pytest.raises(ServerConfigError, match="tuning requires"):
        ContextCacheConfig(max_records=1)
    with pytest.raises(ServerConfigError, match="unsupported context-cache"):
        ContextCacheConfig.parse({"enabled": True, "unknown": 1})
    with pytest.raises(ServerConfigError, match="must be a mapping"):
        ContextCacheConfig.parse("enabled")
    with pytest.raises(ServerConfigError, match="non-negative"):
        ContextCacheConfig(enabled=True, max_records=-1)
    with pytest.raises(ServerConfigError, match="signed 32-bit"):
        ContextCacheConfig(enabled=True, max_records=2**31)
    with pytest.raises(ServerConfigError, match="signed 64-bit"):
        ContextCacheConfig(enabled=True, recurrent_snapshot_pool_bytes=2**63)


def test_allowed_local_media_path_survives_cli_parsing(tmp_path):
    config = parse_server_config([
        "Qwen/Qwen3.5-0.8B",
        "--allowed-local-media-path",
        str(tmp_path),
    ])
    assert config.api.allowed_local_media_path == str(tmp_path)


@pytest.mark.parametrize("kwargs,match", [
    ({
        "max_queued_requests": -1
    }, "non-negative"),
    ({
        "queue_timeout": 0
    }, "positive"),
])
def test_api_config_rejects_invalid_admission_values(kwargs, match):
    with pytest.raises(ServerConfigError, match=match):
        ApiConfig(**kwargs)


def test_cli_forwards_http_configuration(monkeypatch, tmp_path):
    from experimental.server import cli

    captured = {}

    def create_llm(**kwargs):
        captured["llm"] = kwargs
        return object()

    def create_client(llm, config):
        captured["client"] = (llm, config)
        return SimpleNamespace(
            model_name="test",
            capabilities=SimpleNamespace(max_model_len=1024,
                                         kv_cache_dtype="fp16",
                                         speculative_decoding=False,
                                         context_reuse=False),
        )

    monkeypatch.setattr(cli, "load_model", create_llm)
    monkeypatch.setattr(cli, "EngineClient", create_client)
    monkeypatch.setattr(
        cli,
        "run_http_server",
        lambda client, config: captured.setdefault("server", (client, config)),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "tensorrt-edgellm-serve",
            "Qwen/Qwen3.5-0.8B",
            "--port",
            "9000",
            "--allowed-local-media-path",
            str(tmp_path),
        ],
    )
    cli.main()

    api = captured["server"][1]
    assert api.port == 9000
    assert api.allowed_local_media_path == str(tmp_path)


def test_llm_serve_forwards_allowed_local_media_path(monkeypatch, tmp_path):
    from experimental.server.api import app as app_module
    from experimental.server.runtime import engine_client
    from experimental.server.runtime.engine import LLM

    captured = {}
    llm = LLM.__new__(LLM)

    def create_client(instance, config):
        captured["client"] = (instance, config)
        return object()

    monkeypatch.setattr(engine_client, "EngineClient", create_client)
    monkeypatch.setattr(
        app_module,
        "run_http_server",
        lambda client, config: captured.setdefault("server", (client, config)),
    )
    llm.serve(allowed_local_media_path=str(tmp_path))

    assert captured["client"][0] is llm
    assert captured["server"][1].allowed_local_media_path == str(tmp_path)


def test_paired_speculation_requires_draft_model():
    with pytest.raises(ServerConfigError, match="speculative-config.model"):
        parse_server_config([
            "Qwen/Qwen3-1.7B",
            "--speculative-config",
            '{"method":"eagle3","num_speculative_tokens":3}',
        ])


def test_eagle_build_uses_real_base_and_draft_ids():
    config = parse_server_config([
        "Qwen/Qwen3-1.7B",
        "--speculative-config",
        ('{"method":"eagle3",'
         '"model":"AngelSlim/Qwen3-1.7B_eagle3",'
         '"num_speculative_tokens":3}'),
    ])
    spec = config.model.speculative_config
    assert spec.method == "eagle3"
    assert spec.draft_model == "AngelSlim/Qwen3-1.7B_eagle3"
    assert config.model.draft_step == 3


def test_separate_mtp_checkpoint_selects_gemma_builder_contract():
    config = parse_server_config([
        "nvidia/Gemma-4-26B-A4B-NVFP4",
        "--speculative-config",
        ('{"method":"mtp",'
         '"model":"google/gemma-4-26B-A4B-it-assistant"}'),
    ])
    spec = config.model.speculative_config
    assert spec.method == "mtp"
    assert spec.draft_model == "google/gemma-4-26B-A4B-it-assistant"


def test_native_mtp_uses_base_checkpoint():
    config = parse_server_config([
        "Qwen/Qwen3.5-9B",
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":2}',
    ])
    spec = config.model.speculative_config
    assert spec.method == "mtp"
    assert not spec.draft_model


@pytest.mark.parametrize("payload", [
    '{"method":"mtp","num_speculative_tokens":2}',
    '{"method":"mtp"}',
    '{"method":"eagle3","model":"org/draft-ckpt"}',
])
def test_speculative_config_parse_is_idempotent(payload):
    """LLM.__init__ re-parses whatever parse_server_config already produced."""
    once = SpeculativeConfig.parse(payload)
    assert SpeculativeConfig.parse(once) == once


def test_speculative_config_still_rejects_a_blank_draft_model():
    with pytest.raises(ServerConfigError, match="checkpoint path"):
        SpeculativeConfig.parse('{"method":"mtp","model":"  "}')


def test_rejects_unknown_speculative_keys():
    with pytest.raises(ServerConfigError, match="unknown_key"):
        parse_server_config([
            "Qwen/Qwen3-0.6B",
            "--speculative-config",
            '{"method":"mtp","unknown_key":true}',
        ])


def test_reasoning_parser_handles_split_delimiters():
    parser = REASONING_PARSERS.resolve("qwen3", "")
    stream = parser.stream()
    deltas = []
    for chunk in ("<th", "ink>plan</th", "ink>answer"):
        deltas.extend(stream.feed(chunk))
    deltas.extend(stream.flush())
    reasoning = "".join(d.text for d in deltas if d.field == "reasoning")
    content = "".join(d.text for d in deltas if d.field == "content")
    assert reasoning == "plan"
    assert content == "answer"


def test_reasoning_parser_matches_missing_start_behavior():
    parser = REASONING_PARSERS.resolve("qwen3", "")
    assert parser.extract("plan</think>answer") == ("plan", "answer")
    assert parser.extract("unfinished reasoning") == ("unfinished reasoning",
                                                      None)


def test_reasoning_parser_uses_model_metadata_after_relocation(tmp_path):
    checkpoint = tmp_path / "renamed-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"text_config":{"model_type":"qwen3_moe"}}', encoding="utf-8")
    assert REASONING_PARSERS.resolve("auto", str(checkpoint)) is not None
