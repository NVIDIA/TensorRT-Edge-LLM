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

import re
from pathlib import Path

from conftest import EnvironmentConfig
from defs import config as config_module
from defs import test_checkpoint_export as checkpoint_export

_PARAM = "gemma-4-E2B-it-fp16-fp8kv"


def _checkpoint(path: Path, *, quantized: bool = False) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").touch()
    if quantized:
        (path / "hf_quant_config.json").write_text("{}", encoding="utf-8")


def test_gemma4_fp8_kv_l1_quantizes_then_exports_generated_checkpoint(
        tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    base_checkpoint = model_root / "gemma-4-E2B-it"
    _checkpoint(base_checkpoint)
    data_root = tmp_path / "data"
    dataset_dir = model_root / "datasets" / "cnn_dailymail"
    dataset_dir.mkdir(parents=True)

    onnx_dir = tmp_path / "onnx"
    test_cache_dir = tmp_path / "test-cache"
    test_cache_dir.mkdir()
    env = EnvironmentConfig(
        llm_sdk_dir=str(Path(__file__).resolve().parents[2]),
        llm_models_dir=str(model_root),
        edgellm_data_dir=str(data_root),
        onnx_dir=str(onnx_dir),
        engine_dir=None,
        build_dir="build",
        test_log_dir=str(tmp_path / "logs"),
        trt_package_dir=None,
    )
    monkeypatch.setenv("TEST_CACHE_DIR", str(test_cache_dir))
    monkeypatch.delenv("QUANT_CHECKPOINT_DIR", raising=False)

    inventory_resolutions = []
    original_resolve_checkpoint = config_module.resolve_checkpoint

    def _resolve_checkpoint(*args, **kwargs):
        inventory_resolutions.append((args, kwargs))
        assert kwargs.get("quantized_variant") is None
        return original_resolve_checkpoint(*args, **kwargs)

    calls = []
    generated_checkpoint = None

    def _run_command_list(commands, label, test_logger, env_vars=None):
        nonlocal generated_checkpoint
        del test_logger, env_vars
        calls.append("quantize")
        assert label == "Pre-export"
        assert len(commands) == 1
        quantize_shell = commands[0][0][2]
        assert f"--model_dir={base_checkpoint}" in quantize_shell
        assert "--kv_cache_quantization=fp8" in quantize_shell
        output_match = re.search(r"--output_dir=([^ ]+)", quantize_shell)
        assert output_match
        generated_checkpoint = Path(output_match.group(1))
        assert generated_checkpoint.parent.parent == test_cache_dir
        assert generated_checkpoint.name == "gemma-4-E2B-it-FP8-KV"
        _checkpoint(generated_checkpoint, quantized=True)

    exported = {}

    def _run_checkpoint_export(config, test_logger, model_dir=None, **_kwargs):
        del test_logger
        calls.append("export")
        exported["model_dir"] = model_dir
        exported["source_exists"] = Path(model_dir).is_dir()
        exported["onnx_dir"] = config.get_llm_onnx_dir()

    monkeypatch.setattr(config_module, "resolve_checkpoint",
                        _resolve_checkpoint)
    monkeypatch.setattr(checkpoint_export, "run_command_list",
                        _run_command_list)
    monkeypatch.setattr(checkpoint_export, "run_checkpoint_export",
                        _run_checkpoint_export)

    checkpoint_export.test_checkpoint_export_with_quantization(
        _PARAM, None, env)

    assert inventory_resolutions
    assert calls == ["quantize", "export"]
    assert exported["model_dir"] == str(generated_checkpoint)
    assert exported["source_exists"]
    assert exported["onnx_dir"] == str(onnx_dir / "gemma-4-E2B-it" /
                                       "llm-fp16-fp16-fp8kv")
    assert not generated_checkpoint.exists()
