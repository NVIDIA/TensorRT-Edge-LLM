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

import conftest
import pytest
import torch
from defs import ckpt_layout_validator
from defs import config as defs_config
from defs.utils import checkpoint_export_helpers, command_generation
from safetensors import torch as safetensors_torch

_TOKEN_PARAM = "gemma-4-E2B-it-nvfp4-tokembnvfp4-mxsl2048-mxbs1-mxil1024"
_PLE_PARAM = "gemma-4-E2B-it-nvfp4-tokembfp8-plefp8-mxsl2048-mxbs1-mxil1024"


@pytest.fixture(autouse=True)
def _offline_calibration_dataset(monkeypatch):
    monkeypatch.setattr(defs_config.TestConfig,
                        "get_cnn_dailymail_dataset_dir",
                        lambda self: "/data/cnn_dailymail")


def _config(param, task_type=defs_config.TaskType.BUILD):
    env = conftest.EnvironmentConfig(llm_sdk_dir=".",
                                     llm_models_dir="/models",
                                     edgellm_data_dir="/data",
                                     onnx_dir="/onnx",
                                     engine_dir="/engine",
                                     build_dir="/build",
                                     test_log_dir="/logs",
                                     trt_package_dir="/trt")
    return defs_config.TestConfig.from_param_string(param,
                                                    defs_config.ModelType.LLM,
                                                    task_type,
                                                    env,
                                                    validate_environment=False)


def test_embedding_precision_selects_its_own_onnx_directory():
    """Exports that differ only in embedding precision do not share an ONNX tree."""
    plain = _config(_TOKEN_PARAM.replace("-tokembnvfp4", ""))
    token = _config(_TOKEN_PARAM)
    ple = _config(_PLE_PARAM)

    assert len({c.get_llm_onnx_dir() for c in (plain, token, ple)}) == 3


def test_quantize_command_carries_embedding_flags_for_fp16_backbone():
    """An FP16 backbone still quantizes when only embedding tables are requested."""
    config = _config("gemma-4-E2B-it-fp16-tokembfp8-plefp8",
                     defs_config.TaskType.EXPORT)

    commands = command_generation._generate_quantization_commands(config)

    assert len(commands) == 1
    shell = commands[0][0][-1]
    assert "--token_embedding_quantization=fp8" in shell
    assert "--ple_embedding_quantization=fp8" in shell
    assert "--quantization" not in shell
    assert "--output_dir=/onnx/gemma-4-E2B-it-TOKEMBFP8-PLEFP8" in shell


def test_export_reads_the_directory_the_embedding_quantization_wrote():
    """The exporter consumes the checkpoint that the quantize step produced."""
    config = _config("gemma-4-E2B-it-fp16-tokembfp8",
                     defs_config.TaskType.EXPORT)

    shell = command_generation._generate_quantization_commands(
        config)[0][0][-1]

    export_dir = checkpoint_export_helpers.get_export_model_dir(config)
    assert f"--output_dir={export_dir}" in shell


def _write_checkpoint(path, embed_dtype, ple_dtype=None):
    tensors = {
        "model.embed_tokens.weight":
        torch.zeros(4, 16, dtype=embed_dtype),
        "model.layers.0.self_attn.q_proj.weight":
        torch.zeros(4, 16, dtype=torch.float16),
    }
    if embed_dtype == torch.float8_e4m3fn:
        tensors["model.embed_tokens.weight_scale"] = torch.ones(
            1, dtype=torch.float32)
    if ple_dtype is not None:
        tensors["model.embed_tokens_per_layer.weight"] = torch.zeros(
            4, 16, dtype=ple_dtype)
    safetensors_torch.save_file(tensors, str(path / "model.safetensors"))


def test_layout_validator_accepts_matching_token_and_ple_precision(tmp_path):
    """A checkpoint whose token and PLE tables hold the requested dtype passes."""
    _write_checkpoint(tmp_path, torch.float8_e4m3fn, torch.float8_e4m3fn)

    ckpt_layout_validator.validate_ckpt_layout(str(tmp_path),
                                               body_precision="fp16",
                                               token_embedding_precision="fp8",
                                               ple_embedding_precision="fp8")


def test_layout_validator_rejects_unquantized_token_table(tmp_path):
    """A requested token-table quantization that left the table in FP16 fails."""
    _write_checkpoint(tmp_path, torch.float16)

    with pytest.raises(AssertionError, match="embed"):
        ckpt_layout_validator.validate_ckpt_layout(
            str(tmp_path),
            body_precision="fp16",
            token_embedding_precision="fp8")


def test_layout_validator_rejects_unquantized_ple_table(tmp_path):
    """A requested PLE quantization that left the table in FP16 fails."""
    _write_checkpoint(tmp_path, torch.float16, torch.float16)

    with pytest.raises(AssertionError, match="ple"):
        ckpt_layout_validator.validate_ckpt_layout(
            str(tmp_path),
            body_precision="fp16",
            ple_embedding_precision="fp8")
