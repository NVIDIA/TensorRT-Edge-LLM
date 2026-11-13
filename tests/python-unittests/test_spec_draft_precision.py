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
import onnx
import pytest
from defs import ckpt_layout_validator
from defs import config as defs_config


def _write_checkpoint(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"")


def _export_config(param, root):
    env = conftest.EnvironmentConfig(llm_sdk_dir=".",
                                     llm_models_dir=str(root / "models"),
                                     edgellm_data_dir=str(root / "data"),
                                     onnx_dir=str(root / "onnx"),
                                     engine_dir=None,
                                     build_dir=str(root / "build"),
                                     test_log_dir=str(root / "logs"),
                                     trt_package_dir=None)
    return defs_config.TestConfig.from_param_string(
        param,
        defs_config.ModelType.LLM,
        defs_config.TaskType.EXPORT,
        env,
        validate_environment=False)


def test_quantized_draft_export_reads_prequantized_checkpoint(
        tmp_path, monkeypatch):
    """A quantized Domino draft name resolves to its pre-quantized checkpoint, not the fp16 draft."""
    monkeypatch.setenv("QUANT_CHECKPOINT_DIR", str(tmp_path / "quant"))
    monkeypatch.setenv("HF_MODELS_DIR", str(tmp_path / "hf"))
    _write_checkpoint(tmp_path / "models/Qwen3-8B-Domino-b16")
    expected = tmp_path / "models/Qwen3-8B-Domino-b16-NVFP4"
    _write_checkpoint(expected)

    resolved = _export_config("Qwen3-8B-NVFP4-nvfp4-domino-b16-nvfp4",
                              tmp_path).get_block_draft_checkpoint_dir()

    assert resolved == str(expected)


def test_draft_onnx_layout_rejects_fp16_decoder(tmp_path):
    """An fp16 decoder exported under an nvfp4 draft name fails."""
    # Exported drafts list the fp16 ``fc`` ahead of the decoder layers.
    names = ["_model.fc", "_model.layers.0.mlp.down_proj", "_model.lm_head"]
    graph = onnx.helper.make_graph(
        [],
        "draft", [], [],
        initializer=[
            onnx.helper.make_tensor(f"{name}.weight", onnx.TensorProto.FLOAT16,
                                    [2, 2], [0, 0, 0, 0]) for name in names
        ])
    onnx_path = str(tmp_path / "model.onnx")
    onnx.save(onnx.helper.make_model(graph), onnx_path)

    with pytest.raises(AssertionError, match=r"example module: \S+_proj"):
        ckpt_layout_validator.validate_onnx_layout(onnx_path,
                                                   body_precision="nvfp4",
                                                   lm_head_precision="fp16")
