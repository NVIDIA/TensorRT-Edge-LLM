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
"""CPU tests for the INT8 runtime embedding sidecar."""

import importlib.util
import inspect
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _load_repo_module(relative_path, module_name):
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(module_name,
                                                  repo_root / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


embedding_quantization = _load_repo_module(
    "tensorrt_edgellm/checkpoint/embedding_quantization.py",
    "embedding_quantization_under_test")


def test_int8_per_row_quantization_is_deterministic_and_bounded():
    weight = torch.tensor([[0.0, 0.0, 0.0, 0.0], [-2.0, -0.5, 0.5, 2.0],
                           [-0.03, 0.17, 0.61, 1.27]],
                          dtype=torch.float16)

    quantized, scales = embedding_quantization.quantize_embedding_to_int8(
        weight)
    repeated, repeated_scales = \
        embedding_quantization.quantize_embedding_to_int8(weight)

    assert quantized.dtype == torch.int8
    assert scales.dtype == torch.float32
    assert quantized.shape == weight.shape
    assert scales.shape == (weight.shape[0], )
    assert torch.equal(quantized, repeated)
    assert torch.equal(scales, repeated_scales)
    assert scales[0].item() == 1.0
    assert torch.count_nonzero(quantized[0]).item() == 0
    assert quantized.min().item() >= -127
    assert quantized.max().item() <= 127

    dequantized = quantized.float() * scales.unsqueeze(1)
    error = (weight.float() - dequantized).abs()
    error_bound = scales.unsqueeze(1) / 2 + torch.finfo(torch.float32).eps
    assert torch.all(error <= error_bound)


def test_int8_chunked_quantization_matches_single_chunk(monkeypatch):
    weight = torch.tensor([[0.0, -1.0, 1.0, 0.5], [0.25, 0.5, -0.75, 1.0],
                           [8.0, -8.0, 0.125, -0.125]],
                          dtype=torch.float16)
    monkeypatch.setattr(embedding_quantization,
                        "INT8_QUANTIZATION_CHUNK_ELEMENTS", weight.numel())
    expected = embedding_quantization.quantize_embedding_to_int8(weight)
    monkeypatch.setattr(embedding_quantization,
                        "INT8_QUANTIZATION_CHUNK_ELEMENTS", weight.shape[1])
    actual = embedding_quantization.quantize_embedding_to_int8(weight)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_int8_grouped_quantization_uses_independent_scales():
    weight = torch.tensor([[0.1, -0.1, 10.0, -10.0], [0.0, 0.0, 0.5, -0.5]],
                          dtype=torch.float16)

    quantized, scales = embedding_quantization.quantize_embedding_to_int8(
        weight, num_groups=2)

    assert quantized.shape == weight.shape
    assert scales.shape == (2, 2)
    assert scales[0, 0] < scales[0, 1]
    assert scales[1, 0].item() == 1.0
    dequantized = quantized.float().view(2, 2, 2) * scales.unsqueeze(2)
    error = (weight.float().view(2, 2, 2) - dequantized).abs()
    assert torch.all(error <= scales.unsqueeze(2) / 2 +
                     torch.finfo(torch.float32).eps)


def test_int8_grouped_chunked_quantization_matches_single_chunk(monkeypatch):
    weight = torch.tensor([[0.0, -1.0, 1.0, 0.5], [0.25, 0.5, -0.75, 1.0],
                           [8.0, -8.0, 0.125, -0.125]],
                          dtype=torch.float16)
    monkeypatch.setattr(embedding_quantization,
                        "INT8_QUANTIZATION_CHUNK_ELEMENTS", weight.numel())
    expected = embedding_quantization.quantize_embedding_to_int8(weight,
                                                                 num_groups=2)
    monkeypatch.setattr(embedding_quantization,
                        "INT8_QUANTIZATION_CHUNK_ELEMENTS", weight.shape[1])
    actual = embedding_quantization.quantize_embedding_to_int8(weight,
                                                               num_groups=2)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize("num_groups, width, message", [(0, 4, "positive"),
                                                        (2, 3, "divisible")])
def test_int8_quantization_rejects_invalid_groups(num_groups, width, message):
    with pytest.raises(ValueError, match=message):
        embedding_quantization.quantize_embedding_to_int8(
            torch.ones(2, width), num_groups=num_groups)


@pytest.mark.parametrize(
    "weight, message",
    [(torch.ones(2, 3, 4), "2D"), (torch.empty(0, 3), "positive"),
     (torch.ones(2, 3, dtype=torch.int32), "floating-point"),
     (torch.tensor([[math.inf, 0.0]]), "non-finite"),
     (torch.tensor([[math.nan, 0.0]]), "non-finite")])
def test_int8_quantization_rejects_invalid_tables(weight, message):
    with pytest.raises(ValueError, match=message):
        embedding_quantization.quantize_embedding_to_int8(weight)


@pytest.mark.parametrize(
    "tensor_role, value_name, scale_name, num_groups, scale_shape, fmt",
    [("embedding", "embedding", "embedding_scale", 1,
      (2, ), "int8_symmetric_per_row"),
     ("ple_embedding", "weight", "weight_scale", 3,
      (2, 3), "int8_symmetric_column_groups_per_row")])
def test_int8_sidecar_file_contract(tmp_path, tensor_role, value_name,
                                    scale_name, num_groups, scale_shape, fmt):
    weight = torch.tensor([[0.0, -1.0, 1.0, 0.5, -0.5, 0.25],
                           [0.25, 0.5, 0.75, -0.25, -0.5, -0.75]],
                          dtype=torch.float16)
    quantized, scales = embedding_quantization.quantize_embedding_to_int8(
        weight, num_groups=num_groups)
    path = tmp_path / f"{tensor_role}.safetensors"
    metadata = embedding_quantization.int8_sidecar_metadata(tensor_role)

    save_file({
        value_name: quantized,
        scale_name: scales
    },
              str(path),
              metadata=metadata)

    with safe_open(path, framework="pt", device="cpu") as sidecar:
        assert set(sidecar.keys()) == {value_name, scale_name}
        assert sidecar.get_tensor(value_name).dtype == torch.int8
        assert sidecar.get_tensor(scale_name).dtype == torch.float32
        assert sidecar.get_tensor(scale_name).shape == scale_shape
        assert sidecar.metadata() == metadata
        assert sidecar.metadata()["trt_edge_llm_quantization"] == fmt


def test_int8_sidecar_metadata_rejects_unknown_role():
    with pytest.raises(ValueError, match="tensor role"):
        embedding_quantization.int8_sidecar_metadata("unknown")


def _writer_under_test(monkeypatch, reduced_vocab_calls=None):
    reduced_vocab_calls = [] if reduced_vocab_calls is None else reduced_vocab_calls
    checkpoint_utils = _load_repo_module(
        "tensorrt_edgellm/checkpoint/checkpoint_utils.py",
        "checkpoint_utils_writer_under_test")
    monkeypatch.setattr(checkpoint_utils, "build_runtime_llm_config_dict",
                        lambda *_args, **_kwargs: {})

    package = ModuleType("tensorrt_edgellm")
    package.__path__ = []
    checkpoint_package = ModuleType("tensorrt_edgellm.checkpoint")
    checkpoint_package.__path__ = []
    checkpoint_package.embedding_quantization = embedding_quantization
    safetensors_io = ModuleType("tensorrt_edgellm._safetensors_io")
    safetensors_io.save_file = save_file
    chat_template = ModuleType("tensorrt_edgellm.chat_template")
    chat_template.write_chat_template = lambda *_args: None
    vocab_package = ModuleType("tensorrt_edgellm.vocab_reduction")
    vocab_package.__path__ = []
    vocab_export = ModuleType("tensorrt_edgellm.vocab_reduction.onnx_export")
    vocab_export.copy_reduced_vocab_artifacts = (
        lambda *args: reduced_vocab_calls.append(args))

    monkeypatch.setattr(checkpoint_utils, "__package__",
                        "tensorrt_edgellm.checkpoint")
    for name, module in {
            "tensorrt_edgellm": package,
            "tensorrt_edgellm._safetensors_io": safetensors_io,
            "tensorrt_edgellm.chat_template": chat_template,
            "tensorrt_edgellm.checkpoint": checkpoint_package,
            "tensorrt_edgellm.checkpoint.embedding_quantization":
            embedding_quantization,
            "tensorrt_edgellm.vocab_reduction": vocab_package,
            "tensorrt_edgellm.vocab_reduction.onnx_export": vocab_export,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    return checkpoint_utils


def _spec_flags():
    return {
        name: False
        for name in ("is_eagle3_draft", "is_dflash_draft", "is_jetspec_draft",
                     "is_dspark_draft", "is_mtp_draft", "gemma4_mtp_draft",
                     "eagle_base", "dflash_base", "jetspec_base",
                     "dspark_base", "mtp_base", "gemma4_mtp_base")
    }


def _llama_model():
    embedding = torch.tensor([[0.0, -1.0, 1.0, 0.5], [0.25, 0.5, -0.75, 1.0]],
                             dtype=torch.float16)
    config = SimpleNamespace(hidden_size=4,
                             model_type="llama",
                             ple_enabled=False,
                             **_spec_flags())
    return SimpleNamespace(config=config,
                           embed_tokens=SimpleNamespace(weight=embedding))


def test_runtime_artifact_writer_exports_int8_embedding(tmp_path, monkeypatch):
    checkpoint_utils = _writer_under_test(monkeypatch)
    out_dir = tmp_path / "artifacts"

    checkpoint_utils.write_runtime_artifacts(_llama_model(),
                                             str(tmp_path),
                                             str(out_dir),
                                             int8_embedding=True)

    with safe_open(out_dir / "embedding.safetensors",
                   framework="pt",
                   device="cpu") as sidecar:
        assert set(sidecar.keys()) == {"embedding", "embedding_scale"}
        assert sidecar.get_tensor("embedding").dtype == torch.int8
        assert sidecar.get_tensor("embedding_scale").shape == (2, )
        assert sidecar.metadata()["trt_edge_llm_tensor_role"] == "embedding"


def _gemma4_model():
    embedding = torch.tensor([[0.0, -1.0, 1.0, 0.5], [0.25, 0.5, -0.75, 1.0]],
                             dtype=torch.float16)
    ple = torch.tensor([[0.0, -1.0, 8.0, -8.0], [0.25, 0.5, -2.0, 2.0]],
                       dtype=torch.float16)
    config = SimpleNamespace(hidden_size=4,
                             hidden_size_per_layer_input=2,
                             model_type="gemma4_text",
                             num_hidden_layers=2,
                             ple_enabled=True,
                             **_spec_flags())
    return SimpleNamespace(
        config=config,
        embed_tokens=SimpleNamespace(weight=embedding),
        model=SimpleNamespace(embed_tokens_per_layer=SimpleNamespace(
            weight=ple)),
    )


def test_runtime_artifact_writer_exports_int8_embedding_and_ple(
        tmp_path, monkeypatch):
    checkpoint_utils = _writer_under_test(monkeypatch)
    out_dir = tmp_path / "artifacts"

    checkpoint_utils.write_runtime_artifacts(_gemma4_model(),
                                             str(tmp_path),
                                             str(out_dir),
                                             int8_embedding=True)

    with safe_open(out_dir / "embedding.safetensors",
                   framework="pt",
                   device="cpu") as sidecar:
        assert sidecar.get_tensor("embedding").dtype == torch.int8
        assert sidecar.get_tensor("embedding_scale").shape == (2, )
    with safe_open(out_dir / "ple_embedding.safetensors",
                   framework="pt",
                   device="cpu") as sidecar:
        assert set(sidecar.keys()) == {"weight", "weight_scale"}
        assert sidecar.get_tensor("weight").dtype == torch.int8
        assert sidecar.get_tensor("weight_scale").dtype == torch.float32
        assert sidecar.get_tensor("weight_scale").shape == (2, 2)
        assert sidecar.metadata(
        )["trt_edge_llm_tensor_role"] == "ple_embedding"
        assert sidecar.metadata()["trt_edge_llm_quantization"] == \
            "int8_symmetric_column_groups_per_row"


def test_runtime_artifact_writer_writes_2d_ple_scales_for_one_layer(
        tmp_path, monkeypatch):
    checkpoint_utils = _writer_under_test(monkeypatch)
    model = _gemma4_model()
    model.config.num_hidden_layers = 1
    model.config.hidden_size_per_layer_input = 4
    out_dir = tmp_path / "artifacts"

    checkpoint_utils.write_runtime_artifacts(model,
                                             str(tmp_path),
                                             str(out_dir),
                                             int8_embedding=True)

    with safe_open(out_dir / "ple_embedding.safetensors",
                   framework="pt",
                   device="cpu") as sidecar:
        assert sidecar.get_tensor("weight").shape == (2, 4)
        assert sidecar.get_tensor("weight_scale").shape == (2, 1)


def test_runtime_artifact_writer_keeps_fp16_ple_without_flag(
        tmp_path, monkeypatch):
    checkpoint_utils = _writer_under_test(monkeypatch)
    out_dir = tmp_path / "artifacts"

    checkpoint_utils.write_runtime_artifacts(_gemma4_model(), str(tmp_path),
                                             str(out_dir))

    with safe_open(out_dir / "ple_embedding.safetensors",
                   framework="pt",
                   device="cpu") as sidecar:
        assert set(sidecar.keys()) == {"weight"}
        assert sidecar.get_tensor("weight").dtype == torch.float16
        assert not sidecar.metadata()


def test_runtime_artifact_writer_keeps_positional_arguments(
        tmp_path, monkeypatch):
    reduced_vocab_calls = []
    checkpoint_utils = _writer_under_test(monkeypatch, reduced_vocab_calls)
    out_dir = tmp_path / "artifacts"

    # A caller written before ``int8_embedding`` existed: reduced_vocab_dir is
    # the fifth positional argument and must keep that meaning.
    checkpoint_utils.write_runtime_artifacts(_llama_model(), str(tmp_path),
                                             str(out_dir), False, "reduced")

    assert reduced_vocab_calls and reduced_vocab_calls[0][2] == "reduced"
    with safe_open(out_dir / "embedding.safetensors",
                   framework="pt",
                   device="cpu") as sidecar:
        assert set(sidecar.keys()) == {"embedding"}
        assert sidecar.get_tensor("embedding").dtype == torch.float16


def test_quantize_embedding_to_int8_num_groups_is_keyword_only():
    parameter = inspect.signature(
        embedding_quantization.quantize_embedding_to_int8
    ).parameters["num_groups"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 1


def test_embedding_quantization_modes_are_mutually_exclusive():
    checkpoint_utils = _load_repo_module(
        "tensorrt_edgellm/checkpoint/checkpoint_utils.py",
        "checkpoint_utils_under_test")

    with pytest.raises(ValueError, match="mutually exclusive"):
        checkpoint_utils.write_runtime_artifacts(None,
                                                 "",
                                                 "",
                                                 fp8_embedding=True,
                                                 int8_embedding=True)


def test_int8_runtime_artifacts_reject_qwen3_omni():
    checkpoint_utils = _load_repo_module(
        "tensorrt_edgellm/checkpoint/checkpoint_utils.py",
        "checkpoint_utils_omni_under_test")
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_omni"))

    with pytest.raises(ValueError, match="talker runtimes"):
        checkpoint_utils.write_runtime_artifacts(model,
                                                 "",
                                                 "",
                                                 int8_embedding=True)
