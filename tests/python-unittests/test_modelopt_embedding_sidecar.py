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

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tensorrt_edgellm.checkpoint.embedding_quantization import \
    write_modelopt_embedding


@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
def test_export_fp8_uses_one_scale_without_block_alignment(dtype):
    from tensorrt_edgellm.checkpoint.embedding_quantization import \
        quantize_embedding_to_fp8
    weight = torch.linspace(-3, 2, 17 * 24).reshape(17, 24).to(dtype)
    packed, scale = quantize_embedding_to_fp8(weight)
    assert scale.shape == (1, 1) and scale.dtype == torch.float32
    torch.testing.assert_close(scale, torch.tensor([[3.0 / 448.0]]))
    expected = (weight.float() / scale).to(torch.float8_e4m3fn)
    assert torch.equal(packed.view(torch.uint8), expected.view(torch.uint8))


def test_checkpoint_direct_builder_writes_modelopt_fp8_embedding(tmp_path):
    """The direct builder keeps FP8 token rows and their scalar scale."""
    from tensorrt_edgellm.builder.core.artifacts.embeddings import \
        write_embedding
    from tensorrt_edgellm.builder.core.weights import Weights

    key = "model.embed_tokens.weight"
    save_file(
        {
            key: torch.ones(256, 128).to(torch.float8_e4m3fn),
            key + "_scale": torch.tensor(2.0),
        }, str(tmp_path / "model.safetensors"))
    weights = Weights(str(tmp_path))
    try:
        write_embedding(weights, SimpleNamespace(embedding_scale=4.0),
                        SimpleNamespace(fp8_embedding=False), str(tmp_path))
    finally:
        weights.close()
    with safe_open(tmp_path / "embedding.safetensors", framework="pt") as file:
        assert file.get_tensor("embedding").dtype == torch.float8_e4m3fn
        torch.testing.assert_close(file.get_tensor("embedding_scale"),
                                   torch.tensor([[8.0]]))


def test_direct_export_fp8_matches_common_packer(tmp_path):
    from tensorrt_edgellm.builder.core.artifacts.embeddings import \
        write_embedding
    from tensorrt_edgellm.builder.core.weights import Weights
    from tensorrt_edgellm.checkpoint.embedding_quantization import \
        quantize_embedding_to_fp8

    weight = torch.linspace(-3, 2, 17 * 24).reshape(17, 24).half()
    save_file({"model.embed_tokens.weight": weight},
              str(tmp_path / "model.safetensors"))
    weights = Weights(str(tmp_path))
    multiplier = 1.234567
    try:
        write_embedding(weights, SimpleNamespace(embedding_scale=multiplier),
                        SimpleNamespace(fp8_embedding=True), str(tmp_path))
    finally:
        weights.close()
    expected, scale = quantize_embedding_to_fp8(weight)
    with safe_open(tmp_path / "embedding.safetensors", framework="pt") as file:
        torch.testing.assert_close(file.get_tensor("embedding").float(),
                                   expected.float(),
                                   rtol=0,
                                   atol=0)
        torch.testing.assert_close(file.get_tensor("embedding_scale"),
                                   scale * multiplier,
                                   rtol=0,
                                   atol=0)


@pytest.mark.parametrize("format", ["fp8", "nvfp4"])
def test_modelopt_embedding_sidecar(tmp_path, format):
    embedding = torch.nn.Embedding(256, 128)
    prefix = "model.embed_tokens.weight"
    if format == "fp8":
        tensors = {
            prefix: torch.ones(256, 128).to(torch.float8_e4m3fn),
            prefix + "_scale": torch.tensor(2.0),
        }
    else:
        tensors = {
            prefix: torch.full((256, 64), 0x11, dtype=torch.uint8),
            prefix + "_scale": torch.ones(256, 8).to(torch.float8_e4m3fn),
            prefix + "_scale_2": torch.tensor(0.25),
        }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    output = tmp_path / "embedding.safetensors"
    assert write_modelopt_embedding(str(tmp_path), embedding, 4.0, str(output))

    with safe_open(output, framework="pt", device="cpu") as file:
        if format == "fp8":
            assert file.get_tensor("embedding").shape == (256, 128)
            torch.testing.assert_close(file.get_tensor("embedding_scale"),
                                       torch.tensor([[8.0]]))
        else:
            assert file.get_tensor("embedding").shape == (256, 64)
            assert file.get_tensor("embedding_block_scale").shape == (256, 8)
            torch.testing.assert_close(
                file.get_tensor("embedding_multiplier_scale"),
                torch.tensor([0.25]))
            descriptor = json.loads(
                file.metadata()["tensorrt_edge_llm.embedding_descriptor"])
            assert descriptor == {
                "logical_shape": [256, 128],
                "group_size": 16,
                "model_output_scale": 4.0
            }
