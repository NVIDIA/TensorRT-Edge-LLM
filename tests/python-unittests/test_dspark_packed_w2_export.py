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

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_edgellm.onnx.cli import _load_dspark_head_tensors


@pytest.mark.parametrize(
    "scale_dtype",
    [torch.float8_e4m3fn, torch.int8, torch.uint8, torch.float32])
def test_optional_packed_w2_preserves_dense_export(tmp_path, scale_dtype):
    # Given equivalent scale encodings, when exporting, then dense values and packed bytes agree.
    weight = torch.arange(32, dtype=torch.uint8).reshape(2, 16)
    scales = torch.ones((2, 2)).to(torch.float8_e4m3fn)
    scales = (scales.view(scale_dtype) if scale_dtype
              in (torch.int8, torch.uint8) else scales.to(scale_dtype))
    save_file(
        {
            "markov_head.markov_w2.weight": weight,
            "markov_head.markov_w2.weight_scale": scales,
            "markov_head.markov_w2.weight_scale_2": torch.tensor([0.75]),
        },
        str(tmp_path / "model.safetensors"),
    )
    reference, _ = _load_dspark_head_tensors(str(tmp_path), {"markov_w2"})
    packed = {}
    actual, _ = _load_dspark_head_tensors(str(tmp_path), {"markov_w2"},
                                          packed_w2=packed)
    torch.testing.assert_close(actual["markov_w2"],
                               reference["markov_w2"],
                               rtol=0,
                               atol=0)
    from tensorrt_edgellm.models.ops import _nvfp4_dequantize_eager

    expected = _nvfp4_dequantize_eager(
        weight,
        torch.ones((2, 2)).to(torch.float8_e4m3fn), torch.tensor([0.75]), 16)
    torch.testing.assert_close(actual["markov_w2"], expected, rtol=0, atol=0)
    if scale_dtype == torch.float32:
        assert not packed
    else:
        assert torch.equal(packed["packed_weight"], weight)
        assert torch.equal(packed["block_scales"], scales.view(torch.uint8))
        assert packed["global_scale"].shape == (1, )


@pytest.mark.parametrize("quantization", [{}, {
    "quant_algo": "NVFP4",
    "group_size": 16
}])
def test_dense_reexport_removes_stale_packed_w2(tmp_path, quantization):
    # Given a stale packed sidecar, when exporting dense heads, then it is removed.
    from tensorrt_edgellm.onnx import cli as export

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({
            "dspark_config": {
                "markov_head_type": "vanilla",
                "markov_rank": 32
            },
            "quantization_config": quantization
        }))
    save_file(
        {
            "markov_head.markov_w1.weight":
            torch.ones((2, 32), dtype=torch.float16),
            "markov_head.markov_w2.weight":
            torch.ones((2, 32), dtype=torch.float16),
        },
        str(checkpoint / "model.safetensors"),
    )
    stale = output / "dspark_w2_nvfp4.safetensors"
    stale.write_bytes(b"previous checkpoint packed weights")
    export._export_dspark_sidecars(str(checkpoint), str(output))
    assert not stale.exists()
    assert (output / "dspark_heads.safetensors").is_file()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_global_scale_rejected(tmp_path, value):
    # Given an invalid global scale, when loading heads, then export fails explicitly.
    save_file(
        {
            "markov_head.markov_w2.weight":
            torch.zeros((2, 16), dtype=torch.uint8),
            "markov_head.markov_w2.weight_scale":
            torch.ones((2, 2)).to(torch.float8_e4m3fn),
            "markov_head.markov_w2.weight_scale_2":
            torch.tensor([value]),
        }, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="finite positive scalar"):
        _load_dspark_head_tensors(str(tmp_path), {"markov_w2"}, packed_w2={})


@pytest.mark.parametrize("scale", [
    torch.tensor([1.0, 2.0]),
    torch.tensor([1], dtype=torch.int32),
    torch.tensor([1e100], dtype=torch.float64),
    torch.tensor([1e-100], dtype=torch.float64)
])
def test_global_scale_contract_rejected(tmp_path, scale):
    # Given an invalid shape, dtype, or FP32 range, when exporting, then the checkpoint is rejected.
    save_file(
        {
            "markov_head.markov_w2.weight":
            torch.zeros((2, 16), dtype=torch.uint8),
            "markov_head.markov_w2.weight_scale":
            torch.ones((2, 2)).to(torch.float8_e4m3fn),
            "markov_head.markov_w2.weight_scale_2":
            scale,
        }, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="finite positive scalar"):
        _load_dspark_head_tensors(str(tmp_path), {"markov_w2"}, packed_w2={})


def test_native_nvfp4_export_does_not_dequantize_w2(tmp_path, monkeypatch):
    # Given native NVFP4 W2, when exporting, then retain bytes without dense W2 conversion.
    from safetensors.torch import load_file

    from tensorrt_edgellm.models import ops
    from tensorrt_edgellm.onnx import cli as export

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({
            "dspark_config": {
                "markov_head_type": "vanilla",
                "markov_rank": 32
            },
            "quantization_config": {
                "quant_algo": "NVFP4",
                "group_size": 16
            }
        }))
    weight = torch.arange(32, dtype=torch.uint8).reshape(2, 16)
    save_file(
        {
            "markov_head.markov_w1.weight":
            torch.ones((2, 32), dtype=torch.float16),
            "markov_head.markov_w2.weight":
            weight,
            "markov_head.markov_w2.weight_scale":
            torch.ones((2, 2)).to(torch.float8_e4m3fn),
            "markov_head.markov_w2.weight_scale_2":
            torch.tensor([0.75]),
        }, str(checkpoint / "model.safetensors"))

    def reject_dequantization(*args, **kwargs):
        raise AssertionError("Native W2 must remain packed during export")

    monkeypatch.setattr(ops, "nvfp4_dequantize", reject_dequantization)
    export._export_dspark_sidecars(str(checkpoint), str(output))
    heads = load_file(str(output / "dspark_heads.safetensors"))
    assert set(heads) == {"markov_w1"}
    packed = load_file(str(output / "dspark_w2_nvfp4.safetensors"))
    assert torch.equal(packed["packed_weight"], weight)
    info = json.loads((output / "dspark_heads_info.json").read_text())
    assert info["w2_format"] == "nvfp4"


@pytest.mark.parametrize("shape", [(2, 17), (0, 16), (2, 0), (32, )])
def test_native_packed_shape_rejected(tmp_path, shape):
    # Given malformed packed dimensions, when preserving native W2, then export rejects them.
    save_file(
        {
            "markov_head.markov_w2.weight":
            torch.zeros(shape, dtype=torch.uint8),
            "markov_head.markov_w2.weight_scale":
            torch.ones((2, 2)).to(torch.float8_e4m3fn),
            "markov_head.markov_w2.weight_scale_2":
            torch.tensor([0.75]),
        }, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="Invalid packed DSpark head shape"):
        _load_dspark_head_tensors(str(tmp_path), {"markov_w2"},
                                  packed_w2={},
                                  preserve_packed_w2=True)
