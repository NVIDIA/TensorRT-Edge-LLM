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
"""AWQ checkpoint correction in the exporter linear layer."""

import numpy as np
import pytest
import torch

from experimental.builder.core import quantization
from experimental.builder.core.weights import ParameterSpec, Weights
from experimental.builder.weight_packing import int4
from tensorrt_edgellm.checkpoint import repacking
from tensorrt_edgellm.models import linear
from tensorrt_edgellm.onnx import export as onnx_export


def _pack(values):
    order = (0, 2, 4, 6, 1, 3, 5, 7)
    result = torch.zeros(values.shape[0],
                         values.shape[1] // 8,
                         dtype=torch.int32)
    for bit, channel in enumerate(order):
        result |= values[:, channel::8].to(torch.int32) << (4 * bit)
    return result


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("shape", [(3, 64), (2, 3, 64)])
def test_awq_export_linear_applies_zero_correction(monkeypatch, version,
                                                   shape):
    generator = torch.Generator().manual_seed(456)
    weights = torch.randint(0,
                            16, (64, 64),
                            generator=generator,
                            dtype=torch.int32)
    zeros = torch.randint(0,
                          16, (2, 64),
                          generator=generator,
                          dtype=torch.int32)
    scales = torch.rand(2, 64, generator=generator,
                        dtype=torch.float32).half() / 16
    x = torch.randn(*shape, generator=generator, dtype=torch.float32).half()
    layer = linear.AWQLinear(64, 64, group_size=32)
    layer.qweight = _pack(weights)
    layer.qzeros = _pack(zeros)
    layer.scales = scales
    monkeypatch.setattr(repacking, "int4_gemm_plugin_version", lambda: version)
    repacking._repack_awq_weights(layer)

    expected_correction = (8 - zeros).float() * scales.float()
    torch.testing.assert_close(layer.zero_correction,
                               expected_correction,
                               rtol=0,
                               atol=0)
    expanded_scales = scales.float().repeat_interleave(32, dim=0)
    centered_weights = (weights.float() - 8) * expanded_scales

    def centered_gemm(inputs, *args):
        return torch.matmul(inputs.float(), centered_weights).half()

    monkeypatch.setattr(linear, "_int4_groupwise_gemm_dispatch", centered_gemm)
    actual = layer(x)
    original_weights = (
        weights - zeros.repeat_interleave(32, dim=0)).float() * expanded_scales
    expected = torch.matmul(x.float(), original_weights).half()
    torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.005)


@pytest.mark.parametrize("version", [1, 2])
def test_direct_builder_and_exporter_use_identical_awq_representation(
        monkeypatch, version):
    generator = torch.Generator().manual_seed(734)
    weights = torch.randint(0,
                            16, (64, 64),
                            generator=generator,
                            dtype=torch.int32)
    zeros = torch.randint(0,
                          16, (2, 64),
                          generator=generator,
                          dtype=torch.int32)
    scales = torch.rand(2, 64, generator=generator).half()
    qweight = _pack(weights)
    qzeros = _pack(zeros)
    layer = linear.AWQLinear(64, 64, group_size=32)
    layer.qweight, layer.qzeros, layer.scales = qweight.clone(), qzeros.clone(
    ), scales.clone()
    monkeypatch.setattr(repacking, "int4_gemm_plugin_version", lambda: version)
    repacking._repack_awq_weights(layer)
    np.testing.assert_array_equal(
        layer.qweight.numpy(),
        int4.repack_awq(qweight.numpy(),
                        qzeros.numpy(),
                        version,
                        center_at_eight=True))
    np.testing.assert_array_equal(
        layer.zero_correction.numpy(),
        int4.awq_zero_correction(qzeros.numpy(), scales.numpy()))


@pytest.mark.parametrize("name", ["zero_correction", "layer.zero_correction"])
def test_onnx_fixup_preserves_awq_correction_fp32(tmp_path, name):
    import onnx
    from onnx import helper, numpy_helper

    initializer = numpy_helper.from_array(np.ones((2, 64), dtype=np.float32),
                                          name)
    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [3, 2])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [3, 64])
    node = helper.make_node("MatMul", ["x", initializer.name], ["y"])
    model = helper.make_model(
        helper.make_graph([node], "correction", [x], [y], [initializer]))
    path = tmp_path / "awq.onnx"
    onnx.save(model, path)
    onnx_export._fix_initializer_dtypes(str(path))
    result = onnx.load(path)
    assert result.graph.initializer[0].data_type == onnx.TensorProto.FLOAT
    onnx.checker.check_model(result)


@pytest.mark.parametrize("version", [1, 2])
def test_direct_builder_materialized_and_external_awq_recipes(
        tmp_path, version):
    from safetensors.numpy import save_file

    rng = np.random.default_rng(12)
    q = torch.from_numpy(rng.integers(0, 16, (64, 64), dtype=np.int32))
    z = torch.from_numpy(rng.integers(0, 16, (2, 64), dtype=np.int32))
    scales = rng.random((2, 64)).astype(np.float16)
    save_file(
        {
            "layer.qweight": _pack(q).numpy(),
            "layer.qzeros": _pack(z).numpy(),
            "layer.scales": scales
        }, str(tmp_path / "model.safetensors"))
    quant = quantization.QuantConfig(quant_type=quantization.QUANT_INT4_AWQ,
                                     group_size=32)
    weights = Weights(str(tmp_path),
                      quant=quant,
                      int4_gemm_plugin_version=version)
    try:
        actual = weights.linear("layer", quantization.QUANT_INT4_AWQ)
        metadata = weights.linear_metadata("layer",
                                           quantization.QUANT_INT4_AWQ)
        assert isinstance(metadata.zero_correction, ParameterSpec)
        assert metadata.zero_correction.shape == (2, 64)
        assert metadata.zero_correction.dtype == np.dtype(np.float32)
        assert metadata.weight_recipe["assemble"] == "awq_ffn_qweight_center8"
        assert metadata.zero_correction_recipe[
            "assemble"] == "awq_zero_correction"
        assert actual.weight_recipe["assemble"] == metadata.weight_recipe[
            "assemble"]
        np.testing.assert_array_equal(actual.zero_correction,
                                      (8 - z.numpy()).astype(np.float32) *
                                      scales.astype(np.float32))
        with pytest.raises(NotImplementedError, match="tensor-parallel"):
            Weights.shard_linear(metadata, "column", 2, 0)
    finally:
        weights.close()
