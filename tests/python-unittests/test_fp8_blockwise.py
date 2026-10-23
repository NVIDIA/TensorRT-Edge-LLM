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

from types import SimpleNamespace

import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper
from torch._subclasses.fake_tensor import FakeTensorMode

from tensorrt_edgellm.config import (QUANT_FP8, QUANT_FP8_BLOCK, QuantConfig,
                                     _parse_quant)
from tensorrt_edgellm.models.default.modeling_default import (
    Attention, fuse_qkv_projections)
from tensorrt_edgellm.models.linear import (FP8BlockLinear, FP8Linear, TPMode,
                                            make_linear)
from tensorrt_edgellm.models.ops import fp8_block_gemm
from tensorrt_edgellm.onnx.dynamo_translations import \
    build_custom_translation_table
from tensorrt_edgellm.onnx.export import _fix_initializer_dtypes
from tensorrt_edgellm.onnx.onnx_custom_schemas import \
    register_tensorrt_edgellm_onnx_custom_schemas


def test_fp8_block_config_detection(tmp_path):
    block_config = {
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
    }
    block_quant = _parse_quant(str(tmp_path), block_config)
    assert block_quant.quant_type == QUANT_FP8_BLOCK
    assert block_quant.weight_block_size == [128, 128]
    assert block_quant.uses_fp8_block_weights

    tensor_quant = _parse_quant(
        str(tmp_path), {"quantization_config": {
            "quant_method": "fp8"
        }})
    assert tensor_quant.quant_type == QUANT_FP8
    assert tensor_quant.weight_block_size is None
    assert not tensor_quant.uses_fp8_block_weights


def test_fp8_block_linear_buffers():
    layer = FP8BlockLinear(256, 384)
    assert layer.weight.shape == (384, 256)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale_inv.shape == (3, 2)
    assert layer.weight_scale_inv.dtype == torch.float32
    assert layer.bias is None


def test_make_linear_selects_fp8_variants():
    config = SimpleNamespace(quant=QuantConfig(quant_type=QUANT_FP8_BLOCK),
                             tp_size=1,
                             tie_word_embeddings=False)
    assert isinstance(make_linear(config, 128, 128), FP8BlockLinear)

    config.quant = QuantConfig(quant_type=QUANT_FP8)
    assert isinstance(make_linear(config, 128, 128), FP8Linear)


def test_fp8_block_linear_repack_is_bit_view():
    layer = FP8BlockLinear(128, 128)
    layer.weight.copy_(torch.randn(128, 128).to(torch.float8_e4m3fn))
    original_bits = layer.weight.view(torch.int8).clone()

    layer.repack_for_export()

    assert layer.weight.dtype == torch.int8
    assert torch.equal(layer.weight, original_bits)
    assert torch.equal(
        layer.weight.view(torch.float8_e4m3fn).view(torch.int8), original_bits)
    layer.repack_for_export()


def test_fp8_block_gemm_eager_reference():
    x = torch.stack((torch.ones(256), torch.zeros(256))).to(torch.float16)
    weight = torch.ones(128, 256, dtype=torch.float8_e4m3fn)
    weight_scale = torch.tensor([[0.5, 0.25]], dtype=torch.float32)

    output = fp8_block_gemm(x, weight, weight_scale)

    expected = torch.stack((torch.full(
        (128, ), 96.0), torch.zeros(128))).to(torch.float16)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_fp8_block_gemm_fake_shape_propagation():
    with FakeTensorMode():
        x = torch.empty(2, 3, 256, dtype=torch.float16)
        weight = torch.empty(384, 256, dtype=torch.int8)
        weight_scale = torch.empty(3, 2, dtype=torch.float32)
        output = fp8_block_gemm(x, weight, weight_scale)

    assert output.shape == (2, 3, 384)
    assert output.dtype == torch.float16


def _make_attention(*, bias: bool = False) -> Attention:
    attention = Attention.__new__(Attention)
    nn.Module.__init__(attention)
    attention.enable_fp8_kv_cache = False
    attention.q_proj = FP8BlockLinear(128, 128, bias=bias)
    attention.k_proj = FP8BlockLinear(128, 128, bias=bias)
    attention.v_proj = FP8BlockLinear(128, 128, bias=bias)
    return attention


def test_fp8_block_qkv_fusion():
    attention = _make_attention()
    projections = [attention.q_proj, attention.k_proj, attention.v_proj]
    for value, projection in enumerate(projections, start=1):
        projection.weight.fill_(value)
        projection.weight_scale_inv.fill_(value / 10)
        projection.tp_mode = TPMode.COL
    expected_weight = torch.cat([p.weight for p in projections], dim=0)
    expected_scale = torch.cat([p.weight_scale_inv for p in projections],
                               dim=0)
    model = nn.Module()
    model.attention = attention

    assert fuse_qkv_projections(model) == 1
    assert isinstance(attention.qkv_proj_fused, FP8BlockLinear)
    assert attention.qkv_proj_fused.tp_mode == TPMode.COL
    assert torch.equal(attention.qkv_proj_fused.weight, expected_weight)
    assert torch.equal(attention.qkv_proj_fused.weight_scale_inv,
                       expected_scale)
    assert not hasattr(attention, "q_proj")
    assert not hasattr(attention, "k_proj")
    assert not hasattr(attention, "v_proj")


def test_fp8_block_qkv_fusion_rejects_bias():
    attention = _make_attention(bias=True)
    model = nn.Module()
    model.attention = attention

    assert fuse_qkv_projections(model) == 0
    assert not hasattr(attention, "qkv_proj_fused")
    assert all(
        hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj"))


def test_fp8_block_plugin_initializer_dtypes_are_preserved(tmp_path):
    weight = numpy_helper.from_array(
        torch.ones(128, 128, dtype=torch.int8).numpy(), "weight")
    scale = numpy_helper.from_array(torch.ones(1, 1).numpy(), "scale")
    unrelated = numpy_helper.from_array(torch.ones(2, 2).numpy(), "unrelated")
    node = helper.make_node("Fp8BlockGemmPlugin", ["x", "weight", "scale"],
                            ["y"],
                            domain="trt_edgellm")
    graph = helper.make_graph(
        [node], "fp8_block",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 128])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 128])],
        [weight, scale, unrelated])
    model = helper.make_model(graph,
                              opset_imports=[
                                  helper.make_opsetid("", 24),
                                  helper.make_opsetid("trt_edgellm", 23)
                              ])
    path = str(tmp_path / "fp8_block.onnx")
    onnx.save(model, path)

    _fix_initializer_dtypes(path)

    fixed = onnx.load(path)
    dtypes = {init.name: init.data_type for init in fixed.graph.initializer}
    assert dtypes["weight"] == TensorProto.INT8
    assert dtypes["scale"] == TensorProto.FLOAT
    assert dtypes["unrelated"] == TensorProto.FLOAT16


def test_fp8_block_plugin_schema_registration():
    register_tensorrt_edgellm_onnx_custom_schemas()
    schema = onnx.defs.get_schema("Fp8BlockGemmPlugin", 23, "trt_edgellm")
    assert schema.domain == "trt_edgellm"
    assert len(schema.inputs) == 3


def test_fp8_block_linear_exports_plugin_node(tmp_path):
    layer = FP8BlockLinear(128, 128)
    layer.weight.copy_(torch.ones_like(layer.weight))
    layer.weight_scale_inv.fill_(0.5)
    layer.repack_for_export()
    path = str(tmp_path / "linear.onnx")

    program = torch.onnx.export(
        layer,
        (torch.ones(1, 128, dtype=torch.float16), ),
        dynamo=True,
        opset_version=24,
        custom_translation_table=build_custom_translation_table(),
        optimize=False,
    )
    program.save(path)

    exported = onnx.load(path)
    plugin_nodes = [
        node for node in exported.graph.node if node.domain == "trt_edgellm"
        and node.op_type == "Fp8BlockGemmPlugin"
    ]
    assert len(plugin_nodes) == 1
    initializers = {init.name: init for init in exported.graph.initializer}
    plugin = plugin_nodes[0]
    assert initializers[plugin.input[1]].data_type == TensorProto.INT8
    assert initializers[plugin.input[2]].data_type == TensorProto.FLOAT
