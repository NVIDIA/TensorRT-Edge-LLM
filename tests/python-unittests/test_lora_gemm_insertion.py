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
"""Dynamic-LoRA insertion contracts for token-major ONNX Gemm nodes."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from tensorrt_edgellm.lora.lora import (_match_fp8_gemm, _match_fp16_gemm,
                                        insert_lora_and_save)


def _gemm_graph(weight_name="_model.layers.0.mlp.up_proj.weight", **attrs):
    hidden = gs.Variable("hidden", dtype=np.float16, shape=["tokens", 16])
    weight = gs.Constant(weight_name,
                         values=np.zeros((32, 16), dtype=np.float16))
    output = gs.Variable("output", dtype=np.float16, shape=["tokens", 32])
    gemm_attrs = {"alpha": 1.0, "transA": 0, "transB": 1}
    gemm_attrs.update(attrs)
    node = gs.Node(op="Gemm",
                   name="node_Gemm_0",
                   attrs=gemm_attrs,
                   inputs=[hidden, weight],
                   outputs=[output])
    return gs.Graph(nodes=[node], inputs=[hidden], outputs=[output])


def _fp8_gemm_graph():
    hidden = gs.Variable("hidden", dtype=np.float16, shape=["tokens", 16])
    activation_scale = gs.Constant("activation_scale",
                                   values=np.array(1.0, dtype=np.float16))
    hidden_fp8 = gs.Variable("hidden_fp8", shape=["tokens", 16])
    hidden_dq = gs.Variable("hidden_dq",
                            dtype=np.float16,
                            shape=["tokens", 16])
    weight = gs.Constant("_model.layers.0.mlp.up_proj.weight",
                         values=np.zeros((32, 16), dtype=np.uint8))
    weight_scale = gs.Constant("weight_scale",
                               values=np.array(1.0, dtype=np.float16))
    weight_dq = gs.Variable("weight_dq", dtype=np.float16, shape=[32, 16])
    output = gs.Variable("output", dtype=np.float16, shape=["tokens", 32])
    nodes = [
        gs.Node(op="QuantizeLinear",
                attrs={"output_dtype": 17},
                inputs=[hidden, activation_scale],
                outputs=[hidden_fp8]),
        gs.Node(op="DequantizeLinear",
                inputs=[hidden_fp8, activation_scale],
                outputs=[hidden_dq]),
        gs.Node(op="DequantizeLinear",
                inputs=[weight, weight_scale],
                outputs=[weight_dq]),
        gs.Node(op="Gemm",
                name="node_Gemm_0",
                attrs={
                    "alpha": 1.0,
                    "transA": 0,
                    "transB": 1
                },
                inputs=[hidden_dq, weight_dq],
                outputs=[output]),
    ]
    return gs.Graph(nodes=nodes, inputs=[hidden], outputs=[output])


def test_fp16_gemm_match_preserves_transposed_weight_semantics():
    hidden = gs.Variable("hidden", dtype=np.float16, shape=["tokens", 16])
    weight = gs.Constant("_model.layers.0.mlp.up_proj.weight",
                         values=np.zeros((32, 16), dtype=np.float16))
    bias = gs.Constant("bias", values=np.zeros((32, ), dtype=np.float16))
    output = gs.Variable("output", dtype=np.float16, shape=["tokens", 32])
    node = gs.Node(op="Gemm",
                   name="node_Gemm_0",
                   attrs={
                       "alpha": 1.0,
                       "beta": 1.0,
                       "transA": 0,
                       "transB": 1
                   },
                   inputs=[hidden, weight, bias],
                   outputs=[output])
    graph = gs.Graph(nodes=[node], inputs=[hidden], outputs=[output])

    matches = _match_fp16_gemm(graph)

    assert len(matches) == 1
    assert matches[0].input is hidden
    assert matches[0].output is output
    assert matches[0].weight_shape == (16, 32)
    assert matches[0].name == "/layers/0/mlp/up_proj/MatMul"
    assert node.attrs == {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 1}


def test_fp8_gemm_match_supports_token_major_gemm_export():
    matches = _match_fp8_gemm(_fp8_gemm_graph())

    assert len(matches) == 1
    assert matches[0].input.name == "hidden"
    assert matches[0].output.name == "output"
    assert matches[0].weight_shape == (16, 32)
    assert matches[0].name == "/layers/0/mlp/up_proj/MatMul"


def test_insert_lora_rejects_graph_without_eligible_linear(tmp_path):
    value = onnx.helper.make_tensor_value_info("value",
                                               onnx.TensorProto.FLOAT16,
                                               ["tokens", 16])
    output = onnx.helper.make_tensor_value_info("output",
                                                onnx.TensorProto.FLOAT16,
                                                ["tokens", 16])
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["value"], ["output"])],
        "no-linear", [value], [output])
    onnx.save(onnx.helper.make_model(graph), tmp_path / "model.onnx")

    with pytest.raises(ValueError, match="no eligible linear layers"):
        insert_lora_and_save(str(tmp_path))


def test_fp16_gemm_match_skips_scaled_base_output():
    assert _match_fp16_gemm(_gemm_graph(alpha=0.5)) == []


def test_fp16_gemm_match_skips_noncanonical_weight_initializer():
    assert _match_fp16_gemm(_gemm_graph(weight_name="weight")) == []


def test_canonical_lm_head_is_excluded_actionably(tmp_path):
    graph = _gemm_graph(weight_name="_model.lm_head.weight")
    assert [match.name
            for match in _match_fp16_gemm(graph)] == ["/lm_head/MatMul"]
    onnx.save(gs.export_onnx(graph), tmp_path / "model.onnx")

    with pytest.raises(ValueError, match="no eligible linear layers"):
        insert_lora_and_save(str(tmp_path))


@pytest.mark.parametrize("graph", [
    _gemm_graph(alpha=0.5),
    _gemm_graph(weight_name="weight"),
])
def test_insert_lora_rejects_when_every_gemm_is_skipped(tmp_path, graph):
    onnx.save(gs.export_onnx(graph), tmp_path / "model.onnx")

    with pytest.raises(ValueError, match="no eligible linear layers"):
        insert_lora_and_save(str(tmp_path))
