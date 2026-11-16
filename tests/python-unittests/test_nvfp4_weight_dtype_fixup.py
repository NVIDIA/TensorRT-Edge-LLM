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
"""The post-export NVFP4 fixup keeps the dequantize value_info consistent."""

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

from tensorrt_edgellm.onnx.export import _fix_nvfp4_weight_dtype


def _value_info(model, name):
    for info in list(model.graph.value_info) + list(model.graph.output):
        if info.name == name:
            tensor_type = info.type.tensor_type
            return tensor_type.elem_type, [
                dim.dim_value for dim in tensor_type.shape.dim
            ]
    raise KeyError(name)


def _save(graph, tmp_path, opsets):
    path = tmp_path / "model.onnx"
    onnx.save_model(helper.make_model(graph, opset_imports=opsets), str(path))
    return str(path)


def test_moe_block_dequantize_output_is_fp16(tmp_path):
    """The stacked MoE FP4 buffers become FLOAT4E2M1 with the unpacked last
    dim, and the TRT_BlockDequantize output keeps the node's fp16 type."""
    experts, out, half_in = 2, 4, 16
    weight = numpy_helper.from_array(
        np.zeros((experts, out, half_in), np.int8),
        "layers.0.mlp.nvfp4_gate_weight")
    scale = numpy_helper.from_array(np.ones((experts, out, 2), np.float16),
                                    "layers.0.mlp.nvfp4_gate_block_scale")
    dequantize = helper.make_node("TRT_BlockDequantize",
                                  [weight.name, scale.name], ["gate_dq"],
                                  domain="trt",
                                  block_shape=[1, 1, 16],
                                  output_dtype=TP.FLOAT16)
    output = helper.make_tensor_value_info("gate_dq", TP.FLOAT16,
                                           [experts, out, half_in])
    graph = helper.make_graph([dequantize],
                              "moe", [], [output],
                              initializer=[weight, scale])
    path = _save(graph, tmp_path,
                 [helper.make_opsetid("", 21),
                  helper.make_opsetid("trt", 1)])

    _fix_nvfp4_weight_dtype(path)

    model = onnx.load(path, load_external_data=False)
    initializers = {init.name: init for init in model.graph.initializer}
    assert initializers[weight.name].data_type == TP.FLOAT4E2M1
    assert list(initializers[weight.name].dims) == [experts, out, 2 * half_in]
    assert initializers[scale.name].data_type == TP.FLOAT16
    assert _value_info(model,
                       "gate_dq") == (TP.FLOAT16, [experts, out, 2 * half_in])


def test_dense_dequantize_linear_output_stays_fp32(tmp_path):
    """The dense NVFP4 path dequantizes to fp32 (and casts later)."""
    out, half_in = 4, 16
    weight = numpy_helper.from_array(np.zeros((out, half_in), np.int8),
                                     "layers.0.self_attn.o_proj.weight")
    scale = numpy_helper.from_array(np.ones((out, 2), np.float32), "ws")
    dequantize = helper.make_node("DequantizeLinear",
                                  [weight.name, scale.name], ["w_dq"],
                                  axis=-1,
                                  block_size=16)
    output = helper.make_tensor_value_info("w_dq", TP.FLOAT, [out, half_in])
    graph = helper.make_graph([dequantize],
                              "dense", [], [output],
                              initializer=[weight, scale])
    path = _save(graph, tmp_path, [helper.make_opsetid("", 21)])

    _fix_nvfp4_weight_dtype(path)

    model = onnx.load(path, load_external_data=False)
    assert _value_info(model, "w_dq") == (TP.FLOAT, [out, 2 * half_in])
