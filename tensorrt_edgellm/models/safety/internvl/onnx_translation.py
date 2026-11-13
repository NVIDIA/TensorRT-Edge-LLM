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
"""ONNX translation for the InternVL Safety attention interface."""

import onnx
import onnxscript
import torch
from onnxscript import script

from . import modeling_internvl  # noqa: F401 - registers the Safety custom op

_trt = onnxscript.values.Opset("trt", 1)
_op21 = onnxscript.opset21


@script()
def _safety_attention_plugin_translation(
    qkv: onnxscript.FLOAT16,
    past_key_value: onnxscript.FLOAT16,
    context_lengths: onnxscript.INT32,
    rope_rotary_cos_sin: onnxscript.FLOAT,
    kvcache_start_index: onnxscript.INT32,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[onnxscript.FLOAT16, onnxscript.FLOAT16]:
    output, present = _trt.AttentionPlugin(qkv,
                                           past_key_value,
                                           context_lengths,
                                           rope_rotary_cos_sin,
                                           kvcache_start_index,
                                           num_q_heads=num_q_heads,
                                           num_kv_heads=num_kv_heads,
                                           head_size=head_size,
                                           enable_tree_attention=0,
                                           plugin_version="1",
                                           plugin_namespace="trt",
                                           _outputs=2)
    return output, present


@script()
def _fp8_quantize_translation(
    hidden_states: onnxscript.FLOAT16,
    scale: onnxscript.FLOAT16,
) -> onnxscript.FLOAT8E4M3FN:
    return _op21.QuantizeLinear(
        hidden_states,
        scale,
        output_dtype=int(onnx.TensorProto.FLOAT8E4M3FN),
    )


@script()
def _fp8_dequantize_translation(
    x: onnxscript.FLOAT8E4M3FN,
    scale: onnxscript.FLOAT16,
) -> onnxscript.FLOAT16:
    return _op21.DequantizeLinear(x, scale)


@script()
def _gather_nd_translation(
    value: onnxscript.FLOAT16,
    indices: onnxscript.INT64,
) -> onnxscript.FLOAT16:
    axes = _op21.Constant(value_ints=[-1])
    indices_3d = _op21.Unsqueeze(indices, axes)
    return _op21.GatherND(value, indices_3d, batch_dims=1)


def get_custom_translation_table() -> dict:
    """Return the FP16/FP8 translations used by the Safety InternVL models."""
    return {
        torch.ops.trt.safety_attention_plugin.default:
        _safety_attention_plugin_translation,
        torch.ops.trt.fp8_quantize.default: _fp8_quantize_translation,
        torch.ops.trt.fp8_dequantize.default: _fp8_dequantize_translation,
        torch.ops.trt.gather_nd.default: _gather_nd_translation,
    }
