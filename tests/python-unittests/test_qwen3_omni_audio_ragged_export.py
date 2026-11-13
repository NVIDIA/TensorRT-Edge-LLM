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
"""ONNX graph contract of the Qwen3-Omni audio tower's default ragged export.

Kept out of the *plugin*.py files: lowering needs onnx and the 4-stage encoder
variant's modeling needs transformers, neither of which the device plugin-test
environment installs.
"""

from __future__ import annotations

import onnx
import pytest
import torch
from test_qwen3_omni_audio_attention_plugin import (_EXPORT_CONFIG, _HEAD_DIM,
                                                    _SCALE, _encoder,
                                                    _fill_weights)

from tensorrt_edgellm.models.qwen3_asr import modeling_qwen3_asr_audio
from tensorrt_edgellm.models.qwen3_omni_next.modeling_qwen3_omni_next_audio import \
    Qwen3OmniNextAudioEncoder
from tensorrt_edgellm.onnx.export_encoder import _run_dynamo_export

# Same head_dim as the real tower, small enough to lower to ONNX in seconds.
_TINY_ENCODER = dict(d_model=128,
                     num_layers=2,
                     num_heads=2,
                     ffn_dim=256,
                     output_dim=64,
                     downsample_hidden=32)


def test_exported_graph_runs_every_layer_on_the_plugin(monkeypatch, tmp_path):
    """Every layer's ViTAttentionPlugin reads cu_seqlens and max_seqlen_carrier
    straight from the graph inputs the C++ runtime binds, carries the head
    geometry, and no dense softmax is left."""
    monkeypatch.delenv("USE_AUDIO_ATTN_PLUGIN", raising=False)
    encoder = _encoder(modeling_qwen3_asr_audio.QwenAudioEncoder,
                       **_TINY_ENCODER)
    _fill_weights(encoder)
    encoder = encoder.half().eval()
    inputs, names, outputs, dynamic_shapes = encoder.get_onnx_export_args(
        _EXPORT_CONFIG, "cpu")
    path = tmp_path / "audio_encoder.onnx"
    _run_dynamo_export(encoder, inputs, str(path), names, outputs,
                       dynamic_shapes)
    graph = onnx.load(str(path), load_external_data=False).graph

    assert [tensor.name for tensor in graph.input] == [
        "padded_feature", "padded_mask_after_cnn_indices", "cu_seqlens",
        "max_seqlen_carrier"
    ]
    plugins = [
        node for node in graph.node if node.op_type == "ViTAttentionPlugin"
    ]
    assert len(plugins) == _TINY_ENCODER["num_layers"]
    for node in plugins:
        assert list(node.input[3:5]) == ["cu_seqlens", "max_seqlen_carrier"]
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        assert attributes["num_heads"] == _TINY_ENCODER["num_heads"]
        assert attributes["head_size"] == _HEAD_DIM
        assert attributes["attention_scale"] == pytest.approx(_SCALE)
    assert not any(node.op_type == "Softmax" for node in graph.node)


def test_forward_override_passes_the_window_partition_to_every_layer(
        monkeypatch):
    """The 4-stage encoder variant, which overrides forward, hands the caller's
    cu_seqlens and max_seqlen_carrier to every layer."""
    monkeypatch.delenv("USE_AUDIO_ATTN_PLUGIN", raising=False)
    encoder = _encoder(Qwen3OmniNextAudioEncoder, **_TINY_ENCODER)
    _fill_weights(encoder)
    encoder = encoder.half().eval()
    inputs, _, _, _ = encoder.get_onnx_export_args(_EXPORT_CONFIG, "cpu")

    calls = []
    original = modeling_qwen3_asr_audio.ops.vit_attention_plugin

    def recorder(query, key, value, cu_seqlens, carrier, **kwargs):
        calls.append((cu_seqlens, carrier))
        return original(query, key, value, cu_seqlens, carrier, **kwargs)

    monkeypatch.setattr(modeling_qwen3_asr_audio.ops, "vit_attention_plugin",
                        recorder)
    with torch.no_grad():
        encoder(**inputs)

    assert len(calls) == _TINY_ENCODER["num_layers"]
    for cu_seqlens, carrier in calls:
        assert cu_seqlens is inputs["cu_seqlens"]
        assert carrier is inputs["max_seqlen_carrier"]
