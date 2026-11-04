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
"""LFM3-VL vision convolution and visual export contract.

The vision tower's depthwise 3x3 convolution runs per frame over a packed,
merge-block-ordered sequence. The exported graph evaluates it as a gather over
``conv_neighbor_index``; these tests pin that formulation to ``nn.Conv2d``
applied on each frame's raster grid, the way the reference model computes it.
"""

import json
import os
import sys

import pytest

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from tensorrt_edgellm.config import ModelConfig  # noqa: E402
from tensorrt_edgellm.models.lfm3_vl import build_lfm3_vl_visual  # noqa: E402
from tensorrt_edgellm.models.lfm3_vl import conv_neighbor_index
from tensorrt_edgellm.models.lfm3_vl.modeling_lfm3_vl_visual import \
    Lfm3VlVisionConv2d  # noqa: E402

_MERGE = 2
_HIDDEN = 32


def _vision_config() -> dict:
    return {
        "hidden_size": _HIDDEN,
        "intermediate_size": 48,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "layer_types": ["conv", "full_attention"],
        "conv_kernel_size": 3,
        "norm_eps": 1e-5,
        "patch_size": 4,
        "temporal_patch_size": 2,
        "in_channels": 3,
        "spatial_merge_size": _MERGE,
        "rope_parameters": {
            "rope_theta": 1000000.0,
            "rope_type": "axial"
        },
    }


def _model_config(tmp_path) -> ModelConfig:
    text_config = {
        "model_type": "lfm3",
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "layer_types": ["conv", "full_attention"],
        "conv_L_cache": 7,
        "norm_eps": 1e-5,
        "vocab_size": 32,
        "rope_parameters": {
            "full_attention": {
                "rope_theta": 1000000.0,
                "rope_type": "default"
            },
            "sliding_attention": {
                "rope_theta": 10000,
                "rope_type": "default"
            },
        },
    }
    with open(os.path.join(tmp_path, "config.json"), "w") as f:
        json.dump(text_config, f)
    safetensors_torch.save_file({"model.embed_tokens.weight": torch.zeros(1)},
                                os.path.join(tmp_path, "model.safetensors"))
    return ModelConfig.from_pretrained(str(tmp_path),
                                       lambda head_dim: head_dim**-0.5)


def _conv2d_reference(gated: torch.Tensor, weight: torch.Tensor,
                      frames: list) -> torch.Tensor:
    """Per frame: merge-block order -> raster grid, depthwise Conv2d, back."""
    outputs, start = [], 0
    for height, width in frames:
        frame = gated[start:start + height * width]
        frame = frame.view(height // _MERGE, width // _MERGE, _MERGE, _MERGE,
                           -1)
        raster = frame.permute(4, 0, 2, 1, 3).reshape(1, -1, height, width)
        conv = torch.nn.functional.conv2d(raster,
                                          weight,
                                          padding=1,
                                          groups=_HIDDEN)
        conv = conv.view(-1, height // _MERGE, _MERGE, width // _MERGE,
                         _MERGE).permute(1, 3, 2, 4, 0)
        outputs.append(conv.reshape(height * width, -1))
        start += height * width
    return torch.cat(outputs)


def test_neighbor_gather_convolution_matches_conv2d(tmp_path):
    torch.manual_seed(0)
    frames = [(6, 8), (4, 2)]
    conv = Lfm3VlVisionConv2d(_vision_config(), _model_config(tmp_path),
                              "vision_tower.layers.0.conv")
    for linear in (conv.in_proj, conv.out_proj):
        linear.weight.data = torch.randn_like(linear.weight) * 0.2
    weight = torch.randn(_HIDDEN, 1, 3, 3)
    conv.spatial_conv.weight = weight.half()
    total = sum(h * w for h, w in frames)
    hidden = torch.randn(total, _HIDDEN).half()
    neighbors = conv_neighbor_index(frames, _MERGE)

    with torch.no_grad():
        actual = conv(hidden, neighbors)
        b_gate, c_gate, x = conv.in_proj(hidden).chunk(3, dim=-1)
        reference = _conv2d_reference((b_gate * x).float(), weight, frames)
        expected = conv.out_proj(c_gate * reference.half())

    torch.testing.assert_close(actual.float(),
                               expected.float(),
                               rtol=1e-2,
                               atol=1e-2)
    assert int(neighbors.max()) == total  # zero-padding taps exist


def test_visual_export_declares_neighbor_index_input(tmp_path):
    config = {
        "vision_config": _vision_config(),
        "text_config": {
            "hidden_size": 64
        },
        "downsample_factor": 2,
        "projector_hidden_size": 48,
        "projector_bias": True,
        "projector_hidden_act": "gelu",
        "projector_layer_norm_eps": 1e-6,
    }
    model = build_lfm3_vl_visual(config, {}, _model_config(tmp_path))
    inputs, names, outputs, shapes = model.get_onnx_export_args(config, "cpu")

    assert names[:4] == [
        "input", "rotary_pos_emb", "cu_seqlens", "conv_neighbor_index"
    ]
    assert outputs == ["output"]
    num_patches = inputs["hidden_states"].shape[0]
    assert inputs["conv_neighbor_index"].shape == (num_patches, 9)
    assert inputs["rotary_pos_emb"].shape[1] == (_HIDDEN // 4) // 2
    assert shapes["conv_neighbor_index"][0] is shapes["hidden_states"][0]
