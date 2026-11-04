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
"""LFM3 config parsing and runtime sidecar contract.

LFM3 mixes gated short-convolution layers (conv state only) with sliding and
full attention layers. Full attention rotates only part of the head while keeping
the full-head frequency progression, which maps onto the runtime's proportional
RoPE rather than a shrunken rotary prefix.
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

from tensorrt_edgellm.checkpoint.checkpoint_utils import \
    build_runtime_llm_config_dict  # noqa: E402
from tensorrt_edgellm.config import LAYER_CONV, ModelConfig  # noqa: E402
from tensorrt_edgellm.models.lfm3 import Lfm3ForCausalLM  # noqa: E402

_LAYER_TYPES = ["conv", "full_attention", "conv", "sliding_attention"]


def _write_lfm3_checkpoint(path) -> str:
    config = {
        "architectures": ["Lfm3ForCausalLM"],
        "model_type": "lfm3",
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": len(_LAYER_TYPES),
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "layer_types": _LAYER_TYPES,
        "conv_L_cache": 7,
        "conv_bias": False,
        "norm_eps": 1e-5,
        "sliding_window": 8,
        "use_sandwich_norm": True,
        "tie_word_embeddings": True,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.5,
                "rope_theta": 1000000.0,
                "rope_type": "default",
            },
            "sliding_attention": {
                "rope_theta": 10000,
                "rope_type": "default",
            },
        },
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    # QK-norm is detected from checkpoint key names.
    safetensors_torch.save_file(
        {"model.layers.1.self_attn.q_norm.weight": torch.ones(16)},
        os.path.join(path, "model.safetensors"))
    return str(path)


def _load_config(tmp_path) -> ModelConfig:
    return ModelConfig.from_pretrained(_write_lfm3_checkpoint(tmp_path),
                                       lambda head_dim: head_dim**-0.5)


def test_lfm3_parses_conv_layers_and_proportional_full_rope(tmp_path):
    config = _load_config(tmp_path)

    assert config.layer_types == [
        LAYER_CONV, "full_attention", LAYER_CONV, "sliding_attention"
    ]
    assert config.num_conv_layers == 2 and config.num_attn_layers == 2
    assert config.is_hybrid
    assert (config.short_conv_cfg.conv_dim,
            config.short_conv_cfg.conv_kernel) == (64, 7)
    assert config.rms_norm_eps == pytest.approx(1e-5)
    assert config.use_sandwich_norm and config.has_qk_norm
    assert config.sliding_window_size == 8

    full_rope = config.full_rope_config
    assert full_rope["rope_scaling"]["type"] == "proportional"
    assert full_rope["rope_scaling"]["partial_rotary_factor"] == 0.5
    assert full_rope["rope_theta"] == 1000000.0
    assert config.sliding_rope_config["rope_scaling"]["type"] == "default"


def test_lfm3_rejects_unsupported_partial_rope_type(tmp_path):
    path = _write_lfm3_checkpoint(tmp_path)
    with open(os.path.join(path, "config.json")) as f:
        raw = json.load(f)
    raw["rope_parameters"]["full_attention"]["rope_type"] = "linear"
    raw["rope_parameters"]["full_attention"]["factor"] = 2.0
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(raw, f)
    with pytest.raises(NotImplementedError, match="rope_type"):
        ModelConfig.from_pretrained(path, lambda head_dim: head_dim**-0.5)


def test_lfm3_runtime_sidecar_declares_conv_only_recurrent_layers(tmp_path):
    config = _load_config(tmp_path)
    model = Lfm3ForCausalLM(config)

    out = build_runtime_llm_config_dict(model)

    assert out["layer_types"] == ["mamba", "attention", "mamba", "attention"]
    assert out["kv_layer_configs"] == [
        None, {
            "num_kv_heads": 2,
            "head_dim": 16
        }, None, {
            "num_kv_heads": 2,
            "head_dim": 16
        }
    ]
    assert out["num_linear_attn_layers"] == 2
    assert out["num_attention_layers"] == 2
    assert (out["recurrent_state_num_heads"], out["recurrent_state_head_dim"],
            out["recurrent_state_size"]) == (0, 0, 0)
    assert (out["conv_dim"], out["conv_kernel"]) == (64, 7)
    assert out["conv_state_dtype"] == "fp16"
    assert out["full_rope_config"]["rope_scaling"]["type"] == "proportional"


def test_lfm3_onnx_spec_binds_layer_type_rope_and_conv_states(tmp_path):
    config = _load_config(tmp_path)
    spec = Lfm3ForCausalLM(config).onnx_export_spec()

    assert "rope_rotary_cos_sin_sliding" in spec.input_names
    assert "rope_rotary_cos_sin_full" in spec.input_names
    assert "rope_rotary_cos_sin" not in spec.input_names
    assert [n for n in spec.input_names if n.startswith("conv_state_")
            ] == ["conv_state_0", "conv_state_1"]
    assert not any(n.startswith("recurrent_state") for n in spec.input_names)
    assert spec.output_names == [
        "logits", "present_key_values_0", "present_key_values_1",
        "present_conv_state_0", "present_conv_state_1"
    ]
    # Both RoPE tables span the full head: sliding rotates it all, full is
    # proportional.
    rope_full = spec.args[spec.input_names.index("rope_rotary_cos_sin_full")]
    assert rope_full.shape[-1] == config.head_dim
    assert len(spec.args) == len(spec.input_names) == len(spec.dynamic_shapes)
