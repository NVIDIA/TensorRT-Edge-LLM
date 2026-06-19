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
"""Structural tests for the Gemma 4 (``gemma4_text``) modeling.

Uses a *reduced* config (small vocab / few layers) that keeps the Gemma 4
structure — interleaved local/global layers, KV sharing, Per-Layer Embeddings —
so the model can be built, run and ONNX-exported on CPU with random weights
without materializing the full E2B embedding tables.  These checks validate
structure (shapes, KV-share layout, export I/O contract), not numerics.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from tensorrt_edgellm.config import ModelConfig  # noqa: E402
from tensorrt_edgellm.models.gemma4.modeling_gemma4_text import \
    Gemma4CausalLM  # noqa: E402

# A small gemma4_text config: 10 layers as (4 sliding + 1 full) x 2, with the
# trailing 4 layers KV-shared.  Mirrors the E2B architecture at toy sizes.
_REDUCED_GEMMA4_CONFIG = {
    "attention_bias": False,
    "attention_k_eq_v": False,
    "enable_moe_block": False,
    "expert_intermediate_size": None,
    "final_logit_softcapping": 30.0,
    "global_head_dim": 64,
    "head_dim": 32,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 128,
    "hidden_size_per_layer_input": 16,
    "intermediate_size": 256,
    "layer_types": (["sliding_attention"] * 4 + ["full_attention"]) * 2,
    "max_position_embeddings": 4096,
    "model_type": "gemma4_text",
    "num_attention_heads": 4,
    "num_experts": None,
    "num_global_key_value_heads": None,
    "num_hidden_layers": 10,
    "num_key_value_heads": 1,
    "num_kv_shared_layers": 4,
    "moe_intermediate_size": None,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {
            "rope_theta": 10000.0,
            "rope_type": "default",
        },
    },
    "sliding_window": 8,
    "tie_word_embeddings": True,
    "top_k_experts": None,
    "use_double_wide_mlp": True,
    "vocab_size": 512,
    "vocab_size_per_layer_input": 512,
}


def _build_config(tmp_path) -> ModelConfig:
    root = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "tie_word_embeddings": True,
        "text_config": dict(_REDUCED_GEMMA4_CONFIG),
    }
    with open(os.path.join(tmp_path, "config.json"), "w") as f:
        json.dump(root, f)
    return ModelConfig.from_pretrained(str(tmp_path))


def _build_model(tmp_path) -> Gemma4CausalLM:
    config = _build_config(tmp_path)
    model = Gemma4CausalLM(config).eval()
    # Real deployment loads fp16 weights; random-init here, so cast to fp16
    # before tie_weights (which clones embed_tokens into lm_head).
    model.half()
    model.tie_weights()
    return model


def test_kv_share_layout(tmp_path):
    """The trailing 4 layers share KV; layers 4 and 5 store full-length KV."""
    model = _build_model(tmp_path)
    shared = [
        i for i, layer in enumerate(model.model.layers)
        if layer.self_attn.is_kv_shared_layer
    ]
    stores = [
        i for i, layer in enumerate(model.model.layers)
        if layer.self_attn.store_full_length_kv
    ]
    assert shared == [6, 7, 8, 9]
    # Last non-shared full layer (4) and last non-shared sliding layer (5).
    assert stores == [4, 5]
    # Shared layers carry no k/v/k_norm/v_norm weights.
    for i in shared:
        attn = model.model.layers[i].self_attn
        assert attn.k_proj is None and attn.v_proj is None
        assert attn.k_norm is None and attn.v_norm is None


def test_per_layer_head_dims(tmp_path):
    """Local (sliding) layers use head_dim 32; global (full) use 64."""
    model = _build_model(tmp_path)
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if attn.is_global:
            assert attn.head_dim == 64
            assert attn.sliding_window_size == -1
        else:
            assert attn.head_dim == 32
            assert attn.sliding_window_size == 8


def test_double_wide_mlp_on_shared_layers(tmp_path):
    """KV-shared layers get a double-wide MLP intermediate (use_double_wide)."""
    model = _build_model(tmp_path)
    # Non-shared layer: intermediate == 256; shared layer: 512.
    assert model.model.layers[0].mlp.gate_proj.weight.shape[0] == 256
    assert model.model.layers[9].mlp.gate_proj.weight.shape[0] == 512


def test_forward_shapes(tmp_path):
    """A forward over the flat export wrapper yields logits + per-layer KV."""
    model = _build_model(tmp_path)
    spec = model.onnx_export_spec()
    with torch.no_grad():
        out = spec.wrapped(*spec.args)
    logits = out[0]
    present = out[1:]
    assert logits.shape == (1, 1, _REDUCED_GEMMA4_CONFIG["vocab_size"])
    assert logits.dtype == torch.float32
    assert len(present) == _REDUCED_GEMMA4_CONFIG["num_hidden_layers"]


def test_export_io_contract(tmp_path):
    """The ONNX spec exposes the dual RoPE tables and the PLE input."""
    model = _build_model(tmp_path)
    spec = model.onnx_export_spec()
    assert "rope_rotary_cos_sin_local" in spec.input_names
    assert "rope_rotary_cos_sin_global" in spec.input_names
    assert "per_layer_inputs" in spec.input_names
    na = _REDUCED_GEMMA4_CONFIG["num_hidden_layers"]
    assert spec.output_names[0] == "logits"
    assert len(spec.output_names) == 1 + na
    # One dynamic-shape descriptor per positional arg.
    assert len(spec.dynamic_shapes) == len(spec.args)


def test_onnx_export(tmp_path):
    """End-to-end: the modeling exports to ONNX via the dynamo export path."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    from tensorrt_edgellm.onnx.export import export_onnx

    model = _build_model(tmp_path)
    onnx_path = os.path.join(str(tmp_path), "model.onnx")
    export_onnx(model, onnx_path, model_dir="")
    assert os.path.exists(onnx_path)
    assert os.path.getsize(onnx_path) > 0
