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
"""Unit tests for Gemma 4 (``gemma4_text``) checkpoint config parsing.

Covers :func:`tensorrt_edgellm.config._parse_gemma4_cfg`, the
:class:`tensorrt_edgellm.config.Gemma4Config` it produces, and the
``ModelConfig.from_pretrained`` integration — including the regression that a
dense Gemma 4 checkpoint (which sets optional MoE fields to ``null``) parses
without crashing.  Also checks that the registered :class:`Gemma4CausalLM`
fails with an actionable message until export support lands.
"""

import json
import os

import pytest

from tensorrt_edgellm.config import ModelConfig, _parse_gemma4_cfg

# The text sub-config of google/gemma-4-E2B-it (the LLM backbone fields the
# export frontend consumes).  Kept faithful to the published checkpoint.
_GEMMA4_E2B_TEXT_CONFIG = {
    "attention_bias": False,
    "attention_k_eq_v": False,
    "enable_moe_block": False,
    "expert_intermediate_size": None,
    "final_logit_softcapping": 30.0,
    "global_head_dim": 512,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 1536,
    "hidden_size_per_layer_input": 256,
    "intermediate_size": 6144,
    "layer_types": (["sliding_attention"] * 4 + ["full_attention"]) * 7,
    "max_position_embeddings": 131072,
    "model_type": "gemma4_text",
    "num_attention_heads": 8,
    "num_experts": None,
    "num_global_key_value_heads": None,
    "num_hidden_layers": 35,
    "num_key_value_heads": 1,
    "num_kv_shared_layers": 20,
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
    "sliding_window": 512,
    "tie_word_embeddings": True,
    "top_k_experts": None,
    "use_double_wide_mlp": True,
    "vocab_size": 262144,
    "vocab_size_per_layer_input": 262144,
}


def _write_checkpoint(tmp_path) -> str:
    """Write a minimal multimodal Gemma 4 config.json and return its directory."""
    root = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "tie_word_embeddings": True,
        "text_config": dict(_GEMMA4_E2B_TEXT_CONFIG),
    }
    with open(os.path.join(tmp_path, "config.json"), "w") as f:
        json.dump(root, f)
    return str(tmp_path)


def test_parse_gemma4_cfg_fields():
    cfg = _parse_gemma4_cfg(_GEMMA4_E2B_TEXT_CONFIG)
    assert cfg is not None
    assert cfg.local_head_dim == 256
    assert cfg.global_head_dim == 512
    assert cfg.sliding_window == 512
    assert cfg.num_kv_shared_layers == 20
    # num_global_key_value_heads is null -> falls back to num_key_value_heads.
    assert cfg.num_global_key_value_heads == 1
    assert cfg.attention_k_eq_v is False
    assert cfg.sliding_rope_theta == 10000.0
    assert cfg.global_rope_theta == 1000000.0
    assert cfg.global_partial_rotary_factor == 0.25
    assert cfg.global_rope_type == "proportional"
    assert cfg.hidden_size_per_layer_input == 256
    assert cfg.vocab_size_per_layer_input == 262144
    assert cfg.final_logit_softcapping == 30.0
    assert cfg.use_double_wide_mlp is True
    assert cfg.hidden_activation == "gelu_pytorch_tanh"


def test_parse_gemma4_cfg_layer_pattern():
    cfg = _parse_gemma4_cfg(_GEMMA4_E2B_TEXT_CONFIG)
    assert len(cfg.is_global_layer) == 35
    assert cfg.num_global_layers == 7
    assert cfg.num_local_layers == 28
    assert cfg.uses_per_layer_embeddings is True
    assert cfg.first_kv_shared_layer_idx(35) == 15
    # Every 5th layer (the trailing layer of each 4-local + 1-global block).
    assert [i for i, g in enumerate(cfg.is_global_layer)
            if g] == [4, 9, 14, 19, 24, 29, 34]


@pytest.mark.parametrize("model_type", ["llama", "qwen3_moe", "gemma2", ""])
def test_parse_gemma4_cfg_inert_for_non_gemma4(model_type):
    assert _parse_gemma4_cfg({"model_type": model_type}) is None


def test_model_config_from_pretrained(tmp_path):
    config = ModelConfig.from_pretrained(_write_checkpoint(tmp_path))
    assert config.is_gemma4 is True
    assert config.gemma4_cfg is not None
    assert config.model_type == "gemma4_text"
    assert config.num_hidden_layers == 35
    assert config.hidden_size == 1536
    assert config.num_attention_heads == 8
    assert config.num_key_value_heads == 1
    assert config.head_dim == 256
    assert config.intermediate_size == 6144
    assert config.vocab_size == 262144
    assert config.tie_word_embeddings is True
    # The flat rope_theta falls back to the local (sliding) theta; the precise
    # per-layer-type thetas live on gemma4_cfg.
    assert config.rope_theta == 10000.0
    assert config.gemma4_cfg.global_rope_theta == 1000000.0


def test_dense_gemma4_null_moe_fields_do_not_crash(tmp_path):
    """Dense Gemma 4 sets optional MoE keys to null; parsing must not raise."""
    config = ModelConfig.from_pretrained(_write_checkpoint(tmp_path))
    assert config.num_experts == 0
    assert config.moe_intermediate_size == 0


def test_registered_gemma4_class_fails_loudly(tmp_path):
    """A Gemma 4 checkpoint dispatches to Gemma4CausalLM, which fails loudly."""
    pytest.importorskip("torch")
    from tensorrt_edgellm import AutoModel

    with pytest.raises(NotImplementedError) as excinfo:
        AutoModel.from_pretrained(_write_checkpoint(tmp_path))
    message = str(excinfo.value)
    assert "gemma4_text" in message
    assert "issues/72" in message
