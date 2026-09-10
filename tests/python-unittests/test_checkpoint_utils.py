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
"""Tests for runtime metadata emitted by
``tensorrt_edgellm/checkpoint/checkpoint_utils.py``.

Transformers v5 moved RoPE config from ``rope_scaling`` to ``rope_parameters``
(with the old key left as ``null``), and for VLMs the settings live inside
``text_config``. The C++ runtime's ``collectRopeConfig``
keys off ``rope_scaling.mrope_section`` to detect MRope — if the promoted LLM
dict lacks that, ``ropeConfig.type`` stays at the default and MRope-based runners
(QwenViTRunner, Qwen3OmniAudioRunner) fail fast on inference.
"""

import json
import os
import sys
import tempfile

import pytest

# Load the package from the repository root without installing it.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from tensorrt_edgellm.checkpoint.checkpoint_utils import \
        load_checkpoint_config_dicts
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"tensorrt_edgellm not importable: {exc}",
                allow_module_level=True)


def _write_config(tmpdir: str, data: dict) -> None:
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(data, f)


def _qwen3_vl_v5_config(mrope_section):
    """Minimal transformers-v5 style Qwen3-VL config: rope_scaling null,
    rope_parameters carries mrope_section under text_config."""
    return {
        "model_type": "qwen3_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 6144,
            "vocab_size": 151936,
            "head_dim": 128,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_scaling": None,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": mrope_section,
                "rope_theta": 5000000,
                "rope_type": "default",
            },
        },
        "vision_config": {
            "model_type": "qwen3_vl",
            "hidden_size": 1024,
        },
    }


def _assert_kmrope(rope_scaling):
    """Assert the dict would make collectRopeConfig() return kMRope."""
    assert isinstance(
        rope_scaling,
        dict), ("rope_scaling must be a dict to recover MRope; "
                "null would collapse to kDefault in the C++ parser")
    assert rope_scaling.get("mrope_section"), (
        "mrope_section must be preserved for MRope detection")
    rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
    assert rope_type in ("default", "mrope"), (
        "rope type must be 'default' (+ mrope_section) or 'mrope' "
        f"for kMRope; got {rope_type!r}")


def test_qwen3_vl_transformers_v5_recovers_rope_from_text_config_rope_parameters(
):
    """Transformers v5 puts the VLM RoPE dict in
    ``text_config.rope_parameters`` (with ``rope_scaling: null``). The promoted
    LLM dict must contain ``rope_scaling`` with ``mrope_section``, or the C++
    runtime classifies the engine as non-MRope and QwenViTRunner rejects the
    request."""
    with tempfile.TemporaryDirectory() as td:
        _write_config(td, _qwen3_vl_v5_config([24, 20, 20]))
        _root, llm = load_checkpoint_config_dicts(td)
        _assert_kmrope(llm.get("rope_scaling"))
        assert llm["rope_scaling"]["mrope_section"] == [24, 20, 20]


def test_top_level_rope_parameters_recovered_when_rope_scaling_null():
    """Non-VLM single-text case where transformers v5 exposes rope_parameters
    at the top level. The fallback should pick it up from the promoted LLM
    dict directly, no sub-config lookup needed."""
    with tempfile.TemporaryDirectory() as td:
        _write_config(
            td, {
                "model_type": "qwen3",
                "hidden_size": 2048,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 6144,
                "vocab_size": 151936,
                "head_dim": 128,
                "max_position_embeddings": 32768,
                "rms_norm_eps": 1e-6,
                "rope_scaling": None,
                "rope_parameters": {
                    "mrope_section": [16, 16, 16],
                    "rope_theta": 1000000,
                    "rope_type": "default",
                },
            })
        _root, llm = load_checkpoint_config_dicts(td)
        _assert_kmrope(llm.get("rope_scaling"))


def test_populated_rope_scaling_preserved_unchanged():
    """Pre-v5 HF configs and non-MRope v5 configs still work: if
    rope_scaling is already populated, promotion is a no-op."""
    with tempfile.TemporaryDirectory() as td:
        _write_config(
            td, {
                "model_type": "qwen2",
                "hidden_size": 896,
                "num_hidden_layers": 24,
                "num_attention_heads": 14,
                "num_key_value_heads": 2,
                "intermediate_size": 4864,
                "vocab_size": 151936,
                "head_dim": 64,
                "max_position_embeddings": 32768,
                "rms_norm_eps": 1e-6,
                "rope_scaling": {
                    "type": "default",
                    "rope_type": "default",
                    "mrope_section": [16, 24, 24],
                },
                "rope_theta": 1000000.0,
            })
        _root, llm = load_checkpoint_config_dicts(td)
        _assert_kmrope(llm.get("rope_scaling"))
        assert llm["rope_scaling"]["mrope_section"] == [16, 24, 24]


def test_non_mrope_model_unaffected():
    """A plain LLM with only rope_type/rope_theta (no mrope_section) should
    still end up with a rope_scaling dict; collectRopeConfig classifies it as
    kDefault because mrope_section is absent."""
    with tempfile.TemporaryDirectory() as td:
        _write_config(
            td, {
                "model_type": "qwen3",
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 3072,
                "vocab_size": 151936,
                "head_dim": 64,
                "max_position_embeddings": 32768,
                "rms_norm_eps": 1e-6,
                "rope_scaling": None,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 1000000,
                },
            })
        _root, llm = load_checkpoint_config_dicts(td)
        rs = llm.get("rope_scaling")
        assert isinstance(rs, dict)
        assert "mrope_section" not in rs
        # C++ parser: type='default' + no mrope_section -> kDefault (correct)
        assert rs.get("type") == "default"
