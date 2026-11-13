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
from types import SimpleNamespace

import pytest

# Load the package from the repository root without installing it.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from tensorrt_edgellm.checkpoint.checkpoint_utils import (
        _build_alpamayo_tokenizer, load_checkpoint_config_dicts,
        validate_external_speculative_target)
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


def test_alpamayo_tokenizer_does_not_require_processor_dependencies(
        monkeypatch, tmp_path):
    """Build Alpamayo tokenizer artifacts without loading a VLM processor."""
    calls = []

    class FakeTokenizer:

        def __init__(self):
            self.added_tokens = []

        def add_tokens(self, tokens, special_tokens=False):
            self.added_tokens.append((tokens, special_tokens))

        def save_pretrained(self, output_dir):
            for filename in ("tokenizer.json", "tokenizer_config.json"):
                with open(os.path.join(output_dir, filename), "w") as output:
                    output.write("{}")

        def __len__(self):
            return 42

    tokenizer = FakeTokenizer()

    class FakeAutoTokenizer:

        @staticmethod
        def from_pretrained(reference, **kwargs):
            calls.append((reference, kwargs))
            return tokenizer

    monkeypatch.setitem(sys.modules, "transformers",
                        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))

    _build_alpamayo_tokenizer(
        {
            "vlm_name_or_path": "base-vlm",
            "traj_vocab_size": 2,
            "add_special_tokens": False,
        }, str(tmp_path))

    assert calls == [("base-vlm", {"trust_remote_code": True})]
    assert tokenizer.added_tokens[0] == (["<i0>", "<i1>"], False)
    assert tokenizer.added_tokens[1][1] is True
    assert (tmp_path / "tokenizer.json").is_file()
    assert (tmp_path / "tokenizer_config.json").is_file()


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


def test_external_speculative_target_accepts_supported_verifier():
    validate_external_speculative_target({"model_type": "qwen3"})


def test_eagle3_provider_layers_are_preserved():
    with tempfile.TemporaryDirectory() as td:
        _write_config(
            td, {
                "speculators_model_type": "eagle3",
                "eagle_aux_hidden_state_layer_ids": [2, 18, 33],
                "transformer_layer_config": {
                    "model_type": "llama",
                    "hidden_size": 4096,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 32,
                },
            })
        _, llm = load_checkpoint_config_dicts(td)

    assert llm["target_layer_ids"] == [2, 18, 33]


@pytest.mark.parametrize("speculator_type", ["dflash", "dspark"])
def test_speculators_aux_layers_use_edge_layer_output_indices(speculator_type):
    with tempfile.TemporaryDirectory() as td:
        _write_config(
            td, {
                "speculators_model_type": speculator_type,
                "aux_hidden_state_layer_ids": [2, 10, 18, 26, 34],
                "block_size": 8,
                "transformer_layer_config": {
                    "model_type": "qwen3",
                    "hidden_size": 4096,
                    "num_hidden_layers": 5,
                    "num_attention_heads": 32,
                },
            })
        _, llm = load_checkpoint_config_dicts(td)

    config_name = f"{speculator_type}_config"
    assert llm[config_name]["target_layer_ids"] == [1, 9, 17, 25, 33]


def test_dspark_promotes_provider_dflash_metadata():
    with tempfile.TemporaryDirectory() as td:
        provider_contract = {
            "target_layer_ids": [4, 16, 28, 40, 52],
            "mask_token_id": 248077,
            "projector_type": "dspark",
        }
        _write_config(
            td, {
                "model_type": "qwen3",
                "architectures": ["DSparkDraftModel"],
                "hidden_size": 5120,
                "num_hidden_layers": 5,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 10240,
                "vocab_size": 248320,
                "dflash_config": provider_contract,
            })

        _, llm = load_checkpoint_config_dicts(td)

    assert llm["dspark_config"] == provider_contract


def test_external_speculative_target_rejects_nemotron():
    # Given a Nemotron target with no supported recurrent drafter selected,
    # when validating external speculation, then the unsupported route is rejected.
    with pytest.raises(ValueError, match="Nemotron external"):
        validate_external_speculative_target({"model_type": "nemotron_h"})


@pytest.mark.parametrize("mode", ["dflash", "dspark", "dspark_onnx"])
def test_nemotron_h_supported_external_speculative_modes(mode):
    """Protect the supported Nemotron-H external-drafter mode contract."""
    # Given a Nemotron-H target and a supported external drafter producer.
    config = {"model_type": "nemotron_h"}
    # When the producer-specific mode is validated, then the supported hybrid
    # verifier route is accepted.
    validate_external_speculative_target(config, mode=mode)


@pytest.mark.parametrize("mode", ["", "eagle3", "jetspec", "other"])
def test_nemotron_h_external_speculative_mode_contract(mode):
    # Given a Nemotron-H target without a supported recurrent drafter mode.
    config = {"model_type": "nemotron_h"}
    # When the route is validated, then unsupported producers are rejected.
    with pytest.raises(ValueError, match="Nemotron external"):
        validate_external_speculative_target(config, mode=mode)


def test_dspark_onnx_does_not_allow_unknown_nemotron_architecture():
    # Given a Nemotron family without the Nemotron-H DSpark contract.
    config = {"model_type": "nemotron_unknown"}
    # When DSpark ONNX is selected, then the unsupported target is rejected.
    with pytest.raises(ValueError, match="Nemotron external"):
        validate_external_speculative_target(config, mode="dspark_onnx")


def test_dspark_cli_rejects_unknown_nemotron(tmp_path, capsys):
    from tensorrt_edgellm.onnx import cli as export_script

    # Given an unsupported Nemotron checkpoint and a DSpark CLI request.
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "nemotron_unknown"}))
    # When the CLI validates the target, then it rejects the architecture.
    with pytest.raises(SystemExit):
        export_script.main(
            [str(checkpoint),
             str(tmp_path / "output"), "--dspark-base"])
    assert "Nemotron external" in capsys.readouterr().err


@pytest.mark.parametrize(("flag", "expected_mode"),
                         [("--eagle-base", "eagle3"),
                          ("--dflash-base", "dflash"),
                          ("--jetspec-base", "jetspec"),
                          ("--dspark-base", "dspark_onnx")])
def test_official_export_passes_external_speculative_producer_mode(
        tmp_path, capsys, monkeypatch, flag, expected_mode):
    """Protect producer-specific target validation in the official exporter."""
    from tensorrt_edgellm.onnx import cli as export_script

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type":
                                                        "qwen3"}))

    def validate(config, *, mode):
        assert config["model_type"] == "qwen3"
        raise ValueError(f"validated mode={mode}")

    monkeypatch.setattr(export_script.checkpoint_utils,
                        "validate_external_speculative_target", validate)

    # Given an official external-spec export request, when target validation
    # runs, then the producer is identified explicitly at the API boundary.
    with pytest.raises(SystemExit):
        export_script.main([str(checkpoint), str(tmp_path / "output"), flag])
    assert f"validated mode={expected_mode}" in capsys.readouterr().err


@pytest.mark.parametrize("draft_vocab_size", [None, 64, 128])
def test_dspark_optional_draft_vocab_size(tmp_path, draft_vocab_size):
    from tensorrt_edgellm import config as model_config

    # Given a full-vocabulary or explicitly reduced DSpark checkpoint.
    data = {
        "model_type": "qwen3",
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "vocab_size": 128,
        "draft_vocab_size": draft_vocab_size,
        "dspark_config": {
            "target_layer_ids": [1]
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    # When the draft configuration is loaded, then null means full vocabulary.
    config = model_config.make_dspark_draft_config(
        str(tmp_path), default_attention_scale=lambda d: d**-0.5)
    assert config.draft_vocab_size == (128 if draft_vocab_size is None else
                                       draft_vocab_size)
    assert config.reduced_vocab_size == (64
                                         if draft_vocab_size == 64 else None)
