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
"""Alpamayo runtime artifacts preserve the FP8 KV dequant contract."""

import ast
import json
import logging
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from tensorrt_edgellm.builder.models.alpamayo import runtime_config, weights
from tensorrt_edgellm.checkpoint import checkpoint_utils
from tensorrt_edgellm.config import Mapping
from tensorrt_edgellm.models.linear import (FP16Linear, NVFP4LinearMethod,
                                            ReplicatedLinear)
from tensorrt_edgellm.vocab_reduction.onnx_export import \
    copy_reduced_vocab_artifacts


def _args():
    return SimpleNamespace(model_dir="checkpoint",
                           spec_type=None,
                           spec_role=None)


def _load_visual_processor_export():
    export_path = os.path.join(os.path.dirname(__file__), "..", "..",
                               "tensorrt_edgellm", "onnx", "cli.py")
    with open(export_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=export_path)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "_save_alpamayo_visual_processor")
    namespace = {
        "json": json,
        "logger": logging.getLogger(__name__),
        "os": os,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), export_path,
                "exec"), namespace)
    return namespace["_save_alpamayo_visual_processor"]


def _load_alpamayo_generation_vocab_export():
    export_path = os.path.join(os.path.dirname(__file__), "..", "..",
                               "tensorrt_edgellm", "onnx", "cli.py")
    with open(export_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=export_path)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "_apply_alpamayo_generation_vocab")
    namespace = {
        "__package__": "tensorrt_edgellm.scripts",
        "json": json,
        "os": os,
        "torch": torch,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), export_path,
                "exec"), namespace)
    return namespace["_apply_alpamayo_generation_vocab"]


@pytest.mark.parametrize("quant_type", ["fp16", "nvfp4"])
def test_alpamayo_generation_vocab_preserves_embedding_and_control_tokens(
        tmp_path, quant_type):
    """Trim generated logits without removing embeddings or control tokens."""
    config = {
        "vocab_size": 12,
        "traj_token_start_idx": 3,
        "traj_vocab_size": 3,
        "traj_token_ids": {
            "future_start": 9,
            "future_end": 10,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(vocab_size=12)
            self.embed_tokens = torch.nn.Embedding(12, 16)
            if quant_type == "fp16":
                self.lm_head = FP16Linear(16, 12)
            else:
                self.lm_head = ReplicatedLinear(16, 12, False, torch.float16,
                                                Mapping(), NVFP4LinearMethod())
            with torch.no_grad():
                values = torch.arange(self.lm_head.weight.numel()).reshape_as(
                    self.lm_head.weight)
                self.lm_head.weight.copy_(values)

    model = Model()
    original_embedding = model.embed_tokens.weight.detach().clone()
    original_head = model.lm_head.weight.detach().clone()
    _load_alpamayo_generation_vocab_export()(model, str(tmp_path), "")

    expected_ids = [0, 1, 2, 6, 7, 8, 9, 10, 11]
    assert model.config.vocab_size == 12
    assert model.config.reduced_vocab_size == len(expected_ids)
    assert torch.equal(model.embed_tokens.weight, original_embedding)
    assert torch.equal(model.lm_head.weight, original_head[expected_ids])
    assert model.lm_head.out_features == len(expected_ids)

    copy_reduced_vocab_artifacts(model, str(tmp_path / "output"))
    with safe_open(tmp_path / "output" / "vocab_map.safetensors",
                   framework="pt") as f:
        assert f.get_tensor("vocab_map").tolist() == expected_ids


def test_visual_processor_config_does_not_require_image_dependencies(
        monkeypatch, tmp_path):
    """Export visual metadata without loading processor image dependencies."""
    source = tmp_path / "source_preprocessor_config.json"
    source.write_text(json.dumps({
        "image_mean": [0.5, 0.5, 0.5],
        "patch_size": 16,
        "size": {
            "longest_edge": 1,
            "shortest_edge": 1,
        },
    }),
                      encoding="utf-8")
    calls = []

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(source)

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        SimpleNamespace(hf_hub_download=fake_hf_hub_download))
    output = tmp_path / "visual"

    save_visual_processor = _load_visual_processor_export()
    save_visual_processor({"vlm_name_or_path": "base-vlm"}, str(output))

    assert calls == [("base-vlm", "preprocessor_config.json", {})]
    with open(output / "preprocessor_config.json", encoding="utf-8") as f:
        config = json.load(f)
    assert config["image_mean"] == [0.5, 0.5, 0.5]
    assert config["patch_size"] == 16
    assert config["size"] == {
        "longest_edge": 16777216,
        "shortest_edge": 65536,
    }


def test_fp8_llm_config_persists_per_layer_kv_scales(monkeypatch):
    """Persist decoder FP8 KV scales in layer order."""
    calls = []

    class FakeWeights:

        def __init__(self, model_dir, **kwargs):
            calls.append(("open", model_dir, kwargs))

        def qkv_scales(self, prefix):
            calls.append(("scales", prefix))
            layer = int(prefix.split(".")[2])
            return (1.0, 0.25 + layer, 0.5 + layer)

        def close(self):
            calls.append(("close", ))

    # Given an FP8 Alpamayo LLM configuration backed by quantized checkpoint metadata.
    monkeypatch.setattr(runtime_config, "Weights", FakeWeights)
    config = {}
    cfg = SimpleNamespace(kv_cache_dtype="fp8",
                          component="llm",
                          num_hidden_layers=2)

    # When the runtime artifact hook runs.
    runtime_config.update_llm_config(config, {}, cfg, _args())

    # Then it stores K/V scales in layer order and closes the checkpoint reader.
    assert config["kv_cache_scales"] == [[0.25, 0.5], [1.25, 1.5]]
    assert [call[1] for call in calls if call[0] == "scales"
            ] == ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    assert calls[-1] == ("close", )


def test_fp16_llm_config_does_not_open_checkpoint(monkeypatch):
    """Avoid checkpoint reads when FP8 KV scales are unnecessary."""

    class UnexpectedWeights:

        def __init__(self, *args, **kwargs):
            raise AssertionError("FP16 runtime config must not open weights")

    # Given an Alpamayo LLM whose KV cache remains FP16.
    monkeypatch.setattr(runtime_config, "Weights", UnexpectedWeights)
    config = {}
    cfg = SimpleNamespace(kv_cache_dtype="fp16",
                          component="llm",
                          num_hidden_layers=2)

    # When the runtime artifact hook runs.
    runtime_config.update_llm_config(config, {}, cfg, _args())

    # Then no scale metadata is added and the checkpoint is not opened.
    assert "kv_cache_scales" not in config


def test_standard_export_persists_alpamayo_qwen_backbone_fp8_kv_scales():
    """Collect FP8 KV scales only from Alpamayo decoder layers."""

    class FakeAttention:

        enable_fp8_kv_cache = True

        def __init__(self, k_scale, v_scale):
            self._qkv_scales_float = [1.0, k_scale, v_scale]

    class FakeModel:

        def named_modules(self):
            return [
                ("", self),
                ("vlm.visual.layers.0.self_attn", FakeAttention(9.0, 10.0)),
                ("vlm.model.language_model.layers.1.self_attn",
                 FakeAttention(1.25, 1.5)),
                ("vlm.model.language_model.layers.0.self_attn",
                 FakeAttention(0.25, 0.5)),
            ]

    # Given an Alpamayo export with calibrated attention scale triples.
    # When runtime sidecar metadata is collected.
    scales = checkpoint_utils._alpamayo_kv_cache_scales(
        FakeModel(), {"model_type": "alpamayo_r1"}, {
            "kv_cache_dtype": "fp8",
            "num_hidden_layers": 2,
        })

    # Then each decoder layer retains its K/V scales in order.
    assert scales == [[0.25, 0.5], [1.25, 1.5]]


def test_standard_export_omits_scales_for_unrelated_models():
    """Leave unrelated VLM exports free of Alpamayo KV metadata."""

    class UnexpectedModel:

        def named_modules(self):
            raise AssertionError(
                "non-Alpamayo export must not inspect modules")

    # Given a non-Alpamayo VLM with FP8 KV metadata.
    # When the Alpamayo sidecar hook is queried.
    scales = checkpoint_utils._alpamayo_kv_cache_scales(
        UnexpectedModel(), {"model_type": "qwen3_vl"}, {
            "kv_cache_dtype": "fp8",
            "num_hidden_layers": 2,
        })

    # Then the unrelated export is left unchanged.
    assert scales is None


def test_official_action_weight_names_are_direct_candidates():
    """Resolve official action-expert weights without prefix remapping."""
    # Given an official unprefixed Alpamayo action-expert weight.
    name = "expert.layers.0.self_attn.q_proj.weight"

    # When direct-builder candidates are resolved.
    candidates = weights.resolve_candidates(name,
                                            component="action",
                                            spec_type="none",
                                            spec_role="none",
                                            quant_type="fp16")

    # Then the checkpoint key is the preferred candidate.
    assert candidates[0] == name
