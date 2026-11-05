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
"""CLI tests for the embedding sidecar quantization flags."""

import inspect
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_edgellm import model as model_module
from tensorrt_edgellm.checkpoint import checkpoint_utils
from tensorrt_edgellm.models import linear as linear_module
from tensorrt_edgellm.onnx import export as onnx_export
from tensorrt_edgellm.scripts import export as export_script


def _checkpoint(tmp_path, model_type):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(
        {"model_type": model_type}),
                                            encoding="utf-8")
    return checkpoint


def _run_export_main(monkeypatch, checkpoint, tmp_path, *options):
    monkeypatch.setattr(sys, "argv", [
        "tensorrt-edgellm-export",
        str(checkpoint),
        str(tmp_path / "output"), *options
    ])
    export_script.main()


def _capture_export_llm(monkeypatch):
    captured = {}

    def fake_export_llm(model_dir, out_dir, **kwargs):
        captured.update(model_dir=model_dir, out_dir=out_dir, **kwargs)

    monkeypatch.setattr(export_script, "_export_llm", fake_export_llm)
    return captured


def test_fp8_and_int8_embedding_are_mutually_exclusive(monkeypatch, tmp_path,
                                                       capsys):
    checkpoint = _checkpoint(tmp_path, "qwen3")

    with pytest.raises(SystemExit, match="2"):
        _run_export_main(monkeypatch, checkpoint, tmp_path, "--fp8-embedding",
                         "--int8-embedding")

    assert "not allowed with argument --fp8-embedding" in capsys.readouterr(
    ).err


def test_int8_embedding_rejects_qwen3_omni(monkeypatch, tmp_path, capsys):
    checkpoint = _checkpoint(tmp_path, "qwen3_omni_moe")

    with pytest.raises(SystemExit, match="2"):
        _run_export_main(monkeypatch, checkpoint, tmp_path, "--int8-embedding")

    assert "--int8-embedding is not supported by Qwen3-Omni" in capsys.readouterr(
    ).err


def test_cosmos3_reasoning_forwards_int8_embedding(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path, "cosmos3_edge")
    captured = _capture_export_llm(monkeypatch)

    _run_export_main(monkeypatch, checkpoint, tmp_path, "--task", "reasoning",
                     "--skip-visual", "--int8-embedding")

    assert captured["model_type"] == "cosmos3_edge"
    assert captured["int8_embedding"] is True


def test_dflash_base_with_fp8_embedding_is_not_rejected_at_parse(
        monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path, "qwen3")
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    captured = _capture_export_llm(monkeypatch)

    _run_export_main(monkeypatch, checkpoint, tmp_path, "--dflash-base",
                     "--dflash-draft-dir", str(draft_dir), "--fp8-embedding")

    assert captured["fp8_embedding"] is True
    assert captured["dflash_draft_dir"] == str(draft_dir)


def _write_draft(tmp_path, with_embed_tokens):
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text(json.dumps(
        {"dflash_config": {
            "mask_token_id": 1
        }}),
                                           encoding="utf-8")
    if with_embed_tokens:
        save_file({"embed_tokens.weight": torch.ones(4, 8)},
                  str(draft_dir / "model.safetensors"))
    return draft_dir


def _write_quantized_sidecar(tmp_path, int8):
    llm_dir = tmp_path / "llm"
    llm_dir.mkdir()
    table = (torch.ones(4, 8, dtype=torch.int8)
             if int8 else torch.ones(4, 8, dtype=torch.float16))
    save_file({
        "embedding": table,
        "embedding_scale": torch.ones(4)
    }, str(llm_dir / "embedding.safetensors"))
    return llm_dir


def test_dflash_fold_skips_quantized_sidecar_without_draft_embedding(tmp_path):
    llm_dir = _write_quantized_sidecar(tmp_path, int8=True)
    draft_dir = _write_draft(tmp_path, with_embed_tokens=False)

    export_script._patch_dflash_mask_embedding(str(llm_dir), str(draft_dir))


@pytest.mark.parametrize("int8, flag", [(True, "--int8-embedding"),
                                        (False, "--fp8-embedding")])
def test_dflash_fold_rejects_quantized_sidecar_naming_the_flag(
        tmp_path, int8, flag):
    llm_dir = _write_quantized_sidecar(tmp_path, int8=int8)
    draft_dir = _write_draft(tmp_path, with_embed_tokens=True)

    with pytest.raises(ValueError, match=flag):
        export_script._patch_dflash_mask_embedding(str(llm_dir),
                                                   str(draft_dir))


@pytest.mark.parametrize("function", [
    checkpoint_utils.write_runtime_artifacts, onnx_export.export_onnx,
    export_script._export_llm, export_script._export_diffusion_gemma
])
def test_int8_embedding_is_a_keyword_only_parameter(function):
    parameter = inspect.signature(function).parameters["int8_embedding"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


def _patch_loader_and_exporter(monkeypatch, fake_model):
    captured = {}
    monkeypatch.setattr(model_module.AutoModel, "from_pretrained",
                        lambda *_args, **_kwargs: fake_model)

    def fake_export_onnx(model, output_path, **kwargs):
        captured["model"] = model
        captured.update(kwargs)

    monkeypatch.setattr(onnx_export, "export_onnx", fake_export_onnx)
    return captured


def test_export_llm_forwards_int8_embedding_to_export_onnx(
        monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path, "qwen3")
    fake_model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))
    captured = _patch_loader_and_exporter(monkeypatch, fake_model)

    export_script._export_llm(str(checkpoint),
                              str(tmp_path / "llm"),
                              model_type="qwen3",
                              int8_embedding=True)

    assert captured["model"] is fake_model
    assert captured["int8_embedding"] is True
    assert captured["fp8_embedding"] is False


def test_export_diffusion_gemma_forwards_int8_embedding_to_export_onnx(
        monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path, "diffusion_gemma")
    fake_backbone = SimpleNamespace(self_conditioning=SimpleNamespace(
        gate_proj=object(), up_proj=object(), down_proj=object()))
    captured = _patch_loader_and_exporter(monkeypatch, fake_backbone)
    # The export checks isinstance(module, FP16Linear) on the fake modules.
    monkeypatch.setattr(linear_module, "FP16Linear", object)

    export_script._export_diffusion_gemma(str(checkpoint),
                                          str(tmp_path / "out"),
                                          int8_embedding=True)

    assert captured["model"] is fake_backbone
    assert captured["int8_embedding"] is True
    assert captured["fp8_embedding"] is False
