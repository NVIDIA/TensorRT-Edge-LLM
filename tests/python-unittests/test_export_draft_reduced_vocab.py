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

import json
import sys
from types import SimpleNamespace

import pytest

from tensorrt_edgellm import model as model_module
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
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tensorrt-edgellm-export",
            str(checkpoint),
            str(tmp_path / "output"),
            *options,
        ],
    )
    export_script.main()


def test_mtp_tree_rejects_draft_reduced_vocab(monkeypatch, tmp_path, capsys):
    checkpoint = _checkpoint(tmp_path, "qwen3_5")

    with pytest.raises(SystemExit, match="2"):
        _run_export_main(monkeypatch, checkpoint, tmp_path, "--mtp-tree-base",
                         "--draft-reduced-vocab-dir", "reduced")

    assert "not supported with --mtp-tree-base" in capsys.readouterr().err


def test_draft_reduced_vocab_requires_consuming_stage(monkeypatch, tmp_path,
                                                      capsys):
    checkpoint = _checkpoint(tmp_path, "cosmos3_edge")

    with pytest.raises(SystemExit, match="2"):
        _run_export_main(monkeypatch, checkpoint, tmp_path,
                         "--draft-reduced-vocab-dir", "reduced")

    assert "requires a consuming draft stage" in capsys.readouterr().err


def test_component_filter_cannot_silently_drop_mtp_draft_reduction(
        monkeypatch, tmp_path, capsys):
    checkpoint = _checkpoint(tmp_path, "qwen3_5")

    with pytest.raises(SystemExit, match="2"):
        _run_export_main(monkeypatch, checkpoint, tmp_path, "--mtp",
                         "--draft-reduced-vocab-dir", "reduced",
                         "--components", "thinker")

    assert "requires a consuming draft stage" in capsys.readouterr().err


def test_cosmos_export_cannot_silently_drop_mtp_draft_reduction(
        monkeypatch, tmp_path, capsys):
    checkpoint = _checkpoint(tmp_path, "cosmos3_edge")

    with pytest.raises(SystemExit, match="2"):
        _run_export_main(monkeypatch, checkpoint, tmp_path, "--mtp",
                         "--draft-reduced-vocab-dir", "reduced")

    assert "requires a consuming draft stage" in capsys.readouterr().err


def test_mtp_draft_forwards_reduced_vocab_to_model_loader(
        monkeypatch, tmp_path):
    captured = {}
    fake_model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=128, reduced_vocab_size=4))

    def from_pretrained(model_dir, **kwargs):
        captured.update(model_dir=model_dir, **kwargs)
        return fake_model

    monkeypatch.setattr(model_module.AutoModel, "from_pretrained",
                        from_pretrained)
    monkeypatch.setattr(onnx_export, "export_onnx",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(export_script, "_write_draft_vocab_sidecar",
                        lambda *args, **kwargs: None)

    export_script._export_mtp_draft(
        "checkpoint",
        str(tmp_path / "mtp_draft"),
        draft_reduced_vocab_dir="reduced",
    )

    assert captured == {
        "model_dir": "checkpoint",
        "device": "cpu",
        "mtp_draft": True,
        "reduced_vocab_dir": "reduced",
    }
