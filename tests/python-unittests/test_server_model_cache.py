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
"""Complete-model cache and runtime-routing tests."""

import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from experimental.server.runtime import engine_build
from experimental.server.runtime.engine import LLM
from experimental.server.runtime.engine_build import (BuildOptions,
                                                      PreparedModel)
from experimental.server.runtime.engine_layout import (EngineType,
                                                       inspect_bundle)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _engine_config(*, model="qwen2", spec="none", external=True):
    value = {
        "model": model,
        "spec_decode_type": spec,
        "engine_role": "llm" if spec == "none" else "base",
        "builder_config": {
            "max_input_len": 32,
            "max_batch_size": 1,
            "max_kv_cache_capacity": 96,
        },
    }
    if external:
        value["checkpoint_identity"] = {"version": 1, "sources": {}}
    return value


def _checkpoint(path: Path, model_type="qwen2") -> Path:
    path.mkdir()
    _write_json(path / "config.json", {"model_type": model_type})
    (path / "model.safetensors").touch()
    return path


def test_complete_bundle_layout(tmp_path):
    (tmp_path / "llm.engine").touch()
    _write_json(tmp_path / "config.json", _engine_config())
    (tmp_path / "visual").mkdir()
    (tmp_path / "visual" / "visual.engine").touch()
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "audio_encoder.engine").touch()
    for component, filename in (("talker", "llm.engine"), ("code_predictor",
                                                           "llm.engine"),
                                ("code2wav", "code2wav.engine")):
        (tmp_path / component).mkdir()
        (tmp_path / component / filename).touch()

    layout = inspect_bundle(str(tmp_path))
    assert layout.engine_type == EngineType.LLM
    assert layout.media_dir == str(tmp_path)
    assert layout.visual_dir == str(tmp_path / "visual")
    assert layout.audio_dir == str(tmp_path / "audio")
    assert layout.has_speech


def test_spec_bundle_layout(tmp_path):
    (tmp_path / "spec_base.engine").touch()
    (tmp_path / "spec_draft.engine").touch()
    _write_json(tmp_path / "base_config.json", _engine_config(spec="eagle3"))
    _write_json(
        tmp_path / "draft_config.json", {
            "engine_role": "draft",
            "checkpoint_identity": {
                "version": 1,
                "sources": {}
            },
        })
    assert inspect_bundle(str(tmp_path)).engine_type == EngineType.SPEC_DECODE


def test_builder_argv_builds_every_component_and_externalizes_weights():
    options = BuildOptions(spec_type="mtp", draft_model_dir="assistant")
    argv = options.to_argv("base", "bundle")
    assert argv[argv.index("--components") + 1] == "all"
    assert argv[argv.index("--spec-type") + 1] == "gemma4_mtp"
    assert argv[argv.index("--draft-model-dir") + 1] == "assistant"
    assert argv[argv.index("--externalize-weights") + 1] == "all"


def test_cache_path_binds_checkpoint_and_profile(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    cache = str(tmp_path / "cache")
    first = engine_build.bundle_cache_path(str(checkpoint), cache,
                                           BuildOptions(max_input_len=32))
    assert first == engine_build.bundle_cache_path(
        str(checkpoint), cache, BuildOptions(max_input_len=32))
    assert first != engine_build.bundle_cache_path(
        str(checkpoint), cache, BuildOptions(max_input_len=64))

    _write_json(checkpoint / "config.json", {
        "model_type": "qwen2",
        "revision": 2,
    })
    assert first != engine_build.bundle_cache_path(
        str(checkpoint), cache, BuildOptions(max_input_len=32))


def test_cache_identity_survives_checkpoint_relocation_and_mtime_changes(
        tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    cache = str(tmp_path / "cache")
    options = BuildOptions(max_input_len=32)
    expected = engine_build.bundle_cache_path(str(checkpoint), cache, options)

    for path in checkpoint.iterdir():
        os.utime(path, ns=(1, 1))
    assert engine_build.bundle_cache_path(str(checkpoint), cache,
                                          options) == expected

    relocated = tmp_path / "renamed-copy"
    shutil.copytree(checkpoint, relocated, copy_function=shutil.copy)
    assert engine_build.bundle_cache_path(str(relocated), cache,
                                          options) == expected


def test_engine_cache_prunes_lru_and_can_be_cleared(tmp_path):
    cache = tmp_path / "cache"
    old = cache / "engines" / "model-a" / "profile-old"
    active = cache / "engines" / "model-b" / "profile-active"
    old.mkdir(parents=True)
    active.mkdir(parents=True)
    (old / "llm.engine").write_bytes(b"old!")
    (active / "llm.engine").write_bytes(b"new!")
    os.utime(old, ns=(1, 1))
    os.utime(active, ns=(2, 2))
    checkpoint = cache / "checkpoints" / "model" / "config.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}", encoding="utf-8")

    assert engine_build.prune_engine_cache(str(cache), 4,
                                           keep=str(active)) == 1
    assert not old.exists()
    assert active.exists()
    assert engine_build.clear_engine_cache(str(cache)) == 1
    assert not active.exists()
    assert checkpoint.exists()


def test_prepare_model_builds_once_and_publishes_atomically(
        tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    cache = tmp_path / "cache"
    bundle = Path(
        engine_build.bundle_cache_path(str(checkpoint), str(cache),
                                       BuildOptions()))
    bundle.parent.mkdir(parents=True)
    bundle.parent.chmod(0o777)
    calls = []

    def is_ready(_model_dir, bundle_dir, _options):
        return Path(bundle_dir, "ready").is_file()

    def build(argv):
        calls.append(argv)
        output = Path(argv[argv.index("--engine-dir") + 1])
        (output / "ready").touch()

    monkeypatch.setattr(engine_build, "_is_ready", is_ready)
    monkeypatch.setattr(engine_build, "_resolve_plugin_path",
                        lambda _: "/plugin.so")
    monkeypatch.setattr("experimental.builder.cli.main", build)

    first = engine_build.prepare_model(str(checkpoint), str(cache))
    second = engine_build.prepare_model(str(checkpoint), str(cache))
    assert first.built
    assert not second.built
    assert first.bundle_dir == second.bundle_dir
    assert first.model_dir == str(checkpoint)
    assert Path(first.bundle_dir, "ready").is_file()
    assert stat.S_IMODE(Path(first.bundle_dir).stat().st_mode) == 0o777
    assert len(calls) == 1


def test_cache_reuse_requires_profile_and_checkpoint_identity(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "llm.engine").touch()
    options = BuildOptions(max_input_len=32,
                           max_batch_size=1,
                           max_kv_cache_capacity=96)

    _write_json(bundle / "config.json", _engine_config(external=False))
    assert not engine_build._is_ready(str(checkpoint), str(bundle), options)
    _write_json(bundle / "config.json", _engine_config())
    assert engine_build._is_ready(str(checkpoint), str(bundle), options)
    assert not engine_build._is_ready(
        str(checkpoint), str(bundle),
        BuildOptions(
            max_input_len=64, max_batch_size=1, max_kv_cache_capacity=96))


def test_model_source_rejects_cache_artifacts_and_onnx(tmp_path):
    onnx = tmp_path / "onnx"
    onnx.mkdir()
    (onnx / "model.onnx").touch()
    with pytest.raises(ValueError, match="ONNX"):
        engine_build.resolve_model_dir(str(onnx))

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "llm.engine").touch()
    with pytest.raises(ValueError, match="cache artifacts"):
        engine_build.resolve_model_dir(str(bundle))


class _FakeRuntime:

    def __init__(self, *args):
        self.args = args

    def capture_decoding_cuda_graph(self):
        return True

    def has_draft_model(self):
        return len(self.args) == 9


class _FakeContextCacheConfig:

    def __init__(self):
        self.enabled = False
        self.max_records = 0
        self.recurrent_snapshot_pool_bytes = 0
        self.partial_kv_snapshot_pool_bytes = 0


class _FakeBindings:
    LLMRuntime = _FakeRuntime
    ContextCacheConfig = _FakeContextCacheConfig


def test_llm_pairs_cached_bundle_with_resolved_checkpoints(
        tmp_path, monkeypatch):
    base = _checkpoint(tmp_path / "base")
    draft = _checkpoint(tmp_path / "draft")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "spec_base.engine").touch()
    (bundle / "spec_draft.engine").touch()
    _write_json(bundle / "base_config.json", _engine_config(spec="eagle3"))
    _write_json(
        bundle / "draft_config.json", {
            "engine_role": "draft",
            "checkpoint_identity": {
                "version": 1,
                "sources": {}
            },
        })

    monkeypatch.setattr(
        "experimental.server.runtime.engine_build.prepare_model",
        lambda *_args, **_kwargs: PreparedModel(str(bundle), str(base),
                                                str(draft)),
    )
    monkeypatch.setattr("experimental.server.runtime.engine._import_runtime",
                        lambda: _FakeBindings)
    llm = LLM(
        model="Qwen/Qwen3-1.7B",
        cache_dir=str(tmp_path / "cache"),
        build_options=BuildOptions(
            spec_type="eagle3",
            draft_model_dir="AngelSlim/Qwen3-1.7B_eagle3",
            max_input_len=32,
            max_batch_size=1,
            max_kv_cache_capacity=96,
        ),
    )
    assert llm.model_id == "Qwen/Qwen3-1.7B"
    assert llm.model_dir == str(base)
    assert llm.bundle_dir == str(bundle)
    assert llm._runtime.args[-4:-2] == (str(base), str(draft))
    assert not llm._runtime.args[-2].enabled
    assert llm._runtime.args[-1] == 0
