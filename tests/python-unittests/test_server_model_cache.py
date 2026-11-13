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
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_edgellm.server.config import (ServerConfigError,
                                            parse_server_config)
from tensorrt_edgellm.server.runtime import engine_build
from tensorrt_edgellm.server.runtime.engine import LLM
from tensorrt_edgellm.server.runtime.engine_build import (BuildOptions,
                                                          PreparedModel)
from tensorrt_edgellm.server.runtime.engine_layout import (EngineType,
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
    if spec != "none":
        value["builder_config"]["max_spec_verify_tokens"] = 16
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


def test_builder_argv_builds_every_component_with_default_weight_policy():
    options = BuildOptions(spec_type="mtp",
                           draft_model_dir="assistant",
                           max_spec_verify_tokens=16,
                           max_spec_draft_tokens=8)
    argv = options.to_argv("base", "bundle")
    assert argv[argv.index("--components") + 1] == "all"
    assert argv[argv.index("--spec-type") + 1] == "gemma4_mtp"
    assert argv[argv.index("--draft-model-dir") + 1] == "assistant"
    assert argv[argv.index("--max-spec-verify-tokens") + 1] == "16"
    assert argv[argv.index("--max-spec-draft-tokens") + 1] == "8"
    assert "--externalize-weights" not in argv


def test_server_build_uses_packaged_plugin(monkeypatch, tmp_path):
    """A wheel server build passes its selected native plugin to the builder."""
    plugin = tmp_path / "libNvInfer_edgellm_plugin.so"
    monkeypatch.delenv("EDGELLM_PLUGIN_PATH", raising=False)
    monkeypatch.setattr("tensorrt_edgellm._native.load.resolve_payload",
                        lambda: SimpleNamespace(plugin=plugin))
    assert engine_build._resolve_plugin_path("") == str(plugin)


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


def test_cache_path_binds_engine_bundle_contract(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    cache = str(tmp_path / "cache")
    options = BuildOptions(max_input_len=32)
    current = engine_build.bundle_cache_path(str(checkpoint), cache, options)

    monkeypatch.setattr(engine_build, "_CACHE_LAYOUT_VERSION",
                        engine_build._CACHE_LAYOUT_VERSION + 1)

    assert current != engine_build.bundle_cache_path(str(checkpoint), cache,
                                                     options)


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


def test_builder_reports_each_component(tmp_path, monkeypatch):
    pytest.importorskip("tensorrt")
    from tensorrt_edgellm.builder import cli
    from tensorrt_edgellm.builder.core import builder

    args = types.SimpleNamespace(plugin_path="/plugin.so")
    bundle = types.SimpleNamespace(root_model_type="qwen3_5")
    labels = ["spec-base", "spec-draft", "visual"]
    plan = [(label, args, bundle, label) for label in labels]
    progress = []

    def build_one(_args, _bundle, component, _plugin):
        assert len(progress) == labels.index(component) + 1
        path = tmp_path / (component + ".engine")
        path.write_bytes(b"engine")
        return str(path)

    monkeypatch.setattr(cli, "_resolve_build_selection", lambda _:
                        (bundle, []))
    monkeypatch.setattr(cli, "_build_plan", lambda *_: plan)
    monkeypatch.setattr(cli, "_preflight_plan", lambda *_: None)
    monkeypatch.setattr(cli, "_build_one", build_one)
    monkeypatch.setattr(builder, "load_plugin_library", lambda _: None)
    cli._build(args, on_progress=progress.append)
    assert len(progress) == len(labels)
    for index, name in enumerate(["base", "draft", "vision"], 1):
        stage = progress[index - 1]
        assert name in stage.lower()
        assert f"{index}/{len(labels)}" in stage


@pytest.fixture
def model_build(tmp_path, monkeypatch):
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

    def build(argv, *, on_progress=None):
        calls.append(argv)
        if on_progress is not None:
            on_progress("component progress")
        output = Path(argv[argv.index("--engine-dir") + 1])
        (output / "ready").touch()

    monkeypatch.setattr(engine_build, "_is_ready", is_ready)
    monkeypatch.setattr(engine_build, "_resolve_plugin_path",
                        lambda _: "/plugin.so")
    monkeypatch.setattr("tensorrt_edgellm.builder.cli.main", build)
    return types.SimpleNamespace(checkpoint=str(checkpoint),
                                 cache=str(cache),
                                 calls=calls)


def test_prepare_model_builds_once_and_publishes_atomically(model_build):
    first = engine_build.prepare_model(model_build.checkpoint,
                                       model_build.cache)
    second = engine_build.prepare_model(model_build.checkpoint,
                                        model_build.cache)
    assert first.built
    assert not second.built
    assert first.bundle_dir == second.bundle_dir
    assert first.model_dir == model_build.checkpoint
    assert Path(first.bundle_dir, "ready").is_file()
    assert stat.S_IMODE(Path(first.bundle_dir).stat().st_mode) == 0o777
    assert len(model_build.calls) == 1


def test_prepare_model_reports_build_and_cache_progress(model_build):
    progress = []
    engine_build.prepare_model(model_build.checkpoint,
                               model_build.cache,
                               on_progress=progress.append)
    forwarded = progress.index("component progress")
    assert any("building" in stage.lower() for stage in progress[:forwarded])
    assert any("complete" in stage.lower()
               for stage in progress[forwarded + 1:])

    progress.clear()
    engine_build.prepare_model(model_build.checkpoint,
                               model_build.cache,
                               on_progress=progress.append)
    assert "reusing" in progress[-1].lower()
    assert "cache" in progress[-1].lower()
    assert "component progress" not in progress


def test_cache_reuse_accepts_baked_weights_and_validates_external_weights(
        tmp_path):
    """Cache reuse accepts baked engines but requires identity for bindings."""
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "llm.engine").touch()
    options = BuildOptions(max_input_len=32,
                           max_batch_size=1,
                           max_kv_cache_capacity=96)

    _write_json(bundle / "config.json", _engine_config(external=False))
    assert engine_build._is_ready(str(checkpoint), str(bundle), options)
    external = _engine_config(external=False)
    external["checkpoint_weight_bindings"] = [{"kind": "fp16"}]
    _write_json(bundle / "config.json", external)
    assert not engine_build._is_ready(str(checkpoint), str(bundle), options)
    external["checkpoint_identity"] = {"version": 1, "sources": {}}
    _write_json(bundle / "config.json", external)
    assert engine_build._is_ready(str(checkpoint), str(bundle), options)
    assert not engine_build._is_ready(
        str(checkpoint), str(bundle),
        BuildOptions(
            max_input_len=64, max_batch_size=1, max_kv_cache_capacity=96))


@pytest.mark.parametrize(
    "field", ["max_input_len", "max_batch_size", "max_kv_cache_capacity"])
def test_cache_reuse_requires_the_requested_engine_limits(tmp_path, field):
    """A cached bundle is reused only when its builder_config holds the batch size, input length and KV capacity the
    server was started with, so the server's pre-build checks against those flags see the served engine's limits."""
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "llm.engine").touch()
    _write_json(bundle / "config.json", _engine_config())
    limits = dict(max_input_len=32, max_batch_size=1, max_kv_cache_capacity=96)
    assert engine_build._is_ready(str(checkpoint), str(bundle),
                                  BuildOptions(**limits))
    limits[field] *= 2
    assert not engine_build._is_ready(str(checkpoint), str(bundle),
                                      BuildOptions(**limits))


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
    resolved_policy = SimpleNamespace(mode="mixed", execution="compact")

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def capture_decoding_cuda_graph(self):
        return True

    def has_draft_model(self):
        return self.kwargs.get("draft_checkpoint_dir") is not None


class _FakeContextCacheConfig:

    def __init__(self):
        self.enabled = False
        self.max_records = 0
        self.max_snapshots = 0


class _FakeBindings:
    LLMRuntime = _FakeRuntime
    ContextCacheConfig = _FakeContextCacheConfig
    PleRuntimeConfig = SimpleNamespace
    PleResidency = SimpleNamespace(RESIDENT="resident", CACHED="cached")
    RequestEngine = _FakeRuntime

    @staticmethod
    def validate_scheduling_options(**_options):
        pass


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
        "tensorrt_edgellm.server.runtime.engine_build.prepare_model",
        lambda *_args, **_kwargs: PreparedModel(str(bundle), str(base),
                                                str(draft)),
    )
    monkeypatch.setattr(
        "tensorrt_edgellm.server.runtime.engine._import_runtime",
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
    # Given a resolved paired checkpoint, when loading its cached bundle, then both weight roots reach the runtime.
    assert llm._runtime.kwargs["checkpoint_dir"] == str(base)
    assert llm._runtime.kwargs["draft_checkpoint_dir"] == str(draft)
    assert not llm._runtime.kwargs["context_cache_config"].enabled


def test_muse_glimmer_dflash2_server_loads_paired_runtime(
        tmp_path, monkeypatch):
    base = _checkpoint(tmp_path / "Muse-Glimmer-NVFP4", "muse_glimmer")
    draft = _checkpoint(tmp_path / "Muse-Glimmer-30B-DFlash2", "qwen3")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "spec_base.engine").touch()
    (bundle / "spec_draft.engine").touch()
    base_config = _engine_config(model="muse_glimmer_text", spec="dflash")
    base_config["builder_config"]["max_spec_verify_tokens"] = 16
    _write_json(bundle / "base_config.json", base_config)
    _write_json(
        bundle / "draft_config.json", {
            "engine_role": "draft",
            "builder_config": {
                "max_spec_draft_tokens": 16
            },
            "dflash_config": {
                "version": 2,
                "block_size": 16,
                "supports_probabilistic_sampling": True,
            },
            "checkpoint_identity": {
                "version": 1,
                "sources": {}
            },
        })
    prepared_options = []

    def prepare_model(_model, _cache_dir, options, **_kwargs):
        prepared_options.append(options)
        return PreparedModel(str(bundle), str(base), str(draft))

    monkeypatch.setattr(
        "tensorrt_edgellm.server.runtime.engine_build.prepare_model",
        prepare_model,
    )
    monkeypatch.setattr(
        "tensorrt_edgellm.server.runtime.engine._import_runtime",
        lambda: _FakeBindings)

    llm = LLM(
        model="RadixArk/Muse-Glimmer-NVFP4",
        cache_dir=str(tmp_path / "cache"),
        max_image_tokens=4096,
        max_image_tokens_per_image=4096,
        max_spec_verify_tokens=64,
        max_spec_draft_tokens=16,
        speculative_config={
            "method": "dflash",
            "model": "incoai/Muse-Glimmer-30B-DFlash2",
        },
    )

    assert prepared_options[0].spec_type == "dflash"
    assert prepared_options[0].draft_model_dir == \
        "incoai/Muse-Glimmer-30B-DFlash2"
    assert prepared_options[0].max_image_tokens == 4096
    assert prepared_options[0].max_image_tokens_per_image == 4096
    assert prepared_options[0].max_spec_verify_tokens == 64
    assert prepared_options[0].max_spec_draft_tokens == 16
    assert llm.has_draft_model
    # Given omitted P/Q, when loading a paired bundle, then native configuration resolves both defaults.
    assert llm._runtime.kwargs["draft_top_k"] == 1
    assert llm._runtime.kwargs["draft_step"] == 1
    assert llm._runtime.kwargs["proposal_tokens"] is None
    assert llm._runtime.kwargs["verify_tokens"] is None
    assert llm._runtime.kwargs["checkpoint_dir"] == str(base)
    assert llm._runtime.kwargs["draft_checkpoint_dir"] == str(draft)


@pytest.mark.parametrize("ifb", [False, True])
def test_server_ple_options_reach_native_runtime_without_changing_engine_cache(
        tmp_path, monkeypatch, ifb):
    checkpoint = _checkpoint(tmp_path / "checkpoint", "gemma4")
    cache = tmp_path / "cache"
    selections = []

    def prepare(model, cache_dir, options, **kwargs):
        bundle = Path(engine_build.bundle_cache_path(model, cache_dir,
                                                     options))
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "llm.engine").touch()
        _write_json(bundle / "config.json", _engine_config(model="gemma4"))
        selections.append((str(bundle), options.to_argv(model, str(bundle))))
        return PreparedModel(str(bundle), model)

    monkeypatch.setattr(engine_build, "prepare_model", prepare)
    monkeypatch.setattr(
        "tensorrt_edgellm.server.runtime.engine._import_runtime",
        lambda: _FakeBindings)
    common = [str(checkpoint), "--cache-dir", str(cache)]
    if ifb:
        common.append("--enable-in-flight-batching")
    default = LLM(**parse_server_config(common).model.llm_kwargs())
    selected = LLM(**parse_server_config(common + [
        "--ple-residency", "cached", "--ple-cache-slots", "513", "--ple-file",
        "/local/checkpoint-shard.safetensors"
    ]).model.llm_kwargs())
    default_native = default._engine if ifb else default._runtime
    selected_native = selected._engine if ifb else selected._runtime
    assert vars(default_native.kwargs["ple_config"]) == {
        "residency": "resident",
        "cache_slots": 0,
        "file_path": ""
    }
    assert vars(selected_native.kwargs["ple_config"]) == {
        "residency": "cached",
        "cache_slots": 513,
        "file_path": "/local/checkpoint-shard.safetensors"
    }
    assert default_native.args[3] == str(checkpoint)
    assert selected_native.args[3] == str(checkpoint)
    assert selections[0] == selections[1]
    assert default.bundle_dir == selected.bundle_dir


def test_llm_rejects_invalid_ple_before_checkpoint_loading():
    with pytest.raises(ServerConfigError,
                       match="Cached PLE requires cache_slots >= 2"):
        LLM(model="/nonexistent/checkpoint",
            ple_config={
                "residency": "cached",
                "cache_slots": 1
            })


def test_dflash2_tree_speculative_config_sizes_base_and_runtime(
        tmp_path, monkeypatch):
    """A DFlash2 tree request builds the base for its verify width and runs
    with that fanout and width, so the chain bundle (block-sized base) and the
    tree bundle never share a cache entry."""
    base = _checkpoint(tmp_path / "Qwen3.8-27B-NVFP4")
    draft = _checkpoint(tmp_path / "Qwen3.8-27B-DFlash2", "qwen3")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "spec_base.engine").touch()
    (bundle / "spec_draft.engine").touch()
    base_config = _engine_config(spec="dflash")
    base_config["builder_config"]["max_spec_verify_tokens"] = 32
    _write_json(bundle / "base_config.json", base_config)
    _write_json(
        bundle / "draft_config.json", {
            "engine_role": "draft",
            "builder_config": {
                "max_spec_draft_tokens": 8
            },
            "dflash_config": {
                "version": 2,
                "block_size": 8
            },
            "checkpoint_identity": {
                "version": 1,
                "sources": {}
            },
        })
    prepared_options = []

    def prepare_model(_model, _cache_dir, options, **_kwargs):
        prepared_options.append(options)
        return PreparedModel(str(bundle), str(base), str(draft))

    monkeypatch.setattr(
        "tensorrt_edgellm.server.runtime.engine_build.prepare_model",
        prepare_model)
    monkeypatch.setattr(
        "tensorrt_edgellm.server.runtime.engine._import_runtime",
        lambda: _FakeBindings)

    llm = LLM(model="Qwen/Qwen3.8-27B",
              cache_dir=str(tmp_path / "cache"),
              speculative_config={
                  "method": "dflash",
                  "model": "z-lab/Qwen3.8-27B-DFlash2",
                  "draft_top_k": 4,
                  "verify_tokens": 32,
              })

    assert prepared_options[0].max_spec_verify_tokens == 32
    assert prepared_options[0].max_spec_draft_tokens is None
    assert llm._runtime.kwargs["draft_top_k"] == 4
    assert llm._runtime.kwargs["verify_tokens"] == 32
    chain_argv = BuildOptions(spec_type="dflash").to_argv("m", "b")
    assert prepared_options[0].to_argv("m", "b") != chain_argv


@pytest.mark.parametrize("method", ["mtp", "dflash", "jetspec", "dspark"])
def test_server_build_options_are_independent_of_runtime_topology(
        tmp_path, monkeypatch, method):
    # Given one checkpoint and capacity, when runtime K changes, then the server builds the same target artifact.
    calls = []

    class BuildReached(Exception):
        pass

    def capture_build(model, cache, options, **kwargs):
        calls.append(options.to_argv(model, "bundle"))
        raise BuildReached

    monkeypatch.setattr(engine_build, "prepare_model", capture_build)
    for top_k in (1, 2):
        with pytest.raises(BuildReached):
            LLM(model="checkpoint",
                cache_dir=str(tmp_path),
                build_options=BuildOptions(spec_type=method),
                draft_top_k=top_k)
    assert calls[0] == calls[1]


def test_server_build_options_reject_removed_topology_field():
    # Given topology belongs to runtime, when constructing build options with the old field, then it is rejected.
    with pytest.raises(TypeError, match="tree_base"):
        BuildOptions(tree_base=True)
