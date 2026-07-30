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
"""Tests for `LLM(engine_dir=...)` routing to LLM vs spec-decode paths."""

from types import SimpleNamespace

import pytest

from experimental.server.engine import LLM, SamplingParams
from experimental.server.engine_layout import (EngineType, detect_engine_type,
                                               validate_spec_decode_engine_dir)


def _touch(path):
    with open(path, "w"):
        pass


# ---------------------------------------------------------------------------
# engine_layout helpers
# ---------------------------------------------------------------------------


def test_detect_engine_type_spec_decode(tmp_path):
    _touch(tmp_path / "spec_base.engine")
    _touch(tmp_path / "spec_draft.engine")
    assert detect_engine_type(str(tmp_path)) == EngineType.SPEC_DECODE


def test_detect_engine_type_llm(tmp_path):
    _touch(tmp_path / "llm.engine")
    assert detect_engine_type(str(tmp_path)) == EngineType.LLM


def test_detect_engine_type_unknown(tmp_path):
    assert detect_engine_type(str(tmp_path)) == EngineType.UNKNOWN


def test_validate_spec_decode_engine_dir_requires_both(tmp_path):
    assert not validate_spec_decode_engine_dir(str(tmp_path))
    _touch(tmp_path / "spec_base.engine")
    assert not validate_spec_decode_engine_dir(str(tmp_path))
    _touch(tmp_path / "spec_draft.engine")
    assert validate_spec_decode_engine_dir(str(tmp_path))


# ---------------------------------------------------------------------------
# LLM._init_from_engine routing
# ---------------------------------------------------------------------------


class _BareLLM(LLM):
    """LLM subclass that skips runtime loading, exposing just the routing."""

    # pylint: disable=super-init-not-called
    def __init__(self, engine_dir: str, visual_engine_dir: str = ""):
        self._eagle_engine_dir = ""
        self._tool_template_formatter = None
        self._model_id = "test"
        self._init_from_engine(engine_dir, visual_engine_dir)


def test_init_from_engine_routes_spec_decode_dir(tmp_path):
    """Spec-decode dirs promote engine_dir to _eagle_engine_dir."""
    _touch(tmp_path / "spec_base.engine")
    _touch(tmp_path / "spec_draft.engine")

    llm = _BareLLM(str(tmp_path))

    assert llm._engine_dir == str(tmp_path)
    assert llm._model_dir == str(tmp_path)
    # Key routing side-effect: spec-decode dispatch downstream reads
    # `_eagle_engine_dir`, so this promotion is what actually enables the fix.
    assert llm._eagle_engine_dir == str(tmp_path)
    assert llm._visual_engine_dir == ""
    assert llm._is_vlm is False


def test_init_from_engine_ignores_visual_dir_for_spec_decode(tmp_path, caplog):
    """visual_engine_dir is meaningless for spec-decode and must not raise."""
    _touch(tmp_path / "spec_base.engine")
    _touch(tmp_path / "spec_draft.engine")
    visual_dir = tmp_path / "visual"
    visual_dir.mkdir()

    llm = _BareLLM(str(tmp_path), visual_engine_dir=str(visual_dir))

    # Visual dir was ignored (spec-decode engines have no vision).
    assert llm._visual_engine_dir == ""
    assert llm._is_vlm is False


def test_init_from_engine_spec_decode_missing_draft_engine(tmp_path):
    """Half-populated spec-decode dir must raise a clear error."""
    _touch(tmp_path / "spec_base.engine")
    # spec_draft.engine deliberately absent

    with pytest.raises(ValueError, match="spec_base.engine/spec_draft.engine"):
        _BareLLM(str(tmp_path))


def test_init_from_engine_vanilla_llm_still_works(tmp_path):
    """Vanilla llm.engine dirs must continue to route through the LLM path."""
    _touch(tmp_path / "llm.engine")

    llm = _BareLLM(str(tmp_path))

    assert llm._engine_dir == str(tmp_path)
    # Vanilla path does NOT set _eagle_engine_dir.
    assert llm._eagle_engine_dir == ""
    assert llm._is_vlm is False


def test_init_from_engine_unknown_dir_raises(tmp_path):
    """A directory with neither llm.engine nor spec_base.engine is rejected."""
    with pytest.raises(ValueError, match="llm.engine not found"):
        _BareLLM(str(tmp_path))


def test_engine_dir_rejects_separate_spec_decode_engine_dir():
    with pytest.raises(ValueError, match="complete pre-built engine bundle"):
        LLM(engine_dir="/engine", eagle_engine_dir="/other-spec-engine")


def test_load_runtime_spec_decode_passes_context_cache_config(monkeypatch):
    captured_args = []

    class FakeContextCacheConfig:

        def __init__(self):
            self.enabled = False
            self.max_records = 0
            self.recurrent_snapshot_pool_bytes = 0
            self.partial_kv_snapshot_pool_bytes = 0

    class FakeLLMRuntime:

        def __init__(self, *args):
            captured_args.append(args)

        def capture_decoding_cuda_graph(self):
            pass

    fake_rt = SimpleNamespace(ContextCacheConfig=FakeContextCacheConfig,
                              LLMRuntime=FakeLLMRuntime)
    monkeypatch.setattr("experimental.server.engine._import_runtime",
                        lambda: fake_rt)

    llm = object.__new__(LLM)
    llm._engine_dir = "/spec-engine"
    llm._visual_engine_dir = ""
    llm._eagle_engine_dir = "/spec-engine"
    llm._draft_top_k = 8
    llm._draft_step = 6
    llm._verify_tree_size = 60
    llm._enable_context_reuse = True
    llm._context_cache_max_records = 17
    llm._context_cache_recurrent_snapshot_pool_bytes = 1024
    llm._context_cache_partial_kv_snapshot_pool_bytes = 256

    llm._load_runtime()

    assert len(captured_args) == 1
    args = captured_args[0]
    assert args[:6] == ("/spec-engine", "", {}, 8, 6, 60)
    context_cache = args[6]
    assert context_cache.enabled is True
    assert context_cache.max_records == 17
    assert context_cache.recurrent_snapshot_pool_bytes == 1024
    assert context_cache.partial_kv_snapshot_pool_bytes == 256


@pytest.mark.parametrize(
    "enable_context_reuse,prefill_state_only,expected_policy",
    [
        (True, True, "prefill-state-only"),
        (True, False, "default"),
        (False, True, "default"),
    ],
)
def test_make_generation_request_sets_prefill_only_context_cache_policy(
        monkeypatch, enable_context_reuse, prefill_state_only,
        expected_policy):

    class FakeGenerationRequest:

        def __init__(self):
            self.context_cache_commit_policy = "default"
            self.context_cache_replay_tail_length = 0

    class FakeRequest:

        def __init__(self, messages):
            self.messages = messages

    class FakeRuntime:
        LLMGenerationRequest = FakeGenerationRequest
        Request = FakeRequest
        ContextCacheCommitPolicy = SimpleNamespace(
            PREFILL_STATE_ONLY="prefill-state-only")

        @staticmethod
        def has_draft_model():
            return False

    llm = object.__new__(LLM)
    llm._rt = FakeRuntime
    llm._runtime = FakeRuntime()
    llm._enable_context_reuse = enable_context_reuse
    llm._context_cache_prefill_state_only = prefill_state_only
    llm._recurrent_capture_interval = 0
    llm._prepare_messages_for_runtime = lambda *args, **kwargs: ([], [], True,
                                                                 True, 7)
    monkeypatch.setattr("experimental.server.engine._load_audio_buffers",
                        lambda *args: [])

    request = llm._make_generation_request([{
        "role": "user",
        "content": "hello"
    }], SamplingParams())

    assert request.context_cache_commit_policy == expected_policy
    assert request.context_cache_replay_tail_length == 7
