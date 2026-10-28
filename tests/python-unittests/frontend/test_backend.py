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
"""Frontend adapter: option forwarding, reasoning, runtime details, and rates."""

import json
import types

import pytest

from experimental import server
from experimental.frontend import backend
from experimental.server.runtime import engine

from . import helpers


def test_public_cancellable_iterator_preserves_iteration_and_cleanup():
    # Given a public wrapper around an active generator.
    # When cancelled and closed, then cancellation is separate from one-time
    # cleanup.
    cancelled, drained = [], []

    def chunks():
        try:
            yield "first"
        finally:
            drained.append(True)

    stream = server.CancellableIterator(chunks(),
                                        lambda: cancelled.append(True))
    assert iter(stream) is stream
    assert next(stream) == "first"
    stream.cancel()
    assert cancelled == [True]
    assert drained == []
    stream.close()
    assert cancelled == [True, True]
    assert drained == [True]
    stream.close()
    assert cancelled == [True, True]
    assert drained == [True]
    with pytest.raises(StopIteration):
        next(stream)


def test_public_cancellable_iterator_closes_before_iteration():
    # Given a stream that has not started.
    # When closed twice, then cancellation runs once without starting the
    # producer.
    started, cancelled = [], []

    def chunks():
        started.append(True)
        yield "unexpected"

    stream = server.CancellableIterator(chunks(),
                                        lambda: cancelled.append(True))
    stream.close()
    stream.close()
    assert cancelled == [True]
    assert list(stream) == []
    assert started == []


def test_launch_options_reach_runtime_sampling_and_builder(
        tmp_path, monkeypatch):
    # Given explicit launch options.
    # When the backend loads, then runtime and sampling settings retain them.
    result = helpers.settings(
        tmp_path, "--max-input-len", "1024", "--max-kv-cache-capacity", "4096",
        "--max-batch-size", "2", "--tp-size", "1", "--plugin-path",
        "/tmp/plugin.so", "--max-image-tokens", "512",
        "--max-image-tokens-per-image", "256", "--clear-engine-cache",
        "--draft-top-k", "1", "--draft-step", "3", "--verify-tree-size", "4",
        "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}',
        "--context-cache-config", '{"enabled":true,"max_records":8}',
        "--reasoning-effort", "low", "--disable-spec-decode", "--num-logprobs",
        "2", "--stop", "END", "--stop", "DONE", "--logit-bias", '{"42":-5}',
        "--guided-decoding", '{"regex":"[a-z]+"}', "--no-reuse-context",
        "--no-cache-generated-tokens", "--no-skip-special-tokens")
    calls = {}

    def load(model, **kwargs):
        calls.update(kwargs)
        return types.SimpleNamespace(model_dir=str(tmp_path))

    monkeypatch.setattr(engine, "LLM", load)
    instance = backend.Backend(result, lambda _: None)
    assert calls["clear_engine_cache"]
    assert calls["speculative_config"].method == "mtp"
    assert calls["context_cache_config"].max_records == 8
    assert (calls["draft_top_k"], calls["draft_step"],
            calls["verify_tree_size"]) == (1, 3, 4)
    build = calls["build_options"]
    assert (build.max_input_len, build.max_kv_cache_capacity,
            build.max_batch_size) == (1024, 4096, 2)
    argv = build.to_argv(str(tmp_path), "/tmp/engine")
    for flag, value in (("--tp-size", "1"), ("--plugin-path",
                                             "/tmp/plugin.so"),
                        ("--max-image-tokens",
                         "512"), ("--max-image-tokens-per-image", "256")):
        assert argv[argv.index(flag) + 1] == value
    sampling = instance.sampling
    assert sampling.stop == ["END", "DONE"]
    assert sampling.logit_bias == {42: -5.0}
    assert sampling.guided_decoding == ("regex", "[a-z]+")
    assert sampling.reasoning_effort == "low" and sampling.disable_spec_decode
    assert sampling.num_logprobs == 2
    assert not sampling.reuse_context and not sampling.cache_generated_tokens
    assert not sampling.skip_special_tokens


@pytest.mark.parametrize("thinking,skip_special", [(True, False),
                                                   (False, True)])
def test_backend_splits_thinking_and_keeps_native_sampling(
        tmp_path, thinking, skip_special):
    # Given streamed thinking and special tokens.
    # When adapted, then reasoning is separated without changing sampling.
    (tmp_path / "config.json").write_text(json.dumps({"model_type":
                                                      "qwen3_5"}))
    model = backend.Backend.__new__(backend.Backend)
    model.sampling = helpers.settings(
        tmp_path, "--enable-thinking" if thinking else "--no-enable-thinking",
        "--skip-special-tokens"
        if skip_special else "--no-skip-special-tokens").sampling
    closed = []
    text = "<think>Let me check.</think>Four." if thinking else "Four."

    def generate(_messages, sampling, on_metrics):
        assert sampling.enable_thinking == thinking
        assert sampling.skip_special_tokens == skip_special
        decoded = text + ("" if sampling.skip_special_tokens else "<|im_end|>")
        chunks = (engine.StreamDelta(text=char) for char in decoded)
        return server.CancellableIterator(chunks, lambda: closed.append(True))

    model.llm = types.SimpleNamespace(model_dir=str(tmp_path),
                                      generate_stream=generate)
    stream = model.stream([], lambda _: None)
    parts = list(stream)
    assert closed == [True]
    stream.close()
    assert "".join(
        part.text for part in parts
        if part.field == "content") == ("Four." +
                                        ("" if skip_special else "<|im_end|>"))
    assert "".join(
        part.text for part in parts
        if part.field == "reasoning") == ("Let me check." if thinking else "")
    assert closed == [True, True]
    assert model.sampling.skip_special_tokens == skip_special


@pytest.mark.parametrize("quantized", [False, True])
def test_runtime_details_use_checkpoint_and_loaded_speculation(
        tmp_path, monkeypatch, quantized):
    # Given checkpoint metadata.
    # When details are requested, then labels reflect weights and active
    # speculation.
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"text_config": {
            "dtype": "bfloat16"
        }}))
    (tmp_path / "base_config.json").write_text(
        json.dumps({"spec_decode_type": "mtp"}))
    if quantized:
        (model_dir / "hf_quant_config.json").write_text(
            json.dumps({"quantization": {
                "quant_algo": "NVFP4"
            }}))
    model = backend.Backend.__new__(backend.Backend)
    model.llm = types.SimpleNamespace(model_dir=str(model_dir),
                                      model_id="Qwen3.5-27B",
                                      bundle_dir=str(tmp_path),
                                      max_kv_cache_capacity=8192,
                                      has_draft_model=quantized)
    model.sampling = engine.SamplingParams()
    monkeypatch.setattr(backend, "_device_name", lambda: "NVIDIA Thor")
    details = model.details()
    assert model.context_capacity == 8192
    assert details == {
        "model": "Qwen3.5-27B",
        "device": "NVIDIA Thor",
        "weights": f"{'NVFP4' if quantized else 'BF16'} (checkpoint)",
        "speculative mode": "mtp" if quantized else "off",
    }


@pytest.mark.parametrize("tokens,decoded,seconds,expected", [
    (1, 0, 0.0, None),
    (3, 2, 0.25, 8.0),
    (3, 3, 0.25, 12.0),
    (3, 2, 0.0, None),
])
def test_decode_rate_uses_native_phase_count(tokens, decoded, seconds,
                                             expected):
    # Given native phase metrics, when rates are rendered, then only decoded
    # tokens count.
    metrics = engine.GenerationMetrics(generated_tokens=tokens,
                                       decoded_tokens=decoded,
                                       generation_time=seconds)
    assert backend.rates(metrics) == (None, expected)
