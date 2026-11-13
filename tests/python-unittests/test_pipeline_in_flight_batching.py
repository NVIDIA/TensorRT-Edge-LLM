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
"""In-flight batching coverage of model cases.

``engine.in_flight_batching`` starts the server under
``--enable-in-flight-batching``. It is a serving option, so it belongs to the
server lifecycle phase and never to an engine build. ``compact_engine_tests``
runs the real-engine compact session tests on an ONNX case's engine,
``ifb_trace`` gates mixed steps on it, and ``ifb_server`` checks the server's
scheduling options end to end on a server case.
"""

import logging
from copy import deepcopy

import pytest
from conftest import EnvironmentConfig
from defs import test_llm_pipeline as llm_pipeline
from defs import test_server_pipeline as server_pipeline
from defs.config import ModelType, TaskType, TestConfig
from defs.model_test_config import _parse_model_document

_SERVER_DOCUMENT = {
    "schema_version":
    1,
    "model_id":
    "Qwen/Qwen3.5-0.8B",
    "inputs": ["text", "image"],
    "outputs": ["text"],
    "cases": [{
        "name":
        "in-flight-batching-server",
        "mode":
        "server",
        "platform":
        "l0_e2e_a30",
        "quantized_model_id_or_dir":
        "/models/Qwen3.5-0.8B",
        "inputs": ["text"],
        "requirements": {
            "min_sm": 80,
            "min_gpu_memory_gb": 5,
        },
        "engine": {
            "max_kv_cache_capacity": 4096,
            "max_input_len": 1024,
            "max_batch_size": 4,
            "in_flight_batching": True,
            "context_reuse": True,
        },
        "inference": [{
            "name": "basic",
            "scenario": "openai",
            "input_file": "tests/test_cases/llm_basic.json",
        }],
    }],
}


def _environment(tmp_path):
    return EnvironmentConfig(llm_sdk_dir=".",
                             llm_models_dir="/models",
                             edgellm_data_dir="/data",
                             onnx_dir="/onnx",
                             engine_dir="/engine",
                             build_dir="/build",
                             test_log_dir=str(tmp_path),
                             trt_package_dir="/trt")


def test_server_case_keeps_serving_options():
    tests = _parse_model_document(_SERVER_DOCUMENT, "test")

    (server, ) = tests
    assert server.node.endswith("::test_http_server_lifecycle")
    assert server.model_parameters["in_flight_batching"] is True
    assert server.model_parameters["context_reuse"] is True
    assert server.model_parameters["max_batch_size"] == 4


# yapf: disable
@pytest.mark.parametrize("switch, case, error", [
    (True, {}, None),
    (False, {}, "requires engine.in_flight_batching"),
    (True, dict(spec_decode_config="mtp"), "no speculative decoding"),
], ids=["ifb", "without_ifb", "speculative"])
# yapf: enable
def test_ifb_server_scenario_runs_its_own_server_phase(switch, case, error):
    """An ifb_server item of a server LLM case with engine.in_flight_batching becomes its own
    TestHLAPI::test_ifb_http_server phase beside the openai lifecycle, with the case's engine limits; a case
    without in_flight_batching or with speculative decoding is refused."""
    document = deepcopy(_SERVER_DOCUMENT)
    document["cases"][0].update(case)
    document["cases"][0]["engine"]["in_flight_batching"] = switch
    document["cases"][0]["inference"].append({
        "name": "ifb",
        "scenario": "ifb_server"
    })
    if error:
        with pytest.raises(ValueError, match=error):
            _parse_model_document(document, "test")
        return

    lifecycle, ifb = _parse_model_document(document, "test")
    assert lifecycle.node.endswith("::test_http_server_lifecycle")
    assert [item["scenario"] for item in lifecycle.inference] == ["openai"]
    assert ifb.node.endswith("::TestHLAPI::test_ifb_http_server")
    assert ifb.model_parameters["in_flight_batching"] is True
    assert ifb.model_parameters["max_batch_size"] == 4


def _with_mode(document, mode):
    case = document["cases"][0]
    case["mode"] = mode
    case["inference"] = [{
        "name": "basic",
        "scenario": "generate",
        "input_file": "tests/test_cases/llm_basic.json",
        "checks": {
            "accuracy": {
                "required": ["rouge1"],
                "max_regression": 0.3,
            }
        },
    }]


def _with_evaluator(document):
    document["cases"][0]["inference"] = [{
        "name": "mmlu",
        "scenario": "nemo_evaluator",
        "evaluator": {
            "eval_type": "mmlu"
        },
        "score": {
            "key": "score",
            "min": 0.2
        },
    }]


def _with_string_switch(document):
    document["cases"][0]["engine"]["in_flight_batching"] = "yes"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda document: _with_mode(document, "onnx"),
         "requires mode: server"),
        (lambda document: _with_mode(document, "direct_builder"),
         "requires mode: server"),
        (_with_evaluator, "not by nemo_evaluator"),
        (_with_string_switch, "in_flight_batching"),
    ],
    ids=["onnx", "direct_builder", "nemo_evaluator", "not_a_boolean"],
)
def test_switch_rejected_unless_boolean(mutate, message):
    document = deepcopy(_SERVER_DOCUMENT)
    mutate(document)

    with pytest.raises(ValueError, match=message):
        _parse_model_document(document, "test")


@pytest.mark.parametrize("in_flight_batching", [False, True])
def test_server_command_follows_the_switch(tmp_path, monkeypatch,
                                           in_flight_batching):
    param = "Qwen3.5-0.8B-fp16-mxsl4096-mxbs4-mxil1024-llm_basic"
    if in_flight_batching:
        param = param.replace("-llm_basic", "-ifb-llm_basic")
    config = TestConfig.from_param_string(param,
                                          ModelType.LLM,
                                          TaskType.INFERENCE,
                                          _environment(tmp_path),
                                          validate_environment=False)
    assert bool(config.in_flight_batching) is in_flight_batching
    monkeypatch.setattr(
        server_pipeline.TestHLAPI, "_checkpoint_model_and_spec",
        staticmethod(lambda *_: ("/models/Qwen3.5-0.8B", "none", "")))

    command = server_pipeline.TestHLAPI._server_cli_args(
        config, _environment(tmp_path), None, logging.getLogger("test"))

    assert ("--enable-in-flight-batching" in command) is in_flight_batching
    assert command[command.index("--max-batch-size") + 1] == "4"


def _compact_engine_document(**inference):
    """_SERVER_DOCUMENT as a text-only ONNX case with one compact_engine_tests
    phase."""
    document = deepcopy(_SERVER_DOCUMENT)
    case = document["cases"][0]
    case.update(mode="onnx", platform="l0_e2e_b100")
    case["engine"] = dict(case["engine"], in_flight_batching=False)
    case["inference"] = [
        dict(name="compact-engine-tests",
             scenario="compact_engine_tests",
             **inference)
    ]
    return document


def test_compact_engine_phase_runs_on_the_case_engine(tmp_path):
    """A compact_engine_tests phase runs the real-engine compact session tests against the engine its case built."""
    _, build, phase = _parse_model_document(_compact_engine_document(), "test")

    assert phase.node.endswith("::TestLLMPipeline::test_compact_engine_tests")
    configs = [
        TestConfig.from_param_string(param, ModelType.LLM, task, _environment(tmp_path), validate_environment=False)
        for param, task in ((build, TaskType.BUILD), (phase, TaskType.INFERENCE))
    ]  # yapf: disable
    assert configs[0].get_llm_engine_dir() == configs[1].get_llm_engine_dir()


@pytest.mark.parametrize(
    "case, inference, message",
    [
        (dict(mode="direct_builder"), {}, "requires an ONNX LLM deployment"),
        (dict(inputs=["text", "image"]), {},
         "requires an ONNX LLM deployment"),
        (dict(spec_decode_config="mtp"), {}, "without speculative decoding"),
        ({}, dict(test_case="llm_basic"), "test_case"),
    ],
    ids=["direct_builder", "vlm", "speculative", "test_case"],
)
def test_compact_engine_scenario_rejected_outside_onnx_llm(
        case, inference, message):
    """compact_engine_tests accepts only a vanilla ONNX LLM case and no ignored options."""
    document = _compact_engine_document(**inference)
    document["cases"][0].update(case)

    with pytest.raises(ValueError, match=message):
        _parse_model_document(document, "test")


# yapf: disable
_ENGINE_TEST_FAILURES = {
    "skipped": ("[  PASSED  ] 1 test.\n[  SKIPPED ] 1 test, listed below:\n",
                "1 CompactSessionEngineTest.* case(s) skipped"),
    "none_ran": ("[  PASSED  ] 0 tests.\n", "no CompactSessionEngineTest.* case ran"),
}
# yapf: enable


@pytest.mark.parametrize("output, message",
                         _ENGINE_TEST_FAILURES.values(),
                         ids=_ENGINE_TEST_FAILURES.keys())
def test_compact_engine_phase_fails_unless_every_case_ran_and_passed(
        output, message):
    """The compact_engine_tests phase fails when a selected real-engine case skipped or none ran."""
    check = llm_pipeline._compact_engine_test_failures
    assert check("[  PASSED  ] 2 tests.\n") == []

    failures = check(output)
    assert any(message in failure for failure in failures), failures


def _trace_document(**inference):
    """_SERVER_DOCUMENT as a text-only ONNX case with one ifb_trace phase."""
    document = deepcopy(_SERVER_DOCUMENT)
    case = document["cases"][0]
    case.update(mode="onnx", platform="l0_e2e_b100")
    case["engine"] = dict(case["engine"], in_flight_batching=False)
    case["inference"] = [dict(name="ifb", scenario="ifb_trace", **inference)]
    return document


@pytest.mark.parametrize("inference, trace",
                         [({}, "ifb_smoke"),
                          (dict(test_case="ifb_priority"), "ifb_priority")],
                         ids=["default_trace", "named_trace"])
def test_trace_phase_runs_on_the_case_engine(tmp_path, inference, trace):
    """An ifb_trace phase drives the engine its case built, with the trace its test_case names (ifb_smoke by
    default)."""
    _, build, phase = _parse_model_document(_trace_document(**inference),
                                            "test")

    assert phase.node.endswith("::TestServerPipeline::test_ifb_trace")
    configs = [
        TestConfig.from_param_string(param, ModelType.LLM, task, _environment(tmp_path), validate_environment=False)
        for param, task in ((build, TaskType.BUILD), (phase, TaskType.INFERENCE))
    ]  # yapf: disable
    assert configs[0].get_llm_engine_dir() == configs[1].get_llm_engine_dir()
    assert configs[1].test_case == trace


# yapf: disable
@pytest.mark.parametrize("case, inference, message", [
    (dict(mode="direct_builder"), {}, "requires an ONNX LLM deployment"),
    (dict(inputs=["text", "image"]), {}, "requires an ONNX LLM deployment"),
    (dict(spec_decode_config="mtp"), {}, "without speculative decoding"),
    ({}, dict(input_file="tests/test_cases/llm_basic.json"), "input_file"),
    ({}, dict(test_case="ifb_multimodal"), "test_case"),
], ids=["direct_builder", "vlm", "speculative", "input_file", "media_trace"])
# yapf: enable
def test_trace_scenario_rejected_outside_onnx_llm(case, inference, message):
    """ifb_trace accepts only a vanilla ONNX LLM case, a text-only trace and no ignored options."""
    document = _trace_document(**inference)
    document["cases"][0].update(case)

    with pytest.raises(ValueError, match=message):
        _parse_model_document(document, "test")


def _trace_result(mode="run", resolved="mixed", mixed=5, partial=1):
    run = dict(label="run", policy=dict(resolved=resolved),
               metrics_delta=dict(mixed_steps=mixed, partial_prompt_chunks=partial))  # yapf: disable
    return dict(mode=mode, runs=[run])


# yapf: disable
@pytest.mark.parametrize("outcome, message", [
    (_trace_result(resolved="prefillFirst"), "resolved policy 'prefillFirst'"),
    (_trace_result(mixed=0), "mixed_steps=0"),
    (_trace_result(partial=None), "partial_prompt_chunks=None"),
    (_trace_result(mode="reference"), "no run-mode runs"),
], ids=["prefill_first", "no_mixed_steps", "no_partial_chunks", "reference"])
# yapf: enable
def test_trace_phase_fails_without_mixed_steps(outcome, message):
    """The ifb_trace phase fails unless every trace run resolved to mixed and formed mixed, chunked steps."""
    check = server_pipeline._ifb_mixed_step_failures
    assert check(_trace_result()) == []
    assert any(message in failure
               for failure in check(outcome)), check(outcome)
