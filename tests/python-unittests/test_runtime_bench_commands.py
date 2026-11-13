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

import copy

import pytest
from conftest import EnvironmentConfig
from defs.config import ModelType, TaskType, TestConfig
from defs.utils.command_execution import _validate_runtime_bench_report
from defs.utils.command_generation import generate_kernel_bench_commands


def _config(tmp_path, suffix=""):
    environment = EnvironmentConfig(llm_sdk_dir=".",
                                    llm_models_dir="/models",
                                    edgellm_data_dir="/data",
                                    onnx_dir="/onnx",
                                    engine_dir="/engine",
                                    build_dir="/build",
                                    test_log_dir=str(tmp_path),
                                    trt_package_dir="/trt")
    return TestConfig.from_param_string(
        "Qwen3.5-0.8B-FP8-fp8-mxsl2048-mtp-mxbs1-mxil1024" + suffix,
        ModelType.LLM,
        TaskType.KERNEL_BENCH,
        environment,
        validate_environment=False)


def test_runtime_bench_uses_existing_engine_and_preserves_drafting(tmp_path):
    # Given the same built deployment with a fixed-round workload,
    # when creating benchmark commands, then workload controls do not change its engine identity or drafting.
    config = _config(tmp_path, "-mode_fixed_round-il32")
    baseline = _config(tmp_path)
    first = generate_kernel_bench_commands(config,
                                           {"llm_bench": "llm_bench"})[0][0]
    second = generate_kernel_bench_commands(config,
                                            {"llm_bench": "llm_bench"})[0][0]
    assert config.get_llm_engine_dir() == baseline.get_llm_engine_dir()
    assert f"--engineDir={baseline.get_llm_engine_dir()}" in first
    assert "--mode=fixed_round" in first
    assert "--inputLen=32" in first
    assert "--specDecode" in first
    assert f"--specDraftStep={config.eagle_draft_step}" in first
    assert f"--specVerifyTokens={config.max_spec_verify_tokens}" in first
    assert "--profile" not in first
    assert next(x for x in first if x.startswith("--outputDir=")) != next(
        x for x in second if x.startswith("--outputDir="))


@pytest.mark.parametrize("suffix",
                         ["-mode_prefill", "-mode_fixed_round-pkv32"])
def test_runtime_bench_rejects_legacy_component_semantics(tmp_path, suffix):
    # Given legacy component controls, when generating a request benchmark,
    # then reject them instead of silently reporting a different timing boundary.
    with pytest.raises(ValueError, match="legacy"):
        generate_kernel_bench_commands(_config(tmp_path, suffix),
                                       {"llm_bench": "llm_bench"})


def _report():
    return {
        "schema_version":
        2,
        "mode":
        "fixed_round",
        "iterations":
        1,
        "samples": [{
            "wall_ms": 2.0,
            "output_token_counts": [2, 2],
            "decode_round_counts": [1, 1],
            "invocations": [{
                "stage": "llm_prefill"
            }]
        }]
    }


def test_runtime_bench_report_requires_completed_requested_work():
    # Given a report covering both requested sequences, when validating it,
    # then accept it and reject a complete-looking report for a different workload.
    report = _report()
    _validate_runtime_bench_report(report, 1, 2, "fixed_round")
    for changes in [{
            "schema_version": 1
    }, {
            "mode": "generate"
    }, {
            "iterations": 0
    }, {
            "samples": []
    }]:
        changed = dict(report, **changes)
        with pytest.raises(ValueError):
            _validate_runtime_bench_report(changed, 1, 2, "fixed_round")


@pytest.mark.parametrize("changes", [
    {
        "wall_ms": float("nan")
    },
    {
        "wall_ms": 0
    },
    {
        "output_token_counts": []
    },
    {
        "output_token_counts": [2, 0]
    },
    {
        "decode_round_counts": [1, 2]
    },
    {
        "invocations": []
    },
])
def test_runtime_bench_report_rejects_partial_or_invalid_samples(changes):
    # Given a process that exits successfully but emits invalid measurement data,
    # when checking its report, then reject the sample instead of accepting a false green.
    report = copy.deepcopy(_report())
    report["samples"][0].update(changes)
    with pytest.raises(ValueError):
        _validate_runtime_bench_report(report, 1, 2, "fixed_round")


def test_model_owned_l0_selects_bench_against_the_built_deployment(tmp_path):
    # Given the current model-owned B100 priority, when collecting its benchmark,
    # then its workload runs on the same checkpoint/engine as the selected MTP build.
    from conftest import _get_test_list_file

    selected = _get_test_list_file("l0_e2e_b100")["tests"]
    bench, = [
        item for item in selected
        if getattr(item, "node", "").endswith("::test_kernel_bench")
    ]
    deployment = bench.case_id.rsplit("/", 1)[0]
    build = next(item for item in selected
                 if getattr(item, "case_id", None) == deployment + "/build")
    environment = EnvironmentConfig(llm_sdk_dir=".",
                                    llm_models_dir="/models",
                                    edgellm_data_dir="/data",
                                    onnx_dir="/onnx",
                                    engine_dir="/engine",
                                    build_dir="/build",
                                    test_log_dir=str(tmp_path),
                                    trt_package_dir="/trt")
    bench_config = TestConfig.from_param_string(bench,
                                                ModelType.LLM,
                                                TaskType.KERNEL_BENCH,
                                                environment,
                                                validate_environment=False)
    build_config = TestConfig.from_param_string(build,
                                                ModelType.LLM,
                                                TaskType.BUILD,
                                                environment,
                                                validate_environment=False)
    assert (bench_config.get_llm_engine_dir(), bench_config.checkpoint_dir,
            bench_config.bench_mode, bench_config.input_len,
            bench_config.is_mtp) == (build_config.get_llm_engine_dir(),
                                     build_config.checkpoint_dir,
                                     "fixed_round", 32, True)
