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

import conftest
import pytest
from defs import config, model_test_config
from defs.utils import command_generation

_CUSTOM_DESCRIPTION = """## Summary

<!-- EDGE_LLM_L0_TESTS -->
```yaml
schema_version: 1
model_id: Qwen/Qwen3-VL-8B-Instruct
inputs: [text, image]
outputs: [text]
cases:
  - name: vitdp-tp2-smoke
    mode: onnx
    platform: l0_e2e_super_thor
    quantized_model_id_or_dir: /models/Qwen3-VL-8B-Instruct-NVFP4
    requirements:
      min_sm: 100
      min_gpu_memory_gb: 12
    engine:
      tp_size: 2
      max_batch_size: 1
      max_input_len: 4096
      max_kv_cache_capacity: 4608
      min_image_tokens: 4
      max_image_tokens: 4096
      max_image_tokens_per_image: 512
    inference:
      - name: basic
        scenario: generate
        input_file: tests/test_cases/vlm_basic.json
        checks:
          accuracy:
            required: [rouge1, rouge_l]
            max_regression: 0.30
          performance:
            required: [prefill_tokens_per_second, generation_tokens_per_second]
            max_regression: 0.30
```
"""


def _case(phase, tp_size):
    priority = ("l0_onnx_export_1"
                if phase == "export" else "l0_e2e_super_thor")
    description = _CUSTOM_DESCRIPTION.replace("tp_size: 2",
                                              f"tp_size: {tp_size}")
    return next(
        case
        for case in model_test_config.load_model_tests(priority, description)
        if case.case_id == f"Qwen3-VL-8B-Instruct/vitdp-tp2-smoke/{phase}")


def _config(tmp_path, phase, task_type, tp_size=2):
    environment = conftest.EnvironmentConfig(llm_sdk_dir=".",
                                             llm_models_dir="/models",
                                             edgellm_data_dir="/data",
                                             onnx_dir="/onnx",
                                             engine_dir="/engine",
                                             build_dir="/build",
                                             test_log_dir=str(tmp_path),
                                             trt_package_dir="/trt")
    return config.TestConfig.from_param_string(_case(phase, tp_size),
                                               config.ModelType.VLM,
                                               task_type,
                                               environment,
                                               validate_environment=False)


def test_vlm_tp2_custom_export_accepts_tensor_parallelism(tmp_path):
    """Route a custom VLM TP2 case to TP export and accept its TP size."""
    case = _case("export", 2)
    assert case.node.endswith("::test_llm_loader_tp_export")
    export = _config(tmp_path, "export", config.TaskType.EXPORT)
    assert export.tp_size == 2


@pytest.mark.parametrize("tp_size", [1, 2])
def test_vlm_build_shards_only_llm(tmp_path, tp_size):
    """Build each LLM rank and exactly one shared visual engine."""
    build = _config(tmp_path, "build", config.TaskType.BUILD, tp_size)
    commands = command_generation.generate_build_commands(
        build, {
            "llm_build": "llm_build",
            "visual_build": "visual_build"
        })
    assert [command[0] for command, _ in commands
            ] == (["llm_build"] * tp_size + ["visual_build"])
    for rank, (command, _) in enumerate(commands[:-1]):
        expected = ([f"--tpSize={tp_size}", f"--tpRank={rank}"]
                    if tp_size > 1 else [])
        assert [arg for arg in command if arg.startswith("--tp")] == expected
    visual_command = commands[-1][0]
    assert not any(arg.startswith("--tp") for arg in visual_command)
    assert "--minImageTokens=4" in visual_command
    assert "--maxImageTokens=4096" in visual_command
    assert "--maxImageTokensPerImage=512" in visual_command


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize(
    "task_type", [config.TaskType.INFERENCE, config.TaskType.E2E_BENCH])
def test_vlm_runtime_preserves_tp_size_and_visual_engine(
        tmp_path, tp_size, task_type):
    """Inference and benchmarking use the same TP size and engines as build."""
    build = _config(tmp_path, "build", config.TaskType.BUILD, tp_size)
    runtime = _config(tmp_path, "basic", task_type, tp_size)
    input_file = tmp_path / "input.json"
    input_file.write_text("{}", encoding="utf-8")
    runtime.input_file = str(input_file)
    generate = (command_generation.generate_inference_commands
                if task_type == config.TaskType.INFERENCE else
                command_generation.generate_e2e_bench_commands)
    commands = generate(runtime, {"llm_inference": "llm_inference"})
    assert len(commands) == 1
    command = commands[0][0]
    assert command[0] == "llm_inference"
    assert f"--engineDir={build.get_llm_engine_dir()}" in command
    assert f"--multimodalEngineDir={build.get_visual_engine_dir()}" in command
    expected = [f"--tpSize={tp_size}"] if tp_size > 1 else []
    assert [arg for arg in command if arg.startswith("--tp")] == expected


def test_vlm_tp2_rejects_speculative_build(tmp_path):
    """Reject unsupported TP speculative decoding before building any engine."""
    build = _config(tmp_path, "build", config.TaskType.BUILD)
    build.is_eagle = True
    with pytest.raises(ValueError,
                       match="Tensor-parallel speculative decoding"):
        command_generation.generate_build_commands(
            build, {
                "llm_build": "llm_build",
                "visual_build": "visual_build"
            })
