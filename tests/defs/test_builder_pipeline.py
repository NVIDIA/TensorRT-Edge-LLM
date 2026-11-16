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
"""ONNX-less engine builds and C++ runtime execution.

`test_build_and_run` builds and runs one case locally. `test_build` builds once
on the local or remote target and `test_dataset` reuses that engine for each
model-list dataset case.
"""

import json
import os
import pathlib
import shutil
import sys
from typing import Any, Dict, List, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import run_with_trt_env, timer_context

from . import test_server_pipeline
from .config import (ModelType, TaskType, TestConfig,
                     infer_checkpoint_export_model_type)
from .utils import command_execution, multistage_parity
from .utils.accuracy import check_accuracy_with_dataset
from .utils.command_execution import (_GUIDED_DECODING_TEST_CASES,
                                      _check_baseline_regression,
                                      _check_guided_decoding_conformance,
                                      check_declared_metrics,
                                      check_result_failures,
                                      prepare_spec_decode_input)


def _engine_dir(config: TestConfig, weightless_engine: bool = False) -> str:
    """Return a distinct artifact directory for each weight transport mode."""
    suffix = "-weightless" if weightless_engine else ""
    return os.path.join(config.get_engine_base_dir(),
                        f"onnxless-{config.get_engine_id()}{suffix}")


def _speculative_build_args(config: TestConfig) -> List[str]:
    if config.is_mtp:
        if config.model_name.lower().startswith("gemma-"):
            return [
                "--spec-type", "gemma4_mtp", "--draft-model-dir",
                config.get_gemma4_mtp_assistant_model_dir()
            ]
        return ["--spec-type", "mtp"]
    if config.is_dflash:
        return [
            "--spec-type", "dflash", "--draft-model-dir",
            config.get_dflash_draft_model_dir()
        ]
    if config.is_jetspec:
        return [
            "--spec-type", "jetspec", "--draft-model-dir",
            config.get_jetspec_draft_model_dir()
        ]
    if config.is_domino:
        return [
            "--spec-type", "domino", "--draft-model-dir",
            config.get_domino_draft_model_dir()
        ]
    if config.is_dspark:
        return [
            "--spec-type", "dspark", "--draft-model-dir",
            config.get_dspark_draft_model_dir()
        ]
    if config.is_eagle:
        return [
            "--spec-type", "eagle3", "--draft-model-dir",
            config.get_eagle_draft_checkpoint_dir()
        ]
    return []


def _build_command(config: TestConfig,
                   model_dir: str,
                   engine_dir: str,
                   env_config: EnvironmentConfig,
                   tp_size: int = 1,
                   tp_rank: int = 0,
                   weightless_engine: bool = False) -> List[str]:
    """Construct the direct-builder command with an optional experimental transport."""
    components = "llm" if tp_size > 1 else "all"
    command = [
        sys.executable,
        "-m",
        "tensorrt_edgellm.builder",
        "--model-dir",
        model_dir,
        "--engine-dir",
        engine_dir,
        "--components",
        components,
        "--plugin-path",
        os.path.join(env_config.build_dir, "lib",
                     "libNvInfer_edgellm_plugin.so"),
        "--max-input-len",
        str(config.max_input_len),
        "--max-kv-cache-capacity",
        str(config.max_kv_cache_capacity or config.max_seq_len),
        "--max-batch-size",
        str(config.max_batch_size),
        "--max-lora-rank",
        str(config.max_lora_rank or 0),
        "--max-spec-verify-tokens",
        str(config.max_spec_verify_tokens),
        "--max-spec-draft-tokens",
        str(config.max_spec_draft_tokens),
        "--min-image-tokens",
        str(config.min_image_tokens or 4),
        "--max-image-tokens",
        str(config.max_image_tokens or 1024),
        "--max-image-tokens-per-image",
        str(config.max_image_tokens_per_image or 512),
        "--min-time-steps",
        str(config.min_time_steps or 100),
        "--max-time-steps",
        str(config.max_time_steps or 6000),
    ]
    command.extend(_speculative_build_args(config))
    if weightless_engine:
        command.append("--weightless-engine")
    if tp_size > 1:
        command.extend(["--tp-size", str(tp_size), "--tp-rank", str(tp_rank)])
    if config.fp8_embedding:
        command.append("--fp8-embedding")
    if config.debug:
        command.append("--verbose")
    return command


def _runtime_command(config: TestConfig,
                     model_dir: str,
                     engine_dir: str,
                     executables: Dict[str, str],
                     input_file: str,
                     tp_size: int = 1) -> List[str]:
    common = [
        f"--inputFile={input_file}",
        f"--outputFile={config.get_output_json_file()}",
        "--dumpProfile",
    ]
    if config.model_type == ModelType.TTS:
        return [
            executables["qwen3_tts_inference"],
            f"--talkerEngineDir={os.path.join(engine_dir, 'talker')}",
            f"--code2wavEngineDir={os.path.join(engine_dir, 'code2wav')}",
            f"--tokenizerDir={os.path.join(engine_dir, 'talker')}",
            f"--checkpointDir={model_dir}",
            f"--outputAudioDir={config.get_output_audio_dir()}",
            *common,
        ]
    if config.model_type == ModelType.VLA:
        return [
            executables["action_inference"],
            f"--engineDir={engine_dir}",
            f"--multimodalEngineDir={engine_dir}",
            f"--checkpointDir={model_dir}",
            *common,
        ]

    command = [
        executables["llm_inference"],
        f"--engineDir={engine_dir}",
        f"--checkpointDir={model_dir}",
        *common,
    ]
    if config.model_type in (ModelType.VLM, ModelType.ASR, ModelType.OMNI):
        command.append(f"--multimodalEngineDir={engine_dir}")
    if (config.is_eagle or config.is_mtp or config.is_dflash
            or config.is_jetspec or config.is_domino or config.is_dspark):
        command.extend([
            "--specDecode",
            f"--specDraftTopK={config.eagle_draft_top_k}",
            f"--specVerifyTokens={config.spec_verify_tokens}",
        ])
        if not (config.is_dflash or config.is_jetspec or config.is_dspark):
            command.append(f"--specDraftStep={config.eagle_draft_step}")
        if config.is_dspark:
            command.append(
                f"--dsparkMaxProposalLen={config.max_spec_draft_tokens}")
        if config.is_mtp:
            draft_checkpoint = (config.get_gemma4_mtp_assistant_model_dir() if
                                config.model_name.lower().startswith("gemma-")
                                else None)
        elif config.is_dflash:
            draft_checkpoint = config.get_dflash_draft_model_dir()
        elif config.is_jetspec:
            draft_checkpoint = config.get_jetspec_draft_model_dir()
        elif config.is_domino:
            draft_checkpoint = config.get_domino_draft_model_dir()
        elif config.is_dspark:
            draft_checkpoint = config.get_dspark_draft_model_dir()
        elif config.is_eagle:
            draft_checkpoint = config.get_eagle_draft_checkpoint_dir()
        else:
            draft_checkpoint = None
        if draft_checkpoint:
            command.append(f"--draftCheckpointDir={draft_checkpoint}")
    talker_dir = os.path.join(engine_dir, "talker")
    if (config.model_type == ModelType.OMNI
            and os.path.isfile(os.path.join(talker_dir, "llm.engine"))):
        command.extend([
            "--enableAudioOutput",
            f"--talkerEngineDir={talker_dir}",
            f"--code2wavEngineDir={os.path.join(engine_dir, 'code2wav')}",
            f"--outputAudioDir={config.get_output_audio_dir()}",
        ])
    if tp_size > 1:
        command.append(f"--tpSize={tp_size}")
    if config.ple_residency is not None:
        command.append(f"--pleResidency={config.ple_residency}")
    if config.ple_cache_slots is not None:
        command.append(f"--pleCacheSlots={config.ple_cache_slots}")
    if config.batch_size is not None:
        command.append(f"--batchSize={config.batch_size}")
    if config.output_seq_len is not None:
        command.append(f"--maxGenerateLength={config.output_seq_len}")
    if config.warmup:
        command.append(f"--warmup={config.warmup}")
    return command


def _assert_component_engines(model_dir: str,
                              engine_dir: str,
                              config: TestConfig,
                              tp_size: int = 1) -> None:
    from tensorrt_edgellm.builder.core import contracts
    from tensorrt_edgellm.builder.core.config import BundleConfig

    bundle = BundleConfig.from_pretrained(model_dir)
    speculative = bool(config.is_eagle or config.is_mtp or config.is_dflash
                       or config.is_jetspec or config.is_domino
                       or config.is_dspark)
    expected = []
    if tp_size > 1:
        # A tensor-parallel build emits the LLM component alone, one per rank.
        spec = contracts.component_spec(contracts.Component.LLM)
        expected = [
            spec.output_path(engine_dir, tp_size=tp_size, tp_rank=rank)
            for rank in range(tp_size)
        ]
    else:
        for component in bundle.components:
            spec = contracts.component_spec(component)
            if component == contracts.Component.LLM and speculative:
                expected.extend((
                    spec.output_path(engine_dir, contracts.SpecRole.BASE),
                    spec.output_path(engine_dir, contracts.SpecRole.DRAFT),
                ))
            else:
                expected.append(spec.output_path(engine_dir))

    missing = [path for path in expected if not os.path.isfile(path)]
    empty = [
        path for path in expected
        if os.path.isfile(path) and os.path.getsize(path) == 0
    ]
    if missing or empty:
        pytest.fail(f"component engine contract failed; missing={missing}, "
                    f"empty={empty}")


def _has_talker(engine_dir: str) -> bool:
    return os.path.isfile(os.path.join(engine_dir, "talker", "llm.engine"))


def _assert_multimodal_prompt_contract(engine_dir: str,
                                       responses: List[dict]) -> None:
    template_paths = (os.path.join(engine_dir, "chat_template.jinja"),
                      os.path.join(engine_dir, "chat_template.model"))
    placeholders = {
        "image": ("<|image_pad|>", "<image>", "<|image_1|>"),
        "audio": ("<|audio_pad|>", "<so_embedding>", "<|audio_1|>"),
        "video": ("<|video_pad|>", "<video>"),
    }
    media_types = {"image", "audio", "video"}
    for response in responses:
        present_types = {
            item.get("type")
            for message in response.get("messages", [])
            for item in message.get("content", [])
            if isinstance(item, dict) and item.get("type") in media_types
        }
        if not present_types:
            continue
        if not any(os.path.isfile(path) for path in template_paths):
            pytest.fail("multimodal runtime has no provider chat template")
        formatted = response.get("formatted_complete_request", "")
        for media_type in present_types:
            expected = placeholders[media_type]
            if not any(placeholder in formatted for placeholder in expected):
                pytest.fail(f"formatted request dropped the {media_type} "
                            f"placeholder; expected one of {expected!r}")


def _assert_runtime_contract(config: TestConfig, engine_dir: str,
                             output: dict) -> None:
    responses = output.get("responses") or []
    if not responses:
        pytest.fail("runtime produced no responses")
    _assert_multimodal_prompt_contract(engine_dir, responses)
    if config.model_type == ModelType.VLA:
        if not all(
                response.get("output_trajectory") for response in responses):
            pytest.fail("VLA runtime produced no action trajectory")
    if config.model_type == ModelType.TTS or _has_talker(engine_dir):
        audio_dir = config.get_output_audio_dir()
        audio_files = [] if not os.path.isdir(audio_dir) else [
            name for name in os.listdir(audio_dir) if name.endswith(".wav")
        ]
        if not audio_files:
            pytest.fail("TTS runtime produced no audio file")


def _run(command: List[str], name: str, timeout: int, env_config,
         test_logger) -> Dict[str, Any]:
    test_logger.info("Starting %s", name)
    result = run_with_trt_env(command, None, timeout, test_logger, env_config)
    if not result["success"]:
        pytest.fail(
            f"{name} failed: {result.get('error') or result['output']}")
    return result


def _check_multistage_parity(config: TestConfig, engine_dir: str,
                             executables: Dict[str, str],
                             runtime_command: List[str], output: dict,
                             env_config: EnvironmentConfig,
                             test_logger) -> None:
    """multistage_inference on the same engines and input must reproduce *output*.

    A spoken case runs it again with its default vocoder windows, which must
    reproduce its whole-utterance run.
    """
    stem = os.path.join(config.test_log_dir, f"{config.param_str}-multistage")
    speech = config.model_type == ModelType.TTS
    # The existing run speaks an Omni reply exactly when the build has a talker.
    omni_speech = config.model_type == ModelType.OMNI and _has_talker(
        engine_dir)
    speaks = speech or omni_speech

    def run(run_stem: str, whole_utterance: bool, name: str) -> dict:
        if speaks:
            # A stale dump would stand in for one this run failed to write.
            shutil.rmtree(run_stem, ignore_errors=True)
        _run(
            multistage_parity.command(executables["multistage_inference"],
                                      runtime_command,
                                      engine_dir,
                                      f"{run_stem}.json",
                                      run_stem if speaks else None,
                                      whole_utterance=whole_utterance), name,
            6000, env_config, test_logger)
        with open(f"{run_stem}.json", encoding="utf-8") as stream:
            return json.load(stream)

    candidate = run(stem, speaks, "multistage_inference parity run")
    reference_name = os.path.basename(runtime_command[0])
    if speech:
        mismatches = multistage_parity.speech_mismatches(
            output, config.get_output_audio_dir(), candidate, stem,
            reference_name)
    elif omni_speech:
        mismatches = multistage_parity.omni_mismatches(
            output, config.get_output_audio_dir(), candidate, stem,
            reference_name)
    else:
        mismatches = multistage_parity.text_mismatches(output, candidate,
                                                       reference_name)
    outputs = [config.get_output_json_file(), f"{stem}.json"]
    if speaks:
        windowed_stem = f"{stem}-windowed"
        windowed = run(windowed_stem, False,
                       "multistage_inference windowed vocoding run")
        mismatches += multistage_parity.windowed_mismatches(
            candidate, stem, windowed, windowed_stem)
        outputs.append(f"{windowed_stem}.json")
    if mismatches:
        pytest.fail(f"multistage_inference differs on the same engines "
                    f"(outputs: {', '.join(outputs)}):\n" +
                    "\n".join(mismatches))
    test_logger.info("multistage_inference matches %s on %d requests",
                     reference_name, len(output["responses"]))


def test_build_and_run(test_param: str, executable_files: Dict[str, str],
                       test_logger, env_config: EnvironmentConfig,
                       request: pytest.FixtureRequest) -> None:
    """Build every checkpoint component once, then execute its runtime once."""
    # Given a checkpoint case, when direct build completes, then runtime satisfies its declared checks.
    _build_and_run(
        test_param,
        executable_files,
        test_logger,
        env_config,
        weightless_engine=request.config.getoption("--weightless-engine"))


def test_build_and_run_weightless(test_param: str, executable_files: Dict[str,
                                                                          str],
                                  test_logger,
                                  env_config: EnvironmentConfig) -> None:
    """Run refit coverage only with an explicitly qualified native runtime."""
    # Given a checkpoint case, when built weightless, then refitted runtime satisfies the same declared checks.
    if os.environ.get("ENABLE_EXPERIMENTAL_WEIGHTLESS") != "ON":
        pytest.skip("Requires ENABLE_EXPERIMENTAL_WEIGHTLESS=ON and a "
                    "qualified TensorRT/Myelin package")
    _build_and_run(test_param,
                   executable_files,
                   test_logger,
                   env_config,
                   weightless_engine=True)


def _build_and_run(test_param: str, executable_files: Dict[str, str],
                   test_logger, env_config: EnvironmentConfig, *,
                   weightless_engine: bool) -> None:
    """Run either weight transport without changing its baseline metric identity."""
    model_type = infer_checkpoint_export_model_type(test_param)
    config = TestConfig.from_param_string(test_param, model_type,
                                          TaskType.CHECKPOINT_BUILD,
                                          env_config)
    if weightless_engine:
        config.test_log_dir = os.path.join(config.test_log_dir, "weightless")
    config.check_trt_native_attn()
    tp_size = config.tp_size or 1
    if tp_size > 1 and (config.is_eagle or config.is_mtp or config.is_dflash
                        or config.is_jetspec or config.is_domino
                        or config.is_dspark):
        pytest.fail(
            "Tensor-parallel direct builds do not support speculative decoding."
        )
    model_dir = config.get_torch_model_dir()
    engine_dir = _engine_dir(config, weightless_engine)
    shutil.rmtree(engine_dir, ignore_errors=True)
    if config.model_type in (ModelType.TTS, ModelType.OMNI):
        shutil.rmtree(config.get_output_audio_dir(), ignore_errors=True)
    os.makedirs(os.path.dirname(engine_dir), exist_ok=True)
    output_file = config.get_output_json_file()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with timer_context(f"direct build and runtime for {config.model_name}",
                       test_logger):
        for tp_rank in range(tp_size):
            label = ("single all-component engine build" if tp_size == 1 else
                     f"engine build for rank {tp_rank}/{tp_size}")
            _run(
                _build_command(config, model_dir, engine_dir, env_config,
                               tp_size, tp_rank, weightless_engine), label,
                7200, env_config, test_logger)
        _assert_component_engines(model_dir, engine_dir, config, tp_size)

        input_file = prepare_spec_decode_input(config, test_logger)
        runtime_command = _runtime_command(config, model_dir, engine_dir,
                                           executable_files, input_file,
                                           tp_size)
        runtime_result = _run(runtime_command,
                              "single end-to-end runtime execution", 6000,
                              env_config, test_logger)

    with open(output_file, encoding="utf-8") as stream:
        output = json.load(stream)
    _assert_runtime_contract(config, engine_dir, output)
    if config.multistage_parity:
        _check_multistage_parity(config, engine_dir, executable_files,
                                 runtime_command, output, env_config,
                                 test_logger)

    reference = config.get_reference_json_file() or config.get_test_case_file()
    accuracy = check_accuracy_with_dataset(output_file, reference,
                                           config.test_case, test_logger)
    runtime_result.update(accuracy)
    if config.test_case in _GUIDED_DECODING_TEST_CASES:
        _check_guided_decoding_conformance(
            config,
            test_logger,
            require_completion=config.test_case ==
            command_execution._JUMP_FORWARD_TEST_CASE)
    if config.test_case == command_execution._JUMP_FORWARD_TEST_CASE:
        command_execution._check_jump_forward_ran(
            runtime_result.get('output', ''), test_logger)
    if config.metric_checks:
        check_declared_metrics(config, runtime_result, "test_build_and_run",
                               test_logger)
    elif (config.is_eagle or config.is_mtp or config.is_dflash
          or config.is_jetspec or config.is_domino or config.is_dspark
          ) and config.test_case not in _GUIDED_DECODING_TEST_CASES:
        # A guided fixture without metric checks is guarded by conformance, not by a baseline.
        _check_baseline_regression(config,
                                   "test_build_and_run",
                                   runtime_result,
                                   test_logger,
                                   check_perf=True)
    check_result_failures(runtime_result)


def _run_on_target(command, remote, env, logger, timeout=6000):
    result = run_with_trt_env(command, remote, timeout, logger, env)
    assert result["success"], result.get("output", result.get("error", ""))
    return result


def _engine_dir_on_target(config: TestConfig, env: EnvironmentConfig,
                          remote: Optional[RemoteConfig]) -> str:
    """Directory the direct builder writes a case's engine bundle to.

    It is under the workspace of the target the build and the runtime run on.
    """
    target_root = remote.remote_workspace if remote else env.llm_sdk_dir
    config.engine_dir = os.path.join(target_root, config.engine_dir)
    return _engine_dir(config)


def _checkpoint_case(test_param, env, remote, logger, task_type):
    config = TestConfig.from_param_string(
        test_param,
        infer_checkpoint_export_model_type(test_param),
        task_type,
        env,
        validate_environment=False)
    # Build entries carry no dataset, so they cannot use CHECKPOINT_BUILD,
    # which is the only task type that sets the checkpoint root.
    config.llm_models_dir = env.llm_models_dir
    env.validate_for_checkpoint_builder_tests()
    engine = _engine_dir_on_target(config, env, remote)
    config.test_log_dir = os.path.abspath(config.test_log_dir)
    assert (config.tp_size
            or 1) == 1, "This checkpoint list path is single-device"
    model, _, _ = test_server_pipeline.TestHLAPI._checkpoint_model_and_spec(
        config, remote, logger)
    return config, model, engine


def test_build(test_param, remote_config, test_logger, env_config):
    """Given a checkpoint profile, build all components on the selected device."""
    config, model, engine = _checkpoint_case(test_param, env_config,
                                             remote_config, test_logger,
                                             TaskType.BUILD)
    command = _build_command(config, model, engine, env_config)
    command[0] = test_server_pipeline.TestHLAPI._hlapi_python(
        env_config, remote_config)
    _run_on_target(command,
                   remote_config,
                   env_config,
                   test_logger,
                   timeout=7200)


def test_dataset(test_param, executable_files, remote_config, test_logger,
                 env_config):
    """Given a built checkpoint, run the whole dataset natively and score it."""
    config, model, engine = _checkpoint_case(test_param, env_config,
                                             remote_config, test_logger,
                                             TaskType.INFERENCE)
    # Without bs, llm_inference uses the batch size from the input file.
    assert (config.batch_size or 1) <= config.max_batch_size
    source = config.get_test_case_file()
    with open(source, encoding="utf-8") as stream:
        requests = json.load(stream)["requests"]
    output = config.get_output_json_file()
    # The remote fetch helper otherwise accepts an existing controller file.
    pathlib.Path(output).unlink(missing_ok=True)
    _run_on_target(["mkdir", "-p", os.path.dirname(output)], remote_config,
                   env_config, test_logger, 30)
    _run_on_target(["rm", "-f", output], remote_config, env_config,
                   test_logger, 30)
    target_root = remote_config.remote_workspace if remote_config else env_config.llm_sdk_dir
    executables = {
        key: os.path.join(target_root, path)
        for key, path in executable_files.items()
    }
    command = _runtime_command(config, model, engine, executables, source)
    # Shared datasets contain paths relative to updated_datasets; basic
    # inputs contain paths relative to the SDK working directory.
    if config.test_case not in {"llm_basic", "vlm_basic"}:
        base = os.path.join(config.edgellm_data_dir, "updated_datasets")
        command = [
            "bash", "-c", 'cd "$1" && shift && exec "$@"', "dataset", base,
            *command
        ]
    result = _run_on_target(command, remote_config, env_config, test_logger)
    command_execution._sync_remote_output_file(output, remote_config,
                                               test_logger)
    with open(output, encoding="utf-8") as stream:
        responses = json.load(stream).get("responses", [])
    # Scoring is positional, so a short output would misalign answers.
    assert len(responses) == len(requests), "Incomplete dataset output"
    assert all(str(r.get("output_text", "")).strip()
               for r in responses), "Empty response"
    result.update(
        check_accuracy_with_dataset(output, source, config.test_case,
                                    test_logger))
    if config.test_case in {"coco", "mtbench"}:
        assert result.get("metric_type") == "skipped"
        test_logger.info(
            f"Generation coverage only; {config.test_case} has no automatic judge"
        )
        return
    assert result.get("metric_type") in {"rouge",
                                         "accuracy"}, "Expected dataset metric"
    # A found baseline clears threshold_failure, so check the floor first.
    check_result_failures(result)
    _check_baseline_regression(config, "test_dataset", result, test_logger)
    check_result_failures(result)
