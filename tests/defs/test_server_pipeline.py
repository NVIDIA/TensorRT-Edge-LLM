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
"""
End-to-end tests for the Python server (nanobind runtime).

Tests the full pipeline: build nanobind extension -> load TRT engine via
Python API -> run inference -> validate output.
"""
import json
import logging
import os
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import run_command, timer_context

from .config import (ModelType, TaskType, TestConfig,
                     infer_checkpoint_export_model_type)
from .utils.command_execution import _sync_remote_output_file

# Server-deps venv provisioned by test_build_nanobind, relative to the repo root
# (workspace root remotely, LLM_SDK_DIR locally).
_SERVER_VENV = 'venv/server'

_OPENAI_REQUEST_FIELDS = {
    "add_generation_prompt",
    "apply_chat_template",
    "audio",
    "cache_generated_tokens",
    "chat_template_kwargs",
    "disable_spec_decode",
    "enable_thinking",
    "frequency_penalty",
    "guided_decoding",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "messages",
    "modalities",
    "model",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "reuse_context",
    "seed",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_k",
    "top_logprobs",
    "top_p",
}


def _openai_messages(messages: Any, location: str) -> List[Dict[str, Any]]:
    if (not isinstance(messages, list) or not messages
            or any(not isinstance(message, Mapping) for message in messages)):
        raise ValueError(f"{location}.messages must be a non-empty list")

    converted = deepcopy(messages)
    for message in converted:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, item in enumerate(content):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"{location}.messages content must be a mapping")
            media_type = item.get("type")
            if media_type not in {"image", "video", "audio"}:
                continue
            source = item.get(media_type)
            if not isinstance(source, str) or not source:
                raise ValueError(
                    f"{location}.messages {media_type} source must be a string"
                )
            openai_item = dict(item)
            openai_item.pop(media_type)
            openai_item["type"] = f"{media_type}_url"
            openai_item[f"{media_type}_url"] = {"url": source}
            content[index] = openai_item
    return converted


def _convert_openai_request(document: Mapping[str, Any], request: Any,
                            location: str) -> Dict[str, Any]:
    if not isinstance(request, Mapping):
        raise ValueError(f"{location} must be a mapping")

    payload = {
        name: deepcopy(document[name])
        for name in _OPENAI_REQUEST_FIELDS if name in document
    }
    payload.update({
        name: deepcopy(request[name])
        for name in _OPENAI_REQUEST_FIELDS if name in request
    })
    payload["messages"] = _openai_messages(request.get("messages"), location)

    max_tokens = request.get("max_generate_length",
                             document.get("max_generate_length"))
    if max_tokens is not None and "max_tokens" not in payload:
        payload["max_tokens"] = max_tokens

    num_logprobs = request.get("num_logprobs", document.get("num_logprobs"))
    if num_logprobs is not None:
        if (isinstance(num_logprobs, bool)
                or not isinstance(num_logprobs, int) or num_logprobs < 0):
            raise ValueError(f"{location}.num_logprobs must be non-negative")
        payload["logprobs"] = num_logprobs > 0
        if num_logprobs:
            payload["top_logprobs"] = min(num_logprobs, 50)

    lookup_policy = request.get(
        "context_cache_lookup_policy",
        document.get("context_cache_lookup_policy", "use_cache"))
    if lookup_policy not in {"use_cache", "bypass"}:
        raise ValueError(f"{location}.context_cache_lookup_policy is invalid")
    payload["reuse_context"] = lookup_policy == "use_cache"

    commit_policy = request.get(
        "context_cache_commit_policy",
        document.get("context_cache_commit_policy",
                     "including_generated_tokens"))
    if commit_policy not in {
            "including_generated_tokens", "prefill_state_only"
    }:
        raise ValueError(f"{location}.context_cache_commit_policy is invalid")
    payload["cache_generated_tokens"] = \
        commit_policy == "including_generated_tokens"
    return payload


def _openai_requests_from_inputs(inferences: Sequence[Mapping[str, Any]],
                                 repo_root: str) -> List[Dict[str, Any]]:
    payloads = []
    for inference in inferences:
        path = Path(inference["input_file"])
        if not path.is_absolute():
            path = Path(repo_root) / path
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read server input file {path}: {error}") \
                from error
        if not isinstance(document, Mapping):
            raise ValueError(
                f"server input file {path} must contain an object")
        requests = document.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError(
                f"server input file {path} must contain non-empty requests")
        payloads.extend(
            _convert_openai_request(document, request,
                                    f"{path}.requests[{index}]")
            for index, request in enumerate(requests))
    return payloads


def _ifb_mixed_step_failures(result: Mapping[str, Any]) -> List[str]:
    """Reasons a run_ifb_trace.py run-mode result did not execute mixed steps."""
    runs = result.get("runs") if result.get("mode") == "run" else None
    if not runs:
        return ["the trace result holds no run-mode runs"]
    failures = []
    for run in runs:
        label = run.get("label", "?")
        resolved = (run.get("policy") or {}).get("resolved")
        if resolved != "mixed":
            failures.append(
                f"{label}: resolved policy {resolved!r} != 'mixed'")
        delta = run.get("metrics_delta") or {}
        for metric in ("mixed_steps", "partial_prompt_chunks"):
            value = delta.get(metric)
            if isinstance(value,
                          bool) or not isinstance(value, int) or value <= 0:
                failures.append(f"{label}: {metric}={value!r}, expected > 0")
    return failures


def test_build_nanobind(env_config: EnvironmentConfig,
                        remote_config: Optional[RemoteConfig],
                        test_logger: logging.Logger):
    """Build the nanobind extension by reconfiguring the existing build directory.

    Uses subdirectory mode (BUILD_PYTHON_BINDINGS=ON) so all cmake
    variables (CUDA, TRT, toolchain) are inherited from the prior
    test_build_project configuration.
    """
    build_dir = env_config.build_dir

    repo_root = '.' if remote_config else env_config.llm_sdk_dir
    nanobind_venv = _SERVER_VENV
    install_cmd = (f'cd {shlex.quote(repo_root)}'
                   f' && python3 -m venv {nanobind_venv}'
                   f' && {nanobind_venv}/bin/pip install -q'
                   ' -e ".[builder,runtime]"')
    if env_config.trt_package_dir:
        trt_python_dir = shlex.quote(
            os.path.join(env_config.trt_package_dir, "python"))
        install_cmd += (
            f' && PY_TAG=$({nanobind_venv}/bin/python -c '
            "'import sys; print(\"cp%d%d\" % sys.version_info[:2])')"
            f' && TRT_WHL=$(find {trt_python_dir} -maxdepth 1 -type f '
            "-name \"tensorrt-*${PY_TAG}*.whl\" -print -quit)"
            ' && test -n "$TRT_WHL"'
            f' && {nanobind_venv}/bin/pip install -q "$TRT_WHL"')
    # A fresh venv installs the builder and runtime extras from the package
    # index, which can take minutes.
    result = run_command(cmd=['bash', '-c', install_cmd],
                         remote_config=remote_config,
                         timeout=int(
                             os.environ.get('CI_PIP_INSTALL_TIMEOUT_SECONDS',
                                            '1200')),
                         logger=test_logger)
    if not result['success']:
        pytest.fail(
            f"Failed to install server dependencies: {result.get('error')}")

    nanobind_python = f'{repo_root}/{nanobind_venv}/bin/python'
    nanobind_dir_expr = (f'$({shlex.quote(nanobind_python)}'
                         ' -c "import nanobind; print(nanobind.cmake_dir())")')
    build_cmd = (f'NANOBIND_DIR={nanobind_dir_expr}'
                 f' && cd {build_dir}'
                 f' && cmake .. -DBUILD_PYTHON_BINDINGS=ON'
                 f' -Dnanobind_DIR=$NANOBIND_DIR'
                 f' && make -j$(nproc) _edgellm_runtime')

    with timer_context("Building nanobind extension", test_logger):
        result = run_command(cmd=['bash', '-c', build_cmd],
                             remote_config=remote_config,
                             timeout=600,
                             logger=test_logger)

    if not result['success']:
        pytest.fail(f"Nanobind build failed: {result.get('error')}")

    nanobind_output_dir = f'{build_dir}/lib'
    result = run_command(
        cmd=['bash', '-c', f'ls {nanobind_output_dir}/*_edgellm_runtime*.so'],
        remote_config=remote_config,
        timeout=10,
        logger=test_logger)
    if not result['success']:
        pytest.fail(
            f"_edgellm_runtime.so not found in {nanobind_output_dir} after build"
        )


class TestServerPipeline:
    """E2E tests for inference via the Python server API."""

    def test_server_inference(self, test_param: str,
                              executable_files: Dict[str, str],
                              remote_config: Optional[RemoteConfig],
                              test_logger: logging.Logger,
                              env_config: EnvironmentConfig) -> None:
        """Test inference using the nanobind Python runtime directly."""
        is_vlm = "-mnit" in test_param
        model_type = ModelType.VLM if is_vlm else ModelType.LLM
        config = TestConfig.from_param_string(test_param, model_type,
                                              TaskType.INFERENCE, env_config)

        engine_dir = config.get_llm_engine_dir()
        multimodal_engine_dir = config.get_visual_engine_dir(
        ) if is_vlm else ""
        test_logger.info("Using engine dir: %s", engine_dir)
        if multimodal_engine_dir:
            test_logger.info("Using visual engine dir: %s",
                             multimodal_engine_dir)

        nanobind_build_dir = os.path.join(env_config.build_dir, "lib")
        prompt = "Please introduce the company NVIDIA and its CEO."
        max_tokens = 128

        script = f"""\
import sys, os
sys.path.insert(0, {nanobind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({nanobind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
if not so_files:
    raise RuntimeError('_edgellm_runtime.so not found in ' + {nanobind_build_dir!r})
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({nanobind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

engine_dir = {engine_dir!r}
multimodal_engine_dir = {multimodal_engine_dir!r}
runtime = rt.LLMRuntime(engine_dir, multimodal_engine_dir, {{}})
runtime.capture_decoding_cuda_graph()

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
request.requests = [req]
request.temperature = 0.7
request.top_p = 0.9
request.top_k = 50
request.max_generate_length = {max_tokens}
request.apply_chat_template = True
request.add_generation_prompt = True
request.enable_thinking = False
request.disable_spec_decode = False

response = runtime.handle_request(request)
text = response.output_texts[0] if response.output_texts else ''
ids = response.output_ids[0] if response.output_ids else []
print(f'OUTPUT_TEXT_LEN={{len(text)}}')
print(f'OUTPUT_IDS_LEN={{len(ids)}}')
print(f'OUTPUT_TEXT={{text[:200]}}')
assert len(text) > 0, 'Empty output text'
assert len(ids) > 0, 'Empty output token ids'
print('SERVER_INFERENCE_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"Server inference for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Server inference failed: {result.get('error', 'Unknown')}")

        output = result.get('output', '')
        if 'SERVER_INFERENCE_PASSED' not in output:
            pytest.fail(
                f"Server inference did not produce expected output. Output:\n{output}"
            )

    def test_server_inference_with_audio(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Audio nanobind path: in-memory PCM -> AudioData -> Request.audio_buffers -> runtime."""
        is_asr = "-asr" in test_param
        is_omni = "-omni" in test_param
        if not (is_asr or is_omni):
            pytest.skip(
                "audio test requires test_param with '-asr' or '-omni'")
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        audio_engine_dir = (getattr(config, "get_audio_engine_dir",
                                    lambda: "")()
                            or os.environ.get("AUDIO_ENCODER_ENGINE_DIR", ""))
        if not audio_engine_dir:
            pytest.skip("AUDIO_ENCODER_ENGINE_DIR not set")
        test_wav = (getattr(config, "get_audio_test_wav", lambda: "")()
                    or os.environ.get("AUDIO_TEST_WAV", ""))
        if not test_wav:
            pytest.skip("AUDIO_TEST_WAV not set")

        nanobind_build_dir = os.path.join(env_config.build_dir, "lib")
        script = f"""\
import sys, os, importlib.util
sys.path.insert(0, {nanobind_build_dir!r})
so_files = [f for f in os.listdir({nanobind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({nanobind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
with open({test_wav!r}, 'rb') as _f: _audio_bytes = _f.read()
runtime = rt.LLMRuntime({engine_dir!r}, {audio_engine_dir!r}, {{}})
runtime.capture_decoding_cuda_graph()
request = rt.LLMGenerationRequest()
msg = rt.Message(); msg.role = 'user'; msg.contents = [rt.MessageContent('audio', '')]
req = rt.Request(messages=[msg]); req.image_buffers = []
req.audio_buffers = [rt.load_audio_buffer_from_bytes(_audio_bytes)]
request.requests = [req]; request.temperature = 1.0; request.top_p = 1.0; request.top_k = 50
request.max_generate_length = 128; request.apply_chat_template = True; request.add_generation_prompt = True
response = runtime.handle_request(request)
ids = response.output_ids[0] if response.output_ids else []
assert len(ids) > 0
print('SERVER_INFERENCE_WITH_AUDIO_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']
        env_vars = {
            "LD_LIBRARY_PATH":
            f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
        } if env_config.trt_package_dir else None
        with timer_context(f"Server audio inference for {config.model_name}",
                           test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        if not result[
                'success'] or 'SERVER_INFERENCE_WITH_AUDIO_PASSED' not in result.get(
                    'output', ''):
            pytest.fail(f"audio inference failed:\n{result.get('output', '')}")

    def test_server_streaming(self, test_param: str,
                              executable_files: Dict[str, str],
                              remote_config: Optional[RemoteConfig],
                              test_logger: logging.Logger,
                              env_config: EnvironmentConfig) -> None:
        """Test streaming inference using StreamChannel via nanobind."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)

        engine_dir = config.get_llm_engine_dir()
        test_logger.info("Using engine dir: %s", engine_dir)

        nanobind_build_dir = os.path.join(env_config.build_dir, "lib")
        prompt = "Count from 1 to 10."
        max_tokens = 128

        script = f"""\
import sys, os, threading
sys.path.insert(0, {nanobind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({nanobind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
if not so_files:
    raise RuntimeError('_edgellm_runtime.so not found in ' + {nanobind_build_dir!r})
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({nanobind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

engine_dir = {engine_dir!r}
runtime = rt.LLMRuntime(engine_dir, '', {{}})
runtime.capture_decoding_cuda_graph()

channel = rt.StreamChannel.create()
channel.set_skip_special_tokens(True)

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
request.requests = [req]
request.stream_channels = [channel]
request.temperature = 0.7
request.top_p = 0.9
request.top_k = 50
request.max_generate_length = {max_tokens}
request.apply_chat_template = True
request.add_generation_prompt = True
request.enable_thinking = False
request.disable_spec_decode = False

def run_inference():
    runtime.handle_request(request)

worker = threading.Thread(target=run_inference, daemon=True)
worker.start()

chunks = []
while True:
    chunk = channel.wait_pop(timeout_ms=500)
    if chunk is None:
        if channel.is_finished() or channel.is_cancelled():
            break
        continue
    chunks.append(chunk)
    if chunk.finished:
        break

worker.join(timeout=10)

total_text = ''.join(c.text for c in chunks)
total_ids = sum(len(c.token_ids) for c in chunks)
print(f'STREAM_CHUNKS={{len(chunks)}}')
print(f'STREAM_TEXT_LEN={{len(total_text)}}')
print(f'STREAM_IDS={{total_ids}}')
print(f'STREAM_TEXT={{total_text[:200]}}')
assert len(chunks) > 1, f'Expected multiple chunks, got {{len(chunks)}}'
assert len(total_text) > 0, 'Empty streamed text'
assert any(c.finished for c in chunks), 'No terminal chunk received'
print('SERVER_STREAMING_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"Server streaming for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Server streaming failed: {result.get('error', 'Unknown')}")

        output = result.get('output', '')
        if 'SERVER_STREAMING_PASSED' not in output:
            pytest.fail(
                f"Server streaming did not produce expected output. Output:\n{output}"
            )

    def test_server_streaming_with_guided_decoding(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Streaming and guided decoding must hold together.

        The mask is applied before sampling and chunks are emitted after, so nothing
        unconstrained can reach the client -- but the two features share the per-slot
        vectors that `performBatchEvict` compacts, and the error path writes a stream
        terminal reason. This pins that: the text assembled from the deltas satisfies the
        schema, and it really arrived as deltas rather than one final blob.
        """
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)

        engine_dir = config.get_llm_engine_dir()
        test_logger.info("Streaming with guided decoding: engine=%s",
                         engine_dir)

        nanobind_build_dir = os.path.join(env_config.build_dir, "lib")
        prompt = "Give me a person record with a name and an age."
        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string"
                    },
                    "age": {
                        "type": "integer"
                    },
                },
                "required": ["name", "age"],
            },
            sort_keys=True)

        script = f"""\
import sys, os, json, threading
sys.path.insert(0, {nanobind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({nanobind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
if not so_files:
    raise RuntimeError('_edgellm_runtime.so not found in ' + {nanobind_build_dir!r})
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({nanobind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

runtime = rt.LLMRuntime({engine_dir!r}, '', {{}})
runtime.capture_decoding_cuda_graph()

channel = rt.StreamChannel.create()
channel.set_skip_special_tokens(True)

msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
req.guided_decoding = rt.GuidedDecodingParams(rt.GuideType.JSON_SCHEMA, {schema!r})

request = rt.LLMGenerationRequest()
request.requests = [req]
request.stream_channels = [channel]
request.temperature = 1.0
request.top_p = 1.0
request.top_k = 1
request.max_generate_length = 128
request.apply_chat_template = True
request.add_generation_prompt = True
request.enable_thinking = False
request.disable_spec_decode = True

worker = threading.Thread(target=lambda: runtime.handle_request(request), daemon=True)
worker.start()

chunks = []
while True:
    chunk = channel.wait_pop(timeout_ms=500)
    if chunk is None:
        if channel.is_finished() or channel.is_cancelled():
            break
        continue
    chunks.append(chunk)
    if chunk.finished:
        break
worker.join(timeout=30)

text = ''.join(c.text for c in chunks)
print(f'GUIDED_STREAM_CHUNKS={{len(chunks)}}')
print(f'GUIDED_STREAM_TEXT={{text!r}}')
print(f'GUIDED_STREAM_REASON={{channel.get_reason()}}')

assert any(c.finished for c in chunks), 'no terminal chunk received'
# More than one chunk is what distinguishes streaming from a single final emit; a
# guided run that silently fell back to non-streaming would still satisfy the schema.
assert len(chunks) > 1, f'expected deltas, got {{len(chunks)}} chunk(s)'

document = json.loads(text)
assert isinstance(document, dict), f'streamed text is not a JSON object: {{text!r}}'
assert isinstance(document.get('name'), str), f'name must be a string: {{document!r}}'
assert isinstance(document.get('age'), int), f'age must be an integer: {{document!r}}'

print('SERVER_STREAMING_WITH_GUIDED_DECODING_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(
                f"Streaming with guided decoding for {config.model_name}",
                test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        output = result.get('output', '')
        if not result['success']:
            pytest.fail("Streaming with guided decoding failed: "
                        f"{result.get('error', 'Unknown')}")
        if 'SERVER_STREAMING_WITH_GUIDED_DECODING_PASSED' not in output:
            pytest.fail(f"Streaming with guided decoding output:\n{output}")

    def test_server_inference_with_logprobs(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Test non-streaming inference returns per-token logprobs via nanobind."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        test_logger.info("Server inference with logprobs: engine=%s",
                         engine_dir)

        nanobind_build_dir = os.path.join(env_config.build_dir, "lib")
        prompt = "Count from 1 to 5."
        max_tokens = 32
        num_logprobs = 3

        script = f"""\
import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, {nanobind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({nanobind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
if not so_files:
    raise RuntimeError('_edgellm_runtime.so not found in ' + {nanobind_build_dir!r})
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({nanobind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
sys.modules['_edgellm_runtime'] = rt  # make the HLAPI reuse this exact module
spec.loader.exec_module(rt)

engine_dir = {engine_dir!r}
runtime = rt.LLMRuntime(engine_dir, '', {{}})
runtime.capture_decoding_cuda_graph()

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
request.requests = [req]
request.temperature = 0.0
request.top_p = 1.0
request.top_k = 1
request.max_generate_length = {max_tokens}
request.apply_chat_template = True
request.add_generation_prompt = True
request.num_logprobs = {num_logprobs}

response = runtime.handle_request(request)
ids = response.output_ids[0] if response.output_ids else []
lps = response.logprobs[0] if response.logprobs else []
print(f'OUTPUT_IDS_LEN={{len(ids)}}')
print(f'LOGPROBS_STEPS={{len(lps)}}')
assert len(ids) > 0, 'Empty output ids'
assert len(lps) == len(ids), f'logprobs steps {{len(lps)}} != token count {{len(ids)}}'
for step in lps:
    assert len(step) == {num_logprobs}, f'Expected {num_logprobs} entries per step, got {{len(step)}}'
    for e in step:
        assert isinstance(e.token_id, int) and e.token_id >= 0, f'Bad token_id {{e.token_id}}'
        assert e.logprob <= 0.0, f'logprob should be <= 0, got {{e.logprob}}'
        assert isinstance(e.piece, bytes), f'piece should be bytes, got {{type(e.piece)}}'

# Also exercise the OpenAI-compatible formatting layer on the same raw response.
from tensorrt_edgellm.server.api.serving_chat import _format_logprob_steps
obj = _format_logprob_steps(ids, lps, True)
assert obj is not None and 'content' in obj, f'missing content: {{obj}}'
oc = obj['content']
assert len(oc) == len(ids), f'content {{len(oc)}} != token count {{len(ids)}}'
for c in oc:
    assert set(c) >= {{'token', 'token_id', 'bytes', 'logprob', 'top_logprobs'}}, f'missing keys: {{sorted(c)}}'
    assert isinstance(c['token'], str) and isinstance(c['token_id'], int) and isinstance(c['bytes'], list)
    assert c['logprob'] is None or c['logprob'] <= 0.0, f'bad chosen logprob {{c["logprob"]}}'
    top = c['top_logprobs']
    assert len(top) == {num_logprobs}, f'expected {num_logprobs} top_logprobs, got {{len(top)}}'
    for t in top:
        assert set(t) >= {{'token', 'token_id', 'bytes', 'logprob'}}, f'missing top keys: {{sorted(t)}}'
        assert isinstance(t['token'], str) and isinstance(t['bytes'], list) and t['logprob'] <= 0.0
print('SERVER_INFERENCE_LOGPROBS_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"Server inference with logprobs for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Server inference with logprobs failed: {result.get('error', 'Unknown')}"
            )

        if 'SERVER_INFERENCE_LOGPROBS_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"Server inference logprobs output:\n{result.get('output', '')}"
            )

    def test_server_streaming_with_logprobs(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Test streaming inference delivers per-token logprobs on each chunk via nanobind."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        test_logger.info("Server streaming with logprobs: engine=%s",
                         engine_dir)

        nanobind_build_dir = os.path.join(env_config.build_dir, "lib")
        prompt = "Count from 1 to 5."
        max_tokens = 32
        num_logprobs = 3

        script = f"""\
import sys, os, threading
sys.path.insert(0, {nanobind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({nanobind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
if not so_files:
    raise RuntimeError('_edgellm_runtime.so not found in ' + {nanobind_build_dir!r})
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({nanobind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

engine_dir = {engine_dir!r}
runtime = rt.LLMRuntime(engine_dir, '', {{}})
runtime.capture_decoding_cuda_graph()

channel = rt.StreamChannel.create()
channel.set_skip_special_tokens(True)

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
request.requests = [req]
request.stream_channels = [channel]
request.temperature = 0.0
request.top_p = 1.0
request.top_k = 1
request.max_generate_length = {max_tokens}
request.apply_chat_template = True
request.add_generation_prompt = True
request.num_logprobs = {num_logprobs}

def run_inference():
    runtime.handle_request(request)

worker = threading.Thread(target=run_inference, daemon=True)
worker.start()

chunks = []
while True:
    chunk = channel.wait_pop(timeout_ms=500)
    if chunk is None:
        if channel.is_finished() or channel.is_cancelled():
            break
        continue
    chunks.append(chunk)
    if chunk.finished:
        break

worker.join(timeout=10)

total_tokens = sum(len(c.token_ids) for c in chunks)
total_lp_steps = sum(len(c.logprobs) for c in chunks)
print(f'STREAM_TOTAL_TOKENS={{total_tokens}}')
print(f'STREAM_TOTAL_LP_STEPS={{total_lp_steps}}')
assert total_tokens > 0, 'No tokens received'
assert total_lp_steps == total_tokens, f'logprob steps {{total_lp_steps}} != token count {{total_tokens}}'
for chunk in chunks:
    for step in chunk.logprobs:
        assert len(step) == {num_logprobs}, f'Expected {num_logprobs} entries per step, got {{len(step)}}'
        for e in step:
            assert isinstance(e.token_id, int) and e.token_id >= 0, f'Bad token_id {{e.token_id}}'
            assert e.logprob <= 0.0, f'logprob should be <= 0, got {{e.logprob}}'
            assert isinstance(e.piece, bytes), f'piece should be bytes, got {{type(e.piece)}}'
print('SERVER_STREAMING_LOGPROBS_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"Server streaming with logprobs for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Server streaming with logprobs failed: {result.get('error', 'Unknown')}"
            )

        if 'SERVER_STREAMING_LOGPROBS_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"Server streaming logprobs output:\n{result.get('output', '')}"
            )

    def test_server_inference_with_stop(self, test_param: str,
                                        executable_files: Dict[str, str],
                                        remote_config: Optional[RemoteConfig],
                                        test_logger: logging.Logger,
                                        env_config: EnvironmentConfig) -> None:
        """Non-streaming nanobind path: Request.stop_strings + response.finish_reasons round-trip."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        nanobind_build_dir = os.path.join(env_config.build_dir, "lib")
        prompt = "List three colors, separated by commas. End your list with '###'."
        stop = "###"

        script = f"""\
import sys, os
sys.path.insert(0, {nanobind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({nanobind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({nanobind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

runtime = rt.LLMRuntime({engine_dir!r}, '', {{}})
runtime.capture_decoding_cuda_graph()

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
req.stop_strings = [{stop!r}]
request.requests = [req]
request.temperature = 0.0
request.top_p = 1.0
request.top_k = 1
request.max_generate_length = 128
request.apply_chat_template = True
request.add_generation_prompt = True

response = runtime.handle_request(request)
text = response.output_texts[0] if response.output_texts else ''
reasons = list(response.finish_reasons) if response.finish_reasons else []
print(f'OUTPUT_TEXT={{text!r}}')
print(f'FINISH_REASON={{reasons[0] if reasons else None}}')
assert {stop!r} not in text, f'Stop string leaked into output: {{text!r}}'
assert len(reasons) == 1, f'Expected 1 finish_reason, got {{len(reasons)}}'
assert reasons[0] == rt.FinishReason.STOP_WORDS, f'Expected STOP_WORDS, got {{reasons[0]}}'
print('SERVER_INFERENCE_WITH_STOP_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(
                f"Server inference with stop for {config.model_name}",
                test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Server inference with stop failed: {result.get('error', 'Unknown')}"
            )
        if 'SERVER_INFERENCE_WITH_STOP_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"Server inference with stop did not produce expected output. Output:\n{result.get('output', '')}"
            )

    def test_server_streaming_with_stop(self, test_param: str,
                                        executable_files: Dict[str, str],
                                        remote_config: Optional[RemoteConfig],
                                        test_logger: logging.Logger,
                                        env_config: EnvironmentConfig) -> None:
        """Streaming nanobind path: stop string trims chunk text, terminal chunk has STOP_WORDS reason."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        nanobind_build_dir = os.path.join(env_config.build_dir, "lib")
        prompt = "List three colors, separated by commas. End your list with '###'."
        stop = "###"

        script = f"""\
import sys, os, threading
sys.path.insert(0, {nanobind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({nanobind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({nanobind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

runtime = rt.LLMRuntime({engine_dir!r}, '', {{}})
runtime.capture_decoding_cuda_graph()

channel = rt.StreamChannel.create()
channel.set_skip_special_tokens(True)

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
req.stop_strings = [{stop!r}]
request.requests = [req]
request.stream_channels = [channel]
request.temperature = 0.0
request.top_p = 1.0
request.top_k = 1
request.max_generate_length = 128
request.apply_chat_template = True
request.add_generation_prompt = True

threading.Thread(target=lambda: runtime.handle_request(request), daemon=True).start()
chunks = []
while True:
    c = channel.wait_pop(timeout_ms=500)
    if c is None:
        if channel.is_finished() or channel.is_cancelled(): break
        continue
    chunks.append(c)
    if c.finished: break

text = ''.join(c.text for c in chunks)
terminal = next((c for c in chunks if c.finished), None)
print(f'STREAM_TEXT={{text!r}}')
print(f'TERMINAL_REASON={{terminal.reason if terminal else None}}')
assert {stop!r} not in text, f'Stop string leaked into streamed text: {{text!r}}'
assert terminal is not None, 'No terminal chunk'
assert terminal.reason == rt.FinishReason.STOP_WORDS, f'Expected STOP_WORDS, got {{terminal.reason}}'
print('SERVER_STREAMING_WITH_STOP_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(
                f"Server streaming with stop for {config.model_name}",
                test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Server streaming with stop failed: {result.get('error', 'Unknown')}"
            )
        if 'SERVER_STREAMING_WITH_STOP_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"Server streaming with stop did not produce expected output. Output:\n{result.get('output', '')}"
            )

    def test_server_inference_length_finish_reason(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Verify response.finish_reasons reports LENGTH when max_generate_length hit (no stops set)."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        nanobind_build_dir = os.path.join(env_config.build_dir, "lib")
        prompt = "Write a long detailed essay about transformer neural networks."

        script = f"""\
import sys, os
sys.path.insert(0, {nanobind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({nanobind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({nanobind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

runtime = rt.LLMRuntime({engine_dir!r}, '', {{}})
runtime.capture_decoding_cuda_graph()

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
request.requests = [req]
request.temperature = 0.0
request.top_p = 1.0
request.top_k = 1
request.max_generate_length = 8  # tiny → should hit LENGTH
request.apply_chat_template = True
request.add_generation_prompt = True

response = runtime.handle_request(request)
reasons = list(response.finish_reasons) if response.finish_reasons else []
print(f'FINISH_REASON={{reasons[0] if reasons else None}}')
assert len(reasons) == 1
assert reasons[0] == rt.FinishReason.LENGTH, f'Expected LENGTH, got {{reasons[0]}}'
print('SERVER_INFERENCE_LENGTH_REASON_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(
                f"Server inference length-reason for {config.model_name}",
                test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Length-reason test failed: {result.get('error', 'Unknown')}")
        if 'SERVER_INFERENCE_LENGTH_REASON_PASSED' not in result.get(
                'output', ''):
            pytest.fail(
                f"Length-reason test did not produce expected output. Output:\n{result.get('output', '')}"
            )

    def test_ifb_trace(self, test_param: str, executable_files: Dict[str, str],
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger,
                       env_config: EnvironmentConfig) -> None:
        """Mixed in-flight-batching steps form on the built engine and match
        the single-cohort reference outputs."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        repo_root = '.' if remote_config else env_config.llm_sdk_dir
        build_lib = os.path.join(env_config.build_dir, 'lib')
        plugin = os.path.join(build_lib, 'libNvInfer_edgellm_plugin.so')
        out_dir = os.path.join(config.test_log_dir,
                               f"{config.param_str}-ifb-trace")
        trace_timeout = 900
        command = [
            'bash', f'{repo_root}/tests/ifb/ifb_test.sh', '--tier', 'quick',
            '--traces', config.test_case, '--mode', 'mixed', '--engine',
            engine_dir, '--out', out_dir, '--python',
            f'{repo_root}/{_SERVER_VENV}/bin/python3', '--nanobind-dir',
            build_lib, '--timeout',
            str(trace_timeout)
        ]
        # A stale run JSON would let the checker pass a run that never wrote one.
        cmd = [
            'bash', '-c', f'rm -rf -- {shlex.quote(out_dir)} && '
            f'{shlex.join(command)}'
        ]
        library_path = f"{build_lib}:$LD_LIBRARY_PATH"
        if env_config.trt_package_dir:
            library_path += f":{env_config.trt_package_dir}/lib"
        env_vars = {
            'LD_LIBRARY_PATH': library_path,
            'EDGELLM_PLUGIN_LIB': plugin,
            'EDGELLM_PLUGIN_PATH': plugin,
        }

        with timer_context(f"IFB trace for {config.model_name}", test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=2 * trace_timeout + 600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        if not result['success']:
            pytest.fail(f"IFB trace check failed:\n{result.get('output', '')}"
                        f"{result.get('error', '')}")

        run_file = os.path.join(out_dir,
                                os.path.basename(os.path.normpath(engine_dir)),
                                f"{config.test_case}.run.mixed.json")
        with open(_sync_remote_output_file(run_file, remote_config,
                                           test_logger),
                  encoding='utf-8') as handle:
            failures = _ifb_mixed_step_failures(json.load(handle))
        if failures:
            pytest.fail("IFB trace did not run mixed steps:\n" +
                        "\n".join(failures))


class TestHLAPI:
    """E2E tests for checkpoint-direct high-level Python inference."""

    @staticmethod
    def _checkpoint_config(test_param: str, model_type: ModelType,
                           env_config: EnvironmentConfig) -> TestConfig:
        """Parse inference options with checkpoint-direct path validation."""
        config = TestConfig.from_param_string(test_param,
                                              model_type,
                                              TaskType.INFERENCE,
                                              env_config,
                                              validate_environment=False)
        if config.checkpoint_dir:
            env_config.validate_for_checkpoint_runtime_tests()
        else:
            env_config.validate_for_checkpoint_builder_tests()
        return config

    @staticmethod
    def _stage_remote_checkpoint(model_dir: str,
                                 remote_config: Optional[RemoteConfig],
                                 test_logger: logging.Logger) -> str:
        """Make a controller-local checkpoint visible to a remote board."""
        if remote_config is None:
            return model_dir

        visible = run_command(
            ["test", "-f",
             os.path.join(model_dir, "config.json")],
            remote_config,
            timeout=15,
            logger=test_logger,
        )
        if visible["success"]:
            return model_dir

        staged_dir = os.path.join(remote_config.remote_workspace,
                                  ".ci-checkpoints",
                                  os.path.basename(model_dir.rstrip("/")))
        created = run_command(["mkdir", "-p", staged_dir],
                              remote_config,
                              timeout=30,
                              logger=test_logger)
        if not created["success"]:
            raise RuntimeError(f"Cannot create remote checkpoint directory: "
                               f"{staged_dir}")

        remote_host = f"{remote_config.user}@{remote_config.host}"
        synced = run_command(
            [
                "sshpass", "-e", "rsync", "-aL", "-e",
                "ssh -o StrictHostKeyChecking=no", f"{model_dir.rstrip('/')}/",
                f"{remote_host}:{staged_dir}/"
            ],
            remote_config=None,
            timeout=1800,
            logger=test_logger,
            env_vars={"SSHPASS": remote_config.password},
        )
        if not synced["success"]:
            raise RuntimeError(f"Cannot stage checkpoint on remote board: "
                               f"{model_dir}")
        return staged_dir

    @staticmethod
    def _build_hlapi_env_setup(trt_package_dir: str = "") -> str:
        """Return inline script preamble that sets up sys.path and LD_LIBRARY_PATH."""
        parts = [
            "import sys, os",
            "sys.path.insert(0, os.getcwd())",
        ]
        if trt_package_dir:
            parts.append("os.environ.setdefault('LD_LIBRARY_PATH', '')")
            parts.append(
                f"os.environ['LD_LIBRARY_PATH'] += ':{trt_package_dir}/lib'")
        return "\n".join(parts)

    @staticmethod
    def _hlapi_python(env_config: EnvironmentConfig,
                      remote_config: Optional[RemoteConfig]) -> str:
        """Interpreter for the injected HLAPI script: the server/build venv
        created by test_build_nanobind under the same repository root."""
        repo_root = '.' if remote_config else env_config.llm_sdk_dir
        return f'{repo_root}/{_SERVER_VENV}/bin/python3'

    @staticmethod
    def _spec_type(config: TestConfig) -> str:
        if config.is_mtp:
            return "mtp"
        if config.is_dflash:
            return "dflash"
        if config.is_jetspec:
            return "jetspec"
        if config.is_domino:
            return "domino"
        if config.is_dspark:
            return "dspark"
        if config.is_eagle:
            return "eagle3"
        return "none"

    @staticmethod
    def _checkpoint_model_and_spec(
            config: TestConfig, remote_config: Optional[RemoteConfig],
            test_logger: logging.Logger) -> Tuple[str, str, str]:
        """Resolve model and optional draft paths for checkpoint-direct APIs."""
        try:
            model_dir = config.get_torch_model_dir()
            draft_model_dir = ""
            spec_type = TestHLAPI._spec_type(config)
            if spec_type == "mtp":
                if config.model_name.lower().startswith("gemma-"):
                    draft_model_dir = \
                        config.get_gemma4_mtp_assistant_model_dir()
            elif spec_type == "dflash":
                draft_model_dir = config.get_dflash_draft_model_dir()
            elif spec_type == "jetspec":
                draft_model_dir = config.get_jetspec_draft_model_dir()
            elif spec_type == "domino":
                draft_model_dir = config.get_domino_draft_model_dir()
            elif spec_type == "dspark":
                draft_model_dir = config.get_dspark_draft_model_dir()
            elif spec_type == "eagle3":
                draft_model_dir = config.get_eagle_draft_checkpoint_dir()

            model_dir = TestHLAPI._stage_remote_checkpoint(
                model_dir, remote_config, test_logger)
            if draft_model_dir:
                draft_model_dir = TestHLAPI._stage_remote_checkpoint(
                    draft_model_dir, remote_config, test_logger)
        except ValueError as exc:
            pytest.skip("checkpoint-direct source is unavailable on "
                        f"this runner: {exc}")
        return model_dir, spec_type, draft_model_dir

    @staticmethod
    def _server_cli_args(config: TestConfig, env_config: EnvironmentConfig,
                         remote_config: Optional[RemoteConfig],
                         test_logger: logging.Logger) -> List[str]:
        """Build the public server command for one model-owned case."""
        model_dir, spec_type, draft_model_dir = \
            TestHLAPI._checkpoint_model_and_spec(config, remote_config,
                                                 test_logger)

        repo_root = '.' if remote_config else env_config.llm_sdk_dir
        max_kv = config.max_kv_cache_capacity or config.max_seq_len
        command = [
            f'{repo_root}/{_SERVER_VENV}/bin/tensorrt-edgellm-serve',
            model_dir,
            '--host',
            '127.0.0.1',
            '--cache-dir',
            os.path.join(config.engine_dir, 'experimental-server-cache'),
            '--max-input-len',
            str(config.max_input_len),
            '--max-batch-size',
            str(config.max_batch_size),
            '--max-kv-cache-capacity',
            str(max_kv),
            '--allowed-local-media-path',
            '.',
            '--max-queued-requests',
            '0',
            '--queue-timeout',
            '5',
            '--log-level',
            'info' if config.context_reuse else 'error',
        ]
        if config.max_image_tokens is not None:
            command.extend(
                ['--max-image-tokens',
                 str(config.max_image_tokens)])
        if config.max_image_tokens_per_image is not None:
            command.extend([
                '--max-image-tokens-per-image',
                str(config.max_image_tokens_per_image)
            ])
        if spec_type != "none":
            speculative = {'method': spec_type}
            if draft_model_dir:
                speculative['model'] = draft_model_dir
            command.extend([
                '--speculative-config',
                json.dumps(speculative, separators=(',', ':')),
                '--draft-top-k',
                str(config.eagle_draft_top_k),
                '--spec-verify-tokens',
                str(config.max_spec_verify_tokens),
            ])
            if spec_type in {"mtp", "eagle3"}:
                command.extend(
                    ['--spec-draft-step',
                     str(config.eagle_draft_step)])
        if config.context_reuse:
            command.append('--enable-context-reuse')
            if config.context_cache_max_snapshots:
                command.extend([
                    '--context-cache-max-snapshots',
                    str(config.context_cache_max_snapshots)
                ])
        if config.in_flight_batching:
            command.append('--enable-in-flight-batching')
        return command

    @staticmethod
    def _llm_init_script(config: TestConfig, env_config: EnvironmentConfig,
                         remote_config: Optional[RemoteConfig],
                         test_logger: logging.Logger) -> str:
        """Construct the public checkpoint/cache API with the CI profile."""
        model_dir, spec_type, draft_model_dir = \
            TestHLAPI._checkpoint_model_and_spec(config, remote_config,
                                                 test_logger)

        cache_dir = os.path.join(config.engine_dir,
                                 "experimental-server-cache")
        plugin_path = os.path.join(env_config.build_dir, "lib",
                                   "libNvInfer_edgellm_plugin.so")
        max_kv = config.max_kv_cache_capacity or config.max_seq_len
        draft_step = (config.eagle_draft_step
                      if spec_type in {"mtp", "eagle3"} else None)
        verify_tokens = (config.max_spec_verify_tokens
                         if spec_type != "none" else None)

        return f"""\
from tensorrt_edgellm.server.runtime.engine_build import BuildOptions
import sys
sys.path.insert(0, "tests")
from defs.utils import ci_server_cache
ci_server_cache.install()
llm = LLM(
    model={model_dir!r},
    cache_dir={cache_dir!r},
    max_input_len={config.max_input_len},
    max_batch_size={config.max_batch_size},
    max_kv_cache_capacity={max_kv},
    draft_top_k={config.eagle_draft_top_k},
    draft_step={draft_step!r},
    proposal_tokens=None,
    verify_tokens={verify_tokens!r},
    context_cache_config={{
        'enabled': {bool(config.context_reuse)!r},
        'max_snapshots': {config.context_cache_max_snapshots or 0},
    }},
    build_options=BuildOptions(
        spec_type={spec_type!r},
        draft_model_dir={draft_model_dir!r},
        max_input_len={config.max_input_len},
        max_batch_size={config.max_batch_size},
        max_kv_cache_capacity={max_kv},
        max_image_tokens={config.max_image_tokens!r},
        max_image_tokens_per_image={config.max_image_tokens_per_image!r},
        plugin_path={plugin_path!r},
    ),
)"""

    def test_http_server_lifecycle(self, test_param: str,
                                   executable_files: Dict[str, str],
                                   remote_config: Optional[RemoteConfig],
                                   test_logger: logging.Logger,
                                   env_config: EnvironmentConfig) -> None:
        """Exercise the installed server CLI and its OpenAI HTTP API."""
        model_type = infer_checkpoint_export_model_type(test_param)
        config = self._checkpoint_config(test_param, model_type, env_config)
        test_logger.info("HTTP server lifecycle: model=%s", config.model_name)

        requests = _openai_requests_from_inputs(
            getattr(test_param, "inference", ()), env_config.llm_sdk_dir)

        server_command = self._server_cli_args(config, env_config,
                                               remote_config, test_logger)
        spec_type = self._spec_type(config)
        script = f"""\
import http.client
import json
import re
import socket
import subprocess
import tempfile
import time


with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]

def request(path, *, payload=None, raw=None, method='POST'):
    body = raw
    if body is None and payload is not None:
        body = json.dumps(payload).encode()
    headers = {{'content-type': 'application/json'}} if body is not None else {{}}
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=180)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return (response.status,
                {{key.lower(): value for key, value in response.getheaders()}},
                response.read())
    finally:
        connection.close()


command = {server_command!r} + ['--port', str(port)]
with tempfile.TemporaryFile(mode='w+') as server_log:
    process = subprocess.Popen(command, stdout=server_log,
                               stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 7200
        health = None
        while process.poll() is None and time.monotonic() < deadline:
            try:
                status, _, body = request('/health', method='GET')
                if status == 200:
                    health = json.loads(body)
                    break
            except OSError:
                pass
            time.sleep(0.25)
        if health is None:
            server_log.flush()
            server_log.seek(0)
            raise AssertionError('server did not become ready:\\n' +
                                 server_log.read())

        assert health['status'] == 'healthy'
        assert health['capabilities']['context_reuse'] is \
            {bool(config.context_reuse)!r}
        assert health['capabilities']['speculative_decoding'] is \
            {bool(spec_type != 'none')!r}
        assert health['capabilities']['speculative_method'] == {spec_type!r}
        in_flight_batching = {bool(config.in_flight_batching)!r}
        assert health['capabilities']['in_flight_batching'] is \
            in_flight_batching, health
        # Under in-flight batching the overlap dimension is the engine batch;
        # the blocking path serves one sequence at a time.
        expected_seqs = {config.max_batch_size} if in_flight_batching else 1
        assert health['capabilities']['max_num_seqs'] == expected_seqs, health

        status, _, body = request('/v1/models', method='GET')
        models = json.loads(body)
        assert status == 200 and models['data'][0]['id']

        status, _, body = request('/v1/chat/completions',
                                  raw=b'{{\"messages\":')
        assert status == 400
        assert json.loads(body)['error']['type'] == 'invalid_request_error'

        for payload in {requests!r}:
            if payload.get('stream'):
                payload.setdefault('stream_options', {{'include_usage': True}})
            status, headers, body = request('/v1/chat/completions',
                                            payload=payload)
            assert status == 200, body.decode(errors='replace')
            if not payload.get('stream'):
                result = json.loads(body)
                choice = result['choices'][0]
                assert (choice['message']['content']
                        or (payload.get('stop')
                            and choice['finish_reason'] == 'stop'))
                assert result['usage']['completion_tokens'] > 0
                continue

            assert headers['content-type'].startswith('text/event-stream')
            frames = [frame for frame in body.decode().split('\\n\\n') if frame]
            assert frames[-1] == 'data: [DONE]'
            payloads = [json.loads(frame.removeprefix('data: '))
                        for frame in frames[:-1]]
            choices = [item['choices'][0] for item in payloads
                       if item.get('choices')]
            assert choices[0]['delta']['role'] == 'assistant'
            assert any(choice['delta'].get('content') for choice in choices)
            assert any(choice.get('finish_reason') for choice in choices)
            usage = [item['usage'] for item in payloads if item.get('usage')]
            assert len(usage) == 1 and usage[0]['completion_tokens'] > 0

        # In-flight batching logs no per-request reuse line; its cumulative
        # counters are on /health.
        reused = []
        if in_flight_batching and {bool(config.context_reuse)!r}:
            status, _, body = request('/health', method='GET')
            assert status == 200
            reused = [json.loads(body)['context_cache']['reused_tokens']]
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        server_log.flush()
        server_log.seek(0)
        server_output = server_log.read()
        print(server_output)
    if {bool(config.context_reuse)!r}:
        reused += [int(value) for value in re.findall(
            r'cumulative: reused=([0-9]+)', server_output)]
        assert reused and max(reused) > 0, server_output[-4000:]
    print('HTTP_SERVER_LIFECYCLE_PASSED')
"""
        python = self._hlapi_python(env_config, remote_config)
        cmd = ['bash', '-c', f'{python} -c {shlex.quote(script)}']
        env_vars = {
            'BUILD_DIR':
            env_config.build_dir,
            'EDGELLM_PLUGIN_PATH':
            os.path.join(env_config.build_dir, 'lib',
                         'libNvInfer_edgellm_plugin.so'),
            'EDGELLM_NANOBIND_DIR':
            os.path.join(env_config.build_dir, 'lib'),
        }
        if env_config.trt_package_dir:
            env_vars['LD_LIBRARY_PATH'] = \
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"

        with timer_context(f"HTTP server lifecycle for {config.model_name}",
                           test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=7500,
                                 logger=test_logger,
                                 env_vars=env_vars)

        output = result.get('output', '')
        if (not result['success']
                or 'HTTP_SERVER_LIFECYCLE_PASSED' not in output):
            pytest.fail(f"HTTP server lifecycle failed:\n{output}")

    def test_ifb_http_server(self, test_param: str,
                             executable_files: Dict[str, str],
                             remote_config: Optional[RemoteConfig],
                             test_logger: logging.Logger,
                             env_config: EnvironmentConfig) -> None:
        """The in-flight batching server applies its scheduling flags end to
        end: /health reports the policy it resolved, concurrent requests all
        complete, a priority -1 request starts at once in the reserved slot
        while every other slot is busy and pauses the others while it
        prefills, a priority below --min-request-priority gets HTTP 400, and
        an option the engine refuses stops the launch naming the flag."""
        config = self._checkpoint_config(test_param, ModelType.LLM, env_config)
        if config.max_batch_size != 4:
            pytest.fail("ifb_server is written for max_batch_size: 4, got "
                        f"{config.max_batch_size}")
        scheduling_flags = [
            '--ifb-scheduling-mode', 'mixed', '--scheduling-policy',
            'priority', '--preemption', 'pause', '--priority-reserved-slots',
            '1', '--max-pause-steps', '4', '--max-num-tokens', '1024',
            '--max-chunk-tokens', '256', '--min-request-priority', '-1'
        ]
        server_command = self._server_cli_args(
            config, env_config, remote_config, test_logger) + scheduling_flags
        script = f"""\
import http.client
import json
import socket
import subprocess
import tempfile
import threading
import time


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


port = free_port()


def request(path, *, payload=None, method='POST'):
    body = None if payload is None else json.dumps(payload).encode()
    headers = {{'content-type': 'application/json'}} if body else {{}}
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=180)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def stream(payload, record):
    \"\"\"Streams one completion into record: when it was sent, its status,
    the time of its first and last chunk, its text and finish reason, or the
    error.\"\"\"
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=600)
    record['sent'] = time.monotonic()
    try:
        connection.request('POST', '/v1/completions',
                           body=json.dumps({{**payload, 'stream': True}}).encode(),
                           headers={{'content-type': 'application/json'}})
        response = connection.getresponse()
        record['status'] = response.status
        if response.status != 200:
            record['error'] = response.read().decode(errors='replace')
            return
        record['text'] = ''
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data: '):
                continue
            if line == 'data: [DONE]':
                break
            chunk = json.loads(line.removeprefix('data: '))
            if 'error' in chunk:
                record['error'] = chunk['error']
                break
            for choice in chunk.get('choices', []):
                if choice.get('text'):
                    record.setdefault('first', time.monotonic())
                    record['text'] += choice['text']
                if choice.get('finish_reason'):
                    record['finish_reason'] = choice['finish_reason']
    except Exception as error:
        record['error'] = repr(error)
    finally:
        record['done'] = time.monotonic()
        connection.close()


def launch(command, log):
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            text=True)


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def output_of(log):
    log.flush()
    log.seek(0)
    return log.read()


command = {server_command!r}
with tempfile.TemporaryFile(mode='w+') as server_log:
    process = launch(command + ['--port', str(port)], server_log)
    try:
        deadline = time.monotonic() + 7200
        health = None
        while process.poll() is None and time.monotonic() < deadline:
            try:
                status, body = request('/health', method='GET')
                if status == 200:
                    health = json.loads(body)
                    break
            except OSError:
                pass
            time.sleep(0.25)
        assert health is not None, 'server did not become ready:\\n' + output_of(server_log)

        assert health['status'] == 'healthy', health
        assert health['capabilities']['in_flight_batching'] is True, health
        assert health['capabilities']['max_num_seqs'] == 4, health
        assert health['engine'] == dict(
            serving=True, unavailable_reason=None, scheduling_mode='mixed',
            scheduling_mode_source='requested', scheduling_reason=None,
            execution='compact', scheduling_policy='priority',
            preemption='pause', max_num_tokens=1024, max_chunk_tokens=256,
            priority_reserved_slots=1, max_pause_steps=4, warnings=[],
            compact_unavailable_reason=None), health['engine']
        status, body = request('/health/ready', method='GET')
        assert status == 200 and json.loads(body)['status'] == 'ready', body

        status, body = request('/v1/completions',
                               payload={{'prompt': 'Hello', 'priority': -2}})
        assert status == 400, body
        assert json.loads(body)['error']['param'] == 'priority', body

        sampling = {{'temperature': 0.0}}
        story = 'Write a long story about a lighthouse keeper and the sea.'
        # Each runs to its own length, so one leaves while the others decode.
        normal = [{{'prompt': story, 'max_tokens': tokens, 'ignore_eos': True,
                    **sampling}} for tokens in (1500, 1700, 1900)]
        records = [{{}} for _ in normal]
        threads = [threading.Thread(target=stream, args=(payload, record))
                   for payload, record in zip(normal, records)]
        for thread in threads:
            thread.start()
        started = time.monotonic() + 300
        while (not all('first' in record or 'done' in record for record in records)
               and time.monotonic() < started):
            time.sleep(0.01)
        assert all('first' in record for record in records), records

        # Three residents leave one free slot, which the priority -1 request
        # reserves: this 600-token prompt waits for a resident to leave and
        # then runs in 256-token chunks beside the others' decoding.
        waiting, urgent = {{}}, {{}}
        long_prompt = ' '.join(['The tide rose over the rocks below the old lighthouse.'] * 50)
        later = [
            threading.Thread(target=stream, args=(
                {{'prompt': long_prompt, 'max_tokens': 32, **sampling}}, waiting)),
            threading.Thread(target=stream, args=(
                {{'prompt': 'What is the capital of France?', 'max_tokens': 16,
                  'priority': -1, **sampling}}, urgent)),
        ]
        later[0].start()
        time.sleep(0.2)
        later[1].start()
        for thread in threads + later:
            thread.join(timeout=900)

        for record in records + [waiting, urgent]:
            assert record.get('status') == 200 and not record.get('error'), record
            assert record['text'].strip() and record.get('finish_reason'), record
        assert all(record['finish_reason'] == 'length' for record in records), records
        first_departure = min(record['done'] for record in records)
        print('urgent: sent %.3f s, first token %.3f s, waiting request first token %.3f s, '
              'after the first running request left at %.3f s' % tuple(
                  t - records[0]['sent'] for t in (urgent['sent'], urgent['first'],
                                                   waiting['first'], first_departure)))
        assert urgent['sent'] < first_departure, (
            'a running request finished before the others were sent; raise its max_tokens')
        assert urgent['first'] < first_departure, (
            'the priority -1 request waited for a running request to leave')
        assert waiting['first'] > first_departure, (
            'a priority 0 request took the reserved slot')

        status, body = request('/health', method='GET')
        scheduling = json.loads(body)['scheduling']
        assert scheduling['paused_resident_steps'] >= 1, scheduling
        assert scheduling['mixed_steps'] >= 1, scheduling
        assert scheduling['partial_prompt_chunks'] >= 1, scheduling
        assert scheduling['completed'] >= 5 and scheduling['failed'] == 0, scheduling
    finally:
        stop(process)
        print(output_of(server_log))

# The engine, not the argument parser, refuses this budget (B + P = 132):
# the launch fails naming the flag.
with tempfile.TemporaryFile(mode='w+') as refused_log:
    refused = launch(command + ['--port', str(free_port()),
                                '--max-num-tokens', '131'], refused_log)
    try:
        refused.wait(timeout=1800)
    finally:
        stop(refused)
    output = output_of(refused_log)
    assert refused.returncode != 0, output
    assert 'invalid server configuration:' in output, output
    assert '--max-num-tokens 131 must be in [B + P, T_max] = [132, 4096]' in output, output
print('IFB_HTTP_SERVER_PASSED')
"""
        python = self._hlapi_python(env_config, remote_config)
        cmd = ['bash', '-c', f'{python} -c {shlex.quote(script)}']
        env_vars = {
            'BUILD_DIR':
            env_config.build_dir,
            'EDGELLM_PLUGIN_PATH':
            os.path.join(env_config.build_dir, 'lib',
                         'libNvInfer_edgellm_plugin.so'),
            'EDGELLM_NANOBIND_DIR':
            os.path.join(env_config.build_dir, 'lib'),
        }
        if env_config.trt_package_dir:
            env_vars['LD_LIBRARY_PATH'] = \
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"

        with timer_context(f"IFB HTTP server for {config.model_name}",
                           test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=9600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        output = result.get('output', '')
        if not result['success'] or 'IFB_HTTP_SERVER_PASSED' not in output:
            pytest.fail(f"IFB HTTP server check failed:\n{output}")

    def test_hlapi_generate(self, test_param: str, executable_files: Dict[str,
                                                                          str],
                            remote_config: Optional[RemoteConfig],
                            test_logger: logging.Logger,
                            env_config: EnvironmentConfig) -> None:
        """Test LLM.generate() from a checkpoint and shared build cache."""
        is_vlm = "-mnit" in test_param
        model_type = ModelType.VLM if is_vlm else ModelType.LLM
        config = self._checkpoint_config(test_param, model_type, env_config)

        test_logger.info("HLAPI generate: model=%s", config.model_name)

        prompt = "Please introduce the company NVIDIA and its CEO."
        max_tokens = 128

        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config, remote_config,
                                         test_logger)

        script = f"""\
{setup}
from tensorrt_edgellm.server import LLM, SamplingParams

{llm_init}
outputs = llm.generate(
    [{prompt!r}],
    SamplingParams(temperature=0.7, max_tokens={max_tokens}),
)
text = outputs[0].text
ids = outputs[0].token_ids
print(f'HLAPI_TEXT_LEN={{len(text)}}')
print(f'HLAPI_IDS_LEN={{len(ids)}}')
print(f'HLAPI_TEXT={{text[:200]}}')
assert len(text) > 0, 'Empty output text'
assert len(ids) > 0, 'Empty output token ids'
print('HLAPI_GENERATE_PASSED')
"""
        script_escaped = shlex.quote(script)
        python = self._hlapi_python(env_config, remote_config)
        cmd = ['bash', '-c', f'{python} -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"HLAPI generate for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"HLAPI generate failed: {result.get('error', 'Unknown')}")

        output = result.get('output', '')
        if 'HLAPI_GENERATE_PASSED' not in output:
            pytest.fail(
                f"HLAPI generate did not produce expected output. Output:\n{output}"
            )

    def test_hlapi_generate_with_logprobs(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Test LLM.generate() returns per-token logprobs when num_logprobs > 0."""
        config = self._checkpoint_config(test_param, ModelType.LLM, env_config)
        test_logger.info("HLAPI generate with logprobs: model=%s",
                         config.model_name)

        prompt = "Count from 1 to 5."
        max_tokens = 32
        num_logprobs = 3
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config, remote_config,
                                         test_logger)

        script = f"""\
{setup}
from tensorrt_edgellm.server import LLM, SamplingParams

{llm_init}
outputs = llm.generate(
    [[{{"role": "user", "content": {prompt!r}}}]],
    SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens={max_tokens},
                   num_logprobs={num_logprobs}),
)
out = outputs[0]
ids = out.token_ids
lps = out.logprobs
print(f'HLAPI_IDS_LEN={{len(ids)}}')
print(f'HLAPI_LOGPROBS_STEPS={{len(lps)}}')
assert len(ids) > 0, 'Empty output token ids'
assert len(lps) == len(ids), f'logprobs steps {{len(lps)}} != token count {{len(ids)}}'
for step in lps:
    assert len(step) == {num_logprobs}, f'Expected {num_logprobs} entries per step, got {{len(step)}}'
    for e in step:
        assert isinstance(e.token_id, int) and e.token_id >= 0, f'Bad token_id {{e.token_id}}'
        assert e.logprob <= 0.0, f'logprob should be <= 0, got {{e.logprob}}'
        assert isinstance(e.token, str), f'token should be str, got {{type(e.token)}}'
        assert isinstance(e.bytes, list), f'bytes should be list, got {{type(e.bytes)}}'
        print('HLAPI_GENERATE_LOGPROBS_PASSED')
"""
        script_escaped = shlex.quote(script)
        python = self._hlapi_python(env_config, remote_config)
        cmd = ['bash', '-c', f'{python} -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"HLAPI generate with logprobs for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"HLAPI generate with logprobs failed: {result.get('error', 'Unknown')}"
            )

        if 'HLAPI_GENERATE_LOGPROBS_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"HLAPI generate logprobs output:\n{result.get('output', '')}")

    def test_hlapi_generate_with_audio(self, test_param: str,
                                       executable_files: Dict[str, str],
                                       remote_config: Optional[RemoteConfig],
                                       test_logger: logging.Logger,
                                       env_config: EnvironmentConfig) -> None:
        """HLAPI audio path, generate + streaming: OpenAI input_audio.data base64 wav -> transcription."""
        is_asr = "-asr" in test_param
        is_omni = "-omni" in test_param
        if not (is_asr or is_omni):
            pytest.skip(
                "audio HLAPI test requires '-asr' or '-omni' test_param")
        # ASR/OMNI param strings carry audio-only tokens (mnts/mxts), which
        # only those model types are allowed to parse.
        model_type = ModelType.ASR if is_asr else ModelType.OMNI
        config = self._checkpoint_config(test_param, model_type, env_config)
        test_wav = (getattr(config, "get_audio_test_wav", lambda: "")()
                    or os.environ.get("AUDIO_TEST_WAV", ""))
        if test_wav:
            wav_setup = f"""\
with open({test_wav!r}, 'rb') as f:
    wav_bytes = f.read()"""
        else:
            # Synthesize a 1 s sine wav with the stdlib on the machine that
            # runs inference; the transcription is meaningless but the full
            # decode -> mel -> audio-encoder -> LLM path still executes.
            wav_setup = """\
import io, math, struct, wave
buf = io.BytesIO()
with wave.open(buf, 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b''.join(
        struct.pack('<h', int(12000 * math.sin(2 * math.pi * 440 * t / 16000)))
        for t in range(16000)))
wav_bytes = buf.getvalue()"""

        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config, remote_config,
                                         test_logger)
        script = f"""\
{setup}
import base64
{wav_setup}
wav_b64 = base64.b64encode(wav_bytes).decode()
from tensorrt_edgellm.server import LLM, SamplingParams
{llm_init}
messages = [{{'role': 'user', 'content': [
    {{'type': 'input_audio', 'input_audio': {{'data': wav_b64, 'format': 'wav'}}}}
]}}]
outputs = llm.generate([messages], SamplingParams(temperature=1.0, max_tokens=128))
assert len(outputs[0].token_ids) > 0
print('HLAPI_GENERATE_WITH_AUDIO_PASSED')
chunks = list(llm.generate_stream(messages, SamplingParams(temperature=1.0, max_tokens=32)))
# a sine input can transcribe to almost nothing, so a single terminal
# chunk is a legal stream; require termination, not chunk count.
assert chunks and any(c.finished for c in chunks), 'Audio stream failed'
print('HLAPI_STREAM_WITH_AUDIO_PASSED')
"""
        script_escaped = shlex.quote(script)
        python = self._hlapi_python(env_config, remote_config)
        cmd = ['bash', '-c', f'{python} -c {script_escaped}']
        env_vars = {
            "LD_LIBRARY_PATH":
            f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
        } if env_config.trt_package_dir else None
        with timer_context(f"HLAPI audio generate for {config.model_name}",
                           test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        output = result.get('output', '')
        if (not result['success']
                or 'HLAPI_GENERATE_WITH_AUDIO_PASSED' not in output
                or 'HLAPI_STREAM_WITH_AUDIO_PASSED' not in output):
            pytest.fail(f"HLAPI audio generate failed:\n{output}")

    def test_hlapi_generate_with_logit_bias(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Validate non-streaming HLAPI logit_bias behavior.

        Runs generation in a subprocess against a real engine. The +100 case
        selects a non-special tokenizer ID and verifies that deterministic
        generation returns it for both the prefill-sampled token and a decode
        token. The -100 case first records the baseline greedy token, then
        verifies biasing that token suppresses it. A speculative bundle remains
        on its native speculative path.
        """
        config = self._checkpoint_config(test_param, ModelType.LLM, env_config)

        test_logger.info("HLAPI logit_bias: model=%s", config.model_name)

        prompt = "Complete this sentence with one short word: NVIDIA makes"
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config, remote_config,
                                         test_logger)

        script = f"""\
{setup}
import json
import os
from tensorrt_edgellm.server import LLM, SamplingParams

def pick_positive_bias_target_id(model_dir):
    tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
    with open(tokenizer_path, encoding='utf-8') as f:
        tokenizer = json.load(f)

    special_ids = set()
    for token in tokenizer.get('added_tokens', []):
        token_id = token.get('id')
        if token.get('special') and isinstance(token_id, int):
            special_ids.add(token_id)
    for token_id, token in tokenizer.get('added_tokens_decoder', {{}}).items():
        if token.get('special'):
            try:
                special_ids.add(int(token_id))
            except ValueError:
                pass

    vocab = tokenizer.get('model', {{}}).get('vocab', {{}})
    preferred_pieces = (
        ' NVIDIA', 'NVIDIA', ' hello', 'Hello', ' the', 'The',
        ' answer', 'Answer', ' cat', 'cat', '!', '.',
        'ĠNVIDIA', 'Ġhello', 'Ġthe', 'Ġanswer', 'Ġcat',
        '▁NVIDIA', '▁hello', '▁the', '▁answer', '▁cat',
    )
    for piece in preferred_pieces:
        token_id = vocab.get(piece)
        if isinstance(token_id, int) and token_id not in special_ids:
            return token_id

    vocab_items = (
        (piece, token_id) for piece, token_id in vocab.items()
        if isinstance(token_id, int)
    )
    for piece, token_id in sorted(vocab_items, key=lambda item: item[1]):
        if (
            token_id not in special_ids
            and piece
            and not piece.startswith(('<', '[', '{{'))
        ):
            return token_id

    raise RuntimeError('Could not find a non-special token ID for logit_bias')

def generate_ids(llm, *, max_tokens=1, logit_bias=None):
    outputs = llm.generate(
        [{prompt!r}],
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            max_tokens=max_tokens,
            logit_bias=logit_bias or {{}},
        ),
    )
    ids = outputs[0].token_ids
    assert ids, 'Expected at least one generated token id'
    return ids

{llm_init}

target_token_id = pick_positive_bias_target_id(llm.model_dir)
forced_token_count = 2
positive_token_ids = generate_ids(
    llm,
    max_tokens=forced_token_count,
    logit_bias={{target_token_id: 100.0}},
)
print(f'HLAPI_POSITIVE_TARGET_ID={{target_token_id}}')
print(f'HLAPI_POSITIVE_TOKEN_IDS={{positive_token_ids}}')
assert len(positive_token_ids) == forced_token_count, (
    f'Expected {{forced_token_count}} generated tokens, got {{positive_token_ids}}'
)
assert all(token_id == target_token_id for token_id in positive_token_ids), (
    f'Expected +100 logit_bias to force {{target_token_id}} for prefill and decode, '
    f'got {{positive_token_ids}}'
)

baseline_token_id = generate_ids(llm)[0]
negative_token_id = generate_ids(
    llm, logit_bias={{baseline_token_id: -100.0}}
)[0]
print(f'HLAPI_NEGATIVE_BANNED_ID={{baseline_token_id}}')
print(f'HLAPI_NEGATIVE_TOKEN_ID={{negative_token_id}}')
assert negative_token_id != baseline_token_id, (
    f'Expected -100 logit_bias to suppress {{baseline_token_id}}, got {{negative_token_id}}'
)
print('HLAPI_GENERATE_WITH_LOGIT_BIAS_PASSED')
"""
        script_escaped = shlex.quote(script)
        python = self._hlapi_python(env_config, remote_config)
        cmd = ['bash', '-c', f'{python} -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(f"HLAPI logit_bias for {config.model_name}",
                           test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        if not result['success']:
            pytest.fail(
                f"HLAPI logit_bias failed: {result.get('error', 'Unknown')}")
        if 'HLAPI_GENERATE_WITH_LOGIT_BIAS_PASSED' not in result.get(
                'output', ''):
            pytest.fail(
                f"HLAPI logit_bias output:\n{result.get('output', '')}")

    def test_hlapi_generate_with_guided_decoding(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Validate guided decoding end to end through the HLAPI.

        Generation is constrained token by token, so the assertion is that the output
        really satisfies the guide -- a quality metric would not catch a violation. Covers
        the OpenAI-compatible `response_format` surface, the low-level `guided_decoding`
        surface (which additionally reaches regex), and the pre-check that turns an
        unenforceable schema keyword into a clean error instead of a silently wrong answer.
        """
        config = self._checkpoint_config(test_param, ModelType.LLM, env_config)

        test_logger.info("HLAPI guided decoding: model=%s", config.model_name)

        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config, remote_config,
                                         test_logger)

        script = f"""\
{setup}
import json
import re
from tensorrt_edgellm.server import LLM, SamplingParams

{llm_init}

def generate(prompt, **kwargs):
    params = SamplingParams(temperature=0.0, max_tokens=96, **kwargs)
    return llm.generate(prompt, params)[0].text

schema = {{
    'type': 'object',
    'properties': {{'name': {{'type': 'string'}}, 'age': {{'type': 'integer'}}}},
    'required': ['name', 'age'],
}}

text = generate('Give me a person record with a name and an age.',
                guided_decoding=('json_schema', json.dumps(schema, sort_keys=True)))
print(f'HLAPI_GUIDED_JSON_SCHEMA={{text!r}}')
document = json.loads(text)
assert isinstance(document, dict), f'expected a JSON object, got {{document!r}}'
assert isinstance(document.get('name'), str), f'name must be a string: {{document!r}}'
assert isinstance(document.get('age'), int), f'age must be an integer: {{document!r}}'

pattern = '[a-z]+@[a-z]+\\.(com|org)'
text = generate('What is an email address for support?',
                guided_decoding=('regex', pattern))
print(f'HLAPI_GUIDED_REGEX={{text!r}}')
assert re.fullmatch(pattern, text), f'{{text!r}} does not match {{pattern!r}}'

from tensorrt_edgellm.server.runtime.engine import _import_runtime
runtime = _import_runtime()
params = runtime.GuidedDecodingParams(
    runtime.GuideType.JSON_SCHEMA, json.dumps({{'type': 'integer', 'multipleOf': 5}}))
valid, reason = runtime.validate_guided_decoding_params(params)
print(f'HLAPI_GUIDED_PRECHECK={{valid}} {{reason!r}}')
assert not valid and 'multipleOf' in reason, (
    f'multipleOf should be rejected up front, got valid={{valid}} reason={{reason!r}}'
)

print('HLAPI_GENERATE_WITH_GUIDED_DECODING_PASSED')
"""
        script_escaped = shlex.quote(script)
        python = self._hlapi_python(env_config, remote_config)
        cmd = ['bash', '-c', f'{python} -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(f"HLAPI guided decoding for {config.model_name}",
                           test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=900,
                                 logger=test_logger,
                                 env_vars=env_vars)
        if not result['success']:
            pytest.fail("HLAPI guided decoding failed: "
                        f"{result.get('error', 'Unknown')}")
        if 'HLAPI_GENERATE_WITH_GUIDED_DECODING_PASSED' not in result.get(
                'output', ''):
            pytest.fail(
                f"HLAPI guided decoding output:\n{result.get('output', '')}")

    def test_hlapi_video_generate(self, test_param: str,
                                  executable_files: Dict[str, str],
                                  remote_config: Optional[RemoteConfig],
                                  test_logger: logging.Logger,
                                  env_config: EnvironmentConfig) -> None:
        """HLAPI video path, generate + streaming: local clip -> decode+sample ->
        video ImageData -> per-model ViT runner, covering video-capable VLM and
        Omni models. ``VIDEO_TEST_CLIP`` overrides the synthetic PyAV clip
        encoded on the inference machine; ``VIDEO_TEST_NFRAMES`` must fit the
        visual engine's profile."""
        if "-mnit" not in test_param:
            pytest.skip("video HLAPI test requires a multimodal test_param")
        model_type = (ModelType.OMNI
                      if "-omni" in test_param.lower() else ModelType.VLM)
        config = self._checkpoint_config(test_param, model_type, env_config)
        test_clip = os.environ.get("VIDEO_TEST_CLIP", "")
        nframes = int(os.environ.get("VIDEO_TEST_NFRAMES", "8"))

        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config, remote_config,
                                         test_logger)
        if test_clip:
            clip_setup = f"clip_path = {test_clip!r}"
        else:
            # Encode a deterministic synthetic clip (moving gradient bar)
            # where the inference runs; decode uses the same PyAV the server
            # video path depends on.
            clip_setup = f"""\
import numpy as np
import av
import tempfile
clip_path = tempfile.mktemp(suffix='.mp4')
container = av.open(clip_path, 'w')
stream = container.add_stream('libx264', rate=4)
# 256x256 keeps the sampled frames above the ViT engine's minimum
# totalSeqLength for the mnit128 profiles used in CI.
stream.width, stream.height, stream.pix_fmt = 256, 256, 'yuv420p'
for i in range({max(16, nframes * 2)}):
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, 256, dtype=np.uint8)[None, :]
    frame[(i * 16) % 256:(i * 16) % 256 + 32, :, 1] = 255
    for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')):
        container.mux(packet)
for packet in stream.encode():
    container.mux(packet)
container.close()"""
        script = f"""\
{setup}
{clip_setup}
from tensorrt_edgellm.server import LLM, SamplingParams
{llm_init}
messages = [{{'role': 'user', 'content': [
    {{'type': 'video', 'video': clip_path, 'nframes': {nframes}}},
    {{'type': 'text', 'text': 'Describe what happens in this video.'}}
]}}]
outputs = llm.generate([messages], SamplingParams(temperature=0.0, max_tokens=64))
text = outputs[0].text
print(f'HLAPI_VIDEO_TEXT={{text[:200]}}')
assert len(outputs[0].token_ids) > 0, 'Empty output'
print('HLAPI_GENERATE_WITH_VIDEO_PASSED')
chunks = list(llm.generate_stream(messages, SamplingParams(temperature=0.0, max_tokens=32)))
stream_text = ''.join(c.text for c in chunks)
assert len(chunks) > 1 and stream_text, 'Empty video stream'
assert any(c.finished for c in chunks), 'No terminal chunk'
print('HLAPI_STREAM_WITH_VIDEO_PASSED')
too_long = [{{'role': 'user', 'content': [
    {{'type': 'video', 'video': clip_path, 'nframes': {nframes}}},
    {{'type': 'text', 'text': 'word ' * 20000}}
]}}]
try:
    llm.generate([too_long], SamplingParams(temperature=0.0, max_tokens=8))
    raise AssertionError('over-long video request unexpectedly succeeded')
except AssertionError:
    raise
except Exception as exc:
    assert 'EDGELLM_INPUT_TOO_LONG' in str(exc), f'wrong error: {{exc}}'
print('HLAPI_VIDEO_TOO_LONG_PASSED')
"""
        script_escaped = shlex.quote(script)
        python = self._hlapi_python(env_config, remote_config)
        cmd = ['bash', '-c', f'{python} -c {script_escaped}']
        env_vars = {
            "LD_LIBRARY_PATH":
            f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
        } if env_config.trt_package_dir else None
        with timer_context(f"HLAPI video generate for {config.model_name}",
                           test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        output = result.get('output', '')
        if (not result['success']
                or 'HLAPI_GENERATE_WITH_VIDEO_PASSED' not in output
                or 'HLAPI_STREAM_WITH_VIDEO_PASSED' not in output
                or 'HLAPI_VIDEO_TOO_LONG_PASSED' not in output):
            pytest.fail(f"HLAPI video generate failed:\n{output}")

    def test_hlapi_streaming(self, test_param: str,
                             executable_files: Dict[str, str],
                             remote_config: Optional[RemoteConfig],
                             test_logger: logging.Logger,
                             env_config: EnvironmentConfig) -> None:
        """Test LLM.generate_stream() from the checkpoint cache."""
        config = self._checkpoint_config(test_param, ModelType.LLM, env_config)

        test_logger.info("HLAPI streaming: model=%s", config.model_name)

        prompt = "Count from 1 to 10."
        max_tokens = 128

        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config, remote_config,
                                         test_logger)

        script = f"""\
{setup}
from tensorrt_edgellm.server import LLM, SamplingParams

{llm_init}
chunks = list(llm.generate_stream(
    [{{"role": "user", "content": {prompt!r}}}],
    SamplingParams(temperature=0.7, max_tokens={max_tokens}),
))
total_text = ''.join(c.text for c in chunks)
print(f'HLAPI_STREAM_CHUNKS={{len(chunks)}}')
print(f'HLAPI_STREAM_TEXT_LEN={{len(total_text)}}')
print(f'HLAPI_STREAM_TEXT={{total_text[:200]}}')
assert len(chunks) > 1, f'Expected multiple chunks, got {{len(chunks)}}'
assert len(total_text) > 0, 'Empty streamed text'
assert any(c.finished for c in chunks), 'No terminal chunk received'
print('HLAPI_STREAMING_PASSED')
"""
        script_escaped = shlex.quote(script)
        python = self._hlapi_python(env_config, remote_config)
        cmd = ['bash', '-c', f'{python} -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"HLAPI streaming for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"HLAPI streaming failed: {result.get('error', 'Unknown')}")

        output = result.get('output', '')
        if 'HLAPI_STREAMING_PASSED' not in output:
            pytest.fail(
                f"HLAPI streaming did not produce expected output. Output:\n{output}"
            )

    def test_hlapi_streaming_with_logprobs(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Test LLM.generate_stream() delivers per-token logprobs matching token count."""
        config = self._checkpoint_config(test_param, ModelType.LLM, env_config)
        test_logger.info("HLAPI streaming with logprobs: model=%s",
                         config.model_name)

        prompt = "Count from 1 to 5."
        max_tokens = 32
        num_logprobs = 3
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config, remote_config,
                                         test_logger)

        script = f"""\
{setup}
from tensorrt_edgellm.server import LLM, SamplingParams

{llm_init}
chunks = list(llm.generate_stream(
    [{{"role": "user", "content": {prompt!r}}}],
    SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens={max_tokens},
                   num_logprobs={num_logprobs}),
))
total_tokens = sum(len(c.token_ids) for c in chunks)
total_lp_steps = sum(len(c.logprobs) for c in chunks)
print(f'HLAPI_STREAM_TOTAL_TOKENS={{total_tokens}}')
print(f'HLAPI_STREAM_TOTAL_LP_STEPS={{total_lp_steps}}')
assert total_tokens > 0, 'No tokens received'
assert total_lp_steps == total_tokens, f'logprob steps {{total_lp_steps}} != token count {{total_tokens}}'
for chunk in chunks:
    for step in chunk.logprobs:
        assert len(step) == {num_logprobs}, f'Expected {num_logprobs} entries per step, got {{len(step)}}'
        for e in step:
            assert isinstance(e.token_id, int) and e.token_id >= 0, f'Bad token_id {{e.token_id}}'
            assert e.logprob <= 0.0, f'logprob should be <= 0, got {{e.logprob}}'
            assert isinstance(e.token, str), f'token should be str, got {{type(e.token)}}'
            assert isinstance(e.bytes, list), f'bytes should be list, got {{type(e.bytes)}}'

# Also exercise the real serving layer on the same engine: streaming
# choices[0].logprobs must be the OpenAI object shape {{"content": [...]}} with
# nested top_logprobs.
import asyncio as _asyncio
import json as _json
from tensorrt_edgellm.server.config import ApiConfig
from tensorrt_edgellm.server.api.protocol import ChatCompletionRequest
from tensorrt_edgellm.server.api.serving_chat import OpenAIServingChat
from tensorrt_edgellm.server.runtime.engine_client import EngineClient
_request = ChatCompletionRequest(
    messages=[{{"role": "user", "content": {prompt!r}}}],
    stream=True,
    logprobs=True,
    top_logprobs={num_logprobs},
    max_tokens={max_tokens},
    temperature=0.0,
)
_config = ApiConfig()
_handler = OpenAIServingChat(EngineClient(llm, _config), _config)
async def _collect_sse():
    return [chunk async for chunk in _handler.stream_chat_completion(_request)]
lp_objs = []
for sse in _asyncio.run(_collect_sse()):
    if not sse.startswith('data: ') or sse.strip() == 'data: [DONE]':
        continue
    _choice = _json.loads(sse[len('data: '):].strip())['choices'][0]
    assert 'logprobs' not in _choice.get('delta', {{}}), 'logprobs must not live inside delta'
    if _choice.get('logprobs') is not None:
        lp_objs.append(_choice['logprobs'])
assert lp_objs, 'no logprobs found in any SSE chunk'
for lp in lp_objs:
    assert isinstance(lp, dict) and 'content' in lp, f'SSE logprobs not OpenAI-shaped: {{lp}}'
    for c in lp['content']:
        assert set(c) >= {{'token', 'token_id', 'bytes', 'logprob', 'top_logprobs'}}, f'missing keys: {{sorted(c)}}'
        assert len(c['top_logprobs']) == {num_logprobs}, f'expected {num_logprobs} top_logprobs, got {{len(c["top_logprobs"])}}'
print('HLAPI_STREAMING_LOGPROBS_PASSED')
"""
        script_escaped = shlex.quote(script)
        python = self._hlapi_python(env_config, remote_config)
        cmd = ['bash', '-c', f'{python} -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"HLAPI streaming with logprobs for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"HLAPI streaming with logprobs failed: {result.get('error', 'Unknown')}"
            )

        if 'HLAPI_STREAMING_LOGPROBS_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"HLAPI streaming logprobs output:\n{result.get('output', '')}"
            )

    def test_hlapi_generate_with_stop(self, test_param: str,
                                      executable_files: Dict[str, str],
                                      remote_config: Optional[RemoteConfig],
                                      test_logger: logging.Logger,
                                      env_config: EnvironmentConfig) -> None:
        """HLAPI non-streaming: SamplingParams(stop=[...]) trims output, finish_reason == 'stop'."""
        config = self._checkpoint_config(test_param, ModelType.LLM, env_config)
        prompt = "List three colors, separated by commas. End your list with '###'."
        stop = "###"
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config, remote_config,
                                         test_logger)

        script = f"""\
{setup}
from tensorrt_edgellm.server import LLM, SamplingParams

{llm_init}
outputs = llm.generate(
    [{prompt!r}],
    SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens=128, stop=[{stop!r}]),
)
text = outputs[0].text
reason = outputs[0].finish_reason
print(f'HLAPI_TEXT={{text!r}}')
print(f'HLAPI_REASON={{reason}}')
assert {stop!r} not in text, f'Stop string leaked into output: {{text!r}}'
assert reason == 'stop', f'Expected reason=stop, got {{reason}}'
print('HLAPI_GENERATE_WITH_STOP_PASSED')
"""
        script_escaped = shlex.quote(script)
        python = self._hlapi_python(env_config, remote_config)
        cmd = ['bash', '-c', f'{python} -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(f"HLAPI generate with stop for {config.model_name}",
                           test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        if not result['success']:
            pytest.fail(
                f"HLAPI generate with stop failed: {result.get('error', 'Unknown')}"
            )
        if 'HLAPI_GENERATE_WITH_STOP_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"HLAPI generate with stop output:\n{result.get('output', '')}"
            )

    def test_hlapi_streaming_with_stop(self, test_param: str,
                                       executable_files: Dict[str, str],
                                       remote_config: Optional[RemoteConfig],
                                       test_logger: logging.Logger,
                                       env_config: EnvironmentConfig) -> None:
        """HLAPI streaming: stop string trimmed from chunks, last chunk reason == 'stop'."""
        config = self._checkpoint_config(test_param, ModelType.LLM, env_config)
        prompt = "List three colors, separated by commas. End your list with '###'."
        stop = "###"
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config, remote_config,
                                         test_logger)

        script = f"""\
{setup}
from tensorrt_edgellm.server import LLM, SamplingParams

{llm_init}
chunks = list(llm.generate_stream(
    [{{"role": "user", "content": {prompt!r}}}],
    SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens=128, stop=[{stop!r}]),
))
text = ''.join(c.text for c in chunks)
terminal_reason = next((c.finish_reason for c in chunks if c.finished), None)
print(f'HLAPI_STREAM_TEXT={{text!r}}')
print(f'HLAPI_TERMINAL_REASON={{terminal_reason}}')
assert {stop!r} not in text, f'Stop string leaked: {{text!r}}'
assert terminal_reason == 'stop', f'Expected reason=stop, got {{terminal_reason}}'
print('HLAPI_STREAMING_WITH_STOP_PASSED')
"""
        script_escaped = shlex.quote(script)
        python = self._hlapi_python(env_config, remote_config)
        cmd = ['bash', '-c', f'{python} -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(
                f"HLAPI streaming with stop for {config.model_name}",
                test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        if not result['success']:
            pytest.fail(
                f"HLAPI streaming with stop failed: {result.get('error', 'Unknown')}"
            )
        if 'HLAPI_STREAMING_WITH_STOP_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"HLAPI streaming with stop output:\n{result.get('output', '')}"
            )

    def test_hlapi_generate_length_finish_reason(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Verify HLAPI non-streaming reports finish_reason='length' on max_tokens hit."""
        config = self._checkpoint_config(test_param, ModelType.LLM, env_config)
        prompt = "Write a long detailed essay about transformer neural networks."
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config, remote_config,
                                         test_logger)

        script = f"""\
{setup}
from tensorrt_edgellm.server import LLM, SamplingParams

{llm_init}
outputs = llm.generate(
    [{prompt!r}],
    SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens=8),
)
print(f'HLAPI_REASON={{outputs[0].finish_reason}}')
assert outputs[0].finish_reason == 'length', f'Expected length, got {{outputs[0].finish_reason}}'
print('HLAPI_GENERATE_LENGTH_REASON_PASSED')
"""
        script_escaped = shlex.quote(script)
        python = self._hlapi_python(env_config, remote_config)
        cmd = ['bash', '-c', f'{python} -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(f"HLAPI length-reason for {config.model_name}",
                           test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        if not result['success']:
            pytest.fail(
                f"HLAPI length-reason failed: {result.get('error', 'Unknown')}"
            )
        if 'HLAPI_GENERATE_LENGTH_REASON_PASSED' not in result.get(
                'output', ''):
            pytest.fail(
                f"HLAPI length-reason output:\n{result.get('output', '')}")

    def test_hlapi_streaming_length_finish_reason(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """HLAPI streaming: terminal chunk reports finish_reason='length' on max_tokens hit."""
        config = self._checkpoint_config(test_param, ModelType.LLM, env_config)
        prompt = "Write a long detailed essay about transformer neural networks."
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config, remote_config,
                                         test_logger)

        script = f"""\
{setup}
from tensorrt_edgellm.server import LLM, SamplingParams

{llm_init}
chunks = list(llm.generate_stream(
    [{{"role": "user", "content": {prompt!r}}}],
    SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens=8),
))
terminal_reason = next((c.finish_reason for c in chunks if c.finished), None)
print(f'HLAPI_TERMINAL_REASON={{terminal_reason}}')
assert terminal_reason == 'length', f'Expected length, got {{terminal_reason}}'
print('HLAPI_STREAMING_LENGTH_REASON_PASSED')
"""
        script_escaped = shlex.quote(script)
        python = self._hlapi_python(env_config, remote_config)
        cmd = ['bash', '-c', f'{python} -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(
                f"HLAPI streaming length-reason for {config.model_name}",
                test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        if not result['success']:
            pytest.fail(
                f"HLAPI streaming length-reason failed: {result.get('error', 'Unknown')}"
            )
        if 'HLAPI_STREAMING_LENGTH_REASON_PASSED' not in result.get(
                'output', ''):
            pytest.fail(
                f"HLAPI streaming length-reason output:\n{result.get('output', '')}"
            )
