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
End-to-end tests for the experimental Python server (pybind11 runtime).

Tests the full pipeline: build pybind extension -> load TRT engine via
Python API -> run inference -> validate output.
"""
import logging
import os
import shlex
from typing import Dict, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import run_command, timer_context

from .config import ModelType, TaskType, TestConfig

# Server-deps venv provisioned by test_build_pybind, relative to the repo root
# (workspace root remotely, LLM_SDK_DIR locally).
_SERVER_VENV = 'venv/server'


def test_build_pybind(env_config: EnvironmentConfig,
                      remote_config: Optional[RemoteConfig],
                      test_logger: logging.Logger):
    """Build the pybind extension by reconfiguring the existing build directory.

    Uses subdirectory mode (BUILD_PYTHON_BINDINGS=ON) so all cmake
    variables (CUDA, TRT, toolchain) are inherited from the prior
    test_build_project configuration.
    """
    build_dir = env_config.build_dir

    repo_root = '.' if remote_config else env_config.llm_sdk_dir
    pybind_venv = _SERVER_VENV
    install_cmd = (f'cd {shlex.quote(repo_root)}'
                   f' && python3 -m venv {pybind_venv}'
                   f' && {pybind_venv}/bin/pip install -q'
                   ' -e ".[server,native-build]"')
    if env_config.trt_package_dir:
        trt_python_dir = shlex.quote(
            os.path.join(env_config.trt_package_dir, "python"))
        install_cmd += (
            f' && PY_TAG=$({pybind_venv}/bin/python -c '
            "'import sys; print(\"cp%d%d\" % sys.version_info[:2])')"
            f' && TRT_WHL=$(find {trt_python_dir} -maxdepth 1 -type f '
            "-name \"tensorrt-*${PY_TAG}*.whl\" -print -quit)"
            ' && test -n "$TRT_WHL"'
            f' && {pybind_venv}/bin/pip install -q "$TRT_WHL"')
    result = run_command(cmd=['bash', '-c', install_cmd],
                         remote_config=remote_config,
                         timeout=120,
                         logger=test_logger)
    if not result['success']:
        pytest.fail(
            f"Failed to install server dependencies: {result.get('error')}")

    pybind_python = f'{repo_root}/{pybind_venv}/bin/python'
    pybind11_dir_expr = (
        f'$({shlex.quote(pybind_python)}'
        ' -c "import pybind11; print(pybind11.get_cmake_dir())")')
    build_cmd = (f'PYBIND11_DIR={pybind11_dir_expr}'
                 f' && cd {build_dir}'
                 f' && cmake .. -DBUILD_PYTHON_BINDINGS=ON'
                 f' -Dpybind11_DIR=$PYBIND11_DIR'
                 f' && make -j$(nproc) _edgellm_runtime')

    with timer_context("Building pybind extension", test_logger):
        result = run_command(cmd=['bash', '-c', build_cmd],
                             remote_config=remote_config,
                             timeout=600,
                             logger=test_logger)

    if not result['success']:
        pytest.fail(f"Pybind build failed: {result.get('error')}")

    pybind_output_dir = f'{build_dir}/pybind'
    result = run_command(
        cmd=['bash', '-c', f'ls {pybind_output_dir}/*_edgellm_runtime*.so'],
        remote_config=remote_config,
        timeout=10,
        logger=test_logger)
    if not result['success']:
        pytest.fail(
            f"_edgellm_runtime.so not found in {pybind_output_dir} after build"
        )


class TestServerPipeline:
    """E2E tests for inference via the experimental Python server API."""

    def test_server_inference(self, test_param: str,
                              executable_files: Dict[str, str],
                              remote_config: Optional[RemoteConfig],
                              test_logger: logging.Logger,
                              env_config: EnvironmentConfig) -> None:
        """Test inference using the pybind Python runtime directly."""
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

        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "Please introduce the company NVIDIA and its CEO."
        max_tokens = 128

        script = f"""\
import sys, os
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
if not so_files:
    raise RuntimeError('_edgellm_runtime.so not found in ' + {pybind_build_dir!r})
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
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
        """Audio pybind path: in-memory PCM -> AudioData -> Request.audio_buffers -> runtime."""
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

        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        script = f"""\
import sys, os, importlib.util
sys.path.insert(0, {pybind_build_dir!r})
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
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
        """Test streaming inference using StreamChannel via pybind."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)

        engine_dir = config.get_llm_engine_dir()
        test_logger.info("Using engine dir: %s", engine_dir)

        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "Count from 1 to 10."
        max_tokens = 128

        script = f"""\
import sys, os, threading
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
if not so_files:
    raise RuntimeError('_edgellm_runtime.so not found in ' + {pybind_build_dir!r})
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
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

    def test_server_inference_with_logprobs(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Test non-streaming inference returns per-token logprobs via pybind."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        test_logger.info("Server inference with logprobs: engine=%s",
                         engine_dir)

        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "Count from 1 to 5."
        max_tokens = 32
        num_logprobs = 3

        script = f"""\
import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
if not so_files:
    raise RuntimeError('_edgellm_runtime.so not found in ' + {pybind_build_dir!r})
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
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
from experimental.server.api.serving_chat import _format_logprob_steps
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
        """Test streaming inference delivers per-token logprobs on each chunk via pybind."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        test_logger.info("Server streaming with logprobs: engine=%s",
                         engine_dir)

        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "Count from 1 to 5."
        max_tokens = 32
        num_logprobs = 3

        script = f"""\
import sys, os, threading
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
if not so_files:
    raise RuntimeError('_edgellm_runtime.so not found in ' + {pybind_build_dir!r})
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
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
        """Non-streaming pybind path: Request.stop_strings + response.finish_reasons round-trip."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "List three colors, separated by commas. End your list with '###'."
        stop = "###"

        script = f"""\
import sys, os
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
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
        """Streaming pybind path: stop string trims chunk text, terminal chunk has STOP_WORDS reason."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "List three colors, separated by commas. End your list with '###'."
        stop = "###"

        script = f"""\
import sys, os, threading
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
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
        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "Write a long detailed essay about transformer neural networks."

        script = f"""\
import sys, os
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
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


class TestHLAPI:
    """E2E tests for checkpoint-direct high-level Python inference."""

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
        created by test_build_pybind under the same repository root."""
        repo_root = '.' if remote_config else env_config.llm_sdk_dir
        return f'{repo_root}/{_SERVER_VENV}/bin/python3'

    @staticmethod
    def _llm_init_script(config: TestConfig,
                         env_config: EnvironmentConfig) -> str:
        """Construct the public checkpoint/cache API with the CI profile."""
        try:
            model_dir = config.get_torch_model_dir()
            spec_type = "none"
            draft_model_dir = ""
            if config.is_mtp:
                spec_type = "mtp"
                if config.model_name.lower().startswith("gemma-"):
                    draft_model_dir = \
                        config.get_gemma4_mtp_assistant_model_dir()
            elif config.is_dflash:
                spec_type = "dflash"
                draft_model_dir = config.get_dflash_draft_model_dir()
            elif config.is_dspark:
                spec_type = "dspark"
                draft_model_dir = config.get_dspark_draft_model_dir()
            elif config.is_eagle:
                spec_type = "eagle3"
                draft_model_dir = config.get_eagle_draft_checkpoint_dir()
        except ValueError as exc:
            pytest.skip("checkpoint-direct HLAPI source is unavailable on "
                        f"this runner: {exc}")

        cache_dir = os.path.join(config.engine_dir,
                                 "experimental-server-cache")
        plugin_path = os.path.join(env_config.build_dir,
                                   "libNvInfer_edgellm_plugin.so")
        max_kv = config.max_kv_cache_capacity or config.max_seq_len

        return f"""\
from experimental.server.runtime.engine_build import BuildOptions
llm = LLM(
    model={model_dir!r},
    cache_dir={cache_dir!r},
    max_input_len={config.max_input_len},
    max_batch_size={config.max_batch_size},
    max_kv_cache_capacity={max_kv},
    draft_top_k={config.eagle_draft_top_k},
    draft_step={config.eagle_draft_step},
    verify_tree_size={config.max_verify_tree_size},
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
        """Exercise the real ASGI API, including stream cancellation."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        test_logger.info("HTTP server lifecycle: model=%s", config.model_name)

        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config)
        script = f"""\
{setup}
import http.client
import json
import socket
import threading
import time

import uvicorn

from experimental.server import LLM
from experimental.server.api.app import create_app
from experimental.server.config import ApiConfig
from experimental.server.runtime.engine_client import EngineClient

{llm_init}
api_config = ApiConfig(max_queued_requests=0, queue_timeout=5.0)
engine_client = EngineClient(llm, api_config)
app = create_app(engine_client, api_config)


with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]

server = uvicorn.Server(uvicorn.Config(
    app,
    host='127.0.0.1',
    port=port,
    log_level='error',
    access_log=False,
))
server_error = []


def run_server():
    try:
        server.run()
    except BaseException as error:
        server_error.append(error)


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


worker = threading.Thread(target=run_server, daemon=True)
worker.start()
deadline = time.monotonic() + 60
while not server.started and worker.is_alive() and time.monotonic() < deadline:
    time.sleep(0.05)
assert server.started and not server_error, server_error

try:
    status, _, body = request('/health', method='GET')
    health = json.loads(body)
    assert status == 200 and health['status'] == 'healthy'
    assert health['capabilities']['max_num_seqs'] == 1

    status, _, body = request('/v1/models', method='GET')
    models = json.loads(body)
    assert status == 200 and models['data'][0]['id']

    status, _, body = request('/v1/chat/completions',
                              raw=b'{{\"messages\":')
    assert status == 400
    assert json.loads(body)['error']['type'] == 'invalid_request_error'
    assert engine_client.active_requests == 0

    completion = {{
        'messages': [{{'role': 'user', 'content': 'Say hello briefly.'}}],
        'temperature': 0,
        'max_tokens': 16,
    }}
    status, _, body = request('/v1/chat/completions', payload=completion)
    result = json.loads(body)
    assert status == 200, result
    assert result['choices'][0]['message']['content']
    assert result['usage']['completion_tokens'] > 0

    streaming = {{
        'messages': [{{
            'role': 'user',
            'content': 'Count from one to five, one number per line.',
        }}],
        'temperature': 0,
        'max_tokens': 32,
        'stream': True,
        'stream_options': {{'include_usage': True}},
    }}
    status, headers, body = request('/v1/chat/completions',
                                    payload=streaming)
    assert status == 200
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
    assert engine_client.active_requests == 0
    assert engine_client.queued_requests == 0
    print('HTTP_SERVER_LIFECYCLE_PASSED')
finally:
    server.should_exit = True
    worker.join(timeout=180)
    assert not worker.is_alive(), 'HTTP server did not shut down'
    assert not server_error, server_error
"""
        python = self._hlapi_python(env_config, remote_config)
        cmd = ['bash', '-c', f'{python} -c {shlex.quote(script)}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(f"HTTP server lifecycle for {config.model_name}",
                           test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=900,
                                 logger=test_logger,
                                 env_vars=env_vars)

        output = result.get('output', '')
        if (not result['success']
                or 'HTTP_SERVER_LIFECYCLE_PASSED' not in output):
            pytest.fail(f"HTTP server lifecycle failed:\n{output}")

    def test_hlapi_generate(self, test_param: str, executable_files: Dict[str,
                                                                          str],
                            remote_config: Optional[RemoteConfig],
                            test_logger: logging.Logger,
                            env_config: EnvironmentConfig) -> None:
        """Test LLM.generate() from a checkpoint and shared build cache."""
        is_vlm = "-mnit" in test_param
        model_type = ModelType.VLM if is_vlm else ModelType.LLM
        config = TestConfig.from_param_string(test_param, model_type,
                                              TaskType.INFERENCE, env_config)

        test_logger.info("HLAPI generate: model=%s", config.model_name)

        prompt = "Please introduce the company NVIDIA and its CEO."
        max_tokens = 128

        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config)

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

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
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        test_logger.info("HLAPI generate with logprobs: model=%s",
                         config.model_name)

        prompt = "Count from 1 to 5."
        max_tokens = 32
        num_logprobs = 3
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config)

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

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
        config = TestConfig.from_param_string(test_param, model_type,
                                              TaskType.INFERENCE, env_config)
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
        llm_init = self._llm_init_script(config, env_config)
        script = f"""\
{setup}
import base64
{wav_setup}
wav_b64 = base64.b64encode(wav_bytes).decode()
from experimental.server import LLM, SamplingParams
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
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)

        test_logger.info("HLAPI logit_bias: model=%s", config.model_name)

        prompt = "Complete this sentence with one short word: NVIDIA makes"
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config)

        script = f"""\
{setup}
import json
import os
from experimental.server import LLM, SamplingParams

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

    def test_hlapi_video_generate(self, test_param: str,
                                  executable_files: Dict[str, str],
                                  remote_config: Optional[RemoteConfig],
                                  test_logger: logging.Logger,
                                  env_config: EnvironmentConfig) -> None:
        """HLAPI video path, generate + streaming: local clip -> decode+sample ->
        video ImageData -> per-model ViT runner, covering Qwen-VL, InternVL3,
        and Nemotron-Omni. ``VIDEO_TEST_CLIP`` overrides the synthetic PyAV
        clip encoded on the inference machine; ``VIDEO_TEST_NFRAMES`` must fit
        the visual engine's profile."""
        if "-mnit" not in test_param:
            pytest.skip("video HLAPI test requires a multimodal test_param")
        model_type = (ModelType.OMNI
                      if "-omni" in test_param.lower() else ModelType.VLM)
        config = TestConfig.from_param_string(test_param, model_type,
                                              TaskType.INFERENCE, env_config)
        test_clip = os.environ.get("VIDEO_TEST_CLIP", "")
        nframes = int(os.environ.get("VIDEO_TEST_NFRAMES", "8"))

        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config)
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
from experimental.server import LLM, SamplingParams
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
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)

        test_logger.info("HLAPI streaming: model=%s", config.model_name)

        prompt = "Count from 1 to 10."
        max_tokens = 128

        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config)

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

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
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        test_logger.info("HLAPI streaming with logprobs: model=%s",
                         config.model_name)

        prompt = "Count from 1 to 5."
        max_tokens = 32
        num_logprobs = 3
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config)

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

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
from experimental.server.config import ApiConfig
from experimental.server.api.protocol import ChatCompletionRequest
from experimental.server.api.serving_chat import OpenAIServingChat
from experimental.server.runtime.engine_client import EngineClient
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
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        prompt = "List three colors, separated by commas. End your list with '###'."
        stop = "###"
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config)

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

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
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        prompt = "List three colors, separated by commas. End your list with '###'."
        stop = "###"
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config)

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

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
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        prompt = "Write a long detailed essay about transformer neural networks."
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config)

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

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
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        prompt = "Write a long detailed essay about transformer neural networks."
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        llm_init = self._llm_init_script(config, env_config)

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

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
