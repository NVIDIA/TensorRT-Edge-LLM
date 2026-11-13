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
"""Run the media IPC server example on an L0 case's engines."""

import json
import logging
import os
import re
import shlex
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import run_command, timer_context

from .config import ModelType, TaskType, TestConfig

_ITERATIONS = 2
_MEDIA_DIR = os.path.join("test_media", "ipc_example")
_IMAGE = "giant_panda_1000x1000.nv12:1000x1000:NV12_PL:0"
_AUDIO = "speech_f32le_16000hz_mono.raw"
_PROMPTS = {
    ModelType.VLM: "Please describe the image.",
    ModelType.ASR: "",
    ModelType.OMNI: "Describe the image, then transcribe the speech.",
}
_IMAGE_WORDS = ("panda", )
# From the asr_basic reference transcript of the same clip.
_AUDIO_WORDS = ("returned", "tents")
_RESULT_PREFIX = "MEDIA_IPC_RESULT="
_PERF_LINE = re.compile(r"^perf: requestId=(\d+) .*\bosl=(\d+)[ \t]*$",
                        re.MULTILINE)

# Runs on the target. The server logs to a file, so a full pipe never stalls it.
_DRIVER = """\
import json, signal, subprocess, tempfile, time

result = {{"ready": False}}
with tempfile.TemporaryFile() as server_log:
    server = subprocess.Popen({server!r}, stdout=server_log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + {ready_timeout}
        while server.poll() is None and time.monotonic() < deadline:
            server_log.seek(0)
            if b"listening on " in server_log.read():
                result["ready"] = True
                break
            time.sleep(2)
        if result["ready"]:
            try:
                client = subprocess.run({client!r}, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout={client_timeout})
                result["client_returncode"] = client.returncode
                result["client_output"] = client.stdout.decode(errors="replace")
            except subprocess.TimeoutExpired as error:
                result["client_returncode"] = "timeout"
                result["client_output"] = (error.stdout or b"").decode(errors="replace")
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout={stop_timeout})
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
                result["server_killed"] = True
        result["server_returncode"] = server.returncode
        server_log.seek(0)
        result["server_log"] = server_log.read()[-10000:].decode(errors="replace")
print({prefix!r} + json.dumps(result))
"""
_READY_TIMEOUT = 600
_CLIENT_TIMEOUT = 600
_STOP_TIMEOUT = 60


def media_ipc_commands(
        model_type: ModelType, executables: Dict[str, str],
        llm_engine_dir: str, multimodal_engine_dir: str,
        media_dir: str) -> Tuple[List[str], List[str], Tuple[str, ...]]:
    """Server and client command lines, and the words every answer must contain."""
    server = [
        executables['media_ipc_server'], '--llm-engine-dir', llm_engine_dir,
        '--multimodal-engine-dir', multimodal_engine_dir
    ]
    client = [
        executables['media_ipc_client'], '--iterations',
        str(_ITERATIONS), '--topK', '1', '--systemPrompt', '', '--prompt',
        _PROMPTS[model_type]
    ]
    words: Tuple[str, ...] = ()
    if model_type in (ModelType.VLM, ModelType.OMNI):
        client += ['--image', os.path.join(media_dir, _IMAGE)]
        words += _IMAGE_WORDS
    if model_type in (ModelType.ASR, ModelType.OMNI):
        client += ['--audio', os.path.join(media_dir, _AUDIO)]
        words += _AUDIO_WORDS
    return server, client, words


def media_ipc_failures(client_output: str, words: Sequence[str],
                       iterations: int) -> List[str]:
    """Why the client output does not show every request answered about the media."""
    requests = list(_PERF_LINE.finditer(client_output))
    failures = []
    if len(requests) != iterations:
        failures.append(f"{len(requests)} of {iterations} requests finished")
    start = 0
    for request in requests:
        answer = client_output[start:request.start()].lower()
        start = request.end()
        request_id, output_tokens = request.groups()
        if int(output_tokens) == 0:
            failures.append(f"request {request_id} generated no tokens")
        missing = [word for word in words if word not in answer]
        if missing:
            failures.append(f"request {request_id} answer lacks {missing}")
    return failures


def _multimodal_engine_dir(config: TestConfig) -> str:
    return {
        ModelType.VLM: config.get_visual_engine_dir,
        ModelType.ASR: config.get_audio_engine_dir,
        ModelType.OMNI: config.get_multimodal_engine_dir,
    }[config.model_type]()


def test_media_ipc_server(test_param: str, executable_files: Dict[str, str],
                          remote_config: Optional[RemoteConfig],
                          test_logger: logging.Logger,
                          env_config: EnvironmentConfig) -> None:
    """The media IPC server answers every client request about the shared image
    or audio, then exits cleanly on SIGTERM."""
    config = TestConfig.from_param_string(test_param,
                                          ModelType(test_param.model_type),
                                          TaskType.INFERENCE, env_config)
    if not env_config.edgellm_data_dir:
        pytest.fail("EDGELLM_DATA_DIR must hold test_media/ipc_example")
    server, client, words = media_ipc_commands(
        config.model_type, executable_files, config.get_llm_engine_dir(),
        _multimodal_engine_dir(config),
        os.path.join(env_config.edgellm_data_dir, _MEDIA_DIR))
    script = _DRIVER.format(server=server,
                            client=client,
                            ready_timeout=_READY_TIMEOUT,
                            client_timeout=_CLIENT_TIMEOUT,
                            stop_timeout=_STOP_TIMEOUT,
                            prefix=_RESULT_PREFIX)
    library_path = (f"{os.path.join(env_config.build_dir, 'lib')}"
                    ":$LD_LIBRARY_PATH")
    if env_config.trt_package_dir:
        library_path += f":{env_config.trt_package_dir}/lib"

    with timer_context(f"Media IPC server for {config.model_name}",
                       test_logger):
        result = run_command(
            cmd=['bash', '-c', f'python3 -c {shlex.quote(script)}'],
            remote_config=remote_config,
            timeout=_READY_TIMEOUT + _CLIENT_TIMEOUT + _STOP_TIMEOUT + 120,
            logger=test_logger,
            env_vars={'LD_LIBRARY_PATH': library_path})

    output = result.get('output', '')
    lines = [
        line for line in output.splitlines() if line.startswith(_RESULT_PREFIX)
    ]
    if not result['success'] or not lines:
        pytest.fail(f"Media IPC driver failed:\n{output}"
                    f"{result.get('error', '')}")
    run = json.loads(lines[-1][len(_RESULT_PREFIX):])
    client_output = run.get('client_output', '')
    failures = []
    if not run['ready']:
        failures.append("server never logged 'listening on'")
    else:
        if run.get('client_returncode') != 0:
            failures.append(f"client exit code {run.get('client_returncode')}")
        failures += media_ipc_failures(client_output, words, _ITERATIONS)
    if run.get('server_killed') or run['server_returncode'] != 0:
        failures.append(
            f"server exit code {run['server_returncode']} after SIGTERM")
    if failures:
        pytest.fail("\n".join(failures) +
                    f"\n--- client output ---\n{client_output}"
                    f"\n--- server log ---\n{run['server_log']}")
