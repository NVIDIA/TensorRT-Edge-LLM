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
"""MiniMax-H3 text-to-video generation on engines built by `test_builder_pipeline.test_build`.

The build writes four engines and `multistage_config.json`. This test runs
`multistage_inference` on them with a request file and checks the video and
audio it writes. A request file loads the weights once for all its requests.
"""

import json
import os
import pathlib
import shutil

from .config import TaskType
from .test_builder_pipeline import _checkpoint_case, _run_on_target
from .utils import command_execution, minimax_h3_video


def test_video_generation(test_param, executable_files, remote_config,
                          test_logger, env_config):
    """Given a built checkpoint, generate every request and check its media."""
    config, _, engine = _checkpoint_case(test_param, env_config, remote_config,
                                         test_logger, TaskType.INFERENCE)
    source = config.get_test_case_file()
    with open(source, encoding="utf-8") as stream:
        requests = json.load(stream)["requests"]
    target_root = (remote_config.remote_workspace
                   if remote_config else env_config.llm_sdk_dir)
    executable = os.path.join(
        target_root, executable_files[minimax_h3_video.MULTISTAGE_INFERENCE])

    output = config.get_output_json_file()
    output_dir = os.path.splitext(output)[0] + "-media"
    # The remote fetch helper otherwise accepts files left by an earlier run.
    pathlib.Path(output).unlink(missing_ok=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    _run_on_target(["rm", "-rf", output, output_dir], remote_config,
                   env_config, test_logger, 60)
    _run_on_target(["mkdir", "-p", output_dir], remote_config, env_config,
                   test_logger, 60)

    _run_on_target(minimax_h3_video.validate_command(executable, engine),
                   remote_config, env_config, test_logger, 300)
    # Each request file path is relative to the SDK directory on the target.
    input_file = os.path.join(target_root, source)
    _run_on_target(
        minimax_h3_video.generate_command(executable, engine, input_file,
                                          output, output_dir), remote_config,
        env_config, test_logger, 7200)

    command_execution._sync_remote_output_file(output, remote_config,
                                               test_logger)
    with open(output, encoding="utf-8") as stream:
        predictions = json.load(stream)
    for path in minimax_h3_video.media_paths(predictions):
        command_execution._sync_remote_output_file(path, remote_config,
                                                   test_logger)

    problems = minimax_h3_video.output_mismatches(requests, predictions)
    assert not problems, "MiniMax-H3 output check failed:\n" + "\n".join(
        problems)
