# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import shlex
import subprocess
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from conftest import RemoteConfig


def run_command(cmd: List[str],
                remote_config: Optional[RemoteConfig],
                timeout: int = 300,
                logger=None,
                env_vars: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Unified command execution with comprehensive logging"""
    if remote_config is not None:
        remote_host = f"{remote_config.user}@{remote_config.host}"
        ssh_cmd = [
            'sshpass', '-p', remote_config.password, 'ssh', '-o',
            'StrictHostKeyChecking=no', remote_host
        ]

        # Build the command string with environment variables if provided
        env_prefix = ""
        if env_vars:
            env_parts = []
            for key, value in env_vars.items():
                env_parts.append(f"export {key}={shlex.quote(value)}")
            env_prefix = " && ".join(env_parts) + " && "

        if remote_config.remote_workspace:
            cmd_str = f"{env_prefix}cd {shlex.quote(remote_config.remote_workspace)} && {' '.join(shlex.quote(arg) for arg in cmd)}"
        else:
            cmd_str = f"{env_prefix}{' '.join(shlex.quote(arg) for arg in cmd)}"
        ssh_cmd.append(cmd_str)
        final_cmd = ssh_cmd
        cmd_display = f"[REMOTE] {' '.join(cmd)}"
    else:
        final_cmd = cmd
        cmd_display = ' '.join(cmd)

    if logger:
        logger.info(f"Running with timeout {timeout}s: {cmd_display}")

    # Prepare environment for local execution
    local_env = os.environ.copy() if env_vars else None
    if env_vars and not remote_config:
        local_env.update(env_vars)

    try:
        result = subprocess.run(final_cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                timeout=timeout,
                                env=local_env)

        success = result.returncode == 0

        if logger:
            if success:
                # Log successful command completion
                logger.info(
                    f"Command completed successfully (code {result.returncode}): {cmd_display}"
                )
            else:
                logger.error(
                    f"Command failed (code {result.returncode}): {cmd_display}"
                )
            # Log all output (stdout + stderr merged) in chronological order
            if result.stdout and result.stdout.strip():
                logger.info("Command output:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")

        return {
            'success': success,
            'returncode': result.returncode,
            'output': result.stdout or '',
            'error': '',
            'combined_output': result.stdout or ''
        }

    except subprocess.TimeoutExpired:
        error_msg = f"Command timed out after {timeout}s: {cmd_display}"
        if logger:
            logger.error(error_msg)
        return {
            'success': False,
            'returncode': -1,
            'output': '',
            'error': error_msg,
            'combined_output': error_msg
        }
    except Exception as e:
        error_msg = f"Command execution failed: {str(e)}"
        if logger:
            logger.error(f"{error_msg} - Command: {cmd_display}")
        return {
            'success': False,
            'returncode': -1,
            'output': '',
            'error': error_msg,
            'combined_output': error_msg
        }


def check_file_exists(filepath: str,
                      remote_config: Optional[RemoteConfig],
                      logger=None) -> bool:
    """Simple file existence check"""
    result = run_command(['test', '-f', filepath], remote_config, 10, logger)
    return result['success']


def get_file_content(filepath: str,
                     remote_config: Optional[RemoteConfig],
                     logger=None) -> str:
    """Get file content with proper error handling"""
    if not check_file_exists(filepath, remote_config, logger):
        raise FileNotFoundError(f"File not found: {filepath}")

    result = run_command(['cat', filepath], remote_config, 10, logger)
    if not result['success']:
        raise RuntimeError(
            f"Failed to read file content from {filepath}: {result.get('error', 'Unknown error')}"
        )

    content = result['output']
    if not content or not content.strip():
        raise ValueError(
            f"File {filepath} is empty or contains only whitespace")

    return content


def run_with_trt_env(cmd, remote_config, timeout, logger, env_config):
    """Run command with TensorRT LD_LIBRARY_PATH if available"""
    env_vars = None
    if env_config and env_config.trt_package_dir:
        # Unified approach for both local and remote execution
        trt_lib_path = f"{env_config.trt_package_dir}/lib"
        env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib_path}"}

    return run_command(cmd, remote_config, timeout, logger, env_vars)


@contextmanager
def timer_context(description: str, logger=None):
    """Timer context manager with proper logging"""
    if logger:
        logger.info(f"Starting: {description}")

    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        completion_msg = f"Completed: {description} ({elapsed:.2f}s)"
        if logger:
            logger.info(completion_msg)
