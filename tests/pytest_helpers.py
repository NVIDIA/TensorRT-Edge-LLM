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
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from conftest import RemoteConfig

_SENSITIVE_ENV_KEY_PARTS = ("TOKEN", "PASSWORD", "SECRET", "API_KEY",
                            "SSHPASS")
_SHARED_STORAGE_UMASK = "0000"


def _quote_env_assignment(key: str, value: str, mask_sensitive: bool) -> str:
    display_value = "***" if mask_sensitive and any(
        part in key.upper() for part in _SENSITIVE_ENV_KEY_PARTS) else value
    return f"{key}={shlex.quote(str(display_value))}"


def _format_command_for_display(
    cmd: List[str],
    env_vars: Optional[Dict[str, str]] = None,
    mask_sensitive_env: bool = True,
) -> str:
    cmd_display = ' '.join(shlex.quote(arg) for arg in cmd)
    if not env_vars:
        return cmd_display

    env_display = ' '.join(
        _quote_env_assignment(key, value, mask_sensitive_env)
        for key, value in env_vars.items())
    return f"{env_display} {cmd_display}"


# Library size reports collected during the run. Pytest's fd capture hides
# everything a passing test prints, so conftest replays these in the terminal
# summary, which is where a CI job log can show them. This lives here rather
# than in conftest.py because pytest registers conftest as its own plugin
# module instance -- a test importing `conftest` gets a different object and
# would append to a list the hook never sees.
library_size_reports: List[str] = []


def record_library_size_report(text: str) -> None:
    """Queue a library size report for the end-of-run terminal summary."""
    if text:
        library_size_reports.append(text)


def _tee_line_to_process_stdout(text: str) -> None:
    """Send one line to the real OS stdout (fd 1) so the outer edge-llm-qa
    runner (run_cmd_direct) sees it. Pytest's output capture replaces sys.stdout;
    writing fd 1 still reaches the parent pipe.
    """
    if not text:
        return
    line = text if text.endswith("\n") else text + "\n"
    data = line.encode("utf-8", errors="replace")
    try:
        os.write(1, data)
    except OSError:
        try:
            sys.__stdout__.write(line)
        except (OSError, TypeError, AttributeError):
            pass


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
            'StrictHostKeyChecking=no', '-o', 'ServerAliveInterval=30', '-o',
            'ServerAliveCountMax=10', '-o', 'TCPKeepAlive=yes', remote_host
        ]

        remote_steps = [f"umask {_SHARED_STORAGE_UMASK}"]
        if env_vars:
            for key, value in env_vars.items():
                remote_steps.append(f"export {key}={shlex.quote(value)}")

        base_cmd_display = _format_command_for_display(cmd, env_vars)
        if remote_config.remote_workspace:
            remote_steps.append(
                f"cd {shlex.quote(remote_config.remote_workspace)}")
        remote_steps.append(' '.join(shlex.quote(arg) for arg in cmd))

        display_steps = [f"umask {_SHARED_STORAGE_UMASK}"]
        if remote_config.remote_workspace:
            display_steps.append(
                f"cd {shlex.quote(remote_config.remote_workspace)}")
        display_steps.append(base_cmd_display)

        cmd_str = " && ".join(remote_steps)
        cmd_display = f"[REMOTE] {' && '.join(display_steps)}"
        ssh_cmd.append(cmd_str)
        final_cmd = ssh_cmd
    else:
        final_cmd = cmd
        cmd_display = _format_command_for_display(cmd, env_vars)

    if logger:
        logger.info(f"Running with timeout {timeout}s: {cmd_display}")

    # Prepare environment for local execution
    local_env = os.environ.copy() if env_vars else None
    if env_vars and not remote_config:
        local_env.update(env_vars)

    try:
        process = subprocess.Popen(final_cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   bufsize=1,
                                   env=local_env)

        output_lines = []
        # Real-time line-by-line output
        for line in iter(process.stdout.readline, ''):
            line = line.rstrip('\n')
            output_lines.append(line)
            if logger:
                logger.info(f"  {line}")
            _tee_line_to_process_stdout(line)

        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        success = returncode == 0
        combined_output = '\n'.join(output_lines)

        if logger:
            if success:
                logger.info(
                    f"Command completed successfully (code {returncode}): {cmd_display}"
                )
            else:
                logger.error(
                    f"Command failed (code {returncode}): {cmd_display}")

        return {
            'success': success,
            'returncode': returncode,
            'output': combined_output,
            'error': '',
            'combined_output': combined_output
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
