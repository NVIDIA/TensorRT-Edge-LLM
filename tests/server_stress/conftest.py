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
"""Opt-in pressure tests, isolated from model-export fixture requirements."""

import asyncio
import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from .support import ServerProcess, Settings, wait_ready, write_json


def pytest_addoption(parser):
    group = parser.getgroup('server stress')
    group.addoption('--server-stress-config',
                    help='JSON workload and dedicated server configuration')
    group.addoption('--server-stress-output',
                    default='logs/server-stress',
                    help='Parent directory for unique run artifacts')


def pytest_configure(config):
    if config.getoption('--server-stress-config') and (
            getattr(config.option, 'numprocesses', 0)
            or hasattr(config, 'workerinput')):
        raise pytest.UsageError(
            'Run server stress tests serially; use one dedicated server per invocation'
        )


def pytest_generate_tests(metafunc):
    path = metafunc.config.getoption('--server-stress-config')
    settings = Settings.read(path) if path else Settings()
    if 'concurrency' in metafunc.fixturenames:
        metafunc.parametrize('concurrency', settings.concurrency)
    if 'load_attempt' in metafunc.fixturenames:
        metafunc.parametrize('load_attempt', range(settings.load_repeats))


@pytest.fixture(scope='session', autouse=True)
def setup_environment():
    """HTTP pressure tests own their artifacts and require no export directories."""


@pytest.fixture(autouse=True)
def test_logger():
    """Use pytest logging for the HTTP suite."""
    return logging.getLogger('server_stress')


@pytest.fixture(scope='session')
def stress_settings(request):
    path = request.config.getoption('--server-stress-config')
    if not path:
        pytest.skip('Enable live pressure tests with --server-stress-config')
    return Settings.read(path)


@pytest.fixture(scope='session')
def run_directory(request, stress_settings):
    parent = Path(request.config.getoption('--server-stress-output')).resolve()
    directory = parent / datetime.now(
        timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory.mkdir(parents=True)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'],
                              capture_output=True,
                              text=True,
                              check=False)
    write_json(
        directory / 'environment.json', {
            'platform': platform.platform(),
            'python': platform.python_version(),
            'hostname': platform.node(),
            'test_revision': revision.stdout.strip(),
            'configuration': {
                k: v
                for k, v in vars(stress_settings).items() if k != 'command'
            },
            'command_recorded': False,
        })
    request.config._server_stress_directory = directory
    return directory


@pytest.fixture
def case_directory(request, run_directory):
    directory = run_directory / request.node.name
    directory.mkdir()
    request.node._stress_directory = directory
    return directory


@pytest.fixture
def server(stress_settings, case_directory):
    owned = ServerProcess(stress_settings,
                          case_directory) if stress_settings.command else None
    try:
        if owned:
            owned.start()
        asyncio.run(wait_ready(stress_settings, owned))
        yield owned
    finally:
        if owned:
            owned.close()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    directory = getattr(item, '_stress_directory', None)
    if directory:
        with (directory / 'pytest.jsonl').open('a') as stream:
            stream.write(
                json.
                dumps({
                    'phase': report.when,
                    'outcome': report.outcome,
                    'duration_s': report.duration,
                    'detail': str(report.longrepr) if report.failed else None
                }) + '\n')


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    directory = getattr(config, '_server_stress_directory', None)
    if directory:
        terminalreporter.write_sep('=',
                                   f'Server stress artifacts: {directory}')
        summaries = {}
        for path in sorted(directory.glob('*/pytest.jsonl')):
            data = {
                'pytest':
                [json.loads(line) for line in path.read_text().splitlines()]
            }
            for relative, key in (('summary.json',
                                   None), ('benchmark/summary.json', None),
                                  ('exit.json', 'exit'), ('drain.json',
                                                          'drain')):
                artifact = path.parent / relative
                if artifact.exists():
                    value = json.loads(artifact.read_text())
                    if key:
                        data[key] = value
                    else:
                        data.update(value)
            summaries[path.parent.name] = data
            terminalreporter.write_line(
                f'{path.parent.name}: {data.get("outcomes")} '
                f'rps={data.get("success_rps")} p99={data.get("latency_s", {}).get("p99")}s'
            )
        write_json(directory / 'summary.json', {
            'pytest_exitstatus': int(exitstatus),
            'cases': summaries
        })
