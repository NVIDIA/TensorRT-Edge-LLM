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
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add tests directory to path for imports
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

import pytest
import yaml


@dataclass
class RemoteConfig:
    """Configuration for remote test execution"""
    host: str
    user: str
    password: str
    remote_workspace: str


@dataclass
class EnvironmentConfig:
    """Configuration for test environment paths and directories"""
    llm_sdk_dir: str
    llm_models_dir: Optional[str]
    edgellm_data_dir: Optional[str]
    onnx_dir: str
    engine_dir: Optional[str]
    build_dir: str
    test_log_dir: str
    trt_package_dir: Optional[str]

    @classmethod
    def from_environment(cls) -> 'EnvironmentConfig':
        """Create EnvironmentConfig from required environment variables"""
        llm_sdk_dir = os.environ.get('LLM_SDK_DIR')
        if not llm_sdk_dir:
            raise ValueError(
                "LLM_SDK_DIR environment variable is required. "
                "Please set it to the root directory of the TensorRT Edge-LLM project."
            )

        onnx_dir = os.environ.get('ONNX_DIR')
        if not onnx_dir:
            raise ValueError("ONNX_DIR environment variable is required. "
                             "Please set it to the directory for ONNX models.")

        # LLM models directory with default fallback
        llm_models_dir = os.environ.get('LLM_MODELS_DIR')
        if not llm_models_dir:
            # Try default paths
            default_paths = [
                '/scratch.trt_llm_data/llm-models',
                '/home/scratch.trt_llm_data/llm-models'
            ]
            for path in default_paths:
                if os.path.exists(path):
                    llm_models_dir = path
                    break

        # EdgeLLM data directory with default fallback
        edgellm_data_dir = os.environ.get('EDGELLM_DATA_DIR')
        if not edgellm_data_dir:
            # Try default paths
            default_paths = [
                '/scratch.edge_llm_cache', '/home/edge_llm_cache',
                '/home/scratch.edge_llm_cache'
            ]
            for path in default_paths:
                if os.path.exists(path):
                    edgellm_data_dir = path
                    break

        # Optional directories - will be validated when needed
        engine_dir = os.environ.get('ENGINE_DIR')
        trt_package_dir = os.environ.get('TRT_PACKAGE_DIR')

        build_dir = os.environ.get('BUILD_DIR', 'build')
        test_log_dir = os.environ.get('TEST_LOG_DIR', 'logs')

        return cls(llm_sdk_dir=llm_sdk_dir,
                   llm_models_dir=llm_models_dir,
                   edgellm_data_dir=edgellm_data_dir,
                   onnx_dir=onnx_dir,
                   engine_dir=engine_dir,
                   build_dir=build_dir,
                   test_log_dir=test_log_dir,
                   trt_package_dir=trt_package_dir)

    def validate_for_export_tests(self):
        """Validate that required directories are set for export tests"""
        if not self.llm_models_dir:
            raise ValueError(
                "LLM_MODELS_DIR environment variable is required for export tests. "
                "Please set it to the directory containing LLM torch models, or ensure one of the default paths exists: "
                "/scratch.trt_llm_data/llm-models, /home/scratch.trt_llm_data/llm-models"
            )

    def validate_for_pipeline_tests(self):
        """Validate that required directories are set for pipeline tests"""
        if not self.engine_dir:
            raise ValueError(
                "ENGINE_DIR environment variable is required for pipeline tests. "
                "Please set it to the directory for TensorRT engines.")

    def validate_trt_package(self, remote_config=None, logger=None):
        """Validate that TensorRT package directory exists and contains required files"""
        if not self.trt_package_dir:
            raise ValueError("TRT_PACKAGE_DIR environment variable not set")

        from pytest_helpers import run_command

        result = run_command([
            'bash', '-c', f'test -f {self.trt_package_dir}/include/NvInfer.h'
        ], remote_config, 10, logger)
        if not result['success']:
            raise ValueError(
                f"Failed to get TensorRT package directory from {self.trt_package_dir}"
            )

        return self.trt_package_dir


@pytest.fixture(scope="session")
def env_config():
    """Load test environment config from required environment variables"""
    return EnvironmentConfig.from_environment()


@pytest.fixture(scope="session", autouse=True)
def setup_environment(env_config):
    """Setup environment and library paths"""
    os.makedirs(env_config.onnx_dir, exist_ok=True)
    if env_config.engine_dir:
        os.makedirs(env_config.engine_dir, exist_ok=True)
    os.makedirs(env_config.test_log_dir, exist_ok=True)


@pytest.fixture
def executable_files(env_config):
    """Paths to build executables"""
    build_dir = env_config.build_dir
    return {
        'llm_build': f"{build_dir}/examples/llm/llm_build",
        'llm_inference': f"{build_dir}/examples/llm/llm_inference",
        'visual_build': f"{build_dir}/examples/multimodal/visual_build",
        'unit_test': f"{build_dir}/unitTest"
    }


@pytest.fixture
def remote_config(request):
    """Get remote configuration from command line"""
    execution_mode_str = request.config.getoption("--execution-mode")

    if execution_mode_str == "remote":
        password = request.config.getoption("--remote-password")
        if not password:
            password = os.environ.get('BOARD_PASSWORD_NVKS')

        if not password:
            pytest.fail(
                "Remote password required for remote execution. "
                "Use --remote-password or set BOARD_PASSWORD_NVKS environment variable."
            )

        remote_user = request.config.getoption("--remote-user")
        if not remote_user:
            remote_user = os.environ.get('BOARD_USER')
        if not remote_user:
            pytest.fail(
                "Remote user required for remote execution. "
                "Use --remote-user or set BOARD_USER environment variable.")
        remote_host = request.config.getoption("--remote-host")
        if not remote_host:
            pytest.fail(
                "Remote host required for remote execution. "
                "Use --remote-host or set BOARD_HOST environment variable.")

        remote_workspace = request.config.getoption("--remote-workspace")
        if not remote_workspace:
            remote_workspace = os.environ.get('REMOTE_WORKSPACE')
        if not remote_workspace:
            pytest.fail(
                "Remote workspace required for remote execution. "
                "Use --remote-workspace or set REMOTE_WORKSPACE environment variable."
            )
        return RemoteConfig(host=remote_host,
                            user=remote_user,
                            password=password,
                            remote_workspace=remote_workspace)

    return None


@pytest.fixture(autouse=True)
def test_logger(request, env_config):
    """Create individual logger for each test"""
    test_name = request.node.name
    test_function = request.function.__name__

    if hasattr(request, 'param') or '[' in test_name:
        if '[' in test_name and ']' in test_name:
            param_part = test_name.split('[')[1].split(']')[0]
            param_clean = param_part.replace('/', '_').replace(':',
                                                               '_').replace(
                                                                   '-', '_')
            log_filename = f"{test_function}_{param_clean}.log"
        else:
            log_filename = f"{test_name}.log"
    else:
        log_filename = f"{test_function}.log"

    log_file = os.path.join(env_config.test_log_dir, log_filename)

    logger = logging.getLogger(f"test_{test_name}")
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    formatter_file = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter_file)

    logger.addHandler(file_handler)

    logger.info(f"Starting: {test_name}")

    request.node.test_logger = logger

    yield logger

    logger.info(f"Completed: {test_name}")

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption("--priority",
                     action="store",
                     help="Test priority level (l0, l1, etc.)")
    parser.addoption("--execution-mode",
                     action="store",
                     default="local",
                     choices=["local", "remote"],
                     help="Execution mode: local or remote")
    parser.addoption("--remote-host",
                     action="store",
                     help="Remote host for remote execution")
    parser.addoption("--remote-user",
                     action="store",
                     help="Remote user for remote execution")
    parser.addoption("--remote-password",
                     action="store",
                     help="Remote password for remote execution")
    parser.addoption("--remote-workspace",
                     action="store",
                     help="Remote workspace directory")


def pytest_runtest_makereport(item, call):
    """Enhanced test reporting with failure details"""
    if call.when == "call":
        test_logger = getattr(item, 'test_logger', None)

        if call.excinfo is not None and test_logger:
            test_logger.error(f"TEST FAILED: {item.name}")
            test_logger.error(
                f"Exception: {call.excinfo.type.__name__}: {str(call.excinfo.value)}"
            )

            if hasattr(call.excinfo, 'traceback') and call.excinfo.traceback:
                tb_entries = list(call.excinfo.traceback)
                for tb in reversed(tb_entries):
                    if 'tests/' in str(tb.path) and not str(
                            tb.path).endswith('conftest.py'):
                        test_logger.error(f"Location: {tb.path}:{tb.lineno}")
                        break


_test_config_cache = {}


def _get_test_list_file(priority):
    """Get test configuration with caching"""
    if priority not in _test_config_cache:
        if priority.endswith(('.yml', '.yaml')):
            config_file = priority
        else:
            config_file = f"tests/test_lists/{priority}.yml"

        try:
            with open(config_file, 'r') as f:
                _test_config_cache[priority] = yaml.safe_load(f)
        except FileNotFoundError:
            _test_config_cache[priority] = None
    return _test_config_cache[priority]


def pytest_generate_tests(metafunc):
    """Generate parameterized tests based on YAML configuration"""
    if "test_param" in metafunc.fixturenames:
        priority = metafunc.config.getoption("--priority", "l0")
        test_list_file = _get_test_list_file(priority)

        if not test_list_file:
            return

        test_cases = test_list_file.get('tests', [])
        current_test_name = metafunc.function.__name__

        relevant_tests = []
        current_test_file = metafunc.module.__name__

        for test_case in test_cases:
            if isinstance(test_case, str):
                if '::' in test_case:
                    test_file_path, test_function_part = test_case.split(
                        '::', 1)
                    test_module = test_file_path.replace('/', '.').replace(
                        '.py', '')

                    if (test_module == current_test_file
                            and current_test_name in test_function_part):
                        if '[' in test_case and ']' in test_case:
                            param = test_case.split('[')[1].split(']')[0]
                            relevant_tests.append(param)

        if relevant_tests:
            metafunc.parametrize("test_param", relevant_tests)


def pytest_collection_modifyitems(config, items):
    """Keep only test functions that are explicitly configured in the YAML file"""
    priority = config.getoption("--priority", "l0")
    test_list_file = _get_test_list_file(priority)

    if not test_list_file:
        return

    configured_tests = set()
    for test_case in test_list_file.get('tests', []):
        if isinstance(test_case, str):
            test_name = test_case.split('::')[-1]
            configured_tests.add(test_name)

    items[:] = [item for item in items if item.name in configured_tests]
