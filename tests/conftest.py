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
# Repo root: test modules import in-tree packages (experimental.*) directly.
repo_root = tests_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

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
    onnx_dir: Optional[str]
    engine_dir: Optional[str]
    build_dir: str
    test_log_dir: str
    trt_package_dir: Optional[str]
    vlmevalkit_dir: Optional[str] = None
    vlmevalkit_data_dir: Optional[str] = None
    vlmevalkit_work_dir: Optional[str] = None

    @classmethod
    def from_environment(cls) -> 'EnvironmentConfig':
        """Create an EnvironmentConfig from the process environment."""
        llm_sdk_dir = os.environ.get('LLM_SDK_DIR')
        if not llm_sdk_dir:
            raise ValueError(
                "LLM_SDK_DIR environment variable is required. "
                "Please set it to the root directory of the TensorRT Edge-LLM project."
            )

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
        onnx_dir = os.environ.get('ONNX_DIR')
        engine_dir = os.environ.get('ENGINE_DIR')
        trt_package_dir = os.environ.get('TRT_PACKAGE_DIR')
        vlmevalkit_dir = os.environ.get('VLMEVALKIT_DIR')
        vlmevalkit_data_dir = os.environ.get('VLMEVALKIT_DATA_DIR')
        vlmevalkit_work_dir = os.environ.get('VLMEVALKIT_WORK_DIR')

        build_dir = os.environ.get('BUILD_DIR', 'build')
        test_log_dir = os.environ.get('TEST_LOG_DIR', 'logs')

        return cls(llm_sdk_dir=llm_sdk_dir,
                   llm_models_dir=llm_models_dir,
                   edgellm_data_dir=edgellm_data_dir,
                   onnx_dir=onnx_dir,
                   engine_dir=engine_dir,
                   build_dir=build_dir,
                   test_log_dir=test_log_dir,
                   trt_package_dir=trt_package_dir,
                   vlmevalkit_dir=vlmevalkit_dir,
                   vlmevalkit_data_dir=vlmevalkit_data_dir,
                   vlmevalkit_work_dir=vlmevalkit_work_dir)

    @staticmethod
    def _require_setting(value: Optional[str], variable: str,
                         purpose: str) -> None:
        if not value:
            raise ValueError(
                f"{variable} environment variable is required for {purpose}.")

    def validate_for_export_tests(self) -> None:
        """Validate the checkpoint export environment."""
        self._require_setting(self.llm_models_dir, 'LLM_MODELS_DIR',
                              'export tests')
        self._require_setting(self.onnx_dir, 'ONNX_DIR', 'export tests')

    def validate_for_pipeline_tests(self) -> None:
        """Validate an ONNX-based build or runtime environment."""
        self._require_setting(self.onnx_dir, 'ONNX_DIR', 'pipeline tests')
        self._require_setting(self.engine_dir, 'ENGINE_DIR', 'pipeline tests')

    def validate_for_checkpoint_builder_tests(self) -> None:
        """Validate a direct checkpoint-to-engine test environment."""
        self._require_setting(self.llm_models_dir, 'LLM_MODELS_DIR',
                              'checkpoint builder tests')
        self._require_setting(self.engine_dir, 'ENGINE_DIR',
                              'checkpoint builder tests')

    def validate_for_vlmevalkit_tests(self):
        """Validate that required directories are set for VLMEvalKit tests."""
        if not self.vlmevalkit_dir:
            raise ValueError(
                "VLMEVALKIT_DIR environment variable is required for VLMEvalKit tests. "
                "Please set it to a VLMEvalKit checkout.")
        if not os.path.isdir(self.vlmevalkit_dir):
            raise ValueError(
                f"VLMEVALKIT_DIR does not exist: {self.vlmevalkit_dir}")

        data_root = self.vlmevalkit_data_dir or self.edgellm_data_dir
        if not data_root:
            raise ValueError(
                "VLMEVALKIT_DATA_DIR or EDGELLM_DATA_DIR is required for VLMEvalKit tests. "
                "Please set it to the directory containing updated_datasets.")
        if not os.path.isdir(data_root):
            raise ValueError(
                f"VLMEvalKit data directory does not exist: {data_root}")

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
    """Create only the directories declared by the selected test job."""
    if env_config.onnx_dir:
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
        'llm_bench': f"{build_dir}/examples/llm/llm_bench",
        'visual_build': f"{build_dir}/examples/multimodal/visual_build",
        'audio_build': f"{build_dir}/examples/multimodal/audio_build",
        'action_build': f"{build_dir}/examples/multimodal/action_build",
        'action_inference':
        f"{build_dir}/examples/multimodal/action_inference",
        'qwen3_tts_inference':
        f"{build_dir}/examples/omni/qwen3_tts_inference",
        # The unit tests are several ctest-registered executables; this points
        # at the directory ctest is invoked from, not a single binary.
        'unit_test': f"{build_dir}"
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


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Replay library size reports so they reach the CI job log."""
    del exitstatus, config
    from pytest_helpers import library_size_reports

    for report in library_size_reports:
        terminalreporter.write_line(report)


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption("--priority",
                     action="store",
                     default=None,
                     help="Test priority level (l0, l1, etc.)")
    parser.addoption("--test-param",
                     action="append",
                     default=None,
                     help="Run a single test case directly without YAML. "
                     "Can be specified multiple times for multiple cases.")
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

_MODEL_QUANTIZATION_TEST_NAMES = (
    "test_llm_model_quantization",
    "test_tts_model_quantization",
    "test_vlm_model_quantization",
    "test_asr_model_quantization",
    "test_omni_model_quantization",
)


def _test_item_function_name(item):
    return getattr(item, "originalname", item.name.split("[", 1)[0])


def _preferred_model_quantization_test_name(test_param: str) -> str:
    lower = test_param.lower()
    if "omni" in lower:
        return "test_omni_model_quantization"
    if "asr" in lower:
        return "test_asr_model_quantization"
    if "tts" in lower:
        return "test_tts_model_quantization"
    vlm_hints = ("-vl-", "internvl", "multimodal", "cosmos", "vitfp8",
                 "qwen3.5-", "qwen3.6-", "qwen3.8-")
    if any(hint in lower for hint in vlm_hints):
        return "test_vlm_model_quantization"
    return "test_llm_model_quantization"


def _filter_direct_model_quantization_items(items):
    groups = {}
    for item in items:
        test_name = _test_item_function_name(item)
        if test_name not in _MODEL_QUANTIZATION_TEST_NAMES:
            continue
        callspec = getattr(item, "callspec", None)
        if not callspec or "test_param" not in callspec.params:
            continue
        groups.setdefault(callspec.params["test_param"], []).append(item)

    keep_ids = {id(item) for item in items}
    for test_param, group in groups.items():
        if len(group) <= 1:
            continue
        preferred_name = _preferred_model_quantization_test_name(test_param)
        preferred_items = [
            item for item in group
            if _test_item_function_name(item) == preferred_name
        ]
        selected = preferred_items[0] if preferred_items else group[0]
        for item in group:
            if item is not selected:
                keep_ids.discard(id(item))

    items[:] = [item for item in items if id(item) in keep_ids]


def _get_test_list_file(priority):
    """Get test configuration with caching"""
    if priority is None:
        return None
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
    """Generate parameterized tests from --test-param CLI args or YAML config"""
    if "test_param" not in metafunc.fixturenames:
        return

    # --test-param takes precedence: inject directly, skip YAML entirely
    direct_params = metafunc.config.getoption("--test-param")
    if direct_params:
        metafunc.parametrize("test_param", direct_params)
        return

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
                test_file_path, test_function_part = test_case.split('::', 1)
                test_module = test_file_path.replace('/',
                                                     '.').replace('.py', '')

                if test_module != current_test_file:
                    continue
                # Match class name when both YAML entry and test belong to a class
                if '::' in test_function_part:
                    yaml_class, yaml_func = test_function_part.split('::', 1)
                    if metafunc.cls and metafunc.cls.__name__ != yaml_class:
                        continue
                    # Exact function-name match: substring matching leaked
                    # params across prefix-sharing tests (test_hlapi_generate
                    # vs test_hlapi_generate_with_audio).
                    if yaml_func.split('[')[0] != current_test_name:
                        continue
                elif test_function_part.split('[')[0] != current_test_name:
                    continue

                if '[' in test_case and ']' in test_case:
                    param = test_case.split('[')[1].split(']')[0]
                    relevant_tests.append(param)

    if relevant_tests:
        metafunc.parametrize("test_param", relevant_tests)


def pytest_collection_modifyitems(config, items):
    """Filter and reorder tests to strictly follow YAML declaration order"""
    # --test-param bypasses YAML filtering entirely
    if config.getoption("--test-param"):
        _filter_direct_model_quantization_items(items)
        return

    priority = config.getoption("--priority", "l0")
    test_list_file = _get_test_list_file(priority)

    if not test_list_file:
        return

    item_by_name = {}
    item_by_base = {}
    for item in items:
        item_by_name.setdefault(item.name, []).append(item)
        if '[' in item.name:
            base = item.name.split('[')[0]
            item_by_base.setdefault(base, []).append(item)

    ordered = []
    seen = set()
    for test_case in test_list_file.get('tests', []):
        if not isinstance(test_case, str):
            continue
        if '::' not in test_case:
            test_path = Path(test_case)
            if not test_path.is_absolute():
                test_path = repo_root / test_path
            test_path = test_path.resolve()
            for item in items:
                if (Path(str(item.path)).resolve() == test_path
                        and id(item) not in seen):
                    ordered.append(item)
                    seen.add(id(item))
            continue

        test_name = test_case.split('::')[-1]

        for item in item_by_name.get(test_name, []):
            if id(item) not in seen:
                ordered.append(item)
                seen.add(id(item))

        if '[' not in test_name:
            for item in item_by_base.get(test_name, []):
                if id(item) not in seen:
                    ordered.append(item)
                    seen.add(id(item))

    items[:] = ordered
