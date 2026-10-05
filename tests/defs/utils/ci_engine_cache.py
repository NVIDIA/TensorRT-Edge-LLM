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
"""Fingerprint and reuse the latest complete TensorRT engine bundle."""

import contextlib
import ctypes
import enum
import hashlib
import json
import mmap
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import typing
import uuid

ci_cache_event = None

_COLD_ENVIRONMENT = (
    "EDGE_LLM_CACHE_FORCE_COLD",
    "EDGE_LLM_ENGINE_CACHE_FORCE_COLD",
    "EDGE_LLM_STABILITY_COLD",
    "CI_STABILITY_RUN",
)
_ENGINE_KIND_PREFIX = "trt-engine-"
_CACHE_SCHEMA_VERSION = "edgellm-ci-engine-cache/v1"
_BUNDLE_MANIFEST = ".engine_meta"
_ENGINE_ARGUMENTS = ("--engineDir=", "--onnxDir=")
_ONNX_METADATA = {".export_meta"}
_LDD_PATH = re.compile(r"(?:=>\s+)?(/[^\s(]+)")
_REMOTE_DESCRIPTOR_MARKER = "EDGE_LLM_REMOTE_ENGINE_DESCRIPTOR="
_REMOTE_HELPER = "tests/defs/utils/ci_engine_cache.py"
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


class CacheError(RuntimeError):
    """An invalid or unavailable cache operation."""


class CacheMode(enum.Enum):
    """Supported cache read and write policies."""

    OFF = "off"
    READ = "read"
    READ_WRITE = "read-write"
    REFRESH = "refresh"

    @property
    def reads(self):
        return self in (CacheMode.READ, CacheMode.READ_WRITE)

    @property
    def writes(self):
        return self in (CacheMode.READ_WRITE, CacheMode.REFRESH)


def _content_key(value) -> str:
    encoded = json.dumps(value,
                         sort_keys=True,
                         separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _truthy(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


def _log(logger, level: str, message: str, *args) -> None:
    if logger:
        getattr(logger, level)(message, *args)


def _cache_event(logger,
                 artifact: str,
                 operation: str,
                 started: float,
                 decision: str,
                 key: str = "",
                 result=None,
                 detail: str = "") -> None:
    payload = {
        "artifact": artifact,
        "operation": operation,
        "decision": decision,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }
    if key:
        payload["key"] = key
    if result is not None:
        payload.update({
            name: result[name]
            for name in ("namespace", "files", "bytes") if name in result
        })
        if not detail:
            detail = result.get("detail", "")
    if detail:
        payload["detail"] = detail
    _log(logger, "info", "EDGE_LLM_CACHE_RESULT=%s",
         json.dumps(payload, sort_keys=True))
    if ci_cache_event is None or operation == "key":
        return

    telemetry_decision = decision
    reason = detail or decision
    rejected = {
        "corruption", "descriptor-change", "deserialization-failure",
        "invalid-inventory", "invalid-manifest", "not-found",
        "restore-cost-gate", "schema-mismatch", "unrepresentable-input"
    }
    if decision == "miss" and reason in rejected:
        telemetry_decision = reason
    elif decision == "error":
        if "quota" in detail:
            telemetry_decision = reason = "oversized"
        else:
            reason = "error"

    event = {
        "artifact": "engine",
        "operation": operation,
        "decision": telemetry_decision,
        "reason": reason,
        "elapsed_seconds": payload["elapsed_seconds"],
    }
    if artifact:
        event["lane"] = artifact.lower()
    if key:
        event["key_prefix"] = key[:12]
    if result is not None:
        if result.get("namespace"):
            event["namespace"] = result["namespace"].lower()
        if result.get("files") is not None:
            event["files"] = result["files"]
        if result.get("bytes") is not None:
            name = ("bytes_read"
                    if operation == "restore" else "bytes_written")
            event[name] = result["bytes"]
    if operation == "restore" and telemetry_decision == "hit":
        event["materialization"] = "copy"
        event["build_skipped"] = True
    ci_cache_event.record(event)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file_descriptor(
        path: pathlib.Path) -> typing.Dict[str, typing.Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise CacheError(
            "engine cache input is not a regular file: {}".format(path))
    return {
        "name": path.name,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _tree_descriptor(path_text: str) -> typing.Dict[str, typing.Any]:
    root = pathlib.Path(path_text)
    if root.is_symlink() or not root.is_dir():
        raise CacheError(
            "ONNX input is not a regular directory: {}".format(root))
    files = []
    directories = []
    for current, dir_names, file_names in os.walk(str(root),
                                                  followlinks=False):
        current_path = pathlib.Path(current)
        dir_names.sort()
        file_names.sort()
        for name in dir_names:
            candidate = current_path / name
            if candidate.is_symlink() or not candidate.is_dir():
                raise CacheError(
                    "unsafe ONNX directory entry: {}".format(candidate))
            directories.append(candidate.relative_to(root).as_posix())
        for name in file_names:
            if (current_path == root and name in _ONNX_METADATA):
                continue
            candidate = current_path / name
            if candidate.is_symlink() or not candidate.is_file():
                raise CacheError(
                    "unsafe ONNX file entry: {}".format(candidate))
            files.append({
                "path": candidate.relative_to(root).as_posix(),
                "bytes": candidate.stat().st_size,
                "sha256": _sha256(candidate),
            })
    return {"directories": directories, "files": files}


def _argument_value(command: typing.Sequence[str],
                    prefix: str) -> typing.Optional[str]:
    return next((value[len(prefix):]
                 for value in command if value.startswith(prefix)), None)


def _normalized_commands(
    commands: typing.Sequence[typing.Tuple[typing.List[str], int]]
) -> typing.List[typing.Dict[str, typing.Any]]:
    normalized = []
    for command, timeout in commands:
        arguments = []
        for value in command[1:]:
            replacement = value
            for prefix in _ENGINE_ARGUMENTS:
                if value.startswith(prefix):
                    replacement = prefix + "<path>"
                    break
            arguments.append(replacement)
        normalized.append({
            "builder": pathlib.Path(command[0]).name,
            "arguments": arguments,
            "timeout": timeout,
        })
    return normalized


def _linked_libraries(executable: pathlib.Path) -> typing.List[pathlib.Path]:
    result = subprocess.run(["ldd", str(executable)],
                            capture_output=True,
                            check=False,
                            text=True,
                            timeout=30)
    if result.returncode != 0:
        raise CacheError("ldd failed for {}: {}".format(
            executable, (result.stderr or result.stdout).strip()))
    paths = []
    for line in result.stdout.splitlines():
        match = _LDD_PATH.search(line)
        if match:
            candidate = pathlib.Path(match.group(1)).resolve(strict=True)
            if candidate.is_file():
                paths.append(candidate)
    return paths


def _plugin_libraries(executable: pathlib.Path) -> typing.List[pathlib.Path]:
    paths = []
    configured = os.environ.get("EDGELLM_PLUGIN_PATH", "").strip()
    if configured:
        plugin = pathlib.Path(configured).resolve(strict=True)
        if not plugin.is_file():
            raise CacheError(
                "configured Edge-LLM plugin is not a regular file: {}".format(
                    plugin))
        paths.append(plugin)
    current = executable.parent
    for _ in range(6):
        paths.extend(current.glob("libNvInfer_edgellm_plugin.so*"))
        if current == current.parent:
            break
        current = current.parent
    return [path.resolve(strict=True) for path in paths if path.is_file()]


def _builder_descriptor(
    commands: typing.Sequence[typing.Tuple[typing.List[str], int]]
) -> typing.Tuple[typing.List[typing.Dict[str, typing.Any]],
                  typing.List[pathlib.Path]]:
    executables = sorted({
        pathlib.Path(command[0]).resolve(strict=True)
        for command, _ in commands
    })
    inputs = set(executables)
    plugins = set()
    for executable in executables:
        if not executable.is_file():
            raise CacheError(
                "builder executable is missing: {}".format(executable))
        inputs.update(_linked_libraries(executable))
        discovered_plugins = _plugin_libraries(executable)
        inputs.update(discovered_plugins)
        plugins.update(discovered_plugins)
    descriptors = []
    for path in sorted(inputs):
        descriptor = _regular_file_descriptor(path)
        descriptor["role"] = "builder" if path in executables else "library"
        descriptors.append(descriptor)
    return descriptors, sorted(plugins)


def _nvidia_smi_gpu_descriptor() -> typing.Dict[str, str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    selected_device = visible_devices.partition(",")[0].strip() or "0"
    query = subprocess.run([
        "nvidia-smi", "-i", selected_device,
        "--query-gpu=name,compute_cap,driver_version",
        "--format=csv,noheader,nounits"
    ],
                           capture_output=True,
                           check=False,
                           text=True,
                           timeout=30)
    if query.returncode != 0 or not query.stdout.strip():
        raise CacheError(
            "GPU identity is unavailable; refusing engine cache reuse")
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise CacheError(
            "selected GPU identity is ambiguous; refusing engine cache reuse")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3 or any(not field for field in fields):
        raise CacheError(
            "selected GPU identity is malformed; refusing engine cache reuse")
    return {
        "name": fields[0],
        "compute_capability": fields[1],
        "driver_version": fields[2],
    }


def _cuda_driver_gpu_descriptor() -> typing.Dict[str, str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    selected = visible_devices.partition(",")[0].strip()
    ordinal = int(selected) if selected.isdecimal() else 0
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGetName.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_int
    ]
    cuda.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
    ]
    cuda.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    device = ctypes.c_int()
    name = ctypes.create_string_buffer(256)
    major = ctypes.c_int()
    minor = ctypes.c_int()
    driver = ctypes.c_int()
    calls = (
        cuda.cuInit(0),
        cuda.cuDeviceGet(ctypes.byref(device), ordinal),
        cuda.cuDeviceGetName(name, len(name), device),
        cuda.cuDeviceGetAttribute(
            ctypes.byref(major), _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device),
        cuda.cuDeviceGetAttribute(
            ctypes.byref(minor), _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device),
        cuda.cuDriverGetVersion(ctypes.byref(driver)),
    )
    if any(result != 0 for result in calls):
        raise CacheError("CUDA driver could not identify the selected GPU")
    driver_identity = str(driver.value)
    driver_path = pathlib.Path("/proc/driver/nvidia/version")
    if driver_path.is_file():
        driver_identity = "kernel-" + _sha256(driver_path)
    return {
        "name": name.value.decode("utf-8"),
        "compute_capability": "{}.{}".format(major.value, minor.value),
        "driver_version": driver_identity,
    }


def _target_descriptor() -> typing.Dict[str, typing.Any]:
    try:
        gpu = _nvidia_smi_gpu_descriptor()
    except (CacheError, OSError, subprocess.SubprocessError):
        gpu = _cuda_driver_gpu_descriptor()
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "gpu": gpu,
    }


def _platform_descriptor(target=None) -> typing.Dict[str, typing.Any]:
    descriptor = dict(target or _target_descriptor())
    descriptor.update({
        "cuda_version":
        os.environ.get("CUDA_VERSION", "<unset>"),
        "trt_version":
        os.environ.get("TRT_VERSION", "<unset>"),
        "trt_rtx_version":
        os.environ.get("TRT_RTX_VERSION", "<unset>"),
        "job_image":
        os.environ.get("CI_JOB_IMAGE", os.environ.get("IMAGE", "<unset>")),
        "lunowud":
        os.environ.get("__LUNOWUD", "<unset>"),
    })
    return descriptor


def _remote_descriptor(commands, runner):
    builders = sorted({command[0] for command, _ in commands})
    result = runner([
        "python3", _REMOTE_HELPER, "describe",
        json.dumps(builders, separators=(",", ":"))
    ], 900)
    if not result.get("success"):
        raise CacheError(
            "remote engine cache fingerprinting failed: {}".format(
                result.get("error") or result.get("output", "unknown error")))
    line = next((value for value in result.get("output", "").splitlines()
                 if value.startswith(_REMOTE_DESCRIPTOR_MARKER)), None)
    if line is None:
        raise CacheError("remote engine cache fingerprint is missing")
    descriptor = json.loads(line[len(_REMOTE_DESCRIPTOR_MARKER):])
    if descriptor.get(
            "schema_version") != "edgellm-ci-remote-engine-descriptor/v1":
        raise CacheError("remote engine cache fingerprint schema is invalid")
    builders = descriptor.get("builders")
    plugins = descriptor.get("plugins")
    target = descriptor.get("target")
    if (not isinstance(builders, list) or not isinstance(plugins, list)
            or not all(isinstance(value, str) for value in plugins)
            or not isinstance(target, dict)):
        raise CacheError("remote engine cache fingerprint payload is invalid")
    return builders, [pathlib.Path(value) for value in plugins], target


def _remote_probe_engines(bundle: pathlib.Path,
                          plugin_paths: typing.Sequence[pathlib.Path], runner,
                          trtexec_path: str) -> bool:
    engine_paths = sorted(bundle.rglob("*.engine"))
    if not engine_paths:
        return False
    for engine in engine_paths:
        command = [
            trtexec_path, "--loadEngine={}".format(engine), "--skipInference"
        ]
        command.extend("--dynamicPlugins={}".format(path)
                       for path in plugin_paths)
        if not runner(command, 900).get("success"):
            return False
    return True


def _cache_mode():
    raw = os.environ.get("EDGE_LLM_ENGINE_CACHE_MODE", "off")
    try:
        mode = CacheMode(raw)
    except ValueError:
        return CacheMode.OFF
    if (mode is not CacheMode.REFRESH and any(
            _truthy(os.environ.get(name, "")) for name in _COLD_ENVIRONMENT)):
        return CacheMode.OFF
    return mode


def _namespace_policy(environment=None):
    environment = os.environ if environment is None else environment
    project = environment.get("CI_PROJECT_ID", "local")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", project):
        raise CacheError("CI_PROJECT_ID is not a safe cache component")
    prefix = "project-{}".format(project)

    def trusted_for(branch):
        if branch == "main":
            return "{}-main".format(prefix)
        if branch.startswith("release/"):
            release = branch[len("release/"):]
            if not re.fullmatch(r"[A-Za-z0-9._-]+", release):
                raise CacheError(
                    "release branch is not a safe cache component")
            return "{}-release-{}".format(prefix, release)
        return "{}-main".format(prefix)

    merge_request = environment.get("CI_MERGE_REQUEST_IID", "")
    if merge_request:
        if not merge_request.isdecimal():
            raise CacheError("CI_MERGE_REQUEST_IID must be decimal")
        overlay = "{}-mr-{}".format(prefix, merge_request)
        trusted = trusted_for(
            environment.get("CI_MERGE_REQUEST_TARGET_BRANCH_NAME", "main"))
        return [overlay, trusted], overlay

    branch = environment.get("CI_COMMIT_BRANCH", "")
    protected = environment.get("CI_COMMIT_REF_PROTECTED",
                                "").lower() == "true"
    if protected and (branch == "main" or branch.startswith("release/")):
        trusted = trusted_for(branch)
        return [trusted], trusted

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-",
                  environment.get("CI_COMMIT_REF_SLUG",
                                  "local")).strip(".-")[:64]
    overlay = "{}-ref-{}".format(prefix, slug or "local")
    return [overlay], overlay


def _cache_root() -> pathlib.Path:
    value = (os.environ.get("EDGE_LLM_ARTIFACT_CACHE_DIR")
             or os.environ.get("EDGE_LLM_CI_CACHE_ROOT", ""))
    root = pathlib.Path(value)
    if not value or not root.is_absolute() or root == pathlib.Path(
            "/") or root.is_symlink():
        raise CacheError("artifact cache root is not a safe absolute path")
    return root


def _configured_limit():
    name = ("EDGE_LLM_ENGINE_MR_CACHE_BYTES"
            if os.environ.get("CI_MERGE_REQUEST_IID") else
            "EDGE_LLM_ENGINE_TRUSTED_CACHE_BYTES")
    default = 20 * 1024**3 if os.environ.get(
        "CI_MERGE_REQUEST_IID") else 200 * 1024**3
    value = os.environ.get(name, str(default))
    if not value.isdecimal():
        raise CacheError("{} must be a nonnegative integer".format(name))
    return int(value)


def _bundle_roots(commands: typing.Sequence[typing.Tuple[typing.List[str],
                                                         int]],
                  base: pathlib.Path) -> typing.List[pathlib.PurePosixPath]:
    resolved_base = base.resolve(strict=False)
    roots = set()
    for command, _ in commands:
        value = _argument_value(command, "--engineDir=")
        if not value:
            continue
        output = pathlib.Path(value).resolve(strict=False)
        try:
            relative = output.relative_to(resolved_base)
        except ValueError as error:
            raise CacheError(
                "engine output is outside its TestConfig bundle root: {}".
                format(output)) from error
        if not relative.parts:
            raise CacheError("engine bundle root cannot be the model root")
        roots.add(pathlib.PurePosixPath(*relative.parts))
    ordered = sorted(roots,
                     key=lambda item: (len(item.parts), item.as_posix()))
    selected = []
    for root in ordered:
        nested = False
        for parent in selected:
            try:
                root.relative_to(parent)
                nested = True
                break
            except ValueError:
                pass
        if not nested:
            selected.append(root)
    return selected


def _validate_tree(root: pathlib.Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise CacheError(
            "engine bundle component is missing or unsafe: {}".format(root))
    for candidate in root.rglob("*"):
        if candidate.is_symlink() or (not candidate.is_dir()
                                      and not candidate.is_file()):
            raise CacheError(
                "engine bundle contains an unsafe entry: {}".format(candidate))


def _copy_bundle(source_base: pathlib.Path, target_base: pathlib.Path,
                 roots: typing.Sequence[pathlib.PurePosixPath]) -> None:
    for relative in roots:
        source = source_base.joinpath(*relative.parts)
        _validate_tree(source)
        destination = target_base.joinpath(*relative.parts)
        for current, dir_names, file_names in os.walk(str(source),
                                                      followlinks=False):
            dir_names.sort()
            file_names.sort()
            current_path = pathlib.Path(current)
            relative_current = current_path.relative_to(source)
            destination_current = destination / relative_current
            destination_current.mkdir(parents=True, exist_ok=True)
            for name in file_names:
                source_file = current_path / name
                if source_file.is_symlink() or not source_file.is_file():
                    raise CacheError(
                        "engine bundle contains an unsafe file: {}".format(
                            source_file))
                shutil.copy2(str(source_file), str(destination_current / name))


def _install_bundle(source_base: pathlib.Path, target_base: pathlib.Path,
                    roots: typing.Sequence[pathlib.PurePosixPath]) -> None:
    staged = []
    backups = []
    installed = []
    preserved_backups = set()
    target_base.mkdir(parents=True, exist_ok=True)
    try:
        for relative in roots:
            source = source_base.joinpath(*relative.parts)
            _validate_tree(source)
            destination = target_base.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(".{}.restore.{}".format(
                destination.name,
                uuid.uuid4().hex))
            os.rename(str(source), str(temporary))
            staged.append((temporary, destination))
        for temporary, destination in staged:
            backup = destination.with_name(".{}.backup.{}".format(
                destination.name,
                uuid.uuid4().hex))
            if destination.exists():
                os.rename(str(destination), str(backup))
                backups.append((backup, destination))
            os.rename(str(temporary), str(destination))
            installed.append(destination)
        for backup, _ in backups:
            shutil.rmtree(str(backup), ignore_errors=True)
    except BaseException:
        for destination in reversed(installed):
            with contextlib.suppress(OSError):
                if destination.exists():
                    shutil.rmtree(str(destination))
        for backup, destination in reversed(backups):
            try:
                if destination.exists():
                    shutil.rmtree(str(destination))
                os.rename(str(backup), str(destination))
            except OSError:
                preserved_backups.add(backup)
        raise
    finally:
        for temporary, _ in staged:
            shutil.rmtree(str(temporary), ignore_errors=True)
        for backup, _ in backups:
            if backup not in preserved_backups:
                shutil.rmtree(str(backup), ignore_errors=True)


def _probe_engines(bundle: pathlib.Path,
                   plugin_paths: typing.Sequence[pathlib.Path]) -> bool:
    try:
        import tensorrt
        for plugin in plugin_paths:
            ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
        logger = tensorrt.Logger(tensorrt.Logger.ERROR)
        runtime = tensorrt.Runtime(logger)
        engine_paths = sorted(bundle.rglob("*.engine"))
        if not engine_paths:
            return False
        for path in engine_paths:
            with path.open("rb") as stream:
                with mmap.mmap(stream.fileno(), 0,
                               access=mmap.ACCESS_READ) as serialized:
                    engine = runtime.deserialize_cuda_engine(serialized)
                    if engine is None:
                        return False
                    del engine
        return True
    except Exception:
        return False


def _cache_entry(root: pathlib.Path, namespace: str,
                 kind: str) -> pathlib.Path:
    return root / "engines" / namespace / kind


def _tree_size(root: pathlib.Path) -> typing.Tuple[int, int]:
    files = 0
    size = 0
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise CacheError("symlinked engine cache entry is not allowed")
        if candidate.is_file():
            files += 1
            size += candidate.stat().st_size
        elif not candidate.is_dir():
            raise CacheError("special engine cache entry is not allowed")
    return files, size


def _replace_directory(staged: pathlib.Path,
                       destination: pathlib.Path) -> None:
    backup = destination.with_name(".{}.backup.{}".format(
        destination.name,
        uuid.uuid4().hex))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        os.rename(str(destination), str(backup))
    try:
        os.rename(str(staged), str(destination))
    except BaseException:
        if backup.exists():
            os.rename(str(backup), str(destination))
        raise
    shutil.rmtree(str(backup), ignore_errors=True)


def _read_metadata(entry: pathlib.Path, key: str,
                   roots: typing.Sequence[pathlib.PurePosixPath]) -> None:
    metadata = json.loads(
        (entry / _BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    cached_roots = [
        pathlib.PurePosixPath(value) for value in metadata["roots"]
    ]
    if (metadata.get("schema_version") != _CACHE_SCHEMA_VERSION
            or metadata.get("fingerprint") != key or cached_roots != roots):
        raise CacheError("engine bundle identity does not match the request")


def _allow_shared_read(root: pathlib.Path) -> None:
    os.chmod(root, root.stat().st_mode | 0o055)
    for candidate in root.rglob("*"):
        mode = candidate.stat().st_mode
        os.chmod(candidate, mode | (0o055 if candidate.is_dir() else 0o044))


class EngineBundleCache:
    """Cache coordinator for one TestConfig engine bundle.

    Attributes:
        key: Content key covering ONNX, build commands, builder payload, and platform.
    """

    def __init__(self,
                 config,
                 commands,
                 logger=None,
                 probe=None,
                 remote_runner=None,
                 trtexec_path=None):
        """Initialize a fail-open cache coordinator.

        Args:
            config: TestConfig that owns the engine output root.
            commands: Generated engine builder commands and timeouts.
            logger: Optional pytest logger.
            probe: Optional engine-deserialization callback for tests.
            remote_runner: Optional command callback for a shared-storage board.
            trtexec_path: Target-side trtexec used to validate remote bundles.
        """
        self._logger = logger
        self._commands = commands
        self._probe = probe or _probe_engines
        self._mode = _cache_mode()
        self._base = pathlib.Path(config.get_engine_base_dir())
        self._roots = []
        self._cache_root = None
        self._read_namespaces = []
        self._write_namespace = ""
        self._plugin_paths = []
        self._kind = ""
        self._engine_descriptor = None
        self._remote = remote_runner is not None
        self.key = ""
        if self._mode is CacheMode.OFF:
            started = time.monotonic()
            _cache_event(logger,
                         "engine",
                         "lookup",
                         started,
                         "policy-disabled",
                         detail="policy-disabled")
            return
        if not commands:
            return
        key_started = time.monotonic()
        try:
            self._cache_root = _cache_root()
            self._read_namespaces, self._write_namespace = _namespace_policy()
            self._roots = _bundle_roots(commands, self._base)
            if not self._roots:
                raise CacheError(
                    "no engine output roots were found in build commands")
            if remote_runner is None:
                builders, self._plugin_paths = _builder_descriptor(commands)
                platform_identity = _platform_descriptor()
            else:
                if not trtexec_path:
                    raise CacheError(
                        "remote engine validation requires trtexec")
                builders, self._plugin_paths, target = _remote_descriptor(
                    commands, remote_runner)
                platform_identity = _platform_descriptor(target)

                def remote_probe(bundle, plugins):
                    return _remote_probe_engines(bundle, plugins,
                                                 remote_runner, trtexec_path)

                self._probe = remote_probe
            excluded_gpus = {
                value.strip()
                for value in os.environ.get(
                    "EDGE_LLM_ENGINE_CACHE_EXCLUDED_GPUS", "").split(",")
                if value.strip()
            }
            if platform_identity["gpu"]["name"] in excluded_gpus:
                raise CacheError(
                    "engine cache is performance-gated for {}".format(
                        platform_identity["gpu"]["name"]))
            onnx_inputs = []
            seen = set()
            for command, _ in commands:
                value = _argument_value(command, "--onnxDir=")
                if value and value not in seen:
                    seen.add(value)
                    onnx_inputs.append(
                        _tree_descriptor(
                            str(pathlib.Path(value).resolve(strict=True))))
            if not onnx_inputs:
                raise CacheError("no ONNX inputs were found in build commands")
            normalized_commands = _normalized_commands(commands)
            epoch = os.environ.get("EDGE_LLM_ENGINE_CACHE_EPOCH", "1")
            engine_descriptor = {
                "schema_version": "edgellm-ci-engine-descriptor/v1",
                "artifact": "trt-engine-bundle",
                "builders": builders,
                "platform": platform_identity,
                "epoch": epoch,
                "commands": normalized_commands,
                "onnx": onnx_inputs,
            }
            self.key = _content_key(engine_descriptor)
            logical_descriptor = {
                "schema_version": "edgellm-ci-engine-kind/v1",
                "commands": normalized_commands,
                "platform": {
                    "system":
                    platform_identity["system"],
                    "machine":
                    platform_identity["machine"],
                    "gpu_name":
                    platform_identity["gpu"]["name"],
                    "compute_capability":
                    platform_identity["gpu"]["compute_capability"],
                    "cuda_version":
                    platform_identity["cuda_version"],
                    "trt_version":
                    platform_identity["trt_version"],
                    "trt_rtx_version":
                    platform_identity["trt_rtx_version"],
                },
            }
            self._kind = (_ENGINE_KIND_PREFIX +
                          _content_key(logical_descriptor)[:24])
            self._engine_descriptor = engine_descriptor
            _cache_event(self._logger, self._kind, "key", key_started, "ready",
                         self.key)
        except (CacheError, OSError, RuntimeError, subprocess.SubprocessError,
                ValueError) as error:
            _log(logger, "warning", "TensorRT cache disabled: %s", error)
            decision = ("restore-cost-gate" if "performance-gated"
                        in str(error) else "unrepresentable-input")
            _cache_event(logger,
                         "engine",
                         "lookup",
                         key_started,
                         decision,
                         detail=decision)
            self._mode = CacheMode.OFF

    @property
    def enabled(self) -> bool:
        """Return whether strict cache identity was established."""
        return self._mode is not CacheMode.OFF and bool(self.key)

    def prepare_build(self) -> None:
        """Remove declared bundle roots before rebuilding after a cache miss."""
        if not self.enabled:
            return
        try:
            for relative in self._roots:
                destination = self._base.joinpath(*relative.parts)
                if destination.is_symlink():
                    destination.unlink()
                elif destination.exists():
                    if not destination.is_dir():
                        raise CacheError(
                            "engine output root is not a directory: {}".format(
                                destination))
                    shutil.rmtree(str(destination))
        except (CacheError, OSError) as error:
            _log(
                self._logger, "warning",
                "Could not clean TensorRT engine roots; disabling publication: %s",
                error)
            self._mode = CacheMode.OFF

    def restore_bundle(self) -> bool:
        """Restore and deserialize-probe a complete bundle; return hit status."""
        if not self.enabled:
            return False
        started = time.monotonic()
        if not self._mode.reads:
            _cache_event(self._logger,
                         self._kind,
                         "restore",
                         started,
                         "forced-cold",
                         self.key,
                         detail="forced-cold")
            return False
        miss_reason = "not-found"
        self._base.parent.mkdir(parents=True, exist_ok=True)
        for namespace in self._read_namespaces:
            entry = _cache_entry(self._cache_root, namespace, self._kind)
            if not entry.is_dir() or entry.is_symlink():
                continue
            try:
                _read_metadata(entry, self.key, self._roots)
                with tempfile.TemporaryDirectory(
                        prefix=".edgellm-engine-restore-",
                        dir=str(self._base.parent)) as temporary:
                    target = pathlib.Path(temporary) / "bundle"
                    _copy_bundle(entry, target, self._roots)
                    if self._remote:
                        _allow_shared_read(pathlib.Path(temporary))
                    if not self._probe(target, self._plugin_paths):
                        raise CacheError(
                            "TensorRT could not deserialize every cached engine"
                        )
                    files, size = _tree_size(target)
                    _install_bundle(target, self._base, self._roots)
                result = {
                    "namespace": namespace,
                    "files": files,
                    "bytes": size,
                }
                _cache_event(self._logger, self._kind, "restore", started,
                             "hit", self.key, result)
                return True
            except json.JSONDecodeError as error:
                miss_reason = "invalid-manifest"
                _log(self._logger, "warning",
                     "Ignoring TensorRT engine cache entry %s: %s", namespace,
                     error)
            except (KeyError, TypeError, ValueError) as error:
                miss_reason = "invalid-manifest"
                _log(self._logger, "warning",
                     "Ignoring TensorRT engine cache entry %s: %s", namespace,
                     error)
            except CacheError as error:
                detail = str(error)
                if "deserialize" in detail:
                    miss_reason = "deserialization-failure"
                elif "identity" in detail:
                    miss_reason = "descriptor-change"
                elif "missing" in detail or "unsafe" in detail:
                    miss_reason = "invalid-inventory"
                else:
                    miss_reason = "corruption"
                _log(self._logger, "warning",
                     "Ignoring TensorRT engine cache entry %s: %s", namespace,
                     error)
            except OSError as error:
                miss_reason = "corruption"
                _log(self._logger, "warning",
                     "Ignoring TensorRT engine cache entry %s: %s", namespace,
                     error)
        _cache_event(self._logger,
                     self._kind,
                     "restore",
                     started,
                     "miss",
                     self.key,
                     detail=miss_reason)
        return False

    def publish_bundle(self) -> None:
        """Replace the latest bundle after all builders succeed."""
        if not self.enabled or not self._mode.writes:
            return
        started = time.monotonic()
        entry = _cache_entry(self._cache_root, self._write_namespace,
                             self._kind)
        temporary = entry.with_name(".{}.publish.{}".format(
            entry.name,
            uuid.uuid4().hex))
        try:
            if not self._probe(self._base, self._plugin_paths):
                raise CacheError(
                    "refusing to publish an invalid TensorRT engine bundle")
            shutil.rmtree(str(temporary), ignore_errors=True)
            _copy_bundle(self._base, temporary, self._roots)
            if self._remote:
                _allow_shared_read(temporary)
            files, size = _tree_size(temporary)
            if size > _configured_limit():
                raise CacheError(
                    "TensorRT engine bundle exceeds the configured cache quota"
                )
            (temporary / _BUNDLE_MANIFEST).write_text(json.dumps(
                {
                    "schema_version": _CACHE_SCHEMA_VERSION,
                    "fingerprint": self.key,
                    "roots": [root.as_posix() for root in self._roots],
                    "key_descriptor": self._engine_descriptor,
                },
                indent=2,
                sort_keys=True) + "\n",
                                                      encoding="utf-8")
            _replace_directory(temporary, entry)
            result = {
                "namespace": self._write_namespace,
                "files": files,
                "bytes": size,
            }
            _cache_event(self._logger, self._kind, "publish", started,
                         "published", self.key, result)
        except (CacheError, OSError, ValueError) as error:
            _log(self._logger, "warning",
                 "TensorRT engine cache publish failed open: %s", error)
            _cache_event(self._logger,
                         self._kind,
                         "publish",
                         started,
                         "error",
                         self.key,
                         detail=str(error))
        finally:
            shutil.rmtree(str(temporary), ignore_errors=True)


def _describe_remote(builders_json: str) -> typing.Dict[str, typing.Any]:
    builders = json.loads(builders_json)
    if (not isinstance(builders, list) or not builders
            or any(not isinstance(value, str) or not value
                   for value in builders)):
        raise CacheError("remote builder list is invalid")
    descriptors, plugins = _builder_descriptor([([value], 0)
                                                for value in builders])
    return {
        "schema_version": "edgellm-ci-remote-engine-descriptor/v1",
        "builders": descriptors,
        "plugins": [str(path) for path in plugins],
        "target": _target_descriptor(),
    }


def _main(argv: typing.Sequence[str]) -> int:
    if len(argv) != 2 or argv[0] != "describe":
        raise SystemExit("usage: ci_engine_cache.py describe <builders-json>")
    descriptor = _describe_remote(argv[1])
    print(_REMOTE_DESCRIPTOR_MARKER + json.dumps(descriptor, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
