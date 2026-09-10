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
"""Detect raw platform facts without embedding the supported-release policy."""

from __future__ import annotations

import ctypes
import ctypes.util
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import NativeDetectionError

_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
_DGX_SPARK_MODEL_MARKERS = ("dgx spark", "gb10")


@dataclass(frozen=True)
class DetectedPlatform:
    """Raw facts used to select an installed native payload."""

    python_abi: str
    platform_probe_source: str
    platform_probe_value: str
    cpu_arch: str
    cuda_runtime_soname: str
    tensorrt_runtime_soname: str
    gpu_sm: int

    def as_dict(self) -> Dict[str, object]:
        """Return raw facts for selection diagnostics."""
        return {
            "python_abi": self.python_abi,
            "platform_probe_source": self.platform_probe_source,
            "platform_probe_value": self.platform_probe_value,
            "cpu_arch": self.cpu_arch,
            "cuda_runtime_soname": self.cuda_runtime_soname,
            "tensorrt_runtime_soname": self.tensorrt_runtime_soname,
            "gpu_sm": self.gpu_sm,
        }


def _python_abi() -> str:
    cache_tag = getattr(sys.implementation, "cache_tag", "") or ""
    match = re.fullmatch(r"cpython-(\d{2,3})(?:t)?", cache_tag)
    if not match or cache_tag.endswith("t"):
        raise NativeDetectionError(
            "TensorRT Edge-LLM wheels require a supported GIL-enabled CPython; "
            f"detected cache tag {cache_tag!r}.")
    return f"cp{match.group(1)}"


def _cpu_arch() -> str:
    machine = platform.machine().lower()
    normalized = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    if normalized not in {"x86_64", "aarch64"}:
        raise NativeDetectionError(
            f"Unsupported CPU architecture {machine!r}.")
    return normalized


def _read_key_values(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8",
                               errors="replace").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def _device_model_probe() -> Optional[Tuple[str, str]]:
    path = Path("/proc/device-tree/model")
    if path.is_file():
        model = path.read_text(encoding="utf-8",
                               errors="replace").strip("\x00\n ")
        if model:
            return "device-model", model

    # DGX Spark is ACPI/DMI based and does not expose a device-tree model.
    path = Path("/sys/class/dmi/id/product_name")
    if not path.is_file():
        return None
    model = path.read_text(encoding="utf-8", errors="replace").strip()
    normalized = re.sub(r"[_-]+", " ", model)
    if any(marker in normalized.casefold()
           for marker in _DGX_SPARK_MODEL_MARKERS):
        return "device-model", normalized
    return None


def _drive_probe() -> Optional[Tuple[str, str]]:
    release = Path("/etc/driveos-release")
    values = _read_key_values(release)
    text = values.get("VERSION_ID", values.get("VERSION", ""))
    if not text:
        model = _device_model_probe()
        if model is not None and any(marker in model[1].casefold()
                                     for marker in _DGX_SPARK_MODEL_MARKERS):
            return None
        rootfs = Path("/etc/nvidia/version-ubuntu-rootfs.txt")
        if not rootfs.is_file():
            return None
        text = rootfs.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\b(\d+\.\d+)(?:\.\d+)*\b", text)
    if match is None:
        raise NativeDetectionError(
            "A DRIVE release marker exists but has no parseable version.")
    return "driveos", match.group(1)


def _l4t_probe() -> Optional[Tuple[str, str]]:
    path = Path("/etc/nv_tegra_release")
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"R(\d+)\s*\(release\).*?REVISION:\s*(\d+)(?:\.\d+)?",
                      text)
    if match is None:
        match = re.search(r"R(\d+)\.(\d+)", text)
    if match is None:
        raise NativeDetectionError(
            f"Cannot parse the NVIDIA platform release in {path}.")
    return "l4t", f"R{int(match.group(1))}.{int(match.group(2))}"


def _ubuntu_probe(cpu_arch: str) -> Optional[Tuple[str, str]]:
    values = _read_key_values(Path("/etc/os-release"))
    if cpu_arch == "x86_64" and values.get("ID") == "ubuntu":
        return "os-release", f"ubuntu:{values.get('VERSION_ID', 'unknown')}"
    return None


def _platform_probe(cpu_arch: str) -> Tuple[str, str]:
    for probe in (_drive_probe, _l4t_probe, _device_model_probe):
        result = probe()
        if result is not None:
            return result
    ubuntu = _ubuntu_probe(cpu_arch)
    if ubuntu is not None:
        return ubuntu
    raise NativeDetectionError(
        "Cannot identify a supported platform from system release markers.")


def _find_soname(library: str) -> str:
    soname = ctypes.util.find_library(library)
    if not soname:
        raise NativeDetectionError(
            f"Required system library {library!r} was not found.")
    return Path(soname).name


def _load_cuda_device_api() -> Any:
    driver_name = ctypes.util.find_library("cuda") or "libcuda.so.1"
    try:
        driver = ctypes.CDLL(driver_name)
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuDeviceGetCount.restype = ctypes.c_int
        driver.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
        ]
        driver.cuDeviceGetAttribute.restype = ctypes.c_int
    except (AttributeError, OSError) as error:
        raise NativeDetectionError(
            f"Cannot load the CUDA driver API from {driver_name}: {error}"
        ) from error
    return driver


def _cuda_device_count(driver: Any) -> int:
    result = driver.cuInit(0)
    if result != 0:
        raise NativeDetectionError(
            f"CUDA driver initialization failed with error {result}.")
    count = ctypes.c_int()
    result = driver.cuDeviceGetCount(ctypes.byref(count))
    if result != 0 or count.value < 1:
        raise NativeDetectionError(
            "No CUDA device is visible. Set CUDA_VISIBLE_DEVICES before "
            "starting Python.")
    return count.value


def _cuda_device_sm(driver: Any, device: int) -> int:
    major = ctypes.c_int()
    minor = ctypes.c_int()
    major_result = driver.cuDeviceGetAttribute(
        ctypes.byref(major), _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        device)
    minor_result = driver.cuDeviceGetAttribute(
        ctypes.byref(minor), _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        device)
    if major_result != 0 or minor_result != 0:
        raise NativeDetectionError(
            f"Cannot read compute capability for CUDA device {device}.")
    return major.value * 10 + minor.value


def _cuda_device_sms() -> Tuple[int, ...]:
    driver = _load_cuda_device_api()
    return tuple(
        _cuda_device_sm(driver, device)
        for device in range(_cuda_device_count(driver)))


def detect_platform() -> DetectedPlatform:
    """Detect the exact raw inputs required for payload selection."""
    cpu_arch = _cpu_arch()
    probe_source, probe_value = _platform_probe(cpu_arch)
    distinct_sms = sorted(set(_cuda_device_sms()))
    if len(distinct_sms) != 1:
        rendered = ", ".join(f"SM{sm}" for sm in distinct_sms) or "none"
        raise NativeDetectionError(
            "Visible CUDA devices must use one architecture; detected "
            f"{rendered}. Relaunch with CUDA_VISIBLE_DEVICES=<GPU-UUID> or "
            "expose a homogeneous-SM set.")
    return DetectedPlatform(
        python_abi=_python_abi(),
        platform_probe_source=probe_source,
        platform_probe_value=probe_value,
        cpu_arch=cpu_arch,
        cuda_runtime_soname=_find_soname("cudart"),
        tensorrt_runtime_soname=_find_soname("nvinfer"),
        gpu_sm=distinct_sms[0],
    )
