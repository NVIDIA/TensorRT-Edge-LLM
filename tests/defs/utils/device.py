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
"""Device detection and config for local/remote execution"""

from dataclasses import dataclass
from typing import Optional

from pytest_helpers import run_command


@dataclass
class DeviceConfig:
    arch: str
    cuda_version: Optional[str]
    target: str
    compute_capability: Optional[int]

    @classmethod
    def auto_detect(cls, remote_config=None, logger=None):
        """Auto-detect device configuration"""
        arch = _get_arch(remote_config, logger)
        cuda_version = _get_cuda_version(remote_config, logger)
        compute_cap = _get_compute_capability(remote_config, logger)

        target = cls._map_compute_cap_to_target(compute_cap)

        device_config = cls(arch, cuda_version, target, compute_cap)

        # Log the detected configuration
        if logger:
            logger.info(
                f"Detected: {device_config.arch}, CUDA {device_config.cuda_version}, "
                f"target {device_config.target} (compute cap {device_config.compute_capability})"
            )

        return device_config

    @staticmethod
    def _map_compute_cap_to_target(compute_cap: Optional[int]) -> str:
        """Map compute capability to target"""
        if compute_cap == 87:
            return 'jetson-orin'
        # Assume auto-thor uses CUDA 12.8 and compute cap 101
        elif compute_cap == 101:
            return 'auto-thor'
        # Assume jetson-thor uses CUDA 13.0 and compute cap 110. Need to adjust this later with build system.
        elif compute_cap == 110:
            return 'jetson-thor'
        elif compute_cap == 121:
            return 'gb10'
        else:
            return 'x86'  # let cmake decide for x86 architectures


def _get_arch(remote_config=None, logger=None) -> str:
    """Get system architecture"""
    result = run_command(['uname', '-m'], remote_config, 10, logger)
    if not result['success']:
        raise RuntimeError(
            f"Failed to get architecture from uname -m. {result}")
    return result['output'].strip()


def _get_cuda_version(remote_config=None, logger=None) -> str:
    """Get CUDA version using nvcc --version"""
    nvcc_result = run_command(['/usr/local/cuda/bin/nvcc', '--version'],
                              remote_config, 60, logger)
    if not nvcc_result['success']:
        raise RuntimeError(
            f"Failed to get CUDA version from nvcc --version. {nvcc_result}")

    if nvcc_result['output']:
        try:
            # Parse output like:
            # nvcc: NVIDIA (R) Cuda compiler driver
            # Copyright (c) 2005-2025 NVIDIA Corporation
            # Built on Fri_Feb_21_20:28:40_PST_2025
            # Cuda compilation tools, release 12.8, V12.8.93
            output = nvcc_result['output']
            for line in output.split('\n'):
                if 'release' in line.lower() and 'v' in line.lower():
                    # Extract version from "Cuda compilation tools, release 12.8, V12.8.93" format
                    parts = line.split(',')
                    for part in parts:
                        if 'release' in part.lower():
                            version_part = part.split('release')[-1].strip()
                            if version_part and '.' in version_part:
                                # Extract major.minor version
                                version_nums = version_part.split('.')[:2]
                                if len(version_nums) >= 2:
                                    return f"{version_nums[0]}.{version_nums[1]}"
        except Exception:
            pass
    raise RuntimeError(
        f"Failed to get CUDA version from nvcc --version. {nvcc_result}")


def _get_compute_capability(remote_config=None, logger=None) -> Optional[int]:
    """Get compute capability using nvidia-smi with ctypes fallback"""

    # Method 1: Try nvidia-smi first (most reliable)
    nvidia_smi_result = run_command([
        'nvidia-smi', '--query-gpu=compute_cap',
        '--format=csv,noheader,nounits'
    ], remote_config, 60, logger)

    if nvidia_smi_result['success'] and nvidia_smi_result['output'].strip():
        try:
            # Handle multiple GPUs - take the one with highest compute capability
            cap_lines = nvidia_smi_result['output'].strip().split('\n')
            max_compute_cap = 0
            for cap_str in cap_lines:
                if cap_str.strip():
                    major, minor = cap_str.strip().split('.')
                    compute_cap = int(major) * 10 + int(minor)
                    max_compute_cap = max(max_compute_cap, compute_cap)
            if max_compute_cap > 0:
                return max_compute_cap
        except Exception:
            if logger:
                logger.warning(
                    "Failed to get compute capability from nvidia-smi")

    # Method 2: Try ctypes method using /usr/local/cuda/lib64/libcudart.so (IMPORTANT FALLBACK)
    python_cmd = '''python3 -c "
import ctypes
class cudaDeviceProp(ctypes.Structure):
    _fields_ = [
        ('name', ctypes.c_char * 256),
        ('uuid', ctypes.c_ubyte * 16),
        ('luid', ctypes.c_char * 8),
        ('luidDeviceNodeMask', ctypes.c_uint),
        ('totalGlobalMem', ctypes.c_size_t),
        ('sharedMemPerBlock', ctypes.c_size_t),
        ('regsPerBlock', ctypes.c_int),
        ('warpSize', ctypes.c_int),
        ('memPitch', ctypes.c_size_t),
        ('maxThreadsPerBlock', ctypes.c_int),
        ('maxThreadsDim', ctypes.c_int * 3),
        ('maxGridSize', ctypes.c_int * 3),
        ('clockRate', ctypes.c_int),
        ('totalConstMem', ctypes.c_size_t),
        ('major', ctypes.c_int),
        ('minor', ctypes.c_int)
    ]
try:
    cuda = ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so')
    cuda.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuda.cudaGetDeviceCount.restype = ctypes.c_int
    cuda.cudaGetDeviceProperties.argtypes = [ctypes.POINTER(cudaDeviceProp), ctypes.c_int]
    cuda.cudaGetDeviceProperties.restype = ctypes.c_int

    device_count = ctypes.c_int()
    result = cuda.cudaGetDeviceCount(ctypes.byref(device_count))

    if result == 0 and device_count.value > 0:
        prop = cudaDeviceProp()
        result = cuda.cudaGetDeviceProperties(ctypes.byref(prop), 0)
        if result == 0:
            print(f'{prop.major}.{prop.minor}')
        else:
            exit(1)
    else:
        exit(1)
except:
    exit(1)
"'''

    cap_result = run_command(['bash', '-c', python_cmd], remote_config, 60,
                             logger)
    if cap_result['success'] and cap_result['output'].strip():
        try:
            cap_str = cap_result['output'].strip()
            major, minor = cap_str.split('.')
            compute_cap = int(major) * 10 + int(minor)
            return compute_cap
        except Exception:
            if logger:
                logger.warning("Failed to get compute capability from ctypes")

    raise RuntimeError(
        "Failed to get compute capability from nvidia-smi or ctypes")
