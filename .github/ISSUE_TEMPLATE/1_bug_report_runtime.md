---
name: Bug report - C++ Runtime
about: Submit a bug report for the C++ runtime (engine building/inference)
title: ''
labels: bug, runtime
assignees: ''
---

## Describe the bug
<!-- Description of what the bug is, its impact (blocker, should have, nice to have) and any stack traces or error messages. -->

### Steps/Code to reproduce bug
<!-- Please list *minimal* steps or code snippet for us to be able to reproduce the bug. -->
<!-- A helpful guide on on how to craft a minimal bug report http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports. -->

**Build configuration:**
```bash
# Paste your CMake command, for example:
# cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/path/to/TensorRT -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake -DEMBEDDED_TARGET=jetson-thor
```

**Runtime command used:**
```bash
# Paste the exact command(s) you ran, for example:
# ./build/examples/llm/llm_build --onnxDir ./onnx_models/qwen3-0.6b --engineDir ./engines/qwen3-0.6b --maxBatchSize 1
# ./build/examples/llm/llm_inference --engineDir ./engines/qwen3-0.6b --inputFile input.json --outputFile output.json
```

### Expected behavior

## System information (Edge Device)

- Platform (e.g., NVIDIA Jetson Thor): ?
- Software release (e.g., JetPack 7.1): ?
- CPU architecture: ? <!-- Should be aarch64 for Edge devices -->
- GPU compute capability (e.g., SM110 for Jetson Thor): ?
- Total device memory: ?
- Build type (e.g., Release, Debug): ?
- Library versions:
  - TensorRT Edge-LLM version or commit hash: ?
  - CUDA: ?
  - TensorRT: ?
  - C++ compiler (e.g., GCC 11.4): ?
- CMake options used:
  - CMAKE_TOOLCHAIN_FILE: ?
  - EMBEDDED_TARGET: ?
  - TRT_PACKAGE_DIR: ?
- Any other details that may help: ?

<details>
<summary><b>Click to expand: Python script to automatically collect system information</b></summary>

```python
import platform
import re
import subprocess


def get_cuda_version():
    try:
        nvcc_output = subprocess.check_output("nvcc --version", shell=True).decode("utf-8")
        match = re.search(r"release (\d+\.\d+)", nvcc_output)
        if match:
            return match.group(1)
    except Exception:
        return "?"


def get_tensorrt_version():
    try:
        dpkg_output = subprocess.check_output("dpkg -l | grep tensorrt", shell=True).decode("utf-8")
        match = re.search(r"(\d+\.\d+\.\d+)", dpkg_output)
        if match:
            return match.group(1)
    except Exception:
        return "?"


def get_gcc_version():
    try:
        gcc_output = subprocess.check_output("gcc --version", shell=True).decode("utf-8")
        match = re.search(r"gcc.*?(\d+\.\d+\.\d+)", gcc_output)
        if match:
            return f"GCC {match.group(1)}"
    except Exception:
        return "?"


def get_gpu_compute_capability():
    try:
        # Try to get GPU compute capability from nvidia-smi
        smi_output = subprocess.check_output(
            "nvidia-smi --query-gpu=compute_cap --format=csv,noheader",
            shell=True
        ).decode("utf-8").strip()
        if smi_output:
            cap = smi_output.replace(".", "")
            return f"SM{cap}"
    except Exception:
        pass
    return "?"


def get_total_memory():
    try:
        mem_output = subprocess.check_output("free -h | grep Mem", shell=True).decode("utf-8")
        match = re.search(r"\s+(\d+\.\d+[GM]i?)\s+", mem_output)
        if match:
            return match.group(1)
    except Exception:
        return "?"


# Get system info
cpu_arch = platform.machine()
platform_name = "?"
software_release = "?"
gpu_compute_cap = get_gpu_compute_capability()
total_memory = get_total_memory()
cuda_version = get_cuda_version()
tensorrt_version = get_tensorrt_version()
gcc_version = get_gcc_version()

# Print system information in the format required for the issue template
print("=" * 70)
print("## System information (Edge Device)")
print()
print("- Platform (e.g., NVIDIA Jetson Thor, NVIDIA DRIVE Thor): " + platform_name)
print("- Software release (e.g., JetPack 7.1, NVIDIA DRIVE OS 7): " + software_release)
print("- CPU architecture: " + cpu_arch)
print("- GPU compute capability (e.g., SM87 for Jetson Thor): " + gpu_compute_cap)
print("- Total device memory: " + total_memory)
print("- Build type (e.g., Release, Debug): " + "?")
print("- Library versions:")
print("  - TensorRT Edge-LLM version or commit hash: " + "?")
print("  - CUDA: " + cuda_version)
print("  - TensorRT: " + tensorrt_version)
print("  - C++ compiler (e.g., GCC 11.4): " + gcc_version)
print("- CMake options used:")
print("  - CMAKE_TOOLCHAIN_FILE: " + "?")
print("  - EMBEDDED_TARGET: " + "?")
print("  - TRT_PACKAGE_DIR: " + "?")
print("- Any other details that may help: " + "?")
print("=" * 70)
```

</details>
