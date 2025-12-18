---
name: Bug report - Python Export Pipeline
about: Submit a bug report for the Python export pipeline (model quantization/export)
title: ''
labels: bug, export
assignees: ''
---

## Describe the bug
<!-- Description of what the bug is, its impact (blocker, should have, nice to have) and any stack traces or error messages. -->

### Steps/Code to reproduce bug
<!-- Please list *minimal* steps or code snippet for us to be able to reproduce the bug. -->
<!-- A helpful guide on on how to craft a minimal bug report http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports. -->

**Installation method:**
<!-- How did you install TensorRT Edge-LLM? (e.g., pip install ., from source) -->

**Export command used:**
```bash
# Paste the exact command(s) you ran, for example:
# tensorrt-edgellm-quantize-llm --model_dir Qwen/Qwen3-0.6B --output_dir ./quantized/qwen3-0.6b --quantization fp8
# tensorrt-edgellm-export-llm --model_dir ./quantized/qwen3-0.6b --output_dir ./onnx_models/qwen3-0.6b
```

### Expected behavior

## System information (x86 Host with GPU)

- Container used (if applicable): ?
- OS (e.g., Ubuntu 22.04, CentOS 7): ? <!-- Export pipeline requires x86-64 Linux -->
- CPU architecture: ? <!-- Should be x86_64 for export pipeline -->
- GPU name (e.g. H100, A100, RTX 4090): ?
- GPU memory size: ?
- Number of GPUs: ?
- Library versions:
  - Python: ?
  - TensorRT Edge-LLM version or commit hash: ?
  - CUDA: ?
  - PyTorch: ?
  - Transformers: ?
  - ModelOpt: ?
  - ONNX: ?
- Any other details that may help: ?

<details>
<summary><b>Click to expand: Python script to automatically collect system information</b></summary>

```python
import platform
import re
import subprocess


def get_nvidia_gpu_info():
    try:
        nvidia_smi = (
            subprocess.check_output(
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits",
                shell=True,
            )
            .decode("utf-8")
            .strip()
            .split("\n")
        )
        if len(nvidia_smi) > 0:
            gpu_name = nvidia_smi[0].split(",")[0].strip()
            gpu_memory = round(float(nvidia_smi[0].split(",")[1].strip()) / 1024, 1)
            gpu_count = len(nvidia_smi)
            return gpu_name, f"{gpu_memory} GB", gpu_count
    except Exception:
        return "?", "?", "?"


def get_cuda_version():
    try:
        nvcc_output = subprocess.check_output("nvcc --version", shell=True).decode("utf-8")
        match = re.search(r"release (\d+\.\d+)", nvcc_output)
        if match:
            return match.group(1)
    except Exception:
        return "?"


def get_package_version(package):
    try:
        return getattr(__import__(package), "__version__", "?")
    except Exception:
        return "?"


def get_tensorrt_edgellm_version():
    try:
        import tensorrt_edgellm
        return tensorrt_edgellm.__version__
    except Exception:
        return "?"


# Get system info
os_info = f"{platform.system()} {platform.release()}"
if platform.system() == "Linux":
    try:
        os_info = (
            subprocess.check_output("cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2", shell=True)
            .decode("utf-8")
            .strip()
            .strip('"')
        )
    except Exception:
        pass

cpu_arch = platform.machine()
gpu_name, gpu_memory, gpu_count = get_nvidia_gpu_info()
cuda_version = get_cuda_version()

# Print system information in the format required for the issue template
print("=" * 70)
print("## System information (x86 Host with GPU)")
print()
print("- Container used (if applicable): " + "?")
print("- OS (e.g., Ubuntu 22.04, CentOS 7): " + os_info)
print("- CPU architecture: " + cpu_arch)
print("- GPU name (e.g. H100, A100, RTX 4090): " + gpu_name)
print("- GPU memory size: " + gpu_memory)
print("- Number of GPUs: " + str(gpu_count))
print("- Library versions:")
print("  - Python: " + platform.python_version())
print("  - TensorRT Edge-LLM version or commit hash: " + get_tensorrt_edgellm_version())
print("  - CUDA: " + cuda_version)
print("  - PyTorch: " + get_package_version("torch"))
print("  - Transformers: " + get_package_version("transformers"))
print("  - ModelOpt: " + get_package_version("modelopt"))
print("  - ONNX: " + get_package_version("onnx"))
print("- Any other details that may help: " + "?")
print("=" * 70)
```

</details>
