# Installation

TensorRT Edge-LLM has two separate components that need to be installed on different systems:

1. **Export and quantization** (runs on an x86 host; only quantization requires a GPU)
2. **C++ Runtime** (Jetson Thor, NVIDIA DRIVE / DriveOS, NVIDIA DGX Spark, or optional x86 developer build)

---

## Part 1: Export and Quantization (x86 Host)

The Python frontend exports Hugging Face checkpoints and optionally quantizes
FP16/BF16 checkpoints before export. Export runs on CPU. Quantization requires
an NVIDIA GPU.

### System Requirements

- **Platform**: x86-64 Linux system
- **Recommended OS**: Ubuntu 22.04, 24.04
- **GPU for quantization**: NVIDIA GPU with Compute Capability 8.0+ (Ampere or newer)
- **CUDA for quantization**: 12.x or 13.x
- **Python**: 3.10+

#### Memory Requirements

- Export: at least 1.5 times the checkpoint size in CPU memory. No GPU is
  required.
- Quantization: GPU memory at least equal to the FP16 checkpoint size.

**Verify Your Prerequisites:**

```bash
# Check CUDA installation when quantizing
nvcc --version
# Should show CUDA 12.x or 13.x

# Check the GPU and available memory when quantizing
nvidia-smi
# Look for GPU memory (e.g., "24576MiB" for 24GB)

# Check Python version
python3 --version
# Should show Python 3.10 or higher
```

**If CUDA is not installed:**

Download and install CUDA Toolkit from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads). Choose version 12.x or 13.x for your system.

After installation, verify with `nvcc --version` and `nvidia-smi`.

### Installing

For a containerized environment for clean installation, it is recommended to use the NVIDIA PyTorch Docker image:

```bash
# Pull the recommended Docker image
docker pull nvcr.io/nvidia/pytorch:25.12-py3

# Run the container with GPU support
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.12-py3 \
    bash
```

**1. Clone Repository**

```bash
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git submodule update --init --recursive
```

**2. Install Python Dependencies**

If you are not using container, it is recommended to use a virtual environment:
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
```

Install the package and its dependencies. The base install registers the CLI
entry points (`tensorrt-edgellm-export`, `tensorrt-edgellm-quantize`, etc.) and
pulls the core checkpoint export dependencies. Optional tool dependencies stay
out of the base environment so export-only and server images do not pull
quantization, audio, and LoRA-merge packages unnecessarily.

```bash
# Install the package (registers CLI entry points and core export dependencies)
pip3 install -e .

# Required for quantization, LoRA merge, vocabulary reduction, audio preprocessing,
# and tokenizer helpers (re-installs with the tools extra)
pip3 install -e ".[tools]"

# Required only for the experimental high-level Python API and server
pip3 install -e ".[server]"
```

The base install includes:
- PyTorch
- Transformers
- ONNX
- ONNX Script and ONNX GraphSurgeon

The optional `tools` extra adds NVIDIA Model Optimizer, calibration datasets,
audio preprocessing dependencies, LoRA merge dependencies, and tokenizer helpers.
The optional `server` extra adds FastAPI, Uvicorn, and pybind11 for the
experimental high-level Python API and OpenAI-compatible server.

> **Note:** Accuracy evaluation dependencies live under `examples/accuracy/requirements.txt`.

**3. Verify the Checkpoint Export Workflow**

Use the virtual environment created in Step 2 for this checkout. Do not mix
packages from older release branches into the same environment.

Export an unquantized or supported pre-quantized Hugging Face checkpoint with
`tensorrt-edgellm-export`. Run `tensorrt-edgellm-quantize` first only when you
need to create a quantized checkpoint from an FP16/BF16 source checkpoint.

```bash
# Included in the base package
tensorrt-edgellm-export --help

# Available after installing the tools extra
tensorrt-edgellm-quantize --help
tensorrt-edgellm-merge-lora --help
tensorrt-edgellm-reduce-vocab --help
```

**4. Configure HuggingFace Access (Optional)**

Some models on HuggingFace require you to accept terms before downloading.

**Models that require HuggingFace login:**
- Llama family (Llama 3.x)
- Phi-4-Multimodal
- Alpamayo-R1-10B
- Other models marked as "gated" on HuggingFace

**To configure access:**

```bash
# Install HuggingFace CLI and login
hf auth login
# Enter your HuggingFace access token when prompted
```

> **How to get a token:** Visit [HuggingFace Settings - Tokens](https://huggingface.co/settings/tokens), create a new token (read access is sufficient), and copy it.

**You're done with export pipeline setup!** You can now quantize and export models with the checkpoint-based workflow. The ONNX files will be transferred to the Edge device for runtime deployment.

---

## Part 2: C++ Runtime (Edge Device)

The C++ runtime builds TensorRT engines and runs inference on the target. For
the authoritative JetPack, DriveOS, CUDA, TensorRT, and TensorRT Edge-LLM
compatibility table, see the [Official Support Matrix](support-matrix.md).
Then use the matching platform command below for your device or SDK image.

Jetson Orin does not support FP8, MXFP8, FP4, or NVFP4 runtime precision in
this release. Use FP16, INT8, or INT4 checkpoints for Orin.

### System Requirements

- CUDA and TensorRT from the target JetPack, DriveOS SDK, or DGX Spark software release
- Disk space: ~20-50GB for ONNX files and TensorRT engines

### Build Instructions

**1. Install System Dependencies (on Edge device)**

```bash
sudo apt update
sudo apt install -y \
    cmake \
    build-essential \
    git
```

**2. Verify CUDA and TensorRT Installation**

After JetPack is installed, inside the DriveOS SDK Docker image, or on DGX
Spark, TensorRT should be installed in `/usr`.

```bash
# Check CUDA version
nvcc --version  # Should match the CUDA_CTK_VERSION for your platform below

# Check TensorRT version
dpkg -l | grep tensorrt  # Should show TensorRT 10.x+
```

**3. Clone Repository (on Edge device)**

```bash
# Clone to your chosen source directory
cd /path/to/parent-directory
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git submodule update --init --recursive
```

**4. Configure Build**

Use the CMake command for your platform. All commands enable CuTe DSL kernels
because Qwen3.5 and several other model paths require them.

**JetPack 7.0/7.1 Thor**

```bash
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor \
    -DCUDA_CTK_VERSION=13.0 \
    -DENABLE_CUTE_DSL=ALL
```

**JetPack 7.2 Thor**

```bash
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor \
    -DCUDA_CTK_VERSION=13.2 \
    -DENABLE_CUTE_DSL=ALL
```

**DriveOS 7.2 Thor**

Run this inside the DriveOS SDK Docker image, then copy `build/` to the DRIVE
system.

```bash
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=auto-thor \
    -DCUDA_CTK_VERSION=13.3 \
    -DENABLE_CUTE_DSL=ALL
```

**DGX Spark (GB10)**

Run this directly on the DGX Spark system. Use `gb10` as the embedded target
and CUDA Toolkit 13.0.

```bash
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=gb10 \
    -DCUDA_CTK_VERSION=13.0 \
    -DENABLE_CUTE_DSL=ALL
```

**JetPack 7.2 Orin**

```bash
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-orin \
    -DCUDA_CTK_VERSION=13.2 \
    -DENABLE_CUTE_DSL=ALL
```

**JetPack 6.2+ Orin**

```bash
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-orin \
    -DCUDA_CTK_VERSION=12.6 \
    -DENABLE_CUTE_DSL=ALL
```

**Alternative: Building on x86 GPU Systems (Optional for Developers)**

If you want to build and test on an x86 workstation with NVIDIA GPU (for development purposes before deploying to Edge devices), you can use this configuration instead:

```bash
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr/local/TensorRT-10.x.x \
    -DCUDA_CTK_VERSION=<YOUR_CUDA_VERSION> \
    -DCUTE_DSL_ARTIFACT_TAG=<YOUR_SM> \
    -DENABLE_CUTE_DSL=ALL
```

> **Note:** Replace `/usr/local/TensorRT-10.x.x` with your actual TensorRT installation path. Use `dpkg -l | grep tensorrt` to find it, or download from [NVIDIA TensorRT downloads](https://developer.nvidia.com/tensorrt). Replace `<YOUR_CUDA_VERSION>` with your actual CUDA version (e.g., `13.0`). Use `nvcc --version` to check your CUDA version.
> Replace `<YOUR_SM>` with the generated CuTe DSL artifact tag, for example
> `sm_80`, `sm_100`, or `sm_120`.

**CMake Options:**

| Option | Description | Default |
|:-------|:------------|:--------|
| `TRT_PACKAGE_DIR` | Path to TensorRT installation. Auto-detected; manual hint to disambiguate multiple versions. | N/A |
| `CMAKE_TOOLCHAIN_FILE` | **Required for Edge devices**: Use `cmake/aarch64_linux_toolchain.cmake` for Edge device builds. **Not needed for GPU builds** | N/A |
| `EMBEDDED_TARGET` | **Required for Edge devices**: `jetson-thor` (Jetson Thor), `auto-thor` (DRIVE Thor / DriveOS), `gb10` (DGX Spark), or `jetson-orin` (Jetson Orin). **Not needed for GPU builds** | N/A |
| `CUDA_CTK_VERSION` | CUDA Toolkit version. Use the platform command above to select `13.3`, `13.2`, `13.0`, or `12.6`. Do not pass `-DCUDA_VERSION`; CMake reserves that name for CUDA headers and rejects it. | target default |
| `BUILD_UNIT_TESTS` | Build unit tests | OFF |
| `ENABLE_COVERAGE` | Enable gcov code coverage instrumentation (see [Code Coverage](../../developer_guide/testing/code-coverage.md)) | OFF |
| `ENABLE_CUTE_DSL` | Select generated CuTe DSL kernels: `fmha`, `ALL`, or a group list such as `gdn`, `gemm`, or `ssd`. Any selection also links `fmha`, which the attention plugins require. Use `ALL` for customer builds. | fmha |
| `CUTE_DSL_ARTIFACT_TAG` | Artifact tag under `cpp/kernels/cuteDSLArtifact/<arch>/`, for example `sm_87`, `sm_110`, or `sm_121`. Edge targets infer it from `EMBEDDED_TARGET`; pass it explicitly for x86 prebuilt artifacts or when multiple local tags exist for one CPU architecture. | auto |

**CuTe DSL Kernel Artifacts**

CuTe DSL binaries are generated with `kernelSrcs/build_cutedsl.py` before
configuring CMake. A normal build defaults to the canonical `fmha` family and
therefore requires a matching artifact. This family provides Context/ViT
attention on supported GPUs and adds the optimized Blackwell implementation
on SM100/SM101/SM110 when available.

The platform commands above pass `-DENABLE_CUTE_DSL=ALL` because Qwen3.5 and
several other model paths require optional groups. Selecting a narrower group
still includes the `fmha` baseline; for example, `-DENABLE_CUTE_DSL=gdn`
enables both GDN and FMHA.

If you have multiple local artifact tags for the same CPU architecture, also
pass `-DCUTE_DSL_ARTIFACT_TAG=<tag>`.

For B200 or other SM100 build hosts without a matching prebuilt artifact, install
the CuTe DSL package expected by `kernelSrcs/build_cutedsl.py`, then generate the
artifact before running CMake:

```bash
pip install 'nvidia-cutlass-dsl==4.6.1'
python kernelSrcs/build_cutedsl.py --gpu_arch sm_100
```

For cross-compilation, pass `--arch aarch64` when the artifact must be consumed
by an AArch64 target build.

> **For supported model families, precisions, and hardware notes**, see [Supported Models](supported-models.md).

**5. Build Project**

```bash
make -j$(nproc)
```

Build time: ~1-2 minutes depending on hardware.

**6. Verify Build**

```bash
# Test C++ examples
./examples/llm/llm_build --help
./examples/llm/llm_inference --help
```

**You're done with C++ runtime setup!** You can now build engines and run inference on the Edge device.

---

## Next Steps

After installation, proceed to the [Quick Start Guide](quick-start-guide.md) for a complete end-to-end workflow, or see the [Examples](../examples/) for detailed pipeline stages and advanced use cases.

---

## Troubleshooting

### Common Installation Issues

**Issue: Python module import errors**

Solution: Activate the virtual environment and reinstall the package from the
current checkout:
```bash
source venv/bin/activate
python -m pip install -e .
tensorrt-edgellm-export --help
```

**Issue: `nvcc: command not found`**

Solution: Ensure the target JetPack release, DriveOS SDK Docker image, or DGX
Spark software stack is installed with CUDA support:
```bash
# Verify CUDA installation
nvcc --version
# Should match the CUDA_CTK_VERSION used for CMake
```

**Issue: `TensorRT not found` during CMake**

Solution: Specify TensorRT package directory. This directory should contain `lib` and `include` directories, and we are looking for the `nvinfer` library and header:
```bash
cmake .. \
    -DTRT_PACKAGE_DIR=/usr/local/TensorRT-10.x.x \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=<jetson-thor|auto-thor|gb10|jetson-orin> \
    -DCUDA_CTK_VERSION=<target CUDA version> \
    -DENABLE_CUTE_DSL=ALL
```

**Issue: Thread issue during C++ build**

Solution: Reduce parallel jobs or even use sequential build:
```bash
make -j  # Instead of make -j$(nproc)
```

### Getting Help

- **Documentation**: Check the `docs/source/developer_guide` directory
- **Issues**: Report bugs on [GitHub Issues](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/NVIDIA/TensorRT-Edge-LLM/discussions)
- **Community**: Join the NVIDIA Developer Forums

## Uninstalling

**Quantization and `tensorrt_edgellm` (x86 Host):**
- Deactivate and remove virtual environment: `deactivate && rm -rf venv`
- Remove repository (optional): `rm -rf TensorRT-Edge-LLM`

**C++ Runtime (Edge Device):**
- Remove build directory: `rm -rf build`
- Remove repository (optional): `rm -rf TensorRT-Edge-LLM`
