# Installation

## Choose a deployment workflow

| Workflow | Use it when | Data flow |
|---|---|---|
| [C++ source deployment](#source-workflow-c-runtime) | The application uses the supported ONNX export, engine build, and C++ runtime workflow. | Hugging Face checkpoint → optional quantization → ONNX export → C++ engine build → C++ inference |
| [Python from source](#optional-python-frontend) | The application uses the experimental checkpoint-direct builder or Python server from the same source and build tree. | Hugging Face checkpoint → checkpoint-direct builder → TensorRT engine → Python inference |
| [Experimental local wheel](#experimental-local-wheel) | A developer needs to evaluate a relocatable Python installation on the current target. | Local source build → target-specific wheel → Python inference |

## Source workflow: C++ runtime

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

#### Optional Python frontend

Skip this step for the C++ and ONNX workflow. To also use the experimental
checkpoint-direct builder or OpenAI-compatible server, create the Python
environment and install the binding build dependency before configuring CMake:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install pybind11==3.0.4
```

Retain every argument from the complete platform command in Step 4 and append
`-DBUILD_PYTHON_BINDINGS=ON` and
`-Dpybind11_DIR="$(python -m pybind11 --cmakedir)"`. The directory argument is
required because pip installs the pybind11 CMake configuration outside CMake's
default search prefixes.

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

**QNX Standard 8.0 (AArch64 cross-compilation)**

QNX is a C++ source-deployment workflow. Install the QNX SDP 8.0 host and
target trees, a cross-capable host CUDA Toolkit, the matching QNX CUDA target
package, and a QNX TensorRT package. `QNX_HOST` must contain the host `qcc` and
`q++` tools; `QNX_TARGET` is the AArch64 QNX sysroot. The TensorRT root passed
to CMake must expose target headers and libraries under `include` and `lib`, or
under the `include/aarch64-qnx` and `lib/aarch64-qnx` subdirectories.

```bash
export QNX_HOST=/path/to/qnx800/host/linux/x86_64
export QNX_TARGET=/path/to/qnx800/target/qnx

cmake -S . -B build-qnx \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_qnx_toolchain.cmake \
    -DTRT_PACKAGE_DIR=/path/to/tensorrt-qnx \
    -DCUDA_CTK_VERSION=13.3 \
    -DCUDA_TOOLKIT_ROOT=/usr/local/cuda-13.3 \
    -DQNX_CUDA_TARGET_ROOT=/usr/local/cuda-safe-13.3 \
    -DENABLE_CUTE_DSL=OFF

cmake --build build-qnx --parallel "$(nproc)"
```

`QNX_CUDA_TARGET_ROOT` must contain `targets/aarch64-qnx`. The toolchain also
uses `${QNX_CUDA_TARGET_ROOT}/thor/targets/aarch64-qnx` by default; override
`CUDA_TARGET_DIR` when that additional target tree is elsewhere. CUDA Toolkit
13.x defaults to SM110a. CUDA Toolkit 12.7 through 12.x defaults to SM101a;
set `CMAKE_CUDA_ARCHITECTURES` explicitly for another supported target.

Deploy the cross-built binaries and libraries from `build-qnx/` with the
matching QNX CUDA and TensorRT runtime libraries. CuTe DSL kernels are not
available for QNX; CMake rejects `ENABLE_CUTE_DSL` values other than `OFF`.
The standard autoregressive LLM and VLM paths require CuTe DSL FMHA for
prefill, so their `llm_build` and `llm_inference` workflows are not supported
by this QNX build. Components implemented entirely with TensorRT-native or
CUDA operators can be cross-compiled, but this release does not claim a
model-level QNX qualification for them. Python wheels and the experimental
Python server are not part of this cross-compilation workflow.

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
| `CUDA_CTK_VERSION` | CUDA Toolkit version. Use the platform command above to select `13.3`, `13.2`, or `13.0`. Do not pass `-DCUDA_VERSION`; CMake reserves that name for CUDA headers and rejects it. | target default |
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
pip install 'nvidia-cutlass-dsl==4.7.0'
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

#### Install and launch the Python server

If you enabled the optional Python frontend, install Edge-LLM and the server
dependencies after building the native bindings. Run the install and server
from the source checkout because the native artifacts remain in its build
directory. The editable install ensures that the server command resolves those
artifacts from the checkout. The server accepts a model ID or local checkpoint
and builds its engines on first use:

```bash
cd /path/to/TensorRT-Edge-LLM
source .venv/bin/activate
python -m pip install -e ".[server,server-tools]"
tensorrt-edgellm-serve Qwen/Qwen3.5-0.8B
```

See [Experimental Python API and Server](../examples/experimental-server.md)
for server options, requests, and limitations.

---

## Source workflow: export and quantization

The Python frontend exports Hugging Face checkpoints and optionally quantizes
FP16/BF16 checkpoints before export. Export runs on CPU. Quantization requires
an NVIDIA GPU.

### System Requirements

- **Platform**: x86-64 Linux system
- **Recommended OS**: Ubuntu 22.04, 24.04
- **GPU for quantization**: NVIDIA GPU with Compute Capability 8.0+ (Ampere or newer)
- **CUDA for quantization**: 12.x or 13.x
- **TensorRT**: matching Python package and runtime libraries
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

# Check the preinstalled TensorRT Python package
python3 -c "import tensorrt as trt; print(trt.__version__)"
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

Install the dependency set for the host-side ONNX workflow:

```bash
# PyTorch/ONNX checkpoint exporter
pip3 install -e ".[export]"

# Export plus quantization, LoRA, vocabulary, and audio tools
pip3 install -e ".[tools]"
```

The `tools` extra remains a superset of `export`. Checkpoint-direct engine build
and Python inference use the optional Python frontend above and remain separate
from this source-export procedure.

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

## Experimental local wheel

Wheels are not published or the default installation path in 0.10.1. To
evaluate a target-specific wheel locally, install the packaging requirements
and run the local builder:

```bash
python -m pip install -r packaging/wheel-toolchain-requirements.txt
python packaging/wheel_cli.py build-wheel \
    --local \
    --trt-package-dir /path/to/TensorRT \
    --output-dir dist/local
python -m pip install dist/local/tensorrt_edgellm-*.whl
```

This experimental path requires a matching unpublished CuTe DSL tarball and
checksum under `kernelSrcs/cuteDSLPrebuilt/`; see `packaging/README.md` in the
source checkout. The resulting wheel supports only the detected target and is
not a general release artifact.

---

## Next Steps

For the maintained ONNX and C++ workflow, proceed to the
[Quick Start Guide](quick-start-guide.md). For model-specific input and output
contracts, see [Examples](../examples/index.md).

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
