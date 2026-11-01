# Installation

(choose-a-deployment-workflow)=
## Choose an installation method

| Installation method | Choose this when |
|---|---|
| [Python wheel (recommended)](#published-python-wheel) | You want to run chat, an HTTP server, or the Python API on a [supported system](support-matrix.md#wheel-packaging-matrix). |
| [C++ from source](#source-workflow-c-runtime) | You want to build and run the C++ runtime, or modify its source code. |
| [Python from source](#optional-python-frontend) | You want to develop or modify Edge-LLM and use its Python interface. |
| [Build a wheel from source](#build-a-local-wheel-from-source) | You need to package a source build as an installable Python wheel. |

## Published Python wheel

Install a wheel to run Edge-LLM without downloading or compiling its source code.

Use CPython 3.10–3.12 on a target listed in the
[wheel support matrix](support-matrix.md#wheel-packaging-matrix). Install the
required NVIDIA driver, CUDA, and TensorRT, including its Python bindings,
before continuing. See the [TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)
for your supported TensorRT version. On Jetson and DRIVE, use the TensorRT
version supplied by the platform SDK.

If you extracted TensorRT into a local directory, follow the
[local SDK setup instructions](#using-tensorrt-from-a-local-sdk-directory)
to make it available in your Python environment.

(recommended-python-api-and-server)=
(install-and-chat)=
### Install

The `[server]` extra installs Python dependencies for chat and HTTP serving.
CUDA and TensorRT must already be installed.

Run the commands below. Use a supported Python interpreter for `python3`.
The TensorRT Python bindings must match that interpreter's Python version.
`--system-site-packages` makes bindings installed for that interpreter available
in the new environment; it does not provide bindings for a different Python version.

```bash
python3 -m venv --system-site-packages .venv-edgellm
source .venv-edgellm/bin/activate
python -m pip install --upgrade pip
python -m pip install "tensorrt-edgellm[server]"
```

### Verify your installation

If TensorRT is in a local SDK directory, complete the
[local SDK setup](#using-tensorrt-from-a-local-sdk-directory) in the environment
you just activated before running the check below.

To check that the Edge-LLM runtime can load without downloading a model, run:

```bash
python -c "import tensorrt; from tensorrt_edgellm import runtime; runtime.load()"
```

Successful completion confirms that the Edge-LLM runtime can load. If the check
reports no compatible payload, compare your platform, CUDA, TensorRT, and GPU
with the [wheel support matrix](support-matrix.md#wheel-packaging-matrix).
Use a listed configuration or [build from source](#source-workflow-c-runtime).
For other installation errors, see [Wheel installation issues](#wheel-installation-issues).

To download a model and run chat or inference, continue to
[Quick Start](quick-start-guide.md).

The installation above includes the Python dependencies needed for chat and
HTTP serving. For quantization, ONNX export, or build tools, see
[Optional Python dependencies](#optional-python-dependencies).

(using-a-standalone-tensorrt-sdk)=
### Using TensorRT from a local SDK directory

If TensorRT comes from a standalone SDK, install its Python wheel in your
activated environment. Choose the wheel matching your Python version and CPU
architecture:

```bash
python -m pip install "/path/to/TensorRT/python/tensorrt-<version>-<python-abi>-none-linux_<arch>.whl"
```

If the SDK's shared libraries are not on the system library path, expose them
before starting Python. Replace `/path/to/TensorRT` with the SDK directory:

```bash
export TRT_PACKAGE_DIR=/path/to/TensorRT
export LD_LIBRARY_PATH="$TRT_PACKAGE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

---

## Inference Prerequisite

The target must have at least the deployed model size plus 2 GB of available
device memory before starting inference. Treat this as a minimum: KV cache,
multimodal components, speculative draft engines, and larger batch or sequence
profiles can require additional memory.

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

**IGX Thor with an RTX SM120 GPU**

An IGX Thor system with both the integrated Thor GPU and an attached RTX
Blackwell GPU can use one plugin/runtime build. Generate one AArch64 CuTe DSL
artifact containing every supported SM110 and SM120 kernel, then compile the
CUDA sources for both architectures in one CMake build:

```bash
python kernelSrcs/build_cutedsl.py \
    --kernels ALL \
    --gpu_arch sm_110,sm_120 \
    --arch aarch64 \
    --cuda-version 13 \
    --clean

cmake -S . -B build-igx-thor \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=igx-thor \
    -DCUDA_CTK_VERSION=13.0 \
    -DENABLE_CUTE_DSL=ALL

cmake --build build-igx-thor --parallel
```

`EMBEDDED_TARGET=igx-thor` compiles the CUDA sources for `110a` and `120`,
requires CUDA 13 or newer, and selects the combined `sm_110_sm_120` CuTe DSL
artifact. On a native IGX host, set
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, then use
`CUDA_VISIBLE_DEVICES=0` for Thor or `CUDA_VISIBLE_DEVICES=1` for RTX. Confirm
the selection before building or running an engine:

```bash
python3 -c 'import torch; print(torch.cuda.get_device_name(0))'
```

In an NVIDIA container use `NVIDIA_VISIBLE_DEVICES=0` or
`NVIDIA_VISIBLE_DEVICES=1`. The selected physical GPU appears as CUDA device 0
inside the container, so do not copy the physical ordinal into
`CUDA_VISIBLE_DEVICES` there. GPU selection is fixed for the life of an
inference process. To switch GPUs, stop the process, update the selector, use
the engine built for the selected SM, and restart. TensorRT engines remain
architecture specific even though the plugin binary is shared.

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
    -DCUTE_DSL_ARTIFACT_TAG=sm_<YOUR_SM> \
    -DENABLE_CUTE_DSL=ALL
```

> **Note:** Replace `/usr/local/TensorRT-10.x.x` with your actual TensorRT installation path. Use `dpkg -l | grep tensorrt` to find it, or download from [NVIDIA TensorRT downloads](https://developer.nvidia.com/tensorrt). Replace `<YOUR_CUDA_VERSION>` with your actual CUDA version (e.g., `13.0`). Use `nvcc --version` to check your CUDA version.
> Replace `<YOUR_SM>` with the numeric compute capability, for example `80`,
> `100`, or `120`.

<a id="windows-x64-with-tensorrt-rtx"></a>

**Windows x64 with TensorRT-RTX**

Run CMake from an x64 Native Tools Command Prompt for Visual Studio:

```powershell
py -m venv .venv-cutedsl
.\.venv-cutedsl\Scripts\python.exe -m pip install `
    --no-cache-dir `
    --only-binary=:all: `
    --index-url https://pypi.org/simple `
    "nvidia-cutlass-dsl[cu13]==4.8.0" `
    cupy-cuda13x==13.6.0
.\.venv-cutedsl\Scripts\python.exe kernelSrcs\build_cutedsl.py `
    --kernels ALL `
    --gpu_arch sm_<YOUR_SM> `
    --cuda-version <YOUR_CUDA_VERSION> `
    --clean

cmake -S . -B build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DTRT_PACKAGE_DIR=C:/path/to/tensorrt-rtx `
    -DCUDA_CTK_VERSION=<YOUR_CUDA_VERSION> `
    -DCMAKE_CUDA_ARCHITECTURES=<YOUR_SM> `
    -DENABLE_CUTE_DSL=ALL `
    -DCUTE_DSL_ARTIFACT_TAG=sm_<YOUR_SM>
cmake --build build --parallel
```

Replace `<YOUR_CUDA_VERSION>` with the installed CUDA Toolkit version, for
example `13.2`, and `<YOUR_SM>` with the numeric compute capability, for example
`120`. `13.6.0` in the pip command is the CuPy release version; `cuda13x` selects
its CUDA 13.x package. Generate the Windows CuTe DSL artifact with all kernel
groups before running CMake. Windows artifacts contain COFF objects and are
stored under `cpp/kernels/cuteDSLArtifact/windows/x86_64/<tag>`; Linux ELF
artifacts remain under `cpp/kernels/cuteDSLArtifact/x86_64/<tag>`. The
`llm_stream` example is not built on Windows.

The TensorRT-RTX package must include its ONNX parser component. Add the
TensorRT-RTX `bin` directory to `PATH` when running the examples so the runtime,
ONNX parser, and Edge-LLM plugin DLLs can be loaded.

**CMake Options:**

| Option | Description | Default |
|:-------|:------------|:--------|
| `TRT_PACKAGE_DIR` | Path to TensorRT installation. Auto-detected; manual hint to disambiguate multiple versions. | N/A |
| `CMAKE_TOOLCHAIN_FILE` | **Required for Edge devices**: Use `cmake/aarch64_linux_toolchain.cmake` for Edge device builds. **Not needed for GPU builds** | N/A |
| `EMBEDDED_TARGET` | **Required for Edge devices**: `jetson-thor` (Jetson Thor), `igx-thor` (IGX Thor plus RTX SM120), `auto-thor` (DRIVE Thor / DriveOS), `gb10` (DGX Spark), or `jetson-orin` (Jetson Orin). **Not needed for GPU builds** | N/A |
| `CUDA_CTK_VERSION` | CUDA Toolkit version. Use the platform command above to select `13.3`, `13.2`, or `13.0`. Do not pass `-DCUDA_VERSION`; CMake reserves that name for CUDA headers and rejects it. | target default |
| `BUILD_UNIT_TESTS` | Build unit tests | OFF |
| `ENABLE_COVERAGE` | Enable gcov code coverage instrumentation (see [Code Coverage](../../developer_guide/testing/code-coverage.md)) | OFF |
| `ENABLE_CUTE_DSL` | Select generated CuTe DSL kernels: `fmha`, `ALL`, or a group list such as `gdn`, `gemm`, or `ssd`. Any selection also links `fmha`, which the attention plugins require. Use `ALL` for customer builds. | fmha |
| `CUTE_DSL_ARTIFACT_TAG` | Artifact tag under `cpp/kernels/cuteDSLArtifact/<arch>/` (`cpp/kernels/cuteDSLArtifact/windows/<arch>/` on Windows), for example `sm_87`, `sm_110`, `sm_110_sm_120`, or `sm_121`. Edge targets infer it from `EMBEDDED_TARGET`; pass it explicitly for x86 prebuilt artifacts or when multiple local tags exist for one CPU architecture. | auto |

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
pip install 'nvidia-cutlass-dsl==4.8.0'
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
trtedge serve Qwen/Qwen3.5-0.8B
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

For other tasks, choose the extras listed in
[Optional Python dependencies](#optional-python-dependencies).

The `tools` extra includes the export dependencies. Checkpoint-direct engine
build and Python inference use the optional Python frontend above.

> **Note:** Accuracy evaluation dependencies live under `examples/accuracy/requirements.txt`.

**3. Verify the Checkpoint Export Workflow**

Use the virtual environment created in Step 2 for this checkout. Do not mix
packages from older release branches into the same environment.

Export an unquantized or supported pre-quantized Hugging Face checkpoint with
`trtedge export`. Run `trtedge quantize` first only when you
need to create a quantized checkpoint from an FP16/BF16 source checkpoint.

```bash
# Available after installing the export extra
trtedge export --help

# Available after installing the tools extra
trtedge quantize --help
trtedge merge-lora --help
trtedge reduce-vocab --help
```

**4. Configure Hugging Face Access (Optional)**

Some Hugging Face checkpoints require accepting the provider's terms before
download. After accepting those terms, authenticate with a read token:

```bash
hf auth login
```

> **How to get a token:** Visit [Hugging Face Settings - Tokens](https://huggingface.co/settings/tokens) and create a read token.

The environment is now ready to quantize or export a supported checkpoint.

---

## Build a local wheel from source

Use this path to package the current checkout for one detected target. Unlike a
published architecture wheel, `--local` includes only the exact platform,
CUDA/TensorRT, and GPU payload detected during the build.

Clone the repository with submodules and generate the matching CuTe DSL archive
as described in the repository's
[Wheel Tooling](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/packaging/README.md)
guide. Then build and install the wheel:

```bash
python3 -m venv --system-site-packages .venv-wheel
source .venv-wheel/bin/activate
python -m pip install -r packaging/wheel-toolchain-requirements.txt

export TRT_PACKAGE_DIR=/usr  # Use the TensorRT SDK root on this target.
python packaging/wheel_cli.py build-wheel \
    --local \
    --trt-package-dir "$TRT_PACKAGE_DIR" \
    --output-dir dist/local

WHEEL=$(find dist/local -maxdepth 1 -name 'tensorrt_edgellm-*.whl' -print -quit)
python3 -m venv --system-site-packages .venv-install
.venv-install/bin/python -m pip install "$WHEEL"
.venv-install/bin/python -c \
    "import tensorrt_edgellm; print(tensorrt_edgellm.__version__)"
.venv-install/bin/trtedge build --help
```

The local wheel supports only the detected platform release, CPU architecture,
CUDA/TensorRT ABI, GPU architecture, and Python ABI. Use a standalone TensorRT
SDK root instead of `/usr` on an x86 workstation.

---

(install-additional-tools)=
## Optional Python dependencies

Extras are optional groups of Python dependencies available with both wheel
and source installations. They do not install CUDA or TensorRT.

The recommended wheel installation already includes `[server]`. No other
extras are needed for chat, model downloads, the high-level Python API, or
HTTP serving.

| Install selection | When you need it |
|---|---|
| `[server]` | Use `trtedge chat`, `trtedge serve`, model downloads, or the high-level `LLM` API. |
| `[export]` | Export checkpoints to ONNX. |
| `[tools]` | Export and quantize checkpoints, or use LoRA, vocabulary, and audio tools. |
| `[server-tools]` | Generate missing tokenizer artifacts or use Transformers-based tools. |
| `[native-build]` | Build Python bindings from source. |
| [No extra](#install-without-optional-dependencies) | Call the low-level runtime or build engines directly from complete local checkpoints. |

The `[tools]` extra includes the export dependencies. The `[native-build]`
extra is not needed for the automatic engine building performed by
`trtedge chat` and `trtedge serve` with a published wheel.
`[builder]` and `[dev]` are compatibility extras that add no dependencies.

Combine extras in one install or add them later. To add quantization and ONNX
export dependencies to the wheel installation above, run:

```bash
python -m pip install "tensorrt-edgellm[server,tools]"
```

For a source installation, select extras after `.` instead of the package
name. For example, from the source checkout:

```bash
python -m pip install -e ".[tools]"
```

Installing `.` or using `-e .` switches to a source installation; use these
commands only when following the source installation instructions.

(minimal-installation-advanced)=
(use-the-native-runtime-directly)=
### Install without optional dependencies

Use this option for applications that call the C++ runtime through Python and
use local model files. It omits the optional dependencies used by chat, model
downloads, and HTTP serving. For those features, use the
[published wheel installation](#published-python-wheel) with `[server]`.

Keep the same CUDA and TensorRT prerequisites. Create a separate environment
so packages from an earlier installation are not included:

```bash
python3 -m venv .venv-edgellm-base
source .venv-edgellm-base/bin/activate
python -m pip install "tensorrt-edgellm"
```

Install the matching TensorRT Python bindings in this environment. If TensorRT
is in a local SDK directory, follow the
[local SDK setup instructions](#using-tensorrt-from-a-local-sdk-directory),
including the library path setup if needed.

To build an engine from local model files or use an existing engine, follow
[Run inference with the Python runtime](../examples/python-runtime.md).

---

## Next Steps

Follow the [Quick Start Guide](quick-start-guide.md) to start a chat or launch
an HTTP server. The guide also covers inference with the C++ runtime.
For model-specific examples, see [Examples](../examples/index.md).

---

## Troubleshooting

### Wheel installation issues

**Virtual environment creation fails because `ensurepip` is unavailable**

On Debian or Ubuntu, install the venv package matching your Python interpreter.
For Python 3.12, run:

```bash
sudo apt update
sudo apt install -y python3.12-venv
```

Then repeat the [install commands](#install).

**`trtedge: command not found`**

Activate the environment where you installed the wheel. If you followed the
install commands above, run this from the directory containing `.venv-edgellm`:

```bash
source .venv-edgellm/bin/activate
```

If the command is still unavailable, repeat the [wheel installation](#install)
in that environment.

**`No module named tensorrt`**

Install TensorRT's Python bindings for the Python version used by the active
environment. To use system-installed bindings, create the environment with
`--system-site-packages` using the matching interpreter. For bindings supplied
with a local SDK, follow the [SDK setup instructions](#using-tensorrt-from-a-local-sdk-directory).

**No matching distribution for `tensorrt-edgellm`**

Check your Python version, CPU architecture, and operating system against the
[wheel support matrix](support-matrix.md#wheel-packaging-matrix). Use the
[install commands](#install) to update pip and retry the installation.
If your configuration is not supported by a published wheel, follow the
[source installation instructions](#source-workflow-c-runtime).

**The runtime cannot find or load a compatible compiled component**

Compare your GPU, platform or SDK release, CUDA, and TensorRT versions with the
[wheel support matrix](support-matrix.md#wheel-packaging-matrix). A wheel can
install successfully even when its compiled components do not match your
runtime environment. For missing TensorRT shared libraries from a local SDK,
follow the [library path setup](#using-tensorrt-from-a-local-sdk-directory).
Use a supported configuration or [build from source](#source-workflow-c-runtime).

(common-installation-issues)=
### Source installation issues

**Issue: Python module import errors in a source installation**

Solution: Activate the virtual environment and reinstall the package from the
current checkout:
```bash
source venv/bin/activate
python -m pip install -e .
trtedge export --help
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
    -DEMBEDDED_TARGET=<jetson-thor|igx-thor|auto-thor|gb10|jetson-orin> \
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
