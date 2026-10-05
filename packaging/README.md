# TensorRT Edge-LLM wheel tooling

Public releases contain six wheels: CPython 3.10, 3.11, and 3.12, each for
`x86_64` and `aarch64`. The filename identifies the Python ABI and CPU
architecture; each wheel contains every qualified platform, CUDA/TensorRT, and
GPU-SM payload for that architecture. The installed runtime detects those
properties and loads one exact payload. It does not fall back to another SM or
TensorRT major.

The wheels package the Python APIs and
[checkpoint-direct builder](../docs/source/user_guide/getting_started/direct-engine-builder.md)
model families together with the native runtime and plugins. Model-specific
C++ executables under `experimental_models/` are built separately from source.

## Install a published wheel

Published wheels do not require a checkout or the build tools below. Follow
the [installation guide](../docs/source/user_guide/getting_started/installation.md#published-python-wheel)
to prepare CUDA, TensorRT, and a compatible Python environment, then install:

```bash
python -m pip install --only-binary=tensorrt-edgellm \
  --extra-index-url https://pypi.nvidia.com "tensorrt-edgellm[server]==0.11.0"
```

`[server]` is recommended for high-level inference. See the
[extras guide](../docs/source/user_guide/getting_started/installation.md#optional-python-dependencies)
for export/tools workflows, or the [minimal base example](../docs/source/user_guide/getting_started/installation.md#minimal-installation-advanced)
for low-level runtime use without extras. Do not install the repository package
over the published wheel.

`pip` selects the matching Python/architecture wheel. Edge-LLM then validates
the platform release, CUDA and TensorRT SONAMEs, and GPU SM when loading its
native runtime. See the
[Wheel Packaging Matrix](../docs/source/user_guide/getting_started/support-matrix.md#wheel-packaging-matrix)
for the exact payloads included in the release.

## Build wheels from source

The source tree can build a wheel for the current machine, a compatible subset
of configured GPUs, or every configured payload for one CPU architecture.
Normal package installation does not install these build-only tools.

### Prerequisites

Clone the repository with submodules and create a build environment using the
same CPython minor version as the wheel:

```bash
git clone --recurse-submodules https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
python3 -m venv --system-site-packages .venv-wheel
source .venv-wheel/bin/activate
python -m pip install -r packaging/wheel-toolchain-requirements.txt
python packaging/wheel_cli.py validate-matrix
```

The payload build also requires a compatible CUDA toolkit, TensorRT SDK, C/C++
compiler, and CuTe DSL archive. Generate the archive as documented in
[`kernelSrcs/README.md`](../kernelSrcs/README.md), or place an archive and its
`.sha256` file in `kernelSrcs/cuteDSLPrebuilt`.

### Reference x86_64 build container

`packaging/docker/Dockerfile` provides the Python and C/C++ wheel toolchain. It
does not include TensorRT or CuTe DSL archives, and its CUDA development base
must match the selected row in `packaging/variants.toml`.

Build the image from the repository root, selecting a public CUDA development
image with the required Ubuntu and CUDA versions:

```bash
export WHEEL_BASE_IMAGE=nvidia/cuda:CUDA_TAG-devel-ubuntu24.04
docker build \
    --build-arg "BASE_IMAGE=$WHEEL_BASE_IMAGE" \
    -f packaging/docker/Dockerfile \
    -t tensorrt-edgellm-wheel-builder \
    .
```

Mount the source checkout and a compatible TensorRT SDK into the container.
The matching CuTe DSL archive and checksum must already exist under
`kernelSrcs/cuteDSLPrebuilt` in the checkout:

```bash
export TRT_PACKAGE_DIR=/absolute/path/to/TensorRT
docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    -e CUDA_VISIBLE_DEVICES="GPU-<UUID>" \
    -e HOME=/tmp \
    -v "$PWD:/workspace" \
    -v "$TRT_PACKAGE_DIR:/opt/tensorrt:ro" \
    -w /workspace \
    tensorrt-edgellm-wheel-builder \
    bash -lc '
        export TRT_PACKAGE_DIR=/opt/tensorrt
        export LD_LIBRARY_PATH="$TRT_PACKAGE_DIR/lib:${LD_LIBRARY_PATH:-}"
        python packaging/wheel_cli.py build-wheel \
            --local \
            --trt-package-dir "$TRT_PACKAGE_DIR" \
            --output-dir dist/local
    '
```

This reference image supports native x86_64 builds. An aarch64 cross build
requires an appropriate platform SDK image or environment that supplies the
target toolchain, sysroot, TensorRT SDK, and Python headers; overriding
`BASE_IMAGE` alone is not sufficient.

### Build for the current target

Expose one GPU architecture and provide the TensorRT SDK root:

```bash
export TRT_PACKAGE_DIR=/path/to/TensorRT
export LD_LIBRARY_PATH="$TRT_PACKAGE_DIR/lib:${LD_LIBRARY_PATH:-}"

CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n '1p')" \
python packaging/wheel_cli.py build-wheel \
    --local \
    --trt-package-dir "$TRT_PACKAGE_DIR" \
    --output-dir dist/local
```

Local detection matches the CPU architecture, platform release, CUDA runtime,
TensorRT runtime, and visible GPU SM to one row in `packaging/variants.toml`.
On a heterogeneous host, select a GPU by the stable UUID reported by
`nvidia-smi --query-gpu=uuid,name,compute_cap --format=csv,noheader`.

On IGX Thor, the `igx-thor-cu13-sm110-sm120` row is one native payload with
SM110 and SM120 device images. Select either physical GPU before `--local`;
both selections resolve to the same build row and resulting payload.

#### Install the local wheel

Install into a clean environment on the same target configuration:

```bash
WHEEL=$(find dist/local -maxdepth 1 -name 'tensorrt_edgellm-*.whl' -print -quit)
python3 -m venv --system-site-packages .venv-install
.venv-install/bin/python -m pip install "$WHEEL"
.venv-install/bin/python -c \
    "import tensorrt_edgellm; print(tensorrt_edgellm.__version__)"
.venv-install/bin/tensorrt-edgellm-build --help
```

Build and install with the same CPython minor version. The runtime rejects a
wheel whose platform, CUDA/TensorRT ABI, or GPU architecture does not match.

### Build for selected GPUs

Repeat `--variant` to combine compatible SM payloads built with the same
platform, CUDA, TensorRT, and toolchain context:

```bash
python packaging/wheel_cli.py build-wheel \
    --variant x86-ubuntu2404-cu13-sm86 \
    --variant x86-ubuntu2404-cu13-sm100 \
    --trt-package-dir /path/to/TensorRT-10 \
    --output-dir dist/selected
```

A selected-target wheel receives a deterministic `subset` build tag and its
runtime manifest lists only the included variants. It fails with a supported-row
diagnostic on another target rather than loading an incompatible binary.

For an aarch64 cross build, also pass `--toolchain-file`, `--target-sysroot`,
and `--target-python-include-dir`. Selected variants must share one build
context; build incompatible platform or TensorRT variants separately.

The IGX Thor dual-GPU payload is already represented by one variant, so do not
repeat `--variant` for its two SMs:

```bash
python packaging/wheel_cli.py build-wheel \
    --variant igx-thor-cu13-sm110-sm120 \
    --trt-package-dir /usr \
    --output-dir dist/igx-thor
```

This produces one normal AArch64 platform wheel. Its runtime manifest contains
exact SM110 and SM120 identities that reference the same extension and plugin,
so the CuTe DSL archive and native targets are compiled and packaged once.

### Build a complete architecture wheel

A complete x86_64 or aarch64 wheel combines payloads produced in several
platform-specific SDK environments. Build and verify each matrix row with the
low-level commands below, collect their stage directories under one payload
root, and assemble them from the same clean source revision:

```bash
python packaging/wheel_cli.py build-wheel \
    --all-for-arch x86_64 \
    --payload-root /path/to/verified/payloads \
    --output-dir dist/x86_64
```

Complete assembly requires every matrix row for the requested architecture and
preserves the release-compatible wheel name. Missing, extra, stale, or
revision-mismatched payloads are rejected.

Final x86_64 wheels use `manylinux_2_35_x86_64`, matching the oldest selected
Ubuntu 22.04 payload. Final aarch64 wheels use `manylinux_2_39_aarch64`, matching
the Ubuntu 24.04 platform baseline of the configured Jetson, DRIVE, and DGX
Spark payloads. The runtime selects one exact platform payload before loading
native code, so newer mutually exclusive payloads do not raise the x86_64
installation floor. Payload verification audits ELF architecture, dependencies,
RPATHs, and target-library resolution before fan-in; release validation rejects
other platform tags before publication.

### Low-level commands

Every stage remains independently reviewable and usable for custom build
environments:

```bash
python packaging/wheel_cli.py validate-source --output artifacts/provenance/source.json
python packaging/wheel_cli.py build-base --output-dir artifacts/base/cp312
python packaging/wheel_cli.py prepare-cutedsl \
    --variant VARIANT --artifact-dir kernelSrcs/cuteDSLPrebuilt
python packaging/wheel_cli.py build-payload \
    --variant VARIANT --python-abi cp312 \
    --trt-package-dir /path/to/TensorRT \
    --output-dir artifacts/payloads/VARIANT-cp312
python packaging/wheel_cli.py verify-payload \
    --stage artifacts/payloads/VARIANT-cp312
python packaging/wheel_cli.py assemble \
    --base-wheel artifacts/base/cp312/tensorrt_edgellm-*.whl \
    --payload-root artifacts/payloads \
    --cpu-arch x86_64 --python-abi cp312 \
    --output-dir dist/x86_64/cp312
```

Pass one or more `--variant` options to `assemble` for a subset wheel. Omit the
option to require the complete architecture partition.

The build commands require clean output directories and a clean tracked source
checkout by default. Use distinct `--work-dir` and `--output-dir` paths for a
new run. `--allow-dirty-source` and `--no-device-image-check` are explicit
development overrides and must not be used for release artifacts.

## OSS release builds (internal checkout)

Wheel tooling stages a tracked-source copy, including pinned submodules, and
applies the repository OSS policy before packaging Python or compiling native
code. It does not sanitize the developer's checkout. Public source exports
retain the source-build workflow above without the internal release policy.

Internal wheel builds require separate OSS CuTe archives; ordinary internal
test archives are not accepted. In the CuTe builder environment, generate them
from the same checkout before building wheels (Git must also be installed):

```bash
python3 packaging/wheel_cli.py build-oss-cutedsl \
    --output-dir kernelSrcs/cuteDSLOssPrebuilt
```

The internal `build-wheel` default uses this directory. Each archive has a
checksum and an OSS provenance receipt binding the policy, sanitized kernel
sources, and extracted files. Policy or kernel changes require regeneration.
CI generates these separately from internal-test kernels.

Base packaging, native payload verification, final assembly, and publication
enforce the OSS checks. Final wheel checks scan all members, including metadata
and native library bytes; internal CI paths, nested build archives, and loose
kernels are rejected. Wheel builds omit CUDA device line information, which can
embed private build paths even after host debug-symbol stripping. Normal source
builds retain CUDA line information for profiling.
Legacy unstamped artifacts must be rebuilt, not relabeled.
