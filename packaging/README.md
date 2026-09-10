# TensorRT Edge-LLM wheel tooling

The source tree can build a wheel for the current machine, a compatible subset
of configured GPUs, or every configured payload for one CPU architecture.
Normal package installation does not install these build-only tools.

## Prerequisites

Clone the repository with submodules and create a build environment using the
same CPython minor version as the wheel:

```bash
git clone --recurse-submodules https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
python3.12 -m venv .venv-wheel
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

## Build for the current target

Expose one GPU architecture and provide the TensorRT SDK root:

```bash
export TRT_PACKAGE_DIR=/path/to/TensorRT
export LD_LIBRARY_PATH="$TRT_PACKAGE_DIR/lib:${LD_LIBRARY_PATH:-}"

CUDA_VISIBLE_DEVICES=GPU-<UUID> \
python packaging/wheel_cli.py build-wheel \
    --local \
    --trt-package-dir "$TRT_PACKAGE_DIR" \
    --output-dir dist/local
```

Local detection matches the CPU architecture, platform release, CUDA runtime,
TensorRT runtime, and visible GPU SM to one row in `packaging/variants.toml`.
On a heterogeneous host, select a GPU by the stable UUID reported by
`nvidia-smi --query-gpu=uuid,name,compute_cap --format=csv,noheader`.

## Build for selected GPUs

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

## Build a complete architecture wheel

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

## Low-level commands

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
