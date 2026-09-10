# CuTe DSL Artifact Generation

`kernelSrcs/build_cutedsl.py` is the unified entry point for generating local
CuTe DSL artifacts consumed by `cmake/CuteDsl.cmake`.

The dependency, artifact-generation, CMake, and cross-compile steps documented
here are **shared by all CuTe DSL kernel groups**. Group READMEs cover only
kernel-specific behavior, variant coverage, and standalone testing (see
[Group-Specific Details](#group-specific-details)).

## What It Generates

Running the script produces a local artifact directory under:

```text
cpp/kernels/cuteDSLArtifact/<arch>/<artifact_tag>/
  libcutedsl_<arch>.a
  metadata.json
  include/
    cutedsl_all.h
    ...
```

`artifact_tag` is currently `sm_<NN>`, for example `sm_80`, `sm_110`, or
`sm_121`.

These artifacts are local build inputs for CMake. They are not intended to be
checked into git by default.

## Dependencies

These dependencies are required **only** when you regenerate an artifact through
the [CuTe DSL kernel development workflow](#cute-dsl-kernel-development-workflow).
A normal CuTe DSL-enabled build consumes a matching prebuilt artifact from
`kernelSrcs/cuteDSLPrebuilt/` and needs no Python environment or GPU.

Use a separate virtual environment for each CuTe DSL compiler flavor. See
[Option B: Incremental Build in a Local Virtual Environment](#option-b-incremental-build-in-a-local-virtual-environment)
for installation commands and CUDA 12/13 host-runtime guidance.

| Dependency | Version | Notes |
|---|---|---|
| `nvidia-cutlass-dsl` | `4.7.0` | Pinned; isolate cu12 and cu13 compiler backends by environment |
| `cupy-cuda12x` | `12.3.0` | CUDA 12 build host |
| `cupy-cuda13x` | `13.6.0` | CUDA 13 build host |
| `cuda-python` | matches the build host | Required by every group's AOT export |

## Common Commands

From the repository root:

```bash
# Build all groups supported by the current GPU
python kernelSrcs/build_cutedsl.py

# Build one group for a specific target SM
python kernelSrcs/build_cutedsl.py --kernels gdn --gpu_arch sm_87
python kernelSrcs/build_cutedsl.py --kernels fmha --gpu_arch sm_110 --arch aarch64
python kernelSrcs/build_cutedsl.py --kernels gemm --gpu_arch sm_121 --arch aarch64
```

## Docker Artifact Builder

The image build installs dependencies only. Kernel generation runs under
`docker run` because the AOT scripts require a visible GPU:

```bash
# Context = kernelSrcs/ (small, self-contained); the repo root would drag
# local build trees into the docker context.
CUTE_DSL_BUILDER_VERSION=4.7.0
docker build \
  -f kernelSrcs/Dockerfile.cutedsl \
  --build-arg "CUTE_DSL_BUILDER_VERSION=${CUTE_DSL_BUILDER_VERSION}" \
  -t "tensorrt-edge-llm/cutedsl-kernel-builder:${CUTE_DSL_BUILDER_VERSION}" \
  kernelSrcs

mkdir -p cutedsl-out
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/cutedsl-out:/out" \
  "tensorrt-edge-llm/cutedsl-kernel-builder:${CUTE_DSL_BUILDER_VERSION}"
```

The default matrix generates these tarballs and matching `.sha256` files under
`cutedsl-out/`:

```text
cutedsl_x86_64_sm_80_cuda13.tar.gz
cutedsl_x86_64_sm_86_cuda13.tar.gz
cutedsl_x86_64_sm_90_cuda13.tar.gz
cutedsl_x86_64_sm_100_cuda13.tar.gz
cutedsl_x86_64_sm_120_cuda13.tar.gz
cutedsl_x86_64_sm_120_cuda12.tar.gz
cutedsl_aarch64_sm_87_cuda13.tar.gz
cutedsl_aarch64_sm_90_cuda13.tar.gz
cutedsl_aarch64_sm_101_cuda12.tar.gz
cutedsl_aarch64_sm_110_cuda13.tar.gz
cutedsl_aarch64_sm_121_cuda12.tar.gz
cutedsl_aarch64_sm_121_cuda13.tar.gz
```

List the configured outputs without a GPU, or override the matrix:

```bash
docker run --rm \
  "tensorrt-edge-llm/cutedsl-kernel-builder:${CUTE_DSL_BUILDER_VERSION}" \
  --list

docker run --rm --gpus all \
  -v "$PWD/cutedsl-out:/out" \
  -e CUTE_DSL_MATRIX="x86_64:sm_100:12,aarch64:sm_110:13" \
  -e CUTE_DSL_JOBS=2 \
  "tensorrt-edge-llm/cutedsl-kernel-builder:${CUTE_DSL_BUILDER_VERSION}"
```

Expanded build directories are retained under
`cutedsl-out/artifacts/cuda<MAJOR>/<arch>/<sm>/`. Each tarball is validated for
metadata consistency, target ELF architecture, layout, and checksum before the
builder exits successfully.

## CuTe DSL Kernel Development Workflow

Use this workflow only when you change a CuTe DSL kernel or its registry entry.
Normal builds skip it entirely: CMake extracts the matching prebuilt tarball from
`kernelSrcs/cuteDSLPrebuilt/` automatically. A prebuilt tarball does not contain
your source changes, so it is never a valid way to test them.

Pick one of the two manual build environments below. For a new or clean
artifact, generate `fmha` together with the affected group into
`cpp/kernels/cuteDSLArtifact/<arch>/<sm>/`, then configure CMake and run the
relevant build and runtime tests. The remaining pieces of the workflow live in
dedicated sections:

- Dependency versions — [Dependencies](#dependencies)
- Artifact layout — [What It Generates](#what-it-generates)
- `build_cutedsl.py` flags — [Script Options](#script-options)
- CMake configuration — [CMake Integration](#cmake-integration)
- Cross-compilation (AArch64 / Thor) — [Cross-Compiling for AArch64 (Thor) and Runtime Deployment](#cross-compiling-for-aarch64-thor-and-runtime-deployment)

Use `--clean` for a first build, after changing the variant registry, or before
publishing an artifact. During the inner loop, omit `--clean` so
`build_cutedsl.py` can replace rebuilt archive members, refresh generated
headers, and merge compatible metadata without rebuilding unrelated groups.

### Option A: Incremental Build in Docker (Recommended)

Build the image once. Bind-mount `kernelSrcs/` so subsequent source edits do not
require rebuilding the image, and bind-mount the expanded artifact directory so
the generated objects and headers update the tree consumed by CMake:

```bash
CUTE_DSL_BUILDER_VERSION=4.7.0
docker build \
  -f kernelSrcs/Dockerfile.cutedsl \
  --build-arg "CUTE_DSL_BUILDER_VERSION=${CUTE_DSL_BUILDER_VERSION}" \
  -t "tensorrt-edge-llm/cutedsl-kernel-builder:${CUTE_DSL_BUILDER_VERSION}" \
  kernelSrcs

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/kernelSrcs:/workspace/kernelSrcs:ro" \
  -v "$PWD/cpp/kernels/cuteDSLArtifact:/artifacts" \
  --entrypoint /opt/cutedsl/venvs/cu13/bin/python \
  "tensorrt-edge-llm/cutedsl-kernel-builder:${CUTE_DSL_BUILDER_VERSION}" \
  kernelSrcs/build_cutedsl.py \
    --kernels fmha \
    --gpu_arch sm_120 \
    --arch x86_64 \
    --cuda-version 13 \
    --output_dir /artifacts
```

For a non-FMHA change in a new or clean output, use
`--kernels fmha,<affected-group>` so the artifact remains runnable. If the
expanded artifact already contains the complete FMHA baseline, an incremental
build may select only the affected group and omit `--clean`; metadata is merged.
Selection is group-level, not individual-variant-level. For a CUDA 12 artifact,
use the `cu12` interpreter, pass `--cuda-version 12`, and configure CMake with
the matching CUDA 12 toolchain before rebuilding the group.

### Option B: Incremental Build in a Local Virtual Environment

Create one virtual environment per CuTe DSL compiler flavor. The following
example builds a CUDA 13 artifact on a CUDA 13 host:

```bash
python3 -m venv .venv-cutedsl-cu13
source .venv-cutedsl-cu13/bin/activate
python -m pip install --upgrade pip wheel

export CUTE_DSL_VERSION=4.7.0
python -m pip install \
  "nvidia-cutlass-dsl[cu13]==${CUTE_DSL_VERSION}" \
  cupy-cuda13x==13.6.0 \
  cuda-python

python kernelSrcs/build_cutedsl.py \
  --kernels fmha \
  --gpu_arch sm_120 \
  --arch x86_64 \
  --cuda-version 13 \
  --output_dir cpp/kernels/cuteDSLArtifact
```

For CUDA 12 artifacts, create a separate environment with the `cu12` extra.
CuPy must match the **build host**: use `cupy-cuda12x` on a CUDA 12 host or
`cupy-cuda13x` on a CUDA 13 host. `--cuda-version 12` selects the CUDA 12 CuTe
DSL compiler/runtime flavor independently of the host CuPy runtime.

The original tarball does not need to change for local development. If a
modified artifact must be shared, create a new tarball and checksum from the
expanded directory:

```bash
tar -C cpp/kernels/cuteDSLArtifact/x86_64 \
  -czf cutedsl_x86_64_sm_120_cuda13.tar.gz sm_120
sha256sum cutedsl_x86_64_sm_120_cuda13.tar.gz \
  > cutedsl_x86_64_sm_120_cuda13.tar.gz.sha256
```

Use incremental rebuilds only for the developer inner loop. If variants are
removed or renamed, or if an artifact will be published for CI/release, perform
a clean full-matrix rebuild so stale archive members cannot be retained.

## Script Options

| Flag | Default | Description |
|---|---|---|
| `--kernels GROUPS` | `ALL` | A registered group such as `f16_moe`, `fmha`, `gdn`, `gemm`, `gemm_nvfp4`, `int4_fp16_gemm`, `nvfp4_moe`, `nvfp4_fused_moe`, or `ssd`; a comma-separated list; or `ALL`. `fmha` is the attention family: the FMHA-v2 kernels plus the optimized Blackwell kernels on SM100/SM101/SM110. Variants whose `supported_sms` excludes the target SM are skipped. |
| `--gpu_arch SM` | auto-detected | Target GPU SM (e.g. `sm_100`); auto-detected via cupy / nvidia-smi when omitted. The CuTe DSL compile architecture is derived automatically, including the required Blackwell `a` suffix. |
| `--arch ARCH` | auto-detected | Target CPU arch `x86_64` or `aarch64`. If it differs from the build host, kernels are cross-compiled (target host objects). |
| `--cuda-version VERSION` | host CUDA | Artifact CUDA flavor used to select `cu12` or `cu13` runtime objects. |
| `--runtime-libs-version VERSION` | CuTe DSL package version | Target-architecture runtime-libs wheel version; use when compiler and runtime-libs package versions differ. |
| `--output_dir DIR` | `cpp/kernels/cuteDSLArtifact` | Root output dir (artifacts go under `{DIR}/{arch}/sm_<NN>/`). |
| `-j JOBS` | CPU count | Parallel compile jobs, defaulting to the CPUs available to the process (use `-j 1` if GPU memory is limited). |
| `--verbose` | off | Show per-variant kernel script output. |
| `--clean` | off | Remove the selected target artifact directory before building. |

## CMake Integration

Enable one or more groups with `ENABLE_CUTE_DSL`:

```bash
cmake .. -DENABLE_CUTE_DSL=fmha
cmake .. -DENABLE_CUTE_DSL=gdn
cmake .. -DENABLE_CUTE_DSL=gemm
cmake .. -DENABLE_CUTE_DSL="fmha;gdn;gemm"
```

`ENABLE_CUTE_DSL` defaults to `fmha`, the Context/ViT attention kernels. On
SM100/SM101/SM110 the same family also activates the optimized Blackwell
kernels when its variants are present. Every selection implicitly includes
`fmha`, because the attention runner is compiled unconditionally.

`cmake/CuteDsl.cmake` then:

1. Detects the host/target CPU architecture.
2. Resolves the artifact tag from `CUTE_DSL_ARTIFACT_TAG` or the target platform
   when unambiguous.
3. Reads `metadata.json` to determine which groups/variants are available and
   sets the matching compile definitions.
4. Validates `libcutedsl_{arch}.a` and `include/cutedsl_all.h` exist, then links
   the self-contained kernel archive. The CuTe DSL static runtime shim objects
   are already embedded in that archive, so CMake does not link
   `libcute_dsl_runtime.so`.

If the expanded artifact is missing, CMake automatically extracts an
architecture-, SM-, and CUDA-major-matched tarball from
`kernelSrcs/cuteDSLPrebuilt/`. If neither form exists, configuration fails with
a command for `build_cutedsl.py`. CMake never invokes Python or compiles a CuTe
DSL kernel.

Release packages can include prebuilt artifacts for the deployment matrix
listed under [Docker Artifact Builder](#docker-artifact-builder).

When the same CPU architecture has multiple local artifact tags, select one
explicitly:

```bash
cmake .. -DENABLE_CUTE_DSL=gemm -DCUTE_DSL_ARTIFACT_TAG=sm_110
cmake .. -DENABLE_CUTE_DSL=gemm -DCUTE_DSL_ARTIFACT_TAG=sm_121
```

`EMBEDDED_TARGET=gb10`, `auto-thor`, `jetson-thor`, and `jetson-orin` map to a
default artifact tag when unambiguous. `thor-all` requires an explicit
`CUTE_DSL_ARTIFACT_TAG`.

## Cross-Compiling for AArch64 (Thor) and Runtime Deployment

When `--arch` differs from the build host (e.g. `--arch aarch64` on an x86_64
machine), `build_cutedsl.py`:

- compiles the kernels for the target via the CuTe DSL `--host-target`
  (`linux-aarch64`), and
- embeds the target-arch CuTe DSL static runtime shim objects into
  `libcutedsl_<arch>.a`. Native builds use the installed package's
  `libcuda_dialect_runtime_static.a` directly. Cross builds download the
  target-arch `nvidia-cutlass-dsl-libs-cuXX` wheel with `--no-deps`, extract its
  `libcuda_dialect_runtime_static.a`, and merge those objects into the archive.

Because the target-arch static runtime shim is embedded in the artifact, the
link and target runtime environments do **not** need `nvidia-cutlass-dsl` or
`libcute_dsl_runtime.so` to use a relocated/prebuilt artifact (e.g. an extracted
tarball in a CI job or Docker build).

Example cross build + CMake configure for Thor:

```bash
python kernelSrcs/build_cutedsl.py --kernels fmha --gpu_arch sm_110 --arch aarch64

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=auto-thor \
    -DENABLE_CUTE_DSL=fmha
```

## Group-Specific Details

See the group-specific READMEs for kernel coverage and standalone testing:

- `kernelSrcs/fmha_cutedsl_blackwell/README.md` — optimized Blackwell/Thor fused multi-head attention (LLM + ViT)
- `kernelSrcs/f16_moe_cutedsl/README.md` — Cross-platform FP16 grouped MoE GEMM
- `kernelSrcs/gdn_cutedsl/README.md` — Gated Delta Net decode/prefill
- `kernelSrcs/ssd_cutedsl/README.md` — Mamba2 SSD prefill
- `kernelSrcs/gemm_cutedsl/README.md` — FP16 Talker MLP GEMM (`gemm`) and NVFP4 blockscaled GEMM (`gemm_nvfp4`)
- `kernelSrcs/int4_fp16_gemm_cutedsl/README.md` — Ampere-floor W4A16 GEMM
- `kernelSrcs/nvfp4_moe_cutedsl/README.md` — NVFP4 MoE
- `kernelSrcs/nvfp4_fused_moe_cutedsl/README.md` — NVFP4 fused MoE
