<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Official Support Matrix

This page lists supported software stacks. See [Supported Models](supported-models.md)
for checkpoint IDs and [Installation](installation.md) for build commands.

## Platforms

| Platform | Level | OS / SDK | CUDA Toolkit | TensorRT | Build location | Precision constraint |
|---|---|---|---|---|---|---|
| Jetson Thor | Official | JetPack 7.0 / 7.1 | 13.0 | JetPack package | Device | Model-dependent |
| Jetson Thor | Official | JetPack 7.2.x | 13.2 | JetPack package | Device | Model-dependent |
| NVIDIA DRIVE Thor | Official | DriveOS 7.2 | 13.3 | DriveOS SDK package | SDK container, then deploy `build/` | Model-dependent |
| NVIDIA DGX Spark (GB10) | Official | DGX Spark software stack | 13.0 | System package | Device | Model-dependent |
| Jetson Orin | Official | JetPack 7.2.x | 13.2 | JetPack package | Device | FP16, INT8, and INT4 only |
| NVIDIA IGX Thor | Official | Current Linux stack | 13.0 | 10.13.3.9 | Device | Model-dependent; SM110 iGPU or SM120 dGPU |
| x86-64 Linux GPU | Developer | Ubuntu 22.04 / 24.04 | 12.x or 13.x | Compatible user package | Workstation | Development and validation |
| Windows x64 GPU | Developer | Windows x64 / MSVC | 13 | TensorRT RTX 1.7.1.80 | Workstation (source build) | Model- and GPU-dependent |

`Official` combinations are release-tested deployment targets. `Compatible`
combinations are expected to work with the stated constraints. `Developer`
combinations support development but are not edge deployment targets.

Jetson Orin does not run FP8 or FP4 model engines. Edge deployments normally
use the TensorRT version supplied by the platform SDK; x86 builds must use
mutually compatible TensorRT and CUDA packages.

## Wheel Packaging Matrix

Release wheels are built for CPython 3.10, 3.11, and 3.12 on `x86_64` (glibc 2.35+)
and `aarch64` (ARM64, glibc 2.39+). `pip` selects the wheel for your Python version
and CPU architecture. Each wheel includes the compiled components for the matching
configurations below; you do not need a separate download for each GPU.

Your platform, CUDA and TensorRT runtime libraries, and GPU must match a listed
configuration. Edge-LLM checks this when it loads. See
[published-wheel installation](installation.md#published-python-wheel) to get
started.

| CPU architecture | Platform and OS / SDK | CUDA | GPU (SM) | TensorRT |
|---|---|---|---|---|
| x86_64 | Ubuntu 22.04 | 13 | 80 | 10 |
| x86_64 | Ubuntu 24.04 | 13 | 86, 100, 120 | 10 |
| x86_64 | Ubuntu 24.04 | 13 | 80, 86, 100, 120 | 11 |
| aarch64 | Jetson Orin, JetPack 7.2.x | 13 | 87 | Platform-provided 10 |
| aarch64 | Jetson Thor, JetPack 7.0 / 7.1 / 7.2.x | 13 | 110 | Platform-provided 10 |
| aarch64 | DRIVE Thor, DriveOS 7.2 | 13 | 110 | Platform-provided 10 |
| aarch64 | IGX Thor, current platform software | 13 | 110, 120 | Platform-provided 10 |
| aarch64 | DGX Spark, current platform software | 13 | 121 | Platform-provided 10 |

**SM** identifies the GPU's compute capability: for example, SM80 means compute
capability 8.0. The GPU must match a listed SM; Edge-LLM does not substitute
components built for another GPU or TensorRT major version.

Use the NVIDIA driver required by your CUDA version and platform SDK.

If your configuration is not listed, follow the
[source installation instructions](installation.md#source-workflow-c-runtime).
