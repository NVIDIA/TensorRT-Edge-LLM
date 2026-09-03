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
| Jetson Thor | Official | JetPack 7.2 | 13.2 | JetPack package | Device | Model-dependent |
| NVIDIA DRIVE Thor | Official | DriveOS 7.2 | 13.3 | DriveOS SDK package | SDK container, then deploy `build/` | Model-dependent |
| NVIDIA DGX Spark (GB10) | Official | DGX Spark software stack | 13.0 | System package | Device | Model-dependent |
| Jetson Orin | Official | JetPack 7.2 | 13.2 | JetPack package | Device | FP16, INT8, and INT4 only |
| x86-64 Linux GPU | Developer | Ubuntu 22.04 / 24.04 | 12.x or 13.x | Compatible user package | Workstation | Development and validation |

`Official` combinations are release-tested deployment targets. `Compatible`
combinations are expected to work with the stated constraints. `Developer`
combinations support development but are not edge deployment targets.

Jetson Orin does not run FP8 or FP4 model engines. Edge deployments normally
use the TensorRT version supplied by the platform SDK; x86 builds must use
mutually compatible TensorRT and CUDA packages.

## Wheel Packaging Matrix

The wheel tooling is configured to assemble one x86_64 and one aarch64 wheel
for each supported CPython minor: 3.10, 3.11, and 3.12. Native payload selection
is exact; the loader does not guess a nearest SM or TensorRT major.

| Wheel architecture | Configured runtime rows |
|---|---|
| x86_64 | Ubuntu 22.04, CUDA 13, SM80, TensorRT 10 |
| x86_64 | Ubuntu 24.04, CUDA 13, SM86/SM100/SM120, TensorRT 10 |
| x86_64 | Ubuntu 24.04, CUDA 13, SM80/SM86/SM100/SM120, TensorRT 11 |
| aarch64 | Jetson Orin: JetPack 7.2, CUDA 13, SM87, platform TensorRT 10 |
| aarch64 | Jetson Thor: JetPack 7.0/7.1/7.2, CUDA 13, SM110, platform TensorRT 10 |
| aarch64 | DRIVE Thor: DriveOS 7.2, CUDA 13, SM110, platform TensorRT 10 |
| aarch64 | DGX Spark current stack, CUDA 13, SM121, platform TensorRT 10 |

Release qualification installs the final wheel into a clean environment using
the Python ABI provided by each target system, builds a small model through the
installed high-level API, and runs inference through the installed runtime. An
architecture wheel is ready for release only after every configured target row
has passed this behavioral check.

The wheel contract matches the observed platform release, CUDA and TensorRT
SONAMEs, and GPU SM exactly. It does not claim NVIDIA driver-version ranges;
driver compatibility remains part of the CUDA/platform support contract. The
broader x86 developer-source row remains useful for source builds but is
not a promise that every OS/CUDA/SM cross-product is present in version-1
wheels. Adding a wheel row requires a canonical exact-SM CuTe artifact and
passing installed-wheel build and inference validation.
