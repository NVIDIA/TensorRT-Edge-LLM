<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Official Support Matrix

This page lists supported software stacks and KV cache reuse combinations. See
[Supported Models](supported-models.md) for checkpoint IDs and
[Installation](installation.md) for build commands.

## Platforms

| Platform | Level | OS / SDK | CUDA Toolkit | TensorRT | Build location | Precision constraint |
|---|---|---|---|---|---|---|
| Jetson Thor | Official | JetPack 7.0 / 7.1 | 13.0 | JetPack package | Device | Model-dependent |
| Jetson Thor | Official | JetPack 7.2 | 13.2 | JetPack package | Device | Model-dependent |
| NVIDIA DRIVE Thor | Official | DriveOS 7.2 | 13.3 | DriveOS SDK package | SDK container, then deploy `build/` | Model-dependent |
| NVIDIA DGX Spark (GB10) | Official | DGX Spark software stack | 13.0 | System package | Device | Model-dependent |
| Jetson Orin | Official | JetPack 7.2 | 13.2 | JetPack package | Device | FP16, INT8, and INT4 only |
| Jetson Orin | Compatible | JetPack 6.2+ | 12.6 | JetPack package | Device | FP16, INT8, and INT4 only |
| x86-64 Linux GPU | Developer | Ubuntu 22.04 / 24.04 | 12.x or 13.x | Compatible user package | Workstation | Development and validation |

`Official` combinations are release-tested deployment targets. `Compatible`
combinations are expected to work with the stated constraints. `Developer`
combinations support development but are not edge deployment targets.

Jetson Orin does not run FP8 or FP4 model engines. Edge deployments normally
use the TensorRT version supplied by the platform SDK; x86 builds must use
mutually compatible TensorRT and CUDA packages.

## KV Cache Reuse Support

| Deployment scenario | Generalized reuse | Requirements or limitation |
|---|---:|---|
| Text input, attention-only model, vanilla decoding | Yes | FP16 KV cache |
| Text input, pure recurrent model, vanilla decoding | Yes | Non-zero recurrent snapshot pool |
| Text input, hybrid attention/recurrent model, vanilla decoding | Yes | FP16 KV cache plus recurrent and partial-KV snapshot pools |
| Image-input VLM, vanilla decoding | Yes | FP16 KV cache plus the recurrent snapshot pools required by the model |
| Text input, attention-only EAGLE | Yes | Independent FP16 KV caches for matched base and draft engines |
| Hybrid EAGLE | No | Recurrent EAGLE state is not reusable |
| Vision-bidirectional attention | No | State is not managed by the context cache |
| MTP, DFlash, DSpark, or Gemma4 MTP | No | Run without context reuse |
| FP8 KV cache | No | Generalized reuse requires FP16 KV pages |
| Audio-input runtime | No | Not covered by release validation |
| Speech-output runtime | No | Audio-generation requests bypass cache lookup |
| Action runtime | No | Action runners cannot initialize context reuse |
| DiffusionGemma / block diffusion | No | No reusable autoregressive context contract |

See [KV Cache Reuse](../features/kv-cache-reuse.md) for build capacity, runtime
flags, and request policies.
