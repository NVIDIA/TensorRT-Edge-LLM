/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda.h>

namespace trt_edgellm::safety::fmha
{
struct Artifact
{
    unsigned char const* cubin;
    size_t bytes;
    char const* symbol;
    unsigned int sharedMemory;
    int headSize;
    bool visual;
    int tileM;
    int threads;
};

//! One immutable binding set per plugin instance, populated during Initialization warmup.
struct PreparedBindings
{
    CUtensorMap maps[4]{};
    std::array<void const*, 4> pointers{};
    std::array<int32_t, 7> dimensions{};
    bool ready{};
};

struct ContiguousArguments
{
    void const* validity;
    void const* q;
    void const* k;
    void const* v;
    void* output;
    int32_t const* cuQ;
    int32_t const* cuK;
    int32_t queryLength;
    int32_t qHeads;
    int32_t kvHeads;
    int32_t capacity;
    float scale;
};

struct VisualArguments
{
    void const* validity;
    void const* q;
    void const* k;
    void const* v;
    void* output;
    int32_t const* cuSeqLens;
    int32_t totalTokens;
    int32_t qHeads;
    int32_t kvHeads;
    int32_t sequences;
    int32_t maxSeqLength;
    float scale;
};

namespace detail
{
struct Tensor32
{
    void const* data;
    int32_t shape0;
    int32_t shape1;
    int64_t stride0;
    int64_t stride1;
};
struct Tensor24
{
    void const* data;
    int32_t tokens;
    int32_t heads;
    int64_t tokenStride;
};
struct Lengths16
{
    void const* data;
    int32_t size;
    int32_t padding;
};
static_assert(sizeof(void*) == 8);
static_assert(sizeof(Tensor32) == 32 && offsetof(Tensor32, shape0) == 8 && offsetof(Tensor32, stride0) == 16);
static_assert(sizeof(Tensor24) == 24 && offsetof(Tensor24, tokens) == 8 && offsetof(Tensor24, tokenStride) == 16);
static_assert(sizeof(Lengths16) == 16 && offsetof(Lengths16, size) == 8);
} // namespace detail

// Adapted from kernelSrcs/fmha_v2_cutedsl/fmha.py at
// ade55abc28de3c75ac3a96f4a1e4f8c7c8df3878: FMHAV2Ampere.__call__ launch configuration.
// Safety adaptation: replace generated lazy CUDA library loading with a prepared
// CUfunction and an explicitly checked raw descriptor ABI; preserve launch geometry.
//! Launch the pinned contiguous CuTe device ABI; loading and function configuration belong to Initialization.
inline CUresult launchContiguous(
    CUfunction function, Artifact const& artifact, ContiguousArguments const& args, CUstream stream) noexcept
{
    if (function == nullptr || stream == nullptr || artifact.visual || artifact.tileM <= 0)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    int64_t const qStride = static_cast<int64_t>(args.qHeads) * artifact.headSize;
    int64_t const kvStride = static_cast<int64_t>(args.kvHeads) * artifact.headSize;
    detail::Tensor32 q{args.q, args.queryLength, args.qHeads, args.queryLength * qStride, qStride};
    // BSND descriptors address BHSD cache storage: token stride D, head stride capacity * D.
    detail::Tensor32 k{args.k, args.capacity, args.kvHeads, args.capacity * kvStride,
        static_cast<int64_t>(args.capacity) * artifact.headSize};
    detail::Tensor32 v{k};
    v.data = args.v;
    detail::Tensor32 output{args.output, args.queryLength, args.qHeads, args.queryLength * qStride, qStride};
    void const* validity = args.validity;
    int32_t const* cuQ = args.cuQ;
    int32_t const* cuK = args.cuK;
    float scaleLog2 = args.scale * 1.4426950408889634F;
    int32_t kvHeads = args.kvHeads;
    // The generator verifies these descriptor types and argument order against the lowered device launch.
    void* parameters[]{&validity, &q, &k, &v, &output, &cuQ, &cuK, &scaleLog2, &kvHeads};
    return cuLaunchKernel(function, (args.queryLength + artifact.tileM - 1) / artifact.tileM, args.qHeads, 1,
        artifact.threads, 1, 1, artifact.sharedMemory, stream, parameters, nullptr);
}

// Adapted from kernelSrcs/fmha_v2_cutedsl/fmha.py at
// ade55abc28de3c75ac3a96f4a1e4f8c7c8df3878: FMHAV2Ampere.__call_vit__ launch configuration.
// Safety adaptation: replace generated lazy CUDA library loading with a prepared
// CUfunction and an explicitly checked raw descriptor ABI; preserve launch geometry.
//! Launch packed bidirectional attention using the same pinned device ABI as the copied Standard kernel.
inline CUresult launchVisual(
    CUfunction function, Artifact const& artifact, VisualArguments const& args, CUstream stream) noexcept
{
    if (function == nullptr || stream == nullptr || !artifact.visual || artifact.tileM <= 0)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    detail::Tensor24 q{args.q, args.totalTokens, args.qHeads, static_cast<int64_t>(args.qHeads) * artifact.headSize};
    detail::Tensor24 k{args.k, args.totalTokens, args.kvHeads, static_cast<int64_t>(args.kvHeads) * artifact.headSize};
    detail::Tensor24 v{args.v, args.totalTokens, args.kvHeads, static_cast<int64_t>(args.kvHeads) * artifact.headSize};
    detail::Tensor24 output{
        args.output, args.totalTokens, args.qHeads, static_cast<int64_t>(args.qHeads) * artifact.headSize};
    detail::Lengths16 lengths{args.cuSeqLens, args.sequences + 1, 0};
    void const* validity = args.validity;
    float scaleLog2 = args.scale * 1.4426950408889634F;
    int32_t kvHeads = args.kvHeads;
    int32_t maxSequence = args.maxSeqLength;
    void* parameters[]{&validity, &q, &k, &v, &output, &lengths, &lengths, &scaleLog2, &kvHeads, &maxSequence};
    return cuLaunchKernel(function, (args.maxSeqLength + artifact.tileM - 1) / artifact.tileM, args.sequences,
        args.qHeads, artifact.threads, 1, 1, artifact.sharedMemory, stream, parameters, nullptr);
}
} // namespace trt_edgellm::safety::fmha
