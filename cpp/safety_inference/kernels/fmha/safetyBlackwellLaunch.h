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

#include "safetyFmhaLaunch.h"

#include <algorithm>

namespace trt_edgellm::safety::fmha
{
// Adapted from kernelSrcs/fmha_cutedsl_blackwell/fmha.py at
// 8697bde5b6d43a9f70581c41f7aeca77e694e8dd: Blackwell FMHA launch descriptors.
// Safety owns module preparation and fixed TMA bindings; no DSL host loader is deployed.
inline CUresult prepareBlackwellBindings(PreparedBindings& state, Artifact const& artifact,
    std::array<void const*, 4> const& pointers, std::array<int32_t, 7> const& dimensions) noexcept
{
    auto const [tokens, qHeads, kvHeads, capacity, sequences, maxSequence, headSize] = dimensions;
    (void) sequences;
    (void) maxSequence;
    uint64_t const d = static_cast<uint64_t>(headSize);
    uint64_t const ratio = static_cast<uint64_t>(qHeads / kvHeads);
    for (int i = 0; i < 4; ++i)
    {
        bool const qo = i == 0 || i == 3;
        // Q/O: [D, token, grouped-query head, KV head]; K/V omit the broadcast GQA mode.
        uint64_t sizes[]{d, static_cast<uint64_t>(qo || artifact.visual ? tokens : capacity),
            qo ? ratio : static_cast<uint64_t>(kvHeads), static_cast<uint64_t>(kvHeads)};
        uint64_t strides[]{2 * d
                * static_cast<uint64_t>(qo ? qHeads
                        : artifact.visual  ? kvHeads
                                           : 1),
            2 * d * static_cast<uint64_t>(qo || artifact.visual ? 1 : capacity), 2 * d * ratio};
        uint32_t box[]{64, 128, 1, 1};
        uint32_t steps[]{1, 1, 1, 1};
        auto const status = cuTensorMapEncodeTiled(&state.maps[i], CU_TENSOR_MAP_DATA_TYPE_FLOAT16, qo ? 4 : 3,
            const_cast<void*>(pointers[i]), sizes, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (status != CUDA_SUCCESS)
        {
            return status;
        }
    }
    state.pointers = pointers;
    state.dimensions = dimensions;
    state.ready = true;
    return CUDA_SUCCESS;
}

namespace detail
{
struct alignas(16) MmaDescriptor
{
    uint8_t flags[3]{};
    alignas(16) int32_t coordinates[4]{};
};
struct Coordinates
{
    int32_t tokens;
    int32_t ratio;
    int32_t kvHeads;
};
struct Output32
{
    void* data;
    Coordinates shape;
    int32_t tokenStride;
    int32_t headStride;
};
static_assert(sizeof(MmaDescriptor) == 32 && offsetof(MmaDescriptor, coordinates) == 16);
static_assert(sizeof(Coordinates) == 12 && sizeof(Output32) == 32 && offsetof(Output32, tokenStride) == 20);
} // namespace detail

//! Operational uses only prepared maps, bounded stack arguments and the admitted driver launch.
inline CUresult launchBlackwell(CUfunction function, Artifact const& artifact, PreparedBindings& state,
    int32_t const* cuQ, int32_t const* cuK, int32_t const* valid, float scale, int32_t smCount,
    CUstream stream) noexcept
{
    auto const [tokens, qHeads, kvHeads, capacity, sequences, maxSequence, headSize] = state.dimensions;
    detail::MmaDescriptor mma{};
    detail::Coordinates q{tokens, qHeads / kvHeads, kvHeads};
    detail::Coordinates k{artifact.visual ? tokens : capacity, q.ratio, kvHeads};
    detail::Output32 output{const_cast<void*>(state.pointers[3]), q, qHeads * headSize, q.ratio * headSize};
    detail::Lengths16 queryLengths{cuQ, sequences + 1, 0};
    detail::Lengths16 keyLengths{cuK, sequences + 1, 0};
    float scaleLog2 = scale * 1.4426950408889634F;
    float outputScale = 1.0F;
    int32_t rightWindow = 0;
    int32_t tiles = (maxSequence + artifact.tileM - 1) / artifact.tileM;
    int32_t heads = qHeads;
    int32_t batches = sequences;
    void const* validity = valid;
    void* parameters[]{&mma, &mma, &state.maps[0], &q, &state.maps[1], &k, &state.maps[2], &k, &state.maps[3], &q,
        &output, &queryLengths, &keyLengths, &scaleLog2, &scale, &outputScale, &rightWindow, &tiles, &heads, &batches,
        &validity};
    if (artifact.visual)
    {
        // Bidirectional specialization has no causal right-window argument.
        for (int i = 16; i < 20; ++i)
        {
            parameters[i] = parameters[i + 1];
        }
    }
    int32_t const blocks = std::min(smCount, tiles * heads * batches);
    return cuLaunchKernel(
        function, blocks, 1, 1, artifact.threads, 1, 1, artifact.sharedMemory, stream, parameters, nullptr);
}
} // namespace trt_edgellm::safety::fmha
