/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "greedyArgmaxKernel.cuh"

namespace trt_edgellm::safety
{
namespace
{
constexpr int32_t kGreedyArgmaxThreads{256};
constexpr int32_t kGreedyArgmaxWarpSize{32};
constexpr int32_t kGreedyArgmaxWarps{kGreedyArgmaxThreads / kGreedyArgmaxWarpSize};
static_assert(kGreedyArgmaxThreads % kGreedyArgmaxWarpSize == 0);

__device__ __forceinline__ ArgmaxPair betterArgmaxPair(ArgmaxPair candidate, ArgmaxPair current) noexcept
{
    bool const candidateWins = candidate.valid != 0
        && (current.valid == 0 || candidate.value > current.value
            || (candidate.value == current.value && candidate.index < current.index));
    return candidateWins ? candidate : current;
}

__device__ __forceinline__ ArgmaxPair reduceArgmaxWarp(ArgmaxPair candidate, unsigned mask) noexcept
{
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % warpSize;
    int32_t const active = __popc(mask);
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2)
    {
        ArgmaxPair peer{__shfl_down_sync(mask, candidate.value, offset),
            __shfl_down_sync(mask, candidate.index, offset), __shfl_down_sync(mask, candidate.valid, offset)};
        if (lane + offset < active)
        {
            candidate = betterArgmaxPair(peer, candidate);
        }
    }
    return candidate;
}

__global__ void greedyArgmaxBlock(float const* logits, int32_t vocabularySize, ArgmaxPair* result)
{
    extern __shared__ ArgmaxPair warpBest[];
    ArgmaxPair candidate{};
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < vocabularySize; index += blockDim.x)
    {
        float const value = logits[index];
        if (!isnan(value))
        {
            candidate = betterArgmaxPair({value, index, 1}, candidate);
        }
    }
    unsigned const mask = __activemask();
    candidate = reduceArgmaxWarp(candidate, mask);
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % warpSize;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) / warpSize;
    if (lane == 0)
    {
        warpBest[warp] = candidate;
    }
    __syncthreads();
    if (warp == 0)
    {
        int32_t const warpCount = (static_cast<int32_t>(blockDim.x) + warpSize - 1) / warpSize;
        candidate = lane < warpCount ? warpBest[lane] : ArgmaxPair{};
        candidate = reduceArgmaxWarp(candidate, __ballot_sync(mask, lane < warpCount));
        if (lane == 0)
        {
            result[0] = candidate;
        }
    }
}
} // namespace

cudaError_t launchGreedyArgmax(
    float const* logits, int32_t vocabularySize, ArgmaxPair* result, cudaStream_t stream) noexcept
{
    if (logits == nullptr || result == nullptr || vocabularySize <= 0 || stream == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    greedyArgmaxBlock<<<1, kGreedyArgmaxThreads, kGreedyArgmaxWarps * sizeof(ArgmaxPair), stream>>>(
        logits, vocabularySize, result);
    return cudaPeekAtLastError();
}

} // namespace trt_edgellm::safety
