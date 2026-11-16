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

#include "common/checkMacros.h"
#include "initializeCosSinCache.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace trt_edgellm
{
namespace kernel
{

template <int32_t RotaryDim>
__global__ void initializeNormalRopeCosSinKernel(
    float* cosSinCache, float rotaryBaseFrequency, float rotaryScale, int32_t rotaryEmbeddingMaxPositions)
{
    // In this kernel, each warp compute one "position" of the cos/sin cache, and loop until max position.
    // Each CTA will be assigned 4 warps so it proceeds 4 positions in an iteration.
    static_assert(RotaryDim % 64 == 0, "rotaryDim must be multiple of 64");

    uint32_t const bIdx = blockIdx.x;
    uint32_t const tIdx = threadIdx.x;
    uint32_t const tIdy = threadIdx.y;

    uint32_t const bDimY = blockDim.y;
    uint32_t const gDimX = gridDim.x;

    uint32_t const startPosIdx = bIdx * bDimY + tIdy;
    uint32_t const posStride = gDimX * bDimY;

    float ropeConstants[RotaryDim / 64];

#pragma unroll
    for (uint32_t i = 0; i < RotaryDim / 64; ++i)
    {
        uint32_t zid = tIdx + i * 32;
        ropeConstants[i] = pow(rotaryBaseFrequency, 2 * zid / (float) RotaryDim);
    }

    for (uint32_t posIdx = startPosIdx; posIdx < rotaryEmbeddingMaxPositions; posIdx += posStride)
    {
        uint32_t cosSinOffset = posIdx * RotaryDim;

#pragma unroll
        for (uint32_t i = 0; i < RotaryDim / 64; ++i)
        {
            float invFreq = posIdx * rotaryScale / ropeConstants[i];
            float cosVal = cos(invFreq);
            float sinVal = sin(invFreq);

            uint32_t zid = tIdx + i * 32;
            cosSinCache[cosSinOffset + zid] = cosVal;
            cosSinCache[cosSinOffset + zid + RotaryDim / 2] = sinVal;
        }
    }
}

template <int32_t RotaryDim>
__global__ void initializeNormalRopeCosSinWithIdentityTailKernel(float* cosSinCache, float rotaryBaseFrequency,
    float rotaryScale, float partialRotaryFactor, int32_t rotaryEmbeddingMaxPositions)
{
    // Use half-warp groups for valid rotary dims that are not multiples of 64
    // and for full-width caches that materialize a partial_rotary_factor tail as
    // identity values.
    static_assert(RotaryDim % 32 == 0, "rotaryDim must be multiple of 32");

    uint32_t const bIdx = blockIdx.x;
    uint32_t const tIdx = threadIdx.x;
    uint32_t const tIdy = threadIdx.y;

    uint32_t const bDimY = blockDim.y;
    uint32_t const gDimX = gridDim.x;

    uint32_t const startPosIdx = bIdx * bDimY + tIdy;
    uint32_t const posStride = gDimX * bDimY;

    float const clampedPartialRotaryFactor
        = partialRotaryFactor < 0.0F ? 0.0F : (partialRotaryFactor > 1.0F ? 1.0F : partialRotaryFactor);
    uint32_t const rotatedAngles
        = static_cast<uint32_t>(clampedPartialRotaryFactor * static_cast<float>(RotaryDim / 2));
    float ropeConstants[RotaryDim / 32];

#pragma unroll
    for (uint32_t i = 0; i < RotaryDim / 32; ++i)
    {
        uint32_t zid = tIdx + i * 16;
        ropeConstants[i] = pow(rotaryBaseFrequency, 2 * zid / (float) RotaryDim);
    }

    for (uint32_t posIdx = startPosIdx; posIdx < rotaryEmbeddingMaxPositions; posIdx += posStride)
    {
        uint32_t cosSinOffset = posIdx * RotaryDim;

#pragma unroll
        for (uint32_t i = 0; i < RotaryDim / 32; ++i)
        {
            uint32_t zid = tIdx + i * 16;
            if (zid >= rotatedAngles)
            {
                cosSinCache[cosSinOffset + zid] = 1.0F;
                cosSinCache[cosSinOffset + zid + RotaryDim / 2] = 0.0F;
                continue;
            }

            float invFreq = posIdx * rotaryScale / ropeConstants[i];
            float cosVal = cos(invFreq);
            float sinVal = sin(invFreq);

            cosSinCache[cosSinOffset + zid] = cosVal;
            cosSinCache[cosSinOffset + zid + RotaryDim / 2] = sinVal;
        }
    }
}

void initializeNormalRopeCosSin(float* cosSinCache, float rotaryBaseFrequency, float rotaryScale,
    float partialRotaryFactor, int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, cudaStream_t stream)
{
    bool const useIdentityTail = partialRotaryFactor < 1.0F;
    bool const useHalfWarpKernel = useIdentityTail || rotaryDim == 32 || rotaryDim == 96;
    // blockDim.x threads cover rotaryDim/2 dimensions via an inner stride loop in the kernel
    // (e.g., 32 threads × 8 iterations = 256 dims for rotaryDim=512).
    dim3 block = useHalfWarpKernel ? dim3(16, 8) : dim3(32, 4);

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    int32_t const numSMs = deviceProp.multiProcessorCount;

    void* kernelPtr{nullptr};
    switch (rotaryDim)
    {
    case 32: kernelPtr = (void*) initializeNormalRopeCosSinWithIdentityTailKernel<32>; break;
    case 64:
        kernelPtr = useHalfWarpKernel ? (void*) initializeNormalRopeCosSinWithIdentityTailKernel<64>
                                      : (void*) initializeNormalRopeCosSinKernel<64>;
        break;
    case 96: kernelPtr = (void*) initializeNormalRopeCosSinWithIdentityTailKernel<96>; break;
    case 128:
        kernelPtr = useHalfWarpKernel ? (void*) initializeNormalRopeCosSinWithIdentityTailKernel<128>
                                      : (void*) initializeNormalRopeCosSinKernel<128>;
        break;
    case 256:
        kernelPtr = useHalfWarpKernel ? (void*) initializeNormalRopeCosSinWithIdentityTailKernel<256>
                                      : (void*) initializeNormalRopeCosSinKernel<256>;
        break;
    case 512:
        kernelPtr = useHalfWarpKernel ? (void*) initializeNormalRopeCosSinWithIdentityTailKernel<512>
                                      : (void*) initializeNormalRopeCosSinKernel<512>;
        break;
    default:
        throw std::runtime_error(
            "Un-implemented rotaryDim for initializeNormalRopeCosSin: " + std::to_string(rotaryDim));
    }
    int32_t const blockThreads = block.x * block.y;
    int32_t maxBlockPerSM{};
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlockPerSM, kernelPtr, blockThreads, 0));

    int32_t const positionsPerBlock = useHalfWarpKernel ? 8 : 4;
    int32_t const numBlocks
        = std::min(maxBlockPerSM * numSMs, (rotaryEmbeddingMaxPositions + positionsPerBlock - 1) / positionsPerBlock);
    dim3 grid(numBlocks);

    void* kernelArgs[] = {reinterpret_cast<void*>(&cosSinCache), reinterpret_cast<void*>(&rotaryBaseFrequency),
        reinterpret_cast<void*>(&rotaryScale), reinterpret_cast<void*>(&rotaryEmbeddingMaxPositions)};
    void* identityTailKernelArgs[] = {reinterpret_cast<void*>(&cosSinCache),
        reinterpret_cast<void*>(&rotaryBaseFrequency), reinterpret_cast<void*>(&rotaryScale),
        reinterpret_cast<void*>(&partialRotaryFactor), reinterpret_cast<void*>(&rotaryEmbeddingMaxPositions)};
    CUDA_CHECK(
        cudaLaunchKernel(kernelPtr, grid, block, useHalfWarpKernel ? identityTailKernelArgs : kernelArgs, 0, stream));
}

template <int32_t RotaryDim>
__global__ void initializeLongRopeCosSinKernel(float* cosSinCache, float* extFactors, float rotaryBaseFrequency,
    int32_t rotaryEmbeddingMaxPositions, float scalingFactor)
{
    // In this kernel, each warp compute one "position" of the cos/sin cache, and loop until max position.
    // Each CTA will be assigned 4 warps, 16 thread proceeds 1 position， so it proceeds 8 positions in an iteration.
    static_assert(RotaryDim % 32 == 0, "rotaryDim must be multiple of 32");

    uint32_t const bIdx = blockIdx.x;
    uint32_t const tIdx = threadIdx.x;
    uint32_t const tIdy = threadIdx.y;

    uint32_t const bDimY = blockDim.y;
    uint32_t const gDimX = gridDim.x;

    uint32_t const startPosIdx = bIdx * bDimY + tIdy;
    uint32_t const posStride = gDimX * bDimY;

    float ropeConstants[RotaryDim / 32];

#pragma unroll
    for (uint32_t i = 0; i < RotaryDim / 32; ++i)
    {
        uint32_t zid = tIdx + i * 16;
        ropeConstants[i] = extFactors[zid] * pow(rotaryBaseFrequency, 2 * zid / (float) RotaryDim);
    }

    for (uint32_t posIdx = startPosIdx; posIdx < rotaryEmbeddingMaxPositions; posIdx += posStride)
    {
        uint32_t cosSinOffset = posIdx * RotaryDim;

#pragma unroll
        for (uint32_t i = 0; i < RotaryDim / 32; ++i)
        {
            uint32_t zid = tIdx + i * 16;
            float invFreq = posIdx / ropeConstants[i];
            float cosVal = cos(invFreq) * scalingFactor;
            float sinVal = sin(invFreq) * scalingFactor;

            cosSinCache[cosSinOffset + zid] = cosVal;
            cosSinCache[cosSinOffset + zid + RotaryDim / 2] = sinVal;
        }
    }
}

void initializeLongRopeCosSin(float* shortCosSinCache, float* longCosSinCache, float* shortFactor, float* longFactors,
    float rotaryBaseFrequency, int32_t rotaryDim, int32_t rotaryEmbeddingMaxPositions, int32_t maxPositionEmbeddings,
    int32_t originalMaxPositionEmbeddings, cudaStream_t stream)
{
    // rotaryEmbeddingMaxPositions: length of position embeddings
    //     shortCosSinCache/longCosSinCache shape: [1, rotaryEmbeddingMaxPositions, rotaryDim]
    // maxPositionEmbeddings: config.max_position_embeddings
    // originalMaxPositionEmbeddings: config.original_max_position_embeddings

    float scalingFactor = 1.0f;
    float scale = (float) maxPositionEmbeddings / (float) originalMaxPositionEmbeddings;
    if (scale > 1.0f)
    {
        scalingFactor = std::sqrt(1 + std::log(scale) / std::log(originalMaxPositionEmbeddings));
    }

    // Each CTA get assigned 128 threads.
    dim3 block(16, 8);

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    int32_t const numSMs = deviceProp.multiProcessorCount;

    void* kernelPtr{nullptr};
    switch (rotaryDim)
    {
    case 32: kernelPtr = (void*) initializeLongRopeCosSinKernel<32>; break;
    case 64: kernelPtr = (void*) initializeLongRopeCosSinKernel<64>; break;
    case 96: kernelPtr = (void*) initializeLongRopeCosSinKernel<96>; break;
    case 128: kernelPtr = (void*) initializeLongRopeCosSinKernel<128>; break;
    default:
        throw std::runtime_error("Un-implemented rotaryDim for initializeLongRopeCosSin: " + std::to_string(rotaryDim));
    }

    int32_t maxBlockPerSM{};
    int32_t numBlocks{};
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlockPerSM, kernelPtr, 128, 0));

    // Initialized longCosSinCache for context length > originalMaxPositionEmbeddings
    // For all positions, use longFactors to compute cosSinCache
    numBlocks = std::min(maxBlockPerSM * numSMs, rotaryEmbeddingMaxPositions / 8);
    dim3 longGrid(numBlocks);

    void* longKernelArgs[] = {reinterpret_cast<void*>(&longCosSinCache), reinterpret_cast<void*>(&longFactors),
        reinterpret_cast<void*>(&rotaryBaseFrequency), reinterpret_cast<void*>(&rotaryEmbeddingMaxPositions),
        reinterpret_cast<void*>(&scalingFactor)};
    CUDA_CHECK(cudaLaunchKernel(kernelPtr, longGrid, block, longKernelArgs, 0, stream));

    // Initialized shortCosSinCache for context length <= originalMaxPositionEmbeddings
    // For positions <= originalMaxPositionEmbeddings, use shortFactor to compute cosSinCache
    // For positions > originalMaxPositionEmbeddings, use longFactors to compute cosSinCache. Copy from longCosSinCache.
    int32_t shortMaxPositions = std::min(originalMaxPositionEmbeddings, rotaryEmbeddingMaxPositions);
    numBlocks = std::min(maxBlockPerSM * numSMs, shortMaxPositions / 8);
    dim3 shortGrid(numBlocks);

    void* shortKernelArgs[] = {reinterpret_cast<void*>(&shortCosSinCache), reinterpret_cast<void*>(&shortFactor),
        reinterpret_cast<void*>(&rotaryBaseFrequency), reinterpret_cast<void*>(&shortMaxPositions),
        reinterpret_cast<void*>(&scalingFactor)};
    CUDA_CHECK(cudaLaunchKernel(kernelPtr, shortGrid, block, shortKernelArgs, 0, stream));

    if (rotaryEmbeddingMaxPositions > shortMaxPositions)
    {
        CUDA_CHECK(cudaMemcpyAsync(shortCosSinCache + shortMaxPositions * rotaryDim,
            longCosSinCache + shortMaxPositions * rotaryDim,
            (rotaryEmbeddingMaxPositions - shortMaxPositions) * rotaryDim * sizeof(float), cudaMemcpyDeviceToDevice,
            stream));
    }
}

template <int32_t RotaryDim>
__global__ void initializeYarnCosSinKernel(
    float* cosSinCache, float const* invFreq, float mscale, int32_t rotaryEmbeddingMaxPositions)
{
    // Same launch geometry as the LongRope kernel: each half-warp (16 threads)
    // fills one position, 8 positions per CTA. invFreq is the precomputed
    // per-dimension inverse frequency; this kernel only does the position sweep.
    static_assert(RotaryDim % 32 == 0, "rotaryDim must be multiple of 32");

    uint32_t const bIdx = blockIdx.x;
    uint32_t const tIdx = threadIdx.x;
    uint32_t const tIdy = threadIdx.y;

    uint32_t const bDimY = blockDim.y;
    uint32_t const gDimX = gridDim.x;

    uint32_t const startPosIdx = bIdx * bDimY + tIdy;
    uint32_t const posStride = gDimX * bDimY;

    float invFreqLocal[RotaryDim / 32];
#pragma unroll
    for (uint32_t i = 0; i < RotaryDim / 32; ++i)
    {
        invFreqLocal[i] = invFreq[tIdx + i * 16];
    }

    for (uint32_t posIdx = startPosIdx; posIdx < rotaryEmbeddingMaxPositions; posIdx += posStride)
    {
        uint32_t cosSinOffset = posIdx * RotaryDim;

#pragma unroll
        for (uint32_t i = 0; i < RotaryDim / 32; ++i)
        {
            uint32_t zid = tIdx + i * 16;
            float angle = posIdx * invFreqLocal[i];
            cosSinCache[cosSinOffset + zid] = cos(angle) * mscale;
            cosSinCache[cosSinOffset + zid + RotaryDim / 2] = sin(angle) * mscale;
        }
    }
}

void initializeYarnCosSin(float* cosSinCache, float const* invFreq, float mscale, int32_t rotaryDim,
    int32_t rotaryEmbeddingMaxPositions, cudaStream_t stream)
{
    dim3 block(16, 8);

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    int32_t const numSMs = deviceProp.multiProcessorCount;

    void* kernelPtr{nullptr};
    switch (rotaryDim)
    {
    case 64: kernelPtr = (void*) initializeYarnCosSinKernel<64>; break;
    case 128: kernelPtr = (void*) initializeYarnCosSinKernel<128>; break;
    default:
        throw std::runtime_error("Un-implemented rotaryDim for initializeYarnCosSin: " + std::to_string(rotaryDim));
    }

    int32_t maxBlockPerSM{};
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlockPerSM, kernelPtr, 128, 0));
    int32_t const numBlocks = std::max(1, std::min(maxBlockPerSM * numSMs, (rotaryEmbeddingMaxPositions + 7) / 8));
    dim3 grid(numBlocks);

    void* kernelArgs[] = {reinterpret_cast<void*>(&cosSinCache), reinterpret_cast<void*>(&invFreq),
        reinterpret_cast<void*>(&mscale), reinterpret_cast<void*>(&rotaryEmbeddingMaxPositions)};
    CUDA_CHECK(cudaLaunchKernel(kernelPtr, grid, block, kernelArgs, 0, stream));
}

template <int32_t RotaryDim, typename TPos, bool UseInvFreq>
__global__ void initializeMRopeCosSinKernel(float* cosSinCache, TPos* mropePositionIds, float rotaryBaseFrequency,
    float const* invFreqTable, float mscale, int64_t rotaryEmbeddingMaxPositions, bool interleaved, int32_t sectionH,
    int32_t sectionW)
{
    // mropePositionIds: [bs, 3, rotaryEmbeddingMaxPositions]
    //     Represent [T, H, W] information for each token position
    // cosSinCache: [bs, rotaryEmbeddingMaxPositions, rotaryDim]
    //
    // Interleaved format: [THWTHW...TT] where H covers slice(1, sectionH*3, 3)
    //     and W covers slice(2, sectionW*3, 3), rest is T.
    // Non-interleaved format: [TTT...HHH...WWW] chunked by section.

    uint32_t const bIdx = blockIdx.x;
    uint32_t const tIdx = threadIdx.x;
    uint32_t const tIdy = threadIdx.y;

    uint32_t const bDimY = blockDim.y;
    uint32_t const gDimX = gridDim.x;

    uint32_t const startPosIdx = bIdx * bDimY + tIdy;
    uint32_t const posStride = gDimX * bDimY;
    uint32_t const batchIdx = blockIdx.y;
    uint32_t batchPositionIdsOffset = batchIdx * 3 * rotaryEmbeddingMaxPositions;

    // H occupies pair indices {1, 4, ..., 1+3*(sectionH-1)}, max index = sectionH*3 - 2
    // W occupies pair indices {2, 5, ..., 2+3*(sectionW-1)}, max index = sectionW*3 - 1
    int32_t const interleavedHLimit = sectionH * 3;
    int32_t const interleavedWLimit = sectionW * 3;

    // Non-interleaved section boundaries (in units of 8-dim iteration blocks)
    int32_t const numTemporalPairs = RotaryDim / 2 - sectionH - sectionW;
    int32_t const nonInterleavedTEnd = numTemporalPairs / 8;
    int32_t const nonInterleavedHEnd = nonInterleavedTEnd + sectionH / 8;

    // Per-dimension divisors theta^(2d/RotaryDim); inverse frequencies instead when UseInvFreq.
    float ropeConstants[RotaryDim / 16];

#pragma unroll
    for (uint32_t i = 0; i < RotaryDim / 16; ++i)
    {
        uint32_t zid = tIdx + i * 8;
        if constexpr (UseInvFreq)
        {
            ropeConstants[i] = invFreqTable[zid];
        }
        else
        {
            ropeConstants[i] = pow(rotaryBaseFrequency, 2 * zid / (float) RotaryDim);
        }
    }

    for (uint32_t posIdx = startPosIdx; posIdx < rotaryEmbeddingMaxPositions; posIdx += posStride)
    {
        uint32_t cosSinOffset = batchIdx * rotaryEmbeddingMaxPositions * RotaryDim + posIdx * RotaryDim;

#pragma unroll
        for (uint32_t i = 0; i < RotaryDim / 16; ++i)
        {
            uint32_t zid = tIdx + i * 8;

            int32_t groupIdx;
            if (interleaved)
            {
                int32_t mod3 = zid % 3;
                if (mod3 == 1 && zid < interleavedHLimit)
                {
                    groupIdx = 1;
                }
                else if (mod3 == 2 && zid < interleavedWLimit)
                {
                    groupIdx = 2;
                }
                else
                {
                    groupIdx = 0;
                }
            }
            else
            {
                if (i < nonInterleavedTEnd)
                {
                    groupIdx = 0;
                }
                else if (i < nonInterleavedHEnd)
                {
                    groupIdx = 1;
                }
                else
                {
                    groupIdx = 2;
                }
            }
            TPos const mropePosIdx
                = mropePositionIds[batchPositionIdsOffset + groupIdx * rotaryEmbeddingMaxPositions + posIdx];

            float cosVal{};
            float sinVal{};
            if constexpr (UseInvFreq)
            {
                float const angle = mropePosIdx * ropeConstants[i];
                cosVal = cos(angle) * mscale;
                sinVal = sin(angle) * mscale;
            }
            else
            {
                float invFreq = mropePosIdx / ropeConstants[i];
                cosVal = cos(invFreq);
                sinVal = sin(invFreq);
            }

            cosSinCache[cosSinOffset + zid] = cosVal;
            cosSinCache[cosSinOffset + zid + RotaryDim / 2] = sinVal;
        }
    }
}

template <typename TPos>
void initializeMRopeCosSinImpl(float* cosSinCache, TPos* mropePositionIds, float rotaryBaseFrequency, int64_t rotaryDim,
    int64_t rotaryEmbeddingMaxPositions, int64_t batchSize, bool interleaved, int32_t sectionH, int32_t sectionW,
    cudaStream_t stream, float const* invFreq, float mscale)
{
    if (!interleaved)
    {
        int64_t const halfDim = rotaryDim / 2;
        int64_t const numTemporalPairs = halfDim - sectionH - sectionW;
        ELLM_CHECK(numTemporalPairs % 8 == 0 && sectionH % 8 == 0 && sectionW % 8 == 0,
            "Non-interleaved MRoPE sections must be divisible by 8, got T=" + std::to_string(numTemporalPairs)
                + " H=" + std::to_string(sectionH) + " W=" + std::to_string(sectionW));
    }

    // Each CTA get assigned 128 threads.
    dim3 block(8, 16);

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    int32_t const numSMs = deviceProp.multiProcessorCount;

    bool const useInvFreq = invFreq != nullptr;
    void* kernelPtr{nullptr};
    switch (rotaryDim)
    {
    case 64:
        kernelPtr = useInvFreq ? (void*) initializeMRopeCosSinKernel<64, TPos, true>
                               : (void*) initializeMRopeCosSinKernel<64, TPos, false>;
        break;
    case 128:
        kernelPtr = useInvFreq ? (void*) initializeMRopeCosSinKernel<128, TPos, true>
                               : (void*) initializeMRopeCosSinKernel<128, TPos, false>;
        break;
    default:
        throw std::runtime_error("Un-implemented rotaryDim for initializeMRopeCosSin: " + std::to_string(rotaryDim));
    }
    int32_t maxBlockPerSM{};
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlockPerSM, kernelPtr, 128, 0));

    int64_t const numBlocks = std::max(static_cast<int64_t>(1),
        std::min(static_cast<int64_t>(maxBlockPerSM * numSMs), rotaryEmbeddingMaxPositions / 16));
    dim3 grid(numBlocks, batchSize);

    void* kernelArgs[] = {reinterpret_cast<void*>(&cosSinCache), reinterpret_cast<void*>(&mropePositionIds),
        static_cast<void*>(&rotaryBaseFrequency), static_cast<void*>(&invFreq), static_cast<void*>(&mscale),
        static_cast<void*>(&rotaryEmbeddingMaxPositions), static_cast<void*>(&interleaved),
        static_cast<void*>(&sectionH), static_cast<void*>(&sectionW)};
    CUDA_CHECK(cudaLaunchKernel(kernelPtr, grid, block, kernelArgs, 0, stream));
}

void initializeMRopeCosSin(float* cosSinCache, int64_t* mropePositionIds, float rotaryBaseFrequency, int64_t rotaryDim,
    int64_t rotaryEmbeddingMaxPositions, int64_t batchSize, bool interleaved, int32_t sectionH, int32_t sectionW,
    cudaStream_t stream, float const* invFreq, float mscale)
{
    initializeMRopeCosSinImpl(cosSinCache, mropePositionIds, rotaryBaseFrequency, rotaryDim,
        rotaryEmbeddingMaxPositions, batchSize, interleaved, sectionH, sectionW, stream, invFreq, mscale);
}

void initializeMRopeCosSin(float* cosSinCache, float* mropePositionIds, float rotaryBaseFrequency, int64_t rotaryDim,
    int64_t rotaryEmbeddingMaxPositions, int64_t batchSize, bool interleaved, int32_t sectionH, int32_t sectionW,
    cudaStream_t stream, float const* invFreq, float mscale)
{
    initializeMRopeCosSinImpl(cosSinCache, mropePositionIds, rotaryBaseFrequency, rotaryDim,
        rotaryEmbeddingMaxPositions, batchSize, interleaved, sectionH, sectionW, stream, invFreq, mscale);
}

void initializeTextOnlyMRopeCosSin(float* cosSinCache, float rotaryBaseFrequency, int64_t rotaryDim,
    int64_t maxPositions, int64_t batchSize, cudaStream_t stream, float const* invFreq, float mscale)
{
    // When all 3 MRoPE sections (T, H, W) use sequential positions pos[i] = i, the MRoPE
    // formula reduces to 1-D RoPE over the same frequencies. Reuse the 1-D initializers
    // to avoid constructing temporary position ID buffers. Initialize one slice and
    // replicate it across batch slots so later runtime reshape([activeBatch, ...]) is valid.
    if (invFreq != nullptr)
    {
        initializeYarnCosSin(
            cosSinCache, invFreq, mscale, static_cast<int32_t>(rotaryDim), static_cast<int32_t>(maxPositions), stream);
    }
    else
    {
        initializeNormalRopeCosSin(cosSinCache, rotaryBaseFrequency, 1.0f, 1.0f, static_cast<int32_t>(rotaryDim),
            static_cast<int32_t>(maxPositions), stream);
    }

    if (batchSize <= 1)
    {
        return;
    }

    size_t const oneBatchBytes = static_cast<size_t>(maxPositions * rotaryDim * sizeof(float));
    for (int64_t b = 1; b < batchSize; ++b)
    {
        CUDA_CHECK(cudaMemcpyAsync(static_cast<void*>(cosSinCache + b * maxPositions * rotaryDim),
            static_cast<void const*>(cosSinCache), oneBatchBytes, cudaMemcpyDeviceToDevice, stream));
    }
}

} // namespace kernel
} // namespace trt_edgellm
