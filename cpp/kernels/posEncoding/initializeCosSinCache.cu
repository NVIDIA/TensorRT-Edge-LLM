/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

void initializeNormalRopeCosSin(float* cosSinCache, float rotaryBaseFrequency, float rotaryScale, int32_t rotaryDim,
    int32_t rotaryEmbeddingMaxPositions, cudaStream_t stream)
{
    // Each CTA get assigned 128 threads.
    dim3 block(32, 4);

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    int32_t const numSMs = deviceProp.multiProcessorCount;

    void* kernelPtr{nullptr};
    switch (rotaryDim)
    {
    case 64: kernelPtr = (void*) initializeNormalRopeCosSinKernel<64>; break;
    case 128: kernelPtr = (void*) initializeNormalRopeCosSinKernel<128>; break;
    default:
        throw std::runtime_error(
            "Un-implemented rotaryDim for initializeNormalRopeCosSin: " + std::to_string(rotaryDim));
    }
    int32_t maxBlockPerSM{};
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlockPerSM, kernelPtr, 128, 0));

    int32_t const numBlocks = std::min(maxBlockPerSM * numSMs, rotaryEmbeddingMaxPositions / 4);
    dim3 grid(numBlocks);

    void* kernelArgs[] = {reinterpret_cast<void*>(&cosSinCache), reinterpret_cast<void*>(&rotaryBaseFrequency),
        reinterpret_cast<void*>(&rotaryScale), reinterpret_cast<void*>(&rotaryEmbeddingMaxPositions)};
    CUDA_CHECK(cudaLaunchKernel(kernelPtr, grid, block, kernelArgs, 0, stream));
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
__global__ void initializeMRopeCosSinKernel(float* cosSinCache, int64_t* mropePositionIds, float rotaryBaseFrequency,
    int64_t rotaryEmbeddingMaxPositions, bool interleaved)
{
    // In this kernel, each warp compute 4 "position" of the cos/sin cache, and loop until max position.
    // Each CTA will be assigned 4 warps so it proceeds 16 positions in an iteration.
    // mropePositionIds: [bs, 3, rotaryEmbeddingMaxPositions]
    //     Represent [T, H, W] information for each token position
    // cosSinCache: [bs, rotaryEmbeddingMaxPositions, rotaryDim]
    //     Combine [T, H, W] information into rotaryDim. Each will take [16, 24, 24] dims.

    uint32_t const bIdx = blockIdx.x;
    uint32_t const tIdx = threadIdx.x;
    uint32_t const tIdy = threadIdx.y;

    uint32_t const bDimY = blockDim.y;
    uint32_t const gDimX = gridDim.x;

    uint32_t const startPosIdx = bIdx * bDimY + tIdy;
    uint32_t const posStride = gDimX * bDimY;
    uint32_t const batchIdx = blockIdx.y;
    uint32_t batchPositionIdsOffset = batchIdx * 3 * rotaryEmbeddingMaxPositions;

    float ropeConstants[RotaryDim / 16];

#pragma unroll
    for (uint32_t i = 0; i < RotaryDim / 16; ++i)
    {
        uint32_t zid = tIdx + i * 8;
        ropeConstants[i] = pow(rotaryBaseFrequency, 2 * zid / (float) RotaryDim);
    }

    for (uint32_t posIdx = startPosIdx; posIdx < rotaryEmbeddingMaxPositions; posIdx += posStride)
    {
        uint32_t cosSinOffset = batchIdx * rotaryEmbeddingMaxPositions * RotaryDim + posIdx * RotaryDim;

#pragma unroll
        for (uint32_t i = 0; i < RotaryDim / 16; ++i)
        {
            uint32_t zid = tIdx + i * 8;

            // Determine which group of T, H, W to use based on interleaved flag.
            // Non-interleaved format: [TTT...HHH...WWW] Qwen2-VL
            //     mrope section is [16, 24, 24], dims 0~15 is T, 16~39 is H, 40~63 is W
            //     Each iteration i processes 8 dims, i 0~1 is T, 2~4 is H, 5~7 is W
            // Interleaved format: [THWTHWHTHW...TTTT] Qwen3-VL
            //     mrope section is [24, 20, 20]
            //     [THWTHWHTHW...TTTT] has 20 groups of THW and 4 additional T
            int32_t groupIdx;
            if (interleaved)
            {
                if (zid >= 60)
                {
                    groupIdx = 0;
                }
                else
                {
                    groupIdx = zid % 3;
                }
            }
            else
            {
                if (i < 2)
                {
                    groupIdx = 0;
                }
                else if (i < 5)
                {
                    groupIdx = 1;
                }
                else
                {
                    groupIdx = 2;
                }
            }
            int64_t mropePosIdx
                = mropePositionIds[batchPositionIdsOffset + groupIdx * rotaryEmbeddingMaxPositions + posIdx];

            float invFreq = mropePosIdx / ropeConstants[i];
            float cosVal = cos(invFreq);
            float sinVal = sin(invFreq);

            cosSinCache[cosSinOffset + zid] = cosVal;
            cosSinCache[cosSinOffset + zid + RotaryDim / 2] = sinVal;
        }
    }
}

void initializeMRopeCosSin(float* cosSinCache, int64_t* mropePositionIds, float rotaryBaseFrequency, int64_t rotaryDim,
    int64_t rotaryEmbeddingMaxPositions, int64_t batchSize, bool interleaved, cudaStream_t stream)
{
    // Each CTA get assigned 128 threads.
    dim3 block(8, 16);

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    int32_t const numSMs = deviceProp.multiProcessorCount;

    void* kernelPtr{nullptr};
    switch (rotaryDim)
    {
    case 128: kernelPtr = (void*) initializeMRopeCosSinKernel<128>; break;
    default:
        throw std::runtime_error("Un-implemented rotaryDim for initializeMRopeCosSin: " + std::to_string(rotaryDim));
    }
    int32_t maxBlockPerSM{};
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlockPerSM, kernelPtr, 128, 0));

    int64_t const numBlocks = std::min(static_cast<int64_t>(maxBlockPerSM * numSMs), rotaryEmbeddingMaxPositions / 16);
    dim3 grid(numBlocks, batchSize);

    void* kernelArgs[] = {reinterpret_cast<void*>(&cosSinCache), reinterpret_cast<void*>(&mropePositionIds),
        static_cast<void*>(&rotaryBaseFrequency), static_cast<void*>(&rotaryEmbeddingMaxPositions),
        static_cast<void*>(&interleaved)};
    CUDA_CHECK(cudaLaunchKernel(kernelPtr, grid, block, kernelArgs, 0, stream));
}

} // namespace kernel
} // namespace trt_edgellm
