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

#include "common/cudaUtils.h"
#include "moeAlignSumKernels.h"
#include <algorithm>
#include <cub/cub.cuh>

namespace trt_edgellm
{
namespace kernel
{

constexpr int32_t WARP_SIZE = 32;

// Count slots per expert using shared memory to reduce global atomic contention.
// Uses warp-level work distribution for better load balancing.
__global__ void countSlotsPerExpertKernel(int32_t const* topkIndices, int32_t* slotsPerExpertWorkspace,
    int32_t numTokens, int32_t topK, int32_t numExperts, int32_t paddedNumExperts, int32_t expertsPerWarp)
{
    extern __shared__ int32_t sharedCounts[];

    int32_t totalSlots = numTokens * topK;

    // Initialize shared memory counts (each warp handles a group of experts)
    int32_t warpId = threadIdx.x / WARP_SIZE;
    int32_t expertStart = warpId * expertsPerWarp;

    for (int32_t i = 0; i < expertsPerWarp; ++i)
    {
        if (expertStart + i < paddedNumExperts)
        {
            sharedCounts[warpId * expertsPerWarp + i] = 0;
        }
    }

    __syncthreads();

    // Count slots per expert in shared memory
    int32_t tid = threadIdx.x;
    int32_t stride = blockDim.x;

    for (int32_t slot = tid; slot < totalSlots; slot += stride)
    {
        int32_t expertId = topkIndices[slot];
        if (expertId < 0 || expertId >= numExperts)
        {
            continue;
        }
        int32_t warpIdx = expertId / expertsPerWarp;
        int32_t expertOffset = expertId % expertsPerWarp;
        atomicAdd(&sharedCounts[warpIdx * expertsPerWarp + expertOffset], 1);
    }

    __syncthreads();

    // Write shared counts to global memory
    if (tid < numExperts)
    {
        int32_t warpIdx = tid / expertsPerWarp;
        int32_t expertOffset = tid % expertsPerWarp;
        slotsPerExpertWorkspace[tid] = sharedCounts[warpIdx * expertsPerWarp + expertOffset];
    }
}

// Compute padded offsets using CUB BlockScan for parallel prefix sum.
// Requires exactly 1024 threads for BlockScan template.
__global__ void computePaddedOffsetsKernel(int32_t const* counts, int32_t* paddedCounts, int32_t* paddedOffsets,
    int32_t* numTokensPostPadded, int32_t numExperts, int32_t moeBlockSize)
{
    using BlockScan = cub::BlockScan<int32_t, 1024>;
    __shared__ typename BlockScan::TempStorage tempStorage;

    int32_t expertId = threadIdx.x;
    int32_t count = 0;
    int32_t paddedCount = 0;

    if (expertId < numExperts)
    {
        count = counts[expertId];
        paddedCount = (count > 0) ? static_cast<int32_t>(divUp(count, moeBlockSize)) * moeBlockSize : 0;
        paddedCounts[expertId] = paddedCount;
    }

    // Parallel prefix sum using CUB
    int32_t offset = 0;
    BlockScan(tempStorage).ExclusiveSum(paddedCount, offset);

    if (expertId < numExperts)
    {
        paddedOffsets[expertId] = offset;
    }

    // Last thread writes total
    if (expertId == numExperts - 1)
    {
        numTokensPostPadded[0] = offset + paddedCount;
    }
}

__global__ void buildSlotListsKernel(int32_t const* topkIndices, int32_t* slotsByExpertWorkspace,
    int32_t* slotsPerExpertWorkspace, int32_t numTokens, int32_t topK, int32_t numExperts)
{
    int32_t totalSlots = numTokens * topK;
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;

    for (int32_t slot = idx; slot < totalSlots; slot += stride)
    {
        int32_t expertId = topkIndices[slot];
        if (expertId < 0 || expertId >= numExperts)
        {
            continue;
        }
        int32_t offset = atomicAdd(&slotsPerExpertWorkspace[expertId], 1);
        slotsByExpertWorkspace[expertId * totalSlots + offset] = slot;
    }
}

constexpr int32_t kSlotListThreads = 256;
constexpr int32_t kGroupedSlotListExperts = 8;
constexpr int32_t kGroupedSlotListThreads = kGroupedSlotListExperts * WARP_SIZE;

template <int32_t kItems>
__global__ void buildStableSlotListsKernel(int32_t const* topkIndices, int32_t* slotsByExpertWorkspace,
    int32_t* slotsPerExpertWorkspace, int32_t numTokens, int32_t topK)
{
    using BlockLoad = cub::BlockLoad<int32_t, kSlotListThreads, kItems,
        kItems == 1 ? cub::BLOCK_LOAD_DIRECT : cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockScan = cub::BlockScan<int32_t, kSlotListThreads>;
    __shared__ union
    {
        typename BlockLoad::TempStorage load;
        typename BlockScan::TempStorage scan;
    } storage;
    int32_t const expertId = blockIdx.x;
    int32_t const totalSlots = numTokens * topK;
    int32_t count = 0;
    // Stable expert rows prevent atomic arrival order from changing Marlin reduction partitions.
    for (int64_t start = 0; start < totalSlots; start += kSlotListThreads * kItems)
    {
        int32_t ids[kItems];
        int32_t const validSlots
            = static_cast<int32_t>(min(static_cast<int64_t>(kSlotListThreads * kItems), totalSlots - start));
        BlockLoad(storage.load).Load(topkIndices + start, ids, validSlots, -1);
        if constexpr (kItems > 1)
        {
            __syncthreads();
        }
        // A bit per consecutive slot keeps the local scan in registers.
        uint32_t mask = 0;
#pragma unroll
        for (int32_t item = 0; item < kItems; ++item)
        {
            mask |= static_cast<uint32_t>(ids[item] == expertId) << item;
        }
        int32_t offset;
        int32_t tileCount;
        BlockScan(storage.scan).ExclusiveSum(__popc(mask), offset, tileCount);
        while (mask)
        {
            int32_t const item = __ffs(mask) - 1;
            slotsByExpertWorkspace[static_cast<int64_t>(expertId) * totalSlots + count + offset++]
                = static_cast<int32_t>(start + threadIdx.x * kItems + item);
            mask &= mask - 1;
        }
        count += tileCount;
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        slotsPerExpertWorkspace[expertId] = count;
    }
}

__global__ void buildStableSlotListsGroupedKernel(int32_t const* topkIndices, int32_t* slotsByExpertWorkspace,
    int32_t* slotsPerExpertWorkspace, int32_t numTokens, int32_t topK, int32_t numExperts)
{
    __shared__ int32_t expertIds[kGroupedSlotListThreads];
    int32_t const laneId = threadIdx.x % WARP_SIZE;
    int32_t const warpId = threadIdx.x / WARP_SIZE;
    int32_t const expertId = blockIdx.x * kGroupedSlotListExperts + warpId;
    int32_t const totalSlots = numTokens * topK;
    int32_t count = 0;

    for (int32_t start = 0; start < totalSlots; start += kGroupedSlotListThreads)
    {
        int32_t const validSlots = min(kGroupedSlotListThreads, totalSlots - start);
        expertIds[threadIdx.x] = threadIdx.x < validSlots ? topkIndices[start + threadIdx.x] : -1;
        __syncthreads();

        if (expertId < numExperts)
        {
#pragma unroll
            for (int32_t warpStart = 0; warpStart < kGroupedSlotListThreads; warpStart += WARP_SIZE)
            {
                int32_t const localSlot = warpStart + laneId;
                bool const selected = localSlot < validSlots && expertIds[localSlot] == expertId;
                uint32_t const selectedMask = __ballot_sync(0xffffffffU, selected);
                if (selected)
                {
                    int32_t const rank = __popc(selectedMask & ((1U << laneId) - 1));
                    slotsByExpertWorkspace[static_cast<int64_t>(expertId) * totalSlots + count + rank]
                        = start + localSlot;
                }
                count += __popc(selectedMask);
            }
        }
        __syncthreads();
    }

    if (expertId < numExperts && laneId == 0)
    {
        slotsPerExpertWorkspace[expertId] = count;
    }
}

void launchCountSlotsPerExpertKernel(int32_t const* topkIndices, int32_t* slotsPerExpertWorkspace, int32_t numTokens,
    int32_t topK, int32_t numExperts, cudaStream_t stream)
{
    int32_t const expertsPerWarp = WARP_SIZE;
    int32_t const paddedNumExperts = static_cast<int32_t>(trt_edgellm::divUp(numExperts, WARP_SIZE)) * WARP_SIZE;
    int32_t threads = 1024;
    threads = static_cast<int32_t>(trt_edgellm::divUp(threads, WARP_SIZE)) * WARP_SIZE;
    size_t sharedMemSize = trt_edgellm::divUp(paddedNumExperts, expertsPerWarp) * expertsPerWarp * sizeof(int32_t);
    countSlotsPerExpertKernel<<<1, threads, sharedMemSize, stream>>>(
        topkIndices, slotsPerExpertWorkspace, numTokens, topK, numExperts, paddedNumExperts, expertsPerWarp);
}

void launchComputePaddedOffsetsKernel(int32_t const* counts, int32_t* paddedCounts, int32_t* paddedOffsets,
    int32_t* numTokensPostPadded, int32_t numExperts, int32_t moeBlockSize, cudaStream_t stream)
{
    computePaddedOffsetsKernel<<<1, 1024, 0, stream>>>(
        counts, paddedCounts, paddedOffsets, numTokensPostPadded, numExperts, moeBlockSize);
}

bool shouldUseGroupedStableSlotLists(int32_t smVersion, int32_t numExperts)
{
    return smVersion == 110 && numExperts >= 128;
}

void launchBuildStableSlotListsKernel(int32_t const* topkIndices, int32_t* slotsByExpertWorkspace,
    int32_t* slotsPerExpertWorkspace, int32_t numTokens, int32_t topK, int32_t numExperts, cudaStream_t stream)
{
    // SM110 benefits from sharing each input tile across expert warps; SM100 favors the per-expert grid.
    if (shouldUseGroupedStableSlotLists(getSMVersion(), numExperts))
    {
        int32_t const blocks = static_cast<int32_t>(divUp(numExperts, kGroupedSlotListExperts));
        buildStableSlotListsGroupedKernel<<<blocks, kGroupedSlotListThreads, 0, stream>>>(
            topkIndices, slotsByExpertWorkspace, slotsPerExpertWorkspace, numTokens, topK, numExperts);
    }
    else if (static_cast<int64_t>(numTokens) * topK <= kSlotListThreads)
    {
        buildStableSlotListsKernel<1><<<numExperts, kSlotListThreads, 0, stream>>>(
            topkIndices, slotsByExpertWorkspace, slotsPerExpertWorkspace, numTokens, topK);
    }
    else
    {
        buildStableSlotListsKernel<32><<<numExperts, kSlotListThreads, 0, stream>>>(
            topkIndices, slotsByExpertWorkspace, slotsPerExpertWorkspace, numTokens, topK);
    }
}

void launchBuildSlotListsKernel(int32_t const* topkIndices, int32_t* slotsByExpertWorkspace,
    int32_t* slotsPerExpertWorkspace, int32_t numTokens, int32_t topK, int32_t numExperts, cudaStream_t stream)
{
    int32_t dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int32_t numSMs = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));
    int32_t totalSlots = numTokens * topK;
    int32_t threads = 256;
    int32_t blocks = static_cast<int32_t>(trt_edgellm::divUp(totalSlots, threads));
    blocks = std::min(blocks, numSMs * 4);
    buildSlotListsKernel<<<blocks, threads, 0, stream>>>(
        topkIndices, slotsByExpertWorkspace, slotsPerExpertWorkspace, numTokens, topK, numExperts);
}

} // namespace kernel
} // namespace trt_edgellm
