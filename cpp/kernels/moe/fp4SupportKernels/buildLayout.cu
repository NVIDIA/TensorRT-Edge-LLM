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

// MoE layout builder: GPU single-CTA kernel.
// All buffers are pre-allocated by the caller.

#include "buildLayout.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

// ---- GPU kernel ----

constexpr int32_t kBlockSize = 256;

__global__ void __launch_bounds__(kBlockSize) build_layout_kernel(int32_t const* __restrict__ token_selected_experts,
    int32_t* __restrict__ permuted_idx, int32_t* __restrict__ tile_group_idx, int32_t* __restrict__ tile_mn_limit,
    int32_t* __restrict__ num_non_exiting_tiles, int32_t num_expanded, int32_t local_num_experts, int32_t tile_size,
    int32_t max_mPadded)
{
    extern __shared__ int32_t smem[];
    int32_t* s_expert_counts = smem;
    int32_t* s_expert_offsets = smem + local_num_experts;
    int32_t* s_scatter_counters = smem + 2 * local_num_experts;

    int32_t const tid = threadIdx.x;
    int32_t const blockDim_x = blockDim.x;

    // Phase 1: Histogram
    for (int32_t e = tid; e < local_num_experts; e += blockDim_x)
        s_expert_counts[e] = 0;
    __syncthreads();

    for (int32_t i = tid; i < num_expanded; i += blockDim_x)
    {
        int32_t expert = token_selected_experts[i];
        if (expert >= 0 && expert < local_num_experts)
            atomicAdd_block(&s_expert_counts[expert], 1);
    }
    __syncthreads();

    // Phase 2: Prefix sum + tile metadata + fill -1. Each thread owns a contiguous expert range; one block-wide scan
    // of tile counts gives every expert's first tile, and its padded offset is that tile index times tile_size.
    using BlockScan = cub::BlockScan<int32_t, kBlockSize>;
    __shared__ typename BlockScan::TempStorage scan_storage;
    int32_t const experts_per_thread = (local_num_experts + kBlockSize - 1) / kBlockSize;
    int32_t const first_expert = min(tid * experts_per_thread, local_num_experts);
    int32_t const end_expert = min(first_expert + experts_per_thread, local_num_experts);
    int32_t thread_tiles = 0;
    for (int32_t e = first_expert; e < end_expert; e++)
        thread_tiles += (s_expert_counts[e] + tile_size - 1) / tile_size;
    int32_t running_tiles = 0;
    int32_t total_tiles = 0;
    BlockScan(scan_storage).ExclusiveSum(thread_tiles, running_tiles, total_tiles);
    for (int32_t e = first_expert; e < end_expert; e++)
    {
        int32_t count = s_expert_counts[e];
        int32_t ntiles = (count + tile_size - 1) / tile_size;
        int32_t running_offset = running_tiles * tile_size;
        s_expert_offsets[e] = running_offset;
        for (int32_t t = 0; t < ntiles; t++)
        {
            tile_group_idx[running_tiles + t] = e;
            int32_t valid_limit = (t + 1) * tile_size;
            if (valid_limit > count)
                valid_limit = count;
            tile_mn_limit[running_tiles + t] = running_offset + valid_limit;
        }
        running_tiles += ntiles;
    }
    if (tid == 0)
        num_non_exiting_tiles[0] = total_tiles;
    __syncthreads();

    for (int32_t i = tid; i < max_mPadded; i += blockDim_x)
        permuted_idx[i] = -1;
    for (int32_t e = tid; e < local_num_experts; e += blockDim_x)
        s_scatter_counters[e] = 0;
    __syncthreads();

    // Phase 3: Scatter
    for (int32_t i = tid; i < num_expanded; i += blockDim_x)
    {
        int32_t expert = token_selected_experts[i];
        if (expert >= 0 && expert < local_num_experts)
        {
            int32_t pos = atomicAdd_block(&s_scatter_counters[expert], 1);
            permuted_idx[s_expert_offsets[expert] + pos] = i;
        }
    }
}

} // namespace

// =========================================================================
// GPU layout builder — writes to pre-allocated MoELayoutBuffers
// =========================================================================

void buildLayoutGpu(MoELayoutBuffers& buf, int32_t const* tokenSelectedExperts, int32_t numTokens, int32_t topK,
    int32_t localNumExperts, int32_t tileSize, cudaStream_t stream)
{
    assert(numTokens >= 0 && topK > 0 && localNumExperts > 0 && tileSize > 0);

    int32_t const maxMPadded = static_cast<int32_t>(buf.permutedIdxToExpandedIdx.getShape()[0]);
    int32_t const numExpanded = numTokens * topK;
    if (numExpanded == 0)
    {
        cudaMemsetAsync(buf.numNonExitingTiles.dataPointer<int32_t>(), 0, sizeof(int32_t), stream);
        if (maxMPadded > 0)
            cudaMemsetAsync(buf.permutedIdxToExpandedIdx.dataPointer<int32_t>(), 0xFF,
                static_cast<size_t>(maxMPadded) * sizeof(int32_t), stream);
        return;
    }

    size_t const smemBytes = static_cast<size_t>(3 * localNumExperts) * sizeof(int32_t);

    build_layout_kernel<<<1, kBlockSize, smemBytes, stream>>>(tokenSelectedExperts,
        buf.permutedIdxToExpandedIdx.dataPointer<int32_t>(), buf.tileIdxToGroupIdx.dataPointer<int32_t>(),
        buf.tileIdxToMnLimit.dataPointer<int32_t>(), buf.numNonExitingTiles.dataPointer<int32_t>(), numExpanded,
        localNumExperts, tileSize, maxMPadded);
}

} // namespace kernel
} // namespace trt_edgellm
