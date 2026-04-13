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

/*
 * CUDA kernels for AlpamayoExpertRunner.
 */

#include "alpamayoExpertKernels.h"

namespace trt_edgellm
{
namespace kernel
{

namespace
{

__global__ void kvCacheReshapeRepeatKernel(float* __restrict__ dst, half const* __restrict__ src, int32_t numLayers,
    int32_t numKVHeads, int32_t seqLen, int32_t headDim, int32_t numCandidates, int64_t totalElements)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= totalElements)
        return;

    // Decompose: dst[layer2, cand, head, s, d]
    int64_t rem = idx;
    int32_t const d = rem % headDim;
    rem /= headDim;
    int32_t const s = rem % seqLen;
    rem /= seqLen;
    int32_t const head = rem % numKVHeads;
    rem /= numKVHeads;
    rem /= numCandidates; // skip cand dimension (broadcast)
    int32_t const layer2 = rem;

    int32_t const layer = layer2 / 2;
    int32_t const kv = layer2 % 2;

    // src layout: [layer, kv, head, s, d] contiguous
    int64_t const hsd = static_cast<int64_t>(numKVHeads) * seqLen * headDim;
    int64_t const srcIdx = (static_cast<int64_t>(layer) * 2 + kv) * hsd + static_cast<int64_t>(head) * seqLen * headDim
        + static_cast<int64_t>(s) * headDim + d;

    dst[idx] = __half2float(src[srcIdx]);
}

__global__ void buildPositionIdsKernel(int64_t* __restrict__ posIds, int32_t total, int32_t numTokens, int64_t basePos)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    posIds[idx] = basePos + (idx % numTokens);
}

__global__ void fillTimestepKernel(float* __restrict__ dst, int32_t n, float val)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        dst[idx] = val;
}

__global__ void eulerUpdateKernel(float* __restrict__ x, float const* __restrict__ v, float dt, int32_t n)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        x[idx] += v[idx] * dt;
}

} // anonymous namespace

void kvCacheReshapeRepeat(float* dst, half const* src, int32_t numLayers, int32_t numKVHeads, int32_t seqLen,
    int32_t headDim, int32_t numCandidates, cudaStream_t stream)
{
    int64_t const totalElements = static_cast<int64_t>(numLayers) * 2 * numCandidates * numKVHeads * seqLen * headDim;
    int32_t const blockSize = 256;
    int32_t const numBlocks = static_cast<int32_t>((totalElements + blockSize - 1) / blockSize);
    kvCacheReshapeRepeatKernel<<<numBlocks, blockSize, 0, stream>>>(
        dst, src, numLayers, numKVHeads, seqLen, headDim, numCandidates, totalElements);
}

void buildPositionIds(int64_t* posIds, int32_t numCandidates, int32_t numTokens, int64_t basePos, cudaStream_t stream)
{
    int32_t const total = 3 * numCandidates * numTokens;
    int32_t const blockSize = 256;
    int32_t const numBlocks = (total + blockSize - 1) / blockSize;
    buildPositionIdsKernel<<<numBlocks, blockSize, 0, stream>>>(posIds, total, numTokens, basePos);
}

void fillTimestep(float* dst, int32_t numCandidates, float tVal, cudaStream_t stream)
{
    int32_t const blockSize = 256;
    int32_t const numBlocks = (numCandidates + blockSize - 1) / blockSize;
    fillTimestepKernel<<<numBlocks, blockSize, 0, stream>>>(dst, numCandidates, tVal);
}

void eulerUpdate(float* x, float const* v, float dt, int32_t n, cudaStream_t stream)
{
    int32_t const blockSize = 256;
    int32_t const numBlocks = (n + blockSize - 1) / blockSize;
    eulerUpdateKernel<<<numBlocks, blockSize, 0, stream>>>(x, v, dt, n);
}

} // namespace kernel
} // namespace trt_edgellm
