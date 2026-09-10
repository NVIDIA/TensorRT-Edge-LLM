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
#include "dartGatherKernels.h"

namespace trt_edgellm
{
namespace kernel
{

namespace
{

constexpr int32_t kBlockSize = 256;

__global__ void gatherRowsVec16Kernel(uint4* dst, uint4 const* src, int32_t const* rowIndices, int64_t rowVecs)
{
    int64_t const dstRow = blockIdx.x;
    int64_t const srcRow = rowIndices[dstRow];
    uint4* dstPtr = dst + dstRow * rowVecs;
    uint4 const* srcPtr = src + srcRow * rowVecs;
    for (int64_t v = threadIdx.x; v < rowVecs; v += blockDim.x)
    {
        dstPtr[v] = srcPtr[v];
    }
}

__global__ void gatherRowsByteKernel(uint8_t* dst, uint8_t const* src, int32_t const* rowIndices, int64_t rowBytes)
{
    int64_t const dstRow = blockIdx.x;
    int64_t const srcRow = rowIndices[dstRow];
    uint8_t* dstPtr = dst + dstRow * rowBytes;
    uint8_t const* srcPtr = src + srcRow * rowBytes;
    for (int64_t b = threadIdx.x; b < rowBytes; b += blockDim.x)
    {
        dstPtr[b] = srcPtr[b];
    }
}

bool isAligned16(void const* p)
{
    return (reinterpret_cast<uintptr_t>(p) % 16U) == 0U;
}

} // namespace

void gatherRows(
    void* dst, void const* src, int32_t const* rowIndices, int64_t numRows, int64_t rowBytes, cudaStream_t stream)
{
    check::check(dst != nullptr && src != nullptr && rowIndices != nullptr, "gatherRows: null pointer");
    check::check(numRows >= 0 && rowBytes > 0, "gatherRows: invalid extents");
    if (numRows == 0)
    {
        return;
    }

    if (rowBytes % 16 == 0 && isAligned16(dst) && isAligned16(src))
    {
        gatherRowsVec16Kernel<<<static_cast<uint32_t>(numRows), kBlockSize, 0, stream>>>(
            static_cast<uint4*>(dst), static_cast<uint4 const*>(src), rowIndices, rowBytes / 16);
    }
    else
    {
        gatherRowsByteKernel<<<static_cast<uint32_t>(numRows), kBlockSize, 0, stream>>>(
            static_cast<uint8_t*>(dst), static_cast<uint8_t const*>(src), rowIndices, rowBytes);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
