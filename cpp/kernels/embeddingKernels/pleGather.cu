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

#include "pleGather.h"

#include <algorithm>
#include <limits>

namespace trt_edgellm
{
namespace kernel
{
namespace
{
__global__ void pleGatherKernel(
    uint8_t const* pool, int32_t const* rowMap, uint8_t* output, int64_t rows, int64_t rowBytes)
{
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x)
    {
        int64_t const source = static_cast<int64_t>(rowMap[row]) * rowBytes;
        int64_t const destination = row * rowBytes;
        for (int64_t column = threadIdx.x; column < rowBytes; column += blockDim.x)
        {
            output[destination + column] = pool[source + column];
        }
    }
}

__global__ void gemmaPleResidentMapKernel(int32_t const* tokenIds, int32_t* rowMap, int64_t count,
    int64_t vocabularySize, int32_t padTokenId, int32_t imageTokenId, int32_t audioTokenId, int32_t const* queryLengths,
    int64_t queryWidth)
{
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
        i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int32_t const token = tokenIds[i];
        bool const media = (imageTokenId >= 0 && token == imageTokenId) || (audioTokenId >= 0 && token == audioTokenId);
        int32_t const lookupToken = media ? padTokenId : token;
        bool const validPosition = queryLengths == nullptr || i % queryWidth < queryLengths[i / queryWidth];
        rowMap[i] = validPosition && lookupToken >= 0 && lookupToken < vocabularySize ? lookupToken + 1 : 0;
    }
}
} // namespace

cudaError_t launchPleGather(void const* pool, int32_t const* rowMap, void* output, int64_t rowCount, int64_t rowBytes,
    cudaStream_t stream) noexcept
{
    if (pool == nullptr || rowMap == nullptr || output == nullptr || stream == nullptr || rowCount <= 0 || rowBytes <= 0
        || rowCount > std::numeric_limits<int64_t>::max() / rowBytes)
    {
        return cudaErrorInvalidValue;
    }
    int32_t const blocks = static_cast<int32_t>(std::min<int64_t>(rowCount, 65535));
    pleGatherKernel<<<blocks, 128, 0, stream>>>(
        static_cast<uint8_t const*>(pool), rowMap, static_cast<uint8_t*>(output), rowCount, rowBytes);
    return cudaGetLastError();
}

cudaError_t launchGemmaPleResidentMap(int32_t const* tokenIds, int32_t* rowMap, int64_t count, int64_t vocabularySize,
    int32_t padTokenId, int32_t imageTokenId, int32_t audioTokenId, cudaStream_t stream, int32_t const* queryLengths,
    int64_t queryWidth) noexcept
{
    if (tokenIds == nullptr || rowMap == nullptr || stream == nullptr || count <= 0 || vocabularySize <= 0
        || vocabularySize >= std::numeric_limits<int32_t>::max()
        || (queryLengths != nullptr && (queryWidth <= 0 || count % queryWidth != 0)))
    {
        return cudaErrorInvalidValue;
    }
    auto const blocks = static_cast<int32_t>(std::min<int64_t>((count - 1) / 256 + 1, 65535));
    gemmaPleResidentMapKernel<<<blocks, 256, 0, stream>>>(
        tokenIds, rowMap, count, vocabularySize, padTokenId, imageTokenId, audioTokenId, queryLengths, queryWidth);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
