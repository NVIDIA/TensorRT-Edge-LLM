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

#include "fp16EmbeddingKernels.cuh"

#include <cuda_runtime_api.h>
#include <limits>

namespace trt_edgellm::safety
{
namespace
{
constexpr unsigned kMaximumGridY{65535U};
}

cudaError_t launchFp16EmbeddingLookup(int32_t const* tokenIds, std::size_t tokenCount, half const* table,
    int32_t vocabularySize, int32_t hiddenSize, half* output, cudaStream_t stream) noexcept
{
    if (tokenIds == nullptr || tokenCount == 0 || table == nullptr || vocabularySize <= 0 || hiddenSize <= 0
        || output == nullptr || stream == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    unsigned const columnBlocks
        = (static_cast<unsigned>(hiddenSize) + kFp16EmbeddingThreads - 1) / kFp16EmbeddingThreads;
    if (tokenCount > std::numeric_limits<unsigned>::max() || columnBlocks > kMaximumGridY)
    {
        return cudaErrorInvalidValue;
    }
    dim3 const grid{static_cast<unsigned>(tokenCount), columnBlocks, 1};
    fp16EmbeddingLookup<<<grid, kFp16EmbeddingThreads, 0, stream>>>(
        tokenIds, tokenCount, table, vocabularySize, hiddenSize, output);
    return cudaPeekAtLastError();
}

} // namespace trt_edgellm::safety
