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

#include "fp16Embedding.h"

#include <cuda_runtime.h>

namespace trt_edgellm::safety
{

inline constexpr int32_t kFp16EmbeddingThreads{256};

__global__ void fp16EmbeddingLookup(int32_t const* tokenIds, std::size_t tokenCount, half const* table,
    int32_t vocabularySize, int32_t hiddenSize, half* output)
{
    std::size_t const row = blockIdx.x;
    int32_t const column = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (row >= tokenCount || column >= hiddenSize)
    {
        return;
    }
    int32_t const token = tokenIds[row];
    if (token >= 0 && token < vocabularySize)
    {
        std::size_t const outputIndex = row * static_cast<std::size_t>(hiddenSize) + column;
        output[outputIndex] = table[static_cast<std::size_t>(token) * hiddenSize + column];
    }
}

} // namespace trt_edgellm::safety
