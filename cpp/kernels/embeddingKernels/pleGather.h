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

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace trt_edgellm
{
namespace kernel
{
//! Copy rows verbatim. Every rowMap entry must identify an allocated, ready pool row.
cudaError_t launchPleGather(void const* pool, int32_t const* rowMap, void* output, int64_t rowCount, int64_t rowBytes,
    cudaStream_t stream) noexcept;

//! Map Gemma token IDs to resident slots; optional query lengths mask entry-padded tokens.
cudaError_t launchGemmaPleResidentMap(int32_t const* tokenIds, int32_t* rowMap, int64_t count, int64_t vocabularySize,
    int32_t padTokenId, int32_t imageTokenId, int32_t audioTokenId, cudaStream_t stream,
    int32_t const* queryLengths = nullptr, int64_t queryWidth = 0) noexcept;

} // namespace kernel
} // namespace trt_edgellm
