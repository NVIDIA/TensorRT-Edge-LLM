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

#include "ropeCache.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace trt_edgellm::safety
{
namespace
{
SafetyStatus invalidRope() noexcept
{
    return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid RoPE cache configuration");
}

bool computeCacheBytes(RopeCacheSpec const& spec, std::size_t& rowBytes, std::size_t& cacheBytes) noexcept
{
    if (spec.contextLength <= 0 || spec.rotaryDimension <= 0 || spec.rotaryDimension % 2 != 0 || spec.theta <= 0.0F)
    {
        return false;
    }
    rowBytes = static_cast<std::size_t>(spec.rotaryDimension) * sizeof(float);
    if (rowBytes / sizeof(float) != static_cast<std::size_t>(spec.rotaryDimension)
        || static_cast<std::size_t>(spec.contextLength) > std::numeric_limits<std::size_t>::max() / rowBytes)
    {
        return false;
    }
    cacheBytes = static_cast<std::size_t>(spec.contextLength) * rowBytes;
    return true;
}

void fillRows(float* output, RopeCacheSpec const& spec, int32_t first, int32_t rows) noexcept
{
    int32_t const half = spec.rotaryDimension / 2;
    for (int32_t row = 0; row < rows; ++row)
    {
        for (int32_t column = 0; column < half; ++column)
        {
            float const denominator = std::pow(spec.theta, 2.0F * column / spec.rotaryDimension);
            float const angle = static_cast<float>(first + row) * spec.scale / denominator;
            output[row * spec.rotaryDimension + column] = std::cos(angle);
            output[row * spec.rotaryDimension + half + column] = std::sin(angle);
        }
    }
}

cudaError_t copyRows(
    void* destination, void const* source, std::size_t bytes, cudaStream_t stream, cudaEvent_t completion) noexcept
{
    auto status = cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess)
    {
        status = cudaEventRecord(completion, stream);
    }
    if (status == cudaSuccess)
    {
        status = cudaEventSynchronize(completion);
    }
    return status;
}
} // namespace

SafetyStatus initializeRopeCache(RopeCacheSpec const& spec, RuntimeMemoryView upload, RuntimeMemoryView destination,
    cudaStream_t stream, cudaEvent_t completion) noexcept
{
    std::size_t rowBytes{0};
    std::size_t cacheBytes{0};
    if (!computeCacheBytes(spec, rowBytes, cacheBytes) || upload.data == nullptr || upload.sizeInBytes < rowBytes
        || upload.kind != RuntimeMemoryKind::kPINNED || destination.data == nullptr
        || destination.sizeInBytes < cacheBytes || destination.kind != RuntimeMemoryKind::kDEVICE || stream == nullptr
        || completion == nullptr)
    {
        return invalidRope();
    }
    int32_t const rowsPerChunk = static_cast<int32_t>(upload.sizeInBytes / rowBytes);
    for (int32_t first = 0; first < spec.contextLength; first += rowsPerChunk)
    {
        int32_t const rows = std::min(spec.contextLength - first, rowsPerChunk);
        fillRows(static_cast<float*>(upload.data), spec, first, rows);
        auto* target = static_cast<char*>(destination.data) + static_cast<std::size_t>(first) * rowBytes;
        if (copyRows(target, upload.data, static_cast<std::size_t>(rows) * rowBytes, stream, completion) != cudaSuccess)
        {
            return SafetyStatus::fatal(SafetyStatusCode::kCudaError, "RoPE cache upload failed");
        }
    }
    return SafetyStatus::ok();
}

} // namespace trt_edgellm::safety
