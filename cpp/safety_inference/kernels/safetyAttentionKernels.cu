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

#include "safetyAttentionKernels.h"
#include "safetyAttentionModules.h"
#include "safetyRopeKernel.cuh"

#include <cmath>
#include <cstddef>

namespace trt_edgellm
{
namespace safety
{
namespace
{
constexpr int32_t kBLOCK_SIZE{128};
constexpr uint64_t kWORKSPACE_ALIGNMENT{256};
// RuntimeErrorType bit assignments from NvInferSafeRecorder.h.
constexpr uint64_t kINDEX_ERROR{(1ULL << 2) | (1ULL << 3)};
constexpr uint64_t kNONFINITE_ERROR{(1ULL << 0) | (1ULL << 1)};

struct AttentionMetadata
{
    int32_t valid;
    int32_t cumulativeQueryLengths[2];
    int32_t cumulativeKeyLengths[2];
};
static_assert(offsetof(AttentionMetadata, valid) == 0);
static_assert(offsetof(AttentionMetadata, cumulativeQueryLengths) == sizeof(int32_t));
static_assert(offsetof(AttentionMetadata, cumulativeKeyLengths) == 3 * sizeof(int32_t));
static_assert(sizeof(AttentionMetadata) == 5 * sizeof(int32_t));
static_assert(sizeof(AttentionMetadata) <= kWORKSPACE_ALIGNMENT);

__device__ void reportDeviceError(uint64_t* mask, uint64_t bits)
{
    atomicOr(reinterpret_cast<unsigned long long*>(mask), static_cast<unsigned long long>(bits));
}

__global__ void validateLlm(
    AttentionLaunch shape, int32_t const* lengths, int32_t const* starts, AttentionMetadata* metadata, uint64_t* error)
{
    int32_t const start = starts == nullptr ? 0 : starts[0];
    int32_t const length = lengths[0];
    bool const good = start >= 0 && length > 0 && length <= shape.tokens && start <= shape.ropeTokens - length
        && start <= shape.capacity - length;
    metadata->valid = good ? 1 : 0;
    // FMHA prefix sums delimit the single sequence: Q covers this chunk, K includes its cached prefix.
    // Both start at logical offset zero; starts[0] is a cache position, not a packed-sequence offset.
    metadata->cumulativeQueryLengths[0] = 0;
    metadata->cumulativeQueryLengths[1] = good ? length : 0;
    metadata->cumulativeKeyLengths[0] = 0;
    metadata->cumulativeKeyLengths[1] = good ? start + length : 0;
    if (!good)
    {
        reportDeviceError(error, kINDEX_ERROR);
    }
}

__global__ void validateVisual(AttentionLaunch shape, int32_t const* ends, int32_t* valid, uint64_t* error)
{
    bool good = ends[0] == 0 && ends[shape.sequences] == shape.tokens;
    for (int32_t i = 0; i < shape.sequences; ++i)
    {
        good = good && ends[i] >= 0 && ends[i + 1] > ends[i] && ends[i + 1] <= shape.tokens
            && ends[i + 1] - ends[i] <= shape.ropeTokens;
    }
    *valid = good ? 1 : 0;
    if (!good)
    {
        reportDeviceError(error, kINDEX_ERROR);
    }
}

__global__ void copyPool(half const* source, half* destination, int64_t count, int32_t const* valid)
{
    int64_t const i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (*valid != 0 && i < count)
    {
        destination[i] = source[i];
    }
}

__global__ void initializeQueryPadding(
    AttentionLaunch shape, int32_t const* lengths, half* queries, int32_t const* valid)
{
    int64_t const i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const rowSize = static_cast<int64_t>(shape.qHeads) * shape.headSize;
    if (*valid != 0 && i < shape.tokens * rowSize && i >= lengths[0] * rowSize)
    {
        queries[i] = __float2half(0.F);
    }
}

__global__ void checkAttentionOutput(
    AttentionLaunch shape, int32_t const* lengths, half* output, int32_t const* valid, uint64_t* error)
{
    int64_t const i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const rowSize = static_cast<int64_t>(shape.qHeads) * shape.headSize;
    if (i >= shape.tokens * rowSize)
    {
        return;
    }
    if (*valid == 0 || (lengths != nullptr && i >= lengths[0] * rowSize))
    {
        output[i] = __float2half(0.F);
    }
    else if (!isfinite(__half2float(output[i])))
    {
        reportDeviceError(error, kNONFINITE_ERROR);
    }
}

template <typename... Args>
cudaError_t launch(void const* kernel, dim3 grid, dim3 block, cudaStream_t stream, Args&... args) noexcept
{
    void* parameters[]{static_cast<void*>(&args)...};
    return cudaLaunchKernel(kernel, grid, block, parameters, 0, stream);
}
} // namespace

uint64_t attentionWorkspaceBytes(AttentionLaunch const& shape) noexcept
{
    return kWORKSPACE_ALIGNMENT + static_cast<uint64_t>(shape.tokens) * shape.qHeads * shape.headSize * sizeof(half);
}

cudaError_t prepareAttentionKernels() noexcept
{
    void const* functions[]{reinterpret_cast<void const*>(validateLlm), reinterpret_cast<void const*>(validateVisual),
        reinterpret_cast<void const*>(copyPool), reinterpret_cast<void const*>(safetyApplyRopeFromPackedToSplit),
        reinterpret_cast<void const*>(initializeQueryPadding), reinterpret_cast<void const*>(checkAttentionOutput)};
    for (void const* function : functions)
    {
        cudaFuncAttributes attributes{};
        auto const status = cudaFuncGetAttributes(&attributes, function);
        if (status != cudaSuccess)
        {
            return status;
        }
    }
    return cudaSuccess;
}

cudaError_t launchSafetyLlmAttention(SafetyAttentionModules& modules, fmha::PreparedBindings& bindings,
    AttentionLaunch const& launchShape, half const* qkv, half const* past, int32_t const* lengths, float const* rope,
    int32_t const* starts, half* output, half* present, void* workspace, uint64_t* errorMask,
    cudaStream_t stream) noexcept
{
    if (stream == nullptr || workspace == nullptr || errorMask == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    AttentionLaunch shape = launchShape;
    auto* metadata = static_cast<AttentionMetadata*>(workspace);
    auto* valid = &metadata->valid;
    // Reserve the first aligned workspace block for metadata, followed by Q[tokens, qHeads, headSize].
    auto* queries = reinterpret_cast<half*>(static_cast<char*>(workspace) + kWORKSPACE_ALIGNMENT);
    if (modules.prepareBindings(bindings, shape, false, queries, present,
            present + static_cast<int64_t>(shape.kvHeads) * shape.capacity * shape.headSize, output)
        != CUDA_SUCCESS)
    {
        return cudaErrorInvalidValue;
    }
    auto status = launch(reinterpret_cast<void const*>(validateLlm), dim3(1), dim3(1), stream, shape, lengths, starts,
        metadata, errorMask);
    if (status != cudaSuccess)
    {
        return status;
    }
    if (past != present)
    {
        int64_t count = static_cast<int64_t>(2) * shape.capacity * shape.kvHeads * shape.headSize;
        status = launch(reinterpret_cast<void const*>(copyPool), dim3((count + kBLOCK_SIZE - 1) / kBLOCK_SIZE),
            dim3(kBLOCK_SIZE), stream, past, present, count, valid);
        if (status != cudaSuccess)
        {
            return status;
        }
    }
    int64_t count = static_cast<int64_t>(shape.tokens) * shape.qHeads * shape.headSize;
    dim3 const elements((count + kBLOCK_SIZE - 1) / kBLOCK_SIZE);
    status = launch(reinterpret_cast<void const*>(initializeQueryPadding), elements, dim3(kBLOCK_SIZE), stream, shape,
        lengths, queries, valid);
    if (status != cudaSuccess)
    {
        return status;
    }
    dim3 const ropeBlock = safetyRopeBlock(shape);
    dim3 const ropeGrid((shape.tokens + ropeBlock.y - 1) / ropeBlock.y, shape.qHeads + shape.kvHeads);
    status = launch(reinterpret_cast<void const*>(safetyApplyRopeFromPackedToSplit), ropeGrid, ropeBlock, stream, shape,
        qkv, lengths, starts, rope, queries, present, valid, errorMask);
    if (status != cudaSuccess)
    {
        return status;
    }
    if (modules.launch(bindings, shape, false, queries, present,
            present + static_cast<int64_t>(shape.kvHeads) * shape.capacity * shape.headSize,
            metadata->cumulativeQueryLengths, metadata->cumulativeKeyLengths, nullptr, output, valid, stream)
        != CUDA_SUCCESS)
    {
        return cudaErrorLaunchFailure;
    }
    return launch(reinterpret_cast<void const*>(checkAttentionOutput), elements, dim3(kBLOCK_SIZE), stream, shape,
        lengths, output, valid, errorMask);
}

cudaError_t launchSafetyVisualAttention(SafetyAttentionModules& modules, fmha::PreparedBindings& bindings,
    AttentionLaunch const& launchShape, half const* q, half const* k, half const* v, int32_t const* sequenceEnds,
    half* output, void* workspace, uint64_t* errorMask, cudaStream_t stream) noexcept
{
    if (stream == nullptr || workspace == nullptr || errorMask == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    AttentionLaunch shape = launchShape;
    auto* valid = static_cast<int32_t*>(workspace);
    if (modules.prepareBindings(bindings, shape, true, q, k, v, output) != CUDA_SUCCESS)
    {
        return cudaErrorInvalidValue;
    }
    auto status = launch(
        reinterpret_cast<void const*>(validateVisual), dim3(1), dim3(1), stream, shape, sequenceEnds, valid, errorMask);
    if (status != cudaSuccess)
    {
        return status;
    }
    if (modules.launch(bindings, shape, true, q, k, v, nullptr, nullptr, sequenceEnds, output, valid, stream)
        != CUDA_SUCCESS)
    {
        return cudaErrorLaunchFailure;
    }
    int64_t count = static_cast<int64_t>(shape.tokens) * shape.qHeads * shape.headSize;
    int32_t const* lengths = nullptr;
    return launch(reinterpret_cast<void const*>(checkAttentionOutput), dim3((count + kBLOCK_SIZE - 1) / kBLOCK_SIZE),
        dim3(kBLOCK_SIZE), stream, shape, lengths, output, valid, errorMask);
}
} // namespace safety
} // namespace trt_edgellm
