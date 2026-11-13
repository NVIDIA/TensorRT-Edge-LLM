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

#include "kernels/lora/groupedLoraGemm.h"

#if defined(CUTE_DSL_LORA_BLACKWELL_ENABLED)
#include "kernels/cuteDslModuleLoader.h"
#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kRANK_ALIGNMENT{8};
constexpr int32_t kMAX_CUTE_DSL_GROUPS{256};
constexpr int32_t kTENSORMAPS_PER_BLOCK{3};
constexpr int32_t kBYTES_PER_TENSORMAP{128};
constexpr size_t kWORKSPACE_ALIGNMENT{256};

constexpr int32_t getRankStride(int32_t maxRank)
{
    return (maxRank + kRANK_ALIGNMENT - 1) / kRANK_ALIGNMENT * kRANK_ALIGNMENT;
}

#if defined(CUTE_DSL_LORA_BLACKWELL_ENABLED)
detail::LazyKernelModule<lora_blackwell_grouped_fp16_Kernel_Module_t> gLoraModule{};
detail::LazyKernelModule<lora_blackwell_grouped_fp16_fused_Kernel_Module_t> gFusedLoraModule{};
detail::LazyKernelModule<lora_blackwell_metadata_Kernel_Module_t> gMetadataModule{};
std::atomic<int32_t> gPersistentBlockCount{0};
#endif

size_t alignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

struct WorkspaceLayout
{
    size_t intermediate{};
    size_t problemShapes{};
    size_t strides{};
    size_t addresses{};
    size_t logicalKSizes{};
    size_t groupScales{};
    size_t tensormapScratch{};
    size_t total{};
};

WorkspaceLayout buildWorkspaceLayout(int32_t totalRows, int32_t rankStride)
{
    WorkspaceLayout layout{};
    size_t cursor{};
    auto place = [&cursor](size_t& offset, size_t bytes) {
        offset = cursor;
        cursor = alignUp(cursor + bytes, kWORKSPACE_ALIGNMENT);
    };
    place(layout.intermediate, static_cast<size_t>(totalRows) * rankStride * sizeof(half));
    place(layout.problemShapes, static_cast<size_t>(kMAX_CUTE_DSL_GROUPS) * 4 * sizeof(int32_t));
    place(layout.strides, static_cast<size_t>(kMAX_CUTE_DSL_GROUPS) * 6 * sizeof(int32_t));
    place(layout.addresses, static_cast<size_t>(kMAX_CUTE_DSL_GROUPS) * 3 * sizeof(int64_t));
    place(layout.logicalKSizes, static_cast<size_t>(kMAX_CUTE_DSL_GROUPS) * sizeof(int32_t));
    place(layout.groupScales, static_cast<size_t>(kMAX_CUTE_DSL_GROUPS) * sizeof(float));
    place(layout.tensormapScratch,
        static_cast<size_t>(kMAX_CUTE_DSL_GROUPS) * kTENSORMAPS_PER_BLOCK * kBYTES_PER_TENSORMAP);
    layout.total = cursor;
    return layout;
}

void* offsetPointer(half* base, size_t offset)
{
    return reinterpret_cast<std::byte*>(base) + offset;
}

bool hasValidContract(GroupedLoraGemmParams const& params, cudaStream_t stream) noexcept
{
    if (stream == nullptr || params.input == nullptr || params.loraA == nullptr || params.loraB == nullptr
        || params.segmentOffsets == nullptr || params.segmentAdapterIds == nullptr || params.adapterRanks == nullptr
        || params.adapterScales == nullptr || params.workspace == nullptr || params.output == nullptr)
    {
        return false;
    }
    if (params.totalRows <= 0 || params.inputSize <= 0 || params.outputSize <= 0 || params.maxRank <= 0
        || params.maxRank > std::numeric_limits<int32_t>::max() - kRANK_ALIGNMENT + 1 || params.numSegments <= 0
        || params.numSegments > params.totalRows || params.numAdapters <= 0)
    {
        return false;
    }
    constexpr int32_t kMAX_GRID_Y{65535};
    int64_t const outputTiles = (static_cast<int64_t>(params.outputSize) + 31) / 32;
    return outputTiles <= kMAX_GRID_Y;
}

bool canUseCuteDsl(GroupedLoraGemmParams const& params) noexcept
{
#if defined(CUTE_DSL_LORA_BLACKWELL_ENABLED)
    return gPersistentBlockCount.load(std::memory_order_acquire) > 0 && params.numSegments <= kMAX_CUTE_DSL_GROUPS
        && params.inputSize % 8 == 0 && params.outputSize % 8 == 0;
#else
    static_cast<void>(params);
    return false;
#endif
}

cudaError_t launchCuteDsl(GroupedLoraGemmParams const& params, WorkspaceLayout const& layout, cudaStream_t stream)
{
#if defined(CUTE_DSL_LORA_BLACKWELL_ENABLED)
    int32_t const rankStride = getRankStride(params.maxRank);
    auto* const problemShapes = static_cast<int32_t*>(offsetPointer(params.workspace, layout.problemShapes));
    auto* const strides = static_cast<int32_t*>(offsetPointer(params.workspace, layout.strides));
    auto* const addresses = static_cast<int64_t*>(offsetPointer(params.workspace, layout.addresses));
    auto* const logicalKSizes = static_cast<int32_t*>(offsetPointer(params.workspace, layout.logicalKSizes));
    auto* const groupScales = static_cast<float*>(offsetPointer(params.workspace, layout.groupScales));
    void* const tensormapScratch = offsetPointer(params.workspace, layout.tensormapScratch);
    int32_t result = cute_dsl_lora_blackwell_metadata_wrapper(&gMetadataModule.module, const_cast<half*>(params.input),
        const_cast<half*>(params.loraA), const_cast<half*>(params.loraB), params.workspace, params.output,
        const_cast<int32_t*>(params.segmentOffsets), const_cast<int32_t*>(params.segmentAdapterIds),
        const_cast<int32_t*>(params.adapterRanks), const_cast<float*>(params.adapterScales), problemShapes, strides,
        addresses, logicalKSizes, groupScales, params.totalRows, params.inputSize, params.outputSize, params.maxRank,
        params.numSegments, params.numAdapters, 0, params.numSegments, stream);
    if (result != 0)
    {
        return cudaErrorUnknown;
    }
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }
    result = cute_dsl_lora_blackwell_grouped_fp16_wrapper(&gLoraModule.module, const_cast<half*>(params.input),
        const_cast<half*>(params.loraA), params.workspace, problemShapes, strides, addresses, logicalKSizes,
        groupScales, tensormapScratch, params.totalRows, rankStride, params.inputSize, params.numSegments,
        gPersistentBlockCount.load(std::memory_order_acquire), stream);
    if (result != 0)
    {
        return cudaErrorUnknown;
    }
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }

    result = cute_dsl_lora_blackwell_metadata_wrapper(&gMetadataModule.module, const_cast<half*>(params.input),
        const_cast<half*>(params.loraA), const_cast<half*>(params.loraB), params.workspace, params.output,
        const_cast<int32_t*>(params.segmentOffsets), const_cast<int32_t*>(params.segmentAdapterIds),
        const_cast<int32_t*>(params.adapterRanks), const_cast<float*>(params.adapterScales), problemShapes, strides,
        addresses, logicalKSizes, groupScales, params.totalRows, params.inputSize, params.outputSize, params.maxRank,
        params.numSegments, params.numAdapters, 1, params.numSegments, stream);
    if (result != 0)
    {
        return cudaErrorUnknown;
    }
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }
    result = cute_dsl_lora_blackwell_grouped_fp16_fused_wrapper(&gFusedLoraModule.module, params.workspace,
        const_cast<half*>(params.loraB), params.output, problemShapes, strides, addresses, logicalKSizes, groupScales,
        tensormapScratch, params.totalRows, params.outputSize, rankStride, params.numSegments,
        gPersistentBlockCount.load(std::memory_order_acquire), stream);
    if (result != 0)
    {
        return cudaErrorUnknown;
    }
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }

    return status;
#else
    static_cast<void>(params);
    static_cast<void>(layout);
    static_cast<void>(stream);
    return cudaErrorNotSupported;
#endif
}

} // namespace

int32_t getGroupedLoraGemmRankStride(int32_t maxRank) noexcept
{
    if (maxRank <= 0 || maxRank > std::numeric_limits<int32_t>::max() - kRANK_ALIGNMENT + 1)
    {
        return 0;
    }
    return getRankStride(maxRank);
}

size_t getGroupedLoraGemmWorkspaceSize(int32_t totalRows, int32_t outputSize, int32_t maxRank) noexcept
{
    int32_t const rankStride = getGroupedLoraGemmRankStride(maxRank);
    if (totalRows <= 0 || outputSize <= 0 || rankStride == 0)
    {
        return 0;
    }
    size_t const rows = static_cast<size_t>(totalRows);
    size_t const rankStrideSize = static_cast<size_t>(rankStride);
    constexpr size_t kMETADATA_UPPER_BOUND = static_cast<size_t>(kMAX_CUTE_DSL_GROUPS) * 4 * sizeof(int32_t)
        + static_cast<size_t>(kMAX_CUTE_DSL_GROUPS) * 6 * sizeof(int32_t)
        + static_cast<size_t>(kMAX_CUTE_DSL_GROUPS) * 3 * sizeof(int64_t)
        + static_cast<size_t>(kMAX_CUTE_DSL_GROUPS) * sizeof(int32_t)
        + static_cast<size_t>(kMAX_CUTE_DSL_GROUPS) * sizeof(float)
        + static_cast<size_t>(kMAX_CUTE_DSL_GROUPS) * kTENSORMAPS_PER_BLOCK * kBYTES_PER_TENSORMAP
        + 7 * kWORKSPACE_ALIGNMENT;
    if (kMETADATA_UPPER_BOUND >= std::numeric_limits<size_t>::max()
        || rows > (std::numeric_limits<size_t>::max() - kMETADATA_UPPER_BOUND) / rankStrideSize / sizeof(half))
    {
        return 0;
    }
    return buildWorkspaceLayout(totalRows, rankStride).total;
}

cudaError_t initializeGroupedLoraGemmCuteDsl(cudaStream_t stream) noexcept
{
#if defined(CUTE_DSL_LORA_BLACKWELL_ENABLED)
    if (stream == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    int32_t device{};
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess)
    {
        return status;
    }
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess)
    {
        return status;
    }
    int32_t const smVersion = properties.major * 10 + properties.minor;
#if defined(CUTE_DSL_LORA_ARTIFACT_SM)
    if (smVersion != CUTE_DSL_LORA_ARTIFACT_SM)
    {
        return cudaErrorNotSupported;
    }
#endif
    if (smVersion != 100 && smVersion != 101 && smVersion != 103 && smVersion != 110)
    {
        return cudaErrorNotSupported;
    }
    if (!detail::ensureModuleLoaded<lora_blackwell_grouped_fp16_Kernel_Module_Load,
            lora_blackwell_grouped_fp16_Kernel_Module_Unload>(gLoraModule, "lora_blackwell_grouped_fp16", stream))
    {
        return cudaErrorInitializationError;
    }
    if (!detail::ensureModuleLoaded<lora_blackwell_grouped_fp16_fused_Kernel_Module_Load,
            lora_blackwell_grouped_fp16_fused_Kernel_Module_Unload>(
            gFusedLoraModule, "lora_blackwell_grouped_fp16_fused", stream))
    {
        return cudaErrorInitializationError;
    }
    if (!detail::ensureModuleLoaded<lora_blackwell_metadata_Kernel_Module_Load,
            lora_blackwell_metadata_Kernel_Module_Unload>(gMetadataModule, "lora_blackwell_metadata", stream))
    {
        return cudaErrorInitializationError;
    }
    gPersistentBlockCount.store(
        std::min(kMAX_CUTE_DSL_GROUPS, properties.multiProcessorCount), std::memory_order_release);
    return cudaSuccess;
#else
    static_cast<void>(stream);
    return cudaErrorNotSupported;
#endif
}

cudaError_t launchGroupedLoraGemm(GroupedLoraGemmParams const& params, cudaStream_t stream) noexcept
{
    if (!hasValidContract(params, stream))
    {
        return cudaErrorInvalidValue;
    }

    if (!canUseCuteDsl(params))
    {
        return cudaErrorNotSupported;
    }
    WorkspaceLayout const layout = buildWorkspaceLayout(params.totalRows, getRankStride(params.maxRank));
    return launchCuteDsl(params, layout, stream);
}

} // namespace kernel
} // namespace trt_edgellm
