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

#ifdef CUTE_DSL_W2A16_MOE_ENABLED

#include "cuteDslW2A16MoeRunner.h"

#include "common/logger.h"
#include "kernels/cuteDslModuleLoader.h"
#include "kernels/moe/moeAlignSumKernels.h"
#include "kernels/moe/moeMarlinIndicesKernels.h"
#include "w2a16SmallMMoe.h"

#include <algorithm>
#include <exception>
#include <initializer_list>

namespace trt_edgellm
{
#ifdef CUTE_DSL_W2A16_MOE_THOR_ENABLED
detail::LazyKernelModule<w2a16_moe_umma_n48_Kernel_Module_t> CuteDslW2A16MoeRunner::sUmmaN48{};
detail::LazyKernelModule<w2a16_moe_umma_n96_Kernel_Module_t> CuteDslW2A16MoeRunner::sUmmaN96{};
detail::LazyKernelModule<w2a16_moe_umma_n192_Kernel_Module_t> CuteDslW2A16MoeRunner::sUmmaN192{};
#endif
#ifdef CUTE_DSL_W2A16_MOE_FUSED_ENABLED
detail::LazyKernelModule<w2a16_moe_decode_Kernel_Module_t> CuteDslW2A16MoeRunner::sDecode{};
detail::LazyKernelModule<w2a16_moe_prefill_m8_Kernel_Module_t> CuteDslW2A16MoeRunner::sPrefillM8{};
detail::LazyKernelModule<w2a16_moe_prefill_m16_Kernel_Module_t> CuteDslW2A16MoeRunner::sPrefillM16{};
detail::LazyKernelModule<w2a16_moe_prefill_m32_Kernel_Module_t> CuteDslW2A16MoeRunner::sPrefillM32{};
detail::LazyKernelModule<w2a16_moe_prefill_m48_Kernel_Module_t> CuteDslW2A16MoeRunner::sPrefillM48{};
detail::LazyKernelModule<w2a16_moe_prefill_m64_Kernel_Module_t> CuteDslW2A16MoeRunner::sPrefillM64{};
#endif

namespace
{
constexpr size_t kDeviceAlignment = 256;
constexpr size_t kTensorAlignment = 16;
constexpr size_t kRoutingHeaderBytes = 2304 * sizeof(int32_t);

int32_t normalizeSmVersion(int32_t smVersion)
{
    // CUDA 13 renamed Thor's SM101 to SM110.
    return smVersion == 101 ? 110 : smVersion;
}

constexpr int32_t kSM_THOR = 110;

bool isArtifactSmVersion(int32_t smVersion)
{
    for (int32_t const sm : {CUTE_DSL_W2A16_MOE_ARTIFACT_SMS})
    {
        if (normalizeSmVersion(sm) == smVersion)
        {
            return true;
        }
    }
    return false;
}

//! Thor runs UMMA; every other supported SM runs the fused AOT kernels. An artifact that carries only one of the
//! two paths has no choice.
bool useUmma(int32_t smVersion)
{
#if defined(CUTE_DSL_W2A16_MOE_THOR_ENABLED) && defined(CUTE_DSL_W2A16_MOE_FUSED_ENABLED)
    return normalizeSmVersion(smVersion) == kSM_THOR;
#elif defined(CUTE_DSL_W2A16_MOE_THOR_ENABLED)
    static_cast<void>(smVersion);
    return true;
#else
    static_cast<void>(smVersion);
    return false;
#endif
}

#ifdef CUTE_DSL_W2A16_MOE_FUSED_ENABLED
constexpr int32_t kSM_GB10 = 121;
//! GB10 keeps the fused kernels at 1 token and above 8 tokens, where they measured faster than the small-M kernels.
constexpr int32_t kGB10_SMALL_M_MIN_TOKENS = 2;
constexpr int32_t kGB10_SMALL_M_MAX_TOKENS = 8;

int32_t smallMMinTokens(int32_t smVersion)
{
    return smVersion == kSM_GB10 ? kGB10_SMALL_M_MIN_TOKENS : 1;
}

int32_t smallMMaxTokens(int32_t smVersion)
{
    return smVersion == kSM_GB10 ? kGB10_SMALL_M_MAX_TOKENS : kernel::kW2A16_SMALL_M_MOE_MAX_TOKENS;
}

//! Token counts that run on the small-M kernels instead of the fused ones.
bool useSmallM(int32_t smVersion, int32_t numTokens)
{
    return numTokens >= smallMMinTokens(smVersion) && numTokens <= smallMMaxTokens(smVersion);
}
#endif

size_t locksBytes(int32_t numSms)
{
    return (4 * static_cast<size_t>(numSms) + 2) * sizeof(int32_t);
}

size_t alignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

bool isShapeSupported(int32_t numTokens, int32_t numExperts, int32_t moeInterSize)
{
    if (numTokens < 1 || numTokens > CuteDslW2A16MoeRunner::kMaxNumTokens)
    {
        return false;
    }
    bool supported = false;
    switch (numExperts)
    {
    case 240: supported = moeInterSize == 1536 || moeInterSize == 2048 || moeInterSize == 2304; break;
    case 304:
    case 368: supported = moeInterSize == 2048 || moeInterSize == 2304 || moeInterSize == 2560; break;
    case 432: supported = moeInterSize == 2304 || moeInterSize == 2560; break;
    case 496: supported = moeInterSize == 2560; break;
    case 256:
        supported = moeInterSize == 1536 || moeInterSize == 2048 || moeInterSize == 2304 || moeInterSize == 2560;
        break;
    case 320: supported = moeInterSize == 2560 || moeInterSize == 2816; break;
    case 384:
    case 448: supported = moeInterSize == 2304 || moeInterSize == 2560 || moeInterSize == 2816; break;
    case 512: supported = moeInterSize == 2560 || moeInterSize == 2816; break;
    default: break;
    }
    return supported;
}

int32_t selectRowTile(int32_t numTokens, int32_t numExperts)
{
    // Match b12x's 90% target fill using the average routed rows per expert.
    for (int32_t const tile : {8, 16, 32, 48, 64})
    {
        if (int64_t{numTokens} * CuteDslW2A16MoeRunner::kTopK * 10 < int64_t{numExperts} * tile * 9)
        {
            return tile;
        }
    }
    return 64;
}

#ifdef CUTE_DSL_W2A16_MOE_THOR_ENABLED
// Average routed rows per active expert, chosen on the host because the runner never reads device IDs.
// Each doubling of the token tile halves how often a weight tile is decoded but halves TMEM residency.
// On Thor N96 wins once experts average more than 48 rows and N192 once they average more than 96.
constexpr int64_t kUmmaN48MaxRows = 48;
constexpr int64_t kUmmaN96MaxRows = 96;

int32_t selectUmmaTokens(int32_t numTokens, int32_t numExperts)
{
    int64_t const rows = int64_t{numTokens} * CuteDslW2A16MoeRunner::kTopK;
    int64_t const active = std::min<int64_t>(rows, numExperts);
    int64_t const rowsPerExpert = (rows + active - 1) / active;
    if (rowsPerExpert > kUmmaN96MaxRows)
    {
        return 192;
    }
    return rowsPerExpert > kUmmaN48MaxRows ? 96 : 48;
}
#endif

//! Byte offsets shared by workspace sizing and launch binding.
struct WorkspaceLayout
{
    size_t activated{};
    size_t slots{};
    size_t scratch1{};
    size_t scratch2{};
    size_t locks{};
    size_t counts{};
    size_t paddedCounts{};
    size_t offsets{};
    size_t slotLists{};
    size_t routes{};
    size_t blockExperts{};
    size_t routeCount{};
    size_t unusedSortedWeights{};
    size_t total{};
};

WorkspaceLayout buildWorkspaceLayout(
    int32_t numTokens, int32_t numExperts, int32_t moeInterSize, int32_t numSms, bool umma)
{
    WorkspaceLayout layout{};
    auto place = [&layout](size_t& offset, size_t bytes) {
        offset = layout.total;
        layout.total = alignUp(layout.total + bytes, kDeviceAlignment);
    };
    size_t const rows = static_cast<size_t>(numTokens) * CuteDslW2A16MoeRunner::kTopK;
    if (umma)
    {
        layout.total = kRoutingHeaderBytes;
        place(layout.slotLists, static_cast<size_t>(numExperts) * rows * sizeof(int32_t));
        place(layout.activated, rows * moeInterSize * sizeof(uint16_t));
        place(layout.slots, rows * CuteDslW2A16MoeRunner::kHiddenSize * sizeof(uint16_t));
        return layout;
    }
    int32_t const rowTile = selectRowTile(numTokens, numExperts);
    size_t const padded = alignUp(rows + static_cast<size_t>(numExperts) * (rowTile - 1), kDeviceAlignment);
    place(layout.activated, rows * moeInterSize * sizeof(uint16_t));
    place(layout.slots, rows * CuteDslW2A16MoeRunner::kHiddenSize * sizeof(uint16_t));
    // Split-K bounds four CTAs per SM, each with at most a 64x256 FP32 tile.
    size_t const scratchBytes = 4 * static_cast<size_t>(numSms) * 64 * 256 * sizeof(float);
    place(layout.scratch1, scratchBytes);
    place(layout.scratch2, scratchBytes);
    place(layout.locks, locksBytes(numSms));
    if (numTokens > CuteDslW2A16MoeRunner::kDecodeMaxTokens)
    {
        place(layout.counts, numExperts * sizeof(int32_t));
        place(layout.paddedCounts, numExperts * sizeof(int32_t));
        place(layout.offsets, numExperts * sizeof(int32_t));
        place(layout.slotLists, numExperts * rows * sizeof(int32_t));
        place(layout.routes, padded * sizeof(int32_t));
        place(layout.blockExperts, padded / rowTile * sizeof(int32_t));
        place(layout.routeCount, sizeof(int32_t));
        place(layout.unusedSortedWeights, padded * sizeof(float));
    }
    return layout;
}

#ifdef CUTE_DSL_W2A16_MOE_FUSED_ENABLED
template <auto Launch, typename Module>
int32_t launchMoe(Module& module, CuteDslW2A16MoeParams const& params, void* workspace, WorkspaceLayout const& layout,
    int32_t* routes, int32_t* experts, int32_t* count, cudaStream_t stream)
{
    auto* base = static_cast<std::byte*>(workspace);
    return Launch(&module, const_cast<void*>(params.hiddenStates), const_cast<void*>(params.fc1QWeights),
        const_cast<void*>(params.fc2QWeights), const_cast<void*>(params.fc1BlocksScale),
        const_cast<void*>(params.fc2BlocksScale), const_cast<float*>(params.unitScales),
        const_cast<void*>(params.lookupTable), routes, experts, count, const_cast<float*>(params.topkWeights),
        base + layout.activated, base + layout.slots, params.output, base + layout.scratch1, base + layout.scratch2,
        base + layout.locks, params.numTokens, params.numExperts, params.moeInterSize, stream);
}
#endif

template <typename T>
T* workspacePointer(void* workspace, size_t offset)
{
    return reinterpret_cast<T*>(static_cast<std::byte*>(workspace) + offset);
}
} // namespace

bool CuteDslW2A16MoeRunner::isHardwareSupported(int32_t smVersion, int32_t numSms) noexcept
{
    return isArtifactSmVersion(normalizeSmVersion(smVersion)) && numSms > 0
        && (useUmma(smVersion) || numSms >= kFusedNumSms);
}

bool CuteDslW2A16MoeRunner::canImplement(
    int32_t numTokens, int32_t numExperts, int32_t moeInterSize, int32_t smVersion, int32_t numSms) noexcept
{
    return isHardwareSupported(smVersion, numSms) && isShapeSupported(numTokens, numExperts, moeInterSize);
}

size_t CuteDslW2A16MoeRunner::getWorkspaceSize(
    int32_t numTokens, int32_t numExperts, int32_t moeInterSize, int32_t smVersion, int32_t numSms) noexcept
{
    if (!canImplement(numTokens, numExperts, moeInterSize, smVersion, numSms))
    {
        return 0;
    }
    bool const umma = useUmma(smVersion);
#ifdef CUTE_DSL_W2A16_MOE_FUSED_ENABLED
    // A profile-maximum reservation also covers the small-M kernels of every smaller token count.
    int32_t const maxSmall = smallMMaxTokens(smVersion);
    int32_t const smallTokens = numTokens < maxSmall ? numTokens : maxSmall;
    if (!umma && smallTokens >= smallMMinTokens(smVersion))
    {
        size_t const layout = buildWorkspaceLayout(numTokens, numExperts, moeInterSize, numSms, umma).total;
        size_t const small = kernel::w2a16SmallMMoeWorkspaceSize(smallTokens, moeInterSize);
        return layout > small ? layout : small;
    }
#endif
    return buildWorkspaceLayout(numTokens, numExperts, moeInterSize, numSms, umma).total;
}

bool CuteDslW2A16MoeRunner::ensureKernelModules(
    CuteDslW2A16MoeParams const& params, int32_t smVersion, int32_t numSms, cudaStream_t stream) noexcept
{
    if (!isShapeSupported(params.numTokens, params.numExperts, params.moeInterSize))
    {
        return false;
    }
    if (!canImplement(params.numTokens, params.numExperts, params.moeInterSize, smVersion, numSms))
    {
        return false;
    }
#ifdef CUTE_DSL_W2A16_MOE_THOR_ENABLED
    if (useUmma(smVersion))
    {
        switch (selectUmmaTokens(params.numTokens, params.numExperts))
        {
        case 192:
            return detail::ensureModuleLoaded<w2a16_moe_umma_n192_Kernel_Module_Load,
                w2a16_moe_umma_n192_Kernel_Module_Unload>(sUmmaN192, "w2a16_moe_umma_n192", stream);
        case 96:
            return detail::ensureModuleLoaded<w2a16_moe_umma_n96_Kernel_Module_Load,
                w2a16_moe_umma_n96_Kernel_Module_Unload>(sUmmaN96, "w2a16_moe_umma_n96", stream);
        default: break;
        }
        return detail::ensureModuleLoaded<w2a16_moe_umma_n48_Kernel_Module_Load,
            w2a16_moe_umma_n48_Kernel_Module_Unload>(sUmmaN48, "w2a16_moe_umma_n48", stream);
    }
#endif
#ifdef CUTE_DSL_W2A16_MOE_FUSED_ENABLED
    bool loaded = false;
    if (params.numTokens <= kDecodeMaxTokens)
    {
        loaded = detail::ensureModuleLoaded<w2a16_moe_decode_Kernel_Module_Load, w2a16_moe_decode_Kernel_Module_Unload>(
            sDecode, "w2a16_moe_decode", stream);
    }
    else
    {
        switch (selectRowTile(params.numTokens, params.numExperts))
        {
        case 8:
            loaded = detail::ensureModuleLoaded<w2a16_moe_prefill_m8_Kernel_Module_Load,
                w2a16_moe_prefill_m8_Kernel_Module_Unload>(sPrefillM8, "w2a16_moe_prefill_m8", stream);
            break;
        case 16:
            loaded = detail::ensureModuleLoaded<w2a16_moe_prefill_m16_Kernel_Module_Load,
                w2a16_moe_prefill_m16_Kernel_Module_Unload>(sPrefillM16, "w2a16_moe_prefill_m16", stream);
            break;
        case 32:
            loaded = detail::ensureModuleLoaded<w2a16_moe_prefill_m32_Kernel_Module_Load,
                w2a16_moe_prefill_m32_Kernel_Module_Unload>(sPrefillM32, "w2a16_moe_prefill_m32", stream);
            break;
        case 48:
            loaded = detail::ensureModuleLoaded<w2a16_moe_prefill_m48_Kernel_Module_Load,
                w2a16_moe_prefill_m48_Kernel_Module_Unload>(sPrefillM48, "w2a16_moe_prefill_m48", stream);
            break;
        case 64:
            loaded = detail::ensureModuleLoaded<w2a16_moe_prefill_m64_Kernel_Module_Load,
                w2a16_moe_prefill_m64_Kernel_Module_Unload>(sPrefillM64, "w2a16_moe_prefill_m64", stream);
            break;
        default: break;
        }
    }

    return loaded;
#else
    return false;
#endif
}

#ifdef CUTE_DSL_W2A16_MOE_THOR_ENABLED
int32_t CuteDslW2A16MoeRunner::runUmma(CuteDslW2A16MoeParams const& params, void* workspace, cudaStream_t stream)
{
    auto const layout = buildWorkspaceLayout(params.numTokens, params.numExperts, params.moeInterSize, 0, true);
    auto* counts = static_cast<int32_t*>(workspace);
    auto* lists = workspacePointer<int32_t>(workspace, layout.slotLists);
    cudaError_t const status = cudaMemsetAsync(counts, 0, (params.numExperts + 1) * sizeof(int32_t), stream);
    if (status != cudaSuccess)
    {
        return static_cast<int32_t>(status);
    }
    auto const launch = [&](auto function, auto& module) {
        auto* base = static_cast<std::byte*>(workspace);
        return function(&module, const_cast<void*>(params.hiddenStates), const_cast<void*>(params.fc1QWeights),
            const_cast<void*>(params.fc2QWeights), const_cast<void*>(params.fc1BlocksScale),
            const_cast<void*>(params.fc2BlocksScale), const_cast<void*>(params.lookupTable), lists, counts,
            const_cast<float*>(params.topkWeights), base + layout.activated, base + layout.slots, params.output,
            const_cast<int32_t*>(params.topkIds), params.numTokens, params.numExperts, params.moeInterSize, stream);
    };
    switch (selectUmmaTokens(params.numTokens, params.numExperts))
    {
    case 192: return launch(cute_dsl_w2a16_moe_umma_n192_wrapper, sUmmaN192.module);
    case 96: return launch(cute_dsl_w2a16_moe_umma_n96_wrapper, sUmmaN96.module);
    default: break;
    }
    return launch(cute_dsl_w2a16_moe_umma_n48_wrapper, sUmmaN48.module);
}
#endif

#ifdef CUTE_DSL_W2A16_MOE_FUSED_ENABLED
int32_t CuteDslW2A16MoeRunner::runDecode(
    CuteDslW2A16MoeParams const& params, void* workspace, int32_t numSms, cudaStream_t stream)
{
    auto const layout = buildWorkspaceLayout(params.numTokens, params.numExperts, params.moeInterSize, numSms, false);
    auto* base = static_cast<std::byte*>(workspace);
    auto* routes = const_cast<int32_t*>(params.topkIds);
    // Direct routing does not read the packed expert/count arguments.
    auto* experts = workspacePointer<int32_t>(workspace, layout.locks);
    auto* count = experts;
    cudaError_t const status = cudaMemsetAsync(base + layout.locks, 0, locksBytes(numSms), stream);
    if (status != cudaSuccess)
    {
        return static_cast<int32_t>(status);
    }
    return launchMoe<cute_dsl_w2a16_moe_decode_wrapper>(
        sDecode.module, params, workspace, layout, routes, experts, count, stream);
}

int32_t CuteDslW2A16MoeRunner::runPrefill(
    CuteDslW2A16MoeParams const& params, void* workspace, int32_t numSms, cudaStream_t stream)
{
    int32_t const rowTile = selectRowTile(params.numTokens, params.numExperts);
    auto const layout = buildWorkspaceLayout(params.numTokens, params.numExperts, params.moeInterSize, numSms, false);
    auto* base = static_cast<std::byte*>(workspace);
    auto* routes = workspacePointer<int32_t>(workspace, layout.routes);
    auto* experts = workspacePointer<int32_t>(workspace, layout.blockExperts);
    auto* count = workspacePointer<int32_t>(workspace, layout.routeCount);
    auto* counts = workspacePointer<int32_t>(workspace, layout.counts);
    auto* paddedCounts = workspacePointer<int32_t>(workspace, layout.paddedCounts);
    auto* offsets = workspacePointer<int32_t>(workspace, layout.offsets);
    auto* slotLists = workspacePointer<int32_t>(workspace, layout.slotLists);
    auto* sortedWeights = workspacePointer<float>(workspace, layout.unusedSortedWeights);
    cudaError_t status = cudaMemsetAsync(base + layout.locks, 0, locksBytes(numSms), stream);
    if (status != cudaSuccess)
    {
        return static_cast<int32_t>(status);
    }
    kernel::launchCountSlotsPerExpertKernel(params.topkIds, counts, params.numTokens, kTopK, params.numExperts, stream);
    kernel::launchComputePaddedOffsetsKernel(counts, paddedCounts, offsets, count, params.numExperts, rowTile, stream);
    status = cudaMemsetAsync(counts, 0, params.numExperts * sizeof(int32_t), stream);
    if (status != cudaSuccess)
    {
        return static_cast<int32_t>(status);
    }
    kernel::launchBuildSlotListsKernel(
        params.topkIds, slotLists, counts, params.numTokens, kTopK, params.numExperts, stream);
    kernel::launchBuildMarlinIndicesKernel(slotLists, counts, paddedCounts, offsets, params.topkWeights, routes,
        sortedWeights, experts, params.numTokens, kTopK, params.numExperts, rowTile, stream);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return static_cast<int32_t>(status);
    }
    int32_t result = static_cast<int32_t>(cudaErrorInvalidValue);
    switch (rowTile)
    {
    case 8:
        result = launchMoe<cute_dsl_w2a16_moe_prefill_m8_wrapper>(
            sPrefillM8.module, params, workspace, layout, routes, experts, count, stream);
        break;
    case 16:
        result = launchMoe<cute_dsl_w2a16_moe_prefill_m16_wrapper>(
            sPrefillM16.module, params, workspace, layout, routes, experts, count, stream);
        break;
    case 32:
        result = launchMoe<cute_dsl_w2a16_moe_prefill_m32_wrapper>(
            sPrefillM32.module, params, workspace, layout, routes, experts, count, stream);
        break;
    case 48:
        result = launchMoe<cute_dsl_w2a16_moe_prefill_m48_wrapper>(
            sPrefillM48.module, params, workspace, layout, routes, experts, count, stream);
        break;
    case 64:
        result = launchMoe<cute_dsl_w2a16_moe_prefill_m64_wrapper>(
            sPrefillM64.module, params, workspace, layout, routes, experts, count, stream);
        break;
    default: break;
    }
    if (result != 0)
    {
        return result;
    }
    kernel::launchAggregateSlotOutputsBf16Kernel(
        base + layout.slots, params.output, params.numTokens, kTopK, kHiddenSize, stream);
    return static_cast<int32_t>(cudaGetLastError());
}

#endif

int32_t CuteDslW2A16MoeRunner::run(CuteDslW2A16MoeParams const& params, void* workspace, size_t workspaceBytes,
    int32_t smVersion, int32_t numSms, cudaStream_t stream) noexcept
{
    if (!isShapeSupported(params.numTokens, params.numExperts, params.moeInterSize))
    {
        return static_cast<int32_t>(cudaErrorInvalidValue);
    }
    try
    {
        if (!canImplement(params.numTokens, params.numExperts, params.moeInterSize, smVersion, numSms))
        {
            return static_cast<int32_t>(cudaErrorNotSupported);
        }
        bool const umma = useUmma(smVersion);
        size_t required
            = buildWorkspaceLayout(params.numTokens, params.numExperts, params.moeInterSize, numSms, umma).total;
#ifdef CUTE_DSL_W2A16_MOE_FUSED_ENABLED
        bool const smallM = !umma && useSmallM(smVersion, params.numTokens);
        if (smallM)
        {
            required = kernel::w2a16SmallMMoeWorkspaceSize(params.numTokens, params.moeInterSize);
        }
#endif
        if (required == 0 || workspace == nullptr || workspaceBytes < required
            || reinterpret_cast<uintptr_t>(workspace) % kDeviceAlignment != 0)
        {
            return static_cast<int32_t>(cudaErrorInvalidValue);
        }
        for (void const* ptr : {params.hiddenStates, params.fc1QWeights, params.fc2QWeights, params.fc1BlocksScale,
                 params.fc2BlocksScale, static_cast<void const*>(params.unitScales), params.lookupTable,
                 static_cast<void const*>(params.topkIds), static_cast<void const*>(params.topkWeights),
                 static_cast<void const*>(params.output)})
        {
            if (ptr == nullptr || reinterpret_cast<uintptr_t>(ptr) % kTensorAlignment != 0)
            {
                return static_cast<int32_t>(cudaErrorInvalidValue);
            }
        }
#ifdef CUTE_DSL_W2A16_MOE_FUSED_ENABLED
        if (smallM)
        {
            kernel::W2A16SmallMMoeParams const small{params.numTokens, params.numExperts, params.moeInterSize,
                params.hiddenStates, params.fc1QWeights, params.fc2QWeights, params.fc1BlocksScale,
                params.fc2BlocksScale, params.unitScales, params.lookupTable, params.topkIds, params.topkWeights,
                params.output};
            return static_cast<int32_t>(kernel::launchW2A16SmallMMoe(small, workspace, stream));
        }
#endif
        if (!ensureKernelModules(params, smVersion, numSms, stream))
        {
            return static_cast<int32_t>(cudaErrorInitializationError);
        }
#ifdef CUTE_DSL_W2A16_MOE_THOR_ENABLED
        if (umma)
        {
            return runUmma(params, workspace, stream);
        }
#endif
#ifdef CUTE_DSL_W2A16_MOE_FUSED_ENABLED
        if (params.numTokens <= kDecodeMaxTokens)
        {
            return runDecode(params, workspace, numSms, stream);
        }
        return runPrefill(params, workspace, numSms, stream);
#else
        return static_cast<int32_t>(cudaErrorNotSupported);
#endif
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("CuteDslW2A16MoeRunner: %s", error.what());
    }
    catch (...)
    {
        LOG_ERROR("CuteDslW2A16MoeRunner: unknown launch failure");
    }
    return static_cast<int32_t>(cudaErrorUnknown);
}

} // namespace trt_edgellm
#endif // CUTE_DSL_W2A16_MOE_ENABLED
