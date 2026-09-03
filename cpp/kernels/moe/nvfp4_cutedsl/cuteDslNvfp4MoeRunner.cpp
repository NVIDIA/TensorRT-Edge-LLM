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

#ifdef CUTE_DSL_NVFP4_FUSED_MOE_ENABLED

#include "cuteDslNvfp4MoeRunner.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace trt_edgellm
{

// AOT module handles -- 2 backends x 6 activations x n128 = 12.
detail::LazyKernelModule<nvfp4_fused_moe_decode_identity_m128_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sDecodeIdentity_m128_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_decode_silu_m128_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sDecodeSiLU_m128_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_decode_swiglu_m128_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sDecodeSwiGLU_m128_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_decode_gelu_m128_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sDecodeGeLU_m128_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_decode_relu2_m128_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sDecodeReLU2_m128_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_decode_geglu_m128_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sDecodeGeGLU_m128_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_decode_identity_m64_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sDecodeIdentity_m64_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_decode_silu_m64_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sDecodeSiLU_m64_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_decode_swiglu_m64_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sDecodeSwiGLU_m64_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_decode_gelu_m64_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sDecodeGeLU_m64_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_decode_relu2_m64_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sDecodeReLU2_m64_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_decode_geglu_m64_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sDecodeGeGLU_m64_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_prefill_identity_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sPrefillIdentity_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_prefill_silu_n128_Kernel_Module_t> CuteDslNvfp4MoeRunner::sPrefillSiLU_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_prefill_swiglu_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sPrefillSwiGLU_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_prefill_gelu_n128_Kernel_Module_t> CuteDslNvfp4MoeRunner::sPrefillGeLU_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_prefill_relu2_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sPrefillReLU2_n128{};
detail::LazyKernelModule<nvfp4_fused_moe_prefill_geglu_n128_Kernel_Module_t>
    CuteDslNvfp4MoeRunner::sPrefillGeGLU_n128{};

bool CuteDslNvfp4MoeRunner::canImplement(int32_t hiddenSize, int32_t moeInterSize, int32_t numExperts, int32_t topK,
    int32_t smVersion, CuteDslMoeActivation activation, CuteDslMoeIoDtype ioDtype, CuteDslMoeBackend backend)
{
    // SM120/SM121 only.
    if (smVersion < 120 || smVersion > 121)
    {
        return false;
    }
    // FP16 io_dtype is the only supported dtype. All six CuteDslMoeActivation values
    // (identity / silu / swiglu / gelu / relu2 / geglu) are wired -- each backed by its own AOT module.
    if (ioDtype != CuteDslMoeIoDtype::kFP16)
    {
        return false;
    }
    switch (activation)
    {
    case CuteDslMoeActivation::kIdentity:
    case CuteDslMoeActivation::kSiLU:
    case CuteDslMoeActivation::kSwiGLU:
    case CuteDslMoeActivation::kGeLU:
    case CuteDslMoeActivation::kReLU2:
    case CuteDslMoeActivation::kGeGLU: break;
    default: return false;
    }
    // Both backends are supported; kAuto picks between them based on routed-row count.
    (void) backend;
    // Shape-polymorphism contract (enforced by the shape-polymorphic AOT
    // wrappers): I must be positive and a multiple of the smallest MMA
    // N-tile (kLevelTileN = 128); topK / numExperts must be non-degenerate.
    // hiddenSize (K) is runtime; the only constraint is divisibility by
    // kHiddenSizeAlignment (= kCuteDslTileK * kStaticAbStage), which keeps the
    // K-tile pipeline phase-aligned. This replaces the Python-level ab_stage
    // divisor loop in moe_{decode,prefill}_kernel.py, which could not trace
    // with a symbolic K.
    if (moeInterSize <= 0 || numExperts <= 0 || topK <= 0)
    {
        return false;
    }
    if (topK > numExperts)
    {
        return false;
    }
    if (moeInterSize % kLevelTileN != 0)
    {
        return false;
    }
    if (hiddenSize <= 0 || hiddenSize % kHiddenSizeAlignment != 0)
    {
        return false;
    }
    return true;
}

int32_t CuteDslNvfp4MoeRunner::selectMmaTilerN(int32_t moeInterSize)
{
    // DISABLED: n256 variants hit `ab_stage=1` bug in fz_crSFB indexing — always
    // return n128 until the kernel-side fix lands. Re-enable by restoring the
    // divisibility check below.
    (void) moeInterSize;
    return kLevelTileN;
    // return (moeInterSize % kLevelTileNLarge == 0) ? kLevelTileNLarge : kLevelTileN;
}

int32_t CuteDslNvfp4MoeRunner::selectMmaTilerM(int32_t routedRows)
{
    return routedRows <= kTileMSmallMaxRows ? kLevelTileMSmall : kLevelTileM;
}

bool CuteDslNvfp4MoeRunner::ensureKernelModules(CuteDslNvfp4MoeParams const& params, cudaStream_t stream)
{
    CuteDslMoeBackend const backend = resolveBackend(params.backend, params.numTokens, params.topK);
    if (backend == CuteDslMoeBackend::kDecode)
    {
// One entry per activation; SUFFIX picks the M-tile variant (n128 = m128).
// clang-format off
#define ENSURE_DECODE_MODULES(SUFFIX)                                                                                  \
    switch (params.activation)                                                                                         \
    {                                                                                                                  \
    case CuteDslMoeActivation::kIdentity:                                                                              \
        return detail::ensureModuleLoaded<nvfp4_fused_moe_decode_identity_##SUFFIX##_Kernel_Module_Load,                \
            nvfp4_fused_moe_decode_identity_##SUFFIX##_Kernel_Module_Unload>(                                          \
            sDecodeIdentity_##SUFFIX, "nvfp4_fused_moe_decode_identity_" #SUFFIX, stream);                             \
    case CuteDslMoeActivation::kSiLU:                                                                                  \
        return detail::ensureModuleLoaded<nvfp4_fused_moe_decode_silu_##SUFFIX##_Kernel_Module_Load,                    \
            nvfp4_fused_moe_decode_silu_##SUFFIX##_Kernel_Module_Unload>(                                              \
            sDecodeSiLU_##SUFFIX, "nvfp4_fused_moe_decode_silu_" #SUFFIX, stream);                                     \
    case CuteDslMoeActivation::kSwiGLU:                                                                                \
        return detail::ensureModuleLoaded<nvfp4_fused_moe_decode_swiglu_##SUFFIX##_Kernel_Module_Load,                  \
            nvfp4_fused_moe_decode_swiglu_##SUFFIX##_Kernel_Module_Unload>(                                            \
            sDecodeSwiGLU_##SUFFIX, "nvfp4_fused_moe_decode_swiglu_" #SUFFIX, stream);                                 \
    case CuteDslMoeActivation::kGeLU:                                                                                  \
        return detail::ensureModuleLoaded<nvfp4_fused_moe_decode_gelu_##SUFFIX##_Kernel_Module_Load,                    \
            nvfp4_fused_moe_decode_gelu_##SUFFIX##_Kernel_Module_Unload>(                                              \
            sDecodeGeLU_##SUFFIX, "nvfp4_fused_moe_decode_gelu_" #SUFFIX, stream);                                     \
    case CuteDslMoeActivation::kReLU2:                                                                                 \
        return detail::ensureModuleLoaded<nvfp4_fused_moe_decode_relu2_##SUFFIX##_Kernel_Module_Load,                   \
            nvfp4_fused_moe_decode_relu2_##SUFFIX##_Kernel_Module_Unload>(                                             \
            sDecodeReLU2_##SUFFIX, "nvfp4_fused_moe_decode_relu2_" #SUFFIX, stream);                                   \
    case CuteDslMoeActivation::kGeGLU:                                                                                 \
        return detail::ensureModuleLoaded<nvfp4_fused_moe_decode_geglu_##SUFFIX##_Kernel_Module_Load,                   \
            nvfp4_fused_moe_decode_geglu_##SUFFIX##_Kernel_Module_Unload>(                                             \
            sDecodeGeGLU_##SUFFIX, "nvfp4_fused_moe_decode_geglu_" #SUFFIX, stream);                                   \
    }
        // clang-format on
        if (selectMmaTilerM(params.numTokens * params.topK) == kLevelTileMSmall)
        {
            ENSURE_DECODE_MODULES(m64_n128)
        }
        else
        {
            ENSURE_DECODE_MODULES(m128_n128)
        }
#undef ENSURE_DECODE_MODULES
    }
    if (backend == CuteDslMoeBackend::kPrefill)
    {
        switch (params.activation)
        {
        case CuteDslMoeActivation::kIdentity:
            return detail::ensureModuleLoaded<nvfp4_fused_moe_prefill_identity_n128_Kernel_Module_Load,
                nvfp4_fused_moe_prefill_identity_n128_Kernel_Module_Unload>(
                sPrefillIdentity_n128, "nvfp4_fused_moe_prefill_identity_n128", stream);
        case CuteDslMoeActivation::kSiLU:
            return detail::ensureModuleLoaded<nvfp4_fused_moe_prefill_silu_n128_Kernel_Module_Load,
                nvfp4_fused_moe_prefill_silu_n128_Kernel_Module_Unload>(
                sPrefillSiLU_n128, "nvfp4_fused_moe_prefill_silu_n128", stream);
        case CuteDslMoeActivation::kSwiGLU:
            return detail::ensureModuleLoaded<nvfp4_fused_moe_prefill_swiglu_n128_Kernel_Module_Load,
                nvfp4_fused_moe_prefill_swiglu_n128_Kernel_Module_Unload>(
                sPrefillSwiGLU_n128, "nvfp4_fused_moe_prefill_swiglu_n128", stream);
        case CuteDslMoeActivation::kGeLU:
            return detail::ensureModuleLoaded<nvfp4_fused_moe_prefill_gelu_n128_Kernel_Module_Load,
                nvfp4_fused_moe_prefill_gelu_n128_Kernel_Module_Unload>(
                sPrefillGeLU_n128, "nvfp4_fused_moe_prefill_gelu_n128", stream);
        case CuteDslMoeActivation::kReLU2:
            return detail::ensureModuleLoaded<nvfp4_fused_moe_prefill_relu2_n128_Kernel_Module_Load,
                nvfp4_fused_moe_prefill_relu2_n128_Kernel_Module_Unload>(
                sPrefillReLU2_n128, "nvfp4_fused_moe_prefill_relu2_n128", stream);
        case CuteDslMoeActivation::kGeGLU:
            return detail::ensureModuleLoaded<nvfp4_fused_moe_prefill_geglu_n128_Kernel_Module_Load,
                nvfp4_fused_moe_prefill_geglu_n128_Kernel_Module_Unload>(
                sPrefillGeGLU_n128, "nvfp4_fused_moe_prefill_geglu_n128", stream);
        }
    }
    LOG_ERROR("CuteDslNvfp4MoeRunner: unsupported activation/backend module selection");
    return false;
}

namespace
{
//! Device alignment (256 bytes) matches \c kDEVICE_ALIGNMENT used by other plugin workspaces.
constexpr size_t kDeviceAlignment = 256;

inline size_t alignUp(size_t value, size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

inline int32_t alignUpInt(int32_t value, int32_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

//! Phase 0 initializes every other workspace buffer and output before its first resident-grid barrier.
inline void zeroGridBarrier(
    std::byte* workspace, size_t countOffset, size_t epochOffset, size_t barrierBytes, cudaStream_t stream)
{
    size_t const end = epochOffset + barrierBytes;
    CUDA_CHECK(cudaMemsetAsync(workspace + countOffset, 0, end - countOffset, stream));
}

//! Describes the byte layout of the decode backend's workspace. total is the running
//! sum; each field is a byte offset from the workspace base (all aligned up to
//! kDeviceAlignment so each buffer is individually addressable without further alignment).
struct DecodeWorkspaceLayout
{
    // Byte offsets from the workspace base.
    size_t rowCounts{};
    size_t tokenMap{};
    size_t tokenWeights{};
    size_t packedInput{};
    size_t packedInputScale{};
    size_t barrierCount{};
    size_t barrierEpoch{};
    size_t activeExpertCount{};
    size_t total{};

    // Per-buffer byte sizes (convenient for zeroing).
    size_t rowCountsBytes{};
    size_t tokenMapBytes{};
    size_t tokenWeightsBytes{};
    size_t packedInputBytes{};
    size_t packedInputScaleBytes{};
    size_t barrierBytes{}; //!< each of barrier_count/epoch/active_expert_count

    // Padded shape metadata (needed for the wrapper call).
    int32_t rowsPadded{}; //!< max_rows padded up to kRowTileAlign.
};

// weight_expert_ids / global_to_local_expert used to live in the workspace and be
// re-initialized to arange(E) on every launch. They are now provided by the caller
// via CuteDslNvfp4MoeParams::weightExpertIds / globalToLocalExpertIds; the plugin
// owns a 2*E identity buffer allocated through IGpuAllocator at attachToContext
// time, and the runner reads non-aliasing halves from that buffer. This eliminates
// the two arange kernels on the hot path AND keeps cudaMalloc / cudaMemcpy off
// enqueue.
DecodeWorkspaceLayout buildDecodeLayout(int32_t stateE, int32_t /*weightE*/, int32_t maxRows, int32_t k)
{
    DecodeWorkspaceLayout L{};
    int32_t const rowsPad = alignUpInt(maxRows, CuteDslNvfp4MoeRunner::kRowTileAlign);
    int32_t const colsPad = alignUpInt(k / CuteDslNvfp4MoeRunner::kNvfp4SfVecSize, 4);

    L.rowsPadded = rowsPad;
    L.rowCountsBytes = static_cast<size_t>(stateE) * sizeof(int32_t);
    L.tokenMapBytes = static_cast<size_t>(stateE) * static_cast<size_t>(maxRows) * sizeof(int32_t);
    L.tokenWeightsBytes = static_cast<size_t>(stateE) * static_cast<size_t>(maxRows) * sizeof(float);
    L.packedInputBytes = static_cast<size_t>(stateE) * static_cast<size_t>(maxRows) * static_cast<size_t>(k / 2);
    L.packedInputScaleBytes = static_cast<size_t>(stateE) * static_cast<size_t>(rowsPad) * static_cast<size_t>(colsPad);
    L.barrierBytes = sizeof(int32_t);

    auto place = [](size_t& off, size_t& cursor, size_t bytes) {
        off = cursor;
        cursor = alignUp(cursor + bytes, kDeviceAlignment);
    };

    size_t cur = 0;
    place(L.rowCounts, cur, L.rowCountsBytes);
    place(L.tokenMap, cur, L.tokenMapBytes);
    place(L.tokenWeights, cur, L.tokenWeightsBytes);
    place(L.barrierCount, cur, L.barrierBytes);
    place(L.barrierEpoch, cur, L.barrierBytes);
    place(L.activeExpertCount, cur, L.barrierBytes);
    place(L.packedInput, cur, L.packedInputBytes);
    place(L.packedInputScale, cur, L.packedInputScaleBytes);
    L.total = cur;
    return L;
}

//! Mirrors \c _prefill_task_geometry in \c moe_dispatch.py.
struct PrefillTaskGeometry
{
    int32_t maxMTiles{};
    int32_t gateTileCnt{};
    int32_t maxTasks{};
    int32_t physicalTiles{};
    int32_t rowsPadded{};
};

PrefillTaskGeometry computePrefillGeometry(int32_t stateE, int32_t n, int32_t routedRows)
{
    PrefillTaskGeometry g{};
    int32_t const rows = std::max(1, routedRows);
    int32_t const baseMTiles
        = alignUpInt(rows, CuteDslNvfp4MoeRunner::kRowTileAlign) / CuteDslNvfp4MoeRunner::kRowTileAlign;
    int32_t const activeUpper = std::min(stateE, rows);
    g.maxMTiles = std::max(1, baseMTiles + activeUpper - 1);
    g.gateTileCnt = std::max(1, (n + CuteDslNvfp4MoeRunner::kLevelTileN - 1) / CuteDslNvfp4MoeRunner::kLevelTileN);
    int32_t const sliceGroups = std::max(
        1, (g.gateTileCnt + CuteDslNvfp4MoeRunner::kPrefillSliceChunk - 1) / CuteDslNvfp4MoeRunner::kPrefillSliceChunk);
    g.maxTasks = g.maxMTiles * sliceGroups;
    g.physicalTiles = g.maxMTiles;
    g.rowsPadded = g.physicalTiles * CuteDslNvfp4MoeRunner::kRowTileAlign;
    return g;
}

//! Byte layout of the prefill backend's workspace. Mirrors
//! allocate_sm120_prefill_workspace in moe_dispatch.py.
struct PrefillWorkspaceLayout
{
    // Byte offsets from the workspace base (each buffer aligned up to kDeviceAlignment).
    size_t rowCounts{};
    size_t tokenMap{};
    size_t tokenWeights{};
    size_t packedInput{};
    size_t packedInputScale{};
    size_t barrierCount{};
    size_t barrierEpoch{};
    size_t pairHead{};
    size_t producersDoneCount{};
    size_t allWorkPublished{};
    size_t taskHead{};
    size_t taskTail{};
    size_t expertWriteRows{};
    size_t expertTileBase{};
    size_t taskReady{};
    size_t taskExpert{};
    size_t taskMTile{};
    size_t taskSliceBegin{};
    size_t taskSliceCount{};
    size_t taskValidRows{};
    size_t tileWriteCount{};
    size_t total{};

    // Per-buffer byte sizes (needed for zero-memsets).
    size_t rowCountsBytes{};
    size_t tokenMapBytes{};
    size_t tokenWeightsBytes{};
    size_t packedInputBytes{};
    size_t packedInputScaleBytes{};
    size_t barrierBytes{}; //!< each of 7 int32[1] counters
    size_t expertWriteRowsBytes{};
    size_t expertTileBaseBytes{};
    size_t taskBytes{}; //!< each of 6 int32[max_tasks] arrays
    size_t tileWriteCountBytes{};

    // Padded shape metadata forwarded to the wrapper.
    int32_t rowsPadded{};
    int32_t physicalTiles{};
    int32_t maxTasks{};
};

PrefillWorkspaceLayout buildPrefillLayout(int32_t stateE, int32_t routedRows, int32_t k, int32_t n)
{
    PrefillWorkspaceLayout L{};
    auto const geo = computePrefillGeometry(stateE, n, routedRows);
    int32_t const colsPad = alignUpInt(k / CuteDslNvfp4MoeRunner::kNvfp4SfVecSize, 4);

    L.rowsPadded = geo.rowsPadded;
    L.physicalTiles = geo.physicalTiles;
    L.maxTasks = geo.maxTasks;

    L.rowCountsBytes = static_cast<size_t>(stateE) * sizeof(int32_t);
    L.tokenMapBytes = static_cast<size_t>(geo.rowsPadded) * sizeof(int32_t);
    L.tokenWeightsBytes = static_cast<size_t>(geo.rowsPadded) * sizeof(float);
    // Prefill backend's packed_input is [1, rows_padded, k/2] (not per-expert like decode).
    L.packedInputBytes = static_cast<size_t>(geo.rowsPadded) * static_cast<size_t>(k / 2);
    L.packedInputScaleBytes = static_cast<size_t>(geo.rowsPadded) * static_cast<size_t>(colsPad);
    L.barrierBytes = sizeof(int32_t);
    L.expertWriteRowsBytes = static_cast<size_t>(stateE) * sizeof(int32_t);
    L.expertTileBaseBytes = static_cast<size_t>(stateE + 1) * sizeof(int32_t);
    L.taskBytes = static_cast<size_t>(geo.maxTasks) * sizeof(int32_t);
    L.tileWriteCountBytes = static_cast<size_t>(geo.physicalTiles) * sizeof(int32_t);

    auto place = [](size_t& off, size_t& cursor, size_t bytes) {
        off = cursor;
        cursor = alignUp(cursor + bytes, kDeviceAlignment);
    };

    size_t cur = 0;
    place(L.rowCounts, cur, L.rowCountsBytes);
    place(L.tokenMap, cur, L.tokenMapBytes);
    place(L.tokenWeights, cur, L.tokenWeightsBytes);
    place(L.barrierCount, cur, L.barrierBytes);
    place(L.barrierEpoch, cur, L.barrierBytes);
    place(L.pairHead, cur, L.barrierBytes);
    place(L.producersDoneCount, cur, L.barrierBytes);
    place(L.allWorkPublished, cur, L.barrierBytes);
    place(L.taskHead, cur, L.barrierBytes);
    place(L.taskTail, cur, L.barrierBytes);
    place(L.expertWriteRows, cur, L.expertWriteRowsBytes);
    place(L.expertTileBase, cur, L.expertTileBaseBytes);
    place(L.taskReady, cur, L.taskBytes);
    place(L.taskExpert, cur, L.taskBytes);
    place(L.taskMTile, cur, L.taskBytes);
    place(L.taskSliceBegin, cur, L.taskBytes);
    place(L.taskSliceCount, cur, L.taskBytes);
    place(L.taskValidRows, cur, L.taskBytes);
    place(L.tileWriteCount, cur, L.tileWriteCountBytes);
    place(L.packedInput, cur, L.packedInputBytes);
    place(L.packedInputScale, cur, L.packedInputScaleBytes);
    L.total = cur;
    return L;
}

inline void* offsetPtr(void* base, size_t byteOffset)
{
    return static_cast<void*>(static_cast<std::byte*>(base) + byteOffset);
}
} // namespace

size_t CuteDslNvfp4MoeRunner::getWorkspaceSize(int32_t maxNumTokens, int32_t maxRoutedRows, int32_t numExperts,
    int32_t topK, int32_t hiddenSize, int32_t moeInterSize, CuteDslMoeBackend backend)
{
    (void) topK;
    if (maxNumTokens <= 0 || maxRoutedRows <= 0 || numExperts <= 0 || hiddenSize <= 0 || moeInterSize <= 0)
    {
        return 0;
    }

    // Size the decode sub-layout with the same cap runDecode() uses (see decodeCapRoutedRows), and
    // only when decode can actually run. 2*moeInterSize covers the worst-case prefill N1 (gated
    // activations have N1=2*I) so the task queue fits any activation.
    size_t decodeTotal = 0;
    if (backend != CuteDslMoeBackend::kPrefill)
    {
        int32_t const maxRowsDecode = decodeCapRoutedRows(backend, maxRoutedRows);
        decodeTotal = buildDecodeLayout(numExperts, numExperts, maxRowsDecode, hiddenSize).total;
    }
    auto const prefillL = buildPrefillLayout(numExperts, maxRoutedRows, hiddenSize, 2 * moeInterSize);

    // Max of both layouts so kAuto can dispatch either backend; + device-alignment tail pad.
    return std::max(decodeTotal, prefillL.total) + kDeviceAlignment;
}

// Backend resolution
CuteDslMoeBackend CuteDslNvfp4MoeRunner::resolveBackend(CuteDslMoeBackend backend, int32_t numTokens, int32_t topK)
{
    // Explicit decode / prefill override is honored as-is.
    if (backend == CuteDslMoeBackend::kDecode || backend == CuteDslMoeBackend::kPrefill)
    {
        return backend;
    }
    // kAuto: mirror select_sm120_moe_backend in moe_dispatch.py — small routed sets go
    // to the decode backend; larger sets go to the prefill backend.
    int64_t const routedRows = static_cast<int64_t>(std::max(numTokens, 1)) * static_cast<int64_t>(std::max(topK, 1));
    return routedRows <= static_cast<int64_t>(kDecodePrefillCutoverRoutedRows) ? CuteDslMoeBackend::kDecode
                                                                               : CuteDslMoeBackend::kPrefill;
}

int32_t CuteDslNvfp4MoeRunner::decodeCapRoutedRows(CuteDslMoeBackend backend, int32_t maxRoutedRows)
{
    // Only valid for backends that can dispatch decode; kPrefill never does.
    ELLM_CHECK(
        backend != CuteDslMoeBackend::kPrefill, "decodeCapRoutedRows must not be queried for the prefill backend");
    // Explicit kDecode bypasses resolveBackend()'s cutover and may see up to the full profile-wide
    // maxRoutedRows; kAuto only routes num_tokens*top_k <= the cutover to decode.
    int32_t const cap = (backend == CuteDslMoeBackend::kDecode)
        ? maxRoutedRows
        : std::min(maxRoutedRows, kDecodePrefillCutoverRoutedRows);
    return std::max(1, cap);
}

// Decode backend launch

// Macro that builds the per-variant Tensor_* structs and calls the AOT wrapper. The
// PREFIX token must be one of the full variant names, e.g.
// nvfp4_fused_moe_decode_identity_m128_n128 / _swiglu_n128 etc.
// MODULE must be the matching static module member.
//
// Relies on these local variables being in scope:
//   params, L, ws, static_cast<int32_t>(params.numExperts) (state_E/weight_E, single-device),
//   barrierCountT, barrierEpochT, activeExpertCountT, bW13T, bDownT,
//   packedAPtr, sfaPtr, packedAStoragePtr, scaleStoragePtr,
//   rowCountsPtr, weightExpertIdsPtr, globalToLocalExpertPtr,
//   tokenMapPtr, tokenWeightsPtr,
//   numTokens, maxRows, stateE, weightE, rowsPadded, K, N, numTopk, stream, ret.
// clang-format off
#define CALL_DECODE_MOE(PREFIX, MODULE)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        PREFIX##_Tensor_barrier_count_t _barrierCount{};                                                               \
        _barrierCount.data = barrierCountT;                                                                            \
        PREFIX##_Tensor_barrier_epoch_t _barrierEpoch{};                                                               \
        _barrierEpoch.data = barrierEpochT;                                                                            \
        PREFIX##_Tensor_active_expert_count_t _activeExpertCount{};                                                    \
        _activeExpertCount.data = activeExpertCountT;                                                                  \
        ret = cute_dsl_##PREFIX##_wrapper(&(MODULE).module,                                                            \
            const_cast<void*>(static_cast<void const*>(params.hiddenStates)),                                          \
            const_cast<void*>(static_cast<void const*>(params.topkIds)),                                               \
            const_cast<void*>(static_cast<void const*>(params.topkWeights)),                                           \
            packedAPtr, sfaPtr, packedAStoragePtr, scaleStoragePtr,                                                    \
            &_barrierCount, &_barrierEpoch, bW13T,                                                                     \
            const_cast<void*>(static_cast<void const*>(params.fc1BlocksScale)),                                        \
            bDownT,                                                                                                    \
            const_cast<void*>(static_cast<void const*>(params.fc2BlocksScale)),                                        \
            rowCountsPtr, &_activeExpertCount, weightExpertIdsPtr, globalToLocalExpertPtr,                             \
            const_cast<void*>(static_cast<void const*>(params.inputGlobalScale)),                                      \
            const_cast<void*>(static_cast<void const*>(params.fc1Alpha)),                                              \
            const_cast<void*>(static_cast<void const*>(params.fc2Alpha)),                                              \
            const_cast<void*>(static_cast<void const*>(params.downInputScale)),                                        \
            params.output, tokenMapPtr, tokenWeightsPtr,                                                               \
            numTokens, maxRows, stateE, weightE, rowsPadded, K, N, numTopk,                                               \
            trt_edgellm::getDeviceMultiProcessorCount(), stream);                                                                  \
    } while (0)
// clang-format on

// Nested dispatch: activation x N-tile variant = 8 possible decode wrappers.
// clang-format off
#define DISPATCH_DECODE_ACTIVATION(N_SUFFIX)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        switch (params.activation)                                                                                     \
        {                                                                                                              \
        case CuteDslMoeActivation::kIdentity:                                                                          \
            CALL_DECODE_MOE(nvfp4_fused_moe_decode_identity_##N_SUFFIX,                                                \
                sDecodeIdentity_##N_SUFFIX);                                                                           \
            break;                                                                                                     \
        case CuteDslMoeActivation::kSiLU:                                                                              \
            CALL_DECODE_MOE(nvfp4_fused_moe_decode_silu_##N_SUFFIX,                                                    \
                sDecodeSiLU_##N_SUFFIX);                                                                               \
            break;                                                                                                     \
        case CuteDslMoeActivation::kSwiGLU:                                                                            \
            CALL_DECODE_MOE(nvfp4_fused_moe_decode_swiglu_##N_SUFFIX,                                                  \
                sDecodeSwiGLU_##N_SUFFIX);                                                                             \
            break;                                                                                                     \
        case CuteDslMoeActivation::kGeLU:                                                                              \
            CALL_DECODE_MOE(nvfp4_fused_moe_decode_gelu_##N_SUFFIX,                                                    \
                sDecodeGeLU_##N_SUFFIX);                                                                               \
            break;                                                                                                     \
        case CuteDslMoeActivation::kReLU2:                                                                             \
            CALL_DECODE_MOE(nvfp4_fused_moe_decode_relu2_##N_SUFFIX,                                                   \
                sDecodeReLU2_##N_SUFFIX);                                                                              \
            break;                                                                                                     \
        case CuteDslMoeActivation::kGeGLU:                                                                             \
            CALL_DECODE_MOE(nvfp4_fused_moe_decode_geglu_##N_SUFFIX,                                                   \
                sDecodeGeGLU_##N_SUFFIX);                                                                              \
            break;                                                                                                     \
        default:                                                                                                       \
            LOG_ERROR("CuteDslNvfp4MoeRunner: unsupported activation for decode backend");                             \
            return -1;                                                                                                 \
        }                                                                                                              \
    } while (0)
// clang-format on

int32_t CuteDslNvfp4MoeRunner::runDecode(CuteDslNvfp4MoeParams const& params, void* workspace, cudaStream_t stream)
{
    int32_t const stateE = params.numExperts;
    int32_t const weightE = params.numExperts; // single-device.
    // Size the per-expert workspace from the actual routed rows (num_tokens * top_k), capped by the
    // same decodeCapRoutedRows() getWorkspaceSize() reserved for -- not the profile-wide
    // max_routed_rows, which would overflow the kernel's int32 addressing (num_experts *
    // max_routed_rows * K/2) and over-allocate. Mirrors runPrefill().
    int32_t const decodeCap = decodeCapRoutedRows(params.backend, params.maxRoutedRows);
    int64_t const routedRows64
        = static_cast<int64_t>(std::max(1, params.numTokens)) * static_cast<int64_t>(std::max(1, params.topK));
    int32_t const maxRows = std::max<int32_t>(1, static_cast<int32_t>(std::min<int64_t>(routedRows64, decodeCap)));
    auto const L = buildDecodeLayout(stateE, weightE, maxRows, params.hiddenSize);

    std::byte* ws = static_cast<std::byte*>(workspace);

    zeroGridBarrier(ws, L.barrierCount, L.barrierEpoch, L.barrierBytes, stream);

    // weight_expert_ids and global_to_local_expert are the identity map
    // (single-device: state_E == weight_E == numExperts). The plugin owns
    // these tables (allocated via IGpuAllocator at attachToContext time)
    // and threads them through CuteDslNvfp4MoeParams so the runner has no
    // allocations or host-blocking copies on the enqueue path. Two
    // distinct, non-aliasing copies are required because the decode kernel
    // aliases them internally (empirical: aliasing collapsed cos_sim to
    // ~0.73).
    if (params.weightExpertIds == nullptr || params.globalToLocalExpertIds == nullptr)
    {
        LOG_ERROR(
            "CuteDslNvfp4MoeRunner: identity expert tables not provided by the caller "
            "(weightExpertIds / globalToLocalExpertIds must be non-null device pointers)");
        return -1;
    }
    int32_t* const weightExpertIdsPtr = const_cast<int32_t*>(params.weightExpertIds);
    int32_t* const globalToLocalExpertPtr = const_cast<int32_t*>(params.globalToLocalExpertIds);

    void* const barrierCountT = offsetPtr(workspace, L.barrierCount);
    void* const barrierEpochT = offsetPtr(workspace, L.barrierEpoch);
    void* const activeExpertCountT = offsetPtr(workspace, L.activeExpertCount);
    void* const bW13T = const_cast<void*>(static_cast<void const*>(params.fc1QWeights));
    void* const bDownT = const_cast<void*>(static_cast<void const*>(params.fc2QWeights));
    void* const packedAPtr = offsetPtr(workspace, L.packedInput);
    void* const sfaPtr = offsetPtr(workspace, L.packedInputScale);
    void* const packedAStoragePtr = offsetPtr(workspace, L.packedInput);
    void* const scaleStoragePtr = offsetPtr(workspace, L.packedInputScale);
    void* const rowCountsPtr = offsetPtr(workspace, L.rowCounts);
    void* const tokenMapPtr = offsetPtr(workspace, L.tokenMap);
    void* const tokenWeightsPtr = offsetPtr(workspace, L.tokenWeights);

    int32_t const numTokens = params.numTokens;
    int32_t const rowsPadded = L.rowsPadded;
    int32_t const K = params.hiddenSize;
    int32_t const N = params.moeInterSize;
    int32_t const numTopk = params.topK;
    int32_t const mmaTilerN = selectMmaTilerN(N);

    int32_t ret = -1;
    // n256 path disabled — selectMmaTilerN() always returns kLevelTileN.
    (void) mmaTilerN;
    if (selectMmaTilerM(numTokens * numTopk) == kLevelTileMSmall)
    {
        DISPATCH_DECODE_ACTIVATION(m64_n128);
    }
    else
    {
        DISPATCH_DECODE_ACTIVATION(m128_n128);
    }

    if (ret != 0)
    {
        LOG_ERROR("CuteDslNvfp4MoeRunner: decode fused MoE kernel returned error code: %d", ret);
    }
    return ret;
}

#undef CALL_DECODE_MOE
#undef DISPATCH_DECODE_ACTIVATION

// Prefill backend launch
//
// Wrapper signature (see nvfp4_fused_moe_prefill_*_n{128,256}.h):
//   module, a_ptr, topk_ids_ptr, topk_weights_ptr,
//   packed_a_ptr, sfa_ptr, packed_a_storage_ptr, scale_storage_ptr,
//   &barrier_count, &barrier_epoch, &pair_head, &producers_done_count,
//   &all_work_published, &task_head, &task_tail,
//   task_ready_ptr, task_expert_ptr, task_m_tile_ptr, task_slice_begin_ptr,
//   task_slice_count_ptr, task_valid_rows_ptr, tile_write_count_ptr,
//   b_w13_ptr, sfb_w13_ptr, b_down_ptr, sfb_down_ptr,
//   row_counts_ptr, expert_write_rows_ptr, expert_tile_base_ptr,
//   input_gs_ptr, alpha_ptr, down_alpha_ptr, global_scale_ptr,
//   scatter_ptr, token_map_ptr, token_weights_ptr,
//   num_tokens, max_rows, rows_padded, max_tasks, max_phys_tiles,
//   K, N, weight_E, num_topk, stream.
//
// The 7 Tensor_* struct types the macro still needs (barrier_count,
// barrier_epoch, pair_head, producers_done_count, all_work_published,
// task_head, task_tail) all come from the PREFIX-parameterised header;
// everything else is passed as a raw void*.

// clang-format off
#define CALL_PREFILL_MOE(PREFIX, MODULE)                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        PREFIX##_Tensor_barrier_count_t _barrierCount{};                                                               \
        _barrierCount.data = barrierCountT;                                                                            \
        PREFIX##_Tensor_barrier_epoch_t _barrierEpoch{};                                                               \
        _barrierEpoch.data = barrierEpochT;                                                                            \
        PREFIX##_Tensor_pair_head_t _pairHead{};                                                                       \
        _pairHead.data = pairHeadT;                                                                                    \
        PREFIX##_Tensor_producers_done_count_t _producersDone{};                                                       \
        _producersDone.data = producersDoneT;                                                                          \
        PREFIX##_Tensor_all_work_published_t _allWorkPublished{};                                                      \
        _allWorkPublished.data = allWorkPublishedT;                                                                    \
        PREFIX##_Tensor_task_head_t _taskHead{};                                                                       \
        _taskHead.data = taskHeadT;                                                                                    \
        PREFIX##_Tensor_task_tail_t _taskTail{};                                                                       \
        _taskTail.data = taskTailT;                                                                                    \
        ret = cute_dsl_##PREFIX##_wrapper(&(MODULE).module,                                                            \
            const_cast<void*>(static_cast<void const*>(params.hiddenStates)),                                          \
            const_cast<void*>(static_cast<void const*>(params.topkIds)),                                               \
            const_cast<void*>(static_cast<void const*>(params.topkWeights)),                                           \
            packedAPtr, sfaPtr, packedAStoragePtr, scaleStoragePtr,                                                    \
            &_barrierCount, &_barrierEpoch,                                                                            \
            &_pairHead, &_producersDone, &_allWorkPublished, &_taskHead, &_taskTail,                                   \
            taskReadyPtr, taskExpertPtr, taskMTilePtr, taskSliceBeginPtr, taskSliceCountPtr, taskValidRowsPtr,         \
            tileWriteCountPtr,                                                                                         \
            bW13T,                                                                                                     \
            const_cast<void*>(static_cast<void const*>(params.fc1BlocksScale)),                                        \
            bDownT,                                                                                                    \
            const_cast<void*>(static_cast<void const*>(params.fc2BlocksScale)),                                        \
            rowCountsT, expertWriteRowsT, expertTileBaseT,                                                             \
            const_cast<void*>(static_cast<void const*>(params.inputGlobalScale)),                                      \
            const_cast<void*>(static_cast<void const*>(params.fc1Alpha)),                                              \
            const_cast<void*>(static_cast<void const*>(params.fc2Alpha)),                                              \
            const_cast<void*>(static_cast<void const*>(params.downInputScale)),                                        \
            params.output, tokenMapPtr, tokenWeightsPtr,                                                               \
            numTokens, maxRows, rowsPadded, maxTasks, physicalTiles,                                                   \
            K, N, weightE, numTopk, trt_edgellm::getDeviceMultiProcessorCount(), stream);                                          \
    } while (0)
// clang-format on

// Nested dispatch: activation x N-tile variant = 8 possible prefill wrappers.
// clang-format off
#define DISPATCH_PREFILL_ACTIVATION(N_SUFFIX)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        switch (params.activation)                                                                                     \
        {                                                                                                              \
        case CuteDslMoeActivation::kIdentity:                                                                          \
            CALL_PREFILL_MOE(nvfp4_fused_moe_prefill_identity_##N_SUFFIX,                                              \
                sPrefillIdentity_##N_SUFFIX);                                                                          \
            break;                                                                                                     \
        case CuteDslMoeActivation::kSiLU:                                                                              \
            CALL_PREFILL_MOE(nvfp4_fused_moe_prefill_silu_##N_SUFFIX,                                                  \
                sPrefillSiLU_##N_SUFFIX);                                                                              \
            break;                                                                                                     \
        case CuteDslMoeActivation::kSwiGLU:                                                                            \
            CALL_PREFILL_MOE(nvfp4_fused_moe_prefill_swiglu_##N_SUFFIX,                                                \
                sPrefillSwiGLU_##N_SUFFIX);                                                                            \
            break;                                                                                                     \
        case CuteDslMoeActivation::kGeLU:                                                                              \
            CALL_PREFILL_MOE(nvfp4_fused_moe_prefill_gelu_##N_SUFFIX,                                                  \
                sPrefillGeLU_##N_SUFFIX);                                                                              \
            break;                                                                                                     \
        case CuteDslMoeActivation::kReLU2:                                                                             \
            CALL_PREFILL_MOE(nvfp4_fused_moe_prefill_relu2_##N_SUFFIX,                                                 \
                sPrefillReLU2_##N_SUFFIX);                                                                             \
            break;                                                                                                     \
        case CuteDslMoeActivation::kGeGLU:                                                                             \
            CALL_PREFILL_MOE(nvfp4_fused_moe_prefill_geglu_##N_SUFFIX,                                                 \
                sPrefillGeGLU_##N_SUFFIX);                                                                             \
            break;                                                                                                     \
        default:                                                                                                       \
            LOG_ERROR("CuteDslNvfp4MoeRunner: unsupported activation for prefill backend");                            \
            return -1;                                                                                                 \
        }                                                                                                              \
    } while (0)
// clang-format on

int32_t CuteDslNvfp4MoeRunner::runPrefill(CuteDslNvfp4MoeParams const& params, void* workspace, cudaStream_t stream)
{
    int32_t const stateE = params.numExperts;
    int32_t const routedRows = params.numTokens * params.topK;
    // For gated activations (swiglu) the fc1 weight has 2*moeInterSize N-tiles;
    // pass the correct N1 so the task-queue sizing matches the kernel's gate_tile_cnt.
    int32_t const n1
        = (params.activation == CuteDslMoeActivation::kSwiGLU) ? 2 * params.moeInterSize : params.moeInterSize;
    auto const L = buildPrefillLayout(stateE, routedRows, params.hiddenSize, n1);

    std::byte* ws = static_cast<std::byte*>(workspace);

    zeroGridBarrier(ws, L.barrierCount, L.barrierEpoch, L.barrierBytes, stream);

    void* const barrierCountT = offsetPtr(workspace, L.barrierCount);
    void* const barrierEpochT = offsetPtr(workspace, L.barrierEpoch);
    void* const pairHeadT = offsetPtr(workspace, L.pairHead);
    void* const producersDoneT = offsetPtr(workspace, L.producersDoneCount);
    void* const allWorkPublishedT = offsetPtr(workspace, L.allWorkPublished);
    void* const taskHeadT = offsetPtr(workspace, L.taskHead);
    void* const taskTailT = offsetPtr(workspace, L.taskTail);
    void* const rowCountsT = offsetPtr(workspace, L.rowCounts);
    void* const expertWriteRowsT = offsetPtr(workspace, L.expertWriteRows);
    void* const expertTileBaseT = offsetPtr(workspace, L.expertTileBase);
    void* const bW13T = const_cast<void*>(static_cast<void const*>(params.fc1QWeights));
    void* const bDownT = const_cast<void*>(static_cast<void const*>(params.fc2QWeights));
    // packed_a and packed_a_storage alias the same byte buffer; same for sfa / scale_storage
    // (matches the two Python-side views in allocate_sm120_prefill_workspace).
    void* const packedAPtr = offsetPtr(workspace, L.packedInput);
    void* const sfaPtr = offsetPtr(workspace, L.packedInputScale);
    void* const packedAStoragePtr = offsetPtr(workspace, L.packedInput);
    void* const scaleStoragePtr = offsetPtr(workspace, L.packedInputScale);
    void* const taskReadyPtr = offsetPtr(workspace, L.taskReady);
    void* const taskExpertPtr = offsetPtr(workspace, L.taskExpert);
    void* const taskMTilePtr = offsetPtr(workspace, L.taskMTile);
    void* const taskSliceBeginPtr = offsetPtr(workspace, L.taskSliceBegin);
    void* const taskSliceCountPtr = offsetPtr(workspace, L.taskSliceCount);
    void* const taskValidRowsPtr = offsetPtr(workspace, L.taskValidRows);
    void* const tileWriteCountPtr = offsetPtr(workspace, L.tileWriteCount);
    void* const tokenMapPtr = offsetPtr(workspace, L.tokenMap);
    void* const tokenWeightsPtr = offsetPtr(workspace, L.tokenWeights);

    int32_t const numTokens = params.numTokens;
    // Python passes workspace.max_rows (== rows_padded in the prefill workspace) for both
    // the max_rows and rows_padded args -- they are the same number.
    int32_t const maxRows = L.rowsPadded;
    int32_t const rowsPadded = L.rowsPadded;
    int32_t const maxTasks = L.maxTasks;
    int32_t const physicalTiles = L.physicalTiles;
    int32_t const K = params.hiddenSize;
    int32_t const N = params.moeInterSize;
    int32_t const weightE = params.numExperts;
    int32_t const numTopk = params.topK;
    int32_t const mmaTilerN = selectMmaTilerN(N);

    int32_t ret = -1;
    // n256 path disabled — selectMmaTilerN() always returns kLevelTileN.
    (void) mmaTilerN;
    DISPATCH_PREFILL_ACTIVATION(n128);

    if (ret != 0)
    {
        LOG_ERROR("CuteDslNvfp4MoeRunner: prefill fused MoE kernel returned error code: %d", ret);
    }
    return ret;
}

#undef CALL_PREFILL_MOE
#undef DISPATCH_PREFILL_ACTIVATION

int32_t CuteDslNvfp4MoeRunner::run(CuteDslNvfp4MoeParams const& params, void* workspace, cudaStream_t stream)
{
    if (workspace == nullptr)
    {
        LOG_ERROR("CuteDslNvfp4MoeRunner: null workspace pointer");
        return -1;
    }
    if (params.ioDtype != CuteDslMoeIoDtype::kFP16)
    {
        LOG_ERROR("CuteDslNvfp4MoeRunner: only FP16 io_dtype is supported");
        return -1;
    }

    // resolveBackend picks decode or prefill based on routed-row count (for kAuto) or
    // honors an explicit override. Both backends are supported.
    auto const backend = resolveBackend(params.backend, params.numTokens, params.topK);
    if (!ensureKernelModules(params, stream))
    {
        return -1;
    }
    if (backend == CuteDslMoeBackend::kPrefill)
    {
        return runPrefill(params, workspace, stream);
    }
    return runDecode(params, workspace, stream);
}

} // namespace trt_edgellm

#endif // CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
