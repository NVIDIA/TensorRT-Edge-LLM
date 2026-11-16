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

#include "attentionPlugin.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/executionPhase.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "common/trtInterop.h"
#include "kernels/contextAttentionKernels/attentionScaleUtils.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "kernels/decodeAttentionKernels/ctmDecodeAttentionRunner.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "plugins/utils/pluginUtils.h"
#include "plugins/utils/raggedPluginMetadata.h"

// CuTe DSL FMHA kernel (Blackwell SM100+)
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
#include "kernels/contextAttentionKernels/cuteDslFMHARunner.h"
#endif

// FMHA-v2 CuTe DSL FMHA kernels.
#include "kernels/contextAttentionKernels/cuteDslFMHAV2Runner.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{
using nvinfer1::DataType;

namespace
{
constexpr char const* kATTENTION_PLUGIN_VERSION{"1"};
constexpr char const* kATTENTION_PLUGIN_NAME{"AttentionPlugin"};
//! Self-describing blob of (XQAJitKey, cubin) pairs; see serializeXQAJitKernels.
constexpr char const* kXQA_JIT_KERNELS_FIELD{"xqa_jit_kernels"};
constexpr int32_t kROPE_MIN_PDL_SM_VERSION{90};

// Select KV cache storage datatype based on FP8 enablement
static inline DataType selectKvCacheDataType(bool enableFp8KVCache)
{
    return enableFp8KVCache ? DataType::kFP8 : DataType::kHALF;
}

bool requestRopePdl()
{
    static bool const enabled = []() {
        char const* const value = std::getenv("EDGELLM_ENABLE_ROPE_PDL");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

// Define the mapping of input and output indices of the AttentionPlugin.
// Packed-QKV contract: Q/K/V concatenated on the last dim into one input tensor (a single
// fused QKV GEMM output). RoPE + split happens internally via launchApplyRopeFromPackedToSplit.
// See applyRopeWriteKV.h for kernel docs.
//
// Optional inputs follow the required ones in a fixed relative order:
//   [q_norm_gamma, k_norm_gamma]        when enable_qk_norm         (engine-weight constants)
//   [context_mask_selector]             when enable_context_mask_selector
//   [attention_mask, attention_pos_id]  when enable_tree_attention
//   [vision_block_ids]                  when enable_vision_block_attention
//   [skip_softmax_prefill_scale]        when skip_softmax_prefill_input (prefill table deployed)
//   [skip_softmax_decode_scale]         when skip_softmax_decode_input  (decode table deployed)
//   [swa_kv_cache_mode]                 when bounded SWA storage is supported
//   [attention_sinks]                   when enable_attention_sink   (engine-weight constant)
//   [query_start_offsets, attention_sequence_lengths, execution_phase_marker,
//    context_sequence_count_carrier]    always
//   [prefix_key_value, prefix_page_table] when external_kv_prefix; main KV inputs then describe FP16 action scratch
//
// The gamma weights are engine weights: FP16 Constant initializers wired to plugin inputs,
// baked into the engine at build time (device-resident, no runtime upload). Models without
// qk_norm do not wire these inputs at all.
constexpr int32_t kIN_QKV_IDX{0};
constexpr int32_t kIN_KV_CACHE_IDX{1};
constexpr int32_t kIN_QUERY_LENGTH_IDX{2};
constexpr int32_t kIN_ROPE_COS_SIN_IDX{3};
constexpr int32_t kIN_KV_CACHE_START_IDX{4};
constexpr int32_t kIN_KV_PAGE_TABLE_IDX{5};
constexpr int32_t kOUT_ATTENTION_IDX{0};
constexpr int32_t kOUT_KV_CACHE_IDX{1};

constexpr int32_t kNUM_REQUIRED_INPUTS{6};
constexpr int32_t kNUM_TOKEN_METADATA_INPUTS{4};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{2};
constexpr int64_t kFULL_KV_CACHE_MODE_LENGTH{0};
constexpr int64_t kBOUNDED_SWA_CACHE_MODE_LENGTH{1};
} // namespace

struct AttentionInputLayout
{
    int32_t qNormGamma{-1};
    int32_t kNormGamma{-1};
    int32_t contextMaskSelector{-1};
    int32_t attentionMask{-1};
    int32_t attentionPosId{-1};
    int32_t visionBlockIds{-1};
    int32_t skipSoftmaxPrefillScale{-1};
    int32_t skipSoftmaxDecodeScale{-1};
    int32_t swaCacheMode{-1};
    int32_t attentionSink{-1};
    int32_t prefixPool{-1};
    int32_t prefixPageTable{-1};
    int32_t raggedMetadataStart{-1};
    int32_t numInputs{0};
};

namespace
{
//! Which optional input groups a layer carries; resolveAttentionInputLayout turns it into slot indices.
struct AttentionInputFlags
{
    bool qkNorm{false};
    bool contextMaskSelector{false};
    bool treeAttention{false};
    bool visionBlock{false};
    bool skipSoftmaxPrefillScale{false};
    bool skipSoftmaxDecodeScale{false};
    bool boundedKVCache{false};
    bool attentionSink{false};
    bool externalKVPrefix{false};
};

constexpr AttentionInputLayout resolveAttentionInputLayout(AttentionInputFlags const& flags)
{
    AttentionInputLayout layout;
    int32_t cursor = kNUM_REQUIRED_INPUTS;
    if (flags.qkNorm)
    {
        layout.qNormGamma = cursor++;
        layout.kNormGamma = cursor++;
    }
    if (flags.contextMaskSelector)
    {
        layout.contextMaskSelector = cursor++;
    }
    if (flags.treeAttention)
    {
        layout.attentionMask = cursor++;
        layout.attentionPosId = cursor++;
    }
    if (flags.visionBlock)
    {
        layout.visionBlockIds = cursor++;
    }
    if (flags.skipSoftmaxPrefillScale)
    {
        layout.skipSoftmaxPrefillScale = cursor++;
    }
    if (flags.skipSoftmaxDecodeScale)
    {
        layout.skipSoftmaxDecodeScale = cursor++;
    }
    if (flags.boundedKVCache)
    {
        layout.swaCacheMode = cursor++;
    }
    if (flags.attentionSink)
    {
        layout.attentionSink = cursor++;
    }
    layout.raggedMetadataStart = cursor;
    cursor += kNUM_TOKEN_METADATA_INPUTS;
    if (flags.externalKVPrefix)
    {
        layout.prefixPool = cursor++;
        layout.prefixPageTable = cursor++;
    }
    layout.numInputs = cursor;
    return layout;
}

enum class AttentionExecutionMode
{
    kINVALID,
    kNORMAL_PREFILL,
    kCHUNKED_PREFILL,
    kVANILLA_DECODING,
    kTREE_DECODING
};

constexpr AttentionExecutionMode resolveAttentionExecutionMode(rt::ExecutionPhase phase, bool enableTreeAttention)
{
    switch (phase)
    {
    case rt::ExecutionPhase::kContextPrefill:
    case rt::ExecutionPhase::kDiffusionCommit: return AttentionExecutionMode::kNORMAL_PREFILL;
    // A denoise step attends over its whole chunk at once, which is the tree kernel's shape
    // even though nothing about it is speculative.
    case rt::ExecutionPhase::kDiffusionDenoise:
        return enableTreeAttention ? AttentionExecutionMode::kTREE_DECODING : AttentionExecutionMode::kNORMAL_PREFILL;
    case rt::ExecutionPhase::kContextChunk: return AttentionExecutionMode::kCHUNKED_PREFILL;
    case rt::ExecutionPhase::kAutoregressiveDecode: return AttentionExecutionMode::kVANILLA_DECODING;
    case rt::ExecutionPhase::kSpecDraftProposal:
        return enableTreeAttention ? AttentionExecutionMode::kTREE_DECODING : AttentionExecutionMode::kNORMAL_PREFILL;
    case rt::ExecutionPhase::kSpecTargetVerify: return AttentionExecutionMode::kTREE_DECODING;
    case rt::ExecutionPhase::kMixedPrefillDecode: return AttentionExecutionMode::kINVALID;
    }
    return AttentionExecutionMode::kINVALID;
}

constexpr bool attentionModeUsesQueryLengths(AttentionExecutionMode mode)
{
    return mode == AttentionExecutionMode::kNORMAL_PREFILL || mode == AttentionExecutionMode::kCHUNKED_PREFILL;
}

static_assert(attentionModeUsesQueryLengths(
    resolveAttentionExecutionMode(rt::ExecutionPhase::kSpecDraftProposal, /*enableTreeAttention=*/false)));
static_assert(!attentionModeUsesQueryLengths(
    resolveAttentionExecutionMode(rt::ExecutionPhase::kSpecDraftProposal, /*enableTreeAttention=*/true)));
static_assert(attentionModeUsesQueryLengths(
    resolveAttentionExecutionMode(rt::ExecutionPhase::kDiffusionDenoise, /*enableTreeAttention=*/false)));
static_assert(!attentionModeUsesQueryLengths(
    resolveAttentionExecutionMode(rt::ExecutionPhase::kDiffusionDenoise, /*enableTreeAttention=*/true)));

enum class RuntimeContextMaskMode
{
    kDefault,
    kPadding,
};

RuntimeContextMaskMode selectRuntimeContextMaskMode(
    PluginTensorDesc const* inputDesc, bool enableContextMaskSelector, bool enableQKNorm)
{
    if (!enableContextMaskSelector)
    {
        return RuntimeContextMaskMode::kDefault;
    }

    AttentionInputFlags flags;
    flags.qkNorm = enableQKNorm;
    flags.contextMaskSelector = enableContextMaskSelector;
    AttentionInputLayout const layout = resolveAttentionInputLayout(flags);
    PluginTensorDesc const& selectorDesc = inputDesc[layout.contextMaskSelector];
    return selectorDesc.dims.d[0] > 0 ? RuntimeContextMaskMode::kPadding : RuntimeContextMaskMode::kDefault;
}

bool isPaddingContextMask(RuntimeContextMaskMode const mode)
{
    return mode == RuntimeContextMaskMode::kPadding;
}

struct FMHAKernelSelection
{
    ContextFMHABackend backend{ContextFMHABackend::kNONE};
    bool canImplement{false};
};

FMHAKernelSelection selectFMHAKernels(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t smVersion,
    nvinfer1::DataType dataType, bool useSlidingWindow, bool enableFp8KVCache,
    [[maybe_unused]] bool allowPagedBlackwell = true, bool requirePagedRagged = false)
{
    CuteDslFMHAV2MaskType const fmhaV2Mask
        = useSlidingWindow ? CuteDslFMHAV2MaskType::kSLIDING_CAUSAL : CuteDslFMHAV2MaskType::kCAUSAL;
    if (requirePagedRagged)
    {
        bool const canImplementRagged = !enableFp8KVCache
            && CuteDslFMHAV2Runner::canImplementPagedRagged(
                numQHeads, numKVHeads, headSize, smVersion, fromTrt(dataType), fmhaV2Mask);
        return canImplementRagged ? FMHAKernelSelection{ContextFMHABackend::kCUTE_DSL_FMHA_V2, true}
                                  : FMHAKernelSelection{};
    }
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
    if (allowPagedBlackwell && CuteDslFMHARunner::canImplement(headSize, smVersion))
    {
        return {ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL, true};
    }
#endif
    // Native-paged FMHA-v2 is FP16-only. Preserve the FP8 route, including D512, through the dense FMHA-v2 kernels
    // and the page-table-aware split-KV gather.
    bool const canImplementFMHAV2 = enableFp8KVCache
        ? CuteDslFMHAV2Runner::canImplement(numQHeads, numKVHeads, headSize, smVersion, fromTrt(dataType), fmhaV2Mask)
        : CuteDslFMHAV2Runner::canImplementPaged(
              numQHeads, numKVHeads, headSize, smVersion, fromTrt(dataType), fmhaV2Mask);
    if (canImplementFMHAV2)
    {
        return {ContextFMHABackend::kCUTE_DSL_FMHA_V2, true};
    }

    return {};
}

bool canImplementPaddingFMHA(ContextFMHABackend backend, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t smVersion, nvinfer1::DataType dataType)
{
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
    if (backend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL && (headSize == 256 || headSize == 512)
        && CuteDslFMHARunner::canImplement(headSize, smVersion))
    {
        return true;
    }
#endif
    if (backend == ContextFMHABackend::kCUTE_DSL_FMHA_V2
        && CuteDslFMHAV2Runner::canImplement(
            numQHeads, numKVHeads, headSize, smVersion, fromTrt(dataType), CuteDslFMHAV2MaskType::kPADDING))
    {
        return true;
    }
    return false;
}

std::vector<uint8_t> parsePluginBytesField(char const* fieldName, PluginFieldCollection const* fc)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        if (std::string(fieldName) == fc->fields[i].name && fc->fields[i].length > 0)
        {
            ELLM_CHECK(fc->fields[i].type == PluginFieldType::kCHAR,
                std::string(fieldName) + " must use PluginFieldType::kCHAR.");
            ELLM_CHECK(fc->fields[i].data != nullptr, std::string(fieldName) + " data must not be null.");
            auto const* data = static_cast<uint8_t const*>(fc->fields[i].data);
            return std::vector<uint8_t>(data, data + fc->fields[i].length);
        }
    }
    return {};
}

// Workspace layout (cumulative, worst-case across all execution paths):
//
//   Slot  | Shape                            | Type  | Used by
//   ------+----------------------------------+-------+------------------------------------------
//   0     | [B+1]                            | INT32 | cuQSeqLens          (prefill)
//   1     | [B+1]                            | INT32 | cuKVSeqLens         (prefill)
//   2     | [B]                              | INT32 | kvCacheEndIdxs      (prefill)
//   3     | [B+1]                            | INT32 | paddedCuKVSeqLens   (prefill, CuTe DSL)
//   4*    | [B, 2, Hkv, Smax, D]             | HALF  | splitPagedKV out    (FP8/padding/vision)
//   5*    | [B, S, Hq, D]                    | FP8   | fp8Q                (CuTe DSL + FP8 prefill only)
//   6*    | [B, S] x 2                       | INT32 | blockBegin/blockEnd (vision / D512 padding prefill)
//   7*    | [(B*Hq+8) * 32*(D+2)]            | FP32  | CTM decode split-K scratch (decode)
//
//   * Slots 4-7 are conditionally allocated (dense K/V gather fallbacks /
//     CuTe DSL + FP8 KV cache / vision-block or D512 padding attention / CTM decode respectively).
//
// Total allocation is the sum of all conditional slots (safe upper bound).
size_t getAttentionWorkspaceSize(int64_t batchSize, int64_t seqLen, int64_t kvCacheCapacity, int32_t numQHeads,
    int32_t numKVHeads, int32_t headSize, bool useCuteDslFMHA, bool enableFp8KVCache, bool enableBlockRanges,
    size_t ctmDecodeWorkspaceElems, bool reserveDenseKV)
{
    size_t workspaceSize = 0;

    // CuQSeqLens for FMHA.
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize + 1}, DataType::kINT32);

    // Chunked prefill sequence metadata.
    workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize + 1}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize + 1}, DataType::kINT32);
    if (reserveDenseKV)
    {
        workspaceSize = accumulateWorkspaceSize(
            workspaceSize, rt::Coords{batchSize, 2, numKVHeads, kvCacheCapacity, headSize}, DataType::kHALF);
    }

    // Roped Q is written to a scratch tensor (always needed).
    workspaceSize
        = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize, seqLen, numQHeads, headSize}, DataType::kHALF);

    // Scratch K/V remain necessary for dense FP8 FMHA-v2 normal prefill; allocate unconditionally.
    workspaceSize
        = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize, seqLen, numKVHeads, headSize}, DataType::kHALF);

    workspaceSize
        = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize, seqLen, numKVHeads, headSize}, DataType::kHALF);

    // FP8 Q output: RoPE kernel writes FP8 Q to this workspace buffer (CuTe DSL FMHA path).
    if (useCuteDslFMHA && enableFp8KVCache)
    {
        workspaceSize = accumulateWorkspaceSize(
            workspaceSize, rt::Coords{batchSize, seqLen, numQHeads, headSize}, DataType::kFP8);
    }

    // Per-position intervals for vision-block and FMHA-v2 D512 padding prefill.
    if (enableBlockRanges)
    {
        workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize, seqLen}, DataType::kINT32);
        workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize, seqLen}, DataType::kINT32);
    }

    if (ctmDecodeWorkspaceElems > 0)
    {
        workspaceSize = accumulateWorkspaceSize(
            workspaceSize, rt::Coords{static_cast<int64_t>(ctmDecodeWorkspaceElems)}, DataType::kFLOAT);
    }

    return workspaceSize;
}

//! Phase 8 allocations of a layer that is not bounded-SWA-capable, in enqueueCompactImpl order with Np, Nd <= B and
//! T_p <= T_max: prefix sums, prefill Q and FP8 Q, the FMHA-v2 FP8 dense K/V and its page table, then the decode
//! rows' Q and CTM scratch. XQA scratch follows every layout and is added by the caller.
size_t getCompactWorkspaceSize(int64_t batchSize, int64_t numTokens, int64_t kvCacheCapacity, int64_t maxPagesPerSeq,
    int32_t numQHeads, int32_t numKVHeads, int32_t headSize, bool enableFp8KVCache, bool fmhaV2Prefill,
    size_t ctmDecodeWorkspaceElems)
{
    size_t workspaceSize = 0;
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize + 1}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize + 1}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {numTokens, numQHeads, headSize}, DataType::kHALF);
    if (enableFp8KVCache)
    {
        workspaceSize = accumulateWorkspaceSize(workspaceSize, {numTokens, numQHeads, headSize}, DataType::kFP8);
        if (fmhaV2Prefill)
        {
            workspaceSize = accumulateWorkspaceSize(
                workspaceSize, {2, batchSize, kvCacheCapacity, numKVHeads, headSize}, DataType::kHALF);
            workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize, 2, maxPagesPerSeq}, DataType::kINT32);
        }
    }
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize, numQHeads, headSize}, DataType::kHALF);
    if (ctmDecodeWorkspaceElems > 0)
    {
        workspaceSize = accumulateWorkspaceSize(
            workspaceSize, rt::Coords{static_cast<int64_t>(ctmDecodeWorkspaceElems)}, DataType::kFLOAT);
    }
    return workspaceSize;
}

//! Phase 8 prefill allocations of a bounded-SWA-capable layer (enqueueCompactImpl order), plus the decode rows' Q.
size_t getSwaCompactWorkspaceSize(int64_t batchSize, int64_t numTokens, int32_t slidingWindowSize, int32_t numQHeads,
    int32_t numKVHeads, int32_t headSize, int64_t maxPagesPerSeq)
{
    int64_t const assembledPages = std::min(
        batchSize * maxPagesPerSeq, kernel::swaCompactAssembledPages(batchSize, numTokens, slidingWindowSize));
    size_t workspaceSize = 0;
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize + 1}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize + 1}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {numTokens, numQHeads, headSize}, DataType::kHALF);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {numTokens, numKVHeads, headSize}, DataType::kHALF);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {numTokens, numKVHeads, headSize}, DataType::kHALF);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize, 2, maxPagesPerSeq}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, {2, assembledPages, rt::kTOKENS_PER_PAGE, numKVHeads, headSize}, DataType::kHALF);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {numTokens, numQHeads, headSize}, DataType::kHALF);
    return workspaceSize;
}

size_t getSwaKVCacheWorkspaceSize(int64_t batchSize, int64_t seqLen, int32_t slidingWindowSize, int32_t numQHeads,
    int32_t numKVHeads, int32_t headSize, bool enableVisionBlockAttention)
{
    size_t workspaceSize = 0;
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize + 1}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize + 1}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize + 1}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize, seqLen, numQHeads, headSize}, DataType::kHALF);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize, seqLen, numKVHeads, headSize}, DataType::kHALF);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize, seqLen, numKVHeads, headSize}, DataType::kHALF);
    workspaceSize = accumulateWorkspaceSize(workspaceSize,
        {batchSize, 2, numKVHeads, static_cast<int64_t>(slidingWindowSize) + seqLen, headSize}, DataType::kHALF);
    if (enableVisionBlockAttention)
    {
        workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize, seqLen}, DataType::kINT32);
        workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize, seqLen}, DataType::kINT32);
    }
    return workspaceSize;
}

bool isPagedPoolShape(Dims const& shape, int32_t numKVHeads, int32_t headSize, bool allowUnknownNumPages)
{
    if (shape.nbDims != 5)
    {
        return false;
    }
    bool const validNumPages = allowUnknownNumPages ? (shape.d[1] == -1 || shape.d[1] > 0) : shape.d[1] > 0;
    return shape.d[0] == 2 && validNumPages && shape.d[2] == rt::kTOKENS_PER_PAGE && shape.d[3] == numKVHeads
        && shape.d[4] == headSize;
}

bool isPagedPoolShape(rt::Coords const& shape, int32_t numKVHeads, int32_t headSize)
{
    return shape.getNumDims() == 5 && shape[0] == 2 && shape[2] == rt::kTOKENS_PER_PAGE && shape[3] == numKVHeads
        && shape[4] == headSize && shape[1] > 0;
}

bool isKVCacheDescriptor(PluginTensorDesc const& tensorDesc, bool enableFp8KVCache)
{
    bool const validType = enableFp8KVCache ? (tensorDesc.type == DataType::kFP8 || tensorDesc.type == DataType::kINT8)
                                            : tensorDesc.type == DataType::kHALF;
    return validType && tensorDesc.format == TensorFormat::kLINEAR;
}

bool isKVPageTableDescriptor(PluginTensorDesc const& tensorDesc)
{
    return tensorDesc.type == DataType::kINT32 && tensorDesc.format == TensorFormat::kLINEAR;
}

bool haveSameShape(Dims const& lhs, Dims const& rhs)
{
    if (lhs.nbDims != rhs.nbDims)
    {
        return false;
    }
    for (int32_t dimIdx = 0; dimIdx < lhs.nbDims; ++dimIdx)
    {
        if (lhs.d[dimIdx] != rhs.d[dimIdx])
        {
            return false;
        }
    }
    return true;
}

bool hasConcretePagedKVContract(Dims const& qkv, Dims const& sequenceLengths, Dims const& kvCacheInput,
    Dims const& kvCacheOutput, Dims const& kvPageTable, int32_t numKVHeads, int32_t headSize,
    bool allowSparseLogicalTable, bool allowEmptyQKV = false)
{
    bool const validExecutionShape = qkv.nbDims == 2 && (qkv.d[0] > 0 || (allowEmptyQKV && qkv.d[0] == 0))
        && sequenceLengths.nbDims == 1 && sequenceLengths.d[0] > 0;
    int32_t const batchSize = sequenceLengths.d[0];
    return validExecutionShape && isPagedPoolShape(kvCacheInput, numKVHeads, headSize, false)
        && haveSameShape(kvCacheInput, kvCacheOutput) && kvPageTable.nbDims == 3 && kvPageTable.d[0] == batchSize
        && kvPageTable.d[1] == 2 && kvPageTable.d[2] > 0
        && (allowSparseLogicalTable || kvCacheInput.d[1] >= kvPageTable.d[2]);
}

bool hasConcretePagedKVContract(PluginTensorDesc const* in, PluginTensorDesc const* out, int32_t numKVHeads,
    int32_t headSize, bool enableFp8KVCache, bool allowSparseLogicalTable)
{
    return isKVCacheDescriptor(in[kIN_KV_CACHE_IDX], enableFp8KVCache)
        && isKVCacheDescriptor(out[kOUT_KV_CACHE_IDX], enableFp8KVCache)
        && isKVPageTableDescriptor(in[kIN_KV_PAGE_TABLE_IDX])
        && hasConcretePagedKVContract(in[kIN_QKV_IDX].dims, in[kIN_QUERY_LENGTH_IDX].dims, in[kIN_KV_CACHE_IDX].dims,
            out[kOUT_KV_CACHE_IDX].dims, in[kIN_KV_PAGE_TABLE_IDX].dims, numKVHeads, headSize, allowSparseLogicalTable);
}

} // namespace

// Static class fields initialization
PluginFieldCollection AttentionPluginCreator::mFieldCollection{};
std::vector<PluginField> AttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AttentionPluginCreator);

// FMHA-v2 fallback paths consume separate FP16 K/V tensors, so gather logical pages from the pool into workspace.
// The gather follows the page table and dequantizes FP8 pools.
std::pair<rt::Tensor, rt::Tensor> AttentionPlugin::splitPagedKV(rt::Tensor const& poolTensor, int32_t const* pageTable,
    int32_t const* kvSeqLens, int32_t maxPagesPerSeq, std::byte*& workspacePtr, int32_t batchSize, int32_t numKVHeads,
    int32_t capPadded, int32_t headSize, int32_t seqLen, float kScale, float vScale, cudaStream_t stream)
{
    // The pool binding is [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim].
    check::check(isPagedPoolShape(poolTensor.getShape(), numKVHeads, headSize),
        "splitPagedKV requires kv_cache paged-pool contract [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim].");

    trt_edgellm::DataType const dtype = poolTensor.getDataType();
    // Split-K/V consumers read FP16. FP16 pools byte-copy; FP8 pools dequantize.
    // Any other pool dtype would be reinterpreted as half by the consumers, so reject it loudly.
    bool const isFp8KV = dtype == trt_edgellm::DataType::kFP8 || dtype == trt_edgellm::DataType::kINT8;
    bool const validType = dtype == trt_edgellm::DataType::kHALF || isFp8KV;
    check::check(validType,
        "splitPagedKV: separate K/V consumers require an FP16 or FP8 KV pool; other dtypes would be "
        "reinterpreted as FP16 by the split-KV consumers.");
    size_t const elemSize = rt::utils::getTypeSize(dtype);

    // seqLen == 0 gathers full capacity; seqLen > 0 gathers only the first seqLen tokens (compact),
    // so downstream kernels can derive the batch stride from the output's S dimension. The gather
    // kernel uses this value as both the destination stride and the token bound (zero-filling
    // between a slot's live length and the bound).
    int32_t const outSeqDim = (seqLen > 0) ? seqLen : capPadded;

    // Split-KV workspace is ALWAYS FP16 (the consumer dtype), independent of the pool dtype.
    // kTensor fills the first kTensorElems elements of the workspace; vTensor starts right after.
    size_t const kTensorElems = static_cast<size_t>(batchSize) * outSeqDim * numKVHeads * headSize;
    rt::Tensor kvWorkspaceTensor
        = assignTensorFromWorkspace(workspacePtr, {batchSize, 2, numKVHeads, outSeqDim, headSize}, DataType::kHALF);
    auto* ptr = static_cast<std::byte*>(kvWorkspaceTensor.rawPointer());
    rt::Tensor kTensor(ptr, rt::Coords{batchSize, outSeqDim, numKVHeads, headSize}, rt::DeviceType::kGPU,
        trt_edgellm::DataType::kHALF);
    rt::Tensor vTensor(ptr + kTensorElems * sizeof(half), rt::Coords{batchSize, outSeqDim, numKVHeads, headSize},
        rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);

    kernel::gatherPagedKVToSplit(poolTensor.rawPointer(), kTensor.rawPointer(), vTensor.rawPointer(), pageTable,
        kvSeqLens, maxPagesPerSeq, batchSize, outSeqDim, numKVHeads, headSize, elemSize, isFp8KV, kScale, vScale,
        stream);

    return std::make_pair(std::move(kTensor), std::move(vTensor));
}

bool AttentionPlugin::canUseCuteDslBidirectionalForPrefill() const noexcept
{
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
    return mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL && mCanImplementFMHA
        && mCanImplementCuteDslBidirectionalFMHA && !mEnableFp8KVCache;
#else
    return false;
#endif
}

void AttentionPlugin::enforceVisionBlockKernelSupport() const
{
    bool const canImplementXQA = canCompileXQAJitKernel();
    ELLM_CHECK(mUseFMHAV2VisionBlockFMHA || canUseCuteDslBidirectionalForPrefill(),
        "AttentionPlugin: vision-block prefill requires a supported FMHA-v2 or paged CuTe DSL "
        "bidirectional kernel for headSize="
            + std::to_string(mHeadSize) + " on SM" + std::to_string(mSMVersion) + "; none is available.");
    ELLM_CHECK(canImplementXQA,
        "AttentionPlugin: vision-block decode requires XQA decode kernels for Hq=" + std::to_string(mNumQHeads)
            + ", Hkv=" + std::to_string(mNumKVHeads) + ", headSize=" + std::to_string(mHeadSize) + " on SM"
            + std::to_string(mSMVersion) + ", which are not available.");
}

AttentionPlugin::AttentionPlugin(std::string const& name, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t enableTreeAttention, int32_t enableFp8KVCache, int32_t enableVisionBlockAttention,
    int32_t enableContextMaskSelector, bool supportsBoundedKVCache, int32_t slidingWindowSize,
    std::vector<float> const& qkvScales, std::optional<float> attentionScale, int32_t tpSize, int32_t externalKVPrefix,
    int32_t fp8KVPrefix)
    : mLayerName(name)
    , mNumQHeads(numQHeads)
    , mNumKVHeads(numKVHeads)
    , mHeadSize(headSize)
    , mAttentionScale(resolveAttentionScale(attentionScale, headSize))
    , mEnableTreeAttention(enableTreeAttention)
    , mEnableVisionBlockAttention(enableVisionBlockAttention)
    , mExternalKVPrefix(externalKVPrefix)
    , mFp8KVPrefix(fp8KVPrefix)
    , mEnableFp8KVCache(enableFp8KVCache)
    , mEnableContextMaskSelector(enableContextMaskSelector)
    , mQkvScales((enableFp8KVCache || fp8KVPrefix) ? qkvScales : std::vector<float>{1.f, 1.f, 1.f})
    , mSlidingWindowSize(slidingWindowSize)
    , mSupportsBoundedKVCache(supportsBoundedKVCache)
    , mTpSize(tpSize)
{
    ELLM_CHECK(!mExternalKVPrefix
            || (!mEnableFp8KVCache && mHeadSize == 128 && mDataType == DataType::kHALF && !mEnableTreeAttention
                && !mEnableVisionBlockAttention && !mEnableKVShared && !mEnableQKNorm && !mSupportsBoundedKVCache
                && mSlidingWindowSize <= 0 && mEnableContextMaskSelector && !mEnableAttentionSink
                && mSkipSoftmaxPrefillInput == 0 && mSkipSoftmaxDecodeInput == 0),
        "Separate-prefix attention requires FP16 action QKV, D128, and non-causal full attention.");
    ELLM_CHECK(
        !mFp8KVPrefix || (mExternalKVPrefix && mQkvScales.size() == 3), "FP8 prefix requires calibrated K/V scales.");
    // The fused-norm warp reduction covers at most 32 lanes (headDim / vec width 8)
    // => no fused-norm kernel above head_size 256; fail at construction.
    ELLM_CHECK(!mEnableQKNorm || mHeadSize <= 256,
        "enable_qk_norm requires head_size <= 256 (no fused-norm kernel for larger heads).");
    ELLM_CHECK(!(mEnableQKNorm && mEnableKVShared),
        "enable_qk_norm with a shared-KV (Q-only) layer is not supported: the Q-only path has "
        "no fused-norm kernel.");
    ELLM_CHECK(!(mEnableTreeAttention && mEnableVisionBlockAttention),
        "Tree attention and vision block attention are mutually exclusive.");
    ELLM_CHECK(!mEnableAttentionSink || (!mEnableFp8KVCache && !mEnableVisionBlockAttention),
        "Learned attention sinks require an FP16 KV cache without vision masks.");
    ELLM_CHECK(!mEnableFp8KVCache || mQkvScales.size() == 3,
        "FP8 KV cache enabled but qkv_scales has "
            + std::to_string(mQkvScales.size()) + " elements (expected 3). "
            "Re-export the model to include QKV scales [q, k, v].");
    ELLM_CHECK(!mSupportsBoundedKVCache
            || (mSlidingWindowSize > 0 && !mEnableTreeAttention && !mEnableQKNorm && !mEnableContextMaskSelector),
        "Bounded KV cache capability requires vanilla sliding-window attention.");

    mSMVersion = getSMVersion();
    mCtmDecodeOnly = decodesOnCtmOnly(mSMVersion, mTpSize);
    applyThorSMRenumberWAR(mSMVersion);

    FMHAKernelSelection const fmhaSelection = selectFMHAKernels(mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion,
        mDataType, mSlidingWindowSize > 0, mEnableFp8KVCache, /*allowPagedBlackwell=*/!mSupportsBoundedKVCache,
        /*requirePagedRagged=*/mEnableAttentionSink != 0);
    mContextFMHABackend = fmhaSelection.backend;
    mCanImplementFMHA = fmhaSelection.canImplement;
    // Tree-attention layers apply sinks in speculative tree decode (CTM or XQA); enqueue rejects sinks in prefill.
    ELLM_CHECK(!mEnableAttentionSink || mEnableTreeAttention || mCanImplementFMHA,
        "Learned attention sinks require an FMHA-v2 ragged paged kernel for this head shape, dtype, CUDA, and SM.");
    // Blackwell decodes on the CTM kernels only: a layer they cannot serve is a construction error, not an XQA
    // fallback.
    ELLM_CHECK(!mCtmDecodeOnly || (useCtmDecode() && (mEnableTreeAttention == 0 || useCtmTreeDecode())),
        "AttentionPlugin: Blackwell decode requires a CTM kernel, none for Hq=" + std::to_string(mNumQHeads)
            + " Hkv=" + std::to_string(mNumKVHeads) + " headSize=" + std::to_string(mHeadSize)
            + " fp8KV=" + std::to_string(mEnableFp8KVCache) + " window=" + std::to_string(mSlidingWindowSize)
            + " sinks=" + std::to_string(mEnableAttentionSink) + " tree=" + std::to_string(mEnableTreeAttention) + ".");
    LOG_DEBUG("AttentionPlugin FMHA backend: %d, sliding_window: %s", static_cast<int32_t>(mContextFMHABackend),
        mSlidingWindowSize > 0 ? std::to_string(mSlidingWindowSize).c_str() : "disabled");

#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
    mCanImplementCuteDslBidirectionalFMHA = mEnableVisionBlockAttention && mHeadSize == 512
        && mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL && mCanImplementFMHA;
#endif

    ELLM_CHECK(!mSupportsBoundedKVCache
            || (mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_V2 && mCanImplementFMHA
                && canCompileXQAJitKernel()),
        "Bounded SWA cache requires FMHA-v2 prefill and paged XQA decode kernels.");
    ELLM_CHECK(!(mHeadSize == 512 && mSlidingWindowSize > 0 && !mEnableVisionBlockAttention) || mCanImplementFMHA,
        "D512 sliding-window prefill requires the CuTe DSL paged FMHA kernel.");

    // Kernel selection priority for prefill and decode:
    //   1. Vision-block attention        — paged CuTe DSL d512 bidirectional mask
    //      or FMHA-v2 CuTe DSL vision-block; XQA decode.
    //   2. FMHA (prefill) + XQA (decode) — common paged path, including
    //      full-causal FP16 D512 normal/chunked/shared prefill.
    //   3. XQA (decode) only             — prefill unsupported for this head size.
    //   4. None                          — fatal, cannot serve this configuration.

    if (mEnableContextMaskSelector && mCanImplementFMHA)
    {
        mCanImplementPaddingFMHA
            = canImplementPaddingFMHA(mContextFMHABackend, mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion, mDataType);
    }

    if (mEnableVisionBlockAttention)
    {
        // The FMHA-v2 split-K/V vision-block kernels cover d256 sliding layers and the d512 global
        // layers; the latter pass an unbounded window through the same runtime argument.
        mUseFMHAV2VisionBlockFMHA = CuteDslFMHAV2Runner::canImplement(
            mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion, fromTrt(mDataType), CuteDslFMHAV2MaskType::kVISION_BLOCK);
        enforceVisionBlockKernelSupport();

        LOG_INFO(
            "AttentionPlugin: vision-block attention (headSize=%d, Hq=%d, Hkv=%d, window=%d) — prefill via %s, "
            "decode via XQA.",
            mHeadSize, mNumQHeads, mNumKVHeads, mSlidingWindowSize,
            canUseCuteDslBidirectionalForPrefill() ? "paged CuTe DSL d512 bidirectional mask"
                                                   : "FMHA-v2 CuTe DSL vision-block");
    }
    bool const canImplementXQA = canCompileXQAJitKernel();
    if (!mEnableVisionBlockAttention && !mCanImplementFMHA && !canImplementXQA)
    {
        LOG_ERROR("Cannot implement AttentionPlugin configuration. SM: %d, HeadSize: %d, NumQHeads: %d, NumKVHeads: %d",
            mSMVersion, mHeadSize, mNumQHeads, mNumKVHeads);
        throw std::runtime_error("Cannot implement the AttentionPlugin configuration.");
    }
    else if (!mEnableVisionBlockAttention && !mCanImplementFMHA)
    {
        LOG_WARNING("AttentionPlugin: no prefill kernel for headSize=%d; only decode (XQA) is supported.", mHeadSize);
    }
}

AttentionPlugin::AttentionPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
    , mNumQHeads(parsePluginScalarField<int32_t>("num_q_heads", fc).value_or(0))
    , mNumKVHeads(parsePluginScalarField<int32_t>("num_kv_heads", fc).value_or(0))
    , mHeadSize(parsePluginScalarField<int32_t>("head_size", fc).value_or(0))
    , mAttentionScale(resolveAttentionScale(parsePluginScalarField<float>("attention_scale", fc), mHeadSize))
    , mEnableTreeAttention(parsePluginScalarField<int32_t>("enable_tree_attention", fc).value_or(0))
    , mEnableQKNorm(parsePluginScalarField<int32_t>("enable_qk_norm", fc).value_or(0))
    , mQKNormPostRope(parsePluginScalarField<int32_t>("qk_norm_post_rope", fc).value_or(0))
    , mEnableKVShared(parsePluginScalarField<int32_t>("enable_kv_shared", fc).value_or(0))
    , mEnableContiguousQuerySwa(parsePluginScalarField<int32_t>("enable_contiguous_query_swa", fc).value_or(0))
    , mEnableAttentionSink(parsePluginScalarField<int32_t>("enable_attention_sink", fc).value_or(0))
    , mExternalKVPrefix(parsePluginScalarField<int32_t>("external_kv_prefix", fc).value_or(0))
    , mFp8KVPrefix(parsePluginScalarField<int32_t>("fp8_kv_prefix", fc).value_or(0))
    , mEnableFp8KVCache(parsePluginScalarField<int32_t>("enable_fp8_kv_cache", fc).value_or(0))
    , mSlidingWindowSize(parsePluginScalarField<int32_t>("sliding_window_size", fc).value_or(-1))
    , mSupportsBoundedKVCache(parsePluginScalarField<int32_t>("supports_bounded_kv_cache", fc).value_or(0) != 0)
    , mTpSize(parsePluginScalarField<int32_t>("tp_size", fc).value_or(1))
    , mSkipSoftmaxPrefillInput(parsePluginScalarField<int32_t>("skip_softmax_prefill_input", fc).value_or(0))
    , mSkipSoftmaxDecodeInput(parsePluginScalarField<int32_t>("skip_softmax_decode_input", fc).value_or(0))
{
    mEnableVisionBlockAttention = parsePluginScalarField<int32_t>("enable_vision_block_attention", fc).value_or(0);
    mEnableContextMaskSelector = parsePluginScalarField<int32_t>("enable_context_mask_selector", fc).value_or(0);
    mCtmSemaphoreCapacityElems = parsePluginScalarField<int32_t>("ctm_semaphore_capacity", fc).value_or(0);
    // The fused-norm warp reduction covers at most 32 lanes (headDim / vec width 8)
    // => no fused-norm kernel above head_size 256; fail at construction.
    ELLM_CHECK(!mEnableQKNorm || mHeadSize <= 256,
        "enable_qk_norm requires head_size <= 256 (no fused-norm kernel for larger heads).");
    ELLM_CHECK(!(mEnableQKNorm && mEnableKVShared),
        "enable_qk_norm with a shared-KV (Q-only) layer is not supported: the Q-only path has "
        "no fused-norm kernel.");
    ELLM_CHECK(!mQKNormPostRope || mEnableQKNorm, "qk_norm_post_rope requires enable_qk_norm.");

    ELLM_CHECK(!(mEnableTreeAttention && mEnableVisionBlockAttention),
        "Tree attention and vision block attention are mutually exclusive.");
    ELLM_CHECK(!mEnableAttentionSink || (!mEnableFp8KVCache && !mEnableVisionBlockAttention),
        "Learned attention sinks require an FP16 KV cache without vision masks.");
    ELLM_CHECK(!mSupportsBoundedKVCache
            || (mSlidingWindowSize > 0 && !mEnableTreeAttention && !mEnableQKNorm && !mEnableContextMaskSelector),
        "Bounded KV cache capability requires vanilla sliding-window attention.");

    // Parse qkv_scales float array
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        if (std::string("qkv_scales") == fc->fields[i].name)
        {
            auto const* data = static_cast<float const*>(fc->fields[i].data);
            mQkvScales.assign(data, data + fc->fields[i].length);
            break;
        }
    }
    mXqaJitBlob = parsePluginBytesField(kXQA_JIT_KERNELS_FIELD, fc);
    if (!mXqaJitBlob.empty())
    {
        mXqaJitKernels = deserializeXQAJitKernels(mXqaJitBlob.data(), mXqaJitBlob.size());
    }

    // The q/k gamma weights are not plugin fields — they arrive as optional engine-weight
    // constant inputs (see the input-layout block at the top of this file).
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        std::string const name = fc->fields[i].name;
        if (name == "rms_norm_eps")
        {
            mRmsNormEps = *static_cast<float const*>(fc->fields[i].data);
        }
    }

    if (!mEnableFp8KVCache && !mFp8KVPrefix)
    {
        mQkvScales = {1.f, 1.f, 1.f};
    }
    else
    {
        ELLM_CHECK(mQkvScales.size() == 3,
            "FP8 KV cache enabled but qkv_scales missing or incomplete "
            "in plugin fields (expected 3). Re-export the model with QKV scales [q, k, v].");
    }

    mSMVersion = getSMVersion();
    mCtmDecodeOnly = decodesOnCtmOnly(mSMVersion, mTpSize);
    ELLM_CHECK(!mExternalKVPrefix
            || (!mEnableFp8KVCache && mHeadSize == 128 && mDataType == DataType::kHALF && !mEnableTreeAttention
                && !mEnableVisionBlockAttention && !mEnableKVShared && !mEnableQKNorm && !mSupportsBoundedKVCache
                && mSlidingWindowSize <= 0 && mEnableContextMaskSelector && !mEnableAttentionSink
                && mSkipSoftmaxPrefillInput == 0 && mSkipSoftmaxDecodeInput == 0),
        "Separate-prefix attention requires FP16 action QKV, D128, and non-causal full attention.");
    ELLM_CHECK(
        !mFp8KVPrefix || (mExternalKVPrefix && mQkvScales.size() == 3), "FP8 prefix requires calibrated K/V scales.");
    applyThorSMRenumberWAR(mSMVersion);

    FMHAKernelSelection const fmhaSelection = selectFMHAKernels(mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion,
        mDataType, mSlidingWindowSize > 0, mEnableFp8KVCache,
        /*allowPagedBlackwell=*/!mSupportsBoundedKVCache, /*requirePagedRagged=*/mEnableAttentionSink != 0);
    mContextFMHABackend = fmhaSelection.backend;
    mCanImplementFMHA = fmhaSelection.canImplement;
    // Tree-attention layers apply sinks in speculative tree decode (CTM or XQA); enqueue rejects sinks in prefill.
    ELLM_CHECK(!mEnableAttentionSink || mEnableTreeAttention || mCanImplementFMHA,
        "Learned attention sinks require an FMHA-v2 ragged paged kernel for this head shape, dtype, CUDA, and SM.");
    // Blackwell decodes on the CTM kernels only: a layer they cannot serve is a construction error, not an XQA
    // fallback.
    ELLM_CHECK(!mCtmDecodeOnly || (useCtmDecode() && (mEnableTreeAttention == 0 || useCtmTreeDecode())),
        "AttentionPlugin: Blackwell decode requires a CTM kernel, none for Hq=" + std::to_string(mNumQHeads)
            + " Hkv=" + std::to_string(mNumKVHeads) + " headSize=" + std::to_string(mHeadSize)
            + " fp8KV=" + std::to_string(mEnableFp8KVCache) + " window=" + std::to_string(mSlidingWindowSize)
            + " sinks=" + std::to_string(mEnableAttentionSink) + " tree=" + std::to_string(mEnableTreeAttention) + ".");
    LOG_DEBUG("AttentionPlugin FMHA backend: %d", static_cast<int32_t>(mContextFMHABackend));

#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
    mCanImplementCuteDslBidirectionalFMHA = mEnableVisionBlockAttention && mHeadSize == 512
        && mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL && mCanImplementFMHA;
#endif

    ELLM_CHECK(!mSupportsBoundedKVCache
            || (mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_V2 && mCanImplementFMHA
                && canCompileXQAJitKernel()),
        "Bounded SWA cache requires FMHA-v2 prefill and paged XQA decode kernels.");
    ELLM_CHECK(!(mHeadSize == 512 && mSlidingWindowSize > 0 && !mEnableVisionBlockAttention) || mCanImplementFMHA,
        "D512 sliding-window prefill requires the CuTe DSL paged FMHA kernel.");

    if (mEnableContextMaskSelector && mCanImplementFMHA)
    {
        mCanImplementPaddingFMHA
            = canImplementPaddingFMHA(mContextFMHABackend, mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion, mDataType);
    }

    // Vision prefill production path: FMHA-v2 CuTe DSL (d256 sliding and d512 global layers).
    if (mEnableVisionBlockAttention)
    {
        mUseFMHAV2VisionBlockFMHA = CuteDslFMHAV2Runner::canImplement(
            mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion, fromTrt(mDataType), CuteDslFMHAV2MaskType::kVISION_BLOCK);
        enforceVisionBlockKernelSupport();
    }
}

AttentionPlugin::~AttentionPlugin()
{
    releaseCtmSemaphores();
}

AttentionLayerTraits AttentionPlugin::layerTraits(bool qOnlyInput) const noexcept
{
    AttentionLayerTraits traits;
    traits.numQHeads = mNumQHeads;
    traits.numKVHeads = mNumKVHeads;
    traits.headSize = mHeadSize;
    traits.smVersion = mSMVersion;
    traits.tpSize = mTpSize;
    traits.dataType = fromTrt(mDataType);
    traits.fp8KVCache = mEnableFp8KVCache != 0;
    traits.slidingWindow = mSlidingWindowSize > 0;
    traits.supportsBoundedKVCache = mSupportsBoundedKVCache;
    traits.kvShared = mEnableKVShared != 0;
    traits.qOnlyInput = qOnlyInput;
    traits.treeAttention = mEnableTreeAttention != 0;
    traits.visionBlockAttention = mEnableVisionBlockAttention != 0;
    traits.attentionSink = mEnableAttentionSink != 0;
    traits.contextMaskSelector = mEnableContextMaskSelector != 0;
    return traits;
}

std::string AttentionPlugin::describeLayer() const
{
    std::string shape = attentionLayerShape(layerTraits());
    if (mSlidingWindowSize > 0)
    {
        shape += ", sliding window " + std::to_string(mSlidingWindowSize);
    }
    if (mEnableTreeAttention != 0)
    {
        shape += ", tree attention";
    }
    if (mEnableVisionBlockAttention != 0)
    {
        shape += ", vision-block attention";
    }
    if (mEnableAttentionSink != 0)
    {
        shape += ", attention sinks";
    }
    if (mEnableContextMaskSelector != 0)
    {
        shape += ", context mask selector";
    }
    return describePluginLayer("AttentionPlugin", mLayerName, shape);
}

AttentionInputLayout AttentionPlugin::inputLayout() const noexcept
{
    AttentionInputFlags flags;
    flags.qkNorm = mEnableQKNorm != 0;
    flags.contextMaskSelector = mEnableContextMaskSelector != 0;
    flags.treeAttention = mEnableTreeAttention != 0;
    flags.visionBlock = mEnableVisionBlockAttention != 0;
    flags.skipSoftmaxPrefillScale = mSkipSoftmaxPrefillInput != 0;
    flags.skipSoftmaxDecodeScale = mSkipSoftmaxDecodeInput != 0;
    flags.boundedKVCache = mSupportsBoundedKVCache;
    flags.attentionSink = mEnableAttentionSink != 0;
    flags.externalKVPrefix = mExternalKVPrefix != 0;
    return resolveAttentionInputLayout(flags);
}

bool AttentionPlugin::useCtmDecode() const noexcept
{
    return canUseCtmDecode(layerTraits());
}

bool AttentionPlugin::useCtmTreeDecode() const noexcept
{
#if defined(CUTE_DSL_FMHA_DECODE_CTM_ENABLED)
    // CTM kernels exist only for the CTM-only SMs; see canUseCtmDecode().
    return isCtmOnlySM(mSMVersion) && !ctmDecodeDisabledByEnv(mSMVersion) && mTpSize == 1 && mEnableTreeAttention != 0
        && mDataType == nvinfer1::DataType::kHALF
        && CtmDecodeAttentionRunner::supportedTree(mNumQHeads, mNumKVHeads, mHeadSize, mEnableFp8KVCache != 0,
            mSlidingWindowSize > 0, mEnableContiguousQuerySwa != 0);
#else
    return false;
#endif
}

size_t AttentionPlugin::ctmSemaphoreElems(int32_t batchSize, int32_t qSeqLen) const noexcept
{
    if (batchSize <= 0 || mNumKVHeads <= 0 || mNumQHeads % mNumKVHeads != 0)
    {
        return 0;
    }
    size_t elems = CtmDecodeAttentionRunner::semaphoreElems(batchSize, mNumKVHeads);
    if (mEnableTreeAttention != 0)
    {
        elems = std::max(elems,
            CtmDecodeAttentionRunner::treeSemaphoreElems(batchSize, std::max(1, qSeqLen), mNumQHeads, mNumKVHeads));
    }
    return elems;
}

bool AttentionPlugin::allocateCtmSemaphores() noexcept
{
    if (mCtmSemaphores != nullptr)
    {
        return true;
    }
    if (mCtmSemaphoreCapacityElems <= 0)
    {
        LOG_ERROR("AttentionPlugin: engine predates the ctm_semaphore_capacity plugin field; rebuild the engine.");
        return false;
    }
    void* buffer{nullptr};
    size_t const bytes = static_cast<size_t>(mCtmSemaphoreCapacityElems) * sizeof(int32_t);
    cudaError_t const allocError = cudaMalloc(&buffer, bytes);
    if (allocError != cudaSuccess)
    {
        LOG_ERROR("AttentionPlugin: cudaMalloc for CTM decode semaphores failed: %s", cudaGetErrorString(allocError));
        return false;
    }
    // The kernel returns every counter to zero itself; only the fresh allocation needs clearing.
    cudaError_t const memsetError = cudaMemset(buffer, 0, bytes);
    if (memsetError != cudaSuccess)
    {
        LOG_ERROR("AttentionPlugin: cudaMemset for CTM decode semaphores failed: %s", cudaGetErrorString(memsetError));
        cudaFree(buffer);
        return false;
    }
    mCtmSemaphores = static_cast<int32_t*>(buffer);
    mCtmSemaphoreElems = static_cast<size_t>(mCtmSemaphoreCapacityElems);
    return true;
}

void AttentionPlugin::releaseCtmSemaphores() noexcept
{
    if (mCtmSemaphores != nullptr)
    {
        cudaError_t const freeError = cudaFree(mCtmSemaphores);
        if (freeError != cudaSuccess)
        {
            LOG_ERROR("AttentionPlugin: cudaFree for CTM decode semaphores failed: %s", cudaGetErrorString(freeError));
        }
    }
    mCtmSemaphores = nullptr;
    mCtmSemaphoreElems = 0;
}

XQAJitKey AttentionPlugin::getXQAJitKey() const noexcept
{
    XQAJitKey key{};
    key.sm = mSMVersion;
    key.dataType = fromTrt(mDataType);
    key.kvDataType = fromTrt(selectKvCacheDataType(mEnableFp8KVCache));
    key.headSize = mHeadSize;
    key.qHeadsPerKv = mNumKVHeads > 0 ? mNumQHeads / mNumKVHeads : 0;
    key.tokensPerPage = rt::kTOKENS_PER_PAGE;
    key.slidingWindow = mSlidingWindowSize > 0;
    key.specDecode = static_cast<bool>(mEnableTreeAttention);
    key.contiguousQuerySwa = mEnableContiguousQuerySwa && key.specDecode && key.slidingWindow;
    return key;
}

bool AttentionPlugin::canCompileXQAJitKernel() const noexcept
{
    return canUseXqaDecode(layerTraits());
}

ContextFMHABackend AttentionPlugin::selectCompactPrefillBackend() const noexcept
{
    switch (selectCompactPrefillKernel(layerTraits()))
    {
    case CompactPrefillKernel::kCUTE_DSL_FMHA_BLACKWELL: return ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL;
    case CompactPrefillKernel::kCUTE_DSL_FMHA_V2: return ContextFMHABackend::kCUTE_DSL_FMHA_V2;
    case CompactPrefillKernel::kNONE: return ContextFMHABackend::kNONE;
    }
    return ContextFMHABackend::kNONE;
}

bool AttentionPlugin::supportsCompactPhase(int64_t qkvWidth) const noexcept
{
    bool const qOnly = qkvWidth <= 0 || qkvWidth == static_cast<int64_t>(mNumQHeads) * mHeadSize;
    return !mExternalKVPrefix && attentionCompactSupport(layerTraits(qOnly)).supported;
}

void AttentionPlugin::compileXQAJitKernelForBuild()
{
    if (mExternalKVPrefix)
    {
        return; // Two-pool action attention never executes a decode kernel.
    }
    // A layer with no XQA decode kernel is not fatal on its own: the prefill
    // path may still be served by FMHA, and the constructor already rejects
    // the case where no backend at all can implement the layer.
    if (mCtmDecodeOnly)
    {
        return; // Blackwell decodes on the CTM kernels; the engine carries no XQA blobs.
    }
    if (!canCompileXQAJitKernel())
    {
        LOG_INFO("AttentionPlugin: no XQA decode kernel for Hq=%d, Hkv=%d, headSize=%d on SM%d; skipping XQA JIT.",
            mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion);
        return;
    }

    auto compileAndLoad = [](XQAJitKey const& jitKey) -> XQAJitKernel {
        XQAJitKernel kernel = compileXQAKernel(jitKey);
        ELLM_CHECK(!kernel.cubin.empty(), "NVRTC returned an empty XQA cubin.");
        bool const loaded
            = DecoderXQARunner::loadDecodeXQAKernelFromCubin(jitKey, kernel.cubin.data(), kernel.cubin.size());
        ELLM_CHECK(loaded, "Failed to load the NVRTC-compiled XQA kernel.");
        return kernel;
    };

    XQAJitKey const key = getXQAJitKey();
    mXqaJitKernels.clear();

    // Vanilla decode is always needed: tree-attention engines still execute it
    // for single-token steps and fallback requests.
    XQAJitKey vanillaKey = key;
    vanillaKey.specDecode = false;
    vanillaKey.contiguousQuerySwa = false;
    mXqaJitKernels.push_back(compileAndLoad(vanillaKey));

    // Only tree-attention engines need a second, distinct spec-decode kernel.
    // Emitting one entry per distinct key (rather than a fixed pair of slots)
    // is what keeps a non-tree layer from serializing the same cubin twice.
    if (key.specDecode)
    {
        mXqaJitKernels.push_back(compileAndLoad(key));
    }

    mXqaJitBlob = serializeXQAJitKernels(mXqaJitKernels);
}

void AttentionPlugin::loadSerializedXQAJitKernel()
{
    ELLM_CHECK(!mXqaJitKernels.empty(), "Serialized XQA JIT kernel field is empty.");

    for (XQAJitKernel const& kernel : mXqaJitKernels)
    {
        ELLM_CHECK(kernel.key.sm == mSMVersion,
            format::fmtstr(
                "Serialized XQA JIT cubin targets SM%d, but runtime GPU is SM%d.", kernel.key.sm, mSMVersion));
        // Register under the key the kernel was compiled for, which travels in
        // the blob. Recomputing it from live plugin fields would silently bind
        // the cubin to the wrong slot if any contributing field changed.
        bool const loaded
            = DecoderXQARunner::loadDecodeXQAKernelFromCubin(kernel.key, kernel.cubin.data(), kernel.cubin.size());
        ELLM_CHECK(loaded, "Failed to load serialized XQA JIT cubin.");
    }
}

// ---------------------------------------------------------------------------
// IPluginV3
// ---------------------------------------------------------------------------

IPluginCapability* AttentionPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuildV2*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

IPluginV3* AttentionPlugin::clone() noexcept
{
    try
    {
        // TODO: every new plugin attribute must be added here by hand (ctor arg or
        // manual assignment), and a forgotten entry fails silently — the clone
        // constructs fine and the feature is just off at runtime (TensorRT executes
        // the clone, not the creator-made instance). If this ever bites, switch to
        // a memberwise copy (protected `AttentionPlugin(AttentionPlugin const&) =
        // default` + clear the serialization scratch), which clones future fields
        // by construction.
        auto p = std::make_unique<AttentionPlugin>(mLayerName, mNumQHeads, mNumKVHeads, mHeadSize, mEnableTreeAttention,
            mEnableFp8KVCache, mEnableVisionBlockAttention, mEnableContextMaskSelector, mSupportsBoundedKVCache,
            mSlidingWindowSize, mQkvScales, mAttentionScale, mTpSize, mExternalKVPrefix, mFp8KVPrefix);
        p->mEnableQKNorm = mEnableQKNorm;
        p->mQKNormPostRope = mQKNormPostRope;
        p->mEnableKVShared = mEnableKVShared;
        p->mEnableContiguousQuerySwa = mEnableContiguousQuerySwa;
        p->mEnableAttentionSink = mEnableAttentionSink;
        p->mRmsNormEps = mRmsNormEps;
        p->mSkipSoftmaxPrefillInput = mSkipSoftmaxPrefillInput;
        p->mSkipSoftmaxDecodeInput = mSkipSoftmaxDecodeInput;
        p->mCtmSemaphoreCapacityElems = mCtmSemaphoreCapacityElems;
        p->mXqaJitKernels = mXqaJitKernels;
        p->mXqaJitBlob = mXqaJitBlob;
        if (!p->mXqaJitKernels.empty())
        {
            // Registration is keyed and idempotent, so re-running it on a clone
            // of an already-registered plugin is a cache hit, not a reload.
            p->loadSerializedXQAJitKernel();
        }
        p->setPluginNamespace(mNamespace.c_str());
        return p.release();
    }
    catch (...)
    {
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// IPluginV3OneCore — metadata
// ---------------------------------------------------------------------------

char const* AttentionPlugin::getPluginName() const noexcept
{
    return kATTENTION_PLUGIN_NAME;
}

char const* AttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void AttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* AttentionPlugin::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

// ---------------------------------------------------------------------------
// IPluginV3OneBuild — shape / format
// ---------------------------------------------------------------------------

int32_t AttentionPlugin::getNbOutputs() const noexcept
{
    // At both context and generation phase, output attention result and kv-cache.
    return 2;
}

int32_t AttentionPlugin::getOutputDataTypes(DataType* outputTypes, [[maybe_unused]] int32_t nbOutputs,
    DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    try
    {
        assert(nbOutputs == kNUM_REQUIRED_OUTPUTS);
        // Output[0] (attention): always FP16 (follows Q input dtype).
        // Output[1] (KV cache) follows KV input dtype (HALF or FP8).
        outputTypes[kOUT_ATTENTION_IDX] = inputTypes[kIN_QKV_IDX];
        outputTypes[kOUT_KV_CACHE_IDX] = inputTypes[kIN_KV_CACHE_IDX];
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t AttentionPlugin::getOutputShapes(DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs,
    [[maybe_unused]] int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        assert(nbOutputs == kNUM_REQUIRED_OUTPUTS);
        outputs[kOUT_ATTENTION_IDX].nbDims = 3;
        outputs[kOUT_ATTENTION_IDX].d[0] = inputs[kIN_QKV_IDX].d[0];
        outputs[kOUT_ATTENTION_IDX].d[1] = exprBuilder.constant(mNumQHeads);
        outputs[kOUT_ATTENTION_IDX].d[2] = exprBuilder.constant(mHeadSize);

        // Output[1] is the KV cache, same shape as the input KV cache: the paged pool
        // [2, numPages, 128, Hkv, D] (in-place aliased). numPages (dim 1) is dynamic.
        outputs[kOUT_KV_CACHE_IDX].nbDims = 5;
        outputs[kOUT_KV_CACHE_IDX].d[0] = inputs[kIN_KV_CACHE_IDX].d[0];
        outputs[kOUT_KV_CACHE_IDX].d[1] = inputs[kIN_KV_CACHE_IDX].d[1];
        outputs[kOUT_KV_CACHE_IDX].d[2] = inputs[kIN_KV_CACHE_IDX].d[2];
        outputs[kOUT_KV_CACHE_IDX].d[3] = inputs[kIN_KV_CACHE_IDX].d[3];
        outputs[kOUT_KV_CACHE_IDX].d[4] = inputs[kIN_KV_CACHE_IDX].d[4];

        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

bool AttentionPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // Support context/generation phase inputs:
    //      packed QKV tensor (linear FP16) with shape [T_exec, (Hq + 2*Hkv) * D]
    //      KV-cache tensor (linear FP16/FP8): paged pool [2, numPages, kTOKENS_PER_PAGE, Hkv, D].
    //      (numPages can select the bounded or full pool for an SWA-capable engine.)
    //      Real context length: [B] (a vector of scalars) with type int32_t.
    //      RoPE cos/sin cache: [B or 1, Smax, D] (a tensor of scalars) with type float.
    //            Rope CosSin can be ND vector depending on rope type.
    //      Start index of the KVCache [B, 0~1] (a vector of scalars) with type int32_t.
    //            0 length indicates there is no existing KVCache for inference.
    //      Optional context mask selector: [0] for causal/default, [B] for non-causal PADDING mask.
    //      Optional packed tree attention mask: [T_exec, ceil(S/32)] with type int32_t.
    //      Optional tree attention position ids: [T_exec] with type int32_t.
    //      Optional SWA cache mode: INT32 [0] for full storage or [1] for bounded O(W) storage.

    // Support context/generation phase outputs:
    //      attention result (linear FP16) with shape [T_exec, Hq, D]
    //      KV-cache tensor, same as the above.
    // Packed-QKV input channel count: (Hq + 2*Hkv) * D for normal layers, and either
    // Hq * D (Q only) or (Hq + 2*Hkv) * D (Q plus transient donor K/V) for shared-KV
    // layers. No channel assertion here — Gemma4 uses heterogeneous per-layer head
    // configurations; the exact count is validated in enqueue().
    auto checkPackedQKV = [](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 2;
        return status;
    };

    auto checkKVCache = [this](PluginTensorDesc const& tensorDesc) {
        return isKVCacheDescriptor(tensorDesc, mEnableFp8KVCache)
            && isPagedPoolShape(tensorDesc.dims, mNumKVHeads, mHeadSize, true);
    };

    auto checkSequenceLen = [](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        return status;
    };

    auto checkPosEncodingCosSin = [this](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kFLOAT;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 2;
        status &= tensorDesc.dims.d[1] <= mHeadSize;
        return status;
    };

    auto checkContextMaskSelector = [](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        return status;
    };

    auto checkAttentionMask = [](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 2;
        return status;
    };

    auto checkAttentionPosId = [](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        return status;
    };

    auto checkVisionBlockIds = [](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        return status;
    };

    auto checkKVCacheStartIdx = [](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        return status;
    };

    // kv_page_table: INT32 [batch, 2, maxPagesPerSeq] — K page ids then derived V page ids.
    auto checkKVPageTable = [](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        if (status)
        {
            status &= tensorDesc.dims.d[1] == 2;
        }
        return status;
    };

    auto checkAttentionOutput = [this](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        if (status)
        {
            auto const tensorDim = tensorDesc.dims;
            status &= tensorDim.d[1] == mNumQHeads;
            status &= tensorDim.d[2] == mHeadSize;
        }
        return status;
    };

    AttentionInputLayout const inputLayout = this->inputLayout();
    int32_t const expectedNbInputs = inputLayout.numInputs;
    bool const checkNumIOs = nbInputs == expectedNbInputs && nbOutputs == kNUM_REQUIRED_OUTPUTS;
    if (inOut == nullptr || !checkNumIOs || pos < 0 || pos >= nbInputs + nbOutputs)
    {
        LOG_ERROR(
            "Invalid format query for AttentionPlugin '%s': expected %d inputs and %d outputs, got %d inputs "
            "and %d outputs at position %d.",
            mLayerName.c_str(), expectedNbInputs, kNUM_REQUIRED_OUTPUTS, nbInputs, nbOutputs, pos);
        return false;
    }

    bool result{true};

    if (pos < nbInputs)
    {
        switch (pos)
        {
        case kIN_QKV_IDX: result = checkPackedQKV(inOut[pos].desc); break;
        case kIN_KV_CACHE_IDX: result = checkKVCache(inOut[pos].desc); break;
        case kIN_QUERY_LENGTH_IDX: result = checkSequenceLen(inOut[pos].desc); break;
        case kIN_ROPE_COS_SIN_IDX: result = checkPosEncodingCosSin(inOut[pos].desc); break;
        case kIN_KV_CACHE_START_IDX: result = checkKVCacheStartIdx(inOut[pos].desc); break;
        case kIN_KV_PAGE_TABLE_IDX: result = checkKVPageTable(inOut[pos].desc); break;
        default: break;
        }

        // Handle optional inputs (qk_norm gammas, context selector, then tree/vision inputs) with dynamic ordering.
        if (result && pos >= kNUM_REQUIRED_INPUTS)
        {
            if (pos == inputLayout.prefixPool)
            {
                result = isKVCacheDescriptor(inOut[pos].desc, mFp8KVPrefix) && inOut[pos].desc.dims.d[3] == mNumKVHeads
                    && inOut[pos].desc.dims.d[4] == mHeadSize;
            }
            if (pos == inputLayout.prefixPageTable)
            {
                result = checkKVPageTable(inOut[pos].desc);
            }
            if (pos == inputLayout.qNormGamma || pos == inputLayout.kNormGamma)
            {
                result = inOut[pos].desc.type == DataType::kHALF && inOut[pos].desc.format == TensorFormat::kLINEAR
                    && inOut[pos].desc.dims.nbDims == 1 && inOut[pos].desc.dims.d[0] == mHeadSize;
            }
            if (pos == inputLayout.contextMaskSelector)
            {
                result = checkContextMaskSelector(inOut[pos].desc);
            }
            if (pos == inputLayout.attentionMask)
            {
                result = checkAttentionMask(inOut[pos].desc);
            }
            if (pos == inputLayout.attentionPosId)
            {
                result = checkAttentionPosId(inOut[pos].desc);
            }
            if (pos == inputLayout.visionBlockIds)
            {
                result = checkVisionBlockIds(inOut[pos].desc);
            }
            if (pos == inputLayout.skipSoftmaxPrefillScale || pos == inputLayout.skipSoftmaxDecodeScale)
            {
                // Skip-softmax scale tables: 1-D INT32 of flattened (seqlen_bin, S) pairs; length 0 = dense.
                result = inOut[pos].desc.type == DataType::kINT32 && inOut[pos].desc.format == TensorFormat::kLINEAR
                    && inOut[pos].desc.dims.nbDims == 1;
            }
            if (pos == inputLayout.swaCacheMode)
            {
                result = inOut[pos].desc.type == DataType::kINT32 && inOut[pos].desc.format == TensorFormat::kLINEAR
                    && inOut[pos].desc.dims.nbDims == 1;
            }
            if (pos == inputLayout.swaCacheMode)
            {
                // Shape-only carrier: length 1 selects bounded O(W) storage and length 0 selects
                // the ordinary full paged cache. The tensor data is never read.
                result = inOut[pos].desc.type == DataType::kINT32 && inOut[pos].desc.format == TensorFormat::kLINEAR
                    && inOut[pos].desc.dims.nbDims == 1;
            }
            if (pos == inputLayout.attentionSink)
            {
                result = inOut[pos].desc.type == DataType::kFLOAT && inOut[pos].desc.format == TensorFormat::kLINEAR
                    && inOut[pos].desc.dims.nbDims == 1 && inOut[pos].desc.dims.d[0] == mNumQHeads;
            }
            int32_t const raggedMetadataStart = inputLayout.raggedMetadataStart;
            if (pos >= raggedMetadataStart && pos < raggedMetadataStart + 3)
            {
                result = checkSequenceLen(inOut[pos].desc);
            }
            if (pos == raggedMetadataStart + 3)
            {
                result = inOut[pos].desc.type == DataType::kINT32 && inOut[pos].desc.format == TensorFormat::kLINEAR
                    && inOut[pos].desc.dims.nbDims == 1;
            }
        }
    }
    else
    {
        int32_t outPos = pos - nbInputs;
        switch (outPos)
        {
        case kOUT_ATTENTION_IDX: result = checkAttentionOutput(inOut[pos].desc); break;
        case kOUT_KV_CACHE_IDX: result = checkKVCache(inOut[pos].desc); break;
        default: break;
        }
    }

    return result;
}

int32_t AttentionPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    AttentionInputLayout const inputLayout = this->inputLayout();
    int32_t const expectedNbInputs = inputLayout.numInputs;
    if (in == nullptr || out == nullptr || nbInputs != expectedNbInputs || nbOutputs != kNUM_REQUIRED_OUTPUTS)
    {
        LOG_ERROR("AttentionPlugin: expected %d inputs and %d outputs, but got %d inputs and %d outputs.",
            expectedNbInputs, kNUM_REQUIRED_OUTPUTS, nbInputs, nbOutputs);
        return -1;
    }

    bool const matchingDescriptors = isKVCacheDescriptor(in[kIN_KV_CACHE_IDX].desc, mEnableFp8KVCache)
        && isKVCacheDescriptor(out[kOUT_KV_CACHE_IDX].desc, mEnableFp8KVCache)
        && isKVPageTableDescriptor(in[kIN_KV_PAGE_TABLE_IDX].desc);
    int32_t const swaModeIdx = inputLayout.swaCacheMode;
    bool const validSwaModeProfile = !mSupportsBoundedKVCache
        || (in[swaModeIdx].min.d[0] == kFULL_KV_CACHE_MODE_LENGTH
            && in[swaModeIdx].opt.d[0] == kBOUNDED_SWA_CACHE_MODE_LENGTH
            && in[swaModeIdx].max.d[0] == kBOUNDED_SWA_CACHE_MODE_LENGTH);
    // TensorRT can floor a derived token axis to zero at the min endpoint, e.g. when a DFlash2 draft flattens its
    // [batch, rows / batch, hidden] block view back to tokens; onShapeChange and enqueue reject a real empty QKV.
    bool const validProfiles = matchingDescriptors && validSwaModeProfile
        && hasConcretePagedKVContract(in[kIN_QKV_IDX].min, in[kIN_QUERY_LENGTH_IDX].min, in[kIN_KV_CACHE_IDX].min,
            out[kOUT_KV_CACHE_IDX].min, in[kIN_KV_PAGE_TABLE_IDX].min, mNumKVHeads, mHeadSize, mSupportsBoundedKVCache,
            /*allowEmptyQKV=*/true)
        && hasConcretePagedKVContract(in[kIN_QKV_IDX].opt, in[kIN_QUERY_LENGTH_IDX].opt, in[kIN_KV_CACHE_IDX].opt,
            out[kOUT_KV_CACHE_IDX].opt, in[kIN_KV_PAGE_TABLE_IDX].opt, mNumKVHeads, mHeadSize, mSupportsBoundedKVCache)
        && hasConcretePagedKVContract(in[kIN_QKV_IDX].max, in[kIN_QUERY_LENGTH_IDX].max, in[kIN_KV_CACHE_IDX].max,
            out[kOUT_KV_CACHE_IDX].max, in[kIN_KV_PAGE_TABLE_IDX].max, mNumKVHeads, mHeadSize, mSupportsBoundedKVCache);
    if (!validProfiles)
    {
        LOG_ERROR(
            "AttentionPlugin: KV cache profiles must use pool shape [2, N, %d, %d, %d], preserve the KV output "
            "shape, and use INT32 LINEAR kv_page_table [B, 2, M] with B matching packed QKV; full-cache-only "
            "profiles require M <= N.",
            rt::kTOKENS_PER_PAGE, mNumKVHeads, mHeadSize);
        return -1;
    }
    if (mExternalKVPrefix)
    {
        auto const& pool = in[inputLayout.prefixPool];
        auto const& table = in[inputLayout.prefixPageTable];
        bool const valid = isKVCacheDescriptor(pool.desc, mFp8KVPrefix) && isKVPageTableDescriptor(table.desc)
            && hasConcretePagedKVContract(in[kIN_QKV_IDX].min, in[kIN_QUERY_LENGTH_IDX].min, pool.min, pool.min,
                table.min, mNumKVHeads, mHeadSize, false)
            && hasConcretePagedKVContract(in[kIN_QKV_IDX].opt, in[kIN_QUERY_LENGTH_IDX].opt, pool.opt, pool.opt,
                table.opt, mNumKVHeads, mHeadSize, false)
            && hasConcretePagedKVContract(in[kIN_QKV_IDX].max, in[kIN_QUERY_LENGTH_IDX].max, pool.max, pool.max,
                table.max, mNumKVHeads, mHeadSize, false);
        if (!valid)
        {
            LOG_ERROR("AttentionPlugin: invalid retained prefix pool/table profile.");
            return -1;
        }
        return 0;
    }
    // The packed QKV binding is token-major [tokens, width]; the batch comes from query_lengths. Called once
    // per optimization profile; keep the widest so the runtime buffer covers every profile.
    int64_t const maxBatchSize = in[kIN_QUERY_LENGTH_IDX].max.d[0];
    int64_t const maxSeqLen = maxBatchSize > 0 ? in[kIN_QKV_IDX].max.d[0] / maxBatchSize : 1;
    size_t const capacity = ctmSemaphoreElems(static_cast<int32_t>(maxBatchSize), static_cast<int32_t>(maxSeqLen));
    if (capacity > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    {
        LOG_ERROR("AttentionPlugin: CTM semaphore capacity %zu exceeds the serializable range.", capacity);
        return -1;
    }
    mCtmSemaphoreCapacityElems = std::max(mCtmSemaphoreCapacityElems, static_cast<int32_t>(capacity));
    return 0;
}

size_t AttentionPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    AttentionInputLayout const inputLayout = this->inputLayout();
    int32_t const expectedNbInputs = inputLayout.numInputs;
    if (inputs == nullptr || outputs == nullptr || nbInputs != expectedNbInputs || nbOutputs != kNUM_REQUIRED_OUTPUTS)
    {
        LOG_ERROR(
            "AttentionPlugin::getWorkspaceSize: expected %d inputs and %d outputs, but got %d inputs and %d "
            "outputs.",
            expectedNbInputs, kNUM_REQUIRED_OUTPUTS, nbInputs, nbOutputs);
        return 0;
    }

    bool const reservesCompact = supportsCompactPhase(inputs[kIN_QKV_IDX].max.d[1]);
    reportCompactWorkspace("AttentionPlugin", mLayerName, reservesCompact);

    // Packed QKV: max batch/seq derived from packed input's first two dims (same as Q).
    int64_t const maxBatchSize = inputs[kIN_QUERY_LENGTH_IDX].max.d[0];
    int64_t const maxSeqLen = inputs[kIN_QKV_IDX].max.d[0] / maxBatchSize;
    // Split-K scratch for whichever CTM decode module the layer can dispatch to; the tree module's
    // [B*S*Hq] partial-O rows dominate whenever tree decode is enabled.
    if (mExternalKVPrefix)
    {
        size_t size = 0;
        for (int i = 0; i < 4; ++i)
        {
            size = accumulateWorkspaceSize(size, rt::Coords{maxBatchSize + 1}, DataType::kINT32);
        }
        return accumulateWorkspaceSize(
            size, rt::Coords{maxBatchSize, maxSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
    }
    size_t ctmWorkspaceElems = 0;
    if (useCtmDecode())
    {
        ctmWorkspaceElems
            = CtmDecodeAttentionRunner::workspaceElems(static_cast<int32_t>(maxBatchSize), mNumQHeads, mHeadSize);
    }
    if (useCtmTreeDecode())
    {
        // Tree masks pack 32 tree tokens per word, so the mask width bounds the tree query length.
        constexpr int64_t kTREE_MASK_BITS_PER_WORD{32};
        int64_t const maxTreeQueryLen
            = std::min(maxSeqLen, inputs[inputLayout.attentionMask].max.d[1] * kTREE_MASK_BITS_PER_WORD);
        ctmWorkspaceElems = std::max(ctmWorkspaceElems,
            CtmDecodeAttentionRunner::treeWorkspaceElems(static_cast<int32_t>(maxBatchSize),
                static_cast<int32_t>(maxTreeQueryLen), mNumQHeads, mNumKVHeads, mHeadSize));
    }
    // KV binding is the paged pool [2, numPages, 128, Hkv, D]; the per-slot padded capacity is the
    // page-table width times the page size (kv_page_table is [batch, 2, maxPagesPerSeq]).
    int64_t const maxKVCacheCapacity = inputs[kIN_KV_PAGE_TABLE_IDX].max.d[2] * rt::kTOKENS_PER_PAGE;
    size_t workspaceSize{};
    if (mSupportsBoundedKVCache)
    {
        // Full-cache mode stays native-paged, while the bounded path and vision-block fallback need at most
        // W + current-sequence scratch, so neither policy needs a dense max-sequence KV workspace.
        workspaceSize = getSwaKVCacheWorkspaceSize(maxBatchSize, maxSeqLen, mSlidingWindowSize, mNumQHeads, mNumKVHeads,
            mHeadSize, mEnableVisionBlockAttention != 0);
        if (reservesCompact)
        {
            workspaceSize = std::max(workspaceSize,
                getSwaCompactWorkspaceSize(maxBatchSize, inputs[kIN_QKV_IDX].max.d[0], mSlidingWindowSize, mNumQHeads,
                    mNumKVHeads, mHeadSize, inputs[kIN_KV_PAGE_TABLE_IDX].max.d[2]));
        }
        if (ctmWorkspaceElems > 0)
        {
            workspaceSize = accumulateWorkspaceSize(
                workspaceSize, rt::Coords{static_cast<int64_t>(ctmWorkspaceElems)}, DataType::kFLOAT);
        }
    }
    else
    {
        bool const blackwellFMHA = mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL;
        // Blackwell paged FMHA reads the pool directly; only FMHA-v2 prefill, including the vision-block and FP8
        // shared-KV routes, gathers dense K/V.
        bool const reserveDenseKV
            = !blackwellFMHA || mUseFMHAV2VisionBlockFMHA || (mEnableKVShared != 0 && mEnableFp8KVCache != 0);
        workspaceSize = getAttentionWorkspaceSize(maxBatchSize, maxSeqLen, maxKVCacheCapacity, mNumQHeads, mNumKVHeads,
            mHeadSize, blackwellFMHA, mEnableFp8KVCache,
            mEnableVisionBlockAttention != 0 || mEnableContextMaskSelector != 0, ctmWorkspaceElems, reserveDenseKV);
        if (reservesCompact)
        {
            workspaceSize = std::max(workspaceSize,
                getCompactWorkspaceSize(maxBatchSize, inputs[kIN_QKV_IDX].max.d[0], maxKVCacheCapacity,
                    inputs[kIN_KV_PAGE_TABLE_IDX].max.d[2], mNumQHeads, mNumKVHeads, mHeadSize, mEnableFp8KVCache,
                    selectCompactPrefillBackend() == ContextFMHABackend::kCUTE_DSL_FMHA_V2, ctmWorkspaceElems));
        }
    }
    if (!mCtmDecodeOnly && mEnableTreeAttention == 0)
    {
        DecoderXQARunner const runner(fromTrt(mDataType), fromTrt(selectKvCacheDataType(mEnableFp8KVCache)),
            maxBatchSize, mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion);
        workspaceSize += runner.getWorkspaceSize(static_cast<uint32_t>(maxKVCacheCapacity));
    }

    LOG_DEBUG("AttentionPlugin workspace size: %zu bytes", workspaceSize);
    return workspaceSize;
}

int32_t AttentionPlugin::getAliasedInput([[maybe_unused]] int32_t outputIndex) noexcept
{
    // Myelin materializes a redundant per-layer copy for a declared read-write
    // alias. The runtime binds both tensors to the same paged KV pool.
    return -1;
}

// ---------------------------------------------------------------------------
// IPluginV3OneRuntime — execution
// ---------------------------------------------------------------------------

int32_t AttentionPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // enqueue is noexcept: an exception escaping a kernel dispatch (e.g. a
    // missing-cubin check) would terminate the process. Turn it into a failed
    // enqueue instead.
    try
    {
        return enqueueImpl(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("AttentionPlugin: enqueue failed: %s", e.what());
        return -1;
    }
    catch (...)
    {
        LOG_ERROR("AttentionPlugin: enqueue failed with a non-standard exception.");
        return -1;
    }
}

//! Resolve one qk_norm gamma engine-weight input to a device pointer (nullptr when absent
//! or zero-length => RoPE-only path). Not noexcept: throws are caught by enqueue().
half const* AttentionPlugin::resolveNormGammaInput(
    PluginTensorDesc const* inputDesc, void const* const* inputs, int32_t inputIdx) const
{
    if (inputDesc[inputIdx].dims.d[0] <= 0)
    {
        return nullptr;
    }
    check::check(inputDesc[inputIdx].dims.d[0] == mHeadSize, "qk_norm gamma length must equal head_size.");
    return static_cast<half const*>(inputs[inputIdx]);
}

float const* AttentionPlugin::resolveAttentionSinkInput(
    PluginTensorDesc const* inputDesc, void const* const* inputs, int32_t inputIdx) const
{
    if (inputDesc[inputIdx].dims.d[0] <= 0)
    {
        return nullptr;
    }
    check::check(inputDesc[inputIdx].dims.d[0] == mNumQHeads, "attention_sinks length must equal num_q_heads.");
    check::check(inputDesc[inputIdx].type == DataType::kFLOAT, "attention_sinks must be FP32.");
    return static_cast<float const*>(inputs[inputIdx]);
}

int32_t AttentionPlugin::enqueueImpl(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    check::check(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr,
        "AttentionPlugin received null enqueue descriptors or bindings.");
    check::check(hasConcretePagedKVContract(
                     inputDesc, outputDesc, mNumKVHeads, mHeadSize, mEnableFp8KVCache, mSupportsBoundedKVCache),
        "AttentionPlugin requires kv_cache pool [2, N, kTOKENS_PER_PAGE, numKVHeads, headDim], identical KV input "
        "and output shapes, and INT32 LINEAR kv_page_table [B, 2, M] with B matching packed QKV and M <= N.");
    check::check(inputs[kIN_KV_CACHE_IDX] != nullptr && inputs[kIN_KV_PAGE_TABLE_IDX] != nullptr
            && outputs[kOUT_KV_CACHE_IDX] != nullptr,
        "AttentionPlugin requires non-null KV pool and page-table bindings.");
    AttentionInputLayout const inputLayout = this->inputLayout();

    if (mExternalKVPrefix)
    {
        int32_t const batch = inputDesc[kIN_QUERY_LENGTH_IDX].dims.d[0];
        int32_t const tokens = inputDesc[kIN_QKV_IDX].dims.d[0];
        check::check(batch > 0 && tokens % batch == 0, "Action QKV must contain equal-sized batch rows.");
        int32_t const seq = tokens / batch;
        check::check(inputDesc[inputLayout.contextMaskSelector].dims.d[0] == batch
                && inputDesc[inputLayout.raggedMetadataStart + 2].dims.d[0]
                    == static_cast<int32_t>(rt::ExecutionPhase::kDiffusionDenoise),
            "Separate-prefix attention requires the action non-causal execution phase.");
        auto const& prefixDesc = inputDesc[inputLayout.prefixPool];
        auto const& prefixTableDesc = inputDesc[inputLayout.prefixPageTable];
        check::check(isKVCacheDescriptor(prefixDesc, mFp8KVPrefix) && prefixDesc.dims.d[3] == mNumKVHeads
                && prefixDesc.dims.d[4] == mHeadSize && isKVPageTableDescriptor(prefixTableDesc)
                && prefixTableDesc.dims.d[0] == batch && prefixTableDesc.dims.d[2] <= prefixDesc.dims.d[1],
            "Invalid retained prefix pool/table.");
        check::check(inputDesc[kIN_KV_PAGE_TABLE_IDX].dims.d[2] * rt::kTOKENS_PER_PAGE >= seq,
            "Action scratch does not cover the action sequence.");
        CuteDslFMHAV2Runner runner(
            mNumQHeads, mNumKVHeads, mHeadSize, batch, seq, prefixTableDesc.dims.d[2] * rt::kTOKENS_PER_PAGE, false);
        if (!runner.preflightPagedTwoPoolNonCausal(stream, mFp8KVPrefix != 0))
        {
            LOG_ERROR("AttentionPlugin: required two-pool attention kernel is unavailable.");
            return -1;
        }
        auto tensor = [&](int idx) {
            auto const& d = inputDesc[idx];
            return rt::Tensor(
                const_cast<void*>(inputs[idx]), rt::toCoords(d.dims), rt::DeviceType::kGPU, fromTrt(d.type));
        };
        rt::Tensor queryLengths = tensor(kIN_QUERY_LENGTH_IDX);
        rt::Tensor prefixLengths = tensor(kIN_KV_CACHE_START_IDX);
        rt::Tensor rope(const_cast<void*>(inputs[kIN_ROPE_COS_SIN_IDX]),
            {batch, seq, inputDesc[kIN_ROPE_COS_SIN_IDX].dims.d[1]}, rt::DeviceType::kGPU,
            trt_edgellm::DataType::kFLOAT);
        rt::Tensor actionPool = tensor(kIN_KV_CACHE_IDX);
        rt::Tensor packed(const_cast<void*>(inputs[kIN_QKV_IDX]), {batch, seq, mNumQHeads + 2 * mNumKVHeads, mHeadSize},
            rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
        auto* scratch = static_cast<std::byte*>(workspace);
        auto cuQ = assignTensorFromWorkspace(scratch, {batch + 1}, DataType::kINT32);
        auto cuKV = assignTensorFromWorkspace(scratch, {batch + 1}, DataType::kINT32);
        auto ends = assignTensorFromWorkspace(scratch, {batch}, DataType::kINT32);
        auto padded = assignTensorFromWorkspace(scratch, {batch + 1}, DataType::kINT32);
        auto q = assignTensorFromWorkspace(scratch, {batch, seq, mNumQHeads, mHeadSize}, DataType::kHALF);
        kernel::calCuQCuKVSeqLensAndKVEndIdxs(queryLengths, prefixLengths, cuQ, cuKV, ends, padded, seq, stream);
        check::check(outputs[kOUT_KV_CACHE_IDX] == inputs[kIN_KV_CACHE_IDX], "Action pool bindings must alias.");
        kernel::launchApplyRopeFromPackedToSplit(rope, rt::OptionalInputTensor{queryLengths}, {}, packed, q, actionPool,
            1.F, 1.F, stream, static_cast<int32_t const*>(inputs[kIN_KV_PAGE_TABLE_IDX]),
            inputDesc[kIN_KV_PAGE_TABLE_IDX].dims.d[2], nullptr, nullptr, nullptr, 1.F, nullptr, nullptr, 1e-6F, false,
            rt::OptionalInputTensor{cuQ}, true, false, true);
        return runner.runPagedTwoPoolNonCausal(q.rawPointer(), inputs[inputLayout.prefixPool],
                   static_cast<int32_t const*>(inputs[inputLayout.prefixPageTable]), actionPool.rawPointer(),
                   static_cast<int32_t const*>(inputs[kIN_KV_PAGE_TABLE_IDX]), outputs[kOUT_ATTENTION_IDX],
                   cuQ.dataPointer<int32_t>(), cuKV.dataPointer<int32_t>(), 2 * prefixDesc.dims.d[1],
                   prefixTableDesc.dims.d[2], 2 * actionPool.getShape()[1], inputDesc[kIN_KV_PAGE_TABLE_IDX].dims.d[2],
                   stream, mAttentionScale, mFp8KVPrefix != 0, mQkvScales[1], mQkvScales[2])
            ? 0
            : -1;
    }

    PluginTensorDesc const& phaseMarkerDesc = inputDesc[inputLayout.raggedMetadataStart + 2];
    if (phaseMarkerDesc.dims.nbDims == 1
        && phaseMarkerDesc.dims.d[0] == static_cast<int64_t>(rt::ExecutionPhase::kMixedPrefillDecode))
    {
        return enqueueCompactImpl(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    if (mEnableAttentionSink != 0 && phaseMarkerDesc.dims.nbDims == 1
        && (phaseMarkerDesc.dims.d[0] == static_cast<int64_t>(rt::ExecutionPhase::kContextPrefill)
            || phaseMarkerDesc.dims.d[0] == static_cast<int64_t>(rt::ExecutionPhase::kContextChunk)))
    {
        LOG_ERROR("AttentionPlugin: attention sinks require compact ragged prefill (phase 8).");
        return -1;
    }

    // Packed QKV input, layout selected by the enable_kv_shared field:
    //   0: [T_exec, (Hq+2*Hkv)*D] — Q+K+V; K/V are written to the KV cache.
    //   1: [T_exec, Hq*D] — Q only, or [T_exec, (Hq+2*Hkv)*D] with the donor's
    //      current K/V appended. Shared layers never write the donated cache.
    PluginTensorDesc const& packedQKVInputDesc = inputDesc[kIN_QKV_IDX];
    int32_t const runtimeBatchSize = static_cast<int32_t>(inputDesc[kIN_QUERY_LENGTH_IDX].dims.d[0]);
    int32_t const physicalTokens = static_cast<int32_t>(packedQKVInputDesc.dims.d[0]);
    check::check(runtimeBatchSize > 0 && physicalTokens % runtimeBatchSize == 0,
        "AttentionPlugin entry-padded backend requires T_exec to be divisible by N");
    int32_t const runtimeSeqLen = physicalTokens / runtimeBatchSize;
    int32_t const actualChannels = static_cast<int32_t>(packedQKVInputDesc.dims.d[1]);
    bool useBoundedSwaCache{false};
    if (mSupportsBoundedKVCache)
    {
        int32_t const modeIdx = inputLayout.swaCacheMode;
        PluginTensorDesc const& modeDesc = inputDesc[modeIdx];
        check::check(modeDesc.dims.nbDims == 1
                && (modeDesc.dims.d[0] == kFULL_KV_CACHE_MODE_LENGTH
                    || modeDesc.dims.d[0] == kBOUNDED_SWA_CACHE_MODE_LENGTH),
            "AttentionPlugin swa_kv_cache_mode must have runtime shape [0] or [1].");
        useBoundedSwaCache = modeDesc.dims.d[0] == kBOUNDED_SWA_CACHE_MODE_LENGTH;
    }
    bool const sharedKV = (mEnableKVShared != 0);
    int32_t const qChannels = mNumQHeads * mHeadSize;
    int32_t const qkvChannels = (mNumQHeads + 2 * mNumKVHeads) * mHeadSize;
    bool const sharedKVWithCurrent = sharedKV && actualChannels == qkvChannels;
    check::check((!sharedKV && actualChannels == qkvChannels)
            || (sharedKV && (actualChannels == qChannels || sharedKVWithCurrent)),
        "Packed QKV input last dim does not match enable_kv_shared: expected (Hq + 2*Hkv)*head_dim for "
        "owned KV, or Hq*head_dim / (Hq + 2*Hkv)*head_dim for shared KV.");
    int32_t const combinedHeads = sharedKVWithCurrent || !sharedKV ? (mNumQHeads + 2 * mNumKVHeads) : mNumQHeads;

    rt::Tensor packedQKVTensor(const_cast<void*>(inputs[kIN_QKV_IDX]),
        rt::Coords{runtimeBatchSize, runtimeSeqLen, combinedHeads, mHeadSize}, rt::DeviceType::kGPU,
        fromTrt(packedQKVInputDesc.type));

    // qInputTensor / kInputTensor / vInputTensor are assigned from workspace below and
    // populated by launchApplyRopeFromPackedToSplit.
    rt::Tensor qInputTensor;
    rt::Tensor kInputTensor;
    rt::Tensor vInputTensor;

    // The external Q-only binding is token-major. Alias its storage through the entry-padded internal view expected
    // by the shared-KV RoPE kernels.
    auto aliasPackedAsQInput = [&]() {
        return rt::Tensor(const_cast<void*>(inputs[kIN_QKV_IDX]),
            rt::Coords{runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, rt::DeviceType::kGPU,
            fromTrt(packedQKVInputDesc.type));
    };

    int32_t const raggedMetadataStart = inputLayout.raggedMetadataStart;
    int32_t const queryStartOffsetIdx = raggedMetadataStart;
    int32_t const attentionSequenceLengthIdx = raggedMetadataStart + 1;
    int32_t const phaseIdx = raggedMetadataStart + 2;
    int32_t const contextSequenceCountIdx = raggedMetadataStart + 3;
    RaggedPluginMetadata const ragged
        = decodeRaggedPluginMetadata("AttentionPlugin", packedQKVInputDesc, inputDesc[kIN_QUERY_LENGTH_IDX],
            inputDesc[queryStartOffsetIdx], inputDesc[phaseIdx], inputDesc[contextSequenceCountIdx]);
    rt::ExecutionPhase const phase = ragged.phase;
    PluginTensorDesc const& posEncodingCosSinDesc = inputDesc[kIN_ROPE_COS_SIN_IDX];
    rt::Coords const ropeShape{runtimeBatchSize, runtimeSeqLen, posEncodingCosSinDesc.dims.d[1]};
    rt::Tensor const ropeCosSinTensor(const_cast<void*>(inputs[kIN_ROPE_COS_SIN_IDX]), ropeShape, rt::DeviceType::kGPU,
        fromTrt(posEncodingCosSinDesc.type));

    PluginTensorDesc const& kvCacheStartIdxInputDesc = inputDesc[kIN_KV_CACHE_START_IDX];
    rt::Tensor const kvCacheStartIdxTensor(const_cast<void*>(inputs[kIN_KV_CACHE_START_IDX]),
        rt::toCoords(kvCacheStartIdxInputDesc.dims), rt::DeviceType::kGPU, fromTrt(kvCacheStartIdxInputDesc.type));

    PluginTensorDesc const& attentionOutputDesc = outputDesc[kOUT_ATTENTION_IDX];
    rt::Coords const attentionOutputShape{runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize};
    rt::Tensor attentionOutputTensor(
        outputs[kOUT_ATTENTION_IDX], attentionOutputShape, rt::DeviceType::kGPU, fromTrt(attentionOutputDesc.type));

    // Construct the KV cache tensors from the past-KV input descriptor. The buffer is the paged
    // pool [2, numPages, 128, Hkv, D] (K page array then V page array); the present-KV output
    // in-place aliases it (output 1 aliases input 1). Shared-KV draft layers are read-only views
    // of the target/base cache, so they must read the input binding rather than the plugin's
    // present-KV output; the present-KV output is not consumed in shared-KV mode and must remain
    // unwritten by every shared-KV path below.
    PluginTensorDesc const& kvCacheInputDesc = inputDesc[kIN_KV_CACHE_IDX];
    rt::Tensor pastKVCacheTensor(const_cast<void*>(inputs[kIN_KV_CACHE_IDX]), rt::toCoords(kvCacheInputDesc.dims),
        rt::DeviceType::kGPU, fromTrt(kvCacheInputDesc.type));
    rt::Tensor presentKVCacheTensor(outputs[kOUT_KV_CACHE_IDX], rt::toCoords(kvCacheInputDesc.dims),
        rt::DeviceType::kGPU, fromTrt(kvCacheInputDesc.type));
    rt::Tensor& kvCacheTensor = sharedKV ? pastKVCacheTensor : presentKVCacheTensor;
    // numPages feeds native paged CuTe DSL prefill; writes and XQA decode derive
    // capacity from the page table.
    int32_t const numPages = static_cast<int32_t>(kvCacheInputDesc.dims.d[1]);

    // The runtime-supplied page table [batch, 2, maxPagesPerSeq] carries the K-then-V kernel view
    // (KVPageTable convention: V page id = K page id + numPages). It is a required input with no
    // plugin-internal identity fallback — an empty/missing table is a hard error.
    PluginTensorDesc const& kvPageTableInputDesc = inputDesc[kIN_KV_PAGE_TABLE_IDX];
    rt::Tensor const kvPageTableTensor(const_cast<void*>(inputs[kIN_KV_PAGE_TABLE_IDX]),
        rt::toCoords(kvPageTableInputDesc.dims), rt::DeviceType::kGPU, fromTrt(kvPageTableInputDesc.type));
    check::check(!kvPageTableTensor.isEmpty(), "AttentionPlugin: kv_page_table input is required for paged attention.");
    int32_t const* const pageTable = kvPageTableTensor.dataPointer<int32_t>();
    int32_t const maxPagesPerSeq = static_cast<int32_t>(kvPageTableInputDesc.dims.d[2]);
    check::check(useBoundedSwaCache || numPages >= maxPagesPerSeq,
        "AttentionPlugin full-cache mode requires at least one physical page per logical page-table column.");
    // Padded per-slot token capacity spanned by the page table (each page holds kTOKENS_PER_PAGE).
    int32_t const kvCacheCapacity = maxPagesPerSeq * rt::kTOKENS_PER_PAGE;

    // Optional inputs use the same compact dynamic ordering as supportsFormatCombination().
    bool const enableQKNorm = mEnableQKNorm != 0;
    bool const enableContextMaskSelector = mEnableContextMaskSelector != 0;
    RuntimeContextMaskMode const runtimeContextMaskMode
        = selectRuntimeContextMaskMode(inputDesc, enableContextMaskSelector, enableQKNorm);

    rt::Tensor attentionMaskTensor{};
    rt::Tensor attentionPosIdTensor{};
    rt::Tensor visionBlockIdsTensor{};
    if (mEnableTreeAttention)
    {
        int32_t const maskIdx = inputLayout.attentionMask;
        int32_t const posIdIdx = inputLayout.attentionPosId;
        PluginTensorDesc const& attentionMaskInputDesc = inputDesc[maskIdx];
        PluginTensorDesc const& attentionPosIdInputDesc = inputDesc[posIdIdx];
        attentionMaskTensor = rt::Tensor(const_cast<void*>(inputs[maskIdx]),
            rt::Coords{runtimeBatchSize, runtimeSeqLen, attentionMaskInputDesc.dims.d[1]}, rt::DeviceType::kGPU,
            fromTrt(attentionMaskInputDesc.type));
        attentionPosIdTensor = rt::Tensor(const_cast<void*>(inputs[posIdIdx]),
            rt::Coords{runtimeBatchSize, runtimeSeqLen}, rt::DeviceType::kGPU, fromTrt(attentionPosIdInputDesc.type));
    }
    else if (mEnableVisionBlockAttention)
    {
        // The vision-block-ID tensor rides the attention_mask slot; its position
        // depends on whether qk_norm gamma and context selector inputs precede it.
        int32_t const maskIdx = inputLayout.visionBlockIds;
        PluginTensorDesc const& visionBlockIdsDesc = inputDesc[maskIdx];
        rt::Coords const visionBlockShape{runtimeBatchSize, runtimeSeqLen};
        visionBlockIdsTensor = rt::Tensor(const_cast<void*>(inputs[maskIdx]), visionBlockShape, rt::DeviceType::kGPU,
            fromTrt(visionBlockIdsDesc.type));
    }
    bool const useExplicitPositionIds = mEnableTreeAttention && !attentionPosIdTensor.isEmpty()
        && attentionPosIdTensor.getShape().getNumDims() == 2 && attentionPosIdTensor.getShape()[1] == runtimeSeqLen;
    float const kScale = mQkvScales[1];
    float const vScale = mQkvScales[2];

    AttentionExecutionMode const executionMode = resolveAttentionExecutionMode(phase, mEnableTreeAttention);

    // For invalid execution mode, log error and report error return value.
    if (executionMode == AttentionExecutionMode::kINVALID)
    {
        LOG_ERROR("Invalid attention execution mode detected. Abort the AttentionPlugin enqueue() call.");
        return 1;
    }

    // Skip-softmax (BLASST) scale tables: the prefill modes read skip_softmax_prefill_scale, vanilla decode reads
    // skip_softmax_decode_scale, tree decode runs dense. Never skip on sliding-window layers.
    int32_t const* prefillSkipTable = nullptr;
    int32_t prefillSkipTableLen = 0;
    int32_t const* decodeSkipTable = nullptr;
    int32_t decodeSkipTableLen = 0;
    if (mSlidingWindowSize <= 0)
    {
        if (inputLayout.skipSoftmaxPrefillScale >= 0)
        {
            prefillSkipTable = static_cast<int32_t const*>(inputs[inputLayout.skipSoftmaxPrefillScale]);
            prefillSkipTableLen = static_cast<int32_t>(inputDesc[inputLayout.skipSoftmaxPrefillScale].dims.d[0]);
        }
        if (inputLayout.skipSoftmaxDecodeScale >= 0 && executionMode == AttentionExecutionMode::kVANILLA_DECODING)
        {
            decodeSkipTable = static_cast<int32_t const*>(inputs[inputLayout.skipSoftmaxDecodeScale]);
            decodeSkipTableLen = static_cast<int32_t>(inputDesc[inputLayout.skipSoftmaxDecodeScale].dims.d[0]);
        }
    }

    int32_t const contextLengthIdx
        = attentionModeUsesQueryLengths(executionMode) ? kIN_QUERY_LENGTH_IDX : attentionSequenceLengthIdx;
    PluginTensorDesc const& contextLengthInputDesc = inputDesc[contextLengthIdx];
    rt::Tensor const contextLengthTensor(const_cast<void*>(inputs[contextLengthIdx]),
        rt::toCoords(contextLengthInputDesc.dims), rt::DeviceType::kGPU, fromTrt(contextLengthInputDesc.type));

    check::check(!mEnableAttentionSink
            || (executionMode != AttentionExecutionMode::kNORMAL_PREFILL
                && executionMode != AttentionExecutionMode::kCHUNKED_PREFILL),
        "Learned attention sinks are not yet implemented in the padded FMHA prefill path.");
    auto* alignedWorkspacePtr = static_cast<std::byte*>(workspace);
    if (alignedWorkspacePtr == nullptr
        || reinterpret_cast<uintptr_t>(alignedWorkspacePtr) % static_cast<uintptr_t>(kDEVICE_ALIGNMENT) != 0)
    {
        LOG_ERROR("Workspace pointer is not aligned to device alignment granularity");
        return 1;
    }

    // Gamma engine-weight inputs are device-resident at engine load. Models without
    // qk_norm do not wire them ⇒ nullptr ⇒ the kernel takes the RoPE-only path.
    half const* qNormGammaDevicePtr
        = mEnableQKNorm ? resolveNormGammaInput(inputDesc, inputs, inputLayout.qNormGamma) : nullptr;
    half const* kNormGammaDevicePtr
        = mEnableQKNorm ? resolveNormGammaInput(inputDesc, inputs, inputLayout.kNormGamma) : nullptr;
    float const* attentionSinkDevicePtr
        = mEnableAttentionSink ? resolveAttentionSinkInput(inputDesc, inputs, inputLayout.attentionSink) : nullptr;
    float const rmsNormEpsVal = mRmsNormEps;
    bool const enableRopePdl = requestRopePdl() && mSMVersion >= kROPE_MIN_PDL_SM_VERSION;
    bool const qkNormPostRope = mQKNormPostRope != 0;

    // ==================== Prefill path ====================
    // Dispatch order: vision-block attention first (paged CuTe DSL or
    // FMHA-v2 CuTe DSL), then sharedKV, then own-KV.
    // Within each: FMHA, or reject the prefill when no kernel serves the head size.
    if (executionMode == AttentionExecutionMode::kNORMAL_PREFILL
        || executionMode == AttentionExecutionMode::kCHUNKED_PREFILL)
    {
        bool const usePaddingContextMask = isPaddingContextMask(runtimeContextMaskMode);
        bool const useSwaChunkAssembly = executionMode == AttentionExecutionMode::kCHUNKED_PREFILL
            && mSupportsBoundedKVCache && (useBoundedSwaCache || mEnableFp8KVCache);
        bool const useDenseSwaCurrentKV = useBoundedSwaCache || (mSupportsBoundedKVCache && mEnableFp8KVCache);
        bool const useSharedFp8FMHAV2 = sharedKV && mEnableFp8KVCache;
        if (mEnableAttentionSink)
        {
            check::check(!usePaddingContextMask && !useBoundedSwaCache && !useSwaChunkAssembly && !mEnableFp8KVCache
                    && !mEnableVisionBlockAttention && !mEnableTreeAttention
                    && mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_V2,
                "Learned attention sinks require causal/sliding FP16 FMHA-v2 ragged paged prefill; padding masks, "
                "bounded SWA, FP8, tree, and vision paths are unsupported.");
        }
        // Optimized CuTe DSL FMHA consumes FP8 Q/K/V directly; FMHA-v2 retains the page-table-aware
        // gather that dequantizes K/V to FP16 before launching a dense kernel. Both need an FMHA
        // backend, so mCanImplementFMHA alone decides whether either route is available.
        if (mEnableFp8KVCache && executionMode == AttentionExecutionMode::kCHUNKED_PREFILL && !mCanImplementFMHA)
        {
            LOG_ERROR(
                "AttentionPlugin: FP8 KV cache chunked prefill has no direct or split-KV FMHA backend for "
                "headSize=%d on SM %d.",
                mHeadSize, mSMVersion);
            return -1;
        }
        if (mEnableVisionBlockAttention)
        {
            if (executionMode != AttentionExecutionMode::kNORMAL_PREFILL
                || (useDenseSwaCurrentKV && sharedKV && !sharedKVWithCurrent))
            {
                LOG_ERROR(
                    "AttentionPlugin: Gemma4 vision-block attention supports only normal prefill; SWA shared KV "
                    "additionally requires explicit current donor KV (mode=%d, bounded=%d, sharedKV=%d, "
                    "currentKV=%d).",
                    static_cast<int32_t>(executionMode), static_cast<int32_t>(useBoundedSwaCache),
                    static_cast<int32_t>(sharedKV), static_cast<int32_t>(sharedKVWithCurrent));
                return 1;
            }
            if (visionBlockIdsTensor.getShape()[0] != runtimeBatchSize
                || visionBlockIdsTensor.getShape()[1] != runtimeSeqLen)
            {
                LOG_ERROR(
                    "AttentionPlugin: vision_block_ids must have shape [B,S] matching Q; got [%lld,%lld], "
                    "expected [%d,%d].",
                    static_cast<long long>(visionBlockIdsTensor.getShape()[0]),
                    static_cast<long long>(visionBlockIdsTensor.getShape()[1]), runtimeBatchSize, runtimeSeqLen);
                return 1;
            }

            bool visionKernelPreflighted{false};
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
            if (canUseCuteDslBidirectionalForPrefill())
            {
                int32_t const slidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize - 1 : INT_MAX;
                CuteDslFMHARunner runner(
                    mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);
                if (!runner.preflightPaged(stream, slidingWindow, /*fp8Input=*/false, /*isCausal=*/true,
                        /*skipSoftmaxScaleTable=*/nullptr, /*skipSoftmaxScaleTableLen=*/0, /*useBidirectional=*/true))
                {
                    return -1;
                }
                visionKernelPreflighted = true;
            }
#endif
            if (!visionKernelPreflighted && mUseFMHAV2VisionBlockFMHA)
            {
                CuteDslFMHAV2Runner runner(
                    mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, runtimeSeqLen);
                if (!runner.preflightVisionBlock(stream))
                {
                    return -1;
                }
                visionKernelPreflighted = true;
            }
            if (!visionKernelPreflighted)
            {
                LOG_ERROR("AttentionPlugin: selected vision-block prefill kernel is unavailable.");
                return -1;
            }

            rt::Tensor cuQSeqLensTensor
                = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);
            rt::Tensor cuKVSeqLensTensor
                = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);
            rt::Tensor kvCacheEndIdxsTensor
                = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize}, DataType::kINT32);
            rt::Tensor paddedCuKVSeqLensTensor
                = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);
            kernel::calCuQCuKVSeqLensAndKVEndIdxs(contextLengthTensor, kvCacheStartIdxTensor, cuQSeqLensTensor,
                cuKVSeqLensTensor, kvCacheEndIdxsTensor, paddedCuKVSeqLensTensor, runtimeSeqLen, stream);

            if (sharedKV && !sharedKVWithCurrent)
            {
                // Full-cache shared layers already have the donor's complete K/V in the paged pool.
                qInputTensor = aliasPackedAsQInput();
                kernel::launchApplyRopeQOnly(ropeCosSinTensor, qInputTensor, stream);
            }
            else
            {
                // Unpack packed QKV: roped Q to qScratch, optional current K/V to dense scratch, and owned K/V to
                // the paged pool.
                qInputTensor = assignTensorFromWorkspace(
                    alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
                if (useDenseSwaCurrentKV)
                {
                    kInputTensor = assignTensorFromWorkspace(alignedWorkspacePtr,
                        {runtimeBatchSize, runtimeSeqLen, mNumKVHeads, mHeadSize}, DataType::kHALF);
                    vInputTensor = assignTensorFromWorkspace(alignedWorkspacePtr,
                        {runtimeBatchSize, runtimeSeqLen, mNumKVHeads, mHeadSize}, DataType::kHALF);
                }
                kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor, rt::OptionalInputTensor{},
                    rt::OptionalInputTensor{}, packedQKVTensor, qInputTensor, kvCacheTensor, kScale, vScale, stream,
                    pageTable, maxPagesPerSeq, useDenseSwaCurrentKV ? kInputTensor.rawPointer() : nullptr,
                    useDenseSwaCurrentKV ? vInputTensor.rawPointer() : nullptr, nullptr /* fp8QOut */,
                    1.0f /* qScale */, qNormGammaDevicePtr, kNormGammaDevicePtr, rmsNormEpsVal, qkNormPostRope,
                    std::nullopt, !sharedKV, enableRopePdl, true /* tokenAlignedRope */);
            }

            if (useDenseSwaCurrentKV)
            {
                if (!mUseFMHAV2VisionBlockFMHA)
                {
                    LOG_ERROR(
                        "AttentionPlugin bounded SWA: no FMHA-v2 vision-block kernel for S=%d (headSize=%d, SM=%d).",
                        runtimeSeqLen, mHeadSize, mSMVersion);
                    return 1;
                }

                LOG_DEBUG(
                    "AttentionPlugin SWA dense-current: vision-block prefill via FMHA-v2 "
                    "(B=%d, S=%d, Hq=%d, Hkv=%d, window=%d)",
                    runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads, mSlidingWindowSize);

                rt::Tensor blockBeginTensor = assignTensorFromWorkspace(
                    alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen}, DataType::kINT32);
                rt::Tensor blockEndTensor = assignTensorFromWorkspace(
                    alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen}, DataType::kINT32);
                kernel::launchBuildVisionBlockRanges(visionBlockIdsTensor.dataPointer<int32_t>(),
                    contextLengthTensor.dataPointer<int32_t>(), blockBeginTensor.dataPointer<int32_t>(),
                    blockEndTensor.dataPointer<int32_t>(), runtimeBatchSize, runtimeSeqLen, stream);

                CuteDslFMHAV2Runner runner(
                    mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, runtimeSeqLen);
                if (!runner.runVisionBlock(qInputTensor.dataPointer<half>(), kInputTensor.dataPointer<half>(),
                        vInputTensor.dataPointer<half>(), attentionOutputTensor.dataPointer<half>(),
                        paddedCuKVSeqLensTensor.dataPointer<int32_t>(), blockBeginTensor.dataPointer<int32_t>(),
                        blockEndTensor.dataPointer<int32_t>(), stream, mAttentionScale, mSlidingWindowSize - 1))
                {
                    return -1;
                }
                return 0;
            }

            rt::Tensor blockBeginTensor{};
            rt::Tensor blockEndTensor{};
            bool const useVisionRanges = canUseCuteDslBidirectionalForPrefill() || mUseFMHAV2VisionBlockFMHA;
            if (useVisionRanges)
            {
                blockBeginTensor = assignTensorFromWorkspace(
                    alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen}, DataType::kINT32);
                blockEndTensor = assignTensorFromWorkspace(
                    alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen}, DataType::kINT32);
                kernel::launchBuildVisionBlockRanges(visionBlockIdsTensor.dataPointer<int32_t>(),
                    contextLengthTensor.dataPointer<int32_t>(), blockBeginTensor.dataPointer<int32_t>(),
                    blockEndTensor.dataPointer<int32_t>(), runtimeBatchSize, runtimeSeqLen, stream);
            }

#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
            if (canUseCuteDslBidirectionalForPrefill())
            {
                LOG_DEBUG(
                    "AttentionPlugin: vision-block prefill via paged CuTe DSL d512 bidirectional mask "
                    "(B=%d, S=%d, Hq=%d, Hkv=%d, "
                    "window=%d)",
                    runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads, mSlidingWindowSize);

                int32_t const slidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize - 1 : INT_MAX;
                CuteDslFMHARunner runner(
                    mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);
                if (!runner.runPaged(qInputTensor.dataPointer<half>(), kvCacheTensor.rawPointer(), pageTable,
                        attentionOutputTensor.dataPointer<half>(), paddedCuKVSeqLensTensor.dataPointer<int32_t>(),
                        2 * numPages, maxPagesPerSeq, rt::kTOKENS_PER_PAGE, kvCacheTensor.getDataType(), stream,
                        mAttentionScale, slidingWindow, /*fp8Input=*/false, 1.0F, 1.0F, 1.0F,
                        /*isCausal=*/true, /*skipSoftmaxScaleTable=*/nullptr, /*skipSoftmaxScaleTableLen=*/0,
                        /*skipSoftmaxCuKVSeqLens=*/nullptr, blockBeginTensor.dataPointer<int32_t>(),
                        blockEndTensor.dataPointer<int32_t>()))
                {
                    return -1;
                }
                return 0;
            }
#endif

            if (mUseFMHAV2VisionBlockFMHA)
            {
                LOG_DEBUG(
                    "AttentionPlugin: vision-block prefill via FMHA-v2 CuTe DSL d%d (B=%d, S=%d, Hq=%d, "
                    "Hkv=%d, window=%d)",
                    mHeadSize, runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads, mSlidingWindowSize);

                auto [kSplit, vSplit]
                    = splitPagedKV(kvCacheTensor, pageTable, kvCacheEndIdxsTensor.dataPointer<int32_t>(),
                        maxPagesPerSeq, alignedWorkspacePtr, runtimeBatchSize, mNumKVHeads, kvCacheCapacity, mHeadSize,
                        /*seqLen=*/runtimeSeqLen, kScale, vScale, stream);
                int32_t const slidingWindow = mSlidingWindowSize - 1;
                CuteDslFMHAV2Runner runner(
                    mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, runtimeSeqLen);
                if (!runner.runVisionBlock(qInputTensor.dataPointer<half>(), kSplit.dataPointer<half>(),
                        vSplit.dataPointer<half>(), attentionOutputTensor.dataPointer<half>(),
                        paddedCuKVSeqLensTensor.dataPointer<int32_t>(), blockBeginTensor.dataPointer<int32_t>(),
                        blockEndTensor.dataPointer<int32_t>(), stream, mAttentionScale, slidingWindow))
                {
                    return -1;
                }
                return 0;
            }

            LOG_ERROR("AttentionPlugin: selected vision-block prefill kernel is unavailable.");
            return -1;
        }

        // Keep this selected-variant preflight routing synchronized with the dispatch routing below. The exact
        // module must be loaded before any QKV, RoPE, or KV-cache mutation.
        bool fmhaKernelPreflighted{false};
        if (mEnableAttentionSink)
        {
            int32_t const slidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize - 1 : INT_MAX;
            CuteDslFMHAV2Runner runner(
                mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);
            if (!runner.preflightPagedRagged(stream, slidingWindow))
            {
                return -1;
            }
            fmhaKernelPreflighted = true;
        }
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
        if (!fmhaKernelPreflighted && mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL
            && (!usePaddingContextMask || mCanImplementPaddingFMHA) && !(mEnableFp8KVCache && sharedKV)
            && !useSwaChunkAssembly)
        {
            int32_t const slidingWindow
                = (mSlidingWindowSize > 0 && !usePaddingContextMask) ? mSlidingWindowSize - 1 : INT_MAX;
            bool const fp8Input = !sharedKV && mEnableFp8KVCache;
            int32_t const preflightPrefillSkipTableLen = fp8Input ? 0 : prefillSkipTableLen;
            CuteDslFMHARunner runner(
                mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);
            if (!runner.preflightPaged(stream, slidingWindow, fp8Input, !usePaddingContextMask, prefillSkipTable,
                    preflightPrefillSkipTableLen, /*useBidirectional=*/false))
            {
                return -1;
            }
            fmhaKernelPreflighted = true;
        }
#endif
        if (!fmhaKernelPreflighted
            && (mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_V2 || useSharedFp8FMHAV2)
            && (!usePaddingContextMask || mCanImplementPaddingFMHA))
        {
            bool const useSmallD64 = executionMode == AttentionExecutionMode::kNORMAL_PREFILL;
            bool const boundedSwaDense = useDenseSwaCurrentKV || useSwaChunkAssembly;
            bool const useNativePagedFMHA = !boundedSwaDense && !usePaddingContextMask && !mEnableFp8KVCache;
            int32_t const boundedKVSeqLen = executionMode == AttentionExecutionMode::kCHUNKED_PREFILL
                ? mSlidingWindowSize + runtimeSeqLen
                : runtimeSeqLen;
            int32_t const preflightKVSeqLen = boundedSwaDense ? boundedKVSeqLen
                : useNativePagedFMHA                          ? kvCacheCapacity
                : useSmallD64                                 ? runtimeSeqLen
                                                              : kvCacheCapacity;
            CuteDslFMHAV2Runner runner(
                mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, preflightKVSeqLen, useSmallD64);
            int32_t const slidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize - 1 : INT_MAX;
            bool const preflightSucceeded = usePaddingContextMask ? runner.preflightPadding(stream)
                : useNativePagedFMHA
                ? runner.preflightPaged(stream, slidingWindow, prefillSkipTable, prefillSkipTableLen)
                : runner.preflightLlm(stream, slidingWindow);
            if (!preflightSucceeded)
            {
                return -1;
            }
            fmhaKernelPreflighted = true;
        }
        if (!fmhaKernelPreflighted)
        {
            LOG_ERROR("AttentionPlugin: selected prefill kernel is unavailable (paddingMask=%d, D=%d, SM=%d).",
                usePaddingContextMask ? 1 : 0, mHeadSize, mSMVersion);
            return -1;
        }

        // Allocate workspace tensors for cumulative sequence lengths.
        rt::Tensor cuQSeqLensTensor
            = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);
        rt::Tensor cuKVSeqLensTensor
            = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);
        rt::Tensor kvCacheEndIdxsTensor
            = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize}, DataType::kINT32);

        // Padded cu_kv_seqlens for CuTe DSL FMHA bottom_right_align (see utilKernels.h for details).
        rt::Tensor paddedCuKVSeqLensTensor
            = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);
        kernel::calCuQCuKVSeqLensAndKVEndIdxs(contextLengthTensor, kvCacheStartIdxTensor, cuQSeqLensTensor,
            cuKVSeqLensTensor, kvCacheEndIdxsTensor, paddedCuKVSeqLensTensor, runtimeSeqLen, stream);

        auto splitLenForFallback = [&]() {
            // Compact to runtimeSeqLen only when prefill starts from an empty KV cache. Chunked prefill and
            // DiffusionGemma denoise/commit (normal-prefill mode over a cached prefix) must keep the prefix.
            bool const compact = phase == rt::ExecutionPhase::kContextPrefill;
            return compact ? runtimeSeqLen : 0;
        };
        [[maybe_unused]] auto physicalLenForFallback
            = [&]() { return splitLenForFallback() > 0 ? runtimeSeqLen : kvCacheCapacity; };
        // FMHA-v2 D512 padding reuses the vision-block kernel with full-KV intervals.
        auto paddingBlockRanges = [&]() -> std::pair<int32_t const*, int32_t const*> {
            if (mHeadSize != 512)
            {
                return {nullptr, nullptr};
            }
            rt::Tensor blockBegin
                = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen}, DataType::kINT32);
            rt::Tensor blockEnd
                = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen}, DataType::kINT32);
            kernel::launchBuildPaddingBlockRanges(cuKVSeqLensTensor.dataPointer<int32_t>(),
                blockBegin.dataPointer<int32_t>(), blockEnd.dataPointer<int32_t>(), runtimeBatchSize, runtimeSeqLen,
                stream);
            return {blockBegin.dataPointer<int32_t>(), blockEnd.dataPointer<int32_t>()};
        };

        // --- Shared KV prefill: Q gets RoPE, K/V read from donor layer's cache ---
        if (sharedKV)
        {
            if (sharedKVWithCurrent)
            {
                qInputTensor = assignTensorFromWorkspace(
                    alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
                if (useDenseSwaCurrentKV)
                {
                    kInputTensor = assignTensorFromWorkspace(alignedWorkspacePtr,
                        {runtimeBatchSize, runtimeSeqLen, mNumKVHeads, mHeadSize}, DataType::kHALF);
                    vInputTensor = assignTensorFromWorkspace(alignedWorkspacePtr,
                        {runtimeBatchSize, runtimeSeqLen, mNumKVHeads, mHeadSize}, DataType::kHALF);
                }
                kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor,
                    rt::OptionalInputTensor{kvCacheEndIdxsTensor}, rt::OptionalInputTensor{}, packedQKVTensor,
                    qInputTensor, kvCacheTensor, kScale, vScale, stream, pageTable, maxPagesPerSeq,
                    useDenseSwaCurrentKV ? kInputTensor.rawPointer() : nullptr,
                    useDenseSwaCurrentKV ? vInputTensor.rawPointer() : nullptr, nullptr /* fp8QOut */,
                    1.0F /* qScale */, qNormGammaDevicePtr, kNormGammaDevicePtr, rmsNormEpsVal, qkNormPostRope,
                    rt::OptionalInputTensor{cuQSeqLensTensor}, false /* writeKVCache */, enableRopePdl,
                    true /* tokenAlignedRope */);
            }
            else
            {
                check::check(!useDenseSwaCurrentKV,
                    "SWA shared-KV prefill requires the donor's current K/V in the packed input.");
                // Shared-KV: RoPE Q in-place only; the donor layer's cache is already populated.
                qInputTensor = aliasPackedAsQInput();
                kernel::launchApplyRopeQOnly(ropeCosSinTensor, qInputTensor, stream);
            }

            if (useDenseSwaCurrentKV)
            {
                rt::Coords const currentKVShape{runtimeBatchSize, runtimeSeqLen, mNumKVHeads, mHeadSize};
                rt::Tensor currentKInputTensor(
                    kInputTensor.rawPointer(), currentKVShape, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
                rt::Tensor currentVInputTensor(
                    vInputTensor.rawPointer(), currentKVShape, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
                half const* kData = currentKInputTensor.dataPointer<half>();
                half const* vData = currentVInputTensor.dataPointer<half>();
                int32_t kvSeqLen = runtimeSeqLen;
                if (executionMode == AttentionExecutionMode::kCHUNKED_PREFILL)
                {
                    kernel::calSWAChunkedPrefillMetadata(contextLengthTensor, kvCacheStartIdxTensor, cuQSeqLensTensor,
                        cuKVSeqLensTensor, kvCacheEndIdxsTensor, paddedCuKVSeqLensTensor, runtimeSeqLen,
                        mSlidingWindowSize, stream);
                    kvSeqLen = mSlidingWindowSize + runtimeSeqLen;
                    rt::Tensor kvWorkspaceTensor = assignTensorFromWorkspace(
                        alignedWorkspacePtr, {runtimeBatchSize, 2, mNumKVHeads, kvSeqLen, mHeadSize}, DataType::kHALF);
                    size_t const halfSize = static_cast<size_t>(runtimeBatchSize) * kvSeqLen * mNumKVHeads * mHeadSize;
                    half* const workspace = kvWorkspaceTensor.dataPointer<half>();
                    kInputTensor = rt::Tensor(workspace, {runtimeBatchSize, kvSeqLen, mNumKVHeads, mHeadSize},
                        rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
                    vInputTensor
                        = rt::Tensor(workspace + halfSize, {runtimeBatchSize, kvSeqLen, mNumKVHeads, mHeadSize},
                            rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
                    kernel::assemblePagedSWAChunkedPrefillFMHAKV(kvCacheTensor, kvPageTableTensor, currentKInputTensor,
                        currentVInputTensor, contextLengthTensor, kvCacheStartIdxTensor, kInputTensor, vInputTensor,
                        mSlidingWindowSize, kScale, vScale, stream);
                    kData = kInputTensor.dataPointer<half>();
                    vData = vInputTensor.dataPointer<half>();
                }

                CuteDslFMHAV2Runner runner(
                    mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvSeqLen, false);
                if (!runner.run(qInputTensor.dataPointer<half>(), kData, vData,
                        attentionOutputTensor.dataPointer<half>(), paddedCuKVSeqLensTensor.dataPointer<int32_t>(),
                        stream, mAttentionScale, mSlidingWindowSize - 1))
                {
                    return -1;
                }
                return 0;
            }

            // Run FMHA reading from the donor's KV cache (bound to this layer's KV cache input).
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
            if (mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL
                && (!usePaddingContextMask || mCanImplementPaddingFMHA) && !mEnableFp8KVCache)
            {
                // CuTe DSL FMHA reads the paged KV pool natively.
                // Padding attention is dense and consumes actual KV lengths;
                // causal attention keeps bottom-right alignment with padded lengths.
                int32_t const slidingWindow
                    = (mSlidingWindowSize > 0 && !usePaddingContextMask) ? mSlidingWindowSize - 1 : INT_MAX;
                int32_t const* const fmhaCuKVSeqLens = usePaddingContextMask
                    ? cuKVSeqLensTensor.dataPointer<int32_t>()
                    : paddedCuKVSeqLensTensor.dataPointer<int32_t>();
                CuteDslFMHARunner runner(
                    mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);
                if (!runner.runPaged(qInputTensor.dataPointer<half>(), // Q  [b, s_q, h_q, d]
                        kvCacheTensor.rawPointer(),                // paged KV pool [2, numPages, 128, h_k, d] (donor)
                        pageTable,                                 // page table [b, 2, maxPagesPerSeq]
                        attentionOutputTensor.dataPointer<half>(), // O  [b, s_q, h_q, d]
                        fmhaCuKVSeqLens, 2 * numPages, maxPagesPerSeq, rt::kTOKENS_PER_PAGE,
                        kvCacheTensor.getDataType(), stream, mAttentionScale, slidingWindow, /*fp8Input=*/false, 1.0F,
                        kScale, vScale, !usePaddingContextMask, prefillSkipTable, prefillSkipTableLen,
                        cuKVSeqLensTensor.dataPointer<int32_t>()))
                {
                    return -1;
                }
            }
            else
#endif
                if ((mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_V2 || useSharedFp8FMHAV2)
                    && (!usePaddingContextMask || mCanImplementPaddingFMHA))
            {
                int32_t const slidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize - 1 : INT_MAX;
                if (!usePaddingContextMask && !mEnableFp8KVCache)
                {
                    LOG_DEBUG(
                        "AttentionPlugin: shared-KV %s prefill via native paged FP16 FMHA-v2 "
                        "(B=%d, Sq=%d, capacity=%d, Hq=%d, Hkv=%d, D=%d)",
                        executionMode == AttentionExecutionMode::kCHUNKED_PREFILL ? "chunked" : "normal",
                        runtimeBatchSize, runtimeSeqLen, kvCacheCapacity, mNumQHeads, mNumKVHeads, mHeadSize);
                    // Non-padding shared-KV prefill routes to native-paged FMHA-v2.
                    CuteDslFMHAV2Runner runner(
                        mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);
                    if (!runner.runPaged(qInputTensor.dataPointer<half>(), kvCacheTensor.rawPointer(), pageTable,
                            attentionOutputTensor.dataPointer<half>(), cuQSeqLensTensor.dataPointer<int32_t>(),
                            cuKVSeqLensTensor.dataPointer<int32_t>(), 2 * numPages, maxPagesPerSeq,
                            rt::kTOKENS_PER_PAGE, stream, mAttentionScale, slidingWindow, prefillSkipTable,
                            prefillSkipTableLen))
                    {
                        return -1;
                    }
                }
                else
                {
                    // FP8 donor pages are dequantized into split FP16 K/V before FMHA-v2. The padding kernel uses
                    // the same split contract for either pool dtype.
                    int32_t const splitSeqLen = splitLenForFallback();
                    auto [kSplit, vSplit] = splitPagedKV(kvCacheTensor, pageTable,
                        kvCacheEndIdxsTensor.dataPointer<int32_t>(), maxPagesPerSeq, alignedWorkspacePtr,
                        runtimeBatchSize, mNumKVHeads, kvCacheCapacity, mHeadSize, splitSeqLen, kScale, vScale, stream);
                    int32_t const kvExtent = physicalLenForFallback();
                    auto const [blockBegin, blockEnd]
                        = usePaddingContextMask ? paddingBlockRanges() : std::pair<int32_t const*, int32_t const*>{};
                    CuteDslFMHAV2Runner runner(
                        mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvExtent);
                    bool const ranKernel = usePaddingContextMask
                        ? runner.runPadding(qInputTensor.dataPointer<half>(), kSplit.dataPointer<half>(),
                              vSplit.dataPointer<half>(), attentionOutputTensor.dataPointer<half>(),
                              cuQSeqLensTensor.dataPointer<int32_t>(), cuKVSeqLensTensor.dataPointer<int32_t>(), stream,
                              mAttentionScale, blockBegin, blockEnd)
                        : runner.run(qInputTensor.dataPointer<half>(), kSplit.dataPointer<half>(),
                              vSplit.dataPointer<half>(), attentionOutputTensor.dataPointer<half>(),
                              paddedCuKVSeqLensTensor.dataPointer<int32_t>(), stream, mAttentionScale, slidingWindow);
                    if (!ranKernel)
                    {
                        return -1;
                    }
                }
            }
            else
            {
                LOG_ERROR(
                    "AttentionPlugin: selected shared-KV prefill kernel is unavailable (paddingMask=%d, D=%d, SM=%d).",
                    usePaddingContextMask ? 1 : 0, mHeadSize, mSMVersion);
                return -1;
            }
            return 0;
        }

        if (useSwaChunkAssembly)
        {
            kernel::calSWAChunkedPrefillMetadata(contextLengthTensor, kvCacheStartIdxTensor, cuQSeqLensTensor,
                cuKVSeqLensTensor, kvCacheEndIdxsTensor, paddedCuKVSeqLensTensor, runtimeSeqLen, mSlidingWindowSize,
                stream);

            qInputTensor = assignTensorFromWorkspace(
                alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
            kInputTensor = assignTensorFromWorkspace(
                alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumKVHeads, mHeadSize}, DataType::kHALF);
            vInputTensor = assignTensorFromWorkspace(
                alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumKVHeads, mHeadSize}, DataType::kHALF);

            // Split packed QKV into current-chunk scratch and persist only logical pages mapped by the SWA table.
            // Unmapped earlier chunk pages are intentionally skipped: the temporary W + chunk source still owns them.
            kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor, rt::OptionalInputTensor{kvCacheEndIdxsTensor},
                rt::OptionalInputTensor{}, packedQKVTensor, qInputTensor, kvCacheTensor, kScale, vScale, stream,
                pageTable, maxPagesPerSeq, kInputTensor.rawPointer(), vInputTensor.rawPointer(), nullptr /* fp8QOut */,
                1.0F /* qScale */, qNormGammaDevicePtr, kNormGammaDevicePtr, rmsNormEpsVal, qkNormPostRope,
                std::nullopt, true /* writeKVCache */, enableRopePdl, true /* tokenAlignedRope */);

            int32_t const workspaceSeqLen = mSlidingWindowSize + runtimeSeqLen;
            rt::Tensor kvWorkspaceTensor = assignTensorFromWorkspace(
                alignedWorkspacePtr, {runtimeBatchSize, 2, mNumKVHeads, workspaceSeqLen, mHeadSize}, DataType::kHALF);
            size_t const halfSize = static_cast<size_t>(runtimeBatchSize) * workspaceSeqLen * mNumKVHeads * mHeadSize;
            half* const workspace = kvWorkspaceTensor.dataPointer<half>();
            rt::Tensor kWorkspaceTensor(workspace, {runtimeBatchSize, workspaceSeqLen, mNumKVHeads, mHeadSize},
                rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
            rt::Tensor vWorkspaceTensor(workspace + halfSize,
                {runtimeBatchSize, workspaceSeqLen, mNumKVHeads, mHeadSize}, rt::DeviceType::kGPU,
                trt_edgellm::DataType::kHALF);
            kernel::assemblePagedSWAChunkedPrefillFMHAKV(kvCacheTensor, kvPageTableTensor, kInputTensor, vInputTensor,
                contextLengthTensor, kvCacheStartIdxTensor, kWorkspaceTensor, vWorkspaceTensor, mSlidingWindowSize,
                kScale, vScale, stream);

            CuteDslFMHAV2Runner runner(
                mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, workspaceSeqLen, false);
            if (!runner.run(qInputTensor.dataPointer<half>(), kWorkspaceTensor.dataPointer<half>(),
                    vWorkspaceTensor.dataPointer<half>(), attentionOutputTensor.dataPointer<half>(),
                    paddedCuKVSeqLensTensor.dataPointer<int32_t>(), stream, mAttentionScale, mSlidingWindowSize - 1))
            {
                return -1;
            }
            return 0;
        }

        // --- Own KV prefill: RoPE Q+K, write K/V to cache, then run attention kernel ---

#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
        if (mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL
            && (!usePaddingContextMask || mCanImplementPaddingFMHA))
        {
            // CuTe DSL FMHA uses a single packed kernel that splits QKV, applies RoPE
            // (+ optional fused qk_norm RMSNorm), and writes K/V to the paged pool.
            float const qScale = mQkvScales[0];
            // windowSizeLeft excludes the query itself; sliding_window_size
            // counts it (last W keys, the XQA/HF convention), so pass W - 1.
            int32_t const slidingWindow
                = (mSlidingWindowSize > 0 && !usePaddingContextMask) ? mSlidingWindowSize - 1 : INT_MAX;
            int32_t const* const fmhaCuKVSeqLens = usePaddingContextMask
                ? cuKVSeqLensTensor.dataPointer<int32_t>()
                : paddedCuKVSeqLensTensor.dataPointer<int32_t>();
            CuteDslFMHARunner runner(
                mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);

            // qScratch: roped Q [B, S, Hq, D], always allocated. On the FP8 path the
            // kernel writes the quantized Q to fp8QTensor instead and qScratch is unused.
            qInputTensor = assignTensorFromWorkspace(
                alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
            if (mEnableFp8KVCache)
            {
                // FP8 Q workspace: packed RoPE kernel quantizes roped Q to FP8 using calibrated qScale.
                rt::Tensor fp8QTensor = assignTensorFromWorkspace(
                    alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kFP8);

                // Packed kernel: RoPE Q to FP8 (fp8QTensor), RoPE K + write FP8 K/V to the
                // paged pool. The CuTe DSL runner reads K/V from the pool only.
                kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor,
                    rt::OptionalInputTensor{kvCacheEndIdxsTensor}, rt::OptionalInputTensor{}, packedQKVTensor,
                    qInputTensor, kvCacheTensor, kScale, vScale, stream, pageTable, maxPagesPerSeq,
                    nullptr /* kScratch */, nullptr /* vScratch */, fp8QTensor.rawPointer(), qScale,
                    qNormGammaDevicePtr, kNormGammaDevicePtr, rmsNormEpsVal, qkNormPostRope,
                    rt::OptionalInputTensor{cuQSeqLensTensor}, true /* writeKVCache */, enableRopePdl,
                    true /* tokenAlignedRope */);

                if (!runner.runPaged(fp8QTensor.rawPointer(),      // Q  [b, s_q, h_q, d] FP8
                        kvCacheTensor.rawPointer(),                // paged KV pool [2, numPages, 128, h_k, d] FP8
                        pageTable,                                 // page table [b, 2, maxPagesPerSeq]
                        attentionOutputTensor.dataPointer<half>(), // O  [b, s_q, h_q, d] FP16
                        fmhaCuKVSeqLens, 2 * numPages, maxPagesPerSeq, rt::kTOKENS_PER_PAGE,
                        fromTrt(selectKvCacheDataType(mEnableFp8KVCache)), stream, mAttentionScale, slidingWindow,
                        /*fp8Input=*/true, qScale, kScale, vScale, !usePaddingContextMask))
                {
                    return -1;
                }
            }
            else
            {
                // FP16 path: RoPE Q to qScratch, write FP16 K/V to the paged pool.
                // The CuTe DSL runner reads K/V from the pool only.
                kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor,
                    rt::OptionalInputTensor{kvCacheEndIdxsTensor}, rt::OptionalInputTensor{}, packedQKVTensor,
                    qInputTensor, kvCacheTensor, kScale, vScale, stream, pageTable, maxPagesPerSeq,
                    nullptr /* kScratch */, nullptr /* vScratch */, nullptr /* fp8QOut */, 1.0f /* qScale */,
                    qNormGammaDevicePtr, kNormGammaDevicePtr, rmsNormEpsVal, qkNormPostRope,
                    rt::OptionalInputTensor{cuQSeqLensTensor}, true /* writeKVCache */, enableRopePdl,
                    true /* tokenAlignedRope */);

                if (!runner.runPaged(qInputTensor.dataPointer<half>(), // Q  [b, s_q, h_q, d]
                        kvCacheTensor.rawPointer(),                    // paged KV pool [2, numPages, 128, h_k, d]
                        pageTable,                                     // page table [b, 2, maxPagesPerSeq]
                        attentionOutputTensor.dataPointer<half>(),     // O  [b, s_q, h_q, d]
                        fmhaCuKVSeqLens, 2 * numPages, maxPagesPerSeq, rt::kTOKENS_PER_PAGE,
                        kvCacheTensor.getDataType(), stream, mAttentionScale, slidingWindow, /*fp8Input=*/false, 1.0F,
                        1.0F, 1.0F, !usePaddingContextMask, prefillSkipTable, prefillSkipTableLen,
                        cuKVSeqLensTensor.dataPointer<int32_t>()))
                {
                    return -1;
                }
            }
        }
        else
#endif
            if (mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_V2
                && (!usePaddingContextMask || mCanImplementPaddingFMHA))
        {
            int32_t const slidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize - 1 : INT_MAX;

            // FMHA-v2 always reads the RoPE-transformed Q from scratch.
            qInputTensor = assignTensorFromWorkspace(
                alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);

            if (!mEnableFp8KVCache && !usePaddingContextMask && !useBoundedSwaCache)
            {
                // FP16 causal/sliding prefill writes K/V to the cache and reads that paged pool directly.
                kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor,
                    rt::OptionalInputTensor{kvCacheEndIdxsTensor}, rt::OptionalInputTensor{}, packedQKVTensor,
                    qInputTensor, kvCacheTensor, kScale, vScale, stream, pageTable, maxPagesPerSeq,
                    nullptr /* kScratch */, nullptr /* vScratch */, nullptr /* fp8QOut */, 1.0f /* qScale */,
                    qNormGammaDevicePtr, kNormGammaDevicePtr, rmsNormEpsVal, qkNormPostRope,
                    rt::OptionalInputTensor{cuQSeqLensTensor}, true /* writeKVCache */, enableRopePdl,
                    true /* tokenAlignedRope */);

                LOG_DEBUG(
                    "AttentionPlugin: own-KV %s prefill via native paged FP16 FMHA-v2 "
                    "(B=%d, Sq=%d, capacity=%d, Hq=%d, Hkv=%d, D=%d)",
                    executionMode == AttentionExecutionMode::kCHUNKED_PREFILL ? "chunked" : "normal", runtimeBatchSize,
                    runtimeSeqLen, kvCacheCapacity, mNumQHeads, mNumKVHeads, mHeadSize);
                CuteDslFMHAV2Runner runner(
                    mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);
                if (!runner.runPaged(qInputTensor.dataPointer<half>(), kvCacheTensor.rawPointer(), pageTable,
                        attentionOutputTensor.dataPointer<half>(), cuQSeqLensTensor.dataPointer<int32_t>(),
                        cuKVSeqLensTensor.dataPointer<int32_t>(), 2 * numPages, maxPagesPerSeq, rt::kTOKENS_PER_PAGE,
                        stream, mAttentionScale, slidingWindow, prefillSkipTable, prefillSkipTableLen))
                {
                    return -1;
                }
            }
            else
            {
                // Preserve dense K/V for bounded normal prefill, FP8 KV, and the independent padding contract.
                bool const gatherKV
                    = executionMode == AttentionExecutionMode::kCHUNKED_PREFILL || usePaddingContextMask;
                if (gatherKV)
                {
                    kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor,
                        rt::OptionalInputTensor{kvCacheEndIdxsTensor}, rt::OptionalInputTensor{}, packedQKVTensor,
                        qInputTensor, kvCacheTensor, kScale, vScale, stream, pageTable, maxPagesPerSeq,
                        nullptr /* kScratch */, nullptr /* vScratch */, nullptr /* fp8QOut */, 1.0f /* qScale */,
                        qNormGammaDevicePtr, kNormGammaDevicePtr, rmsNormEpsVal, qkNormPostRope,
                        rt::OptionalInputTensor{cuQSeqLensTensor}, true /* writeKVCache */, enableRopePdl,
                        true /* tokenAlignedRope */);

                    int32_t const splitSeqLen = splitLenForFallback();
                    auto [kSplit, vSplit] = splitPagedKV(kvCacheTensor, pageTable,
                        kvCacheEndIdxsTensor.dataPointer<int32_t>(), maxPagesPerSeq, alignedWorkspacePtr,
                        runtimeBatchSize, mNumKVHeads, kvCacheCapacity, mHeadSize, splitSeqLen, kScale, vScale, stream);
                    auto const [blockBegin, blockEnd]
                        = usePaddingContextMask ? paddingBlockRanges() : std::pair<int32_t const*, int32_t const*>{};
                    CuteDslFMHAV2Runner runner(mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen,
                        physicalLenForFallback(), splitSeqLen > 0);
                    bool const ranKernel = usePaddingContextMask
                        ? runner.runPadding(qInputTensor.dataPointer<half>(), kSplit.dataPointer<half>(),
                              vSplit.dataPointer<half>(), attentionOutputTensor.dataPointer<half>(),
                              cuQSeqLensTensor.dataPointer<int32_t>(), cuKVSeqLensTensor.dataPointer<int32_t>(), stream,
                              mAttentionScale, blockBegin, blockEnd)
                        : runner.run(qInputTensor.dataPointer<half>(), kSplit.dataPointer<half>(),
                              vSplit.dataPointer<half>(), attentionOutputTensor.dataPointer<half>(),
                              paddedCuKVSeqLensTensor.dataPointer<int32_t>(), stream, mAttentionScale, slidingWindow);
                    if (!ranKernel)
                    {
                        return -1;
                    }
                }
                else
                {
                    // Normal bounded/FP8 prefill keeps the dense FP16 scratch contract.
                    kInputTensor = assignTensorFromWorkspace(alignedWorkspacePtr,
                        {runtimeBatchSize, runtimeSeqLen, mNumKVHeads, mHeadSize}, DataType::kHALF);
                    vInputTensor = assignTensorFromWorkspace(alignedWorkspacePtr,
                        {runtimeBatchSize, runtimeSeqLen, mNumKVHeads, mHeadSize}, DataType::kHALF);
                    kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor, rt::OptionalInputTensor{},
                        rt::OptionalInputTensor{}, packedQKVTensor, qInputTensor, kvCacheTensor, kScale, vScale, stream,
                        pageTable, maxPagesPerSeq, kInputTensor.rawPointer(), vInputTensor.rawPointer(),
                        nullptr /* fp8QOut */, 1.0f /* qScale */, qNormGammaDevicePtr, kNormGammaDevicePtr,
                        rmsNormEpsVal, qkNormPostRope, rt::OptionalInputTensor{cuQSeqLensTensor},
                        true /* writeKVCache */, enableRopePdl, true /* tokenAlignedRope */);

                    CuteDslFMHAV2Runner runner(
                        mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, runtimeSeqLen, true);
                    if (!runner.run(qInputTensor.dataPointer<half>(), kInputTensor.dataPointer<half>(),
                            vInputTensor.dataPointer<half>(), attentionOutputTensor.dataPointer<half>(),
                            paddedCuKVSeqLensTensor.dataPointer<int32_t>(), stream, mAttentionScale, slidingWindow))
                    {
                        return -1;
                    }
                }
            }
        }
        else
        {
            LOG_ERROR("AttentionPlugin: selected own-KV prefill kernel is unavailable (paddingMask=%d, D=%d, SM=%d).",
                usePaddingContextMask ? 1 : 0, mHeadSize, mSMVersion);
            return -1;
        }
    }
    // ==================== Decode path (vanilla or tree) ====================
    else
    {
        DecodeRows const decodeRows{packedQKVTensor, ropeCosSinTensor, contextLengthTensor, attentionPosIdTensor,
            attentionMaskTensor, kvCacheTensor, attentionOutputTensor,
            static_cast<int32_t const*>(inputs[kIN_QUERY_LENGTH_IDX]), pageTable, maxPagesPerSeq, qNormGammaDevicePtr,
            kNormGammaDevicePtr, attentionSinkDevicePtr, decodeSkipTable, decodeSkipTableLen, useExplicitPositionIds};
        return enqueueDecodeRows(
            decodeRows, executionMode == AttentionExecutionMode::kTREE_DECODING, alignedWorkspacePtr, stream);
    }
    return 0;
}

int32_t AttentionPlugin::enqueueDecodeRows(
    DecodeRows const& rows, bool tree, std::byte* alignedWorkspacePtr, cudaStream_t stream)
{
    rt::Coords const packedShape = rows.packedQKV.getShape();
    int32_t const runtimeBatchSize = static_cast<int32_t>(packedShape[0]);
    int32_t const runtimeSeqLen = static_cast<int32_t>(packedShape[1]);
    bool const sharedKV = mEnableKVShared != 0;
    bool const sharedKVWithCurrent = sharedKV && packedShape[2] == mNumQHeads + 2 * mNumKVHeads;
    int32_t const numPages = static_cast<int32_t>(rows.kvCache.getShape()[1]);
    int32_t const kvCacheCapacity = rows.maxPagesPerSeq * rt::kTOKENS_PER_PAGE;
    float const kScale = mQkvScales[1];
    float const vScale = mQkvScales[2];
    bool const qkNormPostRope = mQKNormPostRope != 0;
    bool const enableRopePdl = requestRopePdl() && mSMVersion >= kROPE_MIN_PDL_SM_VERSION;
    rt::Tensor qInputTensor;

    // RoPE setup: sharedKV → Q only; own-KV → packed kernel (Q to qScratch + KV cache
    // write). Decode reads K/V from the KV cache via XQA — no scratch K/V needed.
    if (sharedKV)
    {
        if (sharedKVWithCurrent)
        {
            qInputTensor = assignTensorFromWorkspace(
                alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
            rt::OptionalInputTensor const tokenPosIds = (tree || rows.explicitPositionIds)
                ? rt::OptionalInputTensor{rows.positionIds}
                : rt::OptionalInputTensor{};
            kernel::launchApplyRopeFromPackedToSplit(rows.ropeCosSin, rt::OptionalInputTensor{rows.sequenceLengths},
                tokenPosIds, rows.packedQKV, qInputTensor, rows.kvCache, kScale, vScale, stream, rows.pageTable,
                rows.maxPagesPerSeq, nullptr /* kScratch */, nullptr /* vScratch */, nullptr /* fp8QOut */,
                1.0F /* qScale */, rows.qNormGamma, rows.kNormGamma, mRmsNormEps, qkNormPostRope, std::nullopt,
                false /* writeKVCache */, enableRopePdl, true /* tokenAlignedRope */);
        }
        else
        {
            qInputTensor = rt::Tensor(const_cast<void*>(rows.packedQKV.rawPointer()),
                rt::Coords{runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, rt::DeviceType::kGPU,
                rows.packedQKV.getDataType());
            kernel::launchApplyRopeQOnly(rows.ropeCosSin, qInputTensor, stream);
        }
        // The Q-only path has no fused-norm kernel; enable_qk_norm + enable_kv_shared is
        // rejected at plugin construction.
    }
    else if (tree)
    {
        // Tree decoding: pass tokenPosIds for per-token RoPE positions.
        // Allocate qScratch (roped Q output written by launchApplyRopeFromPackedToSplit).
        qInputTensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
        kernel::launchApplyRopeFromPackedToSplit(rows.ropeCosSin, rt::OptionalInputTensor{rows.sequenceLengths},
            rt::OptionalInputTensor{rows.positionIds}, rows.packedQKV, qInputTensor, rows.kvCache, kScale, vScale,
            stream, rows.pageTable, rows.maxPagesPerSeq, nullptr /* kScratch */, nullptr /* vScratch */,
            nullptr /* fp8QOut */, 1.0f /* qScale */, rows.qNormGamma, rows.kNormGamma, mRmsNormEps, qkNormPostRope,
            rt::OptionalInputTensor{}, true /* writeKVCache */, enableRopePdl, true /* tokenAlignedRope */);
    }
    else
    {
        // Vanilla decoding: derive RoPE position from rows.sequenceLengths (kvCacheEndLens).
        // Allocate qScratch (roped Q output written by launchApplyRopeFromPackedToSplit).
        qInputTensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
        kernel::launchApplyRopeFromPackedToSplit(rows.ropeCosSin, rt::OptionalInputTensor{rows.sequenceLengths},
            rt::OptionalInputTensor{}, rows.packedQKV, qInputTensor, rows.kvCache, kScale, vScale, stream,
            rows.pageTable, rows.maxPagesPerSeq, nullptr /* kScratch */, nullptr /* vScratch */, nullptr /* fp8QOut */,
            1.0f /* qScale */, rows.qNormGamma, rows.kNormGamma, mRmsNormEps, qkNormPostRope, rt::OptionalInputTensor{},
            true /* writeKVCache */, enableRopePdl, true /* tokenAlignedRope */);
    }

    // Vision-block decode goes to XQA (a hard construction-time
    // requirement): newly generated tokens are text, so decode is pure
    // causal/sliding — identical masking to the non-vision path.
    // Cache-state parity: the vision prefill paths write the cache with
    // launchApplyRopeWriteKV (roped K + original V into the paged pool)
    // and the decode RoPE above appended the
    // new token the same way as the non-vision path, so XQA reads exactly
    // the cache state it would see without vision.

    half* const outputPtr = rows.output.dataPointer<half>();
    half* const qPtr = qInputTensor.dataPointer<half>();
    int32_t const* const seqLensPtr = rows.sequenceLengths.dataPointer<int32_t>();
    bool ctmLaunched = false;
    if ((tree ? useCtmTreeDecode() : useCtmDecode()) && mCtmSemaphores != nullptr)
    {
        size_t const ctmWorkspaceElems = tree
            ? CtmDecodeAttentionRunner::treeWorkspaceElems(
                  runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads, mHeadSize)
            : CtmDecodeAttentionRunner::workspaceElems(runtimeBatchSize, mNumQHeads, mHeadSize);
        rt::Tensor ctmWorkspaceTensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, {static_cast<int64_t>(ctmWorkspaceElems)}, DataType::kFLOAT);
        CtmDecodeParams ctmParams{};
        ctmParams.output = outputPtr;
        ctmParams.q = qPtr;
        ctmParams.kvPool = rows.kvCache.rawPointer();
        ctmParams.pageTable = rows.pageTable;
        ctmParams.seqLens = seqLensPtr;
        ctmParams.sinks = rows.attentionSinks;
        ctmParams.workspace = ctmWorkspaceTensor.rawPointer();
        ctmParams.semaphores = mCtmSemaphores;
        ctmParams.batchSize = runtimeBatchSize;
        ctmParams.numQHeads = mNumQHeads;
        ctmParams.numKVHeads = mNumKVHeads;
        ctmParams.headSize = mHeadSize;
        ctmParams.numPages = numPages;
        ctmParams.maxPagesPerSeq = rows.maxPagesPerSeq;
        ctmParams.slidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize : 0;
        ctmParams.attentionScale = mAttentionScale;
        ctmParams.fp8KVCache = mEnableFp8KVCache != 0;
        ctmParams.kScale = kScale;
        ctmParams.vScale = vScale;
        ctmParams.fallbackAvailable = !mCtmDecodeOnly;
        ctmParams.skipSoftmaxScaleTable = rows.skipSoftmaxDecodeScaleTable;
        ctmParams.skipSoftmaxScaleTableLen = rows.skipSoftmaxDecodeScaleTableLen;
        ctmParams.stream = stream;
        if (tree)
        {
            ctmParams.qSeqLen = runtimeSeqLen;
            ctmParams.treeMask = rows.treeMask.dataPointer<int32_t>();
            ctmParams.maskWords = static_cast<int32_t>(rows.treeMask.getShape()[2]);
            ctmParams.contiguousQuerySwa = mEnableContiguousQuerySwa != 0 && mSlidingWindowSize > 0;
        }
        ctmLaunched = CtmDecodeAttentionRunner::run(ctmParams);
    }
    if (!ctmLaunched)
    {
        if (mCtmDecodeOnly)
        {
            LOG_ERROR("AttentionPlugin: CTM decode launch failed and XQA is not a decode backend on Blackwell.");
            return -1;
        }
        // XQA decode kernel dispatch (the decode backend on pre-Blackwell SMs).
        auto xqaRunner = DecoderXQARunner(fromTrt(mDataType), fromTrt(selectKvCacheDataType(mEnableFp8KVCache)),
            runtimeBatchSize, mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion);
        XQALaunchParams params = xqaRunner.initXQAParams();
        params.attentionScale = mAttentionScale;
        if (mEnableFp8KVCache)
        {
            params.kScale = kScale;
            params.vScale = vScale;
        }
        params.output = outputPtr;
        params.qInputPtr = qPtr;
        // Paged KV cache (layout-1): pool base + page table. maxNbPagesPerSeq = capacity / tokensPerPage,
        // so capacity is the padded per-slot capacity (maxPagesPerSeq * kTOKENS_PER_PAGE).
        params.kvCache.data = rows.kvCache.rawPointer();
        params.kvCache.sequence_lengths = seqLensPtr;
        params.kvCache.capacity = static_cast<uint32_t>(kvCacheCapacity);
        params.kvCache.pageList = rows.pageTable;
        params.kvCache.tokensPerPage = static_cast<uint32_t>(rt::kTOKENS_PER_PAGE);
        params.slidingWinSize = mSlidingWindowSize > 0 ? static_cast<uint32_t>(mSlidingWindowSize) : 0U;
        params.attentionSinks = rows.attentionSinks;
        if (tree)
        {
            params.treeAttnMask = rows.treeMask.dataPointer<int32_t>();
            params.qSeqLen = runtimeSeqLen;
            params.qSeqLens = rows.queryLengths;
            params.contiguousQuerySwa = mEnableContiguousQuerySwa && params.slidingWinSize > 0;
            xqaRunner.dispatchSpecDecodeXQAKernel(params, stream);
        }
        else
        {
            if (mEnableTreeAttention == 0)
            {
                params.workspaceSize = xqaRunner.getWorkspaceSize(params.kvCache.capacity);
                params.workspace = alignedWorkspacePtr;
                params.semaphores = mXqaSemaphores.dataPointer<int32_t>();
            }
            xqaRunner.dispatchXQAKernel(params, stream);
        }
    }
    return 0;
}

int32_t AttentionPlugin::enqueueCompactImpl(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    AttentionInputLayout const inputLayout = this->inputLayout();
    if (!supportsCompactPhase(inputDesc[kIN_QKV_IDX].dims.d[1]) || (mCtmDecodeOnly && mCtmSemaphores == nullptr))
    {
        LOG_ERROR("%s: compact execution is not supported for this layer on this GPU.", describeLayer().c_str());
        return -1;
    }
    int32_t const queryStartOffsetIdx = inputLayout.raggedMetadataStart;
    int32_t const attentionSequenceLengthIdx = inputLayout.raggedMetadataStart + 1;
    RaggedPluginMetadata const ragged = decodeRaggedPluginMetadata("AttentionPlugin", inputDesc[kIN_QKV_IDX],
        inputDesc[kIN_QUERY_LENGTH_IDX], inputDesc[queryStartOffsetIdx], inputDesc[inputLayout.raggedMetadataStart + 2],
        inputDesc[inputLayout.raggedMetadataStart + 3], RaggedLayoutSupport::kPaddedAndCompact);
    int32_t const numSequences = ragged.numSequences;
    int32_t const numPrefill = ragged.numContextSequences;
    int32_t const numDecode = ragged.numDecodeSequences;
    int32_t const numTokens = ragged.physicalTokens;
    int32_t const numPrefillTokens = ragged.contextExecutionRows;
    bool useBoundedSwaCache{false};
    if (mSupportsBoundedKVCache)
    {
        PluginTensorDesc const& modeDesc = inputDesc[inputLayout.swaCacheMode];
        check::check(modeDesc.dims.nbDims == 1
                && (modeDesc.dims.d[0] == kFULL_KV_CACHE_MODE_LENGTH
                    || modeDesc.dims.d[0] == kBOUNDED_SWA_CACHE_MODE_LENGTH),
            "AttentionPlugin swa_kv_cache_mode must have runtime shape [0] or [1].");
        useBoundedSwaCache = modeDesc.dims.d[0] == kBOUNDED_SWA_CACHE_MODE_LENGTH;
    }
    if (mEnableAttentionSink && useBoundedSwaCache)
    {
        LOG_ERROR("%s: compact execution is not supported for this layer on this GPU.", describeLayer().c_str());
        return -1;
    }
    bool const sharedKV = mEnableKVShared != 0;
    bool const fp8KV = mEnableFp8KVCache != 0;
    int32_t const qkvChannels = (mNumQHeads + 2 * mNumKVHeads) * mHeadSize;
    int32_t const inputChannels = static_cast<int32_t>(inputDesc[kIN_QKV_IDX].dims.d[1]);
    // Shared-KV layers carry Q only, or Q with the donor's current K/V, and never write the donor cache.
    bool const sharedKVQOnly = sharedKV && inputChannels == mNumQHeads * mHeadSize;
    check::check(inputChannels == qkvChannels || sharedKVQOnly,
        "AttentionPlugin compact execution requires packed QKV with (Hq + 2*Hkv)*head_dim channels, or "
        "Hq*head_dim for a shared-KV layer.");
    // Bounded-SWA-capable layers prefill from an FP16 assembly of the in-window history and the current rows, as
    // their chunked prefill does, when the cache is bounded or FP8.
    bool const useSwaAssembly = mSupportsBoundedKVCache && (useBoundedSwaCache || fp8KV);
    check::check(inputDesc[kIN_KV_CACHE_START_IDX].dims.nbDims == 1
            && inputDesc[kIN_KV_CACHE_START_IDX].dims.d[0] == numSequences
            && inputDesc[attentionSequenceLengthIdx].dims.d[0] == numSequences,
        "AttentionPlugin compact execution requires kvcache_start_index and attention_sequence_lengths of shape [N].");
    check::check(inputDesc[kIN_ROPE_COS_SIN_IDX].dims.d[0] == numTokens,
        "AttentionPlugin compact execution requires one rope cos/sin row per physical token.");

    PluginTensorDesc const& kvCacheDesc = inputDesc[kIN_KV_CACHE_IDX];
    void* const kvCachePtr = sharedKV ? const_cast<void*>(inputs[kIN_KV_CACHE_IDX]) : outputs[kOUT_KV_CACHE_IDX];
    rt::Tensor kvCacheTensor(
        kvCachePtr, rt::toCoords(kvCacheDesc.dims), rt::DeviceType::kGPU, fromTrt(kvCacheDesc.type));
    int32_t const numPages = static_cast<int32_t>(kvCacheDesc.dims.d[1]);
    int32_t const maxPagesPerSeq = static_cast<int32_t>(inputDesc[kIN_KV_PAGE_TABLE_IDX].dims.d[2]);
    check::check(useBoundedSwaCache || numPages >= maxPagesPerSeq,
        "AttentionPlugin full-cache mode requires at least one physical page per logical page-table column.");
    int32_t const kvCacheCapacity = maxPagesPerSeq * rt::kTOKENS_PER_PAGE;
    int32_t const* const pageTable = static_cast<int32_t const*>(inputs[kIN_KV_PAGE_TABLE_IDX]);

    auto* alignedWorkspacePtr = static_cast<std::byte*>(workspace);
    check::check(alignedWorkspacePtr != nullptr
            && reinterpret_cast<uintptr_t>(alignedWorkspacePtr) % static_cast<uintptr_t>(kDEVICE_ALIGNMENT) == 0,
        "AttentionPlugin: workspace pointer is not aligned to device alignment granularity.");
    half const* qNormGammaDevicePtr
        = mEnableQKNorm ? resolveNormGammaInput(inputDesc, inputs, inputLayout.qNormGamma) : nullptr;
    half const* kNormGammaDevicePtr
        = mEnableQKNorm ? resolveNormGammaInput(inputDesc, inputs, inputLayout.kNormGamma) : nullptr;
    [[maybe_unused]] float const* attentionSinkDevicePtr
        = mEnableAttentionSink ? resolveAttentionSinkInput(inputDesc, inputs, inputLayout.attentionSink) : nullptr;
    PluginTensorDesc const& ropeDesc = inputDesc[kIN_ROPE_COS_SIN_IDX];
    int64_t const ropeRowElems = ropeDesc.dims.d[1];
    auto* const ropePtr = static_cast<float*>(const_cast<void*>(inputs[kIN_ROPE_COS_SIN_IDX]));
    auto* const packedPtr = static_cast<half*>(const_cast<void*>(inputs[kIN_QKV_IDX]));
    auto* const outputPtr = static_cast<half*>(outputs[kOUT_ATTENTION_IDX]);
    auto* const offsetsPtr = static_cast<int32_t*>(const_cast<void*>(inputs[queryStartOffsetIdx]));
    auto* const sequenceLengthsPtr = static_cast<int32_t*>(const_cast<void*>(inputs[attentionSequenceLengthIdx]));
    int32_t const packedHeads = inputChannels / mHeadSize;

    if (numPrefill > 0)
    {
        float const qScale = mQkvScales[0];
        float const kScale = mQkvScales[1];
        float const vScale = mQkvScales[2];
        // windowSizeLeft excludes the query itself while sliding_window_size counts it.
        int32_t const prefillSlidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize - 1 : INT_MAX;
        [[maybe_unused]] DataType const prefillInputType = fp8KV ? DataType::kFP8 : DataType::kHALF;

        // Load the prefill kernel module before the first QKV or KV-cache mutation.
        ContextFMHABackend const prefillBackend = selectCompactPrefillBackend();
        if (prefillBackend == ContextFMHABackend::kNONE)
        {
            LOG_ERROR("%s: compact execution is not supported for this layer on this GPU.", describeLayer().c_str());
            return -1;
        }
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
        if (prefillBackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL)
        {
            CuteDslFMHARunner runner(mNumQHeads, mNumKVHeads, mHeadSize, numPrefill, numPrefillTokens, kvCacheCapacity);
            if (!runner.preflightPagedRagged(stream, prefillSlidingWindow, fromTrt(prefillInputType)))
            {
                LOG_ERROR("%s: failed to load the compact prefill kernel module.", describeLayer().c_str());
                return -1;
            }
        }
#endif
        if (prefillBackend == ContextFMHABackend::kCUTE_DSL_FMHA_V2)
        {
            CuteDslFMHAV2Runner runner(
                mNumQHeads, mNumKVHeads, mHeadSize, numPrefill, numPrefillTokens, kvCacheCapacity);
            if (!runner.preflightPagedRagged(stream, prefillSlidingWindow))
            {
                LOG_ERROR("%s: failed to load the compact prefill kernel module.", describeLayer().c_str());
                return -1;
            }
        }

        // The FMHA reads the same offsets and lengths as the KV write: Q intervals from query_start_offsets, KV
        // extents from attention_sequence_lengths. The Q prefix sum written by the helper is unused.
        rt::Tensor const prefillOffsets(
            offsetsPtr, rt::Coords{numPrefill + 1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
        rt::Tensor const prefillEnds(
            sequenceLengthsPtr, rt::Coords{numPrefill}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
        rt::Tensor unusedCuSeqLensTensor
            = assignTensorFromWorkspace(alignedWorkspacePtr, {numPrefill + 1}, DataType::kINT32);
        rt::Tensor cuKVSeqLensTensor
            = assignTensorFromWorkspace(alignedWorkspacePtr, {numPrefill + 1}, DataType::kINT32);
        kernel::calCuQCuKVSeqLens(prefillEnds, rt::Tensor{}, unusedCuSeqLensTensor, cuKVSeqLensTensor, stream);

        rt::Tensor const prefillRope(
            ropePtr, rt::Coords{1, numPrefillTokens, ropeRowElems}, rt::DeviceType::kGPU, fromTrt(ropeDesc.type));
        bool const fp8Q = fp8KV && prefillBackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL;
        rt::Tensor qTensor;
        rt::Tensor fp8QTensor;
        rt::Tensor kScratchTensor;
        rt::Tensor vScratchTensor;
        if (sharedKVQOnly)
        {
            // Same in-place Q RoPE as phases 1-3 of a Q-only shared-KV layer.
            qTensor = rt::Tensor(packedPtr, rt::Coords{1, numPrefillTokens, mNumQHeads, mHeadSize},
                rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
            kernel::launchApplyRopeQOnly(prefillRope, qTensor, stream);
        }
        else
        {
            rt::Tensor const prefillQKV(packedPtr, rt::Coords{1, numPrefillTokens, packedHeads, mHeadSize},
                rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
            qTensor = assignTensorFromWorkspace(
                alignedWorkspacePtr, {1, numPrefillTokens, mNumQHeads, mHeadSize}, DataType::kHALF);
            if (fp8Q)
            {
                fp8QTensor = assignTensorFromWorkspace(
                    alignedWorkspacePtr, {1, numPrefillTokens, mNumQHeads, mHeadSize}, DataType::kFP8);
            }
            if (useSwaAssembly)
            {
                kScratchTensor = assignTensorFromWorkspace(
                    alignedWorkspacePtr, {numPrefillTokens, mNumKVHeads, mHeadSize}, DataType::kHALF);
                vScratchTensor = assignTensorFromWorkspace(
                    alignedWorkspacePtr, {numPrefillTokens, mNumKVHeads, mHeadSize}, DataType::kHALF);
            }
            kernel::launchApplyRopeFromPackedToSplit(prefillRope, rt::OptionalInputTensor{prefillEnds},
                rt::OptionalInputTensor{}, prefillQKV, qTensor, kvCacheTensor, kScale, vScale, stream, pageTable,
                maxPagesPerSeq, useSwaAssembly ? kScratchTensor.rawPointer() : nullptr,
                useSwaAssembly ? vScratchTensor.rawPointer() : nullptr, fp8Q ? fp8QTensor.rawPointer() : nullptr,
                fp8Q ? qScale : 1.0F, qNormGammaDevicePtr, kNormGammaDevicePtr, mRmsNormEps, mQKNormPostRope != 0,
                rt::OptionalInputTensor{prefillOffsets}, !sharedKV /* writeKVCache */,
                requestRopePdl() && mSMVersion >= kROPE_MIN_PDL_SM_VERSION, true /* tokenAlignedRope */,
                true /* compactLayout */, numPrefill);
        }

        // maxQSeqLen is the T_p upper bound; tiles past a sequence's own length exit early. Compact prefill rows run
        // dense: runPagedRagged takes no skip table.
        bool prefillLaunched{false};
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
        if (prefillBackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL)
        {
            CuteDslFMHARunner runner(mNumQHeads, mNumKVHeads, mHeadSize, numPrefill, numPrefillTokens, kvCacheCapacity);
            void const* const prefillQPtr = fp8Q ? fp8QTensor.rawPointer() : qTensor.rawPointer();
            prefillLaunched = runner.runPagedRagged(prefillQPtr, kvCacheTensor.rawPointer(), pageTable, outputPtr,
                offsetsPtr, cuKVSeqLensTensor.dataPointer<int32_t>(), numPrefillTokens, numPrefillTokens, 2 * numPages,
                maxPagesPerSeq, rt::kTOKENS_PER_PAGE, stream, mAttentionScale, prefillSlidingWindow,
                fromTrt(prefillInputType), qScale, kScale, vScale, attentionSinkDevicePtr);
        }
#endif
        if (prefillBackend == ContextFMHABackend::kCUTE_DSL_FMHA_V2)
        {
            void const* prefillPool = kvCacheTensor.rawPointer();
            int32_t const* prefillPageTable = pageTable;
            int32_t prefillFlatPages = 2 * numPages;
            if (useSwaAssembly)
            {
                int64_t const assembledPages = std::min<int64_t>(static_cast<int64_t>(numPrefill) * maxPagesPerSeq,
                    kernel::swaCompactAssembledPages(numPrefill, numPrefillTokens, mSlidingWindowSize));
                rt::Tensor assembledPageTableTensor
                    = assignTensorFromWorkspace(alignedWorkspacePtr, {numPrefill, 2, maxPagesPerSeq}, DataType::kINT32);
                rt::Tensor assembledPoolTensor = assignTensorFromWorkspace(alignedWorkspacePtr,
                    {2, assembledPages, rt::kTOKENS_PER_PAGE, mNumKVHeads, mHeadSize}, DataType::kHALF);
                kernel::assemblePagedSWACompactPrefillFMHAKV(kvCacheTensor, pageTable, maxPagesPerSeq, kScratchTensor,
                    vScratchTensor, offsetsPtr, sequenceLengthsPtr, numPrefill, mSlidingWindowSize, kScale, vScale,
                    assembledPoolTensor, assembledPageTableTensor.dataPointer<int32_t>(), stream);
                prefillPool = assembledPoolTensor.rawPointer();
                prefillPageTable = assembledPageTableTensor.dataPointer<int32_t>();
                prefillFlatPages = static_cast<int32_t>(2 * assembledPages);
            }
            else if (fp8KV)
            {
                // Dequantize the prefill sequences' K/V into dense [Np, cap, Hkv, D] K and V buffers and address them
                // as a page pool through a dense page table.
                int64_t const denseElems = static_cast<int64_t>(numPrefill) * kvCacheCapacity * mNumKVHeads * mHeadSize;
                rt::Tensor denseKVTensor = assignTensorFromWorkspace(
                    alignedWorkspacePtr, {2, numPrefill, kvCacheCapacity, mNumKVHeads, mHeadSize}, DataType::kHALF);
                rt::Tensor densePageTableTensor
                    = assignTensorFromWorkspace(alignedWorkspacePtr, {numPrefill, 2, maxPagesPerSeq}, DataType::kINT32);
                half* const kDense = denseKVTensor.dataPointer<half>();
                kernel::gatherPagedKVToSplit(kvCacheTensor.rawPointer(), kDense, kDense + denseElems, pageTable,
                    sequenceLengthsPtr, maxPagesPerSeq, numPrefill, kvCacheCapacity, mNumKVHeads, mHeadSize,
                    sizeof(int8_t), true /* dequantFp8 */, kScale, vScale, stream);
                kernel::launchBuildDensePageTable(densePageTableTensor.dataPointer<int32_t>(), numPrefill,
                    maxPagesPerSeq, numPrefill * maxPagesPerSeq, stream);
                prefillPool = kDense;
                prefillPageTable = densePageTableTensor.dataPointer<int32_t>();
                prefillFlatPages = 2 * numPrefill * maxPagesPerSeq;
            }
            CuteDslFMHAV2Runner runner(
                mNumQHeads, mNumKVHeads, mHeadSize, numPrefill, numPrefillTokens, kvCacheCapacity);
            CuteDslFMHAV2Runner::PagedRaggedParams params{};
            params.qPtr = qTensor.dataPointer<half>();
            params.pagedKVPoolPtr = prefillPool;
            params.kvCachePageList = prefillPageTable;
            params.oPtr = outputPtr;
            params.cuQSeqLens = offsetsPtr;
            params.cuKVSeqLens = cuKVSeqLensTensor.dataPointer<int32_t>();
            params.totalQSeqLen = numPrefillTokens;
            params.maxQSeqLen = numPrefillTokens;
            params.numFlatPages = prefillFlatPages;
            params.maxPagesPerSeq = maxPagesPerSeq;
            params.tokensPerPage = rt::kTOKENS_PER_PAGE;
            params.stream = stream;
            params.attentionScale = mAttentionScale;
            params.slidingWindowSize = prefillSlidingWindow;
            params.attentionSinks = mEnableAttentionSink
                ? resolveAttentionSinkInput(inputDesc, inputs, inputLayout.attentionSink)
                : nullptr;
            prefillLaunched = runner.runPagedRagged(params);
        }
        if (!prefillLaunched)
        {
            return -1;
        }
    }
    if (numDecode == 0)
    {
        return 0;
    }

    // Decode rows [T_p, T) of sequences [Np, N), viewed exactly as a phase 3 step of Nd sequences.
    rt::Tensor const decodeQKV(packedPtr + static_cast<int64_t>(numPrefillTokens) * inputChannels,
        rt::Coords{numDecode, 1, packedHeads, mHeadSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    rt::Tensor const decodeRope(ropePtr + numPrefillTokens * ropeRowElems, rt::Coords{numDecode, 1, ropeRowElems},
        rt::DeviceType::kGPU, fromTrt(ropeDesc.type));
    rt::Tensor const decodeLengths(
        sequenceLengthsPtr + numPrefill, rt::Coords{numDecode}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor decodeOutput(outputPtr + static_cast<int64_t>(numPrefillTokens) * mNumQHeads * mHeadSize,
        rt::Coords{numDecode, 1, mNumQHeads, mHeadSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    rt::Tensor const noPositionIds{};
    rt::Tensor noTreeMask{};
    int32_t const* decodeSkipTable = nullptr;
    int32_t decodeSkipTableLen = 0;
    if (inputLayout.skipSoftmaxDecodeScale >= 0 && mSlidingWindowSize <= 0)
    {
        decodeSkipTable = static_cast<int32_t const*>(inputs[inputLayout.skipSoftmaxDecodeScale]);
        decodeSkipTableLen = static_cast<int32_t>(inputDesc[inputLayout.skipSoftmaxDecodeScale].dims.d[0]);
    }
    DecodeRows const decodeRows{decodeQKV, decodeRope, decodeLengths, noPositionIds, noTreeMask, kvCacheTensor,
        decodeOutput, nullptr /* queryLengths */, pageTable + static_cast<int64_t>(numPrefill) * 2 * maxPagesPerSeq,
        maxPagesPerSeq, qNormGammaDevicePtr, kNormGammaDevicePtr,
        mEnableAttentionSink ? resolveAttentionSinkInput(inputDesc, inputs, inputLayout.attentionSink) : nullptr,
        decodeSkipTable, decodeSkipTableLen, false /* explicitPositionIds */};
    return enqueueDecodeRows(decodeRows, false /* tree */, alignedWorkspacePtr, stream);
}

int32_t AttentionPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    AttentionInputLayout const inputLayout = this->inputLayout();
    int32_t const expectedNbInputs = inputLayout.numInputs;
    if (in == nullptr || out == nullptr || nbInputs != expectedNbInputs || nbOutputs != kNUM_REQUIRED_OUTPUTS)
    {
        LOG_ERROR("AttentionPlugin: expected %d inputs and %d outputs, but got %d inputs and %d outputs.",
            expectedNbInputs, kNUM_REQUIRED_OUTPUTS, nbInputs, nbOutputs);
        return -1;
    }
    if (!hasConcretePagedKVContract(in, out, mNumKVHeads, mHeadSize, mEnableFp8KVCache, mSupportsBoundedKVCache))
    {
        LOG_ERROR(
            "AttentionPlugin: kv_cache must use pool shape [2, N, %d, %d, %d], preserve its output shape, and "
            "use INT32 LINEAR kv_page_table [B, 2, M] with B matching packed QKV; full-cache-only profiles "
            "require M <= N.",
            rt::kTOKENS_PER_PAGE, mNumKVHeads, mHeadSize);
        return -1;
    }
    if (mExternalKVPrefix)
    {
        auto const& pool = in[inputLayout.prefixPool];
        auto const& table = in[inputLayout.prefixPageTable];
        if (!isKVCacheDescriptor(pool, mFp8KVPrefix) || !isKVPageTableDescriptor(table)
            || !hasConcretePagedKVContract(in[kIN_QKV_IDX].dims, in[kIN_QUERY_LENGTH_IDX].dims, pool.dims, pool.dims,
                table.dims, mNumKVHeads, mHeadSize, false))
        {
            LOG_ERROR("AttentionPlugin: invalid retained prefix pool/table shape.");
            return -1;
        }
        return 0;
    }
    LOG_DEBUG("AttentionPlugin[%s]: CTM decode %s, tree %s (D=%d, Hq/Hkv=%d/%d, fp8KV=%d, window=%d, sinks=%d).",
        mLayerName.c_str(), useCtmDecode() ? "enabled" : "disabled", useCtmTreeDecode() ? "enabled" : "disabled",
        mHeadSize, mNumQHeads, mNumKVHeads, mEnableFp8KVCache, mSlidingWindowSize, mEnableAttentionSink);
    if (useCtmDecode() || useCtmTreeDecode())
    {
        // The packed QKV binding is token-major [tokens, width]; the batch comes from query_lengths.
        int32_t const batchSize = static_cast<int32_t>(in[kIN_QUERY_LENGTH_IDX].dims.d[0]);
        int32_t const physicalTokens = static_cast<int32_t>(in[kIN_QKV_IDX].dims.d[0]);
        int32_t const qSeqLen = batchSize > 0 ? physicalTokens / batchSize : 1;
        // One profile-max buffer for the life of the plugin: a captured CUDA graph keeps replaying
        // with the address it recorded, so the buffer must never move.
        if (!allocateCtmSemaphores())
        {
            return -1;
        }
        size_t const required = ctmSemaphoreElems(batchSize, qSeqLen);
        if (required > mCtmSemaphoreElems)
        {
            LOG_ERROR("AttentionPlugin: shape needs %zu CTM semaphores but the profile capacity is %zu.", required,
                mCtmSemaphoreElems);
            return -1;
        }
    }
    try
    {
        DecoderXQARunner const runner(fromTrt(mDataType), fromTrt(selectKvCacheDataType(mEnableFp8KVCache)),
            in[kIN_QUERY_LENGTH_IDX].dims.d[0], mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion);
        uint32_t const capacity = static_cast<uint32_t>(in[kIN_KV_PAGE_TABLE_IDX].dims.d[2] * rt::kTOKENS_PER_PAGE);
        if (!mCtmDecodeOnly && mEnableTreeAttention == 0 && mXqaSemaphores.isEmpty()
            && runner.getWorkspaceSize(capacity) > 0U)
        {
            rt::Tensor counters(
                {DecoderXQARunner::getSemaphoreCount()}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
            cudaStream_t stream{};
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)> const streamGuard(
                stream, cudaStreamDestroy);
            CUDA_CHECK(cudaMemsetAsync(counters.rawPointer(), 0, counters.getMemoryCapacity(), stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            // The last split CTA wraps its counter to zero. Keep this allocation alive across shape changes.
            mXqaSemaphores = std::move(counters);
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("AttentionPlugin: failed to initialize XQA counters: %s", e.what());
        return -1;
    }
    return 0;
}

IPluginV3* AttentionPlugin::attachToContext([[maybe_unused]] IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* AttentionPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("num_q_heads", &mNumQHeads, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("num_kv_heads", &mNumKVHeads, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("head_size", &mHeadSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("attention_scale", &mAttentionScale, PluginFieldType::kFLOAT32, 1);
    mDataToSerialize.emplace_back("enable_tree_attention", &mEnableTreeAttention, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("enable_qk_norm", &mEnableQKNorm, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("qk_norm_post_rope", &mQKNormPostRope, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("enable_kv_shared", &mEnableKVShared, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(
        "enable_contiguous_query_swa", &mEnableContiguousQuerySwa, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("enable_attention_sink", &mEnableAttentionSink, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("external_kv_prefix", &mExternalKVPrefix, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("fp8_kv_prefix", &mFp8KVPrefix, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("enable_fp8_kv_cache", &mEnableFp8KVCache, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(
        "enable_vision_block_attention", &mEnableVisionBlockAttention, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(
        "enable_context_mask_selector", &mEnableContextMaskSelector, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("sliding_window_size", &mSlidingWindowSize, PluginFieldType::kINT32, 1);
    mSupportsBoundedKVCacheToSerialize = static_cast<int32_t>(mSupportsBoundedKVCache);
    mDataToSerialize.emplace_back(
        "supports_bounded_kv_cache", &mSupportsBoundedKVCacheToSerialize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("tp_size", &mTpSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("skip_softmax_prefill_input", &mSkipSoftmaxPrefillInput, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("skip_softmax_decode_input", &mSkipSoftmaxDecodeInput, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(
        "qkv_scales", mQkvScales.data(), PluginFieldType::kFLOAT32, static_cast<int32_t>(mQkvScales.size()));
    // Serialize the RMSNorm eps for the fused qk_norm path. The gamma WEIGHTS live as
    // engine-weight constant inputs (baked at build time), not plugin fields.
    mDataToSerialize.emplace_back("rms_norm_eps", &mRmsNormEps, PluginFieldType::kFLOAT32, 1);
    mDataToSerialize.emplace_back("ctm_semaphore_capacity", &mCtmSemaphoreCapacityElems, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(kXQA_JIT_KERNELS_FIELD, mXqaJitBlob.empty() ? nullptr : mXqaJitBlob.data(),
        PluginFieldType::kCHAR, static_cast<int32_t>(mXqaJitBlob.size()));
    mDataToSerialize.emplace_back(layerNameField(mLayerName));
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ---------------------------------------------------------------------------
// Creator
// ---------------------------------------------------------------------------

AttentionPluginCreator::AttentionPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_q_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_kv_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("attention_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    // Make enable_fp8_kv_cache optional with default value 0 (disable by default)
    mPluginAttributes.emplace_back(PluginField("enable_tree_attention", nullptr, PluginFieldType::kINT32, 0));
    // Optional (default 0). Adds the gamma engine-weight inputs and fuses per-head RMSNorm.
    mPluginAttributes.emplace_back(PluginField("enable_qk_norm", nullptr, PluginFieldType::kINT32, 0));
    // Optional (default 0). QK-norm order: 0 = norm then rotate (Qwen3), 1 = rotate then
    // norm (HunYuan V1). Meaningful only with enable_qk_norm.
    mPluginAttributes.emplace_back(PluginField("qk_norm_post_rope", nullptr, PluginFieldType::kINT32, 0));
    // Optional (default 0). Shared-KV layer: packed input is Q only; no KV-cache write.
    mPluginAttributes.emplace_back(PluginField("enable_kv_shared", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("enable_attention_sink", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("enable_contiguous_query_swa", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("external_kv_prefix", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("fp8_kv_prefix", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("enable_fp8_kv_cache", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("enable_vision_block_attention", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("enable_context_mask_selector", nullptr, PluginFieldType::kINT32, 0));
    // Sliding window size (-1 = no sliding window, >0 = window size)
    mPluginAttributes.emplace_back(PluginField("sliding_window_size", nullptr, PluginFieldType::kINT32, 0));
    // Capability marker for runtime-selectable bounded KV storage.
    mPluginAttributes.emplace_back(PluginField("supports_bounded_kv_cache", nullptr, PluginFieldType::kINT32, 0));
    // Tensor-parallel world size of the exporting rank (default 1); TP>1 decodes on XQA.
    mPluginAttributes.emplace_back(PluginField("tp_size", nullptr, PluginFieldType::kINT32, 0));
    // Optional (default 0). 1 wires the skip_softmax_prefill_scale / skip_softmax_decode_scale table input.
    mPluginAttributes.emplace_back(PluginField("skip_softmax_prefill_input", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("skip_softmax_decode_input", nullptr, PluginFieldType::kINT32, 0));
    // Optional QKV dequant scales [q, k, v] for FP8 attention
    mPluginAttributes.emplace_back(PluginField("qkv_scales", nullptr, PluginFieldType::kFLOAT32, 0));
    // Optional per-head RMSNorm gamma weights (length == head_size) and eps. Empty / missing ->
    // qk_norm disabled. When supplied, the plugin fuses RMSNorm into the RoPE+KVWrite kernel.
    mPluginAttributes.emplace_back(PluginField("rms_norm_eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("ctm_semaphore_capacity", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField(kXQA_JIT_KERNELS_FIELD, nullptr, PluginFieldType::kCHAR, 0));
    // Enforce Core parameters are specified.
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* AttentionPluginCreator::getPluginName() const noexcept
{
    return kATTENTION_PLUGIN_NAME;
}

PluginFieldCollection const* AttentionPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void AttentionPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace ? libNamespace : "";
}

char const* AttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* AttentionPluginCreator::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

IPluginV3* AttentionPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        auto plugin = std::make_unique<AttentionPlugin>(resolvePluginLayerName(name, fc), fc);
        if (phase == TensorRTPhase::kBUILD)
        {
            plugin->compileXQAJitKernelForBuild();
        }
        // An engine built for a config with no XQA decode kernel carries an
        // empty blob; that is a valid engine as long as some other backend
        // implements the layer, so only load when there is something to load.
        if (phase == TensorRTPhase::kRUNTIME && plugin->hasSerializedXQAJitKernels())
        {
            plugin->loadSerializedXQAJitKernel();
        }
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create AttentionPlugin: %s", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
