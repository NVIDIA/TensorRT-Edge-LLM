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
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "kernels/contextAttentionKernels/contextFMHARunner.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "plugins/utils/pluginUtils.h"

// CuTe DSL FMHA kernel (Blackwell SM100+)
#ifdef CUTE_DSL_FMHA_ENABLED
#include "kernels/contextAttentionKernels/cuteDslFMHARunner.h"
#endif

// Portable CuTe DSL FMHA-v2 kernel, including native paged FP16 KV.
#ifdef CUTE_DSL_FMHA_V2_ENABLED
#include "kernels/contextAttentionKernels/cuteDslFMHAV2Runner.h"
#endif

// CuTe DSL FFPA kernel (headSize=512 fallback)
#ifdef CUTE_DSL_FFPA_ENABLED
#include "kernels/contextAttentionKernels/cuteDslFFPARunner.h"
#endif

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kATTENTION_PLUGIN_VERSION{"1"};
constexpr char const* kATTENTION_PLUGIN_NAME{"AttentionPlugin"};

// Select KV cache storage datatype based on FP8 enablement
static inline DataType selectKvCacheDataType(bool enableFp8KVCache)
{
    return enableFp8KVCache ? DataType::kFP8 : DataType::kHALF;
}

// Define the mapping of input and output indices of the AttentionPlugin.
constexpr int32_t kIN_Q_IDX{0};
constexpr int32_t kIN_K_IDX{1};
constexpr int32_t kIN_V_IDX{2};
constexpr int32_t kIN_KV_CACHE_IDX{3};
constexpr int32_t kIN_CONTEXT_LENGTH_IDX{4};
constexpr int32_t kIN_ROPE_COS_SIN_IDX{5};
constexpr int32_t kIN_KV_CACHE_START_IDX{6};
constexpr int32_t kIN_KV_PAGE_TABLE_IDX{7};
constexpr int32_t kIN_OPTIONAL_ATTN_MASK_IDX{8};
constexpr int32_t kIN_OPTIONAL_ATTN_POS_ID_IDX{9};
constexpr int32_t kOUT_ATTENTION_IDX{0};
constexpr int32_t kOUT_KV_CACHE_IDX{1};

// Reflect the count of Inputs and Outputs of the AttentionPlugin,
// these definitions shall be consistent.
constexpr int32_t kNUM_REQUIRED_INPUTS{8};
constexpr int32_t kNUM_TREE_ATTN_OPTIONAL_INPUTS{2};
constexpr int32_t kNUM_VISION_BLOCK_OPTIONAL_INPUTS{1};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{2};

// Support Tree Attention decoding schema up to 128 tokens in the draft tree per batch.
// We are unable to check this property during shape checking since prefill length is much larger than this value.
constexpr int64_t kMAX_EAGLE_DECODING_TOKENS = 128;

enum class AttentionExecutionMode
{
    kINVALID,
    kNORMAL_PREFILL,
    kCHUNKED_PREFILL,
    kVANILLA_DECODING,
    kTREE_DECODING
};

AttentionExecutionMode deduceModeVanilla(rt::Tensor const& qInputTensor, rt::Tensor const& kvCacheStartIdxTensor)
{
    // Empty KVCache Start indices means normal prefill without previous KVCache. Notice single token is also a valid
    // prefill length.
    if (kvCacheStartIdxTensor.getShape()[0] == 0)
    {
        return AttentionExecutionMode::kNORMAL_PREFILL;
    }

    // Otherwise, distinguish between chunked prefill and vanilla decoding based on the runtime Sequence Length.
    // Vanilla decoding should always have runtime sequence length of 1.
    int64_t const runtimeSeqLen = qInputTensor.getShape()[1];
    if (runtimeSeqLen > 1)
    {
        return AttentionExecutionMode::kCHUNKED_PREFILL;
    }
    return AttentionExecutionMode::kVANILLA_DECODING;
}

AttentionExecutionMode deduceModeTreeAttention(
    rt::Tensor const& qInputTensor, rt::Tensor const& kvCacheStartIdxTensor, rt::Tensor const& attentionPosIdTensor)
{
    // Normal prefill if there is no previous KVCache.
    if (kvCacheStartIdxTensor.getShape()[0] == 0)
    {
        return AttentionExecutionMode::kNORMAL_PREFILL;
    }

    // Under tree attention, each token will be associated with a position id (within the sequence) to perform correct
    // positional encoding. Even for casual decoding with multiple tokens, the position id is still required to be
    // supplied.

    // Note, chunked prefill is very similar to tree decoding, the difference is chunked prefill will have contiguous
    // tokens in the sequence while tree decoding has a "tree" structure described by attention mask and position ids.
    // By convention, we will supply 1 shape for position id tensor under prefill execution.
    int64_t const runtimeSeqLen = qInputTensor.getShape()[1];
    int64_t const positionIdLen = attentionPosIdTensor.getShape()[1];

    if (runtimeSeqLen == 1)
    {
        // Also supports single token decoding mode when tree attention is enabled.
        return AttentionExecutionMode::kVANILLA_DECODING;
    }
    else if (positionIdLen == runtimeSeqLen)
    {
        return AttentionExecutionMode::kTREE_DECODING;
    }
    else if (positionIdLen == 1)
    {
        return AttentionExecutionMode::kCHUNKED_PREFILL;
    }

    return AttentionExecutionMode::kINVALID;
}

bool loadFMHAKernels(bool& useCuteDslFMHA, bool& useCuteDslFMHAV2, int32_t numQHeads, int32_t numKVHeads,
    int32_t headSize, int32_t smVersion, nvinfer1::DataType dataType, bool useSlidingWindow, bool enableFp8KVCache)
{
#ifdef CUTE_DSL_FMHA_ENABLED
    if (useCuteDslFMHA)
    {
        if (CuteDslFMHARunner::canImplement(headSize, smVersion) && CuteDslFMHARunner::loadLLMKernelModule())
        {
            useCuteDslFMHAV2 = false;
            LOG_DEBUG("CuTe DSL FMHA kernel loaded for SM%d", smVersion);
            return true;
        }
        LOG_DEBUG("Optimized CuTe DSL FMHA not available (headSize=%d, SM%d)", headSize, smVersion);
        useCuteDslFMHA = false;
    }
#endif

#ifdef CUTE_DSL_FMHA_V2_ENABLED
    if (useCuteDslFMHAV2)
    {
        CuteDslFMHAV2MaskType const maskType
            = useSlidingWindow ? CuteDslFMHAV2MaskType::kSLIDING_CAUSAL : CuteDslFMHAV2MaskType::kCAUSAL;
        if (!enableFp8KVCache
            && CuteDslFMHAV2Runner::canImplementPaged(numQHeads, numKVHeads, headSize, smVersion, dataType, maskType)
            && CuteDslFMHAV2Runner::loadLLMKernelModule())
        {
            LOG_DEBUG("Native paged CuTe DSL FMHA-v2 kernel loaded for SM%d", smVersion);
            return true;
        }
        LOG_DEBUG("Native paged CuTe DSL FMHA-v2 unavailable (headSize=%d, SM%d, fp8KV=%d)", headSize, smVersion,
            enableFp8KVCache ? 1 : 0);
        useCuteDslFMHAV2 = false;
    }
#endif

    bool canImplementFMHA
        = ContextFMHARunner::canImplement(headSize, smVersion, dataType, AttentionInputLayout::SEPARATE_Q_K_V,
            useSlidingWindow ? ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL : ContextAttentionMaskType::CAUSAL);
    if (canImplementFMHA && !ContextFMHARunner::loadContextFMHAKernels(smVersion, dataType))
    {
        LOG_ERROR("Failed to load dense FMHA_v2 cubins for SM%d", smVersion);
        canImplementFMHA = false;
    }
    return canImplementFMHA;
}

// Workspace layout (cumulative, worst-case across all execution paths):
//
//   Slot  | Shape                            | Type  | Used by
//   ------+----------------------------------+-------+------------------------------------------
//   0     | [B+1]                            | INT32 | cuQSeqLens          (prefill)
//   1     | [B+1]                            | INT32 | cuKVSeqLens         (prefill)
//   2     | [B]                              | INT32 | kvCacheEndIdxs      (prefill)
//   3     | [B+1]                            | INT32 | paddedCuKVSeqLens   (prefill, CuTe DSL)
//   4     | [B, 2, Hkv, Smax, D]             | HALF  | splitPagedKV out    (FMHA_v2/FFPA fallback, always FP16)
//   5*    | [B, S, Hq, D]                    | FP8   | fp8Q                (CuTe DSL + FP8 prefill only)
//   6*    | [B, S] x 2                       | INT32 | blockBegin/blockEnd (vision FFPA overlay prefill only)
//   7*    | [packedMaskWords(B, S, S)]       | INT32 | packedMask          (vision FMHA CUSTOM_MASK prefill only)
//   8*    | [B+1]                            | INT32 | cuMaskRows          (vision FMHA CUSTOM_MASK prefill only)
//
//   * Slots 5-8 are conditionally allocated (CuTe DSL + FP8 KV cache /
//     vision-block attention respectively).
//
// Total allocation is the sum of all conditional slots (safe upper bound).
size_t getAttentionWorkspaceSize(int64_t batchSize, int64_t seqLen, int64_t kvCacheCapacity, int32_t numQHeads,
    int32_t numKVHeads, int32_t headSize, bool useCuteDslFMHA, bool enableFp8KVCache, bool enableVisionBlockAttention)
{
    size_t workspaceSize = 0;

    // CuQSeqLens for FMHA.
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize + 1}, DataType::kINT32);

    // Always reserve workspace memory to prepare for chunked prefill decoding. The implementation should be further
    // optimized to avoid the workspace size overhead.
    workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize + 1}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize + 1}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{batchSize, 2, numKVHeads, kvCacheCapacity, headSize}, DataType::kHALF);

    // FP8 Q output: RoPE kernel writes FP8 Q to this workspace buffer (CuTe DSL FMHA path).
    if (useCuteDslFMHA && enableFp8KVCache)
    {
        workspaceSize = accumulateWorkspaceSize(
            workspaceSize, rt::Coords{batchSize, seqLen, numQHeads, headSize}, DataType::kFP8);
    }

    // Per-position vision-block intervals for the FFPA overlay prefill, plus
    // the FMHA CUSTOM_MASK packed mask (+ per-batch mask-row offsets) for the
    // sliding d256 prefill.
    if (enableVisionBlockAttention)
    {
        workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize, seqLen}, DataType::kINT32);
        workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize, seqLen}, DataType::kINT32);
        workspaceSize = accumulateWorkspaceSize(
            workspaceSize, rt::Coords{kernel::getPackedMaskSizeInWords(batchSize, seqLen, seqLen)}, DataType::kINT32);
        workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize + 1}, DataType::kINT32);
    }

    return workspaceSize;
}

// Shared shape check for every numPages reader (runPaged's dims.d[1] read and splitPagedKV's
// getShape()[1] read): the kv_cache binding must be the paged-pool contract
// [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim]. A legacy/mismatched binding (e.g. the
// pre-relayout [batch, 2, Hkv, capacity, D] shape) would otherwise silently misread numPages and
// corrupt prefill instead of failing loudly.
bool isPagedPoolShape(rt::Coords const& shape, int32_t numKVHeads, int32_t headSize)
{
    return shape.getNumDims() == 5 && shape[0] == 2 && shape[2] == rt::kTOKENS_PER_PAGE && shape[3] == numKVHeads
        && shape[4] == headSize;
}

} // namespace

// Static class fields initialization
PluginFieldCollection AttentionPluginCreator::mFieldCollection{};
std::vector<PluginField> AttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AttentionPluginCreator);

// WAR for split-KV prefill consumers (FMHA_v2 fallback, FFPA d512) that cannot read the paged pool
// [2, numPages, 128, Hkv, D] directly and consume FP16 via dataPointer<half>().
//
// Design: ALWAYS device-gather the page table into an FP16 workspace -- never alias the pool in
// place. Aliasing was only valid under a hardcoded identity-table guarantee whose release build silently
// aliased the wrong pages for a non-identity table; selecting alias-vs-gather correctly requires knowing
// the table's contiguity on the host, which needs a per-enqueue D2H stream sync on this path. Instead we
// unconditionally gather: the gather follows any table (identity or scrambled) correctly and identically
// in debug and release, and its cost is a single live-length copy on an already-non-primary fallback
// path (only reached when the CuTe DSL / paged-XQA backends do not cover the head size). The gather also
// dequantizes an FP8 pool to FP16 with the K/V scales, so consumers never reinterpret FP8 bytes as half,
std::pair<rt::Tensor, rt::Tensor> AttentionPlugin::splitPagedKV(rt::Tensor const& poolTensor, int32_t const* pageTable,
    int32_t const* kvSeqLens, int32_t maxPagesPerSeq, std::byte*& workspacePtr, int32_t batchSize, int32_t numKVHeads,
    int32_t capPadded, int32_t headSize, int32_t seqLen, float kScale, float vScale, cudaStream_t stream)
{
    // Single chokepoint for every splitPagedKV caller (FMHA_v2 fallback + FFPA d512 shared-KV):
    // the pool binding must be [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim] before dim 1
    // is trusted as numPages below — same contract and same failure mode as runPaged's guard.
    check::check(isPagedPoolShape(poolTensor.getShape(), numKVHeads, headSize),
        "splitPagedKV: kv_cache binding is not the paged-pool contract [2, numPages, kTOKENS_PER_PAGE, "
        "numKVHeads, headDim]; the export/builder KV-cache binding is not pool-shaped.");

    DataType const dtype = poolTensor.getDataType();
    // The fallback consumers read the split K/V as FP16. FP16 pools byte-copy; FP8 pools dequantize.
    // Any other pool dtype would be reinterpreted as half by the consumers, so reject it loudly.
    bool const isFp8KV = (dtype == DataType::kFP8);
    check::check(dtype == DataType::kHALF || isFp8KV,
        "splitPagedKV: FMHA_v2/FFPA fallback requires an FP16 or FP8 KV pool; other dtypes would be "
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
    rt::Tensor kTensor(
        ptr, rt::Coords{batchSize, outSeqDim, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor(ptr + kTensorElems * sizeof(half), rt::Coords{batchSize, outSeqDim, numKVHeads, headSize},
        rt::DeviceType::kGPU, DataType::kHALF);

    kernel::gatherPagedKVToSplit(poolTensor.rawPointer(), kTensor.rawPointer(), vTensor.rawPointer(), pageTable,
        kvSeqLens, maxPagesPerSeq, batchSize, outSeqDim, numKVHeads, headSize, elemSize, isFp8KV, kScale, vScale,
        stream);

    return std::make_pair(std::move(kTensor), std::move(vTensor));
}

#ifdef CUTE_DSL_FFPA_ENABLED
void AttentionPlugin::dispatchFFPAKernel(half const* q, half const* k, half const* v, half* o, int32_t const* cuSeqLenQ,
    int32_t const* cuSeqLenK, int32_t batchSize, int32_t seqlenQ, int32_t seqlenK, cudaStream_t stream)
{
    CuteDslFFPAParams ffpaParams{};
    ffpaParams.q = q;
    ffpaParams.k = k;
    ffpaParams.v = v;
    ffpaParams.o = o;
    ffpaParams.cuSeqLenQ = cuSeqLenQ;
    ffpaParams.cuSeqLenK = cuSeqLenK;
    ffpaParams.batchSize = batchSize;
    ffpaParams.seqlenQ = seqlenQ;
    ffpaParams.seqlenK = seqlenK;
    ffpaParams.numQHeads = mNumQHeads;
    ffpaParams.numKVHeads = mNumKVHeads;
    ffpaParams.headDim = mHeadSize;
    ffpaParams.softmaxScale = mAttentionScale;
    CuteDslFFPARunner::run(ffpaParams, stream);
}
#endif

bool AttentionPlugin::canUseFFPAOverlayForVisionPrefill() const noexcept
{
#ifdef CUTE_DSL_FFPA_ENABLED
    // FFPA is full-causal: only the non-sliding (global) layers may use the
    // overlay.  mCanImplementFFPA implies headSize == 512 and a loaded module.
    return mCanImplementFFPA && mSlidingWindowSize <= 0
        && CuteDslFFPARunner::canImplementVisionBlock(mHeadSize, mSMVersion);
#else
    return false;
#endif
}

void AttentionPlugin::enforceVisionBlockKernelSupport() const
{
    // Vision-block attention has no fallback path: every layer must have a
    // production kernel, otherwise fail loudly at plugin construction (a
    // clear build/load-time error beats a silently wrong deployment).
    if (mHeadSize == 512 && mSlidingWindowSize <= 0)
    {
        ELLM_CHECK(canUseFFPAOverlayForVisionPrefill(),
            "AttentionPlugin: vision-block prefill (headSize=512, full-causal) requires the CuTe DSL FFPA d512 "
            "vision-block overlay kernel, which is unavailable on SM"
                + std::to_string(mSMVersion)
                + " in this build. Rebuild with a CuTe DSL FFPA artifact (ffpa_d512_causal_visionblock variant) "
                  "matching this SM.");
    }
    else
    {
        ELLM_CHECK(mCanImplementCustomMaskFMHA,
            "AttentionPlugin: vision-block prefill requires FMHA_v2 CUSTOM_MASK cubins (SEPARATE_Q_K_V) for headSize="
                + std::to_string(mHeadSize) + " on SM" + std::to_string(mSMVersion)
                + ", which are missing from the cubin metadata table.");
    }
    ELLM_CHECK(mCanImplementXQA,
        "AttentionPlugin: vision-block decode requires XQA decode kernels for Hq=" + std::to_string(mNumQHeads)
            + ", Hkv=" + std::to_string(mNumKVHeads) + ", headSize=" + std::to_string(mHeadSize) + " on SM"
            + std::to_string(mSMVersion) + ", which are not available.");
}

AttentionPlugin::AttentionPlugin(std::string const& name, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t enableTreeAttention, int32_t enableFp8KVCache, int32_t enableVisionBlockAttention,
    int32_t slidingWindowSize, std::vector<float> const& qkvScales, std::optional<float> attentionScale)
    : mLayerName(name)
    , mNumQHeads(numQHeads)
    , mNumKVHeads(numKVHeads)
    , mHeadSize(headSize)
    , mAttentionScale(resolveAttentionScale(attentionScale, headSize))
    , mEnableTreeAttention(enableTreeAttention)
    , mEnableVisionBlockAttention(enableVisionBlockAttention)
    , mEnableFp8KVCache(enableFp8KVCache)
    , mQkvScales(enableFp8KVCache ? qkvScales : std::vector<float>{1.f, 1.f, 1.f})
    , mSlidingWindowSize(slidingWindowSize)
{
    ELLM_CHECK(!(mEnableTreeAttention && mEnableVisionBlockAttention),
        "Tree attention and vision block attention are mutually exclusive.");
    ELLM_CHECK(!mEnableVisionBlockAttention || !mEnableFp8KVCache, "Vision block attention requires an FP16 KV cache.");
    ELLM_CHECK(!mEnableFp8KVCache || mQkvScales.size() == 3,
        "FP8 KV cache enabled but qkv_scales has "
            + std::to_string(mQkvScales.size()) + " elements (expected 3). "
            "Re-export the model to include QKV scales [q, k, v].");

    mSMVersion = getSMVersion();
    applyThorSMRenumberWAR(mSMVersion);

    mCanImplementFMHA = loadFMHAKernels(mUseCuteDslFMHA, mUseCuteDslFMHAV2, mNumQHeads, mNumKVHeads, mHeadSize,
        mSMVersion, mDataType, mSlidingWindowSize > 0, mEnableFp8KVCache != 0);
    LOG_DEBUG("AttentionPlugin FMHA path: %s, sliding_window: %s",
        mUseCuteDslFMHA ? "optimized CuTe DSL FMHA"
                        : (mUseCuteDslFMHAV2 ? "native paged CuTe DSL FMHA-v2" : "dense FMHA-v2"),
        mSlidingWindowSize > 0 ? std::to_string(mSlidingWindowSize).c_str() : "disabled");

    // XQA decode kernels are needed for decode path when available. Decode always reads the paged
    // pool (identity page table while cross-request reuse is off), so load the paged KV cache kernels.
    bool const useSpecDecode = true;
    mCanImplementXQA = DecoderXQARunner::canImplement(mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion, mDataType,
        selectKvCacheDataType(mEnableFp8KVCache), /*usePagedKVCache=*/true);
    if (mCanImplementXQA)
    {
        DecoderXQARunner::loadDecodeXQAKernels(
            mSMVersion, mDataType, selectKvCacheDataType(mEnableFp8KVCache), useSpecDecode, /*usePagedKVCache=*/true);
    }

    // Kernel selection priority for prefill and decode:
    //   1. Vision-block attention        — per-layer routing: FFPA d512 vision-block overlay
    //      (full-causal d512 prefill) or FMHA CUSTOM_MASK (sliding d256-class prefill); XQA
    //      decode.  All three are hard requirements (no fallback; construction fails loudly).
    //   2. FMHA (prefill) + XQA (decode) — standard path for most head sizes.
    //   3. FFPA (prefill) + XQA (decode) — fallback for headSize=512 where FMHA has no cubins.
    //   4. FFPA (prefill) only           — headSize=512 without XQA decode support.
    //   5. XQA (decode) only             — prefill unsupported for this head size.
    //   6. None                          — fatal, cannot serve this configuration.

    // FMHA unavailable — try to load the FFPA d512 kernel module.  It serves
    // both the plain headSize=512 prefill and (when the visionblock AOT
    // variant is present) the vision-block overlay prefill.
    if (!mCanImplementFMHA)
    {
#ifdef CUTE_DSL_FFPA_ENABLED
        if (mHeadSize == 512 && CuteDslFFPARunner::canImplement(mHeadSize, mSMVersion, mNumQHeads, mNumKVHeads))
        {
            if (CuteDslFFPARunner::loadKernelModule())
            {
                mCanImplementFFPA = true;
            }
        }
#endif
    }

    if (mEnableVisionBlockAttention)
    {
        // Sliding d256-class prefill production path: FMHA_v2 CUSTOM_MASK.
        // Availability is discovered from the cubin metadata table.
        mCanImplementCustomMaskFMHA = ContextFMHARunner::canImplement(mHeadSize, mSMVersion, mDataType,
                                          AttentionInputLayout::SEPARATE_Q_K_V, ContextAttentionMaskType::CUSTOM_MASK)
            && ContextFMHARunner::loadContextFMHAKernels(mSMVersion, mDataType);

        enforceVisionBlockKernelSupport();

        LOG_INFO(
            "AttentionPlugin: vision-block attention (headSize=%d, Hq=%d, Hkv=%d, window=%d) — prefill via %s, "
            "decode via XQA.",
            mHeadSize, mNumQHeads, mNumKVHeads, mSlidingWindowSize,
            canUseFFPAOverlayForVisionPrefill() ? "FFPA d512 vision-block overlay" : "FMHA CUSTOM_MASK");
    }
    else if (!mCanImplementFMHA && !mCanImplementFFPA && !mCanImplementXQA)
    {
        LOG_ERROR("Cannot implement AttentionPlugin configuration. SM: %d, HeadSize: %d, NumQHeads: %d, NumKVHeads: %d",
            mSMVersion, mHeadSize, mNumQHeads, mNumKVHeads);
        throw std::runtime_error("Cannot implement the AttentionPlugin configuration.");
    }
    else if (!mCanImplementFMHA)
    {
        if (mCanImplementFFPA)
        {
            LOG_INFO("AttentionPlugin: FMHA unsupported for headSize=%d, using FFPA for prefill%s.", mHeadSize,
                mCanImplementXQA ? " + XQA for decode" : "");
        }
        else
        {
            LOG_WARNING(
                "AttentionPlugin: no prefill kernel for headSize=%d; only decode (XQA) is supported.", mHeadSize);
        }
    }
}

AttentionPlugin::AttentionPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
    , mNumQHeads(parsePluginScalarField<int32_t>("num_q_heads", fc).value_or(0))
    , mNumKVHeads(parsePluginScalarField<int32_t>("num_kv_heads", fc).value_or(0))
    , mHeadSize(parsePluginScalarField<int32_t>("head_size", fc).value_or(0))
    , mAttentionScale(resolveAttentionScale(parsePluginScalarField<float>("attention_scale", fc), mHeadSize))
    , mEnableTreeAttention(parsePluginScalarField<int32_t>("enable_tree_attention", fc).value_or(0))
    , mEnableFp8KVCache(parsePluginScalarField<int32_t>("enable_fp8_kv_cache", fc).value_or(0))
    , mSlidingWindowSize(parsePluginScalarField<int32_t>("sliding_window_size", fc).value_or(-1))
{
    mEnableVisionBlockAttention = parsePluginScalarField<int32_t>("enable_vision_block_attention", fc).value_or(0);
    ELLM_CHECK(!(mEnableTreeAttention && mEnableVisionBlockAttention),
        "Tree attention and vision block attention are mutually exclusive.");
    ELLM_CHECK(!mEnableVisionBlockAttention || !mEnableFp8KVCache, "Vision block attention requires an FP16 KV cache.");

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

    if (!mEnableFp8KVCache)
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
    applyThorSMRenumberWAR(mSMVersion);

    mCanImplementFMHA = loadFMHAKernels(mUseCuteDslFMHA, mUseCuteDslFMHAV2, mNumQHeads, mNumKVHeads, mHeadSize,
        mSMVersion, mDataType, mSlidingWindowSize > 0, mEnableFp8KVCache != 0);
    LOG_DEBUG("AttentionPlugin FMHA path: %s",
        mUseCuteDslFMHA ? "optimized CuTe DSL FMHA"
                        : (mUseCuteDslFMHAV2 ? "native paged CuTe DSL FMHA-v2" : "dense FMHA-v2"));

    // XQA decode kernels. Decode always reads the paged pool, so load the paged KV cache kernels.
    mCanImplementXQA = DecoderXQARunner::canImplement(mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion, mDataType,
        selectKvCacheDataType(mEnableFp8KVCache), /*usePagedKVCache=*/true);
    if (mCanImplementXQA)
    {
        DecoderXQARunner::loadDecodeXQAKernels(mSMVersion, mDataType, selectKvCacheDataType(mEnableFp8KVCache),
            /*useSpecDecodeKernels=*/true, /*usePagedKVCache=*/true);
    }

    if (!mCanImplementFMHA)
    {
#ifdef CUTE_DSL_FFPA_ENABLED
        if (mHeadSize == 512 && CuteDslFFPARunner::canImplement(mHeadSize, mSMVersion, mNumQHeads, mNumKVHeads))
        {
            if (CuteDslFFPARunner::loadKernelModule())
            {
                mCanImplementFFPA = true;
            }
        }
#endif
    }

    // Sliding d256-class vision prefill production path: FMHA_v2 CUSTOM_MASK,
    // discovered from the cubin metadata table.
    if (mEnableVisionBlockAttention)
    {
        mCanImplementCustomMaskFMHA = ContextFMHARunner::canImplement(mHeadSize, mSMVersion, mDataType,
                                          AttentionInputLayout::SEPARATE_Q_K_V, ContextAttentionMaskType::CUSTOM_MASK)
            && ContextFMHARunner::loadContextFMHAKernels(mSMVersion, mDataType);

        enforceVisionBlockKernelSupport();
    }
}

AttentionPlugin::~AttentionPlugin() = default;

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
        auto* p = new AttentionPlugin(mLayerName, mNumQHeads, mNumKVHeads, mHeadSize, mEnableTreeAttention,
            mEnableFp8KVCache, mEnableVisionBlockAttention, mSlidingWindowSize, mQkvScales, mAttentionScale);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
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
        outputTypes[kOUT_ATTENTION_IDX] = inputTypes[kIN_Q_IDX];
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
        // Output[0] is attention result, has shape [B, S, Hq, D]. Refers to Q shape [B, S, Hq*D]
        outputs[kOUT_ATTENTION_IDX].nbDims = 4;
        outputs[kOUT_ATTENTION_IDX].d[0] = inputs[kIN_Q_IDX].d[0];
        outputs[kOUT_ATTENTION_IDX].d[1] = inputs[kIN_Q_IDX].d[1];
        outputs[kOUT_ATTENTION_IDX].d[2] = exprBuilder.constant(mNumQHeads);
        outputs[kOUT_ATTENTION_IDX].d[3] = exprBuilder.constant(mHeadSize);

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
    //      Q tensor (linear FP16) with shape [B, S, Hq, D]
    //      K tensor (linear FP16) with shape [B, S, Hkv, D]
    //      V tensor (linear FP16) with shape [B, S, Hkv, D]
    //      KV-cache tensor (linear FP16/FP8): paged pool [2, numPages, kTOKENS_PER_PAGE, Hkv, D].
    //      (numPages is a fixed value per engine build; see setupKVCacheProfiles in llmBuilder.cpp.)
    //      Real context length: [B] (a vector of scalars) with type int32_t.
    //      RoPE cos/sin cache: [B or 1, Smax, D] (a tensor of scalars) with type float.
    //            Rope CosSin can be ND vector depending on rope type.
    //      Start index of the KVCache [B, 0~1] (a vector of scalars) with type int32_t.
    //            0 length indicates there is no existing KVCache for inference.
    //      Optional tree attention mask: [B, S, S] (a tensor of scalars) with type int32_t.
    //      Optional tree attention position ids: [B, S] (a tensor of scalars) with type int32_t.

    // Support context/generation phase outputs:
    //      attention result (linear FP16) with shape [B, S, Hq, D]
    //      KV-cache tensor, same as the above.
    auto checkQ = [this](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[2] == mNumQHeads * mHeadSize;
        }
        return status;
    };

    auto checkKV = [this](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[2] == mNumKVHeads * mHeadSize;
        }
        return status;
    };

    auto checkKVCache = [this](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        // Support FP16 or FP8 storage;
        if (mEnableFp8KVCache)
        {
            status &= (tensorDesc.type == DataType::kFP8);
        }
        else
        {
            status &= (tensorDesc.type == DataType::kHALF);
        }
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        // Paged pool [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim]: numPages is not known
        // here, so only the rank is checked; enqueue validates the full pool contract
        // (isPagedPoolShape) before trusting dim 1 as numPages.
        status &= tensorDesc.dims.nbDims == 5;
        return status;
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
        status &= tensorDesc.dims.nbDims == 3;
        status &= tensorDesc.dims.d[2] <= mHeadSize;
        return status;
    };

    auto checkAttentionMask = [](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        return status;
    };

    auto checkAttentionPosId = [](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 2;
        return status;
    };

    auto checkVisionBlockIds = [](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 2;
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
        status &= tensorDesc.dims.nbDims == 4;
        if (status)
        {
            auto const tensorDim = tensorDesc.dims;
            status &= tensorDim.d[2] == mNumQHeads;
            status &= tensorDim.d[3] == mHeadSize;
        }
        return status;
    };

    int32_t const expectedNbInputs = kNUM_REQUIRED_INPUTS + (mEnableTreeAttention ? kNUM_TREE_ATTN_OPTIONAL_INPUTS : 0)
        + (mEnableVisionBlockAttention ? kNUM_VISION_BLOCK_OPTIONAL_INPUTS : 0);
    bool const checkNumIOs = nbInputs == expectedNbInputs && nbOutputs == kNUM_REQUIRED_OUTPUTS;
    if (!checkNumIOs)
    {
        LOG_ERROR(
            "Invalid number of inputs or outputs for the AttentionPlugin '%s'. Expected %d inputs and %d outputs, but "
            "got %d inputs and %d outputs.",
            mLayerName.c_str(), expectedNbInputs, kNUM_REQUIRED_OUTPUTS, nbInputs, nbOutputs);
        return false;
    }

    bool result{true};

    if (pos < nbInputs)
    {
        switch (pos)
        {
        case kIN_Q_IDX: result = checkQ(inOut[pos].desc); break;
        case kIN_K_IDX: result = checkKV(inOut[pos].desc); break;
        case kIN_V_IDX: result = checkKV(inOut[pos].desc); break;
        case kIN_KV_CACHE_IDX: result = checkKVCache(inOut[pos].desc); break;
        case kIN_CONTEXT_LENGTH_IDX: result = checkSequenceLen(inOut[pos].desc); break;
        case kIN_ROPE_COS_SIN_IDX: result = checkPosEncodingCosSin(inOut[pos].desc); break;
        case kIN_KV_CACHE_START_IDX: result = checkKVCacheStartIdx(inOut[pos].desc); break;
        case kIN_KV_PAGE_TABLE_IDX: result = checkKVPageTable(inOut[pos].desc); break;
        default: break;
        }

        // Handle optional inputs (tree attention mask/pos and FP8 scales) with dynamic ordering
        if (result && pos > kIN_KV_PAGE_TABLE_IDX)
        {
            int32_t currentOptionalInputIdx = kIN_KV_PAGE_TABLE_IDX + 1;
            if (mEnableTreeAttention)
            {
                if (pos == currentOptionalInputIdx)
                {
                    result = checkAttentionMask(inOut[pos].desc);
                }
                currentOptionalInputIdx++;
                if (pos == currentOptionalInputIdx)
                {
                    result = checkAttentionPosId(inOut[pos].desc);
                }
                currentOptionalInputIdx++;
            }
            if (mEnableVisionBlockAttention && pos == currentOptionalInputIdx)
            {
                result = checkVisionBlockIds(inOut[pos].desc);
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

int32_t AttentionPlugin::configurePlugin([[maybe_unused]] DynamicPluginTensorDesc const* in,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] DynamicPluginTensorDesc const* out,
    [[maybe_unused]] int32_t nbOutputs) noexcept
{
    return 0; // No need to configure anything since we will only use the runtime tensor shapes.
}

size_t AttentionPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    [[maybe_unused]] DynamicPluginTensorDesc const* outputs, [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    // Guard against a stale pre-kv_page_table engine (7 inputs) deserialized against this newer
    // .so: that path skips supportsFormatCombination() and would index inputs[7] out of bounds
    // below. This method is noexcept with no failure sentinel, so log and return 0 instead of
    // throwing (TRT calls it at context creation, surfacing the failure at load time).
    if (nbInputs <= kIN_KV_PAGE_TABLE_IDX)
    {
        LOG_ERROR(
            "AttentionPlugin::getWorkspaceSize: nbInputs (%d) is too small for kv_page_table input index %d -- "
            "this engine was likely serialized by an older, incompatible build of this plugin; re-export and "
            "rebuild.",
            nbInputs, kIN_KV_PAGE_TABLE_IDX);
        return 0;
    }

    int64_t const maxBatchSize = inputs[kIN_Q_IDX].max.d[0];
    int64_t const maxSeqLen = inputs[kIN_Q_IDX].max.d[1];
    // KV binding is the paged pool [2, numPages, 128, Hkv, D]; the per-slot padded capacity is the
    // page-table width times the page size (kv_page_table is [batch, 2, maxPagesPerSeq]).
    int64_t const maxKVCacheCapacity = inputs[kIN_KV_PAGE_TABLE_IDX].max.d[2] * rt::kTOKENS_PER_PAGE;
    size_t const workspaceSize = getAttentionWorkspaceSize(maxBatchSize, maxSeqLen, maxKVCacheCapacity, mNumQHeads,
        mNumKVHeads, mHeadSize, mUseCuteDslFMHA, mEnableFp8KVCache, mEnableVisionBlockAttention != 0);

    LOG_DEBUG("AttentionPlugin workspace size: %zu bytes", workspaceSize);
    return workspaceSize;
}

int32_t AttentionPlugin::getAliasedInput(int32_t outputIndex) noexcept
{
    // WAR:this is not the correct plugin API usage. The
    // plugin updates the KV cache in place, so the correct return is
    // kIN_KV_CACHE_IDX (output kOUT_KV_CACHE_IDX aliases that input). We return -1
    // to drop the alias because declaring it makes Myelin keep a redundant
    // per-layer KV copy (the perf regression). In-place read-write still works
    // because the runtime binds past and present KV to the same address. TODO:
    // restore the alias declaration once the Myelin issue is fixed.
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

int32_t AttentionPlugin::enqueueImpl(PluginTensorDesc const* inputDesc,
    [[maybe_unused]] PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream)
{
    // Construct non-owned tensor objects from I/O data pointers and shapes.
    // Q input in the graph will be in shape [B, S, Hq x D], for convenience,
    // we will use shape of [B, S, Hq, D] to represent the tensor.
    // K and V inputs are in shape [B, S, Hkv x D], represented as [B, S, Hkv, D].
    PluginTensorDesc const& qInputDesc = inputDesc[kIN_Q_IDX];
    PluginTensorDesc const& kInputDesc = inputDesc[kIN_K_IDX];
    PluginTensorDesc const& vInputDesc = inputDesc[kIN_V_IDX];
    int32_t const runtimeBatchSize = static_cast<int32_t>(qInputDesc.dims.d[0]);
    int32_t const runtimeSeqLen = static_cast<int32_t>(qInputDesc.dims.d[1]);
    int32_t const kvSeqLen = static_cast<int32_t>(kInputDesc.dims.d[1]);
    bool const sharedKV = (kvSeqLen == 0);

    check::check(kInputDesc.dims.d[0] == runtimeBatchSize && vInputDesc.dims.d[0] == runtimeBatchSize,
        "Batch size must be consistent across Q/K/V inputs.");
    check::check(kInputDesc.dims.d[1] == vInputDesc.dims.d[1], "K and V sequence lengths must be consistent.");
    if (!sharedKV)
    {
        check::check(
            kvSeqLen == runtimeSeqLen, "K/V sequence length must equal Q sequence length when not in shared-KV mode.");
    }
    check::check(qInputDesc.dims.d[2] == mNumQHeads * mHeadSize, "Q input shape shall be consistent.");
    check::check(kInputDesc.dims.d[2] == mNumKVHeads * mHeadSize, "K input shape shall be consistent.");
    check::check(vInputDesc.dims.d[2] == mNumKVHeads * mHeadSize, "V input shape shall be consistent.");

    rt::Tensor qInputTensor(const_cast<void*>(inputs[kIN_Q_IDX]),
        rt::Coords{runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, rt::DeviceType::kGPU, qInputDesc.type);
    rt::Tensor kInputTensor(const_cast<void*>(inputs[kIN_K_IDX]),
        rt::Coords{runtimeBatchSize, kvSeqLen, mNumKVHeads, mHeadSize}, rt::DeviceType::kGPU, kInputDesc.type);
    rt::Tensor vInputTensor(const_cast<void*>(inputs[kIN_V_IDX]),
        rt::Coords{runtimeBatchSize, kvSeqLen, mNumKVHeads, mHeadSize}, rt::DeviceType::kGPU, vInputDesc.type);

    PluginTensorDesc const& contextLengthInputDesc = inputDesc[kIN_CONTEXT_LENGTH_IDX];
    rt::Tensor const contextLengthTensor(const_cast<void*>(inputs[kIN_CONTEXT_LENGTH_IDX]),
        rt::Coords{contextLengthInputDesc.dims}, rt::DeviceType::kGPU, contextLengthInputDesc.type);

    PluginTensorDesc const& posEncodingCosSinDesc = inputDesc[kIN_ROPE_COS_SIN_IDX];
    rt::Tensor const ropeCosSinTensor(const_cast<void*>(inputs[kIN_ROPE_COS_SIN_IDX]),
        rt::Coords{posEncodingCosSinDesc.dims}, rt::DeviceType::kGPU, posEncodingCosSinDesc.type);

    PluginTensorDesc const& kvCacheStartIdxInputDesc = inputDesc[kIN_KV_CACHE_START_IDX];
    rt::Tensor const kvCacheStartIdxTensor(const_cast<void*>(inputs[kIN_KV_CACHE_START_IDX]),
        rt::Coords{kvCacheStartIdxInputDesc.dims}, rt::DeviceType::kGPU, kvCacheStartIdxInputDesc.type);

    PluginTensorDesc const& attentionOutputDesc = outputDesc[kOUT_ATTENTION_IDX];
    rt::Tensor attentionOutputTensor(outputs[kOUT_ATTENTION_IDX], rt::Coords{attentionOutputDesc.dims},
        rt::DeviceType::kGPU, attentionOutputDesc.type);

    // Construct the KV cache tensors from the past-KV input descriptor. The buffer is the paged
    // pool [2, numPages, 128, Hkv, D] (K page array then V page array); the present-KV output
    // in-place aliases it (output 1 aliases input 3). Shared-KV draft layers are read-only views
    // of the target/base cache, so they must read the input binding rather than the plugin's
    // present-KV output; the present-KV output is not consumed in shared-KV mode and must remain
    // unwritten by every shared-KV path below.
    PluginTensorDesc const& kvCacheInputDesc = inputDesc[kIN_KV_CACHE_IDX];
    rt::Tensor pastKVCacheTensor(const_cast<void*>(inputs[kIN_KV_CACHE_IDX]), rt::Coords{kvCacheInputDesc.dims},
        rt::DeviceType::kGPU, kvCacheInputDesc.type);
    rt::Tensor presentKVCacheTensor(
        outputs[kOUT_KV_CACHE_IDX], rt::Coords{kvCacheInputDesc.dims}, rt::DeviceType::kGPU, kvCacheInputDesc.type);
    rt::Tensor& kvCacheTensor = sharedKV ? pastKVCacheTensor : presentKVCacheTensor;
#if defined(CUTE_DSL_FMHA_ENABLED) || defined(CUTE_DSL_FMHA_V2_ENABLED)
    // numPages feeds runPaged; only the CuTe DSL prefill reads it (writes and XQA decode derive
    // capacity from the page table), so it lives behind the same #ifdef as its call sites.
    int32_t const numPages = static_cast<int32_t>(kvCacheInputDesc.dims.d[1]);

    // Guard against a non-pool-shaped kv_cache binding reaching runPaged (CuTe DSL prefill): that
    // kernel trusts dims.d[1] as numPages, so a legacy/mismatched binding (e.g. the pre-relayout
    // [batch, 2, Hkv, capacity, D] shape) silently misreads numPages and corrupts prefill instead of
    // failing loudly.
    auto validatePagedKVCacheShape = [&]() -> bool {
        bool const ok = isPagedPoolShape(rt::Coords{kvCacheInputDesc.dims}, mNumKVHeads, mHeadSize);
        if (!ok)
        {
            LOG_ERROR(
                "AttentionPlugin: kv_cache binding shape [%ld,%ld,%ld,%ld,%ld] does not match the required "
                "paged-pool contract [2, numPages, %d, %d, %d] (runPaged reads numPages from dim 1); the "
                "export/builder KV-cache binding is not pool-shaped.",
                static_cast<long>(kvCacheInputDesc.dims.d[0]), static_cast<long>(kvCacheInputDesc.dims.d[1]),
                static_cast<long>(kvCacheInputDesc.dims.d[2]), static_cast<long>(kvCacheInputDesc.dims.d[3]),
                static_cast<long>(kvCacheInputDesc.dims.d[4]), rt::kTOKENS_PER_PAGE, mNumKVHeads, mHeadSize);
        }
        return ok;
    };
#endif

    // The runtime-supplied page table [batch, 2, maxPagesPerSeq] carries the K-then-V kernel view
    // (KVPageTable convention: V page id = K page id + numPages). It is a required input with no
    // plugin-internal identity fallback — an empty/missing table is a hard error.
    PluginTensorDesc const& kvPageTableInputDesc = inputDesc[kIN_KV_PAGE_TABLE_IDX];
    rt::Tensor const kvPageTableTensor(const_cast<void*>(inputs[kIN_KV_PAGE_TABLE_IDX]),
        rt::Coords{kvPageTableInputDesc.dims}, rt::DeviceType::kGPU, kvPageTableInputDesc.type);
    check::check(!kvPageTableTensor.isEmpty(), "AttentionPlugin: kv_page_table input is required for paged attention.");
    int32_t const* const pageTable = kvPageTableTensor.dataPointer<int32_t>();
    int32_t const maxPagesPerSeq = static_cast<int32_t>(kvPageTableInputDesc.dims.d[2]);
    // Padded per-slot token capacity spanned by the page table (each page holds kTOKENS_PER_PAGE).
    int32_t const kvCacheCapacity = maxPagesPerSeq * rt::kTOKENS_PER_PAGE;

    // Batch-shaped view of the same pool buffer for the paged applyRopeWriteKV* kernels: they validate
    // and index off a `[B, 2, Hkv, capPadded, D]` descriptor (the flat page pool addressed via the page
    // table) rather than the raw pool binding `[2, numPages, 128, Hkv, D]`. Same base pointer.
    rt::Tensor kvCacheWriteView(outputs[kOUT_KV_CACHE_IDX],
        rt::Coords{runtimeBatchSize, 2, mNumKVHeads, kvCacheCapacity, mHeadSize}, rt::DeviceType::kGPU,
        kvCacheInputDesc.type);

    // Optional Inputs that are not used with Tree Attention enabled.
    rt::Tensor attentionMaskTensor{};
    rt::Tensor attentionPosIdTensor{};
    rt::Tensor visionBlockIdsTensor{};
    if (mEnableTreeAttention)
    {
        PluginTensorDesc const& attentionMaskInputDesc = inputDesc[kIN_OPTIONAL_ATTN_MASK_IDX];
        PluginTensorDesc const& attentionPosIdInputDesc = inputDesc[kIN_OPTIONAL_ATTN_POS_ID_IDX];
        attentionMaskTensor = rt::Tensor(const_cast<void*>(inputs[kIN_OPTIONAL_ATTN_MASK_IDX]),
            rt::Coords{attentionMaskInputDesc.dims}, rt::DeviceType::kGPU, attentionMaskInputDesc.type);
        attentionPosIdTensor = rt::Tensor(const_cast<void*>(inputs[kIN_OPTIONAL_ATTN_POS_ID_IDX]),
            rt::Coords{attentionPosIdInputDesc.dims}, rt::DeviceType::kGPU, attentionPosIdInputDesc.type);
    }
    else if (mEnableVisionBlockAttention)
    {
        PluginTensorDesc const& visionBlockIdsDesc = inputDesc[kIN_OPTIONAL_ATTN_MASK_IDX];
        visionBlockIdsTensor = rt::Tensor(const_cast<void*>(inputs[kIN_OPTIONAL_ATTN_MASK_IDX]),
            rt::Coords{visionBlockIdsDesc.dims}, rt::DeviceType::kGPU, visionBlockIdsDesc.type);
    }
    bool const useExplicitPositionIds = mEnableTreeAttention && !attentionPosIdTensor.isEmpty()
        && attentionPosIdTensor.getShape().getNumDims() == 2 && attentionPosIdTensor.getShape()[1] == runtimeSeqLen;

    float const kScale = mQkvScales[1];
    float const vScale = mQkvScales[2];

    // Determine the attention execution mode based on the input tensors.
    AttentionExecutionMode executionMode{};
    if (!mEnableTreeAttention)
    {
        executionMode = deduceModeVanilla(qInputTensor, kvCacheStartIdxTensor);
    }
    else
    {
        executionMode = deduceModeTreeAttention(qInputTensor, kvCacheStartIdxTensor, attentionPosIdTensor);
    }

    // For invalid execution mode, log error and report error return value.
    if (executionMode == AttentionExecutionMode::kINVALID)
    {
        LOG_ERROR("Invalid attention execution mode detected. Abort the AttentionPlugin enqueue() call.");
        return 1;
    }

    auto* alignedWorkspacePtr = static_cast<std::byte*>(workspace);
    if (alignedWorkspacePtr == nullptr
        || reinterpret_cast<uintptr_t>(alignedWorkspacePtr) % static_cast<uintptr_t>(kDEVICE_ALIGNMENT) != 0)
    {
        LOG_ERROR("Workspace pointer is not aligned to device alignment granularity");
        return 1;
    }

    // ==================== Prefill path ====================
    // Dispatch order: vision-block attention first (FFPA d512 overlay or FMHA
    // CUSTOM_MASK, early return), then sharedKV (early return), then own-KV.
    // Within each: FMHA (standard), FFPA (headSize=512), or reject the
    // prefill when neither kernel serves the head size.
    if (executionMode == AttentionExecutionMode::kNORMAL_PREFILL
        || executionMode == AttentionExecutionMode::kCHUNKED_PREFILL)
    {
        // No prefill backend can read an FP8 donor cache.
        if (mEnableFp8KVCache && sharedKV)
        {
            LOG_ERROR("AttentionPlugin: shared-KV prefill cannot read an FP8 donor cache.");
            return -1;
        }
        // Chunked prefill reads the cache back. The optimized CuTe DSL kernel
        // consumes FP8 directly; dense FMHA-v2 uses splitPagedKV to dequantize
        // the page-table-selected K/V into its FP16 workspace.
        if (mEnableFp8KVCache && executionMode == AttentionExecutionMode::kCHUNKED_PREFILL && !mCanImplementFMHA)
        {
            LOG_ERROR(
                "AttentionPlugin: FP8 KV cache chunked prefill has no direct or split-KV FMHA backend "
                "for headSize=%d on SM %d.",
                mHeadSize, mSMVersion);
            return -1;
        }

        if (mEnableVisionBlockAttention)
        {
            if (executionMode != AttentionExecutionMode::kNORMAL_PREFILL || sharedKV)
            {
                LOG_ERROR(
                    "AttentionPlugin: Gemma4 vision-block attention supports only normal prefill with owned KV "
                    "(mode=%d, sharedKV=%d).",
                    static_cast<int32_t>(executionMode), static_cast<int32_t>(sharedKV));
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

            // Keep K input unmodified: Gemma4 full-attention layers may alias
            // V=K before RoPE. The cache receives roped K and the original V,
            // and both vision prefill paths read K/V from that canonical
            // cache layout.
            kernel::launchApplyRopeWriteKV(ropeCosSinTensor, std::nullopt, qInputTensor, kInputTensor, vInputTensor,
                kvCacheWriteView, kScale, vScale, stream, false, pageTable, maxPagesPerSeq);

#ifdef CUTE_DSL_FFPA_ENABLED
            if (canUseFFPAOverlayForVisionPrefill())
            {
                // Production path for the full-causal d512 global layers: the
                // FFPA vision-block overlay kernel with per-row [blockBegin,
                // blockEnd] intervals expanded from vision_block_ids.
                LOG_DEBUG(
                    "AttentionPlugin: vision-block prefill via FFPA d512 overlay (B=%d, S=%d, Hq=%d, Hkv=%d, cap=%d)",
                    runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads, kvCacheCapacity);

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

                // Expand vision_block_ids into per-position block intervals
                // (-1/-1 sentinel for text and padding positions).
                rt::Tensor blockBeginTensor = assignTensorFromWorkspace(
                    alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen}, DataType::kINT32);
                rt::Tensor blockEndTensor = assignTensorFromWorkspace(
                    alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen}, DataType::kINT32);
                kernel::launchBuildVisionBlockRanges(visionBlockIdsTensor.dataPointer<int32_t>(),
                    contextLengthTensor.dataPointer<int32_t>(), blockBeginTensor.dataPointer<int32_t>(),
                    blockEndTensor.dataPointer<int32_t>(), runtimeBatchSize, runtimeSeqLen, stream);

                // Read K/V back from the just-updated paged pool (roped K,
                // original V — required because V may alias K on the K=V
                // layers).  Compact gather (first runtimeSeqLen tokens) keeps
                // the physical batch stride at runtimeSeqLen; the logical KV
                // length per batch is the same as the Q length (normal
                // prefill, bottom-right offset 0), so cuQSeqLens bounds both
                // sides.
                auto [kSplit, vSplit]
                    = splitPagedKV(kvCacheTensor, pageTable, kvCacheEndIdxsTensor.dataPointer<int32_t>(),
                        maxPagesPerSeq, alignedWorkspacePtr, runtimeBatchSize, mNumKVHeads, kvCacheCapacity, mHeadSize,
                        /*seqLen=*/runtimeSeqLen, kScale, vScale, stream);
                CuteDslFFPAParams ffpaParams{};
                ffpaParams.q = qInputTensor.dataPointer<half>();
                ffpaParams.k = kSplit.dataPointer<half>();
                ffpaParams.v = vSplit.dataPointer<half>();
                ffpaParams.o = attentionOutputTensor.dataPointer<half>();
                ffpaParams.cuSeqLenQ = cuQSeqLensTensor.dataPointer<int32_t>();
                ffpaParams.cuSeqLenK = cuQSeqLensTensor.dataPointer<int32_t>();
                ffpaParams.blockBegin = blockBeginTensor.dataPointer<int32_t>();
                ffpaParams.blockEnd = blockEndTensor.dataPointer<int32_t>();
                ffpaParams.batchSize = runtimeBatchSize;
                ffpaParams.seqlenQ = runtimeSeqLen;
                ffpaParams.seqlenK = runtimeSeqLen;
                ffpaParams.numQHeads = mNumQHeads;
                ffpaParams.numKVHeads = mNumKVHeads;
                ffpaParams.headDim = mHeadSize;
                ffpaParams.softmaxScale = mAttentionScale;
                CuteDslFFPARunner::run(ffpaParams, stream);
                return 0;
            }
#endif

            // Production path for the sliding d256-class layers: FMHA_v2 with
            // ContextAttentionMaskType::CUSTOM_MASK.  The sliding-causal OR
            // same-vision-block predicate is packed into the FMHA bitmask by
            // launchBuildVisionPackedMask, so the kernel needs no window
            // parameter.  Cubin availability is a hard construction-time
            // requirement (enforceVisionBlockKernelSupport); only the exact
            // per-sequence-length kernel (tiled vs non-tiled) is probed here.
            auto fmhaRunner = ContextFMHARunner(mDataType, runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads,
                mHeadSize, mSMVersion, AttentionInputLayout::SEPARATE_Q_K_V, ContextAttentionMaskType::CUSTOM_MASK);
            if (!fmhaRunner.isKernelAvailable())
            {
                LOG_ERROR(
                    "AttentionPlugin: no FMHA CUSTOM_MASK kernel for S=%d (headSize=%d, SM=%d); vision-block "
                    "prefill has no fallback path.",
                    runtimeSeqLen, mHeadSize, mSMVersion);
                return 1;
            }

            LOG_DEBUG(
                "AttentionPlugin: vision-block prefill via FMHA CUSTOM_MASK (B=%d, S=%d, Hq=%d, Hkv=%d, "
                "window=%d)",
                runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads, mSlidingWindowSize);

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

            // Pack the vision-block mask into the FMHA CUSTOM_MASK bit
            // layout (sliding window folded into the bits).
            int64_t const packedMaskWords
                = kernel::getPackedMaskSizeInWords(runtimeBatchSize, runtimeSeqLen, runtimeSeqLen);
            rt::Tensor packedMaskTensor
                = assignTensorFromWorkspace(alignedWorkspacePtr, {packedMaskWords}, DataType::kINT32);
            rt::Tensor cuMaskRowsTensor
                = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);
            kernel::launchBuildVisionPackedMask(visionBlockIdsTensor.dataPointer<int32_t>(),
                contextLengthTensor.dataPointer<int32_t>(),
                reinterpret_cast<uint32_t*>(packedMaskTensor.dataPointer<int32_t>()),
                cuMaskRowsTensor.dataPointer<int32_t>(), runtimeBatchSize, runtimeSeqLen, mSlidingWindowSize, stream);

            // Read K/V back from the just-updated paged pool (roped K,
            // original V — required because V may alias K on the K=V
            // layers).  Compact gather keeps the physical batch
            // stride at runtimeSeqLen, matching s_kv below.
            auto [kSplit, vSplit] = splitPagedKV(kvCacheTensor, pageTable, kvCacheEndIdxsTensor.dataPointer<int32_t>(),
                maxPagesPerSeq, alignedWorkspacePtr, runtimeBatchSize, mNumKVHeads, kvCacheCapacity, mHeadSize,
                /*seqLen=*/runtimeSeqLen, kScale, vScale, stream);

            FusedMultiheadAttentionParamsV2 params{};
            fmhaRunner.setupParams(params, mAttentionScale);
            params.s_kv = runtimeSeqLen;
            params.q_ptr = qInputTensor.dataPointer<half>();
            params.k_ptr = kSplit.dataPointer<half>();
            params.v_ptr = vSplit.dataPointer<half>();
            params.o_ptr = attentionOutputTensor.dataPointer<half>();
            params.cu_q_seqlens = cuQSeqLensTensor.dataPointer<int32_t>();
            params.cu_kv_seqlens = cuQSeqLensTensor.dataPointer<int32_t>();
            params.cu_mask_rows = cuMaskRowsTensor.dataPointer<int32_t>();
            params.packed_mask_ptr = packedMaskTensor.rawPointer();
            params.packed_mask_stride_in_bytes = kernel::getPackedMaskRowStrideInBytes(runtimeSeqLen);
            fmhaRunner.dispatchFMHAKernel(params, stream);
            return 0;
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

        if (sharedKV)
        {
            if (useExplicitPositionIds)
            {
                kernel::launchApplyRopeQOnlyTreeDecoding(ropeCosSinTensor, attentionPosIdTensor, qInputTensor, stream);
            }
            else
            {
                kernel::launchApplyRopeQOnly(ropeCosSinTensor, kvCacheEndIdxsTensor, qInputTensor, stream);
            }

            // Shared-KV without FMHA cubins: FFPA for headSize=512, reject any
            // other head size.
            if (!mCanImplementFMHA)
            {
#ifdef CUTE_DSL_FFPA_ENABLED
                if (!mCanImplementFFPA)
                {
                    LOG_ERROR("AttentionPlugin: no prefill kernel for headSize=%d (FMHA unsupported%s).", mHeadSize,
                        mHeadSize == 512 ? ", FFPA module failed to load" : "; FFPA serves headSize=512 only");
                    return -1;
                }

                LOG_DEBUG(
                    "AttentionPlugin: headSize=512 shared-KV prefill via FFPA native GQA "
                    "(B=%d, S=%d, Hq=%d, Hkv=%d, D=%d, cap=%d)",
                    runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads, mHeadSize, kvCacheCapacity);

                // Per-batch cu_seqlens bound the logical lengths inside the kernel
                // Ragged padding keys/rows are masked and the boundary
                // tile is zero-filled, so no output zeroing WAR is needed.
                // Assemble split K/V from the donor's paged pool via the page-table-aware
                // gather (zero-fills unmapped in-range pages). Chunked prefill gathers the
                // full capacity so the Q chunk can attend the cache prefix, with per-batch
                // cuKVSeqLens (prefix + chunk) driving the bottom-right causal offset
                // inside the kernel; normal prefill gathers compactly (first runtimeSeqLen
                // tokens) so the physical stride matches seqlenK.
                bool const chunked = (executionMode == AttentionExecutionMode::kCHUNKED_PREFILL);
                auto [kSplit, vSplit]
                    = splitPagedKV(kvCacheTensor, pageTable, kvCacheEndIdxsTensor.dataPointer<int32_t>(),
                        maxPagesPerSeq, alignedWorkspacePtr, runtimeBatchSize, mNumKVHeads, kvCacheCapacity, mHeadSize,
                        /*seqLen=*/chunked ? 0 : runtimeSeqLen, kScale, vScale, stream);
                dispatchFFPAKernel(qInputTensor.dataPointer<half>(), kSplit.dataPointer<half>(),
                    vSplit.dataPointer<half>(), attentionOutputTensor.dataPointer<half>(),
                    cuQSeqLensTensor.dataPointer<int32_t>(), cuKVSeqLensTensor.dataPointer<int32_t>(), runtimeBatchSize,
                    runtimeSeqLen, chunked ? kvCacheCapacity : runtimeSeqLen, stream);
#else
                LOG_ERROR(
                    "AttentionPlugin: no prefill kernel for headSize=%d shared-KV prefill (FMHA unsupported, "
                    "built without CUTE_DSL_FFPA_ENABLED).",
                    mHeadSize);
                return -1;
#endif
                return 0;
            }

            // Run FMHA reading from the donor's KV cache (bound to this layer's KV cache input).
#ifdef CUTE_DSL_FMHA_ENABLED
            if (mUseCuteDslFMHA)
            {
                if (!validatePagedKVCacheShape())
                {
                    return 1;
                }

                // CuTe DSL FMHA reads interleaved KV cache natively.
                // windowSizeLeft excludes the query itself; sliding_window_size counts it
                // (last W keys, the XQA/HF convention), so pass W - 1.
                int32_t const slidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize - 1 : INT_MAX;
                CuteDslFMHARunner runner(
                    mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);
                runner.runPaged(qInputTensor.dataPointer<half>(), // Q  [b, s_q, h_q, d]
                    kvCacheTensor.rawPointer(),                   // paged KV pool [2, numPages, 128, h_k, d] (donor)
                    pageTable,                                    // page table [b, 2, maxPagesPerSeq]
                    attentionOutputTensor.dataPointer<half>(),    // O  [b, s_q, h_q, d]
                    paddedCuKVSeqLensTensor.dataPointer<int32_t>(), 2 * numPages, maxPagesPerSeq, rt::kTOKENS_PER_PAGE,
                    kvCacheTensor.getDataType(), stream, mAttentionScale, slidingWindow);
            }
            else
#endif
#ifdef CUTE_DSL_FMHA_V2_ENABLED
                if (mUseCuteDslFMHAV2)
            {
                if (!validatePagedKVCacheShape())
                {
                    return 1;
                }

                int32_t const slidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize - 1 : INT_MAX;
                CuteDslFMHAV2Runner runner(
                    mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);
                if (!runner.runPaged(qInputTensor.dataPointer<half>(), kvCacheTensor.rawPointer(), pageTable,
                        attentionOutputTensor.dataPointer<half>(), cuQSeqLensTensor.dataPointer<int32_t>(),
                        cuKVSeqLensTensor.dataPointer<int32_t>(), 2 * numPages, maxPagesPerSeq, rt::kTOKENS_PER_PAGE,
                        stream, mAttentionScale, slidingWindow))
                {
                    return -1;
                }
            }
            else
#endif
            {
                auto fmhaRunner = ContextFMHARunner(mDataType, runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads,
                    mHeadSize, mSMVersion, AttentionInputLayout::SEPARATE_Q_K_V,
                    mSlidingWindowSize > 0 ? ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL
                                           : ContextAttentionMaskType::CAUSAL);
                FusedMultiheadAttentionParamsV2 params{};
                fmhaRunner.setupParams(params, mAttentionScale);
                if (mSlidingWindowSize > 0)
                {
                    params.sliding_window_size = mSlidingWindowSize;
                }
                params.cu_q_seqlens = cuQSeqLensTensor.dataPointer<int32_t>();

                // Normal prefill: compact gather (seqLen tokens) so FMHA_v2's
                // s_kv-derived batch stride matches the physical layout.
                // Chunked prefill: full gather (seqLen = 0), s_kv = kvCacheCapacity.
                bool const compact = (executionMode == AttentionExecutionMode::kNORMAL_PREFILL);
                auto [kSplit, vSplit]
                    = splitPagedKV(kvCacheTensor, pageTable, kvCacheEndIdxsTensor.dataPointer<int32_t>(),
                        maxPagesPerSeq, alignedWorkspacePtr, runtimeBatchSize, mNumKVHeads, kvCacheCapacity, mHeadSize,
                        /*seqLen=*/compact ? runtimeSeqLen : 0, kScale, vScale, stream);
                params.s_kv = compact ? runtimeSeqLen : kvCacheCapacity;
                params.q_ptr = qInputTensor.dataPointer<half>();
                params.k_ptr = kSplit.dataPointer<half>();
                params.v_ptr = vSplit.dataPointer<half>();
                params.cu_kv_seqlens = cuKVSeqLensTensor.dataPointer<int32_t>();
                params.o_ptr = attentionOutputTensor.dataPointer<half>();
                fmhaRunner.dispatchFMHAKernel(params, stream);
            }
            return 0;
        }

        // --- Own KV prefill: RoPE Q+K, write K/V to cache, then run attention kernel ---

        // Own-KV without FMHA cubins: FFPA for headSize=512, reject any other
        // head size.
        if (!mCanImplementFMHA)
        {
            // headSize=512 prefill: apply RoPE, write K/V to cache, then use FFPA.
            kernel::launchApplyRopeWriteKV(ropeCosSinTensor, kvCacheEndIdxsTensor, qInputTensor, kInputTensor,
                vInputTensor, kvCacheWriteView, kScale, vScale, stream, true, pageTable, maxPagesPerSeq);

#ifdef CUTE_DSL_FFPA_ENABLED
            if (!mCanImplementFFPA)
            {
                LOG_ERROR("AttentionPlugin: no prefill kernel for headSize=%d (FMHA unsupported%s).", mHeadSize,
                    mHeadSize == 512 ? ", FFPA module failed to load" : "; FFPA serves headSize=512 only");
                return -1;
            }

            // Use FFPA d512 causal kernel with native GQA support (no K/V expansion needed).
            LOG_DEBUG(
                "AttentionPlugin: headSize=512 own-KV prefill via FFPA native GQA "
                "(B=%d, S=%d, Hq=%d, Hkv=%d, D=%d)",
                runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads, mHeadSize);

            if (executionMode == AttentionExecutionMode::kCHUNKED_PREFILL)
            {
                // Chunked prefill: the Q chunk must also attend the KV-cache prefix, so
                // gather K/V back from the just-updated paged pool (full capacity;
                // zero-fills unmapped in-range pages).  Per-batch cuKVSeqLens
                // (prefix + chunk) drive the bottom-right causal offset inside the kernel.
                auto [kSplit, vSplit]
                    = splitPagedKV(kvCacheTensor, pageTable, kvCacheEndIdxsTensor.dataPointer<int32_t>(),
                        maxPagesPerSeq, alignedWorkspacePtr, runtimeBatchSize, mNumKVHeads, kvCacheCapacity, mHeadSize,
                        /*seqLen=*/0, kScale, vScale, stream);
                dispatchFFPAKernel(qInputTensor.dataPointer<half>(), kSplit.dataPointer<half>(),
                    vSplit.dataPointer<half>(), attentionOutputTensor.dataPointer<half>(),
                    cuQSeqLensTensor.dataPointer<int32_t>(), cuKVSeqLensTensor.dataPointer<int32_t>(), runtimeBatchSize,
                    runtimeSeqLen, kvCacheCapacity, stream);
            }
            else
            {
                // Normal prefill: K/V inputs are the current (right-padded) sequences, so
                // per-batch Q and KV logical lengths coincide (offset 0) — pass cuQSeqLens
                // for both and ragged padding rows/keys are masked.
                dispatchFFPAKernel(qInputTensor.dataPointer<half>(), kInputTensor.dataPointer<half>(),
                    vInputTensor.dataPointer<half>(), attentionOutputTensor.dataPointer<half>(),
                    cuQSeqLensTensor.dataPointer<int32_t>(), cuQSeqLensTensor.dataPointer<int32_t>(), runtimeBatchSize,
                    runtimeSeqLen, runtimeSeqLen, stream);
            }
#else
            LOG_ERROR(
                "AttentionPlugin: no prefill kernel for headSize=%d own-KV prefill (FMHA unsupported, built "
                "without CUTE_DSL_FFPA_ENABLED).",
                mHeadSize);
            return -1;
#endif
        }
        else
        {
#ifdef CUTE_DSL_FMHA_ENABLED
            if (mUseCuteDslFMHA)
            {
                if (!validatePagedKVCacheShape())
                {
                    return 1;
                }

                // CuTe DSL FMHA uses SplitQKV RoPE variant that writes K/V to interleaved cache.
                float const qScale = mQkvScales[0];
                // windowSizeLeft excludes the query itself; sliding_window_size counts it
                // (last W keys, the XQA/HF convention), so pass W - 1.
                int32_t const slidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize - 1 : INT_MAX;

                CuteDslFMHARunner runner(
                    mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);

                if (mEnableFp8KVCache)
                {
                    // FP8 Q workspace: RoPE kernel quantizes roped Q to FP8 using calibrated qScale.
                    rt::Tensor fp8QTensor = assignTensorFromWorkspace(
                        alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kFP8);

                    // Single kernel: RoPE Q → FP8 output, RoPE K + write FP8 K/V to the paged pool.
                    kernel::launchApplyRopeWriteKVSplitQKV(ropeCosSinTensor, kvCacheEndIdxsTensor, qInputTensor,
                        kInputTensor, vInputTensor, kvCacheWriteView, kScale, vScale, stream, pageTable, maxPagesPerSeq,
                        fp8QTensor.rawPointer(), qScale);

                    runner.runPaged(fp8QTensor.rawPointer(),       // Q  [b, s_q, h_q, d] FP8
                        kvCacheTensor.rawPointer(),                // paged KV pool [2, numPages, 128, h_k, d] FP8
                        pageTable,                                 // page table [b, 2, maxPagesPerSeq]
                        attentionOutputTensor.dataPointer<half>(), // O  [b, s_q, h_q, d] FP16
                        paddedCuKVSeqLensTensor.dataPointer<int32_t>(), 2 * numPages, maxPagesPerSeq,
                        rt::kTOKENS_PER_PAGE, kvCacheTensor.getDataType(), stream, mAttentionScale, slidingWindow,
                        /*fp8Input=*/true, qScale, kScale, vScale);
                }
                else
                {
                    // FP16 path: RoPE Q in-place, write FP16 K/V to the paged pool.
                    kernel::launchApplyRopeWriteKVSplitQKV(ropeCosSinTensor, kvCacheEndIdxsTensor, qInputTensor,
                        kInputTensor, vInputTensor, kvCacheWriteView, kScale, vScale, stream, pageTable, maxPagesPerSeq,
                        /*fp8QOut=*/nullptr, /*qScale=*/1.0f);

                    runner.runPaged(qInputTensor.dataPointer<half>(), // Q  [b, s_q, h_q, d]
                        kvCacheTensor.rawPointer(),                   // paged KV pool [2, numPages, 128, h_k, d]
                        pageTable,                                    // page table [b, 2, maxPagesPerSeq]
                        attentionOutputTensor.dataPointer<half>(),    // O  [b, s_q, h_q, d]
                        paddedCuKVSeqLensTensor.dataPointer<int32_t>(), 2 * numPages, maxPagesPerSeq,
                        rt::kTOKENS_PER_PAGE, kvCacheTensor.getDataType(), stream, mAttentionScale, slidingWindow);
                }
            }
            else
#endif
#ifdef CUTE_DSL_FMHA_V2_ENABLED
                if (mUseCuteDslFMHAV2)
            {
                if (!validatePagedKVCacheShape())
                {
                    return 1;
                }

                kernel::launchApplyRopeWriteKVSplitQKV(ropeCosSinTensor, kvCacheEndIdxsTensor, qInputTensor,
                    kInputTensor, vInputTensor, kvCacheWriteView, kScale, vScale, stream, pageTable, maxPagesPerSeq,
                    /*fp8QOut=*/nullptr, /*qScale=*/1.0f);

                int32_t const slidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize - 1 : INT_MAX;
                CuteDslFMHAV2Runner runner(
                    mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);
                if (!runner.runPaged(qInputTensor.dataPointer<half>(), kvCacheTensor.rawPointer(), pageTable,
                        attentionOutputTensor.dataPointer<half>(), cuQSeqLensTensor.dataPointer<int32_t>(),
                        cuKVSeqLensTensor.dataPointer<int32_t>(), 2 * numPages, maxPagesPerSeq, rt::kTOKENS_PER_PAGE,
                        stream, mAttentionScale, slidingWindow))
                {
                    return -1;
                }
            }
            else
#endif
            {
                auto fmhaRunner = ContextFMHARunner(mDataType, runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads,
                    mHeadSize, mSMVersion, AttentionInputLayout::SEPARATE_Q_K_V,
                    mSlidingWindowSize > 0 ? ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL
                                           : ContextAttentionMaskType::CAUSAL);
                FusedMultiheadAttentionParamsV2 params{};
                fmhaRunner.setupParams(params, mAttentionScale);
                if (mSlidingWindowSize > 0)
                {
                    params.sliding_window_size = mSlidingWindowSize;
                }
                params.cu_q_seqlens = cuQSeqLensTensor.dataPointer<int32_t>();

                if (executionMode == AttentionExecutionMode::kCHUNKED_PREFILL)
                {
                    // Chunked: RoPE + write to the paged pool, then assemble split K/V for FMHA_v2.
                    kernel::launchApplyRopeWriteKV(ropeCosSinTensor, kvCacheEndIdxsTensor, qInputTensor, kInputTensor,
                        vInputTensor, kvCacheWriteView, kScale, vScale, stream, false, pageTable, maxPagesPerSeq);

                    // Chunked prefill: full gather (seqLen = 0) — cu_kv_seqlens spans the whole
                    // cache context, so s_kv (and the split K/V batch stride) is the capacity.
                    auto [kSplit, vSplit]
                        = splitPagedKV(kvCacheTensor, pageTable, kvCacheEndIdxsTensor.dataPointer<int32_t>(),
                            maxPagesPerSeq, alignedWorkspacePtr, runtimeBatchSize, mNumKVHeads, kvCacheCapacity,
                            mHeadSize, /*seqLen=*/0, kScale, vScale, stream);
                    params.s_kv = kvCacheCapacity;
                    params.q_ptr = qInputTensor.dataPointer<half>();
                    params.k_ptr = kSplit.dataPointer<half>();
                    params.v_ptr = vSplit.dataPointer<half>();
                    params.cu_kv_seqlens = cuKVSeqLensTensor.dataPointer<int32_t>();
                    params.o_ptr = attentionOutputTensor.dataPointer<half>();
                }
                else
                { // SEPARATE_Q_K_V
                    kernel::launchApplyRopeWriteKV(ropeCosSinTensor, std::nullopt, qInputTensor, kInputTensor,
                        vInputTensor, kvCacheWriteView, kScale, vScale, stream, true, pageTable, maxPagesPerSeq);

                    params.s_kv = runtimeSeqLen;
                    params.q_ptr = qInputTensor.dataPointer<half>();
                    params.k_ptr = kInputTensor.dataPointer<half>();
                    params.v_ptr = vInputTensor.dataPointer<half>();
                    params.cu_kv_seqlens = cuQSeqLensTensor.dataPointer<int32_t>();
                    params.o_ptr = attentionOutputTensor.dataPointer<half>();
                }

                // Dispatch FMHA kernel
                fmhaRunner.dispatchFMHAKernel(params, stream);
            }
        } // end mCanImplementFMHA
    }
    else
    {
        // Apply RoPE and (optionally) write K/V to cache.
        // Shared KV: RoPE Q only, skip KV write (donor's cache is already populated).
        // Non-shared: RoPE Q+K, write K/V to cache.
        if (executionMode == AttentionExecutionMode::kTREE_DECODING)
        {
            if (sharedKV)
            {
                kernel::launchApplyRopeQOnlyTreeDecoding(ropeCosSinTensor, attentionPosIdTensor, qInputTensor, stream);
            }
            else
            {
                kernel::launchApplyRopeWriteKVTreeDecoding(ropeCosSinTensor, contextLengthTensor, attentionPosIdTensor,
                    qInputTensor, kInputTensor, vInputTensor, kvCacheWriteView, kScale, vScale, stream, pageTable,
                    maxPagesPerSeq);
            }
        }
        else
        {
            if (sharedKV)
            {
                if (useExplicitPositionIds)
                {
                    kernel::launchApplyRopeQOnlyTreeDecoding(
                        ropeCosSinTensor, attentionPosIdTensor, qInputTensor, stream);
                }
                else
                {
                    kernel::launchApplyRopeQOnly(ropeCosSinTensor, contextLengthTensor, qInputTensor, stream);
                }
            }
            else
            {
                kernel::launchApplyRopeWriteKV(ropeCosSinTensor, contextLengthTensor, qInputTensor, kInputTensor,
                    vInputTensor, kvCacheWriteView, kScale, vScale, stream, false, pageTable, maxPagesPerSeq);
            }
        }

        // Vision-block decode goes to XQA (a hard construction-time
        // requirement): newly generated tokens are text, so decode is pure
        // causal/sliding — identical masking to the non-vision path.
        // Cache-state parity: the vision prefill paths write the cache with
        // launchApplyRopeWriteKV (roped K + original V into the paged pool
        // via the batch-shaped write view) and the decode RoPE above appended the
        // new token the same way as the non-vision path, so XQA reads exactly
        // the cache state it would see without vision.

        // XQA decode kernel dispatch.
        auto xqaRunner = DecoderXQARunner(mDataType, selectKvCacheDataType(mEnableFp8KVCache), runtimeBatchSize,
            mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion);
        XQALaunchParams params = xqaRunner.initXQAParams();
        params.attentionScale = mAttentionScale;
        if (mEnableFp8KVCache)
        {
            params.kScale = kScale;
            params.vScale = vScale;
        }
        params.output = attentionOutputTensor.dataPointer<half>();
        params.qInputPtr = qInputTensor.dataPointer<half>();
        // Paged KV cache (layout-1): pool base + page table. maxNbPagesPerSeq = capacity / tokensPerPage,
        // so capacity is the padded per-slot capacity (maxPagesPerSeq * kTOKENS_PER_PAGE).
        params.kvCache.data = kvCacheTensor.rawPointer();
        params.kvCache.sequence_lengths = contextLengthTensor.dataPointer<int32_t>();
        params.kvCache.capacity = static_cast<uint32_t>(kvCacheCapacity);
        params.kvCache.pageList = pageTable;
        params.kvCache.tokensPerPage = static_cast<uint32_t>(rt::kTOKENS_PER_PAGE);
        params.slidingWinSize = mSlidingWindowSize > 0 ? static_cast<uint32_t>(mSlidingWindowSize) : 0U;
        if (executionMode == AttentionExecutionMode::kTREE_DECODING)
        {
            // Execute tree attention decoding.
            params.treeAttnMask = attentionMaskTensor.dataPointer<int32_t>();
            params.qSeqLen = runtimeSeqLen;
            xqaRunner.dispatchSpecDecodeXQAKernel(params, stream);
        }
        else
        {
            // Execute vanilla decoding.
            xqaRunner.dispatchXQAKernel(params, stream);
        }
    }
    return 0;
}

int32_t AttentionPlugin::onShapeChange([[maybe_unused]] PluginTensorDesc const* in, [[maybe_unused]] int32_t nbInputs,
    [[maybe_unused]] PluginTensorDesc const* out, [[maybe_unused]] int32_t nbOutputs) noexcept
{
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
    mDataToSerialize.emplace_back("enable_fp8_kv_cache", &mEnableFp8KVCache, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(
        "enable_vision_block_attention", &mEnableVisionBlockAttention, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("sliding_window_size", &mSlidingWindowSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(
        "qkv_scales", mQkvScales.data(), PluginFieldType::kFLOAT32, static_cast<int32_t>(mQkvScales.size()));
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
    mPluginAttributes.emplace_back(PluginField("enable_fp8_kv_cache", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("enable_vision_block_attention", nullptr, PluginFieldType::kINT32, 0));
    // Sliding window size (-1 = no sliding window, >0 = window size)
    mPluginAttributes.emplace_back(PluginField("sliding_window_size", nullptr, PluginFieldType::kINT32, 0));
    // Optional QKV dequant scales [q, k, v] for FP8 attention
    mPluginAttributes.emplace_back(PluginField("qkv_scales", nullptr, PluginFieldType::kFLOAT32, 0));
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
    char const* name, PluginFieldCollection const* fc, [[maybe_unused]] TensorRTPhase phase) noexcept
{
    try
    {
        auto* plugin = new AttentionPlugin(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create AttentionPlugin: %s", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
