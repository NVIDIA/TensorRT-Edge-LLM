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
#include "kernels/contextAttentionKernels/attentionScaleUtils.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "plugins/utils/pluginUtils.h"

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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
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
//! Self-describing blob of (XQAJitKey, cubin) pairs; see serializeXQAJitKernels.
constexpr char const* kXQA_JIT_KERNELS_FIELD{"xqa_jit_kernels"};

// Select KV cache storage datatype based on FP8 enablement
static inline DataType selectKvCacheDataType(bool enableFp8KVCache)
{
    return enableFp8KVCache ? DataType::kFP8 : DataType::kHALF;
}

bool isFp8KVCacheSupportedSM(int32_t smVersion)
{
    // FP8 KV cache needs an FP8 XQA decode kernel, which is available only for these SMs.
    return smVersion == 89 || smVersion == 90 || smVersion == 100 || smVersion == 101 || smVersion == 110
        || smVersion == 120 || smVersion == 121;
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
//
// The gamma weights are engine weights: FP16 Constant initializers wired to plugin inputs,
// baked into the engine at build time (device-resident, no runtime upload). Models without
// qk_norm do not wire these inputs at all.
constexpr int32_t kIN_QKV_IDX{0};
constexpr int32_t kIN_KV_CACHE_IDX{1};
constexpr int32_t kIN_CONTEXT_LENGTH_IDX{2};
constexpr int32_t kIN_ROPE_COS_SIN_IDX{3};
constexpr int32_t kIN_KV_CACHE_START_IDX{4};
constexpr int32_t kIN_KV_PAGE_TABLE_IDX{5};
constexpr int32_t kOUT_ATTENTION_IDX{0};
constexpr int32_t kOUT_KV_CACHE_IDX{1};

// Reflect the count of Inputs and Outputs of the AttentionPlugin,
// these definitions shall be consistent.
constexpr int32_t kNUM_REQUIRED_INPUTS{6};
constexpr int32_t kNUM_QK_NORM_OPTIONAL_INPUTS{2};
constexpr int32_t kNUM_CONTEXT_MASK_SELECTOR_OPTIONAL_INPUTS{1};
constexpr int32_t kNUM_TREE_ATTN_OPTIONAL_INPUTS{2};
constexpr int32_t kNUM_VISION_BLOCK_OPTIONAL_INPUTS{1};
constexpr int32_t kNUM_SKIP_SCALE_OPTIONAL_INPUTS{1};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{2};

int32_t getExpectedNbInputs(int32_t enableQKNorm, int32_t enableContextMaskSelector, int32_t enableTreeAttention,
    int32_t enableVisionBlockAttention, float skipSoftmaxScaleFactor)
{
    return kNUM_REQUIRED_INPUTS + (enableQKNorm ? kNUM_QK_NORM_OPTIONAL_INPUTS : 0)
        + (enableContextMaskSelector ? kNUM_CONTEXT_MASK_SELECTOR_OPTIONAL_INPUTS : 0)
        + (enableTreeAttention ? kNUM_TREE_ATTN_OPTIONAL_INPUTS : 0)
        + (enableVisionBlockAttention ? kNUM_VISION_BLOCK_OPTIONAL_INPUTS : 0)
        + (skipSoftmaxScaleFactor > 0.F ? kNUM_SKIP_SCALE_OPTIONAL_INPUTS : 0);
}

// Dynamic input-index helpers for the optional inputs (positions depend on which optional
// groups are enabled; qk_norm gammas always precede the tree-attention inputs).
constexpr int32_t qNormGammaInputIdx()
{
    return kNUM_REQUIRED_INPUTS; // valid only when enable_qk_norm
}
constexpr int32_t kNormGammaInputIdx()
{
    return kNUM_REQUIRED_INPUTS + 1; // valid only when enable_qk_norm
}
constexpr int32_t contextMaskSelectorInputIdx(bool enableQKNorm)
{
    return kNUM_REQUIRED_INPUTS + (enableQKNorm ? kNUM_QK_NORM_OPTIONAL_INPUTS : 0);
}
constexpr int32_t attnMaskInputIdx(bool enableQKNorm, bool enableContextMaskSelector)
{
    return contextMaskSelectorInputIdx(enableQKNorm)
        + (enableContextMaskSelector ? kNUM_CONTEXT_MASK_SELECTOR_OPTIONAL_INPUTS : 0);
}
constexpr int32_t attnPosIdInputIdx(bool enableQKNorm, bool enableContextMaskSelector)
{
    return attnMaskInputIdx(enableQKNorm, enableContextMaskSelector) + 1;
}
constexpr int32_t skipSoftmaxScaleInputIdx(
    bool enableQKNorm, bool enableContextMaskSelector, bool enableTreeAttention, bool enableVisionBlock)
{
    return attnMaskInputIdx(enableQKNorm, enableContextMaskSelector)
        + (enableTreeAttention ? kNUM_TREE_ATTN_OPTIONAL_INPUTS : 0)
        + (enableVisionBlock ? kNUM_VISION_BLOCK_OPTIONAL_INPUTS : 0);
}

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

    PluginTensorDesc const& selectorDesc = inputDesc[contextMaskSelectorInputIdx(enableQKNorm)];
    return selectorDesc.dims.d[0] > 0 ? RuntimeContextMaskMode::kPadding : RuntimeContextMaskMode::kDefault;
}

bool isPaddingContextMask(RuntimeContextMaskMode const mode)
{
    return mode == RuntimeContextMaskMode::kPadding;
}

AttentionExecutionMode deduceModeVanilla(rt::Tensor const& packedQKVTensor, rt::Tensor const& kvCacheStartIdxTensor)
{
    // Empty KVCache Start indices means normal prefill without previous KVCache. Notice single token is also a valid
    // prefill length.
    if (kvCacheStartIdxTensor.getShape()[0] == 0)
    {
        return AttentionExecutionMode::kNORMAL_PREFILL;
    }

    // Otherwise, distinguish between chunked prefill and vanilla decoding based on the runtime Sequence Length.
    // Vanilla decoding should always have runtime sequence length of 1.
    int64_t const runtimeSeqLen = packedQKVTensor.getShape()[1];
    if (runtimeSeqLen > 1)
    {
        return AttentionExecutionMode::kCHUNKED_PREFILL;
    }
    return AttentionExecutionMode::kVANILLA_DECODING;
}

#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
//! Skip-softmax (BLASST): derive the runtime threshold from the calibrated scale
//! factor S as lambda = S / L, passed to the kernel as log2(lambda). Returns a
//! finite negative log2(lambda) when skip applies, or 0.0 — the runner's disable
//! sentinel (log2 of the degenerate lambda = 1) — when it does not.
float computeSkipSoftmaxThreshold(float scaleFactor, int32_t slidingWindowSize, int32_t kvCacheCapacity)
{
    if (scaleFactor <= 0.F || slidingWindowSize > 0)
    {
        return 0.F;
    }
    float const lambda = scaleFactor / static_cast<float>(std::max(kvCacheCapacity, 1));
    if (lambda >= 1.F)
    {
        // Degenerate threshold (would mark every tile skippable) — run dense instead.
        return 0.F;
    }
    return std::log2(lambda);
}
#endif // CUTE_DSL_FMHA_BLACKWELL_ENABLED

AttentionExecutionMode deduceModeTreeAttention(
    rt::Tensor const& packedQKVTensor, rt::Tensor const& kvCacheStartIdxTensor, rt::Tensor const& attentionPosIdTensor)
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
    int64_t const runtimeSeqLen = packedQKVTensor.getShape()[1];
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

struct FMHAKernelSelection
{
    ContextFMHABackend backend{ContextFMHABackend::kNONE};
    bool canImplement{false};
};

FMHAKernelSelection selectFMHAKernels(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t smVersion,
    nvinfer1::DataType dataType, bool useSlidingWindow, bool enableFp8KVCache)
{
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
    if (CuteDslFMHARunner::canImplement(headSize, smVersion))
    {
        return {ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL, true};
    }
#endif

    CuteDslFMHAV2MaskType const fmhaV2Mask
        = useSlidingWindow ? CuteDslFMHAV2MaskType::kSLIDING_CAUSAL : CuteDslFMHAV2MaskType::kCAUSAL;
    // Native-paged FMHA-v2 is FP16-only. Preserve the FP8 route, including D512, through the dense FMHA-v2 kernels
    // and the page-table-aware split-KV gather.
    bool const canImplementFMHAV2 = enableFp8KVCache
        ? CuteDslFMHAV2Runner::canImplement(numQHeads, numKVHeads, headSize, smVersion, dataType, fmhaV2Mask)
        : CuteDslFMHAV2Runner::canImplementPaged(numQHeads, numKVHeads, headSize, smVersion, dataType, fmhaV2Mask);
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
            numQHeads, numKVHeads, headSize, smVersion, dataType, CuteDslFMHAV2MaskType::kPADDING))
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
//   4     | [B, 2, Hkv, Smax, D]             | HALF  | splitPagedKV out    (FP8/padding/vision)
//   5*    | [B, S, Hq, D]                    | FP8   | fp8Q                (CuTe DSL + FP8 prefill only)
//   6*    | [B, S] x 2                       | INT32 | blockBegin/blockEnd (vision CuTe DSL prefill)
//
//   * Slots 5-6 are conditionally allocated (CuTe DSL + FP8 KV cache /
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

    // Per-position vision-block intervals for CuTe DSL prefill.
    if (enableVisionBlockAttention)
    {
        workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize, seqLen}, DataType::kINT32);
        workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{batchSize, seqLen}, DataType::kINT32);
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
    return tensorDesc.type == (enableFp8KVCache ? DataType::kFP8 : DataType::kHALF)
        && tensorDesc.format == TensorFormat::kLINEAR;
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

bool hasConcretePagedKVContract(Dims const& qkv, Dims const& kvCacheInput, Dims const& kvCacheOutput,
    Dims const& kvPageTable, int32_t numKVHeads, int32_t headSize)
{
    return qkv.nbDims == 3 && qkv.d[0] > 0 && isPagedPoolShape(kvCacheInput, numKVHeads, headSize, false)
        && haveSameShape(kvCacheInput, kvCacheOutput) && kvPageTable.nbDims == 3 && kvPageTable.d[0] == qkv.d[0]
        && kvPageTable.d[1] == 2 && kvPageTable.d[2] > 0 && kvCacheInput.d[1] >= kvPageTable.d[2];
}

bool hasConcretePagedKVContract(PluginTensorDesc const* in, PluginTensorDesc const* out, int32_t numKVHeads,
    int32_t headSize, bool enableFp8KVCache)
{
    return isKVCacheDescriptor(in[kIN_KV_CACHE_IDX], enableFp8KVCache)
        && isKVCacheDescriptor(out[kOUT_KV_CACHE_IDX], enableFp8KVCache)
        && isKVPageTableDescriptor(in[kIN_KV_PAGE_TABLE_IDX])
        && hasConcretePagedKVContract(in[kIN_QKV_IDX].dims, in[kIN_KV_CACHE_IDX].dims, out[kOUT_KV_CACHE_IDX].dims,
            in[kIN_KV_PAGE_TABLE_IDX].dims, numKVHeads, headSize);
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

    DataType const dtype = poolTensor.getDataType();
    // Split-K/V consumers read FP16. FP16 pools byte-copy; FP8 pools dequantize.
    // Any other pool dtype would be reinterpreted as half by the consumers, so reject it loudly.
    bool const isFp8KV = (dtype == DataType::kFP8);
    check::check(dtype == DataType::kHALF || isFp8KV,
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
    rt::Tensor kTensor(
        ptr, rt::Coords{batchSize, outSeqDim, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor(ptr + kTensorElems * sizeof(half), rt::Coords{batchSize, outSeqDim, numKVHeads, headSize},
        rt::DeviceType::kGPU, DataType::kHALF);

    kernel::gatherPagedKVToSplit(poolTensor.rawPointer(), kTensor.rawPointer(), vTensor.rawPointer(), pageTable,
        kvSeqLens, maxPagesPerSeq, batchSize, outSeqDim, numKVHeads, headSize, elemSize, isFp8KV, kScale, vScale,
        stream);

    return std::make_pair(std::move(kTensor), std::move(vTensor));
}

bool AttentionPlugin::canUseCuteDslBidirectionalForPrefill() const noexcept
{
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
    return mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL && mCanImplementFMHA
        && mCanImplementCuteDslBidirectionalFMHA;
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
    int32_t enableContextMaskSelector, int32_t slidingWindowSize, std::vector<float> const& qkvScales,
    std::optional<float> attentionScale)
    : mLayerName(name)
    , mNumQHeads(numQHeads)
    , mNumKVHeads(numKVHeads)
    , mHeadSize(headSize)
    , mAttentionScale(resolveAttentionScale(attentionScale, headSize))
    , mEnableTreeAttention(enableTreeAttention)
    , mEnableVisionBlockAttention(enableVisionBlockAttention)
    , mEnableFp8KVCache(enableFp8KVCache)
    , mEnableContextMaskSelector(enableContextMaskSelector)
    , mQkvScales(enableFp8KVCache ? qkvScales : std::vector<float>{1.f, 1.f, 1.f})
    , mSlidingWindowSize(slidingWindowSize)
{
    // The fused-norm warp reduction covers at most 32 lanes (headDim / vec width 8)
    // => no fused-norm kernel above head_size 256; fail at construction.
    ELLM_CHECK(!mEnableQKNorm || mHeadSize <= 256,
        "enable_qk_norm requires head_size <= 256 (no fused-norm kernel for larger heads).");
    ELLM_CHECK(!(mEnableQKNorm && mEnableKVShared),
        "enable_qk_norm with a shared-KV (Q-only) layer is not supported: the Q-only path has "
        "no fused-norm kernel.");
    ELLM_CHECK(!(mEnableTreeAttention && mEnableVisionBlockAttention),
        "Tree attention and vision block attention are mutually exclusive.");
    ELLM_CHECK(!mEnableVisionBlockAttention || selectKvCacheDataType(mEnableFp8KVCache) == DataType::kHALF,
        "Vision block attention does not support an FP8 KV cache.");
    ELLM_CHECK(!mEnableFp8KVCache || mQkvScales.size() == 3,
        "FP8 KV cache enabled but qkv_scales has "
            + std::to_string(mQkvScales.size()) + " elements (expected 3). "
            "Re-export the model to include QKV scales [q, k, v].");

    mSMVersion = getSMVersion();
    ELLM_CHECK(!mEnableFp8KVCache || isFp8KVCacheSupportedSM(mSMVersion),
        "FP8 KV cache is supported only on SM89, SM90, SM100, SM101, SM110, SM120, and SM121; got SM"
            + std::to_string(mSMVersion) + ".");
    applyThorSMRenumberWAR(mSMVersion);

    FMHAKernelSelection const fmhaSelection = selectFMHAKernels(
        mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion, mDataType, mSlidingWindowSize > 0, mEnableFp8KVCache);
    mContextFMHABackend = fmhaSelection.backend;
    mCanImplementFMHA = fmhaSelection.canImplement;
    LOG_DEBUG("AttentionPlugin FMHA backend: %d, sliding_window: %s", static_cast<int32_t>(mContextFMHABackend),
        mSlidingWindowSize > 0 ? std::to_string(mSlidingWindowSize).c_str() : "disabled");

#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
    mCanImplementCuteDslBidirectionalFMHA = mEnableVisionBlockAttention && mHeadSize == 512
        && mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL && mCanImplementFMHA;
#endif

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
            mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion, mDataType, CuteDslFMHAV2MaskType::kVISION_BLOCK);
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
    , mEnableKVShared(parsePluginScalarField<int32_t>("enable_kv_shared", fc).value_or(0))
    , mEnableFp8KVCache(parsePluginScalarField<int32_t>("enable_fp8_kv_cache", fc).value_or(0))
    , mSlidingWindowSize(parsePluginScalarField<int32_t>("sliding_window_size", fc).value_or(-1))
    , mSkipSoftmaxScaleFactor(parsePluginScalarField<float>("skip_softmax_scale_factor", fc).value_or(0.f))
{
    mEnableVisionBlockAttention = parsePluginScalarField<int32_t>("enable_vision_block_attention", fc).value_or(0);
    mEnableContextMaskSelector = parsePluginScalarField<int32_t>("enable_context_mask_selector", fc).value_or(0);
    // The fused-norm warp reduction covers at most 32 lanes (headDim / vec width 8)
    // => no fused-norm kernel above head_size 256; fail at construction.
    ELLM_CHECK(!mEnableQKNorm || mHeadSize <= 256,
        "enable_qk_norm requires head_size <= 256 (no fused-norm kernel for larger heads).");
    ELLM_CHECK(!(mEnableQKNorm && mEnableKVShared),
        "enable_qk_norm with a shared-KV (Q-only) layer is not supported: the Q-only path has "
        "no fused-norm kernel.");

    ELLM_CHECK(!(mEnableTreeAttention && mEnableVisionBlockAttention),
        "Tree attention and vision block attention are mutually exclusive.");
    ELLM_CHECK(!mEnableVisionBlockAttention || selectKvCacheDataType(mEnableFp8KVCache) == DataType::kHALF,
        "Vision block attention does not support an FP8 KV cache.");

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
    ELLM_CHECK(!mEnableFp8KVCache || isFp8KVCacheSupportedSM(mSMVersion),
        "FP8 KV cache is supported only on SM89, SM90, SM100, SM101, SM110, SM120, and SM121; got SM"
            + std::to_string(mSMVersion) + ".");
    applyThorSMRenumberWAR(mSMVersion);

    FMHAKernelSelection const fmhaSelection = selectFMHAKernels(
        mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion, mDataType, mSlidingWindowSize > 0, mEnableFp8KVCache);
    mContextFMHABackend = fmhaSelection.backend;
    mCanImplementFMHA = fmhaSelection.canImplement;
    LOG_DEBUG("AttentionPlugin FMHA backend: %d", static_cast<int32_t>(mContextFMHABackend));

#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
    mCanImplementCuteDslBidirectionalFMHA = mEnableVisionBlockAttention && mHeadSize == 512
        && mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL && mCanImplementFMHA;
#endif

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
            mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion, mDataType, CuteDslFMHAV2MaskType::kVISION_BLOCK);
        enforceVisionBlockKernelSupport();
    }
}

AttentionPlugin::~AttentionPlugin() = default;

XQAJitKey AttentionPlugin::getXQAJitKey() const noexcept
{
    XQAJitKey key{};
    key.sm = mSMVersion;
    key.dataType = mDataType;
    key.kvDataType = selectKvCacheDataType(mEnableFp8KVCache);
    key.headSize = mHeadSize;
    key.qHeadsPerKv = mNumKVHeads > 0 ? mNumQHeads / mNumKVHeads : 0;
    key.tokensPerPage = rt::kTOKENS_PER_PAGE;
    key.slidingWindow = mSlidingWindowSize > 0;
    key.specDecode = static_cast<bool>(mEnableTreeAttention);
    return key;
}

bool AttentionPlugin::canCompileXQAJitKernel() const noexcept
{
    return canCompileXQAKernel(
        mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion, mDataType, selectKvCacheDataType(mEnableFp8KVCache));
}

void AttentionPlugin::compileXQAJitKernelForBuild()
{
    // A layer with no XQA decode kernel is not fatal on its own: the prefill
    // path may still be served by FMHA, and the constructor already rejects
    // the case where no backend at all can implement the layer.
    if (!canCompileXQAJitKernel())
    {
        LOG_INFO("AttentionPlugin: no XQA decode kernel for Hq=%d, Hkv=%d, headSize=%d on SM%d; skipping XQA JIT.",
            mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion);
        return;
    }

    auto compileAndLoad = [](XQAJitKey const& jitKey) -> std::vector<uint8_t> {
        XQAJitResult result = compileXQAKernel(jitKey);
        ELLM_CHECK(!result.cubin.empty(), "NVRTC returned an empty XQA cubin.");
        bool const loaded
            = DecoderXQARunner::loadDecodeXQAKernelFromCubin(jitKey, result.cubin.data(), result.cubin.size());
        ELLM_CHECK(loaded, "Failed to load the NVRTC-compiled XQA kernel.");
        return std::move(result.cubin);
    };

    XQAJitKey const key = getXQAJitKey();
    mXqaJitKernels.clear();

    // Vanilla decode is always needed: tree-attention engines still execute it
    // for single-token steps and fallback requests.
    XQAJitKey vanillaKey = key;
    vanillaKey.specDecode = false;
    mXqaJitKernels.push_back({vanillaKey, compileAndLoad(vanillaKey)});

    // Only tree-attention engines need a second, distinct spec-decode kernel.
    // Emitting one entry per distinct key (rather than a fixed pair of slots)
    // is what keeps a non-tree layer from serializing the same cubin twice.
    if (key.specDecode)
    {
        mXqaJitKernels.push_back({key, compileAndLoad(key)});
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
            mEnableFp8KVCache, mEnableVisionBlockAttention, mEnableContextMaskSelector, mSlidingWindowSize, mQkvScales,
            mAttentionScale);
        p->mEnableQKNorm = mEnableQKNorm;
        p->mEnableKVShared = mEnableKVShared;
        p->mRmsNormEps = mRmsNormEps;
        p->mSkipSoftmaxScaleFactor = mSkipSoftmaxScaleFactor;
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
        // Output[0] is attention result, has shape [B, S, Hq, D]. Refers to Q shape [B, S, Hq*D]
        outputs[kOUT_ATTENTION_IDX].nbDims = 4;
        // B,S taken from packed QKV input (same as Q's B,S since they're concat-ed on head dim)
        outputs[kOUT_ATTENTION_IDX].d[0] = inputs[kIN_QKV_IDX].d[0];
        outputs[kOUT_ATTENTION_IDX].d[1] = inputs[kIN_QKV_IDX].d[1];
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
    //      packed QKV tensor (linear FP16) with shape [B, S, (Hq + 2*Hkv) * D]
    //      KV-cache tensor (linear FP16/FP8): paged pool [2, numPages, kTOKENS_PER_PAGE, Hkv, D].
    //      (numPages is a fixed value per engine build; see setupKVCacheProfiles in llmBuilder.cpp.)
    //      Real context length: [B] (a vector of scalars) with type int32_t.
    //      RoPE cos/sin cache: [B or 1, Smax, D] (a tensor of scalars) with type float.
    //            Rope CosSin can be ND vector depending on rope type.
    //      Start index of the KVCache [B, 0~1] (a vector of scalars) with type int32_t.
    //            0 length indicates there is no existing KVCache for inference.
    //      Optional context mask selector: [0] for causal/default, [B] for non-causal PADDING mask.
    //      Optional tree attention mask: [B, S, S] (a tensor of scalars) with type int32_t.
    //      Optional tree attention position ids: [B, S] (a tensor of scalars) with type int32_t.

    // Support context/generation phase outputs:
    //      attention result (linear FP16) with shape [B, S, Hq, D]
    //      KV-cache tensor, same as the above.
    // Packed-QKV input channel count: (Hq + 2*Hkv) * D for normal layers, Hq * D for
    // shared-KV layers (Q only). No channel assertion here — Gemma4 uses heterogeneous
    // per-layer head configurations; the exact count is validated in enqueue().
    auto checkPackedQKV = [](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
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
        status &= tensorDesc.dims.nbDims == 3;
        status &= tensorDesc.dims.d[2] <= mHeadSize;
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

    int32_t const expectedNbInputs = getExpectedNbInputs(mEnableQKNorm, mEnableContextMaskSelector,
        mEnableTreeAttention, mEnableVisionBlockAttention, mSkipSoftmaxScaleFactor);
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
        case kIN_CONTEXT_LENGTH_IDX: result = checkSequenceLen(inOut[pos].desc); break;
        case kIN_ROPE_COS_SIN_IDX: result = checkPosEncodingCosSin(inOut[pos].desc); break;
        case kIN_KV_CACHE_START_IDX: result = checkKVCacheStartIdx(inOut[pos].desc); break;
        case kIN_KV_PAGE_TABLE_IDX: result = checkKVPageTable(inOut[pos].desc); break;
        default: break;
        }

        // Handle optional inputs (qk_norm gammas, context selector, then tree/vision inputs) with dynamic ordering.
        if (result && pos >= kNUM_REQUIRED_INPUTS)
        {
            int32_t currentOptionalInputIdx = kNUM_REQUIRED_INPUTS;
            if (mEnableQKNorm)
            {
                if (pos == currentOptionalInputIdx || pos == currentOptionalInputIdx + 1)
                {
                    // Engine-weight constant: FP16 1-D vector of length head_size,
                    // baked into the engine at build time.
                    result = inOut[pos].desc.type == DataType::kHALF && inOut[pos].desc.format == TensorFormat::kLINEAR
                        && inOut[pos].desc.dims.nbDims == 1 && inOut[pos].desc.dims.d[0] == mHeadSize;
                }
                currentOptionalInputIdx += kNUM_QK_NORM_OPTIONAL_INPUTS;
            }
            if (mEnableContextMaskSelector)
            {
                if (pos == currentOptionalInputIdx)
                {
                    result = checkContextMaskSelector(inOut[pos].desc);
                }
                currentOptionalInputIdx += kNUM_CONTEXT_MASK_SELECTOR_OPTIONAL_INPUTS;
            }
            if (mEnableTreeAttention)
            {
                if (pos == currentOptionalInputIdx)
                {
                    result = checkAttentionMask(inOut[pos].desc);
                }
                if (pos == currentOptionalInputIdx + 1)
                {
                    result = checkAttentionPosId(inOut[pos].desc);
                }
                currentOptionalInputIdx += kNUM_TREE_ATTN_OPTIONAL_INPUTS;
            }
            if (mEnableVisionBlockAttention)
            {
                if (pos == currentOptionalInputIdx)
                {
                    result = checkVisionBlockIds(inOut[pos].desc);
                }
                currentOptionalInputIdx += kNUM_VISION_BLOCK_OPTIONAL_INPUTS;
            }
            if (mSkipSoftmaxScaleFactor > 0.F && pos == currentOptionalInputIdx)
            {
                // Shape-only carrier: 1-D INT8 dummy whose length encodes the runtime
                // skip-softmax scale-factor override. Length is dynamic (>= 0).
                result = inOut[pos].desc.type == DataType::kINT8 && inOut[pos].desc.format == TensorFormat::kLINEAR
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
    int32_t const expectedNbInputs = getExpectedNbInputs(mEnableQKNorm, mEnableContextMaskSelector,
        mEnableTreeAttention, mEnableVisionBlockAttention, mSkipSoftmaxScaleFactor);
    if (in == nullptr || out == nullptr || nbInputs != expectedNbInputs || nbOutputs != kNUM_REQUIRED_OUTPUTS)
    {
        LOG_ERROR("AttentionPlugin: expected %d inputs and %d outputs, but got %d inputs and %d outputs.",
            expectedNbInputs, kNUM_REQUIRED_OUTPUTS, nbInputs, nbOutputs);
        return -1;
    }

    bool const matchingDescriptors = isKVCacheDescriptor(in[kIN_KV_CACHE_IDX].desc, mEnableFp8KVCache)
        && isKVCacheDescriptor(out[kOUT_KV_CACHE_IDX].desc, mEnableFp8KVCache)
        && isKVPageTableDescriptor(in[kIN_KV_PAGE_TABLE_IDX].desc);
    bool const validProfiles = matchingDescriptors
        && hasConcretePagedKVContract(in[kIN_QKV_IDX].min, in[kIN_KV_CACHE_IDX].min, out[kOUT_KV_CACHE_IDX].min,
            in[kIN_KV_PAGE_TABLE_IDX].min, mNumKVHeads, mHeadSize)
        && hasConcretePagedKVContract(in[kIN_QKV_IDX].opt, in[kIN_KV_CACHE_IDX].opt, out[kOUT_KV_CACHE_IDX].opt,
            in[kIN_KV_PAGE_TABLE_IDX].opt, mNumKVHeads, mHeadSize)
        && hasConcretePagedKVContract(in[kIN_QKV_IDX].max, in[kIN_KV_CACHE_IDX].max, out[kOUT_KV_CACHE_IDX].max,
            in[kIN_KV_PAGE_TABLE_IDX].max, mNumKVHeads, mHeadSize);
    if (!validProfiles)
    {
        LOG_ERROR(
            "AttentionPlugin: KV cache profiles must use pool shape [2, N, %d, %d, %d], preserve the KV output "
            "shape, and use INT32 LINEAR kv_page_table [B, 2, M] with B matching packed QKV and M <= N.",
            rt::kTOKENS_PER_PAGE, mNumKVHeads, mHeadSize);
        return -1;
    }
    return 0;
}

size_t AttentionPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    int32_t const expectedNbInputs = getExpectedNbInputs(mEnableQKNorm, mEnableContextMaskSelector,
        mEnableTreeAttention, mEnableVisionBlockAttention, mSkipSoftmaxScaleFactor);
    if (inputs == nullptr || outputs == nullptr || nbInputs != expectedNbInputs || nbOutputs != kNUM_REQUIRED_OUTPUTS)
    {
        LOG_ERROR(
            "AttentionPlugin::getWorkspaceSize: expected %d inputs and %d outputs, but got %d inputs and %d "
            "outputs.",
            expectedNbInputs, kNUM_REQUIRED_OUTPUTS, nbInputs, nbOutputs);
        return 0;
    }

    // Packed QKV: max batch/seq derived from packed input's first two dims (same as Q).
    int64_t const maxBatchSize = inputs[kIN_QKV_IDX].max.d[0];
    int64_t const maxSeqLen = inputs[kIN_QKV_IDX].max.d[1];
    // KV binding is the paged pool [2, numPages, 128, Hkv, D]; the per-slot padded capacity is the
    // page-table width times the page size (kv_page_table is [batch, 2, maxPagesPerSeq]).
    int64_t const maxKVCacheCapacity = inputs[kIN_KV_PAGE_TABLE_IDX].max.d[2] * rt::kTOKENS_PER_PAGE;
    size_t const workspaceSize = getAttentionWorkspaceSize(maxBatchSize, maxSeqLen, maxKVCacheCapacity, mNumQHeads,
        mNumKVHeads, mHeadSize, mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL, mEnableFp8KVCache,
        mEnableVisionBlockAttention != 0);

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

int32_t AttentionPlugin::enqueueImpl(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    check::check(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr,
        "AttentionPlugin received null enqueue descriptors or bindings.");
    check::check(hasConcretePagedKVContract(inputDesc, outputDesc, mNumKVHeads, mHeadSize, mEnableFp8KVCache),
        "AttentionPlugin requires kv_cache pool [2, N, kTOKENS_PER_PAGE, numKVHeads, headDim], identical KV input "
        "and output shapes, and INT32 LINEAR kv_page_table [B, 2, M] with B matching packed QKV and M <= N.");
    check::check(inputs[kIN_KV_CACHE_IDX] != nullptr && inputs[kIN_KV_PAGE_TABLE_IDX] != nullptr
            && outputs[kOUT_KV_CACHE_IDX] != nullptr,
        "AttentionPlugin requires non-null KV pool and page-table bindings.");

    // Packed QKV input, layout selected by the enable_kv_shared field:
    //   0: [B, S, (Hq+2*Hkv)*D] — Q+K+V; K/V are written to the KV cache.
    //   1: [B, S, Hq*D] — Q only; K/V come from a donated cache and are not written.
    PluginTensorDesc const& packedQKVInputDesc = inputDesc[kIN_QKV_IDX];
    int32_t const runtimeBatchSize = static_cast<int32_t>(packedQKVInputDesc.dims.d[0]);
    int32_t const runtimeSeqLen = static_cast<int32_t>(packedQKVInputDesc.dims.d[1]);
    int32_t const actualChannels = static_cast<int32_t>(packedQKVInputDesc.dims.d[2]);
    bool const sharedKV = (mEnableKVShared != 0);
    int32_t const expectedChannels = (sharedKV ? mNumQHeads : (mNumQHeads + 2 * mNumKVHeads)) * mHeadSize;
    check::check(actualChannels == expectedChannels,
        "Packed QKV input last dim does not match enable_kv_shared: expected (Hq + 2*Hkv)*head_dim "
        "(enable_kv_shared=0) or Hq*head_dim (enable_kv_shared=1).");
    int32_t const combinedHeads = sharedKV ? mNumQHeads : (mNumQHeads + 2 * mNumKVHeads);

    rt::Tensor packedQKVTensor(const_cast<void*>(inputs[kIN_QKV_IDX]),
        rt::Coords{runtimeBatchSize, runtimeSeqLen, combinedHeads, mHeadSize}, rt::DeviceType::kGPU,
        packedQKVInputDesc.type);

    // qInputTensor / kInputTensor / vInputTensor are assigned from workspace below and
    // populated by launchApplyRopeFromPackedToSplit.
    rt::Tensor qInputTensor;
    rt::Tensor kInputTensor;
    rt::Tensor vInputTensor;

    // Shared-KV helper: the packed input is already [B, S, Hq, D] (Q only) — alias it as
    // qInputTensor so the Q-only RoPE kernels run in-place.
    auto aliasPackedAsQInput = [&]() {
        return rt::Tensor(const_cast<void*>(inputs[kIN_QKV_IDX]),
            rt::Coords{runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, rt::DeviceType::kGPU,
            packedQKVInputDesc.type);
    };

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
    // in-place aliases it (output 1 aliases input 1). Shared-KV draft layers are read-only views
    // of the target/base cache, so they must read the input binding rather than the plugin's
    // present-KV output; the present-KV output is not consumed in shared-KV mode and must remain
    // unwritten by every shared-KV path below.
    PluginTensorDesc const& kvCacheInputDesc = inputDesc[kIN_KV_CACHE_IDX];
    rt::Tensor pastKVCacheTensor(const_cast<void*>(inputs[kIN_KV_CACHE_IDX]), rt::Coords{kvCacheInputDesc.dims},
        rt::DeviceType::kGPU, kvCacheInputDesc.type);
    rt::Tensor presentKVCacheTensor(
        outputs[kOUT_KV_CACHE_IDX], rt::Coords{kvCacheInputDesc.dims}, rt::DeviceType::kGPU, kvCacheInputDesc.type);
    rt::Tensor& kvCacheTensor = sharedKV ? pastKVCacheTensor : presentKVCacheTensor;
    // numPages feeds native paged CuTe DSL prefill; writes and XQA decode derive
    // capacity from the page table.
    int32_t const numPages = static_cast<int32_t>(kvCacheInputDesc.dims.d[1]);

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

#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
    // Skip-softmax (BLASST): resolve the effective scale factor S (engine-carried
    // calibrated default, overridden by the optional skip_softmax_scale input's
    // SHAPE when present).
    float skipSoftmaxScaleFactor = mSkipSoftmaxScaleFactor;
    if (mSkipSoftmaxScaleFactor > 0.F)
    {
        int32_t const skipScaleIdx = skipSoftmaxScaleInputIdx(mEnableQKNorm != 0, mEnableContextMaskSelector != 0,
            mEnableTreeAttention != 0, mEnableVisionBlockAttention != 0);
        int64_t const overrideS = inputDesc[skipScaleIdx].dims.d[0];
        if (overrideS > 0)
        {
            skipSoftmaxScaleFactor = static_cast<float>(overrideS);
        }
    }
    float const skipSoftmaxThresholdLog2
        = computeSkipSoftmaxThreshold(skipSoftmaxScaleFactor, mSlidingWindowSize, kvCacheCapacity);
#endif // CUTE_DSL_FMHA_BLACKWELL_ENABLED

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
        int32_t const maskIdx = attnMaskInputIdx(enableQKNorm, enableContextMaskSelector);
        int32_t const posIdIdx = attnPosIdInputIdx(enableQKNorm, enableContextMaskSelector);
        PluginTensorDesc const& attentionMaskInputDesc = inputDesc[maskIdx];
        PluginTensorDesc const& attentionPosIdInputDesc = inputDesc[posIdIdx];
        attentionMaskTensor = rt::Tensor(const_cast<void*>(inputs[maskIdx]), rt::Coords{attentionMaskInputDesc.dims},
            rt::DeviceType::kGPU, attentionMaskInputDesc.type);
        attentionPosIdTensor = rt::Tensor(const_cast<void*>(inputs[posIdIdx]), rt::Coords{attentionPosIdInputDesc.dims},
            rt::DeviceType::kGPU, attentionPosIdInputDesc.type);
    }
    else if (mEnableVisionBlockAttention)
    {
        // The vision-block-ID tensor rides the attention_mask slot; its position
        // depends on whether qk_norm gamma and context selector inputs precede it.
        int32_t const maskIdx = attnMaskInputIdx(enableQKNorm, enableContextMaskSelector);
        PluginTensorDesc const& visionBlockIdsDesc = inputDesc[maskIdx];
        visionBlockIdsTensor = rt::Tensor(const_cast<void*>(inputs[maskIdx]), rt::Coords{visionBlockIdsDesc.dims},
            rt::DeviceType::kGPU, visionBlockIdsDesc.type);
    }
    bool const useExplicitPositionIds = mEnableTreeAttention && !attentionPosIdTensor.isEmpty()
        && attentionPosIdTensor.getShape().getNumDims() == 2 && attentionPosIdTensor.getShape()[1] == runtimeSeqLen;
    float const kScale = mQkvScales[1];
    float const vScale = mQkvScales[2];

    // Determine the attention execution mode based on the input tensors.
    // deduceMode* only reads seq_len (.getShape()[1]), which equals Q's seq_len.
    AttentionExecutionMode executionMode{};
    if (!mEnableTreeAttention)
    {
        if (isPaddingContextMask(runtimeContextMaskMode) && kvCacheStartIdxTensor.getShape()[0] != 0)
        {
            executionMode = AttentionExecutionMode::kCHUNKED_PREFILL;
        }
        else
        {
            executionMode = deduceModeVanilla(packedQKVTensor, kvCacheStartIdxTensor);
        }
    }
    else
    {
        executionMode = deduceModeTreeAttention(packedQKVTensor, kvCacheStartIdxTensor, attentionPosIdTensor);
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

    // Gamma engine-weight inputs are device-resident at engine load. Models without
    // qk_norm do not wire them ⇒ nullptr ⇒ the kernel takes the RoPE-only path.
    half const* qNormGammaDevicePtr
        = mEnableQKNorm ? resolveNormGammaInput(inputDesc, inputs, qNormGammaInputIdx()) : nullptr;
    half const* kNormGammaDevicePtr
        = mEnableQKNorm ? resolveNormGammaInput(inputDesc, inputs, kNormGammaInputIdx()) : nullptr;
    float const rmsNormEpsVal = mRmsNormEps;

    // ==================== Prefill path ====================
    // Dispatch order: vision-block attention first (paged CuTe DSL or
    // FMHA-v2 CuTe DSL), then sharedKV, then own-KV.
    // Within each: FMHA, or reject the prefill when no kernel serves the head size.
    if (executionMode == AttentionExecutionMode::kNORMAL_PREFILL
        || executionMode == AttentionExecutionMode::kCHUNKED_PREFILL)
    {
        bool const usePaddingContextMask = isPaddingContextMask(runtimeContextMaskMode);
        // Shared layers do not own the donor cache's K/V quantization scales, so they cannot safely dequantize an
        // FP8 donor cache during prefill.
        if (mEnableFp8KVCache && sharedKV)
        {
            LOG_ERROR("AttentionPlugin: shared-KV prefill cannot read an FP8 donor cache.");
            return -1;
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

            bool visionKernelPreflighted{false};
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
            if (canUseCuteDslBidirectionalForPrefill())
            {
                int32_t const slidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize - 1 : INT_MAX;
                CuteDslFMHARunner runner(
                    mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);
                if (!runner.preflightPaged(stream, slidingWindow, /*fp8Input=*/false, /*isCausal=*/true,
                        /*skipSoftmaxThresholdLog2=*/0.0F, /*useBidirectional=*/true))
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

            // Unpack packed QKV: roped Q to qScratch, roped K + V to the paged pool.
            qInputTensor = assignTensorFromWorkspace(
                alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
            kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor, rt::OptionalInputTensor{},
                rt::OptionalInputTensor{}, packedQKVTensor, qInputTensor, kvCacheTensor, kScale, vScale, stream,
                pageTable, maxPagesPerSeq, nullptr /* kScratchOut */, nullptr /* vScratchOut */, nullptr /* fp8QOut */,
                1.0f /* qScale */, qNormGammaDevicePtr, kNormGammaDevicePtr, rmsNormEpsVal);

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
                        /*isCausal=*/true, /*skipSoftmaxThresholdLog2=*/0.0F, blockBeginTensor.dataPointer<int32_t>(),
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
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
        if (mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL
            && (!usePaddingContextMask || mCanImplementPaddingFMHA))
        {
            int32_t const slidingWindow
                = (mSlidingWindowSize > 0 && !usePaddingContextMask) ? mSlidingWindowSize - 1 : INT_MAX;
            bool const fp8Input = !sharedKV && mEnableFp8KVCache;
            float const preflightSkipSoftmaxThresholdLog2 = fp8Input ? 0.0F : skipSoftmaxThresholdLog2;
            CuteDslFMHARunner runner(
                mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);
            if (!runner.preflightPaged(stream, slidingWindow, fp8Input, !usePaddingContextMask,
                    preflightSkipSoftmaxThresholdLog2, /*useBidirectional=*/false))
            {
                return -1;
            }
            fmhaKernelPreflighted = true;
        }
#endif
        if (!fmhaKernelPreflighted && mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_V2
            && (!usePaddingContextMask || mCanImplementPaddingFMHA))
        {
            bool const useSmallD64 = executionMode == AttentionExecutionMode::kNORMAL_PREFILL;
            bool const useNativePagedFMHA = !usePaddingContextMask && !mEnableFp8KVCache;
            int32_t const preflightKVSeqLen
                = useNativePagedFMHA ? kvCacheCapacity : (useSmallD64 ? runtimeSeqLen : kvCacheCapacity);
            CuteDslFMHAV2Runner runner(
                mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, preflightKVSeqLen, useSmallD64);
            int32_t const slidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize - 1 : INT_MAX;
            bool const preflightSucceeded = usePaddingContextMask ? runner.preflightPadding(stream)
                : useNativePagedFMHA                              ? runner.preflightPaged(stream, slidingWindow)
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
            // It is only safe to compact the split-KV workspace to runtimeSeqLen
            // when the prefill starts from an empty KV cache. Chunked prefill,
            // including DiffusionGemma's runtime-selected padding mask, must
            // keep the existing prefix KV visible: its logical K length is
            // kvcache_start_index + runtimeSeqLen, not just runtimeSeqLen.
            bool const compact = executionMode == AttentionExecutionMode::kNORMAL_PREFILL;
            return compact ? runtimeSeqLen : 0;
        };
        [[maybe_unused]] auto physicalLenForFallback
            = [&]() { return splitLenForFallback() > 0 ? runtimeSeqLen : kvCacheCapacity; };

        // --- Shared KV prefill: Q gets RoPE, K/V read from donor layer's cache ---
        if (sharedKV)
        {
            // Shared-KV: RoPE Q in-place only; the donor layer's cache is already populated.
            qInputTensor = aliasPackedAsQInput();
            if (useExplicitPositionIds)
            {
                kernel::launchApplyRopeQOnlyTreeDecoding(ropeCosSinTensor, attentionPosIdTensor, qInputTensor, stream);
            }
            else
            {
                kernel::launchApplyRopeQOnly(ropeCosSinTensor, kvCacheEndIdxsTensor, qInputTensor, stream);
            }

            // Run FMHA reading from the donor's KV cache (bound to this layer's KV cache input).
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
            if (mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_BLACKWELL
                && (!usePaddingContextMask || mCanImplementPaddingFMHA))
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
                        kScale, vScale, !usePaddingContextMask, skipSoftmaxThresholdLog2))
                {
                    return -1;
                }
            }
            else
#endif
                if (mContextFMHABackend == ContextFMHABackend::kCUTE_DSL_FMHA_V2
                    && (!usePaddingContextMask || mCanImplementPaddingFMHA))
            {
                if (!usePaddingContextMask)
                {
                    int32_t const slidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize - 1 : INT_MAX;
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
                            rt::kTOKENS_PER_PAGE, stream, mAttentionScale, slidingWindow))
                    {
                        return -1;
                    }
                }
                else
                {
                    // The special padding kernel retains its independent split-K/V contract.
                    int32_t const splitSeqLen = splitLenForFallback();
                    auto [kSplit, vSplit] = splitPagedKV(kvCacheTensor, pageTable,
                        kvCacheEndIdxsTensor.dataPointer<int32_t>(), maxPagesPerSeq, alignedWorkspacePtr,
                        runtimeBatchSize, mNumKVHeads, kvCacheCapacity, mHeadSize, splitSeqLen, kScale, vScale, stream);
                    int32_t const kvExtent = physicalLenForFallback();
                    CuteDslFMHAV2Runner runner(
                        mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvExtent);
                    if (!runner.runPadding(qInputTensor.dataPointer<half>(), kSplit.dataPointer<half>(),
                            vSplit.dataPointer<half>(), attentionOutputTensor.dataPointer<half>(),
                            cuQSeqLensTensor.dataPointer<int32_t>(), cuKVSeqLensTensor.dataPointer<int32_t>(), stream,
                            mAttentionScale))
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
                    qNormGammaDevicePtr, kNormGammaDevicePtr, rmsNormEpsVal);

                if (!runner.runPaged(fp8QTensor.rawPointer(),      // Q  [b, s_q, h_q, d] FP8
                        kvCacheTensor.rawPointer(),                // paged KV pool [2, numPages, 128, h_k, d] FP8
                        pageTable,                                 // page table [b, 2, maxPagesPerSeq]
                        attentionOutputTensor.dataPointer<half>(), // O  [b, s_q, h_q, d] FP16
                        fmhaCuKVSeqLens, 2 * numPages, maxPagesPerSeq, rt::kTOKENS_PER_PAGE,
                        kvCacheTensor.getDataType(), stream, mAttentionScale, slidingWindow, /*fp8Input=*/true, qScale,
                        kScale, vScale, !usePaddingContextMask))
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
                    qNormGammaDevicePtr, kNormGammaDevicePtr, rmsNormEpsVal);

                if (!runner.runPaged(qInputTensor.dataPointer<half>(), // Q  [b, s_q, h_q, d]
                        kvCacheTensor.rawPointer(),                    // paged KV pool [2, numPages, 128, h_k, d]
                        pageTable,                                     // page table [b, 2, maxPagesPerSeq]
                        attentionOutputTensor.dataPointer<half>(),     // O  [b, s_q, h_q, d]
                        fmhaCuKVSeqLens, 2 * numPages, maxPagesPerSeq, rt::kTOKENS_PER_PAGE,
                        kvCacheTensor.getDataType(), stream, mAttentionScale, slidingWindow, /*fp8Input=*/false, 1.0F,
                        1.0F, 1.0F, !usePaddingContextMask, skipSoftmaxThresholdLog2))
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

            if (!mEnableFp8KVCache && !usePaddingContextMask)
            {
                // FP16 causal/sliding prefill writes K/V to the cache and reads that paged pool directly.
                kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor,
                    rt::OptionalInputTensor{kvCacheEndIdxsTensor}, rt::OptionalInputTensor{}, packedQKVTensor,
                    qInputTensor, kvCacheTensor, kScale, vScale, stream, pageTable, maxPagesPerSeq,
                    nullptr /* kScratch */, nullptr /* vScratch */, nullptr /* fp8QOut */, 1.0f /* qScale */,
                    qNormGammaDevicePtr, kNormGammaDevicePtr, rmsNormEpsVal, rt::OptionalInputTensor{cuQSeqLensTensor});

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
                        stream, mAttentionScale, slidingWindow))
                {
                    return -1;
                }
            }
            else
            {
                // Preserve the legacy dense path for FP8 KV and the independent padding contract.
                bool const gatherKV
                    = executionMode == AttentionExecutionMode::kCHUNKED_PREFILL || usePaddingContextMask;
                if (gatherKV)
                {
                    kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor,
                        rt::OptionalInputTensor{kvCacheEndIdxsTensor}, rt::OptionalInputTensor{}, packedQKVTensor,
                        qInputTensor, kvCacheTensor, kScale, vScale, stream, pageTable, maxPagesPerSeq,
                        nullptr /* kScratch */, nullptr /* vScratch */, nullptr /* fp8QOut */, 1.0f /* qScale */,
                        qNormGammaDevicePtr, kNormGammaDevicePtr, rmsNormEpsVal);

                    int32_t const splitSeqLen = splitLenForFallback();
                    auto [kSplit, vSplit] = splitPagedKV(kvCacheTensor, pageTable,
                        kvCacheEndIdxsTensor.dataPointer<int32_t>(), maxPagesPerSeq, alignedWorkspacePtr,
                        runtimeBatchSize, mNumKVHeads, kvCacheCapacity, mHeadSize, splitSeqLen, kScale, vScale, stream);
                    CuteDslFMHAV2Runner runner(mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen,
                        physicalLenForFallback(), splitSeqLen > 0);
                    bool const ranKernel = usePaddingContextMask
                        ? runner.runPadding(qInputTensor.dataPointer<half>(), kSplit.dataPointer<half>(),
                              vSplit.dataPointer<half>(), attentionOutputTensor.dataPointer<half>(),
                              cuQSeqLensTensor.dataPointer<int32_t>(), cuKVSeqLensTensor.dataPointer<int32_t>(), stream,
                              mAttentionScale)
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
                    // Normal FP8-KV prefill keeps the existing dense FP16 scratch contract.
                    kInputTensor = assignTensorFromWorkspace(alignedWorkspacePtr,
                        {runtimeBatchSize, runtimeSeqLen, mNumKVHeads, mHeadSize}, DataType::kHALF);
                    vInputTensor = assignTensorFromWorkspace(alignedWorkspacePtr,
                        {runtimeBatchSize, runtimeSeqLen, mNumKVHeads, mHeadSize}, DataType::kHALF);
                    kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor, rt::OptionalInputTensor{},
                        rt::OptionalInputTensor{}, packedQKVTensor, qInputTensor, kvCacheTensor, kScale, vScale, stream,
                        pageTable, maxPagesPerSeq, kInputTensor.rawPointer(), vInputTensor.rawPointer(),
                        nullptr /* fp8QOut */, 1.0f /* qScale */, qNormGammaDevicePtr, kNormGammaDevicePtr,
                        rmsNormEpsVal);

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
        // RoPE setup: sharedKV → Q only; own-KV → packed kernel (Q to qScratch + KV cache
        // write). Decode reads K/V from the KV cache via XQA — no scratch K/V needed.
        if (sharedKV)
        {
            // Shared-KV: packed input is [B,S,Hq,D] (Q only) — alias as qInputTensor
            // and RoPE in-place.
            qInputTensor = aliasPackedAsQInput();
            if (executionMode == AttentionExecutionMode::kTREE_DECODING || useExplicitPositionIds)
            {
                kernel::launchApplyRopeQOnlyTreeDecoding(ropeCosSinTensor, attentionPosIdTensor, qInputTensor, stream);
            }
            else
            {
                kernel::launchApplyRopeQOnly(ropeCosSinTensor, contextLengthTensor, qInputTensor, stream);
            }
            // The Q-only path has no fused-norm kernel; enable_qk_norm + enable_kv_shared is
            // rejected at plugin construction.
        }
        else if (executionMode == AttentionExecutionMode::kTREE_DECODING)
        {
            // Tree decoding: pass tokenPosIds for per-token RoPE positions.
            // Allocate qScratch (roped Q output written by launchApplyRopeFromPackedToSplit).
            qInputTensor = assignTensorFromWorkspace(
                alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
            kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor, rt::OptionalInputTensor{contextLengthTensor},
                rt::OptionalInputTensor{attentionPosIdTensor}, packedQKVTensor, qInputTensor, kvCacheTensor, kScale,
                vScale, stream, pageTable, maxPagesPerSeq, nullptr /* kScratch */, nullptr /* vScratch */,
                nullptr /* fp8QOut */, 1.0f /* qScale */, qNormGammaDevicePtr, kNormGammaDevicePtr, rmsNormEpsVal);
        }
        else
        {
            // Vanilla decoding: derive RoPE position from contextLengthTensor (kvCacheEndLens).
            // Allocate qScratch (roped Q output written by launchApplyRopeFromPackedToSplit).
            qInputTensor = assignTensorFromWorkspace(
                alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
            kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor, rt::OptionalInputTensor{contextLengthTensor},
                rt::OptionalInputTensor{}, packedQKVTensor, qInputTensor, kvCacheTensor, kScale, vScale, stream,
                pageTable, maxPagesPerSeq, nullptr /* kScratch */, nullptr /* vScratch */, nullptr /* fp8QOut */,
                1.0f /* qScale */, qNormGammaDevicePtr, kNormGammaDevicePtr, rmsNormEpsVal);
        }

        // Vision-block decode goes to XQA (a hard construction-time
        // requirement): newly generated tokens are text, so decode is pure
        // causal/sliding — identical masking to the non-vision path.
        // Cache-state parity: the vision prefill paths write the cache with
        // launchApplyRopeWriteKV (roped K + original V into the paged pool)
        // and the decode RoPE above appended the
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

int32_t AttentionPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    int32_t const expectedNbInputs = getExpectedNbInputs(mEnableQKNorm, mEnableContextMaskSelector,
        mEnableTreeAttention, mEnableVisionBlockAttention, mSkipSoftmaxScaleFactor);
    if (in == nullptr || out == nullptr || nbInputs != expectedNbInputs || nbOutputs != kNUM_REQUIRED_OUTPUTS)
    {
        LOG_ERROR("AttentionPlugin: expected %d inputs and %d outputs, but got %d inputs and %d outputs.",
            expectedNbInputs, kNUM_REQUIRED_OUTPUTS, nbInputs, nbOutputs);
        return -1;
    }
    if (!hasConcretePagedKVContract(in, out, mNumKVHeads, mHeadSize, mEnableFp8KVCache))
    {
        LOG_ERROR(
            "AttentionPlugin: kv_cache must use pool shape [2, N, %d, %d, %d], preserve its output shape, and "
            "use INT32 LINEAR kv_page_table [B, 2, M] with B matching packed QKV and M <= N.",
            rt::kTOKENS_PER_PAGE, mNumKVHeads, mHeadSize);
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
    mDataToSerialize.emplace_back("enable_kv_shared", &mEnableKVShared, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("enable_fp8_kv_cache", &mEnableFp8KVCache, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(
        "enable_vision_block_attention", &mEnableVisionBlockAttention, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(
        "enable_context_mask_selector", &mEnableContextMaskSelector, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("sliding_window_size", &mSlidingWindowSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("skip_softmax_scale_factor", &mSkipSoftmaxScaleFactor, PluginFieldType::kFLOAT32, 1);
    mDataToSerialize.emplace_back(
        "qkv_scales", mQkvScales.data(), PluginFieldType::kFLOAT32, static_cast<int32_t>(mQkvScales.size()));
    // Serialize the RMSNorm eps for the fused qk_norm path. The gamma WEIGHTS live as
    // engine-weight constant inputs (baked at build time), not plugin fields.
    mDataToSerialize.emplace_back("rms_norm_eps", &mRmsNormEps, PluginFieldType::kFLOAT32, 1);
    mDataToSerialize.emplace_back(kXQA_JIT_KERNELS_FIELD, mXqaJitBlob.empty() ? nullptr : mXqaJitBlob.data(),
        PluginFieldType::kCHAR, static_cast<int32_t>(mXqaJitBlob.size()));
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
    // Optional (default 0). Shared-KV layer: packed input is Q only; no KV-cache write.
    mPluginAttributes.emplace_back(PluginField("enable_kv_shared", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("enable_fp8_kv_cache", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("enable_vision_block_attention", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("enable_context_mask_selector", nullptr, PluginFieldType::kINT32, 0));
    // Sliding window size (-1 = no sliding window, >0 = window size)
    mPluginAttributes.emplace_back(PluginField("sliding_window_size", nullptr, PluginFieldType::kINT32, 0));
    // Skip-softmax (BLASST) calibrated scale factor S (0 = disabled, the default)
    mPluginAttributes.emplace_back(PluginField("skip_softmax_scale_factor", nullptr, PluginFieldType::kFLOAT32, 0));
    // Optional QKV dequant scales [q, k, v] for FP8 attention
    mPluginAttributes.emplace_back(PluginField("qkv_scales", nullptr, PluginFieldType::kFLOAT32, 0));
    // Optional per-head RMSNorm gamma weights (length == head_size) and eps. Empty / missing ->
    // qk_norm disabled. When supplied, the plugin fuses RMSNorm into the RoPE+KVWrite kernel.
    mPluginAttributes.emplace_back(PluginField("rms_norm_eps", nullptr, PluginFieldType::kFLOAT32, 1));
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
        auto plugin = std::make_unique<AttentionPlugin>(std::string(name), fc);
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
