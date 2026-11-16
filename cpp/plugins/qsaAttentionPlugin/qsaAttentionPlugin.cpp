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

#include "qsaAttentionPlugin.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "common/trtInterop.h"
#include "kernels/contextAttentionKernels/attentionScaleUtils.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/qsaAttention/cuteDslQsaSparseRunner.h"
#include "kernels/qsaIndexer/qsaIndexerKernels.h"
#include "kernels/qsaIndexer/qsaIndexerRunner.h"
#include "plugins/utils/pluginUtils.h"
#include "plugins/utils/raggedPluginMetadata.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{
using nvinfer1::DataType;

namespace
{
constexpr char const* kQSA_ATTENTION_PLUGIN_VERSION{"1"};
constexpr char const* kQSA_ATTENTION_PLUGIN_NAME{"QsaAttentionPlugin"};

// Input / output index mapping. The 15 base inputs are required, so the four ragged-ABI
// carriers always sit at 11-14; use_spec_verify_state appends three inputs.
// Activations are token-major and entry-padded: T_exec = N * S_pad, N = query_lengths extent.
constexpr int32_t kIN_QKV_IDX{0};                         //!< [T_exec, (Hq + 2*Hkv) * D] FP16 packed QKV
constexpr int32_t kIN_INDEX_QK_IDX{1};                    //!< [T_exec, (idxHeads + 1) * idxDim] FP16 indexer projection
constexpr int32_t kIN_KV_CACHE_IDX{2};                    //!< [2, numPages, 128, Hkv, D + idxDim] FP16 widened pool
constexpr int32_t kIN_QUERY_LENGTHS_IDX{3};               //!< [N] INT32 valid query rows per entry (prefill lengths)
constexpr int32_t kIN_ROPE_COS_SIN_IDX{4};                //!< [T_exec, 64] FP32 token-aligned rope rows (main, indexer)
constexpr int32_t kIN_PAST_LENGTHS_IDX{5};                //!< [N] INT32 cached prefix lengths; read by context chunk
constexpr int32_t kIN_KV_PAGE_TABLE_IDX{6};               //!< [N, 2, maxPagesPerSeq] INT32
constexpr int32_t kIN_Q_NORM_GAMMA_IDX{7};                //!< [D] FP16 Constant, pre-folded (1 + w)
constexpr int32_t kIN_K_NORM_GAMMA_IDX{8};                //!< [D] FP16 Constant, pre-folded (1 + w)
constexpr int32_t kIN_INDEXER_Q_NORM_GAMMA_IDX{9};        //!< [idxDim] FP16 Constant, RAW w (kernel adds 1)
constexpr int32_t kIN_INDEXER_K_NORM_GAMMA_IDX{10};       //!< [idxDim] FP16 Constant, RAW w
constexpr int32_t kIN_QUERY_START_OFFSETS_IDX{11};        //!< [N + 1] INT32 physical row starts (descriptor only)
constexpr int32_t kIN_ATTENTION_SEQUENCE_LENGTHS_IDX{12}; //!< [N] INT32 TOTAL (past + query) lengths; read by decode
constexpr int32_t kIN_EXECUTION_PHASE_MARKER_IDX{13};     //!< [phase] INT32 shape-only carrier (rt::ExecutionPhase)
constexpr int32_t kIN_CONTEXT_SEQUENCE_COUNT_IDX{14};     //!< [N_ctx] INT32 shape-only carrier
constexpr int32_t kIN_TREE_PARENT_IDS_IDX{15};            //!< [T_exec] INT32 (use_spec_verify_state)
constexpr int32_t kIN_TREE_DEPTHS_IDX{16};                //!< [T_exec] INT32 (use_spec_verify_state)
constexpr int32_t kIN_SPEC_VERIFY_LAYOUT_IDX{17};         //!< [layout] INT32 shape-only carrier (rt::SpecVerifyLayout)

constexpr int32_t kOUT_ATTENTION_IDX{0}; //!< [T_exec, Hq, D] FP16
constexpr int32_t kOUT_KV_CACHE_IDX{1};  //!< paged pool, in-place aliased with input 2

constexpr int32_t kNUM_REQUIRED_INPUTS{15};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{2};
constexpr int32_t kNUM_SPEC_VERIFY_OPTIONAL_INPUTS{3};

//! The shared token-aligned rope rows serve the main partial rope (64 of 256) and the indexer
//! rope (64 of 128); their width is therefore pinned to the indexer rotary dim.
constexpr int32_t kQSA_ROPE_TABLE_DIM{kernel::kQSA_INDEXER_ROTARY_DIM};

//! The QSA pool head dimension is DERIVED as head_size + indexer_head_dim (no extra plugin
//! attribute): each pool row is [roped K or raw V | indexer-state tail]. The tail layout is
//! owned by the indexer (see the class doc and qsaIndexerKernels.h).
bool isQsaPagedPoolShape(Dims const& shape, int32_t numKVHeads, int32_t headSize, int32_t indexerHeadDim)
{
    if (shape.nbDims != 5)
    {
        return false;
    }
    bool const validNumPages = shape.d[1] == -1 || shape.d[1] > 0;
    return shape.d[0] == 2 && validNumPages && shape.d[2] == rt::kTOKENS_PER_PAGE && shape.d[3] == numKVHeads
        && shape.d[4] == headSize + indexerHeadDim;
}

// Context workspace layout (carve order in enqueueImpl must match): cuQSeqLens, cuKVSeqLens,
// kvCacheEndIdxs, qScratch [B, S, Hq, D], outIdx [B, S, kQSA_INDEX_WIDTH], indexer scratch.
size_t getQsaContextWorkspaceSize(
    int64_t batchSize, int64_t seqLen, int32_t numQHeads, int32_t headSize, size_t indexerWorkspaceBytes)
{
    size_t workspaceSize = 0;
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize + 1}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize + 1}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize, seqLen, numQHeads, headSize}, DataType::kHALF);
    workspaceSize
        = accumulateWorkspaceSize(workspaceSize, {batchSize, seqLen, kernel::kQSA_INDEX_WIDTH}, DataType::kINT32);
    workspaceSize
        = accumulateWorkspaceSize(workspaceSize, {static_cast<int64_t>(indexerWorkspaceBytes)}, DataType::kINT8);
    return workspaceSize;
}

size_t getQsaPrefillWorkspaceSize(
    int64_t batchSize, int64_t seqLen, int32_t maxBlocks, int32_t numQHeads, int32_t headSize)
{
    return getQsaContextWorkspaceSize(batchSize, seqLen, numQHeads, headSize,
        kernel::getQsaIndexerWorkspaceSize(
            static_cast<int32_t>(batchSize), static_cast<int32_t>(seqLen), maxBlocks, true));
}

size_t getQsaVerifyWorkspaceSize(
    int64_t batchSize, int64_t seqLen, int32_t maxBlocks, int32_t numQHeads, int32_t headSize)
{
    return getQsaContextWorkspaceSize(batchSize, seqLen, numQHeads, headSize,
        kernel::getQsaIndexerVerifyWorkspaceSize(
            static_cast<int32_t>(batchSize), static_cast<int32_t>(seqLen), maxBlocks));
}

//! Compressed-block capacity implied by a page-table width: every page holds
//! kTOKENS_PER_PAGE tokens, and blocks compress kQSA_COMPRESS_RATIO tokens each.
int64_t qsaMaxBlocksForPages(int64_t maxPagesPerSeq)
{
    return maxPagesPerSeq * (rt::kTOKENS_PER_PAGE / kernel::kQSA_COMPRESS_RATIO);
}

// Decode workspace layout (cumulative; each slot 128-byte aligned; carve order in
// enqueueDecode must match):
//
//   Slot  | Shape                             | Type  | Used by
//   ------+-----------------------------------+-------+----------------------------------
//   0     | [B, 1, Hq, D]                     | HALF  | qScratch      (roped Q)
//   1     | [B, 1, kQSA_INDEX_WIDTH]          | INT32 | outIdx        (indexer output)
//   2     | [B*Hkv*kMaxSplits, kPartialRows, D] | FP32 | partialO    (split-K partials)
//   3     | [B*Hkv*kMaxSplits, 2, kPartialRows] | FP32 | partialStats (row max / denom)
//   4     | [B, Hkv]                          | INT32 | splitCounters (zeroed by B3 each step)
//   5     | getQsaIndexerDecodeWorkspaceSize  | BYTE  | indexer scratch (qNormed + logits)
size_t getQsaDecodeWorkspaceSize(
    int64_t batchSize, int64_t maxPagesPerSeq, int32_t numQHeads, int32_t numKVHeads, int32_t headSize)
{
    int64_t const partialSlots = batchSize * numKVHeads * CuteDslQsaSparseDecodeRunner::kMaxSplits;
    size_t workspaceSize = 0;
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize, 1, numQHeads, headSize}, DataType::kHALF);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize, 1, kernel::kQSA_INDEX_WIDTH}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, {partialSlots, CuteDslQsaSparseDecodeRunner::kPartialRows, headSize}, DataType::kFLOAT);
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, {partialSlots, 2, CuteDslQsaSparseDecodeRunner::kPartialRows}, DataType::kFLOAT);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize, numKVHeads}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize,
        {static_cast<int64_t>(kernel::getQsaIndexerDecodeWorkspaceSize(
            static_cast<int32_t>(batchSize), static_cast<int32_t>(qsaMaxBlocksForPages(maxPagesPerSeq))))},
        DataType::kINT8);
    return workspaceSize;
}

} // namespace

// Static class fields initialization
PluginFieldCollection QsaAttentionPluginCreator::mFieldCollection{};
std::vector<PluginField> QsaAttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(QsaAttentionPluginCreator);

QsaAttentionPlugin::QsaAttentionPlugin(std::string const& name, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t indexerNumHeads, int32_t indexerHeadDim, int32_t indexerBudget, int32_t indexerCompressRatio,
    float attentionScale, float rmsNormEps, bool useSpecVerifyState)
    : mLayerName(name)
    , mNumQHeads(numQHeads)
    , mNumKVHeads(numKVHeads)
    , mHeadSize(headSize)
    , mIndexerNumHeads(indexerNumHeads)
    , mIndexerHeadDim(indexerHeadDim)
    , mIndexerBudget(indexerBudget)
    , mIndexerCompressRatio(indexerCompressRatio)
    , mAttentionScale(attentionScale)
    , mRmsNormEps(rmsNormEps)
    , mUseSpecVerifyState(useSpecVerifyState)
{
    validateConfiguration();
}

QsaAttentionPlugin::QsaAttentionPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
    , mNumQHeads(parsePluginScalarField<int32_t>("num_q_heads", fc).value_or(0))
    , mNumKVHeads(parsePluginScalarField<int32_t>("num_kv_heads", fc).value_or(0))
    , mHeadSize(parsePluginScalarField<int32_t>("head_size", fc).value_or(0))
    , mIndexerNumHeads(parsePluginScalarField<int32_t>("indexer_n_heads", fc).value_or(0))
    , mIndexerHeadDim(parsePluginScalarField<int32_t>("indexer_head_dim", fc).value_or(0))
    , mIndexerBudget(parsePluginScalarField<int32_t>("indexer_budget", fc).value_or(0))
    , mIndexerCompressRatio(parsePluginScalarField<int32_t>("indexer_compress_ratio", fc).value_or(0))
    , mRmsNormEps(parsePluginScalarField<float>("rms_norm_eps", fc).value_or(1e-6F))
{
    std::optional<int32_t> const useSpecVerifyState = parsePluginScalarField<int32_t>("use_spec_verify_state", fc);
    ELLM_CHECK(useSpecVerifyState.has_value(),
        "QsaAttentionPlugin: missing use_spec_verify_state; re-export and rebuild artifacts.");
    mUseSpecVerifyState = *useSpecVerifyState != 0;

    // The stub always emits attention_scale; <= 0 selects the 1/sqrt(head_size) default.
    std::optional<float> attentionScale = parsePluginScalarField<float>("attention_scale", fc);
    if (attentionScale.has_value() && *attentionScale <= 0.0F)
    {
        attentionScale.reset();
    }
    mAttentionScale = resolveAttentionScale(attentionScale, mHeadSize);

    validateConfiguration();
}

void QsaAttentionPlugin::validateConfiguration() const
{
    ELLM_CHECK(mNumQHeads > 0 && mNumKVHeads > 0 && mHeadSize > 0,
        "QsaAttentionPlugin requires positive num_q_heads, num_kv_heads and head_size.");
    ELLM_CHECK(mNumQHeads % mNumKVHeads == 0, "QsaAttentionPlugin requires num_q_heads % num_kv_heads == 0.");
    // The indexer kernels are specialized to the Qwen3.8-Flash-Next configuration.
    ELLM_CHECK(mIndexerNumHeads == kernel::kQSA_INDEXER_NUM_HEADS,
        "QsaAttentionPlugin requires indexer_n_heads == " + std::to_string(kernel::kQSA_INDEXER_NUM_HEADS) + ".");
    ELLM_CHECK(mIndexerHeadDim == kernel::kQSA_INDEXER_HEAD_DIM,
        "QsaAttentionPlugin requires indexer_head_dim == " + std::to_string(kernel::kQSA_INDEXER_HEAD_DIM) + ".");
    ELLM_CHECK(mIndexerBudget == kernel::kQSA_INDEX_BUDGET,
        "QsaAttentionPlugin requires indexer_budget == " + std::to_string(kernel::kQSA_INDEX_BUDGET) + ".");
    ELLM_CHECK(mIndexerCompressRatio == kernel::kQSA_COMPRESS_RATIO,
        "QsaAttentionPlugin requires indexer_compress_ratio == " + std::to_string(kernel::kQSA_COMPRESS_RATIO) + ".");
    ELLM_CHECK(std::isfinite(mRmsNormEps) && mRmsNormEps > 0.0F,
        "QsaAttentionPlugin requires a positive finite rms_norm_eps.");
    validateAttentionScale(mAttentionScale);

    int32_t const indexWidth = mIndexerBudget + mIndexerCompressRatio - 1;
    ELLM_CHECK(indexWidth == kernel::kQSA_INDEX_WIDTH,
        "QsaAttentionPlugin index width must equal " + std::to_string(kernel::kQSA_INDEX_WIDTH) + ".");

    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);

    ELLM_CHECK(CuteDslQsaSparsePrefillRunner::canImplement(mNumQHeads, mNumKVHeads, mHeadSize,
                   mHeadSize + mIndexerHeadDim, smVersion, trt_edgellm::DataType::kHALF),
        "QsaAttentionPlugin: no QSA sparse prefill kernel for Hq=" + std::to_string(mNumQHeads)
            + ", Hkv=" + std::to_string(mNumKVHeads) + ", D=" + std::to_string(mHeadSize) + " on SM"
            + std::to_string(smVersion) + " (build with -DENABLE_CUTE_DSL=\"fmha;qsa\").");
    ELLM_CHECK(CuteDslQsaSparseDecodeRunner::canImplement(mNumQHeads, mNumKVHeads, mHeadSize,
                   mHeadSize + mIndexerHeadDim, smVersion, trt_edgellm::DataType::kHALF),
        "QsaAttentionPlugin: no QSA sparse decode kernel for Hq=" + std::to_string(mNumQHeads)
            + ", Hkv=" + std::to_string(mNumKVHeads) + ", D=" + std::to_string(mHeadSize) + " on SM"
            + std::to_string(smVersion) + " (build with -DENABLE_CUTE_DSL=\"fmha;qsa\").");
}

// ---------------------------------------------------------------------------
// IPluginV3
// ---------------------------------------------------------------------------

IPluginCapability* QsaAttentionPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* QsaAttentionPlugin::clone() noexcept
{
    try
    {
        auto* p = new QsaAttentionPlugin(mLayerName, mNumQHeads, mNumKVHeads, mHeadSize, mIndexerNumHeads,
            mIndexerHeadDim, mIndexerBudget, mIndexerCompressRatio, mAttentionScale, mRmsNormEps, mUseSpecVerifyState);
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

char const* QsaAttentionPlugin::getPluginName() const noexcept
{
    return kQSA_ATTENTION_PLUGIN_NAME;
}

char const* QsaAttentionPlugin::getPluginVersion() const noexcept
{
    return kQSA_ATTENTION_PLUGIN_VERSION;
}

char const* QsaAttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void QsaAttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

// ---------------------------------------------------------------------------
// IPluginV3OneBuildV2 — shape / format
// ---------------------------------------------------------------------------

int32_t QsaAttentionPlugin::getNbInputs() const noexcept
{
    return kNUM_REQUIRED_INPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_INPUTS : 0);
}

int32_t QsaAttentionPlugin::getNbOutputs() const noexcept
{
    return kNUM_REQUIRED_OUTPUTS;
}

int32_t QsaAttentionPlugin::getOutputDataTypes(DataType* outputTypes, [[maybe_unused]] int32_t nbOutputs,
    DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    try
    {
        assert(nbInputs == getNbInputs());
        assert(nbOutputs == getNbOutputs());
        outputTypes[kOUT_ATTENTION_IDX] = DataType::kHALF;
        outputTypes[kOUT_KV_CACHE_IDX] = inputTypes[kIN_KV_CACHE_IDX];
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t QsaAttentionPlugin::getOutputShapes(DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs,
    [[maybe_unused]] int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        assert(nbInputs == getNbInputs());
        assert(nbOutputs == getNbOutputs());
        // attn_output: [T_exec, Hq, D] — the token dimension from the packed QKV input.
        outputs[kOUT_ATTENTION_IDX].nbDims = 3;
        outputs[kOUT_ATTENTION_IDX].d[0] = inputs[kIN_QKV_IDX].d[0];
        outputs[kOUT_ATTENTION_IDX].d[1] = exprBuilder.constant(mNumQHeads);
        outputs[kOUT_ATTENTION_IDX].d[2] = exprBuilder.constant(mHeadSize);
        // present_key_value: the pool shape, written in place.
        outputs[kOUT_KV_CACHE_IDX] = inputs[kIN_KV_CACHE_IDX];
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

bool QsaAttentionPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || pos < 0 || pos >= nbInputs + nbOutputs)
    {
        return false;
    }
    if (nbInputs != getNbInputs() || nbOutputs != getNbOutputs())
    {
        LOG_ERROR("QsaAttentionPlugin '%s' expects %d inputs and %d outputs, got %d inputs and %d outputs.",
            mLayerName.c_str(), getNbInputs(), getNbOutputs(), nbInputs, nbOutputs);
        return false;
    }

    PluginTensorDesc const& desc = inOut[pos].desc;
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    auto isHalfVector = [&desc](int32_t expectedLength) {
        return desc.type == DataType::kHALF && desc.dims.nbDims == 1
            && (desc.dims.d[0] == -1 || desc.dims.d[0] == expectedLength);
    };

    if (pos >= nbInputs)
    {
        int32_t const outPos = pos - nbInputs;
        switch (outPos)
        {
        case kOUT_ATTENTION_IDX: return desc.type == DataType::kHALF && desc.dims.nbDims == 3;
        case kOUT_KV_CACHE_IDX:
            return desc.type == inOut[kIN_KV_CACHE_IDX].desc.type
                && isQsaPagedPoolShape(desc.dims, mNumKVHeads, mHeadSize, mIndexerHeadDim);
        default: return false;
        }
    }

    switch (pos)
    {
    case kIN_QKV_IDX:
        return desc.type == DataType::kHALF && desc.dims.nbDims == 2
            && (desc.dims.d[1] == -1 || desc.dims.d[1] == (mNumQHeads + 2 * mNumKVHeads) * mHeadSize);
    case kIN_INDEX_QK_IDX:
        return desc.type == DataType::kHALF && desc.dims.nbDims == 2
            && (desc.dims.d[1] == -1 || desc.dims.d[1] == (mIndexerNumHeads + 1) * mIndexerHeadDim);
    case kIN_KV_CACHE_IDX:
        return desc.type == DataType::kHALF && isQsaPagedPoolShape(desc.dims, mNumKVHeads, mHeadSize, mIndexerHeadDim);
    case kIN_ROPE_COS_SIN_IDX:
        return desc.type == DataType::kFLOAT && desc.dims.nbDims == 2
            && (desc.dims.d[1] == -1 || desc.dims.d[1] == kQSA_ROPE_TABLE_DIM);
    case kIN_KV_PAGE_TABLE_IDX:
        return desc.type == DataType::kINT32 && desc.dims.nbDims == 3 && (desc.dims.d[1] == -1 || desc.dims.d[1] == 2);
    case kIN_Q_NORM_GAMMA_IDX: return isHalfVector(mHeadSize);
    case kIN_K_NORM_GAMMA_IDX: return isHalfVector(mHeadSize);
    case kIN_INDEXER_Q_NORM_GAMMA_IDX: return isHalfVector(mIndexerHeadDim);
    case kIN_INDEXER_K_NORM_GAMMA_IDX: return isHalfVector(mIndexerHeadDim);
    // Per-entry lengths and the ragged-ABI carriers: rank-1 INT32; every shape relation
    // (N + 1 offsets, phase extent, N_ctx) is validated by decodeRaggedPluginMetadata at enqueue.
    case kIN_QUERY_LENGTHS_IDX:
    case kIN_PAST_LENGTHS_IDX:
    case kIN_QUERY_START_OFFSETS_IDX:
    case kIN_ATTENTION_SEQUENCE_LENGTHS_IDX:
    case kIN_EXECUTION_PHASE_MARKER_IDX:
    case kIN_CONTEXT_SEQUENCE_COUNT_IDX:
    case kIN_TREE_PARENT_IDS_IDX:
    case kIN_TREE_DEPTHS_IDX:
    case kIN_SPEC_VERIFY_LAYOUT_IDX: return desc.type == DataType::kINT32 && desc.dims.nbDims == 1;
    default: return false;
    }
}

int32_t QsaAttentionPlugin::configurePlugin(DynamicPluginTensorDesc const* in, [[maybe_unused]] int32_t nbInputs,
    DynamicPluginTensorDesc const* out, [[maybe_unused]] int32_t nbOutputs) noexcept
{
    try
    {
        assert(nbInputs == getNbInputs());
        assert(nbOutputs == getNbOutputs());
        // KV pool in/out must agree at build time (in-place contract), and both profile
        // extremes must describe a well-formed paged pool and page table.
        check::check(in[kIN_KV_CACHE_IDX].desc.dims.nbDims == out[kOUT_KV_CACHE_IDX].desc.dims.nbDims,
            "QsaAttentionPlugin: present_key_value must match past_key_value rank.");
        check::check(isQsaPagedPoolShape(in[kIN_KV_CACHE_IDX].max, mNumKVHeads, mHeadSize, mIndexerHeadDim)
                && isQsaPagedPoolShape(out[kOUT_KV_CACHE_IDX].max, mNumKVHeads, mHeadSize, mIndexerHeadDim),
            "QsaAttentionPlugin: KV pool profile max must be [2, numPages, kTOKENS_PER_PAGE, Hkv, "
            "head_size + indexer_head_dim].");
        check::check(in[kIN_KV_PAGE_TABLE_IDX].max.nbDims == 3 && in[kIN_KV_PAGE_TABLE_IDX].max.d[1] == 2,
            "QsaAttentionPlugin: kv_page_table profile max must be [N, 2, maxPagesPerSeq].");
        // Token-major contract at every profile extreme: rank-2 activations and rope rows, N from
        // query_lengths, and a page-table row per entry.
        auto const validExecutionProfile = [](Dims const& qkv, Dims const& indexQk, Dims const& rope,
                                               Dims const& queryLengths, Dims const& pageTable) {
            return qkv.nbDims == 2 && qkv.d[0] > 0 && indexQk.nbDims == 2 && rope.nbDims == 2
                && rope.d[1] == kQSA_ROPE_TABLE_DIM && queryLengths.nbDims == 1 && queryLengths.d[0] > 0
                && pageTable.d[0] == queryLengths.d[0];
        };
        check::check(validExecutionProfile(in[kIN_QKV_IDX].min, in[kIN_INDEX_QK_IDX].min, in[kIN_ROPE_COS_SIN_IDX].min,
                         in[kIN_QUERY_LENGTHS_IDX].min, in[kIN_KV_PAGE_TABLE_IDX].min)
                && validExecutionProfile(in[kIN_QKV_IDX].opt, in[kIN_INDEX_QK_IDX].opt, in[kIN_ROPE_COS_SIN_IDX].opt,
                    in[kIN_QUERY_LENGTHS_IDX].opt, in[kIN_KV_PAGE_TABLE_IDX].opt)
                && validExecutionProfile(in[kIN_QKV_IDX].max, in[kIN_INDEX_QK_IDX].max, in[kIN_ROPE_COS_SIN_IDX].max,
                    in[kIN_QUERY_LENGTHS_IDX].max, in[kIN_KV_PAGE_TABLE_IDX].max),
            "QsaAttentionPlugin: qkv / index_qk must be token-major [T_exec, C], rope_rotary_cos_sin [T_exec, 64], and "
            "kv_page_table rows must equal the query_lengths extent at every profile extreme.");
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("QsaAttentionPlugin '%s' configurePlugin failed: %s", mLayerName.c_str(), e.what());
        return -1;
    }
}

size_t QsaAttentionPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, [[maybe_unused]] int32_t nbInputs,
    [[maybe_unused]] DynamicPluginTensorDesc const* outputs, [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    try
    {
        // Sized from the profile MAX dims day-one: the returned size is serialized into the
        // engine, so it must cover every runtime (N, S_pad) split in all supported phases.
        // The indexer sort slots grow with S_pad at fixed T_exec while the per-entry slots grow
        // with N (qsaIndexerRunner.cu, getQsaIndexerWorkspaceSize), so no single split
        // dominates: take the max over every N the profile admits.
        int64_t const maxTokens = inputs[kIN_QKV_IDX].max.d[0];
        int64_t const maxBatchSize = std::max<int64_t>(1, inputs[kIN_QUERY_LENGTHS_IDX].max.d[0]);
        int64_t const maxPagesPerSeq = inputs[kIN_KV_PAGE_TABLE_IDX].max.d[2];
        int32_t const chunkBlocks = static_cast<int32_t>(qsaMaxBlocksForPages(maxPagesPerSeq));
        size_t prefillBytes = 0;
        for (int64_t numSequences = 1; numSequences <= maxBatchSize; ++numSequences)
        {
            int64_t const seqLen = std::max<int64_t>(1, maxTokens / numSequences);
            int32_t const initialBlocks
                = static_cast<int32_t>((seqLen + kernel::kQSA_COMPRESS_RATIO - 1) / kernel::kQSA_COMPRESS_RATIO);
            prefillBytes = std::max(
                {prefillBytes, getQsaPrefillWorkspaceSize(numSequences, seqLen, initialBlocks, mNumQHeads, mHeadSize),
                    getQsaPrefillWorkspaceSize(numSequences, seqLen, chunkBlocks, mNumQHeads, mHeadSize)});
            if (mUseSpecVerifyState)
            {
                // Verify rejects S_pad beyond the kernel's node bound, so size it at that bound.
                int64_t const verifySeqLen = std::min<int64_t>(seqLen, kernel::kQSA_VERIFY_MAX_NODES);
                prefillBytes = std::max(prefillBytes,
                    getQsaVerifyWorkspaceSize(numSequences, verifySeqLen, chunkBlocks, mNumQHeads, mHeadSize));
            }
        }
        return std::max(
            prefillBytes, getQsaDecodeWorkspaceSize(maxBatchSize, maxPagesPerSeq, mNumQHeads, mNumKVHeads, mHeadSize));
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("QsaAttentionPlugin '%s' getWorkspaceSize failed: %s", mLayerName.c_str(), e.what());
        return 0;
    }
}

int32_t QsaAttentionPlugin::getAliasedInput(int32_t outputIndex) noexcept
{
    // The present-KV output aliases the past-KV input at runtime, but declaring it makes
    // Myelin reject the engine (same WAR as AttentionPlugin::getAliasedInput) — the LLM
    // runtime binds past/present to the same address instead.
    static_cast<void>(outputIndex);
    return -1;
}

// ---------------------------------------------------------------------------
// IPluginV3OneRuntime — execution
// ---------------------------------------------------------------------------

int32_t QsaAttentionPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        return enqueueImpl(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("QsaAttentionPlugin '%s': enqueue failed: %s", mLayerName.c_str(), e.what());
        return -1;
    }
    catch (...)
    {
        LOG_ERROR("QsaAttentionPlugin '%s': enqueue failed with a non-standard exception.", mLayerName.c_str());
        return -1;
    }
}

int32_t QsaAttentionPlugin::enqueueImpl(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    check::check(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr,
        "QsaAttentionPlugin received null enqueue descriptors or bindings.");

    // Phase and geometry come from the shared ragged decoder: it validates the carrier ranks,
    // the [N + 1] offsets, the phase extent, N_ctx and T_exec % N == 0, and throws into
    // enqueue()'s try/catch (phase 8 included). Context phases use entry-padded rows;
    // single-token decode requires T_exec == N, guaranteed by the decoder.
    RaggedPluginMetadata const ragged = decodeRaggedPluginMetadata("QsaAttentionPlugin", inputDesc[kIN_QKV_IDX],
        inputDesc[kIN_QUERY_LENGTHS_IDX], inputDesc[kIN_QUERY_START_OFFSETS_IDX],
        inputDesc[kIN_EXECUTION_PHASE_MARKER_IDX], inputDesc[kIN_CONTEXT_SEQUENCE_COUNT_IDX]);
    if (mUseSpecVerifyState)
    {
        // Only validates the marker extent against the phase: a chain arrives as straight-line
        // parents / depths, so both layouts take the same verify path.
        decodeSpecVerifyLayout("QsaAttentionPlugin", inputDesc[kIN_SPEC_VERIFY_LAYOUT_IDX], ragged.phase);
    }

    // Decode (3) has its own path. Context (1, 2) and verify (4, 5) share the pipeline below and differ
    // only in the indexer: verify scores tree paths, and its nodes complete no block in the pool; the seal
    // at the start of the next chunk or verify compresses the accepted ones.
    bool verify = false;
    switch (ragged.phase)
    {
    case rt::ExecutionPhase::kContextPrefill:
    case rt::ExecutionPhase::kContextChunk: break;
    case rt::ExecutionPhase::kSpecDraftProposal:
    case rt::ExecutionPhase::kSpecTargetVerify:
        check::check(mUseSpecVerifyState, "QsaAttentionPlugin: speculative phases require use_spec_verify_state.");
        verify = true;
        break;
    case rt::ExecutionPhase::kAutoregressiveDecode:
        return enqueueDecode(inputDesc, outputDesc, inputs, outputs, workspace, stream, ragged.numSequences);
    default:
        LOG_ERROR(
            "QsaAttentionPlugin: execution phase extent %d is not supported; QSA accepts only CONTEXT_PREFILL "
            "(1), CONTEXT_CHUNK (2), AUTOREGRESSIVE_DECODE (3), SPEC_DRAFT_PROPOSAL (4) and SPEC_TARGET_VERIFY (5).",
            static_cast<int32_t>(ragged.phase));
        return -1;
    }

    bool const hasPast = ragged.phase != rt::ExecutionPhase::kContextPrefill;
    PluginTensorDesc const& packedQKVInputDesc = inputDesc[kIN_QKV_IDX];
    int32_t const runtimeBatchSize = ragged.numSequences;
    int32_t const physicalTokens = ragged.physicalTokens;
    int32_t const runtimeSeqLen = physicalTokens / runtimeBatchSize; // S_pad: the decoder guarantees T_exec % N == 0
    int32_t const combinedHeads = mNumQHeads + 2 * mNumKVHeads;
    check::check(packedQKVInputDesc.dims.nbDims == 2 && packedQKVInputDesc.dims.d[1] == combinedHeads * mHeadSize,
        "QsaAttentionPlugin: packed QKV must be [T_exec, (Hq + 2*Hkv) * head_size].");

    PluginTensorDesc const& indexQkInputDesc = inputDesc[kIN_INDEX_QK_IDX];
    check::check(indexQkInputDesc.dims.nbDims == 2 && indexQkInputDesc.dims.d[0] == physicalTokens
            && indexQkInputDesc.dims.d[1] == (mIndexerNumHeads + 1) * mIndexerHeadDim,
        "QsaAttentionPlugin: index_qk must be [T_exec, (indexer_n_heads + 1) * indexer_head_dim] matching QKV.");

    // The runtime's token-aligned rope gather: one [64] cos/sin row per physical token, shared
    // by the main partial rope and the indexer rope (padding rows are never read).
    PluginTensorDesc const& ropeCosSinDesc = inputDesc[kIN_ROPE_COS_SIN_IDX];
    check::check(ropeCosSinDesc.dims.nbDims == 2 && ropeCosSinDesc.dims.d[0] == physicalTokens
            && ropeCosSinDesc.dims.d[1] == kQSA_ROPE_TABLE_DIM,
        "QsaAttentionPlugin: rope_rotary_cos_sin must be the token-aligned [T_exec, 64] gather.");

    PluginTensorDesc const& kvCacheInputDesc = inputDesc[kIN_KV_CACHE_IDX];
    check::check(isQsaPagedPoolShape(kvCacheInputDesc.dims, mNumKVHeads, mHeadSize, mIndexerHeadDim)
            && kvCacheInputDesc.dims.d[1] > 0,
        "QsaAttentionPlugin requires the widened paged KV pool [2, numPages, kTOKENS_PER_PAGE, Hkv, "
        "head_size + indexer_head_dim].");
    check::check(inputs[kIN_KV_CACHE_IDX] != nullptr && inputs[kIN_KV_PAGE_TABLE_IDX] != nullptr
            && outputs[kOUT_KV_CACHE_IDX] != nullptr,
        "QsaAttentionPlugin requires non-null KV pool and page-table bindings.");

    PluginTensorDesc const& kvPageTableInputDesc = inputDesc[kIN_KV_PAGE_TABLE_IDX];
    check::check(kvPageTableInputDesc.dims.d[0] == runtimeBatchSize && kvPageTableInputDesc.dims.d[1] == 2
            && kvPageTableInputDesc.dims.d[2] > 0,
        "QsaAttentionPlugin: kv_page_table must be [N, 2, maxPagesPerSeq].");
    int32_t const maxPagesPerSeq = static_cast<int32_t>(kvPageTableInputDesc.dims.d[2]);
    int32_t const numPages = static_cast<int32_t>(kvCacheInputDesc.dims.d[1]);
    check::check(numPages >= maxPagesPerSeq,
        "QsaAttentionPlugin requires at least one physical page per logical page-table column.");
    check::check(static_cast<int64_t>(maxPagesPerSeq) * rt::kTOKENS_PER_PAGE >= runtimeSeqLen,
        "QsaAttentionPlugin: the page table must cover the runtime sequence length.");
    if (verify)
    {
        check::check(runtimeSeqLen <= kernel::kQSA_VERIFY_MAX_NODES,
            "QsaAttentionPlugin: verify supports at most " + std::to_string(kernel::kQSA_VERIFY_MAX_NODES)
                + " padded tree nodes per entry.");
        check::check(inputDesc[kIN_TREE_PARENT_IDS_IDX].dims.nbDims == 1
                && inputDesc[kIN_TREE_PARENT_IDS_IDX].dims.d[0] == physicalTokens
                && inputDesc[kIN_TREE_DEPTHS_IDX].dims.nbDims == 1
                && inputDesc[kIN_TREE_DEPTHS_IDX].dims.d[0] == physicalTokens
                && inputs[kIN_TREE_PARENT_IDS_IDX] != nullptr && inputs[kIN_TREE_DEPTHS_IDX] != nullptr,
            "QsaAttentionPlugin: verify requires tree_parent_ids / tree_depths [T_exec].");
    }

    // Gamma engine-weight inputs are device-resident at engine load; all four are required.
    check::check(inputDesc[kIN_Q_NORM_GAMMA_IDX].dims.d[0] == mHeadSize
            && inputDesc[kIN_K_NORM_GAMMA_IDX].dims.d[0] == mHeadSize,
        "QsaAttentionPlugin: q/k_norm_gamma length must equal head_size.");
    check::check(inputDesc[kIN_INDEXER_Q_NORM_GAMMA_IDX].dims.d[0] == mIndexerHeadDim
            && inputDesc[kIN_INDEXER_K_NORM_GAMMA_IDX].dims.d[0] == mIndexerHeadDim,
        "QsaAttentionPlugin: indexer q/k norm gamma length must equal indexer_head_dim.");
    auto const* qNormGammaDevicePtr = static_cast<half const*>(inputs[kIN_Q_NORM_GAMMA_IDX]);
    auto const* kNormGammaDevicePtr = static_cast<half const*>(inputs[kIN_K_NORM_GAMMA_IDX]);
    auto const* indexerQGammaDevicePtr = static_cast<half const*>(inputs[kIN_INDEXER_Q_NORM_GAMMA_IDX]);
    auto const* indexerKGammaDevicePtr = static_cast<half const*>(inputs[kIN_INDEXER_K_NORM_GAMMA_IDX]);

    auto* alignedWorkspacePtr = static_cast<std::byte*>(workspace);
    if (alignedWorkspacePtr == nullptr
        || reinterpret_cast<uintptr_t>(alignedWorkspacePtr) % static_cast<uintptr_t>(kDEVICE_ALIGNMENT) != 0)
    {
        LOG_ERROR("QsaAttentionPlugin workspace pointer is not aligned to device alignment granularity.");
        return -1;
    }

    // Non-owned tensor views over the bindings.
    rt::Tensor const packedQKVTensor(const_cast<void*>(inputs[kIN_QKV_IDX]),
        rt::Coords{runtimeBatchSize, runtimeSeqLen, combinedHeads, mHeadSize}, rt::DeviceType::kGPU,
        fromTrt(packedQKVInputDesc.type));
    // Query lengths count valid rows in this call; chunk writes begin at past_lengths.
    rt::Tensor const contextLengthTensor(const_cast<void*>(inputs[kIN_QUERY_LENGTHS_IDX]),
        rt::toCoords(inputDesc[kIN_QUERY_LENGTHS_IDX].dims), rt::DeviceType::kGPU,
        fromTrt(inputDesc[kIN_QUERY_LENGTHS_IDX].type));
    // The [T_exec, 64] rows viewed as [N, S_pad, 64]: bit-identical memory, and the batch extent
    // N is what tokenAlignedRope needs (a batch-1 view would collapse every entry to row 0).
    rt::Tensor const ropeCosSinTensor(const_cast<void*>(inputs[kIN_ROPE_COS_SIN_IDX]),
        rt::Coords{runtimeBatchSize, runtimeSeqLen, kQSA_ROPE_TABLE_DIM}, rt::DeviceType::kGPU,
        fromTrt(ropeCosSinDesc.type));
    check::check(!hasPast
            || (inputDesc[kIN_PAST_LENGTHS_IDX].dims.nbDims == 1
                && inputDesc[kIN_PAST_LENGTHS_IDX].dims.d[0] == runtimeBatchSize
                && inputs[kIN_PAST_LENGTHS_IDX] != nullptr),
        "QsaAttentionPlugin: appending to a past requires past_lengths [N].");
    rt::Tensor const kvCacheStartIdxTensor = hasPast
        ? rt::Tensor(const_cast<void*>(inputs[kIN_PAST_LENGTHS_IDX]), rt::Coords{runtimeBatchSize},
              rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32)
        : rt::Tensor{};
    rt::Tensor presentKVCacheTensor(outputs[kOUT_KV_CACHE_IDX], rt::toCoords(kvCacheInputDesc.dims),
        rt::DeviceType::kGPU, fromTrt(kvCacheInputDesc.type));
    rt::Tensor const kvPageTableTensor(const_cast<void*>(inputs[kIN_KV_PAGE_TABLE_IDX]),
        rt::toCoords(kvPageTableInputDesc.dims), rt::DeviceType::kGPU, fromTrt(kvPageTableInputDesc.type));
    int32_t const* const pageTable = kvPageTableTensor.dataPointer<int32_t>();
    // Rank-3 [T_exec, Hq, D] output viewed as [N, S_pad, Hq, D] for the kernels.
    rt::Tensor attentionOutputTensor(outputs[kOUT_ATTENTION_IDX],
        rt::Coords{runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, rt::DeviceType::kGPU,
        fromTrt(outputDesc[kOUT_ATTENTION_IDX].type));

    // ---- Workspace carving (order matches the selected context workspace helper) ----
    rt::Tensor cuQSeqLensTensor
        = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);
    rt::Tensor cuKVSeqLensTensor
        = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);
    rt::Tensor kvCacheEndIdxsTensor
        = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize}, DataType::kINT32);
    rt::Tensor qInputTensor = assignTensorFromWorkspace(
        alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
    rt::Tensor outIdxTensor = assignTensorFromWorkspace(
        alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, kernel::kQSA_INDEX_WIDTH}, DataType::kINT32);
    int32_t const maxBlocks = hasPast ? static_cast<int32_t>(qsaMaxBlocksForPages(maxPagesPerSeq))
                                      : (runtimeSeqLen + kernel::kQSA_COMPRESS_RATIO - 1) / kernel::kQSA_COMPRESS_RATIO;
    size_t const indexerWorkspaceBytes = verify
        ? kernel::getQsaIndexerVerifyWorkspaceSize(runtimeBatchSize, runtimeSeqLen, maxBlocks)
        : kernel::getQsaIndexerWorkspaceSize(runtimeBatchSize, runtimeSeqLen, maxBlocks, true);
    rt::Tensor indexerWorkspaceTensor = assignTensorFromWorkspace(
        alignedWorkspacePtr, {static_cast<int64_t>(indexerWorkspaceBytes)}, DataType::kINT8);

    // ---- Stage 0: cumulative-length metadata (cuQSeqLens feeds the rope kernel's ragged
    // Q zeroing and padded-row K/V write skip; the padded prefix sums are not needed). ----
    kernel::calCuQCuKVSeqLensAndKVEndIdxs(contextLengthTensor, kvCacheStartIdxTensor, cuQSeqLensTensor,
        cuKVSeqLensTensor, kvCacheEndIdxsTensor, std::nullopt, runtimeSeqLen, stream);

    // The rope kernel subtracts S_pad; metadata supplies past + S_pad even for ragged queries.
    kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor, rt::OptionalInputTensor{kvCacheEndIdxsTensor},
        rt::OptionalInputTensor{}, packedQKVTensor, qInputTensor, presentKVCacheTensor, 1.0F /* kScale */,
        1.0F /* vScale */, stream, pageTable, maxPagesPerSeq, nullptr /* kScratch */, nullptr /* vScratch */,
        nullptr /* fp8QOut */, 1.0F /* qScale */, qNormGammaDevicePtr, kNormGammaDevicePtr, mRmsNormEps,
        false /* qkNormPostRope */, rt::OptionalInputTensor{cuQSeqLensTensor}, true /* writeKVCache */,
        false /* enablePdl */, true /* tokenAlignedRope */);

    // ---- Stage 2: QSA indexer — per-query top-512 blocks expanded to int32 token indices (padding rows -1). ----
    kernel::QsaIndexerPoolState poolState{};
    poolState.poolPtr = presentKVCacheTensor.rawPointer();
    poolState.pageTable = pageTable;
    poolState.maxPagesPerSeq = maxPagesPerSeq;
    poolState.numPages = numPages;
    poolState.numKVHeads = mNumKVHeads;
    poolState.poolHeadDim = mHeadSize + mIndexerHeadDim;
    poolState.headSize = mHeadSize;
    if (verify)
    {
        kernel::runQsaIndexerVerify<half>(outIdxTensor.dataPointer<int32_t>(),
            static_cast<half const*>(inputs[kIN_INDEX_QK_IDX]), ropeCosSinTensor.dataPointer<float>(),
            static_cast<int32_t const*>(inputs[kIN_TREE_PARENT_IDS_IDX]),
            static_cast<int32_t const*>(inputs[kIN_TREE_DEPTHS_IDX]), contextLengthTensor.dataPointer<int32_t>(),
            static_cast<int32_t const*>(kvCacheStartIdxTensor.rawPointer()), indexerQGammaDevicePtr,
            indexerKGammaDevicePtr, mRmsNormEps, poolState, indexerWorkspaceTensor.rawPointer(), indexerWorkspaceBytes,
            runtimeBatchSize, runtimeSeqLen, maxBlocks, stream);
    }
    else
    {
        kernel::runQsaIndexerPrefill<half>(outIdxTensor.dataPointer<int32_t>(),
            static_cast<half const*>(inputs[kIN_INDEX_QK_IDX]), ropeCosSinTensor.dataPointer<float>(),
            contextLengthTensor.dataPointer<int32_t>(), static_cast<int32_t const*>(kvCacheStartIdxTensor.rawPointer()),
            indexerQGammaDevicePtr, indexerKGammaDevicePtr, mRmsNormEps, &poolState,
            indexerWorkspaceTensor.rawPointer(), indexerWorkspaceBytes, runtimeBatchSize, runtimeSeqLen, maxBlocks,
            stream);
    }

    QsaSparsePrefillParams params{};
    params.qPtr = qInputTensor.rawPointer();
    params.kvPoolPtr = presentKVCacheTensor.rawPointer();
    params.pageTable = pageTable;
    params.indices = outIdxTensor.dataPointer<int32_t>();
    params.queryLengths = contextLengthTensor.dataPointer<int32_t>();
    params.oPtr = attentionOutputTensor.rawPointer();
    params.batchSize = runtimeBatchSize;
    params.seqLen = runtimeSeqLen;
    params.numQHeads = mNumQHeads;
    params.numKVHeads = mNumKVHeads;
    params.headDim = mHeadSize;
    params.poolHeadDim = mHeadSize + mIndexerHeadDim;
    params.numFlatPages = 2 * numPages;
    params.maxPagesPerSeq = maxPagesPerSeq;
    params.topK = kernel::kQSA_INDEX_WIDTH;
    params.attentionScale = mAttentionScale;
    params.stream = stream;
    return CuteDslQsaSparsePrefillRunner::run(trt_edgellm::DataType::kHALF, params) ? 0 : -1;
}

int32_t QsaAttentionPlugin::enqueueDecode(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream, int32_t runtimeBatchSize)
{
    // Reached from enqueueImpl for AUTOREGRESSIVE_DECODE only: the ragged decoder has already
    // established T_exec == N (one new token per entry). The TOTAL per-sequence lengths
    // including the token being decoded come from attention_sequence_lengths; query_lengths
    // is 1 for every decode row and must not be used here.
    PluginTensorDesc const& packedQKVInputDesc = inputDesc[kIN_QKV_IDX];
    int32_t const combinedHeads = mNumQHeads + 2 * mNumKVHeads;
    check::check(packedQKVInputDesc.dims.nbDims == 2 && packedQKVInputDesc.dims.d[0] == runtimeBatchSize
            && packedQKVInputDesc.dims.d[1] == combinedHeads * mHeadSize,
        "QsaAttentionPlugin: decode packed QKV must be [N, (Hq + 2*Hkv) * head_size].");

    PluginTensorDesc const& indexQkInputDesc = inputDesc[kIN_INDEX_QK_IDX];
    check::check(indexQkInputDesc.dims.nbDims == 2 && indexQkInputDesc.dims.d[0] == runtimeBatchSize
            && indexQkInputDesc.dims.d[1] == (mIndexerNumHeads + 1) * mIndexerHeadDim,
        "QsaAttentionPlugin: decode index_qk must be [N, (indexer_n_heads + 1) * indexer_head_dim].");

    // Token-aligned rope rows: row b is the cos/sin of entry b's new token (position ctx_b - 1,
    // resolved by the runtime). The block-completion rope of a PAST position comes from the
    // block rope row kept in the pool tails (qsaIndexerKernels.h), not from this binding.
    PluginTensorDesc const& ropeCosSinDesc = inputDesc[kIN_ROPE_COS_SIN_IDX];
    check::check(ropeCosSinDesc.dims.nbDims == 2 && ropeCosSinDesc.dims.d[0] == runtimeBatchSize
            && ropeCosSinDesc.dims.d[1] == kQSA_ROPE_TABLE_DIM,
        "QsaAttentionPlugin: decode rope_rotary_cos_sin must be the token-aligned [N, 64] gather.");

    PluginTensorDesc const& kvCacheInputDesc = inputDesc[kIN_KV_CACHE_IDX];
    check::check(isQsaPagedPoolShape(kvCacheInputDesc.dims, mNumKVHeads, mHeadSize, mIndexerHeadDim)
            && kvCacheInputDesc.dims.d[1] > 0,
        "QsaAttentionPlugin requires the widened paged KV pool [2, numPages, kTOKENS_PER_PAGE, Hkv, "
        "head_size + indexer_head_dim].");
    check::check(inputs[kIN_KV_CACHE_IDX] != nullptr && inputs[kIN_KV_PAGE_TABLE_IDX] != nullptr
            && outputs[kOUT_KV_CACHE_IDX] != nullptr,
        "QsaAttentionPlugin requires non-null KV pool and page-table bindings.");

    PluginTensorDesc const& kvPageTableInputDesc = inputDesc[kIN_KV_PAGE_TABLE_IDX];
    check::check(kvPageTableInputDesc.dims.d[0] == runtimeBatchSize && kvPageTableInputDesc.dims.d[1] == 2
            && kvPageTableInputDesc.dims.d[2] > 0,
        "QsaAttentionPlugin: kv_page_table must be [N, 2, maxPagesPerSeq].");
    int32_t const maxPagesPerSeq = static_cast<int32_t>(kvPageTableInputDesc.dims.d[2]);
    int32_t const numPages = static_cast<int32_t>(kvCacheInputDesc.dims.d[1]);
    check::check(numPages >= maxPagesPerSeq,
        "QsaAttentionPlugin requires at least one physical page per logical page-table column.");

    check::check(inputDesc[kIN_Q_NORM_GAMMA_IDX].dims.d[0] == mHeadSize
            && inputDesc[kIN_K_NORM_GAMMA_IDX].dims.d[0] == mHeadSize,
        "QsaAttentionPlugin: q/k_norm_gamma length must equal head_size.");
    check::check(inputDesc[kIN_INDEXER_Q_NORM_GAMMA_IDX].dims.d[0] == mIndexerHeadDim
            && inputDesc[kIN_INDEXER_K_NORM_GAMMA_IDX].dims.d[0] == mIndexerHeadDim,
        "QsaAttentionPlugin: indexer q/k norm gamma length must equal indexer_head_dim.");
    auto const* qNormGammaDevicePtr = static_cast<half const*>(inputs[kIN_Q_NORM_GAMMA_IDX]);
    auto const* kNormGammaDevicePtr = static_cast<half const*>(inputs[kIN_K_NORM_GAMMA_IDX]);
    auto const* indexerQGammaDevicePtr = static_cast<half const*>(inputs[kIN_INDEXER_Q_NORM_GAMMA_IDX]);
    auto const* indexerKGammaDevicePtr = static_cast<half const*>(inputs[kIN_INDEXER_K_NORM_GAMMA_IDX]);

    auto* alignedWorkspacePtr = static_cast<std::byte*>(workspace);
    if (alignedWorkspacePtr == nullptr
        || reinterpret_cast<uintptr_t>(alignedWorkspacePtr) % static_cast<uintptr_t>(kDEVICE_ALIGNMENT) != 0)
    {
        LOG_ERROR("QsaAttentionPlugin workspace pointer is not aligned to device alignment granularity.");
        return -1;
    }

    // Non-owned tensor views over the bindings.
    rt::Tensor const packedQKVTensor(const_cast<void*>(inputs[kIN_QKV_IDX]),
        rt::Coords{runtimeBatchSize, 1, combinedHeads, mHeadSize}, rt::DeviceType::kGPU,
        fromTrt(packedQKVInputDesc.type));
    // TOTAL lengths (past + the new token) from the ragged tail: slot 12, never slot 3.
    rt::Tensor const contextLengthTensor(const_cast<void*>(inputs[kIN_ATTENTION_SEQUENCE_LENGTHS_IDX]),
        rt::toCoords(inputDesc[kIN_ATTENTION_SEQUENCE_LENGTHS_IDX].dims), rt::DeviceType::kGPU,
        fromTrt(inputDesc[kIN_ATTENTION_SEQUENCE_LENGTHS_IDX].type));
    check::check(contextLengthTensor.getShape()[0] == runtimeBatchSize,
        "QsaAttentionPlugin: attention_sequence_lengths must have shape [N].");
    // [N, 64] rows viewed as [N, 1, 64] so tokenAlignedRope reads row b for entry b.
    rt::Tensor const ropeCosSinTensor(const_cast<void*>(inputs[kIN_ROPE_COS_SIN_IDX]),
        rt::Coords{runtimeBatchSize, 1, kQSA_ROPE_TABLE_DIM}, rt::DeviceType::kGPU, fromTrt(ropeCosSinDesc.type));
    rt::Tensor presentKVCacheTensor(outputs[kOUT_KV_CACHE_IDX], rt::toCoords(kvCacheInputDesc.dims),
        rt::DeviceType::kGPU, fromTrt(kvCacheInputDesc.type));
    rt::Tensor const kvPageTableTensor(const_cast<void*>(inputs[kIN_KV_PAGE_TABLE_IDX]),
        rt::toCoords(kvPageTableInputDesc.dims), rt::DeviceType::kGPU, fromTrt(kvPageTableInputDesc.type));
    int32_t const* const pageTable = kvPageTableTensor.dataPointer<int32_t>();
    // Rank-3 [N, Hq, D] output viewed as [N, 1, Hq, D] for the kernels.
    rt::Tensor attentionOutputTensor(outputs[kOUT_ATTENTION_IDX],
        rt::Coords{runtimeBatchSize, 1, mNumQHeads, mHeadSize}, rt::DeviceType::kGPU,
        fromTrt(outputDesc[kOUT_ATTENTION_IDX].type));

    // ---- Workspace carving (order must match getQsaDecodeWorkspaceSize) ----
    int64_t const partialSlots
        = static_cast<int64_t>(runtimeBatchSize) * mNumKVHeads * CuteDslQsaSparseDecodeRunner::kMaxSplits;
    rt::Tensor qInputTensor
        = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize, 1, mNumQHeads, mHeadSize}, DataType::kHALF);
    rt::Tensor outIdxTensor = assignTensorFromWorkspace(
        alignedWorkspacePtr, {runtimeBatchSize, 1, kernel::kQSA_INDEX_WIDTH}, DataType::kINT32);
    rt::Tensor partialOTensor = assignTensorFromWorkspace(
        alignedWorkspacePtr, {partialSlots, CuteDslQsaSparseDecodeRunner::kPartialRows, mHeadSize}, DataType::kFLOAT);
    rt::Tensor partialStatsTensor = assignTensorFromWorkspace(
        alignedWorkspacePtr, {partialSlots, 2, CuteDslQsaSparseDecodeRunner::kPartialRows}, DataType::kFLOAT);
    rt::Tensor splitCountersTensor
        = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize, mNumKVHeads}, DataType::kINT32);
    int32_t const maxBlocks = static_cast<int32_t>(qsaMaxBlocksForPages(maxPagesPerSeq));
    size_t const indexerWorkspaceBytes = kernel::getQsaIndexerDecodeWorkspaceSize(runtimeBatchSize, maxBlocks);
    rt::Tensor indexerWorkspaceTensor = assignTensorFromWorkspace(
        alignedWorkspacePtr, {static_cast<int64_t>(indexerWorkspaceBytes)}, DataType::kINT8);

    // ---- Stage 1: rope + append the new token's K/V to the pool. The rope reads the
    // token-aligned row b; kvCacheEndLens = attention_sequence_lengths (TOTAL lengths) so the
    // kernel writes the pool row ctx - 1 (precedent: AttentionPlugin vanilla decoding; widened
    // rows: see launchApplyRopeFromPackedToSplit). ----
    kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor, rt::OptionalInputTensor{contextLengthTensor},
        rt::OptionalInputTensor{}, packedQKVTensor, qInputTensor, presentKVCacheTensor, 1.0F /* kScale */,
        1.0F /* vScale */, stream, pageTable, maxPagesPerSeq, nullptr /* kScratch */, nullptr /* vScratch */,
        nullptr /* fp8QOut */, 1.0F /* qScale */, qNormGammaDevicePtr, kNormGammaDevicePtr, mRmsNormEps,
        false /* qkNormPostRope */, std::nullopt /* cuQSeqLens */, true /* writeKVCache */, false /* enablePdl */,
        true /* tokenAlignedRope */);

    // ---- Stage 2: indexer decode — B1 writes the new token's tails and compresses the block
    // it completes, B2 scores the visible blocks, B3 zeroes the split counters (guaranteeing
    // zero-on-entry for stage 3, including the first step on a garbage workspace) and emits
    // the top-512 expanded index list. ----
    kernel::QsaIndexerPoolState poolState{};
    poolState.poolPtr = presentKVCacheTensor.rawPointer();
    poolState.pageTable = pageTable;
    poolState.maxPagesPerSeq = maxPagesPerSeq;
    poolState.numPages = numPages;
    poolState.numKVHeads = mNumKVHeads;
    poolState.poolHeadDim = mHeadSize + mIndexerHeadDim;
    poolState.headSize = mHeadSize;
    kernel::runQsaIndexerDecode<half>(outIdxTensor.dataPointer<int32_t>(),
        static_cast<half const*>(inputs[kIN_INDEX_QK_IDX]), ropeCosSinTensor.dataPointer<float>(),
        contextLengthTensor.dataPointer<int32_t>(), indexerQGammaDevicePtr, indexerKGammaDevicePtr, mRmsNormEps,
        poolState, splitCountersTensor.dataPointer<int32_t>(), indexerWorkspaceTensor.rawPointer(),
        indexerWorkspaceBytes, runtimeBatchSize, maxBlocks, stream);

    // ---- Stage 3: single-launch split-K sparse attention over the paged pool. ----
    QsaSparseDecodeParams params{};
    params.qPtr = qInputTensor.rawPointer();
    params.kvPoolPtr = presentKVCacheTensor.rawPointer();
    params.pageTable = pageTable;
    params.indices = outIdxTensor.dataPointer<int32_t>();
    params.contextLengths = contextLengthTensor.dataPointer<int32_t>();
    params.oPtr = attentionOutputTensor.rawPointer();
    params.partialO = partialOTensor.dataPointer<float>();
    params.partialStats = partialStatsTensor.dataPointer<float>();
    params.splitCounters = splitCountersTensor.dataPointer<int32_t>();
    params.batchSize = runtimeBatchSize;
    params.numQHeads = mNumQHeads;
    params.numKVHeads = mNumKVHeads;
    params.headDim = mHeadSize;
    params.poolHeadDim = mHeadSize + mIndexerHeadDim;
    params.numFlatPages = 2 * numPages;
    params.maxPagesPerSeq = maxPagesPerSeq;
    params.topK = kernel::kQSA_INDEX_WIDTH;
    params.attentionScale = mAttentionScale;
    params.stream = stream;
    if (!CuteDslQsaSparseDecodeRunner::run(trt_edgellm::DataType::kHALF, params))
    {
        return -1;
    }
    return 0;
}

int32_t QsaAttentionPlugin::onShapeChange([[maybe_unused]] PluginTensorDesc const* in,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] PluginTensorDesc const* out,
    [[maybe_unused]] int32_t nbOutputs) noexcept
{
    // Both kernel families load lazily on first use (the indexer via NVRTC + driver
    // load, the CuTe-DSL sparse kernel via cudaLibraryLoad in preflight). Warm them up
    // here so a CUDA Graph capture of enqueue never observes those one-time host-side
    // steps.
    try
    {
        kernel::ensureQsaIndexerKernelsLoaded<half>();
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("QsaAttentionPlugin '%s' failed to JIT the QSA indexer kernels: %s", mLayerName.c_str(), e.what());
        return -1;
    }
    if (!CuteDslQsaSparsePrefillRunner::preflight(trt_edgellm::DataType::kHALF, nullptr)
        || !CuteDslQsaSparseDecodeRunner::preflight(trt_edgellm::DataType::kHALF, nullptr))
    {
        LOG_ERROR("QsaAttentionPlugin '%s' failed to load the CuTe DSL sparse kernel modules.", mLayerName.c_str());
        return -1;
    }
    return 0;
}

IPluginV3* QsaAttentionPlugin::attachToContext([[maybe_unused]] IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* QsaAttentionPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("num_q_heads", &mNumQHeads, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("num_kv_heads", &mNumKVHeads, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("head_size", &mHeadSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("indexer_n_heads", &mIndexerNumHeads, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("indexer_head_dim", &mIndexerHeadDim, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("indexer_budget", &mIndexerBudget, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("indexer_compress_ratio", &mIndexerCompressRatio, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("attention_scale", &mAttentionScale, PluginFieldType::kFLOAT32, 1);
    mDataToSerialize.emplace_back("rms_norm_eps", &mRmsNormEps, PluginFieldType::kFLOAT32, 1);
    mUseSpecVerifyStateField = mUseSpecVerifyState ? 1 : 0;
    mDataToSerialize.emplace_back("use_spec_verify_state", &mUseSpecVerifyStateField, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ---------------------------------------------------------------------------
// Creator
// ---------------------------------------------------------------------------

QsaAttentionPluginCreator::QsaAttentionPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_q_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_kv_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("indexer_n_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("indexer_head_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("indexer_budget", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("indexer_compress_ratio", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("attention_scale", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("rms_norm_eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_spec_verify_state", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* QsaAttentionPluginCreator::getPluginName() const noexcept
{
    return kQSA_ATTENTION_PLUGIN_NAME;
}

PluginFieldCollection const* QsaAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void QsaAttentionPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace ? libNamespace : "";
}

char const* QsaAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* QsaAttentionPluginCreator::getPluginVersion() const noexcept
{
    return kQSA_ATTENTION_PLUGIN_VERSION;
}

IPluginV3* QsaAttentionPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, [[maybe_unused]] TensorRTPhase phase) noexcept
{
    try
    {
        auto plugin = std::make_unique<QsaAttentionPlugin>(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create QsaAttentionPlugin: %s", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
