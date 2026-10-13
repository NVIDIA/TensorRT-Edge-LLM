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
#include "kernels/contextAttentionKernels/attentionScaleUtils.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/qsaAttention/cuteDslQsaSparseRunner.h"
#include "kernels/qsaIndexer/qsaIndexerKernels.h"
#include "kernels/qsaIndexer/qsaIndexerRunner.h"
#include "plugins/utils/pluginUtils.h"

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

namespace
{
constexpr char const* kQSA_ATTENTION_PLUGIN_VERSION{"1"};
constexpr char const* kQSA_ATTENTION_PLUGIN_NAME{"QsaAttentionPlugin"};

// Input / output index mapping. ALL inputs are required — there are no optional slots, so
// the positional contract is fixed.
constexpr int32_t kIN_QKV_IDX{0};                   //!< [B, S, (Hq + 2*Hkv) * D] FP16 packed QKV
constexpr int32_t kIN_INDEX_QK_IDX{1};              //!< [B, S, (idxHeads + 1) * idxDim] FP16 indexer projection
constexpr int32_t kIN_KV_CACHE_IDX{2};              //!< [2, numPages, 128, Hkv, D + idxDim] FP16 widened pool
constexpr int32_t kIN_CONTEXT_LENGTH_IDX{3};        //!< [B] INT32 live lengths
constexpr int32_t kIN_ROPE_COS_SIN_IDX{4};          //!< [1, maxPos, 64] FP32 rope table (main + indexer)
constexpr int32_t kIN_KV_CACHE_START_IDX{5};        //!< [0] INT32 = prefill; [B] (values = past lengths) = decode
constexpr int32_t kIN_KV_PAGE_TABLE_IDX{6};         //!< [B, 2, maxPagesPerSeq] INT32
constexpr int32_t kIN_Q_NORM_GAMMA_IDX{7};          //!< [D] FP16 Constant, pre-folded (1 + w)
constexpr int32_t kIN_K_NORM_GAMMA_IDX{8};          //!< [D] FP16 Constant, pre-folded (1 + w)
constexpr int32_t kIN_INDEXER_Q_NORM_GAMMA_IDX{9};  //!< [idxDim] FP16 Constant, RAW w (kernel adds 1)
constexpr int32_t kIN_INDEXER_K_NORM_GAMMA_IDX{10}; //!< [idxDim] FP16 Constant, RAW w

constexpr int32_t kOUT_ATTENTION_IDX{0}; //!< [B, S, Hq, D] FP16
constexpr int32_t kOUT_KV_CACHE_IDX{1};  //!< paged pool, in-place aliased with input 2

constexpr int32_t kNUM_REQUIRED_INPUTS{11};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{2};

//! The shared rope table serves the main partial rope (64 of 256) and the indexer rope
//! (64 of 128); its last dim is therefore pinned to the indexer rotary dim.
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

// Prefill workspace layout (cumulative; each slot 128-byte aligned):
//
//   Slot  | Shape                       | Type  | Used by
//   ------+-----------------------------+-------+------------------------------------------
//   0     | [B+1]                       | INT32 | cuQSeqLens          (rope ragged-Q zeroing)
//   1     | [B+1]                       | INT32 | cuKVSeqLens         (metadata kernel output)
//   2     | [B]                         | INT32 | kvCacheEndIdxs      (metadata kernel output)
//   3     | [B, S, Hq, D]               | HALF  | qScratch            (roped Q mirror)
//   4     | [B, S, Hkv, D]              | HALF  | kScratch            (roped K mirror)
//   5     | [B, S, Hkv, D]              | HALF  | vScratch            (V mirror)
//   6     | [B, S, kQSA_INDEX_WIDTH]    | INT32 | outIdx              (indexer output)
//   7     | getQsaIndexerWorkspaceSize  | BYTE  | indexer scratch     (scores / sort / temp)
size_t getQsaPrefillWorkspaceSize(
    int64_t batchSize, int64_t seqLen, int32_t numQHeads, int32_t numKVHeads, int32_t headSize)
{
    size_t workspaceSize = 0;
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize + 1}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize + 1}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize, seqLen, numQHeads, headSize}, DataType::kHALF);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize, seqLen, numKVHeads, headSize}, DataType::kHALF);
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {batchSize, seqLen, numKVHeads, headSize}, DataType::kHALF);
    workspaceSize
        = accumulateWorkspaceSize(workspaceSize, {batchSize, seqLen, kernel::kQSA_INDEX_WIDTH}, DataType::kINT32);
    workspaceSize = accumulateWorkspaceSize(workspaceSize,
        {static_cast<int64_t>(
            kernel::getQsaIndexerWorkspaceSize(static_cast<int32_t>(batchSize), static_cast<int32_t>(seqLen)))},
        DataType::kINT8);
    return workspaceSize;
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
    float attentionScale, float rmsNormEps)
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

    ELLM_CHECK(
        CuteDslQsaSparsePrefillRunner::canImplement(mNumQHeads, mNumKVHeads, mHeadSize, smVersion, DataType::kHALF),
        "QsaAttentionPlugin: no QSA sparse prefill kernel for Hq=" + std::to_string(mNumQHeads)
            + ", Hkv=" + std::to_string(mNumKVHeads) + ", D=" + std::to_string(mHeadSize) + " on SM"
            + std::to_string(smVersion) + " (build with -DENABLE_CUTE_DSL=\"fmha;qsa\").");
    ELLM_CHECK(CuteDslQsaSparseDecodeRunner::canImplement(
                   mNumQHeads, mNumKVHeads, mHeadSize, mHeadSize + mIndexerHeadDim, smVersion, DataType::kHALF),
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
            mIndexerHeadDim, mIndexerBudget, mIndexerCompressRatio, mAttentionScale, mRmsNormEps);
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

int32_t QsaAttentionPlugin::getNbOutputs() const noexcept
{
    return kNUM_REQUIRED_OUTPUTS;
}

int32_t QsaAttentionPlugin::getOutputDataTypes(DataType* outputTypes, [[maybe_unused]] int32_t nbOutputs,
    DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    try
    {
        assert(nbInputs == kNUM_REQUIRED_INPUTS);
        assert(nbOutputs == kNUM_REQUIRED_OUTPUTS);
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
        assert(nbInputs == kNUM_REQUIRED_INPUTS);
        assert(nbOutputs == kNUM_REQUIRED_OUTPUTS);
        // attn_output: [B, S, Hq, D] — B/S from the packed QKV input.
        outputs[kOUT_ATTENTION_IDX].nbDims = 4;
        outputs[kOUT_ATTENTION_IDX].d[0] = inputs[kIN_QKV_IDX].d[0];
        outputs[kOUT_ATTENTION_IDX].d[1] = inputs[kIN_QKV_IDX].d[1];
        outputs[kOUT_ATTENTION_IDX].d[2] = exprBuilder.constant(mNumQHeads);
        outputs[kOUT_ATTENTION_IDX].d[3] = exprBuilder.constant(mHeadSize);
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
    if (nbInputs != kNUM_REQUIRED_INPUTS || nbOutputs != kNUM_REQUIRED_OUTPUTS)
    {
        LOG_ERROR("QsaAttentionPlugin '%s' expects %d inputs and %d outputs, got %d inputs and %d outputs.",
            mLayerName.c_str(), kNUM_REQUIRED_INPUTS, kNUM_REQUIRED_OUTPUTS, nbInputs, nbOutputs);
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

    switch (pos)
    {
    case kIN_QKV_IDX:
        return desc.type == DataType::kHALF && desc.dims.nbDims == 3
            && (desc.dims.d[2] == -1 || desc.dims.d[2] == (mNumQHeads + 2 * mNumKVHeads) * mHeadSize);
    case kIN_INDEX_QK_IDX:
        return desc.type == DataType::kHALF && desc.dims.nbDims == 3
            && (desc.dims.d[2] == -1 || desc.dims.d[2] == (mIndexerNumHeads + 1) * mIndexerHeadDim);
    case kIN_KV_CACHE_IDX:
        return desc.type == DataType::kHALF && isQsaPagedPoolShape(desc.dims, mNumKVHeads, mHeadSize, mIndexerHeadDim);
    case kIN_CONTEXT_LENGTH_IDX: return desc.type == DataType::kINT32 && desc.dims.nbDims == 1;
    case kIN_ROPE_COS_SIN_IDX:
        return desc.type == DataType::kFLOAT && desc.dims.nbDims == 3
            && (desc.dims.d[2] == -1 || desc.dims.d[2] == kQSA_ROPE_TABLE_DIM);
    case kIN_KV_CACHE_START_IDX: return desc.type == DataType::kINT32 && desc.dims.nbDims == 1;
    case kIN_KV_PAGE_TABLE_IDX:
        return desc.type == DataType::kINT32 && desc.dims.nbDims == 3 && (desc.dims.d[1] == -1 || desc.dims.d[1] == 2);
    case kIN_Q_NORM_GAMMA_IDX: return isHalfVector(mHeadSize);
    case kIN_K_NORM_GAMMA_IDX: return isHalfVector(mHeadSize);
    case kIN_INDEXER_Q_NORM_GAMMA_IDX: return isHalfVector(mIndexerHeadDim);
    case kIN_INDEXER_K_NORM_GAMMA_IDX: return isHalfVector(mIndexerHeadDim);
    default: break;
    }

    int32_t const outPos = pos - kNUM_REQUIRED_INPUTS;
    if (outPos == kOUT_ATTENTION_IDX)
    {
        return desc.type == DataType::kHALF && desc.dims.nbDims == 4;
    }
    if (outPos == kOUT_KV_CACHE_IDX)
    {
        return desc.type == inOut[kIN_KV_CACHE_IDX].desc.type
            && isQsaPagedPoolShape(desc.dims, mNumKVHeads, mHeadSize, mIndexerHeadDim);
    }
    return false;
}

int32_t QsaAttentionPlugin::configurePlugin(DynamicPluginTensorDesc const* in, [[maybe_unused]] int32_t nbInputs,
    DynamicPluginTensorDesc const* out, [[maybe_unused]] int32_t nbOutputs) noexcept
{
    try
    {
        assert(nbInputs == kNUM_REQUIRED_INPUTS);
        assert(nbOutputs == kNUM_REQUIRED_OUTPUTS);
        // KV pool in/out must agree at build time (in-place contract), and both profile
        // extremes must describe a well-formed paged pool and page table.
        check::check(in[kIN_KV_CACHE_IDX].desc.dims.nbDims == out[kOUT_KV_CACHE_IDX].desc.dims.nbDims,
            "QsaAttentionPlugin: present_key_value must match past_key_value rank.");
        check::check(isQsaPagedPoolShape(in[kIN_KV_CACHE_IDX].max, mNumKVHeads, mHeadSize, mIndexerHeadDim)
                && isQsaPagedPoolShape(out[kOUT_KV_CACHE_IDX].max, mNumKVHeads, mHeadSize, mIndexerHeadDim),
            "QsaAttentionPlugin: KV pool profile max must be [2, numPages, kTOKENS_PER_PAGE, Hkv, "
            "head_size + indexer_head_dim].");
        check::check(in[kIN_KV_PAGE_TABLE_IDX].max.nbDims == 3 && in[kIN_KV_PAGE_TABLE_IDX].max.d[1] == 2,
            "QsaAttentionPlugin: kv_page_table profile max must be [B, 2, maxPagesPerSeq].");
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
        // engine, so it must cover every runtime shape of this profile in both modes.
        int64_t const maxBatchSize = inputs[kIN_QKV_IDX].max.d[0];
        int64_t const maxSeqLen = inputs[kIN_QKV_IDX].max.d[1];
        int64_t const maxPagesPerSeq = inputs[kIN_KV_PAGE_TABLE_IDX].max.d[2];
        return std::max(getQsaPrefillWorkspaceSize(maxBatchSize, maxSeqLen, mNumQHeads, mNumKVHeads, mHeadSize),
            getQsaDecodeWorkspaceSize(maxBatchSize, maxPagesPerSeq, mNumQHeads, mNumKVHeads, mHeadSize));
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
        LOG_ERROR("QsaAttentionPlugin: enqueue failed: %s", e.what());
        return -1;
    }
    catch (...)
    {
        LOG_ERROR("QsaAttentionPlugin: enqueue failed with a non-standard exception.");
        return -1;
    }
}

int32_t QsaAttentionPlugin::enqueueImpl(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    check::check(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr,
        "QsaAttentionPlugin received null enqueue descriptors or bindings.");

    // Mode deduction (same convention as AttentionPlugin): kvcache_start_index runtime shape
    // [0] selects NORMAL_PREFILL; shape [B] selects decode and requires S == 1 (chunked
    // prefill and speculative decode are not supported). In decode the start values are the
    // per-sequence PAST lengths; the kernels derive everything from context_lengths (the
    // TOTAL lengths including the new token), so the values are never dereferenced here.
    int64_t const startIdxLen = inputDesc[kIN_KV_CACHE_START_IDX].dims.d[0];
    int64_t const modeBatchSize = inputDesc[kIN_QKV_IDX].dims.d[0];
    int64_t const modeSeqLen = inputDesc[kIN_QKV_IDX].dims.d[1];
    if (startIdxLen != 0)
    {
        if (startIdxLen != modeBatchSize)
        {
            LOG_ERROR(
                "QsaAttentionPlugin: kvcache_start_index must have runtime shape [0] (prefill) or [B] "
                "(decode); got [%lld] with B=%lld.",
                static_cast<long long>(startIdxLen), static_cast<long long>(modeBatchSize));
            return -1;
        }
        if (modeSeqLen != 1)
        {
            LOG_ERROR(
                "QsaAttentionPlugin: decode (kvcache_start_index shape [B]) requires S == 1; got S=%lld. "
                "Chunked prefill / speculative decode are not supported.",
                static_cast<long long>(modeSeqLen));
            return -1;
        }
        return enqueueDecode(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }

    PluginTensorDesc const& packedQKVInputDesc = inputDesc[kIN_QKV_IDX];
    int32_t const runtimeBatchSize = static_cast<int32_t>(packedQKVInputDesc.dims.d[0]);
    int32_t const runtimeSeqLen = static_cast<int32_t>(packedQKVInputDesc.dims.d[1]);
    int32_t const combinedHeads = mNumQHeads + 2 * mNumKVHeads;
    check::check(packedQKVInputDesc.dims.d[2] == combinedHeads * mHeadSize,
        "QsaAttentionPlugin: packed QKV last dim must equal (Hq + 2*Hkv) * head_size.");

    PluginTensorDesc const& indexQkInputDesc = inputDesc[kIN_INDEX_QK_IDX];
    check::check(indexQkInputDesc.dims.d[0] == runtimeBatchSize && indexQkInputDesc.dims.d[1] == runtimeSeqLen
            && indexQkInputDesc.dims.d[2] == (mIndexerNumHeads + 1) * mIndexerHeadDim,
        "QsaAttentionPlugin: index_qk must be [B, S, (indexer_n_heads + 1) * indexer_head_dim] matching QKV.");

    // One rope table serves the main partial rope and the indexer rope; the indexer kernels
    // take a batch-invariant [maxPos, 64] view, so a batch-1 table is required.
    PluginTensorDesc const& ropeCosSinDesc = inputDesc[kIN_ROPE_COS_SIN_IDX];
    check::check(ropeCosSinDesc.dims.d[0] == 1 && ropeCosSinDesc.dims.d[2] == kQSA_ROPE_TABLE_DIM,
        "QsaAttentionPlugin requires a batch-1 rope table [1, maxPos, 64].");
    check::check(ropeCosSinDesc.dims.d[1] >= runtimeSeqLen,
        "QsaAttentionPlugin: rope table must cover the runtime sequence length.");

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
        "QsaAttentionPlugin: kv_page_table must be [B, 2, maxPagesPerSeq].");
    int32_t const maxPagesPerSeq = static_cast<int32_t>(kvPageTableInputDesc.dims.d[2]);
    int32_t const numPages = static_cast<int32_t>(kvCacheInputDesc.dims.d[1]);
    check::check(numPages >= maxPagesPerSeq,
        "QsaAttentionPlugin requires at least one physical page per logical page-table column.");
    check::check(static_cast<int64_t>(maxPagesPerSeq) * rt::kTOKENS_PER_PAGE >= runtimeSeqLen,
        "QsaAttentionPlugin: the page table must cover the runtime sequence length.");

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
        packedQKVInputDesc.type);
    rt::Tensor const contextLengthTensor(const_cast<void*>(inputs[kIN_CONTEXT_LENGTH_IDX]),
        rt::Coords{inputDesc[kIN_CONTEXT_LENGTH_IDX].dims}, rt::DeviceType::kGPU,
        inputDesc[kIN_CONTEXT_LENGTH_IDX].type);
    check::check(contextLengthTensor.getShape()[0] == runtimeBatchSize,
        "QsaAttentionPlugin: context_lengths must have shape [B].");
    rt::Tensor const ropeCosSinTensor(const_cast<void*>(inputs[kIN_ROPE_COS_SIN_IDX]), rt::Coords{ropeCosSinDesc.dims},
        rt::DeviceType::kGPU, ropeCosSinDesc.type);
    rt::Tensor const kvCacheStartIdxTensor(const_cast<void*>(inputs[kIN_KV_CACHE_START_IDX]),
        rt::Coords{inputDesc[kIN_KV_CACHE_START_IDX].dims}, rt::DeviceType::kGPU,
        inputDesc[kIN_KV_CACHE_START_IDX].type);
    rt::Tensor presentKVCacheTensor(
        outputs[kOUT_KV_CACHE_IDX], rt::Coords{kvCacheInputDesc.dims}, rt::DeviceType::kGPU, kvCacheInputDesc.type);
    rt::Tensor const kvPageTableTensor(const_cast<void*>(inputs[kIN_KV_PAGE_TABLE_IDX]),
        rt::Coords{kvPageTableInputDesc.dims}, rt::DeviceType::kGPU, kvPageTableInputDesc.type);
    int32_t const* const pageTable = kvPageTableTensor.dataPointer<int32_t>();
    rt::Tensor attentionOutputTensor(outputs[kOUT_ATTENTION_IDX], rt::Coords{outputDesc[kOUT_ATTENTION_IDX].dims},
        rt::DeviceType::kGPU, outputDesc[kOUT_ATTENTION_IDX].type);

    // ---- Workspace carving (order must match getQsaWorkspaceSize) ----
    rt::Tensor cuQSeqLensTensor
        = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);
    rt::Tensor cuKVSeqLensTensor
        = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);
    rt::Tensor kvCacheEndIdxsTensor
        = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize}, DataType::kINT32);
    rt::Tensor qInputTensor = assignTensorFromWorkspace(
        alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
    rt::Tensor kInputTensor = assignTensorFromWorkspace(
        alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumKVHeads, mHeadSize}, DataType::kHALF);
    rt::Tensor vInputTensor = assignTensorFromWorkspace(
        alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumKVHeads, mHeadSize}, DataType::kHALF);
    rt::Tensor outIdxTensor = assignTensorFromWorkspace(
        alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, kernel::kQSA_INDEX_WIDTH}, DataType::kINT32);
    size_t const indexerWorkspaceBytes = kernel::getQsaIndexerWorkspaceSize(runtimeBatchSize, runtimeSeqLen);
    rt::Tensor indexerWorkspaceTensor = assignTensorFromWorkspace(
        alignedWorkspacePtr, {static_cast<int64_t>(indexerWorkspaceBytes)}, DataType::kINT8);

    // ---- Stage 0: cumulative-length metadata (cuQSeqLens feeds the rope kernel's ragged
    // Q zeroing and padded-row K/V write skip; the padded prefix sums are not needed). ----
    kernel::calCuQCuKVSeqLensAndKVEndIdxs(contextLengthTensor, kvCacheStartIdxTensor, cuQSeqLensTensor,
        cuKVSeqLensTensor, kvCacheEndIdxsTensor, std::nullopt, runtimeSeqLen, stream);

    // ---- Stage 1: split packed QKV, fused qk-norm (pre-folded 1 + w gammas) + partial
    // rope, paged KV write-through, and the dense K/V mirrors the sparse kernel consumes.
    // The widened pool row (head_size + indexer_head_dim) flows through the pool tensor shape
    // (see launchApplyRopeFromPackedToSplit); the mirrors stay head_size-dense. ----
    kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor, rt::OptionalInputTensor{}, rt::OptionalInputTensor{},
        packedQKVTensor, qInputTensor, presentKVCacheTensor, 1.0F /* kScale */, 1.0F /* vScale */, stream, pageTable,
        maxPagesPerSeq, kInputTensor.rawPointer(), vInputTensor.rawPointer(), nullptr /* fp8QOut */, 1.0F /* qScale */,
        qNormGammaDevicePtr, kNormGammaDevicePtr, mRmsNormEps, false /* qkNormPostRope */,
        rt::OptionalInputTensor{cuQSeqLensTensor});

    // ---- Stage 2: QSA indexer — per-query top-512 block selection expanded to the int32
    // token-index lists (padding rows all -1). The pool state makes the pipeline also
    // persist every complete block's kbar and the trailing incomplete block's raw index-K
    // into the pool tails, which is what the decode steps resume from. ----
    kernel::QsaIndexerPoolState poolState{};
    poolState.poolPtr = presentKVCacheTensor.rawPointer();
    poolState.pageTable = pageTable;
    poolState.maxPagesPerSeq = maxPagesPerSeq;
    poolState.numPages = numPages;
    poolState.numKVHeads = mNumKVHeads;
    poolState.poolHeadDim = mHeadSize + mIndexerHeadDim;
    poolState.headSize = mHeadSize;
    kernel::runQsaIndexerPrefill<half>(outIdxTensor.dataPointer<int32_t>(),
        static_cast<half const*>(inputs[kIN_INDEX_QK_IDX]), ropeCosSinTensor.dataPointer<float>(),
        contextLengthTensor.dataPointer<int32_t>(), indexerQGammaDevicePtr, indexerKGammaDevicePtr, mRmsNormEps,
        indexerWorkspaceTensor.rawPointer(), indexerWorkspaceBytes, runtimeBatchSize, runtimeSeqLen, &poolState,
        stream);

    // ---- Stage 3: sparse GQA attention over the dense mirrors. ----
    if (!CuteDslQsaSparsePrefillRunner::preflight(DataType::kHALF, stream))
    {
        return -1;
    }
    QsaSparsePrefillParams params{};
    params.qPtr = qInputTensor.rawPointer();
    params.kPtr = kInputTensor.rawPointer();
    params.vPtr = vInputTensor.rawPointer();
    params.oPtr = attentionOutputTensor.rawPointer();
    params.indices = outIdxTensor.dataPointer<int32_t>();
    params.contextLengths = contextLengthTensor.dataPointer<int32_t>();
    params.batchSize = runtimeBatchSize;
    params.seqLen = runtimeSeqLen;
    params.numQHeads = mNumQHeads;
    params.numKVHeads = mNumKVHeads;
    params.headDim = mHeadSize;
    params.topK = kernel::kQSA_INDEX_WIDTH;
    params.attentionScale = mAttentionScale;
    params.stream = stream;
    if (!CuteDslQsaSparsePrefillRunner::run(DataType::kHALF, params))
    {
        return -1;
    }
    return 0;
}

int32_t QsaAttentionPlugin::enqueueDecode(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    // Callers reach this only through enqueueImpl's mode deduction: S == 1 and
    // kvcache_start_index has runtime shape [B]. context_lengths carries the TOTAL
    // per-sequence lengths including the token being decoded.
    PluginTensorDesc const& packedQKVInputDesc = inputDesc[kIN_QKV_IDX];
    int32_t const runtimeBatchSize = static_cast<int32_t>(packedQKVInputDesc.dims.d[0]);
    int32_t const combinedHeads = mNumQHeads + 2 * mNumKVHeads;
    check::check(packedQKVInputDesc.dims.d[2] == combinedHeads * mHeadSize,
        "QsaAttentionPlugin: packed QKV last dim must equal (Hq + 2*Hkv) * head_size.");

    PluginTensorDesc const& indexQkInputDesc = inputDesc[kIN_INDEX_QK_IDX];
    check::check(indexQkInputDesc.dims.d[0] == runtimeBatchSize && indexQkInputDesc.dims.d[1] == 1
            && indexQkInputDesc.dims.d[2] == (mIndexerNumHeads + 1) * mIndexerHeadDim,
        "QsaAttentionPlugin: decode index_qk must be [B, 1, (indexer_n_heads + 1) * indexer_head_dim].");

    // The rope table must cover every decode position; positions are device-side
    // (context_lengths - 1), so coverage against the profile-declared table length is the
    // builder's contract — only the batch-1 layout is validated here.
    PluginTensorDesc const& ropeCosSinDesc = inputDesc[kIN_ROPE_COS_SIN_IDX];
    check::check(ropeCosSinDesc.dims.d[0] == 1 && ropeCosSinDesc.dims.d[2] == kQSA_ROPE_TABLE_DIM,
        "QsaAttentionPlugin requires a batch-1 rope table [1, maxPos, 64].");

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
        "QsaAttentionPlugin: kv_page_table must be [B, 2, maxPagesPerSeq].");
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
        rt::Coords{runtimeBatchSize, 1, combinedHeads, mHeadSize}, rt::DeviceType::kGPU, packedQKVInputDesc.type);
    rt::Tensor const contextLengthTensor(const_cast<void*>(inputs[kIN_CONTEXT_LENGTH_IDX]),
        rt::Coords{inputDesc[kIN_CONTEXT_LENGTH_IDX].dims}, rt::DeviceType::kGPU,
        inputDesc[kIN_CONTEXT_LENGTH_IDX].type);
    check::check(contextLengthTensor.getShape()[0] == runtimeBatchSize,
        "QsaAttentionPlugin: context_lengths must have shape [B].");
    rt::Tensor const ropeCosSinTensor(const_cast<void*>(inputs[kIN_ROPE_COS_SIN_IDX]), rt::Coords{ropeCosSinDesc.dims},
        rt::DeviceType::kGPU, ropeCosSinDesc.type);
    rt::Tensor presentKVCacheTensor(
        outputs[kOUT_KV_CACHE_IDX], rt::Coords{kvCacheInputDesc.dims}, rt::DeviceType::kGPU, kvCacheInputDesc.type);
    rt::Tensor const kvPageTableTensor(const_cast<void*>(inputs[kIN_KV_PAGE_TABLE_IDX]),
        rt::Coords{kvPageTableInputDesc.dims}, rt::DeviceType::kGPU, kvPageTableInputDesc.type);
    int32_t const* const pageTable = kvPageTableTensor.dataPointer<int32_t>();
    rt::Tensor attentionOutputTensor(outputs[kOUT_ATTENTION_IDX], rt::Coords{outputDesc[kOUT_ATTENTION_IDX].dims},
        rt::DeviceType::kGPU, outputDesc[kOUT_ATTENTION_IDX].type);

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

    // ---- Stage 1: rope + append the new token's K/V to the pool. kvCacheEndLens =
    // context_lengths (TOTAL lengths) so the kernel ropes at position ctx - 1 and writes the
    // pool row ctx - 1 (precedent: AttentionPlugin vanilla decoding; widened rows: see
    // launchApplyRopeFromPackedToSplit). ----
    kernel::launchApplyRopeFromPackedToSplit(ropeCosSinTensor, rt::OptionalInputTensor{contextLengthTensor},
        rt::OptionalInputTensor{}, packedQKVTensor, qInputTensor, presentKVCacheTensor, 1.0F /* kScale */,
        1.0F /* vScale */, stream, pageTable, maxPagesPerSeq, nullptr /* kScratch */, nullptr /* vScratch */,
        nullptr /* fp8QOut */, 1.0F /* qScale */, qNormGammaDevicePtr, kNormGammaDevicePtr, mRmsNormEps);

    // ---- Stage 2: indexer decode — B1 either persists the new token's raw index-K tail
    // (block still incomplete) or compresses the block it completes into a kbar V-tail,
    // B2 scores the visible blocks, B3 zeroes
    // the split counters (guaranteeing zero-on-entry for stage 3, including the first step
    // on a garbage workspace) and emits the top-512 expanded index list. ----
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
    if (!CuteDslQsaSparseDecodeRunner::preflight(DataType::kHALF, stream))
    {
        return -1;
    }
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
    if (!CuteDslQsaSparseDecodeRunner::run(DataType::kHALF, params))
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
    if (!CuteDslQsaSparsePrefillRunner::preflight(DataType::kHALF, nullptr)
        || !CuteDslQsaSparseDecodeRunner::preflight(DataType::kHALF, nullptr))
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
