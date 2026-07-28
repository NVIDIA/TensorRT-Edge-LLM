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

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "kernels/contextAttentionKernels/contextFMHARunner.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "plugins/utils/pluginUtils.h"

// CuTe DSL FMHA kernel (Blackwell SM100+)
#ifdef CUTE_DSL_FMHA_ENABLED
#include "kernels/contextAttentionKernels/cuteDslFMHARunner.h"
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

// Serialization tag that distinguishes the newer [tag][context_attention_mask_type] trailing payload
// from the legacy single-int32 bidirectional-prefill flag written by older engines.
constexpr int32_t kContextMaskSerializationTag{0x4D41534B}; // 'MASK'

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
constexpr int32_t kIN_OPTIONAL_ATTN_MASK_IDX{7};
constexpr int32_t kIN_OPTIONAL_ATTN_POS_ID_IDX{8};
constexpr int32_t kOUT_ATTENTION_IDX{0};
constexpr int32_t kOUT_KV_CACHE_IDX{1};

// Reflect the count of Inputs and Outputs of the AttentionPlugin,
// these definitions shall be consistent.
constexpr int32_t kNUM_REQUIRED_INPUTS{7};
constexpr int32_t kNUM_TREE_ATTN_OPTIONAL_INPUTS{2};
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

} // namespace

// Static class fields initialization
PluginFieldCollection AttentionPluginCreator::mFieldCollection{};
std::vector<PluginField> AttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AttentionPluginCreator);

AttentionPlugin::AttentionPlugin(std::string const& name, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t enableTreeAttention, int32_t enableFp8KVCache, int32_t slidingWindowSize,
    std::vector<float> const& qkvScales, int32_t contextAttentionMaskType)
    : mLayerName(name)
    , mNumQHeads(numQHeads)
    , mNumKVHeads(numKVHeads)
    , mHeadSize(headSize)
    , mEnableTreeAttention(enableTreeAttention)
    , mEnableFp8KVCache(enableFp8KVCache)
    , mQkvScales(enableFp8KVCache ? qkvScales : std::vector<float>{1.f, 1.f, 1.f})
    , mSlidingWindowSize(slidingWindowSize)
    , mContextAttentionMaskType(contextAttentionMaskType)
{
    if (mEnableFp8KVCache && mQkvScales.size() != 3)
    {
        throw std::runtime_error("FP8 KV cache enabled but qkv_scales has "
            + std::to_string(mQkvScales.size()) + " elements (expected 3). "
            "Re-export the model to include QKV scales [q, k, v].");
    }

    mSMVersion = getSMVersion();
    applyThorSMRenumberWAR(mSMVersion);

    LOG_DEBUG("AttentionPlugin FMHA path: %s, sliding_window: %s", mUseCuteDslFMHA ? "CuTe DSL FMHA" : "FMHA_v2",
        mSlidingWindowSize > 0 ? std::to_string(mSlidingWindowSize).c_str() : "disabled");

    auto const prefillMaskType = static_cast<ContextAttentionMaskType>(mContextAttentionMaskType);

    // Check FMHA implementation support and load the corresponding kernel module.
    bool canImplementFMHA = false;
#ifdef CUTE_DSL_FMHA_ENABLED
    // CuTe DSL FMHA currently supports the causal prefill path only.
    bool const useCausalFMHA = prefillMaskType == ContextAttentionMaskType::CAUSAL;
    if (useCausalFMHA && mUseCuteDslFMHA && CuteDslFMHARunner::canImplement(mHeadSize, mSMVersion))
    {
        if (CuteDslFMHARunner::loadLLMKernelModule())
        {
            canImplementFMHA = true;
            LOG_DEBUG("CuTe DSL FMHA kernel loaded for SM%d", mSMVersion);
        }
        else
        {
            LOG_WARNING("CuTe DSL FMHA kernel failed to load, falling back to FMHA_v2");
            mUseCuteDslFMHA = false;
        }
    }
    if (!canImplementFMHA)
#endif
    {
        // Fallback to FMHA_v2 cubins.
        canImplementFMHA = ContextFMHARunner::canImplement(
            mHeadSize, mSMVersion, mDataType, AttentionInputLayout::SEPARATE_Q_K_V, prefillMaskType);
        if (canImplementFMHA)
        {
            if (!ContextFMHARunner::loadContextFMHAKernels(mSMVersion, mDataType))
            {
                LOG_ERROR("Failed to load FMHA_v2 cubins for SM%d", mSMVersion);
                canImplementFMHA = false;
            }
        }
    }

    // XQA decode kernels are always needed regardless of FMHA path.
    bool const useSpecDecode = static_cast<bool>(mEnableTreeAttention);
    bool canImplementXQA = DecoderXQARunner::canImplement(
        mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion, mDataType, selectKvCacheDataType(mEnableFp8KVCache));
    if (canImplementXQA)
    {
        DecoderXQARunner::loadDecodeXQAKernels(
            mSMVersion, mDataType, selectKvCacheDataType(mEnableFp8KVCache), useSpecDecode);
    }

    if (!canImplementFMHA || !canImplementXQA)
    {
        LOG_ERROR(
            "Cannot implement AttentionPlugin configuration. FMHA: %s, XQA: %s, SM: %d, HeadSize: %d, NumQHeads: %d, "
            "NumKVHeads: %d",
            canImplementFMHA ? "supported" : "NOT supported", canImplementXQA ? "supported" : "NOT supported",
            mSMVersion, mHeadSize, mNumQHeads, mNumKVHeads);
        throw std::runtime_error("Cannot implement the AttentionPlugin configuration.");
    }
}

AttentionPlugin::AttentionPlugin(std::string const& name, std::byte const* data, size_t length)
    : mLayerName(name)
{
    deserializeValue(&data, &length, &mNumQHeads);
    deserializeValue(&data, &length, &mNumKVHeads);
    deserializeValue(&data, &length, &mHeadSize);
    deserializeValue(&data, &length, &mEnableTreeAttention);
    deserializeValue(&data, &length, &mEnableFp8KVCache);
    deserializeValue(&data, &length, &mSlidingWindowSize);

    // Optional trailing fields: [int32 qkv_scale_count][float * count] followed by either the newer
    // [int32 tag][int32 context_attention_mask_type] payload or the legacy [int32 bidirectional_prefill]
    // flag. Engines serialized before these fields default to unit QKV scales and causal prefill.
    if (length >= sizeof(int32_t))
    {
        int32_t qkvScaleCount = 0;
        deserializeValue(&data, &length, &qkvScaleCount);
        if (qkvScaleCount < 0 || length < static_cast<size_t>(qkvScaleCount) * sizeof(float))
        {
            throw std::runtime_error("Invalid qkv_scales payload in serialized AttentionPlugin data.");
        }

        std::vector<float> qkvScales(qkvScaleCount);
        for (int32_t i = 0; i < qkvScaleCount; ++i)
        {
            deserializeValue(&data, &length, &qkvScales[i]);
        }
        if (qkvScaleCount == 3)
        {
            mQkvScales = qkvScales;
        }
    }

    if (mEnableFp8KVCache && mQkvScales.size() != 3)
    {
        throw std::runtime_error(
            "FP8 KV cache enabled but qkv_scales missing or incomplete in serialized AttentionPlugin data.");
    }

    // New engines write [int32 tag][int32 maskType]; legacy engines wrote a single [int32
    // bidirectionalPrefill] flag (0 = causal, 1 = bidirectional/padding). Migrate the legacy flag.
    if (length >= 2 * sizeof(int32_t))
    {
        int32_t maskTag = 0;
        deserializeValue(&data, &length, &maskTag);
        check::check(maskTag == kContextMaskSerializationTag,
            "Unexpected serialization tag for AttentionPlugin context attention mask type");
        deserializeValue(&data, &length, &mContextAttentionMaskType);
    }
    else if (length >= sizeof(int32_t))
    {
        int32_t legacyBidirectionalPrefill = 0;
        deserializeValue(&data, &length, &legacyBidirectionalPrefill);
        mContextAttentionMaskType = (legacyBidirectionalPrefill != 0)
            ? static_cast<int32_t>(ContextAttentionMaskType::PADDING)
            : static_cast<int32_t>(ContextAttentionMaskType::CAUSAL);
    }
    check::check(length == 0, "Unexpected trailing bytes in serialized AttentionPlugin data");

    mSMVersion = getSMVersion();
    applyThorSMRenumberWAR(mSMVersion);

    LOG_DEBUG("AttentionPlugin FMHA path: %s", mUseCuteDslFMHA ? "CuTe DSL FMHA" : "FMHA_v2");

    // Load FMHA kernel module based on implementation support.
#ifdef CUTE_DSL_FMHA_ENABLED
    // CuTe DSL FMHA currently supports the causal prefill path only.
    bool const useCausalFMHA
        = static_cast<ContextAttentionMaskType>(mContextAttentionMaskType) == ContextAttentionMaskType::CAUSAL;
    if (useCausalFMHA && mUseCuteDslFMHA && CuteDslFMHARunner::canImplement(mHeadSize, mSMVersion))
    {
        if (!CuteDslFMHARunner::loadLLMKernelModule())
        {
            LOG_WARNING("CuTe DSL FMHA kernel failed to load, falling back to FMHA_v2");
            mUseCuteDslFMHA = false;
        }
    }
    if (!useCausalFMHA || !mUseCuteDslFMHA)
#endif
    {
        if (!ContextFMHARunner::loadContextFMHAKernels(mSMVersion, mDataType))
        {
            LOG_ERROR("Failed to load FMHA_v2 cubins for SM%d", mSMVersion);
        }
    }

    // XQA decode kernels are always needed regardless of FMHA path.
    bool const useSpecDecode = static_cast<bool>(mEnableTreeAttention);
    DecoderXQARunner::loadDecodeXQAKernels(
        mSMVersion, mDataType, selectKvCacheDataType(mEnableFp8KVCache), useSpecDecode);
}

AttentionPlugin::~AttentionPlugin() {}

IPluginV2DynamicExt* AttentionPlugin::clone() const noexcept
{
    AttentionPlugin* plugin = new AttentionPlugin(mLayerName, mNumQHeads, mNumKVHeads, mHeadSize, mEnableTreeAttention,
        mEnableFp8KVCache, mSlidingWindowSize, mQkvScales, mContextAttentionMaskType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

char const* AttentionPlugin::getPluginType() const noexcept
{
    return kATTENTION_PLUGIN_NAME;
}

char const* AttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void AttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

char const* AttentionPlugin::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

int32_t AttentionPlugin::getNbOutputs() const noexcept
{
    // At both context and generation phase, output attention result and kv-cache.
    return 2;
}

bool AttentionPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // Support context/generation phase inputs:
    //      Q tensor (linear FP16) with shape [B, S, Hq, D]
    //      K tensor (linear FP16) with shape [B, S, Hkv, D]
    //      V tensor (linear FP16) with shape [B, S, Hkv, D]
    //      KV-cache tensor (linear FP16/FP8) with shape [B, 2, Hkv, Smax, D], here Smax is the kvcache capacity
    //      buffer.
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
    auto checkQ = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
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

    auto checkKV = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
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

    auto checkKVCache = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
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
        status &= tensorDesc.dims.nbDims == 5;
        if (status)
        {
            auto const tensorDim = tensorDesc.dims;
            status &= tensorDim.d[1] == 2; // Specify K and V
            status &= tensorDim.d[2] == mNumKVHeads;
            status &= tensorDim.d[4] == mHeadSize;
        }
        return status;
    };

    auto checkSequenceLen = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        return status;
    };

    auto checkPosEncodingCosSin = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kFLOAT;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        status &= tensorDesc.dims.d[2] <= mHeadSize;
        return status;
    };

    auto checkAttentionMask = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        return status;
    };

    auto checkAttentionPosId = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 2;
        return status;
    };

    auto checkKVCacheStartIdx = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        return status;
    };

    auto checkAttentionOutput = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
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

    int32_t const expectedNbInputs = kNUM_REQUIRED_INPUTS + (mEnableTreeAttention ? kNUM_TREE_ATTN_OPTIONAL_INPUTS : 0);
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
        case kIN_Q_IDX: result = checkQ(inOut[pos]); break;
        case kIN_K_IDX: result = checkKV(inOut[pos]); break;
        case kIN_V_IDX: result = checkKV(inOut[pos]); break;
        case kIN_KV_CACHE_IDX: result = checkKVCache(inOut[pos]); break;
        case kIN_CONTEXT_LENGTH_IDX: result = checkSequenceLen(inOut[pos]); break;
        case kIN_ROPE_COS_SIN_IDX: result = checkPosEncodingCosSin(inOut[pos]); break;
        case kIN_KV_CACHE_START_IDX: result = checkKVCacheStartIdx(inOut[pos]); break;
        default: break;
        }

        // Handle optional inputs (tree attention mask/pos and FP8 scales) with dynamic ordering
        if (result && pos > kIN_KV_CACHE_START_IDX)
        {
            int32_t currentOptionalInputIdx = kIN_KV_CACHE_START_IDX + 1;
            if (mEnableTreeAttention)
            {
                if (pos == currentOptionalInputIdx)
                {
                    result = checkAttentionMask(inOut[pos]);
                }
                currentOptionalInputIdx++;
                if (pos == currentOptionalInputIdx)
                {
                    result = checkAttentionPosId(inOut[pos]);
                }
                currentOptionalInputIdx++;
            }
        }
    }
    else
    {
        int32_t outPos = pos - nbInputs;
        switch (outPos)
        {
        case kOUT_ATTENTION_IDX: result = checkAttentionOutput(inOut[pos]); break;
        case kOUT_KV_CACHE_IDX: result = checkKVCache(inOut[pos]); break;
        default: break;
        }
    }

    return result;
}

// IPluginV2Ext Methods
DataType AttentionPlugin::getOutputDataType([[maybe_unused]] int32_t index,
    [[maybe_unused]] nvinfer1::DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    // Output[0] (attention): always FP16 (follows Q input dtype).
    // Output[1] (KV cache) follows KV input dtype (HALF or FP8).
    if (index == kOUT_ATTENTION_IDX)
    {
        return inputTypes[kIN_Q_IDX];
    }
    return inputTypes[kIN_KV_CACHE_IDX];
}

DimsExprs AttentionPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
    [[maybe_unused]] int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Output[0] is attention result, has shape [B, S, Hq, D]. Refers to Q shape [B, S, Hq, D]
    DimsExprs output;
    if (outputIndex == kOUT_ATTENTION_IDX)
    {
        output.nbDims = 4;
        output.d[0] = inputs[kIN_Q_IDX].d[0];
        output.d[1] = inputs[kIN_Q_IDX].d[1];
        output.d[2] = exprBuilder.constant(mNumQHeads);
        output.d[3] = exprBuilder.constant(mHeadSize);
    }
    else if (outputIndex == kOUT_KV_CACHE_IDX)
    {
        // Output[1] is KVCache, same shape as input KV cache
        output.nbDims = 5;
        output.d[0] = inputs[kIN_KV_CACHE_IDX].d[0];
        output.d[1] = inputs[kIN_KV_CACHE_IDX].d[1];
        output.d[2] = inputs[kIN_KV_CACHE_IDX].d[2];
        output.d[3] = inputs[kIN_KV_CACHE_IDX].d[3];
        output.d[4] = inputs[kIN_KV_CACHE_IDX].d[4];
    }
    return output;
}

void AttentionPlugin::configurePlugin([[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* in,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* out,
    [[maybe_unused]] int32_t nbOutputs) noexcept
{
    return; // No need to configure anything since we will only use the runtime tensor shapes.
}

// Workspace layout (cumulative, worst-case across all execution paths):
//
//   Slot  | Shape                            | Type  | Used by
//   ------+----------------------------------+-------+------------------------------------------
//   0     | [B+1]                            | INT32 | cuQSeqLens          (prefill)
//   1     | [B+1]                            | INT32 | cuKVSeqLens         (prefill)
//   2     | [B]                              | INT32 | kvCacheEndIdxs      (prefill)
//   3     | [B+1]                            | INT32 | paddedCuKVSeqLens   (prefill, CuTe DSL)
//   4     | [B, 2, Hkv, Smax, D]             | HALF  | transposedKV        (FMHA_v2 chunked prefill)
//   5*    | [B, S, Hq, D]                    | FP8   | fp8Q                (CuTe DSL + FP8 prefill only)
//
//   * Slot 5 is conditionally allocated (CuTe DSL + FP8 KV cache only).
//
// Total allocation is the sum of all conditional slots (safe upper bound).
size_t AttentionPlugin::getWorkspaceSize([[maybe_unused]] nvinfer1::PluginTensorDesc const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputs,
    [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    // TensorRT will supply max profile shape for each input/output tensor across all optimization profiles.
    // We will request workspace to keep intermediate tensors under prefill/decode phase executions.
    // Obtain max supported batch size from the input tensor shapes. The Q input tensor will be in shape
    // [B, S, Hq, D] where S is padded the max length of the input sequence within this batch.
    PluginTensorDesc const& qInputDesc = inputs[kIN_Q_IDX];
    int64_t const maxBatchSize = qInputDesc.dims.d[0];

    // Obtain max KV cache capacity from the KV cache tensor shape.
    // The KV cache tensor has shape [B, 2, num_kv_heads, capacity, head_dim]
    PluginTensorDesc const& kvCacheDesc = inputs[kIN_KV_CACHE_IDX];
    int64_t const maxKVCacheCapacity = kvCacheDesc.dims.d[3];

    size_t workspaceSize = 0;

    // CuQSeqLens for FMHA.
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {maxBatchSize + 1}, DataType::kINT32);

    // Always reserve workspace memory to prepare for chunked prefill decoding. The implementation should be further
    // optimized to avoid the workspace size overhead.

    // CuTotalKvCacheLens to describe the cumulative length of KV tensors.
    workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{maxBatchSize + 1}, DataType::kINT32);
    // KVCache ends that denote the end index of each KVCache lane after adding current contents.
    workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{maxBatchSize}, DataType::kINT32);
    // Padded cumulative KV sequence lengths for CuTe DSL FMHA.
    workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{maxBatchSize + 1}, DataType::kINT32);
    // KV workspace for split K and V tensors (split into K and V halves by pointer arithmetic in enqueue).
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{maxBatchSize, 2, mNumKVHeads, maxKVCacheCapacity, mHeadSize}, DataType::kHALF);

    // FP8 Q output: RoPE kernel writes FP8 Q to this workspace buffer (CuTe DSL FMHA path).
#ifdef CUTE_DSL_FMHA_ENABLED
    if (mUseCuteDslFMHA && mEnableFp8KVCache)
    {
        int64_t const maxSeqLen = qInputDesc.dims.d[1];
        workspaceSize = accumulateWorkspaceSize(
            workspaceSize, rt::Coords{maxBatchSize, maxSeqLen, mNumQHeads, mHeadSize}, DataType::kFP8);
    }
#endif

    // Request another alignment size to align the workspace pointer.
    workspaceSize += kDEVICE_ALIGNMENT;

    LOG_DEBUG("AttentionPlugin workspace size: %zu bytes", workspaceSize);
    return workspaceSize;
}

int32_t AttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
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
    check::check(kInputDesc.dims.d[0] == runtimeBatchSize && vInputDesc.dims.d[0] == runtimeBatchSize,
        "Batch size must be consistent across Q/K/V inputs.");
    check::check(kInputDesc.dims.d[1] == runtimeSeqLen && vInputDesc.dims.d[1] == runtimeSeqLen,
        "Sequence length must be consistent across Q/K/V inputs.");
    check::check(qInputDesc.dims.d[2] == mNumQHeads * mHeadSize, "Q input shape shall be consistent.");
    check::check(kInputDesc.dims.d[2] == mNumKVHeads * mHeadSize, "K input shape shall be consistent.");
    check::check(vInputDesc.dims.d[2] == mNumKVHeads * mHeadSize, "V input shape shall be consistent.");

    rt::Tensor qInputTensor(const_cast<void*>(inputs[kIN_Q_IDX]),
        rt::Coords{runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, rt::DeviceType::kGPU, qInputDesc.type);
    rt::Tensor kInputTensor(const_cast<void*>(inputs[kIN_K_IDX]),
        rt::Coords{runtimeBatchSize, runtimeSeqLen, mNumKVHeads, mHeadSize}, rt::DeviceType::kGPU, kInputDesc.type);
    rt::Tensor vInputTensor(const_cast<void*>(inputs[kIN_V_IDX]),
        rt::Coords{runtimeBatchSize, runtimeSeqLen, mNumKVHeads, mHeadSize}, rt::DeviceType::kGPU, vInputDesc.type);

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

    // Construct the KVCache tensor from the input KV cache descriptor.
    // This allows KV cache from 0 to maxSeqLen and helps adjust the profile at runtime.
    PluginTensorDesc const& kvCacheInputDesc = inputDesc[kIN_KV_CACHE_IDX];
    rt::Tensor kvCacheTensor(
        outputs[kOUT_KV_CACHE_IDX], rt::Coords{kvCacheInputDesc.dims}, rt::DeviceType::kGPU, kvCacheInputDesc.type);

    // Extract KV cache capacity from the runtime tensor shape.
    int32_t const kvCacheCapacity = static_cast<int32_t>(kvCacheInputDesc.dims.d[3]);

    // Optional Inputs that are not used with Tree Attention enabled.
    rt::Tensor attentionMaskTensor{};
    rt::Tensor attentionPosIdTensor{};
    if (mEnableTreeAttention)
    {
        PluginTensorDesc const& attentionMaskInputDesc = inputDesc[kIN_OPTIONAL_ATTN_MASK_IDX];
        PluginTensorDesc const& attentionPosIdInputDesc = inputDesc[kIN_OPTIONAL_ATTN_POS_ID_IDX];
        attentionMaskTensor = rt::Tensor(const_cast<void*>(inputs[kIN_OPTIONAL_ATTN_MASK_IDX]),
            rt::Coords{attentionMaskInputDesc.dims}, rt::DeviceType::kGPU, attentionMaskInputDesc.type);
        attentionPosIdTensor = rt::Tensor(const_cast<void*>(inputs[kIN_OPTIONAL_ATTN_POS_ID_IDX]),
            rt::Coords{attentionPosIdInputDesc.dims}, rt::DeviceType::kGPU, attentionPosIdInputDesc.type);
    }

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

    auto const nbInputs = kNUM_REQUIRED_INPUTS + (mEnableTreeAttention ? kNUM_TREE_ATTN_OPTIONAL_INPUTS : 0);
    auto const nbOutputs = kNUM_REQUIRED_OUTPUTS;
    size_t space = getWorkspaceSize(inputDesc, nbInputs, outputDesc, nbOutputs);
    // Align the workspace pointer so that each tensor assigned from the workspace will align to the device alignment
    // granularity.
    std::byte* alignedWorkspacePtr
        = static_cast<std::byte*>(std::align(kDEVICE_ALIGNMENT, space - kDEVICE_ALIGNMENT, workspace, space));
    if (alignedWorkspacePtr == nullptr)
    {
        LOG_ERROR("Workspace size is too small to hold all data structures with correct alignment");
        return 1;
    }

    if (executionMode == AttentionExecutionMode::kNORMAL_PREFILL
        || executionMode == AttentionExecutionMode::kCHUNKED_PREFILL)
    {
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

        auto const fmhaMaskType = static_cast<ContextAttentionMaskType>(mContextAttentionMaskType);

#ifdef CUTE_DSL_FMHA_ENABLED
        // Enable CuteDSL FMHA for single batch causal prefill usecase when FP8 KVCache is disabled.
        // TODO: Enable multi-batch prefill and FP8 KVCache after we improve the kernel implementation.
        bool const enableCuteDslFMHA = fmhaMaskType == ContextAttentionMaskType::CAUSAL && mUseCuteDslFMHA
            && !mEnableFp8KVCache && runtimeBatchSize == 1;
        if (enableCuteDslFMHA)
        {
            float const qScale = mQkvScales[0];
            int32_t const slidingWindow = mSlidingWindowSize > 0 ? mSlidingWindowSize : INT_MAX;

            CuteDslFMHARunner runner(
                mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);

            if (mEnableFp8KVCache)
            {
                // FP8 Q workspace: RoPE kernel quantizes roped Q to FP8 using calibrated qScale.
                rt::Tensor fp8QTensor = assignTensorFromWorkspace(
                    alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kFP8);

                // Single kernel: RoPE Q → FP8 output, RoPE K + write FP8 K/V to cache.
                kernel::launchApplyRopeWriteKVSplitQKV(ropeCosSinTensor, kvCacheEndIdxsTensor, qInputTensor,
                    kInputTensor, vInputTensor, kvCacheTensor, kScale, vScale, stream, fp8QTensor.rawPointer(), qScale);

                runner.run(fp8QTensor.rawPointer(),                 // Q  [b, s_q, h_q, d] FP8
                    kvCacheTensor.rawPointer(),                     // KV [b, 2, h_k, cap, d] FP8
                    attentionOutputTensor.dataPointer<half>(),      // O  [b, s_q, h_q, d] FP16
                    paddedCuKVSeqLensTensor.dataPointer<int32_t>(), // cu_kv_seqlens [b+1]
                    stream, slidingWindow, /*fp8Input=*/true, qScale, kScale, vScale);
            }
            else
            {
                // FP16 path: RoPE Q in-place, write FP16 K/V to cache.
                kernel::launchApplyRopeWriteKVSplitQKV(ropeCosSinTensor, kvCacheEndIdxsTensor, qInputTensor,
                    kInputTensor, vInputTensor, kvCacheTensor, kScale, vScale, stream);

                runner.run(qInputTensor.dataPointer<half>(),        // Q  [b, s_q, h_q, d]
                    kvCacheTensor.dataPointer<half>(),              // KV [b, 2, h_k, cap, d]
                    attentionOutputTensor.dataPointer<half>(),      // O  [b, s_q, h_q, d]
                    paddedCuKVSeqLensTensor.dataPointer<int32_t>(), // cu_kv_seqlens [b+1]
                    stream, slidingWindow);
            }
        }
        else
#endif
        {
            auto fmhaRunner = ContextFMHARunner(mDataType, runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads,
                mHeadSize, mSMVersion, AttentionInputLayout::SEPARATE_Q_K_V, fmhaMaskType);

            // Prepare FMHA_v2 params to launch FMHA kernel
            FusedMultiheadAttentionParamsV2 params{};
            fmhaRunner.setupParams(params);
            params.cu_q_seqlens = cuQSeqLensTensor.dataPointer<int32_t>();

            if (executionMode == AttentionExecutionMode::kCHUNKED_PREFILL)
            {
                // kvCache: [b, 2, hkv, s, d] -> split K [b, s, hkv, d] + V [b, s, hkv, d]
                kernel::launchApplyRopeWriteKV(ropeCosSinTensor, kvCacheEndIdxsTensor, qInputTensor, kInputTensor,
                    vInputTensor, kvCacheTensor, kScale, vScale, stream, false);

                // Allocate a single workspace and split into K and V halves by pointer arithmetic.
                rt::Tensor kvWorkspaceTensor = assignTensorFromWorkspace(alignedWorkspacePtr,
                    {runtimeBatchSize, 2, mNumKVHeads, kvCacheCapacity, mHeadSize}, DataType::kHALF);
                size_t const halfSize
                    = static_cast<size_t>(runtimeBatchSize) * kvCacheCapacity * mNumKVHeads * mHeadSize;
                half* kvWorkspacePtr = kvWorkspaceTensor.dataPointer<half>();
                rt::Tensor kWorkspaceTensor(kvWorkspacePtr,
                    rt::Coords{runtimeBatchSize, kvCacheCapacity, mNumKVHeads, mHeadSize}, rt::DeviceType::kGPU,
                    DataType::kHALF);
                rt::Tensor vWorkspaceTensor(kvWorkspacePtr + halfSize,
                    rt::Coords{runtimeBatchSize, kvCacheCapacity, mNumKVHeads, mHeadSize}, rt::DeviceType::kGPU,
                    DataType::kHALF);
                // seqLen == 0: copy the full KV-cache capacity (matches the pre-seqLen kernel behavior).
                kernel::cvtKVLayoutBHSDToSplitKV(
                    kvCacheTensor, kWorkspaceTensor, vWorkspaceTensor, rt::Tensor{}, /*seqLen=*/0, stream);

                // Set device ptr for FMHA kernel.
                params.s_kv = kvCacheCapacity;
                params.q_ptr = qInputTensor.dataPointer<half>();
                params.k_ptr = kWorkspaceTensor.dataPointer<half>();
                params.v_ptr = vWorkspaceTensor.dataPointer<half>();
                params.cu_kv_seqlens = cuKVSeqLensTensor.dataPointer<int32_t>();
                params.o_ptr = attentionOutputTensor.dataPointer<half>();
            }
            else
            { // SEPARATE_Q_K_V
                kernel::launchApplyRopeWriteKV(ropeCosSinTensor, std::nullopt, qInputTensor, kInputTensor, vInputTensor,
                    kvCacheTensor, kScale, vScale, stream, true);

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
    }
    else
    {
        // Prepare Decoding attention runner parameter to dispatch kernel
        if (executionMode == AttentionExecutionMode::kTREE_DECODING)
        {
            // Execute tree attention decoding.
            kernel::launchApplyRopeWriteKVTreeDecoding(ropeCosSinTensor, contextLengthTensor, attentionPosIdTensor,
                qInputTensor, kInputTensor, vInputTensor, kvCacheTensor, kScale, vScale, stream);
        }
        else
        {
            // Execute vanilla decoding.
            kernel::launchApplyRopeWriteKV(ropeCosSinTensor, contextLengthTensor, qInputTensor, kInputTensor,
                vInputTensor, kvCacheTensor, kScale, vScale, stream, false);
        }

        auto xqaRunner = DecoderXQARunner(mDataType, selectKvCacheDataType(mEnableFp8KVCache), runtimeBatchSize,
            mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion);
        XQALaunchParams params = xqaRunner.initXQAParams();
        if (mEnableFp8KVCache)
        {
            params.kScale = kScale;
            params.vScale = vScale;
        }
        params.output = attentionOutputTensor.dataPointer<half>();
        params.qInputPtr = qInputTensor.dataPointer<half>();
        params.kvCache.data = kvCacheTensor.rawPointer();
        params.kvCache.sequence_lengths = contextLengthTensor.dataPointer<int32_t>();
        params.kvCache.capacity = kvCacheCapacity;
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

size_t AttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumQHeads) + sizeof(mNumKVHeads) + sizeof(mHeadSize) + sizeof(mEnableTreeAttention)
        + sizeof(mEnableFp8KVCache) + sizeof(int32_t) /* slidingWindowSize */
        + sizeof(int32_t) /* qkv scale count */ + mQkvScales.size() * sizeof(float)
        + sizeof(int32_t) /* context mask serialization tag */ + sizeof(mContextAttentionMaskType);
}

void AttentionPlugin::serialize(void* buffer) const noexcept
{
    std::byte* byteBuffer = static_cast<std::byte*>(buffer);
    serializeValue(&byteBuffer, mNumQHeads);
    serializeValue(&byteBuffer, mNumKVHeads);
    serializeValue(&byteBuffer, mHeadSize);
    serializeValue(&byteBuffer, mEnableTreeAttention);
    serializeValue(&byteBuffer, mEnableFp8KVCache);
    serializeValue(&byteBuffer, mSlidingWindowSize);
    int32_t const qkvScaleCount = static_cast<int32_t>(mQkvScales.size());
    serializeValue(&byteBuffer, qkvScaleCount);
    for (auto const& s : mQkvScales)
    {
        serializeValue(&byteBuffer, s);
    }
    serializeValue(&byteBuffer, kContextMaskSerializationTag);
    serializeValue(&byteBuffer, mContextAttentionMaskType);
}

int32_t AttentionPlugin::initialize() noexcept
{
    return 0;
}

void AttentionPlugin::terminate() noexcept {}

void AttentionPlugin::destroy() noexcept
{
    delete this;
}

AttentionPluginCreator::AttentionPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_q_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_kv_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, 1));
    // Make enable_fp8_kv_cache optional with default value 0 (disable by default)
    mPluginAttributes.emplace_back(PluginField("enable_tree_attention", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("enable_fp8_kv_cache", nullptr, PluginFieldType::kINT32, 0));
    // Sliding window size (-1 = no sliding window, >0 = window size)
    mPluginAttributes.emplace_back(PluginField("sliding_window_size", nullptr, PluginFieldType::kINT32, 0));
    // Optional context attention mask type (ContextAttentionMaskType: 0 = PADDING, 1 = CAUSAL,
    // 2 = SLIDING_OR_CHUNKED_CAUSAL, 3 = CUSTOM_MASK). Defaults to CAUSAL when unset.
    mPluginAttributes.emplace_back(PluginField("context_attention_mask_type", nullptr, PluginFieldType::kINT32, 0));
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

nvinfer1::PluginFieldCollection const* AttentionPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void AttentionPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* AttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* AttentionPluginCreator::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

nvinfer1::IPluginV2* AttentionPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    try
    {
        std::optional<int32_t> numQHeads = parsePluginScalarField<int32_t>("num_q_heads", fc);
        std::optional<int32_t> numKVHeads = parsePluginScalarField<int32_t>("num_kv_heads", fc);
        std::optional<int32_t> headSize = parsePluginScalarField<int32_t>("head_size", fc);
        std::optional<int32_t> enableTreeAttention = parsePluginScalarField<int32_t>("enable_tree_attention", fc);
        std::optional<int32_t> enableFp8KVCache = parsePluginScalarField<int32_t>("enable_fp8_kv_cache", fc);
        // sliding_window_size: -1 = no sliding window (default), >0 = sliding window size
        int32_t slidingWindowSize = parsePluginScalarField<int32_t>("sliding_window_size", fc).value_or(-1);
        // Make enable_fp8_kv_cache optional with default value 0 (disable by default)
        int32_t enableFp8KVCacheValue = enableFp8KVCache.value_or(0);
        int32_t contextAttentionMaskType = parsePluginScalarField<int32_t>("context_attention_mask_type", fc)
                                               .value_or(static_cast<int32_t>(ContextAttentionMaskType::CAUSAL));
        if (!parsePluginScalarField<int32_t>("context_attention_mask_type", fc).has_value())
        {
            if (auto legacyBidirectionalPrefill = parsePluginScalarField<int32_t>("enable_bidirectional_prefill", fc))
            {
                contextAttentionMaskType = (legacyBidirectionalPrefill.value() != 0)
                    ? static_cast<int32_t>(ContextAttentionMaskType::PADDING)
                    : static_cast<int32_t>(ContextAttentionMaskType::CAUSAL);
            }
        }
        if (contextAttentionMaskType < static_cast<int32_t>(ContextAttentionMaskType::PADDING)
            || contextAttentionMaskType > static_cast<int32_t>(ContextAttentionMaskType::CUSTOM_MASK))
        {
            LOG_ERROR("Invalid context_attention_mask_type %d (expected 0-3).", contextAttentionMaskType);
            return nullptr;
        }

        bool checkRequiredFields = numQHeads.has_value() && headSize.has_value() && numKVHeads.has_value()
            && enableTreeAttention.has_value();
        if (!checkRequiredFields)
        {
            LOG_ERROR("Missing required AttentionPlugin fields.");
            return nullptr;
        }

        std::vector<float> qkvScales;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            if (std::string("qkv_scales") == fc->fields[i].name)
            {
                auto const* data = static_cast<float const*>(fc->fields[i].data);
                qkvScales.assign(data, data + fc->fields[i].length);
                break;
            }
        }

        if (enableFp8KVCacheValue && qkvScales.size() != 3)
        {
            LOG_ERROR(
                "FP8 KV cache enabled but qkv_scales has %zu elements (expected 3). "
                "Re-export the model to include QKV scales [q, k, v].",
                qkvScales.size());
            return nullptr;
        }

        AttentionPlugin* plugin = new AttentionPlugin(std::string(name), numQHeads.value(), numKVHeads.value(),
            headSize.value(), enableTreeAttention.value(), enableFp8KVCacheValue, slidingWindowSize, qkvScales,
            contextAttentionMaskType);

        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create AttentionPlugin: %s", e.what());
    }
    return nullptr;
}

nvinfer1::IPluginV2* AttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new AttentionPlugin(name, static_cast<std::byte const*>(serialData), serialLength);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to deserialize AttentionPlugin: %s", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
