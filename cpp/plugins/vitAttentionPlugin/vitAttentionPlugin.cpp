/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vitAttentionPlugin.h"

#include "common/logger.h"
#include "kernels/vitAttentionKernels/vitAttentionRunner.h"
#include "plugins/utils/pluginUtils.h"

#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>
#include <exception>
#include <mutex>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kPLUGIN_VERSION{kVIT_ATTENTION_PLUGIN_VERSION};
constexpr char const* kPLUGIN_NAME{kVIT_ATTENTION_PLUGIN_NAME};

constexpr int32_t kDENSE_ADDITIVE_MASK_TYPE{static_cast<int32_t>(ViTAttentionMaskType::kDenseAdditive)};
constexpr int32_t kPACKED_CU_SEQLENS_MASK_TYPE{static_cast<int32_t>(ViTAttentionMaskType::kPackedCuSeqLens)};
constexpr int32_t kCOMPACT_BLOCK_MASK_TYPE{static_cast<int32_t>(ViTAttentionMaskType::kCompactBlock)};

constexpr int32_t kIN_QKV_IDX{0};
constexpr int32_t kIN_ROPE_COS_IDX{1};
constexpr int32_t kIN_ROPE_SIN_IDX{2};
constexpr int32_t kIN_ATTENTION_MASK_IDX{3};
constexpr int32_t kOUT_ATTENTION_IDX{0};

constexpr int32_t kNUM_INPUTS{4};
constexpr int32_t kNUM_OUTPUTS{1};

bool isDebugEnabled()
{
    static bool initialized = false;
    static bool enabled = false;
    if (!initialized)
    {
        char const* env = std::getenv("TRT_EDGELLM_DEBUG_PLUGIN");
        enabled = (env != nullptr && (std::string(env) == "1" || std::string(env) == "true"));
        initialized = true;
        if (enabled)
        {
            std::printf("[ViTAttentionPlugin] Debug logging enabled\n");
        }
    }
    return enabled;
}

#define PLUGIN_DEBUG_LOG(...)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (isDebugEnabled())                                                                                          \
        {                                                                                                              \
            std::printf("[ViTAttentionPlugin][%s:%d] ", __FUNCTION__, __LINE__);                                     \
            std::printf(__VA_ARGS__);                                                                                  \
            std::printf("\n");                                                                                       \
            std::fflush(stdout);                                                                                       \
        }                                                                                                              \
    } while (0)

} // namespace

ViTAttentionPlugin::ViTAttentionPlugin(
    std::string const& name, int32_t numHeads, int32_t headSize, int32_t qkvFused, int32_t maskType,
    int32_t maxSeqLen, int32_t maskBlockSize)
    : mLayerName(name)
    , mNumHeads(numHeads)
    , mHeadSize(headSize)
    , mQKVFused(qkvFused)
    , mMaskType(maskType)
    , mMaxSeqLen(maxSeqLen)
    , mMaskBlockSize(maskBlockSize)
{
}

ViTAttentionPlugin::ViTAttentionPlugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{
    deserializeValue(&data, &length, &mNumHeads);
    deserializeValue(&data, &length, &mHeadSize);
    deserializeValue(&data, &length, &mQKVFused);
    deserializeValue(&data, &length, &mMaskType);
    deserializeValue(&data, &length, &mMaxSeqLen);
    if (length >= sizeof(mMaskBlockSize))
    {
        deserializeValue(&data, &length, &mMaskBlockSize);
    }
}

ViTAttentionPlugin::~ViTAttentionPlugin() {}

nvinfer1::IPluginV2DynamicExt* ViTAttentionPlugin::clone() const noexcept
{
    ViTAttentionPlugin* plugin
        = new ViTAttentionPlugin(mLayerName, mNumHeads, mHeadSize, mQKVFused, mMaskType, mMaxSeqLen, mMaskBlockSize);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

int32_t ViTAttentionPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DataType ViTAttentionPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

nvinfer1::DimsExprs ViTAttentionPlugin::getOutputDimensions(
    int32_t outputIndex,
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs output(inputs[0]);
    output.d[2] = exprBuilder.constant(mNumHeads * mHeadSize);
    return output;
}

bool ViTAttentionPlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::PluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept
{
    // Support ViT attention inputs:
    //      QKV tensor (linear FP16/FP32 dense, linear FP16 packed-cu-seqlens) with shape [B, S, 3 * H * D]
    //      RoPE cos/sin tensors, matching the QKV type.
    //      Dense additive mask matching the QKV type, or packed cu_seqlens with INT32 type.
    //
    // Support ViT attention outputs:
    //      attention result, matching the QKV type, with shape [B, S, H * D].
    auto checkQKV = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        if (mMaskType == kPACKED_CU_SEQLENS_MASK_TYPE)
        {
            status &= tensorDesc.type == DataType::kHALF;
        }
        else
        {
            status &= tensorDesc.type == DataType::kFLOAT || tensorDesc.type == DataType::kHALF;
        }
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        return status;
    };

    auto checkRopeCosSin = [&inOut](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == inOut[kIN_QKV_IDX].type;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        return status;
    };

    auto checkAttentionMask = [this, &inOut](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        if (mMaskType == kPACKED_CU_SEQLENS_MASK_TYPE)
        {
            status &= tensorDesc.type == DataType::kINT32;
        }
        else if (mMaskType == kCOMPACT_BLOCK_MASK_TYPE)
        {
            status &= tensorDesc.type == DataType::kINT32;
        }
        else
        {
            status &= tensorDesc.type == inOut[kIN_QKV_IDX].type;
        }
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        return status;
    };

    auto checkAttentionOutput = [&inOut](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == inOut[kIN_QKV_IDX].type;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        return status;
    };

    if (nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS)
    {
        return false;
    }

    bool result{true};

    if (pos < nbInputs)
    {
        switch (pos)
        {
        case kIN_QKV_IDX: result = checkQKV(inOut[pos]); break;
        case kIN_ROPE_COS_IDX:
        case kIN_ROPE_SIN_IDX: result = checkRopeCosSin(inOut[pos]); break;
        case kIN_ATTENTION_MASK_IDX: result = checkAttentionMask(inOut[pos]); break;
        default: result = false; break;
        }
    }
    else
    {
        int32_t const outPos = pos - nbInputs;
        switch (outPos)
        {
        case kOUT_ATTENTION_IDX: result = checkAttentionOutput(inOut[pos]); break;
        default: result = false; break;
        }
    }

    return result;
}

void ViTAttentionPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out,
    int32_t nbOutputs) noexcept
{
    mDataType = in[0].desc.type;
}

size_t ViTAttentionPlugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept
{
    PluginTensorDesc const& qkvInputDesc = inputs[0];
    int32_t const runtimeBatchSize = static_cast<int32_t>(qkvInputDesc.dims.d[0]);
    int32_t const runtimeSeqLen = static_cast<int32_t>(qkvInputDesc.dims.d[1]);
    return ViTAttentionRunner::getWorkspaceSize(
        qkvInputDesc.type, runtimeBatchSize, runtimeSeqLen, mNumHeads, mHeadSize,
        static_cast<ViTAttentionMaskType>(mMaskType));
}

int32_t ViTAttentionPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept
{
    try
    {
        PluginTensorDesc const& qkvInputDesc = inputDesc[0];
        int32_t const runtimeBatchSize = static_cast<int32_t>(qkvInputDesc.dims.d[0]);
        int32_t const runtimeSeqLen = static_cast<int32_t>(qkvInputDesc.dims.d[1]);
        int32_t const hiddenSize = mNumHeads * mHeadSize;

        if (mQKVFused != 1)
        {
            LOG_ERROR("ViTAttentionPlugin only supports fused QKV input currently.");
            return 1;
        }
        if (runtimeBatchSize <= 0 || runtimeSeqLen <= 0)
        {
            LOG_ERROR("Invalid ViTAttentionPlugin runtime shape.");
            return 1;
        }
        if (qkvInputDesc.dims.d[2] != 3 * hiddenSize)
        {
            LOG_ERROR("ViTAttentionPlugin QKV input shape is inconsistent with plugin fields.");
            return 1;
        }
        if (!ViTAttentionRunner::canImplement(mDataType, mNumHeads, mHeadSize))
        {
            LOG_ERROR("Unsupported ViTAttentionPlugin configuration.");
            return 1;
        }
        if (mMaskType != kDENSE_ADDITIVE_MASK_TYPE && mMaskType != kPACKED_CU_SEQLENS_MASK_TYPE
            && mMaskType != kCOMPACT_BLOCK_MASK_TYPE)
        {
            LOG_ERROR("Unsupported ViTAttentionPlugin mask_type.");
            return 1;
        }

        if (inputDesc[1].dims.nbDims != 2 || inputDesc[1].dims.d[0] != runtimeSeqLen
            || inputDesc[1].dims.d[1] != mHeadSize)
        {
            LOG_ERROR("ViTAttentionPlugin RoPE cosine input shape must be [S, head_size].");
            return 1;
        }
        if (inputDesc[2].dims.nbDims != 2 || inputDesc[2].dims.d[0] != runtimeSeqLen
            || inputDesc[2].dims.d[1] != mHeadSize)
        {
            LOG_ERROR("ViTAttentionPlugin RoPE sine input shape must be [S, head_size].");
            return 1;
        }
        int32_t maskRows{};
        if (mMaskType == kDENSE_ADDITIVE_MASK_TYPE)
        {
            if (inputDesc[3].dims.nbDims != 3 || inputDesc[3].dims.d[1] != runtimeSeqLen
                || inputDesc[3].dims.d[2] != runtimeSeqLen)
            {
                LOG_ERROR("ViTAttentionPlugin attention mask input shape must be [1|B|B*H, S, S].");
                return 1;
            }
            maskRows = static_cast<int32_t>(inputDesc[3].dims.d[0]);
        }
        else if (mMaskType == kPACKED_CU_SEQLENS_MASK_TYPE)
        {
            if (inputDesc[3].type != DataType::kINT32 || inputDesc[3].dims.nbDims != 1 || inputDesc[3].dims.d[0] < 2)
            {
                LOG_ERROR("ViTAttentionPlugin cu_seqlens input shape must be [num_segments + 1] with INT32 type.");
                return 1;
            }
            if (!ViTAttentionRunner::canImplementFMHA(mDataType, mHeadSize))
            {
                LOG_ERROR("ViTAttentionPlugin cu_seqlens mode requires FP16 with head_size 64 or 128.");
                return 1;
            }
            maskRows = static_cast<int32_t>(inputDesc[3].dims.d[0]);
        }
        else
        {
            if (inputDesc[3].type != DataType::kINT32 || inputDesc[3].dims.nbDims != 2 || inputDesc[3].dims.d[0] < 1
                || inputDesc[3].dims.d[1] < 1)
            {
                LOG_ERROR("ViTAttentionPlugin compact block mask input shape must be [1|B|B*H, num_blocks] INT32.");
                return 1;
            }
            if (mMaskBlockSize <= 0 || inputDesc[3].dims.d[1] * mMaskBlockSize != runtimeSeqLen)
            {
                LOG_ERROR("ViTAttentionPlugin compact block mask requires num_blocks * mask_block_size == S.");
                return 1;
            }
            maskRows = static_cast<int32_t>(inputDesc[3].dims.d[0]);
        }

        int32_t const maxSeqLen = mMaxSeqLen > 0 ? mMaxSeqLen : runtimeSeqLen;
        if (mMaskType == kPACKED_CU_SEQLENS_MASK_TYPE && maxSeqLen > runtimeSeqLen)
        {
            LOG_ERROR("ViTAttentionPlugin max_seq_len cannot exceed runtime sequence length.");
            return 1;
        }

        PLUGIN_DEBUG_LOG("dispatching ViT attention kernel: B=%d, S=%d, maxS=%d, H=%d, D=%d, mask_type=%d",
            runtimeBatchSize, runtimeSeqLen, maxSeqLen, mNumHeads, mHeadSize, mMaskType);

        ViTAttentionRunner runner(mDataType, runtimeBatchSize, runtimeSeqLen, maxSeqLen, mNumHeads, mHeadSize, maskRows,
            mMaskBlockSize, static_cast<ViTAttentionMaskType>(mMaskType));
        runner.dispatch(inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], workspace, stream);
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("ViTAttentionPlugin enqueue failed: %s", e.what());
        return 1;
    }
}

size_t ViTAttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(mQKVFused) + sizeof(mMaskType) + sizeof(mMaxSeqLen)
        + sizeof(mMaskBlockSize);
}

void ViTAttentionPlugin::serialize(void* buffer) const noexcept
{
    serializeValue(&buffer, mNumHeads);
    serializeValue(&buffer, mHeadSize);
    serializeValue(&buffer, mQKVFused);
    serializeValue(&buffer, mMaskType);
    serializeValue(&buffer, mMaxSeqLen);
    serializeValue(&buffer, mMaskBlockSize);
}

char const* ViTAttentionPlugin::getPluginType() const noexcept
{
    return kPLUGIN_NAME;
}

char const* ViTAttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void ViTAttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

char const* ViTAttentionPlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

int32_t ViTAttentionPlugin::initialize() noexcept
{
    return 0;
}

void ViTAttentionPlugin::terminate() noexcept
{
}

void ViTAttentionPlugin::destroy() noexcept
{
    delete this;
}

nvinfer1::PluginFieldCollection ViTAttentionPluginCreator::mFieldCollection{};
std::vector<nvinfer1::PluginField> ViTAttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(ViTAttentionPluginCreator);

ViTAttentionPluginCreator::ViTAttentionPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("qkv_fused", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("mask_type", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_seq_len", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("mask_block_size", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* ViTAttentionPluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* ViTAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void ViTAttentionPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

char const* ViTAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* ViTAttentionPluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

nvinfer1::IPluginV2* ViTAttentionPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    try
    {
        std::optional<int32_t> numHeads = parsePluginScalarField<int32_t>("num_heads", fc);
        std::optional<int32_t> headSize = parsePluginScalarField<int32_t>("head_size", fc);
        std::optional<int32_t> qkvFused = parsePluginScalarField<int32_t>("qkv_fused", fc);
        std::optional<int32_t> maskType = parsePluginScalarField<int32_t>("mask_type", fc);
        std::optional<int32_t> maxSeqLen = parsePluginScalarField<int32_t>("max_seq_len", fc);
        std::optional<int32_t> maskBlockSize = parsePluginScalarField<int32_t>("mask_block_size", fc);

        if (!numHeads.has_value() || !headSize.has_value())
        {
            return nullptr;
        }

        int32_t qkvFusedValue = qkvFused.value_or(1);
        int32_t maskTypeValue = maskType.value_or(kDENSE_ADDITIVE_MASK_TYPE);
        int32_t maxSeqLenValue = maxSeqLen.value_or(0);
        int32_t maskBlockSizeValue = maskBlockSize.value_or(0);
        return new ViTAttentionPlugin(
            std::string(name), numHeads.value(), headSize.value(), qkvFusedValue, maskTypeValue, maxSeqLenValue,
            maskBlockSizeValue);
    }
    catch (std::exception const&)
    {
        return nullptr;
    }
}

nvinfer1::IPluginV2* ViTAttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new ViTAttentionPlugin(name, serialData, serialLength);
    }
    catch (std::exception const&)
    {
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
