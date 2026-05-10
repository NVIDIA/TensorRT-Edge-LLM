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

ViTAttentionPlugin::ViTAttentionPlugin(std::string const& name, int32_t numHeads, int32_t headSize, int32_t qkvFused)
    : mLayerName(name)
    , mNumHeads(numHeads)
    , mHeadSize(headSize)
    , mQKVFused(qkvFused)
{
}

ViTAttentionPlugin::ViTAttentionPlugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{
    deserializeValue(&data, &length, &mNumHeads);
    deserializeValue(&data, &length, &mHeadSize);
    deserializeValue(&data, &length, &mQKVFused);
}

ViTAttentionPlugin::~ViTAttentionPlugin() {}

nvinfer1::IPluginV2DynamicExt* ViTAttentionPlugin::clone() const noexcept
{
    ViTAttentionPlugin* plugin = new ViTAttentionPlugin(mLayerName, mNumHeads, mHeadSize, mQKVFused);
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
    auto const& desc = inOut[pos];
    if (desc.format != nvinfer1::TensorFormat::kLINEAR)
    {
        return false;
    }

    if (desc.type != nvinfer1::DataType::kFLOAT && desc.type != nvinfer1::DataType::kHALF)
    {
        return false;
    }

    if (pos == 0)
    {
        return true;
    }

    if (pos < nbInputs)
    {
        return inOut[0].type == desc.type;
    }

    return inOut[0].type == desc.type;
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
    return ViTAttentionRunner::getWorkspaceSize(runtimeBatchSize, runtimeSeqLen, mNumHeads, mHeadSize);
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
        if (inputDesc[3].dims.nbDims != 3 || inputDesc[3].dims.d[1] != runtimeSeqLen
            || inputDesc[3].dims.d[2] != runtimeSeqLen)
        {
            LOG_ERROR("ViTAttentionPlugin attention mask input shape must be [1|B|B*H, S, S].");
            return 1;
        }

        PLUGIN_DEBUG_LOG("dispatching ViT attention kernel: B=%d, S=%d, H=%d, D=%d", runtimeBatchSize, runtimeSeqLen,
            mNumHeads, mHeadSize);

        int32_t const maskRows = static_cast<int32_t>(inputDesc[3].dims.d[0]);
        ViTAttentionRunner runner(mDataType, runtimeBatchSize, runtimeSeqLen, mNumHeads, mHeadSize, maskRows);
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
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(mQKVFused);
}

void ViTAttentionPlugin::serialize(void* buffer) const noexcept
{
    serializeValue(&buffer, mNumHeads);
    serializeValue(&buffer, mHeadSize);
    serializeValue(&buffer, mQKVFused);
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

        if (!numHeads.has_value() || !headSize.has_value())
        {
            return nullptr;
        }

        int32_t qkvFusedValue = qkvFused.value_or(1);
        return new ViTAttentionPlugin(std::string(name), numHeads.value(), headSize.value(), qkvFusedValue);
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
