/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInferRuntime.h>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

constexpr char const* kVIT_ATTENTION_PLUGIN_NAME{"ViTAttentionPlugin"};
constexpr char const* kVIT_ATTENTION_PLUGIN_VERSION{"1"};

class ViTAttentionPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    ViTAttentionPlugin(
        std::string const& name, int32_t numHeads, int32_t headSize, int32_t qkvFused, int32_t maskType,
        int32_t maxSeqLen, int32_t maskBlockSize);
    ViTAttentionPlugin(std::string const& name, void const* data, size_t length);

    ViTAttentionPlugin() = delete;
    ViTAttentionPlugin(ViTAttentionPlugin const&) = delete;
    ~ViTAttentionPlugin() override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex,
        nvinfer1::DimsExprs const* inputs,
        int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos,
        nvinfer1::PluginTensorDesc const* inOut,
        int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    void configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in,
        int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(
        nvinfer1::PluginTensorDesc const* inputs,
        int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;
    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    char const* getPluginType() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;

protected:
    std::string mLayerName;
    std::string mNamespace;
    int32_t mNumHeads{};
    int32_t mHeadSize{};
    int32_t mQKVFused{};
    int32_t mMaskType{};
    int32_t mMaxSeqLen{};
    int32_t mMaskBlockSize{};
    nvinfer1::DataType mDataType{nvinfer1::DataType::kFLOAT};
};

class ViTAttentionPluginCreator : public nvinfer1::IPluginCreator
{
public:
    ViTAttentionPluginCreator();
    ~ViTAttentionPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::IPluginV2* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
