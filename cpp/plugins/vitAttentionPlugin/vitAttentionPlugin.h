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

#pragma once

#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

//! Internal ViT attention implementation selected from build artifacts and runtime capabilities.
enum class ViTFMHABackend
{
    kNONE,
    kCUTE_DSL_FMHA_BLACKWELL,
    kCUTE_DSL_FMHA_V2
};

//! \brief TensorRT plugin for ViT attention operations (V3 — IPluginV3).
//!
//! This plugin implements efficient attention mechanisms for ViT.
class ViTAttentionPlugin : public nvinfer1::IPluginV3,
                           public nvinfer1::IPluginV3OneCore,
                           public nvinfer1::IPluginV3OneBuild,
                           public nvinfer1::IPluginV3OneRuntime
{
public:
    //! \brief Constructor for attention plugin with configuration parameters
    //! \param[in] name Plugin instance name
    //! \param[in] numHeads Number of attention heads
    //! \param[in] headSize Head dimension size
    //! \param[in] attentionScale Optional absolute QK^T multiplier; defaults to 1/sqrt(headSize)
    //! \param[in] qkvScales Per-tensor dequant scales ``[qScale, kScale, vScale]`` (FP8 → original)
    //!                      folded into the FMHA softmax/output scales.
    ViTAttentionPlugin(std::string const& name, int32_t numHeads, int32_t headSize,
        std::optional<float> attentionScale = std::nullopt,
        std::vector<float> const& qkvScales = std::vector<float>{1.f, 1.f, 1.f});
    ViTAttentionPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    ViTAttentionPlugin() = delete;
    ViTAttentionPlugin(ViTAttentionPlugin const&) = delete;
    ~ViTAttentionPlugin() override;

    // IPluginV3
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // IPluginV3OneCore
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV3OneBuild
    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    // IPluginV3OneRuntime
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

protected:
    std::string mLayerName; //!< Plugin layer name
    std::string mNamespace; //!< Plugin namespace

    int32_t mNumHeads{};     //!< Number of attention heads
    int32_t mHeadSize{};     //!< Number of elements per head (head dimension)
    float mAttentionScale{}; //!< Absolute QK^T multiplier.

    //! Datatype of attention. Only supports FP16 as of now.
    nvinfer1::DataType const mDataType{nvinfer1::DataType::kHALF};
    int32_t mSMVersion; //!< CUDA SM version
    ViTFMHABackend mFMHABackend{ViTFMHABackend::kNONE};

    //! The actual FP8 mode is chosen per-build by the Q/K/V tensor dtype the graph supplies.
    bool supportsFp8Mha() const noexcept
    {
        return mFMHABackend == ViTFMHABackend::kCUTE_DSL_FMHA_BLACKWELL
            && (mHeadSize == 64 || mHeadSize == 80 || mHeadSize == 96 || mHeadSize == 128);
    }

    //! Host QKV dequant scales [q, k, v] (quant→orig).
    //! - q scale: folded into softmaxScale for FP8 FMHA.
    //! - k scale: folded into softmaxScale for FP8 FMHA.
    //! - v scale: folded into scaleOutput for FP8 FMHA.
    //! Attention output is always FP16; downstream Q/DQ for o_proj is handled by the TRT graph.
    std::vector<float> mQkvScales{1.f, 1.f, 1.f};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

//! \brief Factory class for creating ViTAttentionPlugin instances
class ViTAttentionPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    ViTAttentionPluginCreator();
    ~ViTAttentionPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
