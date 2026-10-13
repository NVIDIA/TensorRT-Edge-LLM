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

#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{
//! Router selection kernel chosen by the \c routing_mode plugin attribute.
enum class Nvfp4MoeRoutingMode : int32_t
{
    kSOFTMAX_TOPK = 0,            //!< softmax + flat top-k + pre-softmax additive bias (DeepSeek, Qwen3).
    kSIGMOID_GROUP_TOPK = 1,      //!< sigmoid + grouped top-k + renormalize + scale (NemotronH).
    kSOFTMAX_TOPK_POST_SCALE = 2, //!< softmax + flat top-k + post-renorm multiplicative scale (Gemma4).
};

/*!
 * @brief TensorRT plugin: NVFP4 MoE — FP16 activations with on-the-fly NVFP4
 * quant. SM100/SM101/SM110 use the split FC1/FC2 CuTeDSL path.
 *
 * Weight layout: FC1 is the 64-row up/gate interleave
 * ``[up_chunk(64), gate_chunk(64), up_chunk(64), ...]`` (the layout the split
 * split FC1 kernel reads natively). For the SM12x fused path, see the sibling
 * \c NvFP4MoEPluginGeforce plugin which consumes the plain
 * ``[up_all, gate_all]`` concat layout.
 *
 * @note This plugin is only supported on SM100, SM101, and SM110.
 * @note This plugin is only supported on FP16 I/O.
 * @note The split FC1/FC2 path supports swiglu and relu2 with E=128, 0 < top_k <= 8.
 * @note Activations use token-major [T, H] and router logits use [T, E].
 */
class Nvfp4MoePlugin : public nvinfer1::IPluginV3,
                       public nvinfer1::IPluginV3OneCore,
                       public nvinfer1::IPluginV3OneBuild,
                       public nvinfer1::IPluginV3OneRuntime
{
public:
    Nvfp4MoePlugin(std::string const& name, int32_t numExperts, int32_t topK, int32_t hiddenSize, int32_t moeInterSize,
        int32_t activationType, int32_t nGroup, int32_t topkGroup, int32_t normTopkProb, float routedScalingFactor,
        int32_t routingMode, int32_t backend, int32_t maxRoutedRows, int32_t ioDtype);

    Nvfp4MoePlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    Nvfp4MoePlugin() = delete;
    Nvfp4MoePlugin(Nvfp4MoePlugin const&) = delete;

    ~Nvfp4MoePlugin() noexcept override;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;

    nvinfer1::IPluginV3* clone() noexcept override;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

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

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    std::string mLayerName;
    std::string mNamespace;

    int32_t mNumExperts{};
    int32_t mTopK{};
    int32_t mHiddenSize{};
    int32_t mMoeInterSize{};
    //! Encoding: 0=identity, 1=silu, 2=swiglu, 3=gelu, 4=relu2. All five values are accepted.
    int32_t mActivationType{};
    //! Router encoding: 0=softmax top-k, 1=sigmoid grouped top-k.
    int32_t mRoutingMode{};
    int32_t mNGroup{1};
    int32_t mTopkGroup{1};
    int32_t mNormTopkProb{1};
    float mRoutedScalingFactor{1.0F};
    //! Encoding: 0=auto, 1=decode, 2=prefill. v1 accepts all three values.
    int32_t mBackend{};
    //! Upper bound on num_tokens * top_k; used to size the decode workspace.
    //! Build-time semantics:
    //!   * 0 from the user → \c configurePlugin populates it from the optimization
    //!     profile (\c max_tokens * \c top_k) and serializes the resolved value so
    //!     the runtime instance always carries a concrete cap.
    //!   * non-zero from the user → must be >= profile \c max_tokens * \c top_k;
    //!     a smaller value is rejected at \c configurePlugin time so undersized
    //!     workspaces cannot escape into runtime.
    //! Runtime semantics: \c onShapeChange and \c enqueue reject any launch whose
    //! \c num_tokens * \c top_k exceeds the resolved cap.
    int32_t mMaxRoutedRows{};
    //! Encoding: 0=bf16, 1=fp16. v1 accepts 1 only.
    int32_t mIoDtype{};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

//! Plugin creator — parses PluginFieldCollection into the attributes above, registers under
//! TensorRT's default namespace, exposes name "Nvfp4MoePlugin" / version "1".
class Nvfp4MoePluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    Nvfp4MoePluginCreator();
    ~Nvfp4MoePluginCreator() override = default;

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
