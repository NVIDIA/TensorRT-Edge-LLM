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
//! Same encoding as the SM110 \c Nvfp4MoePlugin so a single ONNX schema can
//! target either plugin variant.
enum class NvFP4MoEGeforceRoutingMode : int32_t
{
    kSOFTMAX_TOPK = 0,            //!< softmax + flat top-k + pre-softmax additive bias (DeepSeek, Qwen3).
    kSIGMOID_GROUP_TOPK = 1,      //!< sigmoid + grouped top-k + renormalize + scale (NemotronH).
    kSOFTMAX_TOPK_POST_SCALE = 2, //!< softmax + flat top-k + post-renorm multiplicative scale (Gemma4).
};

/*!
 * @brief TensorRT plugin: NVFP4 fused MoE (CuTeDSL SM120/SM121) — FP16 activations,
 * dynamic on-the-fly NVFP4 quant, fused route/pack + FC1 + activation + quant + FC2 + scatter.
 *
 * Per expert: \c y_e = down_proj( act( up_proj(x) ) ) with NVFP4 packed weights.
 *
 * Weight layout: FC1 is the plain ``[up_all, gate_all]`` concat along the M
 * axis (no 64-row up/gate interleave). This matches what the fused SM12x
 * CuTeDSL kernel expects natively. The SM110 \c Nvfp4MoePlugin uses the
 * separate 64-row interleaved layout consumed by the split FC1/FC2 backend.
 *
 * @note This plugin is only supported on SM120 and SM121 (consumer Blackwell).
 * @note This plugin is only supported on FP16 I/O.
 * @note Supported activations: identity, silu, swiglu, gelu, relu2.
 * @note Activations use token-major [T, H] and router logits use [T, E].
 */
class NvFP4MoEPluginGeforce : public nvinfer1::IPluginV3,
                              public nvinfer1::IPluginV3OneCore,
                              public nvinfer1::IPluginV3OneBuild,
                              public nvinfer1::IPluginV3OneRuntime
{
public:
    NvFP4MoEPluginGeforce(std::string const& name, int32_t numExperts, int32_t topK, int32_t hiddenSize,
        int32_t moeInterSize, int32_t activationType, int32_t nGroup, int32_t topkGroup, int32_t normTopkProb,
        float routedScalingFactor, int32_t routingMode, int32_t backend, int32_t maxRoutedRows, int32_t ioDtype);

    NvFP4MoEPluginGeforce(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    NvFP4MoEPluginGeforce() = delete;
    NvFP4MoEPluginGeforce(NvFP4MoEPluginGeforce const&) = delete;

    ~NvFP4MoEPluginGeforce() noexcept override;

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

    //! Per-plugin device buffer holding two contiguous identity copies of
    //! ``[0, 1, ..., mNumExperts-1]`` (total ``2 * mNumExperts`` int32). Allocated
    //! through \c mGpuAllocator in \c attachToContext and freed in the destructor.
    //! Threaded into the runner via \c CuteDslNvfp4MoeParams::weightExpertIds /
    //! \c globalToLocalExpertIds so the runner does no allocation, copy, or
    //! growth on the enqueue path. Build-phase / unattached plugin instances
    //! retain \c nullptr and never enqueue.
    int32_t* mIdentityExpertTable{nullptr};
    //! Allocator captured in \c attachToContext. Non-null iff this plugin owns
    //! \c mIdentityExpertTable and is responsible for freeing it.
    nvinfer1::IGpuAllocator* mGpuAllocator{nullptr};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

//! Plugin creator — parses PluginFieldCollection into the attributes above, registers under
//! TensorRT's default namespace, exposes name "NvFP4MoEPluginGeforce" / version "1".
class NvFP4MoEPluginGeforceCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    NvFP4MoEPluginGeforceCreator();
    ~NvFP4MoEPluginGeforceCreator() override = default;

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
