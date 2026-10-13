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

#include "kernels/moe/nvfp4A16BlackwellMoe/nvfp4A16BlackwellMoeJitRunner.h"

#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

/*!
 * @brief Thor (SM110) TensorRT V3 plugin for FP16 activations and NVFP4 routed-expert weights in the
 * BLACKWELL_MOE_N128_K64_V1 layout.
 *
 * One weight buffer per projection serves both backends: a tcgen05 grouped GEMM (prefill) and
 * CUDA-core GEMV kernels (decode). Inputs mirror Nvfp4A16MoePlugin except that the weights are rank-5
 * `[E, N/128, K/64, 128, 32]` int8 codes (row bytes pre-swizzled with the TMA 32B pattern: rows with bit 2
 * of n%128 set swap their 16-byte halves) plus `[E, N/128, K/64, 128, 4]` raw E4M3 block scales, and the
 * per-expert global scales are the checkpoint's verbatim fp32 `weight_scale_2` (no Marlin 2**7 factor).
 * `moe_inter_size` is the logical intermediate size; FC1 N padding to 128 happens inside the layout.
 *
 * Activations use token-major [T, H] and router logits use [T, E].
 *
 * max_routed_rows is the padded-row capacity of the permuted activation buffer: a runtime shape with T
 * tokens needs at most T * top_k + num_experts * 127 rows (0 == resolve from the optimization profile).
 */
class Nvfp4A16BlackwellMoePlugin : public nvinfer1::IPluginV3,
                                   public nvinfer1::IPluginV3OneCore,
                                   public nvinfer1::IPluginV3OneBuild,
                                   public nvinfer1::IPluginV3OneRuntime
{
public:
    Nvfp4A16BlackwellMoePlugin(std::string const& name, int32_t numExperts, int32_t topK, int32_t hiddenSize,
        int32_t moeInterSize, int32_t activationType, int32_t nGroup, int32_t topkGroup, int32_t normTopkProb,
        float routedScalingFactor, int32_t routingMode, int32_t maxRoutedRows, int32_t layout, int32_t backend);

    Nvfp4A16BlackwellMoePlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    Nvfp4A16BlackwellMoePlugin() = delete;
    Nvfp4A16BlackwellMoePlugin(Nvfp4A16BlackwellMoePlugin const&) = delete;
    Nvfp4A16BlackwellMoePlugin& operator=(Nvfp4A16BlackwellMoePlugin const&) = delete;
    ~Nvfp4A16BlackwellMoePlugin() noexcept override;

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

    //! Serialized-bundle state used by the creator: the build phase must not
    //! receive a bundle, the runtime phase must load one.
    bool hasSerializedJitBundle() const noexcept;
    void loadSerializedJitBundle();

private:
    void validateAttributes() const;
    bool validateTensorDesc(int32_t pos, nvinfer1::PluginTensorDesc const& desc) const noexcept;
    int32_t interSizePadded() const noexcept;

    std::string mLayerName;
    std::string mNamespace;
    int32_t mNumExperts{};
    int32_t mTopK{};
    int32_t mHiddenSize{};
    int32_t mMoeInterSize{}; //!< Logical intermediate size (FC1 N is padded to 128 inside the layout).
    int32_t mActivationType{};
    int32_t mNGroup{};
    int32_t mTopkGroup{};
    int32_t mNormTopkProb{};
    float mRoutedScalingFactor{};
    int32_t mRoutingMode{};
    int32_t mMaxRoutedRows{}; //!< Padded permuted-row capacity (0 == auto from the profile).
    int32_t mLayout{};
    int32_t mBackend{};

    //! NVRTC bundle of this layer's CUDA-core kernels: compiled (and loaded) in
    //! configurePlugin, serialized as the ``moe_jit_bundle`` field, loaded into the
    //! context-keyed module registry at deserialization / clone.
    void compileJitBundle();

    Nvfp4A16BlackwellMoeJitKernel mJitKernel;
    std::vector<uint8_t> mJitBundle;
    Nvfp4A16BlackwellMoeJitRunner mJitRunner;

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

//! Creator for Nvfp4A16BlackwellMoePlugin.
class Nvfp4A16BlackwellMoePluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    Nvfp4A16BlackwellMoePluginCreator();
    ~Nvfp4A16BlackwellMoePluginCreator() override = default;

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
