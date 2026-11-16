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

#include "kernels/dualGemmKernels/cuteDslNvfp4DualGemmRunner.h"

#include <NvInferRuntime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

//! NVFP4 SwiGLU. Inputs in order:
//! activation INT8 [...,K/2], packed weight INT8 [2N,K/2], activation SF INT8 [SfAtomBytes(M,K)],
//! packed weight SF INT8 [2N*K/16], alpha FLOAT [2]. SF bytes encode positive E4M3 scales.
//! For row r and K/16 block b, SF atom coordinates are [r/128,b/4,r%32,(r%128)/32,b%4]
//! in a contiguous [ceil(rows/128),K/64,32,4,4] byte tensor.
//! Weight rows alternate 128 gate rows and 128 up rows; weight scales use that same row order.
//! Attributes: inter_size (N), hidden_size (K), activation (0 = SiLU), quantized_output (0 or 1; default 0).
//! Default output is HALF [...,N]. Quantized mode adds FLOAT global_scale[1] and returns
//! INT8 packed output [...,N/2] plus INT8 SF atoms [ceil(M/128)*128*N/16].
//! global_scale is a positive finite forward scale, retained by the caller for down GEMM:
//! down alpha = global_scale * down_weight_global_scale. FP16 rounding precedes fused output quantization; no H
//! workspace is required. Preserve any graph-defined cast of the dequantization global scale when folding down alpha.
class Nvfp4DualGemmPlugin : public nvinfer1::IPluginV3,
                            public nvinfer1::IPluginV3OneCore,
                            public nvinfer1::IPluginV3OneBuild,
                            public nvinfer1::IPluginV3OneRuntime
{
public:
    Nvfp4DualGemmPlugin(std::string const& name, int32_t interSize, int32_t hiddenSize, int32_t activation,
        int32_t quantizedOutput = 0);
    Nvfp4DualGemmPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    Nvfp4DualGemmPlugin() = delete;
    Nvfp4DualGemmPlugin(Nvfp4DualGemmPlugin const&) = delete;
    Nvfp4DualGemmPlugin& operator=(Nvfp4DualGemmPlugin const&) = delete;
    ~Nvfp4DualGemmPlugin() noexcept override = default;

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
    int32_t getNbTactics() noexcept override;
    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override;
    int32_t setTactic(int32_t tactic) noexcept override;
    char const* getTimingCacheID() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    int32_t inputCount() const noexcept;
    bool validateOutputs(nvinfer1::PluginTensorDesc const* outputs, nvinfer1::Dims const& activation) const noexcept;
    void validateAttributes() const;
    bool validateTensorDesc(int32_t pos, nvinfer1::PluginTensorDesc const& desc) const noexcept;

    std::string mLayerName;
    std::string mNamespace;
    int32_t mInterSize{};
    int32_t mHiddenSize{};
    int32_t mActivation{};
    int32_t mQuantizedOutput{};
    int32_t mMaxActiveClusters{};
    int32_t mTactic{};
    kernels::CuteDslNvfp4DualGemmRunner mRunner;
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class Nvfp4DualGemmPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    Nvfp4DualGemmPluginCreator();
    ~Nvfp4DualGemmPluginCreator() override = default;

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
