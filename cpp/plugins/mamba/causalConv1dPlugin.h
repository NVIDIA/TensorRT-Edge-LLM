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
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

//! \brief TensorRT plugin for depthwise causal conv1d with optional speculative-verify state checkpoints.
//!
//! Normal decode updates one persistent rolling conv state. Normal prefill consumes a linear sequence. For MTP/DDTree
//! verification, the plugin emits intermediate conv states so runtime can commit only the accepted speculative path.
//! DDTree additionally consumes tree parent/depth metadata because adjacent flattened verify nodes are not necessarily
//! parent and child in the proposal tree.
//!
//! Required metadata inputs end with `execution_phase_marker` at slot 7 and the shape-only INT32
//! `context_sequence_count_carrier` at slot 8. Verify-capable instances append parent/depth at slots 9/10 and
//! spec_verify_layout at slot 11.
class CausalConv1dPlugin : public nvinfer1::IPluginV3,
                           public nvinfer1::IPluginV3OneCore,
                           public nvinfer1::IPluginV3OneBuildV2,
                           public nvinfer1::IPluginV3OneRuntime
{
public:
    CausalConv1dPlugin(std::string const& name, int32_t stride, int32_t padding, int32_t dilation, int32_t groups,
        bool useSpecVerifyState = false);
    CausalConv1dPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    CausalConv1dPlugin() = delete;
    CausalConv1dPlugin(CausalConv1dPlugin const&) = delete;
    ~CausalConv1dPlugin() override;

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
    int32_t getAliasedInput(int32_t outputIndex) noexcept override;

    // IPluginV3OneRuntime
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    //! Phase 8 needs a depthwise layer without spec-verify state and a packed varlen kernel for (\p width, mDilation)
    //! on mSMVersion. Decides both the phase 8 workspace reservation and enqueue acceptance.
    bool supportsCompactPhase(int32_t dim, int32_t width) const noexcept;

    //! This layer's name and shape (\p dim channels, kernel \p width) for error logs (describePluginLayer).
    std::string describeLayer(int32_t dim, int32_t width) const;

    //! Phase 8: packed varlen conv on rows [0, T_p) through dense history state when Np > 0, single-step decode on
    //! [T_p, T).
    int32_t enqueueCompact(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;

    //! Single-step decode of \p numSequences rows starting at token row \p firstRow and sequence \p firstSequence;
    //! phase 3 passes (0, 0, N), phase 8 passes (T_p, Np, Nd). Throws on invalid kernel arguments.
    void enqueueDecodeRows(void const* const* inputs, void* const* outputs, int32_t firstRow, int32_t firstSequence,
        int32_t numSequences, int32_t dim, int32_t stateLen, int32_t statePoolRows, nvinfer1::Dims const& weightDims,
        nvinfer1::DataType type, cudaStream_t stream) const;

    std::string mLayerName;
    std::string mNamespace;
    int32_t mStride{1};
    int32_t mPadding{0};
    int32_t mDilation{1};
    int32_t mGroups{0};
    int32_t mSMVersion{0};           //!< Captured device SM version used for the profiled chain specialization
    bool mUseSpecVerifyState{false}; //!< Enable spec-verify intermediate_conv_states output
    int32_t mUseSpecVerifyStateField{0};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class CausalConv1dPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    CausalConv1dPluginCreator();
    ~CausalConv1dPluginCreator() override = default;

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
