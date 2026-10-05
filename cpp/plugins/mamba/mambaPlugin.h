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

//! \brief TensorRT plugin for Mamba Selective State Update (SSM)
//!
//! Registered as "update_ssm_state" under the "trt_edgellm" ONNX domain.
//!
//! Implements the selective state space model update:
//!   new_state = state * exp(A * dt) + B * dt * x
//!   output = sum_i(new_state_i * C_i) + D * x
//!
//! SiLU gating (z) is handled externally by the ONNX graph (gated_rms_norm).
//!
//! Input ordering (see constants defined in mambaPlugin.cpp):
//!   [0] x          [tokens, nheads, dim]                  FP16
//!   [1] A          [nheads]                              FP32
//!   [2] B          [tokens, ngroups, dstate]             FP16
//!   [3] C          [tokens, ngroups, dstate]             FP16
//!   [4] D          [nheads]                              FP16
//!   [5] dt         [tokens, nheads]                      FP16
//!   [6] dt_bias    [nheads]                              FP16
//!   [7] state      [resident_rows, nheads, dim, dstate]  FP16
//!   [8] query_lengths [batch]                            INT32
//!   [9] query_start_offsets [batch + 1]                  INT32
//!   [10] state_indices [batch]                           INT32
//!   [11] execution_phase_marker [1..8]                   INT32
//!   [12] context_sequence_count_carrier [0..batch]       INT32 shape-only context count
//!   [13] tree_parent_ids [tokens]                         INT32 (DDTree engines only)
//!   [14] tree_depths [tokens]                            INT32 (DDTree engines only)
//!
//! Outputs:
//!   [0] output     [tokens, nheads, dim]                  FP16
//!   [1] state_out  [resident_rows, nheads, dim, dstate]  aliased state pool
//!   [2] replay_da  [tokens, nheads]                       FP32 (spec-verify engines only)
//!   [3] replay_u   [tokens, nheads, dim]                  FP32 (spec-verify engines only)
//!   [4] replay_b   [tokens, ngroups, dstate]              FP32 (spec-verify engines only)
//!   [5] replay_dt  [tokens, nheads]                       FP32 (spec-verify engines only)
class MambaPlugin : public nvinfer1::IPluginV3,
                    public nvinfer1::IPluginV3OneCore,
                    public nvinfer1::IPluginV3OneBuild,
                    public nvinfer1::IPluginV3OneRuntime
{
public:
    MambaPlugin(std::string const& name, int32_t dim, int32_t dstate, int32_t nheads, int32_t ngroups,
        int32_t dtSoftplus, int32_t useSpecVerifyState = 0, int32_t useDDTree = 0, int32_t replayFormatVersion = 0);

    MambaPlugin() = delete;
    MambaPlugin(MambaPlugin const&) = delete;
    ~MambaPlugin() override;

    // IPluginV3OneCore
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;
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
    //! The common ragged input contract is unconditional. DDTree engines append parent/depth inputs;
    //! spec-verify engines append replay outputs.
    int32_t numInputs() const noexcept;
    int32_t numOutputs() const noexcept;

    std::string mLayerName;
    std::string mNamespace;

    int32_t mDim{};
    int32_t mDstate{};
    int32_t mNheads{};
    int32_t mNgroups{};
    int32_t mDtSoftplus{};
    //! Emit per-token replay factors for accepted-path state reconstruction.
    int32_t mUseSpecVerifyState{};
    //! DDTree verify consumes parent/depth inputs and evaluates each node from its ancestor state.
    int32_t mUseDDTree{};
    //! Serialized replay format guard; non-spec engines do not consume replay outputs.
    int32_t mReplayFormatVersion{};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class MambaPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    MambaPluginCreator();
    ~MambaPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    char const* getPluginNamespace() const noexcept override;
    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
