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
#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

//! \brief TensorRT plugin for Gated Delta Net (V3 — IPluginV3).
//!
//! Registered as "gated_delta_net". The execution phase selects prefill, decode,
//! linear speculative verify, or DDTree verify.
//! Requires SM80+ and K=V=128.
//!
//! \par Dimension notation
//!   n   = batch size
//!   h   = number of Q/K heads
//!   hv  = number of V heads
//!   k   = head dimension K (must be 128)
//!   v   = head dimension V (must be 128)
//!
//! \par Inputs
//!   [0]  q               [tokens, h,  k]        FP16  query
//!   [1]  k               [tokens, h,  k]        FP16  key
//!   [2]  v               [tokens, hv, v]        FP16  value
//!   [3]  a               [tokens, hv]           FP16  input gate
//!   [4]  b               [tokens, hv]           FP16  output gate
//!   [5]  A_log           [hv]                  FP32  log decay
//!   [6]  dt_bias         [hv]                  FP16  delta-time bias
//!   [7]  h0_source       [resident_rows, hv, k, v] FP32 resident recurrent-state pool
//!   [8]  context_lengths [n]                   INT32 valid token count per batch row
//!   [9]  query_start_offsets [n + 1]            INT32 token-row offsets
//!   [10] state_indices   [n]                    INT32 execution row to resident row
//!   [11] execution_phase_marker [1..8]          INT32 shape-only phase marker
//!   [12] context_sequence_count_carrier [0..n]   INT32 shape-only context count
//!   [13] tree_parent_ids [tokens]                INT32 DDTree parent node indices (optional)
//!   [14] tree_depths     [tokens]                INT32 DDTree node depths (optional)
//!
//! \par Outputs
//!   [0]  o               [tokens, hv, v]        FP16 output
//!   [1]  h0_out          [resident_rows, hv, k, v] FP32 aliased resident-state pool
//!   [2]  intermediate_states [tokens, hv, k, v] FP32 (spec-verify only, optional)
//!        Per-step recurrent state cache for linear speculative-decoding rollback.
//!        Present when spec-verify state tracking is enabled.
//!        DDTree verify keeps h0_source read-only and stores replay data here
//!        for accepted-state commit.
class GatedDeltaNetPlugin : public nvinfer1::IPluginV3,
                            public nvinfer1::IPluginV3OneCore,
                            public nvinfer1::IPluginV3OneBuildV2,
                            public nvinfer1::IPluginV3OneRuntime
{
public:
    //! \param name         Plugin instance name
    //! \param kDim         Head dimension K (must be 128 for CuTe DSL kernel)
    //! \param vDim         Head dimension V (must be 128 for CuTe DSL kernel)
    GatedDeltaNetPlugin(std::string const& name, int32_t kDim = 128, int32_t vDim = 128,
        bool useSpecVerifyState = false, bool useDDTree = false, bool useDiffusionState = false);
    GatedDeltaNetPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    GatedDeltaNetPlugin() = delete;
    GatedDeltaNetPlugin(GatedDeltaNetPlugin const&) = delete;
    ~GatedDeltaNetPlugin() override;

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
    std::string mLayerName;
    std::string mNamespace;
    int32_t mKDim{128};              //!< Head dimension K (kernel supports 128 only)
    int32_t mVDim{128};              //!< Head dimension V (kernel supports 128 only)
    bool mUseSpecVerifyState{false}; //!< Enable spec-verify intermediate_states output
    bool mUseDDTree{false};          //!< Enable DDTree parent/depth metadata inputs for tree-state execution.
    bool mUseDiffusionState{false};  //!< Enable transactional diffusion denoise/commit phases.
    int32_t mSMVersion{0};           //!< Captured device SM version used for build-time capability checks
    int32_t mUseSpecVerifyStateField{0};
    int32_t mUseDDTreeField{0};
    int32_t mUseDiffusionStateField{0};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class GatedDeltaNetPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    GatedDeltaNetPluginCreator();
    ~GatedDeltaNetPluginCreator() override = default;

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
