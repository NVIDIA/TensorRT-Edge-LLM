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

#include <NvInferRuntime.h>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

/// TensorRT plugin for DFlash target KV cache update (V3 — IPluginV3).
///
/// Inputs:
///   0: k_delta              [B, L, numKVHeads, headDim] FP16
///   1: v_delta              [B, L, numKVHeads, headDim] FP16
///   2: past_key_value       Paged KV pool [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim] FP16 —
///                           the same KVCacheManager allocation as the AttentionPlugin binding (K/V
///                           split OUTERMOST), bound unreshaped. maxBatch/capPadded are recovered at
///                           enqueue time from numPages (dim 1) and the build-time `pages_per_slot`
///                           attribute (maxBatch = numPages / pagesPerSlot, cap = pagesPerSlot *
///                           kTOKENS_PER_PAGE) — see setPagesPerSlot().
///   3: rope_cos_sin         [ropeBatch, cosSinSeqLen, rotaryDim] FP32 — cosSinSeqLen <= cap
///   4: delta_start_positions [B] INT32
///   5: delta_lengths         [B] INT32
///
/// Outputs:
///   0: present_key_value    same shape/dtype as past_key_value (aliased)
///
/// BREAKING ABI NOTE (paged-KV substrate): `past_key_value`/`present_key_value` moved to the paged
/// pool layout above and a new serialized `pages_per_slot` field was added (see setPagesPerSlot()).
/// The plugin version string was deliberately NOT bumped -- this project always regenerates ONNX and
/// rebuilds engines together with the runtime, so an ABI break here is accepted rather than
/// versioned. Any ONNX/engine older than this change must be re-exported and rebuilt; it will not
/// load correctly against this plugin.
class DFlashTargetKVCacheUpdatePlugin : public nvinfer1::IPluginV3,
                                        public nvinfer1::IPluginV3OneCore,
                                        public nvinfer1::IPluginV3OneBuildV2,
                                        public nvinfer1::IPluginV3OneRuntime
{
public:
    DFlashTargetKVCacheUpdatePlugin(std::string const& name);
    DFlashTargetKVCacheUpdatePlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    DFlashTargetKVCacheUpdatePlugin() = delete;
    DFlashTargetKVCacheUpdatePlugin(DFlashTargetKVCacheUpdatePlugin const&) = delete;
    ~DFlashTargetKVCacheUpdatePlugin() override = default;

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

    //! @brief Set the pool's pages-per-slot (capPadded / kTOKENS_PER_PAGE), a build-time-only fact
    //! not derivable from the pool-shaped past_key_value binding alone (dim 1 there is numPages =
    //! maxBatch * pagesPerSlot). Called by the builder (llmBuilder.cpp) right after ONNX parsing,
    //! from mBuilderConfig.maxKVCacheCapacity; propagated through clone() and engine
    //! serialization (getFieldsToSerialize) so the deserialized runtime plugin has it too.
    void setPagesPerSlot(int32_t pagesPerSlot) noexcept;

private:
    // Input indices
    static constexpr int32_t kIN_K_DELTA = 0;
    static constexpr int32_t kIN_V_DELTA = 1;
    static constexpr int32_t kIN_PAST_KV = 2;
    static constexpr int32_t kIN_ROPE_COS_SIN = 3;
    static constexpr int32_t kIN_DELTA_START = 4;
    static constexpr int32_t kIN_DELTA_LENGTHS = 5;

    // Output indices
    static constexpr int32_t kOUT_PRESENT_KV = 0;

    std::string mLayerName;
    std::string mNamespace;
    //! Pool pages per slot (capPadded / kTOKENS_PER_PAGE); see setPagesPerSlot(). 0 means unset —
    //! enqueue() rejects that as a builder-configuration bug rather than silently misderiving cap.
    int32_t mPagesPerSlot{0};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class DFlashTargetKVCacheUpdatePluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    DFlashTargetKVCacheUpdatePluginCreator();
    ~DFlashTargetKVCacheUpdatePluginCreator() override = default;

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
