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

#include "common/allReducePath.h"
#include "plugins/utils/pluginUtils.h"

#include <NvInferRuntime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{

namespace plugins
{

//! \brief One device's NCCL resources used by plugin execution paths.
struct NcclAllReducePathRegistration
{
    void* communicator{nullptr};
    void* allReduceFunction{nullptr};
};

//! \brief Consistent snapshot of every registered execution path for one device.
struct AllReducePathRegistrations
{
    NcclAllReducePathRegistration nccl{};
};

//! \brief Register NCCL resources for one CUDA device.
bool registerNcclAllReducePath(int32_t deviceId, void* ncclComm, void* ncclAllReduceFunction) noexcept;

//! \brief Remove NCCL resources only if the expected communicator is still registered.
bool unregisterNcclAllReducePath(int32_t deviceId, void* expectedNcclComm) noexcept;

//! \brief Snapshot all available path registrations while holding the registry lock once.
AllReducePathRegistrations snapshotAllReducePathRegistrationsForDevice(int32_t deviceId) noexcept;

//! \brief TensorRT plugin for tensor-parallel all-reduce
//!
//! This plugin sums intermediate activations during TensorRT engine execution.
//! It is inserted after row-parallel linear layers (attention output projection
//! and MLP down projection) to combine partial results from tensor-parallel ranks.
//!
//! The plugin is a passthrough when tpSize=1 (no tensor parallelism).
//! For TP, it uses an optional accelerated execution path when supported and
//! otherwise falls back to the required NCCL path registered by the runtime.
class AllReducePlugin : public nvinfer1::IPluginV3,
                        public nvinfer1::IPluginV3OneCore,
                        public nvinfer1::IPluginV3OneBuild,
                        public nvinfer1::IPluginV3OneRuntime
{
public:
    //! \brief Constructor with TP configuration
    //! \param[in] name Plugin instance name
    //! \param[in] tpSize Tensor parallel world size
    AllReducePlugin(std::string const& name, int32_t tpSize);

    //! \brief Constructor from TensorRT plugin fields
    //! \param[in] name Plugin instance name
    //! \param[in] fc Plugin field collection
    AllReducePlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    AllReducePlugin() = delete;
    AllReducePlugin(AllReducePlugin const&) = delete;
    ~AllReducePlugin() override;

    //! \name IPluginV3 Methods
    //! @{
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;
    //! @}

    //! \name IPluginV3OneCore Methods
    //! @{
    char const* getPluginName() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    //! @}

    //! \name IPluginV3OneBuild Methods
    //! @{
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
    //! @}

    //! \name IPluginV3OneRuntime Methods
    //! @{
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;
    //! @}

    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    std::string mLayerName; //!< Plugin layer name
    std::string mNamespace; //!< Plugin namespace
    int32_t mTpSize{1};     //!< Tensor parallel world size

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

//! \brief Factory class for creating AllReducePlugin instances
class AllReducePluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    AllReducePluginCreator();
    ~AllReducePluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    char const* getPluginNamespace() const noexcept override;
    char const* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm

// Status-returning lifecycle C ABI used by the runtime. Keep these names
// unmangled so runtime/plugin decoupling does not depend on the C++ ABI.
extern "C" EDGELLM_PLUGIN_EXPORT bool edgellmRegisterNcclCommForAllReducePlugin(
    int deviceId, void* ncclComm, void* ncclAllReduceFunction) noexcept;
extern "C" EDGELLM_PLUGIN_EXPORT bool edgellmUnregisterNcclCommForAllReducePlugin(
    int deviceId, void* ncclComm) noexcept;

