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

#include "allReducePlugin.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "plugins/utils/pluginUtils.h"

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kALL_REDUCE_PLUGIN_VERSION{"1"};
constexpr char const* kALL_REDUCE_PLUGIN_NAME{"AllReducePlugin"};

std::mutex gAllReducePathRegistryMutex;
std::unordered_map<int32_t, NcclAllReducePathRegistration> gNcclRegistrations;

// Input/output indices
constexpr int32_t kIN_TENSOR_IDX{0};
constexpr int32_t kOUT_TENSOR_IDX{0};

// NCCL data type and op constants
constexpr int32_t kNcclFloat16 = 6;
constexpr int32_t kNcclFloat32 = 7;
constexpr int32_t kNcclBfloat16 = 9;
constexpr int32_t kNcclSum = 0;
constexpr int32_t kNcclSuccess = 0;

// Per-device NCCL state for single-process multi-GPU TP.
// Each GPU device gets its own NCCL communicator handle.
using NcclAllReduceFn = int (*)(void const*, void*, size_t, int, int, void*, cudaStream_t);

AllReduceExecutionStatus executeIdentityAllReducePath(PluginTensorDesc const& inputDesc, void const* input,
    void* output, int64_t numElements, int32_t tpSize, cudaStream_t stream)
{
    if (tpSize > 1)
    {
        return AllReduceExecutionStatus::kUnavailable;
    }

    size_t const typeSize = rt::utils::getTypeSize(inputDesc.type);
    cudaError_t const error = cudaMemcpyAsync(output, input, numElements * typeSize, cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess)
    {
        LOG_ERROR("AllReducePlugin: identity path copy failed: %s", cudaGetErrorString(error));
        return AllReduceExecutionStatus::kFailure;
    }
    return AllReduceExecutionStatus::kSuccess;
}

AllReduceExecutionStatus executeNcclAllReducePath(NcclAllReducePathRegistration const& registration,
    PluginTensorDesc const& inputDesc, void const* input, void* output, int64_t numElements, int32_t tpSize,
    int32_t deviceId, cudaStream_t stream)
{
    if (tpSize <= 1 || registration.communicator == nullptr || registration.allReduceFunction == nullptr)
    {
        return AllReduceExecutionStatus::kUnavailable;
    }

    int32_t ncclType = kNcclFloat16;
    if (inputDesc.type == DataType::kFLOAT)
    {
        ncclType = kNcclFloat32;
    }
    else if (inputDesc.type == DataType::kBF16)
    {
        ncclType = kNcclBfloat16;
    }

    auto const ncclAllReduce = reinterpret_cast<NcclAllReduceFn>(registration.allReduceFunction);
    int32_t const result
        = ncclAllReduce(input, output, numElements, ncclType, kNcclSum, registration.communicator, stream);
    if (result != kNcclSuccess)
    {
        LOG_ERROR("AllReducePlugin: NCCL path failed with error %d on device %d", result, deviceId);
        return AllReduceExecutionStatus::kFailure;
    }
    return AllReduceExecutionStatus::kSuccess;
}

} // namespace

bool registerNcclAllReducePath(int32_t deviceId, void* ncclComm, void* ncclAllReduceFunction) noexcept
{
    if (deviceId < 0 || ncclComm == nullptr || ncclAllReduceFunction == nullptr)
    {
        LOG_ERROR("Cannot register NCCL AllReduce path: device=%d, communicator=%p, function=%p.", deviceId, ncclComm,
            ncclAllReduceFunction);
        return false;
    }

    std::lock_guard<std::mutex> lock(gAllReducePathRegistryMutex);
    gNcclRegistrations[deviceId] = NcclAllReducePathRegistration{ncclComm, ncclAllReduceFunction};
    return true;
}

bool unregisterNcclAllReducePath(int32_t deviceId, void* expectedNcclComm) noexcept
{
    if (deviceId < 0 || expectedNcclComm == nullptr)
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(gAllReducePathRegistryMutex);
    auto const registration = gNcclRegistrations.find(deviceId);
    if (registration == gNcclRegistrations.end() || registration->second.communicator != expectedNcclComm)
    {
        return false;
    }
    gNcclRegistrations.erase(registration);
    return true;
}

AllReducePathRegistrations snapshotAllReducePathRegistrationsForDevice(int32_t deviceId) noexcept
{
    AllReducePathRegistrations snapshot{};
    if (deviceId < 0)
    {
        return snapshot;
    }

    std::lock_guard<std::mutex> lock(gAllReducePathRegistryMutex);
    auto const ncclRegistration = gNcclRegistrations.find(deviceId);
    if (ncclRegistration != gNcclRegistrations.end())
    {
        snapshot.nccl = ncclRegistration->second;
    }
    return snapshot;
}

// Static class fields initialization
PluginFieldCollection AllReducePluginCreator::mFieldCollection{};
std::vector<PluginField> AllReducePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AllReducePluginCreator);

// ========================== AllReducePlugin Implementation ==========================

AllReducePlugin::AllReducePlugin(std::string const& name, int32_t tpSize)
    : mLayerName(name)
    , mTpSize(tpSize)
{
    LOG_DEBUG("AllReducePlugin created: name=%s, tpSize=%d", name.c_str(), tpSize);
}

AllReducePlugin::AllReducePlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    if (fc == nullptr)
    {
        throw std::invalid_argument("AllReducePlugin requires plugin fields");
    }
    auto tpSize = parsePluginScalarField<int32_t>("tp_size", fc);
    if (!tpSize.has_value())
    {
        throw std::invalid_argument("AllReducePlugin requires 'tp_size' field");
    }
    mTpSize = tpSize.value();
    LOG_DEBUG("AllReducePlugin deserialized from fields: name=%s, tpSize=%d", name.c_str(), mTpSize);
}

AllReducePlugin::~AllReducePlugin() {}

IPluginCapability* AllReducePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

IPluginV3* AllReducePlugin::clone() noexcept
{
    try
    {
        auto* plugin = new AllReducePlugin(mLayerName, mTpSize);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("AllReducePlugin clone failed: %s", e.what());
        return nullptr;
    }
}

int32_t AllReducePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t AllReducePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        if (nbOutputs != 1 || nbInputs != 1)
        {
            return -1;
        }
        outputTypes[kOUT_TENSOR_IDX] = inputTypes[kIN_TENSOR_IDX];
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t AllReducePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* /* shapeInputs */,
    int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& /* exprBuilder */) noexcept
{
    try
    {
        if (nbInputs != 1 || nbOutputs != 1)
        {
            return -1;
        }
        outputs[kOUT_TENSOR_IDX] = inputs[kIN_TENSOR_IDX];
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

bool AllReducePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != 1 || nbOutputs != 1 || pos >= nbInputs + nbOutputs)
    {
        return false;
    }

    auto const& desc = inOut[pos].desc;
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    if (desc.type != DataType::kHALF && desc.type != DataType::kFLOAT && desc.type != DataType::kBF16)
    {
        return false;
    }

    if (pos >= nbInputs)
    {
        return desc.type == inOut[kIN_TENSOR_IDX].desc.type;
    }

    return true;
}

int32_t AllReducePlugin::configurePlugin(DynamicPluginTensorDesc const* /* in */, int32_t nbInputs,
    DynamicPluginTensorDesc const* /* out */, int32_t nbOutputs) noexcept
{
    return (nbInputs == 1 && nbOutputs == 1) ? 0 : -1;
}

size_t AllReducePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

int32_t AllReducePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    try
    {
        auto const& inDesc = inputDesc[kIN_TENSOR_IDX];

        // Compute total number of elements
        int64_t numElements = 1;
        for (int32_t d = 0; d < inDesc.dims.nbDims; ++d)
        {
            numElements *= inDesc.dims.d[d];
        }

        int currentDevice = -1;
        CUDA_CHECK(cudaGetDevice(&currentDevice));

        AllReducePathRegistrations const registrations = snapshotAllReducePathRegistrationsForDevice(currentDevice);

        AllReduceExecutionStatus status = executeIdentityAllReducePath(
            inDesc, inputs[kIN_TENSOR_IDX], outputs[kOUT_TENSOR_IDX], numElements, mTpSize, stream);
        if (status == AllReduceExecutionStatus::kSuccess)
        {
            return 0;
        }
        if (status == AllReduceExecutionStatus::kFailure)
        {
            return -1;
        }


        status = executeNcclAllReducePath(registrations.nccl, inDesc, inputs[kIN_TENSOR_IDX], outputs[kOUT_TENSOR_IDX],
            numElements, mTpSize, currentDevice, stream);
        if (status == AllReduceExecutionStatus::kSuccess)
        {
            return 0;
        }
        if (status == AllReduceExecutionStatus::kFailure)
        {
            return -1;
        }

        LOG_ERROR(
            "AllReducePlugin: no execution path is available for TP size %d on device %d; the required "
            "NCCL path is not registered",
            mTpSize, currentDevice);
        return -1;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("AllReducePlugin enqueue failed: %s", e.what());
        return -1;
    }
}

int32_t AllReducePlugin::onShapeChange(
    PluginTensorDesc const* /* in */, int32_t nbInputs, PluginTensorDesc const* /* out */, int32_t nbOutputs) noexcept
{
    return (nbInputs == 1 && nbOutputs == 1) ? 0 : -1;
}

IPluginV3* AllReducePlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* AllReducePlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("tp_size", &mTpSize, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

char const* AllReducePlugin::getPluginName() const noexcept
{
    return kALL_REDUCE_PLUGIN_NAME;
}

char const* AllReducePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void AllReducePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* AllReducePlugin::getPluginVersion() const noexcept
{
    return kALL_REDUCE_PLUGIN_VERSION;
}

// ========================== AllReducePluginCreator Implementation ==========================

AllReducePluginCreator::AllReducePluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("tp_size", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* AllReducePluginCreator::getPluginName() const noexcept
{
    return kALL_REDUCE_PLUGIN_NAME;
}

PluginFieldCollection const* AllReducePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void AllReducePluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* AllReducePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* AllReducePluginCreator::getPluginVersion() const noexcept
{
    return kALL_REDUCE_PLUGIN_VERSION;
}

IPluginV3* AllReducePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    try
    {
        auto* plugin = new AllReducePlugin(name, fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("AllReducePluginCreator::createPlugin failed: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm

extern "C" EDGELLM_PLUGIN_EXPORT bool edgellmRegisterNcclCommForAllReducePlugin(
    int deviceId, void* ncclComm, void* ncclAllReduceFunction) noexcept
{
    return trt_edgellm::plugins::registerNcclAllReducePath(deviceId, ncclComm, ncclAllReduceFunction);
}

extern "C" EDGELLM_PLUGIN_EXPORT bool edgellmUnregisterNcclCommForAllReducePlugin(int deviceId, void* ncclComm) noexcept
{
    return trt_edgellm::plugins::unregisterNcclAllReducePath(deviceId, ncclComm);
}

