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

#include "int4GroupwiseGemmPlugin.h"
#include "kernels/int4GroupwiseGemmKernels/int4GroupwiseGemm.h"
#include "plugins/utils/pluginUtils.h"

#include <algorithm>
#include <cassert>
#include <cuda_fp16.h>
#include <mutex>
#include <optional>

#include <iostream>

using namespace nvinfer1;
namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kINT4_GEMM_PLUGIN_VERSION{"1"};
constexpr char const* kINT4_GEMM_PLUGIN_NAME{"Int4GroupwiseGemmPlugin"};

} // namespace

// Static class fields initialization
PluginFieldCollection Int4GroupwiseGemmPluginCreator::mFieldCollection{};
std::vector<PluginField> Int4GroupwiseGemmPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Int4GroupwiseGemmPluginCreator);

Int4GroupwiseGemmPlugin::Int4GroupwiseGemmPlugin(std::string const& name, int32_t N, int32_t K, int32_t groupSize)
    : mLayerName(name)
    , mGemmN(N)
    , mGemmK(K)
    , mGroupSize(groupSize)
{
}

Int4GroupwiseGemmPlugin::Int4GroupwiseGemmPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        std::string fieldName(fc->fields[i].name);
        if (fieldName == "gemm_n")
        {
            mGemmN = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "gemm_k")
        {
            mGemmK = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "group_size")
        {
            mGroupSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
    }
}

Int4GroupwiseGemmPlugin::~Int4GroupwiseGemmPlugin() {}

IPluginCapability* Int4GroupwiseGemmPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Int4GroupwiseGemmPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Int4GroupwiseGemmPlugin(mLayerName, mGemmN, mGemmK, mGroupSize);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

char const* Int4GroupwiseGemmPlugin::getPluginName() const noexcept
{
    return kINT4_GEMM_PLUGIN_NAME;
}

char const* Int4GroupwiseGemmPlugin::getPluginVersion() const noexcept
{
    return kINT4_GEMM_PLUGIN_VERSION;
}

char const* Int4GroupwiseGemmPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Int4GroupwiseGemmPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

int32_t Int4GroupwiseGemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Int4GroupwiseGemmPlugin::getOutputDataTypes(DataType* outputTypes, [[maybe_unused]] int32_t nbOutputs,
    DataType const* /* inputTypes */, int32_t /* nbInputs */) const noexcept
{
    try
    {
        assert(nbOutputs == 1);
        outputTypes[0] = DataType::kHALF;
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t Int4GroupwiseGemmPlugin::getOutputShapes(DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs,
    [[maybe_unused]] int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        assert(nbInputs == 3);
        assert(nbOutputs == 1);
        if (inputs[0].nbDims != 2 && inputs[0].nbDims != 3)
        {
            return -1;
        }
        outputs[0].nbDims = inputs[0].nbDims;
        for (int32_t i = 0; i < inputs[0].nbDims - 1; ++i)
        {
            outputs[0].d[i] = inputs[0].d[i];
        }
        outputs[0].d[inputs[0].nbDims - 1] = exprBuilder.constant(mGemmN);
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

bool Int4GroupwiseGemmPlugin::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] int32_t nbOutputs) noexcept
{
    try
    {
        assert(nbInputs == 3 && nbOutputs == 1);
        assert(pos < (nbInputs + nbOutputs));
        auto const& tensorDesc = inOut[pos].desc;
        bool status{true};

        switch (pos)
        {
        case 0:
        {
            status &= tensorDesc.type == DataType::kHALF;
            status &= tensorDesc.format == PluginFormat::kLINEAR;
            bool const supportedRank = tensorDesc.dims.nbDims == 2 || tensorDesc.dims.nbDims == 3;
            status &= supportedRank;
            status &= supportedRank && tensorDesc.dims.d[tensorDesc.dims.nbDims - 1] == mGemmK;
            break;
        }
        case 1:
        {
            status &= tensorDesc.type == DataType::kINT8;
            status &= tensorDesc.format == PluginFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 2;
            status &= tensorDesc.dims.d[0] == mGemmN / 2;
            status &= tensorDesc.dims.d[1] == mGemmK;
            break;
        }
        case 2:
        {
            status &= tensorDesc.type == DataType::kHALF;
            status &= tensorDesc.format == PluginFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 2;
            status &= tensorDesc.dims.d[0] == mGemmK / mGroupSize;
            status &= tensorDesc.dims.d[1] == mGemmN;
            break;
        }
        case 3:
        {
            status &= tensorDesc.type == DataType::kHALF;
            status &= tensorDesc.format == PluginFormat::kLINEAR;
            bool const supportedRank = tensorDesc.dims.nbDims == 2 || tensorDesc.dims.nbDims == 3;
            status &= supportedRank;
            status &= supportedRank && tensorDesc.dims.d[tensorDesc.dims.nbDims - 1] == mGemmN;
            break;
        }
        default: break;
        }
        return status;
    }
    catch (std::exception const& e)
    {
        return false;
    }
}

int32_t Int4GroupwiseGemmPlugin::configurePlugin(DynamicPluginTensorDesc const* /* in */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    return 0;
}

size_t Int4GroupwiseGemmPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

int32_t Int4GroupwiseGemmPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    try
    {
        auto const& inputDesc0 = inputDesc[0];
        int32_t M{1};
        for (int32_t i = 0; i < inputDesc0.dims.nbDims - 1; ++i)
        {
            M *= inputDesc0.dims.d[i];
        }

        half* gemmInPtr = reinterpret_cast<half*>(const_cast<void*>(inputs[0]));
        int8_t* weightsInPtr = reinterpret_cast<int8_t*>(const_cast<void*>(inputs[1]));
        half* ScaleInPtr = reinterpret_cast<half*>(const_cast<void*>(inputs[2]));
        half* gemmOutDevicePtr = reinterpret_cast<half*>(outputs[0]);

        // gemm_forward_cuda_new requires N divisible by its CTA_N tile size (128).
        // Smaller / non-aligned N (e.g. GDN's in_proj_a / in_proj_b = num_recurrent_heads
        // = 32 in Qwen3.5-4B) must use the GEMV kernel. GEMV only has template
        // instantiations for M=1..6, so split M into <=6-row chunks.
        // TODO: Iteration causes overhead. Need optimization if M is small and not divisible by 128. Possible
        // solutions: 1) Add more GEMV instantiations for larger M (e.g. 8, 16, 32). 2) Add a fallback GEMV kernel that
        // can handle arbitrary M without template instantiation (e.g. using dynamic shared memory).
        bool const useGemv = (M <= trt_edgellm::kernel::kGemvMaxM) || (mGemmN % trt_edgellm::kernel::kGemmCtaN != 0);
        if (useGemv)
        {
            for (int32_t m_start = 0; m_start < M; m_start += trt_edgellm::kernel::kGemvMaxM)
            {
                int32_t const m_chunk = std::min(trt_edgellm::kernel::kGemvMaxM, M - m_start);
                trt_edgellm::kernel::gemv_forward_cuda_new(gemmInPtr + m_start * mGemmK, weightsInPtr, ScaleInPtr,
                    gemmOutDevicePtr + m_start * mGemmN, m_chunk, mGemmN, mGemmK, mGroupSize, stream);
            }
        }
        else
        {
            trt_edgellm::kernel::gemm_forward_cuda_new(
                gemmInPtr, weightsInPtr, ScaleInPtr, gemmOutDevicePtr, M, mGemmN, mGemmK, mGroupSize, stream);
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t Int4GroupwiseGemmPlugin::onShapeChange(PluginTensorDesc const* /* in */, int32_t /* nbInputs */,
    PluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    return 0;
}

IPluginV3* Int4GroupwiseGemmPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* Int4GroupwiseGemmPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("gemm_n", &mGemmN, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("gemm_k", &mGemmK, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("group_size", &mGroupSize, PluginFieldType::kINT32, 1);

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

Int4GroupwiseGemmPluginCreator::Int4GroupwiseGemmPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("gemm_n", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("gemm_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("group_size", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Int4GroupwiseGemmPluginCreator::getPluginName() const noexcept
{
    return kINT4_GEMM_PLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* Int4GroupwiseGemmPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void Int4GroupwiseGemmPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* Int4GroupwiseGemmPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* Int4GroupwiseGemmPluginCreator::getPluginVersion() const noexcept
{
    return kINT4_GEMM_PLUGIN_VERSION;
}

IPluginV3* Int4GroupwiseGemmPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    try
    {
        Int4GroupwiseGemmPlugin* plugin = new Int4GroupwiseGemmPlugin(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
