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

#include "testPluginLoader.h"

#include <NvInferRuntime.h>
#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace test
{

constexpr char const* kAllReducePluginName{"AllReducePlugin"};
constexpr char const* kAllReducePluginVersion{"1"};
constexpr char const* kPluginCreatorV3OneKind{"PLUGIN CREATOR_V3ONE"};

inline nvinfer1::IPluginCreatorV3One* allReducePluginCreator() noexcept
{
    if (loadPluginLibrary() == nullptr)
    {
        return nullptr;
    }

    auto* const registry = getPluginRegistry();
    if (registry == nullptr)
    {
        return nullptr;
    }

    auto* const creator = registry->getCreator(kAllReducePluginName, kAllReducePluginVersion, "");
    if (creator == nullptr || std::string(creator->getInterfaceInfo().kind) != kPluginCreatorV3OneKind)
    {
        return nullptr;
    }
    return static_cast<nvinfer1::IPluginCreatorV3One*>(creator);
}

//! Owns an AllReducePlugin instance built from a tp_size field.
class AllReducePluginOwner
{
public:
    explicit AllReducePluginOwner(int32_t tpSize) noexcept
        : mTpSize(tpSize)
    {
        auto* const creator = allReducePluginCreator();
        if (creator == nullptr)
        {
            return;
        }

        nvinfer1::PluginField field{"tp_size", &mTpSize, nvinfer1::PluginFieldType::kINT32, 1};
        nvinfer1::PluginFieldCollection fc{1, &field};
        mPlugin = creator->createPlugin("allReduceUnitTest", &fc, nvinfer1::TensorRTPhase::kBUILD);
    }

    ~AllReducePluginOwner() noexcept
    {
        delete mPlugin;
    }

    AllReducePluginOwner(AllReducePluginOwner const&) = delete;
    AllReducePluginOwner& operator=(AllReducePluginOwner const&) = delete;

    bool valid() const noexcept
    {
        return mPlugin != nullptr;
    }

    nvinfer1::IPluginV3OneRuntime* runtime() noexcept
    {
        if (mPlugin == nullptr)
        {
            return nullptr;
        }
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(
            mPlugin->getCapabilityInterface(nvinfer1::PluginCapabilityType::kRUNTIME));
    }

    nvinfer1::IPluginV3OneBuild* build() noexcept
    {
        if (mPlugin == nullptr)
        {
            return nullptr;
        }
        return static_cast<nvinfer1::IPluginV3OneBuild*>(
            mPlugin->getCapabilityInterface(nvinfer1::PluginCapabilityType::kBUILD));
    }

private:
    int32_t mTpSize{1};
    nvinfer1::IPluginV3* mPlugin{nullptr};
};

//! Frees device memory acquired for one test case.
class DeviceBuffer
{
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t bytes) noexcept
    {
        if (cudaMalloc(&mData, bytes) != cudaSuccess)
        {
            mData = nullptr;
        }
    }

    ~DeviceBuffer() noexcept
    {
        if (mData != nullptr)
        {
            cudaFree(mData);
        }
    }

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    void* get() const noexcept
    {
        return mData;
    }

private:
    void* mData{nullptr};
};

inline size_t dataTypeSize(nvinfer1::DataType dataType) noexcept
{
    return dataType == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(uint16_t);
}

inline nvinfer1::PluginTensorDesc makeLinearDesc(nvinfer1::DataType dataType, int64_t numElements) noexcept
{
    nvinfer1::PluginTensorDesc desc{};
    desc.dims.nbDims = 2;
    desc.dims.d[0] = 1;
    desc.dims.d[1] = numElements;
    desc.type = dataType;
    desc.format = nvinfer1::TensorFormat::kLINEAR;
    desc.scale = 1.0F;
    return desc;
}

//! Build a host byte buffer of numElements values of dataType, all set to value.
inline std::vector<uint8_t> makeHostBuffer(nvinfer1::DataType dataType, int64_t numElements, float value)
{
    std::vector<uint8_t> bytes(static_cast<size_t>(numElements) * dataTypeSize(dataType));
    for (int64_t index = 0; index < numElements; ++index)
    {
        if (dataType == nvinfer1::DataType::kFLOAT)
        {
            float const element = value;
            std::memcpy(bytes.data() + static_cast<size_t>(index) * sizeof(float), &element, sizeof(float));
        }
        else if (dataType == nvinfer1::DataType::kHALF)
        {
            half const element = __float2half(value);
            std::memcpy(bytes.data() + static_cast<size_t>(index) * sizeof(half), &element, sizeof(half));
        }
        else
        {
            __nv_bfloat16 const element = __float2bfloat16(value);
            std::memcpy(
                bytes.data() + static_cast<size_t>(index) * sizeof(__nv_bfloat16), &element, sizeof(__nv_bfloat16));
        }
    }
    return bytes;
}

//! Read a host byte buffer of dataType back as float values.
inline std::vector<float> readHostBuffer(
    nvinfer1::DataType dataType, std::vector<uint8_t> const& bytes, int64_t numElements)
{
    std::vector<float> values(static_cast<size_t>(numElements), 0.0F);
    for (int64_t index = 0; index < numElements; ++index)
    {
        if (dataType == nvinfer1::DataType::kFLOAT)
        {
            float element = 0.0F;
            std::memcpy(&element, bytes.data() + static_cast<size_t>(index) * sizeof(float), sizeof(float));
            values[static_cast<size_t>(index)] = element;
        }
        else if (dataType == nvinfer1::DataType::kHALF)
        {
            half element{};
            std::memcpy(&element, bytes.data() + static_cast<size_t>(index) * sizeof(half), sizeof(half));
            values[static_cast<size_t>(index)] = __half2float(element);
        }
        else
        {
            __nv_bfloat16 element{};
            std::memcpy(
                &element, bytes.data() + static_cast<size_t>(index) * sizeof(__nv_bfloat16), sizeof(__nv_bfloat16));
            values[static_cast<size_t>(index)] = __bfloat162float(element);
        }
    }
    return values;
}

//! Drive one plugin enqueue with a single input and a single output.
inline int32_t enqueueAllReduce(nvinfer1::IPluginV3OneRuntime* runtime, nvinfer1::PluginTensorDesc const& desc,
    void const* input, void* output, cudaStream_t stream) noexcept
{
    nvinfer1::PluginTensorDesc const outputDesc = desc;
    void const* inputs[]{input};
    void* outputs[]{output};
    return runtime->enqueue(&desc, &outputDesc, inputs, outputs, nullptr, stream);
}

} // namespace test
} // namespace trt_edgellm
