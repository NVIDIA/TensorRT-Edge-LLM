/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensor.h"
#include "checkMacros.h"
#include "logger.h"

#include <sstream>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace rt
{

namespace utils
{

size_t getTypeSize(DataType dataType)
{
    size_t size{0};
    switch (dataType)
    {
    case DataType::kINT64:
    {
        size = 8;
        break;
    }
    case DataType::kFLOAT:
    case DataType::kINT32:
    {
        size = 4;
        break;
    }
    case DataType::kHALF:
    case DataType::kBF16:
    {
        size = 2;
        break;
    }
    case DataType::kFP8:
    case DataType::kINT8:
    case DataType::kUINT8:
    {
        size = 1;
        break;
    }
    default:
    {
        // Sub-byte types cannot be processed here.
        throw std::runtime_error("Other types are not supported");
    }
    }
    return size;
}

std::array<int64_t, kMAX_DIMS> computeStrides(Coords const& shape)
{
    std::array<int64_t, kMAX_DIMS> strides;
    int32_t const numDims = shape.getNumDims();
    strides[numDims - 1] = 1;
    int64_t stride = 1;
    for (int32_t i = numDims - 2; i >= 0; --i)
    {
        stride *= shape[i + 1];
        strides[i] = stride;
    }
    return strides;
}
} // namespace utils

int64_t Coords::volume() const
{
    if (mNumDims == 0)
    {
        return 0;
    }
    int64_t vol = 1;
    for (int32_t i = 0; i < mNumDims; ++i)
    {
        vol *= mDims[i];
    }
    return vol;
}

nvinfer1::Dims Coords::getTRTDims() const
{
    nvinfer1::Dims dims;
    dims.nbDims = mNumDims;
    for (int32_t i = 0; i < mNumDims; ++i)
    {
        dims.d[i] = mDims[i];
    }
    return dims;
}

std::string Coords::formatString() const
{
    std::stringstream ss;
    ss << "[";
    for (int32_t i = 0; i < mNumDims; ++i)
    {
        ss << mDims[i];
        if (i < mNumDims - 1)
        {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

Tensor::Tensor(Coords const& shape, DeviceType deviceType, nvinfer1::DataType dataType, std::string const& name)
{
    if (shape.volume() == 0)
    {
        throw std::runtime_error("Construction of Tensor object with zero volume is prohibited");
    }

    if (dataType == DataType::kINT4 || dataType == DataType::kFP4)
    {
        throw std::runtime_error("Sub-type like kInt4 or kFP4 are not supported");
    }

    mShape = shape;
    mDeviceType = deviceType;
    mDataType = dataType;
    ownMemory = true;
    mStrides = utils::computeStrides(shape);
    mName = name;
    memoryCapacity = shape.volume() * utils::getTypeSize(dataType);
    if (deviceType == DeviceType::kCPU)
    {
        CUDA_CHECK(cudaMallocHost(&data, memoryCapacity));
        LOG_DEBUG("Tensor %s of shape %s with size %ld bytes (%.2f MB) allocated on CPU", name.c_str(),
            shape.formatString().c_str(), memoryCapacity, utils::toMB(memoryCapacity));
    }
    else
    {
        CUDA_CHECK(cudaMalloc(&data, memoryCapacity));
        LOG_DEBUG("Tensor %s of shape %s with size %ld bytes (%.2f MB) allocated on GPU", name.c_str(),
            shape.formatString().c_str(), memoryCapacity, utils::toMB(memoryCapacity));
    }
}

Tensor::Tensor(void* data, Coords const& shape, DeviceType deviceType, nvinfer1::DataType dataType,
    std::string const& name) noexcept
{
    // Populate the tensor information and only serve as a data container with shape.
    mShape = shape;
    mDeviceType = deviceType;
    mDataType = dataType;
    ownMemory = false;

    // Allow construction of a non-owned tensor with zero volume.
    // The data pointer won't be granted to the tensor object since no access is needed for zero-volume tensors.
    if (shape.volume() != 0)
    {
        mStrides = utils::computeStrides(shape);
        this->data = data;
        memoryCapacity = shape.volume() * utils::getTypeSize(dataType);
    }
    else
    {
        this->data = nullptr;
        memoryCapacity = 0;
        mStrides = {};
    }
    mName = name;
}

Tensor::~Tensor()
{
    releaseResource();
}

Tensor::Tensor(Tensor&& other) noexcept
{
    this->data = other.data;
    this->mShape = other.mShape;
    this->mStrides = other.mStrides;
    this->mDeviceType = other.mDeviceType;
    this->mDataType = other.mDataType;
    this->ownMemory = other.ownMemory;
    this->memoryCapacity = other.memoryCapacity;
    this->mName = other.mName;
    // Reset the other tensor.
    other.data = nullptr;
    other.mShape = Coords{};
    other.mStrides = std::array<int64_t, kMAX_DIMS>{};
    other.mDeviceType = DeviceType::kCPU;
    other.mDataType = DataType::kFLOAT;
    other.ownMemory = false;
    other.memoryCapacity = 0;
    other.mName = {};
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this != &other)
    {
        releaseResource();
        this->data = other.data;
        this->mShape = other.mShape;
        this->mStrides = other.mStrides;
        this->mDeviceType = other.mDeviceType;
        this->mDataType = other.mDataType;
        this->ownMemory = other.ownMemory;
        this->memoryCapacity = other.memoryCapacity;
        this->mName = other.mName;
        // Reset the other tensor.
        other.data = nullptr;
        other.mShape = {};
        other.mStrides = {};
        other.mDeviceType = {};
        other.mDataType = {};
        other.ownMemory = false;
        other.memoryCapacity = 0;
        other.mName = {};
    }
    return *this;
}

Coords Tensor::getShape() const noexcept
{
    return mShape;
}

DeviceType Tensor::getDeviceType() const noexcept
{
    return mDeviceType;
}

DataType Tensor::getDataType() const noexcept
{
    return mDataType;
}

std::string const& Tensor::getName() const noexcept
{
    return mName;
}

bool Tensor::getOwnMemory() const noexcept
{
    return ownMemory;
}

int64_t Tensor::getMemoryCapacity() const noexcept
{
    return memoryCapacity;
}

int64_t Tensor::getStride(int32_t idx) const
{
    if (idx < 0 || idx >= mShape.getNumDims())
    {
        throw std::out_of_range("Tensor: indexing of strides out of range");
    }
    return mStrides[idx];
}

Dims Tensor::getTRTDims() const noexcept
{
    Dims dims;
    dims.nbDims = mShape.getNumDims();
    for (int32_t i = 0; i < mShape.getNumDims(); ++i)
    {
        dims.d[i] = mShape[i];
    }
    return dims;
}

bool Tensor::isEmpty() const noexcept
{
    return mShape.volume() == 0;
}

void* Tensor::rawPointer() noexcept
{
    return data;
}

void const* Tensor::rawPointer() const noexcept
{
    return data;
}

bool Tensor::reshape(Coords shape) noexcept
{
    if (!ownMemory)
    {
        return false;
    }

    if (static_cast<int64_t>(shape.volume() * utils::getTypeSize(mDataType)) > memoryCapacity)
    {
        return false;
    }

    mShape = shape;
    mStrides = utils::computeStrides(shape);
    return true;
}

void Tensor::releaseResource()
{
    if (ownMemory)
    {
        if (mDeviceType == DeviceType::kCPU)
        {
            LOG_DEBUG("Tensor %s of shape %s with size %ld bytes (%.2f MB) freed on CPU", mName.c_str(),
                mShape.formatString().c_str(), memoryCapacity, utils::toMB(memoryCapacity));
            CUDA_CHECK(cudaFreeHost(data));
        }
        else
        {
            LOG_DEBUG("Tensor %s of shape %s with size %ld bytes (%.2f MB) freed on GPU", mName.c_str(),
                mShape.formatString().c_str(), memoryCapacity, utils::toMB(memoryCapacity));
            CUDA_CHECK(cudaFree(data));
        }
    }
    data = nullptr;
    ownMemory = false;
    memoryCapacity = 0;
    mShape = Coords{};
    mStrides = std::array<int64_t, kMAX_DIMS>{};
    mDeviceType = {};
    mDataType = {};
    mName = {};
}

} // namespace rt
} // namespace trt_edgellm
