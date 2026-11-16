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

#include "lmStageContext.h"

#include "../runtime/resources/trtResourceManager.h"

#include <algorithm>
#include <exception>

namespace trt_edgellm::safety
{
namespace
{
bool mapData(nvinfer2::safe::TypedArray const& binding, SafetyTensorView& view) noexcept
{
    using nvinfer2::safe::DataType;
    switch (binding.getType())
    {
    case DataType::kFLOAT:
        view.dataType = SafetyTensorDataType::kFLOAT;
        view.data = binding.getFloat();
        break;
    case DataType::kHALF:
        view.dataType = SafetyTensorDataType::kHALF;
        view.data = binding.getHalf();
        break;
    case DataType::kINT64:
        view.dataType = SafetyTensorDataType::kINT64;
        view.data = binding.getInt64();
        break;
    case DataType::kINT32:
        view.dataType = SafetyTensorDataType::kINT32;
        view.data = binding.getInt32();
        break;
    case DataType::kINT8:
        view.dataType = SafetyTensorDataType::kINT8;
        view.data = binding.getInt8();
        break;
    case DataType::kUINT8:
        view.dataType = SafetyTensorDataType::kUINT8;
        view.data = binding.getUint8();
        break;
    case DataType::kBOOL:
        view.dataType = SafetyTensorDataType::kBOOL;
        view.data = binding.getBool();
        break;
    case DataType::kFP8:
    case DataType::kBF16:
    case DataType::kINT4:
    case DataType::kFP4:
    case DataType::kE8M0:
    default: return false;
    }
    return true;
}

bool mapPlacement(nvinfer2::safe::MemoryPlacement placement, SafetyTensorView& view) noexcept
{
    using nvinfer2::safe::MemoryPlacement;
    switch (placement)
    {
    case MemoryPlacement::kNONE:
    case MemoryPlacement::kGPU: view.memPlacement = SafetyTensorMemoryPlacement::kGPU; break;
    case MemoryPlacement::kCPU: view.memPlacement = SafetyTensorMemoryPlacement::kCPU; break;
    case MemoryPlacement::kCPU_PINNED: view.memPlacement = SafetyTensorMemoryPlacement::kCPU_PINNED; break;
    default: return false;
    }
    return true;
}

bool mapIOMode(nvinfer2::safe::TensorIOMode mode, SafetyTensorView& view) noexcept
{
    using nvinfer2::safe::TensorIOMode;
    switch (mode)
    {
    case TensorIOMode::kNONE: view.ioMode = SafetyTensorIOMode::kNONE; break;
    case TensorIOMode::kINPUT: view.ioMode = SafetyTensorIOMode::kINPUT; break;
    case TensorIOMode::kOUTPUT: view.ioMode = SafetyTensorIOMode::kOUTPUT; break;
    default: return false;
    }
    return true;
}

template <size_t kRANK>
bool copyDimensions(int32_t rank, int64_t const* dimensions, std::array<int64_t, kRANK>& output) noexcept
{
    if (rank < -1 || rank > static_cast<int32_t>(output.size()))
    {
        return false;
    }
    if (rank > 0)
    {
        std::copy_n(dimensions, static_cast<size_t>(rank), output.begin());
    }
    return true;
}

SafetyStatus makeTensorView(
    SafetyTensorMetadata const& metadata, nvinfer2::safe::TypedArray const& binding, SafetyTensorView& view) noexcept
{
    auto const& descriptor = metadata.descriptor;
    if (binding.getType() != descriptor.dataType || !mapData(binding, view)
        || !mapPlacement(descriptor.memPlacement, view) || !mapIOMode(descriptor.ioMode, view)
        || !copyDimensions(descriptor.userShape.nbDims, descriptor.userShape.d, view.logicalShape)
        || !copyDimensions(descriptor.shape.nbDims, descriptor.shape.d, view.physicalShape)
        || !copyDimensions(descriptor.stride.nbDims, descriptor.stride.d, view.strides)
        || !copyDimensions(descriptor.strideOrder.nbDims, descriptor.strideOrder.d, view.strideOrder))
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Unsupported Safety tensor view metadata");
    }
    view.name = metadata.name;
    view.logicalRank = descriptor.userShape.nbDims;
    view.physicalRank = descriptor.shape.nbDims;
    view.strideRank = descriptor.stride.nbDims;
    view.strideOrderRank = descriptor.strideOrder.nbDims;
    view.bytesPerComponent = descriptor.bytesPerComponent;
    view.componentsPerVector = descriptor.componentsPerVector;
    view.vectorizedDim = descriptor.vectorizedDim;
    view.profileBytes = descriptor.sizeInBytes;
    view.capacityBytes = binding.getSize();
    if (view.profileBytes > view.capacityBytes || (view.profileBytes != 0 && view.data == nullptr))
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Safety tensor view exceeds its bound storage");
    }
    if (view.logicalRank == -1 && view.vectorizedDim == -1 && view.componentsPerVector == 1 && view.physicalRank >= 0
        && view.physicalRank <= static_cast<int32_t>(view.logicalShape.size()))
    {
        view.logicalRank = view.physicalRank;
        std::copy_n(view.physicalShape.begin(), static_cast<size_t>(view.logicalRank), view.logicalShape.begin());
    }
    return SafetyStatus::ok();
}
} // namespace

SafetyStatus LMStageContext::getTensorView(
    int64_t profile, std::string_view name, SafetyTensorView const*& output) const noexcept
{
    output = nullptr;
    for (auto const& candidate : profiles)
    {
        if (candidate.metadata != nullptr && candidate.metadata->profileIndex == profile)
        {
            for (auto const& tensor : candidate.tensors)
            {
                if (tensor.name == name)
                {
                    output = &tensor;
                    return SafetyStatus::ok();
                }
            }
            break;
        }
    }
    return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Tensor name or profile is not cached");
}

SafetyStatus LMStageContext::getTensorView(std::string_view name, SafetyTensorView const*& output) const noexcept
{
    output = nullptr;
    if (profiles.size() != 1 || profiles.front().metadata == nullptr)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Tensor lookup requires one selected profile");
    }
    return getTensorView(profiles.front().metadata->profileIndex, name, output);
}

namespace detail
{
SafetyStatus initializeTensorViews(LMStageProfileView& profile) noexcept
{
    if (profile.metadata == nullptr || profile.resources == nullptr
        || profile.metadata->profileIndex != profile.resources->profileIndex
        || profile.metadata->tensors.size() != profile.resources->bindings.size() || !profile.tensors.empty())
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid Safety tensor view cache source");
    }
    try
    {
        profile.tensors.reserve(profile.metadata->tensors.size());
        for (size_t index = 0; index < profile.metadata->tensors.size(); ++index)
        {
            SafetyTensorView view;
            auto const status
                = makeTensorView(profile.metadata->tensors[index], profile.resources->bindings[index], view);
            if (!status.okStatus())
            {
                return status;
            }
            profile.tensors.push_back(view);
        }
        return SafetyStatus::ok();
    }
    catch (std::exception const& error)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kFailedAllocation, error.what());
    }
    catch (...)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kFailedAllocation, "Safety tensor view cache allocation failed");
    }
}
} // namespace detail
} // namespace trt_edgellm::safety
