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

#include "runnerResourceResolver.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace trt_edgellm::safety
{
namespace
{

SafetyStatus invalid() noexcept
{
    return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Incompatible Runner resource metadata");
}

} // namespace

SafetyStatus describeRuntimeTensor(SafetyTensorView const& tensor, int64_t profile, RuntimeTensorDesc& output) noexcept
{
    output = {};
    if (profile < 0 || profile > std::numeric_limits<int32_t>::max() || tensor.name.empty() || tensor.logicalRank < 0
        || tensor.logicalRank > nvinfer1::Dims::MAX_DIMS || tensor.physicalRank != tensor.logicalRank
        || tensor.strideRank != tensor.logicalRank || tensor.componentsPerVector != 1 || tensor.vectorizedDim != -1)
    {
        return invalid();
    }
    uint64_t elements{1};
    for (int32_t i = tensor.logicalRank; i-- > 0;)
    {
        auto const dim = tensor.logicalShape[i];
        if (dim <= 0 || tensor.physicalShape[i] != dim || tensor.strides[i] < 0
            || static_cast<uint64_t>(tensor.strides[i]) != elements
            || static_cast<uint64_t>(dim)
                > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / elements)
        {
            return invalid();
        }
        elements *= static_cast<uint64_t>(dim);
    }
    if (tensor.bytesPerComponent == 0
        || elements > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / tensor.bytesPerComponent
        || elements * tensor.bytesPerComponent != tensor.profileBytes || tensor.capacityBytes < tensor.profileBytes
        || tensor.capacityBytes > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    {
        return invalid();
    }
    try
    {
        RuntimeTensorDesc desc{};
        desc.name = tensor.name;
        switch (tensor.dataType)
        {
        case SafetyTensorDataType::kFLOAT: desc.dataType = nvinfer1::DataType::kFLOAT; break;
        case SafetyTensorDataType::kHALF: desc.dataType = nvinfer1::DataType::kHALF; break;
        case SafetyTensorDataType::kINT64: desc.dataType = nvinfer1::DataType::kINT64; break;
        case SafetyTensorDataType::kINT32: desc.dataType = nvinfer1::DataType::kINT32; break;
        default: return invalid();
        }
        if (tensor.ioMode == SafetyTensorIOMode::kINPUT)
        {
            desc.ioMode = nvinfer1::TensorIOMode::kINPUT;
        }
        else if (tensor.ioMode == SafetyTensorIOMode::kOUTPUT)
        {
            desc.ioMode = nvinfer1::TensorIOMode::kOUTPUT;
        }
        else
        {
            return invalid();
        }
        RuntimeTensorProfile metadata{};
        metadata.profileId = static_cast<int32_t>(profile);
        metadata.shape.nbDims = tensor.logicalRank;
        std::copy_n(tensor.logicalShape.begin(), tensor.logicalRank, metadata.shape.d);
        metadata.sizeInBytes = static_cast<std::size_t>(tensor.profileBytes);
        desc.profiles.push_back(metadata);
        auto const status = validateTensorDesc(desc);
        if (status.okStatus())
        {
            output = std::move(desc);
        }
        return status;
    }
    catch (std::bad_alloc const&)
    {
        return SafetyStatus::error(SafetyStatusCode::kFailedAllocation, "Runner tensor metadata allocation failed");
    }
    catch (std::length_error const&)
    {
        return invalid();
    }
    catch (...)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kFailedExecution, "Unexpected Runner metadata failure");
    }
}

RunnerResourceResolver::RunnerResourceResolver(std::vector<ModelLocalDataBinding> const& modelLocalData) noexcept
    : mModelLocalData(modelLocalData)
{
}

SafetyStatus RunnerResourceResolver::prepare(std::vector<LMStageContext> const& contexts) noexcept
{
    if (!mStages.empty())
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "Runner resources already published");
    }
    if (contexts.empty())
    {
        return invalid();
    }
    try
    {
        std::vector<Stage> stages;
        stages.reserve(contexts.size());
        for (auto const& context : contexts)
        {
            if (context.plan.id.empty() || context.mainStream == nullptr || context.plan.profiles.size() != 1
                || context.profiles.size() != 1 || std::any_of(stages.begin(), stages.end(), [&](auto const& other) {
                       return other.id == context.plan.id && other.slot == context.plan.slot;
                   }))
            {
                return invalid();
            }
            stages.emplace_back();
            auto& stage = stages.back();
            stage.id = context.plan.id;
            stage.slot = context.plan.slot;
            stage.stream = context.mainStream;
            auto const& tensors = context.profiles.front().tensors;
            stage.descriptors.reserve(tensors.size());
            stage.tensors.reserve(tensors.size());
            for (std::size_t i = 0; i < tensors.size(); ++i)
            {
                auto const& tensor = tensors[i];
                // TRT can retain shape-only IO with no payload to borrow.
                if (tensor.profileBytes == 0)
                {
                    continue;
                }
                if (tensor.data == nullptr || std::count_if(tensors.begin(), tensors.end(), [&](auto const& other) {
                        return other.name == tensor.name;
                    }) != 1)
                {
                    return invalid();
                }
                stage.descriptors.emplace_back();
                auto const status
                    = describeRuntimeTensor(tensor, context.plan.profiles.front(), stage.descriptors.back());
                if (!status.okStatus())
                {
                    return status;
                }
                RuntimeMemoryKind kind{};
                switch (tensor.memPlacement)
                {
                case SafetyTensorMemoryPlacement::kGPU: kind = RuntimeMemoryKind::kDEVICE; break;
                case SafetyTensorMemoryPlacement::kCPU: kind = RuntimeMemoryKind::kPAGEABLE; break;
                case SafetyTensorMemoryPlacement::kCPU_PINNED: kind = RuntimeMemoryKind::kPINNED; break;
                default: return invalid();
                }
                stage.tensors.push_back(
                    {&stage.descriptors.back(), {tensor.data, static_cast<std::size_t>(tensor.capacityBytes), kind}});
            }
        }
        mStages = std::move(stages);
        return SafetyStatus::ok();
    }
    catch (std::bad_alloc const&)
    {
        return SafetyStatus::error(SafetyStatusCode::kFailedAllocation, "Runner resource metadata allocation failed");
    }
    catch (std::length_error const&)
    {
        return invalid();
    }
    catch (...)
    {
        return SafetyStatus::fatal(
            SafetyStatusCode::kFailedExecution, "Unexpected Runner resource preparation failure");
    }
}

SafetyStatus RunnerResourceResolver::resolve(
    RuntimeBorrowedResourceRequest const& request, RuntimeBorrowedResource& output) const noexcept
{
    output = {};
    if (mStages.empty())
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "Runner resources are not prepared");
    }
    auto const stage = std::find_if(mStages.begin(), mStages.end(),
        [&](auto const& candidate) { return candidate.id == request.stage && candidate.slot == request.resourceSlot; });
    if (stage == mStages.end())
    {
        return invalid();
    }
    switch (request.kind)
    {
    case RuntimeBorrowedResourceKind::kCUDA_STREAM: output.stream = stage->stream; break;
    case RuntimeBorrowedResourceKind::kIO_BINDING:
        for (auto const& tensor : stage->tensors)
        {
            if (tensor.desc->name == request.binding)
            {
                output.tensor = &tensor;
                break;
            }
        }
        break;
    case RuntimeBorrowedResourceKind::kMODEL_LOCAL_DATA:
        for (auto const& entry : mModelLocalData)
        {
            if (entry.slot == request.resourceSlot && entry.name == request.binding)
            {
                if (output.modelLocalData != nullptr || entry.object == nullptr)
                {
                    output = {};
                    return invalid();
                }
                output.modelLocalData = entry.object;
            }
        }
        break;
    }
    return output.tensor != nullptr || output.stream != nullptr || output.modelLocalData != nullptr ? SafetyStatus::ok()
                                                                                                    : invalid();
}

} // namespace trt_edgellm::safety
