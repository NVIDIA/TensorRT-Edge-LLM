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

#include "runtimeResource.h"

#include <algorithm>
#include <limits>

namespace trt_edgellm::safety
{

SafetyStatus validateTensorDesc(RuntimeTensorDesc const& desc) noexcept
{
    if (desc.name.empty() || desc.profiles.empty()
        || (desc.ioMode != nvinfer1::TensorIOMode::kINPUT && desc.ioMode != nvinfer1::TensorIOMode::kOUTPUT))
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid tensor binding metadata");
    }
    std::size_t elementBytes{0};
    switch (desc.dataType)
    {
    case nvinfer1::DataType::kHALF: elementBytes = 2; break;
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32: elementBytes = 4; break;
    case nvinfer1::DataType::kINT64: elementBytes = 8; break;
    default: break;
    }
    if (elementBytes == 0)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Unsupported resource tensor type");
    }
    for (auto const& profile : desc.profiles)
    {
        if (profile.profileId < 0 || profile.shape.nbDims < 0 || profile.shape.nbDims > nvinfer1::Dims::MAX_DIMS
            || std::count_if(
                   desc.profiles.begin(), desc.profiles.end(),
                   [&](auto const& other) { return other.profileId == profile.profileId; })
                != 1)
        {
            return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid tensor profile metadata");
        }
        std::size_t bytes{elementBytes};
        for (int32_t i = 0; i < profile.shape.nbDims; ++i)
        {
            auto const dim = profile.shape.d[i];
            if (dim <= 0
                || static_cast<uint64_t>(dim)
                    > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / bytes)
            {
                return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Tensor shape exceeds capacity");
            }
            bytes *= static_cast<std::size_t>(dim);
        }
        if (bytes != profile.sizeInBytes)
        {
            return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Tensor shape and byte size disagree");
        }
    }
    return SafetyStatus::ok();
}

} // namespace trt_edgellm::safety
