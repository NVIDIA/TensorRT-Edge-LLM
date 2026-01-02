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

#include "pluginUtils.h"
#include "common/checkMacros.h"

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

void* alignDevicePtr(void* ptr)
{
    return reinterpret_cast<void*>(
        (reinterpret_cast<uintptr_t>(ptr) + kDEVICE_ALIGNMENT - 1) & ~(kDEVICE_ALIGNMENT - 1));
}

size_t alignTensorSize(size_t size)
{
    return ((size + kDEVICE_ALIGNMENT - 1) / kDEVICE_ALIGNMENT) * kDEVICE_ALIGNMENT;
}

size_t accumulateWorkspaceSize(size_t currentSize, rt::Coords const& shape, DataType dataType)
{
    size_t alignedSize = alignTensorSize(currentSize);
    size_t tensorSizeBytes = rt::utils::getTypeSize(dataType) * static_cast<size_t>(shape.volume());

    return alignedSize + alignTensorSize(tensorSizeBytes);
}

rt::Tensor assignTensorFromWorkspace(void*& workspace, rt::Coords const& shape, DataType dataType)
{
    check::check(workspace != nullptr && reinterpret_cast<uintptr_t>(workspace) % kDEVICE_ALIGNMENT == 0,
        "Workspace pointer shall be valid and aligned to device alignment granularity.");

    // Create non-owned tensor instance from the workspace pointer.
    rt::Tensor tensor(workspace, shape, rt::DeviceType::kGPU, dataType);

    // Move the workspace pointer to the next aligned position after this tensor.
    size_t alignedSize = alignTensorSize(tensor.getMemoryCapacity());
    uintptr_t newAddr = reinterpret_cast<uintptr_t>(workspace) + alignedSize;
    workspace = reinterpret_cast<void*>(newAddr);
    return tensor;
}

} // namespace plugins
} // namespace trt_edgellm
