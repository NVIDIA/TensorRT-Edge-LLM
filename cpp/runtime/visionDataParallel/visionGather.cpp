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

#include "runtime/visionDataParallel/visionGather.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include <algorithm>
#include <limits>

namespace trt_edgellm::rt
{
bool gatherVisionTensor(Tensor& local, std::array<int64_t, 2> const& tokenCounts, int32_t rank,
    VisionGatherFn const& gather, cudaStream_t stream)
{
    ELLM_CHECK(rank >= 0 && rank < 2 && tokenCounts[0] > 0 && tokenCounts[1] > 0,
        "Joint-image visual gather requires two nonempty shards and a valid rank.");
    Coords const shape = local.getShape();
    ELLM_CHECK(local.getDataType() == DataType::kHALF && local.getDeviceType() == DeviceType::kGPU
            && shape.getNumDims() == 2 && shape[0] == tokenCounts[rank] && shape[1] > 0,
        "Joint-image vision data-parallel shard tensor shape mismatch.");
    int64_t const hiddenSize = shape[1];
    ELLM_CHECK(std::max(tokenCounts[0], tokenCounts[1]) <= std::numeric_limits<int32_t>::max() / hiddenSize,
        "Joint-image visual gather count exceeds int32 capacity.");
    int64_t const fullTokens = tokenCounts[0] + tokenCounts[1];
    ELLM_CHECK(fullTokens * hiddenSize <= local.getMemoryCapacity() / static_cast<int64_t>(sizeof(uint16_t)),
        "Joint-image gathered output exceeds the vision runner's capacity.");
    if (!gather(local.rawPointer(), static_cast<int32_t>(tokenCounts[0] * hiddenSize),
            static_cast<int32_t>(tokenCounts[1] * hiddenSize), stream))
    {
        LOG_ERROR("Joint-image visual gather failed for %s.", local.getName().c_str());
        return false;
    }
    ELLM_CHECK(local.reshape({fullTokens, hiddenSize}), "Failed to reshape joint-image visual output.");
    return true;
}
} // namespace trt_edgellm::rt
