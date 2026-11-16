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

#include "common/tensor.h"
#include <array>
#include <functional>

namespace trt_edgellm::rt
{
//! Gather two FP16 shards already encoded at their final offsets in the output allocation.
using VisionGatherFn = std::function<bool(void*, int32_t, int32_t, cudaStream_t)>;

//! Validate and gather one rank-local visual output, then expose its full token-major shape.
bool gatherVisionTensor(Tensor& local, std::array<int64_t, 2> const& tokenCounts, int32_t rank,
    VisionGatherFn const& gather, cudaStream_t stream);

//! Callbacks enqueue FP16 collectives on stream; each rank's shard is already at its final offset.
//! Callback arguments are (send, receive, count, stream) and (buffer, count, root, stream).
template <typename AllGather, typename Broadcast>
bool gatherVisionShards(void* buffer, int32_t rank0Count, int32_t rank1Count, int32_t rank, AllGather const& allGather,
    Broadcast const& broadcast, cudaStream_t stream)
{
    auto* rank1Buffer = static_cast<uint16_t*>(buffer) + rank0Count;
    if (rank0Count == rank1Count)
    {
        void const* sendBuffer = rank == 0 ? buffer : rank1Buffer;
        return allGather(sendBuffer, buffer, rank0Count, stream);
    }
    return broadcast(buffer, rank0Count, 0, stream) && broadcast(rank1Buffer, rank1Count, 1, stream);
}
} // namespace trt_edgellm::rt
