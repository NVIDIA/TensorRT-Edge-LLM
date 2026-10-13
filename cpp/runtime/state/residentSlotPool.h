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

#include "runtime/exec/scheduledStep.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace trt_edgellm::rt
{

//! Runtime-thread ownership of a fixed resident-state pool. The caller must not release a slot until every CUDA
//! operation that can access that resident row has completed. Release invalidates the identity before returning the
//! slot to the free list, so delayed host completions cannot commit into a later request that reused the same row.
class ResidentSlotPool
{
public:
    ResidentSlotPool() = default;
    explicit ResidentSlotPool(int32_t capacity);

    void reset(int32_t capacity);
    std::optional<ResidentRef> acquire();
    bool release(ResidentRef resident);
    bool contains(ResidentRef resident) const noexcept;

    int32_t capacity() const noexcept;
    int32_t available() const noexcept;

private:
    std::vector<uint64_t> mEpochs;
    std::vector<int8_t> mOccupied;
    std::vector<CacheSlot> mFreeSlots;
};

} // namespace trt_edgellm::rt
