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

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Parallelism type for one logical rank group.
enum class ParallelType
{
    kTensor,
};

//! Process launch mode for parallel execution.
enum class ParallelLaunchMode
{
    kThread,
    kMpi,
};

//! Configuration for one logical parallel group.
struct ParallelGroupConfig
{
    ParallelType type{ParallelType::kTensor}; //!< Logical parallelism type.
    int32_t size{1};                          //!< Number of ranks in this group.
    int32_t rank{0};                          //!< Local rank within this group.
    int32_t globalRank{0};                    //!< Global rank within the full parallel world.
    int32_t localDevice{0};                   //!< CUDA device for this rank.

    char const* typeName() const noexcept;
    std::string toString() const;
};

//! External communication handles for one parallel group (e.g. NCCL communicators).
//! Shared by the public runtime config and the runtime coordinator.
struct ParallelBackendHandles
{
    ParallelType type{ParallelType::kTensor};
    std::vector<void*> handles{};
    bool ownsHandles{true};
};

//! Top-level parallel execution configuration. Size-1 dimensions are the single-device default.
struct ParallelConfig
{
    int32_t tensorParallelSize{1}; //!< Tensor parallel size.
    ParallelLaunchMode launchMode{ParallelLaunchMode::kThread};
    std::vector<int32_t> devices{}; //!< Optional local CUDA device list. Empty means [0, size).


    std::string toString() const;
};

//! Fully resolved tensor-parallel coordinates for one runtime rank.
//! Size-1 dimensions are explicit so single-device execution is the TP size-1 case.
struct ParallelMapping
{
    int32_t worldSize{1};
    int32_t globalRank{0};
    int32_t localDevice{0};

    int32_t tensorParallelSize{1};
    int32_t tensorParallelRank{0};

    bool isParallel() const noexcept
    {
        return worldSize > 1;
    }

    std::string toString() const;
};

char const* parallelTypeName(ParallelType type) noexcept;
char const* parallelLaunchModeName(ParallelLaunchMode mode) noexcept;

//! True when a parallel plan should execute inline on the caller's thread:
//! threaded launch, a single-rank world, and exactly one local rank. In this
//! case the coordinator skips worker threads, condition-variable dispatch, and
//! NUMA pinning so single-device execution keeps its original latency and
//! caller-thread streaming/audio semantics.
bool isInlineSingleRank(ParallelLaunchMode launchMode, int32_t worldSize, size_t localRankCount) noexcept;

//! True when @p localRanks contains every rank in [0, groupSize) in rank order.
bool isFullLocalParallelGroup(int32_t groupSize, std::vector<int32_t> const& localRanks) noexcept;

int32_t parallelGroupSize(ParallelConfig const& config, ParallelType type) noexcept;
int32_t parallelWorldSize(ParallelConfig const& config) noexcept;
int32_t parallelGroupRank(ParallelConfig const& config, ParallelType type, int32_t globalRank) noexcept;
ParallelMapping makeParallelMapping(ParallelConfig const& config, int32_t globalRank, int32_t localDevice) noexcept;
ParallelGroupConfig makeParallelGroupConfig(
    ParallelConfig const& config, ParallelType type, int32_t globalRank, int32_t localDevice);
std::vector<ParallelGroupConfig> activeParallelGroups(
    ParallelConfig const& config, int32_t globalRank, int32_t localDevice);

} // namespace rt
} // namespace trt_edgellm
