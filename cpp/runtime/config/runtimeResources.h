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

#include "runtime/multiDevice/parallelConfig.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Rank placement and execution handles supplied to the coordinator. Empty streams are created by the runtime.
struct RuntimeResources
{
    std::vector<int32_t> localRanks{};
    std::unordered_map<int32_t, int32_t> localRankDevices{};
    std::vector<cudaStream_t> localStreams{};
    bool ownsLocalStreams{true};
    std::vector<ParallelBackendHandles> backendHandles{};
};

} // namespace rt
} // namespace trt_edgellm
