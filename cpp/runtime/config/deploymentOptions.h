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

#include "runtime/config/deploymentConfig.h"
#include "runtime/config/pleRuntimeConfig.h"
#include "runtime/config/runtimeBenchmarkOptions.h"
#include "runtime/multiDevice/parallelConfig.h"
#include "runtime/state/contextCache/contextCacheConfig.h"

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{

//! Caller-selected deployment policies and artifact locations, separate from resolved engine metadata.
struct DeploymentOptions
{
    std::string engineDir{};
    std::string multimodalEngineDir{};
    std::unordered_map<std::string, std::string> loraWeightsMap{};
    std::optional<SpecDecodeDraftingConfig> draftingConfig{};
    ContextCacheConfig contextCacheConfig{};
    PleRuntimeConfig pleConfig{};
    std::string checkpointDir{};
    std::string draftCheckpointDir{};
    ParallelConfig parallelConfig{};
    std::optional<std::filesystem::path> runtimeCachePath{}; //!< TRT-RTX runtime cache file path.
    RuntimeBenchmarkOptions benchmark{};
    bool offloadEmbedding{false}; //!< Keep embedding storage in CUDA-pinned host memory, using the same GPU lookup.
};

} // namespace rt
} // namespace trt_edgellm
