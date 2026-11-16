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
#include <tensorrt_edgellm/common/logger.h>
#include <tensorrt_edgellm/common/trtUtils.h>
#include <tensorrt_edgellm/profiling/timer.h>
#include <tensorrt_edgellm/runtime/llmRuntimeUtils.h>

#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string_view>

namespace trt_edgellm
{
namespace exampleUtils
{
//! Noncryptographic FNV-1a-64 diagnostic fingerprint of the exact byte sequence.
std::string fingerprintBytes(std::string_view bytes);
//! FNV-1a-64 of token IDs encoded individually as little-endian int32, without separators.
std::string fingerprintTokenIds(std::vector<int32_t> const& tokens);
//! FNV-1a-64 of file contents, read outside measured execution.
std::string fingerprintFile(std::filesystem::path const& path);

//! Serialize actual request work and its stage/semantic-forward invocations; never records prompt text.
nlohmann::json makeBenchSample(int32_t iteration, int32_t workload, double wallTimeMs,
    rt::LLMGenerationResponse const& response, std::vector<timer::TimingSample> const& timings,
    std::vector<timer::AcceptanceSample> const& acceptance = {});
//! Request and per-invocation statistics use the shared profile formatter definitions.
nlohmann::json summarizeBenchSamples(nlohmann::json const& samples);
//! Write authoritative JSON and derived request/invocation CSV files into a new directory.
void writeBenchReport(std::filesystem::path const& directory, nlohmann::json const& report);
} // namespace exampleUtils
} // namespace trt_edgellm
