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

#include "memoryMonitor.h"
#include "profiling/metrics.h"
#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
struct ContextCacheMetrics;
} // namespace rt
} // namespace trt_edgellm

//! Per-rank memory summary for multi-device launches.
struct RankMemorySummary
{
    int32_t rank{0};
    int32_t device{0};
    size_t peakGpuMemoryBytes{0};
    size_t peakCpuMemoryBytes{0};
};

//! Statistical analysis results for performance data.
struct StatisticalAnalysis
{
    double min{0.0};
    double max{0.0};
    double mean{0.0};
    double median{0.0};
    double p95{0.0};
    double p99{0.0};
    double stddev{0.0};
    size_t count{0};

    static StatisticalAnalysis calculate(std::vector<float> const& data);
};

//! Output prefill stage summary to ostream
void outputPrefillProfile(std::ostream& output, trt_edgellm::metrics::LLMPrefillMetrics const& prefillMetrics);

//! Output context-cache summary to ostream.
void outputContextCacheProfile(std::ostream& output, trt_edgellm::rt::ContextCacheMetrics const& cacheMetrics);

//! Output generation stage summary (naive decoding) to ostream
void outputGenerationProfile(std::ostream& output, trt_edgellm::metrics::LLMGenerationMetrics const& generationMetrics);

//! Output speculative generation stage summary to ostream
void outputSpecDecodeGenerationProfile(std::ostream& output,
    trt_edgellm::metrics::SpecDecodeGenerationMetrics const& specDecodeGenerationMetrics,
    char const* strategyName = "spec_decode");

//! Output multimodal processing summary to ostream
void outputMultimodalProfile(std::ostream& output, trt_edgellm::metrics::MultimodalMetrics const& multimodalMetrics);

//! Output Talker audio generation metrics summary to ostream
void outputTalkerProfile(std::ostream& output, trt_edgellm::metrics::MultimodalMetrics const& talkerMetrics);

//! Output memory usage summary to ostream
void outputMemoryProfile(std::ostream& output, MemoryMonitor const& memoryMonitor);

//! Output layer profiles summary to ostream
void outputLayerProfiles(std::ostream& output, bool detailed = false);

//! Add JSON for prefill stage to existing json object
void addJsonPrefillSummary(nlohmann::json& summary, trt_edgellm::metrics::LLMPrefillMetrics const& prefillMetrics);

//! Add context-cache metrics under the stable context_cache object.
void addJsonContextCacheSummary(nlohmann::json& summary, trt_edgellm::rt::ContextCacheMetrics const& cacheMetrics);

//! Add JSON for generation stage (naive decoding) to existing json object
void addJsonGenerationSummary(
    nlohmann::json& summary, trt_edgellm::metrics::LLMGenerationMetrics const& generationMetrics);

//! Add JSON for speculative generation stage to existing json object
void addJsonSpecDecodeGenerationSummary(nlohmann::json& summary,
    trt_edgellm::metrics::SpecDecodeGenerationMetrics const& specDecodeGenerationMetrics,
    char const* strategyName = "spec_decode");

//! Add JSON for multimodal processing to existing json object
void addJsonMultimodalSummary(
    nlohmann::json& summary, trt_edgellm::metrics::MultimodalMetrics const& multimodalMetrics);

//! Add JSON for Talker audio generation to existing json object (legacy, backward compat)
void addJsonTalkerSummary(nlohmann::json& summary, trt_edgellm::metrics::MultimodalMetrics const& talkerMetrics);

//! Extend talker_generation stage in stages[] with Omni latency/metadata (TTFA, RTF, audio frames, etc.)
//! Must be called AFTER addJsonTimingStages.
void addJsonOmniStageExtensions(nlohmann::json& summary, trt_edgellm::metrics::OmniTalkerMetrics const& talkerMetrics,
    trt_edgellm::metrics::OmniLatencyMetrics const& latencyMetrics);

//! Output audio latency summary to ostream
void outputOmniProfile(std::ostream& output, trt_edgellm::metrics::OmniTalkerMetrics const& talkerMetrics,
    trt_edgellm::metrics::OmniLatencyMetrics const& latencyMetrics);

//! Add JSON for all timing stages to existing json object
void addJsonTimingStages(nlohmann::json& summary);

//! Add JSON for memory usage to existing json object
void addJsonMemorySummary(nlohmann::json& summary, MemoryMonitor const& memoryMonitor);

//! Add JSON for memory usage with multi-device metadata.
void addJsonMemorySummary(nlohmann::json& summary, MemoryMonitor const& memoryMonitor, int32_t tpSize,
    std::string const& launchMode, std::vector<RankMemorySummary> const& rankMemory = {});

//! Check string for invalid UTF-8 sequences
//! Returns original string if valid, or error message if invalid UTF-8 detected
//! Logs the full original text when invalid UTF-8 is found
std::string sanitizeUtf8ForJson(std::string const& input);
