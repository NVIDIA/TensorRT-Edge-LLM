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

#include "profileFormatter.h"
#include "memoryMonitor.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cuda_runtime.h>
#include <future>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <tensorrt_edgellm/common/checkMacros.h>
#include <tensorrt_edgellm/common/logger.h>
#include <tensorrt_edgellm/common/tensor.h>
#include <tensorrt_edgellm/profiling/timer.h>
#include <tensorrt_edgellm/runtime/state/contextCache/contextCacheMetrics.h>
#include <thread>

using namespace trt_edgellm;

namespace
{

//! Utility function for calculating prefill tokens per second
float getPrefillTokensPerSecond(metrics::LLMPrefillMetrics const& prefillMetrics)
{
    auto timingData = gTimer.getTimingData(metrics::StageNames::kLLM_PREFILL);
    if (!timingData || timingData->getTotalGpuTimeMs() <= 0.0f)
    {
        return 0.0f;
    }

    int64_t totalTokens = prefillMetrics.reusedTokens + prefillMetrics.computedTokens;
    if (totalTokens > 0)
    {
        return static_cast<float>(totalTokens) / (timingData->getTotalGpuTimeMs() / 1000.0f);
    }
    return 0.0f;
}

float getGenerationTokensPerSecond(metrics::LLMGenerationMetrics const& generationMetrics)
{
    auto timingData = gTimer.getTimingData(metrics::StageNames::kLLM_GENERATION);
    if (!timingData || timingData->getTotalGpuTimeMs() <= 0.0f)
    {
        return 0.0f;
    }

    if (generationMetrics.generatedTokens > 0)
    {
        return static_cast<float>(generationMetrics.generatedTokens) / (timingData->getTotalGpuTimeMs() / 1000.0f);
    }
    return 0.0f;
}

//! Utility function for calculating prefill average time per token
float getPrefillAverageTimePerToken(metrics::LLMPrefillMetrics const& prefillMetrics)
{
    auto timingData = gTimer.getTimingData(metrics::StageNames::kLLM_PREFILL);
    if (!timingData || timingData->getTotalGpuTimeMs() <= 0.0f)
    {
        return 0.0f;
    }

    int64_t totalTokens = prefillMetrics.reusedTokens + prefillMetrics.computedTokens;
    if (totalTokens > 0)
    {
        return timingData->getTotalGpuTimeMs() / totalTokens;
    }
    return 0.0f;
}

//! Utility function for calculating prefill average tokens per run
float getPrefillAverageTokensPerRun(metrics::LLMPrefillMetrics const& prefillMetrics)
{
    return static_cast<float>(prefillMetrics.reusedTokens + prefillMetrics.computedTokens)
        / prefillMetrics.getTotalRuns();
}

//! Utility function for calculating prefill average time per run
float getPrefillAverageTimePerRun(metrics::LLMPrefillMetrics const& prefillMetrics)
{
    auto timingData = gTimer.getTimingData(metrics::StageNames::kLLM_PREFILL);
    if (!timingData || timingData->getAverageTimeMs() <= 0.0f)
    {
        return 0.0f;
    }

    return timingData->getAverageTimeMs();
}

float getGenerationAverageTimePerToken(metrics::LLMGenerationMetrics const& generationMetrics)
{
    auto timingData = gTimer.getTimingData(metrics::StageNames::kLLM_GENERATION);
    if (!timingData || timingData->getTotalGpuTimeMs() <= 0.0f)
    {
        return 0.0f;
    }

    if (generationMetrics.generatedTokens > 0)
    {
        return timingData->getTotalGpuTimeMs() / generationMetrics.generatedTokens;
    }
    return 0.0f;
}

//! Utility function for calculating multimodal average time per token.
//! Includes encoder execution and visual-shard communication.
float getMultimodalAverageTimePerToken(metrics::MultimodalMetrics const& multimodalMetrics)
{
    float totalGpuTimeMs = 0.0f;

    auto legacyData = gTimer.getTimingData(metrics::StageNames::kMULTIMODAL_PROCESSING);
    if (legacyData)
    {
        totalGpuTimeMs += legacyData->getTotalGpuTimeMs();
    }
    auto visionData = gTimer.getTimingData(metrics::StageNames::kVISION_ENCODER);
    if (visionData)
    {
        totalGpuTimeMs += visionData->getTotalGpuTimeMs();
    }
    auto gatherData = gTimer.getTimingData(metrics::StageNames::kVISION_GATHER);
    if (gatherData)
    {
        totalGpuTimeMs += gatherData->getTotalGpuTimeMs();
    }
    auto audioData = gTimer.getTimingData(metrics::StageNames::kAUDIO_ENCODER);
    if (audioData)
    {
        totalGpuTimeMs += audioData->getTotalGpuTimeMs();
    }

    if (totalGpuTimeMs <= 0.0f)
    {
        return 0.0f;
    }

    int64_t totalTokens = multimodalMetrics.totalImageTokens + multimodalMetrics.totalAudioTokens;
    if (totalTokens > 0)
    {
        return totalGpuTimeMs / totalTokens;
    }
    return 0.0f;
}

std::string getSpecDecodeDisplayName(char const* strategyName)
{
    std::string const name = strategyName ? strategyName : "";
    if (name == "mtp")
    {
        return "MTP";
    }
    if (name == "eagle")
    {
        return "Eagle";
    }
    return "SpecDecode";
}

std::string getSpecDecodeGenerationJsonKey(char const* strategyName)
{
    std::string const name = strategyName ? strategyName : "";
    if (name == "mtp")
    {
        return "mtp_generation";
    }
    if (name == "eagle")
    {
        return "eagle_generation";
    }
    return "spec_decode_generation";
}

//! Utility function for calculating speculative decoding overall tokens per second (excluding base model prefill)
float getSpecDecodeOverallTokensPerSecond(metrics::SpecDecodeGenerationMetrics const& specDecodeGenerationMetrics)
{
    if (specDecodeGenerationMetrics.totalGeneratedTokens <= 0)
    {
        return 0.0f;
    }

    // Calculate total time for all speculative decoding stages except base prefill.
    float totalTimeMs = 0.0f;

    auto draftPrefillData = gTimer.getTimingData(metrics::StageNames::kSPEC_DECODE_DRAFT_PREFILL);
    if (draftPrefillData)
    {
        totalTimeMs += draftPrefillData->getTotalGpuTimeMs();
    }

    auto constructDraftProposalData = gTimer.getTimingData(metrics::StageNames::kSPEC_DECODE_DRAFT_PROPOSAL);
    if (constructDraftProposalData)
    {
        totalTimeMs += constructDraftProposalData->getTotalGpuTimeMs();
    }

    auto baseVerificationData = gTimer.getTimingData(metrics::StageNames::kSPEC_DECODE_BASE_VERIFICATION);
    if (baseVerificationData)
    {
        totalTimeMs += baseVerificationData->getTotalGpuTimeMs();
    }

    auto draftAcceptData = gTimer.getTimingData(metrics::StageNames::kSPEC_DECODE_DRAFT_ACCEPT);
    if (draftAcceptData)
    {
        totalTimeMs += draftAcceptData->getTotalGpuTimeMs();
    }

    if (totalTimeMs > 0.0f)
    {
        return static_cast<float>(specDecodeGenerationMetrics.totalGeneratedTokens) / (totalTimeMs / 1000.0f);
    }
    return 0.0f;
}

//! Average accept length: generated tokens per drafting-verification iteration
float getSpecDecodeAverageAcceptLength(metrics::SpecDecodeGenerationMetrics const& specDecodeGenerationMetrics)
{
    if (specDecodeGenerationMetrics.totalIterations <= 0)
    {
        return 0.0f;
    }
    return static_cast<float>(specDecodeGenerationMetrics.totalGeneratedTokens)
        / static_cast<float>(specDecodeGenerationMetrics.totalIterations);
}

//! Acceptance rate in [0, 1]: draft tokens accepted by verification over draft tokens
//! proposed, bonus token excluded on both sides.
float getSpecDecodeAcceptanceRate(metrics::SpecDecodeGenerationMetrics const& specDecodeGenerationMetrics)
{
    if (specDecodeGenerationMetrics.totalProposedDraftTokens <= 0)
    {
        return 0.0f;
    }

    return static_cast<float>(specDecodeGenerationMetrics.totalAcceptedDraftTokens)
        / static_cast<float>(specDecodeGenerationMetrics.totalProposedDraftTokens);
}

//! Helper function to append timing data for a stage to an ostream
void appendStageTimingData(std::ostream& summary, std::string const& stageName, std::string const& displayName)
{
    auto timingData = gTimer.getTimingData(stageName);
    if (timingData && timingData->getTotalRuns() > 0)
    {
        summary << displayName << " - Total Runs: " << timingData->getTotalRuns() << ", Total GPU Time: " << std::fixed
                << std::setprecision(2) << timingData->getTotalGpuTimeMs()
                << " ms, Average: " << timingData->getAverageTimeMs() << " ms" << std::endl;
    }
}

} // anonymous namespace

StatisticalAnalysis StatisticalAnalysis::calculate(std::vector<float> const& data)
{
    StatisticalAnalysis stats;
    if (data.empty())
    {
        return stats;
    }

    stats.count = data.size();
    stats.mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();

    auto minmax = std::minmax_element(data.begin(), data.end());
    stats.min = static_cast<double>(*minmax.first);
    stats.max = static_cast<double>(*minmax.second);

    double variance = 0.0;
    for (float value : data)
    {
        double dValue = static_cast<double>(value);
        variance += (dValue - stats.mean) * (dValue - stats.mean);
    }
    stats.stddev = std::sqrt(variance / data.size());

    std::vector<float> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    size_t size = sorted_data.size();
    stats.median = static_cast<double>(sorted_data[size / 2]);

    size_t p95_index = std::min(static_cast<size_t>(size * 0.95), size - 1);
    size_t p99_index = std::min(static_cast<size_t>(size * 0.99), size - 1);

    stats.p95 = static_cast<double>(sorted_data[p95_index]);
    stats.p99 = static_cast<double>(sorted_data[p99_index]);

    return stats;
}

void outputPrefillProfile(std::ostream& output, metrics::LLMPrefillMetrics const& prefillMetrics)
{
    if (prefillMetrics.getTotalRuns() > 0)
    {
        output << "=== LLM Prefill ===" << std::endl;
        output << "Reused Tokens: " << prefillMetrics.reusedTokens << std::endl;
        output << "Computed Tokens: " << prefillMetrics.computedTokens << std::endl;
        if (prefillMetrics.prunedTokens > 0)
        {
            output << "Pruned Tokens (visual-token pruning): " << prefillMetrics.prunedTokens << std::endl;
        }
        output << "Average Tokens per Run: " << std::fixed << std::setprecision(2)
               << getPrefillAverageTokensPerRun(prefillMetrics) << std::endl;
        output << "Average Time per Run: " << std::fixed << std::setprecision(4)
               << getPrefillAverageTimePerRun(prefillMetrics) << " ms" << std::endl;
        output << "Tokens/Second: " << std::fixed << std::setprecision(1) << getPrefillTokensPerSecond(prefillMetrics)
               << std::endl;
        output << "Average Time per Token: " << std::fixed << std::setprecision(4)
               << getPrefillAverageTimePerToken(prefillMetrics) << " ms" << std::endl;
        appendStageTimingData(output, metrics::StageNames::kLLM_PREFILL, "LLM Prefill");
    }
}

void outputContextCacheProfile(std::ostream& output, rt::ContextCacheMetrics const& cacheMetrics)
{
    output << "=== Context Cache ===" << std::endl;
    output << "Sequences: admitted=" << cacheMetrics.admittedSequences << ", hits=" << cacheMetrics.hitSequences
           << ", media_aware=" << cacheMetrics.mediaAwareSequences
           << ", lookup_bypass=" << cacheMetrics.lookupBypassSequences
           << ", forced_cold=" << cacheMetrics.forcedColdSequences << std::endl;
    output << "Tokens: matched=" << cacheMetrics.matchedTokens << ", reused=" << cacheMetrics.reusedTokens << std::endl;
    output << "Plans: standard=" << cacheMetrics.standardPlans
           << ", no_reusable_prefix=" << cacheMetrics.noReusablePrefixPlans
           << ", full_input_rewind=" << cacheMetrics.fullInputRewindPlans << std::endl;
    output << "Publications: attempts=" << cacheMetrics.publicationAttempts
           << ", committed=" << cacheMetrics.committedPublications << ", existing=" << cacheMetrics.existingPublications
           << std::endl;
    output << "Records: current=" << cacheMetrics.currentRecords << ", evicted=" << cacheMetrics.evictedRecords
           << std::endl;
    output << "Base KV pages: free=" << cacheMetrics.baseKvPages.free
           << ", capacity=" << cacheMetrics.baseKvPages.capacity << std::endl;
    output << "Draft KV pages: free=" << cacheMetrics.draftKvPages.free
           << ", capacity=" << cacheMetrics.draftKvPages.capacity << std::endl;
    output << "Recurrent snapshots: free=" << cacheMetrics.recurrentSnapshots.free
           << ", capacity=" << cacheMetrics.recurrentSnapshots.capacity << std::endl;
    output << "Partial KV snapshots: free=" << cacheMetrics.partialKvSnapshots.free
           << ", capacity=" << cacheMetrics.partialKvSnapshots.capacity << std::endl;
    output << "SWA KV pages: free=" << cacheMetrics.swaKvPages.free << ", capacity=" << cacheMetrics.swaKvPages.capacity
           << std::endl;
    output << "Degradation: hybrid_snapshot_pressure_skips=" << cacheMetrics.hybridSnapshotPressureSkips << std::endl;
}

void outputGenerationProfile(std::ostream& output, metrics::LLMGenerationMetrics const& generationMetrics)
{
    output << "=== LLM Generation (Excluding sampling after prefill) ===" << std::endl;

    if (generationMetrics.getTotalRuns() > 0)
    {
        output << "Generated Tokens: " << generationMetrics.generatedTokens << std::endl;
        output << "Average Tokens per Run: " << std::fixed << std::setprecision(2)
               << static_cast<float>(generationMetrics.generatedTokens) / generationMetrics.getTotalRuns() << std::endl;
        output << "Tokens/Second: " << std::fixed << std::setprecision(1)
               << getGenerationTokensPerSecond(generationMetrics) << std::endl;
        output << "Average Time per Token: " << std::fixed << std::setprecision(4)
               << getGenerationAverageTimePerToken(generationMetrics) << " ms" << std::endl;
        appendStageTimingData(output, metrics::StageNames::kLLM_GENERATION, "LLM Generation");
    }
    else
    {
        output << "max_generate_length = 1, the model only runs the prefill stage." << std::endl;
    }
}

void outputSpecDecodeGenerationProfile(std::ostream& output,
    metrics::SpecDecodeGenerationMetrics const& specDecodeGenerationMetrics, char const* strategyName)
{
    if (specDecodeGenerationMetrics.getTotalRuns() > 0)
    {
        output << "=== " << getSpecDecodeDisplayName(strategyName) << " Generation ===" << std::endl;
        output << "Total Iterations: " << specDecodeGenerationMetrics.totalIterations << std::endl;
        output << "Total Generated Tokens: " << specDecodeGenerationMetrics.totalGeneratedTokens << std::endl;
        output << "Average Tokens per Run: " << std::fixed << std::setprecision(2)
               << static_cast<float>(specDecodeGenerationMetrics.totalGeneratedTokens)
                / specDecodeGenerationMetrics.getTotalRuns()
               << std::endl;
        output << "Average Accept Length: " << std::fixed << std::setprecision(2)
               << getSpecDecodeAverageAcceptLength(specDecodeGenerationMetrics) << std::endl;
        output << "Acceptance Rate: " << std::fixed << std::setprecision(3)
               << getSpecDecodeAcceptanceRate(specDecodeGenerationMetrics) << std::endl;
        output << "Overall Tokens/Second (excluding base prefill): " << std::fixed << std::setprecision(1)
               << getSpecDecodeOverallTokensPerSecond(specDecodeGenerationMetrics) << std::endl;

        // Individual speculative decoding stage timing.
        appendStageTimingData(output, metrics::StageNames::kSPEC_DECODE_DRAFT_PREFILL, "Draft Model Prefill");
        appendStageTimingData(output, metrics::StageNames::kSPEC_DECODE_DRAFT_PROPOSAL, "Construct Draft Proposal");
        appendStageTimingData(output, metrics::StageNames::kSPEC_DECODE_BASE_VERIFICATION, "Base Model Verification");
        appendStageTimingData(output, metrics::StageNames::kSPEC_DECODE_DRAFT_ACCEPT, "Draft Model Accept Token");
    }
}

void outputMultimodalProfile(std::ostream& output, metrics::MultimodalMetrics const& multimodalMetrics)
{
    if (multimodalMetrics.getTotalRuns() > 0)
    {
        output << "=== Multimodal Processing ===" << std::endl;

        // Show audio stats if present (Qwen3-Omni)
        if (multimodalMetrics.totalAudios > 0)
        {
            output << "Total Audio Clips: " << multimodalMetrics.totalAudios << std::endl;
            output << "Total Audio Tokens: " << multimodalMetrics.totalAudioTokens << std::endl;
        }

        // Show image stats if present
        if (multimodalMetrics.totalImages > 0)
        {
            output << "Total Images: " << multimodalMetrics.totalImages << std::endl;
            output << "Total Image Tokens: " << multimodalMetrics.totalImageTokens << std::endl;
        }

        // Show combined stats
        int64_t totalTokens = multimodalMetrics.totalImageTokens + multimodalMetrics.totalAudioTokens;
        if (totalTokens > 0)
        {
            output << "Total Multimodal Tokens: " << totalTokens << std::endl;
        }

        output << "Average Time per Token: " << std::fixed << std::setprecision(4)
               << getMultimodalAverageTimePerToken(multimodalMetrics) << " ms" << std::endl;
        appendStageTimingData(output, metrics::StageNames::kMULTIMODAL_PROCESSING, "Multimodal Processing");
        appendStageTimingData(output, metrics::StageNames::kVISION_ENCODER, "Vision Encoder");
        appendStageTimingData(output, metrics::StageNames::kVISION_GATHER, "Vision Gather");
        appendStageTimingData(output, metrics::StageNames::kAUDIO_ENCODER, "Audio Encoder");
    }
}

void outputTalkerProfile(std::ostream& output, metrics::MultimodalMetrics const& talkerMetrics)
{
    if (talkerMetrics.getTotalRuns() > 0)
    {
        output << "=== Talker Audio Generation ===" << std::endl;
        output << "Total Audio Outputs: " << talkerMetrics.totalAudios << std::endl;
        output << "Total Audio Codes: " << talkerMetrics.totalAudioTokens << std::endl;

        if (talkerMetrics.totalAudioTokens > 0)
        {
            output << "Average Codes per Output: " << std::fixed << std::setprecision(1)
                   << static_cast<float>(talkerMetrics.totalAudioTokens) / talkerMetrics.totalAudios << std::endl;
        }

        appendStageTimingData(output, metrics::StageNames::kTALKER_GENERATION, "Talker Generation");
        appendStageTimingData(output, metrics::StageNames::kCODEPREDICTOR_PREFILL, "CodePredictor Prefill");
        appendStageTimingData(output, metrics::StageNames::kCODEPREDICTOR_GENERATION, "CodePredictor Generation");
        appendStageTimingData(output, metrics::StageNames::kCODE2WAV, "Code2Wav");
    }
}

void outputOmniProfile(std::ostream& output, metrics::OmniTalkerMetrics const& talkerMetrics,
    metrics::OmniLatencyMetrics const& latencyMetrics)
{
    if (talkerMetrics.getTotalRuns() == 0)
    {
        return;
    }

    output << "=== Audio Latency ===" << std::endl;
    output << "Mode: " << (talkerMetrics.isStreaming ? "streaming" : "sequential") << std::endl;
    if (latencyMetrics.timeToFirstAudioCodeMs > 0.0f)
    {
        output << "Time to First Audio Code (TTFA): " << std::fixed << std::setprecision(2)
               << latencyMetrics.timeToFirstAudioCodeMs << " ms" << std::endl;
    }
    if (latencyMetrics.endToEndMs > 0.0f)
    {
        output << "End-to-End (request to audio output): " << std::fixed << std::setprecision(2)
               << latencyMetrics.endToEndMs << " ms" << std::endl;
    }
    if (latencyMetrics.realTimeFactor > 0.0f)
    {
        output << "Real-Time Factor (RTF): " << std::fixed << std::setprecision(3) << latencyMetrics.realTimeFactor
               << (latencyMetrics.realTimeFactor < 1.0f ? " (faster than real-time)" : " (slower than real-time)")
               << std::endl;
    }
    if (latencyMetrics.audioDurationSeconds > 0.0f)
    {
        output << "Audio Duration: " << std::fixed << std::setprecision(2) << latencyMetrics.audioDurationSeconds
               << " s (" << latencyMetrics.audioSamples << " samples @ " << latencyMetrics.sampleRate << " Hz)"
               << std::endl;
    }
    output << "Audio Frames: " << talkerMetrics.totalFrames << " (RVQ Codes: " << talkerMetrics.totalRvqCodes << ")"
           << std::endl;
    output << "Exit Reason: " << talkerMetrics.exitReason << std::endl;
    appendStageTimingData(output, metrics::StageNames::kTALKER_PREFILL, "Talker Prefill");
}

void outputMemoryProfile(std::ostream& output, MemoryMonitor const& memoryMonitor)
{
    output << "=== Memory Usage ===" << std::endl;
    size_t const peakGpuMemoryBytes = memoryMonitor.getPeakGpuMemory();
    size_t const peakCpuMemoryBytes = memoryMonitor.getPeakCpuMemory();
    output << "Peak GPU Memory: " << std::fixed << std::setprecision(2) << rt::utils::toMB(peakGpuMemoryBytes)
           << " MB (" << peakGpuMemoryBytes << " bytes)" << std::endl;
    output << "Peak CPU Memory: " << std::fixed << std::setprecision(2) << rt::utils::toMB(peakCpuMemoryBytes)
           << " MB (" << peakCpuMemoryBytes << " bytes)" << std::endl;
    output << "GPU Memory Metric: " << memoryMonitor.getGpuMemoryMetric() << std::endl;
}

void addJsonPrefillSummary(nlohmann::json& summary, metrics::LLMPrefillMetrics const& prefillMetrics)
{
    if (prefillMetrics.getTotalRuns() > 0)
    {
        summary["prefill"]
            = {{"total_runs", prefillMetrics.getTotalRuns()}, {"reused_tokens", prefillMetrics.reusedTokens},
                {"computed_tokens", prefillMetrics.computedTokens}, {"pruned_tokens", prefillMetrics.prunedTokens},
                {"average_tokens_per_run", getPrefillAverageTokensPerRun(prefillMetrics)},
                {"average_time_per_run_ms", getPrefillAverageTimePerRun(prefillMetrics)},
                {"tokens_per_second", getPrefillTokensPerSecond(prefillMetrics)},
                {"average_time_per_token_ms", getPrefillAverageTimePerToken(prefillMetrics)}};
    }
}

void addJsonContextCacheSummary(nlohmann::json& summary, rt::ContextCacheMetrics const& cacheMetrics)
{
    auto pool = [](rt::ContextCachePoolMetrics const& metrics) {
        return nlohmann::json{{"free", metrics.free}, {"capacity", metrics.capacity}};
    };
    summary["context_cache"] = {{"admitted_sequences", cacheMetrics.admittedSequences},
        {"hit_sequences", cacheMetrics.hitSequences}, {"media_aware_sequences", cacheMetrics.mediaAwareSequences},
        {"lookup_bypass_sequences", cacheMetrics.lookupBypassSequences},
        {"forced_cold_sequences", cacheMetrics.forcedColdSequences}, {"matched_tokens", cacheMetrics.matchedTokens},
        {"reused_tokens", cacheMetrics.reusedTokens},
        {"plans",
            {{"standard", cacheMetrics.standardPlans}, {"no_reusable_prefix", cacheMetrics.noReusablePrefixPlans},
                {"full_input_rewind", cacheMetrics.fullInputRewindPlans}}},
        {"publications",
            {{"attempts", cacheMetrics.publicationAttempts}, {"committed", cacheMetrics.committedPublications},
                {"existing", cacheMetrics.existingPublications},
                {"successful_endpoints", cacheMetrics.publishedEndpoints}}},
        {"hybrid",
            {{"restores", cacheMetrics.hybridRestores},
                {"snapshot_pressure_skips", cacheMetrics.hybridSnapshotPressureSkips},
                {"capture_synchronizations", cacheMetrics.hybridCaptureSynchronizations}}},
        {"speculative",
            {{"full_page_replays", cacheMetrics.specFullPageReplays},
                {"pair_publications", cacheMetrics.specPairPublications}}},
        {"planning_nanoseconds", cacheMetrics.planningNanoseconds},
        {"records", {{"current", cacheMetrics.currentRecords}, {"evicted", cacheMetrics.evictedRecords}}},
        {"resource_pools",
            {{"base_kv_pages", pool(cacheMetrics.baseKvPages)}, {"draft_kv_pages", pool(cacheMetrics.draftKvPages)},
                {"recurrent_snapshots", pool(cacheMetrics.recurrentSnapshots)},
                {"partial_kv_snapshots", pool(cacheMetrics.partialKvSnapshots)},
                {"swa_kv_pages", pool(cacheMetrics.swaKvPages)}}},
        {"reclaimed_resources",
            {{"base_kv_pages", cacheMetrics.reclaimedBaseKvPages},
                {"draft_kv_pages", cacheMetrics.reclaimedDraftKvPages},
                {"recurrent_snapshots", cacheMetrics.reclaimedRecurrentSnapshots},
                {"partial_kv_snapshots", cacheMetrics.reclaimedPartialKvSnapshots},
                {"swa_kv_pages", cacheMetrics.reclaimedSwaKvPages}}}};
}

void addJsonGenerationSummary(nlohmann::json& summary, metrics::LLMGenerationMetrics const& generationMetrics)
{
    if (generationMetrics.getTotalRuns() > 0)
    {
        summary["generation"] = {{"total_runs", generationMetrics.getTotalRuns()},
            {"generated_tokens", generationMetrics.generatedTokens},
            {"average_tokens_per_run",
                static_cast<float>(generationMetrics.generatedTokens) / generationMetrics.getTotalRuns()},
            {"tokens_per_second", getGenerationTokensPerSecond(generationMetrics)},
            {"average_time_per_token_ms", getGenerationAverageTimePerToken(generationMetrics)}};
    }
}

void addJsonSpecDecodeGenerationSummary(nlohmann::json& summary,
    metrics::SpecDecodeGenerationMetrics const& specDecodeGenerationMetrics, char const* strategyName)
{
    if (specDecodeGenerationMetrics.getTotalRuns() > 0)
    {
        summary[getSpecDecodeGenerationJsonKey(strategyName)]
            = {{"total_runs", specDecodeGenerationMetrics.getTotalRuns()},
                {"total_iterations", specDecodeGenerationMetrics.totalIterations},
                {"total_generated_tokens", specDecodeGenerationMetrics.totalGeneratedTokens},
                {"average_tokens_per_run",
                    static_cast<float>(specDecodeGenerationMetrics.totalGeneratedTokens)
                        / specDecodeGenerationMetrics.getTotalRuns()},
                {"average_accept_length", getSpecDecodeAverageAcceptLength(specDecodeGenerationMetrics)},
                {"acceptance_rate", getSpecDecodeAcceptanceRate(specDecodeGenerationMetrics)},
                {"overall_tokens_per_second_excluding_base_prefill",
                    getSpecDecodeOverallTokensPerSecond(specDecodeGenerationMetrics)}};
    }
}

void addJsonMultimodalSummary(nlohmann::json& summary, metrics::MultimodalMetrics const& multimodalMetrics)
{
    if (multimodalMetrics.getTotalRuns() > 0)
    {
        int64_t totalTokens = multimodalMetrics.totalImageTokens + multimodalMetrics.totalAudioTokens;

        summary["multimodal"] = {{"total_runs", multimodalMetrics.getTotalRuns()},
            {"total_images", multimodalMetrics.totalImages}, {"total_image_tokens", multimodalMetrics.totalImageTokens},
            {"total_audios", multimodalMetrics.totalAudios}, {"total_audio_tokens", multimodalMetrics.totalAudioTokens},
            {"total_multimodal_tokens", totalTokens},
            {"average_time_per_token_ms", getMultimodalAverageTimePerToken(multimodalMetrics)}};
    }
}

void addJsonTalkerSummary(nlohmann::json& summary, metrics::MultimodalMetrics const& talkerMetrics)
{
    if (talkerMetrics.getTotalRuns() > 0)
    {
        summary["talker"] = {{"total_runs", talkerMetrics.getTotalRuns()},
            {"total_audio_outputs", talkerMetrics.totalAudios}, {"total_audio_codes", talkerMetrics.totalAudioTokens},
            {"average_codes_per_output",
                talkerMetrics.totalAudios > 0
                    ? static_cast<float>(talkerMetrics.totalAudioTokens) / talkerMetrics.totalAudios
                    : 0.0f}};
    }
}

void addJsonOmniStageExtensions(nlohmann::json& summary, metrics::OmniTalkerMetrics const& talkerMetrics,
    metrics::OmniLatencyMetrics const& latencyMetrics)
{
    if (!summary.contains("stages") || talkerMetrics.getTotalRuns() == 0)
    {
        return;
    }

    for (auto& stageJson : summary["stages"])
    {
        if (stageJson["stage_id"] == metrics::StageNames::kTALKER_GENERATION)
        {
            stageJson["mode"] = talkerMetrics.isStreaming ? "streaming" : "sequential";
            stageJson["audio_frames"] = talkerMetrics.totalFrames;
            stageJson["total_rvq_codes"] = talkerMetrics.totalRvqCodes;
            stageJson["exit_reason"] = talkerMetrics.exitReason;

            auto talkerGenData = gTimer.getTimingData(metrics::StageNames::kTALKER_GENERATION);
            float talkerGenMs = talkerGenData ? talkerGenData->getTotalGpuTimeMs() : 0.0f;

            if (talkerMetrics.totalFrames > 0 && talkerGenMs > 0.0f)
            {
                stageJson["talker_frames_per_second"]
                    = static_cast<float>(talkerMetrics.totalFrames) / (talkerGenMs / 1000.0f);
            }

            if (latencyMetrics.timeToFirstAudioCodeMs > 0.0f)
            {
                stageJson["time_to_first_audio_code_ms"] = latencyMetrics.timeToFirstAudioCodeMs;
            }

            if (latencyMetrics.timeToFirstPlayableAudioMs > 0.0f)
            {
                stageJson["time_to_first_playable_audio_ms"] = latencyMetrics.timeToFirstPlayableAudioMs;
            }
            if (latencyMetrics.realTimeFactor > 0.0f)
            {
                stageJson["real_time_factor"] = latencyMetrics.realTimeFactor;
            }
            if (latencyMetrics.endToEndMs > 0.0f)
            {
                stageJson["end_to_end_ms"] = latencyMetrics.endToEndMs;
            }
            if (latencyMetrics.audioSamples > 0)
            {
                stageJson["audio_duration_seconds"] = latencyMetrics.audioDurationSeconds;
                stageJson["audio_samples"] = latencyMetrics.audioSamples;
                stageJson["sample_rate"] = latencyMetrics.sampleRate;
            }
        }

        if (stageJson["stage_id"] == metrics::StageNames::kCODEPREDICTOR_GENERATION)
        {
            int64_t genCodes = talkerMetrics.totalRvqCodes - talkerMetrics.totalFrames;
            stageJson["total_codes"] = genCodes;
            float gpuMs = stageJson.value("total_gpu_time_ms", 0.0f);
            if (genCodes > 0 && gpuMs > 0.0f)
            {
                stageJson["codes_per_second"] = static_cast<float>(genCodes) / (gpuMs / 1000.0f);
            }
        }
    }
}

nlohmann::json timingSampleToJson(trt_edgellm::timer::TimingSample const& sample)
{
    return {{"stage", sample.stageId}, {"parent_stage", sample.parentStage}, {"request_id", sample.requestId},
        {"round", sample.round}, {"batch_size", sample.batchSize}, {"query_tokens", sample.queryTokens},
        {"selected_tokens", sample.selectedTokens}, {"past_lengths", sample.pastLengths}, {"forward", sample.forward},
        {"forward_index", sample.forwardIndex}, {"gpu_ms", sample.gpuTimeMs}};
}

nlohmann::json acceptanceSampleToJson(trt_edgellm::timer::AcceptanceSample const& sample)
{
    return {{"request_id", sample.requestId}, {"round", sample.round}, {"sequence_index", sample.sequenceIndex},
        {"proposed_drafts", sample.proposedDrafts}, {"accepted_drafts", sample.acceptedDrafts},
        {"output_tokens", sample.outputTokens}, {"bonus_tokens", sample.bonusTokens},
        {"base_materialized_length", sample.baseMaterializedLength}, {"provenance_exact", sample.provenanceExact},
        {"proposed_drafts_exact", sample.proposedExact}};
}

void addJsonTimingStages(nlohmann::json& summary)
{
    summary["acceptance"] = nlohmann::json::array();
    for (auto const& sample : gTimer.getAcceptanceSamples())
    {
        summary["acceptance"].push_back(acceptanceSampleToJson(sample));
    }
    summary["invocations"] = nlohmann::json::array();
    for (auto const& sample : gTimer.getSamples())
    {
        summary["invocations"].push_back(timingSampleToJson(sample));
    }
    summary["stages"] = nlohmann::json::array();
    for (auto const& [stageId, timingData] : gTimer.getAllTimingData())
    {
        if (timingData.gpuTimesMs.empty())
        {
            continue;
        }
        nlohmann::json stageJson;
        stageJson["stage_id"] = stageId;
        stageJson["total_runs"] = timingData.getTotalRuns();
        stageJson["total_gpu_time_ms"] = timingData.getTotalGpuTimeMs();
        stageJson["average_time_per_run_ms"] = timingData.getAverageTimeMs();

        auto gpuStats = StatisticalAnalysis::calculate(timingData.gpuTimesMs);
        stageJson["gpu_time_stats"] = {{"count", gpuStats.count}, {"min_ms", gpuStats.min}, {"max_ms", gpuStats.max},
            {"mean_ms", gpuStats.mean}, {"median_ms", gpuStats.median}, {"p95_ms", gpuStats.p95},
            {"p99_ms", gpuStats.p99}, {"stddev_ms", gpuStats.stddev}};

        summary["stages"].push_back(stageJson);
    }
}

void addJsonMemorySummary(nlohmann::json& summary, MemoryMonitor const& memoryMonitor)
{
    size_t const peakGpuMemoryBytes = memoryMonitor.getPeakGpuMemory();
    size_t const peakCpuMemoryBytes = memoryMonitor.getPeakCpuMemory();
    summary["peak_gpu_memory_bytes"] = peakGpuMemoryBytes;
    summary["peak_gpu_memory_mb"] = rt::utils::toMB(peakGpuMemoryBytes);
    summary["peak_cpu_memory_bytes"] = peakCpuMemoryBytes;
    summary["peak_cpu_memory_mb"] = rt::utils::toMB(peakCpuMemoryBytes);
    summary["gpu_memory_metric"] = memoryMonitor.getGpuMemoryMetric();
}

void addJsonMemorySummary(nlohmann::json& summary, MemoryMonitor const& memoryMonitor, int32_t tpSize,
    std::string const& launchMode, std::vector<RankMemorySummary> const& rankMemory)
{
    addJsonMemorySummary(summary, memoryMonitor);

    nlohmann::json memoryJson{
        {"launch_mode", launchMode}, {"tp_size", tpSize}, {"gpu_memory_metric", memoryMonitor.getGpuMemoryMetric()}};
    nlohmann::json rankMemoryJson = nlohmann::json::array();
    size_t totalRankGpuMemoryBytes = 0;
    size_t maxRankGpuMemoryBytes = 0;
    size_t totalRankCpuMemoryBytes = 0;
    size_t maxRankCpuMemoryBytes = 0;
    for (RankMemorySummary const& rank : rankMemory)
    {
        rankMemoryJson.push_back(
            {{"rank", rank.rank}, {"device", rank.device}, {"peak_gpu_memory_bytes", rank.peakGpuMemoryBytes},
                {"peak_gpu_memory_mb", rt::utils::toMB(rank.peakGpuMemoryBytes)},
                {"peak_cpu_memory_bytes", rank.peakCpuMemoryBytes},
                {"peak_cpu_memory_mb", rt::utils::toMB(rank.peakCpuMemoryBytes)}});
        totalRankGpuMemoryBytes += rank.peakGpuMemoryBytes;
        maxRankGpuMemoryBytes = std::max(maxRankGpuMemoryBytes, rank.peakGpuMemoryBytes);
        totalRankCpuMemoryBytes += rank.peakCpuMemoryBytes;
        maxRankCpuMemoryBytes = std::max(maxRankCpuMemoryBytes, rank.peakCpuMemoryBytes);
    }

    if (!rankMemoryJson.empty())
    {
        memoryJson["rank_memory"] = std::move(rankMemoryJson);
        memoryJson["peak_total_rank_gpu_memory_bytes"] = totalRankGpuMemoryBytes;
        memoryJson["peak_total_rank_gpu_memory_mb"] = rt::utils::toMB(totalRankGpuMemoryBytes);
        memoryJson["peak_max_rank_gpu_memory_bytes"] = maxRankGpuMemoryBytes;
        memoryJson["peak_max_rank_gpu_memory_mb"] = rt::utils::toMB(maxRankGpuMemoryBytes);
        memoryJson["peak_total_rank_cpu_memory_bytes"] = totalRankCpuMemoryBytes;
        memoryJson["peak_total_rank_cpu_memory_mb"] = rt::utils::toMB(totalRankCpuMemoryBytes);
        memoryJson["peak_max_rank_cpu_memory_bytes"] = maxRankCpuMemoryBytes;
        memoryJson["peak_max_rank_cpu_memory_mb"] = rt::utils::toMB(maxRankCpuMemoryBytes);
    }
    summary["memory"] = std::move(memoryJson);
}

/**
 * @brief Sanitize a string to ensure it contains valid UTF-8 before JSON serialization
 *
 * This function detects and replaces invalid UTF-8 sequences that would cause nlohmann::json to fail.
 *
 * UTF-8 Encoding Patterns (what we're looking for):
 *   - 1-byte (ASCII):  0xxxxxxx              (0x00-0x7F)
 *   - 2-byte sequence: 110xxxxx 10xxxxxx     (0xC0-0xDF followed by 0x80-0xBF)
 *   - 3-byte sequence: 1110xxxx 10xxxxxx 10xxxxxx  (0xE0-0xEF followed by 2× 0x80-0xBF)
 *   - 4-byte sequence: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx  (0xF0-0xF7 followed by 3× 0x80-0xBF)
 *
 * Invalid patterns we detect and replace:
 *   1. Invalid start bytes: Any byte that doesn't match the patterns above (e.g., 0xFF, 0xC0, 0xF8-0xFF)
 *   2. Incomplete sequences: Start of multi-byte sequence without enough continuation bytes
 *      Example: "\xF0\x9F\x98" (3 bytes) when 4 bytes are needed for emoji
 *   3. Invalid continuation bytes: Multi-byte sequence where continuation bytes don't match 10xxxxxx pattern
 *
 * Replacement: All invalid/incomplete sequences are replaced with U+FFFD (�)
 *   - UTF-8 encoding: 0xEF 0xBF 0xBD
 *   - This is the standard Unicode replacement character for invalid/unknown characters
 *
 * @param input String that may contain invalid UTF-8 (e.g., from tokenizer decode)
 * @return String with valid UTF-8, safe for JSON serialization
 */
std::string sanitizeUtf8ForJson(std::string const& input)
{
    // Use nlohmann::json's built-in UTF-8 validation by attempting to serialize to JSON
    // UTF-8 validation happens during dump(), not during assignment
    try
    {
        nlohmann::json testJson = input;
        // Actually call dump() to trigger UTF-8 validation
        testJson.dump();
        // If successful, the string is valid UTF-8, return as-is
        return input;
    }
    catch (std::exception const& e)
    {
        // Invalid UTF-8 detected - perform byte-by-byte sanitization
        LOG_WARNING("Invalid UTF-8 detected in output: %s", e.what());

        std::string sanitized;
        sanitized.reserve(input.size());

        size_t i = 0;
        size_t len = input.length();

        while (i < len)
        {
            unsigned char c = static_cast<unsigned char>(input[i]);

            // Determine expected UTF-8 sequence length based on first byte pattern
            int64_t seqLen = 0;
            if ((c & 0b10000000) == 0b00000000)
            {
                seqLen = 1; // ASCII: 0xxxxxxx (0x00-0x7F)
            }
            else if ((c & 0b11100000) == 0b11000000)
            {
                seqLen = 2; // 2-byte: 110xxxxx (0xC0-0xDF)
            }
            else if ((c & 0b11110000) == 0b11100000)
            {
                seqLen = 3; // 3-byte: 1110xxxx (0xE0-0xEF)
            }
            else if ((c & 0b11111000) == 0b11110000)
            {
                seqLen = 4; // 4-byte: 11110xxx (0xF0-0xF7)
            }
            else
            {
                // Invalid start byte (e.g., 0xFF, 0xF8-0xFF, or continuation byte in wrong position)
                sanitized += "\xEF\xBF\xBD"; // Replace with U+FFFD (�)
                i++;
                continue;
            }

            // Check if we have enough remaining bytes for the complete sequence
            if (i + seqLen > len)
            {
                // Incomplete sequence at end (e.g., "\xF0\x9F\x98" missing 4th byte for emoji)
                LOG_WARNING("Incomplete UTF-8 sequence at position %zu (need %ld bytes, have %zu)", i, seqLen, len - i);
                sanitized += "\xEF\xBF\xBD"; // Replace with U+FFFD (�)
                break;
            }

            // Validate that all continuation bytes match the pattern 10xxxxxx (0x80-0xBF)
            bool validSequence = true;
            for (int64_t j = 1; j < seqLen; j++)
            {
                unsigned char cont = static_cast<unsigned char>(input[i + j]);
                if ((cont & 0b11000000) != 0b10000000) // Must be 10xxxxxx
                {
                    validSequence = false;
                    break;
                }
            }

            if (validSequence)
            {
                // Valid UTF-8 sequence - copy it to output
                sanitized.append(input, i, seqLen);
                i += seqLen;
            }
            else
            {
                // Invalid continuation bytes - replace with U+FFFD (�)
                sanitized += "\xEF\xBF\xBD";
                i++;
            }
        }

        LOG_WARNING("Sanitized output from %zu to %zu bytes", input.size(), sanitized.size());
        return sanitized;
    }
}
