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
#include "runtime/llmRuntimeUtils.h"
#include "runtime/modelArtifacts.h"
#include "runtime/multiDevice/parallelConfig.h"
#include "runtime/state/contextCache/contextCacheConfig.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace tokenizer
{
class Tokenizer;
}

namespace rt
{

class CollectiveGroup;
class MultiDevicePluginResources;
class LLMRankRuntime;
struct VisualPrunerConfig;

//! Owns local LLM rank workers and communication groups for a runtime coordinator.
class RuntimeCoordinator
{
public:
    struct Config
    {
        std::string engineDir;
        std::string multimodalEngineDir;
        std::unordered_map<std::string, std::string> loraWeightsMap;
        ParallelConfig parallelConfig;
        std::optional<SpecDecodeDraftingConfig> draftingConfig;
        ContextCacheConfig contextCacheConfig;
        std::string checkpointDir;
        std::string draftCheckpointDir;
        std::vector<int32_t> localRanks;
        std::unordered_map<int32_t, int32_t> localRankDevices;
        std::vector<cudaStream_t> localStreams;
        bool ownsLocalStreams{true};
        std::vector<ParallelBackendHandles> backendHandles;
        std::unique_ptr<ModelArtifacts> modelArtifacts;
    };

    explicit RuntimeCoordinator(Config config);
    ~RuntimeCoordinator();

    RuntimeCoordinator(RuntimeCoordinator const&) = delete;
    RuntimeCoordinator& operator=(RuntimeCoordinator const&) = delete;
    RuntimeCoordinator(RuntimeCoordinator&&) = delete;
    RuntimeCoordinator& operator=(RuntimeCoordinator&&) = delete;

    bool captureDecodingCUDAGraph(cudaStream_t stream = nullptr);
    bool dispatchRequest(LLMGenerationRequest const& request, bool enableProfiling,
        bool outputThinkerEmbeddings = false, cudaStream_t stream = nullptr);
    bool genAndSaveSystemPromptKVCache(
        std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream = nullptr);
    void setVisualPrunerConfig(VisualPrunerConfig const& config);

    LLMGenerationResponse takeRankResponse(int32_t globalRank);

    LLMRankRuntime& rootRuntime();
    LLMRankRuntime const& rootRuntime() const;

    bool ownsGlobalRank(int32_t globalRank) const noexcept;
    bool localRanksSucceeded() const noexcept;

private:
    using TokenBroadcastFn = std::function<bool(void* buffer, int32_t count, cudaStream_t stream)>;

    struct RankPlan
    {
        int32_t globalRank{0};
        int32_t localDevice{0};
        ParallelMapping mapping;
        std::vector<ParallelGroupConfig> groups;
    };

    struct WorkerTask
    {
        std::vector<LLMGenerationRequest> requests;
        std::vector<LLMGenerationResponse> responses;
        std::vector<int32_t> statuses;
        std::vector<std::string> errors;
        std::vector<cudaStream_t> requestStreams;
        bool enableProfiling{false};
        bool outputThinkerEmbeddings{false};
    };

    void initialize();
    void validateParallelPlan();
    void initializeCollectiveResources();
    void initializeRankStreams();
    void initializeTokenizer();
    void initializeRankRuntimes();
    void initializeRequestSynchronization();
    void registerCollectiveGroup(MultiDevicePluginResources const& resources);
    void startWorkers();
    void stopWorkers() noexcept;
    void destroyStreams() noexcept;
    bool abortOwnedRuntimeCollectives() noexcept;
    void recordWorkerFailure(int32_t rank, std::string message) noexcept;
    void publishWorkerCompletion() noexcept;
    void initializeWorkerTaskBuffers();
    bool runInline(
        LLMGenerationRequest const& request, bool enableProfiling, bool outputThinkerEmbeddings, cudaStream_t stream);

    std::unique_ptr<LLMRankRuntime> createRankRuntime(int32_t globalRank);
    LLMGenerationRequest prepareRequestState(LLMGenerationRequest const& request) const;
    void prepareRankRequests(LLMGenerationRequest const& request);
    CollectiveGroup const* collectiveGroup(ParallelType type) const noexcept;
    CollectiveGroup* collectiveGroup(ParallelType type) noexcept;
    ParallelGroupConfig const& groupForRank(int32_t globalRank, ParallelType type) const;
    ParallelBackendHandles const* backendHandlesFor(ParallelType type) const noexcept;
    int32_t rankForGroup(int32_t globalRank, ParallelType type) const noexcept;
    int32_t runtimeRankFor(int32_t globalRank) const;
    int32_t deviceForRank(int32_t rank) const;

    Config mConfig;
    ParallelConfig mParallelConfig;
    int32_t mWorldSize{1};
    std::vector<int32_t> mLocalRanks;
    std::vector<RankPlan> mRankPlans;
    std::vector<std::unique_ptr<MultiDevicePluginResources>> mPluginResources;
    std::vector<std::unique_ptr<CollectiveGroup>> mCollectiveGroups;
    std::vector<cudaStream_t> mStreams;
    bool mOwnsStreams{true};
    bool mInlineSingleRank{false};
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;
    std::vector<std::unique_ptr<LLMRankRuntime>> mRuntimes;
    std::vector<TokenBroadcastFn> mTokenSyncFns;

    WorkerTask mWorkerTask;
    std::atomic<bool> mWorkersDone{false};
    std::atomic<bool> mWorkerFailed{false};
    std::atomic<uint64_t> mWorkGeneration{0};
    std::atomic<int32_t> mWorkersFinished{0};
    std::vector<std::string> mWorkerErrors;
    std::mutex mWorkerErrorMutex;
    std::mutex mWorkMutex;
    int32_t mWorkerStartupReports{0}; //!< Protected by mWorkMutex.
    int32_t mLiveWorkers{0};          //!< Protected by mWorkMutex.
    std::condition_variable mWorkCv;
    std::condition_variable mWorkerReadyCv;
    std::mutex mDoneMutex;
    std::condition_variable mDoneCv;
    std::vector<std::thread> mWorkers;
};

} // namespace rt
} // namespace trt_edgellm
