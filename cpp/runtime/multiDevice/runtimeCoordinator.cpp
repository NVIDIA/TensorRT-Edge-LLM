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

#include "runtime/multiDevice/runtimeCoordinator.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/parallelArtifactNames.h"
#include "common/stringUtils.h"
#include "profiling/timer.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/llmRankRuntime.h"
#include "runtime/multiDevice/collectiveGroup.h"
#include "runtime/multiDevice/multiDevicePluginResourceFactory.h"
#include "runtime/multiDevice/multiDevicePluginResources.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace trt_edgellm
{
namespace rt
{


RuntimeCoordinator::RuntimeCoordinator(Config config)
    : mConfig(std::move(config))
    , mParallelConfig(mConfig.parallelConfig)
    , mWorldSize(parallelWorldSize(mParallelConfig))
    , mOwnsStreams(mConfig.ownsLocalStreams)
{
    if (mConfig.localRanks.empty())
    {
        mLocalRanks.reserve(mWorldSize);
        for (int32_t rank = 0; rank < mWorldSize; ++rank)
        {
            mLocalRanks.push_back(rank);
        }
    }
    else
    {
        mLocalRanks = mConfig.localRanks;
    }

    mInlineSingleRank = isInlineSingleRank(mParallelConfig.launchMode, mWorldSize, mLocalRanks.size());
    if (mInlineSingleRank)
    {
        LOG_INFO("RuntimeCoordinator running single-device inline (no worker thread).");
    }

    try
    {
        initialize();
    }
    catch (...)
    {
        // If construction fails, ~RuntimeCoordinator() will not run. Initialization may already have started
        // workers and created rank runtimes, collective resources, plugin resources, and CUDA streams. Abort owned
        // communicators first so a remote peer cannot remain blocked, then release resources in dependency order.
        static_cast<void>(abortOwnedRuntimeCollectives());
        stopWorkers();
        mRuntimes.clear();
        mCollectiveGroups.clear();
        mPluginResources.clear();
        destroyStreams();
        throw;
    }
}

RuntimeCoordinator::~RuntimeCoordinator()
{
    stopWorkers();
    mRuntimes.clear();
    mCollectiveGroups.clear();
    mPluginResources.clear();
    destroyStreams();
}

void RuntimeCoordinator::initialize()
{
    validateParallelPlan();
    initializeCollectiveResources();
    initializeRankStreams();
    initializeTokenizer();
    initializeRankRuntimes();
    initializeRequestSynchronization();
    initializeWorkerTaskBuffers();
    if (!mInlineSingleRank)
    {
        startWorkers();
    }
}

void RuntimeCoordinator::validateParallelPlan()
{
    ELLM_CHECK(mParallelConfig.launchMode == ParallelLaunchMode::kThread
            || mParallelConfig.launchMode == ParallelLaunchMode::kMpi,
        "RuntimeCoordinator currently supports thread and MPI launch modes only.");
    ELLM_CHECK(mWorldSize >= 1, "RuntimeCoordinator requires a positive world size.");
    ELLM_CHECK(!mLocalRanks.empty(), "RuntimeCoordinator requires at least one local rank.");
    ELLM_CHECK(mConfig.modelArtifacts == nullptr || (mWorldSize == 1 && mLocalRanks.size() == 1 && mLocalRanks[0] == 0),
        "Injected model artifacts require inline single-rank execution.");
    ELLM_CHECK(mWorldSize == 1 || !mConfig.draftingConfig.has_value(),
        "Tensor-parallel speculative decoding is not supported.");
    ELLM_CHECK(mWorldSize == 1 || !mConfig.contextCacheConfig.enabled,
        "Tensor-parallel context reuse is not supported in the current multi-device runtime.");
    if (mWorldSize > 1)
    {
        for (ParallelBackendHandles const& backendHandles : mConfig.backendHandles)
        {
            bool const activeMultiRankGroup = parallelGroupSize(mParallelConfig, backendHandles.type) > 1;
            ELLM_CHECK(backendHandles.handles.empty() || !activeMultiRankGroup || backendHandles.ownsHandles,
                "An active multi-rank group requires ownership of externally supplied collective handles so a "
                "fatal rank failure can abort blocked peers safely.");
        }
    }

    for (auto const& localRankDevice : mConfig.localRankDevices)
    {
        int32_t const rank = localRankDevice.first;
        if (std::find(mLocalRanks.begin(), mLocalRanks.end(), rank) == mLocalRanks.end())
        {
            throw std::runtime_error(format::fmtstr(
                "Local CUDA device override for rank %d is invalid because this process does not own that rank.",
                rank));
        }
    }

    int32_t const deviceCount = detectCudaDeviceCount();
    if (!mParallelConfig.devices.empty() && static_cast<int32_t>(mParallelConfig.devices.size()) < mWorldSize)
    {
        throw std::runtime_error("ParallelConfig.devices must contain at least worldSize entries.");
    }
    for (int32_t const rank : mLocalRanks)
    {
        if (rank < 0 || rank >= mWorldSize)
        {
            throw std::runtime_error(
                format::fmtstr("Local rank %d is out of range for world size %d.", rank, mWorldSize));
        }
        int32_t const device = deviceForRank(rank);
        if (device < 0 || device >= deviceCount)
        {
            throw std::runtime_error(format::fmtstr(
                "Requested CUDA device %d for rank %d but CUDA reports only %d devices.", device, rank, deviceCount));
        }
    }

    mRankPlans.clear();
    mRankPlans.reserve(mWorldSize);
    for (int32_t rank = 0; rank < mWorldSize; ++rank)
    {
        RankPlan rankPlan{};
        rankPlan.globalRank = rank;
        rankPlan.localDevice = deviceForRank(rank);
        rankPlan.mapping = makeParallelMapping(mParallelConfig, rank, rankPlan.localDevice);
        rankPlan.groups = activeParallelGroups(mParallelConfig, rank, rankPlan.localDevice);
        bool hasTensorGroup = false;
        for (ParallelGroupConfig const& group : rankPlan.groups)
        {
            hasTensorGroup |= group.type == ParallelType::kTensor;
        }
        if (!hasTensorGroup)
        {
            rankPlan.groups.push_back(
                makeParallelGroupConfig(mParallelConfig, ParallelType::kTensor, rank, rankPlan.localDevice));
        }
        mRankPlans.push_back(std::move(rankPlan));
    }
}

void RuntimeCoordinator::initializeCollectiveResources()
{

    mCollectiveGroups.clear();
    mPluginResources.clear();
    for (ParallelGroupConfig const& groupConfig : mRankPlans[0].groups)
    {
        if (groupConfig.size <= 1)
        {
            continue;
        }
        std::vector<int32_t> localGroupRanks;
        std::vector<int32_t> localDevices;
        localGroupRanks.reserve(mLocalRanks.size());
        localDevices.reserve(mLocalRanks.size());
        for (int32_t const globalRank : mLocalRanks)
        {
            localGroupRanks.push_back(rankForGroup(globalRank, groupConfig.type));
            localDevices.push_back(deviceForRank(globalRank));
        }

        ParallelBackendHandles const* externalHandles = backendHandlesFor(groupConfig.type);
        MultiDevicePluginResourceConfig resourceConfig{};
        resourceConfig.groupConfig = groupConfig;
        resourceConfig.localRanks = std::move(localGroupRanks);
        resourceConfig.localDevices = std::move(localDevices);
        if (externalHandles != nullptr)
        {
            resourceConfig.backendHandles = externalHandles->handles;
            resourceConfig.ownsBackendHandles = externalHandles->ownsHandles;
        }
        auto resources = createMultiDevicePluginResources(resourceConfig);
        mPluginResources.push_back(std::move(resources));
        registerCollectiveGroup(*mPluginResources.back());
    }
}

void RuntimeCoordinator::initializeRankStreams()
{
    mStreams.assign(mWorldSize, nullptr);
    if (!mConfig.localStreams.empty())
    {
        if (mConfig.localStreams.size() != mLocalRanks.size())
        {
            throw std::runtime_error("RuntimeCoordinator localStreams must match localRanks when provided.");
        }
        for (size_t i = 0; i < mLocalRanks.size(); ++i)
        {
            int32_t const rank = mLocalRanks[i];
            mStreams[rank] = mConfig.localStreams[i];
        }
        return;
    }
    for (int32_t const rank : mLocalRanks)
    {
        CUDA_CHECK(cudaSetDevice(deviceForRank(rank)));
        CUDA_CHECK(cudaStreamCreate(&mStreams[rank]));
    }
}

void RuntimeCoordinator::initializeTokenizer()
{
    if (mConfig.modelArtifacts != nullptr)
    {
        ELLM_CHECK(mConfig.modelArtifacts->tokenizer != nullptr, "Injected model artifacts require a tokenizer.");
        mTokenizer = std::move(mConfig.modelArtifacts->tokenizer);
        return;
    }

    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    LOG_INFO("Start loading tokenizer from model directory: %s", mConfig.engineDir.c_str());
    ELLM_CHECK(mTokenizer->loadFromHF(mConfig.engineDir),
        "Failed to load tokenizer from model directory: " + mConfig.engineDir);
    LOG_INFO("Tokenizer successfully loaded from model directory: %s", mConfig.engineDir.c_str());

    // The tokenizer is shared by all rank runtimes. Configure additional EOS IDs
    // once, before the rank initialization threads start, to avoid concurrent writes.
    parallel_artifacts::RankArtifactContext const artifactContext{mWorldSize, /*globalRank=*/0};
    std::filesystem::path const configPath = mConfig.draftingConfig.has_value()
        ? std::filesystem::path(mConfig.engineDir) / "base_config.json"
        : std::filesystem::path(mConfig.engineDir) / parallel_artifacts::configFileName(artifactContext);
    LLMEngineConfig const tokenizerConfig = parseEngineConfig(configPath, /*rank=*/0, mWorldSize);
    if (!tokenizerConfig.eosTokenIds.empty())
    {
        std::vector<tokenizer::Rank> const additionalEos(
            tokenizerConfig.eosTokenIds.begin(), tokenizerConfig.eosTokenIds.end());
        mTokenizer->setAdditionalEosIds(additionalEos);
        LOG_INFO("Loaded %zu EOS token IDs from config", additionalEos.size());
    }
}

void RuntimeCoordinator::initializeRankRuntimes()
{
    mRuntimes.resize(mWorldSize);
    std::vector<std::string> initErrors(mWorldSize);
    std::vector<std::thread> initThreads;
    initThreads.reserve(mWorldSize);

    for (int32_t const rank : mLocalRanks)
    {
        initThreads.emplace_back([this, rank, &initErrors]() {
            try
            {
                CUDA_CHECK(cudaSetDevice(deviceForRank(rank)));
                mRuntimes[rank] = createRankRuntime(rank);
                LOG_INFO("[parallel rank %d/%d] Runtime initialized.", rank, mWorldSize);
            }
            catch (std::exception const& e)
            {
                initErrors[rank] = e.what();
            }
        });
    }

    for (auto& thread : initThreads)
    {
        thread.join();
    }

    for (int32_t const rank : mLocalRanks)
    {
        if (!initErrors[rank].empty())
        {
            throw std::runtime_error(
                format::fmtstr("[parallel rank %d] Failed to initialize runtime: %s", rank, initErrors[rank].c_str()));
        }
    }
}

void RuntimeCoordinator::initializeRequestSynchronization()
{
    mTokenSyncFns.resize(mWorldSize);
    if (mWorldSize <= 1)
    {
        for (int32_t const rank : mLocalRanks)
        {
            mTokenSyncFns[rank] = [](void*, int32_t, cudaStream_t) { return true; };
        }
        return;
    }

    CollectiveGroup const* tokenSyncGroup = collectiveGroup(ParallelType::kTensor);
    if (tokenSyncGroup == nullptr)
    {
        throw std::runtime_error("Tensor parallel collective group is not initialized for token synchronization.");
    }

    for (int32_t const rank : mLocalRanks)
    {
        int32_t const groupRank = rankForGroup(rank, ParallelType::kTensor);
        mTokenSyncFns[rank] = [this, groupRank](void* buffer, int32_t count, cudaStream_t stream) {
            CollectiveGroup const* group = collectiveGroup(ParallelType::kTensor);
            return group != nullptr && group->broadcastInt32(groupRank, buffer, count, /*rootRank=*/0, stream);
        };
    }
}

void RuntimeCoordinator::registerCollectiveGroup(MultiDevicePluginResources const& resources)
{
    RuntimeCollectiveResources const* runtimeCollectives = resources.runtimeCollectives();
    if (runtimeCollectives == nullptr)
    {
        LOG_INFO("No runtime collective resources are available for the '%s' parallel group.",
            parallelTypeName(resources.type()));
        return;
    }

    std::vector<void*> backendHandles;
    backendHandles.reserve(resources.size());
    for (int32_t rank = 0; rank < resources.size(); ++rank)
    {
        backendHandles.push_back(runtimeCollectives->communicatorForRank(rank));
    }

    ParallelGroupConfig groupConfig
        = makeParallelGroupConfig(mParallelConfig, resources.type(), /*globalRank=*/0, deviceForRank(0));
    mCollectiveGroups.push_back(std::make_unique<CollectiveGroup>(groupConfig, std::move(backendHandles)));
}

void RuntimeCoordinator::initializeWorkerTaskBuffers()
{
    mWorkerTask.requests.resize(mWorldSize);
    mWorkerTask.responses.resize(mWorldSize);
    mWorkerTask.statuses.resize(mWorldSize, 0);
    mWorkerTask.errors.resize(mWorldSize);
    mWorkerTask.requestStreams.resize(mWorldSize, nullptr);
    mWorkerErrors.resize(mWorldSize);
}

void RuntimeCoordinator::startWorkers()
{
    {
        std::lock_guard<std::mutex> lock(mWorkMutex);
        mWorkerStartupReports = 0;
        mLiveWorkers = 0;
    }
    mWorkers.reserve(mLocalRanks.size());
    for (int32_t const rank : mLocalRanks)
    {
        mWorkers.emplace_back([this, rank]() {
            uint64_t localGeneration = 0;
            try
            {
                CUDA_CHECK(cudaSetDevice(deviceForRank(rank)));
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("[parallel rank %d/%d] worker initialization failed: %s", rank, mWorldSize, e.what());
                recordWorkerFailure(rank, e.what());
                {
                    std::lock_guard<std::mutex> lock(mWorkMutex);
                    ++mWorkerStartupReports;
                }
                mWorkerReadyCv.notify_one();
                return;
            }
            catch (...)
            {
                LOG_ERROR(
                    "[parallel rank %d/%d] worker initialization failed with an unknown exception.", rank, mWorldSize);
                recordWorkerFailure(rank, "unknown worker initialization failure");
                {
                    std::lock_guard<std::mutex> lock(mWorkMutex);
                    ++mWorkerStartupReports;
                }
                mWorkerReadyCv.notify_one();
                return;
            }

            {
                std::lock_guard<std::mutex> lock(mWorkMutex);
                ++mWorkerStartupReports;
                ++mLiveWorkers;
            }
            mWorkerReadyCv.notify_one();

            try
            {
                while (true)
                {
                    {
                        std::unique_lock<std::mutex> lock(mWorkMutex);
                        mWorkCv.wait(lock, [this, localGeneration]() {
                            return mWorkGeneration.load(std::memory_order_acquire) > localGeneration
                                || mWorkersDone.load(std::memory_order_acquire);
                        });
                        if (mWorkersDone.load(std::memory_order_acquire))
                        {
                            break;
                        }
                        localGeneration = mWorkGeneration.load(std::memory_order_acquire);
                    }

                    try
                    {
                        setProfilingEnabled(rank == 0 && mWorkerTask.enableProfiling);
                        cudaStream_t const executionStream = mWorkerTask.requestStreams[rank] != nullptr
                            ? mWorkerTask.requestStreams[rank]
                            : mStreams[rank];
                        bool const succeeded
                            = mRuntimes[rank]->handleRequest(mWorkerTask.requests[rank], mWorkerTask.responses[rank],
                                executionStream, mWorkerTask.outputThinkerEmbeddings && rank == 0, mTokenSyncFns[rank],
                                runtimeRankFor(rank));
                        mWorkerTask.statuses[rank] = succeeded ? 1 : 0;
                        if (!succeeded)
                        {
                            mWorkerTask.errors[rank] = "handleRequest returned failure";
                            recordWorkerFailure(rank, mWorkerTask.errors[rank]);
                        }
                    }
                    catch (std::exception const& e)
                    {
                        LOG_ERROR("[parallel rank %d/%d] handleRequest failed: %s", rank, mWorldSize, e.what());
                        mWorkerTask.statuses[rank] = 0;
                        mWorkerTask.errors[rank] = e.what();
                        if (mWorkerTask.errors[rank].find("EDGELLM_INPUT_TOO_LONG:") == std::string::npos)
                        {
                            recordWorkerFailure(rank, mWorkerTask.errors[rank]);
                        }
                    }
                    catch (...)
                    {
                        LOG_ERROR(
                            "[parallel rank %d/%d] handleRequest failed with an unknown exception.", rank, mWorldSize);
                        mWorkerTask.statuses[rank] = 0;
                        mWorkerTask.errors[rank] = "unknown handleRequest failure";
                        recordWorkerFailure(rank, mWorkerTask.errors[rank]);
                    }

                    publishWorkerCompletion();
                }
            }
            catch (std::exception const& e)
            {
                bool generationPending = false;
                {
                    std::lock_guard<std::mutex> lock(mWorkMutex);
                    generationPending = mWorkGeneration.load(std::memory_order_acquire) > localGeneration;
                    --mLiveWorkers;
                }
                LOG_ERROR("[parallel rank %d/%d] worker exited unexpectedly: %s", rank, mWorldSize, e.what());
                recordWorkerFailure(rank, e.what());
                if (generationPending)
                {
                    publishWorkerCompletion();
                }
                return;
            }
            catch (...)
            {
                bool generationPending = false;
                {
                    std::lock_guard<std::mutex> lock(mWorkMutex);
                    generationPending = mWorkGeneration.load(std::memory_order_acquire) > localGeneration;
                    --mLiveWorkers;
                }
                LOG_ERROR("[parallel rank %d/%d] worker exited after an unknown failure.", rank, mWorldSize);
                recordWorkerFailure(rank, "unknown failure");
                if (generationPending)
                {
                    publishWorkerCompletion();
                }
                return;
            }

            {
                std::lock_guard<std::mutex> lock(mWorkMutex);
                --mLiveWorkers;
            }
        });
    }

    std::unique_lock<std::mutex> lock(mWorkMutex);
    mWorkerReadyCv.wait(lock, [this]() { return mWorkerStartupReports == static_cast<int32_t>(mLocalRanks.size()); });
    ELLM_CHECK(
        !mWorkerFailed.load(std::memory_order_acquire) && mLiveWorkers == static_cast<int32_t>(mLocalRanks.size()),
        "Failed to start all local parallel-rank workers.");
}

void RuntimeCoordinator::stopWorkers() noexcept
{
    {
        std::lock_guard<std::mutex> lock(mWorkMutex);
        mWorkersDone.store(true, std::memory_order_release);
    }
    mWorkCv.notify_all();
    for (auto& worker : mWorkers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
    mWorkers.clear();
}

void RuntimeCoordinator::destroyStreams() noexcept
{
    for (int32_t rank = 0; rank < static_cast<int32_t>(mStreams.size()); ++rank)
    {
        if (mStreams[rank] != nullptr && mOwnsStreams)
        {
            int32_t const device = deviceForRank(rank);
            cudaError_t const setDeviceErr = cudaSetDevice(device);
            if (setDeviceErr != cudaSuccess)
            {
                LOG_WARNING("[parallel rank %d/%d] Failed to select CUDA device %d before stream cleanup: %s", rank,
                    mWorldSize, device, cudaGetErrorString(setDeviceErr));
                mStreams[rank] = nullptr;
                continue;
            }
            cudaError_t const err = cudaStreamDestroy(mStreams[rank]);
            if (err != cudaSuccess)
            {
                LOG_WARNING("[parallel rank %d/%d] Failed to destroy CUDA stream: %s", rank, mWorldSize,
                    cudaGetErrorString(err));
            }
            mStreams[rank] = nullptr;
        }
        else if (mStreams[rank] != nullptr)
        {
            mStreams[rank] = nullptr;
        }
    }
}

bool RuntimeCoordinator::abortOwnedRuntimeCollectives() noexcept
{
    bool abortedAny = false;
    for (auto& resources : mPluginResources)
    {
        if (resources != nullptr)
        {
            abortedAny |= resources->abortOwnedRuntimeCollectives();
        }
    }
    return abortedAny;
}

void RuntimeCoordinator::recordWorkerFailure(int32_t rank, std::string message) noexcept
{
    try
    {
        if (rank >= 0 && rank < static_cast<int32_t>(mWorkerErrors.size()))
        {
            std::lock_guard<std::mutex> lock(mWorkerErrorMutex);
            mWorkerErrors[rank] = std::move(message);
        }
    }
    catch (...)
    {
    }

    {
        std::lock_guard<std::mutex> lock(mDoneMutex);
        mWorkerFailed.store(true, std::memory_order_release);
    }
    if (mWorldSize > 1 && !abortOwnedRuntimeCollectives())
    {
        LOG_ERROR(
            "Cannot abort the failed parallel request because its NCCL communicators are externally owned. "
            "The communicator owner must abort the group to release peers blocked in collectives.");
    }
}

void RuntimeCoordinator::publishWorkerCompletion() noexcept
{
    bool allWorkersFinished = false;
    {
        std::lock_guard<std::mutex> lock(mDoneMutex);
        allWorkersFinished
            = mWorkersFinished.fetch_add(1, std::memory_order_acq_rel) == static_cast<int32_t>(mLocalRanks.size()) - 1;
    }
    if (allWorkersFinished)
    {
        mDoneCv.notify_one();
    }
}

bool RuntimeCoordinator::captureDecodingCUDAGraph(cudaStream_t stream)
{
    char const* disableCudaGraphEnv = std::getenv("EDGELLM_DISABLE_CUDA_GRAPH");
    if (disableCudaGraphEnv != nullptr)
    {
        LOG_INFO("CUDA graph capture disabled via EDGELLM_DISABLE_CUDA_GRAPH.");
        return true;
    }

    if (mWorkerFailed.load(std::memory_order_acquire))
    {
        LOG_ERROR("Runtime coordinator has failed; refusing decoding CUDA graph capture.");
        return false;
    }

    if (mInlineSingleRank)
    {
        // Inline single-rank capture uses the caller's thread and stream.
        CUDA_CHECK(cudaSetDevice(deviceForRank(mLocalRanks.front())));
        return rootRuntime().captureDecodingCUDAGraph(stream);
    }

    std::vector<int32_t> captureResults(mWorldSize, 0);
    std::vector<std::string> captureErrors(mWorldSize);
    std::vector<std::thread> captureThreads;
    captureThreads.reserve(mLocalRanks.size());
    for (int32_t const rank : mLocalRanks)
    {
        captureThreads.emplace_back([this, rank, stream, &captureResults, &captureErrors]() {
            try
            {
                CUDA_CHECK(cudaSetDevice(deviceForRank(rank)));
                cudaStream_t const captureStream
                    = stream != nullptr && mLocalRanks.size() == 1 ? stream : mStreams[rank];
                captureResults[rank] = mRuntimes[rank]->captureDecodingCUDAGraph(captureStream) ? 1 : 0;
            }
            catch (std::exception const& e)
            {
                static_cast<void>(cudaGetLastError());
                captureErrors[rank] = e.what();
                recordWorkerFailure(rank, captureErrors[rank]);
            }
            catch (...)
            {
                static_cast<void>(cudaGetLastError());
                captureErrors[rank] = "unknown decoding CUDA graph capture failure";
                recordWorkerFailure(rank, captureErrors[rank]);
            }
        });
    }
    for (auto& thread : captureThreads)
    {
        thread.join();
    }

    bool decodingCaptured = true;
    for (int32_t const rank : mLocalRanks)
    {
        if (captureResults[rank] == 0)
        {
            if (!captureErrors[rank].empty())
            {
                LOG_WARNING("[parallel rank %d/%d] Failed to capture decoding CUDA graph: %s. Using normal execution.",
                    rank, mWorldSize, captureErrors[rank].c_str());
            }
            else
            {
                LOG_WARNING("[parallel rank %d/%d] Failed to capture decoding CUDA graph; using normal execution.",
                    rank, mWorldSize);
            }
            decodingCaptured = false;
        }
    }
    return decodingCaptured;
}

bool RuntimeCoordinator::genAndSaveSystemPromptKVCache(
    std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream)
{
    if (mInlineSingleRank)
    {
        int32_t const rank = mLocalRanks.front();
        CUDA_CHECK(cudaSetDevice(deviceForRank(rank)));
        cudaStream_t const executionStream = stream != nullptr ? stream : mStreams[rank];
        return rootRuntime().genAndSaveSystemPromptKVCache(prompt, loraWeightsName, executionStream);
    }
    if (mWorkerFailed.load(std::memory_order_acquire))
    {
        LOG_ERROR("Runtime coordinator has failed; refusing system-prompt KV-cache generation.");
        return false;
    }

    std::vector<int32_t> results(mWorldSize, 0);
    std::vector<std::string> errors(mWorldSize);
    std::vector<std::thread> threads;
    threads.reserve(mLocalRanks.size());
    for (int32_t const rank : mLocalRanks)
    {
        threads.emplace_back([this, rank, stream, &prompt, &loraWeightsName, &results, &errors]() {
            try
            {
                CUDA_CHECK(cudaSetDevice(deviceForRank(rank)));
                cudaStream_t const executionStream
                    = stream != nullptr && mLocalRanks.size() == 1 ? stream : mStreams[rank];
                bool const succeeded
                    = mRuntimes[rank]->genAndSaveSystemPromptKVCache(prompt, loraWeightsName, executionStream);
                results[rank] = succeeded ? 1 : 0;
                if (!succeeded)
                {
                    errors[rank] = "system-prompt KV-cache generation returned failure";
                    recordWorkerFailure(rank, errors[rank]);
                }
            }
            catch (std::exception const& e)
            {
                errors[rank] = e.what();
                recordWorkerFailure(rank, errors[rank]);
            }
            catch (...)
            {
                errors[rank] = "unknown system-prompt KV-cache generation failure";
                recordWorkerFailure(rank, errors[rank]);
            }
        });
    }
    for (auto& thread : threads)
    {
        thread.join();
    }

    bool succeeded = !mWorkerFailed.load(std::memory_order_acquire);
    for (int32_t const rank : mLocalRanks)
    {
        if (results[rank] == 0)
        {
            LOG_ERROR("[parallel rank %d/%d] System-prompt KV-cache generation failed: %s", rank, mWorldSize,
                errors[rank].empty() ? "unknown failure" : errors[rank].c_str());
            succeeded = false;
        }
    }
    return succeeded;
}

void RuntimeCoordinator::setVisualPrunerConfig(VisualPrunerConfig const& config)
{
    ELLM_CHECK(mWorldSize == 1, "Visual-token pruning is not supported with multi-device execution.");
    int32_t const rank = mLocalRanks.front();
    CUDA_CHECK(cudaSetDevice(deviceForRank(rank)));
    rootRuntime().setVisualPrunerConfig(config);
}

bool RuntimeCoordinator::dispatchRequest(
    LLMGenerationRequest const& request, bool enableProfiling, bool outputThinkerEmbeddings, cudaStream_t stream)
{
    if (mInlineSingleRank)
    {
        return runInline(request, enableProfiling, outputThinkerEmbeddings, stream);
    }

    if (mWorkerFailed.load(std::memory_order_acquire))
    {
        LOG_ERROR("Runtime coordinator has a failed worker; refusing to dispatch a new request.");
        return false;
    }

    LLMGenerationRequest preparedRequest;
    try
    {
        preparedRequest = prepareRequestState(request);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to prepare threaded parallel request: %s", e.what());
        return false;
    }
    {
        std::lock_guard<std::mutex> workLock(mWorkMutex);
        if (mWorkerFailed.load(std::memory_order_acquire) || mLiveWorkers != static_cast<int32_t>(mLocalRanks.size()))
        {
            LOG_ERROR("Runtime coordinator does not have a complete live worker set; dispatch cancelled.");
            return false;
        }
        prepareRankRequests(preparedRequest);
        mWorkerTask.enableProfiling = enableProfiling;
        mWorkerTask.outputThinkerEmbeddings = outputThinkerEmbeddings;
        std::fill(mWorkerTask.requestStreams.begin(), mWorkerTask.requestStreams.end(), nullptr);
        if (stream != nullptr && mLocalRanks.size() == 1)
        {
            mWorkerTask.requestStreams[mLocalRanks.front()] = stream;
        }
        for (int32_t const rank : mLocalRanks)
        {
            mWorkerTask.errors[rank].clear();
        }
        {
            std::lock_guard<std::mutex> doneLock(mDoneMutex);
            mWorkersFinished.store(0, std::memory_order_release);
        }
        mWorkGeneration.fetch_add(1, std::memory_order_release);
    }
    mWorkCv.notify_all();

    std::unique_lock<std::mutex> lock(mDoneMutex);
    mDoneCv.wait(lock, [this]() {
        return mWorkersFinished.load(std::memory_order_acquire) >= static_cast<int32_t>(mLocalRanks.size());
    });
    if (mWorkerFailed.load(std::memory_order_acquire))
    {
        std::lock_guard<std::mutex> errorLock(mWorkerErrorMutex);
        for (int32_t rank = 0; rank < static_cast<int32_t>(mWorkerErrors.size()); ++rank)
        {
            if (!mWorkerErrors[rank].empty())
            {
                LOG_ERROR("[parallel rank %d/%d] worker failure: %s", rank, mWorldSize, mWorkerErrors[rank].c_str());
            }
        }
        return false;
    }

    std::string inputTooLongError;
    for (int32_t const rank : mLocalRanks)
    {
        std::string const& requestError = mWorkerTask.errors[rank];
        size_t const markerPosition = requestError.find("EDGELLM_INPUT_TOO_LONG:");
        if (markerPosition != std::string::npos)
        {
            inputTooLongError = requestError.substr(markerPosition);
            break;
        }
    }
    if (!inputTooLongError.empty())
    {
        throw std::runtime_error(inputTooLongError);
    }
    return true;
}

bool RuntimeCoordinator::runInline(
    LLMGenerationRequest const& request, bool enableProfiling, bool outputThinkerEmbeddings, cudaStream_t stream)
{
    int32_t const rank = mLocalRanks.front();
    CUDA_CHECK(cudaSetDevice(deviceForRank(rank)));
    setProfilingEnabled(enableProfiling);
    cudaStream_t const executionStream = stream != nullptr ? stream : mStreams[rank];

    try
    {
        mWorkerTask.requests[rank] = prepareRequestState(request);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("[inline single-rank] Failed to prepare request: %s", e.what());
        mWorkerTask.responses[rank] = LLMGenerationResponse{};
        mWorkerTask.statuses[rank] = 0;
        return false;
    }
    mWorkerTask.responses[rank] = LLMGenerationResponse{};
    mWorkerTask.statuses[rank] = 0;
    try
    {
        // Inline single-rank execution uses the caller's thread with no token broadcast;
        // parallelRank=-1 keeps streaming and audio callbacks on that thread.
        mWorkerTask.statuses[rank] = mRuntimes[rank]->handleRequest(mWorkerTask.requests[rank],
                                         mWorkerTask.responses[rank], executionStream, outputThinkerEmbeddings,
                                         /*tokenBroadcast=*/nullptr, /*parallelRank=*/-1)
            ? 1
            : 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("[inline single-rank] handleRequest failed: %s", e.what());
        mWorkerTask.statuses[rank] = 0;
        if (std::string const message{e.what()}; message.find("EDGELLM_INPUT_TOO_LONG:") != std::string::npos)
        {
            throw;
        }
        return false;
    }
    return mWorkerTask.statuses[rank] == 1;
}

std::unique_ptr<LLMRankRuntime> RuntimeCoordinator::createRankRuntime(int32_t globalRank)
{
    ParallelGroupConfig const& engineGroup = groupForRank(globalRank, ParallelType::kTensor);

    ParallelMapping const& mapping = mRankPlans[globalRank].mapping;
    // Engine shards and token synchronization both use the tensor-parallel group.
    ELLM_CHECK(mapping.tensorParallelSize == engineGroup.size && mapping.tensorParallelRank == engineGroup.rank,
        "Resolved tensor-parallel coordinates do not match the engine parallel group.");

    ELLM_CHECK(mTokenizer != nullptr, "RuntimeCoordinator tokenizer is not initialized.");
    if (mConfig.modelArtifacts != nullptr)
    {
        ELLM_CHECK(
            globalRank == 0 && mWorldSize == 1, "Injected model artifacts require global rank 0 of world size 1.");
        auto artifacts = std::move(mConfig.modelArtifacts);
        return std::make_unique<LLMRankRuntime>(std::move(*artifacts), mConfig.engineDir, mConfig.multimodalEngineDir,
            mConfig.loraWeightsMap, mConfig.draftingConfig, mStreams[globalRank], mapping, *mTokenizer,
            mConfig.contextCacheConfig);
    }
    return std::make_unique<LLMRankRuntime>(mConfig.engineDir, mConfig.multimodalEngineDir, mConfig.loraWeightsMap,
        mConfig.draftingConfig, mStreams[globalRank], mapping, *mTokenizer, mConfig.contextCacheConfig,
        mConfig.checkpointDir, mConfig.draftCheckpointDir);
}

LLMGenerationRequest RuntimeCoordinator::prepareRequestState(LLMGenerationRequest const& request) const
{
    ELLM_CHECK(mTokenizer != nullptr, "RuntimeCoordinator tokenizer is not initialized.");

    LLMGenerationRequest preparedRequest = request;
    int32_t const activeBatchSize = static_cast<int32_t>(preparedRequest.requests.size());
    preparedRequest.formattedRequests.resize(activeBatchSize);
    preparedRequest.preTokenizedInputIds.resize(activeBatchSize);

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        bool const formatted
            = mTokenizer->applyChatTemplate(preparedRequest.requests[i], preparedRequest.formattedRequests[i],
                preparedRequest.applyChatTemplate, preparedRequest.addGenerationPrompt, preparedRequest.enableThinking);
        if (!formatted)
        {
            throw std::runtime_error(format::fmtstr("Failed to apply chat template for request %d in batch.", i));
        }

        if (i < static_cast<int32_t>(request.preTokenizedInputIds.size()) && !request.preTokenizedInputIds[i].empty())
        {
            preparedRequest.preTokenizedInputIds[i] = request.preTokenizedInputIds[i];
        }
        else
        {
            // The formatted chat request already carries the model template's special-token policy.
            preparedRequest.preTokenizedInputIds[i]
                = mTokenizer->encode(preparedRequest.formattedRequests[i].formattedCompleteRequest, false);
        }
        if (preparedRequest.preTokenizedInputIds[i].empty())
        {
            throw std::runtime_error(format::fmtstr("Failed to tokenize input text for request %d in batch.", i));
        }
    }

    // Preserve the established request contract: callers may inspect the
    // mutable formattedRequests field after handleRequest() returns.
    request.formattedRequests = preparedRequest.formattedRequests;

    return preparedRequest;
}

void RuntimeCoordinator::prepareRankRequests(LLMGenerationRequest const& request)
{
    if (mLocalRanks.empty())
    {
        throw std::runtime_error("Request preparation requires at least one local rank.");
    }
    if (request.formattedRequests.size() != request.requests.size()
        || request.preTokenizedInputIds.size() != request.requests.size())
    {
        throw std::runtime_error("Request preparation requires formatted and pre-tokenized request state.");
    }
    CUDA_CHECK(cudaSetDevice(deviceForRank(mLocalRanks.front())));

    for (int32_t const rank : mLocalRanks)
    {
        if (rank < 0 || rank >= static_cast<int32_t>(mWorkerTask.requests.size()))
        {
            throw std::runtime_error("Local rank is out of range during request preparation.");
        }
        mWorkerTask.requests[rank] = request;
        if (rank != 0)
        {
            for (auto& channel : mWorkerTask.requests[rank].streamChannels)
            {
                channel.reset();
            }
            mWorkerTask.requests[rank].onTokenGenerated.reset();
        }
        mWorkerTask.responses[rank] = LLMGenerationResponse{};
        mWorkerTask.statuses[rank] = 0;
    }
}

CollectiveGroup const* RuntimeCoordinator::collectiveGroup(ParallelType type) const noexcept
{
    for (auto const& group : mCollectiveGroups)
    {
        if (group != nullptr && group->type() == type)
        {
            return group.get();
        }
    }
    return nullptr;
}

CollectiveGroup* RuntimeCoordinator::collectiveGroup(ParallelType type) noexcept
{
    for (auto& group : mCollectiveGroups)
    {
        if (group != nullptr && group->type() == type)
        {
            return group.get();
        }
    }
    return nullptr;
}

ParallelGroupConfig const& RuntimeCoordinator::groupForRank(int32_t globalRank, ParallelType type) const
{
    ELLM_CHECK(globalRank >= 0 && globalRank < static_cast<int32_t>(mRankPlans.size()), "Rank is out of range.");
    for (ParallelGroupConfig const& group : mRankPlans[globalRank].groups)
    {
        if (group.type == type)
        {
            return group;
        }
    }
    throw std::runtime_error(
        format::fmtstr("Rank %d is not part of an active '%s' parallel group.", globalRank, parallelTypeName(type)));
}

ParallelBackendHandles const* RuntimeCoordinator::backendHandlesFor(ParallelType type) const noexcept
{
    for (ParallelBackendHandles const& handles : mConfig.backendHandles)
    {
        if (handles.type == type)
        {
            return &handles;
        }
    }
    return nullptr;
}

int32_t RuntimeCoordinator::rankForGroup(int32_t globalRank, ParallelType type) const noexcept
{
    return parallelGroupRank(mParallelConfig, type, globalRank);
}

int32_t RuntimeCoordinator::runtimeRankFor(int32_t globalRank) const
{
    return groupForRank(globalRank, ParallelType::kTensor).rank;
}

bool RuntimeCoordinator::ownsGlobalRank(int32_t globalRank) const noexcept
{
    return std::find(mLocalRanks.begin(), mLocalRanks.end(), globalRank) != mLocalRanks.end();
}

LLMGenerationResponse RuntimeCoordinator::takeRankResponse(int32_t globalRank)
{
    ELLM_CHECK(ownsGlobalRank(globalRank), "Cannot take a response for a rank owned by another process.");
    ELLM_CHECK(globalRank >= 0 && globalRank < static_cast<int32_t>(mWorkerTask.responses.size()),
        "Response rank is out of range.");
    return std::move(mWorkerTask.responses[globalRank]);
}

bool RuntimeCoordinator::localRanksSucceeded() const noexcept
{
    for (int32_t const rank : mLocalRanks)
    {
        if (rank < 0 || rank >= static_cast<int32_t>(mWorkerTask.statuses.size()) || mWorkerTask.statuses[rank] == 0)
        {
            return false;
        }
    }
    return true;
}

LLMRankRuntime& RuntimeCoordinator::rootRuntime()
{
    int32_t const rank = ownsGlobalRank(0) ? 0 : mLocalRanks.front();
    ELLM_CHECK(rank >= 0 && rank < static_cast<int32_t>(mRuntimes.size()) && mRuntimes[rank] != nullptr,
        "RuntimeCoordinator root runtime is not initialized.");
    return *mRuntimes[rank];
}

LLMRankRuntime const& RuntimeCoordinator::rootRuntime() const
{
    int32_t const rank = ownsGlobalRank(0) ? 0 : mLocalRanks.front();
    ELLM_CHECK(rank >= 0 && rank < static_cast<int32_t>(mRuntimes.size()) && mRuntimes[rank] != nullptr,
        "RuntimeCoordinator root runtime is not initialized.");
    return *mRuntimes[rank];
}

int32_t RuntimeCoordinator::deviceForRank(int32_t rank) const
{
    ELLM_CHECK(rank >= 0 && rank < mWorldSize, "Rank is out of range.");
    auto const localDevice = mConfig.localRankDevices.find(rank);
    if (localDevice != mConfig.localRankDevices.end())
    {
        return localDevice->second;
    }
    if (!mParallelConfig.devices.empty())
    {
        return mParallelConfig.devices[rank];
    }
    return rank;
}

} // namespace rt
} // namespace trt_edgellm
