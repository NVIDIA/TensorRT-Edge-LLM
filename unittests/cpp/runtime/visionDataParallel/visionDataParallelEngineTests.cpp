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

// Requires Qwen3-VL TP2 engines on two GPUs: LLM batch=1, visual min/max tokens=4/4096,
// and visual maxImageTokensPerImage >= 2560 for the asymmetric capacity cases.
// Set EDGELLM_VISION_DP_ENGINE_DIR to the LLM engine directory and EDGELLM_VISION_DP_VISUAL_DIR to visual.
// EDGELLM_PLUGIN_PATH and EDGELLM_NCCL_SO_PATH select the matching runtime libraries.

#include "chatTemplate/chatTemplate.h"
#include "common/checkMacros.h"
#include "common/trtUtils.h"
#include "multimodal/qwen3/qwen3vlViTRunner.h"
#include "profiling/metrics.h"
#include "runtime/llmRankRuntime.h"
#include "runtime/multiDevice/collectiveGroup.h"
#include "runtime/multiDevice/tensorParallelPluginResources.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <thread>

namespace trt_edgellm::rt
{
namespace
{
constexpr int32_t kRanks{2};
constexpr size_t kOutputs{4}; // Qwen3-VL embedding and all three deepstack features.
constexpr float kAbsoluteTolerance{0.02F};
constexpr float kRelativeTolerance{0.02F};

struct HostOutput
{
    Coords shape;
    std::vector<float> values;
};

struct RunResult
{
    std::array<std::vector<HostOutput>, kRanks> outputs;
    std::array<std::vector<std::vector<int32_t>>, kRanks> inputIds;
    std::array<int32_t, kRanks> gathers{};
    std::array<std::array<int32_t, kRanks>, kRanks> gatheredElements{};
    std::array<int64_t, kRanks> encodedImages{};
};

::testing::AssertionResult sameFeatures(RunResult const& expected, RunResult const& actual)
{
    float maxAbsoluteError = 0;
    for (int32_t rank = 0; rank < kRanks; ++rank)
    {
        if (actual.inputIds[rank] != expected.inputIds[rank])
        {
            return ::testing::AssertionFailure() << "rank " << rank << ": full-request token order changed";
        }
        if (actual.outputs[rank].size() != kOutputs || expected.outputs[rank].size() != kOutputs)
        {
            return ::testing::AssertionFailure() << "rank " << rank << ": missing embedding/deepstack output";
        }
        for (size_t output = 0; output < kOutputs; ++output)
        {
            auto const& a = actual.outputs[rank][output];
            auto const& e = expected.outputs[rank][output];
            if (a.shape != e.shape || a.values.size() != e.values.size())
            {
                return ::testing::AssertionFailure() << "rank " << rank << " output " << output << ": shape changed";
            }
            for (size_t i = 0; i < e.values.size(); ++i)
            {
                float const error = std::abs(a.values[i] - e.values[i]);
                maxAbsoluteError = std::max(maxAbsoluteError, error);
                if (!std::isfinite(a.values[i]) || !std::isfinite(e.values[i])
                    || error > kAbsoluteTolerance + kRelativeTolerance * std::abs(e.values[i]))
                {
                    return ::testing::AssertionFailure() << "rank " << rank << " output " << output << " element " << i
                                                         << ": " << a.values[i] << " vs " << e.values[i];
                }
            }
        }
    }
    std::cout << "ViT-DP feature comparison: max absolute error=" << maxAbsoluteError << '\n';
    return ::testing::AssertionSuccess();
}

class VisionDataParallelEngineTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto const* llm = std::getenv("EDGELLM_VISION_DP_ENGINE_DIR");
        auto const* visual = std::getenv("EDGELLM_VISION_DP_VISUAL_DIR");
        if (llm == nullptr || visual == nullptr)
        {
            GTEST_SKIP()
                << "Set EDGELLM_VISION_DP_ENGINE_DIR and EDGELLM_VISION_DP_VISUAL_DIR to Qwen3-VL TP2 engines.";
        }
        int32_t devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices < kRanks)
        {
            GTEST_SKIP() << "Requires two CUDA devices.";
        }
        mOptions.engineDir = llm;
        mOptions.multimodalEngineDir = visual;
        mOptions.parallelConfig.tensorParallelSize = kRanks;
        mOptions.contextCacheConfig.encoderEmbeddingCacheBudgetBytes = 0;
        ELLM_CHECK(loadEdgellmPluginLib() != nullptr, "Failed to load EdgeLLM plugin");
        ELLM_CHECK(mTokenizer.loadFromHF(llm), "Failed to load tokenizer");
        ELLM_CHECK(mChatTemplate.load(llm,
                       mTokenizer.getBosId() >= 0 ? mTokenizer.idToPiece(mTokenizer.getBosId(), false) : std::string{},
                       mTokenizer.idToPiece(mTokenizer.getEosId(), false)),
            "Failed to load chat template");
        TensorParallelPluginResourcesConfig config;
        config.tpSize = kRanks;
        mResources = std::make_unique<TensorParallelPluginResources>(config);
        ParallelGroupConfig groupConfig;
        groupConfig.size = kRanks;
        auto const* resources = mResources->runtimeCollectives();
        ELLM_CHECK(resources != nullptr, "NCCL runtime resources unavailable");
        mGroup = std::make_unique<CollectiveGroup>(
            groupConfig, std::vector<void*>{resources->communicatorForRank(0), resources->communicatorForRank(1)});
        for (int32_t rank = 0; rank < kRanks; ++rank)
        {
            CUDA_CHECK(cudaSetDevice(rank));
            CUDA_CHECK(cudaStreamCreateWithFlags(&mStreams[rank], cudaStreamNonBlocking));
        }
    }

    void initialize(int64_t cacheBytes = 0)
    {
        mOptions.contextCacheConfig.encoderEmbeddingCacheBudgetBytes = cacheBytes;
        parallel([&](int32_t rank) {
            mRuntimes[rank] = std::make_unique<LLMRankRuntime>(mOptions, mStreams[rank],
                makeParallelMapping(mOptions.parallelConfig, rank, rank), mTokenizer, mChatTemplate);
        });
    }

    void TearDown() override
    {
        for (int32_t rank = 0; rank < kRanks; ++rank)
        {
            if (mStreams[rank] != nullptr)
            {
                cudaSetDevice(rank);
                cudaStreamSynchronize(mStreams[rank]);
                mRuntimes[rank].reset();
                cudaStreamDestroy(mStreams[rank]);
            }
        }
        mGroup.reset();
        mResources.reset();
    }

    void parallel(std::function<void(int32_t)> const& work)
    {
        std::array<std::exception_ptr, kRanks> errors{};
        std::array<std::thread, kRanks> workers;
        for (int32_t rank = 0; rank < kRanks; ++rank)
        {
            workers[rank] = std::thread([&, rank] {
                try
                {
                    CUDA_CHECK(cudaSetDevice(rank));
                    setProfilingEnabled(true);
                    work(rank);
                    CUDA_CHECK(cudaStreamSynchronize(mStreams[rank]));
                }
                catch (...)
                {
                    errors[rank] = std::current_exception();
                    mResources->abortOwnedRuntimeCollectives();
                }
            });
        }
        for (auto& worker : workers)
        {
            worker.join();
        }
        for (auto const& error : errors)
        {
            if (error)
            {
                std::rethrow_exception(error);
            }
        }
    }

    LLMGenerationRequest makeRequest(std::vector<std::pair<int32_t, int32_t>> const& sizes, int32_t seed = 1)
    {
        LLMGenerationRequest request{};
        request.temperature = 1;
        request.topP = 1;
        request.topK = 1;
        request.maxGenerateLength = 1;
        request.ignoreEos = true;
        request.requests.resize(1);
        Message message;
        message.role = "user";
        for (size_t image = 0; image < sizes.size(); ++image)
        {
            auto const [height, width] = sizes[image];
            Tensor pixels({1, height, width, 3}, DeviceType::kCPU, DataType::kUINT8);
            for (int64_t y = 0; y < height; ++y)
            {
                for (int64_t x = 0; x < width; ++x)
                {
                    for (int64_t c = 0; c < 3; ++c)
                    {
                        pixels.dataPointer<uint8_t>()[(y * width + x) * 3 + c] = static_cast<uint8_t>(
                            (x * (3 + seed) + y * (5 + image) + c * 71 + image * 53 + seed * 29) % 256);
                    }
                }
            }
            request.requests.front().imageBuffers.emplace_back(std::move(pixels));
            request.requests.front().imageBuffers.back().doResize = false;
            message.contents.push_back(Message::MessageContent{"image", "test-image"});
        }
        message.contents.push_back(Message::MessageContent{"text", "Describe these images."});
        request.requests.front().messages.push_back(std::move(message));
        request.formattedRequests.resize(1);
        ELLM_CHECK(mChatTemplate.apply(request.requests.front(), request.formattedRequests.front(),
                       chat_template::ChatTemplate::optionsFrom(request)),
            "Failed to format test request");
        request.preTokenizedInputIds.push_back(
            mTokenizer.encode(request.formattedRequests.front().formattedCompleteRequest, false));
        return request;
    }

    RunResult run(LLMGenerationRequest const& request, bool enableDp)
    {
        RunResult result;
        std::array<LLMGenerationResponse, kRanks> responses;
        std::array<std::unique_ptr<LLMRankRuntime::SteppedGeneration>, kRanks> generations;
        parallel([&](int32_t rank) {
            int64_t const before = mRuntimes[rank]->getMultimodalMetrics().totalImages;
            LLMRankRuntime::VisionGatherFn gather;
            if (enableDp)
            {
                gather = [&, rank](void* buffer, int32_t rank0Count, int32_t rank1Count, cudaStream_t stream) {
                    ++result.gathers[rank];
                    result.gatheredElements[rank] = {rank0Count, rank1Count};
                    return gatherVisionShards(
                        buffer, rank0Count, rank1Count, rank,
                        [&, rank](void const* send, void* recv, int32_t count, cudaStream_t callStream) {
                            return mGroup->allGather(rank, send, recv, count, CollectiveDataType::kFloat16, callStream);
                        },
                        [&, rank](void* data, int32_t count, int32_t root, cudaStream_t callStream) {
                            return mGroup->broadcast(rank, data, count, CollectiveDataType::kFloat16, root, callStream);
                        },
                        stream);
                };
            }
            generations[rank] = mRuntimes[rank]->beginGeneration(
                request, responses[rank], mStreams[rank], false,
                [&, rank](void* buffer, int32_t count, cudaStream_t stream) {
                    return mGroup->broadcastInt32(rank, buffer, count, 0, stream);
                },
                std::move(gather), rank);
            ELLM_CHECK(generations[rank] != nullptr, "Test request admission failed");
            result.encodedImages[rank] = mRuntimes[rank]->getMultimodalMetrics().totalImages - before;
        });
        // Both prefill streams have drained before allocating host snapshot buffers.
        for (int32_t rank = 0; rank < kRanks; ++rank)
        {
            CUDA_CHECK(cudaSetDevice(rank));
            auto const& context = generations[rank]->context;
            ELLM_CHECK(context.visualEmbeddings.has_value(), "Missing visual embedding");
            result.inputIds[rank] = context.rawBatchedInputIds;
            auto snapshot = [&](Tensor const& tensor) {
                Tensor host(tensor.getShape(), DeviceType::kCPU, tensor.getDataType());
                CUDA_CHECK(cudaMemcpyAsync(host.rawPointer(), tensor.rawPointer(),
                    tensor.getShape().volume() * sizeof(half), cudaMemcpyDeviceToHost, mStreams[rank]));
                CUDA_CHECK(cudaStreamSynchronize(mStreams[rank]));
                HostOutput output{tensor.getShape(), {}};
                for (int64_t i = 0; i < tensor.getShape().volume(); ++i)
                {
                    output.values.push_back(__half2float(host.dataPointer<half>()[i]));
                }
                result.outputs[rank].push_back(std::move(output));
            };
            snapshot(context.visualEmbeddings->get());
            for (auto const& feature : context.deepstackFeatures)
            {
                snapshot(feature.get());
            }
        }
        parallel([&](int32_t rank) {
            auto& generation = *generations[rank];
            {
                LLMRankRuntime::GenerationSession session(*mRuntimes[rank], generation.context, *generation.strategy,
                    generation.managedRequest(), generation.kvHeadroom, generation.jumpForwardKvHeadroom,
                    mStreams[rank], false, {});
                ELLM_CHECK(session.primeFromPrefill(), "Failed to consume prefill token");
                while (!session.finished())
                {
                    ELLM_CHECK(session.advance(), "Decode failed");
                }
            }
            ELLM_CHECK(mRuntimes[rank]->finishGeneration(generation, request, responses[rank], mStreams[rank]),
                "Failed to retire test request");
            generations[rank].reset();
        });
        return result;
    }

    DeploymentOptions mOptions;
    tokenizer::Tokenizer mTokenizer;
    chat_template::ChatTemplate mChatTemplate;
    std::unique_ptr<TensorParallelPluginResources> mResources;
    std::unique_ptr<CollectiveGroup> mGroup;
    std::array<cudaStream_t, kRanks> mStreams{};
    std::array<std::unique_ptr<LLMRankRuntime>, kRanks> mRuntimes;
};

TEST_F(VisionDataParallelEngineTest, MatchesReplicatedEmbeddingAndEveryDeepstack)
{
    // Given distinct images with equal or unequal token shards, when TP2 encodes them with DP,
    // then both ranks preserve replicated ViT's full token order, embedding and every deepstack output.
    initialize();
    std::vector<std::vector<std::pair<int32_t, int32_t>>> const cases{{{64, 64}, {64, 64}},
        std::vector<std::pair<int32_t, int32_t>>(6, {64, 64}), std::vector<std::pair<int32_t, int32_t>>(7, {64, 64}),
        {{64, 64}, {64, 128}}, {{64, 128}, {64, 64}}, std::vector<std::pair<int32_t, int32_t>>(2, {384, 1280}),
        std::vector<std::pair<int32_t, int32_t>>(6, {384, 1280}),
        std::vector<std::pair<int32_t, int32_t>>(7, {384, 1280}), {{1280, 2048}, {512, 2048}},
        {{512, 2048}, {1280, 2048}}, {{128, 256}, {64, 64}, {64, 64}, {64, 64}, {64, 64}, {64, 64}},
        {{64, 64}, {64, 64}, {64, 64}, {64, 64}, {64, 64}, {128, 256}}, {{64, 64}, {32, 64}, {32, 64}}};
    for (auto const& sizes : cases)
    {
        SCOPED_TRACE("images=" + std::to_string(sizes.size()));
        auto request = makeRequest(sizes);
        auto const expected = run(request, false);
        auto const actual = run(request, true);
        EXPECT_TRUE(sameFeatures(expected, actual));
        EXPECT_EQ(actual.gathers, (std::array<int32_t, kRanks>{kOutputs, kOutputs}));
        auto const replicatedAfterDp = run(request, false);
        EXPECT_TRUE(sameFeatures(expected, replicatedAfterDp));
    }
}

TEST_F(VisionDataParallelEngineTest, BalancesContiguousShardsByTokenCount)
{
    // Given joint images of equal or mixed sizes, when DP partitions them without reordering,
    // then the collective receives the most balanced token shards allowed by the engine profile.
    initialize();
    struct Case
    {
        std::vector<std::pair<int32_t, int32_t>> sizes;
        std::array<int32_t, kRanks> tokens;
    };
    std::vector<Case> const cases{{{{64, 64}, {64, 64}}, {4, 4}},
        {std::vector<std::pair<int32_t, int32_t>>(6, {64, 64}), {12, 12}},
        {std::vector<std::pair<int32_t, int32_t>>(7, {64, 64}), {16, 12}},
        {{{128, 256}, {64, 64}, {64, 64}, {64, 64}, {64, 64}, {64, 64}}, {32, 20}},
        {{{64, 64}, {64, 64}, {64, 64}, {64, 64}, {64, 64}, {128, 256}}, {20, 32}},
        {{{64, 64}, {32, 64}, {32, 64}}, {4, 4}}};
    for (auto const& testCase : cases)
    {
        SCOPED_TRACE("token shards=" + std::to_string(testCase.tokens[0]) + "+" + std::to_string(testCase.tokens[1]));
        auto const actual = run(makeRequest(testCase.sizes), true);
        int32_t const hiddenSize = static_cast<int32_t>(actual.outputs[0][0].shape[1]);
        std::array<int32_t, kRanks> const expected{testCase.tokens[0] * hiddenSize, testCase.tokens[1] * hiddenSize};
        for (int32_t rank = 0; rank < kRanks; ++rank)
        {
            EXPECT_EQ(actual.gatheredElements[rank], expected);
        }
    }
}

TEST_F(VisionDataParallelEngineTest, FallsBackWhenIndividualShardsAreBelowProfileMinimum)
{
    // Given a legal joint input whose shards fall below the engine minimum, when DP is available,
    // then replicated encoding is used without a collective and all visual outputs match the baseline.
    initialize();
    for (auto const& sizes : std::vector<std::vector<std::pair<int32_t, int32_t>>>{
             {{32, 64}, {32, 64}}, {{32, 64}, {64, 64}}, {{64, 64}, {32, 64}}})
    {
        auto request = makeRequest(sizes);
        auto const expected = run(request, false);
        auto const actual = run(request, true);
        EXPECT_TRUE(sameFeatures(expected, actual));
        EXPECT_EQ(actual.gathers, (std::array<int32_t, kRanks>{0, 0}));
    }
}

TEST_F(VisionDataParallelEngineTest, HonorsImageLimitDespiteSparePlanningCapacity)
{
    // Given an image limit and extra planning capacity, when a request is at or beyond that limit,
    // then DP eligibility and replicated preprocessing enforce the same configured maximum.
    class LimitedImageRunner : public Qwen3VLViTRunner
    {
    public:
        using Qwen3VLViTRunner::Qwen3VLViTRunner;

        bool validateAndFillConfig(std::string const& engineDir) override
        {
            if (!Qwen3VLViTRunner::validateAndFillConfig(engineDir))
            {
                return false;
            }
            // Narrow the loaded profile before buffers are allocated to isolate the image-count boundary.
            ELLM_CHECK(mConfig.maxNumImages >= 4, "Test engine must allow at least four images");
            mConfig.maxNumImages = 4;
            return true;
        }

        void reserveExtraPlanningCapacity()
        {
            mVisionDpPlan.spans.reserve(8);
        }
    };

    std::vector<LLMGenerationRequest> requests;
    for (size_t const count : {2, 4, 5, 8})
    {
        requests.push_back(makeRequest(std::vector<std::pair<int32_t, int32_t>>(count, {64, 64})));
    }
    parallel([&](int32_t rank) {
        LimitedImageRunner runner(mOptions.multimodalEngineDir, 1, 4096, mStreams[rank]);
        runner.initialize(mStreams[rank]);
        runner.initializeDataParallel();
        runner.reserveExtraPlanningCapacity();
        for (auto const& request : requests)
        {
            size_t const count = request.requests.front().imageBuffers.size();
            SCOPED_TRACE("rank=" + std::to_string(rank) + " images=" + std::to_string(count));
            bool const expected = count <= 4;
            EXPECT_EQ(runner.planDataParallel(request) != nullptr, expected);
            bool accepted = false;
            std::vector<std::vector<int32_t>> inputIds;
            try
            {
                accepted = runner.preprocess(request, inputIds, &mTokenizer, std::nullopt, mStreams[rank], true);
            }
            catch (std::runtime_error const& error)
            {
                if (std::string(error.what()).find("EDGELLM_INPUT_TOO_LONG: cuSeqlens") == std::string::npos)
                {
                    throw;
                }
            }
            EXPECT_EQ(accepted, expected);
        }
        CUDA_CHECK(cudaStreamSynchronize(mStreams[rank]));
    });
}

TEST_F(VisionDataParallelEngineTest, RestoresDpOutputsAfterSingleImageCacheHits)
{
    // Given cached single images and joint requests sharing a runtime, when a cache hit interrupts
    // consecutive DP requests, then every resumed DP output matches replicated ViT without stale rows.
    initialize(8LL * 1024 * 1024);
    for (auto const& sizes : std::vector<std::vector<std::pair<int32_t, int32_t>>>{
             {{64, 64}, {64, 64}}, {{64, 64}, {64, 128}}, {{64, 128}, {64, 64}}})
    {
        auto pair = makeRequest(sizes, 17);
        auto single = makeRequest({{64, 128}}, 39);
        auto const expectedPair = run(pair, false);
        auto const expectedSingle = run(single, true);
        for (int32_t repeat = 0; repeat < 3; ++repeat)
        {
            auto const first = run(pair, true);
            EXPECT_TRUE(sameFeatures(expectedPair, first));
            auto const hit = run(single, true);
            EXPECT_TRUE(sameFeatures(expectedSingle, hit));
            EXPECT_EQ(hit.encodedImages, (std::array<int64_t, kRanks>{0, 0}));
            auto const resumed = run(pair, true);
            EXPECT_TRUE(sameFeatures(expectedPair, resumed));
            EXPECT_EQ(resumed.gathers, (std::array<int32_t, kRanks>{kOutputs, kOutputs}));
        }
    }
}

} // namespace
} // namespace trt_edgellm::rt
