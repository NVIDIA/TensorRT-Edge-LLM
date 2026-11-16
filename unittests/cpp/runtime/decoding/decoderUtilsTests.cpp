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

#include "runtime/decoding/decoderUtils.h"

#include "common/cudaUtils.h"
#include "common/pagedKvTypes.h"
#include "kernels/speculative/speculativeKernels.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/decoding/decodingStrategy.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/logitBias.h"
#include "runtime/state/decodingInferenceContext.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"
#include "substituteEngine.h"
#include "testUtils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// This test uses only host-side CUDA runtime APIs; compiling it as C++ avoids
// routing unrelated host-only JSON headers through the CUDA front end.
using namespace trt_edgellm;

TEST(DecoderUtilsTests, IdentityDirectVocabMapConvertsToZeroOffsets)
{
    std::vector<int32_t> vocabMap{0, 1, 2, 3, 4};

    rt::decoder_utils::directVocabMapToOffsets(vocabMap, 5);

    EXPECT_EQ(vocabMap, std::vector<int32_t>({0, 0, 0, 0, 0}));
}

TEST(DecoderUtilsTests, PermutedDirectVocabMapConvertsToReconstructableOffsets)
{
    std::vector<int32_t> const directMap{3, 0, 4, 1, 2};
    std::vector<int32_t> offsets = directMap;

    rt::decoder_utils::directVocabMapToOffsets(offsets, 5);

    for (size_t draftTokenId = 0; draftTokenId < offsets.size(); ++draftTokenId)
    {
        EXPECT_EQ(static_cast<int32_t>(draftTokenId) + offsets[draftTokenId], directMap[draftTokenId]);
    }
}

TEST(DecoderUtilsTests, OutOfRangeDirectVocabMapEntryThrows)
{
    std::vector<int32_t> vocabMap{0, 5, 2};

    EXPECT_THROW(rt::decoder_utils::directVocabMapToOffsets(vocabMap, 5), std::runtime_error);
}

TEST(DecoderUtilsTests, FinishedSlotsIgnoreAcceptedTokens)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    rt::DecodingInferenceContext context;
    context.initialize(2, 8, std::nullopt, rt::OptionalInputTensors{}, "", stream);
    context.finishedStates = {1, 0};
    context.tokenIds = {{100}, {200}};
    context.currentGenerateLengths = {1, 1};
    context.committedLengths = {10, 20};
    context.shouldStopAfterAcceptedToken = [](int32_t, int32_t) { return false; };

    rt::Tensor hostAcceptLengths({2}, rt::DeviceType::kCPU, DataType::kINT32);
    rt::Tensor hostAcceptedTokenIds({2, 2}, rt::DeviceType::kCPU, DataType::kINT32);
    rt::Tensor deviceAcceptLengths({2}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor deviceAcceptedTokenIds({2, 2}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<int32_t>(deviceAcceptLengths, {2, 1});
    copyHostToDevice<int32_t>(deviceAcceptedTokenIds, {101, 102, 201, 202});

    tokenizer::Tokenizer tokenizer;
    rt::decoder_utils::appendAcceptedTokens(context, hostAcceptLengths, hostAcceptedTokenIds, deviceAcceptLengths,
        deviceAcceptedTokenIds, 2, tokenizer, stream);

    EXPECT_EQ(context.tokenIds[0], std::vector<int32_t>({100}));
    EXPECT_EQ(context.tokenIds[1], std::vector<int32_t>({200, 201}));
    EXPECT_EQ(context.currentGenerateLengths, std::vector<int32_t>({1, 2}));
    EXPECT_EQ(context.committedLengths, std::vector<int32_t>({10, 21}));
    EXPECT_EQ(hostAcceptLengths.dataPointer<int32_t>()[0], 0);
    EXPECT_EQ(hostAcceptLengths.dataPointer<int32_t>()[1], 1);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(DecoderUtilsTests, ClampAcceptLengthsToRemainingGenerationClampsOnDevice)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    rt::DecodingInferenceContext context;
    context.initialize(3, 8, std::nullopt, rt::OptionalInputTensors{}, "", stream);
    context.currentGenerateLengths = {6, 7, 9};

    rt::Tensor deviceAcceptLengths({3}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<int32_t>(deviceAcceptLengths, {5, 1, 4});

    rt::decoder_utils::clampAcceptLengthsToRemainingGeneration(context, deviceAcceptLengths, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // remaining = {2, 1, 0}: slot 0 is cut to its budget, slot 1 fits, slot 2 is already over budget.
    EXPECT_EQ(copyDeviceToHost<int32_t>(deviceAcceptLengths), std::vector<int32_t>({2, 1, 0}));

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(DecoderUtilsTests, ClampAcceptLengthsToRemainingGenerationCoversBatchesBeyondOneLaunch)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int32_t batchSize = kernel::kMaxAcceptLengthBudgetsPerLaunch + 6;
    constexpr int32_t maxGenerateLength = 100;
    rt::DecodingInferenceContext context;
    context.initialize(batchSize, maxGenerateLength, std::nullopt, rt::OptionalInputTensors{}, "", stream);
    std::vector<int32_t> expected(batchSize);
    for (int32_t slot = 0; slot < batchSize; ++slot)
    {
        int32_t const remaining = slot % 5;
        context.currentGenerateLengths[slot] = maxGenerateLength - remaining;
        expected[slot] = std::min(3, remaining);
    }

    rt::Tensor deviceAcceptLengths({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<int32_t>(deviceAcceptLengths, std::vector<int32_t>(batchSize, 3));

    rt::decoder_utils::clampAcceptLengthsToRemainingGeneration(context, deviceAcceptLengths, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(deviceAcceptLengths), expected);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(DecoderUtilsTests, SpecPrefillPreservesRestoredPrefixFrontier)
{
    rt::DecodingInferenceContext context;
    context.activeBatchSize = 2;
    context.effectivePrefillLengths = {3, 2};
    context.prefillStartLengths = {0, 5};
    context.residentRefs = {rt::ResidentRef{4, 1}, rt::ResidentRef{1, 2}};

    rt::RaggedExecutionBatch const batch = rt::decoder_utils::buildSpecPrefillRaggedBatch(context, 4);

    EXPECT_EQ(batch.shape.numSequences, 2);
    EXPECT_EQ(batch.shape.validTokens, 5);
    EXPECT_EQ(batch.shape.physicalTokens, 8);
    EXPECT_EQ(batch.shape.queryWidth, 4);
    EXPECT_EQ(batch.queryStartOffsets, std::vector<int32_t>({0, 4, 8}));
    EXPECT_EQ(batch.queryLengths, std::vector<int32_t>({3, 2}));
    EXPECT_EQ(batch.pastLengths, std::vector<int32_t>({0, 5}));
    EXPECT_EQ(batch.attentionSequenceLengths, std::vector<int32_t>({3, 7}));
    EXPECT_EQ(batch.positions, std::vector<int32_t>({0, 1, 2, -1, 5, 6, -1, -1}));
    EXPECT_EQ(batch.stateIndices, std::vector<int32_t>({4, 1}));
    EXPECT_EQ(batch.logitsIndices, std::vector<int64_t>({2, 5}));
}

TEST(DecoderUtilsTests, ContextBatchWithColdAndRestoredRowsUsesChunkPhase)
{
    EXPECT_EQ(rt::decoder_utils::contextPrefillPhase({0, 0}, 2), rt::ExecutionPhase::kContextPrefill);
    EXPECT_EQ(rt::decoder_utils::contextPrefillPhase({0, 5}, 2), rt::ExecutionPhase::kContextChunk);
    EXPECT_THROW(rt::decoder_utils::contextPrefillPhase({0}, 2), std::runtime_error);
    EXPECT_THROW(rt::decoder_utils::contextPrefillPhase({0, -1}, 2), std::runtime_error);
    EXPECT_THROW(rt::decoder_utils::contextPrefillPhase({5, -1}, 2), std::runtime_error);
}

TEST(DecoderUtilsTests, SpecPrefillRejectsPhysicalTokenOverflow)
{
    rt::DecodingInferenceContext context;
    context.activeBatchSize = std::numeric_limits<int32_t>::max();
    EXPECT_THROW(rt::decoder_utils::buildSpecPrefillRaggedBatch(context, 2), std::runtime_error);
}

// Given a sequence after a speculative round, whose last token is the root the next round verifies,
// When the tail the n-gram history rewinds to is taken,
// Then it is the last tokens the engine was fed, leaving the pending root out.
TEST(DecoderUtilsTests, ConsumedTokenTailLeavesOutThePendingRoot)
{
    std::vector<int32_t> tail;
    rt::decoder_utils::consumedTokenTail(
        {1, 2, 3, 4, 10, 11, 12}, {1, 2, 3, 4}, /*committedLength=*/6, /*contextLength=*/2, tail);

    EXPECT_EQ(tail, std::vector<int32_t>({10, 11}));
}

// Given a sequence whose prompt prefix came from the context cache, so its token list starts after it,
// When fewer tokens were fed here than the hash reaches back,
// Then the tail continues into the reused prefix.
TEST(DecoderUtilsTests, ConsumedTokenTailReachesIntoAReusedPrefix)
{
    std::vector<int32_t> tail;
    // Prompt 1..6 with 1..5 reused: 6 was fed here, then 20 became the root.
    rt::decoder_utils::consumedTokenTail({6, 20}, {1, 2, 3, 4, 5, 6}, /*committedLength=*/6, /*contextLength=*/3, tail);

    EXPECT_EQ(tail, std::vector<int32_t>({4, 5, 6}));
}

// Given a spec-decode runtime whose base PipelineIO holds the base engine's prefill and decode skip-softmax tables
// and a draft config with no tables, When the spec ragged bindings are prepared with the draft config,
// Then both table tensors keep their address and contents: draft geometry never reaches the base engine's bindings.
TEST(DecoderUtilsTests, SpecRaggedBindingsLeaveSkipSoftmaxTablesUntouched)
{
    using namespace trt_edgellm::rt;
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    LLMEngineConfig base;
    base.hiddenSize = 16;
    base.outputVocabSize = 32;
    base.numAttentionLayers = 1;
    base.numDecoderLayers = 1;
    base.numKVHeads = 1;
    base.headDim = 8;
    base.rotaryDim = 8;
    base.maxSupportedBatchSize = 2;
    base.maxSupportedInputLength = 4;
    base.maxKVCacheCapacity = 256;
    base.maxPhysicalTokens = 8;
    base.maxNumSequences = 2;
    base.recurrentPoolRows = 2;
    base.kvPoolPages
        = static_cast<int32_t>(computeMinimumKvPoolPages(base.maxSupportedBatchSize, base.maxKVCacheCapacity));
    base.kvCacheDtype = DataType::kHALF;
    base.ropeConfig.type = RopeType::kNoRope;
    base.layerTypes.assign(1, HybridCacheManager::LayerType::kAttention);
    base.kvLayerConfigs = {KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8}};
    base.kvSharingDonors = {-1};
    LLMEngineConfig const draft = base;
    base.skipSoftmaxPrefillScaleTable = {{0, 6000}};
    base.skipSoftmaxDecodeScaleTable = {{2048, 614}, {4096, 1228}};

    DeploymentConfig deployment{};
    deployment.base = base;
    deployment.draft.emplace(draft);
    deployment.specConfig.emplace();
    std::unordered_map<std::string, std::string> const noLoraWeights;
    std::unique_ptr<SharedResources> shared = SharedResources::createForLLM(base, noLoraWeights, nullptr);
    PipelineIO io = PipelineIO::createForLLM(base, stream);
    substitute_engine::MockEngineExecutor executor;
    TensorMap tensorMap;
    EmbeddingData embedding;
    EmbeddingPreprocessor embeddingPreprocessor(embedding, deployment.base);
    StepPreparer stepPreparer(deployment.base);
    tokenizer::Tokenizer tokenizer;
    GuidedDecoder guided;
    LogitBias bias;
    Tensor unused;
    std::filesystem::path checkpoint;
    DecodingRuntimeContext runtime{deployment, 2, checkpoint, checkpoint,
        {executor, tensorMap, *shared, *shared->cacheManagers.at(0), io, {}},
        {stepPreparer, embeddingPreprocessor, embedding, unused, nullptr, nullptr, nullptr}, tokenizer, bias, guided,
        {unused, unused, unused, unused, unused, unused, unused, unused, unused},
        {unused, unused, unused, unused, unused}};

    void const* const prefillAddress = io.skipSoftmaxPrefillScale.rawPointer();
    void const* const decodeAddress = io.skipSoftmaxDecodeScale.rawPointer();
    Tensor attentionPositions({2, 1}, rt::DeviceType::kGPU, DataType::kINT32);
    Tensor committedPastLengths({2}, rt::DeviceType::kGPU, DataType::kINT32);
    Tensor selectedTokenIndices({2}, rt::DeviceType::kGPU, DataType::kINT64);
    copyHostToDevice<int32_t>(attentionPositions, {4, 7});
    copyHostToDevice<int32_t>(committedPastLengths, {4, 7});
    copyHostToDevice<int64_t>(selectedTokenIndices, {0, 1});

    rt::decoder_utils::prepareSpecRaggedBindings(runtime, *deployment.draft, /*kvCacheIndex=*/0, attentionPositions,
        committedPastLengths, /*validCounts=*/nullptr, selectedTokenIndices, /*selectedRows=*/2,
        /*residentRefs=*/nullptr, /*batchSize=*/2, /*queryWidth=*/1, deployment.draft->decodeDims(2), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(io.skipSoftmaxPrefillScale.rawPointer(), prefillAddress);
    EXPECT_EQ(io.skipSoftmaxDecodeScale.rawPointer(), decodeAddress);
    EXPECT_EQ(copyDeviceToHost<int32_t>(io.skipSoftmaxPrefillScale), std::vector<int32_t>({0, 6000}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(io.skipSoftmaxDecodeScale), std::vector<int32_t>({2048, 614, 4096, 1228}));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
