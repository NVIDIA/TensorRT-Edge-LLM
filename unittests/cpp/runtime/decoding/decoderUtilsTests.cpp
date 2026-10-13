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
#include "kernels/speculative/eagleUtilKernels.h"
#include "runtime/state/decodingInferenceContext.h"
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
using namespace nvinfer1;

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
