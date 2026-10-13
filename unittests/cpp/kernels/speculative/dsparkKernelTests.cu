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

#include "common/cudaUtils.h"
#include "kernels/speculative/dsparkKernels.h"
#include "sampler/sampling.h"
#include "testUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;
using namespace nvinfer1;

namespace
{

std::vector<half> toHalf(std::vector<float> const& values)
{
    std::vector<half> out(values.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        out[i] = __float2half(values[i]);
    }
    return out;
}

void expectProbabilityRow(std::vector<float> const& actual, std::vector<float> const& expected, float atol = 1e-5F)
{
    ASSERT_EQ(actual.size(), expected.size());
    float sum = 0.0F;
    for (size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], atol) << "token=" << i;
        sum += actual[i];
    }
    EXPECT_NEAR(sum, 1.0F, 1e-5F);
}

} // namespace

TEST(DSparkKernels, LogitsToProbabilitiesAppliesTopK)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t rows = 1;
    constexpr int32_t vocabSize = 4;

    auto logits = rt::Tensor({rows, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto probabilities = rt::Tensor({rows, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);

    copyHostToDevice<float>(logits, {3.0F, 2.0F, 1.0F, 0.0F});

    dsparkLogitsToProbabilities(
        logits, probabilities, rows, vocabSize, /*temperature=*/1.0F, /*topK=*/2, /*topP=*/1.0F, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float const denom = std::exp(3.0F) + std::exp(2.0F);
    expectProbabilityRow(
        copyDeviceToHost<float>(probabilities), {std::exp(3.0F) / denom, std::exp(2.0F) / denom, 0.0F, 0.0F});
}

TEST(DSparkKernels, TopKLogitsToProbabilitiesScattersDenseDistribution)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t rows = 1;
    constexpr int32_t vocabSize = 4;
    constexpr int32_t topK = 2;

    auto topKValues = rt::Tensor({rows, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto topKIndices = rt::Tensor({rows, topK}, rt::DeviceType::kGPU, DataType::kINT32);
    auto probabilities = rt::Tensor({rows, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);

    copyHostToDevice<float>(topKValues, {3.0F, 2.0F});
    copyHostToDevice<int32_t>(topKIndices, {0, 1});

    topKLogitsToDenseProbabilities(topKValues, topKIndices, probabilities, vocabSize, /*temperature=*/1.0F, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float const denom = std::exp(3.0F) + std::exp(2.0F);
    expectProbabilityRow(
        copyDeviceToHost<float>(probabilities), {std::exp(3.0F) / denom, std::exp(2.0F) / denom, 0.0F, 0.0F});
}

TEST(DSparkKernels, LogitsToProbabilitiesTemperatureZeroIsGreedy)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t rows = 1;
    constexpr int32_t vocabSize = 4;

    auto logits = rt::Tensor({rows, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto probabilities = rt::Tensor({rows, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);

    copyHostToDevice<float>(logits, {-1.0F, 0.5F, 4.0F, 3.0F});

    dsparkLogitsToProbabilities(
        logits, probabilities, rows, vocabSize, /*temperature=*/0.0F, /*topK=*/0, /*topP=*/0.5F, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    expectProbabilityRow(copyDeviceToHost<float>(probabilities), {0.0F, 0.0F, 1.0F, 0.0F});
}

TEST(DSparkKernels, FillUniformsIsDeterministicAndBounded)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t totalElements = 16;

    auto uniformsA = rt::Tensor({totalElements}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto uniformsB = rt::Tensor({totalElements}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto uniformsC = rt::Tensor({totalElements}, rt::DeviceType::kGPU, DataType::kFLOAT);

    dsparkFillUniforms(uniformsA, totalElements, /*philoxSeed=*/1234ULL, /*philoxOffset=*/56ULL, stream);
    dsparkFillUniforms(uniformsB, totalElements, /*philoxSeed=*/1234ULL, /*philoxOffset=*/56ULL, stream);
    dsparkFillUniforms(uniformsC, totalElements, /*philoxSeed=*/1234ULL, /*philoxOffset=*/57ULL, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const valuesA = copyDeviceToHost<float>(uniformsA);
    auto const valuesB = copyDeviceToHost<float>(uniformsB);
    auto const valuesC = copyDeviceToHost<float>(uniformsC);
    bool sawOffsetDifference = false;
    for (int32_t idx = 0; idx < totalElements; ++idx)
    {
        EXPECT_GE(valuesA[idx], 0.0F);
        EXPECT_LT(valuesA[idx], 1.0F);
        EXPECT_EQ(valuesA[idx], valuesB[idx]);
        sawOffsetDifference = sawOffsetDifference || (valuesA[idx] != valuesC[idx]);
    }
    EXPECT_TRUE(sawOffsetDifference);
}

TEST(DSparkKernels, VanillaMarkovSampleMaterializesDistribution)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 1;
    constexpr int32_t proposalLen = 2;
    constexpr int32_t vocabSize = 4;
    constexpr int32_t markovRank = 1;

    auto backboneLogits = rt::Tensor({batchSize, proposalLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto markovW1 = rt::Tensor({vocabSize, markovRank}, rt::DeviceType::kGPU, DataType::kHALF);
    auto markovW2 = rt::Tensor({vocabSize, markovRank}, rt::DeviceType::kGPU, DataType::kHALF);
    auto firstPrevTokens = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto uniforms = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftTokenIds = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftProbabilities = rt::Tensor({batchSize, proposalLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto stepLogits = rt::Tensor({batchSize, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto stepProbabilities = rt::Tensor({batchSize, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);

    copyHostToDevice<float>(backboneLogits,
        {
            0.0F,
            2.0F,
            1.0F,
            -1.0F,
            0.0F,
            0.0F,
            3.0F,
            1.0F,
        });
    copyHostToDevice<half>(markovW1, toHalf({0.0F, 0.0F, 0.0F, 0.0F}));
    copyHostToDevice<half>(markovW2, toHalf({0.0F, 0.0F, 0.0F, 0.0F}));
    copyHostToDevice<int32_t>(firstPrevTokens, {0});
    copyHostToDevice<float>(uniforms, {0.0F, 0.99F});

    dsparkVanillaMarkovSample(backboneLogits, markovW1, markovW2, firstPrevTokens, uniforms, draftTokenIds,
        draftProbabilities, stepLogits, stepProbabilities, batchSize, proposalLen, vocabSize, markovRank,
        /*temperature=*/1.0F, /*topK=*/2, /*topP=*/1.0F, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const tokens = copyDeviceToHost<int32_t>(draftTokenIds);
    EXPECT_EQ(tokens, (std::vector<int32_t>{1, 3}));

    auto const probs = copyDeviceToHost<float>(draftProbabilities);
    float const step0Denom = std::exp(2.0F) + std::exp(1.0F);
    expectProbabilityRow(std::vector<float>(probs.begin(), probs.begin() + vocabSize),
        {0.0F, std::exp(2.0F) / step0Denom, std::exp(1.0F) / step0Denom, 0.0F});
    float const step1Denom = std::exp(3.0F) + std::exp(1.0F);
    expectProbabilityRow(std::vector<float>(probs.begin() + vocabSize, probs.end()),
        {0.0F, 0.0F, std::exp(3.0F) / step1Denom, std::exp(1.0F) / step1Denom});
}

TEST(DSparkKernels, VanillaMarkovSampleTopKMaterializesDistribution)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 1;
    constexpr int32_t proposalLen = 2;
    constexpr int32_t vocabSize = 4;
    constexpr int32_t markovRank = 1;
    constexpr int32_t topK = 2;

    auto backboneLogits = rt::Tensor({batchSize, proposalLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto markovW1 = rt::Tensor({vocabSize, markovRank}, rt::DeviceType::kGPU, DataType::kHALF);
    auto markovW2 = rt::Tensor({vocabSize, markovRank}, rt::DeviceType::kGPU, DataType::kHALF);
    auto firstPrevTokens = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto uniforms = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftTokenIds = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftProbabilities = rt::Tensor({batchSize, proposalLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto stepLogits = rt::Tensor({batchSize, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto stepProbabilities = rt::Tensor({batchSize, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto topKValues = rt::Tensor({batchSize, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto topKIndices = rt::Tensor({batchSize, topK}, rt::DeviceType::kGPU, DataType::kINT32);
    auto workspace = rt::Tensor({static_cast<int64_t>(getSelectAllTopKWorkspaceSize(batchSize, vocabSize, topK))},
        rt::DeviceType::kGPU, DataType::kINT8);

    copyHostToDevice<float>(backboneLogits,
        {
            0.0F,
            2.0F,
            1.0F,
            -1.0F,
            0.0F,
            0.0F,
            3.0F,
            1.0F,
        });
    copyHostToDevice<half>(markovW1, toHalf({0.0F, 0.0F, 0.0F, 0.0F}));
    copyHostToDevice<half>(markovW2, toHalf({0.0F, 0.0F, 0.0F, 0.0F}));
    copyHostToDevice<int32_t>(firstPrevTokens, {0});
    copyHostToDevice<float>(uniforms, {0.0F, 0.99F});

    for (int32_t step = 0; step < proposalLen; ++step)
    {
        dsparkBuildMarkovLogits(backboneLogits, markovW1, markovW2, firstPrevTokens, draftTokenIds, stepLogits,
            batchSize, step, proposalLen, vocabSize, markovRank, stream);
        selectAllTopK(stepLogits, std::ref(topKValues), topKIndices, topK, workspace, stream);
        topKLogitsToDenseProbabilities(
            topKValues, topKIndices, stepProbabilities, vocabSize, /*temperature=*/1.0F, stream);
        dsparkSampleProbabilityRows(
            stepProbabilities, uniforms, draftTokenIds, batchSize, step, proposalLen, vocabSize, stream);
        dsparkStoreDraftStepProbabilities(
            stepProbabilities, draftProbabilities, batchSize, step, proposalLen, vocabSize, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const tokens = copyDeviceToHost<int32_t>(draftTokenIds);
    EXPECT_EQ(tokens, (std::vector<int32_t>{1, 3}));

    auto const probs = copyDeviceToHost<float>(draftProbabilities);
    float const step0Denom = std::exp(2.0F) + std::exp(1.0F);
    expectProbabilityRow(std::vector<float>(probs.begin(), probs.begin() + vocabSize),
        {0.0F, std::exp(2.0F) / step0Denom, std::exp(1.0F) / step0Denom, 0.0F});
    float const step1Denom = std::exp(3.0F) + std::exp(1.0F);
    expectProbabilityRow(std::vector<float>(probs.begin() + vocabSize, probs.end()),
        {0.0F, 0.0F, std::exp(3.0F) / step1Denom, std::exp(1.0F) / step1Denom});
}

TEST(DSparkKernels, ProbabilisticAcceptSamplesResidualOnReject)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 1;
    constexpr int32_t proposalLen = 2;
    constexpr int32_t verifyLen = proposalLen + 1;
    constexpr int32_t vocabSize = 4;

    auto targetProbabilities = rt::Tensor({batchSize, verifyLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftProbabilities = rt::Tensor({batchSize, proposalLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftTokenIds = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto uniforms = rt::Tensor({batchSize, 2 * proposalLen + 1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto proposalLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptedTokenIds = rt::Tensor({batchSize, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptLength = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<float>(targetProbabilities,
        {
            0.10F,
            0.80F,
            0.05F,
            0.05F,
            0.50F,
            0.20F,
            0.10F,
            0.20F,
            0.25F,
            0.25F,
            0.25F,
            0.25F,
        });
    copyHostToDevice<float>(draftProbabilities,
        {
            0.20F,
            0.40F,
            0.20F,
            0.20F,
            0.20F,
            0.30F,
            0.50F,
            0.00F,
        });
    copyHostToDevice<int32_t>(draftTokenIds, {1, 2});
    copyHostToDevice<float>(uniforms, {0.5F, 0.9F, 0.0F, 0.7F, 0.0F});
    copyHostToDevice<int32_t>(proposalLengths, {proposalLen});

    dsparkProbabilisticAccept(targetProbabilities, draftProbabilities, draftTokenIds, proposalLengths, uniforms,
        acceptedTokenIds, acceptLength, batchSize, proposalLen, proposalLen, vocabSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(acceptLength), (std::vector<int32_t>{2}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(acceptedTokenIds), (std::vector<int32_t>{1, 3, 0}));
}

TEST(DSparkKernels, ProbabilisticAcceptSamplesBonusWhenAllAccepted)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 1;
    constexpr int32_t proposalLen = 2;
    constexpr int32_t verifyLen = proposalLen + 1;
    constexpr int32_t vocabSize = 4;

    auto targetProbabilities = rt::Tensor({batchSize, verifyLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftProbabilities = rt::Tensor({batchSize, proposalLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftTokenIds = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto uniforms = rt::Tensor({batchSize, 2 * proposalLen + 1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto proposalLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptedTokenIds = rt::Tensor({batchSize, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptLength = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<float>(targetProbabilities,
        {
            0.10F,
            0.80F,
            0.05F,
            0.05F,
            0.10F,
            0.20F,
            0.60F,
            0.10F,
            0.10F,
            0.20F,
            0.30F,
            0.40F,
        });
    copyHostToDevice<float>(draftProbabilities,
        {
            0.10F,
            0.40F,
            0.40F,
            0.10F,
            0.20F,
            0.20F,
            0.20F,
            0.40F,
        });
    copyHostToDevice<int32_t>(draftTokenIds, {1, 2});
    copyHostToDevice<float>(uniforms, {0.9F, 0.9F, 0.0F, 0.0F, 0.75F});
    copyHostToDevice<int32_t>(proposalLengths, {proposalLen});

    dsparkProbabilisticAccept(targetProbabilities, draftProbabilities, draftTokenIds, proposalLengths, uniforms,
        acceptedTokenIds, acceptLength, batchSize, proposalLen, proposalLen, vocabSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(acceptLength), (std::vector<int32_t>{3}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(acceptedTokenIds), (std::vector<int32_t>{1, 2, 3}));
}

TEST(DSparkKernels, SparseTopKSampleStoresDistribution)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 1;
    constexpr int32_t proposalLen = 2;
    constexpr int32_t topK = 2;

    auto topKValues = rt::Tensor({batchSize, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto topKIndices = rt::Tensor({batchSize, topK}, rt::DeviceType::kGPU, DataType::kINT32);
    auto uniforms = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftTokenIds = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftTopKProbabilities = rt::Tensor({batchSize, proposalLen, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftTopKIndices = rt::Tensor({batchSize, proposalLen, topK}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<float>(uniforms, {0.0F, 0.99F});

    copyHostToDevice<float>(topKValues, {2.0F, 1.0F});
    copyHostToDevice<int32_t>(topKIndices, {1, 2});
    dsparkSampleTopKRowsAndStore(topKValues, topKIndices, uniforms, draftTokenIds, draftTopKProbabilities,
        draftTopKIndices, batchSize, /*step=*/0, proposalLen, topK, /*temperature=*/1.0F, stream);

    copyHostToDevice<float>(topKValues, {3.0F, 1.0F});
    copyHostToDevice<int32_t>(topKIndices, {2, 3});
    dsparkSampleTopKRowsAndStore(topKValues, topKIndices, uniforms, draftTokenIds, draftTopKProbabilities,
        draftTopKIndices, batchSize, /*step=*/1, proposalLen, topK, /*temperature=*/1.0F, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(draftTokenIds), (std::vector<int32_t>{1, 3}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(draftTopKIndices), (std::vector<int32_t>{1, 2, 2, 3}));

    auto const probs = copyDeviceToHost<float>(draftTopKProbabilities);
    float const step0Denom = std::exp(2.0F) + std::exp(1.0F);
    EXPECT_NEAR(probs[0], std::exp(2.0F) / step0Denom, 1e-5F);
    EXPECT_NEAR(probs[1], std::exp(1.0F) / step0Denom, 1e-5F);
    float const step1Denom = std::exp(3.0F) + std::exp(1.0F);
    EXPECT_NEAR(probs[2], std::exp(3.0F) / step1Denom, 1e-5F);
    EXPECT_NEAR(probs[3], std::exp(1.0F) / step1Denom, 1e-5F);
}

TEST(DSparkKernels, ConfidenceThresholdSchedulerSelectsPrefix)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 1;
    constexpr int32_t proposalLen = 3;
    constexpr int32_t hiddenSize = 2;
    constexpr int32_t vocabSize = 4;
    constexpr int32_t markovRank = 1;

    auto hiddenStates = rt::Tensor({batchSize, proposalLen, hiddenSize}, rt::DeviceType::kGPU, DataType::kHALF);
    auto markovW1 = rt::Tensor({vocabSize, markovRank}, rt::DeviceType::kGPU, DataType::kHALF);
    auto confidenceWeight = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, DataType::kHALF);
    auto confidenceBias = rt::Tensor({1}, rt::DeviceType::kGPU, DataType::kHALF);
    auto firstPrevTokens = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftTokenIds = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto confidenceScores = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto proposalLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<half>(hiddenStates, toHalf({2.0F, 0.0F, 1.0F, 0.0F, -2.0F, 0.0F}));
    copyHostToDevice<half>(markovW1, toHalf({0.0F, 0.0F, 0.0F, 0.0F}));
    copyHostToDevice<half>(confidenceWeight, toHalf({1.0F, 0.0F}));
    copyHostToDevice<half>(confidenceBias, toHalf({0.0F}));
    copyHostToDevice<int32_t>(firstPrevTokens, {0});
    copyHostToDevice<int32_t>(draftTokenIds, {1, 2, 3});

    dsparkComputeConfidenceAndProposalLengths(hiddenStates, markovW1, confidenceWeight, confidenceBias, firstPrevTokens,
        draftTokenIds, confidenceScores, proposalLengths, batchSize, proposalLen, hiddenSize, markovRank,
        /*confidenceWithMarkov=*/false, /*threshold=*/0.5F, /*minProposalLen=*/1, /*maxProposalLen=*/proposalLen,
        stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const scores = copyDeviceToHost<float>(confidenceScores);
    EXPECT_NEAR(scores[0], 1.0F / (1.0F + std::exp(-2.0F)), 1e-5F);
    EXPECT_NEAR(scores[1], 1.0F / (1.0F + std::exp(-1.0F)), 1e-5F);
    EXPECT_NEAR(scores[2], 1.0F / (1.0F + std::exp(2.0F)), 1e-5F);
    EXPECT_EQ(copyDeviceToHost<int32_t>(proposalLengths), (std::vector<int32_t>{2}));
}

TEST(DSparkKernels, ConfidenceSPSSchedulerSelectsBestExpectedThroughputPrefix)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 1;
    constexpr int32_t proposalLen = 4;
    constexpr int32_t hiddenSize = 1;
    constexpr int32_t vocabSize = 4;
    constexpr int32_t markovRank = 1;

    auto hiddenStates = rt::Tensor({batchSize, proposalLen, hiddenSize}, rt::DeviceType::kGPU, DataType::kHALF);
    auto markovW1 = rt::Tensor({vocabSize, markovRank}, rt::DeviceType::kGPU, DataType::kHALF);
    auto confidenceWeight = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, DataType::kHALF);
    auto confidenceBias = rt::Tensor({1}, rt::DeviceType::kGPU, DataType::kHALF);
    auto firstPrevTokens = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftTokenIds = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto confidenceScores = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto proposalLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<half>(hiddenStates, toHalf({3.0F, 0.0F, 0.0F, 0.0F}));
    copyHostToDevice<half>(markovW1, toHalf({0.0F, 0.0F, 0.0F, 0.0F}));
    copyHostToDevice<half>(confidenceWeight, toHalf({1.0F}));
    copyHostToDevice<half>(confidenceBias, toHalf({0.0F}));
    copyHostToDevice<int32_t>(firstPrevTokens, {0});
    copyHostToDevice<int32_t>(draftTokenIds, {1, 2, 3, 0});

    dsparkComputeConfidenceAndSPSProposalLengths(hiddenStates, markovW1, confidenceWeight, confidenceBias,
        firstPrevTokens, draftTokenIds, confidenceScores, proposalLengths, batchSize, proposalLen, hiddenSize,
        markovRank, /*confidenceWithMarkov=*/false, /*survivalFloor=*/0.0F, /*minProposalLen=*/1,
        /*maxProposalLen=*/proposalLen, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const scores = copyDeviceToHost<float>(confidenceScores);
    EXPECT_NEAR(scores[0], 1.0F / (1.0F + std::exp(-3.0F)), 1e-5F);
    EXPECT_NEAR(scores[1], 0.5F, 1e-5F);
    EXPECT_EQ(copyDeviceToHost<int32_t>(proposalLengths), (std::vector<int32_t>{2}));
}

TEST(DSparkKernels, BuildVerifyTokensUsesDraftStrideForDynamicVerifyLength)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 1;
    constexpr int32_t draftStride = 3;
    constexpr int32_t verifyProposalLen = 1;
    constexpr int32_t verifyLen = verifyProposalLen + 1;

    auto lastAcceptedTokens = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftTokenIds = rt::Tensor({batchSize, draftStride}, rt::DeviceType::kGPU, DataType::kINT32);
    auto verifyTokenIds = rt::Tensor({batchSize, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<int32_t>(lastAcceptedTokens, {5});
    copyHostToDevice<int32_t>(draftTokenIds, {7, 8, 9});

    dsparkBuildVerifyTokens(
        lastAcceptedTokens, draftTokenIds, verifyTokenIds, batchSize, draftStride, verifyProposalLen, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(verifyTokenIds), (std::vector<int32_t>{5, 7}));
}

TEST(DSparkKernels, ProbabilisticAcceptSupportsDynamicVerifyLength)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 1;
    constexpr int32_t draftStride = 2;
    constexpr int32_t verifyProposalLen = 1;
    constexpr int32_t verifyLen = verifyProposalLen + 1;
    constexpr int32_t vocabSize = 4;

    auto targetProbabilities = rt::Tensor({batchSize, verifyLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftProbabilities = rt::Tensor({batchSize, draftStride, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftTokenIds = rt::Tensor({batchSize, draftStride}, rt::DeviceType::kGPU, DataType::kINT32);
    auto uniforms = rt::Tensor({batchSize, 2 * draftStride + 1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto proposalLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptedTokenIds = rt::Tensor({batchSize, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptLength = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<float>(targetProbabilities,
        {
            0.10F,
            0.80F,
            0.05F,
            0.05F,
            0.10F,
            0.20F,
            0.30F,
            0.40F,
        });
    copyHostToDevice<float>(draftProbabilities,
        {
            0.10F,
            0.40F,
            0.40F,
            0.10F,
            0.25F,
            0.25F,
            0.25F,
            0.25F,
        });
    copyHostToDevice<int32_t>(draftTokenIds, {1, 2});
    copyHostToDevice<float>(uniforms, {0.9F, 0.0F, 0.0F, 0.0F, 0.75F});
    copyHostToDevice<int32_t>(proposalLengths, {verifyProposalLen});

    dsparkProbabilisticAccept(targetProbabilities, draftProbabilities, draftTokenIds, proposalLengths, uniforms,
        acceptedTokenIds, acceptLength, batchSize, draftStride, verifyProposalLen, vocabSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(acceptLength), (std::vector<int32_t>{2}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(acceptedTokenIds), (std::vector<int32_t>{1, 3}));
}

TEST(DSparkKernels, FusedGreedyStepMatchesCpuReference)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 2;
    constexpr int32_t proposalLen = 3;
    constexpr int32_t vocabSize = 131; // multiple fused blocks + tail
    constexpr int32_t markovRank = 512;
    constexpr int32_t depthSize = proposalLen + 1;

    ASSERT_TRUE(dsparkFusedGreedySupported(markovRank));

    std::vector<float> backboneHost(batchSize * proposalLen * vocabSize);
    std::vector<float> w1Host(vocabSize * markovRank);
    std::vector<float> w2Host(vocabSize * markovRank);
    for (size_t i = 0; i < backboneHost.size(); ++i)
    {
        backboneHost[i] = std::sin(static_cast<float>(i) * 0.61F) * 5.0F;
    }
    for (size_t i = 0; i < w1Host.size(); ++i)
    {
        w1Host[i] = std::cos(static_cast<float>(i) * 0.29F);
        w2Host[i] = std::sin(static_cast<float>(i) * 0.83F) * 0.4F;
    }

    auto backboneLogits = rt::Tensor({batchSize, proposalLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto markovW1 = rt::Tensor({vocabSize, markovRank}, rt::DeviceType::kGPU, DataType::kHALF);
    auto markovW2 = rt::Tensor({vocabSize, markovRank}, rt::DeviceType::kGPU, DataType::kHALF);
    auto firstPrevTokens = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto greedySlots = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT64);
    auto draftTokenIds = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto stackedLogits = rt::Tensor({batchSize, depthSize, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);

    copyHostToDevice<float>(backboneLogits, backboneHost);
    copyHostToDevice<half>(markovW1, toHalf(w1Host));
    copyHostToDevice<half>(markovW2, toHalf(w2Host));
    copyHostToDevice<int32_t>(firstPrevTokens, {3, 77});
    CUDA_CHECK(cudaMemsetAsync(greedySlots.rawPointer(), 0, batchSize * proposalLen * sizeof(uint64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(stackedLogits.rawPointer(), 0, stackedLogits.getMemoryCapacity(), stream));

    for (int32_t step = 0; step < proposalLen; ++step)
    {
        dsparkMarkovGreedyFusedStep(backboneLogits, markovW1, markovW2, firstPrevTokens, greedySlots, &stackedLogits,
            step + 1, batchSize, step, proposalLen, vocabSize, markovRank, stream);
    }
    dsparkFinalizeGreedyDraftTokens(greedySlots, draftTokenIds, batchSize * proposalLen, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const gpuTokens = copyDeviceToHost<int32_t>(draftTokenIds);
    auto const gpuStacked = copyDeviceToHost<float>(stackedLogits);

    // CPU reference: fp16-rounded weights, greedy chain.
    std::vector<int32_t> cpuPrev = {3, 77};
    for (int32_t step = 0; step < proposalLen; ++step)
    {
        for (int32_t b = 0; b < batchSize; ++b)
        {
            float bestVal = -1e30F;
            for (int32_t v = 0; v < vocabSize; ++v)
            {
                float bias = 0.0F;
                for (int32_t r = 0; r < markovRank; ++r)
                {
                    bias += __half2float(__float2half(w1Host[cpuPrev[b] * markovRank + r]))
                        * __half2float(__float2half(w2Host[v * markovRank + r]));
                }
                float const corrected = backboneHost[(b * proposalLen + step) * vocabSize + v] + bias;
                EXPECT_NEAR(gpuStacked[(b * depthSize + step + 1) * vocabSize + v], corrected, 2e-3F)
                    << "stacked mismatch step=" << step << " b=" << b << " v=" << v;
                if (corrected > bestVal)
                {
                    bestVal = corrected;
                }
            }
            // Tolerate FP reassociation on near-ties: GPU winner must be within epsilon of the CPU max.
            int32_t const gpuTok = gpuTokens[b * proposalLen + step];
            float cpuBias = 0.0F;
            for (int32_t r = 0; r < markovRank; ++r)
            {
                cpuBias += __half2float(__float2half(w1Host[cpuPrev[b] * markovRank + r]))
                    * __half2float(__float2half(w2Host[gpuTok * markovRank + r]));
            }
            float const gpuVal = backboneHost[(b * proposalLen + step) * vocabSize + gpuTok] + cpuBias;
            EXPECT_NEAR(gpuVal, bestVal, 2e-3F) << "argmax mismatch step=" << step << " b=" << b;
            cpuPrev[b] = gpuTok;
        }
    }
}

TEST(DSparkKernels, FusedGreedyStepFp8MatchesFp16Closely)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 2;
    constexpr int32_t proposalLen = 2;
    constexpr int32_t vocabSize = 103; // exercises the fp8 kernel's vocab tail
    constexpr int32_t markovRank = 512;
    constexpr int32_t depthSize = proposalLen + 1;

    std::vector<float> backboneHost(batchSize * proposalLen * vocabSize);
    std::vector<float> w1Host(vocabSize * markovRank);
    std::vector<float> w2Host(vocabSize * markovRank);
    for (size_t i = 0; i < backboneHost.size(); ++i)
    {
        backboneHost[i] = std::cos(static_cast<float>(i) * 0.41F) * 3.0F;
    }
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (int32_t step = 0; step < proposalLen; ++step)
        {
            int32_t const tokenIdx = 7 + batchIdx * 13 + step * 17;
            backboneHost[(batchIdx * proposalLen + step) * vocabSize + tokenIdx] += 1000.0F;
        }
    }
    for (size_t i = 0; i < w1Host.size(); ++i)
    {
        w1Host[i] = std::sin(static_cast<float>(i) * 0.67F);
        w2Host[i] = std::cos(static_cast<float>(i) * 1.19F) * 0.4F;
    }

    auto backboneLogits = rt::Tensor({batchSize, proposalLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto markovW1 = rt::Tensor({vocabSize, markovRank}, rt::DeviceType::kGPU, DataType::kHALF);
    auto markovW2 = rt::Tensor({vocabSize, markovRank}, rt::DeviceType::kGPU, DataType::kHALF);
    auto w2Fp8 = rt::Tensor({vocabSize, markovRank}, rt::DeviceType::kGPU, DataType::kUINT8);
    auto w2RowScales = rt::Tensor({vocabSize}, rt::DeviceType::kGPU, DataType::kHALF);
    auto firstPrevTokens = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto greedySlotsFp16 = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT64);
    auto greedySlotsFp8 = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT64);
    auto draftTokenIdsFp16 = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftTokenIdsFp8 = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto stackedFp16 = rt::Tensor({batchSize, depthSize, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto stackedFp8 = rt::Tensor({batchSize, depthSize, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);

    copyHostToDevice<float>(backboneLogits, backboneHost);
    copyHostToDevice<half>(markovW1, toHalf(w1Host));
    copyHostToDevice<half>(markovW2, toHalf(w2Host));
    copyHostToDevice<int32_t>(firstPrevTokens, {9, 55});
    dsparkQuantizeMarkovW2Fp8(markovW2, w2Fp8, w2RowScales, vocabSize, markovRank, stream);
    CUDA_CHECK(cudaMemsetAsync(stackedFp16.rawPointer(), 0, stackedFp16.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(stackedFp8.rawPointer(), 0, stackedFp8.getMemoryCapacity(), stream));

    CUDA_CHECK(cudaMemsetAsync(greedySlotsFp16.rawPointer(), 0, batchSize * proposalLen * sizeof(uint64_t), stream));
    for (int32_t step = 0; step < proposalLen; ++step)
    {
        dsparkMarkovGreedyFusedStep(backboneLogits, markovW1, markovW2, firstPrevTokens, greedySlotsFp16, &stackedFp16,
            step + 1, batchSize, step, proposalLen, vocabSize, markovRank, stream);
    }
    dsparkFinalizeGreedyDraftTokens(greedySlotsFp16, draftTokenIdsFp16, batchSize * proposalLen, stream);
    CUDA_CHECK(cudaMemsetAsync(greedySlotsFp8.rawPointer(), 0, batchSize * proposalLen * sizeof(uint64_t), stream));
    for (int32_t step = 0; step < proposalLen; ++step)
    {
        dsparkMarkovGreedyFusedStepFp8(backboneLogits, markovW1, w2Fp8, w2RowScales, firstPrevTokens, greedySlotsFp8,
            &stackedFp8, step + 1, batchSize, step, proposalLen, vocabSize, markovRank, stream);
    }
    dsparkFinalizeGreedyDraftTokens(greedySlotsFp8, draftTokenIdsFp8, batchSize * proposalLen, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const rowsFp16 = copyDeviceToHost<float>(stackedFp16);
    auto const rowsFp8 = copyDeviceToHost<float>(stackedFp8);
    EXPECT_EQ(copyDeviceToHost<int32_t>(draftTokenIdsFp8), copyDeviceToHost<int32_t>(draftTokenIdsFp16));
    // E4M3 relative error is ~2^-4 per element; over a rank-R dot with |w1| <= 1 a
    // conservative bound is R * maxAbsW2 * 2^-3.
    float maxAbsW2 = 0.0F;
    for (auto v : w2Host)
    {
        maxAbsW2 = std::max(maxAbsW2, std::fabs(v));
    }
    float const bound = static_cast<float>(markovRank) * maxAbsW2 * 0.125F;
    for (size_t i = 0; i < rowsFp16.size(); ++i)
    {
        EXPECT_NEAR(rowsFp8[i], rowsFp16[i], bound) << "i=" << i;
    }
}

//! Guided decoding drives every grammar-illegal token's target probability to exactly zero, which
//! makes the vanishing-draft-probability guard reachable with p == 0 as well as q == 0. The guard
//! reads that as p/q -> inf and would accept, emitting a token the target says cannot occur. The
//! uniform draw is 0.0 here because the clamp admits it and it is the one value that would also
//! slip past an `acceptProb == 0` comparison.
TEST(DSparkKernels, ProbabilisticAcceptRejectsTokenTheTargetGivesNoMass)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 1;
    constexpr int32_t proposalLen = 1;
    constexpr int32_t verifyLen = proposalLen + 1;
    constexpr int32_t vocabSize = 4;

    auto targetProbabilities = rt::Tensor({batchSize, verifyLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftProbabilities = rt::Tensor({batchSize, proposalLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftTokenIds = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto uniforms = rt::Tensor({batchSize, 2 * proposalLen + 1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto proposalLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptedTokenIds = rt::Tensor({batchSize, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptLength = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

    // Token 2 is masked away in both verify rows; the residual leaves only token 0.
    copyHostToDevice<float>(targetProbabilities, {0.70F, 0.30F, 0.00F, 0.00F, 0.60F, 0.40F, 0.00F, 0.00F});
    copyHostToDevice<float>(draftProbabilities, {0.20F, 0.80F, 0.00F, 0.00F});
    copyHostToDevice<int32_t>(draftTokenIds, {2});
    copyHostToDevice<float>(uniforms, {0.0F, 0.5F, 0.5F});
    copyHostToDevice<int32_t>(proposalLengths, {proposalLen});

    dsparkProbabilisticAccept(targetProbabilities, draftProbabilities, draftTokenIds, proposalLengths, uniforms,
        acceptedTokenIds, acceptLength, batchSize, proposalLen, proposalLen, vocabSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(acceptLength), (std::vector<int32_t>{1}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(acceptedTokenIds)[0], 0) << "the masked token must not be emitted";
}
