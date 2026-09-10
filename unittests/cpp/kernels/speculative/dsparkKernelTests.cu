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

#include <cmath>
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

std::vector<int32_t> greedyMarkovReference(std::vector<float> const& backboneLogits, std::vector<half> const& markovW1,
    std::vector<half> const& markovW2, std::vector<int32_t> const& firstPrevTokens, int32_t batchSize,
    int32_t proposalLen, int32_t vocabSize, int32_t markovRank)
{
    std::vector<int32_t> out(batchSize * proposalLen, 0);
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (int32_t step = 0; step < proposalLen; ++step)
        {
            int32_t const prevToken = (step == 0) ? firstPrevTokens[batchIdx] : out[batchIdx * proposalLen + step - 1];
            float best = -std::numeric_limits<float>::infinity();
            int32_t bestIdx = 0;
            for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
            {
                float score = backboneLogits[(batchIdx * proposalLen + step) * vocabSize + vocabIdx];
                for (int32_t rankIdx = 0; rankIdx < markovRank; ++rankIdx)
                {
                    score += __half2float(markovW1[prevToken * markovRank + rankIdx])
                        * __half2float(markovW2[vocabIdx * markovRank + rankIdx]);
                }
                if (score > best || (score == best && vocabIdx < bestIdx))
                {
                    best = score;
                    bestIdx = vocabIdx;
                }
            }
            out[batchIdx * proposalLen + step] = bestIdx;
        }
    }
    return out;
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

TEST(DSparkKernels, SparseTopKAcceptSamplesResidualOnReject)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 1;
    constexpr int32_t proposalLen = 2;
    constexpr int32_t verifyLen = proposalLen + 1;
    constexpr int32_t topK = 3;

    auto targetTopKProbabilities = rt::Tensor({batchSize, verifyLen, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto targetTopKIndices = rt::Tensor({batchSize, verifyLen, topK}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftTopKProbabilities = rt::Tensor({batchSize, proposalLen, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftTopKIndices = rt::Tensor({batchSize, proposalLen, topK}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftTokenIds = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto uniforms = rt::Tensor({batchSize, 2 * proposalLen + 1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto proposalLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptedTokenIds = rt::Tensor({batchSize, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptLength = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<float>(targetTopKProbabilities,
        {
            0.80F,
            0.10F,
            0.10F,
            0.50F,
            0.20F,
            0.30F,
            0.25F,
            0.25F,
            0.50F,
        });
    copyHostToDevice<int32_t>(targetTopKIndices,
        {
            1,
            0,
            2,
            0,
            1,
            3,
            0,
            1,
            2,
        });
    copyHostToDevice<float>(draftTopKProbabilities,
        {
            0.40F,
            0.30F,
            0.30F,
            0.50F,
            0.30F,
            0.20F,
        });
    copyHostToDevice<int32_t>(draftTopKIndices,
        {
            1,
            2,
            3,
            2,
            1,
            3,
        });
    copyHostToDevice<int32_t>(draftTokenIds, {1, 2});
    copyHostToDevice<float>(uniforms, {0.5F, 0.9F, 0.0F, 0.9F, 0.0F});
    copyHostToDevice<int32_t>(proposalLengths, {proposalLen});

    dsparkSparseTopKAccept(targetTopKProbabilities, targetTopKIndices, draftTopKProbabilities, draftTopKIndices,
        draftTokenIds, proposalLengths, uniforms, acceptedTokenIds, acceptLength, batchSize, proposalLen, proposalLen,
        topK, topK, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(acceptLength), (std::vector<int32_t>{2}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(acceptedTokenIds), (std::vector<int32_t>{1, 3, 0}));
}

TEST(DSparkKernels, SparseTopKAcceptSamplesBonusWhenAllAccepted)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 1;
    constexpr int32_t proposalLen = 2;
    constexpr int32_t verifyLen = proposalLen + 1;
    constexpr int32_t topK = 2;

    auto targetTopKProbabilities = rt::Tensor({batchSize, verifyLen, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto targetTopKIndices = rt::Tensor({batchSize, verifyLen, topK}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftTopKProbabilities = rt::Tensor({batchSize, proposalLen, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftTopKIndices = rt::Tensor({batchSize, proposalLen, topK}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftTokenIds = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto uniforms = rt::Tensor({batchSize, 2 * proposalLen + 1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto proposalLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptedTokenIds = rt::Tensor({batchSize, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptLength = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<float>(targetTopKProbabilities, {0.80F, 0.20F, 0.60F, 0.40F, 0.40F, 0.60F});
    copyHostToDevice<int32_t>(targetTopKIndices, {1, 0, 2, 3, 3, 2});
    copyHostToDevice<float>(draftTopKProbabilities, {0.40F, 0.60F, 0.20F, 0.80F});
    copyHostToDevice<int32_t>(draftTopKIndices, {1, 0, 2, 3});
    copyHostToDevice<int32_t>(draftTokenIds, {1, 2});
    copyHostToDevice<float>(uniforms, {0.9F, 0.9F, 0.0F, 0.0F, 0.75F});
    copyHostToDevice<int32_t>(proposalLengths, {proposalLen});

    dsparkSparseTopKAccept(targetTopKProbabilities, targetTopKIndices, draftTopKProbabilities, draftTopKIndices,
        draftTokenIds, proposalLengths, uniforms, acceptedTokenIds, acceptLength, batchSize, proposalLen, proposalLen,
        topK, topK, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(acceptLength), (std::vector<int32_t>{3}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(acceptedTokenIds), (std::vector<int32_t>{1, 2, 2}));
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
