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
#include "kernels/speculative/dominoRolloutKernels.h"
#include "testUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{

std::vector<half> randomHalf(std::mt19937& rng, size_t count, float scale)
{
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<half> out(count);
    for (auto& value : out)
    {
        value = __float2half(dist(rng));
    }
    return out;
}

float sigmoidf(float x)
{
    return 1.0F / (1.0F + std::exp(-x));
}

//! embedding . weightIh^T per vocabulary row, FP32 accumulate, half-rounded like the exported table.
std::vector<half> gruInputTableHost(std::vector<half> const& embedding, std::vector<half> const& weightIh,
    int32_t vocabSize, int32_t hiddenSize, int32_t gruHiddenDim)
{
    std::vector<half> table(static_cast<size_t>(vocabSize) * 3 * gruHiddenDim);
    for (int32_t v = 0; v < vocabSize; ++v)
    {
        for (int32_t row = 0; row < 3 * gruHiddenDim; ++row)
        {
            float acc = 0.0F;
            for (int32_t i = 0; i < hiddenSize; ++i)
            {
                acc += __half2float(weightIh[row * hiddenSize + i]) * __half2float(embedding[v * hiddenSize + i]);
            }
            table[v * 3 * gruHiddenDim + row] = __float2half(acc);
        }
    }
    return table;
}

//! Bias-free PyTorch GRU cell (r/z/n row layout) with the input projection read from the table row.
std::vector<float> gruCellReference(std::vector<half> const& table, std::vector<int32_t> const& tokens,
    std::vector<half> const& h, std::vector<half> const& weightHh, int32_t batchSize, int32_t gruHiddenDim)
{
    std::vector<float> out(batchSize * gruHiddenDim);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        half const* inputRow = table.data() + static_cast<size_t>(tokens[b]) * 3 * gruHiddenDim;
        for (int32_t g = 0; g < gruHiddenDim; ++g)
        {
            float gateHidden[3] = {0.0F, 0.0F, 0.0F};
            for (int32_t gate = 0; gate < 3; ++gate)
            {
                for (int32_t i = 0; i < gruHiddenDim; ++i)
                {
                    gateHidden[gate] += __half2float(weightHh[(gate * gruHiddenDim + g) * gruHiddenDim + i])
                        * __half2float(h[b * gruHiddenDim + i]);
                }
            }
            float const r = sigmoidf(__half2float(inputRow[g]) + gateHidden[0]);
            float const z = sigmoidf(__half2float(inputRow[gruHiddenDim + g]) + gateHidden[1]);
            float const n = std::tanh(__half2float(inputRow[2 * gruHiddenDim + g]) + r * gateHidden[2]);
            out[b * gruHiddenDim + g] = (1.0F - z) * n + z * __half2float(h[b * gruHiddenDim + g]);
        }
    }
    return out;
}

} // namespace

TEST(DominoRolloutKernels, GruUpdateMatchesReference)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 3;
    constexpr int32_t vocabSize = 16;
    constexpr int32_t hiddenSize = 32;
    constexpr int32_t gruHiddenDim = 8;

    std::mt19937 rng(42);
    auto const embeddingHost = randomHalf(rng, vocabSize * hiddenSize, 1.0F);
    auto const weightIhHost = randomHalf(rng, 3 * gruHiddenDim * hiddenSize, 0.4F);
    auto const weightHhHost = randomHalf(rng, 3 * gruHiddenDim * gruHiddenDim, 0.4F);
    auto const hInHost = randomHalf(rng, batchSize * gruHiddenDim, 0.8F);
    std::vector<int32_t> const tokensHost{3, 0, 15};

    auto const tableHost = gruInputTableHost(embeddingHost, weightIhHost, vocabSize, hiddenSize, gruHiddenDim);
    auto table = rt::Tensor({vocabSize, 3 * gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightHh = rt::Tensor({3 * gruHiddenDim, gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto hIn = rt::Tensor({batchSize, gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto hOut = rt::Tensor({batchSize, gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto tokens = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<half>(table, tableHost);
    copyHostToDevice<half>(weightHh, weightHhHost);
    copyHostToDevice<half>(hIn, hInHost);
    copyHostToDevice<int32_t>(tokens, tokensHost);

    dominoGruUpdate(
        table, tokens, /*tokenStride=*/1, /*tokenOffset=*/0, weightHh, hIn, hOut, batchSize, gruHiddenDim, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const expected = gruCellReference(tableHost, tokensHost, hInHost, weightHhHost, batchSize, gruHiddenDim);
    auto const actual = copyDeviceToHost<half>(hOut);
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(__half2float(actual[i]), expected[i], 3e-3F) << "element=" << i;
    }
}

TEST(DominoRolloutKernels, GruUpdateReadsStridedTokenSlots)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 2;
    constexpr int32_t vocabSize = 8;
    constexpr int32_t hiddenSize = 16;
    constexpr int32_t gruHiddenDim = 4;
    constexpr int32_t tokenStride = 5;
    constexpr int32_t tokenOffset = 2;

    std::mt19937 rng(7);
    auto const embeddingHost = randomHalf(rng, vocabSize * hiddenSize, 1.0F);
    auto const weightIhHost = randomHalf(rng, 3 * gruHiddenDim * hiddenSize, 0.4F);
    auto const weightHhHost = randomHalf(rng, 3 * gruHiddenDim * gruHiddenDim, 0.4F);
    std::vector<half> const hInHost(batchSize * gruHiddenDim, __float2half(0.0F));
    // Only slot tokenOffset of each strided row is the input token.
    std::vector<int32_t> tokensHost(batchSize * tokenStride, -1);
    tokensHost[0 * tokenStride + tokenOffset] = 6;
    tokensHost[1 * tokenStride + tokenOffset] = 1;

    auto const tableHost = gruInputTableHost(embeddingHost, weightIhHost, vocabSize, hiddenSize, gruHiddenDim);
    auto table = rt::Tensor({vocabSize, 3 * gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightHh = rt::Tensor({3 * gruHiddenDim, gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto hIn = rt::Tensor({batchSize, gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto hOut = rt::Tensor({batchSize, gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto tokens = rt::Tensor({batchSize, tokenStride}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<half>(table, tableHost);
    copyHostToDevice<half>(weightHh, weightHhHost);
    copyHostToDevice<half>(hIn, hInHost);
    copyHostToDevice<int32_t>(tokens, tokensHost);

    dominoGruUpdate(table, tokens, tokenStride, tokenOffset, weightHh, hIn, hOut, batchSize, gruHiddenDim, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> selectedTokens(batchSize);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        selectedTokens[b] = tokensHost[b * tokenStride + tokenOffset];
    }
    auto const expected = gruCellReference(tableHost, selectedTokens, hInHost, weightHhHost, batchSize, gruHiddenDim);
    auto const actual = copyDeviceToHost<half>(hOut);
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(__half2float(actual[i]), expected[i], 3e-3F) << "element=" << i;
    }
}

TEST(DominoRolloutKernels, CorrectStepPicksBiasedArgmaxPerBatch)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 2;
    constexpr int32_t feedbackLen = 3;
    constexpr int32_t stepIdx = 1;
    constexpr int32_t poolSize = 6;
    constexpr int32_t projDim = 4;
    constexpr int32_t gruHiddenDim = 4;
    constexpr int32_t draftTokenStride = 8;
    constexpr int32_t tokenOffset = 3;

    std::mt19937 rng(11);
    auto const zProjHost = randomHalf(rng, batchSize * feedbackLen * projDim, 0.5F);
    auto const codebookHost = randomHalf(rng, batchSize * poolSize * projDim, 0.5F);
    auto const projStateHost = randomHalf(rng, projDim * gruHiddenDim, 0.5F);
    auto const gruHiddenHost = randomHalf(rng, batchSize * gruHiddenDim, 0.8F);
    std::uniform_real_distribution<float> logitDist(-2.0F, 2.0F);
    std::vector<float> poolLogitsHost(batchSize * feedbackLen * poolSize);
    for (auto& value : poolLogitsHost)
    {
        value = logitDist(rng);
    }
    std::vector<int32_t> poolIdsHost(batchSize * poolSize);
    for (size_t i = 0; i < poolIdsHost.size(); ++i)
    {
        poolIdsHost[i] = 1000 + static_cast<int32_t>(i);
    }

    auto zProj = rt::Tensor({batchSize, feedbackLen, projDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto poolLogits = rt::Tensor({batchSize, feedbackLen, poolSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto codebook = rt::Tensor({batchSize, poolSize, projDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto poolIds = rt::Tensor({batchSize, poolSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto projState = rt::Tensor({projDim, gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto gruHidden = rt::Tensor({batchSize, gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto draftTokens = rt::Tensor({batchSize, draftTokenStride}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<half>(zProj, zProjHost);
    copyHostToDevice<float>(poolLogits, poolLogitsHost);
    copyHostToDevice<half>(codebook, codebookHost);
    copyHostToDevice<int32_t>(poolIds, poolIdsHost);
    copyHostToDevice<half>(projState, projStateHost);
    copyHostToDevice<half>(gruHidden, gruHiddenHost);
    copyHostToDevice<int32_t>(draftTokens, std::vector<int32_t>(batchSize * draftTokenStride, -1));

    auto correction = rt::Tensor({batchSize, projDim}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto winnerSlots = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT64);
    dominoCorrectStep(zProj, poolLogits, codebook, poolIds, projState, gruHidden, draftTokens, /*scoresOut=*/nullptr,
        DominoCorrectScratch{correction, winnerSlots},
        DominoStepLayout{stepIdx, feedbackLen, stepIdx, feedbackLen, stepIdx, feedbackLen}, draftTokenStride,
        tokenOffset, batchSize, poolSize, projDim, gruHiddenDim, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const actual = copyDeviceToHost<int32_t>(draftTokens);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        std::vector<float> correction(projDim);
        for (int32_t r = 0; r < projDim; ++r)
        {
            float acc = __half2float(zProjHost[(b * feedbackLen + stepIdx) * projDim + r]);
            for (int32_t g = 0; g < gruHiddenDim; ++g)
            {
                acc += __half2float(projStateHost[r * gruHiddenDim + g])
                    * __half2float(gruHiddenHost[b * gruHiddenDim + g]);
            }
            correction[r] = acc * sigmoidf(acc);
        }
        float bestScore = -std::numeric_limits<float>::infinity();
        int32_t bestPool = 0;
        for (int32_t p = 0; p < poolSize; ++p)
        {
            float score = poolLogitsHost[(b * feedbackLen + stepIdx) * poolSize + p];
            for (int32_t r = 0; r < projDim; ++r)
            {
                score += __half2float(codebookHost[(b * poolSize + p) * projDim + r]) * correction[r];
            }
            if (score > bestScore)
            {
                bestScore = score;
                bestPool = p;
            }
        }
        EXPECT_EQ(actual[b * draftTokenStride + tokenOffset], poolIdsHost[b * poolSize + bestPool]) << "batch=" << b;
        // Slots outside tokenOffset stay untouched.
        EXPECT_EQ(actual[b * draftTokenStride], -1);
    }
}

TEST(DominoRolloutKernels, CorrectStepLargePoolWinnerBeyondFirstWarp)
{
    // Production-sized pool: every thread owns several candidates and the block-level
    // merge across warps must surface a winner that lives in the upper half of the pool.
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 2;
    constexpr int32_t feedbackLen = 2;
    constexpr int32_t stepIdx = 1;
    constexpr int32_t poolSize = 2048;
    constexpr int32_t projDim = 256;
    constexpr int32_t gruHiddenDim = 1024;
    constexpr int32_t draftTokenStride = 16;
    constexpr int32_t tokenOffset = 5;
    constexpr int32_t forcedWinner[batchSize] = {1907, 1300};

    std::mt19937 rng(23);
    auto const zProjHost = randomHalf(rng, batchSize * feedbackLen * projDim, 0.5F);
    auto const codebookHost = randomHalf(rng, batchSize * poolSize * projDim, 0.05F);
    auto const projStateHost = randomHalf(rng, projDim * gruHiddenDim, 0.05F);
    auto const gruHiddenHost = randomHalf(rng, batchSize * gruHiddenDim, 0.8F);
    std::uniform_real_distribution<float> logitDist(-2.0F, 2.0F);
    std::vector<float> poolLogitsHost(batchSize * feedbackLen * poolSize);
    for (auto& value : poolLogitsHost)
    {
        value = logitDist(rng);
    }
    for (int32_t b = 0; b < batchSize; ++b)
    {
        poolLogitsHost[(b * feedbackLen + stepIdx) * poolSize + forcedWinner[b]] = 50.0F;
    }
    std::vector<int32_t> poolIdsHost(batchSize * poolSize);
    for (size_t i = 0; i < poolIdsHost.size(); ++i)
    {
        poolIdsHost[i] = 7 * static_cast<int32_t>(i) + 3;
    }

    auto zProj = rt::Tensor({batchSize, feedbackLen, projDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto poolLogits = rt::Tensor({batchSize, feedbackLen, poolSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto codebook = rt::Tensor({batchSize, poolSize, projDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto poolIds = rt::Tensor({batchSize, poolSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto projState = rt::Tensor({projDim, gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto gruHidden = rt::Tensor({batchSize, gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto draftTokens = rt::Tensor({batchSize, draftTokenStride}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<half>(zProj, zProjHost);
    copyHostToDevice<float>(poolLogits, poolLogitsHost);
    copyHostToDevice<half>(codebook, codebookHost);
    copyHostToDevice<int32_t>(poolIds, poolIdsHost);
    copyHostToDevice<half>(projState, projStateHost);
    copyHostToDevice<half>(gruHidden, gruHiddenHost);
    copyHostToDevice<int32_t>(draftTokens, std::vector<int32_t>(batchSize * draftTokenStride, -1));

    auto correction = rt::Tensor({batchSize, projDim}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto winnerSlots = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT64);
    dominoCorrectStep(zProj, poolLogits, codebook, poolIds, projState, gruHidden, draftTokens, /*scoresOut=*/nullptr,
        DominoCorrectScratch{correction, winnerSlots},
        DominoStepLayout{stepIdx, feedbackLen, stepIdx, feedbackLen, stepIdx, feedbackLen}, draftTokenStride,
        tokenOffset, batchSize, poolSize, projDim, gruHiddenDim, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const actual = copyDeviceToHost<int32_t>(draftTokens);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        std::vector<float> correction(projDim);
        for (int32_t r = 0; r < projDim; ++r)
        {
            float acc = __half2float(zProjHost[(b * feedbackLen + stepIdx) * projDim + r]);
            for (int32_t g = 0; g < gruHiddenDim; ++g)
            {
                acc += __half2float(projStateHost[r * gruHiddenDim + g])
                    * __half2float(gruHiddenHost[b * gruHiddenDim + g]);
            }
            correction[r] = acc * sigmoidf(acc);
        }
        float bestScore = -std::numeric_limits<float>::infinity();
        int32_t bestPool = 0;
        for (int32_t p = 0; p < poolSize; ++p)
        {
            float score = poolLogitsHost[(b * feedbackLen + stepIdx) * poolSize + p];
            for (int32_t r = 0; r < projDim; ++r)
            {
                score += __half2float(codebookHost[(b * poolSize + p) * projDim + r]) * correction[r];
            }
            if (score > bestScore)
            {
                bestScore = score;
                bestPool = p;
            }
        }
        EXPECT_EQ(bestPool, forcedWinner[b]) << "reference winner drifted, batch=" << b;
        EXPECT_EQ(actual[b * draftTokenStride + tokenOffset], poolIdsHost[b * poolSize + bestPool]) << "batch=" << b;
    }
}

TEST(DominoRolloutKernels, CorrectStepBreaksTiesTowardLowestPoolIndex)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 1;
    constexpr int32_t feedbackLen = 1;
    constexpr int32_t poolSize = 5;
    constexpr int32_t projDim = 2;
    constexpr int32_t gruHiddenDim = 2;

    // Zero correction path: SiLU(0) = 0 bias, so scores equal the pool logits.
    std::vector<half> const zeros16(16, __float2half(0.0F));
    std::vector<float> const poolLogitsHost{1.0F, 4.0F, 2.0F, 4.0F, 3.0F};
    std::vector<int32_t> const poolIdsHost{70, 71, 72, 73, 74};

    auto zProj = rt::Tensor({batchSize, feedbackLen, projDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto poolLogits = rt::Tensor({batchSize, feedbackLen, poolSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto codebook = rt::Tensor({batchSize, poolSize, projDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto poolIds = rt::Tensor({batchSize, poolSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto projState = rt::Tensor({projDim, gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto gruHidden = rt::Tensor({batchSize, gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto draftTokens = rt::Tensor({batchSize, 4}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<half>(zProj, std::vector<half>(zeros16.begin(), zeros16.begin() + projDim));
    copyHostToDevice<float>(poolLogits, poolLogitsHost);
    copyHostToDevice<half>(codebook, std::vector<half>(zeros16.begin(), zeros16.begin() + poolSize * projDim));
    copyHostToDevice<int32_t>(poolIds, poolIdsHost);
    copyHostToDevice<half>(projState, std::vector<half>(zeros16.begin(), zeros16.begin() + projDim * gruHiddenDim));
    copyHostToDevice<half>(gruHidden, std::vector<half>(zeros16.begin(), zeros16.begin() + gruHiddenDim));
    copyHostToDevice<int32_t>(draftTokens, std::vector<int32_t>(batchSize * 4, -1));

    auto correction = rt::Tensor({batchSize, projDim}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto winnerSlots = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT64);
    dominoCorrectStep(zProj, poolLogits, codebook, poolIds, projState, gruHidden, draftTokens, /*scoresOut=*/nullptr,
        DominoCorrectScratch{correction, winnerSlots}, DominoStepLayout{0, feedbackLen, 0, feedbackLen, 0, feedbackLen},
        /*draftTokenStride=*/4, /*tokenOffset=*/1, batchSize, poolSize, projDim, gruHiddenDim, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(draftTokens)[1], 71);
}

// Given a correction step with a stacked-scores output
// When the step runs with a layout that maps zProj row 0 onto pool-logits row 1 and score row 1
// Then the full corrected score row lands in scoresOut[b, 1, :] and the other rows stay untouched
TEST(DominoRolloutKernels, CorrectStepWritesTheCorrectedScoreRow)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 2;
    constexpr int32_t zRows = 2;
    constexpr int32_t poolRows = 3;
    constexpr int32_t scoreRows = 4;
    constexpr int32_t poolSize = 6;
    constexpr int32_t projDim = 4;
    constexpr int32_t gruHiddenDim = 4;
    constexpr int32_t draftTokenStride = 8;
    DominoStepLayout const layout{/*zRow=*/0, zRows, /*logitsRow=*/1, poolRows, /*scoreRow=*/1, scoreRows};

    std::mt19937 rng(5);
    auto const zProjHost = randomHalf(rng, batchSize * zRows * projDim, 0.5F);
    auto const codebookHost = randomHalf(rng, batchSize * poolSize * projDim, 0.5F);
    auto const projStateHost = randomHalf(rng, projDim * gruHiddenDim, 0.5F);
    auto const gruHiddenHost = randomHalf(rng, batchSize * gruHiddenDim, 0.8F);
    std::uniform_real_distribution<float> logitDist(-2.0F, 2.0F);
    std::vector<float> poolLogitsHost(batchSize * poolRows * poolSize);
    for (auto& value : poolLogitsHost)
    {
        value = logitDist(rng);
    }
    std::vector<int32_t> poolIdsHost(batchSize * poolSize);
    for (size_t i = 0; i < poolIdsHost.size(); ++i)
    {
        poolIdsHost[i] = 500 + static_cast<int32_t>(i);
    }

    auto zProj = rt::Tensor({batchSize, zRows, projDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto poolLogits = rt::Tensor({batchSize, poolRows, poolSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto codebook = rt::Tensor({batchSize, poolSize, projDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto poolIds = rt::Tensor({batchSize, poolSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto projState = rt::Tensor({projDim, gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto gruHidden = rt::Tensor({batchSize, gruHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto draftTokens = rt::Tensor({batchSize, draftTokenStride}, rt::DeviceType::kGPU, DataType::kINT32);
    auto scores = rt::Tensor({batchSize, scoreRows, poolSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    copyHostToDevice<half>(zProj, zProjHost);
    copyHostToDevice<float>(poolLogits, poolLogitsHost);
    copyHostToDevice<half>(codebook, codebookHost);
    copyHostToDevice<int32_t>(poolIds, poolIdsHost);
    copyHostToDevice<half>(projState, projStateHost);
    copyHostToDevice<half>(gruHidden, gruHiddenHost);
    copyHostToDevice<int32_t>(draftTokens, std::vector<int32_t>(batchSize * draftTokenStride, -1));
    copyHostToDevice<float>(scores, std::vector<float>(batchSize * scoreRows * poolSize, -7.0F));

    auto correction = rt::Tensor({batchSize, projDim}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto winnerSlots = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT64);
    dominoCorrectStep(zProj, poolLogits, codebook, poolIds, projState, gruHidden, draftTokens, &scores,
        DominoCorrectScratch{correction, winnerSlots}, layout, draftTokenStride, /*tokenOffset=*/2, batchSize, poolSize,
        projDim, gruHiddenDim, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const actualScores = copyDeviceToHost<float>(scores);
    auto const actualTokens = copyDeviceToHost<int32_t>(draftTokens);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        std::vector<float> correctionRef(projDim);
        for (int32_t r = 0; r < projDim; ++r)
        {
            float acc = __half2float(zProjHost[(b * zRows + layout.zRow) * projDim + r]);
            for (int32_t g = 0; g < gruHiddenDim; ++g)
            {
                acc += __half2float(projStateHost[r * gruHiddenDim + g])
                    * __half2float(gruHiddenHost[b * gruHiddenDim + g]);
            }
            correctionRef[r] = acc * sigmoidf(acc);
        }
        int32_t bestPool = 0;
        float bestScore = -std::numeric_limits<float>::infinity();
        for (int32_t p = 0; p < poolSize; ++p)
        {
            float score = poolLogitsHost[(b * poolRows + layout.logitsRow) * poolSize + p];
            for (int32_t r = 0; r < projDim; ++r)
            {
                score += __half2float(codebookHost[(b * poolSize + p) * projDim + r]) * correctionRef[r];
            }
            EXPECT_NEAR(actualScores[(b * scoreRows + layout.scoreRow) * poolSize + p], score, 2e-3F)
                << "batch=" << b << " pool=" << p;
            if (score > bestScore)
            {
                bestScore = score;
                bestPool = p;
            }
        }
        EXPECT_EQ(actualTokens[b * draftTokenStride + 2], poolIdsHost[b * poolSize + bestPool]);
        for (int32_t row = 0; row < scoreRows; ++row)
        {
            if (row == layout.scoreRow)
            {
                continue;
            }
            EXPECT_EQ(actualScores[(b * scoreRows + row) * poolSize], -7.0F) << "row " << row << " was overwritten";
        }
    }
}

// Given pool logits whose row 0 is the uncorrected first proposal, with a tie in batch 1
// When the first proposal is selected
// Then the token is the row-0 argmax (lowest pool index on ties) and row 0 is copied into the stacked scores
TEST(DominoRolloutKernels, SelectFirstProposalTakesRowZeroArgmax)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 2;
    constexpr int32_t poolRows = 3;
    constexpr int32_t scoreRows = 4;
    constexpr int32_t poolSize = 5;
    constexpr int32_t draftTokenStride = 6;
    std::vector<float> poolLogitsHost(batchSize * poolRows * poolSize, 0.0F);
    // batch 0 row 0: argmax at pool 3; batch 1 row 0: tie between pool 1 and pool 4.
    float const row0b0[poolSize] = {0.1F, -1.0F, 0.5F, 2.5F, 2.0F};
    float const row0b1[poolSize] = {-3.0F, 4.0F, 1.0F, 0.0F, 4.0F};
    for (int32_t p = 0; p < poolSize; ++p)
    {
        poolLogitsHost[p] = row0b0[p];
        poolLogitsHost[poolRows * poolSize + p] = row0b1[p];
        poolLogitsHost[poolSize + p] = 9.0F; // row 1 must not influence the first proposal
    }
    std::vector<int32_t> poolIdsHost(batchSize * poolSize);
    for (size_t i = 0; i < poolIdsHost.size(); ++i)
    {
        poolIdsHost[i] = 100 + static_cast<int32_t>(i);
    }

    auto poolLogits = rt::Tensor({batchSize, poolRows, poolSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto poolIds = rt::Tensor({batchSize, poolSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftTokens = rt::Tensor({batchSize, draftTokenStride}, rt::DeviceType::kGPU, DataType::kINT32);
    auto scores = rt::Tensor({batchSize, scoreRows, poolSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto winnerSlots = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT64);
    copyHostToDevice<float>(poolLogits, poolLogitsHost);
    copyHostToDevice<int32_t>(poolIds, poolIdsHost);
    copyHostToDevice<int32_t>(draftTokens, std::vector<int32_t>(batchSize * draftTokenStride, -1));
    copyHostToDevice<float>(scores, std::vector<float>(batchSize * scoreRows * poolSize, -7.0F));

    dominoSelectFirstProposal(poolLogits, poolRows, poolIds, draftTokens, &scores, scoreRows, winnerSlots,
        draftTokenStride, /*tokenOffset=*/1, batchSize, poolSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const tokens = copyDeviceToHost<int32_t>(draftTokens);
    EXPECT_EQ(tokens[1], 103);
    EXPECT_EQ(tokens[draftTokenStride + 1], 100 + poolSize + 1);
    EXPECT_EQ(tokens[0], -1);
    auto const actualScores = copyDeviceToHost<float>(scores);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t p = 0; p < poolSize; ++p)
        {
            EXPECT_EQ(actualScores[b * scoreRows * poolSize + p], poolLogitsHost[b * poolRows * poolSize + p]);
        }
        EXPECT_EQ(actualScores[(b * scoreRows + 1) * poolSize], -7.0F);
    }
}

// Given DDTree node tokens in pool-slot space with a root node and padding beyond validCounts
// When the slots are remapped through the candidate pool
// Then nodes 1..validCount-1 carry vocabulary ids in both token tensors, root and padding are untouched
TEST(DominoRolloutKernels, RemapPoolSlotsLeavesRootAndPaddingAlone)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 2;
    constexpr int32_t verifySize = 5;
    constexpr int32_t poolSize = 4;
    std::vector<int32_t> const treeHost{7, 2, 0, 3, 1, /* batch 1 */ 9, 1, 1, 3, 2};
    std::vector<int32_t> const verifyHost{7, 2, 0, 3, 0, /* batch 1 */ 9, 1, 1, 0, 0};
    std::vector<int32_t> const validHost{4, 3};
    std::vector<int32_t> const poolIdsHost{10, 11, 12, 13, /* batch 1 */ 20, 21, 22, 23};

    auto tree = rt::Tensor({batchSize, verifySize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto verify = rt::Tensor({batchSize, verifySize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto valid = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto poolIds = rt::Tensor({batchSize, poolSize}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<int32_t>(tree, treeHost);
    copyHostToDevice<int32_t>(verify, verifyHost);
    copyHostToDevice<int32_t>(valid, validHost);
    copyHostToDevice<int32_t>(poolIds, poolIdsHost);

    dominoRemapPoolSlots(tree, verify, valid, poolIds, batchSize, verifySize, poolSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> const expectedTree{7, 12, 10, 13, 1, /* batch 1 */ 9, 21, 21, 3, 2};
    std::vector<int32_t> const expectedVerify{7, 12, 10, 13, 0, /* batch 1 */ 9, 21, 21, 0, 0};
    EXPECT_EQ(copyDeviceToHost<int32_t>(tree), expectedTree);
    EXPECT_EQ(copyDeviceToHost<int32_t>(verify), expectedVerify);
}
