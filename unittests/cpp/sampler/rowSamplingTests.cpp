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

#include "common/checkMacros.h"
#include "common/tensor.h"
#include "sampler/sampling.h"
#include "testUtils.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;

namespace
{

RowSamplingParams greedy()
{
    return RowSamplingParams{0.0F, 1, 1.0F};
}

RowSamplingParams topK(int32_t k, float temperature = 1.0F)
{
    return RowSamplingParams{temperature, k, 1.0F};
}

RowSamplingParams topP(float p, float temperature = 1.0F)
{
    return RowSamplingParams{temperature, 0, p};
}

RowSamplingParams topKtopP(int32_t k, float p, float temperature = 1.0F)
{
    return RowSamplingParams{temperature, k, p};
}

class RowSamplingTest : public ::testing::Test
{
protected:
    //! 64 by default; a test may widen it so stage 1 spreads a row over all eight lanes.
    int32_t vocabSize{64};

    void SetUp() override
    {
        CUDA_CHECK(cudaSetDevice(0));
    }

    //! Logits with a distinct, well-separated ordering per row so that every parameter set
    //! selects unambiguously and rows can be told apart in the output.
    std::vector<float> makeLogits(int32_t rows)
    {
        std::mt19937 generator(7U);
        std::vector<float> logits(static_cast<size_t>(rows) * vocabSize);
        for (int32_t row = 0; row < rows; ++row)
        {
            std::vector<int32_t> order(vocabSize);
            std::iota(order.begin(), order.end(), 0);
            std::shuffle(order.begin(), order.end(), generator);
            for (int32_t rank = 0; rank < vocabSize; ++rank)
            {
                // Rank 0 gets the largest logit; gaps of 0.25 keep the ordering numerically firm.
                logits[static_cast<size_t>(row) * vocabSize + order[rank]] = 8.0F - 0.25F * static_cast<float>(rank);
            }
        }
        return logits;
    }

    std::vector<float> makeUniforms(int32_t rows)
    {
        std::mt19937 generator(11U);
        std::uniform_real_distribution<float> distribution(0.0F, 1.0F);
        std::vector<float> uniforms(static_cast<size_t>(rows));
        for (float& value : uniforms)
        {
            value = distribution(generator);
        }
        return uniforms;
    }

    //! Sample every row with its own parameters through the per-row launch.
    std::vector<int32_t> samplePerRow(std::vector<float> const& logits, std::vector<RowSamplingParams> const& rows,
        std::vector<float> const& uniforms)
    {
        int32_t const rowCount = static_cast<int32_t>(rows.size());
        std::vector<SamplingRow> hostRows;
        for (size_t i = 0; i < rows.size(); ++i)
        {
            hostRows.push_back(makeSamplingRow(rows[i], uniforms[i]));
        }
        rt::Tensor logitsTensor({rowCount, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
        rt::Tensor deviceRows(
            {static_cast<int64_t>(hostRows.size() * sizeof(SamplingRow))}, rt::DeviceType::kGPU, DataType::kINT8);
        rt::Tensor selected({rowCount, 1}, rt::DeviceType::kGPU, DataType::kINT32);
        rt::Tensor workspace({static_cast<int64_t>(getRowSamplingWorkspaceSize(rowCount, vocabSize))},
            rt::DeviceType::kGPU, DataType::kINT8);
        copyHostToDevice(logitsTensor, logits);
        CUDA_CHECK(cudaMemcpy(
            deviceRows.rawPointer(), hostRows.data(), hostRows.size() * sizeof(SamplingRow), cudaMemcpyHostToDevice));
        sampleRows(logitsTensor, selected, hostRows.data(), deviceRows, rowCount, workspace, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        return copyDeviceToHost<int32_t>(selected);
    }

    //! Sample one row on its own with the whole-batch kernels, exactly as a lone request would.
    int32_t sampleSolo(std::vector<float> const& logits, int32_t row, RowSamplingParams const& params, float uniform)
    {
        rt::Tensor logitsTensor({1, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
        rt::Tensor uniformsTensor({1}, rt::DeviceType::kGPU, DataType::kFLOAT);
        rt::Tensor selected({1, 1}, rt::DeviceType::kGPU, DataType::kINT32);
        std::vector<float> const rowLogits(logits.begin() + static_cast<ptrdiff_t>(row) * vocabSize,
            logits.begin() + static_cast<ptrdiff_t>(row + 1) * vocabSize);
        copyHostToDevice(logitsTensor, rowLogits);
        copyHostToDevice(uniformsTensor, std::vector<float>{uniform});
        if (shouldUseNonGreedySampling(params))
        {
            SamplingParams const samplingParams(1, vocabSize, params);
            rt::Tensor workspace({static_cast<int64_t>(getTopKtopPSamplingWorkspaceSize(1, vocabSize, samplingParams))},
                rt::DeviceType::kGPU, DataType::kINT8);
            topKtopPSamplingFromLogits(
                logitsTensor, selected, samplingParams, workspace, nullptr, 42U, 0U, &uniformsTensor);
        }
        else
        {
            rt::Tensor workspace({static_cast<int64_t>(getSelectAllTopKWorkspaceSize(1, vocabSize, 1))},
                rt::DeviceType::kGPU, DataType::kINT8);
            selectAllTopK(logitsTensor, std::nullopt, selected, 1, workspace, nullptr);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        return copyDeviceToHost<int32_t>(selected).front();
    }

    //! The whole-batch launch of one parameter set over every row.
    std::vector<int32_t> sampleBatch(
        std::vector<float> const& logits, RowSamplingParams const& params, std::vector<float> const& uniforms)
    {
        int32_t const rowCount = static_cast<int32_t>(uniforms.size());
        rt::Tensor logitsTensor({rowCount, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
        rt::Tensor uniformsTensor({rowCount}, rt::DeviceType::kGPU, DataType::kFLOAT);
        rt::Tensor selected({rowCount, 1}, rt::DeviceType::kGPU, DataType::kINT32);
        copyHostToDevice(logitsTensor, logits);
        copyHostToDevice(uniformsTensor, uniforms);
        SamplingParams const samplingParams(rowCount, vocabSize, params);
        rt::Tensor workspace(
            {static_cast<int64_t>(getTopKtopPSamplingWorkspaceSize(rowCount, vocabSize, samplingParams))},
            rt::DeviceType::kGPU, DataType::kINT8);
        topKtopPSamplingFromLogits(
            logitsTensor, selected, samplingParams, workspace, nullptr, 42U, 0U, &uniformsTensor);
        CUDA_CHECK(cudaDeviceSynchronize());
        return copyDeviceToHost<int32_t>(selected);
    }
};

} // namespace

TEST(SamplingRowTest, GreedyRowsBecomeTopKOne)
{
    // Greedy for different reasons: all land on the top-k path as k = 1 with a neutral temperature.
    for (RowSamplingParams const& params :
        {greedy(), RowSamplingParams{1.0F, 1, 0.5F}, RowSamplingParams{1.0F, 0, 1.0F}})
    {
        SamplingRow const row = makeSamplingRow(params, 0.25F);
        EXPECT_EQ(row.topK, 1);
        EXPECT_EQ(row.temperature, 1.0F);
        EXPECT_EQ(row.topP, 1.0F);
        EXPECT_EQ(row.uniform, 0.25F);
    }
    // Sampled rows pass through unchanged; top-k 0 keeps the row on the top-p path.
    SamplingRow const sampled = makeSamplingRow(topKtopP(40, 0.9F, 0.7F), 0.5F);
    EXPECT_EQ(sampled.topK, 40);
    EXPECT_EQ(sampled.temperature, 0.7F);
    EXPECT_EQ(sampled.topP, 0.9F);
    SamplingRow const nucleus = makeSamplingRow(topP(0.9F), 0.5F);
    EXPECT_EQ(nucleus.topK, 0);
    SamplingRow const temperatureOnly = makeSamplingRow(topP(1.0F, 2.0F), 0.5F);
    EXPECT_EQ(temperatureOnly.topK, 0);
    EXPECT_EQ(temperatureOnly.temperature, 2.0F);
}

TEST(SamplingRowTest, WorkspaceCoversBothPaths)
{
    for (int32_t rows : {1, 4, 16})
    {
        for (int32_t vocab : {1000, 32000, 151936})
        {
            size_t const bound = getRowSamplingWorkspaceSize(rows, vocab);
            SamplingParams const widestTopK(rows, vocab, 1.0F, kMaxSamplingTopK, 1.0F);
            SamplingParams const pureTopP(rows, vocab, 1.0F, 0, 0.9F);
            // One launch may run both paths, so the bound holds both regions at once.
            EXPECT_GE(bound,
                getTopKtopPSamplingWorkspaceSize(rows, vocab, widestTopK)
                    + getTopKtopPSamplingWorkspaceSize(rows, vocab, pureTopP));
            EXPECT_GE(bound, getSelectAllTopKWorkspaceSize(rows, vocab, 1));
        }
    }
}

TEST_F(RowSamplingTest, EveryRowMatchesItsSoloLaunch)
{
    std::vector<RowSamplingParams> const rows{greedy(), topK(5, 0.8F), topK(60), topP(0.6F), topKtopP(8, 0.7F),
        greedy(), topP(0.6F), topK(3, 1.5F), topP(1.0F, 2.0F)};
    int32_t const rowCount = static_cast<int32_t>(rows.size());
    auto const logits = makeLogits(rowCount);
    auto const uniforms = makeUniforms(rowCount);

    auto const sampled = samplePerRow(logits, rows, uniforms);

    ASSERT_EQ(sampled.size(), rows.size());
    for (int32_t row = 0; row < rowCount; ++row)
    {
        EXPECT_EQ(sampled[row], sampleSolo(logits, row, rows[row], uniforms[row])) << "row " << row;
    }

    // The same rows in reverse order: a row's token does not depend on its position or neighbours.
    std::vector<RowSamplingParams> reversedRows(rows.rbegin(), rows.rend());
    std::vector<float> reversedUniforms(uniforms.rbegin(), uniforms.rend());
    std::vector<float> reversedLogits;
    for (int32_t row = rowCount - 1; row >= 0; --row)
    {
        reversedLogits.insert(reversedLogits.end(), logits.begin() + static_cast<ptrdiff_t>(row) * vocabSize,
            logits.begin() + static_cast<ptrdiff_t>(row + 1) * vocabSize);
    }
    auto const reversed = samplePerRow(reversedLogits, reversedRows, reversedUniforms);
    EXPECT_EQ(reversed, std::vector<int32_t>(sampled.rbegin(), sampled.rend()));
}

// Stage 1 hands each row to eight blocks that each cover a 256-token stripe, so a 64-token vocabulary
// only exercises lane 0. A wide vocabulary with unequal k puts winners in every lane and makes the
// rows' candidate slots narrower than the batch's widest k.
TEST_F(RowSamplingTest, WideVocabularyFillsEveryLane)
{
    vocabSize = 4096;
    std::vector<RowSamplingParams> const rows{
        topK(5, 0.8F), topK(kMaxSamplingTopK), greedy(), topP(0.7F), topK(300, 0.8F), topKtopP(37, 0.5F), greedy()};
    int32_t const rowCount = static_cast<int32_t>(rows.size());
    auto const logits = makeLogits(rowCount);
    auto const uniforms = makeUniforms(rowCount);

    auto const sampled = samplePerRow(logits, rows, uniforms);

    ASSERT_EQ(sampled.size(), rows.size());
    for (int32_t row = 0; row < rowCount; ++row)
    {
        EXPECT_EQ(sampled[row], sampleSolo(logits, row, rows[row], uniforms[row])) << "row " << row;
    }
    // The argmax of a wide row sits in whichever lane holds it; greedy rows must still find it.
    for (int32_t row : {2, 6})
    {
        auto const begin = logits.begin() + static_cast<ptrdiff_t>(row) * vocabSize;
        EXPECT_EQ(sampled[row], static_cast<int32_t>(std::max_element(begin, begin + vocabSize) - begin));
    }
}

// A temperature alone (no top-k, top-p 1.0) samples the whole vocabulary instead of throwing from the launch.
TEST_F(RowSamplingTest, TemperatureOnlySamplesWholeVocab)
{
    RowSamplingParams const temperatureOnly{2.0F, 0, 1.0F};
    std::vector<RowSamplingParams> const rows{greedy(), temperatureOnly};
    ASSERT_TRUE(shouldUseNonGreedySampling(temperatureOnly));
    int32_t const rowCount = static_cast<int32_t>(rows.size());
    auto const logits = makeLogits(rowCount);
    std::vector<float> const uniforms(rows.size(), 0.999F);

    auto const sampled = samplePerRow(logits, rows, uniforms);

    ASSERT_EQ(sampled.size(), rows.size());
    auto const begin = logits.begin() + static_cast<ptrdiff_t>(1) * vocabSize;
    int32_t const argmax = static_cast<int32_t>(std::max_element(begin, begin + vocabSize) - begin);
    EXPECT_NE(sampled[1], argmax) << "a uniform near one must reach past the argmax when nothing is cut off";
    EXPECT_EQ(sampled[1], sampleSolo(logits, 1, temperatureOnly, uniforms[1]));
}

TEST_F(RowSamplingTest, GreedyRowsIgnoreNeighbours)
{
    std::vector<RowSamplingParams> const rows{topP(0.3F), greedy(), topK(2), greedy()};
    int32_t const rowCount = static_cast<int32_t>(rows.size());
    auto const logits = makeLogits(rowCount);
    // Uniforms close to one would push a sampled row away from its argmax; greedy rows ignore them.
    std::vector<float> const uniforms(rows.size(), 0.999F);

    auto const sampled = samplePerRow(logits, rows, uniforms);

    for (int32_t row : {1, 3})
    {
        auto const begin = logits.begin() + static_cast<ptrdiff_t>(row) * vocabSize;
        int32_t const argmax = static_cast<int32_t>(std::max_element(begin, begin + vocabSize) - begin);
        EXPECT_EQ(sampled[row], argmax) << "row " << row;
    }
}

TEST_F(RowSamplingTest, UniformBatchMatchesBatchLaunch)
{
    constexpr int32_t kRows = 6;
    auto const logits = makeLogits(kRows);
    auto const uniforms = makeUniforms(kRows);
    for (RowSamplingParams const& params : {topKtopP(10, 0.9F, 0.9F), topP(0.8F, 1.2F)})
    {
        std::vector<RowSamplingParams> const rows(kRows, params);
        EXPECT_EQ(samplePerRow(logits, rows, uniforms), sampleBatch(logits, params, uniforms))
            << "top-k " << params.topK;
    }
}

TEST_F(RowSamplingTest, RejectsContractViolations)
{
    std::vector<RowSamplingParams> const tooWide{topK(kMaxSamplingTopK + 1)};
    EXPECT_THROW(samplePerRow(makeLogits(1), tooWide, {0.5F}), std::runtime_error);

    constexpr int32_t kRows = 2;
    std::vector<SamplingRow> const hostRows(kRows, makeSamplingRow(topK(4), 0.5F));
    rt::Tensor logitsTensor({kRows, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor shortRows({static_cast<int64_t>(sizeof(SamplingRow))}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor selected({kRows, 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor workspace(
        {static_cast<int64_t>(getRowSamplingWorkspaceSize(kRows, vocabSize))}, rt::DeviceType::kGPU, DataType::kINT8);
    EXPECT_THROW(
        sampleRows(logitsTensor, selected, hostRows.data(), shortRows, kRows, workspace, nullptr), std::runtime_error);
}
