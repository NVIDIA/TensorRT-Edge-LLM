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
#include "kernels/speculative/lilicorrChainWalk.h"
#include "testUtils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{

struct ChainWalkTensors
{
    ChainWalkTensors(int32_t batch, int32_t steps, int32_t topK)
        : ids({batch, steps, topK}, rt::DeviceType::kGPU, DataType::kINT32)
        , scores({batch, steps, topK, topK}, rt::DeviceType::kGPU, DataType::kFLOAT)
        , uniforms({batch, steps}, rt::DeviceType::kGPU, DataType::kFLOAT)
        , temperatures({batch}, rt::DeviceType::kGPU, DataType::kFLOAT)
        , greedy({batch}, rt::DeviceType::kGPU, DataType::kINT32)
        , tokens({batch, steps}, rt::DeviceType::kGPU, DataType::kINT32)
        , q({batch, steps, topK}, rt::DeviceType::kGPU, DataType::kFLOAT)
    {
    }

    void launch()
    {
        launchLiLiCorrChainWalk(ids, scores, uniforms, temperatures, greedy, tokens, q, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    rt::Tensor ids;
    rt::Tensor scores;
    rt::Tensor uniforms;
    rt::Tensor temperatures;
    rt::Tensor greedy;
    rt::Tensor tokens;
    rt::Tensor q;
};

//! Host scores laid out as [B,S,prev,next].
class ScoreTable
{
public:
    ScoreTable(int32_t batch, int32_t steps, int32_t topK)
        : mSteps(steps)
        , mTopK(topK)
        , mValues(static_cast<size_t>(batch) * steps * topK * topK, 0.0F)
    {
    }

    void setRow(int32_t b, int32_t s, int32_t prev, std::vector<float> const& row)
    {
        for (int32_t next = 0; next < mTopK; ++next)
        {
            mValues[((static_cast<size_t>(b) * mSteps + s) * mTopK + prev) * mTopK + next] = row[next];
        }
    }

    std::vector<float> const& values() const
    {
        return mValues;
    }

private:
    int32_t mSteps;
    int32_t mTopK;
    std::vector<float> mValues;
};

std::vector<int32_t> sequentialIds(int32_t batch, int32_t steps, int32_t topK)
{
    std::vector<int32_t> ids(static_cast<size_t>(batch) * steps * topK);
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t s = 0; s < steps; ++s)
        {
            for (int32_t k = 0; k < topK; ++k)
            {
                ids[(b * steps + s) * topK + k] = 100 * b + 10 * (s + 1) + k;
            }
        }
    }
    return ids;
}

std::vector<float> softmaxReference(std::vector<float> const& row, double temperature)
{
    double maxValue = -INFINITY;
    for (float const v : row)
    {
        maxValue = std::max(maxValue, static_cast<double>(v) / temperature);
    }
    std::vector<double> weights(row.size());
    double sum = 0.0;
    for (size_t k = 0; k < row.size(); ++k)
    {
        weights[k] = std::exp(static_cast<double>(row[k]) / temperature - maxValue);
        sum += weights[k];
    }
    std::vector<float> probabilities(row.size());
    for (size_t k = 0; k < row.size(); ++k)
    {
        probabilities[k] = static_cast<float>(weights[k] / sum);
    }
    return probabilities;
}

void expectOneHot(std::vector<float> const& q, int64_t rowOffset, int32_t topK, int32_t selected)
{
    for (int32_t k = 0; k < topK; ++k)
    {
        EXPECT_EQ(q[rowOffset + k], k == selected ? 1.0F : 0.0F) << "row offset " << rowOffset << " k " << k;
    }
}

void expectProbabilities(std::vector<float> const& q, int64_t rowOffset, std::vector<float> const& expected)
{
    for (size_t k = 0; k < expected.size(); ++k)
    {
        EXPECT_NEAR(q[rowOffset + k], expected[k], 1.0e-6F) << "row offset " << rowOffset << " k " << k;
    }
}

} // namespace

TEST(LiLiCorrChainWalk, GreedyFollowsPreviousIndexRows)
{
    // Given one greedy request with 3 slots of 4 candidates whose transition rows T[s, prev, :] each peak at a
    // different next index, so reading any row other than T[s, previously chosen index, :] (including the transposed
    // column T[s, :, prev]) changes the path,
    // When the chain walk runs,
    // Then the walk starts from index 0, follows indices 2 -> 3 -> 0 through the previously chosen rows, emits the
    // candidate token IDs at those indices, and q is one-hot at each chosen index.
    constexpr int32_t batch{1};
    constexpr int32_t steps{3};
    constexpr int32_t topK{4};
    ChainWalkTensors t(batch, steps, topK);
    ScoreTable table(batch, steps, topK);
    std::vector<std::vector<int32_t>> const peakByPrev{{2, 1, 3, 0}, {0, 2, 3, 1}, {1, 3, 2, 0}};
    for (int32_t s = 0; s < steps; ++s)
    {
        for (int32_t prev = 0; prev < topK; ++prev)
        {
            std::vector<float> row(topK, 0.0F);
            row[peakByPrev[s][prev]] = 4.0F;
            table.setRow(0, s, prev, row);
        }
    }
    copyHostToDevice<int32_t>(t.ids, sequentialIds(batch, steps, topK));
    copyHostToDevice<float>(t.scores, table.values());
    copyHostToDevice<float>(t.uniforms, {0.5F, 0.5F, 0.5F});
    copyHostToDevice<float>(t.temperatures, {1.0F});
    copyHostToDevice<int32_t>(t.greedy, {1});

    t.launch();

    EXPECT_EQ(copyDeviceToHost<int32_t>(t.tokens), (std::vector<int32_t>{12, 23, 30}));
    auto const q = copyDeviceToHost<float>(t.q);
    std::vector<int32_t> const selected{2, 3, 0};
    for (int32_t s = 0; s < steps; ++s)
    {
        expectOneHot(q, s * topK, topK, selected[s]);
    }
}

TEST(LiLiCorrChainWalk, GreedyTieChoosesLowestIndex)
{
    // Given one greedy request whose slot-0 row ties indices 1, 2 and 3 for the maximum and whose slot-1 row ties all
    // four indices, with candidate token IDs decreasing in index,
    // When the chain walk runs,
    // Then each tie resolves to the lowest candidate index (not the highest index or the smallest token ID), giving
    // tokens 30 and 13 with one-hot q at indices 1 and 0.
    constexpr int32_t batch{1};
    constexpr int32_t steps{2};
    constexpr int32_t topK{4};
    ChainWalkTensors t(batch, steps, topK);
    ScoreTable table(batch, steps, topK);
    table.setRow(0, 0, 0, {1.0F, 3.0F, 3.0F, 3.0F});
    table.setRow(0, 1, 1, {5.0F, 5.0F, 5.0F, 5.0F});
    copyHostToDevice<int32_t>(t.ids, {40, 30, 20, 10, 13, 12, 11, 10});
    copyHostToDevice<float>(t.scores, table.values());
    copyHostToDevice<float>(t.uniforms, {0.5F, 0.5F});
    copyHostToDevice<float>(t.temperatures, {1.0F});
    copyHostToDevice<int32_t>(t.greedy, {1});

    t.launch();

    EXPECT_EQ(copyDeviceToHost<int32_t>(t.tokens), (std::vector<int32_t>{30, 13}));
    auto const q = copyDeviceToHost<float>(t.q);
    expectOneHot(q, 0, topK, 1);
    expectOneHot(q, topK, topK, 0);
}

TEST(LiLiCorrChainWalk, SamplingUsesInverseCdfOverSoftmax)
{
    // Given one sampling request at temperature 2 with per-slot uniforms 0.45 and 0.1, a slot-0 row of large scores
    // {200, 201, 202, 199.5} and a slot-1 row T[1, 2, :] = {0, 1, -1, 2} whose other rows favour index 1,
    // When the chain walk runs,
    // Then q at each slot equals softmax(row / 2) of the row selected by the previous index (within 1e-6 and
    // finite despite the large scores), and the inverse CDF of each slot's own uniform picks index 2 then index 0.
    constexpr int32_t batch{1};
    constexpr int32_t steps{2};
    constexpr int32_t topK{4};
    constexpr float temperature{2.0F};
    ChainWalkTensors t(batch, steps, topK);
    ScoreTable table(batch, steps, topK);
    std::vector<float> const slot0Row{200.0F, 201.0F, 202.0F, 199.5F};
    std::vector<float> const slot1Row{0.0F, 1.0F, -1.0F, 2.0F};
    table.setRow(0, 0, 0, slot0Row);
    for (int32_t prev = 0; prev < topK; ++prev)
    {
        table.setRow(0, 1, prev, prev == 2 ? slot1Row : std::vector<float>{-9.0F, 9.0F, -9.0F, -9.0F});
    }
    copyHostToDevice<int32_t>(t.ids, sequentialIds(batch, steps, topK));
    copyHostToDevice<float>(t.scores, table.values());
    copyHostToDevice<float>(t.uniforms, {0.45F, 0.1F});
    copyHostToDevice<float>(t.temperatures, {temperature});
    copyHostToDevice<int32_t>(t.greedy, {0});

    t.launch();

    EXPECT_EQ(copyDeviceToHost<int32_t>(t.tokens), (std::vector<int32_t>{12, 20}));
    auto const q = copyDeviceToHost<float>(t.q);
    expectProbabilities(q, 0, softmaxReference(slot0Row, temperature));
    expectProbabilities(q, topK, softmaxReference(slot1Row, temperature));
}

TEST(LiLiCorrChainWalk, NearZeroTemperatureIsGreedy)
{
    // Given two non-greedy-masked requests at temperatures 1e-4 and 0 whose slot-0 row ties indices 1 and 2 for the
    // maximum, with uniform 0.9 (sampling would pick index 2 at temperature 1e-4 and is undefined at temperature 0),
    // When the chain walk runs,
    // Then both requests take the greedy path (index 1, then the argmax of row 1) and every q entry is finite and
    // one-hot.
    constexpr int32_t batch{2};
    constexpr int32_t steps{2};
    constexpr int32_t topK{4};
    ChainWalkTensors t(batch, steps, topK);
    ScoreTable table(batch, steps, topK);
    for (int32_t b = 0; b < batch; ++b)
    {
        table.setRow(b, 0, 0, {1.0F, 3.0F, 3.0F, 0.0F});
        for (int32_t prev = 0; prev < topK; ++prev)
        {
            table.setRow(b, 1, prev,
                prev == 1 ? std::vector<float>{0.0F, 0.0F, 7.0F, 0.0F} : std::vector<float>{7.0F, 0.0F, 0.0F, 0.0F});
        }
    }
    copyHostToDevice<int32_t>(t.ids, sequentialIds(batch, steps, topK));
    copyHostToDevice<float>(t.scores, table.values());
    copyHostToDevice<float>(t.uniforms, {0.9F, 0.9F, 0.9F, 0.9F});
    copyHostToDevice<float>(t.temperatures, {1.0e-4F, 0.0F});
    copyHostToDevice<int32_t>(t.greedy, {0, 0});

    t.launch();

    EXPECT_EQ(copyDeviceToHost<int32_t>(t.tokens), (std::vector<int32_t>{11, 22, 111, 122}));
    auto const q = copyDeviceToHost<float>(t.q);
    for (float const value : q)
    {
        EXPECT_TRUE(std::isfinite(value));
    }
    for (int32_t b = 0; b < batch; ++b)
    {
        expectOneHot(q, (b * steps + 0) * topK, topK, 1);
        expectOneHot(q, (b * steps + 1) * topK, topK, 2);
    }
}

TEST(LiLiCorrChainWalk, TemperatureAboveGreedyThresholdSamples)
{
    // Given one non-greedy-masked request at temperature 2e-3, just above the 1e-3 greedy threshold, whose row ties
    // indices 1 and 2 for the maximum, with uniform 0.9,
    // When the chain walk runs,
    // Then the request samples instead of taking the greedy index 1: q = {0, 0.5, 0.5, 0} and the inverse CDF picks
    // index 2.
    constexpr int32_t batch{1};
    constexpr int32_t steps{1};
    constexpr int32_t topK{4};
    ChainWalkTensors t(batch, steps, topK);
    ScoreTable table(batch, steps, topK);
    table.setRow(0, 0, 0, {1.0F, 3.0F, 3.0F, 0.0F});
    copyHostToDevice<int32_t>(t.ids, sequentialIds(batch, steps, topK));
    copyHostToDevice<float>(t.scores, table.values());
    copyHostToDevice<float>(t.uniforms, {0.9F});
    copyHostToDevice<float>(t.temperatures, {2.0e-3F});
    copyHostToDevice<int32_t>(t.greedy, {0});

    t.launch();

    EXPECT_EQ(copyDeviceToHost<int32_t>(t.tokens), (std::vector<int32_t>{12}));
    expectProbabilities(copyDeviceToHost<float>(t.q), 0, {0.0F, 0.5F, 0.5F, 0.0F});
}

TEST(LiLiCorrChainWalk, SamplingClampsUniformsAtProductionTopK)
{
    // Given one sampling request with the production K = 8 at temperature 1, a slot-0 row where candidate 0 has zero
    // probability and candidates 1..7 share the mass, a slot-1 row T[1, 1, :] that puts all mass on index 4 (other
    // rows favour index 2), and out-of-range uniforms -0.5 and 2.0,
    // When the chain walk runs,
    // Then the uniforms are clamped into [0, 1): -0.5 never selects the zero-probability candidate 0 and picks index
    // 1, and 2.0 picks index 4 rather than falling back to index K-1, with q = softmax(row) of each selected row.
    constexpr int32_t batch{1};
    constexpr int32_t steps{2};
    constexpr int32_t topK{8};
    ChainWalkTensors t(batch, steps, topK);
    ScoreTable table(batch, steps, topK);
    std::vector<float> slot0Row(topK, 0.0F);
    slot0Row[0] = -1000.0F;
    std::vector<float> slot1Row(topK, -1000.0F);
    slot1Row[4] = 0.0F;
    std::vector<float> otherSlot1Row(topK, -1000.0F);
    otherSlot1Row[2] = 0.0F;
    table.setRow(0, 0, 0, slot0Row);
    for (int32_t prev = 0; prev < topK; ++prev)
    {
        table.setRow(0, 1, prev, prev == 1 ? slot1Row : otherSlot1Row);
    }
    copyHostToDevice<int32_t>(t.ids, sequentialIds(batch, steps, topK));
    copyHostToDevice<float>(t.scores, table.values());
    copyHostToDevice<float>(t.uniforms, {-0.5F, 2.0F});
    copyHostToDevice<float>(t.temperatures, {1.0F});
    copyHostToDevice<int32_t>(t.greedy, {0});

    t.launch();

    EXPECT_EQ(copyDeviceToHost<int32_t>(t.tokens), (std::vector<int32_t>{11, 24}));
    auto const q = copyDeviceToHost<float>(t.q);
    expectProbabilities(q, 0, softmaxReference(slot0Row, 1.0));
    expectProbabilities(q, topK, softmaxReference(slot1Row, 1.0));
}

TEST(LiLiCorrChainWalk, BatchesAreIndependent)
{
    // Given three requests with distinct scores, candidates, uniforms and temperatures, where requests 0 and 2 are
    // greedy-masked and request 1 samples at temperature 0.5,
    // When the chain walk runs once over the batch,
    // Then each request follows only its own rows, mask, temperature and uniforms: request 0 picks indices 1 -> 2,
    // request 1 samples 2 -> 0 with q = softmax(row / 0.5), and request 2 picks 3 -> 0, with one-hot q for the
    // greedy requests.
    constexpr int32_t batch{3};
    constexpr int32_t steps{2};
    constexpr int32_t topK{4};
    constexpr float sampledTemperature{0.5F};
    ChainWalkTensors t(batch, steps, topK);
    ScoreTable table(batch, steps, topK);
    std::vector<float> const sampledSlot0Row{0.0F, 0.5F, 1.0F, 1.5F};
    std::vector<float> const sampledSlot1Row{2.0F, 0.0F, 0.0F, 0.0F};
    for (int32_t prev = 0; prev < topK; ++prev)
    {
        table.setRow(0, 0, prev,
            prev == 0 ? std::vector<float>{0.0F, 5.0F, 1.0F, 2.0F} : std::vector<float>{9.0F, 0.0F, 0.0F, 0.0F});
        table.setRow(0, 1, prev,
            prev == 1 ? std::vector<float>{0.0F, 1.0F, 7.0F, 2.0F} : std::vector<float>{9.0F, 0.0F, 0.0F, 0.0F});
        table.setRow(1, 0, prev, prev == 0 ? sampledSlot0Row : std::vector<float>{0.0F, 0.0F, 0.0F, 9.0F});
        table.setRow(1, 1, prev, prev == 2 ? sampledSlot1Row : std::vector<float>{0.0F, 0.0F, 0.0F, 2.0F});
        table.setRow(2, 0, prev,
            prev == 0 ? std::vector<float>{1.0F, 2.0F, 3.0F, 8.0F} : std::vector<float>{0.0F, 0.0F, 9.0F, 0.0F});
        table.setRow(2, 1, prev,
            prev == 3 ? std::vector<float>{6.0F, 1.0F, 2.0F, 3.0F} : std::vector<float>{0.0F, 0.0F, 9.0F, 0.0F});
    }
    copyHostToDevice<int32_t>(t.ids, sequentialIds(batch, steps, topK));
    copyHostToDevice<float>(t.scores, table.values());
    copyHostToDevice<float>(t.uniforms, {0.99F, 0.99F, 0.2F, 0.5F, 0.99F, 0.99F});
    copyHostToDevice<float>(t.temperatures, {1.0F, sampledTemperature, 1.0F});
    copyHostToDevice<int32_t>(t.greedy, {1, 0, 1});

    t.launch();

    EXPECT_EQ(copyDeviceToHost<int32_t>(t.tokens), (std::vector<int32_t>{11, 22, 112, 120, 213, 220}));
    auto const q = copyDeviceToHost<float>(t.q);
    expectOneHot(q, 0 * topK, topK, 1);
    expectOneHot(q, 1 * topK, topK, 2);
    expectProbabilities(q, 2 * topK, softmaxReference(sampledSlot0Row, sampledTemperature));
    expectProbabilities(q, 3 * topK, softmaxReference(sampledSlot1Row, sampledTemperature));
    expectOneHot(q, 4 * topK, topK, 3);
    expectOneHot(q, 5 * topK, topK, 0);
}

TEST(LiLiCorrChainWalk, RejectsShapeMismatch)
{
    // Given otherwise consistent tensors with K = kLiLiCorrMaxTopK (16), with K = 32, and with K = 4 but
    // transition scores flattened to rank 3 [B, S, K*K],
    // When the chain walk is launched,
    // Then K = 16 is accepted, while K = 32 and the rank-3 scores are rejected with std::runtime_error before launch.
    constexpr int32_t batch{1};
    constexpr int32_t steps{2};
    {
        ChainWalkTensors t(batch, steps, kLiLiCorrMaxTopK);
        copyHostToDevice<int32_t>(t.ids, std::vector<int32_t>(batch * steps * kLiLiCorrMaxTopK, 0));
        copyHostToDevice<float>(
            t.scores, std::vector<float>(batch * steps * kLiLiCorrMaxTopK * kLiLiCorrMaxTopK, 0.0F));
        copyHostToDevice<float>(t.uniforms, std::vector<float>(batch * steps, 0.5F));
        copyHostToDevice<float>(t.temperatures, {1.0F});
        copyHostToDevice<int32_t>(t.greedy, {1});
        EXPECT_NO_THROW(t.launch());
    }
    {
        ChainWalkTensors t(batch, steps, 32);
        EXPECT_THROW(t.launch(), std::runtime_error);
    }
    {
        constexpr int32_t topK{4};
        ChainWalkTensors t(batch, steps, topK);
        rt::Tensor flatScores({batch, steps, topK * topK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        EXPECT_THROW(
            launchLiLiCorrChainWalk(t.ids, flatScores, t.uniforms, t.temperatures, t.greedy, t.tokens, t.q, nullptr),
            std::runtime_error);
    }
}
