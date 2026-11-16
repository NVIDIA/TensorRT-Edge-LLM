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
#include "kernels/speculative/lowRankCandidateHeadKernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{

//! H 128, r 32, a vocabulary of 4000 (a partial last 64-row tile) and Kc 64.
constexpr LowRankCandidateHeadShape kShape{128, 32, 4000, 64};

//! Integer weights: every product and partial sum is an integer far below 2^24 (|s| <= 3H, |score| <= 6rH,
//! |logit| <= 6H), so fp32 accumulation in any order is exact; scores and logits tie often.
struct HostHead
{
    int32_t rows{0};
    std::vector<half> hidden;   //!< [rows, H], entries in [-3, 3]
    std::vector<half> downProj; //!< [r, H], entries in [-1, 1]
    std::vector<half> upProj;   //!< [V, r], entries in [-2, 2]
    std::vector<half> lmHead;   //!< [V, H], entries in [-2, 2]

    HostHead(int32_t numRows, uint32_t seed)
        : rows(numRows)
    {
        std::mt19937 rng(seed);
        auto fill = [&](std::vector<half>& values, size_t count, int32_t bound) {
            std::uniform_int_distribution<int32_t> dist(-bound, bound);
            values.resize(count);
            for (auto& value : values)
            {
                value = __float2half_rn(static_cast<float>(dist(rng)));
            }
        };
        size_t const h = kShape.hiddenSize;
        fill(hidden, static_cast<size_t>(rows) * h, 3);
        fill(downProj, static_cast<size_t>(kShape.rank) * h, 1);
        fill(upProj, static_cast<size_t>(kShape.vocabSize) * kShape.rank, 2);
        fill(lmHead, static_cast<size_t>(kShape.vocabSize) * h, 2);
    }
};

double dot(half const* a, half const* b, int32_t n)
{
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i)
    {
        sum += double{__half2float(a[i])} * __half2float(b[i]);
    }
    return sum;
}

//! CPU reference: s = fp16(downProj . h), scores upProj . s for every vocab row, the top Kc by a full sort on
//! (score desc with NaN after every number, id asc), exact logits lmHead . h (non-finite counts as -inf) sorted by
//! (logit desc, id asc), then log_softmax over the Kc logits. Returns ids and log-probs, [rows, Kc] each.
std::pair<std::vector<int32_t>, std::vector<float>> cpuReference(HostHead const& head)
{
    int32_t const h = kShape.hiddenSize;
    int32_t const r = kShape.rank;
    std::pair<std::vector<int32_t>, std::vector<float>> result;
    std::vector<half> s(r);
    std::vector<double> scores(kShape.vocabSize);
    std::vector<int32_t> order(kShape.vocabSize);
    for (int32_t row = 0; row < head.rows; ++row)
    {
        half const* hRow = head.hidden.data() + static_cast<size_t>(row) * h;
        for (int32_t k = 0; k < r; ++k)
        {
            s[k] = __float2half_rn(static_cast<float>(dot(head.downProj.data() + static_cast<size_t>(k) * h, hRow, h)));
        }
        for (int32_t v = 0; v < kShape.vocabSize; ++v)
        {
            scores[v] = dot(head.upProj.data() + static_cast<size_t>(v) * r, s.data(), r);
        }
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            bool const aNan = std::isnan(scores[a]);
            bool const bNan = std::isnan(scores[b]);
            if (aNan != bNan)
            {
                return bNan;
            }
            return (!aNan && scores[a] != scores[b]) ? scores[a] > scores[b] : a < b;
        });
        std::vector<std::pair<double, int32_t>> logits;
        for (int32_t c = 0; c < kShape.numCandidates; ++c)
        {
            double const logit = dot(head.lmHead.data() + static_cast<size_t>(order[c]) * h, hRow, h);
            logits.emplace_back(std::isfinite(logit) ? logit : -std::numeric_limits<double>::infinity(), order[c]);
        }
        std::sort(logits.begin(), logits.end(),
            [](auto const& a, auto const& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
        double sumExp = 0.0;
        for (auto const& [logit, id] : logits)
        {
            sumExp += std::exp(logit - logits.front().first);
        }
        for (auto const& [logit, id] : logits)
        {
            result.first.push_back(id);
            result.second.push_back(static_cast<float>(logit - logits.front().first - std::log(sumExp)));
        }
    }
    return result;
}

rt::Tensor upload(std::vector<half> const& values, int64_t rows, int64_t cols)
{
    rt::Tensor tensor({rows, cols}, rt::DeviceType::kGPU, DataType::kHALF);
    CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), values.data(), values.size() * sizeof(half), cudaMemcpyHostToDevice));
    return tensor;
}

//! Runs the fused head on every row of `head` with a runner sized for `maxRows`, on a workspace filled with garbage,
//! and checks each row against the CPU reference: identical ids and log-probs within 2e-6 + 2e-7 |log-prob|.
void expectMatchesReference(HostHead const& head, cudaDeviceProp const& prop, int32_t maxRows)
{
    LowRankCandidateHeadRunner const runner(kShape, maxRows, prop);
    auto const [expectedIds, expectedLogProbs] = cpuReference(head);
    rt::Tensor const hidden = upload(head.hidden, head.rows, kShape.hiddenSize);
    rt::Tensor const downProj = upload(head.downProj, kShape.rank, kShape.hiddenSize);
    rt::Tensor const upProj = upload(head.upProj, kShape.vocabSize, kShape.rank);
    rt::Tensor const lmHead = upload(head.lmHead, kShape.vocabSize, kShape.hiddenSize);
    rt::Tensor ids({head.rows, kShape.numCandidates}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor logProbs({head.rows, kShape.numCandidates}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor workspace({static_cast<int64_t>(runner.workspaceBytes())}, rt::DeviceType::kGPU, DataType::kINT8);
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemsetAsync(workspace.rawPointer(), 0xA5, runner.workspaceBytes(), stream));
    runner.run(hidden, downProj, upProj, lmHead, ids, logProbs, workspace.rawPointer(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    size_t const count = expectedIds.size();
    std::vector<int32_t> actualIds(count);
    std::vector<float> actualLogProbs(count);
    CUDA_CHECK(cudaMemcpy(actualIds.data(), ids.rawPointer(), count * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actualLogProbs.data(), logProbs.rawPointer(), count * sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_EQ(actualIds, expectedIds);
    for (size_t i = 0; i < count; ++i)
    {
        float const expected = expectedLogProbs[i];
        if (std::isinf(expected))
        {
            EXPECT_EQ(actualLogProbs[i], expected) << "flat index " << i;
        }
        else
        {
            EXPECT_NEAR(actualLogProbs[i], expected, 2e-6 + 2e-7 * std::abs(expected)) << "flat index " << i;
        }
    }
}

cudaDeviceProp deviceProp()
{
    int32_t device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop;
}

} // namespace

TEST(LowRankCandidateHeadTest, MatchesCpuReference)
{
    // Given integer weights at H 128, r 32, V 4000, Kc 64 (exact fp32 sums, frequent ties in scores and logits) with a
    // zero hidden row 2 (every score and logit 0), a second head whose low-rank scores are NaN for every vocab row
    // except 42 (fewer than Kc) and whose lmHead rows of three of those 42 hold NaN or inf, and runners sized for 70
    // rows with this device's configuration and with the configuration of a 20-SM device (fewer merge CTAs per row),
    // when each runner runs the fused head on the 70 rows of the first head (a 64-row and a 6-row pass) and on the 17
    // rows of the second,
    // then every row returns the CPU reference's ids exactly (the top Kc of a full sort by (score desc with NaN after
    // every number, id asc), in (exact logit desc, id asc) order) and its log_softmax log-probs within
    // 2e-6 + 2e-7 |log-prob|, with -inf for every candidate whose exact logit is not finite.
    HostHead head(70, 17U);
    std::fill_n(head.hidden.begin() + 2 * kShape.hiddenSize, kShape.hiddenSize, __float2half_rn(0.0F));
    HostHead nanHead(17, 29U);
    float const nan = std::numeric_limits<float>::quiet_NaN();
    float const inf = std::numeric_limits<float>::infinity();
    for (int32_t v = 0; v < kShape.vocabSize; ++v)
    {
        if (v % 97 != 5)
        {
            nanHead.upProj[static_cast<size_t>(v) * kShape.rank] = __float2half_rn(nan);
        }
    }
    nanHead.lmHead[static_cast<size_t>(5) * kShape.hiddenSize + 3] = __float2half_rn(nan);
    nanHead.lmHead[static_cast<size_t>(102) * kShape.hiddenSize + 7] = __float2half_rn(inf);
    nanHead.lmHead[static_cast<size_t>(3885) * kShape.hiddenSize] = __float2half_rn(-inf);
    cudaDeviceProp smallDevice = deviceProp();
    smallDevice.multiProcessorCount = 20;
    for (cudaDeviceProp const& prop : {deviceProp(), smallDevice})
    {
        SCOPED_TRACE(testing::Message() << prop.multiProcessorCount << " SMs");
        expectMatchesReference(head, prop, head.rows);
        expectMatchesReference(nanHead, prop, head.rows);
    }
}
