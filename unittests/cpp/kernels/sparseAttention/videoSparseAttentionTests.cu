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
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/sparseAttention/videoSparseAttentionRunner.h"

#include <cuda_bf16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <vector>

using namespace trt_edgellm;

namespace
{

using Runner = VideoSparseAttentionRunner;
constexpr int32_t kBLOCK{Runner::kBLOCK_SIZE};
constexpr int32_t kHEADS{Runner::kNUM_HEADS};
constexpr int32_t kDIM{Runner::kHEAD_SIZE};
//! Only these leading channels of Q and K are nonzero, which keeps the CPU reference cheap.
constexpr int32_t kACTIVE_DIMS{2};
constexpr float kSOFTMAX_SCALE{0x1.6A09E6P-4F};

// Q, K, and V vary by block, lane, and head so that exact and summarized key blocks differ.
__host__ __device__ float queryValue(int32_t token, int32_t head, int32_t dimension)
{
    int32_t const block = token / kBLOCK;
    if (dimension == 0)
    {
        return 8.0F + 2.0F * static_cast<float>(block % 3);
    }
    if (dimension == 1)
    {
        return (head % 2 == 0 ? 6.0F : -6.0F) + (block % 2 == 0 ? -2.0F : 2.0F);
    }
    return 0.0F;
}

__host__ __device__ float keyValue(int32_t token, int32_t head, int32_t dimension)
{
    int32_t const block = token / kBLOCK;
    int32_t const lane = token % kBLOCK;
    if (dimension == 0)
    {
        return static_cast<float>(block % 17 - 8) * 0.125F + static_cast<float>(head % 3 - 1) * 0.0625F
            + (lane % 2 == 0 ? -0.75F : 0.75F);
    }
    if (dimension == 1)
    {
        return static_cast<float>((block * 5 + head) % 13 - 6) * 0.125F + (lane % 4 < 2 ? -0.5625F : 0.5625F);
    }
    return 0.0F;
}

__host__ __device__ float valueValue(int32_t token, int32_t head, int32_t dimension)
{
    int32_t const block = token / kBLOCK;
    int32_t const lane = token % kBLOCK;
    return static_cast<float>(block % 17 - 8) * 0.0625F + static_cast<float>((block * 5 + head) % 13 - 6) * 0.03125F
        + static_cast<float>(dimension % 17 - 8) * 0.015625F
        + (lane % 2 == 0 ? -1.0F : 1.0F) * static_cast<float>(1 + dimension % 5) * 0.09375F
        + (lane % 4 < 2 ? -1.0F : 1.0F) * static_cast<float>(1 + dimension % 3) * 0.046875F;
}

__global__ void fillQkv(__nv_bfloat16* query, __nv_bfloat16* key, __nv_bfloat16* value, int64_t count)
{
    int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
    {
        int32_t const dimension = index % kDIM;
        int32_t const head = (index / kDIM) % kHEADS;
        int32_t const token = index / (kDIM * kHEADS);
        query[index] = __float2bfloat16_rn(queryValue(token, head, dimension));
        key[index] = __float2bfloat16_rn(keyValue(token, head, dimension));
        value[index] = __float2bfloat16_rn(valueValue(token, head, dimension));
    }
}

float bf16(float value)
{
    return __bfloat162float(__float2bfloat16_rn(value));
}

struct Reference
{
    std::array<float, kDIM> output{};
    int32_t exactBlocks{};
};

//! Independent CPU routing and block-summarized softmax for one query row.
Reference reference(int32_t sequence, int32_t prefix, int32_t token, int32_t head, bool routed, float scale)
{
    constexpr float kLOG2E{1.4426950408889634F};
    constexpr float kVARIANCE_EPSILON{1.0E-6F};
    int32_t const blocks = (sequence + kBLOCK - 1) / kBLOCK;
    int32_t const prefixBlocks = (prefix + kBLOCK - 1) / kBLOCK;
    auto const blockRange = [sequence](int32_t block) {
        return std::pair<int32_t, int32_t>{block * kBLOCK, std::min(kBLOCK, sequence - block * kBLOCK)};
    };

    int32_t const queryBlock = token / kBLOCK;
    auto const [queryFirst, queryLength] = blockRange(queryBlock);
    std::array<float, kACTIVE_DIMS> centroid{};
    for (int32_t row = queryFirst; row < queryFirst + queryLength; ++row)
    {
        for (int32_t d = 0; d < kACTIVE_DIMS; ++d)
        {
            centroid[d] += bf16(queryValue(row, head, d)) / static_cast<float>(queryLength);
        }
    }
    std::vector<std::array<float, kACTIVE_DIMS>> keyMeans(blocks);
    std::array<float, kACTIVE_DIMS> total{};
    std::array<float, kACTIVE_DIMS> totalSquare{};
    for (int32_t block = 0; block < blocks; ++block)
    {
        auto const [first, length] = blockRange(block);
        for (int32_t d = 0; d < kACTIVE_DIMS; ++d)
        {
            float sum{};
            for (int32_t row = first; row < first + length; ++row)
            {
                sum += bf16(keyValue(row, head, d));
            }
            keyMeans[block][d] = bf16(sum / static_cast<float>(length));
            total[d] += keyMeans[block][d];
            totalSquare[d] += keyMeans[block][d] * keyMeans[block][d];
        }
    }
    float const log2Scale = scale * kLOG2E;
    float thresholdMean{};
    float thresholdVariance{};
    for (int32_t d = 0; d < kACTIVE_DIMS; ++d)
    {
        float const mean = total[d] / static_cast<float>(blocks);
        thresholdMean += centroid[d] * mean;
        thresholdVariance
            += centroid[d] * centroid[d] * std::max(totalSquare[d] / static_cast<float>(blocks) - mean * mean, 0.0F);
    }
    float const threshold
        = thresholdMean * log2Scale + std::sqrt(thresholdVariance * log2Scale * log2Scale + kVARIANCE_EPSILON);

    Reference result;
    std::array<double, kDIM> numerator{};
    double denominator{};
    for (int32_t block = 0; block < blocks; ++block)
    {
        auto const [first, length] = blockRange(block);
        float routeScore{};
        for (int32_t d = 0; d < kACTIVE_DIMS; ++d)
        {
            routeScore += centroid[d] * keyMeans[block][d];
        }
        bool const exact = routed
            && (routeScore * log2Scale > threshold || std::abs(queryBlock - block) <= 1 || block < prefixBlocks
                || token < prefix);
        if (exact)
        {
            ++result.exactBlocks;
            for (int32_t row = first; row < first + length; ++row)
            {
                float score{};
                for (int32_t d = 0; d < kACTIVE_DIMS; ++d)
                {
                    score += bf16(queryValue(token, head, d)) * bf16(keyValue(row, head, d));
                }
                double const probability = std::exp(static_cast<double>(score * scale));
                denominator += probability;
                for (int32_t d = 0; d < kDIM; ++d)
                {
                    numerator[d] += probability * bf16(valueValue(row, head, d));
                }
            }
            continue;
        }
        float score{};
        for (int32_t d = 0; d < kACTIVE_DIMS; ++d)
        {
            score += bf16(queryValue(token, head, d)) * keyMeans[block][d];
        }
        double const probability = std::exp(static_cast<double>(score * scale));
        denominator += probability * length;
        for (int32_t d = 0; d < kDIM; ++d)
        {
            float valueSum{};
            for (int32_t row = first; row < first + length; ++row)
            {
                valueSum += bf16(valueValue(row, head, d));
            }
            numerator[d] += probability * bf16(valueSum);
        }
    }
    for (int32_t d = 0; d < kDIM; ++d)
    {
        result.output[d] = static_cast<float>(numerator[d] / denominator);
    }
    return result;
}

double cosine(std::array<float, kDIM> const& lhs, std::array<float, kDIM> const& rhs)
{
    double dot{};
    double lhsSquare{};
    double rhsSquare{};
    for (int32_t d = 0; d < kDIM; ++d)
    {
        dot += static_cast<double>(lhs[d]) * rhs[d];
        lhsSquare += static_cast<double>(lhs[d]) * lhs[d];
        rhsSquare += static_cast<double>(rhs[d]) * rhs[d];
    }
    return dot / std::sqrt(lhsSquare * rhsSquare);
}

//! (suffix rows, prefix rows)
class VideoSparseAttentionTest : public ::testing::TestWithParam<std::tuple<int32_t, int32_t>>
{
};

TEST_P(VideoSparseAttentionTest, MatchesIndependentReference)
{
    // Given nonconstant Q/K/V that makes exact and approximate routes observably different, when the production
    // runner executes, then sampled rows match an independent CPU implementation of routing and sparse softmax.
    int32_t const smVersion = getSMVersion();
    if (!Runner::canImplement(kHEADS, kDIM, smVersion))
    {
        GTEST_SKIP() << "No video sparse-attention kernel for SM " << smVersion;
    }
    auto const [suffix, prefix] = GetParam();
    int32_t const sequence = suffix + prefix;
    int64_t const count = static_cast<int64_t>(sequence) * kHEADS * kDIM;
    rt::Tensor query({count}, rt::DeviceType::kGPU, DataType::kBF16);
    rt::Tensor key({count}, rt::DeviceType::kGPU, DataType::kBF16);
    rt::Tensor value({count}, rt::DeviceType::kGPU, DataType::kBF16);
    rt::Tensor output({count}, rt::DeviceType::kGPU, DataType::kBF16);
    // One extra byte lets the workspace start misaligned, as TensorRT may hand it out.
    rt::Tensor workspace(
        {static_cast<int64_t>(Runner::getWorkspaceSize(sequence)) + 1}, rt::DeviceType::kGPU, DataType::kUINT8);
    constexpr int32_t kTHREADS{256};
    fillQkv<<<(count + kTHREADS - 1) / kTHREADS, kTHREADS>>>(query.dataPointer<__nv_bfloat16>(),
        key.dataPointer<__nv_bfloat16>(), value.dataPointer<__nv_bfloat16>(), count);
    CUDA_CHECK(cudaGetLastError());
    ASSERT_TRUE(Runner::run(query.rawPointer(), key.rawPointer(), value.rawPointer(), output.rawPointer(), sequence,
        prefix, workspace.dataPointer<uint8_t>() + 1, smVersion, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    int32_t const blocks = (sequence + kBLOCK - 1) / kBLOCK;
    int32_t const prefixBlocks = (prefix + kBLOCK - 1) / kBLOCK;
    std::array<std::pair<int32_t, int32_t>, 5> const samples{
        {{0, 0}, {prefix - 1, kHEADS - 1}, {prefixBlocks * kBLOCK, 0}, {sequence / 2, 17}, {sequence - 1, kHEADS - 1}}};
    for (auto const& [token, head] : samples)
    {
        std::array<__nv_bfloat16, kDIM> row{};
        CUDA_CHECK(cudaMemcpy(row.data(), output.dataPointer<__nv_bfloat16>() + (int64_t{token} * kHEADS + head) * kDIM,
            sizeof(row), cudaMemcpyDeviceToHost));
        Reference const expected = reference(sequence, prefix, token, head, true, kSOFTMAX_SCALE);
        std::array<float, kDIM> actual{};
        float maxError{};
        for (int32_t d = 0; d < kDIM; ++d)
        {
            actual[d] = __bfloat162float(row[d]);
            ASSERT_TRUE(std::isfinite(actual[d])) << "token " << token << " channel " << d;
            maxError = std::max(maxError, std::abs(actual[d] - expected.output[d]));
        }
        if (token < prefix)
        {
            EXPECT_EQ(expected.exactBlocks, blocks) << "token " << token;
        }
        else
        {
            EXPECT_GT(expected.exactBlocks, prefixBlocks + 2) << "token " << token;
            EXPECT_LT(expected.exactBlocks, blocks - 2) << "token " << token;
        }
        EXPECT_LT(cosine(expected.output, reference(sequence, prefix, token, head, false, kSOFTMAX_SCALE).output), 0.99)
            << "token " << token;
        EXPECT_GT(cosine(actual, expected.output), 0.99999) << "token " << token;
        EXPECT_LT(maxError, 0.005F) << "token " << token;
    }
}

INSTANTIATE_TEST_SUITE_P(PackedSequences, VideoSparseAttentionTest,
    ::testing::Values(std::make_tuple(4608, 248), std::make_tuple(37296, 532)));

} // namespace
