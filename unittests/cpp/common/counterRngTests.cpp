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

#include "common/counterRng.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <set>
#include <vector>

using namespace trt_edgellm;

namespace
{
constexpr uint64_t kSEED{1234};
} // namespace

// Given a grid of (offset, lane) keys under one seed, and the same grid under the next seed
// When random bits are drawn for every key
// Then every key gets its own bits, and changing the seed changes every key's bits
TEST(CounterRngTest, EveryKeyGetsItsOwnBits)
{
    constexpr uint64_t kGRID{64};
    std::set<uint64_t> bits;
    for (uint64_t offset = 0; offset < kGRID; ++offset)
    {
        for (uint64_t lane = 0; lane < kGRID; ++lane)
        {
            bits.insert(counter_rng::counterBits(kSEED, offset, lane));
            EXPECT_NE(counter_rng::counterBits(kSEED + 1, offset, lane), counter_rng::counterBits(kSEED, offset, lane));
        }
    }
    EXPECT_EQ(bits.size(), kGRID * kGRID);
}

// Given uniforms drawn over many (offset, lane) keys
// When they are bucketed into tenths
// Then every value lies in (0, 1] and each tenth holds about a tenth of them
TEST(CounterRngTest, UniformsAreEvenlySpread)
{
    constexpr uint64_t kGRID{256};
    std::array<int64_t, 10> buckets{};
    for (uint64_t offset = 0; offset < kGRID; ++offset)
    {
        for (uint64_t lane = 0; lane < kGRID; ++lane)
        {
            float const value = counter_rng::counterUniform(kSEED, offset, lane);
            ASSERT_GT(value, 0.0F);
            ASSERT_LE(value, 1.0F);
            ++buckets[std::min<size_t>(static_cast<size_t>(value * 10.0F), buckets.size() - 1)];
        }
    }
    for (size_t bucket = 0; bucket < buckets.size(); ++bucket)
    {
        EXPECT_NEAR(static_cast<double>(buckets[bucket]) / (kGRID * kGRID), 0.1, 0.01) << "bucket " << bucket;
    }
}

// Given normal samples drawn from consecutive lanes
// When their values are summarised
// Then they look like independent standard-normal draws: mean near 0, variance near 1, about 68% within one
// standard deviation, and no correlation between neighbouring values
TEST(CounterRngTest, NormalsAreIndependentStandardNormal)
{
    constexpr int64_t kCOUNT{int64_t{1} << 20};
    std::vector<float> samples;
    samples.reserve(kCOUNT);
    for (int64_t lane = 0; lane < kCOUNT; ++lane)
    {
        samples.push_back(counter_rng::normalFromBits(counter_rng::counterBits(kSEED, 0, static_cast<uint64_t>(lane))));
    }

    double sum{0.0};
    double sumSquares{0.0};
    double sumNeighbourProducts{0.0};
    int64_t withinOneSigma{0};
    for (size_t i = 0; i < samples.size(); ++i)
    {
        float const value = samples[i];
        ASSERT_TRUE(std::isfinite(value));
        sum += value;
        sumSquares += static_cast<double>(value) * value;
        withinOneSigma += std::fabs(value) < 1.0F ? 1 : 0;
        if (i + 1 < samples.size())
        {
            sumNeighbourProducts += static_cast<double>(value) * samples[i + 1];
        }
    }
    double const mean = sum / kCOUNT;
    double const variance = sumSquares / kCOUNT - mean * mean;
    EXPECT_NEAR(mean, 0.0, 0.01);
    EXPECT_NEAR(variance, 1.0, 0.02);
    EXPECT_NEAR(static_cast<double>(withinOneSigma) / kCOUNT, 0.6827, 0.01);
    double const lagOneCorrelation = (sumNeighbourProducts / (kCOUNT - 1) - mean * mean) / variance;
    EXPECT_NEAR(lagOneCorrelation, 0.0, 0.01);
}

// Given random bits at the two ends of the u1 range: top 24 bits all zero, then all one
// When they are mapped to a normal sample
// Then both samples are finite: about 5.89 at the smallest u1 and 0 at the largest
TEST(CounterRngTest, NormalFromExtremeBitsIsFinite)
{
    float const smallest = counter_rng::normalFromBits(0ULL);
    EXPECT_TRUE(std::isfinite(smallest));
    EXPECT_NEAR(smallest, std::sqrt(50.0 * std::log(2.0)), 1e-3);
    float const largest = counter_rng::normalFromBits(0xFFFFFF0000000000ULL);
    EXPECT_TRUE(std::isfinite(largest));
    EXPECT_NEAR(largest, 0.0F, 1e-6F);
}
