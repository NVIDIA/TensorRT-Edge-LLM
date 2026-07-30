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

#include "runtime/preprocess/mediaArtifactCache.h"

#include <gtest/gtest.h>

using namespace trt_edgellm::rt;

namespace
{

MediaArtifactKey key(uint64_t value)
{
    return MediaArtifactKey{Hash128{value, value + 1}, {}, MediaModality::kVision};
}

Tensor makeTensor(int64_t elements)
{
    return Tensor({elements}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
}

} // namespace

TEST(MediaArtifactCacheTests, InsertAcquireAndReleasePreserveOwnedCopy)
{
    Tensor source = makeTensor(4);
    auto* values = source.dataPointer<int32_t>();
    values[0] = 7;
    MediaArtifactCache cache(/*capacityBytes=*/64, /*maxEntries=*/2);

    std::optional<MediaArtifactLease> inserted = cache.insert(key(1), source, {}, nullptr);
    ASSERT_TRUE(inserted.has_value());
    values[0] = 9;
    EXPECT_EQ(inserted->embedding().dataPointer<int32_t>()[0], 7);
    inserted.reset();

    std::optional<MediaArtifactLease> hit = cache.acquire(key(1));
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->embedding().dataPointer<int32_t>()[0], 7);
}

TEST(MediaArtifactCacheTests, PinnedHitProtectsItselfFromLruEviction)
{
    Tensor source = makeTensor(4);
    MediaArtifactCache cache(/*capacityBytes=*/32, /*maxEntries=*/2);
    cache.insert(key(1), source, {}, nullptr).reset();
    cache.insert(key(2), source, {}, nullptr).reset();

    std::optional<MediaArtifactLease> pinned = cache.acquire(key(1));
    ASSERT_TRUE(pinned.has_value());
    cache.insert(key(3), source, {}, nullptr).reset();

    EXPECT_TRUE(cache.contains(key(1)));
    EXPECT_FALSE(cache.contains(key(2)));
    EXPECT_TRUE(cache.contains(key(3)));
}

TEST(MediaArtifactCacheTests, OversizeAndPinnedPressureSkipInsertionWithoutMutation)
{
    Tensor small = makeTensor(4);
    Tensor large = makeTensor(9);
    MediaArtifactCache cache(/*capacityBytes=*/32, /*maxEntries=*/2);
    std::optional<MediaArtifactLease> first = cache.insert(key(1), small, {}, nullptr);
    std::optional<MediaArtifactLease> second = cache.insert(key(2), small, {}, nullptr);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());

    EXPECT_FALSE(cache.insert(key(3), small, {}, nullptr).has_value());
    EXPECT_FALSE(cache.insert(key(4), large, {}, nullptr).has_value());
    EXPECT_EQ(cache.size(), 2U);
    EXPECT_EQ(cache.residentBytes(), 32U);
}

TEST(MediaArtifactCacheTests, InfeasibleInsertionDoesNotEvictUnpinnedEntries)
{
    Tensor small = makeTensor(4);
    Tensor fullCapacity = makeTensor(8);
    MediaArtifactCache cache(/*capacityBytes=*/32, /*maxEntries=*/2);
    cache.insert(key(1), small, {}, nullptr).reset();
    std::optional<MediaArtifactLease> pinned = cache.insert(key(2), small, {}, nullptr);
    ASSERT_TRUE(pinned.has_value());

    EXPECT_FALSE(cache.insert(key(3), fullCapacity, {}, nullptr).has_value());
    EXPECT_TRUE(cache.contains(key(1)));
    EXPECT_TRUE(cache.contains(key(2)));
    EXPECT_EQ(cache.size(), 2U);
    EXPECT_EQ(cache.residentBytes(), 32U);
}
