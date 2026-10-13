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

#include "runtime/state/residentSlotPool.h"

#include <gtest/gtest.h>

namespace trt_edgellm::rt
{
namespace
{

TEST(ResidentSlotPoolTest, MiddleRetirementReusesOnlyReleasedSlotWithNewEpoch)
{
    ResidentSlotPool pool(3);
    ResidentRef const a = *pool.acquire();
    ResidentRef const b = *pool.acquire();
    ResidentRef const c = *pool.acquire();

    EXPECT_EQ(a, (ResidentRef{0, 1}));
    EXPECT_EQ(b, (ResidentRef{1, 1}));
    EXPECT_EQ(c, (ResidentRef{2, 1}));
    EXPECT_FALSE(pool.acquire().has_value());

    EXPECT_TRUE(pool.release(b));
    EXPECT_TRUE(pool.contains(a));
    EXPECT_TRUE(pool.contains(c));
    EXPECT_FALSE(pool.contains(b));

    ResidentRef const d = *pool.acquire();
    EXPECT_EQ(d.slot, b.slot);
    EXPECT_GT(d.epoch, b.epoch);
    EXPECT_TRUE(pool.contains(d));
    EXPECT_FALSE(pool.release(b));
    EXPECT_TRUE(pool.contains(d));
}

TEST(ResidentSlotPoolTest, ResetRejectsInvalidCapacityAndRestoresBoundedIdentityOrder)
{
    ResidentSlotPool pool;
    EXPECT_THROW(pool.reset(-1), std::runtime_error);

    pool.reset(2);
    EXPECT_EQ(pool.capacity(), 2);
    EXPECT_EQ(pool.available(), 2);
    EXPECT_EQ(*pool.acquire(), (ResidentRef{0, 1}));
    EXPECT_EQ(*pool.acquire(), (ResidentRef{1, 1}));
    EXPECT_EQ(pool.available(), 0);
    EXPECT_FALSE(pool.acquire());
}

} // namespace
} // namespace trt_edgellm::rt
