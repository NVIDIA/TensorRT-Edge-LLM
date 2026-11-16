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

#include "runtime/multiStage/stage.h"

#include <gtest/gtest.h>

#include <cstdint>

using namespace trt_edgellm::rt::multiStage;

// Given a stage that registered its wake before anyone cancelled,
// When the token is requested,
// Then the wake runs -- this is the only way a stage blocked inside its own decode loop
// learns about a cancel, so losing it would hang the request until it finished naturally.
TEST(AbandonTokenTest, AWakeRegisteredBeforeTheCancelRuns)
{
    AbandonToken token;
    int32_t woken = 0;
    token.onRequest([&woken]() { ++woken; });

    EXPECT_EQ(woken, 0);
    token.request();
    EXPECT_EQ(woken, 1);
}

// Given a cancel that arrived before the stage got as far as registering,
// When the stage registers its wake,
// Then the wake runs immediately. prepare() and cancel() are on different threads, so a
// caller that cancels during admission must not be silently dropped.
TEST(AbandonTokenTest, AWakeRegisteredAfterTheCancelStillRuns)
{
    AbandonToken token;
    token.request();

    int32_t woken = 0;
    token.onRequest([&woken]() { ++woken; });
    EXPECT_EQ(woken, 1);
}

// Given two stages that both registered a wake,
// When the token is requested,
// Then both run: cancellation reaches the whole topology, not whichever stage registered last.
TEST(AbandonTokenTest, EveryRegisteredWakeRuns)
{
    AbandonToken token;
    int32_t first = 0;
    int32_t second = 0;
    token.onRequest([&first]() { ++first; });
    token.onRequest([&second]() { ++second; });

    token.request();
    EXPECT_EQ(first, 1);
    EXPECT_EQ(second, 1);
}

// Given a token that was cancelled and then reset,
// When it is requested again,
// Then the stale wakes do not run -- a reset token belongs to a request that is over, and
// waking its channels would cancel whatever holds them now.
TEST(AbandonTokenTest, ResetDropsTheRegisteredWakes)
{
    AbandonToken token;
    int32_t woken = 0;
    token.onRequest([&woken]() { ++woken; });
    token.request();
    EXPECT_EQ(woken, 1);

    token.reset();
    EXPECT_FALSE(token.requested());
    token.request();
    EXPECT_EQ(woken, 1);
}
