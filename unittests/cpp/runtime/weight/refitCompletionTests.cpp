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

#include "runtime/weight/refitWeights.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace trt_edgellm::rt
{
namespace
{

struct TrackedOwner
{
    std::vector<std::string>& events;
    char const* name;

    ~TrackedOwner()
    {
        events.emplace_back(name);
    }
};

TEST(RefitCompletionTest, FailedEnqueueDrainsBeforeOwnersUnwind)
{
    // Given live refitter and source owners, when enqueue rejects refit, then draining precedes failure and
    // destruction.
    std::vector<std::string> events;
    auto refit = [&] {
        TrackedOwner source{events, "source destroyed"};
        TrackedOwner refitter{events, "refitter destroyed"};
        detail::completeRefit(
            [&] {
                events.emplace_back("enqueue");
                return false;
            },
            [&] {
                events.emplace_back("drain");
                return cudaSuccess;
            });
        events.emplace_back("returned");
    };
    EXPECT_THAT(refit,
        testing::ThrowsMessage<std::runtime_error>(testing::HasSubstr("TensorRT weightless engine refit failed")));
    EXPECT_THAT(events, testing::ElementsAre("enqueue", "drain", "refitter destroyed", "source destroyed"));
}

TEST(RefitCompletionTest, SuccessfulEnqueueDrainsBeforeReturning)
{
    // Given live refitter and source owners, when enqueue succeeds, then completion is drained before returning.
    std::vector<std::string> events;
    {
        TrackedOwner source{events, "source destroyed"};
        TrackedOwner refitter{events, "refitter destroyed"};
        detail::completeRefit(
            [&] {
                events.emplace_back("enqueue");
                return true;
            },
            [&] {
                events.emplace_back("drain");
                return cudaSuccess;
            });
        events.emplace_back("returned");
    }
    EXPECT_THAT(events, testing::ElementsAre("enqueue", "drain", "returned", "refitter destroyed", "source destroyed"));
}

} // namespace
} // namespace trt_edgellm::rt
