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

#include "runtime/multiDevice/collectiveGroup.h"

#include <gtest/gtest.h>
#include <vector>

using namespace trt_edgellm::rt;

namespace
{

ParallelGroupConfig makeGroupConfig(int32_t size)
{
    ParallelGroupConfig config{};
    config.type = ParallelType::kTensor;
    config.size = size;
    config.rank = 0;
    config.globalRank = 0;
    config.localDevice = 0;
    return config;
}

} // namespace

TEST(CollectiveGroupTest, SizeOneCollectivesAreNoOps)
{
    CollectiveGroup const group(makeGroupConfig(1), {});
    int32_t value = 1;
    int32_t output = 0;

    EXPECT_TRUE(group.broadcastInt32(0, nullptr, 0, 0, nullptr));
    EXPECT_TRUE(group.broadcastInt32(0, &value, 1, 0, nullptr));
    EXPECT_TRUE(group.allGather(0, nullptr, nullptr, 0, CollectiveDataType::kInt32, nullptr));
    EXPECT_TRUE(group.allGather(0, &value, &output, 1, CollectiveDataType::kInt32, nullptr));
    EXPECT_TRUE(group.allReduceSum(0, nullptr, nullptr, 0, CollectiveDataType::kInt32, nullptr));
    EXPECT_TRUE(group.allReduceSum(0, &value, &output, 1, CollectiveDataType::kInt32, nullptr));
}

TEST(CollectiveGroupTest, BackendHandleBoundsChecks)
{
    int32_t marker = 0;
    void* handle0 = &marker;
    CollectiveGroup const group(makeGroupConfig(2), {handle0, nullptr});

    EXPECT_EQ(group.backendHandle(-1), nullptr);
    EXPECT_EQ(group.backendHandle(0), handle0);
    EXPECT_EQ(group.backendHandle(1), nullptr);
    EXPECT_EQ(group.backendHandle(2), nullptr);
}

TEST(CollectiveGroupTest, RejectsInvalidBroadcastInputsForMultiRank)
{
    CollectiveGroup const group(makeGroupConfig(2), {nullptr, nullptr});
    int32_t value = 1;

    EXPECT_FALSE(group.broadcastInt32(0, nullptr, 1, 0, nullptr));
    EXPECT_FALSE(group.broadcastInt32(0, &value, 0, 0, nullptr));
}

TEST(CollectiveGroupTest, RejectsInvalidAllGatherInputsForMultiRank)
{
    CollectiveGroup const group(makeGroupConfig(2), {nullptr, nullptr});
    int32_t value = 1;

    EXPECT_FALSE(group.allGather(0, nullptr, &value, 1, CollectiveDataType::kInt32, nullptr));
    EXPECT_FALSE(group.allGather(0, &value, nullptr, 1, CollectiveDataType::kInt32, nullptr));
    EXPECT_FALSE(group.allGather(0, &value, &value, 0, CollectiveDataType::kInt32, nullptr));
}

TEST(CollectiveGroupTest, RejectsInvalidAllReduceInputsForMultiRank)
{
    CollectiveGroup const group(makeGroupConfig(2), {nullptr, nullptr});
    int32_t value = 1;

    EXPECT_FALSE(group.allReduceSum(0, nullptr, &value, 1, CollectiveDataType::kInt32, nullptr));
    EXPECT_FALSE(group.allReduceSum(0, &value, nullptr, 1, CollectiveDataType::kInt32, nullptr));
    EXPECT_FALSE(group.allReduceSum(0, &value, &value, 0, CollectiveDataType::kInt32, nullptr));
}

TEST(CollectiveGroupTest, RejectsMissingBackendHandleForMultiRank)
{
    CollectiveGroup const group(makeGroupConfig(2), {nullptr, nullptr});
    int32_t value = 1;
    int32_t output = 0;

    EXPECT_FALSE(group.broadcastInt32(0, &value, 1, 0, nullptr));
    EXPECT_FALSE(group.allGather(0, &value, &output, 1, CollectiveDataType::kInt32, nullptr));
    EXPECT_FALSE(group.allReduceSum(0, &value, &output, 1, CollectiveDataType::kInt32, nullptr));
}
