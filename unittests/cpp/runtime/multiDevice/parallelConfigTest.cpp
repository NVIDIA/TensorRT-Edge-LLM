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

#include "runtime/multiDevice/parallelConfig.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace trt_edgellm::rt;

TEST(ParallelConfigTest, ComputesWorldSizeFromActiveDimensions)
{
    ParallelConfig config{};
    config.tensorParallelSize = 2;

    EXPECT_EQ(parallelGroupSize(config, ParallelType::kTensor), 2);
    EXPECT_EQ(parallelWorldSize(config), 2);
}

TEST(ParallelConfigTest, ClampsInvalidGroupSizesToOne)
{
    ParallelConfig config{};
    config.tensorParallelSize = 0;

    EXPECT_EQ(parallelGroupSize(config, ParallelType::kTensor), 1);
    EXPECT_EQ(parallelWorldSize(config), 1);
}

TEST(ParallelConfigTest, MapsGlobalRankToTensorRank)
{
    ParallelConfig config{};
    config.tensorParallelSize = 2;

    int32_t const globalRank = 3;
    EXPECT_EQ(parallelGroupRank(config, ParallelType::kTensor, globalRank), 1);
}

TEST(ParallelConfigTest, BuildsRankMappingWithTensorCoordinates)
{
    ParallelConfig config{};
    config.tensorParallelSize = 2;

    ParallelMapping const mapping = makeParallelMapping(config, /*globalRank=*/3, /*localDevice=*/6);

    EXPECT_EQ(mapping.worldSize, 2);
    EXPECT_EQ(mapping.globalRank, 3);
    EXPECT_EQ(mapping.localDevice, 6);
    EXPECT_EQ(mapping.tensorParallelSize, 2);
    EXPECT_EQ(mapping.tensorParallelRank, 1);
    EXPECT_TRUE(mapping.isParallel());
}

TEST(ParallelConfigTest, BuildsSingleDeviceMappingByDefault)
{
    ParallelMapping const mapping = makeParallelMapping(ParallelConfig{}, /*globalRank=*/0, /*localDevice=*/0);

    EXPECT_EQ(mapping.worldSize, 1);
    EXPECT_EQ(mapping.tensorParallelSize, 1);
    EXPECT_FALSE(mapping.isParallel());
}


TEST(ParallelConfigTest, BuildsNamedGroupConfig)
{
    ParallelConfig config{};
    config.tensorParallelSize = 2;

    ParallelGroupConfig const group = makeParallelGroupConfig(config, ParallelType::kTensor, 1, 7);

    EXPECT_EQ(group.type, ParallelType::kTensor);
    EXPECT_EQ(group.size, 2);
    EXPECT_EQ(group.rank, 1);
    EXPECT_EQ(group.globalRank, 1);
    EXPECT_EQ(group.localDevice, 7);
    EXPECT_STREQ(group.typeName(), "tensor");
    EXPECT_NE(group.toString().find("type=tensor"), std::string::npos);
    EXPECT_NE(group.toString().find("localDevice=7"), std::string::npos);
}

TEST(ParallelConfigTest, ReturnsOnlyActiveNonSingletonGroups)
{
    ParallelConfig config{};
    config.tensorParallelSize = 2;

    std::vector<ParallelGroupConfig> const groups = activeParallelGroups(config, 3, 0);

    ASSERT_EQ(groups.size(), 1);
    EXPECT_EQ(groups[0].type, ParallelType::kTensor);
    EXPECT_STREQ(groups[0].typeName(), "tensor");
    EXPECT_EQ(groups[0].rank, 1);
}

TEST(ParallelConfigTest, SingletonTensorGroupIsNotActive)
{
    std::vector<ParallelGroupConfig> const groups = activeParallelGroups(ParallelConfig{}, 0, 0);
    EXPECT_TRUE(groups.empty());
}

TEST(ParallelConfigTest, StringifiesKnownEnums)
{
    EXPECT_STREQ(parallelTypeName(ParallelType::kTensor), "tensor");
    EXPECT_STREQ(parallelLaunchModeName(ParallelLaunchMode::kThread), "thread");
    EXPECT_STREQ(parallelLaunchModeName(ParallelLaunchMode::kMpi), "mpi");
}

TEST(ParallelConfigTest, FormatsParallelConfigAndMapping)
{
    ParallelConfig config{};
    config.tensorParallelSize = 2;
    config.launchMode = ParallelLaunchMode::kMpi;
    config.devices = {2, 3};

    ParallelMapping const mapping = makeParallelMapping(config, /*globalRank=*/1, /*localDevice=*/3);

    EXPECT_NE(config.toString().find("tensorParallelSize=2"), std::string::npos);
    EXPECT_NE(config.toString().find("launchMode=mpi"), std::string::npos);
    EXPECT_NE(config.toString().find("devices=[2, 3]"), std::string::npos);
    EXPECT_NE(mapping.toString().find("worldSize=2"), std::string::npos);
    EXPECT_NE(mapping.toString().find("tensorParallelRank=1"), std::string::npos);
}

TEST(ParallelConfigTest, InlineSingleRankOnlyForThreadedWorldSizeOne)
{
    using trt_edgellm::rt::isInlineSingleRank;
    using trt_edgellm::rt::ParallelLaunchMode;

    // Threaded, world size 1, single local rank -> inline.
    EXPECT_TRUE(isInlineSingleRank(ParallelLaunchMode::kThread, /*worldSize=*/1, /*localRankCount=*/1));

    // World size > 1 is never inline, regardless of local rank count.
    EXPECT_FALSE(isInlineSingleRank(ParallelLaunchMode::kThread, /*worldSize=*/2, /*localRankCount=*/1));

    // MPI launch is never inline (each process is its own rank already).
    EXPECT_FALSE(isInlineSingleRank(ParallelLaunchMode::kMpi, /*worldSize=*/1, /*localRankCount=*/1));

    // Zero or multiple local ranks in one process are not the inline case.
    EXPECT_FALSE(isInlineSingleRank(ParallelLaunchMode::kThread, /*worldSize=*/1, /*localRankCount=*/0));
    EXPECT_FALSE(isInlineSingleRank(ParallelLaunchMode::kThread, /*worldSize=*/1, /*localRankCount=*/2));
}

TEST(ParallelConfigTest, IdentifiesFullLocalParallelGroups)
{
    EXPECT_TRUE(isFullLocalParallelGroup(/*groupSize=*/1, {0}));
    EXPECT_TRUE(isFullLocalParallelGroup(/*groupSize=*/2, {0, 1}));
    EXPECT_FALSE(isFullLocalParallelGroup(/*groupSize=*/0, {}));
    EXPECT_FALSE(isFullLocalParallelGroup(/*groupSize=*/2, {0}));
    EXPECT_FALSE(isFullLocalParallelGroup(/*groupSize=*/2, {1, 0}));
    EXPECT_FALSE(isFullLocalParallelGroup(/*groupSize=*/2, {0, 2}));
}
