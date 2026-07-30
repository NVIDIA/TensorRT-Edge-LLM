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

#include "builder/llmBuilder.h"

#include <gtest/gtest.h>
#include <stdexcept>

using namespace trt_edgellm::builder;

namespace
{

LLMBuilderConfig makeConfig()
{
    LLMBuilderConfig config;
    config.maxBatchSize = 2;
    config.maxKVCacheCapacity = 256;
    return config;
}

} // namespace

TEST(LLMBuilderConfigTest, DefaultPoolPagesResolveToActiveCapacityFloor)
{
    LLMBuilderConfig const config = makeConfig();

    EXPECT_EQ(config.resolvedKVPoolPages(), 4);
    EXPECT_EQ(config.toJson().at("max_kv_pool_pages"), 4);
}

TEST(LLMBuilderConfigTest, ExplicitPoolPagesAreSerialized)
{
    LLMBuilderConfig config = makeConfig();
    config.maxKVPoolPages = 9;

    EXPECT_EQ(config.resolvedKVPoolPages(), 9);
    EXPECT_EQ(config.toJson().at("max_kv_pool_pages"), 9);
}

TEST(LLMBuilderConfigTest, PoolPagesBelowActiveCapacityFloorAreRejected)
{
    LLMBuilderConfig config = makeConfig();
    config.maxKVPoolPages = 3;

    EXPECT_THROW(config.resolvedKVPoolPages(), std::runtime_error);
    EXPECT_THROW(config.toJson(), std::runtime_error);
}

TEST(LLMBuilderConfigTest, PoolPagesRoundTripThroughJson)
{
    LLMBuilderConfig config = makeConfig();
    config.maxKVPoolPages = 9;

    LLMBuilderConfig const parsed = LLMBuilderConfig::fromJson(config.toJson());

    EXPECT_EQ(parsed.maxKVPoolPages, 9);
    EXPECT_EQ(parsed.resolvedKVPoolPages(), 9);
}
