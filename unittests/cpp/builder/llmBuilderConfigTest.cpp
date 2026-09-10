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

#include "common/pagedKvTypes.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

using namespace trt_edgellm;
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

TEST(LLMBuilderConfigTest, DefaultPoolPagesResolveToMinimumActivePages)
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

TEST(LLMBuilderConfigTest, PoolPagesBelowMinimumActivePagesAreRejected)
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

TEST(LLMBuilderConfigTest, PoolPagesRejectDerivedVIdOverflow)
{
    LLMBuilderConfig config = makeConfig();
    config.maxKVPoolPages = rt::kMAX_KV_POOL_PAGES + 1;

    EXPECT_THROW(config.resolvedKVPoolPages(), std::runtime_error);
}

TEST(LLMBuilderConfigTest, KVCapacityRejectsPageAlignmentOverflow)
{
    LLMBuilderConfig config;
    config.maxBatchSize = 1;
    config.maxKVCacheCapacity = static_cast<int64_t>(rt::kMAX_KV_CACHE_CAPACITY) + 1;

    EXPECT_THROW(config.resolvedKVPoolPages(), std::runtime_error);
}

TEST(LLMBuilderConfigTest, MinimumActivePagesRejectNarrowingOverflow)
{
    LLMBuilderConfig config;
    config.maxBatchSize = std::numeric_limits<int32_t>::max();
    config.maxKVCacheCapacity = rt::kMAX_KV_CACHE_CAPACITY;

    EXPECT_THROW(config.resolvedKVPoolPages(), std::runtime_error);
}
