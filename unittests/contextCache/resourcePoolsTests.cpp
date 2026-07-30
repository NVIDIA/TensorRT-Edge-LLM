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

#include "runtime/state/contextCache/resourcePools.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace trt_edgellm::rt;

static_assert(std::is_same_v<decltype(std::declval<ResourceDemand const&>().get(ResourceType::kBaseKvPage)), int32_t>);
static_assert(noexcept(std::declval<ResourceDemand const&>().get(ResourceType::kBaseKvPage)));
static_assert(std::is_same_v<decltype(std::declval<ResourceDemand&>().get(ResourceType::kBaseKvPage)), int32_t&>);
static_assert(noexcept(std::declval<ResourceDemand&>().get(ResourceType::kBaseKvPage)));

TEST(ContextCacheResourcePoolsTests, NewAllocationStartsWithOneActiveReference)
{
    ResourcePools pools(ResourceDemand{2, 0, 1, 1});

    auto const baseResources = pools.allocate(ResourceDemand{2, 0, 0, 0});

    ASSERT_TRUE(baseResources.has_value());
    ASSERT_EQ(baseResources->size(), 2U);
    ResourceId const basePage0{ResourceType::kBaseKvPage, 0};
    ResourceId const basePage1{ResourceType::kBaseKvPage, 1};
    EXPECT_EQ(baseResources->at(0), basePage0);
    EXPECT_EQ(baseResources->at(1), basePage1);
    EXPECT_EQ(pools.activeRefCount(basePage0), 1);
    EXPECT_EQ(pools.cacheRefCount(basePage0), 0);
    EXPECT_EQ(pools.activeRefCount(basePage1), 1);
    EXPECT_EQ(pools.cacheRefCount(basePage1), 0);
    EXPECT_EQ(pools.freeCount(ResourceType::kBaseKvPage), 0);
    EXPECT_EQ(pools.capacity(ResourceType::kBaseKvPage), 2);

    auto const recurrentResources = pools.allocate(ResourceDemand{0, 0, 1, 0});
    ASSERT_TRUE(recurrentResources.has_value());
    ASSERT_EQ(recurrentResources->size(), 1U);
    ResourceId const recurrentSnapshot{ResourceType::kRecurrentSnapshot, 0};
    EXPECT_EQ(recurrentResources->front(), recurrentSnapshot);
    EXPECT_EQ(pools.activeRefCount(recurrentSnapshot), 1);
    EXPECT_EQ(pools.cacheRefCount(recurrentSnapshot), 0);

    auto const partialResources = pools.allocate(ResourceDemand{0, 0, 0, 1});
    ASSERT_TRUE(partialResources.has_value());
    ASSERT_EQ(partialResources->size(), 1U);
    ResourceId const partialSnapshot{ResourceType::kPartialKvSnapshot, 0};
    EXPECT_EQ(partialResources->front(), partialSnapshot);
    EXPECT_EQ(pools.activeRefCount(partialSnapshot), 1);
    EXPECT_EQ(pools.cacheRefCount(partialSnapshot), 0);
}

TEST(ContextCacheResourcePoolsTests, ResourceBecomesFreeOnlyAfterBothReferencesReachZero)
{
    ResourcePools activeFirstPools(ResourceDemand{1, 0, 0, 0});
    auto const activeFirstResources = activeFirstPools.allocate(ResourceDemand{1, 0, 0, 0});
    ASSERT_TRUE(activeFirstResources.has_value());
    ResourceId const activeFirstResource = activeFirstResources->front();

    activeFirstPools.addCacheRef(activeFirstResource);
    activeFirstPools.releaseActiveRef(activeFirstResource);

    EXPECT_EQ(activeFirstPools.activeRefCount(activeFirstResource), 0);
    EXPECT_EQ(activeFirstPools.cacheRefCount(activeFirstResource), 1);
    EXPECT_EQ(activeFirstPools.freeCount(ResourceType::kBaseKvPage), 0);
    EXPECT_FALSE(activeFirstPools.allocate(ResourceDemand{1, 0, 0, 0}).has_value());

    activeFirstPools.releaseCacheRef(activeFirstResource);

    EXPECT_EQ(activeFirstPools.activeRefCount(activeFirstResource), 0);
    EXPECT_EQ(activeFirstPools.cacheRefCount(activeFirstResource), 0);
    EXPECT_EQ(activeFirstPools.freeCount(ResourceType::kBaseKvPage), 1);

    ResourcePools cacheFirstPools(ResourceDemand{1, 0, 0, 0});
    auto const cacheFirstResources = cacheFirstPools.allocate(ResourceDemand{1, 0, 0, 0});
    ASSERT_TRUE(cacheFirstResources.has_value());
    ResourceId const cacheFirstResource = cacheFirstResources->front();

    cacheFirstPools.addCacheRef(cacheFirstResource);
    cacheFirstPools.addActiveRef(cacheFirstResource);
    EXPECT_EQ(cacheFirstPools.activeRefCount(cacheFirstResource), 2);
    EXPECT_EQ(cacheFirstPools.cacheRefCount(cacheFirstResource), 1);

    cacheFirstPools.releaseCacheRef(cacheFirstResource);
    EXPECT_EQ(cacheFirstPools.cacheRefCount(cacheFirstResource), 0);
    EXPECT_EQ(cacheFirstPools.freeCount(ResourceType::kBaseKvPage), 0);
    EXPECT_FALSE(cacheFirstPools.allocate(ResourceDemand{1, 0, 0, 0}).has_value());

    cacheFirstPools.releaseActiveRef(cacheFirstResource);
    EXPECT_EQ(cacheFirstPools.activeRefCount(cacheFirstResource), 1);
    EXPECT_EQ(cacheFirstPools.freeCount(ResourceType::kBaseKvPage), 0);
    EXPECT_FALSE(cacheFirstPools.allocate(ResourceDemand{1, 0, 0, 0}).has_value());

    cacheFirstPools.releaseActiveRef(cacheFirstResource);
    EXPECT_EQ(cacheFirstPools.activeRefCount(cacheFirstResource), 0);
    EXPECT_EQ(cacheFirstPools.freeCount(ResourceType::kBaseKvPage), 1);

    ResourcePools fifoPools(ResourceDemand{3, 0, 0, 0});
    auto const initialResources = fifoPools.allocate(ResourceDemand{3, 0, 0, 0});
    ASSERT_TRUE(initialResources.has_value());
    ASSERT_EQ(initialResources->size(), 3U);
    ResourceId const page0{ResourceType::kBaseKvPage, 0};
    ResourceId const page1{ResourceType::kBaseKvPage, 1};
    ResourceId const page2{ResourceType::kBaseKvPage, 2};
    EXPECT_EQ(*initialResources, (std::vector<ResourceId>{page0, page1, page2}));

    fifoPools.releaseActiveRef(page1);
    fifoPools.releaseActiveRef(page2);
    auto const firstReusedResource = fifoPools.allocate(ResourceDemand{1, 0, 0, 0});
    ASSERT_TRUE(firstReusedResource.has_value());
    ASSERT_EQ(firstReusedResource->size(), 1U);
    EXPECT_EQ(firstReusedResource->front(), page1);

    fifoPools.releaseActiveRef(page0);
    fifoPools.releaseActiveRef(page1);
    ResourcePools fifoSnapshot = fifoPools;

    auto const wrappedResources = fifoPools.allocate(ResourceDemand{3, 0, 0, 0});
    auto const snapshotResources = fifoSnapshot.allocate(ResourceDemand{3, 0, 0, 0});
    std::vector<ResourceId> const expectedWrappedOrder{page2, page0, page1};
    ASSERT_TRUE(wrappedResources.has_value());
    ASSERT_TRUE(snapshotResources.has_value());
    EXPECT_EQ(*wrappedResources, expectedWrappedOrder);
    EXPECT_EQ(*snapshotResources, expectedWrappedOrder);
}

TEST(ContextCacheResourcePoolsTests, MultiPoolAllocationIsAllOrNothing)
{
    ResourcePools pools(ResourceDemand{2, 1, 0, 0});
    int32_t const baseFreeBefore = pools.freeCount(ResourceType::kBaseKvPage);
    int32_t const draftFreeBefore = pools.freeCount(ResourceType::kDraftKvPage);

    auto const resources = pools.allocate(ResourceDemand{1, 2, 0, 0});

    EXPECT_FALSE(resources.has_value());
    EXPECT_EQ(pools.freeCount(ResourceType::kBaseKvPage), baseFreeBefore);
    EXPECT_EQ(pools.freeCount(ResourceType::kDraftKvPage), draftFreeBefore);
    EXPECT_THROW((void) ResourcePools(ResourceDemand{-1, 0, 0, 0}), std::runtime_error);
    EXPECT_THROW((void) pools.allocate(ResourceDemand{0, -1, 0, 0}), std::runtime_error);
    EXPECT_FALSE(pools.canAllocate(ResourceDemand{0, -1, 0, 0}));

    ResourcePools preparedPools(ResourceDemand{2, 1, 1, 1});
    std::vector<ResourceId> preparedResources(4);
    size_t const preparedOutputCapacity = preparedResources.capacity();
    EXPECT_TRUE(preparedPools.allocateInto(ResourceDemand{1, 1, 1, 1}, preparedResources));
    std::vector<ResourceId> const expectedPreparedResources{{ResourceType::kBaseKvPage, 0},
        {ResourceType::kDraftKvPage, 0}, {ResourceType::kRecurrentSnapshot, 0}, {ResourceType::kPartialKvSnapshot, 0}};
    EXPECT_EQ(preparedResources, expectedPreparedResources);
    EXPECT_EQ(preparedResources.capacity(), preparedOutputCapacity);
    EXPECT_EQ(preparedPools.capacity(ResourceType::kBaseKvPage), 2);
    EXPECT_EQ(preparedPools.capacity(ResourceType::kDraftKvPage), 1);
    EXPECT_EQ(preparedPools.capacity(ResourceType::kRecurrentSnapshot), 1);
    EXPECT_EQ(preparedPools.capacity(ResourceType::kPartialKvSnapshot), 1);
    EXPECT_EQ(preparedPools.freeCount(ResourceType::kBaseKvPage), 1);
    EXPECT_EQ(preparedPools.freeCount(ResourceType::kDraftKvPage), 0);
    EXPECT_EQ(preparedPools.freeCount(ResourceType::kRecurrentSnapshot), 0);
    EXPECT_EQ(preparedPools.freeCount(ResourceType::kPartialKvSnapshot), 0);
    for (ResourceId const& resource : preparedResources)
    {
        EXPECT_EQ(preparedPools.activeRefCount(resource), 1);
        EXPECT_EQ(preparedPools.cacheRefCount(resource), 0);
    }

    ResourceId const sentinel{ResourceType::kPartialKvSnapshot, -7};
    std::vector<ResourceId> insufficientResources(2, sentinel);
    EXPECT_FALSE(preparedPools.allocateInto(ResourceDemand{2, 0, 0, 0}, insufficientResources));
    EXPECT_EQ(insufficientResources, std::vector<ResourceId>(2, sentinel));
    EXPECT_EQ(preparedPools.freeCount(ResourceType::kBaseKvPage), 1);
    EXPECT_EQ(preparedPools.activeRefCount(ResourceId{ResourceType::kBaseKvPage, 0}), 1);
    EXPECT_EQ(preparedPools.activeRefCount(ResourceId{ResourceType::kBaseKvPage, 1}), 0);

    std::vector<ResourceId> wrongSize(1);
    EXPECT_THROW((void) preparedPools.allocateInto(ResourceDemand{}, wrongSize), std::runtime_error);
    std::vector<ResourceId> emptyResources;
    EXPECT_THROW((void) preparedPools.allocateInto(ResourceDemand{-1, 0, 0, 0}, emptyResources), std::runtime_error);
    EXPECT_EQ(preparedPools.freeCount(ResourceType::kBaseKvPage), 1);
    EXPECT_EQ(preparedPools.freeCount(ResourceType::kDraftKvPage), 0);
    EXPECT_EQ(preparedPools.activeRefCount(ResourceId{ResourceType::kBaseKvPage, 0}), 1);
    EXPECT_EQ(preparedPools.activeRefCount(ResourceId{ResourceType::kBaseKvPage, 1}), 0);

    ResourcePools emptyPools(ResourceDemand{});
    EXPECT_TRUE(emptyPools.allocateInto(ResourceDemand{}, emptyResources));
    EXPECT_EQ(emptyPools.capacity(ResourceType::kBaseKvPage), 0);
    EXPECT_EQ(emptyPools.freeCount(ResourceType::kBaseKvPage), 0);
}

TEST(ContextCacheResourcePoolsTests, BaseAndDraftIdsArePoolLocal)
{
    ResourcePools pools(ResourceDemand{2, 1, 0, 0});
    auto const baseResources = pools.allocate(ResourceDemand{2, 0, 0, 0});
    auto const draftResources = pools.allocate(ResourceDemand{0, 1, 0, 0});

    ASSERT_TRUE(baseResources.has_value());
    ASSERT_TRUE(draftResources.has_value());
    ASSERT_EQ(baseResources->size(), 2U);
    ResourceId const base0 = baseResources->at(0);
    ResourceId const base1 = baseResources->at(1);
    ResourceId const draft = draftResources->front();
    EXPECT_EQ(base0.index, 0);
    EXPECT_EQ(base1.index, 1);
    EXPECT_EQ(draft.index, 0);
    EXPECT_FALSE(base0 == draft);
    EXPECT_FALSE(base0 == base1);

    std::unordered_set<ResourceId> const resources{base0, base1, draft};
    EXPECT_EQ(resources.size(), 3U);
}

TEST(ContextCacheResourcePoolsTests, ReferenceUnderflowThrows)
{
    ResourcePools pools(ResourceDemand{1, 0, 0, 0});
    auto const resources = pools.allocate(ResourceDemand{1, 0, 0, 0});
    ASSERT_TRUE(resources.has_value());
    ResourceId const resource = resources->front();

    EXPECT_THROW(pools.releaseCacheRef(resource), std::runtime_error);
    EXPECT_EQ(pools.activeRefCount(resource), 1);
    EXPECT_EQ(pools.cacheRefCount(resource), 0);
    EXPECT_EQ(pools.freeCount(ResourceType::kBaseKvPage), 0);

    pools.releaseActiveRef(resource);
    EXPECT_EQ(pools.activeRefCount(resource), 0);
    EXPECT_EQ(pools.cacheRefCount(resource), 0);
    EXPECT_EQ(pools.freeCount(ResourceType::kBaseKvPage), 1);

    EXPECT_THROW(pools.releaseActiveRef(resource), std::runtime_error);
    EXPECT_EQ(pools.activeRefCount(resource), 0);
    EXPECT_EQ(pools.cacheRefCount(resource), 0);
    EXPECT_EQ(pools.freeCount(ResourceType::kBaseKvPage), 1);

    EXPECT_THROW(pools.releaseActiveRef(ResourceId{ResourceType::kBaseKvPage, 1}), std::runtime_error);
    EXPECT_EQ(pools.freeCount(ResourceType::kBaseKvPage), 1);
}
