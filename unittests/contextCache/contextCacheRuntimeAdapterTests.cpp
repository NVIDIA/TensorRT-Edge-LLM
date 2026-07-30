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

#include "runtime/config/llmEngineConfig.h"
#include "runtime/state/contextCache/contextCacheRuntimeAdapter.h"

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

using namespace trt_edgellm::rt;

namespace
{

constexpr int32_t kPAGE_SIZE = 4;
constexpr CacheDomainId kDOMAIN{11, 13};
constexpr BlockHash kHASH_A{17, 19};
constexpr BlockHash kHASH_B{23, 29};
constexpr BlockHash kHASH_C{31, 37};
constexpr DraftEngineSignature kDRAFT_SIGNATURE{41, 43};

static_assert(
    std::is_same_v<decltype(std::declval<ContextCacheRuntimeAdapter&>().manager()), ContextCacheManager const&>);

CacheRequestLease takeLease(RuntimeCacheAcquireResult& result)
{
    if (!result.lease.has_value())
    {
        throw std::runtime_error("Test acquisition did not return a lease");
    }
    return std::move(*result.lease);
}

} // namespace

TEST(ContextCacheRuntimeAdapterTests, HitCapacityFailureFallsBackToForcedCold)
{
    ContextCacheRuntimeAdapter adapter(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, /*maxRecords=*/1);

    RuntimeCacheAcquireResult first = adapter.acquireVanilla(kDOMAIN, {}, kPAGE_SIZE);
    CacheRequestLease firstLease = takeLease(first);
    ASSERT_EQ(firstLease.basePages(), std::vector<PageId>{0});
    ASSERT_EQ(adapter
                  .publish(firstLease,
                      PublishRequest{{kHASH_A}, kPAGE_SIZE, PublicationPoint::kPrefillEnd,
                          CommitPolicy::kIncludingGeneratedTokens})
                  .status,
        PublishStatus::kPublished);
    firstLease.release();

    RuntimeCacheAcquireResult second = adapter.acquireVanilla(kDOMAIN, {kHASH_A}, kPAGE_SIZE + 1);
    ASSERT_EQ(second.status, AcquireStatus::kInsufficientCapacity);
    EXPECT_TRUE(second.forcedCold);
    EXPECT_FALSE(second.lease.has_value());
    EXPECT_EQ(adapter.manager().records().lruToMru().size(), 1U);
    EXPECT_EQ(adapter.manager().baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}), std::optional<PageId>{0});
}

TEST(ContextCacheRuntimeAdapterTests, RewoundFullHitCapacityFailureFallsBackToForcedCold)
{
    ContextCacheRuntimeAdapter adapter(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, /*maxRecords=*/1);

    RuntimeCacheAcquireResult first = adapter.acquireVanilla(kDOMAIN, {}, kPAGE_SIZE);
    CacheRequestLease firstLease = takeLease(first);
    ASSERT_EQ(adapter
                  .publish(firstLease,
                      PublishRequest{{kHASH_A}, kPAGE_SIZE, PublicationPoint::kPrefillEnd,
                          CommitPolicy::kIncludingGeneratedTokens})
                  .status,
        PublishStatus::kPublished);

    RuntimeCacheAcquireResult second = adapter.acquireVanilla(kDOMAIN, {kHASH_A}, kPAGE_SIZE);
    EXPECT_EQ(second.status, AcquireStatus::kInsufficientCapacity);
    EXPECT_TRUE(second.forcedCold);
    EXPECT_FALSE(second.lease.has_value());
}

TEST(ContextCacheRuntimeAdapterTests, HybridAcquisitionHashesStoredExactCandidateLengths)
{
    constexpr RecurrentStateSchemaId kSCHEMA{23, 29};
    ContextCacheRuntimeAdapter adapter(kPAGE_SIZE, ResourceDemand{4, 0, 2, 2}, /*maxRecords=*/2);
    std::vector<int32_t> const prefix{1, 2, 3, 4, 5, 6};

    RuntimeCacheAcquireResult cold = adapter.acquireHybrid(kDOMAIN, prefix, {}, kSCHEMA, /*hasAttention=*/true);
    CacheRequestLease coldLease = takeLease(cold);
    std::optional<HybridSnapshotReservation> const snapshots = adapter.reserveHybridSnapshots(coldLease, true);
    ASSERT_TRUE(snapshots.has_value());
    BlockHash const fullHash = hashBlock(kCHAIN_ROOT, prefix.data(), kPAGE_SIZE);
    BlockHash const exactDigest = hashExactPrefix(prefix.data(), prefix.size(), kPAGE_SIZE);
    ASSERT_EQ(adapter
                  .publishHybrid(coldLease,
                      HybridPublishRequest{{fullHash}, HybridCheckpointKey{kDOMAIN, exactDigest, 6, kSCHEMA},
                          PublicationPoint::kPrefillEnd, CommitPolicy::kIncludingGeneratedTokens, *snapshots})
                  .status,
        PublishStatus::kPublished);
    coldLease.release();

    std::vector<int32_t> const extension{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    RuntimeCacheAcquireResult hit = adapter.acquireHybrid(kDOMAIN, extension, {}, kSCHEMA, /*hasAttention=*/true);
    ASSERT_EQ(hit.status, AcquireStatus::kAcquired);
    CacheRequestLease hitLease = takeLease(hit);
    EXPECT_EQ(hitLease.reuseTokenLength(), 6);
    EXPECT_TRUE(hitLease.recurrentSnapshotSlot().has_value());
    EXPECT_TRUE(hitLease.partialKvSnapshotSlot().has_value());
}

TEST(ContextCacheRuntimeAdapterTests, HybridMtpAcquisitionIsSuccessorAgnostic)
{
    constexpr RecurrentStateSchemaId kSCHEMA{31, 37};
    constexpr int32_t kPUBLISHED_SUCCESSOR = 7;
    ContextCacheRuntimeAdapter adapter(kPAGE_SIZE, ResourceDemand{5, 5, 2, 2}, /*maxRecords=*/2);
    std::vector<int32_t> const prefix{1, 2, 3, 4, 5, 6};

    RuntimeCacheAcquireResult cold = adapter.acquireHybridMtp(kDOMAIN, prefix, {}, kSCHEMA, kDRAFT_SIGNATURE);
    CacheRequestLease coldLease = takeLease(cold);
    std::optional<HybridSnapshotReservation> const snapshots = adapter.reserveHybridSnapshots(coldLease, true);
    ASSERT_TRUE(snapshots.has_value());
    BlockHash const fullHash = hashBlock(kCHAIN_ROOT, prefix.data(), kPAGE_SIZE);
    BlockHash const exactDigest = hashExactPrefix(prefix.data(), prefix.size(), kPAGE_SIZE);
    ASSERT_EQ(adapter
                  .publishHybridMtp(coldLease,
                      HybridMtpPublishRequest{{fullHash},
                          HybridMtpCheckpointKey{kDOMAIN, exactDigest, 6, kSCHEMA, kDRAFT_SIGNATURE},
                          PublicationPoint::kPrefillEnd, CommitPolicy::kPrefillStateOnly, *snapshots})
                  .status,
        PublishStatus::kPublished);
    adapter.retireHybridSnapshotReservation(coldLease, *snapshots);
    coldLease.release();

    // The token that followed the checkpoint at publication time. A consumer sharing the [0, 6) prefix must hit
    // regardless of which token it puts at position 6, because the successor-dependent boundary slot is recomputed at
    // restore rather than matched in the lookup key.
    std::vector<int32_t> const matchingSuccessor{1, 2, 3, 4, 5, 6, kPUBLISHED_SUCCESSOR, 8, 9, 10};
    RuntimeCacheAcquireResult hit = adapter.acquireHybridMtp(kDOMAIN, matchingSuccessor, {}, kSCHEMA, kDRAFT_SIGNATURE);
    ASSERT_EQ(hit.status, AcquireStatus::kAcquired);
    CacheRequestLease hitLease = takeLease(hit);
    EXPECT_EQ(hitLease.reuseTokenLength(), 6);
    EXPECT_EQ(hitLease.basePages().size(), 3U);
    EXPECT_EQ(hitLease.draftPages().size(), 3U);
    EXPECT_TRUE(hitLease.recurrentSnapshotSlot().has_value());
    EXPECT_TRUE(hitLease.partialKvSnapshotSlot().has_value());
    adapter.releaseRestoredHybridSnapshots(hitLease);
    hitLease.release();

    // A different successor token used to force a cold miss; it now hits the same checkpoint with the same reuse.
    std::vector<int32_t> changedSuccessor = matchingSuccessor;
    changedSuccessor[6] = kPUBLISHED_SUCCESSOR + 1;
    RuntimeCacheAcquireResult rehit
        = adapter.acquireHybridMtp(kDOMAIN, changedSuccessor, {}, kSCHEMA, kDRAFT_SIGNATURE);
    ASSERT_EQ(rehit.status, AcquireStatus::kAcquired);
    CacheRequestLease rehitLease = takeLease(rehit);
    EXPECT_EQ(rehitLease.reuseTokenLength(), 6);
    EXPECT_EQ(rehitLease.basePages().size(), 3U);
    EXPECT_EQ(rehitLease.draftPages().size(), 3U);
    EXPECT_TRUE(rehitLease.recurrentSnapshotSlot().has_value());
    EXPECT_TRUE(rehitLease.partialKvSnapshotSlot().has_value());
    adapter.releaseRestoredHybridSnapshots(rehitLease);
    rehitLease.release();
}

TEST(ContextCacheRuntimeAdapterTests, PreparedHybridDigestExcludesMediaStartingAfterCheckpointInSamePage)
{
    constexpr RecurrentStateSchemaId kSCHEMA{47, 53};
    ContextCacheRuntimeAdapter adapter(kPAGE_SIZE, ResourceDemand{4, 0, 2, 2}, /*maxRecords=*/2);
    std::vector<int32_t> const prefix{1, 2, 3, 4, 5, 6};

    RuntimeCacheAcquireResult cold = adapter.acquireHybrid(kDOMAIN, prefix, {}, kSCHEMA, /*hasAttention=*/true);
    CacheRequestLease coldLease = takeLease(cold);
    std::optional<HybridSnapshotReservation> const snapshots = adapter.reserveHybridSnapshots(coldLease, true);
    ASSERT_TRUE(snapshots.has_value());
    BlockHash const prefixFullHash = hashBlock(kCHAIN_ROOT, prefix.data(), kPAGE_SIZE);
    BlockHash const exactDigest = hashExactPrefix(prefix.data(), prefix.size(), kPAGE_SIZE);
    ASSERT_EQ(adapter
                  .publishHybrid(coldLease,
                      HybridPublishRequest{{prefixFullHash}, HybridCheckpointKey{kDOMAIN, exactDigest, 6, kSCHEMA},
                          PublicationPoint::kPrefillEnd, CommitPolicy::kIncludingGeneratedTokens, *snapshots})
                  .status,
        PublishStatus::kPublished);
    coldLease.release();

    std::vector<int32_t> const extension{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    BlockKeyExtras futureMedia;
    futureMedia.media.push_back(MediaSpanKey{{59, 61}, {67, 71}, 3, 0, 1});
    std::vector<BlockHash> const fullHashes
        = hashFullBlocks(extension.data(), extension.size(), kPAGE_SIZE, {{}, futureMedia});
    RuntimeCacheAcquireResult hit = adapter.acquireHybrid(kDOMAIN, {HybridCheckpointCandidate{6, exactDigest}},
        fullHashes, static_cast<int32_t>(extension.size()), kSCHEMA,
        /*hasAttention=*/true);
    ASSERT_EQ(hit.status, AcquireStatus::kAcquired);
    CacheRequestLease hitLease = takeLease(hit);
    EXPECT_EQ(hitLease.reuseTokenLength(), 6);
}

TEST(ContextCacheRuntimeAdapterTests, SpecAcquisitionUsesPairedFullPageReplay)
{
    ContextCacheRuntimeAdapter adapter(kPAGE_SIZE, ResourceDemand{6, 6, 0, 0}, /*maxRecords=*/2);

    RuntimeCacheAcquireResult cold
        = adapter.acquireSpec(SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE);
    CacheRequestLease coldLease = takeLease(cold);
    ASSERT_EQ(coldLease.basePages().size(), 2U);
    ASSERT_EQ(coldLease.draftPages().size(), 2U);
    ASSERT_EQ(adapter
                  .publish(coldLease,
                      PublishRequest{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE, PublicationPoint::kDecodeEnd,
                          CommitPolicy::kIncludingGeneratedTokens, 2 * kPAGE_SIZE})
                  .status,
        PublishStatus::kPublished);
    coldLease.release();

    RuntimeCacheAcquireResult hit = adapter.acquireSpec(
        SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE);
    ASSERT_EQ(hit.status, AcquireStatus::kAcquired);
    EXPECT_EQ(hit.planKind, ReusePlanKind::kStandard);
    EXPECT_FALSE(hit.forcedCold);
    CacheRequestLease hitLease = takeLease(hit);
    EXPECT_EQ(hitLease.reuseTokenLength(), kPAGE_SIZE);
    EXPECT_EQ(hitLease.basePages().size(), 3U);
    EXPECT_EQ(hitLease.draftPages().size(), 3U);
}

TEST(ContextCacheRuntimeAdapterTests, SpecHitCapacityFailureFallsBackToForcedCold)
{
    ContextCacheRuntimeAdapter adapter(kPAGE_SIZE, ResourceDemand{2, 2, 0, 0}, /*maxRecords=*/1);

    RuntimeCacheAcquireResult cold
        = adapter.acquireSpec(SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE);
    CacheRequestLease coldLease = takeLease(cold);
    ASSERT_EQ(adapter
                  .publish(coldLease,
                      PublishRequest{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE, PublicationPoint::kDecodeEnd,
                          CommitPolicy::kIncludingGeneratedTokens, 2 * kPAGE_SIZE})
                  .status,
        PublishStatus::kPublished);
    coldLease.release();

    RuntimeCacheAcquireResult blocked = adapter.acquireSpec(
        SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE);
    EXPECT_EQ(blocked.status, AcquireStatus::kInsufficientCapacity);
    EXPECT_TRUE(blocked.forcedCold);
    EXPECT_FALSE(blocked.lease.has_value());
}

TEST(ContextCacheRuntimeAdapterTests, AbandonedHybridReservationIsReleasedWithTheLease)
{
    constexpr RecurrentStateSchemaId kSCHEMA{73, 79};
    ContextCacheRuntimeAdapter adapter(kPAGE_SIZE, ResourceDemand{0, 0, 1, 1}, /*maxRecords=*/1);

    {
        RuntimeCacheAcquireResult cold
            = adapter.acquireHybrid(kDOMAIN, std::vector<int32_t>{1, 2, 3}, {}, kSCHEMA, /*hasAttention=*/false);
        CacheRequestLease lease = takeLease(cold);
        ASSERT_TRUE(adapter.reserveHybridSnapshots(lease, /*needsPartialKvSnapshot=*/false).has_value());
        EXPECT_EQ(adapter.manager().pools().freeCount(ResourceType::kRecurrentSnapshot), 0);
    }

    EXPECT_EQ(adapter.manager().pools().freeCount(ResourceType::kRecurrentSnapshot), 1);
}
