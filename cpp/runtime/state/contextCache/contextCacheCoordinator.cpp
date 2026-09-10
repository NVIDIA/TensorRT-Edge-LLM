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

#include "runtime/state/contextCache/contextCacheCoordinator.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/contextCache/hybridSnapshotStorage.h"
#include "runtime/state/contextCache/reusePlan.h"
#include "runtime/state/contextCache/specStatePlan.h"
#include "runtime/state/kvPageTable.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{

//! The boundary hidden snapshot holds one base-model hidden vector; the base hidden-states tensor is FP16 in this
//! runtime, so kHALF matches its element type. Pool sizing and slab allocation must agree on it.
constexpr nvinfer1::DataType kBOUNDARY_HIDDEN_TYPE{nvinfer1::DataType::kHALF};

bool shouldLogDegradation(uint64_t count) noexcept
{
    return count != 0U && (count & (count - 1U)) == 0U;
}

void requireKvPageCoverage(ReusePlan& plan, int32_t basePages, int32_t draftPages)
{
    ELLM_CHECK(basePages >= 0 && draftPages >= 0, "Context cache KV-page coverage must be non-negative");
    auto extendDemand = [](int32_t& demand, size_t boundPages, int32_t requiredPages) {
        ELLM_CHECK(boundPages <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
            "Context cache reuse path contains too many pages");
        int32_t const privatePages = std::max(0, requiredPages - static_cast<int32_t>(boundPages));
        demand = std::max(demand, privatePages);
    };
    extendDemand(plan.demand.baseKvPages, plan.basePageBindings.size(), basePages);
    extendDemand(plan.demand.draftKvPages, plan.specPageBindings.size(), draftPages);
}

int32_t snapshotSlotCount(int64_t budgetBytes, size_t bytesPerSlot, char const* label)
{
    ELLM_CHECK(budgetBytes >= 0, std::string("Context cache ") + label + " budget must be non-negative");
    ELLM_CHECK(bytesPerSlot > 0, std::string("Context cache ") + label + " has an empty snapshot schema");
    uint64_t const slots = static_cast<uint64_t>(budgetBytes) / static_cast<uint64_t>(bytesPerSlot);
    ELLM_CHECK(slots <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
        std::string("Context cache ") + label + " budget produces too many snapshot slots");
    return static_cast<int32_t>(slots);
}

ResourceDemand makeResourceCapacities(ContextCachePhysicalResources const& resources,
    ContextCacheDeploymentProfile const& profile, DeploymentConfig const& deployment, ContextCacheConfig const& config)
{
    ELLM_CHECK(config.recurrentSnapshotPoolBytes >= 0 && config.partialKvSnapshotPoolBytes >= 0,
        "Context cache snapshot budgets must be non-negative");
    bool const ownsPagedSpecState
        = profile.specReuseContract.has_value() && profile.specReuseContract->ownsPagedSpecState;
    int32_t const draftPages
        = ownsPagedSpecState && resources.draftPageTable != nullptr ? resources.draftPageTable->numPages() : 0;
    bool const hybrid = profile.usesCheckpointReuse();
    bool const hybridMtp = profile.isHybrid() && profile.isSpeculative();
    int32_t recurrentSlots{};
    int32_t partialKvSlots{};
    if (hybrid)
    {
        // Every slab HybridSnapshotStorage allocates per slot has to be charged to the budget that derives the slot
        // count, or the pool overshoots the caller's declared device-memory ceiling by the unpriced slabs.
        size_t recurrentBytes
            = HybridSnapshotStorage::recurrentBytesPerSlot(resources.baseCache.getMambaCacheManager().getConfig());
        if (hybridMtp)
        {
            recurrentBytes
                += HybridSnapshotStorage::boundaryHiddenBytesPerSlot(deployment.base.hiddenSize, kBOUNDARY_HIDDEN_TYPE);
        }
        recurrentSlots
            = snapshotSlotCount(config.recurrentSnapshotPoolBytes, recurrentBytes, "recurrent snapshot pool");
        if (profile.isHybrid())
        {
            size_t partialKvBytes
                = HybridSnapshotStorage::partialKvBytesPerSlot(resources.baseCache.getKVCacheManager().getConfig());
            if (hybridMtp)
            {
                ELLM_CHECK(resources.draftCache != nullptr,
                    "Hybrid+MTP context reuse requires draft cache resources to size the partial-KV snapshot pool");
                partialKvBytes += HybridSnapshotStorage::partialKvBytesPerSlot(
                    resources.draftCache->getKVCacheManager().getConfig());
            }
            partialKvSlots
                = snapshotSlotCount(config.partialKvSnapshotPoolBytes, partialKvBytes, "partial-KV snapshot pool");
        }
    }
    return ResourceDemand{
        profile.hasAttention() ? resources.basePageTable.numPages() : 0, draftPages, recurrentSlots, partialKvSlots};
}

bool hasBlockIdentity(BlockKeyExtras const& extras) noexcept
{
    return !extras.media.empty() || extras.adapter.has_value() || extras.positionDigest.has_value()
        || extras.customEmbeddingDigest.has_value() || extras.isolationDigest.has_value();
}

std::vector<BlockHash> hashRequestFullBlocks(int32_t const* tokens, size_t tokenCount, BlockKeyExtras const& extras,
    Hash128 const* perPositionMediaHash = nullptr)
{
    if (!hasBlockIdentity(extras))
    {
        return hashFullBlocks(tokens, tokenCount, kTOKENS_PER_PAGE, {}, perPositionMediaHash);
    }

    size_t const pageSize = static_cast<size_t>(kTOKENS_PER_PAGE);
    size_t const blockCount = tokenCount / pageSize;
    std::vector<BlockHash> hashes;
    hashes.reserve(blockCount);
    BlockHash parent = kCHAIN_ROOT;
    for (size_t block = 0; block < blockCount; ++block)
    {
        Hash128 const* blockMediaHash
            = perPositionMediaHash != nullptr ? perPositionMediaHash + block * pageSize : nullptr;
        parent = hashBlock(parent, tokens + block * pageSize, pageSize, extras, blockMediaHash);
        hashes.push_back(parent);
    }
    return hashes;
}

BlockHash hashHybridCandidatePrefix(int32_t const* tokens, int32_t exactLength,
    std::vector<BlockHash> const& fullBlockHashes, BlockKeyExtras const& extras,
    Hash128 const* perPositionMediaHash = nullptr)
{
    ELLM_CHECK(exactLength > 0, "Hybrid context cache candidate length must be positive");
    size_t const pageSize = static_cast<size_t>(kTOKENS_PER_PAGE);
    size_t const tokenCount = static_cast<size_t>(exactLength);
    size_t const precedingFullBlocks = tokenCount / pageSize;
    size_t const partialTokenCount = tokenCount % pageSize;
    ELLM_CHECK(precedingFullBlocks <= fullBlockHashes.size(),
        "Hybrid context cache candidate exceeds the computed full-block hash chain");

    if (partialTokenCount == 0U)
    {
        ELLM_CHECK(precedingFullBlocks > 0U, "Hybrid context cache candidate has no terminal full block");
        return fullBlockHashes[precedingFullBlocks - 1U];
    }

    // The complete prefix is already represented by the chained full-block hash; extend it only with the partial tail.
    BlockHash const parent = precedingFullBlocks == 0U ? kCHAIN_ROOT : fullBlockHashes[precedingFullBlocks - 1U];
    Hash128 const* tailMediaHash
        = perPositionMediaHash != nullptr ? perPositionMediaHash + precedingFullBlocks * pageSize : nullptr;
    return hashBlock(parent, tokens + precedingFullBlocks * pageSize, partialTokenCount, extras, tailMediaHash);
}

BlockHash hashRequestExactPrefix(int32_t const* tokens, size_t tokenCount, BlockKeyExtras const& extras,
    Hash128 const* perPositionMediaHash = nullptr)
{
    std::vector<BlockHash> const fullBlockHashes
        = hashRequestFullBlocks(tokens, tokenCount, extras, perPositionMediaHash);
    return tokenCount % static_cast<size_t>(kTOKENS_PER_PAGE) == 0U
        ? (fullBlockHashes.empty() ? kCHAIN_ROOT : fullBlockHashes.back())
        : hashHybridCandidatePrefix(
              tokens, static_cast<int32_t>(tokenCount), fullBlockHashes, extras, perPositionMediaHash);
}

int32_t pageCountForStateLength(int32_t stateLength)
{
    ELLM_CHECK(stateLength >= 0, "Context cache state length must be non-negative");
    int64_t const pages
        = (static_cast<int64_t>(stateLength) + kTOKENS_PER_PAGE - 1) / static_cast<int64_t>(kTOKENS_PER_PAGE);
    ELLM_CHECK(pages <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "Context cache state requires too many pages");
    return static_cast<int32_t>(pages);
}

void validateCompactionMapping(std::vector<int32_t> const& oldToNew, int32_t oldBatchSize, int32_t newBatchSize)
{
    ELLM_CHECK(static_cast<int32_t>(oldToNew.size()) == oldBatchSize,
        "Context cache compaction mapping must describe every active slot");
    ELLM_CHECK(newBatchSize >= 0 && newBatchSize <= oldBatchSize,
        "Context cache compaction has an invalid destination batch size");

    int32_t nextDestination = 0;
    for (int32_t const destination : oldToNew)
    {
        ELLM_CHECK(destination == -1 || (destination >= 0 && destination < newBatchSize),
            "Context cache compaction contains an invalid destination");
        if (destination >= 0)
        {
            ELLM_CHECK(destination == nextDestination,
                "Context cache compaction must preserve survivor order for in-place state movement");
            ++nextDestination;
        }
    }
    ELLM_CHECK(nextDestination == newBatchSize, "Context cache compaction does not cover every destination");
}

class ActiveRequestRollback final
{
public:
    explicit ActiveRequestRollback(bool& active) noexcept
        : mActive(active)
    {
    }

    ~ActiveRequestRollback() noexcept
    {
        if (mArmed)
        {
            mActive = false;
        }
    }

    void dismiss() noexcept
    {
        mArmed = false;
    }

private:
    bool& mActive;
    bool mArmed{true};
};

} // namespace

struct ContextCacheCoordinator::RequestHandle::Impl
{
    struct RequestSlotToken
    {
        explicit RequestSlotToken(ContextCacheCoordinator& coordinator) noexcept
            : owner(&coordinator)
        {
        }

        RequestSlotToken(RequestSlotToken&& other) noexcept
            : owner(std::exchange(other.owner, nullptr))
        {
        }

        RequestSlotToken& operator=(RequestSlotToken&&) = delete;
        RequestSlotToken(RequestSlotToken const&) = delete;
        RequestSlotToken& operator=(RequestSlotToken const&) = delete;

        ~RequestSlotToken() noexcept
        {
            if (owner != nullptr)
            {
                owner->mRequestActive = false;
            }
        }

        ContextCacheCoordinator* owner{};
    };

    enum class Phase : uint8_t
    {
        kAdmitted,
        kExecuting,
        kFinishing,
    };

    struct StagedHybridPublication
    {
        HybridCheckpointKey checkpoint;
        HybridSnapshotReservation snapshots;
        PublicationPoint point{};
    };

    struct SequenceState
    {
        CacheRequestLease lease;
        std::vector<int32_t> tokenIds;
        BlockKeyExtras keyExtras;
        std::vector<Hash128> perPositionMediaHash;
        int32_t reuseTokenLength{};
        ContextCacheLookupPolicy lookupPolicy{ContextCacheLookupPolicy::kUseCache};
        ContextCacheCommitPolicy commitPolicy{ContextCacheCommitPolicy::kIncludingGeneratedTokens};
        int32_t replayTailLength{};
        int32_t committedStateLength{};
        int32_t publishedFullBlockCount{};
        int32_t publishedExactLength{};
        int32_t commonStateLength{};
        std::optional<int32_t> frozenSpecPrefillLength;
        std::optional<StagedHybridPublication> stagedHybridPublication;
    };

    explicit Impl(ContextCacheCoordinator& coordinator, cudaStream_t requestStream) noexcept
        : requestSlot(coordinator)
        , owner(&coordinator)
        , stream(requestStream)
    {
    }

    //! Declared before sequences so its destructor clears the admission flag only after every lease is released.
    RequestSlotToken requestSlot;
    ContextCacheCoordinator* owner{};
    cudaStream_t stream{};
    bool speculativeRequest{};
    std::vector<int32_t> pendingCompactionMapping;
    int32_t pendingCompactionBatchSize{-1};
    Tensor const* pendingDeviceBatchMapping{};
    std::vector<SequenceState> sequences;

    // --- Request lifecycle state --------------------------------------------------------------------------------
    // Every lifecycle method reads/advances the (phase, deviceWorkPending, specAwaitingFirstCompletion) triple only
    // through the named accessors below -- never the raw fields -- so the legal state graph lives in one place and a
    // mid-method synchronize cannot silently leave the pending flag stale (the bug behind the deviceWorkPending
    // checks: publishHybridMtpEndpoint synchronizes mid-prefill, then re-enqueues the fold and must re-mark pending).
    //
    // Legal (phase, deviceWorkPending, specAwaitingFirstCompletion) tuples:
    //   (kAdmitted,  false, false)  freshly admitted, before prefill
    //   (kExecuting, true,  false)  prefill/decode work enqueued on impl.stream
    //   (kExecuting, true,  true)   EAGLE draft-init enqueued, awaiting first verification round
    //   (kExecuting, false, true)   EAGLE draft-init synchronized, frozen-prefill publication pending (transient)
    //   (kExecuting, false, false)  settled between model steps
    //   (kFinishing, *,     *)      terminal
    bool admitted() const noexcept
    {
        return phase == Phase::kAdmitted;
    }
    bool executing() const noexcept
    {
        return phase == Phase::kExecuting;
    }
    bool hasPendingDeviceWork() const noexcept
    {
        return deviceWorkPending;
    }
    bool awaitingFirstSpecCompletion() const noexcept
    {
        return specAwaitingFirstCompletion;
    }

    //! kAdmitted -> kExecuting with the prefill work enqueued on the stream.
    void beginPrefill() noexcept
    {
        phase = Phase::kExecuting;
        deviceWorkPending = true;
    }
    //! Fresh GPU work was enqueued on impl.stream (prefill grow, decode grow, fold, compaction copy).
    void markDeviceWorkEnqueued() noexcept
    {
        deviceWorkPending = true;
    }
    //! impl.stream was explicitly synchronized (or a verification round already synchronized it).
    void markDeviceWorkSynchronized() noexcept
    {
        deviceWorkPending = false;
    }
    //! The enqueued work is provably terminal without a coordinator synchronize (no async publish work remains).
    void markDeviceWorkResolvedWithoutSync() noexcept
    {
        deviceWorkPending = false;
    }
    //! EAGLE only: the ordered draft initialization awaits its first verification round before publishing.
    void beginSpecDraftInit() noexcept
    {
        specAwaitingFirstCompletion = true;
    }
    void endSpecDraftInit() noexcept
    {
        specAwaitingFirstCompletion = false;
    }
    //! Enter the terminal phase (normal completion at empty batch, or an unrecoverable capacity/growth failure).
    void markFinishing() noexcept
    {
        phase = Phase::kFinishing;
    }

private:
    Phase phase{Phase::kAdmitted};
    bool deviceWorkPending{};
    bool specAwaitingFirstCompletion{};
};

struct ContextCacheCoordinator::AcquireSequenceResult
{
    std::optional<CacheRequestLease> lease;
    AcquireStatus status{AcquireStatus::kInsufficientCapacity};
    ReusePlan plan;
    bool forcedCold{};
};

// ================================================================================================================
// Publication policies. The coordinator drives the shared request lifecycle (admission, working-set growth,
// compaction, synchronization) and delegates every flavor-specific endpoint/snapshot publication to the policy
// selected once per request in beginRequest (makePublicationPolicy). This replaces the deployment/execution-mode
// branching that previously threaded through finalizePrefillPublication / completeDecodeStep / terminalize:
//   - BaseEndpointPolicy    vanilla request in an attention deployment -> publish ready full-block base endpoints
//   - HybridSnapshotPolicy  request in a recurrent/hybrid deployment  -> stage + publish recurrent snapshots
//   - HybridMtpPolicy       MTP request in a Hybrid+MTP deployment     -> adds the successor-boundary MTP checkpoint
//   - EagleSpecPolicy       EAGLE request                              -> two-phase frozen-prefill publication
// Policies are nested in the coordinator, so they reach its private state, primitives (publishReadyEndpoint,
// reserveHybridCapture, ...) and predicates directly. They never name the private RequestHandle::Impl type: each
// re-derives `auto& impl = mCoordinator.checkedImpl(request)`. The shared advance validation stays on the
// coordinator (validateEagleDecodeAdvances / validateVanillaDecodeAdvances / validateMtpDecodeAdvances /
// assertUniqueCompletedSlots) so the
// subtle committed-plus-lookahead invariant lives in exactly one place.
// ================================================================================================================
class ContextCacheCoordinator::PublicationPolicy
{
public:
    explicit PublicationPolicy(ContextCacheCoordinator& coordinator) noexcept
        : mCoordinator(coordinator)
    {
    }
    virtual ~PublicationPolicy() noexcept = default;
    virtual char const* name() const noexcept = 0;

    //! Apply the post-prefill advance and publish (or arm) every ready endpoint for this flavor.
    virtual void onPrefillFinalized(RequestHandle& request, std::vector<ContextCacheSequenceAdvance> const& advances,
        std::vector<int32_t> const* commonStateLengths) = 0;
    //! Apply the post-decode advance and publish endpoints for the completed slots.
    virtual ContextCacheCoordinatorStatus onDecodeCompleted(RequestHandle& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const& publishableCompletedSlots,
        std::vector<int32_t> const* commonStateLengths) = 0;
    //! EAGLE-only: publish the frozen prefill once the first verification round terminalizes. Default no-op.
    virtual ContextCacheCoordinatorStatus onTerminalize(RequestHandle& /*request*/)
    {
        return ContextCacheCoordinatorStatus::kOk;
    }
    //! Hybrid+MTP-only: publish the successor-boundary checkpoint. Unreachable for other flavors.
    virtual ContextCacheCoordinatorStatus publishMtpBoundary(RequestHandle& /*request*/, int32_t /*slot*/,
        int32_t /*residentStateLength*/, Tensor const& /*hidden*/, int32_t /*row*/)
    {
        ELLM_CHECK(false, "Hybrid+MTP endpoint publication requires a Hybrid+MTP request");
        return ContextCacheCoordinatorStatus::kRequestFailed;
    }
    //! Hybrid+MTP-only: restore the checkpoint's saved boundary hidden row. Unreachable for other flavors.
    virtual ContextCacheCoordinatorStatus restoreMtpBoundary(
        RequestHandle& /*request*/, int32_t /*slot*/, Tensor& /*hidden*/, int32_t /*row*/)
    {
        ELLM_CHECK(false, "Hybrid+MTP boundary-hidden restore requires a Hybrid+MTP request");
        return ContextCacheCoordinatorStatus::kRequestFailed;
    }

protected:
    ContextCacheCoordinator& mCoordinator;
};

class ContextCacheCoordinator::BaseEndpointPolicy : public ContextCacheCoordinator::PublicationPolicy
{
public:
    using PublicationPolicy::PublicationPolicy;
    char const* name() const noexcept override
    {
        return "base-endpoint";
    }

    void onPrefillFinalized(RequestHandle& request, std::vector<ContextCacheSequenceAdvance> const& advances,
        std::vector<int32_t> const* commonStateLengths) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        ELLM_CHECK(commonStateLengths == nullptr || commonStateLengths->empty(),
            "Non-speculative prefill supplied speculative common-state lengths");
        impl.markDeviceWorkResolvedWithoutSync();
        mCoordinator.applyAdvances(impl, advances);
        for (int32_t slot = 0; slot < static_cast<int32_t>(impl.sequences.size()); ++slot)
        {
            mCoordinator.publishReadyEndpoint(impl, slot, PublicationPoint::kPrefillEnd);
        }
    }

    ContextCacheCoordinatorStatus onDecodeCompleted(RequestHandle& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const& publishableCompletedSlots,
        std::vector<int32_t> const* commonStateLengths) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        mCoordinator.validateVanillaDecodeAdvances(impl, advances, commonStateLengths);
        mCoordinator.applyAdvances(impl, advances);
        mCoordinator.assertUniqueCompletedSlots(impl, publishableCompletedSlots);
        impl.markDeviceWorkResolvedWithoutSync();
        for (int32_t const slot : publishableCompletedSlots)
        {
            mCoordinator.publishReadyEndpoint(impl, slot, PublicationPoint::kDecodeEnd);
        }
        return ContextCacheCoordinatorStatus::kOk;
    }
};

class ContextCacheCoordinator::HybridSnapshotPolicy : public ContextCacheCoordinator::PublicationPolicy
{
public:
    using PublicationPolicy::PublicationPolicy;
    char const* name() const noexcept override
    {
        return "hybrid-snapshot";
    }

    void onPrefillFinalized(RequestHandle& request, std::vector<ContextCacheSequenceAdvance> const& advances,
        std::vector<int32_t> const* commonStateLengths) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        ELLM_CHECK(commonStateLengths == nullptr || commonStateLengths->empty(),
            "Non-speculative prefill supplied speculative common-state lengths");
        impl.markDeviceWorkResolvedWithoutSync();
        mCoordinator.applyAdvances(impl, advances);
        for (auto& sequence : impl.sequences)
        {
            mCoordinator.mManager.releaseRestoredHybridSnapshots(sequence.lease);
        }
        mCoordinator.publishReadyHybridEndpoints(impl);
    }

    ContextCacheCoordinatorStatus onDecodeCompleted(RequestHandle& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const& publishableCompletedSlots,
        std::vector<int32_t> const* commonStateLengths) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        validateDecodeAdvances(impl, advances, commonStateLengths);
        mCoordinator.applyAdvances(impl, advances);
        mCoordinator.assertUniqueCompletedSlots(impl, publishableCompletedSlots);
        for (int32_t const slot : publishableCompletedSlots)
        {
            int32_t const exactLength = impl.sequences[static_cast<size_t>(slot)].committedStateLength;
            mCoordinator.reserveHybridCapture(impl, slot, exactLength, PublicationPoint::kDecodeEnd);
        }
        bool const hasCaptures = std::any_of(impl.sequences.begin(), impl.sequences.end(),
            [](auto const& sequence) { return sequence.stagedHybridPublication.has_value(); });
        if (hasCaptures)
        {
            mCoordinator.enqueueHybridCaptures(impl);
            ++mCoordinator.mMetrics.hybridCaptureSynchronizations;
            ContextCacheCoordinatorStatus const syncStatus = mCoordinator.synchronizeRequest(request);
            if (syncStatus != ContextCacheCoordinatorStatus::kOk)
            {
                return syncStatus;
            }
            mCoordinator.publishReadyHybridEndpoints(impl);
        }
        else
        {
            impl.markDeviceWorkResolvedWithoutSync();
        }
        return ContextCacheCoordinatorStatus::kOk;
    }

protected:
    //! The snapshot publication flow is identical for vanilla hybrid and Hybrid+MTP, but their decode-advance
    //! invariants are not: MTP commits everything a verification round accepted. Subclasses that decode
    //! speculatively override this instead of duplicating onDecodeCompleted.
    virtual void validateDecodeAdvances(RequestHandle::Impl const& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const* commonStateLengths) const
    {
        mCoordinator.validateVanillaDecodeAdvances(request, advances, commonStateLengths);
    }
};

class ContextCacheCoordinator::HybridMtpPolicy : public ContextCacheCoordinator::HybridSnapshotPolicy
{
public:
    using HybridSnapshotPolicy::HybridSnapshotPolicy;
    char const* name() const noexcept override
    {
        return "hybrid-mtp";
    }

protected:
    void validateDecodeAdvances(RequestHandle::Impl const& request,
        std::vector<ContextCacheSequenceAdvance> const& advances,
        std::vector<int32_t> const* commonStateLengths) const override
    {
        mCoordinator.validateMtpDecodeAdvances(request, advances, commonStateLengths);
    }

public:
    ContextCacheCoordinatorStatus publishMtpBoundary(RequestHandle& request, int32_t slot, int32_t residentStateLength,
        Tensor const& baseHiddenStates, int32_t boundaryHiddenRow) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        ELLM_CHECK(slot >= 0 && slot < static_cast<int32_t>(impl.sequences.size()),
            "Hybrid+MTP endpoint publication slot is outside the active batch");
        auto& sequence = impl.sequences[static_cast<size_t>(slot)];

        // Guard/skip mirrors publishHybridMtpContextCacheBoundary (ref llmInferenceRuntime.cpp :3308) and
        // publishReadyHybridEndpoints: a bypass request never publishes, an already-published boundary is idempotent,
        // and an empty prefix has nothing to snapshot.
        if (sequence.lookupPolicy == ContextCacheLookupPolicy::kBypass || residentStateLength <= 0
            || residentStateLength <= sequence.publishedExactLength)
        {
            return ContextCacheCoordinatorStatus::kOk;
        }
        ELLM_CHECK(static_cast<size_t>(residentStateLength) <= sequence.tokenIds.size(),
            "Hybrid+MTP endpoint length exceeds the logical token history");
        ELLM_CHECK(boundaryHiddenRow >= 0, "Hybrid+MTP publication has an empty prefill chunk for its boundary hidden");

        // Release any hit-restore snapshot pins before reserving producer storage (ref :3336), then reserve the paired
        // recurrent + partial-KV slots. Retention pressure is a soft skip, exactly like reserveHybridCapture.
        mCoordinator.mManager.releaseRestoredHybridSnapshots(sequence.lease);
        std::optional<HybridSnapshotReservation> const reservation
            = mCoordinator.mManager.reserveHybridSnapshots(sequence.lease, /*needsPartialKvSnapshot=*/true);
        if (!reservation.has_value())
        {
            ++mCoordinator.mMetrics.hybridSnapshotPressureSkips;
            if (shouldLogDegradation(mCoordinator.mMetrics.hybridSnapshotPressureSkips))
            {
                LOG_WARNING("Context cache hybrid snapshot-pressure skip count reached %llu",
                    static_cast<unsigned long long>(mCoordinator.mMetrics.hybridSnapshotPressureSkips));
            }
            return ContextCacheCoordinatorStatus::kOk;
        }

        // This is the local form of SpecReuseContract::futureDependencyTokens == 1: keep the successor-dependent
        // boundary token private so its draft KV can be rewritten on restore. partialTokenCount is in
        // [1, kTOKENS_PER_PAGE]; a page-aligned length yields a full-page partial snapshot.
        size_t const fullBlockCount = static_cast<size_t>((residentStateLength - 1) / kTOKENS_PER_PAGE);
        int32_t const partialTokenCount = residentStateLength - static_cast<int32_t>(fullBlockCount) * kTOKENS_PER_PAGE;
        ELLM_CHECK(reservation->partialKvSnapshotSlot.has_value() && fullBlockCount < sequence.lease.basePages().size()
                && fullBlockCount < sequence.lease.draftPages().size(),
            "Hybrid+MTP endpoint has no live paired partial pages");

        mCoordinator.mHybridSnapshots->captureRecurrent(reservation->recurrentSnapshotSlot, slot, impl.stream);
        mCoordinator.mHybridSnapshots->capturePartialKv(*reservation->partialKvSnapshotSlot,
            sequence.lease.basePages()[fullBlockCount], sequence.lease.draftPages()[fullBlockCount], partialTokenCount,
            impl.stream);
        mCoordinator.mHybridSnapshots->captureBoundaryHidden(
            reservation->recurrentSnapshotSlot, baseHiddenStates, slot, boundaryHiddenRow, impl.stream);
        impl.markDeviceWorkEnqueued();

        size_t const fullTokenCount = fullBlockCount * static_cast<size_t>(kTOKENS_PER_PAGE);
        Hash128 const* seqMediaPtr
            = sequence.perPositionMediaHash.empty() ? nullptr : sequence.perPositionMediaHash.data();
        std::vector<BlockHash> hashes
            = hashRequestFullBlocks(sequence.tokenIds.data(), fullTokenCount, sequence.keyExtras, seqMediaPtr);
        HybridCheckpointKey const checkpoint{hashRequestExactPrefix(sequence.tokenIds.data(),
                                                 static_cast<size_t>(residentStateLength), sequence.keyExtras),
            residentStateLength};

        // The snapshot writes must be terminal before the host commit. Sync via the coordinator's synchronizer, which
        // quarantines the request (taking ownership of the lease and its reservation pins) on a failed drain.
        ++mCoordinator.mMetrics.hybridCaptureSynchronizations;
        ContextCacheCoordinatorStatus const syncStatus = mCoordinator.synchronizeRequest(request);
        if (syncStatus != ContextCacheCoordinatorStatus::kOk)
        {
            return syncStatus;
        }

        PublishResult const result = mCoordinator.mManager.publishHybridMtp(
            sequence.lease, HybridPublishRequest{std::move(hashes), checkpoint, *reservation});
        mCoordinator.recordPublication(result.status);
        mCoordinator.mManager.retireHybridSnapshotReservation(sequence.lease, *reservation);
        sequence.publishedExactLength = residentStateLength;
        sequence.publishedFullBlockCount = result.publishedBaseFullBlockCount;
        // This endpoint is published mid-prefill: the capture sync above cleared the request's pending-prefill flag,
        // but the prefill lifecycle is not finalized yet. finalizePrefillPublication still runs to apply the
        // post-prefill advance, and the two-chunk (volatile-tail) paths enqueue more base-prefill work after this call.
        // Restore the invariant so the runtime's completePrefill -> finalizePrefillPublication step is satisfied.
        impl.markDeviceWorkEnqueued();
        return ContextCacheCoordinatorStatus::kOk;
    }

    ContextCacheCoordinatorStatus restoreMtpBoundary(
        RequestHandle& request, int32_t slot, Tensor& baseHiddenStates, int32_t destinationRow) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        ELLM_CHECK(slot >= 0 && slot < static_cast<int32_t>(impl.sequences.size()),
            "Hybrid+MTP boundary-hidden restore slot is outside the active batch");
        ELLM_CHECK(destinationRow >= 0, "Hybrid+MTP boundary-hidden restore has a negative destination row");
        auto& sequence = impl.sequences[static_cast<size_t>(slot)];
        std::optional<int32_t> const recurrentSnapshot = sequence.lease.recurrentSnapshotSlot();
        ELLM_CHECK(recurrentSnapshot.has_value(),
            "Hybrid+MTP boundary-hidden restore requires a bound recurrent snapshot from the cache hit");
        mCoordinator.mHybridSnapshots->restoreBoundaryHidden(
            *recurrentSnapshot, baseHiddenStates, slot, destinationRow, impl.stream);
        return ContextCacheCoordinatorStatus::kOk;
    }
};

class ContextCacheCoordinator::EagleSpecPolicy : public ContextCacheCoordinator::PublicationPolicy
{
public:
    using PublicationPolicy::PublicationPolicy;
    char const* name() const noexcept override
    {
        return "eagle-spec";
    }

    void onPrefillFinalized(RequestHandle& request, std::vector<ContextCacheSequenceAdvance> const& advances,
        std::vector<int32_t> const* commonStateLengths) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        ELLM_CHECK(commonStateLengths != nullptr && commonStateLengths->size() == impl.sequences.size(),
            "EAGLE prefill finalization requires one common state length per sequence");
        mCoordinator.applyAdvances(impl, advances);
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            auto& sequence = impl.sequences[slot];
            int32_t const commonLength = (*commonStateLengths)[slot];
            ELLM_CHECK(commonLength == sequence.committedStateLength,
                "EAGLE draft initialization did not materialize the complete prompt boundary");
            sequence.commonStateLength = commonLength;
            sequence.frozenSpecPrefillLength = sequence.committedStateLength;
        }
        impl.beginSpecDraftInit();
    }

    ContextCacheCoordinatorStatus onDecodeCompleted(RequestHandle& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const& publishableCompletedSlots,
        std::vector<int32_t> const* commonStateLengths) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        mCoordinator.validateEagleDecodeAdvances(impl, advances, commonStateLengths);
        mCoordinator.applyAdvances(impl, advances);
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            impl.sequences[slot].commonStateLength = (*commonStateLengths)[slot];
        }
        // EAGLE verification already performs the round synchronization. It also proves the ordered draft
        // initialization and any page-table uploads that preceded the first verification.
        impl.markDeviceWorkSynchronized();
        impl.endSpecDraftInit();
        mCoordinator.publishFrozenSpecPrefill(impl);
        mCoordinator.assertUniqueCompletedSlots(impl, publishableCompletedSlots);
        for (int32_t const slot : publishableCompletedSlots)
        {
            auto const& sequence = impl.sequences[static_cast<size_t>(slot)];
            mCoordinator.publishSpecEndpoint(impl, slot, sequence.commonStateLength, PublicationPoint::kDecodeEnd);
        }
        return ContextCacheCoordinatorStatus::kOk;
    }

    ContextCacheCoordinatorStatus onTerminalize(RequestHandle& request) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        if (!impl.awaitingFirstSpecCompletion())
        {
            return ContextCacheCoordinatorStatus::kOk;
        }
        ELLM_CHECK(impl.hasPendingDeviceWork(), "EAGLE initialization terminalization requires pending draft work");
        ContextCacheCoordinatorStatus const syncStatus = mCoordinator.synchronizeRequest(request);
        if (syncStatus != ContextCacheCoordinatorStatus::kOk)
        {
            return syncStatus;
        }
        mCoordinator.publishFrozenSpecPrefill(impl);
        impl.endSpecDraftInit();
        return ContextCacheCoordinatorStatus::kOk;
    }
};

class ContextCacheCoordinator::SharedKvSpecPolicy : public ContextCacheCoordinator::PublicationPolicy
{
public:
    using PublicationPolicy::PublicationPolicy;
    char const* name() const noexcept override
    {
        return "shared-kv-spec";
    }

    void onPrefillFinalized(RequestHandle& request, std::vector<ContextCacheSequenceAdvance> const& advances,
        std::vector<int32_t> const* commonStateLengths) override
    {
        std::vector<int32_t> derived;
        if (commonStateLengths == nullptr || commonStateLengths->empty())
        {
            derived.reserve(advances.size());
            for (auto const& advance : advances)
            {
                derived.push_back(advance.committedStateLength);
            }
            commonStateLengths = &derived;
        }
        EagleSpecPolicy{mCoordinator}.onPrefillFinalized(request, advances, commonStateLengths);
    }

    ContextCacheCoordinatorStatus onDecodeCompleted(RequestHandle& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const& publishableCompletedSlots,
        std::vector<int32_t> const* commonStateLengths) override
    {
        std::vector<int32_t> derived;
        if (commonStateLengths == nullptr || commonStateLengths->empty())
        {
            derived.reserve(advances.size());
            for (auto const& advance : advances)
            {
                derived.push_back(advance.committedStateLength);
            }
            commonStateLengths = &derived;
        }
        return EagleSpecPolicy{mCoordinator}.onDecodeCompleted(
            request, advances, publishableCompletedSlots, commonStateLengths);
    }

    ContextCacheCoordinatorStatus onTerminalize(RequestHandle& request) override
    {
        return EagleSpecPolicy{mCoordinator}.onTerminalize(request);
    }
};

std::unique_ptr<ContextCacheCoordinator::PublicationPolicy> ContextCacheCoordinator::makePublicationPolicy(
    bool speculativeRequest)
{
    if (speculativeRequest && mProfile.isHybrid())
    {
        return std::make_unique<HybridMtpPolicy>(*this);
    }
    if (speculativeRequest && ownsPagedSpecState())
    {
        return std::make_unique<EagleSpecPolicy>(*this);
    }
    if (speculativeRequest)
    {
        return std::make_unique<SharedKvSpecPolicy>(*this);
    }
    if (usesCheckpointReuse())
    {
        return std::make_unique<HybridSnapshotPolicy>(*this);
    }
    return std::make_unique<BaseEndpointPolicy>(*this);
}

ContextCacheCoordinator::RequestHandle::RequestHandle(std::unique_ptr<Impl> impl) noexcept
    : mImpl(std::move(impl))
{
}

ContextCacheCoordinator::RequestHandle::RequestHandle(RequestHandle&& other) noexcept = default;

ContextCacheCoordinator::RequestHandle::~RequestHandle() noexcept
{
    if (mImpl != nullptr)
    {
        ContextCacheCoordinator* const owner = mImpl->owner;
        owner->abandon(std::move(mImpl));
    }
}

bool ContextCacheCoordinator::RequestHandle::valid() const noexcept
{
    return mImpl != nullptr;
}

ContextCacheCoordinator::ContextCacheCoordinator(ContextCacheConfig const& config, DeploymentConfig const& deployment,
    ContextCacheDeploymentProfile profile, ContextCachePhysicalResources resources, cudaStream_t stream,
    StreamSynchronizer synchronizer)
    : mProfile(std::move(profile))
    , mManager(kTOKENS_PER_PAGE, makeResourceCapacities(resources, mProfile, deployment, config), config.maxRecords,
          mProfile.specReuseContract)
    , mBaseCache(resources.baseCache)
    , mBasePageTable(resources.basePageTable)
    , mDraftCache(resources.draftCache)
    , mDraftPageTable(resources.draftPageTable)
    , mStream(stream)
    , mSynchronizer(std::move(synchronizer))
{
    ELLM_CHECK(config.enabled, "ContextCacheCoordinator requires an enabled ContextCacheConfig");
    if (isSpecDeployment())
    {
        bool const pagedSpecState = ownsPagedSpecState();
        ELLM_CHECK((mDraftCache != nullptr && mDraftPageTable != nullptr) == pagedSpecState,
            "Speculative context-cache physical resources do not match the state contract");
    }
    else
    {
        ELLM_CHECK(mDraftCache == nullptr && mDraftPageTable == nullptr,
            "Non-speculative context-cache deployment cannot bind draft physical resources");
    }

    KVCacheManager const& baseKv = mBaseCache.getKVCacheManager();
    ELLM_CHECK(baseKv.numPages() == mBasePageTable.numPages(),
        "Context cache base pool and page table have different physical page counts");
    ELLM_CHECK(baseKv.getConfig().maxBatchSize == deployment.base.maxSupportedBatchSize,
        "Context cache base pool and deployment have different maximum batch sizes");
    if (deploymentHasAttention())
    {
        ELLM_CHECK(mBasePageTable.maxPagesPerSeq() == pagesPerSlot(baseKv.maxCapPadded()),
            "Context cache base page-table row does not match the engine KV capacity");
    }

    if (isSpecDeployment() && ownsPagedSpecState())
    {
        KVCacheManager const& draftKv = mDraftCache->getKVCacheManager();
        ELLM_CHECK(draftKv.numPages() == mDraftPageTable->numPages(),
            "Context cache draft pool and page table have different physical page counts");
        ELLM_CHECK(draftKv.getConfig().maxBatchSize == deployment.draft->maxSupportedBatchSize
                && mDraftPageTable->maxPagesPerSeq() == pagesPerSlot(draftKv.maxCapPadded()),
            "Context cache draft resources do not match the draft engine geometry");
    }

    if (usesCheckpointReuse())
    {
        int32_t const recurrentSlots = mManager.pools().capacity(ResourceType::kRecurrentSnapshot);
        int32_t const partialKvSlots = mManager.pools().capacity(ResourceType::kPartialKvSnapshot);
        ELLM_CHECK(recurrentSlots > 0, "Hybrid context reuse requires at least one recurrent snapshot slot");
        if (mProfile.isHybrid())
        {
            ELLM_CHECK(
                partialKvSlots > 0, "Hybrid attention context reuse requires at least one partial-KV snapshot slot");
        }
        if (mProfile.isHybrid() && mProfile.isSpeculative())
        {
            // Hybrid+MTP additionally snapshots the paired draft KV pages and one base-model hidden row per checkpoint
            // (the successor-dependent boundary hidden state); makeResourceCapacities prices both into the slot counts.
            ELLM_CHECK(mDraftCache != nullptr, "Hybrid+MTP context reuse requires validated draft cache resources");
            mHybridSnapshots = std::make_unique<HybridSnapshotStorage>(mBaseCache, recurrentSlots, partialKvSlots,
                mDraftCache, deployment.base.hiddenSize, kBOUNDARY_HIDDEN_TYPE);
        }
        else
        {
            mHybridSnapshots = std::make_unique<HybridSnapshotStorage>(mBaseCache, recurrentSlots, partialKvSlots);
        }
    }

    if (!mSynchronizer)
    {
        mSynchronizer = [](cudaStream_t requestStream) { return cudaStreamSynchronize(requestStream); };
    }
    mHostReuseLengths = std::make_unique<Tensor>(Coords{deployment.base.maxSupportedBatchSize}, DeviceType::kCPU,
        nvinfer1::DataType::kINT32, "ContextCacheCoordinator::hostReuseLengths");
}

ContextCacheCoordinator::~ContextCacheCoordinator() noexcept
{
    if (mQuarantinedRequest != nullptr || mRequestActive)
    {
        LOG_ERROR(
            "ContextCacheCoordinator destroyed with unresolved request ownership; shutdown must prove "
            "quiescence before physical cache destruction.");
        std::terminate();
    }
}

ContextCacheCoordinator::AcquireSequenceResult ContextCacheCoordinator::acquireSequence(
    ContextCacheSequenceAdmission const& admission, bool speculativeRequest, ContextCacheLookupPolicy lookupPolicy,
    DecodingKvHeadroom const& headroom)
{
    ELLM_CHECK(!admission.tokenIds.empty(), "Context cache cannot admit an empty token sequence");
    ELLM_CHECK(admission.tokenIds.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "Context cache input contains too many tokens");
    ELLM_CHECK(headroom.baseExtraTokens > 0 && headroom.draftExtraTokens >= 0,
        "Decoder KV headroom is outside the supported range");
    ELLM_CHECK(speculativeRequest || headroom.draftExtraTokens == 0,
        "A non-speculative request cannot require draft KV headroom");
    ELLM_CHECK(!ownsPagedSpecState() || !speculativeRequest || headroom.draftExtraTokens > 0,
        "A paged speculative request requires positive draft KV headroom");
    ELLM_CHECK(headroom.draftExtraTokens == 0 || ownsPagedSpecState(),
        "Decoder requested draft KV headroom without an independent draft page pool");
    int32_t const inputTokenCount = static_cast<int32_t>(admission.tokenIds.size());
    ELLM_CHECK(inputTokenCount <= std::numeric_limits<int32_t>::max() - headroom.baseExtraTokens
            && inputTokenCount <= std::numeric_limits<int32_t>::max() - headroom.draftExtraTokens,
        "Context cache initial KV headroom length overflow");
    int32_t const basePages
        = deploymentHasAttention() ? pageCountForStateLength(inputTokenCount + headroom.baseExtraTokens) : 0;
    int32_t const draftPages = speculativeRequest && ownsPagedSpecState()
        ? pageCountForStateLength(inputTokenCount + headroom.draftExtraTokens)
        : 0;
    ELLM_CHECK(basePages <= mBasePageTable.maxPagesPerSeq(),
        "Context cache initial base KV headroom exceeds the engine page-table capacity");
    ELLM_CHECK(draftPages == 0 || draftPages <= mDraftPageTable->maxPagesPerSeq(),
        "Context cache initial draft KV headroom exceeds the engine page-table capacity");
    auto const planningStart = std::chrono::steady_clock::now();
    Hash128 const* mediaHashPtr
        = admission.perPositionMediaHash.empty() ? nullptr : admission.perPositionMediaHash.data();
    std::vector<BlockHash> const hashes = hashRequestFullBlocks(
        admission.tokenIds.data(), admission.tokenIds.size(), admission.keyExtras, mediaHashPtr);
    std::vector<HybridCheckpointCandidate> hybridCandidates;
    if (usesCheckpointReuse() && lookupPolicy == ContextCacheLookupPolicy::kUseCache)
    {
        std::vector<int32_t> const candidateLengths
            = mManager.hybridCandidateLengths(static_cast<int32_t>(admission.tokenIds.size()));
        hybridCandidates.reserve(candidateLengths.size());
        for (int32_t const candidateLength : candidateLengths)
        {
            hybridCandidates.push_back(HybridCheckpointCandidate{candidateLength,
                hashHybridCandidatePrefix(
                    admission.tokenIds.data(), candidateLength, hashes, admission.keyExtras, mediaHashPtr)});
        }
    }
    auto makePlan = [&](ContextCacheLookupPolicy policy) {
        if (speculativeRequest && mProfile.isHybrid())
        {
            // Hybrid+MTP binds the base recurrent/attention path and the equally long coherent draft path at an exact
            // checkpoint. The candidate list is built identically to the hybrid branch above.
            return makeHybridMtpReusePlan(
                hybridCandidates, hashes, inputTokenCount, kTOKENS_PER_PAGE, mManager.records(), policy);
        }
        if (usesCheckpointReuse())
        {
            return makeHybridReusePlan(hybridCandidates, hashes, inputTokenCount, kTOKENS_PER_PAGE,
                deploymentHasAttention(), mManager.records(), policy);
        }
        if (speculativeRequest)
        {
            ELLM_CHECK(mProfile.specReuseContract.has_value(),
                "Speculative cache planning requires a speculative deployment contract");
            return makeSpecReusePlan(SpecReusePlanInput{hashes, inputTokenCount, kTOKENS_PER_PAGE, policy,
                                         mManager.baseIndex(), mManager.specIndex(), mManager.records()},
                *mProfile.specReuseContract);
        }
        return makeVanillaReusePlan(hashes, inputTokenCount, kTOKENS_PER_PAGE, mManager.baseIndex(), policy);
    };

    ReusePlan plan = makePlan(lookupPolicy);
    bool forcedCold = false;
    LOG_INFO("Context cache lookup: %d matched pages, %d matched tokens, %zu total hashes, mediaHash=%s",
        static_cast<int32_t>(plan.basePageBindings.size()), plan.matchedTokenLength, hashes.size(),
        admission.perPositionMediaHash.empty() ? "empty" : "present");

    requireKvPageCoverage(plan, basePages, draftPages);
    AcquireResult acquired = mManager.acquire(std::move(plan));
    bool const cacheDerivedPlan
        = acquired.plan.reuseTokenLength > 0 || acquired.plan.kind == ReusePlanKind::kFullInputRewind;
    if (acquired.status == AcquireStatus::kInsufficientCapacity && cacheDerivedPlan
        && lookupPolicy == ContextCacheLookupPolicy::kUseCache)
    {
        ReusePlan coldPlan = makePlan(ContextCacheLookupPolicy::kBypass);
        requireKvPageCoverage(coldPlan, basePages, draftPages);
        acquired = mManager.acquire(std::move(coldPlan));
        forcedCold = true;
    }

    auto const planningEnd = std::chrono::steady_clock::now();
    mMetrics.planningNanoseconds += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(planningEnd - planningStart).count());
    return AcquireSequenceResult{std::move(acquired.lease), acquired.status, std::move(acquired.plan), forcedCold};
}

ContextCacheCoordinator::BeginRequestResult ContextCacheCoordinator::beginRequest(
    ContextCacheBatchAdmission const& admission, DecodingKvHeadroom const& headroom, cudaStream_t stream)
{
    ELLM_CHECK(stream == mStream, "Context cache request stream differs from the coordinator construction stream");
    ELLM_CHECK(admission.lookupPolicy == ContextCacheLookupPolicy::kUseCache
            || admission.lookupPolicy == ContextCacheLookupPolicy::kBypass,
        "Context cache admission has an invalid lookup policy");
    ELLM_CHECK(admission.commitPolicy == ContextCacheCommitPolicy::kIncludingGeneratedTokens
            || admission.commitPolicy == ContextCacheCommitPolicy::kPrefillStateOnly,
        "Context cache admission has an invalid commit policy");
    ELLM_CHECK(!admission.speculativeRequest || isSpecDeployment(),
        "Speculative context-cache request does not match the deployment contract");
    bool const containsMedia = std::any_of(admission.sequences.begin(), admission.sequences.end(),
        [](ContextCacheSequenceAdmission const& sequence) { return !sequence.perPositionMediaHash.empty(); });
    ContextCacheLookupPolicy const lookupPolicy
        = admission.speculativeRequest && containsMedia ? ContextCacheLookupPolicy::kBypass : admission.lookupPolicy;
    if (mPoisoned)
    {
        return BeginRequestResult{ContextCacheCoordinatorStatus::kPoisoned, std::nullopt};
    }

    if (mRequestActive)
    {
        return BeginRequestResult{ContextCacheCoordinatorStatus::kRequestFailed, std::nullopt};
    }
    mRequestActive = true;

    ActiveRequestRollback activeRollback(mRequestActive);
    auto request = std::make_unique<RequestHandle::Impl>(*this, stream);
    activeRollback.dismiss();
    request->speculativeRequest = admission.speculativeRequest;
    mPublicationPolicy = makePublicationPolicy(admission.speculativeRequest);
    int32_t maxBatchSize = mBaseCache.getKVCacheManager().getConfig().maxBatchSize;
    if (admission.speculativeRequest && ownsPagedSpecState())
    {
        maxBatchSize = std::min(maxBatchSize, mDraftCache->getKVCacheManager().getConfig().maxBatchSize);
    }
    ELLM_CHECK(!admission.sequences.empty() && admission.sequences.size() <= static_cast<size_t>(maxBatchSize),
        "Context cache admission batch size is outside the engine range");
    request->sequences.reserve(admission.sequences.size());
    std::vector<int32_t> prefillStarts;
    prefillStarts.reserve(admission.sequences.size());

    for (ContextCacheSequenceAdmission const& sequenceAdmission : admission.sequences)
    {
        AcquireSequenceResult acquired
            = acquireSequence(sequenceAdmission, admission.speculativeRequest, lookupPolicy, headroom);
        if (acquired.status != AcquireStatus::kAcquired || !acquired.lease.has_value())
        {
            return BeginRequestResult{ContextCacheCoordinatorStatus::kRequestFailed, std::nullopt};
        }

        RequestHandle::Impl::SequenceState sequence;
        sequence.lease = std::move(*acquired.lease);
        sequence.tokenIds = sequenceAdmission.tokenIds;
        sequence.keyExtras = sequenceAdmission.keyExtras;
        sequence.perPositionMediaHash = sequenceAdmission.perPositionMediaHash;
        sequence.reuseTokenLength = acquired.plan.reuseTokenLength;
        sequence.lookupPolicy = lookupPolicy;
        sequence.commitPolicy = admission.commitPolicy;
        sequence.replayTailLength = admission.replayTailLength;
        sequence.committedStateLength = acquired.plan.reuseTokenLength;
        sequence.commonStateLength = acquired.plan.reuseTokenLength;
        request->sequences.push_back(std::move(sequence));
        prefillStarts.push_back(acquired.plan.reuseTokenLength);
        LOG_INFO("Context cache: sequence %zu reuse %d/%zu tokens (%d pages)", request->sequences.size() - 1,
            acquired.plan.reuseTokenLength, sequenceAdmission.tokenIds.size(),
            acquired.plan.reuseTokenLength / kTOKENS_PER_PAGE);
        ++mMetrics.admittedSequences;
        mMetrics.mediaAwareSequences += static_cast<uint64_t>(!sequenceAdmission.perPositionMediaHash.empty());
        mMetrics.matchedTokens += static_cast<uint64_t>(acquired.plan.matchedTokenLength);
        mMetrics.reusedTokens += static_cast<uint64_t>(acquired.plan.reuseTokenLength);
        mMetrics.hitSequences += static_cast<uint64_t>(acquired.plan.matchedTokenLength > 0);
        mMetrics.lookupBypassSequences
            += static_cast<uint64_t>(lookupPolicy == ContextCacheLookupPolicy::kBypass || acquired.forcedCold);
        if (acquired.forcedCold)
        {
            ++mMetrics.forcedColdSequences;
            if (shouldLogDegradation(mMetrics.forcedColdSequences))
            {
                LOG_WARNING("Context cache forced-cold fallback count reached %llu",
                    static_cast<unsigned long long>(mMetrics.forcedColdSequences));
            }
        }
        switch (acquired.plan.kind)
        {
        case ReusePlanKind::kStandard: ++mMetrics.standardPlans; break;
        case ReusePlanKind::kNoReusablePrefix: ++mMetrics.noReusablePrefixPlans; break;
        case ReusePlanKind::kFullInputRewind: ++mMetrics.fullInputRewindPlans; break;
        }
        mMetrics.specFullPageReplays
            += static_cast<uint64_t>(acquired.plan.specReplayMode == SpecReplayMode::kFullPage);
    }

    AdmissionResult admitted{RequestHandle(std::move(request)), std::move(prefillStarts)};
    return BeginRequestResult{ContextCacheCoordinatorStatus::kOk, std::move(admitted)};
}

ContextCacheCoordinator::RequestHandle::Impl& ContextCacheCoordinator::checkedImpl(RequestHandle& request) const
{
    ELLM_CHECK(request.mImpl != nullptr && request.mImpl->owner == this,
        "Context cache request handle is invalid or belongs to another coordinator");
    return *request.mImpl;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::preparePrefill(RequestHandle& request)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.admitted() && !impl.hasPendingDeviceWork(),
        "Context cache prefill preparation requires a newly admitted request");

    std::vector<KVPageTableRowUpdate> rows;
    std::vector<KVPageTableRowUpdate> draftRows;
    if (deploymentHasAttention())
    {
        rows.reserve(impl.sequences.size());
    }
    if (runsPairedDraftWorkingSet(impl))
    {
        draftRows.reserve(impl.sequences.size());
    }
    ELLM_CHECK(mHostReuseLengths->reshape({static_cast<int64_t>(impl.sequences.size())}),
        "Context cache reuse-length tensor reshape failed");
    int32_t* const reuseLengths = mHostReuseLengths->dataPointer<int32_t>();
    for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
    {
        auto const& sequence = impl.sequences[slot];
        if (deploymentHasAttention())
        {
            auto const& pages = sequence.lease.basePages();
            rows.push_back(KVPageTableRowUpdate{static_cast<int32_t>(slot), pages.empty() ? nullptr : pages.data(),
                static_cast<int32_t>(pages.size())});
        }
        if (runsPairedDraftWorkingSet(impl))
        {
            auto const& pages = sequence.lease.draftPages();
            draftRows.push_back(KVPageTableRowUpdate{static_cast<int32_t>(slot), pages.empty() ? nullptr : pages.data(),
                static_cast<int32_t>(pages.size())});
        }
        reuseLengths[slot] = sequence.reuseTokenLength;
    }

    impl.beginPrefill();
    if (deploymentHasAttention())
    {
        mBasePageTable.setRows(rows);
        mBasePageTable.upload(impl.stream);
    }
    if (runsPairedDraftWorkingSet(impl))
    {
        mDraftPageTable->setRows(draftRows);
        mDraftPageTable->upload(impl.stream);
    }
    if (usesCheckpointReuse())
    {
        ELLM_CHECK(mHybridSnapshots != nullptr, "Hybrid context cache has no snapshot storage");
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            auto& sequence = impl.sequences[slot];
            std::optional<int32_t> const recurrentSnapshot = sequence.lease.recurrentSnapshotSlot();
            if (!recurrentSnapshot.has_value())
            {
                mHybridSnapshots->zeroRecurrent(static_cast<int32_t>(slot), impl.stream);
                continue;
            }

            mHybridSnapshots->restoreRecurrent(*recurrentSnapshot, static_cast<int32_t>(slot), impl.stream);
            if (std::optional<int32_t> const partialSnapshot = sequence.lease.partialKvSnapshotSlot();
                partialSnapshot.has_value())
            {
                // Hybrid+MTP always keeps the boundary token (reuseLength - 1) in a private partial page so its draft
                // KV can be rewritten by the fold: index the boundary page as (reuseLength - 1) / P (mirrors ref
                // llmInferenceRuntime.cpp :3011-3027). Plain hybrid uses the natural page split.
                bool const isMtp = isSpecRequest(impl) && mProfile.isHybrid();
                size_t const destinationIndex = isMtp
                    ? static_cast<size_t>((sequence.reuseTokenLength - 1) / kTOKENS_PER_PAGE)
                    : static_cast<size_t>(sequence.reuseTokenLength / kTOKENS_PER_PAGE);
                int32_t const validTokenCount
                    = sequence.reuseTokenLength - static_cast<int32_t>(destinationIndex) * kTOKENS_PER_PAGE;
                ELLM_CHECK(validTokenCount > 0 && destinationIndex < sequence.lease.basePages().size(),
                    "Hybrid partial-KV restore has no private destination page");
                if (isMtp)
                {
                    // Restore the paired draft partial page into the lease's draft boundary page. The boundary draft
                    // slot itself is recomputed later by the runtime fold; here we only restore the page contents.
                    ELLM_CHECK(destinationIndex < sequence.lease.draftPages().size(),
                        "Hybrid MTP partial-KV restore has no private draft destination page");
                    mHybridSnapshots->restorePartialKv(*partialSnapshot, sequence.lease.basePages()[destinationIndex],
                        sequence.lease.draftPages()[destinationIndex], validTokenCount, impl.stream);
                }
                else
                {
                    mHybridSnapshots->restorePartialKv(
                        *partialSnapshot, sequence.lease.basePages()[destinationIndex], validTokenCount, impl.stream);
                }
            }
            ++mMetrics.hybridRestores;
        }
    }
    mBaseCache.resetForNewSequences(*mHostReuseLengths, impl.stream);
    // Any paired-draft request (EAGLE or Hybrid+MTP) must reset the draft cache's per-request active batch size / KV
    // lengths here too; otherwise the draft cache keeps the previous request's compacted batch size and the draft-side
    // commitSequenceLength fails on the next request.
    if (runsPairedDraftWorkingSet(impl))
    {
        mDraftCache->resetForNewSequences(*mHostReuseLengths, impl.stream);
    }
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::enqueuePrefillCaptures(RequestHandle& request)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(
        impl.executing() && impl.hasPendingDeviceWork(), "Context cache prefill capture requires pending prefill work");
    if (!usesCheckpointReuse())
    {
        return ContextCacheCoordinatorStatus::kOk;
    }

    for (int32_t slot = 0; slot < static_cast<int32_t>(impl.sequences.size()); ++slot)
    {
        int32_t const exactLength = static_cast<int32_t>(impl.sequences[static_cast<size_t>(slot)].tokenIds.size());
        reserveHybridCapture(impl, slot, exactLength, PublicationPoint::kPrefillEnd);
    }
    enqueueHybridCaptures(impl);
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::applyAdvances(
    RequestHandle::Impl& request, std::vector<ContextCacheSequenceAdvance> const& advances)
{
    ELLM_CHECK(
        advances.size() == request.sequences.size(), "Context cache advances must describe every active sequence");
    for (size_t slot = 0; slot < request.sequences.size(); ++slot)
    {
        ContextCacheSequenceAdvance const& delta = advances[slot];
        auto& sequence = request.sequences[slot];
        ELLM_CHECK(delta.acceptedTokenCount >= 0, "Context cache accepted-token count must be non-negative");
        ELLM_CHECK(delta.acceptedTokenCount == 0 || delta.acceptedTokenIds != nullptr,
            "Context cache accepted-token delta has a null token pointer");
        size_t const acceptedCount = static_cast<size_t>(delta.acceptedTokenCount);
        ELLM_CHECK(sequence.tokenIds.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()) - acceptedCount,
            "Context cache token history exceeds int32");
        size_t const updatedTokenCount = sequence.tokenIds.size() + acceptedCount;
        ELLM_CHECK(delta.committedStateLength >= sequence.committedStateLength
                && static_cast<size_t>(delta.committedStateLength) <= updatedTokenCount,
            "Context cache committed state length is not a monotonic materialized prefix");
    }

    for (size_t slot = 0; slot < request.sequences.size(); ++slot)
    {
        ContextCacheSequenceAdvance const& delta = advances[slot];
        auto& sequence = request.sequences[slot];
        if (delta.acceptedTokenCount > 0)
        {
            sequence.tokenIds.insert(sequence.tokenIds.end(), delta.acceptedTokenIds,
                delta.acceptedTokenIds + static_cast<std::ptrdiff_t>(delta.acceptedTokenCount));
        }
        sequence.committedStateLength = delta.committedStateLength;
    }
    return ContextCacheCoordinatorStatus::kOk;
}

void ContextCacheCoordinator::reserveHybridCapture(
    RequestHandle::Impl& request, int32_t slot, int32_t exactLength, PublicationPoint point)
{
    ELLM_CHECK(usesCheckpointReuse() && mHybridSnapshots != nullptr,
        "Hybrid context capture requires a recurrent deployment and snapshot storage");
    ELLM_CHECK(slot >= 0 && slot < static_cast<int32_t>(request.sequences.size()),
        "Hybrid context capture slot is outside the active batch");
    auto& sequence = request.sequences[static_cast<size_t>(slot)];
    ELLM_CHECK(
        !sequence.stagedHybridPublication.has_value(), "Hybrid context capture already has an unpublished snapshot");
    ELLM_CHECK(exactLength > 0 && static_cast<size_t>(exactLength) <= sequence.tokenIds.size(),
        "Hybrid context capture length is outside the logical token history");
    if (sequence.lookupPolicy == ContextCacheLookupPolicy::kBypass || exactLength <= sequence.publishedExactLength
        || (point == PublicationPoint::kDecodeEnd
            && sequence.commitPolicy == ContextCacheCommitPolicy::kPrefillStateOnly))
    {
        return;
    }

    bool const needsPartialKvSnapshot = deploymentHasAttention() && exactLength % kTOKENS_PER_PAGE != 0;
    std::optional<HybridSnapshotReservation> const reservation
        = mManager.reserveHybridSnapshots(sequence.lease, needsPartialKvSnapshot);
    if (!reservation.has_value())
    {
        ++mMetrics.hybridSnapshotPressureSkips;
        if (shouldLogDegradation(mMetrics.hybridSnapshotPressureSkips))
        {
            LOG_WARNING("Context cache hybrid snapshot-pressure skip count reached %llu",
                static_cast<unsigned long long>(mMetrics.hybridSnapshotPressureSkips));
        }
        return;
    }

    Hash128 const* exactMediaPtr
        = sequence.perPositionMediaHash.empty() ? nullptr : sequence.perPositionMediaHash.data();
    HybridCheckpointKey const checkpoint{hashRequestExactPrefix(sequence.tokenIds.data(),
                                             static_cast<size_t>(exactLength), sequence.keyExtras, exactMediaPtr),
        exactLength};
    sequence.stagedHybridPublication = RequestHandle::Impl::StagedHybridPublication{checkpoint, *reservation, point};
}

void ContextCacheCoordinator::enqueueHybridCaptures(RequestHandle::Impl& request)
{
    ELLM_CHECK(usesCheckpointReuse() && mHybridSnapshots != nullptr,
        "Hybrid context capture requires a recurrent deployment and snapshot storage");
    for (size_t slot = 0; slot < request.sequences.size(); ++slot)
    {
        auto& sequence = request.sequences[slot];
        if (!sequence.stagedHybridPublication.has_value())
        {
            continue;
        }
        auto const& staged = *sequence.stagedHybridPublication;
        mHybridSnapshots->captureRecurrent(
            staged.snapshots.recurrentSnapshotSlot, static_cast<int32_t>(slot), request.stream);
        if (staged.snapshots.partialKvSnapshotSlot.has_value())
        {
            int32_t const validTokenCount = staged.checkpoint.exactLength % kTOKENS_PER_PAGE;
            size_t const sourceIndex = static_cast<size_t>(staged.checkpoint.exactLength / kTOKENS_PER_PAGE);
            ELLM_CHECK(validTokenCount > 0 && sourceIndex < sequence.lease.basePages().size(),
                "Hybrid partial-KV capture has no materialized source page");
            mHybridSnapshots->capturePartialKv(*staged.snapshots.partialKvSnapshotSlot,
                sequence.lease.basePages()[sourceIndex], validTokenCount, request.stream);
        }
    }
}

void ContextCacheCoordinator::publishReadyHybridEndpoints(RequestHandle::Impl& request)
{
    for (auto& sequence : request.sequences)
    {
        if (!sequence.stagedHybridPublication.has_value())
        {
            continue;
        }
        auto const staged = *sequence.stagedHybridPublication;
        ELLM_CHECK(staged.checkpoint.exactLength <= sequence.committedStateLength,
            "Hybrid context snapshot exceeds the ready committed boundary");
        size_t const fullTokenCount = static_cast<size_t>(staged.checkpoint.exactLength / kTOKENS_PER_PAGE)
            * static_cast<size_t>(kTOKENS_PER_PAGE);
        Hash128 const* seqMediaPtr
            = sequence.perPositionMediaHash.empty() ? nullptr : sequence.perPositionMediaHash.data();
        std::vector<BlockHash> hashes
            = hashRequestFullBlocks(sequence.tokenIds.data(), fullTokenCount, sequence.keyExtras, seqMediaPtr);
        PublishResult const result = mManager.publishHybrid(
            sequence.lease, HybridPublishRequest{std::move(hashes), staged.checkpoint, staged.snapshots});
        recordPublication(result.status);
        mManager.retireHybridSnapshotReservation(sequence.lease, staged.snapshots);
        sequence.stagedHybridPublication.reset();
        sequence.publishedExactLength = staged.checkpoint.exactLength;
        sequence.publishedFullBlockCount = result.publishedBaseFullBlockCount;
    }
}

void ContextCacheCoordinator::publishReadyEndpoint(RequestHandle::Impl& request, int32_t slot, PublicationPoint point)
{
    auto& sequence = request.sequences[static_cast<size_t>(slot)];
    if (sequence.lookupPolicy == ContextCacheLookupPolicy::kBypass
        || (point == PublicationPoint::kDecodeEnd
            && sequence.commitPolicy == ContextCacheCommitPolicy::kPrefillStateOnly))
    {
        return;
    }

    int32_t const publishBlocks = sequence.committedStateLength / kTOKENS_PER_PAGE;
    if (publishBlocks <= sequence.publishedFullBlockCount)
    {
        return;
    }

    size_t const tokenCount = static_cast<size_t>(publishBlocks) * static_cast<size_t>(kTOKENS_PER_PAGE);
    Hash128 const* pubMediaPtr = sequence.perPositionMediaHash.empty() ? nullptr : sequence.perPositionMediaHash.data();
    std::vector<BlockHash> hashes
        = hashRequestFullBlocks(sequence.tokenIds.data(), tokenCount, sequence.keyExtras, pubMediaPtr);
    PublishResult const result
        = mManager.publish(sequence.lease, PublishRequest{std::move(hashes), sequence.committedStateLength});
    recordPublication(result.status);
    sequence.publishedFullBlockCount = result.publishedBaseFullBlockCount;
}

void ContextCacheCoordinator::publishSpecEndpoint(
    RequestHandle::Impl& request, int32_t slot, int32_t commonStateLength, PublicationPoint point)
{
    ELLM_CHECK(usesFrozenSpecPublication(request), "Speculative endpoint publication requires an EAGLE request");
    auto& sequence = request.sequences[static_cast<size_t>(slot)];
    ELLM_CHECK(commonStateLength >= sequence.publishedFullBlockCount * kTOKENS_PER_PAGE
            && commonStateLength <= sequence.committedStateLength,
        "EAGLE common state is outside the ready base-state boundary");
    if (sequence.lookupPolicy == ContextCacheLookupPolicy::kBypass
        || (point == PublicationPoint::kDecodeEnd
            && sequence.commitPolicy == ContextCacheCommitPolicy::kPrefillStateOnly))
    {
        return;
    }

    int32_t const publishBlocks = commonStateLength / kTOKENS_PER_PAGE;
    if (publishBlocks <= sequence.publishedFullBlockCount)
    {
        return;
    }
    size_t const tokenCount = static_cast<size_t>(publishBlocks) * static_cast<size_t>(kTOKENS_PER_PAGE);
    Hash128 const* specMediaPtr
        = sequence.perPositionMediaHash.empty() ? nullptr : sequence.perPositionMediaHash.data();
    std::vector<BlockHash> hashes
        = hashRequestFullBlocks(sequence.tokenIds.data(), tokenCount, sequence.keyExtras, specMediaPtr);
    PublishResult const result = mManager.publish(sequence.lease, PublishRequest{std::move(hashes), commonStateLength});
    recordPublication(result.status);
    sequence.publishedFullBlockCount = result.publishedBaseFullBlockCount;
    ++mMetrics.specPairPublications;
}

void ContextCacheCoordinator::recordPublication(PublishStatus status) noexcept
{
    ++mMetrics.publicationAttempts;
    switch (status)
    {
    case PublishStatus::kPublished:
        ++mMetrics.committedPublications;
        ++mMetrics.publishedEndpoints;
        break;
    case PublishStatus::kExistingRecord:
        ++mMetrics.existingPublications;
        ++mMetrics.publishedEndpoints;
        break;
    }
}

void ContextCacheCoordinator::publishFrozenSpecPrefill(RequestHandle::Impl& request)
{
    ELLM_CHECK(usesFrozenSpecPublication(request), "Frozen speculative publication requires an EAGLE request");
    for (int32_t slot = 0; slot < static_cast<int32_t>(request.sequences.size()); ++slot)
    {
        auto& sequence = request.sequences[static_cast<size_t>(slot)];
        if (!sequence.frozenSpecPrefillLength.has_value())
        {
            continue;
        }
        int32_t const readyLength = std::min(*sequence.frozenSpecPrefillLength, sequence.commonStateLength);
        publishSpecEndpoint(request, slot, readyLength, PublicationPoint::kPrefillEnd);
        sequence.frozenSpecPrefillLength.reset();
    }
}

void ContextCacheCoordinator::validateEagleDecodeAdvances(RequestHandle::Impl const& request,
    std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const* commonStateLengths) const
{
    ELLM_CHECK(commonStateLengths != nullptr && commonStateLengths->size() == request.sequences.size(),
        "EAGLE decode completion requires one common state length per sequence");
    for (size_t slot = 0; slot < request.sequences.size(); ++slot)
    {
        auto const& sequence = request.sequences[slot];
        auto const& delta = advances[slot];
        int32_t const commonLength = (*commonStateLengths)[slot];
        int64_t const expectedCommittedStateLength
            = static_cast<int64_t>(sequence.committedStateLength) + static_cast<int64_t>(delta.acceptedTokenCount);
        int64_t const expectedTokenCount = static_cast<int64_t>(sequence.committedStateLength) + 1;
        ELLM_CHECK(delta.acceptedTokenCount > 0 && expectedCommittedStateLength <= std::numeric_limits<int32_t>::max()
                && delta.committedStateLength == expectedCommittedStateLength && expectedTokenCount >= 0
                && sequence.tokenIds.size() == static_cast<size_t>(expectedTokenCount),
            "EAGLE decode progress violates the committed-plus-lookahead invariant");
        ELLM_CHECK(commonLength >= sequence.commonStateLength && commonLength <= delta.committedStateLength,
            "EAGLE common state is not a monotonic base-state prefix");
    }
}

void ContextCacheCoordinator::validateVanillaDecodeAdvances(RequestHandle::Impl const& request,
    std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const* commonStateLengths) const
{
    ELLM_CHECK(commonStateLengths == nullptr || commonStateLengths->empty(),
        "Vanilla decode supplied speculative common-state lengths");
    validateCommittedLookaheadAdvances(request, advances, /*tokensPerStep=*/TokensPerDecodeStep::kExactlyOne);
}

void ContextCacheCoordinator::validateMtpDecodeAdvances(RequestHandle::Impl const& request,
    std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const* commonStateLengths) const
{
    // MTP speculative-decodes without EAGLE's common-state tracking, so it supplies no common-state lengths, but it
    // shares EAGLE's committed-plus-lookahead bookkeeping and accepts a variable number of tokens per step.
    ELLM_CHECK(
        commonStateLengths == nullptr || commonStateLengths->empty(), "MTP decode supplied EAGLE common-state lengths");
    validateCommittedLookaheadAdvances(request, advances, /*tokensPerStep=*/TokensPerDecodeStep::kAcceptedCount);
}

void ContextCacheCoordinator::validateCommittedLookaheadAdvances(RequestHandle::Impl const& request,
    std::vector<ContextCacheSequenceAdvance> const& advances, TokensPerDecodeStep tokensPerStep) const
{
    // Both flavors hold tokenIds as the committed prefix plus exactly one lookahead token. They differ only in how
    // far committedStateLength moves per step: a speculative step commits everything it accepted, a vanilla step
    // commits the single sampled token.
    bool const multiToken = tokensPerStep == TokensPerDecodeStep::kAcceptedCount;
    for (size_t slot = 0; slot < request.sequences.size(); ++slot)
    {
        auto const& sequence = request.sequences[slot];
        auto const& delta = advances[slot];
        int64_t const expectedCommittedStateLength = static_cast<int64_t>(sequence.committedStateLength)
            + (multiToken ? static_cast<int64_t>(delta.acceptedTokenCount) : 1);
        int64_t const expectedTokenCount = static_cast<int64_t>(sequence.committedStateLength) + 1;
        ELLM_CHECK((multiToken ? delta.acceptedTokenCount > 0 : delta.acceptedTokenCount == 1)
                && expectedCommittedStateLength <= std::numeric_limits<int32_t>::max()
                && delta.committedStateLength == expectedCommittedStateLength
                && sequence.tokenIds.size() == static_cast<size_t>(expectedTokenCount),
            "Context cache decode progress violates the committed-plus-lookahead invariant");
    }
}

void ContextCacheCoordinator::assertUniqueCompletedSlots(
    RequestHandle::Impl const& request, std::vector<int32_t> const& publishableCompletedSlots) const
{
    std::vector<uint8_t> published(request.sequences.size(), 0U);
    for (int32_t const slot : publishableCompletedSlots)
    {
        ELLM_CHECK(slot >= 0 && slot < static_cast<int32_t>(request.sequences.size()),
            "Context cache completed slot is outside the active batch");
        ELLM_CHECK(published[static_cast<size_t>(slot)] == 0U, "Context cache completed slot appears more than once");
        published[static_cast<size_t>(slot)] = 1U;
    }
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::finalizePrefillPublication(RequestHandle& request,
    std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const* commonStateLengths)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.executing() && impl.hasPendingDeviceWork(),
        "Context cache prefill finalization requires pending prefill work");
    ELLM_CHECK(
        advances.size() == impl.sequences.size(), "Context cache prefill advances must describe every active sequence");
    for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
    {
        auto const& sequence = impl.sequences[slot];
        auto const& delta = advances[slot];
        ELLM_CHECK(delta.acceptedTokenCount == 1
                && delta.committedStateLength == static_cast<int32_t>(sequence.tokenIds.size()),
            "Context cache prefill progress does not describe a complete input plus one lookahead");
    }
    mPublicationPolicy->onPrefillFinalized(request, advances, commonStateLengths);
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::publishHybridMtpEndpoint(RequestHandle& request, int32_t slot,
    int32_t residentStateLength, Tensor const& baseHiddenStates, int32_t boundaryHiddenRow)
{
    checkedImpl(request);
    ELLM_CHECK(mProfile.isHybrid() && mProfile.isSpeculative() && mHybridSnapshots != nullptr,
        "Hybrid+MTP endpoint publication requires a Hybrid+MTP deployment with snapshot storage");
    return mPublicationPolicy->publishMtpBoundary(
        request, slot, residentStateLength, baseHiddenStates, boundaryHiddenRow);
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::restoreHybridMtpBoundaryHidden(
    RequestHandle& request, int32_t slot, Tensor& baseHiddenStates, int32_t destinationRow)
{
    checkedImpl(request);
    ELLM_CHECK(mProfile.isHybrid() && mProfile.isSpeculative() && mHybridSnapshots != nullptr,
        "Hybrid+MTP boundary-hidden restore requires a Hybrid+MTP deployment with snapshot storage");
    return mPublicationPolicy->restoreMtpBoundary(request, slot, baseHiddenStates, destinationRow);
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::prepareDecodeStep(
    RequestHandle& request, DecodingKvHeadroom const& headroom)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.executing()
            && (!impl.hasPendingDeviceWork()
                || (usesFrozenSpecPublication(impl) && impl.awaitingFirstSpecCompletion())),
        "Context cache decode preparation requires terminal prior work");
    ELLM_CHECK(headroom.baseExtraTokens > 0 && headroom.draftExtraTokens >= 0,
        "Decoder KV headroom is outside the supported range");
    ELLM_CHECK(runsPairedDraftWorkingSet(impl) ? headroom.draftExtraTokens > 0 : headroom.draftExtraTokens == 0,
        "Decoder draft KV headroom does not match the active cache resources");

    // Paired speculative leases grow base and independent-draft working sets atomically and upload both page tables.
    // Hybrid+MTP uses the same path because growBasePages rejects its paired lease.
    if (runsPairedDraftWorkingSet(impl))
    {
        struct PageDemand
        {
            int32_t basePages{};
            int32_t draftPages{};
        };
        std::vector<PageDemand> demands;
        demands.reserve(impl.sequences.size());
        for (auto const& sequence : impl.sequences)
        {
            ELLM_CHECK(sequence.committedStateLength <= std::numeric_limits<int32_t>::max() - headroom.baseExtraTokens
                    && sequence.committedStateLength <= std::numeric_limits<int32_t>::max() - headroom.draftExtraTokens,
                "Context-cache working-set length overflow before decode");
            int32_t const basePages = pageCountForStateLength(sequence.committedStateLength + headroom.baseExtraTokens);
            int32_t const draftPages
                = pageCountForStateLength(sequence.committedStateLength + headroom.draftExtraTokens);
            if (basePages > mBasePageTable.maxPagesPerSeq() || draftPages > mDraftPageTable->maxPagesPerSeq())
            {
                impl.markFinishing();
                return ContextCacheCoordinatorStatus::kRequestFailed;
            }
            demands.push_back(PageDemand{basePages, draftPages});
        }

        std::vector<KVPageTableRowUpdate> baseRows;
        std::vector<KVPageTableRowUpdate> draftRows;
        baseRows.reserve(impl.sequences.size());
        draftRows.reserve(impl.sequences.size());
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            auto& sequence = impl.sequences[slot];
            int32_t const baseGrowth
                = demands[slot].basePages - static_cast<int32_t>(sequence.lease.basePages().size());
            int32_t const draftGrowth
                = demands[slot].draftPages - static_cast<int32_t>(sequence.lease.draftPages().size());
            if (!mManager.growSpecPages(sequence.lease, std::max(0, baseGrowth), std::max(0, draftGrowth)))
            {
                impl.markFinishing();
                return ContextCacheCoordinatorStatus::kRequestFailed;
            }
            if (baseGrowth > 0)
            {
                auto const& basePages = sequence.lease.basePages();
                baseRows.push_back(KVPageTableRowUpdate{
                    static_cast<int32_t>(slot), basePages.data(), static_cast<int32_t>(basePages.size())});
            }
            if (draftGrowth > 0)
            {
                auto const& draftPages = sequence.lease.draftPages();
                draftRows.push_back(KVPageTableRowUpdate{
                    static_cast<int32_t>(slot), draftPages.data(), static_cast<int32_t>(draftPages.size())});
            }
        }
        impl.markDeviceWorkEnqueued();
        if (!baseRows.empty())
        {
            mBasePageTable.setRows(baseRows);
            mBasePageTable.upload(impl.stream);
        }
        if (!draftRows.empty())
        {
            mDraftPageTable->setRows(draftRows);
            mDraftPageTable->upload(impl.stream);
        }
        return ContextCacheCoordinatorStatus::kOk;
    }

    for (auto const& sequence : impl.sequences)
    {
        ELLM_CHECK(sequence.committedStateLength < std::numeric_limits<int32_t>::max(),
            "Context cache sequence length overflow before decode");
    }
    if (deploymentHasAttention())
    {
        std::vector<KVPageTableRowUpdate> rows;
        rows.reserve(impl.sequences.size());
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            auto& sequence = impl.sequences[slot];
            int32_t const baseExtraTokens = headroom.baseExtraTokens;
            ELLM_CHECK(sequence.committedStateLength <= std::numeric_limits<int32_t>::max() - baseExtraTokens,
                "Context cache working-set length overflow before decode");
            int32_t const requiredPages = pageCountForStateLength(sequence.committedStateLength + baseExtraTokens);
            if (requiredPages > mBasePageTable.maxPagesPerSeq())
            {
                impl.markFinishing();
                return ContextCacheCoordinatorStatus::kRequestFailed;
            }
            int32_t const currentPages = static_cast<int32_t>(sequence.lease.basePages().size());
            bool const grewPages = requiredPages <= currentPages
                || (isSpecRequest(impl) ? mManager.growSpecPages(sequence.lease, requiredPages - currentPages, 0)
                                        : mManager.growBasePages(sequence.lease, requiredPages - currentPages));
            if (!grewPages)
            {
                impl.markFinishing();
                return ContextCacheCoordinatorStatus::kRequestFailed;
            }
            if (requiredPages > currentPages)
            {
                auto const& pages = sequence.lease.basePages();
                rows.push_back(
                    KVPageTableRowUpdate{static_cast<int32_t>(slot), pages.data(), static_cast<int32_t>(pages.size())});
            }
        }
        if (!rows.empty())
        {
            mBasePageTable.setRows(rows);
            mBasePageTable.upload(impl.stream);
        }
    }
    impl.markDeviceWorkEnqueued();
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::completeDecodeStep(RequestHandle& request,
    std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const& publishableCompletedSlots,
    std::vector<int32_t> const* commonStateLengths)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.executing() && impl.hasPendingDeviceWork(),
        "Context cache decode completion requires pending decode work");
    ELLM_CHECK(
        advances.size() == impl.sequences.size(), "Context cache decode advances must describe every active sequence");
    return mPublicationPolicy->onDecodeCompleted(request, advances, publishableCompletedSlots, commonStateLengths);
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::beginBatchCompaction(
    RequestHandle& request, std::vector<int32_t> const& oldToNew, int32_t newBatchSize, Tensor& deviceBatchMapping)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    // EAGLE finalizes its two-phase draft init before compaction; other flavors no-op.
    ContextCacheCoordinatorStatus const terminalStatus = mPublicationPolicy->onTerminalize(request);
    if (terminalStatus != ContextCacheCoordinatorStatus::kOk)
    {
        return terminalStatus;
    }
    ELLM_CHECK(impl.executing() && !impl.hasPendingDeviceWork(),
        "Context cache compaction preparation requires terminal model work");
    int32_t const oldBatchSize = static_cast<int32_t>(impl.sequences.size());
    validateCompactionMapping(oldToNew, oldBatchSize, newBatchSize);

    impl.markDeviceWorkEnqueued();
    impl.pendingCompactionMapping = oldToNew;
    impl.pendingCompactionBatchSize = newBatchSize;
    impl.pendingDeviceBatchMapping = &deviceBatchMapping;
    ELLM_CHECK(deviceBatchMapping.reshape({oldBatchSize}), "Context cache batch-mapping tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(deviceBatchMapping.rawPointer(), oldToNew.data(),
        static_cast<size_t>(oldBatchSize) * sizeof(int32_t), cudaMemcpyHostToDevice, impl.stream));
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::synchronizeRequest(RequestHandle& request)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    if (!impl.hasPendingDeviceWork())
    {
        return ContextCacheCoordinatorStatus::kOk;
    }
    cudaError_t const status = mSynchronizer(impl.stream);
    if (status != cudaSuccess)
    {
        quarantine(request);
        return ContextCacheCoordinatorStatus::kPoisoned;
    }
    impl.markDeviceWorkSynchronized();
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::compactBatch(RequestHandle& request)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(
        impl.executing() && impl.hasPendingDeviceWork(), "Context cache compaction requires prepared pending work");
    ELLM_CHECK(impl.pendingCompactionBatchSize >= 0 && impl.pendingDeviceBatchMapping != nullptr,
        "Context cache compaction is missing its authoritative mapping");
    std::vector<int32_t> const& oldToNew = impl.pendingCompactionMapping;
    int32_t const newBatchSize = impl.pendingCompactionBatchSize;
    Tensor const& deviceBatchMapping = *impl.pendingDeviceBatchMapping;
    int32_t const oldBatchSize = static_cast<int32_t>(impl.sequences.size());

    ELLM_CHECK(std::none_of(impl.sequences.begin(), impl.sequences.end(),
                   [](auto const& sequence) {
                       return sequence.stagedHybridPublication.has_value()
                           || sequence.frozenSpecPrefillLength.has_value();
                   }),
        "Context cache cannot compact a batch with an unpublished staged endpoint");

    if (deploymentHasAttention())
    {
        mBasePageTable.compactRows(oldToNew, newBatchSize);
        mBasePageTable.upload(impl.stream);
    }
    // Any request that grew a paired draft page path at decode (EAGLE or Hybrid+MTP) must compact the draft page table
    // and draft cache alongside the base side; otherwise the draft rows/state desynchronize from the survivor batch.
    if (runsPairedDraftWorkingSet(impl))
    {
        mDraftPageTable->compactRows(oldToNew, newBatchSize);
        mDraftPageTable->upload(impl.stream);
    }
    mBaseCache.compactBatchSlotState(deviceBatchMapping, oldBatchSize, newBatchSize, impl.stream);
    if (runsPairedDraftWorkingSet(impl))
    {
        mDraftCache->compactBatchSlotState(deviceBatchMapping, oldBatchSize, newBatchSize, impl.stream);
    }
    ContextCacheCoordinatorStatus const syncStatus = synchronizeRequest(request);
    if (syncStatus != ContextCacheCoordinatorStatus::kOk)
    {
        return syncStatus;
    }

    std::vector<RequestHandle::Impl::SequenceState> survivors(static_cast<size_t>(newBatchSize));
    for (int32_t oldSlot = 0; oldSlot < oldBatchSize; ++oldSlot)
    {
        int32_t const newSlot = oldToNew[static_cast<size_t>(oldSlot)];
        if (newSlot >= 0)
        {
            survivors[static_cast<size_t>(newSlot)] = std::move(impl.sequences[static_cast<size_t>(oldSlot)]);
        }
    }
    impl.sequences = std::move(survivors);
    impl.pendingCompactionMapping.clear();
    impl.pendingCompactionBatchSize = -1;
    impl.pendingDeviceBatchMapping = nullptr;
    mBaseCache.setActiveBatchSize(newBatchSize);
    if (runsPairedDraftWorkingSet(impl))
    {
        mDraftCache->setActiveBatchSize(newBatchSize);
    }
    if (newBatchSize == 0)
    {
        impl.markFinishing();
    }
    return ContextCacheCoordinatorStatus::kOk;
}

void ContextCacheCoordinator::quarantine(RequestHandle& request) noexcept
{
    mPoisoned = true;
    if (mQuarantinedRequest != nullptr)
    {
        std::terminate();
    }
    mQuarantinedRequest = std::move(request.mImpl);
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::finish(RequestHandle& request)
{
    if (!request.valid())
    {
        return ContextCacheCoordinatorStatus::kOk;
    }
    // EAGLE finalizes its two-phase draft init at completion; other flavors no-op.
    ContextCacheCoordinatorStatus const terminalStatus = mPublicationPolicy->onTerminalize(request);
    if (terminalStatus != ContextCacheCoordinatorStatus::kOk)
    {
        return terminalStatus;
    }
    ContextCacheCoordinatorStatus const syncStatus = synchronizeRequest(request);
    if (syncStatus != ContextCacheCoordinatorStatus::kOk)
    {
        return syncStatus;
    }

    request.mImpl.reset();
    return ContextCacheCoordinatorStatus::kOk;
}

void ContextCacheCoordinator::abandon(std::unique_ptr<RequestHandle::Impl> request) noexcept
{
    if (request == nullptr)
    {
        return;
    }
    if (!request->hasPendingDeviceWork())
    {
        request.reset();
        return;
    }

    cudaError_t status{cudaErrorUnknown};
    try
    {
        status = mSynchronizer(request->stream);
    }
    catch (...)
    {
        status = cudaErrorUnknown;
    }
    if (status == cudaSuccess)
    {
        request->markDeviceWorkSynchronized();
        request.reset();
        return;
    }

    mPoisoned = true;
    if (mQuarantinedRequest != nullptr)
    {
        std::terminate();
    }
    mQuarantinedRequest = std::move(request);
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::shutdown() noexcept
{
    if (mQuarantinedRequest != nullptr)
    {
        cudaError_t status{cudaErrorUnknown};
        try
        {
            status = mSynchronizer(mQuarantinedRequest->stream);
        }
        catch (...)
        {
            status = cudaErrorUnknown;
        }
        if (status != cudaSuccess)
        {
            return ContextCacheCoordinatorStatus::kPoisoned;
        }
        mQuarantinedRequest->markDeviceWorkSynchronized();
        mQuarantinedRequest.reset();
    }
    if (mRequestActive)
    {
        return ContextCacheCoordinatorStatus::kRequestFailed;
    }
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheMetrics ContextCacheCoordinator::metrics() const noexcept
{
    ContextCacheMetrics result = mMetrics;
    ResourcePools const& pools = mManager.pools();
    result.currentRecords = static_cast<uint64_t>(mManager.records().size());
    result.baseKvPages = ContextCachePoolMetrics{
        pools.freeCount(ResourceType::kBaseKvPage), pools.capacity(ResourceType::kBaseKvPage)};
    result.draftKvPages = ContextCachePoolMetrics{
        pools.freeCount(ResourceType::kDraftKvPage), pools.capacity(ResourceType::kDraftKvPage)};
    result.recurrentSnapshots = ContextCachePoolMetrics{
        pools.freeCount(ResourceType::kRecurrentSnapshot), pools.capacity(ResourceType::kRecurrentSnapshot)};
    result.partialKvSnapshots = ContextCachePoolMetrics{
        pools.freeCount(ResourceType::kPartialKvSnapshot), pools.capacity(ResourceType::kPartialKvSnapshot)};
    ContextCacheManagerMetrics const& managerMetrics = mManager.metrics();
    result.evictedRecords = managerMetrics.evictedRecords;
    result.reclaimedBaseKvPages = managerMetrics.reclaimedBaseKvPages;
    result.reclaimedDraftKvPages = managerMetrics.reclaimedDraftKvPages;
    result.reclaimedRecurrentSnapshots = managerMetrics.reclaimedRecurrentSnapshots;
    result.reclaimedPartialKvSnapshots = managerMetrics.reclaimedPartialKvSnapshots;
    return result;
}

bool ContextCacheCoordinator::isHybridDeployment() const noexcept
{
    return mProfile.isHybrid();
}

bool ContextCacheCoordinator::isPureRecurrentDeployment() const noexcept
{
    return mProfile.isPureRecurrent();
}

bool ContextCacheCoordinator::usesCheckpointReuse() const noexcept
{
    return mProfile.usesCheckpointReuse();
}

bool ContextCacheCoordinator::isSpecDeployment() const noexcept
{
    return mProfile.isSpeculative();
}

bool ContextCacheCoordinator::ownsPagedSpecState() const noexcept
{
    return mProfile.ownsPagedSpecState();
}

bool ContextCacheCoordinator::isSpecRequest(RequestHandle::Impl const& request) const noexcept
{
    return request.speculativeRequest;
}

bool ContextCacheCoordinator::runsPairedDraftWorkingSet(RequestHandle::Impl const& request) const noexcept
{
    return isSpecRequest(request) && ownsPagedSpecState();
}

bool ContextCacheCoordinator::usesFrozenSpecPublication(RequestHandle::Impl const& request) const noexcept
{
    return isSpecRequest(request) && !usesCheckpointReuse();
}

bool ContextCacheCoordinator::deploymentHasAttention() const noexcept
{
    return mProfile.hasAttention();
}

ContextCacheManager const& ContextCacheCoordinator::manager() const noexcept
{
    return mManager;
}

} // namespace rt
} // namespace trt_edgellm
