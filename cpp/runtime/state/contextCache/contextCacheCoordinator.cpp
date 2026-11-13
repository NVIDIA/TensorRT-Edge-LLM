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
constexpr trt_edgellm::DataType kBOUNDARY_HIDDEN_TYPE{trt_edgellm::DataType::kHALF};

bool shouldLogDegradation(uint64_t count) noexcept
{
    return count != 0U && (count & (count - 1U)) == 0U;
}

//! A slot is free, held only by cache records (evictable), or in use by a request: a restore source not yet released
//! or a reserved capture not yet published. A pressure skip leaves a needed pool with no free or evictable slot.
std::string describeSnapshotPool(ResourcePools const& pools, ResourceType type)
{
    int32_t const capacity = pools.capacity(type);
    int32_t inUse{};
    for (int32_t index = 0; index < capacity; ++index)
    {
        if (pools.activeRefCount(ResourceId{type, index}) > 0)
        {
            ++inUse;
        }
    }
    int32_t const freeSlots = pools.freeCount(type);
    return format::fmtstr(
        "%d free, %d evictable, %d in use of %d", freeSlots, capacity - freeSlots - inUse, inUse, capacity);
}

void logHybridSnapshotPressureSkip(ContextCacheManager const& manager, uint64_t skipCount, int32_t sequenceIndex,
    int32_t residentSlot, int32_t prefixLength)
{
    ResourcePools const& pools = manager.pools();
    LOG_WARNING(
        "Context cache hybrid snapshot-pressure skip #%llu: no checkpoint saved for sequence %d (resident slot %d) at "
        "%d tokens; recurrent snapshots: %s; partial-KV snapshots: %s; cached records: %zu",
        static_cast<unsigned long long>(skipCount), sequenceIndex, residentSlot, prefixLength,
        describeSnapshotPool(pools, ResourceType::kRecurrentSnapshot).c_str(),
        describeSnapshotPool(pools, ResourceType::kPartialKvSnapshot).c_str(), manager.records().size());
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

//! Snapshots per batch slot when maxSnapshots is unset: one being staged, the one the sequence resumed from, and two
//! for idle conversations.
constexpr int32_t kDEFAULT_SNAPSHOTS_PER_BATCH_SLOT{4};

ResourceDemand makeResourceCapacities(ContextCachePhysicalResources const& resources,
    ContextCacheDeploymentProfile const& profile, DeploymentConfig const& deployment, ContextCacheConfig const& config)
{
    ELLM_CHECK(config.maxSnapshots >= 0, "Context cache maxSnapshots must be non-negative");
    bool const ownsPagedSpecState
        = profile.specReuseContract.has_value() && profile.specReuseContract->ownsPagedSpecState;
    int32_t const draftPages
        = ownsPagedSpecState && resources.draftPageTable != nullptr ? resources.draftPageTable->numPages() : 0;
    bool const hybrid = profile.usesCheckpointReuse();
    bool const hybridSpec = profile.isHybrid() && profile.isSpeculative();
    ELLM_CHECK(!hybridSpec || deployment.specDecodeMode() != SpecDecodeMode::kEAGLE,
        "EAGLE+hybrid context reuse requires multi-row boundary hidden snapshot sizing that has not been "
        "validated with a real model.");
    int32_t recurrentSlots{};
    int32_t partialKvSlots{};
    if (hybrid)
    {
        // maxSnapshots unset (0): four per runtime batch slot.
        int32_t const snapshots = config.maxSnapshots > 0
            ? config.maxSnapshots
            : kDEFAULT_SNAPSHOTS_PER_BATCH_SLOT * deployment.maxRuntimeBatchSize();
        size_t recurrentBytes = HybridSnapshotStorage::recurrentBytesPerSlot(resources.baseCache.getConfig());
        if (hybridSpec && profile.specReuseContract->futureDependencyTokens > 0)
        {
            recurrentBytes += HybridSnapshotStorage::boundaryHiddenBytesPerSlot(
                deployment.specConfig->baseOutputHiddenDim, kBOUNDARY_HIDDEN_TYPE);
        }
        recurrentSlots = snapshots;
        size_t partialKvBytes = 0;
        if (profile.isHybrid())
        {
            partialKvBytes
                = HybridSnapshotStorage::partialKvBytesPerSlot(resources.baseCache.getKVCacheManager().getConfig());
            if (hybridSpec)
            {
                ELLM_CHECK(resources.draftCache != nullptr,
                    "Hybrid+spec context reuse requires draft cache resources to size the partial-KV snapshot pool");
                partialKvBytes += HybridSnapshotStorage::partialKvBytesPerSlot(
                    resources.draftCache->getKVCacheManager().getConfig());
            }
            partialKvSlots = snapshots;
        }
        LOG_INFO("Context cache: %d snapshots of %zu recurrent and %zu partial-KV bytes each", snapshots,
            recurrentBytes, partialKvBytes);
    }
    return ResourceDemand{profile.hasAttention() ? resources.basePageTable.numPages() : 0, draftPages, recurrentSlots,
        partialKvSlots,
        profile.usesBoundedSwa() && resources.swaPageTable != nullptr ? resources.swaPageTable->numPages() : 0};
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
    explicit ActiveRequestRollback(int32_t& activeRequests) noexcept
        : mActiveRequests(activeRequests)
    {
    }

    ~ActiveRequestRollback() noexcept
    {
        if (mArmed)
        {
            --mActiveRequests;
        }
    }

    void dismiss() noexcept
    {
        mArmed = false;
    }

private:
    int32_t& mActiveRequests;
    bool mArmed{true};
};

//! The policy enums arrive from the runtime's request surface; reject anything outside the enumerators.
void validateSequencePolicies(ContextCacheSequenceAdmission const& admission)
{
    ELLM_CHECK(admission.lookupPolicy == ContextCacheLookupPolicy::kUseCache
            || admission.lookupPolicy == ContextCacheLookupPolicy::kBypass,
        "Context cache sequence admission has an invalid lookup policy");
    ELLM_CHECK(admission.commitPolicy == ContextCacheCommitPolicy::kIncludingGeneratedTokens
            || admission.commitPolicy == ContextCacheCommitPolicy::kPrefillStateOnly,
        "Context cache sequence admission has an invalid commit policy");
}

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
                --owner->mActiveRequests;
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

    //! Sequences admitted with the request start settled. A live admission stays kAdmitting until its
    //! final prefill chunk commits; an aborted one never publishes and waits for eviction.
    enum class AdmissionState : uint8_t
    {
        kSettled,
        kAdmitting,
        kAborted,
    };

    struct SequenceState
    {
        CacheRequestLease lease;
        AdmissionState admissionState{AdmissionState::kSettled};
        bool admissionCheckpointRestored{};
        //! A prepared prefill chunk whose device work the caller has not yet reported complete.
        bool admissionChunkOpen{};
        ResidentRef resident;
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

    //! Declared before sequences so its destructor drops the active-request count only after every
    //! lease is released.
    RequestSlotToken requestSlot;
    std::unique_ptr<PublicationPolicy> publicationPolicy;
    ContextCacheCoordinator* owner{};
    cudaStream_t stream{};
    bool speculativeRequest{};
    DecodingTokenStateContract tokenStateContract{DecodingTokenStateContract::kCommittedPlusLookahead};
    std::vector<int32_t> pendingCompactionMapping;
    int32_t pendingCompactionBatchSize{-1};
    Tensor const* pendingDeviceBatchMapping{};
    std::vector<SequenceState> sequences;
    //! Execution indices that decode in the prepared step; empty means every sequence.
    std::vector<uint8_t> decodeParticipants;

    // --- Request lifecycle state --------------------------------------------------------------------------------
    // Every lifecycle method reads/advances the (phase, deviceWorkPending, specAwaitingFirstCompletion) triple only
    // through the named accessors below -- never the raw fields -- so the legal state graph lives in one place and a
    // mid-method synchronize cannot silently leave the pending flag stale (the bug behind the deviceWorkPending
    // checks: publishHybridSpecEndpoint synchronizes mid-prefill, then re-enqueues the fold and must re-mark pending).
    //
    // Legal (phase, deviceWorkPending, specAwaitingFirstCompletion) tuples:
    //   (kAdmitted,  false, false)  freshly admitted, before prefill
    //   (kExecuting, true,  false)  prefill/decode work enqueued on impl.stream
    //   (kExecuting, true,  true)   EAGLE draft-init enqueued, awaiting first verification round
    //   (kExecuting, false, true)   EAGLE draft-init synchronized, frozen-prefill publication pending (transient)
    //   (kExecuting, false, false)  settled between model steps, whether or not any sequence is resident
    //   (kFinishing, *,     *)      terminal: an unrecoverable capacity/growth failure
    //
    // deviceWorkPending covers request-wide work only. A sequence in admission tracks its open prefill chunk in
    // SequenceState::admissionChunkOpen, cleared only by that sequence's completion or abort, so one sequence's
    // completion cannot resolve another's chunk. Synchronization and teardown consider both.
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
        return deviceWorkPending || hasOpenAdmissionChunk();
    }
    bool hasPendingRequestWork() const noexcept
    {
        return deviceWorkPending;
    }
    //! A sequence seated into the executing request froze its prompt endpoint over a draft prefill that nothing has
    //! synchronized yet: it is synchronized by the next decode verification or at terminalization.
    bool hasUnsynchronizedSeatedPrefill() const noexcept
    {
        return std::any_of(sequences.begin(), sequences.end(),
            [](auto const& sequence) { return sequence.frozenSpecPrefillLength.has_value(); });
    }
    bool hasOpenAdmissionChunk() const noexcept
    {
        return std::any_of(sequences.begin(), sequences.end(),
            [](SequenceState const& sequence) { return sequence.admissionChunkOpen; });
    }
    bool decodesThisStep(size_t sequenceIndex) const noexcept
    {
        return decodeParticipants.empty() || decodeParticipants[sequenceIndex] != 0U;
    }
    SequenceState& sequenceInAdmission(int32_t sequenceIndex, char const* operation)
    {
        ELLM_CHECK(sequenceIndex >= 0 && sequenceIndex < static_cast<int32_t>(sequences.size()),
            std::string("Context cache sequence-admission ") + operation + " index is outside the batch");
        SequenceState& sequence = sequences[static_cast<size_t>(sequenceIndex)];
        ELLM_CHECK(sequence.admissionState == AdmissionState::kAdmitting,
            std::string("Context cache sequence-admission ") + operation + " requires a sequence in admission");
        return sequence;
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
    //! Enter the terminal phase after an unrecoverable capacity/growth failure. Normal completion is the
    //! caller's finish(); a batch that drained to nothing stays executing until then.
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
    //! Set, with no lease, when the planned uncached suffix exceeds mMaxPrefillLength.
    std::optional<int32_t> overlongPrefillLength{};
};

// ================================================================================================================
// Publication policies. The coordinator drives the shared request lifecycle (admission, working-set growth,
// compaction, synchronization) and delegates every flavor-specific endpoint/snapshot publication to the policy
// selected once per request in beginRequest (makePublicationPolicy). This replaces the deployment/execution-mode
// branching that previously threaded through finalizePrefillPublication / completeDecodeStep / terminalize:
//   - BaseEndpointPolicy    vanilla request in an attention deployment -> publish ready full-block base endpoints
//   - HybridSnapshotPolicy  request in a recurrent/hybrid deployment  -> stage + publish recurrent snapshots
//   - HybridSpecPolicy       spec request in a hybrid deployment           -> paired base+draft checkpoint publication
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
        std::vector<int32_t> const* commonStateLengths, SwaWindowStateView const* swaWindowStates) = 0;
    //! Apply the post-decode advance and publish endpoints for the completed slots.
    virtual ContextCacheCoordinatorStatus onDecodeCompleted(RequestHandle& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const& publishableCompletedSlots,
        std::vector<int32_t> const* commonStateLengths, SwaWindowStateView const* swaWindowStates) = 0;
    //! A sequence admitted into the executing request committed its prompt: publish (or arm) its prefill-end
    //! endpoint. The default publishes the ready full-block base endpoint.
    virtual void onSequenceAdmissionFinalized(RequestHandle& request, int32_t sequenceIndex)
    {
        mCoordinator.publishReadyEndpoint(
            mCoordinator.checkedImpl(request), sequenceIndex, PublicationPoint::kPrefillEnd);
    }
    //! EAGLE-only: publish the frozen prefill once the first verification round terminalizes. Default no-op.
    virtual ContextCacheCoordinatorStatus onTerminalize(RequestHandle& /*request*/)
    {
        return ContextCacheCoordinatorStatus::kOk;
    }
    //! Before a compaction that retires the rows @p oldToNew maps below zero. Default: terminalize.
    virtual ContextCacheCoordinatorStatus onCompaction(RequestHandle& request, std::vector<int32_t> const& /*oldToNew*/)
    {
        return onTerminalize(request);
    }
    //! Hybrid+spec-only: publish the successor-boundary checkpoint. Unreachable for other flavors.
    virtual ContextCacheCoordinatorStatus publishMtpBoundary(RequestHandle& /*request*/, int32_t /*slot*/,
        int32_t /*residentStateLength*/, Tensor const& /*hidden*/, int32_t /*row*/)
    {
        ELLM_CHECK(false, "Hybrid+spec endpoint publication requires a Hybrid+spec request");
        return ContextCacheCoordinatorStatus::kRequestFailed;
    }
    //! Hybrid+spec-only: restore the checkpoint's saved boundary hidden row. Unreachable for other flavors.
    virtual ContextCacheCoordinatorStatus restoreMtpBoundary(
        RequestHandle& /*request*/, int32_t /*slot*/, Tensor& /*hidden*/, int32_t /*row*/)
    {
        ELLM_CHECK(false, "Hybrid+spec boundary-hidden restore requires a Hybrid+spec request");
        return ContextCacheCoordinatorStatus::kRequestFailed;
    }

protected:
    //! Record each decoded sequence's common-state length. A sequence that held its ledger this step keeps its own.
    static void assignCommonStateLengths(RequestHandle::Impl& impl,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const& commonStateLengths)
    {
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            if (advances[slot].acceptedTokenCount > 0)
            {
                impl.sequences[slot].commonStateLength = commonStateLengths[slot];
            }
        }
    }

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
        std::vector<int32_t> const* commonStateLengths, SwaWindowStateView const* swaWindowStates) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        ELLM_CHECK(commonStateLengths == nullptr || commonStateLengths->empty(),
            "Non-speculative prefill supplied speculative common-state lengths");
        impl.markDeviceWorkResolvedWithoutSync();
        mCoordinator.applyAdvances(impl, advances);
        for (int32_t slot = 0; slot < static_cast<int32_t>(impl.sequences.size()); ++slot)
        {
            mCoordinator.publishReadyEndpoint(impl, slot, PublicationPoint::kPrefillEnd,
                swaWindowStates == nullptr ? nullptr : &(*swaWindowStates)[static_cast<size_t>(slot)]);
        }
    }

    ContextCacheCoordinatorStatus onDecodeCompleted(RequestHandle& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const& publishableCompletedSlots,
        std::vector<int32_t> const* commonStateLengths, SwaWindowStateView const* swaWindowStates) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        if (impl.tokenStateContract == DecodingTokenStateContract::kFullyCommitted)
        {
            mCoordinator.validateFullyCommittedDecodeAdvances(impl, advances, commonStateLengths);
        }
        else
        {
            mCoordinator.validateVanillaDecodeAdvances(impl, advances, commonStateLengths);
        }
        mCoordinator.applyAdvances(impl, advances);
        mCoordinator.assertUniqueCompletedSlots(impl, publishableCompletedSlots);
        impl.markDeviceWorkResolvedWithoutSync();
        for (int32_t const slot : publishableCompletedSlots)
        {
            mCoordinator.publishReadyEndpoint(impl, slot, PublicationPoint::kDecodeEnd,
                swaWindowStates == nullptr ? nullptr : &(*swaWindowStates)[static_cast<size_t>(slot)]);
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
        std::vector<int32_t> const* commonStateLengths, SwaWindowStateView const* /*swaWindowStates*/) override
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
        std::vector<int32_t> const* commonStateLengths, SwaWindowStateView const* /*swaWindowStates*/) override
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
        // A prefill-end capture staged for a sequence whose final chunk ran in this step is published by
        // that sequence's finalizeSequenceAdmission, never here.
        auto const stagedAtDecodeEnd = [&impl](int32_t slot) {
            auto const& staged = impl.sequences[static_cast<size_t>(slot)].stagedHybridPublication;
            return staged.has_value() && staged->point == PublicationPoint::kDecodeEnd;
        };
        bool const hasCaptures
            = std::any_of(publishableCompletedSlots.begin(), publishableCompletedSlots.end(), stagedAtDecodeEnd);
        if (hasCaptures)
        {
            for (int32_t const slot : publishableCompletedSlots)
            {
                if (stagedAtDecodeEnd(slot))
                {
                    mCoordinator.enqueueHybridCapture(impl, slot);
                }
            }
            ++mCoordinator.mMetrics.hybridCaptureSynchronizations;
            ContextCacheCoordinatorStatus const syncStatus = mCoordinator.synchronizeRequest(request);
            if (syncStatus != ContextCacheCoordinatorStatus::kOk)
            {
                return syncStatus;
            }
            for (int32_t const slot : publishableCompletedSlots)
            {
                if (stagedAtDecodeEnd(slot))
                {
                    mCoordinator.publishReadyHybridEndpoint(impl, slot);
                }
            }
        }
        else
        {
            impl.markDeviceWorkResolvedWithoutSync();
        }
        return ContextCacheCoordinatorStatus::kOk;
    }

    void onSequenceAdmissionFinalized(RequestHandle& request, int32_t sequenceIndex) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        mCoordinator.mManager.releaseRestoredHybridSnapshots(impl.sequences[static_cast<size_t>(sequenceIndex)].lease);
        mCoordinator.publishReadyHybridEndpoint(impl, sequenceIndex);
    }

protected:
    //! The snapshot publication flow is identical for vanilla hybrid and Hybrid+spec, but their decode-advance
    //! invariants are not: MTP commits everything a verification round accepted. Subclasses that decode
    //! speculatively override this instead of duplicating onDecodeCompleted.
    virtual void validateDecodeAdvances(RequestHandle::Impl const& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const* commonStateLengths) const
    {
        mCoordinator.validateVanillaDecodeAdvances(request, advances, commonStateLengths);
    }
};

class ContextCacheCoordinator::HybridSpecPolicy : public ContextCacheCoordinator::HybridSnapshotPolicy
{
public:
    using HybridSnapshotPolicy::HybridSnapshotPolicy;
    char const* name() const noexcept override
    {
        return "hybrid-spec";
    }

    void onPrefillFinalized(RequestHandle& request, std::vector<ContextCacheSequenceAdvance> const& advances,
        std::vector<int32_t> const* commonStateLengths, SwaWindowStateView const* /*swaWindowStates*/) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        ELLM_CHECK(commonStateLengths != nullptr && commonStateLengths->size() == impl.sequences.size(),
            "Hybrid+spec prefill finalization requires one common state length per sequence");
        impl.markDeviceWorkResolvedWithoutSync();
        mCoordinator.applyAdvances(impl, advances);
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            impl.sequences[slot].commonStateLength = (*commonStateLengths)[slot];
        }
    }

    ContextCacheCoordinatorStatus onDecodeCompleted(RequestHandle& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const& publishableCompletedSlots,
        std::vector<int32_t> const* commonStateLengths, SwaWindowStateView const* swaWindowStates) override
    {
        ContextCacheCoordinatorStatus const status = HybridSnapshotPolicy::onDecodeCompleted(
            request, advances, publishableCompletedSlots, commonStateLengths, swaWindowStates);
        if (status == ContextCacheCoordinatorStatus::kOk && commonStateLengths != nullptr
            && !commonStateLengths->empty())
        {
            assignCommonStateLengths(mCoordinator.checkedImpl(request), advances, *commonStateLengths);
        }
        return status;
    }

    //! The checkpoint was published during the prefill (publishMtpBoundary), once the draft had seated the prompt.
    void onSequenceAdmissionFinalized(RequestHandle& request, int32_t sequenceIndex) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        mCoordinator.mManager.releaseRestoredHybridSnapshots(impl.sequences[static_cast<size_t>(sequenceIndex)].lease);
    }

protected:
    void validateDecodeAdvances(RequestHandle::Impl const& request,
        std::vector<ContextCacheSequenceAdvance> const& advances,
        std::vector<int32_t> const* commonStateLengths) const override
    {
        if (commonStateLengths != nullptr && !commonStateLengths->empty())
        {
            mCoordinator.validateEagleDecodeAdvances(request, advances, commonStateLengths);
        }
        else
        {
            mCoordinator.validateMtpDecodeAdvances(request, advances, commonStateLengths);
        }
    }

public:
    ContextCacheCoordinatorStatus publishMtpBoundary(RequestHandle& request, int32_t slot, int32_t residentStateLength,
        Tensor const& baseHiddenStates, int32_t boundaryHiddenRow) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        ELLM_CHECK(slot >= 0 && slot < static_cast<int32_t>(impl.sequences.size()),
            "Hybrid+spec endpoint publication slot is outside the active batch");
        auto& sequence = impl.sequences[static_cast<size_t>(slot)];
        // A blocking request publishes mid-prefill, with its prefill pending; a sequence admitted into an executing
        // request publishes from its own prefill, with no request-wide work in flight. Either is left as it was.
        bool const requestWorkPending = impl.hasPendingRequestWork();

        // Guard/skip mirrors publishHybridMtpContextCacheBoundary (ref llmInferenceRuntime.cpp :3308) and
        // publishReadyHybridEndpoints: a bypass request never publishes, an already-published boundary is idempotent,
        // and an empty prefix has nothing to snapshot.
        if (sequence.lookupPolicy == ContextCacheLookupPolicy::kBypass || residentStateLength <= 0
            || residentStateLength <= sequence.publishedExactLength)
        {
            return ContextCacheCoordinatorStatus::kOk;
        }
        ELLM_CHECK(static_cast<size_t>(residentStateLength) <= sequence.tokenIds.size(),
            "Hybrid+spec endpoint length exceeds the logical token history");
        ELLM_CHECK(mCoordinator.mProfile.specReuseContract->futureDependencyTokens == 0 || boundaryHiddenRow >= 0,
            "Hybrid+spec publication has an empty prefill chunk for its boundary hidden");

        // When futureDependencyTokens > 0 (MTP, EAGLE): keep the successor-dependent boundary token private so its
        // draft KV can be rewritten on restore. When 0 (DFlash, JetSpec, DSpark): all tokens are publishable.
        int32_t const futureDep = mCoordinator.mProfile.specReuseContract->futureDependencyTokens;
        size_t const fullBlockCount
            = static_cast<size_t>((futureDep > 0 ? residentStateLength - 1 : residentStateLength) / kTOKENS_PER_PAGE);
        if (fullBlockCount == 0)
        {
            return ContextCacheCoordinatorStatus::kOk;
        }
        int32_t const partialTokenCount = residentStateLength - static_cast<int32_t>(fullBlockCount) * kTOKENS_PER_PAGE;
        bool const needsPartialKv = partialTokenCount > 0;

        // Release any hit-restore snapshot pins before reserving producer storage (ref :3336), then reserve the paired
        // recurrent + partial-KV slots. Retention pressure is a soft skip, exactly like reserveHybridCapture.
        mCoordinator.mManager.releaseRestoredHybridSnapshots(sequence.lease);
        std::optional<HybridSnapshotReservation> const reservation
            = mCoordinator.mManager.reserveHybridSnapshots(sequence.lease, needsPartialKv);
        if (!reservation.has_value())
        {
            ++mCoordinator.mMetrics.hybridSnapshotPressureSkips;
            logHybridSnapshotPressureSkip(mCoordinator.mManager, mCoordinator.mMetrics.hybridSnapshotPressureSkips,
                slot, sequence.resident.slot, residentStateLength);
            return ContextCacheCoordinatorStatus::kOk;
        }

        ELLM_CHECK(reservation->partialKvSnapshotSlot.has_value() == needsPartialKv,
            "Hybrid+spec endpoint partial-KV reservation does not match the page alignment");
        ELLM_CHECK(
            fullBlockCount < sequence.lease.basePages().size() && fullBlockCount < sequence.lease.draftPages().size(),
            "Hybrid+spec endpoint has no live paired partial pages");

        mCoordinator.mHybridSnapshots->captureRecurrent(
            reservation->recurrentSnapshotSlot, sequence.resident.slot, impl.stream);
        if (needsPartialKv)
        {
            mCoordinator.mHybridSnapshots->capturePartialKv(*reservation->partialKvSnapshotSlot,
                sequence.lease.basePages()[fullBlockCount], sequence.lease.draftPages()[fullBlockCount],
                partialTokenCount, impl.stream);
        }
        if (futureDep > 0)
        {
            mCoordinator.mHybridSnapshots->captureBoundaryHidden(
                reservation->recurrentSnapshotSlot, baseHiddenStates, boundaryHiddenRow, impl.stream);
        }
        impl.markDeviceWorkEnqueued();

        size_t const fullTokenCount = fullBlockCount * static_cast<size_t>(kTOKENS_PER_PAGE);
        Hash128 const* seqMediaPtr
            = sequence.perPositionMediaHash.empty() ? nullptr : sequence.perPositionMediaHash.data();
        std::vector<BlockHash> hashes
            = hashRequestFullBlocks(sequence.tokenIds.data(), fullTokenCount, sequence.keyExtras, seqMediaPtr);
        HybridCheckpointKey const checkpoint{
            hashRequestExactPrefix(
                sequence.tokenIds.data(), static_cast<size_t>(residentStateLength), sequence.keyExtras, seqMediaPtr),
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
        // A blocking request's endpoint is published mid-prefill: the capture sync above cleared the request's
        // pending-prefill flag, but the prefill lifecycle is not finalized yet. finalizePrefillPublication still runs
        // to apply the post-prefill advance, and the two-chunk (volatile-tail) paths enqueue more base-prefill work
        // after this call. Restore the invariant so the runtime's completePrefill -> finalizePrefillPublication step
        // is satisfied.
        if (requestWorkPending)
        {
            impl.markDeviceWorkEnqueued();
        }
        return ContextCacheCoordinatorStatus::kOk;
    }

    ContextCacheCoordinatorStatus restoreMtpBoundary(
        RequestHandle& request, int32_t slot, Tensor& baseHiddenStates, int32_t destinationRow) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        ELLM_CHECK(slot >= 0 && slot < static_cast<int32_t>(impl.sequences.size()),
            "Hybrid+spec boundary-hidden restore slot is outside the active batch");
        ELLM_CHECK(destinationRow >= 0, "Hybrid+spec boundary-hidden restore has a negative destination row");
        auto& sequence = impl.sequences[static_cast<size_t>(slot)];
        std::optional<int32_t> const recurrentSnapshot = sequence.lease.recurrentSnapshotSlot();
        ELLM_CHECK(recurrentSnapshot.has_value(),
            "Hybrid+spec boundary-hidden restore requires a bound recurrent snapshot from the cache hit");
        mCoordinator.mHybridSnapshots->restoreBoundaryHidden(
            *recurrentSnapshot, baseHiddenStates, destinationRow, impl.stream);
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
        std::vector<int32_t> const* commonStateLengths, SwaWindowStateView const* /*swaWindowStates*/) override
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
        std::vector<int32_t> const* commonStateLengths, SwaWindowStateView const* /*swaWindowStates*/) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        mCoordinator.validateEagleDecodeAdvances(impl, advances, commonStateLengths);
        mCoordinator.applyAdvances(impl, advances);
        assignCommonStateLengths(impl, advances, *commonStateLengths);
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

    //! A sequence admitted into the executing request seated its draft prefill without a synchronization: its
    //! prompt endpoint freezes now and publishes, capped at the common state, at the next decode completion (whose
    //! verification synchronizes) or at terminalization.
    void onSequenceAdmissionFinalized(RequestHandle& request, int32_t sequenceIndex) override
    {
        auto& sequence = mCoordinator.checkedImpl(request).sequences[static_cast<size_t>(sequenceIndex)];
        sequence.frozenSpecPrefillLength = sequence.committedStateLength;
    }

    ContextCacheCoordinatorStatus onTerminalize(RequestHandle& request) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        bool const seatedPrefillFrozen = std::any_of(impl.sequences.begin(), impl.sequences.end(),
            [](auto const& sequence) { return sequence.frozenSpecPrefillLength.has_value(); });
        if (!impl.awaitingFirstSpecCompletion() && !seatedPrefillFrozen)
        {
            return ContextCacheCoordinatorStatus::kOk;
        }
        ELLM_CHECK(!impl.awaitingFirstSpecCompletion() || impl.hasPendingDeviceWork(),
            "EAGLE initialization terminalization requires pending draft work");
        // A seated sequence's draft prefill is stream-ordered work no request-wide flag covers.
        impl.markDeviceWorkEnqueued();
        ContextCacheCoordinatorStatus const syncStatus = mCoordinator.synchronizeRequest(request);
        if (syncStatus != ContextCacheCoordinatorStatus::kOk)
        {
            return syncStatus;
        }
        mCoordinator.publishFrozenSpecPrefill(impl);
        impl.endSpecDraftInit();
        return ContextCacheCoordinatorStatus::kOk;
    }

    //! A seated sequence that survives the compaction keeps its frozen endpoint for its first decode completion;
    //! only a retiring one publishes now.
    ContextCacheCoordinatorStatus onCompaction(RequestHandle& request, std::vector<int32_t> const& oldToNew) override
    {
        auto& impl = mCoordinator.checkedImpl(request);
        if (impl.awaitingFirstSpecCompletion())
        {
            return onTerminalize(request);
        }
        bool retiringFrozen = false;
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            retiringFrozen |= oldToNew[slot] < 0 && impl.sequences[slot].frozenSpecPrefillLength.has_value();
        }
        if (!retiringFrozen)
        {
            return ContextCacheCoordinatorStatus::kOk;
        }
        impl.markDeviceWorkEnqueued();
        ContextCacheCoordinatorStatus const syncStatus = mCoordinator.synchronizeRequest(request);
        if (syncStatus != ContextCacheCoordinatorStatus::kOk)
        {
            return syncStatus;
        }
        mCoordinator.publishFrozenSpecPrefill(impl, &oldToNew);
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
        std::vector<int32_t> const* commonStateLengths, SwaWindowStateView const* swaWindowStates) override
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
        EagleSpecPolicy{mCoordinator}.onPrefillFinalized(request, advances, commonStateLengths, swaWindowStates);
    }

    ContextCacheCoordinatorStatus onDecodeCompleted(RequestHandle& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const& publishableCompletedSlots,
        std::vector<int32_t> const* commonStateLengths, SwaWindowStateView const* swaWindowStates) override
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
            request, advances, publishableCompletedSlots, commonStateLengths, swaWindowStates);
    }

    void onSequenceAdmissionFinalized(RequestHandle& request, int32_t sequenceIndex) override
    {
        EagleSpecPolicy{mCoordinator}.onSequenceAdmissionFinalized(request, sequenceIndex);
    }

    ContextCacheCoordinatorStatus onTerminalize(RequestHandle& request) override
    {
        return EagleSpecPolicy{mCoordinator}.onTerminalize(request);
    }

    ContextCacheCoordinatorStatus onCompaction(RequestHandle& request, std::vector<int32_t> const& oldToNew) override
    {
        return EagleSpecPolicy{mCoordinator}.onCompaction(request, oldToNew);
    }
};

std::unique_ptr<ContextCacheCoordinator::PublicationPolicy> ContextCacheCoordinator::makePublicationPolicy(
    bool speculativeRequest)
{
    if (speculativeRequest && mProfile.isHybrid())
    {
        return std::make_unique<HybridSpecPolicy>(*this);
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
          mProfile.specReuseContract, mProfile.swaConfig)
    , mBaseCache(resources.baseCache)
    , mBasePageTable(resources.basePageTable)
    , mDraftCache(resources.draftCache)
    , mDraftPageTable(resources.draftPageTable)
    , mStream(stream)
    , mSynchronizer(std::move(synchronizer))
    , mMaxPrefillLength(deployment.base.maxSupportedInputLength)
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
    ELLM_CHECK((resources.swaPageTable != nullptr) == mProfile.usesBoundedSwa(),
        "Context-cache bounded-SWA resources do not match the deployment profile");

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
    if (mProfile.usesBoundedSwa())
    {
        ELLM_CHECK(resources.swaPageTable->maxPagesPerSeq() == mBasePageTable.maxPagesPerSeq()
                && resources.swaPageTable->numPages() == deployment.base.numSwaPages,
            "Context cache SWA page table does not match the bounded pool geometry");
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
            ELLM_CHECK(deployment.specDecodeMode() != SpecDecodeMode::kEAGLE,
                "EAGLE+hybrid context reuse requires multi-row boundary hidden snapshot sizing that has not been "
                "validated with a real model.");
            // Hybrid+spec additionally snapshots the paired draft KV pages; modes with futureDependencyTokens > 0 (MTP)
            // also snapshot one base-model hidden row per checkpoint for the successor-dependent boundary state.
            ELLM_CHECK(mDraftCache != nullptr, "Hybrid+spec context reuse requires validated draft cache resources");
            int32_t const boundaryHiddenDim = mProfile.specReuseContract->futureDependencyTokens > 0
                ? deployment.specConfig->baseOutputHiddenDim
                : 0;
            mHybridSnapshots = std::make_unique<HybridSnapshotStorage>(
                mBaseCache, recurrentSlots, partialKvSlots, mDraftCache, boundaryHiddenDim, kBOUNDARY_HIDDEN_TYPE);
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
        trt_edgellm::DataType::kINT32, "ContextCacheCoordinator::hostReuseLengths");
}

ContextCacheCoordinator::~ContextCacheCoordinator() noexcept
{
    if (!mQuarantinedRequests.empty() || mActiveRequests != 0)
    {
        LOG_ERROR(
            "ContextCacheCoordinator destroyed with unresolved request ownership; shutdown must prove "
            "quiescence before physical cache destruction.");
        std::terminate();
    }
}

ContextCacheCoordinator::AcquireSequenceResult ContextCacheCoordinator::acquireSequence(
    ContextCacheSequenceAdmission const& admission, bool speculativeRequest, DecodingKvHeadroom const& headroom)
{
    ContextCacheLookupPolicy const lookupPolicy = admission.lookupPolicy;
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
        LOG_DEBUG("Context cache checkpoint candidates: %zu candidates for input %zu tokens (usesCheckpoint=%d)",
            candidateLengths.size(), admission.tokenIds.size(), usesCheckpointReuse());
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
            // Hybrid+spec binds the base recurrent/attention path and the equally long coherent draft path at an exact
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
        return mManager.planVanilla(hashes, inputTokenCount, policy);
    };

    ReusePlan plan = makePlan(lookupPolicy);
    bool forcedCold = false;
    LOG_INFO("Context cache lookup: %d matched pages, %d matched tokens, %zu total hashes, mediaHash=%s",
        static_cast<int32_t>(plan.basePageBindings.size()), plan.matchedTokenLength, hashes.size(),
        admission.perPositionMediaHash.empty() ? "empty" : "present");

    // Refused before acquire(), which would evict other records for a prefill the engine cannot run.
    int32_t const prefillLength = inputTokenCount - plan.reuseTokenLength;
    if (prefillLength > mMaxPrefillLength)
    {
        AcquireSequenceResult refused;
        refused.plan = std::move(plan);
        refused.overlongPrefillLength = prefillLength;
        return refused;
    }

    requireKvPageCoverage(plan, basePages, draftPages);
    AcquireResult acquired = mManager.acquire(std::move(plan));
    bool const cacheDerivedPlan
        = acquired.plan.reuseTokenLength > 0 || acquired.plan.kind == ReusePlanKind::kFullInputRewind;
    // A cold plan prefills the whole input; past the limit the capacity refusal stands.
    if (acquired.status == AcquireStatus::kInsufficientCapacity && cacheDerivedPlan
        && lookupPolicy == ContextCacheLookupPolicy::kUseCache && inputTokenCount <= mMaxPrefillLength)
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

bool ContextCacheCoordinator::acquireIntoRequest(RequestHandle::Impl& request,
    ContextCacheSequenceAdmission const& admission, DecodingKvHeadroom const& headroom, AdmitSequenceResult& refusal)
{
    AcquireSequenceResult acquired = acquireSequence(admission, request.speculativeRequest, headroom);
    if (acquired.status != AcquireStatus::kAcquired || !acquired.lease.has_value())
    {
        refusal.overlongPrefill = acquired.overlongPrefillLength.has_value();
        refusal.insufficientCapacity
            = !refusal.overlongPrefill && acquired.status == AcquireStatus::kInsufficientCapacity;
        refusal.prefillStart = acquired.plan.reuseTokenLength;
        return false;
    }

    RequestHandle::Impl::SequenceState sequence;
    sequence.lease = std::move(*acquired.lease);
    sequence.tokenIds = admission.tokenIds;
    sequence.keyExtras = admission.keyExtras;
    sequence.perPositionMediaHash = admission.perPositionMediaHash;
    sequence.reuseTokenLength = acquired.plan.reuseTokenLength;
    sequence.lookupPolicy = admission.lookupPolicy;
    sequence.commitPolicy = admission.commitPolicy;
    sequence.replayTailLength = admission.replayTailLength;
    sequence.resident = admission.resident;
    sequence.committedStateLength = acquired.plan.reuseTokenLength;
    sequence.commonStateLength = acquired.plan.reuseTokenLength;
    request.sequences.push_back(std::move(sequence));
    LOG_INFO("Context cache: sequence %zu reuse %d/%zu tokens (%d pages)", request.sequences.size() - 1,
        acquired.plan.reuseTokenLength, admission.tokenIds.size(), acquired.plan.reuseTokenLength / kTOKENS_PER_PAGE);
    ++mMetrics.admittedSequences;
    mMetrics.mediaAwareSequences += static_cast<uint64_t>(!admission.perPositionMediaHash.empty());
    mMetrics.matchedTokens += static_cast<uint64_t>(acquired.plan.matchedTokenLength);
    mMetrics.reusedTokens += static_cast<uint64_t>(acquired.plan.reuseTokenLength);
    mMetrics.hitSequences += static_cast<uint64_t>(acquired.plan.matchedTokenLength > 0);
    mMetrics.lookupBypassSequences
        += static_cast<uint64_t>(admission.lookupPolicy == ContextCacheLookupPolicy::kBypass || acquired.forcedCold);
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
    mMetrics.specFullPageReplays += static_cast<uint64_t>(acquired.plan.specReplayMode == SpecReplayMode::kFullPage);
    return true;
}

ContextCacheCoordinator::BeginRequestResult ContextCacheCoordinator::beginRequest(
    ContextCacheBatchAdmission const& admission, DecodingKvHeadroom const& headroom, cudaStream_t stream)
{
    ELLM_CHECK(stream == mStream, "Context cache request stream differs from the coordinator construction stream");
    ELLM_CHECK(admission.tokenStateContract == DecodingTokenStateContract::kCommittedPlusLookahead
            || admission.tokenStateContract == DecodingTokenStateContract::kFullyCommitted,
        "Context cache admission has an invalid token-state contract");
    ELLM_CHECK(!admission.speculativeRequest || isSpecDeployment(),
        "Speculative context-cache request does not match the deployment contract");
    if (mPoisoned)
    {
        return BeginRequestResult{ContextCacheCoordinatorStatus::kPoisoned, std::nullopt};
    }

    // quarantine() is noexcept and runs on the device-failure path, where a bad_alloc would abort
    // the process instead of holding the pages. Buy the storage here, where throwing is harmless,
    // so every request that can become resident already has a quarantine slot waiting for it.
    mQuarantinedRequests.reserve(static_cast<size_t>(mActiveRequests) + 1U);

    ++mActiveRequests;

    ActiveRequestRollback activeRollback(mActiveRequests);
    auto request = std::make_unique<RequestHandle::Impl>(*this, stream);
    activeRollback.dismiss();
    request->speculativeRequest = admission.speculativeRequest;
    request->tokenStateContract = admission.tokenStateContract;
    request->publicationPolicy = makePublicationPolicy(admission.speculativeRequest);
    int32_t maxBatchSize = mBaseCache.getKVCacheManager().getConfig().maxBatchSize;
    if (admission.speculativeRequest && ownsPagedSpecState())
    {
        maxBatchSize = std::min(maxBatchSize, mDraftCache->getKVCacheManager().getConfig().maxBatchSize);
    }
    // An empty batch is a request that will take every sequence through admitSequence().
    ELLM_CHECK(admission.sequences.size() <= static_cast<size_t>(maxBatchSize),
        "Context cache admission batch size is outside the engine range");
    request->sequences.reserve(admission.sequences.size());
    std::vector<int32_t> prefillStarts;
    prefillStarts.reserve(admission.sequences.size());
    std::vector<bool> residentSlots(static_cast<size_t>(maxBatchSize), false);
    std::vector<SwaKVCacheState> swaStates;
    if (mProfile.usesBoundedSwa())
    {
        swaStates.reserve(admission.sequences.size());
    }

    for (ContextCacheSequenceAdmission const& sequenceAdmission : admission.sequences)
    {
        validateSequencePolicies(sequenceAdmission);
        ELLM_CHECK(sequenceAdmission.resident.slot >= 0 && sequenceAdmission.resident.slot < maxBatchSize,
            "Context cache admission resident slot is outside the engine range");
        ELLM_CHECK(sequenceAdmission.resident.epoch != 0, "Context cache admission resident epoch is invalid");
        ELLM_CHECK(!residentSlots[static_cast<size_t>(sequenceAdmission.resident.slot)],
            "Context cache admission contains a duplicate resident slot");
        residentSlots[static_cast<size_t>(sequenceAdmission.resident.slot)] = true;
        AcquireSequenceResult acquired = acquireSequence(sequenceAdmission, admission.speculativeRequest, headroom);
        if (acquired.status != AcquireStatus::kAcquired || !acquired.lease.has_value())
        {
            return BeginRequestResult{
                ContextCacheCoordinatorStatus::kRequestFailed, std::nullopt, acquired.overlongPrefillLength};
        }

        RequestHandle::Impl::SequenceState sequence;
        sequence.lease = std::move(*acquired.lease);
        sequence.resident = sequenceAdmission.resident;
        sequence.tokenIds = sequenceAdmission.tokenIds;
        sequence.keyExtras = sequenceAdmission.keyExtras;
        sequence.perPositionMediaHash = sequenceAdmission.perPositionMediaHash;
        sequence.reuseTokenLength = acquired.plan.reuseTokenLength;
        sequence.lookupPolicy = sequenceAdmission.lookupPolicy;
        sequence.commitPolicy = sequenceAdmission.commitPolicy;
        sequence.replayTailLength = sequenceAdmission.replayTailLength;
        sequence.committedStateLength = acquired.plan.reuseTokenLength;
        sequence.commonStateLength = acquired.plan.reuseTokenLength;
        std::optional<SwaKVCacheState> swaState = mManager.takeSwaState(sequence.lease);
        ELLM_CHECK(swaState.has_value() == mProfile.usesBoundedSwa(),
            "Context-cache admission returned inconsistent SWA state");
        if (swaState.has_value())
        {
            swaState->resident = sequenceAdmission.resident;
            swaStates.push_back(std::move(*swaState));
        }
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
        mMetrics.lookupBypassSequences += static_cast<uint64_t>(
            sequenceAdmission.lookupPolicy == ContextCacheLookupPolicy::kBypass || acquired.forcedCold);
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

    AdmissionResult admitted{RequestHandle(std::move(request)), std::move(prefillStarts), std::move(swaStates)};
    return BeginRequestResult{ContextCacheCoordinatorStatus::kOk, std::move(admitted)};
}

bool ContextCacheCoordinator::retractSequenceAdmission(RequestHandle& request) noexcept
{
    try
    {
        RequestHandle::Impl& impl = checkedImpl(request);
        if (impl.sequences.empty())
        {
            mPoisoned = true;
            return false;
        }
        int32_t const residentSlot = impl.sequences.back().resident.slot;
        impl.sequences.pop_back(); // The lease releases with the sequence state.
        clearSequenceRows(impl, residentSlot);
        return true;
    }
    catch (...)
    {
        // The lease was either released above or remains owned by the request handle. In either
        // case, refuse further admissions because the device page-table row is not trustworthy.
        mPoisoned = true;
        return false;
    }
}

ContextCacheCoordinator::AdmitSequenceResult ContextCacheCoordinator::admitSequence(
    RequestHandle& request, ContextCacheSequenceAdmission const& admission, DecodingKvHeadroom const& headroom)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.executing() && !impl.hasPendingRequestWork(),
        "Context cache sequence admission requires an executing request with no request-wide work in flight");
    ELLM_CHECK(impl.pendingCompactionBatchSize < 0,
        "Context cache sequence admission cannot interleave a pending batch compaction");
    ELLM_CHECK(supportsLiveSequenceAdmission(), "Context cache sequence admission is limited to attention deployments");
    if (mPoisoned)
    {
        return AdmitSequenceResult{ContextCacheCoordinatorStatus::kPoisoned, false, 0};
    }
    int32_t maxBatchSize = mBaseCache.getKVCacheManager().getConfig().maxBatchSize;
    if (runsPairedDraftWorkingSet(impl))
    {
        maxBatchSize = std::min(maxBatchSize, mDraftCache->getKVCacheManager().getConfig().maxBatchSize);
    }
    ELLM_CHECK(impl.sequences.size() < static_cast<size_t>(maxBatchSize),
        "Context cache sequence admission exceeds the engine batch size");
    ELLM_CHECK(admission.resident.slot >= 0 && admission.resident.slot < maxBatchSize,
        "Context cache sequence admission resident slot is outside the engine range");
    ELLM_CHECK(admission.resident.epoch != 0, "Context cache sequence admission resident epoch is invalid");
    ELLM_CHECK(std::none_of(impl.sequences.begin(), impl.sequences.end(),
                   [&](auto const& sequence) { return sequence.resident.slot == admission.resident.slot; }),
        "Context cache sequence admission resident slot is already in use");

    validateSequencePolicies(admission);

    impl.sequences.reserve(impl.sequences.size() + 1);
    AdmitSequenceResult refusal;
    if (!acquireIntoRequest(impl, admission, headroom, refusal))
    {
        return refusal;
    }

    impl.sequences.back().admissionState = RequestHandle::Impl::AdmissionState::kAdmitting;

    // Bind exactly the new row -- in the draft engine's page table too, for a paired speculative lease; resident
    // rows keep serving the running batch. The device KV length for the slot is the runtime's business -- its
    // seating sequence owns the cache-manager view.
    auto const& sequence = impl.sequences.back();
    int32_t const residentSlot = sequence.resident.slot;
    int32_t const reuseTokenLength = sequence.reuseTokenLength;
    try
    {
        auto const& pages = sequence.lease.basePages();
        mBasePageTable.setRows({KVPageTableRowUpdate{
            residentSlot, pages.empty() ? nullptr : pages.data(), static_cast<int32_t>(pages.size())}});
        mBasePageTable.upload(impl.stream);
        if (runsPairedDraftWorkingSet(impl))
        {
            auto const& draftPages = sequence.lease.draftPages();
            mDraftPageTable->setRows({KVPageTableRowUpdate{residentSlot,
                draftPages.empty() ? nullptr : draftPages.data(), static_cast<int32_t>(draftPages.size())}});
            mDraftPageTable->upload(impl.stream);
        }
    }
    catch (...)
    {
        impl.sequences.pop_back();
        try
        {
            clearSequenceRows(impl, residentSlot);
        }
        catch (...)
        {
            mPoisoned = true;
        }
        throw;
    }
    return AdmitSequenceResult{ContextCacheCoordinatorStatus::kOk, false, reuseTokenLength};
}

bool ContextCacheCoordinator::supportsLiveSequenceAdmission() const noexcept
{
    return !mProfile.usesBoundedSwa() && deploymentHasAttention();
}

int32_t ContextCacheCoordinator::maxPrefillLength() const noexcept
{
    return mMaxPrefillLength;
}

void ContextCacheCoordinator::clearSequenceRows(RequestHandle::Impl& request, int32_t residentSlot)
{
    mBasePageTable.setRows({KVPageTableRowUpdate{residentSlot, nullptr, 0}});
    mBasePageTable.upload(request.stream);
    if (runsPairedDraftWorkingSet(request))
    {
        mDraftPageTable->setRows({KVPageTableRowUpdate{residentSlot, nullptr, 0}});
        mDraftPageTable->upload(request.stream);
    }
}

void ContextCacheCoordinator::restoreHybridSequence(RequestHandle::Impl& request, int32_t sequenceIndex)
{
    ELLM_CHECK(
        usesCheckpointReuse() && mHybridSnapshots != nullptr, "Hybrid sequence restore requires checkpoint storage");
    ELLM_CHECK(sequenceIndex >= 0 && sequenceIndex < static_cast<int32_t>(request.sequences.size()),
        "Hybrid sequence restore slot is outside the active batch");
    auto& sequence = request.sequences[static_cast<size_t>(sequenceIndex)];
    std::optional<int32_t> const recurrentSnapshot = sequence.lease.recurrentSnapshotSlot();
    if (!recurrentSnapshot.has_value())
    {
        mHybridSnapshots->zeroRecurrent(sequence.resident.slot, request.stream);
        return;
    }

    mHybridSnapshots->restoreRecurrent(*recurrentSnapshot, sequence.resident.slot, request.stream);
    if (std::optional<int32_t> const partialSnapshot = sequence.lease.partialKvSnapshotSlot();
        partialSnapshot.has_value())
    {
        bool const isMtp = isSpecRequest(request) && mProfile.isHybrid();
        // Hybrid+MTP keeps the boundary token in a private page so the fold can rewrite draft KV.
        size_t const destinationIndex = isMtp ? static_cast<size_t>((sequence.reuseTokenLength - 1) / kTOKENS_PER_PAGE)
                                              : static_cast<size_t>(sequence.reuseTokenLength / kTOKENS_PER_PAGE);
        int32_t const validTokenCount
            = sequence.reuseTokenLength - static_cast<int32_t>(destinationIndex) * kTOKENS_PER_PAGE;
        ELLM_CHECK(validTokenCount > 0 && destinationIndex < sequence.lease.basePages().size(),
            "Hybrid partial-KV restore has no private destination page");
        if (isMtp)
        {
            ELLM_CHECK(destinationIndex < sequence.lease.draftPages().size(),
                "Hybrid MTP partial-KV restore has no private draft destination page");
            mHybridSnapshots->restorePartialKv(*partialSnapshot, sequence.lease.basePages()[destinationIndex],
                sequence.lease.draftPages()[destinationIndex], validTokenCount, request.stream);
        }
        else
        {
            mHybridSnapshots->restorePartialKv(
                *partialSnapshot, sequence.lease.basePages()[destinationIndex], validTokenCount, request.stream);
        }
    }
    ++mMetrics.hybridRestores;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::prepareSequenceAdmission(
    RequestHandle& request, int32_t sequenceIndex)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.executing() && impl.pendingCompactionBatchSize < 0,
        "Context cache sequence-admission preparation requires an executing request outside a batch compaction");
    auto& sequence = impl.sequenceInAdmission(sequenceIndex, "preparation");
    ELLM_CHECK(!sequence.admissionChunkOpen,
        "Context cache sequence-admission preparation found the previous prefill chunk unresolved");
    sequence.admissionChunkOpen = true;
    if (!sequence.admissionCheckpointRestored)
    {
        sequence.admissionCheckpointRestored = true;
        if (usesCheckpointReuse())
        {
            restoreHybridSequence(impl, sequenceIndex);
        }
    }
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::enqueueSequenceAdmissionCapture(
    RequestHandle& request, int32_t sequenceIndex)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.executing(), "Context cache sequence-admission capture requires an executing request");
    auto& sequence = impl.sequenceInAdmission(sequenceIndex, "capture");
    ELLM_CHECK(sequence.admissionChunkOpen, "Context cache sequence-admission capture requires an open prefill chunk");
    // A speculative sequence's checkpoint pairs base and draft state, so it is captured once the draft has seated
    // the prompt, through publishHybridSpecEndpoint.
    if (usesCheckpointReuse() && !isSpecRequest(impl))
    {
        reserveHybridCapture(
            impl, sequenceIndex, static_cast<int32_t>(sequence.tokenIds.size()), PublicationPoint::kPrefillEnd);
        enqueueHybridCapture(impl, sequenceIndex);
    }
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::completeSequenceAdmissionDeviceWork(
    RequestHandle& request, int32_t sequenceIndex)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.executing(), "Context cache sequence-admission completion requires an executing request");
    auto& sequence = impl.sequenceInAdmission(sequenceIndex, "completion");
    ELLM_CHECK(
        sequence.admissionChunkOpen, "Context cache sequence-admission completion requires an open prefill chunk");
    sequence.admissionChunkOpen = false;
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::advanceSequenceAdmission(
    RequestHandle& request, int32_t sequenceIndex, int32_t committedStateLength)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.executing(), "Context cache sequence-admission advance requires an executing request");
    auto& sequence = impl.sequenceInAdmission(sequenceIndex, "advance");
    ELLM_CHECK(!sequence.admissionChunkOpen, "Context cache sequence-admission advance has unresolved device work");
    if (mPoisoned)
    {
        return ContextCacheCoordinatorStatus::kPoisoned;
    }
    // The final chunk commits through finalizeSequenceAdmission, which also records its lookahead token.
    ELLM_CHECK(committedStateLength > sequence.committedStateLength
            && committedStateLength < static_cast<int32_t>(sequence.tokenIds.size()),
        "Context cache non-final prefill chunk must advance strictly inside the prompt");
    sequence.committedStateLength = committedStateLength;
    sequence.commonStateLength = committedStateLength;
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::abortSequenceAdmission(
    RequestHandle& request, int32_t sequenceIndex)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.executing(), "Context cache sequence-admission abort requires an executing request");
    auto& sequence = impl.sequenceInAdmission(sequenceIndex, "abort");
    // Only an open chunk can still be reading the restored snapshot or writing the staged capture.
    if (sequence.admissionChunkOpen)
    {
        ContextCacheCoordinatorStatus const syncStatus = synchronizeRequest(request);
        if (syncStatus != ContextCacheCoordinatorStatus::kOk)
        {
            return syncStatus;
        }
    }

    sequence.admissionChunkOpen = false;
    sequence.admissionState = RequestHandle::Impl::AdmissionState::kAborted;
    sequence.stagedHybridPublication.reset();
    if (usesCheckpointReuse())
    {
        mManager.releaseRestoredHybridSnapshots(sequence.lease);
    }
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::finalizeSequenceAdmission(RequestHandle& request,
    int32_t sequenceIndex, ContextCacheSequenceAdvance const& advance, std::optional<int32_t> commonStateLength)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.executing(), "Context cache sequence-admission finalization requires an executing request");
    auto& sequence = impl.sequenceInAdmission(sequenceIndex, "finalization");
    ELLM_CHECK(
        !sequence.admissionChunkOpen, "Context cache sequence-admission finalization has unresolved device work");
    if (mPoisoned)
    {
        return ContextCacheCoordinatorStatus::kPoisoned;
    }
    // The single-slot form of finalizePrefillPublication's contract: the seated prefill computed
    // the complete input and sampled one lookahead token. Recording that token here is what keeps
    // the committed-plus-lookahead invariant -- and therefore every published block hash -- aligned
    // with what the pages actually contain.
    ELLM_CHECK(advance.acceptedTokenCount == 1 && advance.acceptedTokenIds != nullptr
            && advance.committedStateLength == static_cast<int32_t>(sequence.tokenIds.size()),
        "Context cache sequence-admission progress does not describe a complete input plus one lookahead");
    // A speculative sequence's draft may trail its prompt: the seated prefill leaves the last prompt position to
    // the next decode step, as a resident's accepted span.
    int32_t const commonLength = commonStateLength.value_or(advance.committedStateLength);
    ELLM_CHECK(commonLength >= sequence.commonStateLength && commonLength <= advance.committedStateLength
            && (isSpecRequest(impl) || commonLength == advance.committedStateLength),
        "Context cache sequence-admission common state is outside the committed prompt");
    sequence.tokenIds.push_back(*advance.acceptedTokenIds);
    sequence.committedStateLength = advance.committedStateLength;
    sequence.commonStateLength = commonLength;
    sequence.admissionState = RequestHandle::Impl::AdmissionState::kSettled;
    impl.publicationPolicy->onSequenceAdmissionFinalized(request, sequenceIndex);
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinator::RequestHandle::Impl& ContextCacheCoordinator::checkedImpl(RequestHandle& request) const
{
    ELLM_CHECK(request.mImpl != nullptr && request.mImpl->owner == this,
        "Context cache request handle is invalid or belongs to another coordinator");
    return *request.mImpl;
}

ContextCacheCoordinator::RequestHandle::Impl const& ContextCacheCoordinator::checkedImpl(
    RequestHandle const& request) const
{
    ELLM_CHECK(request.mImpl != nullptr && request.mImpl->owner == this,
        "Context cache request handle is invalid or belongs to another coordinator");
    return *request.mImpl;
}

bool ContextCacheCoordinator::isExternallyOwnedSwaPage(RequestHandle const& request, int32_t slot, PageId page) const
{
    RequestHandle::Impl const& impl = checkedImpl(request);
    ELLM_CHECK(mProfile.usesBoundedSwa() && slot >= 0 && slot < static_cast<int32_t>(impl.sequences.size()),
        "Context-cache SWA ownership query has an invalid slot or deployment");
    return mManager.isExternallyOwnedSwaPage(impl.sequences[static_cast<size_t>(slot)].lease, page);
}

PageId ContextCacheCoordinator::replaceSwaPage(RequestHandle& request, int32_t slot, PageId page)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(mProfile.usesBoundedSwa() && slot >= 0 && slot < static_cast<int32_t>(impl.sequences.size()),
        "Context-cache SWA retirement has an invalid slot or deployment");
    return mManager.replaceSwaPage(impl.sequences[static_cast<size_t>(slot)].lease, page);
}

void ContextCacheCoordinator::validateSwaWindowStates(RequestHandle::Impl const& request,
    std::vector<ContextCacheSequenceAdvance> const& advances, SwaWindowStateView const* swaWindowStates) const
{
    if (!mProfile.usesBoundedSwa())
    {
        ELLM_CHECK(swaWindowStates == nullptr || swaWindowStates->empty(),
            "Non-SWA context cache completion supplied SWA state");
        return;
    }
    ELLM_CHECK(swaWindowStates != nullptr && swaWindowStates->size() == request.sequences.size(),
        "Bounded-SWA context cache completion requires one settled state per sequence");
    ELLM_CHECK(advances.size() == request.sequences.size(),
        "Bounded-SWA context cache completion requires one sequence advance per settled state");
    for (size_t slot = 0; slot < request.sequences.size(); ++slot)
    {
        ELLM_CHECK((*swaWindowStates)[slot].retentionMode == SwaRetentionMode::kLive
                && (*swaWindowStates)[slot].exactResidentTokenCount == advances[slot].committedStateLength,
            "Bounded-SWA context cache completion supplied state for a different endpoint");
    }
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
            rows.push_back(KVPageTableRowUpdate{
                sequence.resident.slot, pages.empty() ? nullptr : pages.data(), static_cast<int32_t>(pages.size())});
        }
        if (runsPairedDraftWorkingSet(impl))
        {
            auto const& pages = sequence.lease.draftPages();
            draftRows.push_back(KVPageTableRowUpdate{
                sequence.resident.slot, pages.empty() ? nullptr : pages.data(), static_cast<int32_t>(pages.size())});
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
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            restoreHybridSequence(impl, static_cast<int32_t>(slot));
        }
    }
    mBaseCache.resetForNewSequences(*mHostReuseLengths, impl.stream);
    // Any paired-draft request (EAGLE or Hybrid+spec) must reset the draft cache's per-request active batch size / KV
    // lengths here too; otherwise the draft cache keeps the previous request's compacted batch size and the draft-side
    // commitSequenceLength fails on the next request.
    if (runsPairedDraftWorkingSet(impl))
    {
        mDraftCache->resetForNewSequences(*mHostReuseLengths, impl.stream);
    }
    if (impl.sequences.empty())
    {
        // Nothing to prefill, so nothing will report a prefill's completion: the request is
        // settled the moment it enters execution. The length reset enqueued above is ordered
        // ahead of every later step on the same stream, which is all a settled request needs.
        impl.markDeviceWorkResolvedWithoutSync();
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
        if (delta.acceptedTokenCount == 0
            && delta.committedStateLength == ContextCacheSequenceAdvance::kHoldCommittedStateLength)
        {
            continue;
        }
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
        if (delta.acceptedTokenCount == 0
            && delta.committedStateLength == ContextCacheSequenceAdvance::kHoldCommittedStateLength)
        {
            continue;
        }
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
        logHybridSnapshotPressureSkip(
            mManager, mMetrics.hybridSnapshotPressureSkips, slot, sequence.resident.slot, exactLength);
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
    for (int32_t slot = 0; slot < static_cast<int32_t>(request.sequences.size()); ++slot)
    {
        enqueueHybridCapture(request, slot);
    }
}

void ContextCacheCoordinator::enqueueHybridCapture(RequestHandle::Impl& request, int32_t slot)
{
    ELLM_CHECK(usesCheckpointReuse() && mHybridSnapshots != nullptr,
        "Hybrid context capture requires a recurrent deployment and snapshot storage");
    auto& sequence = request.sequences[static_cast<size_t>(slot)];
    if (!sequence.stagedHybridPublication.has_value())
    {
        return;
    }
    auto const& staged = *sequence.stagedHybridPublication;
    mHybridSnapshots->captureRecurrent(staged.snapshots.recurrentSnapshotSlot, sequence.resident.slot, request.stream);
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

void ContextCacheCoordinator::publishReadyHybridEndpoints(RequestHandle::Impl& request)
{
    for (int32_t slot = 0; slot < static_cast<int32_t>(request.sequences.size()); ++slot)
    {
        publishReadyHybridEndpoint(request, slot);
    }
}

void ContextCacheCoordinator::publishReadyHybridEndpoint(RequestHandle::Impl& request, int32_t slot)
{
    auto& sequence = request.sequences[static_cast<size_t>(slot)];
    if (!sequence.stagedHybridPublication.has_value())
    {
        return;
    }
    auto const staged = *sequence.stagedHybridPublication;
    ELLM_CHECK(staged.checkpoint.exactLength <= sequence.committedStateLength,
        "Hybrid context snapshot exceeds the ready committed boundary");
    size_t const fullTokenCount
        = static_cast<size_t>(staged.checkpoint.exactLength / kTOKENS_PER_PAGE) * static_cast<size_t>(kTOKENS_PER_PAGE);
    Hash128 const* seqMediaPtr = sequence.perPositionMediaHash.empty() ? nullptr : sequence.perPositionMediaHash.data();
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

void ContextCacheCoordinator::publishReadyEndpoint(
    RequestHandle::Impl& request, int32_t slot, PublicationPoint point, SwaWindowState const* swaWindowState)
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

    ELLM_CHECK(mProfile.usesBoundedSwa() == (swaWindowState != nullptr),
        "Context cache endpoint has inconsistent bounded-SWA state");

    size_t const tokenCount = static_cast<size_t>(publishBlocks) * static_cast<size_t>(kTOKENS_PER_PAGE);
    Hash128 const* pubMediaPtr = sequence.perPositionMediaHash.empty() ? nullptr : sequence.perPositionMediaHash.data();
    std::vector<BlockHash> hashes
        = hashRequestFullBlocks(sequence.tokenIds.data(), tokenCount, sequence.keyExtras, pubMediaPtr);
    std::optional<SwaWindowState> settledSwaState;
    if (swaWindowState != nullptr)
    {
        settledSwaState = *swaWindowState;
    }
    PublishResult const result = mManager.publish(
        sequence.lease, PublishRequest{std::move(hashes), sequence.committedStateLength, std::move(settledSwaState)});
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

void ContextCacheCoordinator::publishFrozenSpecPrefill(
    RequestHandle::Impl& request, std::vector<int32_t> const* oldToNew)
{
    ELLM_CHECK(usesFrozenSpecPublication(request), "Frozen speculative publication requires an EAGLE request");
    for (int32_t slot = 0; slot < static_cast<int32_t>(request.sequences.size()); ++slot)
    {
        auto& sequence = request.sequences[static_cast<size_t>(slot)];
        bool const retiring = oldToNew == nullptr || (*oldToNew)[static_cast<size_t>(slot)] < 0;
        if (!retiring || !sequence.frozenSpecPrefillLength.has_value())
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
        // A slot cancelled at the top of a step appends nothing that step and holds its ledger, as
        // validateCommittedLookaheadAdvances accepts for the other speculative flavors.
        if (delta.acceptedTokenCount == 0)
        {
            ELLM_CHECK(delta.committedStateLength == ContextCacheSequenceAdvance::kHoldCommittedStateLength,
                "Context cache zero advance must carry the hold sentinel");
            continue;
        }
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

void ContextCacheCoordinator::validateFullyCommittedDecodeAdvances(RequestHandle::Impl const& request,
    std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const* commonStateLengths) const
{
    ELLM_CHECK(commonStateLengths == nullptr || commonStateLengths->empty(),
        "Fully committed decode supplied speculative common-state lengths");
    for (size_t slot = 0; slot < request.sequences.size(); ++slot)
    {
        auto const& sequence = request.sequences[slot];
        auto const& delta = advances[slot];
        int64_t const expectedCommittedStateLength
            = static_cast<int64_t>(sequence.committedStateLength) + static_cast<int64_t>(delta.acceptedTokenCount);
        int64_t const expectedTokenCount
            = static_cast<int64_t>(sequence.tokenIds.size()) + static_cast<int64_t>(delta.acceptedTokenCount);
        ELLM_CHECK(delta.acceptedTokenCount > 0 && expectedCommittedStateLength <= std::numeric_limits<int32_t>::max()
                && delta.committedStateLength == expectedCommittedStateLength
                && expectedTokenCount == expectedCommittedStateLength,
            "Fully committed decode progress must append exactly its committed token prefix without a lookahead token");
    }
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
    // A speculative step holds tokenIds as the committed prefix plus one lookahead token and commits everything it
    // accepted. A vanilla step feeds every token the ledger holds beyond the committed state -- one, or the run a
    // jump-forward appended -- so it commits up to the previous token count and appends at least one token.
    bool const multiToken = tokensPerStep == TokensPerDecodeStep::kAcceptedCount;
    for (size_t slot = 0; slot < request.sequences.size(); ++slot)
    {
        auto const& sequence = request.sequences[slot];
        auto const& delta = advances[slot];
        // A slot cancelled at the top of a step appends nothing that step: a zero advance holding
        // the committed length in place is the legal way to say so. The pages stay leased and
        // consistent; the slot leaves through the ordinary eviction after this step.
        if (delta.acceptedTokenCount == 0)
        {
            ELLM_CHECK(delta.committedStateLength == ContextCacheSequenceAdvance::kHoldCommittedStateLength,
                "Context cache zero advance must carry the hold sentinel");
            continue;
        }
        int64_t const tokenCount = static_cast<int64_t>(sequence.tokenIds.size());
        int64_t const expectedCommittedStateLength = multiToken
            ? static_cast<int64_t>(sequence.committedStateLength) + static_cast<int64_t>(delta.acceptedTokenCount)
            : tokenCount;
        ELLM_CHECK(delta.acceptedTokenCount > 0 && expectedCommittedStateLength <= std::numeric_limits<int32_t>::max()
                && delta.committedStateLength == expectedCommittedStateLength
                && (multiToken ? tokenCount == static_cast<int64_t>(sequence.committedStateLength) + 1
                               : tokenCount >= static_cast<int64_t>(sequence.committedStateLength) + 1),
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
    std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const* commonStateLengths,
    SwaWindowStateView const* swaWindowStates)
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
        if (impl.tokenStateContract == DecodingTokenStateContract::kFullyCommitted)
        {
            ELLM_CHECK(delta.acceptedTokenCount == 0
                    && delta.committedStateLength == static_cast<int32_t>(sequence.tokenIds.size()),
                "Fully committed prefill must publish only its materialized prompt boundary");
        }
        else
        {
            ELLM_CHECK(delta.acceptedTokenCount == 1
                    && delta.committedStateLength == static_cast<int32_t>(sequence.tokenIds.size()),
                "Context cache prefill progress does not describe a complete input plus one lookahead");
        }
    }
    validateSwaWindowStates(impl, advances, swaWindowStates);
    impl.publicationPolicy->onPrefillFinalized(request, advances, commonStateLengths, swaWindowStates);
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::publishHybridSpecEndpoint(RequestHandle& request, int32_t slot,
    int32_t residentStateLength, Tensor const& baseHiddenStates, int32_t boundaryHiddenRow)
{
    checkedImpl(request);
    ELLM_CHECK(mProfile.isHybrid() && mProfile.isSpeculative() && mHybridSnapshots != nullptr,
        "Hybrid+spec endpoint publication requires a Hybrid+spec deployment with snapshot storage");
    return checkedImpl(request).publicationPolicy->publishMtpBoundary(
        request, slot, residentStateLength, baseHiddenStates, boundaryHiddenRow);
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::restoreHybridSpecBoundaryHidden(
    RequestHandle& request, int32_t slot, Tensor& baseHiddenStates, int32_t destinationRow)
{
    checkedImpl(request);
    ELLM_CHECK(mProfile.isHybrid() && mProfile.isSpeculative() && mHybridSnapshots != nullptr,
        "Hybrid+spec boundary-hidden restore requires a Hybrid+spec deployment with snapshot storage");
    return checkedImpl(request).publicationPolicy->restoreMtpBoundary(request, slot, baseHiddenStates, destinationRow);
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::prepareDecodeStep(RequestHandle& request,
    DecodingKvHeadroom const& headroom, std::vector<int32_t> const* extraTokensPerSlot,
    std::vector<int32_t> const* draftExtraTokensPerSlot)
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
    ELLM_CHECK((extraTokensPerSlot == nullptr || extraTokensPerSlot->size() == impl.sequences.size())
            && (draftExtraTokensPerSlot == nullptr || draftExtraTokensPerSlot->size() == impl.sequences.size()),
        "Per-slot decode growth needs one count per sequence");
    impl.decodeParticipants.clear();

    // Paired speculative leases grow base and independent-draft working sets atomically and upload both page tables.
    // Hybrid+spec uses the same path because growBasePages rejects its paired lease.
    if (runsPairedDraftWorkingSet(impl))
    {
        struct PageDemand
        {
            int32_t basePages{};
            int32_t draftPages{};
        };
        std::vector<PageDemand> demands;
        demands.reserve(impl.sequences.size());
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            auto const& sequence = impl.sequences[slot];
            int32_t const baseExtraTokens
                = extraTokensPerSlot != nullptr ? (*extraTokensPerSlot)[slot] : headroom.baseExtraTokens;
            int32_t const draftExtraTokens
                = draftExtraTokensPerSlot != nullptr ? (*draftExtraTokensPerSlot)[slot] : headroom.draftExtraTokens;
            ELLM_CHECK(
                baseExtraTokens > 0 && draftExtraTokens > 0, "Per-slot speculative decode growth must be positive");
            ELLM_CHECK(sequence.committedStateLength <= std::numeric_limits<int32_t>::max() - baseExtraTokens
                    && sequence.committedStateLength <= std::numeric_limits<int32_t>::max() - draftExtraTokens,
                "Context-cache working-set length overflow before decode");
            int32_t const basePages = pageCountForStateLength(sequence.committedStateLength + baseExtraTokens);
            int32_t const draftPages = pageCountForStateLength(sequence.committedStateLength + draftExtraTokens);
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
                    sequence.resident.slot, basePages.data(), static_cast<int32_t>(basePages.size())});
            }
            if (draftGrowth > 0)
            {
                auto const& draftPages = sequence.lease.draftPages();
                draftRows.push_back(KVPageTableRowUpdate{
                    sequence.resident.slot, draftPages.data(), static_cast<int32_t>(draftPages.size())});
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
    ELLM_CHECK(draftExtraTokensPerSlot == nullptr, "Per-slot draft decode growth needs a paired draft working set");
    return growDecodeWorkingSet(impl, headroom, extraTokensPerSlot);
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::prepareDecodeStep(
    RequestHandle& request, DecodingKvHeadroom const& headroom, std::vector<int32_t> const& decodeSequences)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(!isSpecRequest(impl), "Context cache decode-subset preparation is limited to non-speculative requests");
    ELLM_CHECK(impl.executing() && !impl.hasPendingRequestWork() && impl.pendingCompactionBatchSize < 0,
        "Context cache decode-subset preparation requires terminal prior request-wide work");
    ELLM_CHECK(headroom.baseExtraTokens > 0 && headroom.draftExtraTokens == 0,
        "Decoder KV headroom is outside the supported range");
    std::vector<uint8_t> participants(impl.sequences.size(), 0U);
    for (int32_t const sequenceIndex : decodeSequences)
    {
        ELLM_CHECK(sequenceIndex >= 0 && sequenceIndex < static_cast<int32_t>(impl.sequences.size()),
            "Context cache decode sequence is outside the batch");
        ELLM_CHECK(participants[static_cast<size_t>(sequenceIndex)] == 0U,
            "Context cache decode sequence appears more than once");
        ELLM_CHECK(impl.sequences[static_cast<size_t>(sequenceIndex)].admissionState
                == RequestHandle::Impl::AdmissionState::kSettled,
            "Context cache decode sequence has not committed its prompt");
        participants[static_cast<size_t>(sequenceIndex)] = 1U;
    }
    impl.decodeParticipants = std::move(participants);
    return growDecodeWorkingSet(impl, headroom, /*extraTokensPerSlot=*/nullptr);
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::growDecodeWorkingSet(
    RequestHandle::Impl& impl, DecodingKvHeadroom const& headroom, std::vector<int32_t> const* extraTokensPerSlot)
{
    for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
    {
        ELLM_CHECK(!impl.decodesThisStep(slot)
                || impl.sequences[slot].committedStateLength < std::numeric_limits<int32_t>::max(),
            "Context cache sequence length overflow before decode");
    }
    if (deploymentHasAttention())
    {
        std::vector<KVPageTableRowUpdate> rows;
        rows.reserve(impl.sequences.size());
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            if (!impl.decodesThisStep(slot))
            {
                continue;
            }
            auto& sequence = impl.sequences[slot];
            int32_t const baseExtraTokens
                = extraTokensPerSlot != nullptr ? (*extraTokensPerSlot)[slot] : headroom.baseExtraTokens;
            ELLM_CHECK(baseExtraTokens > 0, "Per-slot decode growth must be positive");
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
                    KVPageTableRowUpdate{sequence.resident.slot, pages.data(), static_cast<int32_t>(pages.size())});
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
    std::vector<int32_t> const* commonStateLengths, SwaWindowStateView const* swaWindowStates)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.executing() && impl.hasPendingRequestWork(),
        "Context cache decode completion requires pending decode work");
    ELLM_CHECK(
        advances.size() == impl.sequences.size(), "Context cache decode advances must describe every active sequence");
    for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
    {
        bool const holds = advances[slot].acceptedTokenCount == 0
            && advances[slot].committedStateLength == ContextCacheSequenceAdvance::kHoldCommittedStateLength;
        ELLM_CHECK(holds
                || (impl.decodesThisStep(slot)
                    && impl.sequences[slot].admissionState == RequestHandle::Impl::AdmissionState::kSettled),
            "Context cache decode advance names a sequence that did not decode this step");
    }
    for (int32_t const slot : publishableCompletedSlots)
    {
        ELLM_CHECK(slot >= 0 && slot < static_cast<int32_t>(impl.sequences.size())
                && impl.decodesThisStep(static_cast<size_t>(slot))
                && impl.sequences[static_cast<size_t>(slot)].admissionState
                    == RequestHandle::Impl::AdmissionState::kSettled,
            "Context cache decode publication names a sequence that did not decode this step");
    }
    impl.decodeParticipants.clear();
    validateSwaWindowStates(impl, advances, swaWindowStates);
    return impl.publicationPolicy->onDecodeCompleted(
        request, advances, publishableCompletedSlots, commonStateLengths, swaWindowStates);
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::beginBatchCompaction(
    RequestHandle& request, std::vector<int32_t> const& oldToNew, int32_t newBatchSize, Tensor& deviceBatchMapping)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    int32_t const oldBatchSize = static_cast<int32_t>(impl.sequences.size());
    validateCompactionMapping(oldToNew, oldBatchSize, newBatchSize);
    // EAGLE finalizes its two-phase draft init, and publishes the frozen endpoints of retiring seated sequences,
    // before compaction; other flavors no-op.
    ContextCacheCoordinatorStatus const terminalStatus = impl.publicationPolicy->onCompaction(request, oldToNew);
    if (terminalStatus != ContextCacheCoordinatorStatus::kOk)
    {
        return terminalStatus;
    }
    ELLM_CHECK(impl.executing() && !impl.hasPendingDeviceWork(),
        "Context cache compaction preparation requires terminal model work");
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

    for (int32_t oldSlot = 0; oldSlot < oldBatchSize; ++oldSlot)
    {
        auto const& sequence = impl.sequences[static_cast<size_t>(oldSlot)];
        bool const retiring = oldToNew[static_cast<size_t>(oldSlot)] < 0;
        ELLM_CHECK(!sequence.stagedHybridPublication.has_value()
                && !(retiring && sequence.frozenSpecPrefillLength.has_value()),
            "Context cache cannot compact a batch with an unpublished staged endpoint");
    }

    std::vector<int32_t> retiredResidentSlots;
    for (int32_t oldSlot = 0; oldSlot < oldBatchSize; ++oldSlot)
    {
        if (oldToNew[static_cast<size_t>(oldSlot)] < 0)
        {
            retiredResidentSlots.push_back(impl.sequences[static_cast<size_t>(oldSlot)].resident.slot);
        }
    }

    if (deploymentHasAttention())
    {
        std::vector<KVPageTableRowUpdate> clearedRows;
        clearedRows.reserve(retiredResidentSlots.size());
        for (int32_t const residentSlot : retiredResidentSlots)
        {
            clearedRows.push_back(KVPageTableRowUpdate{residentSlot, nullptr, 0});
        }
        mBasePageTable.setRows(clearedRows);
        mBasePageTable.upload(impl.stream);
    }
    // Any request that grew a paired draft page path at decode (EAGLE or Hybrid+spec) must compact the draft page table
    // and draft cache alongside the base side; otherwise the draft rows/state desynchronize from the survivor batch.
    if (runsPairedDraftWorkingSet(impl))
    {
        std::vector<KVPageTableRowUpdate> clearedRows;
        clearedRows.reserve(retiredResidentSlots.size());
        for (int32_t const residentSlot : retiredResidentSlots)
        {
            clearedRows.push_back(KVPageTableRowUpdate{residentSlot, nullptr, 0});
        }
        mDraftPageTable->setRows(clearedRows);
        mDraftPageTable->upload(impl.stream);
    }
    mBaseCache.compactKVCacheLengths(deviceBatchMapping, oldBatchSize, newBatchSize, impl.stream);
    if (runsPairedDraftWorkingSet(impl))
    {
        mDraftCache->compactKVCacheLengths(deviceBatchMapping, oldBatchSize, newBatchSize, impl.stream);
    }
    for (int32_t const residentSlot : retiredResidentSlots)
    {
        mBaseCache.clearResidentSlot(residentSlot, impl.stream);
        if (runsPairedDraftWorkingSet(impl))
        {
            mDraftCache->clearResidentSlot(residentSlot, impl.stream);
        }
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
    // A batch compacted to nothing leaves the request executing and settled, not finished: whether
    // it is done is the caller's to say through finish(), and a request that keeps admitting
    // sequences after its last resident retired is the ordinary shape of a long-lived session.
    return ContextCacheCoordinatorStatus::kOk;
}

void ContextCacheCoordinator::quarantine(RequestHandle& request) noexcept
{
    mPoisoned = true;
    mQuarantinedRequests.push_back(std::move(request.mImpl));
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::finish(RequestHandle& request)
{
    if (!request.valid())
    {
        return ContextCacheCoordinatorStatus::kOk;
    }
    // EAGLE finalizes its two-phase draft init at completion; other flavors no-op.
    ContextCacheCoordinatorStatus const terminalStatus = checkedImpl(request).publicationPolicy->onTerminalize(request);
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
    // A seated sequence's draft prefill may still be writing the pages its lease holds.
    if (!request->hasPendingDeviceWork() && !request->hasUnsynchronizedSeatedPrefill())
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
    mQuarantinedRequests.push_back(std::move(request));
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::shutdown() noexcept
{
    while (!mQuarantinedRequests.empty())
    {
        std::unique_ptr<RequestHandle::Impl>& quarantined = mQuarantinedRequests.back();
        cudaError_t status{cudaErrorUnknown};
        try
        {
            status = mSynchronizer(quarantined->stream);
        }
        catch (...)
        {
            status = cudaErrorUnknown;
        }
        if (status != cudaSuccess)
        {
            return ContextCacheCoordinatorStatus::kPoisoned;
        }
        quarantined->markDeviceWorkSynchronized();
        mQuarantinedRequests.pop_back();
    }
    if (mActiveRequests != 0)
    {
        return ContextCacheCoordinatorStatus::kRequestFailed;
    }
    return ContextCacheCoordinatorStatus::kOk;
}

int32_t ContextCacheCoordinator::activeRequests() const noexcept
{
    return mActiveRequests;
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
    result.swaKvPages
        = ContextCachePoolMetrics{pools.freeCount(ResourceType::kSwaKvPage), pools.capacity(ResourceType::kSwaKvPage)};
    ContextCacheManagerMetrics const& managerMetrics = mManager.metrics();
    result.evictedRecords = managerMetrics.evictedRecords;
    result.reclaimedBaseKvPages = managerMetrics.reclaimedBaseKvPages;
    result.reclaimedDraftKvPages = managerMetrics.reclaimedDraftKvPages;
    result.reclaimedRecurrentSnapshots = managerMetrics.reclaimedRecurrentSnapshots;
    result.reclaimedPartialKvSnapshots = managerMetrics.reclaimedPartialKvSnapshots;
    result.reclaimedSwaKvPages = managerMetrics.reclaimedSwaKvPages;
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

ContextCacheManager& ContextCacheCoordinator::manager() noexcept
{
    return mManager;
}

} // namespace rt
} // namespace trt_edgellm
