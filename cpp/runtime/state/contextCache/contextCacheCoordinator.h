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

#pragma once

#include "runtime/state/contextCache/contextCacheConfig.h"
#include "runtime/state/contextCache/contextCacheDeployment.h"
#include "runtime/state/contextCache/contextCacheManager.h"
#include "runtime/state/contextCache/contextCacheMetrics.h"
#include "runtime/state/decodingInferenceContext.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class HybridCacheManager;
class HybridSnapshotStorage;
class KVPageTable;
class Tensor;

//! Validated physical resources borrowed from SharedResources for the coordinator lifetime.
struct ContextCachePhysicalResources
{
    HybridCacheManager& baseCache;
    KVPageTable& basePageTable;
    HybridCacheManager* draftCache{};
    KVPageTable* draftPageTable{};
    KVPageTable* swaPageTable{};
};

//! One complete logical input admitted to the cache before its executable suffix is derived.
struct ContextCacheSequenceAdmission
{
    std::vector<int32_t> tokenIds;
    //! Request-wide non-token identity; LoRA/isolation identity is constant for the sequence.
    BlockKeyExtras keyExtras;
    //! Per-position media content hash. Empty means text-only. When non-empty, must have tokenIds.size() entries.
    //! A non-zero Hash128 at position i causes the block hash to consume that 128-bit digest instead of the token ID.
    std::vector<Hash128> perPositionMediaHash;
    //! Scheduler-owned physical state identity; the coordinator may not derive this from logical sequence order.
    ResidentRef resident;
    //! Whether the sequence looks up and publishes reusable state. Bypass still uses managed private
    //! pages but neither looks up nor publishes.
    ContextCacheLookupPolicy lookupPolicy{ContextCacheLookupPolicy::kUseCache};
    //! Which ready endpoints the sequence publishes.
    ContextCacheCommitPolicy commitPolicy{ContextCacheCommitPolicy::kIncludingGeneratedTokens};
    //! Hybrid+spec replay tail length derived before cache admission.
    int32_t replayTailLength{0};
};

//! One serialized runtime request: the batch the cache steps together. Cache policies belong to the
//! sequences; the request carries only what the deployment fixes for every sequence it will hold.
//!
//! The sequences may be empty: such a request opens with no lease and takes every sequence through
//! admitSequence() while it executes.
struct ContextCacheBatchAdmission
{
    std::vector<ContextCacheSequenceAdmission> sequences;
    bool speculativeRequest{};
    DecodingTokenStateContract tokenStateContract{DecodingTokenStateContract::kCommittedPlusLookahead};
};

//! Host-visible sequence advance observed after an existing stream synchronization.
struct ContextCacheSequenceAdvance
{
    //! committedStateLength value for a zero advance that holds the ledger's current committed
    //! length in place. A concrete length with no accepted tokens represents a state-only advance.
    static constexpr int32_t kHoldCommittedStateLength = -1;

    int32_t const* acceptedTokenIds{};
    int32_t acceptedTokenCount{};
    //! Greatest logical token boundary whose model state is materialized in the bound cache.
    int32_t committedStateLength{};
};

enum class ContextCacheCoordinatorStatus : uint8_t
{
    kOk,
    kRequestFailed,
    kPoisoned,
};

//! Owns the complete host/device lifecycle around the CUDA-free ContextCacheManager.
//!
//! The coordinator has no worker of its own; every call is made by the caller's thread and callers must keep those
//! calls serialized. Request handles may overlap in lifetime: several may be admitted and hold a page lease at once,
//! which is the ownership shape in-flight batching needs.
//!
//! Overlapping leases are not overlapping execution, and the distinction is load-bearing. Page-table rows bind to
//! stable physical resident slots, while request-local reuse-length staging is reused across preparations. A prepare
//! issued before the previous request's async copy has landed would corrupt that staging buffer. A caller must
//! therefore still finish one request's steps before preparing the next; what a count rather than a flag permits is
//! that the leases, not the execution, may interleave.
//!
//! Normal publication occurs only at host-visible completion points already present in the runtime. An abnormal exit
//! drains the bound stream before releasing active page references; a failed drain quarantines the entire request and
//! poisons the coordinator until runtime-owned shutdown can establish quiescence. Poisoning gates new admissions, so
//! a request already admitted when the stream failed is quarantined by its own drain rather than by the first
//! failure.
class ContextCacheCoordinator final
{
public:
    using StreamSynchronizer = std::function<cudaError_t(cudaStream_t)>;

    class RequestHandle final
    {
    public:
        RequestHandle(RequestHandle&& other) noexcept;
        RequestHandle& operator=(RequestHandle&& other) = delete;
        RequestHandle(RequestHandle const&) = delete;
        RequestHandle& operator=(RequestHandle const&) = delete;
        ~RequestHandle() noexcept;

        bool valid() const noexcept;

    private:
        friend class ContextCacheCoordinator;
        struct Impl;

        explicit RequestHandle(std::unique_ptr<Impl> impl) noexcept;
        std::unique_ptr<Impl> mImpl;
    };

    struct AdmissionResult
    {
        RequestHandle request;
        //! Per-sequence logical token offset at which runtime prefill begins.
        std::vector<int32_t> prefillStarts;
        //! Seeded states whose page references remain owned by the corresponding context-cache leases.
        std::vector<SwaKVCacheState> swaStates;
    };

    struct BeginRequestResult
    {
        ContextCacheCoordinatorStatus status{ContextCacheCoordinatorStatus::kRequestFailed};
        std::optional<AdmissionResult> admission;
        //! Set on a refusal whose cause is a sequence's uncached suffix exceeding maxPrefillLength(): that suffix
        //! length. Such a sequence is refused before anything is leased, so no record is evicted for it.
        std::optional<int32_t> overlongPrefillLength{};
    };

    ContextCacheCoordinator(ContextCacheConfig const& config, DeploymentConfig const& deployment,
        ContextCacheDeploymentProfile profile, ContextCachePhysicalResources resources, cudaStream_t stream,
        StreamSynchronizer synchronizer = {});
    ~ContextCacheCoordinator() noexcept;
    ContextCacheCoordinator(ContextCacheCoordinator const&) = delete;
    ContextCacheCoordinator& operator=(ContextCacheCoordinator const&) = delete;

    BeginRequestResult beginRequest(
        ContextCacheBatchAdmission const& admission, DecodingKvHeadroom const& headroom, cudaStream_t stream);

    struct AdmitSequenceResult
    {
        ContextCacheCoordinatorStatus status{ContextCacheCoordinatorStatus::kRequestFailed};
        //! True when the refusal is transient pool pressure rather than an error: pages free as
        //! resident sequences retire, so the caller may retry at a later step boundary.
        bool insufficientCapacity{};
        //! Logical token offset at which the runtime's prefill begins (== the reused prefix length); on an
        //! overlongPrefill refusal, the reused length the lookup found.
        int32_t prefillStart{};
        //! Refused because the uncached suffix exceeds maxPrefillLength(); nothing was leased or evicted.
        bool overlongPrefill{};
    };

    //! @brief Join one more sequence to a live request: look up its prefix, lease its pages, and
    //!        bind exactly its page-table row (and its draft page-table row, for a paired speculative lease).
    //!        Attention deployments without bounded SWA only.
    //!
    //! The request may hold no sequence yet: an empty request admitted with no sequences takes
    //! its first one here, through the same path as every later one. The sequence inherits the
    //! request's lookup, commit and replay-tail policies unless its admission overrides the
    //! lookup policy.
    //!
    //! The device KV length for the new slot stays the runtime's business -- its seating sequence
    //! owns the cache-manager view -- and the ledger is not advanced until
    //! advanceSequenceAdmission or finalizeSequenceAdmission reports a committed prefill chunk.
    //!
    //! Other sequences may still be in admission, and their chunk work may be in flight: the
    //! lease comes from pages no enqueued work references, and the row upload is stream-ordered
    //! for a slot no enqueued step reads. Request-wide work (a decode step or a compaction) must
    //! be resolved first.
    AdmitSequenceResult admitSequence(
        RequestHandle& request, ContextCacheSequenceAdmission const& admission, DecodingKvHeadroom const& headroom);

    //! @name Sequence-admission transaction
    //! Every sequence joined through admitSequence is in admission until finalizeSequenceAdmission
    //! commits its final prefill chunk or abortSequenceAdmission discards it. Any number of
    //! sequences may be in admission at once, at any execution index, and each tracks its own
    //! in-flight device work. Per chunk the caller runs prepare -> (final chunk only: capture) ->
    //! its own stream synchronization -> complete -> advance (non-final) or finalize (final).
    //! @{

    //! Open the next prefill chunk of a sequence in admission. Before the first chunk this
    //! restores the sequence's recurrent and partial-KV checkpoint into its stable resident slot
    //! without touching any other resident sequence. @p sequenceIndex is the transient execution
    //! index.
    ContextCacheCoordinatorStatus prepareSequenceAdmission(RequestHandle& request, int32_t sequenceIndex);

    //! Reserve and enqueue the sequence's prefill-end snapshot after its final chunk and before the
    //! runtime's existing sampled-token synchronization. This is a no-op for non-hybrid deployments.
    ContextCacheCoordinatorStatus enqueueSequenceAdmissionCapture(RequestHandle& request, int32_t sequenceIndex);

    //! Resolve the sequence's open chunk after the runtime's stream synchronization. Other
    //! sequences' chunks stay open until their own completion.
    ContextCacheCoordinatorStatus completeSequenceAdmissionDeviceWork(RequestHandle& request, int32_t sequenceIndex);

    //! Record a committed non-final prefill chunk: the slot's state now covers the first
    //! @p committedStateLength prompt tokens. Nothing is published and no snapshot is captured.
    ContextCacheCoordinatorStatus advanceSequenceAdmission(
        RequestHandle& request, int32_t sequenceIndex, int32_t committedStateLength);

    //! Discard a failed or cancelled sequence-admission transaction so ordinary eviction can retire
    //! it; the eviction's compaction releases its lease. Drains the stream only when the sequence
    //! has an open chunk. The sequence never publishes afterwards. Throws for a sequence that is not
    //! in admission (finalized, aborted, or admitted with the request).
    ContextCacheCoordinatorStatus abortSequenceAdmission(RequestHandle& request, int32_t sequenceIndex);

    //! @brief The single-slot form of finalizePrefillPublication: record the final chunk's
    //!        lookahead token, advance the slot's committed prefix to the full input, and publish
    //!        its ready prefix state (the hybrid snapshot captured for this sequence only).
    //!
    //! @p commonStateLength is the prefix a speculative sequence's draft has materialized (the full input when
    //! empty). A Hybrid+spec sequence published its checkpoint during the prefill (publishHybridSpecEndpoint); an
    //! EAGLE-flavor sequence's prompt endpoint publishes at the next decode completion, capped at the common state.
    ContextCacheCoordinatorStatus finalizeSequenceAdmission(RequestHandle& request, int32_t sequenceIndex,
        ContextCacheSequenceAdvance const& advance, std::optional<int32_t> commonStateLength = std::nullopt);
    //! @}

    //! @brief Undo the most recent admitSequence before its slot ever joined the runtime batch:
    //!        release the lease, drop the sequence, and clear its page-table row.
    //!
    //! Only legal while the admitted sequence is still the batch's tail and the runtime holds no
    //! slot for it -- the recovery path for a seating that threw between lease and slot append. A
    //!        seated slot that failed later stays, terminal from birth, and leaves through the
    //!        ordinary eviction instead. Retracting from a request that holds no sequence poisons
    //!        the coordinator: nothing was admitted, so the caller's bookkeeping is already wrong.
    bool retractSequenceAdmission(RequestHandle& request) noexcept;

    //! Bind every admitted row and reset logical cache lengths to the selected reuse boundaries, entering
    //! execution. A request admitted with no sequences has no prefill to complete and is settled at once.
    ContextCacheCoordinatorStatus preparePrefill(RequestHandle& request);
    //! Reserve and enqueue hybrid prefill-end snapshots before the runtime's existing prefill synchronization.
    ContextCacheCoordinatorStatus enqueuePrefillCaptures(RequestHandle& request);
    //! Apply the post-prefill sequence advance and publish every ready full-block endpoint.
    //! For EAGLE, commonStateLengths caps publication at the prefix materialized by both base and draft state.
    ContextCacheCoordinatorStatus finalizePrefillPublication(RequestHandle& request,
        std::vector<ContextCacheSequenceAdvance> const& advances,
        std::vector<int32_t> const* commonStateLengths = nullptr, SwaWindowStateView const* swaWindowStates = nullptr);
    //! Publish one Hybrid+spec checkpoint at the stable predecessor boundary. This is the dedicated MTP publication
    //! entrypoint; the runtime drives it after the folded draft prefill has materialized the boundary draft state,
    //! for a blocking request's sequence or one in sequence admission.
    //! It captures the recurrent state, the paired base+draft partial pages, and the successor-dependent boundary
    //! base-hidden row, then commits the exact checkpoint. Skipped for bypass, already-published, or empty prefixes.
    ContextCacheCoordinatorStatus publishHybridSpecEndpoint(RequestHandle& request, int32_t slot,
        int32_t residentStateLength, Tensor const& baseHiddenStates, int32_t boundaryHiddenRow);
    //! Restore the checkpoint's saved boundary base-hidden row into baseHiddenStates[slot, destinationRow, :] for the
    //! runtime's fold micro-forward. No synchronization: the caller orders this within its prefill stream.
    ContextCacheCoordinatorStatus restoreHybridSpecBoundaryHidden(
        RequestHandle& request, int32_t slot, Tensor& baseHiddenStates, int32_t destinationRow);
    //! Grow and upload every row needed for the next decode working set before model execution.
    //! @p extraTokensPerSlot, when set, is each sequence's own growth beyond its committed state (one, or the pending
    //! run a jump-forward left; a jump-forward slot's wider verify round) and replaces headroom.baseExtraTokens.
    //! @p draftExtraTokensPerSlot likewise replaces headroom.draftExtraTokens; paired speculative requests only.
    ContextCacheCoordinatorStatus prepareDecodeStep(RequestHandle& request, DecodingKvHeadroom const& headroom,
        std::vector<int32_t> const* extraTokensPerSlot = nullptr,
        std::vector<int32_t> const* draftExtraTokensPerSlot = nullptr);
    //! Grow only the rows of @p decodeSequences (execution indices, each with a complete prompt) for a
    //! step in which the other sequences do not decode. Every other sequence must carry the hold advance
    //! in the matching completeDecodeStep. Non-speculative requests only.
    ContextCacheCoordinatorStatus prepareDecodeStep(
        RequestHandle& request, DecodingKvHeadroom const& headroom, std::vector<int32_t> const& decodeSequences);
    //! Apply the post-decode sequence advance after the decoder's existing synchronization. A sequence
    //! that did not decode this step -- outside the prepared decode subset, or still in admission --
    //! must carry the hold advance and cannot be published.
    //! For EAGLE, commonStateLengths excludes any unmaterialized accepted suffix and speculative lookahead.
    ContextCacheCoordinatorStatus completeDecodeStep(RequestHandle& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const& publishableCompletedSlots,
        std::vector<int32_t> const* commonStateLengths = nullptr, SwaWindowStateView const* swaWindowStates = nullptr);
    bool isExternallyOwnedSwaPage(RequestHandle const& request, int32_t slot, PageId page) const;
    PageId replaceSwaPage(RequestHandle& request, int32_t slot, PageId page);
    //! Validate and upload the one authoritative old-to-new mapping before any old-slot retirement work.
    ContextCacheCoordinatorStatus beginBatchCompaction(
        RequestHandle& request, std::vector<int32_t> const& oldToNew, int32_t newBatchSize, Tensor& deviceBatchMapping);
    //! Retire logical sequences while preserving survivor physical rows, then consume the eviction sync.
    ContextCacheCoordinatorStatus compactBatch(RequestHandle& request);
    //! Consume a normally completed request. This is idempotent for an already-empty handle.
    ContextCacheCoordinatorStatus finish(RequestHandle& request);
    //! Drain quarantined ownership before physical cache resources are destroyed.
    ContextCacheCoordinatorStatus shutdown() noexcept;

    ContextCacheMetrics metrics() const noexcept;
    ContextCacheManager const& manager() const noexcept;
    ContextCacheManager& manager() noexcept;

    //! Whether this deployment can lease and bind a new sequence into an executing request.
    bool supportsLiveSequenceAdmission() const noexcept;
    //! The longest uncached suffix an admitted sequence may leave to prefill: the engine's maximum input length.
    int32_t maxPrefillLength() const noexcept;
    //! Requests begun and not yet finished or abandoned, quarantined ones included.
    int32_t activeRequests() const noexcept;

private:
    enum class PublicationPoint : uint8_t
    {
        kPrefillEnd,
        kDecodeEnd,
    };

    struct AcquireSequenceResult;

    //! Per-request publication strategies (defined in the .cpp). The coordinator owns the shared lifecycle and
    //! delegates every flavor-specific endpoint/snapshot publication to the selected policy. Nested so they reach
    //! the coordinator's private state and primitives directly.
    class PublicationPolicy;
    class BaseEndpointPolicy;
    class HybridSnapshotPolicy;
    class HybridSpecPolicy;
    class EagleSpecPolicy;
    class SharedKvSpecPolicy;

    std::unique_ptr<PublicationPolicy> makePublicationPolicy(bool speculativeRequest);

    //! Shared tail of beginRequest and admitSequence: acquire one sequence's lease and file its
    //! SequenceState and metrics. On failure nothing is modified and @p refusal says why.
    bool acquireIntoRequest(RequestHandle::Impl& request, ContextCacheSequenceAdmission const& admission,
        DecodingKvHeadroom const& headroom, AdmitSequenceResult& refusal);

    AcquireSequenceResult acquireSequence(
        ContextCacheSequenceAdmission const& admission, bool speculativeRequest, DecodingKvHeadroom const& headroom);
    ContextCacheCoordinatorStatus applyAdvances(
        RequestHandle::Impl& request, std::vector<ContextCacheSequenceAdvance> const& advances);
    void validateSwaWindowStates(RequestHandle::Impl const& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, SwaWindowStateView const* swaWindowStates) const;
    //! How far committedStateLength advances per decode step, which is the only difference between the vanilla and
    //! MTP flavors of the committed-plus-lookahead check.
    enum class TokensPerDecodeStep : uint8_t
    {
        kExactlyOne,    //!< Vanilla / hybrid decode: the single sampled token.
        kAcceptedCount, //!< Speculative decode: everything the verification accepted this step.
    };
    //! Shared post-advance validation invoked by the policies, so the committed-plus-lookahead invariant math and
    //! the completed-slot dedup live in exactly one place regardless of publication flavor. Each policy calls the
    //! variant matching its decode flavor; PublicationPolicy::validateDecodeAdvances routes to the right one.
    void validateEagleDecodeAdvances(RequestHandle::Impl const& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const* commonStateLengths) const;
    void validateVanillaDecodeAdvances(RequestHandle::Impl const& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const* commonStateLengths) const;
    void validateFullyCommittedDecodeAdvances(RequestHandle::Impl const& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const* commonStateLengths) const;
    void validateMtpDecodeAdvances(RequestHandle::Impl const& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const* commonStateLengths) const;
    void validateCommittedLookaheadAdvances(RequestHandle::Impl const& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, TokensPerDecodeStep tokensPerStep) const;
    void assertUniqueCompletedSlots(
        RequestHandle::Impl const& request, std::vector<int32_t> const& publishableCompletedSlots) const;
    void publishReadyEndpoint(RequestHandle::Impl& request, int32_t slot, PublicationPoint point,
        SwaWindowState const* swaWindowState = nullptr);
    void reserveHybridCapture(RequestHandle::Impl& request, int32_t slot, int32_t exactLength, PublicationPoint point);
    void restoreHybridSequence(RequestHandle::Impl& request, int32_t sequenceIndex);
    void enqueueHybridCaptures(RequestHandle::Impl& request);
    void enqueueHybridCapture(RequestHandle::Impl& request, int32_t slot);
    void publishReadyHybridEndpoints(RequestHandle::Impl& request);
    void publishReadyHybridEndpoint(RequestHandle::Impl& request, int32_t slot);
    //! Shared non-paired decode growth; rows outside request.decodeParticipants keep their pages.
    ContextCacheCoordinatorStatus growDecodeWorkingSet(RequestHandle::Impl& request, DecodingKvHeadroom const& headroom,
        std::vector<int32_t> const* extraTokensPerSlot);
    void publishSpecEndpoint(
        RequestHandle::Impl& request, int32_t slot, int32_t commonStateLength, PublicationPoint point);
    void recordPublication(PublishStatus status) noexcept;
    //! Publish the frozen prefill endpoints, of every sequence or only those @p oldToNew retires.
    void publishFrozenSpecPrefill(RequestHandle::Impl& request, std::vector<int32_t> const* oldToNew = nullptr);
    RequestHandle::Impl& checkedImpl(RequestHandle& request) const;
    RequestHandle::Impl const& checkedImpl(RequestHandle const& request) const;
    bool isHybridDeployment() const noexcept;
    bool isPureRecurrentDeployment() const noexcept;
    bool usesCheckpointReuse() const noexcept;
    bool isSpecDeployment() const noexcept;
    bool ownsPagedSpecState() const noexcept;
    bool isSpecRequest(RequestHandle::Impl const& request) const noexcept;
    //! Request capability predicates. The context-cache subsystem is decoder-agnostic: the adapter collapses the
    //! decoder identity into the per-request speculativeRequest bit at admission. Lifecycle sites
    //! must not test that identity (== kEAGLE / == kMTP) directly -- each names the *capability* it depends on, so a
    //! future decoder that gains or loses a capability changes one predicate body, not a scavenger hunt across call
    //! sites. One capability owns every site that depends on it; splitting a capability across two predicates with the
    //! same body is how those sites drift apart.
    //!
    //! EAGLE and Hybrid+spec both run a paired base+draft cache working set: the lease owns a draft page path alongside
    //! the base one, so the draft engine's page table must carry that path (uploaded at prefill, grown at decode,
    //! compacted on eviction) and the draft cache must be reset at prefill. Leased draft pages are not reachable until
    //! the table names them -- it is identity-mapped otherwise, and restoring snapshot *contents* into a leased page
    //! does not publish the mapping that gets the engine there.
    bool runsPairedDraftWorkingSet(RequestHandle::Impl const& request) const noexcept;
    //! EAGLE's two-phase draft initialization publishes a frozen prefill endpoint after the first verification round
    //! terminalizes the ordered draft init. Hybrid+spec publishes via the hybrid snapshot endpoint path, no frozen
    //! phase.
    bool usesFrozenSpecPublication(RequestHandle::Impl const& request) const noexcept;
    bool deploymentHasAttention() const noexcept;
    //! Clear a resident slot's base page-table row, and its draft row for a paired speculative request.
    void clearSequenceRows(RequestHandle::Impl& request, int32_t residentSlot);
    ContextCacheCoordinatorStatus synchronizeRequest(RequestHandle& request);
    void abandon(std::unique_ptr<RequestHandle::Impl> request) noexcept;
    void quarantine(RequestHandle& request) noexcept;

    ContextCacheDeploymentProfile mProfile;
    ContextCacheManager mManager;
    std::unique_ptr<HybridSnapshotStorage> mHybridSnapshots;
    HybridCacheManager& mBaseCache;
    KVPageTable& mBasePageTable;
    HybridCacheManager* mDraftCache{};
    KVPageTable* mDraftPageTable{};
    cudaStream_t mStream{};
    StreamSynchronizer mSynchronizer;
    int32_t mMaxPrefillLength{};
    ContextCacheMetrics mMetrics;
    std::unique_ptr<Tensor> mHostReuseLengths;
    int32_t mActiveRequests{};
    bool mPoisoned{};
    //! Declared after mManager so quarantined leases are destroyed first during normal shutdown.
    std::vector<std::unique_ptr<RequestHandle::Impl>> mQuarantinedRequests;
};

} // namespace rt
} // namespace trt_edgellm
