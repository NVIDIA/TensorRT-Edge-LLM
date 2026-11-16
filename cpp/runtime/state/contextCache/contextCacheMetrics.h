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

#include <cstdint>

namespace trt_edgellm
{
namespace rt
{

//! Current occupancy for one typed context-cache resource pool.
struct ContextCachePoolMetrics
{
    int32_t free{};
    int32_t capacity{};
};

//! Lightweight coordinator-local diagnostics. Counters are cumulative; occupancy fields are snapshots populated when
//! metrics() is called.
struct ContextCacheMetrics
{
    uint64_t admittedSequences{};
    uint64_t hitSequences{};
    //! Sequences whose block hash incorporated per-position media content hashes.
    uint64_t mediaAwareSequences{};
    //! Sequences whose final executable lookup policy was bypass, including forced-cold retries.
    uint64_t lookupBypassSequences{};
    uint64_t forcedColdSequences{};
    uint64_t standardPlans{};
    uint64_t noReusablePrefixPlans{};
    uint64_t fullInputRewindPlans{};
    uint64_t matchedTokens{};
    uint64_t reusedTokens{};
    uint64_t publicationAttempts{};
    //! PublishStatus::kPublished: a new record or an existing record's draft-state upgrade.
    uint64_t committedPublications{};
    uint64_t existingPublications{};
    //! Successful publication resolutions, including both new and already-existing endpoints.
    uint64_t publishedEndpoints{};
    uint64_t hybridRestores{};
    uint64_t hybridSnapshotPressureSkips{};
    uint64_t hybridCaptureSynchronizations{};
    uint64_t specFullPageReplays{};
    uint64_t specPairPublications{};
    uint64_t planningNanoseconds{};
    uint64_t currentRecords{};
    ContextCachePoolMetrics baseKvPages;
    ContextCachePoolMetrics draftKvPages;
    ContextCachePoolMetrics recurrentSnapshots;
    ContextCachePoolMetrics partialKvSnapshots;
    ContextCachePoolMetrics swaKvPages;
    uint64_t evictedRecords{};
    uint64_t reclaimedBaseKvPages{};
    uint64_t reclaimedDraftKvPages{};
    uint64_t reclaimedRecurrentSnapshots{};
    uint64_t reclaimedPartialKvSnapshots{};
    uint64_t reclaimedSwaKvPages{};
};

} // namespace rt
} // namespace trt_edgellm
