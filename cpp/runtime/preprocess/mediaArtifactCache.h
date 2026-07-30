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

#include "common/tensor.h"
#include "runtime/state/contextCache/blockHash.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

enum class MediaModality : uint8_t
{
    kVision = 1,
    kAudio = 2,
};

//! Identity of one position-independent encoder artifact in this loaded runtime.
struct MediaArtifactKey
{
    Hash128 contentDigest{};
    Hash128 isolationDigest{};
    MediaModality modality{MediaModality::kVision};
};

inline bool operator==(MediaArtifactKey const& lhs, MediaArtifactKey const& rhs) noexcept
{
    return lhs.contentDigest == rhs.contentDigest && lhs.isolationDigest == rhs.isolationDigest
        && lhs.modality == rhs.modality;
}

class MediaArtifactCache;

//! Move-only active pin on one immutable cached encoder artifact.
class MediaArtifactLease
{
public:
    MediaArtifactLease() noexcept = default;
    ~MediaArtifactLease() noexcept;
    MediaArtifactLease(MediaArtifactLease&& other) noexcept;
    MediaArtifactLease& operator=(MediaArtifactLease&& other) noexcept;
    MediaArtifactLease(MediaArtifactLease const&) = delete;
    MediaArtifactLease& operator=(MediaArtifactLease const&) = delete;

    Tensor const& embedding() const;
    std::vector<Tensor const*> deepstackFeatures() const;
    bool valid() const noexcept;
    void release() noexcept;

private:
    friend class MediaArtifactCache;
    struct Entry
    {
        struct ReadyEvent
        {
            ReadyEvent();
            ~ReadyEvent() noexcept;
            ReadyEvent(ReadyEvent const&) = delete;
            ReadyEvent& operator=(ReadyEvent const&) = delete;

            void record(cudaStream_t stream);
            void wait(cudaStream_t stream) const;

            cudaEvent_t event{};
        };

        MediaArtifactKey key;
        Tensor embedding;
        std::vector<Tensor> deepstackFeatures;
        size_t bytes{};
        int32_t activeRefs{};
        //! Recorded after the immutable device copies. Destroying it synchronizes before tensor storage is freed.
        std::unique_ptr<ReadyEvent> ready;
    };

    MediaArtifactCache* mCache{};
    Entry* mEntry{};
};

//! Byte- and entry-bounded LRU of immutable vision/audio encoder outputs.
//!
//! Entries live on the same device as their source tensors. acquire() and insert() return active leases, so
//! artifacts used by an in-flight request cannot be evicted by other insertions from that request. The cache and all
//! leases are confined to the runtime metadata thread; every lease must die before the cache.
class MediaArtifactCache
{
public:
    MediaArtifactCache(size_t capacityBytes, int32_t maxEntries);

    std::optional<MediaArtifactLease> acquire(MediaArtifactKey const& key, cudaStream_t stream = nullptr);
    //! Copy and pin a new artifact. Oversize entries or pinned-capacity pressure return no lease without mutation.
    std::optional<MediaArtifactLease> insert(MediaArtifactKey const& key, Tensor const& embedding,
        std::vector<Tensor const*> const& deepstackFeatures, cudaStream_t stream);

    size_t size() const noexcept;
    size_t residentBytes() const noexcept;
    bool contains(MediaArtifactKey const& key) const noexcept;

private:
    friend class MediaArtifactLease;
    using Entry = MediaArtifactLease::Entry;

    struct KeyHash
    {
        size_t operator()(MediaArtifactKey const& key) const noexcept;
    };

    void release(Entry& entry) noexcept;
    bool canFit(size_t incomingBytes) const;
    bool evictToFit(size_t incomingBytes);

    size_t mCapacityBytes{};
    int32_t mMaxEntries{};
    size_t mResidentBytes{};
    std::list<Entry> mEntries;
    std::unordered_map<MediaArtifactKey, std::list<Entry>::iterator, KeyHash> mIndex;
};

} // namespace rt
} // namespace trt_edgellm
