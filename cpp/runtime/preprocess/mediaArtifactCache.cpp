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

#include "runtime/preprocess/mediaArtifactCache.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/hashUtils.h"

#include <algorithm>
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

size_t tensorBytes(Tensor const& tensor)
{
    ELLM_CHECK(tensor.getShape().volume() >= 0, "Media artifact tensor has a negative volume");
    size_t const elements = static_cast<size_t>(tensor.getShape().volume());
    size_t const elementBytes = utils::getTypeSize(tensor.getDataType());
    ELLM_CHECK(elementBytes == 0 || elements <= std::numeric_limits<size_t>::max() / elementBytes,
        "Media artifact tensor byte size overflows size_t");
    return elements * elementBytes;
}

Tensor copyTensor(Tensor const& source, std::string const& name, cudaStream_t stream)
{
    Tensor copy(source.getShape(), source.getDeviceType(), source.getDataType(), name);
    size_t const bytes = tensorBytes(source);
    if (bytes == 0)
    {
        return copy;
    }
    if (source.getDeviceType() == DeviceType::kCPU)
    {
        std::memcpy(copy.rawPointer(), source.rawPointer(), bytes);
    }
    else
    {
        CUDA_CHECK(cudaMemcpyAsync(copy.rawPointer(), source.rawPointer(), bytes, cudaMemcpyDeviceToDevice, stream));
    }
    return copy;
}

} // namespace

MediaArtifactLease::Entry::ReadyEvent::ReadyEvent()
{
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
}

MediaArtifactLease::Entry::ReadyEvent::~ReadyEvent() noexcept
{
    if (event != nullptr)
    {
        (void) cudaEventSynchronize(event);
        (void) cudaEventDestroy(event);
    }
}

void MediaArtifactLease::Entry::ReadyEvent::record(cudaStream_t stream)
{
    CUDA_CHECK(cudaEventRecord(event, stream));
}

void MediaArtifactLease::Entry::ReadyEvent::wait(cudaStream_t stream) const
{
    CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
}

MediaArtifactLease::~MediaArtifactLease() noexcept
{
    release();
}

MediaArtifactLease::MediaArtifactLease(MediaArtifactLease&& other) noexcept
    : mCache(std::exchange(other.mCache, nullptr))
    , mEntry(std::exchange(other.mEntry, nullptr))
{
}

MediaArtifactLease& MediaArtifactLease::operator=(MediaArtifactLease&& other) noexcept
{
    if (this != &other)
    {
        release();
        mCache = std::exchange(other.mCache, nullptr);
        mEntry = std::exchange(other.mEntry, nullptr);
    }
    return *this;
}

Tensor const& MediaArtifactLease::embedding() const
{
    ELLM_CHECK(valid(), "Cannot access an invalid media artifact lease");
    return mEntry->embedding;
}

std::vector<Tensor const*> MediaArtifactLease::deepstackFeatures() const
{
    ELLM_CHECK(valid(), "Cannot access an invalid media artifact lease");
    std::vector<Tensor const*> features;
    features.reserve(mEntry->deepstackFeatures.size());
    for (Tensor const& feature : mEntry->deepstackFeatures)
    {
        features.push_back(&feature);
    }
    return features;
}

bool MediaArtifactLease::valid() const noexcept
{
    return mCache != nullptr && mEntry != nullptr;
}

void MediaArtifactLease::release() noexcept
{
    if (valid())
    {
        mCache->release(*mEntry);
    }
    mCache = nullptr;
    mEntry = nullptr;
}

MediaArtifactCache::MediaArtifactCache(size_t capacityBytes, int32_t maxEntries)
    : mCapacityBytes(capacityBytes)
    , mMaxEntries(maxEntries)
{
    ELLM_CHECK(maxEntries >= 0, "Media artifact cache entry limit must be non-negative");
}

std::optional<MediaArtifactLease> MediaArtifactCache::acquire(MediaArtifactKey const& key, cudaStream_t stream)
{
    auto const found = mIndex.find(key);
    if (found == mIndex.end())
    {
        return std::nullopt;
    }
    Entry& entry = *found->second;
    if (entry.ready)
    {
        entry.ready->wait(stream);
    }
    mEntries.splice(mEntries.begin(), mEntries, found->second);
    ++entry.activeRefs;
    MediaArtifactLease lease;
    lease.mCache = this;
    lease.mEntry = &entry;
    return lease;
}

std::optional<MediaArtifactLease> MediaArtifactCache::insert(MediaArtifactKey const& key, Tensor const& embedding,
    std::vector<Tensor const*> const& deepstackFeatures, cudaStream_t stream)
{
    if (mIndex.find(key) != mIndex.end())
    {
        return acquire(key, stream);
    }

    size_t bytes = tensorBytes(embedding);
    for (Tensor const* feature : deepstackFeatures)
    {
        ELLM_CHECK(feature != nullptr, "Media artifact deepstack feature cannot be null");
        size_t const featureBytes = tensorBytes(*feature);
        ELLM_CHECK(bytes <= std::numeric_limits<size_t>::max() - featureBytes,
            "Media artifact aggregate byte size overflows size_t");
        bytes += featureBytes;
    }
    if (bytes > mCapacityBytes || mMaxEntries == 0 || !canFit(bytes))
    {
        return std::nullopt;
    }
    ELLM_CHECK(evictToFit(bytes), "Media artifact eviction plan changed during insertion");

    Entry entry;
    entry.key = key;
    entry.embedding = copyTensor(embedding, "MediaArtifactCache::embedding", stream);
    entry.deepstackFeatures.reserve(deepstackFeatures.size());
    for (Tensor const* feature : deepstackFeatures)
    {
        entry.deepstackFeatures.push_back(copyTensor(*feature, "MediaArtifactCache::deepstack", stream));
    }
    entry.bytes = bytes;
    entry.activeRefs = 1;
    bool const deviceCopy = embedding.getDeviceType() == DeviceType::kGPU
        || std::any_of(deepstackFeatures.begin(), deepstackFeatures.end(),
            [](Tensor const* feature) { return feature->getDeviceType() == DeviceType::kGPU; });
    if (deviceCopy)
    {
        entry.ready = std::make_unique<Entry::ReadyEvent>();
        entry.ready->record(stream);
    }
    mEntries.push_front(std::move(entry));
    mIndex.emplace(key, mEntries.begin());
    mResidentBytes += bytes;

    MediaArtifactLease lease;
    lease.mCache = this;
    lease.mEntry = &mEntries.front();
    return lease;
}

size_t MediaArtifactCache::size() const noexcept
{
    return mEntries.size();
}

size_t MediaArtifactCache::residentBytes() const noexcept
{
    return mResidentBytes;
}

bool MediaArtifactCache::contains(MediaArtifactKey const& key) const noexcept
{
    return mIndex.find(key) != mIndex.end();
}

size_t MediaArtifactCache::KeyHash::operator()(MediaArtifactKey const& key) const noexcept
{
    size_t seed{};
    hash_utils::hashCombine(seed, key.contentDigest);
    hash_utils::hashCombine(seed, key.isolationDigest);
    hash_utils::hashCombine(seed, key.modality);
    return seed;
}

void MediaArtifactCache::release(Entry& entry) noexcept
{
    if (entry.activeRefs > 0)
    {
        --entry.activeRefs;
    }
}

bool MediaArtifactCache::canFit(size_t incomingBytes) const
{
    size_t residentBytes = mResidentBytes;
    size_t residentEntries = mEntries.size();
    auto overLimit = [&]() {
        return residentBytes > mCapacityBytes - incomingBytes || residentEntries >= static_cast<size_t>(mMaxEntries);
    };
    for (auto victim = mEntries.rbegin(); overLimit() && victim != mEntries.rend(); ++victim)
    {
        if (victim->activeRefs == 0)
        {
            residentBytes -= victim->bytes;
            --residentEntries;
        }
    }
    return !overLimit();
}

bool MediaArtifactCache::evictToFit(size_t incomingBytes)
{
    auto overLimit = [&]() {
        return mResidentBytes > mCapacityBytes - incomingBytes || mEntries.size() >= static_cast<size_t>(mMaxEntries);
    };
    while (overLimit())
    {
        auto victim = std::find_if(
            mEntries.rbegin(), mEntries.rend(), [](Entry const& entry) { return entry.activeRefs == 0; });
        if (victim == mEntries.rend())
        {
            return false;
        }
        auto const eraseIt = std::prev(victim.base());
        mResidentBytes -= eraseIt->bytes;
        mIndex.erase(eraseIt->key);
        mEntries.erase(eraseIt);
    }
    return true;
}

} // namespace rt
} // namespace trt_edgellm
