/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "runtime/kvCacheManager.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"

#include <limits>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace rt
{

namespace
{

size_t computeLayerBytes(int32_t numPages, KVLayerConfig const& layerConfig, size_t elemSize)
{
    size_t volume = 2;
    for (size_t const factor : {static_cast<size_t>(numPages), static_cast<size_t>(kTOKENS_PER_PAGE),
             static_cast<size_t>(layerConfig.numKVHeads), static_cast<size_t>(layerConfig.headDim), elemSize})
    {
        check::check(factor > 0 && volume <= std::numeric_limits<size_t>::max() / factor,
            "KVCacheManager: layer allocation size overflow.");
        volume *= factor;
    }
    return volume;
}

} // namespace

KVCacheManager::KVCacheManager(Config const& config, cudaStream_t stream)
    : mConfig(config)
{
    (void) stream;
    check::check(mConfig.kvCacheType == nvinfer1::DataType::kHALF || mConfig.kvCacheType == nvinfer1::DataType::kFP8,
        "Unsupported KV cache dtype.");
    check::check(mConfig.numAttentionLayers >= 0, "numAttentionLayers must be non-negative.");
    check::check(mConfig.maxBatchSize > 0, "maxBatchSize must be positive.");
    check::check(mConfig.maxSequenceLength > 0, "maxSequenceLength must be positive.");
    check::check(mConfig.maxSequenceLength <= kMAX_KV_CACHE_CAPACITY,
        "maxSequenceLength exceeds the largest value that remains int32 after page alignment.");
    check::check(static_cast<int32_t>(mConfig.layerConfigs.size()) == mConfig.numAttentionLayers,
        "layerConfigs size must equal numAttentionLayers.");

    // The no-argument APIs retain the full-engine geometry for compatibility. Active bounded layers
    // get separate physical geometry below and never masquerade as this full logical capacity.
    mCapPadded = computeMaxPagesPerSeq(mConfig.maxSequenceLength) * kTOKENS_PER_PAGE;

    // numPages defaults to the active-slot pages. A larger override retains pages across requests.
    int64_t const minimumActivePages = computeMinimumKvPoolPages(mConfig.maxBatchSize, mConfig.maxSequenceLength);
    check::check(minimumActivePages <= kMAX_KV_POOL_PAGES,
        "KVCacheManager: minimum active pages exceed the largest int32-addressable paged-KV pool.");
    check::check(mConfig.numPages == 0 || static_cast<int64_t>(mConfig.numPages) >= minimumActivePages,
        "KVCacheManager: Config::numPages (" + std::to_string(mConfig.numPages)
            + ") must be >= the minimum active pages (" + std::to_string(minimumActivePages) + ") when non-zero.");
    check::check(mConfig.numPages <= kMAX_KV_POOL_PAGES,
        "KVCacheManager: Config::numPages exceeds the largest supported paged-KV pool.");
    mNumPages = (mConfig.numPages == 0) ? static_cast<int32_t>(minimumActivePages) : mConfig.numPages;

    std::optional<int32_t> boundedCapability;
    std::optional<int32_t> boundedCapPadded;
    for (int32_t i = 0; i < mConfig.numAttentionLayers; ++i)
    {
        KVLayerConfig const& lc = mConfig.layerConfigs[i];
        check::check(lc.numKVHeads > 0, "numKVHeads must be positive for layer " + std::to_string(i) + ".");
        check::check(lc.headDim > 0, "headDim must be positive for layer " + std::to_string(i) + ".");
        check::check(lc.kvCacheCapacity >= 0 && lc.kvCacheCapacity <= mConfig.maxSequenceLength,
            "kvCacheCapacity must be in [0, maxSequenceLength] for layer " + std::to_string(i) + ".");

        int32_t const resolvedCapacity = resolveKvCacheCapacity(lc.kvCacheCapacity, mConfig.maxSequenceLength);
        if (resolvedCapacity < mConfig.maxSequenceLength)
        {
            check::check(!boundedCapability.has_value() || *boundedCapability == resolvedCapacity,
                "KVCacheManager: all SWA-capable layers must use one common kv_cache_capacity.");
            boundedCapability = resolvedCapacity;
        }
    }
    check::check(!boundedCapability.has_value() || mConfig.kvCacheType != nvinfer1::DataType::kFP8,
        "KVCacheManager: SWA-capable KV configurations do not support FP8 KV cache.");
    if (boundedCapability.has_value())
    {
        int64_t const minimumSwaPages = computeMinimumSwaPoolPages(mConfig.maxBatchSize, *boundedCapability);
        check::check(minimumSwaPages <= std::numeric_limits<int32_t>::max() && mConfig.numSwaPages >= minimumSwaPages,
            "KVCacheManager: numSwaPages must cover every active slot's private SWA pages.");
        int64_t const privatePages
            = computeSwaPrivatePagesPerSlot(*boundedCapability, kTOKENS_PER_PAGE, kSWA_REPLACEMENT_PAGES);
        check::check(privatePages <= std::numeric_limits<int32_t>::max() / kTOKENS_PER_PAGE,
            "KVCacheManager: per-slot SWA capacity exceeds int32 range.");
        boundedCapPadded = static_cast<int32_t>(privatePages * kTOKENS_PER_PAGE);
        if (mConfig.useBoundedSwaKVCache)
        {
            mReducedKVCacheCapacity = boundedCapability;
        }
    }

    // Pure-Mamba / pure-recurrent models legitimately have zero attention layers.
    // Leave mLayerCaches empty and skip uniformity detection.
    if (mConfig.numAttentionLayers == 0)
    {
        mIsUniform = true;
        return;
    }

    size_t const elemSize = rt::utils::getTypeSize(mConfig.kvCacheType);
    char const* kvCacheTypeStr = (mConfig.kvCacheType == nvinfer1::DataType::kHALF) ? "kHALF" : "kFP8";

    // Determine uniformity: check if all layers share the same numKVHeads and headDim.
    mIsUniform = true;
    for (int32_t i = 1; i < mConfig.numAttentionLayers; ++i)
    {
        if (mConfig.layerConfigs[i].numKVHeads != mConfig.layerConfigs[0].numKVHeads
            || mConfig.layerConfigs[i].headDim != mConfig.layerConfigs[0].headDim)
        {
            mIsUniform = false;
            break;
        }
    }

    // Every layer owns a pool-shaped buffer [2, numPages_i, kTOKENS_PER_PAGE, H, D].
    size_t totalBytes = 0;
    mLayerCaches.reserve(mConfig.numAttentionLayers);
    mLayerCapPadded.reserve(mConfig.numAttentionLayers);
    mLayerNumPages.reserve(mConfig.numAttentionLayers);
    for (int32_t i = 0; i < mConfig.numAttentionLayers; ++i)
    {
        KVLayerConfig const& lc = mConfig.layerConfigs[i];
        bool const isReduced
            = mConfig.useBoundedSwaKVCache && isReducedKvCacheCapacity(lc.kvCacheCapacity, mConfig.maxSequenceLength);
        int32_t const layerNumPages = isReduced ? mConfig.numSwaPages : mNumPages;
        int32_t const layerCapPadded = isReduced ? *boundedCapPadded : mCapPadded;
        mLayerNumPages.push_back(layerNumPages);
        mLayerCapPadded.push_back(layerCapPadded);

        size_t const layerBytes = computeLayerBytes(layerNumPages, lc, elemSize);
        check::check(totalBytes <= std::numeric_limits<size_t>::max() - layerBytes,
            "KVCacheManager: total allocation size overflow.");
        totalBytes += layerBytes;

        mLayerCaches.emplace_back(rt::Tensor({2, layerNumPages, kTOKENS_PER_PAGE, lc.numKVHeads, lc.headDim},
            DeviceType::kGPU, mConfig.kvCacheType, "KVCacheManager::layer_" + std::to_string(i)));
    }

    LOG_INFO("KVCacheManager(dtype=%s, layers=%d, uniform=%s) KV cache pool allocation: %zu bytes (%.2f MiB)",
        kvCacheTypeStr, mConfig.numAttentionLayers, mIsUniform ? "true" : "false", totalBytes,
        static_cast<float>(totalBytes) / (1024.0f * 1024.0f));
}

KVCacheManager::~KVCacheManager() noexcept {}

KVCacheManager::KVCacheManager(KVCacheManager&& other) noexcept
{
    mConfig = std::move(other.mConfig);
    mLayerCaches = std::move(other.mLayerCaches);
    mLayerCapPadded = std::move(other.mLayerCapPadded);
    mLayerNumPages = std::move(other.mLayerNumPages);
    mReducedKVCacheCapacity = other.mReducedKVCacheCapacity;
    mIsUniform = other.mIsUniform;
    mCapPadded = other.mCapPadded;
    mNumPages = other.mNumPages;

    other.mConfig = Config{};
    other.mIsUniform = true;
    other.mCapPadded = 0;
    other.mNumPages = 0;
    other.mReducedKVCacheCapacity.reset();
}

KVCacheManager& KVCacheManager::operator=(KVCacheManager&& other) noexcept
{
    if (this != &other)
    {
        mConfig = std::move(other.mConfig);
        mLayerCaches = std::move(other.mLayerCaches);
        mLayerCapPadded = std::move(other.mLayerCapPadded);
        mLayerNumPages = std::move(other.mLayerNumPages);
        mReducedKVCacheCapacity = other.mReducedKVCacheCapacity;
        mIsUniform = other.mIsUniform;
        mCapPadded = other.mCapPadded;
        mNumPages = other.mNumPages;

        other.mConfig = Config{};
        other.mIsUniform = true;
        other.mCapPadded = 0;
        other.mNumPages = 0;
        other.mReducedKVCacheCapacity.reset();
    }
    return *this;
}

rt::Tensor& KVCacheManager::getCombinedKVCache(int32_t attnLayerIdx) noexcept
{
    return mLayerCaches[attnLayerIdx];
}

rt::Tensor const& KVCacheManager::getCombinedKVCache(int32_t attnLayerIdx) const noexcept
{
    return mLayerCaches[attnLayerIdx];
}

std::pair<rt::Tensor, rt::Tensor> KVCacheManager::getSeparateKVCache(int32_t attnLayerIdx) const noexcept
{
    KVLayerConfig const& lc = mConfig.layerConfigs[attnLayerIdx];
    if (mConfig.useBoundedSwaKVCache && isReducedKvCacheCapacity(lc.kvCacheCapacity, mConfig.maxSequenceLength))
    {
        rt::Coords const poolHalfShape{numPages(attnLayerIdx), kTOKENS_PER_PAGE, lc.numKVHeads, lc.headDim};
        rt::Tensor kView(kPoolPtr(attnLayerIdx), poolHalfShape, DeviceType::kGPU, mConfig.kvCacheType);
        rt::Tensor vView(vPoolPtr(attnLayerIdx), poolHalfShape, DeviceType::kGPU, mConfig.kvCacheType);
        return {std::move(kView), std::move(vView)};
    }
    // K-half / V-half of the NHD pool [2, maxBatch, capPadded, H, D] are each a contiguous
    // [maxBatch, capPadded, H, D] view; kPoolPtr()/vPoolPtr() already carry the V-half offset.
    rt::Tensor kView(kPoolPtr(attnLayerIdx),
        {mConfig.maxBatchSize, mLayerCapPadded[attnLayerIdx], lc.numKVHeads, lc.headDim}, DeviceType::kGPU,
        mConfig.kvCacheType);
    rt::Tensor vView(vPoolPtr(attnLayerIdx),
        {mConfig.maxBatchSize, mLayerCapPadded[attnLayerIdx], lc.numKVHeads, lc.headDim}, DeviceType::kGPU,
        mConfig.kvCacheType);
    return {std::move(kView), std::move(vView)};
}

int32_t KVCacheManager::maxCapPadded() const noexcept
{
    return mCapPadded;
}

int32_t KVCacheManager::maxCapPadded(int32_t attnLayerIdx) const noexcept
{
    return mLayerCapPadded[attnLayerIdx];
}

int32_t KVCacheManager::numPages() const noexcept
{
    return mNumPages;
}

int32_t KVCacheManager::numPages(int32_t attnLayerIdx) const noexcept
{
    return mLayerNumPages[attnLayerIdx];
}

bool KVCacheManager::hasReducedKVCache() const noexcept
{
    return mReducedKVCacheCapacity.has_value();
}

std::optional<int32_t> KVCacheManager::reducedKVCacheCapacity() const noexcept
{
    return mReducedKVCacheCapacity;
}

void* KVCacheManager::kPoolPtr(int32_t attnLayerIdx) const noexcept
{
    return const_cast<void*>(mLayerCaches[attnLayerIdx].rawPointer());
}

void* KVCacheManager::vPoolPtr(int32_t attnLayerIdx) const noexcept
{
    KVLayerConfig const& lc = mConfig.layerConfigs[attnLayerIdx];
    size_t const elemSize = rt::utils::getTypeSize(mConfig.kvCacheType);
    int64_t const kCacheElems
        = static_cast<int64_t>(numPages(attnLayerIdx)) * kTOKENS_PER_PAGE * lc.numKVHeads * lc.headDim;
    return static_cast<char*>(kPoolPtr(attnLayerIdx)) + kCacheElems * static_cast<int64_t>(elemSize);
}

KVLayerConfig const& KVCacheManager::getLayerConfig(int32_t attnLayerIdx) const noexcept
{
    return mConfig.layerConfigs[attnLayerIdx];
}

KVLayerStorageMetadata KVCacheManager::getLayerStorageMetadata(int32_t attnLayerIdx) const noexcept
{
    KVLayerConfig const& layer = mConfig.layerConfigs[attnLayerIdx];
    bool const reduced
        = mConfig.useBoundedSwaKVCache && isReducedKvCacheCapacity(layer.kvCacheCapacity, mConfig.maxSequenceLength);
    return {reduced ? KVCacheStorageKind::kReducedSwa : KVCacheStorageKind::kFull, numPages(attnLayerIdx),
        computeMaxPagesPerSeq(mConfig.maxSequenceLength), layer.numKVHeads, layer.headDim};
}

int32_t KVCacheManager::numLayers() const noexcept
{
    return mConfig.numAttentionLayers;
}

bool KVCacheManager::isUniform() const noexcept
{
    return mIsUniform;
}

KVCacheManager::Config const& KVCacheManager::getConfig() const noexcept
{
    return mConfig;
}

} // namespace rt
} // namespace trt_edgellm
