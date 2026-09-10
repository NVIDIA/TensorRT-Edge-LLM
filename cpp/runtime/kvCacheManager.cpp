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

using namespace nvinfer1;

namespace trt_edgellm
{
namespace rt
{

KVCacheManager::KVCacheManager(Config const& config, cudaStream_t stream)
    : mConfig(config)
{
    check::check(mConfig.kvCacheType == nvinfer1::DataType::kHALF || mConfig.kvCacheType == nvinfer1::DataType::kFP8,
        "Unsupported KV cache dtype.");
    check::check(mConfig.numAttentionLayers >= 0, "numAttentionLayers must be non-negative.");
    check::check(mConfig.maxBatchSize > 0, "maxBatchSize must be positive.");
    check::check(mConfig.maxSequenceLength > 0, "maxSequenceLength must be positive.");
    check::check(mConfig.maxSequenceLength <= kMAX_KV_CACHE_CAPACITY,
        "maxSequenceLength exceeds the largest value that remains int32 after page alignment.");
    check::check(static_cast<int32_t>(mConfig.layerConfigs.size()) == mConfig.numAttentionLayers,
        "layerConfigs size must equal numAttentionLayers.");

    // Token capacity of each active-slot K/V view, padded to a whole number of pages.
    mCapPadded = static_cast<int32_t>(
        ((static_cast<int64_t>(mConfig.maxSequenceLength) + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE)
        * kTOKENS_PER_PAGE);

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

    size_t totalBytes = 0;
    mLayerCaches.reserve(mConfig.numAttentionLayers);
    for (int32_t i = 0; i < mConfig.numAttentionLayers; ++i)
    {
        KVLayerConfig const& lc = mConfig.layerConfigs[i];
        check::check(lc.numKVHeads > 0, "numKVHeads must be positive for layer " + std::to_string(i) + ".");
        check::check(lc.headDim > 0, "headDim must be positive for layer " + std::to_string(i) + ".");

        int64_t const layerVolume = 2 * static_cast<int64_t>(mNumPages) * kTOKENS_PER_PAGE * lc.numKVHeads * lc.headDim;
        size_t const layerBytes = static_cast<size_t>(layerVolume) * elemSize;
        totalBytes += layerBytes;

        mLayerCaches.emplace_back(rt::Tensor({2, mNumPages, kTOKENS_PER_PAGE, lc.numKVHeads, lc.headDim},
            DeviceType::kGPU, mConfig.kvCacheType, "KVCacheManager::layer_" + std::to_string(i)));
    }

    LOG_DEBUG("KVCacheManager(dtype=%s, layers=%d, uniform=%s) allocated %.2f MB total GPU memory", kvCacheTypeStr,
        mConfig.numAttentionLayers, mIsUniform ? "true" : "false",
        static_cast<float>(totalBytes) / (1024.0f * 1024.0f));
}

KVCacheManager::~KVCacheManager() noexcept {}

KVCacheManager::KVCacheManager(KVCacheManager&& other) noexcept
{
    mConfig = std::move(other.mConfig);
    mLayerCaches = std::move(other.mLayerCaches);
    mIsUniform = other.mIsUniform;
    mCapPadded = other.mCapPadded;
    mNumPages = other.mNumPages;

    other.mConfig = Config{};
    other.mIsUniform = true;
    other.mCapPadded = 0;
    other.mNumPages = 0;
}

KVCacheManager& KVCacheManager::operator=(KVCacheManager&& other) noexcept
{
    if (this != &other)
    {
        mConfig = std::move(other.mConfig);
        mLayerCaches = std::move(other.mLayerCaches);
        mIsUniform = other.mIsUniform;
        mCapPadded = other.mCapPadded;
        mNumPages = other.mNumPages;

        other.mConfig = Config{};
        other.mIsUniform = true;
        other.mCapPadded = 0;
        other.mNumPages = 0;
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
    rt::Tensor kView(kPoolPtr(attnLayerIdx), {mConfig.maxBatchSize, mCapPadded, lc.numKVHeads, lc.headDim},
        DeviceType::kGPU, mConfig.kvCacheType);
    rt::Tensor vView(vPoolPtr(attnLayerIdx), {mConfig.maxBatchSize, mCapPadded, lc.numKVHeads, lc.headDim},
        DeviceType::kGPU, mConfig.kvCacheType);
    return {std::move(kView), std::move(vView)};
}

int32_t KVCacheManager::maxCapPadded() const noexcept
{
    return mCapPadded;
}

int32_t KVCacheManager::numPages() const noexcept
{
    return mNumPages;
}

void* KVCacheManager::kPoolPtr(int32_t attnLayerIdx) const noexcept
{
    return const_cast<void*>(mLayerCaches[attnLayerIdx].rawPointer());
}

void* KVCacheManager::vPoolPtr(int32_t attnLayerIdx) const noexcept
{
    KVLayerConfig const& lc = mConfig.layerConfigs[attnLayerIdx];
    size_t const elemSize = rt::utils::getTypeSize(mConfig.kvCacheType);
    int64_t const kCacheElems = static_cast<int64_t>(numPages()) * kTOKENS_PER_PAGE * lc.numKVHeads * lc.headDim;
    return static_cast<char*>(kPoolPtr(attnLayerIdx)) + kCacheElems * static_cast<int64_t>(elemSize);
}

KVLayerConfig const& KVCacheManager::getLayerConfig(int32_t attnLayerIdx) const noexcept
{
    return mConfig.layerConfigs[attnLayerIdx];
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
