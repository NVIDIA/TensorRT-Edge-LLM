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

#include "runtime/hybridCacheManager.h"
#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "kernels/speculative/batchEvictKernels.h"
#include <unordered_map>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace rt
{

HybridCacheManager::HybridCacheManager(Config const& config, cudaStream_t stream)
    : mConfig(config)
{
    int32_t const totalLayers = static_cast<int32_t>(mConfig.layerTypes.size());
    check::check(totalLayers > 0, "HybridCacheManager: layerTypes must not be empty.");
    check::check(mConfig.maxBatchSize > 0, "HybridCacheManager: maxBatchSize must be positive.");

    // Build routing tables: absolute layer index -> local sub-manager index.
    mAbsToKVIndex.resize(totalLayers, -1);
    mAbsToMambaIndex.resize(totalLayers, -1);

    int32_t kvLocalIdx = 0;
    int32_t mambaLocalIdx = 0;
    for (int32_t i = 0; i < totalLayers; ++i)
    {
        if (mConfig.layerTypes[i] == LayerType::kAttention)
        {
            mAbsToKVIndex[i] = kvLocalIdx++;
        }
        else
        {
            mAbsToMambaIndex[i] = mambaLocalIdx++;
        }
    }

    // Validate that local counts match sub-manager configs.
    check::check(kvLocalIdx == mConfig.kvConfig.numAttentionLayers,
        "HybridCacheManager: number of kAttention layers in layerTypes (" + std::to_string(kvLocalIdx)
            + ") must equal kvConfig.numAttentionLayers (" + std::to_string(mConfig.kvConfig.numAttentionLayers)
            + ").");
    check::check(mambaLocalIdx == mConfig.mambaConfig.numRecurrentLayers,
        "HybridCacheManager: number of kMamba layers in layerTypes (" + std::to_string(mambaLocalIdx)
            + ") must equal mambaConfig.numRecurrentLayers (" + std::to_string(mConfig.mambaConfig.numRecurrentLayers)
            + ").");

    // Construct sub-managers.
    mKVCache = KVCacheManager(mConfig.kvConfig, stream);
    mMambaCache = MambaCacheManager(mConfig.mambaConfig, stream);

    // Allocate shared device KV cache lengths tensor (zero-initialised).
    mDeviceKVCacheLengths = rt::Tensor(
        {mConfig.maxBatchSize}, DeviceType::kGPU, DataType::kINT32, "HybridCacheManager::mDeviceKVCacheLengths");
    CUDA_CHECK(
        cudaMemsetAsync(mDeviceKVCacheLengths.rawPointer(), 0, mDeviceKVCacheLengths.getMemoryCapacity(), stream));

    // Pre-build per-headDim groups for batched kernel launches.
    if (mKVCache.numLayers() > 0)
    {
        // Group KV layers by headDim, preserving insertion order via an auxiliary vector.
        std::unordered_map<int32_t, size_t> headDimToGroupIdx;
        for (int32_t i = 0; i < mKVCache.numLayers(); ++i)
        {
            auto const& lc = mKVCache.getLayerConfig(i);
            auto it = headDimToGroupIdx.find(lc.headDim);
            if (it == headDimToGroupIdx.end())
            {
                headDimToGroupIdx[lc.headDim] = mHeadDimGroups.size();
                // Positional init to satisfy -Werror=missing-field-initializers; order matches
                // the HeadDimGroup field declaration in hybridCacheManager.h.
                mHeadDimGroups.push_back(HeadDimGroup{lc.headDim, 0, 0, {}, {}, {}, {}});
            }
            size_t const gIdx = headDimToGroupIdx[lc.headDim];
            auto& group = mHeadDimGroups[gIdx];
            group.localKVIndices.push_back(i);
            group.maxKVHeads = std::max(group.maxKVHeads, lc.numKVHeads);
            group.numLayers++;

            // The active-slot K/V views use capPadded tokens per row.
            kernel::KVLayerInfo info{};
            info.data = mKVCache.kPoolPtr(i);
            info.numKVHeads = lc.numKVHeads;
            info.maxSeqLen = mKVCache.maxCapPadded();
            group.hostInfos.push_back(info);
        }

        // Upload each group's layer info array to the device and pre-allocate a scratch buffer
        // for save/restore operations (avoids per-call cudaMalloc/cudaFree synchronisation).
        for (auto& group : mHeadDimGroups)
        {
            size_t const infoBytes = group.hostInfos.size() * sizeof(kernel::KVLayerInfo);
            group.deviceLayerInfos = rt::Tensor({static_cast<int64_t>(infoBytes)}, DeviceType::kGPU, DataType::kINT8,
                "HybridCacheManager::headDimGroup_" + std::to_string(group.headDim));
            CUDA_CHECK(cudaMemcpyAsync(group.deviceLayerInfos.rawPointer(), group.hostInfos.data(), infoBytes,
                cudaMemcpyHostToDevice, stream));

            group.deviceScratchInfos = rt::Tensor({static_cast<int64_t>(infoBytes)}, DeviceType::kGPU, DataType::kINT8,
                "HybridCacheManager::scratchInfos_" + std::to_string(group.headDim));
        }
    }

    LOG_DEBUG("HybridCacheManager: totalLayers=%d, kvLayers=%d, mambaLayers=%d, maxBatchSize=%d, headDimGroups=%zu",
        totalLayers, kvLocalIdx, mambaLocalIdx, mConfig.maxBatchSize, mHeadDimGroups.size());
}

HybridCacheManager::~HybridCacheManager() noexcept {}

HybridCacheManager::HybridCacheManager(HybridCacheManager&& other) noexcept
{
    mConfig = std::move(other.mConfig);
    mKVCache = std::move(other.mKVCache);
    mMambaCache = std::move(other.mMambaCache);
    mAbsToKVIndex = std::move(other.mAbsToKVIndex);
    mAbsToMambaIndex = std::move(other.mAbsToMambaIndex);
    mDeviceKVCacheLengths = std::move(other.mDeviceKVCacheLengths);
    mActiveBatchSize = other.mActiveBatchSize;
    mKVCacheAllEmpty = other.mKVCacheAllEmpty;
    mHeadDimGroups = std::move(other.mHeadDimGroups);

    other.mConfig = Config{};
    other.mActiveBatchSize = 0;
    other.mKVCacheAllEmpty = true;
}

HybridCacheManager& HybridCacheManager::operator=(HybridCacheManager&& other) noexcept
{
    if (this != &other)
    {
        mConfig = std::move(other.mConfig);
        mKVCache = std::move(other.mKVCache);
        mMambaCache = std::move(other.mMambaCache);
        mAbsToKVIndex = std::move(other.mAbsToKVIndex);
        mAbsToMambaIndex = std::move(other.mAbsToMambaIndex);
        mDeviceKVCacheLengths = std::move(other.mDeviceKVCacheLengths);
        mActiveBatchSize = other.mActiveBatchSize;
        mKVCacheAllEmpty = other.mKVCacheAllEmpty;
        mHeadDimGroups = std::move(other.mHeadDimGroups);

        other.mConfig = Config{};
        other.mActiveBatchSize = 0;
        other.mKVCacheAllEmpty = true;
    }
    return *this;
}

// ------------------------------------------------------------------
// Routing by absolute layer index
// ------------------------------------------------------------------

HybridCacheManager::LayerType HybridCacheManager::getLayerType(int32_t absLayerIdx) const
{
    check::check(absLayerIdx >= 0 && absLayerIdx < static_cast<int32_t>(mConfig.layerTypes.size()),
        "getLayerType: absLayerIdx " + std::to_string(absLayerIdx) + " out of range.");
    return mConfig.layerTypes[absLayerIdx];
}

rt::Tensor& HybridCacheManager::getCombinedKVCache(int32_t absLayerIdx)
{
    check::check(absLayerIdx >= 0 && absLayerIdx < static_cast<int32_t>(mAbsToKVIndex.size()),
        "getCombinedKVCache: absLayerIdx " + std::to_string(absLayerIdx) + " out of range.");
    int32_t const localIdx = mAbsToKVIndex[absLayerIdx];
    check::check(
        localIdx >= 0, "getCombinedKVCache: layer " + std::to_string(absLayerIdx) + " is not an attention layer.");
    return mKVCache.getCombinedKVCache(localIdx);
}

rt::Tensor const& HybridCacheManager::getCombinedKVCache(int32_t absLayerIdx) const
{
    check::check(absLayerIdx >= 0 && absLayerIdx < static_cast<int32_t>(mAbsToKVIndex.size()),
        "getCombinedKVCache: absLayerIdx " + std::to_string(absLayerIdx) + " out of range.");
    int32_t const localIdx = mAbsToKVIndex[absLayerIdx];
    check::check(
        localIdx >= 0, "getCombinedKVCache: layer " + std::to_string(absLayerIdx) + " is not an attention layer.");
    return mKVCache.getCombinedKVCache(localIdx);
}

std::pair<rt::Tensor, rt::Tensor> HybridCacheManager::getSeparateKVCache(int32_t absLayerIdx)
{
    check::check(absLayerIdx >= 0 && absLayerIdx < static_cast<int32_t>(mAbsToKVIndex.size()),
        "getSeparateKVCache: absLayerIdx " + std::to_string(absLayerIdx) + " out of range.");
    int32_t const localIdx = mAbsToKVIndex[absLayerIdx];
    check::check(
        localIdx >= 0, "getSeparateKVCache: layer " + std::to_string(absLayerIdx) + " is not an attention layer.");
    return mKVCache.getSeparateKVCache(localIdx);
}

rt::Tensor& HybridCacheManager::getRecurrentState(int32_t absLayerIdx)
{
    check::check(absLayerIdx >= 0 && absLayerIdx < static_cast<int32_t>(mAbsToMambaIndex.size()),
        "getRecurrentState: absLayerIdx " + std::to_string(absLayerIdx) + " out of range.");
    int32_t const localIdx = mAbsToMambaIndex[absLayerIdx];
    check::check(localIdx >= 0, "getRecurrentState: layer " + std::to_string(absLayerIdx) + " is not a Mamba layer.");
    return mMambaCache.getRecurrentState(localIdx);
}

rt::Tensor& HybridCacheManager::getConvState(int32_t absLayerIdx)
{
    check::check(absLayerIdx >= 0 && absLayerIdx < static_cast<int32_t>(mAbsToMambaIndex.size()),
        "getConvState: absLayerIdx " + std::to_string(absLayerIdx) + " out of range.");
    int32_t const localIdx = mAbsToMambaIndex[absLayerIdx];
    check::check(localIdx >= 0, "getConvState: layer " + std::to_string(absLayerIdx) + " is not a Mamba layer.");
    return mMambaCache.getConvState(localIdx);
}

// ------------------------------------------------------------------
// Sub-manager direct access
// ------------------------------------------------------------------

KVCacheManager& HybridCacheManager::getKVCacheManager() noexcept
{
    return mKVCache;
}

KVCacheManager const& HybridCacheManager::getKVCacheManager() const noexcept
{
    return mKVCache;
}

MambaCacheManager& HybridCacheManager::getMambaCacheManager() noexcept
{
    return mMambaCache;
}

std::vector<HybridCacheManager::KVHeadDimGroupView> HybridCacheManager::getKVHeadDimGroups() const
{
    std::vector<KVHeadDimGroupView> views;
    views.reserve(mHeadDimGroups.size());
    for (auto const& group : mHeadDimGroups)
    {
        views.push_back(KVHeadDimGroupView{
            static_cast<kernel::KVLayerInfo const*>(group.deviceLayerInfos.rawPointer()),
            group.numLayers,
            group.headDim,
            group.maxKVHeads,
        });
    }
    return views;
}

// ------------------------------------------------------------------
// Shared state
// ------------------------------------------------------------------

rt::Tensor& HybridCacheManager::getKVCacheLengths() noexcept
{
    return mDeviceKVCacheLengths;
}

void HybridCacheManager::resetForNewSequences(rt::Tensor const& reuseKVCacheLengths, cudaStream_t stream)
{
    int32_t const batchSize = static_cast<int32_t>(reuseKVCacheLengths.getShape()[0]);
    check::check(
        batchSize <= mConfig.maxBatchSize, "Batch size of request shall not exceed the max supported batch size.");
    check::check(
        reuseKVCacheLengths.getDeviceType() == DeviceType::kCPU, "The reuseKVCacheLengths tensor shall reside on CPU.");
    check::check(reuseKVCacheLengths.getDataType() == mDeviceKVCacheLengths.getDataType(),
        "The data type of the reuseKVCacheLengths tensor shall match the data type of the Device KVCache Lengths.");

    mActiveBatchSize = batchSize;
    check::check(mDeviceKVCacheLengths.reshape({mActiveBatchSize}), "Tensor reshape failed");

    // If all reuse lengths are 0, set mKVCacheAllEmpty to true.
    int32_t const* reuseData = reuseKVCacheLengths.dataPointer<int32_t>();
    bool allEmpty{true};
    for (int32_t i = 0; i < batchSize; ++i)
    {
        if (reuseData[i] != 0)
        {
            allEmpty = false;
            break;
        }
    }
    mKVCacheAllEmpty = allEmpty;
    CUDA_CHECK(cudaMemcpyAsync(mDeviceKVCacheLengths.rawPointer(), reuseKVCacheLengths.rawPointer(),
        reuseKVCacheLengths.getMemoryCapacity(), cudaMemcpyHostToDevice, stream));
}

void HybridCacheManager::commitSequenceLength(rt::Tensor const& newContextLengths, cudaStream_t stream)
{
    check::check(newContextLengths.getDataType() == DataType::kINT32,
        "The newContextLengths tensor shall have data type of int32_t.");
    check::check(
        newContextLengths.getDeviceType() == DeviceType::kGPU, "The newContextLengths tensor shall reside on GPU.");
    check::check(newContextLengths.getShape()[0] == mActiveBatchSize,
        "The newContextLengths tensor shall have the same batch size as the active batch size.");

    kernel::incrementLengthTensor(mDeviceKVCacheLengths, newContextLengths, stream);

    // No longer empty after committing a sequence length.
    mKVCacheAllEmpty = false;
}

void HybridCacheManager::commitSequenceLength(int32_t increment, cudaStream_t stream)
{
    kernel::incrementLengthTensor(mDeviceKVCacheLengths, increment, stream);

    // No longer empty after committing a sequence length.
    mKVCacheAllEmpty = false;
}

// ------------------------------------------------------------------
// Batch management
// ------------------------------------------------------------------

int32_t HybridCacheManager::getActiveBatchSize() const noexcept
{
    return mActiveBatchSize;
}

void HybridCacheManager::setActiveBatchSize(int32_t newActiveBatchSize)
{
    check::check(newActiveBatchSize >= 0 && newActiveBatchSize <= mConfig.maxBatchSize,
        "Invalid active batch size: must be in range [0, maxBatchSize]");
    mActiveBatchSize = newActiveBatchSize;
    check::check(mDeviceKVCacheLengths.reshape({mActiveBatchSize}), "Tensor reshape failed");
}

bool HybridCacheManager::getKVCacheAllEmpty() const noexcept
{
    return mKVCacheAllEmpty;
}

// ------------------------------------------------------------------
// Compaction
// ------------------------------------------------------------------

void HybridCacheManager::compactBatch(
    rt::Tensor const& batchMapping, int32_t oldBatch, int32_t newBatch, cudaStream_t stream)
{
    // Active-slot K/V views have [maxBatch, capPadded, H, D] shape. Compaction moves only each
    // survivor's live prefix while the identity page table retains row == slot.
    for (auto const& group : mHeadDimGroups)
    {
        auto const* layerInfos = static_cast<kernel::KVLayerInfo const*>(group.deviceLayerInfos.rawPointer());
        kernel::compactKVCacheBatched(layerInfos, batchMapping, mDeviceKVCacheLengths, group.numLayers, group.headDim,
            mKVCache.numPages(), mConfig.kvConfig.kvCacheType, oldBatch, newBatch, stream);
    }

    compactBatchSlotState(batchMapping, oldBatch, newBatch, stream);
}

void HybridCacheManager::compactBatchSlotState(
    rt::Tensor const& batchMapping, int32_t oldBatch, int32_t newBatch, cudaStream_t stream)
{

    // Compact the shared KV cache lengths tensor separately after all layers are done.
    kernel::compactTensorBatch(mDeviceKVCacheLengths, batchMapping, mDeviceKVCacheLengths, oldBatch, newBatch, stream);

    // Compact Mamba recurrent and conv states.
    // compactTensorBatch asserts shape[0] == oldBatch, but Mamba tensors are allocated with
    // shape[0] == maxBatchSize. Reshape to [oldBatch, ...] before compaction and [newBatch, ...] after.
    auto const& mambaConfig = mMambaCache.getConfig();
    for (int32_t i = 0; i < mMambaCache.numLayers(); ++i)
    {
        rt::Tensor& recState = mMambaCache.getRecurrentState(i);
        check::check(recState.reshape({oldBatch, mambaConfig.recurrentStateNumHeads, mambaConfig.recurrentStateHeadDim,
                         mambaConfig.recurrentStateSize}),
            "Tensor reshape failed");
        kernel::compactTensorBatch(recState, batchMapping, recState, oldBatch, newBatch, stream);
        check::check(recState.reshape({mambaConfig.maxBatchSize, mambaConfig.recurrentStateNumHeads,
                         mambaConfig.recurrentStateHeadDim, mambaConfig.recurrentStateSize}),
            "Tensor reshape failed");

        rt::Tensor& convState = mMambaCache.getConvState(i);
        check::check(
            convState.reshape({oldBatch, mambaConfig.convDim, mambaConfig.convKernel}), "Tensor reshape failed");
        kernel::compactTensorBatch(convState, batchMapping, convState, oldBatch, newBatch, stream);
        check::check(convState.reshape({mambaConfig.maxBatchSize, mambaConfig.convDim, mambaConfig.convKernel}),
            "Tensor reshape failed");
    }
}

// ------------------------------------------------------------------
// System prompt cache
// ------------------------------------------------------------------

std::vector<rt::Tensor> HybridCacheManager::captureKVCache(
    int32_t batchIdx, int32_t sequenceLength, cudaStream_t stream)
{
    // The batched save/restore kernels only instantiate the `half` template today — match main's
    // contract of throwing loudly on unsupported dtypes instead of silently corrupting.
    check::check(mConfig.kvConfig.kvCacheType == nvinfer1::DataType::kHALF,
        "HybridCacheManager::captureKVCache currently only supports kHALF KV cache; "
        "FP8 system-prompt cache is not implemented.");

    // Allocate all per-layer destination tensors first.
    std::vector<rt::Tensor> result;
    result.reserve(mKVCache.numLayers());
    for (int32_t i = 0; i < mKVCache.numLayers(); ++i)
    {
        KVLayerConfig const& lc = mKVCache.getLayerConfig(i);
        // NHD saved layout: [2, sequenceLength, H, D] (K plane then V plane), each plane a
        // contiguous sequenceLength*H*D span — matches batchedKVCacheCopyKernel's NHD addressing.
        result.emplace_back(rt::Tensor({2, sequenceLength, lc.numKVHeads, lc.headDim}, DeviceType::kGPU,
            mConfig.kvConfig.kvCacheType, "HybridCacheManager::capturedKVCache_" + std::to_string(i)));
    }

    // Batch the copies per headDim group (non-const ref: scratch buffer is overwritten).
    for (auto& group : mHeadDimGroups)
    {
        // Build destination KVLayerInfo array on host, then upload to pre-allocated device scratch.
        // The vector is pageable; cudaMemcpyAsync stages it through the runtime's internal pinned
        // buffer (blocking briefly for the staging), which is what makes it safe to reuse the
        // vector immediately on return. For KB-sized uploads the staging cost is in the microseconds.
        std::vector<kernel::KVLayerInfo> dstInfos(group.numLayers);
        for (int32_t g = 0; g < group.numLayers; ++g)
        {
            int32_t const kvIdx = group.localKVIndices[g];
            KVLayerConfig const& lc = mKVCache.getLayerConfig(kvIdx);
            dstInfos[g].data = result[kvIdx].rawPointer();
            dstInfos[g].numKVHeads = lc.numKVHeads;
            dstInfos[g].maxSeqLen = sequenceLength; // destination tensor's "maxSeqLen" == sequenceLength
        }

        size_t const infoBytes = dstInfos.size() * sizeof(kernel::KVLayerInfo);
        CUDA_CHECK(cudaMemcpyAsync(
            group.deviceScratchInfos.rawPointer(), dstInfos.data(), infoBytes, cudaMemcpyHostToDevice, stream));

        auto const* srcLayerInfos = static_cast<kernel::KVLayerInfo const*>(group.deviceLayerInfos.rawPointer());
        auto const* dstLayerInfos = static_cast<kernel::KVLayerInfo const*>(group.deviceScratchInfos.rawPointer());

        kernel::saveKVCacheBatched(srcLayerInfos, dstLayerInfos, group.numLayers, group.headDim, mKVCache.numPages(),
            batchIdx, sequenceLength, stream);
    }

    return result;
}

void HybridCacheManager::restoreKVCache(std::vector<rt::Tensor> const& saved, int32_t batchIdx, cudaStream_t stream)
{
    // See captureKVCache for the dtype contract.
    check::check(mConfig.kvConfig.kvCacheType == nvinfer1::DataType::kHALF,
        "HybridCacheManager::restoreKVCache currently only supports kHALF KV cache; "
        "FP8 system-prompt cache is not implemented.");

    // Batch the copies per headDim group (non-const ref: scratch buffer is overwritten).
    for (auto& group : mHeadDimGroups)
    {
        // Pageable staging (see captureKVCache for rationale).
        std::vector<kernel::KVLayerInfo> srcInfos(group.numLayers);
        for (int32_t g = 0; g < group.numLayers; ++g)
        {
            int32_t const kvIdx = group.localKVIndices[g];
            KVLayerConfig const& lc = mKVCache.getLayerConfig(kvIdx);
            auto const& srcShape = saved[kvIdx].getShape();
            srcInfos[g].data = const_cast<void*>(saved[kvIdx].rawPointer());
            srcInfos[g].numKVHeads = lc.numKVHeads;
            srcInfos[g].maxSeqLen = srcShape[1]; // sequenceLength from the saved tensor [2, seqLen, H, D]
        }

        size_t const infoBytes = srcInfos.size() * sizeof(kernel::KVLayerInfo);
        CUDA_CHECK(cudaMemcpyAsync(
            group.deviceScratchInfos.rawPointer(), srcInfos.data(), infoBytes, cudaMemcpyHostToDevice, stream));

        auto const* dstLayerInfos = static_cast<kernel::KVLayerInfo const*>(group.deviceLayerInfos.rawPointer());
        auto const* srcLayerInfos = static_cast<kernel::KVLayerInfo const*>(group.deviceScratchInfos.rawPointer());

        int32_t const sequenceLength = saved[group.localKVIndices[0]].getShape()[1];

        kernel::instantiateKVCacheBatched(dstLayerInfos, srcLayerInfos, group.numLayers, group.headDim,
            mKVCache.numPages(), batchIdx, sequenceLength, stream);
    }
}

std::vector<rt::Tensor> HybridCacheManager::captureRecurrentStates(int32_t batchIdx, cudaStream_t stream)
{
    return mMambaCache.captureRecurrentStates(batchIdx, stream);
}

std::vector<rt::Tensor> HybridCacheManager::captureConvStates(int32_t batchIdx, cudaStream_t stream)
{
    return mMambaCache.captureConvStates(batchIdx, stream);
}

} // namespace rt
} // namespace trt_edgellm
