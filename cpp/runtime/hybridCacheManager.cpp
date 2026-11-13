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
#include "kernels/speculative/mtpStateScatterKernels.h"

#include <exception>
#include <unordered_map>

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

    // Allocate any auxiliary conv states (zero-initialised), one per configured entry.
    mAuxConvStates.reserve(mConfig.auxConvStateConfigs.size());
    for (size_t i = 0; i < mConfig.auxConvStateConfigs.size(); ++i)
    {
        auto const& stateConfig = mConfig.auxConvStateConfigs[i];
        check::check(stateConfig.absLayerIdx >= 0 && stateConfig.absLayerIdx < totalLayers,
            "HybridCacheManager: auxiliary conv state names layer " + std::to_string(stateConfig.absLayerIdx)
                + ", which is outside a " + std::to_string(totalLayers) + "-layer stack.");
        check::check(stateConfig.stateIdx > 0,
            "HybridCacheManager: auxiliary conv state index must be above 0, which "
            "belongs to the layer's own mixer.");
        check::check(stateConfig.convDim > 0 && stateConfig.convWidth > 0,
            "HybridCacheManager: auxiliary conv state must have positive dimensions.");
        mAuxConvStates.emplace_back(Coords{mConfig.maxBatchSize, stateConfig.convDim, stateConfig.convWidth},
            DeviceType::kGPU, stateConfig.dataType, "HybridCacheManager::mAuxConvStates" + std::to_string(i));
        CUDA_CHECK(
            cudaMemsetAsync(mAuxConvStates.back().rawPointer(), 0, mAuxConvStates.back().getMemoryCapacity(), stream));
    }

    // One snapshot per verified token, committed after acceptance like the GDN conv state; the two
    // share the switch that turns them on.
    // TODO: the registry declares these bindings from the spec mode alone, so a GDN-free n-gram base
    // would allocate nothing here and fail at the first verify; move both under one commit policy.
    int32_t const intermediateSeqLen = mConfig.mambaConfig.maxIntermediateSeqLen;
    if (!mAuxConvStates.empty() && intermediateSeqLen > 0)
    {
        std::vector<kernel::MtpLayerInfo> auxInfos(mAuxConvStates.size());
        for (size_t i = 0; i < mConfig.auxConvStateConfigs.size(); ++i)
        {
            auto const& stateConfig = mConfig.auxConvStateConfigs[i];
            check::check(stateConfig.dataType == DataType::kHALF,
                "HybridCacheManager: the auxiliary conv state scatter supports FP16 states only.");
            check::check(stateConfig.convDim == mConfig.auxConvStateConfigs[0].convDim
                    && stateConfig.convWidth == mConfig.auxConvStateConfigs[0].convWidth,
                "HybridCacheManager: auxiliary conv states must share one geometry to be scattered together.");
            mIntermediateAuxConvStates.emplace_back(
                Coords{mConfig.maxBatchSize, intermediateSeqLen, stateConfig.convDim, stateConfig.convWidth},
                DeviceType::kGPU, stateConfig.dataType,
                "HybridCacheManager::mIntermediateAuxConvStates" + std::to_string(i));
            CUDA_CHECK(cudaMemsetAsync(mIntermediateAuxConvStates.back().rawPointer(), 0,
                mIntermediateAuxConvStates.back().getMemoryCapacity(), stream));
            auxInfos[i]
                = {nullptr, nullptr, mAuxConvStates[i].rawPointer(), mIntermediateAuxConvStates.back().rawPointer()};
        }
        size_t const infoBytes = auxInfos.size() * sizeof(kernel::MtpLayerInfo);
        mDeviceAuxMtpLayerInfos = rt::Tensor({static_cast<int64_t>(infoBytes)}, DeviceType::kGPU, DataType::kINT8,
            "HybridCacheManager::auxMtpLayerInfos");
        CUDA_CHECK(cudaMemcpyAsync(
            mDeviceAuxMtpLayerInfos.rawPointer(), auxInfos.data(), infoBytes, cudaMemcpyHostToDevice, stream));
        // Sync before auxInfos goes out of scope.
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Pre-build per-headDim groups for batched kernel launches.
    if (mKVCache.numLayers() > 0)
    {
        // Group KV layers by headDim, preserving insertion order via an auxiliary vector.
        std::unordered_map<int32_t, size_t> headDimToGroupIdx;
        for (int32_t i = 0; i < mKVCache.numLayers(); ++i)
        {
            auto const& lc = mKVCache.getLayerConfig(i);
            if (mKVCache.hasReducedKVCache()
                && isReducedKvCacheCapacity(lc.kvCacheCapacity, mConfig.kvConfig.maxSequenceLength))
            {
                continue;
            }
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

            // Dense batched-copy operations only include full-capacity layers.
            kernel::KVLayerInfo info{};
            info.data = mKVCache.kPoolPtr(i);
            info.numKVHeads = lc.numKVHeads;
            info.maxSeqLen = mKVCache.maxCapPadded(i);
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
    mAuxConvStates = std::move(other.mAuxConvStates);
    mIntermediateAuxConvStates = std::move(other.mIntermediateAuxConvStates);
    mDeviceAuxMtpLayerInfos = std::move(other.mDeviceAuxMtpLayerInfos);

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
        mAuxConvStates = std::move(other.mAuxConvStates);
        mIntermediateAuxConvStates = std::move(other.mIntermediateAuxConvStates);
        mDeviceAuxMtpLayerInfos = std::move(other.mDeviceAuxMtpLayerInfos);

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

KVLayerStorageMetadata HybridCacheManager::getKVLayerStorageMetadata(int32_t absLayerIdx) const
{
    check::check(absLayerIdx >= 0 && absLayerIdx < static_cast<int32_t>(mAbsToKVIndex.size()),
        "getKVLayerStorageMetadata: absolute layer index is out of range.");
    int32_t const localIdx = mAbsToKVIndex[absLayerIdx];
    check::check(localIdx >= 0, "getKVLayerStorageMetadata: layer is not an attention layer.");
    return mKVCache.getLayerStorageMetadata(localIdx);
}

rt::Tensor& HybridCacheManager::getRecurrentState(int32_t absLayerIdx)
{
    check::check(absLayerIdx >= 0 && absLayerIdx < static_cast<int32_t>(mAbsToMambaIndex.size()),
        "getRecurrentState: absLayerIdx " + std::to_string(absLayerIdx) + " out of range.");
    int32_t const localIdx = mAbsToMambaIndex[absLayerIdx];
    check::check(localIdx >= 0, "getRecurrentState: layer " + std::to_string(absLayerIdx) + " is not a Mamba layer.");
    return mMambaCache.getRecurrentState(localIdx);
}

HybridCacheManager::Config const& HybridCacheManager::getConfig() const noexcept
{
    return mConfig;
}

int32_t HybridCacheManager::numAuxConvStates() const noexcept
{
    return static_cast<int32_t>(mAuxConvStates.size());
}

HybridCacheManager::AuxConvStateConfig const& HybridCacheManager::auxConvStateConfig(int32_t auxIdx) const
{
    check::check(auxIdx >= 0 && auxIdx < numAuxConvStates(),
        "auxConvStateConfig: index " + std::to_string(auxIdx) + " out of range.");
    return mConfig.auxConvStateConfigs[static_cast<size_t>(auxIdx)];
}

rt::Tensor& HybridCacheManager::getAuxConvState(int32_t auxIdx)
{
    check::check(auxIdx >= 0 && auxIdx < numAuxConvStates(),
        "getAuxConvState: index " + std::to_string(auxIdx) + " out of range.");
    return mAuxConvStates[static_cast<size_t>(auxIdx)];
}

bool HybridCacheManager::hasIntermediateAuxConvStates() const noexcept
{
    return !mIntermediateAuxConvStates.empty();
}

rt::Tensor& HybridCacheManager::getIntermediateAuxConvState(int32_t auxIdx)
{
    check::check(auxIdx >= 0 && auxIdx < static_cast<int32_t>(mIntermediateAuxConvStates.size()),
        "getIntermediateAuxConvState: index " + std::to_string(auxIdx) + " out of range.");
    return mIntermediateAuxConvStates[static_cast<size_t>(auxIdx)];
}

void HybridCacheManager::reshapeIntermediateAuxConvStates(int32_t activeBatchSize, int32_t seqLen)
{
    for (size_t i = 0; i < mIntermediateAuxConvStates.size(); ++i)
    {
        auto const& stateConfig = mConfig.auxConvStateConfigs[i];
        check::check(mIntermediateAuxConvStates[i].reshape(
                         {activeBatchSize, seqLen, stateConfig.convDim, stateConfig.convWidth}),
            "Intermediate auxiliary conv state reshape failed");
    }
}

void HybridCacheManager::scatterAcceptedAuxConvStates(rt::Tensor const& acceptLengths, rt::Tensor const& stateIndices,
    cudaStream_t stream, rt::Tensor const* acceptedStateNodeIds)
{
    if (mIntermediateAuxConvStates.empty())
    {
        return;
    }
    auto const& stateConfig = mConfig.auxConvStateConfigs[0];
    int32_t const activeBatchSize = static_cast<int32_t>(acceptLengths.getShape()[0]);
    check::check(stateIndices.getDeviceType() == DeviceType::kGPU && stateIndices.getDataType() == DataType::kINT32
            && stateIndices.getShape().getNumDims() == 1 && stateIndices.getShape()[0] == activeBatchSize,
        "stateIndices must be a GPU INT32 tensor with shape [activeBatchSize].");
    auto const* layerInfos = static_cast<kernel::MtpLayerInfo const*>(mDeviceAuxMtpLayerInfos.rawPointer());
    int32_t const numLayers = static_cast<int32_t>(mIntermediateAuxConvStates.size());
    int32_t const verifySize = static_cast<int32_t>(mIntermediateAuxConvStates[0].getShape()[1]);
    int32_t const stateElements = stateConfig.convDim * stateConfig.convWidth;
    if (acceptedStateNodeIds == nullptr)
    {
        kernel::mtpScatterConvStates(layerInfos, numLayers, activeBatchSize, mConfig.maxBatchSize, verifySize,
            stateElements, acceptLengths.dataPointer<int32_t>(), stateIndices.dataPointer<int32_t>(), stream);
        return;
    }
    auto const idsShape = acceptedStateNodeIds->getShape();
    check::check(acceptedStateNodeIds->getDeviceType() == DeviceType::kGPU
            && acceptedStateNodeIds->getDataType() == DataType::kINT32 && idsShape.getNumDims() == 2
            && idsShape[0] == activeBatchSize,
        "acceptedStateNodeIds must be a GPU INT32 tensor with shape [activeBatchSize, maxAcceptLen].");
    kernel::mtpScatterAcceptedTreeConvStates(layerInfos, numLayers, activeBatchSize, mConfig.maxBatchSize, verifySize,
        stateElements, acceptedStateNodeIds->dataPointer<int32_t>(), static_cast<int32_t>(idsShape[1]),
        acceptLengths.dataPointer<int32_t>(), stateIndices.dataPointer<int32_t>(), stream);
}

size_t HybridCacheManager::auxConvStateBytesPerSlot(int32_t auxIdx) const
{
    auto const& stateConfig = mConfig.auxConvStateConfigs[static_cast<size_t>(auxIdx)];
    return static_cast<size_t>(stateConfig.convDim) * stateConfig.convWidth
        * rt::utils::getTypeSize(stateConfig.dataType);
}

std::vector<rt::Tensor> HybridCacheManager::captureAuxConvStates(int32_t slot, cudaStream_t stream)
{
    std::vector<rt::Tensor> result;
    result.reserve(mAuxConvStates.size());
    for (int32_t auxIdx = 0; auxIdx < numAuxConvStates(); ++auxIdx)
    {
        auto const& stateConfig = mConfig.auxConvStateConfigs[static_cast<size_t>(auxIdx)];
        size_t const bytes = auxConvStateBytesPerSlot(auxIdx);
        rt::Tensor saved({1, stateConfig.convDim, stateConfig.convWidth}, DeviceType::kGPU, stateConfig.dataType,
            "HybridCacheManager::capturedAuxConvState_" + std::to_string(auxIdx));
        CUDA_CHECK(cudaMemcpyAsync(saved.rawPointer(),
            static_cast<char const*>(mAuxConvStates[static_cast<size_t>(auxIdx)].rawPointer())
                + static_cast<size_t>(slot) * bytes,
            bytes, cudaMemcpyDeviceToDevice, stream));
        result.push_back(std::move(saved));
    }
    return result;
}

void HybridCacheManager::restoreAuxConvStates(std::vector<rt::Tensor> const& saved, int32_t slot, cudaStream_t stream)
{
    for (int32_t auxIdx = 0; auxIdx < numAuxConvStates(); ++auxIdx)
    {
        size_t const bytes = auxConvStateBytesPerSlot(auxIdx);
        auto* destination = static_cast<char*>(mAuxConvStates[static_cast<size_t>(auxIdx)].rawPointer())
            + static_cast<size_t>(slot) * bytes;
        if (static_cast<size_t>(auxIdx) < saved.size())
        {
            CUDA_CHECK(cudaMemcpyAsync(
                destination, saved[static_cast<size_t>(auxIdx)].rawPointer(), bytes, cudaMemcpyDeviceToDevice, stream));
        }
        else
        {
            CUDA_CHECK(cudaMemsetAsync(destination, 0, bytes, stream));
        }
    }
}

void HybridCacheManager::clearAuxConvStates(int32_t slot, cudaStream_t stream)
{
    for (int32_t auxIdx = 0; auxIdx < numAuxConvStates(); ++auxIdx)
    {
        size_t const bytes = auxConvStateBytesPerSlot(auxIdx);
        CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(mAuxConvStates[static_cast<size_t>(auxIdx)].rawPointer())
                + static_cast<size_t>(slot) * bytes,
            0, bytes, stream));
    }
}

rt::Tensor& HybridCacheManager::getConvState(int32_t absLayerIdx, int32_t stateIdx)
{
    if (stateIdx != 0)
    {
        for (int32_t auxIdx = 0; auxIdx < numAuxConvStates(); ++auxIdx)
        {
            auto const& stateConfig = mConfig.auxConvStateConfigs[static_cast<size_t>(auxIdx)];
            if (stateConfig.absLayerIdx == absLayerIdx && stateConfig.stateIdx == stateIdx)
            {
                return mAuxConvStates[static_cast<size_t>(auxIdx)];
            }
        }
        check::check(false,
            "getConvState: layer " + std::to_string(absLayerIdx) + " has no conv state at index "
                + std::to_string(stateIdx) + ".");
    }

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

void HybridCacheManager::appendSequence(int32_t committedLength, cudaStream_t stream)
{
    check::check(committedLength >= 0, "Appended sequence KV length must not be negative.");
    check::check(mActiveBatchSize < mConfig.maxBatchSize, "Cannot append a sequence to a full batch.");
    // The row is written before the batch grows to include it: a device enqueue that fails leaves
    // the manager exactly as it was, so the caller can give the reservation back.
    int32_t const row = mActiveBatchSize;
    rt::Tensor rowView(
        mDeviceKVCacheLengths.dataPointer<int32_t>() + row, {1}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    CUDA_CHECK(cudaMemsetAsync(rowView.rawPointer(), 0, sizeof(int32_t), stream));
    if (committedLength > 0)
    {
        kernel::incrementLengthTensor(rowView, committedLength, stream);
    }
    mActiveBatchSize += 1;
    check::check(mDeviceKVCacheLengths.reshape({mActiveBatchSize}), "Tensor reshape failed");
    if (committedLength > 0)
    {
        mKVCacheAllEmpty = false;
    }
}

void HybridCacheManager::restoreActiveBatchSize(int32_t activeBatchSize) noexcept
{
    mActiveBatchSize = activeBatchSize;
    bool const reshaped = mDeviceKVCacheLengths.reshape({activeBatchSize});
    if (!reshaped)
    {
        std::terminate();
    }
}

void HybridCacheManager::swapExecutionRows(int32_t rowA, int32_t rowB, cudaStream_t stream)
{
    check::check(rowA >= 0 && rowA < mActiveBatchSize && rowB >= 0 && rowB < mActiveBatchSize,
        "Swapped KV-length rows are outside the active batch.");
    kernel::swapLengthTensorEntries(mDeviceKVCacheLengths, rowA, rowB, stream);
}

void HybridCacheManager::materializeExecutionLengths(rt::Tensor const& executionLengths, cudaStream_t stream)
{
    check::check(executionLengths.getDeviceType() == DeviceType::kGPU, "Execution lengths tensor shall reside on GPU.");
    check::check(
        executionLengths.getDataType() == DataType::kINT32, "Execution lengths tensor shall have data type int32_t.");
    check::check(executionLengths.getShape().getNumDims() == 1, "Execution lengths tensor shall be one-dimensional.");
    int32_t const batchSize = static_cast<int32_t>(executionLengths.getShape()[0]);
    check::check(batchSize > 0 && batchSize <= mConfig.maxBatchSize,
        "Execution lengths batch size must be in range [1, maxBatchSize].");

    mActiveBatchSize = batchSize;
    check::check(mDeviceKVCacheLengths.reshape({batchSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mDeviceKVCacheLengths.rawPointer(), executionLengths.rawPointer(),
        static_cast<size_t>(batchSize) * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
}

bool HybridCacheManager::getKVCacheAllEmpty() const noexcept
{
    return mKVCacheAllEmpty;
}

// ------------------------------------------------------------------
// Compaction
// ------------------------------------------------------------------

void HybridCacheManager::compactKVCacheLengths(
    rt::Tensor const& batchMapping, int32_t oldBatch, int32_t newBatch, cudaStream_t stream)
{
    check::check(mActiveBatchSize == oldBatch, "KV length compaction old batch does not match the active batch size");
    check::check(mDeviceKVCacheLengths.reshape({oldBatch}), "Tensor reshape failed");
    kernel::compactTensorBatch(mDeviceKVCacheLengths, batchMapping, mDeviceKVCacheLengths, oldBatch, newBatch, stream);
    check::check(mDeviceKVCacheLengths.reshape({newBatch}), "Tensor reshape failed");
    mActiveBatchSize = newBatch;
}

void HybridCacheManager::clearResidentSlot(int32_t slot, cudaStream_t stream)
{
    check::check(slot >= 0 && slot < mConfig.maxBatchSize, "Resident cache slot is out of range.");
    mMambaCache.clearSlot(slot, stream);
    clearAuxConvStates(slot, stream);
}

// ------------------------------------------------------------------
// System prompt cache
// ------------------------------------------------------------------

std::vector<rt::Tensor> HybridCacheManager::captureKVCache(
    int32_t batchIdx, int32_t sequenceLength, cudaStream_t stream)
{
    check::check(
        !mKVCache.hasReducedKVCache(), "HybridCacheManager::captureKVCache does not support sparse SWA page mappings.");
    // The batched save/restore kernels only instantiate the `half` template today — match main's
    // contract of throwing loudly on unsupported dtypes instead of silently corrupting.
    check::check(mConfig.kvConfig.kvCacheType == trt_edgellm::DataType::kHALF,
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
    check::check(
        !mKVCache.hasReducedKVCache(), "HybridCacheManager::restoreKVCache does not support sparse SWA page mappings.");
    // See captureKVCache for the dtype contract.
    check::check(mConfig.kvConfig.kvCacheType == trt_edgellm::DataType::kHALF,
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
