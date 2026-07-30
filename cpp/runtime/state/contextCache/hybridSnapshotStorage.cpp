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

#include "runtime/state/contextCache/hybridSnapshotStorage.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/pagedKvTypes.h"

#include <limits>
#include <sstream>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace
{

size_t checkedMultiply(size_t lhs, size_t rhs, char const* description)
{
    ELLM_CHECK(rhs == 0 || lhs <= std::numeric_limits<size_t>::max() / rhs,
        std::string("Hybrid snapshot size overflow for ") + description);
    return lhs * rhs;
}

size_t checkedAdd(size_t lhs, size_t rhs, char const* description)
{
    ELLM_CHECK(lhs <= std::numeric_limits<size_t>::max() - rhs,
        std::string("Hybrid snapshot size overflow for ") + description);
    return lhs + rhs;
}

size_t recurrentStateBytes(MambaCacheManager::Config const& config)
{
    size_t elements = checkedMultiply(static_cast<size_t>(config.recurrentStateNumHeads),
        static_cast<size_t>(config.recurrentStateHeadDim), "recurrent heads");
    elements = checkedMultiply(elements, static_cast<size_t>(config.recurrentStateSize), "recurrent state");
    return checkedMultiply(elements, utils::getTypeSize(config.recurrentStateType), "recurrent dtype");
}

size_t convStateBytes(MambaCacheManager::Config const& config)
{
    size_t const elements = checkedMultiply(
        static_cast<size_t>(config.convDim), static_cast<size_t>(config.convKernel), "convolution state");
    return checkedMultiply(elements, utils::getTypeSize(config.convStateType), "convolution dtype");
}

std::byte* byteOffset(void* pointer, size_t offset)
{
    return static_cast<std::byte*>(pointer) + offset;
}

} // namespace

HybridSnapshotStorage::HybridSnapshotStorage(HybridCacheManager& cacheManager, int32_t recurrentSlotCount,
    int32_t partialKvSlotCount, HybridCacheManager* draftCacheManager, int32_t boundaryHiddenDim,
    nvinfer1::DataType boundaryHiddenType)
    : mCacheManager(cacheManager)
    , mDraftCacheManager(draftCacheManager)
    , mRecurrentSlotCount(recurrentSlotCount)
    , mPartialKvSlotCount(partialKvSlotCount)
    , mBoundaryHiddenDim(boundaryHiddenDim)
{
    ELLM_CHECK(recurrentSlotCount >= 0 && partialKvSlotCount >= 0, "Hybrid snapshot slot counts must be non-negative");
    ELLM_CHECK(boundaryHiddenDim >= 0, "Hybrid snapshot boundary hidden dim must be non-negative");
    ELLM_CHECK(boundaryHiddenDim == 0 || draftCacheManager != nullptr,
        "Boundary hidden storage is a Hybrid+MTP feature and requires a draft cache manager");
    HybridCacheManager::Config const& config = cacheManager.getConfig();
    ELLM_CHECK(
        config.mambaConfig.numRecurrentLayers > 0, "Hybrid snapshot storage requires at least one recurrent layer");
    ELLM_CHECK(config.kvConfig.numAttentionLayers > 0 || partialKvSlotCount == 0,
        "Pure-recurrent snapshot storage cannot allocate partial KV slots");

    if (recurrentSlotCount > 0)
    {
        mRecurrentSnapshots.reserve(static_cast<size_t>(config.mambaConfig.numRecurrentLayers));
        mConvSnapshots.reserve(static_cast<size_t>(config.mambaConfig.numRecurrentLayers));
        for (int32_t layer = 0; layer < config.mambaConfig.numRecurrentLayers; ++layer)
        {
            mRecurrentSnapshots.emplace_back(
                Tensor({recurrentSlotCount, config.mambaConfig.recurrentStateNumHeads,
                           config.mambaConfig.recurrentStateHeadDim, config.mambaConfig.recurrentStateSize},
                    DeviceType::kGPU, config.mambaConfig.recurrentStateType,
                    "HybridSnapshotStorage::recurrent_" + std::to_string(layer)));
            mConvSnapshots.emplace_back(Tensor(
                {recurrentSlotCount, config.mambaConfig.convDim, config.mambaConfig.convKernel}, DeviceType::kGPU,
                config.mambaConfig.convStateType, "HybridSnapshotStorage::conv_" + std::to_string(layer)));
        }
    }

    if (partialKvSlotCount > 0)
    {
        mPartialKvSnapshots.reserve(static_cast<size_t>(config.kvConfig.numAttentionLayers));
        for (int32_t layer = 0; layer < config.kvConfig.numAttentionLayers; ++layer)
        {
            KVLayerConfig const& layerConfig = config.kvConfig.layerConfigs[static_cast<size_t>(layer)];
            mPartialKvSnapshots.emplace_back(
                Tensor({partialKvSlotCount, 2, kTOKENS_PER_PAGE, layerConfig.numKVHeads, layerConfig.headDim},
                    DeviceType::kGPU, config.kvConfig.kvCacheType,
                    "HybridSnapshotStorage::partialKv_" + std::to_string(layer)));
        }
        if (mDraftCacheManager != nullptr)
        {
            HybridCacheManager::Config const& draftConfig = mDraftCacheManager->getConfig();
            ELLM_CHECK(draftConfig.kvConfig.numAttentionLayers > 0,
                "Hybrid MTP snapshot storage requires draft attention layers");
            mDraftPartialKvSnapshots.reserve(static_cast<size_t>(draftConfig.kvConfig.numAttentionLayers));
            for (int32_t layer = 0; layer < draftConfig.kvConfig.numAttentionLayers; ++layer)
            {
                KVLayerConfig const& layerConfig = draftConfig.kvConfig.layerConfigs[static_cast<size_t>(layer)];
                mDraftPartialKvSnapshots.emplace_back(
                    Tensor({partialKvSlotCount, 2, kTOKENS_PER_PAGE, layerConfig.numKVHeads, layerConfig.headDim},
                        DeviceType::kGPU, draftConfig.kvConfig.kvCacheType,
                        "HybridSnapshotStorage::draftPartialKv_" + std::to_string(layer)));
            }
        }
    }

    if (mBoundaryHiddenDim > 0 && recurrentSlotCount > 0)
    {
        mBoundaryHiddenSnapshot = Tensor({recurrentSlotCount, mBoundaryHiddenDim}, DeviceType::kGPU, boundaryHiddenType,
            "HybridSnapshotStorage::boundaryHidden");
    }
}

size_t HybridSnapshotStorage::recurrentBytesPerSlot(HybridCacheManager::Config const& config)
{
    ELLM_CHECK(
        config.mambaConfig.numRecurrentLayers >= 0, "Hybrid snapshot recurrent layer count must be non-negative");
    size_t const layerBytes
        = checkedAdd(recurrentStateBytes(config.mambaConfig), convStateBytes(config.mambaConfig), "recurrent layer");
    return checkedMultiply(layerBytes, static_cast<size_t>(config.mambaConfig.numRecurrentLayers), "recurrent layers");
}

size_t HybridSnapshotStorage::partialKvBytesPerSlot(
    HybridCacheManager::Config const& config, HybridCacheManager::Config const* draftConfig)
{
    auto kvBytes = [](KVCacheManager::Config const& kvConfig) {
        ELLM_CHECK(kvConfig.numAttentionLayers >= 0
                && kvConfig.layerConfigs.size() == static_cast<size_t>(kvConfig.numAttentionLayers),
            "Hybrid snapshot attention layer schema is inconsistent");
        size_t bytes{};
        for (KVLayerConfig const& layer : kvConfig.layerConfigs)
        {
            size_t elements = checkedMultiply(
                static_cast<size_t>(layer.numKVHeads), static_cast<size_t>(layer.headDim), "partial KV heads");
            elements = checkedMultiply(elements, static_cast<size_t>(2 * kTOKENS_PER_PAGE), "partial KV page");
            size_t const layerBytes
                = checkedMultiply(elements, utils::getTypeSize(kvConfig.kvCacheType), "partial KV dtype");
            bytes = checkedAdd(bytes, layerBytes, "partial KV layers");
        }
        return bytes;
    };

    size_t bytes = kvBytes(config.kvConfig);
    if (draftConfig != nullptr)
    {
        bytes = checkedAdd(bytes, kvBytes(draftConfig->kvConfig), "base and draft partial KV");
    }
    return bytes;
}

RecurrentStateSchemaId HybridSnapshotStorage::schemaId(
    HybridCacheManager::Config const& config, HybridCacheManager::Config const* draftConfig)
{
    std::ostringstream identity;
    identity << (draftConfig == nullptr ? "hybrid-snapshot-v1;layers=" : "hybrid-mtp-snapshot-v1;layers=");
    for (HybridCacheManager::LayerType const layer : config.layerTypes)
    {
        identity << static_cast<int32_t>(layer) << ',';
    }
    identity << ";kvType=" << static_cast<int32_t>(config.kvConfig.kvCacheType) << ";kv=";
    for (KVLayerConfig const& layer : config.kvConfig.layerConfigs)
    {
        identity << layer.numKVHeads << 'x' << layer.headDim << ',';
    }
    MambaCacheManager::Config const& mamba = config.mambaConfig;
    identity << ";recurrentLayers=" << mamba.numRecurrentLayers << ";recurrent=" << mamba.recurrentStateNumHeads << 'x'
             << mamba.recurrentStateHeadDim << 'x' << mamba.recurrentStateSize << ':'
             << static_cast<int32_t>(mamba.recurrentStateType) << ";conv=" << mamba.convDim << 'x' << mamba.convKernel
             << ':' << static_cast<int32_t>(mamba.convStateType);
    if (draftConfig != nullptr)
    {
        identity << ";draftKvType=" << static_cast<int32_t>(draftConfig->kvConfig.kvCacheType) << ";draftKv=";
        for (KVLayerConfig const& layer : draftConfig->kvConfig.layerConfigs)
        {
            identity << layer.numKVHeads << 'x' << layer.headDim << ',';
        }
    }
    return hashOpaqueIdentity(identity.str());
}

void HybridSnapshotStorage::captureRecurrent(int32_t snapshotSlot, int32_t batchSlot, cudaStream_t stream)
{
    validateRecurrentSlots(snapshotSlot, batchSlot);
    MambaCacheManager& mamba = mCacheManager.getMambaCacheManager();
    MambaCacheManager::Config const& config = mamba.getConfig();
    size_t const recurrentBytes = recurrentStateBytes(config);
    size_t const convBytes = convStateBytes(config);
    for (int32_t layer = 0; layer < mamba.numLayers(); ++layer)
    {
        CUDA_CHECK(cudaMemcpyAsync(byteOffset(mRecurrentSnapshots[static_cast<size_t>(layer)].rawPointer(),
                                       static_cast<size_t>(snapshotSlot) * recurrentBytes),
            byteOffset(mamba.getRecurrentState(layer).rawPointer(), static_cast<size_t>(batchSlot) * recurrentBytes),
            recurrentBytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(byteOffset(mConvSnapshots[static_cast<size_t>(layer)].rawPointer(),
                                       static_cast<size_t>(snapshotSlot) * convBytes),
            byteOffset(mamba.getConvState(layer).rawPointer(), static_cast<size_t>(batchSlot) * convBytes), convBytes,
            cudaMemcpyDeviceToDevice, stream));
    }
}

void HybridSnapshotStorage::restoreRecurrent(int32_t snapshotSlot, int32_t batchSlot, cudaStream_t stream)
{
    validateRecurrentSlots(snapshotSlot, batchSlot);
    MambaCacheManager& mamba = mCacheManager.getMambaCacheManager();
    MambaCacheManager::Config const& config = mamba.getConfig();
    size_t const recurrentBytes = recurrentStateBytes(config);
    size_t const convBytes = convStateBytes(config);
    for (int32_t layer = 0; layer < mamba.numLayers(); ++layer)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            byteOffset(mamba.getRecurrentState(layer).rawPointer(), static_cast<size_t>(batchSlot) * recurrentBytes),
            byteOffset(mRecurrentSnapshots[static_cast<size_t>(layer)].rawPointer(),
                static_cast<size_t>(snapshotSlot) * recurrentBytes),
            recurrentBytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            byteOffset(mamba.getConvState(layer).rawPointer(), static_cast<size_t>(batchSlot) * convBytes),
            byteOffset(
                mConvSnapshots[static_cast<size_t>(layer)].rawPointer(), static_cast<size_t>(snapshotSlot) * convBytes),
            convBytes, cudaMemcpyDeviceToDevice, stream));
    }
}

void HybridSnapshotStorage::capturePartialKv(
    int32_t snapshotSlot, PageId sourcePage, int32_t validTokenCount, cudaStream_t stream)
{
    capturePartialKv(mCacheManager, mPartialKvSnapshots, snapshotSlot, sourcePage, validTokenCount, stream);
}

void HybridSnapshotStorage::restorePartialKv(
    int32_t snapshotSlot, PageId destinationPage, int32_t validTokenCount, cudaStream_t stream)
{
    restorePartialKv(mCacheManager, mPartialKvSnapshots, snapshotSlot, destinationPage, validTokenCount, stream);
}

void HybridSnapshotStorage::capturePartialKv(
    int32_t snapshotSlot, PageId sourceBasePage, PageId sourceDraftPage, int32_t validTokenCount, cudaStream_t stream)
{
    ELLM_CHECK(mDraftCacheManager != nullptr, "Hybrid MTP partial snapshot storage has no draft cache manager");
    capturePartialKv(mCacheManager, mPartialKvSnapshots, snapshotSlot, sourceBasePage, validTokenCount, stream);
    capturePartialKv(
        *mDraftCacheManager, mDraftPartialKvSnapshots, snapshotSlot, sourceDraftPage, validTokenCount, stream);
}

void HybridSnapshotStorage::restorePartialKv(int32_t snapshotSlot, PageId destinationBasePage,
    PageId destinationDraftPage, int32_t validTokenCount, cudaStream_t stream)
{
    ELLM_CHECK(mDraftCacheManager != nullptr, "Hybrid MTP partial snapshot storage has no draft cache manager");
    restorePartialKv(mCacheManager, mPartialKvSnapshots, snapshotSlot, destinationBasePage, validTokenCount, stream);
    restorePartialKv(
        *mDraftCacheManager, mDraftPartialKvSnapshots, snapshotSlot, destinationDraftPage, validTokenCount, stream);
}

void HybridSnapshotStorage::capturePartialKv(HybridCacheManager& cacheManager, std::vector<Tensor>& snapshots,
    int32_t snapshotSlot, PageId sourcePage, int32_t validTokenCount, cudaStream_t stream)
{
    validatePartialSlots(cacheManager, snapshotSlot, sourcePage, validTokenCount);
    KVCacheManager& kvCache = cacheManager.getKVCacheManager();
    ELLM_CHECK(snapshots.size() == static_cast<size_t>(kvCache.numLayers()),
        "Hybrid partial KV snapshot layer count does not match the cache manager");
    for (int32_t layer = 0; layer < kvCache.numLayers(); ++layer)
    {
        KVLayerConfig const& config = kvCache.getLayerConfig(layer);
        size_t const tokenElements = checkedMultiply(
            static_cast<size_t>(config.numKVHeads), static_cast<size_t>(config.headDim), "partial KV token heads");
        size_t const tokenBytes = checkedMultiply(
            tokenElements, utils::getTypeSize(kvCache.getConfig().kvCacheType), "partial KV token dtype");
        size_t const pageBytes = checkedMultiply(tokenBytes, static_cast<size_t>(kTOKENS_PER_PAGE), "partial KV page");
        size_t const copyBytes = checkedMultiply(tokenBytes, static_cast<size_t>(validTokenCount), "partial KV copy");
        size_t const snapshotBase = static_cast<size_t>(snapshotSlot) * 2U * pageBytes;
        size_t const pageOffset = static_cast<size_t>(sourcePage) * pageBytes;
        Tensor& snapshot = snapshots[static_cast<size_t>(layer)];
        CUDA_CHECK(cudaMemcpyAsync(byteOffset(snapshot.rawPointer(), snapshotBase),
            byteOffset(kvCache.kPoolPtr(layer), pageOffset), copyBytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(byteOffset(snapshot.rawPointer(), snapshotBase + pageBytes),
            byteOffset(kvCache.vPoolPtr(layer), pageOffset), copyBytes, cudaMemcpyDeviceToDevice, stream));
    }
}

void HybridSnapshotStorage::restorePartialKv(HybridCacheManager& cacheManager, std::vector<Tensor>& snapshots,
    int32_t snapshotSlot, PageId destinationPage, int32_t validTokenCount, cudaStream_t stream)
{
    validatePartialSlots(cacheManager, snapshotSlot, destinationPage, validTokenCount);
    KVCacheManager& kvCache = cacheManager.getKVCacheManager();
    ELLM_CHECK(snapshots.size() == static_cast<size_t>(kvCache.numLayers()),
        "Hybrid partial KV snapshot layer count does not match the cache manager");
    for (int32_t layer = 0; layer < kvCache.numLayers(); ++layer)
    {
        KVLayerConfig const& config = kvCache.getLayerConfig(layer);
        size_t const tokenElements = checkedMultiply(
            static_cast<size_t>(config.numKVHeads), static_cast<size_t>(config.headDim), "partial KV token heads");
        size_t const tokenBytes = checkedMultiply(
            tokenElements, utils::getTypeSize(kvCache.getConfig().kvCacheType), "partial KV token dtype");
        size_t const pageBytes = checkedMultiply(tokenBytes, static_cast<size_t>(kTOKENS_PER_PAGE), "partial KV page");
        size_t const copyBytes = checkedMultiply(tokenBytes, static_cast<size_t>(validTokenCount), "partial KV copy");
        size_t const snapshotBase = static_cast<size_t>(snapshotSlot) * 2U * pageBytes;
        size_t const pageOffset = static_cast<size_t>(destinationPage) * pageBytes;
        Tensor& snapshot = snapshots[static_cast<size_t>(layer)];
        CUDA_CHECK(cudaMemcpyAsync(byteOffset(kvCache.kPoolPtr(layer), pageOffset),
            byteOffset(snapshot.rawPointer(), snapshotBase), copyBytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(byteOffset(kvCache.vPoolPtr(layer), pageOffset),
            byteOffset(snapshot.rawPointer(), snapshotBase + pageBytes), copyBytes, cudaMemcpyDeviceToDevice, stream));
    }
}

void HybridSnapshotStorage::captureBoundaryHidden(
    int32_t snapshotSlot, Tensor const& sourceHiddenStates, int32_t batchSlot, int32_t position, cudaStream_t stream)
{
    ELLM_CHECK(mBoundaryHiddenDim > 0 && !mBoundaryHiddenSnapshot.isEmpty(),
        "Boundary hidden storage is not configured for this deployment");
    ELLM_CHECK(
        snapshotSlot >= 0 && snapshotSlot < mRecurrentSlotCount, "Boundary hidden snapshot slot is out of range");
    Coords const& shape = sourceHiddenStates.getShape();
    ELLM_CHECK(shape.getNumDims() == 3 && shape[2] == mBoundaryHiddenDim,
        "Boundary hidden source must be a [batch, seq, boundaryHiddenDim] tensor");
    ELLM_CHECK(batchSlot >= 0 && batchSlot < shape[0] && position >= 0 && position < shape[1],
        "Boundary hidden capture indices are out of range");
    ELLM_CHECK(sourceHiddenStates.getDataType() == mBoundaryHiddenSnapshot.getDataType(),
        "Boundary hidden source dtype does not match the snapshot slab");
    size_t const rowBytes = checkedMultiply(static_cast<size_t>(mBoundaryHiddenDim),
        utils::getTypeSize(mBoundaryHiddenSnapshot.getDataType()), "boundary hidden row");
    size_t const sourceRow
        = static_cast<size_t>(batchSlot) * static_cast<size_t>(shape[1]) + static_cast<size_t>(position);
    CUDA_CHECK(
        cudaMemcpyAsync(byteOffset(mBoundaryHiddenSnapshot.rawPointer(), static_cast<size_t>(snapshotSlot) * rowBytes),
            static_cast<std::byte const*>(sourceHiddenStates.rawPointer()) + sourceRow * rowBytes, rowBytes,
            cudaMemcpyDeviceToDevice, stream));
}

void HybridSnapshotStorage::restoreBoundaryHidden(
    int32_t snapshotSlot, Tensor& destinationHiddenStates, int32_t batchSlot, int32_t position, cudaStream_t stream)
{
    ELLM_CHECK(mBoundaryHiddenDim > 0 && !mBoundaryHiddenSnapshot.isEmpty(),
        "Boundary hidden storage is not configured for this deployment");
    ELLM_CHECK(
        snapshotSlot >= 0 && snapshotSlot < mRecurrentSlotCount, "Boundary hidden snapshot slot is out of range");
    Coords const& shape = destinationHiddenStates.getShape();
    ELLM_CHECK(shape.getNumDims() == 3 && shape[2] == mBoundaryHiddenDim,
        "Boundary hidden destination must be a [batch, seq, boundaryHiddenDim] tensor");
    ELLM_CHECK(batchSlot >= 0 && batchSlot < shape[0] && position >= 0 && position < shape[1],
        "Boundary hidden restore indices are out of range");
    ELLM_CHECK(destinationHiddenStates.getDataType() == mBoundaryHiddenSnapshot.getDataType(),
        "Boundary hidden destination dtype does not match the snapshot slab");
    size_t const rowBytes = checkedMultiply(static_cast<size_t>(mBoundaryHiddenDim),
        utils::getTypeSize(mBoundaryHiddenSnapshot.getDataType()), "boundary hidden row");
    size_t const destRow
        = (static_cast<size_t>(batchSlot) * static_cast<size_t>(shape[1]) + static_cast<size_t>(position));
    CUDA_CHECK(cudaMemcpyAsync(byteOffset(destinationHiddenStates.rawPointer(), destRow * rowBytes),
        byteOffset(mBoundaryHiddenSnapshot.rawPointer(), static_cast<size_t>(snapshotSlot) * rowBytes), rowBytes,
        cudaMemcpyDeviceToDevice, stream));
}

int32_t HybridSnapshotStorage::recurrentSlotCount() const noexcept
{
    return mRecurrentSlotCount;
}

int32_t HybridSnapshotStorage::boundaryHiddenDim() const noexcept
{
    return mBoundaryHiddenDim;
}

int32_t HybridSnapshotStorage::partialKvSlotCount() const noexcept
{
    return mPartialKvSlotCount;
}

void HybridSnapshotStorage::validateRecurrentSlots(int32_t snapshotSlot, int32_t batchSlot) const
{
    ELLM_CHECK(
        snapshotSlot >= 0 && snapshotSlot < mRecurrentSlotCount, "Hybrid recurrent snapshot slot is out of range");
    ELLM_CHECK(batchSlot >= 0 && batchSlot < mCacheManager.getConfig().maxBatchSize,
        "Hybrid recurrent live batch slot is out of range");
}

void HybridSnapshotStorage::validatePartialSlots(
    HybridCacheManager& cacheManager, int32_t snapshotSlot, PageId page, int32_t validTokenCount) const
{
    ELLM_CHECK(
        snapshotSlot >= 0 && snapshotSlot < mPartialKvSlotCount, "Hybrid partial KV snapshot slot is out of range");
    ELLM_CHECK(page >= 0 && page < cacheManager.getKVCacheManager().numPages(),
        "Hybrid partial KV physical page is out of range");
    // Hybrid+MTP retains its boundary token in a private partial page even when the checkpoint length is page-aligned,
    // in which case that page holds a full kTOKENS_PER_PAGE tokens; allow the inclusive upper bound.
    ELLM_CHECK(validTokenCount > 0 && validTokenCount <= kTOKENS_PER_PAGE,
        "Hybrid partial KV snapshot must contain a non-empty page");
}

} // namespace rt
} // namespace trt_edgellm
