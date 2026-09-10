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

void validateGpuTensorLayout(Tensor const& tensor, Coords const& expectedShape, nvinfer1::DataType expectedType,
    size_t expectedOuterBytes, std::string const& description)
{
    ELLM_CHECK(tensor.getShape() == expectedShape,
        description + " shape " + tensor.getShape().formatString() + " does not match expected shape "
            + expectedShape.formatString());
    ELLM_CHECK(tensor.getDeviceType() == DeviceType::kGPU, description + " must reside on the GPU");
    ELLM_CHECK(tensor.getDataType() == expectedType, description + " dtype does not match its cache configuration");
    ELLM_CHECK(tensor.rawPointer() != nullptr, description + " has no backing allocation");
    ELLM_CHECK(tensor.getStride(0) >= 0, description + " has an invalid outer stride");
    size_t const actualOuterBytes = checkedMultiply(
        static_cast<size_t>(tensor.getStride(0)), utils::getTypeSize(tensor.getDataType()), description.c_str());
    ELLM_CHECK(actualOuterBytes == expectedOuterBytes,
        description + " outer stride does not match the byte-offset layout used by hybrid snapshots");
    size_t const expectedBytes
        = checkedMultiply(expectedOuterBytes, static_cast<size_t>(expectedShape[0]), description.c_str());
    ELLM_CHECK(tensor.getMemoryCapacity() >= 0 && static_cast<size_t>(tensor.getMemoryCapacity()) >= expectedBytes,
        description + " backing allocation is smaller than its declared layout");
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
    MambaCacheManager& mamba = cacheManager.getMambaCacheManager();
    KVCacheManager& kv = cacheManager.getKVCacheManager();
    MambaCacheManager::Config const& mambaConfig = mamba.getConfig();
    KVCacheManager::Config const& kvConfig = kv.getConfig();
    ELLM_CHECK(mamba.numLayers() > 0, "Hybrid snapshot storage requires at least one recurrent layer");
    ELLM_CHECK(kv.numLayers() > 0 || partialKvSlotCount == 0,
        "Pure-recurrent snapshot storage cannot allocate partial KV slots");

    size_t const recurrentBytes = recurrentStateBytes(mambaConfig);
    size_t const convBytes = convStateBytes(mambaConfig);
    for (int32_t layer = 0; layer < mamba.numLayers(); ++layer)
    {
        std::string const layerSuffix = std::to_string(layer);
        validateGpuTensorLayout(mamba.getRecurrentState(layer),
            {mambaConfig.maxBatchSize, mambaConfig.recurrentStateNumHeads, mambaConfig.recurrentStateHeadDim,
                mambaConfig.recurrentStateSize},
            mambaConfig.recurrentStateType, recurrentBytes, "Live recurrent state layer " + layerSuffix);
        validateGpuTensorLayout(mamba.getConvState(layer),
            {mambaConfig.maxBatchSize, mambaConfig.convDim, mambaConfig.convKernel}, mambaConfig.convStateType,
            convBytes, "Live convolution state layer " + layerSuffix);
    }

    for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
    {
        KVLayerConfig const& layerConfig = kv.getLayerConfig(layer);
        size_t const tokenElements = checkedMultiply(
            static_cast<size_t>(layerConfig.numKVHeads), static_cast<size_t>(layerConfig.headDim), "live KV token");
        size_t const pageElements
            = checkedMultiply(tokenElements, static_cast<size_t>(kTOKENS_PER_PAGE), "live KV page");
        size_t const pageBytes
            = checkedMultiply(pageElements, utils::getTypeSize(kvConfig.kvCacheType), "live KV page dtype");
        size_t const poolHalfBytes
            = checkedMultiply(pageBytes, static_cast<size_t>(kv.numPages()), "live KV pool half");
        Tensor const& pool = kv.getCombinedKVCache(layer);
        std::string const description = "Live KV pool layer " + std::to_string(layer);
        validateGpuTensorLayout(pool, {2, kv.numPages(), kTOKENS_PER_PAGE, layerConfig.numKVHeads, layerConfig.headDim},
            kvConfig.kvCacheType, poolHalfBytes, description);
        ELLM_CHECK(pool.getStride(1) >= 0
                && checkedMultiply(static_cast<size_t>(pool.getStride(1)), utils::getTypeSize(pool.getDataType()),
                       description.c_str())
                    == pageBytes,
            description + " page stride does not match the page-byte layout used by hybrid snapshots");
        ELLM_CHECK(pool.rawPointer() == kv.kPoolPtr(layer), description + " K pool pointer does not match its view");
        ELLM_CHECK(byteOffset(kv.kPoolPtr(layer), poolHalfBytes) == kv.vPoolPtr(layer),
            description + " V pool pointer does not follow the contiguous K pool");
    }

    if (recurrentSlotCount > 0)
    {
        mRecurrentSnapshots.reserve(static_cast<size_t>(mamba.numLayers()));
        mConvSnapshots.reserve(static_cast<size_t>(mamba.numLayers()));
        for (int32_t layer = 0; layer < mamba.numLayers(); ++layer)
        {
            mRecurrentSnapshots.emplace_back(
                Tensor({recurrentSlotCount, mambaConfig.recurrentStateNumHeads, mambaConfig.recurrentStateHeadDim,
                           mambaConfig.recurrentStateSize},
                    DeviceType::kGPU, mambaConfig.recurrentStateType,
                    "HybridSnapshotStorage::recurrent_" + std::to_string(layer)));
            mConvSnapshots.emplace_back(Tensor({recurrentSlotCount, mambaConfig.convDim, mambaConfig.convKernel},
                DeviceType::kGPU, mambaConfig.convStateType, "HybridSnapshotStorage::conv_" + std::to_string(layer)));
            validateGpuTensorLayout(mRecurrentSnapshots.back(),
                {recurrentSlotCount, mambaConfig.recurrentStateNumHeads, mambaConfig.recurrentStateHeadDim,
                    mambaConfig.recurrentStateSize},
                mambaConfig.recurrentStateType, recurrentBytes, "Recurrent snapshot layer " + std::to_string(layer));
            validateGpuTensorLayout(mConvSnapshots.back(),
                {recurrentSlotCount, mambaConfig.convDim, mambaConfig.convKernel}, mambaConfig.convStateType, convBytes,
                "Convolution snapshot layer " + std::to_string(layer));
        }
    }

    if (partialKvSlotCount > 0)
    {
        mPartialKvSnapshots.reserve(static_cast<size_t>(kv.numLayers()));
        for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
        {
            KVLayerConfig const& layerConfig = kv.getLayerConfig(layer);
            size_t const tokenElements = checkedMultiply(static_cast<size_t>(layerConfig.numKVHeads),
                static_cast<size_t>(layerConfig.headDim), "partial KV snapshot token");
            size_t const pageBytes = checkedMultiply(
                checkedMultiply(tokenElements, static_cast<size_t>(kTOKENS_PER_PAGE), "partial KV snapshot page"),
                utils::getTypeSize(kvConfig.kvCacheType), "partial KV snapshot dtype");
            mPartialKvSnapshots.emplace_back(Tensor(
                {partialKvSlotCount, 2, kTOKENS_PER_PAGE, layerConfig.numKVHeads, layerConfig.headDim},
                DeviceType::kGPU, kvConfig.kvCacheType, "HybridSnapshotStorage::partialKv_" + std::to_string(layer)));
            validateGpuTensorLayout(mPartialKvSnapshots.back(),
                {partialKvSlotCount, 2, kTOKENS_PER_PAGE, layerConfig.numKVHeads, layerConfig.headDim},
                kvConfig.kvCacheType, checkedMultiply(2U, pageBytes, "partial KV snapshot slot"),
                "Partial KV snapshot layer " + std::to_string(layer));
        }

        if (mDraftCacheManager != nullptr)
        {
            KVCacheManager& draftKv = mDraftCacheManager->getKVCacheManager();
            KVCacheManager::Config const& draftKvConfig = draftKv.getConfig();
            ELLM_CHECK(
                draftKv.numLayers() > 0, "Hybrid MTP snapshot storage requires at least one draft attention layer");
            mDraftPartialKvSnapshots.reserve(static_cast<size_t>(draftKv.numLayers()));
            for (int32_t layer = 0; layer < draftKv.numLayers(); ++layer)
            {
                KVLayerConfig const& layerConfig = draftKv.getLayerConfig(layer);
                size_t const tokenElements = checkedMultiply(static_cast<size_t>(layerConfig.numKVHeads),
                    static_cast<size_t>(layerConfig.headDim), "draft partial KV snapshot token");
                size_t const pageBytes
                    = checkedMultiply(checkedMultiply(tokenElements, static_cast<size_t>(kTOKENS_PER_PAGE),
                                          "draft partial KV snapshot page"),
                        utils::getTypeSize(draftKvConfig.kvCacheType), "draft partial KV snapshot dtype");
                mDraftPartialKvSnapshots.emplace_back(
                    Tensor({partialKvSlotCount, 2, kTOKENS_PER_PAGE, layerConfig.numKVHeads, layerConfig.headDim},
                        DeviceType::kGPU, draftKvConfig.kvCacheType,
                        "HybridSnapshotStorage::draftPartialKv_" + std::to_string(layer)));
                validateGpuTensorLayout(mDraftPartialKvSnapshots.back(),
                    {partialKvSlotCount, 2, kTOKENS_PER_PAGE, layerConfig.numKVHeads, layerConfig.headDim},
                    draftKvConfig.kvCacheType, checkedMultiply(2U, pageBytes, "draft partial KV snapshot slot"),
                    "Draft partial KV snapshot layer " + std::to_string(layer));
            }
        }
    }

    if (mBoundaryHiddenDim > 0 && recurrentSlotCount > 0)
    {
        size_t const rowBytes = checkedMultiply(
            static_cast<size_t>(mBoundaryHiddenDim), utils::getTypeSize(boundaryHiddenType), "boundary hidden row");
        mBoundaryHiddenSnapshot = Tensor({recurrentSlotCount, mBoundaryHiddenDim}, DeviceType::kGPU, boundaryHiddenType,
            "HybridSnapshotStorage::boundaryHidden");
        validateGpuTensorLayout(mBoundaryHiddenSnapshot, {recurrentSlotCount, mBoundaryHiddenDim}, boundaryHiddenType,
            rowBytes, "Boundary hidden snapshot");
    }
}

size_t HybridSnapshotStorage::recurrentBytesPerSlot(MambaCacheManager::Config const& config)
{
    ELLM_CHECK(config.numRecurrentLayers >= 0, "Hybrid snapshot recurrent layer count must be non-negative");
    size_t const layerBytes = checkedAdd(recurrentStateBytes(config), convStateBytes(config), "recurrent layer");
    return checkedMultiply(layerBytes, static_cast<size_t>(config.numRecurrentLayers), "recurrent layers");
}

size_t HybridSnapshotStorage::partialKvBytesPerSlot(KVCacheManager::Config const& config)
{
    ELLM_CHECK(
        config.numAttentionLayers >= 0 && config.layerConfigs.size() == static_cast<size_t>(config.numAttentionLayers),
        "Hybrid snapshot attention layer schema is inconsistent");
    size_t bytes{};
    for (KVLayerConfig const& layer : config.layerConfigs)
    {
        size_t elements = checkedMultiply(
            static_cast<size_t>(layer.numKVHeads), static_cast<size_t>(layer.headDim), "partial KV heads");
        elements = checkedMultiply(elements, static_cast<size_t>(2 * kTOKENS_PER_PAGE), "partial KV page");
        bytes = checkedAdd(bytes, checkedMultiply(elements, utils::getTypeSize(config.kvCacheType), "partial KV dtype"),
            "partial KV layers");
    }
    return bytes;
}

size_t HybridSnapshotStorage::boundaryHiddenBytesPerSlot(int32_t boundaryHiddenDim, nvinfer1::DataType type)
{
    ELLM_CHECK(boundaryHiddenDim >= 0, "Hybrid snapshot boundary hidden dimension must be non-negative");
    return checkedMultiply(
        static_cast<size_t>(boundaryHiddenDim), utils::getTypeSize(type), "boundary hidden row bytes");
}

void HybridSnapshotStorage::zeroRecurrent(int32_t batchSlot, cudaStream_t stream)
{
    validateBatchSlot(batchSlot);
    MambaCacheManager& mamba = mCacheManager.getMambaCacheManager();
    MambaCacheManager::Config const& config = mamba.getConfig();
    size_t const recurrentBytes = recurrentStateBytes(config);
    size_t const convBytes = convStateBytes(config);
    for (int32_t layer = 0; layer < mamba.numLayers(); ++layer)
    {
        CUDA_CHECK(cudaMemsetAsync(
            byteOffset(mamba.getRecurrentState(layer).rawPointer(), static_cast<size_t>(batchSlot) * recurrentBytes), 0,
            recurrentBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(
            byteOffset(mamba.getConvState(layer).rawPointer(), static_cast<size_t>(batchSlot) * convBytes), 0,
            convBytes, stream));
    }
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

int32_t HybridSnapshotStorage::recurrentSlotCount() const noexcept
{
    return mRecurrentSlotCount;
}

int32_t HybridSnapshotStorage::partialKvSlotCount() const noexcept
{
    return mPartialKvSlotCount;
}

int32_t HybridSnapshotStorage::boundaryHiddenDim() const noexcept
{
    return mBoundaryHiddenDim;
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

void HybridSnapshotStorage::validateRecurrentSlots(int32_t snapshotSlot, int32_t batchSlot) const
{
    ELLM_CHECK(
        snapshotSlot >= 0 && snapshotSlot < mRecurrentSlotCount, "Hybrid recurrent snapshot slot is out of range");
    validateBatchSlot(batchSlot);
}

void HybridSnapshotStorage::validateBatchSlot(int32_t batchSlot) const
{
    ELLM_CHECK(batchSlot >= 0 && batchSlot < mCacheManager.getMambaCacheManager().getConfig().maxBatchSize,
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
