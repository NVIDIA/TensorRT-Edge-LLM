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

#include "runtime/linearKVCache.h"
#include "common/logger.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include <cuda_bf16.h>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace rt
{

LinearKVCache::LinearKVCache(CacheConfig const& config, cudaStream_t stream)
    : mConfig(config)
{
    check::check(
        mConfig.kvCacheTypeTRT == nvinfer1::DataType::kHALF || mConfig.kvCacheTypeTRT == nvinfer1::DataType::kFP8,
        "Unsupported KV cache dtype.");
    mDeviceKVCache = rt::Tensor({mConfig.numAttentionLayers, mConfig.maxBatchSize, 2, mConfig.numKVHeads,
                                    mConfig.maxSequenceLength, mConfig.headDim},
        DeviceType::kGPU, mConfig.kvCacheTypeTRT, "LinearKVCache::mDeviceKVCache");
    int64_t const kvCacheVolume = mConfig.numAttentionLayers * mConfig.maxBatchSize * 2 * mConfig.numKVHeads
        * mConfig.maxSequenceLength * mConfig.headDim;
    size_t const kvCacheElemSize = rt::utils::getTypeSize(mConfig.kvCacheTypeTRT);
    char const* kvCacheTypeStr = (mConfig.kvCacheTypeTRT == nvinfer1::DataType::kHALF) ? "kHALF" : "kFP8";
    LOG_DEBUG(
        "KVCache(dtype=%s) of shape [%ld, %ld, %ld, %ld, %ld, %ld] allocated on GPU with size: %ld bytes (%.2f MB)",
        kvCacheTypeStr, mConfig.numAttentionLayers, mConfig.maxBatchSize, 2, mConfig.numKVHeads,
        mConfig.maxSequenceLength, mConfig.headDim, kvCacheVolume * static_cast<int64_t>(kvCacheElemSize),
        static_cast<float>(kvCacheVolume * static_cast<int64_t>(kvCacheElemSize)) / (1024.0 * 1024.0));
    mDeviceKVCacheLengths = rt::Tensor(
        {mConfig.maxBatchSize}, DeviceType::kGPU, DataType::kINT32, "LinearKVCache::mDeviceKVCacheLengths");
    CUDA_CHECK(
        cudaMemsetAsync(mDeviceKVCacheLengths.rawPointer(), 0, mDeviceKVCacheLengths.getMemoryCapacity(), stream));

    if (mConfig.numLinearAttnLayers > 0)
    {
        mDeviceRecurrentStates
            = rt::Tensor({mConfig.numLinearAttnLayers, mConfig.maxBatchSize, mConfig.recurrentStateNumHeads,
                             mConfig.recurrentStateHeadDim, mConfig.recurrentStateSize},
                DeviceType::kGPU, mConfig.recurrentStateType, "LinearKVCache::mDeviceRecurrentStates");
        CUDA_CHECK(cudaMemsetAsync(
            mDeviceRecurrentStates.rawPointer(), 0, mDeviceRecurrentStates.getMemoryCapacity(), stream));

        mDeviceConvStates
            = rt::Tensor({mConfig.numLinearAttnLayers, mConfig.maxBatchSize, mConfig.convDim, mConfig.convKernel},
                DeviceType::kGPU, mConfig.convStateType, "LinearKVCache::mDeviceConvStates");
        CUDA_CHECK(cudaMemsetAsync(mDeviceConvStates.rawPointer(), 0, mDeviceConvStates.getMemoryCapacity(), stream));
    }
}

LinearKVCache::~LinearKVCache() noexcept {}

LinearKVCache::LinearKVCache(LinearKVCache&& other) noexcept
{
    mConfig = other.mConfig;
    mActiveBatchSize = other.mActiveBatchSize;
    mKVCacheAllEmpty = other.mKVCacheAllEmpty;
    mDeviceKVCache = std::move(other.mDeviceKVCache);
    mDeviceKVCacheLengths = std::move(other.mDeviceKVCacheLengths);
    mDeviceRecurrentStates = std::move(other.mDeviceRecurrentStates);
    mDeviceConvStates = std::move(other.mDeviceConvStates);

    other.mConfig = CacheConfig{};
    other.mActiveBatchSize = 0;
    other.mKVCacheAllEmpty = true;
}

LinearKVCache& LinearKVCache::operator=(LinearKVCache&& other) noexcept
{
    if (this != &other)
    {
        mConfig = other.mConfig;
        mKVCacheAllEmpty = other.mKVCacheAllEmpty;
        mActiveBatchSize = other.mActiveBatchSize;
        mDeviceKVCache = std::move(other.mDeviceKVCache);
        mDeviceKVCacheLengths = std::move(other.mDeviceKVCacheLengths);
        mDeviceRecurrentStates = std::move(other.mDeviceRecurrentStates);
        mDeviceConvStates = std::move(other.mDeviceConvStates);

        other.mConfig = CacheConfig{};
        other.mActiveBatchSize = 0;
        other.mKVCacheAllEmpty = true;
    }
    return *this;
}

rt::Tensor LinearKVCache::getCombinedKVCacheForDecoderLayer(int32_t decoderLayerIdx) noexcept
{
    int64_t const kvCacheOffset
        = decoderLayerIdx * mConfig.maxBatchSize * 2 * mConfig.numKVHeads * mConfig.maxSequenceLength * mConfig.headDim;

    size_t const elemSize = rt::utils::getTypeSize(mConfig.kvCacheTypeTRT);
    void* kvCachePtr = static_cast<void*>(static_cast<char*>(mDeviceKVCache.rawPointer()) + kvCacheOffset * elemSize);

    return rt::Tensor(kvCachePtr,
        {mConfig.maxBatchSize, 2, mConfig.numKVHeads, mConfig.maxSequenceLength, mConfig.headDim}, DeviceType::kGPU,
        mConfig.kvCacheTypeTRT);
}

std::pair<rt::Tensor, rt::Tensor> LinearKVCache::getSeparateKVCacheForDecoderLayer(int32_t decoderLayerIdx) noexcept
{
    // Get the combined KV cache for this layer from base class
    rt::Tensor kvCache = LinearKVCache::getCombinedKVCacheForDecoderLayer(decoderLayerIdx);

    // The KV cache has shape: [maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]
    // K cache is at index 0 of dimension 2, so we just need to point to the beginning
    // and reshape to remove the "2" dimension

    CacheConfig config = getConfig();
    void* kvCachePtr = static_cast<void*>(kvCache.rawPointer());

    void* kCachePtr = static_cast<void*>(kvCachePtr);
    rt::Tensor kCache
        = rt::Tensor(kCachePtr, {config.maxBatchSize, config.numKVHeads, config.maxSequenceLength, config.headDim},
            DeviceType::kGPU, mConfig.kvCacheTypeTRT);

    // Calculate offset to V cache: skip the entire K cache portion
    // Offset = maxBatchSize * 1 (K portion) * numKVHeads * maxSequenceLength * headDim
    int64_t vCacheOffset = config.maxBatchSize * config.numKVHeads * config.maxSequenceLength * config.headDim
        * static_cast<int64_t>(rt::utils::getTypeSize(mConfig.kvCacheTypeTRT));
    void* vCachePtr = static_cast<void*>(static_cast<char*>(kvCachePtr) + vCacheOffset);

    rt::Tensor vCache
        = rt::Tensor(vCachePtr, {config.maxBatchSize, config.numKVHeads, config.maxSequenceLength, config.headDim},
            DeviceType::kGPU, mConfig.kvCacheTypeTRT);
    return {std::move(kCache), std::move(vCache)};
}

rt::Tensor LinearKVCache::getKVCacheBuffer() noexcept
{
    return rt::Tensor(mDeviceKVCache.rawPointer(),
        {mConfig.numAttentionLayers, mConfig.maxBatchSize, 2, mConfig.numKVHeads, mConfig.maxSequenceLength,
            mConfig.headDim},
        DeviceType::kGPU, mConfig.kvCacheTypeTRT);
}

void LinearKVCache::clearRecurrentStates(cudaStream_t stream)
{
    if (mConfig.numLinearAttnLayers == 0)
    {
        return;
    }
    CUDA_CHECK(
        cudaMemsetAsync(mDeviceRecurrentStates.rawPointer(), 0, mDeviceRecurrentStates.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(mDeviceConvStates.rawPointer(), 0, mDeviceConvStates.getMemoryCapacity(), stream));
}

rt::Tensor LinearKVCache::getRecurrentStateForLayer(int32_t recurrentLayerIdx) noexcept
{
    size_t const elemSize = rt::utils::getTypeSize(mConfig.recurrentStateType);
    int64_t const perLayerElems = mConfig.maxBatchSize * mConfig.recurrentStateNumHeads * mConfig.recurrentStateHeadDim
        * mConfig.recurrentStateSize;
    void* ptr = static_cast<char*>(mDeviceRecurrentStates.rawPointer()) + recurrentLayerIdx * perLayerElems * elemSize;
    return rt::Tensor(ptr,
        {mConfig.maxBatchSize, mConfig.recurrentStateNumHeads, mConfig.recurrentStateHeadDim,
            mConfig.recurrentStateSize},
        DeviceType::kGPU, mConfig.recurrentStateType);
}

rt::Tensor LinearKVCache::getConvStateForLayer(int32_t recurrentLayerIdx) noexcept
{
    size_t const elemSize = rt::utils::getTypeSize(mConfig.convStateType);
    int64_t const perLayerElems = mConfig.maxBatchSize * mConfig.convDim * mConfig.convKernel;
    void* ptr = static_cast<char*>(mDeviceConvStates.rawPointer()) + recurrentLayerIdx * perLayerElems * elemSize;
    return rt::Tensor(
        ptr, {mConfig.maxBatchSize, mConfig.convDim, mConfig.convKernel}, DeviceType::kGPU, mConfig.convStateType);
}

std::vector<rt::Tensor> LinearKVCache::captureRecurrentStates(int32_t batchIdx, cudaStream_t stream)
{
    std::vector<rt::Tensor> result;
    if (mConfig.numLinearAttnLayers == 0)
    {
        return result;
    }
    size_t const elemSize = rt::utils::getTypeSize(mConfig.recurrentStateType);
    int64_t const perLayerElems = mConfig.maxBatchSize * mConfig.recurrentStateNumHeads * mConfig.recurrentStateHeadDim
        * mConfig.recurrentStateSize;
    int64_t const perBatchElems
        = mConfig.recurrentStateNumHeads * mConfig.recurrentStateHeadDim * mConfig.recurrentStateSize;
    size_t const perBatchBytes = static_cast<size_t>(perBatchElems) * elemSize;

    result.reserve(mConfig.numLinearAttnLayers);
    for (int32_t layer = 0; layer < mConfig.numLinearAttnLayers; ++layer)
    {
        void const* src = static_cast<char const*>(mDeviceRecurrentStates.rawPointer())
            + static_cast<size_t>(layer * perLayerElems + batchIdx * perBatchElems) * elemSize;
        rt::Tensor saved({1, mConfig.recurrentStateNumHeads, mConfig.recurrentStateHeadDim, mConfig.recurrentStateSize},
            DeviceType::kGPU, mConfig.recurrentStateType,
            "LinearKVCache::capturedRecurrentState_" + std::to_string(layer));
        CUDA_CHECK(cudaMemcpyAsync(saved.rawPointer(), src, perBatchBytes, cudaMemcpyDeviceToDevice, stream));
        result.push_back(std::move(saved));
    }
    return result;
}

std::vector<rt::Tensor> LinearKVCache::captureConvStates(int32_t batchIdx, cudaStream_t stream)
{
    std::vector<rt::Tensor> result;
    if (mConfig.numLinearAttnLayers == 0)
    {
        return result;
    }
    size_t const elemSize = rt::utils::getTypeSize(mConfig.convStateType);
    int64_t const perLayerElems = mConfig.maxBatchSize * mConfig.convDim * mConfig.convKernel;
    int64_t const perBatchElems = mConfig.convDim * mConfig.convKernel;
    size_t const perBatchBytes = static_cast<size_t>(perBatchElems) * elemSize;

    result.reserve(mConfig.numLinearAttnLayers);
    for (int32_t layer = 0; layer < mConfig.numLinearAttnLayers; ++layer)
    {
        void const* src = static_cast<char const*>(mDeviceConvStates.rawPointer())
            + static_cast<size_t>(layer * perLayerElems + batchIdx * perBatchElems) * elemSize;
        rt::Tensor saved({1, mConfig.convDim, mConfig.convKernel}, DeviceType::kGPU, mConfig.convStateType,
            "LinearKVCache::capturedConvState_" + std::to_string(layer));
        CUDA_CHECK(cudaMemcpyAsync(saved.rawPointer(), src, perBatchBytes, cudaMemcpyDeviceToDevice, stream));
        result.push_back(std::move(saved));
    }
    return result;
}

void LinearKVCache::resetForNewSequences(rt::Tensor const& reuseKVCacheLengths, cudaStream_t stream)
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

    // If all reuseSequenceLengths are 0, then we can set flag mKVCacheAllEmpty to true.
    int32_t const* reuseSequenceLengthsData = reuseKVCacheLengths.dataPointer<int32_t>();
    bool allEmpty{true};
    for (int32_t i = 0; i < batchSize; ++i)
    {
        if (reuseSequenceLengthsData[i] != 0)
        {
            allEmpty = false;
            break;
        }
    }
    mKVCacheAllEmpty = allEmpty;
    CUDA_CHECK(cudaMemcpyAsync(mDeviceKVCacheLengths.rawPointer(), reuseKVCacheLengths.rawPointer(),
        reuseKVCacheLengths.getMemoryCapacity(), cudaMemcpyHostToDevice, stream));
}

void LinearKVCache::commitSequenceLength(rt::Tensor const& newContextLengths, cudaStream_t stream)
{
    check::check(newContextLengths.getDataType() == DataType::kINT32,
        "The newContextLengths tensor shall have data type of int32_t.");
    check::check(
        newContextLengths.getDeviceType() == DeviceType::kGPU, "The newContextLengths tensor shall reside on GPU.");
    check::check(newContextLengths.getShape()[0] == mActiveBatchSize,
        "The newContextLengths tensor shall have the same batch size as the active batch size.");

    kernel::incrementLengthTensor(mDeviceKVCacheLengths, newContextLengths, stream);

    // Set flag to false since we have committed a new sequence length.
    mKVCacheAllEmpty = false;
}

void LinearKVCache::commitSequenceLength(int32_t increment, cudaStream_t stream)
{
    kernel::incrementLengthTensor(mDeviceKVCacheLengths, increment, stream);

    // Set flag to false since we have committed a new sequence length.
    mKVCacheAllEmpty = false;
}

rt::Tensor& LinearKVCache::getKVCacheLengths() noexcept
{
    return mDeviceKVCacheLengths;
}

LinearKVCache::CacheConfig LinearKVCache::getConfig() const noexcept
{
    return mConfig;
}

int32_t LinearKVCache::getActiveBatchSize() const noexcept
{
    return mActiveBatchSize;
}

bool LinearKVCache::getKVCacheAllEmpty() const noexcept
{
    return mKVCacheAllEmpty;
}

void LinearKVCache::setActiveBatchSize(int32_t newActiveBatchSize)
{
    check::check(newActiveBatchSize >= 0 && newActiveBatchSize <= mConfig.maxBatchSize,
        "Invalid active batch size: must be in range [0, maxBatchSize]");
    mActiveBatchSize = newActiveBatchSize;
    check::check(mDeviceKVCacheLengths.reshape({mActiveBatchSize}), "Tensor reshape failed");
}

} // namespace rt
} // namespace trt_edgellm
