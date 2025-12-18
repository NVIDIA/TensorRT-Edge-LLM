/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "common/cudaUtils.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include <cuda_bf16.h>
#include <type_traits>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace rt
{

LinearKVCache::LinearKVCache(CacheConfig const& config, cudaStream_t stream)
    : mConfig(config)
{
    int64_t const kvCacheVolume = mConfig.numDecoderLayers * mConfig.maxBatchSize * 2 * mConfig.numKVHeads
        * mConfig.maxSequenceLength * mConfig.headDim;
    CUDA_CHECK(cudaMalloc(&mDeviceKVCache, kvCacheVolume * sizeof(KVCacheType)));
    LOG_DEBUG("KVCache of shape [%ld, %ld, %ld, %ld, %ld, %ld] allocated on GPU with size: %ld bytes (%.2f MB)",
        mConfig.numDecoderLayers, mConfig.maxBatchSize, 2, mConfig.numKVHeads, mConfig.maxSequenceLength,
        mConfig.headDim, kvCacheVolume * sizeof(KVCacheType),
        static_cast<float>(kvCacheVolume * sizeof(KVCacheType)) / (1024.0 * 1024.0));
    mDeviceKVCacheLengths = rt::Tensor(
        {mConfig.maxBatchSize}, DeviceType::kGPU, DataType::kINT32, "LinearKVCache::mDeviceKVCacheLengths");
    CUDA_CHECK(
        cudaMemsetAsync(mDeviceKVCacheLengths.rawPointer(), 0, mDeviceKVCacheLengths.getMemoryCapacity(), stream));
}

LinearKVCache::~LinearKVCache()
{
    CUDA_CHECK(cudaFree(mDeviceKVCache));
    mDeviceKVCache = nullptr;
}

LinearKVCache::LinearKVCache(LinearKVCache&& other) noexcept
{
    mConfig = other.mConfig;
    mActiveBatchSize = other.mActiveBatchSize;
    mKVCacheAllEmpty = other.mKVCacheAllEmpty;
    mDeviceKVCache = other.mDeviceKVCache;
    mDeviceKVCacheLengths = std::move(other.mDeviceKVCacheLengths);

    other.mConfig = CacheConfig{};
    other.mActiveBatchSize = 0;
    other.mKVCacheAllEmpty = true;
    other.mDeviceKVCache = nullptr;
}

LinearKVCache& LinearKVCache::operator=(LinearKVCache&& other) noexcept
{
    if (this != &other)
    {
        // Release current KVCache memory.
        CUDA_CHECK(cudaFree(mDeviceKVCache));
        mConfig = other.mConfig;
        mKVCacheAllEmpty = other.mKVCacheAllEmpty;
        mActiveBatchSize = other.mActiveBatchSize;
        mDeviceKVCache = other.mDeviceKVCache;
        mDeviceKVCacheLengths = std::move(other.mDeviceKVCacheLengths);

        other.mConfig = CacheConfig{};
        other.mActiveBatchSize = 0;
        other.mKVCacheAllEmpty = true;
        other.mDeviceKVCache = nullptr;
    }
    return *this;
}

rt::Tensor LinearKVCache::getKVCacheForDecoderLayer(int32_t decoderLayerIdx)
{
    int64_t const kvCacheOffset
        = decoderLayerIdx * mConfig.maxBatchSize * 2 * mConfig.numKVHeads * mConfig.maxSequenceLength * mConfig.headDim;
    KVCacheType* kvCachePtr = mDeviceKVCache + kvCacheOffset;
    return rt::Tensor(kvCachePtr,
        {mConfig.maxBatchSize, 2, mConfig.numKVHeads, mConfig.maxSequenceLength, mConfig.headDim}, DeviceType::kGPU,
        KVCacheTypeTRT);
}

rt::Tensor LinearKVCache::getKVCacheBuffer()
{
    return rt::Tensor(mDeviceKVCache,
        {mConfig.numDecoderLayers, mConfig.maxBatchSize, 2, mConfig.numKVHeads, mConfig.maxSequenceLength,
            mConfig.headDim},
        DeviceType::kGPU, KVCacheTypeTRT);
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
    mDeviceKVCacheLengths.reshape({mActiveBatchSize});

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

rt::Tensor& LinearKVCache::getKVCacheLengths()
{
    return mDeviceKVCacheLengths;
}

LinearKVCache::CacheConfig LinearKVCache::getConfig() const
{
    return mConfig;
}

int32_t LinearKVCache::getActiveBatchSize() const
{
    return mActiveBatchSize;
}

bool LinearKVCache::getKVCacheAllEmpty() const
{
    return mKVCacheAllEmpty;
}

void LinearKVCache::setActiveBatchSize(int32_t newActiveBatchSize)
{
    check::check(newActiveBatchSize >= 0 && newActiveBatchSize <= mConfig.maxBatchSize,
        "Invalid active batch size: must be in range [0, maxBatchSize]");
    mActiveBatchSize = newActiveBatchSize;
    mDeviceKVCacheLengths.reshape({mActiveBatchSize});
}

} // namespace rt
} // namespace trt_edgellm