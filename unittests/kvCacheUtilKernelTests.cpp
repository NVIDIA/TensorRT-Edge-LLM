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

#include "common/cudaUtils.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "runtime/linearKVCache.h"
#include "testUtils.h"
#include <gtest/gtest.h>

using namespace trt_edgellm;
using namespace nvinfer1;

struct KVCacheParameters
{
    int32_t numDecoderLayers;
    int32_t maxBatchSize;
    int32_t maxSequenceLength;
    int32_t numKVHead;
    int32_t headDim;
};

void TestKVCacheCopyWithTensor(KVCacheParameters const& cacheParams, int32_t copyBatchIdx, int32_t copySequenceLen)
{
    cudaStream_t stream{nullptr};
    rt::LinearKVCache kvCache(rt::LinearKVCache::CacheConfig{cacheParams.numDecoderLayers, cacheParams.maxBatchSize,
                                  cacheParams.maxSequenceLength, cacheParams.numKVHead, cacheParams.headDim},
        stream);
    rt::Tensor cacheTensor
        = rt::Tensor({cacheParams.numDecoderLayers, 2, cacheParams.numKVHead, copySequenceLen, cacheParams.headDim},
            rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kvCacheBuffer = kvCache.getKVCacheBuffer();
    // Instantiate the cache tensor with random data
    std::vector<half> cacheDataHost(cacheTensor.getShape().volume(), 0.0f);
    uniformFloatInitialization(cacheDataHost);

    // Copy the cache tensor to the KVCache
    CUDA_CHECK(cudaMemcpy(
        cacheTensor.rawPointer(), cacheDataHost.data(), cacheTensor.getMemoryCapacity(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(kvCacheBuffer.rawPointer(), 0, kvCacheBuffer.getMemoryCapacity()));

    // Perform the copy from tensor to Cache and pull the data back to host.
    std::vector<half> kvCacheBufferHost(kvCacheBuffer.getShape().volume(), 0.0f);
    kernel::instantiateKVCacheFromTensor(kvCacheBuffer, cacheTensor, copyBatchIdx, stream);
    CUDA_CHECK(cudaMemcpyAsync(kvCacheBufferHost.data(), kvCacheBuffer.rawPointer(), kvCacheBuffer.getMemoryCapacity(),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify the data in the KVCache
    auto compareCacheAndTensorData = [&]() {
        KvCacheIndexer indexer(
            cacheParams.maxBatchSize, cacheParams.numKVHead, cacheParams.maxSequenceLength, cacheParams.headDim);
        for (int32_t idxL = 0; idxL < cacheParams.numDecoderLayers; idxL++)
        {
            int64_t cacheLayerOffset = idxL
                * (cacheParams.maxBatchSize * 2 * cacheParams.numKVHead * cacheParams.maxSequenceLength
                    * cacheParams.headDim);
            int64_t tensorLayerOffset = idxL * (2 * cacheParams.numKVHead * copySequenceLen * cacheParams.headDim);
            for (int32_t idxS = 0; idxS < copySequenceLen; idxS++)
            {
                for (int32_t idxKV = 0; idxKV < cacheParams.numKVHead; idxKV++)
                {
                    for (int32_t idxD = 0; idxD < cacheParams.headDim; idxD++)
                    {
                        // First compare K than V.
                        // layout [numDecoderLayers, 2, numKVHead, maxSequenceLength, headDim]
                        int64_t srcKOffset = tensorLayerOffset + idxKV * copySequenceLen * cacheParams.headDim
                            + idxS * cacheParams.headDim + idxD;
                        int64_t dstKOffset = cacheLayerOffset + indexer.indexK(copyBatchIdx, idxKV, idxS, idxD);
                        if (!isclose(kvCacheBufferHost[dstKOffset], cacheDataHost[srcKOffset], 1e-5, 1e-5))
                        {
                            std::cout << "Mismatch at layer " << idxL << ", sequence " << idxS << ", KV head " << idxKV
                                      << ", dim " << idxD << std::endl;
                            std::cout << "kvCacheBufferHost[dstKOffset]: "
                                      << __half2float(kvCacheBufferHost[dstKOffset])
                                      << ", cacheDataHost[srcKOffset]: " << __half2float(cacheDataHost[srcKOffset])
                                      << std::endl;
                        }
                        ASSERT_TRUE(isclose(kvCacheBufferHost[dstKOffset], cacheDataHost[srcKOffset], 1e-5, 1e-5));

                        int64_t srcVOffset = tensorLayerOffset
                            + (cacheParams.numKVHead + idxKV) * copySequenceLen * cacheParams.headDim
                            + idxS * cacheParams.headDim + idxD;
                        int64_t dstVOffset = cacheLayerOffset + indexer.indexV(copyBatchIdx, idxKV, idxS, idxD);
                        ASSERT_TRUE(isclose(kvCacheBufferHost[dstVOffset], cacheDataHost[srcVOffset], 1e-5, 1e-5));
                    }
                }
            }
        }
    };

    compareCacheAndTensorData();
    std::cout << "Tested copy from tensor to cache with batchIdx " << copyBatchIdx << ", sequence length "
              << copySequenceLen
              << ", KVCacheBuffer shape ([numLayers, maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]): "
              << kvCacheBuffer.getShape().formatString() << std::endl;

    // cudaMemset the cache tensor and test from the other direction.
    CUDA_CHECK(cudaMemsetAsync(cacheTensor.rawPointer(), 0, cacheTensor.getMemoryCapacity(), stream));
    kernel::saveKVCacheIntoTensor(cacheTensor, kvCacheBuffer, copyBatchIdx, stream);
    CUDA_CHECK(cudaMemcpyAsync(cacheDataHost.data(), cacheTensor.rawPointer(), cacheTensor.getMemoryCapacity(),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    compareCacheAndTensorData();

    std::cout << "Tested copy from cache to tensor with batchIdx " << copyBatchIdx << ", sequence length "
              << copySequenceLen
              << ", saved kvCacheTensor shape ([numLayers, 2, numKVHeads, sequenceLength, headDim]): "
              << cacheTensor.getShape().formatString() << std::endl;
}

TEST(KVCacheUtilKernelTests, TestKVCacheCopyWithTensor)
{
    // KVCache: 3 decoder layers, 8 max batch size, 1024 max sequence length, 4 KV heads, 128 head dim.
    // Copy to batchIdx 0 with sequence length 128.
    TestKVCacheCopyWithTensor({3, 8, 1024, 4, 128}, 0, 128);
    // KVCache: 3 decoder layers, 8 max batch size, 1024 max sequence length, 4 KV heads, 128 head dim.
    // Copy to batchIdx 1 with sequence length 97, which is not divisible by 2
    TestKVCacheCopyWithTensor({3, 8, 1024, 4, 128}, 1, 97);
    // KVCache: 3 decoder layers, 4 max batch size, 512 max sequence length, 7 KV heads, 64 head dim.
    // Copy to batchIdx 0 with sequence length 96.
    TestKVCacheCopyWithTensor({3, 4, 512, 7, 64}, 0, 96);
    // KVCache: 3 decoder layers, 4 max batch size, 512 max sequence length, 7 KV heads, 64 head dim.
    // Copy to batchIdx 1 with sequence length 47, which is not divisible by 4.
    TestKVCacheCopyWithTensor({3, 4, 512, 7, 64}, 1, 47);
    // KVCache: 28 decoder layers, 4 max batch size, 2048 max sequence length, 4 KV heads, 128 head dim.
    // Copy to batchIdx 0 with sequence length 1010, simulate the Qwen2-VL config..
    TestKVCacheCopyWithTensor({28, 1, 1024, 4, 128}, 0, 255);
}