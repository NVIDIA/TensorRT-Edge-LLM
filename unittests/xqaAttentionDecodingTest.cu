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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "references.h"
#include "testUtils.h"

using namespace nvinfer1;
using namespace trt_edgellm;

void TestXQAAttentionDecodingAccuracy(
    int32_t batchSize, int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t kvCacheCapacity)
{
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    // Decoding attention length always set qSequenceLength to 1
    constexpr int qSequenceLength = 1;

    std::vector<int32_t> kvCacheLengths(batchSize);
    uniformIntInitialization(kvCacheLengths, kvCacheCapacity / 4, kvCacheCapacity);

    std::vector<half> qInput;
    // Initialize KVCahce buffer to full capacity.
    std::vector<half> kvInput(batchSize * 2 * numKVHeads * kvCacheCapacity * headSize, 0.F);
    std::vector<half> outReference;

    for (int32_t i = 0; i < batchSize; i++)
    {
        int32_t kvLength = kvCacheLengths[i];
        std::vector<half> qi(numQHeads * headSize * qSequenceLength);
        std::vector<half> ki(numKVHeads * headSize * kvLength);
        std::vector<half> vi(numKVHeads * headSize * kvLength);
        uniformFloatInitialization(qi);
        uniformFloatInitialization(ki);
        uniformFloatInitialization(vi);

        auto ref = casualAttentionRef(qi, ki, vi, qSequenceLength, kvLength, numQHeads, numKVHeads, headSize);

        // Add data from batch to input Tensors
        qInput.insert(qInput.end(), qi.begin(), qi.end());

        // Add KV data to KVCache buffer, layout assumed to be [B, 2, Hkv, S, D]
        int32_t const batchOffset = i * 2 * numKVHeads * kvCacheCapacity * headSize;
        int32_t const vOffset = numKVHeads * kvCacheCapacity * headSize;
        for (int32_t hkv = 0; hkv < numKVHeads; hkv++)
        {
            for (int32_t skv = 0; skv < kvLength; skv++)
            {
                for (int32_t d = 0; d < headSize; d++)
                {
                    kvInput[batchOffset + hkv * kvCacheCapacity * headSize + skv * headSize + d]
                        = ki[hkv * kvLength * headSize + skv * headSize + d];
                    kvInput[batchOffset + vOffset + hkv * kvCacheCapacity * headSize + skv * headSize + d]
                        = vi[hkv * kvLength * headSize + skv * headSize + d];
                }
            }
        }
        outReference.insert(outReference.end(), ref.begin(), ref.end());
    }
    // Prepare device memory for kernel execution.
    thrust::device_vector<half> qInputDevice(qInput);
    thrust::device_vector<half> kvInputDevice(kvInput);
    thrust::device_vector<half> outDevice(outReference.size(), 0.0F);
    thrust::device_vector<int32_t> kvCacheLengthDevice(kvCacheLengths);

    EXPECT_TRUE(trt_edgellm::DecoderXQARunner::canImplement(numQHeads, numKVHeads, smVersion, DataType::kHALF));
    trt_edgellm::DecoderXQARunner runner(DataType::kHALF, batchSize, numQHeads, numKVHeads, headSize, smVersion);
    auto params = runner.initXQAParams();
    params.qInputPtr = thrust::raw_pointer_cast(qInputDevice.data());
    params.kvCache.data = thrust::raw_pointer_cast(kvInputDevice.data());
    params.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthDevice.data());
    params.kvCache.capacity = kvCacheCapacity;
    params.output = thrust::raw_pointer_cast(outDevice.data());

    // Use default stream .
    cudaStream_t stream{nullptr};
    runner.dispatchXQAKernel(params, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Check accuracy.
    thrust::host_vector<half> outHost(outDevice.size());
    thrust::copy(outDevice.begin(), outDevice.end(), outHost.begin());

    bool NanValueDetected = false;
    int32_t numErrorWithin1E_3 = 0;
    for (int32_t i = 0; i < batchSize * numQHeads * headSize; ++i)
    {
        EXPECT_TRUE(isclose(outHost[i], outReference[i], 1e-2, 1e-2));
        if (isclose(outHost[i], outReference[i], 1e-3, 1e-3))
        {
            numErrorWithin1E_3++;
        }
        if (__hisnan(outHost[i]))
        {
            NanValueDetected = true;
        }
    }
    float passRate1E_3 = static_cast<float>(numErrorWithin1E_3) / (batchSize * numQHeads * headSize);

    std::cout << "XQA Attention Decoding test. batch_size: " << batchSize << " num_Q_heads: " << numQHeads
              << " num_KV_heads: " << numKVHeads << " head_size: " << headSize << " kvcache lengths: " << kvCacheLengths
              << " pass_rate_1e-3: " << passRate1E_3 << std::endl;
    EXPECT_GT(passRate1E_3, 0.9);
    EXPECT_FALSE(NanValueDetected);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio3)
{
    TestXQAAttentionDecodingAccuracy(1, 24, 8, 128, 1024);
    TestXQAAttentionDecodingAccuracy(2, 24, 8, 128, 512);
    TestXQAAttentionDecodingAccuracy(4, 24, 8, 128, 256);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio4)
{
    TestXQAAttentionDecodingAccuracy(1, 32, 8, 128, 1024);
    TestXQAAttentionDecodingAccuracy(2, 32, 8, 128, 512);
    TestXQAAttentionDecodingAccuracy(4, 32, 8, 128, 256);
    TestXQAAttentionDecodingAccuracy(1, 32, 8, 64, 2048);
    TestXQAAttentionDecodingAccuracy(4, 16, 4, 64, 512);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio5)
{
    TestXQAAttentionDecodingAccuracy(1, 40, 8, 128, 1024);
    TestXQAAttentionDecodingAccuracy(2, 40, 8, 128, 512);
    TestXQAAttentionDecodingAccuracy(4, 40, 8, 128, 512);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio7)
{
    TestXQAAttentionDecodingAccuracy(1, 28, 4, 128, 1024);
    TestXQAAttentionDecodingAccuracy(2, 28, 4, 128, 512);
    TestXQAAttentionDecodingAccuracy(4, 28, 4, 128, 256);
    TestXQAAttentionDecodingAccuracy(1, 28, 4, 64, 1024);
    TestXQAAttentionDecodingAccuracy(4, 14, 2, 64, 512);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio8)
{
    TestXQAAttentionDecodingAccuracy(1, 32, 4, 128, 1024);
    TestXQAAttentionDecodingAccuracy(2, 32, 4, 128, 512);
    TestXQAAttentionDecodingAccuracy(4, 32, 4, 128, 256);
}
