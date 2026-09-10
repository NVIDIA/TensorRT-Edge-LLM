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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "references.h"
#include "testUtils.h"
#include "xqaJitTestUtils.h"

#include <algorithm>
#include <cmath>
#include <optional>

using namespace nvinfer1;
using namespace trt_edgellm;

void TestXQAAttentionDecodingAccuracy(int32_t batchSize, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t kvCacheCapacity, bool useFp8Cache = false, int32_t slidingWindowSize = 0,
    std::optional<float> attentionScale = std::nullopt, int32_t fixedContextLen = 0)
{
    float const resolvedAttentionScale = attentionScale.value_or(1.0F / std::sqrt(static_cast<float>(headSize)));
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (useFp8Cache && smVersion < 89)
    {
        GTEST_SKIP() << "Skipping FP8 KV cache tests: requires SM >= 89, but got SM " << smVersion;
    }
    // Decoding attention length always set qSequenceLength to 1
    constexpr int qSequenceLength = 1;

    std::vector<int32_t> kvCacheLengths(batchSize);
    uniformIntInitialization(kvCacheLengths, kvCacheCapacity / 4, kvCacheCapacity);
    if (fixedContextLen > 0)
    {
        // Deterministic context length to reproduce reported failure shapes.
        ASSERT_LE(fixedContextLen, kvCacheCapacity);
        std::fill(kvCacheLengths.begin(), kvCacheLengths.end(), fixedContextLen);
    }
    if (slidingWindowSize > 0)
    {
        // Cover KV lengths greater than, equal to, and smaller than the sliding window.
        for (int32_t i = 0; i < batchSize; ++i)
        {
            int32_t targetLength = kvCacheCapacity;
            if (i % 3 == 0)
            {
                targetLength = slidingWindowSize + 37;
            }
            else if (i % 3 == 1)
            {
                targetLength = slidingWindowSize;
            }
            else
            {
                targetLength = std::max(1, slidingWindowSize / 2);
            }
            kvCacheLengths[i] = std::min(kvCacheCapacity, targetLength);
        }
    }

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
        uniformFloatInitialization(qi, -1.0F, 1.0F);
        uniformFloatInitialization(ki, -1.0F, 1.0F);
        uniformFloatInitialization(vi, -1.0F, 1.0F);

        int32_t const attentionLength = slidingWindowSize > 0 ? std::min(kvLength, slidingWindowSize) : kvLength;
        auto kiRef = sliceKVWindow(ki, numKVHeads, headSize, kvLength, slidingWindowSize);
        auto viRef = sliceKVWindow(vi, numKVHeads, headSize, kvLength, slidingWindowSize);
        auto ref = casualAttentionRef<half>(qi, kiRef, viRef, qSequenceLength, attentionLength, numQHeads, numKVHeads,
            headSize, resolvedAttentionScale);

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

    EXPECT_TRUE(
        trt_edgellm::canCompileXQAKernel(numQHeads, numKVHeads, headSize, smVersion, DataType::kHALF, DataType::kHALF));
    ASSERT_TRUE(trt_edgellm::loadXQAJitKernelForTest(
        smVersion, DataType::kHALF, DataType::kHALF, headSize, numQHeads, numKVHeads, slidingWindowSize > 0, false));
    trt_edgellm::DecoderXQARunner runner(
        DataType::kHALF, DataType::kHALF, batchSize, numQHeads, numKVHeads, headSize, smVersion);
    auto params = runner.initXQAParams();
    params.qInputPtr = thrust::raw_pointer_cast(qInputDevice.data());
    params.kvCache.data = thrust::raw_pointer_cast(kvInputDevice.data());
    params.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthDevice.data());
    params.kvCache.capacity = kvCacheCapacity;
    params.output = thrust::raw_pointer_cast(outDevice.data());
    params.attentionScale = resolvedAttentionScale;
    params.slidingWinSize = slidingWindowSize > 0 ? static_cast<uint32_t>(slidingWindowSize) : 0U;

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
        if (isnan(__half2float(outHost[i])) || isinf(__half2float(outHost[i])))
        {
            NanValueDetected = true;
        }
    }
    float passRate1E_3 = static_cast<float>(numErrorWithin1E_3) / (batchSize * numQHeads * headSize);

    std::cout << "XQA Attention Decoding test. [FP16 KV cache] batch_size: " << batchSize
              << " num_Q_heads: " << numQHeads << " num_KV_heads: " << numKVHeads << " head_size: " << headSize
              << " sliding_window: " << slidingWindowSize << " kvcache lengths: " << kvCacheLengths
              << " attention_scale: " << resolvedAttentionScale << " pass_rate_1e-3: " << passRate1E_3 << std::endl;
    EXPECT_GT(passRate1E_3, 0.9);
    EXPECT_FALSE(NanValueDetected);

#if SUPPORTS_FP8
    if (useFp8Cache)
    {
        // Compute FP8 amax-based scale for KV cache
        float kAmax = 0.0F;
        float vAmax = 0.0F;
        int32_t const kvStrideHalf = numKVHeads * kvCacheCapacity * headSize; // elements per K or V per batch
        for (int32_t b = 0; b < batchSize; ++b)
        {
            size_t const batchBase = static_cast<size_t>(b) * 2 * kvStrideHalf;
            size_t const vBase = batchBase + kvStrideHalf;
            for (int32_t idx = 0; idx < kvStrideHalf; ++idx)
            {
                kAmax = std::max(kAmax, std::fabs(__half2float(kvInput[batchBase + idx])));
                vAmax = std::max(vAmax, std::fabs(__half2float(kvInput[vBase + idx])));
            }
        }

        // FP8 E4M3 max finite value
        constexpr float FP8_E4M3_MAX = 448.0F;
        assert(kAmax > 0.0F && vAmax > 0.0F);
        float const kScaleQuantOrig = kAmax / FP8_E4M3_MAX;
        float const vScaleQuantOrig = vAmax / FP8_E4M3_MAX;
        float const kScaleOrigQuant = 1.0F / kScaleQuantOrig;
        float const vScaleOrigQuant = 1.0F / vScaleQuantOrig;

        // FP8 decode path: quantize KV cache to FP8 using computed scale and compare against FP16 decoding outputs.
        std::vector<__nv_fp8_e4m3> kvInputFp8(kvInput.size());
        for (int32_t b = 0; b < batchSize; ++b)
        {
            size_t const batchBase = static_cast<size_t>(b) * 2 * kvStrideHalf;
            size_t const vBase = batchBase + kvStrideHalf;
            for (int32_t idx = 0; idx < kvStrideHalf; ++idx)
            {
                kvInputFp8[batchBase + idx] = __nv_fp8_e4m3(__half2float(kvInput[batchBase + idx]) * kScaleOrigQuant);
                kvInputFp8[vBase + idx] = __nv_fp8_e4m3(__half2float(kvInput[vBase + idx]) * vScaleOrigQuant);
            }
        }

        std::vector<half> outReferenceFp8;
        int32_t const qOffset = numQHeads * headSize * qSequenceLength;
        int32_t const batchKvStride = 2 * numKVHeads * kvCacheCapacity * headSize;
        int32_t const vRegionOffset = numKVHeads * kvCacheCapacity * headSize;
        for (int32_t i = 0; i < batchSize; i++)
        {
            int32_t kvLength = kvCacheLengths[i];

            // Reconstruct qi from the flattened qInput buffer.
            std::vector<half> qi(numQHeads * headSize * qSequenceLength);
            std::copy_n(qInput.begin() + i * qOffset, qOffset, qi.begin());

            // Reconstruct compact K/V (shape [Hkv, kvLength, D]) from KV cache layout
            // kvInput layout per batch: [2, Hkv, S=kvCacheCapacity, D]
            std::vector<__nv_fp8_e4m3> ki(numKVHeads * headSize * kvLength);
            std::vector<__nv_fp8_e4m3> vi(numKVHeads * headSize * kvLength);

            int32_t const batchOffset = i * batchKvStride;
            for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
            {
                for (int32_t skv = 0; skv < kvLength; ++skv)
                {
                    for (int32_t d = 0; d < headSize; ++d)
                    {
                        int32_t const compactIdx = hkv * kvLength * headSize + skv * headSize + d;

                        int32_t const kCacheIdx = batchOffset + hkv * kvCacheCapacity * headSize + skv * headSize + d;
                        int32_t const vCacheIdx
                            = batchOffset + vRegionOffset + hkv * kvCacheCapacity * headSize + skv * headSize + d;

                        ki[compactIdx] = kvInputFp8[kCacheIdx];
                        vi[compactIdx] = kvInputFp8[vCacheIdx];
                    }
                }
            }

            int32_t const attentionLength = slidingWindowSize > 0 ? std::min(kvLength, slidingWindowSize) : kvLength;
            auto kiRef = sliceKVWindow(ki, numKVHeads, headSize, kvLength, slidingWindowSize);
            auto viRef = sliceKVWindow(vi, numKVHeads, headSize, kvLength, slidingWindowSize);
            auto ref = casualAttentionRef<__nv_fp8_e4m3>(qi, kiRef, viRef, qSequenceLength, attentionLength, numQHeads,
                numKVHeads, headSize, resolvedAttentionScale, std::nullopt, kScaleQuantOrig, vScaleQuantOrig);
            outReferenceFp8.insert(outReferenceFp8.end(), ref.begin(), ref.end());
        }

        thrust::device_vector<__nv_fp8_e4m3> kvInputFp8Device(kvInputFp8);
        thrust::device_vector<half> outFp8Device(batchSize * numQHeads * headSize, __float2half(0.0F));
        EXPECT_TRUE(trt_edgellm::canCompileXQAKernel(
            numQHeads, numKVHeads, headSize, smVersion, DataType::kHALF, DataType::kFP8));
        ASSERT_TRUE(trt_edgellm::loadXQAJitKernelForTest(
            smVersion, DataType::kHALF, DataType::kFP8, headSize, numQHeads, numKVHeads, slidingWindowSize > 0, false));
        trt_edgellm::DecoderXQARunner runnerFp8(
            DataType::kHALF, DataType::kFP8, batchSize, numQHeads, numKVHeads, headSize, smVersion);
        auto paramsFp8 = runnerFp8.initXQAParams();
        paramsFp8.qInputPtr = thrust::raw_pointer_cast(qInputDevice.data());
        paramsFp8.kvCache.data = thrust::raw_pointer_cast(kvInputFp8Device.data());
        paramsFp8.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthDevice.data());
        paramsFp8.kvCache.capacity = kvCacheCapacity;
        paramsFp8.output = thrust::raw_pointer_cast(outFp8Device.data());
        paramsFp8.attentionScale = resolvedAttentionScale;
        paramsFp8.kScale = kScaleQuantOrig;
        paramsFp8.vScale = vScaleQuantOrig;
        paramsFp8.slidingWinSize = slidingWindowSize > 0 ? static_cast<uint32_t>(slidingWindowSize) : 0U;

        // Reuse the same stream used for FP16 decoding.
        cudaStream_t stream{nullptr};
        runnerFp8.dispatchXQAKernel(paramsFp8, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());

        thrust::host_vector<half> outFp8Host(outFp8Device.size());
        thrust::copy(outFp8Device.begin(), outFp8Device.end(), outFp8Host.begin());

        // Compare FP8 vs FP16 decoding outputs.
        ASSERT_EQ(outReferenceFp8.size(), outFp8Host.size());
        int32_t numClose = 0;
        float maxAbsDiff = 0.0F;
        bool NanValueDetectedFp8 = false;
        for (int32_t i = 0; i < static_cast<int32_t>(outReferenceFp8.size()); ++i)
        {
            float const v16 = __half2float(outReferenceFp8[i]);
            float const v8 = __half2float(outFp8Host[i]);
            float const absDiff = std::fabs(v16 - v8);
            maxAbsDiff = std::max(maxAbsDiff, absDiff);

            if (isclose(outFp8Host[i], outReferenceFp8[i], 1e-3, 1e-3))
            {
                numClose++;
            }
            if (isnan(__half2float(outFp8Host[i])) || isinf(__half2float(outFp8Host[i])))
            {
                NanValueDetectedFp8 = true;
            }
            EXPECT_FALSE(isnan(__half2float(outFp8Host[i])));
        }
        float const matchRate = static_cast<float>(numClose) / static_cast<float>(outHost.size());
        std::cout << "XQA Attention Decoding test. [FP8 KV cache] batch_size: " << batchSize
                  << " num_Q_heads: " << numQHeads << " num_KV_heads: " << numKVHeads << " head_size: " << headSize
                  << " sliding_window: " << slidingWindowSize << " kvcache lengths: " << kvCacheLengths
                  << " attention_scale: " << resolvedAttentionScale << " pass_rate_1e-3: " << passRate1E_3 << std::endl;
        EXPECT_GT(matchRate, 0.9);
        EXPECT_FALSE(NanValueDetectedFp8);
    }
#else
    (void) useFp8Cache;
#endif
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
    TestXQAAttentionDecodingAccuracy(1, 8, 2, 256, 1024);
    TestXQAAttentionDecodingAccuracy(1, 16, 4, 256, 1024);
    TestXQAAttentionDecodingAccuracy(2, 16, 4, 256, 512);
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

TEST(XQAAttentionDecodingTest, accuracyKVRatio8HeadDim256)
{
    TestXQAAttentionDecodingAccuracy(1, 16, 2, 256, 1024);
    TestXQAAttentionDecodingAccuracy(2, 16, 2, 256, 512);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio8HeadDim512)
{
    TestXQAAttentionDecodingAccuracy(1, 16, 2, 512, 256);
    TestXQAAttentionDecodingAccuracy(2, 16, 2, 512, 128);
    TestXQAAttentionDecodingAccuracy(1, 16, 2, 512, 512, false, 0, std::nullopt, 274);
    TestXQAAttentionDecodingAccuracy(1, 8, 1, 512, 1024, false, 0, 1000);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio16HeadDim512)
{
    TestXQAAttentionDecodingAccuracy(2, 16, 1, 512, 128);
    TestXQAAttentionDecodingAccuracy(1, 16, 1, 512, 512, false, 0, std::nullopt, 274);
    TestXQAAttentionDecodingAccuracy(1, 16, 1, 512, 4096, false, 0, std::nullopt, 3200);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio6)
{
    TestXQAAttentionDecodingAccuracy(1, 24, 4, 256, 1024);
    TestXQAAttentionDecodingAccuracy(2, 24, 4, 256, 512);
    TestXQAAttentionDecodingAccuracy(4, 24, 4, 256, 256);
}

TEST(XQAAttentionDecodingTest, slidingWindowAccuracy)
{
    TestXQAAttentionDecodingAccuracy(3, 32, 4, 128, 512, false, 127);
    TestXQAAttentionDecodingAccuracy(2, 32, 8, 64, 384, false, 96);
    TestXQAAttentionDecodingAccuracy(2, 24, 4, 256, 384, false, 129);
    TestXQAAttentionDecodingAccuracy(2, 16, 2, 512, 192, false, 64);
    TestXQAAttentionDecodingAccuracy(2, 32, 4, 128, 96, false, 256);
}

TEST(XQAAttentionDecodingTest, configurableAttentionScale)
{
    TestXQAAttentionDecodingAccuracy(1, 8, 2, 128, 256, false, 0, 1.0F);
    TestXQAAttentionDecodingAccuracy(1, 8, 2, 128, 256, false, 0, 0.37F);
    TestXQAAttentionDecodingAccuracy(1, 8, 1, 256, 1024, false, 512, 1.0F);
    TestXQAAttentionDecodingAccuracy(1, 8, 1, 256, 1024, false, 512, 0.37F);
    TestXQAAttentionDecodingAccuracy(1, 8, 1, 512, 256, false, 0, 1.0F);
    TestXQAAttentionDecodingAccuracy(1, 8, 1, 512, 256, false, 0, 0.37F);
}

#if SUPPORTS_FP8
TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio3)
{
    TestXQAAttentionDecodingAccuracy(1, 24, 8, 128, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 24, 8, 128, 512, true);
    TestXQAAttentionDecodingAccuracy(4, 24, 8, 128, 256, true);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio4)
{
    TestXQAAttentionDecodingAccuracy(1, 32, 8, 128, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 32, 8, 128, 512, true);
    TestXQAAttentionDecodingAccuracy(4, 32, 8, 128, 256, true);
    TestXQAAttentionDecodingAccuracy(1, 32, 8, 64, 2048, true);
    TestXQAAttentionDecodingAccuracy(4, 16, 4, 64, 512, true);
    TestXQAAttentionDecodingAccuracy(1, 8, 2, 256, 1024, true);
    TestXQAAttentionDecodingAccuracy(1, 16, 4, 256, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 16, 4, 256, 512, true);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio5)
{
    TestXQAAttentionDecodingAccuracy(1, 40, 8, 128, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 40, 8, 128, 512, true);
    TestXQAAttentionDecodingAccuracy(4, 40, 8, 128, 512, true);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio7)
{
    TestXQAAttentionDecodingAccuracy(1, 28, 4, 128, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 28, 4, 128, 512, true);
    TestXQAAttentionDecodingAccuracy(4, 28, 4, 128, 256, true);
    TestXQAAttentionDecodingAccuracy(1, 28, 4, 64, 1024, true);
    TestXQAAttentionDecodingAccuracy(4, 14, 2, 64, 512, true);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio8)
{
    TestXQAAttentionDecodingAccuracy(1, 32, 4, 128, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 32, 4, 128, 512, true);
    TestXQAAttentionDecodingAccuracy(4, 32, 4, 128, 256, true);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio8HeadDim256)
{
    TestXQAAttentionDecodingAccuracy(1, 16, 2, 256, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 16, 2, 256, 512, true);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio8HeadDim512)
{
    TestXQAAttentionDecodingAccuracy(1, 16, 2, 512, 256, true);
    TestXQAAttentionDecodingAccuracy(2, 16, 2, 512, 128, true);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio16HeadDim512)
{
    TestXQAAttentionDecodingAccuracy(2, 16, 1, 512, 128, true);
    TestXQAAttentionDecodingAccuracy(1, 16, 1, 512, 512, true, 0, std::nullopt, 274);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio6)
{
    TestXQAAttentionDecodingAccuracy(1, 24, 4, 256, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 24, 4, 256, 512, true);
    TestXQAAttentionDecodingAccuracy(4, 24, 4, 256, 256, true);
}

TEST(XQAAttentionDecodingFP8Test, slidingWindowAccuracy)
{
    TestXQAAttentionDecodingAccuracy(3, 32, 4, 128, 512, true, 127);
    TestXQAAttentionDecodingAccuracy(2, 16, 2, 256, 384, true, 96);
}

TEST(XQAAttentionDecodingFP8Test, configurableAttentionScale)
{
    TestXQAAttentionDecodingAccuracy(1, 8, 2, 128, 256, true, 0, 0.37F);
}
#endif
