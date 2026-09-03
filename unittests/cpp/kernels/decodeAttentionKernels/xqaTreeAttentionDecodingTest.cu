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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "kernels/decodeAttentionKernels/decoderXQAJitCompiler.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "references.h"
#include "testUtils.h"
#include "xqaJitTestUtils.h"

using namespace nvinfer1;
using namespace trt_edgellm;

// Tests pack paged KV into a scalar host buffer, so the final head-dim/D coordinate is explicit here.
// Device code indexes the same pool in GMemCacheHead units and traverses D inside each head.
size_t getPagedKVPoolOffset(int32_t pageIdx, int32_t numKVHeads, int32_t tokensPerPage, int32_t headSize, int32_t hkv,
    int32_t tokenInPage, int32_t d)
{
    return (((static_cast<size_t>(pageIdx) * tokensPerPage + tokenInPage) * numKVHeads + hkv) * headSize + d);
}

void initializeCudaContextForXQATest()
{
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
}

void TestXQATreeAttentionDecodingAccuracy(int32_t batchSize, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t kvSequenceLength, int32_t qSequenceLength, bool useFp8Cache = false, int32_t slidingWindowSize = 0,
    int32_t tokensPerPage = 0, std::optional<float> attentionScale = std::nullopt)
{
    float const resolvedAttentionScale = attentionScale.value_or(1.0F / std::sqrt(static_cast<float>(headSize)));
    initializeCudaContextForXQATest();

    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (useFp8Cache && smVersion < 89)
    {
        GTEST_SKIP() << "Skipping FP8 KV cache tests: requires SM >= 89, but got SM " << smVersion;
    }
    bool const usePagedKVCache = tokensPerPage > 0;
    ASSERT_TRUE(
        trt_edgellm::canCompileXQAKernel(numQHeads, numKVHeads, headSize, smVersion, DataType::kHALF, DataType::kHALF))
        << "No compatible FP16 XQA kernel for SM " << smVersion << " paged_kv=" << usePagedKVCache;
    ASSERT_TRUE(trt_edgellm::loadXQAJitKernelForTest(smVersion, DataType::kHALF, DataType::kHALF, headSize, numQHeads,
        numKVHeads, slidingWindowSize > 0, /*specDecode=*/true, tokensPerPage));
    if (usePagedKVCache)
    {
        ASSERT_EQ(kvSequenceLength % tokensPerPage, 0);
#if SUPPORTS_FP8
        if (useFp8Cache)
        {
            ASSERT_TRUE(trt_edgellm::loadXQAJitKernelForTest(smVersion, DataType::kHALF, DataType::kFP8, headSize,
                numQHeads, numKVHeads, slidingWindowSize > 0, /*specDecode=*/true, tokensPerPage));
        }
#endif
    }

    // Tree attention uses one fixed KV length per shape; boundary coverage is enumerated by sliding-window test cases.
    std::vector<int32_t> kvCacheLength(batchSize, kvSequenceLength);
    std::vector<half> qInput;
    std::vector<half> kvInput;
    std::vector<half> outReference;
    std::vector<int32_t> packedTreeMaskInput;
    // Keep original (unpacked) tree mask per batch for CPU FP8 reference
    std::vector<std::vector<int32_t>> treeMasks;
    treeMasks.reserve(batchSize);

    int32_t const maxNbPagesPerSeq = usePagedKVCache ? kvSequenceLength / tokensPerPage : 0;
    std::vector<int32_t> kvCachePageList;
    if (usePagedKVCache)
    {
        int32_t const nbPoolPages = batchSize * 2 * maxNbPagesPerSeq;
        kvInput.resize(static_cast<size_t>(nbPoolPages) * numKVHeads * tokensPerPage * headSize, __float2half(0.0F));
        kvCachePageList.resize(static_cast<size_t>(batchSize) * 2 * maxNbPagesPerSeq);
        for (int32_t b = 0; b < batchSize; ++b)
        {
            for (int32_t p = 0; p < maxNbPagesPerSeq; ++p)
            {
                kvCachePageList[(b * 2 * maxNbPagesPerSeq) + p] = (b * 2 * maxNbPagesPerSeq) + p;
                kvCachePageList[(b * 2 * maxNbPagesPerSeq) + maxNbPagesPerSeq + p]
                    = (b * 2 * maxNbPagesPerSeq) + maxNbPagesPerSeq + p;
            }
        }
    }

    for (int32_t i = 0; i < batchSize; i++)
    {
        std::vector<half> qi(numQHeads * headSize * qSequenceLength);
        std::vector<half> ki(numKVHeads * headSize * kvSequenceLength);
        std::vector<half> vi(numKVHeads * headSize * kvSequenceLength);
        std::vector<int32_t> treeMaski(qSequenceLength * qSequenceLength);

        uniformFloatInitialization(qi, -1.0F, 1.0F);
        uniformFloatInitialization(ki, -1.0F, 1.0F);
        uniformFloatInitialization(vi, -1.0F, 1.0F);
        uniformIntInitialization(treeMaski, 0, 1);
        int32_t const attentionLength
            = slidingWindowSize > 0 ? std::min(kvSequenceLength, slidingWindowSize) : kvSequenceLength;
        auto kiRef = sliceKVWindow(ki, numKVHeads, headSize, kvSequenceLength, slidingWindowSize);
        auto viRef = sliceKVWindow(vi, numKVHeads, headSize, kvSequenceLength, slidingWindowSize);
        auto ref = casualAttentionRef<half>(qi, kiRef, viRef, qSequenceLength, attentionLength, numQHeads, numKVHeads,
            headSize, resolvedAttentionScale, std::make_optional(treeMaski));

        // Add data from batch to input Tensors
        qInput.insert(qInput.end(), qi.begin(), qi.end());

        if (usePagedKVCache)
        {
            for (int32_t hkv = 0; hkv < numKVHeads; hkv++)
            {
                for (int32_t skv = 0; skv < kvSequenceLength; skv++)
                {
                    int32_t const page = skv / tokensPerPage;
                    int32_t const tokenInPage = skv % tokensPerPage;
                    int32_t const kPageIdx = kvCachePageList[(i * 2 * maxNbPagesPerSeq) + page];
                    int32_t const vPageIdx = kvCachePageList[(i * 2 * maxNbPagesPerSeq) + maxNbPagesPerSeq + page];
                    for (int32_t d = 0; d < headSize; d++)
                    {
                        size_t const compactOffset = (hkv * kvSequenceLength * headSize) + (skv * headSize) + d;
                        kvInput[getPagedKVPoolOffset(
                            kPageIdx, numKVHeads, tokensPerPage, headSize, hkv, tokenInPage, d)] = ki[compactOffset];
                        kvInput[getPagedKVPoolOffset(
                            vPageIdx, numKVHeads, tokensPerPage, headSize, hkv, tokenInPage, d)] = vi[compactOffset];
                    }
                }
            }
        }
        else
        {
            // KVcache layout assumed to have layout of [B, 2n H, S, D]
            kvInput.insert(kvInput.end(), ki.begin(), ki.end());
            kvInput.insert(kvInput.end(), vi.begin(), vi.end());
        }
        outReference.insert(outReference.end(), ref.begin(), ref.end());

        // Prepare packed tree mask. The layout of mask is [qSeqLen, qSeqLen]. Which represent whether two tokens
        // will attend to each other.
        int32_t const numBitsPerPackedMask = 32;
        int32_t const numPackedMasksPerToken = divUp(qSequenceLength, numBitsPerPackedMask);
        std::vector<int32_t> packedMaski(numPackedMasksPerToken * qSequenceLength, 0);
        for (int32_t i = 0; i < qSequenceLength; i++)
        {
            for (int32_t j = 0; j < numPackedMasksPerToken; j++)
            {
                int32_t mask = 0;
                for (int32_t k = 0; k < numBitsPerPackedMask; k++)
                {
                    int32_t const bitIndex = j * numBitsPerPackedMask + k;
                    int32_t maskFlag = 0;
                    if (j * numBitsPerPackedMask + k < qSequenceLength)
                    {
                        maskFlag = treeMaski[i * qSequenceLength + bitIndex];
                    }
                    mask |= maskFlag << k;
                }
                packedMaski[i * numPackedMasksPerToken + j] = mask;
            }
        }
        packedTreeMaskInput.insert(packedTreeMaskInput.end(), packedMaski.begin(), packedMaski.end());
        treeMasks.push_back(std::move(treeMaski));
    }
    // Prepare device memory for kernel execution.
    thrust::device_vector<half> qInputDevice(qInput);
    thrust::device_vector<half> kvInputDevice(kvInput);
    thrust::device_vector<half> outDevice(outReference.size(), 1.0F);
    thrust::device_vector<int32_t> kvCacheLengthDevice(kvCacheLength);
    thrust::device_vector<int32_t> kvCachePageListDevice(kvCachePageList);
    thrust::device_vector<int32_t> packedTreeMaskDevice(packedTreeMaskInput);

    trt_edgellm::DecoderXQARunner runner(
        DataType::kHALF, DataType::kHALF, batchSize, numQHeads, numKVHeads, headSize, smVersion);
    auto params = runner.initXQAParams();
    params.qSeqLen = qSequenceLength;
    params.qInputPtr = thrust::raw_pointer_cast(qInputDevice.data());
    params.kvCache.data = thrust::raw_pointer_cast(kvInputDevice.data());
    params.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthDevice.data());
    params.kvCache.capacity = kvSequenceLength;
    if (usePagedKVCache)
    {
        params.kvCache.pageList = thrust::raw_pointer_cast(kvCachePageListDevice.data());
        params.kvCache.tokensPerPage = static_cast<uint32_t>(tokensPerPage);
    }
    params.output = thrust::raw_pointer_cast(outDevice.data());
    params.treeAttnMask = thrust::raw_pointer_cast(packedTreeMaskDevice.data());
    params.attentionScale = resolvedAttentionScale;
    params.slidingWinSize = slidingWindowSize > 0 ? static_cast<uint32_t>(slidingWindowSize) : 0U;
    // Use default stream .
    cudaStream_t stream{nullptr};
    runner.dispatchSpecDecodeXQAKernel(params, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Check accuracy.
    thrust::host_vector<half> outHost(outDevice.size());
    thrust::copy(outDevice.begin(), outDevice.end(), outHost.begin());

    bool NanValueDetected = false;
    int32_t numErrorWithin1E_3 = 0;
    for (int32_t i = 0; i < batchSize * qSequenceLength * numQHeads * headSize; ++i)
    {
        EXPECT_TRUE(isclose(outHost[i], outReference[i], 1e-2, 1e-2));
        if (isclose(outHost[i], outReference[i], 1e-3, 1e-3))
        {
            numErrorWithin1E_3++;
        }
        if (isnan(__half2float(outHost[i])))
        {
            NanValueDetected = true;
        }
    }
    float passRate1E_3 = static_cast<float>(numErrorWithin1E_3) / (batchSize * qSequenceLength * numQHeads * headSize);

    std::cout << "XQA Tree Attention Decoding test. [FP16 KV cache] batch_size: " << batchSize
              << " num_Q_heads: " << numQHeads << " num_KV_heads: " << numKVHeads << " head_size: " << headSize
              << " kvcache seq_len: " << kvSequenceLength << " q_seq_len: " << qSequenceLength
              << " sliding_window: " << slidingWindowSize << " paged_kv: " << usePagedKVCache
              << " tokens_per_page: " << tokensPerPage << " attention_scale: " << resolvedAttentionScale
              << " pass_rate_1e-3: " << passRate1E_3 << std::endl;
    EXPECT_GT(passRate1E_3, 0.9);
    EXPECT_FALSE(NanValueDetected);

#if SUPPORTS_FP8
    if (useFp8Cache)
    {
        int32_t const qOffset = numQHeads * headSize * qSequenceLength;
        int32_t const kvStrideHalf = numKVHeads * headSize * kvSequenceLength; // compact elements per K or V per batch
        float kAmax = 0.0F;
        float vAmax = 0.0F;
        if (usePagedKVCache)
        {
            for (int32_t b = 0; b < batchSize; ++b)
            {
                for (int32_t p = 0; p < maxNbPagesPerSeq; ++p)
                {
                    int32_t const kPageIdx = kvCachePageList[(b * 2 * maxNbPagesPerSeq) + p];
                    int32_t const vPageIdx = kvCachePageList[(b * 2 * maxNbPagesPerSeq) + maxNbPagesPerSeq + p];
                    for (int32_t hkv = 0; hkv < numKVHeads; hkv++)
                    {
                        for (int32_t tokenInPage = 0; tokenInPage < tokensPerPage; tokenInPage++)
                        {
                            for (int32_t d = 0; d < headSize; d++)
                            {
                                kAmax = std::max(kAmax,
                                    std::fabs(__half2float(kvInput[getPagedKVPoolOffset(
                                        kPageIdx, numKVHeads, tokensPerPage, headSize, hkv, tokenInPage, d)])));
                                vAmax = std::max(vAmax,
                                    std::fabs(__half2float(kvInput[getPagedKVPoolOffset(
                                        vPageIdx, numKVHeads, tokensPerPage, headSize, hkv, tokenInPage, d)])));
                            }
                        }
                    }
                }
            }
        }
        else
        {
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
        if (usePagedKVCache)
        {
            for (int32_t b = 0; b < batchSize; ++b)
            {
                for (int32_t p = 0; p < maxNbPagesPerSeq; ++p)
                {
                    int32_t const kPageIdx = kvCachePageList[(b * 2 * maxNbPagesPerSeq) + p];
                    int32_t const vPageIdx = kvCachePageList[(b * 2 * maxNbPagesPerSeq) + maxNbPagesPerSeq + p];
                    for (int32_t hkv = 0; hkv < numKVHeads; hkv++)
                    {
                        for (int32_t tokenInPage = 0; tokenInPage < tokensPerPage; tokenInPage++)
                        {
                            for (int32_t d = 0; d < headSize; d++)
                            {
                                size_t const kOffset = getPagedKVPoolOffset(
                                    kPageIdx, numKVHeads, tokensPerPage, headSize, hkv, tokenInPage, d);
                                size_t const vOffset = getPagedKVPoolOffset(
                                    vPageIdx, numKVHeads, tokensPerPage, headSize, hkv, tokenInPage, d);
                                kvInputFp8[kOffset] = __nv_fp8_e4m3(__half2float(kvInput[kOffset]) * kScaleOrigQuant);
                                kvInputFp8[vOffset] = __nv_fp8_e4m3(__half2float(kvInput[vOffset]) * vScaleOrigQuant);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            for (int32_t b = 0; b < batchSize; ++b)
            {
                size_t const batchBase = static_cast<size_t>(b) * 2 * kvStrideHalf;
                size_t const vBase = batchBase + kvStrideHalf;
                for (int32_t idx = 0; idx < kvStrideHalf; ++idx)
                {
                    kvInputFp8[batchBase + idx]
                        = __nv_fp8_e4m3(__half2float(kvInput[batchBase + idx]) * kScaleOrigQuant);
                    kvInputFp8[vBase + idx] = __nv_fp8_e4m3(__half2float(kvInput[vBase + idx]) * vScaleOrigQuant);
                }
            }
        }

        std::vector<half> outReferenceFp8;
        for (int32_t b = 0; b < batchSize; ++b)
        {
            // Reconstruct qi from the flattened qInput buffer.
            std::vector<half> qi(qOffset);
            std::copy_n(qInput.begin() + b * qOffset, qOffset, qi.begin());

            // Reconstruct compact K/V (shape [Hkv, kvSequenceLength, D]) from KV cache layout
            std::vector<__nv_fp8_e4m3> ki(kvStrideHalf);
            std::vector<__nv_fp8_e4m3> vi(kvStrideHalf);

            if (usePagedKVCache)
            {
                for (int32_t hkv = 0; hkv < numKVHeads; hkv++)
                {
                    for (int32_t skv = 0; skv < kvSequenceLength; skv++)
                    {
                        int32_t const page = skv / tokensPerPage;
                        int32_t const tokenInPage = skv % tokensPerPage;
                        int32_t const kPageIdx = kvCachePageList[(b * 2 * maxNbPagesPerSeq) + page];
                        int32_t const vPageIdx = kvCachePageList[(b * 2 * maxNbPagesPerSeq) + maxNbPagesPerSeq + page];
                        for (int32_t d = 0; d < headSize; d++)
                        {
                            size_t const compactOffset = (hkv * kvSequenceLength * headSize) + (skv * headSize) + d;
                            ki[compactOffset] = kvInputFp8[getPagedKVPoolOffset(
                                kPageIdx, numKVHeads, tokensPerPage, headSize, hkv, tokenInPage, d)];
                            vi[compactOffset] = kvInputFp8[getPagedKVPoolOffset(
                                vPageIdx, numKVHeads, tokensPerPage, headSize, hkv, tokenInPage, d)];
                        }
                    }
                }
            }
            else
            {
                size_t const batchBase = static_cast<size_t>(b) * 2 * kvStrideHalf;
                size_t const vBase = batchBase + kvStrideHalf;
                for (int32_t idx = 0; idx < kvStrideHalf; ++idx)
                {
                    ki[idx] = kvInputFp8[batchBase + idx];
                    vi[idx] = kvInputFp8[vBase + idx];
                }
            }

            int32_t const attentionLength
                = slidingWindowSize > 0 ? std::min(kvSequenceLength, slidingWindowSize) : kvSequenceLength;
            auto kiRef = sliceKVWindow(ki, numKVHeads, headSize, kvSequenceLength, slidingWindowSize);
            auto viRef = sliceKVWindow(vi, numKVHeads, headSize, kvSequenceLength, slidingWindowSize);
            auto refFp8 = casualAttentionRef<__nv_fp8_e4m3>(qi, kiRef, viRef, qSequenceLength, attentionLength,
                numQHeads, numKVHeads, headSize, resolvedAttentionScale, std::make_optional(treeMasks[b]),
                kScaleQuantOrig, vScaleQuantOrig);
            outReferenceFp8.insert(outReferenceFp8.end(), refFp8.begin(), refFp8.end());
        }

        thrust::device_vector<__nv_fp8_e4m3> kvInputFp8Device(kvInputFp8);
        thrust::device_vector<half> outFp8Device(
            batchSize * qSequenceLength * numQHeads * headSize, __float2half(0.0F));

        EXPECT_TRUE(trt_edgellm::canCompileXQAKernel(
            numQHeads, numKVHeads, headSize, smVersion, DataType::kHALF, DataType::kFP8));
        ASSERT_TRUE(trt_edgellm::loadXQAJitKernelForTest(smVersion, DataType::kHALF, DataType::kFP8, headSize,
            numQHeads, numKVHeads, slidingWindowSize > 0, /*specDecode=*/true, tokensPerPage));
        trt_edgellm::DecoderXQARunner runnerFp8(
            DataType::kHALF, DataType::kFP8, batchSize, numQHeads, numKVHeads, headSize, smVersion);
        auto paramsFp8 = runnerFp8.initXQAParams();
        paramsFp8.qSeqLen = qSequenceLength;
        paramsFp8.qInputPtr = thrust::raw_pointer_cast(qInputDevice.data());
        paramsFp8.kvCache.data = thrust::raw_pointer_cast(kvInputFp8Device.data());
        paramsFp8.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthDevice.data());
        paramsFp8.kvCache.capacity = kvSequenceLength;
        if (usePagedKVCache)
        {
            paramsFp8.kvCache.pageList = thrust::raw_pointer_cast(kvCachePageListDevice.data());
            paramsFp8.kvCache.tokensPerPage = static_cast<uint32_t>(tokensPerPage);
        }
        paramsFp8.output = thrust::raw_pointer_cast(outFp8Device.data());
        paramsFp8.treeAttnMask = thrust::raw_pointer_cast(packedTreeMaskDevice.data());
        paramsFp8.attentionScale = resolvedAttentionScale;
        paramsFp8.kScale = kScaleQuantOrig;
        paramsFp8.vScale = vScaleQuantOrig;
        paramsFp8.slidingWinSize = slidingWindowSize > 0 ? static_cast<uint32_t>(slidingWindowSize) : 0U;

        // Reuse the same stream used for FP16 decoding.
        cudaStream_t stream{nullptr};
        runnerFp8.dispatchSpecDecodeXQAKernel(paramsFp8, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());

        thrust::host_vector<half> outFp8Host(outFp8Device.size());
        thrust::copy(outFp8Device.begin(), outFp8Device.end(), outFp8Host.begin());

        // Compare FP8 vs FP16 decoding outputs (through CPU FP8 reference).
        ASSERT_EQ(outReferenceFp8.size(), outFp8Host.size());
        int32_t numClose = 0;
        bool NanValueDetectedFp8 = false;
        for (int32_t i = 0; i < static_cast<int32_t>(outReferenceFp8.size()); ++i)
        {
            EXPECT_TRUE(isclose(outFp8Host[i], outReferenceFp8[i], 1e-2, 1e-2));
            if (isclose(outFp8Host[i], outReferenceFp8[i], 1e-3, 1e-3))
            {
                numClose++;
            }
            if (isnan(__half2float(outFp8Host[i])))
            {
                NanValueDetectedFp8 = true;
            }
            EXPECT_FALSE(isnan(__half2float(outFp8Host[i])));
        }
        float const fp8PassRate1E_3 = static_cast<float>(numClose) / static_cast<float>(outReferenceFp8.size());

        std::cout << "XQA Tree Attention Decoding test. [FP8 KV cache] batch_size: " << batchSize
                  << " num_Q_heads: " << numQHeads << " num_KV_heads: " << numKVHeads << " head_size: " << headSize
                  << " kvcache seq_len: " << kvSequenceLength << " q_seq_len: " << qSequenceLength
                  << " sliding_window: " << slidingWindowSize << " paged_kv: " << usePagedKVCache
                  << " tokens_per_page: " << tokensPerPage << " attention_scale: " << resolvedAttentionScale
                  << " pass_rate_1e-3: " << fp8PassRate1E_3 << std::endl;
        EXPECT_GT(fp8PassRate1E_3, 0.8F);
        EXPECT_FALSE(NanValueDetectedFp8);
    }
#else
    (void) useFp8Cache;
#endif
}

// Test with capacity > kvSequenceLength (simulates runtime with large allocated KV cache, partially filled).
// This is the runtime scenario for Gemma4: capacity=256, only 16 tokens filled.
void TestXQATreeAttentionDecodingWithPaddedCapacity(int32_t batchSize, int32_t numQHeads, int32_t numKVHeads,
    int32_t headSize, int32_t kvSequenceLength, int32_t qSequenceLength, int32_t capacity)
{
    ASSERT_GE(capacity, kvSequenceLength);
    float const attentionScale = 1.0F / std::sqrt(static_cast<float>(headSize));
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    ASSERT_TRUE(
        trt_edgellm::canCompileXQAKernel(numQHeads, numKVHeads, headSize, smVersion, DataType::kHALF, DataType::kHALF));
    ASSERT_TRUE(trt_edgellm::loadXQAJitKernelForTest(smVersion, DataType::kHALF, DataType::kHALF, headSize, numQHeads,
        numKVHeads, /*slidingWindow=*/false, /*specDecode=*/true));

    std::vector<int32_t> kvCacheLength(batchSize, kvSequenceLength);
    std::vector<half> qInput;
    std::vector<half> kvInput;
    std::vector<half> outReference;
    std::vector<int32_t> packedTreeMaskInput;

    for (int32_t i = 0; i < batchSize; i++)
    {
        std::vector<half> qi(numQHeads * headSize * qSequenceLength);
        // KV allocated at full capacity, but only first kvSequenceLength positions have data
        std::vector<half> ki(numKVHeads * headSize * capacity, __float2half(0.0F));
        std::vector<half> vi(numKVHeads * headSize * capacity, __float2half(0.0F));
        std::vector<int32_t> treeMaski(qSequenceLength * qSequenceLength);

        uniformFloatInitialization(qi);
        // Fill only the first kvSequenceLength positions per KV head
        std::vector<half> kiData(numKVHeads * headSize * kvSequenceLength);
        std::vector<half> viData(numKVHeads * headSize * kvSequenceLength);
        uniformFloatInitialization(kiData);
        uniformFloatInitialization(viData);
        // Copy data into padded buffer: KV layout is [numKVHeads, capacity, headSize]
        for (int32_t h = 0; h < numKVHeads; ++h)
        {
            for (int32_t s = 0; s < kvSequenceLength; ++s)
            {
                for (int32_t d = 0; d < headSize; ++d)
                {
                    ki[h * capacity * headSize + s * headSize + d]
                        = kiData[h * kvSequenceLength * headSize + s * headSize + d];
                    vi[h * capacity * headSize + s * headSize + d]
                        = viData[h * kvSequenceLength * headSize + s * headSize + d];
                }
            }
        }

        uniformIntInitialization(treeMaski, 0, 1);
        auto ref = casualAttentionRef<half>(qi, kiData, viData, qSequenceLength, kvSequenceLength, numQHeads,
            numKVHeads, headSize, attentionScale, treeMaski);

        qInput.insert(qInput.end(), qi.begin(), qi.end());
        // KV layout: [B, 2, H_kv, capacity, D] — K then V per batch
        kvInput.insert(kvInput.end(), ki.begin(), ki.end());
        kvInput.insert(kvInput.end(), vi.begin(), vi.end());
        outReference.insert(outReference.end(), ref.begin(), ref.end());

        int32_t const numBitsPerPackedMask = 32;
        int32_t const numPackedMasksPerToken = divUp(qSequenceLength, numBitsPerPackedMask);
        std::vector<int32_t> packedMaski(numPackedMasksPerToken * qSequenceLength, 0);
        for (int32_t ti = 0; ti < qSequenceLength; ti++)
        {
            for (int32_t j = 0; j < numPackedMasksPerToken; j++)
            {
                int32_t mask = 0;
                for (int32_t k = 0; k < numBitsPerPackedMask; k++)
                {
                    int32_t const bitIndex = j * numBitsPerPackedMask + k;
                    int32_t maskFlag = 0;
                    if (bitIndex < qSequenceLength)
                    {
                        maskFlag = treeMaski[ti * qSequenceLength + bitIndex];
                    }
                    mask |= maskFlag << k;
                }
                packedMaski[ti * numPackedMasksPerToken + j] = mask;
            }
        }
        packedTreeMaskInput.insert(packedTreeMaskInput.end(), packedMaski.begin(), packedMaski.end());
    }

    thrust::device_vector<half> qInputDevice(qInput);
    thrust::device_vector<half> kvInputDevice(kvInput);
    thrust::device_vector<half> outDevice(outReference.size(), 1.0F);
    thrust::device_vector<int32_t> kvCacheLengthDevice(kvCacheLength);
    thrust::device_vector<int32_t> packedTreeMaskDevice(packedTreeMaskInput);

    trt_edgellm::DecoderXQARunner runner(
        DataType::kHALF, DataType::kHALF, batchSize, numQHeads, numKVHeads, headSize, smVersion);
    auto params = runner.initXQAParams();
    params.qSeqLen = qSequenceLength;
    params.qInputPtr = thrust::raw_pointer_cast(qInputDevice.data());
    params.kvCache.data = thrust::raw_pointer_cast(kvInputDevice.data());
    params.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthDevice.data());
    params.kvCache.capacity = capacity; // capacity > kvSequenceLength
    params.output = thrust::raw_pointer_cast(outDevice.data());
    params.treeAttnMask = thrust::raw_pointer_cast(packedTreeMaskDevice.data());
    params.attentionScale = attentionScale;

    cudaStream_t stream{nullptr};
    runner.dispatchSpecDecodeXQAKernel(params, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    thrust::host_vector<half> outHost(outDevice.size());
    thrust::copy(outDevice.begin(), outDevice.end(), outHost.begin());

    bool NanValueDetected = false;
    int32_t numErrorWithin1E_3 = 0;
    for (int32_t i = 0; i < batchSize * qSequenceLength * numQHeads * headSize; ++i)
    {
        EXPECT_TRUE(isclose(outHost[i], outReference[i], 1e-2, 1e-2));
        if (isclose(outHost[i], outReference[i], 1e-3, 1e-3))
        {
            numErrorWithin1E_3++;
        }
        if (isnan(__half2float(outHost[i])))
        {
            NanValueDetected = true;
        }
    }
    float passRate1E_3 = static_cast<float>(numErrorWithin1E_3) / (batchSize * qSequenceLength * numQHeads * headSize);

    std::cout << "XQA Tree Attention Decoding test. [Padded capacity] batch_size: " << batchSize
              << " num_Q_heads: " << numQHeads << " num_KV_heads: " << numKVHeads << " head_size: " << headSize
              << " kvcache seq_len: " << kvSequenceLength << " q_seq_len: " << qSequenceLength
              << " capacity: " << capacity << " pass_rate_1e-3: " << passRate1E_3 << std::endl;
    EXPECT_GT(passRate1E_3, 0.9);
    EXPECT_FALSE(NanValueDetected);
}

TEST(XQATreeAttentionDecodingTest, accuracyBaseLen0Gemma4)
{
    // Gemma4 prefill scenario: baseLen=0 (kvSeqLen==qSeqLen), GQA 8:1, headDim=512
    TestXQATreeAttentionDecodingAccuracy(1, 8, 1, 512, 16, 16);
    TestXQATreeAttentionDecodingAccuracy(1, 8, 1, 512, 32, 32);
    TestXQATreeAttentionDecodingAccuracy(1, 8, 1, 512, 64, 64);
    // Also test with baseLen > 0 for the same config (decode after prefill)
    TestXQATreeAttentionDecodingAccuracy(1, 8, 1, 512, 32, 16);
}

TEST(XQATreeAttentionDecodingTest, accuracyPaddedCapacityGemma4)
{
    // Gemma4 runtime: capacity=256, only 16 tokens filled (baseLen=0)
    TestXQATreeAttentionDecodingWithPaddedCapacity(1, 8, 1, 512, 16, 16, 256);
    // capacity=256, 32 tokens filled
    TestXQATreeAttentionDecodingWithPaddedCapacity(1, 8, 1, 512, 32, 32, 256);
    // capacity=128, 16 tokens filled
    TestXQATreeAttentionDecodingWithPaddedCapacity(1, 8, 1, 512, 16, 16, 128);
}

TEST(XQATreeAttentionDecodingTest, accuracyKVRatio4HeadDim128)
{
    /// KVSequence 256, QSequence 48
    TestXQATreeAttentionDecodingAccuracy(1, 32, 8, 128, 512, 10);
    /// KVSequence 128, QSequence 64
    TestXQATreeAttentionDecodingAccuracy(1, 32, 8, 128, 256, 32);
    /// KVSequence 64, QSequence 128
    TestXQATreeAttentionDecodingAccuracy(1, 32, 8, 128, 320, 60);
}

TEST(XQATreeAttentionDecodingTest, accuracyKVRatio8HeadDim128)
{
    /// KVSequence 256, QSequence 48
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 256, 48);
    /// KVSequence 128, QSequence 64
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 128, 64);
    /// KVSequence 192, QSequence 60 KV-head = 3
    TestXQATreeAttentionDecodingAccuracy(1, 24, 3, 128, 192, 60);
    /// KVSequence 512, QSequence 20， KV-head = 3
    TestXQATreeAttentionDecodingAccuracy(1, 24, 3, 128, 512, 20);
}

TEST(XQATreeAttentionDecodingTest, accuracyKVRatio7HeadDim64)
{
    /// KVSequence 256, QSequence 48
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 256, 48);
    /// KVSequence 128, QSequence 64
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 128, 64);
    /// KVSequence 192, QSequence 60
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 192, 60);
    /// KVSequence 512, QSequence 20
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 512, 20);
}

TEST(XQATreeAttentionDecodingTest, accuracyKVRatio4HeadDim256)
{
    TestXQATreeAttentionDecodingAccuracy(1, 16, 4, 256, 512, 10);
    TestXQATreeAttentionDecodingAccuracy(1, 16, 4, 256, 256, 32);
    TestXQATreeAttentionDecodingAccuracy(1, 8, 2, 256, 512, 20);
}

TEST(XQATreeAttentionDecodingTest, accuracyKVRatio6HeadDim256)
{
    TestXQATreeAttentionDecodingAccuracy(1, 24, 4, 256, 512, 10);
    TestXQATreeAttentionDecodingAccuracy(1, 24, 4, 256, 256, 32);
}

TEST(XQATreeAttentionDecodingTest, accuracyKVRatio8HeadDim512)
{
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 512, 256, 20);
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 512, 128, 33);
    // Gemma4-like: GQA 8:1, headDim=512, baseLen=0 (kvSeqLen==qSeqLen, initial prefill)
    TestXQATreeAttentionDecodingAccuracy(1, 8, 1, 512, 16, 16);
    TestXQATreeAttentionDecodingAccuracy(1, 8, 1, 512, 32, 32);
}

TEST(XQATreeAttentionDecodingTest, pagedKVAccuracyKVRatio8HeadDim512)
{
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 512, 256, 20, false, 0, 128);
}

TEST(XQATreeAttentionDecodingTest, pagedKVAccuracyNon512HeadDims)
{
    TestXQATreeAttentionDecodingAccuracy(1, 8, 2, 32, 128, 16, false, 0, 128);
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 256, 32, false, 0, 128);
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 256, 16, false, 0, 128);
    TestXQATreeAttentionDecodingAccuracy(1, 24, 4, 256, 256, 16, false, 0, 128);
}

TEST(XQATreeAttentionDecodingTest, configurableAttentionScale)
{
    TestXQATreeAttentionDecodingAccuracy(1, 8, 2, 128, 128, 16, false, 0, 0, 1.0F);
    TestXQATreeAttentionDecodingAccuracy(1, 8, 2, 128, 128, 16, false, 0, 0, 0.37F);
}

TEST(XQATreeAttentionDecodingTest, pagedKVCustomAttentionScale)
{
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 256, 16, false, 0, 128, 0.37F);
}

TEST(XQATreeAttentionDecodingTest, slidingWindowAccuracy)
{
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 256, 16, false, 64);
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 192, 32, false, 96);
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 128, 16, false, 128);
    TestXQATreeAttentionDecodingAccuracy(1, 24, 4, 256, 256, 16, false, 129);
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 96, 48, false, 192);
}

#if SUPPORTS_FP8
TEST(XQATreeAttentionDecodingFP8Test, accuracyKVRatio4HeadDim128)
{
    /// KVSequence 256, QSequence 48
    TestXQATreeAttentionDecodingAccuracy(1, 32, 8, 128, 512, 10, true);
    /// KVSequence 128, QSequence 64
    TestXQATreeAttentionDecodingAccuracy(1, 32, 8, 128, 256, 32, true);
    /// KVSequence 64, QSequence 128
    TestXQATreeAttentionDecodingAccuracy(1, 32, 8, 128, 320, 60, true);
}

TEST(XQATreeAttentionDecodingFP8Test, accuracyKVRatio8HeadDim128)
{
    /// KVSequence 256, QSequence 48
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 256, 48, true);
    /// KVSequence 128, QSequence 64
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 128, 64, true);
    /// KVSequence 192, QSequence 60 KV-head = 3
    TestXQATreeAttentionDecodingAccuracy(1, 24, 3, 128, 192, 60, true);
    /// KVSequence 512, QSequence 20， KV-head = 3
    TestXQATreeAttentionDecodingAccuracy(1, 24, 3, 128, 512, 20, true);
}

TEST(XQATreeAttentionDecodingFP8Test, accuracyKVRatio7HeadDim64)
{
    /// KVSequence 256, QSequence 48
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 256, 48, true);
    /// KVSequence 128, QSequence 64
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 128, 64, true);
    /// KVSequence 192, QSequence 60
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 192, 60, true);
    /// KVSequence 512, QSequence 20
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 512, 20, true);
}

TEST(XQATreeAttentionDecodingFP8Test, accuracyKVRatio4HeadDim256)
{
    TestXQATreeAttentionDecodingAccuracy(1, 16, 4, 256, 512, 10, true);
    TestXQATreeAttentionDecodingAccuracy(1, 16, 4, 256, 256, 32, true);
    TestXQATreeAttentionDecodingAccuracy(1, 8, 2, 256, 512, 20, true);
}

TEST(XQATreeAttentionDecodingFP8Test, accuracyKVRatio6HeadDim256)
{
    TestXQATreeAttentionDecodingAccuracy(1, 24, 4, 256, 512, 10, true);
    TestXQATreeAttentionDecodingAccuracy(1, 24, 4, 256, 256, 32, true);
}

TEST(XQATreeAttentionDecodingFP8Test, accuracyKVRatio8HeadDim512)
{
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 512, 256, 20, true);
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 512, 128, 33, true);
}

TEST(XQATreeAttentionDecodingFP8Test, pagedKVAccuracyKVRatio8HeadDim512)
{
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 512, 256, 20, true, 0, 128);
}

TEST(XQATreeAttentionDecodingFP8Test, pagedKVAccuracyNon512HeadDims)
{
    TestXQATreeAttentionDecodingAccuracy(1, 8, 2, 32, 128, 16, true, 0, 128);
    TestXQATreeAttentionDecodingAccuracy(1, 14, 2, 64, 256, 32, true, 0, 128);
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 256, 16, true, 0, 128);
    TestXQATreeAttentionDecodingAccuracy(1, 24, 4, 256, 256, 16, true, 0, 128);
}

TEST(XQATreeAttentionDecodingFP8Test, pagedKVCustomAttentionScale)
{
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 256, 16, true, 0, 128, 0.37F);
}

TEST(XQATreeAttentionDecodingFP8Test, slidingWindowAccuracy)
{
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 256, 16, true, 64);
    TestXQATreeAttentionDecodingAccuracy(1, 32, 4, 128, 128, 16, true, 128);
    TestXQATreeAttentionDecodingAccuracy(1, 24, 4, 256, 256, 16, true, 129);
}
#endif

struct XQATreeAttentionBenchShape
{
    int32_t batchSize{1};
    int32_t numQHeads{32};
    int32_t numKVHeads{4};
    int32_t headSize{512};
    int32_t kvSequenceLength{256};
    int32_t qSequenceLength{20};
    int32_t slidingWindowSize{0};
};

struct XQATreeAttentionBenchConfig
{
    int32_t warmup{20};
    int32_t iterations{100};
    int32_t buffers{32};
    int32_t tokensPerPage{128};
    float peakBwGbps{273.0F};
    float peakTflops{0.0F};
};

int32_t getBenchEnvInt(char const* name, int32_t defaultValue)
{
    char const* value = std::getenv(name);
    if (value == nullptr)
    {
        return defaultValue;
    }
    int32_t const parsed = std::atoi(value);
    return parsed > 0 ? parsed : defaultValue;
}

float getBenchEnvFloat(char const* name, float defaultValue)
{
    char const* value = std::getenv(name);
    if (value == nullptr)
    {
        return defaultValue;
    }
    float const parsed = std::atof(value);
    return parsed > 0.0F ? parsed : defaultValue;
}

XQATreeAttentionBenchConfig getBenchConfigFromEnv()
{
    XQATreeAttentionBenchConfig config;
    config.warmup = getBenchEnvInt("XQA_BENCH_WARMUP", config.warmup);
    config.iterations = getBenchEnvInt("XQA_BENCH_ITERS", config.iterations);
    config.buffers = getBenchEnvInt("XQA_BENCH_BUFFERS", config.buffers);
    config.tokensPerPage = getBenchEnvInt("XQA_BENCH_TOKENS_PER_PAGE", config.tokensPerPage);
    config.peakBwGbps = getBenchEnvFloat("XQA_BENCH_PEAK_BW_GBPS", config.peakBwGbps);
    config.peakTflops = getBenchEnvFloat("XQA_BENCH_PEAK_TFLOPS", config.peakTflops);
    return config;
}

XQATreeAttentionBenchShape getBenchShapeFromEnv()
{
    XQATreeAttentionBenchShape shape;
    shape.batchSize = getBenchEnvInt("XQA_BENCH_BATCH", shape.batchSize);
    shape.numQHeads = getBenchEnvInt("XQA_BENCH_Q_HEADS", shape.numQHeads);
    shape.numKVHeads = getBenchEnvInt("XQA_BENCH_KV_HEADS", shape.numKVHeads);
    shape.headSize = getBenchEnvInt("XQA_BENCH_HEAD_SIZE", shape.headSize);
    shape.kvSequenceLength = getBenchEnvInt("XQA_BENCH_KV_SEQ_LEN", shape.kvSequenceLength);
    shape.qSequenceLength = getBenchEnvInt("XQA_BENCH_Q_SEQ_LEN", shape.qSequenceLength);
    shape.slidingWindowSize = getBenchEnvInt("XQA_BENCH_SLIDING_WINDOW_SIZE", shape.slidingWindowSize);
    return shape;
}

std::string formatBenchFloat(double value, int32_t precision = 4)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

std::string formatBenchMaybeFloat(double value, bool valid)
{
    if (!valid)
    {
        return "n/a";
    }
    return formatBenchFloat(value);
}

std::string shapeToString(XQATreeAttentionBenchShape const& shape)
{
    std::ostringstream oss;
    oss << "B" << shape.batchSize << "_Q" << shape.qSequenceLength << "_KV" << shape.kvSequenceLength << "_H"
        << shape.headSize;
    if (shape.slidingWindowSize > 0)
    {
        oss << "_SW" << shape.slidingWindowSize;
    }
    return oss.str();
}

std::vector<int32_t> makeCausalPackedTreeMask(int32_t batchSize, int32_t qSequenceLength)
{
    int32_t constexpr kBITS_PER_PACKED_MASK{32};
    int32_t const packedMasksPerToken = divUp(qSequenceLength, kBITS_PER_PACKED_MASK);
    std::vector<int32_t> packedMask(batchSize * qSequenceLength * packedMasksPerToken, 0);
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (int32_t row = 0; row < qSequenceLength; ++row)
        {
            for (int32_t packedIdx = 0; packedIdx < packedMasksPerToken; ++packedIdx)
            {
                uint32_t mask = 0U;
                for (int32_t bit = 0; bit < kBITS_PER_PACKED_MASK; ++bit)
                {
                    int32_t const col = packedIdx * kBITS_PER_PACKED_MASK + bit;
                    if (col <= row && col < qSequenceLength)
                    {
                        mask |= (1U << bit);
                    }
                }
                size_t const maskIdx
                    = (static_cast<size_t>(batchIdx) * qSequenceLength + row) * packedMasksPerToken + packedIdx;
                packedMask[maskIdx] = static_cast<int32_t>(mask);
            }
        }
    }
    return packedMask;
}

std::vector<half> makeBenchQHost(XQATreeAttentionBenchShape const& shape)
{
    std::vector<half> qHost(
        static_cast<size_t>(shape.batchSize) * shape.qSequenceLength * shape.numQHeads * shape.headSize);
    uniformFloatInitialization(qHost, -1.0F, 1.0F);
    return qHost;
}

std::vector<half> makeBenchKvHost(XQATreeAttentionBenchShape const& shape)
{
    std::vector<half> kvHost(
        static_cast<size_t>(shape.batchSize) * 2 * shape.numKVHeads * shape.kvSequenceLength * shape.headSize);
    uniformFloatInitialization(kvHost, -1.0F, 1.0F);
    return kvHost;
}

std::vector<int32_t> makeBenchPagedKvPageList(XQATreeAttentionBenchShape const& shape, int32_t tokensPerPage)
{
    int32_t const maxNbPagesPerSeq = shape.kvSequenceLength / tokensPerPage;
    std::vector<int32_t> pageList(static_cast<size_t>(shape.batchSize) * 2 * maxNbPagesPerSeq);
    for (int32_t batchIdx = 0; batchIdx < shape.batchSize; ++batchIdx)
    {
        for (int32_t page = 0; page < maxNbPagesPerSeq; ++page)
        {
            size_t const batchBase = static_cast<size_t>(batchIdx) * 2 * maxNbPagesPerSeq;
            pageList[batchBase + page] = static_cast<int32_t>(batchBase + page);
            pageList[batchBase + maxNbPagesPerSeq + page] = static_cast<int32_t>(batchBase + maxNbPagesPerSeq + page);
        }
    }
    return pageList;
}

template <typename KVType>
std::vector<KVType> makeBenchPagedKvHost(XQATreeAttentionBenchShape const& shape,
    std::vector<KVType> const& compactKvHost, int32_t tokensPerPage, std::vector<int32_t> const& pageList)
{
    int32_t const maxNbPagesPerSeq = shape.kvSequenceLength / tokensPerPage;
    int32_t const nbPoolPages = shape.batchSize * 2 * maxNbPagesPerSeq;
    std::vector<KVType> pagedKvHost(
        static_cast<size_t>(nbPoolPages) * shape.numKVHeads * tokensPerPage * shape.headSize);
    size_t const kvStride = static_cast<size_t>(shape.numKVHeads) * shape.kvSequenceLength * shape.headSize;
    for (int32_t batchIdx = 0; batchIdx < shape.batchSize; ++batchIdx)
    {
        size_t const batchBase = static_cast<size_t>(batchIdx) * 2 * kvStride;
        size_t const vBase = batchBase + kvStride;
        for (int32_t hkv = 0; hkv < shape.numKVHeads; ++hkv)
        {
            for (int32_t skv = 0; skv < shape.kvSequenceLength; ++skv)
            {
                int32_t const page = skv / tokensPerPage;
                int32_t const tokenInPage = skv % tokensPerPage;
                size_t const pageListBase = static_cast<size_t>(batchIdx) * 2 * maxNbPagesPerSeq;
                int32_t const kPageIdx = pageList[pageListBase + page];
                int32_t const vPageIdx = pageList[pageListBase + maxNbPagesPerSeq + page];
                for (int32_t d = 0; d < shape.headSize; ++d)
                {
                    size_t const compactOffset
                        = (static_cast<size_t>(hkv) * shape.kvSequenceLength + skv) * shape.headSize + d;
                    pagedKvHost[getPagedKVPoolOffset(kPageIdx, shape.numKVHeads, tokensPerPage, shape.headSize, hkv,
                        tokenInPage, d)] = compactKvHost[batchBase + compactOffset];
                    pagedKvHost[getPagedKVPoolOffset(vPageIdx, shape.numKVHeads, tokensPerPage, shape.headSize, hkv,
                        tokenInPage, d)] = compactKvHost[vBase + compactOffset];
                }
            }
        }
    }
    return pagedKvHost;
}

#if SUPPORTS_FP8
std::vector<__nv_fp8_e4m3> makeBenchKvHostFp8(
    XQATreeAttentionBenchShape const& shape, std::vector<half> const& kvHost, float& kScale, float& vScale)
{
    float kAmax = 0.0F;
    float vAmax = 0.0F;
    int32_t const kvStrideHalf = shape.numKVHeads * shape.kvSequenceLength * shape.headSize;
    for (int32_t batchIdx = 0; batchIdx < shape.batchSize; ++batchIdx)
    {
        size_t const batchBase = static_cast<size_t>(batchIdx) * 2 * kvStrideHalf;
        size_t const vBase = batchBase + kvStrideHalf;
        for (int32_t idx = 0; idx < kvStrideHalf; ++idx)
        {
            kAmax = std::max(kAmax, std::fabs(__half2float(kvHost[batchBase + idx])));
            vAmax = std::max(vAmax, std::fabs(__half2float(kvHost[vBase + idx])));
        }
    }

    constexpr float kFP8_E4M3_MAX{448.0F};
    kScale = std::max(kAmax, 1e-6F) / kFP8_E4M3_MAX;
    vScale = std::max(vAmax, 1e-6F) / kFP8_E4M3_MAX;
    float const kScaleOrigQuant = 1.0F / kScale;
    float const vScaleOrigQuant = 1.0F / vScale;

    std::vector<__nv_fp8_e4m3> kvHostFp8(kvHost.size());
    for (int32_t batchIdx = 0; batchIdx < shape.batchSize; ++batchIdx)
    {
        size_t const batchBase = static_cast<size_t>(batchIdx) * 2 * kvStrideHalf;
        size_t const vBase = batchBase + kvStrideHalf;
        for (int32_t idx = 0; idx < kvStrideHalf; ++idx)
        {
            kvHostFp8[batchBase + idx] = __nv_fp8_e4m3(__half2float(kvHost[batchBase + idx]) * kScaleOrigQuant);
            kvHostFp8[vBase + idx] = __nv_fp8_e4m3(__half2float(kvHost[vBase + idx]) * vScaleOrigQuant);
        }
    }
    return kvHostFp8;
}
#endif

template <typename KVType>
struct XQATreeAttentionBenchBuffer
{
    thrust::device_vector<half> q;
    thrust::device_vector<KVType> kv;
    thrust::device_vector<half> output;
    thrust::device_vector<int32_t> sequenceLengths;
    thrust::device_vector<int32_t> packedMask;
    thrust::device_vector<int32_t> pageList;
    float kScale{1.0F};
    float vScale{1.0F};

    XQATreeAttentionBenchBuffer(std::vector<half> const& qHost, std::vector<KVType> const& kvHost, size_t outputElems,
        std::vector<int32_t> const& sequenceLengthsHost, std::vector<int32_t> const& packedMaskHost,
        std::vector<int32_t> const& pageListHost, float kScale_, float vScale_)
        : q(qHost)
        , kv(kvHost)
        , output(outputElems, __float2half(0.0F))
        , sequenceLengths(sequenceLengthsHost)
        , packedMask(packedMaskHost)
        , pageList(pageListHost)
        , kScale(kScale_)
        , vScale(vScale_)
    {
    }
};

double estimateXQATreeAttentionFlops(XQATreeAttentionBenchShape const& shape)
{
    int32_t const effectiveKvLength = shape.slidingWindowSize > 0
        ? std::min(shape.kvSequenceLength, shape.slidingWindowSize)
        : shape.kvSequenceLength;
    return 4.0 * static_cast<double>(shape.batchSize) * shape.numQHeads * shape.qSequenceLength * effectiveKvLength
        * shape.headSize;
}

template <typename KVType>
double estimateXQATreeAttentionBytes(XQATreeAttentionBenchShape const& shape, int32_t tokensPerPage)
{
    int32_t const effectiveKvLength = shape.slidingWindowSize > 0
        ? std::min(shape.kvSequenceLength, shape.slidingWindowSize)
        : shape.kvSequenceLength;
    size_t const qBytes = static_cast<size_t>(shape.batchSize) * shape.qSequenceLength * shape.numQHeads
        * shape.headSize * sizeof(half);
    size_t const kvBytes = static_cast<size_t>(shape.batchSize) * 2 * shape.numKVHeads * effectiveKvLength
        * shape.headSize * sizeof(KVType);
    size_t const outputBytes = qBytes;
    size_t const maskBytes = static_cast<size_t>(shape.batchSize) * shape.qSequenceLength
        * divUp(shape.qSequenceLength, 32) * sizeof(int32_t);
    size_t const pageListBytes = tokensPerPage > 0
        ? static_cast<size_t>(shape.batchSize) * 2 * divUp(effectiveKvLength, tokensPerPage) * sizeof(int32_t)
        : 0;
    return static_cast<double>(qBytes + kvBytes + outputBytes + maskBytes + pageListBytes);
}

template <typename KVType>
void runXQATreeAttentionBenchmark(nvinfer1::DataType kvDataType, char const* kvLabel, int32_t tokensPerPage = 0)
{
    initializeCudaContextForXQATest();

    XQATreeAttentionBenchConfig const config = getBenchConfigFromEnv();
    XQATreeAttentionBenchShape const shape = getBenchShapeFromEnv();
    bool const usePagedKVCache = tokensPerPage > 0;
    if (usePagedKVCache)
    {
        ASSERT_EQ(shape.kvSequenceLength % tokensPerPage, 0);
    }

    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (kvDataType == DataType::kFP8 && smVersion < 89)
    {
        GTEST_SKIP() << "Skipping FP8 XQA tree attention benchmark: requires SM >= 89, but got SM " << smVersion;
    }
    ASSERT_TRUE(trt_edgellm::canCompileXQAKernel(
        shape.numQHeads, shape.numKVHeads, shape.headSize, smVersion, DataType::kHALF, kvDataType))
        << "No compatible XQA kernel for SM " << smVersion << " paged_kv=" << usePagedKVCache;
    ASSERT_TRUE(trt_edgellm::loadXQAJitKernelForTest(smVersion, DataType::kHALF, kvDataType, shape.headSize,
        shape.numQHeads, shape.numKVHeads, shape.slidingWindowSize > 0, /*specDecode=*/true, tokensPerPage));

    int32_t deviceId{0};
    CUDA_CHECK(cudaGetDevice(&deviceId));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

    std::vector<int32_t> const sequenceLengths(shape.batchSize, shape.kvSequenceLength);
    std::vector<int32_t> const packedMask = makeCausalPackedTreeMask(shape.batchSize, shape.qSequenceLength);
    std::vector<int32_t> const pageList
        = usePagedKVCache ? makeBenchPagedKvPageList(shape, tokensPerPage) : std::vector<int32_t>{};
    size_t const outputElems
        = static_cast<size_t>(shape.batchSize) * shape.qSequenceLength * shape.numQHeads * shape.headSize;

    std::vector<XQATreeAttentionBenchBuffer<KVType>> buffers;
    buffers.reserve(config.buffers);
    for (int32_t bufferIdx = 0; bufferIdx < config.buffers; ++bufferIdx)
    {
        std::vector<half> qHost = makeBenchQHost(shape);
        std::vector<half> kvHostHalf = makeBenchKvHost(shape);
        float kScale = 1.0F;
        float vScale = 1.0F;
        if constexpr (std::is_same_v<KVType, half>)
        {
            std::vector<half> const& kvHost = kvHostHalf;
            std::vector<half> pagedKvHost;
            if (usePagedKVCache)
            {
                pagedKvHost = makeBenchPagedKvHost(shape, kvHostHalf, tokensPerPage, pageList);
            }
            buffers.emplace_back(qHost, usePagedKVCache ? pagedKvHost : kvHost, outputElems, sequenceLengths,
                packedMask, pageList, kScale, vScale);
        }
#if SUPPORTS_FP8
        else
        {
            std::vector<__nv_fp8_e4m3> kvHostFp8 = makeBenchKvHostFp8(shape, kvHostHalf, kScale, vScale);
            std::vector<__nv_fp8_e4m3> pagedKvHostFp8;
            if (usePagedKVCache)
            {
                pagedKvHostFp8 = makeBenchPagedKvHost(shape, kvHostFp8, tokensPerPage, pageList);
            }
            buffers.emplace_back(qHost, usePagedKVCache ? pagedKvHostFp8 : kvHostFp8, outputElems, sequenceLengths,
                packedMask, pageList, kScale, vScale);
        }
#endif
    }

    trt_edgellm::DecoderXQARunner runner(
        DataType::kHALF, kvDataType, shape.batchSize, shape.numQHeads, shape.numKVHeads, shape.headSize, smVersion);
    std::vector<trt_edgellm::XQALaunchParams> paramsList;
    paramsList.reserve(buffers.size());
    for (auto& buffer : buffers)
    {
        auto params = runner.initXQAParams();
        params.qSeqLen = shape.qSequenceLength;
        params.qInputPtr = thrust::raw_pointer_cast(buffer.q.data());
        params.kvCache.data = thrust::raw_pointer_cast(buffer.kv.data());
        params.kvCache.sequence_lengths = thrust::raw_pointer_cast(buffer.sequenceLengths.data());
        params.kvCache.capacity = shape.kvSequenceLength;
        if (usePagedKVCache)
        {
            params.kvCache.pageList = thrust::raw_pointer_cast(buffer.pageList.data());
            params.kvCache.tokensPerPage = static_cast<uint32_t>(tokensPerPage);
        }
        params.output = thrust::raw_pointer_cast(buffer.output.data());
        params.treeAttnMask = thrust::raw_pointer_cast(buffer.packedMask.data());
        params.attentionScale = 1.0F / std::sqrt(static_cast<float>(shape.headSize));
        params.kScale = buffer.kScale;
        params.vScale = buffer.vScale;
        params.slidingWinSize = shape.slidingWindowSize > 0 ? static_cast<uint32_t>(shape.slidingWindowSize) : 0U;
        paramsList.push_back(params);
    }

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Defer destroyStream{[&stream]() { CUDA_CHECK(cudaStreamDestroy(stream)); }};

    auto dispatch = [&](int32_t iter) {
        auto& params = paramsList[static_cast<size_t>(iter % config.buffers)];
        runner.dispatchSpecDecodeXQAKernel(params, stream);
    };

    for (int32_t iter = 0; iter < config.warmup; ++iter)
    {
        dispatch(iter);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    Defer destroyEvents{[&start, &stop]() {
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }};

    std::vector<float> latenciesMs;
    latenciesMs.reserve(config.iterations);
    for (int32_t iter = 0; iter < config.iterations; ++iter)
    {
        CUDA_CHECK(cudaEventRecord(start, stream));
        dispatch(iter);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsedMs{0.0F};
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
        latenciesMs.push_back(elapsedMs);
    }
    CUDA_CHECK(cudaGetLastError());

    std::sort(latenciesMs.begin(), latenciesMs.end());
    double const medianMs = latenciesMs[latenciesMs.size() / 2];
    int32_t const effectiveKvLength = shape.slidingWindowSize > 0
        ? std::min(shape.kvSequenceLength, shape.slidingWindowSize)
        : shape.kvSequenceLength;
    double const flops = estimateXQATreeAttentionFlops(shape);
    double const bytes = estimateXQATreeAttentionBytes<KVType>(shape, tokensPerPage);
    bool const hasPeakTflops = config.peakTflops > 0.0F;
    double const minComputeMs = hasPeakTflops ? flops / static_cast<double>(config.peakTflops) / 1.0e9 : 0.0;
    double const minMemoryMs = bytes / static_cast<double>(config.peakBwGbps) / 1.0e6;
    double const achievedTflops = flops / medianMs / 1.0e9;
    double const achievedGbps = bytes / medianMs / 1.0e6;
    double const computeUtil = hasPeakTflops ? achievedTflops / static_cast<double>(config.peakTflops) * 100.0 : 0.0;
    double const bandwidthUtil = achievedGbps / static_cast<double>(config.peakBwGbps) * 100.0;

    std::string const minComputeStr = formatBenchMaybeFloat(minComputeMs, hasPeakTflops);
    std::string const computeUtilStr = formatBenchMaybeFloat(computeUtil, hasPeakTflops);
    std::ostringstream benchmarkOutput;
    benchmarkOutput << "\nXQA Tree Attention Decoding Benchmark\n";
    benchmarkOutput << "Device: " << prop.name << " SM " << prop.major << "." << prop.minor
                    << " sm_count: " << prop.multiProcessorCount << "\n";
    benchmarkOutput << "DTypes: Q/O=FP16 KV=" << kvLabel << "\n";
    benchmarkOutput << "KV cache: " << (usePagedKVCache ? "paged" : "contiguous")
                    << " tokens_per_page=" << tokensPerPage << "\n";
    benchmarkOutput << "Warmup: " << config.warmup << " iterations: " << config.iterations
                    << " buffers: " << config.buffers << "\n";
    benchmarkOutput << "Buffer policy: round-robin independent Q/KV/output/mask buffers\n";
    benchmarkOutput << "Workspace: none\n";
    benchmarkOutput << "shape,kv_dtype,paged_kv,tokens_per_page,q_heads,kv_heads,group,q_len,kv_len,sliding_window,"
                    << "effective_kv_len,head_dim,latency_ms,min_compute_ms,min_memory_ms,achieved_tflops,"
                    << "achieved_gbps,compute_util_pct,bw_util_pct\n";
    benchmarkOutput << shapeToString(shape) << "," << kvLabel << "," << usePagedKVCache << "," << tokensPerPage << ","
                    << shape.numQHeads << "," << shape.numKVHeads << "," << shape.numQHeads / shape.numKVHeads << ","
                    << shape.qSequenceLength << "," << shape.kvSequenceLength << "," << shape.slidingWindowSize << ","
                    << effectiveKvLength << "," << shape.headSize << "," << formatBenchFloat(medianMs) << ","
                    << minComputeStr << "," << formatBenchFloat(minMemoryMs) << "," << formatBenchFloat(achievedTflops)
                    << "," << formatBenchFloat(achievedGbps) << "," << computeUtilStr << ","
                    << formatBenchFloat(bandwidthUtil) << "\n";
    benchmarkOutput << "Metric notes: FLOPs estimate = 4 * B * QHeads * QLen * effective_KVLen * HeadDim.\n";
    benchmarkOutput << "Memory estimate includes Q read, effective K/V cache read, output write, packed tree mask read,"
                    << " and page list read when paged.\n";
    benchmarkOutput << "Peak assumptions: bandwidth=" << config.peakBwGbps
                    << " GB/s, compute=" << (hasPeakTflops ? formatBenchFloat(config.peakTflops) : std::string("n/a"))
                    << " TFLOP/s.\n";
    // GTest emits RecordProperty values only in structured output, not stdout. For example:
    // `GTEST_OUTPUT=json:/tmp/xqa_bench.json ./run.py xqa-bench`, then read each testcase's
    // `benchmark_output` CSV from the JSON `testsuites[].testsuite[]` entries.
    ::testing::Test::RecordProperty("benchmark_output", benchmarkOutput.str());
    EXPECT_GT(medianMs, 0.0);
}

TEST(XQATreeAttentionDecodingBenchmark, perfKVRatio8HeadDim512SeparateKV)
{
    runXQATreeAttentionBenchmark<half>(DataType::kHALF, "FP16");
}

TEST(XQATreeAttentionDecodingBenchmark, perfKVRatio8HeadDim512PagedKV)
{
    runXQATreeAttentionBenchmark<half>(DataType::kHALF, "FP16", getBenchConfigFromEnv().tokensPerPage);
}

#if SUPPORTS_FP8
TEST(XQATreeAttentionDecodingBenchmark, perfKVRatio8HeadDim512SeparateKVFp8)
{
    runXQATreeAttentionBenchmark<__nv_fp8_e4m3>(DataType::kFP8, "FP8");
}

TEST(XQATreeAttentionDecodingBenchmark, perfKVRatio8HeadDim512PagedKVFp8)
{
    runXQATreeAttentionBenchmark<__nv_fp8_e4m3>(DataType::kFP8, "FP8", getBenchConfigFromEnv().tokensPerPage);
}
#endif

/// Test to reproduce padding issue: compare padded vs non-padded attention output
/// This test verifies that when we pad the Q sequence, the output for valid tokens
/// should remain the same as the non-padded case.
/// Also compares both outputs against CPU reference implementation.
void TestXQAPaddingConsistency(int32_t batchSize, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t kvSequenceLength, int32_t actualQSeqLen, int32_t paddedQSeqLen)
{
    std::cout << "\n========== Padding Consistency Test ==========" << std::endl;
    std::cout << "batch_size: " << batchSize << ", numQHeads: " << numQHeads << ", numKVHeads: " << numKVHeads
              << ", headSize: " << headSize << std::endl;
    std::cout << "kvSeqLen: " << kvSequenceLength << ", actualQSeqLen: " << actualQSeqLen
              << ", paddedQSeqLen: " << paddedQSeqLen << std::endl;

    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    ASSERT_TRUE(
        trt_edgellm::canCompileXQAKernel(numQHeads, numKVHeads, headSize, smVersion, DataType::kHALF, DataType::kHALF));
    ASSERT_TRUE(trt_edgellm::loadXQAJitKernelForTest(smVersion, DataType::kHALF, DataType::kHALF, headSize, numQHeads,
        numKVHeads, /*slidingWindow=*/false, /*specDecode=*/true));
    float const attentionScale = 1.0F / std::sqrt(static_cast<float>(headSize));

    // Generate random Q, K, V data for actual sequence length
    // Layout: Q is [seqLen, numQHeads, headSize], K/V is [seqLen, numKVHeads, headSize]
    std::vector<half> qActual(numQHeads * headSize * actualQSeqLen);
    std::vector<half> kInput(numKVHeads * headSize * kvSequenceLength);
    std::vector<half> vInput(numKVHeads * headSize * kvSequenceLength);

    uniformFloatInitialization(qActual);
    uniformFloatInitialization(kInput);
    uniformFloatInitialization(vInput);

    // === Compute CPU Reference ===
    // Create causal tree mask for reference: treeMask[i][j] = 1 if token i can attend to token j
    std::vector<int32_t> treeMaskForRef(actualQSeqLen * actualQSeqLen, 0);
    for (int32_t i = 0; i < actualQSeqLen; i++)
    {
        for (int32_t j = 0; j <= i; j++)
        {
            treeMaskForRef[i * actualQSeqLen + j] = 1; // Causal: attend to 0..i
        }
    }

    std::cout << "\n--- Computing CPU Reference ---" << std::endl;
    auto outReference = casualAttentionRef<half>(qActual, kInput, vInput, actualQSeqLen, kvSequenceLength, numQHeads,
        numKVHeads, headSize, attentionScale, std::make_optional(treeMaskForRef));
    std::cout << "Reference output size: " << outReference.size() << std::endl;

    // === Run 1: No padding (actualQSeqLen) ===
    std::vector<half> qNoPadding = qActual;

    // Create packed causal mask for no-padding case
    int32_t const numBitsPerPackedMask = 32;
    int32_t const numPackedMasksPerTokenNoPad = divUp(actualQSeqLen, numBitsPerPackedMask);
    std::vector<int32_t> packedMaskNoPadding(numPackedMasksPerTokenNoPad * actualQSeqLen, 0);

    for (int32_t i = 0; i < actualQSeqLen; i++)
    {
        // Token i attends to tokens 0..i (causal mask)
        int32_t mask = 0;
        for (int32_t j = 0; j <= i && j < 32; j++)
        {
            mask |= (1 << j);
        }
        packedMaskNoPadding[i * numPackedMasksPerTokenNoPad] = mask;
    }

    // Print no-padding mask
    std::cout << "\n--- No-Padding Mask (packed, qSeqLen=" << actualQSeqLen << ") ---" << std::endl;
    for (int32_t i = 0; i < actualQSeqLen; i++)
    {
        std::cout << "Token " << i << ": packed=0x" << std::hex << packedMaskNoPadding[i * numPackedMasksPerTokenNoPad]
                  << std::dec << " -> attends to: ";
        int32_t mask = packedMaskNoPadding[i * numPackedMasksPerTokenNoPad];
        for (int32_t j = 0; j < actualQSeqLen; j++)
        {
            std::cout << ((mask >> j) & 1);
        }
        std::cout << std::endl;
    }

    // Prepare KV cache (layout: [2 * numKVHeads, S, D] for single batch)
    std::vector<half> kvInput;
    kvInput.insert(kvInput.end(), kInput.begin(), kInput.end());
    kvInput.insert(kvInput.end(), vInput.begin(), vInput.end());

    std::vector<int32_t> kvCacheLength(1, kvSequenceLength);

    // Device memory for no-padding run
    thrust::device_vector<half> qNoPaddingDevice(qNoPadding);
    thrust::device_vector<half> kvInputDevice(kvInput);
    thrust::device_vector<half> outNoPaddingDevice(actualQSeqLen * numQHeads * headSize, __float2half(0.0f));
    thrust::device_vector<int32_t> kvCacheLengthDevice(kvCacheLength);
    thrust::device_vector<int32_t> packedMaskNoPaddingDevice(packedMaskNoPadding);

    ASSERT_TRUE(trt_edgellm::loadXQAJitKernelForTest(smVersion, DataType::kHALF, DataType::kHALF, headSize, numQHeads,
        numKVHeads, /*slidingWindow=*/false, /*specDecode=*/true));

    trt_edgellm::DecoderXQARunner runnerNoPad(
        DataType::kHALF, DataType::kHALF, 1, numQHeads, numKVHeads, headSize, smVersion);
    auto paramsNoPad = runnerNoPad.initXQAParams();
    paramsNoPad.qSeqLen = actualQSeqLen;
    paramsNoPad.qInputPtr = thrust::raw_pointer_cast(qNoPaddingDevice.data());
    paramsNoPad.kvCache.data = thrust::raw_pointer_cast(kvInputDevice.data());
    paramsNoPad.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthDevice.data());
    paramsNoPad.kvCache.capacity = kvSequenceLength;
    paramsNoPad.output = thrust::raw_pointer_cast(outNoPaddingDevice.data());
    paramsNoPad.treeAttnMask = thrust::raw_pointer_cast(packedMaskNoPaddingDevice.data());
    paramsNoPad.attentionScale = attentionScale;

    cudaStream_t stream{nullptr};
    runnerNoPad.dispatchSpecDecodeXQAKernel(paramsNoPad, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    thrust::host_vector<half> outNoPaddingHost(outNoPaddingDevice);

    // === Run 2: With padding (paddedQSeqLen) ===
    // Pad Q with zeros
    std::vector<half> qPadded(numQHeads * headSize * paddedQSeqLen, __float2half(0.0f));
    for (int32_t t = 0; t < actualQSeqLen; t++)
    {
        for (int32_t h = 0; h < numQHeads; h++)
        {
            for (int32_t d = 0; d < headSize; d++)
            {
                int32_t srcIdx = t * numQHeads * headSize + h * headSize + d;
                int32_t dstIdx = t * numQHeads * headSize + h * headSize + d;
                qPadded[dstIdx] = qActual[srcIdx];
            }
        }
    }

    // Create mask for padded case:
    // - Valid tokens (0..actualQSeqLen-1): causal mask (attend to self and previous)
    // - Padding tokens (actualQSeqLen..paddedQSeqLen-1): only attend to token 0 (to avoid NaN in softmax)
    int32_t const numPackedMasksPerTokenPad = divUp(paddedQSeqLen, numBitsPerPackedMask);
    std::vector<int32_t> packedMaskPadding(numPackedMasksPerTokenPad * paddedQSeqLen, 0);

    for (int32_t i = 0; i < paddedQSeqLen; i++)
    {
        int32_t mask = 0;
        if (i < actualQSeqLen)
        {
            for (int32_t j = 0; j <= i && j < 32; j++)
            {
                mask |= (1 << j);
            }
        }
        else
        {
            mask = 1;
        }
        packedMaskPadding[i * numPackedMasksPerTokenPad] = mask;
    }

    std::cout << "\n--- Padded Mask (packed, qSeqLen=" << paddedQSeqLen << ", actualQSeqLen=" << actualQSeqLen
              << ") ---" << std::endl;
    for (int32_t i = 0; i < paddedQSeqLen; i++)
    {
        std::string tokenType = (i < actualQSeqLen) ? "valid" : "padding";
        std::cout << "Token " << i << " (" << tokenType << "): packed=0x" << std::hex
                  << packedMaskPadding[i * numPackedMasksPerTokenPad] << std::dec << " -> attends to: ";
        int32_t mask = packedMaskPadding[i * numPackedMasksPerTokenPad];
        for (int32_t j = 0; j < paddedQSeqLen; j++)
        {
            std::cout << ((mask >> j) & 1);
        }
        std::cout << std::endl;
    }

    thrust::device_vector<half> qPaddedDevice(qPadded);
    thrust::device_vector<half> outPaddedDevice(paddedQSeqLen * numQHeads * headSize, __float2half(0.0f));
    thrust::device_vector<int32_t> packedMaskPaddingDevice(packedMaskPadding);

    // For padded case, adjust context length to account for padding tokens
    // This ensures correct context K range: K[0 : paddedContextLen - paddedQSeqLen] = K[0 : kvSequenceLength -
    // actualQSeqLen] Same as what eagleUtilKernels does: sequenceContextLengths = sequenceStartIndices +
    // maxAcceptedTokenNum
    int32_t paddedContextLen = kvSequenceLength + (paddedQSeqLen - actualQSeqLen);
    std::vector<int32_t> kvCacheLengthPadded(1, paddedContextLen);
    thrust::device_vector<int32_t> kvCacheLengthPaddedDevice(kvCacheLengthPadded);

    trt_edgellm::DecoderXQARunner runnerPad(
        DataType::kHALF, DataType::kHALF, 1, numQHeads, numKVHeads, headSize, smVersion);
    auto paramsPad = runnerPad.initXQAParams();
    paramsPad.qSeqLen = paddedQSeqLen;
    paramsPad.qInputPtr = thrust::raw_pointer_cast(qPaddedDevice.data());
    paramsPad.kvCache.data = thrust::raw_pointer_cast(kvInputDevice.data());
    paramsPad.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthPaddedDevice.data());
    paramsPad.kvCache.capacity = kvSequenceLength;
    paramsPad.output = thrust::raw_pointer_cast(outPaddedDevice.data());
    paramsPad.treeAttnMask = thrust::raw_pointer_cast(packedMaskPaddingDevice.data());
    paramsPad.attentionScale = attentionScale;

    runnerPad.dispatchSpecDecodeXQAKernel(paramsPad, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    thrust::host_vector<half> outPaddedHost(outPaddedDevice);

    // === Compare No-padding vs Reference ===
    std::cout << "\n--- No-padding XQA vs CPU Reference ---" << std::endl;
    {
        float maxAbsDiff = 0.0f;
        int32_t numMismatches = 0;
        int32_t totalElements = actualQSeqLen * numQHeads * headSize;

        for (int32_t i = 0; i < totalElements; i++)
        {
            float valXQA = __half2float(outNoPaddingHost[i]);
            float valRef = __half2float(outReference[i]);
            float absDiff = std::abs(valXQA - valRef);
            maxAbsDiff = std::max(maxAbsDiff, absDiff);
            if (!isclose(outNoPaddingHost[i], outReference[i], 1e-2, 1e-2))
            {
                numMismatches++;
            }
        }
        float matchRate = 1.0f - static_cast<float>(numMismatches) / totalElements;
        std::cout << "Max Abs Diff: " << maxAbsDiff << ", Match rate (1e-2): " << matchRate * 100 << "%" << std::endl;

        std::cout << "First 10 values:" << std::endl;
        std::cout << "  XQA:       ";
        for (int32_t i = 0; i < std::min(10, totalElements); i++)
            std::cout << __half2float(outNoPaddingHost[i]) << " ";
        std::cout << std::endl;
        std::cout << "  Reference: ";
        for (int32_t i = 0; i < std::min(10, totalElements); i++)
            std::cout << __half2float(outReference[i]) << " ";
        std::cout << std::endl;

        EXPECT_GT(matchRate, 0.9f) << "No-padding XQA should match reference!";
    }

    // === Compare Padded vs Reference (only valid tokens) ===
    std::cout << "\n--- Padded XQA vs CPU Reference (valid tokens only) ---" << std::endl;
    {
        float maxAbsDiff = 0.0f;
        int32_t numMismatches = 0;
        int32_t totalElements = actualQSeqLen * numQHeads * headSize;

        for (int32_t t = 0; t < actualQSeqLen; t++)
        {
            for (int32_t h = 0; h < numQHeads; h++)
            {
                for (int32_t d = 0; d < headSize; d++)
                {
                    int32_t refIdx = t * numQHeads * headSize + h * headSize + d;
                    int32_t padIdx = t * numQHeads * headSize + h * headSize + d;

                    float valXQA = __half2float(outPaddedHost[padIdx]);
                    float valRef = __half2float(outReference[refIdx]);
                    float absDiff = std::abs(valXQA - valRef);
                    maxAbsDiff = std::max(maxAbsDiff, absDiff);
                    if (!isclose(outPaddedHost[padIdx], outReference[refIdx], 1e-2, 1e-2))
                    {
                        numMismatches++;
                    }
                }
            }
        }
        float matchRate = 1.0f - static_cast<float>(numMismatches) / totalElements;
        std::cout << "Max Abs Diff: " << maxAbsDiff << ", Match rate (1e-2): " << matchRate * 100 << "%" << std::endl;

        std::cout << "First 10 values:" << std::endl;
        std::cout << "  Padded XQA: ";
        for (int32_t i = 0; i < std::min(10, totalElements); i++)
            std::cout << __half2float(outPaddedHost[i]) << " ";
        std::cout << std::endl;
        std::cout << "  Reference:  ";
        for (int32_t i = 0; i < std::min(10, totalElements); i++)
            std::cout << __half2float(outReference[i]) << " ";
        std::cout << std::endl;

        // This is the KEY test: padded output should also match reference!
        EXPECT_GT(matchRate, 0.9f) << "Padded XQA should also match reference for valid tokens!";
    }

    // === Compare No-padding vs Padded (for valid tokens) ===
    std::cout << "\n--- No-padding XQA vs Padded XQA (valid tokens) ---" << std::endl;
    {
        float maxAbsDiff = 0.0f;
        float maxRelDiff = 0.0f;
        int32_t maxAbsDiffIdx = 0;
        int32_t numMismatches = 0;
        int32_t totalElements = actualQSeqLen * numQHeads * headSize;

        for (int32_t i = 0; i < totalElements; i++)
        {
            float valNoPad = __half2float(outNoPaddingHost[i]);
            float valPad = __half2float(outPaddedHost[i]);

            float absDiff = std::abs(valNoPad - valPad);
            float relDiff = absDiff / std::max(std::abs(valNoPad), 1e-6f);

            if (absDiff > maxAbsDiff)
            {
                maxAbsDiff = absDiff;
                maxAbsDiffIdx = i;
            }
            maxRelDiff = std::max(maxRelDiff, relDiff);

            if (!isclose(outNoPaddingHost[i], outPaddedHost[i], 1e-3, 1e-3))
            {
                numMismatches++;
            }
        }

        float matchRate = 1.0f - static_cast<float>(numMismatches) / totalElements;

        std::cout << "Total elements compared: " << totalElements << std::endl;
        std::cout << "Max Absolute Diff: " << maxAbsDiff << " at index " << maxAbsDiffIdx << std::endl;
        std::cout << "  No-pad: " << __half2float(outNoPaddingHost[maxAbsDiffIdx])
                  << ", Padded: " << __half2float(outPaddedHost[maxAbsDiffIdx]) << std::endl;
        std::cout << "Max Relative Diff: " << maxRelDiff * 100 << "%" << std::endl;
        std::cout << "Match rate (1e-3): " << matchRate * 100 << "%" << std::endl;
        std::cout << "Number of mismatches: " << numMismatches << std::endl;

        std::cout << "\nFirst 10 values comparison:" << std::endl;
        std::cout << "No-padding: ";
        for (int32_t i = 0; i < std::min(10, headSize); i++)
            std::cout << __half2float(outNoPaddingHost[i]) << " ";
        std::cout << std::endl;
        std::cout << "Padded:     ";
        for (int32_t i = 0; i < std::min(10, headSize); i++)
            std::cout << __half2float(outPaddedHost[i]) << " ";
        std::cout << std::endl;

        // The key expectation: padded and non-padded should produce same results
        EXPECT_GT(matchRate, 0.99f) << "Padding should not affect valid token outputs!";
        EXPECT_LT(maxAbsDiff, 1e-2f) << "Max absolute diff too large between padded and non-padded!";
    }
}

// Test padding consistency for multi-batch EAGLE.
// The workaround is to pass padded context length (sequenceStartIndices + maxAcceptedTokenNum)
// to ensure correct K range calculation. Only compare actual accepted tokens.
// TODO: Fix XQA kernel to properly handle padding for padded sequence length.
// Currently the XQA tree attention kernel doesn't correctly handle padding tokens
// when actualQSeqLen != paddedQSeqLen. The workaround is to pass padded context length
// (sequenceStartIndices + maxAcceptedTokenNum) to ensure correct K range calculation.
// This test validates padding consistency - enable once the XQA kernel is fixed.
TEST(XQATreeAttentionDecodingTest, PaddingConsistencyTest)
{
    // Reproduce the issue: actualQSeqLen=2, paddedQSeqLen=7
    // This matches the real scenario in EAGLE runtime accept decode token step

    std::cout << "\n=== Test 1: Baseline - no padding (actualQSeqLen == paddedQSeqLen) ===" << std::endl;
    TestXQAPaddingConsistency(1, 32, 8, 128, 256, 2, 2);

    std::cout << "\n=== Test 2: Reproduce issue - actualQSeqLen=2, paddedQSeqLen=7 ===" << std::endl;
    TestXQAPaddingConsistency(1, 32, 8, 128, 256, 2, 7);

    std::cout << "\n=== Test 3: Different padding ratio ===" << std::endl;
    TestXQAPaddingConsistency(1, 32, 8, 128, 256, 3, 10);

    std::cout << "\n=== Test 4: Larger actual sequence ===" << std::endl;
    TestXQAPaddingConsistency(1, 32, 8, 128, 256, 5, 7);
}
