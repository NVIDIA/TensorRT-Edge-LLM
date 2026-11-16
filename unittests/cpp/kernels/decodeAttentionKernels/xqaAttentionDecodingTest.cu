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

#include <cuda.h>
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

using namespace trt_edgellm;

void TestXQAAttentionDecodingAccuracy(int32_t batchSize, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t kvCacheCapacity, bool useFp8Cache = false, int32_t slidingWindowSize = 0,
    std::optional<float> attentionScale = std::nullopt, int32_t fixedContextLen = 0,
    std::optional<std::vector<float>> const& attentionSinks = std::nullopt, bool useMultiBlock = false,
    bool usePagedKV = false, bool expectMultiBlock = true)
{
    ASSERT_FALSE(attentionSinks.has_value() && attentionSinks->size() != static_cast<size_t>(numQHeads));
    float const resolvedAttentionScale = attentionScale.value_or(1.0F / std::sqrt(static_cast<float>(headSize)));
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (useFp8Cache && smVersion < 89)
    {
        GTEST_SKIP() << "Skipping FP8 KV cache tests: requires SM >= 89, but got SM " << smVersion;
    }
    auto dispatch = [&](DecoderXQARunner& runner, XQALaunchParams& params) {
        if (!useMultiBlock)
        {
            runner.dispatchXQAKernel(params, nullptr);
            CUDA_CHECK(cudaStreamSynchronize(nullptr));
            return;
        }
        constexpr size_t kGUARD_BYTES{128};
        constexpr uint8_t kGUARD_VALUE{0xA5};
        params.workspaceSize = runner.getWorkspaceSize(params.kvCache.capacity);
        thrust::device_vector<uint8_t> workspace(params.workspaceSize + 2 * kGUARD_BYTES, kGUARD_VALUE);
        params.workspace = thrust::raw_pointer_cast(workspace.data()) + kGUARD_BYTES;
        thrust::device_vector<int32_t> semaphores(DecoderXQARunner::getSemaphoreCount(), 0);
        params.semaphores = thrust::raw_pointer_cast(semaphores.data());
        cudaStream_t stream{};
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        runner.dispatchXQAKernel(params, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaGraph_t graph{};
        cudaGraphExec_t graphExec{};
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        runner.dispatchXQAKernel(params, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        size_t nodeCount{};
        CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodeCount));
        std::vector<cudaGraphNode_t> nodes(nodeCount);
        CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &nodeCount));
        bool splitKernelFound{false};
        for (auto node : nodes)
        {
            cudaGraphNodeType type{};
            CUDA_CHECK(cudaGraphNodeGetType(node, &type));
            // Given persistent counters, when decode is captured, then no counter reset enters the graph.
            EXPECT_NE(type, cudaGraphNodeTypeMemset);
            if (type == cudaGraphNodeTypeKernel)
            {
                CUDA_KERNEL_NODE_PARAMS kernelParams{};
                CUDA_DRIVER_CHECK(cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(node), &kernelParams));
                uint32_t const headSplits = headSize == 512 && (smVersion == 120 || smVersion == 121) ? 2U : 1U;
                splitKernelFound |= kernelParams.gridDimX > headSplits;
            }
        }
        EXPECT_EQ(splitKernelFound, expectMultiBlock);
        CUDA_CHECK(instantiateCudaGraph(&graphExec, graph));
        for (int repeat = 0; repeat < 20; ++repeat)
        {
            // Workspace may be reused by other layers between replays; counters must survive independently.
            CUDA_CHECK(cudaMemsetAsync(params.workspace, kGUARD_VALUE, params.workspaceSize, stream));
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Given completed split launches, when the counters are read back, then every counter has wrapped to zero.
        thrust::host_vector<int32_t> hostSemaphores(semaphores);
        EXPECT_TRUE(
            std::all_of(hostSemaphores.begin(), hostSemaphores.end(), [](int32_t value) { return value == 0; }));
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
        thrust::host_vector<uint8_t> hostWorkspace(workspace);
        EXPECT_TRUE(std::all_of(hostWorkspace.begin(), hostWorkspace.begin() + kGUARD_BYTES,
            [=](uint8_t value) { return value == kGUARD_VALUE; }));
        EXPECT_TRUE(std::all_of(hostWorkspace.end() - kGUARD_BYTES, hostWorkspace.end(),
            [=](uint8_t value) { return value == kGUARD_VALUE; }));
    };
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
    if (slidingWindowSize > 0 && fixedContextLen == 0)
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
        if (attentionSinks.has_value())
        {
            constexpr int32_t kMODULUS = 257;
            constexpr float kSCALE = 1.0F / 128.0F;
            for (size_t idx = 0; idx < qi.size(); ++idx)
            {
                qi[idx] = __float2half(
                    static_cast<float>((static_cast<int32_t>(idx % kMODULUS) * 37 + i * 17) % kMODULUS - 128) * kSCALE);
            }
            for (size_t idx = 0; idx < ki.size(); ++idx)
            {
                ki[idx] = __float2half(
                    static_cast<float>((static_cast<int32_t>(idx % kMODULUS) * 29 + i * 19) % kMODULUS - 128) * kSCALE);
                vi[idx] = __float2half(
                    static_cast<float>((static_cast<int32_t>(idx % kMODULUS) * 31 + i * 23) % kMODULUS - 128) * kSCALE);
            }
        }
        else
        {
            uniformFloatInitialization(qi, -1.0F, 1.0F);
            uniformFloatInitialization(ki, -1.0F, 1.0F);
            uniformFloatInitialization(vi, -1.0F, 1.0F);
        }

        int32_t const attentionLength = slidingWindowSize > 0 ? std::min(kvLength, slidingWindowSize) : kvLength;
        auto kiRef = sliceKVWindow(ki, numKVHeads, headSize, kvLength, slidingWindowSize);
        auto viRef = sliceKVWindow(vi, numKVHeads, headSize, kvLength, slidingWindowSize);
        auto ref = casualAttentionRef<half>(qi, kiRef, viRef, qSequenceLength, attentionLength, numQHeads, numKVHeads,
            headSize, resolvedAttentionScale, std::nullopt, 1.0F, 1.0F, 0, /*contiguousQuerySwa=*/false,
            attentionSinks);

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
    constexpr int32_t kTOKENS_PER_PAGE{128};
    int32_t const pagesPerSequence = kvCacheCapacity / kTOKENS_PER_PAGE;
    std::vector<int32_t> pageTable(batchSize * 2 * pagesPerSequence, -1);
    auto packKV = [&](auto const& input) {
        auto packed = input;
        for (int32_t b = 0; b < batchSize; ++b)
        {
            for (int32_t kv = 0; kv < 2; ++kv)
            {
                for (int32_t token = 0; token < kvCacheLengths[b]; ++token)
                {
                    int32_t const logicalPage = (b * 2 + kv) * pagesPerSequence + token / kTOKENS_PER_PAGE;
                    int32_t const physicalPage = static_cast<int32_t>(pageTable.size()) - 1 - logicalPage;
                    pageTable[logicalPage] = physicalPage;
                    for (int32_t head = 0; head < numKVHeads; ++head)
                    {
                        size_t const source
                            = ((static_cast<size_t>(b * 2 + kv) * numKVHeads + head) * kvCacheCapacity + token)
                            * headSize;
                        size_t const destination
                            = ((static_cast<size_t>(physicalPage) * kTOKENS_PER_PAGE + token % kTOKENS_PER_PAGE)
                                      * numKVHeads
                                  + head)
                            * headSize;
                        std::copy_n(input.begin() + source, headSize, packed.begin() + destination);
                    }
                }
            }
        }
        return packed;
    };
    // Unused page-table entries stay invalid to catch prefetches from inactive splits.
    auto const kernelKV = usePagedKV ? packKV(kvInput) : kvInput;
    thrust::device_vector<int32_t> pageTableDevice(pageTable);
    // Prepare device memory for kernel execution.
    thrust::device_vector<half> qInputDevice(qInput);
    thrust::device_vector<half> kvInputDevice(kernelKV);
    thrust::device_vector<half> outDevice(outReference.size(), 0.0F);
    thrust::device_vector<int32_t> kvCacheLengthDevice(kvCacheLengths);
    thrust::device_vector<float> attentionSinksDevice(attentionSinks.value_or(std::vector<float>{}));

    EXPECT_TRUE(
        trt_edgellm::canCompileXQAKernel(numQHeads, numKVHeads, headSize, smVersion, DataType::kHALF, DataType::kHALF));
    ASSERT_TRUE(trt_edgellm::loadXQAJitKernelForTest(smVersion, DataType::kHALF, DataType::kHALF, headSize, numQHeads,
        numKVHeads, slidingWindowSize > 0, false, usePagedKV ? kTOKENS_PER_PAGE : 0));
    trt_edgellm::DecoderXQARunner runner(
        DataType::kHALF, DataType::kHALF, batchSize, numQHeads, numKVHeads, headSize, smVersion);
    auto params = runner.initXQAParams();
    params.qInputPtr = thrust::raw_pointer_cast(qInputDevice.data());
    params.kvCache.data = thrust::raw_pointer_cast(kvInputDevice.data());
    params.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthDevice.data());
    params.kvCache.capacity = kvCacheCapacity;
    params.kvCache.tokensPerPage = usePagedKV ? kTOKENS_PER_PAGE : 0U;
    params.kvCache.pageList = usePagedKV ? thrust::raw_pointer_cast(pageTableDevice.data()) : nullptr;
    params.output = thrust::raw_pointer_cast(outDevice.data());
    params.attentionSinks
        = attentionSinks.has_value() ? thrust::raw_pointer_cast(attentionSinksDevice.data()) : nullptr;
    params.attentionScale = resolvedAttentionScale;
    params.slidingWinSize = slidingWindowSize > 0 ? static_cast<uint32_t>(slidingWindowSize) : 0U;

    // Use default stream .
    cudaStream_t stream{nullptr};
    dispatch(runner, params);
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
        // Positions past kvCacheLengths[b] are zero, so scanning and quantizing only the live prefix of each head
        // yields the same amax and the same FP8 buffer as scanning the full capacity.
        for (int32_t b = 0; b < batchSize; ++b)
        {
            size_t const batchBase = static_cast<size_t>(b) * 2 * kvStrideHalf;
            size_t const vBase = batchBase + kvStrideHalf;
            size_t const liveElements = static_cast<size_t>(kvCacheLengths[b]) * headSize;
            for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
            {
                size_t const headOffset = static_cast<size_t>(hkv) * kvCacheCapacity * headSize;
                for (size_t idx = 0; idx < liveElements; ++idx)
                {
                    kAmax = std::max(kAmax, std::fabs(__half2float(kvInput[batchBase + headOffset + idx])));
                    vAmax = std::max(vAmax, std::fabs(__half2float(kvInput[vBase + headOffset + idx])));
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
        for (int32_t b = 0; b < batchSize; ++b)
        {
            size_t const batchBase = static_cast<size_t>(b) * 2 * kvStrideHalf;
            size_t const vBase = batchBase + kvStrideHalf;
            size_t const liveElements = static_cast<size_t>(kvCacheLengths[b]) * headSize;
            for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
            {
                size_t const headOffset = static_cast<size_t>(hkv) * kvCacheCapacity * headSize;
                for (size_t idx = 0; idx < liveElements; ++idx)
                {
                    size_t const k = batchBase + headOffset + idx;
                    size_t const v = vBase + headOffset + idx;
                    kvInputFp8[k] = __nv_fp8_e4m3(__half2float(kvInput[k]) * kScaleOrigQuant);
                    kvInputFp8[v] = __nv_fp8_e4m3(__half2float(kvInput[v]) * vScaleOrigQuant);
                }
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
                numKVHeads, headSize, resolvedAttentionScale, std::nullopt, kScaleQuantOrig, vScaleQuantOrig, 0,
                /*contiguousQuerySwa=*/false, attentionSinks);
            outReferenceFp8.insert(outReferenceFp8.end(), ref.begin(), ref.end());
        }

        auto const kernelKVFp8 = usePagedKV ? packKV(kvInputFp8) : kvInputFp8;
        thrust::device_vector<__nv_fp8_e4m3> kvInputFp8Device(kernelKVFp8);
        thrust::device_vector<half> outFp8Device(batchSize * numQHeads * headSize, __float2half(0.0F));
        EXPECT_TRUE(trt_edgellm::canCompileXQAKernel(
            numQHeads, numKVHeads, headSize, smVersion, DataType::kHALF, DataType::kFP8));
        ASSERT_TRUE(trt_edgellm::loadXQAJitKernelForTest(smVersion, DataType::kHALF, DataType::kFP8, headSize,
            numQHeads, numKVHeads, slidingWindowSize > 0, false, usePagedKV ? kTOKENS_PER_PAGE : 0));
        trt_edgellm::DecoderXQARunner runnerFp8(
            DataType::kHALF, DataType::kFP8, batchSize, numQHeads, numKVHeads, headSize, smVersion);
        auto paramsFp8 = runnerFp8.initXQAParams();
        paramsFp8.qInputPtr = thrust::raw_pointer_cast(qInputDevice.data());
        paramsFp8.kvCache.data = thrust::raw_pointer_cast(kvInputFp8Device.data());
        paramsFp8.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthDevice.data());
        paramsFp8.kvCache.capacity = kvCacheCapacity;
        paramsFp8.kvCache.tokensPerPage = usePagedKV ? kTOKENS_PER_PAGE : 0U;
        paramsFp8.kvCache.pageList = usePagedKV ? thrust::raw_pointer_cast(pageTableDevice.data()) : nullptr;
        paramsFp8.output = thrust::raw_pointer_cast(outFp8Device.data());
        paramsFp8.attentionSinks
            = attentionSinks.has_value() ? thrust::raw_pointer_cast(attentionSinksDevice.data()) : nullptr;
        paramsFp8.attentionScale = resolvedAttentionScale;
        paramsFp8.kScale = kScaleQuantOrig;
        paramsFp8.vScale = vScaleQuantOrig;
        paramsFp8.slidingWinSize = slidingWindowSize > 0 ? static_cast<uint32_t>(slidingWindowSize) : 0U;

        // Reuse the same stream used for FP16 decoding.
        cudaStream_t stream{nullptr};
        dispatch(runnerFp8, paramsFp8);
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
            if (!std::isfinite(__half2float(outFp8Host[i])))
            {
                NanValueDetectedFp8 = true;
            }
            EXPECT_TRUE(std::isfinite(__half2float(outFp8Host[i])));
        }
        float const matchRate = static_cast<float>(numClose) / static_cast<float>(outHost.size());
        std::cout << "XQA Attention Decoding test. [FP8 KV cache] batch_size: " << batchSize
                  << " num_Q_heads: " << numQHeads << " num_KV_heads: " << numKVHeads << " head_size: " << headSize
                  << " sliding_window: " << slidingWindowSize << " kvcache lengths: " << kvCacheLengths
                  << " attention_scale: " << resolvedAttentionScale << " pass_rate_1e-3: " << matchRate << std::endl;
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

TEST(XQAAttentionDecodingTest, accuracyKVRatio16HeadDim256)
{
    TestXQAAttentionDecodingAccuracy(1, 16, 1, 256, 1024);
    TestXQAAttentionDecodingAccuracy(2, 16, 1, 256, 512);
    TestXQAAttentionDecodingAccuracy(1, 32, 2, 256, 512, false, 0, std::nullopt, 274);
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

TEST(XQAAttentionDecodingTest, attentionSinkAccuracy)
{
    constexpr int32_t kNUM_Q_HEADS = 32;
    std::vector<float> attentionSinks(kNUM_Q_HEADS);
    for (int32_t headIdx = 0; headIdx < kNUM_Q_HEADS; ++headIdx)
    {
        attentionSinks[headIdx] = static_cast<float>(headIdx - kNUM_Q_HEADS / 2) * 0.125F;
    }
    TestXQAAttentionDecodingAccuracy(1, kNUM_Q_HEADS, 2, 128, 256, false, 0, std::nullopt, 129, attentionSinks);
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

TEST(XQAAttentionDecodingTest, multiBlockGraphReplay)
{
    // Given different head sizes and live lengths in a large cache, when split decode is replayed in a graph,
    // then FP16/FP8 output matches the reference and workspace guards remain intact.
    for (int32_t const headSize : {64, 128, 256, 512})
    {
        SCOPED_TRACE(headSize);
        for (int32_t const length : {1, 255, 256, 257, 512, 513, 768, 769, 1024, 1025, 2000, 8000, 16000})
        {
            SCOPED_TRACE(length);
            TestXQAAttentionDecodingAccuracy(
                1, 16, 2, headSize, 16384, true, 0, std::nullopt, length, std::nullopt, true, true);
        }
    }
}

TEST(XQAAttentionDecodingTest, multiBlockMixedBatchAndSinks)
{
    // Given heterogeneous batch lengths and attention sinks, when split partials are merged,
    // then the sink is counted once and each request retains its own normalization.
    for (int32_t const headSize : {64, 128, 256, 512})
    {
        SCOPED_TRACE(headSize);
        std::vector<float> sinks(16, 8.F);
        TestXQAAttentionDecodingAccuracy(3, 16, 2, headSize, 16384, true, 0, std::nullopt, 0, sinks, true, true);
        TestXQAAttentionDecodingAccuracy(1, 16, 2, headSize, 16384, true, 0, std::nullopt, 100, sinks, true);
        TestXQAAttentionDecodingAccuracy(1, 16, 2, headSize, 16384, true, 0, std::nullopt, 16000, sinks, true);
        if (headSize != 64)
        {
            sinks.resize(32, 8.F);
            TestXQAAttentionDecodingAccuracy(1, 32, 2, headSize, 16384, true, 0, std::nullopt, 2000, sinks, true, true);
        }
    }
}

TEST(XQAAttentionDecodingTest, multiBlockSlidingWindowGraphReplay)
{
    // Given a long cache and different window widths, when single-token SWA is replayed in a graph,
    // then only multi-tile windows split and FP16/FP8 results and workspace guards remain valid.
    int32_t const smVersion = getSMVersion();
    for (int32_t const headSize : {64, 128, 256, 512})
    {
        SCOPED_TRACE(headSize);
        int32_t const tileSize = headSize == 512 && smVersion != 120 && smVersion != 121 ? 512 : 256;
        for (int32_t const window : {1, tileSize - 1, tileSize, tileSize + 1, 1024, 4096})
        {
            SCOPED_TRACE(window);
            TestXQAAttentionDecodingAccuracy(1, 16, 2, headSize, 16384, true, window, std::nullopt, 16000, std::nullopt,
                true, true, window > tileSize);
        }
    }
}

TEST(XQAAttentionDecodingTest, multiBlockSlidingWindowMixedBatchAndSinks)
{
    // Given mixed live lengths and attention sinks, when single-token SWA uses KV splits,
    // then each request retains its window and counts its sink once.
    std::vector<float> const sinks(16, 8.F);
    for (int32_t const headSize : {64, 128, 256, 512})
    {
        SCOPED_TRACE(headSize);
        TestXQAAttentionDecodingAccuracy(3, 16, 2, headSize, 16384, true, 1024, std::nullopt, 0, sinks, true, true);
        TestXQAAttentionDecodingAccuracy(1, 16, 2, headSize, 16384, true, 4096, std::nullopt, 16000, sinks, true, true);
    }
}
