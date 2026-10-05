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

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "profiling/nvtx_wrapper.h"
#include "references.h"
#include "testUtils.h"
#include "xqaJitTestUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm;

namespace
{

constexpr int32_t kGRAPH_REPLAY_COUNT{8};

struct PdlGraphResult
{
    std::array<std::vector<half>, 2> outputs;
    float averageMs{0.0F};
};

void initializeDeterministicHalf(std::vector<half>& values, int32_t offset)
{
    constexpr int32_t kPERIOD{61};
    constexpr float kSCALE{1.0F / 32.0F};
    for (size_t i = 0; i < values.size(); ++i)
    {
        int32_t const value = (static_cast<int32_t>(i) * 17 + offset * 13) % kPERIOD - kPERIOD / 2;
        values[i] = __float2half(static_cast<float>(value) * kSCALE);
    }
}

std::vector<half> copyToHost(thrust::device_vector<half> const& device)
{
    thrust::host_vector<half> const host(device);
    return std::vector<half>(host.begin(), host.end());
}

template <typename Dispatch>
void captureAndReplayPair(size_t outputSize, half const* input, bool enablePdl, cudaStream_t stream,
    Dispatch&& dispatch, PdlGraphResult& result)
{
    thrust::device_vector<half> first(outputSize, __float2half(0.0F));
    thrust::device_vector<half> second(outputSize, __float2half(0.0F));
    thrust::device_vector<half> firstSnapshots(outputSize * kGRAPH_REPLAY_COUNT);
    thrust::device_vector<half> secondSnapshots(outputSize * kGRAPH_REPLAY_COUNT);
    half* const firstPtr = thrust::raw_pointer_cast(first.data());
    half* const secondPtr = thrust::raw_pointer_cast(second.data());
    half* const firstSnapshotsPtr = thrust::raw_pointer_cast(firstSnapshots.data());
    half* const secondSnapshotsPtr = thrust::raw_pointer_cast(secondSnapshots.data());

    dispatch(input, firstPtr, enablePdl);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph{};
    bool isCapturing{false};
    Defer finishCapture{[&]() {
        if (isCapturing)
        {
            cudaGraph_t abandonedGraph{};
            if (cudaStreamEndCapture(stream, &abandonedGraph) == cudaSuccess && abandonedGraph != nullptr)
            {
                static_cast<void>(cudaGraphDestroy(abandonedGraph));
            }
        }
    }};
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    isCapturing = true;
    dispatch(input, firstPtr, enablePdl);
    dispatch(firstPtr, secondPtr, enablePdl);
    cudaError_t const endCaptureStatus = cudaStreamEndCapture(stream, &graph);
    isCapturing = false;
    CUDA_CHECK(endCaptureStatus);
    ASSERT_NE(graph, nullptr);
    Defer destroyGraph{[&]() { static_cast<void>(cudaGraphDestroy(graph)); }};

    cudaGraphExec_t graphExec{};
    CUDA_CHECK(instantiateCudaGraph(&graphExec, graph));
    Defer destroyGraphExec{[&]() { static_cast<void>(cudaGraphExecDestroy(graphExec)); }};

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    Defer destroyStart{[&]() { static_cast<void>(cudaEventDestroy(start)); }};
    CUDA_CHECK(cudaEventCreate(&stop));
    Defer destroyStop{[&]() { static_cast<void>(cudaEventDestroy(stop)); }};

    NVTX_SCOPED_RANGE(pdlRange, enablePdl ? "xqa_pdl_on" : "xqa_pdl_off");
    float totalMs{0.0F};
    for (int32_t replay = 0; replay < kGRAPH_REPLAY_COUNT; ++replay)
    {
        CUDA_CHECK(cudaMemsetAsync(firstPtr, 0, outputSize * sizeof(half), stream));
        CUDA_CHECK(cudaMemsetAsync(secondPtr, 0, outputSize * sizeof(half), stream));
        CUDA_CHECK(cudaEventRecord(start, stream));
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsedMs{0.0F};
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
        totalMs += elapsedMs;
        size_t const snapshotOffset = static_cast<size_t>(replay) * outputSize;
        CUDA_CHECK(cudaMemcpyAsync(
            firstSnapshotsPtr + snapshotOffset, firstPtr, outputSize * sizeof(half), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(secondSnapshotsPtr + snapshotOffset, secondPtr, outputSize * sizeof(half),
            cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    result.averageMs = totalMs / static_cast<float>(kGRAPH_REPLAY_COUNT);
    result.outputs[0] = copyToHost(firstSnapshots);
    result.outputs[1] = copyToHost(secondSnapshots);
}

void expectPdlMatchesReference(std::string const& label, PdlGraphResult const& pdlOff, PdlGraphResult const& pdlOn,
    std::array<std::vector<half>, 2> const& references)
{
    for (size_t output = 0; output < pdlOff.outputs.size(); ++output)
    {
        size_t const outputSize = references[output].size();
        ASSERT_EQ(pdlOff.outputs[output].size(), outputSize * kGRAPH_REPLAY_COUNT);
        ASSERT_EQ(pdlOn.outputs[output].size(), outputSize * kGRAPH_REPLAY_COUNT);
        for (int32_t replay = 0; replay < kGRAPH_REPLAY_COUNT; ++replay)
        {
            size_t const replayOffset = static_cast<size_t>(replay) * outputSize;
            for (size_t i = 0; i < outputSize; ++i)
            {
                size_t const resultIndex = replayOffset + i;
                EXPECT_TRUE(isclose(pdlOff.outputs[output][resultIndex], references[output][i], 1e-2F, 1e-2F))
                    << label << " PDL-off replay=" << replay << " output=" << output << " index=" << i;
                EXPECT_TRUE(isclose(pdlOn.outputs[output][resultIndex], references[output][i], 1e-2F, 1e-2F))
                    << label << " PDL-on replay=" << replay << " output=" << output << " index=" << i;
                EXPECT_EQ(__half_as_ushort(pdlOff.outputs[output][resultIndex]),
                    __half_as_ushort(pdlOn.outputs[output][resultIndex]))
                    << label << " PDL on/off mismatch at replay=" << replay << " output=" << output << " index=" << i;
            }
        }
    }
    std::cout << label << " graph_pair_ms pdl_off=" << pdlOff.averageMs << " pdl_on=" << pdlOn.averageMs
              << " delta_pct=" << (pdlOn.averageMs / pdlOff.averageMs - 1.0F) * 100.0F << std::endl;
}

void runDecodePdlTest(int32_t headSize, int32_t slidingWindowSize)
{
    constexpr int32_t kBATCH_SIZE{1};
    constexpr int32_t kNUM_Q_HEADS{8};
    constexpr int32_t kNUM_KV_HEADS{1};
    constexpr int32_t kKV_CACHE_CAPACITY{128};
    constexpr int32_t kQ_SEQUENCE_LENGTH{1};

    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    ASSERT_TRUE(
        canCompileXQAKernel(kNUM_Q_HEADS, kNUM_KV_HEADS, headSize, smVersion, DataType::kHALF, DataType::kHALF));
    ASSERT_TRUE(loadXQAJitKernelForTest(smVersion, DataType::kHALF, DataType::kHALF, headSize, kNUM_Q_HEADS,
        kNUM_KV_HEADS, slidingWindowSize > 0, /*specDecode=*/false));

    std::vector<half> qInput(kNUM_Q_HEADS * headSize);
    std::vector<half> kInput(kNUM_KV_HEADS * kKV_CACHE_CAPACITY * headSize);
    std::vector<half> vInput(kNUM_KV_HEADS * kKV_CACHE_CAPACITY * headSize);
    initializeDeterministicHalf(qInput, 1);
    initializeDeterministicHalf(kInput, 2);
    initializeDeterministicHalf(vInput, 3);

    int32_t const attentionLength
        = slidingWindowSize > 0 ? std::min(kKV_CACHE_CAPACITY, slidingWindowSize) : kKV_CACHE_CAPACITY;
    auto const kReference = sliceKVWindow(kInput, kNUM_KV_HEADS, headSize, kKV_CACHE_CAPACITY, slidingWindowSize);
    auto const vReference = sliceKVWindow(vInput, kNUM_KV_HEADS, headSize, kKV_CACHE_CAPACITY, slidingWindowSize);
    float const attentionScale = 1.0F / std::sqrt(static_cast<float>(headSize));
    std::array<std::vector<half>, 2> references;
    references[0] = casualAttentionRef<half>(qInput, kReference, vReference, kQ_SEQUENCE_LENGTH, attentionLength,
        kNUM_Q_HEADS, kNUM_KV_HEADS, headSize, attentionScale);
    references[1] = casualAttentionRef<half>(references[0], kReference, vReference, kQ_SEQUENCE_LENGTH, attentionLength,
        kNUM_Q_HEADS, kNUM_KV_HEADS, headSize, attentionScale);

    std::vector<half> kvInput;
    kvInput.reserve(kInput.size() + vInput.size());
    kvInput.insert(kvInput.end(), kInput.begin(), kInput.end());
    kvInput.insert(kvInput.end(), vInput.begin(), vInput.end());
    std::vector<int32_t> const kvCacheLengths{kKV_CACHE_CAPACITY};

    thrust::device_vector<half> qDevice(qInput);
    thrust::device_vector<half> kvDevice(kvInput);
    thrust::device_vector<int32_t> kvLengthDevice(kvCacheLengths);

    DecoderXQARunner runner(
        DataType::kHALF, DataType::kHALF, kBATCH_SIZE, kNUM_Q_HEADS, kNUM_KV_HEADS, headSize, smVersion);
    XQALaunchParams params = runner.initXQAParams();
    params.qInputPtr = thrust::raw_pointer_cast(qDevice.data());
    params.kvCache.data = thrust::raw_pointer_cast(kvDevice.data());
    params.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvLengthDevice.data());
    params.kvCache.capacity = kKV_CACHE_CAPACITY;
    params.attentionScale = attentionScale;
    params.slidingWinSize = static_cast<uint32_t>(slidingWindowSize);

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    Defer destroyStream{[&]() { static_cast<void>(cudaStreamDestroy(stream)); }};
    auto const dispatch = [&](half const* input, half* output, bool enablePdl) {
        params.qInputPtr = input;
        params.output = output;
        params.enablePdl = enablePdl;
        runner.dispatchXQAKernel(params, stream);
    };

    PdlGraphResult pdlOff;
    PdlGraphResult pdlOn;
    half const* const qInputDevice = thrust::raw_pointer_cast(qDevice.data());
    ASSERT_NO_FATAL_FAILURE(captureAndReplayPair(references[0].size(), qInputDevice, false, stream, dispatch, pdlOff));
    ASSERT_NO_FATAL_FAILURE(captureAndReplayPair(references[0].size(), qInputDevice, true, stream, dispatch, pdlOn));
    expectPdlMatchesReference(
        "decode head_dim=" + std::to_string(headSize) + " sliding=" + std::to_string(slidingWindowSize), pdlOff, pdlOn,
        references);
}

size_t getPagedKVPoolOffset(int32_t pageIdx, int32_t tokensPerPage, int32_t headSize, int32_t tokenInPage, int32_t d)
{
    return (static_cast<size_t>(pageIdx) * tokensPerPage + tokenInPage) * headSize + d;
}

void runSpecDecodePdlTest(bool const usePagedKVCache, int32_t const slidingWindowSize)
{
    constexpr int32_t kBATCH_SIZE{1};
    constexpr int32_t kNUM_Q_HEADS{8};
    constexpr int32_t kNUM_KV_HEADS{1};
    constexpr int32_t kHEAD_SIZE{512};
    constexpr int32_t kKV_SEQUENCE_LENGTH{256};
    constexpr int32_t kQ_SEQUENCE_LENGTH{20};
    constexpr int32_t kTOKENS_PER_PAGE{128};
    constexpr int32_t kMAX_PAGES_PER_SEQUENCE{kKV_SEQUENCE_LENGTH / kTOKENS_PER_PAGE};

    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    ASSERT_TRUE(
        canCompileXQAKernel(kNUM_Q_HEADS, kNUM_KV_HEADS, kHEAD_SIZE, smVersion, DataType::kHALF, DataType::kHALF));
    ASSERT_TRUE(loadXQAJitKernelForTest(smVersion, DataType::kHALF, DataType::kHALF, kHEAD_SIZE, kNUM_Q_HEADS,
        kNUM_KV_HEADS, slidingWindowSize > 0, /*specDecode=*/true, usePagedKVCache ? kTOKENS_PER_PAGE : 0));

    std::vector<half> qInput(kNUM_Q_HEADS * kHEAD_SIZE * kQ_SEQUENCE_LENGTH);
    std::vector<half> kInput(kNUM_KV_HEADS * kKV_SEQUENCE_LENGTH * kHEAD_SIZE);
    std::vector<half> vInput(kNUM_KV_HEADS * kKV_SEQUENCE_LENGTH * kHEAD_SIZE);
    initializeDeterministicHalf(qInput, 4);
    initializeDeterministicHalf(kInput, 5);
    initializeDeterministicHalf(vInput, 6);

    std::vector<int32_t> treeMask(kQ_SEQUENCE_LENGTH * kQ_SEQUENCE_LENGTH);
    for (int32_t row = 0; row < kQ_SEQUENCE_LENGTH; ++row)
    {
        for (int32_t column = 0; column < kQ_SEQUENCE_LENGTH; ++column)
        {
            treeMask[row * kQ_SEQUENCE_LENGTH + column] = column <= row ? 1 : 0;
        }
    }
    std::vector<int32_t> packedTreeMask(kQ_SEQUENCE_LENGTH);
    for (int32_t row = 0; row < kQ_SEQUENCE_LENGTH; ++row)
    {
        int32_t packed{0};
        for (int32_t column = 0; column < kQ_SEQUENCE_LENGTH; ++column)
        {
            packed |= treeMask[row * kQ_SEQUENCE_LENGTH + column] << column;
        }
        packedTreeMask[row] = packed;
    }

    auto const kReference = sliceKVWindow(kInput, kNUM_KV_HEADS, kHEAD_SIZE, kKV_SEQUENCE_LENGTH, slidingWindowSize);
    auto const vReference = sliceKVWindow(vInput, kNUM_KV_HEADS, kHEAD_SIZE, kKV_SEQUENCE_LENGTH, slidingWindowSize);
    int32_t const attentionLength
        = slidingWindowSize > 0 ? std::min(kKV_SEQUENCE_LENGTH, slidingWindowSize) : kKV_SEQUENCE_LENGTH;
    float const attentionScale = 1.0F / std::sqrt(static_cast<float>(kHEAD_SIZE));
    std::array<std::vector<half>, 2> references;
    references[0] = casualAttentionRef<half>(qInput, kReference, vReference, kQ_SEQUENCE_LENGTH, attentionLength,
        kNUM_Q_HEADS, kNUM_KV_HEADS, kHEAD_SIZE, attentionScale, std::make_optional(treeMask));
    references[1] = casualAttentionRef<half>(references[0], kReference, vReference, kQ_SEQUENCE_LENGTH, attentionLength,
        kNUM_Q_HEADS, kNUM_KV_HEADS, kHEAD_SIZE, attentionScale, std::make_optional(treeMask));

    std::vector<int32_t> const pageList{0, 1, 2, 3};
    std::vector<half> kvInput;
    if (usePagedKVCache)
    {
        kvInput.resize(static_cast<size_t>(pageList.size()) * kTOKENS_PER_PAGE * kHEAD_SIZE, __float2half(0.0F));
        for (int32_t token = 0; token < kKV_SEQUENCE_LENGTH; ++token)
        {
            int32_t const page = token / kTOKENS_PER_PAGE;
            int32_t const tokenInPage = token % kTOKENS_PER_PAGE;
            int32_t const kPage = pageList[page];
            int32_t const vPage = pageList[kMAX_PAGES_PER_SEQUENCE + page];
            for (int32_t d = 0; d < kHEAD_SIZE; ++d)
            {
                size_t const compactOffset = static_cast<size_t>(token) * kHEAD_SIZE + d;
                kvInput[getPagedKVPoolOffset(kPage, kTOKENS_PER_PAGE, kHEAD_SIZE, tokenInPage, d)]
                    = kInput[compactOffset];
                kvInput[getPagedKVPoolOffset(vPage, kTOKENS_PER_PAGE, kHEAD_SIZE, tokenInPage, d)]
                    = vInput[compactOffset];
            }
        }
    }
    else
    {
        kvInput.reserve(kInput.size() + vInput.size());
        kvInput.insert(kvInput.end(), kInput.begin(), kInput.end());
        kvInput.insert(kvInput.end(), vInput.begin(), vInput.end());
    }

    std::vector<int32_t> const kvCacheLength{kKV_SEQUENCE_LENGTH};
    thrust::device_vector<half> qDevice(qInput);
    thrust::device_vector<half> kvDevice(kvInput);
    thrust::device_vector<int32_t> kvLengthDevice(kvCacheLength);
    thrust::device_vector<int32_t> pageListDevice(pageList);
    thrust::device_vector<int32_t> treeMaskDevice(packedTreeMask);

    DecoderXQARunner runner(
        DataType::kHALF, DataType::kHALF, kBATCH_SIZE, kNUM_Q_HEADS, kNUM_KV_HEADS, kHEAD_SIZE, smVersion);
    XQALaunchParams params = runner.initXQAParams();
    params.qSeqLen = kQ_SEQUENCE_LENGTH;
    params.qInputPtr = thrust::raw_pointer_cast(qDevice.data());
    params.kvCache.data = thrust::raw_pointer_cast(kvDevice.data());
    params.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvLengthDevice.data());
    params.kvCache.capacity = kKV_SEQUENCE_LENGTH;
    if (usePagedKVCache)
    {
        params.kvCache.pageList = thrust::raw_pointer_cast(pageListDevice.data());
        params.kvCache.tokensPerPage = kTOKENS_PER_PAGE;
    }
    params.treeAttnMask = thrust::raw_pointer_cast(treeMaskDevice.data());
    params.attentionScale = attentionScale;
    params.slidingWinSize = static_cast<uint32_t>(slidingWindowSize);

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    Defer destroyStream{[&]() { static_cast<void>(cudaStreamDestroy(stream)); }};
    auto const dispatch = [&](half const* input, half* output, bool enablePdl) {
        params.qInputPtr = input;
        params.output = output;
        params.enablePdl = enablePdl;
        runner.dispatchSpecDecodeXQAKernel(params, stream);
    };

    PdlGraphResult pdlOff;
    PdlGraphResult pdlOn;
    half const* const qInputDevice = thrust::raw_pointer_cast(qDevice.data());
    ASSERT_NO_FATAL_FAILURE(captureAndReplayPair(references[0].size(), qInputDevice, false, stream, dispatch, pdlOff));
    ASSERT_NO_FATAL_FAILURE(captureAndReplayPair(references[0].size(), qInputDevice, true, stream, dispatch, pdlOn));
    expectPdlMatchesReference(std::string{"spec_decode paged="} + (usePagedKVCache ? "true" : "false")
            + " head_dim=512 sliding=" + std::to_string(slidingWindowSize),
        pdlOff, pdlOn, references);
}

TEST(XQAPdlTest, DecodeAllSupportedHeadDimsGraphReplay)
{
    for (int32_t const headSize : {32, 64, 128, 256, 512})
    {
        SCOPED_TRACE("head_dim=" + std::to_string(headSize));
        runDecodePdlTest(headSize, /*slidingWindowSize=*/0);
    }
}

TEST(XQAPdlTest, DecodeHeadDim512SlidingWindowGraphReplay)
{
    runDecodePdlTest(/*headSize=*/512, /*slidingWindowSize=*/64);
}

TEST(XQAPdlTest, SpecDecodePagedHeadDim512GraphReplay)
{
    runSpecDecodePdlTest(/*usePagedKVCache=*/true, /*slidingWindowSize=*/0);
}

TEST(XQAPdlTest, SpecDecodeHeadDim512SlidingWindowGraphReplay)
{
    runSpecDecodePdlTest(/*usePagedKVCache=*/false, /*slidingWindowSize=*/64);
}

} // namespace
