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

//! Tests for the Gemma4 vision-block FMHA CUSTOM_MASK production path:
//!  1. Bit-exactness of kernel::launchBuildVisionPackedMask against an
//!     independent host derivation of the fmha_v2 packed-mask layout.
//!  2. Parity of the FMHA CUSTOM_MASK prefill against the in-file CPU FP32
//!     semantic oracle on the real Gemma4 Unified 12B sliding-layer shape
//!     (d256, Hq16/Hkv8, W1024, fp16).
//!

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/contextAttentionKernels/contextFMHARunner.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "testUtils.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm;

namespace
{

// ---------------------------------------------------------------------------
// CPU FP32 semantic oracle for vision-block attention (replaced the retired
// GPU fallback kernel):
//   Allowed(q, k) = sliding-causal (k <= q and, for window W > 0, k > q - W)
//                   OR same non-negative vision block
//                   (blockIds[q] >= 0 and blockIds[q] == blockIds[k]).
// Softmax and accumulation are FP32; inputs/outputs are FP16.
// ---------------------------------------------------------------------------
//! Prefill reference.
//! \param q              [B, S, Hq, D] (BSHD).
//! \param kvCache        [B, 2, Hkv, capacity, D] canonical cache (K plane 0, V plane 1).
//! \param blockIds       [B, S]; -1 for text/audio/pad, non-negative per contiguous image run.
//! \param contextLengths [B]; clamped to seqLen.  Rows q >= contextLength are zero-filled.
//! \return               [B, S, Hq, D] FP16 output.
std::vector<half> visionBlockPrefillReference(std::vector<half> const& q, std::vector<half> const& kvCache,
    std::vector<int32_t> const& blockIds, std::vector<int32_t> const& contextLengths, int32_t batchSize, int32_t seqLen,
    int32_t capacity, int32_t numQHeads, int32_t numKVHeads, int32_t headDim, int32_t slidingWindow)
{
    // Convert once to FP32 for speed (the reference is O(S^2 * Hq * D)).
    std::vector<float> qf(q.size());
    std::transform(q.begin(), q.end(), qf.begin(), [](half v) { return __half2float(v); });
    std::vector<float> kvf(kvCache.size());
    std::transform(kvCache.begin(), kvCache.end(), kvf.begin(), [](half v) { return __half2float(v); });

    std::vector<half> reference(q.size(), __float2half(0.0F));
    float const scale = 1.0F / std::sqrt(static_cast<float>(headDim));
    int32_t const groupSize = numQHeads / numKVHeads;

    auto qIdx = [&](int32_t b, int32_t s, int32_t h) {
        return ((static_cast<size_t>(b) * seqLen + s) * numQHeads + h) * headDim;
    };
    auto kvIdx = [&](int32_t b, int32_t kvSel, int32_t h, int32_t s) {
        return (((static_cast<size_t>(b) * 2 + kvSel) * numKVHeads + h) * capacity + s) * headDim;
    };

    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        int32_t const contextLen = std::min(contextLengths[batch], seqLen);
        int32_t const* blocks = blockIds.data() + static_cast<size_t>(batch) * seqLen;

        auto computeRows = [&](int32_t queryBegin, int32_t queryEnd) {
            std::vector<float> logits(contextLen);
            for (int32_t query = queryBegin; query < queryEnd; ++query)
            {
                for (int32_t qHead = 0; qHead < numQHeads; ++qHead)
                {
                    int32_t const kvHead = qHead / groupSize;
                    float const* qRow = qf.data() + qIdx(batch, query, qHead);
                    float maxLogit = -INFINITY;
                    std::fill(logits.begin(), logits.end(), -INFINITY);
                    for (int32_t key = 0; key < contextLen; ++key)
                    {
                        bool const localCausal = key <= query && (slidingWindow <= 0 || key > query - slidingWindow);
                        bool const sameVisionBlock = blocks[query] >= 0 && blocks[query] == blocks[key];
                        if (!localCausal && !sameVisionBlock)
                        {
                            continue;
                        }
                        float const* kRow = kvf.data() + kvIdx(batch, 0, kvHead, key);
                        float dot = 0.0F;
                        for (int32_t d = 0; d < headDim; ++d)
                        {
                            dot += qRow[d] * kRow[d];
                        }
                        logits[key] = dot * scale;
                        maxLogit = std::max(maxLogit, logits[key]);
                    }
                    float denominator = 0.0F;
                    for (float& logit : logits)
                    {
                        logit = std::isfinite(logit) ? std::exp(logit - maxLogit) : 0.0F;
                        denominator += logit;
                    }
                    std::vector<float> value(headDim, 0.0F);
                    for (int32_t key = 0; key < contextLen; ++key)
                    {
                        if (logits[key] == 0.0F)
                        {
                            continue;
                        }
                        float const* vRow = kvf.data() + kvIdx(batch, 1, kvHead, key);
                        for (int32_t d = 0; d < headDim; ++d)
                        {
                            value[d] += logits[key] * vRow[d];
                        }
                    }
                    half* out = reference.data() + qIdx(batch, query, qHead);
                    for (int32_t d = 0; d < headDim; ++d)
                    {
                        out[d] = __float2half(value[d] / denominator);
                    }
                }
            }
        };

        // Parallelize over disjoint query ranges (each thread writes its own rows).
        uint32_t const numThreads = std::max(
            1U, std::min(std::thread::hardware_concurrency(), static_cast<uint32_t>((contextLen + 63) / 64)));
        std::vector<std::thread> workers;
        int32_t const chunk = (contextLen + static_cast<int32_t>(numThreads) - 1) / static_cast<int32_t>(numThreads);
        for (uint32_t t = 0; t < numThreads; ++t)
        {
            int32_t const begin = static_cast<int32_t>(t) * chunk;
            int32_t const end = std::min(contextLen, begin + chunk);
            if (begin < end)
            {
                workers.emplace_back(computeRows, begin, end);
            }
        }
        for (auto& worker : workers)
        {
            worker.join();
        }
    }
    return reference;
}

// ---------------------------------------------------------------------------
// Host reference for the packed-mask layout.
//
// Independent derivation: iterate over every (q, k) position of the dense
// mask and place the bit via the inverse of the fmha_v2 thread mapping
// (mask.h MASK_VERSION 5 / TRT-LLM fmhaPackedMask.cu):
//   64x64 tiles, 128 threads per tile, thread t = warpM * 32 + lane with
//   row = warpM*16 + lane/4 + 8*rowSel, col = (lane%4)*2 + ni*8 + colSel,
//   bit = 4*ni + rowSel*2 + colSel.
// ---------------------------------------------------------------------------

bool visionAllowed(
    std::vector<int32_t> const& blockIds, int32_t batchBase, int32_t ctxLen, int32_t window, int32_t q, int32_t k)
{
    if (q >= ctxLen || k >= ctxLen)
    {
        return false;
    }
    bool const slidingCausal = k <= q && (window <= 0 || k > q - window);
    int32_t const qBlock = blockIds[batchBase + q];
    bool const sameBlock = qBlock >= 0 && qBlock == blockIds[batchBase + k];
    return slidingCausal || sameBlock;
}

std::vector<uint32_t> buildPackedMaskHostReference(std::vector<int32_t> const& blockIds,
    std::vector<int32_t> const& ctxLens, int32_t batchSize, int32_t seqLen, int32_t window)
{
    int64_t const rowsPerSeq = kernel::getPackedMaskRowsPerSeq(seqLen);
    int64_t const strideBytes = kernel::getPackedMaskRowStrideInBytes(seqLen);
    int64_t const mmasM = rowsPerSeq / 64;
    int64_t const mmasN = strideBytes * 8 / 64;
    std::vector<uint32_t> packed(kernel::getPackedMaskSizeInWords(batchSize, seqLen, seqLen), 0U);

    for (int32_t b = 0; b < batchSize; ++b)
    {
        int32_t const ctxLen = std::min(ctxLens[b], seqLen);
        for (int32_t q = 0; q < ctxLen; ++q)
        {
            for (int32_t k = 0; k < ctxLen; ++k)
            {
                if (!visionAllowed(blockIds, b * seqLen, ctxLen, window, q, k))
                {
                    continue;
                }
                // Inverse thread mapping for position (q, k).
                int64_t const mi = q / 64;
                int32_t const r = q % 64;
                int32_t const warpM = r / 16;
                int32_t const rowSel = (r % 16) / 8;
                int32_t const laneDiv4 = r % 8;
                int64_t const nIdx = k / 64;
                int32_t const c = k % 64;
                int32_t const ni = c / 8;
                int32_t const lanePart = (c % 8) / 2;
                int32_t const colSel = c % 2;
                int32_t const lane = laneDiv4 * 4 + lanePart;
                int32_t const tidx = warpM * 32 + lane;
                int32_t const bit = 4 * ni + rowSel * 2 + colSel;
                int64_t const word = (((static_cast<int64_t>(b) * mmasM + mi) * mmasN) + nIdx) * 128 + tidx;
                packed[word] |= 1U << bit;
            }
        }
    }
    return packed;
}

std::vector<uint32_t> buildPackedMaskOnDevice(std::vector<int32_t> const& blockIds, std::vector<int32_t> const& ctxLens,
    int32_t batchSize, int32_t seqLen, int32_t window, std::vector<int32_t>* cuMaskRowsOut = nullptr)
{
    int64_t const words = kernel::getPackedMaskSizeInWords(batchSize, seqLen, seqLen);
    rt::Tensor idsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor lengthsTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor packedTensor({words}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuMaskRowsTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(idsTensor, blockIds);
    copyHostToDevice(lengthsTensor, ctxLens);
    // Poison the output so untouched (padding) words would be caught.
    CUDA_CHECK(cudaMemset(packedTensor.rawPointer(), 0xFF, words * sizeof(uint32_t)));

    cudaStream_t stream{nullptr};
    kernel::launchBuildVisionPackedMask(idsTensor.dataPointer<int32_t>(), lengthsTensor.dataPointer<int32_t>(),
        reinterpret_cast<uint32_t*>(packedTensor.dataPointer<int32_t>()), cuMaskRowsTensor.dataPointer<int32_t>(),
        batchSize, seqLen, window, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (cuMaskRowsOut != nullptr)
    {
        auto const rows = copyDeviceToHost<int32_t>(cuMaskRowsTensor);
        cuMaskRowsOut->assign(rows.begin(), rows.end());
    }
    auto const packedI32 = copyDeviceToHost<int32_t>(packedTensor);
    return std::vector<uint32_t>(reinterpret_cast<uint32_t const*>(packedI32.data()),
        reinterpret_cast<uint32_t const*>(packedI32.data()) + words);
}

void expectPackedMaskBitExact(std::vector<int32_t> const& blockIds, std::vector<int32_t> const& ctxLens,
    int32_t batchSize, int32_t seqLen, int32_t window)
{
    std::vector<int32_t> cuMaskRows;
    auto const actual = buildPackedMaskOnDevice(blockIds, ctxLens, batchSize, seqLen, window, &cuMaskRows);
    auto const expected = buildPackedMaskHostReference(blockIds, ctxLens, batchSize, seqLen, window);
    ASSERT_EQ(actual.size(), expected.size());

    int64_t const rowsPerSeq = kernel::getPackedMaskRowsPerSeq(seqLen);
    for (int32_t b = 0; b <= batchSize; ++b)
    {
        EXPECT_EQ(cuMaskRows[b], b * rowsPerSeq) << "cuMaskRows mismatch at batch " << b;
    }
    for (size_t i = 0; i < actual.size(); ++i)
    {
        ASSERT_EQ(actual[i], expected[i])
            << "Packed mask word mismatch at index " << i << " (S=" << seqLen << ", W=" << window << "): actual=0x"
            << std::hex << actual[i] << " expected=0x" << expected[i] << std::dec;
    }
}

// ---------------------------------------------------------------------------
// Mask builder bit-exactness.
// ---------------------------------------------------------------------------

TEST(VisionPackedMaskBuilderTest, MultiBlockBatchedWithBlockAtSequenceEndAndShortContext)
{
    // Batch 0: block 0 at the very start, block 1 at the sequence end; full
    // context.  Batch 1: contextLength < S with a block clipped by the
    // context length; padding rows/cols must be zero.
    int32_t constexpr batchSize = 2;
    int32_t constexpr seqLen = 300;
    std::vector<int32_t> blockIds(batchSize * seqLen, -1);
    for (int32_t i = 0; i < 20; ++i)
    {
        blockIds[i] = 0;
    }
    for (int32_t i = 270; i < 300; ++i)
    {
        blockIds[i] = 1;
    }
    for (int32_t i = 150; i < 220; ++i)
    {
        blockIds[seqLen + i] = 0;
    }
    expectPackedMaskBitExact(blockIds, {seqLen, 200}, batchSize, seqLen, /*window=*/64);
}

TEST(VisionPackedMaskBuilderTest, LongSequenceAboveWindowPlainCausalAndBlocks)
{
    // S > W exercises masked-out keys far below the causal diagonal.  Also
    // covers window <= 0 (plain causal) in a second pass.
    int32_t constexpr seqLen = 1500;
    std::vector<int32_t> blockIds(seqLen, -1);
    for (int32_t i = 200; i < 480; ++i)
    {
        blockIds[i] = 3;
    }
    expectPackedMaskBitExact(blockIds, {seqLen}, 1, seqLen, /*window=*/1024);
    expectPackedMaskBitExact(blockIds, {seqLen}, 1, seqLen, /*window=*/-1);
}

// ---------------------------------------------------------------------------
// FMHA CUSTOM_MASK parity vs the CPU FP32 vision-block reference oracle.
// Real Gemma4 Unified 12B sliding-layer shape: d256, Hq16/Hkv8, W1024, fp16.
// ---------------------------------------------------------------------------

struct VisionFMHAParityCase
{
    std::string name;
    int32_t seqLen;
    int32_t window;
    //! Pairs of [begin, end) vision-block intervals.
    std::vector<std::pair<int32_t, int32_t>> blocks;
    //! Compare the CUSTOM_MASK output against the plain sliding FMHA kernel
    //! as well (text-only cases: the two kernels must agree tightly).
    bool compareAgainstSlidingFMHA{false};
};

class VisionFMHACustomMaskParityTest : public ::testing::TestWithParam<VisionFMHAParityCase>
{
};

TEST_P(VisionFMHACustomMaskParityTest, MatchesVisionBlockReferenceOracle)
{
    auto const& testCase = GetParam();
    int32_t constexpr numQHeads = 16;
    int32_t constexpr numKVHeads = 8;
    int32_t constexpr headDim = 256;
    int32_t const seqLen = testCase.seqLen;
    int32_t const window = testCase.window;

    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);

    AttentionInputLayout constexpr inputLayout = AttentionInputLayout::SEPARATE_Q_K_V;
    ASSERT_TRUE(ContextFMHARunner::canImplement(
        headDim, smVersion, DataType::kHALF, inputLayout, ContextAttentionMaskType::CUSTOM_MASK))
        << "FMHA CUSTOM_MASK cubins missing from the metadata table for headSize=" << headDim << ", SM=" << smVersion;
    // Establish the CUDA primary context before the driver-API cubin loads
    // below (needed when this test is the first CUDA user in the process).
    CUDA_CHECK(cudaFree(nullptr));
    ASSERT_TRUE(ContextFMHARunner::loadContextFMHAKernels(
        smVersion, DataType::kHALF, ContextAttentionMaskType::CUSTOM_MASK));
    ContextFMHARunner runner(DataType::kHALF, /*batchSize=*/1, seqLen, numQHeads, numKVHeads, headDim, smVersion,
        inputLayout, ContextAttentionMaskType::CUSTOM_MASK);
    ASSERT_TRUE(runner.isKernelAvailable())
        << "FMHA CUSTOM_MASK kernel unavailable for S=" << seqLen << " on SM" << smVersion;

    // Vision block ids.
    std::vector<int32_t> blockIds(seqLen, -1);
    for (size_t blockIdx = 0; blockIdx < testCase.blocks.size(); ++blockIdx)
    {
        auto const [begin, end] = testCase.blocks[blockIdx];
        ASSERT_LE(end, seqLen);
        for (int32_t i = begin; i < end; ++i)
        {
            blockIds[i] = static_cast<int32_t>(blockIdx);
        }
    }

    // Inputs: Q [1, S, Hq, D]; K/V both as separate [1, S, Hkv, D] (FMHA) and
    // as the canonical KV cache [1, 2, Hkv, S, D] (CPU reference oracle).
    std::vector<half> q(static_cast<size_t>(seqLen) * numQHeads * headDim);
    std::vector<half> k(static_cast<size_t>(seqLen) * numKVHeads * headDim);
    std::vector<half> v(static_cast<size_t>(seqLen) * numKVHeads * headDim);
    uniformFloatInitialization(q, -0.5F, 0.5F);
    uniformFloatInitialization(k, -0.5F, 0.5F);
    uniformFloatInitialization(v, -0.5F, 0.5F);

    std::vector<half> kvCache(static_cast<size_t>(2) * numKVHeads * seqLen * headDim);
    for (int32_t s = 0; s < seqLen; ++s)
    {
        for (int32_t h = 0; h < numKVHeads; ++h)
        {
            for (int32_t d = 0; d < headDim; ++d)
            {
                size_t const srcIdx = (static_cast<size_t>(s) * numKVHeads + h) * headDim + d;
                size_t const kIdx = (static_cast<size_t>(h) * seqLen + s) * headDim + d;
                size_t const vIdx = ((static_cast<size_t>(numKVHeads) + h) * seqLen + s) * headDim + d;
                kvCache[kIdx] = k[srcIdx];
                kvCache[vIdx] = v[srcIdx];
            }
        }
    }

    rt::Tensor qTensor({1, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kTensor({1, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({1, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kvCacheTensor({1, 2, numKVHeads, seqLen, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor blockIdsTensor({1, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor lengthsTensor({1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor fmhaOutTensor({1, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(qTensor, q);
    copyHostToDevice(kTensor, k);
    copyHostToDevice(vTensor, v);
    copyHostToDevice(kvCacheTensor, kvCache);
    copyHostToDevice(blockIdsTensor, blockIds);
    copyHostToDevice(lengthsTensor, std::vector<int32_t>{seqLen});

    cudaStream_t stream{nullptr};

    // Oracle: the CPU FP32 vision-block attention reference.
    auto const expected = visionBlockPrefillReference(q, kvCache, blockIds, std::vector<int32_t>{seqLen},
        /*batchSize=*/1, seqLen, /*capacity=*/seqLen, numQHeads, numKVHeads, headDim, window);

    // Packed mask (sliding window folded into the bits).
    int64_t const packedMaskWords = kernel::getPackedMaskSizeInWords(1, seqLen, seqLen);
    rt::Tensor packedMaskTensor({packedMaskWords}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuMaskRowsTensor({2}, rt::DeviceType::kGPU, DataType::kINT32);
    kernel::launchBuildVisionPackedMask(blockIdsTensor.dataPointer<int32_t>(), lengthsTensor.dataPointer<int32_t>(),
        reinterpret_cast<uint32_t*>(packedMaskTensor.dataPointer<int32_t>()), cuMaskRowsTensor.dataPointer<int32_t>(),
        1, seqLen, window, stream);

    rt::Tensor cuSeqLensTensor({2}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(cuSeqLensTensor, std::vector<int32_t>{0, seqLen});

    FusedMultiheadAttentionParamsV2 params{};
    runner.setupParams(params, 1.0F / std::sqrt(static_cast<float>(headDim)));
    params.s_kv = seqLen;
    params.q_ptr = qTensor.rawPointer();
    params.k_ptr = kTensor.rawPointer();
    params.v_ptr = vTensor.rawPointer();
    params.o_ptr = fmhaOutTensor.rawPointer();
    params.cu_q_seqlens = cuSeqLensTensor.dataPointer<int32_t>();
    params.cu_kv_seqlens = cuSeqLensTensor.dataPointer<int32_t>();
    params.cu_mask_rows = cuMaskRowsTensor.dataPointer<int32_t>();
    params.packed_mask_ptr = packedMaskTensor.rawPointer();
    params.packed_mask_stride_in_bytes = kernel::getPackedMaskRowStrideInBytes(seqLen);
    runner.dispatchFMHAKernel(params, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const actual = copyDeviceToHost<half>(fmhaOutTensor);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
    {
        ASSERT_TRUE(isclose(actual[i], expected[i], 2e-2F, 2e-2F))
            << testCase.name << ": FMHA/reference mismatch at " << i << " (q=" << i / (numQHeads * headDim)
            << "): fmha=" << __half2float(actual[i]) << " reference=" << __half2float(expected[i]);
    }

    // Text-only cases: the CUSTOM_MASK kernel must agree tightly with the
    // production sliding-window FMHA kernel (same kernel family, mask via
    // bits vs analytic) — this pins the packed mask to the exact sliding
    // semantics.
    if (testCase.compareAgainstSlidingFMHA)
    {
        ASSERT_TRUE(ContextFMHARunner::canImplement(
            headDim, smVersion, DataType::kHALF, inputLayout, ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL));
        ContextFMHARunner slidingRunner(DataType::kHALF, 1, seqLen, numQHeads, numKVHeads, headDim, smVersion,
            inputLayout, ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL);
        rt::Tensor slidingOutTensor({1, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);

        FusedMultiheadAttentionParamsV2 slidingParams{};
        slidingRunner.setupParams(slidingParams, 1.0F / std::sqrt(static_cast<float>(headDim)));
        slidingParams.s_kv = seqLen;
        slidingParams.sliding_window_size = window;
        slidingParams.q_ptr = qTensor.rawPointer();
        slidingParams.k_ptr = kTensor.rawPointer();
        slidingParams.v_ptr = vTensor.rawPointer();
        slidingParams.o_ptr = slidingOutTensor.rawPointer();
        slidingParams.cu_q_seqlens = cuSeqLensTensor.dataPointer<int32_t>();
        slidingParams.cu_kv_seqlens = cuSeqLensTensor.dataPointer<int32_t>();
        slidingRunner.dispatchFMHAKernel(slidingParams, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto const slidingOut = copyDeviceToHost<half>(slidingOutTensor);
        int64_t bitExactMismatches = 0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            ASSERT_TRUE(isclose(actual[i], slidingOut[i], 1e-3F, 1e-3F))
                << testCase.name << ": CUSTOM_MASK/sliding FMHA mismatch at " << i
                << ": custom=" << __half2float(actual[i]) << " sliding=" << __half2float(slidingOut[i]);
            uint16_t actualBits{};
            uint16_t slidingBits{};
            std::memcpy(&actualBits, &actual[i], sizeof(actualBits));
            std::memcpy(&slidingBits, &slidingOut[i], sizeof(slidingBits));
            if (actualBits != slidingBits)
            {
                ++bitExactMismatches;
            }
        }
        std::cout << testCase.name << ": CUSTOM_MASK vs sliding FMHA bit-exact mismatches: " << bitExactMismatches
                  << " / " << actual.size() << std::endl;
    }
}

INSTANTIATE_TEST_SUITE_P(Gemma4UnifiedSlidingLayerD256, VisionFMHACustomMaskParityTest,
    ::testing::Values(
        // Pure sliding, S above the window: exercises masking below the
        // causal diagonal (keys older than the window) — also compared
        // against the plain sliding FMHA kernel.
        VisionFMHAParityCase{"TextOnlyAboveWindow", 1536, 1024, {}, true},
        // Pure sliding, S below the window (degenerates to causal).
        VisionFMHAParityCase{"TextOnlyBelowWindow", 320, 1024, {}, true},
        // One mid-sequence block crossing the 256-col alignment boundary.
        VisionFMHAParityCase{"MidSequenceBlock", 512, 1024, {{200, 320}}, false},
        // Ragged S (not a multiple of any tile size) with the block ending
        // exactly at the sequence end.
        VisionFMHAParityCase{"BlockAtSequenceEndRaggedS", 333, 64, {{300, 333}}, false},
        // Multiple blocks, S above the window, block at the sequence end;
        // rows near the end must attend their block but not stale text keys.
        VisionFMHAParityCase{"MultiBlockLongSequence", 2100, 1024, {{64, 344}, {700, 980}, {1900, 2100}}, false}),
    [](::testing::TestParamInfo<VisionFMHAParityCase> const& info) { return info.param.name; });

} // namespace
