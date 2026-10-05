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
#include "common/tensor.h"
#include "kernels/qsaIndexer/qsaIndexerKernels.h"
#include "kernels/qsaIndexer/qsaIndexerRunner.h"
#include "qsaIndexerReference.h"
#include "testUtils.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kWidth = kernel::kQSA_INDEX_WIDTH;
constexpr int32_t kNumKVHeadsPool = 2;
constexpr int32_t kPoolHeadDim = 384;
constexpr int32_t kHeadSize = 256;
constexpr float kRmsEps = 1e-6f;

//! These kernels are arch-generic; the only runtime requirement is any CUDA device.
bool hasCudaDevice()
{
    int32_t deviceCount = 0;
    return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
}

uint16_t bitsOf(half h)
{
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

//! cos/sin table [maxPositions, 64] matching initializeNormalRopeCosSin with rotaryDim = 64
//! (cos in columns [0, 32), sin in [32, 64)); theta = 1e7 is the qwen4_exp indexer rope base.
std::vector<float> makeCosSinTable(int32_t maxPositions)
{
    std::vector<float> table(static_cast<size_t>(maxPositions) * qsa_ref::kRotaryDim);
    for (int32_t pos = 0; pos < maxPositions; ++pos)
    {
        for (int32_t i = 0; i < qsa_ref::kRotaryDim / 2; ++i)
        {
            double const angle = pos / std::pow(1e7, 2.0 * i / qsa_ref::kRotaryDim);
            table[static_cast<int64_t>(pos) * qsa_ref::kRotaryDim + i] = static_cast<float>(std::cos(angle));
            table[static_cast<int64_t>(pos) * qsa_ref::kRotaryDim + qsa_ref::kRotaryDim / 2 + i]
                = static_cast<float>(std::sin(angle));
        }
    }
    return table;
}

std::vector<half> randomHalves(size_t count, std::mt19937& rng, float lo = -1.0f, float hi = 1.0f)
{
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<half> values(count);
    for (size_t i = 0; i < count; ++i)
    {
        values[i] = __float2half(dist(rng));
    }
    return values;
}

std::vector<float> toFloats(std::vector<half> const& values)
{
    std::vector<float> out(values.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        out[i] = __half2float(values[i]);
    }
    return out;
}

//! NaN-poisoned synthetic paged pool with a permuted non-identity page table (V ids
//! pre-offset +numPages, unused slots -1). Keeps a host mirror for expected-value checks.
struct PagedPoolHarness
{
    int32_t batch;
    int32_t numPages;
    int32_t maxPagesPerSeq;
    std::vector<int32_t> pageTable;
    std::vector<half> poolH; //!< host mirror; upload() pushes it to the device
    rt::Tensor dPool;
    rt::Tensor dPageTable;

    PagedPoolHarness(int32_t batch_, int32_t numPages_, int32_t maxPagesPerSeq_, int32_t pagesUsed, uint32_t seed)
        : batch(batch_)
        , numPages(numPages_)
        , maxPagesPerSeq(maxPagesPerSeq_)
        , pageTable(qsa_ref::makePermutedPageTable(batch_, maxPagesPerSeq_, pagesUsed, numPages_, seed))
        , poolH(static_cast<size_t>(2) * numPages_ * qsa_ref::kTokensPerPage * kNumKVHeadsPool * kPoolHeadDim,
              __float2half(std::numeric_limits<float>::quiet_NaN()))
    {
        dPool = rt::Tensor({2 * numPages, qsa_ref::kTokensPerPage, kNumKVHeadsPool, kPoolHeadDim}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "pool");
        dPageTable
            = rt::Tensor({batch, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "pageTable");
        upload();
        copyHostToDevice(dPageTable, pageTable);
    }

    void upload()
    {
        copyHostToDevice(dPool, poolH);
    }

    std::vector<half> download() const
    {
        return copyDeviceToHost<half>(dPool);
    }

    kernel::QsaIndexerPoolState state()
    {
        return {dPool.rawPointer(), dPageTable.dataPointer<int32_t>(), maxPagesPerSeq, numPages, kNumKVHeadsPool,
            kPoolHeadDim, kHeadSize};
    }

    //! Host element offset of `token`'s tail column 0 on `plane` (0 = K, 1 = V) of seq b.
    int64_t tailOffset(int32_t b, int32_t plane, int32_t token) const
    {
        int32_t const page = qsa_ref::pageOf(pageTable, maxPagesPerSeq, b, plane, token);
        return qsa_ref::poolOffset(page, token % qsa_ref::kTokensPerPage, 0, kHeadSize, kNumKVHeadsPool, kPoolHeadDim);
    }
};

//! Common inputs of one launchQsaIndexerPreDecode step: one indexQk row per sequence plus
//! the shared rope table, lengths, and norm gammas.
struct PreDecodeHarness
{
    int32_t batch;
    std::vector<int32_t> lens;
    std::vector<half> indexQkH;
    std::vector<float> indexQkF;
    std::vector<float> cosSinF;
    std::vector<float> wQF;
    std::vector<float> wKF;
    rt::Tensor dIndexQk;
    rt::Tensor dCosSin;
    rt::Tensor dLens;
    rt::Tensor dWQ;
    rt::Tensor dWK;
    rt::Tensor dQNormed;

    PreDecodeHarness(std::vector<int32_t> lens_, int32_t maxPositions, uint32_t seed)
        : batch(static_cast<int32_t>(lens_.size()))
        , lens(std::move(lens_))
    {
        std::mt19937 rng(seed);
        indexQkH = randomHalves(static_cast<size_t>(batch) * qsa_ref::kIndexQkWidth, rng);
        indexQkF = toFloats(indexQkH);
        std::vector<half> const wQH = randomHalves(qsa_ref::kHeadDim, rng, -0.5f, 0.5f);
        std::vector<half> const wKH = randomHalves(qsa_ref::kHeadDim, rng, -0.5f, 0.5f);
        wQF = toFloats(wQH);
        wKF = toFloats(wKH);
        cosSinF = makeCosSinTable(maxPositions);

        dIndexQk = rt::Tensor(
            {batch, 1, qsa_ref::kIndexQkWidth}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "indexQk");
        dCosSin = rt::Tensor(
            {maxPositions, qsa_ref::kRotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "cosSin");
        dLens = rt::Tensor({batch}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "contextLengths");
        dWQ = rt::Tensor({qsa_ref::kHeadDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "wQ");
        dWK = rt::Tensor({qsa_ref::kHeadDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "wK");
        dQNormed = rt::Tensor({batch, 1, qsa_ref::kNumHeads, qsa_ref::kHeadDim}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "qNormed");
        copyHostToDevice(dIndexQk, indexQkH);
        copyHostToDevice(dCosSin, cosSinF);
        copyHostToDevice(dLens, lens);
        copyHostToDevice(dWQ, wQH);
        copyHostToDevice(dWK, wKH);
    }

    void run(PagedPoolHarness& pool)
    {
        kernel::launchQsaIndexerPreDecode<half>(dQNormed.dataPointer<half>(), dIndexQk.dataPointer<half>(),
            dCosSin.dataPointer<float>(), dLens.dataPointer<int32_t>(), dWQ.dataPointer<half>(),
            dWK.dataPointer<half>(), kRmsEps, pool.state(), batch, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

//! Verify a decode outIdx row's K4 structure — min(512, ctx/4) complete blocks packed to
//! the front in groups of 4, then the causal tail, then -1 pad — and collect the block ids.
void verifyDecodeRowStructure(int32_t const* row, int32_t contextLen, std::vector<int32_t>* selectedBlocks)
{
    int32_t const numVisible = contextLen / qsa_ref::kRatio;
    int32_t const numSelected = std::min(qsa_ref::kTopk, numVisible);
    int32_t const tailCount = contextLen % qsa_ref::kRatio;
    int32_t const expandedCount = numSelected * qsa_ref::kRatio;
    selectedBlocks->clear();
    std::vector<char> seen(std::max(numVisible, 1), 0);
    for (int32_t i = 0; i < numSelected; ++i)
    {
        int32_t const blockId = row[i * qsa_ref::kRatio] / qsa_ref::kRatio;
        ASSERT_GE(blockId, 0) << "group " << i;
        ASSERT_LT(blockId, numVisible) << "group " << i;
        ASSERT_EQ(seen[blockId], 0) << "duplicate block " << blockId;
        seen[blockId] = 1;
        for (int32_t j = 0; j < qsa_ref::kRatio; ++j)
        {
            ASSERT_EQ(row[i * qsa_ref::kRatio + j], blockId * qsa_ref::kRatio + j)
                << "slot " << i * qsa_ref::kRatio + j;
        }
        selectedBlocks->push_back(blockId);
    }
    for (int32_t j = 0; j < tailCount; ++j) // causal tail, ALWAYS appended
    {
        ASSERT_EQ(row[expandedCount + j], numVisible * qsa_ref::kRatio + j) << "tail slot " << j;
    }
    for (int32_t j = expandedCount + tailCount; j < kWidth; ++j)
    {
        ASSERT_EQ(row[j], -1) << "pad slot " << j;
    }
}

//! Sorted copy for order-insensitive set comparison (the emit order is unspecified).
std::vector<int32_t> sorted(std::vector<int32_t> v)
{
    std::sort(v.begin(), v.end());
    return v;
}

} // namespace

TEST(QsaIndexerDecode, PoolStateRejectsZeroHeadSize)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    // headSize == 0 would put the state tail at column 0, on top of the K/V data: every launcher
    // that dereferences the pool state must reject it before touching the device.
    PagedPoolHarness pool(
        /* batch = */ 1, /* numPages = */ 2, /* maxPagesPerSeq = */ 1, /* pagesUsed = */ 1, /* seed = */ 7);
    kernel::QsaIndexerPoolState bad = pool.state();
    bad.headSize = 0;
    rt::Tensor dQk({1, 1, qsa_ref::kIndexQkWidth}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "indexQk");
    rt::Tensor dLens({1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "contextLengths");
    EXPECT_THROW(kernel::launchQsaRawKTailWritePrefill<half>(dQk.dataPointer<half>(), dLens.dataPointer<int32_t>(), bad,
                     /* batchSize = */ 1, /* seqLen = */ 1, nullptr),
        std::runtime_error);
}

TEST(QsaIndexerDecode, PreDecodeQPrepMatchesReference)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    // Positions 0, 5, and 129 (page-crossing); no length is a multiple of 4, so phase 3
    // stays a device-side no-op.
    PreDecodeHarness h({1, 6, 130}, /* maxPositions = */ 256, /* seed = */ 2001);
    PagedPoolHarness pool(h.batch, /* numPages = */ 8, /* maxPagesPerSeq = */ 3, /* pagesUsed = */ 2, /* seed = */ 7);
    h.run(pool);
    std::vector<half> const got = copyDeviceToHost<half>(h.dQNormed);

    for (int32_t b = 0; b < h.batch; ++b)
    {
        std::vector<float> const ref
            = qsa_ref::refQPrepDecodeRow(h.indexQkF.data() + static_cast<int64_t>(b) * qsa_ref::kIndexQkWidth,
                h.cosSinF, h.lens[b] - 1, h.wQF, kRmsEps);
        for (int32_t i = 0; i < qsa_ref::kQRowWidth; ++i)
        {
            float const v = __half2float(got[static_cast<int64_t>(b) * qsa_ref::kQRowWidth + i]);
            ASSERT_TRUE(isclose(v, ref[i], /* rtol = */ 1e-2f, /* atol = */ 1e-3f))
                << "b=" << b << " i=" << i << " got " << v << " want " << ref[i];
        }
    }
}

TEST(QsaIndexerDecode, PreDecodeRawTailWriteBitExactAndNaNSafe)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    PreDecodeHarness h({1, 6, 130}, /* maxPositions = */ 256, /* seed = */ 2002);
    PagedPoolHarness pool(h.batch, /* numPages = */ 8, /* maxPagesPerSeq = */ 3, /* pagesUsed = */ 2, /* seed = */ 11);
    h.run(pool);
    std::vector<half> const gotPool = pool.download();

    // The only written cells are each sequence's K-tail of the new token (no length is a
    // multiple of 4, so no V-tail compress happened); the values are bit-exact copies of
    // indexQk columns [512, 640).
    std::vector<char> written(gotPool.size(), 0);
    for (int32_t b = 0; b < h.batch; ++b)
    {
        int32_t const pos = h.lens[b] - 1;
        int64_t const base = pool.tailOffset(b, /* plane = */ 0, pos);
        for (int32_t d = 0; d < qsa_ref::kHeadDim; ++d)
        {
            ASSERT_EQ(bitsOf(gotPool[base + d]),
                bitsOf(h.indexQkH[static_cast<int64_t>(b) * qsa_ref::kIndexQkWidth + qsa_ref::kQRowWidth + d]))
                << "b=" << b << " d=" << d;
            written[base + d] = 1;
        }
    }
    for (size_t i = 0; i < gotPool.size(); ++i)
    {
        if (written[i] == 0)
        {
            ASSERT_TRUE(std::isnan(__half2float(gotPool[i]))) << "pool cell " << i << " was clobbered";
        }
    }
}

TEST(QsaIndexerDecode, PreDecodeCompressBoundaryBitParityWithPrefill)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    // b0 hits a block boundary (132 % 4 == 0): block 32 = tokens [128, 131] compresses with
    // three raw keys read back from pool K-tails and the newest forwarded in registers.
    // b1 (130 % 4 != 0) must not compress.
    PreDecodeHarness h({132, 130}, /* maxPositions = */ 256, /* seed = */ 2003);
    PagedPoolHarness pool(h.batch, /* numPages = */ 6, /* maxPagesPerSeq = */ 2, /* pagesUsed = */ 2, /* seed = */ 13);

    // Pre-populate b0's K-tails of tokens 128..130 (as previous decode steps would have).
    std::mt19937 rng(99);
    std::array<std::vector<half>, 4> rawKH;
    for (int32_t j = 0; j < 3; ++j)
    {
        rawKH[j] = randomHalves(qsa_ref::kHeadDim, rng);
        int64_t const base = pool.tailOffset(0, /* plane = */ 0, 128 + j);
        std::copy(rawKH[j].begin(), rawKH[j].end(), pool.poolH.begin() + base);
    }
    pool.upload();
    // Token 131's raw key is the new indexQk row's columns [512, 640).
    rawKH[3].assign(h.indexQkH.begin() + qsa_ref::kQRowWidth, h.indexQkH.begin() + qsa_ref::kIndexQkWidth);

    h.run(pool);
    std::vector<half> const gotPool = pool.download();

    // (1) kbar at the V-tail of token 128 matches the decode compress reference.
    std::array<std::vector<float>, 4> rawKF;
    std::array<float const*, 4> rawKPtr{};
    for (int32_t j = 0; j < 4; ++j)
    {
        rawKF[j] = toFloats(rawKH[j]);
        rawKPtr[j] = rawKF[j].data();
    }
    std::vector<float> const refKbar
        = qsa_ref::refKCompressGroup(rawKPtr, h.cosSinF, /* firstToken = */ 128, h.wKF, kRmsEps);
    int64_t const vBase = pool.tailOffset(0, /* plane = */ 1, 128);
    for (int32_t d = 0; d < qsa_ref::kHeadDim; ++d)
    {
        float const v = __half2float(gotPool[vBase + d]);
        ASSERT_TRUE(isclose(v, refKbar[d], /* rtol = */ 1e-2f, /* atol = */ 1e-3f))
            << "d=" << d << " got " << v << " want " << refKbar[d];
    }

    // (2) BIT-PARITY with prefill K1b over the same 4 raw keys at the same rope position
    // (blockBegin = 32, pastLen = 128 places the block at absolute tokens [128, 131]).
    constexpr int32_t kNumBlocks = 33;
    std::vector<half> prefillQk(static_cast<size_t>(4) * qsa_ref::kIndexQkWidth, __float2half(0.0f));
    for (int32_t j = 0; j < 4; ++j)
    {
        std::copy(rawKH[j].begin(), rawKH[j].end(),
            prefillQk.begin() + static_cast<int64_t>(j) * qsa_ref::kIndexQkWidth + qsa_ref::kQRowWidth);
    }
    rt::Tensor dPrefillQk({1, 4, qsa_ref::kIndexQkWidth}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "prefillQk");
    rt::Tensor dPrefillLens({1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "prefillLens");
    rt::Tensor dKbar({1, kNumBlocks, qsa_ref::kHeadDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "kbar");
    copyHostToDevice(dPrefillQk, prefillQk);
    copyHostToDevice(dPrefillLens, std::vector<int32_t>{132});
    kernel::launchQsaIndexKCompress<half>(dKbar.dataPointer<half>(), dPrefillQk.dataPointer<half>(),
        h.dCosSin.dataPointer<float>(), dPrefillLens.dataPointer<int32_t>(), h.dWK.dataPointer<half>(), kRmsEps,
        /* batchSize = */ 1, /* seqLen = */ 4, kNumBlocks, /* blockBegin = */ 32, /* blockEnd = */ 33,
        /* pastLen = */ 128, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> const prefillKbar = copyDeviceToHost<half>(dKbar);
    for (int32_t d = 0; d < qsa_ref::kHeadDim; ++d)
    {
        ASSERT_EQ(bitsOf(gotPool[vBase + d]), bitsOf(prefillKbar[static_cast<int64_t>(32) * qsa_ref::kHeadDim + d]))
            << "decode compress not bit-identical to prefill K1b at d=" << d;
    }

    // (3) The completing token's raw key is consumed from registers, never persisted: b0's
    // K-tail of token 131 stays NaN.
    {
        int64_t const base = pool.tailOffset(0, /* plane = */ 0, 131);
        for (int32_t d = 0; d < qsa_ref::kHeadDim; ++d)
        {
            ASSERT_TRUE(std::isnan(__half2float(gotPool[base + d]))) << "b0 K-tail of token 131 d=" << d;
        }
    }

    // (4) b1 crossed no boundary: its V plane stays NaN.
    for (int32_t g = 0; g < 130 / qsa_ref::kRatio; ++g)
    {
        int64_t const base = pool.tailOffset(1, /* plane = */ 1, g * qsa_ref::kRatio);
        for (int32_t d = 0; d < qsa_ref::kHeadDim; ++d)
        {
            ASSERT_TRUE(std::isnan(__half2float(gotPool[base + d]))) << "b1 V-tail g=" << g << " d=" << d;
        }
    }
}

TEST(QsaIndexerDecode, ScoresDecodeMatchesReference)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int32_t kBatch = 2;
    constexpr int32_t kMaxBlocks = 128; // grid has 2 column CTAs; the second early-outs for b1
    std::vector<int32_t> const lens = {300, 100};
    PagedPoolHarness pool(kBatch, /* numPages = */ 8, /* maxPagesPerSeq = */ 3, /* pagesUsed = */ 3, /* seed = */ 17);
    std::mt19937 rng(303);

    // Random kbar rows placed at the V-tails; only the V plane is populated (a stray K-tail
    // or non-tail read would surface as NaN logits).
    std::array<std::vector<float>, kBatch> kbarDense;
    for (int32_t b = 0; b < kBatch; ++b)
    {
        int32_t const numVisible = lens[b] / qsa_ref::kRatio;
        kbarDense[b].reserve(static_cast<size_t>(numVisible) * qsa_ref::kHeadDim);
        for (int32_t g = 0; g < numVisible; ++g)
        {
            std::vector<half> const row = randomHalves(qsa_ref::kHeadDim, rng);
            int64_t const base = pool.tailOffset(b, /* plane = */ 1, g * qsa_ref::kRatio);
            std::copy(row.begin(), row.end(), pool.poolH.begin() + base);
            for (half v : row)
            {
                kbarDense[b].push_back(__half2float(v));
            }
        }
    }
    pool.upload();

    std::vector<half> const qNormedH = randomHalves(static_cast<size_t>(kBatch) * qsa_ref::kQRowWidth, rng);
    std::vector<float> const qNormedF = toFloats(qNormedH);
    rt::Tensor dQNormed(
        {kBatch, 1, qsa_ref::kNumHeads, qsa_ref::kHeadDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "qNormed");
    rt::Tensor dLens({kBatch}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "contextLengths");
    copyHostToDevice(dQNormed, qNormedH);
    copyHostToDevice(dLens, lens);

    // Sentinel-filled logits: only the valid prefix may be overwritten.
    constexpr float kSentinel = 1e30f;
    rt::Tensor dLogits({kBatch, kMaxBlocks}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
    copyHostToDevice(dLogits, std::vector<float>(static_cast<size_t>(kBatch) * kMaxBlocks, kSentinel));

    kernel::launchQsaIndexScoresDecode<half>(dLogits.dataPointer<float>(), dQNormed.dataPointer<half>(),
        dLens.dataPointer<int32_t>(), pool.state(), kBatch, kMaxBlocks, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> const got = copyDeviceToHost<float>(dLogits);

    for (int32_t b = 0; b < kBatch; ++b)
    {
        int32_t const numVisible = lens[b] / qsa_ref::kRatio;
        std::vector<float> const qRow(qNormedF.begin() + static_cast<int64_t>(b) * qsa_ref::kQRowWidth,
            qNormedF.begin() + static_cast<int64_t>(b + 1) * qsa_ref::kQRowWidth);
        std::vector<double> const ref = qsa_ref::refLogitsDecodeRow(qRow, kbarDense[b], numVisible);
        for (int32_t g = 0; g < kMaxBlocks; ++g)
        {
            float const v = got[static_cast<int64_t>(b) * kMaxBlocks + g];
            if (g < numVisible)
            {
                ASSERT_TRUE(isclose(v, static_cast<float>(ref[g]), /* rtol = */ 1e-3f, /* atol = */ 5e-3f))
                    << "b=" << b << " g=" << g << " got " << v << " want " << ref[g];
            }
            else
            {
                ASSERT_EQ(v, kSentinel) << "b=" << b << " g=" << g << " written past the valid prefix";
            }
        }
    }
}

namespace
{

//! Run B3 on host-crafted logits; returns outIdx [batch, kWidth] and the counters.
std::vector<int32_t> runTopKExpand(std::vector<float> const& logitsHost, std::vector<int32_t> const& lens,
    int32_t maxBlocks, bool withCounters, std::vector<int32_t>* countersOut, int32_t outIdxPoison = -7)
{
    int32_t const batch = static_cast<int32_t>(lens.size());
    rt::Tensor dLogits({batch, maxBlocks}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
    rt::Tensor dLens({batch}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "contextLengths");
    rt::Tensor dOutIdx({batch, 1, kWidth}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "outIdx");
    rt::Tensor dCounters({batch, kNumKVHeadsPool}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "counters");
    copyHostToDevice(dLogits, logitsHost);
    copyHostToDevice(dLens, lens);
    copyHostToDevice(dOutIdx, std::vector<int32_t>(static_cast<size_t>(batch) * kWidth, outIdxPoison));
    copyHostToDevice(dCounters, std::vector<int32_t>(static_cast<size_t>(batch) * kNumKVHeadsPool, 1234));

    kernel::launchQsaTopKExpandDecode(dOutIdx.dataPointer<int32_t>(), dLogits.dataPointer<float>(),
        dLens.dataPointer<int32_t>(), withCounters ? dCounters.dataPointer<int32_t>() : nullptr, kNumKVHeadsPool, batch,
        maxBlocks, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (countersOut != nullptr)
    {
        *countersOut = copyDeviceToHost<int32_t>(dCounters);
    }
    return copyDeviceToHost<int32_t>(dOutIdx);
}

} // namespace

TEST(QsaIndexerDecode, TopKSelectsReferenceSetOnDistinctLogits)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int32_t kMaxBlocks = 600;
    // nVis 575 / 600, tails 3 / 0. The second row fills maxBlocks EXACTLY — the runner
    // contract requires maxBlocks >= ceilDiv(ctx, 4), so nVis > maxBlocks is not a legal
    // input (and would overflow the host logits fill below).
    std::vector<int32_t> const lens = {2303, 2400};
    std::mt19937 rng(4242);

    // Distinct logits (no tie ambiguity); slots past nVis get +FLT_MAX so any out-of-range
    // read would corrupt the selection.
    std::vector<float> logits(static_cast<size_t>(lens.size()) * kMaxBlocks, FLT_MAX);
    for (size_t b = 0; b < lens.size(); ++b)
    {
        int32_t const numVisible = lens[b] / qsa_ref::kRatio;
        std::vector<float> values(numVisible);
        for (int32_t g = 0; g < numVisible; ++g)
        {
            values[g] = 0.5f + 1e-3f * g;
        }
        std::shuffle(values.begin(), values.end(), rng);
        std::copy(values.begin(), values.end(), logits.begin() + b * kMaxBlocks);
    }

    std::vector<int32_t> counters;
    std::vector<int32_t> const out = runTopKExpand(logits, lens, kMaxBlocks, /* withCounters = */ true, &counters);

    for (size_t b = 0; b < lens.size(); ++b)
    {
        SCOPED_TRACE("b=" + std::to_string(b));
        std::vector<int32_t> gpuBlocks;
        verifyDecodeRowStructure(out.data() + b * kWidth, lens[b], &gpuBlocks);
        if (::testing::Test::HasFatalFailure())
        {
            return;
        }
        std::vector<float> const row(logits.begin() + b * kMaxBlocks, logits.begin() + (b + 1) * kMaxBlocks);
        std::vector<int32_t> const ref = qsa_ref::refTopKBlocks(row, lens[b] / qsa_ref::kRatio, qsa_ref::kTopk);
        ASSERT_EQ(sorted(gpuBlocks), sorted(ref)) << "selected block set mismatch";
    }
    for (int32_t c : counters)
    {
        ASSERT_EQ(c, 0) << "split counter not zeroed";
    }
}

TEST(QsaIndexerDecode, TopKTieBreaksAscendingBlockId)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int32_t kMaxBlocks = 600;
    constexpr int32_t kCtx = 2400; // nVis = 600, tail = 0
    constexpr int32_t kNumVisible = kCtx / qsa_ref::kRatio;
    constexpr int32_t kHighCount = 400;
    constexpr int32_t kTieCount = 150;
    constexpr float kTieValue = 1.0f;
    std::mt19937 rng(777);

    // 400 distinct highs, 150 exact ties at the threshold (quota leaves 112 of them), and
    // 50 distinct lows; positions shuffled so tie ids are scattered.
    std::vector<int32_t> positions(kNumVisible);
    std::iota(positions.begin(), positions.end(), 0);
    std::shuffle(positions.begin(), positions.end(), rng);
    std::vector<float> logits(kMaxBlocks, FLT_MAX);
    std::vector<int32_t> tieIds;
    for (int32_t i = 0; i < kNumVisible; ++i)
    {
        int32_t const g = positions[i];
        if (i < kHighCount)
        {
            logits[g] = 10.0f + 1e-2f * i;
        }
        else if (i < kHighCount + kTieCount)
        {
            logits[g] = kTieValue;
            tieIds.push_back(g);
        }
        else
        {
            logits[g] = 1e-3f * (i - kHighCount - kTieCount + 1);
        }
    }
    std::sort(tieIds.begin(), tieIds.end());

    std::vector<int32_t> const out = runTopKExpand(logits, {kCtx}, kMaxBlocks, /* withCounters = */ true, nullptr);
    std::vector<int32_t> gpuBlocks;
    verifyDecodeRowStructure(out.data(), kCtx, &gpuBlocks);
    if (::testing::Test::HasFatalFailure())
    {
        return;
    }

    // Reference set == cub::DeviceSegmentedRadixSort descending stable behavior.
    std::vector<int32_t> const ref = qsa_ref::refTopKBlocks(logits, kNumVisible, qsa_ref::kTopk);
    ASSERT_EQ(sorted(gpuBlocks), sorted(ref));

    // Explicitly: ALL highs selected, plus exactly the quota of ties in ASCENDING id order.
    constexpr int32_t kTieQuota = qsa_ref::kTopk - kHighCount; // 112
    std::vector<char> selected(kNumVisible, 0);
    for (int32_t g : gpuBlocks)
    {
        selected[g] = 1;
    }
    for (int32_t i = 0; i < kTieCount; ++i)
    {
        ASSERT_EQ(selected[tieIds[i]], i < kTieQuota ? 1 : 0)
            << "tie block " << tieIds[i] << " (ascending tie rank " << i << ")";
    }
}

TEST(QsaIndexerDecode, TopKFastPathSelectsCausalPrefix)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int32_t kMaxBlocks = 600;
    // nVis <= 512 selects every block; ctx = 2051 fills the whole width. Logits are all NaN:
    // the fast path must never read them. Also exercises splitCounters == nullptr.
    std::vector<int32_t> const lens = {1, 3, 2048, 2051};
    std::vector<float> const logits(
        static_cast<size_t>(lens.size()) * kMaxBlocks, std::numeric_limits<float>::quiet_NaN());

    std::vector<int32_t> const out = runTopKExpand(logits, lens, kMaxBlocks, /* withCounters = */ false, nullptr);

    for (size_t b = 0; b < lens.size(); ++b)
    {
        SCOPED_TRACE("b=" + std::to_string(b));
        std::vector<int32_t> gpuBlocks;
        verifyDecodeRowStructure(out.data() + b * kWidth, lens[b], &gpuBlocks);
        if (::testing::Test::HasFatalFailure())
        {
            return;
        }
        // All nVis blocks selected: {0, 1, ..., nVis - 1}.
        std::vector<int32_t> expected(lens[b] / qsa_ref::kRatio);
        std::iota(expected.begin(), expected.end(), 0);
        ASSERT_EQ(sorted(gpuBlocks), expected);
    }
}

TEST(QsaIndexerDecode, TopKDeterministicRuns)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int32_t kMaxBlocks = 600;
    constexpr int32_t kCtx = 2402; // radix path, tail = 2
    constexpr int32_t kNumVisible = kCtx / qsa_ref::kRatio;
    std::mt19937 rng(31337);

    // Heavy ties stress the tie-quota scan; identical runs must be bitwise identical even
    // through differently poisoned output buffers.
    std::uniform_int_distribution<int32_t> coarse(0, 7);
    std::vector<float> logits(kMaxBlocks, FLT_MAX);
    for (int32_t g = 0; g < kNumVisible; ++g)
    {
        logits[g] = 0.125f * coarse(rng);
    }

    std::vector<int32_t> countersFirst;
    std::vector<int32_t> countersSecond;
    std::vector<int32_t> const first
        = runTopKExpand(logits, {kCtx}, kMaxBlocks, /* withCounters = */ true, &countersFirst, /* poison = */ 0x2B2B);
    std::vector<int32_t> const second
        = runTopKExpand(logits, {kCtx}, kMaxBlocks, /* withCounters = */ true, &countersSecond, /* poison = */ 0x5454);
    ASSERT_TRUE(first == second) << "outIdx must be bitwise identical across identical runs";
    for (int32_t c : countersFirst)
    {
        ASSERT_EQ(c, 0);
    }
    ASSERT_TRUE(countersFirst == countersSecond);

    std::vector<int32_t> gpuBlocks;
    verifyDecodeRowStructure(first.data(), kCtx, &gpuBlocks);
    if (::testing::Test::HasFatalFailure())
    {
        return;
    }
    std::vector<int32_t> const ref = qsa_ref::refTopKBlocks(logits, kNumVisible, qsa_ref::kTopk);
    ASSERT_EQ(sorted(gpuBlocks), sorted(ref));
}

TEST(QsaIndexerDecode, PrefillTailPopulationMatchesDenseKbar)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int32_t kBatch = 2;
    constexpr int32_t kSeqLen = 140; // crosses the 128-token page boundary
    constexpr int32_t kNumBlocks = 35;
    std::vector<int32_t> const lens = {137, 90};
    std::mt19937 rng(1234);

    PagedPoolHarness pool(kBatch, /* numPages = */ 5, /* maxPagesPerSeq = */ 2, /* pagesUsed = */ 2, /* seed = */ 19);
    std::vector<half> const indexQkH
        = randomHalves(static_cast<size_t>(kBatch) * kSeqLen * qsa_ref::kIndexQkWidth, rng);
    std::vector<half> const wKH = randomHalves(qsa_ref::kHeadDim, rng, -0.5f, 0.5f);
    std::vector<float> const cosSinF = makeCosSinTable(kSeqLen);

    rt::Tensor dIndexQk(
        {kBatch, kSeqLen, qsa_ref::kIndexQkWidth}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "indexQk");
    rt::Tensor dCosSin({kSeqLen, qsa_ref::kRotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "cosSin");
    rt::Tensor dLens({kBatch}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "contextLengths");
    rt::Tensor dWK({qsa_ref::kHeadDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "wK");
    rt::Tensor dKbarPaged(
        {kBatch, kNumBlocks, qsa_ref::kHeadDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "kbarPaged");
    rt::Tensor dKbarPlain(
        {kBatch, kNumBlocks, qsa_ref::kHeadDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "kbarPlain");
    copyHostToDevice(dIndexQk, indexQkH);
    copyHostToDevice(dCosSin, cosSinF);
    copyHostToDevice(dLens, lens);
    copyHostToDevice(dWK, wKH);

    kernel::launchQsaRawKTailWritePrefill<half>(
        dIndexQk.dataPointer<half>(), dLens.dataPointer<int32_t>(), pool.state(), kBatch, kSeqLen, nullptr);
    kernel::launchQsaIndexKCompressPaged<half>(dKbarPaged.dataPointer<half>(), dIndexQk.dataPointer<half>(),
        dCosSin.dataPointer<float>(), dLens.dataPointer<int32_t>(), dWK.dataPointer<half>(), kRmsEps, pool.state(),
        kBatch, kSeqLen, kNumBlocks, /* blockBegin = */ 0, /* blockEnd = */ kNumBlocks, /* pastLen = */ 0, nullptr);
    kernel::launchQsaIndexKCompress<half>(dKbarPlain.dataPointer<half>(), dIndexQk.dataPointer<half>(),
        dCosSin.dataPointer<float>(), dLens.dataPointer<int32_t>(), dWK.dataPointer<half>(), kRmsEps, kBatch, kSeqLen,
        kNumBlocks, /* blockBegin = */ 0, /* blockEnd = */ kNumBlocks, /* pastLen = */ 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // The paged variant's dense output stays bit-identical to plain K1b.
    std::vector<half> const kbarPaged = copyDeviceToHost<half>(dKbarPaged);
    std::vector<half> const kbarPlain = copyDeviceToHost<half>(dKbarPlain);
    ASSERT_EQ(kbarPaged.size(), kbarPlain.size());
    for (size_t i = 0; i < kbarPaged.size(); ++i)
    {
        ASSERT_EQ(bitsOf(kbarPaged[i]), bitsOf(kbarPlain[i])) << "dense kbar diverged at " << i;
    }

    std::vector<half> const gotPool = pool.download();
    std::vector<char> written(gotPool.size(), 0);
    for (int32_t b = 0; b < kBatch; ++b)
    {
        // K-tails: bit-exact copies of indexQk columns [512, 640) for the trailing INCOMPLETE
        // block's tokens only (137 -> token 136; 90 -> tokens 88, 89); complete blocks store
        // no raw rows (their K-tails must stay NaN, checked below).
        for (int32_t t = lens[b] - lens[b] % qsa_ref::kRatio; t < lens[b]; ++t)
        {
            int64_t const base = pool.tailOffset(b, /* plane = */ 0, t);
            int64_t const src = (static_cast<int64_t>(b) * kSeqLen + t) * qsa_ref::kIndexQkWidth + qsa_ref::kQRowWidth;
            for (int32_t d = 0; d < qsa_ref::kHeadDim; ++d)
            {
                ASSERT_EQ(bitsOf(gotPool[base + d]), bitsOf(indexQkH[src + d]))
                    << "b=" << b << " t=" << t << " d=" << d;
                written[base + d] = 1;
            }
        }
        // V-tails: bit-exact copies of the dense kbar rows for every COMPLETE block.
        for (int32_t g = 0; g < kNumBlocks; ++g)
        {
            if (g * qsa_ref::kRatio + qsa_ref::kRatio - 1 >= lens[b])
            {
                continue; // incomplete block: its V-tail must stay NaN (checked below)
            }
            int64_t const base = pool.tailOffset(b, /* plane = */ 1, g * qsa_ref::kRatio);
            int64_t const src = (static_cast<int64_t>(b) * kNumBlocks + g) * qsa_ref::kHeadDim;
            for (int32_t d = 0; d < qsa_ref::kHeadDim; ++d)
            {
                ASSERT_EQ(bitsOf(gotPool[base + d]), bitsOf(kbarPaged[src + d]))
                    << "b=" << b << " g=" << g << " d=" << d;
                written[base + d] = 1;
            }
        }
    }
    // NaN discipline: nothing else in the pool was touched.
    for (size_t i = 0; i < gotPool.size(); ++i)
    {
        if (written[i] == 0)
        {
            ASSERT_TRUE(std::isnan(__half2float(gotPool[i]))) << "pool cell " << i << " was clobbered";
        }
    }
}

TEST(QsaIndexerDecode, MultiStepLadderMatchesReference)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int32_t kBatch = 2;
    constexpr int32_t kPrefillSeqLen = 64;
    constexpr int32_t kSteps = 16;
    constexpr int32_t kMaxBlocks = 32;
    constexpr int32_t kMaxPositions = 128;
    // Staggered start lengths cross the 4-token block boundary on different steps; b1's
    // first decode boundary (ctx 64) compresses three PREFILL-written K-tails.
    std::vector<int32_t> lens = {64, 63};
    std::vector<int32_t> const prefillLens = lens;
    // A token's raw index-K is persisted only while its block is incomplete: never for the
    // block-completing token (t % 4 == 3), and after a prefill just for its trailing block.
    auto const rawTailWritten = [&prefillLens](int32_t b, int32_t t) {
        int32_t const trailingStart = prefillLens[b] - prefillLens[b] % qsa_ref::kRatio;
        return t >= trailingStart && t % qsa_ref::kRatio != qsa_ref::kRatio - 1;
    };
    std::mt19937 rng(90210);

    PagedPoolHarness pool(kBatch, /* numPages = */ 4, /* maxPagesPerSeq = */ 2, /* pagesUsed = */ 1, /* seed = */ 23);
    std::vector<float> const cosSinF = makeCosSinTable(kMaxPositions);
    std::vector<half> const wQH = randomHalves(qsa_ref::kHeadDim, rng, -0.5f, 0.5f);
    std::vector<half> const wKH = randomHalves(qsa_ref::kHeadDim, rng, -0.5f, 0.5f);
    std::vector<float> const wQF = toFloats(wQH);
    std::vector<float> const wKF = toFloats(wKH);

    // Per-sequence full history of indexQk rows (half-rounded FP32), for the from-scratch
    // host reference at every step.
    std::array<std::vector<float>, kBatch> histF;
    std::vector<half> prefillQkH
        = randomHalves(static_cast<size_t>(kBatch) * kPrefillSeqLen * qsa_ref::kIndexQkWidth, rng);
    // Poison b1's padding row (t = 63): the prefill population kernels must never read it.
    std::fill_n(prefillQkH.begin() + (static_cast<int64_t>(1) * kPrefillSeqLen + 63) * qsa_ref::kIndexQkWidth,
        qsa_ref::kIndexQkWidth, __float2half(std::numeric_limits<float>::quiet_NaN()));
    for (int32_t b = 0; b < kBatch; ++b)
    {
        for (int32_t t = 0; t < lens[b]; ++t)
        {
            int64_t const off = (static_cast<int64_t>(b) * kPrefillSeqLen + t) * qsa_ref::kIndexQkWidth;
            for (int32_t c = 0; c < qsa_ref::kIndexQkWidth; ++c)
            {
                histF[b].push_back(__half2float(prefillQkH[off + c]));
            }
        }
    }

    rt::Tensor dCosSin(
        {kMaxPositions, qsa_ref::kRotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "cosSin");
    rt::Tensor dLens({kBatch}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "contextLengths");
    rt::Tensor dWQ({qsa_ref::kHeadDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "wQ");
    rt::Tensor dWK({qsa_ref::kHeadDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "wK");
    copyHostToDevice(dCosSin, cosSinF);
    copyHostToDevice(dLens, lens);
    copyHostToDevice(dWQ, wQH);
    copyHostToDevice(dWK, wKH);

    // Prefill tail population (kernels 4a + 4b).
    {
        constexpr int32_t kPrefillBlocks = kPrefillSeqLen / qsa_ref::kRatio;
        rt::Tensor dPrefillQk({kBatch, kPrefillSeqLen, qsa_ref::kIndexQkWidth}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "prefillQk");
        rt::Tensor dKbar(
            {kBatch, kPrefillBlocks, qsa_ref::kHeadDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "kbar");
        copyHostToDevice(dPrefillQk, prefillQkH);
        kernel::launchQsaRawKTailWritePrefill<half>(dPrefillQk.dataPointer<half>(), dLens.dataPointer<int32_t>(),
            pool.state(), kBatch, kPrefillSeqLen, nullptr);
        kernel::launchQsaIndexKCompressPaged<half>(dKbar.dataPointer<half>(), dPrefillQk.dataPointer<half>(),
            dCosSin.dataPointer<float>(), dLens.dataPointer<int32_t>(), dWK.dataPointer<half>(), kRmsEps, pool.state(),
            kBatch, kPrefillSeqLen, kPrefillBlocks, /* blockBegin = */ 0, /* blockEnd = */ kPrefillBlocks,
            /* pastLen = */ 0, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    size_t const workspaceBytes = kernel::getQsaIndexerDecodeWorkspaceSize(kBatch, kMaxBlocks);
    rt::Tensor dWorkspace(
        {static_cast<int64_t>(workspaceBytes)}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "workspace");
    rt::Tensor dIndexQk(
        {kBatch, 1, qsa_ref::kIndexQkWidth}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "indexQk");
    rt::Tensor dOutIdx({kBatch, 1, kWidth}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "outIdx");
    rt::Tensor dCounters({kBatch, kNumKVHeadsPool}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "counters");
    // Documented decode workspace layout: slot 0 qNormed [B, 1, 4, 128] T, slot 1 logits
    // [B, maxBlocks] FP32, 128-byte-aligned slots.
    auto const alignUp = [](size_t bytes) { return (bytes + 127) / 128 * 128; };
    size_t const logitsOffset = alignUp(static_cast<size_t>(kBatch) * qsa_ref::kQRowWidth * sizeof(half));

    for (int32_t step = 0; step < kSteps; ++step)
    {
        SCOPED_TRACE("step=" + std::to_string(step));
        for (int32_t b = 0; b < kBatch; ++b)
        {
            lens[b] += 1;
        }
        std::vector<half> const newRowsH = randomHalves(static_cast<size_t>(kBatch) * qsa_ref::kIndexQkWidth, rng);
        for (int32_t b = 0; b < kBatch; ++b)
        {
            for (int32_t c = 0; c < qsa_ref::kIndexQkWidth; ++c)
            {
                histF[b].push_back(__half2float(newRowsH[static_cast<int64_t>(b) * qsa_ref::kIndexQkWidth + c]));
            }
        }
        copyHostToDevice(dIndexQk, newRowsH);
        copyHostToDevice(dLens, lens);
        copyHostToDevice(dOutIdx, std::vector<int32_t>(static_cast<size_t>(kBatch) * kWidth, -7));
        copyHostToDevice(dCounters, std::vector<int32_t>(static_cast<size_t>(kBatch) * kNumKVHeadsPool, 99));

        kernel::runQsaIndexerDecode<half>(dOutIdx.dataPointer<int32_t>(), dIndexQk.dataPointer<half>(),
            dCosSin.dataPointer<float>(), dLens.dataPointer<int32_t>(), dWQ.dataPointer<half>(),
            dWK.dataPointer<half>(), kRmsEps, pool.state(), dCounters.dataPointer<int32_t>(), dWorkspace.rawPointer(),
            workspaceBytes, kBatch, kMaxBlocks, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        // From-scratch host reference at the current lengths: padded dense history rebuild.
        int32_t const maxCtx = std::max(lens[0], lens[1]);
        int32_t const refBlocks = qsa_ref::ceilDiv(maxCtx, qsa_ref::kRatio);
        std::vector<float> denseHist(static_cast<size_t>(kBatch) * maxCtx * qsa_ref::kIndexQkWidth, 0.0f);
        for (int32_t b = 0; b < kBatch; ++b)
        {
            std::copy(histF[b].begin(), histF[b].end(),
                denseHist.begin() + static_cast<int64_t>(b) * maxCtx * qsa_ref::kIndexQkWidth);
        }
        std::vector<float> const kbarRef
            = qsa_ref::halfRound(qsa_ref::refKCompress(denseHist, cosSinF, lens, wKF, kRmsEps, kBatch, maxCtx));

        std::vector<half> qNormedGpu(static_cast<size_t>(kBatch) * qsa_ref::kQRowWidth);
        std::vector<float> logitsGpu(static_cast<size_t>(kBatch) * kMaxBlocks);
        CUDA_CHECK(cudaMemcpy(
            qNormedGpu.data(), dWorkspace.rawPointer(), qNormedGpu.size() * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(logitsGpu.data(), static_cast<std::byte*>(dWorkspace.rawPointer()) + logitsOffset,
            logitsGpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<int32_t> const outIdx = copyDeviceToHost<int32_t>(dOutIdx);
        std::vector<int32_t> const counters = copyDeviceToHost<int32_t>(dCounters);
        std::vector<half> const gotPool = pool.download();

        for (int32_t b = 0; b < kBatch; ++b)
        {
            SCOPED_TRACE("b=" + std::to_string(b));
            int32_t const ctx = lens[b];
            int32_t const numVisible = ctx / qsa_ref::kRatio;

            // (1) qNormed vs the q-prep reference at position ctx - 1.
            std::vector<float> const qRef = qsa_ref::refQPrepDecodeRow(
                histF[b].data() + histF[b].size() - qsa_ref::kIndexQkWidth, cosSinF, ctx - 1, wQF, kRmsEps);
            for (int32_t i = 0; i < qsa_ref::kQRowWidth; ++i)
            {
                float const v = __half2float(qNormedGpu[static_cast<int64_t>(b) * qsa_ref::kQRowWidth + i]);
                ASSERT_TRUE(isclose(v, qRef[i], /* rtol = */ 1e-2f, /* atol = */ 1e-3f)) << "qNormed i=" << i;
            }

            // (2) Raw K-tails: bit-exact copies for exactly the tokens whose block was still
            // incomplete when they arrived (stale copies are left in place, never read).
            for (int32_t t = 0; t < ctx; ++t)
            {
                if (!rawTailWritten(b, t))
                {
                    continue;
                }
                int64_t const base = pool.tailOffset(b, /* plane = */ 0, t);
                float const* src
                    = histF[b].data() + static_cast<int64_t>(t) * qsa_ref::kIndexQkWidth + qsa_ref::kQRowWidth;
                for (int32_t d = 0; d < qsa_ref::kHeadDim; ++d)
                {
                    ASSERT_EQ(bitsOf(gotPool[base + d]), bitsOf(__float2half(src[d]))) << "t=" << t << " d=" << d;
                }
            }

            // (3) kbar parity: every complete block vs the from-scratch compress reference.
            std::vector<float> kbarRows(static_cast<size_t>(numVisible) * qsa_ref::kHeadDim);
            for (int32_t g = 0; g < numVisible; ++g)
            {
                int64_t const base = pool.tailOffset(b, /* plane = */ 1, g * qsa_ref::kRatio);
                for (int32_t d = 0; d < qsa_ref::kHeadDim; ++d)
                {
                    float const v = __half2float(gotPool[base + d]);
                    float const want = kbarRef[(static_cast<int64_t>(b) * refBlocks + g) * qsa_ref::kHeadDim + d];
                    ASSERT_TRUE(isclose(v, want, /* rtol = */ 1e-2f, /* atol = */ 1e-3f))
                        << "kbar g=" << g << " d=" << d << " got " << v << " want " << want;
                    kbarRows[static_cast<int64_t>(g) * qsa_ref::kHeadDim + d] = v;
                }
            }

            // (4) Logits (workspace slot 1) vs the double-precision reference over the
            // GPU-held q and kbar values.
            std::vector<float> qRow(qsa_ref::kQRowWidth);
            for (int32_t i = 0; i < qsa_ref::kQRowWidth; ++i)
            {
                qRow[i] = __half2float(qNormedGpu[static_cast<int64_t>(b) * qsa_ref::kQRowWidth + i]);
            }
            std::vector<double> const logitsRef = qsa_ref::refLogitsDecodeRow(qRow, kbarRows, numVisible);
            for (int32_t g = 0; g < numVisible; ++g)
            {
                float const v = logitsGpu[static_cast<int64_t>(b) * kMaxBlocks + g];
                ASSERT_TRUE(isclose(v, static_cast<float>(logitsRef[g]), /* rtol = */ 1e-3f, /* atol = */ 5e-3f))
                    << "logits g=" << g << " got " << v << " want " << logitsRef[g];
            }

            // (5) Index SET: nVis <= 512 here, so the selection is exactly the causal
            // prefix {0, ..., ctx - 1} (distinct logits — no tie ambiguity by construction).
            std::vector<int32_t> gpuBlocks;
            verifyDecodeRowStructure(outIdx.data() + static_cast<int64_t>(b) * kWidth, ctx, &gpuBlocks);
            if (::testing::Test::HasFatalFailure())
            {
                return;
            }
            std::vector<int32_t> expected(numVisible);
            std::iota(expected.begin(), expected.end(), 0);
            ASSERT_EQ(sorted(gpuBlocks), expected);
        }
        for (int32_t c : counters)
        {
            ASSERT_EQ(c, 0) << "split counter not zeroed";
        }
        if (::testing::Test::HasFatalFailure())
        {
            return;
        }
    }

    // NaN discipline after the full ladder: only the K-tails of tokens that arrived into an
    // incomplete block and the V-tails of complete blocks may have been written; everything
    // else — including every K-tail of the blocks prefill completed and every
    // block-completing token — must still be NaN.
    std::vector<half> const gotPool = pool.download();
    std::vector<char> written(gotPool.size(), 0);
    for (int32_t b = 0; b < kBatch; ++b)
    {
        for (int32_t t = 0; t < lens[b]; ++t)
        {
            if (!rawTailWritten(b, t))
            {
                continue;
            }
            int64_t const base = pool.tailOffset(b, /* plane = */ 0, t);
            std::fill_n(written.begin() + base, qsa_ref::kHeadDim, 1);
        }
        for (int32_t g = 0; g < lens[b] / qsa_ref::kRatio; ++g)
        {
            int64_t const base = pool.tailOffset(b, /* plane = */ 1, g * qsa_ref::kRatio);
            std::fill_n(written.begin() + base, qsa_ref::kHeadDim, 1);
        }
    }
    for (size_t i = 0; i < gotPool.size(); ++i)
    {
        if (written[i] == 0)
        {
            ASSERT_TRUE(std::isnan(__half2float(gotPool[i]))) << "pool cell " << i << " was clobbered";
        }
    }
}
