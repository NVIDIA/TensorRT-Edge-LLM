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

// Round-trip test for the [B, 2, maxPagesPerSeq] page-table-aware applyRopeWriteKV write path.
// The K row carries page ids in [0, numPages); the V row carries their flattened ids offset by
// numPages into the V plane of a [2, numPages, P, Hkv, D] pool. An unmapped or out-of-plane entry
// skips that cache plane's write.

#include <gtest/gtest.h>

#include <algorithm>

#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "references.h"
#include "testUtils.h"

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{

constexpr int32_t kPageSize = rt::kTOKENS_PER_PAGE;
half const kCanary = __float2half(12345.0f);

int32_t padToPage(int32_t v)
{
    return ((v + kPageSize - 1) / kPageSize) * kPageSize;
}

int64_t pagedPoolIndex(int32_t const cachePlane, int32_t const page, int32_t const inPage, int32_t const h,
    int32_t const d, int32_t const numPages, int32_t const numKVHeads, int32_t const headDim)
{
    return (((static_cast<int64_t>(cachePlane) * numPages + page) * kPageSize + inPage) * numKVHeads + h) * headDim + d;
}

struct AttnParams
{
    int32_t numQHeads;
    int32_t numKVHeads;
    int32_t headDim;
    int32_t rotaryDim;
};

enum class Variant
{
    kSplitQKV, // launchApplyRopeWriteKVSplitQKV
    kInPlace   // launchApplyRopeWriteKV (writeKInPlace=true)
};

enum class TableMode
{
    kIdentity,
    kScrambled
};

// Prefill round-trip: batchSize requests, each writing qSeqLen tokens starting at in-cache
// position kvCacheStart[b]. Verifies the paged write lands exactly where the page table (identity
// or a scrambled permutation of page ids) says it should.
void TestWriteKvPagedPrefill(int32_t const batchSize, AttnParams const& attnParams, int32_t const maxSeq,
    int32_t const qSeqLen, std::vector<int32_t> const& kvCacheStart, Variant const variant, TableMode const tableMode,
    float ropeTheta = 10000.0f)
{
    ASSERT_EQ(static_cast<int32_t>(kvCacheStart.size()), batchSize);

    cudaStream_t stream{nullptr};

    int32_t const headDim = attnParams.headDim;
    int32_t const rotaryDim = attnParams.rotaryDim;
    int32_t const numQHeads = attnParams.numQHeads;
    int32_t const numKVHeads = attnParams.numKVHeads;
    int32_t const capPadded = padToPage(maxSeq);
    int32_t const maxPagesPerSeq = capPadded / kPageSize;
    int32_t const numPages = batchSize * maxPagesPerSeq;

    float const ropeScale = 1.0f;
    bool const permuteRope = true;
    rt::Tensor cosSinCacheTensor(rt::Coords{1, capPadded, rotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    initializeNormalRopeCosSin(
        cosSinCacheTensor.dataPointer<float>(), ropeTheta, ropeScale, 1.0f, rotaryDim, capPadded, stream);

    // Build inputs + references. Q/K/V have layout [B, S, H, D].
    std::vector<half> qInput;
    std::vector<half> kInput;
    std::vector<half> vInput;
    std::vector<half> kReference; // roped K, [B, S, Hkv, D]
    std::vector<half> vReference; // V, [B, S, Hkv, D]
    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t s = 0; s < qSeqLen; ++s)
        {
            std::vector<half> qbs(numQHeads * headDim);
            std::vector<half> kbs(numKVHeads * headDim);
            std::vector<half> vbs(numKVHeads * headDim);
            uniformFloatInitialization(qbs);
            uniformFloatInitialization(kbs);
            uniformFloatInitialization(vbs);
            qInput.insert(qInput.end(), qbs.begin(), qbs.end());
            kInput.insert(kInput.end(), kbs.begin(), kbs.end());
            vInput.insert(vInput.end(), vbs.begin(), vbs.end());

            int32_t const pos = kvCacheStart[b] + s;
            std::vector<half> kRoped
                = ropeRef(kbs, numKVHeads, headDim, rotaryDim, pos, ropeScale, ropeTheta, permuteRope);
            kReference.insert(kReference.end(), kRoped.begin(), kRoped.end());
            vReference.insert(vReference.end(), vbs.begin(), vbs.end());
        }
    }

    rt::Tensor qTensor(
        rt::Coords{batchSize, qSeqLen, numQHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor kTensor(
        rt::Coords{batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor vTensor(
        rt::Coords{batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    copyHostToDevice(qTensor, qInput);
    copyHostToDevice(kTensor, kInput);
    copyHostToDevice(vTensor, vInput);

    std::vector<int32_t> kvCacheEndLens(batchSize);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        kvCacheEndLens[b] = kvCacheStart[b] + qSeqLen;
        ASSERT_LE(kvCacheEndLens[b], capPadded) << "request must fit within padded capacity";
    }
    rt::Tensor kvCacheEndLensTensor(rt::Coords{batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(kvCacheEndLensTensor, kvCacheEndLens);

    rt::Tensor kvPoolTensor(
        rt::Coords{2, numPages, kPageSize, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    int64_t const poolVolume = kvPoolTensor.getShape().volume();
    std::vector<half> zeros(static_cast<size_t>(poolVolume), __float2half(0.0f));
    copyHostToDevice(kvPoolTensor, zeros);

    // K page ids: identity range [0, numPages), optionally scrambled by reversal.
    std::vector<int32_t> kPageIds(numPages);
    for (int32_t i = 0; i < numPages; ++i)
    {
        kPageIds[i] = i;
    }
    if (tableMode == TableMode::kScrambled)
    {
        std::reverse(kPageIds.begin(), kPageIds.end());
    }

    // Page table [B, 2, maxPagesPerSeq]: V ids are K ids offset by numPages.
    std::vector<int32_t> pageTableHost(static_cast<size_t>(batchSize) * 2 * maxPagesPerSeq);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t j = 0; j < maxPagesPerSeq; ++j)
        {
            int32_t const kId = kPageIds[static_cast<size_t>(b) * maxPagesPerSeq + j];
            pageTableHost[(static_cast<size_t>(b) * 2 + 0) * maxPagesPerSeq + j] = kId;
            pageTableHost[(static_cast<size_t>(b) * 2 + 1) * maxPagesPerSeq + j] = kId + numPages;
        }
    }
    rt::Tensor pageTableTensor(
        rt::Coords{batchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(pageTableTensor, pageTableHost);
    int32_t const* pageTable = pageTableTensor.dataPointer<int32_t>();

    if (variant == Variant::kSplitQKV)
    {
        launchApplyRopeWriteKVSplitQKV(cosSinCacheTensor, kvCacheEndLensTensor, qTensor, kTensor, vTensor, kvPoolTensor,
            1.0f, 1.0f, stream, pageTable, maxPagesPerSeq, /*fp8QOut=*/nullptr, /*qScale=*/1.0f);
    }
    else
    {
        launchApplyRopeWriteKV(cosSinCacheTensor, kvCacheEndLensTensor, qTensor, kTensor, vTensor, kvPoolTensor, 1.0f,
            1.0f, stream, /*writeKInPlace=*/true, pageTable, maxPagesPerSeq);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const poolOut = copyDeviceToHost<half>(kvPoolTensor);

    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t s = 0; s < qSeqLen; ++s)
        {
            int32_t const pos = kvCacheStart[b] + s;
            int32_t const j = pos / kPageSize;
            int32_t const inPage = pos % kPageSize;
            int32_t const kPage = pageTableHost[(static_cast<size_t>(b) * 2 + 0) * maxPagesPerSeq + j];
            int32_t const vPage = pageTableHost[(static_cast<size_t>(b) * 2 + 1) * maxPagesPerSeq + j];
            for (int32_t h = 0; h < numKVHeads; ++h)
            {
                int32_t const refOffset = b * qSeqLen * numKVHeads * headDim + s * numKVHeads * headDim + h * headDim;
                for (int32_t d = 0; d < headDim; ++d)
                {
                    int64_t const kIdx = pagedPoolIndex(
                        /*cachePlane=*/0, kPage, inPage, h, d, numPages, numKVHeads, headDim);
                    int64_t const vIdx = pagedPoolIndex(
                        /*cachePlane=*/1, vPage - numPages, inPage, h, d, numPages, numKVHeads, headDim);
                    ASSERT_TRUE(isclose(poolOut[kIdx], kReference[refOffset + d], 1e-3, 4e-3))
                        << "K mismatch b=" << b << " s=" << s << " h=" << h << " d=" << d;
                    ASSERT_TRUE(isclose(poolOut[vIdx], vReference[refOffset + d], 1e-3, 4e-3))
                        << "V mismatch b=" << b << " s=" << s << " h=" << h << " d=" << d;
                }
            }
        }
    }
}

} // namespace

TEST(WriteKvPaged, SplitQKVIdentitySpansPageBoundaryStaggeredStart)
{
    // capPadded = 256 (2 pages). Slot 0 starts at pos 126 (crosses the page-0/page-1 boundary
    // within [126,132)); slot 1 starts at a different, non-boundary offset (staggered starts).
    TestWriteKvPagedPrefill(2, {16, 4, 64, 64}, /*maxSeq=*/200, /*qSeqLen=*/6, /*kvCacheStart=*/{126, 50},
        Variant::kSplitQKV, TableMode::kIdentity);
}

TEST(WriteKvPaged, InPlaceIdentitySpansPageBoundaryStaggeredStart)
{
    TestWriteKvPagedPrefill(2, {16, 4, 64, 64}, /*maxSeq=*/200, /*qSeqLen=*/6, /*kvCacheStart=*/{126, 50},
        Variant::kInPlace, TableMode::kIdentity);
}

TEST(WriteKvPaged, InPlaceScrambledTablePlacesWritesOnScrambledPages)
{
    TestWriteKvPagedPrefill(2, {16, 4, 64, 64}, /*maxSeq=*/200, /*qSeqLen=*/6, /*kvCacheStart=*/{126, 50},
        Variant::kInPlace, TableMode::kScrambled);
}

TEST(WriteKvPaged, SplitQKVScrambledTablePlacesWritesOnScrambledPages)
{
    TestWriteKvPagedPrefill(2, {16, 4, 64, 64}, /*maxSeq=*/200, /*qSeqLen=*/6, /*kvCacheStart=*/{126, 50},
        Variant::kSplitQKV, TableMode::kScrambled);
}

TEST(WriteKvPaged, NegativeEntryInLiveRangeSkipsWrite)
{
    // One slot, capPadded = 256 (2 pages). Page 1's table entry is the sentinel -1 in both the K
    // and V rows. This call has no tokenPosIds (kvCacheEndLens-only path), so every position in
    // [0, qSeqLen) is live -- positions [128, 200) landing on the -1 entry are therefore an
    // in-range hole (an internal invariant violation), NOT the legal "beyond live
    // sequence" padding-tail case. The write must be skipped (canary poisoning that region must
    // survive untouched; skip-on-negative-id means no wild-address write). Page 0 (positions
    // [0, 128)) is a live identity entry and must still be written correctly.
    cudaStream_t stream{nullptr};

    int32_t const batchSize = 1;
    int32_t const numQHeads = 4;
    int32_t const numKVHeads = 2;
    int32_t const headDim = 32;
    int32_t const rotaryDim = 32;
    int32_t const qSeqLen = 200;
    int32_t const maxSeq = 200;
    int32_t const capPadded = padToPage(maxSeq);
    int32_t const maxPagesPerSeq = capPadded / kPageSize;
    int32_t const numPages = batchSize * maxPagesPerSeq;
    ASSERT_EQ(maxPagesPerSeq, 2);

    float const ropeTheta = 10000.0f;
    float const ropeScale = 1.0f;
    bool const permuteRope = true;
    rt::Tensor cosSinCacheTensor(rt::Coords{1, capPadded, rotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    initializeNormalRopeCosSin(
        cosSinCacheTensor.dataPointer<float>(), ropeTheta, ropeScale, 1.0f, rotaryDim, capPadded, stream);

    std::vector<half> qInput(static_cast<size_t>(qSeqLen) * numQHeads * headDim);
    std::vector<half> kInput(static_cast<size_t>(qSeqLen) * numKVHeads * headDim);
    std::vector<half> vInput(static_cast<size_t>(qSeqLen) * numKVHeads * headDim);
    uniformFloatInitialization(qInput);
    uniformFloatInitialization(kInput);
    uniformFloatInitialization(vInput);

    std::vector<half> kReference; // roped K, [S, Hkv, D], page 0 tokens only
    for (int32_t s = 0; s < kPageSize; ++s)
    {
        std::vector<half> kbs(kInput.begin() + static_cast<size_t>(s) * numKVHeads * headDim,
            kInput.begin() + static_cast<size_t>(s + 1) * numKVHeads * headDim);
        std::vector<half> kRoped = ropeRef(kbs, numKVHeads, headDim, rotaryDim, s, ropeScale, ropeTheta, permuteRope);
        kReference.insert(kReference.end(), kRoped.begin(), kRoped.end());
    }

    rt::Tensor qTensor(
        rt::Coords{batchSize, qSeqLen, numQHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor kTensor(
        rt::Coords{batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor vTensor(
        rt::Coords{batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    copyHostToDevice(qTensor, qInput);
    copyHostToDevice(kTensor, kInput);
    copyHostToDevice(vTensor, vInput);

    std::vector<int32_t> kvCacheEndLens{qSeqLen};
    rt::Tensor kvCacheEndLensTensor(rt::Coords{batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(kvCacheEndLensTensor, kvCacheEndLens);

    rt::Tensor kvPoolTensor(
        rt::Coords{2, numPages, kPageSize, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    int64_t const poolVolume = kvPoolTensor.getShape().volume();
    std::vector<half> canaryFill(static_cast<size_t>(poolVolume), kCanary);
    copyHostToDevice(kvPoolTensor, canaryFill);

    // Page table [1, 2, 2]: page 0 is identity in both planes; page 1 is the sentinel in both halves.
    std::vector<int32_t> pageTableHost{0, rt::kUNUSED_PAGE_ENTRY, numPages, rt::kUNUSED_PAGE_ENTRY};
    rt::Tensor pageTableTensor(
        rt::Coords{batchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(pageTableTensor, pageTableHost);
    int32_t const* pageTable = pageTableTensor.dataPointer<int32_t>();

    launchApplyRopeWriteKV(cosSinCacheTensor, kvCacheEndLensTensor, qTensor, kTensor, vTensor, kvPoolTensor, 1.0f, 1.0f,
        stream, /*writeKInPlace=*/true, pageTable, maxPagesPerSeq);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const poolOut = copyDeviceToHost<half>(kvPoolTensor);

    // Page 0: valid write, must match the reference.
    for (int32_t s = 0; s < kPageSize; ++s)
    {
        for (int32_t h = 0; h < numKVHeads; ++h)
        {
            int32_t const refOffset = s * numKVHeads * headDim + h * headDim;
            for (int32_t d = 0; d < headDim; ++d)
            {
                int64_t const kIdx = pagedPoolIndex(
                    /*cachePlane=*/0, /*page=*/0, s, h, d, numPages, numKVHeads, headDim);
                int64_t const vIdx = pagedPoolIndex(
                    /*cachePlane=*/1, /*page=*/0, s, h, d, numPages, numKVHeads, headDim);
                ASSERT_TRUE(isclose(poolOut[kIdx], kReference[refOffset + d], 1e-3, 4e-3))
                    << "K mismatch s=" << s << " h=" << h << " d=" << d;
                ASSERT_TRUE(isclose(poolOut[vIdx], vInput[refOffset + d], 1e-3, 4e-3))
                    << "V mismatch s=" << s << " h=" << h << " d=" << d;
            }
        }
    }

    // Page 1 must be untouched because positions [128,200) resolve to the -1 table entry.
    for (int32_t inPage = 0; inPage < kPageSize; ++inPage)
    {
        for (int32_t h = 0; h < numKVHeads; ++h)
        {
            for (int32_t d = 0; d < headDim; ++d)
            {
                int64_t const kIdx = pagedPoolIndex(
                    /*cachePlane=*/0, /*page=*/1, inPage, h, d, numPages, numKVHeads, headDim);
                int64_t const vIdx = pagedPoolIndex(
                    /*cachePlane=*/1, /*page=*/1, inPage, h, d, numPages, numKVHeads, headDim);
                ASSERT_TRUE(isclose(poolOut[kIdx], kCanary, 1e-6, 1e-6))
                    << "K canary clobbered inPage=" << inPage << " h=" << h << " d=" << d;
                ASSERT_TRUE(isclose(poolOut[vIdx], kCanary, 1e-6, 1e-6))
                    << "V canary clobbered inPage=" << inPage << " h=" << h << " d=" << d;
            }
        }
    }
}

TEST(WriteKvPaged, PaddingTokenWithNegativePageEntryIsSkipped)
{
    // Tree-decoding path (has tokenPosIds, so a padding token is possible). One slot, 2 tree
    // positions: token 0 is real (tokenPosIds=127, cache-write position 127, page 0 -- valid);
    // token 1 is padding (tokenPosIds=-1), and its cache-write position (128) deliberately lands on
    // page 1, whose table entry is the unmapped sentinel -1 in both K and V rows. isPaddingToken
    // gates the ENTIRE page-id check (see applyRopeWriteKV.cu), so this negative entry must never be
    // read for a padding token -- the padding tail is skipped without touching the table.
    cudaStream_t stream{nullptr};

    int32_t const batchSize = 1;
    int32_t const numQHeads = 4;
    int32_t const numKVHeads = 2;
    int32_t const headDim = 32;
    int32_t const rotaryDim = 32;
    int32_t const qSeqLen = 2;
    int32_t const capPadded = 256; // 2 pages
    int32_t const maxPagesPerSeq = capPadded / kPageSize;
    int32_t const numPages = batchSize * maxPagesPerSeq;
    ASSERT_EQ(maxPagesPerSeq, 2);

    float const ropeTheta = 10000.0f;
    float const ropeScale = 1.0f;
    rt::Tensor cosSinCacheTensor(rt::Coords{1, capPadded, rotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    initializeNormalRopeCosSin(
        cosSinCacheTensor.dataPointer<float>(), ropeTheta, ropeScale, 1.0f, rotaryDim, capPadded, stream);

    std::vector<half> qInput(static_cast<size_t>(qSeqLen) * numQHeads * headDim);
    std::vector<half> kInput(static_cast<size_t>(qSeqLen) * numKVHeads * headDim);
    std::vector<half> vInput(static_cast<size_t>(qSeqLen) * numKVHeads * headDim);
    uniformFloatInitialization(qInput);
    uniformFloatInitialization(kInput);
    uniformFloatInitialization(vInput);

    rt::Tensor qTensor(
        rt::Coords{batchSize, qSeqLen, numQHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor kTensor(
        rt::Coords{batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor vTensor(
        rt::Coords{batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    copyHostToDevice(qTensor, qInput);
    copyHostToDevice(kTensor, kInput);
    copyHostToDevice(vTensor, vInput);

    // kvCacheStartIdx = kvCacheEndLens[0] - qSeqLen = 127, so token 0 writes to position 127 (page 0)
    // and token 1 writes to position 128 (page 1).
    std::vector<int32_t> const kvCacheEndLens{129};
    rt::Tensor kvCacheEndLensTensor(rt::Coords{batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(kvCacheEndLensTensor, kvCacheEndLens);

    std::vector<int32_t> const tokenPosIds{127, -1}; // token 0 real, token 1 padding
    rt::Tensor tokenPosIdsTensor(rt::Coords{batchSize, qSeqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(tokenPosIdsTensor, tokenPosIds);

    rt::Tensor kvPoolTensor(
        rt::Coords{2, numPages, kPageSize, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    int64_t const poolVolume = kvPoolTensor.getShape().volume();
    std::vector<half> canaryFill(static_cast<size_t>(poolVolume), kCanary);
    copyHostToDevice(kvPoolTensor, canaryFill);

    // Page table: page 0 (real token's slot) is identity-valid; page 1 (padding token's slot) is the
    // unmapped sentinel in both halves.
    std::vector<int32_t> const pageTableHost{0, rt::kUNUSED_PAGE_ENTRY, numPages, rt::kUNUSED_PAGE_ENTRY};
    rt::Tensor pageTableTensor(
        rt::Coords{batchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(pageTableTensor, pageTableHost);
    int32_t const* pageTable = pageTableTensor.dataPointer<int32_t>();

    launchApplyRopeWriteKVTreeDecoding(cosSinCacheTensor, kvCacheEndLensTensor, tokenPosIdsTensor, qTensor, kTensor,
        vTensor, kvPoolTensor, 1.0f, 1.0f, stream, pageTable, maxPagesPerSeq);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Sanity: page 1 (the padding token's slot) must still be untouched by the skipped write.
    auto const poolOut = copyDeviceToHost<half>(kvPoolTensor);
    for (int32_t h = 0; h < numKVHeads; ++h)
    {
        for (int32_t d = 0; d < headDim; ++d)
        {
            int64_t const kIdx = pagedPoolIndex(
                /*cachePlane=*/0, /*page=*/1, /*inPage=*/0, h, d, numPages, numKVHeads, headDim);
            int64_t const vIdx = pagedPoolIndex(
                /*cachePlane=*/1, /*page=*/1, /*inPage=*/0, h, d, numPages, numKVHeads, headDim);
            EXPECT_TRUE(isclose(poolOut[kIdx], kCanary, 1e-6, 1e-6)) << "K canary clobbered h=" << h << " d=" << d;
            EXPECT_TRUE(isclose(poolOut[vIdx], kCanary, 1e-6, 1e-6)) << "V canary clobbered h=" << h << " d=" << d;
        }
    }
}

TEST(WriteKvPaged, ExtraRetainedPagesKeepWriteAndGatherCorrect)
{
    // Identity slots occupy the minimum active pages while the pool retains additional pages.
    cudaStream_t stream{nullptr};

    int32_t const batchSize = 1;
    int32_t const numQHeads = 4;
    int32_t const numKVHeads = 2;
    int32_t const headDim = 32;
    int32_t const rotaryDim = 32;
    int32_t const qSeqLen = 128;
    int32_t const maxSeq = 128;
    int32_t const capPadded = padToPage(maxSeq);
    int32_t const maxPagesPerSeq = capPadded / kPageSize;
    int32_t const minimumActivePages = batchSize * maxPagesPerSeq;
    int32_t const extraRetainedPages = 3;
    int32_t const poolNumPages = minimumActivePages + extraRetainedPages;
    ASSERT_EQ(maxPagesPerSeq, 1);

    float const ropeTheta = 10000.0f;
    float const ropeScale = 1.0f;
    bool const permuteRope = true;
    rt::Tensor cosSinCacheTensor(rt::Coords{1, capPadded, rotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    initializeNormalRopeCosSin(
        cosSinCacheTensor.dataPointer<float>(), ropeTheta, ropeScale, 1.0f, rotaryDim, capPadded, stream);

    std::vector<half> qInput(static_cast<size_t>(qSeqLen) * numQHeads * headDim);
    std::vector<half> kInput(static_cast<size_t>(qSeqLen) * numKVHeads * headDim);
    std::vector<half> vInput(static_cast<size_t>(qSeqLen) * numKVHeads * headDim);
    uniformFloatInitialization(qInput);
    uniformFloatInitialization(kInput);
    uniformFloatInitialization(vInput);

    std::vector<half> kReference; // roped K, [S, Hkv, D]
    for (int32_t s = 0; s < qSeqLen; ++s)
    {
        std::vector<half> kbs(kInput.begin() + static_cast<size_t>(s) * numKVHeads * headDim,
            kInput.begin() + static_cast<size_t>(s + 1) * numKVHeads * headDim);
        std::vector<half> kRoped = ropeRef(kbs, numKVHeads, headDim, rotaryDim, s, ropeScale, ropeTheta, permuteRope);
        kReference.insert(kReference.end(), kRoped.begin(), kRoped.end());
    }

    rt::Tensor qTensor(
        rt::Coords{batchSize, qSeqLen, numQHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor kTensor(
        rt::Coords{batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor vTensor(
        rt::Coords{batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    copyHostToDevice(qTensor, qInput);
    copyHostToDevice(kTensor, kInput);
    copyHostToDevice(vTensor, vInput);

    std::vector<int32_t> const kvCacheEndLens{qSeqLen};
    rt::Tensor kvCacheEndLensTensor(rt::Coords{batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(kvCacheEndLensTensor, kvCacheEndLens);

    // The pool retains extra pages beyond the identity-mapped active pages.
    rt::Tensor kvPoolTensor(
        rt::Coords{2, poolNumPages, kPageSize, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    int64_t const poolVolume = kvPoolTensor.getShape().volume();
    std::vector<half> canaryFill(static_cast<size_t>(poolVolume), kCanary);
    copyHostToDevice(kvPoolTensor, canaryFill);

    std::vector<int32_t> const pageTableHost{/*K=*/0, /*V=*/poolNumPages};
    rt::Tensor pageTableTensor(
        rt::Coords{batchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(pageTableTensor, pageTableHost);
    int32_t const* pageTable = pageTableTensor.dataPointer<int32_t>();

    launchApplyRopeWriteKV(cosSinCacheTensor, kvCacheEndLensTensor, qTensor, kTensor, vTensor, kvPoolTensor, 1.0f, 1.0f,
        stream, /*writeKInPlace=*/true, pageTable, maxPagesPerSeq);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify the write landed at page 0 in both planes, not at an offset based only on active pages.
    auto const poolOut = copyDeviceToHost<half>(kvPoolTensor);
    for (int32_t s = 0; s < qSeqLen; ++s)
    {
        for (int32_t h = 0; h < numKVHeads; ++h)
        {
            int32_t const refOffset = s * numKVHeads * headDim + h * headDim;
            for (int32_t d = 0; d < headDim; ++d)
            {
                int64_t const kIdx = pagedPoolIndex(
                    /*cachePlane=*/0, /*page=*/0, s, h, d, poolNumPages, numKVHeads, headDim);
                int64_t const vIdx = pagedPoolIndex(
                    /*cachePlane=*/1, /*page=*/0, s, h, d, poolNumPages, numKVHeads, headDim);
                ASSERT_TRUE(isclose(poolOut[kIdx], kReference[refOffset + d], 1e-3, 4e-3))
                    << "K mismatch s=" << s << " h=" << h << " d=" << d;
                ASSERT_TRUE(isclose(poolOut[vIdx], vInput[refOffset + d], 1e-3, 4e-3))
                    << "V mismatch s=" << s << " h=" << h << " d=" << d;
            }
        }
    }

    // The extra retained pages [minimumActivePages, poolNumPages) are untouched by identity writes.
    for (int32_t page = minimumActivePages; page < poolNumPages; ++page)
    {
        for (int32_t inPage = 0; inPage < kPageSize; ++inPage)
        {
            for (int32_t h = 0; h < numKVHeads; ++h)
            {
                for (int32_t d = 0; d < headDim; ++d)
                {
                    int64_t const kIdx = pagedPoolIndex(
                        /*cachePlane=*/0, page, inPage, h, d, poolNumPages, numKVHeads, headDim);
                    int64_t const vIdx = pagedPoolIndex(
                        /*cachePlane=*/1, page, inPage, h, d, poolNumPages, numKVHeads, headDim);
                    EXPECT_TRUE(isclose(poolOut[kIdx], kCanary, 1e-6, 1e-6))
                        << "extra retained K page clobbered page=" << page << " inPage=" << inPage;
                    EXPECT_TRUE(isclose(poolOut[vIdx], kCanary, 1e-6, 1e-6))
                        << "extra retained V page clobbered page=" << page << " inPage=" << inPage;
                }
            }
        }
    }
}
