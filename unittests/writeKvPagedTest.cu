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

// Round-trip test for the [B, 2, maxPagesPerSeq] page-table-aware applyRopeWriteKV write path
// (part of the paged-KV substrate).
//
// The page table carries K page ids in row [b*2+0] and the derived V page ids (K + numPages)
// in row [b*2+1]. Both halves address the SAME flat pool buffer -- a single combined tensor of
// shape [B, 2, Hkv, capPadded, D] happens to have exactly 2 * numPages * P * Hkv * D elements
// (numPages = B * maxPagesPerSeq, P = 128), so it doubles as the flat page pool addressed as
// [nPagesTotal, P, Hkv, D]:
//     addr = pool + ((page * P + inPage) * Hkv + h) * D + d
// A negative page-table entry (kUNUSED_PAGE_ENTRY) skips the write for that half.

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

// Flat element index into the combined [nPagesTotal, P, Hkv, D] pool for (page, inPage, h, d).
int64_t pagedPoolIndex(int32_t const page, int32_t const inPage, int32_t const h, int32_t const d,
    int32_t const numKVHeads, int32_t const headDim)
{
    return ((static_cast<int64_t>(page) * kPageSize + inPage) * numKVHeads + h) * headDim + d;
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
    kLegacy    // launchApplyRopeWriteKV (writeKInPlace=true)
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

    // The combined tensor [B, 2, Hkv, capPadded, D] has exactly 2 * numPages * P * Hkv * D
    // elements, so it doubles as the flat [nPagesTotal, P, Hkv, D] pool addressed by the page table.
    rt::Tensor kvPoolTensor(
        rt::Coords{batchSize, 2, numKVHeads, capPadded, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
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

    // Page table [B, 2, maxPagesPerSeq]: row (b*2+0) carries K page ids, row (b*2+1) the derived V
    // ids (K + numPages).
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
                    int64_t const kIdx = pagedPoolIndex(kPage, inPage, h, d, numKVHeads, headDim);
                    int64_t const vIdx = pagedPoolIndex(vPage, inPage, h, d, numKVHeads, headDim);
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

TEST(WriteKvPaged, LegacyIdentitySpansPageBoundaryStaggeredStart)
{
    TestWriteKvPagedPrefill(2, {16, 4, 64, 64}, /*maxSeq=*/200, /*qSeqLen=*/6, /*kvCacheStart=*/{126, 50},
        Variant::kLegacy, TableMode::kIdentity);
}

TEST(WriteKvPaged, LegacyScrambledTablePlacesWritesOnScrambledPages)
{
    TestWriteKvPagedPrefill(2, {16, 4, 64, 64}, /*maxSeq=*/200, /*qSeqLen=*/6, /*kvCacheStart=*/{126, 50},
        Variant::kLegacy, TableMode::kScrambled);
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
        rt::Coords{batchSize, 2, numKVHeads, capPadded, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    int64_t const poolVolume = kvPoolTensor.getShape().volume();
    std::vector<half> canaryFill(static_cast<size_t>(poolVolume), kCanary);
    copyHostToDevice(kvPoolTensor, canaryFill);

    // Page table [1, 2, 2]: page 0 is identity (K=0, V=0+numPages); page 1 is the sentinel in
    // both halves.
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
                int64_t const kIdx = pagedPoolIndex(/*page=*/0, s, h, d, numKVHeads, headDim);
                int64_t const vIdx = pagedPoolIndex(numPages, s, h, d, numKVHeads, headDim);
                ASSERT_TRUE(isclose(poolOut[kIdx], kReference[refOffset + d], 1e-3, 4e-3))
                    << "K mismatch s=" << s << " h=" << h << " d=" << d;
                ASSERT_TRUE(isclose(poolOut[vIdx], vInput[refOffset + d], 1e-3, 4e-3))
                    << "V mismatch s=" << s << " h=" << h << " d=" << d;
            }
        }
    }

    // Page 1 (the sentinel page's physical slots: page ids 1 and numPages+1) must be untouched:
    // the canary must survive since positions [128,200) all resolve to the -1 table entry.
    for (int32_t inPage = 0; inPage < kPageSize; ++inPage)
    {
        for (int32_t h = 0; h < numKVHeads; ++h)
        {
            for (int32_t d = 0; d < headDim; ++d)
            {
                int64_t const kIdx = pagedPoolIndex(/*page=*/1, inPage, h, d, numKVHeads, headDim);
                int64_t const vIdx = pagedPoolIndex(numPages + 1, inPage, h, d, numKVHeads, headDim);
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
        rt::Coords{batchSize, 2, numKVHeads, capPadded, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
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
            int64_t const kIdx = pagedPoolIndex(/*page=*/1, /*inPage=*/0, h, d, numKVHeads, headDim);
            int64_t const vIdx = pagedPoolIndex(numPages + 1, /*inPage=*/0, h, d, numKVHeads, headDim);
            EXPECT_TRUE(isclose(poolOut[kIdx], kCanary, 1e-6, 1e-6)) << "K canary clobbered h=" << h << " d=" << d;
            EXPECT_TRUE(isclose(poolOut[vIdx], kCanary, 1e-6, 1e-6)) << "V canary clobbered h=" << h << " d=" << d;
        }
    }
}

TEST(WriteKvPaged, RetentionPagesBeyondFloorKeepWriteAndGatherCorrect)
{
    // Robustness: the pool may be allocated with pages beyond the
    // active-capacity floor (poolNumPages = floor + extra). Identity slots still occupy the FIRST
    // floor pages of each half; the V-half of the (now larger) pool starts at poolNumPages, not at
    // the floor -- exactly what KVPageTable::setIdentity()/deriveV() compute when constructed with
    // the real (possibly-larger) numPages (see sharedResources.cpp::makeIdentityPageTable). This is
    // a smoke test that a write followed by a gather (read back) round-trips correctly on such a
    // pool: the write path is page-id-driven and must not assume poolNumPages == floor.
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
    int32_t const floorPages = batchSize * maxPagesPerSeq;
    int32_t const extraRetentionPages = 3;
    int32_t const poolNumPages = floorPages + extraRetentionPages; // > floor
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

    // Pool declared as [batchSize, 2, Hkv, poolNumPages*P, D] (launchApplyRopeWriteKV validates
    // dims[0] == batchSize even in page-table mode) with exactly 2 * poolNumPages * P * Hkv * D
    // elements -- poolNumPages pages per half, extraRetentionPages of which are unused tail pages
    // beyond the identity-mapped floor (matches KVCacheManager::getCombinedKVCachePoolView() when
    // Config::numPages > the floor).
    rt::Tensor kvPoolTensor(rt::Coords{batchSize, 2, numKVHeads, poolNumPages * kPageSize, headDim},
        rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    int64_t const poolVolume = kvPoolTensor.getShape().volume();
    std::vector<half> canaryFill(static_cast<size_t>(poolVolume), kCanary);
    copyHostToDevice(kvPoolTensor, canaryFill);

    // Identity page table, but V ids derived from poolNumPages (the real pool size), not floorPages
    // -- exactly KVPageTable::deriveV(k, numPages=poolNumPages).
    std::vector<int32_t> const pageTableHost{/*K=*/0, /*V=*/0 + poolNumPages};
    rt::Tensor pageTableTensor(
        rt::Coords{batchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(pageTableTensor, pageTableHost);
    int32_t const* pageTable = pageTableTensor.dataPointer<int32_t>();

    launchApplyRopeWriteKV(cosSinCacheTensor, kvCacheEndLensTensor, qTensor, kTensor, vTensor, kvPoolTensor, 1.0f, 1.0f,
        stream, /*writeKInPlace=*/true, pageTable, maxPagesPerSeq);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Gather (read back) and verify the write landed at page 0 (K) / page poolNumPages (V) -- NOT
    // at page floorPages, which is where a floor-only V-offset formula would incorrectly place it.
    auto const poolOut = copyDeviceToHost<half>(kvPoolTensor);
    for (int32_t s = 0; s < qSeqLen; ++s)
    {
        for (int32_t h = 0; h < numKVHeads; ++h)
        {
            int32_t const refOffset = s * numKVHeads * headDim + h * headDim;
            for (int32_t d = 0; d < headDim; ++d)
            {
                int64_t const kIdx = pagedPoolIndex(/*page=*/0, s, h, d, numKVHeads, headDim);
                int64_t const vIdx = pagedPoolIndex(poolNumPages, s, h, d, numKVHeads, headDim);
                ASSERT_TRUE(isclose(poolOut[kIdx], kReference[refOffset + d], 1e-3, 4e-3))
                    << "K mismatch s=" << s << " h=" << h << " d=" << d;
                ASSERT_TRUE(isclose(poolOut[vIdx], vInput[refOffset + d], 1e-3, 4e-3))
                    << "V mismatch s=" << s << " h=" << h << " d=" << d;
            }
        }
    }

    // The unused retention pages (indices [floorPages, poolNumPages) in EACH half) must be
    // untouched -- writes only ever target identity-mapped floor pages.
    for (int32_t page = floorPages; page < poolNumPages; ++page)
    {
        for (int32_t inPage = 0; inPage < kPageSize; ++inPage)
        {
            for (int32_t h = 0; h < numKVHeads; ++h)
            {
                for (int32_t d = 0; d < headDim; ++d)
                {
                    int64_t const kIdx = pagedPoolIndex(page, inPage, h, d, numKVHeads, headDim);
                    int64_t const vIdx = pagedPoolIndex(poolNumPages + page, inPage, h, d, numKVHeads, headDim);
                    EXPECT_TRUE(isclose(poolOut[kIdx], kCanary, 1e-6, 1e-6))
                        << "K retention page clobbered page=" << page << " inPage=" << inPage;
                    EXPECT_TRUE(isclose(poolOut[vIdx], kCanary, 1e-6, 1e-6))
                        << "V retention page clobbered page=" << page << " inPage=" << inPage;
                }
            }
        }
    }
}
