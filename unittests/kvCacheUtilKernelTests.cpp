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

#include "common/cudaUtils.h"
#include "common/pagedKvTypes.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "runtime/state/kvPageTable.h"
#include "testUtils.h"
#include <algorithm>
#include <cstdint>
#include <cuda_fp8.h>
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

    // The single-layer copy kernels operate on a classic HND single-layer buffer
    // [maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]. The production NHD pool
    // [2, maxBatch, capPadded, H, D] is exercised by the *batched* path (HybridCacheManager
    // capture/restore) in hybridCacheManagerTests / sysPromptCachePagedTest; here we allocate a
    // standalone HND buffer so this test stays a self-consistent unit test of the single-layer HND
    // kernels (which have no production caller).
    // Test each layer independently using single-layer kernel variants
    for (int32_t idxL = 0; idxL < cacheParams.numDecoderLayers; idxL++)
    {
        rt::Tensor cacheTensor = rt::Tensor(
            {2, cacheParams.numKVHead, copySequenceLen, cacheParams.headDim}, rt::DeviceType::kGPU, DataType::kHALF);
        rt::Tensor kvCacheLayer = rt::Tensor(
            {cacheParams.maxBatchSize, 2, cacheParams.numKVHead, cacheParams.maxSequenceLength, cacheParams.headDim},
            rt::DeviceType::kGPU, DataType::kHALF);

        // Instantiate the cache tensor with random data
        std::vector<half> cacheDataHost(cacheTensor.getShape().volume(), 0.0f);
        uniformFloatInitialization(cacheDataHost);

        // Copy the cache tensor to the KVCache layer
        CUDA_CHECK(cudaMemcpy(
            cacheTensor.rawPointer(), cacheDataHost.data(), cacheTensor.getMemoryCapacity(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(kvCacheLayer.rawPointer(), 0, kvCacheLayer.getMemoryCapacity()));

        // Perform the copy from tensor to Cache and pull the data back to host.
        std::vector<half> kvCacheLayerHost(kvCacheLayer.getShape().volume(), 0.0f);
        kernel::instantiateKVCacheLayerFromTensor(kvCacheLayer, cacheTensor, copyBatchIdx, stream);
        CUDA_CHECK(cudaMemcpyAsync(kvCacheLayerHost.data(), kvCacheLayer.rawPointer(), kvCacheLayer.getMemoryCapacity(),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Verify the data in the KVCache layer
        auto compareCacheAndTensorData = [&]() {
            KvCacheIndexer indexer(
                cacheParams.maxBatchSize, cacheParams.numKVHead, cacheParams.maxSequenceLength, cacheParams.headDim);
            // tensorLayerOffset is 0 since the saved tensor is now per-layer [2, numKVHead, seqLen, headDim]
            for (int32_t idxS = 0; idxS < copySequenceLen; idxS++)
            {
                for (int32_t idxKV = 0; idxKV < cacheParams.numKVHead; idxKV++)
                {
                    for (int32_t idxD = 0; idxD < cacheParams.headDim; idxD++)
                    {
                        // First compare K then V.
                        // Saved tensor layout: [2, numKVHead, sequenceLength, headDim]
                        int64_t srcKOffset
                            = idxKV * copySequenceLen * cacheParams.headDim + idxS * cacheParams.headDim + idxD;
                        int64_t dstKOffset = indexer.indexK(copyBatchIdx, idxKV, idxS, idxD);
                        if (!isclose(kvCacheLayerHost[dstKOffset], cacheDataHost[srcKOffset], 1e-5, 1e-5))
                        {
                            std::cout << "Mismatch at layer " << idxL << ", sequence " << idxS << ", KV head " << idxKV
                                      << ", dim " << idxD << std::endl;
                            std::cout << "kvCacheLayerHost[dstKOffset]: " << __half2float(kvCacheLayerHost[dstKOffset])
                                      << ", cacheDataHost[srcKOffset]: " << __half2float(cacheDataHost[srcKOffset])
                                      << std::endl;
                        }
                        ASSERT_TRUE(isclose(kvCacheLayerHost[dstKOffset], cacheDataHost[srcKOffset], 1e-5, 1e-5));

                        int64_t srcVOffset = (cacheParams.numKVHead + idxKV) * copySequenceLen * cacheParams.headDim
                            + idxS * cacheParams.headDim + idxD;
                        int64_t dstVOffset = indexer.indexV(copyBatchIdx, idxKV, idxS, idxD);
                        ASSERT_TRUE(isclose(kvCacheLayerHost[dstVOffset], cacheDataHost[srcVOffset], 1e-5, 1e-5));
                    }
                }
            }
        };

        compareCacheAndTensorData();
        std::cout << "Tested copy from tensor to cache layer " << idxL << " with batchIdx " << copyBatchIdx
                  << ", sequence length " << copySequenceLen
                  << ", KVCacheLayer shape ([maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]): "
                  << kvCacheLayer.getShape().formatString() << std::endl;

        // cudaMemset the cache tensor and test from the other direction.
        CUDA_CHECK(cudaMemsetAsync(cacheTensor.rawPointer(), 0, cacheTensor.getMemoryCapacity(), stream));
        kernel::saveKVCacheLayerIntoTensor(cacheTensor, kvCacheLayer, copyBatchIdx, stream);
        CUDA_CHECK(cudaMemcpyAsync(cacheDataHost.data(), cacheTensor.rawPointer(), cacheTensor.getMemoryCapacity(),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        compareCacheAndTensorData();

        std::cout << "Tested copy from cache to tensor layer " << idxL << " with batchIdx " << copyBatchIdx
                  << ", sequence length " << copySequenceLen
                  << ", saved kvCacheTensor shape ([2, numKVHeads, sequenceLength, headDim]): "
                  << cacheTensor.getShape().formatString() << std::endl;
    }
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

//=============================================================================
// gatherPagedKVToSplit tests
//=============================================================================
// gatherPagedKVToSplit reads a single flat pool per rt::KVPageTable's convention: V page ids are
// always K page id + numPages ("-1" stays "-1"), so both halves of the [B,2,maxPagesPerSeq] table
// index the SAME pool base. Tests build the table through KVPageTable itself (never hand-roll the
// +numPages convention) so it can't drift from the writer/page-table contract.
namespace
{
// Flat index into a [B, S, H, D] tensor.
size_t splitIdx(int32_t b, int32_t s, int32_t h, int32_t d, int32_t S, int32_t H, int32_t D)
{
    return ((((size_t) b * S + s) * H + h) * D + d);
}

// Builds a [2*numPages, kTOKENS_PER_PAGE, H, D] flat pool (K pages then V pages, per KVPageTable's
// kernel view) from [B, S, H, D] K/V reference arrays, using an identity page mapping: page
// (b * pagesPerBatch + l) holds slot b's K tokens [l*128, l*128+128); the V page at +numPages holds
// the corresponding V tokens. Tail tokens beyond S within the last page are left as zero.
template <typename T>
std::vector<T> buildIdentityFlatPool(std::vector<T> const& refK, std::vector<T> const& refV, int32_t B, int32_t S,
    int32_t H, int32_t D, int32_t pagesPerBatch)
{
    int32_t const numPages = B * pagesPerBatch;
    size_t const pageElems = (size_t) rt::kTOKENS_PER_PAGE * H * D;
    std::vector<T> pool(static_cast<size_t>(2) * numPages * pageElems, T{});
    for (int32_t b = 0; b < B; ++b)
        for (int32_t l = 0; l < pagesPerBatch; ++l)
            for (int32_t t = 0; t < rt::kTOKENS_PER_PAGE; ++t)
            {
                int32_t const s = l * rt::kTOKENS_PER_PAGE + t;
                if (s >= S)
                    continue;
                int32_t const kPage = b * pagesPerBatch + l;
                int32_t const vPage = kPage + numPages;
                for (int32_t h = 0; h < H; ++h)
                    for (int32_t d = 0; d < D; ++d)
                    {
                        size_t const elemOff = (size_t) t * H * D + h * D + d;
                        pool[(size_t) kPage * pageElems + elemOff] = refK[splitIdx(b, s, h, d, S, H, D)];
                        pool[(size_t) vPage * pageElems + elemOff] = refV[splitIdx(b, s, h, d, S, H, D)];
                    }
            }
    return pool;
}

// Uploads a per-slot live-length vector to a device [B] int32 tensor (gatherPagedKVToSplit's kvSeqLens).
rt::Tensor makeKvSeqLens(std::vector<int32_t> const& lens)
{
    rt::Tensor t({static_cast<int64_t>(lens.size())}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(t, lens);
    return t;
}

} // namespace

// Identity page table: gatherPagedKVToSplit's output must match the existing
// cvtKVLayoutBHSDToSplitKV kernel's output (FP16, no dtype conversion involved).
TEST(GatherPagedKVToSplitTest, IdentityMatchesCvtKVLayoutBHSDToSplitKV)
{
    cudaStream_t stream{nullptr};
    int32_t const B = 2, H = 3, D = 64, S = 257; // S spans 3 logical pages (2*128 + 1).
    int32_t const pagesPerBatch = (S + rt::kTOKENS_PER_PAGE - 1) / rt::kTOKENS_PER_PAGE;
    int32_t const numPages = B * pagesPerBatch;

    size_t const splitVol = (size_t) B * S * H * D;
    std::vector<half> refK(splitVol), refV(splitVol);
    uniformFloatInitialization(refK, -4.f, 4.f);
    uniformFloatInitialization(refV, -4.f, 4.f);

    // Reference: build a [B, 2, H, S, D] source and run the existing conversion kernel.
    std::vector<half> bhsdSrc((size_t) B * 2 * H * S * D);
    for (int32_t b = 0; b < B; ++b)
        for (int32_t s = 0; s < S; ++s)
            for (int32_t h = 0; h < H; ++h)
                for (int32_t d = 0; d < D; ++d)
                {
                    size_t const kOff = (((((size_t) b * 2 + 0) * H + h) * S + s) * D + d);
                    size_t const vOff = (((((size_t) b * 2 + 1) * H + h) * S + s) * D + d);
                    bhsdSrc[kOff] = refK[splitIdx(b, s, h, d, S, H, D)];
                    bhsdSrc[vOff] = refV[splitIdx(b, s, h, d, S, H, D)];
                }
    rt::Tensor srcTensor({B, 2, H, S, D}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(srcTensor, bhsdSrc);
    rt::Tensor kRefTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vRefTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    kernel::cvtKVLayoutBHSDToSplitKV(srcTensor, kRefTensor, vRefTensor, rt::Tensor{}, S, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<half> const kRefHost = copyDeviceToHost<half>(kRefTensor);
    std::vector<half> const vRefHost = copyDeviceToHost<half>(vRefTensor);

    // Paged pool + identity table (via KVPageTable) for gatherPagedKVToSplit.
    std::vector<half> const poolHost = buildIdentityFlatPool(refK, refV, B, S, H, D, pagesPerBatch);
    rt::Tensor poolTensor({(int64_t) poolHost.size()}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(poolTensor, poolHost);

    rt::KVPageTable pageTable(B, pagesPerBatch, numPages);
    pageTable.setIdentity();
    pageTable.upload(stream);

    rt::Tensor kDstTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vDstTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kvSeqLens = makeKvSeqLens(std::vector<int32_t>(B, S));
    kernel::gatherPagedKVToSplit(poolTensor.rawPointer(), kDstTensor.rawPointer(), vDstTensor.rawPointer(),
        pageTable.kernelView().dataPointer<int32_t>(), kvSeqLens.dataPointer<int32_t>(), pagesPerBatch, B, S, H, D,
        sizeof(half), /*dequantFp8=*/false, /*kScale=*/1.f, /*vScale=*/1.f, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<half> const kOutHost = copyDeviceToHost<half>(kDstTensor);
    std::vector<half> const vOutHost = copyDeviceToHost<half>(vDstTensor);

    ASSERT_EQ(kOutHost.size(), kRefHost.size());
    for (size_t i = 0; i < kOutHost.size(); ++i)
    {
        ASSERT_EQ(__half_as_ushort(kOutHost[i]), __half_as_ushort(kRefHost[i])) << "K byte mismatch at flat idx " << i;
        ASSERT_EQ(__half_as_ushort(vOutHost[i]), __half_as_ushort(vRefHost[i])) << "V byte mismatch at flat idx " << i;
    }
}

// Dtype-agnostic byte-wise copy: a 1-byte element type (uint8_t, standing in for FP8's width) with an
// identity page table must reproduce the source bytes exactly. gatherPagedKVToSplit never interprets
// element values (unlike cvtKVLayoutBHSDToSplitKV's FP8 dequant path), so this is checked against a
// host-computed reference rather than the existing conversion kernel.
TEST(GatherPagedKVToSplitTest, IdentityByteCopyIsDtypeAgnostic)
{
    cudaStream_t stream{nullptr};
    int32_t const B = 2, H = 2, D = 32, S = 150; // S spans 2 logical pages.
    int32_t const pagesPerBatch = (S + rt::kTOKENS_PER_PAGE - 1) / rt::kTOKENS_PER_PAGE;
    int32_t const numPages = B * pagesPerBatch;

    size_t const splitVol = (size_t) B * S * H * D;
    std::vector<uint8_t> refK(splitVol), refV(splitVol);
    uniformIntInitialization(refK, 0, 255);
    uniformIntInitialization(refV, 0, 255);

    std::vector<uint8_t> const poolHost = buildIdentityFlatPool(refK, refV, B, S, H, D, pagesPerBatch);
    rt::Tensor poolTensor({(int64_t) poolHost.size()}, rt::DeviceType::kGPU, DataType::kUINT8);
    copyHostToDevice(poolTensor, poolHost);

    rt::KVPageTable pageTable(B, pagesPerBatch, numPages);
    pageTable.setIdentity();
    pageTable.upload(stream);

    rt::Tensor kDstTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kUINT8);
    rt::Tensor vDstTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kUINT8);
    rt::Tensor kvSeqLens = makeKvSeqLens(std::vector<int32_t>(B, S));
    kernel::gatherPagedKVToSplit(poolTensor.rawPointer(), kDstTensor.rawPointer(), vDstTensor.rawPointer(),
        pageTable.kernelView().dataPointer<int32_t>(), kvSeqLens.dataPointer<int32_t>(), pagesPerBatch, B, S, H, D,
        sizeof(uint8_t), /*dequantFp8=*/false, /*kScale=*/1.f, /*vScale=*/1.f, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<uint8_t> const kOutHost = copyDeviceToHost<uint8_t>(kDstTensor);
    std::vector<uint8_t> const vOutHost = copyDeviceToHost<uint8_t>(vDstTensor);

    ASSERT_EQ(kOutHost, refK);
    ASSERT_EQ(vOutHost, refV);
}

// Scrambled page table (cross-request-reuse-style), built via KVPageTable::setRow: output must match
// a host-computed gather that follows the same table. One slot's row is short (a natural -1 tail,
// e.g. a slot with fewer allocated pages than the batch's runtime seqLen) to also cover zero-fill.
TEST(GatherPagedKVToSplitTest, ScrambledTableMatchesHostGather)
{
    cudaStream_t stream{nullptr};
    int32_t const B = 2, H = 2, D = 32, maxPagesPerSeq = 4;
    int32_t const S = 3 * rt::kTOKENS_PER_PAGE + 50; // 4 logical pages, last one partial.
    int32_t const numLogicalPages = (S + rt::kTOKENS_PER_PAGE - 1) / rt::kTOKENS_PER_PAGE;
    ASSERT_EQ(numLogicalPages, maxPagesPerSeq);

    int32_t const numPhysicalPages = 10; // Larger than needed so ids can be scrambled/shared/unused.
    size_t const pageElems = (size_t) rt::kTOKENS_PER_PAGE * H * D;
    std::vector<half> poolHost(static_cast<size_t>(2) * numPhysicalPages * pageElems);
    uniformFloatInitialization(poolHost, -4.f, 4.f);

    // Scrambled table: non-identity physical page order; slot 1 only has 2 of 4 pages allocated
    // (the rest fall back to the sentinel tail per KVPageTable's convention).
    rt::KVPageTable pageTable(B, maxPagesPerSeq, numPhysicalPages);
    std::vector<int32_t> const slot0KIds = {9, 3, 6, 5};
    std::vector<int32_t> const slot1KIds = {0, 7};
    pageTable.setRow(0, slot0KIds.data(), static_cast<int32_t>(slot0KIds.size()));
    pageTable.setRow(1, slot1KIds.data(), static_cast<int32_t>(slot1KIds.size()));
    pageTable.upload(stream);

    // Host-computed expected [B, S, H, D] output, reading the table the same way the kernel does:
    // row b's K ids at hostRow(b)[0..maxPagesPerSeq), V ids at hostRow(b)[maxPagesPerSeq..2*maxPagesPerSeq).
    std::vector<half> expectedK((size_t) B * S * H * D, half(0.f)), expectedV((size_t) B * S * H * D, half(0.f));
    for (int32_t b = 0; b < B; ++b)
    {
        int32_t const* row = pageTable.hostRow(b);
        for (int32_t l = 0; l < maxPagesPerSeq; ++l)
        {
            int32_t const tokenStart = l * rt::kTOKENS_PER_PAGE;
            int32_t const tokensInPage = std::min(rt::kTOKENS_PER_PAGE, S - tokenStart);
            int32_t const kPage = row[l];
            int32_t const vPage = row[maxPagesPerSeq + l];
            for (int32_t t = 0; t < tokensInPage; ++t)
                for (int32_t h = 0; h < H; ++h)
                    for (int32_t d = 0; d < D; ++d)
                    {
                        size_t const dstOff = splitIdx(b, tokenStart + t, h, d, S, H, D);
                        size_t const elemOff = (size_t) t * H * D + h * D + d;
                        if (kPage >= 0)
                            expectedK[dstOff] = poolHost[(size_t) kPage * pageElems + elemOff];
                        if (vPage >= 0)
                            expectedV[dstOff] = poolHost[(size_t) vPage * pageElems + elemOff];
                    }
        }
    }

    rt::Tensor poolTensor({(int64_t) poolHost.size()}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(poolTensor, poolHost);

    rt::Tensor kDstTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vDstTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    // Destinations start non-zero so a missed zero-fill would be caught.
    std::vector<half> const canary((size_t) B * S * H * D, half(123.f));
    copyHostToDevice(kDstTensor, canary);
    copyHostToDevice(vDstTensor, canary);

    // Slot 0 is fully live (S tokens); slot 1 only has 2 allocated pages, so its live length is 256 and
    // logical pages 2,3 are the legal padding tail (-1 -> zero-fill, NOT a violation).
    rt::Tensor kvSeqLens = makeKvSeqLens({S, 2 * rt::kTOKENS_PER_PAGE});
    kernel::gatherPagedKVToSplit(poolTensor.rawPointer(), kDstTensor.rawPointer(), vDstTensor.rawPointer(),
        pageTable.kernelView().dataPointer<int32_t>(), kvSeqLens.dataPointer<int32_t>(), maxPagesPerSeq, B, S, H, D,
        sizeof(half), /*dequantFp8=*/false, /*kScale=*/1.f, /*vScale=*/1.f, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<half> const kOutHost = copyDeviceToHost<half>(kDstTensor);
    std::vector<half> const vOutHost = copyDeviceToHost<half>(vDstTensor);

    for (size_t i = 0; i < kOutHost.size(); ++i)
    {
        ASSERT_EQ(__half_as_ushort(kOutHost[i]), __half_as_ushort(expectedK[i])) << "K mismatch at flat idx " << i;
        ASSERT_EQ(__half_as_ushort(vOutHost[i]), __half_as_ushort(expectedV[i])) << "V mismatch at flat idx " << i;
    }
}

// All-unallocated table (KVPageTable's default-constructed state, count=0) modeling a slot with zero
// live tokens (liveLen=0): every logical page is beyond the live range, so the entire output is
// zero-filled and NO violation is flagged.
TEST(GatherPagedKVToSplitTest, AllPagesUnallocatedZeroFills)
{
    cudaStream_t stream{nullptr};
    int32_t const B = 1, H = 2, D = 16, maxPagesPerSeq = 2;
    int32_t const S = 2 * rt::kTOKENS_PER_PAGE;

    rt::KVPageTable pageTable(B, maxPagesPerSeq, /*numPages=*/1);
    pageTable.setRow(0, nullptr, 0);
    pageTable.upload(stream);

    // Pool contents are irrelevant since every entry is -1; keep a valid non-null pointer.
    rt::Tensor poolTensor({1}, rt::DeviceType::kGPU, DataType::kHALF);

    rt::Tensor kDstTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vDstTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    std::vector<half> const canary((size_t) B * S * H * D, half(42.f));
    copyHostToDevice(kDstTensor, canary);
    copyHostToDevice(vDstTensor, canary);

    rt::Tensor kvSeqLens = makeKvSeqLens({0}); // zero live tokens -> all pages are legal tail.
    kernel::gatherPagedKVToSplit(poolTensor.rawPointer(), kDstTensor.rawPointer(), vDstTensor.rawPointer(),
        pageTable.kernelView().dataPointer<int32_t>(), kvSeqLens.dataPointer<int32_t>(), maxPagesPerSeq, B, S, H, D,
        sizeof(half), /*dequantFp8=*/false, /*kScale=*/1.f, /*vScale=*/1.f, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<half> const kOutHost = copyDeviceToHost<half>(kDstTensor);
    std::vector<half> const vOutHost = copyDeviceToHost<half>(vDstTensor);

    for (size_t i = 0; i < kOutHost.size(); ++i)
    {
        ASSERT_EQ(__half_as_ushort(kOutHost[i]), 0u) << "K not zero-filled at flat idx " << i;
        ASSERT_EQ(__half_as_ushort(vOutHost[i]), 0u) << "V not zero-filled at flat idx " << i;
    }
}

// FP8 pool -> FP16 dequant gather: the destination must equal a host-computed dequant reference
// (fp8_value * scale), NOT the raw FP8 bytes. Identity table, distinct K vs V scales.
TEST(GatherPagedKVToSplitTest, Fp8DequantMatchesHostReference)
{
    cudaStream_t stream{nullptr};
    int32_t const B = 2, H = 2, D = 32, S = 200; // spans 2 logical pages.
    int32_t const pagesPerBatch = (S + rt::kTOKENS_PER_PAGE - 1) / rt::kTOKENS_PER_PAGE;
    int32_t const numPages = B * pagesPerBatch;
    float const kScale = 0.5f, vScale = 0.25f;

    // Reference K/V as FP8 e4m3 values (exactly representable so the comparison is exact).
    size_t const splitVol = (size_t) B * S * H * D;
    std::vector<__nv_fp8_e4m3> refK(splitVol), refV(splitVol);
    for (size_t i = 0; i < splitVol; ++i)
    {
        refK[i] = __nv_fp8_e4m3(static_cast<float>((i % 7) - 3)); // small integers, exact in e4m3.
        refV[i] = __nv_fp8_e4m3(static_cast<float>((i % 5) - 2));
    }

    std::vector<__nv_fp8_e4m3> const poolHost = buildIdentityFlatPool(refK, refV, B, S, H, D, pagesPerBatch);
    rt::Tensor poolTensor({(int64_t) poolHost.size()}, rt::DeviceType::kGPU, DataType::kFP8);
    copyHostToDevice(poolTensor, poolHost);

    rt::KVPageTable pageTable(B, pagesPerBatch, numPages);
    pageTable.setIdentity();
    pageTable.upload(stream);

    rt::Tensor kDstTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vDstTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kvSeqLens = makeKvSeqLens(std::vector<int32_t>(B, S));
    kernel::gatherPagedKVToSplit(poolTensor.rawPointer(), kDstTensor.rawPointer(), vDstTensor.rawPointer(),
        pageTable.kernelView().dataPointer<int32_t>(), kvSeqLens.dataPointer<int32_t>(), pagesPerBatch, B, S, H, D,
        sizeof(__nv_fp8_e4m3), /*dequantFp8=*/true, kScale, vScale, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<half> const kOutHost = copyDeviceToHost<half>(kDstTensor);
    std::vector<half> const vOutHost = copyDeviceToHost<half>(vDstTensor);

    for (size_t i = 0; i < splitVol; ++i)
    {
        half const expK = __float2half(static_cast<float>(refK[i]) * kScale);
        half const expV = __float2half(static_cast<float>(refV[i]) * vScale);
        ASSERT_EQ(__half_as_ushort(kOutHost[i]), __half_as_ushort(expK)) << "K dequant mismatch at flat idx " << i;
        ASSERT_EQ(__half_as_ushort(vOutHost[i]), __half_as_ushort(expV)) << "V dequant mismatch at flat idx " << i;
    }
}

// An unmapped page INSIDE the live range (a hole, not a trailing tail) cannot occur by construction
// (the runtime guarantees mapped coverage for live positions); if it ever does, the kernel must fail
// soft by zero-filling that page's destination span -- never reading a wild address or leaving
// uninitialized workspace for downstream consumers -- while still gathering the mapped pages.
TEST(GatherPagedKVToSplitTest, InRangeUnmappedPageZeroFillsDestination)
{
    cudaStream_t stream{nullptr};
    int32_t const B = 1, H = 2, D = 16, maxPagesPerSeq = 3;
    int32_t const S = 3 * rt::kTOKENS_PER_PAGE; // 3 logical pages, all within the live range.

    int32_t const numPhysicalPages = 8;
    size_t const pageElems = (size_t) rt::kTOKENS_PER_PAGE * H * D;
    rt::Tensor poolTensor(
        {static_cast<int64_t>(2) * numPhysicalPages * (int64_t) pageElems}, rt::DeviceType::kGPU, DataType::kHALF);
    std::vector<half> poolHost(static_cast<size_t>(2) * numPhysicalPages * pageElems, __float2half(3.0f));
    copyHostToDevice(poolTensor, poolHost);

    // Logical page 1's K id is -1 while the slot's live length spans all 3 pages -> an in-range hole. This
    // violates KVPageTable's "no live id after a sentinel" invariant on purpose, so we upload the raw
    // [B, 2, maxPagesPerSeq] table directly rather than through KVPageTable::upload.
    std::vector<int32_t> tableHost = {2, rt::kUNUSED_PAGE_ENTRY, 4,          // K row (page 1 unmapped)
        2 + numPhysicalPages, rt::kUNUSED_PAGE_ENTRY, 4 + numPhysicalPages}; // V row = K + numPages
    rt::Tensor pageTableTensor({B, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(pageTableTensor, tableHost);

    // Canary-poison the destinations so "zero-filled" is distinguishable from "left untouched".
    rt::Tensor kDstTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vDstTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    std::vector<half> canary(static_cast<size_t>(B) * S * H * D, __float2half(-9.0f));
    copyHostToDevice(kDstTensor, canary);
    copyHostToDevice(vDstTensor, canary);

    rt::Tensor kvSeqLens = makeKvSeqLens({S}); // all pages live.
    kernel::gatherPagedKVToSplit(poolTensor.rawPointer(), kDstTensor.rawPointer(), vDstTensor.rawPointer(),
        pageTableTensor.dataPointer<int32_t>(), kvSeqLens.dataPointer<int32_t>(), maxPagesPerSeq, B, S, H, D,
        sizeof(half), /*dequantFp8=*/false, /*kScale=*/1.f, /*vScale=*/1.f, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const kOut = copyDeviceToHost<half>(kDstTensor);
    auto const vOut = copyDeviceToHost<half>(vDstTensor);
    size_t const pageSpan = static_cast<size_t>(rt::kTOKENS_PER_PAGE) * H * D;
    for (size_t i = 0; i < static_cast<size_t>(S) * H * D; ++i)
    {
        float const expected = (i >= pageSpan && i < 2 * pageSpan) ? 0.0f : 3.0f;
        ASSERT_EQ(static_cast<float>(kOut[i]), expected) << "K flat idx " << i;
        ASSERT_EQ(static_cast<float>(vOut[i]), expected) << "V flat idx " << i;
    }
}
