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

// Unit tests for KVCacheManager's paged-KV pools.
//
// Full-attention layers retain full-sequence page pools. Reduced SWA layers use
// explicit per-layer physical page counts with bounded active and replacement
// reservations derived from W. Each layer owns a page pool
// [2, numPages, kTOKENS_PER_PAGE, numKVHeads_i, headDim_i], with
// kPoolPtr/vPoolPtr as the contiguous K/V halves.

#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "runtime/kvCacheManager.h"
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <limits>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace trt_edgellm;

namespace
{

int32_t padToPage(int32_t v)
{
    return ((v + rt::kTOKENS_PER_PAGE - 1) / rt::kTOKENS_PER_PAGE) * rt::kTOKENS_PER_PAGE;
}

// Build a heterogeneous KV config: layer 0 uses (h0, d0), layer 1 uses (h1, d1).
rt::KVCacheManager::Config makeHeteroConfig(
    int32_t maxBatch, int32_t maxSeq, int32_t h0, int32_t d0, int32_t h1, int32_t d1, DataType dtype)
{
    std::vector<rt::KVLayerConfig> layers{rt::KVLayerConfig{h0, d0}, rt::KVLayerConfig{h1, d1}};
    return rt::KVCacheManager::Config{/*numAttentionLayers=*/2, maxBatch, maxSeq, layers, dtype};
}

bool hasCudaDevice()
{
    int32_t deviceCount{0};
    cudaError_t const status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess)
    {
        static_cast<void>(cudaGetLastError());
        return false;
    }
    return deviceCount > 0;
}

} // namespace

TEST(KvCacheManagerPagedPoolTest, RejectsOverflowGeometryBeforeAllocation)
{
    rt::KVCacheManager::Config config;
    config.numAttentionLayers = 0;
    config.maxBatchSize = std::numeric_limits<int32_t>::max();
    config.maxSequenceLength = rt::kMAX_KV_CACHE_CAPACITY;
    config.kvCacheType = DataType::kHALF;
    EXPECT_THROW(rt::KVCacheManager(config, /*stream=*/nullptr), std::runtime_error);

    config.maxBatchSize = 1;
    config.maxSequenceLength = rt::kTOKENS_PER_PAGE;
    config.numPages = static_cast<int32_t>(rt::kMAX_KV_POOL_PAGES + 1);
    EXPECT_THROW(rt::KVCacheManager(config, /*stream=*/nullptr), std::runtime_error);
}

TEST(KvCacheManagerPagedPoolTest, SwaPageSizingCoversWindowPageBoundaries)
{
    int32_t const batch = 3;
    EXPECT_EQ(rt::computeSwaPrivatePagesPerSlot(64, rt::kTOKENS_PER_PAGE, rt::kSWA_REPLACEMENT_PAGES), 7);
    EXPECT_EQ(rt::computeSwaPrivatePagesPerSlot(128, rt::kTOKENS_PER_PAGE, rt::kSWA_REPLACEMENT_PAGES), 7);
    EXPECT_EQ(rt::computeSwaPrivatePagesPerSlot(256, rt::kTOKENS_PER_PAGE, rt::kSWA_REPLACEMENT_PAGES), 9);
    EXPECT_EQ(rt::computeSwaPrivatePagesPerSlot(257, rt::kTOKENS_PER_PAGE, rt::kSWA_REPLACEMENT_PAGES), 11);
    EXPECT_EQ(rt::computeSwaPrivatePagesPerSlot(
                  /*slidingWindowCapacity=*/5, /*tokensPerPage=*/4, /*replacementPageCount=*/3),
        11);
    EXPECT_EQ(rt::computeMinimumSwaPoolPages(batch, 257), batch * 11);
}

// maxSeq below (200) is deliberately NOT a multiple of 128 so the page count accounts for padding (-> 256).
TEST(KvCacheManagerPagedPoolTest, CombinedCacheIsPoolShapedWithPaddedCapacity)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 2;
    int32_t const maxSeq = 200; // not a multiple of 128 -> capPadded == 256
    int32_t const h0 = 4, d0 = 64;
    int32_t const h1 = 2, d1 = 128;

    rt::KVCacheManager mgr(makeHeteroConfig(maxBatch, maxSeq, h0, d0, h1, d1, DataType::kHALF), stream);

    int32_t const capPadded = padToPage(maxSeq);
    EXPECT_EQ(capPadded, 256);
    EXPECT_EQ(mgr.maxCapPadded(), capPadded);
    EXPECT_EQ(mgr.numPages(), maxBatch * capPadded / rt::kTOKENS_PER_PAGE);

    int32_t const heads[] = {h0, h1};
    int32_t const dims[] = {d0, d1};
    for (int32_t i = 0; i < 2; ++i)
    {
        auto const& shape = mgr.getCombinedKVCache(i).getShape();
        ASSERT_EQ(shape.getNumDims(), 5) << "layer " << i;
        EXPECT_EQ(shape[0], 2) << "layer " << i;
        EXPECT_EQ(shape[1], mgr.numPages()) << "layer " << i;
        EXPECT_EQ(shape[2], rt::kTOKENS_PER_PAGE) << "layer " << i;
        EXPECT_EQ(shape[3], heads[i]) << "layer " << i;
        EXPECT_EQ(shape[4], dims[i]) << "layer " << i;
    }
}

// kPoolPtr must be the combined buffer's base (the K half); vPoolPtr must start exactly
// numPages * kTOKENS_PER_PAGE * H_i * D_i elements later (the V half), per layer.
TEST(KvCacheManagerPagedPoolTest, PoolPointersAddressTwoContiguousHalves)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 2;
    int32_t const maxSeq = 200; // -> capPadded == 256
    int32_t const h0 = 4, d0 = 64;
    int32_t const h1 = 2, d1 = 128;

    rt::KVCacheManager mgr(makeHeteroConfig(maxBatch, maxSeq, h0, d0, h1, d1, DataType::kHALF), stream);

    size_t const elemSize = rt::utils::getTypeSize(DataType::kHALF);
    int32_t const heads[] = {h0, h1};
    int32_t const dims[] = {d0, d1};
    for (int32_t i = 0; i < 2; ++i)
    {
        rt::Tensor& combined = mgr.getCombinedKVCache(i);

        // K half starts at the combined buffer base.
        EXPECT_EQ(mgr.kPoolPtr(i), combined.rawPointer()) << "layer " << i;

        // V half starts numPages * kTOKENS_PER_PAGE * H_i * D_i elements later.
        int64_t const kCacheElems = static_cast<int64_t>(mgr.numPages()) * rt::kTOKENS_PER_PAGE * heads[i] * dims[i];
        auto const* expectedV
            = static_cast<char const*>(combined.rawPointer()) + kCacheElems * static_cast<int64_t>(elemSize);
        EXPECT_EQ(mgr.vPoolPtr(i), static_cast<void const*>(expectedV)) << "layer " << i;
    }
}

// FP8 pools have the same shape; only the element size (and thus the byte offset between
// kPoolPtr/vPoolPtr) changes. Verifies dtype is carried through and the half offset tracks it.
TEST(KvCacheManagerPagedPoolTest, Fp8PoolHasSameShapeAndCorrectByteOffset)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 3;
    int32_t const maxSeq = 130; // -> capPadded == 256
    int32_t const h = 2, d = 64;

    rt::KVCacheManager mgr(makeHeteroConfig(maxBatch, maxSeq, h, d, h, d, DataType::kFP8), stream);

    int32_t const capPadded = padToPage(maxSeq);
    EXPECT_EQ(capPadded, 256);
    EXPECT_EQ(mgr.maxCapPadded(), capPadded);
    EXPECT_EQ(mgr.numPages(), maxBatch * capPadded / rt::kTOKENS_PER_PAGE);

    rt::Tensor& combined = mgr.getCombinedKVCache(0);
    EXPECT_EQ(combined.getDataType(), DataType::kFP8);
    auto const& shape = combined.getShape();
    ASSERT_EQ(shape.getNumDims(), 5);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], mgr.numPages());
    EXPECT_EQ(shape[2], rt::kTOKENS_PER_PAGE);
    EXPECT_EQ(shape[3], h);
    EXPECT_EQ(shape[4], d);

    size_t const elemSize = rt::utils::getTypeSize(DataType::kFP8); // 1 byte
    int64_t const kCacheElems = static_cast<int64_t>(mgr.numPages()) * rt::kTOKENS_PER_PAGE * h * d;
    auto const* expectedV
        = static_cast<char const*>(combined.rawPointer()) + kCacheElems * static_cast<int64_t>(elemSize);
    EXPECT_EQ(mgr.kPoolPtr(0), combined.rawPointer());
    EXPECT_EQ(mgr.vPoolPtr(0), static_cast<void const*>(expectedV));
}

// Config::numPages may include extra retained pages beyond the minimum active pages.
TEST(KvCacheManagerPagedPoolTest, ExtraRetainedPagesAllocateAndStayConsistent)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 2;
    int32_t const maxSeq = 200; // -> capPadded == 256, minimum active pages == maxBatch * 2 == 4
    int32_t const h = 4, d = 64;

    int32_t const minimumActivePages = maxBatch * padToPage(maxSeq) / rt::kTOKENS_PER_PAGE;
    ASSERT_EQ(minimumActivePages, 4);
    int32_t const extraRetainedPages = 6;
    int32_t const requestedPages = minimumActivePages + extraRetainedPages;

    rt::KVCacheManager::Config config = makeHeteroConfig(maxBatch, maxSeq, h, d, h, d, DataType::kHALF);
    config.numPages = requestedPages;
    rt::KVCacheManager mgr(config, stream);

    EXPECT_EQ(mgr.numPages(), requestedPages);
    EXPECT_EQ(mgr.maxCapPadded(), padToPage(maxSeq));
    for (int32_t i = 0; i < 2; ++i)
    {
        auto const& poolShape = mgr.getCombinedKVCache(i).getShape();
        ASSERT_EQ(poolShape.getNumDims(), 5) << "layer " << i;
        EXPECT_EQ(poolShape[0], 2) << "layer " << i;
        EXPECT_EQ(poolShape[1], requestedPages) << "layer " << i;
        EXPECT_EQ(poolShape[2], rt::kTOKENS_PER_PAGE) << "layer " << i;
        EXPECT_EQ(poolShape[3], h) << "layer " << i;
        EXPECT_EQ(poolShape[4], d) << "layer " << i;

        // vPoolPtr is offset from kPoolPtr by the configured numPages, including extra retained pages.
        size_t const elemSize = rt::utils::getTypeSize(DataType::kHALF);
        int64_t const kCacheElems = static_cast<int64_t>(requestedPages) * rt::kTOKENS_PER_PAGE * h * d;
        auto const* expectedV
            = static_cast<char const*>(mgr.kPoolPtr(i)) + kCacheElems * static_cast<int64_t>(elemSize);
        EXPECT_EQ(mgr.vPoolPtr(i), static_cast<void const*>(expectedV)) << "layer " << i;

        auto [kView, vView] = mgr.getSeparateKVCache(i);
        EXPECT_EQ(kView.rawPointer(), mgr.kPoolPtr(i)) << "layer " << i;
        EXPECT_EQ(vView.rawPointer(), mgr.vPoolPtr(i)) << "layer " << i;
        auto const& kShape = kView.getShape();
        ASSERT_EQ(kShape.getNumDims(), 4) << "layer " << i;
        EXPECT_EQ(kShape[0], maxBatch) << "layer " << i;
        EXPECT_EQ(kShape[1], padToPage(maxSeq)) << "layer " << i;
    }
}

TEST(KvCacheManagerPagedPoolTest, NumPagesOverrideBelowMinimumActivePagesRejected)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 2;
    int32_t const maxSeq = 200; // minimum active pages == 4
    rt::KVCacheManager::Config config = makeHeteroConfig(maxBatch, maxSeq, 4, 64, 4, 64, DataType::kHALF);
    config.numPages = 1; // below the minimum active page count of 4
    EXPECT_THROW(rt::KVCacheManager mgr(config, stream), std::exception);
}

TEST(KvCacheManagerPagedPoolTest, FullAndSwaLayersExposeSeparatePhysicalCounts)
{
    cudaStream_t stream{nullptr};
    int32_t const maxBatch = 2;
    int32_t const maxSeq = 8192;
    int32_t const window = 129;
    int32_t const configuredSwaPages = 64;
    std::vector<rt::KVLayerConfig> layers{
        rt::KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8},
        rt::KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8, /*kvCacheCapacity=*/window},
    };
    rt::KVCacheManager::Config config{/*numAttentionLayers=*/2, maxBatch, maxSeq, layers, DataType::kHALF};
    config.numSwaPages = configuredSwaPages;
    rt::KVCacheManager mgr(config, stream);

    int32_t const fullPages = static_cast<int32_t>(rt::computeMinimumKvPoolPages(maxBatch, maxSeq));
    int32_t const swaPages = configuredSwaPages;
    EXPECT_EQ(mgr.numPages(), fullPages); // Legacy no-argument API remains the full pool.
    EXPECT_EQ(mgr.numPages(0), fullPages);
    EXPECT_EQ(mgr.numPages(1), swaPages);
    EXPECT_EQ(mgr.maxCapPadded(), maxSeq);
    EXPECT_EQ(mgr.maxCapPadded(0), maxSeq);
    EXPECT_EQ(mgr.maxCapPadded(1),
        rt::computeSwaPrivatePagesPerSlot(window, rt::kTOKENS_PER_PAGE, rt::kSWA_REPLACEMENT_PAGES)
            * rt::kTOKENS_PER_PAGE);
    ASSERT_TRUE(mgr.reducedKVCacheCapacity().has_value());
    EXPECT_EQ(*mgr.reducedKVCacheCapacity(), window);

    auto const& fullShape = mgr.getCombinedKVCache(0).getShape();
    auto const& swaShape = mgr.getCombinedKVCache(1).getShape();
    EXPECT_EQ(fullShape[1], fullPages);
    EXPECT_EQ(swaShape[1], swaPages);
    EXPECT_EQ(swaShape[2], rt::kTOKENS_PER_PAGE);
    auto const [swaK, swaV] = mgr.getSeparateKVCache(1);
    EXPECT_EQ(swaK.getShape()[0], swaPages);
    EXPECT_EQ(swaV.getShape()[0], swaPages);

    size_t const elemSize = rt::utils::getTypeSize(DataType::kHALF);
    int64_t const swaHalfElements = static_cast<int64_t>(swaPages) * rt::kTOKENS_PER_PAGE * 8;
    auto const* expectedSwaV
        = static_cast<char const*>(mgr.kPoolPtr(1)) + swaHalfElements * static_cast<int64_t>(elemSize);
    EXPECT_EQ(mgr.vPoolPtr(1), static_cast<void const*>(expectedSwaV));
}

TEST(KvCacheManagerPagedPoolTest, FullRuntimeModePreservesMarkerButAllocatesFullPhysicalCount)
{
    cudaStream_t stream{nullptr};
    constexpr int32_t kMAX_BATCH = 2;
    constexpr int32_t kMAX_SEQ = 8192;
    constexpr int32_t kWINDOW = 129;
    std::vector<rt::KVLayerConfig> layers{
        rt::KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8},
        rt::KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8, /*kvCacheCapacity=*/kWINDOW},
    };
    rt::KVCacheManager::Config config{/*numAttentionLayers=*/2, kMAX_BATCH, kMAX_SEQ, layers, DataType::kHALF};
    config.numSwaPages = static_cast<int32_t>(rt::computeMinimumSwaPoolPages(kMAX_BATCH, kWINDOW));
    config.useBoundedSwaKVCache = false;
    rt::KVCacheManager mgr(config, stream);

    int32_t const fullPages = static_cast<int32_t>(rt::computeMinimumKvPoolPages(kMAX_BATCH, kMAX_SEQ));
    EXPECT_FALSE(mgr.hasReducedKVCache());
    EXPECT_EQ(mgr.numPages(0), fullPages);
    EXPECT_EQ(mgr.numPages(1), fullPages);
    EXPECT_EQ(mgr.getLayerConfig(1).kvCacheCapacity, kWINDOW);
    auto const [kView, vView] = mgr.getSeparateKVCache(1);
    EXPECT_EQ(kView.getShape()[0], kMAX_BATCH);
    EXPECT_EQ(kView.getShape()[1], kMAX_SEQ);
    EXPECT_EQ(vView.getShape()[0], kMAX_BATCH);
}

TEST(KvCacheManagerPagedPoolTest, MixedReducedWindowsAreRejected)
{
    cudaStream_t stream{nullptr};
    std::vector<rt::KVLayerConfig> layers{
        rt::KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8, /*kvCacheCapacity=*/64},
        rt::KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8, /*kvCacheCapacity=*/128},
    };
    rt::KVCacheManager::Config config{/*numAttentionLayers=*/2, /*maxBatchSize=*/1,
        /*maxSequenceLength=*/512, layers, DataType::kHALF};
    EXPECT_THROW(rt::KVCacheManager mgr(config, stream), std::exception);
}

TEST(KvCacheManagerPagedPoolTest, ReducedFp8PoolUsesBoundedGeometryAndOneByteOffsets)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "CUDA device required for FP8 bounded KV allocation geometry.";
    }

    cudaStream_t stream{nullptr};
    constexpr int32_t kMAX_BATCH_SIZE = 2;
    constexpr int32_t kMAX_SEQUENCE_LENGTH = 8192;
    constexpr int32_t kSWA_WINDOW_SIZE = 129;
    constexpr int32_t kNUM_KV_HEADS = 1;
    constexpr int32_t kHEAD_DIM = 8;

    std::vector<rt::KVLayerConfig> layers{
        rt::KVLayerConfig{/*numKVHeads=*/kNUM_KV_HEADS, /*headDim=*/kHEAD_DIM},
        rt::KVLayerConfig{/*numKVHeads=*/kNUM_KV_HEADS, /*headDim=*/kHEAD_DIM,
            /*kvCacheCapacity=*/kSWA_WINDOW_SIZE},
    };
    rt::KVCacheManager::Config config{
        /*numAttentionLayers=*/2, kMAX_BATCH_SIZE, kMAX_SEQUENCE_LENGTH, layers, DataType::kFP8};
    config.numSwaPages = static_cast<int32_t>(rt::computeMinimumSwaPoolPages(kMAX_BATCH_SIZE, kSWA_WINDOW_SIZE));
    int32_t const fullPages
        = static_cast<int32_t>(rt::computeMinimumKvPoolPages(kMAX_BATCH_SIZE, kMAX_SEQUENCE_LENGTH));
    int32_t const boundedCapPadded = static_cast<int32_t>(
        rt::computeSwaPrivatePagesPerSlot(kSWA_WINDOW_SIZE, rt::kTOKENS_PER_PAGE, rt::kSWA_REPLACEMENT_PAGES)
        * rt::kTOKENS_PER_PAGE);

    rt::KVCacheManager boundedMgr(config, stream);
    EXPECT_TRUE(boundedMgr.hasReducedKVCache());
    ASSERT_TRUE(boundedMgr.reducedKVCacheCapacity().has_value());
    EXPECT_EQ(*boundedMgr.reducedKVCacheCapacity(), kSWA_WINDOW_SIZE);
    EXPECT_EQ(boundedMgr.numPages(0), fullPages);
    EXPECT_EQ(boundedMgr.numPages(1), config.numSwaPages);
    EXPECT_EQ(boundedMgr.maxCapPadded(0), kMAX_SEQUENCE_LENGTH);
    EXPECT_EQ(boundedMgr.maxCapPadded(1), boundedCapPadded);
    EXPECT_EQ(boundedMgr.getCombinedKVCache(1).getDataType(), DataType::kFP8);

    auto const [boundedK, boundedV] = boundedMgr.getSeparateKVCache(1);
    EXPECT_EQ(boundedK.getDataType(), DataType::kFP8);
    EXPECT_EQ(boundedV.getDataType(), DataType::kFP8);
    EXPECT_EQ(boundedK.getShape()[0], config.numSwaPages);
    EXPECT_EQ(boundedK.getShape()[1], rt::kTOKENS_PER_PAGE);
    EXPECT_EQ(boundedV.getShape()[0], config.numSwaPages);
    size_t const fp8ElemSize = rt::utils::getTypeSize(DataType::kFP8);
    EXPECT_EQ(fp8ElemSize, 1U);
    int64_t const boundedPlaneElements
        = static_cast<int64_t>(config.numSwaPages) * rt::kTOKENS_PER_PAGE * kNUM_KV_HEADS * kHEAD_DIM;
    auto const* expectedBoundedV
        = static_cast<char const*>(boundedMgr.kPoolPtr(1)) + boundedPlaneElements * static_cast<int64_t>(fp8ElemSize);
    EXPECT_EQ(boundedMgr.vPoolPtr(1), static_cast<void const*>(expectedBoundedV));

    config.useBoundedSwaKVCache = false;
    rt::KVCacheManager fullModeMgr(config, stream);
    EXPECT_FALSE(fullModeMgr.hasReducedKVCache());
    EXPECT_EQ(fullModeMgr.getLayerConfig(1).kvCacheCapacity, kSWA_WINDOW_SIZE);
    EXPECT_EQ(fullModeMgr.numPages(1), fullPages);
    EXPECT_EQ(fullModeMgr.getCombinedKVCache(1).getDataType(), DataType::kFP8);
    auto const [fullModeK, fullModeV] = fullModeMgr.getSeparateKVCache(1);
    EXPECT_EQ(fullModeK.getShape()[0], kMAX_BATCH_SIZE);
    EXPECT_EQ(fullModeK.getShape()[1], kMAX_SEQUENCE_LENGTH);
    EXPECT_EQ(fullModeV.getShape()[0], kMAX_BATCH_SIZE);
    int64_t const fullModePlaneElements
        = static_cast<int64_t>(fullPages) * rt::kTOKENS_PER_PAGE * kNUM_KV_HEADS * kHEAD_DIM;
    auto const* expectedFullModeV
        = static_cast<char const*>(fullModeMgr.kPoolPtr(1)) + fullModePlaneElements * static_cast<int64_t>(fp8ElemSize);
    EXPECT_EQ(fullModeMgr.vPoolPtr(1), static_cast<void const*>(expectedFullModeV));
}

TEST(KvCacheManagerPagedPoolTest, ReducedPoolRequiresExplicitSwaPageBudget)
{
    cudaStream_t stream{nullptr};
    std::vector<rt::KVLayerConfig> layers{
        rt::KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8, /*kvCacheCapacity=*/129},
    };
    rt::KVCacheManager::Config config{/*numAttentionLayers=*/1, /*maxBatchSize=*/2,
        /*maxSequenceLength=*/8192, layers, DataType::kHALF};
    EXPECT_THROW(rt::KVCacheManager mgr(config, stream), std::exception);

    config.numSwaPages
        = static_cast<int32_t>(rt::computeMinimumSwaPoolPages(/*maxBatchSize=*/2, /*slidingWindowCapacity=*/129) - 1);
    EXPECT_THROW(rt::KVCacheManager mgr(config, stream), std::exception);
}
