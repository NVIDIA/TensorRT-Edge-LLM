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

// Unit tests for the paged-KV pool relayout of KVCacheManager.
//
// Each attention layer's combined KV buffer is allocated slot-shaped as
//     [2, maxBatchSize, capPadded, numKVHeads_i, headDim_i]
// (token-major NHD within a slot, K/V split outermost), where
//     capPadded = ceil(maxSequenceLength / kTOKENS_PER_PAGE) * kTOKENS_PER_PAGE.
//
// The same buffer is a bind-time reinterpretation of a page pool
//     [2, numPages, kTOKENS_PER_PAGE, numKVHeads_i, headDim_i]   (numPages = maxBatchSize * capPadded /
//     kTOKENS_PER_PAGE)
// with kPoolPtr/vPoolPtr the two contiguous halves.

#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "runtime/kvCacheManager.h"
#include <gtest/gtest.h>

using namespace trt_edgellm;
using namespace nvinfer1;

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

} // namespace

// The combined per-layer buffer is slot-shaped NHD [2, maxBatch, capPadded, H_i, D_i]. maxSeq
// below (200) is deliberately NOT a multiple of 128 so the padding (-> 256) is exercised.
TEST(KvCacheManagerPagedPoolTest, CombinedCacheIsSlotShapedWithPaddedCapacity)
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
        EXPECT_EQ(shape[0], 2) << "layer " << i;         // K/V split outermost
        EXPECT_EQ(shape[1], maxBatch) << "layer " << i;  // batch
        EXPECT_EQ(shape[2], capPadded) << "layer " << i; // padded capacity (token-major)
        EXPECT_EQ(shape[3], heads[i]) << "layer " << i;  // numKVHeads
        EXPECT_EQ(shape[4], dims[i]) << "layer " << i;   // headDim
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
    EXPECT_EQ(shape[1], maxBatch);
    EXPECT_EQ(shape[2], capPadded);
    EXPECT_EQ(shape[3], h);
    EXPECT_EQ(shape[4], d);

    size_t const elemSize = rt::utils::getTypeSize(DataType::kFP8); // 1 byte
    int64_t const kCacheElems = static_cast<int64_t>(mgr.numPages()) * rt::kTOKENS_PER_PAGE * h * d;
    auto const* expectedV
        = static_cast<char const*>(combined.rawPointer()) + kCacheElems * static_cast<int64_t>(elemSize);
    EXPECT_EQ(mgr.kPoolPtr(0), combined.rawPointer());
    EXPECT_EQ(mgr.vPoolPtr(0), static_cast<void const*>(expectedV));
}

// Robustness: Config::numPages may request a pool larger than the
// active-capacity floor. numPages()/getCombinedKVCachePoolView() must reflect the configured
// (larger) value; the "live" slot-shaped view (getSeparateKVCache/kPoolPtr/vPoolPtr) stays
// correctly addressed (V-half starts at the real numPages, not the floor).
TEST(KvCacheManagerPagedPoolTest, RetentionPagesAllocateBeyondFloorAndStayConsistent)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 2;
    int32_t const maxSeq = 200; // -> capPadded == 256, floor == maxBatch * 2 == 4
    int32_t const h = 4, d = 64;

    int32_t const floorPages = maxBatch * padToPage(maxSeq) / rt::kTOKENS_PER_PAGE;
    ASSERT_EQ(floorPages, 4);
    int32_t const extraRetentionPages = 6;
    int32_t const requestedPages = floorPages + extraRetentionPages;

    rt::KVCacheManager::Config config = makeHeteroConfig(maxBatch, maxSeq, h, d, h, d, DataType::kHALF);
    config.numPages = requestedPages;
    rt::KVCacheManager mgr(config, stream);

    EXPECT_EQ(mgr.numPages(), requestedPages);
    EXPECT_EQ(mgr.maxCapPadded(), padToPage(maxSeq));

    for (int32_t i = 0; i < 2; ++i)
    {
        // Pool-view binding dims reflect the requested (larger) page count.
        auto const& poolShape = mgr.getCombinedKVCachePoolView(i).getShape();
        ASSERT_EQ(poolShape.getNumDims(), 5) << "layer " << i;
        EXPECT_EQ(poolShape[0], 2) << "layer " << i;
        EXPECT_EQ(poolShape[1], requestedPages) << "layer " << i;
        EXPECT_EQ(poolShape[2], rt::kTOKENS_PER_PAGE) << "layer " << i;
        EXPECT_EQ(poolShape[3], h) << "layer " << i;
        EXPECT_EQ(poolShape[4], d) << "layer " << i;

        // vPoolPtr is offset by the REAL numPages (not the floor) from kPoolPtr.
        size_t const elemSize = rt::utils::getTypeSize(DataType::kHALF);
        int64_t const kCacheElems = static_cast<int64_t>(requestedPages) * rt::kTOKENS_PER_PAGE * h * d;
        auto const* expectedV
            = static_cast<char const*>(mgr.kPoolPtr(i)) + kCacheElems * static_cast<int64_t>(elemSize);
        EXPECT_EQ(mgr.vPoolPtr(i), static_cast<void const*>(expectedV)) << "layer " << i;

        // getSeparateKVCache still returns the floor-sized "live" slot view, correctly based at the
        // (offset) kPoolPtr/vPoolPtr rather than a floor-based V offset.
        auto [kView, vView] = mgr.getSeparateKVCache(i);
        EXPECT_EQ(kView.rawPointer(), mgr.kPoolPtr(i)) << "layer " << i;
        EXPECT_EQ(vView.rawPointer(), mgr.vPoolPtr(i)) << "layer " << i;
        auto const& kShape = kView.getShape();
        ASSERT_EQ(kShape.getNumDims(), 4) << "layer " << i;
        EXPECT_EQ(kShape[0], maxBatch) << "layer " << i;
        EXPECT_EQ(kShape[1], padToPage(maxSeq)) << "layer " << i;
    }
}

TEST(KvCacheManagerPagedPoolTest, NumPagesOverrideBelowFloorRejected)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 2;
    int32_t const maxSeq = 200; // floor == 4
    rt::KVCacheManager::Config config = makeHeteroConfig(maxBatch, maxSeq, 4, 64, 4, 64, DataType::kHALF);
    config.numPages = 1; // below the floor of 4
    EXPECT_THROW(rt::KVCacheManager mgr(config, stream), std::exception);
}
