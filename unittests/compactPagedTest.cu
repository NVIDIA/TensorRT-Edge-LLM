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

// Paged-pool compaction round-trip: HybridCacheManager::compactBatch must move each survivor's LIVE
// PREFIX (not the full capPadded*H*D row) in BOTH the K-half and the V-half of the NHD pool
// [2, maxBatch, capPadded, H, D], keeping row == physical slot (Phase-1 compaction discipline,
// identity page table unchanged — the remap step arrives with reuse in MR 3).

#include "common/cudaUtils.h"
#include "runtime/hybridCacheManager.h"
#include "testUtils.h"
#include <cstring>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

// Fill tokens [startTok, endTok) of row `b` of a half-view [maxBatch, capPadded, H, D], leaving the
// rest of the row untouched.
void fillRowRange(rt::Tensor& halfView, int32_t b, int32_t startTok, int32_t endTok, float value)
{
    auto const& shape = halfView.getShape();
    int64_t const tokenStride = shape[2] * shape[3]; // H * D
    int64_t const capPadded = shape[1];
    int64_t const numTok = endTok - startTok;
    std::vector<half> host(static_cast<size_t>(numTok * tokenStride), __float2half(value));
    int64_t const off
        = static_cast<int64_t>(b) * capPadded * tokenStride + static_cast<int64_t>(startTok) * tokenStride;
    CUDA_CHECK(cudaMemcpy(static_cast<half*>(halfView.rawPointer()) + off, host.data(), host.size() * sizeof(half),
        cudaMemcpyHostToDevice));
}

// Assert tokens [startTok, endTok) of row `b` of a half-view all equal `expected`.
void expectRowRangeEq(
    rt::Tensor const& halfView, int32_t b, int32_t startTok, int32_t endTok, float expected, std::string const& what)
{
    auto const& shape = halfView.getShape();
    int64_t const tokenStride = shape[2] * shape[3]; // H * D
    int64_t const capPadded = shape[1];
    int64_t const numTok = endTok - startTok;
    std::vector<half> host(static_cast<size_t>(numTok * tokenStride));
    int64_t const off
        = static_cast<int64_t>(b) * capPadded * tokenStride + static_cast<int64_t>(startTok) * tokenStride;
    CUDA_CHECK(cudaMemcpy(host.data(), static_cast<half const*>(halfView.rawPointer()) + off,
        host.size() * sizeof(half), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < host.size(); ++i)
    {
        ASSERT_TRUE(isclose(host[i], __float2half(expected), 1e-2f, 1e-2f))
            << what << ": row=" << b << " tok=" << (startTok + i / tokenStride) << " got=" << __half2float(host[i])
            << " expected=" << expected;
    }
}

rt::Tensor uploadMapping(std::vector<int32_t> const& mapping)
{
    rt::Tensor t({static_cast<int32_t>(mapping.size())}, rt::DeviceType::kGPU, DataType::kINT32, "batchMapping");
    CUDA_CHECK(cudaMemcpy(t.rawPointer(), mapping.data(), mapping.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    return t;
}

rt::HybridCacheManager::Config makeUniformAttnConfig(
    int32_t numLayers, int32_t maxBatch, int32_t maxSeq, int32_t numKVHeads, int32_t headDim)
{
    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(numLayers, rt::HybridCacheManager::LayerType::kAttention);
    std::vector<rt::KVLayerConfig> layers(numLayers, rt::KVLayerConfig{numKVHeads, headDim});
    cfg.kvConfig = rt::KVCacheManager::Config{numLayers, maxBatch, maxSeq, layers, DataType::kHALF};
    // Zero Mamba layers.
    cfg.mambaConfig = rt::MambaCacheManager::Config{};
    cfg.mambaConfig.numRecurrentLayers = 0;
    cfg.mambaConfig.maxBatchSize = maxBatch;
    cfg.maxBatchSize = maxBatch;
    return cfg;
}

} // namespace

// Evict one slot; survivors must compact densely in BOTH K and V halves, with K and V carrying
// distinct values so a half-swap or single-half copy would be caught.
TEST(CompactPagedTest, CompactMovesLivePrefixInBothHalves)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 8;
    int32_t const oldBatch = 4;
    int32_t const newBatch = 2;
    int32_t const numLayers = 2;
    int32_t const maxSeq = 32; // capPadded -> 128
    int32_t const numKVHeads = 2;
    int32_t const headDim = 64;

    rt::HybridCacheManager mgr(makeUniformAttnConfig(numLayers, maxBatch, maxSeq, numKVHeads, headDim), stream);

    // Sentinel shape check: NHD [2, maxBatch, capPadded, H, D].
    {
        auto const& shape = mgr.getCombinedKVCache(0).getShape();
        ASSERT_EQ(shape[0], 2);
        ASSERT_EQ(shape[1], maxBatch);
        ASSERT_EQ(shape[2], 128); // capPadded
        ASSERT_EQ(shape[3], numKVHeads);
        ASSERT_EQ(shape[4], headDim);
    }

    // Seed the live prefix [0, maxSeq) of each row: K-half row b -> Lbase + b; V-half row b ->
    // Lbase + 1000 + b (distinct from K). Compaction only moves the live prefix (per the KV lengths
    // fed via resetForNewSequences below), so only that range is seeded and checked.
    auto kVal = [](int32_t L, int32_t b) { return static_cast<float>((L + 1) * 10 + b); };
    auto vVal = [](int32_t L, int32_t b) { return static_cast<float>((L + 1) * 10 + 1000 + b); };
    for (int32_t L = 0; L < numLayers; ++L)
    {
        auto [k, v] = mgr.getSeparateKVCache(L);
        for (int32_t b = 0; b < oldBatch; ++b)
        {
            fillRowRange(k, b, 0, maxSeq, kVal(L, b));
            fillRowRange(v, b, 0, maxSeq, vVal(L, b));
        }
    }

    // Live length per slot (equal to maxSeq here) drives how much of each row compaction copies.
    std::vector<int32_t> hostLens(oldBatch, maxSeq);
    rt::Tensor reuseLens({oldBatch}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(reuseLens.rawPointer(), hostLens.data(), hostLens.size() * sizeof(int32_t));
    mgr.resetForNewSequences(reuseLens, stream);

    // Keep old slot 1 -> new 0, old slot 3 -> new 1; evict 0 and 2.
    auto mapping = uploadMapping({-1, 0, -1, 1});
    mgr.compactBatch(mapping, oldBatch, newBatch, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t L = 0; L < numLayers; ++L)
    {
        auto [k, v] = mgr.getSeparateKVCache(L);
        // new 0 <- old 1, new 1 <- old 3, in both halves with their own value spaces, over the live
        // prefix [0, maxSeq).
        expectRowRangeEq(k, 0, 0, maxSeq, kVal(L, 1), "K L=" + std::to_string(L) + " new0");
        expectRowRangeEq(k, 1, 0, maxSeq, kVal(L, 3), "K L=" + std::to_string(L) + " new1");
        expectRowRangeEq(v, 0, 0, maxSeq, vVal(L, 1), "V L=" + std::to_string(L) + " new0");
        expectRowRangeEq(v, 1, 0, maxSeq, vVal(L, 3), "V L=" + std::to_string(L) + " new1");
    }
}
