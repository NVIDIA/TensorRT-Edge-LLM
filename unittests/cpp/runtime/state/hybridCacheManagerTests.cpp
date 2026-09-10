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
#include "runtime/hybridCacheManager.h"
#include "testUtils.h"
#include <cuda_fp16.h>
#include <gtest/gtest.h>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

// Fill every element of one batch slot with a given half value.
void fillSlotHalf(rt::Tensor& tensor, int32_t batchIdx, float value)
{
    auto const& shape = tensor.getShape();
    int64_t batchStride = 1;
    for (int32_t d = 1; d < shape.getNumDims(); ++d)
    {
        batchStride *= shape[d];
    }
    std::vector<half> host(static_cast<size_t>(batchStride), __float2half(value));
    int64_t const elemOffset = static_cast<int64_t>(batchIdx) * batchStride;
    CUDA_CHECK(cudaMemcpy(static_cast<half*>(tensor.rawPointer()) + elemOffset, host.data(),
        static_cast<size_t>(batchStride) * sizeof(half), cudaMemcpyHostToDevice));
}

// Read one batch slot back into a host vector of half.
std::vector<half> readSlotHalf(rt::Tensor const& tensor, int32_t batchIdx)
{
    auto const& shape = tensor.getShape();
    int64_t batchStride = 1;
    for (int32_t d = 1; d < shape.getNumDims(); ++d)
    {
        batchStride *= shape[d];
    }
    std::vector<half> host(static_cast<size_t>(batchStride));
    int64_t const elemOffset = static_cast<int64_t>(batchIdx) * batchStride;
    CUDA_CHECK(cudaMemcpy(host.data(), static_cast<half const*>(tensor.rawPointer()) + elemOffset,
        static_cast<size_t>(batchStride) * sizeof(half), cudaMemcpyDeviceToHost));
    return host;
}

// Assert every element in the slot equals `expected` (half tolerance).
void expectSlotEqHalf(rt::Tensor const& tensor, int32_t batchIdx, float expected, std::string const& what)
{
    auto const host = readSlotHalf(tensor, batchIdx);
    for (size_t i = 0; i < host.size(); ++i)
    {
        ASSERT_TRUE(isclose(host[i], __float2half(expected), 1e-2f, 1e-2f))
            << what << ": slot=" << batchIdx << " elem=" << i << " got=" << __half2float(host[i])
            << " expected=" << expected;
    }
}

// --- NHD-aware slot helpers --------------------------------------------------
//
// A batch slot spans row `b` of the K half and row `b` of the V half, which live one full
// half-pool apart.
// getSeparateKVCache returns the K-half and V-half as [maxBatch, capPadded, H, D] views whose
// dim 0 IS batch, so the plain fill/read/expect helpers above work directly on each view.

// Fill row `b` (both K and V halves) of an attention layer with `value`.
void fillSlotNhd(rt::HybridCacheManager& mgr, int32_t absLayer, int32_t b, float value)
{
    auto [k, v] = mgr.getSeparateKVCache(absLayer);
    fillSlotHalf(k, b, value);
    fillSlotHalf(v, b, value);
}

// Fill tokens [startTok, endTok) of row `b` within a K-or-V half view [maxBatch, capPadded, H, D],
// leaving the rest of the row untouched.
void fillSlotTokenRangeHalf(rt::Tensor& tensor, int32_t batchIdx, int32_t startTok, int32_t endTok, float value)
{
    auto const& shape = tensor.getShape();
    int64_t const tokenStride = shape[2] * shape[3]; // H * D
    int64_t const capPadded = shape[1];
    int64_t const numTok = endTok - startTok;
    std::vector<half> host(static_cast<size_t>(numTok * tokenStride), __float2half(value));
    int64_t const elemOffset
        = static_cast<int64_t>(batchIdx) * capPadded * tokenStride + static_cast<int64_t>(startTok) * tokenStride;
    CUDA_CHECK(cudaMemcpy(static_cast<half*>(tensor.rawPointer()) + elemOffset, host.data(), host.size() * sizeof(half),
        cudaMemcpyHostToDevice));
}

// Assert tokens [startTok, endTok) of row `b` within a K-or-V half view all equal `expected`.
void expectSlotTokenRangeEqHalf(rt::Tensor const& tensor, int32_t batchIdx, int32_t startTok, int32_t endTok,
    float expected, std::string const& what)
{
    auto const& shape = tensor.getShape();
    int64_t const tokenStride = shape[2] * shape[3]; // H * D
    int64_t const capPadded = shape[1];
    int64_t const numTok = endTok - startTok;
    std::vector<half> host(static_cast<size_t>(numTok * tokenStride));
    int64_t const elemOffset
        = static_cast<int64_t>(batchIdx) * capPadded * tokenStride + static_cast<int64_t>(startTok) * tokenStride;
    CUDA_CHECK(cudaMemcpy(host.data(), static_cast<half const*>(tensor.rawPointer()) + elemOffset,
        host.size() * sizeof(half), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < host.size(); ++i)
    {
        ASSERT_TRUE(isclose(host[i], __float2half(expected), 1e-2f, 1e-2f))
            << what << ": slot=" << batchIdx << " tok=" << (startTok + i / tokenStride)
            << " got=" << __half2float(host[i]) << " expected=" << expected;
    }
}

// Fill tokens [startTok, endTok) of row `b` (both K and V halves) of an attention layer.
void fillSlotTokenRangeNhd(
    rt::HybridCacheManager& mgr, int32_t absLayer, int32_t b, int32_t startTok, int32_t endTok, float value)
{
    auto [k, v] = mgr.getSeparateKVCache(absLayer);
    fillSlotTokenRangeHalf(k, b, startTok, endTok, value);
    fillSlotTokenRangeHalf(v, b, startTok, endTok, value);
}

// Assert tokens [startTok, endTok) of row `b` (both K and V halves) of an attention layer equal `value`.
void expectSlotTokenRangeEqNhd(rt::HybridCacheManager& mgr, int32_t absLayer, int32_t b, int32_t startTok,
    int32_t endTok, float value, std::string const& what)
{
    auto [k, v] = mgr.getSeparateKVCache(absLayer);
    expectSlotTokenRangeEqHalf(k, b, startTok, endTok, value, what + " [K]");
    expectSlotTokenRangeEqHalf(v, b, startTok, endTok, value, what + " [V]");
}

// Upload an int32 batch mapping to a GPU tensor of shape [oldActiveBatch].
rt::Tensor uploadMapping(std::vector<int32_t> const& mapping)
{
    rt::Tensor t({static_cast<int32_t>(mapping.size())}, rt::DeviceType::kGPU, DataType::kINT32, "batchMapping");
    CUDA_CHECK(cudaMemcpy(t.rawPointer(), mapping.data(), mapping.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    return t;
}

// Build a uniform KV config for all-attention models.
rt::KVCacheManager::Config makeUniformKVConfig(
    int32_t numLayers, int32_t maxBatch, int32_t maxSeq, int32_t numKVHeads, int32_t headDim)
{
    std::vector<rt::KVLayerConfig> layers(numLayers, rt::KVLayerConfig{numKVHeads, headDim});
    return rt::KVCacheManager::Config{numLayers, maxBatch, maxSeq, layers, DataType::kHALF};
}

// Build a heterogeneous KV config where the first half uses (h0, d0) and the second half uses (h1, d1).
rt::KVCacheManager::Config makeHeteroKVConfig(
    int32_t numLayers, int32_t maxBatch, int32_t maxSeq, int32_t h0, int32_t d0, int32_t h1, int32_t d1)
{
    std::vector<rt::KVLayerConfig> layers;
    layers.reserve(numLayers);
    for (int32_t i = 0; i < numLayers; ++i)
    {
        if (i < numLayers / 2)
        {
            layers.push_back({h0, d0});
        }
        else
        {
            layers.push_back({h1, d1});
        }
    }
    return rt::KVCacheManager::Config{numLayers, maxBatch, maxSeq, layers, DataType::kHALF};
}

rt::MambaCacheManager::Config makeMambaConfig(int32_t numLayers, int32_t maxBatch)
{
    rt::MambaCacheManager::Config cfg{};
    cfg.numRecurrentLayers = numLayers;
    cfg.maxBatchSize = maxBatch;
    cfg.recurrentStateNumHeads = 4;
    cfg.recurrentStateHeadDim = 16;
    cfg.recurrentStateSize = 8;
    cfg.convDim = 32;
    cfg.convKernel = 3;
    cfg.recurrentStateType = DataType::kHALF;
    cfg.convStateType = DataType::kHALF;
    return cfg;
}

} // namespace

// --- Routing ----------------------------------------------------------------

TEST(HybridCacheManagerTests, RoutingUniformKV)
{
    cudaStream_t stream{nullptr};

    int32_t const numLayers = 4;
    int32_t const maxBatch = 2;
    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(numLayers, rt::HybridCacheManager::LayerType::kAttention);
    cfg.kvConfig = makeUniformKVConfig(numLayers, maxBatch, 128, 4, 64);
    cfg.mambaConfig = makeMambaConfig(0, maxBatch);
    cfg.maxBatchSize = maxBatch;

    rt::HybridCacheManager mgr(cfg, stream);

    // Paged-pool layout [2, numPages, kTOKENS_PER_PAGE, H, D] with the K/V split outermost.
    for (int32_t i = 0; i < numLayers; ++i)
    {
        auto& t = mgr.getCombinedKVCache(i);
        EXPECT_EQ(t.getShape()[0], 2); // K/V split outermost
        EXPECT_EQ(t.getShape()[1], mgr.getKVCacheManager().numPages());
        EXPECT_EQ(t.getShape()[2], rt::kTOKENS_PER_PAGE);
        EXPECT_EQ(t.getShape()[3], 4);  // numKVHeads
        EXPECT_EQ(t.getShape()[4], 64); // headDim
    }
    EXPECT_THROW((void) mgr.getCombinedKVCache(-1), std::runtime_error);
    EXPECT_THROW((void) mgr.getCombinedKVCache(numLayers), std::runtime_error);
    // No Mamba layers exist — any absLayerIdx is not a Mamba layer.
    EXPECT_THROW((void) mgr.getRecurrentState(0), std::runtime_error);
}

TEST(HybridCacheManagerTests, RoutingHybridKVAndMamba)
{
    cudaStream_t stream{nullptr};

    // Layer pattern: [Attn, Mamba, Attn, Mamba] — 2 attention, 2 Mamba.
    int32_t const totalLayers = 4;
    int32_t const numAttn = 2;
    int32_t const numMamba = 2;
    int32_t const maxBatch = 2;

    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes = {rt::HybridCacheManager::LayerType::kAttention, rt::HybridCacheManager::LayerType::kMamba,
        rt::HybridCacheManager::LayerType::kAttention, rt::HybridCacheManager::LayerType::kMamba};
    cfg.kvConfig = makeUniformKVConfig(numAttn, maxBatch, 64, 2, 32);
    cfg.mambaConfig = makeMambaConfig(numMamba, maxBatch);
    cfg.maxBatchSize = maxBatch;

    rt::HybridCacheManager mgr(cfg, stream);

    // Attention-only access on attention layers.
    EXPECT_NO_THROW((void) mgr.getCombinedKVCache(0));
    EXPECT_NO_THROW((void) mgr.getCombinedKVCache(2));
    EXPECT_THROW((void) mgr.getCombinedKVCache(1), std::runtime_error); // layer 1 is Mamba
    EXPECT_THROW((void) mgr.getCombinedKVCache(3), std::runtime_error); // layer 3 is Mamba

    // Mamba-only access on Mamba layers.
    EXPECT_NO_THROW((void) mgr.getRecurrentState(1));
    EXPECT_NO_THROW((void) mgr.getConvState(3));
    EXPECT_THROW((void) mgr.getRecurrentState(0), std::runtime_error); // layer 0 is Attn
    EXPECT_THROW((void) mgr.getConvState(2), std::runtime_error);      // layer 2 is Attn

    // Out-of-range.
    EXPECT_THROW((void) mgr.getCombinedKVCache(totalLayers), std::runtime_error);
    EXPECT_THROW((void) mgr.getRecurrentState(totalLayers), std::runtime_error);
}

// Construction rejects layerType/sub-manager count mismatches.
TEST(HybridCacheManagerTests, ConstructionValidatesLayerCounts)
{
    cudaStream_t stream{nullptr};

    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes = {rt::HybridCacheManager::LayerType::kAttention, rt::HybridCacheManager::LayerType::kAttention};
    // Mismatch: kvConfig declares 1 attention layer but layerTypes has 2.
    cfg.kvConfig = makeUniformKVConfig(1, 2, 64, 2, 32);
    cfg.mambaConfig = makeMambaConfig(0, 2);
    cfg.maxBatchSize = 2;

    EXPECT_THROW(rt::HybridCacheManager(cfg, stream), std::runtime_error);
}

// --- Batch + length management ---------------------------------------------

TEST(HybridCacheManagerTests, ResetAndCommitTracksActiveBatchAndEmptyFlag)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 4;
    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(1, rt::HybridCacheManager::LayerType::kAttention);
    cfg.kvConfig = makeUniformKVConfig(1, maxBatch, 16, 1, 8);
    cfg.mambaConfig = makeMambaConfig(0, maxBatch);
    cfg.maxBatchSize = maxBatch;

    rt::HybridCacheManager mgr(cfg, stream);
    EXPECT_TRUE(mgr.getKVCacheAllEmpty());

    // All-zero reuse lengths keep the empty flag true.
    std::vector<int32_t> reuseZero(3, 0);
    rt::Tensor reuseZeroT({3}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(reuseZeroT.rawPointer(), reuseZero.data(), reuseZero.size() * sizeof(int32_t));
    mgr.resetForNewSequences(reuseZeroT, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(mgr.getActiveBatchSize(), 3);
    EXPECT_TRUE(mgr.getKVCacheAllEmpty());
    EXPECT_EQ(mgr.getKVCacheLengths().getShape()[0], 3);

    // Non-zero reuse clears the empty flag.
    std::vector<int32_t> reuseNonZero{0, 5, 0};
    rt::Tensor reuseNonZeroT({3}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(reuseNonZeroT.rawPointer(), reuseNonZero.data(), reuseNonZero.size() * sizeof(int32_t));
    mgr.resetForNewSequences(reuseNonZeroT, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_FALSE(mgr.getKVCacheAllEmpty());

    // setActiveBatchSize rejects out-of-range values.
    EXPECT_THROW(mgr.setActiveBatchSize(-1), std::runtime_error);
    EXPECT_THROW(mgr.setActiveBatchSize(maxBatch + 1), std::runtime_error);
    EXPECT_NO_THROW(mgr.setActiveBatchSize(2));
    EXPECT_EQ(mgr.getActiveBatchSize(), 2);
    EXPECT_EQ(mgr.getKVCacheLengths().getShape()[0], 2);

    // Scalar commit increments per-slot lengths.
    mgr.commitSequenceLength(7, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto lengths = copyDeviceToHost<int32_t>(mgr.getKVCacheLengths());
    ASSERT_EQ(lengths.size(), 2u);
    // reuseNonZero truncated to activeBatch=2 -> {0, 5}, +7 each.
    EXPECT_EQ(lengths[0], 7);
    EXPECT_EQ(lengths[1], 12);
}

// --- Compaction: attention-only, oldBatch < maxBatch ------------------------

TEST(HybridCacheManagerTests, CompactBatchUniformKVSmallerThanMax)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 8;
    int32_t const oldBatch = 4;
    int32_t const newBatch = 2;
    int32_t const numLayers = 3;
    int32_t const maxSeqLen = 32;

    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(numLayers, rt::HybridCacheManager::LayerType::kAttention);
    // headDim must be one of {64, 128, 256, 512} — the batched kernel is template-dispatched.
    cfg.kvConfig = makeUniformKVConfig(numLayers, maxBatch, maxSeqLen, 2, 64);
    cfg.mambaConfig = makeMambaConfig(0, maxBatch);
    cfg.maxBatchSize = maxBatch;

    rt::HybridCacheManager mgr(cfg, stream);

    // Mark the live prefix [0, maxSeqLen) of each slot in every layer (both K and V halves) with
    // value = (layerIdx + 1) * 10 + slot. Compaction only moves the live prefix, so only that range
    // is seeded (and checked below).
    for (int32_t L = 0; L < numLayers; ++L)
    {
        for (int32_t b = 0; b < oldBatch; ++b)
        {
            fillSlotTokenRangeNhd(mgr, L, b, 0, maxSeqLen, static_cast<float>((L + 1) * 10 + b));
        }
    }

    // Keep slot 1 -> new 0, slot 3 -> new 1. Evict 0 and 2.
    auto mapping = uploadMapping({-1, 0, -1, 1});

    // Live length per slot (equal to maxSeqLen here) drives how much of each row compaction copies.
    std::vector<int32_t> hostLens(oldBatch, maxSeqLen);
    rt::Tensor reuseLens({oldBatch}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(reuseLens.rawPointer(), hostLens.data(), hostLens.size() * sizeof(int32_t));
    mgr.resetForNewSequences(reuseLens, stream);

    mgr.compactBatch(mapping, oldBatch, newBatch, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Active slots [0, newBatch) should carry old slots 1 and 3's values in both K and V halves,
    // over the live prefix [0, maxSeqLen).
    for (int32_t L = 0; L < numLayers; ++L)
    {
        expectSlotTokenRangeEqNhd(
            mgr, L, 0, 0, maxSeqLen, static_cast<float>((L + 1) * 10 + 1), "L=" + std::to_string(L) + " newSlot=0");
        expectSlotTokenRangeEqNhd(
            mgr, L, 1, 0, maxSeqLen, static_cast<float>((L + 1) * 10 + 3), "L=" + std::to_string(L) + " newSlot=1");
    }

    // Lengths compacted: all entries are maxSeqLen, so new slots should also be maxSeqLen.
    mgr.setActiveBatchSize(newBatch);
    auto lens = copyDeviceToHost<int32_t>(mgr.getKVCacheLengths());
    ASSERT_EQ(lens.size(), static_cast<size_t>(newBatch));
    EXPECT_EQ(lens[0], maxSeqLen);
    EXPECT_EQ(lens[1], maxSeqLen);
}

// Regression for review finding #8: compaction must copy only each survivor's live prefix, not the
// full capPadded*H*D row. Proven by poisoning the destination row's padded tail (beyond the live
// length) before compaction and asserting it is left untouched afterwards — a full-row copy (the
// pre-fix behavior) would instead overwrite that tail with the survivor's own padded-tail garbage.
TEST(HybridCacheManagerTests, CompactBatchCopiesOnlyLivePrefixNotFullCapacity)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 8;
    int32_t const oldBatch = 4;
    int32_t const newBatch = 2;
    int32_t const numLayers = 2;
    int32_t const maxSeqLen = 32; // capPadded rounds up to one 128-token page (> maxSeqLen).

    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(numLayers, rt::HybridCacheManager::LayerType::kAttention);
    cfg.kvConfig = makeUniformKVConfig(numLayers, maxBatch, maxSeqLen, 2, 64);
    cfg.mambaConfig = makeMambaConfig(0, maxBatch);
    cfg.maxBatchSize = maxBatch;

    rt::HybridCacheManager mgr(cfg, stream);
    int32_t const capPadded = static_cast<int32_t>(mgr.getSeparateKVCache(0).first.getShape()[1]);
    ASSERT_GT(capPadded, maxSeqLen) << "test requires padding beyond the live length to be meaningful";

    // Slot 2 (live length 5) -> new slot 0. Slot 3 (live length 20) -> new slot 1. Slots 0 and 1
    // are evicted AND double as the physical destination rows (compaction is in-place, so new slot
    // k's physical row is row k) — choosing survivors at rows >= newBatch keeps source and
    // destination rows disjoint, so poisoning rows 0/1 up front can't clobber a survivor's own data.
    int32_t const liveLen2 = 5;
    int32_t const liveLen3 = 20;
    constexpr float kPoison = -7.0f;
    constexpr float kGarbage = 777.0f;

    for (int32_t L = 0; L < numLayers; ++L)
    {
        // Destination rows: poison entirely so any spurious write beyond the live length is detectable.
        fillSlotNhd(mgr, L, 0, kPoison);
        fillSlotNhd(mgr, L, 1, kPoison);

        // Survivors: live prefix carries real data; the padded tail carries garbage that must NOT move.
        float const v2 = static_cast<float>((L + 1) * 10 + 2);
        float const v3 = static_cast<float>((L + 1) * 10 + 3);
        fillSlotTokenRangeNhd(mgr, L, 2, 0, liveLen2, v2);
        fillSlotTokenRangeNhd(mgr, L, 2, liveLen2, capPadded, kGarbage);
        fillSlotTokenRangeNhd(mgr, L, 3, 0, liveLen3, v3);
        fillSlotTokenRangeNhd(mgr, L, 3, liveLen3, capPadded, kGarbage);
    }

    std::vector<int32_t> hostLens{0, 0, liveLen2, liveLen3};
    rt::Tensor reuseLens({oldBatch}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(reuseLens.rawPointer(), hostLens.data(), hostLens.size() * sizeof(int32_t));
    mgr.resetForNewSequences(reuseLens, stream);

    auto mapping = uploadMapping({-1, -1, 0, 1});
    mgr.compactBatch(mapping, oldBatch, newBatch, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t L = 0; L < numLayers; ++L)
    {
        float const v2 = static_cast<float>((L + 1) * 10 + 2);
        float const v3 = static_cast<float>((L + 1) * 10 + 3);

        // Live prefix moved onto the destination slot.
        expectSlotTokenRangeEqNhd(mgr, L, 0, 0, liveLen2, v2, "L=" + std::to_string(L) + " newSlot=0 live prefix");
        expectSlotTokenRangeEqNhd(mgr, L, 1, 0, liveLen3, v3, "L=" + std::to_string(L) + " newSlot=1 live prefix");

        // Padded tail beyond the live length was left untouched (still the destination row's
        // poison) — proves the copy did not move the full capPadded row.
        expectSlotTokenRangeEqNhd(
            mgr, L, 0, liveLen2, capPadded, kPoison, "L=" + std::to_string(L) + " newSlot=0 padded tail");
        expectSlotTokenRangeEqNhd(
            mgr, L, 1, liveLen3, capPadded, kPoison, "L=" + std::to_string(L) + " newSlot=1 padded tail");
    }
}

// Validates that lengths compaction carries the correct per-slot values even
// when they differ. Uses a trivial 1-layer KV config so we isolate the
// generic `compactTensorBatch` path for the shared KV lengths tensor.
TEST(HybridCacheManagerTests, CompactBatchSharedLengthsCarriesPerSlotValues)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 8;
    int32_t const oldBatch = 4;
    int32_t const newBatch = 2;
    int32_t const maxSeqLen = 32;

    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(1, rt::HybridCacheManager::LayerType::kAttention);
    cfg.kvConfig = makeUniformKVConfig(1, maxBatch, maxSeqLen, 2, 64);
    cfg.mambaConfig = makeMambaConfig(0, maxBatch);
    cfg.maxBatchSize = maxBatch;

    rt::HybridCacheManager mgr(cfg, stream);

    std::vector<int32_t> hostLens{11, 12, 13, 14};
    rt::Tensor reuseLens({oldBatch}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(reuseLens.rawPointer(), hostLens.data(), hostLens.size() * sizeof(int32_t));
    mgr.resetForNewSequences(reuseLens, stream);

    // Keep slot 1 -> 0, slot 3 -> 1.
    auto mapping = uploadMapping({-1, 0, -1, 1});
    mgr.compactBatch(mapping, oldBatch, newBatch, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    mgr.setActiveBatchSize(newBatch);
    auto lens = copyDeviceToHost<int32_t>(mgr.getKVCacheLengths());
    ASSERT_EQ(lens.size(), static_cast<size_t>(newBatch));
    EXPECT_EQ(lens[0], 12);
    EXPECT_EQ(lens[1], 14);
}

TEST(HybridCacheManagerTests, CompactBatchSlotStateLeavesGlobalKVPagesInPlace)
{
    cudaStream_t stream{nullptr};
    int32_t const maxBatch = 2;
    int32_t const maxSeqLen = 32;

    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(1, rt::HybridCacheManager::LayerType::kAttention);
    cfg.kvConfig = makeUniformKVConfig(1, maxBatch, maxSeqLen, 2, 64);
    cfg.mambaConfig = makeMambaConfig(0, maxBatch);
    cfg.maxBatchSize = maxBatch;
    rt::HybridCacheManager mgr(cfg, stream);

    fillSlotTokenRangeNhd(mgr, 0, 0, 0, maxSeqLen, 10.0F);
    fillSlotTokenRangeNhd(mgr, 0, 1, 0, maxSeqLen, 20.0F);
    std::vector<int32_t> hostLens{5, 7};
    rt::Tensor reuseLens({maxBatch}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(reuseLens.rawPointer(), hostLens.data(), hostLens.size() * sizeof(int32_t));
    mgr.resetForNewSequences(reuseLens, stream);

    auto mapping = uploadMapping({-1, 0});
    mgr.compactBatchSlotState(mapping, maxBatch, /*newBatch=*/1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    expectSlotTokenRangeEqNhd(mgr, 0, 0, 0, maxSeqLen, 10.0F, "global page row 0");
    expectSlotTokenRangeEqNhd(mgr, 0, 1, 0, maxSeqLen, 20.0F, "global page row 1");
    mgr.setActiveBatchSize(1);
    auto const lens = copyDeviceToHost<int32_t>(mgr.getKVCacheLengths());
    ASSERT_EQ(lens.size(), 1U);
    EXPECT_EQ(lens.front(), 7);
}

TEST(HybridCacheManagerTests, CompactBatchHeterogeneousHeadDim)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 4;
    int32_t const oldBatch = 4;
    int32_t const newBatch = 2;
    int32_t const numLayers = 4; // first half headDim=64, second half headDim=128 (two HeadDimGroups)

    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(numLayers, rt::HybridCacheManager::LayerType::kAttention);
    cfg.kvConfig = makeHeteroKVConfig(numLayers, maxBatch, 32, /*h0=*/4, /*d0=*/64, /*h1=*/2, /*d1=*/128);
    cfg.mambaConfig = makeMambaConfig(0, maxBatch);
    cfg.maxBatchSize = maxBatch;

    rt::HybridCacheManager mgr(cfg, stream);

    int32_t const maxSeqLen = 32;
    for (int32_t L = 0; L < numLayers; ++L)
    {
        for (int32_t b = 0; b < oldBatch; ++b)
        {
            fillSlotTokenRangeNhd(mgr, L, b, 0, maxSeqLen, static_cast<float>((L + 1) * 100 + b));
        }
    }

    // Keep slots 0 and 2, drop 1 and 3.
    auto mapping = uploadMapping({0, -1, 1, -1});

    std::vector<int32_t> hostLens(oldBatch, maxSeqLen);
    rt::Tensor reuseLens({oldBatch}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(reuseLens.rawPointer(), hostLens.data(), hostLens.size() * sizeof(int32_t));
    mgr.resetForNewSequences(reuseLens, stream);

    mgr.compactBatch(mapping, oldBatch, newBatch, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t L = 0; L < numLayers; ++L)
    {
        // Slot 0 stays in place; slot 1 receives old slot 2 (live prefix [0, maxSeqLen) only).
        expectSlotTokenRangeEqNhd(
            mgr, L, 0, 0, maxSeqLen, static_cast<float>((L + 1) * 100 + 0), "L=" + std::to_string(L) + " newSlot=0");
        expectSlotTokenRangeEqNhd(
            mgr, L, 1, 0, maxSeqLen, static_cast<float>((L + 1) * 100 + 2), "L=" + std::to_string(L) + " newSlot=1");
    }
}

// --- P0 regression: hybrid model, oldBatch < maxBatchSize -------------------
//
// Before the reshape-compact-reshape workaround in hybridCacheManager.cpp,
// compactTensorBatch on Mamba state tensors (allocated at [maxBatchSize, ...])
// fired `srcShape[0] == oldActiveBatch` and crashed. This test locks that path
// in: it drives a mixed KV+Mamba config with oldBatch (4) strictly less than
// maxBatchSize (8) and verifies that (a) no exception is thrown, (b) KV +
// Mamba states are compacted correctly, and (c) Mamba tensor shapes are
// restored to maxBatchSize after compaction.
TEST(HybridCacheManagerTests, CompactBatchHybridRegressionOldBatchLessThanMax)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 8;
    int32_t const oldBatch = 4;
    int32_t const newBatch = 2;
    int32_t const numAttn = 2;
    int32_t const numMamba = 2;

    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes = {rt::HybridCacheManager::LayerType::kAttention, rt::HybridCacheManager::LayerType::kMamba,
        rt::HybridCacheManager::LayerType::kAttention, rt::HybridCacheManager::LayerType::kMamba};
    // headDim must be one of {64, 128, 256, 512} — the batched kernel is template-dispatched.
    cfg.kvConfig = makeUniformKVConfig(numAttn, maxBatch, 32, 2, 64);
    cfg.mambaConfig = makeMambaConfig(numMamba, maxBatch);
    cfg.maxBatchSize = maxBatch;

    rt::HybridCacheManager mgr(cfg, stream);

    // Seed KV caches (NHD: fill both K and V halves of each row, over the live prefix [0, maxSeqLen)
    // that compaction is expected to copy — see maxSeqLen/hostLens below).
    int32_t const maxSeqLen = 32;
    std::vector<int32_t> const kvLayerAbs{0, 2};
    for (size_t idx = 0; idx < kvLayerAbs.size(); ++idx)
    {
        int32_t const L = kvLayerAbs[idx];
        for (int32_t b = 0; b < oldBatch; ++b)
        {
            fillSlotTokenRangeNhd(mgr, L, b, 0, maxSeqLen, static_cast<float>(L * 10 + b + 1));
        }
    }

    // Seed Mamba states (recurrent + conv) for all slots including inactive
    // ones beyond oldBatch. The compaction code must not touch those.
    std::vector<int32_t> const mambaLayerAbs{1, 3};
    for (size_t idx = 0; idx < mambaLayerAbs.size(); ++idx)
    {
        int32_t const L = mambaLayerAbs[idx];
        rt::Tensor& rec = mgr.getRecurrentState(L);
        rt::Tensor& conv = mgr.getConvState(L);
        for (int32_t b = 0; b < maxBatch; ++b)
        {
            fillSlotHalf(rec, b, static_cast<float>(L * 100 + b + 1));
            fillSlotHalf(conv, b, static_cast<float>(L * 1000 + b + 1));
        }
    }

    // Keep old slot 1 -> new 0, old slot 3 -> new 1; evict 0 and 2.
    auto mapping = uploadMapping({-1, 0, -1, 1});

    std::vector<int32_t> hostLens(oldBatch, maxSeqLen);
    rt::Tensor reuseLens({oldBatch}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(reuseLens.rawPointer(), hostLens.data(), hostLens.size() * sizeof(int32_t));
    mgr.resetForNewSequences(reuseLens, stream);

    // The bug before the fix: this would throw from compactTensorBatch's
    // `srcShape[0] == oldActiveBatch` check on the first Mamba state.
    ASSERT_NO_THROW(mgr.compactBatch(mapping, oldBatch, newBatch, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // KV: new slot 0 <- old 1, new slot 1 <- old 3 (both halves, live prefix [0, maxSeqLen)).
    for (size_t idx = 0; idx < kvLayerAbs.size(); ++idx)
    {
        int32_t const L = kvLayerAbs[idx];
        expectSlotTokenRangeEqNhd(
            mgr, L, 0, 0, maxSeqLen, static_cast<float>(L * 10 + 1 + 1), "kv L=" + std::to_string(L) + " newSlot=0");
        expectSlotTokenRangeEqNhd(
            mgr, L, 1, 0, maxSeqLen, static_cast<float>(L * 10 + 3 + 1), "kv L=" + std::to_string(L) + " newSlot=1");
    }

    // Mamba: active slots compacted; tensor shape restored to maxBatch.
    for (size_t idx = 0; idx < mambaLayerAbs.size(); ++idx)
    {
        int32_t const L = mambaLayerAbs[idx];
        rt::Tensor& rec = mgr.getRecurrentState(L);
        rt::Tensor& conv = mgr.getConvState(L);

        EXPECT_EQ(rec.getShape()[0], maxBatch) << "recurrent shape[0] not restored to maxBatch";
        EXPECT_EQ(conv.getShape()[0], maxBatch) << "conv shape[0] not restored to maxBatch";

        expectSlotEqHalf(rec, 0, static_cast<float>(L * 100 + 1 + 1), "rec L=" + std::to_string(L) + " newSlot=0");
        expectSlotEqHalf(rec, 1, static_cast<float>(L * 100 + 3 + 1), "rec L=" + std::to_string(L) + " newSlot=1");
        expectSlotEqHalf(conv, 0, static_cast<float>(L * 1000 + 1 + 1), "conv L=" + std::to_string(L) + " newSlot=0");
        expectSlotEqHalf(conv, 1, static_cast<float>(L * 1000 + 3 + 1), "conv L=" + std::to_string(L) + " newSlot=1");
    }
}

// --- Capture / restore round-trip ------------------------------------------

TEST(HybridCacheManagerTests, CaptureRestoreRoundTripUniform)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 2;
    int32_t const numLayers = 2;
    int32_t const capturedSeq = 16;

    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(numLayers, rt::HybridCacheManager::LayerType::kAttention);
    // headDim must be one of {64, 128, 256, 512} — the batched kernel is template-dispatched.
    cfg.kvConfig = makeUniformKVConfig(numLayers, maxBatch, 32, 2, 64);
    cfg.mambaConfig = makeMambaConfig(0, maxBatch);
    cfg.maxBatchSize = maxBatch;

    rt::HybridCacheManager mgr(cfg, stream);

    int32_t const captureSlot = 1;
    for (int32_t L = 0; L < numLayers; ++L)
    {
        fillSlotNhd(mgr, L, captureSlot, static_cast<float>(L + 7));
    }

    auto saved = mgr.captureKVCache(captureSlot, capturedSeq, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(saved.size(), static_cast<size_t>(numLayers));
    // Saved tensor is NHD [2, capturedSeq, H, D].
    for (auto const& s : saved)
    {
        EXPECT_EQ(s.getShape()[0], 2);
        EXPECT_EQ(s.getShape()[1], capturedSeq);
    }

    // Zero the slot (both halves), then restore.
    for (int32_t L = 0; L < numLayers; ++L)
    {
        fillSlotNhd(mgr, L, captureSlot, 0.f);
    }

    mgr.restoreKVCache(saved, captureSlot, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify the first `capturedSeq` tokens of the restored row match the seed in BOTH halves.
    // NHD half view is [maxBatch, capPadded, H, D]; row `captureSlot` holds capPadded tokens, of
    // which restore overwrites only [0, capturedSeq). Each token is H*D contiguous elements.
    for (int32_t L = 0; L < numLayers; ++L)
    {
        auto [kView, vView] = mgr.getSeparateKVCache(L);
        float const expected = static_cast<float>(L + 7);
        for (rt::Tensor const* half : {&kView, &vView})
        {
            auto const& shape = half->getShape();
            int32_t const headsTimesDim = static_cast<int32_t>(shape[2] * shape[3]); // H * D
            auto host = readSlotHalf(*half, captureSlot);                            // capPadded * H * D elems
            for (int32_t s = 0; s < capturedSeq; ++s)
            {
                for (int32_t e = 0; e < headsTimesDim; ++e)
                {
                    auto got = host[static_cast<size_t>(static_cast<int64_t>(s) * headsTimesDim + e)];
                    ASSERT_TRUE(isclose(got, __float2half(expected), 1e-2f, 1e-2f))
                        << "L=" << L << " half=" << (half == &kView ? "K" : "V") << " token=" << s << " elem=" << e
                        << " got=" << __half2float(got) << " expected=" << expected;
                }
            }
        }
    }
}

TEST(HybridCacheManagerTests, CaptureRestoreWithExtraRetainedPages)
{
    cudaStream_t stream{nullptr};
    int32_t const maxBatch = 2;
    int32_t const maxSeq = 32;
    int32_t const capturedSeq = 16;
    int64_t const computedMinimumActivePages = rt::computeMinimumKvPoolPages(maxBatch, maxSeq);
    ASSERT_LE(computedMinimumActivePages, rt::kMAX_KV_POOL_PAGES);
    int32_t const minimumActivePages = static_cast<int32_t>(computedMinimumActivePages);

    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(1, rt::HybridCacheManager::LayerType::kAttention);
    cfg.kvConfig = makeUniformKVConfig(/*numLayers=*/1, maxBatch, maxSeq, /*numKVHeads=*/2, /*headDim=*/64);
    cfg.kvConfig.numPages = minimumActivePages + 3;
    cfg.mambaConfig = makeMambaConfig(/*numLayers=*/0, maxBatch);
    cfg.maxBatchSize = maxBatch;

    rt::HybridCacheManager mgr(cfg, stream);
    int32_t const captureSlot = 1;
    fillSlotNhd(mgr, /*absLayer=*/0, captureSlot, 7.0F);

    auto const saved = mgr.captureKVCache(captureSlot, capturedSeq, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fillSlotNhd(mgr, /*absLayer=*/0, captureSlot, 0.0F);
    mgr.restoreKVCache(saved, captureSlot, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    expectSlotTokenRangeEqNhd(mgr, /*absLayer=*/0, captureSlot, /*startTok=*/0, capturedSeq, 7.0F,
        "capture/restore with extra retained pages");
}

TEST(HybridCacheManagerTests, CompactBatchWithExtraRetainedPagesUsesPhysicalVHalfOffset)
{
    cudaStream_t stream{nullptr};
    int32_t const maxBatch = 3;
    int32_t const oldBatch = 3;
    int32_t const newBatch = 2;
    int32_t const maxSeq = 32;
    int64_t const computedMinimumActivePages = rt::computeMinimumKvPoolPages(maxBatch, maxSeq);
    ASSERT_LE(computedMinimumActivePages, rt::kMAX_KV_POOL_PAGES);
    int32_t const minimumActivePages = static_cast<int32_t>(computedMinimumActivePages);

    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(1, rt::HybridCacheManager::LayerType::kAttention);
    cfg.kvConfig = makeUniformKVConfig(/*numLayers=*/1, maxBatch, maxSeq, /*numKVHeads=*/2, /*headDim=*/64);
    cfg.kvConfig.numPages = minimumActivePages + 4;
    cfg.mambaConfig = makeMambaConfig(/*numLayers=*/0, maxBatch);
    cfg.maxBatchSize = maxBatch;

    rt::HybridCacheManager mgr(cfg, stream);
    fillSlotNhd(mgr, /*absLayer=*/0, /*slot=*/0, 10.0F);
    fillSlotNhd(mgr, /*absLayer=*/0, /*slot=*/1, 20.0F);
    fillSlotNhd(mgr, /*absLayer=*/0, /*slot=*/2, 30.0F);
    std::vector<int32_t> const hostLengths(oldBatch, maxSeq);
    rt::Tensor lengths({oldBatch}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(lengths.rawPointer(), hostLengths.data(), hostLengths.size() * sizeof(int32_t));
    mgr.resetForNewSequences(lengths, stream);

    auto const mapping = uploadMapping({-1, 0, 1});
    mgr.compactBatch(mapping, oldBatch, newBatch, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    expectSlotTokenRangeEqNhd(
        mgr, /*absLayer=*/0, /*slot=*/0, /*startTok=*/0, maxSeq, 20.0F, "compact slot 0 with extra retained pages");
    expectSlotTokenRangeEqNhd(
        mgr, /*absLayer=*/0, /*slot=*/1, /*startTok=*/0, maxSeq, 30.0F, "compact slot 1 with extra retained pages");
}

// --- Parametrized headDim coverage -----------------------------------------
//
// Exercises the batched save/restore (captureKVCache + restoreKVCache) and
// compact (compactBatch) paths across every headDim the batched kernels claim
// to support. Prior to this coverage the suite only ran 64/128, which missed
// a HEAD_DIM=512 corruption (kSLEN_PER_WARP=0 in the legacy copy kernel).

class HybridCacheManagerHeadDimTest : public ::testing::TestWithParam<int32_t>
{
};

TEST_P(HybridCacheManagerHeadDimTest, CaptureRestoreRoundTrip)
{
    int32_t const headDim = GetParam();
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 2;
    int32_t const numLayers = 2;
    int32_t const maxSeq = 32;
    int32_t const capturedSeq = 16;

    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(numLayers, rt::HybridCacheManager::LayerType::kAttention);
    cfg.kvConfig = makeUniformKVConfig(numLayers, maxBatch, maxSeq, /*numKVHeads=*/2, headDim);
    cfg.mambaConfig = makeMambaConfig(0, maxBatch);
    cfg.maxBatchSize = maxBatch;

    rt::HybridCacheManager mgr(cfg, stream);

    int32_t const captureSlot = 1;
    for (int32_t L = 0; L < numLayers; ++L)
    {
        fillSlotNhd(mgr, L, captureSlot, static_cast<float>(L + 7));
    }

    auto saved = mgr.captureKVCache(captureSlot, capturedSeq, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(saved.size(), static_cast<size_t>(numLayers));

    for (int32_t L = 0; L < numLayers; ++L)
    {
        fillSlotNhd(mgr, L, captureSlot, 0.f);
    }

    mgr.restoreKVCache(saved, captureSlot, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify the prefix [0, capturedSeq) tokens of the restored row match the seed in both halves.
    for (int32_t L = 0; L < numLayers; ++L)
    {
        auto [kView, vView] = mgr.getSeparateKVCache(L);
        float const expected = static_cast<float>(L + 7);
        for (rt::Tensor const* half : {&kView, &vView})
        {
            auto const& shape = half->getShape();
            int32_t const headsTimesDim = static_cast<int32_t>(shape[2] * shape[3]); // H * D
            auto host = readSlotHalf(*half, captureSlot);
            for (int32_t s = 0; s < capturedSeq; ++s)
            {
                for (int32_t e = 0; e < headsTimesDim; ++e)
                {
                    auto got = host[static_cast<size_t>(static_cast<int64_t>(s) * headsTimesDim + e)];
                    ASSERT_TRUE(isclose(got, __float2half(expected), 1e-2f, 1e-2f))
                        << "headDim=" << headDim << " L=" << L << " half=" << (half == &kView ? "K" : "V")
                        << " token=" << s << " elem=" << e << " got=" << __half2float(got) << " expected=" << expected;
                }
            }
        }
    }
}

TEST_P(HybridCacheManagerHeadDimTest, CompactBatchUniform)
{
    int32_t const headDim = GetParam();
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 4;
    int32_t const oldBatch = 4;
    int32_t const newBatch = 2;
    int32_t const numLayers = 2;
    int32_t const maxSeqLen = 32;

    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(numLayers, rt::HybridCacheManager::LayerType::kAttention);
    cfg.kvConfig = makeUniformKVConfig(numLayers, maxBatch, maxSeqLen, /*numKVHeads=*/2, headDim);
    cfg.mambaConfig = makeMambaConfig(0, maxBatch);
    cfg.maxBatchSize = maxBatch;

    rt::HybridCacheManager mgr(cfg, stream);

    for (int32_t L = 0; L < numLayers; ++L)
    {
        for (int32_t b = 0; b < oldBatch; ++b)
        {
            fillSlotTokenRangeNhd(mgr, L, b, 0, maxSeqLen, static_cast<float>((L + 1) * 10 + b));
        }
    }

    auto mapping = uploadMapping({-1, 0, -1, 1});
    std::vector<int32_t> hostLens(oldBatch, maxSeqLen);
    rt::Tensor reuseLens({oldBatch}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(reuseLens.rawPointer(), hostLens.data(), hostLens.size() * sizeof(int32_t));
    mgr.resetForNewSequences(reuseLens, stream);

    mgr.compactBatch(mapping, oldBatch, newBatch, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t L = 0; L < numLayers; ++L)
    {
        expectSlotTokenRangeEqNhd(mgr, L, 0, 0, maxSeqLen, static_cast<float>((L + 1) * 10 + 1),
            "headDim=" + std::to_string(headDim) + " L=" + std::to_string(L) + " newSlot=0");
        expectSlotTokenRangeEqNhd(mgr, L, 1, 0, maxSeqLen, static_cast<float>((L + 1) * 10 + 3),
            "headDim=" + std::to_string(headDim) + " L=" + std::to_string(L) + " newSlot=1");
    }
}

INSTANTIATE_TEST_SUITE_P(AllSupportedHeadDims, HybridCacheManagerHeadDimTest, ::testing::Values(64, 128, 256, 512),
    [](::testing::TestParamInfo<int32_t> const& info) { return "headDim" + std::to_string(info.param); });

// --- Pure-Mamba construction ------------------------------------------------
//
// Ensures HybridCacheManager tolerates zero attention layers (pure Mamba /
// pure recurrent models). This was previously blocked by a positive-only
// assert inside KVCacheManager's ctor.

TEST(HybridCacheManagerTests, ConstructPureMambaNoAttentionLayers)
{
    cudaStream_t stream{nullptr};
    int32_t const maxBatch = 4;
    int32_t const numMamba = 3;

    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(numMamba, rt::HybridCacheManager::LayerType::kMamba);
    // Zero-attention KV config: empty layerConfigs, numAttentionLayers == 0.
    cfg.kvConfig = rt::KVCacheManager::Config{0, maxBatch, 32, {}, DataType::kHALF};
    cfg.mambaConfig = makeMambaConfig(numMamba, maxBatch);
    cfg.maxBatchSize = maxBatch;

    rt::HybridCacheManager mgr(cfg, stream);
    EXPECT_EQ(mgr.getKVCacheManager().numLayers(), 0);
    EXPECT_EQ(mgr.getMambaCacheManager().numLayers(), numMamba);
}

// --- FP8 capture/restore contract -------------------------------------------
//
// FP8 save/restore is not implemented — the batched copy kernel only
// instantiates the `half` template. Confirm the entry point throws a clear
// error instead of silently corrupting (matches main's single-layer contract).

TEST(HybridCacheManagerTests, CaptureKVCacheRejectsFp8)
{
    cudaStream_t stream{nullptr};
    int32_t const maxBatch = 1;
    int32_t const numLayers = 1;

    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(numLayers, rt::HybridCacheManager::LayerType::kAttention);
    cfg.kvConfig = rt::KVCacheManager::Config{numLayers, maxBatch, 32, {rt::KVLayerConfig{2, 64}}, DataType::kFP8};
    cfg.mambaConfig = makeMambaConfig(0, maxBatch);
    cfg.maxBatchSize = maxBatch;

    rt::HybridCacheManager mgr(cfg, stream);

    EXPECT_THROW(mgr.captureKVCache(0, 8, stream), std::exception);
}
