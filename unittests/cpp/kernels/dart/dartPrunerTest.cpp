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

#include "common/checkMacros.h"
#include "common/tensor.h"
#include "kernels/dart/dartGatherKernels.h"
#include "runtime/preprocess/visualTokenPruner.h"

#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kImageTokenId = 151655;
constexpr int32_t kTextTokenId = 42;

//! CPU mirror of the embedding-level DART selection: L1-norm pivots, then greedy anti-cosine
//! growth over a shrinking candidate pool, both scored on the input embeddings. This matches
//! the `pruned_layer=0` mode of the DART repo's `_select_dart_keep_indices` (the variant this
//! runtime implements), NOT the paper's layer-2 algorithm, which scores pivots with projected
//! key states — see the deviation note in runtime/preprocess/dartPruner.h. Operates on the
//! same half-rounded values the GPU kernels see.
std::vector<int32_t> referenceSelect(std::vector<float> const& embeds, int32_t seqLen, int32_t hidden,
    std::vector<int32_t> const& tokenIds, rt::VisualPrunerConfig const& cfg)
{
    std::vector<int32_t> imagePos;
    std::vector<int32_t> textPos;
    for (int32_t i = 0; i < seqLen; ++i)
    {
        (tokenIds[i] == kImageTokenId ? imagePos : textPos).push_back(i);
    }
    int32_t const numVisual = static_cast<int32_t>(imagePos.size());
    std::vector<int32_t> all(seqLen);
    std::iota(all.begin(), all.end(), 0);
    if (numVisual == 0 || numVisual < cfg.minVisualTokens)
    {
        return all;
    }
    // Skip guard mirrors the runtime: pruning happens only when the sum of per-span quotas is
    // an actual reduction.
    int32_t sumTargets = 0;
    for (size_t i = 0; i < imagePos.size();)
    {
        size_t j = i + 1;
        while (j < imagePos.size() && imagePos[j] == imagePos[j - 1] + 1)
        {
            ++j;
        }
        int32_t const spanLen = static_cast<int32_t>(j - i);
        int32_t t = static_cast<int32_t>(std::ceil(static_cast<double>(spanLen) * (1.0 - cfg.reductionRatio)));
        sumTargets += std::max(1, std::min(spanLen, t));
        i = j;
    }
    if (sumTargets >= numVisual)
    {
        return all;
    }

    std::vector<float> l1(seqLen, 0.0F);
    std::vector<float> l2(seqLen, 0.0F);
    for (int32_t i = 0; i < seqLen; ++i)
    {
        double a = 0.0;
        double s = 0.0;
        for (int32_t h = 0; h < hidden; ++h)
        {
            double const v = embeds[static_cast<int64_t>(i) * hidden + h];
            a += std::abs(v);
            s += v * v;
        }
        l1[i] = static_cast<float>(a);
        l2[i] = static_cast<float>(std::sqrt(s));
    }

    auto topByScore = [](std::vector<int32_t> positions, std::vector<float> const& score, int32_t count) {
        count = std::min<int32_t>(count, static_cast<int32_t>(positions.size()));
        std::partial_sort(positions.begin(), positions.begin() + count, positions.end(),
            [&score](int32_t a, int32_t b) { return score[a] != score[b] ? score[a] > score[b] : a < b; });
        positions.resize(std::max(count, 0));
        return positions;
    };

    std::vector<int32_t> const textPivots = topByScore(textPos, l1, cfg.pivotTextTokens);

    // Per-image-span selection: split visual positions into contiguous runs, give each its
    // proportional quota, and run pivot + greedy growth independently per span (text pivots
    // shared), mirroring the runtime implementation.
    std::vector<int32_t> retained;
    for (size_t i = 0; i < imagePos.size();)
    {
        size_t j = i + 1;
        while (j < imagePos.size() && imagePos[j] == imagePos[j - 1] + 1)
        {
            ++j;
        }
        std::vector<int32_t> const spanPos(imagePos.begin() + i, imagePos.begin() + j);
        int32_t const spanLen = static_cast<int32_t>(spanPos.size());
        int32_t spanTarget = static_cast<int32_t>(std::ceil(static_cast<double>(spanLen) * (1.0 - cfg.reductionRatio)));
        spanTarget = std::max(1, std::min(spanLen, spanTarget));
        i = j;

        std::vector<int32_t> const imagePivots = topByScore(spanPos, l1, std::min(cfg.pivotImageTokens, spanTarget));
        std::vector<int32_t> pivots = imagePivots;
        pivots.insert(pivots.end(), textPivots.begin(), textPivots.end());
        int32_t const numPivots = static_cast<int32_t>(pivots.size());

        std::vector<int32_t> spanRetained = imagePivots;
        std::set<int32_t> retainedSet(spanRetained.begin(), spanRetained.end());
        std::vector<int32_t> candidates;
        for (int32_t p : spanPos)
        {
            if (retainedSet.count(p) == 0)
            {
                candidates.push_back(p);
            }
        }

        for (int32_t offset = 0; offset < numPivots; ++offset)
        {
            int32_t const remaining = spanTarget - static_cast<int32_t>(spanRetained.size());
            if (remaining <= 0 || candidates.empty())
            {
                break;
            }
            int32_t const perPivot = (remaining + (numPivots - offset) - 1) / (numPivots - offset);
            int32_t const pivot = pivots[offset];
            std::vector<float> sim(seqLen, 0.0F);
            for (int32_t c : candidates)
            {
                double dot = 0.0;
                for (int32_t h = 0; h < hidden; ++h)
                {
                    dot += static_cast<double>(embeds[static_cast<int64_t>(pivot) * hidden + h])
                        * embeds[static_cast<int64_t>(c) * hidden + h];
                }
                float const denom = std::max(l2[pivot] * l2[c], 1e-8F);
                sim[c] = -static_cast<float>(dot) / denom;
            }
            std::vector<int32_t> const chosen = topByScore(candidates, sim, perPivot);
            for (int32_t c : chosen)
            {
                spanRetained.push_back(c);
                retainedSet.insert(c);
            }
            candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                 [&retainedSet](int32_t c) { return retainedSet.count(c) != 0; }),
                candidates.end());
        }
        retained.insert(retained.end(), spanRetained.begin(), spanRetained.end());
    }

    std::vector<int32_t> keep = textPos;
    keep.insert(keep.end(), retained.begin(), retained.end());
    std::sort(keep.begin(), keep.end());
    return keep;
}

struct PrunerHarness
{
    static constexpr int32_t kHidden = 256;
    static constexpr int32_t kRotaryDim = 64;
    static constexpr int32_t kCapacity = 1024;

    rt::LLMEngineConfig engineConfig{};
    rt::PipelineIO io{};
    std::vector<float> embedsFloat; //!< half-rounded values the GPU sees
    std::vector<int32_t> tokenIds;
    int32_t seqLen{0};

    PrunerHarness(std::vector<int32_t> tokens, int32_t numDeepstack, uint32_t seed)
        : tokenIds(std::move(tokens))
        , seqLen(static_cast<int32_t>(tokenIds.size()))
    {
        engineConfig.hiddenSize = kHidden;
        engineConfig.rotaryDim = kRotaryDim;
        engineConfig.maxKVCacheCapacity = kCapacity;
        engineConfig.maxSupportedInputLength = 512;
        engineConfig.imageTokenId = kImageTokenId;

        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0F, 1.0F);
        std::vector<half> hostHalf(static_cast<size_t>(seqLen) * kHidden);
        embedsFloat.resize(hostHalf.size());
        for (size_t i = 0; i < hostHalf.size(); ++i)
        {
            hostHalf[i] = __float2half(dist(rng));
            embedsFloat[i] = __half2float(hostHalf[i]);
        }

        io.inputsEmbeds = rt::Tensor({1, seqLen, kHidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "embeds");
        CUDA_CHECK(cudaMemcpy(
            io.inputsEmbeds.rawPointer(), hostHalf.data(), hostHalf.size() * sizeof(half), cudaMemcpyHostToDevice));

        // Deepstack planes: marker rows (row index in every column) so gathers are verifiable.
        for (int32_t d = 0; d < numDeepstack; ++d)
        {
            std::vector<half> marker(static_cast<size_t>(seqLen) * kHidden);
            for (int32_t r = 0; r < seqLen; ++r)
            {
                std::fill_n(marker.begin() + static_cast<int64_t>(r) * kHidden, kHidden, __float2half(float(r)));
            }
            io.deepstackEmbeds.emplace_back(
                rt::Coords{1, seqLen, kHidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "deepstack");
            CUDA_CHECK(cudaMemcpy(io.deepstackEmbeds.back().rawPointer(), marker.data(), marker.size() * sizeof(half),
                cudaMemcpyHostToDevice));
        }

        // Rope rows: marker = source row index in every column.
        std::vector<float> rope(static_cast<size_t>(kCapacity) * kRotaryDim);
        for (int32_t r = 0; r < kCapacity; ++r)
        {
            std::fill_n(rope.begin() + static_cast<int64_t>(r) * kRotaryDim, kRotaryDim, float(r));
        }
        io.mropeActiveCosSin
            = rt::Tensor({1, kCapacity, kRotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "rope");
        CUDA_CHECK(cudaMemcpy(
            io.mropeActiveCosSin.rawPointer(), rope.data(), rope.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    //! Recover the gather indices from the first deepstack marker plane (or rope rows).
    std::vector<int32_t> readDeepstackRows(int32_t count)
    {
        std::vector<half> out(static_cast<size_t>(count) * kHidden);
        CUDA_CHECK(cudaMemcpy(
            out.data(), io.deepstackEmbeds[0].rawPointer(), out.size() * sizeof(half), cudaMemcpyDeviceToHost));
        std::vector<int32_t> rows(count);
        for (int32_t r = 0; r < count; ++r)
        {
            rows[r] = static_cast<int32_t>(__half2float(out[static_cast<int64_t>(r) * kHidden]));
        }
        return rows;
    }

    std::vector<int32_t> readRopeRows(int32_t count)
    {
        std::vector<float> out(static_cast<size_t>(count) * kRotaryDim);
        CUDA_CHECK(cudaMemcpy(
            out.data(), io.mropeActiveCosSin.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<int32_t> rows(count);
        for (int32_t r = 0; r < count; ++r)
        {
            rows[r] = static_cast<int32_t>(out[static_cast<int64_t>(r) * kRotaryDim]);
        }
        return rows;
    }
};

std::vector<int32_t> makeTokens(int32_t preText, int32_t imageTokens, int32_t postText)
{
    std::vector<int32_t> tokens;
    tokens.insert(tokens.end(), preText, kTextTokenId);
    tokens.insert(tokens.end(), imageTokens, kImageTokenId);
    tokens.insert(tokens.end(), postText, kTextTokenId);
    return tokens;
}

//! Multi-slot variant of PrunerHarness: embeds [batch, maxLen, hidden], deepstack marker rows
//! encode (slot, row) in the first two columns, rope planes encode the source row per slot.
struct BatchPrunerHarness
{
    static constexpr int32_t kHidden = 256;
    static constexpr int32_t kRotaryDim = 64;
    static constexpr int32_t kCapacity = 1024;

    rt::LLMEngineConfig engineConfig{};
    rt::PipelineIO io{};
    std::vector<std::vector<float>> embedsFloat; //!< per-slot half-rounded values the GPU sees
    std::vector<std::vector<int32_t>> tokenIds;
    std::vector<int32_t> lens;
    int32_t batch{0};
    int32_t maxLen{0};

    BatchPrunerHarness(std::vector<std::vector<int32_t>> tokens, int32_t numDeepstack, uint32_t seed)
        : tokenIds(std::move(tokens))
        , batch(static_cast<int32_t>(tokenIds.size()))
    {
        engineConfig.hiddenSize = kHidden;
        engineConfig.rotaryDim = kRotaryDim;
        engineConfig.maxKVCacheCapacity = kCapacity;
        engineConfig.maxSupportedInputLength = 512;
        engineConfig.maxSupportedBatchSize = 4;
        engineConfig.imageTokenId = kImageTokenId;

        for (auto const& slotTokens : tokenIds)
        {
            lens.push_back(static_cast<int32_t>(slotTokens.size()));
            maxLen = std::max(maxLen, lens.back());
        }

        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0F, 1.0F);
        std::vector<half> hostHalf(static_cast<size_t>(batch) * maxLen * kHidden, __float2half(0.0F));
        embedsFloat.resize(batch);
        for (int32_t b = 0; b < batch; ++b)
        {
            embedsFloat[b].resize(static_cast<size_t>(lens[b]) * kHidden);
            for (int64_t i = 0; i < static_cast<int64_t>(lens[b]) * kHidden; ++i)
            {
                half const v = __float2half(dist(rng));
                hostHalf[static_cast<int64_t>(b) * maxLen * kHidden + i] = v;
                embedsFloat[b][i] = __half2float(v);
            }
        }
        io.inputsEmbeds
            = rt::Tensor({batch, maxLen, kHidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "embeds");
        CUDA_CHECK(cudaMemcpy(
            io.inputsEmbeds.rawPointer(), hostHalf.data(), hostHalf.size() * sizeof(half), cudaMemcpyHostToDevice));

        // Deepstack planes: column 0 = row index, column 1 = slot index.
        for (int32_t d = 0; d < numDeepstack; ++d)
        {
            std::vector<half> marker(static_cast<size_t>(batch) * maxLen * kHidden, __float2half(0.0F));
            for (int32_t b = 0; b < batch; ++b)
            {
                for (int32_t r = 0; r < maxLen; ++r)
                {
                    int64_t const off = (static_cast<int64_t>(b) * maxLen + r) * kHidden;
                    marker[off] = __float2half(float(r));
                    marker[off + 1] = __float2half(float(b));
                }
            }
            io.deepstackEmbeds.emplace_back(
                rt::Coords{batch, maxLen, kHidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "deepstack");
            CUDA_CHECK(cudaMemcpy(io.deepstackEmbeds.back().rawPointer(), marker.data(), marker.size() * sizeof(half),
                cudaMemcpyHostToDevice));
        }

        // Rope planes: marker = source row index in every column (identical per slot).
        std::vector<float> rope(static_cast<size_t>(batch) * kCapacity * kRotaryDim);
        for (int32_t b = 0; b < batch; ++b)
        {
            for (int32_t r = 0; r < kCapacity; ++r)
            {
                std::fill_n(
                    rope.begin() + (static_cast<int64_t>(b) * kCapacity + r) * kRotaryDim, kRotaryDim, float(r));
            }
        }
        io.mropeActiveCosSin
            = rt::Tensor({batch, kCapacity, kRotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "rope");
        CUDA_CHECK(cudaMemcpy(
            io.mropeActiveCosSin.rawPointer(), rope.data(), rope.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    //! Recover slot `b`'s (row, slot) marker pairs from the first deepstack plane at pitch `pitch`.
    std::vector<std::pair<int32_t, int32_t>> readDeepstackRows(int32_t b, int32_t pitch, int32_t count)
    {
        std::vector<half> out(static_cast<size_t>(count) * kHidden);
        auto const* base = static_cast<half const*>(io.deepstackEmbeds[0].rawPointer());
        CUDA_CHECK(cudaMemcpy(out.data(), base + static_cast<int64_t>(b) * pitch * kHidden, out.size() * sizeof(half),
            cudaMemcpyDeviceToHost));
        std::vector<std::pair<int32_t, int32_t>> rows(count);
        for (int32_t r = 0; r < count; ++r)
        {
            rows[r] = {static_cast<int32_t>(__half2float(out[static_cast<int64_t>(r) * kHidden])),
                static_cast<int32_t>(__half2float(out[static_cast<int64_t>(r) * kHidden + 1]))};
        }
        return rows;
    }

    std::vector<int32_t> readRopeRows(int32_t b, int32_t count)
    {
        std::vector<float> out(static_cast<size_t>(count) * kRotaryDim);
        auto const* base = static_cast<float const*>(io.mropeActiveCosSin.rawPointer());
        CUDA_CHECK(cudaMemcpy(out.data(), base + static_cast<int64_t>(b) * kCapacity * kRotaryDim,
            out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<int32_t> rows(count);
        for (int32_t r = 0; r < count; ++r)
        {
            rows[r] = static_cast<int32_t>(out[static_cast<int64_t>(r) * kRotaryDim]);
        }
        return rows;
    }

    //! Read slot `b`'s embedding rows at pitch `pitch` as floats.
    std::vector<float> readEmbeds(int32_t b, int32_t pitch, int32_t count)
    {
        std::vector<half> out(static_cast<size_t>(count) * kHidden);
        auto const* base = static_cast<half const*>(io.inputsEmbeds.rawPointer());
        CUDA_CHECK(cudaMemcpy(out.data(), base + static_cast<int64_t>(b) * pitch * kHidden, out.size() * sizeof(half),
            cudaMemcpyDeviceToHost));
        std::vector<float> vals(out.size());
        for (size_t i = 0; i < out.size(); ++i)
        {
            vals[i] = __half2float(out[i]);
        }
        return vals;
    }
};

} // namespace

TEST(DartGatherKernels, RowGatherMatchesCPU)
{
    constexpr int64_t kRows = 37;
    constexpr int64_t kRowBytes = 40; // not a multiple of 16 -> byte path
    std::vector<uint8_t> src(kRows * kRowBytes);
    std::mt19937 rng(7);
    for (auto& b : src)
    {
        b = static_cast<uint8_t>(rng());
    }
    std::vector<int32_t> indices = {5, 0, 36, 12, 12, 7};

    rt::Tensor srcDev({kRows * kRowBytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "src");
    rt::Tensor dstDev(
        {static_cast<int64_t>(indices.size()) * kRowBytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "dst");
    rt::Tensor idxDev({static_cast<int64_t>(indices.size())}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "idx");
    CUDA_CHECK(cudaMemcpy(srcDev.rawPointer(), src.data(), src.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(idxDev.rawPointer(), indices.data(), indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    kernel::gatherRows(dstDev.rawPointer(), srcDev.rawPointer(), idxDev.dataPointer<int32_t>(),
        static_cast<int64_t>(indices.size()), kRowBytes, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> dst(indices.size() * kRowBytes);
    CUDA_CHECK(cudaMemcpy(dst.data(), dstDev.rawPointer(), dst.size(), cudaMemcpyDeviceToHost));
    for (size_t r = 0; r < indices.size(); ++r)
    {
        for (int64_t b = 0; b < kRowBytes; ++b)
        {
            ASSERT_EQ(dst[r * kRowBytes + b], src[indices[r] * kRowBytes + b]);
        }
    }
}

TEST(DartPruner, MatchesReferenceSingleImage)
{
    rt::VisualPrunerConfig cfg;
    cfg.enabled = true;
    cfg.reductionRatio = 0.5F;
    auto tokens = makeTokens(20, 200, 30);
    PrunerHarness h(tokens, /*numDeepstack=*/2, /*seed=*/123);

    auto pruner = rt::createVisualTokenPruner(cfg, h.engineConfig);
    int32_t const prunedLen = pruner->pruneForPrefill(h.tokenIds, h.io, h.seqLen, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> const expected = referenceSelect(h.embedsFloat, h.seqLen, PrunerHarness::kHidden, tokens, cfg);
    ASSERT_EQ(prunedLen, static_cast<int32_t>(expected.size()));
    EXPECT_EQ(h.readDeepstackRows(prunedLen), expected);
    EXPECT_EQ(h.io.inputsEmbeds.getShape()[1], prunedLen);

    // Rope rows: [0, P) = keep indices, [P, cap - numPruned) = origLen, origLen+1, ...
    int32_t const numPruned = h.seqLen - prunedLen;
    int32_t const ropeRows = PrunerHarness::kCapacity - numPruned;
    std::vector<int32_t> const gotRope = h.readRopeRows(ropeRows);
    for (int32_t r = 0; r < prunedLen; ++r)
    {
        ASSERT_EQ(gotRope[r], expected[r]);
    }
    for (int32_t r = prunedLen; r < ropeRows; ++r)
    {
        ASSERT_EQ(gotRope[r], h.seqLen + (r - prunedLen));
    }
}

TEST(DartPruner, MatchesReferenceTwoImages)
{
    rt::VisualPrunerConfig cfg;
    cfg.enabled = true;
    cfg.reductionRatio = 0.25F;
    std::vector<int32_t> tokens = makeTokens(10, 120, 5);
    auto const second = makeTokens(8, 90, 12);
    tokens.insert(tokens.end(), second.begin(), second.end());
    PrunerHarness h(tokens, /*numDeepstack=*/1, /*seed=*/321);

    auto pruner = rt::createVisualTokenPruner(cfg, h.engineConfig);
    int32_t const prunedLen = pruner->pruneForPrefill(h.tokenIds, h.io, h.seqLen, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> const expected = referenceSelect(h.embedsFloat, h.seqLen, PrunerHarness::kHidden, tokens, cfg);
    ASSERT_EQ(prunedLen, static_cast<int32_t>(expected.size()));
    EXPECT_EQ(h.readDeepstackRows(prunedLen), expected);
}

TEST(DartPruner, PerImageQuotaPreventsStarvation)
{
    // One small (20-token) and one large (200-token) image: with per-image quotas the small
    // image must keep exactly ceil(20 * 0.5) = 10 tokens regardless of how "duplicated" its
    // content looks relative to the large image (a global pool could starve it entirely).
    rt::VisualPrunerConfig cfg;
    cfg.enabled = true;
    cfg.reductionRatio = 0.5F;
    std::vector<int32_t> tokens = makeTokens(10, 20, 5);
    auto const second = makeTokens(0, 200, 8);
    tokens.insert(tokens.end(), second.begin(), second.end());
    PrunerHarness h(tokens, /*numDeepstack=*/1, /*seed=*/555);

    auto pruner = rt::createVisualTokenPruner(cfg, h.engineConfig);
    int32_t const prunedLen = pruner->pruneForPrefill(h.tokenIds, h.io, h.seqLen, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Spans: [10, 30) len 20 -> quota 10; [35, 235) len 200 -> quota 100. Text = 23.
    ASSERT_EQ(prunedLen, 23 + 10 + 100);
    std::vector<int32_t> const rows = h.readDeepstackRows(prunedLen);
    int32_t smallKept = 0;
    int32_t largeKept = 0;
    for (int32_t r : rows)
    {
        smallKept += (r >= 10 && r < 30);
        largeKept += (r >= 35 && r < 235);
    }
    EXPECT_EQ(smallKept, 10);
    EXPECT_EQ(largeKept, 100);
}

TEST(DartPruner, RejectsInvalidPivotConfig)
{
    auto tokens = makeTokens(4, 32, 4);
    PrunerHarness h(tokens, 0, 21);
    auto makeCfg = [](int32_t imagePivots, int32_t textPivots) {
        rt::VisualPrunerConfig cfg;
        cfg.enabled = true;
        cfg.pivotImageTokens = imagePivots;
        cfg.pivotTextTokens = textPivots;
        return cfg;
    };
    // Negative counts must be rejected even when the sum passes the aggregate checks:
    // {-4, 8} would reach negative iterator arithmetic, {70, -10} would overflow the
    // pivot staging buffer. Each count is also bounded individually before the sum, so two
    // huge values cannot overflow the signed addition itself.
    EXPECT_THROW(rt::createVisualTokenPruner(makeCfg(-4, 8), h.engineConfig), std::runtime_error);
    EXPECT_THROW(rt::createVisualTokenPruner(makeCfg(70, -10), h.engineConfig), std::runtime_error);
    EXPECT_THROW(rt::createVisualTokenPruner(makeCfg(0, 0), h.engineConfig), std::runtime_error);
    EXPECT_THROW(rt::createVisualTokenPruner(makeCfg(40, 40), h.engineConfig), std::runtime_error);
    int32_t const intMax = std::numeric_limits<int32_t>::max();
    EXPECT_THROW(rt::createVisualTokenPruner(makeCfg(intMax, intMax), h.engineConfig), std::runtime_error);
    // Asymmetric-but-valid configs construct fine.
    EXPECT_NO_THROW(rt::createVisualTokenPruner(makeCfg(0, 4), h.engineConfig));
    EXPECT_NO_THROW(rt::createVisualTokenPruner(makeCfg(16, 0), h.engineConfig));
}

TEST(VisualTokenPruner, UnknownAlgorithmThrows)
{
    rt::VisualPrunerConfig cfg;
    cfg.enabled = true;
    cfg.algorithm = "no-such-algo";
    auto tokens = makeTokens(4, 32, 4);
    PrunerHarness h(tokens, 0, 5);
    EXPECT_THROW(rt::createVisualTokenPruner(cfg, h.engineConfig), std::runtime_error);
    auto const names = rt::registeredVisualPrunerNames();
    EXPECT_NE(std::find(names.begin(), names.end(), "dart"), names.end());
}

TEST(VisualTokenPruner, CustomSelectorRegistration)
{
    // A trivial "keep the first N visual tokens" selector registered at runtime.
    // A trivial custom pruner subclass: keeps the first N visual tokens via the base class's
    // shared compaction helper.
    class KeepFirstPruner final : public rt::VisualTokenPruner
    {
    public:
        KeepFirstPruner(rt::VisualPrunerConfig const& cfg, rt::LLMEngineConfig const& engineCfg)
            : rt::VisualTokenPruner(cfg, engineCfg)
        {
        }
        char const* name() const noexcept override
        {
            return "keep-first";
        }

    protected:
        int32_t prune(rt::PruneRequest const& req, rt::PipelineIO& io, cudaStream_t stream) override
        {
            std::vector<int32_t> const retained(
                req.imagePositions->begin(), req.imagePositions->begin() + req.targetImageTokens);
            return compactToKeepList(io, retained, req, stream);
        }
    };
    rt::registerVisualPruner("keep-first", [](rt::VisualPrunerConfig const& cfg, rt::LLMEngineConfig const& engineCfg) {
        return std::unique_ptr<rt::VisualTokenPruner>(std::make_unique<KeepFirstPruner>(cfg, engineCfg));
    });

    rt::VisualPrunerConfig cfg;
    cfg.enabled = true;
    cfg.algorithm = "keep-first";
    cfg.reductionRatio = 0.5F;
    auto tokens = makeTokens(8, 64, 8);
    PrunerHarness h(tokens, /*numDeepstack=*/1, /*seed=*/99);
    auto pruner = rt::createVisualTokenPruner(cfg, h.engineConfig);
    int32_t const prunedLen = pruner->pruneForPrefill(h.tokenIds, h.io, h.seqLen, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(prunedLen, 8 + 32 + 8);
    std::vector<int32_t> const rows = h.readDeepstackRows(prunedLen);
    for (int32_t i = 0; i < 8 + 32; ++i)
    {
        ASSERT_EQ(rows[i], i); // leading text + first 32 visual tokens are contiguous
    }
}

TEST(VisualTokenPruner, RejectsCustomPrunerWithoutSubsetProvenance)
{
    class RewritePruner final : public rt::VisualTokenPruner
    {
    public:
        RewritePruner(rt::VisualPrunerConfig const& cfg, rt::LLMEngineConfig const& engineCfg)
            : rt::VisualTokenPruner(cfg, engineCfg)
        {
        }

        char const* name() const noexcept override
        {
            return "rewrite";
        }

    protected:
        int32_t prune(rt::PruneRequest const& req, rt::PipelineIO&, cudaStream_t) override
        {
            return req.origLen - 1;
        }
    };

    rt::registerVisualPruner("rewrite", [](rt::VisualPrunerConfig const& cfg, rt::LLMEngineConfig const& engineCfg) {
        return std::unique_ptr<rt::VisualTokenPruner>(std::make_unique<RewritePruner>(cfg, engineCfg));
    });

    rt::VisualPrunerConfig cfg;
    cfg.enabled = true;
    cfg.algorithm = "rewrite";
    cfg.reductionRatio = 0.5F;
    auto tokens = makeTokens(8, 64, 8);
    PrunerHarness h(tokens, /*numDeepstack=*/0, /*seed=*/101);
    auto pruner = rt::createVisualTokenPruner(cfg, h.engineConfig);

    EXPECT_THROW(pruner->pruneForPrefill(h.tokenIds, h.io, h.seqLen, nullptr), std::runtime_error);
}

TEST(VisualTokenPruner, RejectsInvalidRetainedIndices)
{
    // A misbehaving custom pruner returning configurable bad indices must be caught by
    // compactToKeepList before any device gather runs.
    static std::vector<int32_t> badIndices;
    class BadPruner final : public rt::VisualTokenPruner
    {
    public:
        BadPruner(rt::VisualPrunerConfig const& cfg, rt::LLMEngineConfig const& engineCfg)
            : rt::VisualTokenPruner(cfg, engineCfg)
        {
        }
        char const* name() const noexcept override
        {
            return "bad";
        }

    protected:
        int32_t prune(rt::PruneRequest const& req, rt::PipelineIO& io, cudaStream_t stream) override
        {
            return compactToKeepList(io, badIndices, req, stream);
        }
    };
    rt::registerVisualPruner("bad", [](rt::VisualPrunerConfig const& cfg, rt::LLMEngineConfig const& engineCfg) {
        return std::unique_ptr<rt::VisualTokenPruner>(std::make_unique<BadPruner>(cfg, engineCfg));
    });

    rt::VisualPrunerConfig cfg;
    cfg.enabled = true;
    cfg.algorithm = "bad";
    cfg.reductionRatio = 0.5F;
    auto tokens = makeTokens(8, 64, 8); // visual positions are [8, 72)
    PrunerHarness h(tokens, 0, 11);
    auto pruner = rt::createVisualTokenPruner(cfg, h.engineConfig);

    badIndices = {8, 9, 10, 1000}; // out of range
    EXPECT_THROW(pruner->pruneForPrefill(h.tokenIds, h.io, h.seqLen, nullptr), std::runtime_error);
    badIndices = {8, 9, 9, 10}; // duplicate
    EXPECT_THROW(pruner->pruneForPrefill(h.tokenIds, h.io, h.seqLen, nullptr), std::runtime_error);
    badIndices = {0, 8, 9, 10}; // position 0 is a text token
    EXPECT_THROW(pruner->pruneForPrefill(h.tokenIds, h.io, h.seqLen, nullptr), std::runtime_error);
    badIndices = {8, 9, 10, 11}; // valid set still works
    EXPECT_EQ(pruner->pruneForPrefill(h.tokenIds, h.io, h.seqLen, nullptr), 8 + 4 + 8);
    CUDA_CHECK(cudaDeviceSynchronize());
}

//! Build a marker feature tensor [rows, hidden]: column 0 = row index.
static rt::Tensor makeMarkerFeatures(int32_t rows, int32_t hidden)
{
    std::vector<half> host(static_cast<size_t>(rows) * hidden, __float2half(0.0F));
    for (int32_t r = 0; r < rows; ++r)
    {
        host[static_cast<int64_t>(r) * hidden] = __float2half(float(r));
    }
    rt::Tensor t({rows, hidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "features");
    CUDA_CHECK(cudaMemcpy(t.rawPointer(), host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice));
    return t;
}

static std::vector<int32_t> readMarkerRows(rt::Tensor const& t)
{
    auto const shape = t.getShape();
    std::vector<half> host(static_cast<size_t>(shape.volume()));
    CUDA_CHECK(cudaMemcpy(host.data(), t.rawPointer(), host.size() * sizeof(half), cudaMemcpyDeviceToHost));
    std::vector<int32_t> rows(shape[0]);
    for (int64_t r = 0; r < shape[0]; ++r)
    {
        rows[r] = static_cast<int32_t>(__half2float(host[r * shape[1]]));
    }
    return rows;
}

TEST(DartPruner, CompactAuxiliaryInputsMatchesKeepLists)
{
    // Two slots: one prunable image request, one text-only passthrough. After the pruning pass,
    // compactAuxiliaryInputs must shorten each slot's token ids to its keep list and gather the
    // kept visual feature rows at their original global ordinals.
    rt::VisualPrunerConfig cfg;
    cfg.enabled = true;
    cfg.reductionRatio = 0.5F;
    std::vector<std::vector<int32_t>> tokens = {makeTokens(20, 200, 30), makeTokens(120, 0, 0)};
    BatchPrunerHarness h(tokens, /*numDeepstack=*/1, /*seed=*/808);
    auto pruner = rt::createVisualTokenPruner(cfg, h.engineConfig);

    std::vector<int32_t> effectiveLens = h.lens;
    std::vector<int32_t> prunedPerSlot;
    int32_t const newMaxLen
        = pruner->pruneBatchForPrefill(h.tokenIds, h.io, effectiveLens, h.maxLen, prunedPerSlot, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_GT(prunedPerSlot[0], 0);
    ASSERT_EQ(prunedPerSlot[1], 0);

    // Expected keep list for slot 0 from the CPU reference.
    std::vector<int32_t> const expected
        = referenceSelect(h.embedsFloat[0], h.lens[0], BatchPrunerHarness::kHidden, tokens[0], cfg);
    ASSERT_EQ(effectiveLens[0], static_cast<int32_t>(expected.size()));

    // Raw feature tensors: 200 visual rows (all from slot 0), marker = row index.
    rt::Tensor const visualSrc = makeMarkerFeatures(200, BatchPrunerHarness::kHidden);
    rt::OptionalInputTensor visual{std::cref(visualSrc)};

    auto tokenIds = tokens; // copy: compacted in place
    pruner->compactAuxiliaryInputs(tokenIds, h.batch, prunedPerSlot, visual, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Token ids compacted to the keep list; text-only slot untouched.
    ASSERT_EQ(static_cast<int32_t>(tokenIds[0].size()), effectiveLens[0]);
    for (size_t k = 0; k < expected.size(); ++k)
    {
        ASSERT_EQ(tokenIds[0][k], tokens[0][expected[k]]);
    }
    ASSERT_EQ(tokenIds[1], tokens[1]);

    // Feature refs rebound to compacted rows holding the kept original ordinals (span [20, 220)
    // of the sequence maps to feature rows 0..200).
    ASSERT_NE(&visual->get(), &visualSrc);
    std::vector<int32_t> expectedOrdinals;
    for (int32_t pos : expected)
    {
        if (tokens[0][pos] == kImageTokenId)
        {
            expectedOrdinals.push_back(pos - 20);
        }
    }
    EXPECT_EQ(readMarkerRows(visual->get()), expectedOrdinals);

    // Nothing-pruned request: refs and ids stay untouched.
    std::vector<std::vector<int32_t>> textTokens = {makeTokens(64, 0, 0)};
    PrunerHarness textOnly(textTokens[0], 0, 9);
    auto textPruner = rt::createVisualTokenPruner(cfg, textOnly.engineConfig);
    EXPECT_EQ(textPruner->pruneForPrefill(textOnly.tokenIds, textOnly.io, textOnly.seqLen, nullptr), textOnly.seqLen);
    rt::OptionalInputTensor noVisual{std::cref(visualSrc)};
    std::vector<int32_t> const noPruned = {0};
    textPruner->compactAuxiliaryInputs(textTokens, 1, noPruned, noVisual, nullptr);
    EXPECT_EQ(&noVisual->get(), &visualSrc);
    EXPECT_EQ(textTokens[0], std::vector<int32_t>(64, kTextTokenId));
    (void) newMaxLen;
}

TEST(DartPruner, CompactAuxiliaryInputsTwoImageSlots)
{
    // Two image-bearing slots plus a text-only slot: the kept feature ordinals of the second
    // image slot must be offset by the FIRST slot's ORIGINAL (unpruned) visual count, matching
    // the batch-major running image-token count the embedding kernel recomputes after
    // compaction.
    rt::VisualPrunerConfig cfg;
    cfg.enabled = true;
    cfg.reductionRatio = 0.5F;
    std::vector<std::vector<int32_t>> tokens = {makeTokens(20, 60, 10), makeTokens(5, 80, 15), makeTokens(90, 0, 0)};
    BatchPrunerHarness h(tokens, /*numDeepstack=*/1, /*seed=*/4242);
    auto pruner = rt::createVisualTokenPruner(cfg, h.engineConfig);
    // Exercise the preallocated path (spec-decode deployments size the buffers up front);
    // the sibling test covers the lazily-grown path.
    pruner->preallocateAuxiliaryBuffers();

    std::vector<int32_t> effectiveLens = h.lens;
    std::vector<int32_t> prunedPerSlot;
    pruner->pruneBatchForPrefill(h.tokenIds, h.io, effectiveLens, h.maxLen, prunedPerSlot, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_GT(prunedPerSlot[0], 0);
    ASSERT_GT(prunedPerSlot[1], 0);
    ASSERT_EQ(prunedPerSlot[2], 0);

    rt::Tensor const visualSrc = makeMarkerFeatures(60 + 80, BatchPrunerHarness::kHidden);
    rt::OptionalInputTensor visual{std::cref(visualSrc)};
    auto tokenIds = tokens;
    pruner->compactAuxiliaryInputs(tokenIds, h.batch, prunedPerSlot, visual, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Expected global feature ordinals: slot 0 kept visual positions map to [0, 60) via
    // (pos - 20); slot 1 kept visual positions map to [60, 140) via 60 + (pos - 5).
    std::vector<int32_t> expectedOrdinals;
    struct SlotSpan
    {
        int32_t slot;
        int32_t visualBegin;
        int32_t ordinalBase;
    };
    for (auto const& span : {SlotSpan{0, 20, 0}, SlotSpan{1, 5, 60}})
    {
        std::vector<int32_t> const keep = referenceSelect(
            h.embedsFloat[span.slot], h.lens[span.slot], BatchPrunerHarness::kHidden, tokens[span.slot], cfg);
        ASSERT_EQ(effectiveLens[span.slot], static_cast<int32_t>(keep.size()));
        for (size_t k = 0; k < keep.size(); ++k)
        {
            ASSERT_EQ(tokenIds[span.slot][k], tokens[span.slot][keep[k]]);
            if (tokens[span.slot][keep[k]] == kImageTokenId)
            {
                expectedOrdinals.push_back(span.ordinalBase + keep[k] - span.visualBegin);
            }
        }
    }
    ASSERT_EQ(tokenIds[2], tokens[2]); // text-only slot untouched
    EXPECT_EQ(readMarkerRows(visual->get()), expectedOrdinals);
}

TEST(DartPruner, BatchMatchesPerSlotReference)
{
    // Three slots with mixed lengths and content: image-heavy, text-only (passthrough), and a
    // second, shorter image request. Each pruned slot must match the single-request reference
    // on its own embeddings, at the new common row pitch, with no cross-slot mixing.
    rt::VisualPrunerConfig cfg;
    cfg.enabled = true;
    cfg.reductionRatio = 0.5F;
    std::vector<std::vector<int32_t>> tokens = {makeTokens(20, 200, 30), makeTokens(150, 0, 0), makeTokens(10, 96, 14)};
    BatchPrunerHarness h(tokens, /*numDeepstack=*/2, /*seed=*/2024);

    auto pruner = rt::createVisualTokenPruner(cfg, h.engineConfig);
    std::vector<int32_t> effectiveLens = h.lens;
    std::vector<int32_t> prunedPerSlot;
    int32_t const newMaxLen
        = pruner->pruneBatchForPrefill(h.tokenIds, h.io, effectiveLens, h.maxLen, prunedPerSlot, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ASSERT_EQ(h.io.inputsEmbeds.getShape()[1], newMaxLen);
    int32_t expectedMax = 0;
    for (int32_t b = 0; b < h.batch; ++b)
    {
        std::vector<int32_t> const expected
            = referenceSelect(h.embedsFloat[b], h.lens[b], BatchPrunerHarness::kHidden, h.tokenIds[b], cfg);
        int32_t const expectedLen = static_cast<int32_t>(expected.size());
        expectedMax = std::max(expectedMax, expectedLen);
        ASSERT_EQ(effectiveLens[b], expectedLen) << "slot " << b;
        ASSERT_EQ(prunedPerSlot[b], h.lens[b] - expectedLen) << "slot " << b;

        // Deepstack markers: kept source rows in order, all from this slot.
        auto const rows = h.readDeepstackRows(b, newMaxLen, expectedLen);
        for (int32_t r = 0; r < expectedLen; ++r)
        {
            ASSERT_EQ(rows[r].first, expected[r]) << "slot " << b << " row " << r;
            ASSERT_EQ(rows[r].second, b) << "slot " << b << " row " << r;
        }

        // Embedding rows: bit-exact copies of the kept source rows.
        auto const got = h.readEmbeds(b, newMaxLen, expectedLen);
        for (int32_t r = 0; r < expectedLen; ++r)
        {
            for (int32_t c = 0; c < BatchPrunerHarness::kHidden; ++c)
            {
                ASSERT_EQ(got[static_cast<int64_t>(r) * BatchPrunerHarness::kHidden + c],
                    h.embedsFloat[b][static_cast<int64_t>(expected[r]) * BatchPrunerHarness::kHidden + c])
                    << "slot " << b << " row " << r;
            }
        }

        // Rope rows: pruned slots get keep + shifted continuation; unpruned slots stay identity.
        int32_t const numPruned = h.lens[b] - expectedLen;
        int32_t const ropeRows = BatchPrunerHarness::kCapacity - numPruned;
        auto const gotRope = h.readRopeRows(b, ropeRows);
        for (int32_t r = 0; r < expectedLen; ++r)
        {
            ASSERT_EQ(gotRope[r], expected[r]) << "slot " << b << " rope row " << r;
        }
        for (int32_t r = expectedLen; r < ropeRows; ++r)
        {
            ASSERT_EQ(gotRope[r], h.lens[b] + (r - expectedLen)) << "slot " << b << " rope row " << r;
        }
    }
    ASSERT_EQ(newMaxLen, expectedMax);
}

TEST(DartPruner, BatchAllSlotsSkippedLeavesBuffersUntouched)
{
    rt::VisualPrunerConfig cfg;
    cfg.enabled = true;
    cfg.reductionRatio = 0.5F;
    cfg.minVisualTokens = 16;
    // Text-only and below-min-visual slots: nothing prunable in the whole batch.
    std::vector<std::vector<int32_t>> tokens = {makeTokens(120, 0, 0), makeTokens(30, 8, 10)};
    BatchPrunerHarness h(tokens, /*numDeepstack=*/1, /*seed=*/77);

    auto pruner = rt::createVisualTokenPruner(cfg, h.engineConfig);
    std::vector<int32_t> effectiveLens = h.lens;
    std::vector<int32_t> prunedPerSlot;
    int32_t const newMaxLen
        = pruner->pruneBatchForPrefill(h.tokenIds, h.io, effectiveLens, h.maxLen, prunedPerSlot, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_EQ(newMaxLen, h.maxLen);
    EXPECT_EQ(effectiveLens, h.lens);
    EXPECT_EQ(prunedPerSlot, std::vector<int32_t>(h.batch, 0));
    EXPECT_EQ(h.io.inputsEmbeds.getShape()[1], h.maxLen);
    for (int32_t b = 0; b < h.batch; ++b)
    {
        auto const rows = h.readDeepstackRows(b, h.maxLen, h.lens[b]);
        for (int32_t r = 0; r < h.lens[b]; ++r)
        {
            ASSERT_EQ(rows[r].first, r);
            ASSERT_EQ(rows[r].second, b);
        }
    }
}

TEST(DartPruner, SkipsWhenGuardsFire)
{
    rt::VisualPrunerConfig cfg;
    cfg.enabled = true;
    cfg.reductionRatio = 0.5F;
    cfg.minVisualTokens = 16;

    // No visual tokens at all.
    {
        auto tokens = makeTokens(64, 0, 0);
        PrunerHarness h(tokens, 0, 1);
        auto pruner = rt::createVisualTokenPruner(cfg, h.engineConfig);
        EXPECT_EQ(pruner->pruneForPrefill(h.tokenIds, h.io, h.seqLen, nullptr), h.seqLen);
    }
    // Fewer visual tokens than minVisualTokens.
    {
        auto tokens = makeTokens(30, 8, 10);
        PrunerHarness h(tokens, 0, 2);
        auto pruner = rt::createVisualTokenPruner(cfg, h.engineConfig);
        EXPECT_EQ(pruner->pruneForPrefill(h.tokenIds, h.io, h.seqLen, nullptr), h.seqLen);
    }
    // Reduction so small the target is not an actual reduction (2 visual tokens, r small).
    {
        rt::VisualPrunerConfig tiny = cfg;
        tiny.minVisualTokens = 1;
        tiny.reductionRatio = 0.01F;
        auto tokens = makeTokens(10, 20, 10);
        PrunerHarness h(tokens, 0, 3);
        auto pruner = rt::createVisualTokenPruner(tiny, h.engineConfig);
        EXPECT_EQ(pruner->pruneForPrefill(h.tokenIds, h.io, h.seqLen, nullptr), h.seqLen);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
