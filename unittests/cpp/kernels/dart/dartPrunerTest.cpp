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
        io.mropeCosSin
            = rt::Tensor({1, kCapacity, kRotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "rope");
        CUDA_CHECK(
            cudaMemcpy(io.mropeCosSin.rawPointer(), rope.data(), rope.size() * sizeof(float), cudaMemcpyHostToDevice));
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
        CUDA_CHECK(
            cudaMemcpy(out.data(), io.mropeCosSin.rawPointer(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
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
