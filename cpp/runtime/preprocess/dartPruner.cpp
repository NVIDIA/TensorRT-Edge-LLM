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

#include "runtime/preprocess/dartPruner.h"

#include "common/checkMacros.h"
#include "kernels/dart/dartSelectorKernels.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

namespace
{

//! Cosine denominator floor, matching torch.nn.functional.cosine_similarity's eps.
constexpr float kCosineEps = 1e-8F;

//! Top-`count` indices of `positions` by descending score, ties broken by lower index.
std::vector<int32_t> topByScore(std::vector<int32_t> const& positions, float const* scores, int32_t count)
{
    std::vector<int32_t> sorted = positions;
    count = std::min<int32_t>(count, static_cast<int32_t>(sorted.size()));
    if (count <= 0)
    {
        return {};
    }
    std::partial_sort(sorted.begin(), sorted.begin() + count, sorted.end(), [scores](int32_t a, int32_t b) {
        if (scores[a] != scores[b])
        {
            return scores[a] > scores[b];
        }
        return a < b;
    });
    sorted.resize(count);
    return sorted;
}

} // namespace

DartPruner::DartPruner(VisualPrunerConfig const& config, LLMEngineConfig const& engineConfig)
    : VisualTokenPruner(config, engineConfig)
    , mPivotImageTokens(config.pivotImageTokens)
    , mPivotTextTokens(config.pivotTextTokens)
{
    // Validate each count independently before the sum: a negative count admitted through a
    // sum-only check would reach invalid iterator arithmetic (negative numImagePivots) or let
    // the other count exceed kDartMaxPivots and overflow the pivot staging buffers — and two
    // huge counts would overflow the signed sum itself before any sum-based check runs.
    check::check(
        mPivotImageTokens >= 0 && mPivotImageTokens <= kernel::kDartMaxPivots, "DART pivotImageTokens out of range");
    check::check(
        mPivotTextTokens >= 0 && mPivotTextTokens <= kernel::kDartMaxPivots, "DART pivotTextTokens out of range");
    check::check(mPivotImageTokens + mPivotTextTokens > 0, "DART requires at least one pivot");
    check::check(
        mPivotImageTokens + mPivotTextTokens <= kernel::kDartMaxPivots, "DART pivot count exceeds kernel limit");
    int32_t const maxInputLen = engineConfig.maxSupportedInputLength;
    // Reserve the host selection scratch up front so per-request/per-span reuse never reallocates.
    mL1Scores.reserve(maxInputLen);
    mInRetained.reserve(maxInputLen);
    mSimScores.reserve(maxInputLen);
    mNormsDevice = Tensor({maxInputLen, 2}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "DartPruner::normsDevice");
    mNormsHost = Tensor({maxInputLen, 2}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "DartPruner::normsHost");
    mDotsDevice = Tensor(
        {kernel::kDartMaxPivots, maxInputLen}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "DartPruner::dots");
    mDotsHost = Tensor(
        {kernel::kDartMaxPivots, maxInputLen}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "DartPruner::dotsHost");
    mPivotIdxDevice
        = Tensor({kernel::kDartMaxPivots}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DartPruner::pivotIdxDevice");
    mPivotIdxHost
        = Tensor({kernel::kDartMaxPivots}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DartPruner::pivotIdxHost");
}

int32_t DartPruner::prune(PruneRequest const& req, PipelineIO& io, cudaStream_t stream)
{
    selectRetainedImageTokens(req, stream);
    return compactToKeepList(io, mRetained, req, stream);
}

void DartPruner::selectRetainedImageTokens(PruneRequest const& req, cudaStream_t stream)
{
    std::vector<int32_t> const& textPositions = *req.textPositions;
    std::vector<ImageSpan> const& spans = *req.imageSpans;
    int32_t const origLen = req.origLen;

    // Stage 1: row L1/L2 norms (device) -> host, once for the whole sequence.
    check::check(mNormsDevice.reshape({origLen, 2}), "Tensor reshape failed");
    kernel::computeDartRowNorms(*req.embeds, mNormsDevice, stream);
    CUDA_CHECK(cudaMemcpyAsync(mNormsHost.rawPointer(), mNormsDevice.rawPointer(),
        static_cast<size_t>(origLen) * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float const* norms = mNormsHost.dataPointer<float>();

    mL1Scores.resize(origLen);
    for (int32_t i = 0; i < origLen; ++i)
    {
        mL1Scores[i] = norms[i * 2 + 0];
    }
    // Text pivots are shared across all spans (mirroring the reference's per-camera mode,
    // where the fixed instruction/text tokens drive every camera's selection). Image pivots
    // are picked per span on the host — norms are already resident, no extra device work.
    std::vector<int32_t> const textPivots = topByScore(textPositions, mL1Scores.data(), mPivotTextTokens);
    std::vector<std::vector<int32_t>> spanPivots(spans.size());
    for (size_t s = 0; s < spans.size(); ++s)
    {
        std::vector<int32_t> spanPositions(spans[s].end - spans[s].begin);
        std::iota(spanPositions.begin(), spanPositions.end(), spans[s].begin);
        spanPivots[s] = topByScore(spanPositions, mL1Scores.data(), std::min(mPivotImageTokens, spans[s].targetTokens));
    }

    // Stages 2+3: batch the pivot-vs-all dot products of as many spans as fit into the pivot
    // buffer per launch (typically all of them), so the synchronization count stays fixed
    // instead of growing with the image count. Each chunk shares one copy of the text pivots.
    mRetained.clear();
    int32_t const textCount = static_cast<int32_t>(textPivots.size());
    size_t chunkBegin = 0;
    while (chunkBegin < spans.size())
    {
        int32_t* pivotHost = mPivotIdxHost.dataPointer<int32_t>();
        int32_t numPivots = 0;
        std::vector<int32_t> spanPivotOffsets; // offset of each chunk member's pivots in pivotHost
        size_t chunkEnd = chunkBegin;
        while (chunkEnd < spans.size())
        {
            int32_t const spanCount = static_cast<int32_t>(spanPivots[chunkEnd].size());
            if (numPivots + spanCount + textCount > kernel::kDartMaxPivots && chunkEnd > chunkBegin)
            {
                break; // chunk full; a single span always fits (ctor bounds pivot counts)
            }
            spanPivotOffsets.push_back(numPivots);
            std::copy(spanPivots[chunkEnd].begin(), spanPivots[chunkEnd].end(), pivotHost + numPivots);
            numPivots += spanCount;
            ++chunkEnd;
        }
        int32_t const textOffset = numPivots;
        std::copy(textPivots.begin(), textPivots.end(), pivotHost + numPivots);
        numPivots += textCount;
        check::check(numPivots > 0, "DART selected no pivots");

        CUDA_CHECK(cudaMemcpyAsync(mPivotIdxDevice.rawPointer(), pivotHost,
            static_cast<size_t>(numPivots) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        check::check(mDotsDevice.reshape({numPivots, origLen}), "Tensor reshape failed");
        kernel::computeDartPivotDots(
            *req.embeds, mPivotIdxDevice.dataPointer<int32_t>(), numPivots, mDotsDevice, stream);
        CUDA_CHECK(cudaMemcpyAsync(mDotsHost.rawPointer(), mDotsDevice.rawPointer(),
            static_cast<size_t>(numPivots) * origLen * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (size_t s = chunkBegin; s < chunkEnd; ++s)
        {
            growSpanGreedily(req, spans[s], spanPivotOffsets[s - chunkBegin],
                static_cast<int32_t>(spanPivots[s].size()), textOffset, textCount, norms);
        }
        chunkBegin = chunkEnd;
    }
}

void DartPruner::growSpanGreedily(PruneRequest const& req, ImageSpan const& span, int32_t spanPivotOffset,
    int32_t numSpanPivots, int32_t textPivotOffset, int32_t numTextPivots, float const* norms)
{
    int32_t const origLen = req.origLen;
    int32_t const target = span.targetTokens;
    int32_t const* pivotHost = mPivotIdxHost.dataPointer<int32_t>();
    float const* dots = mDotsHost.dataPointer<float>();

    // Pivot rows for this span, in reference order: the span's own image pivots, then the
    // shared text pivots (both already present in the chunk's dots matrix).
    std::vector<int32_t> pivotRows(numSpanPivots + numTextPivots);
    std::iota(pivotRows.begin(), pivotRows.begin() + numSpanPivots, spanPivotOffset);
    std::iota(pivotRows.begin() + numSpanPivots, pivotRows.end(), textPivotOffset);
    int32_t const numPivots = static_cast<int32_t>(pivotRows.size());

    // Greedy anti-duplication growth on the host, restricted to this span (mirror of the
    // reference: per pivot, keep the candidates with the most negative cosine similarity;
    // the pairwise similarities are fixed, only the candidate pool shrinks).
    size_t const retainedBefore = mRetained.size();
    mInRetained.assign(origLen, 0);
    for (int32_t i = 0; i < numSpanPivots; ++i)
    {
        int32_t const pos = pivotHost[spanPivotOffset + i];
        mRetained.push_back(pos);
        mInRetained[pos] = 1;
    }
    std::vector<int32_t> candidates;
    candidates.reserve(span.end - span.begin);
    for (int32_t pos = span.begin; pos < span.end; ++pos)
    {
        if (!mInRetained[pos])
        {
            candidates.push_back(pos);
        }
    }

    // Similarity scores are overwritten per candidate before every read, so only the range is sized.
    mSimScores.resize(origLen);
    for (int32_t pivotOffset = 0; pivotOffset < numPivots; ++pivotOffset)
    {
        int32_t const remaining = target - static_cast<int32_t>(mRetained.size() - retainedBefore);
        if (remaining <= 0 || candidates.empty())
        {
            break;
        }
        int32_t const perPivot = (remaining + (numPivots - pivotOffset) - 1) / (numPivots - pivotOffset);
        int32_t const pivotRow = pivotRows[pivotOffset];
        float const pivotL2 = norms[pivotHost[pivotRow] * 2 + 1];
        for (int32_t c : candidates)
        {
            float const denom = std::max(pivotL2 * norms[c * 2 + 1], kCosineEps);
            mSimScores[c] = -(dots[static_cast<int64_t>(pivotRow) * origLen + c] / denom);
        }
        std::vector<int32_t> const chosen = topByScore(candidates, mSimScores.data(), perPivot);
        for (int32_t c : chosen)
        {
            mRetained.push_back(c);
            mInRetained[c] = 1;
        }
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(), [this](int32_t c) { return mInRetained[c] != 0; }),
            candidates.end());
    }
}

} // namespace rt
} // namespace trt_edgellm
