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

#include "multimodal/qwen3/qwen3vlViTRunner.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "multimodal/common/imageUtils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace trt_edgellm
{
namespace rt
{

bool Qwen3VLViTRunner::validateExtraConfig(nlohmann::json const& jsonConfig)
{
    auto visionConfig = jsonConfig["vision_config"];
    auto numPositionEmbeddings = visionConfig["num_position_embeddings"].get<int64_t>();
    mNumGridPerSide = static_cast<int64_t>(std::sqrt(numPositionEmbeddings));
    auto deepstackIndexes = visionConfig.value("deepstack_visual_indexes", std::vector<int64_t>{});
    mNumDeepstackFeatures = deepstackIndexes.size(); // 0 for Qwen3.5 (no deepstack)
    mConfig.mropeInterleaved = true;                 // Qwen3-VL/3.5/Omni use interleaved MRoPE (architectural)
    return true;
}

bool Qwen3VLViTRunner::allocateExtraBuffers(int64_t maxImageTokens)
{
    bool setTensorAddressStatus{true};
    mFastPosEmbIdx = rt::Tensor(
        {4, mConfig.maxHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "Qwen3VLViTRunner::mFastPosEmbIdx");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kFastPosEmbIdx, mFastPosEmbIdx.rawPointer());

    mFastPosEmbWeight = rt::Tensor(
        {4, mConfig.maxHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "Qwen3VLViTRunner::mFastPosEmbWeight");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kFastPosEmbWeight, mFastPosEmbWeight.rawPointer());

    for (int64_t i = 0; i < mNumDeepstackFeatures; ++i)
    {
        std::string const deepstackFeatureName = binding_names::formatDeepstackFeaturesName(i);
        mDeepstackFeatures.emplace_back(rt::Tensor({maxImageTokens, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, deepstackFeatureName));
        setTensorAddressStatus
            &= mVisualContext->setTensorAddress(deepstackFeatureName.c_str(), mDeepstackFeatures.back().rawPointer());
    }
    return setTensorAddressStatus;
}

void Qwen3VLViTRunner::buildExtraInputs(
    std::vector<VisionSpan> const& spans, int64_t totalSeqLength, int64_t totalImageTokens, cudaStream_t stream)
{
    check::check(mFastPosEmbIdx.reshape({4, totalSeqLength}), "Tensor reshape failed");
    check::check(mFastPosEmbWeight.reshape({4, totalSeqLength}), "Tensor reshape failed");

    // Each span's gridT*gridH*gridW fast-pos-emb entries are written starting at vit.patchStart.
    for (auto const& s : spans)
    {
        kernel::initFastPosEmbedQwenViT(mFastPosEmbIdx, mFastPosEmbWeight, {s.vit.gridT, s.vit.gridH, s.vit.gridW},
            mConfig.mergeSize, mNumGridPerSide, s.vit.patchStart, stream);
    }

    for (int64_t i = 0; i < mNumDeepstackFeatures; ++i)
    {
        check::check(mDeepstackFeatures[i].reshape({totalImageTokens, mConfig.outHiddenSize}), "Tensor reshape failed");
    }
}

bool Qwen3VLViTRunner::bindExtraInputShapes()
{
    bool setEngineIOStatus{true};
    setEngineIOStatus
        &= mVisualContext->setInputShape(binding_names::kFastPosEmbIdx, mFastPosEmbIdx.getShape().getTRTDims());
    setEngineIOStatus
        &= mVisualContext->setInputShape(binding_names::kFastPosEmbWeight, mFastPosEmbWeight.getShape().getTRTDims());
    return setEngineIOStatus;
}

std::tuple<int64_t, int64_t> Qwen3VLViTRunner::getResizedImageSize(
    int64_t numFrames, bool isVideo, int64_t height, int64_t width, int64_t maxRatio)
{
    return rt::imageUtils::qwenSmartResize3D(numFrames, isVideo, height, width, mConfig.patchSize, mConfig.mergeSize,
        mConfig.minImageTokensPerImage, mConfig.maxImageTokensPerImage, mConfig.temporalPatchSize, maxRatio);
}

std::tuple<int64_t, int64_t> Qwen3VLViTRunner::computeVisionSpans(
    rt::imageUtils::ImageData const& image, int64_t patchBase, std::vector<VisionSpan>& spans)
{
    int64_t const height = image.height;
    int64_t const width = image.width;
    ELLM_CHECK(
        height % (mConfig.patchSize * mConfig.mergeSize) == 0 && width % (mConfig.patchSize * mConfig.mergeSize) == 0,
        "Image height or width is not divisible by patchSize * mergeSize = "
            + std::to_string(mConfig.patchSize * mConfig.mergeSize) + " got height: " + std::to_string(height)
            + ", width: " + std::to_string(width));

    int64_t const gridT = (image.frames + mConfig.temporalPatchSize - 1) / mConfig.temporalPatchSize;
    int64_t const gridH = height / mConfig.patchSize; // HF grid_h (patch units)
    int64_t const gridW = width / mConfig.patchSize;  // HF grid_w (patch units)
    int64_t const patchesPerFrame = gridH * gridW;
    int64_t const llmGridH = gridH / mConfig.mergeSize; // HF llm_grid_h
    int64_t const llmGridW = gridW / mConfig.mergeSize; // HF llm_grid_w
    int64_t const tokensPerFrame = llmGridH * llmGridW;

    // One frame -> single flat span (== the per-frame loop at gridT==1); image-vs-video is decided in textPreprocess.
    if (image.frames <= 1)
    {
        VisionSpan span;
        span.llm
            = LlmVisionBlock{tokensPerFrame, /*llmGridT*/ 1, llmGridH, llmGridW}; // secondPerGrid unused (gridT==1)
        span.vit = VitFrameGrid{/*gridT*/ 1, gridH, gridW, patchBase};
        spans.push_back(span);
        return {/*totalSeqLen*/ patchesPerFrame, /*totalGridT*/ 1};
    }

    // Video: per-frame sub-spans, each gridT==1.
    // The <X.X seconds> timestamp marker is a textPreprocess concern (recomputed there from frames/fps).
    int64_t patch = patchBase;
    for (int64_t t = 0; t < gridT; ++t)
    {
        VisionSpan span;
        span.llm = LlmVisionBlock{tokensPerFrame, /*llmGridT*/ 1, llmGridH, llmGridW};
        span.vit = VitFrameGrid{/*gridT*/ 1, gridH, gridW, patch};
        spans.push_back(span);
        patch += patchesPerFrame;
    }
    // gridT sub-spans, each 1 frame of patchesPerFrame patches.
    return {/*totalSeqLen*/ gridT * patchesPerFrame, /*totalGridT*/ gridT};
}

rt::OptionalInputTensors Qwen3VLViTRunner::getDeepstackFeatures()
{
    if (mNumDeepstackFeatures == 0) // Qwen3.5: no deepstack
    {
        return {};
    }
    std::vector<std::reference_wrapper<rt::Tensor const>> refs;
    refs.reserve(mDeepstackFeatures.size());
    for (auto const& tensor : mDeepstackFeatures)
    {
        refs.emplace_back(std::cref(tensor));
    }
    return refs;
}

void Qwen3VLViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<VisionSpan> const& spans,
    std::vector<int64_t> const& spansPerRequest, trt_edgellm::tokenizer::Tokenizer const* tokenizer)
{
    // Pads expand to copies of mConfig.imageTokenId (embeddingLookup fills them in order). Two paths:
    //   (a) VIDEO: the <|vision_start|><|video_pad|><|vision_end|> triplet -> one timestamped (<X.X s> + vision_start
    //       + pads + vision_end) group per per-frame sub-span. Detected by the token triplet itself.
    //   (b) IMAGE (and any non-video pad): one flat pad run.
    ELLM_CHECK(spansPerRequest.size() == request.requests.size(),
        "spansPerRequest.size() != request.requests.size(), " + std::to_string(spansPerRequest.size())
            + " != " + std::to_string(request.requests.size()));
    size_t spanIdx = 0;

    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        std::vector<int32_t> ids;
        if (i < batchInputIds.size() && !batchInputIds[i].empty())
        {
            ids = batchInputIds[i]; // already tokenized by another runner
        }
        else
        {
            ids = tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);
        }

        // Per-request span window: a pad/triplet may only consume this request's own spans (batch isolation).
        size_t const spanEnd = spanIdx + static_cast<size_t>(spansPerRequest[i]);

        auto const& imgBuffers = request.requests[i].imageBuffers;
        size_t bufferIdx = 0; // per-request image-buffer cursor (for the video sub-span count)
        std::vector<int32_t> newIds;
        for (size_t j = 0; j < ids.size(); ++j)
        {
            bool const findVideoTriplet = j + 2 < ids.size() && ids[j] == mConfig.visionStartTokenId
                && ids[j + 1] == mConfig.videoTokenId && ids[j + 2] == mConfig.visionEndTokenId;
            if (findVideoTriplet && spanIdx < spanEnd)
            {
                int64_t const tps = mConfig.temporalPatchSize;
                ELLM_CHECK(bufferIdx < imgBuffers.size(),
                    "Video triplet found but no matching image buffer at index " + std::to_string(bufferIdx));
                int64_t const frames = imgBuffers[bufferIdx].frames;
                double const fps = imgBuffers[bufferIdx].fps;
                int64_t const gridT = (frames + tps - 1) / tps;
                // HF replaces the whole triplet: each timestamped frame group carries its own start/end.
                for (int64_t t = 0; t < gridT; ++t)
                {
                    ELLM_CHECK(spanIdx < spanEnd,
                        "Pad token found but no matching vision span at index " + std::to_string(spanIdx));
                    LlmVisionBlock const& block = spans[spanIdx++].llm;
                    // HF _calculate_timestamps: midpoint of the group's first/last source timestamps.
                    // Without per-frame timestamps (pre-sampled frame paths), fall back to renumbered
                    // indices / sample fps — HF's own no-metadata behavior.
                    int64_t const firstIdx = std::min(t * tps, frames - 1);
                    int64_t const lastIdx = std::min(t * tps + tps - 1, frames - 1);
                    auto const& tsList = imgBuffers[bufferIdx].timestamps;
                    double const ts = !tsList.empty() ? (tsList[firstIdx] + tsList[lastIdx]) / 2.0
                                                      : static_cast<double>(firstIdx + lastIdx) / 2.0 / fps;
                    char tsBuf[32];
                    std::snprintf(tsBuf, sizeof(tsBuf), "<%.1f seconds>", ts);
                    std::vector<int32_t> tsTokens = tokenizer->encode(std::string(tsBuf));
                    newIds.insert(newIds.end(), tsTokens.begin(), tsTokens.end());

                    newIds.push_back(mConfig.visionStartTokenId);
                    for (int64_t k = 0; k < block.numTokens; ++k)
                    {
                        newIds.push_back(mConfig.imageTokenId);
                    }
                    newIds.push_back(mConfig.visionEndTokenId);
                }
                ++bufferIdx;
                j += 2; // consumed video_pad + vision_end
            }
            else if (ids[j] == mConfig.imageTokenId || ids[j] == mConfig.videoTokenId)
            {
                ELLM_CHECK(spanIdx < spanEnd,
                    "EDGELLM_BAD_MEDIA_COUNT: Qwen3VLViTRunner::textPreprocess() pad count exceeds this request's "
                    "media count");
                LlmVisionBlock const& block = spans[spanIdx++].llm;
                for (int64_t k = 0; k < block.numTokens; ++k)
                {
                    newIds.push_back(mConfig.imageTokenId);
                }
                ++bufferIdx;
            }
            else
            {
                newIds.push_back(ids[j]);
            }
        }
        ELLM_CHECK(spanIdx == spanEnd,
            "EDGELLM_BAD_MEDIA_COUNT: Qwen3VLViTRunner::textPreprocess() pad count is smaller than this request's "
            "media count");

        if (i < batchInputIds.size())
        {
            batchInputIds[i] = std::move(newIds);
        }
        else
        {
            batchInputIds.emplace_back(std::move(newIds));
        }
    }
}

} // namespace rt
} // namespace trt_edgellm
