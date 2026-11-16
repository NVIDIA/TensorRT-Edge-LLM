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
#include "common/logger.h"
#include "common/trtInterop.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "multimodal/common/imageUtils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>

namespace trt_edgellm
{
namespace rt
{

namespace qwen3vl
{
namespace
{

constexpr std::string_view kIMAGE_TRIPLET{"<|vision_start|><|image_pad|><|vision_end|>"};
constexpr std::string_view kVIDEO_TRIPLET{"<|vision_start|><|video_pad|><|vision_end|>"};

} // namespace

std::vector<int32_t> tokenizeExpandedPrompt(std::string const& formattedPrompt,
    std::vector<rt::imageUtils::ImageData> const& media, std::vector<VisionSpan> const& spans, size_t spanBegin,
    size_t spanEnd, QwenViTConfig const& config, trt_edgellm::tokenizer::Tokenizer const& tokenizer)
{
    ELLM_CHECK(spanBegin <= spanEnd && spanEnd <= spans.size(), "Invalid vision span window");
    ELLM_CHECK(config.temporalPatchSize > 0, "Temporal patch size must be positive");

    std::string expanded;
    expanded.reserve(formattedPrompt.size());
    size_t promptOffset = 0;
    size_t markerCount = 0;
    size_t bufferIdx = 0;

    while (true)
    {
        size_t const imageOffset = formattedPrompt.find(kIMAGE_TRIPLET.data(), promptOffset, kIMAGE_TRIPLET.size());
        size_t const videoOffset = formattedPrompt.find(kVIDEO_TRIPLET.data(), promptOffset, kVIDEO_TRIPLET.size());
        size_t const markerOffset = std::min(imageOffset, videoOffset);
        if (markerOffset == std::string::npos)
        {
            break;
        }
        bool const isVideo = videoOffset < imageOffset;
        ELLM_CHECK(
            bufferIdx < media.size(), "EDGELLM_BAD_MEDIA_COUNT: prompt has more media placeholders than media buffers");
        auto const& buffer = media[bufferIdx++];
        expanded.append(formattedPrompt, promptOffset, markerOffset - promptOffset);

        if (isVideo)
        {
            int64_t const tps = config.temporalPatchSize;
            int64_t const gridT = (buffer.frames + tps - 1) / tps;
            for (int64_t t = 0; t < gridT; ++t)
            {
                ELLM_CHECK(spanBegin + markerCount < spanEnd,
                    "EDGELLM_BAD_MEDIA_COUNT: video has more frame groups than vision spans");
                // HF _calculate_timestamps: midpoint of the group's first/last source timestamps, or
                // renumbered indices / sample fps without per-frame timestamps.
                int64_t const firstIdx = std::min(t * tps, buffer.frames - 1);
                int64_t const lastIdx = std::min(t * tps + tps - 1, buffer.frames - 1);
                double const timestamp = !buffer.timestamps.empty()
                    ? (buffer.timestamps[firstIdx] + buffer.timestamps[lastIdx]) / 2.0
                    : static_cast<double>(firstIdx + lastIdx) / 2.0 / buffer.fps;
                char timestampText[32];
                std::snprintf(timestampText, sizeof(timestampText), "<%.1f seconds>", timestamp);
                expanded.append(timestampText);
                expanded.append(kVIDEO_TRIPLET.data(), kVIDEO_TRIPLET.size());
                ++markerCount;
            }
            promptOffset = markerOffset + kVIDEO_TRIPLET.size();
        }
        else
        {
            ELLM_CHECK(spanBegin + markerCount < spanEnd, "EDGELLM_BAD_MEDIA_COUNT: image has no matching vision span");
            expanded.append(kIMAGE_TRIPLET.data(), kIMAGE_TRIPLET.size());
            ++markerCount;
            promptOffset = markerOffset + kIMAGE_TRIPLET.size();
        }
    }

    ELLM_CHECK(spanBegin + markerCount == spanEnd,
        "EDGELLM_BAD_MEDIA_COUNT: prompt has fewer media placeholders than vision spans");
    expanded.append(formattedPrompt, promptOffset, std::string::npos);

    auto const ids = tokenizer.encode(expanded);
    std::vector<int32_t> expandedIds;
    expandedIds.reserve(ids.size());
    size_t spanIdx = spanBegin;
    for (auto const tokenId : ids)
    {
        if (tokenId == config.imageTokenId || tokenId == config.videoTokenId)
        {
            ELLM_CHECK(
                spanIdx < spanEnd, "EDGELLM_BAD_MEDIA_COUNT: prompt has more visual pad tokens than vision spans");
            int64_t const numTokens = spans[spanIdx++].llm.numTokens;
            ELLM_CHECK(numTokens >= 0, "Vision span token count cannot be negative");
            expandedIds.insert(expandedIds.end(), static_cast<size_t>(numTokens), config.imageTokenId);
        }
        else
        {
            expandedIds.push_back(tokenId);
        }
    }
    ELLM_CHECK(spanIdx == spanEnd, "EDGELLM_BAD_MEDIA_COUNT: prompt has fewer visual pad tokens than vision spans");
    return expandedIds;
}

} // namespace qwen3vl

bool Qwen3VLViTRunner::validateExtraConfig(nlohmann::json const& jsonConfig)
{
    auto visionConfig = jsonConfig["vision_config"];
    // The learned position grid holds (image_size / patch_size)^2 entries. Checkpoints that keep
    // the default omit the count, so it is recovered from that geometry instead.
    int64_t numPositionEmbeddings = visionConfig.value("num_position_embeddings", int64_t{0});
    if (numPositionEmbeddings <= 0)
    {
        int64_t const imageSize = visionConfig.value("image_size", int64_t{0});
        int64_t const patchSize = visionConfig.value("patch_size", int64_t{0});
        ELLM_CHECK(imageSize > 0 && patchSize > 0,
            "vision_config states neither 'num_position_embeddings' nor an 'image_size'/'patch_size' pair to "
            "recover it from");
        int64_t const gridPerSide = imageSize / patchSize;
        numPositionEmbeddings = gridPerSide * gridPerSide;
    }
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
        {4, mConfig.maxHW}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT64, "Qwen3VLViTRunner::mFastPosEmbIdx");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kFastPosEmbIdx, mFastPosEmbIdx.rawPointer());

    mFastPosEmbWeight = rt::Tensor(
        {4, mConfig.maxHW}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "Qwen3VLViTRunner::mFastPosEmbWeight");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kFastPosEmbWeight, mFastPosEmbWeight.rawPointer());

    mDeepstackFeatures.reserve(static_cast<size_t>(mNumDeepstackFeatures));
    for (int64_t i = 0; i < mNumDeepstackFeatures; ++i)
    {
        std::string const deepstackFeatureName = binding_names::formatDeepstackFeaturesName(i);
        mDeepstackFeatures.emplace_back(rt::Tensor({maxImageTokens, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
            trt_edgellm::DataType::kHALF, deepstackFeatureName));
        setTensorAddressStatus
            &= mVisualContext->setTensorAddress(deepstackFeatureName.c_str(), mDeepstackFeatures.back().rawPointer());
    }
    mOutputTensors.reserve(1 + mDeepstackFeatures.size());
    mOutputTensors.emplace_back(mOutputEmbedding);
    for (auto& feature : mDeepstackFeatures)
    {
        mOutputTensors.emplace_back(feature);
    }
    return setTensorAddressStatus;
}

void Qwen3VLViTRunner::buildExtraInputs(
    std::vector<VisionSpan> const& spans, int64_t totalSeqLength, int64_t /*totalImageTokens*/, cudaStream_t stream)
{
    check::check(mFastPosEmbIdx.reshape({4, totalSeqLength}), "Tensor reshape failed");
    check::check(mFastPosEmbWeight.reshape({4, totalSeqLength}), "Tensor reshape failed");

    // Each span's gridT*gridH*gridW fast-pos-emb entries are written starting at vit.patchStart.
    for (auto const& s : spans)
    {
        kernel::initFastPosEmbedQwenViT(mFastPosEmbIdx, mFastPosEmbWeight, {s.vit.gridT, s.vit.gridH, s.vit.gridW},
            mConfig.mergeSize, mNumGridPerSide, s.vit.patchStart, stream);
    }
}

bool Qwen3VLViTRunner::bindExtraInputShapes()
{
    // Cache restoration can change output shapes without changing the last encoded ViT geometry.
    // Restore these on every encode, even when the geometry-dependent input buffers are reused.
    for (auto& feature : mDeepstackFeatures)
    {
        check::check(feature.reshape(mOutputEmbedding.getShape()), "Tensor reshape failed");
    }
    bool setEngineIOStatus{true};
    setEngineIOStatus
        &= mVisualContext->setInputShape(binding_names::kFastPosEmbIdx, toTrtDims(mFastPosEmbIdx.getShape()));
    setEngineIOStatus
        &= mVisualContext->setInputShape(binding_names::kFastPosEmbWeight, toTrtDims(mFastPosEmbWeight.getShape()));
    return setEngineIOStatus;
}

bool Qwen3VLViTRunner::infer(cudaStream_t stream) noexcept
{
    return infer(stream, 0);
}

bool Qwen3VLViTRunner::infer(cudaStream_t stream, int64_t outputTokenOffset) noexcept
{
    int64_t const localTokens = mOutputEmbedding.getShape()[0];
    for (size_t index = 0; index < mOutputTensors.size(); ++index)
    {
        auto& output = mOutputTensors[index].get();
        int64_t const rowBytes
            = output.getShape()[1] * static_cast<int64_t>(rt::utils::getTypeSize(output.getDataType()));
        if (rowBytes <= 0 || outputTokenOffset < 0 || localTokens < 0
            || localTokens > output.getMemoryCapacity() / rowBytes
            || outputTokenOffset > output.getMemoryCapacity() / rowBytes - localTokens)
        {
            LOG_ERROR("Vision shard output exceeds the runner's capacity.");
            return false;
        }
        auto* address = static_cast<uint8_t*>(output.rawPointer()) + outputTokenOffset * rowBytes;
        constexpr uintptr_t kTensorAlignment{256};
        if (reinterpret_cast<uintptr_t>(address) % kTensorAlignment != 0)
        {
            LOG_ERROR("Vision shard output does not satisfy TensorRT address alignment.");
            return false;
        }
        char const* const name = index == 0 ? binding_names::kVisualOutput : output.getName().c_str();
        if (!mVisualContext->setTensorAddress(name, address))
        {
            LOG_ERROR("Failed to bind vision shard output %s.", name);
            return false;
        }
    }
    return QwenViTRunner::infer(stream);
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

std::vector<std::reference_wrapper<rt::Tensor>> Qwen3VLViTRunner::getDeepstackFeatures()
{
    if (mNumDeepstackFeatures == 0) // Qwen3.5: no deepstack
    {
        return {};
    }
    std::vector<std::reference_wrapper<rt::Tensor>> refs;
    refs.reserve(mDeepstackFeatures.size());
    for (auto& tensor : mDeepstackFeatures)
    {
        refs.emplace_back(std::ref(tensor));
    }
    return refs;
}

std::vector<std::reference_wrapper<rt::Tensor>> const& Qwen3VLViTRunner::getOutputTensors() noexcept
{
    return mOutputTensors;
}

void Qwen3VLViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<VisionSpan> const& spans,
    std::vector<int64_t> const& spansPerRequest, trt_edgellm::tokenizer::Tokenizer const* tokenizer)
{
    ELLM_CHECK(spansPerRequest.size() == request.requests.size(),
        "spansPerRequest.size() != request.requests.size(), " + std::to_string(spansPerRequest.size())
            + " != " + std::to_string(request.requests.size()));
    size_t spanIdx = 0;

    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        // Timestamps must be tokenized together with the prompt text, so ids from another runner cannot be reused.
        ELLM_CHECK(i >= batchInputIds.size() || batchInputIds[i].empty(),
            "Qwen3VLViTRunner::textPreprocess() cannot expand a prompt another runner already tokenized");
        // Per-request span window: a pad/triplet may only consume this request's own spans (batch isolation).
        size_t const spanEnd = spanIdx + static_cast<size_t>(spansPerRequest[i]);
        auto ids = qwen3vl::tokenizeExpandedPrompt(request.formattedRequests[i].formattedCompleteRequest,
            request.requests[i].imageBuffers, spans, spanIdx, spanEnd, mConfig, *tokenizer);
        spanIdx = spanEnd;
        if (i < batchInputIds.size())
        {
            batchInputIds[i] = std::move(ids);
        }
        else
        {
            batchInputIds.emplace_back(std::move(ids));
        }
    }
}

} // namespace rt
} // namespace trt_edgellm
