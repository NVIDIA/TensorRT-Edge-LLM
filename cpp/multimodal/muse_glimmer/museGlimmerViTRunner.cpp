/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "multimodal/muse_glimmer/museGlimmerViTRunner.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/trtUtils.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

bool MuseGlimmerViTRunner::validateExtraConfig(nlohmann::json const& jsonConfig)
{
    if (!jsonConfig.contains("vision_config") || !jsonConfig["vision_config"].is_object())
    {
        LOG_ERROR("Muse-Glimmer requires a vision_config object in config.json");
        return false;
    }
    auto const& visionConfig = jsonConfig["vision_config"];

    // The learned position table is square.
    mNumGridPerSide = visionConfig.value("pos_emb_height", int64_t{32});
    int64_t const visionPatchSize = visionConfig.value("patch_size", int64_t{14});
    if (mNumGridPerSide <= 0 || visionPatchSize <= 0)
    {
        LOG_ERROR("Muse-Glimmer pos_emb_height (%ld) and patch_size (%ld) must be positive", mNumGridPerSide,
            visionPatchSize);
        return false;
    }
    mWindowSizePx = mNumGridPerSide * visionPatchSize;

    // Vision and text use independent rotary theta values.
    if (visionConfig.contains("rope_parameters") && visionConfig["rope_parameters"].is_object())
    {
        mVitRopeTheta = visionConfig["rope_parameters"].value("rope_theta", 10000.0f);
    }
    return true;
}

bool MuseGlimmerViTRunner::validateAndFillConfig(std::string const& engineDir)
{
    if (!QwenViTRunner::validateAndFillConfig(engineDir))
    {
        return false;
    }
    // Linear patch embedding consumes temporal-first packed patches.
    int64_t const expectedInputDim = 3 * mConfig.temporalPatchSize * mConfig.patchSize * mConfig.patchSize;
    if (mConfig.inputDim != expectedInputDim)
    {
        LOG_ERROR(
            "Visual engine input dim %ld does not match the Muse-Glimmer Linear patch-embed width %ld "
            "(temporalPatchSize * 3 * patchSize^2; temporalPatchSize = %ld, patchSize = %ld)",
            mConfig.inputDim, expectedInputDim, mConfig.temporalPatchSize, mConfig.patchSize);
        return false;
    }

    std::ifstream processorStream(engineDir + "/processor_config.json");
    if (processorStream.is_open())
    {
        try
        {
            auto const processor = nlohmann::json::parse(processorStream);
            if (processor.contains("image_processor"))
            {
                mMaxImageTokens = processor["image_processor"].value("max_image_tokens", mMaxImageTokens);
            }
            if (processor.contains("video_processor"))
            {
                mMaxVideoFrameTokens
                    = processor["video_processor"].value("max_video_frame_tokens", mMaxVideoFrameTokens);
            }
        }
        catch (nlohmann::json::parse_error const& e)
        {
            LOG_ERROR("Failed to parse Muse-Glimmer processor_config.json: %s", e.what());
            return false;
        }
    }
    return true;
}

int64_t MuseGlimmerViTRunner::vitInputMergeSize() const
{
    // The encoder runs at spatial_merge_size == 1 -> raster-order ViT input patches.
    return 1;
}

bool MuseGlimmerViTRunner::vitPatchTemporalFirst() const
{
    return true;
}

std::tuple<int64_t, int64_t> MuseGlimmerViTRunner::getResizedImageSize(
    int64_t const /*numFrames*/, bool const isVideo, int64_t const height, int64_t const width, int64_t const maxRatio)
{
    ELLM_CHECK(height > 0 && width > 0, "Muse-Glimmer media dimensions must be positive");
    ELLM_CHECK(static_cast<double>(std::max(height, width)) / std::min(height, width) <= maxRatio,
        "Muse-Glimmer media aspect ratio exceeds the supported maximum");

    int64_t const factor = mConfig.patchSize * mConfig.mergeSize;
    int64_t const checkpointCap = isVideo ? mMaxVideoFrameTokens : mMaxImageTokens;
    int64_t const maxTokens = std::min(checkpointCap, mConfig.maxImageTokensPerImage);
    ELLM_CHECK(factor > 0 && maxTokens > 0, "Invalid Muse-Glimmer resize configuration");

    double idealH = static_cast<double>(height) / factor;
    double idealW = static_cast<double>(width) / factor;
    double const ratio = idealW / idealH;
    if (idealH * idealW > maxTokens)
    {
        idealH = std::sqrt(maxTokens / ratio);
        idealW = idealH * ratio;
    }

    std::array<int64_t, 2> const candidateH{
        static_cast<int64_t>(std::floor(idealH)), static_cast<int64_t>(std::ceil(idealH))};
    std::array<int64_t, 2> const candidateW{
        static_cast<int64_t>(std::floor(idealW)), static_cast<int64_t>(std::ceil(idealW))};
    double const targetAspect = static_cast<double>(height) / width;
    double bestError = std::numeric_limits<double>::infinity();
    int64_t bestH = 1;
    int64_t bestW = 1;
    for (int64_t const gridH : candidateH)
    {
        for (int64_t const gridW : candidateW)
        {
            if (gridH < 1 || gridW < 1 || gridH * gridW > maxTokens)
            {
                continue;
            }
            double const error = std::abs(static_cast<double>(gridH) / gridW - targetAspect);
            if (error < bestError)
            {
                bestError = error;
                bestH = gridH;
                bestW = gridW;
            }
        }
    }
    return {bestH * factor, bestW * factor};
}

void MuseGlimmerViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<VisionSpan> const& spans,
    std::vector<int64_t> const& spansPerRequest, trt_edgellm::tokenizer::Tokenizer const* tokenizer)
{
    ELLM_CHECK(spansPerRequest.size() == request.requests.size(), "spansPerRequest and request batch size differ");

    int32_t const imageStartId = tokenizer->getTokenId("<|image_start|>");
    int32_t const imageEndId = tokenizer->getTokenId("<|image_end|>");
    int32_t const videoStartId = tokenizer->getTokenId("<|vid_start|>");
    int32_t const videoEndId = tokenizer->getTokenId("<|vid_end|>");
    int32_t const videoSeparatorId = tokenizer->getTokenId("<|vid_frame_separator|>");
    ELLM_CHECK(imageStartId >= 0 && imageEndId >= 0 && videoStartId >= 0 && videoEndId >= 0 && videoSeparatorId >= 0,
        "Muse-Glimmer tokenizer is missing multimodal special tokens");

    size_t spanIdx = 0;
    for (size_t requestIdx = 0; requestIdx < request.requests.size(); ++requestIdx)
    {
        std::vector<int32_t> ids;
        if (requestIdx < batchInputIds.size() && !batchInputIds[requestIdx].empty())
        {
            ids = std::move(batchInputIds[requestIdx]);
        }
        else
        {
            ids = tokenizer->encode(request.formattedRequests[requestIdx].formattedCompleteRequest);
        }
        auto const& imageBuffers = request.requests[requestIdx].imageBuffers;
        size_t const spanEnd = spanIdx + static_cast<size_t>(spansPerRequest[requestIdx]);
        size_t bufferIdx = 0;
        std::vector<int32_t> expandedIds;
        size_t expandedCapacity = ids.size();
        for (size_t i = spanIdx; i < spanEnd; ++i)
        {
            auto const& block = spans[i].llm;
            expandedCapacity += static_cast<size_t>(block.numTokens + block.llmGridT * 16 + 2);
        }
        expandedIds.reserve(expandedCapacity);

        for (int32_t const id : ids)
        {
            if (id != mConfig.imageTokenId && id != mConfig.videoTokenId)
            {
                expandedIds.push_back(id);
                continue;
            }

            ELLM_CHECK(spanIdx < spanEnd && bufferIdx < imageBuffers.size(),
                "EDGELLM_BAD_MEDIA_COUNT: Muse-Glimmer placeholder count exceeds this request's media count");
            auto const& block = spans[spanIdx++].llm;
            auto const& image = imageBuffers[bufferIdx++];
            if (!image.isVideo)
            {
                expandedIds.push_back(imageStartId);
                expandedIds.insert(expandedIds.end(), block.numTokens, mConfig.imageTokenId);
                expandedIds.push_back(imageEndId);
                continue;
            }

            int64_t const groups = block.llmGridT;
            ELLM_CHECK(groups > 0 && block.numTokens % groups == 0, "Invalid Muse-Glimmer video token geometry");
            int64_t const tokensPerGroup = block.numTokens / groups;
            expandedIds.push_back(videoStartId);
            for (int64_t group = 0; group < groups; ++group)
            {
                int64_t const frame = std::min(group * mConfig.temporalPatchSize, image.frames - 1);
                double const timestamp
                    = !image.timestamps.empty() ? image.timestamps[frame] : static_cast<double>(frame) / image.fps;
                char label[32];
                std::snprintf(label, sizeof(label), "Time: %.1fs", timestamp);
                auto const labelIds = tokenizer->encode(label);
                expandedIds.insert(expandedIds.end(), labelIds.begin(), labelIds.end());
                expandedIds.insert(expandedIds.end(), tokensPerGroup, mConfig.imageTokenId);
                expandedIds.push_back(group + 1 < groups ? videoSeparatorId : videoEndId);
            }
        }
        ELLM_CHECK(spanIdx == spanEnd && bufferIdx == imageBuffers.size(),
            "EDGELLM_BAD_MEDIA_COUNT: Muse-Glimmer placeholder count is smaller than this request's media count");

        if (requestIdx < batchInputIds.size())
        {
            batchInputIds[requestIdx] = std::move(expandedIds);
        }
        else
        {
            batchInputIds.emplace_back(std::move(expandedIds));
        }
    }
}

void MuseGlimmerViTRunner::buildRotaryPosEmb(std::vector<VisionSpan> const& spans, cudaStream_t stream)
{
    // Muse uses raster-order 2D rotary positions offset by one.
    for (auto const& s : spans)
    {
        kernel::initRotaryPosEmbMuseGlimmerViT(
            mRotaryPosEmb, {s.vit.gridT, s.vit.gridH, s.vit.gridW}, s.vit.patchStart, mVitRopeTheta, stream);
    }
}

bool MuseGlimmerViTRunner::allocateExtraBuffers(int64_t /*maxImageTokens*/)
{
    bool setTensorAddressStatus{true};
    // Window metadata spans unmerged visual tokens.
    int64_t const maxTokens = mConfig.maxHW;

    mCuWindowSeqlens = rt::Tensor(
        {maxTokens + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "MuseGlimmerViTRunner::mCuWindowSeqlens");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kCuWindowSeqlens, mCuWindowSeqlens.rawPointer());
    mCuWindowSeqlensHost = rt::Tensor({maxTokens + 1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32,
        "MuseGlimmerViTRunner::mCuWindowSeqlensHost");

    if (mUseTrtNativeVitAttn)
    {
        mHasKvLengthsWindow = isEngineInput(*mVisualEngine, binding_names::kKvLengthsWindow);
        if (mHasKvLengthsWindow)
        {
            mKvLengthsWindow = rt::Tensor({maxTokens + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32,
                "MuseGlimmerViTRunner::mKvLengthsWindow");
            setTensorAddressStatus
                &= mVisualContext->setTensorAddress(binding_names::kKvLengthsWindow, mKvLengthsWindow.rawPointer());
        }
    }

    mWindowIndexHost = rt::Tensor(
        {maxTokens}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64, "MuseGlimmerViTRunner::mWindowIndexHost");
    mWindowInverseHost = rt::Tensor(
        {maxTokens}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64, "MuseGlimmerViTRunner::mWindowInverseHost");
    mWindowIndexDevice = rt::Tensor(
        {maxTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "MuseGlimmerViTRunner::mWindowIndexDevice");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kWindowIndex, mWindowIndexDevice.rawPointer());

    mReverseWindowIndexHost = rt::Tensor(
        {maxTokens}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64, "MuseGlimmerViTRunner::mReverseWindowIndexHost");
    mReverseWindowIndexDevice = rt::Tensor({maxTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64,
        "MuseGlimmerViTRunner::mReverseWindowIndexDevice");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kReverseWindowIndex, mReverseWindowIndexDevice.rawPointer());

    // Fast-pos-embed inputs (4-tap bilinear gather over the learned position table).
    mFastPosEmbIdx = rt::Tensor(
        {4, maxTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "MuseGlimmerViTRunner::mFastPosEmbIdx");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kFastPosEmbIdx, mFastPosEmbIdx.rawPointer());
    mFastPosEmbWeight = rt::Tensor(
        {4, maxTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "MuseGlimmerViTRunner::mFastPosEmbWeight");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kFastPosEmbWeight, mFastPosEmbWeight.rawPointer());

    return setTensorAddressStatus;
}

void MuseGlimmerViTRunner::buildExtraInputs(
    std::vector<VisionSpan> const& spans, int64_t totalSeqLength, int64_t /*totalImageTokens*/, cudaStream_t stream)
{
    check::check(mWindowIndexHost.reshape({totalSeqLength}), "Tensor reshape failed");
    check::check(mWindowIndexDevice.reshape({totalSeqLength}), "Tensor reshape failed");
    check::check(mReverseWindowIndexHost.reshape({totalSeqLength}), "Tensor reshape failed");
    check::check(mReverseWindowIndexDevice.reshape({totalSeqLength}), "Tensor reshape failed");
    check::check(mFastPosEmbIdx.reshape({4, totalSeqLength}), "Tensor reshape failed");
    check::check(mFastPosEmbWeight.reshape({4, totalSeqLength}), "Tensor reshape failed");

    getWindowIndex(spans, totalSeqLength, stream);

    for (auto const& s : spans)
    {
        kernel::initFastPosEmbedMuseGlimmerViT(mFastPosEmbIdx, mFastPosEmbWeight,
            {s.vit.gridT, s.vit.gridH, s.vit.gridW}, mNumGridPerSide, s.vit.patchStart, stream);
    }

    if (mUseTrtNativeVitAttn && mHasKvLengthsWindow)
    {
        int64_t const cuWindowSeqlensSize = mCuWindowSeqlens.getShape()[0];
        check::check(mKvLengthsWindow.reshape({cuWindowSeqlensSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mKvLengthsWindow.rawPointer(), mCuWindowSeqlensHost.rawPointer(),
            cuWindowSeqlensSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    }
}

bool MuseGlimmerViTRunner::bindExtraInputShapes()
{
    bool setEngineIOStatus{true};
    setEngineIOStatus
        &= mVisualContext->setInputShape(binding_names::kCuWindowSeqlens, mCuWindowSeqlens.getShape().getTRTDims());
    if (mUseTrtNativeVitAttn && mHasKvLengthsWindow)
    {
        setEngineIOStatus
            &= mVisualContext->setInputShape(binding_names::kKvLengthsWindow, mKvLengthsWindow.getShape().getTRTDims());
    }
    setEngineIOStatus
        &= mVisualContext->setInputShape(binding_names::kWindowIndex, mWindowIndexDevice.getShape().getTRTDims());
    setEngineIOStatus &= mVisualContext->setInputShape(
        binding_names::kReverseWindowIndex, mReverseWindowIndexDevice.getShape().getTRTDims());
    setEngineIOStatus
        &= mVisualContext->setInputShape(binding_names::kFastPosEmbIdx, mFastPosEmbIdx.getShape().getTRTDims());
    setEngineIOStatus
        &= mVisualContext->setInputShape(binding_names::kFastPosEmbWeight, mFastPosEmbWeight.getShape().getTRTDims());
    return setEngineIOStatus;
}

void MuseGlimmerViTRunner::getWindowIndex(std::vector<VisionSpan> const& spans, int64_t curHW, cudaStream_t stream)
{
    int64_t* windowIndexPtr = mWindowIndexHost.dataPointer<int64_t>();
    int64_t const windowIndexSize = mWindowIndexHost.getShape()[0];
    int64_t const vitMergerWindowSize = mWindowSizePx / mConfig.patchSize;
    int64_t windowIndexPos = 0;
    int64_t windowIndexValue = 0;

    int32_t* cuWindowSeqlensData = mCuWindowSeqlensHost.dataPointer<int32_t>();
    cuWindowSeqlensData[0] = 0;
    int64_t cuWindowSeqlensSize = 1;

    // Build per-token window indices and cumulative lengths.
    for (auto const& span : spans)
    {
        int64_t const T = span.vit.gridT;
        int64_t const H = span.vit.gridH; // full patch grid (not merged)
        int64_t const W = span.vit.gridW;
        int64_t const numWindowsH = (H + vitMergerWindowSize - 1) / vitMergerWindowSize;
        int64_t const numWindowsW = (W + vitMergerWindowSize - 1) / vitMergerWindowSize;

        for (int64_t t = 0; t < T; ++t)
        {
            int64_t const frameBase = windowIndexValue + t * H * W;
            for (int64_t i = 0; i < numWindowsH; ++i)
            {
                for (int64_t j = 0; j < numWindowsW; ++j)
                {
                    int64_t cnt{0};
                    for (int64_t m = 0; m < vitMergerWindowSize; ++m)
                    {
                        for (int64_t n = 0; n < vitMergerWindowSize; ++n)
                        {
                            int64_t const idxH = i * vitMergerWindowSize + m;
                            int64_t const idxW = j * vitMergerWindowSize + n;
                            if (idxH < H && idxW < W)
                            {
                                windowIndexPtr[windowIndexPos++] = idxH * W + idxW + frameBase;
                                ++cnt;
                            }
                        }
                    }
                    ELLM_CHECK(cuWindowSeqlensSize < mCuWindowSeqlensHost.getShape()[0],
                        "cuWindowSeqlens overflow in Muse-Glimmer window attention");
                    int32_t const prevCuWindowSeqlen = cuWindowSeqlensData[cuWindowSeqlensSize - 1];
                    cuWindowSeqlensData[cuWindowSeqlensSize++] = static_cast<int32_t>(prevCuWindowSeqlen + cnt);
                }
            }
        }
        windowIndexValue += T * H * W;
    }

    ELLM_CHECK(windowIndexPos == curHW,
        "Muse-Glimmer windowIndex size does not match totalSeqLength. Got windowIndex size: "
            + std::to_string(windowIndexPos) + ", curHW: " + std::to_string(curHW));

    check::check(mCuWindowSeqlens.reshape({cuWindowSeqlensSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mCuWindowSeqlens.rawPointer(), mCuWindowSeqlensHost.rawPointer(),
        cuWindowSeqlensSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    int64_t* windowInversePtr = mWindowInverseHost.dataPointer<int64_t>();
    for (int64_t wp = 0; wp < windowIndexSize; ++wp)
    {
        windowInversePtr[windowIndexPtr[wp]] = wp;
    }

    int64_t* reverseWindowIndexPtr = mReverseWindowIndexHost.dataPointer<int64_t>();
    int64_t const f = mConfig.mergeSize; // pixel-shuffle 2x2 grouping
    int64_t outPos = 0;
    int64_t spanRasterBase = 0;
    for (auto const& span : spans)
    {
        int64_t const T = span.vit.gridT;
        int64_t const H = span.vit.gridH;
        int64_t const W = span.vit.gridW;
        for (int64_t t = 0; t < T; ++t)
        {
            int64_t const frameRasterBase = spanRasterBase + t * H * W;
            // Pixel-shuffle emits each 2x2 block contiguously in raster order.
            for (int64_t hb = 0; hb < H / f; ++hb)
            {
                for (int64_t wb = 0; wb < W / f; ++wb)
                {
                    for (int64_t fh = 0; fh < f; ++fh)
                    {
                        for (int64_t fw = 0; fw < f; ++fw)
                        {
                            int64_t const raster = (hb * f + fh) * W + (wb * f + fw);
                            reverseWindowIndexPtr[outPos++] = windowInversePtr[frameRasterBase + raster];
                        }
                    }
                }
            }
        }
        spanRasterBase += T * H * W;
    }
    ELLM_CHECK(outPos == curHW,
        "Muse-Glimmer reverse_window_index size does not match totalSeqLength. Got: " + std::to_string(outPos)
            + ", curHW: " + std::to_string(curHW));

    CUDA_CHECK(cudaMemcpyAsync(mWindowIndexDevice.rawPointer(), mWindowIndexHost.rawPointer(),
        windowIndexSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mReverseWindowIndexDevice.rawPointer(), mReverseWindowIndexHost.rawPointer(),
        windowIndexSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
}

} // namespace rt
} // namespace trt_edgellm
