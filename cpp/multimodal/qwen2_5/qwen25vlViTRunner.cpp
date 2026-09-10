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

#include "multimodal/qwen2_5/qwen25vlViTRunner.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include <algorithm>
#include <numeric>

namespace trt_edgellm
{
namespace rt
{

bool Qwen25VLViTRunner::validateExtraConfig(nlohmann::json const& jsonConfig)
{
    mWindowSize = jsonConfig["vision_config"]["window_size"].get<int64_t>();
    // Qwen2.5-VL video MRoPE temporal interval = tokens_per_second * int(temporal_patch_size / fps).
    if (!jsonConfig["vision_config"].contains("tokens_per_second"))
    {
        LOG_ERROR("Qwen2.5-VL requires vision_config.tokens_per_second for video temporal MRoPE");
        return false;
    }
    mTokensPerSecond = jsonConfig["vision_config"]["tokens_per_second"].get<int64_t>();
    return true;
}

bool Qwen25VLViTRunner::allocateExtraBuffers(int64_t maxImageTokens)
{
    bool setTensorAddressStatus{true};
    // Use maxImageTokens as a safe upper bound for cumulative window sequence lengths.
    mCuWindowSeqlens = rt::Tensor(
        {maxImageTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "Qwen25VLViTRunner::mCuWindowSeqlens");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kCuWindowSeqlens, mCuWindowSeqlens.rawPointer());
    mCuWindowSeqlensHost = rt::Tensor(
        {maxImageTokens}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "Qwen25VLViTRunner::mCuWindowSeqlensHost");

    if (mUseTrtNativeVitAttn)
    {
        mHasKvLengthsWindow = isEngineInput(*mVisualEngine, binding_names::kKvLengthsWindow);
        if (mHasKvLengthsWindow)
        {
            mKvLengthsWindow = rt::Tensor({maxImageTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32,
                "Qwen25VLViTRunner::mKvLengthsWindow");
            setTensorAddressStatus
                &= mVisualContext->setTensorAddress(binding_names::kKvLengthsWindow, mKvLengthsWindow.rawPointer());
        }
    }

    mWindowIndexHost = rt::Tensor(
        {maxImageTokens}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64, "Qwen25VLViTRunner::mWindowIndexHost");
    mWindowIndexDevice = rt::Tensor(
        {maxImageTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "Qwen25VLViTRunner::mWindowIndexDevice");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kWindowIndex, mWindowIndexDevice.rawPointer());

    mReverseWindowIndexHost = rt::Tensor({maxImageTokens}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64,
        "Qwen25VLViTRunner::mReverseWindowIndexHost");
    mReverseWindowIndexDevice = rt::Tensor({maxImageTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64,
        "Qwen25VLViTRunner::mReverseWindowIndexDevice");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kReverseWindowIndex, mReverseWindowIndexDevice.rawPointer());

    return setTensorAddressStatus;
}

void Qwen25VLViTRunner::buildExtraInputs(
    std::vector<VisionSpan> const& spans, int64_t totalSeqLength, int64_t totalImageTokens, cudaStream_t stream)
{
    check::check(mWindowIndexHost.reshape({totalImageTokens}), "Tensor reshape failed");
    check::check(mWindowIndexDevice.reshape({totalImageTokens}), "Tensor reshape failed");
    check::check(mReverseWindowIndexHost.reshape({totalImageTokens}), "Tensor reshape failed");
    check::check(mReverseWindowIndexDevice.reshape({totalImageTokens}), "Tensor reshape failed");
    getWindowIndex(spans, totalSeqLength, stream);

    if (mUseTrtNativeVitAttn && mHasKvLengthsWindow)
    {
        int64_t const cuWindowSeqlensSize = mCuWindowSeqlens.getShape()[0];
        check::check(mKvLengthsWindow.reshape({cuWindowSeqlensSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mKvLengthsWindow.rawPointer(), mCuWindowSeqlensHost.rawPointer(),
            cuWindowSeqlensSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    }
}

bool Qwen25VLViTRunner::bindExtraInputShapes()
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
    return setEngineIOStatus;
}

void Qwen25VLViTRunner::getWindowIndex(std::vector<VisionSpan> const& spans, int64_t curHW, cudaStream_t stream)
{
    int64_t* windowIndexPtr = mWindowIndexHost.dataPointer<int64_t>();
    int64_t const windowIndexSize = mWindowIndexHost.getShape()[0];
    int64_t const vitMergerWindowSize = mWindowSize / mConfig.mergeSize / mConfig.patchSize;
    int64_t windowIndexPos = 0;
    int64_t windowIndexValue = 0;

    int32_t* cuWindowSeqlensData = mCuWindowSeqlensHost.dataPointer<int32_t>();
    cuWindowSeqlensData[0] = 0;
    int64_t cuWindowSeqlensSize = 1;

    for (auto const& span : spans)
    {
        int64_t T = span.vit.gridT;
        int64_t llmGridH = span.vit.gridH / mConfig.mergeSize;
        int64_t llmGridW = span.vit.gridW / mConfig.mergeSize;
        int64_t numWindowsH = (llmGridH + vitMergerWindowSize - 1) / vitMergerWindowSize;
        int64_t numWindowsW = (llmGridW + vitMergerWindowSize - 1) / vitMergerWindowSize;

        // Window attention is spatial and applied per temporal frame: the same partition repeats for each of the
        // gridT frames, offset by the frame's token base. (Qwen3-VL video splits into gridT==1 spans; here gridT>1.)
        for (int64_t t = 0; t < T; ++t)
        {
            int64_t const frameBase = windowIndexValue + t * llmGridH * llmGridW;
            for (int64_t i = 0; i < numWindowsH; ++i)
            {
                for (int64_t j = 0; j < numWindowsW; ++j)
                {
                    int64_t cnt{0};
                    for (int64_t m = 0; m < vitMergerWindowSize; ++m)
                    {
                        for (int64_t n = 0; n < vitMergerWindowSize; ++n)
                        {
                            int64_t idxH = i * vitMergerWindowSize + m;
                            int64_t idxW = j * vitMergerWindowSize + n;
                            if (idxH < llmGridH && idxW < llmGridW)
                            {
                                windowIndexPtr[windowIndexPos++] = idxH * llmGridW + idxW + frameBase;
                                ++cnt;
                            }
                        }
                    }

                    ELLM_CHECK(cuWindowSeqlensSize < mCuWindowSeqlensHost.getShape()[0],
                        "cuWindowSeqlens overflow in window attention");
                    int32_t prevCuWindowSeqlen = cuWindowSeqlensData[cuWindowSeqlensSize - 1];
                    cuWindowSeqlensData[cuWindowSeqlensSize++]
                        = static_cast<int32_t>(prevCuWindowSeqlen + cnt * mConfig.mergeSize * mConfig.mergeSize);
                }
            }
        }

        windowIndexValue += T * llmGridH * llmGridW;
    }

    ELLM_CHECK(windowIndexPos * (mConfig.mergeSize * mConfig.mergeSize) == curHW,
        "windowIndex size * (mergeSize * mergeSize) does not match curHW. Got windowIndex size: "
            + std::to_string(windowIndexPos) + ", curHW: " + std::to_string(curHW));

    check::check(mCuWindowSeqlens.reshape({cuWindowSeqlensSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mCuWindowSeqlens.rawPointer(), mCuWindowSeqlensHost.rawPointer(),
        cuWindowSeqlensSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    int64_t* reverseWindowIndexPtr = mReverseWindowIndexHost.dataPointer<int64_t>();
    std::iota(reverseWindowIndexPtr, reverseWindowIndexPtr + windowIndexSize, 0);
    std::sort(reverseWindowIndexPtr, reverseWindowIndexPtr + windowIndexSize,
        [windowIndexPtr](size_t left, size_t right) { return windowIndexPtr[left] < windowIndexPtr[right]; });

    CUDA_CHECK(cudaMemcpyAsync(mWindowIndexDevice.rawPointer(), mWindowIndexHost.rawPointer(),
        windowIndexSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mReverseWindowIndexDevice.rawPointer(), mReverseWindowIndexHost.rawPointer(),
        windowIndexSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
}

std::tuple<int64_t, int64_t> Qwen25VLViTRunner::computeVisionSpans(
    rt::imageUtils::ImageData const& image, int64_t patchBase, std::vector<VisionSpan>& spans)
{
    // Reuse the base flat layout, then fill the fps-derived secondPerGrid.
    auto const extent = QwenViTRunner::computeVisionSpans(image, patchBase, spans);
    spans.back().llm.secondPerGrid = mConfig.temporalPatchSize / image.fps;
    return extent;
}

void Qwen25VLViTRunner::getMRopePositionIds(std::vector<std::vector<int32_t>> const& batchInputIds,
    std::vector<VisionSpan> const& spans, std::vector<int64_t> const& spansPerRequest) noexcept
{
    // Mirrors HF Qwen2_5_VLModel.get_rope_index: per-frame temporal step = secondPerGrid * tokens_per_second
    // (fps-aware); span advance is spatial max(llmGridH, llmGridW).

    // mropePositionIds: (bs, 3, maxPositionEmbeddings), 3 is for T, H, W
    int64_t* mropePositionIdsPtr = mMropePositionIdsHost.dataPointer<int64_t>();
    int64_t const maxPositionEmbeddings = mMropePositionIdsHost.getShape()[2];
    int64_t totalImageIdx = 0;
    int64_t batchOffset = 0;

    mMropeRopeDeltasPerBatch.clear();
    for (size_t r = 0; r < batchInputIds.size(); ++r)
    {
        auto const& inputIds = batchInputIds[r];
        // Per-request span window: a visual block may only consume this request's own spans.
        int64_t const spanEnd = totalImageIdx + (r < spansPerRequest.size() ? spansPerRequest[r] : 0);
        auto start = inputIds.begin();
        auto end = inputIds.end();
        auto it = inputIds.begin();
        int64_t startIdx = 0;
        int64_t remainingStartPos = 0;

        auto searchFrom = start;
        while ((it = std::find(searchFrom, end, mConfig.visionStartTokenId)) != end)
        {
            // A visual block is <|vision_start|> immediately followed by pads (textPreprocess fills the
            // constant imageTokenId); a stray start token, or one beyond this request's span window,
            // must not consume a span.
            if (it + 1 == end || *(it + 1) != mConfig.imageTokenId || totalImageIdx >= spanEnd)
            {
                searchFrom = it + 1;
                continue;
            }
            // Text part
            int64_t textLen = it + 1 - start;
            for (int64_t i = 0; i < 3; ++i)
            {
                for (int64_t j = 0; j < textLen; ++j)
                {
                    mropePositionIdsPtr[batchOffset + i * maxPositionEmbeddings + remainingStartPos + j] = j + startIdx;
                }
            }

            // Visual part — the Nth <|vision_start|> consumes the Nth span (global order). MRoPE reads the LLM view.
            LlmVisionBlock const& block = spans[totalImageIdx].llm;
            int64_t const llmGridT = block.llmGridT;
            int64_t const llmGridH = block.llmGridH;
            int64_t const llmGridW = block.llmGridW;
            // HF get_rope_index truncates: tokens_per_second * int(second_per_grid_ts).
            int64_t const timeInterval = static_cast<int64_t>(block.secondPerGrid) * mTokensPerSecond;
            ++totalImageIdx;

            for (int64_t t = 0; t < llmGridT; ++t)
            {
                for (int64_t h = 0; h < llmGridH; ++h)
                {
                    for (int64_t w = 0; w < llmGridW; ++w)
                    {
                        int64_t idx = remainingStartPos + textLen + t * llmGridH * llmGridW + h * llmGridW + w;
                        mropePositionIdsPtr[batchOffset + 0 * maxPositionEmbeddings + idx]
                            = t * timeInterval + textLen + startIdx;
                        mropePositionIdsPtr[batchOffset + 1 * maxPositionEmbeddings + idx] = h + textLen + startIdx;
                        mropePositionIdsPtr[batchOffset + 2 * maxPositionEmbeddings + idx] = w + textLen + startIdx;
                    }
                }
            }

            start = it + 1 + llmGridT * llmGridH * llmGridW;
            searchFrom = start;
            startIdx += std::max(llmGridH, llmGridW) + textLen; // spatial-only advance (T not counted)
            remainingStartPos = start - inputIds.begin();
        }

        // MRoPE rope delta for this batch: maxMropePositionId + 1 - inputIdSize
        int64_t const maxMropePositionId = startIdx + inputIds.size() - remainingStartPos - 1;
        mMropeRopeDeltasPerBatch.push_back(maxMropePositionId + 1 - inputIds.size());

        // Remaining text part till maxPositionEmbeddings. Treat all generated tokens as text tokens.
        int64_t textLen = maxPositionEmbeddings - remainingStartPos;
        for (int64_t i = 0; i < 3; ++i)
        {
            for (int64_t j = 0; j < textLen; ++j)
            {
                mropePositionIdsPtr[batchOffset + i * maxPositionEmbeddings + remainingStartPos + j] = j + startIdx;
            }
        }

        totalImageIdx = spanEnd; // seal the window: unconsumed spans never leak forward
        batchOffset += 3 * maxPositionEmbeddings;
    }
}

} // namespace rt
} // namespace trt_edgellm
