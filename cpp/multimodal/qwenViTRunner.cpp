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

#include "qwenViTRunner.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/mathUtils.h"
#include "common/trtUtils.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "profiling/timer.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <tuple>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{

QwenViTRunner::QwenViTRunner(
    std::string const& engineDir, int32_t llmMaxBatchSize, int32_t llmMaxSequenceLength, cudaStream_t stream)
    : MultimodalRunner(engineDir, stream)
    , mEngineDir(engineDir)
    , mLLMMaxBatchSize(llmMaxBatchSize)
    , mLLMMaxSequenceLength(llmMaxSequenceLength)
{
}

void QwenViTRunner::initialize(cudaStream_t stream)
{
    if (!validateAndFillConfig(mEngineDir))
    {
        LOG_ERROR("Failed to validate and fill config");
        throw std::runtime_error("QwenViTRunner::initialize(): Failed to validate and fill config");
    }
    if (!allocateBuffer(stream))
    {
        LOG_ERROR("Failed to allocate buffer");
        throw std::runtime_error("QwenViTRunner::initialize(): Failed to allocate buffer");
    }
}

bool QwenViTRunner::validateAndFillConfig(std::string const& engineDir)
{
    Json jsonConfig;

    std::string configPath = engineDir + "/config.json";
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.c_str());
        return false;
    }

    try
    {
        jsonConfig = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config file with error: %s", e.what());
        return false;
    }

    std::string modelTypeStr = jsonConfig["model_type"].get<std::string>();
    mModelType = multimodal::stringToModelType(modelTypeStr);
    if (mModelType != multimodal::ModelType::QWEN2_5_VL && mModelType != multimodal::ModelType::QWEN2_VL
        && mModelType != multimodal::ModelType::QWEN3_VL && mModelType != multimodal::ModelType::QWEN3_5
        && mModelType != multimodal::ModelType::QWEN3_OMNI_VISION_ENCODER)
    {
        LOG_ERROR("Invalid model type: %s", modelTypeStr.c_str());
        return false;
    }

    mConfig.visionStartTokenId = jsonConfig["vision_start_token_id"].get<int32_t>();
    mConfig.visionEndTokenId = jsonConfig.value("vision_end_token_id", 0);
    mConfig.imageTokenId = jsonConfig["image_token_id"].get<int32_t>();
    mConfig.videoTokenId = jsonConfig["video_token_id"].get<int32_t>();

    auto const& subConfig = (jsonConfig.contains("text_config") && jsonConfig["text_config"].is_object())
        ? jsonConfig["text_config"]
        : jsonConfig;
    mConfig.vocabSize = subConfig["vocab_size"].get<int32_t>();
    mConfig.mropeTheta = subConfig["rope_theta"].get<float>();

    // Read mrope_section from rope_parameters or rope_scaling
    auto const& ropeParams = subConfig.contains("rope_scaling") ? subConfig["rope_scaling"] : subConfig;
    if (ropeParams.contains("mrope_section"))
    {
        auto section = ropeParams["mrope_section"].get<std::vector<int32_t>>();
        if (section.size() >= 3)
        {
            mConfig.mropeSectionH = section[1];
            mConfig.mropeSectionW = section[2];
        }
    }
    if (mConfig.mropeSectionH <= 0 || mConfig.mropeSectionW <= 0)
    {
        LOG_ERROR("Failed to parse mrope_section in text_config. Got H=%d, W=%d", mConfig.mropeSectionH,
            mConfig.mropeSectionW);
        return false;
    }

    if (!validateExtraConfig(jsonConfig))
    {
        return false;
    }

    auto builderConfig = jsonConfig["builder_config"];
    mConfig.minImageTokensPerImage = builderConfig["min_image_tokens"].get<int64_t>();
    mConfig.maxImageTokensPerImage = builderConfig["max_image_tokens_per_image"].get<int64_t>();
    mUseTrtNativeVitAttn = builderConfig.value("use_trt_native_vit_attn", false);
    if (mConfig.minImageTokensPerImage <= 0 || mConfig.maxImageTokensPerImage <= 0)
    {
        LOG_ERROR(
            "minImageTokensPerImage and maxImageTokensPerImage must be "
            "positive, got %d and %d",
            mConfig.minImageTokensPerImage, mConfig.maxImageTokensPerImage);
        return false;
    }

    // Get preprocessor config
    Json preprocessorConfig;
    std::string preprocessorConfigPath = engineDir + "/preprocessor_config.json";
    std::ifstream preprocessorConfigFileStream(preprocessorConfigPath);
    if (!preprocessorConfigFileStream.is_open())
    {
        LOG_ERROR("Failed to open preprocessor config file: %s", preprocessorConfigPath.c_str());
        return false;
    }
    try
    {
        preprocessorConfig = Json::parse(preprocessorConfigFileStream);
        preprocessorConfigFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse preprocessor config file with error: %s", e.what());
        return false;
    }

    auto const& imageProcessorConfig
        = (preprocessorConfig.contains("image_processor") && preprocessorConfig["image_processor"].is_object())
        ? preprocessorConfig["image_processor"]
        : preprocessorConfig;
    mConfig.patchSize = imageProcessorConfig["patch_size"].get<int64_t>();
    mConfig.temporalPatchSize = imageProcessorConfig["temporal_patch_size"].get<int64_t>();
    mConfig.mergeSize = imageProcessorConfig["merge_size"].get<int64_t>();
    mConfig.imageMean = imageProcessorConfig["image_mean"].get<std::vector<float>>();
    mConfig.imageStd = imageProcessorConfig["image_std"].get<std::vector<float>>();

    // Get config from engine shapes
    nvinfer1::Dims const inputShapeMax
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims const inputShapeMin
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMIN);
    mConfig.maxHW = inputShapeMax.d[0];
    mConfig.minHW = inputShapeMin.d[0];
    auto maxImageTokens = mConfig.maxHW / (mConfig.mergeSize * mConfig.mergeSize);
    // Mirrors the visual engine's cu_seqlens profile max (maxImageTokens / minImageTokens). For video this also
    // bounds the number of per-frame cu_seqlens entries (one per frame block), not just the image count.
    mConfig.maxNumImages = maxImageTokens / mConfig.minImageTokensPerImage;
    mConfig.inputDim = mVisualContext->getTensorShape(binding_names::kVisualInput).d[1];
    mConfig.vitPosEmbDim = mVisualContext->getTensorShape(binding_names::kRotaryPosEmb).d[1];
    mConfig.outHiddenSize = mVisualEngine->getTensorShape(binding_names::kVisualOutput).d[1];

    return true;
}

bool QwenViTRunner::allocateBuffer(cudaStream_t stream)
{
    bool setTensorAddressStatus{true};
    mVitInput = rt::Tensor(
        {mConfig.maxHW, mConfig.inputDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "QwenViTRunner::mVitInput");
    setTensorAddressStatus &= mVisualContext->setTensorAddress(binding_names::kVisualInput, mVitInput.rawPointer());

    mRotaryPosEmb = rt::Tensor({mConfig.maxHW, mConfig.vitPosEmbDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
        "QwenViTRunner::mRotaryPosEmb");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kRotaryPosEmb, mRotaryPosEmb.rawPointer());

    // The size of the tensor is maxNumImages + 1 because the first element is 0.
    mCuSeqlens = rt::Tensor(
        {mConfig.maxNumImages + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "QwenViTRunner::mCuSeqlens");
    setTensorAddressStatus &= mVisualContext->setTensorAddress(binding_names::kCuSeqlens, mCuSeqlens.rawPointer());
    // Pre-allocate host tensor for cumulative sequence lengths.
    mCuSeqlensHost = rt::Tensor(
        {mConfig.maxNumImages + 1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "QwenViTRunner::mCuSeqlensHost");

    // kv_lengths is required when using TRT-native attention (TRT >= 11).
    // TRT IAttentionV2 requires query_lengths and kv_lengths to be distinct tensors,
    // so we allocate a separate buffer and copy the same cu_seqlens data at runtime.
    if (mUseTrtNativeVitAttn)
    {
        bool const hasKvLengths = isEngineInput(*mVisualEngine, binding_names::kKvLengths);
        if (!hasKvLengths)
        {
            LOG_ERROR("Config has use_trt_native_vit_attn=true but engine is missing kv_lengths binding");
            return false;
        }
        mKvLengths = rt::Tensor(
            {mConfig.maxNumImages + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "QwenViTRunner::mKvLengths");
        setTensorAddressStatus &= mVisualContext->setTensorAddress(binding_names::kKvLengths, mKvLengths.rawPointer());
    }

    mHasMaxSeqLenCarrier = isEngineInput(*mVisualEngine, binding_names::kMaxSeqLenCarrier);
    if (mHasMaxSeqLenCarrier)
    {
        mMaxSeqLenCarrier = rt::Tensor(
            {mConfig.maxHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "QwenViTRunner::mMaxSeqLenCarrier");
        setTensorAddressStatus
            &= mVisualContext->setTensorAddress(binding_names::kMaxSeqLenCarrier, mMaxSeqLenCarrier.rawPointer());
    }

    // In Qwen-VL, VIT input mHW is always numImageTokens * spatial_merge_size ** 2.
    auto const maxImageTokens = mConfig.maxHW / (mConfig.mergeSize * mConfig.mergeSize);
    mOutputEmbedding = rt::Tensor({maxImageTokens, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "QwenViTRunner::mOutputEmbedding");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kVisualOutput, mOutputEmbedding.rawPointer());

    // Allocate model-specific ViT-input buffers
    setTensorAddressStatus &= allocateExtraBuffers(maxImageTokens);

    if (!setTensorAddressStatus)
    {
        LOG_ERROR("Failed to set tensor address to the engine");
        return false;
    }

    // Copy image mean and std to device to be used in normalizeImage
    auto nbBytes = mConfig.imageMean.size() * sizeof(float);
    auto channels = math::cast<int64_t>(mConfig.imageMean.size());
    mImageMean = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "QwenViTRunner::mImageMean");
    mImageStd = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "QwenViTRunner::mImageStd");
    CUDA_CHECK(
        cudaMemcpyAsync(mImageMean.rawPointer(), mConfig.imageMean.data(), nbBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(mImageStd.rawPointer(), mConfig.imageStd.data(), nbBytes, cudaMemcpyHostToDevice, stream));

    // Pre-allocate temporary image buffers for preprocessing
    int64_t const maxImagePixels = mVitInput.getShape().volume();
    // Scratch buffer for resizeImage output. Initial shape is a placeholder; resizeImage reshapes to
    // [T, realH, realW, channels] per call. Lead 1s keep the 4D layout ImageData enforces.
    rt::Tensor resizeBuffer({1, 1, maxImagePixels, channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8,
        "QwenViTRunner::resizeBuffer");
    mResizedImageHost = rt::imageUtils::ImageData(std::move(resizeBuffer));
    mImageDevice
        = rt::Tensor({maxImagePixels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "QwenViTRunner::mImageDevice");
    mNormalizedImageDevice = rt::Tensor(
        {maxImagePixels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "QwenViTRunner::mNormalizedImageDevice");

    // Pre-allocate tensors for MRoPE position IDs
    mMropePositionIdsHost = rt::Tensor({mLLMMaxBatchSize, 3, mLLMMaxSequenceLength}, rt::DeviceType::kCPU,
        nvinfer1::DataType::kINT64, "QwenViTRunner::mMropePositionIdsHost");
    mMropePositionIdsDevice = rt::Tensor({mLLMMaxBatchSize, 3, mLLMMaxSequenceLength}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kINT64, "QwenViTRunner::mMropePositionIdsDevice");

    return true;
}

std::tuple<int64_t, int64_t> QwenViTRunner::computeVisionSpans(
    rt::imageUtils::ImageData const& image, int64_t patchBase, std::vector<VisionSpan>& spans)
{
    int64_t const height = image.height;
    int64_t const width = image.width;

    ELLM_CHECK(
        height % (mConfig.patchSize * mConfig.mergeSize) == 0 && width % (mConfig.patchSize * mConfig.mergeSize) == 0,
        "Image height or width is not divisible by patchSize * mergeSize = "
            + std::to_string(mConfig.patchSize * mConfig.mergeSize) + " got height: " + std::to_string(height)
            + ", width: " + std::to_string(width));

    // gridT = ceil(srcFrames / temporalPatchSize) (HF pads T up to a multiple of temporalPatchSize; image -> 1).
    int64_t const gridT = (image.frames + mConfig.temporalPatchSize - 1) / mConfig.temporalPatchSize;
    int64_t const gridH = height / mConfig.patchSize;   // HF grid_h (patch units)
    int64_t const gridW = width / mConfig.patchSize;    // HF grid_w (patch units)
    int64_t const llmGridH = gridH / mConfig.mergeSize; // HF llm_grid_h
    int64_t const llmGridW = gridW / mConfig.mergeSize; // HF llm_grid_w
    int64_t const tokensPerFrame = llmGridH * llmGridW;

    // Base = one flat span.
    VisionSpan span;
    span.vit = VitFrameGrid{gridT, gridH, gridW, patchBase};
    span.llm = LlmVisionBlock{gridT * tokensPerFrame, /*llmGridT*/ gridT, llmGridH, llmGridW};
    spans.push_back(span);
    return {/*totalSeqLen*/ gridT * gridH * gridW, /*totalGridT*/ gridT};
}

void QwenViTRunner::formatPatch(
    rt::imageUtils::ImageData const& image, std::vector<VisionSpan>& spans, int64_t& patchBase, cudaStream_t stream)
{
    int64_t const height = image.height;
    int64_t const width = image.width;
    int64_t const channels = image.channels;
    unsigned char* imageData = image.data(); // THWC order

    // computeVisionSpans (the one model-aware step) appends this buffer's spans starting at the running patch offset.
    int64_t const prevPatchBase = patchBase;
    auto const [totalSeqLen, totalGridT] = computeVisionSpans(image, prevPatchBase, spans);

    ELLM_CHECK(prevPatchBase + totalSeqLen <= mConfig.maxHW,
        "Visual token count " + std::to_string(prevPatchBase + totalSeqLen)
            + " exceeds the ViT engine budget maxHW = " + std::to_string(mConfig.maxHW)
            + " (for video this grows with frame count; rebuild the visual engine with a larger --maxImageTokens).");
    patchBase += totalSeqLen;

    // Physical frames consumed = grid_t groups * temporalPatchSize (HF pads T to a multiple of temporalPatchSize).
    int64_t const nSourceFrames = image.frames;
    int64_t const tPadded = totalGridT * mConfig.temporalPatchSize;

    // Reshape pre-allocated temporary buffers to current image dimensions
    check::check(mImageDevice.reshape({tPadded, height, width, channels}), "Tensor reshape failed");
    check::check(mNormalizedImageDevice.reshape({tPadded, height, width, channels}), "Tensor reshape failed");

    // Copy the source frames in one transfer, then replicate the last frame into any padded slots.
    int64_t const frameBytes = height * width * channels;
    CUDA_CHECK(cudaMemcpyAsync(
        mImageDevice.rawPointer(), imageData, nSourceFrames * frameBytes, cudaMemcpyHostToDevice, stream));
    unsigned char const* lastFrameSrc = imageData + (nSourceFrames - 1) * frameBytes;
    for (int64_t i = nSourceFrames; i < tPadded; ++i)
    {
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(mImageDevice.rawPointer()) + i * frameBytes, lastFrameSrc,
            frameBytes, cudaMemcpyHostToDevice, stream));
    }

    // Normalize image
    kernel::normalizeImage(mImageDevice, mImageMean, mImageStd, mNormalizedImageDevice, stream);

    // Transpose to patch
    kernel::transposeToPatchQwenViT(mNormalizedImageDevice, mVitInput, prevPatchBase * mConfig.inputDim,
        mConfig.temporalPatchSize, mConfig.patchSize, mConfig.mergeSize, stream);
}

void QwenViTRunner::buildCuSeqlens(
    std::vector<VisionSpan> const& spans, int32_t* cuSeqlensData, int64_t& cuSeqlensSize, int64_t& maxSeqLen) const
{
    // Per-frame cu_seqlens (HF get_vision_cu_seqlens = repeat_interleave(H*W, gridT)): each span expands into its
    // gridT per-frame gridH*gridW blocks.
    int64_t const cuSeqlensCapacity = mCuSeqlensHost.getShape().volume(); // maxNumImages + 1 entries
    cuSeqlensData[0] = 0;
    cuSeqlensSize = 1;
    maxSeqLen = 0;
    int64_t runningCuSeqlen = 0;
    for (auto const& s : spans)
    {
        auto const& g = s.vit;
        int64_t const blockSeqLen = g.gridH * g.gridW;
        for (int64_t t = 0; t < g.gridT; ++t)
        {
            ELLM_CHECK(cuSeqlensSize < cuSeqlensCapacity,
                "cuSeqlens frame count exceeds buffer capacity " + std::to_string(cuSeqlensCapacity)
                    + " (maxNumImages = " + std::to_string(mConfig.maxNumImages) + ") of VIT engine.");
            runningCuSeqlen += blockSeqLen;
            cuSeqlensData[cuSeqlensSize++] = static_cast<int32_t>(runningCuSeqlen);
        }
        maxSeqLen = std::max(maxSeqLen, blockSeqLen);
    }
}

std::tuple<int64_t, int64_t> QwenViTRunner::getResizedImageSize(
    int64_t const /*numFrames*/, int64_t const height, int64_t const width, int64_t const maxRatio)
{
    // According to https://github.com/QwenLM/Qwen2-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
    int64_t const factor = mConfig.patchSize * mConfig.mergeSize;
    int64_t const minPixels = mConfig.minImageTokensPerImage * factor * factor;
    int64_t const maxPixels = mConfig.maxImageTokensPerImage * factor * factor;

    // Banker's rounding (round-half-to-even) to match Python's round() used by the HF reference.
    auto roundByFactor = [](int64_t value, int64_t factor) -> int64_t {
        int64_t q = value / factor;
        int64_t r = value - q * factor;
        int64_t twoR = 2 * r;
        if (twoR > factor || (twoR == factor && (q & 1)))
            ++q;
        return q * factor;
    };
    auto floorByFactor = [](int64_t value, int64_t factor) -> int64_t {
        return std::floor(static_cast<double>(value) / factor) * factor;
    };
    auto ceilByFactor = [](int64_t value, int64_t factor) -> int64_t {
        return std::ceil(static_cast<double>(value) / factor) * factor;
    };

    ELLM_CHECK(std::max(height, width) / std::min(height, width) <= maxRatio,
        "absolute aspect ratio must be smaller than " + std::to_string(maxRatio) + ", got "
            + std::to_string(std::max(height, width) / std::min(height, width)));

    int64_t hBar = std::max(factor, roundByFactor(height, factor));
    int64_t wBar = std::max(factor, roundByFactor(width, factor));

    if (hBar * wBar > maxPixels)
    {
        double beta = std::sqrt(static_cast<double>(height * width) / maxPixels);
        // Clamp to >= factor: a heavily-downscaled image must not yield a sub-factor (or zero) dimension.
        hBar = std::max(factor, floorByFactor(static_cast<int64_t>(height / beta), factor));
        wBar = std::max(factor, floorByFactor(static_cast<int64_t>(width / beta), factor));
    }
    else if (hBar * wBar < minPixels)
    {
        double beta = std::sqrt(static_cast<double>(minPixels) / (height * width));
        hBar = ceilByFactor(static_cast<int64_t>(height * beta), factor);
        wBar = ceilByFactor(static_cast<int64_t>(width * beta), factor);
    }

    return {hBar, wBar};
}

void QwenViTRunner::imagePreprocess(
    rt::LLMGenerationRequest const& request, std::vector<VisionSpan>& spans, bool doResize, cudaStream_t stream)
{
    // Marshal each buffer's frames into the ViT scratch and append its spans. totalSeqLength doubles as the running
    // patch offset threaded across buffers, so after the loop it is the total ViT patch count.
    int64_t totalSeqLength = 0;
    int64_t imageCount = 0;
    for (auto const& req : request.requests)
    {
        for (auto const& image : req.imageBuffers)
        {
            size_t const spanBegin = spans.size();
            if (doResize)
            {
                auto [resizedHeight, resizedWidth] = getResizedImageSize(image.frames, image.height, image.width);
                rt::imageUtils::resizeImage(
                    image, mResizedImageHost, resizedWidth, resizedHeight, rt::imageUtils::InterpolationMode::kBICUBIC);
                formatPatch(mResizedImageHost, spans, totalSeqLength, stream);
            }
            else
            {
                formatPatch(image, spans, totalSeqLength, stream);
            }
            for (size_t spanIndex = spanBegin; spanIndex < spans.size(); ++spanIndex)
            {
                spans[spanIndex].artifactIndex = static_cast<size_t>(imageCount);
            }
            ++imageCount;
        }
    }

    mPreparedArtifactRowCounts.assign(static_cast<size_t>(imageCount), 0);
    for (VisionSpan const& span : spans)
    {
        ELLM_CHECK(span.artifactIndex < mPreparedArtifactRowCounts.size(), "Vision span has no source artifact");
        mPreparedArtifactRowCounts[span.artifactIndex] += span.llm.numTokens;
    }

    if (totalSeqLength == 0)
    {
        check::check(mVitInput.reshape({totalSeqLength, mConfig.inputDim}), "Tensor reshape failed");
        return;
    }

    ELLM_CHECK(totalSeqLength >= mConfig.minHW && totalSeqLength <= mConfig.maxHW,
        "totalSeqLength " + std::to_string(totalSeqLength) + " exceeds the limitation, max = "
            + std::to_string(mConfig.maxHW) + ", min = " + std::to_string(mConfig.minHW) + " of VIT engine.");

    // Reshape tensors
    int64_t totalImageTokens = totalSeqLength / (mConfig.mergeSize * mConfig.mergeSize);
    check::check(mVitInput.reshape({totalSeqLength, mConfig.inputDim}), "Tensor reshape failed");
    check::check(mOutputEmbedding.reshape({totalImageTokens, mConfig.outHiddenSize}), "Tensor reshape failed");
    // Record performance data
    mMultimodalMetrics.recordRun(imageCount, totalImageTokens);

    /*
     * Cache optimization for cu_seqlens, rotary position embeddings, and other image grid dependent
     * input tensors. Reuse the data from last round of computation if the image grid sizes are identical.
     * This reduces inference latency by skipping invariant tensor initialization.
     */
    bool const vitGeometryUnchanged = spans.size() == mLastSpans.size()
        && std::equal(spans.begin(), spans.end(), mLastSpans.begin(),
            [](VisionSpan const& a, VisionSpan const& b) { return a.vit == b.vit; });
    if (!vitGeometryUnchanged)
    {
        // Build cu_seqlens + per-frame maxSeqLen
        int32_t* cuSeqlensData = mCuSeqlensHost.dataPointer<int32_t>();
        int64_t cuSeqlensSize = 1;
        int64_t maxSeqLen = 0;
        buildCuSeqlens(spans, cuSeqlensData, cuSeqlensSize, maxSeqLen);
        check::check(mCuSeqlens.reshape({cuSeqlensSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mCuSeqlens.rawPointer(), mCuSeqlensHost.rawPointer(),
            cuSeqlensSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        if (mUseTrtNativeVitAttn)
        {
            check::check(mKvLengths.reshape({cuSeqlensSize}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemcpyAsync(mKvLengths.rawPointer(), mCuSeqlensHost.rawPointer(),
                cuSeqlensSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        }

        if (mHasMaxSeqLenCarrier)
        {
            check::check(mMaxSeqLenCarrier.reshape({maxSeqLen}), "Tensor reshape failed");
        }

        // Build rotary position embeddings
        check::check(mRotaryPosEmb.reshape({totalSeqLength, mConfig.vitPosEmbDim}), "Tensor reshape failed");
        for (auto const& s : spans)
        {
            kernel::initRotaryPosEmbQwenViT(mRotaryPosEmb, {s.vit.gridT, s.vit.gridH, s.vit.gridW}, mConfig.mergeSize,
                s.vit.patchStart, 10000.0f, 1.0f, stream);
        }

        // Build model-specific ViT inputs.
        buildExtraInputs(spans, totalSeqLength, totalImageTokens, stream);

        mLastSpans = spans;
    }
}

void QwenViTRunner::getMRopePositionIds(
    std::vector<std::vector<int32_t>> const& batchInputIds, std::vector<VisionSpan> const& spans) noexcept
{
    // Mirrors HF Qwen2VLModel.get_rope_index (base = Qwen2-VL; also serves Qwen3-VL, whose sub-spans are llmGridT==1).
    // Temporal step is 1 (HF t_index = arange(grid_t)); span advance is spatial max(llmGridH, llmGridW).

    // mropePositionIds: (bs, 3, maxPositionEmbeddings), 3 is for T, H, W
    int64_t* mropePositionIdsPtr = mMropePositionIdsHost.dataPointer<int64_t>();
    int64_t const maxPositionEmbeddings = mMropePositionIdsHost.getShape()[2];
    int64_t totalImageIdx = 0;
    int64_t batchOffset = 0;

    mMropeRopeDeltasPerBatch.clear();
    for (auto const& inputIds : batchInputIds)
    {
        auto start = inputIds.begin();
        auto end = inputIds.end();
        auto it = inputIds.begin();
        int64_t startIdx = 0;
        int64_t remainingStartPos = 0;

        while ((it = std::find(start, end, mConfig.visionStartTokenId)) != end)
        {
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
            ++totalImageIdx;

            for (int64_t t = 0; t < llmGridT; ++t)
            {
                for (int64_t h = 0; h < llmGridH; ++h)
                {
                    for (int64_t w = 0; w < llmGridW; ++w)
                    {
                        int64_t idx = remainingStartPos + textLen + t * llmGridH * llmGridW + h * llmGridW + w;
                        mropePositionIdsPtr[batchOffset + 0 * maxPositionEmbeddings + idx] = t + textLen + startIdx;
                        mropePositionIdsPtr[batchOffset + 1 * maxPositionEmbeddings + idx] = h + textLen + startIdx;
                        mropePositionIdsPtr[batchOffset + 2 * maxPositionEmbeddings + idx] = w + textLen + startIdx;
                    }
                }
            }

            start = it + 1 + llmGridT * llmGridH * llmGridW;
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

        batchOffset += 3 * maxPositionEmbeddings;
    }
}

void QwenViTRunner::generateMropeParams(std::vector<std::vector<int32_t>> const& batchInputIds,
    std::vector<VisionSpan> const& spans, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    int64_t const activeBatchSize = batchInputIds.size();
    auto ropeRotaryCosSinDim = ropeRotaryCosSinDevice.getShape();
    int64_t const maxPositionEmbeddings = ropeRotaryCosSinDim[1];
    int64_t const rotaryDim = ropeRotaryCosSinDim[2];

    bool checkShapeValid = activeBatchSize <= mLLMMaxBatchSize && maxPositionEmbeddings <= mLLMMaxSequenceLength;
    if (!checkShapeValid)
    {
        LOG_ERROR(
            "mropePositionIdsHost shape is not valid. Allowed shape: [%d, 3, %d]. "
            "Got activeBatchSize: %d, maxPositionEmbeddings: %ld",
            mLLMMaxBatchSize, mLLMMaxSequenceLength, activeBatchSize, maxPositionEmbeddings);
        throw std::runtime_error("mropePositionIdsHost shape validation failed");
    }

    // Initialize mropePositionIds and copy to device
    check::check(mMropePositionIdsHost.reshape({activeBatchSize, 3, maxPositionEmbeddings}), "Tensor reshape failed");
    check::check(mMropePositionIdsDevice.reshape({activeBatchSize, 3, maxPositionEmbeddings}), "Tensor reshape failed");
    getMRopePositionIds(batchInputIds, spans);
    CUDA_CHECK(cudaMemcpyAsync(mMropePositionIdsDevice.rawPointer(), mMropePositionIdsHost.rawPointer(),
        activeBatchSize * 3 * maxPositionEmbeddings * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    // Initialize mrope cosSinCacheDevice
    check::check(
        ropeRotaryCosSinDevice.reshape({activeBatchSize, maxPositionEmbeddings, rotaryDim}), "Tensor reshape failed");
    bool interleaved = mConfig.mropeInterleaved;
    kernel::initializeMRopeCosSin(ropeRotaryCosSinDevice.dataPointer<float>(),
        mMropePositionIdsDevice.dataPointer<int64_t>(), mConfig.mropeTheta, rotaryDim, maxPositionEmbeddings,
        activeBatchSize, interleaved, mConfig.mropeSectionH, mConfig.mropeSectionW, stream);
}

void QwenViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<VisionSpan> const& spans,
    trt_edgellm::tokenizer::Tokenizer const* tokenizer)
{
    // Flat visual-pad expansion: each vision pad (<|image_pad|>/<|video_pad|>) -> its span's numTokens incrementing
    // IDs (>= vocabSize), consumed by embeddingLookupWithImageInsertion.
    int32_t nextImageTokenId = mConfig.vocabSize;
    size_t spanIdx = 0;

    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        // Reuse tokens if another runner already tokenized this request, else tokenize it now.
        std::vector<int32_t> ids;
        if (i < batchInputIds.size() && !batchInputIds[i].empty())
        {
            ids = batchInputIds[i];
        }
        else
        {
            ids = tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);
        }

        std::vector<int32_t> newIds;
        for (size_t j = 0; j < ids.size(); ++j)
        {
            if (ids[j] == mConfig.imageTokenId || ids[j] == mConfig.videoTokenId)
            {
                ELLM_CHECK(spanIdx < spans.size(),
                    "Pad token found but no matching vision span at index " + std::to_string(spanIdx));
                LlmVisionBlock const& block = spans[spanIdx++].llm;
                for (int64_t k = 0; k < block.numTokens; ++k)
                {
                    newIds.push_back(nextImageTokenId++);
                }
            }
            else
            {
                newIds.push_back(ids[j]);
            }
        }

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

bool QwenViTRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
    rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream, bool imageOnly)
{
    std::vector<VisionSpan> spans; // per-request vision layout; lives only across this call

    try
    {
        imagePreprocess(request, spans, !imageOnly, stream);
        if (!imageOnly)
        {
            if (!mropeCosSinOut.has_value())
            {
                LOG_ERROR("mropeCosSinOut is required when imageOnly=false.");
                return false;
            }
            textPreprocess(request, batchedInputIds, spans, tokenizer);
            generateMropeParams(batchedInputIds, spans, mropeCosSinOut.value().get(), stream);
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed: %s", e.what());
        return false;
    }

    return true;
}

bool QwenViTRunner::prepareArtifactSubset(rt::LLMGenerationRequest const& request,
    [[maybe_unused]] std::vector<size_t> const& originalItemIndices, cudaStream_t stream)
{
    std::vector<VisionSpan> spans;
    try
    {
        imagePreprocess(request, spans, /*doResize=*/true, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to prepare selected vision artifacts: %s", e.what());
        return false;
    }
    return true;
}

std::vector<int64_t> QwenViTRunner::preparedArtifactRowCounts() const
{
    return mPreparedArtifactRowCounts;
}

bool QwenViTRunner::preprocessSystemPrompt(std::string const& systemPrompt, tokenizer::Tokenizer const* tokenizer,
    rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream)
{
    if (systemPrompt.empty())
    {
        return true;
    }

    if (!mropeCosSinOut.has_value())
    {
        LOG_ERROR("mropeCosSinOut is required for non-empty system prompts.");
        return false;
    }

    // systemPrompt is already formatted by tokenizer's applyChatTemplate
    std::vector<int32_t> ids = tokenizer->encode(systemPrompt);
    if (ids.empty())
    {
        LOG_ERROR("Failed to encode system prompt.");
        return false;
    }
    std::vector<std::vector<int32_t>> batchedInputIds;
    batchedInputIds.emplace_back(std::move(ids));

    try
    {
        generateMropeParams(batchedInputIds, {}, mropeCosSinOut.value().get(), stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("MRope parameter generation failed: %s", e.what());
        return false;
    }

    return true;
}

bool QwenViTRunner::infer(cudaStream_t stream) noexcept
{
    // Skip VIT inference if there are no images to process
    // Check if the first dimension (sequence length) is 0, indicating no images
    if (mVitInput.getShape()[0] == 0)
    {
        return true;
    }

    // Profile ViT inference with automatic cleanup
    {
        TIME_STAGE(metrics::StageNames::kVISION_ENCODER, stream);

        bool setEngineIOStatus{true};
        setEngineIOStatus
            &= mVisualContext->setInputShape(binding_names::kVisualInput, mVitInput.getShape().getTRTDims());
        setEngineIOStatus
            &= mVisualContext->setInputShape(binding_names::kRotaryPosEmb, mRotaryPosEmb.getShape().getTRTDims());
        setEngineIOStatus
            &= mVisualContext->setInputShape(binding_names::kCuSeqlens, mCuSeqlens.getShape().getTRTDims());
        if (mUseTrtNativeVitAttn)
        {
            setEngineIOStatus
                &= mVisualContext->setInputShape(binding_names::kKvLengths, mKvLengths.getShape().getTRTDims());
        }
        if (mHasMaxSeqLenCarrier)
        {
            setEngineIOStatus &= mVisualContext->setInputShape(
                binding_names::kMaxSeqLenCarrier, mMaxSeqLenCarrier.getShape().getTRTDims());
        }
        // Model-specific ViT-input shapes (window vs fast-pos-emb); base adds none.
        setEngineIOStatus &= bindExtraInputShapes();

        if (!setEngineIOStatus)
        {
            LOG_ERROR("Failed to bind engine input tensors.");
            return false;
        }

        bool enqueueStatus = mVisualContext->enqueueV3(stream);
        if (!enqueueStatus)
        {
            LOG_ERROR("Failed to enqueue engine.");
            return false;
        }
    }

    return true;
}

// ---- Per-model strategy hooks: base implementations = Qwen2-VL. Subclasses override what they add. ----

bool QwenViTRunner::validateExtraConfig(nlohmann::json const& /*jsonConfig*/)
{
    return true;
}

bool QwenViTRunner::allocateExtraBuffers(int64_t /*maxImageTokens*/)
{
    return true;
}

void QwenViTRunner::buildExtraInputs(std::vector<VisionSpan> const& /*spans*/, int64_t /*totalSeqLength*/,
    int64_t /*totalImageTokens*/, cudaStream_t /*stream*/)
{
    return;
}

bool QwenViTRunner::bindExtraInputShapes()
{
    return true;
}

rt::OptionalInputTensors QwenViTRunner::getDeepstackFeatures()
{
    return {};
}

} // namespace rt
} // namespace trt_edgellm
