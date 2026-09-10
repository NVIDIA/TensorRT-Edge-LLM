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

#include "multimodal/gemma4/gemma4UnifiedVisionRunner.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "multimodal/common/imageUtils.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{

Gemma4UnifiedVisionRunner::Gemma4UnifiedVisionRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner(engineDir, stream)
{
    ELLM_CHECK(validateAndFillConfig(engineDir), "Gemma4UnifiedVisionRunner: invalid configuration or engine");
    ELLM_CHECK(allocateBuffer(stream), "Gemma4UnifiedVisionRunner: failed to allocate buffers");
}

bool Gemma4UnifiedVisionRunner::validateAndFillConfig(std::string const& engineDir)
{
    Json config;
    std::string const configPath = engineDir + "/config.json";
    std::ifstream configStream(configPath);
    if (!configStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.c_str());
        return false;
    }
    try
    {
        config = Json::parse(configStream);
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse %s: %s", configPath.c_str(), e.what());
        return false;
    }

    std::string const modelType = config.value("model_type", "");
    mModelType = multimodal::stringToModelType(modelType);
    if (mModelType != multimodal::ModelType::GEMMA4_UNIFIED_VISION)
    {
        LOG_ERROR("Invalid model type for Gemma4 Unified vision: %s", modelType.c_str());
        return false;
    }
    if (!config.contains("image_token_id") || !config.contains("boi_token_id") || !config.contains("eoi_token_id")
        || !config.contains("vision_config"))
    {
        LOG_ERROR(
            "Gemma4 Unified visual config requires image_token_id, boi_token_id, eoi_token_id, and vision_config");
        return false;
    }
    mConfig.imageTokenId = config["image_token_id"].get<int32_t>();
    mConfig.beginImageTokenId = config["boi_token_id"].get<int32_t>();
    mConfig.endImageTokenId = config["eoi_token_id"].get<int32_t>();
    auto const& visionConfig = config["vision_config"];
    mConfig.modelPatchSize = visionConfig.value("model_patch_size", int64_t{48});
    if (!visionConfig.contains("mm_posemb_size"))
    {
        LOG_ERROR("Gemma4 Unified visual config requires vision_config.mm_posemb_size");
        return false;
    }
    mConfig.positionEmbeddingSize = visionConfig["mm_posemb_size"].get<int64_t>();
    if (mConfig.positionEmbeddingSize < 1)
    {
        LOG_ERROR("Gemma4 Unified mm_posemb_size must be positive, got %ld", mConfig.positionEmbeddingSize);
        return false;
    }

    if (!mVisualEngine || !mVisualContext)
    {
        LOG_ERROR("Failed to load Gemma4 Unified visual engine");
        return false;
    }
    if (mVisualEngine->getTensorIOMode(binding_names::kVisualInput) != nvinfer1::TensorIOMode::kINPUT
        || mVisualEngine->getTensorIOMode(binding_names::kPixelPositionIds) != nvinfer1::TensorIOMode::kINPUT
        || mVisualEngine->getTensorIOMode(binding_names::kVisualOutput) != nvinfer1::TensorIOMode::kOUTPUT)
    {
        LOG_ERROR("Gemma4 Unified visual engine is missing required input/output bindings");
        return false;
    }
    if (mVisualEngine->getTensorDataType(binding_names::kVisualInput) != nvinfer1::DataType::kHALF
        || mVisualEngine->getTensorDataType(binding_names::kPixelPositionIds) != nvinfer1::DataType::kINT64
        || mVisualEngine->getTensorDataType(binding_names::kVisualOutput) != nvinfer1::DataType::kHALF)
    {
        LOG_ERROR("Gemma4 Unified visual engine requires FP16 input/output and INT64 pixel_position_ids");
        return false;
    }

    nvinfer1::Dims const inputMin
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMIN);
    nvinfer1::Dims const inputMax
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims const positionsMin
        = mVisualEngine->getProfileShape(binding_names::kPixelPositionIds, 0, nvinfer1::OptProfileSelector::kMIN);
    nvinfer1::Dims const positionsMax
        = mVisualEngine->getProfileShape(binding_names::kPixelPositionIds, 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims const outputShape = mVisualEngine->getTensorShape(binding_names::kVisualOutput);
    if (inputMin.nbDims != 2 || inputMax.nbDims != 2 || positionsMin.nbDims != 2 || positionsMax.nbDims != 2
        || outputShape.nbDims != 2)
    {
        LOG_ERROR("Gemma4 Unified visual bindings have unexpected ranks");
        return false;
    }
    mConfig.minPatches = inputMin.d[0];
    mConfig.maxPatches = inputMax.d[0];
    mConfig.inputDim = inputMax.d[1];
    mConfig.outputHiddenSize = outputShape.d[1];
    if (mConfig.minPatches < 1 || mConfig.maxPatches < mConfig.minPatches)
    {
        LOG_ERROR("Gemma4 Unified visual patch profile must satisfy 1 <= min <= max, got [%ld,%ld]", mConfig.minPatches,
            mConfig.maxPatches);
        return false;
    }
    if (mConfig.modelPatchSize != 48)
    {
        LOG_ERROR("Gemma4 Unified vision requires model_patch_size=48, got %ld", mConfig.modelPatchSize);
        return false;
    }
    if (mConfig.inputDim != mConfig.modelPatchSize * mConfig.modelPatchSize * 3)
    {
        LOG_ERROR("Gemma4 Unified visual input dim must be %ld, got %ld",
            mConfig.modelPatchSize * mConfig.modelPatchSize * 3, mConfig.inputDim);
        return false;
    }
    if (mConfig.outputHiddenSize != 3840)
    {
        LOG_ERROR("Gemma4 Unified visual output hidden size must be 3840, got %ld", mConfig.outputHiddenSize);
        return false;
    }
    if (positionsMin.d[0] != mConfig.minPatches || positionsMax.d[0] != mConfig.maxPatches || positionsMin.d[1] != 2
        || positionsMax.d[1] != 2)
    {
        LOG_ERROR(
            "Gemma4 Unified pixel_position_ids profile must be [N,2] matching the input patch profile, got "
            "min [%ld,%ld], max [%ld,%ld]",
            positionsMin.d[0], positionsMin.d[1], positionsMax.d[0], positionsMax.d[1]);
        return false;
    }

    auto const& builderConfig = config.contains("builder_config") ? config["builder_config"] : Json::object();
    int64_t configuredPerImage
        = builderConfig.value("max_image_tokens_per_image", visionConfig.value("num_soft_tokens", mConfig.maxPatches));
    if (configuredPerImage <= 0)
    {
        configuredPerImage = mConfig.maxPatches;
    }
    mConfig.maxPatchesPerImage = std::min(configuredPerImage, mConfig.maxPatches);

    std::string const preprocessorPath = engineDir + "/preprocessor_config.json";
    std::ifstream preprocessorStream(preprocessorPath);
    if (!preprocessorStream.is_open())
    {
        LOG_ERROR("Failed to open Gemma4 Unified image preprocessor config: %s", preprocessorPath.c_str());
        return false;
    }
    try
    {
        Json const preprocessor = Json::parse(preprocessorStream);
        Json const& imageProcessor
            = preprocessor.contains("image_processor") ? preprocessor["image_processor"] : preprocessor;
        bool const doRescale = imageProcessor.value("do_rescale", true);
        bool const doNormalize = imageProcessor.value("do_normalize", false);
        double const rescaleFactor = imageProcessor.value("rescale_factor", 1.0 / 255.0);
        if (!doRescale || doNormalize || std::abs(rescaleFactor - 1.0 / 255.0) > 1.0e-8)
        {
            LOG_ERROR("Gemma4 Unified runtime requires RGB rescale_factor=1/255 without image normalization");
            return false;
        }
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse %s: %s", preprocessorPath.c_str(), e.what());
        return false;
    }

    LOG_INFO("Gemma4 Unified vision: profile patches=[%ld,%ld], max_per_image=%ld, input_dim=%ld, hidden=%ld",
        mConfig.minPatches, mConfig.maxPatches, mConfig.maxPatchesPerImage, mConfig.inputDim, mConfig.outputHiddenSize);
    return true;
}

bool Gemma4UnifiedVisionRunner::allocateBuffer([[maybe_unused]] cudaStream_t stream)
{
    mVisualInput = rt::Tensor({mConfig.maxPatches, mConfig.inputDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "Gemma4UnifiedVisionRunner::mVisualInput");
    mPixelPositionIds = rt::Tensor({mConfig.maxPatches, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64,
        "Gemma4UnifiedVisionRunner::mPixelPositionIds");
    mPixelPositionIdsHost = rt::Tensor({mConfig.maxPatches, 2}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64,
        "Gemma4UnifiedVisionRunner::mPixelPositionIdsHost");
    mOutputEmbedding = rt::Tensor({mConfig.maxPatches, mConfig.outputHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "Gemma4UnifiedVisionRunner::mOutputEmbedding");

    bool status = true;
    status &= mVisualContext->setTensorAddress(binding_names::kVisualInput, mVisualInput.rawPointer());
    status &= mVisualContext->setTensorAddress(binding_names::kPixelPositionIds, mPixelPositionIds.rawPointer());
    status &= mVisualContext->setTensorAddress(binding_names::kVisualOutput, mOutputEmbedding.rawPointer());
    if (!status)
    {
        LOG_ERROR("Failed to set Gemma4 Unified visual tensor addresses");
        return false;
    }

    std::vector<float> const mean(3, 0.0F);
    std::vector<float> const stddev(3, 1.0F);
    mImageMean
        = rt::Tensor({3}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "Gemma4UnifiedVisionRunner::mImageMean");
    mImageStd
        = rt::Tensor({3}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "Gemma4UnifiedVisionRunner::mImageStd");
    CUDA_CHECK(cudaMemcpy(mImageMean.rawPointer(), mean.data(), mean.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(mImageStd.rawPointer(), stddev.data(), stddev.size() * sizeof(float), cudaMemcpyHostToDevice));

    int64_t const maxSpatialPixels = mConfig.maxPatchesPerImage * mConfig.modelPatchSize * mConfig.modelPatchSize;
    mImageDevice = rt::Tensor({maxSpatialPixels * 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "Gemma4UnifiedVisionRunner::mImageDevice");
    mRescaledImageDevice = rt::Tensor({maxSpatialPixels * 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "Gemma4UnifiedVisionRunner::mRescaledImageDevice");

    // GPU image-resize scratch.
    int64_t const kMaxRawPixels = kernel::kGpuResizeMaxRawDim * kernel::kGpuResizeMaxRawDim;
    // Horizontal-pass scratch holds [rawH, outW, 3] floats. gemma4UnifiedResizeTarget preserves aspect
    // ratio, so rawH * outW <= sqrt(maxSpatialPixels * rawH * rawW) <= sqrt(maxSpatialPixels *
    // kMaxRawPixels).
    int64_t const kMaxResizeTmpElems
        = static_cast<int64_t>(
              std::sqrt(static_cast<double>(maxSpatialPixels) * kMaxRawPixels) * kernel::kGpuResizeScratchMargin)
        * 3;
    kernel::allocateResizeScratch(3, kMaxResizeTmpElems, mRawImageDevice, mResizeTmpDevice);
    return true;
}

void Gemma4UnifiedVisionRunner::formatImage(rt::imageUtils::ImageData const& image, int64_t& patchOffset,
    std::vector<int64_t>& imageTokenLengths, cudaStream_t stream)
{
    ELLM_CHECK(image.frames == 1 && image.channels == 3, "Gemma4 Unified accepts one RGB image per image buffer");
    ELLM_CHECK(image.height % mConfig.modelPatchSize == 0 && image.width % mConfig.modelPatchSize == 0,
        "Gemma4 Unified image dimensions must be divisible by 48");
    int64_t const gridHeight = image.height / mConfig.modelPatchSize;
    int64_t const gridWidth = image.width / mConfig.modelPatchSize;
    int64_t const patchCount = gridHeight * gridWidth;
    ELLM_CHECK(patchCount > 0 && patchCount <= mConfig.maxPatchesPerImage,
        "Gemma4 Unified image exceeds the per-image patch profile");
    ELLM_CHECK(
        patchOffset + patchCount <= mConfig.maxPatches, "Gemma4 Unified request exceeds the visual engine profile");
    ELLM_CHECK(gridHeight <= mConfig.positionEmbeddingSize && gridWidth <= mConfig.positionEmbeddingSize,
        "Gemma4 Unified patch position exceeds mm_posemb_size");

    // mImageDevice already holds the [1, image.height, image.width, 3] image filled by the caller.
    check::check(mRescaledImageDevice.reshape({1, image.height, image.width, 3}), "Tensor reshape failed");
    kernel::normalizeImage(mImageDevice, mImageMean, mImageStd, mRescaledImageDevice, stream);
    kernel::transposeToPatchGemma4ViT(
        mRescaledImageDevice, mVisualInput, patchOffset * mConfig.inputDim, mConfig.modelPatchSize, stream);

    int64_t* positions = mPixelPositionIdsHost.dataPointer<int64_t>();
    for (int64_t y = 0; y < gridHeight; ++y)
    {
        for (int64_t x = 0; x < gridWidth; ++x)
        {
            int64_t const index = patchOffset + y * gridWidth + x;
            positions[index * 2] = x;
            positions[index * 2 + 1] = y;
        }
    }
    patchOffset += patchCount;
    imageTokenLengths.push_back(patchCount);
}

void Gemma4UnifiedVisionRunner::imagePreprocessTokenLengthsOnly(rt::LLMGenerationRequest const& request,
    std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& imagesPerRequest)
{
    int64_t totalPatches = 0;
    for (auto const& req : request.requests)
    {
        imagesPerRequest.push_back(static_cast<int64_t>(req.imageBuffers.size()));
        for (auto const& image : req.imageBuffers)
        {
            auto const [h, w] = image.doResize
                ? rt::imageUtils::gemma4UnifiedResizeTarget(image.height, image.width, mConfig.maxPatchesPerImage,
                      mConfig.modelPatchSize, mConfig.positionEmbeddingSize)
                : std::make_tuple(image.height, image.width);
            int64_t const patchCount = (h / mConfig.modelPatchSize) * (w / mConfig.modelPatchSize);
            totalPatches += patchCount;
            imageTokenLengths.push_back(patchCount);
        }
    }
    if (totalPatches > 0)
    {
        check::check(mOutputEmbedding.reshape({totalPatches, mConfig.outputHiddenSize}), "Tensor reshape failed");
    }
}

void Gemma4UnifiedVisionRunner::imagePreprocess(rt::LLMGenerationRequest const& request,
    std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& imagesPerRequest, cudaStream_t stream)
{
    check::check(mVisualInput.reshape({mConfig.maxPatches, mConfig.inputDim}), "Tensor reshape failed");
    check::check(mPixelPositionIdsHost.reshape({mConfig.maxPatches, 2}), "Tensor reshape failed");
    int64_t totalPatches = 0;
    for (auto const& req : request.requests)
    {
        imagesPerRequest.push_back(static_cast<int64_t>(req.imageBuffers.size()));
        for (auto const& image : req.imageBuffers)
        {
            if (image.doResize)
            {
                auto const [resizedHeight, resizedWidth] = rt::imageUtils::gemma4UnifiedResizeTarget(image.height,
                    image.width, mConfig.maxPatchesPerImage, mConfig.modelPatchSize, mConfig.positionEmbeddingSize);
                kernel::copyImageToDeviceAndResize(image.data(), image.frames, image.height, image.width,
                    image.channels, mRawImageDevice, mResizeTmpDevice, mImageDevice, resizedHeight, resizedWidth,
                    stream);
                formatImage(image.resizedMeta(resizedHeight, resizedWidth), totalPatches, imageTokenLengths, stream);
            }
            else
            {
                LOG_DEBUG("Skipping resize for pre-resized image %ldx%ld", image.height, image.width);
                kernel::copyImageToDeviceAndResize(image.data(), image.frames, image.height, image.width,
                    image.channels, mRawImageDevice, mResizeTmpDevice, mImageDevice, image.height, image.width, stream);
                formatImage(image, totalPatches, imageTokenLengths, stream);
            }
        }
    }

    ELLM_CHECK(totalPatches == 0 || totalPatches >= mConfig.minPatches,
        "Gemma4 Unified visual input is below the engine profile minimum");
    check::check(mVisualInput.reshape({totalPatches, mConfig.inputDim}), "Tensor reshape failed");
    check::check(mPixelPositionIds.reshape({totalPatches, 2}), "Tensor reshape failed");
    check::check(mPixelPositionIdsHost.reshape({totalPatches, 2}), "Tensor reshape failed");
    check::check(mOutputEmbedding.reshape({totalPatches, mConfig.outputHiddenSize}), "Tensor reshape failed");
    if (totalPatches > 0)
    {
        CUDA_CHECK(cudaMemcpyAsync(mPixelPositionIds.rawPointer(), mPixelPositionIdsHost.rawPointer(),
            totalPatches * 2 * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    }
    int64_t const imageCount = std::accumulate(imagesPerRequest.begin(), imagesPerRequest.end(), int64_t{0});
    mMultimodalMetrics.recordRun(imageCount, totalPatches);
}

void Gemma4UnifiedVisionRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, std::vector<int64_t> const& imageTokenLengths,
    std::vector<int64_t> const& imagesPerRequest, tokenizer::Tokenizer const* tokenizer)
{
    ELLM_CHECK(imagesPerRequest.size() == request.requests.size(), "Invalid Gemma4 Unified image request metadata");
    size_t imageIndex = 0;
    for (size_t requestIndex = 0; requestIndex < request.requests.size(); ++requestIndex)
    {
        std::vector<int32_t> ids = requestIndex < batchedInputIds.size() && !batchedInputIds[requestIndex].empty()
            ? batchedInputIds[requestIndex]
            : tokenizer->encode(request.formattedRequests[requestIndex].formattedCompleteRequest);
        check::check(!ids.empty(), "Gemma4 Unified vision failed to tokenize text");
        size_t const expectedEnd = imageIndex + static_cast<size_t>(imagesPerRequest[requestIndex]);
        std::vector<int32_t> expanded;
        for (size_t tokenIndex = 0; tokenIndex < ids.size(); ++tokenIndex)
        {
            int32_t const token = ids[tokenIndex];
            if (token == mConfig.imageTokenId)
            {
                ELLM_CHECK(imageIndex < expectedEnd,
                    "EDGELLM_BAD_MEDIA_COUNT: too many image placeholders in Gemma4 Unified prompt");
                bool const alreadyHasBegin = tokenIndex > 0 && ids[tokenIndex - 1] == mConfig.beginImageTokenId;
                bool const alreadyHasEnd
                    = tokenIndex + 1 < ids.size() && ids[tokenIndex + 1] == mConfig.endImageTokenId;
                if (!alreadyHasBegin)
                {
                    expanded.push_back(mConfig.beginImageTokenId);
                }
                expanded.insert(expanded.end(), static_cast<size_t>(imageTokenLengths.at(imageIndex)), token);
                if (!alreadyHasEnd)
                {
                    expanded.push_back(mConfig.endImageTokenId);
                }
                ++imageIndex;
            }
            else
            {
                expanded.push_back(token);
            }
        }
        ELLM_CHECK(imageIndex == expectedEnd,
            "EDGELLM_BAD_MEDIA_COUNT: image placeholder count does not match Gemma4 Unified image count");
        if (requestIndex < batchedInputIds.size())
        {
            batchedInputIds[requestIndex] = std::move(expanded);
        }
        else
        {
            batchedInputIds.push_back(std::move(expanded));
        }
    }
    ELLM_CHECK(imageIndex == imageTokenLengths.size(), "Unused Gemma4 Unified image embeddings");
}

bool Gemma4UnifiedVisionRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
    [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream, bool imageOnly, bool skipEncoderWork)
{
    try
    {
        std::vector<int64_t> imageTokenLengths;
        std::vector<int64_t> imagesPerRequest;
        if (skipEncoderWork)
        {
            imagePreprocessTokenLengthsOnly(request, imageTokenLengths, imagesPerRequest);
        }
        else
        {
            imagePreprocess(request, imageTokenLengths, imagesPerRequest, stream);
        }
        if (!imageOnly)
        {
            textPreprocess(request, batchedInputIds, imageTokenLengths, imagesPerRequest, tokenizer);
        }
        mLastMediaTokenLengths = imageTokenLengths;
        return true;
    }
    catch (std::exception const& e)
    {
        bool const actionable = isCallerActionable(e);
        if (!actionable)
        {
            LOG_ERROR("Gemma4 Unified vision preprocessing failed: %s", e.what());
        }
        // Drain async H2D copies that may still read the request's image buffers, so the caller can
        // safely release them after the failure -- including when the error propagates.
        cudaStreamSynchronize(stream);
        if (actionable)
        {
            throw;
        }
        return false;
    }
}

bool Gemma4UnifiedVisionRunner::infer(cudaStream_t stream) noexcept
{
    if (mVisualInput.getShape()[0] == 0)
    {
        return true;
    }
    TIME_STAGE(metrics::StageNames::kVISION_ENCODER, stream);
    bool status = true;
    status &= mVisualContext->setInputShape(binding_names::kVisualInput, mVisualInput.getShape().getTRTDims());
    status
        &= mVisualContext->setInputShape(binding_names::kPixelPositionIds, mPixelPositionIds.getShape().getTRTDims());
    if (!status)
    {
        LOG_ERROR("Failed to set Gemma4 Unified visual input shapes");
        return false;
    }
    if (!mVisualContext->enqueueV3(stream))
    {
        LOG_ERROR("Failed to enqueue Gemma4 Unified visual engine");
        return false;
    }
    return true;
}

} // namespace rt
} // namespace trt_edgellm
