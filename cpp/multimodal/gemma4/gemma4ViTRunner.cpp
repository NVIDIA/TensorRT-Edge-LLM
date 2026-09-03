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

#include "multimodal/gemma4/gemma4ViTRunner.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/mathUtils.h"
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

Gemma4ViTRunner::Gemma4ViTRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner(engineDir, stream)
{
    if (!validateAndFillConfig(engineDir))
    {
        LOG_ERROR("Failed to validate and fill config");
        throw std::runtime_error("Gemma4ViTRunner::Gemma4ViTRunner(): Failed to validate and fill config");
    }
    if (!allocateBuffer(stream))
    {
        LOG_ERROR("Failed to allocate buffer");
        throw std::runtime_error("Gemma4ViTRunner::Gemma4ViTRunner(): Failed to allocate buffer");
    }
}

bool Gemma4ViTRunner::validateAndFillConfig(std::string const& engineDir)
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
    if (mModelType != multimodal::ModelType::GEMMA4_VISION)
    {
        LOG_ERROR("Invalid model type: %s", modelTypeStr.c_str());
        return false;
    }

    if (!jsonConfig.contains("image_token_id"))
    {
        LOG_ERROR("Gemma4 visual config is missing image_token_id");
        return false;
    }
    mConfig.imageTokenId = jsonConfig["image_token_id"].get<int32_t>();
    // boi/eoi delimit each image's soft-token span so the model can tell
    // adjacent images apart (mirrors the HF processor layout).
    mConfig.beginImageTokenId = jsonConfig.value("boi_token_id", -1);
    mConfig.endImageTokenId = jsonConfig.value("eoi_token_id", -1);
    if (mConfig.beginImageTokenId < 0 || mConfig.endImageTokenId < 0)
    {
        std::string missingKeys = mConfig.beginImageTokenId < 0 ? "boi_token_id" : "";
        if (mConfig.endImageTokenId < 0)
        {
            missingKeys += missingKeys.empty() ? "eoi_token_id" : " and eoi_token_id";
        }
        LOG_WARNING(
            "Gemma4 visual config is missing %s; image spans will not be delimited and multi-image prompts may be "
            "misread. Re-export the model to add it.",
            missingKeys.c_str());
    }

    auto const& visionConfig = jsonConfig["vision_config"];
    mConfig.patchSize = visionConfig.value("patch_size", 16);
    mConfig.poolingKernelSize = visionConfig.value("pooling_kernel_size", 3);
    auto const& ropeParams = visionConfig.contains("rope_parameters") ? visionConfig["rope_parameters"] : visionConfig;
    mConfig.ropeTheta = ropeParams.value("rope_theta", 100.0F);
    if (mConfig.patchSize <= 0 || mConfig.poolingKernelSize <= 0)
    {
        LOG_ERROR("Gemma4 patch_size and pooling_kernel_size must be positive");
        return false;
    }

    auto const& builderConfig = jsonConfig["builder_config"];
    mConfig.minImageTokensPerImage = builderConfig["min_image_tokens"].get<int64_t>();
    mConfig.maxImageTokens = builderConfig["max_image_tokens"].get<int64_t>();
    mConfig.maxImageTokensPerImage = builderConfig["max_image_tokens_per_image"].get<int64_t>();
    mUseTrtNativeVitAttn = builderConfig.value("use_trt_native_vit_attn", false);
    if (mConfig.minImageTokensPerImage <= 0 || mConfig.maxImageTokens <= 0 || mConfig.maxImageTokensPerImage <= 0)
    {
        LOG_ERROR("Gemma4 visual builder_config image-token limits must be positive");
        return false;
    }
    mConfig.maxNumImages = std::max<int64_t>(1, mConfig.maxImageTokens / mConfig.minImageTokensPerImage);

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
    mConfig.patchSize = imageProcessorConfig.value("patch_size", mConfig.patchSize);
    mConfig.poolingKernelSize = imageProcessorConfig.value("pooling_kernel_size", mConfig.poolingKernelSize);
    mConfig.imageMean = imageProcessorConfig.value("image_mean", std::vector<float>{0.0F, 0.0F, 0.0F});
    mConfig.imageStd = imageProcessorConfig.value("image_std", std::vector<float>{1.0F, 1.0F, 1.0F});
    mConfig.maxPatchesPerImage = mConfig.maxImageTokensPerImage * mConfig.poolingKernelSize * mConfig.poolingKernelSize;

    nvinfer1::Dims const inputShapeMax
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims const inputShapeMin
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMIN);
    mConfig.maxPatches = inputShapeMax.d[0];
    mConfig.minPatches = inputShapeMin.d[0];
    mConfig.inputDim = mVisualContext->getTensorShape(binding_names::kVisualInput).d[1];
    mConfig.rotaryPosEmbDim = mVisualContext->getTensorShape(binding_names::kRotaryPosEmb).d[1];
    mConfig.outHiddenSize = mVisualEngine->getTensorShape(binding_names::kVisualOutput).d[1];
    if (mConfig.rotaryPosEmbDim <= 0)
    {
        LOG_ERROR("Gemma4 visual engine is missing rotary_pos_emb head dimension");
        return false;
    }

    // Clamp per-image patch budget to the engine profile capacity so that
    // gemma4ResizeTarget never produces an image exceeding the inputPatches buffer.
    if (mConfig.maxPatchesPerImage > mConfig.maxPatches)
    {
        LOG_WARNING(
            "Gemma4 max_image_tokens_per_image (%ld patches) exceeds engine profile capacity (%ld patches); "
            "clamping per-image budget to engine capacity",
            mConfig.maxPatchesPerImage, mConfig.maxPatches);
        mConfig.maxPatchesPerImage = mConfig.maxPatches;
        int64_t const k2 = mConfig.poolingKernelSize * mConfig.poolingKernelSize;
        mConfig.maxImageTokensPerImage = mConfig.maxPatches / k2;
    }

    return true;
}

bool Gemma4ViTRunner::allocateBuffer(cudaStream_t stream)
{
    bool setTensorAddressStatus{true};

    mVitInput = rt::Tensor({mConfig.maxPatches, mConfig.inputDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "Gemma4ViTRunner::mVitInput");
    setTensorAddressStatus &= mVisualContext->setTensorAddress(binding_names::kVisualInput, mVitInput.rawPointer());

    mPixelPositionIds = rt::Tensor({mConfig.maxPatches, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64,
        "Gemma4ViTRunner::mPixelPositionIds");
    mPixelPositionIdsHost = rt::Tensor({mConfig.maxPatches, 2}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64,
        "Gemma4ViTRunner::mPixelPositionIdsHost");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kPixelPositionIds, mPixelPositionIds.rawPointer());

    mRotaryPosEmb = rt::Tensor({mConfig.maxPatches, mConfig.rotaryPosEmbDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "Gemma4ViTRunner::mRotaryPosEmb");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kRotaryPosEmb, mRotaryPosEmb.rawPointer());

    mPoolingWeights = rt::Tensor({mConfig.maxImageTokens, mConfig.maxPatches}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "Gemma4ViTRunner::mPoolingWeights");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kPoolingWeights, mPoolingWeights.rawPointer());

    mCuSeqlens = rt::Tensor(
        {mConfig.maxNumImages + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "Gemma4ViTRunner::mCuSeqlens");
    mCuSeqlensHost = rt::Tensor({mConfig.maxNumImages + 1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32,
        "Gemma4ViTRunner::mCuSeqlensHost");
    setTensorAddressStatus &= mVisualContext->setTensorAddress(binding_names::kCuSeqlens, mCuSeqlens.rawPointer());

    if (mUseTrtNativeVitAttn)
    {
        bool const hasKvLengths
            = mVisualEngine->getTensorIOMode(binding_names::kKvLengths) != nvinfer1::TensorIOMode::kNONE;
        if (!hasKvLengths)
        {
            LOG_ERROR("Config has use_trt_native_vit_attn=true but engine is missing kv_lengths binding");
            return false;
        }
        mKvLengths = rt::Tensor({mConfig.maxNumImages + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32,
            "Gemma4ViTRunner::mKvLengths");
        setTensorAddressStatus &= mVisualContext->setTensorAddress(binding_names::kKvLengths, mKvLengths.rawPointer());
    }

    mHasMaxSeqLenCarrier
        = mVisualEngine->getTensorIOMode(binding_names::kMaxSeqLenCarrier) != nvinfer1::TensorIOMode::kNONE;
    if (mHasMaxSeqLenCarrier)
    {
        mMaxSeqLenCarrier = rt::Tensor({mConfig.maxPatchesPerImage}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32,
            "Gemma4ViTRunner::mMaxSeqLenCarrier");
        setTensorAddressStatus
            &= mVisualContext->setTensorAddress(binding_names::kMaxSeqLenCarrier, mMaxSeqLenCarrier.rawPointer());
    }

    mOutputEmbedding = rt::Tensor({mConfig.maxImageTokens, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "Gemma4ViTRunner::mOutputEmbedding");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kVisualOutput, mOutputEmbedding.rawPointer());

    if (!setTensorAddressStatus)
    {
        LOG_ERROR("Failed to set tensor address to the engine");
        return false;
    }

    auto nbBytes = mConfig.imageMean.size() * sizeof(float);
    auto channels = math::cast<int64_t>(mConfig.imageMean.size());
    mImageMean
        = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "Gemma4ViTRunner::mImageMean");
    mImageStd = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "Gemma4ViTRunner::mImageStd");
    CUDA_CHECK(
        cudaMemcpyAsync(mImageMean.rawPointer(), mConfig.imageMean.data(), nbBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(mImageStd.rawPointer(), mConfig.imageStd.data(), nbBytes, cudaMemcpyHostToDevice, stream));

    int64_t const maxImagePixels = mConfig.maxPatchesPerImage * mConfig.patchSize * mConfig.patchSize;
    mImageDevice = rt::Tensor(
        {maxImagePixels * channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "Gemma4ViTRunner::mImageDevice");
    mNormalizedImageDevice = rt::Tensor({maxImagePixels * channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "Gemma4ViTRunner::mNormalizedImageDevice");

    // GPU image-resize scratch.
    int64_t const kMaxRawPixels = kernel::kGpuResizeMaxRawDim * kernel::kGpuResizeMaxRawDim;
    // Horizontal-pass scratch holds [rawH, outW, C] floats. gemma4ResizeTarget preserves aspect ratio,
    // so rawH * outW <= sqrt(maxImagePixels * rawH * rawW) <= sqrt(maxImagePixels * kMaxRawPixels).
    int64_t const kMaxResizeTmpElems
        = static_cast<int64_t>(
              std::sqrt(static_cast<double>(maxImagePixels) * kMaxRawPixels) * kernel::kGpuResizeScratchMargin)
        * channels;
    kernel::allocateResizeScratch(channels, kMaxResizeTmpElems, mRawImageDevice, mResizeTmpDevice);

    return true;
}

void Gemma4ViTRunner::formatPatch(rt::imageUtils::ImageData const& image, std::vector<ImageGrid>& imageGrids,
    std::vector<int64_t>& imageTokenLengths, int32_t* cuSeqlensData, int64_t& cuSeqlensSize, int64_t& maxSeqLen,
    cudaStream_t stream)
{
    int64_t const height = image.height;
    int64_t const width = image.width;
    int64_t const channels = image.channels;
    int64_t const sideMult = mConfig.patchSize * mConfig.poolingKernelSize;

    ELLM_CHECK(channels == static_cast<int64_t>(mConfig.imageMean.size()),
        "Gemma4 image channels mismatch, got " + std::to_string(channels));
    ELLM_CHECK(height % sideMult == 0 && width % sideMult == 0,
        "Gemma4 image height/width must be divisible by patch_size * pooling_kernel_size");

    int64_t const patchHeight = height / mConfig.patchSize;
    int64_t const patchWidth = width / mConfig.patchSize;
    int64_t const curSeqLength = patchHeight * patchWidth;
    int64_t const curSoftLength = curSeqLength / (mConfig.poolingKernelSize * mConfig.poolingKernelSize);
    int64_t const prevCuSeqlen = cuSeqlensData[cuSeqlensSize - 1];
    ELLM_CHECK(prevCuSeqlen + curSeqLength <= mConfig.maxPatches && cuSeqlensSize < (mConfig.maxNumImages + 1),
        "Gemma4 visual input exceeds VIT engine profile");
    imageTokenLengths.emplace_back(curSoftLength);
    imageGrids.push_back(ImageGrid{patchHeight, patchWidth});
    maxSeqLen = std::max(maxSeqLen, curSeqLength);

    // mImageDevice already holds the [1, height, width, channels] image (resized on the GPU by the
    // caller); normalize and patchify consume it in place.
    check::check(mNormalizedImageDevice.reshape({1, height, width, channels}), "Tensor reshape failed");
    kernel::normalizeImage(mImageDevice, mImageMean, mImageStd, mNormalizedImageDevice, stream);
    kernel::transposeToPatchGemma4ViT(
        mNormalizedImageDevice, mVitInput, prevCuSeqlen * mConfig.inputDim, mConfig.patchSize, stream);

    int64_t* positionIds = mPixelPositionIdsHost.dataPointer<int64_t>();
    for (int64_t y = 0; y < patchHeight; ++y)
    {
        for (int64_t x = 0; x < patchWidth; ++x)
        {
            int64_t const idx = prevCuSeqlen + y * patchWidth + x;
            positionIds[2 * idx] = x;
            positionIds[2 * idx + 1] = y;
        }
    }

    cuSeqlensData[cuSeqlensSize++] = static_cast<int32_t>(prevCuSeqlen + curSeqLength);
}

void Gemma4ViTRunner::generatePoolingWeights(
    std::vector<ImageGrid> const& imageGrids, int64_t totalPatches, int64_t totalSoftTokens, cudaStream_t stream)
{
    size_t const numPoolingWeightBytes = static_cast<size_t>(totalSoftTokens) * totalPatches * sizeof(half);
    CUDA_CHECK(cudaMemsetAsync(mPoolingWeights.rawPointer(), 0, numPoolingWeightBytes, stream));

    int64_t patchStart = 0;
    int64_t softStart = 0;
    int64_t const k = mConfig.poolingKernelSize;
    for (auto const& grid : imageGrids)
    {
        kernel::initPoolingWeightsGemma4ViT(
            mPoolingWeights, patchStart, softStart, grid.patchHeight, grid.patchWidth, k, stream);
        patchStart += grid.patchHeight * grid.patchWidth;
        softStart += (grid.patchHeight / k) * (grid.patchWidth / k);
    }
}

void Gemma4ViTRunner::imagePreprocessTokenLengthsOnly(
    rt::LLMGenerationRequest const& request, std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& numImages)
{
    int64_t totalSoftTokens = 0;
    for (auto const& req : request.requests)
    {
        int64_t numImage = 0;
        for (auto const& image : req.imageBuffers)
        {
            auto [resizedHeight, resizedWidth] = image.doResize
                ? rt::imageUtils::gemma4ResizeTarget(image.height, image.width, mConfig.maxImageTokensPerImage,
                      mConfig.poolingKernelSize, mConfig.patchSize)
                : std::make_tuple(image.height, image.width);
            int64_t const patchHeight = resizedHeight / mConfig.patchSize;
            int64_t const patchWidth = resizedWidth / mConfig.patchSize;
            int64_t const softTokens
                = (patchHeight * patchWidth) / (mConfig.poolingKernelSize * mConfig.poolingKernelSize);
            imageTokenLengths.emplace_back(softTokens);
            totalSoftTokens += softTokens;
            ++numImage;
        }
        numImages.emplace_back(numImage);
    }

    if (totalSoftTokens > 0)
    {
        check::check(mOutputEmbedding.reshape({totalSoftTokens, mConfig.outHiddenSize}), "Tensor reshape failed");
    }
}

void Gemma4ViTRunner::imagePreprocess(rt::LLMGenerationRequest const& request, std::vector<ImageGrid>& imageGrids,
    std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& numImages, cudaStream_t stream)
{
    // Restore mVitInput to full engine-profile capacity before writing patches.
    // imagePreprocess shrinks it to {totalPatches, inputDim} at the end (for engine execution),
    // so a subsequent call would see a reduced shape and fail the capacity check in the kernel.
    check::check(mVitInput.reshape({mConfig.maxPatches, mConfig.inputDim}), "Tensor reshape failed");

    int32_t* cuSeqlensData = mCuSeqlensHost.dataPointer<int32_t>();
    cuSeqlensData[0] = 0;
    int64_t cuSeqlensSize = 1;
    int64_t maxSeqLen = 0;

    for (auto const& req : request.requests)
    {
        int64_t numImage = 0;
        for (auto const& image : req.imageBuffers)
        {
            if (image.doResize)
            {
                auto [resizedHeight, resizedWidth] = rt::imageUtils::gemma4ResizeTarget(image.height, image.width,
                    mConfig.maxImageTokensPerImage, mConfig.poolingKernelSize, mConfig.patchSize);
                kernel::copyImageToDeviceAndResize(image.data(), image.frames, image.height, image.width,
                    image.channels, mRawImageDevice, mResizeTmpDevice, mImageDevice, resizedHeight, resizedWidth,
                    stream);
                formatPatch(image.resizedMeta(resizedHeight, resizedWidth), imageGrids, imageTokenLengths,
                    cuSeqlensData, cuSeqlensSize, maxSeqLen, stream);
            }
            else
            {
                LOG_DEBUG("Skipping resize for pre-resized image %ldx%ld", image.height, image.width);
                kernel::copyImageToDeviceAndResize(image.data(), image.frames, image.height, image.width,
                    image.channels, mRawImageDevice, mResizeTmpDevice, mImageDevice, image.height, image.width, stream);
                formatPatch(image, imageGrids, imageTokenLengths, cuSeqlensData, cuSeqlensSize, maxSeqLen, stream);
            }
            ++numImage;
        }
        numImages.emplace_back(numImage);
    }

    int64_t const totalPatches = cuSeqlensData[cuSeqlensSize - 1];
    if (totalPatches == 0)
    {
        check::check(mVitInput.reshape({totalPatches, mConfig.inputDim}), "Tensor reshape failed");
        return;
    }

    int64_t const totalSoftTokens = std::accumulate(imageTokenLengths.begin(), imageTokenLengths.end(), int64_t(0));
    ELLM_CHECK(totalPatches >= mConfig.minPatches && totalPatches <= mConfig.maxPatches,
        "Gemma4 total patch count exceeds VIT engine profile");
    ELLM_CHECK(totalSoftTokens <= mConfig.maxImageTokens, "Gemma4 total soft token count exceeds VIT engine profile");

    check::check(mVitInput.reshape({totalPatches, mConfig.inputDim}), "Tensor reshape failed");
    check::check(mPixelPositionIds.reshape({totalPatches, 2}), "Tensor reshape failed");
    check::check(mPixelPositionIdsHost.reshape({totalPatches, 2}), "Tensor reshape failed");
    check::check(mRotaryPosEmb.reshape({totalPatches, mConfig.rotaryPosEmbDim}), "Tensor reshape failed");
    check::check(mCuSeqlens.reshape({cuSeqlensSize}), "Tensor reshape failed");
    check::check(mOutputEmbedding.reshape({totalSoftTokens, mConfig.outHiddenSize}), "Tensor reshape failed");
    check::check(mPoolingWeights.reshape({totalSoftTokens, totalPatches}), "Tensor reshape failed");
    if (mHasMaxSeqLenCarrier)
    {
        check::check(mMaxSeqLenCarrier.reshape({maxSeqLen}), "Tensor reshape failed");
    }
    if (mUseTrtNativeVitAttn)
    {
        check::check(mKvLengths.reshape({cuSeqlensSize}), "Tensor reshape failed");
    }

    generatePoolingWeights(imageGrids, totalPatches, totalSoftTokens, stream);
    CUDA_CHECK(cudaMemcpyAsync(mPixelPositionIds.rawPointer(), mPixelPositionIdsHost.rawPointer(),
        totalPatches * 2 * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    kernel::initRotaryPosEmbGemma4ViT(mRotaryPosEmb, mPixelPositionIds, mConfig.ropeTheta, stream);
    CUDA_CHECK(cudaMemcpyAsync(mCuSeqlens.rawPointer(), mCuSeqlensHost.rawPointer(), cuSeqlensSize * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    if (mUseTrtNativeVitAttn)
    {
        // TRT native attention expects kv_lengths in cumulative format (same as cu_seqlens),
        // e.g. [0, 256, 512] for two 256-patch images. The naming is historical.
        CUDA_CHECK(cudaMemcpyAsync(mKvLengths.rawPointer(), mCuSeqlensHost.rawPointer(),
            cuSeqlensSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    }

    int64_t imageCount = std::accumulate(numImages.begin(), numImages.end(), int64_t(0));
    mMultimodalMetrics.recordRun(imageCount, totalSoftTokens);
}

void Gemma4ViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int64_t> const& numImages,
    std::vector<int64_t> const& imageTokenLengths, trt_edgellm::tokenizer::Tokenizer const* tokenizer)
{
    if (numImages.size() != request.requests.size())
    {
        std::string errorMsg = "Gemma4ViTRunner::textPreprocess() numImages.size() != request.requests.size(), "
            + std::to_string(numImages.size()) + " != " + std::to_string(request.requests.size());
        LOG_ERROR("%s", errorMsg.c_str());
        throw std::runtime_error(errorMsg);
    }

    int64_t imageIndex = 0;
    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        std::vector<int32_t> ids;
        if (i < batchInputIds.size() && !batchInputIds[i].empty())
        {
            ids = batchInputIds[i];
        }
        else
        {
            ids = tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);
            check::check(!ids.empty(), "Gemma4ViTRunner::textPreprocess() Failed to encode text");
        }

        bool const wrapImages = mConfig.beginImageTokenId >= 0 && mConfig.endImageTokenId >= 0;
        // Per-request media window (numImages[i]): a placeholder may only consume this request's own media (batch
        // isolation).
        int64_t const mediaEnd = imageIndex + numImages[i];

        std::vector<int32_t> newIds;
        // Compute expanded size: non-image tokens + sum of image token lengths
        // (+2 boundary tokens per image when wrapping).
        size_t expandedSize = 0;
        int64_t imgIdx = imageIndex;
        for (auto tokenId : ids)
        {
            expandedSize += (tokenId == mConfig.imageTokenId && imgIdx < mediaEnd)
                ? imageTokenLengths.at(imgIdx++) + (wrapImages ? 2 : 0)
                : 1;
        }
        newIds.reserve(expandedSize);
        for (size_t tokenIndex = 0; tokenIndex < ids.size(); ++tokenIndex)
        {
            int32_t const tokenId = ids[tokenIndex];
            if (tokenId == mConfig.imageTokenId)
            {
                ELLM_CHECK(imageIndex < mediaEnd,
                    "EDGELLM_BAD_MEDIA_COUNT: Gemma4ViTRunner::textPreprocess() placeholder count exceeds this "
                    "request's image count");
                bool const alreadyHasBegin
                    = wrapImages && tokenIndex > 0 && ids[tokenIndex - 1] == mConfig.beginImageTokenId;
                bool const alreadyHasEnd
                    = wrapImages && tokenIndex + 1 < ids.size() && ids[tokenIndex + 1] == mConfig.endImageTokenId;
                if (wrapImages && !alreadyHasBegin)
                {
                    newIds.push_back(mConfig.beginImageTokenId);
                }
                int64_t const numImageTokens = imageTokenLengths.at(imageIndex);
                for (int64_t k = 0; k < numImageTokens; ++k)
                {
                    newIds.push_back(tokenId);
                }
                if (wrapImages && !alreadyHasEnd)
                {
                    newIds.push_back(mConfig.endImageTokenId);
                }
                ++imageIndex;
            }
            else
            {
                newIds.push_back(tokenId);
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
        ELLM_CHECK(imageIndex == mediaEnd,
            "EDGELLM_BAD_MEDIA_COUNT: Gemma4ViTRunner::textPreprocess() placeholder count is smaller than this "
            "request's image count");
    }
}

bool Gemma4ViTRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
    [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream, bool imageOnly, bool skipEncoderWork)
{
    std::vector<ImageGrid> imageGrids;
    std::vector<int64_t> imageTokenLengths;
    std::vector<int64_t> numImages;

    try
    {
        if (skipEncoderWork)
        {
            imagePreprocessTokenLengthsOnly(request, imageTokenLengths, numImages);
        }
        else
        {
            imagePreprocess(request, imageGrids, imageTokenLengths, numImages, stream);
        }
        if (!imageOnly)
        {
            textPreprocess(request, batchedInputIds, numImages, imageTokenLengths, tokenizer);
        }
    }
    catch (std::exception const& e)
    {
        bool const actionable = isCallerActionable(e);
        if (!actionable)
        {
            LOG_ERROR("Failed: %s", e.what());
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

    mLastMediaTokenLengths = imageTokenLengths;
    return true;
}

bool Gemma4ViTRunner::infer(cudaStream_t stream) noexcept
{
    if (mVitInput.getShape()[0] == 0)
    {
        return true;
    }

    {
        TIME_STAGE(metrics::StageNames::kVISION_ENCODER, stream);

        bool setEngineIOStatus{true};
        setEngineIOStatus
            &= mVisualContext->setInputShape(binding_names::kVisualInput, mVitInput.getShape().getTRTDims());
        setEngineIOStatus &= mVisualContext->setInputShape(
            binding_names::kPixelPositionIds, mPixelPositionIds.getShape().getTRTDims());
        setEngineIOStatus
            &= mVisualContext->setInputShape(binding_names::kRotaryPosEmb, mRotaryPosEmb.getShape().getTRTDims());
        setEngineIOStatus
            &= mVisualContext->setInputShape(binding_names::kCuSeqlens, mCuSeqlens.getShape().getTRTDims());
        setEngineIOStatus
            &= mVisualContext->setInputShape(binding_names::kPoolingWeights, mPoolingWeights.getShape().getTRTDims());
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

        if (!setEngineIOStatus)
        {
            LOG_ERROR("Failed to bind Gemma4 visual engine input tensors.");
            return false;
        }

        bool enqueueStatus = mVisualContext->enqueueV3(stream);
        if (!enqueueStatus)
        {
            LOG_ERROR("Failed to enqueue Gemma4 visual engine.");
            return false;
        }
    }

    return true;
}

} // namespace rt
} // namespace trt_edgellm
