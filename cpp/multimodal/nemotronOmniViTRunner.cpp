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

#include "nemotronOmniViTRunner.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "multimodal/imageUtils.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include <fstream>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{

NemotronOmniViTRunner::NemotronOmniViTRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner(engineDir, stream)
{
    bool const configValid = validateAndFillConfig(engineDir);
    ELLM_CHECK(configValid, "NemotronOmniViTRunner: Failed to validate and fill config");
    bool const bufferAllocated = allocateBuffer(stream);
    ELLM_CHECK(bufferAllocated, "NemotronOmniViTRunner: Failed to allocate buffer");
}

bool NemotronOmniViTRunner::validateAndFillConfig(std::string const& engineDir)
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
        LOG_ERROR("Failed to parse config file: %s", e.what());
        return false;
    }

    mModelType = multimodal::ModelType::NEMOTRON_OMNI_VISION_ENCODER;

    if (jsonConfig.contains("llm_config") && jsonConfig["llm_config"].contains("vocab_size"))
    {
        mConfig.vocabSize = jsonConfig["llm_config"]["vocab_size"].get<int32_t>();
    }

    if (jsonConfig.contains("img_context_token_id"))
    {
        mConfig.imgContextTokenId = jsonConfig["img_context_token_id"].get<int32_t>();
    }
    if (jsonConfig.contains("img_start_token_id"))
    {
        mConfig.imgStartTokenId = jsonConfig["img_start_token_id"].get<int32_t>();
    }
    if (jsonConfig.contains("img_end_token_id"))
    {
        mConfig.imgEndTokenId = jsonConfig["img_end_token_id"].get<int32_t>();
    }

    if (jsonConfig.contains("force_image_size"))
    {
        int64_t const imgSize = jsonConfig["force_image_size"].get<int64_t>();
        mConfig.blockImageSizeH = imgSize;
        mConfig.blockImageSizeW = imgSize;
    }
    else
    {
        LOG_ERROR("force_image_size not found in config.json");
        return false;
    }

    if (jsonConfig.contains("norm_mean") && jsonConfig["norm_mean"].is_array())
    {
        auto const& mean = jsonConfig["norm_mean"];
        for (size_t i = 0; i < std::min(mean.size(), size_t(3)); ++i)
        {
            mConfig.imageMean[i] = mean[i].get<float>();
        }
    }
    if (jsonConfig.contains("norm_std") && jsonConfig["norm_std"].is_array())
    {
        auto const& std = jsonConfig["norm_std"];
        for (size_t i = 0; i < std::min(std.size(), size_t(3)); ++i)
        {
            mConfig.imageStd[i] = std[i].get<float>();
        }
    }

    nvinfer1::Dims const inputShapeMax
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims const inputShapeMin
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMIN);
    mConfig.maxNumBlocks = inputShapeMax.d[0];
    mConfig.minNumBlocks = inputShapeMin.d[0];
    mConfig.outHiddenSize = mVisualEngine->getTensorShape(binding_names::kVisualOutput).d[2];

    mConfig.tokensPerBlock = mVisualEngine->getTensorShape(binding_names::kVisualOutput).d[1];

    // Per-image tile budget used for aspect-ratio grid selection in imagePreprocess.
    // Fall back to engine-wide limits when builder_config is absent.
    if (jsonConfig.contains("builder_config"))
    {
        auto const& builderConfig = jsonConfig["builder_config"];
        mConfig.maxImageTokensPerImage
            = builderConfig.value("max_image_tokens_per_image", mConfig.maxNumBlocks * mConfig.tokensPerBlock);
        mConfig.minImageTokensPerImage = builderConfig.value("min_image_tokens", mConfig.tokensPerBlock);
    }
    else
    {
        mConfig.maxImageTokensPerImage = mConfig.maxNumBlocks * mConfig.tokensPerBlock;
        mConfig.minImageTokensPerImage = mConfig.tokensPerBlock;
    }

    LOG_INFO(
        "NemotronOmniViTRunner: image=%ldx%ld, blocks=[%ld,%ld], tokensPerBlock=%ld, "
        "outHidden=%ld, imgContextTokenId=%d",
        mConfig.blockImageSizeH, mConfig.blockImageSizeW, mConfig.minNumBlocks, mConfig.maxNumBlocks,
        mConfig.tokensPerBlock, mConfig.outHiddenSize, mConfig.imgContextTokenId);

    return true;
}

bool NemotronOmniViTRunner::allocateBuffer(cudaStream_t stream)
{
    bool setTensorAddressStatus{true};

    mVitInput
        = rt::Tensor({mConfig.maxNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "NemotronOmniViTRunner::mVitInput");
    setTensorAddressStatus &= mVisualContext->setTensorAddress(binding_names::kVisualInput, mVitInput.rawPointer());

    // Output: [maxNumBlocks * tokensPerBlock, outHiddenSize]
    int64_t const maxTotalTokens = mConfig.maxNumBlocks * mConfig.tokensPerBlock;
    mOutputEmbedding = rt::Tensor({maxTotalTokens, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "NemotronOmniViTRunner::mOutputEmbedding");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kVisualOutput, mOutputEmbedding.rawPointer());

    if (!setTensorAddressStatus)
    {
        LOG_ERROR("Failed to set tensor address to the engine");
        return false;
    }

    // Copy image mean and std to device
    int64_t const channels = static_cast<int64_t>(mConfig.imageMean.size());
    mImageMean = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "mImageMean");
    mImageStd = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "mImageStd");
    CUDA_CHECK(cudaMemcpyAsync(
        mImageMean.rawPointer(), mConfig.imageMean.data(), channels * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mImageStd.rawPointer(), mConfig.imageStd.data(), channels * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Pre-allocate temporary image buffers for preprocessing
    int64_t const maxImagePixels = mVitInput.getShape().volume();
    mImageDevice = rt::Tensor(
        {maxImagePixels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "NemotronOmniViTRunner::mImageDevice");
    mNormalizedImageDevice = rt::Tensor({maxImagePixels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "NemotronOmniViTRunner::mNormalizedImageDevice");
    // Scratch buffer for resizeImage output (4D placeholder shape; resizeImage reshapes per call).
    rt::Tensor resizeBuffer({1, 1, maxImagePixels / channels, channels}, rt::DeviceType::kCPU,
        nvinfer1::DataType::kUINT8, "NemotronOmniViTRunner::resizeBuffer");
    mResizedImageHost = rt::imageUtils::ImageData(std::move(resizeBuffer));
    // Thumbnail image has fixed size: 1 x blockImageSizeH x blockImageSizeW x channels (4D, frames=1).
    rt::Tensor thumbnailBuffer({1, mConfig.blockImageSizeH, mConfig.blockImageSizeW, channels}, rt::DeviceType::kCPU,
        nvinfer1::DataType::kUINT8, "NemotronOmniViTRunner::thumbnailBuffer");
    mThumbnailImageHost = rt::imageUtils::ImageData(std::move(thumbnailBuffer));

    return true;
}

void NemotronOmniViTRunner::formatPatch(rt::imageUtils::ImageData const& image, std::vector<int64_t>& imageTokenLengths,
    int64_t& numImages, int64_t& totalNumBlocks, bool isThumbnail, cudaStream_t stream)
{
    int64_t height = image.height;
    int64_t width = image.width;
    int64_t channels = image.channels;
    unsigned char* imageData = image.data(); // In hwc order

    ELLM_CHECK(channels == mConfig.numChannels,
        "Image channels mismatch, got " + std::to_string(channels) + ", expected "
            + std::to_string(mConfig.numChannels));
    ELLM_CHECK(height % mConfig.blockImageSizeH == 0 && width % mConfig.blockImageSizeW == 0,
        "Image height or width is not divisible by blockImageSizeH or blockImageSizeW, "
        "got height: "
            + std::to_string(height) + ", width: " + std::to_string(width)
            + ", blockImageSizeH: " + std::to_string(mConfig.blockImageSizeH)
            + ", blockImageSizeW: " + std::to_string(mConfig.blockImageSizeW));

    int64_t curNumBlocks = (height / mConfig.blockImageSizeH) * (width / mConfig.blockImageSizeW);
    ELLM_CHECK(totalNumBlocks + curNumBlocks <= mConfig.maxNumBlocks,
        "totalNumBlocks " + std::to_string(totalNumBlocks) + " + curNumBlocks " + std::to_string(curNumBlocks)
            + " exceeds the limitation, max = " + std::to_string(mConfig.maxNumBlocks) + " of VIT engine.");

    int64_t curTokenLength = curNumBlocks * mConfig.tokensPerBlock;
    if (isThumbnail)
    {
        // Add to the last image token length, instead of considered as a new image
        imageTokenLengths.back() += curTokenLength;
    }
    else
    {
        imageTokenLengths.push_back(curTokenLength);
        ++numImages;
    }

    // Reshape pre-allocated temporary buffers to current image dimensions
    check::check(mImageDevice.reshape({1, height, width, channels}), "Tensor reshape failed");
    check::check(mNormalizedImageDevice.reshape({1, height, width, channels}), "Tensor reshape failed");

    // Copy image to device
    CUDA_CHECK(cudaMemcpyAsync(
        mImageDevice.rawPointer(), imageData, height * width * channels, cudaMemcpyHostToDevice, stream));

    // Normalize image
    kernel::normalizeImage(mImageDevice, mImageMean, mImageStd, mNormalizedImageDevice, stream);

    // Transpose to patch
    int64_t offset = totalNumBlocks * mConfig.numChannels * mConfig.blockImageSizeH * mConfig.blockImageSizeW;
    kernel::transposeToPatchInternVLPhi4MM(mNormalizedImageDevice, mVitInput, offset, stream);

    // Update numBlocks
    totalNumBlocks += curNumBlocks;
}

void NemotronOmniViTRunner::imagePreprocess(rt::LLMGenerationRequest const& request,
    std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& numImages, cudaStream_t stream)
{
    mTotalNumBlocks = 0;

    int64_t totalImages = 0;
    for (auto const& req : request.requests)
    {
        totalImages += static_cast<int64_t>(req.imageBuffers.size());
    }

    if (totalImages == 0)
    {
        check::check(mVitInput.reshape({0, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW}),
            "Tensor reshape failed");
        return;
    }

    for (auto const& req : request.requests)
    {
        int64_t numImage = 0;
        for (auto const& image : req.imageBuffers)
        {
            int64_t const blocksBeforePatch = mTotalNumBlocks;

            // Resize image to the aspect-ratio-matched tile grid within the per-image tile budget
            auto [resizedHeight, resizedWidth]
                = imageUtils::computeBestBlockGridForResize(image.height, image.width, mConfig.minImageTokensPerImage,
                    mConfig.maxImageTokensPerImage, mConfig.blockImageSizeH, mConfig.blockImageSizeW);
            rt::imageUtils::resizeImage(
                image, mResizedImageHost, resizedWidth, resizedHeight, rt::imageUtils::InterpolationMode::kBICUBIC);
            formatPatch(mResizedImageHost, imageTokenLengths, numImage, mTotalNumBlocks, false, stream);

            // Only add thumbnail when the image has more than 1 block (matches HuggingFace behavior)
            int64_t const mainImageBlocks = mTotalNumBlocks - blocksBeforePatch;
            if (mainImageBlocks > 1)
            {
                rt::imageUtils::resizeImage(image, mThumbnailImageHost, mConfig.blockImageSizeW,
                    mConfig.blockImageSizeH, rt::imageUtils::InterpolationMode::kBICUBIC);
                formatPatch(mThumbnailImageHost, imageTokenLengths, numImage, mTotalNumBlocks, true, stream);
            }
        }
        numImages.emplace_back(numImage);
    }

    if (mTotalNumBlocks == 0)
    {
        check::check(mVitInput.reshape({0, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW}),
            "Tensor reshape failed");
        return;
    }

    ELLM_CHECK(mTotalNumBlocks >= mConfig.minNumBlocks && mTotalNumBlocks <= mConfig.maxNumBlocks,
        "totalNumBlocks " + std::to_string(mTotalNumBlocks)
            + " exceeds the limitation, max = " + std::to_string(mConfig.maxNumBlocks)
            + ", min = " + std::to_string(mConfig.minNumBlocks) + " of VIT engine.");

    // Calculate total image tokens for profiling
    int64_t const totalImageTokens = mTotalNumBlocks * mConfig.tokensPerBlock;

    // Record performance data
    mMultimodalMetrics.recordRun(mTotalNumBlocks, totalImageTokens);

    check::check(
        mVitInput.reshape({mTotalNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW}),
        "Tensor reshape failed");
    check::check(mOutputEmbedding.reshape({totalImageTokens, mConfig.outHiddenSize}), "Tensor reshape failed");
}

void NemotronOmniViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int64_t> const& numImages,
    std::vector<int64_t> const& imageTokenLengths, trt_edgellm::tokenizer::Tokenizer const* tokenizer)
{
    // Reuse batchInputIds if the audio runner already tokenized (combined image+audio);
    // otherwise tokenize from scratch. Each image placeholder is then repeated to match
    // the image's token count so the runtime injects one visual feature per copy.
    bool const alreadyTokenized = batchInputIds.size() == request.requests.size();
    int imageIndex = 0;

    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        std::vector<int32_t> ids = alreadyTokenized
            ? std::move(batchInputIds[i])
            : tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);
        check::check(!ids.empty(), "Failed to encode text");

        std::vector<int32_t> newIds;
        newIds.reserve(ids.size());
        for (auto const& id : ids)
        {
            if (id == mConfig.imgContextTokenId)
            {
                int64_t const numImageTokens = imageTokenLengths.at(imageIndex);
                for (int64_t k = 0; k < numImageTokens; ++k)
                {
                    newIds.push_back(mConfig.imgContextTokenId);
                }
                ++imageIndex;
            }
            else
            {
                newIds.push_back(id);
            }
        }
        if (alreadyTokenized)
        {
            batchInputIds[i] = std::move(newIds);
        }
        else
        {
            batchInputIds.emplace_back(std::move(newIds));
        }
    }
}

bool NemotronOmniViTRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
    [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream, bool imageOnly) noexcept
{
    std::vector<int64_t> imageTokenLengths;
    std::vector<int64_t> numImages;

    try
    {
        imagePreprocess(request, imageTokenLengths, numImages, stream);
        mPreparedArtifactRowCounts = imageTokenLengths;
        if (!imageOnly)
        {
            textPreprocess(request, batchedInputIds, numImages, imageTokenLengths, tokenizer);
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed: %s", e.what());
        return false;
    }

    return true;
}

bool NemotronOmniViTRunner::prepareArtifactSubset(rt::LLMGenerationRequest const& request,
    [[maybe_unused]] std::vector<size_t> const& originalItemIndices, cudaStream_t stream)
{
    std::vector<int64_t> imageTokenLengths;
    std::vector<int64_t> numImages;
    try
    {
        imagePreprocess(request, imageTokenLengths, numImages, stream);
        mPreparedArtifactRowCounts = imageTokenLengths;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to prepare selected vision artifacts: %s", e.what());
        return false;
    }
    return true;
}

std::vector<int64_t> NemotronOmniViTRunner::preparedArtifactRowCounts() const
{
    return mPreparedArtifactRowCounts;
}

bool NemotronOmniViTRunner::infer(cudaStream_t stream) noexcept
{
    if (mTotalNumBlocks == 0)
    {
        return true; // No images to process
    }

    {
        TIME_STAGE(metrics::StageNames::kMULTIMODAL_PROCESSING, stream);

        nvinfer1::Dims4 inputDims(
            mTotalNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW);
        if (!mVisualContext->setInputShape(binding_names::kVisualInput, inputDims))
        {
            LOG_ERROR("Failed to bind engine input tensors.");
            return false;
        }

        if (!mVisualContext->enqueueV3(stream))
        {
            LOG_ERROR("Failed to enqueue engine.");
            return false;
        }
    }

    return true;
}

} // namespace rt
} // namespace trt_edgellm
