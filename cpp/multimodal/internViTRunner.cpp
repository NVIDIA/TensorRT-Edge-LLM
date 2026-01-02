/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internViTRunner.h"
#include "common/bindingNames.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "multimodal/imageUtils.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
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

InternViTRunner::InternViTRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner(engineDir, stream)
{
    if (!validateAndFillConfig(engineDir))
    {
        LOG_ERROR("InternViTRunner::InternViTRunner(): Failed to validate and fill config");
        throw std::runtime_error("InternViTRunner::InternViTRunner(): Failed to validate and fill config");
    }
    if (!allocateBuffer(stream))
    {
        LOG_ERROR("InternViTRunner::InternViTRunner(): Failed to allocate buffer");
        throw std::runtime_error("InternViTRunner::InternViTRunner(): Failed to allocate buffer");
    }
}

bool InternViTRunner::validateAndFillConfig(std::string const& engineDir)
{
    Json jsonConfig;

    std::string configPath = engineDir + "/config.json";
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("InternViTRunner::validateAndFillConfig(): Failed to open config file: %s", configPath.c_str());
        return false;
    }

    try
    {
        jsonConfig = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("InternViTRunner::validateAndFillConfig(): Failed to parse config file with error: %s", e.what());
        return false;
    }

    std::string modelTypeStr = jsonConfig["model_type"].get<std::string>();
    mModelType = multimodal::stringToModelType(modelTypeStr);
    if (mModelType != multimodal::ModelType::INTERNVL)
    {
        LOG_ERROR("InternViTRunner::validateAndFillConfig(): Invalid model type: %s", modelTypeStr.c_str());
        return false;
    }

    mConfig.imageTokenId = jsonConfig["image_token_id"].get<int32_t>();
    auto textConfig = jsonConfig["text_config"];
    mConfig.vocabSize = textConfig["vocab_size"].get<int32_t>();

    auto visionConfig = jsonConfig["vision_config"];
    mConfig.numChannels = visionConfig["num_channels"].get<int64_t>();
    mConfig.patchSizeH = visionConfig["patch_size"][0].get<int64_t>();
    mConfig.patchSizeW = visionConfig["patch_size"][1].get<int64_t>();
    mConfig.blockImageSizeH = visionConfig["image_size"][0].get<int64_t>();
    mConfig.blockImageSizeW = visionConfig["image_size"][1].get<int64_t>();

    auto builderConfig = jsonConfig["builder_config"];
    mConfig.minImageTokensPerImage = builderConfig["min_image_tokens"].get<int64_t>();
    mConfig.maxImageTokensPerImage = builderConfig["max_image_tokens_per_image"].get<int64_t>();

    // Get config from engine shapes
    nvinfer1::Dims const inputShapeMax
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims const inputShapeMin
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMIN);
    mConfig.maxNumBlocks = inputShapeMax.d[0];
    mConfig.minNumBlocks = inputShapeMin.d[0];
    mConfig.outHiddenSize = mVisualEngine->getTensorShape(binding_names::kVisualOutput).d[1];

    return true;
}

bool InternViTRunner::allocateBuffer(cudaStream_t stream)
{
    bool setTensorAddressStatus{true};
    LOG_INFO(
        "InternViTRunner::allocateBuffer() mConfig.maxNumBlocks: %d, mConfig.numChannels: %d, mConfig.blockImageSizeH: "
        "%d, mConfig.blockImageSizeW: %d",
        mConfig.maxNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW);
    mVitInput
        = rt::Tensor({mConfig.maxNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "InternViTRunner::mVitInput");
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kVisualInput, mVitInput.rawPointer());
    LOG_INFO("InternViTRunner::allocateBuffer() mConfig.maxNumBlocks: %d, mConfig.outHiddenSize: %d",
        mConfig.maxNumBlocks, mConfig.outHiddenSize);
    // In InternVL3, each block generates 256 tokens, so output size is maxNumBlocks*256
    mOutputEmbedding = rt::Tensor({mConfig.maxNumBlocks * 256, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "InternViTRunner::mOutputEmbedding");
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kVisualOutput, mOutputEmbedding.rawPointer());
    if (!setTensorAddressStatus)
    {
        LOG_ERROR("Failed to set tensor address to the engine");
        return false;
    }

    // Copy image mean and std to device to be used in normalizeImage
    int64_t const channels = static_cast<int64_t>(mConfig.imageMean.size());
    mImageMean
        = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "InternViTRunner::mImageMean");
    mImageStd = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "InternViTRunner::mImageStd");
    CUDA_CHECK(cudaMemcpyAsync(
        mImageMean.rawPointer(), mConfig.imageMean.data(), channels * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mImageStd.rawPointer(), mConfig.imageStd.data(), channels * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Pre-allocate temporary image buffers for preprocessing
    int64_t const maxImagePixels = mVitInput.getShape().volume();
    mImageDevice = rt::Tensor(
        {maxImagePixels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "InternViTRunner::mImageDevice");
    mNormalizedImageDevice = rt::Tensor(
        {maxImagePixels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "InternViTRunner::mNormalizedImageDevice");
    // Set max image size to 1xmaxImagePixelsxchannels, will reshape to actual image size in resizeImage
    rt::Tensor resizeBuffer({1, maxImagePixels, channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8,
        "InternViTRunner::resizeBuffer");
    mResizedImageHost = rt::imageUtils::ImageData(std::move(resizeBuffer));
    // Thumbnail image has fixed size: blockImageSizeH x blockImageSizeW x channels)
    rt::Tensor thumbnailBuffer({mConfig.blockImageSizeH, mConfig.blockImageSizeW, channels}, rt::DeviceType::kCPU,
        nvinfer1::DataType::kUINT8, "InternViTRunner::thumbnailBuffer");
    mThumbnailImageHost = rt::imageUtils::ImageData(std::move(thumbnailBuffer));

    return true;
}

void InternViTRunner::formatPatch(imageUtils::ImageData const& image, std::vector<int64_t>& imageTokenLengths,
    int64_t& numImages, int64_t& totalNumBlocks, bool isThumbnail, cudaStream_t stream)
{
    int64_t height = image.height;
    int64_t width = image.width;
    int64_t channels = image.channels;
    unsigned char* imageData = image.data(); // In hwc order

    if (channels != mConfig.numChannels)
    {
        throw std::runtime_error("Image channels mismatch, got " + std::to_string(channels) + ", expected "
            + std::to_string(mConfig.numChannels));
    }
    if (height % mConfig.blockImageSizeH != 0 || width % mConfig.blockImageSizeW != 0)
    {
        throw std::runtime_error(
            "Image height or width is not divisible by blockImageSizeH or blockImageSizeW, "
            "got height: "
            + std::to_string(height) + ", width: " + std::to_string(width)
            + ", blockImageSizeH: " + std::to_string(mConfig.blockImageSizeH)
            + ", blockImageSizeW: " + std::to_string(mConfig.blockImageSizeW));
    }

    int64_t curNumBlocks = (height / mConfig.blockImageSizeH) * (width / mConfig.blockImageSizeW);
    if (totalNumBlocks + curNumBlocks > mConfig.maxNumBlocks)
    {
        throw std::runtime_error("totalNumBlocks " + std::to_string(totalNumBlocks) + " + curNumBlocks "
            + std::to_string(curNumBlocks) + " exceeds the limitation, max = " + std::to_string(mConfig.maxNumBlocks)
            + " of VIT engine.");
    }

    int64_t curTokenLength = curNumBlocks * 256;
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
    mImageDevice.reshape({1, height, width, channels});
    mNormalizedImageDevice.reshape({1, height, width, channels});

    // Copy image to device
    CUDA_CHECK(cudaMemcpyAsync(mImageDevice.rawPointer(), imageData, height * width * channels * sizeof(unsigned char),
        cudaMemcpyHostToDevice, stream));

    // Normalize image
    kernel::normalizeImage(mImageDevice, mImageMean, mImageStd, mNormalizedImageDevice, stream);

    // Transpose to patch
    int64_t offset = totalNumBlocks * mConfig.numChannels * mConfig.blockImageSizeH * mConfig.blockImageSizeW;
    kernel::transposeToPatchInternVLPhi4MM(mNormalizedImageDevice, mVitInput, offset, stream);

    // Update numBlocks
    totalNumBlocks += curNumBlocks;
}

void InternViTRunner::imagePreprocess(rt::LLMGenerationRequest const& request, std::vector<int64_t>& imageTokenLengths,
    std::vector<int64_t>& numImages, bool doResize, cudaStream_t stream)
{
    int64_t totalNumBlocks = 0;

    for (auto const& req : request.requests)
    {
        int64_t numImage = 0;
        for (auto const& image : req.imageBuffers)
        {
            if (doResize)
            {
                auto [resizedHeight, resizedWidth] = imageUtils::computeBestBlockGridForResize(image.height,
                    image.width, mConfig.minImageTokensPerImage, mConfig.maxImageTokensPerImage,
                    mConfig.blockImageSizeH, mConfig.blockImageSizeW);
                rt::imageUtils::resizeImage(image, mResizedImageHost, resizedWidth, resizedHeight);
                formatPatch(mResizedImageHost, imageTokenLengths, numImage, totalNumBlocks, false, stream);
            }
            else
            {
                formatPatch(image, imageTokenLengths, numImage, totalNumBlocks, false, stream);
            }
            // Add thumbnail image by default (use separate buffer to avoid race condition)
            rt::imageUtils::resizeImage(image, mThumbnailImageHost, mConfig.blockImageSizeW, mConfig.blockImageSizeH);
            formatPatch(mThumbnailImageHost, imageTokenLengths, numImage, totalNumBlocks, true, stream);
        }
        numImages.emplace_back(numImage);
    }

    if (totalNumBlocks == 0)
    {
        mVitInput.reshape({totalNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW});
        return;
    }

    if (totalNumBlocks < mConfig.minNumBlocks || totalNumBlocks > mConfig.maxNumBlocks)
    {
        throw std::runtime_error("totalNumBlocks " + std::to_string(totalNumBlocks)
            + " exceeds the limitation, max = " + std::to_string(mConfig.maxNumBlocks)
            + ", min = " + std::to_string(mConfig.minNumBlocks) + " of VIT engine.");
    }

    // Calculate total image tokens for profiling (InternVL: each block generates 256 tokens)
    int64_t totalImageTokens = totalNumBlocks * 256;

    // Record performance data
    int64_t imageCount = std::accumulate(numImages.begin(), numImages.end(), int64_t(0));
    mMultimodalMetrics.recordRun(imageCount, totalImageTokens);

    mVitInput.reshape({totalNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW});
    mOutputEmbedding.reshape({totalImageTokens, mConfig.outHiddenSize});
}

void InternViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int64_t> const& numImages,
    std::vector<int64_t> const& imageTokenLengths, trt_edgellm::tokenizer::Tokenizer* tokenizer)
{
    if (numImages.size() != request.requests.size())
    {
        std::string errorMsg = "InternViTRunner::textPreprocess() numImages.size() != request.requests.size(), "
            + std::to_string(numImages.size()) + " != " + std::to_string(request.requests.size());
        LOG_ERROR("%s", errorMsg.c_str());
        throw std::runtime_error(errorMsg);
    }

    int64_t imageIndex = 0;
    // Image token id will start from vocabSize and increment for each image token position
    int32_t imageTokenId = mConfig.vocabSize;

    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        // Use the formatted complete request
        std::vector<int32_t> ids = tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);

        // replace vis tokens
        std::vector<int32_t> newIds;
        for (size_t j = 0; j < ids.size(); ++j)
        {
            if (ids[j] == mConfig.imageTokenId)
            {
                // Prepend <img> token
                newIds.push_back(mConfig.imgStartTokenId);

                int64_t numImageTokens = imageTokenLengths.at(imageIndex);
                for (int64_t k = 0; k < numImageTokens; ++k)
                {
                    newIds.push_back(imageTokenId);
                    ++imageTokenId;
                }

                // Append </img> token
                newIds.push_back(mConfig.imgEndTokenId);
                ++imageIndex;
            }
            else
            {
                newIds.push_back(ids[j]);
            }
        }
        batchInputIds.emplace_back(std::move(newIds));
    }
}

bool InternViTRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer* tokenizer,
    rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    std::vector<int64_t> imageTokenLengths;
    std::vector<int64_t> numImages;

    try
    {
        imagePreprocess(request, imageTokenLengths, numImages, true, stream);
        textPreprocess(request, batchedInputIds, numImages, imageTokenLengths, tokenizer);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("InternViTRunner::preprocess() failed: %s", e.what());
        return false;
    }

    return true;
}

bool InternViTRunner::infer(cudaStream_t stream)
{
    // Skip VIT inference if there are no images to process
    // Check if the first dimension (sequence length) is 0, indicating no images
    if (mVitInput.getShape()[0] == 0)
    {
        return true;
    }

    // Profile ViT inference with automatic cleanup
    {
        TIME_STAGE(metrics::StageNames::kMULTIMODAL_PROCESSING, stream);

        bool setEngineIOStatus{true};
        setEngineIOStatus &= mContext->setInputShape(binding_names::kVisualInput, mVitInput.getShape().getTRTDims());
        if (!setEngineIOStatus)
        {
            LOG_ERROR("InternViTRunner::infer(): Failed to bind engine input tensors.");
            return false;
        }

        bool enqueueStatus = mContext->enqueueV3(stream);
        if (!enqueueStatus)
        {
            LOG_ERROR("InternViTRunner::infer(): Failed to enqueue engine.");
            return false;
        }
    }

    return true;
}

} // namespace rt
} // namespace trt_edgellm