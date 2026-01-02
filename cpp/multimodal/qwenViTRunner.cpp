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

#include "qwenViTRunner.h"
#include "common/bindingNames.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "profiling/timer.h"
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
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
    , mLLMMaxBatchSize(llmMaxBatchSize)
    , mLLMMaxSequenceLength(llmMaxSequenceLength)
{
    if (!validateAndFillConfig(engineDir))
    {
        LOG_ERROR("QwenViTRunner::QwenViTRunner(): Failed to validate and fill config");
        throw std::runtime_error("QwenViTRunner::QwenViTRunner(): Failed to validate and fill config");
    }
    if (!allocateBuffer(stream))
    {
        LOG_ERROR("QwenViTRunner::QwenViTRunner(): Failed to allocate buffer");
        throw std::runtime_error("QwenViTRunner::QwenViTRunner(): Failed to allocate buffer");
    }
}

bool QwenViTRunner::validateAndFillConfig(std::string const& engineDir)
{
    Json jsonConfig;

    std::string configPath = engineDir + "/config.json";
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("QwenViTRunner::validateAndFillConfig(): Failed to open config file: %s", configPath.c_str());
        return false;
    }

    try
    {
        jsonConfig = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("QwenViTRunner::validateAndFillConfig(): Failed to parse config file with error: %s", e.what());
        return false;
    }

    std::string modelTypeStr = jsonConfig["model_type"].get<std::string>();
    mModelType = multimodal::stringToModelType(modelTypeStr);
    if (mModelType != multimodal::ModelType::QWEN2_5_VL && mModelType != multimodal::ModelType::QWEN2_VL
        && mModelType != multimodal::ModelType::QWEN3_VL)
    {
        LOG_ERROR("QwenViTRunner::validateAndFillConfig(): Invalid model type: %s", modelTypeStr.c_str());
        return false;
    }

    mConfig.visionStartTokenId = jsonConfig["vision_start_token_id"].get<int32_t>();
    mConfig.imageTokenId = jsonConfig["image_token_id"].get<int32_t>();
    mConfig.videoTokenId = jsonConfig["video_token_id"].get<int32_t>();

    auto const& subConfig
        = (mModelType == multimodal::ModelType::QWEN2_VL || mModelType == multimodal::ModelType::QWEN2_5_VL)
        ? jsonConfig
        : jsonConfig["text_config"];
    mConfig.vocabSize = subConfig["vocab_size"].get<int32_t>();
    mConfig.mropeTheta = subConfig["rope_theta"].get<float>();

    if (mModelType == multimodal::ModelType::QWEN2_5_VL)
    {
        mConfig.windowSize = jsonConfig["vision_config"]["window_size"].get<int64_t>();
    }
    else if (mModelType == multimodal::ModelType::QWEN3_VL)
    {
        auto visionConfig = jsonConfig["vision_config"];
        auto numPositionEmbeddings = visionConfig["num_position_embeddings"].get<int64_t>();
        mConfig.numGridPerSide = static_cast<int64_t>(std::sqrt(numPositionEmbeddings));
        mConfig.numDeepstackFeatures = visionConfig["deepstack_visual_indexes"].get<std::vector<int64_t>>().size();
    }

    auto builderConfig = jsonConfig["builder_config"];
    mConfig.minImageTokensPerImage = builderConfig["min_image_tokens"].get<int64_t>();
    mConfig.maxImageTokensPerImage = builderConfig["max_image_tokens_per_image"].get<int64_t>();
    if (mConfig.minImageTokensPerImage <= 0 || mConfig.maxImageTokensPerImage <= 0)
    {
        LOG_ERROR(
            "QwenViTRunner::validateAndFillConfig(): minImageTokensPerImage and maxImageTokensPerImage must be "
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
        LOG_ERROR("QwenViTRunner::validateAndFillConfig(): Failed to open preprocessor config file: %s",
            preprocessorConfigPath.c_str());
        return false;
    }
    try
    {
        preprocessorConfig = Json::parse(preprocessorConfigFileStream);
        preprocessorConfigFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("QwenViTRunner::validateAndFillConfig(): Failed to parse preprocessor config file with error: %s",
            e.what());
        return false;
    }

    mConfig.patchSize = preprocessorConfig["patch_size"].get<int64_t>();
    mConfig.temporalPatchSize = preprocessorConfig["temporal_patch_size"].get<int64_t>();
    mConfig.mergeSize = preprocessorConfig["merge_size"].get<int64_t>();
    mConfig.imageMean = preprocessorConfig["image_mean"].get<std::vector<float>>();
    mConfig.imageStd = preprocessorConfig["image_std"].get<std::vector<float>>();

    // Get config from engine shapes
    nvinfer1::Dims const inputShapeMax
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims const inputShapeMin
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMIN);
    mConfig.maxHW = inputShapeMax.d[0];
    mConfig.minHW = inputShapeMin.d[0];
    auto maxImageTokens = mConfig.maxHW / (mConfig.mergeSize * mConfig.mergeSize);
    mConfig.maxNumImages = maxImageTokens / mConfig.minImageTokensPerImage;
    mConfig.inputDim = mContext->getTensorShape(binding_names::kVisualInput).d[1];
    mConfig.vitPosEmbDim = mContext->getTensorShape(binding_names::kRotaryPosEmb).d[1];
    mConfig.outHiddenSize = mVisualEngine->getTensorShape(binding_names::kVisualOutput).d[1];

    return true;
}

bool QwenViTRunner::allocateBuffer(cudaStream_t stream)
{
    bool setTensorAddressStatus{true};
    mVitInput = rt::Tensor(
        {mConfig.maxHW, mConfig.inputDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "QwenViTRunner::mVitInput");
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kVisualInput, mVitInput.rawPointer());

    mAttentionMask = rt::Tensor({1, mConfig.maxHW, mConfig.maxHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "QwenViTRunner::mAttentionMask");
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kAttentionMask, mAttentionMask.rawPointer());

    mRotaryPosEmb = rt::Tensor({mConfig.maxHW, mConfig.vitPosEmbDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
        "QwenViTRunner::mRotaryPosEmb");
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kRotaryPosEmb, mRotaryPosEmb.rawPointer());

    // In Qwen-VL, VIT input mHW is always numImageTokens * spatial_merge_size ** 2.
    auto const maxImageTokens = mConfig.maxHW / (mConfig.mergeSize * mConfig.mergeSize);
    mOutputEmbedding = rt::Tensor({maxImageTokens, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "QwenViTRunner::mOutputEmbedding");
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kVisualOutput, mOutputEmbedding.rawPointer());

    if (mModelType == multimodal::ModelType::QWEN2_5_VL)
    {
        mWindowAttentionMask = rt::Tensor({1, mConfig.maxHW, mConfig.maxHW}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "QwenViTRunner::mWindowAttentionMask");
        setTensorAddressStatus
            &= mContext->setTensorAddress(binding_names::kWindowAttentionMask, mWindowAttentionMask.rawPointer());

        mWindowIndexHost = rt::Tensor(
            {maxImageTokens}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64, "QwenViTRunner::mWindowIndexHost");
        mWindowIndexDevice = rt::Tensor(
            {maxImageTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "QwenViTRunner::mWindowIndexDevice");
        setTensorAddressStatus
            &= mContext->setTensorAddress(binding_names::kWindowIndex, mWindowIndexDevice.rawPointer());

        mReverseWindowIndexHost = rt::Tensor({maxImageTokens}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64,
            "QwenViTRunner::mReverseWindowIndexHost");
        mReverseWindowIndexDevice = rt::Tensor({maxImageTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64,
            "QwenViTRunner::mReverseWindowIndexDevice");
        setTensorAddressStatus
            &= mContext->setTensorAddress(binding_names::kReverseWindowIndex, mReverseWindowIndexDevice.rawPointer());

        // Use maxImageTokens as a safe upper bound for cumulative window sequence lengths.
        mCuWindowSeqlensHost = rt::Tensor(
            {maxImageTokens}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64, "QwenViTRunner::mCuWindowSeqlensHost");
        mCuWindowSeqlensDevice = rt::Tensor({maxImageTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64,
            "QwenViTRunner::mCuWindowSeqlensDevice");
    }
    else if (mModelType == multimodal::ModelType::QWEN3_VL)
    {
        mFastPosEmbIdx = rt::Tensor(
            {4, mConfig.maxHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "QwenViTRunner::mFastPosEmbIdx");
        setTensorAddressStatus
            &= mContext->setTensorAddress(binding_names::kFastPosEmbIdx, mFastPosEmbIdx.rawPointer());

        mFastPosEmbWeight = rt::Tensor(
            {4, mConfig.maxHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "QwenViTRunner::mFastPosEmbWeight");
        setTensorAddressStatus
            &= mContext->setTensorAddress(binding_names::kFastPosEmbWeight, mFastPosEmbWeight.rawPointer());

        for (int64_t i = 0; i < mConfig.numDeepstackFeatures; ++i)
        {
            // Set tensor name to match the engine binding name.
            std::string const deepstackFeatureName = binding_names::formatDeepstackFeaturesName(i);
            mDeepstackFeatures.emplace_back(rt::Tensor({maxImageTokens, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kHALF, deepstackFeatureName));
            setTensorAddressStatus
                &= mContext->setTensorAddress(deepstackFeatureName.c_str(), mDeepstackFeatures.back().rawPointer());
        }
    }

    if (!setTensorAddressStatus)
    {
        LOG_ERROR("Failed to set tensor address to the engine");
        return false;
    }

    // Copy image mean and std to device to be used in normalizeImage
    auto channels = mConfig.imageMean.size();
    mImageMean = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "QwenViTRunner::mImageMean");
    mImageStd = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "QwenViTRunner::mImageStd");
    CUDA_CHECK(cudaMemcpyAsync(
        mImageMean.rawPointer(), mConfig.imageMean.data(), channels * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mImageStd.rawPointer(), mConfig.imageStd.data(), channels * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Pre-allocate temporary image buffers for preprocessing
    int64_t const maxImagePixels = mVitInput.getShape().volume();
    // Set max image size to 1xmaxImagePixelsxchannels, will reshape to actual image size in resizeImage
    rt::Tensor resizeBuffer(
        {1, maxImagePixels, channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "QwenViTRunner::resizeBuffer");
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

    // Pre-allocate tensors for cumulative sequence lengths.
    // The size of the tensor is maxNumImages + 1 because the first element is 0.
    mCuSeqlensDevice = rt::Tensor({mConfig.maxNumImages + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64,
        "QwenViTRunner::mCuSeqlensDevice");
    mCuSeqlensHost = rt::Tensor(
        {mConfig.maxNumImages + 1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64, "QwenViTRunner::mCuSeqlensHost");

    return true;
}

void QwenViTRunner::formatPatch(rt::imageUtils::ImageData const& image,
    std::vector<std::vector<int64_t>>& imageGridTHWs, std::vector<int64_t>& imageTokenLengths, int64_t* cuSeqlensData,
    int64_t& cuSeqlensSize, cudaStream_t stream)
{
    int64_t height = image.height;
    int64_t width = image.width;
    int64_t channels = image.channels;
    unsigned char* imageData = image.data(); // In hwc order

    if (height % (mConfig.patchSize * mConfig.mergeSize) != 0 || width % (mConfig.patchSize * mConfig.mergeSize) != 0)
    {
        throw std::runtime_error("Image height or width is not divisible by patchSize * mergeSize = "
            + std::to_string(mConfig.patchSize * mConfig.mergeSize) + " got height: " + std::to_string(height)
            + ", width: " + std::to_string(width));
    }

    std::vector<int64_t> curGrid{1, (height / mConfig.patchSize), (width / mConfig.patchSize)};
    imageGridTHWs.emplace_back(curGrid);
    int64_t curSeqLength = (height / mConfig.patchSize) * (width / mConfig.patchSize);
    int64_t prevCuSeqlen = cuSeqlensData[cuSeqlensSize - 1];
    if (prevCuSeqlen + curSeqLength > mConfig.maxHW || cuSeqlensSize > (mConfig.maxNumImages + 1))
    {
        throw std::runtime_error("cuSeqlens " + std::to_string(prevCuSeqlen + curSeqLength)
            + " exceeds the limitation, maxHW = " + std::to_string(mConfig.maxHW)
            + " or maxNumImages = " + std::to_string(mConfig.maxNumImages) + " of VIT engine.");
    }
    imageTokenLengths.emplace_back(curSeqLength / mConfig.mergeSize / mConfig.mergeSize);

    // Reshape pre-allocated temporary buffers to current image dimensions
    mImageDevice.reshape({mConfig.temporalPatchSize, height, width, channels});
    mNormalizedImageDevice.reshape({mConfig.temporalPatchSize, height, width, channels});

    // Copy image to device. Repeat for T = temporalPatchSize
    auto imageSize = height * width * channels;
    for (int64_t i = 0; i < mConfig.temporalPatchSize; ++i)
    {
        CUDA_CHECK(cudaMemcpyAsync(mImageDevice.rawPointer() + i * imageSize * sizeof(unsigned char), imageData,
            imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice, stream));
    }

    // Normalize image
    kernel::normalizeImage(mImageDevice, mImageMean, mImageStd, mNormalizedImageDevice, stream);

    // Transpose to patch
    kernel::transposeToPatchQwenViT(mNormalizedImageDevice, mVitInput, prevCuSeqlen * mConfig.inputDim,
        mConfig.temporalPatchSize, mConfig.patchSize, mConfig.mergeSize, stream);

    // Update sequence length
    cuSeqlensData[cuSeqlensSize++] = prevCuSeqlen + curSeqLength;
}

std::tuple<int64_t, int64_t> QwenViTRunner::getResizedImageSize(
    int64_t const height, int64_t const width, int64_t const maxRatio)
{
    // According to https://github.com/QwenLM/Qwen2-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
    int64_t const factor = mConfig.patchSize * mConfig.mergeSize;
    int64_t const minPixels = mConfig.minImageTokensPerImage * factor * factor;
    int64_t const maxPixels = mConfig.maxImageTokensPerImage * factor * factor;

    auto roundByFactor = [](int64_t value, int64_t factor) -> int64_t {
        return std::round(static_cast<double>(value) / factor) * factor;
    };
    auto floorByFactor = [](int64_t value, int64_t factor) -> int64_t {
        return std::floor(static_cast<double>(value) / factor) * factor;
    };
    auto ceilByFactor = [](int64_t value, int64_t factor) -> int64_t {
        return std::ceil(static_cast<double>(value) / factor) * factor;
    };

    if (std::max(height, width) / std::min(height, width) > maxRatio)
    {
        throw std::runtime_error("absolute aspect ratio must be smaller than " + std::to_string(maxRatio) + ", got "
            + std::to_string(std::max(height, width) / std::min(height, width)));
    }

    int64_t hBar = std::max(factor, roundByFactor(height, factor));
    int64_t wBar = std::max(factor, roundByFactor(width, factor));

    if (hBar * wBar > maxPixels)
    {
        double beta = std::sqrt(static_cast<double>(height * width) / maxPixels);
        hBar = floorByFactor(static_cast<int64_t>(height / beta), factor);
        wBar = floorByFactor(static_cast<int64_t>(width / beta), factor);
    }
    else if (hBar * wBar < minPixels)
    {
        double beta = std::sqrt(static_cast<double>(minPixels) / (height * width));
        hBar = ceilByFactor(static_cast<int64_t>(height * beta), factor);
        wBar = ceilByFactor(static_cast<int64_t>(width * beta), factor);
    }

    return {hBar, wBar};
}

void QwenViTRunner::imagePreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int64_t>>& imageGridTHWs, std::vector<int64_t>& imageTokenLengths,
    std::vector<int64_t>& numImages, bool doResize, cudaStream_t stream)
{
    // Use pre-allocated pinned host tensor for cumulative sequence lengths
    int64_t* cuSeqlensData = mCuSeqlensHost.dataPointer<int64_t>();
    cuSeqlensData[0] = 0;
    int64_t cuSeqlensSize = 1;

    for (auto const& req : request.requests)
    {
        int64_t numImage = 0;
        for (auto const& image : req.imageBuffers)
        {
            if (doResize)
            {
                auto [resizedHeight, resizedWidth] = getResizedImageSize(image.height, image.width);
                rt::imageUtils::resizeImage(image, mResizedImageHost, resizedWidth, resizedHeight);
                formatPatch(mResizedImageHost, imageGridTHWs, imageTokenLengths, cuSeqlensData, cuSeqlensSize, stream);
            }
            else
            {
                formatPatch(image, imageGridTHWs, imageTokenLengths, cuSeqlensData, cuSeqlensSize, stream);
            }
            ++numImage;
        }
        numImages.emplace_back(numImage);
    }

    int64_t totalSeqLength = cuSeqlensData[cuSeqlensSize - 1];
    if (totalSeqLength == 0)
    {
        mVitInput.reshape({totalSeqLength, mConfig.inputDim});
        return;
    }

    if (totalSeqLength < mConfig.minHW || totalSeqLength > mConfig.maxHW)
    {
        throw std::runtime_error("totalSeqLength " + std::to_string(totalSeqLength) + " exceeds the limitation, max = "
            + std::to_string(mConfig.maxHW) + ", min = " + std::to_string(mConfig.minHW) + " of VIT engine.");
    }

    // Reshape tensors
    int64_t totalImageTokens = totalSeqLength / (mConfig.mergeSize * mConfig.mergeSize);
    mVitInput.reshape({totalSeqLength, mConfig.inputDim});
    mAttentionMask.reshape({1, totalSeqLength, totalSeqLength});
    mRotaryPosEmb.reshape({totalSeqLength, mConfig.vitPosEmbDim});
    mOutputEmbedding.reshape({totalImageTokens, mConfig.outHiddenSize});

    // Record performance data
    int64_t imageCount = std::accumulate(numImages.begin(), numImages.end(), int64_t(0));
    mMultimodalMetrics.recordRun(imageCount, totalImageTokens);

    // Compute attention mask
    CUDA_CHECK(cudaMemcpyAsync(mCuSeqlensDevice.rawPointer(), mCuSeqlensHost.rawPointer(),
        cuSeqlensSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    kernel::initAttentionMaskQwenViT(mCuSeqlensDevice, mAttentionMask, stream);

    // Compute rotary position embeddings
    for (int64_t i = 0; i < imageGridTHWs.size(); ++i)
    {
        kernel::initRotaryPosEmbQwenViT(
            mRotaryPosEmb, imageGridTHWs[i], mConfig.mergeSize, cuSeqlensData[i], 10000.0f, 1.0f, stream);
    }

    // Compute additional inputs
    if (mModelType == multimodal::ModelType::QWEN2_5_VL)
    {
        mWindowAttentionMask.reshape({1, totalSeqLength, totalSeqLength});
        mWindowIndexHost.reshape({totalImageTokens});
        mWindowIndexDevice.reshape({totalImageTokens});
        mReverseWindowIndexHost.reshape({totalImageTokens});
        mReverseWindowIndexDevice.reshape({totalImageTokens});

        getWindowIndex(imageGridTHWs, totalSeqLength, stream);
    }
    else if (mModelType == multimodal::ModelType::QWEN3_VL)
    {
        mFastPosEmbIdx.reshape({4, totalSeqLength});
        mFastPosEmbWeight.reshape({4, totalSeqLength});

        for (int64_t i = 0; i < imageGridTHWs.size(); ++i)
        {
            kernel::initFastPosEmbedQwenViT(mFastPosEmbIdx, mFastPosEmbWeight, imageGridTHWs[i], mConfig.mergeSize,
                mConfig.numGridPerSide, cuSeqlensData[i], stream);
        }

        for (int64_t i = 0; i < mConfig.numDeepstackFeatures; ++i)
        {
            mDeepstackFeatures[i].reshape({totalImageTokens, mConfig.outHiddenSize});
        }
    }
}

void QwenViTRunner::getMRopePositionIds(
    std::vector<std::vector<int32_t>> const& batchInputIds, std::vector<std::vector<int64_t>> const& imageGridTHWs)
{
    // According to transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.get_rope_index
    // mropePositionIds: (bs, 3, maxPositionEmbeddings), 3 is for T, H, W
    int64_t* mropePositionIdsPtr = mMropePositionIdsHost.dataPointer<int64_t>();
    int64_t const maxPositionEmbeddings = mMropePositionIdsHost.getShape()[2];
    int64_t totalImageIdx = 0;
    int64_t batchOffset = 0;

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

            // Visual part
            int64_t T = imageGridTHWs[totalImageIdx][0];
            int64_t H = imageGridTHWs[totalImageIdx][1] / mConfig.mergeSize;
            int64_t W = imageGridTHWs[totalImageIdx][2] / mConfig.mergeSize;
            ++totalImageIdx;

            for (int64_t t = 0; t < T; ++t)
            {
                for (int64_t h = 0; h < H; ++h)
                {
                    for (int64_t w = 0; w < W; ++w)
                    {
                        int64_t idx = remainingStartPos + textLen + t * H * W + h * W + w;
                        mropePositionIdsPtr[batchOffset + 0 * maxPositionEmbeddings + idx] = t + textLen + startIdx;
                        mropePositionIdsPtr[batchOffset + 1 * maxPositionEmbeddings + idx] = h + textLen + startIdx;
                        mropePositionIdsPtr[batchOffset + 2 * maxPositionEmbeddings + idx] = w + textLen + startIdx;
                    }
                }
            }

            start = it + 1 + T * H * W;
            startIdx += std::max(T, std::max(H, W)) + textLen;
            remainingStartPos = start - inputIds.begin();
        }

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
    std::vector<std::vector<int64_t>> const& imageGridTHWs, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
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
    mMropePositionIdsHost.reshape({activeBatchSize, 3, maxPositionEmbeddings});
    mMropePositionIdsDevice.reshape({activeBatchSize, 3, maxPositionEmbeddings});
    getMRopePositionIds(batchInputIds, imageGridTHWs);
    CUDA_CHECK(cudaMemcpyAsync(mMropePositionIdsDevice.rawPointer(), mMropePositionIdsHost.rawPointer(),
        activeBatchSize * 3 * maxPositionEmbeddings * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    // Initialize mrope cosSinCacheDevice
    ropeRotaryCosSinDevice.reshape({activeBatchSize, maxPositionEmbeddings, rotaryDim});
    bool interleaved = mModelType == multimodal::ModelType::QWEN3_VL;
    kernel::initializeMRopeCosSin(ropeRotaryCosSinDevice.dataPointer<float>(),
        mMropePositionIdsDevice.dataPointer<int64_t>(), mConfig.mropeTheta, rotaryDim, maxPositionEmbeddings,
        activeBatchSize, interleaved, stream);
}

void QwenViTRunner::getWindowIndex(
    std::vector<std::vector<int64_t>> const& imageGridTHWs, int64_t const curHW, cudaStream_t stream)
{
    // Init windowIndex and cuWindowSeqlens
    int64_t* windowIndexPtr = mWindowIndexHost.dataPointer<int64_t>();
    int64_t const windowIndexSize = mWindowIndexHost.getShape()[0];
    int64_t const vitMergerWindowSize = mConfig.windowSize / mConfig.mergeSize / mConfig.patchSize;
    int64_t windowIndexPos = 0;
    int64_t windowIndexValue = 0;

    // Use pre-allocated pinned host tensor for cumulative window sequence lengths
    int64_t* cuWindowSeqlensData = mCuWindowSeqlensHost.dataPointer<int64_t>();
    cuWindowSeqlensData[0] = 0;
    int64_t cuWindowSeqlensSize = 1;

    for (auto const& grid : imageGridTHWs)
    {
        int64_t T = grid[0], H = grid[1], W = grid[2];
        int64_t llmGridH = H / mConfig.mergeSize;
        int64_t llmGridW = W / mConfig.mergeSize;
        int64_t numWindowsH = (llmGridH + vitMergerWindowSize - 1) / vitMergerWindowSize;
        int64_t numWindowsW = (llmGridW + vitMergerWindowSize - 1) / vitMergerWindowSize;

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
                            windowIndexPtr[windowIndexPos++] = idxH * llmGridW + idxW + windowIndexValue;
                            ++cnt;
                        }
                    }
                }

                int64_t prevCuWindowSeqlen = cuWindowSeqlensData[cuWindowSeqlensSize - 1];
                cuWindowSeqlensData[cuWindowSeqlensSize++]
                    = prevCuWindowSeqlen + cnt * mConfig.mergeSize * mConfig.mergeSize;
            }
        }

        windowIndexValue += T * llmGridH * llmGridW;
    }

    if (windowIndexPos * (mConfig.mergeSize * mConfig.mergeSize) != curHW)
    {
        throw std::runtime_error(
            "windowIndex size * (mergeSize * mergeSize) does not match curHW. Got windowIndex size: "
            + std::to_string(windowIndexPos) + ", curHW: " + std::to_string(curHW));
    }

    int64_t* reverseWindowIndexPtr = mReverseWindowIndexHost.dataPointer<int64_t>();
    std::iota(reverseWindowIndexPtr, reverseWindowIndexPtr + windowIndexSize, 0);
    std::sort(reverseWindowIndexPtr, reverseWindowIndexPtr + windowIndexSize,
        [windowIndexPtr](size_t left, size_t right) { return windowIndexPtr[left] < windowIndexPtr[right]; });

    CUDA_CHECK(cudaMemcpyAsync(mWindowIndexDevice.rawPointer(), mWindowIndexHost.rawPointer(),
        windowIndexSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mReverseWindowIndexDevice.rawPointer(), mReverseWindowIndexHost.rawPointer(),
        windowIndexSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    // Init window attention mask
    CUDA_CHECK(cudaMemcpyAsync(mCuWindowSeqlensDevice.rawPointer(), mCuWindowSeqlensHost.rawPointer(),
        cuWindowSeqlensSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    kernel::initAttentionMaskQwenViT(mCuWindowSeqlensDevice, mWindowAttentionMask, stream);
}

void QwenViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int64_t> const& numImages,
    std::vector<int64_t> const& imageTokenLengths, trt_edgellm::tokenizer::Tokenizer* tokenizer)
{
    if (numImages.size() != request.requests.size())
    {
        std::string errorMsg = "QwenViTRunner::textPreprocess() numImages.size() != request.requests.size(), "
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

        // insert image tokens
        std::vector<int32_t> newIds;
        for (size_t j = 0; j < ids.size(); ++j)
        {
            if (ids[j] == mConfig.imageTokenId || ids[j] == mConfig.videoTokenId)
            {
                int64_t numImageTokens = imageTokenLengths.at(imageIndex);
                for (int64_t k = 0; k < numImageTokens; ++k)
                {
                    newIds.push_back(imageTokenId);
                    ++imageTokenId;
                }
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

bool QwenViTRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer* tokenizer,
    rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    std::vector<std::vector<int64_t>> imageGridTHWs;
    std::vector<int64_t> imageTokenLengths;
    std::vector<int64_t> numImages;

    try
    {
        imagePreprocess(request, imageGridTHWs, imageTokenLengths, numImages, true, stream);
        textPreprocess(request, batchedInputIds, numImages, imageTokenLengths, tokenizer);
        generateMropeParams(batchedInputIds, imageGridTHWs, ropeRotaryCosSinDevice, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("QwenViTRunner::preprocess() failed: %s", e.what());
        return false;
    }

    return true;
}

bool QwenViTRunner::preprocessSystemPrompt(std::string const& systemPrompt, tokenizer::Tokenizer* tokenizer,
    rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    // systemPrompt is already formatted by tokenizer's applyChatTemplate
    std::vector<int32_t> ids = tokenizer->encode(systemPrompt);
    std::vector<std::vector<int32_t>> batchedInputIds;
    batchedInputIds.emplace_back(std::move(ids));
    std::vector<std::vector<int64_t>> imageGridTHWs;

    try
    {
        generateMropeParams(batchedInputIds, imageGridTHWs, ropeRotaryCosSinDevice, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("MRope parameter generation failed: %s", e.what());
        return false;
    }

    return true;
}

bool QwenViTRunner::infer(cudaStream_t stream)
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
        setEngineIOStatus
            &= mContext->setInputShape(binding_names::kAttentionMask, mAttentionMask.getShape().getTRTDims());
        setEngineIOStatus
            &= mContext->setInputShape(binding_names::kRotaryPosEmb, mRotaryPosEmb.getShape().getTRTDims());
        if (mModelType == multimodal::ModelType::QWEN2_5_VL)
        {
            setEngineIOStatus &= mContext->setInputShape(
                binding_names::kWindowAttentionMask, mWindowAttentionMask.getShape().getTRTDims());
            setEngineIOStatus
                &= mContext->setInputShape(binding_names::kWindowIndex, mWindowIndexDevice.getShape().getTRTDims());
            setEngineIOStatus &= mContext->setInputShape(
                binding_names::kReverseWindowIndex, mReverseWindowIndexDevice.getShape().getTRTDims());
        }
        else if (mModelType == multimodal::ModelType::QWEN3_VL)
        {
            setEngineIOStatus
                &= mContext->setInputShape(binding_names::kFastPosEmbIdx, mFastPosEmbIdx.getShape().getTRTDims());
            setEngineIOStatus
                &= mContext->setInputShape(binding_names::kFastPosEmbWeight, mFastPosEmbWeight.getShape().getTRTDims());
        }

        if (!setEngineIOStatus)
        {
            LOG_ERROR("QwenViTRunner::infer(): Failed to bind engine input tensors.");
            return false;
        }

        bool enqueueStatus = mContext->enqueueV3(stream);
        if (!enqueueStatus)
        {
            LOG_ERROR("QwenViTRunner::infer(): Failed to enqueue engine.");
            return false;
        }
    }

    return true;
}

rt::OptionalInputTensors QwenViTRunner::getExtraVisualFeatures()
{
    if (mModelType != multimodal::ModelType::QWEN3_VL)
    {
        return {};
    }

    // Build vector of references to individual tensors
    std::vector<std::reference_wrapper<rt::Tensor const>> refs;
    refs.reserve(mDeepstackFeatures.size());
    for (auto const& tensor : mDeepstackFeatures)
    {
        refs.emplace_back(std::cref(tensor));
    }
    return refs;
}

} // namespace rt
} // namespace trt_edgellm