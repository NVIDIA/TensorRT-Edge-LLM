/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "multimodal/vitRunner.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "multimodal/modelTypes.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{

VitRunner::VitRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner(engineDir, stream)
{
    std::string const configPath = engineDir + "/config.json";
    if (!validateAndFillConfig(configPath))
    {
        LOG_ERROR("VitRunner: failed to validate and fill config");
        throw std::runtime_error("VitRunner: failed to validate and fill config");
    }
    if (!allocateBuffer(stream))
    {
        LOG_ERROR("VitRunner: failed to allocate buffer");
        throw std::runtime_error("VitRunner: failed to allocate buffer");
    }
}

bool VitRunner::validateAndFillConfig(std::string const& configPath)
{
    Json jsonConfig;

    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("VitRunner: failed to open config file: %s", configPath.c_str());
        return false;
    }

    try
    {
        jsonConfig = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("VitRunner: failed to parse config file: %s", e.what());
        return false;
    }

    std::string const modelTypeStr = jsonConfig["model_type"].get<std::string>();
    mModelType = multimodal::stringToModelType(modelTypeStr);
    if (mModelType != multimodal::ModelType::VIT)
    {
        LOG_ERROR("VitRunner: invalid model type: %s", modelTypeStr.c_str());
        return false;
    }

    mConfig.vocabSize = jsonConfig["vocab_size"].get<int32_t>();
    mConfig.imageTokenId = jsonConfig["image_token_id"].get<int32_t>();

    auto const builderConfig = jsonConfig["builder_config"];
    mConfig.seqLenPerImage = builderConfig["seq_len"].get<int32_t>();
    if (builderConfig.contains("image_mean"))
    {
        for (size_t i = 0; i < mConfig.imageMean.size(); ++i)
        {
            mConfig.imageMean[i] = builderConfig["image_mean"].at(i).get<float>();
        }
    }
    if (builderConfig.contains("image_std"))
    {
        for (size_t i = 0; i < mConfig.imageStd.size(); ++i)
        {
            mConfig.imageStd[i] = builderConfig["image_std"].at(i).get<float>();
        }
    }

    // Get config from engine shapes ([batch, H, W, C] HWC input)
    nvinfer1::Dims const inputShapeMax
        = mVisualEngine->getProfileShape(binding_names::kVitInput, 0, nvinfer1::OptProfileSelector::kMAX);
    mConfig.batchSize = inputShapeMax.d[0];
    mConfig.imageHeight = inputShapeMax.d[1];
    mConfig.imageWidth = inputShapeMax.d[2];
    mConfig.numChannels = inputShapeMax.d[3];

    nvinfer1::Dims const outputShape = mVisualEngine->getTensorShape(binding_names::kVitOutput);
    mConfig.numImageTokens = outputShape.d[0];
    mConfig.outHiddenSize = outputShape.d[1];

    return true;
}

bool VitRunner::allocateBuffer(cudaStream_t stream)
{
    bool setTensorAddressStatus{true};

    mVitInput = rt::Tensor({mConfig.batchSize, mConfig.imageHeight, mConfig.imageWidth, mConfig.numChannels},
        rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "VitRunner::mVitInput");
    mOutputEmbedding = rt::Tensor({mConfig.numImageTokens, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "VitRunner::mOutputEmbedding");

    setTensorAddressStatus &= mVisualContext->setTensorAddress(binding_names::kVitInput, mVitInput.rawPointer());
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kVitOutput, mOutputEmbedding.rawPointer());
    if (!setTensorAddressStatus)
    {
        LOG_ERROR("VitRunner: failed to set tensor addresses on the engine");
        return false;
    }

    int64_t const channels = static_cast<int64_t>(mConfig.numChannels);
    mImageMean = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "VitRunner::mImageMean");
    mImageStd = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "VitRunner::mImageStd");
    CUDA_CHECK(cudaMemcpyAsync(
        mImageMean.rawPointer(), mConfig.imageMean.data(), channels * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mImageStd.rawPointer(), mConfig.imageStd.data(), channels * sizeof(float), cudaMemcpyHostToDevice, stream));

    int64_t const maxImagePixels = mConfig.imageHeight * mConfig.imageWidth * mConfig.numChannels;
    mImageDevice = rt::Tensor({maxImagePixels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "VitRunner::mImageDevice");
    mNormalizedImageDevice = rt::Tensor(
        {maxImagePixels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "VitRunner::mNormalizedImageDevice");

    rt::Tensor resizeBuffer({1, maxImagePixels, channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8,
        "VitRunner::resizeBuffer");
    mResizedImageHost = rt::imageUtils::ImageData(std::move(resizeBuffer));

    return true;
}

void VitRunner::formatImage(rt::imageUtils::ImageData const& image, int64_t batchIndex, cudaStream_t stream)
{
    int64_t height = image.height;
    int64_t width = image.width;
    int64_t channels = image.channels;
    if (channels != mConfig.numChannels)
    {
        throw std::runtime_error("VitRunner::formatImage(): image channels mismatch, got " + std::to_string(channels)
            + ", expected " + std::to_string(mConfig.numChannels));
    }

    rt::imageUtils::ImageData const* imageToUse = &image;
    if (height != mConfig.imageHeight || width != mConfig.imageWidth)
    {
        rt::imageUtils::resizeImage(image, mResizedImageHost, mConfig.imageWidth, mConfig.imageHeight,
            rt::imageUtils::InterpolationMode::kLINEAR);
        imageToUse = &mResizedImageHost;
        height = imageToUse->height;
        width = imageToUse->width;
    }

    check::check(mImageDevice.reshape({1, height, width, channels}), "VitRunner::formatImage(): mImageDevice reshape failed");
    check::check(mNormalizedImageDevice.reshape({1, height, width, channels}),
        "VitRunner::formatImage(): mNormalizedImageDevice reshape failed");

    CUDA_CHECK(cudaMemcpyAsync(mImageDevice.rawPointer(), imageToUse->data(), height * width * channels,
        cudaMemcpyHostToDevice, stream));
    kernel::normalizeImage(mImageDevice, mImageMean, mImageStd, mNormalizedImageDevice, stream);

    int64_t const imageBytes = height * width * channels * static_cast<int64_t>(sizeof(half));
    int64_t const batchOffset = batchIndex * imageBytes;
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(mVitInput.rawPointer()) + batchOffset,
        mNormalizedImageDevice.rawPointer(), static_cast<size_t>(imageBytes), cudaMemcpyDeviceToDevice, stream));
}

void VitRunner::imagePreprocess(rt::LLMGenerationRequest const& request, cudaStream_t stream)
{
    int64_t batchIndex = 0;
    for (auto const& req : request.requests)
    {
        for (auto const& image : req.imageBuffers)
        {
            if (batchIndex >= mConfig.batchSize)
            {
                throw std::runtime_error("VitRunner::imagePreprocess(): too many images for fixed engine batch size "
                    + std::to_string(mConfig.batchSize));
            }
            formatImage(image, batchIndex, stream);
            ++batchIndex;
        }
    }

    if (batchIndex == 0)
    {
        return;
    }

    if (batchIndex != mConfig.batchSize)
    {
        throw std::runtime_error("VitRunner::imagePreprocess(): expected " + std::to_string(mConfig.batchSize)
            + " images, got " + std::to_string(batchIndex));
    }

    mMultimodalMetrics.recordRun(batchIndex, mConfig.numImageTokens);
}

void VitRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer)
{
    int32_t nextImageTokenId = mConfig.vocabSize;
    int64_t numExpandedImageTokens{0};

    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        std::vector<int32_t> ids = tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);
        check::check(!ids.empty(), "VitRunner::textPreprocess(): failed to encode text");

        std::vector<int32_t> newIds;
        newIds.reserve(ids.size());
        for (int32_t const id : ids)
        {
            if (id == mConfig.imageTokenId)
            {
                for (int32_t k = 0; k < mConfig.seqLenPerImage; ++k)
                {
                    newIds.push_back(nextImageTokenId);
                    ++nextImageTokenId;
                    ++numExpandedImageTokens;
                }
            }
            else
            {
                newIds.push_back(id);
            }
        }
        batchedInputIds.emplace_back(std::move(newIds));
    }

    check::check(numExpandedImageTokens == mConfig.numImageTokens,
        "VitRunner::textPreprocess(): expanded image token count must match engine output rows");
}

bool VitRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
    rt::OptionalOutputTensor /*mropeCosSinOut*/, cudaStream_t stream, bool imageOnly) noexcept
{
    try
    {
        imagePreprocess(request, stream);
        // VitRunner uses standard RoPE, so mropeCosSinOut is unused. imageOnly skips text tokenization
        // for benchmarking paths that only need the visual engine inputs.
        if (!imageOnly)
        {
            textPreprocess(request, batchedInputIds, tokenizer);
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("VitRunner::preprocess failed: %s", e.what());
        return false;
    }

    return true;
}

bool VitRunner::infer(cudaStream_t stream)
{
    if (mVitInput.getShape().volume() == 0)
    {
        return true;
    }

    TIME_STAGE(metrics::StageNames::kMULTIMODAL_PROCESSING, stream);

    if (!mVisualContext->setInputShape(binding_names::kVitInput, mVitInput.getShape().getTRTDims()))
    {
        LOG_ERROR("VitRunner::infer(): failed to set engine input shape");
        return false;
    }

    if (!mVisualContext->enqueueV3(stream))
    {
        LOG_ERROR("VitRunner::infer(): failed to enqueue engine");
        return false;
    }

    return true;
}

} // namespace rt
} // namespace trt_edgellm
