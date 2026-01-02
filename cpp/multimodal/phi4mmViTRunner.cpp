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

#include "phi4mmViTRunner.h"
#include "common/bindingNames.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "modelTypes.h"
#include "multimodal/imageUtils.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{

Phi4MMViTRunner::Phi4MMViTRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner(engineDir, stream)
    , mEngineDir(engineDir)
{
    std::string const configPath = engineDir + "/config.json";
    if (!validateAndFillConfig(configPath))
    {
        LOG_ERROR("Failed to validate and fill config");
        throw std::runtime_error("Phi4MMViTRunner(): Failed to validate and fill config");
    }
    if (!allocateBuffer(stream))
    {
        LOG_ERROR("Failed to allocate buffer");
        throw std::runtime_error("Phi4MMViTRunner(): Failed to allocate buffer");
    }
}

bool Phi4MMViTRunner::validateAndFillConfig(std::string const& configPath)
{
    Json jsonConfig;

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
    if (mModelType != multimodal::ModelType::PHI4MM)
    {
        LOG_ERROR("Invalid model type: %s", modelTypeStr.c_str());
        return false;
    }

    mConfig.vocabSize = jsonConfig["vocab_size"].get<int32_t>();

    auto visionConfig = jsonConfig["embd_layer"]["image_embd_layer"];
    mConfig.blockImageSizeH = visionConfig["crop_size"].get<int32_t>();
    mConfig.blockImageSizeW = mConfig.blockImageSizeH;
    check::check(mConfig.blockImageSizeH % mConfig.blockDownsampleRatio == 0,
        "blockImageSizeH must be divisible by blockDownsampleRatio");
    mConfig.tokensPerSide = mConfig.blockImageSizeH / mConfig.blockDownsampleRatio;

    auto builderConfig = jsonConfig["builder_config"];
    // Min token per image is the same as min token per batch (minimum number of images = 1).
    mConfig.minImageTokensPerImage = builderConfig["min_image_tokens"].get<int32_t>();
    mConfig.maxImageTokensPerImage = builderConfig["max_image_tokens_per_image"].get<int32_t>();

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

bool Phi4MMViTRunner::allocateBuffer(cudaStream_t stream)
{
    bool setTensorAddressStatus{true};
    LOG_INFO(
        "mConfig.maxNumBlocks: %d, mConfig.numChannels: %d, mConfig.blockImageSizeH: "
        "%d, mConfig.blockImageSizeW: %d",
        mConfig.maxNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW);
    mVitInput
        = rt::Tensor({mConfig.maxNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "Phi4MMViTRunner::mVitInput");
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kVisualInput, mVitInput.rawPointer());
    LOG_INFO("mConfig.maxNumBlocks: %d, mConfig.outHiddenSize: %d", mConfig.maxNumBlocks, mConfig.outHiddenSize);

    // In Phi-4MM, each block generates 256 tokens, so output size is maxNumBlocks*256
    int64_t const vitImageTokens = mConfig.maxNumBlocks * 256;
    // Conservative capacity for postprocessed tokens:
    // - Sub-crops: insert one sub_GN per row → tokensPerSide * max_hBlocks ≤ tokensPerSide * maxNumBlocks
    // - Global: (tokensPerSide newline tokens (one per row in a tokensPerSide x tokensPerSide grid) + 1 glb_GN)
    //           * maxNumImages (≤ maxNumBlocks)
    int64_t const glbExtraPerImage = mConfig.tokensPerSide + 1;
    int64_t const conservativeExtra
        = mConfig.tokensPerSide * mConfig.maxNumBlocks + glbExtraPerImage * mConfig.maxNumBlocks;
    int64_t const totalCapacity = vitImageTokens + conservativeExtra;

    mEngineOutputEmbedding = rt::Tensor({vitImageTokens, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "Phi4MMViTRunner::mEngineOutputEmbedding");
    mOutputEmbedding = rt::Tensor({totalCapacity, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "Phi4MMViTRunner::mOutputEmbedding");
    setTensorAddressStatus
        &= mContext->setTensorAddress(binding_names::kVisualOutput, mEngineOutputEmbedding.rawPointer());
    if (!setTensorAddressStatus)
    {
        LOG_ERROR("Failed to set tensor address to the engine");
        return false;
    }

    // Copy image mean and std to device to be used in normalizeImage
    int64_t const channels = static_cast<int64_t>(mConfig.imageMean.size());
    check::check(channels == mConfig.numChannels, "channel of imageMean != numChannels");
    mImageMean
        = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "Phi4MMViTRunner::mImageMean");
    mImageStd = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "Phi4MMViTRunner::mImageStd");
    CUDA_CHECK(cudaMemcpyAsync(
        mImageMean.rawPointer(), mConfig.imageMean.data(), channels * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mImageStd.rawPointer(), mConfig.imageStd.data(), channels * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Pre-allocate temporary image buffers for preprocessing
    int64_t const maxImagePixels = mVitInput.getShape().volume();
    mImageDevice = rt::Tensor(
        {maxImagePixels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "Phi4MMViTRunner::mImageDevice");
    mNormalizedImageDevice = rt::Tensor(
        {maxImagePixels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "Phi4MMViTRunner::mNormalizedImageDevice");
    // Set max image size to 1xmaxImagePixelsxchannels, will reshape to actual image size in resizeImage
    rt::Tensor resizeBuffer({1, maxImagePixels, channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8,
        "Phi4MMViTRunner::resizeBuffer");
    mResizedImageHost = rt::imageUtils::ImageData(std::move(resizeBuffer));
    // Thumbnail image has fixed size: blockImageSizeH x blockImageSizeW x channels)
    rt::Tensor thumbnailBuffer({mConfig.blockImageSizeH, mConfig.blockImageSizeW, channels}, rt::DeviceType::kCPU,
        nvinfer1::DataType::kUINT8, "Phi4MMViTRunner::thumbnailBuffer");
    mThumbnailImageHost = rt::imageUtils::ImageData(std::move(thumbnailBuffer));

    // Pre-allocate temporary index tensors for Phi4MM postprocess (assign to members)
    mHBlocks = rt::Tensor(
        {mConfig.maxNumBlocks}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "Phi4MMViTRunner::mHBlocks");
    mWBlocks = rt::Tensor(
        {mConfig.maxNumBlocks}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "Phi4MMViTRunner::mWBlocks");
    mSrcGlbStart = rt::Tensor(
        {mConfig.maxNumBlocks}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "Phi4MMViTRunner::mSrcGlbStart");
    mSrcSubStart = rt::Tensor(
        {mConfig.maxNumBlocks}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "Phi4MMViTRunner::mSrcSubStart");
    mDstOutStart = rt::Tensor(
        {mConfig.maxNumBlocks}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "Phi4MMViTRunner::mDstOutStart");
    mSubOutLen = rt::Tensor(
        {mConfig.maxNumBlocks}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "Phi4MMViTRunner::mSubOutLen");

    // Initialize newline embeddings and load from safetensors file
    mSubGNProj = rt::Tensor(
        {mConfig.outHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "Phi4MMViTRunner::mSubGNProj");
    mGlbGNProj = rt::Tensor(
        {mConfig.outHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "Phi4MMViTRunner::mGlbGNProj");
    CUDA_CHECK(cudaMemsetAsync(mSubGNProj.rawPointer(), 0, mSubGNProj.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(mGlbGNProj.rawPointer(), 0, mGlbGNProj.getMemoryCapacity(), stream));
    // Try to load GN tensors from safetensors companion file
    try
    {
        std::filesystem::path gnPath = mEngineDir + "/phi4mm_gn_proj.safetensors";
        std::vector<rt::Tensor> tensors;
        rt::safetensors::loadSafetensors(gnPath, tensors, stream);
        for (auto& t : tensors)
        {
            if (t.getName() == "sub_GN")
            {
                check::check(t.getDataType() == nvinfer1::DataType::kHALF, "sub_GN must be FP16");
                check::check(
                    t.getShape().getNumDims() == 1 && t.getShape()[0] == mConfig.outHiddenSize, "sub_GN size mismatch");
                CUDA_CHECK(cudaMemcpyAsync(mSubGNProj.rawPointer(), t.rawPointer(),
                    mConfig.outHiddenSize * sizeof(half), cudaMemcpyDeviceToDevice, stream));
            }
            else if (t.getName() == "glb_GN")
            {
                check::check(t.getDataType() == nvinfer1::DataType::kHALF, "glb_GN must be FP16");
                check::check(
                    t.getShape().getNumDims() == 1 && t.getShape()[0] == mConfig.outHiddenSize, "glb_GN size mismatch");
                CUDA_CHECK(cudaMemcpyAsync(mGlbGNProj.rawPointer(), t.rawPointer(),
                    mConfig.outHiddenSize * sizeof(half), cudaMemcpyDeviceToDevice, stream));
            }
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Exception while loading GN tensors: %s", e.what());
        return false;
    }

    // Pre-allocate temporary index CPU buffers for Phi4MM postprocess
    mHBlocksHost = rt::Tensor(
        {mConfig.maxNumBlocks}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "Phi4MMViTRunner::mHBlocksHost");
    mWBlocksHost = rt::Tensor(
        {mConfig.maxNumBlocks}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "Phi4MMViTRunner::mWBlocksHost");
    mSrcGlbStartHost = rt::Tensor(
        {mConfig.maxNumBlocks}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64, "Phi4MMViTRunner::mSrcGlbStartHost");
    mSrcSubStartHost = rt::Tensor(
        {mConfig.maxNumBlocks}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64, "Phi4MMViTRunner::mSrcSubStartHost");
    mDstOutStartHost = rt::Tensor(
        {mConfig.maxNumBlocks}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64, "Phi4MMViTRunner::mDstOutStartHost");
    mSubOutLenHost = rt::Tensor(
        {mConfig.maxNumBlocks}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64, "Phi4MMViTRunner::mSubOutLenHost");

    return true;
}

void Phi4MMViTRunner::formatPatch(imageUtils::ImageData const& image, std::vector<int64_t>& imageTokenLengths,
    int64_t& numImages, int64_t& totalNumBlocks, bool isThumbnail, cudaStream_t stream)
{
    int height = image.height;
    int width = image.width;
    int channels = image.channels;
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

    // Compute token length according to Phi4MM hd_feature_transform:
    // - For thumbnail (global): reshape to tokensPerSide x tokensPerSide, add 1 newline per row, then add 1 glb_GN
    // later.
    //   So initial contribution on thumbnail push = tokensPerSide*(tokensPerSide+1) + 1 = 273.
    // - For sub-crops: grid (hBlocks x wBlocks), each block is tokensPerSide x tokensPerSide;
    //   final tokens = (tokensPerSide*hBlocks) * (tokensPerSide*wBlocks + 1).
    int64_t const hBlocks = height / mConfig.blockImageSizeH;
    int64_t const wBlocks = width / mConfig.blockImageSizeW;

    if (isThumbnail)
    {
        int64_t const glbLen
            = mConfig.tokensPerSide * (mConfig.tokensPerSide + 1) + 1; // glb rows with newlines + one glb_GN
        imageTokenLengths.push_back(glbLen);
        ++numImages;
    }
    else
    {
        int64_t const subLen = mConfig.tokensPerSide * hBlocks * (mConfig.tokensPerSide * wBlocks + 1);
        imageTokenLengths.back() += subLen;
    }

    // Copy image to device.
    mImageDevice.reshape({1, height, width, channels});
    CUDA_CHECK(cudaMemcpyAsync(mImageDevice.rawPointer(), imageData, height * width * channels * sizeof(unsigned char),
        cudaMemcpyHostToDevice, stream));

    // Normalize image
    mNormalizedImageDevice.reshape({1, height, width, channels});
    kernel::normalizeImage(mImageDevice, mImageMean, mImageStd, mNormalizedImageDevice, stream);

    // Transpose to patch
    int64_t offset = totalNumBlocks * mConfig.numChannels * mConfig.blockImageSizeH * mConfig.blockImageSizeW;
    kernel::transposeToPatchInternVLPhi4MM(mNormalizedImageDevice, mVitInput, offset, stream);

    // Update numBlocks
    totalNumBlocks += curNumBlocks;
}

void Phi4MMViTRunner::imagePreprocess(rt::LLMGenerationRequest const& request, std::vector<int64_t>& imageTokenLengths,
    std::vector<int64_t>& numImages, std::vector<std::vector<std::vector<int64_t>>>& imagesBlockGridHW, bool doResize,
    cudaStream_t stream)
{
    int64_t totalNumBlocks = 0;
    int64_t totalHBlocks = 0;

    for (auto const& req : request.requests)
    {
        int64_t numImage = 0;
        std::vector<std::vector<int64_t>> blockGridHWPerBatch;
        for (auto const& image : req.imageBuffers)
        {
            // Add thumbnail image by default
            imageUtils::resizeImage(image, mThumbnailImageHost, mConfig.blockImageSizeW, mConfig.blockImageSizeH);
            formatPatch(mThumbnailImageHost, imageTokenLengths, numImage, totalNumBlocks, true, stream);

            if (doResize)
            {
                auto [resizedHeight, resizedWidth] = imageUtils::computeBestBlockGridForResize(image.height,
                    image.width, mConfig.minImageTokensPerImage, mConfig.maxImageTokensPerImage,
                    mConfig.blockImageSizeH, mConfig.blockImageSizeW);
                imageUtils::resizeImage(image, mResizedImageHost, resizedWidth, resizedHeight);
                blockGridHWPerBatch.push_back(
                    {resizedHeight / mConfig.blockImageSizeH, resizedWidth / mConfig.blockImageSizeW});
                formatPatch(mResizedImageHost, imageTokenLengths, numImage, totalNumBlocks, false, stream);
            }
            else
            {
                blockGridHWPerBatch.push_back(
                    {image.height / mConfig.blockImageSizeH, image.width / mConfig.blockImageSizeW});
                formatPatch(image, imageTokenLengths, numImage, totalNumBlocks, false, stream);
            }
        }
        numImages.emplace_back(numImage);
        for (auto hw : blockGridHWPerBatch)
        {
            totalHBlocks += hw[0];
        }
        imagesBlockGridHW.emplace_back(std::move(blockGridHWPerBatch));
    }

    if (totalNumBlocks == 0)
    {
        check::check(
            mVitInput.reshape({totalNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW}),
            "mVitInput.reshape failed for empty input");
        return;
    }

    if (totalNumBlocks < mConfig.minNumBlocks || totalNumBlocks > mConfig.maxNumBlocks)
    {
        throw std::runtime_error("totalNumBlocks " + std::to_string(totalNumBlocks)
            + " exceeds the limitation, max = " + std::to_string(mConfig.maxNumBlocks)
            + ", min = " + std::to_string(mConfig.minNumBlocks) + " of VIT engine.");
    }

    // Calculate total image tokens for profiling (Phi4MM: each block generates 256 tokens)
    int64_t totalImageTokens = totalNumBlocks * 256;
    // Each image adds (tokensPerSide * hBlocks) for sub_GN separators + (tokensPerSide + 1) for glb segment
    int64_t const imageCount = std::accumulate(numImages.begin(), numImages.end(), static_cast<int64_t>(0));
    int64_t const glbExtraPerImage = mConfig.tokensPerSide + 1;
    int64_t const conservativeExtra = mConfig.tokensPerSide * totalHBlocks + glbExtraPerImage * imageCount;
    int64_t const totalOutTokens = totalImageTokens + conservativeExtra;

    // Record performance data
    mMultimodalMetrics.recordRun(imageCount, totalImageTokens);

    check::check(
        mVitInput.reshape({totalNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW}),
        "mVitInput.reshape failed");
    check::check(mEngineOutputEmbedding.reshape({totalImageTokens, mConfig.outHiddenSize}),
        "mEngineOutputEmbedding.reshape failed");
    check::check(mOutputEmbedding.reshape({totalOutTokens, mConfig.outHiddenSize}), "mOutputEmbedding.reshape failed");
}

void Phi4MMViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, std::vector<int64_t> const& numImages,
    std::vector<int64_t> const& imageTokenLengths, tokenizer::Tokenizer* tokenizer)
{
    if (numImages.size() != request.requests.size())
    {
        std::string errorMsg = "Phi4MMViTRunner::textPreprocess() numImages.size() != request.requests.size(), "
            + std::to_string(numImages.size()) + " != " + std::to_string(request.requests.size());
        LOG_ERROR("%s", errorMsg.c_str());
        throw std::runtime_error(errorMsg);
    }

    int32_t imageTokenId = mConfig.vocabSize;

    int imageIndex = 0;

    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        // Use the formatted complete request
        std::vector<int32_t> ids = tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);

        // Replace image placeholder tokens with sequential image token IDs
        std::vector<int32_t> newIds;
        for (size_t j = 0; j < ids.size(); ++j)
        {
            if (ids[j] == mConfig.imageTokenId)
            {
                // Expand to actual number of image tokens
                int64_t numImageTokens = imageTokenLengths.at(imageIndex);
                for (int k = 0; k < numImageTokens; ++k)
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
        batchedInputIds.emplace_back(std::move(newIds));
    }
}

bool Phi4MMViTRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer* tokenizer,
    rt::Tensor& /*ropeRotaryCosSinDevice*/, cudaStream_t stream)
{
    std::vector<int64_t> imageTokenLengths;
    std::vector<int64_t> numImages;
    mImagesBlockGridHW.clear();
    try
    {
        imagePreprocess(request, imageTokenLengths, numImages, mImagesBlockGridHW, true, stream);
        textPreprocess(request, batchedInputIds, numImages, imageTokenLengths, tokenizer);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Preprocess failed: %s", e.what());
        return false;
    }

    return true;
}

bool Phi4MMViTRunner::infer(cudaStream_t stream)
{
    // Skip VIT inference if there are no images to process
    if (mVitInput.getShape()[0] == 0)
    {
        return true;
    }
    {
        TIME_STAGE(metrics::StageNames::kMULTIMODAL_PROCESSING, stream);

        bool setEngineIOStatus{true};
        setEngineIOStatus &= mContext->setInputShape(binding_names::kVisualInput, mVitInput.getShape().getTRTDims());

        if (!setEngineIOStatus)
        {
            LOG_ERROR("Failed to bind engine input tensors.");
            return false;
        }

        bool const enqueueStatus = mContext->enqueueV3(stream);

        // Transform raw ViT tokens on GPU in one batched kernel (Phi4MM HD transform)
        // Flatten all images across batches
        int64_t inStartTok = 0;
        int64_t outStartTok = 0;
        int32_t blockIndex = 0;
        int32_t* hBlocksHostPtr = mHBlocksHost.dataPointer<int32_t>();
        int32_t* wBlocksHostPtr = mWBlocksHost.dataPointer<int32_t>();
        int64_t* srcGlbStartHostPtr = mSrcGlbStartHost.dataPointer<int64_t>();
        int64_t* srcSubStartHostPtr = mSrcSubStartHost.dataPointer<int64_t>();
        int64_t* dstOutStartHostPtr = mDstOutStartHost.dataPointer<int64_t>();
        int64_t* subOutLenHostPtr = mSubOutLenHost.dataPointer<int64_t>();
        for (auto const& blockGridHWPerBatch : mImagesBlockGridHW)
        {
            for (auto const& blockGridHWPerImage : blockGridHWPerBatch)
            {
                int32_t const hb = static_cast<int32_t>(blockGridHWPerImage[0]);
                int32_t const wb = static_cast<int32_t>(blockGridHWPerImage[1]);
                hBlocksHostPtr[blockIndex] = hb;
                wBlocksHostPtr[blockIndex] = wb;
                srcGlbStartHostPtr[blockIndex] = inStartTok;
                srcSubStartHostPtr[blockIndex] = inStartTok + 256;
                int64_t const subOutTokens = 256LL * hb * wb + 16LL * hb;
                subOutLenHostPtr[blockIndex] = subOutTokens;
                dstOutStartHostPtr[blockIndex] = outStartTok;
                outStartTok += subOutTokens + 1 + 16LL * (16 + 1); // sub + glb_GN + glb
                inStartTok += (1LL + static_cast<int64_t>(hb) * wb) * 256LL;
                blockIndex++;
            }
        }
        int32_t const numImagesTotal = blockIndex;
        int64_t const totalOutTokens = outStartTok;
        // Create device index tensors
        CUDA_CHECK(cudaMemcpyAsync(
            mHBlocks.rawPointer(), hBlocksHostPtr, numImagesTotal * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            mWBlocks.rawPointer(), wBlocksHostPtr, numImagesTotal * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(mSrcGlbStart.rawPointer(), srcGlbStartHostPtr, numImagesTotal * sizeof(int64_t),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(mSrcSubStart.rawPointer(), srcSubStartHostPtr, numImagesTotal * sizeof(int64_t),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(mDstOutStart.rawPointer(), dstOutStartHostPtr, numImagesTotal * sizeof(int64_t),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(mSubOutLen.rawPointer(), subOutLenHostPtr, numImagesTotal * sizeof(int64_t),
            cudaMemcpyHostToDevice, stream));
        // Launch single kernel
        int32_t const hidden = static_cast<int32_t>(mEngineOutputEmbedding.getShape()[1]);
        kernel::Phi4MMIndex indices{mHBlocks.dataPointer<int32_t>(), mWBlocks.dataPointer<int32_t>(),
            mSrcGlbStart.dataPointer<int64_t>(), mSrcSubStart.dataPointer<int64_t>(),
            mDstOutStart.dataPointer<int64_t>(), mSubOutLen.dataPointer<int64_t>(), numImagesTotal, hidden,
            totalOutTokens};
        kernel::Phi4MMGN gn{mSubGNProj.dataPointer<half>(), mGlbGNProj.dataPointer<half>()};
        kernel::phi4mmPostprocessVisionTokens(
            mEngineOutputEmbedding, mOutputEmbedding, indices, gn, totalOutTokens, stream);

        if (!enqueueStatus)
        {
            LOG_ERROR("Failed to enqueue engine.");
            return false;
        }
    }
    return true;
}

} // namespace rt
} // namespace trt_edgellm
