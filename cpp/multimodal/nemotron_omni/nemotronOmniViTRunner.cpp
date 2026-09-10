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

#include "multimodal/nemotron_omni/nemotronOmniViTRunner.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#ifdef CUTE_DSL_GEMM_ENABLED
#include "kernels/talkerMLPKernels/cuteDslGemmRunner.h"
#endif
#include "multimodal/common/imageUtils.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{

NemotronOmniViTRunner::NemotronOmniViTRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner(engineDir, stream)
    , mEngineDir(engineDir)
{
    bool const configValid = validateAndFillConfig(engineDir);
    ELLM_CHECK(configValid, "NemotronOmniViTRunner: Failed to validate and fill config");
#ifdef CUTE_DSL_GEMM_ENABLED
    ELLM_CHECK(CuteDslGemmRunner::loadKernelModule(),
        "NemotronOmniViTRunner: CuTe DSL GEMM kernels unavailable on this device — the patch embedder GEMM "
        "cannot run");
#else
    ELLM_CHECK(false,
        "NemotronOmniViTRunner: built without CuTe DSL GEMM (rebuild with -DENABLE_CUTE_DSL=gemm) — the patch "
        "embedder GEMM cannot run");
#endif
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

    if (jsonConfig.contains("patch_size"))
    {
        mConfig.patchSize = jsonConfig["patch_size"].get<int64_t>();
    }
    if (jsonConfig.contains("downsample_ratio"))
    {
        mConfig.downsampleScale
            = static_cast<int64_t>(std::llround(1.0 / jsonConfig["downsample_ratio"].get<double>()));
    }
    if (jsonConfig.contains("video_temporal_patch_size"))
    {
        mConfig.videoTemporalPatchSize = jsonConfig["video_temporal_patch_size"].get<int64_t>();
    }
    if (jsonConfig.contains("video_pruning_rate"))
    {
        mConfig.videoPruningRate = jsonConfig["video_pruning_rate"].get<float>();
    }
    if (jsonConfig.contains("video_target_num_patches"))
    {
        mConfig.videoTargetNumPatches = jsonConfig["video_target_num_patches"].get<int64_t>();
    }
    if (jsonConfig.contains("video_maintain_aspect_ratio"))
    {
        mConfig.videoMaintainAspectRatio = jsonConfig["video_maintain_aspect_ratio"].get<bool>();
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

    // The engine consumes patch embeddings [numBlocks, numPatches, vitHidden]; block and patch
    // budgets come from the optimization profile, output hidden from the (static) output dim.
    nvinfer1::Dims const inputShapeMax
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims const inputShapeMin
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMIN);
    if (inputShapeMax.nbDims != 3)
    {
        LOG_ERROR(
            "Nemotron-Omni visual engine input must be 3D patch embeddings — rebuild the engine from a "
            "re-exported ONNX (the runtime/model boundary moved: the embedder GEMM now runs in the runtime)");
        return false;
    }
    mConfig.maxNumBlocks = inputShapeMax.d[0];
    mConfig.minNumBlocks = inputShapeMin.d[0];
    mConfig.maxNumPatches = inputShapeMax.d[1];
    mConfig.vitHiddenSize = inputShapeMax.d[2];
    mConfig.outHiddenSize = mVisualEngine->getTensorShape(binding_names::kVisualOutput).d[2];

    if (mConfig.patchSize <= 0 || mConfig.downsampleScale <= 0 || mConfig.videoTemporalPatchSize <= 0
        || mConfig.videoTargetNumPatches <= 0)
    {
        LOG_ERROR(
            "Nemotron-Omni visual config: patch_size, downsample_ratio, video_temporal_patch_size, and "
            "video_target_num_patches must be positive");
        return false;
    }
    if (mConfig.videoTargetNumPatches > mConfig.maxNumPatches)
    {
        LOG_ERROR(
            "Nemotron-Omni visual config: video_target_num_patches (%ld) exceeds the engine patch "
            "budget (%ld)",
            mConfig.videoTargetNumPatches, mConfig.maxNumPatches);
        return false;
    }
    if (mConfig.videoPruningRate < 0.0F || mConfig.videoPruningRate > 1.0F)
    {
        LOG_ERROR("Nemotron-Omni visual config: video_pruning_rate must be within [0, 1]");
        return false;
    }

    int64_t const scale = mConfig.downsampleScale;
    mConfig.tokensPerBlock = mConfig.maxNumPatches / (scale * scale);

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
        "NemotronOmniViTRunner: image=%ldx%ld, blocks=[%ld,%ld], tokensPerBlock=%ld, vitHidden=%ld, "
        "outHidden=%ld, imgContextTokenId=%d, videoT=%ld, videoPruningRate=%.2f",
        mConfig.blockImageSizeH, mConfig.blockImageSizeW, mConfig.minNumBlocks, mConfig.maxNumBlocks,
        mConfig.tokensPerBlock, mConfig.vitHiddenSize, mConfig.outHiddenSize, mConfig.imgContextTokenId,
        mConfig.videoTemporalPatchSize, static_cast<double>(mConfig.videoPruningRate));

    return true;
}

bool NemotronOmniViTRunner::loadVEmbedder(cudaStream_t stream)
{
    std::filesystem::path const weightPath = std::filesystem::path(mEngineDir) / "nemotron_omni_embedder.safetensors";
    if (!std::filesystem::exists(weightPath))
    {
        LOG_ERROR(
            "Embedder weight file not found: %s — re-export the visual model (the patch embedder GEMM runs in "
            "the runtime and its weights ship next to the engine)",
            weightPath.string().c_str());
        return false;
    }

    int64_t const patchDim = 3 * mConfig.patchSize * mConfig.patchSize;
    int64_t const videoPatchDim = mConfig.videoTemporalPatchSize * patchDim;

    try
    {
        std::vector<rt::Tensor> tensors;
        rt::safetensors::loadSafetensors(weightPath, tensors, stream);
        rt::Tensor posEmbedRaw;
        for (auto& t : tensors)
        {
            if (t.getName() == "embedder.weight")
            {
                check::check(t.getDataType() == nvinfer1::DataType::kHALF, "embedder.weight must be FP16");
                check::check(t.getShape().getNumDims() == 2 && t.getShape()[0] == mConfig.vitHiddenSize
                        && t.getShape()[1] == patchDim,
                    "embedder.weight shape mismatch");
                mImageEmbedderWeight = std::move(t);
            }
            else if (t.getName() == "video_embedder.weight")
            {
                check::check(t.getDataType() == nvinfer1::DataType::kHALF, "video_embedder.weight must be FP16");
                check::check(t.getShape().getNumDims() == 2 && t.getShape()[0] == mConfig.vitHiddenSize
                        && t.getShape()[1] == videoPatchDim,
                    "video_embedder.weight shape mismatch");
                mVideoEmbedderWeight = std::move(t);
                mHasVideoEmbedder = true;
            }
            else if (t.getName() == "pos_embed")
            {
                check::check(t.getDataType() == nvinfer1::DataType::kHALF, "pos_embed must be FP16");
                check::check(t.getShape().getNumDims() == 3 && t.getShape()[2] == mConfig.vitHiddenSize,
                    "pos_embed shape mismatch");
                posEmbedRaw = std::move(t);
            }
        }
        check::check(mImageEmbedderWeight.getShape().volume() > 0, "weight file missing embedder.weight");
        check::check(posEmbedRaw.getShape().volume() > 0, "weight file missing pos_embed");

        // Keep the raw pos_embed on the host (FP32) for per-grid bilinear interpolation.
        int64_t const storedPatches = posEmbedRaw.getShape()[1];
        mPosEmbedRawSide = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(storedPatches))));
        check::check(mPosEmbedRawSide * mPosEmbedRawSide == storedPatches, "pos_embed grid is not square");
        std::vector<half> posHalf(static_cast<size_t>(storedPatches * mConfig.vitHiddenSize));
        CUDA_CHECK(cudaMemcpyAsync(
            posHalf.data(), posEmbedRaw.rawPointer(), posHalf.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        mPosEmbedRawHost.resize(posHalf.size());
        for (size_t i = 0; i < posHalf.size(); ++i)
        {
            mPosEmbedRawHost[i] = __half2float(posHalf[i]);
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Exception while loading embedder weight file: %s", e.what());
        return false;
    }
    return true;
}

bool NemotronOmniViTRunner::allocateBuffer(cudaStream_t stream)
{
    if (!loadVEmbedder(stream))
    {
        return false;
    }

    bool setTensorAddressStatus{true};

    int64_t const scale = mConfig.downsampleScale;
    mVitInput = rt::Tensor({mConfig.maxNumBlocks, mConfig.maxNumPatches, mConfig.vitHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "NemotronOmniViTRunner::mVitInput");
    setTensorAddressStatus &= mVisualContext->setTensorAddress(binding_names::kVisualInput, mVitInput.rawPointer());

    mShuffleIndices = rt::Tensor({mConfig.maxNumPatches / (scale * scale), scale * scale}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kINT64, "NemotronOmniViTRunner::mShuffleIndices");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kVisualShuffleIndices, mShuffleIndices.rawPointer());

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

    // Pixel staging and GEMM-row scratch. Video tubelets pack T frames per block, so the pixel
    // staging holds up to maxNumBlocks * T frames and GEMM rows are T*3*P*P wide.
    int64_t const T = std::max<int64_t>(1, mConfig.videoTemporalPatchSize);
    mBlockPixels
        = rt::Tensor({mConfig.maxNumBlocks * T, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "NemotronOmniViTRunner::mBlockPixels");
    int64_t const maxPatchDim = T * mConfig.numChannels * mConfig.patchSize * mConfig.patchSize;
    mGemmRows = rt::Tensor({mConfig.maxNumBlocks * mConfig.maxNumPatches, maxPatchDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "NemotronOmniViTRunner::mGemmRows");

    // Position embeddings: static square grid for image tiles, per-grid scratch for video. mPosEmbedGridHost
    // is pinned host staging (kCPU) shared by the image and per-video pos-embed H2D uploads.
    int64_t const gridSide = mConfig.blockImageSizeH / mConfig.patchSize;
    mPosEmbedGridHost = rt::Tensor({mConfig.maxNumPatches, mConfig.vitHiddenSize}, rt::DeviceType::kCPU,
        nvinfer1::DataType::kHALF, "NemotronOmniViTRunner::mPosEmbedGridHost");
    {
        std::vector<half> const posImage = interpolatePosEmbedHost(gridSide, gridSide);
        mPosEmbedImage = rt::Tensor({gridSide * gridSide, mConfig.vitHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "NemotronOmniViTRunner::mPosEmbedImage");
        std::memcpy(mPosEmbedGridHost.rawPointer(), posImage.data(), posImage.size() * sizeof(half));
        CUDA_CHECK(cudaMemcpyAsync(mPosEmbedImage.rawPointer(), mPosEmbedGridHost.rawPointer(),
            posImage.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
        // mPosEmbedGridHost is shared staging the video path overwrites; the image H2D
        // must finish reading it before reuse. Construction-time only, off the hot path.
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    mPosEmbedGrid = rt::Tensor({mConfig.maxNumPatches, mConfig.vitHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "NemotronOmniViTRunner::mPosEmbedGrid");
    mImageShuffleTable = buildShuffleIndicesHost(gridSide, gridSide);

    // EVS scratch (video token pruning).
    mEvsScores = rt::Tensor(
        {maxTotalTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "NemotronOmniViTRunner::mEvsScores");
    mEvsGatherIdx = rt::Tensor(
        {maxTotalTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "NemotronOmniViTRunner::mEvsGatherIdx");
    mEvsScratch = rt::Tensor({maxTotalTokens, mConfig.outHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "NemotronOmniViTRunner::mEvsScratch");

    // Pinned (rt::Tensor kCPU = cudaMallocHost) host staging so the shuffle-index and EVS gather-index
    // H2D copies run truly async instead of the pageable synchronous-staging fallback.
    mShuffleIndicesHost = rt::Tensor({mConfig.maxNumPatches}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64,
        "NemotronOmniViTRunner::mShuffleIndicesHost");
    mEvsGatherIdxHost = rt::Tensor(
        {maxTotalTokens}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "NemotronOmniViTRunner::mEvsGatherIdxHost");
    mEvsScoresHost = rt::Tensor(
        {maxTotalTokens}, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "NemotronOmniViTRunner::mEvsScoresHost");

    // Pre-allocate temporary image buffers for preprocessing
    int64_t const maxImagePixels = mBlockPixels.getShape().volume();
    mImageDevice = rt::Tensor(
        {maxImagePixels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "NemotronOmniViTRunner::mImageDevice");
    mNormalizedImageDevice = rt::Tensor({maxImagePixels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "NemotronOmniViTRunner::mNormalizedImageDevice");

    // Pinned (rt::Tensor kCPU = cudaMallocHost) host buffer for the video CPU-resize output; images
    // use the GPU resize path. Sized for the largest video: maxNumBlocks tubelets x T frames.
    mResizedImageHost = rt::imageUtils::ImageData(
        rt::Tensor({mConfig.maxNumBlocks * T, mConfig.blockImageSizeH, mConfig.blockImageSizeW, mConfig.numChannels},
            rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "NemotronOmniViTRunner::mResizedImageHost"));

    // GPU image-resize scratch.
    // Horizontal-pass scratch holds [rawH, outW, C] floats. computeBestBlockGridForResize snaps to a
    // block grid and does NOT preserve aspect ratio, so bound each dimension independently: rawH by the
    // raw cap and outW by the widest single-row block grid (maxNumBlocks * blockImageSizeW).
    int64_t const kMaxResizeTmpElems
        = kernel::kGpuResizeMaxRawDim * mConfig.maxNumBlocks * mConfig.blockImageSizeW * channels;
    kernel::allocateResizeScratch(channels, kMaxResizeTmpElems, mRawImageDevice, mResizeTmpDevice);

    return true;
}

std::vector<half> NemotronOmniViTRunner::interpolatePosEmbedHost(int64_t gridH, int64_t gridW) const
{
    // Bilinear, align_corners=true — matches the Python export-time transform and F.interpolate.
    int64_t const side = mPosEmbedRawSide;
    int64_t const hidden = mConfig.vitHiddenSize;
    std::vector<half> out(static_cast<size_t>(gridH * gridW * hidden));

    double const scaleH = gridH > 1 ? static_cast<double>(side - 1) / static_cast<double>(gridH - 1) : 0.0;
    double const scaleW = gridW > 1 ? static_cast<double>(side - 1) / static_cast<double>(gridW - 1) : 0.0;
    for (int64_t y = 0; y < gridH; ++y)
    {
        double const srcY = static_cast<double>(y) * scaleH;
        int64_t const y0 = static_cast<int64_t>(srcY);
        int64_t const y1 = std::min(y0 + 1, side - 1);
        float const fy = static_cast<float>(srcY - static_cast<double>(y0));
        for (int64_t x = 0; x < gridW; ++x)
        {
            double const srcX = static_cast<double>(x) * scaleW;
            int64_t const x0 = static_cast<int64_t>(srcX);
            int64_t const x1 = std::min(x0 + 1, side - 1);
            float const fx = static_cast<float>(srcX - static_cast<double>(x0));

            float const w00 = (1.0F - fy) * (1.0F - fx);
            float const w01 = (1.0F - fy) * fx;
            float const w10 = fy * (1.0F - fx);
            float const w11 = fy * fx;
            float const* p00 = mPosEmbedRawHost.data() + (y0 * side + x0) * hidden;
            float const* p01 = mPosEmbedRawHost.data() + (y0 * side + x1) * hidden;
            float const* p10 = mPosEmbedRawHost.data() + (y1 * side + x0) * hidden;
            float const* p11 = mPosEmbedRawHost.data() + (y1 * side + x1) * hidden;
            half* dst = out.data() + (y * gridW + x) * hidden;
            for (int64_t c = 0; c < hidden; ++c)
            {
                dst[c] = __float2half(w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c]);
            }
        }
    }
    return out;
}

std::vector<int64_t> NemotronOmniViTRunner::buildShuffleIndicesHost(int64_t gridH, int64_t gridW) const
{
    // Row m = output cell (i, j); entries are input patch indices (s*i+di)*gridW + (s*j+dj) in
    // (di, dj) row-major order — the token order HF pixel_shuffle (ps_version v2) produces.
    int64_t const s = mConfig.downsampleScale;
    // Logical shape [gridH*gridW/s^2, s^2] — one entry per input patch.
    std::vector<int64_t> table(static_cast<size_t>(gridH * gridW));
    int64_t const outW = gridW / s;
    for (int64_t i = 0; i < gridH / s; ++i)
    {
        for (int64_t j = 0; j < outW; ++j)
        {
            for (int64_t di = 0; di < s; ++di)
            {
                for (int64_t dj = 0; dj < s; ++dj)
                {
                    table[((i * outW + j) * s + di) * s + dj] = (s * i + di) * gridW + (s * j + dj);
                }
            }
        }
    }
    return table;
}

void NemotronOmniViTRunner::bindEngineShapes(int64_t numBlocks, int64_t gridH, int64_t gridW, cudaStream_t stream)
{
    int64_t const s = mConfig.downsampleScale;
    int64_t const numPatches = gridH * gridW;
    // Both ternary operands must be lvalues, else the cache-hit path copies mImageShuffleTable.
    std::vector<int64_t> freshTable;
    bool const useCache
        = gridH == gridW && gridH == mConfig.blockImageSizeH / mConfig.patchSize && !mImageShuffleTable.empty();
    if (!useCache)
    {
        freshTable = buildShuffleIndicesHost(gridH, gridW);
    }
    std::vector<int64_t> const& table = useCache ? mImageShuffleTable : freshTable;
    check::check(mShuffleIndices.reshape({numPatches / (s * s), s * s}), "Tensor reshape failed");
    std::memcpy(mShuffleIndicesHost.rawPointer(), table.data(), table.size() * sizeof(int64_t));
    CUDA_CHECK(cudaMemcpyAsync(mShuffleIndices.rawPointer(), mShuffleIndicesHost.rawPointer(),
        table.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    nvinfer1::Dims3 const inputDims(numBlocks, numPatches, mConfig.vitHiddenSize);
    check::check(
        mVisualContext->setInputShape(binding_names::kVisualInput, inputDims), "Failed to set visual input shape");
    nvinfer1::Dims2 const shuffleDims(numPatches / (s * s), s * s);
    check::check(mVisualContext->setInputShape(binding_names::kVisualShuffleIndices, shuffleDims),
        "Failed to set shuffle indices shape");
}

void NemotronOmniViTRunner::runEmbedderStage(int64_t numGroups, int64_t numPatches, int64_t temporalPatchSize,
    rt::Tensor const& posEmbed, [[maybe_unused]] rt::Tensor const& weight, cudaStream_t stream)
{
    int64_t const patchDim = temporalPatchSize * mConfig.numChannels * mConfig.patchSize * mConfig.patchSize;
    check::check(mGemmRows.reshape({numGroups * numPatches, patchDim}), "Tensor reshape failed");
    kernel::transposeToPatchNemotronViT(mBlockPixels, mGemmRows, temporalPatchSize, mConfig.patchSize, stream);

    check::check(mVitInput.reshape({numGroups, numPatches, mConfig.vitHiddenSize}), "Tensor reshape failed");
#ifdef CUTE_DSL_GEMM_ENABLED
    bool const gemmOk = CuteDslGemmRunner::run(mGemmRows.rawPointer(), weight.rawPointer(), mVitInput.rawPointer(),
        static_cast<int32_t>(numGroups * numPatches), static_cast<int32_t>(mConfig.vitHiddenSize),
        static_cast<int32_t>(patchDim), stream);
    check::check(gemmOk, "Patch embedder GEMM dispatch failed");
#else
    check::check(false,
        "NemotronOmniViTRunner: built without CuTe DSL GEMM (rebuild with -DENABLE_CUTE_DSL=gemm) — the patch "
        "embedder GEMM cannot run");
#endif

    kernel::addPosEmbedNemotronViT(mVitInput, posEmbed, stream);
}

void NemotronOmniViTRunner::formatPatch(rt::imageUtils::ImageData const& image, std::vector<int64_t>& imageTokenLengths,
    int64_t& numImages, int64_t& totalNumBlocks, bool isThumbnail, cudaStream_t stream)
{
    int64_t height = image.height;
    int64_t width = image.width;
    int64_t channels = image.channels;

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

    // mImageDevice already holds the [1, height, width, channels] image, populated by the caller.
    check::check(mNormalizedImageDevice.reshape({1, height, width, channels}), "Tensor reshape failed");

    // Normalize image
    kernel::normalizeImage(mImageDevice, mImageMean, mImageStd, mNormalizedImageDevice, stream);

    // Transpose to block-split CHW staging (embedder GEMM input is patchified from here)
    int64_t offset = totalNumBlocks * mConfig.numChannels * mConfig.blockImageSizeH * mConfig.blockImageSizeW;
    kernel::transposeToPatchInternVLPhi4MM(mNormalizedImageDevice, mBlockPixels, offset, stream);

    // Update numBlocks
    totalNumBlocks += curNumBlocks;
}

void NemotronOmniViTRunner::imagePreprocessTokenLengthsOnly(
    rt::LLMGenerationRequest const& request, std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& numImages)
{
    mTotalNumBlocks = 0;
    for (auto const& req : request.requests)
    {
        int64_t numImage = 0;
        for (auto const& image : req.imageBuffers)
        {
            auto const [h, w] = image.doResize
                ? imageUtils::computeBestBlockGridForResize(image.height, image.width, mConfig.minImageTokensPerImage,
                      mConfig.maxImageTokensPerImage, mConfig.blockImageSizeH, mConfig.blockImageSizeW)
                : std::make_tuple(image.height, image.width);
            int64_t const mainBlocks = (h / mConfig.blockImageSizeH) * (w / mConfig.blockImageSizeW);
            int64_t tokens = mainBlocks * mConfig.tokensPerBlock;
            if (mainBlocks > 1)
            {
                tokens += mConfig.tokensPerBlock; // thumbnail
                mTotalNumBlocks += mainBlocks + 1;
            }
            else
            {
                mTotalNumBlocks += mainBlocks;
            }
            imageTokenLengths.push_back(tokens);
            ++numImage;
        }
        numImages.emplace_back(numImage);
    }

    if (mTotalNumBlocks > 0)
    {
        int64_t const totalImageTokens = mTotalNumBlocks * mConfig.tokensPerBlock;
        check::check(mOutputEmbedding.reshape({totalImageTokens, mConfig.outHiddenSize}), "Tensor reshape failed");
    }
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
        check::check(mVitInput.reshape({0, mConfig.maxNumPatches, mConfig.vitHiddenSize}), "Tensor reshape failed");
        return;
    }

    check::check(mBlockPixels.reshape(
                     {mConfig.maxNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW}),
        "Tensor reshape failed");

    for (auto const& req : request.requests)
    {
        int64_t numImage = 0;
        for (auto const& image : req.imageBuffers)
        {
            int64_t const blocksBeforePatch = mTotalNumBlocks;
            if (image.doResize)
            {
                // Resize image to the aspect-ratio-matched tile grid within the per-image tile budget
                auto [resizedHeight, resizedWidth] = imageUtils::computeBestBlockGridForResize(image.height,
                    image.width, mConfig.minImageTokensPerImage, mConfig.maxImageTokensPerImage,
                    mConfig.blockImageSizeH, mConfig.blockImageSizeW);
                kernel::copyImageToDeviceAndResize(image.data(), image.frames, image.height, image.width,
                    image.channels, mRawImageDevice, mResizeTmpDevice, mImageDevice, resizedHeight, resizedWidth,
                    stream);
                formatPatch(image.resizedMeta(resizedHeight, resizedWidth), imageTokenLengths, numImage,
                    mTotalNumBlocks, false, stream);
            }
            else
            {
                LOG_DEBUG("Skipping resize for pre-resized image %ldx%ld", image.height, image.width);
                kernel::copyImageToDeviceAndResize(image.data(), image.frames, image.height, image.width,
                    image.channels, mRawImageDevice, mResizeTmpDevice, mImageDevice, image.height, image.width, stream);
                formatPatch(image, imageTokenLengths, numImage, mTotalNumBlocks, false, stream);
            }

            // Only add thumbnail when the image has more than 1 block (matches HuggingFace behavior). The
            // thumbnail is a one-block resize of the ORIGINAL image and is not gated by image.doResize.
            int64_t const mainImageBlocks = mTotalNumBlocks - blocksBeforePatch;
            if (mainImageBlocks > 1)
            {
                kernel::copyImageToDeviceAndResize(image.data(), 1, image.height, image.width, image.channels,
                    mRawImageDevice, mResizeTmpDevice, mImageDevice, mConfig.blockImageSizeH, mConfig.blockImageSizeW,
                    stream);
                formatPatch(image.resizedMeta(mConfig.blockImageSizeH, mConfig.blockImageSizeW), imageTokenLengths,
                    numImage, mTotalNumBlocks, true, stream);
            }
        }
        numImages.emplace_back(numImage);
    }

    if (mTotalNumBlocks == 0)
    {
        check::check(mVitInput.reshape({0, mConfig.maxNumPatches, mConfig.vitHiddenSize}), "Tensor reshape failed");
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

    // Embedder stage: patchify + image-embedder GEMM + square-grid pos_embed.
    check::check(
        mBlockPixels.reshape({mTotalNumBlocks, mConfig.numChannels, mConfig.blockImageSizeH, mConfig.blockImageSizeW}),
        "Tensor reshape failed");
    runEmbedderStage(
        mTotalNumBlocks, mConfig.maxNumPatches, /*temporalPatchSize=*/1, mPosEmbedImage, mImageEmbedderWeight, stream);

    check::check(mOutputEmbedding.reshape({totalImageTokens, mConfig.outHiddenSize}), "Tensor reshape failed");
}

std::pair<int64_t, int64_t> NemotronOmniViTRunner::computeVideoTargetPatches(int64_t width, int64_t height) const
{
    // HF/vLLM aspect-preserving video frame sizing (_compute_target_patches_video): ~videoTargetNumPatches
    // patches snapped to the pixel-shuffle factor, but the snap steps down (not up) when a round-up would
    // overshoot, so the grid never exceeds videoTargetNumPatches — an EdgeLLM profile-safe deviation
    // (validateAndFillConfig holds videoTargetNumPatches <= the engine patch budget).
    int64_t const target = mConfig.videoTargetNumPatches;
    int64_t const divisor = mConfig.downsampleScale;
    if (!mConfig.videoMaintainAspectRatio)
    {
        int64_t side = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(target))));
        side = std::max(divisor, (side / divisor) * divisor);
        return {side, side};
    }
    double const aspect = static_cast<double>(width) / static_cast<double>(std::max<int64_t>(height, 1));
    int64_t ph = std::max<int64_t>(std::llround(std::sqrt(static_cast<double>(target) / aspect)), 1);
    int64_t pw = std::max<int64_t>(std::llround(std::sqrt(static_cast<double>(target) * aspect)), 1);
    if (divisor > 1)
    {
        int64_t const remH = ph % divisor;
        int64_t const remW = pw % divisor;
        int64_t const phUp = ph + (remH != 0 ? divisor - remH : 0);
        int64_t const pwUp = pw + (remW != 0 ? divisor - remW : 0);
        if (phUp * pwUp <= target)
        {
            ph = phUp;
            pw = pwUp;
        }
        else
        {
            // Round-down is a no-op when a dim is already divisor-aligned, so step down to fit.
            ph = std::max(divisor, ph - remH);
            pw = std::max(divisor, pw - remW);
            while (ph * pw > target && (ph > divisor || pw > divisor))
            {
                if (pw >= ph && pw > divisor)
                {
                    pw -= divisor;
                }
                else
                {
                    ph -= divisor;
                }
            }
        }
    }
    return {pw, ph};
}

void NemotronOmniViTRunner::videoPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& numImages, cudaStream_t stream)
{
    // Videos follow the HF processor contract: batch of one, a single video, no mixed-in images
    // (the ViT output row order must match placeholder order, and image tiles and video tubelets
    // need separate engine enqueues with different shapes).
    ELLM_CHECK(request.requests.size() == 1, "Nemotron-Omni video: batch size must be 1");
    ELLM_CHECK(request.requests[0].imageBuffers.size() == 1,
        "Nemotron-Omni video: exactly one video per request (no mixed image+video)");
    ELLM_CHECK(mHasVideoEmbedder,
        "Nemotron-Omni video: the embedder weight file has no video_embedder weights — re-export the visual model "
        "from a checkpoint that carries them");

    auto const& video = request.requests[0].imageBuffers[0];
    ELLM_CHECK(video.doResize,
        "Nemotron-Omni video does not support do_resize=false; the runner always resizes to the target grid");
    int64_t const T = mConfig.videoTemporalPatchSize;
    int64_t const P = mConfig.patchSize;
    int64_t const scale = mConfig.downsampleScale;

    auto const [gridW, gridH] = computeVideoTargetPatches(video.width, video.height);
    int64_t const frameH = gridH * P;
    int64_t const frameW = gridW * P;
    int64_t const numPatches = gridH * gridW;

    int64_t const numFrames = video.frames;
    int64_t const paddedFrames = ((numFrames + T - 1) / T) * T;
    int64_t const numGroups = paddedFrames / T;
    ELLM_CHECK(numGroups <= mConfig.maxNumBlocks,
        "Nemotron-Omni video: " + std::to_string(numGroups) + " tubelets exceed the engine block budget "
            + std::to_string(mConfig.maxNumBlocks) + "; sample fewer frames");
    ELLM_CHECK(numPatches <= mConfig.maxNumPatches,
        "Nemotron-Omni video: grid " + std::to_string(gridH) + "x" + std::to_string(gridW) + " = "
            + std::to_string(numPatches) + " patches exceed the engine patch budget "
            + std::to_string(mConfig.maxNumPatches));
    ELLM_CHECK(gridH % scale == 0 && gridW % scale == 0,
        "Nemotron-Omni video: grid " + std::to_string(gridH) + "x" + std::to_string(gridW)
            + " is not divisible by the downsample scale " + std::to_string(scale));

    // Resize all frames to the aspect-preserving grid (host, UINT8 bicubic). The HF processor resizes in
    // FP32 with antialias, so this is a close approximation, not a bit-exact match.
    // resizeImage returns `video` itself when already at target size, so consume the return.
    auto const& resized = rt::imageUtils::resizeImage(
        video, mResizedImageHost, frameW, frameH, rt::imageUtils::InterpolationMode::kBICUBIC);

    // One bulk H2D of the contiguous [numFrames,H,W,C] frames + one normalize. The transpose stays
    // per-frame (kernel launches only); padding to a multiple of T reuses the last frame's view.
    check::check(mBlockPixels.reshape({paddedFrames, mConfig.numChannels, frameH, frameW}), "Tensor reshape failed");
    check::check(mImageDevice.reshape({numFrames, frameH, frameW, mConfig.numChannels}), "Tensor reshape failed");
    check::check(
        mNormalizedImageDevice.reshape({numFrames, frameH, frameW, mConfig.numChannels}), "Tensor reshape failed");
    int64_t const frameBytes = frameH * frameW * mConfig.numChannels;
    CUDA_CHECK(cudaMemcpyAsync(
        mImageDevice.rawPointer(), resized.data(), numFrames * frameBytes, cudaMemcpyHostToDevice, stream));
    kernel::normalizeImage(mImageDevice, mImageMean, mImageStd, mNormalizedImageDevice, stream);
    auto* normBase = static_cast<half*>(mNormalizedImageDevice.rawPointer());
    for (int64_t f = 0; f < paddedFrames; ++f)
    {
        int64_t const srcFrame = std::min(f, numFrames - 1);
        rt::Tensor frameView(normBase + srcFrame * frameBytes, {1, frameH, frameW, mConfig.numChannels},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "NemotronOmniViTRunner::frameView");
        kernel::transposeToPatchInternVLPhi4MM(
            frameView, mBlockPixels, f * mConfig.numChannels * frameH * frameW, stream);
    }

    // Embedder stage with the video weights and the per-grid pos_embed, then the shared engine.
    {
        std::vector<half> const posGrid = interpolatePosEmbedHost(gridH, gridW);
        check::check(mPosEmbedGrid.reshape({numPatches, mConfig.vitHiddenSize}), "Tensor reshape failed");
        std::memcpy(mPosEmbedGridHost.rawPointer(), posGrid.data(), posGrid.size() * sizeof(half));
        CUDA_CHECK(cudaMemcpyAsync(mPosEmbedGrid.rawPointer(), mPosEmbedGridHost.rawPointer(),
            posGrid.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    }
    runEmbedderStage(numGroups, numPatches, T, mPosEmbedGrid, mVideoEmbedderWeight, stream);

    // Run the visual engine here: the EVS-pruned per-tubelet token counts are needed for text
    // placeholder expansion, which happens later in preprocess().
    {
        TIME_STAGE(metrics::StageNames::kMULTIMODAL_PROCESSING, stream);
        bindEngineShapes(numGroups, gridH, gridW, stream);
        ELLM_CHECK(mVisualContext->enqueueV3(stream), "Nemotron-Omni video: visual engine enqueue failed");
    }

    int64_t const tokensPerGroup = numPatches / (scale * scale);
    int64_t const totalTokens = numGroups * tokensPerGroup;
    check::check(mOutputEmbedding.reshape({totalTokens, mConfig.outHiddenSize}), "Tensor reshape failed");

    std::vector<int64_t> groupTokenCounts(static_cast<size_t>(numGroups), tokensPerGroup);
    if (mConfig.videoPruningRate > 0.0F)
    {
        // EVS: rank tokens by temporal dissimilarity of the projected embeddings and keep the
        // top (1-q) fraction, but never less than one tubelet's worth. Group 0 scores the 255
        // sentinel so the first tubelet always survives. Matches HF/vLLM EVS with a stable sort.
        kernel::evsScoresNemotronViT(mOutputEmbedding, mEvsScores, tokensPerGroup, stream);
        float* const scoresHost = mEvsScoresHost.dataPointer<float>();
        CUDA_CHECK(cudaMemcpyAsync(
            scoresHost, mEvsScores.rawPointer(), totalTokens * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int64_t const evsTokens
            = static_cast<int64_t>(static_cast<double>(totalTokens) * (1.0 - mConfig.videoPruningRate));
        int64_t const numKeep = std::max(tokensPerGroup, evsTokens);

        std::vector<int64_t> order(static_cast<size_t>(totalTokens));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(
            order.begin(), order.end(), [&scoresHost](int64_t a, int64_t b) { return scoresHost[a] > scoresHost[b]; });

        std::vector<bool> retained(static_cast<size_t>(totalTokens), false);
        for (int64_t k = 0; k < numKeep; ++k)
        {
            retained[order[k]] = true;
        }

        std::vector<int32_t> gatherIdx;
        gatherIdx.reserve(static_cast<size_t>(numKeep));
        std::fill(groupTokenCounts.begin(), groupTokenCounts.end(), 0);
        for (int64_t i = 0; i < totalTokens; ++i)
        {
            if (retained[i])
            {
                gatherIdx.push_back(static_cast<int32_t>(i));
                ++groupTokenCounts[i / tokensPerGroup];
            }
        }

        // Row compaction is an embedding-table lookup: dst[i] = embeds[retainedIdx[i]].
        check::check(mEvsGatherIdx.reshape({1, numKeep}), "Tensor reshape failed");
        std::memcpy(mEvsGatherIdxHost.rawPointer(), gatherIdx.data(), numKeep * sizeof(int32_t));
        CUDA_CHECK(cudaMemcpyAsync(mEvsGatherIdx.rawPointer(), mEvsGatherIdxHost.rawPointer(),
            numKeep * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        check::check(mEvsScratch.reshape({1, numKeep, mConfig.outHiddenSize}), "Tensor reshape failed");
        kernel::embeddingLookup(mEvsGatherIdx, mOutputEmbedding, std::nullopt, mEvsScratch, stream);
        CUDA_CHECK(cudaMemcpyAsync(mOutputEmbedding.rawPointer(), mEvsScratch.rawPointer(),
            numKeep * mConfig.outHiddenSize * sizeof(half), cudaMemcpyDeviceToDevice, stream));
        check::check(mOutputEmbedding.reshape({numKeep, mConfig.outHiddenSize}), "Tensor reshape failed");
        LOG_INFO("Nemotron-Omni EVS: retained %ld of %ld video tokens (q=%.2f)", numKeep, totalTokens,
            static_cast<double>(mConfig.videoPruningRate));
    }

    int64_t const retainedTokens = std::accumulate(groupTokenCounts.begin(), groupTokenCounts.end(), int64_t{0});
    mMultimodalMetrics.recordRun(numGroups, retainedTokens);

    // One video item; textPreprocess expands its single placeholder into the per-tubelet
    // Frame-label sequence (labels from the source timestamps, one img-group per tubelet at
    // its EVS-pruned token count).
    mVideoTubeletCounts.assign(groupTokenCounts.begin(), groupTokenCounts.end());
    // Frame labels need one timestamp per sampled frame. Callers that skip them
    // (CLI frames-path) get uniform fps spacing via the vLLM formula so the labels
    // still bound the frame count and carry "sampled at".
    if (video.timestamps.empty())
    {
        double const fps = video.fps > 0.0 ? video.fps : 1.0;
        int64_t const perMs = static_cast<int64_t>(1000.0 / fps);
        mVideoTimestamps.resize(static_cast<size_t>(numFrames));
        for (int64_t i = 0; i < numFrames; ++i)
        {
            mVideoTimestamps[static_cast<size_t>(i)] = static_cast<double>(i * perMs) / 1000.0;
        }
    }
    else
    {
        ELLM_CHECK(static_cast<int64_t>(video.timestamps.size()) == numFrames,
            "Nemotron-Omni video: timestamps must have one entry per sampled frame");
        mVideoTimestamps = video.timestamps;
    }
    imageTokenLengths.clear();
    numImages.assign(1, 1);

    // The engine already ran; infer() must be a no-op for this request.
    mTotalNumBlocks = 0;
}

std::vector<int32_t> NemotronOmniViTRunner::buildVideoPlaceholderIds(
    trt_edgellm::tokenizer::Tokenizer const* tokenizer) const
{
    // HF/vLLM Frame-label layout: one "<img>{IMG_CONTEXT x count}</img>" group per tubelet, each
    // prefixed by newline-joined per-frame timestamp labels ("Frame"/"frame" i sampled at t seconds).
    int64_t const T = mConfig.videoTemporalPatchSize;
    int64_t const numFrames = static_cast<int64_t>(mVideoTimestamps.size());
    bool const haveTimes = numFrames > 0;

    std::vector<int32_t> const newline = tokenizer->encode("\n");

    std::vector<int32_t> out;
    for (size_t g = 0; g < mVideoTubeletCounts.size(); ++g)
    {
        if (g > 0)
        {
            out.insert(out.end(), newline.begin(), newline.end());
        }
        std::string label;
        for (int64_t j = 0; j < T; ++j)
        {
            int64_t const fi = static_cast<int64_t>(g) * T + j;
            if (haveTimes && fi >= numFrames)
            {
                break; // last tubelet may be short (odd sampled frame count)
            }
            if (j > 0)
            {
                label += " and ";
            }
            label += (j == 0 ? "Frame " : "frame ") + std::to_string(fi + 1);
            if (haveTimes)
            {
                char buf[32];
                std::snprintf(buf, sizeof(buf), " sampled at %.2f seconds", mVideoTimestamps[fi]);
                label += buf;
            }
        }
        label += ": ";
        std::vector<int32_t> const labelIds = tokenizer->encode(label);
        out.insert(out.end(), labelIds.begin(), labelIds.end());

        out.push_back(mConfig.imgStartTokenId);
        for (int64_t k = 0; k < mVideoTubeletCounts[g]; ++k)
        {
            out.push_back(mConfig.imgContextTokenId);
        }
        out.push_back(mConfig.imgEndTokenId);
    }
    return out;
}

void NemotronOmniViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int64_t> const& numImages,
    std::vector<int64_t> const& imageTokenLengths, trt_edgellm::tokenizer::Tokenizer const* tokenizer)
{
    // Reuse batchInputIds if the audio runner already tokenized, else tokenize here. Each
    // imgContextTokenId marker is then expanded: images repeat it by their token count, the
    // single video marker becomes the per-tubelet Frame-label sequence.
    bool const alreadyTokenized = batchInputIds.size() == request.requests.size();
    int64_t imageIndex = 0;
    std::vector<int32_t> const videoIds
        = mRequestHasVideo ? buildVideoPlaceholderIds(tokenizer) : std::vector<int32_t>{};

    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        std::vector<int32_t> ids = alreadyTokenized
            ? std::move(batchInputIds[i])
            : tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);
        check::check(!ids.empty(), "Failed to encode text");

        // Per-request media window (numImages[i]): a placeholder may only consume this request's own media (batch
        // isolation).
        int64_t const mediaEnd = imageIndex + numImages[i];

        std::vector<int32_t> newIds;
        newIds.reserve(ids.size());
        for (auto const& id : ids)
        {
            if (id != mConfig.imgContextTokenId)
            {
                newIds.push_back(id);
            }
            else if (mRequestHasVideo)
            {
                ELLM_CHECK(imageIndex < mediaEnd,
                    "EDGELLM_BAD_MEDIA_COUNT: NemotronOmniViTRunner::textPreprocess() placeholder count exceeds this "
                    "request's image count");
                newIds.insert(newIds.end(), videoIds.begin(), videoIds.end());
                ++imageIndex;
            }
            else
            {
                ELLM_CHECK(imageIndex < mediaEnd,
                    "EDGELLM_BAD_MEDIA_COUNT: NemotronOmniViTRunner::textPreprocess() placeholder count exceeds this "
                    "request's image count");
                int64_t const numImageTokens = imageTokenLengths.at(imageIndex);
                for (int64_t k = 0; k < numImageTokens; ++k)
                {
                    newIds.push_back(mConfig.imgContextTokenId);
                }
                ++imageIndex;
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
        ELLM_CHECK(imageIndex == mediaEnd,
            "EDGELLM_BAD_MEDIA_COUNT: NemotronOmniViTRunner::textPreprocess() placeholder count is smaller than this "
            "request's image count");
    }
}

bool NemotronOmniViTRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
    [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream, bool imageOnly, bool skipEncoderWork)
{
    std::vector<int64_t> imageTokenLengths;
    std::vector<int64_t> numImages;

    try
    {
        bool hasVideo = false;
        for (auto const& req : request.requests)
        {
            for (auto const& image : req.imageBuffers)
            {
                hasVideo |= image.isVideo;
            }
        }

        mRequestHasVideo = hasVideo;

        if (skipEncoderWork)
        {
            imagePreprocessTokenLengthsOnly(request, imageTokenLengths, numImages);
        }
        else if (hasVideo)
        {
            videoPreprocess(request, imageTokenLengths, numImages, stream);
        }
        else
        {
            imagePreprocess(request, imageTokenLengths, numImages, stream);
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

bool NemotronOmniViTRunner::infer(cudaStream_t stream) noexcept
{
    if (mTotalNumBlocks == 0)
    {
        return true; // No image blocks pending (videos already ran inside preprocess)
    }

    {
        TIME_STAGE(metrics::StageNames::kMULTIMODAL_PROCESSING, stream);

        try
        {
            int64_t const gridSide = mConfig.blockImageSizeH / mConfig.patchSize;
            bindEngineShapes(mTotalNumBlocks, gridSide, gridSide, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to bind engine input tensors: %s", e.what());
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
