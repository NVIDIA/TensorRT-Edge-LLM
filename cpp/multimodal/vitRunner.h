/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "multimodal/multimodalRunner.h"
#include "runtime/imageUtils.h"
#include "tokenizer/tokenizer.h"

#include <array>
#include <cuda_runtime.h>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Configuration for model-agnostic fixed-shape ViT vision engines.
struct VitConfig
{
    int32_t batchSize{1};
    int32_t numChannels{3};
    int32_t imageHeight{0};
    int32_t imageWidth{0};
    int32_t seqLenPerImage{0};
    int32_t numImageTokens{0};
    int32_t outHiddenSize{0};
    int32_t imageTokenId{0};
    int32_t vocabSize{0};
    std::array<float, 3> imageMean{{0.5F, 0.5F, 0.5F}};
    std::array<float, 3> imageStd{{0.5F, 0.5F, 0.5F}};
};

class VitRunner : public MultimodalRunner
{
public:
    VitRunner(std::string const& engineDir, cudaStream_t stream);
    ~VitRunner() noexcept override = default;

    bool validateAndFillConfig(std::string const& configPath) override;
    bool allocateBuffer(cudaStream_t stream) override;

    bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer, rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream,
        bool imageOnly = false) noexcept override;

    bool infer(cudaStream_t stream) override;

private:
    void imagePreprocess(rt::LLMGenerationRequest const& request, cudaStream_t stream);
    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer);
    void formatImage(rt::imageUtils::ImageData const& image, int64_t batchIndex, cudaStream_t stream);

    VitConfig mConfig{};
    rt::Tensor mVitInput{};
    rt::Tensor mImageMean{};
    rt::Tensor mImageStd{};
    rt::Tensor mImageDevice{};
    rt::Tensor mNormalizedImageDevice{};
    rt::imageUtils::ImageData mResizedImageHost{};
};

} // namespace rt
} // namespace trt_edgellm
