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

#pragma once

#include "multimodal/common/multimodalRunner.h"
#include <cuda_fp16.h>
#include <tuple>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! \brief Configuration for Gemma4 vision encoder preprocessing and runtime bindings.
struct Gemma4ViTConfig
{
    int64_t maxPatches{0};             //!< Maximum number of visual patches in a request
    int64_t minPatches{0};             //!< Minimum number of visual patches in a request
    int64_t inputDim{0};               //!< Patch input dimension
    int64_t outHiddenSize{0};          //!< Output hidden dimension size
    int64_t patchSize{16};             //!< Patch size in pixels
    int64_t poolingKernelSize{3};      //!< Vision pooler spatial kernel size
    int64_t maxImageTokens{0};         //!< Maximum soft tokens in a request
    int64_t minImageTokensPerImage{0}; //!< Minimum soft tokens per image
    int64_t maxImageTokensPerImage{0}; //!< Maximum soft tokens per image
    int64_t maxNumImages{0};           //!< Maximum number of images per request
    int64_t maxPatchesPerImage{0};     //!< Maximum patches per single image
    int64_t rotaryPosEmbDim{0};        //!< Gemma4 visual RoPE angle dimension
    float ropeTheta{100.0F};           //!< Gemma4 visual RoPE base frequency
    int32_t imageTokenId{0};           //!< Token ID for image placeholder
    int32_t beginImageTokenId{-1};     //!< boi token ID wrapping each image span (-1 when absent)
    int32_t endImageTokenId{-1};       //!< eoi token ID wrapping each image span (-1 when absent)
    std::vector<float> imageMean{};    //!< Image normalization mean values (RGB)
    std::vector<float> imageStd{};     //!< Image normalization standard deviation values (RGB)
};

//! \brief Runner for Gemma4 vision encoder.
class Gemma4ViTRunner : public MultimodalRunner
{
public:
    //! \brief Constructor for Gemma4ViTRunner.
    Gemma4ViTRunner(std::string const& engineDir, cudaStream_t stream);

    ~Gemma4ViTRunner() noexcept = default;

    bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer, [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut,
        cudaStream_t stream, bool imageOnly = false, bool skipEncoderWork = false) override;

    bool infer(cudaStream_t stream) noexcept override;

    bool validateAndFillConfig(std::string const& engineDir) override;

    bool allocateBuffer(cudaStream_t stream) override;

private:
    struct ImageGrid
    {
        int64_t patchHeight{0};
        int64_t patchWidth{0};
    };

    void formatPatch(rt::imageUtils::ImageData const& image, std::vector<ImageGrid>& imageGrids,
        std::vector<int64_t>& imageTokenLengths, int32_t* cuSeqlensData, int64_t& cuSeqlensSize, int64_t& maxSeqLen,
        cudaStream_t stream);

    void imagePreprocess(rt::LLMGenerationRequest const& request, std::vector<ImageGrid>& imageGrids,
        std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& numImages, cudaStream_t stream);

    //! \brief Compute token lengths from image dimensions without GPU pixel work.
    void imagePreprocessTokenLengthsOnly(rt::LLMGenerationRequest const& request,
        std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& numImages);

    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchInputIds,
        std::vector<int64_t> const& numImages, std::vector<int64_t> const& imageTokenLengths,
        trt_edgellm::tokenizer::Tokenizer const* tokenizer);

    void generatePoolingWeights(
        std::vector<ImageGrid> const& imageGrids, int64_t totalPatches, int64_t totalSoftTokens, cudaStream_t stream);

    Gemma4ViTConfig mConfig{};           //!< Gemma4 vision configuration
    rt::Tensor mVitInput{};              //!< Vision encoder input patches
    rt::Tensor mPixelPositionIds{};      //!< Pixel position IDs device tensor
    rt::Tensor mPixelPositionIdsHost{};  //!< Pixel position IDs host tensor
    rt::Tensor mRotaryPosEmb{};          //!< Gemma4 visual RoPE angle tensor
    rt::Tensor mPoolingWeights{};        //!< Dense pooling weights device tensor
    rt::Tensor mCuSeqlens{};             //!< Cumulative sequence lengths tensor
    rt::Tensor mCuSeqlensHost{};         //!< Cumulative sequence lengths host tensor
    rt::Tensor mKvLengths{};             //!< KV lengths for TRT-native attention
    rt::Tensor mMaxSeqLenCarrier{};      //!< Shape-only max sequence length carrier
    rt::Tensor mImageMean{};             //!< Image mean tensor
    rt::Tensor mImageStd{};              //!< Image standard deviation tensor
    rt::Tensor mImageDevice{};           //!< Temporary image buffer (holds the GPU-resized image)
    rt::Tensor mNormalizedImageDevice{}; //!< Temporary normalized image buffer
    rt::Tensor mRawImageDevice{};        //!< Raw (pre-resize) image device buffer for the GPU resize path
    rt::Tensor mResizeTmpDevice{};       //!< Float scratch (horizontal pass) for the GPU resize

    bool mUseTrtNativeVitAttn{false}; //!< Use TRT IAttentionV2 instead of ViTAttentionPlugin
    bool mHasMaxSeqLenCarrier{false}; //!< Whether the visual engine has max_seqlen_carrier binding
};

} // namespace rt
} // namespace trt_edgellm
