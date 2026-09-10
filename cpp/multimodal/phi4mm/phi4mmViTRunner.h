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

#pragma once

#include "multimodal/common/multimodalRunner.h"
#include "tokenizer/tokenizer.h"
#include <cuda_fp16.h>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! \brief Configuration for Phi4MMViT vision encoder
//!
//! This configuration aggregates vision-tower-derived dimensions (num blocks, channels,
//! output hidden size), tokenizer-related settings for image token expansion, and
//! image normalization parameters used by the CUDA preprocess kernels.
struct Phi4MMViTConfig
{
    int32_t maxNumBlocks{0};      //!< Maximum number of image blocks supported by engine
    int32_t minNumBlocks{0};      //!< Minimum number of image blocks supported by engine
    int32_t numChannels{3};       //!< Image channels (RGB=3)
    int32_t outHiddenSize{0};     //!< Visual output hidden size (projection dim)
    int32_t imageTokenId{200010}; //!< Placeholder token id in text to be expanded into image tokens
    std::array<float, 3> imageMean{{0.5F, 0.5F, 0.5F}}; //!< Mean per channel used in normalize: (val/255 - mean)/std
    std::array<float, 3> imageStd{{0.5F, 0.5F, 0.5F}};  //!< Std per channel used in normalize
    int32_t minImageTokensPerImage{0};                  //!< Min visual tokens per image (for resize/grid selection)
    int32_t maxImageTokensPerImage{0};                  //!< Max visual tokens per image (for resize/grid selection)
    int32_t blockImageSizeH{0};                         //!< Block image height (crop size)
    int32_t blockImageSizeW{0};                         //!< Block image width (crop size)
    int32_t blockDownsampleRatio{28};                   //!< Block downsample ratio
    int32_t tokensPerSide{0};                           //!< Number of tokens per dimension
};

//! \brief Runner for Phi-4MM vision encoder
//!
//! This class handles:
//! - Image preprocessing (HWC uint8 → normalized FP16 HWC on GPU)
//! - Tiling to per-block CHW layout for the TRT visual engine
//! - Running the visual engine to produce raw 256-per-block visual tokens
//! - Batched HD postprocess to assemble sub/global grids with newline and GN tokens
//! - Text preprocessing to expand image placeholders into a contiguous id range
class Phi4MMViTRunner : public MultimodalRunner
{
public:
    //! \brief Constructor for Phi4MMViTRunner
    //! \param[in] engineDir Directory containing the TensorRT engine files
    //! \param[in] stream CUDA stream for execution
    //! \throws std::runtime_error if engineDir does not contain valid engine files
    //! \throws std::runtime_error if buffer allocation fails
    Phi4MMViTRunner(std::string const& engineDir, cudaStream_t stream);
    ~Phi4MMViTRunner() noexcept = default;

    //! \brief Preprocess multimodal input including images and text
    //! \param[in] request LLM generation request containing images and text
    //! \param[in,out] batchedInputIds Batched input token IDs after preprocessing
    //! \param[in] tokenizer Tokenizer for text processing
    //! \param[out] mropeCosSinOut MRope cos/sin cache (unused by Phi-4MM; standard RoPE)
    //! \param[in] stream CUDA stream for execution
    //! \return True if preprocessing succeeded, false otherwise
    bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer, [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut,
        cudaStream_t stream, bool imageOnly = false, bool skipEncoderWork = false) override;

    //! \brief Run inference on the vision encoder and perform HD postprocess
    //! \param[in] stream CUDA stream for execution
    //! \return True if inference succeeded, false otherwise
    //! \throws std::runtime_error if a CUDA operation fails
    bool infer(cudaStream_t stream) override;

    //! \brief Validate and load configuration from JSON file
    //! \param[in] configPath Path to configuration file
    //! \return True if configuration is valid and loaded successfully, false otherwise
    bool validateAndFillConfig(std::string const& configPath) override;

    //! \brief Allocate buffers for inference and postprocess
    //! \param[in] stream CUDA stream for execution
    //! \return True if allocation succeeded, false otherwise
    //! \throws std::runtime_error if a CUDA operation fails
    bool allocateBuffer(cudaStream_t stream) override;

private:
    //! \brief Preprocess images in the request
    //! \param[in] request LLM generation request containing images
    //! \param[out] imageTokenLengths Token lengths for each image
    //! \param[out] numImages Number of images per prompt
    //! \param[in] stream CUDA stream for execution
    //! \throws std::runtime_error if an image reshape fails, or the number of blocks is inconsistent
    void imagePreprocess(rt::LLMGenerationRequest const& request, std::vector<int64_t>& imageTokenLengths,
        std::vector<int64_t>& numImages, std::vector<std::vector<std::vector<int64_t>>>& imagesBlockGridHW,
        cudaStream_t stream);
    void imagePreprocessTokenLengthsOnly(rt::LLMGenerationRequest const& request,
        std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& numImages);

    //! \brief Preprocess text portion of the request
    //! \param[in] request LLM generation request
    //! \param[out] batchInputIds Batch of input token IDs
    //! \param[in] numImages Number of images per request
    //! \param[in] imageTokenLengths Token lengths for each image
    //! \param[in] tokenizer Tokenizer for text processing
    //! \throws std::runtime_error if the text encoding fails, or the number of images does not match the
    //!         number of requests
    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchInputIds,
        std::vector<int64_t> const& numImages, std::vector<int64_t> const& imageTokenLengths,
        tokenizer::Tokenizer const* tokenizer);

    //! \brief Normalize the GPU-resident image, tile to blocks, and update token-length accounting
    //! \throws std::runtime_error if image not divisible into blocks
    void formatPatch(rt::imageUtils::ImageData const& image, std::vector<int64_t>& imageTokenLengths,
        int64_t& numImages, int64_t& totalNumBlocks, bool isThumbnail, cudaStream_t stream);

    Phi4MMViTConfig mConfig{};           //!< Phi-4MM visual configuration
    rt::Tensor mVitInput{};              //!< Visual engine input tensor
    rt::Tensor mImageMean{};             //!< Image mean tensor [C]
    rt::Tensor mImageStd{};              //!< Image std tensor [C]
    rt::Tensor mImageDevice{};           //!< Temporary image buffer (holds the GPU-resized image)
    rt::Tensor mNormalizedImageDevice{}; //!< Temporary normalized image buffer
    rt::Tensor mRawImageDevice{};        //!< Raw (pre-resize) image device buffer for the GPU resize path
    rt::Tensor mResizeTmpDevice{};       //!< Float scratch (horizontal pass) for the GPU resize
    std::vector<std::vector<std::vector<int64_t>>> mImagesBlockGridHW; //!< Per-image block grid sizes [[hb, wb], ...]

    // Buffer for raw ViT outputs from the TRT engine before Phi4MM postprocess (HD transform)
    rt::Tensor mEngineOutputEmbedding{}; //!< Raw visual tokens [numBlocks*256, hidden]
    // Newline embeddings (sub-grid newline and global-GN token), already projected to hidden
    rt::Tensor mSubGNProj{};  //!< Sub-grid newline token vector [hidden_size]
    rt::Tensor mGlbGNProj{};  //!< Global GN single token vector [hidden_size]
    std::string mEngineDir{}; //!< Engine directory for auxiliary assets (e.g., safetensors)

    // Device buffer for Phi4MM postprocess
    rt::Tensor mHBlocks{};     //!< Block height indices
    rt::Tensor mWBlocks{};     //!< Block width indices
    rt::Tensor mSrcGlbStart{}; //!< Global start indices
    rt::Tensor mSrcSubStart{}; //!< Sub-grid start indices
    rt::Tensor mDstOutStart{}; //!< Destination output start indices
    rt::Tensor mSubOutLen{};   //!< Sub-grid output lengths

    // CPU buffer for Phi4MM postprocess
    rt::Tensor mHBlocksHost{};     //!< Block height indices
    rt::Tensor mWBlocksHost{};     //!< Block width indices
    rt::Tensor mSrcGlbStartHost{}; //!< Global start indices
    rt::Tensor mSrcSubStartHost{}; //!< Sub-grid start indices
    rt::Tensor mDstOutStartHost{}; //!< Destination output start indices
    rt::Tensor mSubOutLenHost{};   //!< Sub-grid output lengths
};

} // namespace rt
} // namespace trt_edgellm
