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
#include <tuple>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Runtime configuration for the encoder-free Gemma4 Unified visual embedder.
struct Gemma4UnifiedVisionConfig
{
    int64_t minPatches{0};
    int64_t maxPatches{0};
    int64_t maxPatchesPerImage{0};
    int64_t inputDim{0};
    int64_t outputHiddenSize{0};
    int64_t modelPatchSize{48};
    int64_t positionEmbeddingSize{0};
    int32_t imageTokenId{0};
    int32_t beginImageTokenId{0};
    int32_t endImageTokenId{0};
};

//! Preprocess RGB images directly into 48x48 patches and run the lightweight Unified visual graph.
class Gemma4UnifiedVisionRunner : public MultimodalRunner
{
public:
    Gemma4UnifiedVisionRunner(std::string const& engineDir, cudaStream_t stream);
    ~Gemma4UnifiedVisionRunner() noexcept = default;

    bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer, rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream,
        bool imageOnly = false, bool skipEncoderWork = false) override;
    bool infer(cudaStream_t stream) noexcept override;
    bool validateAndFillConfig(std::string const& engineDir) override;
    bool allocateBuffer(cudaStream_t stream) override;

private:
    void formatImage(rt::imageUtils::ImageData const& image, int64_t& patchOffset,
        std::vector<int64_t>& imageTokenLengths, cudaStream_t stream);
    void imagePreprocess(rt::LLMGenerationRequest const& request, std::vector<int64_t>& imageTokenLengths,
        std::vector<int64_t>& imagesPerRequest, cudaStream_t stream);
    void imagePreprocessTokenLengthsOnly(rt::LLMGenerationRequest const& request,
        std::vector<int64_t>& imageTokenLengths, std::vector<int64_t>& imagesPerRequest);
    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        std::vector<int64_t> const& imageTokenLengths, std::vector<int64_t> const& imagesPerRequest,
        tokenizer::Tokenizer const* tokenizer);

    Gemma4UnifiedVisionConfig mConfig{};
    rt::Tensor mVisualInput{};
    rt::Tensor mPixelPositionIds{};
    rt::Tensor mPixelPositionIdsHost{};
    rt::Tensor mImageMean{};
    rt::Tensor mImageStd{};
    rt::Tensor mImageDevice{};
    rt::Tensor mRescaledImageDevice{};
    rt::Tensor mRawImageDevice{};  //!< Raw (pre-resize) image device buffer for the GPU resize path
    rt::Tensor mResizeTmpDevice{}; //!< Float scratch (horizontal pass) for the GPU resize
};

} // namespace rt
} // namespace trt_edgellm
