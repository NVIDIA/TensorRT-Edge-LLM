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

#include "multimodal/qwen2/qwenViTRunner.h"

namespace trt_edgellm
{
namespace rt
{

//! \brief Qwen3-VL / Qwen3.5 vision encoder. Extends the Qwen2-VL base with FAST-POS-EMBED + DEEPSTACK, a 3D
//!        temporal resize budget, and per-frame timestamped video sub-spans. Mirror of HF modeling_qwen3_vl.py.
class Qwen3VLViTRunner : public QwenViTRunner
{
public:
    using QwenViTRunner::QwenViTRunner;

    rt::OptionalInputTensors getDeepstackFeatures() override;

protected:
    bool validateExtraConfig(nlohmann::json const& jsonConfig) override;
    bool allocateExtraBuffers(int64_t maxImageTokens) override;
    void buildExtraInputs(std::vector<VisionSpan> const& spans, int64_t totalSeqLength, int64_t totalImageTokens,
        cudaStream_t stream) override;
    bool bindExtraInputShapes() override;

    //! Video splits into per-frame sub-spans (each vit.gridT==1), unlike the base single flat span.
    std::tuple<int64_t, int64_t> computeVisionSpans(
        rt::imageUtils::ImageData const& image, int64_t patchBase, std::vector<VisionSpan>& spans) override;

    //! 3D temporal-aware smart-resize budget (t_bar = ceil(N/TPS)*TPS), vs the base 2D per-frame budget.
    std::tuple<int64_t, int64_t> getResizedImageSize(
        int64_t numFrames, bool isVideo, int64_t height, int64_t width, int64_t maxRatio = 200) override;

    //! Adds the Qwen3-VL video path: the <|vision_start|><|video_pad|><|vision_end|> triplet expands to one
    //! timestamped (<X.X s> + vision_start + pads + vision_end) group per per-frame sub-span; images stay flat.
    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchInputIds,
        std::vector<VisionSpan> const& spans, std::vector<int64_t> const& spansPerRequest,
        trt_edgellm::tokenizer::Tokenizer const* tokenizer) override;

    int64_t mNumGridPerSide{0};                   //!< sqrt(num_position_embeddings) for fast position embedding
    int64_t mNumDeepstackFeatures{0};             //!< deepstack_visual_indexes count (0 for Qwen3.5 -> no deepstack)
    rt::Tensor mFastPosEmbIdx{};                  //!< Fast position embeddings index tensor
    rt::Tensor mFastPosEmbWeight{};               //!< Fast position embeddings weight tensor
    std::vector<rt::Tensor> mDeepstackFeatures{}; //!< Deepstack features tensors (empty for Qwen3.5)
};

} // namespace rt
} // namespace trt_edgellm
