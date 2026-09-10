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

//! \brief Qwen2.5-VL vision encoder. Extends the Qwen2-VL base with WINDOW ATTENTION and fps-scaled video MRoPE
//!        (tokens_per_second); 2D resize, incrementing pad — all inherited. Mirror of HF modeling_qwen2_5_vl.py.
class Qwen25VLViTRunner : public QwenViTRunner
{
public:
    using QwenViTRunner::QwenViTRunner;

protected:
    bool validateExtraConfig(nlohmann::json const& jsonConfig) override;
    bool allocateExtraBuffers(int64_t maxImageTokens) override;
    void buildExtraInputs(std::vector<VisionSpan> const& spans, int64_t totalSeqLength, int64_t totalImageTokens,
        cudaStream_t stream) override;
    bool bindExtraInputShapes() override;

    //! Flat span (base layout) + the fps-derived secondPerGrid this model's MRoPE needs.
    std::tuple<int64_t, int64_t> computeVisionSpans(
        rt::imageUtils::ImageData const& image, int64_t patchBase, std::vector<VisionSpan>& spans) override;

    //! Mirrors HF Qwen2_5_VLModel.get_rope_index: temporal step = secondPerGrid * tokens_per_second; spatial advance.
    void getMRopePositionIds(std::vector<std::vector<int32_t>> const& batchInputIds,
        std::vector<VisionSpan> const& spans, std::vector<int64_t> const& spansPerRequest) noexcept override;

    //! \brief Compute window indices + cu_window_seqlens for window attention (each span's gridT frames).
    void getWindowIndex(std::vector<VisionSpan> const& spans, int64_t curHW, cudaStream_t stream);

    int64_t mWindowSize{0};      //!< Window attention size (vision_config.window_size)
    int64_t mTokensPerSecond{0}; //!< Tokens/second for fps-aware video temporal MRoPE (vision_config.tokens_per_second)
    rt::Tensor mCuWindowSeqlensHost{};      //!< Cumulative window sequence lengths host tensor
    rt::Tensor mCuWindowSeqlens{};          //!< Cumulative window sequence lengths device tensor
    rt::Tensor mWindowIndexHost{};          //!< Window index host tensor for window attention
    rt::Tensor mWindowIndexDevice{};        //!< Window index device tensor for window attention
    rt::Tensor mReverseWindowIndexHost{};   //!< Reverse window index host tensor
    rt::Tensor mReverseWindowIndexDevice{}; //!< Reverse window index device tensor
};

} // namespace rt
} // namespace trt_edgellm
