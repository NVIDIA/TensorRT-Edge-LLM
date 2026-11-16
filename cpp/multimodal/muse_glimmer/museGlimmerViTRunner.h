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

#include "multimodal/qwen2/qwenViTRunner.h"

namespace trt_edgellm
{
namespace rt
{

//! \brief Muse-Glimmer vision encoder: window + full attention with a 2D
//!        interleaved rotary and a fast learned position embedding, run in
//!        raster token order (spatial_merge_size == 1); the merge_size == 2
//!        pixel-shuffle is folded into reverse_window_index at the output.
class MuseGlimmerViTRunner : public QwenViTRunner
{
public:
    using QwenViTRunner::QwenViTRunner;

protected:
    bool validateAndFillConfig(std::string const& engineDir) override;
    bool validateExtraConfig(nlohmann::json const& jsonConfig) override;
    bool allocateExtraBuffers(int64_t maxImageTokens) override;
    void buildExtraInputs(std::vector<VisionSpan> const& spans, int64_t totalSeqLength, int64_t totalImageTokens,
        cudaStream_t stream) override;
    bool bindExtraInputShapes() override;

    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchInputIds,
        std::vector<VisionSpan> const& spans, std::vector<int64_t> const& spansPerRequest,
        trt_edgellm::tokenizer::Tokenizer const* tokenizer) override;

    std::tuple<int64_t, int64_t> getResizedImageSize(
        int64_t numFrames, bool isVideo, int64_t height, int64_t width, int64_t maxRatio = 200) override;

    int64_t vitInputMergeSize() const override;
    bool vitPatchTemporalFirst() const override;

    //! 2D rotary in raster order; per-token row = concat(freq_w, freq_h).
    void buildRotaryPosEmb(std::vector<VisionSpan> const& spans, cudaStream_t stream) override;

    //! Compute per-token window_index + cu_window_seqlens and the
    //! pixel-shuffle-folded reverse_window_index. \param curHW == totalSeqLength.
    void getWindowIndex(std::vector<VisionSpan> const& spans, int64_t curHW, cudaStream_t stream);

    int64_t mNumGridPerSide{0}; //!< Side length of the learned position table (pos_emb_height)
    int64_t mWindowSizePx{0};   //!< Window attention size in pixels
    int64_t mMaxImageTokens{4096};
    int64_t mMaxVideoFrameTokens{144};
    float mVitRopeTheta{10000.0f}; //!< Vision rotary theta

    rt::Tensor mCuWindowSeqlensHost{};
    rt::Tensor mCuWindowSeqlens{};
    rt::Tensor mWindowIndexHost{};
    rt::Tensor mWindowInverseHost{};
    rt::Tensor mWindowIndexDevice{};
    rt::Tensor mReverseWindowIndexHost{};
    rt::Tensor mReverseWindowIndexDevice{};

    rt::Tensor mFastPosEmbIdx{};    //!< Fast position-embedding index [4, T]
    rt::Tensor mFastPosEmbWeight{}; //!< Fast position-embedding weight [4, T]
};

} // namespace rt
} // namespace trt_edgellm
