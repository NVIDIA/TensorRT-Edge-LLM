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

#include "multimodal/qwen3/qwen3vlViTRunner.h"

namespace trt_edgellm
{
namespace rt
{

//! \brief Qwen3-Omni vision encoder. Mirror of HF modeling_qwen3_omni.py, whose ViT IS Qwen3-VL's (fast-pos-emb +
//!        deepstack — inherited unchanged) but whose preprocessing/MRoPE follow Qwen2.5-VL: a flat video span, 2D
//!        resize, constant pad fill, position_id_per_seconds, and an MRoPE advance that also counts the T-extent.
class Qwen3OmniViTRunner : public Qwen3VLViTRunner
{
public:
    using Qwen3VLViTRunner::Qwen3VLViTRunner;

protected:
    //! Flat single span (not Qwen3-VL's per-frame sub-spans) — delegates to the base (Qwen2-VL) flat layout.
    std::tuple<int64_t, int64_t> computeVisionSpans(
        rt::imageUtils::ImageData const& image, int64_t patchBase, std::vector<VisionSpan>& spans) override;

    //! 2D resize budget (unlike Qwen3-VL's 3D) — delegates to the base (Qwen2-VL) getResizedImageSize.
    std::tuple<int64_t, int64_t> getResizedImageSize(
        int64_t numFrames, bool isVideo, int64_t height, int64_t width, int64_t maxRatio = 200) override;
    bool validateExtraConfig(nlohmann::json const& jsonConfig) override;

    //! HF keeps this family's temporal positions in float (`(arange(t) * spg).float() * pips`).
    bool usesFractionalMRopePositions() const override;

    //! Mirrors HF Qwen3-Omni get_rope_index: temporal step = secondPerGrid * position_id_per_seconds; the span
    //! advance also counts the temporal extent (its large position_id_per_seconds can make T dominate).
    void getMRopePositionIds(std::vector<std::vector<int32_t>> const& batchInputIds,
        std::vector<VisionSpan> const& spans, std::vector<int64_t> const& spansPerRequest) noexcept override;

    //! Fills visual pads with a constant imageTokenId (embeddingLookup), vs the base's incrementing IDs.
    //! Omni is flat-only (no timestamped sub-spans), so this is a simpler walk than the base's.
    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchInputIds,
        std::vector<VisionSpan> const& spans, std::vector<int64_t> const& spansPerRequest,
        trt_edgellm::tokenizer::Tokenizer const* tokenizer) override;

    int64_t mPositionIdPerSecond{0}; //!< Position-ids/second for fps-aware video temporal MRoPE
};

} // namespace rt
} // namespace trt_edgellm
