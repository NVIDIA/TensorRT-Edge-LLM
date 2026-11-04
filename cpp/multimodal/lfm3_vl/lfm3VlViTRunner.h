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

//! \brief LFM3-VL vision encoder: gated depthwise-conv and bidirectional-attention tower plus projector.
//!        Shares the Qwen2-VL preprocessing (smart resize, 2x2 merge-block patch order, cu_seqlens, axial 2D
//!        rotary) and adds the conv_neighbor_index input. The LFM3 text model uses 1D RoPE, so no MRoPE is built.
class Lfm3VlViTRunner : public QwenViTRunner
{
public:
    using QwenViTRunner::QwenViTRunner;

protected:
    //! Reads the vision rotary theta from vision_config.rope_parameters.
    bool validateExtraConfig(nlohmann::json const& jsonConfig) override;
    bool usesTextMRope() const override;
    void buildRotaryPosEmb(std::vector<VisionSpan> const& spans, cudaStream_t stream) override;
    bool allocateExtraBuffers(int64_t maxImageTokens) override;
    void buildExtraInputs(std::vector<VisionSpan> const& spans, int64_t totalSeqLength, int64_t totalImageTokens,
        cudaStream_t stream) override;
    bool bindExtraInputShapes() override;

private:
    float mVitRopeTheta{0.0F};       //!< Vision axial rotary theta
    rt::Tensor mConvNeighborIndex{}; //!< Device [totalSeqLength, 9] conv_neighbor_index, built on the GPU
};

} // namespace rt
} // namespace trt_edgellm
