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

//! \brief Cosmos3-Edge reasoner vision encoder: SigLIP2 (packed patches) + Qwen3-VL-style PatchMerger.
//!        Follows the qwen3_vl visual engine I/O convention at temporalPatchSize == 1 (per-frame Linear
//!        patch embed, per-frame vision spans/timestamps, learned pos-emb via the fast-pos-emb inputs,
//!        no rotary input, no deepstack). It reuses the Qwen3-VL runner contract while overriding the patch layout
//!        and learned-position interpolation to match Cosmos3.
class Cosmos3EdgeViTRunner : public Qwen3VLViTRunner
{
public:
    using Qwen3VLViTRunner::Qwen3VLViTRunner;

    //! Base load + SigLIP2 sanity: temporalPatchSize must be 1 and the engine's packed-patch input dim must
    //! equal the Linear patch-embed width 3 * patchSize * patchSize.
    bool validateAndFillConfig(std::string const& engineDir) override;

protected:
    //! Reads the pos-emb grid from vision_config.num_patches (SigLIP2 naming) instead of Qwen3-VL's
    //! num_position_embeddings; no deepstack; interleaved mRoPE.
    bool validateExtraConfig(nlohmann::json const& jsonConfig) override;

    //! Cosmos3 resizes its learned 2D position table with bilinear
    //! align_corners=false coordinates, unlike the inherited Qwen3-VL path.
    void buildExtraInputs(std::vector<VisionSpan> const& spans, int64_t totalSeqLength, int64_t totalImageTokens,
        cudaStream_t stream) override;

    //! Cosmos3 patchifies visual inputs as [H,W,C,T], matching its Hugging Face processor.
    bool vitPatchChannelLast() const override
    {
        return true;
    }

    //! SigLIP2 uses full attention with a learned pos-emb — the visual engine has no rotary_pos_emb input.
    bool usesRotaryPosEmb() const override
    {
        return false;
    }
};

} // namespace rt
} // namespace trt_edgellm
