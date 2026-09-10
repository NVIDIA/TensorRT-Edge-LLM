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

#include "runtime/preprocess/visualTokenPruner.h"

namespace trt_edgellm
{
namespace rt
{

//! DART visual-token pruner, based on the paper "Stop Looking for Important Tokens in
//! Multimodal Language Models: Duplication Matters More" and registered as the default "dart".
//!
//! Each visual span keeps a proportional quota. Its highest-L1 image tokens and shared text
//! tokens are pivots; remaining tokens with the lowest cosine similarity to those pivots are kept.
//!
//! This embedding-level variant uses input embeddings because the paper's decoder-layer-2
//! states are unavailable inside the monolithic engine. See the user guide for validation.
//!
//! CUDA kernels compute row norms and batched pivot similarities; greedy selection runs on the host.
class DartPruner final : public VisualTokenPruner
{
public:
    //! Preallocates the selection work buffers from the engine limits.
    //! \throws std::runtime_error on invalid pivot configuration.
    DartPruner(VisualPrunerConfig const& config, LLMEngineConfig const& engineConfig);

    char const* name() const noexcept override
    {
        return "dart";
    }

protected:
    int32_t prune(PruneRequest const& req, PipelineIO& io, cudaStream_t stream) override;

private:
    //! Fills mRetained with the kept visual-token positions: norms + all pivots once, then a
    //! batched pivot-similarity pass covering as many spans as fit per launch (typically all),
    //! followed by per-span greedy growth (each span keeps its own quota).
    void selectRetainedImageTokens(PruneRequest const& req, cudaStream_t stream);

    //! Runs the greedy anti-duplication growth for one image span using the chunk's dots
    //! matrix (span pivots at spanPivotOffset, shared text pivots at textPivotOffset),
    //! appending the span's retained positions to mRetained.
    void growSpanGreedily(PruneRequest const& req, ImageSpan const& span, int32_t spanPivotOffset,
        int32_t numSpanPivots, int32_t textPivotOffset, int32_t numTextPivots, float const* norms);

    int32_t mPivotImageTokens;
    int32_t mPivotTextTokens;
    std::vector<int32_t> mRetained; //!< per-request retained positions scratch (reused)
    std::vector<float> mL1Scores;   //!< per-request L1 norms scratch (reused)
    std::vector<char> mInRetained;  //!< per-span retained-membership scratch (reused)
    std::vector<float> mSimScores;  //!< per-span similarity scratch (reused)
    Tensor mNormsDevice;            //!< [maxInputLen, 2] FP32 (L1, L2 per row)
    Tensor mNormsHost;              //!< pinned mirror
    Tensor mDotsDevice;             //!< [kDartMaxPivots, maxInputLen] FP32
    Tensor mDotsHost;               //!< pinned mirror
    Tensor mPivotIdxDevice;         //!< [kDartMaxPivots] INT32
    Tensor mPivotIdxHost;           //!< pinned mirror
};

} // namespace rt
} // namespace trt_edgellm
