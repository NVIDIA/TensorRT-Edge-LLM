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

#include "nvfp4A16BlackwellMoeJitCompiler.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace trt_edgellm
{

struct Nvfp4A16BlackwellMoeLoadedModule;

//! Loads one compiled MoE JIT bundle (context-keyed module registry, shared
//! between plugin clones) and launches its seven kernels through the driver API.
//! Every launch takes ``enablePdl``: when set, the kernel is launched with
//! programmatic stream serialization (the kernels carry the griddepcontrol
//! wait/trigger unconditionally).  Launches never compile or query the device,
//! so they are CUDA-graph capture safe once load() has run.
class Nvfp4A16BlackwellMoeJitRunner
{
public:
    void load(Nvfp4A16BlackwellMoeJitKernel const& kernel);

    bool isLoaded() const noexcept
    {
        return mLoadedModule != nullptr;
    }

    Nvfp4A16BlackwellMoeJitKey const& getKey() const noexcept
    {
        return mKey;
    }

    //! Warp-per-token sigmoid top-k routing (ungrouped contract).
    void launchRoute(float const* logits, float const* correctionBias, int32_t numTokens, bool normTopkProb,
        float routedScalingFactor, int32_t* topkIndices, float* topkWeights, bool enablePdl, cudaStream_t stream) const;

    //! Single-CTA expert-contiguous tile layout of the numSlots routed rows.
    void launchLayout(int32_t const* topkIndices, int32_t numSlots, int32_t tokenTile, int32_t* permutedIdx,
        int32_t* tileGroupIdx, int32_t* numValidTiles, bool enablePdl, cudaStream_t stream) const;

    //! Permuted-row gather plus output zeroing for the grouped GEMM path.
    void launchGather(void const* hiddenStates, int32_t const* permutedIdx, int32_t const* numValidTiles,
        int32_t tokenTile, int32_t numTokens, int64_t maxRowsPadded, void* permutedActivations, void* output,
        bool enablePdl, cudaStream_t stream) const;

    //! Decode FC1 (+ its split-K reduce when the key's fc1SplitK > 1).
    void launchFc1(void const* hiddenStates, int32_t const* topkIndices, void const* qweights, void const* blockScales,
        float const* globalScales, void* fc1Output, float* partials, int32_t numTokens, bool enablePdl,
        cudaStream_t stream) const;

    //! Decode FC2 (+ its split-K reduce when the key's fc2SplitK > 1).
    void launchFc2(void const* fc1Output, int32_t const* topkIndices, float const* topkWeights, void const* qweights,
        void const* blockScales, float const* globalScales, void* output, float* partials, int32_t numTokens,
        bool enablePdl, cudaStream_t stream) const;

private:
    Nvfp4A16BlackwellMoeJitKey mKey{};
    Nvfp4A16BlackwellMoeJitDigest mDigest{};
    std::shared_ptr<Nvfp4A16BlackwellMoeLoadedModule> mLoadedModule;
};

} // namespace trt_edgellm
