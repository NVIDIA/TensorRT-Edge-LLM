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

#include "common/pi05Common.h"
#include "common/tensor.h"

#include <cuda_runtime.h>
#include <random>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace pi05
{

//! \brief The pi0.5 action expert: the flow-matching denoise loop over a K/V cache the
//! prefix tower has already filled, plus the cond component that hoists the AdaRMS
//! modulation out of that loop.
//!
//! The capture bakes in the batch, the step count and the K/V cache addresses, so generate()
//! drops a stale graph when any of those move. The prefix length is not among them: it
//! reaches only buffers that keep their addresses and are restaged before every launch.
//!
//! Single-stream by construction, NOT thread-safe.
class Pi05ActionRunner
{
public:
    //! \param engineDir Engine root; the expert is loaded from its action/ subdirectory, and
    //!        cond/ as well when the export hoisted the AdaRMS modulation.
    //! \param config The parsed contract of the export \p engineDir holds.
    //! \param stream The instance's only stream, for allocation-time work and every request.
    Pi05ActionRunner(std::string const& engineDir, Pi05PolicyConfig const& config, cudaStream_t stream);
    ~Pi05ActionRunner() noexcept;

    int64_t getRequiredContextMemorySize() const;
    bool setContextMemory(rt::Tensor& pool);

    //! \brief Request batch the expert's profile admits.
    int32_t getMaxBatch() const noexcept
    {
        return mMaxBatch;
    }

    //! \brief Reseed the generator that draws the initial noise trajectory.
    //! It advances across requests, as openpi draws a fresh x_0 per inference;
    //! reseeding is what makes a run reproducible.
    void setNoiseSeed(int32_t seed) noexcept
    {
        mNoiseGen.seed(static_cast<std::mt19937::result_type>(seed));
    }

    //! \brief Use an externally supplied x_0 instead of seeding one locally.
    //! Required for numerical comparison: the host RNGs on the two sides of the
    //! comparison do not produce identical draws from the same seed.
    //! \param noise Row-major [B, actionHorizon, actionDim].
    void setInitialNoise(std::vector<float> noise);

    //! \brief Capture the denoise loop into a CUDA graph on first use.
    //! Valid because every step enqueues the same engine with fixed shapes and
    //! addresses; only buffer *contents* change, updated in place on device.
    void setUseCudaGraph(bool enable) noexcept;

    //! \brief Override the denoise step count for this process (default: contract value).
    void setNumDenoiseSteps(int32_t steps);

    //! \brief Run the flow-matching loop and return the raw action chunk.
    //! \param kvCache Per-layer paged [2, numPages, tokensPerPage, numKVHeads, headDim]
    //!        pool the prefix tower filled; the expert's plugin appends its own tokens
    //!        in place, under the identity page table the runtime owns.
    //! \param batch Request batch the cache was filled at.
    //! \param prefixLen Prefix tokens already in the cache; the expert's rows follow them.
    //! \return Row-major [B, actionHorizon, actionDim], normalized and padded --
    //!         the policy layer converts it to robot-native units. Empty on error.
    std::vector<float> generate(std::vector<rt::Tensor>& kvCache, int32_t batch, int32_t prefixLen);

    //! \brief Stream time of the last generate(), in ms.
    float getElapsedMs() const noexcept;

private:
    void allocateTensors();
    //! Build the request-invariant paged-attention inputs (identity page table,
    //! all-ones mask, uniform query geometry, tree-decoding phase carrier).
    void allocatePagedInputs(int32_t maxBatch);
    //! Fill the identity page table for a pool of \p numPages pages per plane.
    //! The V plane's ids are the K ids offset by that count.
    bool refreshPageTable(int64_t numPages);
    void initializeNoise(int32_t batch);
    //! Run the cond engine over the whole 1 -> 0 schedule, filling mAdarmsMod.
    //! The schedule is a function of the step count and the batch alone -- no observation
    //! and no noise -- so this re-runs only when one of those two changes.
    bool computeModulation();
    //! Drop a captured denoise graph; anything that changes what the capture baked in
    //! must call this or the next launch replays a stale graph.
    void invalidateGraph() noexcept;
    //! Whether a captured graph would replay K/V addresses \p kvCache no longer holds.
    bool capturedKVCacheMoved(std::vector<rt::Tensor> const& kvCache) const noexcept;

    Pi05PolicyConfig mConfig;
    cudaStream_t mStream{nullptr};
    Pi05Component mAction;
    Pi05Component mCond; //!< hoisted-modulation exports only

    cudaEvent_t mBegin{nullptr};
    cudaEvent_t mEnd{nullptr};

    int32_t mMaxBatch{1};
    int32_t mActiveBatch{1};
    int32_t mPrefixLen{0};
    std::mt19937 mNoiseGen{0};
    std::vector<float> mExternalNoise;
    bool mUseCudaGraph{false};
    bool mGraphReady{false};
    cudaGraph_t mGraph{nullptr};
    cudaGraphExec_t mGraphExec{nullptr};
    //! The K/V addresses mGraph baked in; empty while no graph is captured.
    std::vector<void const*> mGraphKVCache;
    int32_t mNumDenoiseSteps{0};
    //! Step count and batch mAdarmsMod currently holds; 0 until the cond engine has run.
    int32_t mModulationSteps{0};
    int32_t mModulationBatch{0};

    rt::Tensor mKVSeqLens;       //!< [B] int32, prefixLen + actionHorizon
    rt::Tensor mKVSeqLensHost;   //!< pinned staging for mKVSeqLens
    rt::Tensor mKVPageTableHost; //!< pinned staging for mKVPageTable

    //! Paged-pool exports only. Request-invariant once built: an identity page table,
    //! an all-ones mask making the action tokens mutually visible, and the start index
    //! the tree-decoding path leaves unread.
    rt::Tensor mKVPageTable;
    rt::Tensor mAttentionMask;
    rt::Tensor mKVCacheStartIdx;
    int64_t mPageTableNumPages{-1};

    //! Token-major query geometry. Every request contributes exactly one action
    //! horizon of rows, so both are staged once and never restaged.
    rt::Tensor mQueryLengths;          //!< [B] int32, actionHorizon
    rt::Tensor mQueryLengthsHost;      //!< pinned staging for mQueryLengths
    rt::Tensor mQueryStartOffsets;     //!< [B + 1] int32, slot * actionHorizon
    rt::Tensor mQueryStartOffsetsHost; //!< pinned staging for mQueryStartOffsets

    //! Shape-only carriers: the plugin reads the bound extent, never the payload.
    rt::Tensor mPhaseMarker; //!< extent = rt::ExecutionPhase::kDiffusionDenoise

    //! [maxPrefixLen + actionHorizon, headDim] packed (cos|sin) for every position; the
    //! expert reads the rows the prefix did not.
    rt::Tensor mRopeCache;
    rt::Tensor mActionRopeCosSin;
    rt::Tensor mActionPosIds;
    rt::Tensor mActionPosIdsHost;

    rt::Tensor mNoiseDevice; //!< current x_t, FLOAT32 [B, H, actionDim]
    rt::Tensor mNoiseHost;
    rt::Tensor mPredDevice; //!< velocity v_t from the action engine
    rt::Tensor mTimestepDevice;
    rt::Tensor mTimestepHost;

    //! [maxDenoiseSteps, maxBatch, numAdarmsSites, modulationDim] fp16: the expert reads
    //! one [B, sites, dim] slice per step, so each step's row is stored once per request.
    rt::Tensor mAdarmsMod;
    rt::Tensor mCondTimestepDevice; //!< the whole schedule, FLOAT32 [maxDenoiseSteps * maxBatch]
    rt::Tensor mCondTimestepHost;
};

} // namespace pi05
} // namespace trt_edgellm
