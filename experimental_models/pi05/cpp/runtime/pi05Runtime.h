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

#include "action/pi05ActionRunner.h"
#include "common/pi05Common.h"
#include "common/tensor.h"
#include "multimodal/pi05VisualRunner.h"

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace pi05
{

//! \brief Stream time of each stage of the last generate() call, in ms.
//! Taken from events the request already had to drain, so collecting them adds
//! no synchronization to the timed path.
struct Pi05StageTimes
{
    float visualMs{0.0F};
    float assembleMs{0.0F};
    float prefixMs{0.0F};
    float actionMs{0.0F};
};

//! \brief pi0.5 policy head: SigLIP -> PaliGemma prefix -> action expert.
//!
//! Owns the prefix tower and the K/V cache both towers share, and drives the vision tower
//! and the action expert through their own runners. The prefix stays here because at
//! batch > 1 its K/V is scattered into the packed halves, which only the cache's owner can do.
//!
//! NOT thread-safe, single-stream: the staging buffers are reused, and safe to refill only
//! after the stream drains.
class Pi05Runtime
{
public:
    //! \param engineDir Directory holding visual/, prefix/ and action/ subdirectories, plus
    //!        cond/ when the export hoisted the AdaRMS modulation (the default).
    //! \param stream The instance's only stream, for allocation-time work and every request.
    Pi05Runtime(std::string const& engineDir, cudaStream_t stream);
    ~Pi05Runtime() noexcept;

    Pi05PolicyConfig const& getConfig() const noexcept
    {
        return mConfig;
    }

    //! \brief Per-stage timing of the last generate().
    Pi05StageTimes const& getStageTimes() const noexcept
    {
        return mStageTimes;
    }

    //! \brief Set the random seed for the initial noise trajectory.
    void setNoiseSeed(int32_t seed) noexcept
    {
        mActionRunner->setNoiseSeed(seed);
    }

    //! \brief Use an externally supplied x_0 instead of seeding one locally.
    //! Required for numerical comparison: the host RNGs on the two sides of the
    //! comparison do not produce identical draws from the same seed.
    //! \param noise Row-major [B, actionHorizon, actionDim].
    void setInitialNoise(std::vector<float> noise)
    {
        mActionRunner->setInitialNoise(std::move(noise));
    }

    //! \brief Capture the denoise loop into a CUDA graph on first use.
    void setUseCudaGraph(bool enable) noexcept
    {
        mActionRunner->setUseCudaGraph(enable);
    }

    //! \brief Override the denoise step count for this process (default: contract value).
    void setNumDenoiseSteps(int32_t steps)
    {
        mActionRunner->setNumDenoiseSteps(steps);
    }

    //! \brief Build the prefix the graph expects: image features followed by token
    //! embeddings scaled by sqrt(hidden_size), replicated across \p batch.
    //!
    //! The scale is applied here and nowhere else: openpi applies it in ``embed_prefix`` and
    //! comments out the tower's own normalizer, so applying it twice shrinks every language
    //! embedding.
    //!
    //! \param imageFeatures Output of the vision tower; its rows lead the prefix.
    //! \param tokenIds Language token ids, already compacted -- no padding.
    //! \return Device FLOAT16 [batch, numImageTokens + tokenIds.size(), hiddenSize].
    rt::Tensor const& assemblePrefix(
        rt::Tensor const& imageFeatures, std::vector<int32_t> const& tokenIds, int32_t batch);

    //! \brief Run the prefix tower once, retaining its per-layer K/V for the loop.
    //! \param inputsEmbeds Device FLOAT16 [B, prefixLen, hiddenSize]; must be COMPACT --
    //!        no missing-camera or language padding, since the graph carries no mask.
    bool prefill(rt::Tensor const& inputsEmbeds);

    int32_t getPrefixLen() const noexcept
    {
        return mPrefixLen;
    }

    //! \brief One whole request: vision tower, prefix assembly, prefill, denoise loop.
    //! \return Row-major [B, actionHorizon, actionDim], normalized and padded --
    //!         the policy layer converts it to robot-native units. Empty on error.
    std::vector<float> generate(rt::Tensor const& pixelValues, std::vector<int32_t> const& tokenIds, int32_t batch);

private:
    //! Size one user-managed scratch pool at the largest component requirement and bind it
    //! to every context. Must run before the first enqueue and before any graph capture.
    void allocateSharedContextMemory();
    void loadEmbedTable(std::string const& engineDir);
    void parseConfigs(std::string const& engineDir);
    void allocateTensors();
    //! Move the batched prefix K/V into the per-request halves of the packed cache.
    void scatterPrefixKV();
    size_t vPlaneOffsetBytes() const noexcept;
    //! Stage boundary \p index on mStream. The runners mark their own spans, so these
    //! only bracket the two stages the runtime runs itself.
    void markStage(int32_t index) noexcept;
    void collectStageTimes() noexcept;
    //! Every batch entry ran the same request, so a spread between them is a
    //! batch-axis defect rather than a numerical one.
    void logBatchSpread(std::vector<float> const& actions, int32_t batch) const;

    Pi05PolicyConfig mConfig{};
    cudaStream_t mStream{nullptr};
    //! Prefix-assembly start, then one per stage the runtime runs itself.
    static constexpr int32_t kNumStageEvents = 3;
    cudaEvent_t mStageEvents[kNumStageEvents]{};
    Pi05StageTimes mStageTimes{};
    int32_t mMaxBatch{1};
    int32_t mActiveBatch{1};
    int32_t mPrefixLen{0};

    //! Scratch shared by every component; its address is baked into the captured
    //! denoise graph, so it is allocated once and never resized. Declared ahead of
    //! everything that binds it, so it outlives those contexts and that graph.
    rt::Tensor mSharedContextMemory;

    std::unique_ptr<Pi05VisualRunner> mVisualRunner;
    std::unique_ptr<Pi05ActionRunner> mActionRunner;
    Pi05Component mPrefix;

    rt::Tensor mInputsEmbeds;
    //! [vocabSize, hiddenSize], staged beside the engines. Loaded once: it is ~1 GB and
    //! a request gathers a handful of rows from it.
    rt::Tensor mEmbedTable;
    rt::Tensor mTokenIds;     //!< [1, len] int32, the ids kernel::embeddingLookup gathers by
    rt::Tensor mTokenIdsHost; //!< pinned staging for mTokenIds
    //! Paged pool per layer, [2, maxBatch * pagesPerSeq, tokensPerPage, numKVHeads, headDim],
    //! plane-major: every request's K pages, then every request's V pages. At batch 1 the
    //! prefix engine writes its two outputs straight into the two planes and the expert's
    //! plugin appends the action tokens, so there is no repack between the towers.
    std::vector<rt::Tensor> mKVCache;
    //! 2 * numHiddenLayers buffers of [maxBatch, maxPrefixLen, numKVHeads, headDim],
    //! K then V per layer. The prefix graph packs its output across requests at the
    //! prefix length while the cache strides by capacity, so batched runs land here
    //! first; empty when the engines were built for batch 1.
    std::vector<rt::Tensor> mPrefixKVStaging;

    //! [maxPrefixLen + actionHorizon, headDim] packed (cos|sin) for every position.
    rt::Tensor mRopeCache;
    rt::Tensor mPrefixRopeCosSin;
    rt::Tensor mPrefixPosIds;
    rt::Tensor mPrefixPosIdsHost;
};

} // namespace pi05
} // namespace trt_edgellm
