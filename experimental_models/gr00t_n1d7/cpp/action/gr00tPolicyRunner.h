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

#include "common/tensor.h"
#include "common/trtUtils.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace gr00t
{

//! \brief Contract parsed from the component config.json files. Every field is
//! required; a missing key fails engine load rather than running on a guess.
struct Gr00tConfig
{
    // vl_backbone
    int32_t numHiddenLayers{0};
    int32_t hiddenSize{0};
    int32_t headDim{0};
    int32_t numDeepstackFeatures{0};
    int32_t maxVlLen{0};
    //! Even DiT blocks cross-attend; the two lists partition them by which span
    //! of the VL context they read.
    std::vector<int32_t> textCrossBlocks;
    std::vector<int32_t> imageCrossBlocks;
    std::vector<int32_t> crossBlocks;
    int32_t ditNumHeads{0};
    int32_t ditHeadDim{0};

    // action
    int32_t actionHorizon{0};
    int32_t actionDim{0};
    int32_t stateDim{0};
    int32_t stateHistoryLength{1};
    int32_t numDenoiseSteps{0};
    int32_t numModulationSites{0};
    int32_t modulationDim{0};
    int32_t innerDim{0};
};

//! \brief GR00T N1.7 policy head: VL backbone prefill, then the DiT denoise loop.
//! The backbone emits the step-invariant cross-attention K/V, bound for the whole loop;
//! the Euler update runs on the device, so the loop captures as one CUDA graph.
class Gr00tPolicyRunner
{
public:
    //! \param engineDir Directory holding vl_backbone/ and action/.
    Gr00tPolicyRunner(std::string const& engineDir, cudaStream_t stream);
    ~Gr00tPolicyRunner() noexcept;

    Gr00tConfig const& getConfig() const noexcept
    {
        return mConfig;
    }

    //! \brief Use an externally supplied x_0 instead of seeding one locally.
    //! Required for numerical comparison: two host RNGs do not agree on a seed.
    //! \param noise Row-major [B, actionHorizon, actionDim].
    void setInitialNoise(std::vector<float> noise);

    //! \brief Restart the x_0 stream; each generate() draws fresh noise from it.
    void setNoiseSeed(int32_t seed) noexcept
    {
        mNoiseGenerator.seed(static_cast<uint32_t>(seed));
    }

    //! \brief Capture the denoise loop into a CUDA graph on first use.
    void setUseCudaGraph(bool enable) noexcept
    {
        if (mUseCudaGraph != enable)
        {
            invalidateGraph();
        }
        mUseCudaGraph = enable;
    }

    //! \brief Run the VL backbone once, retaining the DiT cross-attention K/V.
    //! \param inputsEmbeds Device FLOAT16 [B, vlLen, hiddenSize].
    //! \param deepstackEmbeds ``numDeepstackFeatures`` tensors of the same shape, zero
    //!        outside image-token positions: the graph adds them unconditionally.
    //! \param ropeCosSin Device FLOAT32 [B, vlLen, headDim], packed (cos|sin).
    //! \throws std::runtime_error If the request does not fit the engines or the enqueue fails.
    void prefill(rt::Tensor const& inputsEmbeds, std::vector<rt::Tensor> const& deepstackEmbeds,
        rt::Tensor const& ropeCosSin, rt::Tensor const& posIds, int32_t batch, int32_t vlLen, cudaStream_t stream);

    //! \brief Bind the two cross-attention masks over the prefilled VL context.
    //! Each is [B, 1, 1, vlLen]; true means the DiT may attend to that token.
    void setCrossMasks(
        std::vector<uint8_t> const& textMask, std::vector<uint8_t> const& imageMask, cudaStream_t stream);

    //! \brief Set the state vector, row-major [B, stateHistoryLength, stateDim].
    void setState(std::vector<float> state, cudaStream_t stream);

    //! \brief Run the flow-matching loop and return the raw action chunk.
    //! \return Row-major [B, actionHorizon, actionDim] in the model's normalized, zero-padded space.
    //! \throws std::runtime_error If prefill() has not run or an enqueue fails.
    std::vector<float> generate(cudaStream_t stream);

private:
    struct Engine
    {
        std::unique_ptr<nvinfer1::IRuntime> runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> engine;
        std::unique_ptr<nvinfer1::IExecutionContext> context;
        AuxStreamSet auxStreams; //!< Non-blocking, or the loop cannot be captured.
    };

    void loadComponent(std::string const& engineDir, std::string const& component, Engine& out);
    void parseConfigs(std::string const& engineDir);
    void loadConstantTables(std::string const& engineDir, cudaStream_t stream);
    void allocateTensors();
    void initializeNoise(int32_t batch);
    //! Drop a captured denoise graph. The capture bakes in the step count and the
    //! bound addresses and shapes; anything that changes those must call this.
    void invalidateGraph() noexcept;

    Gr00tConfig mConfig{};
    std::mt19937 mNoiseGenerator{0U};
    std::vector<float> mExternalNoise;
    bool mUseCudaGraph{false};
    bool mGraphReady{false};
    cudaGraph_t mGraph{nullptr};
    cudaGraphExec_t mGraphExec{nullptr};
    int32_t mMaxBatch{1};
    int32_t mActiveBatch{1};
    int32_t mVlLen{0};

    Engine mBackbone;
    Engine mAction;

    //! One [B, maxVlLen, ditHeads, ditHeadDim] buffer per cross block and half; the
    //! backbone writes them and the action engine reads them in place.
    std::vector<rt::Tensor> mCrossKey;
    std::vector<rt::Tensor> mCrossValue;

    rt::Tensor mTextMask;
    rt::Tensor mImageMask;
    rt::Tensor mMaskHost; //!< [2, B, 1, 1, maxVl] pinned staging: text then image
    rt::Tensor mState;
    rt::Tensor mStateHost;

    rt::Tensor mTrajectory; //!< current x_t, FLOAT32 [B, H, actionDim]
    rt::Tensor mTrajectoryHost;
    rt::Tensor mVelocity;

    //! [numDenoiseSteps, sites, modulationDim] fp16 and [numDenoiseSteps, innerDim]:
    //! the whole fixed schedule, evaluated by the builder.
    rt::Tensor mModulation;
    rt::Tensor mTimeBias;
};

} // namespace gr00t
} // namespace trt_edgellm
