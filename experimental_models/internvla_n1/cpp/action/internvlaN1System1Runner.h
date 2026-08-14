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

#include "action/internvlaN1Scheduler.h"
#include "common/tensor.h"

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace trt_edgellm
{
namespace internvla_n1
{

//! \brief InternVLA-N1 System 1: the memory block and the flow-matching trajectory loop.
//!
//! System 2 (a stock Qwen2.5-VL) is served by the core runtime and emits `z_latents` directly,
//! so this runner needs nothing from it but that tensor.
//!
//! Two engines rather than one, deliberately: the memory block runs once per observation window
//! while the trajectory expert runs once per denoising step, so a fused graph would re-encode
//! the frames on every step.
//!
//! Everything the loop does outside the engines is a scheduler lookup and one fused kernel --
//! the projections that used to surround the expert are inside its graph.
//!
//! \note Owns its own execution contexts. It does **not** join the core runtime's shared
//! context-memory pool, which is sized as a max() over components and therefore assumes its
//! members run serialized. Keeping this runner independent is what lets System 1 and System 2
//! overlap.
class InternVLAN1System1Runner
{
public:
    struct Config
    {
        int32_t numSampleTrajs{32};    //!< Trajectories sampled per plan.
        int32_t predictStepNums{32};   //!< Waypoints per trajectory.
        int32_t actionDim{3};          //!< Waypoint width.
        int32_t numInferenceSteps{10}; //!< Denoising steps.
        float guidanceScale{1.0F};     //!< Classifier-free guidance weight.
    };

    //! \brief Create a stream at the greatest priority the device offers.
    //!
    //! System 1 produces the control output. If its rate collapses under a competing load the
    //! trajectories arrive too late to steer with, so it is the workload that must not be
    //! starved -- unlike System 2, which is allowed to take longer.
    //!
    //! **Only effective when System 2 shares this process.** Stream priorities order streams
    //! within one CUDA context; across processes the GPU time-slices between contexts and the
    //! priority is invisible. Measured on Thor both ways, against a competing load:
    //!
    //!   same process    94.0 ms per trajectory at equal priority, 73.3 ms with this stream
    //!   separate process 121 ms either way -- no effect at all
    //!
    //! Alone it is 48.2 ms. So in-process priority recovers roughly half of what contention
    //! costs, and no arrangement recovers all of it: CUDA preempts between kernels, not inside
    //! one. If System 2 runs as its own process, this stream buys nothing and the integration
    //! is what needs fixing, not the priority.
    static cudaStream_t makeControlStream();

    //! \param engineDir Directory holding memory_bf16.engine and traj_dit_bf16.engine.
    InternVLAN1System1Runner(std::string const& engineDir, Config const& config, cudaStream_t stream);
    ~InternVLAN1System1Runner() noexcept = default;

    //! \brief Device memory the two contexts need. Allocated per-runner, not shared with the LLM.
    int64_t getRequiredContextMemorySize() const;

    //! \brief Encode a window of ResNet-normalized frames into memory tokens.
    //! \param images Device tensor `[frames, 3, 224, 224]`, FLOAT32.
    //! \return Device tensor `[1, numQuery, 768]`, owned by this runner.
    rt::Tensor& encodeMemory(rt::Tensor const& images, cudaStream_t stream);

    //! \brief Run the denoising loop and return sampled trajectories.
    //!
    //! \param conditioning Device tensor `[2, condLen, 768]` -- the null row first, then
    //!        memory tokens concatenated with the System-2 `z_latents`. Supplying both rows
    //!        explicitly keeps the guidance convention in the caller, where it is visible.
    //! \param noise Device tensor `[numSampleTrajs, predictStepNums, actionDim]`, the initial
    //!        sample. Passed in rather than drawn here so a run can be reproduced exactly.
    //! \return Device tensor with the same shape as \p noise, owned by this runner.
    rt::Tensor& sampleTrajectory(rt::Tensor const& conditioning, rt::Tensor const& noise, cudaStream_t stream);

    InternVLAN1Scheduler const& scheduler() const noexcept
    {
        return mScheduler;
    }

private:
    void loadEngine(std::string const& path, std::unique_ptr<nvinfer1::ICudaEngine>& engine,
        std::unique_ptr<nvinfer1::IExecutionContext>& context, cudaStream_t stream);

    Config mConfig;
    InternVLAN1Scheduler mScheduler;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> mMemoryEngine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mMemoryContext{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> mDitEngine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mDitContext{nullptr};

    //! Scratch for both contexts. Owned here rather than drawn from the core runtime's shared
    //! pool: that pool is sized as a max() over components and so assumes they never overlap,
    //! which is precisely what System 1 has to do.
    rt::Tensor mContextMemory;
    rt::Tensor mMemoryTokens;   //!< [1, numQuery, 768]
    rt::Tensor mLatents;        //!< [numSampleTrajs, predictStepNums, actionDim]
    rt::Tensor mDoubledLatents; //!< [2 * numSampleTrajs, ...] -- the engine's batch
    rt::Tensor mDoubledCond;    //!< [2 * numSampleTrajs, condLen, 768]
    rt::Tensor mTimesteps;      //!< [2 * numSampleTrajs], one value broadcast per step
    rt::Tensor mModelOutput;    //!< [2 * numSampleTrajs, predictStepNums, actionDim]
};

} // namespace internvla_n1
} // namespace trt_edgellm
