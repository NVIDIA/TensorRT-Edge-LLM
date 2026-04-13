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

/*
 * AlpamayoExpertRunner: runs TRT Diffusion Expert inline after VLM decode.
 * KV cache stays on GPU — no disk I/O, no Python.
 */

#pragma once

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/tensor.h"

#include <NvInfer.h>
#include <curand.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

struct AlpamayoExpertConfig
{
    int32_t numCandidates{6};
    int32_t numDiffusionSteps{10};
    int32_t numDiffusionTokens{64};
    int32_t actionDim{2};
    int32_t numLayers{36};
    int32_t numKVHeads{8};
    int32_t headDim{128};
    int32_t trajFutureStartTokenId{155681};
    int32_t ropeDelta{-2592};
    int64_t seed{42};
    bool multiSeq{false};
};

class AlpamayoExpertRunner
{
public:
    AlpamayoExpertRunner(std::string const& enginePath, AlpamayoExpertConfig const& config, cudaStream_t stream);
    ~AlpamayoExpertRunner() noexcept;

    /*! Run 10-step flow matching diffusion using VLM's KV cache.
     *
     * @param vlmKVCache Contiguous GPU tensor: (numLayers, 2, numKVHeads, seqLen, headDim) fp16
     *                   from kernel::saveKVCacheIntoTensor()
     * @param kvSeqLen   Sequence length of the truncated KV cache (after traj_future_start)
     * @param ropeDelta  RoPE delta (typically -2592 for 4-camera 4-frame setup)
     * @param outputActions Output GPU tensor: (numCandidates, numDiffusionTokens, actionDim) fp32
     * @param stream     CUDA stream
     */
    void runDiffusion(rt::Tensor const& vlmKVCache, int32_t kvSeqLen, int32_t ropeDelta, rt::Tensor& outputActions,
        cudaStream_t stream);

    /*! Find <traj_future_start> in decoded output IDs and return the truncation point.
     *  Returns prefillLen + idx + 1 where idx is the index in outputIds.
     *  Returns -1 if not found.
     */
    int32_t findTrajFutureStart(std::vector<int32_t> const& outputIds, int32_t prefillSeqLen) const noexcept;

    AlpamayoExpertConfig const& getConfig() const noexcept
    {
        return mConfig;
    }
    nvinfer1::IExecutionContext* getContext() noexcept
    {
        return mContext.get();
    }

private:
    void loadEngine(std::string const& path, cudaStream_t stream);
    void allocateBuffers(int32_t maxKVSeqLen, cudaStream_t stream);

    AlpamayoExpertConfig mConfig;

    // TRT engine
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    rt::Tensor mExecContextMemory;

    // Pre-allocated GPU buffers
    rt::Tensor mKVCacheReshaped; // (72, B, 8, maxS, 128) float32
    rt::Tensor mNoisyAction;     // (B, 64, 2) float32
    rt::Tensor mTimestep;        // (B, 1, 1) float32
    rt::Tensor mPositionIds;     // (3, B, 64) int64
    rt::Tensor mAttentionMask;   // (B, 1, 64, maxS+64) float32
    rt::Tensor mPredVelocity;    // (B, 64, 2) float32

    // cuRAND
    curandGenerator_t mRNG{nullptr};
};

} // namespace rt
} // namespace trt_edgellm
