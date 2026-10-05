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

#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "common/trtUtils.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace trt_edgellm
{
namespace pi05
{

//! \brief Configuration parsed from the pi0.5 component config.json files.
//! A missing key fails engine load rather than running on a guess, except for
//! the hoist flag and the cond fields it gates, which an unhoisted export omits.
struct Pi05PolicyConfig
{
    //! The export run every component config.json is stamped with.
    std::string exportId;

    // prefix component
    int32_t numHiddenLayers{0};
    int32_t numKVHeads{0};
    int32_t headDim{0};
    int32_t hiddenSize{0};
    float ropeTheta{0.0F};
    int32_t maxPrefixLen{0};

    // action component
    int32_t actionDim{0};
    int32_t actionHorizon{0};
    int32_t numDenoiseSteps{0};
    //! Modulation hoisted into the cond component instead of recomputed per step.
    bool hoistedAdarmsCond{false};
    int32_t numAdarmsSites{0};
    //! Slots per K/V cache half; the prefix and action tokens share them.
    int32_t kvCacheCapacity{0};

    // cond component (hoisted-modulation exports only)
    int32_t modulationDim{0};
    int32_t maxDenoiseSteps{0};

    // visual component
    int32_t numImageTokens{0};
    int32_t imageSize{0};
};

//! \brief Pages one request's K/V occupies in the paged pool. The capacity is a whole
//! number of pages, so an identity page table reproduces the contiguous layout.
inline int32_t pagesPerSeq(Pi05PolicyConfig const& config) noexcept
{
    return config.kvCacheCapacity / rt::kTOKENS_PER_PAGE;
}

//! \brief One pi0.5 engine and the user-managed context that runs it.
struct Pi05Component
{
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    //! TensorRT's default aux streams are blocking, which is an illegal
    //! synchronization during CUDA-graph capture; non-blocking ones are not.
    AuxStreamSet auxStreams;

    //! Load ``<engineDir>/<component>/<component>.engine``.
    void load(std::string const& engineDir, std::string const& component);

    int64_t getRequiredContextMemorySize() const;

    //! Bind the caller's scratch pool. Must run before the first enqueue and before
    //! any graph capture: the capture bakes the pool address in.
    bool setContextMemory(rt::Tensor& pool);
};

//! \brief Stage the bound RoPE inputs for \p len positions starting at \p posOffset of
//! \p ropeCache. Positions do not vary with the request, so every batch entry gets the
//! same rows, packed at the stride the engine input is bound with.
void stageRopeInputs(rt::Tensor const& ropeCache, int32_t headDim, int32_t batch, int32_t len, int32_t posOffset,
    rt::Tensor& cosSin, rt::Tensor& posIds, rt::Tensor& posIdsHost, cudaStream_t stream);

} // namespace pi05
} // namespace trt_edgellm
