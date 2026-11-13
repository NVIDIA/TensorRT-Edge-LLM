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

#include "common/checkMacros.h"
#include "common/tensor.h"

#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

namespace detail
{

//! Drain even a rejected asynchronous refit before its caller releases refitter or source storage.
template <typename Enqueue, typename Synchronize>
void completeRefit(Enqueue&& enqueue, Synchronize&& synchronize)
{
    bool const refitted = enqueue();
    CUDA_CHECK(synchronize());
    ELLM_CHECK(refitted, "TensorRT weightless engine refit failed");
}

} // namespace detail

//! Temporary device storage and named views consumed together during engine initialization.
//! Views are destroyed before their backing storage. Copying is prohibited by Tensor ownership.
struct RefitWeightSet
{
    Tensor storage;              //!< Owns the temporary refit arena.
    std::vector<Tensor> weights; //!< Non-owning named views into storage.
};

//! Optional pre-context refit; stream must be explicit when weights is non-null.
struct EngineRefitOptions
{
    std::unique_ptr<RefitWeightSet> weights; //!< Ownership transferred from the checkpoint loader.
    cudaStream_t stream{nullptr};            //!< Initialization stream, not the default stream.
};

//! Validate mode metadata and binding destinations before checkpoint allocation.
//! Returns whether refit is required; throws for unsupported builds or malformed contracts.
bool validateWeightlessConfig(nlohmann::json const& config);

//! Apply all supplied weights and wait for completion before returning.
//! The caller must keep weights alive until this returns, including on error.
//! Throws on missing names, incompatible prototypes, or failed refit; no context may exist yet.
void refitEngine(nvinfer1::ICudaEngine& engine, RefitWeightSet const& weights, cudaStream_t stream);

//! Consume optional refit storage before context creation; reject stripped engines without it.
void refitEngineIfNeeded(nvinfer1::ICudaEngine& engine, EngineRefitOptions refit = {});

} // namespace rt
} // namespace trt_edgellm
