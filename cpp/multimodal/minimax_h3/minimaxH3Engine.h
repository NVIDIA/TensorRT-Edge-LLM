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
#include "runtime/state/externalWeightManager.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace minimax_h3
{

//! \brief One MiniMax-H3 engine, its runtime config, and the context that runs it.
//!
//! The context memory and the weights the engine binds from the checkpoint are claimed here, once.
class MiniMaxH3Engine
{
public:
    //! Load ``<artifactDir>/<component>.engine``, its ``config.json``, and its checkpoint weights.
    MiniMaxH3Engine(std::string const& artifactDir, std::string const& component, cudaStream_t stream);

    nlohmann::json const& config() const noexcept
    {
        return mConfig;
    }

    //! Bind a tensor address, and its shape when it is an input.
    void bind(char const* name, rt::Tensor& tensor);
    void bind(char const* name, void* data, rt::Coords const& inputShape);

    void enqueue(cudaStream_t stream);

private:
    std::string mComponent;
    nlohmann::json mConfig;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    rt::ExternalWeightManager mWeights; //!< Outlives mContext, which reads the weights.
    AuxStreamSet mAuxStreams;           //!< Outlives mContext, which uses the streams.
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
};

} // namespace minimax_h3
} // namespace rt
} // namespace trt_edgellm
