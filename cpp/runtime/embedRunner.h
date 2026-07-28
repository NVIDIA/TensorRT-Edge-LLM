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
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/tensor.h"
#include "runtime/wfmRuntimeUtils.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace trt_edgellm
{
namespace rt
{

//! Cosmos vision embed engine: latents + timestep -> gen_seq vision tokens.
class EmbedRunner
{
public:
    //! \p engineDir is the ``embed/`` component directory (contains config.json + engine).
    explicit EmbedRunner(std::string const& engineDir, cudaStream_t stream);

    ~EmbedRunner() noexcept = default;

    int64_t getRequiredContextMemorySize() const;
    bool setContextMemory(rt::Tensor& sharedContextMemory);

    std::string const& getLatentsInputName() const noexcept
    {
        return mLatentsInputName;
    }

    std::string const& getTimestepInputName() const noexcept
    {
        return mTimestepInputName;
    }

    std::string const& getOutputName() const noexcept
    {
        return mOutputName;
    }

    rt::Coords const& getLatentShape() const noexcept
    {
        return mLatentsShape;
    }

    rt::Coords const& getGenSeqShape() const noexcept
    {
        return mOutputShape;
    }

    rt::Tensor& getLatents() noexcept
    {
        return mLatentsTensor;
    }

    rt::Tensor const& getLatents() const noexcept
    {
        return mLatentsTensor;
    }

    rt::Tensor& getTimestep() noexcept
    {
        return mTimestepTensor;
    }

    rt::Tensor const& getTimestep() const noexcept
    {
        return mTimestepTensor;
    }

    rt::Tensor& getGenSeq() noexcept
    {
        return mGenSeqTensor;
    }

    rt::Tensor const& getGenSeq() const noexcept
    {
        return mGenSeqTensor;
    }

    bool copyLatentsFrom(rt::Tensor const& src, cudaStream_t stream);
    bool setTimestep(float timestep, cudaStream_t stream);
    bool embed(cudaStream_t stream) noexcept;

private:
    bool loadConfig();
    bool loadEngine(cudaStream_t stream);
    bool validateAndFillConfig();
    bool allocateBuffers();
    bool bindTensors() noexcept;

    std::string mEngineDir;
    nlohmann::json mConfigJson;

    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;

    std::string mLatentsInputName{"vision_latents"};
    std::string mTimestepInputName{"timestep"};
    std::string mOutputName{"gen_seq"};
    rt::Coords mLatentsShape;
    rt::Coords mTimestepShape;
    rt::Coords mOutputShape;
    bool mTimestepIsScalar{false};
    nvinfer1::DataType mLatentsType{nvinfer1::DataType::kHALF};
    nvinfer1::DataType mTimestepType{nvinfer1::DataType::kFLOAT};
    nvinfer1::DataType mOutputType{nvinfer1::DataType::kHALF};

    rt::Tensor mLatentsTensor;
    rt::Tensor mTimestepTensor;
    rt::Tensor mGenSeqTensor;
};

} // namespace rt
} // namespace trt_edgellm
