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

//! Wan VAE encode engine: pixels [B,3,T,H,W] -> latents [B,C,T',H',W'].
class VaeEncodeRunner
{
public:
    //! \p engineDir is the ``visual_encode/`` component directory (contains config.json + engine).
    explicit VaeEncodeRunner(std::string const& engineDir, cudaStream_t stream);

    ~VaeEncodeRunner() noexcept = default;

    int64_t getRequiredContextMemorySize() const;
    bool setContextMemory(rt::Tensor& sharedContextMemory);

    std::string const& getInputName() const noexcept
    {
        return mInputName;
    }

    std::string const& getOutputName() const noexcept
    {
        return mOutputName;
    }

    rt::Coords const& getPixelShape() const noexcept
    {
        return mInputShape;
    }

    rt::Coords const& getLatentShape() const noexcept
    {
        return mOutputShape;
    }

    rt::Tensor& getPixels() noexcept
    {
        return mPixelsTensor;
    }

    rt::Tensor const& getPixels() const noexcept
    {
        return mPixelsTensor;
    }

    rt::Tensor& getLatents() noexcept
    {
        return mLatentsTensor;
    }

    rt::Tensor const& getLatents() const noexcept
    {
        return mLatentsTensor;
    }

    bool copyPixelsFrom(rt::Tensor const& src, cudaStream_t stream);
    bool copyPixelsFrom(VideoBuffer const& video, cudaStream_t stream);
    bool encode(cudaStream_t stream) noexcept;

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

    std::string mInputName{"pixels"};
    std::string mOutputName{"latents"};
    rt::Coords mInputShape;
    rt::Coords mOutputShape;
    nvinfer1::DataType mInputType{nvinfer1::DataType::kHALF};
    nvinfer1::DataType mOutputType{nvinfer1::DataType::kHALF};

    rt::Tensor mPixelsTensor;
    rt::Tensor mLatentsTensor;
};

} // namespace rt
} // namespace trt_edgellm
