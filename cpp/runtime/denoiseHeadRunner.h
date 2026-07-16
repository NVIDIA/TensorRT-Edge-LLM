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

//! Cosmos vision denoise head: last_hidden_state -> pred vision latents.
class DenoiseHeadRunner
{
public:
    //! \p engineDir is the ``denoise_head/`` component directory (contains config.json + engine).
    explicit DenoiseHeadRunner(std::string const& engineDir, cudaStream_t stream);

    ~DenoiseHeadRunner() noexcept = default;

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

    rt::Coords const& getLastHiddenShape() const noexcept
    {
        return mInputShape;
    }

    rt::Coords const& getPredLatentShape() const noexcept
    {
        return mOutputShape;
    }

    rt::Tensor& getLastHiddenState() noexcept
    {
        return mLastHiddenTensor;
    }

    rt::Tensor const& getLastHiddenState() const noexcept
    {
        return mLastHiddenTensor;
    }

    rt::Tensor& getPredLatents() noexcept
    {
        return mPredLatentsTensor;
    }

    rt::Tensor const& getPredLatents() const noexcept
    {
        return mPredLatentsTensor;
    }

    bool copyLastHiddenFrom(rt::Tensor const& src, cudaStream_t stream);
    bool runHead(cudaStream_t stream) noexcept;

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

    std::string mInputName{"last_hidden_state"};
    std::string mOutputName{"pred_vision_latents"};
    rt::Coords mInputShape;
    rt::Coords mOutputShape;
    nvinfer1::DataType mInputType{nvinfer1::DataType::kHALF};
    nvinfer1::DataType mOutputType{nvinfer1::DataType::kHALF};

    rt::Tensor mLastHiddenTensor;
    rt::Tensor mPredLatentsTensor;
};

} // namespace rt
} // namespace trt_edgellm
