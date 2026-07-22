/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/tensor.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace trt_edgellm
{
namespace rt
{

//! GR00T action-context projection engine: lm_hidden_states -> vl_embs.
//!
//! The exported TensorRT engine owns the model-specific stack
//! (eagle_linear -> vlln -> vl_self_attention). The runner only implements the
//! common runtime contract:
//!
//!   lm_hidden_states -> vl_embs
class ActionContextRunner
{
public:
    ActionContextRunner(std::string const& engineDir, cudaStream_t stream);
    ~ActionContextRunner() noexcept = default;

    int64_t getRequiredContextMemorySize() const;
    bool setContextMemory(rt::Tensor& contextMemory);

    int32_t getMaxBatchSize() const noexcept { return mMaxBatchSize; }
    int32_t getMaxSeqLen() const noexcept { return mMaxSeqLen; }
    int32_t getHiddenSize() const noexcept { return mHiddenSize; }
    int32_t getContextHiddenSize() const noexcept { return mContextHiddenSize; }

    std::string const& getInputName() const noexcept { return mInputName; }
    std::string const& getOutputName() const noexcept { return mOutputName; }

    rt::Tensor& getLmHiddenStatesInput() noexcept { return mInputTensor; }
    rt::Tensor const& getLmHiddenStatesInput() const noexcept { return mInputTensor; }
    rt::Tensor& getVlEmbs() noexcept { return mOutputTensor; }
    rt::Tensor const& getVlEmbs() const noexcept { return mOutputTensor; }

    bool reshapeForContext(int32_t batchSize, int32_t seqLen);
    bool copyLmHiddenFrom(rt::Tensor const& lmHidden, cudaStream_t stream, int32_t validSeqLen = -1);
    bool resetExecutionContext(cudaStream_t stream);
    bool infer(cudaStream_t stream) noexcept;

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

    rt::Tensor mInputTensor;
    rt::Tensor mOutputTensor;

    std::string mInputName{"lm_hidden_states"};
    std::string mOutputName{"vl_embs"};
    rt::Coords mInputShape;
    rt::Coords mOutputShape;
    nvinfer1::DataType mInputType{nvinfer1::DataType::kHALF};
    nvinfer1::DataType mOutputType{nvinfer1::DataType::kHALF};

    int32_t mMaxBatchSize{1};
    int32_t mMaxSeqLen{1};
    int32_t mHiddenSize{0};
    int32_t mContextHiddenSize{0};
};

} // namespace rt
} // namespace trt_edgellm
