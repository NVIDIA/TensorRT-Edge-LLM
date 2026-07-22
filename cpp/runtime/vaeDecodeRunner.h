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

//! Wan VAE decode engine: latents [B,C,T',H',W'] -> pixels [B,3,T,H,W].
class VaeDecodeRunner
{
public:
    //! \p engineDir is the ``visual_decode/`` component directory (contains config.json + engine).
    explicit VaeDecodeRunner(std::string const& engineDir, cudaStream_t stream);

    ~VaeDecodeRunner() noexcept = default;

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

    rt::Coords const& getLatentShape() const noexcept
    {
        return mInputShape;
    }

    rt::Coords const& getPixelShape() const noexcept
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

    rt::Tensor& getPixels() noexcept
    {
        return mPixelsTensor;
    }

    rt::Tensor const& getPixels() const noexcept
    {
        return mPixelsTensor;
    }

    bool copyLatentsFrom(rt::Tensor const& src, cudaStream_t stream);
    bool copyPixelsTo(rt::Tensor& dst, cudaStream_t stream) const;
    bool copyPixelsTo(VideoBuffer& video, cudaStream_t stream) const;
    bool decode(cudaStream_t stream) noexcept;

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

    std::string mInputName{"latents"};
    std::string mOutputName{"pixels"};
    rt::Coords mInputShape;
    rt::Coords mOutputShape;
    nvinfer1::DataType mInputType{nvinfer1::DataType::kHALF};
    nvinfer1::DataType mOutputType{nvinfer1::DataType::kHALF};

    rt::Tensor mLatentsTensor;
    rt::Tensor mPixelsTensor;
};

} // namespace rt
} // namespace trt_edgellm
