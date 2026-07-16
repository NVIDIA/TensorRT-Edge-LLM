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

//! AVAE encode engine: waveform [B,1,T] or [B,T] -> sound_latents [B,C,T'].
class AudioEncodeRunner
{
public:
    //! \p engineDir is the ``audio_encode/`` component directory (contains config.json + engine).
    explicit AudioEncodeRunner(std::string const& engineDir, cudaStream_t stream);

    ~AudioEncodeRunner() noexcept = default;

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

    rt::Coords const& getWaveformShape() const noexcept
    {
        return mInputShape;
    }

    rt::Coords const& getSoundLatentShape() const noexcept
    {
        return mOutputShape;
    }

    rt::Tensor& getWaveform() noexcept
    {
        return mWaveformTensor;
    }

    rt::Tensor const& getWaveform() const noexcept
    {
        return mWaveformTensor;
    }

    rt::Tensor& getSoundLatents() noexcept
    {
        return mSoundLatentsTensor;
    }

    rt::Tensor const& getSoundLatents() const noexcept
    {
        return mSoundLatentsTensor;
    }

    bool copyWaveformFrom(rt::Tensor const& src, cudaStream_t stream);
    bool copyWaveformFrom(AudioBuffer const& audio, cudaStream_t stream);
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

    std::string mInputName{"waveform"};
    std::string mOutputName{"sound_latents"};
    rt::Coords mInputShape;
    rt::Coords mOutputShape;
    nvinfer1::DataType mInputType{nvinfer1::DataType::kHALF};
    nvinfer1::DataType mOutputType{nvinfer1::DataType::kHALF};

    rt::Tensor mWaveformTensor;
    rt::Tensor mSoundLatentsTensor;
};

} // namespace rt
} // namespace trt_edgellm
