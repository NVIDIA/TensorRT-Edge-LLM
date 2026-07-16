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

//! AVAE decode engine: sound_latents [B,C,T'] -> waveform [B,1,T] or [B,T].
class AudioDecodeRunner
{
public:
    //! \p engineDir is the ``audio_decode/`` component directory (contains config.json + engine).
    explicit AudioDecodeRunner(std::string const& engineDir, cudaStream_t stream);

    ~AudioDecodeRunner() noexcept = default;

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

    rt::Coords const& getSoundLatentShape() const noexcept
    {
        return mInputShape;
    }

    rt::Coords const& getWaveformShape() const noexcept
    {
        return mOutputShape;
    }

    rt::Tensor& getSoundLatents() noexcept
    {
        return mSoundLatentsTensor;
    }

    rt::Tensor const& getSoundLatents() const noexcept
    {
        return mSoundLatentsTensor;
    }

    rt::Tensor& getWaveform() noexcept
    {
        return mWaveformTensor;
    }

    rt::Tensor const& getWaveform() const noexcept
    {
        return mWaveformTensor;
    }

    bool copySoundLatentsFrom(rt::Tensor const& src, cudaStream_t stream);
    bool copyWaveformTo(rt::Tensor& dst, cudaStream_t stream) const;
    bool copyWaveformTo(AudioBuffer& audio, cudaStream_t stream) const;
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

    std::string mInputName{"sound_latents"};
    std::string mOutputName{"waveform"};
    rt::Coords mInputShape;
    rt::Coords mOutputShape;
    nvinfer1::DataType mInputType{nvinfer1::DataType::kHALF};
    nvinfer1::DataType mOutputType{nvinfer1::DataType::kHALF};

    rt::Tensor mSoundLatentsTensor;
    rt::Tensor mWaveformTensor;
};

} // namespace rt
} // namespace trt_edgellm
