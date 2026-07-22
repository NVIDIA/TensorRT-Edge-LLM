/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/audioDecodeRunner.h"
#include "runtime/audioEncodeRunner.h"
#include "runtime/cosmosDenoiseRunner.h"
#include "runtime/vaeDecodeRunner.h"
#include "runtime/vaeEncodeRunner.h"
#include "runtime/wfmRuntimeUtils.h"

#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace trt_edgellm
{
namespace tokenizer
{
class Tokenizer;
} // namespace tokenizer

namespace rt
{

//! Orchestrates Cosmos TRT engines: encode → denoise → decode (vision + optional sound for Omni).
class WFMInferenceRuntime
{
public:
    //! \p engineDir must contain config.json and packing_static.json.
    explicit WFMInferenceRuntime(std::string const& engineDir, cudaStream_t stream);

    ~WFMInferenceRuntime() noexcept = default;

    bool handleRequest(WFMGenerationRequest const& request, WFMGenerationResponse& response, cudaStream_t stream);

    CosmosEngineConfig const& getEngineConfig() const noexcept
    {
        return mConfig;
    }

    CosmosPackedStatic const& getPackedStatic() const noexcept
    {
        return mPackedStatic;
    }

private:
    bool examineRequest(WFMGenerationRequest const& request) const noexcept;
    bool prepareTextPhase0(std::string const& prompt, cudaStream_t stream);
    bool allocateZeroSoundLatents(rt::Coords const& shape, nvinfer1::DataType dtype, cudaStream_t stream);

    std::string mEngineDir;
    CosmosEngineConfig mConfig{};
    CosmosPackedStatic mPackedStatic{};
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;
    EmbeddingData mEmbedding{};
    CosmosTextPhase0 mTextPhase0{};
    std::unique_ptr<VaeEncodeRunner> mVaeEncodeRunner;
    std::unique_ptr<VaeDecodeRunner> mVaeDecodeRunner;
    std::unique_ptr<AudioEncodeRunner> mAudioEncodeRunner;
    std::unique_ptr<AudioDecodeRunner> mAudioDecodeRunner;
    std::unique_ptr<CosmosDenoiseRunner> mDenoiseRunner;
    std::shared_ptr<rt::Tensor> mOutputVideo;
    std::shared_ptr<rt::Tensor> mOutputWaveform;
    rt::Tensor mNoisyLatents{};
    rt::Tensor mNoisySoundLatents{};
    rt::Tensor mZeroSoundLatents{};
    rt::Tensor mSharedExecContextMemory{};
};

} // namespace rt
} // namespace trt_edgellm
