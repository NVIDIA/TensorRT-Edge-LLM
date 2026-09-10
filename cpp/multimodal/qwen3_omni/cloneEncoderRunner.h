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

#pragma once

#include "common/tensor.h"
#include "common/trtUtils.h"

#include <cuda_fp16.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! \brief Runner for the Qwen3-TTS voice-clone reference encoders (Base checkpoints)
//!
//! Wraps two engines exported from the Base checkpoint:
//!  - speaker_encoder.engine: 24kHz waveform [1, T] (dynamic) -> x-vector [1, talkerHidden].
//!    Mel extraction (STFT 1024/256, slaney mel-128, log) is folded into the graph as
//!    conv-DFT, so the input is raw PCM.
//!  - speech_tokenizer_encoder.engine: 24kHz waveform [1, bucketLen] (static bucket,
//!    zero-padded) -> RVQ codes [bucketFrames, numQuantizers]. The encoder is causal, so
//!    padding does not affect earlier frames; callers take the first
//!    floor(numSamples / downsampleRate) complete frames.
//!
//! Follows the Code2WavRunner pattern (plain TRT engines, no LLM pipeline).
class CloneEncoderRunner
{
public:
    //! \param engineDir Directory containing speaker_encoder.engine and (optionally)
    //!                  speech_tokenizer_encoder.engine. The tokenizer engine may be absent
    //!                  if only x-vector cloning is needed.
    CloneEncoderRunner(std::string const& engineDir, cudaStream_t stream);
    ~CloneEncoderRunner() noexcept = default;

    //! \brief Extract the speaker x-vector from a 24kHz mono waveform.
    //! \param wav24k    Host PCM float32 in [-1, 1] at 24kHz
    //! \param xvecOut   GPU tensor [hiddenDim] FP16, written in place (must be preallocated)
    bool extractSpeakerEmbedding(std::vector<float> const& wav24k, rt::Tensor& xvecOut, cudaStream_t stream);

    //! \brief Encode reference codec codes from a 24kHz mono waveform.
    //! Codes stay on device — consume them via refCodesDevice() (e.g. the codec-embedding
    //! sum kernel reads the engine output in place; no host round-trip).
    //! \param wav24k    Host PCM float32 in [-1, 1] at 24kHz (truncated to the bucket if longer)
    //! \param numFrames Complete frames = floor(min(len, bucket) / downsampleRate)
    bool encodeReferenceCodes(std::vector<float> const& wav24k, int32_t& numFrames, cudaStream_t stream);

    //! Device pointer to the last encodeReferenceCodes output [bucketFrames, numQuantizers] INT64.
    int64_t const* refCodesDevice() const
    {
        return static_cast<int64_t const*>(mCodesBuffer.rawPointer());
    }

    bool hasTokenizerEncoder() const
    {
        return mTokenizerContext != nullptr;
    }
    int64_t speakerEmbeddingDim() const
    {
        return mXvecDim;
    }
    int32_t numQuantizers() const
    {
        return mNumQuantizers;
    }

private:
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;

    std::unique_ptr<nvinfer1::ICudaEngine> mSpeakerEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mSpeakerContext;
    int64_t mXvecDim{0};
    int64_t mSpeakerMaxSamples{0}; //!< Max wav length of the speaker engine's profile

    std::unique_ptr<nvinfer1::ICudaEngine> mTokenizerEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mTokenizerContext;
    rt::Tensor mSharedContextMemory; //!< One device-memory block shared by both contexts (serial use)
    int64_t mBucketSamples{0};       //!< Static wav bucket length of the tokenizer engine
    int64_t mBucketFrames{0};        //!< Codes rows produced for a full bucket
    int32_t mNumQuantizers{0};       //!< Code groups per frame
    int64_t mDownsampleRate{0};      //!< Samples per codec frame (bucketSamples / bucketFrames)

    rt::Tensor mWavBuffer;   //!< GPU wav staging [1, max(bucket, speakerMax)]
    rt::Tensor mXvecFp32;    //!< GPU x-vector output (engine dtype) [1, dim]
    rt::Tensor mCodesBuffer; //!< GPU codes output [bucketFrames, numQuantizers] INT64
};

} // namespace rt
} // namespace trt_edgellm
