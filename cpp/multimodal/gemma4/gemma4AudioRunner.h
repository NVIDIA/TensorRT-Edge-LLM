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

#include "multimodal/common/multimodalRunner.h"
#include "runtime/audioUtils.h"
#include "runtime/melSpectrogram.h"
#include <cuda_fp16.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! \brief Configuration for Gemma4 audio encoder
struct Gemma4AudioConfig
{
    int32_t melBins{128};          //!< Number of mel-frequency bins
    int32_t audioFeatureDim{1536}; //!< Audio output dimension (text_hidden_size after embedder projection)
    int32_t subsamplingFactor{4};  //!< Temporal downsampling factor (2x stride-2 convs)

    int32_t audioTokenId{0};            //!< Audio placeholder token ID
    int32_t beginAudioTokenId{-1};      //!< boa token ID wrapping each audio span (-1 when absent)
    int32_t endAudioTokenId{-1};        //!< eoa token ID wrapping each audio span (-1 when absent)
    int32_t maxAudioClipsPerRequest{8}; //!< Max audio clips in a single request (for buffer pre-allocation)
};

//! \brief Runner for Gemma4 audio encoder.
//!
//! The encoder is exported at fixed batch=1: the chunked local attention
//! and depthwise convolution are local operators, so cross-clip batching with
//! padding may corrupt outputs. ``preprocess()`` walks every audio clip in the
//! request batch and invokes the encoder once per clip, writing the encoded
//! rows directly into ``mAudioEmbedding`` at sequential offsets.
class Gemma4AudioRunner : public MultimodalRunner
{
public:
    //! \brief Constructor
    //! \param[in] engineDir Directory containing the audio encoder engine
    //! \param[in] stream CUDA stream for execution
    Gemma4AudioRunner(std::string const& engineDir, cudaStream_t stream);

    ~Gemma4AudioRunner() noexcept = default;

    //! \brief Preprocess multimodal input including audio and text
    bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer, [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut,
        cudaStream_t stream, bool imageOnly = false, bool skipEncoderWork = false) override;

    //! \brief Run inference (no-op, encoding done per-clip in preprocess)
    bool infer(cudaStream_t stream) override;

    //! \brief Validate and load configuration from JSON file
    bool validateAndFillConfig(std::string const& engineDir) override;

    //! \brief Allocate buffers for inference
    bool allocateBuffer(cudaStream_t stream) override;

    //! \brief Get audio embeddings from encoder output
    rt::Tensor& getOutputEmbedding() override;

private:
    //! \brief Load pre-computed mel-spectrogram from file
    bool loadMelSpectrogramFromFile(
        std::string const& filePath, std::string const& format, rt::Tensor& melSpectrogram, cudaStream_t stream);

    //! \brief Extract mel-spectrogram from raw PCM and upload to GPU as FP16 [1, T, mel_bins].
    bool extractMelFromPcm(rt::audio::AudioPCM const& pcm, rt::Tensor& melGpu, cudaStream_t stream);

    //! \brief Encode all audio clips in the request batch into mAudioEmbedding.
    bool encodeAllClips(
        rt::LLMGenerationRequest const& request, std::vector<int64_t>& audioTokenLengths, cudaStream_t stream);

    //! \brief Encode a single mel-spectrogram clip. Returns encoded rows.
    bool encodeSingleClip(rt::Tensor const& mel, int64_t destRowOffset, int64_t& encodedRowsOut, cudaStream_t stream);

    //! \brief Resize mAudioEmbedding to hold the given number of rows.
    bool resizeEmbeddingForRows(int64_t rows);

    //! \brief Tokenize text and expand audio placeholder tokens
    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchInputIds,
        std::vector<int64_t> const& audioTokenLengths, tokenizer::Tokenizer const* tokenizer);

    Gemma4AudioConfig mConfig{};  //!< Audio encoder configuration
    rt::Tensor mInputFeatures{};  //!< [1, paddedSeqLen, mel_bins] encoder input
    rt::Tensor mValidMask{};      //!< [1, paddedSeqLen/4] valid mask after downsample
    rt::Tensor mAudioEmbedding{}; //!< [totalEncodedRows, hidden_dim] flat output
    int64_t mMaxSeqLen{0};        //!< Max mel-spectrogram time steps from engine profile

    //! Mel extractor for PCM→mel fallback (Whisper-style, 16kHz, 128 bins).
    rt::audio::MelExtractor mMelExtractor;
};

} // namespace rt
} // namespace trt_edgellm
