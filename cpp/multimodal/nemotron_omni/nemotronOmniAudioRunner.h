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

#include "multimodal/common/audioUtils.h"
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

//! \brief Configuration for Nemotron-Omni Parakeet audio encoder
struct NemotronOmniAudioConfig
{
    int32_t melBins{0};             //!< Number of mel-frequency bins
    int32_t audioFeatureDim{0};     //!< Audio feature dimension (LLM hidden size)
    int32_t subsamplingFactor{0};   //!< Parakeet subsampling factor
    int32_t soundContextTokenId{0}; //!< <so_embedding> token ID
};

//! \brief Runner for Nemotron-Omni Parakeet audio encoder.
//!
//! The encoder is exported and built at fixed batch=1: the Conformer's
//! depthwise convolution is a local operator that ignores attention masks,
//! so cross-clip batching with padding silently corrupts short-clip outputs
//! at their right edge. ``preprocess()`` walks every audio clip in the
//! request batch and invokes the encoder once per clip, writing the encoded
//! rows directly into ``mAudioEmbedding`` at sequential offsets. ``infer()``
//! is a no-op.
class NemotronOmniAudioRunner : public MultimodalRunner
{
public:
    //! \brief Constructor for NemotronOmniAudioRunner
    //! \param[in] engineDir Directory containing the audio encoder engine
    //! \param[in] stream CUDA stream for execution
    //! \throws std::runtime_error if engine loading fails or configuration is invalid
    NemotronOmniAudioRunner(std::string const& engineDir, cudaStream_t stream);

    ~NemotronOmniAudioRunner() noexcept = default;

    //! \brief Preprocess multimodal input including audio and text
    //! \param[in] request LLM generation request containing audio and text
    //! \param[in,out] batchedInputIds Batched input token IDs after preprocessing
    //! \param[in] tokenizer Tokenizer for text processing
    //! \param[in,out] mropeCosSinOut MRoPE cos/sin output tensor (unused by this model)
    //! \param[in] stream CUDA stream for execution
    //! \return True if preprocessing succeeded, false otherwise
    bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer, [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut,
        cudaStream_t stream, bool imageOnly = false, bool skipEncoderWork = false) override;

    //! \brief Run inference on the audio encoder
    //! \param[in] stream CUDA stream for execution
    //! \return True if inference succeeded, false otherwise
    bool infer(cudaStream_t stream) override;

    //! \brief Validate and load configuration from JSON file
    //! \param[in] engineDir Path to engine directory
    //! \return True if configuration is valid and loaded successfully, false otherwise
    bool validateAndFillConfig(std::string const& engineDir) override;

    //! \brief Allocate buffers for inference
    //! \param[in] stream CUDA stream for execution
    //! \return True if allocation succeeded, false otherwise
    bool allocateBuffer(cudaStream_t stream) override;

    //! \brief Get audio embeddings from encoder output
    //! \return Reference to audio embedding tensor
    rt::Tensor& getOutputEmbedding() override;

private:
    //! \brief Encode all audio clips in the request batch into mAudioEmbedding.
    //!
    //! Iterates ``request.requests`` (and each request's audioBuffers) in
    //! placeholder order, runs the bs=1 encoder once per clip, and packs the
    //! valid rows of every clip into a contiguous ``[totalRows, hidden]``
    //! buffer. ``audioTokenLengths`` is filled in the same order so
    //! ``textPreprocess`` can replace each ``<so_embedding>`` placeholder
    //! with the right number of audio tokens.
    bool encodeAllClips(
        rt::LLMGenerationRequest const& request, std::vector<int64_t>& audioTokenLengths, cudaStream_t stream);

    //! \brief Encode a single mel-spectrogram clip into mAudioEmbedding at
    //!        the specified row offset. Returns the number of encoded rows.
    bool encodeSingleClip(rt::Tensor const& mel, int64_t destRowOffset, int64_t& encodedRowsOut, cudaStream_t stream);

    //! \brief Resize mAudioEmbedding to ``[rows, hiddenDim]``, reallocating
    //!        if the existing buffer cannot hold that many rows.
    bool resizeEmbeddingForRows(int64_t rows);

    //! \brief Tokenize text and insert audio placeholder tokens
    //! \param[in] request LLM generation request
    //! \param[out] batchInputIds Batched input IDs after tokenization and audio token insertion
    //! \param[in] audioTokenLengths Token lengths for each audio clip
    //! \param[in] tokenizer Tokenizer for text encoding
    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchInputIds,
        std::vector<int64_t> const& audioTokenLengths, tokenizer::Tokenizer const* tokenizer);

    //! \brief Build the online GPU fbank resources (mel filter / window / FFT
    //!        twiddle + scalar params) from the bound CPU MelExtractor's config,
    //!        validated parakeet-spec. Best-effort: returns false (→ CPU fallback)
    //!        when the extractor isn't parakeet-spec, the CuTe DSL GEMM is
    //!        unavailable for this device/build, or a resource upload fails.
    bool initFbankResources(cudaStream_t stream);

    //! \brief Whether the online GPU fbank can run for this clip: resources ready,
    //!        engine mel width agrees, and the sample rate matches what the kernels
    //!        assume. Checked in pass 1 of encodeAllClips before committing a clip
    //!        to the GPU path.
    bool gpuFbankViable(rt::audio::AudioPCM const& pcm) const;

    //! \brief Pass 2 GPU fbank for one clip: upload PCM and run fbankParakeet into a
    //!        fresh ``[1, numFrames, mel_bins]`` FP16 ``melSpec``. Returns false on
    //!        upload / kernel failure so the caller can fall back to the CPU
    //!        extractor (which yields the same frame count, keeping pass-1 sizing valid).
    bool runGpuFbankClip(
        rt::audio::AudioPCM const& pcm, int64_t const numFrames, rt::Tensor& melSpec, cudaStream_t stream);

    //! \brief Per-clip mel plan: built in pass 1 of encodeAllClips (frame count +
    //!        chosen path), consumed in pass 2 by produceClipMel.
    struct ClipPlan
    {
        rt::audio::AudioPCM const* pcm{nullptr}; //!< non-null → run GPU fbank in pass 2
        rt::Tensor hostMel{};                    //!< valid → CPU mel already extracted (upload in pass 2)
        int64_t numFrames{0};                    //!< T (shapes the GPU melSpec view)
    };

    //! \brief Pass 2: produce one clip's ``[1, T, mel_bins]`` FP16 GPU mel from its
    //!        plan — GPU fbank when planned (falling back to a CPU re-extract on
    //!        runtime failure), otherwise upload the pass-1 CPU mel. The fallback
    //!        yields the same floor(N/hop) frame count, keeping pass-1 sizing valid.
    bool produceClipMel(ClipPlan& plan, rt::Tensor& melSpec, cudaStream_t stream);

    NemotronOmniAudioConfig mConfig{}; //!< Nemotron-Omni Parakeet audio configuration
    rt::audio::MelExtractor mFeMel;    //!< FE for PCM→mel; family bound by validateAndFillConfig
    rt::Tensor mInputFeatures{};       //!< [1, paddedSeqLen, mel_bins] encoder input (rebound per clip)
    rt::Tensor mAudioEmbedding{};      //!< [totalEncodedRows, hidden_dim] flat output for all clips in batch
    int64_t mMaxSeqLen{0};             //!< Max raw mel-spectrogram time steps from engine profile

    // Online GPU fbank state (persistent across clips).
    // When mFbankReady is false the runner uses the CPU MelExtractor (mFeMel).
    bool mFbankReady{false}; //!< true once initFbankResources succeeds
    rt::audioUtils::FbankResourcesParakeet
        mFbankResourcesParakeet{}; //!< weights / tables / scalars + pre-allocated workspace
    rt::Tensor mPcmF32Device{};    //!< [maxPcmSamples] Float — mono FP32 PCM staging, reshaped to [N] per clip
    rt::Tensor mMelSpecDevice{}; //!< [1, maxFrames, nMel] Half — fbank output backing store, viewed at [1, T_out, nMel]
};

} // namespace rt
} // namespace trt_edgellm
