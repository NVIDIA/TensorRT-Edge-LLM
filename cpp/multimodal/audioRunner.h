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

#include "multimodal/audioUtils.h"
#include "multimodalRunner.h"
#include "runtime/audioUtils.h"
#include "runtime/melSpectrogram.h"
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! \brief Configuration for Qwen3-Omni audio encoder
struct AudioConfig
{
    int32_t melBins{128};          //!< Number of mel-frequency bins
    int32_t audioFeatureDim{2560}; //!< Audio feature dimension (output embedding size)
    int32_t nWindow{100};          //!< Window size for audio chunking
    int32_t nWindowInfer{100};     //!< Inference window size

    // Audio special tokens (from tokenizer_config.json)
    int32_t audioTokenId{151675}; //!< <|audio_pad|> token ID
    float mropeTheta{0.0F};       //!< Multi-dimensional RoPE theta (0 = no MRope)
    int32_t mropeSectionH{20};    //!< MRoPE section: frequency pairs for height (default Qwen3-Omni)
    int32_t mropeSectionW{20};    //!< MRoPE section: frequency pairs for width (default Qwen3-Omni)
};

//! \brief Runner for Qwen3-Omni audio encoder
//!
//! This class handles audio preprocessing and encoder inference for Qwen3-Omni model.
class Qwen3OmniAudioRunner : public MultimodalRunner
{
public:
    //! \brief Constructor for Qwen3OmniAudioRunner
    //! \param[in] engineDir Directory containing the audio encoder engine
    //! \param[in] stream CUDA stream for execution
    //! \throws std::runtime_error if engine loading fails or configuration is invalid
    Qwen3OmniAudioRunner(std::string const& engineDir, cudaStream_t stream);

    ~Qwen3OmniAudioRunner() noexcept = default;

    //! \brief Preprocess multimodal input including audio and text
    //! \param[in] request LLM generation request containing audio and text
    //! \param[in,out] batchedInputIds Batched input token IDs after preprocessing
    //! \param[in] tokenizer Tokenizer for text processing
    //! \param[out] mropeCosSinOut MRope cos/sin output cache (required: this runner is MRope-only)
    //! \param[in] stream CUDA stream for execution
    //! \return True if preprocessing succeeded, false otherwise
    bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer, rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream,
        bool imageOnly = false) override;

    //! \brief Run inference on the audio encoder
    //! \param[in] stream CUDA stream for execution
    //! \return True if inference succeeded, false otherwise
    bool infer(cudaStream_t stream) override;
    bool prepareArtifactSubset(rt::LLMGenerationRequest const& request, std::vector<size_t> const& originalItemIndices,
        cudaStream_t stream) override;
    std::vector<int64_t> preparedArtifactRowCounts() const override;
    void discardPreparedInput() noexcept override;

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

    //! \brief Initialize sequential MRope cache for system prompt KVCache saving
    //! \details For audio-only MRope models (e.g. Qwen3-ASR), initialize sequential MRope cache
    //!          since no vision runner will fill it. When a vision runner is present, this is a no-op
    //!          because QwenViTRunner::preprocessSystemPrompt handles MRope initialization.
    //! \param[in] systemPrompt System prompt text
    //! \param[in] tokenizer Tokenizer instance
    //! \param[out] mropeCosSinOut MRope cos/sin output cache (required for non-empty system prompts)
    //! \param[in] stream CUDA stream
    //! \return True on success, false on failure
    bool preprocessSystemPrompt(std::string const& systemPrompt, tokenizer::Tokenizer const* tokenizer,
        rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream) override;

private:
    struct ClipPlan
    {
        std::shared_ptr<rt::audio::AudioPCM> pcm;
        rt::Tensor hostMel;
        int64_t numFrames{};
    };

    //! \brief Inspect audio buffers, select a fbank path, and derive per-clip encoder row counts.
    //! \param[in] audioBuffers Input audio data carrying raw PCM (``AudioData::pcm``)
    //! \param[out] audioTokenLengths Output token lengths for each audio clip
    //! \param[in] stream CUDA stream for execution
    //! \return True if inspection succeeded, false otherwise
    bool inspectAudio(
        std::vector<rt::audioUtils::AudioData> const& audioBuffers, std::vector<int64_t>& audioTokenLengths);

    //! \brief Encode all clips retained by inspectAudio into one offset-correct output tensor.
    bool encodePreparedAudio(cudaStream_t stream);

    //! \brief Materialize one prepared clip as a device mel tensor, using online fbank when available.
    bool produceClipMel(ClipPlan& plan, rt::Tensor& melSpec, cudaStream_t stream);

    //! \brief Encode one device mel tensor into the output at destRowOffset.
    bool encodeClipMel(rt::Tensor const& melSpec, int64_t destRowOffset, int64_t expectedRows, cudaStream_t stream);

    bool resizeEmbeddingForRows(int64_t rows);

    //! \brief Initialize persistent online-GPU-fbank state: load the CuTe DSL
    //!        GEMM module, pre-allocate every device buffer the fbank path
    //!        uses (weights, workspace, PCM staging, mel output) at the engine
    //!        kMAX profile bound, then fill the weight/table tensors.
    //!        Best-effort: returns false (and leaves the runner on
    //!        the CPU MelExtractor path) when the GEMM has no variant for the
    //!        current SM, or mel geometry is not the 128×201 Whisper Slaney
    //!        bank. The mel filter is reused from ``mFeMel`` (single source of
    //!        truth — no external mel_filter.bin), so ``mFeMel`` must already
    //!        be bound (call after validateAndFillConfig).
    //! \param[in] stream CUDA stream for the H2D uploads
    //! \return True if online fbank is ready, false to fall back to CPU mel
    bool initFbankResources(cudaStream_t stream);

    //! \brief Run the online GPU fbank for one clip, if it is applicable.
    //! \details Gates on online-fbank readiness, the engine mel width, and the
    //!          sample rate, then uploads PCM, frames it, and runs fbankWhisper.
    //!          Any gate miss or kernel failure returns false so the caller
    //!          falls back to the CPU MelExtractor; the [1, nMel, T] FP16
    //!          contract is identical on both paths.
    //! \param[in] pcm Host FP32 mono PCM for this clip
    //! \param[out] melSpec [1, nMel, T] FP16 GPU mel, written only on success
    //! \param[in] stream CUDA stream
    //! \return True if melSpec was produced on the GPU, false to use the CPU path
    bool tryOnlineGpuFbank(rt::audio::AudioPCM const& pcm, rt::Tensor& melSpec, cudaStream_t stream);

    //! \brief Return the planned online-fbank frame count, or zero when this clip must use CPU mel extraction.
    int32_t plannedGpuFbankFrames(rt::audio::AudioPCM const& pcm) const;

    //! \brief Tokenize text and insert audio tokens
    //! \param[in] request LLM generation request
    //! \param[out] batchInputIds Batched input IDs after tokenization and audio token insertion
    //! \param[in] audioTokenLengths Token lengths for each audio clip
    //! \param[in] tokenizer Tokenizer for text encoding
    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchInputIds,
        std::vector<int64_t> const& audioTokenLengths, tokenizer::Tokenizer const* tokenizer);

    //! \brief Initialize MRope cos/sin cache with sequential position IDs (T=H=W=[0,1,2,...])
    //! \details Used for audio+text only models (e.g. Qwen3-ASR) where all 3 MRope dimensions
    //!          use identical sequential positions. Skipped when mConfig.mropeTheta == 0.
    //! \param[in] activeBatchSize Number of active sequences in the batch
    //! \param[in,out] ropeRotaryCosSinDevice RoPE cache tensor to fill
    //! \param[in] stream CUDA stream for execution
    //! \return True if initialization succeeded or was skipped, false on failure
    bool initializeSequentialMRopeCache(
        int64_t activeBatchSize, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream);

    AudioConfig mConfig{};                               //!< Audio encoder configuration
    rt::audio::MelExtractor mFeMel;                      //!< FE for PCM→mel; family bound by validateAndFillConfig
    std::unique_ptr<nvinfer1::ICudaEngine> mAudioEngine; //!< Audio encoder TensorRT engine
    std::unique_ptr<nvinfer1::IExecutionContext> mAudioContext; //!< Audio encoder execution context
    rt::Tensor mPaddedFeature{};      //!< [num_chunks, mel_bins, max_chunk_len] Padded audio chunks
    rt::Tensor mPaddedMaskAfterCNN{}; //!< [num_chunks, max_len_after_cnn] Mask for valid tokens
    rt::Tensor mPaddedMaskIndices{};  //!< [num_valid_elements, 2] Nonzero indices from mask
    rt::Tensor mAudioAttentionMask{}; //!< [num_attention_elems, num_attention_elems] Block-diagonal attention mask
    rt::Tensor mCuSeqlens{};          //!< [num_windows + 1] Cumulative sequence lengths (optional TRT input)
    rt::Tensor mCuSeqlensHost{};      //!< Host staging for mCuSeqlens
    rt::Tensor mKvLengths{};          //!< [num_windows + 1] Separate copy of cu_seqlens (TRT-native attention)
    rt::Tensor mAudioEmbedding{};     //!< [num_audio_tokens, hidden_dim] Audio encoder output
    bool mHasCuSeqlens{false};        //!< True when engine exposes cu_seqlens input
    bool mHasKvLengths{false};        //!< True when engine exposes kv_lengths input

    // Online GPU fbank state (persistent across clips). When mFbankReady is
    // false (unsupported SM / non-128 melBins / GEMM module unavailable) the
    // runner transparently falls back to the CPU MelExtractor (mFeMel) — the
    // [1,128,T] FP16 mel contract is identical on both paths. Every device
    // buffer below is allocated once by initFbankResources at the engine kMAX
    // profile bound.
    bool mFbankReady{false};                          //!< true once initFbankResources succeeds
    rt::audioUtils::FbankResources mFbankResources{}; //!< Weights/tables, params, pre-allocated workspace
    rt::Tensor mPcmF32Device{};                       //!< [maxPcmSamples] Float — PCM staging, reshaped to [N] per clip
    rt::Tensor mMelSpecDevice{}; //!< [1, nMel, maxFrames] Half — fbank output backing store, viewed at [1, nMel, T_out]
    std::vector<ClipPlan> mPreparedClipPlans;   //!< Request-local clip inputs retained between inspect and infer
    std::vector<int64_t> mPreparedTokenLengths; //!< Encoder row counts aligned with mPreparedClipPlans
};

} // namespace rt
} // namespace trt_edgellm
