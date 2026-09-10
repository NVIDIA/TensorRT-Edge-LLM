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
#include "runtime/audioLoader.h"
#include "runtime/melSpectrogram.h"
#include "tokenizer/tokenizer.h"

#include <NvInferRuntime.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

/*!
 * \brief Standalone offline transcription runtime for Nemotron-3.5-ASR
 *        (FastConformer encoder + RNN-T LSTM prediction network).
 *
 * Owns two TRT engines and drives the greedy RNN-T transducer loop:
 *
 *  - ``audio_encoder.engine``: ``(input_features [1,T_mel,128] fp16,
 *    prompt_ids [1] i64) → encoder_frames [1,T,640] fp16`` (dynamic T_mel,
 *    batch fixed at 1).
 *  - ``rnnt_step.engine``: one decode step ``(decoder_input_ids [1,1] i64,
 *    hidden_state/cell_state [L,1,H] fp16, encoder_frame [1,H] fp16) →
 *    (logits [1,V] fp16, present_hidden_state, present_cell_state)``.
 *
 * Loop semantics (must match HF greedy RNN-T exactly): the step engine
 * always runs; on a blank prediction the encoder frame pointer advances
 * and the present LSTM states are DISCARDED (state ping-pong buffers are
 * not swapped — blank never updates the prediction network); on a
 * non-blank prediction the token is emitted, the present states are
 * adopted (buffer swap) and the frame pointer stays. A forced advance
 * after ``max_symbols_per_step`` consecutive non-blanks bounds the loop;
 * decoding stops when the frame pointer passes the last encoder frame.
 *
 * Unlike the LLM multimodal audio runners, encoder output frames feed the
 * RNN-T joint network directly — they are never spliced into LLM prompt
 * embeddings.
 */
class NemotronAsrRuntime
{
public:
    //! \brief Transcription result.
    struct Result
    {
        std::vector<int32_t> tokens; //!< Emitted non-blank token ids.
        std::string text;            //!< Detokenized transcript (special tokens kept: language tag).
        int64_t numMelFrames{0};     //!< Mel frames fed to the encoder.
        int64_t numEncoderFrames{0}; //!< Encoder output frames (T).
        int64_t numDecodeSteps{0};   //!< Step-engine invocations (~T + tokens).
    };

    //! \brief Per-phase latency breakdown (populated only when a non-null
    //!        pointer is passed to ``transcribe``). All values in milliseconds.
    //! The phases are the ASR analogues of an LLM's prefill/decode split:
    //! ``encoderMs`` is the single encoder forward (prefill-like) and
    //! ``decodeMs`` is the RNN-T greedy step loop (decode-like).
    struct Timings
    {
        double melMs{0.0};     //!< CPU mel/FFT feature extraction (host).
        double encoderMs{0.0}; //!< Feature H2D + encoder forward (GPU, synced).
        double decodeMs{0.0};  //!< RNN-T greedy step loop (GPU, host-synced/step).
    };

    /*!
     * \brief Load engines, config and tokenizer.
     *
     * \param engineDir Directory containing ``config.json`` and either the
     *        direct-builder ``audio/audio_encoder.engine`` plus
     *        ``rnnt/rnnt_step.engine`` layout, or both engines at its root.
     * \param tokenizerDir Directory containing ``tokenizer.json``.
     * \param stream CUDA stream used for allocations.
     * \throws std::runtime_error on missing files or engine/config mismatch.
     */
    NemotronAsrRuntime(
        std::filesystem::path const& engineDir, std::filesystem::path const& tokenizerDir, cudaStream_t stream);

    ~NemotronAsrRuntime() noexcept;

    NemotronAsrRuntime(NemotronAsrRuntime const&) = delete;
    NemotronAsrRuntime& operator=(NemotronAsrRuntime const&) = delete;

    /*!
     * \brief Transcribe one utterance (offline, batch 1).
     *
     * \param pcm Mono float32 PCM at 16 kHz (see ``audio::loadAudioFile``).
     * \param promptId Language-prompt index (``defaultPromptId()`` = auto
     *        language detection; the model then emits an ``<xx-XX>`` tag).
     * \param stream CUDA stream for the full pipeline.
     * \param timings Optional out-parameter; when non-null, receives the
     *        per-phase latency breakdown. Passing it forces a stream sync
     *        after the encoder to isolate the phases, so leave it null on the
     *        latency-sensitive path.
     * \return Transcription result.
     */
    Result transcribe(audio::AudioPCM const& pcm, int32_t promptId, cudaStream_t stream, Timings* timings = nullptr);

    //! \brief Prompt index for automatic language detection (config.json
    //!        ``default_prompt_id``).
    int32_t defaultPromptId() const noexcept
    {
        return mDefaultPromptId;
    }

    int64_t maxMelFrames() const noexcept
    {
        return mMaxMelFrames;
    }

    tokenizer::Tokenizer const& tokenizer() const noexcept
    {
        return mTokenizer;
    }

private:
    void loadConfig(std::filesystem::path const& configPath);
    void allocateBuffers(cudaStream_t stream);

    //! Run mel extraction + encoder; returns the number of encoder frames.
    //! When ``timings`` is non-null, fills melMs/encoderMs and syncs the stream.
    int64_t runEncoder(
        audio::AudioPCM const& pcm, int32_t promptId, cudaStream_t stream, int64_t& numMelFrames, Timings* timings);

    //! Greedy transducer loop over ``numFrames`` encoder frames.
    //! When ``timings`` is non-null, fills decodeMs.
    Result decodeGreedy(int64_t numFrames, cudaStream_t stream, Timings* timings);

    // -- model hyperparameters (from config.json) ---------------------------
    int32_t mMelBins{128};
    int32_t mBlankTokenId{0};
    int32_t mVocabSize{0};
    int32_t mDecoderHiddenSize{0};
    int32_t mNumDecoderLayers{2};
    int32_t mMaxSymbolsPerStep{10};
    int32_t mDefaultPromptId{101};
    int32_t mNumPrompts{128};

    // -- engine bounds -------------------------------------------------------
    int64_t mMinMelFrames{0};
    int64_t mMaxMelFrames{0};
    int64_t mMaxEncoderFrames{0};

    // -- TRT objects ---------------------------------------------------------
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEncoderEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mEncoderContext;
    std::unique_ptr<nvinfer1::ICudaEngine> mStepEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mStepContext;

    // -- front-end / tokenizer -----------------------------------------------
    audio::MelExtractor mMelExtractor;
    tokenizer::Tokenizer mTokenizer;

    // -- device buffers --------------------------------------------------------
    // The step engine is bound ONCE to fixed addresses so the whole step
    // (enqueueV3 + argmax) is CUDA-graphable: the current encoder frame is
    // D2D-copied into a staging buffer each iteration, and on emit the
    // present LSTM states are D2D-copied back over the current states.
    Tensor mInputFeatures;       //!< [1, maxMelFrames, melBins] fp16
    Tensor mPromptIds;           //!< [1] i64
    Tensor mEncoderFrames;       //!< [1, maxEncoderFrames, H] fp16
    Tensor mEncoderFrameStaging; //!< [1, H] fp16 — fixed step-engine input
    Tensor mDecoderInputIds;     //!< [1, 1] i64
    Tensor mHiddenState[2];      //!< [L, 1, H] fp16 — [0] current, [1] present
    Tensor mCellState[2];        //!< [L, 1, H] fp16 — [0] current, [1] present
    Tensor mLogits;              //!< [1, V] fp16
    Tensor mTokenOut;            //!< [1] i32 — argmax result

    // -- CUDA graph for one decode step (enqueueV3 + argmax) -------------------
    cudaGraph_t mStepGraph{nullptr};
    cudaGraphExec_t mStepGraphExec{nullptr};
    bool mStepGraphCaptureAttempted{false};
    bool mStepBindingsSet{false};

    // -- host staging ----------------------------------------------------------
    Tensor mMelHostFp16;   //!< [maxMelFrames, melBins] fp16 — H2D source for input_features
    Tensor mTokenHostI64;  //!< [1] i64 — H2D source for decoder_input_ids
    Tensor mPromptHostI64; //!< [1] i64 — H2D source for prompt_ids
    Tensor mTokenHostI32;  //!< [1] i32 — D2H destination for mTokenOut
};

} // namespace rt
} // namespace trt_edgellm
