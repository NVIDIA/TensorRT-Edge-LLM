/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "multimodal/qwen3_omni/code2WavRunner.h"
#include "runtime/multiStage/stage.h"
#include "runtime/multiStage/stageRegistry.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace qwen3_omni
{

/*!
 * @brief Vocoding, behind IStage.
 *
 * Drives Code2WavRunner as a stream: one step vocodes one window of codec frames and
 * publishes it, so audio starts playing before the codec sequence is finished.
 *
 * Each window carries left context that is vocoded and then dropped, which is what
 * keeps the seam between windows free of boundary artifacts.
 *
 * Config keys:
 *   chunk_frames   codec frames published per step; defaults to 25 and may not exceed the engine's
 *                  opt_code_len. Together with the vocoder's left context it must fit max_code_len.
 *   stream_depth   PCM slots per member behind the `audio` port; at least 2, and at least as many
 *                  windows as a reader of that port holds at once
 *   checkpoint     optional, checkpoint-backed engine weights
 *   whole_utterance  optional, false by default. True waits for the whole codec sequence and
 *                  vocodes it in one pass when it fits max_code_len, else in windows of the engine's
 *                  chunk size; chunk_frames is then not read. The first PCM slot grows to max_code_len.
 */
class Code2WavStage final : public multiStage::IStage
{
public:
    explicit Code2WavStage(multiStage::StageInitContext const& init);
    ~Code2WavStage() override;

    void preallocate() override;
    bool admit(multiStage::StageExecContext const& ctx, std::string& reason) override;
    void prepare(multiStage::StageExecContext& ctx) override;
    multiStage::StageStatus step(multiStage::StageExecContext& ctx) override;
    void stepCohort(std::vector<multiStage::StageExecContext*> const& members,
        std::vector<multiStage::StageStatus>& statuses) override;
    void finish(multiStage::StageExecContext& ctx, multiStage::FinishReason reason) override;
    void reset() override;

private:
    /*!
     * @brief One cohort member's own streaming position.
     *
     * Members are stepped in turn rather than batched, but each still needs its own cursor and
     * its own PCM slots: a window published on the `audio` port is borrowed by its reader until
     * read, so two members cannot share a ring.
     */
    struct Row
    {
        int32_t totalFrames{0};
        int32_t cursor{0};
        int32_t slot{0};
    };

    Row& rowFor(multiStage::StageExecContext const& ctx);
    //! Start of this member's PCM slot `slot` inside mPcmRing.
    int8_t* slotFor(multiStage::StageExecContext const& ctx, int32_t slot);
    //! Bind one PCM window out of the ring. `samples` of 0 is the terminating window.
    void bindWindow(multiStage::StageExecContext& ctx, int32_t slot, int64_t samples, int32_t frames, bool complete);
    //! Vocode this member's next window and publish it. False on engine failure.
    bool vocodeNextWindow(multiStage::StageExecContext& ctx, bool complete);
    //! Convert one window to little-endian int16 and hand it to the caller's channel.
    //! A no-op when no audio sink is attached, which is how a text-only request pays nothing.
    void emitAudio(
        multiStage::StageExecContext& ctx, int8_t const* window, int64_t samples, int32_t frames, bool complete);

    std::string mStageId;
    std::string mArtifactDir;
    std::string mCheckpointDir;
    cudaStream_t mStream{nullptr};
    int32_t mMaxBatch{1};
    std::unique_ptr<Code2WavRunner> mVocoder;

    int32_t mChunkFrames{0};
    int32_t mLeftContext{0};
    int32_t mStreamDepth{4};
    int32_t mUpsampleRate{0};
    int32_t mNumQuantizers{0};
    bool mWholeUtterance{false};
    int32_t mSpanFrames{0};     //!< The widest window a step can ask for: chunk plus left context, or max_code_len.
    int32_t mSampleBytes{0};    //!< Bytes per PCM sample of the engine's waveform output.
    int64_t mSlotBytes{0};      //!< One chunk of PCM: a window's left context is vocoded but never published.
    int64_t mFirstSlotBytes{0}; //!< Slot 0 also takes a one-pass utterance, which is a request's only window.
    trt_edgellm::DataType mWaveformType{trt_edgellm::DataType::kHALF};

    rt::Tensor mPcmRing;                          //!< maxBatch rings of streamDepth slots, host, claimed once.
    rt::Tensor mWindowPcm;                        //!< [spanFrames * upsampleRate] samples, host, claimed once.
    std::vector<std::vector<int32_t>> mCodeLanes; //!< [numQuantizers][spanFrames], host, capacity claimed once.
    std::vector<Row> mRows;
};

std::unique_ptr<multiStage::IStage> createCode2WavStage(multiStage::StageInitContext const& init);
//! What a code2wav stage offers a stage graph.
multiStage::RunnerDecl code2WavDeclaration();

} // namespace qwen3_omni
} // namespace rt
} // namespace trt_edgellm
