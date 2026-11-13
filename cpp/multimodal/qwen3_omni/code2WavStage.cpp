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

#include "multimodal/qwen3_omni/code2WavStage.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/multiStage/stageRegistry.h"
#include "runtime/streaming.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace qwen3_omni
{

namespace
{

constexpr char const* kRunnerName = "qwen3_omni.code2wav";
constexpr char const* kAudioPort = "audio";
constexpr char const* kCodesInput = "codes";

//! One codec frame is 1920 samples at 24 kHz, so this is two seconds of audio: small
//! enough that the first window lands early, wide enough that the seam cost stays amortised.
constexpr int32_t kDefaultChunkFrames = 25;

//! A caller that cancelled its audio channel is no longer listening.
bool listenerGone(multiStage::StageExecContext const& ctx)
{
    return ctx.audioSink != nullptr && ctx.audioSink->isCancelled();
}

//! A talker and a vocoder that disagree on the codebook count were built from different
//! checkpoints. Padding or dropping lanes would vocode noise instead of failing.
void checkCodecLayout(multiStage::PortValue const& codes, int32_t numQuantizers, std::string const& stageId)
{
    ELLM_CHECK(codes.shape[1] == numQuantizers && codes.elementBytes == static_cast<int32_t>(sizeof(int32_t)),
        "stage '" + stageId + "': the codec stream carries " + std::to_string(codes.shape[1]) + " codes of "
            + std::to_string(codes.elementBytes) + " bytes per frame, but the vocoder takes "
            + std::to_string(numQuantizers) + " int32 codes");
}

} // namespace

Code2WavStage::Code2WavStage(multiStage::StageInitContext const& init)
    : mArtifactDir(init.artifactDir)
    , mStream(init.stream)
    , mMaxBatch(init.maxBatch)
{
    ELLM_CHECK(init.config != nullptr, "vocoder stage built without a StageConfig");
    mStageId = init.config->id;
    auto const& config = init.config->config;
    if (config.contains("checkpoint"))
    {
        auto const value = config.at("checkpoint").get<std::string>();
        mCheckpointDir = value.empty() ? std::string{} : (std::filesystem::path(mArtifactDir) / value).string();
    }
    mChunkFrames = config.value("chunk_frames", 0);
    mWholeUtterance = config.value("whole_utterance", false);
    mStreamDepth = config.value("stream_depth", 4);
    // The ring must hold at least as many distinct payloads as a consumer can be
    // holding, or a published window is overwritten while it is still being read.
    ELLM_CHECK(mStreamDepth >= 2, "vocoder stage 'stream_depth' must be at least 2");
    ELLM_CHECK(mMaxBatch >= 1, "vocoder stage needs a positive max_batch");
}

Code2WavStage::~Code2WavStage() = default;

Code2WavStage::Row& Code2WavStage::rowFor(multiStage::StageExecContext const& ctx)
{
    ELLM_CHECK(ctx.slot != nullptr, "vocoder stage reached without a request slot");
    auto const slot = static_cast<size_t>(ctx.slot->batchSlot);
    ELLM_CHECK(ctx.slot->batchSlot >= 0 && slot < mRows.size(),
        "stage '" + mStageId + "': batch slot " + std::to_string(ctx.slot->batchSlot) + " is outside max_batch "
            + std::to_string(mMaxBatch));
    return mRows[slot];
}

int8_t* Code2WavStage::slotFor(multiStage::StageExecContext const& ctx, int32_t slot)
{
    ELLM_CHECK(ctx.slot != nullptr, "vocoder stage reached without a request slot");
    int64_t const memberBytes = mFirstSlotBytes + static_cast<int64_t>(mStreamDepth - 1) * mSlotBytes;
    int64_t const offset = slot == 0 ? 0 : mFirstSlotBytes + static_cast<int64_t>(slot - 1) * mSlotBytes;
    return mPcmRing.dataPointer<int8_t>() + static_cast<int64_t>(ctx.slot->batchSlot) * memberBytes + offset;
}

void Code2WavStage::preallocate()
{
    mVocoder = std::make_unique<Code2WavRunner>(mArtifactDir, mStream, mCheckpointDir);

    auto const& config = mVocoder->getConfig();
    mUpsampleRate = config.upsampleRate;
    mNumQuantizers = config.numQuantizers;
    mLeftContext = config.leftContextSize;
    if (mWholeUtterance)
    {
        // What qwen3_tts_inference windows an utterance longer than max_code_len by.
        mChunkFrames = config.chunkSize;
    }
    else if (mChunkFrames <= 0)
    {
        // Not opt_code_len: that is the profile's optimization point (300 frames, 24 s, by the
        // builder's default), and waiting that long for the first window is batch synthesis.
        mChunkFrames = std::min(kDefaultChunkFrames, config.chunkSize);
    }
    ELLM_CHECK(mChunkFrames > 0 && mUpsampleRate > 0 && mNumQuantizers > 0 && mLeftContext >= 0,
        "vocoder config is incomplete; chunk size, upsample rate and quantizer count must all be positive");
    ELLM_CHECK(mChunkFrames <= config.chunkSize,
        "vocoder stage 'chunk_frames' of " + std::to_string(mChunkFrames) + " exceeds what this engine was built for ("
            + std::to_string(config.chunkSize) + ")");

    // Every window must be one engine pass -- generateWaveformInto() rejects a longer span --
    // because the skip below relies on a single pass's tail trim.
    mSpanFrames = mChunkFrames + mLeftContext;
    int64_t const maxCodeLength = mVocoder->getMaxCodeLength();
    ELLM_CHECK(mSpanFrames <= maxCodeLength,
        "vocoder stage 'chunk_frames' of " + std::to_string(mChunkFrames) + " plus " + std::to_string(mLeftContext)
            + " frames of left context exceeds the engine's max_code_len of " + std::to_string(maxCodeLength));
    if (mWholeUtterance)
    {
        mSpanFrames = static_cast<int32_t>(maxCodeLength);
    }

    mWaveformType = mVocoder->getWaveformDataType();
    ELLM_CHECK(mWaveformType == trt_edgellm::DataType::kHALF || mWaveformType == trt_edgellm::DataType::kFLOAT,
        "vocoder waveform must be FP16 or FP32");
    mSampleBytes = static_cast<int32_t>(rt::utils::getTypeSize(mWaveformType));

    // A one-pass utterance is its request's first window, and prepare() starts every request at slot 0.
    mSlotBytes = static_cast<int64_t>(mChunkFrames) * mUpsampleRate * mSampleBytes;
    mFirstSlotBytes = mWholeUtterance ? maxCodeLength * mUpsampleRate * mSampleBytes : mSlotBytes;
    int64_t const memberBytes = mFirstSlotBytes + static_cast<int64_t>(mStreamDepth - 1) * mSlotBytes;
    mPcmRing = rt::Tensor({static_cast<int64_t>(mMaxBatch) * memberBytes}, rt::DeviceType::kCPU,
        trt_edgellm::DataType::kINT8, "code2wavPcmRing");
    mRows.resize(static_cast<size_t>(mMaxBatch));
    int64_t const spanSamples = static_cast<int64_t>(mSpanFrames) * mUpsampleRate;
    mWindowPcm = rt::Tensor({spanSamples}, rt::DeviceType::kCPU, mWaveformType, "code2wavWindowPcm");

    // Full-span vectors up front: step() only ever resizes downwards inside this
    // capacity, so no lane reallocates once a request is running.
    mCodeLanes.assign(static_cast<size_t>(mNumQuantizers), std::vector<int32_t>(static_cast<size_t>(mSpanFrames), 0));
}

bool Code2WavStage::admit(multiStage::StageExecContext const&, std::string&)
{
    // Nothing per-request to reject: the window geometry is fixed at load and the codec
    // stream's shape is only known once the producer has published something.
    return true;
}

void Code2WavStage::prepare(multiStage::StageExecContext& ctx)
{
    rowFor(ctx) = Row{};
}

multiStage::StageStatus Code2WavStage::step(multiStage::StageExecContext& ctx)
{
    if (ctx.slot->abandon.requested() || listenerGone(ctx))
    {
        return multiStage::StageStatus::kDone;
    }
    auto& row = rowFor(ctx);

    // Re-read every step: the codec binding grows as the producer publishes.
    auto const* codes = ctx.in != nullptr ? ctx.in->find(kCodesInput) : nullptr;
    if (codes == nullptr || codes->data == nullptr || codes->shape.size() != 2)
    {
        return multiStage::StageStatus::kBlocked;
    }
    checkCodecLayout(*codes, mNumQuantizers, mStageId);
    row.totalFrames = static_cast<int32_t>(codes->shape[0]);

    if (row.cursor >= row.totalFrames)
    {
        // Drained everything published so far. Only the producer knows whether that is
        // "not yet" or "never", which is what complete says.
        return codes->complete ? multiStage::StageStatus::kDone : multiStage::StageStatus::kBlocked;
    }

    // A short window costs a seam and a wasted enqueue, so wait for a whole chunk unless
    // the stream has ended and this is the remainder.
    if ((mWholeUtterance || row.totalFrames - row.cursor < mChunkFrames) && !codes->complete)
    {
        return multiStage::StageStatus::kBlocked;
    }

    if (!vocodeNextWindow(ctx, /*complete=*/false))
    {
        return multiStage::StageStatus::kFailed;
    }
    if (row.cursor < row.totalFrames)
    {
        return multiStage::StageStatus::kMore;
    }
    // Same rule as the drained branch above: an empty remainder only means the end once the
    // producer says so.
    return codes->complete ? multiStage::StageStatus::kDone : multiStage::StageStatus::kBlocked;
}

void Code2WavStage::stepCohort(
    std::vector<multiStage::StageExecContext*> const& members, std::vector<multiStage::StageStatus>& statuses)
{
    // In turn, not batched: every member's cursor and ring are its own, and the shared window
    // scratch is copied into a member's slot before the next member vocodes.
    for (size_t index = 0; index < members.size(); ++index)
    {
        statuses[index] = step(*members[index]);
    }
}

bool Code2WavStage::vocodeNextWindow(multiStage::StageExecContext& ctx, bool complete)
{
    auto const* codes = ctx.in != nullptr ? ctx.in->find(kCodesInput) : nullptr;
    ELLM_CHECK(codes != nullptr && codes->data != nullptr && codes->shape.size() == 2,
        "vocoder asked for a window with no codec binding");
    auto& row = rowFor(ctx);
    auto const* values = static_cast<int32_t const*>(codes->data);

    int32_t const leftContext = std::min(row.cursor, mLeftContext);
    int32_t const begin = row.cursor - leftContext;
    // A whole utterance that fits the engine is one pass, with no seam at all.
    bool const onePass = mWholeUtterance && row.cursor == 0 && row.totalFrames <= mSpanFrames;
    int32_t const end = onePass ? row.totalFrames : std::min(row.cursor + mChunkFrames, row.totalFrames);
    auto const span = static_cast<size_t>(end - begin);
    ELLM_CHECK(span <= static_cast<size_t>(mSpanFrames), "vocoder window is wider than the preallocated span");

    // Transposed into lane-major, which is the layout the engine takes. The caller has checked
    // that a frame holds exactly mNumQuantizers codes.
    for (int32_t lane = 0; lane < mNumQuantizers; ++lane)
    {
        auto& laneCodes = mCodeLanes[static_cast<size_t>(lane)];
        laneCodes.resize(span);
        for (size_t frame = 0; frame < span; ++frame)
        {
            auto const offset = (static_cast<size_t>(begin) + frame) * static_cast<size_t>(mNumQuantizers);
            laneCodes[frame] = values[offset + static_cast<size_t>(lane)];
        }
    }

    int64_t produced = 0;
    if (!mVocoder->generateWaveformInto(mCodeLanes, mWindowPcm, produced, ctx.stream))
    {
        LOG_ERROR("Stage '%s' failed to vocode frames [%d, %d) of request %llu", mStageId.c_str(), begin, end,
            static_cast<unsigned long long>(ctx.slot->requestId));
        return false;
    }

    // The left-context samples were vocoded only to keep the seam clean; they belong to
    // the window that was already published. The engine trims its own tail, so deriving the
    // skip from what it actually produced is what hands the trimmed samples over: they sit
    // at the head of this window, and skipping a nominal leftContext drops them every seam.
    int64_t const fresh
        = std::min(produced, static_cast<int64_t>(end - row.cursor) * static_cast<int64_t>(mUpsampleRate));
    int64_t const skip = produced - fresh;
    ELLM_CHECK(fresh * mSampleBytes <= (row.slot == 0 ? mFirstSlotBytes : mSlotBytes),
        "vocoder window of " + std::to_string(fresh) + " samples exceeds the preallocated PCM slot");

    auto* destination = slotFor(ctx, row.slot);
    std::memcpy(destination, static_cast<int8_t const*>(mWindowPcm.rawPointer()) + skip * mSampleBytes,
        static_cast<size_t>(fresh * mSampleBytes));

    bindWindow(ctx, row.slot, fresh, end - row.cursor, complete);
    row.slot = (row.slot + 1) % mStreamDepth;
    row.cursor = end;
    return true;
}

void Code2WavStage::bindWindow(
    multiStage::StageExecContext& ctx, int32_t slot, int64_t samples, int32_t frames, bool complete)
{
    auto* destination = slotFor(ctx, slot);
    multiStage::PortValue window{destination, {samples}, multiStage::PortKind::kAudio, mSampleBytes};
    window.complete = complete;
    ctx.out->bind(kAudioPort, window);
    emitAudio(ctx, destination, samples, frames, complete);
}

void Code2WavStage::emitAudio(
    multiStage::StageExecContext& ctx, int8_t const* window, int64_t samples, int32_t frames, bool complete)
{
    if (ctx.audioSink == nullptr)
    {
        return;
    }

    AudioChunk chunk;
    chunk.isFinal = complete;
    chunk.numFrames = frames;
    chunk.sampleRate = mVocoder->getConfig().sampleRate;
    if (samples > 0 && window != nullptr)
    {
        // AudioChunk carries int16; the engine emits FP16 or FP32, so the narrowing
        // happens once here rather than in every consumer.
        chunk.pcm16.resize(static_cast<size_t>(samples) * sizeof(int16_t));
        auto* out = reinterpret_cast<int16_t*>(chunk.pcm16.data());
        for (int64_t i = 0; i < samples; ++i)
        {
            float const value = mWaveformType == trt_edgellm::DataType::kHALF
                ? __half2float(reinterpret_cast<__half const*>(window)[i])
                : reinterpret_cast<float const*>(window)[i];
            float const scaled = std::clamp(value, -1.0F, 1.0F) * 32767.0F;
            out[i] = static_cast<int16_t>(std::lround(scaled));
        }
    }
    ctx.audioSink->push(std::move(chunk));
}

void Code2WavStage::finish(multiStage::StageExecContext& ctx, multiStage::FinishReason reason)
{
    // Frames the producer published after this stage's last step would otherwise be
    // dropped, truncating the utterance. Only on the completed path and for a caller still
    // listening: a cancelled request wants the audio to stop, not to run to the end first.
    // A failure here is rethrown only after the stream has been ended below.
    auto& row = rowFor(ctx);
    std::exception_ptr drainError;
    if (ctx.out != nullptr && reason == multiStage::FinishReason::kCompleted && !listenerGone(ctx))
    {
        try
        {
            auto const* codes = ctx.in != nullptr ? ctx.in->find(kCodesInput) : nullptr;
            if (codes != nullptr && codes->data != nullptr && codes->shape.size() == 2)
            {
                checkCodecLayout(*codes, mNumQuantizers, mStageId);
                row.totalFrames = static_cast<int32_t>(codes->shape[0]);
                while (row.cursor < row.totalFrames)
                {
                    if (!vocodeNextWindow(ctx, /*complete=*/false))
                    {
                        drainError = std::make_exception_ptr(std::runtime_error("stage '" + mStageId
                            + "' failed to vocode the tail of request " + std::to_string(ctx.slot->requestId)));
                        break;
                    }
                }
            }
        }
        catch (...)
        {
            drainError = std::current_exception();
        }
    }

    // An empty final window and an explicit end of stream on every path: a cancelled or failed
    // request must still reach the audio terminal, or a listener waits forever on a stream that
    // stopped. A consumer may key on either AudioChunk::isFinal or isFinished().
    if (ctx.out != nullptr)
    {
        bindWindow(ctx, row.slot, 0, /*frames=*/0, /*complete=*/true);
    }
    if (ctx.audioSink != nullptr)
    {
        ctx.audioSink->finish();
    }
    if (reason != multiStage::FinishReason::kCompleted)
    {
        LOG_INFO("Vocoder request %llu ended: %s after %d of %d frames",
            static_cast<unsigned long long>(ctx.slot->requestId), multiStage::toString(reason), row.cursor,
            row.totalFrames);
    }
    if (drainError != nullptr)
    {
        std::rethrow_exception(drainError);
    }
}

void Code2WavStage::reset()
{
    for (auto& row : mRows)
    {
        row = Row{};
    }
}

std::unique_ptr<multiStage::IStage> createCode2WavStage(multiStage::StageInitContext const& init)
{
    return std::make_unique<Code2WavStage>(init);
}

multiStage::RunnerDecl code2WavDeclaration()
{
    multiStage::RunnerDecl decl;
    decl.name = kRunnerName;
    decl.kind = multiStage::StageKind::kIterative;
    decl.ports = {{kAudioPort, multiStage::PortKind::kAudio}};
    decl.inputs = {kCodesInput};
    decl.batches = true;
    return decl;
}

} // namespace qwen3_omni
} // namespace rt
} // namespace trt_edgellm
