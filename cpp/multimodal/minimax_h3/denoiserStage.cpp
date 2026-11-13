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

#include "multimodal/minimax_h3/minimaxH3Stages.h"

#include "common/checkMacros.h"
#include "kernels/minimaxH3Kernels/minimaxH3Kernels.h"
#include "profiling/timer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace minimax_h3
{

namespace
{

constexpr char const* kRunnerName = "minimax_h3.denoiser";
constexpr char const* kTextHiddenInput = "text_hidden";
constexpr char const* kStyleInput = "style";
constexpr char const* kSamplingInput = "sampling";
constexpr char const* kAudioLatentsPort = "audio_latents";
constexpr char const* kVideoLatentsPort = "video_latents";
std::string const kTimerStage = "minimax_h3_denoise_step";

constexpr int32_t kDEFAULT_HEIGHT{768};
constexpr int32_t kDEFAULT_WIDTH{1344};
constexpr int32_t kDEFAULT_FRAMES{124};
constexpr int32_t kAUDIO_LATENTS_PER_SECOND{40};
constexpr uint64_t kVIDEO_NOISE_STREAM{0};
constexpr uint64_t kAUDIO_NOISE_STREAM{1};

std::vector<std::array<float, 2>> readSchedule(nlohmann::json const& schedule)
{
    std::vector<std::array<float, 2>> steps;
    for (auto const& step : schedule)
    {
        steps.push_back({step.at(0).get<float>(), step.at(1).get<float>()});
    }
    return steps;
}

} // namespace

DenoiserStage::DenoiserStage(multiStage::StageInitContext const& init)
    : mArtifactDir(init.artifactDir)
    , mStream(init.stream)
{
}

void DenoiserStage::preallocate()
{
    mEngine = std::make_unique<MiniMaxH3Engine>(mArtifactDir, "gen", mStream);
    auto const& config = mEngine->config();
    mMaxTextTokens = config.at("max_text_tokens");
    mMaxAudioRows = config.at("max_audio_tokens");
    mMaxVideoRows = config.at("max_video_tokens");
    mLatentChannels = config.at("latent_channels");
    mAudioChannels = config.at("audio_channels");
    mVideoSchedule = readSchedule(config.at("video_schedule"));
    mAudioSchedule = readSchedule(config.at("audio_schedule"));
    mLayout = VideoLayout::fromConfig(config.at("vae"));
    ELLM_CHECK(!mVideoSchedule.empty() && mVideoSchedule.size() == mAudioSchedule.size(),
        "MiniMax-H3 video and audio schedules must have the same, positive number of steps");

    int32_t const videoRowSize = mLatentChannels * 4;
    int64_t const maxSequence = static_cast<int64_t>(mMaxTextTokens) + mMaxAudioRows + mMaxVideoRows;
    auto const numSteps = static_cast<int32_t>(mVideoSchedule.size());
    mPositionIds = rt::Tensor({maxSequence, 3}, rt::DeviceType::kGPU, DataType::kFLOAT, "minimax_h3::position_ids");
    mStepIds = rt::Tensor({numSteps}, rt::DeviceType::kGPU, DataType::kINT32, "minimax_h3::step_ids");
    mVideoLatents = rt::Tensor(
        {1, mMaxVideoRows, videoRowSize}, rt::DeviceType::kGPU, DataType::kFLOAT, "minimax_h3::video_latents");
    mVideoVelocity = rt::Tensor(
        {1, mMaxVideoRows, videoRowSize}, rt::DeviceType::kGPU, DataType::kFLOAT, "minimax_h3::video_velocity");
    mAudioLatents = rt::Tensor(
        {1, mMaxAudioRows, mAudioChannels}, rt::DeviceType::kGPU, DataType::kFLOAT, "minimax_h3::audio_latents");
    mAudioVelocity = rt::Tensor(
        {1, mMaxAudioRows, mAudioChannels}, rt::DeviceType::kGPU, DataType::kFLOAT, "minimax_h3::audio_velocity");

    std::vector<int32_t> stepIds(numSteps);
    std::iota(stepIds.begin(), stepIds.end(), 0);
    CUDA_CHECK(
        cudaMemcpy(mStepIds.rawPointer(), stepIds.data(), stepIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
}

DenoiserStage::Plan DenoiserStage::plan(multiStage::PortBindings const* in) const
{
    auto const style = multiStage::jsonInput(in, kStyleInput);
    auto const sampling = multiStage::jsonInput(in, kSamplingInput);
    Plan result{};
    result.height = style.value("height", kDEFAULT_HEIGHT);
    result.width = style.value("width", kDEFAULT_WIDTH);
    int32_t const requestedFrames = style.value("num_frames", kDEFAULT_FRAMES);
    result.seed = sampling.value("seed", uint64_t{0});

    int32_t const shortSide = std::min(result.height, result.width);
    int32_t const longSide = std::max(result.height, result.width);
    ELLM_CHECK(result.height % kFRAME_ALIGNMENT == 0 && result.width % kFRAME_ALIGNMENT == 0
            && shortSide >= mLayout.tilePixels() && longSide <= kMAX_ASPECT_RATIO * shortSide
            && static_cast<int64_t>(result.height) * result.width <= kMAX_FRAME_PIXELS,
        "MiniMax-H3 height and width must be multiples of 32, at least 256, within a 4:1 aspect ratio, and at most "
        "1344x768 pixels");
    ELLM_CHECK(requestedFrames > 0, "MiniMax-H3 needs at least one frame");

    int32_t const clipFrames = mLayout.clipFrames();
    int32_t const numClips = std::max(1, (requestedFrames - mLayout.chunkFrameOverlap + clipFrames - 1) / clipFrames);
    result.numFrames = numClips * clipFrames + mLayout.chunkFrameOverlap;
    result.latentFrames = mLayout.latentFrames(numClips);
    result.latentHeight = result.height / mLayout.spatialRatio;
    result.latentWidth = result.width / mLayout.spatialRatio;
    result.videoRows = result.latentFrames * (result.latentHeight / 2) * (result.latentWidth / 2);
    result.audioFrames
        = static_cast<int32_t>(std::lround(static_cast<double>(result.numFrames) * kAUDIO_LATENTS_PER_SECOND / kFPS));
    ELLM_CHECK(result.videoRows <= mMaxVideoRows && 2 * result.audioFrames <= mMaxAudioRows,
        "MiniMax-H3 request needs " + std::to_string(result.videoRows) + " video and "
            + std::to_string(2 * result.audioFrames) + " audio rows; the engine holds " + std::to_string(mMaxVideoRows)
            + " and " + std::to_string(mMaxAudioRows));
    return result;
}

bool DenoiserStage::admit(multiStage::StageExecContext const& ctx, std::string& reason)
{
    try
    {
        static_cast<void>(plan(ctx.in));
    }
    catch (std::exception const& error)
    {
        reason = error.what();
        return false;
    }
    return true;
}

void DenoiserStage::prepare(multiStage::StageExecContext& ctx)
{
    auto const& textHidden = ctx.in->at(kTextHiddenInput);
    ELLM_CHECK(textHidden.shape.size() == 3 && textHidden.shape[1] <= mMaxTextTokens,
        "MiniMax-H3 denoiser takes at most " + std::to_string(mMaxTextTokens) + " text rows");
    auto const textTokens = static_cast<int32_t>(textHidden.shape[1]);
    mPlan = plan(ctx.in);
    mStep = 0;

    int32_t const audioRows = 2 * mPlan.audioFrames;
    int32_t const sequence = textTokens + audioRows + mPlan.videoRows;
    int32_t const videoRowSize = mLatentChannels * 4;
    ELLM_CHECK(mPositionIds.reshape({sequence, 3}) && mVideoLatents.reshape({1, mPlan.videoRows, videoRowSize})
            && mVideoVelocity.reshape({1, mPlan.videoRows, videoRowSize})
            && mAudioLatents.reshape({1, audioRows, mAudioChannels})
            && mAudioVelocity.reshape({1, audioRows, mAudioChannels}),
        "Request does not fit the denoiser buffers");
    kernel::minimax_h3::ropePositions(mPositionIds.dataPointer<float>(),
        {textTokens, mPlan.audioFrames, mPlan.latentFrames, mPlan.latentHeight, mPlan.latentWidth}, ctx.stream);
    kernel::minimax_h3::randomNormal(mVideoLatents.dataPointer<float>(), mVideoLatents.getShape().volume(), mPlan.seed,
        kVIDEO_NOISE_STREAM, ctx.stream);
    kernel::minimax_h3::randomNormal(mAudioLatents.dataPointer<float>(), mAudioLatents.getShape().volume(), mPlan.seed,
        kAUDIO_NOISE_STREAM, ctx.stream);

    mEngine->bind("text_hidden", textHidden.data, {1, textTokens, textHidden.shape[2]});
    mEngine->bind("audio_rows", mAudioLatents);
    mEngine->bind("video_rows", mVideoLatents);
    mEngine->bind("position_ids", mPositionIds);
    mEngine->bind("audio_pred", mAudioVelocity);
    mEngine->bind("video_pred", mVideoVelocity);
}

multiStage::StageStatus DenoiserStage::step(multiStage::StageExecContext& ctx)
{
    if (ctx.slot->abandon.requested())
    {
        return multiStage::StageStatus::kDone;
    }
    {
        TIME_STAGE(kTimerStage, ctx.stream);
        mEngine->bind("step", mStepIds.dataPointer<int32_t>() + mStep, {1});
        mEngine->enqueue(ctx.stream);
        auto const& [videoSigma, videoRatio] = mVideoSchedule[mStep];
        auto const& [audioSigma, audioRatio] = mAudioSchedule[mStep];
        kernel::minimax_h3::eulerStep({mVideoLatents.dataPointer<float>(), mVideoVelocity.dataPointer<float>(),
                                          mVideoLatents.getShape().volume(), videoSigma, videoRatio},
            {mAudioLatents.dataPointer<float>(), mAudioVelocity.dataPointer<float>(), mAudioLatents.getShape().volume(),
                audioSigma, audioRatio},
            ctx.stream);
    }
    if (++mStep < static_cast<int32_t>(mVideoSchedule.size()))
    {
        return multiStage::StageStatus::kMore;
    }
    ctx.out->bind(kAudioLatentsPort,
        multiStage::PortValue{mAudioLatents.rawPointer(), {2, mPlan.audioFrames, mAudioChannels},
            multiStage::PortKind::kEmbeds, static_cast<int32_t>(sizeof(float)), /*complete=*/true});
    ctx.out->bind(kVideoLatentsPort,
        multiStage::PortValue{mVideoLatents.rawPointer(),
            {mPlan.latentFrames, mPlan.latentHeight / 2, mPlan.latentWidth / 2, mLatentChannels * 4},
            multiStage::PortKind::kEmbeds, static_cast<int32_t>(sizeof(float)), /*complete=*/true});
    return multiStage::StageStatus::kDone;
}

void DenoiserStage::finish(multiStage::StageExecContext&, multiStage::FinishReason) {}

void DenoiserStage::reset()
{
    mPlan = Plan{};
    mStep = 0;
}

nlohmann::json DenoiserStage::diagnostics(multiStage::RequestSlot const&) const
{
    return {{"steps", mStep}, {"height", mPlan.height}, {"width", mPlan.width}, {"frames", mPlan.numFrames},
        {"video_rows", mPlan.videoRows}, {"audio_rows", 2 * mPlan.audioFrames}};
}

std::unique_ptr<multiStage::IStage> createDenoiserStage(multiStage::StageInitContext const& init)
{
    return std::make_unique<DenoiserStage>(init);
}

multiStage::RunnerDecl denoiserDeclaration()
{
    multiStage::RunnerDecl decl;
    decl.name = kRunnerName;
    decl.kind = multiStage::StageKind::kIterative;
    decl.ports
        = {{kAudioLatentsPort, multiStage::PortKind::kEmbeds}, {kVideoLatentsPort, multiStage::PortKind::kEmbeds}};
    decl.inputs = {kTextHiddenInput, kStyleInput, kSamplingInput};
    decl.samplingKeys = {"seed"};
    decl.styleFields = {"height", "width", "num_frames"};
    return decl;
}

} // namespace minimax_h3
} // namespace rt
} // namespace trt_edgellm
