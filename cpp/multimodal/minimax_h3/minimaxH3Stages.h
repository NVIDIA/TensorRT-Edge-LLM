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
#include "multimodal/minimax_h3/minimaxH3Engine.h"
#include "runtime/multiStage/stage.h"
#include "runtime/multiStage/stageRegistry.h"
#include "tokenizer/tokenizer.h"

#include <array>
#include <cstdint>
#include <cuda_fp16.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace minimax_h3
{

//! Frames per second of the generated video.
constexpr int32_t kFPS{24};
//! Frame sides are multiples of this many pixels.
constexpr int32_t kFRAME_ALIGNMENT{32};
//! Decode buffers cover up to this many pixels per frame, the largest supported 1344x768 frame.
constexpr int64_t kMAX_FRAME_PIXELS{1344 * 768};
//! The longer side of a frame is at most this many times the shorter one.
constexpr int32_t kMAX_ASPECT_RATIO{4};

//! \brief How the video VAE splits a request into clips and tiles; the ``config.json`` of the VAE engine.
//!
//! The VAE decodes ``tileLatentFrames`` latent frames at a time, one clip of ``chunkLatentFrames`` per step,
//! and cross-fades ``chunkFrameOverlap`` frames into the next clip.
struct VideoLayout
{
    int32_t spatialRatio{0};
    int32_t temporalRatio{0};
    int32_t tileLatentFrames{0};
    int32_t tileLatentSize{0};
    int32_t tileMinOverlap{0};
    int32_t chunkLatentFrames{0};
    int32_t chunkFramePadding{0};
    int32_t chunkFrameOverlap{0};

    static VideoLayout fromConfig(nlohmann::json const& config);

    int32_t tilePixels() const noexcept
    {
        return tileLatentSize * spatialRatio;
    }

    //! Frames each clip adds to the video.
    int32_t clipFrames() const noexcept
    {
        return chunkLatentFrames * temporalRatio - chunkFramePadding;
    }

    //! Latent frames of a video of @p numClips clips.
    int32_t latentFrames(int32_t numClips) const noexcept
    {
        return numClips * chunkLatentFrames + tileLatentFrames - chunkLatentFrames;
    }
};

/*!
 * @brief Prompt to the hidden state that conditions the denoiser.
 *
 * Runs the Qwen3-VL text tower of ``text_encoder.engine`` on the ``prompt`` input, UTF-8 text or INT32
 * token ids, and publishes ``text_hidden``: FP16 ``[1, tokens, hidden]`` on the device. The tokenizer
 * is read from the artifact directory.
 */
class TextEncoderStage final : public multiStage::IStage
{
public:
    explicit TextEncoderStage(multiStage::StageInitContext const& init);

    void preallocate() override;
    bool admit(multiStage::StageExecContext const& ctx, std::string& reason) override;
    void prepare(multiStage::StageExecContext& ctx) override;
    multiStage::StageStatus step(multiStage::StageExecContext& ctx) override;
    void finish(multiStage::StageExecContext& ctx, multiStage::FinishReason reason) override;
    void reset() override;
    nlohmann::json diagnostics(multiStage::RequestSlot const& slot) const override;

private:
    std::string mArtifactDir;
    cudaStream_t mStream{nullptr};
    tokenizer::Tokenizer mTokenizer;
    std::unique_ptr<MiniMaxH3Engine> mEngine;
    int32_t mMaxTokens{0};
    int32_t mTokens{0};

    rt::Tensor mInputIds;     //!< [1, maxTokens] INT32
    rt::Tensor mInputIdsHost; //!< Pinned staging of mInputIds.
    rt::Tensor mTextHidden;   //!< [1, maxTokens, hidden] FP16
};

/*!
 * @brief Rectified-flow denoising of the packed ``[text | audio | video]`` sequence, one step per call.
 *
 * Reads the frame ``height``, ``width``, and ``num_frames`` from the ``style`` input and the noise
 * ``seed`` from ``sampling``. After the last step of ``gen.engine`` it publishes FP32 device latents:
 * ``audio_latents`` ``[2, frames, channels]`` (left, then right) and ``video_latents``
 * ``[frames, height / 2, width / 2, channels * 4]`` over the latent grid.
 */
class DenoiserStage final : public multiStage::IStage
{
public:
    explicit DenoiserStage(multiStage::StageInitContext const& init);

    void preallocate() override;
    bool admit(multiStage::StageExecContext const& ctx, std::string& reason) override;
    void prepare(multiStage::StageExecContext& ctx) override;
    multiStage::StageStatus step(multiStage::StageExecContext& ctx) override;
    void finish(multiStage::StageExecContext& ctx, multiStage::FinishReason reason) override;
    void reset() override;
    nlohmann::json diagnostics(multiStage::RequestSlot const& slot) const override;

private:
    //! The request's video and audio extents.
    struct Plan
    {
        int32_t height{0};
        int32_t width{0};
        int32_t numFrames{0};
        int32_t latentFrames{0};
        int32_t latentHeight{0};
        int32_t latentWidth{0};
        int32_t videoRows{0};
        int32_t audioFrames{0};
        uint64_t seed{0};
    };

    //! Resolve the request's size and seed; throws naming the first limit it breaks.
    Plan plan(multiStage::PortBindings const* in) const;

    std::string mArtifactDir;
    cudaStream_t mStream{nullptr};
    std::unique_ptr<MiniMaxH3Engine> mEngine;
    VideoLayout mLayout;
    int32_t mMaxTextTokens{0};
    int32_t mMaxAudioRows{0};
    int32_t mMaxVideoRows{0};
    int32_t mLatentChannels{0};
    int32_t mAudioChannels{0};
    std::vector<std::array<float, 2>> mVideoSchedule; //!< Per step: sigma from the timestep, sigma ratio.
    std::vector<std::array<float, 2>> mAudioSchedule;

    Plan mPlan;
    int32_t mStep{0};

    rt::Tensor mPositionIds;  //!< [sequence, 3] FP32 (t, h, w) RoPE positions, written on the device.
    rt::Tensor mStepIds;      //!< [steps] INT32 0..steps-1; each step binds one element.
    rt::Tensor mVideoLatents; //!< [1, videoRows, channels * 4] FP32
    rt::Tensor mVideoVelocity;
    rt::Tensor mAudioLatents; //!< [1, 2 * audioFrames, audioChannels] FP32
    rt::Tensor mAudioVelocity;
};

/*!
 * @brief Stereo waveform from the denoised ``latents`` of the audio VAE, ``audio_decoder.engine``.
 *
 * Emits the request's whole waveform on the ``audio`` terminal as one interleaved 16-bit chunk.
 */
class AudioDecoderStage final : public multiStage::IStage
{
public:
    explicit AudioDecoderStage(multiStage::StageInitContext const& init);

    void preallocate() override;
    bool admit(multiStage::StageExecContext const& ctx, std::string& reason) override;
    void prepare(multiStage::StageExecContext& ctx) override;
    multiStage::StageStatus step(multiStage::StageExecContext& ctx) override;
    void finish(multiStage::StageExecContext& ctx, multiStage::FinishReason reason) override;
    void reset() override;
    nlohmann::json diagnostics(multiStage::RequestSlot const& slot) const override;

private:
    std::string mArtifactDir;
    cudaStream_t mStream{nullptr};
    std::unique_ptr<MiniMaxH3Engine> mEngine;
    int32_t mAudioChannels{0};
    int32_t mMaxFrames{0};
    int32_t mHopLength{0};
    int32_t mSampleRate{0};
    int64_t mSamples{0};

    rt::Tensor mLatents;      //!< [2, audioChannels, frames] FP32
    rt::Tensor mWaveform;     //!< [2, samples] FP32
    rt::Tensor mWaveformHost; //!< Pinned copy of mWaveform.
};

/*!
 * @brief Video frames from the denoised ``latents``, one clip per call, with the tiled VAE of
 * ``vae_decoder.engine``.
 *
 * Emits each clip on the ``video`` terminal as packed 8-bit RGB frames.
 */
class VideoDecoderStage final : public multiStage::IStage
{
public:
    explicit VideoDecoderStage(multiStage::StageInitContext const& init);

    void preallocate() override;
    bool admit(multiStage::StageExecContext const& ctx, std::string& reason) override;
    void prepare(multiStage::StageExecContext& ctx) override;
    multiStage::StageStatus step(multiStage::StageExecContext& ctx) override;
    void finish(multiStage::StageExecContext& ctx, multiStage::FinishReason reason) override;
    void reset() override;
    nlohmann::json diagnostics(multiStage::RequestSlot const& slot) const override;

private:
    //! The canvas clip @p clip is stitched into; consecutive clips alternate between two.
    half* canvasFor(int32_t clip) noexcept;
    //! Decode clip mClip into its canvas.
    void decodeClip(float const* latents, cudaStream_t stream);
    //! Convert @p count canvas frames at @p source to RGB, the first @p blendFrames faded in from @p blendSource, and
    //! hand them to the caller.
    void emitFrames(multiStage::StageExecContext& ctx, half const* source, int32_t count, half const* blendSource,
        int32_t blendFrames);

    std::string mArtifactDir;
    cudaStream_t mStream{nullptr};
    std::unique_ptr<MiniMaxH3Engine> mEngine;
    VideoLayout mLayout;
    int32_t mLatentChannels{0};
    int32_t mMaxTileColumns{0};

    // The request's latent grid and progress.
    int32_t mLatentFrames{0};
    int32_t mLatentHeight{0};
    int32_t mLatentWidth{0};
    int32_t mNumClips{0};
    int32_t mClip{0};
    int32_t mFramesEmitted{0};

    rt::Tensor mVaeTile;      //!< [1, latentChannels, tileFrames, tile, tile] FP32
    rt::Tensor mDecodedTiles; //!< [2, maxTileColumns, 3, tileFrames * 4, tile * 16, tile * 16] FP16 row ring.
    rt::Tensor mClipCanvases; //!< [2, 3, clip + tail frames, pixels] FP16 stitched clips, alternating.
    rt::Tensor mFrames;       //!< [clip frames, pixels, 3] RGB8 of one clip.
    rt::Tensor mFramesHost;   //!< Pinned copy of mFrames.
};

std::unique_ptr<multiStage::IStage> createTextEncoderStage(multiStage::StageInitContext const& init);
std::unique_ptr<multiStage::IStage> createDenoiserStage(multiStage::StageInitContext const& init);
std::unique_ptr<multiStage::IStage> createAudioDecoderStage(multiStage::StageInitContext const& init);
std::unique_ptr<multiStage::IStage> createVideoDecoderStage(multiStage::StageInitContext const& init);

multiStage::RunnerDecl textEncoderDeclaration();
multiStage::RunnerDecl denoiserDeclaration();
multiStage::RunnerDecl audioDecoderDeclaration();
multiStage::RunnerDecl videoDecoderDeclaration();

} // namespace minimax_h3
} // namespace rt
} // namespace trt_edgellm
