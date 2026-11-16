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
#include "runtime/streaming.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace minimax_h3
{

namespace
{

constexpr char const* kRunnerName = "minimax_h3.video_decoder";
constexpr char const* kLatentsInput = "latents";
constexpr char const* kVideoPort = "video";
std::string const kTimerStage = "minimax_h3_video_decode";
constexpr int32_t kRGB_CHANNELS{3};

bool listenerGone(multiStage::StageExecContext const& ctx)
{
    return ctx.videoSink != nullptr && ctx.videoSink->isCancelled();
}

//! Tiles along one axis of Diffusers' tiled decode: the fewest tiles whose overlaps reach the
//! minimum, with the slack added round-robin to the overlaps in alignment steps.
class TileLayout
{
public:
    TileLayout(int32_t length, int32_t tile, int32_t minOverlap, int32_t alignment)
        : mTile(tile)
        , mMinOverlap(minOverlap)
        , mAlignment(alignment)
    {
        mCount = (length + tile - 1) / tile;
        while (tile * mCount - minOverlap * (mCount - 1) < length)
        {
            ++mCount;
        }
        mSlackSteps = (tile * mCount - minOverlap * (mCount - 1) - length) / alignment;
    }

    int32_t count() const noexcept
    {
        return mCount;
    }

    //! Overlap between tile \p index and the next one.
    int32_t overlap(int32_t index) const noexcept
    {
        int32_t const gaps = mCount - 1;
        return mMinOverlap + mAlignment * (mSlackSteps / gaps + (index < mSlackSteps % gaps ? 1 : 0));
    }

    int32_t start(int32_t index) const noexcept
    {
        int32_t origin{0};
        for (int32_t previous = 0; previous < index; ++previous)
        {
            origin += mTile - overlap(previous);
        }
        return origin;
    }

private:
    int32_t mTile;
    int32_t mMinOverlap;
    int32_t mAlignment;
    int32_t mCount{1};
    int32_t mSlackSteps{0};
};

} // namespace

VideoLayout VideoLayout::fromConfig(nlohmann::json const& config)
{
    VideoLayout layout;
    layout.spatialRatio = config.at("spatial_ratio");
    layout.temporalRatio = config.at("temporal_ratio");
    layout.tileLatentFrames = config.at("tile_latent_frames");
    layout.tileLatentSize = config.at("tile_latent_size");
    layout.tileMinOverlap = config.at("tile_min_overlap");
    layout.chunkLatentFrames = config.at("chunk_latent_frames");
    layout.chunkFramePadding = config.at("chunk_frame_padding");
    layout.chunkFrameOverlap = config.at("chunk_frame_overlap");
    ELLM_CHECK(layout.spatialRatio > 0 && layout.temporalRatio > 0 && layout.chunkLatentFrames > 0
            && layout.tileLatentFrames >= layout.chunkLatentFrames && layout.clipFrames() > 0,
        "Invalid MiniMax-H3 video VAE layout");
    return layout;
}

VideoDecoderStage::VideoDecoderStage(multiStage::StageInitContext const& init)
    : mArtifactDir(init.artifactDir)
    , mStream(init.stream)
{
}

void VideoDecoderStage::preallocate()
{
    mEngine = std::make_unique<MiniMaxH3Engine>(mArtifactDir, "vae_decoder", mStream);
    auto const& config = mEngine->config();
    mLayout = VideoLayout::fromConfig(config);
    mLatentChannels = config.at("latent_channels");

    int32_t const tilePixels = mLayout.tilePixels();
    int32_t const tileFrames = mLayout.tileLatentFrames * mLayout.temporalRatio;
    int32_t const clipFrames = mLayout.clipFrames();
    // The widest accepted frame keeps the maximum aspect ratio at the maximum area.
    auto const maxWidth = static_cast<int32_t>(std::sqrt(static_cast<double>(kMAX_FRAME_PIXELS) * kMAX_ASPECT_RATIO))
        / kFRAME_ALIGNMENT * kFRAME_ALIGNMENT;
    mMaxTileColumns = TileLayout(maxWidth, tilePixels, mLayout.tileMinOverlap, mLayout.spatialRatio).count();
    mVaeTile
        = rt::Tensor({1, mLatentChannels, mLayout.tileLatentFrames, mLayout.tileLatentSize, mLayout.tileLatentSize},
            rt::DeviceType::kGPU, DataType::kFLOAT, "minimax_h3::vae_tile");
    mDecodedTiles = rt::Tensor({2 * mMaxTileColumns, kRGB_CHANNELS, tileFrames, tilePixels, tilePixels},
        rt::DeviceType::kGPU, DataType::kHALF, "minimax_h3::decoded_tiles");
    mClipCanvases = rt::Tensor({2, kRGB_CHANNELS, clipFrames + mLayout.chunkFrameOverlap, kMAX_FRAME_PIXELS},
        rt::DeviceType::kGPU, DataType::kHALF, "minimax_h3::clip_canvases");
    mFrames = rt::Tensor(
        {clipFrames, kMAX_FRAME_PIXELS, kRGB_CHANNELS}, rt::DeviceType::kGPU, DataType::kUINT8, "minimax_h3::frames");
    mFramesHost = rt::Tensor({clipFrames, kMAX_FRAME_PIXELS, kRGB_CHANNELS}, rt::DeviceType::kCPU, DataType::kUINT8,
        "minimax_h3::frames_host");
    mEngine->bind("latents", mVaeTile);
}

bool VideoDecoderStage::admit(multiStage::StageExecContext const&, std::string&)
{
    // The latents' extent is known only once the denoiser has planned the request.
    return true;
}

void VideoDecoderStage::prepare(multiStage::StageExecContext& ctx)
{
    auto const& latents = ctx.in->at(kLatentsInput);
    ELLM_CHECK(latents.shape.size() == 4 && latents.shape[3] == mLatentChannels * 4,
        "MiniMax-H3 video decoder takes [frames, height / 2, width / 2, " + std::to_string(mLatentChannels * 4)
            + "] latents");
    mLatentFrames = static_cast<int32_t>(latents.shape[0]);
    mLatentHeight = static_cast<int32_t>(latents.shape[1]) * 2;
    mLatentWidth = static_cast<int32_t>(latents.shape[2]) * 2;
    int32_t const tailLatentFrames = mLayout.tileLatentFrames - mLayout.chunkLatentFrames;
    mNumClips = (mLatentFrames - tailLatentFrames) / mLayout.chunkLatentFrames;
    int32_t const height = mLatentHeight * mLayout.spatialRatio;
    int32_t const width = mLatentWidth * mLayout.spatialRatio;
    ELLM_CHECK(mNumClips > 0 && mLayout.latentFrames(mNumClips) == mLatentFrames,
        "MiniMax-H3 video latents must hold whole clips");
    ELLM_CHECK(static_cast<int64_t>(height) * width <= kMAX_FRAME_PIXELS
            && std::min(height, width) >= mLayout.tilePixels()
            && TileLayout(width, mLayout.tilePixels(), mLayout.tileMinOverlap, mLayout.spatialRatio).count()
                <= mMaxTileColumns,
        "MiniMax-H3 frame of " + std::to_string(width) + "x" + std::to_string(height)
            + " does not fit the video decoder");
    mClip = 0;
    mFramesEmitted = 0;
}

half* VideoDecoderStage::canvasFor(int32_t clip) noexcept
{
    return mClipCanvases.dataPointer<half>() + (clip % 2) * (mClipCanvases.getShape().volume() / 2);
}

void VideoDecoderStage::decodeClip(float const* latents, cudaStream_t stream)
{
    int32_t const tilePixels = mLayout.tilePixels();
    int32_t const tileFrames = mLayout.tileLatentFrames * mLayout.temporalRatio;
    int32_t const chunkFrames = mLayout.chunkLatentFrames * mLayout.temporalRatio;
    int32_t const clipFrames = mLayout.clipFrames();
    int32_t const keptFrames = clipFrames + mLayout.chunkFrameOverlap;
    int32_t const height = mLatentHeight * mLayout.spatialRatio;
    int32_t const width = mLatentWidth * mLayout.spatialRatio;
    int64_t const tileElements = static_cast<int64_t>(kRGB_CHANNELS) * tileFrames * tilePixels * tilePixels;
    TileLayout const rows(height, tilePixels, mLayout.tileMinOverlap, mLayout.spatialRatio);
    TileLayout const columns(width, tilePixels, mLayout.tileMinOverlap, mLayout.spatialRatio);
    auto* const tiles = mDecodedTiles.dataPointer<half>();
    kernel::minimax_h3::VideoLatentGrid const grid{mLatentChannels, mLatentFrames, mLatentHeight, mLatentWidth};

    for (int32_t row = 0; row < rows.count(); ++row)
    {
        half* const rowTiles = tiles + (row % 2) * mMaxTileColumns * tileElements;
        half const* const aboveTiles = tiles + ((row + 1) % 2) * mMaxTileColumns * tileElements;
        for (int32_t column = 0; column < columns.count(); ++column)
        {
            kernel::minimax_h3::gatherVaeTile(latents, mVaeTile.dataPointer<float>(), grid,
                mClip * mLayout.chunkLatentFrames, rows.start(row) / mLayout.spatialRatio,
                columns.start(column) / mLayout.spatialRatio, mLayout.tileLatentFrames, mLayout.tileLatentSize, stream);
            mEngine->bind(
                "pixels", rowTiles + column * tileElements, {1, kRGB_CHANNELS, tileFrames, tilePixels, tilePixels});
            mEngine->enqueue(stream);
        }
        for (int32_t column = 0; column < columns.count(); ++column)
        {
            bool const lastRow = row + 1 == rows.count();
            bool const lastColumn = column + 1 == columns.count();
            kernel::minimax_h3::TilePlacement const placement{tileFrames, tilePixels, mLayout.chunkFramePadding,
                clipFrames, chunkFrames + mLayout.chunkFramePadding, keptFrames, height, width, rows.start(row),
                columns.start(column), tilePixels - (lastRow ? 0 : rows.overlap(row)),
                tilePixels - (lastColumn ? 0 : columns.overlap(column)), row == 0 ? 0 : rows.overlap(row - 1),
                column == 0 ? 0 : columns.overlap(column - 1)};
            kernel::minimax_h3::stitchTile(rowTiles + column * tileElements,
                row == 0 ? nullptr : aboveTiles + column * tileElements,
                column == 0 ? nullptr : rowTiles + (column - 1) * tileElements, canvasFor(mClip), placement, stream);
        }
    }
}

void VideoDecoderStage::emitFrames(
    multiStage::StageExecContext& ctx, half const* source, int32_t count, half const* blendSource, int32_t blendFrames)
{
    int32_t const height = mLatentHeight * mLayout.spatialRatio;
    int32_t const width = mLatentWidth * mLayout.spatialRatio;
    int32_t const keptFrames = mLayout.clipFrames() + mLayout.chunkFrameOverlap;
    size_t const bytes = static_cast<size_t>(count) * height * width * kRGB_CHANNELS;
    kernel::minimax_h3::emitRgbFrames(source, keptFrames, count, blendSource, keptFrames, blendFrames, height, width,
        mFrames.dataPointer<uint8_t>(), ctx.stream);
    CUDA_CHECK(
        cudaMemcpyAsync(mFramesHost.rawPointer(), mFrames.rawPointer(), bytes, cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    mFramesEmitted += count;
    if (ctx.videoSink != nullptr)
    {
        VideoChunk chunk;
        chunk.rgb8.assign(mFramesHost.dataPointer<char>(), bytes);
        chunk.numFrames = count;
        chunk.height = height;
        chunk.width = width;
        chunk.fps = kFPS;
        ctx.videoSink->push(std::move(chunk));
    }
}

multiStage::StageStatus VideoDecoderStage::step(multiStage::StageExecContext& ctx)
{
    if (ctx.slot->abandon.requested() || listenerGone(ctx))
    {
        return multiStage::StageStatus::kDone;
    }
    int32_t const clipFrames = mLayout.clipFrames();
    int32_t const overlap = mLayout.chunkFrameOverlap;
    int64_t const pixels
        = static_cast<int64_t>(mLatentHeight) * mLatentWidth * mLayout.spatialRatio * mLayout.spatialRatio;
    // The tail of a clip, the frames after its first clipFrames, stays in its canvas until the next clip, which
    // decodes into the other canvas, has faded in from it.
    auto const tailOf = [&](int32_t clip) { return canvasFor(clip) + clipFrames * pixels; };
    {
        TIME_STAGE(kTimerStage, ctx.stream);
        decodeClip(static_cast<float const*>(ctx.in->at(kLatentsInput).data), ctx.stream);
        emitFrames(
            ctx, canvasFor(mClip), clipFrames, mClip == 0 ? nullptr : tailOf(mClip - 1), mClip == 0 ? 0 : overlap);
        if (++mClip == mNumClips)
        {
            emitFrames(ctx, tailOf(mClip - 1), overlap, nullptr, 0);
        }
    }
    return mClip < mNumClips ? multiStage::StageStatus::kMore : multiStage::StageStatus::kDone;
}

void VideoDecoderStage::finish(multiStage::StageExecContext& ctx, multiStage::FinishReason)
{
    // An empty final chunk and an explicit end of stream on every path, so a listener never waits on a
    // request that stopped.
    if (ctx.videoSink != nullptr)
    {
        VideoChunk chunk;
        chunk.isFinal = true;
        chunk.height = mLatentHeight * mLayout.spatialRatio;
        chunk.width = mLatentWidth * mLayout.spatialRatio;
        chunk.fps = kFPS;
        ctx.videoSink->push(std::move(chunk));
        ctx.videoSink->finish();
    }
}

void VideoDecoderStage::reset()
{
    mClip = 0;
    mNumClips = 0;
    mFramesEmitted = 0;
}

nlohmann::json VideoDecoderStage::diagnostics(multiStage::RequestSlot const&) const
{
    return {{"frames", mFramesEmitted}, {"height", mLatentHeight * mLayout.spatialRatio},
        {"width", mLatentWidth * mLayout.spatialRatio}, {"fps", kFPS}};
}

std::unique_ptr<multiStage::IStage> createVideoDecoderStage(multiStage::StageInitContext const& init)
{
    return std::make_unique<VideoDecoderStage>(init);
}

multiStage::RunnerDecl videoDecoderDeclaration()
{
    multiStage::RunnerDecl decl;
    decl.name = kRunnerName;
    decl.kind = multiStage::StageKind::kIterative;
    decl.ports = {{kVideoPort, multiStage::PortKind::kImage}};
    decl.inputs = {kLatentsInput};
    return decl;
}

} // namespace minimax_h3
} // namespace rt
} // namespace trt_edgellm
