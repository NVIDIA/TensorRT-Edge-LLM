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

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{
namespace minimax_h3
{

//! \brief Fill \p count FP32 values with standard normal samples of Philox stream \p streamId.
void randomNormal(float* values, int64_t count, uint64_t seed, uint64_t streamId, cudaStream_t stream);

//! \brief FP32 latents of one modality and their predicted velocity.
struct FlowLatents
{
    float* latents;
    float const* velocity;
    int64_t count;
    float sigma; //!< Sigma of this step.
    float ratio; //!< Sigma of the next step over sigma of this one.
};

//! \brief Rectified-flow Euler step ``x = r * x + (1 - r) * (x + sigma * v)`` of the video and audio latents in one
//! launch.
void eulerStep(FlowLatents video, FlowLatents audio, cudaStream_t stream);

//! \brief The packed ``[text | audio | video]`` sequence of one request, for its RoPE positions.
struct SequenceLayout
{
    int32_t textTokens;
    int32_t audioFrames;  //!< Stereo rows are 2 * audioFrames.
    int32_t latentFrames; //!< Video latent grid.
    int32_t latentHeight;
    int32_t latentWidth;
};

//! \brief Write the ``[sequence, 3]`` FP32 ``(t, h, w)`` RoPE positions of every row.
//!
//! Text token i is at ``(i, 0, 0)``. Audio frame f of either channel is at time ``text + f`` on the left or right
//! edge of the video grid. Video latent frames start at time ``text`` and span 5/3 of 1, 4, 4, 4, 4 frames in turn,
//! which rescales 24 fps video onto the 40 Hz audio-latent axis; their rows are centred on a square of side 32.
void ropePositions(float* positions, SequenceLayout layout, cudaStream_t stream);

//! \brief Transpose ``[2 * frames, channels]`` stereo rows to ``[2, channels, frames]`` latents.
void audioRowsToLatents(float const* rows, float* latents, int32_t frames, int32_t channels, cudaStream_t stream);

//! \brief Video latent grid of patchified ``[frames, height / 2, width / 2, channels * 4]`` rows.
struct VideoLatentGrid
{
    int32_t channels;
    int32_t frames;
    int32_t height;
    int32_t width;
};

//! \brief Copy the ``[channels, tileFrames, tileSize, tileSize]`` latent tile at
//! ``(frame0, y0, x0)`` out of the patchified video rows.
void gatherVaeTile(float const* rows, float* tile, VideoLatentGrid grid, int32_t frame0, int32_t y0, int32_t x0,
    int32_t tileFrames, int32_t tileSize, cudaStream_t stream);

//! \brief Where one decoded ``[3, tileFrames, tile, tile]`` VAE tile lands in the clip canvas.
//!
//! The canvas is ``[3, keptFrames, height, width]``: the clip's first ``headFrames`` frames start
//! at tile frame ``headFirst`` and the rest at ``tailFirst``. A tile is linearly blended with the
//! original tiles above and to its left over their overlaps, then cropped, as in Diffusers'
//! tiled decode.
struct TilePlacement
{
    int32_t tileFrames;
    int32_t tileSize;
    int32_t headFirst;
    int32_t headFrames;
    int32_t tailFirst;
    int32_t keptFrames;
    int32_t canvasHeight;
    int32_t canvasWidth;
    int32_t y0; //!< Tile origin in the canvas.
    int32_t x0;
    int32_t height; //!< Rows kept after cropping the overlap with the next tile.
    int32_t width;
    int32_t overlapAbove; //!< Rows blended with the tile above; 0 for the first row.
    int32_t overlapLeft;  //!< Columns blended with the tile to the left; 0 for the first column.
};

void stitchTile(
    half const* tile, half const* above, half const* left, half* canvas, TilePlacement placement, cudaStream_t stream);

//! \brief Write the first ``count`` frames of a ``[3, srcFrames, height, width]`` RGB canvas as
//! packed 8-bit ``[count, height, width, 3]`` RGB.
//!
//! The first \p blendFrames frames are cross-faded from the ``[3, blendSourceFrames, height,
//! width]`` frames of \p blendSource, as Diffusers joins consecutive decoded clips.
void emitRgbFrames(half const* source, int32_t srcFrames, int32_t count, half const* blendSource,
    int32_t blendSourceFrames, int32_t blendFrames, int32_t height, int32_t width, uint8_t* rgb, cudaStream_t stream);

} // namespace minimax_h3
} // namespace kernel
} // namespace trt_edgellm
