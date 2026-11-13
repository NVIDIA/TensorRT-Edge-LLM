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

#include "kernels/minimaxH3Kernels/minimaxH3Kernels.h"

#include "common/checkMacros.h"

#include <curand_kernel.h>

namespace trt_edgellm
{
namespace kernel
{
namespace minimax_h3
{

namespace
{

constexpr int32_t kTHREADS{256};
//! Decorrelates the Philox key of different noise streams sharing one seed.
constexpr uint64_t kSTREAM_KEY_STRIDE{0x9E3779B97F4A7C15ULL};

int32_t blocksFor(int64_t count)
{
    return static_cast<int32_t>((count + kTHREADS - 1) / kTHREADS);
}

__global__ void randomNormalKernel(float* values, int64_t count, uint64_t key)
{
    int64_t const group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const first = group * 4;
    if (first >= count)
    {
        return;
    }
    curandStatePhilox4_32_10_t state;
    curand_init(key, group, 0, &state);
    float4 const samples = curand_normal4(&state);
    float const sampleArray[4]{samples.x, samples.y, samples.z, samples.w};
    for (int32_t index = 0; index < 4 && first + index < count; ++index)
    {
        values[first + index] = sampleArray[index];
    }
}

__global__ void eulerStepKernel(FlowLatents video, FlowLatents audio)
{
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    FlowLatents const& flow = index < video.count ? video : audio;
    index -= index < video.count ? 0 : video.count;
    if (index < flow.count)
    {
        float const sample = flow.latents[index];
        float const denoised = sample + flow.sigma * flow.velocity[index];
        flow.latents[index] = flow.ratio * sample + (1.0F - flow.ratio) * denoised;
    }
}

//! ``np.linspace(left, left + ratio, size / 2, endpoint=False) * 32`` with ``ratio = size / sqrt(area)``.
__device__ double spatialPosition(int32_t size, double sqrtArea, int32_t index)
{
    double const ratio = size / sqrtArea;
    double const left = (1.0 - ratio) / 2.0;
    return (left + index * (ratio / (size / 2))) * 32.0;
}

__global__ void ropePositionsKernel(float* positions, SequenceLayout layout)
{
    int64_t const audioRows = 2LL * layout.audioFrames;
    int64_t const frameRows = static_cast<int64_t>(layout.latentHeight / 2) * (layout.latentWidth / 2);
    int64_t const row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= layout.textTokens + audioRows + layout.latentFrames * frameRows)
    {
        return;
    }
    double const sqrtArea = sqrt(static_cast<double>(layout.latentHeight) * layout.latentWidth);
    double t{0.0};
    double h{0.0};
    double w{0.0};
    if (row < layout.textTokens)
    {
        t = static_cast<double>(row);
    }
    else if (row < layout.textTokens + audioRows)
    {
        int64_t const audio = row - layout.textTokens;
        int32_t const edge = audio < layout.audioFrames ? 0 : layout.latentWidth / 2 - 1;
        t = static_cast<double>(layout.textTokens + audio % layout.audioFrames);
        w = spatialPosition(layout.latentWidth, sqrtArea, edge);
    }
    else
    {
        int64_t const video = row - layout.textTokens - audioRows;
        auto const frame = static_cast<int32_t>(video / frameRows);
        auto const cell = static_cast<int32_t>(video % frameRows);
        // Five latent frames span 1 + 4 + 4 + 4 + 4 = 17 video frames.
        int32_t const phase = frame % 5;
        int32_t const span = 17 * (frame / 5) + (phase == 0 ? 0 : 1 + 4 * (phase - 1));
        t = layout.textTokens + 5.0 / 3.0 * span;
        h = spatialPosition(layout.latentHeight, sqrtArea, cell / (layout.latentWidth / 2));
        w = spatialPosition(layout.latentWidth, sqrtArea, cell % (layout.latentWidth / 2));
    }
    positions[row * 3] = static_cast<float>(t);
    positions[row * 3 + 1] = static_cast<float>(h);
    positions[row * 3 + 2] = static_cast<float>(w);
}

__global__ void audioRowsToLatentsKernel(float const* rows, float* latents, int32_t frames, int32_t channels)
{
    int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const total = 2LL * channels * frames;
    if (index >= total)
    {
        return;
    }
    int32_t const frame = index % frames;
    int32_t const channel = (index / frames) % channels;
    int32_t const stereo = index / (static_cast<int64_t>(frames) * channels);
    latents[index] = rows[(static_cast<int64_t>(stereo) * frames + frame) * channels + channel];
}

__global__ void gatherVaeTileKernel(float const* rows, float* tile, VideoLatentGrid grid, int32_t frame0, int32_t y0,
    int32_t x0, int32_t tileFrames, int32_t tileSize)
{
    int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const total = static_cast<int64_t>(grid.channels) * tileFrames * tileSize * tileSize;
    if (index >= total)
    {
        return;
    }
    int32_t const x = x0 + index % tileSize;
    int32_t const y = y0 + (index / tileSize) % tileSize;
    int32_t const frame = frame0 + (index / (tileSize * tileSize)) % tileFrames;
    int32_t const channel = index / (static_cast<int64_t>(tileSize) * tileSize * tileFrames);
    // Rows are [frame, y / 2, x / 2] with the 2x2 patch innermost in each channel.
    int64_t const row = (static_cast<int64_t>(frame) * (grid.height / 2) + y / 2) * (grid.width / 2) + x / 2;
    tile[index] = rows[row * grid.channels * 4 + channel * 4 + (y % 2) * 2 + x % 2];
}

__global__ void stitchTileKernel(
    half const* tile, half const* above, half const* left, half* canvas, TilePlacement placement)
{
    TilePlacement const& p = placement;
    int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const total = 3LL * p.keptFrames * p.height * p.width;
    if (index >= total)
    {
        return;
    }
    int32_t const col = index % p.width;
    int32_t const row = (index / p.width) % p.height;
    int32_t const kept = (index / (static_cast<int64_t>(p.width) * p.height)) % p.keptFrames;
    int32_t const channel = index / (static_cast<int64_t>(p.width) * p.height * p.keptFrames);
    int32_t const frame = kept < p.headFrames ? p.headFirst + kept : p.tailFirst + kept - p.headFrames;
    auto const at = [&](half const* source, int32_t y, int32_t x) {
        return __half2float(
            source[((static_cast<int64_t>(channel) * p.tileFrames + frame) * p.tileSize + y) * p.tileSize + x]);
    };
    float value = at(tile, row, col);
    if (row < p.overlapAbove)
    {
        float const weight = static_cast<float>(row) / p.overlapAbove;
        value = at(above, p.tileSize - p.overlapAbove + row, col) * (1.0F - weight) + value * weight;
    }
    if (col < p.overlapLeft)
    {
        float const weight = static_cast<float>(col) / p.overlapLeft;
        value = at(left, row, p.tileSize - p.overlapLeft + col) * (1.0F - weight) + value * weight;
    }
    canvas[((static_cast<int64_t>(channel) * p.keptFrames + kept) * p.canvasHeight + p.y0 + row) * p.canvasWidth + p.x0
        + col] = __float2half(value);
}

__global__ void emitRgbFramesKernel(half const* source, int32_t srcFrames, int32_t count, half const* blendSource,
    int32_t blendSourceFrames, int32_t blendFrames, int32_t height, int32_t width, uint8_t* rgb)
{
    int64_t const plane = static_cast<int64_t>(height) * width;
    int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= plane * count)
    {
        return;
    }
    int64_t const pixel = index % plane;
    int32_t const frame = index / plane;
    for (int32_t channel = 0; channel < 3; ++channel)
    {
        float value = __half2float(source[(static_cast<int64_t>(channel) * srcFrames + frame) * plane + pixel]);
        if (frame < blendFrames)
        {
            float const weight = static_cast<float>(frame) / blendFrames;
            value
                = __half2float(blendSource[(static_cast<int64_t>(channel) * blendSourceFrames + frame) * plane + pixel])
                    * (1.0F - weight)
                + value * weight;
        }
        rgb[index * 3 + channel] = static_cast<uint8_t>(rintf(fminf(fmaxf(value, 0.0F), 1.0F) * 255.0F));
    }
}

} // namespace

void randomNormal(float* values, int64_t count, uint64_t seed, uint64_t streamId, cudaStream_t stream)
{
    randomNormalKernel<<<blocksFor((count + 3) / 4), kTHREADS, 0, stream>>>(
        values, count, seed ^ (streamId * kSTREAM_KEY_STRIDE));
    CUDA_CHECK(cudaGetLastError());
}

void eulerStep(FlowLatents video, FlowLatents audio, cudaStream_t stream)
{
    eulerStepKernel<<<blocksFor(video.count + audio.count), kTHREADS, 0, stream>>>(video, audio);
    CUDA_CHECK(cudaGetLastError());
}

void ropePositions(float* positions, SequenceLayout layout, cudaStream_t stream)
{
    int64_t const rows = layout.textTokens + 2LL * layout.audioFrames
        + static_cast<int64_t>(layout.latentFrames) * (layout.latentHeight / 2) * (layout.latentWidth / 2);
    ropePositionsKernel<<<blocksFor(rows), kTHREADS, 0, stream>>>(positions, layout);
    CUDA_CHECK(cudaGetLastError());
}

void audioRowsToLatents(float const* rows, float* latents, int32_t frames, int32_t channels, cudaStream_t stream)
{
    audioRowsToLatentsKernel<<<blocksFor(2LL * channels * frames), kTHREADS, 0, stream>>>(
        rows, latents, frames, channels);
    CUDA_CHECK(cudaGetLastError());
}

void gatherVaeTile(float const* rows, float* tile, VideoLatentGrid grid, int32_t frame0, int32_t y0, int32_t x0,
    int32_t tileFrames, int32_t tileSize, cudaStream_t stream)
{
    int64_t const total = static_cast<int64_t>(grid.channels) * tileFrames * tileSize * tileSize;
    gatherVaeTileKernel<<<blocksFor(total), kTHREADS, 0, stream>>>(
        rows, tile, grid, frame0, y0, x0, tileFrames, tileSize);
    CUDA_CHECK(cudaGetLastError());
}

void stitchTile(
    half const* tile, half const* above, half const* left, half* canvas, TilePlacement placement, cudaStream_t stream)
{
    int64_t const total = 3LL * placement.keptFrames * placement.height * placement.width;
    stitchTileKernel<<<blocksFor(total), kTHREADS, 0, stream>>>(tile, above, left, canvas, placement);
    CUDA_CHECK(cudaGetLastError());
}

void emitRgbFrames(half const* source, int32_t srcFrames, int32_t count, half const* blendSource,
    int32_t blendSourceFrames, int32_t blendFrames, int32_t height, int32_t width, uint8_t* rgb, cudaStream_t stream)
{
    int64_t const total = static_cast<int64_t>(height) * width * count;
    emitRgbFramesKernel<<<blocksFor(total), kTHREADS, 0, stream>>>(
        source, srcFrames, count, blendSource, blendSourceFrames, blendFrames, height, width, rgb);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace minimax_h3
} // namespace kernel
} // namespace trt_edgellm
