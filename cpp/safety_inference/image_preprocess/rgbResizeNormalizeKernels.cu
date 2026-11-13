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

#include "rgbResizeNormalizeKernels.cuh"

namespace trt_edgellm::safety
{
namespace
{
__device__ float bilinearRgbChannel(uint8_t const* source, int32_t width, int32_t height, int32_t rowStride,
    float sourceX, float sourceY, int32_t channel) noexcept
{
    float const x = fminf(fmaxf(sourceX, 0.0F), static_cast<float>(width - 1));
    float const y = fminf(fmaxf(sourceY, 0.0F), static_cast<float>(height - 1));
    int32_t const x0 = static_cast<int32_t>(floorf(x));
    int32_t const y0 = static_cast<int32_t>(floorf(y));
    int32_t const x1 = min(x0 + 1, width - 1);
    int32_t const y1 = min(y0 + 1, height - 1);
    float const wx = x - static_cast<float>(x0);
    float const wy = y - static_cast<float>(y0);
    float const top = source[y0 * rowStride + x0 * kRgbChannels + channel] * (1.0F - wx)
        + source[y0 * rowStride + x1 * kRgbChannels + channel] * wx;
    float const bottom = source[y1 * rowStride + x0 * kRgbChannels + channel] * (1.0F - wx)
        + source[y1 * rowStride + x1 * kRgbChannels + channel] * wx;
    return top * (1.0F - wy) + bottom * wy;
}

__global__ void resizeNormalizeRgb8(uint8_t const* source, int32_t width, int32_t height, int32_t rowStride,
    int32_t outputWidth, int32_t outputHeight, RgbNormalization normalization, half* output)
{
    int32_t const outputX = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    int32_t const outputY = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
    if (outputX >= outputWidth || outputY >= outputHeight)
    {
        return;
    }
    float const sourceX = (static_cast<float>(outputX) + normalization.coordinateOffset) * width / outputWidth
        - normalization.coordinateOffset;
    float const sourceY = (static_cast<float>(outputY) + normalization.coordinateOffset) * height / outputHeight
        - normalization.coordinateOffset;
    auto const pixel = (static_cast<std::size_t>(outputY) * outputWidth + outputX) * kRgbChannels;
    for (int32_t channel = 0; channel < kRgbChannels; ++channel)
    {
        float const value = bilinearRgbChannel(source, width, height, rowStride, sourceX, sourceY, channel);
        output[pixel + channel] = __float2half(
            (value * normalization.scale - normalization.mean[channel]) / normalization.deviation[channel]);
    }
}

bool validNormalization(RgbNormalization const& normalization) noexcept
{
    return normalization.deviation[0] > 0.0F && normalization.deviation[1] > 0.0F && normalization.deviation[2] > 0.0F;
}
} // namespace

cudaError_t launchResizeNormalizeRgb8(uint8_t const* source, int32_t width, int32_t height, int32_t rowStride,
    int32_t outputWidth, int32_t outputHeight, RgbNormalization normalization, half* output,
    cudaStream_t stream) noexcept
{
    if (source == nullptr || output == nullptr || stream == nullptr || width <= 0 || height <= 0 || outputWidth <= 0
        || outputHeight <= 0 || static_cast<int64_t>(rowStride) < static_cast<int64_t>(width) * kRgbChannels
        || !validNormalization(normalization))
    {
        return cudaErrorInvalidValue;
    }
    dim3 const threads{kRgbResizeBlockWidth, kRgbResizeBlockHeight, 1};
    dim3 const grid{(static_cast<unsigned>(outputWidth) + threads.x - 1) / threads.x,
        (static_cast<unsigned>(outputHeight) + threads.y - 1) / threads.y, 1};
    resizeNormalizeRgb8<<<grid, threads, 0, stream>>>(
        source, width, height, rowStride, outputWidth, outputHeight, normalization, output);
    return cudaPeekAtLastError();
}

} // namespace trt_edgellm::safety
