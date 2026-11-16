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

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace trt_edgellm::safety
{

inline constexpr int32_t kRgbChannels{3};
inline constexpr unsigned kRgbResizeBlockWidth{16};
inline constexpr unsigned kRgbResizeBlockHeight{16};

struct RgbNormalization
{
    float scale{1.0F};
    float mean[kRgbChannels]{};
    float deviation[kRgbChannels]{1.0F, 1.0F, 1.0F};
    float coordinateOffset{0.5F};
};

cudaError_t launchResizeNormalizeRgb8(uint8_t const* source, int32_t width, int32_t height, int32_t rowStride,
    int32_t outputWidth, int32_t outputHeight, RgbNormalization normalization, half* output,
    cudaStream_t stream) noexcept;

} // namespace trt_edgellm::safety
