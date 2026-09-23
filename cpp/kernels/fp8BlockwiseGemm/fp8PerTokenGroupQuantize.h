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
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace trt_edgellm::kernels
{

constexpr int32_t kFp8BlockwiseGroupSize{128};

//! Quantize each contiguous 128-element activation group to E4M3 and emit its FP32 dequantization scale.
cudaError_t launchFp8PerTokenGroupQuantize(half const* input, __nv_fp8_e4m3* output, float* scales, int32_t numTokens,
    int32_t hiddenSize, cudaStream_t stream) noexcept;

} // namespace trt_edgellm::kernels
