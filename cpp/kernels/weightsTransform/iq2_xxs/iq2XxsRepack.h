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
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Bytes in one IQ2_XXS checkpoint block: an FP16 base followed by 64 payload bytes for 256 weights.
inline constexpr int32_t kIq2XxsBlockBytes = 66;

//! Reorder a dense IQ2_XXS checkpoint matrix uint8 [N, K/256, 66] for W2A16GemmPlugin. ``values`` receives the
//! payload as [N/P, K/256, P, 64] with P = 128 when N % 128 == 0, else 8; ``bases`` receives the FP16 bases as
//! [K/256, N]. Either output may be null. Bytes are only moved.
cudaError_t launchIq2XxsGemmRepack(
    uint8_t const* source, int8_t* values, uint16_t* bases, int32_t N, int32_t kBlocks, cudaStream_t stream);

//! Reorder per-expert IQ2_XXS checkpoint matrices uint8 [N, K/256, 66] for W2A16MoePlugin. ``words`` receives each
//! expert's payload as int32 [N/64, K/128, 8, 4, 8, 2]; ``bases`` receives each expert's FP16 bases as
//! [K/256, N/16, 8, 2]. Either output may be null; both point at the first expert of this batch.
cudaError_t launchIq2XxsMoeRepackSourceBatch(uint8_t const* const* sources, int32_t count, int32_t* words,
    uint16_t* bases, int32_t N, int32_t kBlocks, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
