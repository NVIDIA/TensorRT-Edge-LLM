/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/cudaMacros.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernels
{

/// Compute the required output buffer size (bytes) for the SfAtom tiled layout.
inline int64_t getSfAtomTiledBufferSize(int32_t numRows, int32_t numKBlocks)
{
    constexpr int kAtomRows = 128;
    constexpr int kAtomKBlocks = 4;
    constexpr int kAtomBytes = kAtomRows * kAtomKBlocks; // 512
    int32_t numAtomRows = (numRows + kAtomRows - 1) / kAtomRows;
    int32_t numAtomKBlocks = (numKBlocks + kAtomKBlocks - 1) / kAtomKBlocks;
    return static_cast<int64_t>(numAtomRows) * numAtomKBlocks * kAtomBytes;
}

/// Fused FP32 -> UE4M3 conversion + SfAtom tiled repack in a single kernel.
///
/// Reads FP32 scale factors from contiguous [numRows, numKBlocks] layout,
/// converts each to UE4M3, and writes directly to the SfAtom tiled offset.
///
/// Output buffer does not need to be pre-zeroed; padding bytes are written by the kernel.
///
/// @param fp32Scales   Input FP32 scales [numRows, numKBlocks] row-major
/// @param tiledOut     Output SfAtom tiled UE4M3 buffer
/// @param numRows      Number of rows (M for SFA, N for SFB)
/// @param numKBlocks   Number of K-dimension scale blocks
/// @param stream       CUDA stream
void fusedFp32ToSfAtom(
    float const* fp32Scales, uint8_t* tiledOut, int32_t numRows, int32_t numKBlocks, cudaStream_t stream);

/// Repack checkpoint block scales into the SfAtom tiled layout.
///
/// Output buffer does not need to be pre-zeroed; padding bytes are written by the kernel.
///
/// @param fp8Scales    Input FP8E4M3FN scales [numRows, numKBlocks] row-major
/// @param tiledOut     Output SfAtom tiled UE4M3 buffer
/// @param numRows      Number of rows
/// @param numKBlocks   Number of K-dimension scale blocks
/// @param stream       CUDA stream
#if SUPPORTS_FP8
void repackE4m3ScalesToSfAtom(
    __nv_fp8_e4m3 const* fp8Scales, uint8_t* tiledOut, int32_t numRows, int32_t numKBlocks, cudaStream_t stream);
#endif // SUPPORTS_FP8

} // namespace kernels
} // namespace trt_edgellm
