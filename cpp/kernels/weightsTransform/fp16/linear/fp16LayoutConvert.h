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

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! These are pointer-level launch primitives by design. Checkpoint sources are
//! const CUDA aliases of file-backed host pages and therefore cannot be modeled
//! faithfully by rt::Tensor, which has one mutable address and one device type.
//! The runtime validates source metadata and Tensor destinations before calling
//! these functions.

//! Copy CUDA-addressable bytes directly into final device storage.
cudaError_t launchCopyBytes(void const* source, void* destination, size_t bytes, cudaStream_t stream);

//! Copy one byte range from each source row into compact destination rows.
cudaError_t launchCopy2DBytes(void const* source, void* destination, int64_t rows, size_t sourceRowBytes,
    size_t sourceColumnOffsetBytes, size_t destinationRowBytes, cudaStream_t stream);

//! Gather and cast a contiguous column slice from BF16/FP32 source rows.
cudaError_t launchBf16SliceToFp16(void const* source, void* destination, int64_t rows, int64_t sourceColumns,
    int64_t sourceColumnOffset, int64_t destinationColumns, cudaStream_t stream);
cudaError_t launchFp32SliceToFp16(void const* source, void* destination, int64_t rows, int64_t sourceColumns,
    int64_t sourceColumnOffset, int64_t destinationColumns, cudaStream_t stream);

//! Cast device BF16 ``[N]`` → FP16 ``[N]`` (elementwise).
cudaError_t launchBf16ToFp16(void const* dBf16, void* dFp16, int64_t n, cudaStream_t stream);

//! Cast device FP32 ``[N]`` → FP16 ``[N]`` (elementwise).
cudaError_t launchFp32ToFp16(void const* dFp32, void* dFp16, int64_t n, cudaStream_t stream);

//! Cast device BF16 ``[N]`` → FP32 ``[N]`` (elementwise).
cudaError_t launchBf16ToFp32(void const* dBf16, void* dFp32, int64_t n, cudaStream_t stream);

//! Cast device FP16 ``[N]`` → FP32 ``[N]`` (elementwise).
cudaError_t launchFp16ToFp32(void const* dFp16, void* dFp32, int64_t n, cudaStream_t stream);

//! Transpose FP16 ``[rows, cols]`` → ``[cols, rows]``.
cudaError_t launchTransposeFp16(
    void const* dSrcRowsCols, void* dDstColsRows, int32_t rows, int32_t cols, cudaStream_t stream);

//! Fused BF16 ``[rows, cols]`` → FP16 ``[cols, rows]`` (cast + transpose).
cudaError_t launchBf16TransposeToFp16(
    void const* dSrcBf16RowsCols, void* dDstFp16ColsRows, int32_t rows, int32_t cols, cudaStream_t stream);

//! Fused FP32 ``[rows, cols]`` → FP16 ``[cols, rows]``.
cudaError_t launchFp32TransposeToFp16(
    void const* dSrcFp32RowsCols, void* dDstFp16ColsRows, int32_t rows, int32_t cols, cudaStream_t stream);

//! Scale FP16 tensor in-place by a scalar.
cudaError_t launchScaleFp16(void* dFp16, int64_t n, float scale, cudaStream_t stream);

//! Fill an FP32 tensor with one scalar value.
cudaError_t launchFillFp32(void* dFp32, int64_t n, float value, cudaStream_t stream);

//! Write up to 256 FP32 values passed as kernel arguments.
cudaError_t launchWriteFp32(float const* values, int32_t count, void* dFp32, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
