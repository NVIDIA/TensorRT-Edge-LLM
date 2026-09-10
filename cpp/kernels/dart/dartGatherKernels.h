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

//! \brief Dtype-agnostic row gather: dst[r, :] = src[rowIndices[r], :].
//!
//! Rows are treated as opaque byte spans of length rowBytes. dst and src must not overlap
//! (gather out-of-place into a scratch buffer, then copy back if compaction in place is
//! desired). Uses 16-byte vectorized copies when rowBytes and both pointers are 16-byte
//! aligned, byte copies otherwise.
//!
//! \param[out] dst Destination base pointer (device), at least numRows * rowBytes bytes
//! \param[in] src Source base pointer (device)
//! \param[in] rowIndices Device INT32 array of source row indices, length numRows
//! \param[in] numRows Number of rows to gather
//! \param[in] rowBytes Size of one row in bytes
//! \param[in] stream CUDA stream for execution
void gatherRows(
    void* dst, void const* src, int32_t const* rowIndices, int64_t numRows, int64_t rowBytes, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
