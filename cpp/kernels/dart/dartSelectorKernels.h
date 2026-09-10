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

#include "common/tensor.h"
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Maximum number of pivot rows a single computeDartPivotDots launch supports. Sized so one
//! batched launch covers the per-image pivots of many spans at once (e.g. 15 images at
//! 4 image pivots each + 4 shared text pivots): multi-image requests then keep a fixed
//! number of stream synchronizations instead of one per image.
constexpr int32_t kDartMaxPivots = 64;

//! \brief Per-row L1 and L2 norms over the hidden dimension.
//!
//! Computes, for every row of a [numRows, hiddenSize] FP16 matrix, the L1 norm (used for
//! DART pivot scoring) and the L2 norm (used as the cosine-similarity denominator).
//!
//! \param[in] embeds Row-major FP16 matrix [numRows, hiddenSize] (a [1, S, H] tensor viewed as [S, H])
//! \param[out] norms FP32 output [numRows, 2]; column 0 = L1, column 1 = L2
//! \param[in] stream CUDA stream for execution
//! \throws std::runtime_error on shape/dtype mismatch
void computeDartRowNorms(rt::Tensor const& embeds, rt::Tensor& norms, cudaStream_t stream);

//! \brief Dot products of a small set of pivot rows against all rows.
//!
//! dots[p][r] = sum_h embeds[pivotIndices[p]][h] * embeds[r][h], accumulated in FP32.
//! Cosine similarity is then dots[p][r] / (L2[pivot_p] * L2[r]) on the host.
//!
//! \param[in] embeds Row-major FP16 matrix [numRows, hiddenSize]
//! \param[in] pivotIndices Device INT32 array of row indices, length numPivots (<= kDartMaxPivots)
//! \param[in] numPivots Number of pivot rows
//! \param[out] dots FP32 output [numPivots, numRows]
//! \param[in] stream CUDA stream for execution
//! \throws std::runtime_error on shape/dtype mismatch or numPivots out of range
void computeDartPivotDots(
    rt::Tensor const& embeds, int32_t const* pivotIndices, int32_t numPivots, rt::Tensor& dots, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
