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
namespace kernels
{

//! Contract of the CuTe DSL dual GEMM kernels (kernelSrcs/gemm_cutedsl/dual_gemm_blackwell.py):
//! H[M, N] = silu(A[M, K] @ Wg[N, K]^T) * (A @ Wu[N, K]^T), FP16 in/out, FP32 accumulation.
namespace dual_gemm_blackwell
{
//! The gate/up weights are consumed as ONE packed FP16 tensor of shape [2N, K]: for every 128-wide output tile
//! and every 64-wide K tile, the 128 rows of Wg then the 128 rows of Wu, each tile row-major
//! (`pack_gate_up_weights` in the kernel script). Every TMA box is one contiguous 16 KB or 32 KB block.
constexpr int32_t kWeightTileN = 128;
constexpr int32_t kWeightTileK = 64;

//! Runtime M ranges of the three AOT variants (tuned on Thor, SM110).
constexpr int32_t kDecodeMaxM = 64;    //!< 64x128 tile, two accumulators (weight-stream bound)
constexpr int32_t kPrefillMaxM = 2048; //!< gate|up merged along N: 256x256 2-CTA MMA, 2x1 cluster
                                       //!< above: 128x128 tile with the grouped raster (A no longer fits L2)

constexpr bool isSupportedProblemShape(int32_t n, int32_t k) noexcept
{
    return n > 0 && k > 0 && n % kWeightTileN == 0 && k % kWeightTileK == 0;
}
} // namespace dual_gemm_blackwell

struct DualGemmBlackwellParams
{
    void const* activation;   //!< A [M, K] FP16, K contiguous
    void const* packedWeight; //!< packed gate/up weight [2N, K] FP16 (dual_gemm_blackwell layout)
    void* output;             //!< H [M, N] FP16, N contiguous
    int32_t numTokens;        //!< M
    int32_t interSize;        //!< N (intermediate size)
    int32_t hiddenSize;       //!< K (hidden size)
};

class CuteDslDualGemmRunner
{
public:
    //! True when the dual GEMM variants were compiled into this binary and the GPU is a supported Blackwell.
    static bool canImplement(int32_t smVersion) noexcept;

    //! Load the AOT module that serves `numTokens` rows (idempotent, thread-safe). Must run outside CUDA graph
    //! capture; call it from configurePlugin / onShapeChange so enqueue only launches.
    static cudaError_t prepare(int32_t numTokens, cudaStream_t stream) noexcept;

    //! Launch the variant selected by `params.numTokens`.
    static cudaError_t run(DualGemmBlackwellParams const& params, cudaStream_t stream) noexcept;
};

} // namespace kernels
} // namespace trt_edgellm
