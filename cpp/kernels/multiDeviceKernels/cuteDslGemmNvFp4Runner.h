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

// C++ runner for the CuTe DSL-compiled Blackwell NVFP4 GEMM kernel.
// Follows the static-module-cache pattern established by
// cpp/kernels/moe/NvFP4MoEContiguousGemmRunner.
//
// Gating
// ------
// Guarded by CUTE_DSL_GEMM_NVFP4_ENABLED (set by cmake/CuteDsl.cmake when
// the gemm_nvfp4 group is in the artifact's metadata.json). When the
// define is absent this compilation unit still produces valid object
// code -- run() returns cudaErrorNotSupported. This lets downstream
// callers compile unconditionally while the runtime reports unsupported
// CuTe DSL GEMM artifacts cleanly.
//
// Layout conventions (match gemm_blackwell_nvfp4.py wrapper):
//   * A   : [M, K] Float4E2M1FN, packed two FP4 values per uint8
//   * B   : [N, K] Float4E2M1FN, packed
//   * SFA : atom-layout Float8E4M3FN, size = _sfBufferBytes(M, K, 16)
//   * SFB : atom-layout Float8E4M3FN, size = _sfBufferBytes(N, K, 16)
//   * D   : [M, N] Float16 row-major   (run())
//   * D   : [M, N] Float8E4M3FN        (runFp8())
//
// Output dtype selection
// ----------------------
// Each (output dtype, mma tiler N) combination is a separate AOT-compiled
// kernel module. Header tags below:
//
//   FP16-out:  gemm_blackwell_nvfp4_fp16_tn{64,128}_Kernel_Module_t
//              + CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP16_TN{64,128}_ENABLED
//   FP8-out:   gemm_blackwell_nvfp4_fp8_tn{64,128}_Kernel_Module_t
//              + CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP8_TN{64,128}_ENABLED
//
// Shape handling
// --------------
// The DSL kernel requires N to be divisible by 128 and K by 64
// (=sf_vec_size*4). Those constraints are naturally satisfied by TP-split
// weight layouts. M may be any positive value supported by the AOT kernel's
// direct-output epilogue.

#pragma once

#ifdef CUTE_DSL_GEMM_NVFP4_ENABLED
#include "cutedsl_gemm_nvfp4_all.h"
#endif

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>

namespace trt_edgellm
{
namespace kernels
{

class CuteDslGemmNvFp4Runner
{
public:
    /// Construct with the desired MMA N-tile. Must be one of {64, 128}.
    explicit CuteDslGemmNvFp4Runner(int32_t mmaTilerN = 64);

    ~CuteDslGemmNvFp4Runner();

    /// Load all AOT kernel modules. Thread-safe, idempotent. Call once
    /// per process before the first `run()`.
    static bool loadKernelModules();

    /// Unload modules (call at process shutdown if resource cleanup matters).
    static void unloadKernelModules();

    /// Is there an AOT variant compiled for this GPU + tile combo?
    static bool canImplement(int32_t smVersion, int32_t mmaTilerN);

    /// Launch the blockscaled NVFP4 GEMM with FP16 output.
    ///
    /// @param a      FP4 packed activation, device ptr to [M, K/2] uint8
    /// @param b      FP4 packed weight,     device ptr to [N, K/2] uint8
    /// @param aSF    UE4M3 atom-layout SF buffer for A, device ptr
    /// @param bSF    UE4M3 atom-layout SF buffer for B, device ptr
    /// @param d      FP16 output buffer, device ptr to [M, N] half
    /// @param alpha  device ptr to the per tensor FP32 dequant scale
    /// @param M, N, K dynamic shapes. N must be multiple of 128 and K a
    ///               multiple of 64 (satisfied by TP layouts). M may be any
    ///               positive integer.
    /// @param stream CUDA stream
    cudaError_t run(void const* a, void const* b, void const* aSF, void const* bSF, void* d, void const* alpha,
        int32_t M, int32_t N, int32_t K, cudaStream_t stream);

    /// Launch the blockscaled NVFP4 GEMM with FP8 E4M3 output.
    ///
    /// Same A / B / SFA / SFB layout as run(); only the output dtype
    /// differs. The kernel's epilogue casts FP32 accumulator → FP8 E4M3.
    ///
    /// @param d      FP8 E4M3 output, device ptr to [M, N] uint8 (1 byte/elem)
    /// @param other params same as run()
    cudaError_t runFp8(void const* a, void const* b, void const* aSF, void const* bSF, void* d, void const* alpha,
        int32_t M, int32_t N, int32_t K, cudaStream_t stream);

private:
    int32_t mMmaTilerN;

#ifdef CUTE_DSL_GEMM_NVFP4_ENABLED
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP16_TN64_ENABLED
    static gemm_blackwell_nvfp4_fp16_tn64_Kernel_Module_t sModFp16Tn64;
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP16_TN128_ENABLED
    static gemm_blackwell_nvfp4_fp16_tn128_Kernel_Module_t sModFp16Tn128;
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP8_TN64_ENABLED
    static gemm_blackwell_nvfp4_fp8_tn64_Kernel_Module_t sModFp8Tn64;
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP8_TN128_ENABLED
    static gemm_blackwell_nvfp4_fp8_tn128_Kernel_Module_t sModFp8Tn128;
#endif
    // Warp-specialised variant slots. Coexist with the non-WS slots above.
    // When a `_WS_*_ENABLED` define is present, dispatch uses it as the
    // production path; otherwise it falls through to the non-WS sibling.
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN64_ENABLED
    static gemm_blackwell_nvfp4_ws_fp16_tn64_Kernel_Module_t sModWsFp16Tn64;
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN128_ENABLED
    static gemm_blackwell_nvfp4_ws_fp16_tn128_Kernel_Module_t sModWsFp16Tn128;
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN64_ENABLED
    static gemm_blackwell_nvfp4_ws_fp8_tn64_Kernel_Module_t sModWsFp8Tn64;
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN128_ENABLED
    static gemm_blackwell_nvfp4_ws_fp8_tn128_Kernel_Module_t sModWsFp8Tn128;
#endif
#endif // CUTE_DSL_GEMM_NVFP4_ENABLED
    static std::mutex sMutex;
    static bool sLoaded;
};

} // namespace kernels
} // namespace trt_edgellm
