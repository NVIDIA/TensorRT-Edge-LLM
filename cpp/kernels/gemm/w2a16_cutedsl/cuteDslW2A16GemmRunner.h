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
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#if defined(CUTE_DSL_W2A16_GEMM_ENABLED)
namespace trt_edgellm
{
//! Contiguous current-device buffers; all pointers must be 16-byte aligned.
struct CuteDslW2A16GemmParams
{
    int32_t m{};
    int32_t n{};
    int32_t k{};
    void const* input{};  //!< BF16 [M,K].
    void const* values{}; //!< Byte payload [N/B,K/256,B,64], B=128 when N%128=0, otherwise 8.
    void const* bases{};  //!< FP16 bases [K/256,N], stored without N padding.
    void const* lookup{}; //!< BF16 [256,8] magnitude grid (4096 bytes).
    void* output{};       //!< BF16 [M,N]; must not overlap inputs or workspace.
};

//! Standalone SM110 (Thor), SM120 and GB10 (SM121) IQ2_XXS dense runner; modules remain resident for process lifetime.
class CuteDslW2A16GemmRunner
{
public:
    //! Pure host shape/hardware admission; no CUDA calls or pointer access.
    static bool canImplement(int32_t m, int32_t n, int32_t k, int32_t smVersion, int32_t numSms) noexcept;
    //! FP32 split-K scratch bytes, zero for no split or unsupported shapes.
    //! Allocate 256-byte aligned scratch; concurrent calls require disjoint scratch/output.
    static size_t getWorkspaceSize(int32_t m, int32_t n, int32_t k) noexcept;
    //! Shape-only preload before capture, once for each selected variant.
    static bool ensureKernelModules(CuteDslW2A16GemmParams const& params, cudaStream_t stream) noexcept;
    //! Enqueue into an explicit non-default stream; requires prior preload.
    //! No allocation, compilation, module loading or host synchronization during run/replay.
    //! Returns a CUDA error code, zero on success. Device ownership is a caller precondition.
    int32_t run(CuteDslW2A16GemmParams const& params, void* workspace, size_t workspaceBytes,
        cudaStream_t stream) const noexcept;
};
} // namespace trt_edgellm
#endif
