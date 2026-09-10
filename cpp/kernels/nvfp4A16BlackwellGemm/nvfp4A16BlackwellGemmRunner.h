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

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace trt_edgellm
{
namespace kernels
{

enum class Nvfp4A16BlackwellDtype : int32_t
{
    kFp16 = 0,
    kBf16 = 1,
};

enum class Nvfp4A16BlackwellTokenTile : int32_t
{
    kM8 = 8,
    kM16 = 16,
    kM32 = 32,
    kM64 = 64,
    kM128 = 128,
    kM256 = 256,
};

struct Nvfp4A16BlackwellGemmParams
{
    void const* activation;
    void const* qweight;
    void const* blockScales;
    float const* globalScale;
    void* output;
    int32_t numTokens;
    int32_t outFeatures;
    int32_t inFeatures;
    Nvfp4A16BlackwellDtype dtype;
};

//! Dense W4A16 GEMM for the export-time SM110 opaque weight layout.
//!
//! Logical tensors are activation [numTokens, inFeatures], weight
//! [outFeatures, inFeatures], and output [numTokens, outFeatures]. The opaque
//! inputs are qweight [N/128, K/64, 128, 32] packed E2M1 bytes and blockScales
//! [N/128, K/64, 128, 4] raw E4M3 bytes. globalScale is a device FP32 scalar.
//! AOT variants bake only the activation dtype and post-transpose MMA
//! (TM,TN,TK) tile. Logical M=numTokens maps to MMA N.
//! Runtime N/K define the tensor descriptors after the runner validates their
//! positive, aligned dimensions.
class Nvfp4A16BlackwellGemmRunner
{
public:
    //! Load only the AOT module selected by this runtime dtype and token tile. Call this
    //! outside CUDA graph capture to prewarm the exact variant that run()
    //! will dispatch.
    static cudaError_t prepare(Nvfp4A16BlackwellDtype dtype, int32_t numTokens, int32_t outFeatures, int32_t inFeatures,
        cudaStream_t stream) noexcept;

    //! Legacy test helper that eagerly loads every compiled variant. Runtime
    //! plugin paths must use prepare() instead.
    static cudaError_t loadKernelModules(cudaStream_t stream) noexcept;

    static bool isSupported(int32_t smVersion, Nvfp4A16BlackwellDtype dtype, int32_t numTokens, int32_t outFeatures,
        int32_t inFeatures) noexcept;

    static Nvfp4A16BlackwellTokenTile selectTokenTile(int32_t numTokens) noexcept;

    static size_t getWorkspaceSize(Nvfp4A16BlackwellGemmParams const& params) noexcept;

    static cudaError_t run(
        Nvfp4A16BlackwellGemmParams const& params, void* workspace, size_t workspaceSize, cudaStream_t stream) noexcept;
};

} // namespace kernels
} // namespace trt_edgellm
