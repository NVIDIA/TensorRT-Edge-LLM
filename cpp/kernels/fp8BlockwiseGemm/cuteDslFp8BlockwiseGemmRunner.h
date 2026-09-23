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

#include <cstdint>

namespace trt_edgellm::kernels
{

struct CuteDslFp8BlockwiseGemmParams
{
    void const* activation;
    void const* weight;
    float const* activationScales;
    float const* weightScales;
    void* output;
    int32_t numTokens;
    int32_t outFeatures;
    int32_t inFeatures;
};

//! Dispatch the SM110 CuTe DSL blockwise E4M3 GEMM AOT module.
class CuteDslFp8BlockwiseGemmRunner
{
public:
    //! Load the module before CUDA graph capture without launching a GEMM.
    static cudaError_t prepare(cudaStream_t stream) noexcept;

    static bool isSupported(int32_t smVersion, int32_t numTokens, int32_t outFeatures, int32_t inFeatures) noexcept;

    static cudaError_t run(CuteDslFp8BlockwiseGemmParams const& params, cudaStream_t stream) noexcept;
};

} // namespace trt_edgellm::kernels
