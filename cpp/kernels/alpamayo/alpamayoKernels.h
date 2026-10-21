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

#include <cuda_runtime_api.h>

//! FP8 buffers and dimensions for the Alpamayo 2B gate/up kernel.
struct AlpamayoFusedGateUpParams
{
    void const* x;
    void const* w;
    void* out;
    float const* gateDecode;
    float const* upEncode;
    int m;
    int n;
    int k;
    int gridN;
    int gridSize;
};

//! SM110-only entry points. Return zero on success or a CUDA driver/runtime error code.
extern "C"
{
    //! Set kernel attributes on the current device before execution or graph capture.
    int initializeAlpamayoFusedGateUp();

    //! FP8 x[M, K] @ packed W[N, K]^T with SwiGLU; FP8 out[M, N / 2].
    int launch_alpamayo_fp8_gate_up_swiglu_2cta_m256o256k128_s3_tanh(
        AlpamayoFusedGateUpParams const& params, cudaStream_t stream);
}
