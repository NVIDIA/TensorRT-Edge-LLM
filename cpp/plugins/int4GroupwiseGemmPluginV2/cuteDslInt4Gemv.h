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

// Adapter that lets Int4GroupwiseGemmPluginV2 dispatch decode-regime (small M)
// launches to the CUDA-core INT4 FP16 GEMV variants.  The GEMV consumes the
// SAME fragment-layout weight buffer as the GEMM tactics (bN=128/bK=64), so the
// plugin routes small M here and larger M to the GEMM tactics off one weight
// copy.  Gated on the cuteDSL artifact being built with the int4_fp16_gemv group.
#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace plugins
{

//! Largest M the GEMV kernel has a compiled variant for (one per M in [1, 8]).
constexpr int32_t kGemvMaxM = 8;

//! Load all GEMV kernel modules once (handle-checked). Returns false on any
//! load failure (so the caller can fall back to the GEMM path).
bool cuteDslInt4GemvLoadModules();

//! True if a compiled GEMV variant exists for this M (1 <= M <= kGemvMaxM).
bool cuteDslInt4GemvSupported(int32_t M);

//! Launch the M-variant GEMV. A: [M,K] fp16; fragW: fragment-order uint32 weights
//! (the plugin's INT8 input[1] reinterpreted); scales: [ceil(K/G),N] fp16; Out:
//! [M,N] fp16 written directly. Single launch, no workspace/locks. Returns 0 on
//! success, -1 if M is unsupported or a launch fails.
int32_t cuteDslInt4GemvLaunch(int32_t M, void const* A, void const* fragW, void const* scales, void* Out, int32_t N,
    int32_t K, int32_t groupSize, cudaStream_t stream);

} // namespace plugins
} // namespace trt_edgellm

#endif // CUTE_DSL_INT4_FP16_GEMM_ENABLED
