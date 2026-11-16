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

// Adapter that lets Int4GroupwiseGemmPluginV2 dispatch to the CuTe DSL INT4 FP16
// GEMM AOT variants as autotuned tactics. Weights are consumed already in the
// cuteDSL fragment layout (repacked at export time, bN=128 pinned), so there is
// no runtime repack and no cached second buffer. Gated on the cuteDSL artifact
// being built with the int4_fp16_gemm group.
#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace plugins
{

//! One compiled cuteDSL variant (CTA tile + pipeline depth + serial split-K).
//! bN is pinned to 128 for every variant (tile-independent weight layout).
struct CuteDslInt4Variant
{
    int32_t bM;
    int32_t bN;
    int32_t bK;
    int32_t stages;
    int32_t splitK;
};

//! Number of compiled cuteDSL variants in the table (16: the subset of
//! the 4 bM x {2,3,4} stages x {1,2,4,8,16} split-K = 60 space; see the
//! INT4_FP16_GEMM_VARIANTS X-list in cuteDslInt4Gemm.cpp).
int32_t cuteDslInt4NumVariants();

//! Variant at table index [0, cuteDslInt4NumVariants()).
CuteDslInt4Variant const& cuteDslInt4VariantAt(int32_t index);

//! True if this variant can serve a positive (N, K) problem where K%64==0 and
//! splitK divides ceil(K/bK). (bN is always 128; N residue is predicated.)
bool cuteDslInt4VariantValid(CuteDslInt4Variant const& v, int32_t N, int32_t K);

//! Load all cuteDSL kernel modules once (handle-checked). Returns false on any
//! load failure.
bool cuteDslInt4GemmLoadModules();

//! Number of uint32 words (== the fragment weight buffer row count x 128) for a
//! (N,K) tile. Rows = ceil(N/bN)*ceil(K/bK)*(bK/16)*(bN/64); each row is 128 words.
int64_t cuteDslInt4RepackedWords(int32_t N, int32_t K, int32_t bN, int32_t bK);

//! Number of int32 lock entries (serial split-K semaphore) for a launch grid.
int64_t cuteDslInt4LockCount(int32_t M, int32_t N, int32_t bM, int32_t bN);

//! Launch the variant. A: [M,K] fp16; fragW: fragment-order uint32 weights (the
//! plugin's INT8 input[1] reinterpreted); scales: [ceil(K/G),N] fp16; C: [M,N]
//! fp16 (also serves as the workspace placeholder); locks: zero-initialized int32
//! [cuteDslInt4LockCount]; swizzle: grouped-M width. Returns 0 on success.
int32_t cuteDslInt4GemmLaunch(CuteDslInt4Variant const& v, void const* A, void const* fragW, void const* scales,
    void* C, void* locks, int32_t M, int32_t N, int32_t K, int32_t groupSize, int32_t swizzle, cudaStream_t stream);

} // namespace plugins
} // namespace trt_edgellm

#endif // CUTE_DSL_INT4_FP16_GEMM_ENABLED
