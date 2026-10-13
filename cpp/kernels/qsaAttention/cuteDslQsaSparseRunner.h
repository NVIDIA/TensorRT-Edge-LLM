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

#if defined(CUTE_DSL_QSA_ENABLED)
#include <cuda.h>

#if CUDA_VERSION >= 12000 && CUDA_VERSION < 12080
typedef CUlibrary cudaLibrary_t;
extern "C" cudaError_t cudaLibraryUnload(cudaLibrary_t library);
#endif

#include "kernels/cuteDslModuleLoader.h"

#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_qsa_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif // defined(CUTE_DSL_QSA_ENABLED)

#include <NvInferRuntime.h>
#include <cstdint>

namespace trt_edgellm
{

//! Per-launch parameters for the QSA sparse-GQA prefill kernel.
//!
//! Q/K/V/O are dense padded BSND tensors: Q/O are [batchSize, seqLen, numQHeads, headDim] and
//! K/V are [batchSize, seqLen, numKVHeads, headDim], all fp16 (or bf16), contiguous with D
//! innermost. @c indices is the QSA indexer output [batchSize, seqLen, topK] Int32: per query
//! row the selected KV token ids, -1-padded on the right, unsorted, distinct, all < token + 1.
//! Causality lives entirely in the index list — the kernel applies no positional mask.
//! @c contextLengths is [batchSize] Int32; output rows at or past the live length store exact
//! zeros, and K/V rows at or past it are never referenced (indices only name live tokens).
struct QsaSparsePrefillParams
{
    void const* qPtr{};
    void const* kPtr{};
    void const* vPtr{};
    void* oPtr{};
    int32_t const* indices{};
    int32_t const* contextLengths{};
    int32_t batchSize{};
    int32_t seqLen{};
    int32_t numQHeads{};
    int32_t numKVHeads{};
    int32_t headDim{};
    int32_t topK{};
    float attentionScale{};
    cudaStream_t stream{};
};

//! Runner for the CuTe DSL QSA sparse-GQA prefill kernels (Qwen3.8-Flash-Next).
//!
//! One CTA per (query token, KV head); the GQA head group is the MMA M tile and the KV
//! traversal is a per-row cp.async gather over the index list. The AOT family bakes only the
//! head dim and tile tuning; batch, seq, head counts, topK and strides stay runtime-dynamic.
class CuteDslQsaSparsePrefillRunner
{
public:
    CuteDslQsaSparsePrefillRunner() = delete;

    //! Returns whether the QSA AOT family covers this shape on this SM.
    static bool canImplement(
        int32_t numQHeads, int32_t numKVHeads, int32_t headDim, int32_t smVersion, nvinfer1::DataType dataType);

    //! Ensures the variant selected by run() is loaded (CUDA-graph-capture aware).
    static bool preflight(nvinfer1::DataType dataType, cudaStream_t stream);

    //! Launches sparse prefill attention over the gathered index lists.
    static bool run(nvinfer1::DataType dataType, QsaSparsePrefillParams const& params);

private:
#if defined(CUTE_DSL_QSA_ENABLED)
    static detail::LazyKernelModule<qsa_sparse_d256_fp16_Kernel_Module_t> sSparseD256Fp16;
    static detail::LazyKernelModule<qsa_sparse_d256_bf16_Kernel_Module_t> sSparseD256Bf16;
#endif // defined(CUTE_DSL_QSA_ENABLED)
};

//! Per-launch parameters for the QSA split-K sparse DECODE kernel.
//!
//! One query token per sequence: @c qPtr / @c oPtr are [batchSize, 1, numQHeads, headDim].
//! K/V come from the paged pool [2*numPages, 128, numKVHeads, poolHeadDim] through
//! @c pageTable [batchSize, 2, maxPagesPerSeq] (V page ids pre-offset by +numPages); only
//! columns [0, headDim) of each row are read — the tail carries the QSA indexer state.
//! @c indices is [batchSize, 1, topK] Int32 (-1 padded). @c contextLengths is the TOTAL
//! per-sequence length including the new token. @c partialO (fp32
//! [batchSize*numKVHeads*kMaxSplits, kPartialRows, headDim]), @c partialStats (fp32
//! [batchSize*numKVHeads*kMaxSplits, 2, kPartialRows]) and @c splitCounters (int32
//! [batchSize*numKVHeads]) live in plugin workspace; counters MUST be zero on entry
//! (the kernel release-resets them, but fresh workspaces start as garbage).
struct QsaSparseDecodeParams
{
    void const* qPtr{};
    void const* kvPoolPtr{};
    int32_t const* pageTable{};
    int32_t const* indices{};
    int32_t const* contextLengths{};
    void* oPtr{};
    float* partialO{};
    float* partialStats{};
    int32_t* splitCounters{};
    int32_t batchSize{};
    int32_t numQHeads{};
    int32_t numKVHeads{};
    int32_t headDim{};
    int32_t poolHeadDim{};
    int32_t numFlatPages{}; //!< 2 * numPages (both planes)
    int32_t maxPagesPerSeq{};
    int32_t topK{};
    float attentionScale{};
    cudaStream_t stream{};
};

//! Runner for the single-launch split-K QSA sparse decode kernel.
class CuteDslQsaSparseDecodeRunner
{
public:
    //! Grid split dimension baked into the AOT variants (build_cutedsl.py --max_splits).
    static constexpr int32_t kMaxSplits{8};
    //! Accumulator row count of the partial workspaces (m_block_size).
    static constexpr int32_t kPartialRows{16};

    CuteDslQsaSparseDecodeRunner() = delete;

    static bool canImplement(int32_t numQHeads, int32_t numKVHeads, int32_t headDim, int32_t poolHeadDim,
        int32_t smVersion, nvinfer1::DataType dataType);

    //! Ensures the decode variant is loaded (CUDA-graph-capture aware).
    static bool preflight(nvinfer1::DataType dataType, cudaStream_t stream);

    static bool run(nvinfer1::DataType dataType, QsaSparseDecodeParams const& params);

private:
#if defined(CUTE_DSL_QSA_ENABLED)
    static detail::LazyKernelModule<qsa_sparse_decode_d256_fp16_Kernel_Module_t> sDecodeD256Fp16;
    static detail::LazyKernelModule<qsa_sparse_decode_d256_bf16_Kernel_Module_t> sDecodeD256Bf16;
#endif // defined(CUTE_DSL_QSA_ENABLED)
};

} // namespace trt_edgellm
