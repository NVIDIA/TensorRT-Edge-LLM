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

#include "common/dataType.h"
#include <cstdint>

namespace trt_edgellm
{

//! Initial and continuation prefill queries attend to individual causal token-index lists.
//!
//! Q/O are [batchSize, seqLen, numQHeads, headDim]. The pool and page table use
//! the decode layout; columns [headDim, poolHeadDim) are never read.
//! @c indices is [batchSize, seqLen, topK], with causal token ids followed by -1.
//! @c queryLengths is [batchSize]; padded query rows store exact zeros without
//! reading their Q, index list, or KV data.
struct QsaSparsePrefillParams
{
    void const* qPtr{};
    void const* kvPoolPtr{};
    int32_t const* pageTable{};
    int32_t const* indices{};
    int32_t const* queryLengths{};
    void* oPtr{};
    int32_t batchSize{};
    int32_t seqLen{};
    int32_t numQHeads{};
    int32_t numKVHeads{};
    int32_t headDim{};
    int32_t poolHeadDim{};
    int32_t numFlatPages{};
    int32_t maxPagesPerSeq{};
    int32_t topK{};
    float attentionScale{};
    cudaStream_t stream{};
};

//! Sparse attention with one CTA per query token and KV head.
class CuteDslQsaSparsePrefillRunner
{
public:
    CuteDslQsaSparsePrefillRunner() = delete;

    //! Returns whether the paged QSA AOT family covers this shape on this SM.
    static bool canImplement(int32_t numQHeads, int32_t numKVHeads, int32_t headDim, int32_t poolHeadDim,
        int32_t smVersion, trt_edgellm::DataType dataType);

    //! Loads the paged prefill variant for this dtype before CUDA graph capture.
    static bool preflight(trt_edgellm::DataType dataType, cudaStream_t stream);

    //! Launches all prefill queries against their individual causal index lists.
    static bool run(trt_edgellm::DataType dataType, QsaSparsePrefillParams const& params);

private:
#if defined(CUTE_DSL_QSA_ENABLED)
    static detail::LazyKernelModule<qsa_sparse_chunk_prefill_d256_fp16_Kernel_Module_t> sSparseD256Fp16;
    static detail::LazyKernelModule<qsa_sparse_chunk_prefill_d256_bf16_Kernel_Module_t> sSparseD256Bf16;
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
        int32_t smVersion, trt_edgellm::DataType dataType);

    //! Ensures the decode variant is loaded (CUDA-graph-capture aware).
    static bool preflight(trt_edgellm::DataType dataType, cudaStream_t stream);

    static bool run(trt_edgellm::DataType dataType, QsaSparseDecodeParams const& params);

private:
#if defined(CUTE_DSL_QSA_ENABLED)
    static detail::LazyKernelModule<qsa_sparse_decode_d256_fp16_Kernel_Module_t> sDecodeD256Fp16;
    static detail::LazyKernelModule<qsa_sparse_decode_d256_bf16_Kernel_Module_t> sDecodeD256Bf16;
#endif // defined(CUTE_DSL_QSA_ENABLED)
};

} // namespace trt_edgellm
