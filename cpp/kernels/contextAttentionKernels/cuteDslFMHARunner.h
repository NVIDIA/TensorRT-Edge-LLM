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

#include "kernels/cuteDslModuleLoader.h"

#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK

#include <NvInferRuntime.h>
#include <climits>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{

/**
 * @brief Unified runner for CuTe DSL compiled FMHA kernels (Blackwell SM100/101/110).
 *
 * Supports two execution modes via AOT-compiled kernel variants:
 *
 * 1. LLM prefill/chunked-prefill: batched Q [B,S_q,H_q,D] + combined KV cache
 *    [B,2,H_kv,Cap,D] with causal masking and optional sliding window.
 *
 * 2. ViT: packed varlen separate Q/K/V [total_S,H,D] with cu_seqlens [B+1]
 *    for ragged batching, bidirectional attention (no causal mask).
 *
 * Each mode has its own kernel modules and run() overload.
 */
class CuteDslFMHARunner
{
public:
    CuteDslFMHARunner(int32_t numQHeads, int32_t numKVHeads, int32_t headDim, int32_t batchSize = 0,
        int32_t seqLenQ = 0, int32_t kvCacheCapacity = 0);

    ~CuteDslFMHARunner() = default;
    CuteDslFMHARunner(CuteDslFMHARunner const&) = delete;
    CuteDslFMHARunner& operator=(CuteDslFMHARunner const&) = delete;

    static bool canImplement(int32_t headSize, int32_t smVersion);
    static bool canImplementViT(int32_t headSize, int32_t smVersion);

    //! Ensures the exact dense LLM variant selected by run() is loaded.
    bool preflightLlm(cudaStream_t stream, int32_t slidingWindowSize = INT_MAX, bool fp8Input = false,
        float skipSoftmaxThresholdLog2 = 0.0F);

    //! Ensures the exact paged LLM variant selected by runPaged() is loaded.
    bool preflightPaged(cudaStream_t stream, int32_t slidingWindowSize = INT_MAX, bool fp8Input = false,
        bool isCausal = true, float skipSoftmaxThresholdLog2 = 0.0F, bool useBidirectional = false);

    //! Ensures the exact packed ViT variant selected by run() is loaded.
    bool preflightViT(cudaStream_t stream);

    /**
     * @brief LLM FMHA: batched Q + combined KV cache with causal masking.
     *
     * Output is always FP16. Selects kernel variant based on fp8Input:
     *   - fp8Input=false → FP16 kernels (all scales ignored)
     *   - fp8Input=true  → FP8-input / FP16-output kernels
     *
     * @param qPtr Query [B, S_q, H_q, D]
     * @param kvPtr Combined KV cache [B, 2, H_kv, Cap, D]
     * @param oPtr Output [B, S_q, H_q, D] (always FP16)
     * @param cuKVSeqLens Cumulative KV sequence lengths [B+1]
     * @param stream CUDA stream
     * @param attentionScale Model-defined multiplier applied to QK^T before softmax. For FP8 input, the effective
     *        softmax scale is attentionScale * qScale * kScale.
     * @param slidingWindowSize Sliding window size (INT_MAX = disabled)
     * @param fp8Input Whether Q/KV are FP8 E4M3
     * @param qScale Q dequant scale (quant→orig), ignored when fp8Input=false
     * @param kScale K dequant scale (quant→orig), ignored when fp8Input=false
     * @param vScale V dequant scale (quant→orig), applied to the attention output and ignored when fp8Input=false
     * @param skipSoftmaxThresholdLog2 Skip-softmax (BLASST) threshold as log2(lambda). A finite negative value
     *        (lambda in (0,1)) dispatches the skip-softmax kernel variant, which skips the P*V GEMM of KV
     *        tiles whose contribution is negligible — approximate, FP16 causal only. 0.0 (the default,
     *        log2 of the degenerate lambda = 1) disables skip, mirroring the slidingWindowSize = INT_MAX
     *        sentinel convention.
     */
    bool run(void const* qPtr, void const* kvPtr, void* oPtr, int32_t const* cuKVSeqLens, cudaStream_t stream,
        float attentionScale, int32_t slidingWindowSize = INT_MAX, bool fp8Input = false, float qScale = 1.0F,
        float kScale = 1.0F, float vScale = 1.0F, float skipSoftmaxThresholdLog2 = 0.0F);

    /**
     * @brief LLM FMHA over a paged KV cache.
     *
     * Dispatches a dedicated CuTe DSL AOT variant that reads K/V directly from
     * a paged pool using kvCachePageList. The logical descriptor shape is
     * [numPages, H_kv, tokensPerPage, D], while the physical pool is fixed to
     * NHD [numPages, tokensPerPage, H_kv, D]. This path maps one logical K/V
     * TMA tile to one physical page. The current CuTe DSL variants use a K/V
     * tile width of 128 tokens, so tokensPerPage must be 128 to avoid
     * multi-page tile stitching or a gather workspace. Setting both block
     * range pointers selects the FP16 D512 bidirectional-mask variant; mixed
     * null pointers are invalid.
     *
     * @param qPtr Query [B, S_q, H_q, D]
     * @param pagedKVPoolPtr Paged KV pool [numPages, tokensPerPage, H_kv, D]
     * @param kvCachePageList Page table [B, 2, maxPagesPerSeq], K pages then V pages
     * @param oPtr Output [B, S_q, H_q, D] (always FP16)
     * @param cuKVSeqLens Cumulative KV sequence lengths [B+1]
     * @param numPages Number of pages in the paged KV pool
     * @param maxPagesPerSeq Max logical pages per sequence
     * @param tokensPerPage Number of tokens per page
     * @param kvDataType Paged KV cache dtype (FP16 or FP8)
     * @param stream CUDA stream
     * @param slidingWindowSize Sliding window size (INT_MAX = disabled)
     * @param fp8Input Whether Q/KV are FP8 E4M3
     * @param qScale Q dequant scale, ignored when fp8Input=false
     * @param kScale K dequant scale, ignored when fp8Input=false
     * @param vScale V dequant scale, ignored when fp8Input=false
     * @param isCausal Whether to dispatch a causal or dense non-causal variant
     * @param skipSoftmaxThresholdLog2 Skip-softmax threshold as log2(lambda), or 0 to disable
     * @param bidirectionalBlockBegin Optional inclusive bidirectional-block begin positions [B, S_q].
     * Text/padding rows use -1; every row in a disjoint contiguous vision run
     * must repeat that run's begin position.
     * @param bidirectionalBlockEnd Optional inclusive bidirectional-block end positions [B, S_q].
     * Text/padding rows use -1; every row in a disjoint contiguous vision run
     * must repeat that run's end position.
     */
    bool runPaged(void const* qPtr, void const* pagedKVPoolPtr, int32_t const* kvCachePageList, void* oPtr,
        int32_t const* cuKVSeqLens, int32_t numPages, int32_t maxPagesPerSeq, int32_t tokensPerPage,
        nvinfer1::DataType kvDataType, cudaStream_t stream, float attentionScale, int32_t slidingWindowSize = INT_MAX,
        bool fp8Input = false, float qScale = 1.0F, float kScale = 1.0F, float vScale = 1.0F, bool isCausal = true,
        float skipSoftmaxThresholdLog2 = 0.0F, int32_t const* bidirectionalBlockBegin = nullptr,
        int32_t const* bidirectionalBlockEnd = nullptr);

    /**
     * @brief ViT FMHA: packed varlen separate Q/K/V, bidirectional.
     *
     * Output is always FP16. Selects kernel variant based on fp8Input:
     *   - fp8Input=false → FP16 kernels (all scales ignored)
     *   - fp8Input=true  → FP8-input / FP16-output kernels
     *
     * @param qPtr  Query  [total_S, H, D]
     * @param kPtr  Key    [total_S, H, D]
     * @param vPtr  Value  [total_S, H, D]
     * @param oPtr  Output [total_S, H, D] (always FP16)
     * @param cuSeqLens Cumulative sequence lengths [B+1]
     * @param totalSeqLen Sum of all sequence lengths
     * @param maxSeqLen Longest individual sequence length
     * @param batchSize Number of sequences
     * @param stream CUDA stream
     * @param attentionScale Absolute multiplier applied to QK^T before softmax. For FP8 input, the effective
     *        softmax scale is attentionScale * qScale * kScale.
     * @param fp8Input Whether Q/K/V are FP8 E4M3
     * @param qScale Q dequant scale (quant→orig), ignored when fp8Input=false
     * @param kScale K dequant scale (quant→orig), ignored when fp8Input=false
     * @param vScale V dequant scale (quant→orig), applied to the attention output and ignored when fp8Input=false
     */
    bool run(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuSeqLens,
        int32_t totalSeqLen, int32_t maxSeqLen, int32_t batchSize, cudaStream_t stream, float attentionScale,
        bool fp8Input = false, float qScale = 1.0F, float kScale = 1.0F, float vScale = 1.0F);

private:
    int32_t mBatchSize{};
    int32_t mSeqLenQ{};
    int32_t mKVCacheCapacity{};
    int32_t mNumHeadsQ{};
    int32_t mNumHeadsK{};
    int32_t mHeadDim{};

    // LLM kernel modules (FP16)
    static detail::LazyKernelModule<fmha_d64_Kernel_Module_t> sLLM_d64;
    static detail::LazyKernelModule<fmha_d128_Kernel_Module_t> sLLM_d128;
    static detail::LazyKernelModule<fmha_d256_Kernel_Module_t> sLLM_d256;
    static detail::LazyKernelModule<fmha_d64_sw_Kernel_Module_t> sLLM_d64_sw;
    static detail::LazyKernelModule<fmha_d128_sw_Kernel_Module_t> sLLM_d128_sw;
    static detail::LazyKernelModule<fmha_d256_sw_Kernel_Module_t> sLLM_d256_sw;

    // LLM skip-softmax (BLASST) kernel modules (FP16, causal, no sliding window)
    static detail::LazyKernelModule<fmha_d64_skipsoftmax_Kernel_Module_t> sLLM_d64_skipsoftmax;
    static detail::LazyKernelModule<fmha_d128_skipsoftmax_Kernel_Module_t> sLLM_d128_skipsoftmax;

    // LLM kernel modules (FP8 input, FP16 output)
    static detail::LazyKernelModule<fmha_d64_fp8_Kernel_Module_t> sLLM_d64_fp8;
    static detail::LazyKernelModule<fmha_d128_fp8_Kernel_Module_t> sLLM_d128_fp8;
    static detail::LazyKernelModule<fmha_d256_fp8_Kernel_Module_t> sLLM_d256_fp8;
    static detail::LazyKernelModule<fmha_d64_sw_fp8_Kernel_Module_t> sLLM_d64_sw_fp8;
    static detail::LazyKernelModule<fmha_d128_sw_fp8_Kernel_Module_t> sLLM_d128_sw_fp8;
    static detail::LazyKernelModule<fmha_d256_sw_fp8_Kernel_Module_t> sLLM_d256_sw_fp8;

    // LLM paged KV cache kernel modules (FP16)
    static detail::LazyKernelModule<fmha_d64_paged_Kernel_Module_t> sLLM_d64_paged;
    static detail::LazyKernelModule<fmha_d128_paged_Kernel_Module_t> sLLM_d128_paged;
    static detail::LazyKernelModule<fmha_d64_skipsoftmax_paged_Kernel_Module_t> sLLM_d64_skipsoftmax_paged;
    static detail::LazyKernelModule<fmha_d128_skipsoftmax_paged_Kernel_Module_t> sLLM_d128_skipsoftmax_paged;
    static detail::LazyKernelModule<fmha_d256_paged_Kernel_Module_t> sLLM_d256_paged;
    static detail::LazyKernelModule<fmha_d256_dense_paged_Kernel_Module_t> sLLM_d256_dense_paged;
    static detail::LazyKernelModule<fmha_d512_paged_Kernel_Module_t> sLLM_d512_paged;
    static detail::LazyKernelModule<fmha_d512_dense_paged_Kernel_Module_t> sLLM_d512_dense_paged;
    static detail::LazyKernelModule<fmha_d64_sw_paged_Kernel_Module_t> sLLM_d64_sw_paged;
    static detail::LazyKernelModule<fmha_d128_sw_paged_Kernel_Module_t> sLLM_d128_sw_paged;
    static detail::LazyKernelModule<fmha_d256_sw_paged_Kernel_Module_t> sLLM_d256_sw_paged;
    static detail::LazyKernelModule<fmha_d512_sw_paged_Kernel_Module_t> sLLM_d512_sw_paged;

    // D512 paged bidirectional-mask kernel module (FP16, runtime sliding window)
    static detail::LazyKernelModule<fmha_d512_paged_bidirectional_Kernel_Module_t> sLLM_d512_paged_bidirectional;

    // LLM paged KV cache kernel modules (FP8 input, FP16 output)
    static detail::LazyKernelModule<fmha_d64_paged_fp8_Kernel_Module_t> sLLM_d64_paged_fp8;
    static detail::LazyKernelModule<fmha_d128_paged_fp8_Kernel_Module_t> sLLM_d128_paged_fp8;
    static detail::LazyKernelModule<fmha_d256_paged_fp8_Kernel_Module_t> sLLM_d256_paged_fp8;
    static detail::LazyKernelModule<fmha_d256_dense_paged_fp8_Kernel_Module_t> sLLM_d256_dense_paged_fp8;
    static detail::LazyKernelModule<fmha_d512_paged_fp8_Kernel_Module_t> sLLM_d512_paged_fp8;
    static detail::LazyKernelModule<fmha_d512_dense_paged_fp8_Kernel_Module_t> sLLM_d512_dense_paged_fp8;
    static detail::LazyKernelModule<fmha_d64_sw_paged_fp8_Kernel_Module_t> sLLM_d64_sw_paged_fp8;
    static detail::LazyKernelModule<fmha_d128_sw_paged_fp8_Kernel_Module_t> sLLM_d128_sw_paged_fp8;
    static detail::LazyKernelModule<fmha_d256_sw_paged_fp8_Kernel_Module_t> sLLM_d256_sw_paged_fp8;
    static detail::LazyKernelModule<fmha_d512_sw_paged_fp8_Kernel_Module_t> sLLM_d512_sw_paged_fp8;

    // ViT kernel modules (FP16)
    static detail::LazyKernelModule<vit_fmha_d64_Kernel_Module_t> sViT_d64;
    static detail::LazyKernelModule<vit_fmha_d72_Kernel_Module_t> sViT_d72;
    static detail::LazyKernelModule<vit_fmha_d80_Kernel_Module_t> sViT_d80;
    static detail::LazyKernelModule<vit_fmha_d96_Kernel_Module_t> sViT_d96;
    static detail::LazyKernelModule<vit_fmha_d128_Kernel_Module_t> sViT_d128;

    static detail::LazyKernelModule<vit_fmha_d64_fp8_Kernel_Module_t> sViT_d64_fp8;
    static detail::LazyKernelModule<vit_fmha_d80_fp8_Kernel_Module_t> sViT_d80_fp8;
    static detail::LazyKernelModule<vit_fmha_d96_fp8_Kernel_Module_t> sViT_d96_fp8;
    static detail::LazyKernelModule<vit_fmha_d128_fp8_Kernel_Module_t> sViT_d128_fp8;
};

} // namespace trt_edgellm
