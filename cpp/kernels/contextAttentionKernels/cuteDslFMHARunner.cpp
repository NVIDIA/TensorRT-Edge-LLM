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

#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED

#include "cuteDslFMHARunner.h"

#include "attentionScaleUtils.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "cuteDslFmhaParams.h"
#include "cuteDslTensorDescriptors.h"

#include <climits>
#include <cmath>
#include <tuple>

namespace trt_edgellm
{

namespace
{

bool isSupportedBlackwellFmha(int32_t smVersion)
{
    return smVersion == 100 || smVersion == 101 || smVersion == 110;
}

template <auto moduleLoader, auto moduleUnloader, typename Module>
bool preflightVariant(detail::LazyKernelModule<Module>& state, char const* moduleName, cudaStream_t stream)
{
    return detail::ensureModuleLoaded<moduleLoader, moduleUnloader>(state, moduleName, stream);
}

} // namespace

// =====================================================================
// Static member initialization
// =====================================================================

// LLM (FP16)
detail::LazyKernelModule<fmha_d64_Kernel_Module_t> CuteDslFMHARunner::sLLM_d64{};
detail::LazyKernelModule<fmha_d128_Kernel_Module_t> CuteDslFMHARunner::sLLM_d128{};
detail::LazyKernelModule<fmha_d256_Kernel_Module_t> CuteDslFMHARunner::sLLM_d256{};
detail::LazyKernelModule<fmha_d64_sw_Kernel_Module_t> CuteDslFMHARunner::sLLM_d64_sw{};
detail::LazyKernelModule<fmha_d128_sw_Kernel_Module_t> CuteDslFMHARunner::sLLM_d128_sw{};
detail::LazyKernelModule<fmha_d256_sw_Kernel_Module_t> CuteDslFMHARunner::sLLM_d256_sw{};
// LLM skip-softmax (BLASST, FP16 causal)
detail::LazyKernelModule<fmha_d64_skipsoftmax_Kernel_Module_t> CuteDslFMHARunner::sLLM_d64_skipsoftmax{};
detail::LazyKernelModule<fmha_d128_skipsoftmax_Kernel_Module_t> CuteDslFMHARunner::sLLM_d128_skipsoftmax{};
detail::LazyKernelModule<fmha_d64_skipsoftmax_paged_Kernel_Module_t> CuteDslFMHARunner::sLLM_d64_skipsoftmax_paged{};
detail::LazyKernelModule<fmha_d128_skipsoftmax_paged_Kernel_Module_t> CuteDslFMHARunner::sLLM_d128_skipsoftmax_paged{};
// LLM (FP8 input, FP16 output)
detail::LazyKernelModule<fmha_d64_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d64_fp8{};
detail::LazyKernelModule<fmha_d128_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d128_fp8{};
detail::LazyKernelModule<fmha_d256_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d256_fp8{};
detail::LazyKernelModule<fmha_d64_sw_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d64_sw_fp8{};
detail::LazyKernelModule<fmha_d128_sw_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d128_sw_fp8{};
detail::LazyKernelModule<fmha_d256_sw_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d256_sw_fp8{};
// LLM paged KV cache (FP16)
detail::LazyKernelModule<fmha_d64_paged_Kernel_Module_t> CuteDslFMHARunner::sLLM_d64_paged{};
detail::LazyKernelModule<fmha_d128_paged_Kernel_Module_t> CuteDslFMHARunner::sLLM_d128_paged{};
detail::LazyKernelModule<fmha_d256_paged_Kernel_Module_t> CuteDslFMHARunner::sLLM_d256_paged{};
detail::LazyKernelModule<fmha_d256_dense_paged_Kernel_Module_t> CuteDslFMHARunner::sLLM_d256_dense_paged{};
detail::LazyKernelModule<fmha_d512_paged_Kernel_Module_t> CuteDslFMHARunner::sLLM_d512_paged{};
detail::LazyKernelModule<fmha_d512_dense_paged_Kernel_Module_t> CuteDslFMHARunner::sLLM_d512_dense_paged{};
detail::LazyKernelModule<fmha_d64_sw_paged_Kernel_Module_t> CuteDslFMHARunner::sLLM_d64_sw_paged{};
detail::LazyKernelModule<fmha_d128_sw_paged_Kernel_Module_t> CuteDslFMHARunner::sLLM_d128_sw_paged{};
detail::LazyKernelModule<fmha_d256_sw_paged_Kernel_Module_t> CuteDslFMHARunner::sLLM_d256_sw_paged{};
detail::LazyKernelModule<fmha_d512_sw_paged_Kernel_Module_t> CuteDslFMHARunner::sLLM_d512_sw_paged{};
detail::LazyKernelModule<fmha_d512_paged_bidirectional_Kernel_Module_t>
    CuteDslFMHARunner::sLLM_d512_paged_bidirectional{};
// LLM paged KV cache (FP8 input, FP16 output)
detail::LazyKernelModule<fmha_d64_paged_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d64_paged_fp8{};
detail::LazyKernelModule<fmha_d128_paged_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d128_paged_fp8{};
detail::LazyKernelModule<fmha_d256_paged_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d256_paged_fp8{};
detail::LazyKernelModule<fmha_d256_dense_paged_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d256_dense_paged_fp8{};
detail::LazyKernelModule<fmha_d512_paged_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d512_paged_fp8{};
detail::LazyKernelModule<fmha_d512_dense_paged_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d512_dense_paged_fp8{};
detail::LazyKernelModule<fmha_d64_sw_paged_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d64_sw_paged_fp8{};
detail::LazyKernelModule<fmha_d128_sw_paged_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d128_sw_paged_fp8{};
detail::LazyKernelModule<fmha_d256_sw_paged_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d256_sw_paged_fp8{};
detail::LazyKernelModule<fmha_d512_sw_paged_fp8_Kernel_Module_t> CuteDslFMHARunner::sLLM_d512_sw_paged_fp8{};

// ViT
detail::LazyKernelModule<vit_fmha_d64_Kernel_Module_t> CuteDslFMHARunner::sViT_d64{};
detail::LazyKernelModule<vit_fmha_d72_Kernel_Module_t> CuteDslFMHARunner::sViT_d72{};
detail::LazyKernelModule<vit_fmha_d80_Kernel_Module_t> CuteDslFMHARunner::sViT_d80{};
detail::LazyKernelModule<vit_fmha_d96_Kernel_Module_t> CuteDslFMHARunner::sViT_d96{};
detail::LazyKernelModule<vit_fmha_d128_Kernel_Module_t> CuteDslFMHARunner::sViT_d128{};
detail::LazyKernelModule<vit_fmha_d64_fp8_Kernel_Module_t> CuteDslFMHARunner::sViT_d64_fp8{};
detail::LazyKernelModule<vit_fmha_d80_fp8_Kernel_Module_t> CuteDslFMHARunner::sViT_d80_fp8{};
detail::LazyKernelModule<vit_fmha_d96_fp8_Kernel_Module_t> CuteDslFMHARunner::sViT_d96_fp8{};
detail::LazyKernelModule<vit_fmha_d128_fp8_Kernel_Module_t> CuteDslFMHARunner::sViT_d128_fp8{};

bool CuteDslFMHARunner::canImplement(int32_t headSize, int32_t smVersion)
{
    bool const supportedHeadSize = headSize == 64 || headSize == 128 || headSize == 256 || headSize == 512;
    return isSupportedBlackwellFmha(smVersion) && supportedHeadSize;
}

bool CuteDslFMHARunner::canImplementViT(int32_t headSize, int32_t smVersion)
{
    return isSupportedBlackwellFmha(smVersion)
        && (headSize == 64 || headSize == 72 || headSize == 80 || headSize == 96 || headSize == 128);
}

bool CuteDslFMHARunner::preflightLlm(
    cudaStream_t stream, int32_t slidingWindowSize, bool fp8Input, float skipSoftmaxThresholdLog2)
{
    bool const useSlidingWindow = slidingWindowSize < INT_MAX;
    bool const enableSkipSoftmax = std::isfinite(skipSoftmaxThresholdLog2) && skipSoftmaxThresholdLog2 < 0.0F;

    if (enableSkipSoftmax)
    {
        if (fp8Input || useSlidingWindow)
        {
            LOG_ERROR("CuTe DSL LLM FMHA: skip-softmax variant is FP16 causal only (fp8Input=%s, sw=%s)",
                fp8Input ? "true" : "false", useSlidingWindow ? "true" : "false");
            return false;
        }
        switch (mHeadDim)
        {
        case 64:
            return preflightVariant<fmha_d64_skipsoftmax_Kernel_Module_Load, fmha_d64_skipsoftmax_Kernel_Module_Unload>(
                sLLM_d64_skipsoftmax, "fmha_d64_skipsoftmax", stream);
        case 128:
            return preflightVariant<fmha_d128_skipsoftmax_Kernel_Module_Load,
                fmha_d128_skipsoftmax_Kernel_Module_Unload>(sLLM_d128_skipsoftmax, "fmha_d128_skipsoftmax", stream);
        default: LOG_ERROR("CuTe DSL LLM FMHA: unsupported head_dim=%d", mHeadDim); return false;
        }
    }

    if (fp8Input)
    {
        switch (mHeadDim)
        {
        case 64:
            return useSlidingWindow
                ? preflightVariant<fmha_d64_sw_fp8_Kernel_Module_Load, fmha_d64_sw_fp8_Kernel_Module_Unload>(
                      sLLM_d64_sw_fp8, "fmha_d64_sw_fp8", stream)
                : preflightVariant<fmha_d64_fp8_Kernel_Module_Load, fmha_d64_fp8_Kernel_Module_Unload>(
                      sLLM_d64_fp8, "fmha_d64_fp8", stream);
        case 128:
            return useSlidingWindow
                ? preflightVariant<fmha_d128_sw_fp8_Kernel_Module_Load, fmha_d128_sw_fp8_Kernel_Module_Unload>(
                      sLLM_d128_sw_fp8, "fmha_d128_sw_fp8", stream)
                : preflightVariant<fmha_d128_fp8_Kernel_Module_Load, fmha_d128_fp8_Kernel_Module_Unload>(
                      sLLM_d128_fp8, "fmha_d128_fp8", stream);
        case 256:
            return useSlidingWindow
                ? preflightVariant<fmha_d256_sw_fp8_Kernel_Module_Load, fmha_d256_sw_fp8_Kernel_Module_Unload>(
                      sLLM_d256_sw_fp8, "fmha_d256_sw_fp8", stream)
                : preflightVariant<fmha_d256_fp8_Kernel_Module_Load, fmha_d256_fp8_Kernel_Module_Unload>(
                      sLLM_d256_fp8, "fmha_d256_fp8", stream);
        default: LOG_ERROR("CuTe DSL LLM FMHA: unsupported head_dim=%d", mHeadDim); return false;
        }
    }

    switch (mHeadDim)
    {
    case 64:
        return useSlidingWindow ? preflightVariant<fmha_d64_sw_Kernel_Module_Load, fmha_d64_sw_Kernel_Module_Unload>(
                                      sLLM_d64_sw, "fmha_d64_sw", stream)
                                : preflightVariant<fmha_d64_Kernel_Module_Load, fmha_d64_Kernel_Module_Unload>(
                                      sLLM_d64, "fmha_d64", stream);
    case 128:
        return useSlidingWindow ? preflightVariant<fmha_d128_sw_Kernel_Module_Load, fmha_d128_sw_Kernel_Module_Unload>(
                                      sLLM_d128_sw, "fmha_d128_sw", stream)
                                : preflightVariant<fmha_d128_Kernel_Module_Load, fmha_d128_Kernel_Module_Unload>(
                                      sLLM_d128, "fmha_d128", stream);
    case 256:
        return useSlidingWindow ? preflightVariant<fmha_d256_sw_Kernel_Module_Load, fmha_d256_sw_Kernel_Module_Unload>(
                                      sLLM_d256_sw, "fmha_d256_sw", stream)
                                : preflightVariant<fmha_d256_Kernel_Module_Load, fmha_d256_Kernel_Module_Unload>(
                                      sLLM_d256, "fmha_d256", stream);
    default: LOG_ERROR("CuTe DSL LLM FMHA: unsupported head_dim=%d", mHeadDim); return false;
    }
}

bool CuteDslFMHARunner::preflightPaged(cudaStream_t stream, int32_t slidingWindowSize, bool fp8Input, bool isCausal,
    float skipSoftmaxThresholdLog2, bool useBidirectional)
{
    bool const useSlidingWindow = slidingWindowSize < INT_MAX;

    if (useBidirectional)
    {
        if (mHeadDim != 512 || !isCausal || fp8Input)
        {
            LOG_ERROR("CuTe DSL paged BIDIRECTIONAL FMHA requires causal FP16 with head_dim=512.");
            return false;
        }
        return preflightVariant<fmha_d512_paged_bidirectional_Kernel_Module_Load,
            fmha_d512_paged_bidirectional_Kernel_Module_Unload>(
            sLLM_d512_paged_bidirectional, "fmha_d512_paged_bidirectional", stream);
    }

    if (!isCausal)
    {
        if (useSlidingWindow || (mHeadDim != 256 && mHeadDim != 512))
        {
            LOG_ERROR("CuTe DSL dense non-causal paged FMHA requires no sliding window and head_dim=256 or 512.");
            return false;
        }
        if (fp8Input)
        {
            return mHeadDim == 256 ? preflightVariant<fmha_d256_dense_paged_fp8_Kernel_Module_Load,
                                         fmha_d256_dense_paged_fp8_Kernel_Module_Unload>(
                                         sLLM_d256_dense_paged_fp8, "fmha_d256_dense_paged_fp8", stream)
                                   : preflightVariant<fmha_d512_dense_paged_fp8_Kernel_Module_Load,
                                         fmha_d512_dense_paged_fp8_Kernel_Module_Unload>(
                                         sLLM_d512_dense_paged_fp8, "fmha_d512_dense_paged_fp8", stream);
        }
        return mHeadDim == 256
            ? preflightVariant<fmha_d256_dense_paged_Kernel_Module_Load, fmha_d256_dense_paged_Kernel_Module_Unload>(
                  sLLM_d256_dense_paged, "fmha_d256_dense_paged", stream)
            : preflightVariant<fmha_d512_dense_paged_Kernel_Module_Load, fmha_d512_dense_paged_Kernel_Module_Unload>(
                  sLLM_d512_dense_paged, "fmha_d512_dense_paged", stream);
    }

    bool const enableSkipSoftmax
        = std::isfinite(skipSoftmaxThresholdLog2) && skipSoftmaxThresholdLog2 < 0.0F && !fp8Input && !useSlidingWindow;
    if (enableSkipSoftmax && (mHeadDim == 64 || mHeadDim == 128))
    {
        return mHeadDim == 64 ? preflightVariant<fmha_d64_skipsoftmax_paged_Kernel_Module_Load,
                                    fmha_d64_skipsoftmax_paged_Kernel_Module_Unload>(
                                    sLLM_d64_skipsoftmax_paged, "fmha_d64_skipsoftmax_paged", stream)
                              : preflightVariant<fmha_d128_skipsoftmax_paged_Kernel_Module_Load,
                                    fmha_d128_skipsoftmax_paged_Kernel_Module_Unload>(
                                    sLLM_d128_skipsoftmax_paged, "fmha_d128_skipsoftmax_paged", stream);
    }

    if (fp8Input)
    {
        switch (mHeadDim)
        {
        case 64:
            return useSlidingWindow
                ? preflightVariant<fmha_d64_sw_paged_fp8_Kernel_Module_Load,
                      fmha_d64_sw_paged_fp8_Kernel_Module_Unload>(
                      sLLM_d64_sw_paged_fp8, "fmha_d64_sw_paged_fp8", stream)
                : preflightVariant<fmha_d64_paged_fp8_Kernel_Module_Load, fmha_d64_paged_fp8_Kernel_Module_Unload>(
                      sLLM_d64_paged_fp8, "fmha_d64_paged_fp8", stream);
        case 128:
            return useSlidingWindow
                ? preflightVariant<fmha_d128_sw_paged_fp8_Kernel_Module_Load,
                      fmha_d128_sw_paged_fp8_Kernel_Module_Unload>(
                      sLLM_d128_sw_paged_fp8, "fmha_d128_sw_paged_fp8", stream)
                : preflightVariant<fmha_d128_paged_fp8_Kernel_Module_Load, fmha_d128_paged_fp8_Kernel_Module_Unload>(
                      sLLM_d128_paged_fp8, "fmha_d128_paged_fp8", stream);
        case 256:
            return useSlidingWindow
                ? preflightVariant<fmha_d256_sw_paged_fp8_Kernel_Module_Load,
                      fmha_d256_sw_paged_fp8_Kernel_Module_Unload>(
                      sLLM_d256_sw_paged_fp8, "fmha_d256_sw_paged_fp8", stream)
                : preflightVariant<fmha_d256_paged_fp8_Kernel_Module_Load, fmha_d256_paged_fp8_Kernel_Module_Unload>(
                      sLLM_d256_paged_fp8, "fmha_d256_paged_fp8", stream);
        case 512:
            return useSlidingWindow
                ? preflightVariant<fmha_d512_sw_paged_fp8_Kernel_Module_Load,
                      fmha_d512_sw_paged_fp8_Kernel_Module_Unload>(
                      sLLM_d512_sw_paged_fp8, "fmha_d512_sw_paged_fp8", stream)
                : preflightVariant<fmha_d512_paged_fp8_Kernel_Module_Load, fmha_d512_paged_fp8_Kernel_Module_Unload>(
                      sLLM_d512_paged_fp8, "fmha_d512_paged_fp8", stream);
        default: LOG_ERROR("CuTe DSL paged LLM FMHA: unsupported head_dim=%d", mHeadDim); return false;
        }
    }

    switch (mHeadDim)
    {
    case 64:
        return useSlidingWindow
            ? preflightVariant<fmha_d64_sw_paged_Kernel_Module_Load, fmha_d64_sw_paged_Kernel_Module_Unload>(
                  sLLM_d64_sw_paged, "fmha_d64_sw_paged", stream)
            : preflightVariant<fmha_d64_paged_Kernel_Module_Load, fmha_d64_paged_Kernel_Module_Unload>(
                  sLLM_d64_paged, "fmha_d64_paged", stream);
    case 128:
        return useSlidingWindow
            ? preflightVariant<fmha_d128_sw_paged_Kernel_Module_Load, fmha_d128_sw_paged_Kernel_Module_Unload>(
                  sLLM_d128_sw_paged, "fmha_d128_sw_paged", stream)
            : preflightVariant<fmha_d128_paged_Kernel_Module_Load, fmha_d128_paged_Kernel_Module_Unload>(
                  sLLM_d128_paged, "fmha_d128_paged", stream);
    case 256:
        return useSlidingWindow
            ? preflightVariant<fmha_d256_sw_paged_Kernel_Module_Load, fmha_d256_sw_paged_Kernel_Module_Unload>(
                  sLLM_d256_sw_paged, "fmha_d256_sw_paged", stream)
            : preflightVariant<fmha_d256_paged_Kernel_Module_Load, fmha_d256_paged_Kernel_Module_Unload>(
                  sLLM_d256_paged, "fmha_d256_paged", stream);
    case 512:
        return useSlidingWindow
            ? preflightVariant<fmha_d512_sw_paged_Kernel_Module_Load, fmha_d512_sw_paged_Kernel_Module_Unload>(
                  sLLM_d512_sw_paged, "fmha_d512_sw_paged", stream)
            : preflightVariant<fmha_d512_paged_Kernel_Module_Load, fmha_d512_paged_Kernel_Module_Unload>(
                  sLLM_d512_paged, "fmha_d512_paged", stream);
    default: LOG_ERROR("CuTe DSL paged LLM FMHA: unsupported head_dim=%d", mHeadDim); return false;
    }
}

bool CuteDslFMHARunner::preflightViT(cudaStream_t stream)
{
    switch (mHeadDim)
    {
    case 64:
        return preflightVariant<vit_fmha_d64_Kernel_Module_Load, vit_fmha_d64_Kernel_Module_Unload>(
            sViT_d64, "vit_fmha_d64", stream);
    case 72:
        return preflightVariant<vit_fmha_d72_Kernel_Module_Load, vit_fmha_d72_Kernel_Module_Unload>(
            sViT_d72, "vit_fmha_d72", stream);
    case 80:
        return preflightVariant<vit_fmha_d80_Kernel_Module_Load, vit_fmha_d80_Kernel_Module_Unload>(
            sViT_d80, "vit_fmha_d80", stream);
    case 96:
        return preflightVariant<vit_fmha_d96_Kernel_Module_Load, vit_fmha_d96_Kernel_Module_Unload>(
            sViT_d96, "vit_fmha_d96", stream);
    case 128:
        return preflightVariant<vit_fmha_d128_Kernel_Module_Load, vit_fmha_d128_Kernel_Module_Unload>(
            sViT_d128, "vit_fmha_d128", stream);
    default: LOG_ERROR("CuTe DSL ViT FMHA: unsupported head_dim=%d", mHeadDim); return false;
    }
}

// =====================================================================
// Constructors
// =====================================================================

CuteDslFMHARunner::CuteDslFMHARunner(
    int32_t numQHeads, int32_t numKVHeads, int32_t headDim, int32_t batchSize, int32_t seqLenQ, int32_t kvCacheCapacity)
    : mBatchSize(batchSize)
    , mSeqLenQ(seqLenQ)
    , mKVCacheCapacity(kvCacheCapacity)
    , mNumHeadsQ(numQHeads)
    , mNumHeadsK(numKVHeads)
    , mHeadDim(headDim)
{
}

namespace
{

using cutedsl::makeCuSeqLenTensor;
using cutedsl::makePackedTensor;
using cutedsl::makeStridedTensor;
using cutedsl::WrapperArgT;
using cutedsl::WrapperArity;

//! Launch a dense (combined KV cache) LLM FMHA variant. The exported signature is
//!   (module, q_tensor, kv_cache, o_tensor, cum_seqlen_k, window_size_left, attention_scale,
//!    scale_q, scale_k, scale_v, inv_scale_o, sm_count, stream)
//! The skip-softmax (BLASST) variants carry one extra trailing runtime float,
//! skip_softmax_threshold_log2, placed AFTER stream (mirrors the generated
//! fmha_d{64,128}_skipsoftmax.h signatures) — selected here by wrapper arity at compile time.
//! @tparam cuteDslKernelWrapper Generated CuTe DSL kernel wrapper function. Its signature supplies the module
//! and tensor descriptor types at compile time.
template <auto cuteDslKernelWrapper, auto moduleLoader, auto moduleUnloader>
int32_t callLlmFmha(detail::LazyKernelModule<WrapperArgT<0, decltype(cuteDslKernelWrapper)>>& state,
    char const* moduleName, LlmFmhaParams const& params)
{
    constexpr size_t kArity = WrapperArity<decltype(cuteDslKernelWrapper)>::value;
    static_assert(kArity == 13 || kArity == 14,
        "callLlmFmha: not a dense LLM FMHA wrapper (module, q_tensor, kv_cache, o_tensor, cum_seqlen_k, "
        "window_size_left, attention_scale, scale_q, scale_k, scale_v, inv_scale_o, sm_count, stream "
        "[, skip_softmax_threshold_log2]).");

    if (!detail::ensureModuleLoaded<moduleLoader, moduleUnloader>(state, moduleName, params.stream))
    {
        return -1;
    }
    auto& module = state.module;

    auto qTensor = makePackedTensor<WrapperArgT<1, decltype(cuteDslKernelWrapper)>>(
        params.qPtr, {params.batchSize, params.seqLenQ, params.numQHeads, params.headDim});
    auto kvTensor = makePackedTensor<WrapperArgT<2, decltype(cuteDslKernelWrapper)>>(
        params.kvPtr, {params.batchSize, 2, params.numKVHeads, params.kvCacheCapacity, params.headDim});
    auto oTensor = makePackedTensor<WrapperArgT<3, decltype(cuteDslKernelWrapper)>>(
        params.oPtr, {params.batchSize, params.seqLenQ, params.numQHeads, params.headDim});
    auto cumSeqlenK
        = makeCuSeqLenTensor<WrapperArgT<4, decltype(cuteDslKernelWrapper)>>(params.cuKVSeqLens, params.batchSize + 1);

    if constexpr (kArity == 14)
    {
        return cuteDslKernelWrapper(&module, &qTensor, &kvTensor, &oTensor, &cumSeqlenK, params.windowSizeLeft,
            params.attentionScale, params.scaleQ, params.scaleK, params.scaleV, params.invScaleO,
            getDeviceMultiProcessorCount(), params.stream, params.skipSoftmaxThresholdLog2);
    }
    else
    {
        return cuteDslKernelWrapper(&module, &qTensor, &kvTensor, &oTensor, &cumSeqlenK, params.windowSizeLeft,
            params.attentionScale, params.scaleQ, params.scaleK, params.scaleV, params.invScaleO,
            getDeviceMultiProcessorCount(), params.stream);
    }
}

//! Launch a paged LLM FMHA variant. The exported signature matches callLlmFmha() except that
//! kv_cache is replaced by the (kv_cache_pool, kv_cache_page_list) pair. The bidirectional-mask
//! specialization appends block_begin and block_end descriptors after cum_seqlen_k. The skip-softmax
//! (BLASST) variants carry one extra trailing runtime float, skip_softmax_threshold_log2, after stream.
//! @tparam cuteDslKernelWrapper Generated CuTe DSL kernel wrapper function. Its signature supplies the module
//! and tensor descriptor types at compile time.
template <auto cuteDslKernelWrapper, auto moduleLoader, auto moduleUnloader>
int32_t callLlmFmhaPaged(detail::LazyKernelModule<WrapperArgT<0, decltype(cuteDslKernelWrapper)>>& state,
    char const* moduleName, LlmFmhaPagedParams const& params)
{
    constexpr std::size_t kWrapperArity = WrapperArity<decltype(cuteDslKernelWrapper)>::value;
    static_assert(kWrapperArity == 14 || kWrapperArity == 15 || kWrapperArity == 16,
        "callLlmFmhaPaged: not a paged LLM FMHA wrapper (module, q_tensor, kv_cache_pool, kv_cache_page_list, "
        "o_tensor, cum_seqlen_k, [block_begin, block_end,] window_size_left, attention_scale, scale_q, scale_k, "
        "scale_v, inv_scale_o, sm_count, stream [, skip_softmax_threshold_log2]).");

    if (!detail::ensureModuleLoaded<moduleLoader, moduleUnloader>(state, moduleName, params.stream))
    {
        return -1;
    }
    auto& module = state.module;

    // WrapperArgT supplies the generated ABI descriptor type. All runtime pointers and shapes come from params.
    auto qTensor = makePackedTensor<WrapperArgT<1, decltype(cuteDslKernelWrapper)>>(
        params.qPtr, {params.batchSize, params.seqLenQ, params.numQHeads, params.headDim});

    // The physical paged KV pool layout is fixed to NHD [numPages, tokensPerPage, H_kv, D]. CuTe DSL still
    // receives logical shape [numPages, H_kv, tokensPerPage, D], and the permutation lives in these strides,
    // so this is the one descriptor here that is not packed row-major.
    auto kvPoolTensor = makeStridedTensor<WrapperArgT<2, decltype(cuteDslKernelWrapper)>>(params.pagedKVPoolPtr,
        {params.numPages, params.numKVHeads, params.tokensPerPage, params.headDim},
        {static_cast<int64_t>(params.numKVHeads) * params.tokensPerPage * params.headDim,
            static_cast<int64_t>(params.headDim), static_cast<int64_t>(params.numKVHeads) * params.headDim});

    auto pageListTensor = makePackedTensor<WrapperArgT<3, decltype(cuteDslKernelWrapper)>>(
        params.kvCachePageList, {params.batchSize, 2, params.maxPagesPerSeq});
    auto oTensor = makePackedTensor<WrapperArgT<4, decltype(cuteDslKernelWrapper)>>(
        params.oPtr, {params.batchSize, params.seqLenQ, params.numQHeads, params.headDim});
    auto cumSeqlenK
        = makeCuSeqLenTensor<WrapperArgT<5, decltype(cuteDslKernelWrapper)>>(params.cuKVSeqLens, params.batchSize + 1);

    auto const tensorArgs = std::make_tuple(&module, &qTensor, &kvPoolTensor, &pageListTensor, &oTensor, &cumSeqlenK);
    auto const runtimeArgs = std::make_tuple(params.windowSizeLeft, params.attentionScale, params.scaleQ, params.scaleK,
        params.scaleV, params.invScaleO, getDeviceMultiProcessorCount(), params.stream);

    if constexpr (kWrapperArity == 16)
    {
        check::check(params.bidirectionalBlockBegin != nullptr && params.bidirectionalBlockEnd != nullptr,
            "Bidirectional paged LLM FMHA requires bidirectionalBlockBegin and bidirectionalBlockEnd.");
        auto bidirectionalBlockBeginTensor = makePackedTensor<WrapperArgT<6, decltype(cuteDslKernelWrapper)>>(
            params.bidirectionalBlockBegin, {params.batchSize, params.seqLenQ});
        auto bidirectionalBlockEndTensor = makePackedTensor<WrapperArgT<7, decltype(cuteDslKernelWrapper)>>(
            params.bidirectionalBlockEnd, {params.batchSize, params.seqLenQ});
        auto const args = std::tuple_cat(
            tensorArgs, std::make_tuple(&bidirectionalBlockBeginTensor, &bidirectionalBlockEndTensor), runtimeArgs);
        return std::apply(cuteDslKernelWrapper, args);
    }
    else if constexpr (kWrapperArity == 15)
    {
        check::check(params.bidirectionalBlockBegin == nullptr && params.bidirectionalBlockEnd == nullptr,
            "Standard paged LLM FMHA does not accept bidirectional block ranges.");
        auto const args = std::tuple_cat(tensorArgs, runtimeArgs, std::make_tuple(params.skipSoftmaxThresholdLog2));
        return std::apply(cuteDslKernelWrapper, args);
    }
    else
    {
        check::check(params.bidirectionalBlockBegin == nullptr && params.bidirectionalBlockEnd == nullptr,
            "Standard paged LLM FMHA does not accept bidirectional block ranges.");
        auto const args = std::tuple_cat(tensorArgs, runtimeArgs);
        return std::apply(cuteDslKernelWrapper, args);
    }
}

//! Launch a ViT FMHA variant over packed varlen [total_S, H, D] Q/K/V.
//! @tparam cuteDslKernelWrapper Generated CuTe DSL kernel wrapper function. Its signature supplies the module
//! and tensor descriptor types at compile time.
template <auto cuteDslKernelWrapper, auto moduleLoader, auto moduleUnloader>
int32_t callVitFmha(detail::LazyKernelModule<WrapperArgT<0, decltype(cuteDslKernelWrapper)>>& state,
    char const* moduleName, VitFmhaParams const& params)
{
    static_assert(WrapperArity<decltype(cuteDslKernelWrapper)>::value == 12,
        "callVitFmha: not a ViT FMHA wrapper (module, q_tensor, k_tensor, v_tensor, o_tensor, cu_seqlens, "
        "max_seqlen, scale_softmax_log2, scale_softmax, scale_output, sm_count, stream).");

    if (!detail::ensureModuleLoaded<moduleLoader, moduleUnloader>(state, moduleName, params.stream))
    {
        return -1;
    }
    auto& module = state.module;

    int32_t const shape[] = {params.totalSeqLen, params.numHeads, params.headDim};
    auto qTensor = makePackedTensor<WrapperArgT<1, decltype(cuteDslKernelWrapper)>>(params.qPtr, shape);
    auto kTensor = makePackedTensor<WrapperArgT<2, decltype(cuteDslKernelWrapper)>>(params.kPtr, shape);
    auto vTensor = makePackedTensor<WrapperArgT<3, decltype(cuteDslKernelWrapper)>>(params.vPtr, shape);
    auto oTensor = makePackedTensor<WrapperArgT<4, decltype(cuteDslKernelWrapper)>>(params.oPtr, shape);
    auto cuSeqlensTensor
        = makeCuSeqLenTensor<WrapperArgT<5, decltype(cuteDslKernelWrapper)>>(params.cuSeqLens, params.batchSize + 1);

    return cuteDslKernelWrapper(&module, &qTensor, &kTensor, &vTensor, &oTensor, &cuSeqlensTensor, params.maxSeqLen,
        params.scaleSoftmaxLog2, params.attentionScale, params.scaleOutput, getDeviceMultiProcessorCount(),
        params.stream);
}

} // namespace

// =====================================================================
// LLM run: batched Q + combined KV cache (FP16 / FP8→FP16 / FP8→FP8)
// =====================================================================

bool CuteDslFMHARunner::run(void const* qPtr, void const* kvPtr, void* oPtr, int32_t const* cuKVSeqLens,
    cudaStream_t stream, float attentionScale, int32_t slidingWindowSize, bool fp8Input, float qScale, float kScale,
    float vScale, float skipSoftmaxThresholdLog2)
{
    validateAttentionScale(attentionScale);

    int32_t const headDim = mHeadDim;
    bool const useSlidingWindow = (slidingWindowSize < INT_MAX);
    int32_t constexpr kNoLimit = 1 << 30;

    // Skip-softmax threshold sentinel: a finite negative log2(lambda) enables the
    // skip variant; 0.0 (log2 of the degenerate lambda = 1) means disabled. Any
    // other value is a caller bug — warn and dispatch dense.
    bool const enableSkipSoftmax = std::isfinite(skipSoftmaxThresholdLog2) && skipSoftmaxThresholdLog2 < 0.0F;
    if (!enableSkipSoftmax && skipSoftmaxThresholdLog2 != 0.0F)
    {
        LOG_WARNING(
            "CuTe DSL LLM FMHA: invalid skipSoftmaxThresholdLog2=%f (want finite < 0, or 0 to disable); "
            "dispatching dense kernel instead.",
            skipSoftmaxThresholdLog2);
    }

    LlmFmhaParams params{};
    params.qPtr = qPtr;
    params.kvPtr = kvPtr;
    params.oPtr = oPtr;
    params.cuKVSeqLens = cuKVSeqLens;
    params.batchSize = mBatchSize;
    params.seqLenQ = mSeqLenQ;
    params.numQHeads = mNumHeadsQ;
    params.numKVHeads = mNumHeadsK;
    params.headDim = headDim;
    params.kvCacheCapacity = mKVCacheCapacity;
    params.windowSizeLeft = useSlidingWindow ? slidingWindowSize : kNoLimit;
    params.attentionScale = attentionScale;
    params.scaleQ = qScale;
    params.scaleK = kScale;
    params.scaleV = vScale;
    params.invScaleO = 1.0F;
    params.skipSoftmaxThresholdLog2 = skipSoftmaxThresholdLog2;
    params.stream = stream;

    int32_t ret = -1;

    if (enableSkipSoftmax)
    {
        if (fp8Input || useSlidingWindow)
        {
            LOG_ERROR("CuTe DSL LLM FMHA: skip-softmax variant is FP16 causal only (fp8Input=%s, sw=%s)",
                fp8Input ? "true" : "false", useSlidingWindow ? "true" : "false");
            return false;
        }
        switch (headDim)
        {
        case 64:
            ret = callLlmFmha<cute_dsl_fmha_d64_skipsoftmax_wrapper, fmha_d64_skipsoftmax_Kernel_Module_Load,
                fmha_d64_skipsoftmax_Kernel_Module_Unload>(sLLM_d64_skipsoftmax, "fmha_d64_skipsoftmax", params);
            break;
        case 128:
            ret = callLlmFmha<cute_dsl_fmha_d128_skipsoftmax_wrapper, fmha_d128_skipsoftmax_Kernel_Module_Load,
                fmha_d128_skipsoftmax_Kernel_Module_Unload>(sLLM_d128_skipsoftmax, "fmha_d128_skipsoftmax", params);
            break;
        default: LOG_ERROR("CuTe DSL LLM FMHA: unsupported head_dim=%d", headDim); return false;
        }
    }
    else if (fp8Input)
    {
        switch (headDim)
        {
        case 64:
            ret = useSlidingWindow
                ? callLlmFmha<cute_dsl_fmha_d64_sw_fp8_wrapper, fmha_d64_sw_fp8_Kernel_Module_Load,
                      fmha_d64_sw_fp8_Kernel_Module_Unload>(sLLM_d64_sw_fp8, "fmha_d64_sw_fp8", params)
                : callLlmFmha<cute_dsl_fmha_d64_fp8_wrapper, fmha_d64_fp8_Kernel_Module_Load,
                      fmha_d64_fp8_Kernel_Module_Unload>(sLLM_d64_fp8, "fmha_d64_fp8", params);
            break;
        case 128:
            ret = useSlidingWindow
                ? callLlmFmha<cute_dsl_fmha_d128_sw_fp8_wrapper, fmha_d128_sw_fp8_Kernel_Module_Load,
                      fmha_d128_sw_fp8_Kernel_Module_Unload>(sLLM_d128_sw_fp8, "fmha_d128_sw_fp8", params)
                : callLlmFmha<cute_dsl_fmha_d128_fp8_wrapper, fmha_d128_fp8_Kernel_Module_Load,
                      fmha_d128_fp8_Kernel_Module_Unload>(sLLM_d128_fp8, "fmha_d128_fp8", params);
            break;
        case 256:
            ret = useSlidingWindow
                ? callLlmFmha<cute_dsl_fmha_d256_sw_fp8_wrapper, fmha_d256_sw_fp8_Kernel_Module_Load,
                      fmha_d256_sw_fp8_Kernel_Module_Unload>(sLLM_d256_sw_fp8, "fmha_d256_sw_fp8", params)
                : callLlmFmha<cute_dsl_fmha_d256_fp8_wrapper, fmha_d256_fp8_Kernel_Module_Load,
                      fmha_d256_fp8_Kernel_Module_Unload>(sLLM_d256_fp8, "fmha_d256_fp8", params);
            break;
        default: LOG_ERROR("CuTe DSL LLM FMHA: unsupported head_dim=%d", headDim); return false;
        }
    }
    else
    {
        switch (headDim)
        {
        case 64:
            ret = useSlidingWindow
                ? callLlmFmha<cute_dsl_fmha_d64_sw_wrapper, fmha_d64_sw_Kernel_Module_Load,
                      fmha_d64_sw_Kernel_Module_Unload>(sLLM_d64_sw, "fmha_d64_sw", params)
                : callLlmFmha<cute_dsl_fmha_d64_wrapper, fmha_d64_Kernel_Module_Load, fmha_d64_Kernel_Module_Unload>(
                      sLLM_d64, "fmha_d64", params);
            break;
        case 128:
            ret = useSlidingWindow
                ? callLlmFmha<cute_dsl_fmha_d128_sw_wrapper, fmha_d128_sw_Kernel_Module_Load,
                      fmha_d128_sw_Kernel_Module_Unload>(sLLM_d128_sw, "fmha_d128_sw", params)
                : callLlmFmha<cute_dsl_fmha_d128_wrapper, fmha_d128_Kernel_Module_Load, fmha_d128_Kernel_Module_Unload>(
                      sLLM_d128, "fmha_d128", params);
            break;
        case 256:
            ret = useSlidingWindow
                ? callLlmFmha<cute_dsl_fmha_d256_sw_wrapper, fmha_d256_sw_Kernel_Module_Load,
                      fmha_d256_sw_Kernel_Module_Unload>(sLLM_d256_sw, "fmha_d256_sw", params)
                : callLlmFmha<cute_dsl_fmha_d256_wrapper, fmha_d256_Kernel_Module_Load, fmha_d256_Kernel_Module_Unload>(
                      sLLM_d256, "fmha_d256", params);
            break;
        default: LOG_ERROR("CuTe DSL LLM FMHA: unsupported head_dim=%d", headDim); return false;
        }
    }

    if (ret != 0)
    {
        LOG_ERROR("CuTe DSL LLM FMHA kernel (d=%d, sw=%s, fp8in=%s) failed with error code: %d", headDim,
            useSlidingWindow ? "true" : "false", fp8Input ? "true" : "false", ret);
    }
    return ret == 0;
}

bool CuteDslFMHARunner::runPaged(void const* qPtr, void const* pagedKVPoolPtr, int32_t const* kvCachePageList,
    void* oPtr, int32_t const* cuKVSeqLens, int32_t numPages, int32_t maxPagesPerSeq, int32_t tokensPerPage,
    nvinfer1::DataType kvDataType, cudaStream_t stream, float attentionScale, int32_t slidingWindowSize, bool fp8Input,
    float qScale, float kScale, float vScale, bool isCausal, float skipSoftmaxThresholdLog2,
    int32_t const* bidirectionalBlockBegin, int32_t const* bidirectionalBlockEnd)
{
    check::check(qPtr != nullptr, "CuTe DSL paged FMHA qPtr must not be null.");
    check::check(pagedKVPoolPtr != nullptr, "CuTe DSL paged FMHA KV pool must not be null.");
    check::check(kvCachePageList != nullptr, "CuTe DSL paged FMHA page list must not be null.");
    check::check(oPtr != nullptr, "CuTe DSL paged FMHA oPtr must not be null.");
    check::check(cuKVSeqLens != nullptr, "CuTe DSL paged FMHA cuKVSeqLens must not be null.");
    check::check(numPages > 0 && maxPagesPerSeq > 0 && tokensPerPage > 0,
        "CuTe DSL paged FMHA requires positive numPages/maxPagesPerSeq/tokensPerPage.");
    // Direct paged CuTe DSL keeps the existing TMA load pipeline: each logical K/V tile maps to one physical page.
    // These AOT variants use tile_N=128, so smaller pages would require stitching one tile from multiple pages.
    check::check(tokensPerPage == 128,
        "CuTe DSL direct paged FMHA requires tokensPerPage == 128 because one K/V TMA tile maps to one page.");
    check::check(mKVCacheCapacity == maxPagesPerSeq * tokensPerPage,
        "CuTe DSL paged FMHA runner capacity must equal maxPagesPerSeq * tokensPerPage.");
    check::check(kvDataType == nvinfer1::DataType::kHALF || kvDataType == nvinfer1::DataType::kFP8,
        "CuTe DSL paged FMHA supports FP16 or FP8 KV cache.");
    check::check((kvDataType == nvinfer1::DataType::kFP8) == fp8Input,
        "CuTe DSL paged FMHA requires fp8Input to match the paged KV cache dtype.");
    bool const useBidirectional = bidirectionalBlockBegin != nullptr || bidirectionalBlockEnd != nullptr;
    check::check((bidirectionalBlockBegin == nullptr) == (bidirectionalBlockEnd == nullptr),
        "CuTe DSL paged FMHA requires bidirectionalBlockBegin and bidirectionalBlockEnd to be both set or both null.");
    bool const useSlidingWindow = (slidingWindowSize < INT_MAX);
    check::check(!useBidirectional || isCausal, "CuTe DSL paged BIDIRECTIONAL FMHA requires causal attention.");
    check::check(isCausal || slidingWindowSize == INT_MAX,
        "CuTe DSL dense non-causal paged FMHA does not support sliding-window masking.");
    check::check(isCausal || mHeadDim == 256 || mHeadDim == 512,
        "CuTe DSL dense non-causal paged FMHA currently supports head_dim=256 or 512 only.");
    if (useBidirectional)
    {
        check::check(kvDataType == nvinfer1::DataType::kHALF && !fp8Input,
            "CuTe DSL paged BIDIRECTIONAL FMHA supports only FP16.");
    }

    int32_t const headDim = mHeadDim;
    int32_t constexpr kNoLimit = 1 << 30;

    // Skip-softmax threshold sentinel: a finite negative log2(lambda) enables the
    // skip variant; 0.0 means disabled.
    bool const thresholdValid = std::isfinite(skipSoftmaxThresholdLog2) && skipSoftmaxThresholdLog2 < 0.0F;
    if (!thresholdValid && skipSoftmaxThresholdLog2 != 0.0F)
    {
        LOG_WARNING(
            "CuTe DSL paged LLM FMHA: invalid skipSoftmaxThresholdLog2=%f (want finite < 0, or 0 to disable); "
            "dispatching dense paged kernel instead.",
            skipSoftmaxThresholdLog2);
    }
    bool const enableSkipSoftmax = thresholdValid && !fp8Input && isCausal && !useSlidingWindow;

    LlmFmhaPagedParams params{};
    params.qPtr = qPtr;
    params.pagedKVPoolPtr = pagedKVPoolPtr;
    params.kvCachePageList = kvCachePageList;
    params.oPtr = oPtr;
    params.cuKVSeqLens = cuKVSeqLens;
    params.bidirectionalBlockBegin = bidirectionalBlockBegin;
    params.bidirectionalBlockEnd = bidirectionalBlockEnd;
    params.batchSize = mBatchSize;
    params.seqLenQ = mSeqLenQ;
    params.numQHeads = mNumHeadsQ;
    params.numKVHeads = mNumHeadsK;
    params.headDim = headDim;
    params.numPages = numPages;
    params.maxPagesPerSeq = maxPagesPerSeq;
    params.tokensPerPage = tokensPerPage;
    params.windowSizeLeft = useSlidingWindow ? slidingWindowSize : kNoLimit;
    params.attentionScale = attentionScale;
    params.scaleQ = qScale;
    params.scaleK = kScale;
    params.scaleV = vScale;
    params.invScaleO = 1.0F;
    params.skipSoftmaxThresholdLog2 = skipSoftmaxThresholdLog2;
    params.stream = stream;

    int32_t ret = -1;

    if (useBidirectional)
    {
        check::check(headDim == 512, "CuTe DSL paged BIDIRECTIONAL FMHA dispatch requires head dimension 512.");
        ret = callLlmFmhaPaged<cute_dsl_fmha_d512_paged_bidirectional_wrapper,
            fmha_d512_paged_bidirectional_Kernel_Module_Load, fmha_d512_paged_bidirectional_Kernel_Module_Unload>(
            sLLM_d512_paged_bidirectional, "fmha_d512_paged_bidirectional", params);
    }
    else if (!isCausal)
    {
        // The check::check above already narrowed the dense non-causal path to head_dim 256 or 512.
        if (fp8Input)
        {
            ret = headDim == 256
                ? callLlmFmhaPaged<cute_dsl_fmha_d256_dense_paged_fp8_wrapper,
                      fmha_d256_dense_paged_fp8_Kernel_Module_Load, fmha_d256_dense_paged_fp8_Kernel_Module_Unload>(
                      sLLM_d256_dense_paged_fp8, "fmha_d256_dense_paged_fp8", params)
                : callLlmFmhaPaged<cute_dsl_fmha_d512_dense_paged_fp8_wrapper,
                      fmha_d512_dense_paged_fp8_Kernel_Module_Load, fmha_d512_dense_paged_fp8_Kernel_Module_Unload>(
                      sLLM_d512_dense_paged_fp8, "fmha_d512_dense_paged_fp8", params);
        }
        else
        {
            ret = headDim == 256
                ? callLlmFmhaPaged<cute_dsl_fmha_d256_dense_paged_wrapper, fmha_d256_dense_paged_Kernel_Module_Load,
                      fmha_d256_dense_paged_Kernel_Module_Unload>(
                      sLLM_d256_dense_paged, "fmha_d256_dense_paged", params)
                : callLlmFmhaPaged<cute_dsl_fmha_d512_dense_paged_wrapper, fmha_d512_dense_paged_Kernel_Module_Load,
                      fmha_d512_dense_paged_Kernel_Module_Unload>(
                      sLLM_d512_dense_paged, "fmha_d512_dense_paged", params);
        }
    }
    else if (enableSkipSoftmax && (headDim == 64 || headDim == 128))
    {
        // Skip-softmax paged prefill: causal, FP16, non-sliding, d64/d128 only.
        ret = headDim == 64
            ? callLlmFmhaPaged<cute_dsl_fmha_d64_skipsoftmax_paged_wrapper,
                  fmha_d64_skipsoftmax_paged_Kernel_Module_Load, fmha_d64_skipsoftmax_paged_Kernel_Module_Unload>(
                  sLLM_d64_skipsoftmax_paged, "fmha_d64_skipsoftmax_paged", params)
            : callLlmFmhaPaged<cute_dsl_fmha_d128_skipsoftmax_paged_wrapper,
                  fmha_d128_skipsoftmax_paged_Kernel_Module_Load, fmha_d128_skipsoftmax_paged_Kernel_Module_Unload>(
                  sLLM_d128_skipsoftmax_paged, "fmha_d128_skipsoftmax_paged", params);
    }
    else if (fp8Input)
    {
        switch (headDim)
        {
        case 64:
            ret = useSlidingWindow
                ? callLlmFmhaPaged<cute_dsl_fmha_d64_sw_paged_fp8_wrapper, fmha_d64_sw_paged_fp8_Kernel_Module_Load,
                      fmha_d64_sw_paged_fp8_Kernel_Module_Unload>(
                      sLLM_d64_sw_paged_fp8, "fmha_d64_sw_paged_fp8", params)
                : callLlmFmhaPaged<cute_dsl_fmha_d64_paged_fp8_wrapper, fmha_d64_paged_fp8_Kernel_Module_Load,
                      fmha_d64_paged_fp8_Kernel_Module_Unload>(sLLM_d64_paged_fp8, "fmha_d64_paged_fp8", params);
            break;
        case 128:
            ret = useSlidingWindow
                ? callLlmFmhaPaged<cute_dsl_fmha_d128_sw_paged_fp8_wrapper, fmha_d128_sw_paged_fp8_Kernel_Module_Load,
                      fmha_d128_sw_paged_fp8_Kernel_Module_Unload>(
                      sLLM_d128_sw_paged_fp8, "fmha_d128_sw_paged_fp8", params)
                : callLlmFmhaPaged<cute_dsl_fmha_d128_paged_fp8_wrapper, fmha_d128_paged_fp8_Kernel_Module_Load,
                      fmha_d128_paged_fp8_Kernel_Module_Unload>(sLLM_d128_paged_fp8, "fmha_d128_paged_fp8", params);
            break;
        case 256:
            ret = useSlidingWindow
                ? callLlmFmhaPaged<cute_dsl_fmha_d256_sw_paged_fp8_wrapper, fmha_d256_sw_paged_fp8_Kernel_Module_Load,
                      fmha_d256_sw_paged_fp8_Kernel_Module_Unload>(
                      sLLM_d256_sw_paged_fp8, "fmha_d256_sw_paged_fp8", params)
                : callLlmFmhaPaged<cute_dsl_fmha_d256_paged_fp8_wrapper, fmha_d256_paged_fp8_Kernel_Module_Load,
                      fmha_d256_paged_fp8_Kernel_Module_Unload>(sLLM_d256_paged_fp8, "fmha_d256_paged_fp8", params);
            break;
        case 512:
            ret = useSlidingWindow
                ? callLlmFmhaPaged<cute_dsl_fmha_d512_sw_paged_fp8_wrapper, fmha_d512_sw_paged_fp8_Kernel_Module_Load,
                      fmha_d512_sw_paged_fp8_Kernel_Module_Unload>(
                      sLLM_d512_sw_paged_fp8, "fmha_d512_sw_paged_fp8", params)
                : callLlmFmhaPaged<cute_dsl_fmha_d512_paged_fp8_wrapper, fmha_d512_paged_fp8_Kernel_Module_Load,
                      fmha_d512_paged_fp8_Kernel_Module_Unload>(sLLM_d512_paged_fp8, "fmha_d512_paged_fp8", params);
            break;
        default: LOG_ERROR("CuTe DSL paged LLM FMHA: unsupported head_dim=%d", headDim); return false;
        }
    }
    else
    {
        switch (headDim)
        {
        case 64:
            ret = useSlidingWindow
                ? callLlmFmhaPaged<cute_dsl_fmha_d64_sw_paged_wrapper, fmha_d64_sw_paged_Kernel_Module_Load,
                      fmha_d64_sw_paged_Kernel_Module_Unload>(sLLM_d64_sw_paged, "fmha_d64_sw_paged", params)
                : callLlmFmhaPaged<cute_dsl_fmha_d64_paged_wrapper, fmha_d64_paged_Kernel_Module_Load,
                      fmha_d64_paged_Kernel_Module_Unload>(sLLM_d64_paged, "fmha_d64_paged", params);
            break;
        case 128:
            ret = useSlidingWindow
                ? callLlmFmhaPaged<cute_dsl_fmha_d128_sw_paged_wrapper, fmha_d128_sw_paged_Kernel_Module_Load,
                      fmha_d128_sw_paged_Kernel_Module_Unload>(sLLM_d128_sw_paged, "fmha_d128_sw_paged", params)
                : callLlmFmhaPaged<cute_dsl_fmha_d128_paged_wrapper, fmha_d128_paged_Kernel_Module_Load,
                      fmha_d128_paged_Kernel_Module_Unload>(sLLM_d128_paged, "fmha_d128_paged", params);
            break;
        case 256:
            ret = useSlidingWindow
                ? callLlmFmhaPaged<cute_dsl_fmha_d256_sw_paged_wrapper, fmha_d256_sw_paged_Kernel_Module_Load,
                      fmha_d256_sw_paged_Kernel_Module_Unload>(sLLM_d256_sw_paged, "fmha_d256_sw_paged", params)
                : callLlmFmhaPaged<cute_dsl_fmha_d256_paged_wrapper, fmha_d256_paged_Kernel_Module_Load,
                      fmha_d256_paged_Kernel_Module_Unload>(sLLM_d256_paged, "fmha_d256_paged", params);
            break;
        case 512:
            ret = useSlidingWindow
                ? callLlmFmhaPaged<cute_dsl_fmha_d512_sw_paged_wrapper, fmha_d512_sw_paged_Kernel_Module_Load,
                      fmha_d512_sw_paged_Kernel_Module_Unload>(sLLM_d512_sw_paged, "fmha_d512_sw_paged", params)
                : callLlmFmhaPaged<cute_dsl_fmha_d512_paged_wrapper, fmha_d512_paged_Kernel_Module_Load,
                      fmha_d512_paged_Kernel_Module_Unload>(sLLM_d512_paged, "fmha_d512_paged", params);
            break;
        default: LOG_ERROR("CuTe DSL paged LLM FMHA: unsupported head_dim=%d", headDim); return false;
        }
    }

    if (ret != 0)
    {
        LOG_ERROR(
            "CuTe DSL paged LLM FMHA kernel (d=%d, causal=%s, sw=%s, fp8in=%s, bidirectional=%s) failed with error "
            "code: %d",
            headDim, isCausal ? "true" : "false", useSlidingWindow ? "true" : "false", fp8Input ? "true" : "false",
            useBidirectional ? "true" : "false", ret);
    }
    return ret == 0;
}

// =====================================================================
// ViT run: packed varlen separate Q/K/V
// =====================================================================

bool CuteDslFMHARunner::run(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuSeqLens,
    int32_t totalSeqLen, int32_t maxSeqLen, int32_t batchSize, cudaStream_t stream, float attentionScale, bool fp8Input,
    float qScale, float kScale, float vScale)
{
    validateAttentionScale(attentionScale);

    int32_t const headDim = mHeadDim;

    // FP8 dequant scales fold into the softmax scale (q*k) and the output scale (v).
    float const softmaxScale = attentionScale * qScale * kScale;

    VitFmhaParams params{};
    params.qPtr = qPtr;
    params.kPtr = kPtr;
    params.vPtr = vPtr;
    params.oPtr = oPtr;
    params.cuSeqLens = cuSeqLens;
    params.totalSeqLen = totalSeqLen;
    params.numHeads = mNumHeadsQ;
    params.headDim = headDim;
    params.maxSeqLen = maxSeqLen;
    params.batchSize = batchSize;
    params.scaleSoftmaxLog2 = softmaxScale * static_cast<float>(M_LOG2E);
    params.attentionScale = softmaxScale;
    params.scaleOutput = vScale;
    params.stream = stream;

    int32_t ret = -1;

    if (fp8Input)
    {
        switch (headDim)
        {
        case 64:
            ret = callVitFmha<cute_dsl_vit_fmha_d64_fp8_wrapper, vit_fmha_d64_fp8_Kernel_Module_Load,
                vit_fmha_d64_fp8_Kernel_Module_Unload>(sViT_d64_fp8, "vit_fmha_d64_fp8", params);
            break;
        case 80:
            ret = callVitFmha<cute_dsl_vit_fmha_d80_fp8_wrapper, vit_fmha_d80_fp8_Kernel_Module_Load,
                vit_fmha_d80_fp8_Kernel_Module_Unload>(sViT_d80_fp8, "vit_fmha_d80_fp8", params);
            break;
        case 96:
            ret = callVitFmha<cute_dsl_vit_fmha_d96_fp8_wrapper, vit_fmha_d96_fp8_Kernel_Module_Load,
                vit_fmha_d96_fp8_Kernel_Module_Unload>(sViT_d96_fp8, "vit_fmha_d96_fp8", params);
            break;
        case 128:
            ret = callVitFmha<cute_dsl_vit_fmha_d128_fp8_wrapper, vit_fmha_d128_fp8_Kernel_Module_Load,
                vit_fmha_d128_fp8_Kernel_Module_Unload>(sViT_d128_fp8, "vit_fmha_d128_fp8", params);
            break;
        default:
            // d=72 has no direct kernel: SM100 TMA requires 16-byte-aligned innermost
            // GMEM strides, and 72 FP8 elements are 72 bytes. Callers zero-pad to
            // d=80 and pass the real softmax scale, then dispatch case 80 above.
            LOG_ERROR("CuTe DSL ViT FP8 FMHA: unsupported head_dim=%d (supported: 64, 80, 96, 128)", headDim);
            return false;
        }
    }
    else
    {
        switch (headDim)
        {
        case 64:
            ret = callVitFmha<cute_dsl_vit_fmha_d64_wrapper, vit_fmha_d64_Kernel_Module_Load,
                vit_fmha_d64_Kernel_Module_Unload>(sViT_d64, "vit_fmha_d64", params);
            break;
        case 72:
            ret = callVitFmha<cute_dsl_vit_fmha_d72_wrapper, vit_fmha_d72_Kernel_Module_Load,
                vit_fmha_d72_Kernel_Module_Unload>(sViT_d72, "vit_fmha_d72", params);
            break;
        case 80:
            ret = callVitFmha<cute_dsl_vit_fmha_d80_wrapper, vit_fmha_d80_Kernel_Module_Load,
                vit_fmha_d80_Kernel_Module_Unload>(sViT_d80, "vit_fmha_d80", params);
            break;
        case 96:
            ret = callVitFmha<cute_dsl_vit_fmha_d96_wrapper, vit_fmha_d96_Kernel_Module_Load,
                vit_fmha_d96_Kernel_Module_Unload>(sViT_d96, "vit_fmha_d96", params);
            break;
        case 128:
            ret = callVitFmha<cute_dsl_vit_fmha_d128_wrapper, vit_fmha_d128_Kernel_Module_Load,
                vit_fmha_d128_Kernel_Module_Unload>(sViT_d128, "vit_fmha_d128", params);
            break;
        default: LOG_ERROR("CuTe DSL ViT FMHA: unsupported head_dim=%d", headDim); return false;
        }
    }

    if (ret != 0)
    {
        LOG_ERROR("CuTe DSL ViT FMHA kernel (d=%d, fp8in=%s) failed with error code: %d", headDim,
            fp8Input ? "true" : "false", ret);
    }
    return ret == 0;
}

} // namespace trt_edgellm

#endif // CUTE_DSL_FMHA_BLACKWELL_ENABLED
