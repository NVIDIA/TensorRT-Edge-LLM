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

#include "cuteDslFMHAV2Runner.h"

#if defined(CUTE_DSL_FMHA_ENABLED)

#include "attentionScaleUtils.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "cuteDslFmhaParams.h"
#include "cuteDslTensorDescriptors.h"

#include <cmath>

namespace trt_edgellm
{

detail::LazyKernelModule<fmha_v2_d64_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d64{};
detail::LazyKernelModule<fmha_v2_d64_small_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d64Small{};
detail::LazyKernelModule<fmha_v2_d128_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d128{};
detail::LazyKernelModule<fmha_v2_d256_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d256{};
detail::LazyKernelModule<fmha_v2_d512_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d512{};
detail::LazyKernelModule<fmha_v2_d256_padding_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d256Padding{};
detail::LazyKernelModule<fmha_v2_d64_sw_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d64Sw{};
detail::LazyKernelModule<fmha_v2_d128_sw_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d128Sw{};
detail::LazyKernelModule<fmha_v2_d256_sw_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d256Sw{};
detail::LazyKernelModule<fmha_v2_d512_sw_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d512Sw{};
detail::LazyKernelModule<fmha_v2_d256_bidirectional_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d256Bidirectional{};
detail::LazyKernelModule<fmha_v2_d512_bidirectional_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d512Bidirectional{};
detail::LazyKernelModule<fmha_v2_d64_paged_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d64Paged{};
detail::LazyKernelModule<fmha_v2_d64_small_paged_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d64SmallPaged{};
detail::LazyKernelModule<fmha_v2_d128_paged_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d128Paged{};
detail::LazyKernelModule<fmha_v2_d256_paged_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d256Paged{};
detail::LazyKernelModule<fmha_v2_d512_paged_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d512Paged{};
detail::LazyKernelModule<fmha_v2_d64_sw_paged_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d64SwPaged{};
detail::LazyKernelModule<fmha_v2_d128_sw_paged_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d128SwPaged{};
detail::LazyKernelModule<fmha_v2_d256_sw_paged_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d256SwPaged{};
detail::LazyKernelModule<fmha_v2_d512_sw_paged_Kernel_Module_t> CuteDslFMHAV2Runner::sLLM_d512SwPaged{};

detail::LazyKernelModule<fmha_v2_vit_d64_Kernel_Module_t> CuteDslFMHAV2Runner::sViT_d64{};
detail::LazyKernelModule<fmha_v2_vit_d72_Kernel_Module_t> CuteDslFMHAV2Runner::sViT_d72{};
detail::LazyKernelModule<fmha_v2_vit_d80_Kernel_Module_t> CuteDslFMHAV2Runner::sViT_d80{};
detail::LazyKernelModule<fmha_v2_vit_d128_Kernel_Module_t> CuteDslFMHAV2Runner::sViT_d128{};

namespace
{

bool isFMHAV2SM(int32_t smVersion)
{
    return smVersion == 80 || smVersion == 86 || smVersion == 87 || smVersion == 89 || smVersion == 90
        || smVersion == 100 || smVersion == 101 || smVersion == 110 || smVersion == 120 || smVersion == 121;
}

bool isFMHAV2PagedLlmHeadSize(int32_t headSize)
{
    switch (headSize)
    {
    case 64:
    case 128:
    case 256:
    case 512: return true;
    default: return false;
    }
}

template <auto moduleLoader, auto moduleUnloader, typename Module>
bool preflightVariant(detail::LazyKernelModule<Module>& state, char const* moduleName, cudaStream_t stream)
{
    return detail::ensureModuleLoaded<moduleLoader, moduleUnloader>(state, moduleName, stream);
}

} // namespace

bool CuteDslFMHAV2Runner::canImplement(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t smVersion,
    nvinfer1::DataType dataType, CuteDslFMHAV2MaskType maskType)
{
    if (!isFMHAV2SM(smVersion) || dataType != nvinfer1::DataType::kHALF || numQHeads <= 0 || numKVHeads <= 0
        || numQHeads < numKVHeads || numQHeads % numKVHeads != 0)
    {
        return false;
    }

    switch (maskType)
    {
    case CuteDslFMHAV2MaskType::kCAUSAL:
    case CuteDslFMHAV2MaskType::kSLIDING_CAUSAL:
        return headSize == 64 || headSize == 128 || headSize == 256 || headSize == 512;
    case CuteDslFMHAV2MaskType::kVISION_BLOCK: return headSize == 256 || headSize == 512;
    case CuteDslFMHAV2MaskType::kPADDING: return headSize == 256;
    }
    return false;
}

bool CuteDslFMHAV2Runner::canImplementPaged(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t smVersion,
    nvinfer1::DataType dataType, CuteDslFMHAV2MaskType maskType)
{
    if (!isFMHAV2SM(smVersion) || !isFMHAV2PagedLlmHeadSize(headSize) || dataType != nvinfer1::DataType::kHALF
        || numQHeads <= 0 || numKVHeads <= 0 || numQHeads < numKVHeads || numQHeads % numKVHeads != 0)
    {
        return false;
    }

    switch (maskType)
    {
    case CuteDslFMHAV2MaskType::kCAUSAL:
    case CuteDslFMHAV2MaskType::kSLIDING_CAUSAL: return true;
    case CuteDslFMHAV2MaskType::kPADDING:
    case CuteDslFMHAV2MaskType::kVISION_BLOCK: return false;
    }
    return false;
}

bool CuteDslFMHAV2Runner::canImplementViT(int32_t headSize, int32_t smVersion, nvinfer1::DataType dataType)
{
    return isFMHAV2SM(smVersion) && dataType == nvinfer1::DataType::kHALF
        && (headSize == 64 || headSize == 72 || headSize == 80 || headSize == 128);
}

bool CuteDslFMHAV2Runner::preflightLlm(cudaStream_t stream, int32_t slidingWindowSize)
{
    bool const useSlidingWindow = slidingWindowSize < INT_MAX;

    switch (mHeadDim)
    {
    case 64:
        if (useSlidingWindow)
        {
            return preflightVariant<fmha_v2_d64_sw_Kernel_Module_Load, fmha_v2_d64_sw_Kernel_Module_Unload>(
                sLLM_d64Sw, "fmha_v2_d64_sw", stream);
        }
        if (mUseSmallD64 && mSeqLenQ <= 512)
        {
            return preflightVariant<fmha_v2_d64_small_Kernel_Module_Load, fmha_v2_d64_small_Kernel_Module_Unload>(
                sLLM_d64Small, "fmha_v2_d64_small", stream);
        }
        return preflightVariant<fmha_v2_d64_Kernel_Module_Load, fmha_v2_d64_Kernel_Module_Unload>(
            sLLM_d64, "fmha_v2_d64", stream);
    case 128:
        return useSlidingWindow
            ? preflightVariant<fmha_v2_d128_sw_Kernel_Module_Load, fmha_v2_d128_sw_Kernel_Module_Unload>(
                  sLLM_d128Sw, "fmha_v2_d128_sw", stream)
            : preflightVariant<fmha_v2_d128_Kernel_Module_Load, fmha_v2_d128_Kernel_Module_Unload>(
                  sLLM_d128, "fmha_v2_d128", stream);
    case 256:
        return useSlidingWindow
            ? preflightVariant<fmha_v2_d256_sw_Kernel_Module_Load, fmha_v2_d256_sw_Kernel_Module_Unload>(
                  sLLM_d256Sw, "fmha_v2_d256_sw", stream)
            : preflightVariant<fmha_v2_d256_Kernel_Module_Load, fmha_v2_d256_Kernel_Module_Unload>(
                  sLLM_d256, "fmha_v2_d256", stream);
    case 512:
        return useSlidingWindow
            ? preflightVariant<fmha_v2_d512_sw_Kernel_Module_Load, fmha_v2_d512_sw_Kernel_Module_Unload>(
                  sLLM_d512Sw, "fmha_v2_d512_sw", stream)
            : preflightVariant<fmha_v2_d512_Kernel_Module_Load, fmha_v2_d512_Kernel_Module_Unload>(
                  sLLM_d512, "fmha_v2_d512", stream);
    default: LOG_ERROR("FMHA-v2 CuTe DSL LLM FMHA: unsupported head_dim=%d", mHeadDim); return false;
    }
}

bool CuteDslFMHAV2Runner::preflightPaged(cudaStream_t stream, int32_t slidingWindowSize)
{
    bool const useSlidingWindow = slidingWindowSize < INT_MAX;

    switch (mHeadDim)
    {
    case 64:
        if (useSlidingWindow)
        {
            return preflightVariant<fmha_v2_d64_sw_paged_Kernel_Module_Load, fmha_v2_d64_sw_paged_Kernel_Module_Unload>(
                sLLM_d64SwPaged, "fmha_v2_d64_sw_paged", stream);
        }
        if (mSeqLenQ <= 512)
        {
            return preflightVariant<fmha_v2_d64_small_paged_Kernel_Module_Load,
                fmha_v2_d64_small_paged_Kernel_Module_Unload>(sLLM_d64SmallPaged, "fmha_v2_d64_small_paged", stream);
        }
        return preflightVariant<fmha_v2_d64_paged_Kernel_Module_Load, fmha_v2_d64_paged_Kernel_Module_Unload>(
            sLLM_d64Paged, "fmha_v2_d64_paged", stream);
    case 128:
        return useSlidingWindow
            ? preflightVariant<fmha_v2_d128_sw_paged_Kernel_Module_Load, fmha_v2_d128_sw_paged_Kernel_Module_Unload>(
                  sLLM_d128SwPaged, "fmha_v2_d128_sw_paged", stream)
            : preflightVariant<fmha_v2_d128_paged_Kernel_Module_Load, fmha_v2_d128_paged_Kernel_Module_Unload>(
                  sLLM_d128Paged, "fmha_v2_d128_paged", stream);
    case 256:
        return useSlidingWindow
            ? preflightVariant<fmha_v2_d256_sw_paged_Kernel_Module_Load, fmha_v2_d256_sw_paged_Kernel_Module_Unload>(
                  sLLM_d256SwPaged, "fmha_v2_d256_sw_paged", stream)
            : preflightVariant<fmha_v2_d256_paged_Kernel_Module_Load, fmha_v2_d256_paged_Kernel_Module_Unload>(
                  sLLM_d256Paged, "fmha_v2_d256_paged", stream);
    case 512:
        return useSlidingWindow
            ? preflightVariant<fmha_v2_d512_sw_paged_Kernel_Module_Load, fmha_v2_d512_sw_paged_Kernel_Module_Unload>(
                  sLLM_d512SwPaged, "fmha_v2_d512_sw_paged", stream)
            : preflightVariant<fmha_v2_d512_paged_Kernel_Module_Load, fmha_v2_d512_paged_Kernel_Module_Unload>(
                  sLLM_d512Paged, "fmha_v2_d512_paged", stream);
    default: LOG_ERROR("FMHA-v2 native paged FMHA: unsupported head_dim=%d.", mHeadDim); return false;
    }
}

bool CuteDslFMHAV2Runner::preflightPadding(cudaStream_t stream)
{
    if (mHeadDim != 256)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL padding FMHA requires head_dim=256.");
        return false;
    }
    return preflightVariant<fmha_v2_d256_padding_Kernel_Module_Load, fmha_v2_d256_padding_Kernel_Module_Unload>(
        sLLM_d256Padding, "fmha_v2_d256_padding", stream);
}

bool CuteDslFMHAV2Runner::preflightVisionBlock(cudaStream_t stream)
{
    if (mHeadDim != 256 && mHeadDim != 512)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL bidirectional FMHA requires head_dim=256 or head_dim=512.");
        return false;
    }
    return mHeadDim == 512 ? preflightVariant<fmha_v2_d512_bidirectional_Kernel_Module_Load,
                                 fmha_v2_d512_bidirectional_Kernel_Module_Unload>(
                                 sLLM_d512Bidirectional, "fmha_v2_d512_bidirectional", stream)
                           : preflightVariant<fmha_v2_d256_bidirectional_Kernel_Module_Load,
                                 fmha_v2_d256_bidirectional_Kernel_Module_Unload>(
                                 sLLM_d256Bidirectional, "fmha_v2_d256_bidirectional", stream);
}

bool CuteDslFMHAV2Runner::preflightViT(cudaStream_t stream)
{
    switch (mHeadDim)
    {
    case 64:
        return preflightVariant<fmha_v2_vit_d64_Kernel_Module_Load, fmha_v2_vit_d64_Kernel_Module_Unload>(
            sViT_d64, "fmha_v2_vit_d64", stream);
    case 72:
        return preflightVariant<fmha_v2_vit_d72_Kernel_Module_Load, fmha_v2_vit_d72_Kernel_Module_Unload>(
            sViT_d72, "fmha_v2_vit_d72", stream);
    case 80:
        return preflightVariant<fmha_v2_vit_d80_Kernel_Module_Load, fmha_v2_vit_d80_Kernel_Module_Unload>(
            sViT_d80, "fmha_v2_vit_d80", stream);
    case 128:
        return preflightVariant<fmha_v2_vit_d128_Kernel_Module_Load, fmha_v2_vit_d128_Kernel_Module_Unload>(
            sViT_d128, "fmha_v2_vit_d128", stream);
    default: LOG_ERROR("FMHA-v2 CuTe DSL ViT FMHA: unsupported head_dim=%d", mHeadDim); return false;
    }
}

CuteDslFMHAV2Runner::CuteDslFMHAV2Runner(int32_t numQHeads, int32_t numKVHeads, int32_t headDim, int32_t batchSize,
    int32_t seqLenQ, int32_t kvSeqLen, bool useSmallD64)
    : mBatchSize(batchSize)
    , mSeqLenQ(seqLenQ)
    , mKVSeqLen(kvSeqLen)
    , mNumHeadsQ(numQHeads)
    , mNumHeadsKV(numKVHeads)
    , mHeadDim(headDim)
    , mUseSmallD64(useSmallD64)
{
}

namespace
{

using cutedsl::makeCuSeqLenTensor;
using cutedsl::makePackedTensor;
using cutedsl::makeStridedTensor;
using cutedsl::kShapeRank;
using cutedsl::WrapperArity;
using cutedsl::WrapperArgT;

//! Populate a [B, S, H] descriptor over a contiguous [B, S, H, D] buffer. The FMHA-v2 kernels bake
//! the head dim in, so D is not one of the extents and these descriptors are never packed over
//! their own extents — the strides have to be spelled out.
template <class TensorT>
TensorT makeBshTensor(void const* data, int32_t batchSize, int32_t seqLen, int32_t numHeads, int32_t headDim)
{
    return makeStridedTensor<TensorT>(data, {batchSize, seqLen, numHeads},
        {static_cast<int64_t>(seqLen) * numHeads * headDim, static_cast<int64_t>(numHeads) * headDim});
}

//! Populate the generated descriptor for a contiguous BSND tensor. Some AOT
//! variants keep D static in the descriptor while others expose all four axes.
template <class TensorT>
TensorT makeBsndTensor(void const* data, int32_t batchSize, int32_t seqLen, int32_t numHeads, int32_t headDim)
{
    if constexpr (kShapeRank<TensorT> == 3)
    {
        return makeBshTensor<TensorT>(data, batchSize, seqLen, numHeads, headDim);
    }
    else
    {
        static_assert(kShapeRank<TensorT> == 4, "FMHA-v2 BSND descriptors must expose three or four extents.");
        return makePackedTensor<TensorT>(data, {batchSize, seqLen, numHeads, headDim});
    }
}

//! Packed-varlen counterpart of makeBshTensor(): a [total_S, H] descriptor over [total_S, H, D].
template <class TensorT>
TensorT makeShTensor(void const* data, int32_t totalSeqLen, int32_t numHeads, int32_t headDim)
{
    return makeStridedTensor<TensorT>(data, {totalSeqLen, numHeads}, {static_cast<int64_t>(numHeads) * headDim});
}

//! Launch an FMHA-v2 LLM variant over separate padded [B, S, H, D] Q/K/V.
//! @tparam cuteDslKernelWrapper Generated CuTe DSL kernel wrapper function. Its signature supplies the module
//! and tensor descriptor types at compile time.
template <auto cuteDslKernelWrapper, auto moduleLoader, auto moduleUnloader>
int32_t callFmhaV2Llm(detail::LazyKernelModule<WrapperArgT<0, decltype(cuteDslKernelWrapper)>>& state,
    char const* moduleName, FmhaV2LlmParams const& params)
{
    static_assert(WrapperArity<decltype(cuteDslKernelWrapper)>::value == 14,
        "callFmhaV2Llm: not an FMHA-v2 LLM wrapper (module, q_tensor, k_tensor, v_tensor, o_tensor, cum_seqlen_k, "
        "window_size_left, attention_scale, scale_q, scale_k, scale_v, inv_scale_o, sm_count, stream).");

    if (!detail::ensureModuleLoaded<moduleLoader, moduleUnloader>(state, moduleName, params.stream))
    {
        return -1;
    }
    auto& module = state.module;

    auto qTensor = makeBshTensor<WrapperArgT<1, decltype(cuteDslKernelWrapper)>>(
        params.qPtr, params.batchSize, params.seqLenQ, params.numQHeads, params.headDim);
    auto kTensor = makeBshTensor<WrapperArgT<2, decltype(cuteDslKernelWrapper)>>(
        params.kPtr, params.batchSize, params.kvSeqLen, params.numKVHeads, params.headDim);
    auto vTensor = makeBshTensor<WrapperArgT<3, decltype(cuteDslKernelWrapper)>>(
        params.vPtr, params.batchSize, params.kvSeqLen, params.numKVHeads, params.headDim);
    auto oTensor = makeBshTensor<WrapperArgT<4, decltype(cuteDslKernelWrapper)>>(
        params.oPtr, params.batchSize, params.seqLenQ, params.numQHeads, params.headDim);
    auto cumSeqlenK
        = makeCuSeqLenTensor<WrapperArgT<5, decltype(cuteDslKernelWrapper)>>(params.cuKVSeqLens, params.batchSize + 1);

    return cuteDslKernelWrapper(&module, &qTensor, &kTensor, &vTensor, &oTensor, &cumSeqlenK, params.windowSizeLeft,
        params.attentionScale, params.scaleQ, params.scaleK, params.scaleV, params.invScaleO,
        getDeviceMultiProcessorCount(), params.stream);
}

//! Launch an FMHA-v2 bidirectional variant over separate padded [B, S, H, D] Q/K/V.
//! @tparam cuteDslKernelWrapper Generated CuTe DSL kernel wrapper function. Its signature supplies the module
//! and tensor descriptor types at compile time.
template <auto cuteDslKernelWrapper, auto moduleLoader, auto moduleUnloader>
int32_t callFmhaV2Bidirectional(detail::LazyKernelModule<WrapperArgT<0, decltype(cuteDslKernelWrapper)>>& state,
    char const* moduleName, FmhaV2LlmParams const& params)
{
    static_assert(WrapperArity<decltype(cuteDslKernelWrapper)>::value == 16,
        "callFmhaV2Bidirectional: not an FMHA-v2 bidirectional wrapper (module, q_tensor, k_tensor, v_tensor, "
        "o_tensor, cum_seqlen_k, block_begin, block_end, window_size_left, attention_scale, scale_q, scale_k, scale_v, "
        "inv_scale_o, sm_count, stream).");

    if (!detail::ensureModuleLoaded<moduleLoader, moduleUnloader>(state, moduleName, params.stream))
    {
        return -1;
    }
    auto& module = state.module;

    auto qTensor = makeBshTensor<WrapperArgT<1, decltype(cuteDslKernelWrapper)>>(
        params.qPtr, params.batchSize, params.seqLenQ, params.numQHeads, params.headDim);
    auto kTensor = makeBshTensor<WrapperArgT<2, decltype(cuteDslKernelWrapper)>>(
        params.kPtr, params.batchSize, params.kvSeqLen, params.numKVHeads, params.headDim);
    auto vTensor = makeBshTensor<WrapperArgT<3, decltype(cuteDslKernelWrapper)>>(
        params.vPtr, params.batchSize, params.kvSeqLen, params.numKVHeads, params.headDim);
    auto oTensor = makeBshTensor<WrapperArgT<4, decltype(cuteDslKernelWrapper)>>(
        params.oPtr, params.batchSize, params.seqLenQ, params.numQHeads, params.headDim);
    auto cumSeqlenK
        = makeCuSeqLenTensor<WrapperArgT<5, decltype(cuteDslKernelWrapper)>>(params.cuKVSeqLens, params.batchSize + 1);

    // The per-query block ranges are packed [B, S_q], unlike the Q/K/V/O descriptors above.
    auto blockBeginTensor = makePackedTensor<WrapperArgT<6, decltype(cuteDslKernelWrapper)>>(
        params.blockBegin, {params.batchSize, params.seqLenQ});
    auto blockEndTensor = makePackedTensor<WrapperArgT<7, decltype(cuteDslKernelWrapper)>>(
        params.blockEnd, {params.batchSize, params.seqLenQ});

    return cuteDslKernelWrapper(&module, &qTensor, &kTensor, &vTensor, &oTensor, &cumSeqlenK, &blockBeginTensor,
        &blockEndTensor, params.windowSizeLeft, params.attentionScale, params.scaleQ, params.scaleK, params.scaleV,
        params.invScaleO, getDeviceMultiProcessorCount(), params.stream);
}

//! Launch an FMHA-v2 LLM variant directly against the Edge-LLM NHD paged KV pool.
//! @tparam cuteDslKernelWrapper Generated CuTe DSL kernel wrapper function. Its signature supplies the module
//! and tensor descriptor types at compile time.
template <auto cuteDslKernelWrapper, auto moduleLoader, auto moduleUnloader>
int32_t callFmhaV2Paged(detail::LazyKernelModule<WrapperArgT<0, decltype(cuteDslKernelWrapper)>>& state,
    char const* moduleName, LlmFmhaPagedParams const& params)
{
    static_assert(WrapperArity<decltype(cuteDslKernelWrapper)>::value == 11,
        "callFmhaV2Paged: not an FMHA-v2 paged LLM wrapper (module, q_tensor, kv_cache_pool, "
        "kv_cache_page_list, o_tensor, cum_seqlen_q, cum_seqlen_k, window_size_left, attention_scale, sm_count, "
        "stream).");

    if (!detail::ensureModuleLoaded<moduleLoader, moduleUnloader>(state, moduleName, params.stream))
    {
        return -1;
    }
    auto& module = state.module;

    auto qTensor = makeBsndTensor<WrapperArgT<1, decltype(cuteDslKernelWrapper)>>(
        params.qPtr, params.batchSize, params.seqLenQ, params.numQHeads, params.headDim);

    // The physical pool is NHD [2P, T, Hkv, D]. The AOT kernel consumes a
    // logical [2P, Hkv, T, D] view; the permutation is encoded by the strides.
    auto kvPoolTensor = makeStridedTensor<WrapperArgT<2, decltype(cuteDslKernelWrapper)>>(params.pagedKVPoolPtr,
        {params.numPages, params.numKVHeads, params.tokensPerPage, params.headDim},
        {static_cast<int64_t>(params.tokensPerPage) * params.numKVHeads * params.headDim,
            static_cast<int64_t>(params.headDim), static_cast<int64_t>(params.numKVHeads) * params.headDim});

    auto pageListTensor = makePackedTensor<WrapperArgT<3, decltype(cuteDslKernelWrapper)>>(
        params.kvCachePageList, {params.batchSize, 2, params.maxPagesPerSeq});
    auto oTensor = makeBsndTensor<WrapperArgT<4, decltype(cuteDslKernelWrapper)>>(
        params.oPtr, params.batchSize, params.seqLenQ, params.numQHeads, params.headDim);
    auto cumSeqlenQ
        = makeCuSeqLenTensor<WrapperArgT<5, decltype(cuteDslKernelWrapper)>>(params.cuQSeqLens, params.batchSize + 1);
    auto cumSeqlenK
        = makeCuSeqLenTensor<WrapperArgT<6, decltype(cuteDslKernelWrapper)>>(params.cuKVSeqLens, params.batchSize + 1);

    return cuteDslKernelWrapper(&module, &qTensor, &kvPoolTensor, &pageListTensor, &oTensor, &cumSeqlenQ, &cumSeqlenK,
        params.windowSizeLeft, params.attentionScale, getDeviceMultiProcessorCount(), params.stream);
}

//! Launch an FMHA-v2 ViT variant over packed varlen [total_S, H, D] Q/K/V.
//! @tparam cuteDslKernelWrapper Generated CuTe DSL kernel wrapper function. Its signature supplies the module
//! and tensor descriptor types at compile time.
template <auto cuteDslKernelWrapper, auto moduleLoader, auto moduleUnloader>
int32_t callFmhaV2Vit(detail::LazyKernelModule<WrapperArgT<0, decltype(cuteDslKernelWrapper)>>& state,
    char const* moduleName, FmhaV2VitParams const& params)
{
    static_assert(WrapperArity<decltype(cuteDslKernelWrapper)>::value == 12,
        "callFmhaV2Vit: not an FMHA-v2 ViT wrapper (module, q_tensor, k_tensor, v_tensor, o_tensor, cu_seqlens, "
        "max_seqlen, scale_softmax_log2, scale_softmax, scale_output, sm_count, stream).");

    if (!detail::ensureModuleLoaded<moduleLoader, moduleUnloader>(state, moduleName, params.stream))
    {
        return -1;
    }
    auto& module = state.module;

    auto qTensor = makeShTensor<WrapperArgT<1, decltype(cuteDslKernelWrapper)>>(
        params.qPtr, params.totalSeqLen, params.numQHeads, params.headDim);
    auto kTensor = makeShTensor<WrapperArgT<2, decltype(cuteDslKernelWrapper)>>(
        params.kPtr, params.totalSeqLen, params.numKVHeads, params.headDim);
    auto vTensor = makeShTensor<WrapperArgT<3, decltype(cuteDslKernelWrapper)>>(
        params.vPtr, params.totalSeqLen, params.numKVHeads, params.headDim);
    auto oTensor = makeShTensor<WrapperArgT<4, decltype(cuteDslKernelWrapper)>>(
        params.oPtr, params.totalSeqLen, params.numQHeads, params.headDim);
    auto cuSeqlensTensor
        = makeCuSeqLenTensor<WrapperArgT<5, decltype(cuteDslKernelWrapper)>>(params.cuSeqLens, params.batchSize + 1);

    return cuteDslKernelWrapper(&module, &qTensor, &kTensor, &vTensor, &oTensor, &cuSeqlensTensor, params.maxSeqLen,
        params.scaleSoftmaxLog2, params.attentionScale, params.scaleOutput, getDeviceMultiProcessorCount(),
        params.stream);
}

} // namespace

bool CuteDslFMHAV2Runner::run(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr,
    int32_t const* cuKVSeqLens, cudaStream_t stream, float attentionScale, int32_t slidingWindowSize)
{
    validateAttentionScale(attentionScale);
    int32_t constexpr kNO_LIMIT = 1 << 30;
    bool const useSlidingWindow = slidingWindowSize < INT_MAX;

    FmhaV2LlmParams params{};
    params.qPtr = qPtr;
    params.kPtr = kPtr;
    params.vPtr = vPtr;
    params.oPtr = oPtr;
    params.cuKVSeqLens = cuKVSeqLens;
    params.batchSize = mBatchSize;
    params.seqLenQ = mSeqLenQ;
    params.kvSeqLen = mKVSeqLen;
    params.numQHeads = mNumHeadsQ;
    params.numKVHeads = mNumHeadsKV;
    params.headDim = mHeadDim;
    params.windowSizeLeft = useSlidingWindow ? slidingWindowSize : kNO_LIMIT;
    params.attentionScale = attentionScale;
    params.scaleQ = 1.0F;
    params.scaleK = 1.0F;
    params.scaleV = 1.0F;
    params.invScaleO = 1.0F;
    params.stream = stream;

    int32_t ret = -1;

    switch (mHeadDim)
    {
    case 64:
        if (useSlidingWindow)
        {
            ret = callFmhaV2Llm<cute_dsl_fmha_v2_d64_sw_wrapper, fmha_v2_d64_sw_Kernel_Module_Load,
                fmha_v2_d64_sw_Kernel_Module_Unload>(sLLM_d64Sw, "fmha_v2_d64_sw", params);
        }
        else if (mUseSmallD64 && mSeqLenQ <= 512)
        {
            ret = callFmhaV2Llm<cute_dsl_fmha_v2_d64_small_wrapper, fmha_v2_d64_small_Kernel_Module_Load,
                fmha_v2_d64_small_Kernel_Module_Unload>(sLLM_d64Small, "fmha_v2_d64_small", params);
        }
        else
        {
            ret = callFmhaV2Llm<cute_dsl_fmha_v2_d64_wrapper, fmha_v2_d64_Kernel_Module_Load,
                fmha_v2_d64_Kernel_Module_Unload>(sLLM_d64, "fmha_v2_d64", params);
        }
        break;
    case 128:
        ret = useSlidingWindow ? callFmhaV2Llm<cute_dsl_fmha_v2_d128_sw_wrapper, fmha_v2_d128_sw_Kernel_Module_Load,
                                     fmha_v2_d128_sw_Kernel_Module_Unload>(sLLM_d128Sw, "fmha_v2_d128_sw", params)
                               : callFmhaV2Llm<cute_dsl_fmha_v2_d128_wrapper, fmha_v2_d128_Kernel_Module_Load,
                                     fmha_v2_d128_Kernel_Module_Unload>(sLLM_d128, "fmha_v2_d128", params);
        break;
    case 256:
        ret = useSlidingWindow ? callFmhaV2Llm<cute_dsl_fmha_v2_d256_sw_wrapper, fmha_v2_d256_sw_Kernel_Module_Load,
                                     fmha_v2_d256_sw_Kernel_Module_Unload>(sLLM_d256Sw, "fmha_v2_d256_sw", params)
                               : callFmhaV2Llm<cute_dsl_fmha_v2_d256_wrapper, fmha_v2_d256_Kernel_Module_Load,
                                     fmha_v2_d256_Kernel_Module_Unload>(sLLM_d256, "fmha_v2_d256", params);
        break;
    case 512:
        ret = useSlidingWindow ? callFmhaV2Llm<cute_dsl_fmha_v2_d512_sw_wrapper, fmha_v2_d512_sw_Kernel_Module_Load,
                                     fmha_v2_d512_sw_Kernel_Module_Unload>(sLLM_d512Sw, "fmha_v2_d512_sw", params)
                               : callFmhaV2Llm<cute_dsl_fmha_v2_d512_wrapper, fmha_v2_d512_Kernel_Module_Load,
                                     fmha_v2_d512_Kernel_Module_Unload>(sLLM_d512, "fmha_v2_d512", params);
        break;
    default: LOG_ERROR("FMHA-v2 CuTe DSL LLM FMHA: unsupported head_dim=%d", mHeadDim); return false;
    }

    if (ret != 0)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL LLM FMHA kernel (d=%d, sw=%s) failed with error code: %d", mHeadDim,
            useSlidingWindow ? "true" : "false", ret);
    }
    return ret == 0;
}

bool CuteDslFMHAV2Runner::runPaged(void const* qPtr, void const* pagedKVPoolPtr, int32_t const* kvCachePageList,
    void* oPtr, int32_t const* cuQSeqLens, int32_t const* cuKVSeqLens, int32_t numFlatPages, int32_t maxPagesPerSeq,
    int32_t tokensPerPage, cudaStream_t stream, float attentionScale, int32_t slidingWindowSize)
{
    check::check(qPtr != nullptr, "FMHA-v2 native paged FMHA qPtr must not be null.");
    check::check(pagedKVPoolPtr != nullptr, "FMHA-v2 native paged FMHA KV pool must not be null.");
    check::check(kvCachePageList != nullptr, "FMHA-v2 native paged FMHA page list must not be null.");
    check::check(oPtr != nullptr, "FMHA-v2 native paged FMHA oPtr must not be null.");
    check::check(cuQSeqLens != nullptr, "FMHA-v2 native paged FMHA cuQSeqLens must not be null.");
    check::check(cuKVSeqLens != nullptr, "FMHA-v2 native paged FMHA cuKVSeqLens must not be null.");
    check::check(mBatchSize > 0 && mSeqLenQ > 0 && mKVSeqLen > 0 && mNumHeadsQ > 0 && mNumHeadsKV > 0,
        "FMHA-v2 native paged FMHA requires positive tensor extents.");
    check::check(mNumHeadsQ >= mNumHeadsKV && mNumHeadsQ % mNumHeadsKV == 0,
        "FMHA-v2 native paged FMHA requires Q heads to be divisible by KV heads.");
    check::check(numFlatPages > 0 && numFlatPages % 2 == 0 && maxPagesPerSeq > 0,
        "FMHA-v2 native paged FMHA requires an even positive flattened page count and maxPagesPerSeq.");
    check::check(tokensPerPage == 128,
        "FMHA-v2 native paged FMHA requires tokensPerPage == 128 because one K/V tile maps to one page.");
    check::check(mKVSeqLen == maxPagesPerSeq * tokensPerPage,
        "FMHA-v2 native paged FMHA runner capacity must equal maxPagesPerSeq * tokensPerPage.");
    check::check(slidingWindowSize >= 0 || slidingWindowSize == INT_MAX,
        "FMHA-v2 native paged FMHA slidingWindowSize must be non-negative or INT_MAX.");
    bool const useSlidingWindow = slidingWindowSize < INT_MAX;
    validateAttentionScale(attentionScale);
    int32_t constexpr kNO_LIMIT = 1 << 30;

    LlmFmhaPagedParams params{};
    params.qPtr = qPtr;
    params.pagedKVPoolPtr = pagedKVPoolPtr;
    params.kvCachePageList = kvCachePageList;
    params.oPtr = oPtr;
    params.cuQSeqLens = cuQSeqLens;
    params.cuKVSeqLens = cuKVSeqLens;
    params.batchSize = mBatchSize;
    params.seqLenQ = mSeqLenQ;
    params.numQHeads = mNumHeadsQ;
    params.numKVHeads = mNumHeadsKV;
    params.headDim = mHeadDim;
    params.numPages = numFlatPages;
    params.maxPagesPerSeq = maxPagesPerSeq;
    params.tokensPerPage = tokensPerPage;
    params.windowSizeLeft = useSlidingWindow ? slidingWindowSize : kNO_LIMIT;
    params.attentionScale = attentionScale;
    params.stream = stream;

    int32_t ret = -1;

    switch (mHeadDim)
    {
    case 64:
        ret = useSlidingWindow
            ? callFmhaV2Paged<cute_dsl_fmha_v2_d64_sw_paged_wrapper, fmha_v2_d64_sw_paged_Kernel_Module_Load,
                  fmha_v2_d64_sw_paged_Kernel_Module_Unload>(sLLM_d64SwPaged, "fmha_v2_d64_sw_paged", params)
            : (mSeqLenQ <= 512
                      ? callFmhaV2Paged<cute_dsl_fmha_v2_d64_small_paged_wrapper,
                            fmha_v2_d64_small_paged_Kernel_Module_Load, fmha_v2_d64_small_paged_Kernel_Module_Unload>(
                            sLLM_d64SmallPaged, "fmha_v2_d64_small_paged", params)
                      : callFmhaV2Paged<cute_dsl_fmha_v2_d64_paged_wrapper, fmha_v2_d64_paged_Kernel_Module_Load,
                            fmha_v2_d64_paged_Kernel_Module_Unload>(sLLM_d64Paged, "fmha_v2_d64_paged", params));
        break;
    case 128:
        ret = useSlidingWindow
            ? callFmhaV2Paged<cute_dsl_fmha_v2_d128_sw_paged_wrapper, fmha_v2_d128_sw_paged_Kernel_Module_Load,
                  fmha_v2_d128_sw_paged_Kernel_Module_Unload>(sLLM_d128SwPaged, "fmha_v2_d128_sw_paged", params)
            : callFmhaV2Paged<cute_dsl_fmha_v2_d128_paged_wrapper, fmha_v2_d128_paged_Kernel_Module_Load,
                  fmha_v2_d128_paged_Kernel_Module_Unload>(sLLM_d128Paged, "fmha_v2_d128_paged", params);
        break;
    case 256:
        ret = useSlidingWindow
            ? callFmhaV2Paged<cute_dsl_fmha_v2_d256_sw_paged_wrapper, fmha_v2_d256_sw_paged_Kernel_Module_Load,
                  fmha_v2_d256_sw_paged_Kernel_Module_Unload>(sLLM_d256SwPaged, "fmha_v2_d256_sw_paged", params)
            : callFmhaV2Paged<cute_dsl_fmha_v2_d256_paged_wrapper, fmha_v2_d256_paged_Kernel_Module_Load,
                  fmha_v2_d256_paged_Kernel_Module_Unload>(sLLM_d256Paged, "fmha_v2_d256_paged", params);
        break;
    case 512:
        ret = useSlidingWindow
            ? callFmhaV2Paged<cute_dsl_fmha_v2_d512_sw_paged_wrapper, fmha_v2_d512_sw_paged_Kernel_Module_Load,
                  fmha_v2_d512_sw_paged_Kernel_Module_Unload>(sLLM_d512SwPaged, "fmha_v2_d512_sw_paged", params)
            : callFmhaV2Paged<cute_dsl_fmha_v2_d512_paged_wrapper, fmha_v2_d512_paged_Kernel_Module_Load,
                  fmha_v2_d512_paged_Kernel_Module_Unload>(sLLM_d512Paged, "fmha_v2_d512_paged", params);
        break;
    default: LOG_ERROR("FMHA-v2 native paged FMHA: unsupported head_dim=%d.", mHeadDim); return false;
    }

    if (ret != 0)
    {
        LOG_ERROR("FMHA-v2 native paged FMHA kernel (d=%d, sw=%s) failed with error code: %d", mHeadDim,
            useSlidingWindow ? "true" : "false", ret);
    }
    return ret == 0;
}

bool CuteDslFMHAV2Runner::runPadding(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr,
    int32_t const* cuQSeqLens, int32_t const* cuKVSeqLens, cudaStream_t stream, float attentionScale)
{
    if (mHeadDim != 256)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL padding FMHA requires head_dim=256.");
        return false;
    }

    check::check(qPtr != nullptr, "FMHA-v2 CuTe DSL padding FMHA qPtr must not be null.");
    check::check(kPtr != nullptr, "FMHA-v2 CuTe DSL padding FMHA kPtr must not be null.");
    check::check(vPtr != nullptr, "FMHA-v2 CuTe DSL padding FMHA vPtr must not be null.");
    check::check(oPtr != nullptr, "FMHA-v2 CuTe DSL padding FMHA oPtr must not be null.");
    check::check(cuQSeqLens != nullptr, "FMHA-v2 CuTe DSL padding FMHA cuQSeqLens must not be null.");
    check::check(cuKVSeqLens != nullptr, "FMHA-v2 CuTe DSL padding FMHA cuKVSeqLens must not be null.");
    check::check(mBatchSize > 0 && mSeqLenQ > 0 && mKVSeqLen > 0 && mNumHeadsQ > 0 && mNumHeadsKV > 0,
        "FMHA-v2 CuTe DSL padding FMHA requires positive tensor extents.");
    check::check(mNumHeadsQ >= mNumHeadsKV && mNumHeadsQ % mNumHeadsKV == 0,
        "FMHA-v2 CuTe DSL padding FMHA requires Q heads to be divisible by KV heads.");

    validateAttentionScale(attentionScale);

    using WrapperFn = decltype(&cute_dsl_fmha_v2_d256_padding_wrapper);
    static_assert(WrapperArity<WrapperFn>::value == 10,
        "FMHA-v2 padding wrapper signature changed (module, mQ, mK, mV, mO, mCuSeqLenQ, mCuSeqLenK, attention_scale, "
        "num_heads_kv, stream).");

    auto qTensor = makeBshTensor<WrapperArgT<1, WrapperFn>>(qPtr, mBatchSize, mSeqLenQ, mNumHeadsQ, mHeadDim);
    auto kTensor = makeBshTensor<WrapperArgT<2, WrapperFn>>(kPtr, mBatchSize, mKVSeqLen, mNumHeadsKV, mHeadDim);
    auto vTensor = makeBshTensor<WrapperArgT<3, WrapperFn>>(vPtr, mBatchSize, mKVSeqLen, mNumHeadsKV, mHeadDim);
    auto oTensor = makeBshTensor<WrapperArgT<4, WrapperFn>>(oPtr, mBatchSize, mSeqLenQ, mNumHeadsQ, mHeadDim);
    auto cumSeqlenQ = makeCuSeqLenTensor<WrapperArgT<5, WrapperFn>>(cuQSeqLens, mBatchSize + 1);
    auto cumSeqlenK = makeCuSeqLenTensor<WrapperArgT<6, WrapperFn>>(cuKVSeqLens, mBatchSize + 1);

    if (!detail::ensureModuleLoaded<fmha_v2_d256_padding_Kernel_Module_Load, fmha_v2_d256_padding_Kernel_Module_Unload>(
            sLLM_d256Padding, "fmha_v2_d256_padding", stream))
    {
        return false;
    }
    int32_t const ret = cute_dsl_fmha_v2_d256_padding_wrapper(&sLLM_d256Padding.module, &qTensor, &kTensor, &vTensor,
        &oTensor, &cumSeqlenQ, &cumSeqlenK, attentionScale, mNumHeadsKV, stream);
    if (ret != 0)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL padding FMHA kernel failed with error code: %d", ret);
    }
    return ret == 0;
}

bool CuteDslFMHAV2Runner::runVisionBlock(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr,
    int32_t const* cuKVSeqLens, int32_t const* blockBegin, int32_t const* blockEnd, cudaStream_t stream,
    float attentionScale, int32_t slidingWindowSize)
{
    if (mHeadDim != 256 && mHeadDim != 512)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL bidirectional FMHA requires head_dim=256 or head_dim=512; got %d.", mHeadDim);
        return false;
    }
    if (blockBegin == nullptr || blockEnd == nullptr)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL bidirectional FMHA requires both blockBegin and blockEnd.");
        return false;
    }

    validateAttentionScale(attentionScale);
    // Both bidirectional variants bake in the sliding-window branch and take the window as a runtime
    // argument, so the Gemma4 global layers (no window) pass an unreachable bound instead of a
    // negative sentinel the kernel would clamp to zero.
    int32_t constexpr kNO_LIMIT = 1 << 30;

    FmhaV2LlmParams params{};
    params.qPtr = qPtr;
    params.kPtr = kPtr;
    params.vPtr = vPtr;
    params.oPtr = oPtr;
    params.cuKVSeqLens = cuKVSeqLens;
    params.batchSize = mBatchSize;
    params.seqLenQ = mSeqLenQ;
    params.kvSeqLen = mKVSeqLen;
    params.numQHeads = mNumHeadsQ;
    params.numKVHeads = mNumHeadsKV;
    params.headDim = mHeadDim;
    params.windowSizeLeft = slidingWindowSize >= 0 ? slidingWindowSize : kNO_LIMIT;
    params.attentionScale = attentionScale;
    params.scaleQ = 1.0F;
    params.scaleK = 1.0F;
    params.scaleV = 1.0F;
    params.invScaleO = 1.0F;
    params.stream = stream;
    params.blockBegin = blockBegin;
    params.blockEnd = blockEnd;

    int32_t const ret = mHeadDim == 512
        ? callFmhaV2Bidirectional<cute_dsl_fmha_v2_d512_bidirectional_wrapper,
              fmha_v2_d512_bidirectional_Kernel_Module_Load, fmha_v2_d512_bidirectional_Kernel_Module_Unload>(
              sLLM_d512Bidirectional, "fmha_v2_d512_bidirectional", params)
        : callFmhaV2Bidirectional<cute_dsl_fmha_v2_d256_bidirectional_wrapper,
              fmha_v2_d256_bidirectional_Kernel_Module_Load, fmha_v2_d256_bidirectional_Kernel_Module_Unload>(
              sLLM_d256Bidirectional, "fmha_v2_d256_bidirectional", params);
    if (ret != 0)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL bidirectional FMHA kernel (d=%d) failed with error code: %d", mHeadDim, ret);
    }
    return ret == 0;
}

bool CuteDslFMHAV2Runner::run(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr,
    int32_t const* cuSeqLens, int32_t totalSeqLen, int32_t maxSeqLen, int32_t batchSize, cudaStream_t stream,
    float attentionScale)
{
    validateAttentionScale(attentionScale);

    FmhaV2VitParams params{};
    params.qPtr = qPtr;
    params.kPtr = kPtr;
    params.vPtr = vPtr;
    params.oPtr = oPtr;
    params.cuSeqLens = cuSeqLens;
    params.totalSeqLen = totalSeqLen;
    params.numQHeads = mNumHeadsQ;
    params.numKVHeads = mNumHeadsKV;
    params.headDim = mHeadDim;
    params.maxSeqLen = maxSeqLen;
    params.batchSize = batchSize;
    params.scaleSoftmaxLog2 = attentionScale * static_cast<float>(M_LOG2E);
    params.attentionScale = attentionScale;
    params.scaleOutput = 1.0F;
    params.stream = stream;

    int32_t ret = -1;

    switch (mHeadDim)
    {
    case 64:
        ret = callFmhaV2Vit<cute_dsl_fmha_v2_vit_d64_wrapper, fmha_v2_vit_d64_Kernel_Module_Load,
            fmha_v2_vit_d64_Kernel_Module_Unload>(sViT_d64, "fmha_v2_vit_d64", params);
        break;
    case 72:
        ret = callFmhaV2Vit<cute_dsl_fmha_v2_vit_d72_wrapper, fmha_v2_vit_d72_Kernel_Module_Load,
            fmha_v2_vit_d72_Kernel_Module_Unload>(sViT_d72, "fmha_v2_vit_d72", params);
        break;
    case 80:
        ret = callFmhaV2Vit<cute_dsl_fmha_v2_vit_d80_wrapper, fmha_v2_vit_d80_Kernel_Module_Load,
            fmha_v2_vit_d80_Kernel_Module_Unload>(sViT_d80, "fmha_v2_vit_d80", params);
        break;
    case 128:
        ret = callFmhaV2Vit<cute_dsl_fmha_v2_vit_d128_wrapper, fmha_v2_vit_d128_Kernel_Module_Load,
            fmha_v2_vit_d128_Kernel_Module_Unload>(sViT_d128, "fmha_v2_vit_d128", params);
        break;
    default: LOG_ERROR("FMHA-v2 CuTe DSL ViT FMHA: unsupported head_dim=%d", mHeadDim); return false;
    }

    if (ret != 0)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL ViT FMHA kernel (d=%d) failed with error code: %d", mHeadDim, ret);
    }
    return ret == 0;
}

} // namespace trt_edgellm

#else

// Keep symbols available for unconditional callers; false reports that CuTe DSL kernels are unavailable.
namespace trt_edgellm
{

CuteDslFMHAV2Runner::CuteDslFMHAV2Runner(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, bool) {}

bool CuteDslFMHAV2Runner::canImplement(int32_t, int32_t, int32_t, int32_t, nvinfer1::DataType, CuteDslFMHAV2MaskType)
{
    return false;
}

bool CuteDslFMHAV2Runner::canImplementPaged(
    int32_t, int32_t, int32_t, int32_t, nvinfer1::DataType, CuteDslFMHAV2MaskType)
{
    return false;
}

bool CuteDslFMHAV2Runner::canImplementViT(int32_t, int32_t, nvinfer1::DataType)
{
    return false;
}

bool CuteDslFMHAV2Runner::preflightLlm(cudaStream_t, int32_t)
{
    return false;
}

bool CuteDslFMHAV2Runner::preflightPaged(cudaStream_t, int32_t)
{
    return false;
}

bool CuteDslFMHAV2Runner::preflightPadding(cudaStream_t)
{
    return false;
}

bool CuteDslFMHAV2Runner::preflightVisionBlock(cudaStream_t)
{
    return false;
}

bool CuteDslFMHAV2Runner::preflightViT(cudaStream_t)
{
    return false;
}

bool CuteDslFMHAV2Runner::run(
    void const*, void const*, void const*, void*, int32_t const*, cudaStream_t, float, int32_t)
{
    return false;
}

bool CuteDslFMHAV2Runner::runPaged(void const*, void const*, int32_t const*, void*, int32_t const*, int32_t const*,
    int32_t, int32_t, int32_t, cudaStream_t, float, int32_t)
{
    return false;
}

bool CuteDslFMHAV2Runner::runPadding(
    void const*, void const*, void const*, void*, int32_t const*, int32_t const*, cudaStream_t, float)
{
    return false;
}

bool CuteDslFMHAV2Runner::runVisionBlock(void const*, void const*, void const*, void*, int32_t const*, int32_t const*,
    int32_t const*, cudaStream_t, float, int32_t)
{
    return false;
}

bool CuteDslFMHAV2Runner::run(
    void const*, void const*, void const*, void*, int32_t const*, int32_t, int32_t, int32_t, cudaStream_t, float)
{
    return false;
}

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_FMHA_ENABLED)
