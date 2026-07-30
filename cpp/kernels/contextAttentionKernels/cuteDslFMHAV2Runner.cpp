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

#ifdef CUTE_DSL_FMHA_V2_ENABLED

#include "cuteDslFMHAV2Runner.h"

#include "attentionScaleUtils.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "cuteDslFmhaParams.h"
#include "cuteDslTensorDescriptors.h"

#include <cmath>
#include <stdexcept>

namespace trt_edgellm
{

fmha_v2_d64_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d64{};
fmha_v2_d64_small_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d64Small{};
fmha_v2_d128_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d128{};
fmha_v2_d256_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d256{};
fmha_v2_d256_padding_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d256Padding{};
fmha_v2_d64_sw_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d64Sw{};
fmha_v2_d128_sw_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d128Sw{};
fmha_v2_d256_sw_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d256Sw{};
fmha_v2_d256_bidirectional_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d256Bidirectional{};
fmha_v2_d64_paged_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d64Paged{};
fmha_v2_d64_small_paged_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d64SmallPaged{};
fmha_v2_d128_paged_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d128Paged{};
fmha_v2_d256_paged_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d256Paged{};
fmha_v2_d512_paged_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d512Paged{};
fmha_v2_d64_sw_paged_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d64SwPaged{};
fmha_v2_d128_sw_paged_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d128SwPaged{};
fmha_v2_d256_sw_paged_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d256SwPaged{};
fmha_v2_d512_sw_paged_Kernel_Module_t CuteDslFMHAV2Runner::sLLM_d512SwPaged{};
bool CuteDslFMHAV2Runner::sLLMLoaded{false};
std::mutex CuteDslFMHAV2Runner::sLLMMutex;

fmha_v2_vit_d64_Kernel_Module_t CuteDslFMHAV2Runner::sViT_d64{};
fmha_v2_vit_d72_Kernel_Module_t CuteDslFMHAV2Runner::sViT_d72{};
fmha_v2_vit_d80_Kernel_Module_t CuteDslFMHAV2Runner::sViT_d80{};
fmha_v2_vit_d128_Kernel_Module_t CuteDslFMHAV2Runner::sViT_d128{};
bool CuteDslFMHAV2Runner::sViTLoaded{false};
std::mutex CuteDslFMHAV2Runner::sViTMutex;

namespace
{

bool isFMHAV2SM(int32_t smVersion)
{
    return smVersion == 120 || smVersion == 121;
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

template <typename Module, typename Loader>
void loadModule(Module& module, Loader loader, char const* name)
{
    loader(&module);
    if (module.module == nullptr)
    {
        throw std::runtime_error(std::string("CuTe DSL module loader returned a null handle for ") + name);
    }
}

//! Unload through the generated helper: it is emitted per artifact flavour and
//! calls cuModuleUnload or cudaLibraryUnload to match. The helper reports errors
//! through CUTE_DSL_CUDA_ERROR_CHECK, which throws here, so contain it -- callers
//! include the partial-load cleanup path.
template <typename Module, typename Unloader>
void unloadModule(Module& module, Unloader unloader, char const* name) noexcept
{
    if (module.module == nullptr)
    {
        return;
    }
    try
    {
        unloader(&module);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to unload CuTe DSL module %s: %s", name, e.what());
    }
    module.module = nullptr;
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
    case CuteDslFMHAV2MaskType::kSLIDING_CAUSAL: return headSize == 64 || headSize == 128 || headSize == 256;
    case CuteDslFMHAV2MaskType::kVISION_BLOCK: return headSize == 256;
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

bool CuteDslFMHAV2Runner::loadLLMKernelModule()
{
    std::lock_guard<std::mutex> lock{sLLMMutex};
    if (sLLMLoaded)
    {
        return true;
    }

    try
    {
        loadModule(sLLM_d64, fmha_v2_d64_Kernel_Module_Load, "fmha_v2_d64");
        loadModule(sLLM_d64Small, fmha_v2_d64_small_Kernel_Module_Load, "fmha_v2_d64_small");
        loadModule(sLLM_d128, fmha_v2_d128_Kernel_Module_Load, "fmha_v2_d128");
        loadModule(sLLM_d256, fmha_v2_d256_Kernel_Module_Load, "fmha_v2_d256");
        loadModule(sLLM_d256Padding, fmha_v2_d256_padding_Kernel_Module_Load, "fmha_v2_d256_padding");
        loadModule(sLLM_d64Sw, fmha_v2_d64_sw_Kernel_Module_Load, "fmha_v2_d64_sw");
        loadModule(sLLM_d128Sw, fmha_v2_d128_sw_Kernel_Module_Load, "fmha_v2_d128_sw");
        loadModule(sLLM_d256Sw, fmha_v2_d256_sw_Kernel_Module_Load, "fmha_v2_d256_sw");
        loadModule(sLLM_d256Bidirectional, fmha_v2_d256_bidirectional_Kernel_Module_Load, "fmha_v2_d256_bidirectional");
        loadModule(sLLM_d64Paged, fmha_v2_d64_paged_Kernel_Module_Load, "fmha_v2_d64_paged");
        loadModule(sLLM_d64SmallPaged, fmha_v2_d64_small_paged_Kernel_Module_Load, "fmha_v2_d64_small_paged");
        loadModule(sLLM_d128Paged, fmha_v2_d128_paged_Kernel_Module_Load, "fmha_v2_d128_paged");
        loadModule(sLLM_d256Paged, fmha_v2_d256_paged_Kernel_Module_Load, "fmha_v2_d256_paged");
        loadModule(sLLM_d512Paged, fmha_v2_d512_paged_Kernel_Module_Load, "fmha_v2_d512_paged");
        loadModule(sLLM_d64SwPaged, fmha_v2_d64_sw_paged_Kernel_Module_Load, "fmha_v2_d64_sw_paged");
        loadModule(sLLM_d128SwPaged, fmha_v2_d128_sw_paged_Kernel_Module_Load, "fmha_v2_d128_sw_paged");
        loadModule(sLLM_d256SwPaged, fmha_v2_d256_sw_paged_Kernel_Module_Load, "fmha_v2_d256_sw_paged");
        loadModule(sLLM_d512SwPaged, fmha_v2_d512_sw_paged_Kernel_Module_Load, "fmha_v2_d512_sw_paged");
        sLLMLoaded = true;
        LOG_DEBUG("FMHA-v2 CuTe DSL LLM FMHA kernel modules loaded");
        return true;
    }
    catch (...)
    {
        unloadModule(sLLM_d64, fmha_v2_d64_Kernel_Module_Unload, "fmha_v2_d64");
        unloadModule(sLLM_d64Small, fmha_v2_d64_small_Kernel_Module_Unload, "fmha_v2_d64_small");
        unloadModule(sLLM_d128, fmha_v2_d128_Kernel_Module_Unload, "fmha_v2_d128");
        unloadModule(sLLM_d256, fmha_v2_d256_Kernel_Module_Unload, "fmha_v2_d256");
        unloadModule(sLLM_d256Padding, fmha_v2_d256_padding_Kernel_Module_Unload, "fmha_v2_d256_padding");
        unloadModule(sLLM_d64Sw, fmha_v2_d64_sw_Kernel_Module_Unload, "fmha_v2_d64_sw");
        unloadModule(sLLM_d128Sw, fmha_v2_d128_sw_Kernel_Module_Unload, "fmha_v2_d128_sw");
        unloadModule(sLLM_d256Sw, fmha_v2_d256_sw_Kernel_Module_Unload, "fmha_v2_d256_sw");
        unloadModule(
            sLLM_d256Bidirectional, fmha_v2_d256_bidirectional_Kernel_Module_Unload, "fmha_v2_d256_bidirectional");
        unloadModule(sLLM_d64Paged, fmha_v2_d64_paged_Kernel_Module_Unload, "fmha_v2_d64_paged");
        unloadModule(sLLM_d64SmallPaged, fmha_v2_d64_small_paged_Kernel_Module_Unload, "fmha_v2_d64_small_paged");
        unloadModule(sLLM_d128Paged, fmha_v2_d128_paged_Kernel_Module_Unload, "fmha_v2_d128_paged");
        unloadModule(sLLM_d256Paged, fmha_v2_d256_paged_Kernel_Module_Unload, "fmha_v2_d256_paged");
        unloadModule(sLLM_d512Paged, fmha_v2_d512_paged_Kernel_Module_Unload, "fmha_v2_d512_paged");
        unloadModule(sLLM_d64SwPaged, fmha_v2_d64_sw_paged_Kernel_Module_Unload, "fmha_v2_d64_sw_paged");
        unloadModule(sLLM_d128SwPaged, fmha_v2_d128_sw_paged_Kernel_Module_Unload, "fmha_v2_d128_sw_paged");
        unloadModule(sLLM_d256SwPaged, fmha_v2_d256_sw_paged_Kernel_Module_Unload, "fmha_v2_d256_sw_paged");
        unloadModule(sLLM_d512SwPaged, fmha_v2_d512_sw_paged_Kernel_Module_Unload, "fmha_v2_d512_sw_paged");
        LOG_ERROR("Failed to load FMHA-v2 CuTe DSL LLM FMHA kernel modules");
        return false;
    }
}

void CuteDslFMHAV2Runner::unloadLLMKernelModule()
{
    std::lock_guard<std::mutex> lock{sLLMMutex};
    if (!sLLMLoaded)
    {
        return;
    }

    unloadModule(sLLM_d64, fmha_v2_d64_Kernel_Module_Unload, "fmha_v2_d64");
    unloadModule(sLLM_d64Small, fmha_v2_d64_small_Kernel_Module_Unload, "fmha_v2_d64_small");
    unloadModule(sLLM_d128, fmha_v2_d128_Kernel_Module_Unload, "fmha_v2_d128");
    unloadModule(sLLM_d256, fmha_v2_d256_Kernel_Module_Unload, "fmha_v2_d256");
    unloadModule(sLLM_d256Padding, fmha_v2_d256_padding_Kernel_Module_Unload, "fmha_v2_d256_padding");
    unloadModule(sLLM_d64Sw, fmha_v2_d64_sw_Kernel_Module_Unload, "fmha_v2_d64_sw");
    unloadModule(sLLM_d128Sw, fmha_v2_d128_sw_Kernel_Module_Unload, "fmha_v2_d128_sw");
    unloadModule(sLLM_d256Sw, fmha_v2_d256_sw_Kernel_Module_Unload, "fmha_v2_d256_sw");
    unloadModule(sLLM_d256Bidirectional, fmha_v2_d256_bidirectional_Kernel_Module_Unload, "fmha_v2_d256_bidirectional");
    unloadModule(sLLM_d64Paged, fmha_v2_d64_paged_Kernel_Module_Unload, "fmha_v2_d64_paged");
    unloadModule(sLLM_d64SmallPaged, fmha_v2_d64_small_paged_Kernel_Module_Unload, "fmha_v2_d64_small_paged");
    unloadModule(sLLM_d128Paged, fmha_v2_d128_paged_Kernel_Module_Unload, "fmha_v2_d128_paged");
    unloadModule(sLLM_d256Paged, fmha_v2_d256_paged_Kernel_Module_Unload, "fmha_v2_d256_paged");
    unloadModule(sLLM_d512Paged, fmha_v2_d512_paged_Kernel_Module_Unload, "fmha_v2_d512_paged");
    unloadModule(sLLM_d64SwPaged, fmha_v2_d64_sw_paged_Kernel_Module_Unload, "fmha_v2_d64_sw_paged");
    unloadModule(sLLM_d128SwPaged, fmha_v2_d128_sw_paged_Kernel_Module_Unload, "fmha_v2_d128_sw_paged");
    unloadModule(sLLM_d256SwPaged, fmha_v2_d256_sw_paged_Kernel_Module_Unload, "fmha_v2_d256_sw_paged");
    unloadModule(sLLM_d512SwPaged, fmha_v2_d512_sw_paged_Kernel_Module_Unload, "fmha_v2_d512_sw_paged");
    sLLMLoaded = false;
}

bool CuteDslFMHAV2Runner::loadViTKernelModule()
{
    std::lock_guard<std::mutex> lock{sViTMutex};
    if (sViTLoaded)
    {
        return true;
    }

    try
    {
        loadModule(sViT_d64, fmha_v2_vit_d64_Kernel_Module_Load, "fmha_v2_vit_d64");
        loadModule(sViT_d72, fmha_v2_vit_d72_Kernel_Module_Load, "fmha_v2_vit_d72");
        loadModule(sViT_d80, fmha_v2_vit_d80_Kernel_Module_Load, "fmha_v2_vit_d80");
        loadModule(sViT_d128, fmha_v2_vit_d128_Kernel_Module_Load, "fmha_v2_vit_d128");
        sViTLoaded = true;
        LOG_DEBUG("FMHA-v2 CuTe DSL ViT FMHA kernel modules loaded");
        return true;
    }
    catch (...)
    {
        unloadModule(sViT_d64, fmha_v2_vit_d64_Kernel_Module_Unload, "fmha_v2_vit_d64");
        unloadModule(sViT_d72, fmha_v2_vit_d72_Kernel_Module_Unload, "fmha_v2_vit_d72");
        unloadModule(sViT_d80, fmha_v2_vit_d80_Kernel_Module_Unload, "fmha_v2_vit_d80");
        unloadModule(sViT_d128, fmha_v2_vit_d128_Kernel_Module_Unload, "fmha_v2_vit_d128");
        LOG_ERROR("Failed to load FMHA-v2 CuTe DSL ViT FMHA kernel modules");
        return false;
    }
}

void CuteDslFMHAV2Runner::unloadViTKernelModule()
{
    std::lock_guard<std::mutex> lock{sViTMutex};
    if (!sViTLoaded)
    {
        return;
    }

    unloadModule(sViT_d64, fmha_v2_vit_d64_Kernel_Module_Unload, "fmha_v2_vit_d64");
    unloadModule(sViT_d72, fmha_v2_vit_d72_Kernel_Module_Unload, "fmha_v2_vit_d72");
    unloadModule(sViT_d80, fmha_v2_vit_d80_Kernel_Module_Unload, "fmha_v2_vit_d80");
    unloadModule(sViT_d128, fmha_v2_vit_d128_Kernel_Module_Unload, "fmha_v2_vit_d128");
    sViTLoaded = false;
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
template <auto cuteDslKernelWrapper>
int32_t callFmhaV2Llm(WrapperArgT<0, decltype(cuteDslKernelWrapper)>& module, FmhaV2LlmParams const& params)
{
    static_assert(WrapperArity<decltype(cuteDslKernelWrapper)>::value == 14,
        "callFmhaV2Llm: not an FMHA-v2 LLM wrapper (module, q_tensor, k_tensor, v_tensor, o_tensor, cum_seqlen_k, "
        "window_size_left, attention_scale, scale_q, scale_k, scale_v, inv_scale_o, sm_count, stream).");

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

//! Launch an FMHA-v2 LLM variant directly against the Edge-LLM NHD paged KV pool.
//! @tparam cuteDslKernelWrapper Generated CuTe DSL kernel wrapper function. Its signature supplies the module
//! and tensor descriptor types at compile time.
template <auto cuteDslKernelWrapper>
int32_t callFmhaV2Paged(WrapperArgT<0, decltype(cuteDslKernelWrapper)>& module, LlmFmhaPagedParams const& params)
{
    static_assert(WrapperArity<decltype(cuteDslKernelWrapper)>::value == 11,
        "callFmhaV2Paged: not an FMHA-v2 paged LLM wrapper (module, q_tensor, kv_cache_pool, "
        "kv_cache_page_list, o_tensor, cum_seqlen_q, cum_seqlen_k, window_size_left, attention_scale, sm_count, "
        "stream).");

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
template <auto cuteDslKernelWrapper>
int32_t callFmhaV2Vit(WrapperArgT<0, decltype(cuteDslKernelWrapper)>& module, FmhaV2VitParams const& params)
{
    static_assert(WrapperArity<decltype(cuteDslKernelWrapper)>::value == 12,
        "callFmhaV2Vit: not an FMHA-v2 ViT wrapper (module, q_tensor, k_tensor, v_tensor, o_tensor, cu_seqlens, "
        "max_seqlen, scale_softmax_log2, scale_softmax, scale_output, sm_count, stream).");

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
    if (!sLLMLoaded)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL LLM FMHA kernel module not loaded.");
        return false;
    }

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
            ret = callFmhaV2Llm<cute_dsl_fmha_v2_d64_sw_wrapper>(sLLM_d64Sw, params);
        }
        else if (mUseSmallD64 && mSeqLenQ <= 512)
        {
            ret = callFmhaV2Llm<cute_dsl_fmha_v2_d64_small_wrapper>(sLLM_d64Small, params);
        }
        else
        {
            ret = callFmhaV2Llm<cute_dsl_fmha_v2_d64_wrapper>(sLLM_d64, params);
        }
        break;
    case 128:
        ret = useSlidingWindow ? callFmhaV2Llm<cute_dsl_fmha_v2_d128_sw_wrapper>(sLLM_d128Sw, params)
                               : callFmhaV2Llm<cute_dsl_fmha_v2_d128_wrapper>(sLLM_d128, params);
        break;
    case 256:
        ret = useSlidingWindow ? callFmhaV2Llm<cute_dsl_fmha_v2_d256_sw_wrapper>(sLLM_d256Sw, params)
                               : callFmhaV2Llm<cute_dsl_fmha_v2_d256_wrapper>(sLLM_d256, params);
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
    if (!sLLMLoaded)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL LLM FMHA kernel module not loaded.");
        return false;
    }

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
            ? callFmhaV2Paged<cute_dsl_fmha_v2_d64_sw_paged_wrapper>(sLLM_d64SwPaged, params)
            : (mSeqLenQ <= 512 ? callFmhaV2Paged<cute_dsl_fmha_v2_d64_small_paged_wrapper>(sLLM_d64SmallPaged, params)
                               : callFmhaV2Paged<cute_dsl_fmha_v2_d64_paged_wrapper>(sLLM_d64Paged, params));
        break;
    case 128:
        ret = useSlidingWindow ? callFmhaV2Paged<cute_dsl_fmha_v2_d128_sw_paged_wrapper>(sLLM_d128SwPaged, params)
                               : callFmhaV2Paged<cute_dsl_fmha_v2_d128_paged_wrapper>(sLLM_d128Paged, params);
        break;
    case 256:
        ret = useSlidingWindow ? callFmhaV2Paged<cute_dsl_fmha_v2_d256_sw_paged_wrapper>(sLLM_d256SwPaged, params)
                               : callFmhaV2Paged<cute_dsl_fmha_v2_d256_paged_wrapper>(sLLM_d256Paged, params);
        break;
    case 512:
        ret = useSlidingWindow ? callFmhaV2Paged<cute_dsl_fmha_v2_d512_sw_paged_wrapper>(sLLM_d512SwPaged, params)
                               : callFmhaV2Paged<cute_dsl_fmha_v2_d512_paged_wrapper>(sLLM_d512Paged, params);
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
    if (!sLLMLoaded)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL padding FMHA kernel module not loaded.");
        return false;
    }
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

    int32_t const ret = cute_dsl_fmha_v2_d256_padding_wrapper(&sLLM_d256Padding, &qTensor, &kTensor, &vTensor, &oTensor,
        &cumSeqlenQ, &cumSeqlenK, attentionScale, mNumHeadsKV, stream);
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
    if (!sLLMLoaded)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL bidirectional FMHA kernel module not loaded.");
        return false;
    }
    if (mHeadDim != 256 || slidingWindowSize < 0)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL bidirectional FMHA requires head_dim=256 and a non-negative left window.");
        return false;
    }

    validateAttentionScale(attentionScale);
    float const scaleQ = 1.0F;
    float const scaleK = 1.0F;
    float const scaleV = 1.0F;
    float const invScaleO = 1.0F;

    using WrapperFn = decltype(&cute_dsl_fmha_v2_d256_bidirectional_wrapper);
    static_assert(WrapperArity<WrapperFn>::value == 16,
        "FMHA-v2 bidirectional wrapper signature changed (module, q_tensor, k_tensor, v_tensor, o_tensor, "
        "cum_seqlen_k, block_begin, block_end, window_size_left, attention_scale, scale_q, scale_k, scale_v, "
        "inv_scale_o, sm_count, stream).");

    auto qTensor = makeBshTensor<WrapperArgT<1, WrapperFn>>(qPtr, mBatchSize, mSeqLenQ, mNumHeadsQ, mHeadDim);
    auto kTensor = makeBshTensor<WrapperArgT<2, WrapperFn>>(kPtr, mBatchSize, mKVSeqLen, mNumHeadsKV, mHeadDim);
    auto vTensor = makeBshTensor<WrapperArgT<3, WrapperFn>>(vPtr, mBatchSize, mKVSeqLen, mNumHeadsKV, mHeadDim);
    auto oTensor = makeBshTensor<WrapperArgT<4, WrapperFn>>(oPtr, mBatchSize, mSeqLenQ, mNumHeadsQ, mHeadDim);
    auto cumSeqlenK = makeCuSeqLenTensor<WrapperArgT<5, WrapperFn>>(cuKVSeqLens, mBatchSize + 1);

    // The per-query block ranges are packed [B, S_q], unlike the Q/K/V/O descriptors above.
    auto blockBeginTensor = makePackedTensor<WrapperArgT<6, WrapperFn>>(blockBegin, {mBatchSize, mSeqLenQ});
    auto blockEndTensor = makePackedTensor<WrapperArgT<7, WrapperFn>>(blockEnd, {mBatchSize, mSeqLenQ});

    int32_t const ret = cute_dsl_fmha_v2_d256_bidirectional_wrapper(&sLLM_d256Bidirectional, &qTensor, &kTensor,
        &vTensor, &oTensor, &cumSeqlenK, &blockBeginTensor, &blockEndTensor, slidingWindowSize, attentionScale, scaleQ,
        scaleK, scaleV, invScaleO, getDeviceMultiProcessorCount(), stream);
    if (ret != 0)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL bidirectional FMHA kernel failed with error code: %d", ret);
    }
    return ret == 0;
}

bool CuteDslFMHAV2Runner::run(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr,
    int32_t const* cuSeqLens, int32_t totalSeqLen, int32_t maxSeqLen, int32_t batchSize, cudaStream_t stream,
    float attentionScale)
{
    if (!sViTLoaded)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL ViT FMHA kernel module not loaded.");
        return false;
    }

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
    case 64: ret = callFmhaV2Vit<cute_dsl_fmha_v2_vit_d64_wrapper>(sViT_d64, params); break;
    case 72: ret = callFmhaV2Vit<cute_dsl_fmha_v2_vit_d72_wrapper>(sViT_d72, params); break;
    case 80: ret = callFmhaV2Vit<cute_dsl_fmha_v2_vit_d80_wrapper>(sViT_d80, params); break;
    case 128: ret = callFmhaV2Vit<cute_dsl_fmha_v2_vit_d128_wrapper>(sViT_d128, params); break;
    default: LOG_ERROR("FMHA-v2 CuTe DSL ViT FMHA: unsupported head_dim=%d", mHeadDim); return false;
    }

    if (ret != 0)
    {
        LOG_ERROR("FMHA-v2 CuTe DSL ViT FMHA kernel (d=%d) failed with error code: %d", mHeadDim, ret);
    }
    return ret == 0;
}

} // namespace trt_edgellm

#endif // CUTE_DSL_FMHA_V2_ENABLED
