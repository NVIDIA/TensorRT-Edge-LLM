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

#include "compactPhaseSupport.h"

#include "common/logger.h"
#include "kernels/contextAttentionKernels/cuteDslFMHAV2Runner.h"
#include "kernels/decodeAttentionKernels/ctmDecodeAttentionRunner.h"
#include "kernels/decodeAttentionKernels/xqaKernelSupport.h"
#include "kernels/mamba/causalConv1d.h"

#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
#include "kernels/contextAttentionKernels/cuteDslFMHARunner.h"
#endif

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iterator>
#include <string>

namespace trt_edgellm
{

bool isCtmOnlySM(int32_t smVersion) noexcept
{
    return smVersion == 100 || smVersion == 101 || smVersion == 103 || smVersion == 110;
}

int32_t thorRenumberedSm(int32_t smVersion) noexcept
{
    constexpr int32_t kTHOR_SM{110};
    constexpr int32_t kTHOR_KERNEL_SM{101};
    return smVersion == kTHOR_SM ? kTHOR_KERNEL_SM : smVersion;
}

bool ctmDecodeDisabledByEnv(int32_t smVersion) noexcept
{
    static bool const sRequested = std::getenv("EDGELLM_DISABLE_CTM_DECODE") != nullptr;
    return sRequested && !isCtmOnlySM(smVersion);
}

bool cuteDslArtifactCarriesSm([[maybe_unused]] int32_t smVersion) noexcept
{
#if defined(CUTE_DSL_ARTIFACT_SMS)
    constexpr int32_t kARTIFACT_SMS[]{CUTE_DSL_ARTIFACT_SMS};
    return std::find(std::begin(kARTIFACT_SMS), std::end(kARTIFACT_SMS), smVersion) != std::end(kARTIFACT_SMS);
#else
    return false;
#endif
}

bool decodesOnCtmOnly(int32_t smVersion, int32_t tpSize) noexcept
{
    return isCtmOnlySM(smVersion) && tpSize == 1;
}

bool canUseCtmDecode([[maybe_unused]] AttentionLayerTraits const& layer) noexcept
{
#if defined(CUTE_DSL_FMHA_DECODE_CTM_ENABLED)
    // Bounded-SWA page tables (retired slots = -1, pool of live pages only) are safe: the window kernels
    // touch no page before (seq_len - window) / kTOKENS_PER_PAGE, which is exactly the retained range.
    // The CTM kernels are AOT-built only for the CTM-only SMs (`supported_sms` of the ctm_decode_* variants in
    // kernelSrcs/build_cutedsl.py), yet a multi-SM artifact such as sm_110_sm_120 defines the macro above for its SM110
    // kernels. On any other SM (e.g. SM120) its dispatcher aborts the process, so decode must not select CTM there.
    return isCtmOnlySM(layer.smVersion) && !ctmDecodeDisabledByEnv(layer.smVersion) && layer.tpSize == 1
        && layer.dataType == trt_edgellm::DataType::kHALF
        && CtmDecodeAttentionRunner::supported(layer.numQHeads, layer.numKVHeads, layer.headSize, layer.fp8KVCache,
            layer.slidingWindow, layer.attentionSink);
#else
    return false;
#endif
}

bool canUseXqaDecode(AttentionLayerTraits const& layer) noexcept
{
    return canCompileXQAKernel(layer.numQHeads, layer.numKVHeads, layer.headSize, thorRenumberedSm(layer.smVersion),
        layer.dataType, layer.fp8KVCache ? trt_edgellm::DataType::kFP8 : trt_edgellm::DataType::kHALF);
}

CompactPrefillKernel selectCompactPrefillKernel(AttentionLayerTraits const& layer) noexcept
{
    int32_t const sm = thorRenumberedSm(layer.smVersion);
    auto const maskType = layer.visionBlockAttention ? CuteDslFMHAMaskType::kVISION_BLOCK
        : layer.slidingWindow                        ? CuteDslFMHAMaskType::kSLIDING_CAUSAL
                                                     : CuteDslFMHAMaskType::kCAUSAL;
    if (layer.attentionSink)
    {
        if (!isCtmOnlySM(sm)
            && CuteDslFMHAV2Runner::canImplementPagedRagged(
                layer.numQHeads, layer.numKVHeads, layer.headSize, sm, layer.dataType, maskType))
        {
            return CompactPrefillKernel::kCUTE_DSL_FMHA_V2;
        }
        return CompactPrefillKernel::kNONE;
    }
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
    // Bounded-SWA-capable layers and FP8 shared-KV layers prefill on FMHA-v2 in phases 1/2 as well
    // (allowPagedBlackwell, useSharedFp8FMHAV2). The Blackwell ragged kernels read FP8 Q and KV directly.
    if (!layer.supportsBoundedKVCache && !(layer.fp8KVCache && layer.kvShared)
        && CuteDslFMHARunner::canImplementPagedRagged(layer.numQHeads, layer.numKVHeads, layer.headSize, sm,
            layer.fp8KVCache ? trt_edgellm::DataType::kFP8 : layer.dataType, maskType))
    {
        return CompactPrefillKernel::kCUTE_DSL_FMHA_BLACKWELL;
    }
#endif
    // FMHA-v2 reads FP16 K/V; an FP8 cache is dequantized into a dense workspace first, as in chunked prefill.
    if (!layer.attentionSink
        && CuteDslFMHAV2Runner::canImplementPagedRagged(
            layer.numQHeads, layer.numKVHeads, layer.headSize, sm, layer.dataType, maskType))
    {
        return CompactPrefillKernel::kCUTE_DSL_FMHA_V2;
    }
    return CompactPrefillKernel::kNONE;
}

CompactPhaseSupport attentionCompactSupport(AttentionLayerTraits const& layer) noexcept
{
    if (layer.contextMaskSelector)
    {
        return {false, "no compact path for the context mask selector"};
    }
    if (layer.treeAttention || layer.visionBlockAttention)
    {
        return {false, "tree and vision-block layers lack compact per-row mask preparation"};
    }
    bool const ctmOnly = decodesOnCtmOnly(layer.smVersion, layer.tpSize);
    bool const canDecode = ctmOnly ? canUseCtmDecode(layer) : (canUseCtmDecode(layer) || canUseXqaDecode(layer));
    if (!canDecode)
    {
        return {false,
            ctmOnly ? "no CTM decode kernel for these head counts, head dim and KV dtype in this build"
                    : "no CTM or XQA decode kernel for these head counts, head dim and KV dtype on this SM"};
    }
    if (selectCompactPrefillKernel(layer) == CompactPrefillKernel::kNONE)
    {
        return {false, "no ragged paged prefill kernel for this SM, head counts and head dim in this build"};
    }
    if (layer.kvShared && layer.supportsBoundedKVCache && layer.qOnlyInput)
    {
        return {false, "a Q-only shared-KV bounded-SWA-capable layer lacks the current K/V the bounded assembly needs"};
    }
    return {true, nullptr};
}

std::string attentionLayerShape(AttentionLayerTraits const& layer)
{
    std::string shape = std::to_string(layer.numQHeads) + " query heads, " + std::to_string(layer.numKVHeads)
        + " KV heads, head dim " + std::to_string(layer.headSize);
    if (layer.fp8KVCache)
    {
        shape += ", FP8 KV cache";
    }
    if (layer.supportsBoundedKVCache)
    {
        shape += ", bounded-SWA-capable";
    }
    if (layer.kvShared)
    {
        shape += ", KV-shared";
    }
    if (layer.tpSize != 1)
    {
        shape += ", TP " + std::to_string(layer.tpSize);
    }
    return shape;
}

CompactPhaseSupport causalConv1dCompactSupport(
    int32_t smVersion, int32_t dim, int32_t width, int32_t dilation, int32_t groups, bool specVerifyState) noexcept
{
    if (specVerifyState)
    {
        return {false, "spec-verify layers have no compact execution path"};
    }
    if (!(groups == 0 || groups == dim))
    {
        return {false, "the packed-varlen conv kernel is depthwise only"};
    }
    try
    {
        if (!mamba_ssm::canImplementPackedVarlen(smVersion, width, dilation))
        {
            return {false, "no packed-varlen causal_conv1d kernel for this SM, width and dilation in this build"};
        }
        if (!cuteDslArtifactCarriesSm(smVersion))
        {
            return {false, "the linked CuTe DSL artifact has no kernels for this SM"};
        }
        return {true, nullptr};
    }
    catch (std::exception const& e)
    {
        LOG_WARNING("causal_conv1d: compact capability query failed (%s); compact execution disabled", e.what());
        return {false, "the packed-varlen causal_conv1d capability query failed"};
    }
}

} // namespace trt_edgellm
