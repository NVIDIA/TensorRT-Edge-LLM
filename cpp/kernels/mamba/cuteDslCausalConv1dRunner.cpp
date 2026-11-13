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

#if defined(CUTE_DSL_CAUSAL_CONV1D_ENABLED)

#include "cuteDslCausalConv1dRunner.h"

#include "common/logger.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <exception>
#include <limits>

namespace trt_edgellm
{

detail::LazyKernelModule<causal_conv1d_fp16_w2_prefill_Kernel_Module_t>
    CuteDslCausalConv1dRunner::sWidth2PrefillModule{};
detail::LazyKernelModule<causal_conv1d_fp16_w3_prefill_Kernel_Module_t>
    CuteDslCausalConv1dRunner::sWidth3PrefillModule{};
detail::LazyKernelModule<causal_conv1d_fp16_w4_prefill_Kernel_Module_t>
    CuteDslCausalConv1dRunner::sWidth4PrefillModule{};
detail::LazyKernelModule<causal_conv1d_fp16_w5_prefill_Kernel_Module_t>
    CuteDslCausalConv1dRunner::sWidth5PrefillModule{};
detail::LazyKernelModule<causal_conv1d_fp16_w6_prefill_Kernel_Module_t>
    CuteDslCausalConv1dRunner::sWidth6PrefillModule{};
detail::LazyKernelModule<causal_conv1d_fp16_w7_prefill_Kernel_Module_t>
    CuteDslCausalConv1dRunner::sWidth7PrefillModule{};
detail::LazyKernelModule<causal_conv1d_fp16_w8_prefill_Kernel_Module_t>
    CuteDslCausalConv1dRunner::sWidth8PrefillModule{};
detail::LazyKernelModule<causal_conv1d_fp16_w2_short_Kernel_Module_t> CuteDslCausalConv1dRunner::sWidth2ShortModule{};
detail::LazyKernelModule<causal_conv1d_fp16_w3_short_Kernel_Module_t> CuteDslCausalConv1dRunner::sWidth3ShortModule{};
detail::LazyKernelModule<causal_conv1d_fp16_w4_short_Kernel_Module_t> CuteDslCausalConv1dRunner::sWidth4ShortModule{};
detail::LazyKernelModule<causal_conv1d_fp16_w5_short_Kernel_Module_t> CuteDslCausalConv1dRunner::sWidth5ShortModule{};
detail::LazyKernelModule<causal_conv1d_fp16_w6_short_Kernel_Module_t> CuteDslCausalConv1dRunner::sWidth6ShortModule{};
detail::LazyKernelModule<causal_conv1d_fp16_w7_short_Kernel_Module_t> CuteDslCausalConv1dRunner::sWidth7ShortModule{};
detail::LazyKernelModule<causal_conv1d_fp16_w8_short_Kernel_Module_t> CuteDslCausalConv1dRunner::sWidth8ShortModule{};

namespace
{

template <typename XTensor, typename CuSeqLensTensor, typename StateIndicesTensor, typename HasInitialStateTensor,
    typename WeightTensor, typename BiasTensor, typename ConvStatesTensor, typename OutTensor, auto Wrapper,
    typename Module>
int32_t launchVariant(
    detail::LazyKernelModule<Module>& module, CuteDslCausalConv1dParams const& params, cudaStream_t stream) noexcept
{
    int32_t const historyLen = params.width - 1;

    XTensor xTensor{};
    xTensor.data = const_cast<void*>(params.x != nullptr ? params.x : params.weight);
    xTensor.dynamic_shapes[0] = params.totalTokens * params.dim;

    CuSeqLensTensor cuSeqLensTensor{};
    cuSeqLensTensor.data = const_cast<int32_t*>(params.cuSeqLens);
    cuSeqLensTensor.dynamic_shapes[0] = params.numSequences + 1;

    StateIndicesTensor stateIndicesTensor{};
    stateIndicesTensor.data = const_cast<int32_t*>(params.stateIndices);
    stateIndicesTensor.dynamic_shapes[0] = params.numSequences;

    HasInitialStateTensor hasInitialStateTensor{};
    hasInitialStateTensor.data = const_cast<int32_t*>(params.hasInitialState);
    hasInitialStateTensor.dynamic_shapes[0] = params.numSequences;

    WeightTensor weightTensor{};
    weightTensor.data = const_cast<void*>(params.weight);
    weightTensor.dynamic_shapes[0] = params.dim * params.width;

    BiasTensor biasTensor{};
    biasTensor.data = const_cast<void*>(params.bias != nullptr ? params.bias : params.weight);
    biasTensor.dynamic_shapes[0] = params.dim;

    ConvStatesTensor convStatesTensor{};
    convStatesTensor.data = params.convStates;
    convStatesTensor.dynamic_shapes[0] = params.numSlots * params.dim * historyLen;

    OutTensor outTensor{};
    outTensor.data = params.out != nullptr ? params.out : params.convStates;
    outTensor.dynamic_shapes[0] = params.totalTokens * params.dim;

    int32_t const result = Wrapper(&module.module, &xTensor, &cuSeqLensTensor, &stateIndicesTensor,
        &hasInitialStateTensor, &weightTensor, &biasTensor, &convStatesTensor, &outTensor, params.totalTokens,
        params.numSequences, params.dim, params.numSlots, params.padSlotId, params.bias != nullptr ? 1 : 0, stream);
    if (result != 0)
    {
        LOG_ERROR("CuteDslCausalConv1dRunner: AOT wrapper failed with code %d for T=%d N=%d dim=%d width=%d", result,
            params.totalTokens, params.numSequences, params.dim, params.width);
        return result;
    }
    cudaError_t const launchError = cudaGetLastError();
    if (launchError != cudaSuccess)
    {
        LOG_ERROR("CuteDslCausalConv1dRunner: kernel launch failed: %s (%s)", cudaGetErrorName(launchError),
            cudaGetErrorString(launchError));
        return -1;
    }
    return 0;
}

bool hasRepresentableFlatShapes(CuteDslCausalConv1dParams const& params) noexcept
{
    int64_t constexpr kMaxDynamicShape = std::numeric_limits<int32_t>::max();
    return static_cast<int64_t>(params.totalTokens) * params.dim <= kMaxDynamicShape
        && static_cast<int64_t>(params.dim) * params.width <= kMaxDynamicShape
        && static_cast<int64_t>(params.numSlots) * params.dim * (params.width - 1) <= kMaxDynamicShape;
}

} // namespace

CuteDslCausalConv1dRunner::Variant CuteDslCausalConv1dRunner::selectVariant(int32_t width) noexcept
{
    switch (width)
    {
    case 2: return Variant::kWidth2;
    case 3: return Variant::kWidth3;
    case 4: return Variant::kWidth4;
    case 5: return Variant::kWidth5;
    case 6: return Variant::kWidth6;
    case 7: return Variant::kWidth7;
    case 8: return Variant::kWidth8;
    default: return Variant::kNone;
    }
}

bool CuteDslCausalConv1dRunner::ensureKernelModule(Variant variant, bool sequenceMode, cudaStream_t stream) noexcept
{
    switch (variant)
    {
    case Variant::kWidth2:
        if (sequenceMode)
        {
            return detail::ensureModuleLoaded<causal_conv1d_fp16_w2_short_Kernel_Module_Load,
                causal_conv1d_fp16_w2_short_Kernel_Module_Unload>(
                sWidth2ShortModule, "causal_conv1d_fp16_w2_short", stream);
        }
        return detail::ensureModuleLoaded<causal_conv1d_fp16_w2_prefill_Kernel_Module_Load,
            causal_conv1d_fp16_w2_prefill_Kernel_Module_Unload>(
            sWidth2PrefillModule, "causal_conv1d_fp16_w2_prefill", stream);
    case Variant::kWidth3:
        if (sequenceMode)
        {
            return detail::ensureModuleLoaded<causal_conv1d_fp16_w3_short_Kernel_Module_Load,
                causal_conv1d_fp16_w3_short_Kernel_Module_Unload>(
                sWidth3ShortModule, "causal_conv1d_fp16_w3_short", stream);
        }
        return detail::ensureModuleLoaded<causal_conv1d_fp16_w3_prefill_Kernel_Module_Load,
            causal_conv1d_fp16_w3_prefill_Kernel_Module_Unload>(
            sWidth3PrefillModule, "causal_conv1d_fp16_w3_prefill", stream);
    case Variant::kWidth4:
        if (sequenceMode)
        {
            return detail::ensureModuleLoaded<causal_conv1d_fp16_w4_short_Kernel_Module_Load,
                causal_conv1d_fp16_w4_short_Kernel_Module_Unload>(
                sWidth4ShortModule, "causal_conv1d_fp16_w4_short", stream);
        }
        return detail::ensureModuleLoaded<causal_conv1d_fp16_w4_prefill_Kernel_Module_Load,
            causal_conv1d_fp16_w4_prefill_Kernel_Module_Unload>(
            sWidth4PrefillModule, "causal_conv1d_fp16_w4_prefill", stream);
    case Variant::kWidth5:
        if (sequenceMode)
        {
            return detail::ensureModuleLoaded<causal_conv1d_fp16_w5_short_Kernel_Module_Load,
                causal_conv1d_fp16_w5_short_Kernel_Module_Unload>(
                sWidth5ShortModule, "causal_conv1d_fp16_w5_short", stream);
        }
        return detail::ensureModuleLoaded<causal_conv1d_fp16_w5_prefill_Kernel_Module_Load,
            causal_conv1d_fp16_w5_prefill_Kernel_Module_Unload>(
            sWidth5PrefillModule, "causal_conv1d_fp16_w5_prefill", stream);
    case Variant::kWidth6:
        if (sequenceMode)
        {
            return detail::ensureModuleLoaded<causal_conv1d_fp16_w6_short_Kernel_Module_Load,
                causal_conv1d_fp16_w6_short_Kernel_Module_Unload>(
                sWidth6ShortModule, "causal_conv1d_fp16_w6_short", stream);
        }
        return detail::ensureModuleLoaded<causal_conv1d_fp16_w6_prefill_Kernel_Module_Load,
            causal_conv1d_fp16_w6_prefill_Kernel_Module_Unload>(
            sWidth6PrefillModule, "causal_conv1d_fp16_w6_prefill", stream);
    case Variant::kWidth7:
        if (sequenceMode)
        {
            return detail::ensureModuleLoaded<causal_conv1d_fp16_w7_short_Kernel_Module_Load,
                causal_conv1d_fp16_w7_short_Kernel_Module_Unload>(
                sWidth7ShortModule, "causal_conv1d_fp16_w7_short", stream);
        }
        return detail::ensureModuleLoaded<causal_conv1d_fp16_w7_prefill_Kernel_Module_Load,
            causal_conv1d_fp16_w7_prefill_Kernel_Module_Unload>(
            sWidth7PrefillModule, "causal_conv1d_fp16_w7_prefill", stream);
    case Variant::kWidth8:
        if (sequenceMode)
        {
            return detail::ensureModuleLoaded<causal_conv1d_fp16_w8_short_Kernel_Module_Load,
                causal_conv1d_fp16_w8_short_Kernel_Module_Unload>(
                sWidth8ShortModule, "causal_conv1d_fp16_w8_short", stream);
        }
        return detail::ensureModuleLoaded<causal_conv1d_fp16_w8_prefill_Kernel_Module_Load,
            causal_conv1d_fp16_w8_prefill_Kernel_Module_Unload>(
            sWidth8PrefillModule, "causal_conv1d_fp16_w8_prefill", stream);
    case Variant::kNone: return false;
    }
    return false;
}

int32_t CuteDslCausalConv1dRunner::run(CuteDslCausalConv1dParams const& params, cudaStream_t stream) noexcept
{
    try
    {
        if (params.cuSeqLens == nullptr || params.stateIndices == nullptr || params.hasInitialState == nullptr
            || params.weight == nullptr || params.convStates == nullptr
            || (params.totalTokens > 0 && (params.x == nullptr || params.out == nullptr)))
        {
            LOG_ERROR("CuteDslCausalConv1dRunner: required tensor pointers must be non-null");
            return -1;
        }
        if (params.totalTokens < 0 || params.numSequences <= 0 || params.dim <= 0 || params.numSlots <= 0)
        {
            LOG_ERROR("CuteDslCausalConv1dRunner: invalid shape T=%d N=%d dim=%d slots=%d", params.totalTokens,
                params.numSequences, params.dim, params.numSlots);
            return -1;
        }
        if (!hasRepresentableFlatShapes(params))
        {
            LOG_ERROR("CuteDslCausalConv1dRunner: flattened tensor size exceeds INT32_MAX");
            return -1;
        }

        // Sequence mode walks each sequence's tokens serially; past 64 tokens per sequence the token-parallel
        // prefill mode is faster.
        int32_t constexpr kMaxSequenceModeTokens{64};
        bool const sequenceMode
            = params.totalTokens <= static_cast<int64_t>(params.numSequences) * kMaxSequenceModeTokens;
        Variant const variant = selectVariant(params.width);
        if (variant == Variant::kNone || !ensureKernelModule(variant, sequenceMode, stream))
        {
            LOG_ERROR("CuteDslCausalConv1dRunner: no loadable AOT variant for width=%d", params.width);
            return -1;
        }

#define LAUNCH_WIDTH_MODE(WIDTH, MODE, MODE_NAME)                                                                      \
    return launchVariant<causal_conv1d_fp16_w##WIDTH##_##MODE##_Tensor_x_t,                                            \
        causal_conv1d_fp16_w##WIDTH##_##MODE##_Tensor_cu_seqlens_t,                                                    \
        causal_conv1d_fp16_w##WIDTH##_##MODE##_Tensor_state_indices_t,                                                 \
        causal_conv1d_fp16_w##WIDTH##_##MODE##_Tensor_has_initial_state_t,                                             \
        causal_conv1d_fp16_w##WIDTH##_##MODE##_Tensor_weight_t, causal_conv1d_fp16_w##WIDTH##_##MODE##_Tensor_bias_t,  \
        causal_conv1d_fp16_w##WIDTH##_##MODE##_Tensor_conv_states_t,                                                   \
        causal_conv1d_fp16_w##WIDTH##_##MODE##_Tensor_out_t, cute_dsl_causal_conv1d_fp16_w##WIDTH##_##MODE##_wrapper,  \
        causal_conv1d_fp16_w##WIDTH##_##MODE##_Kernel_Module_t>(sWidth##WIDTH##MODE_NAME##Module, params, stream)

#define LAUNCH_WIDTH(WIDTH)                                                                                            \
    if (sequenceMode)                                                                                                  \
    {                                                                                                                  \
        LAUNCH_WIDTH_MODE(WIDTH, short, Short);                                                                        \
    }                                                                                                                  \
    LAUNCH_WIDTH_MODE(WIDTH, prefill, Prefill)

        switch (variant)
        {
        case Variant::kWidth2: LAUNCH_WIDTH(2);
        case Variant::kWidth3: LAUNCH_WIDTH(3);
        case Variant::kWidth4: LAUNCH_WIDTH(4);
        case Variant::kWidth5: LAUNCH_WIDTH(5);
        case Variant::kWidth6: LAUNCH_WIDTH(6);
        case Variant::kWidth7: LAUNCH_WIDTH(7);
        case Variant::kWidth8: LAUNCH_WIDTH(8);
        case Variant::kNone: break;
        }

#undef LAUNCH_WIDTH
#undef LAUNCH_WIDTH_MODE
    }
    catch (std::exception const& exception)
    {
        LOG_ERROR("CuteDslCausalConv1dRunner: exception: %s", exception.what());
        return -1;
    }
    catch (...)
    {
        LOG_ERROR("CuteDslCausalConv1dRunner: unknown exception");
        return -1;
    }
    return -1;
}

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_CAUSAL_CONV1D_ENABLED)
