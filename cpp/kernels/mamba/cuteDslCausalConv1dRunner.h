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

#if defined(CUTE_DSL_CAUSAL_CONV1D_ENABLED)

#include "kernels/cuteDslModuleLoader.h"

#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_causal_conv1d_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{

struct CuteDslCausalConv1dParams
{
    void const* x{};
    int32_t const* cuSeqLens{};
    int32_t const* stateIndices{};
    int32_t const* hasInitialState{};
    void const* weight{};
    void const* bias{};
    void* convStates{};
    void* out{};
    int32_t totalTokens{};
    int32_t numSequences{};
    int32_t dim{};
    int32_t width{};
    int32_t numSlots{};
    int32_t padSlotId{};
};

class CuteDslCausalConv1dRunner
{
public:
    CuteDslCausalConv1dRunner() = delete;

    static int32_t run(CuteDslCausalConv1dParams const& params, cudaStream_t stream) noexcept;

private:
    enum class Variant : int32_t
    {
        kNone,
        kWidth2,
        kWidth3,
        kWidth4,
        kWidth5,
        kWidth6,
        kWidth7,
        kWidth8,
    };

    static Variant selectVariant(int32_t width) noexcept;
    static bool ensureKernelModule(Variant variant, bool sequenceMode, cudaStream_t stream) noexcept;

    static detail::LazyKernelModule<causal_conv1d_fp16_w2_prefill_Kernel_Module_t> sWidth2PrefillModule;
    static detail::LazyKernelModule<causal_conv1d_fp16_w3_prefill_Kernel_Module_t> sWidth3PrefillModule;
    static detail::LazyKernelModule<causal_conv1d_fp16_w4_prefill_Kernel_Module_t> sWidth4PrefillModule;
    static detail::LazyKernelModule<causal_conv1d_fp16_w5_prefill_Kernel_Module_t> sWidth5PrefillModule;
    static detail::LazyKernelModule<causal_conv1d_fp16_w6_prefill_Kernel_Module_t> sWidth6PrefillModule;
    static detail::LazyKernelModule<causal_conv1d_fp16_w7_prefill_Kernel_Module_t> sWidth7PrefillModule;
    static detail::LazyKernelModule<causal_conv1d_fp16_w8_prefill_Kernel_Module_t> sWidth8PrefillModule;
    static detail::LazyKernelModule<causal_conv1d_fp16_w2_short_Kernel_Module_t> sWidth2ShortModule;
    static detail::LazyKernelModule<causal_conv1d_fp16_w3_short_Kernel_Module_t> sWidth3ShortModule;
    static detail::LazyKernelModule<causal_conv1d_fp16_w4_short_Kernel_Module_t> sWidth4ShortModule;
    static detail::LazyKernelModule<causal_conv1d_fp16_w5_short_Kernel_Module_t> sWidth5ShortModule;
    static detail::LazyKernelModule<causal_conv1d_fp16_w6_short_Kernel_Module_t> sWidth6ShortModule;
    static detail::LazyKernelModule<causal_conv1d_fp16_w7_short_Kernel_Module_t> sWidth7ShortModule;
    static detail::LazyKernelModule<causal_conv1d_fp16_w8_short_Kernel_Module_t> sWidth8ShortModule;
};

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_CAUSAL_CONV1D_ENABLED)
