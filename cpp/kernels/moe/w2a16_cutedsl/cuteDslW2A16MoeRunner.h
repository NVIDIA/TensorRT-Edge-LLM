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

#ifdef CUTE_DSL_W2A16_MOE_ENABLED

#if !defined(CUTE_DSL_W2A16_MOE_ARTIFACT_SMS)
#error "CUTE_DSL_W2A16_MOE_ARTIFACT_SMS must list the SMs of the linked target artifact"
#endif

#include "kernels/cuteDslModuleLoader.h"

#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_w2a16_moe_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
//! DIGITS IQ2_XXS/BF16, H=1024, topK=9, ReLU2. All pointers are device pointers.
//! See kernelSrcs/w2a16_moe_cutedsl/README.md for packing and precision.
struct CuteDslW2A16MoeParams
{
    int32_t numTokens{};
    int32_t numExperts{};
    int32_t moeInterSize{};

    void const* hiddenStates{}; //!< BF16 [M,1024].
    void const* fc1QWeights{};  //!< Packed IQ2_XXS int32 payload.
    void const* fc2QWeights{};
    void const* fc1BlocksScale{}; //!< Tiled FP16 bases, exposed as int32 words.
    void const* fc2BlocksScale{};
    float const* unitScales{};  //!< FP32 [E], all ones, prepared once.
    void const* lookupTable{};  //!< 2048-byte pair-selector table.
    int32_t const* topkIds{};   //!< int32 [M,9], every entry in [0,E).
    float const* topkWeights{}; //!< FP32 [M,9], finite.
    void* output{};             //!< BF16 [M,1024]. Must not alias any input/workspace.
};

//! Native runner for the independent IQ2_XXS/BF16 MoE kernel family.
//! Selected AOT modules remain resident for process lifetime.
class CuteDslW2A16MoeRunner
{
public:
    static constexpr int32_t kHiddenSize = 1024;
    static constexpr int32_t kTopK = 9;
    static constexpr int32_t kFusedNumSms = 48;
    static constexpr int32_t kMaxNumTokens = 8192;
    static constexpr int32_t kDecodeMaxTokens = 8;

    CuteDslW2A16MoeRunner() = default;
    ~CuteDslW2A16MoeRunner() = default;
    CuteDslW2A16MoeRunner(CuteDslW2A16MoeRunner const&) = delete;
    CuteDslW2A16MoeRunner& operator=(CuteDslW2A16MoeRunner const&) = delete;

    //! Check hardware support alone: the SM must be in the linked artifact and a fused target needs enough SMs.
    static bool isHardwareSupported(int32_t smVersion, int32_t numSms) noexcept;

    //! Check shape and hardware support without accessing CUDA or device pointers.
    static bool canImplement(
        int32_t numTokens, int32_t numExperts, int32_t moeInterSize, int32_t smVersion, int32_t numSms) noexcept;

    //! Hardware arguments must describe the execution device for preload, workspace sizing and run.
    //! Preload the selected implementation before graph capture.
    //! Tensor pointers and workspace are not accessed; no kernel is launched.
    static bool ensureKernelModules(
        CuteDslW2A16MoeParams const& params, int32_t smVersion, int32_t numSms, cudaStream_t stream) noexcept;

    //! Size workspace for the supplied hardware; return zero for unsupported shapes or hardware.
    //! Workspace must be 256-byte aligned.
    //! Concurrent streams require distinct workspaces; calls on one stream may reuse it.
    static size_t getWorkspaceSize(
        int32_t numTokens, int32_t numExperts, int32_t moeInterSize, int32_t smVersion, int32_t numSms) noexcept;

    //! Returns zero on success or a CUDA error code on failure.
    //! Thor requires standard top-k routing with distinct expert IDs per token on every replay.
    //! Thor (SM101/SM110) uses UMMA; SM120 and SM121 use fused AOT. A combined artifact selects by device SM.
    //! Validates and loads the selected module before mutating workspace or output.
    //! After module preload, performs no allocation or host/device synchronization.
    int32_t run(CuteDslW2A16MoeParams const& params, void* workspace, size_t workspaceBytes, int32_t smVersion,
        int32_t numSms, cudaStream_t stream) noexcept;

private:
#ifdef CUTE_DSL_W2A16_MOE_THOR_ENABLED
    int32_t runUmma(CuteDslW2A16MoeParams const& params, void* workspace, cudaStream_t stream);
    static detail::LazyKernelModule<w2a16_moe_umma_n48_Kernel_Module_t> sUmmaN48;
    static detail::LazyKernelModule<w2a16_moe_umma_n96_Kernel_Module_t> sUmmaN96;
    static detail::LazyKernelModule<w2a16_moe_umma_n192_Kernel_Module_t> sUmmaN192;
#endif
#ifdef CUTE_DSL_W2A16_MOE_FUSED_ENABLED
    int32_t runDecode(CuteDslW2A16MoeParams const& params, void* workspace, int32_t numSms, cudaStream_t stream);
    int32_t runPrefill(CuteDslW2A16MoeParams const& params, void* workspace, int32_t numSms, cudaStream_t stream);

    static detail::LazyKernelModule<w2a16_moe_decode_Kernel_Module_t> sDecode;
    static detail::LazyKernelModule<w2a16_moe_prefill_m8_Kernel_Module_t> sPrefillM8;
    static detail::LazyKernelModule<w2a16_moe_prefill_m16_Kernel_Module_t> sPrefillM16;
    static detail::LazyKernelModule<w2a16_moe_prefill_m32_Kernel_Module_t> sPrefillM32;
    static detail::LazyKernelModule<w2a16_moe_prefill_m48_Kernel_Module_t> sPrefillM48;
    static detail::LazyKernelModule<w2a16_moe_prefill_m64_Kernel_Module_t> sPrefillM64;
#endif
};

} // namespace trt_edgellm
#endif // CUTE_DSL_W2A16_MOE_ENABLED
