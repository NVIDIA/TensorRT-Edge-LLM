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

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED

#include "kernels/cuteDslModuleLoader.h"

#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{

//! Per-launch parameters for the SM100/SM101/SM110 decomposed NVFP4 MoE path.
//! Device pointers are non-owning and follow the Nvfp4MoePlugin tensor ABI.
struct CuteDslNvfp4MoeSm110Params
{
    int32_t numTokens{};
    int32_t numExperts{};
    int32_t topK{};
    int32_t hiddenSize{};
    int32_t moeInterSize{};

    void const* hiddenStates{}; //!< FP16 [T, H].
    int32_t const* topkIds{};   //!< INT32 [T, topK].
    float const* topkWeights{}; //!< FP32 [T, topK].

    void const* fc1QWeights{};    //!< INT8 [E, N1, H/2], N1=2*I for swiglu else I.
    void const* fc1BlocksScale{}; //!< INT8 [E, ceil(N1/128), ceil((H/16)/4), 32, 4, 4].
    float const* fc1Alpha{};      //!< FP32 [E].

    void const* fc2QWeights{};    //!< INT8 [E, H, I/2].
    void const* fc2BlocksScale{}; //!< INT8 [E, ceil(H/128), ceil((I/16)/4), 32, 4, 4].
    float const* fc2Alpha{};      //!< FP32 [E].

    float const* inputGlobalScale{}; //!< FP32 [E], per-expert FC1 activation global scale.
    float const* downInputScale{};   //!< FP32 [E], per-expert FC2 activation global scale.

    void* output{}; //!< FP16 [T, H].

    //! Nvfp4MoePlugin activation encoding: 2=swiglu, 4=relu2.
    int32_t activationType{};
};

//! Runner for the SM100/SM101/SM110 decomposed NVFP4 MoE backend.
//!
//! Pipeline:
//!   topK ids/weights provided by the plugin -> GPU layout build ->
//!   routed-row linear-SF FP4 activation pack -> FC1 gather grouped GEMM + activation + FP4 requant ->
//!   zero output -> FC2 finalize/scatter.
class CuteDslNvfp4MoeSm110Runner
{
public:
    static constexpr int32_t kNvfp4SfVecSize = 16;
    static constexpr int32_t kRowTileAlign = 128;
    static constexpr int32_t kLevelTileN = 128;
    static constexpr int32_t kLevelTileNLarge = 256;
    static constexpr int32_t kHiddenSizeAlignment = 128;
    //! The FC1/FC2 CuTe-DSL cubins are runtime-polymorphic in L (num_experts);
    //! the product contract restricts the runner to this discrete set.
    static constexpr int32_t kMaxNumExperts = 512;
    static constexpr int32_t kSupportedNumExperts[] = {128, 256, 512};
    static constexpr int32_t kMaxTopK = 32;

    //! True iff numExperts is one of the product-supported expert counts.
    static constexpr bool isSupportedNumExperts(int32_t numExperts)
    {
        for (int32_t supported : kSupportedNumExperts)
        {
            if (numExperts == supported)
            {
                return true;
            }
        }
        return false;
    }

    CuteDslNvfp4MoeSm110Runner() = default;
    ~CuteDslNvfp4MoeSm110Runner() = default;
    CuteDslNvfp4MoeSm110Runner(CuteDslNvfp4MoeSm110Runner const&) = delete;
    CuteDslNvfp4MoeSm110Runner& operator=(CuteDslNvfp4MoeSm110Runner const&) = delete;

    static bool canImplement(int32_t hiddenSize, int32_t moeInterSize, int32_t numExperts, int32_t topK,
        int32_t smVersion, int32_t activationType, int32_t ioDtype, int32_t backend);

    //! Load the selected FC1 activation module and the shared FC2 module. The
    //! plugin calls this before routing so failure cannot mutate device buffers.
    static bool ensureKernelModules(CuteDslNvfp4MoeSm110Params const& params, cudaStream_t stream);

    static size_t getWorkspaceSize(int32_t maxNumTokens, int32_t maxRoutedRows, int32_t numExperts, int32_t topK,
        int32_t hiddenSize, int32_t moeInterSize);

    int32_t run(CuteDslNvfp4MoeSm110Params const& params, void* workspace, cudaStream_t stream);

private:
    static int32_t selectMmaTilerN(int32_t moeInterSize);

    static detail::LazyKernelModule<nvfp4_moe_sm110_fc1_relu2_n128_Kernel_Module_t> sFC1Relu2N128;
    static detail::LazyKernelModule<nvfp4_moe_sm110_fc1_swiglu_n128_Kernel_Module_t> sFC1SwiGLUN128;
    static detail::LazyKernelModule<nvfp4_moe_sm110_fc1_geglu_n128_Kernel_Module_t> sFC1GeGLUN128;
    static detail::LazyKernelModule<nvfp4_moe_sm110_fc2_n128_fp16_Kernel_Module_t> sFC2N128Fp16;
};

} // namespace trt_edgellm

#endif // CUTE_DSL_NVFP4_MOE_ENABLED
