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

#if defined(CUTE_DSL_F16_MOE_ENABLED)

#include <cuda.h>
#if defined(TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT)
#include <cuda_runtime.h>
#if CUDA_VERSION >= 12000 && CUDA_VERSION < 12080
typedef CUlibrary cudaLibrary_t;
static inline cudaError_t cudaLibraryUnload(cudaLibrary_t lib)
{
    CUresult r = cuLibraryUnload(lib);
    return static_cast<cudaError_t>(r);
}
#endif // CUDA_VERSION >= 12000 && CUDA_VERSION < 12080
#endif // TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT

#include "kernels/cuteDslModuleLoader.h"

#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_f16_moe_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace trt_edgellm
{

//! Per-launch parameters for the cross-platform homogeneous-FP16 MoE path.
struct CuteDslF16MoeParams
{
    int32_t numTokens{};
    int32_t numExperts{};
    int32_t topK{};
    int32_t hiddenSize{};
    int32_t moeInterSize{};
    int32_t activationType{};       //!< 2=SwiGLU, 4=ReLU2.
    int32_t persistentBlockCount{}; //!< Cached on plugin context attachment.

    void const* hiddenStates{}; //!< FP16 [T, H].
    int32_t const* topkIds{};   //!< INT32 [T, topK].
    float const* topkWeights{}; //!< FP32 [T, topK].
    void const* fc1Weights{};   //!< FP16 [E, 2*I, H] for SwiGLU, [E, I, H] for ReLU2.
    void const* fc2Weights{};   //!< FP16 [E, H, I].
    void* output{};             //!< FP16 [T, H].
};

//! One plugin-side runner dispatching target-specific CuTeDSL grouped-GEMM modules.
class CuteDslF16MoeRunner
{
public:
    static constexpr int32_t kFc1NAlignment{128};
    //! Expert count is a runtime kernel argument (group_count); the runner
    //! restricts it to this set, capped by export_common.MAX_NUM_EXPERTS.
    static constexpr int32_t kMaxNumExperts{256};
    static constexpr int32_t kSupportedNumExperts[] = {128, 256};
    static constexpr int32_t kMaxTopK{8};
    static constexpr int32_t kHiddenSizeAlignment{128};
    static constexpr int32_t kInterSizeAlignment{64};

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

    CuteDslF16MoeRunner() = delete;

    static bool canImplement(int32_t hiddenSize, int32_t moeInterSize, int32_t numExperts, int32_t topK,
        int32_t smVersion, int32_t activationType) noexcept;

    //! Load only the architecture module selected for \p smVersion. The plugin
    //! calls this before routing so failure cannot mutate device buffers.
    static bool ensureKernelModules(int32_t smVersion, cudaStream_t stream) noexcept;
    static int32_t getPersistentBlockCount() noexcept;

    static size_t getWorkspaceSize(int32_t maxRoutedRows, int32_t numExperts, int32_t hiddenSize, int32_t moeInterSize,
        int32_t activationType) noexcept;

    static int32_t run(CuteDslF16MoeParams const& params, void* workspace, cudaStream_t stream) noexcept;

private:
    enum class Variant : int32_t
    {
        kNone,
        kAmpere,
        kBlackwell,
        kBlackwellGeforce,
    };

#if defined(CUTE_DSL_F16_MOE_AMPERE_ENABLED)
    static detail::LazyKernelModule<f16_moe_ampere_grouped_fp16_Kernel_Module_t> sAmpereModule;
#endif
#if defined(CUTE_DSL_F16_MOE_BLACKWELL_ENABLED)
    static detail::LazyKernelModule<f16_moe_blackwell_grouped_fp16_Kernel_Module_t> sBlackwellModule;
#endif
#if defined(CUTE_DSL_F16_MOE_BLACKWELL_GEFORCE_ENABLED)
    static detail::LazyKernelModule<f16_moe_blackwell_geforce_grouped_fp16_Kernel_Module_t> sBlackwellGeforceModule;
#endif
    static Variant selectVariant(int32_t smVersion) noexcept;
    static std::unordered_map<int32_t, int32_t> sPersistentBlockCounts;
    static std::mutex sPersistentBlockCountsMutex;
};

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_F16_MOE_ENABLED)
