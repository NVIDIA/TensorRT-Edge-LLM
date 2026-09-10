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

#ifdef CUTE_DSL_SSD_ENABLED

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
#include "cutedsl_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{

/// Device pointers and dimensions for the SSD (Mamba2 SSM scan) kernel.
struct SSDParams
{
    void* x{};                     ///< [batch, seq_len, nheads, dim]        input features (fp16)
    void* dt{};                    ///< [batch, seq_len, nheads]             delta/timestep (fp16)
    void* A{};                     ///< [nheads]                             state decay (fp32, negative)
    void* B{};                     ///< [batch, seq_len, ngroups, dstate]    input projection (fp16)
    void* C{};                     ///< [batch, seq_len, ngroups, dstate]    output projection (fp16)
    void* D{};                     ///< [nheads]                             skip connection (fp16, nullable)
    void* dt_bias{};               ///< [nheads]                             dt bias (fp16, nullable)
    void* z{};                     ///< [batch, seq_len, nheads, dim]        gating (fp16, nullable)
    void* state{};                 ///< [batch, nheads, dim, dstate]         SSM state in/out (fp16)
    void* output{};                ///< [batch, seq_len, nheads, dim]        output (fp16)
    void const* context_lengths{}; ///< [batch] int32 actual length per batch (nullptr = uniform = seq_len for all)

    int32_t batch{};
    int32_t seq_len{};
    int32_t nheads{};
    int32_t dim{};
    int32_t dstate{};
    int32_t ngroups{};
    int32_t smVersion{}; ///< GPU SM version for dispatch (e.g. 80, 100)

    bool dt_softplus{false};
    bool has_D{false};
    bool has_z{false};
    /// Caller provides an initial hidden state at chunk 0 (false = zero-state, faster).
    bool has_init_states{false};

    void* workspace{}; ///< Pre-allocated workspace for intermediate buffers
};

/// Loads AOT-compiled SSD kernel, fills tensor structs from SSDParams, calls generated wrapper.
class CuteDslSSDRunner
{
public:
    CuteDslSSDRunner() = default;
    ~CuteDslSSDRunner() = default;
    CuteDslSSDRunner(CuteDslSSDRunner const&) = delete;
    CuteDslSSDRunner& operator=(CuteDslSSDRunner const&) = delete;

    //! Returns true if this runner can handle the given configuration.
    //!
    //! SM80+: dim ∈ {64, 128}, dstate ∈ {64, 128} (SM80 cp.async kernel, runs on all GPUs).
    //! SM100-110: dim == 64 uses the Blackwell persistent kernel.
    //! SM100/101/110: dim == 80, dstate == 128 uses one persistent launch with two D64 work tiles.
    static bool canImplement(int32_t dim, int32_t dstate, int32_t smVersion);

    //! Load only the module selected by \p params. The plugin calls this before
    //! its state copy so a load failure cannot mutate output buffers.
    static bool ensureKernelModules(SSDParams const& params, cudaStream_t stream);

    /// Run SSD prefill (Blackwell path when smVersion >= 100, else Ampere path).
    int run(SSDParams const& params, cudaStream_t stream);

    /// Workspace size in bytes for intermediate buffers.
    static size_t getWorkspaceSize(
        int32_t batch, int32_t seqLen, int32_t nheads, int32_t dim, int32_t dstate, int32_t ngroups);

private:
    int runPrefill(SSDParams const& params, cudaStream_t stream);

#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED
    int runPrefillBlackwell(SSDParams const& params, cudaStream_t stream);
#endif

    // SM80 modules — one per (dim, dstate) combination
    static detail::LazyKernelModule<ssd_prefill_d128_n128_Kernel_Module_t> sD128N128Module;
    static detail::LazyKernelModule<ssd_prefill_d64_n128_Kernel_Module_t> sD64N128Module;
    static detail::LazyKernelModule<ssd_prefill_d128_n64_Kernel_Module_t> sD128N64Module;
    static detail::LazyKernelModule<ssd_prefill_d64_n64_Kernel_Module_t> sD64N64Module;
#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED
    // Two has_init_states variants per (D, N); runner dispatches on params.has_init_states.
    static detail::LazyKernelModule<ssd_prefill_blackwell_d64_n128_Kernel_Module_t> sBlackwellD64N128Module;
    static detail::LazyKernelModule<ssd_prefill_blackwell_d64_n128_init_states_Kernel_Module_t>
        sBlackwellD64N128InitStatesModule;
    static detail::LazyKernelModule<ssd_prefill_blackwell_d80_n128_Kernel_Module_t> sBlackwellD80N128Module;
    static detail::LazyKernelModule<ssd_prefill_blackwell_d80_n128_init_states_Kernel_Module_t>
        sBlackwellD80N128InitStatesModule;
    static detail::LazyKernelModule<ssd_prefill_blackwell_d64_n64_Kernel_Module_t> sBlackwellD64N64Module;
    static detail::LazyKernelModule<ssd_prefill_blackwell_d64_n64_init_states_Kernel_Module_t>
        sBlackwellD64N64InitStatesModule;
#endif
};

} // namespace trt_edgellm

#endif // CUTE_DSL_SSD_ENABLED
