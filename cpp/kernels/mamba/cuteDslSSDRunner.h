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
    //! [batch + 1] int32 offsets of packed sequences on one flat token axis of total_tokens tokens; x, dt, B, C and
    //! output are then [total_tokens, ...] and seq_len and context_lengths are unused. nullptr selects the
    //! entry-padded layout. Blackwell path only; every sequence must own at least one token.
    void const* cu_seqlens{};
    int32_t total_tokens{};

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

//! Contiguous packed FP16 SSD inputs with an in-place state pool.
//! cuSeqlens is device int32[batch+1], monotonic from 0 to totalTokens.
//! stateIndices is device int32[batch], unique and in [0, stateSlots).
//! Empty sequences preserve their state; fresh sequences require zeroed slots.
//! D and dtBias are required; dt uses softplus and gating is not fused.
//! All pointers must be 16-byte aligned. Metadata validity is the caller's contract.
struct PackedSSDParams
{
    void* x{};            //!< [totalTokens, nheads, dim], FP16
    void* dt{};           //!< [totalTokens, nheads], FP16
    void* A{};            //!< [nheads], FP32
    void* B{};            //!< [totalTokens, ngroups, dstate], FP16
    void* C{};            //!< [totalTokens, ngroups, dstate], FP16
    void* D{};            //!< [nheads], FP16
    void* dtBias{};       //!< [nheads], FP16
    void* cuSeqlens{};    //!< [batch+1], INT32
    void* stateIndices{}; //!< [batch], INT32
    void* state{};        //!< [stateSlots, nheads, dim, dstate], FP16 in/out
    void* output{};       //!< [totalTokens, nheads, dim], FP16
    void* workspace{};    //!< Caller-owned scratch, at least workspaceSize bytes
    size_t workspaceSize{};
    int32_t totalTokens{};
    int32_t batch{};
    int32_t stateSlots{};
    int32_t nheads{};
    int32_t dim{};
    int32_t dstate{};
    int32_t ngroups{};
    int32_t smVersion{};
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
    //! SM100/101/103/110: dim == 80, dstate == 128 uses one persistent launch with two D64 work tiles.
    static bool canImplement(int32_t dim, int32_t dstate, int32_t smVersion);

    //! Whether packed (cu_seqlens) prefill is available: the Blackwell persistent kernel for (dim, dstate) on
    //! SM100/101/103/110.
    static bool canImplementPacked(int32_t dim, int32_t dstate, int32_t smVersion);

    //! Whether padded prefill has a kernel that does not read the initial state buffer.
    static bool supportsZeroInitialState(int32_t dim, int32_t dstate, int32_t smVersion);

    //! Workspace for any packed call with at most \p maxSeqs sequences on at most \p maxTokens tokens.
    static size_t getPackedWorkspaceSize(int32_t maxSeqs, int32_t maxTokens, int32_t nheads, int32_t dim);

    //! Packed state-pool SM120/121 capability, including whether its AOT modules were built.
    static bool canImplementPackedStatePool(int32_t dim, int32_t dstate, int32_t smVersion);

    //! Bytes for two FP32 [totalTokens,nheads] scratch arrays, each span rounded to 256 bytes.
    static size_t getPackedWorkspaceSize(int32_t totalTokens, int32_t nheads);

    //! Run packed prefill on the explicit stream; warm up once before graph capture.
    //! Validates host descriptors without reading device metadata or synchronizing.
    int32_t runPacked(PackedSSDParams const& params, cudaStream_t stream);

    //! Load only the module selected by \p params. The plugin calls this before
    //! its state copy so a load failure cannot mutate output buffers.
    static bool ensureKernelModules(SSDParams const& params, cudaStream_t stream);

    /// Run SSD prefill (Blackwell path when smVersion >= 100, else Ampere path).
    int run(SSDParams const& params, cudaStream_t stream);

    /// Workspace size in bytes for intermediate buffers.
    static size_t getWorkspaceSize(
        int32_t batch, int32_t seqLen, int32_t nheads, int32_t dim, int32_t dstate, int32_t ngroups);

    /// Maximum workspace for padded matrices whose total physical rows do not exceed maxTokens.
    static size_t getWorkspaceSizeForTokenBudget(
        int32_t maxBatch, int32_t maxTokens, int32_t nheads, int32_t dim, int32_t dstate, int32_t ngroups);

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
