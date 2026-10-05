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

#ifdef CUTE_DSL_GDN_ENABLED

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

/** Device pointers and dimensions; runner fills generated tensor structs from these. */
struct GDNParams
{
    void* q{};
    void* k{};
    void* v{};
    void* a{};
    void* b{};
    void* A_log{};
    void* dt_bias{};
    void* h0_source{};
    void* context_lengths{};   ///< [N] int32 — valid length per batch (decode / prefill; unused in MTP)
    void* state_indices{};     ///< [N] int32 — execution row to resident state-pool row.
    void* cu_seqlens{};        ///< [N+1] int32 — prefix-sum of context_lengths (Blackwell prefill)
    void* h0_scratch{};        ///< [N, hv, k, v] f32 — dense state scratch for SM100/101/110 optimized prefill;
                               ///<   must be provided by the caller on those architectures.
    void* tensormap_scratch{}; ///< Tail-store TMA descriptors (Blackwell GeForce prefill).
    void* o{};

    // Kernel mode is selected by the plugin's explicit execution phase.
    bool use_prefill{false}; ///< Context/chunk/diffusion or multi-row state commit.

    // MTP (multi-token) verify fields — used only when use_mtp == true.
    void* intermediate_states{}; ///< [N, seq_len, HV, K, V] FP32 — per-step h cache for rollback.
                                 ///<   Must be non-null when use_mtp == true.
    bool use_mtp{false};         ///< true → MTP decode path (any seq_len).
    bool enablePdl{false};       ///< Request PDL for the SM120/121 prefill path; the runner applies final gates.

    int32_t n{};
    int32_t seq_len{};
    int32_t h{};
    int32_t hv{};
    int32_t k_dim{};
    int32_t v_dim{};
    int32_t state_pool_rows{};
    int32_t smVersion{}; // GPU SM version for dispatch (e.g. 87, 110)
};

enum class GDNBackend
{
    kDecode,
    kPrefill,
    kPrefillBlackwell,
    kPrefillBlackwellGeforce,
    kDecodeMTP,
};

/** Lazily loads the selected AOT module, fills tensor structs from GDNParams, and calls its generated wrapper.
 *
 *  Dispatch table (evaluated in order):
 *    use_mtp == true              → runDecodeMTP()       (MTP: any seq_len)
 *    use_prefill && SM120/121     → runPrefillBlackwellGeforce() (Blackwell GeForce warp-MMA prefill)
 *    use_prefill && SM100/101/110 → runPrefillBlackwell() (Blackwell prefill)
 *    use_prefill                  → runPrefill()          (sequential prefill)
 *    otherwise                    → runDecode()           (single-token decode)
 *
 *  MTP note: all batch items process seq_len (T) draft tokens uniformly.
 *  context_lengths is not used in MTP mode.
 */
class CuteDslGDNRunner
{
public:
    // Blackwell GeForce only; must match TENSOR_MAP_DESCRIPTOR_BYTES in gdn_prefill_sm12x_helpers.py.
    static constexpr int32_t kBlackwellGeforceTensorMapDescriptorBytes = 128;
    static constexpr int32_t kBlackwellGeforceMaxSMCount = 256;

    CuteDslGDNRunner() = default;
    ~CuteDslGDNRunner() = default;
    CuteDslGDNRunner(CuteDslGDNRunner const&) = delete;
    CuteDslGDNRunner& operator=(CuteDslGDNRunner const&) = delete;

    static bool canImplement(int32_t kDim, int32_t vDim, int32_t smVersion);

    static GDNBackend selectBackend(GDNParams const& params);

    //! Load only the module selected by \p params. This is exposed so the plugin can
    //! fail before enqueue-side preprocessing mutates device buffers.
    static bool ensureKernelModules(GDNParams const& params, cudaStream_t stream);

    /** Run GDN kernel. See class-level dispatch table. */
    int run(GDNParams const& params, cudaStream_t stream);

private:
    int runDecode(GDNParams const& params, cudaStream_t stream);
    int runPrefill(GDNParams const& params, cudaStream_t stream);
    int runPrefillBlackwell(GDNParams const& params, cudaStream_t stream);
    int runPrefillBlackwellGeforce(GDNParams const& params, cudaStream_t stream);
    int runDecodeMTP(GDNParams const& params, cudaStream_t stream);

    static detail::LazyKernelModule<gdn_decode_Kernel_Module_t> sDecodeModule;
    static detail::LazyKernelModule<gdn_prefill_Kernel_Module_t> sPrefillModule;
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    static detail::LazyKernelModule<gdn_prefill_blackwell_Kernel_Module_t> sBlackwellPrefillModule;
#endif
#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
    static detail::LazyKernelModule<gdn_prefill_blackwell_geforce_Kernel_Module_t> sBlackwellGeforcePrefillModule;
#endif
    static detail::LazyKernelModule<gdn_decode_mtp_cache_Kernel_Module_t> sMTPDecodeCacheModule;
};

} // namespace trt_edgellm

#endif // CUTE_DSL_GDN_ENABLED
