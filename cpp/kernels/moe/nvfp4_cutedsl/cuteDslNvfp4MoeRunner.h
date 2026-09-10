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

#ifdef CUTE_DSL_NVFP4_FUSED_MOE_ENABLED

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

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{

//! Activation variants exposed by the fused NVFP4 MoE CuTeDSL kernel family.
//! All six activations (identity, silu, swiglu, gelu, relu2, geglu) are wired through the runner:
//! each is backed by its own AOT module (KernelVariant rows in kernelSrcs/build_cutedsl.py)
//! and loaded lazily by the decode/prefill switch ladders.
enum class CuteDslMoeActivation : int32_t
{
    kIdentity = 0,
    kSiLU = 1,
    kSwiGLU = 2,
    kGeLU = 3,
    kReLU2 = 4,
    kGeGLU = 5,
};

//! Decode vs prefill backend selection. kAuto picks decode when
//! num_tokens*top_k <= kDecodePrefillCutoverRoutedRows, else prefill -- mirrors
//! select_sm120_moe_backend in kernelSrcs/nvfp4_fused_moe_cutedsl/moe_dispatch.py.
//!
//! Both backends' AOT binaries are shape-polymorphic in K / N / E / top_k
//! (the wrapper builds cute.Tensor objects from pointers + runtime Int32
//! extents at launch). The runner currently dispatches the n128 MMA N-tile
//! variant; add another dispatch axis when n256 is re-enabled.
enum class CuteDslMoeBackend : int32_t
{
    kAuto = 0,
    kDecode = 1,
    kPrefill = 2,
};

//! Hidden-state dtype. now only accepts kFP16; BF16 kernels are deferred until the AOT scripts
//! regain an --io_dtype flag and the artifact group is rebuilt with both dtypes.
enum class CuteDslMoeIoDtype : int32_t
{
    kBF16 = 0,
    kFP16 = 1,
};

//! Per-launch parameters for CuteDslNvfp4MoeRunner::run. Device pointers are non-owning —
//! the plugin / caller is responsible for lifetime and shape validation.
struct CuteDslNvfp4MoeParams
{
    int32_t numTokens{};     //!< T = B*S routed tokens.
    int32_t numExperts{};    //!< E — total expert count; matches state_E and weight_E in v1.
    int32_t topK{};          //!< Top-K per token.
    int32_t hiddenSize{};    //!< H (kernel's K dim).
    int32_t moeInterSize{};  //!< I (kernel's N dim, per expert).
    int32_t maxRoutedRows{}; //!< Upper bound on max rows per expert; sets workspace padding.

    // Inputs (device). All weights use native CuTeDSL layouts; see
    // cpp/plugins/nvfp4MoePlugin/README.md for the exact shapes.
    void const* hiddenStates{}; //!< FP16 [B,S,H] row-major.
    int32_t const* topkIds{};   //!< int32 [T*K].
    float const* topkWeights{}; //!< fp32  [T*K] (renormalized, sums to 1 per row).

    void const* fc1QWeights{};    //!< int8 [E, N1, H/2]; N1 = 2*I for swiglu else I.
    void const* fc1BlocksScale{}; //!< int8 bytes of FP8 E4M3 SF in 6D MMA layout.
    float const* fc1Alpha{};      //!< fp32 [E].

    void const* fc2QWeights{};    //!< int8 [E, H, I/2].
    void const* fc2BlocksScale{}; //!< int8 bytes of FP8 E4M3 SF in 6D MMA layout.
    float const* fc2Alpha{};      //!< fp32 [E].

    float const* inputGlobalScale{}; //!< fp32 [E] — per-expert FC1 activation global scale.
    float const* downInputScale{};   //!< fp32 [E] — per-expert FC2 activation global scale.

    //! int32 [numExperts] identity table [0, 1, ..., E-1] used as weight_expert_ids
    //! by the decode wrapper. Non-owning; caller must keep the buffer alive for the
    //! whole launch and ensure it does not alias \c globalToLocalExpertIds. The
    //! plugin owns this table via \c IGpuAllocator and frees it at destruction —
    //! the runner does not allocate or grow it on the hot path.
    int32_t const* weightExpertIds{};
    //! int32 [numExperts] identity table [0, 1, ..., E-1] used as
    //! global_to_local_expert by the decode wrapper. Must not alias
    //! \c weightExpertIds (the kernel collapses cos_sim if they alias).
    int32_t const* globalToLocalExpertIds{};

    // Output (device).
    void* output{}; //!< FP16 [B,S,H] row-major; runner will zero before scatter-add.

    // Configuration.
    CuteDslMoeActivation activation{CuteDslMoeActivation::kSwiGLU};
    CuteDslMoeIoDtype ioDtype{CuteDslMoeIoDtype::kFP16};
    CuteDslMoeBackend backend{CuteDslMoeBackend::kAuto};
};

//! Runner for the SM120/SM121 fused NVFP4 MoE kernel (decode + prefill backends).
//!
//! Each (activation x backend x N-tile) module is loaded on its first selected
//! dispatch and then remains resident for process lifetime. Shape axes H / I /
//! E / top_k and the batch / sequence dims are runtime; H must be a positive
//! multiple of kHiddenSizeAlignment (= kCuteDslTileK * kStaticAbStage) so the
//! K-tile pipeline drains cleanly. canImplement() enforces this contract.
class CuteDslNvfp4MoeRunner
{
public:
    //! Decode-vs-prefill cutover threshold on num_tokens*top_k. Matches
    //! _DECODE_PREFILL_CUTOVER_PAIRS in moe_dispatch.py.
    static constexpr int32_t kDecodePrefillCutoverRoutedRows = 640;

    //! Prefill backend slice-chunk multiplier (_PREFILL_SLICE_CHUNK).
    static constexpr int32_t kPrefillSliceChunk = 2;

    //! NVFP4 block-scale group size (K dim grouping).
    static constexpr int32_t kNvfp4SfVecSize = 16;

    //! Alignment of max_rows to multiples of 128 (MMA tile-M alignment).
    static constexpr int32_t kRowTileAlign = 128;

    //! MMA tile N -- current compile-time N-tile variant (n128). Weights
    //! must have I (moeInterSize) that is a multiple of this tile.
    static constexpr int32_t kLevelTileN = 128;

    //! Largest planned MMA N-tile variant (n256). Dispatch is disabled until
    //! the kernel-side n256 indexing issue is fixed.
    static constexpr int32_t kLevelTileNLarge = 256;

    //! MMA tile M. Decode routes only top_k rows per token, so the default
    //! 128-row tile runs mostly empty; 64 is the scheduler's minimum (it
    //! asserts divisibility by 64). Measured on SM121 with E=256/I=512/top_k=8:
    //! m64 is 12.9-15.2% faster than m128 over routed_rows 8..56.
    static constexpr int32_t kLevelTileM = 128;
    static constexpr int32_t kLevelTileMSmall = 64;

    //! Upper bound of the routed_rows range where m64 was measured. Past it we
    //! keep m128 rather than extrapolate.
    static constexpr int32_t kTileMSmallMaxRows = 128;

    //! MMA tile-K (compile-time, bolted to NVFP4 block-scale geometry).
    //! Mirrors ``tile_k = sf_vec_size * 8`` in moe_{decode,prefill}_kernel.py.
    static constexpr int32_t kCuteDslTileK = kNvfp4SfVecSize * 8;

    //! AOT-baked mainloop pipeline depth (number of A/B operand smem buffers).
    //! MUST mirror ``_AB_STAGE_DEFAULT`` in moe_{decode,prefill}_kernel.py;
    //! changing it requires a kernel rebuild AND updating kHiddenSizeAlignment.
    static constexpr int32_t kStaticAbStage = 2;

    //! Hidden-size divisibility contract: ensures ``(H / kCuteDslTileK)`` is
    //! divisible by ``kStaticAbStage`` so the K-tile pipeline drains cleanly
    //! (no phase mismatch). Equivalent to ``H % 256 == 0`` with the current
    //! constants. Validated for H in {1024, 2048}; other multiples of
    //! kHiddenSizeAlignment are expected to work by shape polymorphism.
    static constexpr int32_t kHiddenSizeAlignment = kCuteDslTileK * kStaticAbStage;

    CuteDslNvfp4MoeRunner() = default;
    ~CuteDslNvfp4MoeRunner() = default;
    CuteDslNvfp4MoeRunner(CuteDslNvfp4MoeRunner const&) = delete;
    CuteDslNvfp4MoeRunner& operator=(CuteDslNvfp4MoeRunner const&) = delete;

    //! True if the given dimensions and hardware can use this runner. The AOT
    //! kernels are shape-polymorphic in H / N / E / top_k (wrappers build
    //! cute.Tensor layouts from runtime Int32 extents). hiddenSize must be a
    //! positive multiple of kHiddenSizeAlignment; other dims must meet the
    //! alignment / divisibility rules compatible with the compile-time MMA
    //! tile and the NVFP4 block-scale vector size.
    static bool canImplement(int32_t hiddenSize, int32_t moeInterSize, int32_t numExperts, int32_t topK,
        int32_t smVersion, CuteDslMoeActivation activation, CuteDslMoeIoDtype ioDtype, CuteDslMoeBackend backend);

    //! Load only the activation/backend module selected by \p params. The plugin
    //! calls this before routing so failure cannot mutate device buffers.
    static bool ensureKernelModules(CuteDslNvfp4MoeParams const& params, cudaStream_t stream);

    //! Workspace size (bytes) upper bound for the given runtime caps.
    //!
    //! Includes (a) the top-k softmax scratch reused from kernel::moeTopkSoftmax (which runs
    //! before the fused kernel — the kernel takes pre-computed topk_ids / topk_weights),
    //! and (b) every buffer enumerated in allocate_sm120_decode_workspace +
    //! allocate_sm120_prefill_workspace in moe_dispatch.py. Returns the max of the two
    //! backends so backend == kAuto can pick either at runtime without re-querying.
    //! \p backend sizes the decode sub-layout appropriately: under kAuto the decode backend is
    //! only ever selected for num_tokens*top_k <= kDecodePrefillCutoverRoutedRows, so the decode
    //! workspace is capped at the cutover instead of the (potentially huge) profile-wide
    //! maxRoutedRows. See decodeCapRoutedRows().
    static size_t getWorkspaceSize(int32_t maxNumTokens, int32_t maxRoutedRows, int32_t numExperts, int32_t topK,
        int32_t hiddenSize, int32_t moeInterSize, CuteDslMoeBackend backend);

    //! Dispatch one fused MoE launch. Returns 0 on success, non-zero on dispatch failure
    //! (unsupported dim, missing module, kernel returned non-zero).
    //! workspace must point to at least getWorkspaceSize(...) bytes.
    int32_t run(CuteDslNvfp4MoeParams const& params, void* workspace, cudaStream_t stream);

private:
    //! Pick decode/prefill for backend == kAuto. Mirrors select_sm120_moe_backend in
    //! moe_dispatch.py: decode when num_tokens*top_k <= kDecodePrefillCutoverRoutedRows,
    //! else prefill. Explicit kDecode / kPrefill are propagated as-is.
    static CuteDslMoeBackend resolveBackend(CuteDslMoeBackend backend, int32_t numTokens, int32_t topK);

    //! Upper bound on the routed rows the decode backend can ever process, used to size BOTH
    //! the decode workspace allocation (getWorkspaceSize) and the per-launch decode layout
    //! (runDecode) so they stay consistent. kPrefill never dispatches decode (minimal); kAuto is
    //! bounded by kDecodePrefillCutoverRoutedRows; explicit kDecode bypasses the cutover and is
    //! bounded only by the profile-wide maxRoutedRows.
    static int32_t decodeCapRoutedRows(CuteDslMoeBackend backend, int32_t maxRoutedRows);

    //! Pick the compile-time MMA N-tile variant. Currently returns n128 only;
    //! N must already be a multiple of kLevelTileN (enforced by canImplement).
    static int32_t selectMmaTilerN(int32_t moeInterSize);

    //! Picks the MMA M-tile from the routed-row count. Decode-only: the
    //! prefill backend keeps m128 because it already fills the tile.
    static int32_t selectMmaTilerM(int32_t routedRows);

    //! Decode-backend launch -- dispatches to the activation x N-tile variant.
    int32_t runDecode(CuteDslNvfp4MoeParams const& params, void* workspace, cudaStream_t stream);

    //! Prefill-backend launch -- dispatches to the activation x N-tile variant.
    //! Workspace must be at least getWorkspaceSize(...) bytes (which is max(decode, prefill)).
    int32_t runPrefill(CuteDslNvfp4MoeParams const& params, void* workspace, cudaStream_t stream);

    // AOT kernel modules -- FP16 io_dtype, all six activations
    // (identity/silu/swiglu/gelu/relu2/geglu) x both backends (decode/prefill) x n128 = 12 modules.
    static detail::LazyKernelModule<nvfp4_fused_moe_decode_identity_m128_n128_Kernel_Module_t>
        sDecodeIdentity_m128_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_decode_silu_m128_n128_Kernel_Module_t> sDecodeSiLU_m128_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_decode_swiglu_m128_n128_Kernel_Module_t> sDecodeSwiGLU_m128_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_decode_gelu_m128_n128_Kernel_Module_t> sDecodeGeLU_m128_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_decode_relu2_m128_n128_Kernel_Module_t> sDecodeReLU2_m128_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_decode_geglu_m128_n128_Kernel_Module_t> sDecodeGeGLU_m128_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_decode_identity_m64_n128_Kernel_Module_t> sDecodeIdentity_m64_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_decode_silu_m64_n128_Kernel_Module_t> sDecodeSiLU_m64_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_decode_swiglu_m64_n128_Kernel_Module_t> sDecodeSwiGLU_m64_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_decode_gelu_m64_n128_Kernel_Module_t> sDecodeGeLU_m64_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_decode_relu2_m64_n128_Kernel_Module_t> sDecodeReLU2_m64_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_decode_geglu_m64_n128_Kernel_Module_t> sDecodeGeGLU_m64_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_prefill_identity_n128_Kernel_Module_t> sPrefillIdentity_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_prefill_silu_n128_Kernel_Module_t> sPrefillSiLU_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_prefill_swiglu_n128_Kernel_Module_t> sPrefillSwiGLU_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_prefill_gelu_n128_Kernel_Module_t> sPrefillGeLU_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_prefill_relu2_n128_Kernel_Module_t> sPrefillReLU2_n128;
    static detail::LazyKernelModule<nvfp4_fused_moe_prefill_geglu_n128_Kernel_Module_t> sPrefillGeGLU_n128;
};

} // namespace trt_edgellm

#endif // CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
