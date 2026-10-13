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

#ifdef CUTE_DSL_GDN_ENABLED

#include "cuteDslGDNRunner.h"
#include "gdnKernelUtils.cuh"
#include "gdnPdlConfig.h"

#include "common/cudaUtils.h"
#include "common/logger.h"

#include <cmath>
#include <cstring>

namespace trt_edgellm
{

namespace
{

#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
bool isBlackwellSm(int32_t smVersion)
{
    return smVersion == 100 || smVersion == 101 || smVersion == 110;
}
#endif

#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
bool isBlackwellGeforceSm(int32_t smVersion)
{
    return smVersion == 120 || smVersion == 121;
}
#endif

} // namespace

detail::LazyKernelModule<gdn_decode_Kernel_Module_t> CuteDslGDNRunner::sDecodeModule{};
detail::LazyKernelModule<gdn_prefill_Kernel_Module_t> CuteDslGDNRunner::sPrefillModule{};
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
detail::LazyKernelModule<gdn_prefill_blackwell_Kernel_Module_t> CuteDslGDNRunner::sBlackwellPrefillModule{};
#endif
#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
detail::LazyKernelModule<gdn_prefill_blackwell_geforce_Kernel_Module_t>
    CuteDslGDNRunner::sBlackwellGeforcePrefillModule{};
#endif
detail::LazyKernelModule<gdn_decode_mtp_cache_Kernel_Module_t> CuteDslGDNRunner::sMTPDecodeCacheModule{};

#define SET_4D_TENSOR(tensor, data_ptr, dim0, dim1, dim2, dim3)                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (dim0);                                                                           \
        (tensor).dynamic_shapes[1] = (dim1);                                                                           \
        (tensor).dynamic_shapes[2] = (dim2);                                                                           \
        (tensor).dynamic_shapes[3] = (dim3);                                                                           \
        (tensor).dynamic_strides[0] = static_cast<int64_t>(dim1) * (dim2) * (dim3);                                    \
        (tensor).dynamic_strides[1] = static_cast<int64_t>(dim2) * (dim3);                                             \
        (tensor).dynamic_strides[2] = static_cast<int64_t>(dim3);                                                      \
    } while (0)

#define SET_3D_TENSOR(tensor, data_ptr, dim0, dim1, dim2)                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (dim0);                                                                           \
        (tensor).dynamic_shapes[1] = (dim1);                                                                           \
        (tensor).dynamic_shapes[2] = (dim2);                                                                           \
        (tensor).dynamic_strides[0] = static_cast<int64_t>(dim1) * (dim2);                                             \
        (tensor).dynamic_strides[1] = static_cast<int64_t>(dim2);                                                      \
    } while (0)

#define SET_2D_TENSOR(tensor, data_ptr, dim0, dim1)                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (dim0);                                                                           \
        (tensor).dynamic_shapes[1] = (dim1);                                                                           \
        (tensor).dynamic_strides[0] = static_cast<int64_t>(dim1);                                                      \
    } while (0)

#define SET_1D_TENSOR(tensor, data_ptr, dim0)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (dim0);                                                                           \
    } while (0)

bool CuteDslGDNRunner::canImplement(int32_t kDim, int32_t vDim, int32_t smVersion)
{
    return (smVersion >= 80) && (kDim == 128) && (vDim == 128);
}

GDNBackend CuteDslGDNRunner::selectBackend(GDNParams const& params)
{
    if (params.use_mtp)
    {
        return GDNBackend::kDecodeMTP;
    }
    if (!params.use_prefill)
    {
        return GDNBackend::kDecode;
    }
#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
    if (isBlackwellGeforceSm(params.smVersion) && params.h > 0 && params.hv % params.h == 0)
    {
        return GDNBackend::kPrefillBlackwellGeforce;
    }
#endif
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    if (isBlackwellSm(params.smVersion))
    {
        return GDNBackend::kPrefillBlackwell;
    }
#endif
    return GDNBackend::kPrefill;
}

bool CuteDslGDNRunner::ensureKernelModules(GDNParams const& params, cudaStream_t stream)
{
    switch (selectBackend(params))
    {
    case GDNBackend::kDecodeMTP:
        return detail::ensureModuleLoaded<gdn_decode_mtp_cache_Kernel_Module_Load,
            gdn_decode_mtp_cache_Kernel_Module_Unload>(sMTPDecodeCacheModule, "gdn_decode_mtp_cache", stream);
    case GDNBackend::kDecode:
        return detail::ensureModuleLoaded<gdn_decode_Kernel_Module_Load, gdn_decode_Kernel_Module_Unload>(
            sDecodeModule, "gdn_decode", stream);
    case GDNBackend::kPrefillBlackwellGeforce:
#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
        return detail::ensureModuleLoaded<gdn_prefill_blackwell_geforce_Kernel_Module_Load,
            gdn_prefill_blackwell_geforce_Kernel_Module_Unload>(
            sBlackwellGeforcePrefillModule, "gdn_prefill_blackwell_geforce", stream);
#else
        return false;
#endif
    case GDNBackend::kPrefillBlackwell:
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
        return detail::ensureModuleLoaded<gdn_prefill_blackwell_Kernel_Module_Load,
            gdn_prefill_blackwell_Kernel_Module_Unload>(sBlackwellPrefillModule, "gdn_prefill_blackwell", stream);
#else
        return false;
#endif
    case GDNBackend::kPrefill:
        return detail::ensureModuleLoaded<gdn_prefill_Kernel_Module_Load, gdn_prefill_Kernel_Module_Unload>(
            sPrefillModule, "gdn_prefill", stream);
    }
    return false;
}

int CuteDslGDNRunner::run(GDNParams const& params, cudaStream_t stream)
{
    if (!ensureKernelModules(params, stream))
    {
        return -1;
    }

    switch (selectBackend(params))
    {
    case GDNBackend::kDecodeMTP: return runDecodeMTP(params, stream);
    case GDNBackend::kDecode: return runDecode(params, stream);
    case GDNBackend::kPrefillBlackwellGeforce:
#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
        return runPrefillBlackwellGeforce(params, stream);
#else
        return -1;
#endif
    case GDNBackend::kPrefillBlackwell:
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
        return runPrefillBlackwell(params, stream);
#else
        return -1;
#endif
    case GDNBackend::kPrefill: return runPrefill(params, stream);
    }
    return -1;
}

int CuteDslGDNRunner::runDecode(GDNParams const& params, cudaStream_t stream)
{
    int32_t const n = params.n;
    int32_t const h = params.h;
    int32_t const hv = params.hv;
    int32_t const k = params.k_dim;
    int32_t const v = params.v_dim;

    if (params.seq_len != 1)
    {
        LOG_ERROR("CuTe DSL GDN decode requires seq_len == 1, got %d", params.seq_len);
        return -1;
    }

    gdn_decode_Tensor_q_t qTensor{};
    SET_4D_TENSOR(qTensor, params.q, n, 1, h, k);

    gdn_decode_Tensor_k_t kTensor{};
    SET_4D_TENSOR(kTensor, params.k, n, 1, h, k);

    gdn_decode_Tensor_v_t vTensor{};
    SET_4D_TENSOR(vTensor, params.v, n, 1, hv, v);

    gdn_decode_Tensor_a_t aTensor{};
    SET_3D_TENSOR(aTensor, params.a, n, 1, hv);

    gdn_decode_Tensor_b_t bTensor{};
    SET_3D_TENSOR(bTensor, params.b, n, 1, hv);

    gdn_decode_Tensor_A_log_t A_logTensor{};
    SET_1D_TENSOR(A_logTensor, params.A_log, hv);

    gdn_decode_Tensor_dt_bias_t dt_biasTensor{};
    SET_1D_TENSOR(dt_biasTensor, params.dt_bias, hv);

    gdn_decode_Tensor_h0_source_t h0_sourceTensor{};
    h0_sourceTensor.data = params.h0_source;
    h0_sourceTensor.dynamic_shapes[0] = params.state_pool_rows > 0 ? params.state_pool_rows : n;
    h0_sourceTensor.dynamic_shapes[1] = params.hv;
    h0_sourceTensor.dynamic_strides[0] = static_cast<int64_t>(params.hv) * params.k_dim * params.v_dim;

    gdn_decode_Tensor_context_lengths_t contextLengthsTensor{};
    SET_1D_TENSOR(contextLengthsTensor, params.context_lengths, n);

    gdn_decode_Tensor_state_indices_t stateIndicesTensor{};
    SET_1D_TENSOR(
        stateIndicesTensor, params.state_indices != nullptr ? params.state_indices : params.context_lengths, n);

    gdn_decode_Tensor_o_t oTensor{};
    SET_4D_TENSOR(oTensor, params.o, n, 1, hv, v);

    return cute_dsl_gdn_decode_wrapper(&sDecodeModule.module, &qTensor, &kTensor, &vTensor, &aTensor, &bTensor,
        &A_logTensor, &dt_biasTensor, &h0_sourceTensor, &stateIndicesTensor, params.state_indices != nullptr,
        &contextLengthsTensor, &oTensor, stream);
}

int CuteDslGDNRunner::runPrefill(GDNParams const& params, cudaStream_t stream)
{
    int32_t const n = params.n;
    int32_t const seq_len = params.seq_len;
    int32_t const h = params.h;
    int32_t const hv = params.hv;
    int32_t const k = params.k_dim;
    int32_t const v = params.v_dim;

    gdn_prefill_Tensor_q_t qTensor{};
    SET_4D_TENSOR(qTensor, params.q, n, seq_len, h, k);

    gdn_prefill_Tensor_k_t kTensor{};
    SET_4D_TENSOR(kTensor, params.k, n, seq_len, h, k);

    gdn_prefill_Tensor_v_t vTensor{};
    SET_4D_TENSOR(vTensor, params.v, n, seq_len, hv, v);

    gdn_prefill_Tensor_a_t aTensor{};
    SET_3D_TENSOR(aTensor, params.a, n, seq_len, hv);

    gdn_prefill_Tensor_b_t bTensor{};
    SET_3D_TENSOR(bTensor, params.b, n, seq_len, hv);

    gdn_prefill_Tensor_A_log_t A_logTensor{};
    SET_1D_TENSOR(A_logTensor, params.A_log, hv);

    gdn_prefill_Tensor_dt_bias_t dt_biasTensor{};
    SET_1D_TENSOR(dt_biasTensor, params.dt_bias, hv);

    gdn_prefill_Tensor_h0_source_t h0_sourceTensor{};
    h0_sourceTensor.data = params.h0_source;
    h0_sourceTensor.dynamic_shapes[0] = params.state_pool_rows > 0 ? params.state_pool_rows : n;
    h0_sourceTensor.dynamic_shapes[1] = params.hv;
    h0_sourceTensor.dynamic_strides[0] = static_cast<int64_t>(params.hv) * params.k_dim * params.v_dim;

    gdn_prefill_Tensor_context_lengths_t contextLengthsTensor{};
    SET_1D_TENSOR(contextLengthsTensor, params.context_lengths, n);

    gdn_prefill_Tensor_state_indices_t stateIndicesTensor{};
    SET_1D_TENSOR(
        stateIndicesTensor, params.state_indices != nullptr ? params.state_indices : params.context_lengths, n);

    gdn_prefill_Tensor_o_t oTensor{};
    SET_4D_TENSOR(oTensor, params.o, n, seq_len, hv, v);

    return cute_dsl_gdn_prefill_wrapper(&sPrefillModule.module, &qTensor, &kTensor, &vTensor, &aTensor, &bTensor,
        &A_logTensor, &dt_biasTensor, &h0_sourceTensor, &stateIndicesTensor, params.state_indices != nullptr,
        &contextLengthsTensor, &oTensor, seq_len, stream);
}

int CuteDslGDNRunner::runPrefillBlackwell(GDNParams const& params, cudaStream_t stream)
{
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    int32_t const n = params.n;
    int32_t const seq_len = params.seq_len;
    int32_t const h = params.h;
    int32_t const hv = params.hv;
    int32_t const k = params.k_dim;
    int32_t const v = params.v_dim;

    // L2-normalize Q and K in-place.  The Blackwell kernel expects pre-normalized
    // Q/K so that QK^T, KK^T, state update, and output paths are all consistent.
    launchGdnL2NormQK(params.q, params.k, n, seq_len, h, k, stream);

    // Set up tensor structs for fused Blackwell kernel (g/beta computed inline in warp 7)
    gdn_prefill_blackwell_Tensor_q_t qTensor{};
    SET_4D_TENSOR(qTensor, params.q, n, seq_len, h, k);

    gdn_prefill_blackwell_Tensor_k_t kTensor{};
    SET_4D_TENSOR(kTensor, params.k, n, seq_len, h, k);

    gdn_prefill_blackwell_Tensor_v_t vTensor{};
    SET_4D_TENSOR(vTensor, params.v, n, seq_len, hv, v);

    gdn_prefill_blackwell_Tensor_a_t aTensor{};
    SET_3D_TENSOR(aTensor, params.a, n, seq_len, hv);

    gdn_prefill_blackwell_Tensor_b_t bTensor{};
    SET_3D_TENSOR(bTensor, params.b, n, seq_len, hv);

    // A_log and dt_bias are constant-shape tensors in the Blackwell kernel — data pointer only.
    gdn_prefill_blackwell_Tensor_A_log_t A_logTensor{};
    A_logTensor.data = params.A_log;

    gdn_prefill_blackwell_Tensor_dt_bias_t dt_biasTensor{};
    dt_biasTensor.data = params.dt_bias;

    // The Blackwell MMA computes state = V^T * K in V-major (d_v, d_k) order,
    // while sequential/decode kernels use K-major (d_k, d_v).  We transpose
    // h0 before and after the kernel to bridge this convention mismatch.
    // The kernel is not in-place safe, so we ping-pong between h0_source and h0_scratch.
    size_t const h0ScratchBytes = static_cast<size_t>(n) * hv * k * v * sizeof(float);
    if (params.h0_scratch == nullptr)
    {
        LOG_WARNING(
            "GDN Blackwell prefill: h0_scratch not provided in GDNParams — "
            "caller must allocate [n=%d, hv=%d, k=%d, v=%d] f32 scratch and set params.h0_scratch.",
            n, hv, k, v);
        return -1;
    }

    int32_t const numStateBlocks = n * hv;

    // Step 1: Gather selected resident rows, if needed, and transpose K-major state into dense V-major scratch.
    if (params.state_indices != nullptr)
    {
        launchGdnStateGatherTranspose(
            params.h0_source, params.h0_scratch, params.state_indices, n, params.state_pool_rows, hv, k, stream);
    }
    else
    {
        launchGdnStateTranspose(params.h0_source, params.h0_scratch, numStateBlocks, k, stream);
    }

    gdn_prefill_blackwell_Tensor_h0_in_t h0InTensor{};
    h0InTensor.data = params.h0_scratch; // V-major initial state
    h0InTensor.dynamic_shapes[0] = n;
    h0InTensor.dynamic_shapes[1] = hv;
    h0InTensor.dynamic_strides[0] = static_cast<int64_t>(hv) * k * v;

    gdn_prefill_blackwell_Tensor_h0_out_t h0OutTensor{};
    h0OutTensor.data = params.h0_source; // V-major output written here
    h0OutTensor.dynamic_shapes[0] = params.state_pool_rows > 0 ? params.state_pool_rows : n;
    h0OutTensor.dynamic_shapes[1] = hv;
    h0OutTensor.dynamic_strides[0] = static_cast<int64_t>(hv) * k * v;

    gdn_prefill_blackwell_Tensor_state_indices_t stateIndicesTensor{};
    SET_1D_TENSOR(
        stateIndicesTensor, params.state_indices != nullptr ? params.state_indices : params.context_lengths, n);

    // cu_seqlens [N+1]: prefix-sum of context_lengths, computed by the plugin before launch.
    gdn_prefill_blackwell_Tensor_cu_seqlens_t cuSeqLensTensor{};
    SET_1D_TENSOR(cuSeqLensTensor, params.cu_seqlens, n + 1);

    gdn_prefill_blackwell_Tensor_o_t oTensor{};
    SET_4D_TENSOR(oTensor, params.o, n, seq_len, hv, v);

    // Step 2: Run the Blackwell prefill kernel.
    int32_t const result = cute_dsl_gdn_prefill_blackwell_wrapper(&sBlackwellPrefillModule.module, &qTensor, &kTensor,
        &vTensor, &aTensor, &bTensor, &A_logTensor, &dt_biasTensor, &h0InTensor, &h0OutTensor, &stateIndicesTensor,
        params.state_indices != nullptr, &oTensor, &cuSeqLensTensor, getDeviceMultiProcessorCount(), stream);
    if (result != 0)
    {
        return result;
    }

    if (params.state_indices != nullptr)
    {
        launchGdnStateIndexedTransposeInPlace(
            params.h0_source, params.state_indices, n, params.state_pool_rows, hv, k, stream);
    }
    else
    {
        launchGdnStateTranspose(params.h0_source, params.h0_scratch, numStateBlocks, k, stream);
        CUDA_CHECK(
            cudaMemcpyAsync(params.h0_source, params.h0_scratch, h0ScratchBytes, cudaMemcpyDeviceToDevice, stream));
    }
    return 0;
#else
    static_cast<void>(params);
    static_cast<void>(stream);
    LOG_ERROR("Blackwell GDN prefill not compiled in this build.");
    return -1;
#endif
}

int CuteDslGDNRunner::runPrefillBlackwellGeforce(GDNParams const& params, cudaStream_t stream)
{
#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
    if (params.tensormap_scratch == nullptr)
    {
        LOG_ERROR("GDN Blackwell GeForce prefill requires tensor-map scratch.");
        return -1;
    }

    int32_t const n = params.n;
    int32_t const seqLen = params.seq_len;
    int32_t const h = params.h;
    int32_t const hv = params.hv;
    int32_t const k = params.k_dim;
    int32_t const v = params.v_dim;
    int32_t const smCount = getDeviceMultiProcessorCount();
    if (smCount > kBlackwellGeforceMaxSMCount)
    {
        LOG_ERROR(
            "GDN Blackwell GeForce prefill supports at most %d SMs, got %d.", kBlackwellGeforceMaxSMCount, smCount);
        return -1;
    }

    if (params.state_indices != nullptr && params.state_pool_rows <= 0)
    {
        LOG_ERROR("GDN Blackwell GeForce indexed prefill requires a non-empty resident state pool.");
        return -1;
    }

    // Match the existing optimized Blackwell path: all consumers must observe
    // the same normalized Q/K values. The effective PDL value is shared by the
    // dual-role combined Q/K and generated fused GDN launches.
    int32_t const enablePdl = useGdnPdl(params.enablePdl, seqLen, params.smVersion) ? 1 : 0;
    cudaError_t const qkNormResult
        = launchGdnL2NormQKFusedSm12x(params.q, params.k, n, seqLen, h, k, enablePdl != 0, stream);
    if (qkNormResult != cudaSuccess)
    {
        LOG_ERROR("GDN Blackwell GeForce Q/K normalization launch failed: %s", cudaGetErrorString(qkNormResult));
        return -1;
    }

    gdn_prefill_blackwell_geforce_Tensor_q_t qTensor{};
    SET_4D_TENSOR(qTensor, params.q, n, seqLen, h, k);
    gdn_prefill_blackwell_geforce_Tensor_k_t kTensor{};
    SET_4D_TENSOR(kTensor, params.k, n, seqLen, h, k);
    gdn_prefill_blackwell_geforce_Tensor_v_t vTensor{};
    SET_4D_TENSOR(vTensor, params.v, n, seqLen, hv, v);
    gdn_prefill_blackwell_geforce_Tensor_a_t aTensor{};
    SET_3D_TENSOR(aTensor, params.a, n, seqLen, hv);
    gdn_prefill_blackwell_geforce_Tensor_b_t bTensor{};
    SET_3D_TENSOR(bTensor, params.b, n, seqLen, hv);

    gdn_prefill_blackwell_geforce_Tensor_A_log_t ALogTensor{};
    SET_1D_TENSOR(ALogTensor, params.A_log, hv);
    gdn_prefill_blackwell_geforce_Tensor_dt_bias_t dtBiasTensor{};
    SET_1D_TENSOR(dtBiasTensor, params.dt_bias, hv);

    // Active resident rows are unique, so each CTA exclusively owns one
    // state slice until its register-resident update is written back.
    gdn_prefill_blackwell_geforce_Tensor_h0_in_t h0InTensor{};
    h0InTensor.data = params.h0_source;
    h0InTensor.dynamic_shapes[0] = params.state_pool_rows > 0 ? params.state_pool_rows : n;
    h0InTensor.dynamic_shapes[1] = hv;
    h0InTensor.dynamic_strides[0] = static_cast<int64_t>(hv) * k * v;

    gdn_prefill_blackwell_geforce_Tensor_h0_out_t h0OutTensor{};
    h0OutTensor.data = params.h0_source;
    h0OutTensor.dynamic_shapes[0] = params.state_pool_rows > 0 ? params.state_pool_rows : n;
    h0OutTensor.dynamic_shapes[1] = hv;
    h0OutTensor.dynamic_strides[0] = static_cast<int64_t>(hv) * k * v;

    gdn_prefill_blackwell_geforce_Tensor_state_indices_t stateIndicesTensor{};
    SET_1D_TENSOR(
        stateIndicesTensor, params.state_indices != nullptr ? params.state_indices : params.context_lengths, n);

    gdn_prefill_blackwell_geforce_Tensor_context_lengths_t contextLengthsTensor{};
    SET_1D_TENSOR(contextLengthsTensor, params.context_lengths, n);
    gdn_prefill_blackwell_geforce_Tensor_o_t oTensor{};
    SET_4D_TENSOR(oTensor, params.o, n, seqLen, hv, v);

    gdn_prefill_blackwell_geforce_Tensor_tensormap_scratch_t tensormapScratchTensor{};
    SET_1D_TENSOR(tensormapScratchTensor, params.tensormap_scratch,
        kBlackwellGeforceMaxSMCount * kBlackwellGeforceTensorMapDescriptorBytes);

    return cute_dsl_gdn_prefill_blackwell_geforce_wrapper(&sBlackwellGeforcePrefillModule.module, &qTensor, &kTensor,
        &vTensor, &aTensor, &bTensor, &ALogTensor, &dtBiasTensor, &h0InTensor, &h0OutTensor, &stateIndicesTensor,
        params.state_indices != nullptr, &contextLengthsTensor, &oTensor, &tensormapScratchTensor, enablePdl, stream);
#else
    static_cast<void>(params);
    static_cast<void>(stream);
    LOG_ERROR("Blackwell GeForce GDN prefill not compiled in this build.");
    return -1;
#endif
}

int CuteDslGDNRunner::runDecodeMTP(GDNParams const& params, cudaStream_t stream)
{
    int32_t const n = params.n;
    int32_t const seq_len = params.seq_len;
    int32_t const h = params.h;
    int32_t const hv = params.hv;
    int32_t const k = params.k_dim;
    int32_t const v = params.v_dim;

    if (seq_len < 1)
    {
        LOG_ERROR("CuTe DSL GDN MTP decode requires seq_len >= 1, got %d", seq_len);
        return -1;
    }

    // Must match T_MAX_AOT in gdn_decode_mtp.py.
    constexpr int32_t kMTPMaxSeqLen = 16;
    if (seq_len > kMTPMaxSeqLen)
    {
        LOG_ERROR("CuTe DSL GDN MTP decode requires seq_len <= %d, got %d", kMTPMaxSeqLen, seq_len);
        return -1;
    }

    // q / k : [n, seq_len, h, k]
    gdn_decode_mtp_cache_Tensor_q_t qTensor{};
    SET_4D_TENSOR(qTensor, params.q, n, seq_len, h, k);

    gdn_decode_mtp_cache_Tensor_k_t kTensor{};
    SET_4D_TENSOR(kTensor, params.k, n, seq_len, h, k);

    // v : [n, seq_len, hv, v]
    gdn_decode_mtp_cache_Tensor_v_t vTensor{};
    SET_4D_TENSOR(vTensor, params.v, n, seq_len, hv, v);

    // a / b : [n, seq_len, hv]
    gdn_decode_mtp_cache_Tensor_a_t aTensor{};
    SET_3D_TENSOR(aTensor, params.a, n, seq_len, hv);

    gdn_decode_mtp_cache_Tensor_b_t bTensor{};
    SET_3D_TENSOR(bTensor, params.b, n, seq_len, hv);

    // A_log / dt_bias : [hv]
    gdn_decode_mtp_cache_Tensor_A_log_t A_logTensor{};
    SET_1D_TENSOR(A_logTensor, params.A_log, hv);

    gdn_decode_mtp_cache_Tensor_dt_bias_t dt_biasTensor{};
    SET_1D_TENSOR(dt_biasTensor, params.dt_bias, hv);

    // h0_source : [n, hv, k, v]  (batch-dense, in-place updated)
    gdn_decode_mtp_cache_Tensor_h0_source_t h0Tensor{};
    h0Tensor.data = params.h0_source;
    h0Tensor.dynamic_shapes[0] = n;
    h0Tensor.dynamic_shapes[1] = hv;
    h0Tensor.dynamic_strides[0] = static_cast<int64_t>(hv) * k * v;

    // o : [n, seq_len, hv, v]
    gdn_decode_mtp_cache_Tensor_o_t oTensor{};
    SET_4D_TENSOR(oTensor, params.o, n, seq_len, hv, v);

    if (params.intermediate_states == nullptr)
    {
        LOG_ERROR("CuTe DSL GDN MTP: intermediate_states is null.");
        return -1;
    }

    // intermediate_states : [n, seq_len, hv, k, v]
    gdn_decode_mtp_cache_Tensor_intermediate_states_t intermTensor{};
    intermTensor.data = params.intermediate_states;
    intermTensor.dynamic_shapes[0] = n;
    intermTensor.dynamic_shapes[1] = seq_len;
    intermTensor.dynamic_shapes[2] = hv;
    intermTensor.dynamic_strides[0] = static_cast<int64_t>(seq_len) * hv * k * v;
    intermTensor.dynamic_strides[1] = static_cast<int64_t>(hv) * k * v;
    intermTensor.dynamic_strides[2] = static_cast<int64_t>(k) * v;
    intermTensor.dynamic_strides[3] = static_cast<int64_t>(v);

    return cute_dsl_gdn_decode_mtp_cache_wrapper(&sMTPDecodeCacheModule.module, &h0Tensor, &qTensor, &kTensor, &vTensor,
        &aTensor, &bTensor, &A_logTensor, &dt_biasTensor, &oTensor, &intermTensor, seq_len, stream);
}

} // namespace trt_edgellm

#endif // CUTE_DSL_GDN_ENABLED
