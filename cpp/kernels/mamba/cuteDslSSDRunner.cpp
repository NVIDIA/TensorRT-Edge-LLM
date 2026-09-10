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

#ifdef CUTE_DSL_SSD_ENABLED

#include "cuteDslSSDRunner.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "ssdVarlenMetadata.h"

#include <cmath>

namespace trt_edgellm
{

namespace
{

constexpr int32_t kCHUNK_SIZE{128};

#if defined(CUTE_DSL_SSD_BLACKWELL_ENABLED)
constexpr int32_t kD80_DIM{80};
constexpr int32_t kD80_DSTATE{128};
#endif

bool isGenericConfiguration(int32_t dim, int32_t dstate)
{
    return (dim == 64 || dim == 128) && (dstate == 64 || dstate == 128);
}

#if defined(CUTE_DSL_SSD_BLACKWELL_ENABLED)
bool isD80BlackwellSm(int32_t smVersion)
{
    return smVersion == 100 || smVersion == 101 || smVersion == 110;
}

bool isD80BlackwellConfiguration(int32_t dim, int32_t dstate, int32_t smVersion)
{
    return dim == kD80_DIM && dstate == kD80_DSTATE && isD80BlackwellSm(smVersion);
}
#endif

bool hasCompiledConfiguration(int32_t dim, int32_t dstate)
{
    if (isGenericConfiguration(dim, dstate))
    {
        return true;
    }
#if defined(CUTE_DSL_SSD_BLACKWELL_ENABLED)
    return dim == kD80_DIM && dstate == kD80_DSTATE;
#else
    return false;
#endif
}

size_t getGenericWorkspaceSize(
    int32_t batch, int32_t seqLen, int32_t nheads, int32_t dim, int32_t dstate, int32_t ngroups)
{
    int32_t const nchunks = (seqLen + kCHUNK_SIZE - 1) / kCHUNK_SIZE;
    size_t size{};
    size += static_cast<size_t>(batch) * nheads * nchunks * kCHUNK_SIZE * sizeof(float);
    size += static_cast<size_t>(batch) * nheads * nchunks * kCHUNK_SIZE * sizeof(float);
    size += static_cast<size_t>(batch) * nchunks * nheads * dim * dstate * sizeof(float);
    size += static_cast<size_t>(batch) * nchunks * nheads * dim * dstate * sizeof(float);
    size += static_cast<size_t>(batch) * nchunks * ngroups * kCHUNK_SIZE * kCHUNK_SIZE * sizeof(float);
    size += static_cast<size_t>(batch) * sizeof(int32_t);
    return size;
}

#if defined(CUTE_DSL_SSD_BLACKWELL_ENABLED)
size_t getBlackwellWorkspaceSize(int32_t batch, int32_t seqLen, int32_t nheads, int32_t dim)
{
    int32_t const nchunks = (seqLen + kCHUNK_SIZE - 1) / kCHUNK_SIZE;
    int32_t const totalSeq = batch * seqLen;
    int32_t const totalChunks = (totalSeq + kCHUNK_SIZE - 1) / kCHUNK_SIZE;
    int32_t const logicalChunksPerSeqUpper = nchunks + ((seqLen % kCHUNK_SIZE) == 0 ? 0 : 1);
    size_t size{};
    size += static_cast<size_t>(totalChunks) * nheads * kCHUNK_SIZE * sizeof(float);
    size += static_cast<size_t>(totalChunks) * nheads * kCHUNK_SIZE * sizeof(__half);
    size += static_cast<size_t>(totalChunks) * nheads * dim * kCHUNK_SIZE * sizeof(__half);
    size += static_cast<size_t>(batch) * seqLen * sizeof(int32_t);
    size += static_cast<size_t>(batch) * logicalChunksPerSeqUpper * sizeof(int32_t);
    size += static_cast<size_t>(batch) * logicalChunksPerSeqUpper * sizeof(int32_t);
    size += static_cast<size_t>(batch + 1) * sizeof(int32_t);
    size += static_cast<size_t>(batch) * sizeof(int32_t);
    return size;
}

#endif

} // namespace

detail::LazyKernelModule<ssd_prefill_d128_n128_Kernel_Module_t> CuteDslSSDRunner::sD128N128Module{};
detail::LazyKernelModule<ssd_prefill_d64_n128_Kernel_Module_t> CuteDslSSDRunner::sD64N128Module{};
detail::LazyKernelModule<ssd_prefill_d128_n64_Kernel_Module_t> CuteDslSSDRunner::sD128N64Module{};
detail::LazyKernelModule<ssd_prefill_d64_n64_Kernel_Module_t> CuteDslSSDRunner::sD64N64Module{};
#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED
detail::LazyKernelModule<ssd_prefill_blackwell_d64_n128_Kernel_Module_t> CuteDslSSDRunner::sBlackwellD64N128Module{};
detail::LazyKernelModule<ssd_prefill_blackwell_d64_n128_init_states_Kernel_Module_t>
    CuteDslSSDRunner::sBlackwellD64N128InitStatesModule{};
detail::LazyKernelModule<ssd_prefill_blackwell_d80_n128_Kernel_Module_t> CuteDslSSDRunner::sBlackwellD80N128Module{};
detail::LazyKernelModule<ssd_prefill_blackwell_d80_n128_init_states_Kernel_Module_t>
    CuteDslSSDRunner::sBlackwellD80N128InitStatesModule{};
detail::LazyKernelModule<ssd_prefill_blackwell_d64_n64_Kernel_Module_t> CuteDslSSDRunner::sBlackwellD64N64Module{};
detail::LazyKernelModule<ssd_prefill_blackwell_d64_n64_init_states_Kernel_Module_t>
    CuteDslSSDRunner::sBlackwellD64N64InitStatesModule{};
#endif

// Tensor struct helpers — contiguous row-major layout.
#define SET_5D_TENSOR(tensor, data_ptr, d0, d1, d2, d3, d4)                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (d0);                                                                             \
        (tensor).dynamic_shapes[1] = (d1);                                                                             \
        (tensor).dynamic_shapes[2] = (d2);                                                                             \
        (tensor).dynamic_shapes[3] = (d3);                                                                             \
        (tensor).dynamic_shapes[4] = (d4);                                                                             \
        (tensor).dynamic_strides[0] = static_cast<int64_t>(d1) * (d2) * (d3) * (d4);                                   \
        (tensor).dynamic_strides[1] = static_cast<int64_t>(d2) * (d3) * (d4);                                          \
        (tensor).dynamic_strides[2] = static_cast<int64_t>(d3) * (d4);                                                 \
        (tensor).dynamic_strides[3] = static_cast<int64_t>(d4);                                                        \
    } while (0)

#define SET_4D_TENSOR(tensor, data_ptr, d0, d1, d2, d3)                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (d0);                                                                             \
        (tensor).dynamic_shapes[1] = (d1);                                                                             \
        (tensor).dynamic_shapes[2] = (d2);                                                                             \
        (tensor).dynamic_shapes[3] = (d3);                                                                             \
        (tensor).dynamic_strides[0] = static_cast<int64_t>(d1) * (d2) * (d3);                                          \
        (tensor).dynamic_strides[1] = static_cast<int64_t>(d2) * (d3);                                                 \
        (tensor).dynamic_strides[2] = static_cast<int64_t>(d3);                                                        \
    } while (0)

#define SET_3D_TENSOR(tensor, data_ptr, d0, d1, d2)                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (d0);                                                                             \
        (tensor).dynamic_shapes[1] = (d1);                                                                             \
        (tensor).dynamic_shapes[2] = (d2);                                                                             \
        (tensor).dynamic_strides[0] = static_cast<int64_t>(d1) * (d2);                                                 \
        (tensor).dynamic_strides[1] = static_cast<int64_t>(d2);                                                        \
    } while (0)

#define SET_1D_TENSOR(tensor, data_ptr, d0)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (d0);                                                                             \
    } while (0)

bool CuteDslSSDRunner::canImplement(int32_t dim, int32_t dstate, int32_t smVersion)
{
    if (smVersion < 80)
    {
        return false;
    }
    if (isGenericConfiguration(dim, dstate))
    {
        return true;
    }
#if defined(CUTE_DSL_SSD_BLACKWELL_ENABLED)
    return isD80BlackwellConfiguration(dim, dstate, smVersion);
#else
    return false;
#endif
}

bool CuteDslSSDRunner::ensureKernelModules(SSDParams const& params, cudaStream_t stream)
{
#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED
    if (isD80BlackwellConfiguration(params.dim, params.dstate, params.smVersion))
    {
        if (params.has_init_states)
        {
            return detail::ensureModuleLoaded<ssd_prefill_blackwell_d80_n128_init_states_Kernel_Module_Load,
                ssd_prefill_blackwell_d80_n128_init_states_Kernel_Module_Unload>(
                sBlackwellD80N128InitStatesModule, "ssd_prefill_blackwell_d80_n128_init_states", stream);
        }
        return detail::ensureModuleLoaded<ssd_prefill_blackwell_d80_n128_Kernel_Module_Load,
            ssd_prefill_blackwell_d80_n128_Kernel_Module_Unload>(
            sBlackwellD80N128Module, "ssd_prefill_blackwell_d80_n128", stream);
    }
    if (params.smVersion >= 100 && params.smVersion < 120 && params.dim == 64)
    {
        if (params.dstate == 128 && params.has_init_states)
        {
            return detail::ensureModuleLoaded<ssd_prefill_blackwell_d64_n128_init_states_Kernel_Module_Load,
                ssd_prefill_blackwell_d64_n128_init_states_Kernel_Module_Unload>(
                sBlackwellD64N128InitStatesModule, "ssd_prefill_blackwell_d64_n128_init_states", stream);
        }
        if (params.dstate == 128)
        {
            return detail::ensureModuleLoaded<ssd_prefill_blackwell_d64_n128_Kernel_Module_Load,
                ssd_prefill_blackwell_d64_n128_Kernel_Module_Unload>(
                sBlackwellD64N128Module, "ssd_prefill_blackwell_d64_n128", stream);
        }
        if (params.dstate == 64 && params.has_init_states)
        {
            return detail::ensureModuleLoaded<ssd_prefill_blackwell_d64_n64_init_states_Kernel_Module_Load,
                ssd_prefill_blackwell_d64_n64_init_states_Kernel_Module_Unload>(
                sBlackwellD64N64InitStatesModule, "ssd_prefill_blackwell_d64_n64_init_states", stream);
        }
        if (params.dstate == 64)
        {
            return detail::ensureModuleLoaded<ssd_prefill_blackwell_d64_n64_Kernel_Module_Load,
                ssd_prefill_blackwell_d64_n64_Kernel_Module_Unload>(
                sBlackwellD64N64Module, "ssd_prefill_blackwell_d64_n64", stream);
        }
        return false;
    }
#endif
    if (params.dim == 128 && params.dstate == 128)
    {
        return detail::ensureModuleLoaded<ssd_prefill_d128_n128_Kernel_Module_Load,
            ssd_prefill_d128_n128_Kernel_Module_Unload>(sD128N128Module, "ssd_prefill_d128_n128", stream);
    }
    if (params.dim == 64 && params.dstate == 128)
    {
        return detail::ensureModuleLoaded<ssd_prefill_d64_n128_Kernel_Module_Load,
            ssd_prefill_d64_n128_Kernel_Module_Unload>(sD64N128Module, "ssd_prefill_d64_n128", stream);
    }
    if (params.dim == 128 && params.dstate == 64)
    {
        return detail::ensureModuleLoaded<ssd_prefill_d128_n64_Kernel_Module_Load,
            ssd_prefill_d128_n64_Kernel_Module_Unload>(sD128N64Module, "ssd_prefill_d128_n64", stream);
    }
    if (params.dim == 64 && params.dstate == 64)
    {
        return detail::ensureModuleLoaded<ssd_prefill_d64_n64_Kernel_Module_Load,
            ssd_prefill_d64_n64_Kernel_Module_Unload>(sD64N64Module, "ssd_prefill_d64_n64", stream);
    }
    return false;
}

int CuteDslSSDRunner::run(SSDParams const& params, cudaStream_t stream)
{
    if (!ensureKernelModules(params, stream))
    {
        return -1;
    }
#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED
    if (isD80BlackwellConfiguration(params.dim, params.dstate, params.smVersion))
    {
        return runPrefillBlackwell(params, stream);
    }
    // SM100-110 with D=64: use Blackwell persistent kernel (TMA/wgmma/TMEM).
    // SM120+ lacks TMEM/wgmma — falls through to SM80 kernel.
    if (params.smVersion >= 100 && params.smVersion < 120 && params.dim == 64
        && (params.dstate == 128 || params.dstate == 64))
        return runPrefillBlackwell(params, stream);
#endif
    return runPrefill(params, stream);
}

// Macro to fill tensor structs and call the SM80 wrapper for a given variant prefix.
// All SM80 variants share the same struct layout — only the type name prefix differs.
// Relies on local variables: params, n, seq_len, nheads, dim, dstate, ngroups, nchunks,
// nheads_ngroups_ratio, stream.
// clang-format off
#define CALL_SSD_PREFILL(PREFIX, MODULE)                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        PREFIX##_Tensor_x_t xTensor{};                                                                                 \
        SET_4D_TENSOR(xTensor, params.x, n, seq_len, nheads, dim);                                                     \
                                                                                                                       \
        PREFIX##_Tensor_dt_in_t dtTensor{};                                                                            \
        SET_3D_TENSOR(dtTensor, params.dt, n, seq_len, nheads);                                                        \
                                                                                                                       \
        PREFIX##_Tensor_A_t aTensor{};                                                                                 \
        SET_1D_TENSOR(aTensor, params.A, nheads);                                                                      \
                                                                                                                       \
        PREFIX##_Tensor_B_t bTensor{};                                                                                 \
        bTensor.data = params.B;                                                                                       \
        bTensor.dynamic_shapes[0] = n;                                                                                 \
        bTensor.dynamic_shapes[1] = seq_len;                                                                           \
        bTensor.dynamic_shapes[2] = ngroups;                                                                           \
        bTensor.dynamic_strides[0] = static_cast<int64_t>(seq_len) * ngroups * dstate;                                 \
        bTensor.dynamic_strides[1] = static_cast<int64_t>(ngroups) * dstate;                                           \
                                                                                                                       \
        PREFIX##_Tensor_C_t cTensor{};                                                                                 \
        cTensor.data = params.C;                                                                                       \
        cTensor.dynamic_shapes[0] = n;                                                                                 \
        cTensor.dynamic_shapes[1] = seq_len;                                                                           \
        cTensor.dynamic_shapes[2] = ngroups;                                                                           \
        cTensor.dynamic_strides[0] = static_cast<int64_t>(seq_len) * ngroups * dstate;                                 \
        cTensor.dynamic_strides[1] = static_cast<int64_t>(ngroups) * dstate;                                           \
                                                                                                                       \
        PREFIX##_Tensor_D_t dTensor{};                                                                                 \
        SET_1D_TENSOR(dTensor, params.D, nheads);                                                                      \
                                                                                                                       \
        PREFIX##_Tensor_dt_bias_t dtBiasTensor{};                                                                      \
        SET_1D_TENSOR(dtBiasTensor, params.dt_bias, nheads);                                                           \
                                                                                                                       \
        PREFIX##_Tensor_z_t zTensor{};                                                                                 \
        SET_4D_TENSOR(zTensor, params.z ? params.z : params.x, n, seq_len, nheads, dim);                               \
                                                                                                                       \
        PREFIX##_Tensor_output_t outputTensor{};                                                                       \
        SET_4D_TENSOR(outputTensor, params.output, n, seq_len, nheads, dim);                                           \
                                                                                                                       \
        PREFIX##_Tensor_final_states_t finalStatesTensor{};                                                            \
        SET_4D_TENSOR(finalStatesTensor, params.state, n, nheads, dim, dstate);                                        \
                                                                                                                       \
        /* init_states aliases params.state: kernel reads init at chunk 0, overwrites with final. */                   \
        PREFIX##_Tensor_init_states_t initStatesTensor{};                                                              \
        SET_4D_TENSOR(initStatesTensor, params.state, n, nheads, dim, dstate);                                         \
                                                                                                                       \
        void* ws = params.workspace;                                                                                   \
        size_t offset = 0;                                                                                             \
                                                                                                                       \
        PREFIX##_Tensor_dA_cumsum_t dACumsumTensor{};                                                                  \
        size_t dACumsumBytes = static_cast<size_t>(n) * nheads * nchunks * 128 * sizeof(float);                        \
        SET_4D_TENSOR(dACumsumTensor, static_cast<char*>(ws) + offset, n, nheads, nchunks, 128);                       \
        offset += dACumsumBytes;                                                                                       \
                                                                                                                       \
        PREFIX##_Tensor_dt_proc_t dtProcTensor{};                                                                      \
        SET_4D_TENSOR(dtProcTensor, static_cast<char*>(ws) + offset, n, nheads, nchunks, 128);                         \
        offset += dACumsumBytes;                                                                                       \
                                                                                                                       \
        PREFIX##_Tensor_states_t statesTensor{};                                                                       \
        size_t statesBytes = static_cast<size_t>(n) * nchunks * nheads * dim * dstate * sizeof(float);                 \
        SET_5D_TENSOR(statesTensor, static_cast<char*>(ws) + offset, n, nchunks, nheads, dim, dstate);                 \
        offset += statesBytes;                                                                                         \
                                                                                                                       \
        PREFIX##_Tensor_prev_states_t prevStatesTensor{};                                                              \
        SET_5D_TENSOR(prevStatesTensor, static_cast<char*>(ws) + offset, n, nchunks, nheads, dim, dstate);             \
        offset += statesBytes;                                                                                         \
                                                                                                                       \
        PREFIX##_Tensor_CB_t cbTensor{};                                                                               \
        size_t cbBytes = static_cast<size_t>(n) * nchunks * ngroups * 128 * 128 * sizeof(float);                       \
        SET_5D_TENSOR(cbTensor, static_cast<char*>(ws) + offset, n, nchunks, ngroups, 128, 128);                       \
        offset += cbBytes;                                                                                             \
                                                                                                                       \
        /* Memset must precede cl synth: same-stream memset would clobber the just-filled cl[]. */                     \
        cudaMemsetAsync(ws, 0, offset, stream);                                                                        \
                                                                                                                       \
        /* Synth cl[b] = seq_len when caller passes null (kernel always reads context_lengths). */                     \
        int32_t const* clPtr = reinterpret_cast<int32_t const*>(params.context_lengths);                               \
        if (clPtr == nullptr)                                                                                          \
        {                                                                                                              \
            size_t clBytes = static_cast<size_t>(n) * sizeof(int32_t);                                                 \
            int32_t* dCl = reinterpret_cast<int32_t*>(static_cast<char*>(ws) + offset);                                \
            offset += clBytes;                                                                                         \
            ::trt_edgellm::mamba::fillUniformValidLens(dCl, n, seq_len, stream);                                       \
            clPtr = dCl;                                                                                               \
        }                                                                                                              \
        PREFIX##_Tensor_context_lengths_t clTensor{};                                                                  \
        clTensor.data = const_cast<int32_t*>(clPtr);                                                                   \
        clTensor.dynamic_shapes[0] = n;                                                                                \
                                                                                                                       \
        return cute_dsl_##PREFIX##_wrapper(&(MODULE), &xTensor, &dtTensor, &aTensor, &bTensor, &cTensor, &dTensor,     \
            &dtBiasTensor, &zTensor, &outputTensor, &finalStatesTensor, &initStatesTensor,                             \
            &dACumsumTensor, &dtProcTensor, &statesTensor, &prevStatesTensor, &cbTensor, &clTensor,                    \
            nchunks, n, nheads, nheads_ngroups_ratio, ngroups, stream);                                                \
    } while (0)
// clang-format on

int CuteDslSSDRunner::runPrefill(SSDParams const& params, cudaStream_t stream)
{
    // buildSSDVarlenMetadata launches a single-block kernel; batch must fit one CTA's
    // 1024-thread cap. Production Nemotron-H workloads use batch <= 8.
    if (params.batch > 1024)
    {
        LOG_ERROR("CuTe DSL SSD prefill: batch=%d exceeds supported maximum (1024)", params.batch);
        return -1;
    }

    int32_t const n = params.batch;
    int32_t const seq_len = params.seq_len;
    int32_t const nheads = params.nheads;
    int32_t const dim = params.dim;
    int32_t const dstate = params.dstate;
    int32_t const ngroups = params.ngroups;
    int32_t const nchunks = (seq_len + 127) / 128; // CHUNK_SIZE = 128
    int32_t const nheads_ngroups_ratio = nheads / ngroups;

    if (dim == 128 && dstate == 128)
    {
        CALL_SSD_PREFILL(ssd_prefill_d128_n128, sD128N128Module.module);
    }
    else if (dim == 64 && dstate == 128)
    {
        CALL_SSD_PREFILL(ssd_prefill_d64_n128, sD64N128Module.module);
    }
    else if (dim == 128 && dstate == 64)
    {
        CALL_SSD_PREFILL(ssd_prefill_d128_n64, sD128N64Module.module);
    }
    else if (dim == 64 && dstate == 64)
    {
        CALL_SSD_PREFILL(ssd_prefill_d64_n64, sD64N64Module.module);
    }

    LOG_ERROR("CuTe DSL SSD prefill: unsupported dim=%d dstate=%d", dim, dstate);
    return -1;
}

#undef CALL_SSD_PREFILL

#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED

// Macro to fill tensor structs and call the Blackwell wrapper for a given variant prefix.
// All Blackwell variants share the same struct layout: only the type name prefix differs.
// Relies on local variables: params, n, seq_len, nheads, dim, dstate, ngroups, nchunks, stream.
// Plugin contract is padded [B,S,...]+context_lengths; runner reshapes to [1,B*S,...] via
// stride change (zero data movement), passes context_lengths as valid_lens for end-of-seq clamp.
// clang-format off
#define CALL_SSD_PREFILL_BLACKWELL(PREFIX, MODULE)                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        /* Reshape padded [B,T,...] to [1,B*T,...] (zero data movement); kernel hardcodes b_idx=0. */                  \
        int32_t const total_seq = n * seq_len;                                                                         \
        int32_t const total_chunks = (total_seq + 127) / 128;                                                          \
        int32_t const logical_chunks_per_seq_upper = nchunks + ((seq_len % 128) == 0 ? 0 : 1);                         \
        int32_t const numLogicalChunks = n * logical_chunks_per_seq_upper;                                             \
                                                                                                                       \
        PREFIX##_Tensor_x_t xTensor{};                                                                                 \
        SET_4D_TENSOR(xTensor, params.x, 1, total_seq, nheads, dim);                                                   \
                                                                                                                       \
        PREFIX##_Tensor_dt_in_t dtTensor{};                                                                            \
        SET_3D_TENSOR(dtTensor, params.dt, 1, total_seq, nheads);                                                      \
                                                                                                                       \
        PREFIX##_Tensor_A_t aTensor{};                                                                                 \
        SET_1D_TENSOR(aTensor, params.A, nheads);                                                                      \
                                                                                                                       \
        PREFIX##_Tensor_B_t bTensor{};                                                                                 \
        SET_4D_TENSOR(bTensor, params.B, 1, total_seq, ngroups, dstate);                                               \
                                                                                                                       \
        PREFIX##_Tensor_C_t cTensor{};                                                                                 \
        SET_4D_TENSOR(cTensor, params.C, 1, total_seq, ngroups, dstate);                                               \
                                                                                                                       \
        PREFIX##_Tensor_Dvec_t dTensor{};                                                                              \
        SET_1D_TENSOR(dTensor, params.D, nheads);                                                                      \
                                                                                                                       \
        PREFIX##_Tensor_dt_bias_t dtBiasTensor{};                                                                      \
        SET_1D_TENSOR(dtBiasTensor, params.dt_bias, nheads);                                                           \
                                                                                                                       \
        PREFIX##_Tensor_output_t outputTensor{};                                                                       \
        SET_4D_TENSOR(outputTensor, params.output, 1, total_seq, nheads, dim);                                         \
                                                                                                                       \
        PREFIX##_Tensor_state_t stateTensor{};                                                                         \
        SET_4D_TENSOR(stateTensor, params.state, n, nheads, dim, dstate);                                              \
                                                                                                                       \
        void* ws = params.workspace;                                                                                   \
        size_t offset = 0;                                                                                             \
                                                                                                                       \
        /* cumsum / dt_proc / y_ws workspaces: [1, C, EH, ...] row-major. */                                          \
        size_t cumsumBytes = static_cast<size_t>(1) * total_chunks * nheads * 128 * sizeof(float);                     \
        PREFIX##_Tensor_dA_cumsum_t cumsumTensor{};                                                                    \
        SET_4D_TENSOR(cumsumTensor, static_cast<char*>(ws) + offset, 1, total_chunks, nheads, 128);                    \
        offset += cumsumBytes;                                                                                         \
                                                                                                                       \
        size_t dtProcBytes = static_cast<size_t>(1) * total_chunks * nheads * 128 * sizeof(__half);                    \
        PREFIX##_Tensor_dt_proc_t dtProcTensor{};                                                                      \
        SET_4D_TENSOR(dtProcTensor, static_cast<char*>(ws) + offset, 1, total_chunks, nheads, 128);                    \
        offset += dtProcBytes;                                                                                         \
                                                                                                                       \
        size_t yBytes = static_cast<size_t>(1) * total_chunks * nheads * dim * 128 * sizeof(__half);                   \
        PREFIX##_Tensor_y_ws_t yTensor{};                                                                              \
        SET_5D_TENSOR(yTensor, static_cast<char*>(ws) + offset, 1, total_chunks, nheads, dim, 128);                    \
        offset += yBytes;                                                                                              \
                                                                                                                       \
        /* Varlen metadata (always-varlen at AOT: uniform batch synthesizes degenerate metadata). */                   \
        size_t seqIdxBytes = static_cast<size_t>(n) * seq_len * sizeof(int32_t);                                       \
        int32_t* dSeqIdx = reinterpret_cast<int32_t*>(static_cast<char*>(ws) + offset);                                \
        offset += seqIdxBytes;                                                                                         \
                                                                                                                       \
        size_t chunkBytes = static_cast<size_t>(numLogicalChunks) * sizeof(int32_t);                                  \
        int32_t* dChunkIndices = reinterpret_cast<int32_t*>(static_cast<char*>(ws) + offset);                          \
        offset += chunkBytes;                                                                                          \
        int32_t* dChunkOffsets = reinterpret_cast<int32_t*>(static_cast<char*>(ws) + offset);                          \
        offset += chunkBytes;                                                                                          \
                                                                                                                       \
        size_t cumsumChunkBytes = static_cast<size_t>(n + 1) * sizeof(int32_t);                                        \
        int32_t* dSeqChunkCumsum = reinterpret_cast<int32_t*>(static_cast<char*>(ws) + offset);                        \
        offset += cumsumChunkBytes;                                                                                    \
                                                                                                                       \
        cudaMemsetAsync(ws, 0, offset, stream);                                                                        \
                                                                                                                       \
        /* Build metadata fully on-device: CUDA-graph-compatible, no D2H sync. */                                      \
        int32_t const numSeqs = n;                                                                                     \
        ::trt_edgellm::mamba::buildSSDVarlenMetadata(                                                                  \
            dSeqIdx, dChunkIndices, dChunkOffsets, dSeqChunkCumsum,                                                    \
            reinterpret_cast<int32_t const*>(params.context_lengths),                                                  \
            n, seq_len, 128, stream);                                                                                  \
                                                                                                                       \
        PREFIX##_Tensor_seq_idx_t seqIdxTensor{};                                                                      \
        seqIdxTensor.data = dSeqIdx;                                                                                   \
        seqIdxTensor.dynamic_shapes[0] = 1;                                                                            \
        seqIdxTensor.dynamic_shapes[1] = total_seq;                                                                    \
        seqIdxTensor.dynamic_strides[0] = total_seq;                                                                   \
                                                                                                                       \
        PREFIX##_Tensor_chunk_indices_t chunkIndicesTensor{};                                                          \
        chunkIndicesTensor.data = dChunkIndices;                                                                       \
        chunkIndicesTensor.dynamic_shapes[0] = numLogicalChunks;                                                       \
                                                                                                                       \
        PREFIX##_Tensor_chunk_offsets_t chunkOffsetsTensor{};                                                          \
        chunkOffsetsTensor.data = dChunkOffsets;                                                                       \
        chunkOffsetsTensor.dynamic_shapes[0] = numLogicalChunks;                                                       \
                                                                                                                       \
        PREFIX##_Tensor_seq_chunk_cumsum_t seqChunkCumsumTensor{};                                                     \
        seqChunkCumsumTensor.data = dSeqChunkCumsum;                                                                   \
        seqChunkCumsumTensor.dynamic_shapes[0] = numSeqs + 1;                                                          \
                                                                                                                       \
        /* Synth valid_lens[b] = seq_len when caller passes null (kernel always reads context_lengths). */             \
        int32_t const* validLensPtr = reinterpret_cast<int32_t const*>(params.context_lengths);                        \
        int32_t* dValidLensSynth = nullptr;                                                                            \
        if (validLensPtr == nullptr)                                                                                   \
        {                                                                                                              \
            size_t validLensBytes = static_cast<size_t>(n) * sizeof(int32_t);                                          \
            dValidLensSynth = reinterpret_cast<int32_t*>(static_cast<char*>(ws) + offset);                             \
            offset += validLensBytes;                                                                                  \
            ::trt_edgellm::mamba::fillUniformValidLens(dValidLensSynth, n, seq_len, stream);                           \
            validLensPtr = dValidLensSynth;                                                                            \
        }                                                                                                              \
        PREFIX##_Tensor_valid_lens_t validLensTensor{};                                                                \
        validLensTensor.data = const_cast<int32_t*>(validLensPtr);                                                     \
        validLensTensor.dynamic_shapes[0] = n;                                                                         \
                                                                                                                       \
        return cute_dsl_##PREFIX##_wrapper(&(MODULE), &xTensor, &dtTensor, &aTensor, &bTensor, &cTensor, &dTensor,     \
            &dtBiasTensor, &outputTensor, &stateTensor,                                                                \
            &cumsumTensor, &dtProcTensor, &yTensor,                                                                    \
            &seqIdxTensor, &chunkIndicesTensor, &chunkOffsetsTensor, &seqChunkCumsumTensor,                            \
            &validLensTensor,                                                                                          \
            total_seq, total_chunks, numLogicalChunks, numSeqs, getDeviceMultiProcessorCount(), stream);              \
    } while (0)
// clang-format on

int CuteDslSSDRunner::runPrefillBlackwell(SSDParams const& params, cudaStream_t stream)
{
    if (params.batch > 1024)
    {
        LOG_ERROR("CuTe DSL SSD Blackwell prefill: batch=%d exceeds supported maximum (1024)", params.batch);
        return -1;
    }

    int32_t const n = params.batch;
    int32_t const seq_len = params.seq_len;
    int32_t const nheads = params.nheads;
    int32_t const dim = params.dim;
    int32_t const dstate = params.dstate;
    int32_t const ngroups = params.ngroups;
    int32_t const nchunks = (seq_len + 127) / 128;

    // Dispatch on (dim, dstate, has_init_states).
    if (dim == 80 && dstate == 128)
    {
        if (params.has_init_states)
            CALL_SSD_PREFILL_BLACKWELL(
                ssd_prefill_blackwell_d80_n128_init_states, sBlackwellD80N128InitStatesModule.module);
        else
            CALL_SSD_PREFILL_BLACKWELL(ssd_prefill_blackwell_d80_n128, sBlackwellD80N128Module.module);
    }
    else if (dim == 64 && dstate == 128)
    {
        if (params.has_init_states)
            CALL_SSD_PREFILL_BLACKWELL(
                ssd_prefill_blackwell_d64_n128_init_states, sBlackwellD64N128InitStatesModule.module);
        else
            CALL_SSD_PREFILL_BLACKWELL(ssd_prefill_blackwell_d64_n128, sBlackwellD64N128Module.module);
    }
    else if (dim == 64 && dstate == 64)
    {
        if (params.has_init_states)
            CALL_SSD_PREFILL_BLACKWELL(
                ssd_prefill_blackwell_d64_n64_init_states, sBlackwellD64N64InitStatesModule.module);
        else
            CALL_SSD_PREFILL_BLACKWELL(ssd_prefill_blackwell_d64_n64, sBlackwellD64N64Module.module);
    }

    LOG_ERROR("CuTe DSL SSD Blackwell prefill: unsupported dim=%d dstate=%d", dim, dstate);
    return -1;
}

#undef CALL_SSD_PREFILL_BLACKWELL
#endif

size_t CuteDslSSDRunner::getWorkspaceSize(
    int32_t batch, int32_t seqLen, int32_t nheads, int32_t dim, int32_t dstate, int32_t ngroups)
{
    if (!hasCompiledConfiguration(dim, dstate))
    {
        return 0;
    }

#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED
    if (dim == kD80_DIM && dstate == kD80_DSTATE)
    {
        return getBlackwellWorkspaceSize(batch, seqLen, nheads, dim);
    }
    return std::max(getGenericWorkspaceSize(batch, seqLen, nheads, dim, dstate, ngroups),
        getBlackwellWorkspaceSize(batch, seqLen, nheads, dim));
#else
    return getGenericWorkspaceSize(batch, seqLen, nheads, dim, dstate, ngroups);
#endif
}

} // namespace trt_edgellm

#endif // CUTE_DSL_SSD_ENABLED
