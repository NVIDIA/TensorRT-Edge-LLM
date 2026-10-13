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

/*
 * This file contains code derived from FlashInfer (https://github.com/flashinfer-ai/flashinfer)
 * Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
 * Licensed under the Apache License, Version 2.0.
 *
 * Modifications by NVIDIA:
 * - Ported simple selective state update kernel for TensorRT Edge-LLM
 * - Added explicit stride-based memory access for padded layouts
 * - Renamed namespace from flashinfer::mamba to mamba_ssm
 * - Added BFloat16 template instantiation
 * - Replaced FLASHINFER_CHECK with direct std::runtime_error throws
 */

#include "selectiveStateUpdate.h"

#include "common.cuh"
#include "common/checkMacros.h"
#include "conversion.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mamba_ssm
{

namespace
{

constexpr int32_t kStateCopyBlockSize{256};

template <bool Scatter>
__global__ void copyIndexedStateRowsKernel(
    uint8_t const* source, uint8_t* destination, int32_t const* stateIndices, int32_t residentRows, int64_t rowBytes)
{
    int64_t const activeRow = blockIdx.x;
    int64_t const residentRow = stateIndices[activeRow];
    bool const validResidentRow = residentRow >= 0 && residentRow < residentRows;
    if constexpr (Scatter)
    {
        if (!validResidentRow)
        {
            return;
        }
    }
    int64_t const sourceRow = Scatter ? activeRow : residentRow;
    int64_t const destinationRow = Scatter ? residentRow : activeRow;
    uint8_t const* sourceRowPtr = validResidentRow ? source + sourceRow * rowBytes : nullptr;
    uint8_t* destinationRowPtr = destination + destinationRow * rowBytes;
    if (rowBytes % static_cast<int64_t>(sizeof(uint4)) == 0)
    {
        int64_t const vectorCount = rowBytes / static_cast<int64_t>(sizeof(uint4));
        auto const* sourceVectors = reinterpret_cast<uint4 const*>(sourceRowPtr);
        auto* destinationVectors = reinterpret_cast<uint4*>(destinationRowPtr);
        for (int64_t vector = threadIdx.x; vector < vectorCount; vector += blockDim.x)
        {
            destinationVectors[vector] = validResidentRow ? sourceVectors[vector] : uint4{};
        }
    }
    else
    {
        for (int64_t byte = threadIdx.x; byte < rowBytes; byte += blockDim.x)
        {
            destinationRowPtr[byte] = validResidentRow ? sourceRowPtr[byte] : uint8_t{};
        }
    }
}

__global__ void clearMambaStateKernel(MambaStateLayerInfo const* layerInfos, int32_t residentRows, int32_t slot,
    int64_t recurrentRowBytes, int64_t convRowBytes)
{
    int32_t const layer = static_cast<int32_t>(blockIdx.x);
    int64_t const firstRow = slot < 0 ? 0 : slot;
    int64_t const rows = slot < 0 ? residentRows : 1;
    int64_t const recurrentBytes = rows * recurrentRowBytes;
    int64_t const convBytes = rows * convRowBytes;
    int64_t const totalBytes = recurrentBytes + convBytes;
    int64_t const byte = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(gridDim.y) * blockDim.x;

    auto const info = layerInfos[layer];
    auto* recurrent = static_cast<uint8_t*>(info.recurrentState) + firstRow * recurrentRowBytes;
    auto* conv = static_cast<uint8_t*>(info.convState) + firstRow * convRowBytes;
    for (int64_t offset = byte; offset < totalBytes; offset += stride)
    {
        if (offset < recurrentBytes)
        {
            recurrent[offset] = uint8_t{};
        }
        else
        {
            conv[offset - recurrentBytes] = uint8_t{};
        }
    }
}

int64_t logicalStateRowBytes(trt_edgellm::rt::Tensor const& state)
{
    auto const& shape = state.getShape();
    ELLM_CHECK(shape.getNumDims() > 0 && shape[0] > 0, "Mamba state must have a positive row extent.");
    int64_t const elements = shape.volume() / shape[0];
    int64_t const elementBytes = static_cast<int64_t>(trt_edgellm::rt::utils::getTypeSize(state.getDataType()));
    ELLM_CHECK(elements > 0 && elementBytes > 0, "Mamba state row size must be positive.");
    return elements * elementBytes;
}

void validateStateCopyTensors(trt_edgellm::rt::Tensor const& residentState, trt_edgellm::rt::Tensor const& activeState,
    trt_edgellm::rt::Tensor const& stateIndices)
{
    ELLM_CHECK(residentState.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && activeState.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && stateIndices.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU,
        "Mamba active state, resident state, and state indices must be GPU tensors.");
    ELLM_CHECK(
        residentState.getDataType() == activeState.getDataType(), "Mamba active and resident state types must match.");
    ELLM_CHECK(residentState.getShape().getNumDims() == 4 && activeState.getShape().getNumDims() == 4,
        "Mamba active and resident state must be rank four.");
    ELLM_CHECK(stateIndices.getDataType() == nvinfer1::DataType::kINT32 && stateIndices.getShape().getNumDims() == 1,
        "Mamba state indices must be rank-one INT32.");
    ELLM_CHECK(activeState.getShape()[0] == stateIndices.getShape()[0],
        "Mamba active state and state-index extents must match.");
    for (int32_t axis = 1; axis < 4; ++axis)
    {
        ELLM_CHECK(activeState.getShape()[axis] == residentState.getShape()[axis],
            "Mamba active and resident state trailing dimensions must match.");
    }
}

} // namespace

// Internal parameter struct (not exposed in the public header).
struct SelectiveStateUpdateParams
{
    uint32_t batch{}, nheads{}, dim{}, dstate{}, ngroups{}, state_cache_size{};
    int32_t pad_slot_id{-1};

    // Batch strides
    int64_t x_stride_batch{};
    int64_t dt_stride_batch{};
    int64_t B_stride_batch{};
    int64_t C_stride_batch{};
    int64_t out_stride_batch{};
    int64_t z_stride_batch{};
    int64_t state_stride_batch{};

    // Head strides (for padded layouts, set to padded dimension size)
    int64_t x_stride_head{};
    int64_t z_stride_head{};
    int64_t out_stride_head{};
    int64_t state_stride_head{};
    int64_t state_stride_dim{};

    // Sequence strides (prefill kernel only)
    int64_t x_stride_seq{};
    int64_t dt_stride_seq{};
    int64_t B_stride_seq{};
    int64_t C_stride_seq{};
    int64_t out_stride_seq{};
    int32_t seq_len{1};
    int32_t const* context_lengths{nullptr}; // Per-batch actual token count (prefill only)

    // MTP spec-verify (replay): rather than snapshotting the full [nheads, dim, dstate] recurrent
    // state after every draft token, the prefill kernel stashes the minimal per-token replay inputs
    // needed to rebuild the accepted state. During verification the committed state is left read-only
    // (update_state=false); after acceptance the runtime replays the recurrence
    //   dB = B * dt; S = dA * S + dB * x
    // over the first ``p`` accepted tokens from the read-only committed state (see
    // invokeMambaReplayReconstruct). The stash buffers are fp32 and contiguous:
    //   replay_dA [batch, replay_seq_len, nheads]           per-token decay exp(A * softplus(dt))
    //   replay_u  [batch, replay_seq_len, nheads, dim]      per-token unscaled input x
    //   replay_B  [batch, replay_seq_len, ngroups, dstate]  per-token key B
    //   replay_dt [batch, replay_seq_len, nheads]           per-token discretization step dt
    // replay_seq_len is the seq-dim capacity; writes at t >= replay_seq_len are dropped (CUDA-graph
    // capture may pass a dummy context length exceeding the buffer extent).
    void* __restrict__ replay_dA{nullptr};
    void* __restrict__ replay_u{nullptr};
    void* __restrict__ replay_B{nullptr};
    void* __restrict__ replay_dt{nullptr};
    int64_t replay_seq_len{};

    void* __restrict__ state{nullptr};
    void* __restrict__ x{nullptr};
    void* __restrict__ dt{nullptr};
    void* __restrict__ dt_bias{nullptr};
    void* __restrict__ A{nullptr};
    void* __restrict__ B{nullptr};
    void* __restrict__ C{nullptr};
    void* __restrict__ D{nullptr};
    void* __restrict__ z{nullptr};
    void* __restrict__ output{nullptr};
    void* __restrict__ state_batch_indices{nullptr};

    bool dt_softplus{false};
    bool update_state{true};
};

static void setOptionalStateIndices(
    SelectiveStateUpdateParams& params, trt_edgellm::rt::OptionalInputTensor const& stateIndices, int32_t activeRows)
{
    if (!stateIndices.has_value())
    {
        return;
    }
    auto const& indices = stateIndices->get();
    ELLM_CHECK(indices.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU
            && indices.getDataType() == nvinfer1::DataType::kINT32 && indices.getShape().getNumDims() == 1
            && indices.getShape()[0] == activeRows,
        "Mamba state indices must be GPU INT32 with shape [activeBatchSize].");
    params.state_batch_indices = const_cast<int32_t*>(indices.dataPointer<int32_t>());
}

inline void setContiguousStrides(SelectiveStateUpdateParams& params)
{
    params.x_stride_head = params.dim;
    params.z_stride_head = params.dim;
    params.out_stride_head = params.dim;
    params.state_stride_dim = params.dstate;
    params.state_stride_head = params.dim * params.dstate;
    params.x_stride_batch = params.nheads * params.x_stride_head;
    params.z_stride_batch = params.nheads * params.z_stride_head;
    params.out_stride_batch = params.nheads * params.out_stride_head;
    params.state_stride_batch = params.nheads * params.state_stride_head;
    params.dt_stride_batch = params.nheads;
    params.B_stride_batch = params.ngroups * params.dstate;
    params.C_stride_batch = params.ngroups * params.dstate;
}

using namespace conversion;

template <typename state_t>
__device__ __forceinline__ float roundTripState(float value)
{
    state_t rounded;
    convertAndStore(&rounded, value);
    return toFloat(rounded);
}

// Allowed (dim, dstate) for kernel instantiation
using AllowedDims = std::integer_sequence<int, 64, 80, 128, 256>;
using AllowedDstates = std::integer_sequence<int, 64, 80, 128, 256>;

// Shared memory structure for simple kernel.
template <typename input_t, int dim, int dstate>
struct SharedStorageSimple
{
    alignas(alignof(PackedAligned<input_t>)) input_t x[dim];
    alignas(alignof(PackedAligned<input_t>)) input_t z[dim];
    alignas(alignof(PackedAligned<input_t>)) input_t B[dstate];
    alignas(alignof(PackedAligned<input_t>)) input_t C[dstate];
    float out[dim];
};

// Simple selective state update kernel (works on all GPU architectures).
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t, int DIM,
    int DSTATE, int numWarps>
__global__ void selective_state_update_kernel_simple(SelectiveStateUpdateParams params)
{
    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);
    auto* __restrict__ state = reinterpret_cast<state_t*>(params.state);

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt);
    auto const* __restrict__ A = reinterpret_cast<matrixA_t const*>(params.A);
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);
    auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
    auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
    auto const* __restrict__ state_batch_indices = reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
    bool const dt_softplus = params.dt_softplus;

    int const nheads = params.nheads;
    int const ngroups = params.ngroups;

    constexpr auto rowsPerWarp = (DIM + numWarps - 1) / numWarps;

    auto const batch = blockIdx.x;
    auto const head = blockIdx.y;
    auto const group = head / (nheads / ngroups);
    auto lane = threadIdx.x % kWARP_SIZE;
    auto warp = threadIdx.y;

    int32_t const state_batch
        = state_batch_indices ? static_cast<int32_t>(state_batch_indices[batch]) : static_cast<int32_t>(batch);
    bool const validSlot = state_batch >= 0 && state_batch < static_cast<int32_t>(params.state_cache_size)
        && (params.pad_slot_id < 0 || state_batch != params.pad_slot_id);
    state_t* const stateRow = validSlot
        ? state + static_cast<int64_t>(state_batch) * params.state_stride_batch + head * params.state_stride_head
        : nullptr;

    __shared__ SharedStorageSimple<input_t, DIM, DSTATE> sram;

    static constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
    using load_state_t = PackedAligned<state_t, stateLoadSize>;
    using load_input_t = PackedAligned<input_t>;

    auto const A_value = toFloat(A[head]);

    auto dt_value = toFloat(dt[batch * params.dt_stride_batch + head]);
    if (dt_bias)
        dt_value += toFloat(dt_bias[head]);
    if (dt_softplus)
    {
        dt_value = thresholded_softplus(dt_value);
    }

    auto const dA = __expf(A_value * dt_value);

    auto d_value = D ? toFloat(D[head]) : 0.f;

    // Load x and B (warp 0)
    if (warp == 0)
    {
        for (auto d = lane * load_input_t::count; d < DIM; d += kWARP_SIZE * load_input_t::count)
        {
            auto* dst = reinterpret_cast<load_input_t*>(&sram.x[d]);
            *dst = *reinterpret_cast<load_input_t const*>(
                &x[batch * params.x_stride_batch + head * params.x_stride_head + d]);
        }
        for (auto i = lane * load_input_t::count; i < DSTATE; i += kWARP_SIZE * load_input_t::count)
        {
            auto* dst = reinterpret_cast<load_input_t*>(&sram.B[i]);
            *dst = *reinterpret_cast<load_input_t const*>(&B[batch * params.B_stride_batch + group * DSTATE + i]);
        }
    }
    // Load z and C (warp 1)
    else if (warp == 1)
    {
        for (auto d = lane * load_input_t::count; d < DIM; d += kWARP_SIZE * load_input_t::count)
        {
            auto* dst = reinterpret_cast<load_input_t*>(&sram.z[d]);
            *dst = z ? *reinterpret_cast<load_input_t const*>(
                           &z[batch * params.z_stride_batch + head * params.z_stride_head + d])
                     : make_zeros<load_input_t>();
        }
        for (auto i = lane * load_input_t::count; i < DSTATE; i += kWARP_SIZE * load_input_t::count)
        {
            auto* dst = reinterpret_cast<load_input_t*>(&sram.C[i]);
            *dst = *reinterpret_cast<load_input_t const*>(&C[batch * params.C_stride_batch + group * DSTATE + i]);
        }
    }
    __syncthreads();

    // Main computation loop: each warp processes a subset of dim rows
    for (auto _d = warp * rowsPerWarp; _d < (warp + 1) * rowsPerWarp; _d++)
    {
        auto d = _d;
        if (d >= DIM)
            break;

        float x_value = toFloat(sram.x[_d]);
        // D*x is a scalar contribution added once per dim row; only lane 0 seeds it
        // so after warpReduceSum it is included exactly once in the final output.
        float out_value = (lane == 0) ? (d_value * x_value) : 0.0f;

        // Process state dimension
        for (int i = lane * load_state_t::count; i < DSTATE; i += kWARP_SIZE * load_state_t::count)
        {
            auto rState = make_zeros<load_state_t>();
            if (validSlot)
            {
                rState = *reinterpret_cast<load_state_t*>(&stateRow[d * params.state_stride_dim + i]);
            }

            for (int ii = 0; ii < load_state_t::count; ii++)
            {
                auto state_value = toFloat(rState.val[ii]);
                auto B_value = toFloat(sram.B[i + ii]);
                auto C_value = toFloat(sram.C[i + ii]);

                auto const dB = B_value * dt_value;
                auto const new_state = state_value * dA + dB * x_value;

                convertAndStore(&rState.val[ii], new_state);

                out_value += new_state * C_value;
            }
            if (params.update_state && validSlot)
            {
                *reinterpret_cast<load_state_t*>(&stateRow[d * params.state_stride_dim + i]) = rState;
            }
        }

        // Warp reduce the output value
        out_value = warpReduceSum(out_value);
        if (lane == 0)
        {
            sram.out[_d] = out_value;
        }
    }

    __syncthreads();

    // Write output with optional SiLU gating
    for (int l = lane; l < rowsPerWarp; l += kWARP_SIZE)
    {
        auto d = warp * rowsPerWarp + l;
        if (d < DIM)
        {
            auto out_value = sram.out[d];
            if (z)
            {
                float z_value = toFloat(sram.z[d]);
                float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
                float silu_z = z_value * sig_z;
                out_value *= silu_z;
            }
            convertAndStore(&output[batch * params.out_stride_batch + head * params.out_stride_head + d], out_value);
        }
    }
}

// Prefill kernel: full-sequence SSM scan with fp32 state kept in registers.
// Unlike selective_state_update_kernel_simple (which is launched once per token
// from a host-side loop), this kernel processes the entire token sequence inside
// a single CUDA kernel.  The running SSM state is accumulated as float in
// registers across all time steps and written to global memory only once at the
// end, eliminating the fp16 quantisation round-trip that the host loop incurs on
// every token.
//
// Loop nesting:  (d-outer) → (t-middle) → (dstate-inner)
// Each lane holds DSTATE/kWARP_SIZE = 4 float registers for its state slice.
// No shared memory is used, so no __syncthreads() is required.
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t, int DIM,
    int DSTATE, int numWarps>
__global__ void selective_state_update_prefill_kernel_simple(SelectiveStateUpdateParams params)
{
    // Ceil division: e.g. DSTATE=80, kWARP_SIZE=32 → dstatePerLane=3.
    // Last few lanes may own elements with i >= DSTATE; those are guarded below.
    constexpr int dstatePerLane = (DSTATE + kWARP_SIZE - 1) / kWARP_SIZE;

    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);
    auto* __restrict__ state = reinterpret_cast<state_t*>(params.state);

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt);
    auto const* __restrict__ A = reinterpret_cast<matrixA_t const*>(params.A);
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);
    auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);

    int const nheads = params.nheads;
    int const ngroups = params.ngroups;

    auto const batch = blockIdx.x;
    auto const* stateIndices = static_cast<int32_t const*>(params.state_batch_indices);
    auto const stateBatch = stateIndices ? stateIndices[batch] : static_cast<int32_t>(batch);
    auto const head = blockIdx.y;
    auto const group = head / (nheads / ngroups);
    auto const lane = threadIdx.x % kWARP_SIZE;
    auto const warp = threadIdx.y;

    bool const validSlot = stateBatch >= 0 && stateBatch < static_cast<int32_t>(params.state_cache_size)
        && (params.pad_slot_id < 0 || stateBatch != params.pad_slot_id);
    state_t* const stateRow = validSlot
        ? state + static_cast<int64_t>(stateBatch) * params.state_stride_batch + head * params.state_stride_head
        : nullptr;

    auto const A_value = toFloat(A[head]);
    auto const d_value = D ? toFloat(D[head]) : 0.f;
    auto const effectiveSeqLen = params.context_lengths ? params.context_lengths[batch] : params.seq_len;

    constexpr auto rowsPerWarp = (DIM + numWarps - 1) / numWarps;

    // Process each dim row assigned to this warp.
    for (int _d = warp * rowsPerWarp; _d < (warp + 1) * rowsPerWarp; _d++)
    {
        if (_d >= DIM)
            break;

        // Load initial SSM state into fp32 registers (no quantisation here).
        float runState[dstatePerLane];
#pragma unroll
        for (int ii = 0; ii < dstatePerLane; ++ii)
        {
            int const i = lane * dstatePerLane + ii;
            runState[ii] = (validSlot && i < DSTATE) ? toFloat(stateRow[_d * params.state_stride_dim + i]) : 0.f;
        }

        // Scan over the token sequence, state stays fp32 in registers.
        for (int32_t t = 0; t < effectiveSeqLen; ++t)
        {
            // dt[batch, t, head]
            float dt_val = toFloat(dt[batch * params.dt_stride_batch + t * params.dt_stride_seq + head]);
            if (dt_bias)
                dt_val += toFloat(dt_bias[head]);
            if (params.dt_softplus)
                dt_val = thresholded_softplus(dt_val);
            float const dA = __expf(A_value * dt_val);

            // x[batch, t, head, _d]
            float const x_val = toFloat(
                x[batch * params.x_stride_batch + t * params.x_stride_seq + head * params.x_stride_head + _d]);

            // D * x contribution: only lane 0 adds it to avoid double-counting
            // in the warp reduce that follows.
            float out_val = (lane == 0) ? d_value * x_val : 0.f;

            // State update and output accumulation (over this lane's dstate slice).
#pragma unroll
            for (int ii = 0; ii < dstatePerLane; ++ii)
            {
                int const i = lane * dstatePerLane + ii;
                if (i < DSTATE)
                {
                    float const B_val
                        = toFloat(B[batch * params.B_stride_batch + t * params.B_stride_seq + group * DSTATE + i]);
                    float const C_val
                        = toFloat(C[batch * params.C_stride_batch + t * params.C_stride_seq + group * DSTATE + i]);
                    // The current-token output uses FP32 state before the token-boundary state_t round-trip below.
                    float const dB = B_val * dt_val;
                    runState[ii] = runState[ii] * dA + dB * x_val;
                    out_val += runState[ii] * C_val;
                }
            }

            // Warp reduce: sum contributions across lanes (each handles a dstate slice).
            out_val = warpReduceSum(out_val);

            // Lane 0 writes the output token to global memory.
            if (lane == 0)
            {
                convertAndStore(&output[batch * params.out_stride_batch + t * params.out_stride_seq
                                    + head * params.out_stride_head + _d],
                    out_val);
            }

            // MTP spec-verify (replay): stash the minimal per-token replay inputs so the accepted
            // state can be reconstructed from the read-only committed state after verification.
            // Clamp to the buffer's seq capacity: during CUDA-graph capture effectiveSeqLen can
            // exceed replay_seq_len (dummy context lengths), which would write OOB.
            if (params.replay_u && t < params.replay_seq_len)
            {
                int64_t const tokenRow = static_cast<int64_t>(batch) * params.replay_seq_len + t;
                // x is identical across lanes, so only lane 0 writes.
                if (lane == 0)
                {
                    reinterpret_cast<float*>(params.replay_u)[(tokenRow * nheads + head) * DIM + _d] = x_val;
                    // dA[batch, t, head] — one value per (batch, t, head).
                    if (_d == 0)
                    {
                        reinterpret_cast<float*>(params.replay_dA)[tokenRow * nheads + head] = dA;
                        reinterpret_cast<float*>(params.replay_dt)[tokenRow * nheads + head] = dt_val;
                    }
                }
                // B[batch, t, group, i] — one value per (batch, t, group); the first head of each
                // group (first dim row) writes it, lanes split the dstate slice.
                if (_d == 0 && (head % (nheads / ngroups)) == 0)
                {
                    auto* __restrict__ replayB = reinterpret_cast<float*>(params.replay_B);
                    int64_t const bOff = (tokenRow * ngroups + group) * DSTATE;
#pragma unroll
                    for (int ii = 0; ii < dstatePerLane; ++ii)
                    {
                        int const i = lane * dstatePerLane + ii;
                        if (i < DSTATE)
                            replayB[bOff + i] = toFloat(
                                B[batch * params.B_stride_batch + t * params.B_stride_seq + group * DSTATE + i]);
                    }
                }
            }

            // Decode persists state_t after every token. Preserve that boundary during
            // speculative verification so later rows do not depend on the verify grouping.
            if (params.replay_u && t + 1 < effectiveSeqLen)
            {
#pragma unroll
                for (int ii = 0; ii < dstatePerLane; ++ii)
                {
                    int const i = lane * dstatePerLane + ii;
                    if (i < DSTATE)
                    {
                        runState[ii] = roundTripState<state_t>(runState[ii]);
                    }
                }
            }
        }

        // Zero output at padded positions.
        for (int32_t t = effectiveSeqLen; t < params.seq_len; ++t)
        {
            if (lane == 0)
            {
                convertAndStore(&output[batch * params.out_stride_batch + t * params.out_stride_seq
                                    + head * params.out_stride_head + _d],
                    0.f);
            }
        }

        // Write final state to global memory (one quantisation per sequence).
        if (params.update_state && validSlot)
        {
#pragma unroll
            for (int ii = 0; ii < dstatePerLane; ++ii)
            {
                int const i = lane * dstatePerLane + ii;
                if (i < DSTATE)
                    convertAndStore(&stateRow[_d * params.state_stride_dim + i], runState[ii]);
            }
        }
    }
}

// DDTree verify reconstructs every node from the committed state along its own ancestor path.
constexpr int32_t kMAMBA_TREE_MAX_PATH{128};

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, int DIM, int DSTATE, int numWarps>
__global__ void mamba_tree_verify_kernel(
    SelectiveStateUpdateParams params, int32_t const* treeParentIds, int32_t const* treeDepths)
{
    constexpr int32_t dstatePerLane = (DSTATE + kWARP_SIZE - 1) / kWARP_SIZE;
    __shared__ int32_t path[kMAMBA_TREE_MAX_PATH];
    __shared__ int32_t pathLength;
    __shared__ float pathDt[kMAMBA_TREE_MAX_PATH];
    __shared__ float pathDA[kMAMBA_TREE_MAX_PATH];

    int32_t const batchNode = blockIdx.x;
    int32_t const batch = batchNode / params.seq_len;
    int32_t const node = batchNode % params.seq_len;
    int32_t const head = blockIdx.y;
    int32_t const lane = threadIdx.x;
    int32_t const warp = threadIdx.y;
    int32_t const group = head / (params.nheads / params.ngroups);
    auto const* stateIndices = static_cast<int32_t const*>(params.state_batch_indices);
    int32_t const stateBatch = stateIndices != nullptr ? stateIndices[batch] : batch;
    bool const validSlot = stateBatch >= 0 && stateBatch < static_cast<int32_t>(params.state_cache_size)
        && (params.pad_slot_id < 0 || stateBatch != params.pad_slot_id);

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        int32_t const treeOffset = batch * params.seq_len;
        int32_t const depth = treeDepths[treeOffset + node];
        bool valid = depth >= 0 && depth < kMAMBA_TREE_MAX_PATH;
        int32_t current = node;
        if (valid)
        {
            for (int32_t pos = depth; pos >= 0; --pos)
            {
                if (current < 0 || current >= params.seq_len)
                {
                    valid = false;
                    break;
                }
                path[pos] = current;
                int32_t const parent = treeParentIds[treeOffset + current];
                if ((pos == 0 && (current != 0 || parent >= 0)) || (pos > 0 && (parent < 0 || parent >= current)))
                {
                    valid = false;
                    break;
                }
                current = parent;
            }
        }
        pathLength = valid ? depth + 1 : 0;
    }
    __syncthreads();

    auto const* x = static_cast<input_t const*>(params.x);
    auto const* dt = static_cast<weight_t const*>(params.dt);
    auto const* dtBias = static_cast<weight_t const*>(params.dt_bias);
    auto const* A = static_cast<matrixA_t const*>(params.A);
    auto const* B = static_cast<input_t const*>(params.B);
    auto const* C = static_cast<input_t const*>(params.C);
    auto const* D = static_cast<input_t const*>(params.D);
    auto const* state = validSlot ? static_cast<state_t const*>(params.state)
            + static_cast<int64_t>(stateBatch) * params.state_stride_batch + head * params.state_stride_head
                                  : nullptr;
    auto* output = static_cast<input_t*>(params.output);

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        for (int32_t pos = 0; pos < pathLength; ++pos)
        {
            int32_t const token = path[pos];
            float dtValue = toFloat(dt[batch * params.dt_stride_batch + token * params.dt_stride_seq + head]);
            if (dtBias != nullptr)
            {
                dtValue += toFloat(dtBias[head]);
            }
            if (params.dt_softplus)
            {
                dtValue = thresholded_softplus(dtValue);
            }
            pathDt[pos] = dtValue;
            pathDA[pos] = __expf(toFloat(A[head]) * dtValue);
        }
    }
    __syncthreads();

    constexpr int32_t rowsPerWarp = (DIM + numWarps - 1) / numWarps;
    for (int32_t d = warp * rowsPerWarp; d < (warp + 1) * rowsPerWarp && d < DIM; ++d)
    {
        float runState[dstatePerLane];
#pragma unroll
        for (int32_t ii = 0; ii < dstatePerLane; ++ii)
        {
            int32_t const i = lane * dstatePerLane + ii;
            runState[ii] = i < DSTATE && validSlot ? toFloat(state[d * params.state_stride_dim + i]) : 0.F;
        }

        for (int32_t pos = 0; pos < pathLength; ++pos)
        {
            int32_t const token = path[pos];
            float const dtValue = pathDt[pos];
            float const dA = pathDA[pos];
            float const xValue = toFloat(
                x[batch * params.x_stride_batch + token * params.x_stride_seq + head * params.x_stride_head + d]);

#pragma unroll
            for (int32_t ii = 0; ii < dstatePerLane; ++ii)
            {
                int32_t const i = lane * dstatePerLane + ii;
                if (i < DSTATE)
                {
                    float const bValue
                        = toFloat(B[batch * params.B_stride_batch + token * params.B_stride_seq + group * DSTATE + i]);
                    float const dB = bValue * dtValue;
                    runState[ii] = runState[ii] * dA + dB * xValue;
                }
            }

            if (token == node)
            {
                int64_t const tokenRow = static_cast<int64_t>(batch) * params.replay_seq_len + node;
                if (lane == 0)
                {
                    reinterpret_cast<float*>(params.replay_u)[(tokenRow * params.nheads + head) * DIM + d] = xValue;
                    if (d == 0)
                    {
                        reinterpret_cast<float*>(params.replay_dA)[tokenRow * params.nheads + head] = dA;
                        reinterpret_cast<float*>(params.replay_dt)[tokenRow * params.nheads + head] = dtValue;
                    }
                }
                if (d == 0 && head % (params.nheads / params.ngroups) == 0)
                {
                    float* replayB
                        = reinterpret_cast<float*>(params.replay_B) + (tokenRow * params.ngroups + group) * DSTATE;
#pragma unroll
                    for (int32_t ii = 0; ii < dstatePerLane; ++ii)
                    {
                        int32_t const i = lane * dstatePerLane + ii;
                        if (i < DSTATE)
                        {
                            replayB[i] = toFloat(
                                B[batch * params.B_stride_batch + token * params.B_stride_seq + group * DSTATE + i]);
                        }
                    }
                }
            }

            if (pos + 1 < pathLength)
            {
#pragma unroll
                for (int32_t ii = 0; ii < dstatePerLane; ++ii)
                {
                    int32_t const i = lane * dstatePerLane + ii;
                    if (i < DSTATE)
                    {
                        runState[ii] = roundTripState<state_t>(runState[ii]);
                    }
                }
            }
        }

        float outValue = 0.F;
        if (pathLength > 0)
        {
            float const xValue = toFloat(
                x[batch * params.x_stride_batch + node * params.x_stride_seq + head * params.x_stride_head + d]);
            outValue = lane == 0 && D != nullptr ? toFloat(D[head]) * xValue : 0.F;
#pragma unroll
            for (int32_t ii = 0; ii < dstatePerLane; ++ii)
            {
                int32_t const i = lane * dstatePerLane + ii;
                if (i < DSTATE)
                {
                    float const cValue
                        = toFloat(C[batch * params.C_stride_batch + node * params.C_stride_seq + group * DSTATE + i]);
                    outValue += runState[ii] * cValue;
                }
            }
            outValue = warpReduceSum(outValue);
        }
        if (lane == 0)
        {
            convertAndStore(&output[batch * params.out_stride_batch + node * params.out_stride_seq
                                + head * params.out_stride_head + d],
                outValue);
        }
    }
}

// MTP spec-verify (replay): rebuild the committed recurrent state after acceptance by re-running the
// SSD recurrence  S = dA * S + u ⊗ B  over the first ``p`` accepted tokens (params.context_lengths[b])
// from the read-only committed state. The stash (replay_dA/u/B) was produced by the prefill kernel.
// Same block/warp/lane mapping as the forward scan; the update is in-place on disjoint state elements.
template <typename state_t, int DIM, int DSTATE, int numWarps>
__global__ void mamba_replay_reconstruct_batched_kernel(MambaReplayLayerInfo const* __restrict__ layerInfos,
    SelectiveStateUpdateParams params, int32_t const* __restrict__ acceptedNodeIds, int32_t maxAcceptLen)
{
    constexpr int dstatePerLane = (DSTATE + kWARP_SIZE - 1) / kWARP_SIZE;

    int32_t const layer = blockIdx.z;
    MambaReplayLayerInfo const& layerInfo = layerInfos[layer];
    auto* __restrict__ state = static_cast<state_t*>(layerInfo.stateDst);
    auto const* __restrict__ replayDA = static_cast<float const*>(layerInfo.replayDa);
    auto const* __restrict__ replayU = static_cast<float const*>(layerInfo.replayU);
    auto const* __restrict__ replayB = static_cast<float const*>(layerInfo.replayB);
    auto const* __restrict__ replayDt = static_cast<float const*>(layerInfo.replayDt);

    int const nheads = params.nheads;
    int const ngroups = params.ngroups;

    auto const batch = blockIdx.x;
    auto const* stateIndices = static_cast<int32_t const*>(params.state_batch_indices);
    auto const stateBatch = stateIndices ? stateIndices[batch] : static_cast<int32_t>(batch);
    auto const head = blockIdx.y;
    auto const group = head / (nheads / ngroups);
    auto const lane = threadIdx.x % kWARP_SIZE;
    auto const warp = threadIdx.y;

    bool const validSlot = stateBatch >= 0 && stateBatch < static_cast<int32_t>(params.state_cache_size)
        && (params.pad_slot_id < 0 || stateBatch != params.pad_slot_id);
    // Accepted-token count for this batch; 0 accepted => committed state already correct.
    int const p = params.context_lengths ? params.context_lengths[batch] : params.seq_len;
    if (!validSlot || p <= 0)
        return;

    state += static_cast<int64_t>(stateBatch) * params.state_stride_batch + head * params.state_stride_head;
    constexpr auto rowsPerWarp = (DIM + numWarps - 1) / numWarps;

    for (int _d = warp * rowsPerWarp; _d < (warp + 1) * rowsPerWarp; _d++)
    {
        if (_d >= DIM)
            break;

        float runState[dstatePerLane];
#pragma unroll
        for (int ii = 0; ii < dstatePerLane; ++ii)
        {
            int const i = lane * dstatePerLane + ii;
            runState[ii] = (i < DSTATE) ? toFloat(state[_d * params.state_stride_dim + i]) : 0.f;
        }

        int const replaySeqLen = static_cast<int>(params.replay_seq_len);
        int const replayCount
            = acceptedNodeIds != nullptr ? min(min(p, replaySeqLen), maxAcceptLen) : min(p, replaySeqLen);
        for (int t = 0; t < replayCount; ++t)
        {
            int32_t const token = acceptedNodeIds != nullptr ? acceptedNodeIds[batch * maxAcceptLen + t] : t;
            if (token < 0 || token >= params.replay_seq_len)
            {
                continue;
            }
            int64_t const tokenRow = static_cast<int64_t>(batch) * params.replay_seq_len + token;
            float const dA = replayDA[tokenRow * nheads + head];
            float const x = replayU[(tokenRow * nheads + head) * DIM + _d];
            float const dt = replayDt[tokenRow * nheads + head];
            int64_t const bOff = (tokenRow * ngroups + group) * DSTATE;
#pragma unroll
            for (int ii = 0; ii < dstatePerLane; ++ii)
            {
                int const i = lane * dstatePerLane + ii;
                if (i < DSTATE)
                {
                    float const dB = replayB[bOff + i] * dt;
                    runState[ii] = roundTripState<state_t>(runState[ii] * dA + dB * x);
                }
            }
        }

#pragma unroll
        for (int ii = 0; ii < dstatePerLane; ++ii)
        {
            int const i = lane * dstatePerLane + ii;
            if (i < DSTATE)
                convertAndStore(&state[_d * params.state_stride_dim + i], runState[ii]);
        }
    }
}

// Kernel launcher functors (at namespace scope for nvcc compatibility).
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t>
struct SsmKernelLauncher
{
    SelectiveStateUpdateParams& params;
    cudaStream_t stream;

    template <int DIM, int DSTATE>
    void operator()()
    {
        constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
        using load_state_t = PackedAligned<state_t, stateLoadSize>;

        auto const stateAlign = std::to_string(sizeof(load_state_t));
        ELLM_CHECK(reinterpret_cast<uintptr_t>(params.state) % sizeof(load_state_t) == 0,
            "state pointer must be aligned to " + stateAlign + " bytes");
        ELLM_CHECK((params.dim * params.dstate * sizeof(state_t)) % sizeof(load_state_t) == 0,
            "state head stride must be aligned to " + stateAlign + " bytes");

        constexpr int numWarps = 4;
        dim3 block(kWARP_SIZE, numWarps);
        dim3 grid(params.batch, params.nheads);
        selective_state_update_kernel_simple<input_t, weight_t, matrixA_t, state_t, stateIndex_t, DIM, DSTATE, numWarps>
            <<<grid, block, 0, stream>>>(params);
    }
};

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t>
struct SsmPrefillKernelLauncher
{
    SelectiveStateUpdateParams& params;
    cudaStream_t stream;

    template <int DIM, int DSTATE>
    void operator()()
    {
        constexpr int numWarps = 4;
        dim3 block(kWARP_SIZE, numWarps);
        dim3 grid(params.batch, params.nheads);
        selective_state_update_prefill_kernel_simple<input_t, weight_t, matrixA_t, state_t, stateIndex_t, DIM, DSTATE,
            numWarps><<<grid, block, 0, stream>>>(params);
    }
};

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t>
struct SsmTreeVerifyLauncher
{
    SelectiveStateUpdateParams& params;
    int32_t const* treeParentIds;
    int32_t const* treeDepths;
    cudaStream_t stream;

    template <int DIM, int DSTATE>
    void operator()()
    {
        constexpr int32_t numWarps = 16;
        dim3 const block(kWARP_SIZE, numWarps);
        dim3 const grid(params.batch * params.seq_len, params.nheads);
        mamba_tree_verify_kernel<input_t, weight_t, matrixA_t, state_t, DIM, DSTATE, numWarps>
            <<<grid, block, 0, stream>>>(params, treeParentIds, treeDepths);
    }
};
template <typename state_t>
struct SsmReplayReconstructLauncher
{
    SelectiveStateUpdateParams& params;
    MambaReplayLayerInfo const* layerInfos;
    int32_t numLayers;
    int32_t const* acceptedNodeIds;
    int32_t maxAcceptLen;
    cudaStream_t stream;

    template <int DIM, int DSTATE>
    void operator()()
    {
        constexpr int numWarps = 4;
        dim3 block(kWARP_SIZE, numWarps);
        dim3 grid(params.batch, params.nheads, numLayers);
        mamba_replay_reconstruct_batched_kernel<state_t, DIM, DSTATE, numWarps>
            <<<grid, block, 0, stream>>>(layerInfos, params, acceptedNodeIds, maxAcceptLen);
    }
};

// Internal kernel dispatch (params-based); not part of the public API.
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t>
static void invokeSelectiveStateUpdateImpl(SelectiveStateUpdateParams& params, cudaStream_t stream)
{
    check_ptr_alignment_input_vars<input_t>(params);

    SsmKernelLauncher<input_t, weight_t, matrixA_t, state_t, stateIndex_t> launcher{params, stream};
    dispatchDimDstate(params, AllowedDims{}, AllowedDstates{}, launcher);
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t>
static void invokeSelectiveStateUpdatePrefillImpl(SelectiveStateUpdateParams& params, cudaStream_t stream)
{
    check_ptr_alignment_input_vars<input_t>(params);

    SsmPrefillKernelLauncher<input_t, weight_t, matrixA_t, state_t, stateIndex_t> launcher{params, stream};
    dispatchDimDstate(params, AllowedDims{}, AllowedDstates{}, launcher);
}

// Fill params from Tensor refs (common for decode and prefill).
static void fillCommonParamsFromTensors(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& A,
    trt_edgellm::rt::Tensor const& B, trt_edgellm::rt::Tensor const& C, trt_edgellm::rt::Tensor const& dt,
    trt_edgellm::rt::OptionalInputTensor dt_bias, trt_edgellm::rt::OptionalInputTensor D,
    trt_edgellm::rt::OptionalInputTensor z, trt_edgellm::rt::Tensor& state, trt_edgellm::rt::Tensor& output,
    bool dt_softplus, SelectiveStateUpdateParams& params)
{
    params.batch = static_cast<uint32_t>(x.getShape()[0]);
    params.state_cache_size = static_cast<uint32_t>(state.getShape()[0]);
    params.nheads = static_cast<uint32_t>(state.getShape()[1]);
    params.dim = static_cast<uint32_t>(state.getShape()[2]);
    params.dstate = static_cast<uint32_t>(state.getShape()[3]);
    auto const bndims = B.getShape().getNumDims();
    params.ngroups = static_cast<uint32_t>(B.getShape()[bndims - 2]);
    params.dt_softplus = dt_softplus;
    params.update_state = true;

    params.state_stride_batch = state.getStride(0);
    params.state_stride_head = state.getStride(1);
    params.state_stride_dim = state.getStride(2);

    params.x = const_cast<void*>(x.rawPointer());
    params.A = const_cast<void*>(A.rawPointer());
    params.B = const_cast<void*>(B.rawPointer());
    params.C = const_cast<void*>(C.rawPointer());
    params.dt = const_cast<void*>(dt.rawPointer());
    params.dt_bias = dt_bias.has_value() ? const_cast<void*>(dt_bias->get().rawPointer()) : nullptr;
    params.D = D.has_value() ? const_cast<void*>(D->get().rawPointer()) : nullptr;
    params.z = z.has_value() ? const_cast<void*>(z->get().rawPointer()) : nullptr;
    params.state = state.rawPointer();
    params.output = output.rawPointer();
}

// Public non-templated API (decode).
void invokeSelectiveStateUpdate(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& A,
    trt_edgellm::rt::Tensor const& B, trt_edgellm::rt::Tensor const& C, trt_edgellm::rt::Tensor const& dt,
    trt_edgellm::rt::OptionalInputTensor dt_bias, trt_edgellm::rt::OptionalInputTensor D,
    trt_edgellm::rt::OptionalInputTensor z, trt_edgellm::rt::Tensor& state, trt_edgellm::rt::Tensor& output,
    bool dt_softplus, trt_edgellm::rt::OptionalInputTensor stateIndices, cudaStream_t stream)
{
    SelectiveStateUpdateParams params{};
    fillCommonParamsFromTensors(x, A, B, C, dt, dt_bias, D, z, state, output, dt_softplus, params);
    setOptionalStateIndices(params, stateIndices, static_cast<int32_t>(x.getShape()[0]));

    params.x_stride_batch = x.getStride(0);
    params.x_stride_head = x.getStride(1);
    params.dt_stride_batch = dt.getStride(0);
    params.B_stride_batch = B.getStride(0);
    params.C_stride_batch = C.getStride(0);
    params.out_stride_batch = output.getStride(0);
    params.out_stride_head = output.getStride(1);
    if (z.has_value())
    {
        params.z_stride_batch = z->get().getStride(0);
        params.z_stride_head = z->get().getStride(1);
    }

    if (x.getDataType() == nvinfer1::DataType::kHALF && dt.getDataType() == nvinfer1::DataType::kHALF)
    {
        invokeSelectiveStateUpdateImpl<half, half, float, half, int32_t>(params, stream);
    }
    else if (x.getDataType() == nvinfer1::DataType::kHALF && dt.getDataType() == nvinfer1::DataType::kFLOAT)
    {
        invokeSelectiveStateUpdateImpl<half, float, float, half, int32_t>(params, stream);
    }
    else
    {
        throw std::runtime_error("invokeSelectiveStateUpdate: only (x=half, dt=half or float) is supported.");
    }
}

// Public non-templated API (prefill).
void invokeSelectiveStateUpdatePrefill(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& A,
    trt_edgellm::rt::Tensor const& B, trt_edgellm::rt::Tensor const& C, trt_edgellm::rt::Tensor const& dt,
    trt_edgellm::rt::OptionalInputTensor dt_bias, trt_edgellm::rt::OptionalInputTensor D,
    trt_edgellm::rt::OptionalInputTensor z, trt_edgellm::rt::Tensor& state, trt_edgellm::rt::Tensor& output,
    bool dt_softplus, trt_edgellm::rt::OptionalInputTensor contextLengths,
    trt_edgellm::rt::OptionalInputTensor stateIndices, trt_edgellm::rt::OptionalOutputTensor replayDA,
    trt_edgellm::rt::OptionalOutputTensor replayU, trt_edgellm::rt::OptionalOutputTensor replayB,
    trt_edgellm::rt::OptionalOutputTensor replayDT, cudaStream_t stream)
{
    SelectiveStateUpdateParams params{};
    fillCommonParamsFromTensors(x, A, B, C, dt, dt_bias, D, z, state, output, dt_softplus, params);
    params.context_lengths = contextLengths.has_value() ? contextLengths->get().dataPointer<int32_t>() : nullptr;
    setOptionalStateIndices(params, stateIndices, static_cast<int32_t>(x.getShape()[0]));

    // MTP spec-verify (replay): optional per-token replay stash. When present, keep the committed
    // recurrent state read-only (the runtime reconstructs the accepted state after verification via
    // invokeMambaReplayReconstruct) and stash dA/dt/x/B [batch, seq_len, ...] for that replay.
    bool const hasReplay = replayDA.has_value() || replayU.has_value() || replayB.has_value() || replayDT.has_value();
    ELLM_CHECK(
        !hasReplay || (replayDA.has_value() && replayU.has_value() && replayB.has_value() && replayDT.has_value()),
        "Mamba replay outputs must provide dA, u, B, and dt together.");
    if (hasReplay)
    {
        params.replay_dA = replayDA->get().rawPointer();
        params.replay_u = replayU->get().rawPointer();
        params.replay_B = replayB->get().rawPointer();
        params.replay_seq_len = replayU->get().getShape()[1];
        params.replay_dt = replayDT->get().rawPointer();
        params.update_state = false;
    }

    params.seq_len = static_cast<int32_t>(x.getShape()[1]);
    params.x_stride_batch = x.getStride(0);
    params.x_stride_seq = x.getStride(1);
    params.x_stride_head = x.getStride(2);
    params.dt_stride_batch = dt.getStride(0);
    params.dt_stride_seq = dt.getStride(1);
    params.B_stride_batch = B.getStride(0);
    params.B_stride_seq = B.getStride(1);
    params.C_stride_batch = C.getStride(0);
    params.C_stride_seq = C.getStride(1);
    params.out_stride_batch = output.getStride(0);
    params.out_stride_seq = output.getStride(1);
    params.out_stride_head = output.getStride(2);
    if (z.has_value())
    {
        params.z_stride_batch = z->get().getStride(0);
        params.z_stride_head = z->get().getStride(2);
    }

    if (x.getDataType() == nvinfer1::DataType::kHALF && dt.getDataType() == nvinfer1::DataType::kHALF)
    {
        invokeSelectiveStateUpdatePrefillImpl<half, half, float, half, int32_t>(params, stream);
    }
    else if (x.getDataType() == nvinfer1::DataType::kHALF && dt.getDataType() == nvinfer1::DataType::kFLOAT)
    {
        invokeSelectiveStateUpdatePrefillImpl<half, float, float, half, int32_t>(params, stream);
    }
    else
    {
        throw std::runtime_error("invokeSelectiveStateUpdatePrefill: only (x=half, dt=half or float) is supported.");
    }
}

// Public non-templated API (DDTree spec-verify).
void invokeSelectiveStateUpdateDDTree(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& A,
    trt_edgellm::rt::Tensor const& B, trt_edgellm::rt::Tensor const& C, trt_edgellm::rt::Tensor const& dt,
    trt_edgellm::rt::OptionalInputTensor dtBias, trt_edgellm::rt::OptionalInputTensor D, trt_edgellm::rt::Tensor& state,
    trt_edgellm::rt::Tensor& output, trt_edgellm::rt::Tensor const& treeParentIds,
    trt_edgellm::rt::Tensor const& treeDepths, bool dtSoftplus, trt_edgellm::rt::OptionalInputTensor stateIndices,
    trt_edgellm::rt::Tensor& replayDA, trt_edgellm::rt::Tensor& replayU, trt_edgellm::rt::Tensor& replayB,
    trt_edgellm::rt::Tensor& replayDT, cudaStream_t stream)
{
    int32_t const seqLen = static_cast<int32_t>(x.getShape()[1]);
    if (treeParentIds.getShape().getNumDims() != 2 || treeDepths.getShape().getNumDims() != 2
        || treeParentIds.getShape()[0] != x.getShape()[0] || treeDepths.getShape()[0] != x.getShape()[0]
        || treeParentIds.getShape()[1] != seqLen || treeDepths.getShape()[1] != seqLen)
    {
        throw std::runtime_error("invokeSelectiveStateUpdateDDTree: tree metadata must have shape [batch, seq_len].");
    }
    if (treeParentIds.getDeviceType() != trt_edgellm::rt::DeviceType::kGPU
        || treeDepths.getDeviceType() != trt_edgellm::rt::DeviceType::kGPU
        || treeParentIds.getDataType() != nvinfer1::DataType::kINT32
        || treeDepths.getDataType() != nvinfer1::DataType::kINT32)
    {
        throw std::runtime_error("invokeSelectiveStateUpdateDDTree: tree metadata must be GPU INT32.");
    }

    SelectiveStateUpdateParams params{};
    fillCommonParamsFromTensors(x, A, B, C, dt, dtBias, D, std::nullopt, state, output, dtSoftplus, params);
    setOptionalStateIndices(params, stateIndices, static_cast<int32_t>(x.getShape()[0]));
    params.seq_len = seqLen;
    params.x_stride_batch = x.getStride(0);
    params.x_stride_seq = x.getStride(1);
    params.x_stride_head = x.getStride(2);
    params.dt_stride_batch = dt.getStride(0);
    params.dt_stride_seq = dt.getStride(1);
    params.B_stride_batch = B.getStride(0);
    params.B_stride_seq = B.getStride(1);
    params.C_stride_batch = C.getStride(0);
    params.C_stride_seq = C.getStride(1);
    params.out_stride_batch = output.getStride(0);
    params.out_stride_seq = output.getStride(1);
    params.out_stride_head = output.getStride(2);
    params.replay_dA = replayDA.rawPointer();
    params.replay_u = replayU.rawPointer();
    params.replay_B = replayB.rawPointer();
    params.replay_seq_len = replayU.getShape()[1];
    params.replay_dt = replayDT.rawPointer();
    params.update_state = false;

    if (x.getDataType() == nvinfer1::DataType::kHALF && dt.getDataType() == nvinfer1::DataType::kHALF)
    {
        SsmTreeVerifyLauncher<half, half, float, half> launcher{
            params, treeParentIds.dataPointer<int32_t>(), treeDepths.dataPointer<int32_t>(), stream};
        dispatchDimDstate(params, AllowedDims{}, AllowedDstates{}, launcher);
    }
    else if (x.getDataType() == nvinfer1::DataType::kHALF && dt.getDataType() == nvinfer1::DataType::kFLOAT)
    {
        SsmTreeVerifyLauncher<half, float, float, half> launcher{
            params, treeParentIds.dataPointer<int32_t>(), treeDepths.dataPointer<int32_t>(), stream};
        dispatchDimDstate(params, AllowedDims{}, AllowedDstates{}, launcher);
    }
    else
    {
        throw std::runtime_error("invokeSelectiveStateUpdateDDTree: only (x=half, dt=half or float) is supported.");
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

void invokeMambaStateClear(MambaStateLayerInfo const* deviceLayerInfos, int32_t numLayers, int32_t residentRows,
    int32_t slot, int64_t recurrentRowBytes, int64_t convRowBytes, cudaStream_t stream)
{
    if (numLayers == 0)
    {
        return;
    }
    ELLM_CHECK(deviceLayerInfos != nullptr, "Mamba state layer info must not be null.");
    ELLM_CHECK(numLayers > 0 && residentRows > 0 && slot >= -1 && slot < residentRows,
        "Mamba state clear received invalid row extents.");
    ELLM_CHECK(recurrentRowBytes > 0 && convRowBytes > 0, "Mamba state clear received an empty state row.");

    int64_t const rows = slot < 0 ? residentRows : 1;
    int64_t const totalBytes = rows * (recurrentRowBytes + convRowBytes);
    constexpr int32_t kMaxTilesPerLayer = 32;
    int64_t const tileBytes = static_cast<int64_t>(kStateCopyBlockSize) * 16;
    int64_t const requiredTiles = (totalBytes + tileBytes - 1) / tileBytes;
    int32_t const tiles = static_cast<int32_t>(requiredTiles < kMaxTilesPerLayer ? requiredTiles : kMaxTilesPerLayer);
    dim3 const grid(static_cast<uint32_t>(numLayers), static_cast<uint32_t>(tiles));
    clearMambaStateKernel<<<grid, kStateCopyBlockSize, 0, stream>>>(
        deviceLayerInfos, residentRows, slot, recurrentRowBytes, convRowBytes);
    CUDA_CHECK(cudaGetLastError());
}

// Public non-templated API (MTP spec-verify replay reconstruction).
void invokeMambaReplayReconstructBatched(MambaReplayLayerInfo const* deviceLayerInfos, int32_t numLayers,
    trt_edgellm::rt::Tensor const& state, trt_edgellm::rt::Tensor const& replayU,
    trt_edgellm::rt::Tensor const& replayB, trt_edgellm::rt::Tensor const& replayDT,
    trt_edgellm::rt::Tensor const& acceptedLengths, trt_edgellm::rt::Tensor const& stateIndices,
    int32_t activeBatchSize, cudaStream_t stream, trt_edgellm::rt::Tensor const* acceptedNodeIds, int32_t maxAcceptLen)
{
    if (activeBatchSize <= 0 || numLayers <= 0)
    {
        return;
    }
    if (deviceLayerInfos == nullptr)
    {
        throw std::runtime_error("invokeMambaReplayReconstructBatched: layer info must not be null.");
    }
    if (state.getShape().getNumDims() != 4 || activeBatchSize > static_cast<int32_t>(state.getShape()[0]))
    {
        throw std::runtime_error(
            "invokeMambaReplayReconstructBatched: state must be rank four and cover the active batch.");
    }
    if (stateIndices.getDeviceType() != trt_edgellm::rt::DeviceType::kGPU
        || stateIndices.getDataType() != nvinfer1::DataType::kINT32 || stateIndices.getShape().getNumDims() != 1
        || stateIndices.getShape()[0] != activeBatchSize)
    {
        throw std::runtime_error(
            "invokeMambaReplayReconstructBatched: stateIndices must be GPU INT32 [activeBatchSize].");
    }
    if (acceptedLengths.getDeviceType() != trt_edgellm::rt::DeviceType::kGPU
        || acceptedLengths.getDataType() != nvinfer1::DataType::kINT32 || acceptedLengths.getShape().getNumDims() != 1
        || acceptedLengths.getShape()[0] != activeBatchSize)
    {
        throw std::runtime_error(
            "invokeMambaReplayReconstructBatched: acceptedLengths must be GPU INT32 [activeBatchSize].");
    }
    if (replayU.getShape().getNumDims() != 4 || replayB.getShape().getNumDims() != 4
        || replayDT.getShape().getNumDims() != 3 || replayU.getShape()[0] != activeBatchSize
        || replayB.getShape()[0] != activeBatchSize || replayDT.getShape()[0] != activeBatchSize
        || replayU.getShape()[1] != replayB.getShape()[1] || replayU.getShape()[1] != replayDT.getShape()[1])
    {
        throw std::runtime_error(
            "invokeMambaReplayReconstructBatched: replay tensors must share active-batch and sequence extents.");
    }
    if (acceptedNodeIds != nullptr)
    {
        auto const idsShape = acceptedNodeIds->getShape();
        if (maxAcceptLen <= 0 || idsShape.getNumDims() != 2 || idsShape[0] < activeBatchSize
            || idsShape[1] < maxAcceptLen || acceptedNodeIds->getDataType() != nvinfer1::DataType::kINT32
            || acceptedNodeIds->getDeviceType() != trt_edgellm::rt::DeviceType::kGPU)
        {
            throw std::runtime_error(
                "invokeMambaReplayReconstructBatched: acceptedNodeIds must be GPU INT32 [batch, maxAcceptLen].");
        }
    }

    SelectiveStateUpdateParams params{};
    // State pools are sized to maxBatch; bound the reconstruction to the active
    // sequences so the kernel does not read past acceptedLengths[activeBatch).
    params.batch = static_cast<uint32_t>(activeBatchSize);
    params.state_cache_size = static_cast<uint32_t>(state.getShape()[0]);
    params.state_batch_indices = const_cast<int32_t*>(stateIndices.dataPointer<int32_t>());
    params.nheads = static_cast<uint32_t>(state.getShape()[1]);
    params.dim = static_cast<uint32_t>(state.getShape()[2]);
    params.dstate = static_cast<uint32_t>(state.getShape()[3]);
    params.ngroups = static_cast<uint32_t>(replayB.getShape()[2]); // [batch, seq_len, ngroups, dstate]

    params.state_stride_batch = state.getStride(0);
    params.state_stride_head = state.getStride(1);
    params.state_stride_dim = state.getStride(2);

    params.replay_seq_len = replayU.getShape()[1]; // [batch, seq_len, nheads, dim]
    params.context_lengths = acceptedLengths.dataPointer<int32_t>();

    if (state.getDataType() == nvinfer1::DataType::kHALF)
    {
        SsmReplayReconstructLauncher<half> launcher{params, deviceLayerInfos, numLayers,
            acceptedNodeIds != nullptr ? acceptedNodeIds->dataPointer<int32_t>() : nullptr, maxAcceptLen, stream};
        dispatchDimDstate(params, AllowedDims{}, AllowedDstates{}, launcher);
    }
    else if (state.getDataType() == nvinfer1::DataType::kFLOAT)
    {
        SsmReplayReconstructLauncher<float> launcher{params, deviceLayerInfos, numLayers,
            acceptedNodeIds != nullptr ? acceptedNodeIds->dataPointer<int32_t>() : nullptr, maxAcceptLen, stream};
        dispatchDimDstate(params, AllowedDims{}, AllowedDstates{}, launcher);
    }
    else
    {
        throw std::runtime_error("invokeMambaReplayReconstructBatched: state must be half or float.");
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

void invokeMambaStateGather(trt_edgellm::rt::Tensor const& residentState, trt_edgellm::rt::Tensor& activeState,
    trt_edgellm::rt::Tensor const& stateIndices, cudaStream_t stream)
{
    validateStateCopyTensors(residentState, activeState, stateIndices);
    int32_t const activeRows = static_cast<int32_t>(activeState.getShape()[0]);
    if (activeRows == 0)
    {
        return;
    }
    int64_t const rowBytes = logicalStateRowBytes(residentState);
    int32_t const residentRows = static_cast<int32_t>(residentState.getShape()[0]);
    copyIndexedStateRowsKernel<false><<<activeRows, kStateCopyBlockSize, 0, stream>>>(
        static_cast<uint8_t const*>(residentState.rawPointer()), static_cast<uint8_t*>(activeState.rawPointer()),
        stateIndices.dataPointer<int32_t>(), residentRows, rowBytes);
    CUDA_CHECK(cudaGetLastError());
}

void invokeMambaStateScatter(trt_edgellm::rt::Tensor const& activeState, trt_edgellm::rt::Tensor& residentState,
    trt_edgellm::rt::Tensor const& stateIndices, cudaStream_t stream)
{
    validateStateCopyTensors(residentState, activeState, stateIndices);
    int32_t const activeRows = static_cast<int32_t>(activeState.getShape()[0]);
    if (activeRows == 0)
    {
        return;
    }
    int64_t const rowBytes = logicalStateRowBytes(residentState);
    int32_t const residentRows = static_cast<int32_t>(residentState.getShape()[0]);
    copyIndexedStateRowsKernel<true><<<activeRows, kStateCopyBlockSize, 0, stream>>>(
        static_cast<uint8_t const*>(activeState.rawPointer()), static_cast<uint8_t*>(residentState.rawPointer()),
        stateIndices.dataPointer<int32_t>(), residentRows, rowBytes);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace mamba_ssm
