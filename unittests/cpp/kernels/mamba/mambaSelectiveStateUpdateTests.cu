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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/mamba/selectiveStateUpdate.h"
#include "testUtils.h"

using namespace mamba_ssm;
using namespace trt_edgellm;

namespace
{

// =============================================================================
// CPU Reference Implementation
// =============================================================================

float softplus(float x)
{
    return std::log(1.f + std::exp(x));
}

float thresholdedSoftplus(float x)
{
    constexpr float threshold = 20.f;
    return (x <= threshold) ? softplus(x) : x;
}

float silu(float x)
{
    return x / (1.f + std::exp(-x));
}

//! \brief CPU reference implementation for selective state update
//!
//! Computes:
//!   dA = exp(A * dt)
//!   dB = B * dt
//!   new_state = state * dA + dB * x
//!   output = sum_i(new_state_i * C_i) + D * x
//!   if z: output *= silu(z)
void selectiveStateUpdateReference(int32_t batch, int32_t nheads, int32_t dim, int32_t dstate, int32_t ngroups,
    std::vector<half> const& state,   // [batch, nheads, dim, dstate]
    std::vector<half> const& x,       // [batch, nheads, dim]
    std::vector<float> const& dt,     // [batch, nheads]
    std::vector<float> const& A,      // [nheads]
    std::vector<half> const& B,       // [batch, ngroups, dstate]
    std::vector<half> const& C,       // [batch, ngroups, dstate]
    std::vector<float> const* D,      // [nheads] (optional)
    std::vector<float> const* dtBias, // [nheads] (optional)
    std::vector<half> const* z,       // [batch, nheads, dim] (optional)
    bool dtSoftplus,                  //
    std::vector<half>& outputRef,     // [batch, nheads, dim]
    std::vector<half>& stateRef       // [batch, nheads, dim, dstate]
)
{
    // Copy state for in-place update
    stateRef = state;

    int32_t const headsPerGroup = nheads / ngroups;

    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t h = 0; h < nheads; ++h)
        {
            int32_t const group = h / headsPerGroup;

            // Get dt value and apply bias + softplus if needed
            float dtVal = dt[b * nheads + h];
            if (dtBias)
            {
                dtVal += (*dtBias)[h];
            }
            if (dtSoftplus)
            {
                dtVal = thresholdedSoftplus(dtVal);
            }

            // Get A value and compute dA = exp(A * dt)
            float const aVal = A[h];
            float const dA = std::exp(aVal * dtVal);

            // Get D value if available
            float const dVal = D ? (*D)[h] : 0.f;

            for (int32_t d = 0; d < dim; ++d)
            {
                // Get x value
                float const xVal = __half2float(x[b * nheads * dim + h * dim + d]);

                // Compute output accumulator (start with D*x)
                float outVal = dVal * xVal;

                // Process each state dimension
                for (int32_t s = 0; s < dstate; ++s)
                {
                    // Get current state
                    int64_t const stateIdx = static_cast<int64_t>(b) * nheads * dim * dstate
                        + static_cast<int64_t>(h) * dim * dstate + static_cast<int64_t>(d) * dstate + s;
                    float stateVal = __half2float(stateRef[stateIdx]);

                    // Get B and C values (indexed by group)
                    float const bVal = __half2float(B[b * ngroups * dstate + group * dstate + s]);
                    float const cVal = __half2float(C[b * ngroups * dstate + group * dstate + s]);

                    // Compute dB = B * dt
                    float const dB = bVal * dtVal;

                    // Update state: new_state = state * dA + dB * x
                    float const newState = stateVal * dA + dB * xVal;

                    // Store updated state
                    stateRef[stateIdx] = __float2half(newState);

                    // Accumulate output: output += new_state * C
                    outVal += newState * cVal;
                }

                // Apply SiLU gating if z is provided
                if (z)
                {
                    float const zVal = __half2float((*z)[b * nheads * dim + h * dim + d]);
                    outVal *= silu(zVal);
                }

                // Store output
                outputRef[b * nheads * dim + h * dim + d] = __float2half(outVal);
            }
        }
    }
}

//! \brief CPU reference implementation for multi-step selective state update
//!
//! Computes for each t in [0, seqLen):
//!   dA = exp(A * dt)
//!   dB = B * dt
//!   new_state = state * dA + dB * x
//!   output = sum_i(new_state_i * C_i) + D * x
void selectiveStateUpdateMultiStepReferenceFp32(
    int32_t batch, int32_t nheads, int32_t dim, int32_t dstate, int32_t ngroups, int32_t seqLen,
    std::vector<half> const& stateInit, // [batch, nheads, dim, dstate]
    std::vector<half> const& x,         // [batch, seqLen, nheads, dim]
    std::vector<half> const& dt,        // [batch, seqLen, nheads]   (half for weight_t=half)
    std::vector<float> const& A,        // [nheads]
    std::vector<half> const& B,         // [batch, seqLen, ngroups, dstate]
    std::vector<half> const& C,         // [batch, seqLen, ngroups, dstate]
    std::vector<half> const* D,         // [nheads] (optional, half)
    std::vector<half> const* dtBias,    // [nheads] (optional, half)
    std::vector<half> const* z,         // [batch, seqLen, nheads, dim] (optional)
    bool dtSoftplus,
    std::vector<half>& outputRef,                     // [batch, seqLen, nheads, dim]
    std::vector<half>& stateRef,                      // [batch, nheads, dim, dstate]
    std::vector<int32_t> const* contextLens = nullptr // [batch] (optional)
)
{
    int32_t const headsPerGroup = nheads / ngroups;

    std::vector<float> fp32State(stateInit.size());
    for (size_t i = 0; i < stateInit.size(); ++i)
        fp32State[i] = __half2float(stateInit[i]);

    for (int32_t t = 0; t < seqLen; ++t)
    {
        for (int32_t b = 0; b < batch; ++b)
        {
            int32_t const cl = contextLens ? (*contextLens)[b] : seqLen;
            if (t >= cl)
            {
                for (int32_t h = 0; h < nheads; ++h)
                    for (int32_t d = 0; d < dim; ++d)
                        outputRef[b * seqLen * nheads * dim + t * nheads * dim + h * dim + d] = __float2half(0.f);
                continue;
            }

            for (int32_t h = 0; h < nheads; ++h)
            {
                int32_t const group = h / headsPerGroup;

                float dtVal = __half2float(dt[b * seqLen * nheads + t * nheads + h]);
                if (dtBias)
                    dtVal += __half2float((*dtBias)[h]);
                if (dtSoftplus)
                    dtVal = thresholdedSoftplus(dtVal);

                float const aVal = A[h];
                float const dA = std::exp(aVal * dtVal);
                float const dVal = D ? __half2float((*D)[h]) : 0.f;

                for (int32_t d = 0; d < dim; ++d)
                {
                    float const xVal = __half2float(x[b * seqLen * nheads * dim + t * nheads * dim + h * dim + d]);
                    float outVal = dVal * xVal;

                    for (int32_t s = 0; s < dstate; ++s)
                    {
                        int64_t const si = static_cast<int64_t>(b) * nheads * dim * dstate
                            + static_cast<int64_t>(h) * dim * dstate + static_cast<int64_t>(d) * dstate + s;

                        float const bVal = __half2float(
                            B[b * seqLen * ngroups * dstate + t * ngroups * dstate + group * dstate + s]);
                        float const cVal = __half2float(
                            C[b * seqLen * ngroups * dstate + t * ngroups * dstate + group * dstate + s]);

                        fp32State[si] = fp32State[si] * dA + bVal * dtVal * xVal;
                        outVal += fp32State[si] * cVal;
                    }

                    if (z)
                    {
                        float const zVal
                            = __half2float((*z)[b * seqLen * nheads * dim + t * nheads * dim + h * dim + d]);
                        outVal *= silu(zVal);
                    }

                    outputRef[b * seqLen * nheads * dim + t * nheads * dim + h * dim + d] = __float2half(outVal);
                }
            }
        }
    }

    stateRef.resize(fp32State.size());
    for (size_t i = 0; i < fp32State.size(); ++i)
        stateRef[i] = __float2half(fp32State[i]);
}

// =============================================================================
// Test Helper Functions
// =============================================================================

struct MambaTestConfig
{
    int32_t batch;
    int32_t nheads;
    int32_t dim;            // mamba_head_dim in config
    int32_t dstate;         // ssm_state_size in config
    int32_t ngroups;        // n_groups in config
    bool useSiluGating;     // If true, applies output *= silu(z) gating
    bool useSkipConnection; // If true, adds D * x to output (skip/residual connection)
    bool useDtBias;
    bool dtSoftplus;
    int32_t paddedDim{0};    // 0 means no padding for dim
    int32_t paddedDstate{0}; // 0 means no padding for dstate
};

//! \brief Copy data from contiguous to padded layout for tensors with shape [batch, nheads, dim]
//! Used for x, z, output tensors when dim is padded
void copyTensorToPaddedDim(
    std::vector<half> const& src, std::vector<half>& dst, int32_t batch, int32_t nheads, int32_t dim, int32_t paddedDim)
{
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t h = 0; h < nheads; ++h)
        {
            for (int32_t d = 0; d < dim; ++d)
            {
                int64_t const srcIdx = static_cast<int64_t>(b) * nheads * dim + static_cast<int64_t>(h) * dim + d;
                int64_t const dstIdx
                    = static_cast<int64_t>(b) * nheads * paddedDim + static_cast<int64_t>(h) * paddedDim + d;
                dst[dstIdx] = src[srcIdx];
            }
        }
    }
}

//! \brief Copy data from padded to contiguous layout for tensors with shape [batch, nheads, dim]
void copyTensorFromPaddedDim(
    std::vector<half> const& src, std::vector<half>& dst, int32_t batch, int32_t nheads, int32_t dim, int32_t paddedDim)
{
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t h = 0; h < nheads; ++h)
        {
            for (int32_t d = 0; d < dim; ++d)
            {
                int64_t const srcIdx
                    = static_cast<int64_t>(b) * nheads * paddedDim + static_cast<int64_t>(h) * paddedDim + d;
                int64_t const dstIdx = static_cast<int64_t>(b) * nheads * dim + static_cast<int64_t>(h) * dim + d;
                dst[dstIdx] = src[srcIdx];
            }
        }
    }
}

//! \brief Copy data from contiguous to padded layout for state tensor
//! Contiguous: [batch, nheads, dim, dstate]
//! Padded:     [batch, nheads, paddedDim, paddedDstate]
void copyStateToPadded(std::vector<half> const& src, std::vector<half>& dst, int32_t batch, int32_t nheads, int32_t dim,
    int32_t dstate, int32_t paddedDim, int32_t paddedDstate)
{
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t h = 0; h < nheads; ++h)
        {
            for (int32_t d = 0; d < dim; ++d)
            {
                for (int32_t s = 0; s < dstate; ++s)
                {
                    int64_t const srcIdx = static_cast<int64_t>(b) * nheads * dim * dstate
                        + static_cast<int64_t>(h) * dim * dstate + static_cast<int64_t>(d) * dstate + s;
                    int64_t const dstIdx = static_cast<int64_t>(b) * nheads * paddedDim * paddedDstate
                        + static_cast<int64_t>(h) * paddedDim * paddedDstate + static_cast<int64_t>(d) * paddedDstate
                        + s;
                    dst[dstIdx] = src[srcIdx];
                }
            }
        }
    }
}

//! \brief Copy data from padded to contiguous layout for state tensor
void copyStateFromPadded(std::vector<half> const& src, std::vector<half>& dst, int32_t batch, int32_t nheads,
    int32_t dim, int32_t dstate, int32_t paddedDim, int32_t paddedDstate)
{
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t h = 0; h < nheads; ++h)
        {
            for (int32_t d = 0; d < dim; ++d)
            {
                for (int32_t s = 0; s < dstate; ++s)
                {
                    int64_t const srcIdx = static_cast<int64_t>(b) * nheads * paddedDim * paddedDstate
                        + static_cast<int64_t>(h) * paddedDim * paddedDstate + static_cast<int64_t>(d) * paddedDstate
                        + s;
                    int64_t const dstIdx = static_cast<int64_t>(b) * nheads * dim * dstate
                        + static_cast<int64_t>(h) * dim * dstate + static_cast<int64_t>(d) * dstate + s;
                    dst[dstIdx] = src[srcIdx];
                }
            }
        }
    }
}

void runMambaSelectiveStateUpdateTest(MambaTestConfig const& config)
{
    cudaStream_t stream{nullptr};

    int32_t const batch = config.batch;
    int32_t const nheads = config.nheads;
    int32_t const dim = config.dim;
    int32_t const dstate = config.dstate;
    int32_t const ngroups = config.ngroups;

    // Determine if we're using padded layout
    bool const usePaddedDim = config.paddedDim > dim;
    bool const usePaddedDstate = config.paddedDstate > dstate;
    bool const usePadding = usePaddedDim || usePaddedDstate;
    int32_t const paddedDim = usePaddedDim ? config.paddedDim : dim;
    int32_t const paddedDstate = usePaddedDstate ? config.paddedDstate : dstate;

    // Allocate and initialize host buffers (contiguous layout for reference)
    std::vector<half> stateHostContiguous(batch * nheads * dim * dstate);
    std::vector<half> xHostContiguous(batch * nheads * dim);
    std::vector<float> dtHost(batch * nheads);
    std::vector<float> AHost(nheads);
    std::vector<half> BHost(batch * ngroups * dstate);
    std::vector<half> CHost(batch * ngroups * dstate);
    std::vector<float> DHost(nheads);
    std::vector<float> dtBiasHost(nheads);
    std::vector<half> zHostContiguous(batch * nheads * dim);

    // Initialize with random values
    uniformFloatInitialization<half>(stateHostContiguous, -1.f, 1.f);
    uniformFloatInitialization<half>(xHostContiguous, -1.f, 1.f);
    uniformFloatInitialization<float>(dtHost, 0.1f, 2.f);
    uniformFloatInitialization<float>(AHost, -1.f, -0.1f); // A is typically negative
    uniformFloatInitialization<half>(BHost, -1.f, 1.f);
    uniformFloatInitialization<half>(CHost, -1.f, 1.f);
    uniformFloatInitialization<float>(DHost, -1.f, 1.f);
    uniformFloatInitialization<float>(dtBiasHost, -0.5f, 0.5f);
    uniformFloatInitialization<half>(zHostContiguous, -2.f, 2.f);

    // Prepare tensors for GPU (padded if needed)
    std::vector<half> stateHostForGpu;
    std::vector<half> xHostForGpu;
    std::vector<half> zHostForGpu;
    size_t outputGpuSize;

    // State needs padding if either dim or dstate is padded
    if (usePadding)
    {
        stateHostForGpu.resize(batch * nheads * paddedDim * paddedDstate, __float2half(0.f));
        copyStateToPadded(stateHostContiguous, stateHostForGpu, batch, nheads, dim, dstate, paddedDim, paddedDstate);
    }
    else
    {
        stateHostForGpu = stateHostContiguous;
    }

    // x, z, output only need padding if dim is padded (they don't have dstate dimension)
    if (usePaddedDim)
    {
        xHostForGpu.resize(batch * nheads * paddedDim, __float2half(0.f));
        copyTensorToPaddedDim(xHostContiguous, xHostForGpu, batch, nheads, dim, paddedDim);

        zHostForGpu.resize(batch * nheads * paddedDim, __float2half(0.f));
        copyTensorToPaddedDim(zHostContiguous, zHostForGpu, batch, nheads, dim, paddedDim);

        outputGpuSize = batch * nheads * paddedDim;
    }
    else
    {
        xHostForGpu = xHostContiguous;
        zHostForGpu = zHostContiguous;
        outputGpuSize = batch * nheads * dim;
    }

    // Compute CPU reference using contiguous layout
    std::vector<half> outputRef(batch * nheads * dim);
    std::vector<half> stateRef;
    selectiveStateUpdateReference(batch, nheads, dim, dstate, ngroups, stateHostContiguous, xHostContiguous, dtHost,
        AHost, BHost, CHost, config.useSkipConnection ? &DHost : nullptr, config.useDtBias ? &dtBiasHost : nullptr,
        config.useSiluGating ? &zHostContiguous : nullptr, config.dtSoftplus, outputRef, stateRef);

    auto stateDevice = rt::Tensor({batch, nheads, paddedDim, paddedDstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto xDevice = rt::Tensor({batch, nheads, paddedDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto dtDevice = rt::Tensor({batch, nheads}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto ADevice = rt::Tensor({nheads}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto BDevice = rt::Tensor({batch, ngroups, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto CDevice = rt::Tensor({batch, ngroups, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outputDevice = rt::Tensor({batch, nheads, paddedDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor DDevice;
    rt::Tensor dtBiasDevice;
    rt::Tensor zDevice;

    copyHostToDevice<half>(stateDevice, stateHostForGpu);
    copyHostToDevice<half>(xDevice, xHostForGpu);
    copyHostToDevice<float>(dtDevice, dtHost);
    copyHostToDevice<float>(ADevice, AHost);
    copyHostToDevice<half>(BDevice, BHost);
    copyHostToDevice<half>(CDevice, CHost);
    CUDA_CHECK(cudaMemset(outputDevice.rawPointer(), 0, outputDevice.getMemoryCapacity()));

    if (config.useSkipConnection)
    {
        DDevice = rt::Tensor({nheads}, rt::DeviceType::kGPU, DataType::kFLOAT);
        copyHostToDevice<float>(DDevice, DHost);
    }
    if (config.useDtBias)
    {
        dtBiasDevice = rt::Tensor({nheads}, rt::DeviceType::kGPU, DataType::kFLOAT);
        copyHostToDevice<float>(dtBiasDevice, dtBiasHost);
    }
    if (config.useSiluGating)
    {
        zDevice = rt::Tensor({batch, nheads, paddedDim}, rt::DeviceType::kGPU, DataType::kHALF);
        copyHostToDevice<half>(zDevice, zHostForGpu);
    }

    if (usePaddedDstate)
    {
        auto BPadded = rt::Tensor({batch, ngroups, paddedDstate}, rt::DeviceType::kGPU, DataType::kHALF);
        auto CPadded = rt::Tensor({batch, ngroups, paddedDstate}, rt::DeviceType::kGPU, DataType::kHALF);
        CUDA_CHECK(cudaMemset(BPadded.rawPointer(), 0, BPadded.getMemoryCapacity()));
        CUDA_CHECK(cudaMemset(CPadded.rawPointer(), 0, CPadded.getMemoryCapacity()));
        for (int32_t b = 0; b < batch; ++b)
        {
            for (int32_t g = 0; g < ngroups; ++g)
            {
                auto* dst = static_cast<std::byte*>(BPadded.rawPointer())
                    + (static_cast<size_t>(b) * ngroups * paddedDstate + g * paddedDstate) * sizeof(half);
                auto const* src = static_cast<std::byte const*>(BDevice.rawPointer())
                    + (static_cast<size_t>(b) * ngroups * dstate + g * dstate) * sizeof(half);
                CUDA_CHECK(cudaMemcpy(dst, src, dstate * sizeof(half), cudaMemcpyDeviceToDevice));

                dst = static_cast<std::byte*>(CPadded.rawPointer())
                    + (static_cast<size_t>(b) * ngroups * paddedDstate + g * paddedDstate) * sizeof(half);
                src = static_cast<std::byte const*>(CDevice.rawPointer())
                    + (static_cast<size_t>(b) * ngroups * dstate + g * dstate) * sizeof(half);
                CUDA_CHECK(cudaMemcpy(dst, src, dstate * sizeof(half), cudaMemcpyDeviceToDevice));
            }
        }
        BDevice = std::move(BPadded);
        CDevice = std::move(CPadded);
    }

    namespace rt = trt_edgellm::rt;
    rt::OptionalInputTensor dtBiasOpt = dtBiasDevice.isEmpty() ? std::nullopt : std::optional(std::cref(dtBiasDevice));
    rt::OptionalInputTensor DOpt = DDevice.isEmpty() ? std::nullopt : std::optional(std::cref(DDevice));
    rt::OptionalInputTensor zOpt = zDevice.isEmpty() ? std::nullopt : std::optional(std::cref(zDevice));

    invokeSelectiveStateUpdate(xDevice, ADevice, BDevice, CDevice, dtDevice, dtBiasOpt, DOpt, zOpt, stateDevice,
        outputDevice, config.dtSoftplus, /* stateIndices=*/std::nullopt, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const outputFromGpu = copyDeviceToHost<half>(outputDevice);
    auto const stateResultFromGpu = copyDeviceToHost<half>(stateDevice);

    // Convert results from padded to contiguous for comparison if needed
    std::vector<half> outputHost;
    std::vector<half> stateResultContiguous;

    // Output only needs conversion if dim was padded
    if (usePaddedDim)
    {
        outputHost.resize(outputRef.size());
        copyTensorFromPaddedDim(outputFromGpu, outputHost, batch, nheads, dim, paddedDim);
    }
    else
    {
        outputHost = outputFromGpu;
    }

    // State needs conversion if either dim or dstate was padded
    if (usePadding)
    {
        stateResultContiguous.resize(stateRef.size());
        copyStateFromPadded(
            stateResultFromGpu, stateResultContiguous, batch, nheads, dim, dstate, paddedDim, paddedDstate);
    }
    else
    {
        stateResultContiguous = stateResultFromGpu;
    }

    // Compare output
    auto [rtol, atol] = getTolerance<half>();
    int32_t outputMismatches = 0;
    for (size_t i = 0; i < outputRef.size(); ++i)
    {
        if (!isclose(outputHost[i], outputRef[i], rtol, atol))
        {
            if (outputMismatches < 10)
            {
                std::cout << "Output mismatch at index " << i << ": got " << __half2float(outputHost[i])
                          << ", expected " << __half2float(outputRef[i]) << std::endl;
            }
            outputMismatches++;
        }
    }
    EXPECT_EQ(outputMismatches, 0) << "Output has " << outputMismatches << " / " << outputRef.size() << " mismatches";

    // Compare state
    int32_t stateMismatches = 0;
    for (size_t i = 0; i < stateRef.size(); ++i)
    {
        if (!isclose(stateResultContiguous[i], stateRef[i], rtol, atol))
        {
            if (stateMismatches < 10)
            {
                std::cout << "State mismatch at index " << i << ": got " << __half2float(stateResultContiguous[i])
                          << ", expected " << __half2float(stateRef[i]) << std::endl;
            }
            stateMismatches++;
        }
    }
    EXPECT_EQ(stateMismatches, 0) << "State has " << stateMismatches << " / " << stateRef.size() << " mismatches";

    // Print summary
    std::cout << "MambaSelectiveStateUpdate Accuracy: batch=" << batch << ", nheads=" << nheads << ", dim=" << dim;
    if (usePaddedDim)
    {
        std::cout << " (padded to " << paddedDim << ")";
    }
    std::cout << ", dstate=" << dstate;
    if (usePaddedDstate)
    {
        std::cout << " (padded to " << paddedDstate << ")";
    }
    std::cout << ", ngroups=" << ngroups << ", siluGating=" << config.useSiluGating
              << ", skipConn=" << config.useSkipConnection << ", dtSoftplus=" << config.dtSoftplus << std::endl;
}

} // namespace

// =============================================================================
// Test Cases
// =============================================================================

TEST(MambaSelectiveStateUpdate, Basic_Batch1_Dim64_Dstate64)
{
    MambaTestConfig config{};
    config.batch = 1;
    config.nheads = 8;
    config.dim = 64;
    config.dstate = 64;
    config.ngroups = 8;
    config.useSiluGating = false;
    config.useSkipConnection = true;
    config.useDtBias = true;
    config.dtSoftplus = true;
    runMambaSelectiveStateUpdateTest(config);
}

TEST(MambaSelectiveStateUpdate, Basic_Batch4_Dim128_Dstate128)
{
    MambaTestConfig config{};
    config.batch = 4;
    config.nheads = 8;
    config.dim = 128;
    config.dstate = 128;
    config.ngroups = 8;
    config.useSiluGating = false;
    config.useSkipConnection = true;
    config.useDtBias = true;
    config.dtSoftplus = true;
    runMambaSelectiveStateUpdateTest(config);
}

TEST(MambaSelectiveStateUpdate, WithZ_Batch1_Dim64_Dstate128)
{
    MambaTestConfig config{};
    config.batch = 1;
    config.nheads = 8;
    config.dim = 64;
    config.dstate = 128;
    config.ngroups = 8;
    config.useSiluGating = true;
    config.useSkipConnection = true;
    config.useDtBias = true;
    config.dtSoftplus = true;
    runMambaSelectiveStateUpdateTest(config);
}

TEST(MambaSelectiveStateUpdate, NoOptionals_Batch1_Dim128_Dstate64)
{
    MambaTestConfig config{};
    config.batch = 1;
    config.nheads = 8;
    config.dim = 128;
    config.dstate = 64;
    config.ngroups = 8;
    config.useSiluGating = false;
    config.useSkipConnection = false;
    config.useDtBias = false;
    config.dtSoftplus = false;
    runMambaSelectiveStateUpdateTest(config);
}

TEST(MambaSelectiveStateUpdate, NonPaddedNemotronLike)
{
    MambaTestConfig config{};
    config.batch = 1;
    config.nheads = 96;  // mamba_num_heads
    config.dim = 80;     // mamba_head_dim
    config.dstate = 128; // ssm_state_size
    config.ngroups = 8;  // n_groups
    config.useSiluGating = true;
    config.useSkipConnection = true;
    config.useDtBias = true;
    config.dtSoftplus = true;
    runMambaSelectiveStateUpdateTest(config);
}

TEST(MambaSelectiveStateUpdate, Nemotron9B)
{
    // Nemotron-H 9B config
    MambaTestConfig config{};
    config.batch = 1;
    config.nheads = 128; // mamba_num_heads
    config.dim = 80;     // mamba_head_dim
    config.dstate = 128; // ssm_state_size
    config.ngroups = 8;  // n_groups
    config.useSiluGating = true;
    config.useSkipConnection = true;
    config.useDtBias = true;
    config.dtSoftplus = true;
    runMambaSelectiveStateUpdateTest(config);
}

// =============================================================================
// Padded Layout Test Cases
// =============================================================================

TEST(MambaSelectiveStateUpdate, PaddedDimOnly_80to128)
{
    // Test case 1: Only dim is padded (dim=80 padded to 128)
    // Like Nemotron-9B with TMA alignment on the head dimension
    // x, z, output, and state all have padded dim; dstate is contiguous
    MambaTestConfig config{};
    config.batch = 2;
    config.nheads = 16;
    config.dim = 80;        // Logical head dim (Nemotron-9B)
    config.paddedDim = 128; // Allocated head dim (for TMA alignment)
    config.dstate = 128;    // No padding on dstate
    config.ngroups = 8;
    config.useSiluGating = true;
    config.useSkipConnection = true;
    config.useDtBias = true;
    config.dtSoftplus = true;
    runMambaSelectiveStateUpdateTest(config);
}

TEST(MambaSelectiveStateUpdate, PaddedDstateOnly_80to128)
{
    // Test case 2: Only dstate is padded (dstate=80 padded to 128)
    // x, z, output are contiguous; only state has padded dstate dimension
    MambaTestConfig config{};
    config.batch = 2;
    config.nheads = 16;
    config.dim = 64;           // No padding on dim
    config.dstate = 80;        // Logical dstate
    config.paddedDstate = 128; // Allocated dstate (for alignment)
    config.ngroups = 8;
    config.useSiluGating = true;
    config.useSkipConnection = true;
    config.useDtBias = true;
    config.dtSoftplus = true;
    runMambaSelectiveStateUpdateTest(config);
}

TEST(MambaSelectiveStateUpdate, PaddedBoth_Dim80to128_Dstate80to128)
{
    // Test case 3: Both dim and dstate are padded
    // All tensors have padded layouts
    MambaTestConfig config{};
    config.batch = 2;
    config.nheads = 16;
    config.dim = 80;           // Logical head dim
    config.paddedDim = 128;    // Allocated head dim
    config.dstate = 80;        // Logical dstate
    config.paddedDstate = 128; // Allocated dstate
    config.ngroups = 8;
    config.useSiluGating = true;
    config.useSkipConnection = true;
    config.useDtBias = true;
    config.dtSoftplus = true;
    runMambaSelectiveStateUpdateTest(config);
}

// =============================================================================
// Multi-step (seq_len > 1) Test
// Validates that iterating the single-step kernel over seq_len produces the
// same result as calling the reference one step at a time.
// =============================================================================

void runMambaMultiStepTest(
    MambaTestConfig const& config, int32_t seqLen, std::vector<int32_t> const* contextLens = nullptr)
{
    cudaStream_t stream{nullptr};

    int32_t const batch = config.batch;
    int32_t const nheads = config.nheads;
    int32_t const dim = config.dim;
    int32_t const dstate = config.dstate;
    int32_t const ngroups = config.ngroups;

    // Allocate host buffers with seq_len dimension
    // x: [batch, seqLen, nheads, dim], dt: [batch, seqLen, nheads],
    // B/C: [batch, seqLen, ngroups, dstate]
    int64_t const xSize = static_cast<int64_t>(batch) * seqLen * nheads * dim;
    int64_t const dtSize = static_cast<int64_t>(batch) * seqLen * nheads;
    int64_t const bcSize = static_cast<int64_t>(batch) * seqLen * ngroups * dstate;
    int64_t const stateSize = static_cast<int64_t>(batch) * nheads * dim * dstate;

    std::vector<half> stateHostInit(stateSize);
    std::vector<half> xHost(xSize);
    std::vector<half> dtHost(dtSize);
    std::vector<float> AHost(nheads);
    std::vector<half> BHost(bcSize);
    std::vector<half> CHost(bcSize);
    std::vector<half> DHost(nheads);
    std::vector<half> dtBiasHost(nheads);
    std::vector<half> zHost(xSize);

    uniformFloatInitialization<half>(stateHostInit, -1.f, 1.f);
    uniformFloatInitialization<half>(xHost, -1.f, 1.f);
    uniformFloatInitialization<half>(dtHost, 0.1f, 2.f);
    uniformFloatInitialization<float>(AHost, -1.f, -0.1f);
    uniformFloatInitialization<half>(BHost, -1.f, 1.f);
    uniformFloatInitialization<half>(CHost, -1.f, 1.f);
    uniformFloatInitialization<half>(DHost, -1.f, 1.f);
    uniformFloatInitialization<half>(dtBiasHost, -0.5f, 0.5f);
    uniformFloatInitialization<half>(zHost, -2.f, 2.f);

    // CPU reference: iterate one step at a time, feeding state from previous step
    std::vector<half> refOutput(xSize);
    std::vector<half> refState;
    selectiveStateUpdateMultiStepReferenceFp32(batch, nheads, dim, dstate, ngroups, seqLen, stateHostInit, xHost,
        dtHost, AHost, BHost, CHost, config.useSkipConnection ? &DHost : nullptr,
        config.useDtBias ? &dtBiasHost : nullptr, config.useSiluGating ? &zHost : nullptr, config.dtSoftplus, refOutput,
        refState, contextLens);

    auto stateDevice = rt::Tensor({batch, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto xDevice = rt::Tensor({batch, seqLen, nheads, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto dtDevice = rt::Tensor({batch, seqLen, nheads}, rt::DeviceType::kGPU, DataType::kHALF);
    auto ADevice = rt::Tensor({nheads}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto BDevice = rt::Tensor({batch, seqLen, ngroups, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto CDevice = rt::Tensor({batch, seqLen, ngroups, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outputDevice = rt::Tensor({batch, seqLen, nheads, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor DDevice;
    rt::Tensor dtBiasDevice;
    rt::Tensor zDevice;

    copyHostToDevice<half>(stateDevice, stateHostInit);
    copyHostToDevice<half>(xDevice, xHost);
    copyHostToDevice<half>(dtDevice, dtHost);
    copyHostToDevice<float>(ADevice, AHost);
    copyHostToDevice<half>(BDevice, BHost);
    copyHostToDevice<half>(CDevice, CHost);
    CUDA_CHECK(cudaMemset(outputDevice.rawPointer(), 0, outputDevice.getMemoryCapacity()));

    if (config.useSkipConnection)
    {
        DDevice = rt::Tensor({nheads}, rt::DeviceType::kGPU, DataType::kHALF);
        copyHostToDevice<half>(DDevice, DHost);
    }
    if (config.useDtBias)
    {
        dtBiasDevice = rt::Tensor({nheads}, rt::DeviceType::kGPU, DataType::kHALF);
        copyHostToDevice<half>(dtBiasDevice, dtBiasHost);
    }
    if (config.useSiluGating)
    {
        zDevice = rt::Tensor({batch, seqLen, nheads, dim}, rt::DeviceType::kGPU, DataType::kHALF);
        copyHostToDevice<half>(zDevice, zHost);
    }

    namespace rt = trt_edgellm::rt;
    rt::OptionalInputTensor dtBiasOpt = dtBiasDevice.isEmpty() ? std::nullopt : std::optional(std::cref(dtBiasDevice));
    rt::OptionalInputTensor DOpt = DDevice.isEmpty() ? std::nullopt : std::optional(std::cref(DDevice));
    rt::OptionalInputTensor zOpt = zDevice.isEmpty() ? std::nullopt : std::optional(std::cref(zDevice));

    rt::Tensor clDevice;
    if (contextLens)
    {
        clDevice = rt::Tensor({batch}, rt::DeviceType::kGPU, DataType::kINT32);
        CUDA_CHECK(cudaMemcpy(
            clDevice.rawPointer(), contextLens->data(), contextLens->size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    rt::OptionalInputTensor clOpt = clDevice.isEmpty() ? std::nullopt : std::optional(std::cref(clDevice));

    invokeSelectiveStateUpdatePrefill(xDevice, ADevice, BDevice, CDevice, dtDevice, dtBiasOpt, DOpt, zOpt, stateDevice,
        outputDevice, config.dtSoftplus, clOpt, /* stateIndices=*/std::nullopt, /*replayDA=*/std::nullopt,
        /*replayU=*/std::nullopt, /*replayB=*/std::nullopt, /*replayDT=*/std::nullopt, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const gpuOutput = copyDeviceToHost<half>(outputDevice);
    auto const gpuState = copyDeviceToHost<half>(stateDevice);

    auto [rtol, atol] = getTolerance<half>();
    int32_t outputMismatches = 0;
    for (size_t i = 0; i < refOutput.size(); ++i)
    {
        if (!isclose(gpuOutput[i], refOutput[i], rtol, atol))
        {
            if (outputMismatches < 10)
            {
                std::cout << "MultiStep output mismatch at index " << i << ": got " << __half2float(gpuOutput[i])
                          << ", expected " << __half2float(refOutput[i]) << std::endl;
            }
            outputMismatches++;
        }
    }
    EXPECT_EQ(outputMismatches, 0) << "MultiStep output has " << outputMismatches << " / " << refOutput.size()
                                   << " mismatches";

    int32_t stateMismatches = 0;
    for (size_t i = 0; i < refState.size(); ++i)
    {
        if (!isclose(gpuState[i], refState[i], rtol, atol))
        {
            if (stateMismatches < 10)
            {
                std::cout << "MultiStep state mismatch at index " << i << ": got " << __half2float(gpuState[i])
                          << ", expected " << __half2float(refState[i]) << std::endl;
            }
            stateMismatches++;
        }
    }
    EXPECT_EQ(stateMismatches, 0) << "MultiStep state has " << stateMismatches << " / " << refState.size()
                                  << " mismatches";

    std::cout << "MambaMultiStep: batch=" << batch << ", seqLen=" << seqLen << ", nheads=" << nheads << ", dim=" << dim
              << ", dstate=" << dstate << ", ngroups=" << ngroups << std::endl;
}

TEST(MambaSelectiveStateUpdate, MultiStep_SeqLen4_Batch1)
{
    MambaTestConfig config{};
    config.batch = 1;
    config.nheads = 8;
    config.dim = 64;
    config.dstate = 64;
    config.ngroups = 8;
    config.useSiluGating = false;
    config.useSkipConnection = true;
    config.useDtBias = true;
    config.dtSoftplus = true;
    runMambaMultiStepTest(config, 4);
}

TEST(MambaSelectiveStateUpdate, MultiStep_SeqLen16_Batch2)
{
    MambaTestConfig config{};
    config.batch = 2;
    config.nheads = 8;
    config.dim = 64;
    config.dstate = 128;
    config.ngroups = 8;
    config.useSiluGating = false;
    config.useSkipConnection = true;
    config.useDtBias = true;
    config.dtSoftplus = true;
    runMambaMultiStepTest(config, 16);
}

TEST(MambaSelectiveStateUpdate, MultiStep_NemotronLike_SeqLen8)
{
    MambaTestConfig config{};
    config.batch = 1;
    config.nheads = 128;
    config.dim = 80;
    config.dstate = 128;
    config.ngroups = 8;
    config.useSiluGating = false;
    config.useSkipConnection = true;
    config.useDtBias = true;
    config.dtSoftplus = true;
    runMambaMultiStepTest(config, 8);
}

TEST(MambaSelectiveStateUpdate, Padding_MixedContextLengths)
{
    MambaTestConfig config{};
    config.batch = 2;
    config.nheads = 8;
    config.dim = 64;
    config.dstate = 64;
    config.ngroups = 1;
    config.useSiluGating = false;
    config.useSkipConnection = true;
    config.useDtBias = true;
    config.dtSoftplus = true;
    std::vector<int32_t> cl = {5, 16};
    runMambaMultiStepTest(config, 16, &cl);
}

TEST(MambaSelectiveStateUpdate, Padding_AllShorter)
{
    MambaTestConfig config{};
    config.batch = 2;
    config.nheads = 8;
    config.dim = 64;
    config.dstate = 64;
    config.ngroups = 1;
    config.useSiluGating = false;
    config.useSkipConnection = true;
    config.useDtBias = true;
    config.dtSoftplus = true;
    std::vector<int32_t> cl = {3, 7};
    runMambaMultiStepTest(config, 16, &cl);
}

TEST(MambaSelectiveStateUpdate, DDTreeSiblingPaddingAndNoncontiguousReplay)
{
    int32_t constexpr batch = 2;
    int32_t constexpr seqLen = 5;
    int32_t constexpr nheads = 1;
    int32_t constexpr dim = 64;
    int32_t constexpr dstate = 64;
    int32_t constexpr ngroups = 1;

    std::vector<half> stateHost(batch * nheads * dim * dstate);
    std::vector<half> xHost(batch * seqLen * nheads * dim);
    std::vector<half> dtHost(batch * seqLen * nheads);
    std::vector<float> aHost(nheads, -0.2F);
    std::vector<half> bHost(batch * seqLen * ngroups * dstate);
    std::vector<half> cHost(batch * seqLen * ngroups * dstate);
    std::vector<half> dHost(nheads, __float2half(0.1F));
    std::vector<half> dtBiasHost(nheads, __float2half(0.02F));
    for (size_t i = 0; i < stateHost.size(); ++i)
    {
        stateHost[i] = __float2half((static_cast<int32_t>(i % 7) - 3) * 0.01F);
    }
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t t = 0; t < seqLen; ++t)
        {
            dtHost[(b * seqLen + t) * nheads] = __float2half(0.05F * (t + 1));
            for (int32_t d = 0; d < dim; ++d)
            {
                xHost[((b * seqLen + t) * nheads) * dim + d]
                    = __float2half(0.01F * (b + 1) + 0.02F * (t + 1) + 0.001F * (d % 5));
            }
            for (int32_t i = 0; i < dstate; ++i)
            {
                bHost[((b * seqLen + t) * ngroups) * dstate + i] = __float2half(0.01F * (t + 1) + 0.001F * (i % 3));
                cHost[((b * seqLen + t) * ngroups) * dstate + i] = __float2half(0.02F + 0.001F * (i % 5));
            }
        }
    }

    std::vector<int32_t> const parents{-1, 0, 0, 1, 2, -1, 0, 0, -1, -1};
    std::vector<int32_t> const depths{0, 1, 1, 2, 2, 0, 1, 1, 0, 0};
    std::vector<int32_t> const stateIndicesHost{0, 1};
    std::vector<std::vector<int32_t>> const paths{{0}, {0, 1}, {0, 2}, {0, 1, 3}, {0, 2, 4}};

    auto applyToken = [&](int32_t b, int32_t token, std::vector<float>& state, std::vector<half>* output) {
        float dtValue
            = thresholdedSoftplus(__half2float(dtHost[(b * seqLen + token) * nheads]) + __half2float(dtBiasHost[0]));
        float const decay = std::exp(aHost[0] * dtValue);
        for (int32_t row = 0; row < dim; ++row)
        {
            float const xValue = __half2float(xHost[((b * seqLen + token) * nheads) * dim + row]);
            float outValue = __half2float(dHost[0]) * xValue;
            for (int32_t i = 0; i < dstate; ++i)
            {
                int64_t const stateIdx = static_cast<int64_t>(row) * dstate + i;
                float const bValue = __half2float(bHost[((b * seqLen + token) * ngroups) * dstate + i]);
                float const dB = bValue * dtValue;
                float const newState = state[stateIdx] * decay + dB * xValue;
                outValue += newState * __half2float(cHost[((b * seqLen + token) * ngroups) * dstate + i]);
                state[stateIdx] = __half2float(__float2half(newState));
            }
            if (output != nullptr)
            {
                (*output)[((b * seqLen + token) * nheads) * dim + row] = __float2half(outValue);
            }
        }
    };

    std::vector<half> outputRef(batch * seqLen * nheads * dim, __float2half(0.F));
    for (int32_t b = 0; b < batch; ++b)
    {
        int32_t const validNodes = b == 0 ? seqLen : 3;
        for (int32_t node = 0; node < validNodes; ++node)
        {
            std::vector<float> pathState(dim * dstate);
            int64_t const stateBase = static_cast<int64_t>(b) * dim * dstate;
            for (int32_t i = 0; i < dim * dstate; ++i)
            {
                pathState[i] = __half2float(stateHost[stateBase + i]);
            }
            for (int32_t token : paths[node])
            {
                applyToken(b, token, pathState, token == node ? &outputRef : nullptr);
            }
        }
    }

    auto stateDevice = rt::Tensor({batch, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto xDevice = rt::Tensor({batch, seqLen, nheads, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto dtDevice = rt::Tensor({batch, seqLen, nheads}, rt::DeviceType::kGPU, DataType::kHALF);
    auto aDevice = rt::Tensor({nheads}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto bDevice = rt::Tensor({batch, seqLen, ngroups, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto cDevice = rt::Tensor({batch, seqLen, ngroups, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto dDevice = rt::Tensor({nheads}, rt::DeviceType::kGPU, DataType::kHALF);
    auto dtBiasDevice = rt::Tensor({nheads}, rt::DeviceType::kGPU, DataType::kHALF);
    auto parentDevice = rt::Tensor({batch, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto depthDevice = rt::Tensor({batch, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto stateIndicesDevice = rt::Tensor({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    auto outputDevice = rt::Tensor({batch, seqLen, nheads, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto replayDa = rt::Tensor({batch, seqLen, nheads}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto replayU = rt::Tensor({batch, seqLen, nheads, dim}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto replayB = rt::Tensor({batch, seqLen, ngroups, dstate}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto replayDt = rt::Tensor({batch, seqLen, nheads}, rt::DeviceType::kGPU, DataType::kFLOAT);
    copyHostToDevice<half>(stateDevice, stateHost);
    copyHostToDevice<half>(xDevice, xHost);
    copyHostToDevice<half>(dtDevice, dtHost);
    copyHostToDevice<float>(aDevice, aHost);
    copyHostToDevice<half>(bDevice, bHost);
    copyHostToDevice<half>(cDevice, cHost);
    copyHostToDevice<half>(dDevice, dHost);
    copyHostToDevice<half>(dtBiasDevice, dtBiasHost);
    copyHostToDevice<int32_t>(parentDevice, parents);
    copyHostToDevice<int32_t>(depthDevice, depths);
    copyHostToDevice<int32_t>(stateIndicesDevice, stateIndicesHost);

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    rt::OptionalInputTensor const dtBiasOpt{std::cref(dtBiasDevice)};
    rt::OptionalInputTensor const dOpt{std::cref(dDevice)};
    invokeSelectiveStateUpdateDDTree(xDevice, aDevice, bDevice, cDevice, dtDevice, dtBiasOpt, dOpt, stateDevice,
        outputDevice, parentDevice, depthDevice, true, std::optional(std::cref(stateIndicesDevice)), replayDa, replayU,
        replayB, replayDt, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const outputHost = copyDeviceToHost<half>(outputDevice);
    auto const stateAfterVerify = copyDeviceToHost<half>(stateDevice);
    EXPECT_EQ(stateAfterVerify, stateHost);
    auto [rtol, atol] = getTolerance<half>();
    for (size_t i = 0; i < outputRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(outputHost[i], outputRef[i], rtol, atol)) << "output mismatch at " << i;
    }
    EXPECT_NE(outputHost[1 * dim], outputHost[2 * dim]);

    std::vector<int32_t> const acceptedIds{0, 2, 4, 0, 1, -1};
    std::vector<int32_t> const acceptedLengths{3, 2};
    auto acceptedIdsDevice = rt::Tensor({batch, 3}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptedLengthsDevice = rt::Tensor({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<int32_t>(acceptedIdsDevice, acceptedIds);
    copyHostToDevice<int32_t>(acceptedLengthsDevice, acceptedLengths);
    MambaReplayLayerInfo const hostInfo{stateDevice.rawPointer(), replayDa.rawPointer(), replayU.rawPointer(),
        replayB.rawPointer(), replayDt.rawPointer()};
    auto infoDevice
        = rt::Tensor({static_cast<int64_t>(sizeof(MambaReplayLayerInfo))}, rt::DeviceType::kGPU, DataType::kUINT8);
    CUDA_CHECK(cudaMemcpyAsync(infoDevice.rawPointer(), &hostInfo, sizeof(hostInfo), cudaMemcpyHostToDevice, stream));
    invokeMambaReplayReconstructBatched(static_cast<MambaReplayLayerInfo const*>(infoDevice.rawPointer()), 1,
        stateDevice, replayU, replayB, replayDt, acceptedLengthsDevice, stateIndicesDevice, batch, stream,
        &acceptedIdsDevice, 3);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<half> replayStateRef(stateHost.size());
    for (int32_t b = 0; b < batch; ++b)
    {
        std::vector<float> state(dim * dstate);
        int64_t const stateBase = static_cast<int64_t>(b) * dim * dstate;
        for (int32_t i = 0; i < dim * dstate; ++i)
        {
            state[i] = __half2float(stateHost[stateBase + i]);
        }
        for (int32_t i = 0; i < acceptedLengths[b]; ++i)
        {
            applyToken(b, acceptedIds[b * 3 + i], state, nullptr);
        }
        for (int32_t i = 0; i < dim * dstate; ++i)
        {
            replayStateRef[stateBase + i] = __float2half(state[i]);
        }
    }
    auto const replayStateHost = copyDeviceToHost<half>(stateDevice);
    for (size_t i = 0; i < replayStateRef.size(); ++i)
    {
        EXPECT_EQ(replayStateHost[i], replayStateRef[i]) << "replay mismatch at " << i;
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(MambaSelectiveStateUpdate, NonIdentityStateIndicesPreserveResidentPool)
{
    constexpr int32_t batch = 2;
    constexpr int32_t stateRows = 4;
    constexpr int32_t nheads = 1;
    constexpr int32_t dim = 64;
    constexpr int32_t dstate = 64;
    constexpr int32_t ngroups = 1;

    std::vector<half> stateHost(static_cast<size_t>(stateRows) * nheads * dim * dstate);
    for (int32_t row = 0; row < stateRows; ++row)
    {
        auto const value = __float2half(static_cast<float>(row + 1));
        auto const begin = stateHost.begin() + static_cast<size_t>(row) * nheads * dim * dstate;
        std::fill(begin, begin + nheads * dim * dstate, value);
    }
    std::vector<half> xHost(static_cast<size_t>(batch) * nheads * dim, __float2half(0.0F));
    std::vector<half> dtHost(static_cast<size_t>(batch) * nheads, __float2half(0.0F));
    std::vector<float> aHost(nheads, 0.0F);
    std::vector<half> bHost(static_cast<size_t>(batch) * ngroups * dstate, __float2half(0.0F));
    std::vector<half> cHost(static_cast<size_t>(batch) * ngroups * dstate, __float2half(1.0F));
    std::vector<int32_t> const stateIndicesHost{3, 1};

    auto state = rt::Tensor({stateRows, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto x = rt::Tensor({batch, nheads, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto dt = rt::Tensor({batch, nheads}, rt::DeviceType::kGPU, DataType::kHALF);
    auto a = rt::Tensor({nheads}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto b = rt::Tensor({batch, ngroups, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto c = rt::Tensor({batch, ngroups, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto output = rt::Tensor({batch, nheads, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto stateIndices = rt::Tensor({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(state, stateHost);
    copyHostToDevice(x, xHost);
    copyHostToDevice(dt, dtHost);
    copyHostToDevice(a, aHost);
    copyHostToDevice(b, bHost);
    copyHostToDevice(c, cHost);
    copyHostToDevice(stateIndices, stateIndicesHost);

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    rt::OptionalInputTensor const indices = std::optional(std::cref(stateIndices));
    invokeSelectiveStateUpdate(x, a, b, c, dt, std::nullopt, std::nullopt, std::nullopt, state, output,
        /* dtSoftplus=*/false, indices, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    auto const outputHost = copyDeviceToHost<half>(output);
    auto const stateAfter = copyDeviceToHost<half>(state);
    for (int32_t row = 0; row < batch; ++row)
    {
        float const expected = static_cast<float>(stateIndicesHost[row] + 1) * dstate;
        for (int32_t d = 0; d < dim; ++d)
        {
            EXPECT_NEAR(__half2float(outputHost[static_cast<size_t>(row) * dim + d]), expected, 0.5F);
        }
    }
    ASSERT_EQ(stateAfter.size(), stateHost.size());
    for (size_t i = 0; i < stateHost.size(); ++i)
    {
        EXPECT_EQ(__half2float(stateAfter[i]), __half2float(stateHost[i]));
    }
}

TEST(MambaSelectiveStateUpdate, GatherScatterNonContiguousResidentRows)
{
    constexpr int32_t activeRows = 2;
    constexpr int32_t residentRows = 5;
    constexpr int32_t nheads = 1;
    constexpr int32_t dim = 2;
    constexpr int32_t dstate = 4;
    constexpr int32_t rowElements = nheads * dim * dstate;

    std::vector<half> residentHost(static_cast<size_t>(residentRows) * rowElements);
    for (int32_t row = 0; row < residentRows; ++row)
    {
        std::fill_n(residentHost.begin() + static_cast<size_t>(row) * rowElements, rowElements,
            __float2half(static_cast<float>(row + 1)));
    }
    std::vector<int32_t> const indicesHost{3, 1};
    auto resident = rt::Tensor({residentRows, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto active = rt::Tensor({activeRows, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto indices = rt::Tensor({activeRows}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(resident, residentHost);
    copyHostToDevice(indices, indicesHost);

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    invokeMambaStateGather(resident, active, indices, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto activeHost = copyDeviceToHost<half>(active);
    for (int32_t activeRow = 0; activeRow < activeRows; ++activeRow)
    {
        for (int32_t element = 0; element < rowElements; ++element)
        {
            EXPECT_EQ(__half2float(activeHost[activeRow * rowElements + element]),
                static_cast<float>(indicesHost[activeRow] + 1));
        }
    }

    std::fill_n(activeHost.begin(), rowElements, __float2half(11.0F));
    std::fill_n(activeHost.begin() + rowElements, rowElements, __float2half(12.0F));
    copyHostToDevice(active, activeHost);
    invokeMambaStateScatter(active, resident, indices, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    auto const residentAfter = copyDeviceToHost<half>(resident);
    for (int32_t row = 0; row < residentRows; ++row)
    {
        float const expected = row == 3 ? 11.0F : (row == 1 ? 12.0F : static_cast<float>(row + 1));
        for (int32_t element = 0; element < rowElements; ++element)
        {
            EXPECT_EQ(__half2float(residentAfter[row * rowElements + element]), expected);
        }
    }
}

TEST(MambaSelectiveStateUpdate, GatherScatterUsesLogicalRowShapeAfterReshape)
{
    constexpr int32_t logicalRows = 2;
    constexpr int32_t allocatedRows = 4;
    constexpr int32_t nheads = 1;
    constexpr int32_t dim = 2;
    constexpr int32_t dstate = 4;
    constexpr int32_t rowElements = nheads * dim * dstate;

    std::vector<half> residentHost(static_cast<size_t>(allocatedRows) * rowElements);
    std::vector<half> activeHost(static_cast<size_t>(allocatedRows) * rowElements, __float2half(-1.0F));
    for (int32_t row = 0; row < allocatedRows; ++row)
    {
        std::fill_n(residentHost.begin() + static_cast<size_t>(row) * rowElements, rowElements,
            __float2half(static_cast<float>(row + 1)));
    }
    std::vector<int32_t> const indicesHost{1, 0};

    auto resident = rt::Tensor({allocatedRows, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto active = rt::Tensor({allocatedRows, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(resident, residentHost);
    copyHostToDevice(active, activeHost);
    ASSERT_TRUE(resident.reshape({logicalRows, nheads, dim, dstate}));
    ASSERT_TRUE(active.reshape({logicalRows, nheads, dim, dstate}));
    auto indices = rt::Tensor({logicalRows}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(indices, indicesHost);

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    invokeMambaStateGather(resident, active, indices, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    activeHost = copyDeviceToHost<half>(active);
    for (int32_t row = 0; row < logicalRows; ++row)
    {
        for (int32_t element = 0; element < rowElements; ++element)
        {
            EXPECT_EQ(__half2float(activeHost[row * rowElements + element]), static_cast<float>(indicesHost[row] + 1));
        }
    }
    ASSERT_TRUE(active.reshape({allocatedRows, nheads, dim, dstate}));
    activeHost = copyDeviceToHost<half>(active);
    for (int32_t row = logicalRows; row < allocatedRows; ++row)
    {
        for (int32_t element = 0; element < rowElements; ++element)
        {
            EXPECT_EQ(__half2float(activeHost[row * rowElements + element]), -1.0F);
        }
    }
    ASSERT_TRUE(active.reshape({logicalRows, nheads, dim, dstate}));
    activeHost.resize(static_cast<size_t>(logicalRows) * rowElements);

    std::fill_n(activeHost.begin(), rowElements, __float2half(11.0F));
    std::fill_n(activeHost.begin() + rowElements, rowElements, __float2half(12.0F));
    copyHostToDevice(active, activeHost);
    invokeMambaStateScatter(active, resident, indices, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    ASSERT_TRUE(resident.reshape({allocatedRows, nheads, dim, dstate}));
    residentHost = copyDeviceToHost<half>(resident);
    for (int32_t row = 0; row < allocatedRows; ++row)
    {
        float const expected = row == 0 ? 12.0F : (row == 1 ? 11.0F : static_cast<float>(row + 1));
        for (int32_t element = 0; element < rowElements; ++element)
        {
            EXPECT_EQ(__half2float(residentHost[row * rowElements + element]), expected);
        }
    }
}

TEST(MambaSelectiveStateUpdate, InvalidResidentRowsAreIsolated)
{
    constexpr int32_t activeRows = 3;
    constexpr int32_t residentRows = 2;
    constexpr int32_t nheads = 1;
    constexpr int32_t dim = 2;
    constexpr int32_t dstate = 4;
    constexpr int32_t rowElements = nheads * dim * dstate;

    constexpr int32_t backingRows = 4;
    std::vector<half> residentHost(static_cast<size_t>(backingRows) * rowElements);
    std::fill_n(residentHost.begin(), rowElements, __float2half(3.0F));
    std::fill_n(residentHost.begin() + rowElements, rowElements, __float2half(7.0F));
    std::fill_n(residentHost.begin() + 2 * rowElements, rowElements, __float2half(23.0F));
    std::fill_n(residentHost.begin() + 3 * rowElements, rowElements, __float2half(29.0F));
    std::vector<half> activeHost(static_cast<size_t>(activeRows) * rowElements, __float2half(19.0F));
    std::vector<int32_t> const indicesHost{1, -1, residentRows};

    auto residentBacking = rt::Tensor({backingRows, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto resident = rt::Tensor(
        residentBacking.rawPointer(), {residentRows, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto active = rt::Tensor({activeRows, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto indices = rt::Tensor({activeRows}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(residentBacking, residentHost);
    copyHostToDevice(active, activeHost);
    copyHostToDevice(indices, indicesHost);

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    invokeMambaStateGather(resident, active, indices, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    activeHost = copyDeviceToHost<half>(active);
    for (int32_t element = 0; element < rowElements; ++element)
    {
        EXPECT_EQ(__half2float(activeHost[element]), 7.0F);
        EXPECT_EQ(__half2float(activeHost[rowElements + element]), 0.0F);
        EXPECT_EQ(__half2float(activeHost[2 * rowElements + element]), 0.0F);
    }

    std::fill_n(activeHost.begin(), rowElements, __float2half(11.0F));
    std::fill_n(activeHost.begin() + rowElements, rowElements, __float2half(13.0F));
    std::fill_n(activeHost.begin() + 2 * rowElements, rowElements, __float2half(17.0F));
    copyHostToDevice(active, activeHost);
    invokeMambaStateScatter(active, resident, indices, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    auto const residentAfter = copyDeviceToHost<half>(residentBacking);
    for (int32_t element = 0; element < rowElements; ++element)
    {
        EXPECT_EQ(__half2float(residentAfter[element]), 3.0F);
        EXPECT_EQ(__half2float(residentAfter[rowElements + element]), 11.0F);
        EXPECT_EQ(__half2float(residentAfter[2 * rowElements + element]), 23.0F);
        EXPECT_EQ(__half2float(residentAfter[3 * rowElements + element]), 29.0F);
    }
}

TEST(MambaSelectiveStateUpdate, ReplayPreservesDecodeFmaRounding)
{
    // Given captured FP32 replay operands near FP16 cancellation,
    // when committing a root through prefix or indexed replay,
    // then the state matches the decode path's fused input contribution.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<rt::Tensor> hostBuffers;
    auto upload = [&](rt::Coords const& shape, DataType dtype, auto value) {
        rt::Tensor device(shape, rt::DeviceType::kGPU, dtype);
        hostBuffers.emplace_back(shape, rt::DeviceType::kCPU, dtype);
        auto& host = hostBuffers.back();
        std::fill_n(static_cast<decltype(value)*>(host.rawPointer()), host.getShape().volume(), value);
        CUDA_CHECK(cudaMemcpyAsync(device.rawPointer(), host.rawPointer(), host.getShape().volume() * sizeof(value),
            cudaMemcpyHostToDevice, stream));
        return device;
    };
    auto da = upload({1, 1, 1}, DataType::kFLOAT, 0.5937744975090027F);
    auto x = upload({1, 1, 1, 64}, DataType::kFLOAT, 0.02862548828125F);
    auto b = upload({1, 1, 1, 128}, DataType::kFLOAT, 0.374267578125F);
    auto dt = upload({1, 1, 1}, DataType::kFLOAT, 13.443361282348633F);
    auto accepted = upload({1}, DataType::kINT32, int32_t{1});
    auto stateIndices = upload({1}, DataType::kINT32, int32_t{0});
    auto acceptedIndices = upload({1, 1}, DataType::kINT32, int32_t{0});
    for (bool indexed : {false, true})
    {
        auto state = upload({1, 1, 64, 128}, DataType::kHALF, __float2half(-0.2427978515625F));
        rt::Tensor hostInfo(
            {static_cast<int64_t>(sizeof(MambaReplayLayerInfo))}, rt::DeviceType::kCPU, DataType::kUINT8);
        *static_cast<MambaReplayLayerInfo*>(hostInfo.rawPointer())
            = {state.rawPointer(), da.rawPointer(), x.rawPointer(), b.rawPointer(), dt.rawPointer()};
        rt::Tensor info(hostInfo.getShape(), rt::DeviceType::kGPU, DataType::kUINT8);
        CUDA_CHECK(cudaMemcpyAsync(
            info.rawPointer(), hostInfo.rawPointer(), sizeof(MambaReplayLayerInfo), cudaMemcpyHostToDevice, stream));
        invokeMambaReplayReconstructBatched(static_cast<MambaReplayLayerInfo const*>(info.rawPointer()), 1, state, x, b,
            dt, accepted, stateIndices, 1, stream, indexed ? &acceptedIndices : nullptr, 1);
        rt::Tensor result(state.getShape(), rt::DeviceType::kCPU, DataType::kHALF);
        CUDA_CHECK(cudaMemcpyAsync(result.rawPointer(), state.rawPointer(), state.getShape().volume() * sizeof(half),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto const* values = static_cast<half const*>(result.rawPointer());
        std::vector<half> const expected(state.getShape().volume(), __float2half(-0.0001404285430908203F));
        EXPECT_EQ(std::vector<half>(values, values + state.getShape().volume()), expected) << "indexed=" << indexed;
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(MambaSelectiveStateUpdate, VerifyPreservesDecodeFmaRounding)
{
    // Given a cancelling state/input update with unit decay and a single output weight,
    // when evaluating the same root in decode, chain verify, or tree verify,
    // then all routes preserve the independently computed FP16 result.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<rt::Tensor> hostBuffers;
    auto upload = [&](rt::Coords const& shape, DataType dtype, auto value) {
        rt::Tensor device(shape, rt::DeviceType::kGPU, dtype);
        hostBuffers.emplace_back(shape, rt::DeviceType::kCPU, dtype);
        auto& host = hostBuffers.back();
        std::fill_n(static_cast<decltype(value)*>(host.rawPointer()), host.getShape().volume(), value);
        CUDA_CHECK(cudaMemcpyAsync(device.rawPointer(), host.rawPointer(), host.getShape().volume() * sizeof(value),
            cudaMemcpyHostToDevice, stream));
        return device;
    };
    auto a = upload({1}, DataType::kFLOAT, 0.0F);
    auto parents = upload({1, 1}, DataType::kINT32, int32_t{-1});
    auto depths = upload({1, 1}, DataType::kINT32, int32_t{0});
    auto da = upload({1, 1, 1}, DataType::kFLOAT, 0.0F);
    auto u = upload({1, 1, 1, 64}, DataType::kFLOAT, 0.0F);
    auto replayB = upload({1, 1, 1, 128}, DataType::kFLOAT, 0.0F);
    auto replayDt = upload({1, 1, 1}, DataType::kFLOAT, 0.0F);
    for (int32_t caseIndex = 0; caseIndex < 6; ++caseIndex)
    {
        int32_t const route = caseIndex % 3;
        bool const halfWeights = caseIndex >= 3;
        auto state = upload(
            {1, 1, 64, 128}, DataType::kHALF, __float2half(halfWeights ? -0.10479736328125F : -0.14404296875F));
        auto x = upload({1, 1, 1, 64}, DataType::kHALF, __float2half(0.02862548828125F));
        auto b = upload({1, 1, 1, 128}, DataType::kHALF, __float2half(0.374267578125F));
        auto c = upload({1, 1, 1, 128}, DataType::kHALF, __float2half(0.0F));
        auto dt = halfWeights ? upload({1, 1, 1}, DataType::kHALF, __float2half(9.78125F))
                              : upload({1, 1, 1}, DataType::kFLOAT, 13.442407608032227F);
        rt::Tensor cHost(c.getShape(), rt::DeviceType::kCPU, DataType::kHALF);
        auto* cValues = static_cast<half*>(cHost.rawPointer());
        std::fill_n(cValues, 128, __float2half(0.0F));
        cValues[0] = __float2half(1.0F);
        CUDA_CHECK(
            cudaMemcpyAsync(c.rawPointer(), cHost.rawPointer(), 128 * sizeof(half), cudaMemcpyHostToDevice, stream));
        rt::Tensor output({1, 1, 1, 64}, rt::DeviceType::kGPU, DataType::kHALF);
        if (route == 0)
        {
            ASSERT_TRUE(x.reshape({1, 1, 64}));
            ASSERT_TRUE(b.reshape({1, 1, 128}));
            ASSERT_TRUE(c.reshape({1, 1, 128}));
            ASSERT_TRUE(dt.reshape({1, 1}));
            ASSERT_TRUE(output.reshape({1, 1, 64}));
            invokeSelectiveStateUpdate(
                x, a, b, c, dt, std::nullopt, std::nullopt, std::nullopt, state, output, false, std::nullopt, stream);
        }
        else if (route == 1)
        {
            invokeSelectiveStateUpdatePrefill(x, a, b, c, dt, std::nullopt, std::nullopt, std::nullopt, state, output,
                false, std::nullopt, std::nullopt, std::ref(da), std::ref(u), std::ref(replayB), std::ref(replayDt),
                stream);
        }
        else
        {
            invokeSelectiveStateUpdateDDTree(x, a, b, c, dt, std::nullopt, std::nullopt, state, output, parents, depths,
                false, std::nullopt, da, u, replayB, replayDt, stream);
        }
        rt::Tensor result({64}, rt::DeviceType::kCPU, DataType::kHALF);
        CUDA_CHECK(cudaMemcpyAsync(
            result.rawPointer(), output.rawPointer(), 64 * sizeof(half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto const* values = static_cast<half const*>(result.rawPointer());
        std::vector<half> const expected(
            64, __float2half(halfWeights ? -0.000005066394805908203125F : -0.000026524066925048828125F));
        EXPECT_EQ(std::vector<half>(values, values + 64), expected) << "case=" << caseIndex;
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

namespace
{

struct ScanGeometry
{
    char const* name;
    int32_t heads;
    int32_t dim;
    int32_t dstate;
    int32_t groups;
};

// Bolt (Nemotron 3.5) and Nano 9B v2 head layouts plus the remaining instantiated dstate extents.
std::vector<ScanGeometry> const kScanGeometries{
    {"bolt_h128_d64_n128_g8", 128, 64, 128, 8},
    {"nano9b_h128_d80_n128_g8", 128, 80, 128, 8},
    {"h16_d64_n64_g2", 16, 64, 64, 2},
    {"h8_d128_n256_g1", 8, 128, 256, 1},
};
ScanGeometry const kBoltGeometry = kScanGeometries[0];

constexpr int32_t kScanBatch{2};
constexpr int32_t kScanResidentRows{3};
// Longer than the 8-token multi-token path, so a scan with this extent runs the token-serial kernel.
constexpr int32_t kTokenSerialSeqLen{16};

struct ScanInputs
{
    ScanGeometry geometry;
    // Token-major operands with kTokenSerialSeqLen tokens per sequence.
    std::vector<half> x;
    std::vector<half> b;
    std::vector<half> c;
    std::vector<half> dtHalf;
    std::vector<float> dtFloat;
    std::vector<float> a;
    std::vector<half> dtBias;
    std::vector<half> d;
    std::vector<half> state;
    std::vector<int32_t> stateIndices{2, 0};

    size_t xElems() const
    {
        return static_cast<size_t>(geometry.heads) * geometry.dim;
    }
    size_t bcElems() const
    {
        return static_cast<size_t>(geometry.groups) * geometry.dstate;
    }
    size_t rowElems() const
    {
        return xElems() * geometry.dstate;
    }
};

ScanInputs makeScanInputs(ScanGeometry const& geometry, uint32_t seed)
{
    std::mt19937 gen{seed};
    auto fill = [&gen](auto& values, size_t count, float low, float high) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        std::uniform_real_distribution<float> dist{low, high};
        values.resize(count);
        for (auto& value : values)
        {
            value = Value(dist(gen));
        }
    };
    ScanInputs in{geometry};
    size_t const tokens = static_cast<size_t>(kScanBatch) * kTokenSerialSeqLen;
    fill(in.x, tokens * in.xElems(), -1.F, 1.F);
    fill(in.b, tokens * in.bcElems(), -1.F, 1.F);
    fill(in.c, tokens * in.bcElems(), -1.F, 1.F);
    fill(in.dtHalf, tokens * geometry.heads, -4.F, 2.F);
    fill(in.dtFloat, tokens * geometry.heads, -4.F, 2.F);
    fill(in.a, geometry.heads, -8.F, -0.5F);
    fill(in.dtBias, geometry.heads, -0.5F, 0.5F);
    fill(in.d, geometry.heads, -1.F, 1.F);
    fill(in.state, kScanResidentRows * in.rowElems(), -1.F, 1.F);
    return in;
}

template <typename T>
std::vector<T> selectTokens(std::vector<T> const& src, int32_t firstToken, int32_t seqLen, size_t tokenElements)
{
    std::vector<T> dst(static_cast<size_t>(kScanBatch) * seqLen * tokenElements);
    for (int32_t batch = 0; batch < kScanBatch; ++batch)
    {
        auto const begin = src.begin() + (static_cast<size_t>(batch) * kTokenSerialSeqLen + firstToken) * tokenElements;
        std::copy(
            begin, begin + seqLen * tokenElements, dst.begin() + static_cast<size_t>(batch) * seqLen * tokenElements);
    }
    return dst;
}

//! Resident state pool; @p offsetElements shifts the view off the allocation's alignment.
struct StatePool
{
    rt::Tensor storage;
    rt::Tensor view;
};

StatePool uploadState(ScanInputs const& in, std::vector<half> const& host, int64_t offsetElements = 0)
{
    ScanGeometry const& g = in.geometry;
    StatePool pool;
    pool.storage
        = rt::Tensor({static_cast<int64_t>(host.size()) + offsetElements}, rt::DeviceType::kGPU, DataType::kHALF);
    half* const base = static_cast<half*>(pool.storage.rawPointer()) + offsetElements;
    CUDA_CHECK(cudaMemcpy(base, host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice));
    pool.view = rt::Tensor(
        base, rt::Coords{kScanResidentRows, g.heads, g.dim, g.dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    return pool;
}

std::vector<half> downloadState(StatePool const& pool)
{
    std::vector<half> host(static_cast<size_t>(pool.view.getShape().volume()));
    CUDA_CHECK(cudaMemcpy(host.data(), pool.view.rawPointer(), host.size() * sizeof(half), cudaMemcpyDeviceToHost));
    return host;
}

struct ScanRun
{
    std::vector<half> output; // [batch, seqLen, heads, dim]
    std::vector<float> replayDA;
    std::vector<float> replayU;
    std::vector<float> replayB;
    std::vector<float> replayDT;
};

//! Runs tokens [firstToken, firstToken + seqLen) of every sequence through invokeSelectiveStateUpdatePrefill
//! against the resident pool @p state. With @p replayStash, the stash tensors are written and kept.
ScanRun runScan(ScanInputs const& in, int32_t firstToken, int32_t seqLen, std::vector<int32_t> const* contextLengths,
    bool floatDt, rt::Tensor& state, std::vector<rt::Tensor>* replayStash, cudaStream_t stream)
{
    ScanGeometry const& g = in.geometry;
    auto x = rt::Tensor({kScanBatch, seqLen, g.heads, g.dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto b = rt::Tensor({kScanBatch, seqLen, g.groups, g.dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto c = rt::Tensor({kScanBatch, seqLen, g.groups, g.dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    DataType const weightType = floatDt ? DataType::kFLOAT : DataType::kHALF;
    auto dt = rt::Tensor({kScanBatch, seqLen, g.heads}, rt::DeviceType::kGPU, weightType);
    auto dtBias = rt::Tensor({g.heads}, rt::DeviceType::kGPU, weightType);
    auto d = rt::Tensor({g.heads}, rt::DeviceType::kGPU, weightType);
    auto a = rt::Tensor({g.heads}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto output = rt::Tensor({kScanBatch, seqLen, g.heads, g.dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto stateIndices = rt::Tensor({kScanBatch}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(x, selectTokens(in.x, firstToken, seqLen, in.xElems()));
    copyHostToDevice(b, selectTokens(in.b, firstToken, seqLen, in.bcElems()));
    copyHostToDevice(c, selectTokens(in.c, firstToken, seqLen, in.bcElems()));
    if (floatDt)
    {
        copyHostToDevice(dt, selectTokens(in.dtFloat, firstToken, seqLen, g.heads));
        std::vector<float> biasHost(g.heads);
        std::vector<float> dHost(g.heads);
        for (int32_t head = 0; head < g.heads; ++head)
        {
            biasHost[head] = __half2float(in.dtBias[head]);
            dHost[head] = __half2float(in.d[head]);
        }
        copyHostToDevice(dtBias, biasHost);
        copyHostToDevice(d, dHost);
    }
    else
    {
        copyHostToDevice(dt, selectTokens(in.dtHalf, firstToken, seqLen, g.heads));
        copyHostToDevice(dtBias, in.dtBias);
        copyHostToDevice(d, in.d);
    }
    copyHostToDevice(a, in.a);
    copyHostToDevice(stateIndices, in.stateIndices);
    CUDA_CHECK(cudaMemsetAsync(output.rawPointer(), 0xFF, output.getMemoryCapacity(), stream));

    rt::Tensor lengths;
    if (contextLengths != nullptr)
    {
        lengths = rt::Tensor({kScanBatch}, rt::DeviceType::kGPU, DataType::kINT32);
        copyHostToDevice(lengths, *contextLengths);
    }
    rt::OptionalOutputTensor replayDA = std::nullopt;
    rt::OptionalOutputTensor replayU = std::nullopt;
    rt::OptionalOutputTensor replayB = std::nullopt;
    rt::OptionalOutputTensor replayDT = std::nullopt;
    if (replayStash != nullptr)
    {
        replayStash->clear();
        replayStash->emplace_back(rt::Coords{kScanBatch, seqLen, g.heads}, rt::DeviceType::kGPU, DataType::kFLOAT);
        replayStash->emplace_back(
            rt::Coords{kScanBatch, seqLen, g.heads, g.dim}, rt::DeviceType::kGPU, DataType::kFLOAT);
        replayStash->emplace_back(
            rt::Coords{kScanBatch, seqLen, g.groups, g.dstate}, rt::DeviceType::kGPU, DataType::kFLOAT);
        replayStash->emplace_back(rt::Coords{kScanBatch, seqLen, g.heads}, rt::DeviceType::kGPU, DataType::kFLOAT);
        for (auto& tensor : *replayStash)
        {
            CUDA_CHECK(cudaMemsetAsync(tensor.rawPointer(), 0, tensor.getMemoryCapacity(), stream));
        }
        replayDA = std::ref((*replayStash)[0]);
        replayU = std::ref((*replayStash)[1]);
        replayB = std::ref((*replayStash)[2]);
        replayDT = std::ref((*replayStash)[3]);
    }
    invokeSelectiveStateUpdatePrefill(x, a, b, c, dt, std::cref(dtBias), std::cref(d), std::nullopt, state, output,
        /* dtSoftplus=*/true, contextLengths != nullptr ? std::optional(std::cref(lengths)) : std::nullopt,
        std::cref(stateIndices), replayDA, replayU, replayB, replayDT, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ScanRun run;
    run.output = copyDeviceToHost<half>(output);
    if (replayStash != nullptr)
    {
        run.replayDA = copyDeviceToHost<float>((*replayStash)[0]);
        run.replayU = copyDeviceToHost<float>((*replayStash)[1]);
        run.replayB = copyDeviceToHost<float>((*replayStash)[2]);
        run.replayDT = copyDeviceToHost<float>((*replayStash)[3]);
    }
    return run;
}

template <typename T>
bool sameBits(T lhs, T rhs)
{
    return std::memcmp(&lhs, &rhs, sizeof(T)) == 0;
}

//! Counts bitwise differences between token rows [0, tokens[b]) of two [batch, seq, elements] buffers.
template <typename T>
int64_t countTokenMismatches(std::vector<T> const& lhs, int32_t lhsSeq, std::vector<T> const& rhs, int32_t rhsSeq,
    std::vector<int32_t> const& tokens, size_t elements)
{
    int64_t mismatches = 0;
    for (int32_t batch = 0; batch < kScanBatch; ++batch)
    {
        for (int32_t token = 0; token < tokens[batch]; ++token)
        {
            size_t const lhsBase = (static_cast<size_t>(batch) * lhsSeq + token) * elements;
            size_t const rhsBase = (static_cast<size_t>(batch) * rhsSeq + token) * elements;
            for (size_t i = 0; i < elements; ++i)
            {
                mismatches += sameBits(lhs[lhsBase + i], rhs[rhsBase + i]) ? 0 : 1;
            }
        }
    }
    return mismatches;
}

template <typename T>
int64_t countMismatches(std::vector<T> const& lhs, std::vector<T> const& rhs)
{
    EXPECT_EQ(lhs.size(), rhs.size());
    int64_t mismatches = 0;
    for (size_t i = 0; i < std::min(lhs.size(), rhs.size()); ++i)
    {
        mismatches += sameBits(lhs[i], rhs[i]) ? 0 : 1;
    }
    return mismatches;
}

//! Runs a short scan (seqLen tokens, context lengths @p shortLengths) and the token-serial scan over the same tokens
//! (context lengths @p serialLengths), then expects bitwise-equal outputs for every token below seqLen, an identical
//! resident pool and, for verify, an identical stash for the tokens each sequence consumed.
void expectShortScanMatchesTokenSerial(ScanInputs const& in, int32_t seqLen, bool verify, bool floatDt,
    std::vector<int32_t> const* shortLengths, std::vector<int32_t> const& serialLengths, int64_t stateOffset,
    std::string const& label, cudaStream_t stream)
{
    std::vector<rt::Tensor> shortStash;
    std::vector<rt::Tensor> serialStash;
    StatePool shortState = uploadState(in, in.state, stateOffset);
    StatePool serialState = uploadState(in, in.state);
    ScanRun const shortRun
        = runScan(in, 0, seqLen, shortLengths, floatDt, shortState.view, verify ? &shortStash : nullptr, stream);
    ScanRun const serialRun = runScan(
        in, 0, kTokenSerialSeqLen, &serialLengths, floatDt, serialState.view, verify ? &serialStash : nullptr, stream);

    std::vector<int32_t> const allTokens(kScanBatch, seqLen);
    EXPECT_EQ(
        countTokenMismatches(shortRun.output, seqLen, serialRun.output, kTokenSerialSeqLen, allTokens, in.xElems()), 0)
        << label << " output";
    auto const shortStateHost = downloadState(shortState);
    EXPECT_EQ(countMismatches(shortStateHost, downloadState(serialState)), 0) << label << " state";
    if (verify)
    {
        EXPECT_EQ(countMismatches(shortStateHost, in.state), 0) << label << " committed state written";
        int32_t const heads = in.geometry.heads;
        EXPECT_EQ(countTokenMismatches(
                      shortRun.replayDA, seqLen, serialRun.replayDA, kTokenSerialSeqLen, serialLengths, heads),
            0)
            << label << " dA";
        EXPECT_EQ(countTokenMismatches(
                      shortRun.replayU, seqLen, serialRun.replayU, kTokenSerialSeqLen, serialLengths, in.xElems()),
            0)
            << label << " u";
        EXPECT_EQ(countTokenMismatches(
                      shortRun.replayB, seqLen, serialRun.replayB, kTokenSerialSeqLen, serialLengths, in.bcElems()),
            0)
            << label << " B";
        EXPECT_EQ(countTokenMismatches(
                      shortRun.replayDT, seqLen, serialRun.replayDT, kTokenSerialSeqLen, serialLengths, heads),
            0)
            << label << " dt";
    }
}

} // namespace

TEST(MambaSelectiveStateUpdate, MultiTokenScanMatchesTokenSerialScanBitwise)
{
    // Given Bolt, Nano 9B v2, dstate-64 and dstate-256 head layouts, a two-sequence batch mapped onto non-identity
    // resident state rows, and 1 to 8 tokens per sequence,
    // when the short-sequence scan runs as decode (state written back) or as chain verify (replay stash written,
    // second sequence one token shorter),
    // then outputs, the resident state pool, and the replay stash are bitwise identical to the token-serial scan.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (ScanGeometry const& geometry : kScanGeometries)
    {
        ScanInputs const in = makeScanInputs(geometry, 20261019U);
        for (bool const floatDt : {false, true})
        {
            for (bool const verify : {false, true})
            {
                for (int32_t seqLen = 1; seqLen <= 8; ++seqLen)
                {
                    std::vector<int32_t> const lengths = verify ? std::vector<int32_t>{seqLen, std::max(1, seqLen - 1)}
                                                                : std::vector<int32_t>{seqLen, seqLen};
                    std::string const label = std::string(geometry.name) + " floatDt=" + std::to_string(floatDt)
                        + " verify=" + std::to_string(verify) + " seqLen=" + std::to_string(seqLen);
                    expectShortScanMatchesTokenSerial(
                        in, seqLen, verify, floatDt, verify ? &lengths : nullptr, lengths, 0, label, stream);
                }
            }
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(MambaSelectiveStateUpdate, MultiTokenScanEdgeSlotsAndLengthsMatchTokenSerialScan)
{
    // Given a Bolt-layout batch where one sequence maps to the pad slot (-1), or where context lengths exceed the
    // token extent or are zero,
    // when the short-sequence scan runs as decode or chain verify,
    // then it matches the token-serial scan run with the lengths clamped to the token extent: the pad sequence
    // reads zero state and writes no state, and a zero-length sequence emits zero outputs and leaves its state and
    // stash untouched.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ScanInputs padIn = makeScanInputs(kBoltGeometry, 7U);
    padIn.stateIndices = {-1, 0};
    ScanInputs const lengthIn = makeScanInputs(kBoltGeometry, 8U);
    for (bool const verify : {false, true})
    {
        for (int32_t const seqLen : {1, 5, 8})
        {
            std::string const suffix = " verify=" + std::to_string(verify) + " seqLen=" + std::to_string(seqLen);
            std::vector<int32_t> const full{seqLen, seqLen};
            expectShortScanMatchesTokenSerial(
                padIn, seqLen, verify, false, &full, full, 0, "pad slot" + suffix, stream);
            std::vector<int32_t> const requested{seqLen + 3, 0};
            std::vector<int32_t> const clamped{seqLen, 0};
            expectShortScanMatchesTokenSerial(
                lengthIn, seqLen, verify, false, &requested, clamped, 0, "lengths" + suffix, stream);
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(MambaSelectiveStateUpdate, MultiTokenScanWithUnalignedStateMatchesTokenSerialScan)
{
    // Given a resident state pool whose base is only 4-byte aligned, so whole dstate slices cannot be vector loaded,
    // when decode and chain verify run with 1, 2, and 8 tokens,
    // then results are bitwise identical to the token-serial scan.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ScanInputs const in = makeScanInputs(kBoltGeometry, 9U);
    for (bool const verify : {false, true})
    {
        for (int32_t const seqLen : {1, 2, 8})
        {
            std::vector<int32_t> const lengths{seqLen, seqLen};
            std::string const label = "verify=" + std::to_string(verify) + " seqLen=" + std::to_string(seqLen);
            expectShortScanMatchesTokenSerial(
                in, seqLen, verify, false, verify ? &lengths : nullptr, lengths, /* stateOffset=*/2, label, stream);
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(MambaSelectiveStateUpdate, ChainVerifyThenPartialReplayMatchesSequentialDecode)
{
    // Given a Bolt-layout two-sequence batch on non-identity resident rows and eight chain-verify tokens per sequence,
    // when the tokens are verified in one pass and the replay commit then accepts a different prefix per sequence,
    // including zero, one, and all tokens,
    // then every verify output equals the output of decoding that token alone after its predecessors, and each
    // committed resident row equals the state reached by decoding exactly the accepted prefix; the unmapped
    // resident row is untouched.
    constexpr int32_t kVerifyTokens{8};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ScanInputs const in = makeScanInputs(kBoltGeometry, 1838U);
    size_t const xElems = in.xElems();
    size_t const rowElems = in.rowElems();

    std::vector<std::vector<half>> decodeOutputs;
    std::vector<std::vector<half>> decodeStates{in.state};
    StatePool decodeState = uploadState(in, in.state);
    for (int32_t token = 0; token < kVerifyTokens; ++token)
    {
        decodeOutputs.push_back(runScan(in, token, 1, nullptr, false, decodeState.view, nullptr, stream).output);
        decodeStates.push_back(downloadState(decodeState));
    }

    std::vector<int32_t> const verifyLengths{kVerifyTokens, kVerifyTokens};
    for (std::vector<int32_t> const& accepted :
        {std::vector<int32_t>{3, 8}, std::vector<int32_t>{0, 1}, std::vector<int32_t>{8, 5}})
    {
        std::vector<rt::Tensor> stash;
        StatePool state = uploadState(in, in.state);
        ScanRun const verifyRun = runScan(in, 0, kVerifyTokens, &verifyLengths, false, state.view, &stash, stream);
        for (int32_t batch = 0; batch < kScanBatch; ++batch)
        {
            int64_t mismatches = 0;
            for (int32_t token = 0; token < kVerifyTokens; ++token)
            {
                for (size_t i = 0; i < xElems; ++i)
                {
                    mismatches
                        += sameBits(verifyRun.output[(static_cast<size_t>(batch) * kVerifyTokens + token) * xElems + i],
                               decodeOutputs[token][static_cast<size_t>(batch) * xElems + i])
                        ? 0
                        : 1;
                }
            }
            EXPECT_EQ(mismatches, 0) << "verify output, batch=" << batch;
        }

        MambaReplayLayerInfo const hostInfo{state.view.rawPointer(), stash[0].rawPointer(), stash[1].rawPointer(),
            stash[2].rawPointer(), stash[3].rawPointer()};
        rt::Tensor info({static_cast<int64_t>(sizeof(MambaReplayLayerInfo))}, rt::DeviceType::kGPU, DataType::kUINT8);
        CUDA_CHECK(cudaMemcpy(info.rawPointer(), &hostInfo, sizeof(hostInfo), cudaMemcpyHostToDevice));
        auto acceptedDevice = rt::Tensor({kScanBatch}, rt::DeviceType::kGPU, DataType::kINT32);
        auto stateIndices = rt::Tensor({kScanBatch}, rt::DeviceType::kGPU, DataType::kINT32);
        copyHostToDevice(acceptedDevice, accepted);
        copyHostToDevice(stateIndices, in.stateIndices);
        invokeMambaReplayReconstructBatched(static_cast<MambaReplayLayerInfo const*>(info.rawPointer()), 1, state.view,
            stash[1], stash[2], stash[3], acceptedDevice, stateIndices, kScanBatch, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto const committed = downloadState(state);
        std::vector<half> expected = in.state;
        for (int32_t batch = 0; batch < kScanBatch; ++batch)
        {
            size_t const row = static_cast<size_t>(in.stateIndices[batch]) * rowElems;
            auto const& source = decodeStates[accepted[batch]];
            std::copy(source.begin() + row, source.begin() + row + rowElems, expected.begin() + row);
        }
        EXPECT_EQ(countMismatches(committed, expected), 0) << "accepted={" << accepted[0] << ", " << accepted[1] << "}";
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

namespace
{

//! Host replay of the accepted tokens with the device's FP32 operation order: dB = B * dt,
//! S = fma(dB, u, S * dA), rounded to the state type after every token.
template <typename State>
std::vector<State> replayReference(std::vector<State> const& state, std::vector<float> const& da,
    std::vector<float> const& u, std::vector<float> const& b, std::vector<float> const& dt,
    std::vector<std::vector<int32_t>> const& tokens, std::vector<int32_t> const& stateIndices, int32_t seqLen,
    int32_t heads, int32_t dim, int32_t dstate, int32_t groups)
{
    std::vector<State> result = state;
    for (int32_t batch = 0; batch < static_cast<int32_t>(tokens.size()); ++batch)
    {
        for (int32_t head = 0; head < heads; ++head)
        {
            int32_t const group = head / (heads / groups);
            for (int32_t d = 0; d < dim; ++d)
            {
                for (int32_t n = 0; n < dstate; ++n)
                {
                    size_t const index
                        = ((static_cast<size_t>(stateIndices[batch]) * heads + head) * dim + d) * dstate + n;
                    float s = static_cast<float>(result[index]);
                    for (int32_t const token : tokens[batch])
                    {
                        size_t const tokenRow = static_cast<size_t>(batch) * seqLen + token;
                        float const dB = b[(tokenRow * groups + group) * dstate + n] * dt[tokenRow * heads + head];
                        s = std::fma(dB, u[(tokenRow * heads + head) * dim + d], s * da[tokenRow * heads + head]);
                        s = static_cast<float>(State(s));
                    }
                    result[index] = State(s);
                }
            }
        }
    }
    return result;
}

template <typename State>
void expectReplayMatchesReference(int32_t dstate, int64_t stateOffset, bool indexed, std::string const& label)
{
    constexpr int32_t kHeads{16};
    constexpr int32_t kDim{64};
    constexpr int32_t kGroups{2};
    constexpr int32_t kSeqLen{8};
    DataType const stateType = std::is_same_v<State, float> ? DataType::kFLOAT : DataType::kHALF;
    std::mt19937 gen{static_cast<uint32_t>(dstate * 31 + stateOffset * 7 + (indexed ? 1 : 0))};
    auto uniform = [&gen](size_t count, float low, float high) {
        std::uniform_real_distribution<float> dist{low, high};
        std::vector<float> values(count);
        for (auto& value : values)
        {
            value = dist(gen);
        }
        return values;
    };
    size_t const stateElems = static_cast<size_t>(kScanResidentRows) * kHeads * kDim * dstate;
    std::vector<State> stateHost(stateElems);
    for (auto& value : stateHost)
    {
        value = State(std::uniform_real_distribution<float>{-1.F, 1.F}(gen));
    }
    auto const da = uniform(static_cast<size_t>(kScanBatch) * kSeqLen * kHeads, 0.3F, 1.F);
    auto const u = uniform(static_cast<size_t>(kScanBatch) * kSeqLen * kHeads * kDim, -1.F, 1.F);
    auto const b = uniform(static_cast<size_t>(kScanBatch) * kSeqLen * kGroups * dstate, -1.F, 1.F);
    auto const dt = uniform(static_cast<size_t>(kScanBatch) * kSeqLen * kHeads, 0.1F, 2.F);
    std::vector<int32_t> const stateIndices{2, 0};
    std::vector<int32_t> const accepted{indexed ? 3 : 5, indexed ? 2 : 0};
    std::vector<int32_t> const ids{0, 2, 5, 1, 7, -1};
    std::vector<std::vector<int32_t>> tokens(kScanBatch);
    for (int32_t batch = 0; batch < kScanBatch; ++batch)
    {
        for (int32_t t = 0; t < accepted[batch]; ++t)
        {
            tokens[batch].push_back(indexed ? ids[batch * 3 + t] : t);
        }
    }

    rt::Tensor storage({static_cast<int64_t>(stateElems) + stateOffset}, rt::DeviceType::kGPU, stateType);
    State* const base = static_cast<State*>(storage.rawPointer()) + stateOffset;
    CUDA_CHECK(cudaMemcpy(base, stateHost.data(), stateElems * sizeof(State), cudaMemcpyHostToDevice));
    rt::Tensor state(base, rt::Coords{kScanResidentRows, kHeads, kDim, dstate}, rt::DeviceType::kGPU, stateType);
    auto daDevice = rt::Tensor({kScanBatch, kSeqLen, kHeads}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto uDevice = rt::Tensor({kScanBatch, kSeqLen, kHeads, kDim}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto bDevice = rt::Tensor({kScanBatch, kSeqLen, kGroups, dstate}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto dtDevice = rt::Tensor({kScanBatch, kSeqLen, kHeads}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto acceptedDevice = rt::Tensor({kScanBatch}, rt::DeviceType::kGPU, DataType::kINT32);
    auto indicesDevice = rt::Tensor({kScanBatch}, rt::DeviceType::kGPU, DataType::kINT32);
    auto idsDevice = rt::Tensor({kScanBatch, 3}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(daDevice, da);
    copyHostToDevice(uDevice, u);
    copyHostToDevice(bDevice, b);
    copyHostToDevice(dtDevice, dt);
    copyHostToDevice(acceptedDevice, accepted);
    copyHostToDevice(indicesDevice, stateIndices);
    copyHostToDevice(idsDevice, ids);
    MambaReplayLayerInfo const hostInfo{
        base, daDevice.rawPointer(), uDevice.rawPointer(), bDevice.rawPointer(), dtDevice.rawPointer()};
    rt::Tensor info({static_cast<int64_t>(sizeof(MambaReplayLayerInfo))}, rt::DeviceType::kGPU, DataType::kUINT8);
    CUDA_CHECK(cudaMemcpy(info.rawPointer(), &hostInfo, sizeof(hostInfo), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    invokeMambaReplayReconstructBatched(static_cast<MambaReplayLayerInfo const*>(info.rawPointer()), 1, state, uDevice,
        bDevice, dtDevice, acceptedDevice, indicesDevice, kScanBatch, stream, indexed ? &idsDevice : nullptr,
        indexed ? 3 : 0);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::vector<State> committed(stateElems);
    CUDA_CHECK(cudaMemcpy(committed.data(), base, stateElems * sizeof(State), cudaMemcpyDeviceToHost));
    auto const expected
        = replayReference(stateHost, da, u, b, dt, tokens, stateIndices, kSeqLen, kHeads, kDim, dstate, kGroups);
    EXPECT_EQ(countMismatches(committed, expected), 0) << label;
}

} // namespace

TEST(MambaSelectiveStateUpdate, ReplayCommitMatchesFmaReferenceBitwise)
{
    // Given FP16 and FP32 resident states, aligned and 4-byte-offset pools, dstate 128 and 80, and prefix
    // (lengths 5 and 0) or indexed (ids {0,2,5} and {1,7}) acceptance on non-identity resident rows,
    // when the replay commit runs,
    // then every committed state element equals a host replay that rounds dB * u + S * dA once per token to the
    // state type, and rows outside the accepted sequences are untouched.
    for (bool const indexed : {false, true})
    {
        for (int32_t const dstate : {128, 80})
        {
            for (int64_t const offset : {int64_t{0}, int64_t{1}})
            {
                std::string const label = "dstate=" + std::to_string(dstate) + " offset=" + std::to_string(offset)
                    + " indexed=" + std::to_string(indexed);
                expectReplayMatchesReference<half>(dstate, offset, indexed, "half " + label);
                expectReplayMatchesReference<float>(dstate, offset, indexed, "float " + label);
            }
        }
    }
}

class MambaStateCopy : public ::testing::TestWithParam<std::array<int32_t, 3>>
{
protected:
    void SetUp() override
    {
        cudaStream_t stream{};
        CUDA_CHECK(cudaStreamCreate(&stream));
        mStream.reset(stream);
    }

    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)> mStream{
        nullptr, cudaStreamDestroy};
};

TEST_P(MambaStateCopy, GatherScatterNonContiguousResidentRows)
{
    // Given patterned state rows of varying sizes and noncontiguous resident indices,
    // when gathering and scattering one or two active rows,
    // then every selected element is copied exactly and unselected rows remain unchanged.
    for (int32_t const activeRows : {1, 2})
    {
        constexpr int32_t residentRows = 5;
        int32_t const nheads = GetParam().at(0);
        int32_t const dim = GetParam().at(1);
        int32_t const dstate = GetParam().at(2);
        int32_t const rowElements = nheads * dim * dstate;
        SCOPED_TRACE(activeRows);

        std::vector<half> residentHost(static_cast<size_t>(residentRows) * rowElements);
        for (int32_t row = 0; row < residentRows; ++row)
        {
            for (int32_t element = 0; element < rowElements; ++element)
            {
                residentHost.at(row * rowElements + element)
                    = __float2half(static_cast<float>(row * 256 + element % 251));
            }
        }
        std::vector<int32_t> indicesHost{3, 1};
        indicesHost.resize(activeRows);
        auto resident = rt::Tensor({residentRows, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
        auto active = rt::Tensor({activeRows, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
        auto indices = rt::Tensor({activeRows}, rt::DeviceType::kGPU, DataType::kINT32);
        copyHostToDevice(resident, residentHost);
        copyHostToDevice(indices, indicesHost);
        std::vector<half> activeHost(static_cast<size_t>(activeRows) * rowElements, __float2half(-1.0F));
        copyHostToDevice(active, activeHost);

        cudaStream_t const stream = mStream.get();
        invokeMambaStateGather(resident, active, indices, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        activeHost = copyDeviceToHost<half>(active);
        for (int32_t activeRow = 0; activeRow < activeRows; ++activeRow)
        {
            for (int32_t element = 0; element < rowElements; ++element)
            {
                ASSERT_EQ(__half2float(activeHost.at(activeRow * rowElements + element)),
                    __half2float(residentHost.at(indicesHost.at(activeRow) * rowElements + element)))
                    << element;
            }
        }

        for (int32_t row = 0; row < activeRows; ++row)
        {
            for (int32_t element = 0; element < rowElements; ++element)
            {
                activeHost.at(row * rowElements + element)
                    = __float2half(static_cast<float>(-2 - row * 256 - element % 251));
            }
            std::copy_n(activeHost.begin() + row * rowElements, rowElements,
                residentHost.begin() + indicesHost.at(row) * rowElements);
        }
        copyHostToDevice(active, activeHost);
        invokeMambaStateScatter(active, resident, indices, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto const residentAfter = copyDeviceToHost<half>(resident);
        for (int32_t row = 0; row < residentRows; ++row)
        {
            for (int32_t element = 0; element < rowElements; ++element)
            {
                ASSERT_EQ(__half2float(residentAfter.at(row * rowElements + element)),
                    __half2float(residentHost.at(row * rowElements + element)))
                    << row << ":" << element;
            }
        }
    }
}

TEST_P(MambaStateCopy, GatherScatterUsesLogicalRowShapeAfterReshape)
{
    // Given state buffers whose logical row count is smaller than their allocation,
    // when gathering and scattering through the logical views,
    // then selected rows match exactly and the unused backing rows remain unchanged.
    constexpr int32_t logicalRows = 2;
    constexpr int32_t allocatedRows = 4;
    int32_t const nheads = GetParam().at(0);
    int32_t const dim = GetParam().at(1);
    int32_t const dstate = GetParam().at(2);
    int32_t const rowElements = nheads * dim * dstate;

    std::vector<half> residentHost(static_cast<size_t>(allocatedRows) * rowElements);
    std::vector<half> activeHost(static_cast<size_t>(allocatedRows) * rowElements, __float2half(-1.0F));
    for (int32_t row = 0; row < allocatedRows; ++row)
    {
        std::fill_n(residentHost.begin() + static_cast<size_t>(row) * rowElements, rowElements,
            __float2half(static_cast<float>(row + 1)));
    }
    std::vector<int32_t> const indicesHost{1, 0};

    auto resident = rt::Tensor({allocatedRows, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto active = rt::Tensor({allocatedRows, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(resident, residentHost);
    copyHostToDevice(active, activeHost);
    ASSERT_TRUE(resident.reshape({logicalRows, nheads, dim, dstate}));
    ASSERT_TRUE(active.reshape({logicalRows, nheads, dim, dstate}));
    auto indices = rt::Tensor({logicalRows}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(indices, indicesHost);

    cudaStream_t const stream = mStream.get();
    invokeMambaStateGather(resident, active, indices, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    activeHost = copyDeviceToHost<half>(active);
    for (int32_t row = 0; row < logicalRows; ++row)
    {
        for (int32_t element = 0; element < rowElements; ++element)
        {
            EXPECT_EQ(
                __half2float(activeHost.at(row * rowElements + element)), static_cast<float>(indicesHost.at(row) + 1));
        }
    }
    ASSERT_TRUE(active.reshape({allocatedRows, nheads, dim, dstate}));
    activeHost = copyDeviceToHost<half>(active);
    for (int32_t row = logicalRows; row < allocatedRows; ++row)
    {
        for (int32_t element = 0; element < rowElements; ++element)
        {
            EXPECT_EQ(__half2float(activeHost.at(row * rowElements + element)), -1.0F);
        }
    }
    ASSERT_TRUE(active.reshape({logicalRows, nheads, dim, dstate}));
    activeHost.resize(static_cast<size_t>(logicalRows) * rowElements);

    std::fill_n(activeHost.begin(), rowElements, __float2half(11.0F));
    std::fill_n(activeHost.begin() + rowElements, rowElements, __float2half(12.0F));
    copyHostToDevice(active, activeHost);
    invokeMambaStateScatter(active, resident, indices, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ASSERT_TRUE(resident.reshape({allocatedRows, nheads, dim, dstate}));
    residentHost = copyDeviceToHost<half>(resident);
    for (int32_t row = 0; row < allocatedRows; ++row)
    {
        float const expected = row == 0 ? 12.0F : (row == 1 ? 11.0F : static_cast<float>(row + 1));
        for (int32_t element = 0; element < rowElements; ++element)
        {
            EXPECT_EQ(__half2float(residentHost.at(row * rowElements + element)), expected);
        }
    }
}

TEST_P(MambaStateCopy, InvalidResidentRowsAreIsolated)
{
    // Given valid, negative, and out-of-range resident indices with guarded backing rows,
    // when gathering and scattering the active state,
    // then invalid gathers produce zero and invalid scatters leave resident storage unchanged.
    constexpr int32_t activeRows = 3;
    constexpr int32_t residentRows = 2;
    int32_t const nheads = GetParam().at(0);
    int32_t const dim = GetParam().at(1);
    int32_t const dstate = GetParam().at(2);
    int32_t const rowElements = nheads * dim * dstate;

    constexpr int32_t backingRows = 4;
    std::vector<half> residentHost(static_cast<size_t>(backingRows) * rowElements);
    std::fill_n(residentHost.begin(), rowElements, __float2half(3.0F));
    std::fill_n(residentHost.begin() + rowElements, rowElements, __float2half(7.0F));
    std::fill_n(residentHost.begin() + 2 * rowElements, rowElements, __float2half(23.0F));
    std::fill_n(residentHost.begin() + 3 * rowElements, rowElements, __float2half(29.0F));
    std::vector<half> activeHost(static_cast<size_t>(activeRows) * rowElements, __float2half(19.0F));
    std::vector<int32_t> const indicesHost{1, -1, residentRows};

    auto residentBacking = rt::Tensor({backingRows, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto resident = rt::Tensor(
        residentBacking.rawPointer(), {residentRows, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto active = rt::Tensor({activeRows, nheads, dim, dstate}, rt::DeviceType::kGPU, DataType::kHALF);
    auto indices = rt::Tensor({activeRows}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(residentBacking, residentHost);
    copyHostToDevice(active, activeHost);
    copyHostToDevice(indices, indicesHost);

    cudaStream_t const stream = mStream.get();
    invokeMambaStateGather(resident, active, indices, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    activeHost = copyDeviceToHost<half>(active);
    for (int32_t element = 0; element < rowElements; ++element)
    {
        EXPECT_EQ(__half2float(activeHost.at(element)), 7.0F);
        EXPECT_EQ(__half2float(activeHost.at(rowElements + element)), 0.0F);
        EXPECT_EQ(__half2float(activeHost.at(2 * rowElements + element)), 0.0F);
    }

    std::fill_n(activeHost.begin(), rowElements, __float2half(11.0F));
    std::fill_n(activeHost.begin() + rowElements, rowElements, __float2half(13.0F));
    std::fill_n(activeHost.begin() + 2 * rowElements, rowElements, __float2half(17.0F));
    copyHostToDevice(active, activeHost);
    invokeMambaStateScatter(active, resident, indices, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const residentAfter = copyDeviceToHost<half>(residentBacking);
    for (int32_t element = 0; element < rowElements; ++element)
    {
        EXPECT_EQ(__half2float(residentAfter.at(element)), 3.0F);
        EXPECT_EQ(__half2float(residentAfter.at(rowElements + element)), 11.0F);
        EXPECT_EQ(__half2float(residentAfter.at(2 * rowElements + element)), 23.0F);
        EXPECT_EQ(__half2float(residentAfter.at(3 * rowElements + element)), 29.0F);
    }
}

INSTANTIATE_TEST_SUITE_P(StateRowShapes, MambaStateCopy,
    ::testing::Values(std::array<int32_t, 3>{1, 2, 4}, std::array<int32_t, 3>{1, 1, 8191},
        std::array<int32_t, 3>{1, 1, 8192}, std::array<int32_t, 3>{1, 1, 8193}, std::array<int32_t, 3>{1, 1, 8200},
        std::array<int32_t, 3>{96, 80, 128}),
    [](auto const& info) {
        auto const& shape = info.param;
        return "H" + std::to_string(shape.at(0)) + "D" + std::to_string(shape.at(1)) + "N"
            + std::to_string(shape.at(2));
    });
