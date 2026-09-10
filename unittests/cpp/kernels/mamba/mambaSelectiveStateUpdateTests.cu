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

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/mamba/selectiveStateUpdate.h"
#include "testUtils.h"

using namespace mamba_ssm;
using namespace trt_edgellm;
using namespace nvinfer1;

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
        outputDevice, config.dtSoftplus, stream);
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
        outputDevice, config.dtSoftplus, clOpt, /*replayDA=*/std::nullopt, /*replayU=*/std::nullopt,
        /*replayB=*/std::nullopt, stream);

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
