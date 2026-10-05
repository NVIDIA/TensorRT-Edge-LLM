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
