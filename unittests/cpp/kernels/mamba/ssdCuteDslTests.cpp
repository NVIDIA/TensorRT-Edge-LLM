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

/// Unit tests for CuTe DSL SSD (Mamba2 chunk scan) prefill kernel.
/// Verifies correctness by comparing against the serial invokeSelectiveStateUpdatePrefill reference.

#ifdef CUTE_DSL_SSD_ENABLED

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/mamba/cuteDslSSDRunner.h"
#include "kernels/mamba/selectiveStateUpdate.h"
#include "kernels/mamba/ssdVarlenMetadata.h"

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

// =============================================================================
// CPU Reference (token-by-token, same as invokeSelectiveStateUpdatePrefill)
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

size_t roundUpTo(size_t value, size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

class GuardedDeviceBuffer
{
public:
    explicit GuardedDeviceBuffer(size_t logicalBytes)
    {
        try
        {
            CUDA_DRIVER_CHECK(cuInit(0));

            int device{};
            CUDA_CHECK(cudaGetDevice(&device));
            CUdevice cuDevice{};
            CUDA_DRIVER_CHECK(cuDeviceGet(&cuDevice, device));

            int vmmSupported{};
            CUDA_DRIVER_CHECK(
                cuDeviceGetAttribute(&vmmSupported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cuDevice));
            if (vmmSupported == 0)
            {
                throw std::runtime_error("CUDA VMM is not supported on this device");
            }

            CUmemAllocationProp prop{};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device;
            prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

            size_t granularity{};
            CUDA_DRIVER_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
            mMappedBytes = roundUpTo(logicalBytes, granularity);
            mReservedBytes = mMappedBytes + granularity;

            CUDA_DRIVER_CHECK(cuMemAddressReserve(&mBase, mReservedBytes, granularity, 0, 0));
            mHasAddress = true;
            CUDA_DRIVER_CHECK(cuMemCreate(&mHandle, mMappedBytes, &prop, 0));
            mHasHandle = true;
            CUDA_DRIVER_CHECK(cuMemMap(mBase, mMappedBytes, 0, mHandle, 0));
            mIsMapped = true;

            CUmemAccessDesc access{};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CUDA_DRIVER_CHECK(cuMemSetAccess(mBase, mMappedBytes, &access, 1));

            mData = mBase + (mMappedBytes - logicalBytes);
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    GuardedDeviceBuffer(GuardedDeviceBuffer const&) = delete;
    GuardedDeviceBuffer& operator=(GuardedDeviceBuffer const&) = delete;

    ~GuardedDeviceBuffer()
    {
        cleanup();
    }

    void* data() const
    {
        return reinterpret_cast<void*>(mData);
    }

private:
    void cleanup() noexcept
    {
        if (mIsMapped)
        {
            (void) cuMemUnmap(mBase, mMappedBytes);
            mIsMapped = false;
        }
        if (mHasHandle)
        {
            (void) cuMemRelease(mHandle);
            mHasHandle = false;
        }
        if (mHasAddress)
        {
            (void) cuMemAddressFree(mBase, mReservedBytes);
            mHasAddress = false;
        }
        mBase = 0;
        mData = 0;
        mMappedBytes = 0;
        mReservedBytes = 0;
    }

    size_t mMappedBytes{};
    size_t mReservedBytes{};
    CUdeviceptr mBase{};
    CUdeviceptr mData{};
    CUmemGenericAllocationHandle mHandle{};
    bool mHasAddress{};
    bool mHasHandle{};
    bool mIsMapped{};
};

enum class GuardedSsdInput
{
    kX,
    kDt,
    kB,
    kC,
    kAll,
};

/// CPU reference for SSD prefill (sequential scan, matches selectiveStateUpdatePrefill).
void ssdPrefillReference(int32_t batch, int32_t seqLen, int32_t nheads, int32_t dim, int32_t dstate, int32_t ngroups,
    std::vector<half> const& x,       // [batch, seqLen, nheads, dim]
    std::vector<half> const& dt,      // [batch, seqLen, nheads]
    std::vector<float> const& A,      // [nheads]
    std::vector<half> const& B,       // [batch, seqLen, ngroups, dstate]
    std::vector<half> const& C,       // [batch, seqLen, ngroups, dstate]
    std::vector<float> const& D,      // [nheads]
    std::vector<float> const& dtBias, // [nheads]
    bool dtSoftplus,
    std::vector<float>& stateRef,                        // [batch, nheads, dim, dstate] -- in/out
    std::vector<half>& outputRef,                        // [batch, seqLen, nheads, dim]
    std::vector<int32_t> const* contextLengths = nullptr // [batch] (optional, nullptr = uniform = seqLen)
)
{
    int32_t const headsPerGroup = nheads / ngroups;

    for (int32_t b = 0; b < batch; ++b)
    {
        int32_t const cl = contextLengths ? (*contextLengths)[b] : seqLen;
        for (int32_t t = 0; t < seqLen; ++t)
        {
            // Padding region: state stays unchanged, output writes 0.
            // Mirrors selectiveStateUpdateMultiStepReferenceFp32 in
            // mambaSelectiveStateUpdateTests.cu:191-197.
            if (t >= cl)
            {
                for (int32_t h = 0; h < nheads; ++h)
                    for (int32_t d = 0; d < dim; ++d)
                        outputRef[((b * seqLen + t) * nheads + h) * dim + d] = __float2half(0.f);
                continue;
            }
            for (int32_t h = 0; h < nheads; ++h)
            {
                int32_t const g = h / headsPerGroup;
                float dtVal = __half2float(dt[((b * seqLen + t) * nheads) + h]) + dtBias[h];
                if (dtSoftplus)
                    dtVal = thresholdedSoftplus(dtVal);
                float const aVal = A[h];
                float const dA = std::exp(aVal * dtVal);
                float const dVal = D[h];

                for (int32_t d = 0; d < dim; ++d)
                {
                    float const xVal = __half2float(x[((b * seqLen + t) * nheads + h) * dim + d]);
                    float outVal = dVal * xVal;

                    for (int32_t ds = 0; ds < dstate; ++ds)
                    {
                        float const bVal = __half2float(B[((b * seqLen + t) * ngroups + g) * dstate + ds]);
                        float const cVal = __half2float(C[((b * seqLen + t) * ngroups + g) * dstate + ds]);

                        int64_t const sIdx
                            = static_cast<int64_t>(b) * nheads * dim * dstate + h * dim * dstate + d * dstate + ds;
                        float const newState = dA * stateRef[sIdx] + dtVal * bVal * xVal;
                        stateRef[sIdx] = newState;
                        outVal += newState * cVal;
                    }
                    outputRef[((b * seqLen + t) * nheads + h) * dim + d] = __float2half(outVal);
                }
            }
        }
    }
}

// =============================================================================
// Test fixture
// =============================================================================

struct SsdCuteDslTestConfig
{
    int32_t batch;
    int32_t seqLen;
    int32_t nheads;
    int32_t dim;
    int32_t dstate;
    int32_t ngroups;
    std::vector<int32_t> contextLengths{}; // empty = uniform (cl[b] == seqLen for all)
    bool useNonzeroInitState{false};       // when true, initialize state with deterministic random values
    bool poisonPadding{false};             // when true, fill every padded input element with a large value
};

class SsdCuteDslTest : public ::testing::TestWithParam<SsdCuteDslTestConfig>
{
};

TEST(SsdCuteDslCapability, SupportedConfigurationsMatchCompiledKernels)
{
    EXPECT_FALSE(CuteDslSSDRunner::canImplement(64, 128, 79));
    EXPECT_TRUE(CuteDslSSDRunner::canImplement(64, 128, 80));
    EXPECT_TRUE(CuteDslSSDRunner::canImplement(128, 64, 120));
    EXPECT_FALSE(CuteDslSSDRunner::canImplement(80, 64, 110));
    EXPECT_FALSE(CuteDslSSDRunner::canImplement(96, 128, 110));
    EXPECT_EQ(CuteDslSSDRunner::getWorkspaceSize(1, 128, 8, 80, 64, 1), 0);

#if defined(CUTE_DSL_SSD_BLACKWELL_ENABLED)
    EXPECT_TRUE(CuteDslSSDRunner::canImplement(80, 128, 100));
    EXPECT_TRUE(CuteDslSSDRunner::canImplement(80, 128, 101));
    EXPECT_TRUE(CuteDslSSDRunner::canImplement(80, 128, 110));
    EXPECT_FALSE(CuteDslSSDRunner::canImplement(80, 128, 80));
    EXPECT_FALSE(CuteDslSSDRunner::canImplement(80, 128, 90));
    EXPECT_FALSE(CuteDslSSDRunner::canImplement(80, 128, 120));
    EXPECT_GT(CuteDslSSDRunner::getWorkspaceSize(1, 128, 96, 80, 128, 8), 0);
#else
    EXPECT_FALSE(CuteDslSSDRunner::canImplement(80, 128, 100));
    EXPECT_EQ(CuteDslSSDRunner::getWorkspaceSize(1, 128, 96, 80, 128, 8), 0);
#endif
}

TEST_P(SsdCuteDslTest, CorrectnessVsSerialReference)
{
    auto const& cfg = GetParam();
    int32_t const batch = cfg.batch;
    int32_t const seqLen = cfg.seqLen;
    int32_t const nheads = cfg.nheads;
    int32_t const dim = cfg.dim;
    int32_t const dstate = cfg.dstate;
    int32_t const ngroups = cfg.ngroups;

    if (!CuteDslSSDRunner::canImplement(dim, dstate, 80))
    {
        GTEST_SKIP() << "CuteDslSSDRunner cannot implement dim=" << dim << " dstate=" << dstate;
    }

    // ssd_prefill_d128_n128 cubin fails to load on sm_87 (root cause TBD).
    {
        int sm_major{}, sm_minor{};
        cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, 0);
        cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, 0);
        if (sm_major == 8 && sm_minor == 7 && dim == 128 && dstate == 128)
        {
            GTEST_SKIP() << "ssd_prefill_d128_n128 cubin load fails on sm_87";
        }
    }

    // Allocate host data
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.f, 0.5f);
    std::uniform_real_distribution<float> uniform(0.1f, 0.6f);

    size_t const xSize = batch * seqLen * nheads * dim;
    size_t const dtSize = batch * seqLen * nheads;
    size_t const bSize = batch * seqLen * ngroups * dstate;
    size_t const stateSize = batch * nheads * dim * dstate;
    size_t const outSize = xSize;

    std::vector<half> xHost(xSize), dtHost(dtSize), bHost(bSize), cHost(bSize);
    std::vector<float> aHost(nheads), dHost(nheads), dtBiasHost(nheads);
    std::vector<float> stateHost(stateSize, 0.f);
    std::vector<half> outHost(outSize, __float2half(0.f));

    for (auto& v : xHost)
        v = __float2half(normal(rng));
    for (auto& v : dtHost)
        v = __float2half(uniform(rng));
    for (auto& v : bHost)
        v = __float2half(normal(rng));
    for (auto& v : cHost)
        v = __float2half(normal(rng));
    for (auto& v : aHost)
        v = -(uniform(rng) + 0.5f);
    for (auto& v : dHost)
        v = normal(rng) * 0.1f;
    for (auto& v : dtBiasHost)
        v = normal(rng) * 0.1f;
    // Initial SSM state: zeros by default, deterministic small values when test exercises has_init_states.
    if (cfg.useNonzeroInitState)
    {
        std::normal_distribution<float> stateNormal(0.f, 0.1f);
        for (auto& v : stateHost)
            v = stateNormal(rng);
    }

    // Optional per-batch context_lengths
    std::vector<int32_t> const* contextLengthsPtr = cfg.contextLengths.empty() ? nullptr : &cfg.contextLengths;
    if (cfg.poisonPadding)
    {
        ASSERT_EQ(static_cast<int32_t>(cfg.contextLengths.size()), batch);
        half const poison = __float2half(8.f);
        for (int32_t b = 0; b < batch; ++b)
        {
            for (int32_t t = cfg.contextLengths[b]; t < seqLen; ++t)
            {
                for (int32_t h = 0; h < nheads; ++h)
                {
                    dtHost[(b * seqLen + t) * nheads + h] = poison;
                    for (int32_t d = 0; d < dim; ++d)
                    {
                        xHost[((b * seqLen + t) * nheads + h) * dim + d] = poison;
                    }
                }
                for (int32_t g = 0; g < ngroups; ++g)
                {
                    for (int32_t ds = 0; ds < dstate; ++ds)
                    {
                        size_t const index = ((b * seqLen + t) * ngroups + g) * dstate + ds;
                        bHost[index] = poison;
                        cHost[index] = poison;
                    }
                }
            }
        }
    }

    // CPU reference (refState is the initial state; mutated to final by reference scan).
    std::vector<float> refState = stateHost;
    std::vector<half> refOut(outSize, __float2half(0.f));
    ssdPrefillReference(batch, seqLen, nheads, dim, dstate, ngroups, xHost, dtHost, aHost, bHost, cHost, dHost,
        dtBiasHost, true, refState, refOut, contextLengthsPtr);

    // SM80 wrapper now takes fp16 D / dt_bias / state (matches plugin contract).
    std::vector<half> dHostFp16(nheads), dtBiasHostFp16(nheads), stateHostFp16(stateSize);
    for (size_t i = 0; i < dHostFp16.size(); ++i)
        dHostFp16[i] = __float2half(dHost[i]);
    for (size_t i = 0; i < dtBiasHostFp16.size(); ++i)
        dtBiasHostFp16[i] = __float2half(dtBiasHost[i]);
    for (size_t i = 0; i < stateSize; ++i)
        stateHostFp16[i] = __float2half(stateHost[i]);

    // GPU: allocate and copy
    void *dX, *dDt, *dA, *dB, *dC, *dD, *dDtBias, *dState, *dOutput;
    CUDA_CHECK(cudaMalloc(&dX, xSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dDt, dtSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dA, nheads * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, bSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, bSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dD, nheads * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dDtBias, nheads * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dState, stateSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dOutput, outSize * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(dX, xHost.data(), xSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDt, dtHost.data(), dtSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA, aHost.data(), nheads * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, bHost.data(), bSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, cHost.data(), bSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dD, dHostFp16.data(), nheads * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDtBias, dtBiasHostFp16.data(), nheads * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dState, stateHostFp16.data(), stateSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dOutput, 0, outSize * sizeof(half)));

    // Allocate workspace for chunk scan intermediates
    size_t const wsSize = CuteDslSSDRunner::getWorkspaceSize(batch, seqLen, nheads, dim, dstate, ngroups);
    void* dWorkspace = nullptr;
    if (wsSize > 0)
    {
        CUDA_CHECK(cudaMalloc(&dWorkspace, wsSize));
        CUDA_CHECK(cudaMemset(dWorkspace, 0, wsSize));
    }

    // Run CuteDSL kernel
    SSDParams params{};
    params.x = dX;
    params.dt = dDt;
    params.A = dA;
    params.B = dB;
    params.C = dC;
    params.D = dD;
    params.dt_bias = dDtBias;
    params.z = nullptr;
    params.state = dState;
    params.output = dOutput;
    params.workspace = dWorkspace;
    params.batch = batch;
    params.seq_len = seqLen;
    params.nheads = nheads;
    params.dim = dim;
    params.dstate = dstate;
    params.ngroups = ngroups;
    params.smVersion = 80;
    params.dt_softplus = true;
    params.has_D = true;
    params.has_z = false;
    // Tests that exercise non-zero initial states need the runner to dispatch
    // to the Blackwell variant that reads init state at chunk 0.
    params.has_init_states = cfg.useNonzeroInitState;

    // Optional context_lengths device tensor
    void* dContextLengths = nullptr;
    if (contextLengthsPtr != nullptr)
    {
        CUDA_CHECK(cudaMalloc(&dContextLengths, batch * sizeof(int32_t)));
        CUDA_CHECK(
            cudaMemcpy(dContextLengths, contextLengthsPtr->data(), batch * sizeof(int32_t), cudaMemcpyHostToDevice));
        params.context_lengths = dContextLengths;
    }

    CuteDslSSDRunner runner;
    ASSERT_EQ(runner.run(params, nullptr), 0) << "CuteDslSSDRunner::run failed";
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back and compare
    std::vector<half> gpuOut(outSize);
    CUDA_CHECK(cudaMemcpy(gpuOut.data(), dOutput, outSize * sizeof(half), cudaMemcpyDeviceToHost));

    float maxDiff = 0.f;
    float refMax = 0.f;
    for (size_t i = 0; i < outSize; ++i)
    {
        float const a = __half2float(gpuOut[i]);
        float const b = __half2float(refOut[i]);
        maxDiff = std::max(maxDiff, std::abs(a - b));
        refMax = std::max(refMax, std::abs(b));
    }
    float const relErr = maxDiff / (refMax + 1e-8f);

    EXPECT_LT(relErr, 0.05f) << "Relative error " << relErr << " exceeds threshold. "
                             << "maxDiff=" << maxDiff << " refMax=" << refMax;

    // Cleanup
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dDt));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dD));
    CUDA_CHECK(cudaFree(dDtBias));
    CUDA_CHECK(cudaFree(dState));
    CUDA_CHECK(cudaFree(dOutput));
    if (dWorkspace)
    {
        CUDA_CHECK(cudaFree(dWorkspace));
    }
    if (dContextLengths)
    {
        CUDA_CHECK(cudaFree(dContextLengths));
    }
}

// SM80 test configurations — all D×N combos: {128,64} × {128,64}
INSTANTIATE_TEST_SUITE_P(SsdCuteDslSM80, SsdCuteDslTest,
    ::testing::Values(
        // batch, seqLen, nheads, dim, dstate, ngroups
        // D=128, N=128
        SsdCuteDslTestConfig{1, 128, 8, 128, 128, 1}, SsdCuteDslTestConfig{1, 256, 8, 128, 128, 1},
        SsdCuteDslTestConfig{1, 512, 8, 128, 128, 1}, SsdCuteDslTestConfig{1, 1024, 8, 128, 128, 1},
        SsdCuteDslTestConfig{4, 128, 8, 128, 128, 1}, SsdCuteDslTestConfig{4, 256, 8, 128, 128, 1},
        SsdCuteDslTestConfig{1, 256, 64, 128, 128, 1}, SsdCuteDslTestConfig{1, 256, 64, 128, 128, 8},
        // D=64, N=128
        SsdCuteDslTestConfig{1, 128, 8, 64, 128, 1}, SsdCuteDslTestConfig{1, 256, 8, 64, 128, 1},
        SsdCuteDslTestConfig{1, 512, 8, 64, 128, 1}, SsdCuteDslTestConfig{4, 128, 8, 64, 128, 1},
        // D=128, N=64
        SsdCuteDslTestConfig{1, 128, 8, 128, 64, 1}, SsdCuteDslTestConfig{1, 256, 8, 128, 64, 1},
        SsdCuteDslTestConfig{1, 512, 8, 128, 64, 1}, SsdCuteDslTestConfig{4, 128, 8, 128, 64, 1},
        // D=64, N=64
        SsdCuteDslTestConfig{1, 128, 8, 64, 64, 1}, SsdCuteDslTestConfig{1, 256, 8, 64, 64, 1},
        SsdCuteDslTestConfig{1, 512, 8, 64, 64, 1}, SsdCuteDslTestConfig{4, 128, 8, 64, 64, 1},
        // Varlen: explicit context_lengths
        SsdCuteDslTestConfig{1, 256, 8, 128, 128, 1, {256}},               // batch=1 cl==seqLen sanity
        SsdCuteDslTestConfig{1, 256, 8, 64, 128, 1, {200}},                // batch=1 partial
        SsdCuteDslTestConfig{4, 256, 8, 64, 128, 1, {100, 256, 200, 256}}, // batch>1 mixed
        SsdCuteDslTestConfig{4, 384, 8, 64, 128, 1, {100, 200, 250, 384}}, // mixed + partial last chunk
        SsdCuteDslTestConfig{2, 128, 8, 64, 64, 1, {32, 96}},              // small mixed
        SsdCuteDslTestConfig{8, 128, 8, 128, 128, 1, {1, 64, 128, 128, 100, 128, 50, 128}},
        // Nonzero initial state: exercises has_init_states=True kernel path on SM80. Reference
        // scan is seeded with the same state, so output must match. Validates chunked prefill /
        // continuous batching on Ampere fallback.
        SsdCuteDslTestConfig{1, 256, 8, 128, 128, 1, {}, true},                  // D=128 N=128, init nonzero
        SsdCuteDslTestConfig{1, 512, 8, 64, 128, 1, {}, true},                   // multi-chunk, init nonzero
        SsdCuteDslTestConfig{4, 256, 8, 64, 128, 1, {100, 256, 200, 256}, true}, // mixed varlen + init
        SsdCuteDslTestConfig{2, 128, 8, 64, 64, 1, {32, 96}, true}),             // small mixed + init
    [](testing::TestParamInfo<SsdCuteDslTestConfig> const& info) {
        auto const& c = info.param;
        std::string name = "b" + std::to_string(c.batch) + "_s" + std::to_string(c.seqLen) + "_h"
            + std::to_string(c.nheads) + "_d" + std::to_string(c.dim) + "_ds" + std::to_string(c.dstate) + "_g"
            + std::to_string(c.ngroups);
        if (!c.contextLengths.empty())
        {
            name += "_vl";
            for (int32_t cl : c.contextLengths)
                name += std::to_string(cl) + "x";
            name.pop_back();
        }
        if (c.useNonzeroInitState)
            name += "_initstate";
        if (c.poisonPadding)
            name += "_poison";
        return name;
    });

TEST(SsdCuteDslVarlenMetadata, HandlesUnalignedPaddedRows)
{
    int32_t constexpr batch = 3;
    int32_t constexpr seqLen = 462;
    int32_t constexpr chunkSize = 128;
    int32_t constexpr chunksPerSeqUpper = 5;
    int32_t constexpr totalSlots = batch * chunksPerSeqUpper;

    std::vector<int32_t> const contextLengths{78, 175, 462};
    std::vector<int32_t> const expectedSeqChunkCumsum{0, 1, 3, 7};
    std::vector<int32_t> const expectedChunkIndices{0, 3, 4, 7, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1, -1};
    std::vector<int32_t> const expectedChunkOffsets{0, 78, 0, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    int32_t* dSeqIdx = nullptr;
    int32_t* dChunkIndices = nullptr;
    int32_t* dChunkOffsets = nullptr;
    int32_t* dSeqChunkCumsum = nullptr;
    int32_t* dContextLengths = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSeqIdx), batch * seqLen * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dChunkIndices), totalSlots * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dChunkOffsets), totalSlots * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSeqChunkCumsum), (batch + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dContextLengths), batch * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(dContextLengths, contextLengths.data(), batch * sizeof(int32_t), cudaMemcpyHostToDevice));

    mamba::buildSSDVarlenMetadata(
        dSeqIdx, dChunkIndices, dChunkOffsets, dSeqChunkCumsum, dContextLengths, batch, seqLen, chunkSize, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> seqChunkCumsum(batch + 1);
    std::vector<int32_t> chunkIndices(totalSlots);
    std::vector<int32_t> chunkOffsets(totalSlots);
    std::vector<int32_t> seqIdx(batch * seqLen);
    CUDA_CHECK(
        cudaMemcpy(seqChunkCumsum.data(), dSeqChunkCumsum, (batch + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(chunkIndices.data(), dChunkIndices, totalSlots * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(chunkOffsets.data(), dChunkOffsets, totalSlots * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(seqIdx.data(), dSeqIdx, batch * seqLen * sizeof(int32_t), cudaMemcpyDeviceToHost));

    EXPECT_EQ(seqChunkCumsum, expectedSeqChunkCumsum);
    EXPECT_EQ(chunkIndices, expectedChunkIndices);
    EXPECT_EQ(chunkOffsets, expectedChunkOffsets);
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t t = 0; t < seqLen; ++t)
        {
            int32_t const expected = (t < contextLengths[b]) ? b : -1;
            EXPECT_EQ(seqIdx[b * seqLen + t], expected) << "b=" << b << " t=" << t;
        }
    }

    CUDA_CHECK(cudaFree(dSeqIdx));
    CUDA_CHECK(cudaFree(dChunkIndices));
    CUDA_CHECK(cudaFree(dChunkOffsets));
    CUDA_CHECK(cudaFree(dSeqChunkCumsum));
    CUDA_CHECK(cudaFree(dContextLengths));
}

#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED

char const* guardedSsdInputName(GuardedSsdInput guardedInput)
{
    switch (guardedInput)
    {
    case GuardedSsdInput::kX: return "X";
    case GuardedSsdInput::kDt: return "Dt";
    case GuardedSsdInput::kB: return "B";
    case GuardedSsdInput::kC: return "C";
    case GuardedSsdInput::kAll: return "All";
    }
    return "Unknown";
}

rt::Tensor makeGpuTensor(int64_t elements, DataType dataType, std::string const& name)
{
    return rt::Tensor(rt::Coords{elements}, rt::DeviceType::kGPU, dataType, name);
}

// =============================================================================
// Blackwell test fixture (SM100+, dim=64, dstate=128)
// =============================================================================
// The Blackwell kernel is a single persistent kernel that takes pre-computed
// cumsum_delta and dt_processed as inputs (unlike the SM80 kernel which does
// cumsum internally). This test verifies end-to-end correctness via
// CuteDslSSDRunner::run() with smVersion=100.

class SsdCuteDslBlackwellTest : public ::testing::TestWithParam<SsdCuteDslTestConfig>
{
protected:
    void SetUp() override
    {
        // Check that GPU actually supports SM100+
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        if (prop.major < 10)
        {
            GTEST_SKIP() << "Blackwell tests require SM100+ GPU (got SM" << prop.major << prop.minor << ")";
        }
    }
};

TEST_P(SsdCuteDslBlackwellTest, CorrectnessVsSerialReference)
{
    auto const& cfg = GetParam();
    int32_t const batch = cfg.batch;
    int32_t const seqLen = cfg.seqLen;
    int32_t const nheads = cfg.nheads;
    int32_t const dim = cfg.dim;
    int32_t const dstate = cfg.dstate;
    int32_t const ngroups = cfg.ngroups;

    int32_t dispatchSmVersion = 100;
    if (dim == 80)
    {
        int device{};
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        dispatchSmVersion = prop.major * 10 + prop.minor;
    }
    if (!CuteDslSSDRunner::canImplement(dim, dstate, dispatchSmVersion))
    {
        GTEST_SKIP() << "CuteDslSSDRunner cannot implement dim=" << dim << " dstate=" << dstate << " for SM"
                     << dispatchSmVersion;
    }

    // Both Blackwell native (dim=64) and SM80 fallback (dim=128) wrappers take fp16
    // D/state/dt_bias (matches plugin kIN_D_IDX / kIN_DT_BIAS_IDX / kIN_STATE_IDX = kHALF).
    // Allocate host data
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.f, 0.5f);
    std::uniform_real_distribution<float> uniform(0.1f, 0.6f);

    size_t const xSize = batch * seqLen * nheads * dim;
    size_t const dtSize = batch * seqLen * nheads;
    size_t const bSize = batch * seqLen * ngroups * dstate;
    size_t const stateSize = batch * nheads * dim * dstate;
    size_t const outSize = xSize;

    std::vector<half> xHost(xSize), dtHost(dtSize), bHost(bSize), cHost(bSize);
    std::vector<float> aHost(nheads), dHost(nheads), dtBiasHost(nheads);
    std::vector<float> stateHost(stateSize, 0.f);
    std::vector<half> outHost(outSize, __float2half(0.f));

    for (auto& v : xHost)
        v = __float2half(normal(rng));
    for (auto& v : dtHost)
        v = __float2half(uniform(rng));
    for (auto& v : bHost)
        v = __float2half(normal(rng));
    for (auto& v : cHost)
        v = __float2half(normal(rng));
    for (auto& v : aHost)
        v = -(uniform(rng) + 0.5f);
    for (auto& v : dHost)
        v = normal(rng) * 0.1f;
    for (auto& v : dtBiasHost)
        v = normal(rng) * 0.1f;
    // Initial SSM state: zeros by default, deterministic small values when test exercises has_init_states.
    if (cfg.useNonzeroInitState)
    {
        std::normal_distribution<float> stateNormal(0.f, 0.1f);
        for (auto& v : stateHost)
            v = stateNormal(rng);
    }

    // Optional per-batch context_lengths
    std::vector<int32_t> const* contextLengthsPtr = cfg.contextLengths.empty() ? nullptr : &cfg.contextLengths;
    if (cfg.poisonPadding)
    {
        ASSERT_EQ(static_cast<int32_t>(cfg.contextLengths.size()), batch);
        half const poison = __float2half(8.f);
        for (int32_t b = 0; b < batch; ++b)
        {
            for (int32_t t = cfg.contextLengths[b]; t < seqLen; ++t)
            {
                for (int32_t h = 0; h < nheads; ++h)
                {
                    dtHost[(b * seqLen + t) * nheads + h] = poison;
                    for (int32_t d = 0; d < dim; ++d)
                    {
                        xHost[((b * seqLen + t) * nheads + h) * dim + d] = poison;
                    }
                }
                for (int32_t g = 0; g < ngroups; ++g)
                {
                    for (int32_t ds = 0; ds < dstate; ++ds)
                    {
                        size_t const index = ((b * seqLen + t) * ngroups + g) * dstate + ds;
                        bHost[index] = poison;
                        cHost[index] = poison;
                    }
                }
            }
        }
    }

    // CPU reference (refState is the initial state; mutated to final by reference scan).
    std::vector<float> refState = stateHost;
    std::vector<half> refOut(outSize, __float2half(0.f));
    ssdPrefillReference(batch, seqLen, nheads, dim, dstate, ngroups, xHost, dtHost, aHost, bHost, cHost, dHost,
        dtBiasHost, true, refState, refOut, contextLengthsPtr);

    std::vector<half> dHostFp16(nheads), dtBiasHostFp16(nheads), stateHostFp16(stateSize);
    for (size_t i = 0; i < dHostFp16.size(); ++i)
        dHostFp16[i] = __float2half(dHost[i]);
    for (size_t i = 0; i < dtBiasHostFp16.size(); ++i)
        dtBiasHostFp16[i] = __float2half(dtBiasHost[i]);
    for (size_t i = 0; i < stateSize; ++i)
        stateHostFp16[i] = __float2half(stateHost[i]);

    // GPU: allocate and copy
    void *dX, *dDt, *dA, *dB, *dC, *dD, *dDtBias, *dState, *dOutput;
    CUDA_CHECK(cudaMalloc(&dX, xSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dDt, dtSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dA, nheads * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, bSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, bSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dD, nheads * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dDtBias, nheads * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dState, stateSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dOutput, outSize * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(dX, xHost.data(), xSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDt, dtHost.data(), dtSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA, aHost.data(), nheads * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, bHost.data(), bSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, cHost.data(), bSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dD, dHostFp16.data(), nheads * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDtBias, dtBiasHostFp16.data(), nheads * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dState, stateHostFp16.data(), stateSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dOutput, 0, outSize * sizeof(half)));

    size_t const wsSize = CuteDslSSDRunner::getWorkspaceSize(batch, seqLen, nheads, dim, dstate, ngroups);
    void* dWorkspace = nullptr;
    if (wsSize > 0)
    {
        CUDA_CHECK(cudaMalloc(&dWorkspace, wsSize));
        CUDA_CHECK(cudaMemset(dWorkspace, 0, wsSize));
    }

    SSDParams params{};
    params.x = dX;
    params.dt = dDt;
    params.A = dA;
    params.B = dB;
    params.C = dC;
    params.D = dD;
    params.dt_bias = dDtBias;
    params.z = nullptr;
    params.state = dState;
    params.output = dOutput;
    params.workspace = dWorkspace;
    params.batch = batch;
    params.seq_len = seqLen;
    params.nheads = nheads;
    params.dim = dim;
    params.dstate = dstate;
    params.ngroups = ngroups;
    params.smVersion = dispatchSmVersion; // Force Blackwell path for existing variants; use the exact SM for D80.
    params.dt_softplus = true;
    params.has_D = true;
    params.has_z = false;
    // Tests with non-zero initial state dispatch to the init-states variant.
    params.has_init_states = cfg.useNonzeroInitState;

    // Optional context_lengths device tensor
    void* dContextLengths = nullptr;
    if (contextLengthsPtr != nullptr)
    {
        CUDA_CHECK(cudaMalloc(&dContextLengths, batch * sizeof(int32_t)));
        CUDA_CHECK(
            cudaMemcpy(dContextLengths, contextLengthsPtr->data(), batch * sizeof(int32_t), cudaMemcpyHostToDevice));
        params.context_lengths = dContextLengths;
    }

    CuteDslSSDRunner runner;
    ASSERT_EQ(runner.run(params, nullptr), 0) << "CuteDslSSDRunner::run (Blackwell) failed";
    CUDA_CHECK(cudaDeviceSynchronize());

    // Kernel writes y directly to params.output [B, S, H, D] (no transpose adapter).
    std::vector<half> gpuOut(outSize);
    CUDA_CHECK(cudaMemcpy(gpuOut.data(), dOutput, outSize * sizeof(half), cudaMemcpyDeviceToHost));

    float maxDiff = 0.f;
    float refMax = 0.f;
    for (size_t i = 0; i < outSize; ++i)
    {
        float const a = __half2float(gpuOut[i]);
        float const b = __half2float(refOut[i]);
        maxDiff = std::max(maxDiff, std::abs(a - b));
        refMax = std::max(refMax, std::abs(b));
    }
    float const relErr = maxDiff / (refMax + 1e-8f);

    EXPECT_LT(relErr, 0.05f) << "Relative error " << relErr << " exceeds threshold. "
                             << "maxDiff=" << maxDiff << " refMax=" << refMax;

    std::vector<half> gpuState(stateSize);
    CUDA_CHECK(cudaMemcpy(gpuState.data(), dState, stateSize * sizeof(half), cudaMemcpyDeviceToHost));

    float maxStateDiff = 0.f;
    float refStateMax = 0.f;
    for (size_t i = 0; i < stateSize; ++i)
    {
        float const a = __half2float(gpuState[i]);
        float const b = refState[i];
        maxStateDiff = std::max(maxStateDiff, std::abs(a - b));
        refStateMax = std::max(refStateMax, std::abs(b));
    }
    float const stateRelErr = maxStateDiff / (refStateMax + 1e-8f);

    EXPECT_LT(stateRelErr, 0.05f) << "Final state relative error " << stateRelErr << " exceeds threshold. "
                                  << "maxDiff=" << maxStateDiff << " refMax=" << refStateMax;

    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dDt));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dD));
    CUDA_CHECK(cudaFree(dDtBias));
    CUDA_CHECK(cudaFree(dState));
    CUDA_CHECK(cudaFree(dOutput));
    if (dWorkspace)
    {
        CUDA_CHECK(cudaFree(dWorkspace));
    }
    if (dContextLengths)
    {
        CUDA_CHECK(cudaFree(dContextLengths));
    }
}

int runSsdTmaBoundsCase(GuardedSsdInput guardedInput)
{
    try
    {
        int device{};
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        if (prop.major < 10)
        {
            std::cerr << "Blackwell tests require SM100+ GPU (got SM" << prop.major << prop.minor << ")\n";
            return 1;
        }

        CUDA_DRIVER_CHECK(cuInit(0));
        CUdevice cuDevice{};
        CUDA_DRIVER_CHECK(cuDeviceGet(&cuDevice, device));
        int vmmSupported{};
        CUDA_DRIVER_CHECK(
            cuDeviceGetAttribute(&vmmSupported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cuDevice));
        if (vmmSupported == 0)
        {
            std::cerr << "CUDA VMM is required for the guard-page allocation\n";
            return 1;
        }

        int32_t constexpr batch = 1;
        int32_t constexpr seqLen = 129;
        int32_t constexpr nheads = 8;
        int32_t constexpr dim = 64;
        int32_t constexpr dstate = 128;
        int32_t constexpr ngroups = 1;
        if (!CuteDslSSDRunner::canImplement(dim, dstate, 100))
        {
            std::cerr << "CuteDslSSDRunner cannot implement dim=" << dim << " dstate=" << dstate << " for SM100\n";
            return 1;
        }

        size_t constexpr xSize = static_cast<size_t>(batch) * seqLen * nheads * dim;
        size_t constexpr dtSize = static_cast<size_t>(batch) * seqLen * nheads;
        size_t constexpr bSize = static_cast<size_t>(batch) * seqLen * ngroups * dstate;
        size_t constexpr stateSize = static_cast<size_t>(batch) * nheads * dim * dstate;

        std::mt19937 rng(2026);
        std::normal_distribution<float> normal(0.f, 0.5f);
        std::uniform_real_distribution<float> uniform(0.1f, 0.6f);

        std::vector<half> xHost(xSize), dtHost(dtSize), bHost(bSize), cHost(bSize);
        std::vector<float> aHost(nheads);
        std::vector<half> dHost(nheads), dtBiasHost(nheads), stateHost(stateSize, __float2half(0.f));
        for (auto& v : xHost)
            v = __float2half(normal(rng));
        for (auto& v : dtHost)
            v = __float2half(uniform(rng));
        for (auto& v : bHost)
            v = __float2half(normal(rng));
        for (auto& v : cHost)
            v = __float2half(normal(rng));
        for (auto& v : aHost)
            v = -(uniform(rng) + 0.5f);
        for (auto& v : dHost)
            v = __float2half(normal(rng) * 0.1f);
        for (auto& v : dtBiasHost)
            v = __float2half(normal(rng) * 0.1f);

        bool const guardX = guardedInput == GuardedSsdInput::kX || guardedInput == GuardedSsdInput::kAll;
        bool const guardDt = guardedInput == GuardedSsdInput::kDt || guardedInput == GuardedSsdInput::kAll;
        bool const guardB = guardedInput == GuardedSsdInput::kB || guardedInput == GuardedSsdInput::kAll;
        bool const guardC = guardedInput == GuardedSsdInput::kC || guardedInput == GuardedSsdInput::kAll;

        std::unique_ptr<GuardedDeviceBuffer> xGuard;
        std::unique_ptr<GuardedDeviceBuffer> dtGuard;
        std::unique_ptr<GuardedDeviceBuffer> bGuard;
        std::unique_ptr<GuardedDeviceBuffer> cGuard;
        std::optional<rt::Tensor> xTensor;
        std::optional<rt::Tensor> dtTensor;
        std::optional<rt::Tensor> bTensor;
        std::optional<rt::Tensor> cTensor;
        void *dX{}, *dDt{}, *dA{}, *dB{}, *dC{}, *dD{}, *dDtBias{}, *dState{}, *dOutput{};
        if (guardX)
        {
            xGuard = std::make_unique<GuardedDeviceBuffer>(xSize * sizeof(half));
            dX = xGuard->data();
        }
        else
        {
            xTensor.emplace(makeGpuTensor(static_cast<int64_t>(xSize), DataType::kHALF, "x"));
            dX = xTensor->rawPointer();
        }
        if (guardDt)
        {
            dtGuard = std::make_unique<GuardedDeviceBuffer>(dtSize * sizeof(half));
            dDt = dtGuard->data();
        }
        else
        {
            dtTensor.emplace(makeGpuTensor(static_cast<int64_t>(dtSize), DataType::kHALF, "dt"));
            dDt = dtTensor->rawPointer();
        }
        if (guardB)
        {
            bGuard = std::make_unique<GuardedDeviceBuffer>(bSize * sizeof(half));
            dB = bGuard->data();
        }
        else
        {
            bTensor.emplace(makeGpuTensor(static_cast<int64_t>(bSize), DataType::kHALF, "B"));
            dB = bTensor->rawPointer();
        }
        if (guardC)
        {
            cGuard = std::make_unique<GuardedDeviceBuffer>(bSize * sizeof(half));
            dC = cGuard->data();
        }
        else
        {
            cTensor.emplace(makeGpuTensor(static_cast<int64_t>(bSize), DataType::kHALF, "C"));
            dC = cTensor->rawPointer();
        }
        rt::Tensor aTensor = makeGpuTensor(nheads, DataType::kFLOAT, "A");
        rt::Tensor dTensor = makeGpuTensor(nheads, DataType::kHALF, "D");
        rt::Tensor dtBiasTensor = makeGpuTensor(nheads, DataType::kHALF, "dtBias");
        rt::Tensor stateTensor = makeGpuTensor(static_cast<int64_t>(stateSize), DataType::kHALF, "state");
        rt::Tensor outputTensor = makeGpuTensor(static_cast<int64_t>(xSize), DataType::kHALF, "output");
        dA = aTensor.rawPointer();
        dD = dTensor.rawPointer();
        dDtBias = dtBiasTensor.rawPointer();
        dState = stateTensor.rawPointer();
        dOutput = outputTensor.rawPointer();

        CUDA_CHECK(cudaMemcpy(dX, xHost.data(), xSize * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dDt, dtHost.data(), dtSize * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dA, aHost.data(), nheads * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, bHost.data(), bSize * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dC, cHost.data(), bSize * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dD, dHost.data(), nheads * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dDtBias, dtBiasHost.data(), nheads * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dState, stateHost.data(), stateSize * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dOutput, 0, xSize * sizeof(half)));

        size_t const wsSize = CuteDslSSDRunner::getWorkspaceSize(batch, seqLen, nheads, dim, dstate, ngroups);
        std::optional<rt::Tensor> workspaceTensor;
        void* dWorkspace{};
        if (wsSize > 0)
        {
            workspaceTensor.emplace(makeGpuTensor(static_cast<int64_t>(wsSize), DataType::kUINT8, "workspace"));
            dWorkspace = workspaceTensor->rawPointer();
            CUDA_CHECK(cudaMemset(dWorkspace, 0, wsSize));
        }

        SSDParams params{};
        params.x = dX;
        params.dt = dDt;
        params.A = dA;
        params.B = dB;
        params.C = dC;
        params.D = dD;
        params.dt_bias = dDtBias;
        params.z = nullptr;
        params.state = dState;
        params.output = dOutput;
        params.workspace = dWorkspace;
        params.batch = batch;
        params.seq_len = seqLen;
        params.nheads = nheads;
        params.dim = dim;
        params.dstate = dstate;
        params.ngroups = ngroups;
        params.smVersion = 100;
        params.dt_softplus = true;
        params.has_D = true;
        params.has_z = false;
        params.has_init_states = false;

        CuteDslSSDRunner runner;
        int const runStatus = runner.run(params, nullptr);
        cudaError_t const syncStatus = cudaDeviceSynchronize();

        if (runStatus != 0)
        {
            std::cerr << "CuteDslSSDRunner::run (Blackwell) failed while guarding " << guardedSsdInputName(guardedInput)
                      << "\n";
            return 1;
        }
        if (syncStatus != cudaSuccess)
        {
            std::cerr << "Final partial TMA tile read past a logical tensor bound while guarding "
                      << guardedSsdInputName(guardedInput) << ": " << cudaGetErrorString(syncStatus) << "\n";
            return 1;
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        std::cerr << "SSD TMA bounds case failed while guarding " << guardedSsdInputName(guardedInput) << ": "
                  << e.what() << "\n";
        return 1;
    }
}

class SsdCuteDslBlackwellTmaBounds : public ::testing::TestWithParam<GuardedSsdInput>
{
};

TEST_P(SsdCuteDslBlackwellTmaBounds, FinalPartialChunkDoesNotReadPastTensor)
{
    GuardedSsdInput const guardedInput = GetParam();
    int device{};
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 10)
    {
        GTEST_SKIP() << "Blackwell tests require SM100+ GPU (got SM" << prop.major << prop.minor << ")";
    }

    CUDA_DRIVER_CHECK(cuInit(0));
    CUdevice cuDevice{};
    CUDA_DRIVER_CHECK(cuDeviceGet(&cuDevice, device));
    int vmmSupported{};
    CUDA_DRIVER_CHECK(
        cuDeviceGetAttribute(&vmmSupported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cuDevice));
    if (vmmSupported == 0)
    {
        GTEST_SKIP() << "CUDA VMM is required for the guard-page allocation";
    }

    int32_t constexpr dim = 64;
    int32_t constexpr dstate = 128;
    if (!CuteDslSSDRunner::canImplement(dim, dstate, 100))
    {
        GTEST_SKIP() << "CuteDslSSDRunner cannot implement dim=" << dim << " dstate=" << dstate << " for SM100";
    }

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(
        {
            int const exitCode = runSsdTmaBoundsCase(guardedInput);
            std::_Exit(exitCode);
        },
        ::testing::ExitedWithCode(0), "");
}

INSTANTIATE_TEST_SUITE_P(GuardedInput, SsdCuteDslBlackwellTmaBounds,
    ::testing::Values(
        GuardedSsdInput::kX, GuardedSsdInput::kDt, GuardedSsdInput::kB, GuardedSsdInput::kC, GuardedSsdInput::kAll),
    [](testing::TestParamInfo<GuardedSsdInput> const& info) { return guardedSsdInputName(info.param); });

int runSsdD80BoundsCase()
{
    try
    {
        int device{};
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        int32_t const smVersion = prop.major * 10 + prop.minor;

        int32_t constexpr batch = 1;
        int32_t constexpr seqLen = 129;
        int32_t constexpr nheads = 8;
        int32_t constexpr dim = 80;
        int32_t constexpr dstate = 128;
        int32_t constexpr ngroups = 1;
        if (!CuteDslSSDRunner::canImplement(dim, dstate, smVersion))
        {
            std::cerr << "CuteDslSSDRunner cannot implement D80/N128 for SM" << smVersion << "\n";
            return 1;
        }

        size_t constexpr xSize = static_cast<size_t>(batch) * seqLen * nheads * dim;
        size_t constexpr dtSize = static_cast<size_t>(batch) * seqLen * nheads;
        size_t constexpr bSize = static_cast<size_t>(batch) * seqLen * ngroups * dstate;
        size_t constexpr stateSize = static_cast<size_t>(batch) * nheads * dim * dstate;
        size_t const workspaceSize = CuteDslSSDRunner::getWorkspaceSize(batch, seqLen, nheads, dim, dstate, ngroups);

        GuardedDeviceBuffer xBuffer(xSize * sizeof(half));
        GuardedDeviceBuffer stateBuffer(stateSize * sizeof(half));
        GuardedDeviceBuffer outputBuffer(xSize * sizeof(half));
        GuardedDeviceBuffer workspaceBuffer(workspaceSize);

        rt::Tensor dtTensor = makeGpuTensor(static_cast<int64_t>(dtSize), DataType::kHALF, "dt");
        rt::Tensor aTensor = makeGpuTensor(nheads, DataType::kFLOAT, "A");
        rt::Tensor bTensor = makeGpuTensor(static_cast<int64_t>(bSize), DataType::kHALF, "B");
        rt::Tensor cTensor = makeGpuTensor(static_cast<int64_t>(bSize), DataType::kHALF, "C");
        rt::Tensor dTensor = makeGpuTensor(nheads, DataType::kHALF, "D");
        rt::Tensor dtBiasTensor = makeGpuTensor(nheads, DataType::kHALF, "dtBias");

        CUDA_CHECK(cudaMemset(xBuffer.data(), 0, xSize * sizeof(half)));
        CUDA_CHECK(cudaMemset(dtTensor.rawPointer(), 0, dtSize * sizeof(half)));
        CUDA_CHECK(cudaMemset(aTensor.rawPointer(), 0, nheads * sizeof(float)));
        CUDA_CHECK(cudaMemset(bTensor.rawPointer(), 0, bSize * sizeof(half)));
        CUDA_CHECK(cudaMemset(cTensor.rawPointer(), 0, bSize * sizeof(half)));
        CUDA_CHECK(cudaMemset(dTensor.rawPointer(), 0, nheads * sizeof(half)));
        CUDA_CHECK(cudaMemset(dtBiasTensor.rawPointer(), 0, nheads * sizeof(half)));
        CUDA_CHECK(cudaMemset(stateBuffer.data(), 0, stateSize * sizeof(half)));
        CUDA_CHECK(cudaMemset(outputBuffer.data(), 0, xSize * sizeof(half)));
        CUDA_CHECK(cudaMemset(workspaceBuffer.data(), 0, workspaceSize));

        SSDParams params{};
        params.x = xBuffer.data();
        params.dt = dtTensor.rawPointer();
        params.A = aTensor.rawPointer();
        params.B = bTensor.rawPointer();
        params.C = cTensor.rawPointer();
        params.D = dTensor.rawPointer();
        params.dt_bias = dtBiasTensor.rawPointer();
        params.state = stateBuffer.data();
        params.output = outputBuffer.data();
        params.workspace = workspaceBuffer.data();
        params.batch = batch;
        params.seq_len = seqLen;
        params.nheads = nheads;
        params.dim = dim;
        params.dstate = dstate;
        params.ngroups = ngroups;
        params.smVersion = smVersion;
        params.dt_softplus = true;
        params.has_D = true;
        params.has_init_states = true;

        CuteDslSSDRunner runner;
        int const runStatus = runner.run(params, nullptr);
        cudaError_t const syncStatus = cudaDeviceSynchronize();
        if (runStatus != 0)
        {
            std::cerr << "CuteDslSSDRunner::run failed for guarded D80\n";
            return 1;
        }
        if (syncStatus != cudaSuccess)
        {
            std::cerr << "D80 kernel accessed beyond an exact tensor or workspace: " << cudaGetErrorString(syncStatus)
                      << "\n";
            return 1;
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        std::cerr << "D80 bounds case failed: " << e.what() << "\n";
        return 1;
    }
}

TEST(SsdCuteDslBlackwellD80Bounds, ExactBuffersDoNotOverflow)
{
    int device{};
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int32_t const smVersion = prop.major * 10 + prop.minor;
    if (!CuteDslSSDRunner::canImplement(80, 128, smVersion))
    {
        GTEST_SKIP() << "D80 kernel requires SM100, SM101, or SM110";
    }

    CUDA_DRIVER_CHECK(cuInit(0));
    CUdevice cuDevice{};
    CUDA_DRIVER_CHECK(cuDeviceGet(&cuDevice, device));
    int vmmSupported{};
    CUDA_DRIVER_CHECK(
        cuDeviceGetAttribute(&vmmSupported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cuDevice));
    if (vmmSupported == 0)
    {
        GTEST_SKIP() << "CUDA VMM is required for the guard-page allocation";
    }

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(
        {
            int const exitCode = runSsdD80BoundsCase();
            std::_Exit(exitCode);
        },
        ::testing::ExitedWithCode(0), "");
}

// =============================================================================
// Chunked prefill simulation -- exercises has_init_states correctness end-to-end.
// Splits a single seq of length 2*chunkLen into two consecutive runner calls; the
// second call's initial state is the first call's final output state. The combined
// output must match a single-shot run of the full sequence.
// =============================================================================
TEST(SsdCuteDslBlackwellChunkedPrefill, StateCarriesAcrossCalls)
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 10)
    {
        GTEST_SKIP() << "Blackwell tests require SM100+ GPU";
    }
    int32_t const batch = 1;
    int32_t const chunkLen = 256;
    int32_t const totalLen = chunkLen * 2; // 512 -- exercises multi-chunk per call
    int32_t const nheads = 8, dim = 64, dstate = 128, ngroups = 1;
    if (!CuteDslSSDRunner::canImplement(dim, dstate, 100))
    {
        GTEST_SKIP() << "CuteDslSSDRunner cannot implement dim=" << dim << " dstate=" << dstate;
    }

    std::mt19937 rng(123);
    std::normal_distribution<float> normal(0.f, 0.5f);
    std::uniform_real_distribution<float> uniform(0.1f, 0.6f);

    size_t const xSize = batch * totalLen * nheads * dim;
    size_t const dtSize = batch * totalLen * nheads;
    size_t const bSize = batch * totalLen * ngroups * dstate;
    size_t const stateSize = batch * nheads * dim * dstate;

    std::vector<half> xFull(xSize), dtFull(dtSize), bFull(bSize), cFull(bSize);
    std::vector<float> aHost(nheads), dHost(nheads), dtBiasHost(nheads);
    std::vector<float> stateZero(stateSize, 0.f);

    for (auto& v : xFull)
        v = __float2half(normal(rng));
    for (auto& v : dtFull)
        v = __float2half(uniform(rng));
    for (auto& v : bFull)
        v = __float2half(normal(rng));
    for (auto& v : cFull)
        v = __float2half(normal(rng));
    for (auto& v : aHost)
        v = -(uniform(rng) + 0.5f);
    for (auto& v : dHost)
        v = normal(rng) * 0.1f;
    for (auto& v : dtBiasHost)
        v = normal(rng) * 0.1f;

    // Blackwell wrapper takes fp16 D / dt_bias / state (matches plugin contract).
    std::vector<half> dHostFp16(nheads), dtBiasHostFp16(nheads);
    for (size_t i = 0; i < dHostFp16.size(); ++i)
        dHostFp16[i] = __float2half(dHost[i]);
    for (size_t i = 0; i < dtBiasHostFp16.size(); ++i)
        dtBiasHostFp16[i] = __float2half(dtBiasHost[i]);

    auto runOnce
        = [&](int32_t seqLen, std::vector<half> const& x, std::vector<half> const& dt, std::vector<half> const& Bv,
              std::vector<half> const& Cv, std::vector<float>& stateInOut, std::vector<half>& outBuf) {
              size_t const n_x = batch * seqLen * nheads * dim;
              size_t const n_dt = batch * seqLen * nheads;
              size_t const n_bc = batch * seqLen * ngroups * dstate;

              // Convert state float->half for input; convert back to float on output.
              std::vector<half> stateFp16In(stateSize);
              for (size_t i = 0; i < stateSize; ++i)
                  stateFp16In[i] = __float2half(stateInOut[i]);

              void *dX, *dDt, *dA, *dB, *dC, *dD, *dDtBias, *dState, *dOutBuf;
              CUDA_CHECK(cudaMalloc(&dX, n_x * sizeof(half)));
              CUDA_CHECK(cudaMalloc(&dDt, n_dt * sizeof(half)));
              CUDA_CHECK(cudaMalloc(&dA, nheads * sizeof(float)));
              CUDA_CHECK(cudaMalloc(&dB, n_bc * sizeof(half)));
              CUDA_CHECK(cudaMalloc(&dC, n_bc * sizeof(half)));
              CUDA_CHECK(cudaMalloc(&dD, nheads * sizeof(half)));
              CUDA_CHECK(cudaMalloc(&dDtBias, nheads * sizeof(half)));
              CUDA_CHECK(cudaMalloc(&dState, stateSize * sizeof(half)));
              CUDA_CHECK(cudaMalloc(&dOutBuf, n_x * sizeof(half)));
              CUDA_CHECK(cudaMemcpy(dX, x.data(), n_x * sizeof(half), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemcpy(dDt, dt.data(), n_dt * sizeof(half), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemcpy(dA, aHost.data(), nheads * sizeof(float), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemcpy(dB, Bv.data(), n_bc * sizeof(half), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemcpy(dC, Cv.data(), n_bc * sizeof(half), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemcpy(dD, dHostFp16.data(), nheads * sizeof(half), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemcpy(dDtBias, dtBiasHostFp16.data(), nheads * sizeof(half), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemcpy(dState, stateFp16In.data(), stateSize * sizeof(half), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemset(dOutBuf, 0, n_x * sizeof(half)));

              size_t const wsSize = CuteDslSSDRunner::getWorkspaceSize(batch, seqLen, nheads, dim, dstate, ngroups);
              void* dWs = nullptr;
              if (wsSize > 0)
              {
                  CUDA_CHECK(cudaMalloc(&dWs, wsSize));
                  CUDA_CHECK(cudaMemset(dWs, 0, wsSize));
              }

              SSDParams params{};
              params.x = dX;
              params.dt = dDt;
              params.A = dA;
              params.B = dB;
              params.C = dC;
              params.D = dD;
              params.dt_bias = dDtBias;
              params.z = nullptr;
              params.state = dState;
              params.output = dOutBuf;
              params.workspace = dWs;
              params.batch = batch;
              params.seq_len = seqLen;
              params.nheads = nheads;
              params.dim = dim;
              params.dstate = dstate;
              params.ngroups = ngroups;
              params.smVersion = 100;
              params.dt_softplus = true;
              params.has_D = true;
              params.has_z = false;
              // ChunkedPrefill exercises the carry-over path: each call reads its
              // initial state from the prior call's output state.
              params.has_init_states = true;

              CuteDslSSDRunner runner;
              ASSERT_EQ(runner.run(params, nullptr), 0) << "runner.run failed";
              CUDA_CHECK(cudaDeviceSynchronize());

              outBuf.resize(n_x);
              CUDA_CHECK(cudaMemcpy(outBuf.data(), dOutBuf, n_x * sizeof(half), cudaMemcpyDeviceToHost));
              std::vector<half> stateFp16Out(stateSize);
              CUDA_CHECK(cudaMemcpy(stateFp16Out.data(), dState, stateSize * sizeof(half), cudaMemcpyDeviceToHost));
              for (size_t i = 0; i < stateSize; ++i)
                  stateInOut[i] = __half2float(stateFp16Out[i]);

              CUDA_CHECK(cudaFree(dX));
              CUDA_CHECK(cudaFree(dDt));
              CUDA_CHECK(cudaFree(dA));
              CUDA_CHECK(cudaFree(dB));
              CUDA_CHECK(cudaFree(dC));
              CUDA_CHECK(cudaFree(dD));
              CUDA_CHECK(cudaFree(dDtBias));
              CUDA_CHECK(cudaFree(dState));
              CUDA_CHECK(cudaFree(dOutBuf));
              if (dWs)
                  CUDA_CHECK(cudaFree(dWs));
          };

    // 1. One-shot reference: full seq in single runner call.
    std::vector<float> stateOneShot = stateZero;
    std::vector<half> outOneShot;
    runOnce(totalLen, xFull, dtFull, bFull, cFull, stateOneShot, outOneShot);

    // 2. Two-call: split [0,chunkLen) and [chunkLen,totalLen). Second call uses first call's final state.
    auto sliceTokens = [&](std::vector<half> const& src, int32_t start, int32_t len, int32_t per_token_len) {
        std::vector<half> out(len * per_token_len);
        std::copy(src.begin() + static_cast<size_t>(start) * per_token_len,
            src.begin() + static_cast<size_t>(start + len) * per_token_len, out.begin());
        return out;
    };
    int32_t const x_per_token = nheads * dim;
    int32_t const dt_per_token = nheads;
    int32_t const bc_per_token = ngroups * dstate;

    std::vector<half> x1 = sliceTokens(xFull, 0, chunkLen, x_per_token);
    std::vector<half> dt1 = sliceTokens(dtFull, 0, chunkLen, dt_per_token);
    std::vector<half> b1 = sliceTokens(bFull, 0, chunkLen, bc_per_token);
    std::vector<half> c1 = sliceTokens(cFull, 0, chunkLen, bc_per_token);
    std::vector<half> x2 = sliceTokens(xFull, chunkLen, chunkLen, x_per_token);
    std::vector<half> dt2 = sliceTokens(dtFull, chunkLen, chunkLen, dt_per_token);
    std::vector<half> b2 = sliceTokens(bFull, chunkLen, chunkLen, bc_per_token);
    std::vector<half> c2 = sliceTokens(cFull, chunkLen, chunkLen, bc_per_token);

    std::vector<float> stateChunked = stateZero;
    std::vector<half> out1, out2;
    runOnce(chunkLen, x1, dt1, b1, c1, stateChunked, out1);
    runOnce(chunkLen, x2, dt2, b2, c2, stateChunked, out2);

    // Reconstruct full output: out1 ++ out2.
    std::vector<half> outChunked(xSize);
    std::copy(out1.begin(), out1.end(), outChunked.begin());
    std::copy(out2.begin(), out2.end(), outChunked.begin() + out1.size());

    // Compare one-shot vs chunked. Tolerances allow for fp16 rounding accumulated through the scan.
    float maxDiffY = 0.f, refMaxY = 0.f;
    for (size_t i = 0; i < outChunked.size(); ++i)
    {
        float a = __half2float(outChunked[i]);
        float b = __half2float(outOneShot[i]);
        maxDiffY = std::max(maxDiffY, std::abs(a - b));
        refMaxY = std::max(refMaxY, std::abs(b));
    }
    float relErrY = maxDiffY / (refMaxY + 1e-8f);
    EXPECT_LT(relErrY, 0.05f) << "chunked-prefill output diverges from one-shot. relErr=" << relErrY
                              << " maxDiff=" << maxDiffY;

    float maxDiffS = 0.f, refMaxS = 0.f;
    for (size_t i = 0; i < stateChunked.size(); ++i)
    {
        maxDiffS = std::max(maxDiffS, std::abs(stateChunked[i] - stateOneShot[i]));
        refMaxS = std::max(refMaxS, std::abs(stateOneShot[i]));
    }
    float relErrS = maxDiffS / (refMaxS + 1e-8f);
    EXPECT_LT(relErrS, 0.05f) << "chunked-prefill final state diverges from one-shot. relErr=" << relErrS
                              << " maxDiff=" << maxDiffS;
}

// Blackwell test configurations: D=64 (Blackwell TMA kernel) + D=128/N=64 (SM80 fallback)
INSTANTIATE_TEST_SUITE_P(SsdCuteDslBlackwell, SsdCuteDslBlackwellTest,
    ::testing::Values(
        // batch, seqLen, nheads, dim, dstate, ngroups
        // D=64, N=128: Blackwell persistent kernel (native)
        SsdCuteDslTestConfig{1, 128, 8, 64, 128, 1}, SsdCuteDslTestConfig{1, 256, 8, 64, 128, 1},
        SsdCuteDslTestConfig{1, 512, 8, 64, 128, 1}, SsdCuteDslTestConfig{1, 1024, 8, 64, 128, 1},
        SsdCuteDslTestConfig{4, 128, 8, 64, 128, 1}, SsdCuteDslTestConfig{1, 256, 64, 64, 128, 1},
        SsdCuteDslTestConfig{1, 256, 64, 64, 128, 8},
        // D=64, N=64: Blackwell persistent kernel (native)
        SsdCuteDslTestConfig{1, 128, 8, 64, 64, 1}, SsdCuteDslTestConfig{1, 256, 8, 64, 64, 1},
        SsdCuteDslTestConfig{1, 512, 8, 64, 64, 1}, SsdCuteDslTestConfig{4, 128, 8, 64, 64, 1},
        // D=128, N=128: SM80 cp.async kernel running on Blackwell GPU (fallback)
        SsdCuteDslTestConfig{1, 128, 8, 128, 128, 1}, SsdCuteDslTestConfig{1, 256, 8, 128, 128, 1},
        SsdCuteDslTestConfig{1, 1024, 8, 128, 128, 1},
        // D=128, N=64: SM80 fallback
        SsdCuteDslTestConfig{1, 128, 8, 128, 64, 1}, SsdCuteDslTestConfig{1, 256, 8, 128, 64, 1},
        // D=80, N=128: one launch with two D=64 scheduler work tiles. Covers the routing boundary,
        // a partial chunk, multi-chunk execution, poisoned ragged padding, restored state,
        // and the exact Nemotron-3 Nano H=96/G=8 shape.
        SsdCuteDslTestConfig{1, 128, 8, 80, 128, 1}, SsdCuteDslTestConfig{1, 129, 8, 80, 128, 1},
        SsdCuteDslTestConfig{1, 512, 8, 80, 128, 1},
        SsdCuteDslTestConfig{2, 384, 8, 80, 128, 1, {131, 384}, false, true},
        SsdCuteDslTestConfig{1, 256, 8, 80, 128, 1, {}, true}, SsdCuteDslTestConfig{1, 128, 96, 80, 128, 8},
        // Varlen: explicit context_lengths (Blackwell varlen path target)
        SsdCuteDslTestConfig{1, 256, 8, 64, 128, 1, {256}},                // batch=1 cl==seqLen sanity
        SsdCuteDslTestConfig{1, 256, 8, 64, 128, 1, {200}},                // batch=1 partial
        SsdCuteDslTestConfig{4, 256, 8, 64, 128, 1, {100, 256, 200, 256}}, // batch>1 mixed
        SsdCuteDslTestConfig{4, 384, 8, 64, 128, 1, {100, 200, 250, 384}}, // mixed + partial last chunk
        SsdCuteDslTestConfig{2, 128, 8, 64, 64, 1, {32, 96}},              // small mixed
        SsdCuteDslTestConfig{8, 128, 8, 64, 128, 1, {1, 64, 128, 128, 100, 128, 50, 128}},
        // Padding-heavy varlen ports of mambaSelectiveStateUpdateTests Padding_*
        // (scaled to seqLen >= 128 so CuTeDSL Blackwell path is exercised).
        SsdCuteDslTestConfig{2, 128, 8, 64, 64, 1, {40, 128}},            // port of Padding_MixedContextLengths
        SsdCuteDslTestConfig{2, 128, 8, 64, 64, 1, {24, 56}},             // port of Padding_AllShorter
        SsdCuteDslTestConfig{2, 256, 8, 64, 128, 1, {40, 256}},           // partial seq + full seq
        SsdCuteDslTestConfig{4, 512, 8, 64, 128, 1, {77, 200, 333, 512}}, // mixed partials, seqLen=512
        SsdCuteDslTestConfig{2, 1024, 8, 64, 128, 1, {500, 1024}},        // large seqLen mixed
        // Nonzero initial state: exercises has_init_states=True kernel path. Reference scan
        // also seeds from this state, so output must match exactly.
        SsdCuteDslTestConfig{1, 128, 8, 64, 128, 1, {}, true}, // batch=1 init_state != 0, single chunk
        SsdCuteDslTestConfig{1, 512, 8, 64, 128, 1, {}, true}, // batch=1 init_state != 0, multi-chunk
        SsdCuteDslTestConfig{
            4, 256, 8, 64, 128, 1, {100, 256, 200, 256}, true},       // batch>1 mixed cl + nonzero init_state
        SsdCuteDslTestConfig{2, 256, 8, 64, 64, 1, {200, 256}, true}, // partial cl + nonzero init_state (D=64 N=64)
        // Long-seq coverage at production batch shapes (bs <= 8).
        SsdCuteDslTestConfig{4, 2048, 8, 64, 128, 1}, SsdCuteDslTestConfig{4, 4096, 8, 64, 128, 1},
        SsdCuteDslTestConfig{8, 1024, 8, 64, 128, 1}, SsdCuteDslTestConfig{8, 2048, 8, 64, 128, 1},
        SsdCuteDslTestConfig{8, 4096, 8, 64, 128, 1},
        // Production-shaped Nemotron-H MMLU batch: h=64, g=8, mixed lengths, unaligned padded row stride,
        // including a sub-128 valid sequence away from batch row 0.
        SsdCuteDslTestConfig{
            16, 462, 64, 64, 128, 8, {78, 175, 210, 248, 96, 325, 360, 400, 462, 462, 462, 462, 462, 462, 462, 462}}),
    [](testing::TestParamInfo<SsdCuteDslTestConfig> const& info) {
        auto const& c = info.param;
        std::string name = "b" + std::to_string(c.batch) + "_s" + std::to_string(c.seqLen) + "_h"
            + std::to_string(c.nheads) + "_d" + std::to_string(c.dim) + "_ds" + std::to_string(c.dstate) + "_g"
            + std::to_string(c.ngroups);
        if (!c.contextLengths.empty())
        {
            name += "_vl";
            for (int32_t cl : c.contextLengths)
                name += std::to_string(cl) + "x";
            name.pop_back();
        }
        if (c.useNonzeroInitState)
            name += "_initstate";
        if (c.poisonPadding)
            name += "_poison";
        return name;
    });

#endif // CUTE_DSL_SSD_BLACKWELL_ENABLED

} // anonymous namespace

#endif // CUTE_DSL_SSD_ENABLED
