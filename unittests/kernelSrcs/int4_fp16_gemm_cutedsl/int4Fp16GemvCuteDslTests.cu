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

// Kernel-level accuracy test for the CuTe DSL INT4 (W4A16) FP16 decode GEMV
// (narrow-CTA). Like the GEMM test, it drives the AOT-exported artifact DIRECTLY
// (load module -> marshal the generated tensor structs -> call the generated
// wrapper). The GEMV consumes the SAME fragment weight buffer as the GEMM
// (bN=128/bK=64); M is the baked dimension, so it covers all 8 compiled
// variants (M = 1..8), each at a few decode-relevant (N, K) shapes.

#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED

// The AOT-generated headers use cudaLibrary_t, which only exists in the CUDA
// 12.8+ runtime. On CUDA < 12.8 the build defines TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT
// (manually on x86, auto for jetson-orin); route it to the driver CUlibrary type.
#include <cuda.h>
#if defined(TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT)
#include <cuda_runtime.h>
#if CUDA_VERSION < 12800
typedef CUlibrary cudaLibrary_t;
static inline cudaError_t cudaLibraryUnload(cudaLibrary_t lib)
{
    return static_cast<cudaError_t>(cuLibraryUnload(lib));
}
#endif // CUDA_VERSION < 12800
#endif // TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT

#include "cutedsl_all.h"

#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace
{

#define ASSERT_CUDA_OK(expr) ASSERT_EQ((expr), cudaSuccess) << cudaGetErrorString(cudaGetLastError())

constexpr int32_t kGroupSize = 128;
constexpr int32_t kBN = 128; // fragment N-block (shared with the GEMM)
constexpr int32_t kBK = 64;  // fragment K-tile

inline int32_t ceilDiv(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

// RAII device buffer: frees on scope exit so a mid-test ASSERT_* early-return
// cannot leak device memory across the suite.
template <typename T>
struct DeviceBuffer
{
    T* ptr{nullptr};
    DeviceBuffer() = default;
    ~DeviceBuffer()
    {
        if (ptr != nullptr)
        {
            cudaFree(ptr);
        }
    }
    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;
    cudaError_t alloc(size_t count)
    {
        return cudaMalloc(&ptr, count * sizeof(T));
    }
};

// ---------------------------------------------------------------------------
// The 8 compiled variants (one exported function per M). X(prefix, M) is expanded for
// module declaration, loading, and runtime dispatch below.
// ---------------------------------------------------------------------------
#define INT4_FP16_GEMV_VARIANTS(X)                                                                                     \
    X(int4_fp16_gemv_m1, 1)                                                                                            \
    X(int4_fp16_gemv_m2, 2)                                                                                            \
    X(int4_fp16_gemv_m3, 3)                                                                                            \
    X(int4_fp16_gemv_m4, 4)                                                                                            \
    X(int4_fp16_gemv_m5, 5)                                                                                            \
    X(int4_fp16_gemv_m6, 6)                                                                                            \
    X(int4_fp16_gemv_m7, 7)                                                                                            \
    X(int4_fp16_gemv_m8, 8)

// ---------------------------------------------------------------------------
// Module storage + loading (all 8, loaded once).
// ---------------------------------------------------------------------------
#define DECL_MODULE(prefix, M) prefix##_Kernel_Module_t g_##prefix{};
INT4_FP16_GEMV_VARIANTS(DECL_MODULE)
#undef DECL_MODULE

bool gLoaded = false;
std::mutex gLoadMutex;

bool loadModules()
{
    std::lock_guard<std::mutex> lock(gLoadMutex);
    if (gLoaded)
    {
        return true;
    }
    // Ensure the CUDA primary context exists before the module loads: on CUDA
    // < 12.8 the generated loader routes cudaLibrary_t through the driver
    // cuLibrary* API, which needs a current context, else the first load fails.
    cudaFree(nullptr);
    (void) cudaGetLastError();
    bool ok = true;
#define LOAD_MODULE(prefix, M)                                                                                         \
    prefix##_Kernel_Module_Load(&g_##prefix);                                                                          \
    ok = ok && (g_##prefix.module != nullptr);
    INT4_FP16_GEMV_VARIANTS(LOAD_MODULE)
#undef LOAD_MODULE
    if (!ok || cudaGetLastError() != cudaSuccess)
    {
        return false;
    }
    gLoaded = true;
    return true;
}

// ---------------------------------------------------------------------------
// Tensor-struct marshalling + dispatch to the matching M-variant.
// ABI: (mA, mQW, mScales, mOut, stream) -- single launch, no workspace/locks.
// ---------------------------------------------------------------------------
#define SET_2D_TENSOR(t, ptr, d0, d1)                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        (t).data = const_cast<void*>(static_cast<void const*>(ptr));                                                   \
        (t).dynamic_shapes[0] = (d0);                                                                                  \
        (t).dynamic_shapes[1] = (d1);                                                                                  \
        (t).dynamic_strides[0] = static_cast<int64_t>(d1);                                                             \
    } while (0)

// Relies on locals: A, QW, scales, Out, M, N, K, qwRows, scaleRows, stream, ret.
#define LAUNCH_ONE(PREFIX)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        PREFIX##_Tensor_mA_t tA{};                                                                                     \
        SET_2D_TENSOR(tA, A, M, K);                                                                                    \
        PREFIX##_Tensor_mQW_t tQW{};                                                                                   \
        SET_2D_TENSOR(tQW, QW, qwRows, 128);                                                                           \
        PREFIX##_Tensor_mScales_t tS{};                                                                                \
        SET_2D_TENSOR(tS, scales, scaleRows, N);                                                                       \
        PREFIX##_Tensor_mOut_t tOut{};                                                                                 \
        SET_2D_TENSOR(tOut, Out, M, N);                                                                                \
        ret = cute_dsl_##PREFIX##_wrapper(&g_##PREFIX, &tA, &tQW, &tS, &tOut, stream);                                 \
    } while (0)

int launchVariant(
    int32_t M, void const* A, void const* QW, void const* scales, void* Out, int32_t N, int32_t K, cudaStream_t stream)
{
    int32_t const kn = (kBK / 16) * (kBN / 64);
    int32_t const qwRows = ceilDiv(N, kBN) * ceilDiv(K, kBK) * kn;
    int32_t const scaleRows = ceilDiv(K, kGroupSize);
    int32_t ret = -1;
#define DISPATCH_ONE(prefix, VM)                                                                                       \
    if (M == (VM))                                                                                                     \
    {                                                                                                                  \
        LAUNCH_ONE(prefix);                                                                                            \
        return ret;                                                                                                    \
    }
    INT4_FP16_GEMV_VARIANTS(DISPATCH_ONE)
#undef DISPATCH_ONE
    return -1; // no matching variant
}

// ---------------------------------------------------------------------------
// Host helpers: INT4 unpack, FP32 reference, and the offline B repack
// (fragment-order uint32; identical to the GEMM's, bN=128/bK=64).
// ---------------------------------------------------------------------------
inline int32_t signedNibble(std::vector<uint8_t> const& qweight, int32_t n, int32_t k, int32_t K)
{
    uint8_t const packed = qweight[static_cast<size_t>(n / 2) * K + k];
    int32_t const nib = (n % 2 == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
    return nib >= 8 ? nib - 16 : nib;
}

void int4GemmReference(std::vector<half> const& A, std::vector<uint8_t> const& qweight, std::vector<half> const& scales,
    std::vector<half>& C, int32_t M, int32_t N, int32_t K, int32_t groupSize)
{
    for (int32_t m = 0; m < M; ++m)
    {
        for (int32_t n = 0; n < N; ++n)
        {
            float acc = 0.f;
            for (int32_t k = 0; k < K; ++k)
            {
                float const a = __half2float(A[static_cast<size_t>(m) * K + k]);
                float const s = __half2float(scales[static_cast<size_t>(k / groupSize) * N + n]);
                acc += a * (static_cast<float>(signedNibble(qweight, n, k, K)) * s);
            }
            C[static_cast<size_t>(m) * N + n] = __float2half(acc);
        }
    }
}

std::vector<uint32_t> repackWeights(std::vector<uint8_t> const& qweight, int32_t N, int32_t K, int32_t bN, int32_t bK)
{
    int32_t const kBlocks = bK / 16;
    int32_t const nPairs = bN / 64;
    int32_t const kn = kBlocks * nPairs;
    int32_t const numNBlocks = ceilDiv(N, bN);
    int32_t const numKTiles = ceilDiv(K, bK);
    int64_t const rows = static_cast<int64_t>(numNBlocks) * numKTiles * kn;
    std::vector<uint32_t> out(static_cast<size_t>(rows) * 128, 0u);

    auto biased = [&](int32_t n, int32_t k) -> uint32_t {
        if (n >= N || k >= K)
        {
            return 8u;
        }
        uint8_t const packed = qweight[static_cast<size_t>(n / 2) * K + k];
        uint32_t const nib = (n % 2 == 0) ? (packed & 0xFu) : ((packed >> 4) & 0xFu);
        return (nib + 8u) & 0xFu;
    };

    struct Nibble
    {
        int32_t shift;
        bool hi;
        int32_t koff;
    };
    static constexpr Nibble kNibbles[8] = {
        {0, false, 0},
        {4, false, 8},
        {8, true, 0},
        {12, true, 8},
        {16, false, 1},
        {20, false, 9},
        {24, true, 1},
        {28, true, 9},
    };

    for (int32_t t = 0; t < 128; ++t)
    {
        int32_t const nBase = t / 4;
        int32_t const kb = 2 * (t % 4);
        for (int32_t nb = 0; nb < numNBlocks; ++nb)
        {
            for (int32_t kt = 0; kt < numKTiles; ++kt)
            {
                for (int32_t kbl = 0; kbl < kBlocks; ++kbl)
                {
                    for (int32_t p = 0; p < nPairs; ++p)
                    {
                        int32_t const idx = kbl * nPairs + p;
                        int32_t const nLo = nb * bN + (nBase + 64 * p);
                        int32_t const nHi = nLo + 32;
                        int32_t const k0 = kt * bK + (16 * kbl + kb);
                        uint32_t word = 0u;
                        for (auto const& nbl : kNibbles)
                        {
                            int32_t const nrow = nbl.hi ? nHi : nLo;
                            int32_t const kcol = k0 + nbl.koff;
                            word |= biased(nrow, kcol) << nbl.shift;
                        }
                        int64_t const row = (static_cast<int64_t>(nb * numKTiles + kt) * kn + idx);
                        out[static_cast<size_t>(row) * 128 + t] = word;
                    }
                }
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Parametrized test over the 8 M-variants x a few decode-relevant (N, K).
// ---------------------------------------------------------------------------
struct Int4GemvTestConfig
{
    int32_t M;
    int32_t N;
    int32_t K;
};

std::vector<Int4GemvTestConfig> buildConfigs()
{
    std::vector<Int4GemvTestConfig> v;
    // (N, K) shapes: a canonical square-ish shape and Qwen2.5-0.5B projections
    // (incl. tall-K down_proj) -- N unconstrained, K % 64 == 0.
    int32_t const shapes[][2] = {{160, 256}, {256, 1024}, {896, 896}, {4864, 896}, {896, 4864}};
    for (int32_t M = 1; M <= 8; ++M)
    {
        for (auto const& s : shapes)
        {
            v.push_back(Int4GemvTestConfig{M, s[0], s[1]});
        }
    }
    return v;
}

class Int4Fp16GemvCuteDslTest : public ::testing::TestWithParam<Int4GemvTestConfig>
{
protected:
    void SetUp() override
    {
        ASSERT_TRUE(loadModules()) << "Failed to load INT4 FP16 GEMV CuTe DSL kernel modules";
    }
};

TEST_P(Int4Fp16GemvCuteDslTest, CorrectnessVsFp32Reference)
{
    auto const& cfg = GetParam();
    int32_t const M = cfg.M, N = cfg.N, K = cfg.K;
    int32_t const groupSize = kGroupSize;

    int32_t smMajor{}, smMinor{};
    ASSERT_CUDA_OK(cudaDeviceGetAttribute(&smMajor, cudaDevAttrComputeCapabilityMajor, 0));
    ASSERT_CUDA_OK(cudaDeviceGetAttribute(&smMinor, cudaDevAttrComputeCapabilityMinor, 0));
    int32_t const smVersion = smMajor * 10 + smMinor;

    // Decode GEMV: K % 64 == 0 (N unconstrained; repack pads to 128); Ampere+.
    if (smVersion < 80 || K % 64 != 0)
    {
        GTEST_SKIP() << "Unsupported config";
    }

    // ---- Host inputs ----
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> aDist(-1.f, 1.f);
    std::uniform_real_distribution<float> scaleDist(0.01f, 0.2f);
    std::uniform_int_distribution<int32_t> byteDist(0, 255);

    int32_t const packedRows = (N + 1) / 2;
    int32_t const scaleRows = ceilDiv(K, groupSize);

    std::vector<half> hA(static_cast<size_t>(M) * K);
    std::vector<uint8_t> hQW(static_cast<size_t>(packedRows) * K);
    std::vector<half> hScales(static_cast<size_t>(scaleRows) * N);
    for (auto& v : hA)
        v = __float2half(aDist(rng));
    for (auto& v : hQW)
        v = static_cast<uint8_t>(byteDist(rng));
    for (auto& v : hScales)
        v = __float2half(scaleDist(rng));

    // ---- Reference ----
    std::vector<half> hRef(static_cast<size_t>(M) * N, __float2half(0.f));
    int4GemmReference(hA, hQW, hScales, hRef, M, N, K, groupSize);

    // ---- Offline weight repack (bN=128/bK=64; shared with the GEMM) ----
    std::vector<uint32_t> hRepacked = repackWeights(hQW, N, K, kBN, kBK);

    // ---- Device buffers ----
    DeviceBuffer<half> dA, dScales, dOutput;
    DeviceBuffer<uint32_t> dQW;
    ASSERT_CUDA_OK(dA.alloc(hA.size()));
    ASSERT_CUDA_OK(dQW.alloc(hRepacked.size()));
    ASSERT_CUDA_OK(dScales.alloc(hScales.size()));
    ASSERT_CUDA_OK(dOutput.alloc(static_cast<size_t>(M) * N));

    ASSERT_CUDA_OK(cudaMemcpy(dA.ptr, hA.data(), hA.size() * sizeof(half), cudaMemcpyHostToDevice));
    ASSERT_CUDA_OK(cudaMemcpy(dQW.ptr, hRepacked.data(), hRepacked.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    ASSERT_CUDA_OK(cudaMemcpy(dScales.ptr, hScales.data(), hScales.size() * sizeof(half), cudaMemcpyHostToDevice));
    ASSERT_CUDA_OK(cudaMemset(dOutput.ptr, 0, static_cast<size_t>(M) * N * sizeof(half)));

    // ---- Run ----
    ASSERT_EQ(launchVariant(M, dA.ptr, dQW.ptr, dScales.ptr, dOutput.ptr, N, K, /*stream=*/nullptr), 0)
        << "INT4 FP16 GEMV kernel launch failed";
    ASSERT_CUDA_OK(cudaDeviceSynchronize());

    std::vector<half> hOutput(static_cast<size_t>(M) * N);
    ASSERT_CUDA_OK(cudaMemcpy(hOutput.data(), dOutput.ptr, hOutput.size() * sizeof(half), cudaMemcpyDeviceToHost));

    // ---- Compare ----
    float maxDiff = 0.f;
    float refMax = 0.f;
    for (size_t i = 0; i < hOutput.size(); ++i)
    {
        float const a = __half2float(hOutput[i]);
        float const b = __half2float(hRef[i]);
        maxDiff = std::max(maxDiff, std::abs(a - b));
        refMax = std::max(refMax, std::abs(b));
    }
    float const relErr = maxDiff / (refMax + 1e-8f);

    EXPECT_LT(relErr, 0.05f) << "relErr=" << relErr << " maxDiff=" << maxDiff << " refMax=" << refMax;
}

INSTANTIATE_TEST_SUITE_P(Int4Fp16GemvAllVariants, Int4Fp16GemvCuteDslTest, ::testing::ValuesIn(buildConfigs()),
    [](testing::TestParamInfo<Int4GemvTestConfig> const& info) {
        auto const& c = info.param;
        return "M" + std::to_string(c.M) + "_N" + std::to_string(c.N) + "_K" + std::to_string(c.K);
    });

} // namespace

#endif // CUTE_DSL_INT4_FP16_GEMM_ENABLED
