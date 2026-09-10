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

// Kernel-level accuracy test for the CuTe DSL INT4 (W4A16) FP16 GEMM. This
// kernel has no C++ consumer yet, so the test drives the AOT-exported artifact
// DIRECTLY (load module -> marshal the generated tensor structs -> call the
// generated wrapper), rather than going through a runner. It covers all compiled
// variants -- the 16 variant subset (bN=128; see build_cutedsl.py / the plugin's
// INT4_FP16_GEMM_VARIANTS) of the full 4-tile x {2,3,4}-stage x {1,2,4,8,16}
// split-K = 60 space.

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

inline int32_t ceilDiv(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

// RAII device buffer: frees on scope exit so a mid-test ASSERT_* early-return
// (Google Test ASSERT_* returns immediately on failure) cannot leak device
// memory across this multi-config suite.
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
// X(prefix, bM, bN, bK, stages, splitK) is expanded for module declaration,
// loading, and runtime dispatch below.
// ---------------------------------------------------------------------------
#define INT4_FP16_GEMM_VARIANTS(X)                                                                                     \
    X(int4_fp16_gemm_16x128x64_s3_sk1, 16, 128, 64, 3, 1)                                                              \
    X(int4_fp16_gemm_16x128x64_s4_sk1, 16, 128, 64, 4, 1)                                                              \
    X(int4_fp16_gemm_16x128x64_s4_sk2, 16, 128, 64, 4, 2)                                                              \
    X(int4_fp16_gemm_16x128x64_s4_sk4, 16, 128, 64, 4, 4)                                                              \
    X(int4_fp16_gemm_32x128x64_s2_sk1, 32, 128, 64, 2, 1)                                                              \
    X(int4_fp16_gemm_32x128x64_s3_sk2, 32, 128, 64, 3, 2)                                                              \
    X(int4_fp16_gemm_32x128x64_s3_sk4, 32, 128, 64, 3, 4)                                                              \
    X(int4_fp16_gemm_32x128x64_s4_sk1, 32, 128, 64, 4, 1)                                                              \
    X(int4_fp16_gemm_64x128x64_s2_sk1, 64, 128, 64, 2, 1)                                                              \
    X(int4_fp16_gemm_64x128x64_s3_sk1, 64, 128, 64, 3, 1)                                                              \
    X(int4_fp16_gemm_64x128x64_s3_sk4, 64, 128, 64, 3, 4)                                                              \
    X(int4_fp16_gemm_64x128x64_s4_sk2, 64, 128, 64, 4, 2)                                                              \
    X(int4_fp16_gemm_64x128x64_s4_sk4, 64, 128, 64, 4, 4)                                                              \
    X(int4_fp16_gemm_128x128x64_s2_sk1, 128, 128, 64, 2, 1)                                                            \
    X(int4_fp16_gemm_128x128x64_s2_sk4, 128, 128, 64, 2, 4)                                                            \
    X(int4_fp16_gemm_128x128x64_s4_sk1, 128, 128, 64, 4, 1)

// ---------------------------------------------------------------------------
// Module storage + loading (all compiled gemm variants, loaded once).
// ---------------------------------------------------------------------------
#define DECL_MODULE(prefix, bM, bN, bK, st, sk) prefix##_Kernel_Module_t g_##prefix{};
INT4_FP16_GEMM_VARIANTS(DECL_MODULE)
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
    // cuLibrary* API, which needs a current context, else the first load fails
    // with cudaErrorUnknown. (Runtime library API on CUDA >= 12.8 auto-inits.)
    cudaFree(nullptr);
    // Clear any stale error so the post-load check reflects only the loads below.
    (void) cudaGetLastError();
    // The generated *_Kernel_Module_Load is void and only prints CUDA errors
    // internally, so detect failure here: each module's library handle must be
    // non-null after loading (a failed driver/runtime library load leaves it
    // null), and no CUDA error may be latched.
    bool ok = true;
#define LOAD_MODULE(prefix, bM, bN, bK, st, sk)                                                                        \
    prefix##_Kernel_Module_Load(&g_##prefix);                                                                          \
    ok = ok && (g_##prefix.module != nullptr);
    INT4_FP16_GEMM_VARIANTS(LOAD_MODULE)
#undef LOAD_MODULE
    if (!ok || cudaGetLastError() != cudaSuccess)
    {
        // Leave gLoaded false so SetUp's ASSERT_TRUE(loadModules()) fails loudly.
        return false;
    }
    gLoaded = true;
    return true;
}

// ---------------------------------------------------------------------------
// Tensor-struct marshalling + dispatch to the matching variant.
// ---------------------------------------------------------------------------
#define SET_2D_TENSOR(t, ptr, d0, d1)                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        (t).data = const_cast<void*>(static_cast<void const*>(ptr));                                                   \
        (t).dynamic_shapes[0] = (d0);                                                                                  \
        (t).dynamic_shapes[1] = (d1);                                                                                  \
        (t).dynamic_strides[0] = static_cast<int64_t>(d1);                                                             \
    } while (0)

#define SET_1D_TENSOR(t, ptr, d0)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        (t).data = const_cast<void*>(static_cast<void const*>(ptr));                                                   \
        (t).dynamic_shapes[0] = (d0);                                                                                  \
    } while (0)

// mWorkspace is the C placeholder (serial split-K reduces in place into C). The
// repacked weight buffer is [qwRows, 128] (128 threads); qwRows depends on the
// tile's bN/bK. Relies on locals: A, QW, scales, C, locks, M, N, K, qwRows,
// scaleRows, nTiles, swizzle, stream, ret.
#define LAUNCH_ONE(PREFIX)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        PREFIX##_Tensor_mA_t tA{};                                                                                     \
        SET_2D_TENSOR(tA, A, M, K);                                                                                    \
        PREFIX##_Tensor_mQW_t tQW{};                                                                                   \
        SET_2D_TENSOR(tQW, QW, qwRows, 128);                                                                           \
        PREFIX##_Tensor_mScales_t tS{};                                                                                \
        SET_2D_TENSOR(tS, scales, scaleRows, N);                                                                       \
        PREFIX##_Tensor_mC_t tC{};                                                                                     \
        SET_2D_TENSOR(tC, C, M, N);                                                                                    \
        PREFIX##_Tensor_mWorkspace_t tW{};                                                                             \
        SET_2D_TENSOR(tW, C, M, N);                                                                                    \
        PREFIX##_Tensor_mLocks_t tL{};                                                                                 \
        SET_1D_TENSOR(tL, locks, nTiles);                                                                              \
        ret = cute_dsl_##PREFIX##_wrapper(&g_##PREFIX, &tA, &tQW, &tS, &tC, &tW, &tL, swizzle, stream);                \
    } while (0)

int launchVariant(int32_t tileBM, int32_t tileBN, int32_t tileBK, int32_t stages, int32_t splitK, void const* A,
    void const* QW, void const* scales, void* C, void* locks, int32_t M, int32_t N, int32_t K, int32_t swizzle,
    cudaStream_t stream)
{
    int32_t const kn = (tileBK / 16) * (tileBN / 64);
    int32_t const qwRows = ceilDiv(N, tileBN) * ceilDiv(K, tileBK) * kn;
    int32_t const scaleRows = ceilDiv(K, kGroupSize);
    int32_t const nTiles = ceilDiv(M, tileBM) * ceilDiv(N, tileBN);
    int32_t ret = -1;
#define DISPATCH_ONE(prefix, bM, bN, bK, st, sk)                                                                       \
    if (tileBM == (bM) && tileBN == (bN) && tileBK == (bK) && stages == (st) && splitK == (sk))                        \
    {                                                                                                                  \
        LAUNCH_ONE(prefix);                                                                                            \
        return ret;                                                                                                    \
    }
    INT4_FP16_GEMM_VARIANTS(DISPATCH_ONE)
#undef DISPATCH_ONE
    return -1; // no matching variant
}

// ---------------------------------------------------------------------------
// Host helpers: INT4 unpack, FP32 reference, and the offline B repack
// (fragment-order uint32; mirrors int4_reference.repack_b_for_tile).
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

// Repack QW[ceil(N/2), K] uint8 into the fragment-order uint32 buffer for a tile
// of (bN, bK). Biased nibble = (raw + 8) & 0xF; padding -> 8. Returns [rows, 128].
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
// Parametrized test over the full variant set + a few edge-case shapes.
// ---------------------------------------------------------------------------
struct Int4GemmTestConfig
{
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t bM;
    int32_t bN;
    int32_t bK;
    int32_t stages;
    int32_t splitK;
    int32_t swizzle;
};

std::vector<Int4GemmTestConfig> buildConfigs()
{
    std::vector<Int4GemmTestConfig> v;
    // Every compiled variant at a canonical shape: K=1024 -> ceil(K/64)=16 is
    // divisible by every split_k {1,2,4,8,16}; N=256 and M=128 work for all tiles.
    constexpr int32_t M = 128, N = 256, K = 1024;
#define ADD_VARIANT(prefix, bM, bN, bK, st, sk) v.push_back(Int4GemmTestConfig{M, N, K, bM, bN, bK, st, sk, 1});
    INT4_FP16_GEMM_VARIANTS(ADD_VARIANT)
#undef ADD_VARIANT
    // Extra edge-case shapes on the 16x128x64 tile.
    v.push_back(Int4GemmTestConfig{200, 160, 256, 16, 128, 64, 4, 1, 1}); // M and N residues; N % 64 != 0
    v.push_back(Int4GemmTestConfig{512, 256, 512, 16, 128, 64, 4, 1, 4}); // grouped-M swizzle = 4
    v.push_back(Int4GemmTestConfig{64, 128, 256, 16, 128, 64, 4, 2, 1});  // small low-M split-K shape
    return v;
}

class Int4Fp16GemmCuteDslTest : public ::testing::TestWithParam<Int4GemmTestConfig>
{
protected:
    void SetUp() override
    {
        ASSERT_TRUE(loadModules()) << "Failed to load INT4 FP16 GEMM CuTe DSL kernel modules";
    }
};

TEST_P(Int4Fp16GemmCuteDslTest, CorrectnessVsFp32Reference)
{
    auto const& cfg = GetParam();
    int32_t const M = cfg.M, N = cfg.N, K = cfg.K;
    int32_t const groupSize = kGroupSize;

    int32_t smMajor{}, smMinor{};
    ASSERT_CUDA_OK(cudaDeviceGetAttribute(&smMajor, cudaDevAttrComputeCapabilityMajor, 0));
    ASSERT_CUDA_OK(cudaDeviceGetAttribute(&smMinor, cudaDevAttrComputeCapabilityMinor, 0));
    int32_t const smVersion = smMajor * 10 + smMinor;

    // N is predicated; K%64==0. A baked split_k is correct only when it
    // divides ceil(K/bK) (split_k=1 always works); runs on Ampere or newer.
    bool const ok
        = smVersion >= 80 && N > 0 && K % 64 == 0 && (cfg.splitK == 1 || ceilDiv(K, cfg.bK) % cfg.splitK == 0);
    if (!ok)
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

    // ---- Offline weight repack (tile-specific bN/bK) ----
    std::vector<uint32_t> hRepacked = repackWeights(hQW, N, K, cfg.bN, cfg.bK);

    // ---- Device buffers (RAII: freed on scope exit even if an ASSERT below
    // returns early) ----
    int64_t const nTiles = static_cast<int64_t>(ceilDiv(M, cfg.bM)) * ceilDiv(N, cfg.bN);
    DeviceBuffer<half> dA, dScales, dC;
    DeviceBuffer<uint32_t> dQW;
    DeviceBuffer<int32_t> dLocks;
    ASSERT_CUDA_OK(dA.alloc(hA.size()));
    ASSERT_CUDA_OK(dQW.alloc(hRepacked.size()));
    ASSERT_CUDA_OK(dScales.alloc(hScales.size()));
    ASSERT_CUDA_OK(dC.alloc(static_cast<size_t>(M) * N));
    ASSERT_CUDA_OK(dLocks.alloc(static_cast<size_t>(nTiles)));

    ASSERT_CUDA_OK(cudaMemcpy(dA.ptr, hA.data(), hA.size() * sizeof(half), cudaMemcpyHostToDevice));
    ASSERT_CUDA_OK(cudaMemcpy(dQW.ptr, hRepacked.data(), hRepacked.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    ASSERT_CUDA_OK(cudaMemcpy(dScales.ptr, hScales.data(), hScales.size() * sizeof(half), cudaMemcpyHostToDevice));
    ASSERT_CUDA_OK(cudaMemset(dC.ptr, 0, static_cast<size_t>(M) * N * sizeof(half)));
    ASSERT_CUDA_OK(cudaMemset(dLocks.ptr, 0, static_cast<size_t>(nTiles) * sizeof(int32_t)));

    // ---- Run ----
    ASSERT_EQ(launchVariant(cfg.bM, cfg.bN, cfg.bK, cfg.stages, cfg.splitK, dA.ptr, dQW.ptr, dScales.ptr, dC.ptr,
                  dLocks.ptr, M, N, K, cfg.swizzle, /*stream=*/nullptr),
        0)
        << "INT4 FP16 GEMM kernel launch failed";
    ASSERT_CUDA_OK(cudaDeviceSynchronize());

    std::vector<half> hC(static_cast<size_t>(M) * N);
    ASSERT_CUDA_OK(cudaMemcpy(hC.data(), dC.ptr, hC.size() * sizeof(half), cudaMemcpyDeviceToHost));

    // ---- Compare ----
    float maxDiff = 0.f;
    float refMax = 0.f;
    for (size_t i = 0; i < hC.size(); ++i)
    {
        float const a = __half2float(hC[i]);
        float const b = __half2float(hRef[i]);
        maxDiff = std::max(maxDiff, std::abs(a - b));
        refMax = std::max(refMax, std::abs(b));
    }
    float const relErr = maxDiff / (refMax + 1e-8f);

    EXPECT_LT(relErr, 0.05f) << "relErr=" << relErr << " maxDiff=" << maxDiff << " refMax=" << refMax;
}

INSTANTIATE_TEST_SUITE_P(Int4Fp16GemmAllVariants, Int4Fp16GemmCuteDslTest, ::testing::ValuesIn(buildConfigs()),
    [](testing::TestParamInfo<Int4GemmTestConfig> const& info) {
        auto const& c = info.param;
        return "t" + std::to_string(c.bM) + "x" + std::to_string(c.bN) + "x" + std::to_string(c.bK) + "_s"
            + std::to_string(c.stages) + "_sk" + std::to_string(c.splitK) + "_M" + std::to_string(c.M) + "_N"
            + std::to_string(c.N) + "_K" + std::to_string(c.K) + "_sw" + std::to_string(c.swizzle);
    });

} // namespace

#endif // CUTE_DSL_INT4_FP16_GEMM_ENABLED
