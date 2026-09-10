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
 * Per-kernel correctness tests for the five audioFbankKernels wrappers in
 * the FP16+TC pipeline:
 *
 *   1. pcmToFramesAndWindow       — reflect-pad + frame + Hann window
 *   2. stftR2C400FusedMagsq       — radix-25×16 self-written R2C FFT, with
 *                                   |·|^2 + FP16 cast fused into Stage 4
 *                                   (T-major [N_pad, K_pad] mag output)
 *   3. melLinearGemmFp16TC        — AOT CuTe DSL FP16 GEMM (Tensor Core)
 *   4. log10MaxReduce             — FP16 read + 2-phase custom global max
 *   5. logMelNormalizeAndCastF16  — FP16 read + scale + reshape
 *
 * Reference values are computed in pure host C++ from the same synthetic
 * input. Tolerances are tuned to F16 round-off on the kernels that read or
 * write FP16, F32 round-off where everything stays FP32.
 *
 * Purpose: pin individual kernels. When a future kernel swap introduces a
 * regression, these tests narrow it down to one stage instead of one big
 * e2e failure.
 */

#include "common/checkMacros.h"
#include "common/tensor.h"
#include "kernels/preprocessKernels/audioFbankKernels.h"
#ifdef CUTE_DSL_GEMM_ENABLED
#include "kernels/talkerMLPKernels/cuteDslGemmRunner.h"
#endif
#include "multimodal/common/audioUtils.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kNFft = 400;
constexpr int32_t kHop = 160;
constexpr int32_t kPad = 200;
constexpr int32_t kNFftBins = 201; // kNFft / 2 + 1
constexpr int32_t kNumMel = 128;
constexpr int32_t kKPad = rt::audioUtils::kFbankKPad; // 208
constexpr float kMelFloor = 1e-10f;

void makeHannPeriodic(std::vector<float>& w)
{
    w.resize(kNFft);
    for (int32_t i = 0; i < kNFft; ++i)
    {
        w[i] = 0.5f
            * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(kNFft)));
    }
}

void makeFftTwiddle(std::vector<float>& twoChan)
{
    constexpr double kTwoPiNeg = -2.0 * 3.14159265358979323846;
    twoChan.resize(static_cast<size_t>(kNFft) * 2);
    for (int32_t k = 0; k < kNFft; ++k)
    {
        double const ang = kTwoPiNeg * static_cast<double>(k) / static_cast<double>(kNFft);
        twoChan[2 * k + 0] = static_cast<float>(std::cos(ang));
        twoChan[2 * k + 1] = static_cast<float>(std::sin(ang));
    }
}

rt::Tensor uploadF32(std::vector<float> const& host, rt::Coords const& shape, cudaStream_t stream)
{
    rt::Tensor t(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    CUDA_CHECK(
        cudaMemcpyAsync(t.rawPointer(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    return t;
}

rt::Tensor uploadFp16(std::vector<__half> const& host, rt::Coords const& shape, cudaStream_t stream)
{
    rt::Tensor t(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    CUDA_CHECK(
        cudaMemcpyAsync(t.rawPointer(), host.data(), host.size() * sizeof(__half), cudaMemcpyHostToDevice, stream));
    return t;
}

void downloadF32(rt::Tensor const& gpu, std::vector<float>& host, cudaStream_t stream)
{
    int64_t const n = gpu.getShape().volume();
    host.resize(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpyAsync(host.data(), gpu.rawPointer(), n * sizeof(float), cudaMemcpyDeviceToHost, stream));
}

void downloadFp16(rt::Tensor const& gpu, std::vector<__half>& host, cudaStream_t stream)
{
    int64_t const n = gpu.getShape().volume();
    host.resize(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpyAsync(host.data(), gpu.rawPointer(), n * sizeof(__half), cudaMemcpyDeviceToHost, stream));
}

// reflectIdx: numpy-style reflect-101 (no-edge-repeat) for index i over signal length N.
// Used as the contract behind the virtual reflect-pad in pcmToFramesAndWindow.
int64_t reflectIdx(int64_t i, int64_t N)
{
    if (N <= 1)
    {
        return 0;
    }
    int64_t const period = 2 * (N - 1);
    int64_t k = i % period;
    if (k < 0)
    {
        k += period;
    }
    return (k < N) ? k : (period - k);
}

float maxAbsDiff(std::vector<float> const& a, std::vector<float> const& b)
{
    EXPECT_EQ(a.size(), b.size());
    float m = 0.0f;
    size_t const n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i)
    {
        float const d = std::abs(a[i] - b[i]);
        if (d > m)
        {
            m = d;
        }
    }
    return m;
}

} // anonymous namespace

// ============================================================================
// 1. pcmToFramesAndWindow
// ============================================================================
//
// Generate a deterministic sine FP32 PCM, run the kernel, and verify against a
// host reference that does the same reflect-pad / frame / Hann pipeline.

TEST(AudioFbankPerKernelTest, PcmToFramesAndWindow)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int64_t N = 16000;         // 1 sec @ 16 kHz
    int64_t const T_full = N / kHop + 1; // 101 frames

    // Deterministic sine: 440 Hz, amplitude 0.5 (mono FP32 in [-1, 1], matching
    // the AudioPCM::samples contract the runner uploads).
    std::vector<float> pcmHost(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i)
    {
        pcmHost[i] = static_cast<float>(0.5 * std::sin(2.0 * M_PI * 440.0 * static_cast<double>(i) / 16000.0));
    }

    rt::Tensor pcmF32 = uploadF32(pcmHost, {N}, stream);

    // pcmToFramesAndWindow takes a plain periodic Hann window: the input is
    // already FP32 [-1, 1], so there is no S16→F32 normaliser (1/32768) to bake
    // in. Mirror that here so the reference below matches the kernel bit-for-bit.
    std::vector<float> hannHost;
    makeHannPeriodic(hannHost);
    rt::Tensor hannWindow = uploadF32(hannHost, {kNFft}, stream);

    rt::Tensor framed({T_full, kNFft}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    kernel::pcmToFramesAndWindow(pcmF32, hannWindow, framed, kNFft, kHop, kPad, stream);

    std::vector<float> gpuOut;
    downloadF32(framed, gpuOut, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Host reference
    std::vector<float> refOut(static_cast<size_t>(T_full) * kNFft);
    for (int64_t t = 0; t < T_full; ++t)
    {
        for (int64_t n = 0; n < kNFft; ++n)
        {
            int64_t const srcIdx = t * kHop + n - kPad;
            int64_t const refIdx = reflectIdx(srcIdx, N);
            // FP32 PCM × plain Hann — matches the kernel's single mul per element.
            refOut[static_cast<size_t>(t * kNFft + n)] = pcmHost[refIdx] * hannHost[n];
        }
    }

    float const absMax = maxAbsDiff(gpuOut, refOut);
    std::cout << "[pcmToFramesAndWindow] T_full=" << T_full << ", abs_max=" << absMax << std::endl;
    // F32 multiplication is bit-exact for this expression; small slack for any
    // FMA-vs-mul-then-add ordering on the GPU.
    EXPECT_LE(absMax, 1e-6f);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// 2. stftR2C400FusedMagsq
// ============================================================================
//
// Build a tiny framed batch (3 frames: impulse, all-ones, sine), run kernel,
// compare to a naive DFT |·|^2 host reference (O(N^2) per frame; fine for N=400).
// Output is T-major FP16 mag at [N_pad, K_pad], so we verify on the active
// [T_out, nFreq) sub-window plus a spot-check that the padding stays zero.

TEST(AudioFbankPerKernelTest, StftR2C400FusedMagsq)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int32_t T_out = 3;
    constexpr int32_t T_full = T_out + 1; // grid_x = T_out; provide one trailing frame
    constexpr int32_t N_pad = 128;        // round_up(T_out=3, 128)

    std::vector<float> framedHost(static_cast<size_t>(T_full) * kNFft, 0.0f);
    // Frame 0: impulse at n=0
    framedHost[0] = 1.0f;
    // Frame 1: constant 0.1 (flat spectrum). DC magsq = (400*0.1)^2 = 1600
    // stays within FP16 range, so the magnitude comparison below is meaningful.
    for (int n = 0; n < kNFft; ++n)
    {
        framedHost[kNFft + n] = 0.1f;
    }
    // Frame 2: cos(2*pi*5*n/400) → energy concentrated at bin 5
    for (int n = 0; n < kNFft; ++n)
    {
        framedHost[2 * kNFft + n] = std::cos(2.0f * static_cast<float>(M_PI) * 5.0f * n / kNFft);
    }
    // Frame 3 (T_full-1) is dummy — kernel grid is T_out and drops it.

    rt::Tensor framed = uploadF32(framedHost, {T_full, kNFft}, stream);
    std::vector<float> twiddleHost;
    makeFftTwiddle(twiddleHost);
    rt::Tensor fftTwiddle = uploadF32(twiddleHost, {kNFft, 2}, stream);

    // Pre-zero the mag buffer (caller contract).
    rt::Tensor magFp16({N_pad, kKPad}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaMemsetAsync(magFp16.rawPointer(), 0, static_cast<size_t>(N_pad) * kKPad * sizeof(__half), stream));

    kernel::stftR2C400FusedMagsq(framed, fftTwiddle, magFp16, T_out, kKPad, stream);

    std::vector<__half> gpuOut;
    downloadFp16(magFp16, gpuOut, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Host reference: naive DFT, take |·|² of first kNFftBins bins.
    float absMax = 0.0f;
    for (int32_t t = 0; t < T_out; ++t)
    {
        for (int32_t k = 0; k < kNFftBins; ++k)
        {
            double re = 0.0, im = 0.0;
            for (int n = 0; n < kNFft; ++n)
            {
                double const ang = -2.0 * M_PI * static_cast<double>(k) * n / kNFft;
                double const x = framedHost[static_cast<size_t>(t * kNFft + n)];
                re += x * std::cos(ang);
                im += x * std::sin(ang);
            }
            float const ref = static_cast<float>(re * re + im * im);
            // Cast through F16 to match the kernel's quantisation.
            float const refF16 = __half2float(__float2half_rn(ref));
            float const got = __half2float(gpuOut[static_cast<size_t>(t * kKPad + k)]);
            float const d = std::abs(got - refF16);
            if (d > absMax)
            {
                absMax = d;
            }
        }
    }

    // Verify the K-axis padding stays zero (cols [kNFftBins, kKPad)).
    bool kPadZero = true;
    for (int32_t t = 0; t < T_out; ++t)
    {
        for (int32_t k = kNFftBins; k < kKPad; ++k)
        {
            if (__half2float(gpuOut[static_cast<size_t>(t * kKPad + k)]) != 0.0f)
            {
                kPadZero = false;
            }
        }
    }
    // And the N-axis padding (rows [T_out, N_pad)).
    bool nPadZero = true;
    for (int32_t t = T_out; t < N_pad; ++t)
    {
        for (int32_t k = 0; k < kKPad; ++k)
        {
            if (__half2float(gpuOut[static_cast<size_t>(t * kKPad + k)]) != 0.0f)
            {
                nPadZero = false;
            }
        }
    }

    std::cout << "[stftR2C400FusedMagsq] abs_max=" << absMax << ", k_pad_zero=" << kPadZero
              << ", n_pad_zero=" << nPadZero << std::endl;
    // Largest magsq across the 3 frames is the cos frame's bin-5 = 200² = 40000
    // (impulse → all bins = 1; constant-0.1 frame DC = (400*0.1)² = 1600). All
    // stay finite in FP16 (max 65504); F16 ULP near 40000 is ~32, so 100 bounds it.
    EXPECT_LE(absMax, 100.0f);
    EXPECT_TRUE(kPadZero) << "K-padding cols [201, 208) must stay zero.";
    EXPECT_TRUE(nPadZero) << "N-padding rows [T_out, N_pad) must stay zero.";

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// 3. melLinearGemmFp16TC
// ============================================================================
//
// Real-shape problem (M=128 mel, K_pad=208, N_pad=128 for T_out=16). FP16 K-major
// filter + FP16 K-major mag (K-padding zeroed), compare to triple-loop host
// SGEMM run only over the active [M, T_out] sub-window. Padded rows/cols are
// asserted to stay at the GEMM-input zero values (math invariant).

TEST(AudioFbankPerKernelTest, MelLinearGemmFp16TC)
{
    // CuTe DSL gemm module is shared process-wide; skip if not built or
    // the GPU has no compiled variant. Check BEFORE creating the stream so the
    // GTEST_SKIP early return does not leak it.
#ifndef CUTE_DSL_GEMM_ENABLED
    GTEST_SKIP() << "CuTe DSL GEMM not enabled in this build (rebuild with -DENABLE_CUTE_DSL=gemm).";
#else
    if (!CuteDslGemmRunner::loadKernelModule())
    {
        GTEST_SKIP() << "CuteDslGemmRunner module unavailable on this build / GPU.";
    }
#endif

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int32_t M = kNumMel;
    constexpr int32_t K = kKPad; // 208
    constexpr int32_t T_out = 16;
    constexpr int32_t N = 128;           // round_up(T_out, 128)
    constexpr int32_t nFreq = kNFftBins; // active K-window 201; cols [201, 208) zero

    std::mt19937 gen(0xDEAD);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // mel filter is non-negative

    // A: melFilterFp16Kmajor [M, K_pad] — fill [0, nFreq), zero-pad rest.
    std::vector<__half> aHost(static_cast<size_t>(M) * K, __half(0.0f));
    for (int32_t m = 0; m < M; ++m)
    {
        for (int32_t k = 0; k < nFreq; ++k)
        {
            aHost[static_cast<size_t>(m) * K + k] = __float2half_rn(dist(gen));
        }
    }
    // B: magFp16 [N_pad, K_pad] — fill [0, T_out) × [0, nFreq), zero-pad rest.
    std::vector<__half> bHost(static_cast<size_t>(N) * K, __half(0.0f));
    for (int32_t t = 0; t < T_out; ++t)
    {
        for (int32_t k = 0; k < nFreq; ++k)
        {
            bHost[static_cast<size_t>(t) * K + k] = __float2half_rn(dist(gen));
        }
    }

    rt::Tensor aDev = uploadFp16(aHost, {M, K}, stream);
    rt::Tensor bDev = uploadFp16(bHost, {N, K}, stream);
    rt::Tensor cDev({M, N}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    kernel::melLinearGemmFp16TC(aDev, bDev, cDev, stream);

    std::vector<__half> gpuOut;
    downloadFp16(cDev, gpuOut, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Reference: C[m, n] = sum_k A[m, k] * B[n, k] (only k in [0, nFreq) contribute).
    float absMax = 0.0f;
    for (int32_t m = 0; m < M; ++m)
    {
        for (int32_t n = 0; n < T_out; ++n)
        {
            double acc = 0.0;
            for (int32_t k = 0; k < nFreq; ++k)
            {
                acc += static_cast<double>(__half2float(aHost[static_cast<size_t>(m) * K + k]))
                    * static_cast<double>(__half2float(bHost[static_cast<size_t>(n) * K + k]));
            }
            float const refF16 = __half2float(__float2half_rn(static_cast<float>(acc)));
            float const got = __half2float(gpuOut[static_cast<size_t>(m) * N + n]);
            float const d = std::abs(got - refF16);
            if (d > absMax)
            {
                absMax = d;
            }
        }
    }

    std::cout << "[melLinearGemmFp16TC] abs_max=" << absMax << " (M=" << M << ", N=" << N << ", K=" << K
              << ", T_out=" << T_out << ", nFreq=" << nFreq << ")" << std::endl;
    // FP32-accumulate FP16 GEMM with K=201 active inner-product terms; the
    // accumulation order differs from the host triple loop and rounding to
    // FP16 output adds another LSB.
    EXPECT_LE(absMax, 1.0f);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// 4. log10MaxReduce (FP16 input + N_pad stride)
// ============================================================================

TEST(AudioFbankPerKernelTest, Log10MaxReduce)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int32_t M = kNumMel;
    constexpr int32_t T_out = 32;
    constexpr int32_t N_pad = 128; // round_up(T_out, 128)

    std::mt19937 gen(0xBEEF);
    // Include values below melFloor to exercise the clamp path.
    std::uniform_real_distribution<float> dist(-2.0f, 5.0f); // log-domain range
    // melPower laid out as [M, N_pad] with [0, T_out) active, [T_out, N_pad) zero
    // (mirrors what the GEMM produces on a real run).
    std::vector<__half> melHost(static_cast<size_t>(M) * N_pad, __half(0.0f));
    for (int32_t m = 0; m < M; ++m)
    {
        for (int32_t t = 0; t < T_out; ++t)
        {
            float const v = std::exp(dist(gen)); // positive, spans ~exp(-2)..exp(5) ≈ 0.13..150
            melHost[static_cast<size_t>(m) * N_pad + t] = __float2half_rn(v);
        }
    }
    // Force one element to be very small (< kMelFloor)
    melHost[0] = __float2half_rn(1e-15f);

    rt::Tensor melPower = uploadFp16(melHost, {M, N_pad}, stream);
    rt::Tensor maxLog({1}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

    kernel::log10MaxReduce(melPower, T_out, maxLog, kMelFloor, stream);
    std::vector<float> gpuOut;
    downloadF32(maxLog, gpuOut, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Reference: only the active [M, T_out) window contributes.
    float refMax = -std::numeric_limits<float>::infinity();
    for (int32_t m = 0; m < M; ++m)
    {
        for (int32_t t = 0; t < T_out; ++t)
        {
            float const v = __half2float(melHost[static_cast<size_t>(m) * N_pad + t]);
            float const clamped = std::max(v, kMelFloor);
            float const logged = std::log10(clamped);
            if (logged > refMax)
            {
                refMax = logged;
            }
        }
    }

    std::cout << "[log10MaxReduce] gpu=" << gpuOut[0] << ", ref=" << refMax
              << ", abs_diff=" << std::abs(gpuOut[0] - refMax) << std::endl;
    // F16 → F32 round-trip on the max-bearing element is bit-exact, but log10
    // is computed in F32 so the max may pick up one ulp from the LDG read.
    EXPECT_NEAR(gpuOut[0], refMax, 1e-3f);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// 5. logMelNormalizeAndCastF16 (FP16 input + N_pad stride)
// ============================================================================

TEST(AudioFbankPerKernelTest, LogMelNormalizeAndCastF16)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int32_t M = kNumMel;
    constexpr int32_t T_out = 24;
    constexpr int32_t N_pad = 128; // round_up(T_out, 128)

    std::mt19937 gen(0xF00D);
    std::uniform_real_distribution<float> dist(-5.0f, 3.0f);
    // [M, N_pad] padded layout; only the active [0, T_out) window is read.
    std::vector<__half> melHost(static_cast<size_t>(M) * N_pad, __half(0.0f));
    for (int32_t m = 0; m < M; ++m)
    {
        for (int32_t t = 0; t < T_out; ++t)
        {
            float const v = std::exp(dist(gen));
            melHost[static_cast<size_t>(m) * N_pad + t] = __float2half_rn(v);
        }
    }
    // Threshold for the (maxLog - 8.0) floor branch.
    melHost[5] = __float2half_rn(1e-20f);

    // Pick a deterministic maxLog (computed from inputs).
    float refMax = -std::numeric_limits<float>::infinity();
    for (int32_t m = 0; m < M; ++m)
    {
        for (int32_t t = 0; t < T_out; ++t)
        {
            float const v = __half2float(melHost[static_cast<size_t>(m) * N_pad + t]);
            float const logged = std::log10(std::max(v, kMelFloor));
            if (logged > refMax)
            {
                refMax = logged;
            }
        }
    }

    rt::Tensor melPower = uploadFp16(melHost, {M, N_pad}, stream);
    std::vector<float> maxLogHost{refMax};
    rt::Tensor maxLog = uploadF32(maxLogHost, {1}, stream);
    rt::Tensor melOut({1, M, T_out}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    kernel::logMelNormalizeAndCastF16(melPower, T_out, maxLog, melOut, kMelFloor, stream);

    int64_t const n = melOut.getShape().volume();
    std::vector<half> gpuOut(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpyAsync(gpuOut.data(), melOut.rawPointer(), n * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Reference: y = max(log10(max(x, melFloor)), maxLog - 8.0); out = (y+4)/4 as F16
    float absMax = 0.0f;
    for (int32_t m = 0; m < M; ++m)
    {
        for (int32_t t = 0; t < T_out; ++t)
        {
            float const x = __half2float(melHost[static_cast<size_t>(m) * N_pad + t]);
            float const logged = std::log10(std::max(x, kMelFloor));
            float const y = std::max(logged, refMax - 8.0f);
            float const refF32 = (y + 4.0f) * 0.25f;
            // Cast through F16 to match the kernel's F32→F16 quantisation.
            float const refF16AsF32 = __half2float(__float2half(refF32));
            float const gpuF32 = __half2float(gpuOut[static_cast<size_t>(m) * T_out + t]);
            float const d = std::abs(gpuF32 - refF16AsF32);
            if (d > absMax)
            {
                absMax = d;
            }
        }
    }

    std::cout << "[logMelNormalizeAndCastF16] abs_max=" << absMax << std::endl;
    EXPECT_LE(absMax, 1e-3f); // F16 LSB at unit magnitude

    CUDA_CHECK(cudaStreamDestroy(stream));
}
