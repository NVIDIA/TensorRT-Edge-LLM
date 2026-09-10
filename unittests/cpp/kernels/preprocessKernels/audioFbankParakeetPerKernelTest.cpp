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

// Per-kernel correctness tests for the five Parakeet (Nemotron-Omni) FP16-in /
// FP32-out audioFbank kernels: pcmPreemphFramesAndWindow, stftR2C512FusedMagsq,
// melLinearGemmFp16inFp32out, melStatsLnPerFeature, melNormalizeZScoreTimeFirst
// (plus a MelStatsLnSilenceNoNaN guard for the fmaxf(var, 0) clamp). References
// are pure host C++; tolerances track F16 round-off where a kernel reads/writes
// FP16, F32 round-off otherwise. Runs alongside the Whisper per-kernel suite via
// --gtest_filter=AudioFbank*.

#include "common/checkMacros.h"
#include "common/tensor.h"
#include "kernels/preprocessKernels/audioFbankKernels.h"
#ifdef CUTE_DSL_GEMM_ENABLED
#include "kernels/talkerMLPKernels/cuteDslGemmRunner.h"
#endif
#include "multimodal/common/audioUtils.h"
#include "testUtils.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

using namespace trt_edgellm;

namespace
{

// Parakeet (Nemotron-Omni) geometry — cf. Whisper's 400 / 160 / 200 / 201 / 208.
constexpr int32_t kNFft = 512;
constexpr int32_t kHop = 160;
constexpr int32_t kWin = 400;       // symmetric-Hann taps, centered in the 512 FFT
constexpr int32_t kCenterPad = 256; // nFft / 2 (virtual zero pad)
constexpr int32_t kNFreq = 257;     // nFft / 2 + 1
constexpr int32_t kNumMel = 128;
constexpr int32_t kKPad = 272;                  // fbankKPadParakeet(257) = round_up(257, 16)
constexpr float kLogGuard = 1.0f / 16777216.0f; // 2^-24, natural-log input floor
constexpr float kNormEps = 1e-5f;               // per-feature std epsilon
constexpr float kPreemph = 0.97f;

// Symmetric-Hann (torch.hann_window(periodic=False), denominator winLength-1)
// taps for a `kWin`-wide window, embedded centered in the `kNFft`-wide FFT input
// via the same helper initFbankResources uses — so the test window matches the
// production window bit-for-bit.
void makeParakeetWindowHost(std::vector<float>& out)
{
    std::vector<float> taps(static_cast<size_t>(kWin));
    for (int32_t i = 0; i < kWin; ++i)
    {
        taps[static_cast<size_t>(i)] = 0.5f
            * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(kWin - 1)));
    }
    rt::audioUtils::makeCentredWindowHost(taps, kNFft, out);
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
// 1. pcmPreemphFramesAndWindow
// ============================================================================
//
// Deterministic sine FP32 PCM → kernel → host reference implementing the same
// preemphasis recurrence + virtual center zero-pad + symmetric-Hann window.

TEST(AudioFbankParakeetPerKernelTest, PcmPreemphFramesAndWindow)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int64_t N = 16000;                          // 1 sec @ 16 kHz
    int32_t const T_out = static_cast<int32_t>(N / kHop); // floor(N / hop) = 100

    // Deterministic sine: 440 Hz, amplitude 0.5 (mono FP32 in [-1, 1], matching
    // AudioPCM::samples — no S16→F32 normaliser, and none is baked into window).
    std::vector<float> pcmHost(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i)
    {
        pcmHost[static_cast<size_t>(i)]
            = static_cast<float>(0.5 * std::sin(2.0 * M_PI * 440.0 * static_cast<double>(i) / 16000.0));
    }

    rt::Tensor pcmF32({N}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice(pcmF32, pcmHost);

    std::vector<float> windowHost;
    makeParakeetWindowHost(windowHost); // [kNFft], symmetric Hann centered at offset 56
    rt::Tensor windowF32({kNFft}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice(windowF32, windowHost);

    rt::Tensor framed({T_out, kNFft}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    kernel::pcmPreemphFramesAndWindow(pcmF32, windowF32, framed, kNFft, kHop, kCenterPad, kPreemph, T_out, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> const gpuOut = copyDeviceToHost<float>(framed);

    // Host reference (see audioFbankKernels.h pcmPreemphFramesAndWindow):
    //   idx = t*hop + n - centerPad
    //   y   = 0                              if idx < 0 or idx >= N
    //       = pcm[0]                         if idx == 0
    //       = pcm[idx] - preemph*pcm[idx-1]  otherwise
    //   framed[t, n] = y * window[n]
    std::vector<float> refOut(static_cast<size_t>(T_out) * kNFft, 0.0f);
    for (int32_t t = 0; t < T_out; ++t)
    {
        for (int32_t n = 0; n < kNFft; ++n)
        {
            int64_t const idx = static_cast<int64_t>(t) * kHop + n - kCenterPad;
            float y;
            if (idx < 0 || idx >= N)
            {
                y = 0.0f;
            }
            else if (idx == 0)
            {
                y = pcmHost[0];
            }
            else
            {
                y = pcmHost[static_cast<size_t>(idx)] - kPreemph * pcmHost[static_cast<size_t>(idx - 1)];
            }
            refOut[static_cast<size_t>(t) * kNFft + n] = y * windowHost[static_cast<size_t>(n)];
        }
    }

    float const absMax = maxAbsDiff(gpuOut, refOut);
    std::cout << "[pcmPreemphFramesAndWindow] T_out=" << T_out << ", abs_max=" << absMax << std::endl;
    // FP32 multiply + single FMA (a - c*b); slack covers FMA-vs-mul-then-add ordering.
    EXPECT_LE(absMax, 1e-6f);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// 2. stftR2C512FusedMagsq
// ============================================================================
//
// Tiny framed batch (3 frames: impulse / constant 0.1 / half-amplitude cosine),
// run kernel, compare to a naive DFT |·|^2 host reference (O(N^2) per frame;
// fine for N=512). Output is T-major FP16 mag at [N_pad, K_pad]; verify on the
// active [T_out, nFreq) sub-window plus a spot-check that the padding stays zero.
//
// FP16 overflow trap: a full-amplitude cosine at N=512 peaks at
// magsq = (512/2)^2 = 65536 > FP16 max 65504 → inf → inf-inf=NaN → silent pass.
// So frame 2 uses amplitude 0.5 (peak magsq = 128^2 = 16384) and frame 1 uses
// constant 0.1 (DC magsq = (512*0.1)^2 = 2621) — every magsq stays finite in FP16.

TEST(AudioFbankParakeetPerKernelTest, StftR2C512FusedMagsq)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int32_t T_out = 3;
    constexpr int32_t N_pad = 128; // round_up(T_out, 128)

    // Parakeet framing emits exactly T_out rows (no Whisper stft[..., :-1] trim),
    // so the framed buffer is [T_out, kNFft] with all rows real.
    std::vector<float> framedHost(static_cast<size_t>(T_out) * kNFft, 0.0f);
    // Frame 0: impulse at n=0 → flat |X|^2 = 1 across all bins.
    framedHost[0] = 1.0f;
    // Frame 1: constant 0.1 → DC magsq = (512*0.1)^2 = 2621, rest ~0.
    for (int32_t n = 0; n < kNFft; ++n)
    {
        framedHost[static_cast<size_t>(kNFft) + n] = 0.1f;
    }
    // Frame 2: 0.5 * cos(2*pi*5*n/512) → energy concentrated at bin 5, peak
    // magsq = (0.5 * 512/2)^2 = 128^2 = 16384 (safely < FP16 max).
    for (int32_t n = 0; n < kNFft; ++n)
    {
        framedHost[static_cast<size_t>(2) * kNFft + n] = 0.5f
            * std::cos(2.0f * static_cast<float>(M_PI) * 5.0f * static_cast<float>(n) / static_cast<float>(kNFft));
    }

    rt::Tensor framed({T_out, kNFft}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice(framed, framedHost);
    std::vector<float> twiddleHost;
    rt::audioUtils::makeFftTwiddleHost(kNFft, twiddleHost); // [kNFft, 2] — same table production builds
    rt::Tensor fftTwiddle({kNFft, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice(fftTwiddle, twiddleHost);

    // Pre-zero the mag buffer (caller contract; the AOT GEMM has no residue handling).
    rt::Tensor magFp16({N_pad, kKPad}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaMemsetAsync(magFp16.rawPointer(), 0, static_cast<size_t>(N_pad) * kKPad * sizeof(__half), stream));

    kernel::stftR2C512FusedMagsq(framed, fftTwiddle, magFp16, T_out, kKPad, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<__half> const gpuOut = copyDeviceToHost<__half>(magFp16);

    // Host reference: naive DFT, take |·|^2 of the first kNFreq bins, cast through
    // F16 to match the kernel's fused quantisation.
    float absMax = 0.0f;
    for (int32_t t = 0; t < T_out; ++t)
    {
        for (int32_t k = 0; k < kNFreq; ++k)
        {
            double re = 0.0, im = 0.0;
            for (int32_t n = 0; n < kNFft; ++n)
            {
                double const ang = -2.0 * M_PI * static_cast<double>(k) * static_cast<double>(n) / kNFft;
                double const x = framedHost[static_cast<size_t>(t) * kNFft + n];
                re += x * std::cos(ang);
                im += x * std::sin(ang);
            }
            float const refF16 = __half2float(__float2half_rn(static_cast<float>(re * re + im * im)));
            float const got = __half2float(gpuOut[static_cast<size_t>(t) * kKPad + k]);
            float const d = std::abs(got - refF16);
            if (d > absMax)
            {
                absMax = d;
            }
        }
    }

    // Verify the K-axis padding stays zero (cols [kNFreq, kKPad)) ...
    bool kPadZero = true;
    for (int32_t t = 0; t < T_out; ++t)
    {
        for (int32_t k = kNFreq; k < kKPad; ++k)
        {
            if (__half2float(gpuOut[static_cast<size_t>(t) * kKPad + k]) != 0.0f)
            {
                kPadZero = false;
            }
        }
    }
    // ... and the N-axis padding (rows [T_out, N_pad)).
    bool nPadZero = true;
    for (int32_t t = T_out; t < N_pad; ++t)
    {
        for (int32_t k = 0; k < kKPad; ++k)
        {
            if (__half2float(gpuOut[static_cast<size_t>(t) * kKPad + k]) != 0.0f)
            {
                nPadZero = false;
            }
        }
    }

    std::cout << "[stftR2C512FusedMagsq] abs_max=" << absMax << ", k_pad_zero=" << kPadZero
              << ", n_pad_zero=" << nPadZero << std::endl;
    // Largest magsq is the cosine frame's bin-5 = 16384; F16 ULP there is ~16, so
    // 100 bounds the fp32-FFT-vs-double-DFT residual plus one F16 rounding step.
    EXPECT_LE(absMax, 100.0f);
    EXPECT_TRUE(kPadZero) << "K-padding cols [257, 272) must stay zero.";
    EXPECT_TRUE(nPadZero) << "N-padding rows [T_out, N_pad) must stay zero.";

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// 3. melLinearGemmFp16inFp32out
// ============================================================================
//
// Real-shape problem (M=nMel=128, K_pad=272, N_pad=128 for T_out=16). FP16
// K-major filter + FP16 K-major mag (K-padding zeroed), compared to a triple-loop
// host SGEMM over the active [M, T_out] window. The C output is FP32 — the
// Parakeet-specific deviation — so the reference is NOT re-quantised to F16 and
// the tolerance can be tighter than the Whisper FP16-out path.

TEST(AudioFbankParakeetPerKernelTest, MelLinearGemmFp16inFp32out)
{
    // The CuTe DSL gemm module is shared process-wide; skip if not built or the
    // GPU has no compiled variant. Check BEFORE creating the stream so the
    // GTEST_SKIP early return does not leak it.
#ifndef CUTE_DSL_GEMM_ENABLED
    GTEST_SKIP() << "CuTe DSL GEMM not enabled in this build (rebuild with -DENABLE_CUTE_DSL=gemm).";
#elif !defined(CUTE_DSL_GEMM_BLACKWELL_SMALL_FP16IN_FP32OUT_ENABLED)
    GTEST_SKIP() << "FP16-in/FP32-out GEMM variant (gemm_blackwell_small_fp16in_fp32out) not compiled "
                    "in this artifact.";
#else
    if (!CuteDslGemmRunner::loadKernelModule())
    {
        GTEST_SKIP() << "CuteDslGemmRunner module unavailable on this build / GPU.";
    }
#endif

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int32_t M = kNumMel;
    constexpr int32_t K = kKPad; // 272
    constexpr int32_t T_out = 16;
    constexpr int32_t N = 128;        // round_up(T_out, 128)
    constexpr int32_t nFreq = kNFreq; // active K-window 257; cols [257, 272) zero

    std::mt19937 gen(0xDEAD);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // mel filter + mag are non-negative

    // A: melFilterFp16Kmajor [M, K_pad] — fill [0, nFreq), zero-pad rest.
    std::vector<__half> aHost(static_cast<size_t>(M) * K, __half(0.0f));
    for (int32_t m = 0; m < M; ++m)
    {
        for (int32_t k = 0; k < nFreq; ++k)
        {
            aHost[static_cast<size_t>(m) * K + k] = __float2half_rn(dist(gen));
        }
    }
    // B: magFp16 [N_pad, K_pad] — fill [0, T_out) x [0, nFreq), zero-pad rest.
    std::vector<__half> bHost(static_cast<size_t>(N) * K, __half(0.0f));
    for (int32_t t = 0; t < T_out; ++t)
    {
        for (int32_t k = 0; k < nFreq; ++k)
        {
            bHost[static_cast<size_t>(t) * K + k] = __float2half_rn(dist(gen));
        }
    }

    rt::Tensor aDev({M, K}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    copyHostToDevice(aDev, aHost);
    rt::Tensor bDev({N, K}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    copyHostToDevice(bDev, bHost);
    rt::Tensor cDev({M, N}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT); // FP32 out

    kernel::melLinearGemmFp16inFp32out(aDev, bDev, cDev, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> const gpuOut = copyDeviceToHost<float>(cDev);

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
            float const got = gpuOut[static_cast<size_t>(m) * N + n];
            float const d = std::abs(got - static_cast<float>(acc));
            if (d > absMax)
            {
                absMax = d;
            }
        }
    }

    std::cout << "[melLinearGemmFp16inFp32out] abs_max=" << absMax << " (M=" << M << ", N=" << N << ", K=" << K
              << ", T_out=" << T_out << ", nFreq=" << nFreq << ")" << std::endl;
    // FP32-accumulate FP16 GEMM with K=257 active terms vs a double host loop; no
    // F16 output quantisation floor. The residual is dominated by the FP16 input
    // quantisation (deterministic given the inputs, stable across Blackwell SMs),
    // not the FP32 accumulator, so the bound holds cross-platform. Observed
    // residual ~1e-4; 1e-3 keeps ~10x headroom while still flagging a gross
    // kernel regression.
    EXPECT_LE(absMax, 1e-3f);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// 4. melStatsLnPerFeature
// ============================================================================
//
// Random positive FP32 mel power (with forced sub-floor elements to exercise the
// ln floor), compare per-bin mean and invDenom to a host reference using the same
// unbiased (N-1) variance expansion (Sx2 - Sx*mean) the kernel computes.

TEST(AudioFbankParakeetPerKernelTest, MelStatsLnPerFeature)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int32_t M = kNumMel;
    constexpr int32_t T_out = 64;
    constexpr int32_t N_pad = 128; // round_up(T_out, 128)

    std::mt19937 gen(0xBEEF);
    std::uniform_real_distribution<float> dist(-2.0f, 5.0f); // ln-domain range
    // melPower is FP32 [M, N_pad] (GEMM C is FP32 for Parakeet); [0, T_out) active,
    // [T_out, N_pad) zero (not read). Values span exp(-2)..exp(5) ~ 0.13..150.
    std::vector<float> melHost(static_cast<size_t>(M) * N_pad, 0.0f);
    for (int32_t m = 0; m < M; ++m)
    {
        for (int32_t t = 0; t < T_out; ++t)
        {
            melHost[static_cast<size_t>(m) * N_pad + t] = std::exp(dist(gen));
        }
    }
    // Force sub-floor elements (< kLogGuard) to exercise the ln floor branch.
    melHost[0] = 1e-20f;
    melHost[static_cast<size_t>(5) * N_pad + 0] = 1e-20f;

    rt::Tensor melPower({M, N_pad}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice(melPower, melHost);
    rt::Tensor mean({M}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor invDenom({M}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

    kernel::melStatsLnPerFeature(melPower, T_out, kLogGuard, kNormEps, mean, invDenom, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> const gpuMean = copyDeviceToHost<float>(mean);
    std::vector<float> const gpuInv = copyDeviceToHost<float>(invDenom);

    // Reference: per bin x = ln(power + logGuard); mean = Sx/T;
    // var = max((Sx2 - Sx*mean)/(T-1), 0); invDenom = 1/(sqrt(var) + normEps).
    float meanAbs = 0.0f;
    float invRel = 0.0f;
    for (int32_t m = 0; m < M; ++m)
    {
        double s = 0.0, s2 = 0.0;
        for (int32_t t = 0; t < T_out; ++t)
        {
            double const x = std::log(static_cast<double>(melHost[static_cast<size_t>(m) * N_pad + t]) + kLogGuard);
            s += x;
            s2 += x * x;
        }
        double const refMean = s / T_out;
        double var = (s2 - s * refMean) / (T_out - 1);
        if (var < 0.0)
        {
            var = 0.0;
        }
        double const refInv = 1.0 / (std::sqrt(var) + kNormEps);
        meanAbs = std::max(meanAbs, std::abs(gpuMean[static_cast<size_t>(m)] - static_cast<float>(refMean)));
        invRel = std::max(invRel,
            std::abs(gpuInv[static_cast<size_t>(m)] - static_cast<float>(refInv))
                / static_cast<float>(std::abs(refInv)));
    }

    std::cout << "[melStatsLnPerFeature] mean_abs=" << meanAbs << ", inv_rel=" << invRel << std::endl;
    EXPECT_LE(meanAbs, 1e-3f);
    EXPECT_LE(invRel, 1e-3f);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// 5. melNormalizeZScoreTimeFirst
// ============================================================================
//
// Random positive FP32 mel power + self-consistent mean/invDenom (same formula
// as #4), run kernel, compare the time-first [1, T_out, nMel] F16 output to a
// host z-score reference. Inputs are kept well-conditioned (no extreme outliers)
// so the z-score range and thus the F16 quantisation floor stay bounded; the ln
// floor branch itself is covered by #4.

TEST(AudioFbankParakeetPerKernelTest, MelNormalizeZScoreTimeFirst)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int32_t M = kNumMel;
    constexpr int32_t T_out = 64;
    constexpr int32_t N_pad = 128; // round_up(T_out, 128)

    std::mt19937 gen(0xF00D);
    std::uniform_real_distribution<float> dist(-2.0f, 5.0f);
    std::vector<float> melHost(static_cast<size_t>(M) * N_pad, 0.0f);
    for (int32_t m = 0; m < M; ++m)
    {
        for (int32_t t = 0; t < T_out; ++t)
        {
            melHost[static_cast<size_t>(m) * N_pad + t] = std::exp(dist(gen));
        }
    }

    // Deterministic mean/invDenom from the inputs (same unbiased N-1 stats as #4).
    std::vector<float> meanHost(static_cast<size_t>(M));
    std::vector<float> invHost(static_cast<size_t>(M));
    for (int32_t m = 0; m < M; ++m)
    {
        double s = 0.0, s2 = 0.0;
        for (int32_t t = 0; t < T_out; ++t)
        {
            double const x = std::log(static_cast<double>(melHost[static_cast<size_t>(m) * N_pad + t]) + kLogGuard);
            s += x;
            s2 += x * x;
        }
        double const mn = s / T_out;
        double var = (s2 - s * mn) / (T_out - 1);
        if (var < 0.0)
        {
            var = 0.0;
        }
        meanHost[static_cast<size_t>(m)] = static_cast<float>(mn);
        invHost[static_cast<size_t>(m)] = static_cast<float>(1.0 / (std::sqrt(var) + kNormEps));
    }

    rt::Tensor melPower({M, N_pad}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice(melPower, melHost);
    rt::Tensor mean({M}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice(mean, meanHost);
    rt::Tensor invDenom({M}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice(invDenom, invHost);
    rt::Tensor melOut({1, T_out, M}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF); // time-first

    kernel::melNormalizeZScoreTimeFirst(melPower, T_out, kLogGuard, mean, invDenom, melOut, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<__half> const gpuOut = copyDeviceToHost<__half>(melOut);

    // Reference: out[t, m] = (ln(power[m, t] + logGuard) - mean[m]) * invDenom[m],
    // written time-first [T_out, nMel], cast through F16.
    float absMax = 0.0f;
    for (int32_t t = 0; t < T_out; ++t)
    {
        for (int32_t m = 0; m < M; ++m)
        {
            double const x = std::log(static_cast<double>(melHost[static_cast<size_t>(m) * N_pad + t]) + kLogGuard);
            float const z
                = static_cast<float>((x - meanHost[static_cast<size_t>(m)]) * invHost[static_cast<size_t>(m)]);
            float const refF16AsF32 = __half2float(__float2half(z));
            float const got = __half2float(gpuOut[static_cast<size_t>(t) * M + m]);
            float const d = std::abs(got - refF16AsF32);
            if (d > absMax)
            {
                absMax = d;
            }
        }
    }

    std::cout << "[melNormalizeZScoreTimeFirst] abs_max=" << absMax << std::endl;
    // F16 LSB near the z-score range this test spans (|z| up to ~4 → ULP ~2^-9).
    // The bound is the ULP-derived limit — a tighter 1e-3 would be optimistic once
    // |z| exceeds ~2. F16 rounding is IEEE-standard, so it is platform-independent.
    // Observed residual ~4.9e-4 confirms 2e-3 holds.
    EXPECT_LE(absMax, 2e-3f);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// 6. MelStatsLnSilenceNoNaN
// ============================================================================
//
// A silent clip drives every mel bin to a constant ln(logGuard), so the variance
// expansion (Sx2 - Sx*mean) cancels to a tiny FP32 residual that can go negative.
// Without the fmaxf(var, 0) clamp that becomes sqrt(negative) = NaN → NaN
// invDenom → the whole mel row is NaN into the encoder. This test pins that the
// clamp keeps every output finite. No GEMM, no skip.

TEST(AudioFbankParakeetPerKernelTest, MelStatsLnSilenceNoNaN)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int32_t M = kNumMel;
    constexpr int32_t T_out = 64;
    constexpr int32_t N_pad = 128;

    // Silence: all mel power = 0 → x = ln(0 + logGuard) constant across time →
    // exact variance 0 → the degenerate cancellation case.
    std::vector<float> melHost(static_cast<size_t>(M) * N_pad, 0.0f);

    rt::Tensor melPower({M, N_pad}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice(melPower, melHost);
    rt::Tensor mean({M}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor invDenom({M}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

    kernel::melStatsLnPerFeature(melPower, T_out, kLogGuard, kNormEps, mean, invDenom, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> const gpuMean = copyDeviceToHost<float>(mean);
    std::vector<float> const gpuInv = copyDeviceToHost<float>(invDenom);

    float const refMean = std::log(kLogGuard);     // ~ -16.6355
    float const invUpper = 1.0f / kNormEps + 1.0f; // var >= 0 => invDenom <= 1/eps
    bool allFinite = true;
    for (int32_t m = 0; m < M; ++m)
    {
        if (!std::isfinite(gpuMean[static_cast<size_t>(m)]) || !std::isfinite(gpuInv[static_cast<size_t>(m)]))
        {
            allFinite = false;
        }
        // var clamped to >= 0 => 0 < invDenom <= 1/normEps.
        EXPECT_GT(gpuInv[static_cast<size_t>(m)], 0.0f) << "bin " << m;
        EXPECT_LE(gpuInv[static_cast<size_t>(m)], invUpper) << "bin " << m;
        EXPECT_NEAR(gpuMean[static_cast<size_t>(m)], refMean, 1e-2f) << "bin " << m;
    }

    std::cout << "[MelStatsLnSilenceNoNaN] all_finite=" << allFinite << ", mean[0]=" << gpuMean[0]
              << ", invDenom[0]=" << gpuInv[0] << std::endl;
    EXPECT_TRUE(allFinite) << "silent clip produced NaN/Inf stats — fmaxf(var, 0) clamp regressed";

    CUDA_CHECK(cudaStreamDestroy(stream));
}
