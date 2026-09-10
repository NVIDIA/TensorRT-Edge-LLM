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

#include <gtest/gtest.h>

#include "common/checkMacros.h"
#include "kernels/audioAttentionKernels/gemma4AudioAttention.h"
#include "testUtils.h"

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{

// Gemma 4 audio defaults (the config the kernel is specialized for).
int constexpr kH = 8, kD = 128, kC = 12, kL = 12, kM = 24, kP = 13;
float constexpr kCap = 50.f;
float constexpr kLn2 = 0.6931471805599453f;

template <typename T>
float toFloat(T x)
{
    if constexpr (std::is_same_v<T, half>)
    {
        return __half2float(x);
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return __bfloat162float(x);
    }
    else
    {
        return x; // float
    }
}

template <typename T>
T fromFloat(float x)
{
    if constexpr (std::is_same_v<T, half>)
    {
        return __float2half(x);
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return __float2bfloat16(x);
    }
    else
    {
        return x; // float
    }
}

inline float softplusf(float x)
{
    return x > 20.f ? x : std::log1p(std::exp(x));
}

//! \brief Map the element type to the TensorRT data type carried by an rt::Tensor.
template <typename T>
nvinfer1::DataType trtDataType()
{
    if constexpr (std::is_same_v<T, half>)
    {
        return nvinfer1::DataType::kHALF;
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return nvinfer1::DataType::kBF16;
    }
    else
    {
        return nvinfer1::DataType::kFLOAT; // float
    }
}

//! \brief CPU fp32 reference for the Gemma 4 audio attention body. Operates on the
//! low-precision-rounded inputs (upcast to fp32), mirroring the kernel's view of the data,
//! so the comparison isolates the kernel's internal low-precision accumulation.
void reference(std::vector<float> const& q, std::vector<float> const& k, std::vector<float> const& v,
    std::vector<float> const& gamma, std::vector<float> const& rel, std::vector<char> const& valid,
    std::vector<float>& out, int B, int S)
{
    int const H = kH, D = kD, C = kC, L = kL, M = kM, P = kP;
    int const nb = (S + C - 1) / C;
    float const qScalar = static_cast<float>(std::pow(static_cast<double>(D), -0.5) / kLn2);
    float const kScale = static_cast<float>(std::log1p(std::exp(1.0)) / kLn2);

    std::vector<float> sp(D);
    for (int r = 0; r < D; ++r)
    {
        sp[r] = softplusf(gamma[r]);
    }
    auto Q = [&](int b, int s, int h, int r) {
        return q[((static_cast<size_t>(b * S + s) * H + h) * D + r)] * qScalar * sp[r];
    };
    auto K = [&](int b, int s, int h, int r) { return k[((static_cast<size_t>(b * S + s) * H + h) * D + r)] * kScale; };
    auto V = [&](int b, int s, int h, int r) { return v[((static_cast<size_t>(b * S + s) * H + h) * D + r)]; };
    auto Rk = [&](int t, int h, int r) { return rel[(static_cast<size_t>(t * H + h) * D + r)]; };

    std::fill(out.begin(), out.end(), 0.f);
    for (int b = 0; b < B; ++b)
    {
        for (int h = 0; h < H; ++h)
        {
            for (int n = 0; n < nb; ++n)
            {
                std::vector<float> relraw(C * P, 0.f);
                for (int a = 0; a < C; ++a)
                {
                    int i = n * C + a;
                    if (i >= S)
                    {
                        continue;
                    }
                    for (int t = 0; t < P; ++t)
                    {
                        float acc = 0.f;
                        for (int r = 0; r < D; ++r)
                        {
                            acc += Q(b, i, h, r) * Rk(t, h, r);
                        }
                        relraw[a * P + t] = acc;
                    }
                }
                for (int a = 0; a < C; ++a)
                {
                    int i = n * C + a;
                    if (i >= S)
                    {
                        continue;
                    }
                    std::vector<float> lg(M);
                    for (int m = 0; m < M; ++m)
                    {
                        int j = n * C - L + m;
                        int dist = i - j;
                        bool inWindow = dist >= 0 && dist < L;
                        bool jin = j >= 0 && j < S;
                        bool vj = jin && valid[static_cast<size_t>(b) * S + j];
                        if (inWindow && jin && vj)
                        {
                            float sc = 0.f;
                            for (int r = 0; r < D; ++r)
                            {
                                sc += Q(b, i, h, r) * K(b, j, h, r);
                            }
                            int p = a * M + m, as = p / (M + 1), ts = p % (M + 1);
                            float sr = (as < C && ts < P) ? relraw[as * P + ts] : 0.f;
                            lg[m] = kCap * std::tanh((sc + sr) / kCap);
                        }
                        else
                        {
                            lg[m] = -1e9f;
                        }
                    }
                    float mx = -1e30f;
                    for (float x : lg)
                    {
                        mx = std::max(mx, x);
                    }
                    float sum = 0.f;
                    for (float& x : lg)
                    {
                        x = std::exp(x - mx);
                        sum += x;
                    }
                    for (float& x : lg)
                    {
                        x /= sum;
                    }
                    for (int r = 0; r < D; ++r)
                    {
                        float acc = 0.f;
                        for (int m = 0; m < M; ++m)
                        {
                            int j = n * C - L + m;
                            if (j >= 0 && j < S)
                            {
                                acc += lg[m] * V(b, j, h, r);
                            }
                        }
                        out[((static_cast<size_t>(b * S + i) * H + h) * D + r)] = acc;
                    }
                }
            }
        }
    }
}

enum class Mask
{
    AllValid,
    RightPadded
};

//! \brief Build inputs, round to T, run the kernel, and compare against the fp32 reference.
template <typename T>
void runCase(int B, int S, Mask mask, float atol)
{
    size_t const elems = static_cast<size_t>(B) * S * kH * kD;
    std::vector<float> q(elems), k(elems), v(elems), rel(static_cast<size_t>(kP) * kH * kD), gamma(kD);
    uniformFloatInitialization(q, -1.f, 1.f);
    uniformFloatInitialization(k, -1.f, 1.f);
    uniformFloatInitialization(v, -1.f, 1.f);
    uniformFloatInitialization(rel, -1.f, 1.f);
    uniformFloatInitialization(gamma, -0.2f, 0.2f);

    std::vector<char> valid(static_cast<size_t>(B) * S, 1);
    if (mask == Mask::RightPadded)
    {
        for (int b = 0; b < B; ++b)
        {
            int len = std::max(1, (S * (b + 1)) / (B + 1));
            for (int s = len; s < S; ++s)
            {
                valid[static_cast<size_t>(b) * S + s] = 0;
            }
        }
    }

    // Round inputs to T (what both kernel and reference see); gamma stays fp32.
    auto roundView = [](std::vector<float> const& f) {
        std::vector<float> o(f.size());
        for (size_t i = 0; i < f.size(); ++i)
        {
            o[i] = toFloat<T>(fromFloat<T>(f[i]));
        }
        return o;
    };
    std::vector<float> ref(elems);
    reference(roundView(q), roundView(k), roundView(v), gamma, roundView(rel), valid, ref, B, S);

    // Host buffers in the kernel's element type (gamma stays fp32, valid stays 1-byte bool).
    auto toHostT = [](std::vector<float> const& f) {
        std::vector<T> h(f.size());
        for (size_t i = 0; i < f.size(); ++i)
        {
            h[i] = fromFloat<T>(f[i]);
        }
        return h;
    };
    std::vector<T> hq = toHostT(q), hk = toHostT(k), hv = toHostT(v), hrel = toHostT(rel);

    // Owning GPU tensors; each carries its shape and element type.
    nvinfer1::DataType const dt = trtDataType<T>();
    auto onGpu = rt::DeviceType::kGPU;
    rt::Tensor qT(rt::Coords{B, S, kH, kD}, onGpu, dt);
    rt::Tensor kT(rt::Coords{B, S, kH, kD}, onGpu, dt);
    rt::Tensor vT(rt::Coords{B, S, kH, kD}, onGpu, dt);
    rt::Tensor outT(rt::Coords{B, S, kH, kD}, onGpu, dt);
    rt::Tensor gammaT(rt::Coords{kD}, onGpu, nvinfer1::DataType::kFLOAT);
    rt::Tensor relT(rt::Coords{kP, kH, kD}, onGpu, dt);
    rt::Tensor validT(rt::Coords{B, S}, onGpu, nvinfer1::DataType::kBOOL);

    auto upload = [](rt::Tensor& t, void const* src, size_t bytes) {
        CUDA_CHECK(cudaMemcpy(t.rawPointer(), src, bytes, cudaMemcpyHostToDevice));
    };
    upload(qT, hq.data(), hq.size() * sizeof(T));
    upload(kT, hk.data(), hk.size() * sizeof(T));
    upload(vT, hv.data(), hv.size() * sizeof(T));
    upload(relT, hrel.data(), hrel.size() * sizeof(T));
    upload(gammaT, gamma.data(), gamma.size() * sizeof(float));
    upload(validT, valid.data(), valid.size() * sizeof(char)); // char 0/1 is a valid bool byte

    Gemma4AudioAttentionParams params{kC, kL, kM, kCap};
    cudaStream_t stream{nullptr};
    gemma4AudioAttentionForward(qT, kT, vT, gammaT, relT, validT, outT, params, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> outHost(elems);
    CUDA_CHECK(cudaMemcpy(outHost.data(), outT.rawPointer(), elems * sizeof(T), cudaMemcpyDeviceToHost));
    float maxErr = 0.f;
    for (size_t i = 0; i < elems; ++i)
    {
        maxErr = std::max(maxErr, std::fabs(toFloat<T>(outHost[i]) - ref[i]));
    }
    EXPECT_LT(maxErr, atol) << "B=" << B << " S=" << S << " mask=" << static_cast<int>(mask)
                            << " max_abs_err=" << maxErr;
}

} // namespace

TEST(Gemma4AudioAttention, Fp32ShapeSweep)
{
    // fp32 is the HF-faithful, full-precision path: kernel vs CPU fp32 reference
    // differs only by summation order and tanh/exp approximation (round-off).
    for (int S : {1, 12, 13, 24, 25, 47, 96})
    {
        for (Mask m : {Mask::AllValid, Mask::RightPadded})
        {
            runCase<float>(2, S, m, 1e-3f);
        }
    }
}

TEST(Gemma4AudioAttention, Fp16ShapeSweep)
{
    for (int S : {1, 12, 13, 24, 25, 47, 96})
    {
        for (Mask m : {Mask::AllValid, Mask::RightPadded})
        {
            runCase<half>(2, S, m, 5e-3f);
        }
    }
}

TEST(Gemma4AudioAttention, Bf16ShapeSweep)
{
    for (int S : {1, 12, 13, 24, 25, 47, 96})
    {
        for (Mask m : {Mask::AllValid, Mask::RightPadded})
        {
            runCase<__nv_bfloat16>(2, S, m, 2.5e-2f);
        }
    }
}
