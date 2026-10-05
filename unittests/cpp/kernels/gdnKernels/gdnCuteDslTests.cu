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

#ifdef CUTE_DSL_GDN_ENABLED

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include "common/cudaUtils.h"
#include "kernels/gdnKernels/cuteDslGDNRunner.h"
#include "kernels/gdnKernels/gdnKernelUtils.cuh"
#include "testUtils.h"

using namespace trt_edgellm;

// ---------------------------------------------------------------------------
// SM helpers
// ---------------------------------------------------------------------------

/** True if the SM version supports an optimized Blackwell GDN prefill kernel. */
static inline bool isBlackwellSM(int32_t sm)
{
    return sm == 100 || sm == 101 || sm == 110 || sm == 120 || sm == 121;
}

static inline bool isBlackwellGeforceSM(int32_t sm)
{
    return sm == 120 || sm == 121;
}

static void* allocTensorMapScratch(int32_t smVersion)
{
    if (!isBlackwellGeforceSM(smVersion))
        return nullptr;
    void* scratch = nullptr;
    size_t const bytes = static_cast<size_t>(CuteDslGDNRunner::kBlackwellGeforceMaxSMCount)
        * CuteDslGDNRunner::kBlackwellGeforceTensorMapDescriptorBytes;
    CUDA_CHECK(cudaMalloc(&scratch, bytes));
    return scratch;
}

/** One step of a seeded LCG PRNG; returns a float in [-0.5, 0.5). */
static float lcgStep(uint32_t& s)
{
    s = s * 1664525u + 1013904223u;
    return (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) - 0.5f;
}

/**
 * Allocate a [N+1] int32 device buffer and compute cu_seqlens from context_lengths.
 * Caller must cudaFree the returned pointer.
 */
static void* allocCuSeqlens(void* d_context_lengths, int32_t n, cudaStream_t stream = nullptr)
{
    void* d_cu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cu, static_cast<size_t>(n + 1) * sizeof(int32_t)));
    launchGdnCalCuSeqLens(d_context_lengths, d_cu, n, stream);
    return d_cu;
}

namespace
{

template <typename T>
class PinnedBuffer
{
public:
    explicit PinnedBuffer(size_t const count)
        : mCount(count)
    {
        CUDA_CHECK(cudaMallocHost(&mData, count * sizeof(T)));
    }

    explicit PinnedBuffer(std::vector<T> const& values)
        : PinnedBuffer(values.size())
    {
        std::copy(values.begin(), values.end(), mData);
    }

    ~PinnedBuffer()
    {
        if (mData != nullptr)
        {
            (void) cudaFreeHost(mData);
        }
    }

    PinnedBuffer(PinnedBuffer const&) = delete;
    PinnedBuffer& operator=(PinnedBuffer const&) = delete;

    T* data() const noexcept
    {
        return mData;
    }

    size_t size() const noexcept
    {
        return mCount;
    }

    T* begin() const noexcept
    {
        return mData;
    }

    T& operator[](size_t const index) const noexcept
    {
        return mData[index];
    }

private:
    T* mData{};
    size_t mCount{};
};

static inline float halfToFloat(__half h)
{
    return __half2float(h);
}

static inline __half floatToHalf(float f)
{
    return __float2half_rn(f);
}

/** softplus(x) = log(1+exp(beta*x))/beta with beta=1, cap at threshold (linear above). */
static float softplus(float x, float beta = 1.f, float threshold = 20.f)
{
    float bx = beta * x;
    if (bx <= threshold)
        return (1.f / beta) * std::log(1.f + std::exp(bx));
    return x;
}

/**
 * CPU reference for GDN decode (non-varlen): q/k [n,1,h,k], v [n,1,hv,v], a/b [n,1,hv],
 * A_log [hv], dt_bias [hv], h0 [n,hv,k,v] batch-dense. Writes o [n,1,hv,v].
 * Matches Python: scale = 1/sqrt(k), use_qk_l2norm=true.
 */
static void gdnDecodeReference(float const* q, float const* k, float const* v, float const* a, float const* b,
    float const* A_log, float const* dt_bias, float* h0, float* o_ref, int32_t n, int32_t h, int32_t hv, int32_t k_dim,
    int32_t v_dim)
{
    float const scale = 1.f / std::sqrt(static_cast<float>(k_dim));
    int32_t const hk = h * k_dim;
    int32_t const hvv = hv * v_dim;
    int32_t const kv = k_dim * v_dim;

    for (int32_t i_n = 0; i_n < n; ++i_n)
    {
        for (int32_t i_hv = 0; i_hv < hv; ++i_hv)
        {
            int32_t const i_h = h > 0 ? i_hv / (hv / h) : 0;
            std::vector<float> H(k_dim * v_dim);
            for (int32_t ik = 0; ik < k_dim; ++ik)
                for (int32_t iv = 0; iv < v_dim; ++iv)
                    H[ik * v_dim + iv] = h0[(i_n * hv + i_hv) * kv + ik * v_dim + iv];

            int32_t const q_off = i_n * hk + i_h * k_dim;
            int32_t const v_off = i_n * hvv + i_hv * v_dim;
            int32_t const ab_off = i_n * hv + i_hv;

            float nq = 1e-6f, nk = 1e-6f;
            for (int32_t i = 0; i < k_dim; ++i)
            {
                float qv = q[q_off + i], kv = k[q_off + i];
                nq += qv * qv;
                nk += kv * kv;
            }
            nq = std::sqrt(nq);
            nk = std::sqrt(nk);
            std::vector<float> q_eff(k_dim), k_eff(k_dim);
            for (int32_t i = 0; i < k_dim; ++i)
            {
                q_eff[i] = (q[q_off + i] / nq) * scale;
                k_eff[i] = k[q_off + i] / nk;
            }

            float const a_val = a[ab_off], b_val = b[ab_off];
            float const A_val = A_log[i_hv], dt_val = dt_bias[i_hv];
            float const sp = softplus(a_val + dt_val, 1.f, 20.f);
            float const g = std::exp(-std::exp(A_val) * sp);
            float const beta = 1.f / (1.f + std::exp(-b_val));

            std::vector<float> H_gated(k_dim * v_dim);
            for (int32_t i = 0; i < k_dim * v_dim; ++i)
                H_gated[i] = H[i] * g;

            std::vector<float> corr(v_dim);
            for (int32_t iv = 0; iv < v_dim; ++iv)
            {
                float dot = 0.f;
                for (int32_t ik = 0; ik < k_dim; ++ik)
                    dot += H_gated[ik * v_dim + iv] * k_eff[ik];
                corr[iv] = (v[v_off + iv] - dot) * beta;
            }

            for (int32_t ik = 0; ik < k_dim; ++ik)
                for (int32_t iv = 0; iv < v_dim; ++iv)
                    H[ik * v_dim + iv] = H_gated[ik * v_dim + iv] + k_eff[ik] * corr[iv];

            for (int32_t iv = 0; iv < v_dim; ++iv)
            {
                float dot = 0.f;
                for (int32_t ik = 0; ik < k_dim; ++ik)
                    dot += H[ik * v_dim + iv] * q_eff[ik];
                o_ref[i_n * hvv + i_hv * v_dim + iv] = dot;
            }
            for (int32_t ik = 0; ik < k_dim; ++ik)
                for (int32_t iv = 0; iv < v_dim; ++iv)
                    h0[(i_n * hv + i_hv) * kv + ik * v_dim + iv] = H[ik * v_dim + iv];
        }
    }
}

/**
 * CPU reference for GDN prefill: same math as kernel. context_lengths[i] = valid token count for batch row i.
 */
static void gdnPrefillReference(float const* q, float const* k, float const* v, float const* a, float const* b,
    float const* A_log, float const* dt_bias, float* h0, float* o_ref, int32_t n, int32_t seq_len, int32_t h,
    int32_t hv, int32_t k_dim, int32_t v_dim, int32_t const* context_lengths)
{
    float const scale = 1.f / std::sqrt(static_cast<float>(k_dim));
    int32_t const t_hk = seq_len * h * k_dim;
    int32_t const t_hvv = seq_len * hv * v_dim;
    int32_t const t_hv = seq_len * hv;
    int32_t const kv = k_dim * v_dim;

    for (int32_t i_n = 0; i_n < n; ++i_n)
    {
        int32_t const max_t = context_lengths[i_n];
        for (int32_t i_hv = 0; i_hv < hv; ++i_hv)
        {
            int32_t const i_h = h > 0 ? i_hv / (hv / h) : 0;
            std::vector<float> H(k_dim * v_dim);
            for (int32_t ik = 0; ik < k_dim; ++ik)
                for (int32_t iv = 0; iv < v_dim; ++iv)
                    H[ik * v_dim + iv] = h0[(i_n * hv + i_hv) * kv + ik * v_dim + iv];

            for (int32_t t = 0; t < seq_len; ++t)
            {
                int32_t const o_base = i_n * t_hvv + t * hv * v_dim + i_hv * v_dim;
                if (t >= max_t)
                {
                    for (int32_t iv = 0; iv < v_dim; ++iv)
                        o_ref[o_base + iv] = 0.f;
                    continue;
                }

                int32_t const q_off = i_n * t_hk + t * h * k_dim + i_h * k_dim;
                int32_t const v_off = i_n * t_hvv + t * hv * v_dim + i_hv * v_dim;
                int32_t const ab_off = i_n * t_hv + t * hv + i_hv;

                float nq = 1e-6f, nk = 1e-6f;
                for (int32_t i = 0; i < k_dim; ++i)
                {
                    nq += q[q_off + i] * q[q_off + i];
                    nk += k[q_off + i] * k[q_off + i];
                }
                nq = std::sqrt(nq);
                nk = std::sqrt(nk);
                std::vector<float> q_eff(k_dim), k_eff(k_dim);
                for (int32_t i = 0; i < k_dim; ++i)
                {
                    q_eff[i] = (q[q_off + i] / nq) * scale;
                    k_eff[i] = k[q_off + i] / nk;
                }

                float const a_val = a[ab_off], b_val = b[ab_off];
                float const A_val = A_log[i_hv], dt_val = dt_bias[i_hv];
                float const sp = softplus(a_val + dt_val, 1.f, 20.f);
                float const g = std::exp(-std::exp(A_val) * sp);
                float const beta = 1.f / (1.f + std::exp(-b_val));

                for (int32_t i = 0; i < k_dim * v_dim; ++i)
                    H[i] *= g;

                std::vector<float> corr(v_dim);
                for (int32_t iv = 0; iv < v_dim; ++iv)
                {
                    float dot = 0.f;
                    for (int32_t ik = 0; ik < k_dim; ++ik)
                        dot += H[ik * v_dim + iv] * k_eff[ik];
                    corr[iv] = (v[v_off + iv] - dot) * beta;
                }
                for (int32_t ik = 0; ik < k_dim; ++ik)
                    for (int32_t iv = 0; iv < v_dim; ++iv)
                        H[ik * v_dim + iv] += k_eff[ik] * corr[iv];

                for (int32_t iv = 0; iv < v_dim; ++iv)
                {
                    float dot = 0.f;
                    for (int32_t ik = 0; ik < k_dim; ++ik)
                        dot += H[ik * v_dim + iv] * q_eff[ik];
                    o_ref[o_base + iv] = dot;
                }
            }
            for (int32_t ik = 0; ik < k_dim; ++ik)
                for (int32_t iv = 0; iv < v_dim; ++iv)
                    h0[(i_n * hv + i_hv) * kv + ik * v_dim + iv] = H[ik * v_dim + iv];
        }
    }
}

void runGDNDecodeTest()
{
    // Test config: AOT supports dynamic shape; use arbitrary dims.
    int32_t const n = 4;
    int32_t const h = 8;
    int32_t const hv = 8;
    int32_t const k = 128;
    int32_t const v = 128;

    size_t const qkvLen = static_cast<size_t>(n) * 1 * h * k;
    size_t const vLen = static_cast<size_t>(n) * 1 * hv * v;
    size_t const abLen = static_cast<size_t>(n) * 1 * hv;
    size_t const h0Len = static_cast<size_t>(n) * hv * k * v;
    size_t const oLen = static_cast<size_t>(n) * 1 * hv * v;

    size_t const qkvBytes = qkvLen * sizeof(half);
    size_t const vBytes = vLen * sizeof(half);
    size_t const abBytes = abLen * sizeof(half);
    size_t const A_logBytes = static_cast<size_t>(hv) * sizeof(float);
    size_t const dt_biasBytes = static_cast<size_t>(hv) * sizeof(half);
    size_t const h0Bytes = h0Len * sizeof(float);
    size_t const oBytes = oLen * sizeof(half);

    std::vector<float> h_q(qkvLen), h_k(qkvLen), h_v(vLen), h_a(abLen), h_b(abLen);
    std::vector<float> h_A_log(hv), h_dt_bias(hv), h_h0(h0Len);
    for (size_t i = 0; i < qkvLen; ++i)
        h_q[i] = 0.1f * (1.f + static_cast<float>(i % 5));
    for (size_t i = 0; i < qkvLen; ++i)
        h_k[i] = 0.1f * (1.f + static_cast<float>((i + 1) % 5));
    for (size_t i = 0; i < vLen; ++i)
        h_v[i] = 0.1f * (1.f + static_cast<float>((i + 2) % 5));
    for (size_t i = 0; i < abLen; ++i)
    {
        h_a[i] = 0.25f * (static_cast<float>(i % 5) - 2.f);
        h_b[i] = 0.25f * (static_cast<float>((i + 1) % 5) - 2.f);
    }
    for (int32_t i = 0; i < hv; ++i)
    {
        h_A_log[i] = -2.f + 0.25f * (i % 4);
        h_dt_bias[i] = 0.02f * (i + 1);
    }
    for (size_t i = 0; i < h0Len; ++i)
        h_h0[i] = 0.01f * (1.f + static_cast<float>(i % 10));

    std::vector<half> h_q_half(qkvLen), h_k_half(qkvLen), h_v_half(vLen), h_a_half(abLen), h_b_half(abLen),
        h_dt_half(hv);
    for (size_t i = 0; i < qkvLen; ++i)
        h_q_half[i] = floatToHalf(h_q[i]);
    for (size_t i = 0; i < qkvLen; ++i)
        h_k_half[i] = floatToHalf(h_k[i]);
    for (size_t i = 0; i < vLen; ++i)
        h_v_half[i] = floatToHalf(h_v[i]);
    for (size_t i = 0; i < abLen; ++i)
    {
        h_a_half[i] = floatToHalf(h_a[i]);
        h_b_half[i] = floatToHalf(h_b[i]);
    }
    for (int32_t i = 0; i < hv; ++i)
        h_dt_half[i] = floatToHalf(h_dt_bias[i]);

    void* d_q = nullptr;
    void* d_k = nullptr;
    void* d_v = nullptr;
    void* d_a = nullptr;
    void* d_b = nullptr;
    void* d_A_log = nullptr;
    void* d_dt_bias = nullptr;
    void* d_h0_source = nullptr;
    void* d_context_lengths = nullptr;
    void* d_o = nullptr;

    CUDA_CHECK(cudaMalloc(&d_q, qkvBytes));
    CUDA_CHECK(cudaMalloc(&d_k, qkvBytes));
    CUDA_CHECK(cudaMalloc(&d_v, vBytes));
    CUDA_CHECK(cudaMalloc(&d_a, abBytes));
    CUDA_CHECK(cudaMalloc(&d_b, abBytes));
    CUDA_CHECK(cudaMalloc(&d_A_log, A_logBytes));
    CUDA_CHECK(cudaMalloc(&d_dt_bias, dt_biasBytes));
    CUDA_CHECK(cudaMalloc(&d_h0_source, h0Bytes));
    CUDA_CHECK(cudaMalloc(&d_context_lengths, static_cast<size_t>(n) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_o, oBytes));

    std::vector<int32_t> h_ctx_decode(n, 1);
    CUDA_CHECK(cudaMemcpy(
        d_context_lengths, h_ctx_decode.data(), static_cast<size_t>(n) * sizeof(int32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_q, h_q_half.data(), qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k_half.data(), qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v_half.data(), vBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, h_a_half.data(), abBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b_half.data(), abBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_log, h_A_log.data(), A_logBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dt_bias, h_dt_half.data(), dt_biasBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h0_source, h_h0.data(), h0Bytes, cudaMemcpyHostToDevice));

    GDNParams params{};
    params.q = d_q;
    params.k = d_k;
    params.v = d_v;
    params.a = d_a;
    params.b = d_b;
    params.A_log = d_A_log;
    params.dt_bias = d_dt_bias;
    params.h0_source = d_h0_source;
    params.context_lengths = d_context_lengths;
    params.o = d_o;
    params.n = n;
    params.seq_len = 1;
    params.h = h;
    params.hv = hv;
    params.k_dim = k;
    params.v_dim = v;

    CuteDslGDNRunner runner;
    int ret = runner.run(params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(ret, 0) << "GDN decode run failed";

    std::vector<half> h_o_half(oLen);
    CUDA_CHECK(cudaMemcpy(h_o_half.data(), d_o, oBytes, cudaMemcpyDeviceToHost));
    std::vector<float> h_o_float(oLen);
    for (size_t i = 0; i < oLen; ++i)
        h_o_float[i] = halfToFloat(h_o_half[i]);

    std::vector<float> h0_ref(h_h0);
    std::vector<float> o_ref(oLen, 0.f);
    gdnDecodeReference(h_q.data(), h_k.data(), h_v.data(), h_a.data(), h_b.data(), h_A_log.data(), h_dt_bias.data(),
        h0_ref.data(), o_ref.data(), n, h, hv, k, v);

    float const atol = 0.2f;
    float const rtol = 0.02f;
    for (size_t i = 0; i < oLen; ++i)
    {
        EXPECT_TRUE(isclose(h_o_float[i], o_ref[i], rtol, atol))
            << "Decode output mismatch at " << i << ": got " << h_o_float[i] << ", ref " << o_ref[i];
    }

    // Second output: updated recurrent state h0 [n, hv, k, v]
    std::vector<float> h_h0_out(h0Len);
    CUDA_CHECK(cudaMemcpy(h_h0_out.data(), d_h0_source, h0Bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < h0Len; ++i)
    {
        EXPECT_TRUE(isclose(h_h0_out[i], h0_ref[i], rtol, atol))
            << "Decode h0 output mismatch at " << i << ": got " << h_h0_out[i] << ", ref " << h0_ref[i];
    }

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_A_log));
    CUDA_CHECK(cudaFree(d_dt_bias));
    CUDA_CHECK(cudaFree(d_h0_source));
    CUDA_CHECK(cudaFree(d_context_lengths));
    CUDA_CHECK(cudaFree(d_o));
}

void runGDNPrefillTest()
{
    // Detect SM version: optimized prefill runs on SM100/101/110 and SM120/121, sequential otherwise.
    int32_t const smVersion = getSMVersion();
    bool const onBlackwell = isBlackwellSM(smVersion);

    // seq_len=128 satisfies Blackwell chunk_size=128 requirement and is valid for sequential too.
    int32_t const n = 8;
    int32_t const seq_len = 128;
    int32_t const h = 8;
    int32_t const hv = 8;
    int32_t const k = 128;
    int32_t const v = 128;

    size_t const qkvLen = static_cast<size_t>(n) * seq_len * h * k;
    size_t const vLen = static_cast<size_t>(n) * seq_len * hv * v;
    size_t const abLen = static_cast<size_t>(n) * seq_len * hv;
    size_t const h0Len = static_cast<size_t>(n) * hv * k * v;
    size_t const oLen = static_cast<size_t>(n) * seq_len * hv * v;

    size_t const qkvBytes = qkvLen * sizeof(half);
    size_t const vBytes = vLen * sizeof(half);
    size_t const abBytes = abLen * sizeof(half);
    size_t const A_logBytes = static_cast<size_t>(hv) * sizeof(float);
    size_t const dt_biasBytes = static_cast<size_t>(hv) * sizeof(half);
    size_t const h0Bytes = h0Len * sizeof(float);
    size_t const oBytes = oLen * sizeof(half);

    // The Blackwell kernel uses chunk-wise matrix inversion which requires full-rank keys.
    // Periodic patterns (e.g. i%5) produce near-rank-1 key matrices => NaN in inversion.
    // Use a seeded LCG PRNG on Blackwell to generate random-looking (but deterministic) inputs.
    std::vector<float> h_q(qkvLen), h_k(qkvLen), h_v(vLen), h_a(abLen), h_b(abLen);
    std::vector<float> h_A_log(hv), h_dt_bias(hv), h_h0(h0Len);
    if (onBlackwell)
    {
        uint32_t seed = 0x42u;
        for (size_t i = 0; i < qkvLen; ++i)
            h_q[i] = lcgStep(seed) * 0.2f;
        for (size_t i = 0; i < qkvLen; ++i)
            h_k[i] = lcgStep(seed) * 0.2f;
        for (size_t i = 0; i < vLen; ++i)
            h_v[i] = lcgStep(seed) * 0.2f;
        for (size_t i = 0; i < abLen; ++i)
            h_a[i] = lcgStep(seed) * 0.5f;
        for (size_t i = 0; i < abLen; ++i)
            h_b[i] = lcgStep(seed) * 0.5f;
        for (int32_t i = 0; i < hv; ++i)
            h_A_log[i] = -2.f + 0.25f * (i % 4);
        for (int32_t i = 0; i < hv; ++i)
            h_dt_bias[i] = 0.02f * (i + 1);
        for (size_t i = 0; i < h0Len; ++i)
            h_h0[i] = lcgStep(seed) * 0.01f;
    }
    else
    {
        /* Regular inputs: q/k/v in {0.1..0.5}, a/b in {-0.5..0.5}, A_log/dt_bias/h0 simple steps. */
        for (size_t i = 0; i < qkvLen; ++i)
            h_q[i] = 0.1f * (1.f + static_cast<float>(i % 5));
        for (size_t i = 0; i < qkvLen; ++i)
            h_k[i] = 0.1f * (1.f + static_cast<float>((i + 1) % 5));
        for (size_t i = 0; i < vLen; ++i)
            h_v[i] = 0.1f * (1.f + static_cast<float>((i + 2) % 5));
        for (size_t i = 0; i < abLen; ++i)
        {
            h_a[i] = 0.25f * (static_cast<float>(i % 5) - 2.f);
            h_b[i] = 0.25f * (static_cast<float>((i + 1) % 5) - 2.f);
        }
        for (int32_t i = 0; i < hv; ++i)
        {
            h_A_log[i] = -2.f + 0.25f * (i % 4);
            h_dt_bias[i] = 0.02f * (i + 1);
        }
        for (size_t i = 0; i < h0Len; ++i)
            h_h0[i] = 0.01f * (1.f + static_cast<float>(i % 10));
    }

    std::vector<half> h_q_half(qkvLen), h_k_half(qkvLen), h_v_half(vLen), h_a_half(abLen), h_b_half(abLen),
        h_dt_half(hv);
    for (size_t i = 0; i < qkvLen; ++i)
        h_q_half[i] = floatToHalf(h_q[i]);
    for (size_t i = 0; i < qkvLen; ++i)
        h_k_half[i] = floatToHalf(h_k[i]);
    for (size_t i = 0; i < vLen; ++i)
        h_v_half[i] = floatToHalf(h_v[i]);
    for (size_t i = 0; i < abLen; ++i)
    {
        h_a_half[i] = floatToHalf(h_a[i]);
        h_b_half[i] = floatToHalf(h_b[i]);
    }
    for (int32_t i = 0; i < hv; ++i)
        h_dt_half[i] = floatToHalf(h_dt_bias[i]);

    void* d_q = nullptr;
    void* d_k = nullptr;
    void* d_v = nullptr;
    void* d_a = nullptr;
    void* d_b = nullptr;
    void* d_A_log = nullptr;
    void* d_dt_bias = nullptr;
    void* d_h0_source = nullptr;
    void* d_context_lengths = nullptr;
    void* d_o = nullptr;

    CUDA_CHECK(cudaMalloc(&d_q, qkvBytes));
    CUDA_CHECK(cudaMalloc(&d_k, qkvBytes));
    CUDA_CHECK(cudaMalloc(&d_v, vBytes));
    CUDA_CHECK(cudaMalloc(&d_a, abBytes));
    CUDA_CHECK(cudaMalloc(&d_b, abBytes));
    CUDA_CHECK(cudaMalloc(&d_A_log, A_logBytes));
    CUDA_CHECK(cudaMalloc(&d_dt_bias, dt_biasBytes));
    CUDA_CHECK(cudaMalloc(&d_h0_source, h0Bytes));
    CUDA_CHECK(cudaMalloc(&d_context_lengths, static_cast<size_t>(n) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_o, oBytes));

    std::vector<int32_t> h_ctx_prefill(n);
    int32_t const span = (seq_len > 1) ? (seq_len - 1) : 1;
    for (int32_t i = 0; i < n; ++i)
    {
        int32_t const len = seq_len - (i % span);
        h_ctx_prefill[i] = (len < 1) ? 1 : len;
    }
    CUDA_CHECK(cudaMemcpy(
        d_context_lengths, h_ctx_prefill.data(), static_cast<size_t>(n) * sizeof(int32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_q, h_q_half.data(), qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k_half.data(), qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v_half.data(), vBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, h_a_half.data(), abBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b_half.data(), abBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_log, h_A_log.data(), A_logBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dt_bias, h_dt_half.data(), dt_biasBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h0_source, h_h0.data(), h0Bytes, cudaMemcpyHostToDevice));

    void* d_cu_seqlens = nullptr;
    if (onBlackwell)
        d_cu_seqlens = allocCuSeqlens(d_context_lengths, n);

    void* d_h0_scratch = nullptr;
    if (onBlackwell)
        CUDA_CHECK(cudaMalloc(&d_h0_scratch, h0Bytes));
    void* d_tensormap_scratch = allocTensorMapScratch(smVersion);

    GDNParams params{};
    params.q = d_q;
    params.k = d_k;
    params.v = d_v;
    params.a = d_a;
    params.b = d_b;
    params.A_log = d_A_log;
    params.dt_bias = d_dt_bias;
    params.h0_source = d_h0_source;
    params.context_lengths = d_context_lengths;
    params.cu_seqlens = d_cu_seqlens;
    params.h0_scratch = d_h0_scratch;
    params.tensormap_scratch = d_tensormap_scratch;
    params.o = d_o;
    params.n = n;
    params.seq_len = seq_len;
    params.use_prefill = true;
    params.h = h;
    params.hv = hv;
    params.k_dim = k;
    params.v_dim = v;
    params.smVersion = smVersion;

    CuteDslGDNRunner runner;
    int ret = runner.run(params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(ret, 0) << "GDN prefill run failed (SM=" << smVersion
                      << ", path=" << (onBlackwell ? "Blackwell" : "Sequential") << ")";

    std::vector<half> h_o_half(oLen);
    CUDA_CHECK(cudaMemcpy(h_o_half.data(), d_o, oBytes, cudaMemcpyDeviceToHost));
    std::vector<float> h_o_float(oLen);
    for (size_t i = 0; i < oLen; ++i)
        h_o_float[i] = halfToFloat(h_o_half[i]);

    std::vector<float> h0_ref(h_h0);
    std::vector<float> o_ref(oLen, 0.f);
    gdnPrefillReference(h_q.data(), h_k.data(), h_v.data(), h_a.data(), h_b.data(), h_A_log.data(), h_dt_bias.data(),
        h0_ref.data(), o_ref.data(), n, seq_len, h, hv, k, v, h_ctx_prefill.data());

    // Blackwell uses fp16 + TF32 matrix inversion, use higher tolerance for Blackwell.
    // Sequential path uses exact fp32 ref: tight tolerance OK.
    float const atol = onBlackwell ? 5e-2f : 1e-4f;
    float const rtol = onBlackwell ? 5e-2f : 1e-4f;
    for (size_t i = 0; i < oLen; ++i)
    {
        EXPECT_TRUE(isclose(h_o_float[i], o_ref[i], rtol, atol))
            << "Prefill output mismatch at " << i << ": got " << h_o_float[i] << ", ref " << o_ref[i];
    }

    std::vector<float> h_h0_out(h0Len);
    CUDA_CHECK(cudaMemcpy(h_h0_out.data(), d_h0_source, h0Bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < h0Len; ++i)
    {
        EXPECT_TRUE(isclose(h_h0_out[i], h0_ref[i], rtol, atol))
            << "Prefill h0 mismatch at " << i << ": got " << h_h0_out[i] << ", ref " << h0_ref[i];
    }

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_A_log));
    CUDA_CHECK(cudaFree(d_dt_bias));
    CUDA_CHECK(cudaFree(d_h0_source));
    CUDA_CHECK(cudaFree(d_context_lengths));
    if (d_cu_seqlens)
        CUDA_CHECK(cudaFree(d_cu_seqlens));
    if (d_h0_scratch)
        CUDA_CHECK(cudaFree(d_h0_scratch));
    if (d_tensormap_scratch)
        CUDA_CHECK(cudaFree(d_tensormap_scratch));
    CUDA_CHECK(cudaFree(d_o));
}

/**
 * SM-aware padding test: context_lengths < seq_len for some batch items.
 * On SM100/101/110 and SM120/121 the runner dispatches to an optimized Blackwell kernel.
 * On SM80   the runner dispatches to sequential kernel (context_lengths masking).
 * Verifies: output at padding positions is 0 (or close to 0), valid positions match reference.
 */
void runGDNPrefillPaddingTest()
{
    int32_t const smVersion = getSMVersion();
    bool const onBlackwell = isBlackwellSM(smVersion);

    int32_t const n = 4;
    int32_t const seq_len = 128; // multiple of chunk_size=128
    int32_t const h = 8;
    int32_t const hv = 8;
    int32_t const k = 128;
    int32_t const v = 128;

    size_t const qkvLen = static_cast<size_t>(n) * seq_len * h * k;
    size_t const vLen = static_cast<size_t>(n) * seq_len * hv * v;
    size_t const abLen = static_cast<size_t>(n) * seq_len * hv;
    size_t const h0Len = static_cast<size_t>(n) * hv * k * v;
    size_t const oLen = static_cast<size_t>(n) * seq_len * hv * v;

    // Mixed context_lengths: [64, 128, 96, 128] — items 0 and 2 have padding.
    std::vector<int32_t> h_ctx = {64, 128, 96, 128};

    // Blackwell kernel requires full-rank keys (matrix inversion). Use seeded LCG PRNG.
    std::vector<float> h_q(qkvLen), h_k(qkvLen), h_v(vLen), h_a(abLen), h_b(abLen);
    std::vector<float> h_A_log(hv), h_dt_bias(hv), h_h0(h0Len);
    if (onBlackwell)
    {
        uint32_t seed = 0x43u;
        for (size_t i = 0; i < qkvLen; ++i)
            h_q[i] = lcgStep(seed) * 0.2f;
        for (size_t i = 0; i < qkvLen; ++i)
            h_k[i] = lcgStep(seed) * 0.2f;
        for (size_t i = 0; i < vLen; ++i)
            h_v[i] = lcgStep(seed) * 0.2f;
        for (size_t i = 0; i < abLen; ++i)
            h_a[i] = lcgStep(seed) * 0.5f;
        for (size_t i = 0; i < abLen; ++i)
            h_b[i] = lcgStep(seed) * 0.5f;
        for (int32_t i = 0; i < hv; ++i)
            h_A_log[i] = -2.f + 0.25f * (i % 4);
        for (int32_t i = 0; i < hv; ++i)
            h_dt_bias[i] = 0.02f * (i + 1);
        for (size_t i = 0; i < h0Len; ++i)
            h_h0[i] = lcgStep(seed) * 0.01f;
    }
    else
    {
        for (size_t i = 0; i < qkvLen; ++i)
            h_q[i] = 0.1f * (1.f + static_cast<float>(i % 5));
        for (size_t i = 0; i < qkvLen; ++i)
            h_k[i] = 0.1f * (1.f + static_cast<float>((i + 1) % 5));
        for (size_t i = 0; i < vLen; ++i)
            h_v[i] = 0.1f * (1.f + static_cast<float>((i + 2) % 5));
        for (size_t i = 0; i < abLen; ++i)
        {
            h_a[i] = 0.25f * (static_cast<float>(i % 5) - 2.f);
            h_b[i] = 0.25f * (static_cast<float>((i + 1) % 5) - 2.f);
        }
        for (int32_t i = 0; i < hv; ++i)
        {
            h_A_log[i] = -2.f + 0.25f * (i % 4);
            h_dt_bias[i] = 0.02f * (i + 1);
        }
        for (size_t i = 0; i < h0Len; ++i)
            h_h0[i] = 0.01f * (1.f + static_cast<float>(i % 10));
    }

    std::vector<half> h_q_h(qkvLen), h_k_h(qkvLen), h_v_h(vLen);
    std::vector<half> h_a_h(abLen), h_b_h(abLen), h_dt_h(hv);
    for (size_t i = 0; i < qkvLen; ++i)
        h_q_h[i] = floatToHalf(h_q[i]);
    for (size_t i = 0; i < qkvLen; ++i)
        h_k_h[i] = floatToHalf(h_k[i]);
    for (size_t i = 0; i < vLen; ++i)
        h_v_h[i] = floatToHalf(h_v[i]);
    for (size_t i = 0; i < abLen; ++i)
    {
        h_a_h[i] = floatToHalf(h_a[i]);
        h_b_h[i] = floatToHalf(h_b[i]);
    }
    for (int32_t i = 0; i < hv; ++i)
        h_dt_h[i] = floatToHalf(h_dt_bias[i]);

    void *d_q, *d_k, *d_v, *d_a, *d_b, *d_A_log, *d_dt_bias, *d_h0_src, *d_ctx, *d_o;
    CUDA_CHECK(cudaMalloc(&d_q, qkvLen * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k, qkvLen * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v, vLen * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_a, abLen * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, abLen * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_A_log, hv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dt_bias, hv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_h0_src, h0Len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ctx, n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_o, oLen * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q_h.data(), qkvLen * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k_h.data(), qkvLen * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v_h.data(), vLen * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, h_a_h.data(), abLen * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b_h.data(), abLen * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_log, h_A_log.data(), hv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dt_bias, h_dt_h.data(), hv * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h0_src, h_h0.data(), h0Len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ctx, h_ctx.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
    std::vector<half> const h_o_sentinel(oLen, floatToHalf(1.f));
    CUDA_CHECK(cudaMemcpy(d_o, h_o_sentinel.data(), oLen * sizeof(half), cudaMemcpyHostToDevice));

    void* d_cu_seqlens = nullptr;
    if (onBlackwell)
        d_cu_seqlens = allocCuSeqlens(d_ctx, n);

    void* d_h0_scratch = nullptr;
    if (onBlackwell)
        CUDA_CHECK(cudaMalloc(&d_h0_scratch, h0Len * sizeof(float)));
    void* d_tensormap_scratch = allocTensorMapScratch(smVersion);

    GDNParams params{};
    params.q = d_q;
    params.k = d_k;
    params.v = d_v;
    params.a = d_a;
    params.b = d_b;
    params.A_log = d_A_log;
    params.dt_bias = d_dt_bias;
    params.h0_source = d_h0_src;
    params.context_lengths = d_ctx;
    params.cu_seqlens = d_cu_seqlens;
    params.h0_scratch = d_h0_scratch;
    params.tensormap_scratch = d_tensormap_scratch;
    params.o = d_o;
    params.n = n;
    params.seq_len = seq_len;
    params.use_prefill = true;
    params.h = h;
    params.hv = hv;
    params.k_dim = k;
    params.v_dim = v;
    params.smVersion = smVersion;

    CuteDslGDNRunner runner;
    int ret = runner.run(params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(ret, 0) << "GDN prefill padding run failed (SM=" << smVersion
                      << ", path=" << (onBlackwell ? "Blackwell" : "Sequential") << ")";

    std::vector<half> h_o_h(oLen);
    CUDA_CHECK(cudaMemcpy(h_o_h.data(), d_o, oLen * sizeof(half), cudaMemcpyDeviceToHost));
    std::vector<float> h_o(oLen);
    for (size_t i = 0; i < oLen; ++i)
        h_o[i] = halfToFloat(h_o_h[i]);

    // CPU reference (respects context_lengths masking)
    std::vector<float> h0_ref(h_h0);
    std::vector<float> o_ref(oLen, 0.f);
    gdnPrefillReference(h_q.data(), h_k.data(), h_v.data(), h_a.data(), h_b.data(), h_A_log.data(), h_dt_bias.data(),
        h0_ref.data(), o_ref.data(), n, seq_len, h, hv, k, v, h_ctx.data());

    // Blackwell uses fp16 + TF32 matrix inversion, use higher tolerance for Blackwell.
    float const atol = onBlackwell ? 5e-2f : 1e-4f;
    float const rtol = onBlackwell ? 5e-2f : 1e-4f;
    size_t const hvv = static_cast<size_t>(hv) * v;
    for (int32_t b = 0; b < n; ++b)
    {
        int32_t const valid = h_ctx[static_cast<size_t>(b)];
        for (int32_t t = 0; t < seq_len; ++t)
        {
            size_t const base = (static_cast<size_t>(b) * seq_len + t) * hvv;
            if (t >= valid)
            {
                // Padding positions: output should be near zero.
                // Blackwell fp16 masking leaves residuals up to ~3e-3; allow 3e-3.
                float const pad_tol = onBlackwell ? 3e-3f : 1e-3f;
                for (size_t idx = 0; idx < hvv; ++idx)
                    EXPECT_NEAR(h_o[base + idx], 0.f, pad_tol)
                        << "Expected zero at padding b=" << b << " t=" << t << " idx=" << idx;
            }
            else
            {
                // Valid positions: must match reference within tolerance.
                for (size_t idx = 0; idx < hvv; ++idx)
                    EXPECT_TRUE(isclose(h_o[base + idx], o_ref[base + idx], rtol, atol))
                        << "Valid token mismatch b=" << b << " t=" << t << " idx=" << idx << ": got " << h_o[base + idx]
                        << ", ref " << o_ref[base + idx];
            }
        }
    }

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_A_log));
    CUDA_CHECK(cudaFree(d_dt_bias));
    CUDA_CHECK(cudaFree(d_h0_src));
    CUDA_CHECK(cudaFree(d_ctx));
    if (d_cu_seqlens)
        CUDA_CHECK(cudaFree(d_cu_seqlens));
    if (d_h0_scratch)
        CUDA_CHECK(cudaFree(d_h0_scratch));
    if (d_tensormap_scratch)
        CUDA_CHECK(cudaFree(d_tensormap_scratch));
    CUDA_CHECK(cudaFree(d_o));
}

void runGDNIndexedStateTest(
    int32_t seqLen, bool usePrefill, std::vector<int32_t> const& stateIndices, int32_t statePoolRows)
{
    int32_t const n = static_cast<int32_t>(stateIndices.size());
    int32_t constexpr h = 1;
    int32_t constexpr hv = 2;
    int32_t constexpr dim = 128;
    size_t const qkLen = static_cast<size_t>(n) * seqLen * h * dim;
    size_t const valueLen = static_cast<size_t>(n) * seqLen * hv * dim;
    size_t const abLen = static_cast<size_t>(n) * seqLen * hv;
    size_t const stateStride = static_cast<size_t>(hv) * dim * dim;
    size_t const stateLen = static_cast<size_t>(statePoolRows) * stateStride;

    uint32_t seed = 0x127u;
    std::vector<float> qHost(qkLen), kHost(qkLen), valueHost(valueLen), aHost(abLen), bHost(abLen);
    for (float& value : qHost)
    {
        value = lcgStep(seed) * 0.2F;
    }
    for (float& value : kHost)
    {
        value = lcgStep(seed) * 0.2F;
    }
    for (float& value : valueHost)
    {
        value = lcgStep(seed) * 0.2F;
    }
    for (float& value : aHost)
    {
        value = lcgStep(seed) * 0.5F;
    }
    for (float& value : bHost)
    {
        value = lcgStep(seed) * 0.5F;
    }
    std::vector<float> logA(hv), dtBiasHost(hv);
    for (int32_t head = 0; head < hv; ++head)
    {
        logA[static_cast<size_t>(head)] = -2.0F + 0.25F * (head % 4);
        dtBiasHost[static_cast<size_t>(head)] = 0.02F * (head + 1);
    }
    auto toHalf = [](std::vector<float> const& input) {
        std::vector<half> output(input.size());
        std::transform(input.begin(), input.end(), output.begin(), [](float value) { return __float2half(value); });
        return output;
    };
    std::vector<half> const q = toHalf(qHost);
    std::vector<half> const k = toHalf(kHost);
    std::vector<half> const values = toHalf(valueHost);
    std::vector<half> const a = toHalf(aHost);
    std::vector<half> const b = toHalf(bHost);
    std::vector<half> const dtBias = toHalf(dtBiasHost);
    auto roundTripHalf = [](std::vector<half> const& input, std::vector<float>& output) {
        std::transform(input.begin(), input.end(), output.begin(), [](half value) { return __half2float(value); });
    };
    roundTripHalf(q, qHost);
    roundTripHalf(k, kHost);
    roundTripHalf(values, valueHost);
    roundTripHalf(a, aHost);
    roundTripHalf(b, bHost);
    roundTripHalf(dtBias, dtBiasHost);
    std::vector<int32_t> queryLengths(n, seqLen);
    for (int32_t row = 0; row < n; ++row)
    {
        queryLengths[row] = std::max(1, seqLen - row);
    }
    std::vector<float> state(stateLen);
    for (int32_t slot = 0; slot < statePoolRows; ++slot)
    {
        size_t const begin = static_cast<size_t>(slot) * stateStride;
        for (size_t element = 0; element < stateStride; ++element)
        {
            state[begin + element] = 0.01F * (slot + 1) + 1e-6F * element;
        }
    }

    PinnedBuffer<half> const qPinned(q);
    PinnedBuffer<half> const kPinned(k);
    PinnedBuffer<half> const valuesPinned(values);
    PinnedBuffer<half> const aPinned(a);
    PinnedBuffer<half> const bPinned(b);
    PinnedBuffer<float> const logAPinned(logA);
    PinnedBuffer<half> const dtBiasPinned(dtBias);
    PinnedBuffer<float> const statePinned(state);
    PinnedBuffer<int32_t> const queryLengthsPinned(queryLengths);
    PinnedBuffer<int32_t> const stateIndicesPinned(stateIndices);
    std::vector<half> const outputSentinel(values.size(), __float2half(1.0F));
    PinnedBuffer<half> const outputSentinelPinned(outputSentinel);

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    void *dQ{}, *dK{}, *dV{}, *dA{}, *dB{}, *dLogA{}, *dDtBias{}, *dState{}, *dLengths{}, *dIndices{}, *deviceOutput{},
        *deviceScratch{}, *deviceCuSeqlens{}, *deviceTensorMapScratch{};
    CUDA_CHECK(cudaMalloc(&dQ, q.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dK, k.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dV, values.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dA, a.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dB, b.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dLogA, logA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDtBias, dtBias.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dState, state.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dLengths, queryLengths.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dIndices, stateIndices.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&deviceOutput, values.size() * sizeof(half)));
    CUDA_CHECK(cudaMemcpyAsync(dQ, qPinned.data(), q.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dK, kPinned.data(), k.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dV, valuesPinned.data(), values.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dA, aPinned.data(), a.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dB, bPinned.data(), b.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dLogA, logAPinned.data(), logA.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(dDtBias, dtBiasPinned.data(), dtBias.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(dState, statePinned.data(), state.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        dLengths, queryLengthsPinned.data(), queryLengths.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        dIndices, stateIndicesPinned.data(), stateIndices.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(deviceOutput, outputSentinelPinned.data(), outputSentinel.size() * sizeof(half),
        cudaMemcpyHostToDevice, stream));

    GDNParams params{};
    params.q = dQ;
    params.k = dK;
    params.v = dV;
    params.a = dA;
    params.b = dB;
    params.A_log = dLogA;
    params.dt_bias = dDtBias;
    params.h0_source = dState;
    params.context_lengths = dLengths;
    params.state_indices = dIndices;
    params.o = deviceOutput;
    params.n = n;
    params.seq_len = seqLen;
    params.use_prefill = usePrefill;
    params.h = h;
    params.hv = hv;
    params.k_dim = dim;
    params.v_dim = dim;
    params.state_pool_rows = statePoolRows;
    params.smVersion = getSMVersion();
    if (params.seq_len > 1 && (params.smVersion == 100 || params.smVersion == 101 || params.smVersion == 110))
    {
        CUDA_CHECK(cudaMalloc(&deviceScratch, static_cast<size_t>(n) * stateStride * sizeof(float)));
        deviceCuSeqlens = allocCuSeqlens(dLengths, n, stream);
        params.h0_scratch = deviceScratch;
        params.cu_seqlens = deviceCuSeqlens;
    }
    if (params.seq_len > 1 && isBlackwellGeforceSM(params.smVersion))
    {
        deviceTensorMapScratch = allocTensorMapScratch(params.smVersion);
        params.tensormap_scratch = deviceTensorMapScratch;
    }

    CuteDslGDNRunner runner;
    ASSERT_EQ(runner.run(params, stream), 0);
    PinnedBuffer<float> const actual(stateLen);
    PinnedBuffer<half> const output(values.size());
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), dState, actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(output.data(), deviceOutput, output.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> activeState(static_cast<size_t>(n) * stateStride, 0.0F);
    std::vector<float> expectedState(state);
    for (int32_t row = 0; row < n; ++row)
    {
        int32_t const slot = stateIndices[static_cast<size_t>(row)];
        if (slot >= 0 && slot < statePoolRows)
        {
            std::copy_n(state.begin() + static_cast<size_t>(slot) * stateStride, stateStride,
                activeState.begin() + static_cast<size_t>(row) * stateStride);
        }
    }
    std::vector<float> outputReference(valueLen, 0.0F);
    if (usePrefill)
    {
        gdnPrefillReference(qHost.data(), kHost.data(), valueHost.data(), aHost.data(), bHost.data(), logA.data(),
            dtBiasHost.data(), activeState.data(), outputReference.data(), n, seqLen, h, hv, dim, dim,
            queryLengths.data());
    }
    else
    {
        gdnDecodeReference(qHost.data(), kHost.data(), valueHost.data(), aHost.data(), bHost.data(), logA.data(),
            dtBiasHost.data(), activeState.data(), outputReference.data(), n, h, hv, dim, dim);
    }
    for (int32_t row = 0; row < n; ++row)
    {
        int32_t const slot = stateIndices[static_cast<size_t>(row)];
        if (slot >= 0 && slot < statePoolRows)
        {
            std::copy_n(activeState.begin() + static_cast<size_t>(row) * stateStride, stateStride,
                expectedState.begin() + static_cast<size_t>(slot) * stateStride);
        }
    }
    bool const datacenterBlackwellPrefill = CuteDslGDNRunner::selectBackend(params) == GDNBackend::kPrefillBlackwell;
    float const outputTolerance = datacenterBlackwellPrefill ? 5e-2F : 1e-4F;
    float const stateTolerance = datacenterBlackwellPrefill ? 5e-2F : 1e-4F;
    for (size_t element = 0; element < output.size(); ++element)
    {
        EXPECT_TRUE(isclose(__half2float(output[element]), outputReference[element], outputTolerance, outputTolerance))
            << "Output mismatch at element " << element;
    }
    for (int32_t slot = 0; slot < statePoolRows; ++slot)
    {
        auto const begin = static_cast<size_t>(slot) * stateStride;
        if (std::find(stateIndices.begin(), stateIndices.end(), slot) != stateIndices.end())
        {
            for (size_t element = 0; element < stateStride; ++element)
            {
                EXPECT_NEAR(actual[begin + element], expectedState[begin + element], stateTolerance)
                    << "Resident state mismatch for slot " << slot << " at element " << element;
            }
        }
        else
        {
            EXPECT_TRUE(std::equal(actual.begin() + begin, actual.begin() + begin + stateStride, state.begin() + begin))
                << "Unselected resident slot changed: " << slot;
        }
    }

    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dK));
    CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dLogA));
    CUDA_CHECK(cudaFree(dDtBias));
    CUDA_CHECK(cudaFree(dState));
    CUDA_CHECK(cudaFree(dLengths));
    CUDA_CHECK(cudaFree(dIndices));
    CUDA_CHECK(cudaFree(deviceOutput));
    if (deviceScratch)
    {
        CUDA_CHECK(cudaFree(deviceScratch));
    }
    if (deviceCuSeqlens)
    {
        CUDA_CHECK(cudaFree(deviceCuSeqlens));
    }
    if (deviceTensorMapScratch)
    {
        CUDA_CHECK(cudaFree(deviceTensorMapScratch));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------
// MTP (Multi-Token Processing) reference and test functions
// ---------------------------------------------------------------------------

/**
 * CPU reference for GDN MTP decode.
 *
 * Evolves the recurrent state over seq_len time steps per batch item.
 * All batch items process the same number of steps (uniform T = seq_len).
 *
 * Also fills intermediate_states_ref[i_n * seq_len + t, i_hv, k, v] (if non-null)
 * with the h-state immediately after step t.  Used to verify rollback.
 */
static void gdnDecodeMTPReference(float const* q, // [n, seq_len, h, k]
    float const* k,                               // [n, seq_len, h, k]
    float const* v,                               // [n, seq_len, hv, v]
    float const* a,                               // [n, seq_len, hv]
    float const* b,                               // [n, seq_len, hv]
    float const* A_log,                           // [hv]
    float const* dt_bias,                         // [hv]
    float* h0,                                    // [n, hv, k_dim, v_dim]  — updated in-place
    float* o_ref,                                 // [n, seq_len, hv, v_dim]
    float* intermediate_states_ref,               // [n, seq_len, hv, k_dim, v_dim] or nullptr
    int32_t n, int32_t seq_len, int32_t h, int32_t hv, int32_t k_dim, int32_t v_dim)
{
    float const scale = 1.f / std::sqrt(static_cast<float>(k_dim));
    int32_t const t_hk = seq_len * h * k_dim;
    int32_t const t_hvv = seq_len * hv * v_dim;
    int32_t const t_hv = seq_len * hv;
    int32_t const kv = k_dim * v_dim;

    for (int32_t i_n = 0; i_n < n; ++i_n)
    {
        for (int32_t i_hv = 0; i_hv < hv; ++i_hv)
        {
            int32_t const i_h = (h > 0) ? (i_hv / (hv / h)) : 0;
            std::vector<float> H(k_dim * v_dim);
            for (int32_t ik = 0; ik < k_dim; ++ik)
                for (int32_t iv = 0; iv < v_dim; ++iv)
                    H[ik * v_dim + iv] = h0[(i_n * hv + i_hv) * kv + ik * v_dim + iv];

            for (int32_t t = 0; t < seq_len; ++t)
            {
                int32_t const o_base = i_n * t_hvv + t * hv * v_dim + i_hv * v_dim;
                int32_t const q_off = i_n * t_hk + t * h * k_dim + i_h * k_dim;
                int32_t const v_off = i_n * t_hvv + t * hv * v_dim + i_hv * v_dim;
                int32_t const ab_off = i_n * t_hv + t * hv + i_hv;

                // L2-normalise q and k, apply scale.
                float nq = 1e-6f, nk = 1e-6f;
                for (int32_t i = 0; i < k_dim; ++i)
                {
                    nq += q[q_off + i] * q[q_off + i];
                    nk += k[q_off + i] * k[q_off + i];
                }
                nq = std::sqrt(nq);
                nk = std::sqrt(nk);
                std::vector<float> q_eff(k_dim), k_eff(k_dim);
                for (int32_t i = 0; i < k_dim; ++i)
                {
                    q_eff[i] = (q[q_off + i] / nq) * scale;
                    k_eff[i] = k[q_off + i] / nk;
                }

                // Gate values.
                float const a_val = a[ab_off], b_val = b[ab_off];
                float const A_val = A_log[i_hv], dt_val = dt_bias[i_hv];
                float const sp = softplus(a_val + dt_val, 1.f, 20.f);
                float const g = std::exp(-std::exp(A_val) * sp);
                float const beta = 1.f / (1.f + std::exp(-b_val));

                // Decay, delta-rule update.
                for (int32_t i = 0; i < k_dim * v_dim; ++i)
                    H[i] *= g;

                std::vector<float> corr(v_dim);
                for (int32_t iv = 0; iv < v_dim; ++iv)
                {
                    float dot = 0.f;
                    for (int32_t ik = 0; ik < k_dim; ++ik)
                        dot += H[ik * v_dim + iv] * k_eff[ik];
                    corr[iv] = (v[v_off + iv] - dot) * beta;
                }
                for (int32_t ik = 0; ik < k_dim; ++ik)
                    for (int32_t iv = 0; iv < v_dim; ++iv)
                        H[ik * v_dim + iv] += k_eff[ik] * corr[iv];

                // Output h @ q.
                for (int32_t iv = 0; iv < v_dim; ++iv)
                {
                    float dot = 0.f;
                    for (int32_t ik = 0; ik < k_dim; ++ik)
                        dot += H[ik * v_dim + iv] * q_eff[ik];
                    o_ref[o_base + iv] = dot;
                }

                // Cache intermediate state for step t (if requested).
                if (intermediate_states_ref != nullptr)
                {
                    // Layout: [n, seq_len, hv, k_dim, v_dim]
                    int32_t const interm_base = ((i_n * seq_len + t) * hv + i_hv) * kv;
                    for (int32_t ik = 0; ik < k_dim; ++ik)
                        for (int32_t iv = 0; iv < v_dim; ++iv)
                            intermediate_states_ref[interm_base + ik * v_dim + iv] = H[ik * v_dim + iv];
                }
            }

            // Write final h-state back (after last valid step).
            for (int32_t ik = 0; ik < k_dim; ++ik)
                for (int32_t iv = 0; iv < v_dim; ++iv)
                    h0[(i_n * hv + i_hv) * kv + ik * v_dim + iv] = H[ik * v_dim + iv];
        }
    }
}

/**
 * Run one MTP decode test configuration.
 *
 * @param seq_len    Number of draft tokens (T); all batch items process the same T.
 * @param with_cache Whether to allocate + verify intermediate_states.
 */
static void runGDNDecodeMTPTestConfig(int32_t seq_len, bool with_cache)
{
    int32_t const n = 4;
    int32_t const h = 8;
    int32_t const hv = 8;
    int32_t const k = 128;
    int32_t const v = 128;

    size_t const qkvLen = static_cast<size_t>(n) * seq_len * h * k;
    size_t const vLen = static_cast<size_t>(n) * seq_len * hv * v;
    size_t const abLen = static_cast<size_t>(n) * seq_len * hv;
    size_t const h0Len = static_cast<size_t>(n) * hv * k * v;
    size_t const oLen = static_cast<size_t>(n) * seq_len * hv * v;
    size_t const intermLen = with_cache ? (static_cast<size_t>(n) * seq_len * hv * k * v) : 1UL;

    size_t const qkvBytes = qkvLen * sizeof(half);
    size_t const vBytes = vLen * sizeof(half);
    size_t const abBytes = abLen * sizeof(half);
    size_t const A_logBytes = static_cast<size_t>(hv) * sizeof(float);
    size_t const dtBytes = static_cast<size_t>(hv) * sizeof(half);
    size_t const h0Bytes = h0Len * sizeof(float);
    size_t const oBytes = oLen * sizeof(half);
    size_t const intermBytes = intermLen * sizeof(float);

    // Host float32 data.
    std::vector<float> h_q(qkvLen), h_k(qkvLen), h_v(vLen), h_a(abLen), h_b(abLen);
    std::vector<float> h_A_log(hv), h_dt_bias(hv), h_h0(h0Len);
    for (size_t i = 0; i < qkvLen; ++i)
        h_q[i] = 0.1f * (1.f + static_cast<float>(i % 5));
    for (size_t i = 0; i < qkvLen; ++i)
        h_k[i] = 0.1f * (1.f + static_cast<float>((i + 1) % 5));
    for (size_t i = 0; i < vLen; ++i)
        h_v[i] = 0.1f * (1.f + static_cast<float>((i + 2) % 5));
    for (size_t i = 0; i < abLen; ++i)
    {
        h_a[i] = 0.25f * (static_cast<float>(i % 5) - 2.f);
        h_b[i] = 0.25f * (static_cast<float>((i + 1) % 5) - 2.f);
    }
    for (int32_t i = 0; i < hv; ++i)
    {
        h_A_log[i] = -2.f + 0.25f * (i % 4);
        h_dt_bias[i] = 0.02f * (i + 1);
    }
    for (size_t i = 0; i < h0Len; ++i)
        h_h0[i] = 0.01f * (1.f + static_cast<float>(i % 10));

    // FP16 converted inputs.
    std::vector<half> h_q_h(qkvLen), h_k_h(qkvLen), h_v_h(vLen);
    std::vector<half> h_a_h(abLen), h_b_h(abLen), h_dt_h(hv);
    for (size_t i = 0; i < qkvLen; ++i)
        h_q_h[i] = floatToHalf(h_q[i]);
    for (size_t i = 0; i < qkvLen; ++i)
        h_k_h[i] = floatToHalf(h_k[i]);
    for (size_t i = 0; i < vLen; ++i)
        h_v_h[i] = floatToHalf(h_v[i]);
    for (size_t i = 0; i < abLen; ++i)
        h_a_h[i] = floatToHalf(h_a[i]);
    for (size_t i = 0; i < abLen; ++i)
        h_b_h[i] = floatToHalf(h_b[i]);
    for (int32_t i = 0; i < hv; ++i)
        h_dt_h[i] = floatToHalf(h_dt_bias[i]);

    // Device allocations.
    void* d_q = nullptr;
    void* d_k = nullptr;
    void* d_v = nullptr;
    void* d_a = nullptr;
    void* d_b = nullptr;
    void* d_A_log = nullptr;
    void* d_dt = nullptr;
    void* d_h0 = nullptr;
    void* d_o = nullptr;
    void* d_interm = nullptr;

    CUDA_CHECK(cudaMalloc(&d_q, qkvBytes));
    CUDA_CHECK(cudaMalloc(&d_k, qkvBytes));
    CUDA_CHECK(cudaMalloc(&d_v, vBytes));
    CUDA_CHECK(cudaMalloc(&d_a, abBytes));
    CUDA_CHECK(cudaMalloc(&d_b, abBytes));
    CUDA_CHECK(cudaMalloc(&d_A_log, A_logBytes));
    CUDA_CHECK(cudaMalloc(&d_dt, dtBytes));
    CUDA_CHECK(cudaMalloc(&d_h0, h0Bytes));
    CUDA_CHECK(cudaMalloc(&d_o, oBytes));
    CUDA_CHECK(cudaMalloc(&d_interm, intermBytes));

    CUDA_CHECK(cudaMemcpy(d_q, h_q_h.data(), qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k_h.data(), qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v_h.data(), vBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, h_a_h.data(), abBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b_h.data(), abBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_log, h_A_log.data(), A_logBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dt, h_dt_h.data(), dtBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h0, h_h0.data(), h0Bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_o, 0, oBytes));
    CUDA_CHECK(cudaMemset(d_interm, 0, intermBytes));

    // Configure and run.
    GDNParams params{};
    params.q = d_q;
    params.k = d_k;
    params.v = d_v;
    params.a = d_a;
    params.b = d_b;
    params.A_log = d_A_log;
    params.dt_bias = d_dt;
    params.h0_source = d_h0;
    params.o = d_o;
    params.intermediate_states = d_interm;
    params.use_mtp = true;
    params.n = n;
    params.seq_len = seq_len;
    params.h = h;
    params.hv = hv;
    params.k_dim = k;
    params.v_dim = v;
    params.smVersion = getSMVersion();

    CuteDslGDNRunner runner;
    int32_t const ret = runner.run(params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(ret, 0) << "GDN MTP decode run failed";

    // Copy outputs back.
    std::vector<half> h_o_half(oLen);
    std::vector<float> h_h0_out(h0Len);
    std::vector<float> h_interm_out(intermLen, 0.f);
    CUDA_CHECK(cudaMemcpy(h_o_half.data(), d_o, oBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_h0_out.data(), d_h0, h0Bytes, cudaMemcpyDeviceToHost));
    if (with_cache)
        CUDA_CHECK(cudaMemcpy(h_interm_out.data(), d_interm, intermBytes, cudaMemcpyDeviceToHost));

    std::vector<float> h_o(oLen);
    for (size_t i = 0; i < oLen; ++i)
        h_o[i] = halfToFloat(h_o_half[i]);

    // CPU reference.
    std::vector<float> h0_ref(h_h0);
    std::vector<float> o_ref(oLen, 0.f);
    std::vector<float> interm_ref(intermLen, 0.f);
    gdnDecodeMTPReference(h_q.data(), h_k.data(), h_v.data(), h_a.data(), h_b.data(), h_A_log.data(), h_dt_bias.data(),
        h0_ref.data(), o_ref.data(), with_cache ? interm_ref.data() : nullptr, n, seq_len, h, hv, k, v);

    float const atol = 0.2f;
    float const rtol = 0.02f;

    // Verify output tokens.
    for (size_t i = 0; i < oLen; ++i)
    {
        EXPECT_TRUE(isclose(h_o[i], o_ref[i], rtol, atol))
            << "MTP output mismatch at " << i << ": got=" << h_o[i] << " ref=" << o_ref[i];
    }

    // Verify final h-state.
    for (size_t i = 0; i < h0Len; ++i)
    {
        EXPECT_TRUE(isclose(h_h0_out[i], h0_ref[i], rtol, atol))
            << "MTP h0 mismatch at " << i << ": got=" << h_h0_out[i] << " ref=" << h0_ref[i];
    }

    // Verify intermediate states (rollback support).
    if (with_cache)
    {
        for (size_t i = 0; i < intermLen; ++i)
        {
            EXPECT_TRUE(isclose(h_interm_out[i], interm_ref[i], rtol, atol))
                << "MTP intermediate state mismatch at " << i << ": got=" << h_interm_out[i]
                << " ref=" << interm_ref[i];
        }
    }

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_A_log));
    CUDA_CHECK(cudaFree(d_dt));
    CUDA_CHECK(cudaFree(d_h0));
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_interm));
}

} // namespace

TEST(GDNCuteDsl, Decode)
{
    runGDNDecodeTest();
}

TEST(GDNCuteDsl, Prefill)
{
    runGDNPrefillTest();
}

TEST(GDNCuteDsl, PrefillPadding)
{
    runGDNPrefillPaddingTest();
}

TEST(GDNCuteDsl, IndexedResidentDecode)
{
    runGDNIndexedStateTest(1, false, {2, 0}, 4);
}

TEST(GDNCuteDsl, IndexedResidentPaddedPrefill)
{
    runGDNIndexedStateTest(3, true, {3, 1}, 5);
}

TEST(GDNCuteDsl, InvalidIndexedResidentPaddedPrefill)
{
#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
    if (!isBlackwellGeforceSM(getSMVersion()))
    {
        GTEST_SKIP() << "Defensive invalid-index behavior is specific to SM120/121";
    }
    runGDNIndexedStateTest(3, true, {-1, 5}, 5);
#else
    GTEST_SKIP() << "SM120/121 optimized GDN prefill is not compiled";
#endif
}

TEST(GDNCuteDsl, IndexedStateTranspose)
{
    int32_t constexpr poolRows = 5;
    int32_t constexpr batchSize = 2;
    int32_t constexpr numHeads = 1;
    int32_t constexpr dim = 4;
    std::vector<int32_t> const indices{3, 1};
    std::vector<float> source(static_cast<size_t>(poolRows) * numHeads * dim * dim);
    for (int32_t slot = 0; slot < poolRows; ++slot)
    {
        for (int32_t row = 0; row < dim; ++row)
        {
            for (int32_t col = 0; col < dim; ++col)
            {
                source[(static_cast<size_t>(slot) * dim + row) * dim + col]
                    = static_cast<float>(slot * 100 + row * 10 + col);
            }
        }
    }

    float *deviceSource{}, *deviceGathered{};
    int32_t* deviceIndices{};
    CUDA_CHECK(cudaMalloc(&deviceSource, source.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&deviceGathered, static_cast<size_t>(batchSize) * numHeads * dim * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&deviceIndices, indices.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(deviceSource, source.data(), source.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceIndices, indices.data(), indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    launchGdnStateGatherTranspose(
        deviceSource, deviceGathered, deviceIndices, batchSize, poolRows, numHeads, dim, nullptr);
    std::vector<float> gathered(static_cast<size_t>(batchSize) * numHeads * dim * dim);
    CUDA_CHECK(cudaMemcpy(gathered.data(), deviceGathered, gathered.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        for (int32_t row = 0; row < dim; ++row)
        {
            for (int32_t col = 0; col < dim; ++col)
            {
                EXPECT_EQ(gathered[(static_cast<size_t>(batch) * dim + row) * dim + col],
                    source[(static_cast<size_t>(indices[batch]) * dim + col) * dim + row]);
            }
        }
    }

    launchGdnStateIndexedTransposeInPlace(deviceSource, deviceIndices, batchSize, poolRows, numHeads, dim, nullptr);
    std::vector<float> transposed(source.size());
    CUDA_CHECK(cudaMemcpy(transposed.data(), deviceSource, transposed.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int32_t slot = 0; slot < poolRows; ++slot)
    {
        bool const selected = std::find(indices.begin(), indices.end(), slot) != indices.end();
        for (int32_t row = 0; row < dim; ++row)
        {
            for (int32_t col = 0; col < dim; ++col)
            {
                float const expected = selected ? source[(static_cast<size_t>(slot) * dim + col) * dim + row]
                                                : source[(static_cast<size_t>(slot) * dim + row) * dim + col];
                EXPECT_EQ(transposed[(static_cast<size_t>(slot) * dim + row) * dim + col], expected);
            }
        }
    }

    CUDA_CHECK(cudaFree(deviceSource));
    CUDA_CHECK(cudaFree(deviceGathered));
    CUDA_CHECK(cudaFree(deviceIndices));
}

TEST(GDNCuteDsl, IndexedStateGather)
{
    int32_t constexpr poolRows = 4;
    int32_t constexpr batchSize = 4;
    int32_t constexpr rowElements = 8;
    std::vector<int32_t> const indices{2, -1, 0, poolRows};
    std::vector<float> source(static_cast<size_t>(poolRows) * rowElements);
    std::vector<float> dense(static_cast<size_t>(batchSize) * rowElements);
    for (size_t index = 0; index < source.size(); ++index)
    {
        source[index] = static_cast<float>(index);
    }
    PinnedBuffer<float> const sourcePinned(source);
    PinnedBuffer<int32_t> const indicesPinned(indices);
    PinnedBuffer<float> const densePinned(dense.size());

    float *deviceSource{}, *deviceDense{};
    int32_t* deviceIndices{};
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&deviceSource, source.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&deviceDense, dense.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&deviceIndices, indices.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpyAsync(
        deviceSource, sourcePinned.data(), source.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        deviceIndices, indicesPinned.data(), indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    launchGdnStateGather(deviceSource, deviceDense, deviceIndices, batchSize, 1, poolRows, 1, rowElements, stream);
    CUDA_CHECK(
        cudaMemcpyAsync(densePinned.data(), deviceDense, dense.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        for (int32_t element = 0; element < rowElements; ++element)
        {
            int32_t const residentRow = indices[batch];
            float const expected = residentRow >= 0 && residentRow < poolRows
                ? source[static_cast<size_t>(residentRow) * rowElements + element]
                : 0.0F;
            EXPECT_EQ(densePinned[static_cast<size_t>(batch) * rowElements + element], expected);
        }
    }

    CUDA_CHECK(cudaFree(deviceSource));
    CUDA_CHECK(cudaFree(deviceDense));
    CUDA_CHECK(cudaFree(deviceIndices));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

/**
 * GDN prefill with the exact Qwen3.5-4B parameters: n=1, h=16, hv=32, seq_len=164.
 * This configuration triggers grouped value attention (h_r=2) and tail masking on optimized Blackwell paths.
 */
void runGDNPrefillQwen35Test(float atol = 5e-2f, float rtol = 5e-2f, bool enablePdl = false)
{
    int32_t const smVersion = getSMVersion();
    bool const onBlackwell = isBlackwellSM(smVersion);
    if (!onBlackwell)
    {
        GTEST_SKIP() << "Qwen3.5-4B prefill test requires SM100/101/110 or SM120/121, skipping on SM" << smVersion;
        return;
    }

    int32_t const n = 1;
    int32_t const seq_len = 164; // non-multiple of 128 to test tail masking
    int32_t const h = 16;
    int32_t const hv = 32;
    int32_t const k = 128;
    int32_t const v = 128;

    size_t const qkvLen = static_cast<size_t>(n) * seq_len * h * k;
    size_t const vLen = static_cast<size_t>(n) * seq_len * hv * v;
    size_t const abLen = static_cast<size_t>(n) * seq_len * hv;
    size_t const h0Len = static_cast<size_t>(n) * hv * k * v;
    size_t const oLen = static_cast<size_t>(n) * seq_len * hv * v;

    std::vector<float> h_q(qkvLen), h_k(qkvLen), h_v(vLen), h_a(abLen), h_b(abLen);
    std::vector<float> h_A_log(hv), h_dt_bias(hv), h_h0(h0Len);
    uint32_t seed = 0x44u;
    for (size_t i = 0; i < qkvLen; ++i)
        h_q[i] = lcgStep(seed) * 0.2f;
    for (size_t i = 0; i < qkvLen; ++i)
        h_k[i] = lcgStep(seed) * 0.2f;
    for (size_t i = 0; i < vLen; ++i)
        h_v[i] = lcgStep(seed) * 0.2f;
    for (size_t i = 0; i < abLen; ++i)
        h_a[i] = lcgStep(seed) * 0.5f;
    for (size_t i = 0; i < abLen; ++i)
        h_b[i] = lcgStep(seed) * 0.5f;
    for (int32_t i = 0; i < hv; ++i)
        h_A_log[i] = -2.f + 0.25f * (i % 4);
    for (int32_t i = 0; i < hv; ++i)
        h_dt_bias[i] = 0.02f * (i + 1);
    for (size_t i = 0; i < h0Len; ++i)
        h_h0[i] = lcgStep(seed) * 0.01f;

    std::vector<half> h_q_h(qkvLen), h_k_h(qkvLen), h_v_h(vLen);
    std::vector<half> h_a_h(abLen), h_b_h(abLen), h_dt_h(hv);
    for (size_t i = 0; i < qkvLen; ++i)
        h_q_h[i] = floatToHalf(h_q[i]);
    for (size_t i = 0; i < qkvLen; ++i)
        h_k_h[i] = floatToHalf(h_k[i]);
    for (size_t i = 0; i < vLen; ++i)
        h_v_h[i] = floatToHalf(h_v[i]);
    for (size_t i = 0; i < abLen; ++i)
    {
        h_a_h[i] = floatToHalf(h_a[i]);
        h_b_h[i] = floatToHalf(h_b[i]);
    }
    for (int32_t i = 0; i < hv; ++i)
        h_dt_h[i] = floatToHalf(h_dt_bias[i]);

    std::vector<int32_t> h_ctx(n, seq_len); // full context

    void *d_q, *d_k, *d_v, *d_a, *d_b, *d_A_log, *d_dt_bias, *d_h0_src, *d_ctx, *d_o;
    CUDA_CHECK(cudaMalloc(&d_q, qkvLen * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k, qkvLen * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v, vLen * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_a, abLen * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, abLen * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_A_log, hv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dt_bias, hv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_h0_src, h0Len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ctx, n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_o, oLen * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q_h.data(), qkvLen * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k_h.data(), qkvLen * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v_h.data(), vLen * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, h_a_h.data(), abLen * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b_h.data(), abLen * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_log, h_A_log.data(), hv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dt_bias, h_dt_h.data(), hv * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h0_src, h_h0.data(), h0Len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ctx, h_ctx.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));

    void* d_cu_seqlens = allocCuSeqlens(d_ctx, n);
    void* d_h0_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_h0_scratch, h0Len * sizeof(float)));
    void* d_tensormap_scratch = allocTensorMapScratch(smVersion);

    GDNParams params{};
    params.q = d_q;
    params.k = d_k;
    params.v = d_v;
    params.a = d_a;
    params.b = d_b;
    params.A_log = d_A_log;
    params.dt_bias = d_dt_bias;
    params.h0_source = d_h0_src;
    params.context_lengths = d_ctx;
    params.cu_seqlens = d_cu_seqlens;
    params.h0_scratch = d_h0_scratch;
    params.tensormap_scratch = d_tensormap_scratch;
    params.o = d_o;
    params.n = n;
    params.seq_len = seq_len;
    params.use_prefill = true;
    params.h = h;
    params.hv = hv;
    params.k_dim = k;
    params.v_dim = v;
    params.smVersion = smVersion;
    params.enablePdl = enablePdl;

    CuteDslGDNRunner runner;
    int ret = runner.run(params, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(ret, 0) << "GDN prefill (Qwen3.5-4B params) run failed";

    std::vector<half> h_o_h(oLen);
    CUDA_CHECK(cudaMemcpy(h_o_h.data(), d_o, oLen * sizeof(half), cudaMemcpyDeviceToHost));
    std::vector<float> h_o(oLen);
    for (size_t i = 0; i < oLen; ++i)
        h_o[i] = halfToFloat(h_o_h[i]);

    // CPU reference
    std::vector<float> h0_ref(h_h0);
    std::vector<float> o_ref(oLen, 0.f);
    gdnPrefillReference(h_q.data(), h_k.data(), h_v.data(), h_a.data(), h_b.data(), h_A_log.data(), h_dt_bias.data(),
        h0_ref.data(), o_ref.data(), n, seq_len, h, hv, k, v, h_ctx.data());

    // Check output is non-zero
    float maxAbsOut = 0.f;
    for (size_t i = 0; i < oLen; ++i)
        maxAbsOut = std::max(maxAbsOut, std::abs(h_o[i]));
    printf("  GDN Qwen3.5-4B: max |output| = %.6f\n", maxAbsOut);
    EXPECT_GT(maxAbsOut, 1e-4f) << "GDN output is all zeros/near-zero";

    float maxAbsRef = 0.f;
    for (size_t i = 0; i < oLen; ++i)
        maxAbsRef = std::max(maxAbsRef, std::abs(o_ref[i]));
    printf("  Reference: max |output| = %.6f\n", maxAbsRef);

    size_t mismatches = 0;
    for (size_t i = 0; i < oLen && mismatches < 10; ++i)
    {
        if (!isclose(h_o[i], o_ref[i], rtol, atol))
        {
            printf("  Mismatch at %zu: got %.6f, ref %.6f\n", i, h_o[i], o_ref[i]);
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u) << "GDN Qwen3.5-4B output mismatches found";

    // Check state
    std::vector<float> h_h0_out(h0Len);
    CUDA_CHECK(cudaMemcpy(h_h0_out.data(), d_h0_src, h0Len * sizeof(float), cudaMemcpyDeviceToHost));
    size_t stateMismatches = 0;
    for (size_t i = 0; i < h0Len && stateMismatches < 10; ++i)
    {
        if (!isclose(h_h0_out[i], h0_ref[i], rtol, atol))
        {
            printf("  State mismatch at %zu: got %.6f, ref %.6f\n", i, h_h0_out[i], h0_ref[i]);
            ++stateMismatches;
        }
    }
    EXPECT_EQ(stateMismatches, 0u) << "GDN Qwen3.5-4B state mismatches found";

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_A_log));
    CUDA_CHECK(cudaFree(d_dt_bias));
    CUDA_CHECK(cudaFree(d_h0_src));
    CUDA_CHECK(cudaFree(d_ctx));
    CUDA_CHECK(cudaFree(d_cu_seqlens));
    CUDA_CHECK(cudaFree(d_h0_scratch));
    if (d_tensormap_scratch)
        CUDA_CHECK(cudaFree(d_tensormap_scratch));
    CUDA_CHECK(cudaFree(d_o));
}

TEST(GDNCuteDsl, PrefillQwen35)
{
    runGDNPrefillQwen35Test();
}

namespace
{

struct GdnL2NormResult
{
    std::vector<uint16_t> qInputBits;
    std::vector<uint16_t> kInputBits;
    std::vector<uint16_t> qBits;
    std::vector<uint16_t> kBits;
    std::vector<uint16_t> qPrefixGuard;
    std::vector<uint16_t> qSuffixGuard;
    std::vector<uint16_t> kPrefixGuard;
    std::vector<uint16_t> kSuffixGuard;
};

enum class GdnL2NormLaunchMode
{
    kLegacyTwoLaunch,
    kFusedPdlOff,
    kFusedPdlOn,
};

GdnL2NormResult runGdnL2NormPartialCtaCase(GdnL2NormLaunchMode mode)
{
    // Eight warps process eight rows per CTA, so nine rows exercise a partial
    // final CTA while keeping the production head dimension.
    constexpr int32_t n = 1;
    constexpr int32_t seqLen = 3;
    constexpr int32_t h = 3;
    constexpr int32_t headDim = 128;
    constexpr size_t guardElements = 32;
    constexpr uint16_t guardBits = 0xA55Au;
    size_t const activeElements = static_cast<size_t>(n) * seqLen * h * headDim;
    size_t const allocationElements = guardElements + activeElements + guardElements;

    std::vector<half> qInput(activeElements), kInput(activeElements);
    uint32_t seed = 0x925u;
    for (size_t i = 0; i < activeElements; ++i)
    {
        // Keep all rows non-zero and cover positive and negative values.
        qInput[i] = floatToHalf(0.05f + lcgStep(seed));
        kInput[i] = floatToHalf(-0.05f + lcgStep(seed));
    }

    auto runOneAllocation = [&](std::vector<half> const& input, void** allocation, half** active) {
        CUDA_CHECK(cudaMalloc(allocation, allocationElements * sizeof(uint16_t)));
        std::vector<uint16_t> initialized(allocationElements, guardBits);
        std::memcpy(initialized.data() + guardElements, input.data(), activeElements * sizeof(half));
        CUDA_CHECK(
            cudaMemcpy(*allocation, initialized.data(), initialized.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        *active = static_cast<half*>(*allocation) + guardElements;
    };

    void *qAllocation = nullptr, *kAllocation = nullptr;
    half *qActive = nullptr, *kActive = nullptr;
    runOneAllocation(qInput, &qAllocation, &qActive);
    runOneAllocation(kInput, &kAllocation, &kActive);
    auto allocationsGuard = Defer([&] {
        cudaFree(qAllocation);
        cudaFree(kAllocation);
    });

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto streamGuard = Defer([&] { cudaStreamDestroy(stream); });
    switch (mode)
    {
    case GdnL2NormLaunchMode::kLegacyTwoLaunch:
        launchGdnL2NormQK(qActive, kActive, n, seqLen, h, headDim, stream);
        break;
    case GdnL2NormLaunchMode::kFusedPdlOff:
        CUDA_CHECK(launchGdnL2NormQKFusedSm12x(qActive, kActive, n, seqLen, h, headDim, /*enablePdl=*/false, stream));
        break;
    case GdnL2NormLaunchMode::kFusedPdlOn:
        CUDA_CHECK(launchGdnL2NormQKFusedSm12x(qActive, kActive, n, seqLen, h, headDim, /*enablePdl=*/true, stream));
        break;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    auto readAllocation = [&](void* allocation, std::vector<uint16_t>& activeBits, std::vector<uint16_t>& prefix,
                              std::vector<uint16_t>& suffix) {
        std::vector<uint16_t> host(allocationElements);
        CUDA_CHECK(cudaMemcpy(host.data(), allocation, host.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        prefix.assign(host.begin(), host.begin() + guardElements);
        activeBits.assign(host.begin() + guardElements, host.begin() + guardElements + activeElements);
        suffix.assign(host.begin() + guardElements + activeElements, host.end());
    };

    GdnL2NormResult result;
    result.qInputBits.resize(activeElements);
    result.kInputBits.resize(activeElements);
    std::memcpy(result.qInputBits.data(), qInput.data(), activeElements * sizeof(uint16_t));
    std::memcpy(result.kInputBits.data(), kInput.data(), activeElements * sizeof(uint16_t));
    readAllocation(qAllocation, result.qBits, result.qPrefixGuard, result.qSuffixGuard);
    readAllocation(kAllocation, result.kBits, result.kPrefixGuard, result.kSuffixGuard);
    return result;
}

void expectGdnL2NormGuardsIntact(GdnL2NormResult const& result)
{
    constexpr uint16_t guardBits = 0xA55Au;
    auto expectGuard = [](std::vector<uint16_t> const& guard, char const* label) {
        EXPECT_TRUE(std::all_of(guard.begin(), guard.end(), [](uint16_t bits) { return bits == guardBits; })) << label;
    };
    expectGuard(result.qPrefixGuard, "Q prefix canary changed");
    expectGuard(result.qSuffixGuard, "Q suffix canary changed");
    expectGuard(result.kPrefixGuard, "K prefix canary changed");
    expectGuard(result.kSuffixGuard, "K suffix canary changed");
}

void expectNormalizedRowsFromInput(std::vector<uint16_t> const& outputBits, std::vector<uint16_t> const& inputBits,
    int32_t numRows, int32_t headDim, char const* label)
{
    constexpr float normEpsilon = 1e-6f;
    constexpr float tolerance = 2e-3f;
    ASSERT_EQ(outputBits.size(), inputBits.size());
    ASSERT_EQ(outputBits.size(), static_cast<size_t>(numRows) * headDim);
    std::vector<half> output(outputBits.size());
    std::vector<half> input(inputBits.size());
    std::memcpy(output.data(), outputBits.data(), outputBits.size() * sizeof(uint16_t));
    std::memcpy(input.data(), inputBits.data(), inputBits.size() * sizeof(uint16_t));
    for (int32_t row = 0; row < numRows; ++row)
    {
        float inputNormSq = normEpsilon;
        for (int32_t d = 0; d < headDim; ++d)
        {
            float const value = halfToFloat(input[static_cast<size_t>(row) * headDim + d]);
            inputNormSq += value * value;
        }
        float const inverseInputNorm = 1.0f / std::sqrt(inputNormSq);

        float normSq = 0.0f;
        for (int32_t d = 0; d < headDim; ++d)
        {
            size_t const index = static_cast<size_t>(row) * headDim + d;
            float const outputValue = halfToFloat(output[index]);
            float const expectedValue = halfToFloat(floatToHalf(halfToFloat(input[index]) * inverseInputNorm));
            ASSERT_TRUE(std::isfinite(outputValue)) << label << " row=" << row << " d=" << d;
            EXPECT_NEAR(outputValue, expectedValue, tolerance) << label << " row=" << row << " d=" << d;
            normSq += outputValue * outputValue;
        }
        EXPECT_NEAR(std::sqrt(normSq), 1.0f, tolerance) << label << " row=" << row;
    }
}

} // namespace

TEST(GDNCuteDsl, L2NormQKSingleProducerMatchesLegacyForPartialFinalCtaAndPreservesCanaries)
{
    int32_t const smVersion = getSMVersion();
    if (!isBlackwellGeforceSM(smVersion))
    {
        GTEST_SKIP() << "GDN L2 norm PDL test requires SM120 or SM121, got SM" << smVersion;
    }

    GdnL2NormResult const legacy = runGdnL2NormPartialCtaCase(GdnL2NormLaunchMode::kLegacyTwoLaunch);
    GdnL2NormResult const fusedPdlOff = runGdnL2NormPartialCtaCase(GdnL2NormLaunchMode::kFusedPdlOff);
    GdnL2NormResult const fusedPdlOn = runGdnL2NormPartialCtaCase(GdnL2NormLaunchMode::kFusedPdlOn);
    expectGdnL2NormGuardsIntact(legacy);
    expectGdnL2NormGuardsIntact(fusedPdlOff);
    expectGdnL2NormGuardsIntact(fusedPdlOn);
    EXPECT_EQ(fusedPdlOff.qInputBits, legacy.qInputBits);
    EXPECT_EQ(fusedPdlOff.kInputBits, legacy.kInputBits);
    EXPECT_EQ(fusedPdlOn.qInputBits, legacy.qInputBits);
    EXPECT_EQ(fusedPdlOn.kInputBits, legacy.kInputBits);
    EXPECT_EQ(fusedPdlOff.qBits, legacy.qBits);
    EXPECT_EQ(fusedPdlOff.kBits, legacy.kBits);
    EXPECT_EQ(fusedPdlOn.qBits, legacy.qBits);
    EXPECT_EQ(fusedPdlOn.kBits, legacy.kBits);
    EXPECT_NE(legacy.qBits, legacy.kBits);
    expectNormalizedRowsFromInput(legacy.qBits, legacy.qInputBits, /*numRows=*/9, /*headDim=*/128, "Q");
    expectNormalizedRowsFromInput(legacy.kBits, legacy.kInputBits, /*numRows=*/9, /*headDim=*/128, "K");
}

#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED

namespace
{

enum class GdnPdlExecutionMode
{
    kEager,
    kCudaGraph,
    kBenchmark,
};

struct GdnCapturedGraphSummary
{
    size_t numNodes{};
    size_t numKernelNodes{};
    size_t numEdges{};
    size_t numDefaultEdges{};
    size_t numProgrammaticEdges{};
    size_t numProgrammaticPortEdges{};
};

struct GdnPdlRunResult
{
    int32_t status{-1};
    std::vector<uint16_t> qBits;
    std::vector<uint16_t> kBits;
    std::vector<uint16_t> outputBits;
    std::vector<uint32_t> stateBits;
    std::vector<uint16_t> observedOutputBits;
    std::vector<uint32_t> observedStateBits;
    GdnCapturedGraphSummary graph;
};

void readGdnPdlOutputs(GdnPdlRunResult& result, void const* q, void const* k, void const* output, void const* state,
    size_t qkElements, size_t outputElements, size_t stateElements, void const* observedOutput = nullptr,
    void const* observedState = nullptr)
{
    result.qBits.resize(qkElements);
    result.kBits.resize(qkElements);
    result.outputBits.resize(outputElements);
    result.stateBits.resize(stateElements);
    CUDA_CHECK(cudaMemcpy(result.qBits.data(), q, qkElements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.kBits.data(), k, qkElements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.outputBits.data(), output, outputElements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.stateBits.data(), state, stateElements * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (observedOutput != nullptr && observedState != nullptr)
    {
        result.observedOutputBits.resize(outputElements);
        result.observedStateBits.resize(stateElements);
        CUDA_CHECK(cudaMemcpy(result.observedOutputBits.data(), observedOutput, outputElements * sizeof(uint16_t),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            result.observedStateBits.data(), observedState, stateElements * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
}

#if CUDART_VERSION >= 12030

cudaError_t getCapturedGraphEdges(
    cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to, cudaGraphEdgeData* edgeData, size_t* numEdges)
{
#if CUDART_VERSION >= 13000
    return cudaGraphGetEdges(graph, from, to, edgeData, numEdges);
#else
    return cudaGraphGetEdges_v2(graph, from, to, edgeData, numEdges);
#endif
}

GdnCapturedGraphSummary summarizeCapturedGdnGraph(cudaGraph_t graph)
{
    GdnCapturedGraphSummary summary;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &summary.numNodes));
    std::vector<cudaGraphNode_t> nodes(summary.numNodes);
    if (!nodes.empty())
    {
        CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &summary.numNodes));
        nodes.resize(summary.numNodes);
    }
    for (cudaGraphNode_t const node : nodes)
    {
        cudaGraphNodeType type{};
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        summary.numKernelNodes += type == cudaGraphNodeTypeKernel ? 1 : 0;
    }

    // Supplying edge-data storage on the first edge query avoids a lossy query
    // for the enabled graph. numNodes^2 is a conservative upper bound for the
    // captured producer-consumer chain.
    summary.numEdges = summary.numNodes * summary.numNodes;
    std::vector<cudaGraphNode_t> from(summary.numEdges), to(summary.numEdges);
    std::vector<cudaGraphEdgeData> edgeData(summary.numEdges);
    if (!edgeData.empty())
    {
        CUDA_CHECK(getCapturedGraphEdges(graph, from.data(), to.data(), edgeData.data(), &summary.numEdges));
        from.resize(summary.numEdges);
        to.resize(summary.numEdges);
        edgeData.resize(summary.numEdges);
    }

    for (size_t i = 0; i < summary.numEdges; ++i)
    {
        if (edgeData[i].type == cudaGraphDependencyTypeDefault)
        {
            ++summary.numDefaultEdges;
            continue;
        }
        if (edgeData[i].type != cudaGraphDependencyTypeProgrammatic)
        {
            continue;
        }
        ++summary.numProgrammaticEdges;
        summary.numProgrammaticPortEdges += edgeData[i].from_port == cudaGraphKernelNodePortProgrammatic ? 1 : 0;
    }
    return summary;
}

#endif // CUDART_VERSION >= 12030

template <bool EnablePdl>
__global__ void gdnPdlInputProducerKernel(half* q, half* k, half const* qSeed, half const* kSeed, size_t qkElements)
{
#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (EnablePdl)
    {
        // Release the fused Q/K grid before publishing the payload. Its PDL
        // wait is responsible for producer completion and visibility.
        if (threadIdx.x == 0)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
#endif

    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < qkElements;
        index += static_cast<size_t>(gridDim.x) * blockDim.x)
    {
        q[index] = qSeed[index];
        k[index] = kSeed[index];
    }
}

template <bool EnablePdl>
__global__ void gdnPdlOutputObserverKernel(half const* output, half* observedOutput, size_t outputElements,
    float const* state, float* observedState, size_t stateElements)
{
#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (EnablePdl)
    {
        // The GDN STORE_O warp triggers while its math warp may still be
        // publishing recurrent state. Do not consume either result until the
        // whole producer grid has completed.
        asm volatile("griddepcontrol.wait;\n" ::: "memory");
    }
#endif

    size_t const first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t const stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < outputElements; index += stride)
    {
        observedOutput[index] = output[index];
    }
    for (size_t index = first; index < stateElements; index += stride)
    {
        observedState[index] = state[index];
    }
}

void launchGdnPdlInputProducer(
    void* q, void* k, void const* qSeed, void const* kSeed, size_t qkElements, bool enablePdl, cudaStream_t stream)
{
    constexpr int32_t threadsPerBlock = 256;
    int32_t const numBlocks = static_cast<int32_t>((qkElements + threadsPerBlock - 1) / threadsPerBlock);
    if (enablePdl)
    {
        gdnPdlInputProducerKernel<true><<<numBlocks, threadsPerBlock, 0, stream>>>(static_cast<half*>(q),
            static_cast<half*>(k), static_cast<half const*>(qSeed), static_cast<half const*>(kSeed), qkElements);
    }
    else
    {
        gdnPdlInputProducerKernel<false><<<numBlocks, threadsPerBlock, 0, stream>>>(static_cast<half*>(q),
            static_cast<half*>(k), static_cast<half const*>(qSeed), static_cast<half const*>(kSeed), qkElements);
    }
}

template <bool EnablePdl>
void launchGdnPdlOutputObserver(void const* output, void* observedOutput, size_t outputElements, void const* state,
    void* observedState, size_t stateElements, cudaStream_t stream)
{
    constexpr int32_t threadsPerBlock = 256;
    size_t const numElements = std::max(outputElements, stateElements);
    int32_t const numBlocks = static_cast<int32_t>((numElements + threadsPerBlock - 1) / threadsPerBlock);
#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
    cudaLaunchAttribute pdlAttribute{};
    cudaLaunchConfig_t launchConfig{};
    launchConfig.gridDim = dim3(numBlocks);
    launchConfig.blockDim = dim3(threadsPerBlock);
    launchConfig.dynamicSmemBytes = 0;
    launchConfig.stream = stream;
    launchConfig.attrs = EnablePdl ? &pdlAttribute : nullptr;
    launchConfig.numAttrs = EnablePdl ? 1U : 0U;
    if constexpr (EnablePdl)
    {
        pdlAttribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        pdlAttribute.val.programmaticStreamSerializationAllowed = 1;
    }
    CUDA_CHECK(cudaLaunchKernelEx(&launchConfig, gdnPdlOutputObserverKernel<EnablePdl>,
        static_cast<half const*>(output), static_cast<half*>(observedOutput), outputElements,
        static_cast<float const*>(state), static_cast<float*>(observedState), stateElements));
#else
    gdnPdlOutputObserverKernel<EnablePdl><<<numBlocks, threadsPerBlock, 0, stream>>>(static_cast<half const*>(output),
        static_cast<half*>(observedOutput), outputElements, static_cast<float const*>(state),
        static_cast<float*>(observedState), stateElements);
#endif
}

GdnPdlRunResult runGdnPdlCase(int32_t seqLen, bool enablePdl, GdnPdlExecutionMode mode, int32_t replayCount = 1,
    bool includeBoundaryKernels = false)
{
    constexpr int32_t n = 1;
    constexpr int32_t h = 16;
    constexpr int32_t hv = 32;
    constexpr int32_t k = 128;
    constexpr int32_t v = 128;
    constexpr uint8_t qPoison = 0x5A;
    constexpr uint8_t kPoison = 0xA5;
    constexpr uint8_t statePoison = 0xC3;
    constexpr uint8_t outputPoison = 0x3C;
    size_t const qkElements = static_cast<size_t>(n) * seqLen * h * k;
    size_t const vElements = static_cast<size_t>(n) * seqLen * hv * v;
    size_t const abElements = static_cast<size_t>(n) * seqLen * hv;
    size_t const stateElements = static_cast<size_t>(n) * hv * k * v;
    size_t const outputElements = vElements;
    size_t const tensorMapBytes = static_cast<size_t>(CuteDslGDNRunner::kBlackwellGeforceMaxSMCount)
        * CuteDslGDNRunner::kBlackwellGeforceTensorMapDescriptorBytes;

    std::vector<half> qInput(qkElements), kInput(qkElements), vInput(vElements), aInput(abElements), bInput(abElements),
        dtBiasInput(hv);
    std::vector<float> aLogInput(hv), stateInput(stateElements);
    std::vector<int32_t> contextLengths(n, seqLen);
    uint32_t seed = 0x925120u + static_cast<uint32_t>(seqLen);
    for (size_t i = 0; i < qkElements; ++i)
    {
        qInput[i] = floatToHalf(lcgStep(seed) * 0.2f);
        kInput[i] = floatToHalf(lcgStep(seed) * 0.2f);
    }
    for (size_t i = 0; i < vElements; ++i)
    {
        vInput[i] = floatToHalf(lcgStep(seed) * 0.2f);
    }
    for (size_t i = 0; i < abElements; ++i)
    {
        aInput[i] = floatToHalf(lcgStep(seed) * 0.5f);
        bInput[i] = floatToHalf(lcgStep(seed) * 0.5f);
    }
    for (int32_t i = 0; i < hv; ++i)
    {
        aLogInput[i] = -2.0f + 0.25f * static_cast<float>(i % 4);
        dtBiasInput[i] = floatToHalf(0.02f * static_cast<float>(i + 1));
    }
    for (size_t i = 0; i < stateElements; ++i)
    {
        stateInput[i] = lcgStep(seed) * 0.01f;
    }

    void *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dA = nullptr, *dB = nullptr;
    void *dALog = nullptr, *dDtBias = nullptr, *dState = nullptr, *dContextLengths = nullptr;
    void *dOutput = nullptr, *dTensorMapScratch = nullptr;
    void *dObservedOutput = nullptr, *dObservedState = nullptr;
    void *dQSeed = nullptr, *dKSeed = nullptr, *dStateSeed = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ, qkElements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dK, qkElements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dV, vElements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dA, abElements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dB, abElements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dALog, static_cast<size_t>(hv) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDtBias, static_cast<size_t>(hv) * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dState, stateElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dContextLengths, static_cast<size_t>(n) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dOutput, outputElements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dTensorMapScratch, tensorMapBytes));
    CUDA_CHECK(cudaMalloc(&dQSeed, qkElements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dKSeed, qkElements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dStateSeed, stateElements * sizeof(float)));
    if (includeBoundaryKernels)
    {
        CUDA_CHECK(cudaMalloc(&dObservedOutput, outputElements * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dObservedState, stateElements * sizeof(float)));
    }
    auto allocationsGuard = Defer([&] {
        cudaFree(dQ);
        cudaFree(dK);
        cudaFree(dV);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dALog);
        cudaFree(dDtBias);
        cudaFree(dState);
        cudaFree(dContextLengths);
        cudaFree(dOutput);
        cudaFree(dTensorMapScratch);
        cudaFree(dQSeed);
        cudaFree(dKSeed);
        cudaFree(dStateSeed);
        cudaFree(dObservedOutput);
        cudaFree(dObservedState);
    });

    CUDA_CHECK(cudaMemcpy(dQSeed, qInput.data(), qkElements * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dKSeed, kInput.data(), qkElements * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, vInput.data(), vElements * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA, aInput.data(), abElements * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, bInput.data(), abElements * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dALog, aLogInput.data(), static_cast<size_t>(hv) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDtBias, dtBiasInput.data(), static_cast<size_t>(hv) * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dStateSeed, stateInput.data(), stateElements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        dContextLengths, contextLengths.data(), static_cast<size_t>(n) * sizeof(int32_t), cudaMemcpyHostToDevice));

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto streamGuard = Defer([&] { cudaStreamDestroy(stream); });

    auto resetProducedBuffers = [&] {
        // Make stale-data reuse observable before restoring the exact inputs.
        CUDA_CHECK(cudaMemsetAsync(dQ, qPoison, qkElements * sizeof(half), stream));
        CUDA_CHECK(cudaMemsetAsync(dK, kPoison, qkElements * sizeof(half), stream));
        CUDA_CHECK(cudaMemsetAsync(dState, statePoison, stateElements * sizeof(float), stream));
        if (!includeBoundaryKernels)
        {
            CUDA_CHECK(cudaMemcpyAsync(dQ, dQSeed, qkElements * sizeof(half), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dK, dKSeed, qkElements * sizeof(half), cudaMemcpyDeviceToDevice, stream));
        }
        CUDA_CHECK(
            cudaMemcpyAsync(dState, dStateSeed, stateElements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(dOutput, outputPoison, outputElements * sizeof(half), stream));
        CUDA_CHECK(cudaMemsetAsync(dTensorMapScratch, outputPoison, tensorMapBytes, stream));
        if (includeBoundaryKernels)
        {
            CUDA_CHECK(cudaMemsetAsync(dObservedOutput, outputPoison, outputElements * sizeof(half), stream));
            CUDA_CHECK(cudaMemsetAsync(dObservedState, statePoison, stateElements * sizeof(float), stream));
        }
    };

    GDNParams params{};
    params.q = dQ;
    params.k = dK;
    params.v = dV;
    params.a = dA;
    params.b = dB;
    params.A_log = dALog;
    params.dt_bias = dDtBias;
    params.h0_source = dState;
    params.context_lengths = dContextLengths;
    params.tensormap_scratch = dTensorMapScratch;
    params.o = dOutput;
    params.enablePdl = enablePdl;
    params.use_prefill = true;
    params.n = n;
    params.seq_len = seqLen;
    params.h = h;
    params.hv = hv;
    params.k_dim = k;
    params.v_dim = v;
    params.smVersion = getSMVersion();

    GdnPdlRunResult result;
    CuteDslGDNRunner runner;
    if (!CuteDslGDNRunner::ensureKernelModules(params, stream))
    {
        return result;
    }
    // Cache device properties before capture so graph contents contain only
    // the standalone Q/K-to-GDN pair or the boundary-inclusive four-kernel chain.
    (void) getDeviceMultiProcessorCount();

    auto launchKernelChain = [&]() {
        if (includeBoundaryKernels)
        {
            launchGdnPdlInputProducer(dQ, dK, dQSeed, dKSeed, qkElements, enablePdl, stream);
        }
        int32_t const status = runner.run(params, stream);
        if (status == 0 && includeBoundaryKernels)
        {
            if (enablePdl)
            {
                launchGdnPdlOutputObserver<true>(
                    dOutput, dObservedOutput, outputElements, dState, dObservedState, stateElements, stream);
            }
            else
            {
                launchGdnPdlOutputObserver<false>(
                    dOutput, dObservedOutput, outputElements, dState, dObservedState, stateElements, stream);
            }
        }
        return status;
    };

    auto runAndRead = [&]() {
        resetProducedBuffers();
        result.status = launchKernelChain();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());
        if (result.status == 0)
        {
            readGdnPdlOutputs(result, dQ, dK, dOutput, dState, qkElements, outputElements, stateElements,
                dObservedOutput, dObservedState);
        }
    };

    if (mode == GdnPdlExecutionMode::kBenchmark)
    {
        constexpr int32_t warmupCount = 3;
        constexpr int32_t measuredCount = 30;
        for (int32_t warmup = 0; warmup < warmupCount; ++warmup)
        {
            resetProducedBuffers();
            result.status = launchKernelChain();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaGetLastError());
            if (result.status != 0)
            {
                return result;
            }
        }

        cudaEvent_t start{}, stop{};
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        auto eventGuard = Defer([&] {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        });

        std::vector<float> samplesMs;
        samplesMs.reserve(measuredCount);
        char const* const pdlMode = enablePdl ? "on" : "off";
        for (int32_t sample = 0; sample < measuredCount; ++sample)
        {
            // Reset/poison operations precede the start event on the same
            // stream, so only combined Q/K norm and fused GDN are timed.
            resetProducedBuffers();
            CUDA_CHECK(cudaEventRecord(start, stream));
            result.status = launchKernelChain();
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaGetLastError());
            if (result.status != 0)
            {
                return result;
            }

            float elapsedMs{};
            CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
            samplesMs.push_back(elapsedMs);
            std::printf(
                "{\"schema\":\"gdn_pdl_bench.v1\",\"kind\":\"sample\",\"pdl\":\"%s\","
                "\"sample\":%d,\"duration_ms\":%.9f,\"n\":1,\"seq_len\":2048,\"h\":16,\"hv\":32,"
                "\"k\":128,\"v\":128}\n",
                pdlMode, sample, static_cast<double>(elapsedMs));
        }

        std::vector<float> sortedMs(samplesMs);
        std::sort(sortedMs.begin(), sortedMs.end());
        double sumMs = 0.0;
        for (float const sampleMs : samplesMs)
        {
            sumMs += sampleMs;
        }
        double const medianMs
            = 0.5 * static_cast<double>(sortedMs[measuredCount / 2 - 1] + sortedMs[measuredCount / 2]);
        std::printf(
            "{\"schema\":\"gdn_pdl_bench.v1\",\"kind\":\"summary\",\"pdl\":\"%s\","
            "\"warmups\":%d,\"samples\":%d,\"mean_ms\":%.9f,\"median_ms\":%.9f,\"min_ms\":%.9f,"
            "\"max_ms\":%.9f,\"n\":1,\"seq_len\":2048,\"h\":16,\"hv\":32,\"k\":128,\"v\":128}\n",
            pdlMode, warmupCount, measuredCount, sumMs / measuredCount, medianMs, static_cast<double>(sortedMs.front()),
            static_cast<double>(sortedMs.back()));

        readGdnPdlOutputs(result, dQ, dK, dOutput, dState, qkElements, outputElements, stateElements, dObservedOutput,
            dObservedState);
        return result;
    }

    if (mode == GdnPdlExecutionMode::kEager)
    {
        GdnPdlRunResult first;
        for (int32_t replay = 0; replay < replayCount; ++replay)
        {
            runAndRead();
            if (result.status != 0)
            {
                return result;
            }
            if (replay == 0)
            {
                first = result;
            }
            else
            {
                EXPECT_TRUE(result.qBits == first.qBits) << "Q changed on eager replay=" << replay;
                EXPECT_TRUE(result.kBits == first.kBits) << "K changed on eager replay=" << replay;
                EXPECT_TRUE(result.outputBits == first.outputBits) << "output changed on eager replay=" << replay;
                EXPECT_TRUE(result.stateBits == first.stateBits) << "state changed on eager replay=" << replay;
                EXPECT_TRUE(result.observedOutputBits == first.observedOutputBits)
                    << "observed output changed on eager replay=" << replay;
                EXPECT_TRUE(result.observedStateBits == first.observedStateBits)
                    << "observed state changed on eager replay=" << replay;
            }
        }
        return first;
    }

#if CUDART_VERSION >= 12030
    // First-use wrapper/device setup is intentionally outside capture.
    runAndRead();
    if (result.status != 0)
    {
        return result;
    }
    resetProducedBuffers();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph{};
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    result.status = launchKernelChain();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    auto graphGuard = Defer([&] {
        if (graph != nullptr)
        {
            cudaGraphDestroy(graph);
        }
    });
    if (result.status != 0)
    {
        return result;
    }
    result.graph = summarizeCapturedGdnGraph(graph);

    cudaGraphExec_t graphExec{};
    CUDA_CHECK(instantiateCudaGraph(&graphExec, graph));
    auto graphExecGuard = Defer([&] {
        if (graphExec != nullptr)
        {
            cudaGraphExecDestroy(graphExec);
        }
    });

    GdnPdlRunResult first;
    for (int32_t replay = 0; replay < replayCount; ++replay)
    {
        resetProducedBuffers();
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());
        result.status = 0;
        readGdnPdlOutputs(result, dQ, dK, dOutput, dState, qkElements, outputElements, stateElements, dObservedOutput,
            dObservedState);
        if (replay == 0)
        {
            first = result;
        }
        else
        {
            EXPECT_TRUE(result.qBits == first.qBits) << "Q changed on graph replay=" << replay;
            EXPECT_TRUE(result.kBits == first.kBits) << "K changed on graph replay=" << replay;
            EXPECT_TRUE(result.outputBits == first.outputBits) << "output changed on graph replay=" << replay;
            EXPECT_TRUE(result.stateBits == first.stateBits) << "state changed on graph replay=" << replay;
            EXPECT_TRUE(result.observedOutputBits == first.observedOutputBits)
                << "observed output changed on graph replay=" << replay;
            EXPECT_TRUE(result.observedStateBits == first.observedStateBits)
                << "observed state changed on graph replay=" << replay;
        }
    }
    first.graph = result.graph;
    return first;
#else
    ADD_FAILURE() << "GDN PDL graph test requires CUDA Toolkit 12.3 or newer";
    return result;
#endif
}

void expectGdnPdlResultsBitwiseEqual(GdnPdlRunResult const& pdlOff, GdnPdlRunResult const& pdlOn)
{
    ASSERT_EQ(pdlOff.status, 0);
    ASSERT_EQ(pdlOn.status, 0);
    EXPECT_TRUE(pdlOn.qBits == pdlOff.qBits) << "PDL changed normalized Q";
    EXPECT_TRUE(pdlOn.kBits == pdlOff.kBits) << "PDL changed normalized K";
    EXPECT_TRUE(pdlOn.outputBits == pdlOff.outputBits) << "PDL changed fused output";
    EXPECT_TRUE(pdlOn.stateBits == pdlOff.stateBits) << "PDL changed recurrent state";
    EXPECT_TRUE(pdlOn.observedOutputBits == pdlOff.observedOutputBits) << "PDL changed downstream-observed output";
    EXPECT_TRUE(pdlOn.observedStateBits == pdlOff.observedStateBits) << "PDL changed downstream-observed state";
    if (!pdlOff.observedOutputBits.empty() || !pdlOff.observedStateBits.empty())
    {
        EXPECT_TRUE(pdlOff.observedOutputBits == pdlOff.outputBits) << "PDL-off downstream observed stale output";
        EXPECT_TRUE(pdlOff.observedStateBits == pdlOff.stateBits) << "PDL-off downstream observed stale state";
        EXPECT_TRUE(pdlOn.observedOutputBits == pdlOn.outputBits) << "PDL-on downstream observed stale output";
        EXPECT_TRUE(pdlOn.observedStateBits == pdlOn.stateBits) << "PDL-on downstream observed stale state";
    }
}

void runPdlFocusedBenchmark(bool enablePdl)
{
    int32_t const smVersion = getSMVersion();
    if (!isBlackwellGeforceSM(smVersion))
    {
        GTEST_SKIP() << "Focused GDN PDL benchmark requires SM120 or SM121, got SM" << smVersion;
    }
#if !SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
    GTEST_SKIP() << "Focused GDN PDL benchmark requires toolchain PDL support";
#endif

    GdnPdlRunResult const result = runGdnPdlCase(
        /*seqLen=*/2048, enablePdl, GdnPdlExecutionMode::kBenchmark);
    ASSERT_EQ(result.status, 0);
}

} // namespace

TEST(GDNCuteDsl, PdlMatchesDisabledForShortAndLongSequences)
{
    int32_t const smVersion = getSMVersion();
    if (!isBlackwellGeforceSM(smVersion))
    {
        GTEST_SKIP() << "GDN PDL comparison requires SM120 or SM121, got SM" << smVersion;
    }

    for (int32_t const seqLen : {164, 2048})
    {
        SCOPED_TRACE(::testing::Message() << "seqLen=" << seqLen);
        GdnPdlRunResult const pdlOff = runGdnPdlCase(seqLen, /*enablePdl=*/false, GdnPdlExecutionMode::kEager);
        GdnPdlRunResult const pdlOn = runGdnPdlCase(seqLen, /*enablePdl=*/true, GdnPdlExecutionMode::kEager);
        expectGdnPdlResultsBitwiseEqual(pdlOff, pdlOn);
    }
}

TEST(GDNCuteDsl, PdlEagerNonDefaultStreamResetStressIsDeterministic)
{
    int32_t const smVersion = getSMVersion();
    if (!isBlackwellGeforceSM(smVersion))
    {
        GTEST_SKIP() << "GDN PDL eager stress requires SM120 or SM121, got SM" << smVersion;
    }

    GdnPdlRunResult const pdlOff
        = runGdnPdlCase(/*seqLen=*/164, /*enablePdl=*/false, GdnPdlExecutionMode::kEager, /*replayCount=*/8,
            /*includeBoundaryKernels=*/true);
    GdnPdlRunResult const pdlOn
        = runGdnPdlCase(/*seqLen=*/164, /*enablePdl=*/true, GdnPdlExecutionMode::kEager, /*replayCount=*/8,
            /*includeBoundaryKernels=*/true);
    expectGdnPdlResultsBitwiseEqual(pdlOff, pdlOn);
}

TEST(GDNCuteDsl, DISABLED_PdlFocusedBenchmarkOff)
{
    runPdlFocusedBenchmark(/*enablePdl=*/false);
}

TEST(GDNCuteDsl, DISABLED_PdlFocusedBenchmarkOn)
{
    runPdlFocusedBenchmark(/*enablePdl=*/true);
}

#if CUDART_VERSION >= 12030

TEST(GDNCuteDsl, PdlCudaGraphReplayUsesThreeProgrammaticEdgesAcrossDualRoleKernels)
{
    int32_t const smVersion = getSMVersion();
    if (!isBlackwellGeforceSM(smVersion))
    {
        GTEST_SKIP() << "GDN PDL CUDA Graph test requires SM120 or SM121, got SM" << smVersion;
    }
#if !SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
    GTEST_SKIP() << "GDN PDL CUDA Graph test requires toolchain PDL support";
#endif

    GdnPdlRunResult const pdlOff
        = runGdnPdlCase(/*seqLen=*/164, /*enablePdl=*/false, GdnPdlExecutionMode::kCudaGraph, /*replayCount=*/4,
            /*includeBoundaryKernels=*/true);
    GdnPdlRunResult const pdlOn
        = runGdnPdlCase(/*seqLen=*/164, /*enablePdl=*/true, GdnPdlExecutionMode::kCudaGraph, /*replayCount=*/4,
            /*includeBoundaryKernels=*/true);
    expectGdnPdlResultsBitwiseEqual(pdlOff, pdlOn);

    EXPECT_EQ(pdlOff.graph.numNodes, 4u);
    EXPECT_EQ(pdlOff.graph.numKernelNodes, 4u);
    EXPECT_EQ(pdlOff.graph.numEdges, 3u);
    EXPECT_EQ(pdlOff.graph.numDefaultEdges, 3u);
    EXPECT_EQ(pdlOff.graph.numProgrammaticEdges, 0u);
    EXPECT_EQ(pdlOff.graph.numProgrammaticPortEdges, 0u);
    EXPECT_EQ(pdlOn.graph.numNodes, 4u);
    EXPECT_EQ(pdlOn.graph.numKernelNodes, 4u);
    EXPECT_EQ(pdlOn.graph.numEdges, 3u);
    EXPECT_EQ(pdlOn.graph.numDefaultEdges, 0u);
    EXPECT_EQ(pdlOn.graph.numProgrammaticEdges, 3u);
    EXPECT_EQ(pdlOn.graph.numProgrammaticPortEdges, 3u);
}

#endif // CUDART_VERSION >= 12030

#endif // CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED

/**
 * Compare Blackwell vs Sequential kernel output for the same input data.
 * Run both kernels and compare output and state.
 */
void runGDNBlackwellVsSequentialTest()
{
    int32_t const smVersion = getSMVersion();
    if (!isBlackwellSM(smVersion))
    {
        GTEST_SKIP() << "Blackwell vs Sequential test requires SM100/101/110 or SM120/121";
        return;
    }

    int32_t const n = 1;
    int32_t const seq_len = 164;
    int32_t const h = 16;
    int32_t const hv = 32;
    int32_t const k = 128;
    int32_t const v = 128;

    size_t const qkvLen = static_cast<size_t>(n) * seq_len * h * k;
    size_t const vLen = static_cast<size_t>(n) * seq_len * hv * v;
    size_t const abLen = static_cast<size_t>(n) * seq_len * hv;
    size_t const h0Len = static_cast<size_t>(n) * hv * k * v;
    size_t const oLen = static_cast<size_t>(n) * seq_len * hv * v;

    std::vector<float> h_q(qkvLen), h_k(qkvLen), h_v(vLen), h_a(abLen), h_b(abLen);
    std::vector<float> h_A_log(hv), h_dt_bias(hv), h_h0(h0Len);
    uint32_t seed = 0x44u;
    for (size_t i = 0; i < qkvLen; ++i)
        h_q[i] = lcgStep(seed) * 0.2f;
    for (size_t i = 0; i < qkvLen; ++i)
        h_k[i] = lcgStep(seed) * 0.2f;
    for (size_t i = 0; i < vLen; ++i)
        h_v[i] = lcgStep(seed) * 0.2f;
    for (size_t i = 0; i < abLen; ++i)
        h_a[i] = lcgStep(seed) * 0.5f;
    for (size_t i = 0; i < abLen; ++i)
        h_b[i] = lcgStep(seed) * 0.5f;
    for (int32_t i = 0; i < hv; ++i)
        h_A_log[i] = -2.f + 0.25f * (i % 4);
    for (int32_t i = 0; i < hv; ++i)
        h_dt_bias[i] = 0.02f * (i + 1);
    for (size_t i = 0; i < h0Len; ++i)
        h_h0[i] = lcgStep(seed) * 0.01f;

    std::vector<half> h_q_h(qkvLen), h_k_h(qkvLen), h_v_h(vLen);
    std::vector<half> h_a_h(abLen), h_b_h(abLen), h_dt_h(hv);
    for (size_t i = 0; i < qkvLen; ++i)
        h_q_h[i] = floatToHalf(h_q[i]);
    for (size_t i = 0; i < qkvLen; ++i)
        h_k_h[i] = floatToHalf(h_k[i]);
    for (size_t i = 0; i < vLen; ++i)
        h_v_h[i] = floatToHalf(h_v[i]);
    for (size_t i = 0; i < abLen; ++i)
    {
        h_a_h[i] = floatToHalf(h_a[i]);
        h_b_h[i] = floatToHalf(h_b[i]);
    }
    for (int32_t i = 0; i < hv; ++i)
        h_dt_h[i] = floatToHalf(h_dt_bias[i]);

    std::vector<int32_t> h_ctx(n, seq_len);

    // Allocate two sets of device buffers: one for Blackwell, one for sequential
    void *d_q, *d_k, *d_v, *d_a, *d_b, *d_A_log, *d_dt_bias, *d_ctx;
    CUDA_CHECK(cudaMalloc(&d_q, qkvLen * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k, qkvLen * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v, vLen * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_a, abLen * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, abLen * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_A_log, hv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dt_bias, hv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ctx, n * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q_h.data(), qkvLen * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k_h.data(), qkvLen * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v_h.data(), vLen * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, h_a_h.data(), abLen * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b_h.data(), abLen * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_log, h_A_log.data(), hv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dt_bias, h_dt_h.data(), hv * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ctx, h_ctx.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Blackwell run
    void *d_h0_bw, *d_o_bw;
    CUDA_CHECK(cudaMalloc(&d_h0_bw, h0Len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o_bw, oLen * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_h0_bw, h_h0.data(), h0Len * sizeof(float), cudaMemcpyHostToDevice));
    void* d_cu_seqlens = allocCuSeqlens(d_ctx, n);
    void* d_h0_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_h0_scratch, h0Len * sizeof(float)));
    void* d_tensormap_scratch = allocTensorMapScratch(smVersion);

    GDNParams bwParams{};
    bwParams.q = d_q;
    bwParams.k = d_k;
    bwParams.v = d_v;
    bwParams.a = d_a;
    bwParams.b = d_b;
    bwParams.A_log = d_A_log;
    bwParams.dt_bias = d_dt_bias;
    bwParams.h0_source = d_h0_bw;
    bwParams.context_lengths = d_ctx;
    bwParams.cu_seqlens = d_cu_seqlens;
    bwParams.h0_scratch = d_h0_scratch;
    bwParams.tensormap_scratch = d_tensormap_scratch;
    bwParams.o = d_o_bw;
    bwParams.n = n;
    bwParams.seq_len = seq_len;
    bwParams.use_prefill = true;
    bwParams.h = h;
    bwParams.hv = hv;
    bwParams.k_dim = k;
    bwParams.v_dim = v;
    bwParams.smVersion = smVersion;

    CuteDslGDNRunner runner;
    int ret = runner.run(bwParams, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(ret, 0) << "Blackwell run failed";

    // Sequential run (force sequential by setting smVersion < 100)
    void *d_h0_seq, *d_o_seq;
    CUDA_CHECK(cudaMalloc(&d_h0_seq, h0Len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o_seq, oLen * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_h0_seq, h_h0.data(), h0Len * sizeof(float), cudaMemcpyHostToDevice));

    GDNParams seqParams{};
    seqParams.q = d_q;
    seqParams.k = d_k;
    seqParams.v = d_v;
    seqParams.a = d_a;
    seqParams.b = d_b;
    seqParams.A_log = d_A_log;
    seqParams.dt_bias = d_dt_bias;
    seqParams.h0_source = d_h0_seq;
    seqParams.context_lengths = d_ctx;
    seqParams.o = d_o_seq;
    seqParams.n = n;
    seqParams.seq_len = seq_len;
    seqParams.use_prefill = true;
    seqParams.h = h;
    seqParams.hv = hv;
    seqParams.k_dim = k;
    seqParams.v_dim = v;
    seqParams.smVersion = 89; // Force sequential path

    ret = runner.run(seqParams, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(ret, 0) << "Sequential run failed";

    // Compare outputs
    std::vector<half> h_o_bw(oLen), h_o_seq(oLen);
    CUDA_CHECK(cudaMemcpy(h_o_bw.data(), d_o_bw, oLen * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_o_seq.data(), d_o_seq, oLen * sizeof(half), cudaMemcpyDeviceToHost));

    float maxDiff = 0.f, maxRelDiff = 0.f;
    float maxAbsBw = 0.f, maxAbsSeq = 0.f;
    size_t largeDiffCount = 0;
    for (size_t i = 0; i < oLen; ++i)
    {
        float bw = halfToFloat(h_o_bw[i]), sq = halfToFloat(h_o_seq[i]);
        float diff = std::abs(bw - sq);
        float denom = std::max(std::abs(sq), 1e-6f);
        maxDiff = std::max(maxDiff, diff);
        maxRelDiff = std::max(maxRelDiff, diff / denom);
        maxAbsBw = std::max(maxAbsBw, std::abs(bw));
        maxAbsSeq = std::max(maxAbsSeq, std::abs(sq));
        if (diff > 0.1f)
            ++largeDiffCount;
    }
    printf("  BW vs SEQ output: maxDiff=%.6f maxRelDiff=%.6f maxAbsBw=%.6f maxAbsSeq=%.6f largeDiffs=%zu/%zu\n",
        maxDiff, maxRelDiff, maxAbsBw, maxAbsSeq, largeDiffCount, oLen);

    // Compare states
    std::vector<float> h_h0_bw(h0Len), h_h0_seq(h0Len);
    CUDA_CHECK(cudaMemcpy(h_h0_bw.data(), d_h0_bw, h0Len * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_h0_seq.data(), d_h0_seq, h0Len * sizeof(float), cudaMemcpyDeviceToHost));

    float maxStateDiff = 0.f;
    for (size_t i = 0; i < h0Len; ++i)
        maxStateDiff = std::max(maxStateDiff, std::abs(h_h0_bw[i] - h_h0_seq[i]));
    printf("  BW vs SEQ state: maxDiff=%.6f\n", maxStateDiff);

    EXPECT_LT(maxDiff, 0.5f) << "Blackwell vs Sequential output diverges too much";

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_A_log));
    CUDA_CHECK(cudaFree(d_dt_bias));
    CUDA_CHECK(cudaFree(d_ctx));
    CUDA_CHECK(cudaFree(d_h0_bw));
    CUDA_CHECK(cudaFree(d_o_bw));
    CUDA_CHECK(cudaFree(d_cu_seqlens));
    CUDA_CHECK(cudaFree(d_h0_scratch));
    if (d_tensormap_scratch)
        CUDA_CHECK(cudaFree(d_tensormap_scratch));
    CUDA_CHECK(cudaFree(d_h0_seq));
    CUDA_CHECK(cudaFree(d_o_seq));
}

TEST(GDNCuteDsl, BlackwellVsSequential)
{
    runGDNBlackwellVsSequentialTest();
}

TEST(GDNCuteDsl, PrefillQwen35BlackwellGeforce)
{
#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
    if (!isBlackwellGeforceSM(getSMVersion()))
        GTEST_SKIP() << "Blackwell GeForce-specific coverage requires SM120 or SM121";
    runGDNPrefillQwen35Test(1e-4f, 1e-4f, /*enablePdl=*/true);
#else
    GTEST_SKIP() << "Blackwell GeForce GDN artifact is not enabled";
#endif
}

TEST(GDNCuteDsl, CanImplement)
{
    EXPECT_TRUE(CuteDslGDNRunner::canImplement(128, 128, 80));
    EXPECT_TRUE(CuteDslGDNRunner::canImplement(128, 128, 89));
    EXPECT_FALSE(CuteDslGDNRunner::canImplement(64, 128, 80));
    EXPECT_FALSE(CuteDslGDNRunner::canImplement(128, 128, 70));
}

TEST(GDNCuteDsl, BackendSelection)
{
    GDNParams params{};
    params.h = 16;
    params.hv = 32;
    params.smVersion = 120;
    int32_t stateIndex{};
    params.state_indices = &stateIndex;

    params.use_mtp = true;
    EXPECT_EQ(CuteDslGDNRunner::selectBackend(params), GDNBackend::kDecodeMTP);

    params.use_mtp = false;
    params.use_prefill = false;
    EXPECT_EQ(CuteDslGDNRunner::selectBackend(params), GDNBackend::kDecode);

    params.use_prefill = true;
    params.smVersion = 87;
    EXPECT_EQ(CuteDslGDNRunner::selectBackend(params), GDNBackend::kPrefill);

#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    params.smVersion = 110;
    EXPECT_EQ(CuteDslGDNRunner::selectBackend(params), GDNBackend::kPrefillBlackwell);
#else
    params.smVersion = 110;
    EXPECT_EQ(CuteDslGDNRunner::selectBackend(params), GDNBackend::kPrefill);
#endif

#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
    params.smVersion = 120;
    EXPECT_EQ(CuteDslGDNRunner::selectBackend(params), GDNBackend::kPrefillBlackwellGeforce);
    params.smVersion = 121;
    EXPECT_EQ(CuteDslGDNRunner::selectBackend(params), GDNBackend::kPrefillBlackwellGeforce);
#else
    params.smVersion = 120;
    EXPECT_EQ(CuteDslGDNRunner::selectBackend(params), GDNBackend::kPrefill);
#endif

    params.h = 0;
    EXPECT_EQ(CuteDslGDNRunner::selectBackend(params), GDNBackend::kPrefill);
}

/**
 * MTP decode: uniform T=4, all batch items process all 4 steps.
 * Verifies output, final h-state, and per-step intermediate states
 * against CPU reference — enabling runtime rollback to any accepted token count.
 */
TEST(GDNCuteDsl, MTPDecodeWithIntermediateStates)
{
    runGDNDecodeMTPTestConfig(/*seq_len=*/4, /*with_cache=*/true);
}

#endif // CUTE_DSL_GDN_ENABLED
