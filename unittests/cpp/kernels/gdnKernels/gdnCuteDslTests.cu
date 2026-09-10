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

#include <cmath>
#include <cstdio>
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
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return d_cu;
}

namespace
{

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

/**
 * GDN prefill with the exact Qwen3.5-4B parameters: n=1, h=16, hv=32, seq_len=164.
 * This configuration triggers grouped value attention (h_r=2) and tail masking on optimized Blackwell paths.
 */
void runGDNPrefillQwen35Test(float atol = 5e-2f, float rtol = 5e-2f)
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
    params.h = h;
    params.hv = hv;
    params.k_dim = k;
    params.v_dim = v;
    params.smVersion = smVersion;

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
    if (!isBlackwellGeforceSM(getSMVersion()))
        GTEST_SKIP() << "Blackwell GeForce-specific coverage requires SM120 or SM121";
    runGDNPrefillQwen35Test(1e-4f, 1e-4f);
}

TEST(GDNCuteDsl, CanImplement)
{
    EXPECT_TRUE(CuteDslGDNRunner::canImplement(128, 128, 80));
    EXPECT_TRUE(CuteDslGDNRunner::canImplement(128, 128, 89));
    EXPECT_FALSE(CuteDslGDNRunner::canImplement(64, 128, 80));
    EXPECT_FALSE(CuteDslGDNRunner::canImplement(128, 128, 70));
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
