/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Unit tests for the stateless chunk-form GDN tree verify + replay commit
// kernels (gdnTreeChunkKernels.cu) against a double-precision host reference.
//
// The verify kernels stage H0/Attn/R through fp16 and accumulate on tensor
// cores, so the reference models the fp16 staging explicitly and the
// comparisons use mixed absolute/relative tolerances. The replay kernel is
// fp32 throughout, so its tolerance is tighter. Node counts include a
// non-multiple-of-16 size to exercise the wmma padding paths, and trees
// include an invalid (masked-out) node whose output rows must be exact zeros.

#include "kernels/gdnKernels/gdnTreeChunkKernels.h"
#include "kernels/speculative/mtpStateScatterKernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace trt_edgellm::kernel;

namespace
{

constexpr int32_t kDk = 128;
constexpr int32_t kDv = 128;

template <typename T>
T* uploadVec(std::vector<T> const& host)
{
    T* dev = nullptr;
    EXPECT_EQ(cudaMalloc(&dev, host.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return dev;
}

template <typename T>
std::vector<T> downloadVec(T const* dev, size_t count)
{
    std::vector<T> host(count);
    EXPECT_EQ(cudaMemcpy(host.data(), dev, count * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return host;
}

//! Mirrors the (internal) KS/QS staging base used by the verify kernels: the
//! staging area sits past the last stash cell of each intermediate row,
//! 256-byte aligned.
size_t ksqsBaseBytes(int32_t h, int32_t hv)
{
    size_t const nodeBytes = static_cast<size_t>(h) * kDk * sizeof(float) + static_cast<size_t>(hv) * 2 * sizeof(float)
        + static_cast<size_t>(hv) * kDk * sizeof(__half);
    return (static_cast<size_t>(kGDN_TREE_CHUNK_MAX_NODES) * nodeBytes + 255) & ~static_cast<size_t>(255);
}

//! Row tail = [KS/QS: hv x 2 x MAX_NODES x dk f32][prep: hv x PREP_WORDS f32],
//! matching the launcher's gdnTreeScratchEndBytes capacity requirement.
size_t stashRowBytes(int32_t h, int32_t hv)
{
    return ksqsBaseBytes(h, hv) + static_cast<size_t>(hv) * 2 * kGDN_TREE_CHUNK_MAX_NODES * kDk * sizeof(float)
        + static_cast<size_t>(hv) * kGDN_TREE_CHUNK_PREP_WORDS * sizeof(float);
}

float halfRound(double x)
{
    return __half2float(__float2half(static_cast<float>(x)));
}

double softplusRef(double x)
{
    return (x > 20.0) ? x : std::log1p(std::exp(x));
}

//! Random test problem for one verify/replay configuration.
struct ChunkProblem
{
    int32_t batch;
    int32_t numNodes;
    int32_t h;
    int32_t hv;
    bool addInvalidNode; // makes the last node a masked-out padding node

    std::vector<int32_t> parents; // [batch][numNodes]
    std::vector<__half> q, k;     // [batch][N][h][128]
    std::vector<__half> v;        // [batch][N][hv][128]
    std::vector<__half> a, b;     // [batch][N][hv]
    std::vector<float> logA;      // [hv]
    std::vector<__half> dtBias;   // [hv]
    std::vector<float> h0;        // [batch][hv][128][128]

    void generate(uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uni(-1.f, 1.f);
        std::uniform_real_distribution<float> uniSmall(-0.5f, 0.5f);
        std::uniform_real_distribution<float> uniALog(-2.f, 0.f);

        parents.resize(static_cast<size_t>(batch) * numNodes);
        for (int32_t bi = 0; bi < batch; ++bi)
        {
            for (int32_t n = 0; n < numNodes; ++n)
            {
                // Node 0 is the root; every other node hangs off a random
                // earlier node (topological order holds by construction).
                int32_t parent = (n == 0) ? -1 : static_cast<int32_t>(rng() % static_cast<uint32_t>(n));
                parents[bi * numNodes + n] = parent;
            }
            if (addInvalidNode && numNodes >= 4)
            {
                parents[bi * numNodes + numNodes - 1] = -1; // invalid at n > 0
            }
        }

        auto fillHalf = [&](std::vector<__half>& dst, size_t count, auto& dist) {
            dst.resize(count);
            for (auto& x : dst)
            {
                x = __float2half(dist(rng));
            }
        };
        size_t const nQK = static_cast<size_t>(batch) * numNodes * h * kDk;
        size_t const nV = static_cast<size_t>(batch) * numNodes * hv * kDv;
        size_t const nAB = static_cast<size_t>(batch) * numNodes * hv;
        fillHalf(q, nQK, uni);
        fillHalf(k, nQK, uni);
        fillHalf(v, nV, uni);
        fillHalf(a, nAB, uni);
        fillHalf(b, nAB, uni);
        fillHalf(dtBias, hv, uniSmall);
        logA.resize(hv);
        for (auto& x : logA)
        {
            x = uniALog(rng);
        }
        h0.resize(static_cast<size_t>(batch) * hv * kDk * kDv);
        for (auto& x : h0)
        {
            x = uniSmall(rng);
        }
    }

    bool nodeValid(int32_t bi, int32_t n) const
    {
        int32_t const p = parents[bi * numNodes + n];
        return (n == 0) ? (p < 0) : (p >= 0 && p < n);
    }

    //! Inclusive ancestor chain of node n (empty for invalid nodes).
    std::vector<int32_t> ancestors(int32_t bi, int32_t n) const
    {
        std::vector<int32_t> chain;
        if (!nodeValid(bi, n))
        {
            return chain;
        }
        int32_t iter = n;
        while (iter >= 0)
        {
            chain.push_back(iter);
            iter = parents[bi * numNodes + iter];
        }
        return chain;
    }
};

//! Per-(batch, hv) intermediate quantities of the host reference.
struct RefState
{
    std::vector<double> logg, beta, invK, invQ, cum, gamma; // [N]
    std::vector<double> M, Attn;                            // [N][N]
    std::vector<uint8_t> valid;                             // [N]
};

RefState referencePrep(ChunkProblem const& p, int32_t bi, int32_t iHv, float scale, bool useL2Norm)
{
    int32_t const N = p.numNodes;
    int32_t const iH = iHv / (p.hv / p.h);
    RefState s;
    s.logg.resize(N);
    s.beta.resize(N);
    s.invK.resize(N);
    s.invQ.resize(N);
    s.cum.resize(N);
    s.gamma.resize(N);
    s.M.assign(static_cast<size_t>(N) * N, 0.0);
    s.Attn.assign(static_cast<size_t>(N) * N, 0.0);
    s.valid.resize(N);

    auto qAt = [&](int32_t n, int32_t kk) {
        return static_cast<double>(__half2float(p.q[((static_cast<size_t>(bi) * N + n) * p.h + iH) * kDk + kk]));
    };
    auto kAt = [&](int32_t n, int32_t kk) {
        return static_cast<double>(__half2float(p.k[((static_cast<size_t>(bi) * N + n) * p.h + iH) * kDk + kk]));
    };

    for (int32_t n = 0; n < N; ++n)
    {
        s.valid[n] = p.nodeValid(bi, n) ? 1 : 0;
        double const av = __half2float(p.a[(static_cast<size_t>(bi) * N + n) * p.hv + iHv]);
        double const bv = __half2float(p.b[(static_cast<size_t>(bi) * N + n) * p.hv + iHv]);
        double const x = av + __half2float(p.dtBias[iHv]);
        s.logg[n] = -std::exp(static_cast<double>(p.logA[iHv])) * softplusRef(x);
        s.beta[n] = 1.0 / (1.0 + std::exp(-bv));
        double sk2 = 0.0;
        double sq2 = 0.0;
        for (int32_t kk = 0; kk < kDk; ++kk)
        {
            sk2 += kAt(n, kk) * kAt(n, kk);
            sq2 += qAt(n, kk) * qAt(n, kk);
        }
        s.invK[n] = useL2Norm ? 1.0 / std::sqrt(sk2 + 1e-6) : 1.0;
        s.invQ[n] = (useL2Norm ? 1.0 / std::sqrt(sq2 + 1e-6) : 1.0) * scale;
    }
    for (int32_t n = 0; n < N; ++n)
    {
        double c = 0.0;
        for (int32_t ancNode : p.ancestors(bi, n))
        {
            c += s.logg[ancNode];
        }
        s.cum[n] = c;
        s.gamma[n] = std::exp(c);
    }
    for (int32_t n = 0; n < N; ++n)
    {
        for (int32_t ancNode : p.ancestors(bi, n))
        {
            double const d = std::exp(s.cum[n] - s.cum[ancNode]);
            double kkDot = 0.0;
            double qkDot = 0.0;
            for (int32_t kk = 0; kk < kDk; ++kk)
            {
                kkDot += kAt(n, kk) * kAt(ancNode, kk);
                qkDot += qAt(n, kk) * kAt(ancNode, kk);
            }
            kkDot *= s.invK[n] * s.invK[ancNode];
            qkDot *= s.invQ[n] * s.invK[ancNode];
            s.M[static_cast<size_t>(n) * N + ancNode] = (ancNode == n) ? 0.0 : s.beta[n] * d * kkDot;
            s.Attn[static_cast<size_t>(n) * N + ancNode] = d * qkDot;
        }
    }
    return s;
}

//! Host reference of the verify output o[b, n, hv, dv]. Models the kernel's
//! fp16 staging of H0 (KS/QS GEMM operand), Attn, and R.
std::vector<double> referenceVerifyOutput(ChunkProblem const& p, float scale, bool useL2Norm)
{
    int32_t const N = p.numNodes;
    std::vector<double> o(static_cast<size_t>(p.batch) * N * p.hv * kDv, 0.0);

    for (int32_t bi = 0; bi < p.batch; ++bi)
    {
        for (int32_t iHv = 0; iHv < p.hv; ++iHv)
        {
            RefState const s = referencePrep(p, bi, iHv, scale, useL2Norm);
            int32_t const iH = iHv / (p.hv / p.h);

            std::vector<double> B(static_cast<size_t>(N) * kDv);
            std::vector<double> QS(static_cast<size_t>(N) * kDv);
            for (int32_t n = 0; n < N; ++n)
            {
                for (int32_t vv = 0; vv < kDv; ++vv)
                {
                    double ks = 0.0;
                    double qs = 0.0;
                    for (int32_t kk = 0; kk < kDk; ++kk)
                    {
                        double const h0v
                            = halfRound(p.h0[((static_cast<size_t>(bi) * p.hv + iHv) * kDk + kk) * kDv + vv]);
                        ks += __half2float(p.k[((static_cast<size_t>(bi) * N + n) * p.h + iH) * kDk + kk]) * h0v;
                        qs += __half2float(p.q[((static_cast<size_t>(bi) * N + n) * p.h + iH) * kDk + kk]) * h0v;
                    }
                    ks *= s.invK[n];
                    qs *= s.invQ[n];
                    double const vraw = __half2float(p.v[((static_cast<size_t>(bi) * N + n) * p.hv + iHv) * kDv + vv]);
                    B[static_cast<size_t>(n) * kDv + vv] = s.beta[n] * (vraw - s.gamma[n] * ks);
                    QS[static_cast<size_t>(n) * kDv + vv] = qs;
                }
            }
            // Forward substitution: R[n] = B[n] - sum_{j<n} M[n][j] * R[j].
            for (int32_t n = 1; n < N; ++n)
            {
                for (int32_t vv = 0; vv < kDv; ++vv)
                {
                    double acc = 0.0;
                    for (int32_t j = 0; j < n; ++j)
                    {
                        acc += s.M[static_cast<size_t>(n) * N + j] * B[static_cast<size_t>(j) * kDv + vv];
                    }
                    B[static_cast<size_t>(n) * kDv + vv] -= acc;
                }
            }
            // O = valid ? gamma*QS + Attn(fp16) @ R(fp16) : 0.
            for (int32_t n = 0; n < N; ++n)
            {
                if (!s.valid[n])
                {
                    continue;
                }
                for (int32_t vv = 0; vv < kDv; ++vv)
                {
                    double acc = s.gamma[n] * QS[static_cast<size_t>(n) * kDv + vv];
                    for (int32_t j = 0; j < N; ++j)
                    {
                        acc += halfRound(s.Attn[static_cast<size_t>(n) * N + j])
                            * halfRound(B[static_cast<size_t>(j) * kDv + vv]);
                    }
                    o[((static_cast<size_t>(bi) * N + n) * p.hv + iHv) * kDv + vv] = acc;
                }
            }
        }
    }
    return o;
}

//! Host reference of the replay-committed recurrent state, recomputed from
//! the raw inputs with the same staging the verify stash applies (k as
//! fp32(fp16 k)*invK, g/beta fp32, v as fp16).
std::vector<double> referenceReplayState(
    ChunkProblem const& p, float scale, bool useL2Norm, std::vector<std::vector<int32_t>> const& acceptedPaths)
{
    int32_t const N = p.numNodes;
    std::vector<double> h(static_cast<size_t>(p.batch) * p.hv * kDk * kDv);
    for (size_t i = 0; i < h.size(); ++i)
    {
        h[i] = p.h0[i];
    }
    for (int32_t bi = 0; bi < p.batch; ++bi)
    {
        for (int32_t iHv = 0; iHv < p.hv; ++iHv)
        {
            RefState const s = referencePrep(p, bi, iHv, scale, useL2Norm);
            int32_t const iH = iHv / (p.hv / p.h);
            double* hd = h.data() + (static_cast<size_t>(bi) * p.hv + iHv) * kDk * kDv;
            for (int32_t node : acceptedPaths[bi])
            {
                double const g = static_cast<double>(std::exp(static_cast<float>(s.logg[node])));
                double const beta = s.beta[node];
                std::vector<double> kt(kDk);
                for (int32_t kk = 0; kk < kDk; ++kk)
                {
                    // The stash stores fp32(k_half) * invK as fp32.
                    kt[kk] = static_cast<double>(static_cast<float>(
                        __half2float(p.k[((static_cast<size_t>(bi) * N + node) * p.h + iH) * kDk + kk])
                        * static_cast<float>(s.invK[node])));
                }
                for (int32_t vv = 0; vv < kDv; ++vv)
                {
                    double sumHk = 0.0;
                    for (int32_t kk = 0; kk < kDk; ++kk)
                    {
                        hd[static_cast<size_t>(kk) * kDv + vv] *= g;
                        sumHk += hd[static_cast<size_t>(kk) * kDv + vv] * kt[kk];
                    }
                    double const vraw
                        = __half2float(p.v[((static_cast<size_t>(bi) * N + node) * p.hv + iHv) * kDv + vv]);
                    double const vNew = (vraw - sumHk) * beta;
                    for (int32_t kk = 0; kk < kDk; ++kk)
                    {
                        hd[static_cast<size_t>(kk) * kDv + vv] += kt[kk] * vNew;
                    }
                }
            }
        }
    }
    return h;
}

//! Device-side setup shared by the tests: uploads inputs, builds masks, and
//! runs the verify kernels. Owns the device buffers.
struct DeviceRun
{
    int32_t batch{};
    int32_t numNodes{};
    int32_t h{};
    int32_t hv{};
    size_t rowBytes{};

    int32_t* dParents{};
    uint32_t* dMasks{};
    __half *dQ{}, *dK{}, *dV{}, *dA{}, *dB{}, *dDtBias{};
    float* dALog{};
    float* dH0{};
    __half* dO{};
    char* dStash{};
    cudaStream_t stream{};

    //! residentRows: when non-empty, request b's h0 is placed in row residentRows[b] of a poolRows-row resident pool
    //! (other rows hold unrelated values) and verification reads it through those state indices.
    void run(ChunkProblem const& p, float scale, bool useL2Norm, size_t physicalRowBytes = 0,
        std::vector<int32_t> const& residentRows = {}, int32_t poolRows = 0,
        GdnTreeVerifyPath path = GdnTreeVerifyPath::kAuto)
    {
        batch = p.batch;
        numNodes = p.numNodes;
        h = p.h;
        hv = p.hv;
        rowBytes = physicalRowBytes == 0 ? stashRowBytes(h, hv) : physicalRowBytes;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        dParents = uploadVec(p.parents);
        dQ = uploadVec(p.q);
        dK = uploadVec(p.k);
        dV = uploadVec(p.v);
        dA = uploadVec(p.a);
        dB = uploadVec(p.b);
        dALog = uploadVec(p.logA);
        dDtBias = uploadVec(p.dtBias);
        int32_t* dIdx = nullptr;
        if (residentRows.empty())
        {
            dH0 = uploadVec(p.h0);
        }
        else
        {
            size_t const rowElems = static_cast<size_t>(hv) * kDk * kDv;
            std::vector<float> pool(static_cast<size_t>(poolRows) * rowElems);
            std::mt19937 rng(99);
            std::uniform_real_distribution<float> uni(-4.f, 4.f);
            for (auto& x : pool)
            {
                x = uni(rng);
            }
            for (int32_t bi = 0; bi < batch; ++bi)
            {
                int32_t const row = residentRows[bi];
                if (row >= 0 && row < poolRows)
                {
                    std::copy_n(p.h0.begin() + static_cast<size_t>(bi) * rowElems, rowElems,
                        pool.begin() + static_cast<size_t>(row) * rowElems);
                }
            }
            dH0 = uploadVec(pool);
            dIdx = uploadVec(residentRows);
        }

        size_t const oCount = static_cast<size_t>(batch) * numNodes * hv * kDv;
        ASSERT_EQ(cudaMalloc(&dO, oCount * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&dStash, static_cast<size_t>(batch) * rowBytes), cudaSuccess);
        ASSERT_EQ(
            cudaMalloc(&dMasks, static_cast<size_t>(batch) * numNodes * kGDN_TREE_CHUNK_MASK_WORDS * sizeof(uint32_t)),
            cudaSuccess);

        ASSERT_EQ(
            gdnTreeVerifyChunk(dH0, dIdx, residentRows.empty() ? batch : poolRows, dQ, dK, dV, dA, dB, dALog, dDtBias,
                dParents, dMasks, dO, dStash, rowBytes, batch, numNodes, h, hv, scale, useL2Norm, stream, path),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        cudaFree(dIdx);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    }

    ~DeviceRun()
    {
        if (stream != nullptr)
        {
            cudaStreamDestroy(stream);
        }
        for (void* ptr :
            {static_cast<void*>(dParents), static_cast<void*>(dMasks), static_cast<void*>(dQ), static_cast<void*>(dK),
                static_cast<void*>(dV), static_cast<void*>(dA), static_cast<void*>(dB), static_cast<void*>(dDtBias),
                static_cast<void*>(dALog), static_cast<void*>(dH0), static_cast<void*>(dO), static_cast<void*>(dStash)})
        {
            cudaFree(ptr);
        }
    }
};

void expectClose(double actual, double ref, double absTol, double relTol, std::string const& where)
{
    double const tol = absTol + relTol * std::abs(ref);
    EXPECT_NEAR(actual, ref, tol) << where;
}

} // anonymous namespace

// ============================================================================
// Verify (prep + apply) vs host reference
// ============================================================================

struct GdnTreeChunkVerifyParams
{
    int32_t numNodes;
    bool useL2Norm;
    int32_t heads{2};
};

class GdnTreeChunkVerifyTest : public ::testing::TestWithParam<GdnTreeChunkVerifyParams>
{
};

// Given two independent requests and the plugin's exact intermediate row capacity,
// When chunk verification runs across short and padded node counts,
// Then both requests match the numerical reference without requiring a larger buffer.
TEST_P(GdnTreeChunkVerifyTest, MatchesReference)
{
    auto const [numNodes, useL2Norm, heads] = GetParam();
    ChunkProblem p{};
    p.batch = 2;
    p.numNodes = numNodes;
    p.h = heads;
    p.hv = 4;
    p.addInvalidNode = true;
    p.generate(/*seed=*/1234 + numNodes);

    float const scale = 1.f / std::sqrt(static_cast<float>(kDk));
    DeviceRun dev;
    size_t const physicalRowBytes = static_cast<size_t>(numNodes) * p.hv * kDk * kDv * sizeof(float);
    ASSERT_NO_FATAL_FAILURE(dev.run(p, scale, useL2Norm, physicalRowBytes));

    std::vector<double> const ref = referenceVerifyOutput(p, scale, useL2Norm);
    std::vector<__half> const out = downloadVec(dev.dO, static_cast<size_t>(p.batch) * numNodes * p.hv * kDv);

    int32_t mismatches = 0;
    for (int32_t bi = 0; bi < p.batch; ++bi)
    {
        for (int32_t n = 0; n < numNodes; ++n)
        {
            bool const valid = p.nodeValid(bi, n);
            for (int32_t iHv = 0; iHv < p.hv; ++iHv)
            {
                for (int32_t vv = 0; vv < kDv; ++vv)
                {
                    size_t const idx = ((static_cast<size_t>(bi) * numNodes + n) * p.hv + iHv) * kDv + vv;
                    double const actual = __half2float(out[idx]);
                    if (!valid)
                    {
                        // Invalid (masked-out) nodes must produce exact zeros.
                        if (actual != 0.0)
                        {
                            ++mismatches;
                            EXPECT_EQ(actual, 0.0)
                                << "invalid node b=" << bi << " n=" << n << " hv=" << iHv << " vv=" << vv;
                        }
                        continue;
                    }
                    // fp16 output + fp16-staged TC accumulation over up to N
                    // terms: mixed tolerance.
                    double const tol = 2e-2 + 2e-2 * std::abs(ref[idx]);
                    if (!std::isfinite(actual) || std::abs(actual - ref[idx]) > tol)
                    {
                        ++mismatches;
                        EXPECT_NEAR(actual, ref[idx], tol) << "b=" << bi << " n=" << n << " hv=" << iHv << " vv=" << vv;
                    }
                    if (mismatches > 16)
                    {
                        FAIL() << "too many mismatches, aborting detailed reporting";
                    }
                }
            }
        }
    }
}

// N=7 is the chain-6 production shape. N=16/17 straddle the specialized
// two-warp apply dispatch boundary, while 33 exercises WMMA row padding.
INSTANTIATE_TEST_SUITE_P(Configurations, GdnTreeChunkVerifyTest,
    ::testing::Values(GdnTreeChunkVerifyParams{1, true}, GdnTreeChunkVerifyParams{4, true},
        GdnTreeChunkVerifyParams{7, true}, GdnTreeChunkVerifyParams{16, true}, GdnTreeChunkVerifyParams{17, true},
        GdnTreeChunkVerifyParams{32, true}, GdnTreeChunkVerifyParams{33, true}, GdnTreeChunkVerifyParams{48, true},
        GdnTreeChunkVerifyParams{64, true}, GdnTreeChunkVerifyParams{7, false}, GdnTreeChunkVerifyParams{2, true},
        GdnTreeChunkVerifyParams{1, true, 4}, GdnTreeChunkVerifyParams{2, true, 4}, GdnTreeChunkVerifyParams{15, true},
        GdnTreeChunkVerifyParams{31, true}, GdnTreeChunkVerifyParams{63, true}));

TEST(GdnTreeChunkVerifyRegressionTest, CorrelatedChainUsesSubstitutedRows)
{
    constexpr int32_t kNumNodes{7};
    ChunkProblem p{};
    p.batch = 1;
    p.numNodes = kNumNodes;
    p.h = 1;
    p.hv = 1;
    p.addInvalidNode = false;
    p.generate(/*seed=*/2026);

    for (int32_t n = 0; n < kNumNodes; ++n)
    {
        p.parents[n] = n - 1;
    }
    std::fill(p.q.begin(), p.q.end(), __float2half(1.f));
    std::fill(p.k.begin(), p.k.end(), __float2half(1.f));
    std::fill(p.v.begin(), p.v.end(), __float2half(0.5f));
    std::fill(p.a.begin(), p.a.end(), __float2half(-10.f));
    std::fill(p.b.begin(), p.b.end(), __float2half(10.f));
    std::fill(p.dtBias.begin(), p.dtBias.end(), __float2half(0.f));
    std::fill(p.logA.begin(), p.logA.end(), -10.f);
    std::fill(p.h0.begin(), p.h0.end(), 0.f);

    float const scale = 1.f / std::sqrt(static_cast<float>(kDk));
    DeviceRun dev;
    dev.run(p, scale, /*useL2Norm=*/true);

    std::vector<double> const ref = referenceVerifyOutput(p, scale, true);
    std::vector<__half> const out = downloadVec(dev.dO, ref.size());
    size_t const deepestNode = static_cast<size_t>(kNumNodes - 1) * kDv;
    expectClose(__half2float(out[deepestNode]), ref[deepestNode], 2e-2, 2e-2, "deepest chain node");
}

TEST(GdnTreeChunkVerifyRegressionTest, SameDepthSiblingsMatchReference)
{
    constexpr int32_t kNumNodes{17};
    ChunkProblem p{};
    p.batch = 1;
    p.numNodes = kNumNodes;
    p.h = 1;
    p.hv = 1;
    p.addInvalidNode = false;
    p.generate(/*seed=*/2027);

    for (int32_t n = 1; n < kNumNodes; ++n)
    {
        p.parents[n] = (n - 1) / 2;
    }
    std::fill(p.q.begin(), p.q.end(), __float2half(1.f));
    std::fill(p.k.begin(), p.k.end(), __float2half(1.f));

    float const scale = 1.f / std::sqrt(static_cast<float>(kDk));
    DeviceRun dev;
    dev.run(p, scale, /*useL2Norm=*/true);

    std::vector<double> const ref = referenceVerifyOutput(p, scale, true);
    std::vector<__half> const out = downloadVec(dev.dO, ref.size());
    for (size_t i = 0; i < ref.size(); ++i)
    {
        expectClose(__half2float(out[i]), ref[i], 2e-2, 2e-2, "same-depth sibling element " + std::to_string(i));
    }
}

// ============================================================================
// Verify -> replay commit end-to-end (stash handoff) vs host reference
// ============================================================================

TEST(GdnTreeChunkReplayTest, VerifyThenReplayMatchesReference)
{
    constexpr int32_t kNumLayers = 2;
    constexpr int32_t kStatePoolRows = 4;
    constexpr int32_t kBatch = 2;
    constexpr int32_t kH = 2;
    constexpr int32_t kHv = 6;
    std::vector<int32_t> const hostStateIndices{2, 0};
    std::vector<std::vector<int32_t>> const paths{
        {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30}, {0, 17, 23, 31, 39, 47}};
    ChunkProblem probs[kNumLayers];
    float const scale = 1.f / std::sqrt(static_cast<float>(kDk));

    DeviceRun devs[kNumLayers];
    std::vector<float*> persistent(kNumLayers);
    for (int32_t layer = 0; layer < kNumLayers; ++layer)
    {
        auto& p = probs[layer];
        p.batch = kBatch;
        p.numNodes = 48;
        p.h = kH;
        p.hv = kHv;
        p.addInvalidNode = false;
        p.generate(/*seed=*/777 + layer);
        for (int32_t batchIdx = 0; batchIdx < kBatch; ++batchIdx)
        {
            for (size_t i = 1; i < paths[batchIdx].size(); ++i)
            {
                p.parents[batchIdx * p.numNodes + paths[batchIdx][i]] = paths[batchIdx][i - 1];
            }
        }
        devs[layer].run(p, scale, /*useL2Norm=*/true);

        size_t const rowElements = probs[layer].h0.size() / kBatch;
        std::vector<float> statePool(static_cast<size_t>(kStatePoolRows) * rowElements, -17.F);
        for (int32_t batchIdx = 0; batchIdx < kBatch; ++batchIdx)
        {
            auto const inputBegin = probs[layer].h0.begin() + static_cast<size_t>(batchIdx) * rowElements;
            auto const outputBegin = statePool.begin() + static_cast<size_t>(hostStateIndices[batchIdx]) * rowElements;
            std::copy(inputBegin, inputBegin + rowElements, outputBegin);
        }
        persistent[layer] = uploadVec(statePool);
    }

    std::vector<int32_t> hostIndices(static_cast<size_t>(kBatch) * kGDN_TREE_CHUNK_MAX_ACCEPT, 0);
    std::vector<int32_t> hostLens(kBatch);
    for (int32_t batchIdx = 0; batchIdx < kBatch; ++batchIdx)
    {
        hostLens[batchIdx] = static_cast<int32_t>(paths[batchIdx].size());
        for (size_t i = 0; i < paths[batchIdx].size(); ++i)
        {
            hostIndices[static_cast<size_t>(batchIdx) * kGDN_TREE_CHUNK_MAX_ACCEPT + i] = paths[batchIdx][i];
        }
    }
    ASSERT_EQ(hostLens[0], kGDN_TREE_CHUNK_MAX_ACCEPT);
    int32_t* dIndices = uploadVec(hostIndices);
    int32_t* dLens = uploadVec(hostLens);
    int32_t* dStateIndices = uploadVec(hostStateIndices);

    std::vector<MtpLayerInfo> infos(kNumLayers);
    for (int32_t layer = 0; layer < kNumLayers; ++layer)
    {
        infos[layer].recurrentDst = persistent[layer];
        infos[layer].recurrentSrc = devs[layer].dStash;
        infos[layer].convDst = nullptr;
        infos[layer].convSrc = nullptr;
    }
    MtpLayerInfo* dInfos = uploadVec(infos);

    ASSERT_EQ(gdnTreeReplayCommitBatched(dInfos, kNumLayers, stashRowBytes(kH, kHv), dIndices, dLens, dStateIndices,
                  kBatch, kStatePoolRows, kGDN_TREE_CHUNK_MAX_ACCEPT, probs[0].numNodes, kH, kHv, nullptr),
        cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    for (int32_t layer = 0; layer < kNumLayers; ++layer)
    {
        std::vector<double> const ref = referenceReplayState(probs[layer], scale, true, paths);
        size_t const rowElements = ref.size() / kBatch;
        std::vector<float> const out
            = downloadVec(persistent[layer], static_cast<size_t>(kStatePoolRows) * rowElements);
        for (int32_t batchIdx = 0; batchIdx < kBatch; ++batchIdx)
        {
            for (size_t i = 0; i < rowElements; ++i)
            {
                // The replay chain is fp32 with an fp16 v; the stash k/g/beta are
                // fp32 rounded once.
                size_t const refIdx = static_cast<size_t>(batchIdx) * rowElements + i;
                size_t const outIdx = static_cast<size_t>(hostStateIndices[batchIdx]) * rowElements + i;
                expectClose(out[outIdx], ref[refIdx], 2e-3, 2e-3,
                    "layer " + std::to_string(layer) + " batch " + std::to_string(batchIdx) + " elem "
                        + std::to_string(i));
            }
        }
        for (int32_t slot = 0; slot < kStatePoolRows; ++slot)
        {
            if (std::find(hostStateIndices.begin(), hostStateIndices.end(), slot) != hostStateIndices.end())
            {
                continue;
            }
            auto const begin = out.begin() + static_cast<size_t>(slot) * rowElements;
            EXPECT_TRUE(std::all_of(begin, begin + rowElements, [](float value) { return value == -17.F; }))
                << "Unselected resident slot changed: layer=" << layer << " slot=" << slot;
        }
        cudaFree(persistent[layer]);
    }
    cudaFree(dIndices);
    cudaFree(dLens);
    cudaFree(dStateIndices);
    cudaFree(dInfos);
}

TEST(GdnTreeChunkReplayTest, ZeroAcceptLengthLeavesStateUnchanged)
{
    ChunkProblem p{};
    p.batch = 1;
    p.numNodes = 16;
    p.h = 2;
    p.hv = 4;
    p.addInvalidNode = false;
    p.generate(/*seed=*/4242);

    float const scale = 1.f / std::sqrt(static_cast<float>(kDk));
    DeviceRun dev;
    dev.run(p, scale, true);

    float* persistent = uploadVec(p.h0);
    std::vector<int32_t> const hostIndices(kGDN_TREE_CHUNK_MAX_ACCEPT, 0);
    int32_t* dIndices = uploadVec(hostIndices);
    std::vector<int32_t> const hostLens{0};
    int32_t* dLens = uploadVec(hostLens);
    std::vector<int32_t> const hostStateIndices{0};
    int32_t* dStateIndices = uploadVec(hostStateIndices);

    std::vector<MtpLayerInfo> infos(1);
    infos[0].recurrentDst = persistent;
    infos[0].recurrentSrc = dev.dStash;
    infos[0].convDst = nullptr;
    infos[0].convSrc = nullptr;
    MtpLayerInfo* dInfos = uploadVec(infos);

    ASSERT_EQ(gdnTreeReplayCommitBatched(dInfos, 1, stashRowBytes(2, 4), dIndices, dLens, dStateIndices, 1,
                  /*residentPoolRows=*/1, kGDN_TREE_CHUNK_MAX_ACCEPT, p.numNodes, 2, 4, nullptr),
        cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<float> const out = downloadVec(persistent, p.h0.size());
    for (size_t i = 0; i < p.h0.size(); ++i)
    {
        ASSERT_EQ(out[i], p.h0[i]) << "state changed at " << i << " despite acceptLength == 0";
    }
    cudaFree(persistent);
    cudaFree(dIndices);
    cudaFree(dLens);
    cudaFree(dStateIndices);
    cudaFree(dInfos);
}

TEST(GdnTreeChunkReplayTest, InvalidAcceptedPathLeavesPersistentStateUnchanged)
{
    ChunkProblem p{};
    p.batch = 1;
    p.numNodes = 16;
    p.h = 2;
    p.hv = 4;
    p.addInvalidNode = false;
    p.generate(/*seed=*/5150);

    float const scale = 1.f / std::sqrt(static_cast<float>(kDk));
    DeviceRun dev;
    dev.run(p, scale, true);

    struct InvalidPath
    {
        int32_t acceptLength;
        int32_t acceptedNode;
        int32_t declaredNumNodes;
    };
    std::vector<InvalidPath> const invalidPaths{
        {kGDN_TREE_CHUNK_MAX_ACCEPT + 1, 0, p.numNodes},
        // Node 15 exists in the allocated stash, but is outside the declared [0, 15) tree. This
        // validates the declared node domain without relying on an out-of-bounds device read.
        {1, p.numNodes - 1, p.numNodes - 1},
    };

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    for (auto const& invalid : invalidPaths)
    {
        float* persistent = uploadVec(p.h0);
        std::vector<int32_t> hostIndices(kGDN_TREE_CHUNK_MAX_ACCEPT, 0);
        hostIndices[0] = invalid.acceptedNode;
        int32_t* dIndices = uploadVec(hostIndices);
        int32_t* dLens = uploadVec(std::vector<int32_t>{invalid.acceptLength});
        int32_t* dStateIndices = uploadVec(std::vector<int32_t>{0});
        std::vector<MtpLayerInfo> infos(1);
        infos[0].recurrentDst = persistent;
        infos[0].recurrentSrc = dev.dStash;
        MtpLayerInfo* dInfos = uploadVec(infos);

        ASSERT_EQ(gdnTreeReplayCommitBatched(dInfos, 1, stashRowBytes(2, 4), dIndices, dLens, dStateIndices, 1,
                      /*residentPoolRows=*/1, kGDN_TREE_CHUNK_MAX_ACCEPT, invalid.declaredNumNodes, 2, 4, stream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        std::vector<float> const out = downloadVec(persistent, p.h0.size());
        EXPECT_EQ(out, p.h0) << "malformed accepted path committed persistent state: length=" << invalid.acceptLength
                             << " node=" << invalid.acceptedNode << " numNodes=" << invalid.declaredNumNodes;
        cudaFree(persistent);
        cudaFree(dIndices);
        cudaFree(dLens);
        cudaFree(dStateIndices);
        cudaFree(dInfos);
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

class GdnTreeChunkPrefixReplayTest : public ::testing::TestWithParam<int32_t>
{
};

TEST_P(GdnTreeChunkPrefixReplayTest, ImplicitPrefixMatchesNonzeroStateReference)
{
    // Given nonzero resident states and exact plugin-sized intermediate rows,
    // When replay commits unequal implicit prefixes after chunk verification,
    // Then each mapped resident state matches its own prefix reference.
    ChunkProblem problem;
    problem.batch = 2;
    problem.numNodes = GetParam();
    problem.h = 2;
    problem.hv = 4;
    problem.addInvalidNode = false;
    problem.generate(917);
    for (int32_t batch = 0; batch < problem.batch; ++batch)
    {
        for (int32_t node = 0; node < problem.numNodes; ++node)
        {
            problem.parents[batch * problem.numNodes + node] = node - 1;
        }
    }
    float const scale = 1.F / std::sqrt(static_cast<float>(kDk));
    DeviceRun device;
    size_t const physicalRowBytes = static_cast<size_t>(problem.numNodes) * problem.hv * kDk * kDv * sizeof(float);
    ASSERT_NO_FATAL_FAILURE(device.run(problem, scale, true, physicalRowBytes));
    size_t const rowElements = problem.h0.size() / problem.batch;
    std::vector<float> states(problem.h0.size());
    std::copy(problem.h0.begin(), problem.h0.begin() + rowElements, states.begin() + rowElements);
    std::copy(problem.h0.begin() + rowElements, problem.h0.end(), states.begin());
    float* persistent = uploadVec(states);
    int32_t const firstLength = std::min(3, problem.numNodes);
    int32_t const secondLength = firstLength - 1;
    int32_t* lengths = uploadVec(std::vector<int32_t>{firstLength, secondLength});
    int32_t* slots = uploadVec(std::vector<int32_t>{1, 0});
    MtpLayerInfo info{};
    info.recurrentDst = persistent;
    info.recurrentSrc = device.dStash;
    MtpLayerInfo* infos = uploadVec(std::vector<MtpLayerInfo>{info});
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_EQ(gdnTreeReplayCommitBatched(
                  infos, 1, physicalRowBytes, nullptr, lengths, slots, 2, 2, 4, problem.numNodes, 2, 4, stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    auto const actual = downloadVec(persistent, states.size());
    std::vector<std::vector<int32_t>> paths(2);
    for (int32_t node = 0; node < firstLength; ++node)
    {
        paths[0].push_back(node);
        if (node < secondLength)
        {
            paths[1].push_back(node);
        }
    }
    auto const reference = referenceReplayState(problem, scale, true, paths);
    double maxError = 0;
    for (int32_t batch = 0; batch < 2; ++batch)
    {
        for (size_t element = 0; element < rowElements; ++element)
        {
            double const error
                = std::abs(actual[(1 - batch) * rowElements + element] - reference[batch * rowElements + element]);
            maxError = std::isfinite(error) ? std::max(maxError, error) : std::numeric_limits<double>::infinity();
        }
    }
    EXPECT_LT(maxError, 0.003);
    cudaFree(infos);
    cudaFree(slots);
    cudaFree(lengths);
    cudaFree(persistent);
    cudaStreamDestroy(stream);
}

INSTANTIATE_TEST_SUITE_P(PhysicalRows, GdnTreeChunkPrefixReplayTest, ::testing::Values(1, 2, 8));

// ============================================================================
// Row independence: a node's output depends only on its own inclusive ancestor chain
// ============================================================================

namespace
{

//! Tree B: tree A with every input that a node off `keep`'s ancestor chain owns redrawn, and that node re-parented
//! at random, so the two trees agree only on `keep`'s chain.
ChunkProblem redrawOutside(ChunkProblem const& a, int32_t keep, uint32_t seed)
{
    ChunkProblem b = a;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<int32_t> const chain = a.ancestors(0, keep);
    auto const onChain = [&](int32_t n) { return std::find(chain.begin(), chain.end(), n) != chain.end(); };
    for (int32_t n = 1; n < a.numNodes; ++n)
    {
        if (onChain(n))
        {
            continue;
        }
        // Chain nodes keep their parents, so re-parenting any other node leaves every chain node's ancestors as is.
        b.parents[n] = static_cast<int32_t>(rng() % static_cast<uint32_t>(n));
        for (int32_t i = 0; i < a.h * kDk; ++i)
        {
            b.q[static_cast<size_t>(n) * a.h * kDk + i] = __float2half(uni(rng));
            b.k[static_cast<size_t>(n) * a.h * kDk + i] = __float2half(uni(rng));
        }
        for (int32_t i = 0; i < a.hv * kDv; ++i)
        {
            b.v[static_cast<size_t>(n) * a.hv * kDv + i] = __float2half(uni(rng));
        }
        for (int32_t i = 0; i < a.hv; ++i)
        {
            b.a[static_cast<size_t>(n) * a.hv + i] = __float2half(uni(rng));
            b.b[static_cast<size_t>(n) * a.hv + i] = __float2half(uni(rng));
        }
    }
    return b;
}

std::vector<uint16_t> nodeBits(std::vector<__half> const& out, int32_t numNodes, int32_t hv, int32_t node)
{
    std::vector<uint16_t> bits(static_cast<size_t>(hv) * kDv);
    for (int32_t i = 0; i < hv * kDv; ++i)
    {
        __half const value = out[static_cast<size_t>(node) * hv * kDv + i];
        std::memcpy(&bits[static_cast<size_t>(i)], &value, sizeof(uint16_t));
    }
    return bits;
}

} // namespace

class GdnTreeChunkRowIndependenceTest : public ::testing::TestWithParam<int32_t>
{
};

// Given one request's state h0 and a random tree of N nodes, and a second tree of N nodes that keeps the root's inputs
// and the inputs and parents of one leaf's ancestor chain but redraws every other node's q, k, v, a, b and parent
// When chunk verification runs on both trees
// Then the root row and that leaf's row are bitwise identical between the two trees, and so is the root row of a
// one-node verification of the same root: a verify row reads nothing outside its own ancestor chain, whatever the tree
// width, so all-reject speculative decoding commits the same root output under any draft
TEST_P(GdnTreeChunkRowIndependenceTest, RowsDependOnlyOnTheirAncestorChain)
{
    int32_t const numNodes = GetParam();
    ChunkProblem a{};
    a.batch = 1;
    a.numNodes = numNodes;
    a.h = 2;
    a.hv = 4;
    a.addInvalidNode = false;
    a.generate(/*seed=*/4242 + numNodes);
    int32_t const leaf = numNodes - 1;
    ChunkProblem const b = redrawOutside(a, leaf, /*seed=*/77 + numNodes);
    ChunkProblem root = a;
    root.numNodes = 1;
    root.parents = {-1};
    root.q.resize(static_cast<size_t>(a.h) * kDk);
    root.k.resize(static_cast<size_t>(a.h) * kDk);
    root.v.resize(static_cast<size_t>(a.hv) * kDv);
    root.a.resize(a.hv);
    root.b.resize(a.hv);

    float const scale = 1.f / std::sqrt(static_cast<float>(kDk));
    auto const verify = [&](ChunkProblem const& p) {
        DeviceRun dev;
        size_t const rowBytes = static_cast<size_t>(p.numNodes) * p.hv * kDk * kDv * sizeof(float);
        dev.run(p, scale, /*useL2Norm=*/true, std::max(rowBytes, stashRowBytes(p.h, p.hv)));
        return downloadVec(dev.dO, static_cast<size_t>(p.numNodes) * p.hv * kDv);
    };
    std::vector<__half> const outA = verify(a);
    std::vector<__half> const outB = verify(b);
    std::vector<__half> const outRoot = verify(root);

    // The fixture must change what a leaky kernel would read: some row off the kept chain differs.
    std::vector<int32_t> const chain = a.ancestors(0, leaf);
    bool offChainRowDiffers = false;
    for (int32_t n = 1; n < numNodes; ++n)
    {
        if (std::find(chain.begin(), chain.end(), n) == chain.end())
        {
            offChainRowDiffers |= nodeBits(outA, numNodes, a.hv, n) != nodeBits(outB, numNodes, a.hv, n);
        }
    }
    EXPECT_TRUE(offChainRowDiffers || static_cast<int32_t>(chain.size()) == numNodes)
        << "the redrawn tree left every row off the kept chain unchanged";
    EXPECT_EQ(nodeBits(outA, numNodes, a.hv, 0), nodeBits(outB, numNodes, a.hv, 0)) << "root row";
    EXPECT_EQ(nodeBits(outA, numNodes, a.hv, leaf), nodeBits(outB, numNodes, a.hv, leaf)) << "kept leaf row";
    EXPECT_EQ(nodeBits(outA, numNodes, a.hv, 0), nodeBits(outRoot, 1, a.hv, 0)) << "root row vs one-node verify";
}

// 16/17 straddle the split path's warp-apply / block-apply boundary where the fused kernel does not fit; 33 and 64
// add WMMA K tiles.
INSTANTIATE_TEST_SUITE_P(TreeSizes, GdnTreeChunkRowIndependenceTest, ::testing::Values(2, 7, 16, 17, 33, 64));

// ============================================================================
// Fused vs split verify kernels
// ============================================================================

class GdnTreeChunkFusedSplitTest : public ::testing::TestWithParam<int32_t>
{
};

// Given three random tree problems of four requests with 16 v-heads and a padding node, read from a six-row resident
// pool
// When chunk verification runs once on the fused kernel and once on the split prep + apply kernels
// Then the outputs and the replay stash are bitwise identical: the fused kernel keeps the split kernels' arithmetic
// and order across the warp-apply (N <= 16) and block-apply (N > 16) sizes. The fp16 output hides most last-bit fp32
// differences, hence many outputs per size.
TEST_P(GdnTreeChunkFusedSplitTest, FusedMatchesSplitBitwise)
{
    int32_t const numNodes = GetParam();
    if (!gdnTreeVerifyFusedFits(numNodes))
    {
        GTEST_SKIP() << "fused verify does not fit this device's shared memory at N=" << numNodes;
    }
    for (uint32_t const seed : {909u, 1909u, 2909u})
    {
        ChunkProblem p{};
        p.batch = 4;
        p.numNodes = numNodes;
        p.h = 4;
        p.hv = 16;
        p.addInvalidNode = true;
        p.generate(seed + static_cast<uint32_t>(numNodes));

        float const scale = 1.f / std::sqrt(static_cast<float>(kDk));
        size_t const rowBytes
            = std::max(static_cast<size_t>(numNodes) * p.hv * kDk * kDv * sizeof(float), stashRowBytes(p.h, p.hv));
        std::vector<int32_t> const rows{3, 0, 5, 1};
        DeviceRun fused;
        ASSERT_NO_FATAL_FAILURE(
            fused.run(p, scale, /*useL2Norm=*/true, rowBytes, rows, /*poolRows=*/6, GdnTreeVerifyPath::kFused));
        DeviceRun split;
        ASSERT_NO_FATAL_FAILURE(
            split.run(p, scale, /*useL2Norm=*/true, rowBytes, rows, /*poolRows=*/6, GdnTreeVerifyPath::kSplit));

        size_t const oCount = static_cast<size_t>(p.batch) * numNodes * p.hv * kDv;
        std::vector<__half> const oFused = downloadVec(fused.dO, oCount);
        std::vector<__half> const oSplit = downloadVec(split.dO, oCount);
        EXPECT_EQ(0, std::memcmp(oFused.data(), oSplit.data(), oCount * sizeof(__half))) << "outputs, seed " << seed;
        size_t const stashBytes = static_cast<size_t>(numNodes) * gdnTreeStashNodeBytes(p.h, p.hv);
        for (int32_t bi = 0; bi < p.batch; ++bi)
        {
            std::vector<char> const sFused = downloadVec(fused.dStash + bi * rowBytes, stashBytes);
            std::vector<char> const sSplit = downloadVec(split.dStash + bi * rowBytes, stashBytes);
            EXPECT_EQ(0, std::memcmp(sFused.data(), sSplit.data(), stashBytes))
                << "replay stash, request " << bi << ", seed " << seed;
        }
    }
}

// 16/17 straddle the split path's warp/block apply boundary; 33, 48 and 64 add Attn @ R K tiles.
INSTANTIATE_TEST_SUITE_P(TreeSizes, GdnTreeChunkFusedSplitTest, ::testing::Values(1, 7, 16, 17, 33, 48, 64));

// Given the Qwen3.8-27B GDN geometry (16 key heads, 48 value heads) and every tree size the chunk verify accepts
// When the verify row stride is computed on this device
// Then a size the fused kernel runs holds only the replay stash, and a size the split kernels run also holds their
// KS/QS and prep scratch
TEST(GdnTreeChunkRowBytesTest, RowsHoldSplitScratchOnlyWhereSplitRuns)
{
    int32_t const h = 16;
    int32_t const hv = 48;
    for (int32_t numNodes = 1; numNodes <= kGDN_TREE_CHUNK_MAX_NODES; ++numNodes)
    {
        size_t const stashBytes
            = (static_cast<size_t>(numNodes) * gdnTreeStashNodeBytes(h, hv) + 255) & ~static_cast<size_t>(255);
        size_t const rowBytes = gdnTreeVerifyRowBytes(h, hv, numNodes);
        if (gdnTreeVerifyFusedFits(numNodes))
        {
            EXPECT_EQ(rowBytes, stashBytes) << "fused, numNodes " << numNodes;
        }
        else
        {
            EXPECT_GT(rowBytes, stashBytes) << "split, numNodes " << numNodes;
        }
    }
}
