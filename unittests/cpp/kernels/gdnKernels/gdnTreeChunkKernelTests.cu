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

    void run(ChunkProblem const& p, float scale, bool useL2Norm)
    {
        batch = p.batch;
        numNodes = p.numNodes;
        h = p.h;
        hv = p.hv;
        rowBytes = stashRowBytes(h, hv);

        dParents = uploadVec(p.parents);
        dQ = uploadVec(p.q);
        dK = uploadVec(p.k);
        dV = uploadVec(p.v);
        dA = uploadVec(p.a);
        dB = uploadVec(p.b);
        dALog = uploadVec(p.logA);
        dDtBias = uploadVec(p.dtBias);
        dH0 = uploadVec(p.h0);

        size_t const oCount = static_cast<size_t>(batch) * numNodes * hv * kDv;
        ASSERT_EQ(cudaMalloc(&dO, oCount * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&dStash, static_cast<size_t>(batch) * rowBytes), cudaSuccess);
        ASSERT_EQ(
            cudaMalloc(&dMasks, static_cast<size_t>(batch) * numNodes * kGDN_TREE_CHUNK_MASK_WORDS * sizeof(uint32_t)),
            cudaSuccess);

        ASSERT_EQ(gdnTreeBuildAncestorMasks(dParents, dMasks, batch, numNodes, numNodes, nullptr), cudaSuccess);
        ASSERT_EQ(gdnTreeVerifyChunk(dH0, dQ, dK, dV, dA, dB, dALog, dDtBias, dMasks, dO, dStash, rowBytes, batch,
                      numNodes, h, hv, scale, useL2Norm, nullptr),
            cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    }

    ~DeviceRun()
    {
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
};

class GdnTreeChunkVerifyTest : public ::testing::TestWithParam<GdnTreeChunkVerifyParams>
{
};

TEST_P(GdnTreeChunkVerifyTest, MatchesReference)
{
    auto const [numNodes, useL2Norm] = GetParam();
    ChunkProblem p{};
    p.batch = 2;
    p.numNodes = numNodes;
    p.h = 2;
    p.hv = 4;
    p.addInvalidNode = true;
    p.generate(/*seed=*/1234 + numNodes);

    float const scale = 1.f / std::sqrt(static_cast<float>(kDk));
    DeviceRun dev;
    dev.run(p, scale, useL2Norm);

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
                    if (std::abs(actual - ref[idx]) > tol)
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
        GdnTreeChunkVerifyParams{64, true}, GdnTreeChunkVerifyParams{7, false}));

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
