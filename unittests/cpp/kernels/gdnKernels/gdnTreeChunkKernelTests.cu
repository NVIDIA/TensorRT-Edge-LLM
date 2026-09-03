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
    ChunkProblem const& p, float scale, bool useL2Norm, std::vector<int32_t> const& acceptedPath)
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
            for (int32_t node : acceptedPath)
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

class GdnTreeChunkVerifyTest : public ::testing::TestWithParam<int32_t>
{
};

TEST_P(GdnTreeChunkVerifyTest, MatchesReference)
{
    int32_t const numNodes = GetParam();
    ChunkProblem p{};
    p.batch = 2;
    p.numNodes = numNodes;
    p.h = 2;
    p.hv = 4;
    p.addInvalidNode = true;
    p.generate(/*seed=*/1234 + numNodes);

    float const scale = 1.f / std::sqrt(static_cast<float>(kDk));
    DeviceRun dev;
    dev.run(p, scale, /*useL2Norm=*/true);

    std::vector<double> const ref = referenceVerifyOutput(p, scale, true);
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

// 33 exercises the nPad > N wmma padding paths; 4 exercises tiny trees.
INSTANTIATE_TEST_SUITE_P(NodeCounts, GdnTreeChunkVerifyTest, ::testing::Values(4, 16, 33, 48, 64));

// ============================================================================
// Verify -> replay commit end-to-end (stash handoff) vs host reference
// ============================================================================

TEST(GdnTreeChunkReplayTest, VerifyThenReplayMatchesReference)
{
    constexpr int32_t kNumLayers = 2;
    ChunkProblem probs[kNumLayers];
    float const scale = 1.f / std::sqrt(static_cast<float>(kDk));

    DeviceRun devs[kNumLayers];
    std::vector<float*> persistent(kNumLayers);
    for (int32_t layer = 0; layer < kNumLayers; ++layer)
    {
        auto& p = probs[layer];
        p.batch = 1;
        p.numNodes = 48;
        p.h = 2;
        p.hv = 4;
        p.addInvalidNode = false;
        p.generate(/*seed=*/777 + layer);
        devs[layer].run(p, scale, /*useL2Norm=*/true);

        // Persistent recurrent state starts at h0 and is committed in place.
        persistent[layer] = uploadVec(probs[layer].h0);
    }

    // Accepted path: a root-to-leaf chain in layer 0's tree (trees differ per
    // layer only in the stash contents; the accepted indices are shared).
    std::vector<int32_t> path;
    {
        // Walk from the last node up to the root, then reverse.
        int32_t iter = probs[0].numNodes - 1;
        while (iter >= 0 && static_cast<int32_t>(path.size()) < kGDN_TREE_CHUNK_MAX_ACCEPT)
        {
            path.push_back(iter);
            iter = probs[0].parents[iter];
        }
        std::reverse(path.begin(), path.end());
    }
    int32_t const acceptLen = static_cast<int32_t>(path.size());
    ASSERT_GE(acceptLen, 2);

    std::vector<int32_t> hostIndices(kGDN_TREE_CHUNK_MAX_ACCEPT, 0);
    for (int32_t i = 0; i < acceptLen; ++i)
    {
        hostIndices[i] = path[i];
    }
    int32_t* dIndices = uploadVec(hostIndices);
    std::vector<int32_t> const hostLens{acceptLen};
    int32_t* dLens = uploadVec(hostLens);

    std::vector<MtpLayerInfo> infos(kNumLayers);
    for (int32_t layer = 0; layer < kNumLayers; ++layer)
    {
        infos[layer].recurrentDst = persistent[layer];
        infos[layer].recurrentSrc = devs[layer].dStash;
        infos[layer].convDst = nullptr;
        infos[layer].convSrc = nullptr;
    }
    MtpLayerInfo* dInfos = uploadVec(infos);

    ASSERT_EQ(gdnTreeReplayCommitBatched(dInfos, kNumLayers, stashRowBytes(2, 4), dIndices, dLens, /*batch=*/1,
                  kGDN_TREE_CHUNK_MAX_ACCEPT, probs[0].numNodes, /*h=*/2, /*hv=*/4, nullptr),
        cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    for (int32_t layer = 0; layer < kNumLayers; ++layer)
    {
        std::vector<double> const ref = referenceReplayState(probs[layer], scale, true, path);
        std::vector<float> const out = downloadVec(persistent[layer], ref.size());
        for (size_t i = 0; i < ref.size(); ++i)
        {
            // The replay chain is fp32 with an fp16 v; the stash k/g/beta are
            // fp32 rounded once.
            expectClose(out[i], ref[i], 2e-3, 2e-3, "layer " + std::to_string(layer) + " elem " + std::to_string(i));
        }
        cudaFree(persistent[layer]);
    }
    cudaFree(dIndices);
    cudaFree(dLens);
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

    std::vector<MtpLayerInfo> infos(1);
    infos[0].recurrentDst = persistent;
    infos[0].recurrentSrc = dev.dStash;
    infos[0].convDst = nullptr;
    infos[0].convSrc = nullptr;
    MtpLayerInfo* dInfos = uploadVec(infos);

    ASSERT_EQ(gdnTreeReplayCommitBatched(dInfos, 1, stashRowBytes(2, 4), dIndices, dLens, 1, kGDN_TREE_CHUNK_MAX_ACCEPT,
                  p.numNodes, 2, 4, nullptr),
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
    cudaFree(dInfos);
}
