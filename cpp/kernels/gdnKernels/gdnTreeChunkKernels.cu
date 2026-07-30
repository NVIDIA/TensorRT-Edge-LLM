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

// Stateless chunk-form GDN speculative verify + replay commit kernels. Both a
// branching DDTree and a linear verification chain are represented by packed
// ancestor masks. The implementation uses a V-split grid for occupancy (one
// CTA per (batch, hv-head, v-quarter)), zero-barrier per-column forward
// substitution, vectorized phases, and batched replay across layers.
// Validated against the fp64 golden in workspace/gdn-tree-poc (ulp_max=1).

#include "gdnTreeChunkKernels.h"

#include "common/logger.h"
#include "kernels/speculative/mtpStateScatterKernels.h"

#include <cstdlib>
#include <cstring>
#include <mma.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

// Problem shape (fixed by the GDN plugin contract).
constexpr int32_t kTILE_K{128};                            // head dim (dk = dv); also the h0 state edge (128x128)
constexpr int32_t kN_MAX{kGDN_TREE_CHUNK_MAX_NODES};       // static tree-size cap: smem M/Attn tiles and the prep
                                                           // block are laid out [kN_MAX][...] so offsets/strides are
                                                           // N-independent; larger trees are unsupported
constexpr int32_t kMASK_WORDS{kGDN_TREE_CHUNK_MASK_WORDS}; // uint32 words per ancestor bitmask (kN_MAX / 32)

// Parallelization strategy.
constexpr int32_t kNUM_THREADS{256};          // threads per CTA (8 warps), all kernels
constexpr int32_t kV_SPLIT{4};                // apply kernel: v/output columns split across this many CTAs per
                                              // (batch, hv) — the per-column forward substitution is the serial
                                              // bottleneck, so splitting columns raises occupancy
constexpr int32_t kVCOLS{kTILE_K / kV_SPLIT}; // output columns owned by one v-split CTA

// Verify -> replay handoff cell. The chunk-form verify is stateless (reads
// h0 only), so for each tree node it stashes just the quantities the replay
// commit needs to re-run the state update S <- gamma*S + beta*(v - S*k)k^T
// along the accepted path — ~160x smaller than materializing a full
// per-node [hv, 128, 128] fp32 state checkpoint.
//
// Byte offsets within one (batch, node) cell, densely packed:
// [kNorm fp32 h*128 | g fp32 hv | beta fp32 hv | v fp16 hv*128]
struct StashOffsets
{
    size_t kNorm;     // L2-normalized k, one 128-dim fp32 vector per k-head
    size_t g;         // log decay gate (log gamma), one fp32 per v-head
    size_t beta;      // write strength (post-sigmoid), one fp32 per v-head
    size_t v;         // value vectors, one 128-dim fp16 vector per v-head
    size_t nodeBytes; // total cell size; cells are packed at this stride
};

__host__ __device__ inline StashOffsets stashOffsets(int32_t h, int32_t hv)
{
    StashOffsets o;
    o.kNorm = 0;
    o.g = o.kNorm + static_cast<size_t>(h) * kTILE_K * sizeof(float);
    o.beta = o.g + static_cast<size_t>(hv) * sizeof(float);
    o.v = o.beta + static_cast<size_t>(hv) * sizeof(float);
    o.nodeBytes = o.v + static_cast<size_t>(hv) * kTILE_K * sizeof(__half);
    return o;
}

// Turns the DDTree parent-pointer array into per-node inclusive ancestor
// bitmasks — the causal mask of the chunk-form tree attention. The verify
// kernel treats the whole tree as one N-token chunk; node n may only absorb
// contributions from nodes on its own root path, so every (n, j) element of
// the M/Attn matrices is gated on bit j of mask[n].
//
// One CTA per batch, one thread per node: each thread walks its parent chain
// up to the root and sets one bit per visited node (O(tree depth) steps).
//
// Invalid/padding nodes: node 0 must be the root (parent < 0) and every
// other node must point at an earlier node (topological order). A node
// failing this check skips the walk, so its zero-initialized local mask is
// written out as-is — an all-zero mask, which gates the node's entire
// M/Attn row to zero in the verify kernel and yields exact-zero outputs.
//
// The check covers each node's own parent only, NOT the whole chain: a
// "valid" node whose ancestor chain passes through a padding node would get
// a truncated mask that stops there instead of at the root. Correctness
// therefore relies on the DDTree build contract — trees are prefix-closed
// and padding nodes are never referenced as a parent — which
// buildDDTreeKernel guarantees. maxDepth only bounds the walk against
// malformed (cyclic) input.
//
// Inputs:
//   parentIds [batch, numNodes] int32 — parent index per node, root = -1
// Outputs:
//   masksOut [batch, numNodes, kMASK_WORDS] uint32 — inclusive ancestor
//     bitmask per node; bit j of mask[b][n] = node j is on n's root path
__global__ void buildAncestorMasksKernel(
    int32_t const* parentIds, uint32_t* masksOut, int32_t numNodes, int32_t maxDepth)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const node = threadIdx.x;
    if (node >= numNodes)
    {
        return;
    }

    uint32_t words[kMASK_WORDS] = {};
    if (parentIds == nullptr)
    {
        // Linear speculative verification is a degenerate tree: node n's
        // ancestors are exactly [0, n]. Build its causal mask without adding
        // tree metadata to the exported engine contract.
        for (int32_t ancestor = 0; ancestor <= node; ++ancestor)
        {
            words[ancestor / 32] |= (1u << (ancestor % 32));
        }
    }
    else
    {
        int32_t const* parents = parentIds + static_cast<int64_t>(batchIdx) * numNodes;
        bool const valid = (node == 0) ? (parents[0] < 0) : (parents[node] >= 0 && parents[node] < node);
        if (valid)
        {
            int32_t iter = node;
            for (int32_t d = 0; d <= maxDepth && iter >= 0; ++d)
            {
                words[iter / 32] |= (1u << (iter % 32));
                iter = parents[iter];
            }
        }
    }

    uint32_t* out = masksOut + (static_cast<int64_t>(batchIdx) * numNodes + node) * kMASK_WORDS;
#pragma unroll
    for (int32_t w = 0; w < kMASK_WORDS; ++w)
    {
        out[w] = words[w];
    }
}

// Prep-block layout in the plugin workspace, one block per (batch, hv-head).
// Word strides fixed by kN_MAX so the layout is N-independent; see
// kGDN_TREE_CHUNK_PREP_WORDS in the header.
constexpr int32_t kPREP_M{0};                       // [kN_MAX * kN_MAX] f32
constexpr int32_t kPREP_ATTN{kN_MAX * kN_MAX};      // [kN_MAX * kN_MAX] f32
constexpr int32_t kPREP_GAMMA{2 * kN_MAX * kN_MAX}; // [kN_MAX] f32
constexpr int32_t kPREP_BETA{kPREP_GAMMA + kN_MAX}; // [kN_MAX] f32
constexpr int32_t kPREP_INVK{kPREP_BETA + kN_MAX};  // [kN_MAX] f32
constexpr int32_t kPREP_INVQ{kPREP_INVK + kN_MAX};  // [kN_MAX] f32
constexpr int32_t kPREP_DEPTH{kPREP_INVQ + kN_MAX}; // [kN_MAX] i32 + maxDepth i32

// Verify scratch lives in the unused tail of each intermediate-states batch
// row (past the last stash cell) — it must NOT live in the plugin workspace:
// engine plans serialize the workspace requirement at build time, so growing
// it without rebuilding engines causes out-of-bounds writes. Row-tail layout
// (all 256B-aligned):
//   [stash cells][KS/QS: hv x 2 x kN_MAX x kTILE_K f32][prep: hv x PREP_WORDS f32]
// The launcher validates capacity and refuses to launch on overflow.
__host__ __device__ inline size_t gdnTreeKsQsBaseBytes(StashOffsets const& so)
{
    return (static_cast<size_t>(kN_MAX) * so.nodeBytes + 255) & ~static_cast<size_t>(255);
}

__host__ __device__ inline size_t gdnTreePrepBaseBytes(StashOffsets const& so, int32_t hv)
{
    return gdnTreeKsQsBaseBytes(so) + static_cast<size_t>(hv) * 2 * kN_MAX * kTILE_K * sizeof(float);
}

// Prep/apply split: Phases 1-3 (q/k load, gates, norms, cumlog/depth, replay stash,
// M/Attn dual masked GEMM) are independent of the v-split, so they used to be
// computed redundantly by all kV_SPLIT CTAs. The prep kernel now runs them
// once per (batch, hv-head) and exports M/Attn + per-node vectors to the
// intermediate-states row tail (see gdnTreePrepBaseBytes); the apply kernel
// (one CTA per (batch, hv, v-quarter)) consumes them for Phases 4a-c.
__global__ void __launch_bounds__(kNUM_THREADS) treeVerifyChunkPrepKernel(__half const* __restrict__ q,
    __half const* __restrict__ k, __half const* __restrict__ v, __half const* __restrict__ a,
    __half const* __restrict__ b, float const* __restrict__ A_log, __half const* __restrict__ dt_bias,
    uint32_t const* __restrict__ masks, char* __restrict__ stash, size_t stashBatchStrideBytes,
    float const* __restrict__ h0, int32_t numNodes, int32_t H, int32_t HV, float scale, int32_t useQKL2Norm)
{
    using namespace nvcuda;

    int32_t const stateIdx = blockIdx.x;
    int32_t const iN = stateIdx / HV;
    int32_t const iHv = stateIdx % HV;
    int32_t const iH = iHv / (HV / H);
    int32_t const tid = threadIdx.x;
    int32_t const N = numNodes;

    extern __shared__ char smemRaw[];
    __half* sK = reinterpret_cast<__half*>(smemRaw);                // [N][K]
    __half* sQ = sK + kN_MAX * kTILE_K;                             // [N][K]
    __half* sH0h = sQ + kN_MAX * kTILE_K;                           // [K][K] fp16 (KS/QS GEMM B)
    float* sM = reinterpret_cast<float*>(sH0h + kTILE_K * kTILE_K); // [N][N]
    float* sAttn = sM + kN_MAX * kN_MAX;                            // [N][N]
    float* sCum = sAttn + kN_MAX * kN_MAX;
    float* sGamma = sCum + kN_MAX;
    float* sLogG = sGamma + kN_MAX;
    float* sBeta = sLogG + kN_MAX;
    float* sInvK = sBeta + kN_MAX;
    float* sInvQ = sInvK + kN_MAX;
    uint32_t* sMask = reinterpret_cast<uint32_t*>(sInvQ + kN_MAX);
    int32_t* sDepth = reinterpret_cast<int32_t*>(sMask + kN_MAX * kMASK_WORDS); // [N] tree depth per node
    int32_t* sMaxDepth = sDepth + kN_MAX;                                       // single int

    // ---- Phase 1: vectorized q/k load, gates, norms, masks ----------------
    {
        __half2 const* qg = reinterpret_cast<__half2 const*>(q + (static_cast<int64_t>(iN) * N * H + iH) * kTILE_K);
        __half2 const* kg = reinterpret_cast<__half2 const*>(k + (static_cast<int64_t>(iN) * N * H + iH) * kTILE_K);
        __half2* sK2 = reinterpret_cast<__half2*>(sK);
        __half2* sQ2 = reinterpret_cast<__half2*>(sQ);
        int32_t const rowStrideH2 = H * kTILE_K / 2;
        for (int32_t idx = tid; idx < N * kTILE_K / 2; idx += kNUM_THREADS)
        {
            int32_t const n = idx / (kTILE_K / 2);
            int32_t const kk = idx % (kTILE_K / 2);
            sK2[n * kTILE_K / 2 + kk] = kg[static_cast<int64_t>(n) * rowStrideH2 + kk];
            sQ2[n * kTILE_K / 2 + kk] = qg[static_cast<int64_t>(n) * rowStrideH2 + kk];
        }
    }
    if (tid < N * kMASK_WORDS)
    {
        sMask[tid] = masks[(static_cast<int64_t>(iN) * N) * kMASK_WORDS + tid];
    }
    // Full h0 tile [K][K] as fp16: B operand of the KS/QS GEMM below.
    for (int32_t idx = tid; idx < kTILE_K * kTILE_K; idx += kNUM_THREADS)
    {
        sH0h[idx] = __float2half(h0[(static_cast<int64_t>(iN) * HV + iHv) * kTILE_K * kTILE_K + idx]);
    }
    __syncthreads();

    if (tid < N)
    {
        float const av = __half2float(a[(static_cast<int64_t>(iN) * N + tid) * HV + iHv]);
        float const bv = __half2float(b[(static_cast<int64_t>(iN) * N + tid) * HV + iHv]);
        float const x = av + __half2float(dt_bias[iHv]);
        float const sp = (x > 20.f) ? x : log1pf(expf(x));
        sLogG[tid] = -expf(A_log[iHv]) * sp;
        sBeta[tid] = 1.f / (1.f + expf(-bv));
        __half2 const* k2 = reinterpret_cast<__half2 const*>(sK + tid * kTILE_K);
        __half2 const* q2 = reinterpret_cast<__half2 const*>(sQ + tid * kTILE_K);
        float sk2 = 0.f;
        float sq2 = 0.f;
        for (int32_t kk = 0; kk < kTILE_K / 2; ++kk)
        {
            float2 const kf = __half22float2(k2[kk]);
            float2 const qf = __half22float2(q2[kk]);
            sk2 += kf.x * kf.x + kf.y * kf.y;
            sq2 += qf.x * qf.x + qf.y * qf.y;
        }
        sInvK[tid] = useQKL2Norm ? rsqrtf(sk2 + 1e-6f) : 1.f;
        sInvQ[tid] = (useQKL2Norm ? rsqrtf(sq2 + 1e-6f) : 1.f) * scale;
    }
    if (tid == 0)
    {
        *sMaxDepth = 0;
    }
    __syncthreads();

    // ---- Phase 2: cumlog via mask, gamma, node depth -----------------------
    if (tid < N)
    {
        float c = 0.f;
        int32_t pc = 0;
        for (int32_t w = 0; w < kMASK_WORDS; ++w)
        {
            uint32_t m = sMask[tid * kMASK_WORDS + w];
            pc += __popc(m);
            while (m)
            {
                c += sLogG[w * 32 + __ffs(m) - 1];
                m &= m - 1;
            }
        }
        sCum[tid] = c;
        sGamma[tid] = expf(c);
        // Ancestor mask includes self: depth = popcount - 1 (root = 0,
        // invalid nodes with empty masks = -1, skipped by the level loop).
        sDepth[tid] = pc - 1;
        atomicMax(sMaxDepth, pc - 1);
    }
    __syncthreads();

    // ---- Phase 2b: replay stash (written once per (batch, hv)) ------------
    {
        StashOffsets const so = stashOffsets(H, HV);
        char* nodeBase = stash + static_cast<size_t>(iN) * stashBatchStrideBytes;
        for (int32_t n = 0; n < N; ++n)
        {
            char* cell = nodeBase + static_cast<size_t>(n) * so.nodeBytes;
            if (iHv % (HV / H) == 0)
            {
                float* kDst = reinterpret_cast<float*>(cell + so.kNorm) + static_cast<size_t>(iH) * kTILE_K;
                for (int32_t kk = tid; kk < kTILE_K; kk += kNUM_THREADS)
                {
                    kDst[kk] = __half2float(sK[n * kTILE_K + kk]) * sInvK[n];
                }
            }
            if (tid == 0)
            {
                reinterpret_cast<float*>(cell + so.g)[iHv] = expf(sLogG[n]);
                reinterpret_cast<float*>(cell + so.beta)[iHv] = sBeta[n];
            }
            __half* vDst = reinterpret_cast<__half*>(cell + so.v) + static_cast<size_t>(iHv) * kTILE_K;
            for (int32_t vv = tid; vv < kTILE_K; vv += kNUM_THREADS)
            {
                vDst[vv] = v[(static_cast<int64_t>(iN) * N + n) * HV * kTILE_K + iHv * kTILE_K + vv];
            }
        }
    }

    // ---- Phase 3: M (strict) / Attn (inclusive), masked pairs only --------
    for (int32_t pair = tid; pair < N * N; pair += kNUM_THREADS)
    {
        int32_t const n = pair / N;
        int32_t const j = pair % N;
        uint32_t const bit = (sMask[n * kMASK_WORDS + j / 32] >> (j % 32)) & 1u;
        float mm = 0.f;
        float at = 0.f;
        if (bit)
        {
            float const d = expf(sCum[n] - sCum[j]);
            __half2 const* kn = reinterpret_cast<__half2 const*>(sK + n * kTILE_K);
            __half2 const* qn = reinterpret_cast<__half2 const*>(sQ + n * kTILE_K);
            __half2 const* kj = reinterpret_cast<__half2 const*>(sK + j * kTILE_K);
            float kkDot = 0.f;
            float qkDot = 0.f;
#pragma unroll 8
            for (int32_t kk = 0; kk < kTILE_K / 2; ++kk)
            {
                float2 const kjf = __half22float2(kj[kk]);
                float2 const knf = __half22float2(kn[kk]);
                float2 const qnf = __half22float2(qn[kk]);
                kkDot += knf.x * kjf.x + knf.y * kjf.y;
                qkDot += qnf.x * kjf.x + qnf.y * kjf.y;
            }
            kkDot *= sInvK[n] * sInvK[j];
            qkDot *= sInvQ[n] * sInvK[j];
            mm = (j == n) ? 0.f : sBeta[n] * d * kkDot;
            at = d * qkDot;
        }
        sM[n * kN_MAX + j] = mm;
        sAttn[n * kN_MAX + j] = at;
    }
    __syncthreads();

    // ---- Export prep block for the apply kernel ----------------------------
    {
        StashOffsets const soKq = stashOffsets(H, HV);
        char* rowTail = stash + static_cast<size_t>(iN) * stashBatchStrideBytes;
        float* blk = reinterpret_cast<float*>(rowTail + gdnTreePrepBaseBytes(soKq, HV))
            + static_cast<size_t>(iHv) * kGDN_TREE_CHUNK_PREP_WORDS;
        for (int32_t idx = tid; idx < N * kN_MAX; idx += kNUM_THREADS)
        {
            blk[kPREP_M + idx] = sM[idx];
            blk[kPREP_ATTN + idx] = sAttn[idx];
        }
        int32_t* blkI = reinterpret_cast<int32_t*>(blk + kPREP_DEPTH);
        if (tid < N)
        {
            blk[kPREP_GAMMA + tid] = sGamma[tid];
            blk[kPREP_BETA + tid] = sBeta[tid];
            blk[kPREP_INVK + tid] = sInvK[tid];
            blk[kPREP_INVQ + tid] = sInvQ[tid];
            blkI[tid] = sDepth[tid];
        }
        if (tid == 0)
        {
            blkI[kN_MAX] = *sMaxDepth;
        }

        // Raw KS/QS = [K|Q] x H0 on tensor cores, exported for all four
        // v-quarters at once (the apply kernel used to recompute its own
        // 32-column slice per split CTA). Accumulators store straight to the
        // KS/QS staging area in the intermediate-states row. Rows in
        // [N, nPad) are garbage and are never read by apply (all consumers
        // guard on n < N).
        float* ksOut = reinterpret_cast<float*>(rowTail + gdnTreeKsQsBaseBytes(soKq))
            + static_cast<size_t>(iHv) * (2 * kN_MAX * kTILE_K);
        float* qsOut = ksOut + kN_MAX * kTILE_K;
        int32_t const nPad = (N + 15) & ~15;
        int32_t const warp = tid / 32;
        constexpr int32_t kColTiles = kTILE_K / 16; // 8
        int32_t const numTiles = (nPad / 16) * kColTiles;
        for (int32_t tile = warp; tile < numTiles; tile += kNUM_THREADS / 32)
        {
            int32_t const rowTile = tile / kColTiles;
            int32_t const colTile = tile % kColTiles;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> aK;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> aQ;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accK;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accQ;
            wmma::fill_fragment(accK, 0.f);
            wmma::fill_fragment(accQ, 0.f);
            for (int32_t kt = 0; kt < kTILE_K / 16; ++kt)
            {
                wmma::load_matrix_sync(bf, sH0h + (kt * 16) * kTILE_K + colTile * 16, kTILE_K);
                wmma::load_matrix_sync(aK, sK + (rowTile * 16) * kTILE_K + kt * 16, kTILE_K);
                wmma::mma_sync(accK, aK, bf, accK);
                wmma::load_matrix_sync(aQ, sQ + (rowTile * 16) * kTILE_K + kt * 16, kTILE_K);
                wmma::mma_sync(accQ, aQ, bf, accQ);
            }
            wmma::store_matrix_sync(
                ksOut + (rowTile * 16) * kTILE_K + colTile * 16, accK, kTILE_K, wmma::mem_row_major);
            wmma::store_matrix_sync(
                qsOut + (rowTile * 16) * kTILE_K + colTile * 16, accQ, kTILE_K, wmma::mem_row_major);
        }
    }
}

// Apply kernel: one CTA per (batch, hv-head, v-quarter). Consumes the prep
// block (M/Attn/vectors/depth) and runs Phases 4a-c. See
// gdnTreeChunkKernels.h and the validated fp64 reference
// (workspace/gdn-tree-poc/reference.py).
//
// Tensor-core path: the KS/QS GEMM lives in the prep kernel (computed once
// per (batch, hv) instead of per v-quarter); apply consumes its 32-column
// slice, keeps the fp32 Phase-4b substitution, and runs 4c (Attn x R) on
// tensor cores (wmma m16n16k16, fp16 in / fp32 accumulate). Attn and R stage
// through fp16; padding rows/columns beyond N are explicitly zeroed where
// they enter the wmma K-dimension. Not bit-exact vs a scalar implementation
// (TC accumulation order + fp16 staging); gated on output-text equality.
__global__ void __launch_bounds__(kNUM_THREADS)
    treeVerifyChunkApplyKernel(__half const* __restrict__ v, uint32_t const* __restrict__ masks, __half* __restrict__ o,
        char const* __restrict__ stash, size_t stashBatchStrideBytes, int32_t numNodes, int32_t H, int32_t HV)
{
    using namespace nvcuda;

    int32_t const split = blockIdx.x % kV_SPLIT;
    int32_t const stateIdx = blockIdx.x / kV_SPLIT;
    int32_t const iN = stateIdx / HV;
    int32_t const iHv = stateIdx % HV;
    int32_t const tid = threadIdx.x;
    int32_t const N = numNodes;
    int32_t const vBase = split * kVCOLS;
    int32_t const nPad = (N + 15) & ~15; // wmma tile-padded node count

    extern __shared__ char smemRaw[];
    float* sM = reinterpret_cast<float*>(smemRaw);                    // [N][N] fp32 (4b)
    __half* sAttnH = reinterpret_cast<__half*>(sM + kN_MAX * kN_MAX); // [N][N] fp16, zero-padded (4c A)
    float* sB = reinterpret_cast<float*>(sAttnH + kN_MAX * kN_MAX);   // [N][VCOLS] KS -> B -> R
    float* sQS = sB + kN_MAX * kVCOLS;                                // [N][VCOLS]
    __half* sRh = reinterpret_cast<__half*>(sQS + kN_MAX * kVCOLS);   // [N][VCOLS] fp16, zero-padded (4c B)
    float* sOt = reinterpret_cast<float*>(sRh + kN_MAX * kVCOLS);     // [N][VCOLS] 4c acc staging
    float* sGamma = sOt + kN_MAX * kVCOLS;
    float* sBeta = sGamma + kN_MAX;
    float* sInvK = sBeta + kN_MAX;
    float* sInvQ = sInvK + kN_MAX;
    uint32_t* sMask = reinterpret_cast<uint32_t*>(sInvQ + kN_MAX);
    int32_t* sDepth = reinterpret_cast<int32_t*>(sMask + kN_MAX * kMASK_WORDS); // [N] tree depth per node
    int32_t* sMaxDepth = sDepth + kN_MAX;                                       // single int

    // ---- Load masks and the prep block (incl. this split's KS/QS slice) ---
    if (tid < N * kMASK_WORDS)
    {
        sMask[tid] = masks[(static_cast<int64_t>(iN) * N) * kMASK_WORDS + tid];
    }
    {
        StashOffsets const soKq = stashOffsets(H, HV);
        char const* rowTail = stash + static_cast<size_t>(iN) * stashBatchStrideBytes;
        float const* blk = reinterpret_cast<float const*>(rowTail + gdnTreePrepBaseBytes(soKq, HV))
            + static_cast<size_t>(iHv) * kGDN_TREE_CHUNK_PREP_WORDS;
        for (int32_t idx = tid; idx < N * kN_MAX; idx += kNUM_THREADS)
        {
            sM[idx] = blk[kPREP_M + idx];
            // Attn as fp16 for the 4c GEMM; zero the j >= N garbage the prep
            // export never wrote (it would enter the wmma K-dimension).
            int32_t const j = idx % kN_MAX;
            sAttnH[idx] = (j < N) ? __float2half(blk[kPREP_ATTN + idx]) : __half(0);
        }
        float const* ksIn = reinterpret_cast<float const*>(rowTail + gdnTreeKsQsBaseBytes(soKq))
            + static_cast<size_t>(iHv) * (2 * kN_MAX * kTILE_K);
        float const* qsIn = ksIn + kN_MAX * kTILE_K;
        for (int32_t idx = tid; idx < N * kVCOLS; idx += kNUM_THREADS)
        {
            int32_t const n = idx / kVCOLS;
            int32_t const vv = idx % kVCOLS;
            sB[idx] = ksIn[n * kTILE_K + vBase + vv];
            sQS[idx] = qsIn[n * kTILE_K + vBase + vv];
        }
        // Zero the wmma K-padding rows of R once.
        for (int32_t idx = N * kVCOLS + tid; idx < nPad * kVCOLS; idx += kNUM_THREADS)
        {
            sRh[idx] = __half(0);
        }
        int32_t const* blkI = reinterpret_cast<int32_t const*>(blk + kPREP_DEPTH);
        if (tid < N)
        {
            sGamma[tid] = blk[kPREP_GAMMA + tid];
            sBeta[tid] = blk[kPREP_BETA + tid];
            sInvK[tid] = blk[kPREP_INVK + tid];
            sInvQ[tid] = blk[kPREP_INVQ + tid];
            sDepth[tid] = blkI[tid];
        }
        if (tid == 0)
        {
            *sMaxDepth = blkI[kN_MAX];
        }
    }
    __syncthreads();

    // Elementwise: apply the norm scales, build B from raw KS (same operation
    // order as the scalar version: t = invK*ks; B = beta*(vraw - gamma*t)).
    // Also seed R-as-fp16: nodes at depth <= 0 are never touched by 4b, so
    // their staged value is final; substituted nodes overwrite theirs in 4b.
    for (int32_t idx = tid; idx < N * kVCOLS; idx += kNUM_THREADS)
    {
        int32_t const n = idx / kVCOLS;
        int32_t const vv = idx % kVCOLS;
        float const ks = sInvK[n] * sB[idx];
        float const vraw
            = __half2float(v[(static_cast<int64_t>(iN) * N + n) * HV * kTILE_K + iHv * kTILE_K + vBase + vv]);
        float const bVal = sBeta[n] * (vraw - sGamma[n] * ks);
        sB[idx] = bVal;
        sRh[idx] = __float2half(bVal);
        sQS[idx] *= sInvQ[n];
    }
    __syncthreads();

    // ---- Phase 4b: depth-level-scheduled forward substitution (w1) --------
    // R[n] = B[n] - sum_j M[n][j]*R[j]. M[n][j] is nonzero only for strict
    // ancestors j (depth[j] < depth[n]), so nodes at the same depth are
    // independent: process one depth level per round with the whole block
    // (was: 32 threads x N serial steps). In-place sB updates are safe within
    // a round because same-depth entries are multiplied by an exact 0 in M.
    // The j-ascending pairwise accumulation matches the serial version for
    // bit-identical results.
    {
        int32_t const maxDepth = *sMaxDepth;
        for (int32_t d = 1; d <= maxDepth; ++d)
        {
            for (int32_t idx = tid; idx < N * kVCOLS; idx += kNUM_THREADS)
            {
                int32_t const n = idx / kVCOLS;
                if (sDepth[n] != d)
                {
                    continue;
                }
                int32_t const col = idx % kVCOLS;
                float const* Mrow = sM + n * kN_MAX;
                float a0 = 0.f;
                float a1 = 0.f;
                int32_t j = 0;
                for (; j + 1 < n; j += 2)
                {
                    a0 += Mrow[j] * sB[j * kVCOLS + col];
                    a1 += Mrow[j + 1] * sB[(j + 1) * kVCOLS + col];
                }
                if (j < n)
                {
                    a0 += Mrow[j] * sB[j * kVCOLS + col];
                }
                float const rVal = sB[n * kVCOLS + col] - (a0 + a1);
                sB[n * kVCOLS + col] = rVal;                // in place: sB becomes R
                sRh[n * kVCOLS + col] = __float2half(rVal); // fp16 copy for the 4c GEMM
            }
            __syncthreads();
        }
    }
    __syncthreads();

    // ---- Phase 4c: O' = Attn @ R on tensor cores ---------------------------
    {
        int32_t const warp = tid / 32;
        constexpr int32_t kColTiles = kVCOLS / 16; // 2
        int32_t const numTiles = (nPad / 16) * kColTiles;
        if (warp < numTiles)
        {
            int32_t const rowTile = warp / kColTiles;
            int32_t const colTile = warp % kColTiles;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
            for (int32_t kt = 0; kt < nPad / 16; ++kt)
            {
                wmma::load_matrix_sync(a, sAttnH + (rowTile * 16) * kN_MAX + kt * 16, kN_MAX);
                wmma::load_matrix_sync(b, sRh + (kt * 16) * kVCOLS + colTile * 16, kVCOLS);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sOt + (rowTile * 16) * kVCOLS + colTile * 16, acc, kVCOLS, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Final store: O = valid ? gamma*QS + O' : 0.
    for (int32_t idx = tid; idx < N * kVCOLS; idx += kNUM_THREADS)
    {
        int32_t const n = idx / kVCOLS;
        int32_t const vv = idx % kVCOLS;
        bool const valid = ((sMask[n * kMASK_WORDS + n / 32] >> (n % 32)) & 1u) != 0u;
        float const val = valid ? (sGamma[n] * sQS[idx] + sOt[idx]) : 0.f;
        o[(static_cast<int64_t>(iN) * N + n) * HV * kTILE_K + iHv * kTILE_K + vBase + vv] = __float2half(val);
    }
}

// Batched replay commit: one launch covers all recurrent layers.
// Grid = numLayers x batch x hv (z-major on layer via blockIdx.y not needed;
// flat decode). Per (layer, batch, hv-head) the state tile is SMEM-resident
// across the accepted path; consumes the stash written by verify.
//
// Key optimizations:
//   1. Phase 1 pre-loads v from stash into sV[L][K] (half), eliminating per-step
//      global stash reads from the inner loop — inner loop is now pure SMEM.
//   2. sH double-buffered (sH[2][K*TILE_V]) with CP-ASYNC:
//      each V-tile's h-load overlaps with the previous tile's L-step compute.
//   3. No block barriers inside the L-step loop (was 2 per step): each warp
//      exclusively owns 4 sH columns and the k·H reduction is intra-warp, so
//      the sHK SMEM broadcast and both per-step __syncthreads were removable.
//      Barriers per V-tile drop from 2L+2 to 2.
//   4. 16-byte cp.async.cg tile loads (4x fewer async ops) and float4 tile
//      writeback.
__global__ void __launch_bounds__(kNUM_THREADS) treeReplayCommitBatchedKernel(MtpLayerInfo const* __restrict__ infos,
    size_t stashBatchStrideBytes, int32_t const* __restrict__ acceptedIndices,
    int32_t const* __restrict__ acceptLengths, int32_t maxAcceptLen, int32_t numNodes, int32_t H, int32_t HV)
{
    constexpr int32_t kTILE_V{32};
    constexpr int32_t kNUM_V_TILES{kTILE_K / kTILE_V}; // 4
    constexpr int32_t kNUM_STAGES{2};

    int32_t const layer = blockIdx.y;
    int32_t const stateIdx = blockIdx.x;
    int32_t const iN = stateIdx / HV;
    int32_t const iHv = stateIdx % HV;
    int32_t const iH = iHv / (HV / H);
    int32_t const tid = threadIdx.x;
    int32_t const L = min(acceptLengths[iN], min(maxAcceptLen, kGDN_TREE_CHUNK_MAX_ACCEPT));

    float* h0 = static_cast<float*>(infos[layer].recurrentDst);
    char const* stash = static_cast<char const*>(infos[layer].recurrentSrc);

    // Double-buffered h-tile (32KB), k/g/beta from stash (8KB), v pre-cache (4KB).
    __shared__ float sH[kNUM_STAGES][kTILE_K * kTILE_V];          // 2×16KB = 32KB
    __shared__ float sKvec[kGDN_TREE_CHUNK_MAX_ACCEPT * kTILE_K]; // 8KB
    __shared__ float sG[kGDN_TREE_CHUNK_MAX_ACCEPT];
    __shared__ float sBetaS[kGDN_TREE_CHUNK_MAX_ACCEPT];
    __shared__ __half sV[kGDN_TREE_CHUNK_MAX_ACCEPT * kTILE_K]; // 4KB — v pre-cache

    StashOffsets const so = stashOffsets(H, HV);
    char const* nodeBase = stash + static_cast<size_t>(iN) * stashBatchStrideBytes;

    // ---- Phase 1: pre-load k/g/beta/v for all L accepted steps from stash --
    for (int32_t i = 0; i < L; ++i)
    {
        int32_t const node = acceptedIndices[static_cast<int64_t>(iN) * maxAcceptLen + i];
        char const* cell = nodeBase + static_cast<size_t>(node) * so.nodeBytes;

        float const* kSrc = reinterpret_cast<float const*>(cell + so.kNorm) + static_cast<size_t>(iH) * kTILE_K;
        for (int32_t kk = tid; kk < kTILE_K; kk += kNUM_THREADS)
            sKvec[i * kTILE_K + kk] = kSrc[kk];

        if (tid == 0)
        {
            sG[i] = reinterpret_cast<float const*>(cell + so.g)[iHv];
            sBetaS[i] = reinterpret_cast<float const*>(cell + so.beta)[iHv];
        }

        // Pre-load all K v-values so the inner loop reads from SMEM, not global.
        __half const* vSrc = reinterpret_cast<__half const*>(cell + so.v) + static_cast<size_t>(iHv) * kTILE_K;
        for (int32_t kk = tid; kk < kTILE_K; kk += kNUM_THREADS)
            sV[i * kTILE_K + kk] = vSrc[kk];
    }

    int32_t const warp = tid / 32;
    int32_t const lane = tid % 32;
    int32_t const kLocal = lane / 4;
    int32_t const vLocal = lane % 4;
    int32_t const vIdx = warp * 4 + vLocal;

    // Macro: issue cp.async load of h0 V-tile (vt_) into sH[stage_].
    // 16-byte copies (cp.async.cg) — each of the kNUM_THREADS threads
    // issues kTILE_K*kTILE_V/4/kNUM_THREADS=4 float4 async copies. Both the
    // global row (kTILE_K-float stride) and the SMEM row (kTILE_V floats) are
    // 16B-aligned at every 4-column boundary.
#define REPLAY_ISSUE_LOAD(stage_, vt_)                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        constexpr int32_t _kV4 = kTILE_V / 4;                                                                          \
        for (int32_t _i4 = tid; _i4 < kTILE_K * _kV4; _i4 += kNUM_THREADS)                                             \
        {                                                                                                              \
            int32_t const _kk = _i4 / _kV4;                                                                            \
            int32_t const _v4 = (_i4 % _kV4) * 4;                                                                      \
            float const* _src                                                                                          \
                = h0 + ((static_cast<int64_t>(iN) * HV + iHv) * kTILE_K + _kk) * kTILE_K + (vt_) * kTILE_V + _v4;      \
            uint32_t _dst = __cvta_generic_to_shared(&sH[(stage_)][_kk * kTILE_V + _v4]);                              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(_dst), "l"(_src));                          \
        }                                                                                                              \
        asm volatile("cp.async.commit_group;");                                                                        \
    } while (0)

    __syncthreads(); // Phase 1 writes visible to all threads

    // Prime the pipeline: issue load for vt=0 into stage 0.
    REPLAY_ISSUE_LOAD(0, 0);

    // ---- V-tile loop: double-buffered CP-ASYNC (compute || next-tile load) --
    for (int32_t vt = 0; vt < kNUM_V_TILES; ++vt)
    {
        int32_t const stage = vt % kNUM_STAGES;
        int32_t const nextVt = vt + 1;

        // Wait for current stage's async load to complete, then sync.
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // Issue next tile's prefetch into the opposite stage while we compute.
        if (nextVt < kNUM_V_TILES)
            REPLAY_ISSUE_LOAD(nextVt % kNUM_STAGES, nextVt);

        // ---- L accepted steps, entirely in SMEM (overlaps next-tile load) ---
        // No block barriers inside this loop. Each warp exclusively owns
        // sH columns 4*warp .. 4*warp+3 (vIdx = warp*4 + vLocal), the h
        // read-modify-write per (kk, vIdx) is lane-private, and the sumHk
        // reduction over kLocal is intra-warp via xor-shuffle — nothing
        // crosses warp boundaries until the writeback below.
        for (int32_t i = 0; i < L; ++i)
        {
            float const g = sG[i];
            float const beta = sBetaS[i];

            float sumHk = 0.f;
            for (int32_t it = 0; it < kTILE_K / 8; ++it)
            {
                int32_t const kk = it * 8 + kLocal;
                float const hg = sH[stage][kk * kTILE_V + vIdx] * g;
                sH[stage][kk * kTILE_V + vIdx] = hg;
                sumHk += hg * sKvec[i * kTILE_K + kk];
            }
#pragma unroll
            for (int32_t off = 4; off < 32; off <<= 1)
                sumHk += __shfl_xor_sync(0xffffffffu, sumHk, off);
            // Every lane now holds the full k·H sum for its vIdx column.

            // v from SMEM pre-cache: sV[i][vt * kTILE_V + vIdx]
            float const vraw = __half2float(sV[i * kTILE_K + vt * kTILE_V + vIdx]);
            float const vNew = (vraw - sumHk) * beta;

            for (int32_t it = 0; it < kTILE_K / 8; ++it)
            {
                int32_t const kk = it * 8 + kLocal;
                sH[stage][kk * kTILE_V + vIdx] += sKvec[i * kTILE_K + kk] * vNew;
            }
        }

        // All warps must finish their columns before the cross-warp writeback.
        __syncthreads();

        // Write updated h-tile back to global h0 (float4).
        for (int32_t i4 = tid; i4 < kTILE_K * (kTILE_V / 4); i4 += kNUM_THREADS)
        {
            int32_t const kk = i4 / (kTILE_V / 4);
            int32_t const v4 = (i4 % (kTILE_V / 4)) * 4;
            *reinterpret_cast<float4*>(
                &h0[((static_cast<int64_t>(iN) * HV + iHv) * kTILE_K + kk) * kTILE_K + vt * kTILE_V + v4])
                = *reinterpret_cast<float4 const*>(&sH[stage][kk * kTILE_V + v4]);
        }
        // No barrier here: the next iteration's post-cp.async __syncthreads
        // orders this writeback before any thread's prefetch can overwrite
        // this stage buffer (tile vt+2 reuses stage vt%2).
    }

#undef REPLAY_ISSUE_LOAD
}

} // namespace

bool gdnTreeChunkVerifyEnabled(int32_t treeSize)
{
    // Retain the old environment variable only as an explicit rejection. An
    // engine with the compact output contract has no checkpoint-shaped buffer
    // to fall back to.
    static bool const checkpointRequested = [] {
        char const* env = std::getenv("EDGELLM_GDN_TREE_IMPL");
        return env != nullptr && std::strcmp(env, "checkpoint") == 0;
    }();
    return !checkpointRequested && treeSize > 0 && treeSize <= kGDN_TREE_CHUNK_MAX_NODES;
}

// Launch error tracer: logs which launch failed and returns the error so the
// caller can propagate it. Call immediately after a launch — cudaGetLastError()
// also clears the sticky error, so callers must detect failure via the return
// value, NOT a later cudaGetLastError().
inline cudaError_t gdnTreeTraceLaunch(char const* tag)
{
    cudaError_t const e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        LOG_ERROR("gdnTreeChunk: launch failed at %s: %s", tag, cudaGetErrorString(e));
    }
    return e;
}

cudaError_t gdnTreeBuildAncestorMasks(int32_t const* parentIds, uint32_t* masksOut, int32_t batch, int32_t numNodes,
    int32_t maxDepth, cudaStream_t stream)
{
    buildAncestorMasksKernel<<<batch, kN_MAX, 0, stream>>>(parentIds, masksOut, numNodes, maxDepth);
    return gdnTreeTraceLaunch("buildAncestorMasks");
}

cudaError_t gdnLinearBuildCausalMasks(uint32_t* masksOut, int32_t batch, int32_t numNodes, cudaStream_t stream)
{
    buildAncestorMasksKernel<<<batch, kN_MAX, 0, stream>>>(nullptr, masksOut, numNodes, numNodes);
    return gdnTreeTraceLaunch("buildLinearCausalMasks");
}

cudaError_t gdnTreeVerifyChunk(float const* h0, __half const* q, __half const* k, __half const* v, __half const* a,
    __half const* b, float const* A_log, __half const* dt_bias, uint32_t const* masks, __half* o, void* stash,
    size_t stashBatchStrideBytes, int32_t batch, int32_t numNodes, int32_t h, int32_t hv, float scale, bool useQKL2Norm,
    cudaStream_t stream)
{
    size_t const smemPrep = 2 * kN_MAX * kTILE_K * sizeof(__half) // sK, sQ
        + kTILE_K * kTILE_K * sizeof(__half)                      // sH0h (fp16, KS/QS GEMM)
        + 2 * kN_MAX * kN_MAX * sizeof(float)                     // sM, sAttn
        + 6 * kN_MAX * sizeof(float)                              // scalars
        + kN_MAX * kMASK_WORDS * sizeof(uint32_t)                 // sMask
        + (kN_MAX + 1) * sizeof(int32_t);                         // sDepth + sMaxDepth
    size_t const smemApply = kN_MAX * kN_MAX * sizeof(float)      // sM (fp32, 4b)
        + kN_MAX * kN_MAX * sizeof(__half)                        // sAttnH (fp16, 4c)
        + 2 * kN_MAX * kVCOLS * sizeof(float)                     // sB, sQS
        + kN_MAX * kVCOLS * sizeof(__half)                        // sRh (fp16, 4c)
        + kN_MAX * kVCOLS * sizeof(float)                         // sOt (4c staging)
        + 4 * kN_MAX * sizeof(float)                              // gamma/beta/invK/invQ
        + kN_MAX * kMASK_WORDS * sizeof(uint32_t)                 // sMask
        + (kN_MAX + 1) * sizeof(int32_t);                         // sDepth + sMaxDepth
    cudaError_t attrErr
        = cudaFuncSetAttribute(treeVerifyChunkPrepKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemPrep);
    if (attrErr != cudaSuccess)
    {
        LOG_ERROR(
            "gdnTreeChunk: cudaFuncSetAttribute(verifyPrep, %zu) failed: %s", smemPrep, cudaGetErrorString(attrErr));
        return attrErr;
    }
    attrErr = cudaFuncSetAttribute(treeVerifyChunkApplyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemApply);
    if (attrErr != cudaSuccess)
    {
        LOG_ERROR(
            "gdnTreeChunk: cudaFuncSetAttribute(verifyApply, %zu) failed: %s", smemApply, cudaGetErrorString(attrErr));
        return attrErr;
    }
    // KS/QS + prep staging must fit in the unused tail of each
    // intermediate-states batch row (past the last stash cell). Refuse to
    // launch on overflow — launching anyway would silently corrupt the next
    // batch row (or whatever follows the buffer).
    {
        StashOffsets const so = stashOffsets(h, hv);
        size_t const scratchEnd = gdnTreeChunkBufferBytes(h, hv);
        if (scratchEnd > stashBatchStrideBytes)
        {
            LOG_ERROR(
                "gdnTreeChunk: verify scratch (%zu B) exceeds the intermediate row stride (%zu B); "
                "refusing to launch",
                scratchEnd, stashBatchStrideBytes);
            return cudaErrorInvalidValue;
        }
    }
    treeVerifyChunkPrepKernel<<<batch * hv, kNUM_THREADS, smemPrep, stream>>>(q, k, v, a, b, A_log, dt_bias, masks,
        static_cast<char*>(stash), stashBatchStrideBytes, h0, numNodes, h, hv, scale, useQKL2Norm ? 1 : 0);
    if (cudaError_t const e = gdnTreeTraceLaunch("treeVerifyChunkPrep"); e != cudaSuccess)
    {
        return e;
    }
    treeVerifyChunkApplyKernel<<<batch * hv * kV_SPLIT, kNUM_THREADS, smemApply, stream>>>(
        v, masks, o, static_cast<char const*>(stash), stashBatchStrideBytes, numNodes, h, hv);
    return gdnTreeTraceLaunch("treeVerifyChunkApply");
}

cudaError_t gdnTreeReplayCommitBatched(MtpLayerInfo const* deviceLayerInfos, int32_t numLayers,
    size_t stashBatchStrideBytes, int32_t const* acceptedIndices, int32_t const* acceptLengths, int32_t batch,
    int32_t maxAcceptLen, int32_t numNodes, int32_t h, int32_t hv, cudaStream_t stream)
{
    static bool const kTimeCommit = (std::getenv("EDGELLM_GDN_TIME_COMMIT") != nullptr);
    static int64_t sCallCount = 0;

    cudaEvent_t t0{}, t1{};
    if (kTimeCommit)
    {
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0, stream);
    }

    dim3 const grid(batch * hv, numLayers);
    treeReplayCommitBatchedKernel<<<grid, kNUM_THREADS, 0, stream>>>(
        deviceLayerInfos, stashBatchStrideBytes, acceptedIndices, acceptLengths, maxAcceptLen, numNodes, h, hv);
    cudaError_t const launchErr = gdnTreeTraceLaunch("treeReplayCommitBatched");

    if (kTimeCommit)
    {
        cudaEventRecord(t1, stream);
        cudaEventSynchronize(t1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, t0, t1);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        ++sCallCount;
        LOG_INFO("gdnTreeChunk commit-time: call=%ld batch=%d hv=%d layers=%d nodes=%d  %.1f us", (long) sCallCount,
            batch, hv, numLayers, numNodes, ms * 1000.f);
    }
    return launchErr;
}

} // namespace kernel
} // namespace trt_edgellm
