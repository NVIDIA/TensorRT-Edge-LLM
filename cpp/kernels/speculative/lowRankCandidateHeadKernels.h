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

#pragma once

#include "common/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{

//! Dimensions of a low-rank candidate head: hidden [R, hiddenSize], downProj [rank, hiddenSize],
//! upProj [vocabSize, rank], lmHead [vocabSize, hiddenSize]; numCandidates (Kc) candidates per row.
struct LowRankCandidateHeadShape
{
    int32_t hiddenSize{0};
    int32_t rank{0};
    int32_t vocabSize{0};
    int32_t numCandidates{0};
};

//! Selection-kernel (K1) launch configuration for passes of up to maxRows rows.
struct LowRankCandidateHeadBucket
{
    int32_t maxRows{0};    //!< Largest pass row count served by this bucket (8, 16, 32 or 64).
    int32_t capacity{0};   //!< Survivor entries per row (CAP).
    int32_t grid{0};       //!< K1 CTAs (P); CTA p scans vocab slab p.
    int32_t slabRows{0};   //!< Vocab rows per slab (a multiple of 16, at most 65536).
    int32_t listHead{0};   //!< Each per-CTA candidate list stores its listHead largest keys first.
    int32_t pilotTiles{0}; //!< Sample tiles whose per-row maxima seed the row thresholds (0: no pilot).
    size_t sharedBytes{0}; //!< K1 dynamic shared memory.
};

//! Per-device configuration, fixed at construction. Results never depend on it.
struct LowRankCandidateHeadConfig
{
    static constexpr int32_t kMaxBuckets = 4;
    static constexpr int32_t kMaxK2Groups = 8;

    int32_t smVersion{0};
    int32_t numSMs{0};
    int32_t rowsPerPass{0};      //!< Rp: R > Rp runs in ceil(R / Rp) passes, each re-reading upProj once.
    int32_t tileRows{0};         //!< T: upProj rows per pipeline stage.
    int32_t stages{0};           //!< S: pipeline depth.
    int32_t projectionSplits{0}; //!< K0 k splits (depends on hiddenSize only).
    size_t k2SharedBytes{0};     //!< K2 dynamic shared memory.
    int32_t k2CtasPerWave{0};    //!< K2 CTAs that are co-resident on the device.
    //! Co-resident (1, G, 1) K2 clusters for each G (0 where G cannot be placed); used when useCluster is set.
    std::array<int32_t, kMaxK2Groups + 1> k2ClustersPerWave{};
    bool usePdl{false};      //!< K1 and K2 use programmatic dependent launch (SM90+).
    bool useCluster{false};  //!< K2 merges once per row through a (1, G, 1) cluster (SM90+).
    bool evictFirstA{false}; //!< upProj streams with an L2 evict-first policy (upProj does not fit in L2).
    int32_t numBuckets{0};
    std::array<LowRankCandidateHeadBucket, kMaxBuckets> buckets{};
};

struct LowRankCandidateHeadOverrides;

//! Fused low-rank candidate head: K0 s = fp16(hidden downProj^T); for passes above 8 rows, or every pass when upProj
//! fits in L2, a pilot bound per row from the maxima of sampled upProj tiles; K1 tensor-core low-rank scores of every
//! vocab row with a per-CTA streaming top-Kc (upProj read once per pass, scores never written); K2 merge of the per-CTA
//! lists, lmHead gather-dot, sort by (exact logit desc, id asc), then log_softmax over the Kc exact logits.
class LowRankCandidateHeadRunner
{
public:
    //! Largest pass: workspaceBytes(shape, maxRows) is the same for every maxRows >= kMaxRowsPerPass and never larger
    //! below it.
    static constexpr int32_t kMaxRowsPerPass = 64;

    //! Picks the per-device configuration (rows per pass, tile, stages, survivor capacity, grid) from prop's SM
    //! version, SM count and shared-memory limit, and sets the kernel attributes on the current device, whose
    //! occupancy queries size the grids (prop normally describes the current device). Throws for unsupported shapes
    //! or devices (SM < 80).
    LowRankCandidateHeadRunner(LowRankCandidateHeadShape shape, int32_t maxRows, cudaDeviceProp const& prop);

    size_t workspaceBytes() const noexcept;

    //! Same configuration rules (and current-device queries) as the constructor.
    static size_t workspaceBytes(LowRankCandidateHeadShape shape, int32_t maxRows, cudaDeviceProp const& prop);

    //! The smallest workspace that any configuration withinWorkspace may choose needs on this device; results do not
    //! depend on the configuration, only the speed does.
    static size_t minimumWorkspaceBytes(LowRankCandidateHeadShape shape, int32_t maxRows, cudaDeviceProp const& prop);

    //! The device's preferred runner if its workspace fits `workspaceLimit`, else the first of a fixed sequence of
    //! smaller configurations (fewer pilot witnesses, no pilot, smaller survivor buffers, fewer CTAs per pass) that
    //! fits. Throws when none does, that is below minimumWorkspaceBytes.
    static LowRankCandidateHeadRunner withinWorkspace(
        LowRankCandidateHeadShape shape, int32_t maxRows, cudaDeviceProp const& prop, size_t workspaceLimit);

    //! hidden [R, H] fp16 with 0 <= R <= maxRows; downProj [r, H], upProj [V, r], lmHead [V, H] fp16, all 16-byte
    //! aligned. Writes candidateIds [R, Kc] int32 and candidateLogProbs [R, Kc] fp32 sorted by (logit desc, id asc).
    //! workspace: workspaceBytes() of scratch; no state persists across calls. No allocation and no host sync, so
    //! the call can be captured in a CUDA graph.
    void run(rt::Tensor const& hidden, rt::Tensor const& downProj, rt::Tensor const& upProj, rt::Tensor const& lmHead,
        rt::Tensor& candidateIds, rt::Tensor& candidateLogProbs, void* workspace, cudaStream_t stream) const;

private:
    LowRankCandidateHeadRunner(LowRankCandidateHeadShape shape, int32_t maxRows, cudaDeviceProp const& prop,
        LowRankCandidateHeadOverrides const& overrides);

    LowRankCandidateHeadShape mShape{};
    int32_t mMaxRows{0};
    LowRankCandidateHeadConfig mConfig{};
    size_t mWorkspaceBytes{0};
};

} // namespace kernel
} // namespace trt_edgellm
