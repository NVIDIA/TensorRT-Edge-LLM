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

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/pagedKvTypes.h"
#include "kernels/decodeAttentionKernels/ctmDecodeAttentionRunner.h"
#include "testUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstddef>
#include <cstdint>

namespace
{

using trt_edgellm::CtmDecodeAttentionRunner;
using trt_edgellm::CtmDecodeParams;
using trt_edgellm::rt::kTOKENS_PER_PAGE;

constexpr int32_t kContractLimit = CtmDecodeAttentionRunner::kMaxTreeQuerySeqLen;
constexpr int32_t kEagleTreeTokens = 60;           //!< EAGLE3 verify budget, the widest production tree below 64
constexpr int32_t kFirstTreePastTwoMaskWords = 65; //!< the first width whose mask needs a third word
constexpr int32_t kTwiceTheContract = 2 * kContractLimit;

// Given the speculative-decoding contract allows trees up to kMaxTreeQuerySeqLen (128) tokens,
// When a tree of up to that many tokens is requested,
// Then the runner reports it as supported, so EAGLE (60), JetSpec (128) and every size between run on CTM.
TEST(CtmDecodeTreeQuerySeqLenTest, AcceptsTreesUpToTheContractLimit)
{
    EXPECT_TRUE(CtmDecodeAttentionRunner::supportsTreeQuerySeqLen(1));
    EXPECT_TRUE(CtmDecodeAttentionRunner::supportsTreeQuerySeqLen(kEagleTreeTokens));
    EXPECT_TRUE(CtmDecodeAttentionRunner::supportsTreeQuerySeqLen(kFirstTreePastTwoMaskWords));
    EXPECT_TRUE(CtmDecodeAttentionRunner::supportsTreeQuerySeqLen(kContractLimit));
}

// Given a tree wider than the speculative-decoding contract, or an empty one,
// When the runner is asked whether it can serve that tree,
// Then it declines, so the caller rejects the configuration instead of running an unvalidated shape.
TEST(CtmDecodeTreeQuerySeqLenTest, RejectsTreesPastTheContractLimit)
{
    EXPECT_FALSE(CtmDecodeAttentionRunner::supportsTreeQuerySeqLen(kContractLimit + 1));
    EXPECT_FALSE(CtmDecodeAttentionRunner::supportsTreeQuerySeqLen(kTwiceTheContract));
    EXPECT_FALSE(CtmDecodeAttentionRunner::supportsTreeQuerySeqLen(0));
}

//! Device buffers for one fp16 D = 128 sequence whose whole KV span is the tree (no prefix, every mask bit
//! set), sized for the widest tree the tests ask for.
class CtmTreeLaunchFixture
{
public:
    static constexpr int32_t kBatch = 1;
    static constexpr int32_t kQHeads = 8;
    static constexpr int32_t kKVHeads = 1;
    static constexpr int32_t kHeadSize = 128;
    static constexpr int32_t kMaxTreeTokens = kContractLimit + 1;
    static constexpr int32_t kMaskWords = (kMaxTreeTokens + 31) / 32;
    static constexpr int32_t kPages = (kMaxTreeTokens + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE;

    CtmTreeLaunchFixture()
        : mQ(static_cast<size_t>(kBatch) * kMaxTreeTokens * kQHeads * kHeadSize, __float2half(0.F))
        , mOut(mQ.size(), __float2half(0.F))
        , mPool(static_cast<size_t>(2 * kPages) * kTOKENS_PER_PAGE * kKVHeads * kHeadSize, __float2half(0.F))
        , mPageTable(makePageTable())
        , mSeqLens(1, kMaxTreeTokens)
        , mTreeMask(static_cast<size_t>(kBatch) * kMaxTreeTokens * kMaskWords, -1)
        , mWorkspace(
              CtmDecodeAttentionRunner::treeWorkspaceElems(kBatch, kMaxTreeTokens, kQHeads, kKVHeads, kHeadSize), 0.F)
        , mSemaphores(CtmDecodeAttentionRunner::treeSemaphoreElems(kBatch, kMaxTreeTokens, kQHeads, kKVHeads), 0)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
    }

    ~CtmTreeLaunchFixture()
    {
        cudaStreamDestroy(mStream);
    }

    CtmDecodeParams params(int32_t treeTokens)
    {
        mSeqLens[0] = treeTokens;
        CtmDecodeParams p{};
        p.output = thrust::raw_pointer_cast(mOut.data());
        p.q = thrust::raw_pointer_cast(mQ.data());
        p.kvPool = thrust::raw_pointer_cast(mPool.data());
        p.pageTable = thrust::raw_pointer_cast(mPageTable.data());
        p.seqLens = thrust::raw_pointer_cast(mSeqLens.data());
        p.treeMask = thrust::raw_pointer_cast(mTreeMask.data());
        p.workspace = thrust::raw_pointer_cast(mWorkspace.data());
        p.semaphores = thrust::raw_pointer_cast(mSemaphores.data());
        p.batchSize = kBatch;
        p.qSeqLen = treeTokens;
        p.maskWords = kMaskWords;
        p.numQHeads = kQHeads;
        p.numKVHeads = kKVHeads;
        p.headSize = kHeadSize;
        p.numPages = kPages;
        p.maxPagesPerSeq = kPages;
        p.attentionScale = 1.F / 11.3137F;
        p.stream = mStream;
        return p;
    }

    cudaStream_t stream() const
    {
        return mStream;
    }

private:
    static thrust::host_vector<int32_t> makePageTable()
    {
        thrust::host_vector<int32_t> table(2 * kPages);
        for (int32_t i = 0; i < kPages; ++i)
        {
            table[i] = i;
            table[kPages + i] = kPages + i; // V page ids are pre-offset by the K page count
        }
        return table;
    }

    thrust::device_vector<half> mQ;
    thrust::device_vector<half> mOut;
    thrust::device_vector<half> mPool;
    thrust::device_vector<int32_t> mPageTable;
    thrust::device_vector<int32_t> mSeqLens;
    thrust::device_vector<int32_t> mTreeMask;
    thrust::device_vector<float> mWorkspace;
    thrust::device_vector<int32_t> mSemaphores;
    cudaStream_t mStream{nullptr};
};

bool treeModuleAvailable()
{
    return hasSm100FamilyCuteDslKernels(trt_edgellm::getSMVersion())
        && CtmDecodeAttentionRunner::supportedTree(CtmTreeLaunchFixture::kQHeads, CtmTreeLaunchFixture::kKVHeads,
            CtmTreeLaunchFixture::kHeadSize, false, false, false);
}

// Given valid device buffers for a tree exactly as wide as the contract allows,
// When run() is called,
// Then the tree module launches and completes, so a decline on the same buffers is a decision, not a broken setup.
TEST(CtmDecodeTreeQuerySeqLenTest, RunLaunchesTreeAtTheContractLimit)
{
    if (!treeModuleAvailable())
    {
        GTEST_SKIP() << "CTM tree decode module not built for this configuration or device SM";
    }
    CtmTreeLaunchFixture fixture;
    EXPECT_TRUE(CtmDecodeAttentionRunner::run(fixture.params(kContractLimit)));
    EXPECT_EQ(cudaStreamSynchronize(fixture.stream()), cudaSuccess);
}

// Given the same valid buffers but a tree one token wider than the contract,
// When run() is called,
// Then it declines without launching, so the plugin fails loudly (CTM-only) or falls back (XQA) instead of
// running an unvalidated tree width.
TEST(CtmDecodeTreeQuerySeqLenTest, RunDeclinesTreePastTheContractLimit)
{
    if (!treeModuleAvailable())
    {
        GTEST_SKIP() << "CTM tree decode module not built for this configuration or device SM";
    }
    CtmTreeLaunchFixture fixture;
    EXPECT_FALSE(CtmDecodeAttentionRunner::run(fixture.params(kContractLimit + 1)));
    EXPECT_EQ(cudaStreamSynchronize(fixture.stream()), cudaSuccess);
}

} // namespace
