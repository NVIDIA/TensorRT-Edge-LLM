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

#include "common/cudaUtils.h"
#include "kernels/speculative/speculativeKernels.h"
#include "references.h"
#include "testUtils.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <limits>

using namespace trt_edgellm;

TEST(SpecGreedyAcceptKernels, ChainPreservesMismatchCountsAndIndependentOutputStride)
{
    // Given root-inclusive chains with unequal valid counts and poisoned padding,
    // when walking every mismatch and output width, then only the valid prefix and bonus are emitted.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    for (int32_t capacity : {0, 1, 3, 7})
    {
        for (int32_t mismatch = 0; mismatch <= 4; ++mismatch)
        {
            rt::Tensor target({2, 5}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
            rt::Tensor tokens({2, 5}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
            rt::Tensor valid({2}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
            rt::Tensor ids({2, std::max(1, capacity)}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
            rt::Tensor rows({2, std::max(1, capacity)}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
            rt::Tensor counts({2}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
            ELLM_CHECK(ids.reshape({2, capacity}), "Test output reshape failed");
            ELLM_CHECK(rows.reshape({2, capacity}), "Test predictor reshape failed");
            std::vector<int32_t> predictions{1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
            if (mismatch < 4)
                predictions[mismatch] = 99;
            copyHostToDevice<int32_t>(target, predictions);
            copyHostToDevice<int32_t>(tokens, {0, 1, 2, 3, 4, 0, 1, 2, 3, 4});
            copyHostToDevice<int32_t>(valid, {5, 3});
            kernel::acceptSpecGreedyChain(target, tokens, &valid, nullptr, ids, counts, &rows, 2, 5, capacity, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<int32_t> expectedCounts{std::min(capacity, mismatch + 1), std::min(capacity, 3)};
            std::vector<int32_t> expectedIds(2 * capacity, 0), expectedRows(2 * capacity, -1);
            for (int32_t batch = 0; batch < 2; ++batch)
            {
                for (int32_t i = 0; i < expectedCounts[batch]; ++i)
                {
                    expectedIds[batch * capacity + i] = predictions[batch * 5 + i];
                    expectedRows[batch * capacity + i] = i;
                }
            }
            auto actual = copyDeviceToHost<int32_t>(counts);
            auto expected = expectedCounts;
            if (capacity > 0)
            {
                auto const actualIds = copyDeviceToHost<int32_t>(ids);
                auto const actualRows = copyDeviceToHost<int32_t>(rows);
                actual.insert(actual.end(), actualIds.begin(), actualIds.end());
                actual.insert(actual.end(), actualRows.begin(), actualRows.end());
                expected.insert(expected.end(), expectedIds.begin(), expectedIds.end());
                expected.insert(expected.end(), expectedRows.begin(), expectedRows.end());
            }
            EXPECT_EQ(actual, expected) << "capacity=" << capacity << " mismatch=" << mismatch;
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SpecGreedyAcceptKernels, TreeUsesImmediateParentsAndMinimumMatchingChild)
{
    // Given non-BFS trees with duplicate siblings and a shorter second request,
    // when walking target predictions, then immediate parents, earliest child and valid counts determine the path.
    rt::Tensor target({2, 6}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor tokens({2, 6}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor parents({2, 6}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor valid({2}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor ids({2, 5}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor rows({2, 5}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor counts({2}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    copyHostToDevice<int32_t>(target, {7, 8, 9, 99, 99, 10, 7, 8, 9, 99, 99, 10});
    copyHostToDevice<int32_t>(tokens, {0, 7, 8, 7, 9, 9, 0, 7, 8, 7, 9, 9});
    copyHostToDevice<int32_t>(parents, {-1, 0, 1, 0, 3, 2, -1, 0, 1, 0, 3, 2});
    copyHostToDevice<int32_t>(valid, {6, 5});
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    kernel::acceptSpecGreedyTree(target, tokens, parents, &valid, nullptr, ids, counts, &rows, 2, 6, 5, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto actual = copyDeviceToHost<int32_t>(counts);
    auto const actualIds = copyDeviceToHost<int32_t>(ids);
    auto const actualRows = copyDeviceToHost<int32_t>(rows);
    actual.insert(actual.end(), actualIds.begin(), actualIds.end());
    actual.insert(actual.end(), actualRows.begin(), actualRows.end());
    EXPECT_EQ(actual, (std::vector<int32_t>{4, 3, 7, 8, 9, 10, 0, 7, 8, 9, 0, 0, 0, 1, 2, 5, -1, 0, 1, 2, -1, -1}));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SpecGreedyAcceptKernels, BorrowedSelectorPreservesMappedTiesAndExceptionalLogits)
{
    // Given mapped vocabulary rows with finite ties, NaNs, infinities and masked values,
    // when selecting target IDs separately, then the old selector results and workspace ownership are preserved.
    constexpr int32_t batch = 7;
    constexpr int32_t vocab = 128;
    rt::Tensor logits({batch, vocab}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT);
    rt::Tensor mapping({vocab}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    size_t const bytes = kernel::getSpecTargetSelectionWorkspaceSize(batch, 1);
    rt::Tensor workspace({static_cast<int64_t>(bytes)}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8);
    std::vector<float> values(batch * vocab, -10.0F);
    values[3] = 10.0F;
    for (int32_t i = 0; i < vocab; ++i)
    {
        values[vocab + i] = 0.0F;
        values[2 * vocab + i] = std::numeric_limits<float>::quiet_NaN();
        values[3 * vocab + i] = -std::numeric_limits<float>::infinity();
        values[5 * vocab + i] = -std::numeric_limits<float>::max();
    }
    values[4 * vocab + 9] = std::numeric_limits<float>::infinity();
    values[4 * vocab + 33] = std::numeric_limits<float>::infinity();
    values[6 * vocab] = std::numeric_limits<float>::quiet_NaN();
    values[6 * vocab + 64] = 20.0F;
    std::vector<int32_t> map(vocab);
    for (int32_t i = 0; i < vocab; ++i)
        map[i] = i + 1000;
    copyHostToDevice<float>(logits, values);
    copyHostToDevice<int32_t>(mapping, map);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto selected = kernel::selectSpecTargetTokens(logits, std::cref(mapping), workspace, batch, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(copyDeviceToHost<int32_t>(selected), (std::vector<int32_t>{1003, 1127, 1000, 1000, 1033, 1000, 1064}));
    EXPECT_EQ(selected.rawPointer(), workspace.rawPointer());
    CUDA_CHECK(cudaStreamDestroy(stream));
}

namespace
{
//! Run greedy acceptance with optional forced flags and valid counts; return counts, ids and rows concatenated.
std::vector<int32_t> acceptWithFlags(bool tree, int32_t requests, int32_t width, int32_t capacity,
    std::vector<int32_t> const& predictions, std::vector<int32_t> const& tokens, std::vector<int32_t> const& parents,
    std::vector<int8_t> const* flags, std::vector<int32_t> const* validCounts)
{
    rt::Tensor target({requests, width}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor tokenIds({requests, width}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor parentIds({requests, width}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor forced({requests, width}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8);
    rt::Tensor valid({requests}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor ids({requests, capacity}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor rows({requests, capacity}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor counts({requests}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    copyHostToDevice<int32_t>(target, predictions);
    copyHostToDevice<int32_t>(tokenIds, tokens);
    if (tree)
    {
        copyHostToDevice<int32_t>(parentIds, parents);
    }
    if (flags != nullptr)
    {
        copyHostToDevice<int8_t>(forced, *flags);
    }
    if (validCounts != nullptr)
    {
        copyHostToDevice<int32_t>(valid, *validCounts);
    }
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (tree)
    {
        kernel::acceptSpecGreedyTree(target, tokenIds, parentIds, validCounts == nullptr ? nullptr : &valid,
            flags == nullptr ? nullptr : &forced, ids, counts, &rows, requests, width, capacity, stream);
    }
    else
    {
        kernel::acceptSpecGreedyChain(target, tokenIds, validCounts == nullptr ? nullptr : &valid,
            flags == nullptr ? nullptr : &forced, ids, counts, &rows, requests, width, capacity, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    auto result = copyDeviceToHost<int32_t>(counts);
    auto const actualIds = copyDeviceToHost<int32_t>(ids);
    auto const actualRows = copyDeviceToHost<int32_t>(rows);
    result.insert(result.end(), actualIds.begin(), actualIds.end());
    result.insert(result.end(), actualRows.begin(), actualRows.end());
    return result;
}

// Root 0 with a draft decoy (node 1, token 5) and a forced chain F1 -> F2 -> F3 (nodes 2..4) ending in a draft
// child d4 (node 5, token 30); the root prediction matches the decoy.
std::vector<int32_t> const kFORCED_CHAIN_TOKENS{0, 5, 20, 21, 22, 30};
std::vector<int32_t> const kFORCED_CHAIN_PARENTS{-1, 0, 0, 2, 3, 4};
std::vector<int8_t> const kFORCED_CHAIN_FLAGS{0, 0, 1, 1, 1, 0};
std::vector<int32_t> const kFORCED_CHAIN_PREDICTIONS{5, 99, 99, 99, 30, 40};

//! The ancestors-or-self rows the reference reads, derived from parents as the host draft tree derives them.
std::vector<int8_t> maskFromParents(std::vector<int32_t> const& parents)
{
    auto const width = static_cast<int32_t>(parents.size());
    std::vector<int8_t> mask(static_cast<size_t>(width) * static_cast<size_t>(width), 0);
    for (int32_t node = 0; node < width; ++node)
    {
        if (parents[node] >= 0)
        {
            std::copy_n(mask.begin() + parents[node] * width, width, mask.begin() + node * width);
        }
        mask[static_cast<size_t>(node) * width + node] = 1;
    }
    return mask;
}

//! The kernel's accepted path for one request and the reference's, with the per-position forced flags derived from
//! the kernel's rows the way the runtime derives them: position k is forced when row k + 1 is a forced node.
struct AcceptedPath
{
    std::vector<int32_t> ids;
    std::vector<int32_t> rows;
    std::vector<int8_t> forcedFromRows;
    EagleAcceptResult ref;
};

AcceptedPath acceptBoth(std::vector<int32_t> const& predictions, std::vector<int32_t> const& tokens,
    std::vector<int32_t> const& parents, std::vector<int8_t> const& flags, int32_t capacity)
{
    constexpr int32_t kVocab{128};
    auto const width = static_cast<int32_t>(tokens.size());
    auto const out
        = acceptWithFlags(/* tree = */ true, 1, width, capacity, predictions, tokens, parents, &flags, nullptr);
    int32_t const count = out[0];
    AcceptedPath path;
    path.ids.assign(out.begin() + 1, out.begin() + 1 + count);
    path.rows.assign(out.begin() + 1 + capacity, out.begin() + 1 + capacity + count);
    for (int32_t k = 0; k < count; ++k)
    {
        path.forcedFromRows.push_back(k + 1 < count && flags[path.rows[k + 1]] != 0 ? 1 : 0);
    }
    std::vector<float> logits(static_cast<size_t>(width) * kVocab, 0.0F);
    for (int32_t node = 0; node < width; ++node)
    {
        logits[static_cast<size_t>(node) * kVocab + predictions[node]] = 1.0F;
    }
    path.ref = eagleAcceptRef(logits, tokens, maskFromParents(parents), 1, width, kVocab, capacity, {}, flags, {});
    return path;
}
} // namespace

TEST(SpecGreedyAcceptKernels, TreeForcedChildOutranksMatchingSiblingAtTheRootAndAtDepthTwo)
{
    // Given a root whose children are a token-matching draft sibling at the lowest index and two forced children,
    // and a root whose matched child has both a token-matching draft child and a forced child,
    // when accepting, then at the root the lowest-indexed forced child is taken, its token is emitted and the walk
    // continues from it with a normal match, and after the matched step the forced child is taken and its token
    // emitted from the parent's row, the draft below it continues on a match, and the reference agrees position by
    // position, including which positions were forced.
    std::vector<int8_t> const flags{0, 0, 1, 1, 0, 0};
    auto const actual = acceptWithFlags(
        /* tree = */ true, 1, 6, 4, {7, 99, 11, 99, 42, 99}, {0, 7, 9, 8, 11, 5}, {-1, 0, 0, 0, 2, 3}, &flags, nullptr);
    EXPECT_EQ(actual, (std::vector<int32_t>{3, 9, 11, 42, 0, 0, 2, 4, -1}));

    std::vector<int32_t> const tokens{0, 7, 8, 20, 30};
    std::vector<int32_t> const parents{-1, 0, 1, 1, 3};
    std::vector<int8_t> const depthTwoFlags{0, 0, 0, 1, 0};
    auto const path = acceptBoth({7, 8, 99, 30, 40}, tokens, parents, depthTwoFlags, 5);
    EXPECT_EQ(path.ids, (std::vector<int32_t>{7, 20, 30, 40}));
    EXPECT_EQ(path.rows, (std::vector<int32_t>{0, 1, 3, 4}));
    EXPECT_EQ(path.forcedFromRows, (std::vector<int8_t>{0, 1, 0, 0}));
    EXPECT_EQ(path.ref.acceptedTokenIds, path.ids);
    EXPECT_EQ(path.ref.acceptedLogitsIndices, path.rows);
    EXPECT_EQ(path.ref.acceptedForced, path.forcedFromRows);
}

TEST(SpecGreedyAcceptKernels, TreeFollowsForcedChainPastDraftDepthThenMatchesTop1)
{
    // Given a forced chain deeper than the draft branch and a draft child under its tail,
    // when accepting, then every chain token is emitted in order, the tail's top-1 match continues the path and
    // the final bonus token closes it.
    auto const actual = acceptWithFlags(/* tree = */ true, 1, 6, 6, kFORCED_CHAIN_PREDICTIONS, kFORCED_CHAIN_TOKENS,
        kFORCED_CHAIN_PARENTS, &kFORCED_CHAIN_FLAGS, nullptr);
    std::vector<int32_t> const countAndIds(actual.begin(), actual.begin() + 7);
    EXPECT_EQ(countAndIds, (std::vector<int32_t>{5, 20, 21, 22, 30, 40, 0}));
}

TEST(SpecGreedyAcceptKernels, TreeForcedStepsKeepParentRowsAlongAChainAndAcrossAdjacentSegments)
{
    // Given forced steps along a chain, and a forced chain at the root whose tail has a draft child that carries its
    // own forced chain, when accepting, then each emitted token's row is the node that predicted it, never the forced
    // child itself: both segments are emitted from their parents' rows with the draft match between them, and the
    // reference agrees position by position.
    auto const actual = acceptWithFlags(/* tree = */ true, 1, 6, 6, kFORCED_CHAIN_PREDICTIONS, kFORCED_CHAIN_TOKENS,
        kFORCED_CHAIN_PARENTS, &kFORCED_CHAIN_FLAGS, nullptr);
    std::vector<int32_t> const rows(actual.begin() + 7, actual.end());
    EXPECT_EQ(rows, (std::vector<int32_t>{0, 2, 3, 4, 5, -1}));

    std::vector<int32_t> const tokens{0, 20, 21, 7, 22};
    std::vector<int32_t> const parents{-1, 0, 1, 2, 3};
    std::vector<int8_t> const flags{0, 1, 1, 0, 1};
    auto const path = acceptBoth({5, 99, 7, 99, 60}, tokens, parents, flags, 6);
    EXPECT_EQ(path.ids, (std::vector<int32_t>{20, 21, 7, 22, 60}));
    EXPECT_EQ(path.rows, (std::vector<int32_t>{0, 1, 2, 3, 4}));
    EXPECT_EQ(path.forcedFromRows, (std::vector<int8_t>{1, 1, 0, 1, 0}));
    EXPECT_EQ(path.ref.acceptedTokenIds, path.ids);
    EXPECT_EQ(path.ref.acceptedLogitsIndices, path.rows);
    EXPECT_EQ(path.ref.acceptedForced, path.forcedFromRows);
}

TEST(SpecGreedyAcceptKernels, ChainForcedSuccessorIsAcceptedWithItsToken)
{
    // Given chains whose next candidate is forced although the prediction disagrees, at the first and at a later
    // step, when accepting, then the forced candidate is emitted with its own token and ordinary matching resumes.
    std::vector<int8_t> const flags{0, 1, 0, 0, 0, 0, 0, 0, 1, 0};
    auto const actual = acceptWithFlags(/* tree = */ false, 2, 5, 5, {99, 8, 55, 99, 99, 7, 8, 99, 10, 77},
        {0, 7, 8, 9, 10, 0, 7, 8, 9, 10}, {}, &flags, nullptr);
    EXPECT_EQ(actual, (std::vector<int32_t>{3, 5, 7, 8, 55, 0, 0, 7, 8, 9, 10, 77, 0, 1, 2, -1, -1, 0, 1, 2, 3, 4}));
}

TEST(SpecGreedyAcceptKernels, ForcedFlagsBeyondValidCountAreIgnored)
{
    // Given a forced flag on a padding node at or past the request's valid count in both layouts,
    // when accepting, then the walk ends at the valid prefix as if the flag were absent.
    std::vector<int8_t> const treeFlags{0, 0, 0, 1, 0, 0};
    std::vector<int32_t> const treeValid{3};
    auto const tree = acceptWithFlags(/* tree = */ true, 1, 6, 4, {7, 99, 11, 99, 42, 99}, {0, 7, 9, 8, 11, 5},
        {-1, 0, 0, 0, 2, 3}, &treeFlags, &treeValid);
    EXPECT_EQ(tree, (std::vector<int32_t>{2, 7, 99, 0, 0, 0, 1, -1, -1}));
    std::vector<int8_t> const chainFlags{0, 0, 1, 0, 0};
    std::vector<int32_t> const chainValid{2};
    auto const chain = acceptWithFlags(
        /* tree = */ false, 1, 5, 5, {7, 99, 55, 55, 55}, {0, 7, 8, 9, 10}, {}, &chainFlags, &chainValid);
    EXPECT_EQ(chain, (std::vector<int32_t>{2, 7, 99, 0, 0, 0, 0, 1, -1, -1, -1}));
}

TEST(SpecGreedyAcceptKernels, ZeroForcedFlagsMatchPlainAcceptance)
{
    // Given forced flags that are all zero and predictions that skip the lowest-indexed child or break the chain,
    // when accepting in both layouts, then the results equal acceptance without flags.
    std::vector<int8_t> const treeFlags(6, 0);
    std::vector<int32_t> const treePredictions{7, 99, 11, 99, 42, 99};
    std::vector<int32_t> const treeTokens{0, 9, 7, 8, 11, 5};
    std::vector<int32_t> const treeParents{-1, 0, 0, 0, 2, 3};
    auto const tree
        = acceptWithFlags(/* tree = */ true, 1, 6, 4, treePredictions, treeTokens, treeParents, &treeFlags, nullptr);
    auto const treeWithoutFlags
        = acceptWithFlags(/* tree = */ true, 1, 6, 4, treePredictions, treeTokens, treeParents, nullptr, nullptr);
    EXPECT_EQ(treeWithoutFlags, (std::vector<int32_t>{3, 7, 11, 42, 0, 0, 2, 4, -1}));
    EXPECT_EQ(tree, treeWithoutFlags);
    std::vector<int8_t> const chainFlags(5, 0);
    std::vector<int32_t> const chainPredictions{7, 99, 55, 55, 55};
    std::vector<int32_t> const chainTokens{0, 7, 8, 9, 10};
    auto const chain
        = acceptWithFlags(/* tree = */ false, 1, 5, 5, chainPredictions, chainTokens, {}, &chainFlags, nullptr);
    auto const chainWithoutFlags
        = acceptWithFlags(/* tree = */ false, 1, 5, 5, chainPredictions, chainTokens, {}, nullptr, nullptr);
    EXPECT_EQ(chainWithoutFlags, (std::vector<int32_t>{2, 7, 99, 0, 0, 0, 0, 1, -1, -1, -1}));
    EXPECT_EQ(chain, chainWithoutFlags);
}

TEST(SpecGreedyAcceptKernels, TreeForcedChainsOnTwoBranchesFollowThePredictionAtTheFork)
{
    // Given two draft branches under the root, each ending in a forced child, when the root's prediction names one
    // branch or the other, then that branch's forced chain is taken and the other is left, and the reference agrees.
    std::vector<int32_t> const tokens{0, 7, 8, 20, 21};
    std::vector<int32_t> const parents{-1, 0, 0, 1, 2};
    std::vector<int8_t> const flags{0, 0, 0, 1, 1};
    auto const right = acceptBoth({8, 99, 99, 50, 50}, tokens, parents, flags, 5);
    EXPECT_EQ(right.ids, (std::vector<int32_t>{8, 21, 50}));
    EXPECT_EQ(right.rows, (std::vector<int32_t>{0, 2, 4}));
    EXPECT_EQ(right.forcedFromRows, (std::vector<int8_t>{0, 1, 0}));
    EXPECT_EQ(right.ref.acceptedTokenIds, right.ids);
    EXPECT_EQ(right.ref.acceptedLogitsIndices, right.rows);
    EXPECT_EQ(right.ref.acceptedForced, right.forcedFromRows);
    auto const left = acceptBoth({7, 99, 99, 50, 50}, tokens, parents, flags, 5);
    EXPECT_EQ(left.ids, (std::vector<int32_t>{7, 20, 50}));
    EXPECT_EQ(left.rows, (std::vector<int32_t>{0, 1, 3}));
    EXPECT_EQ(left.ref.acceptedForced, (std::vector<int8_t>{0, 1, 0}));
}
