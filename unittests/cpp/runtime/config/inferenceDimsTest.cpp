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

#include "runtime/config/inferenceDims.h"

#include <gtest/gtest.h>
#include <vector>

using namespace trt_edgellm::rt;

namespace
{

InferenceDims makeValid()
{
    return InferenceDims{
        /*.batch=*/2,
        /*.seqLen=*/128,
        /*.kvLen=*/4096,
        /*.selectLen=*/1,
        /*.attnMaskSeqLen=*/1,
        /*.ropeBatch=*/1,
        /*.packedMaskLen=*/4,
        /*.contextMaskSelectorLen=*/0,
        /*.startIndexLen=*/2,
        /*.specVerifyPhaseLen=*/0,
        /*.skipSoftmaxScaleLen=*/0,
    };
}

std::vector<int64_t InferenceDims::*> allReferenced()
{
    return {
        &InferenceDims::batch,
        &InferenceDims::seqLen,
        &InferenceDims::kvLen,
        &InferenceDims::selectLen,
        &InferenceDims::attnMaskSeqLen,
        &InferenceDims::ropeBatch,
        &InferenceDims::packedMaskLen,
        &InferenceDims::contextMaskSelectorLen,
        &InferenceDims::startIndexLen,
        &InferenceDims::specVerifyPhaseLen,
        &InferenceDims::skipSoftmaxScaleLen,
    };
}

} // namespace

// ---------------------------------------------------------------------------
// dimName round-trip: every entry in kDimNames must map back via dimName.
// ---------------------------------------------------------------------------

TEST(InferenceDimsTest, DimNameKnownMembers)
{
    EXPECT_EQ(dimName(&InferenceDims::batch), "batch");
    EXPECT_EQ(dimName(&InferenceDims::seqLen), "seq_len");
    EXPECT_EQ(dimName(&InferenceDims::kvLen), "kv_len");
    EXPECT_EQ(dimName(&InferenceDims::selectLen), "select_len");
    EXPECT_EQ(dimName(&InferenceDims::attnMaskSeqLen), "attn_seq_len");
    EXPECT_EQ(dimName(&InferenceDims::ropeBatch), "rope_batch");
    EXPECT_EQ(dimName(&InferenceDims::packedMaskLen), "packed_mask_len");
    EXPECT_EQ(dimName(&InferenceDims::contextMaskSelectorLen), "context_mask_selector_len");
    EXPECT_EQ(dimName(&InferenceDims::startIndexLen), "start_index_len");
    EXPECT_EQ(dimName(&InferenceDims::specVerifyPhaseLen), "spec_verify_phase_len");
    EXPECT_EQ(dimName(&InferenceDims::skipSoftmaxScaleLen), "skip_softmax_scale_len");
}

TEST(InferenceDimsTest, DimNameUnknownReturnsEmpty)
{
    int64_t InferenceDims::* unknown = nullptr;
    EXPECT_TRUE(dimName(unknown).empty());
}

// ---------------------------------------------------------------------------
// toString: owning, every field present.
// ---------------------------------------------------------------------------

TEST(InferenceDimsTest, ToStringContainsAllFields)
{
    InferenceDims const d = makeValid();
    std::string const s = toString(d);
    EXPECT_NE(s.find("batch=2"), std::string::npos) << s;
    EXPECT_NE(s.find("seq_len=128"), std::string::npos) << s;
    EXPECT_NE(s.find("kv_len=4096"), std::string::npos) << s;
    EXPECT_NE(s.find("select_len=1"), std::string::npos) << s;
    EXPECT_NE(s.find("attn_seq_len=1"), std::string::npos) << s;
    EXPECT_NE(s.find("rope_batch=1"), std::string::npos) << s;
    EXPECT_NE(s.find("packed_mask_len=4"), std::string::npos) << s;
    EXPECT_NE(s.find("context_mask_selector_len=0"), std::string::npos) << s;
    EXPECT_NE(s.find("start_index_len=2"), std::string::npos) << s;
    EXPECT_NE(s.find("spec_verify_phase_len=0"), std::string::npos) << s;
}

// ---------------------------------------------------------------------------
// firstInvalidMember — 5 cases from design §6.1.
// ---------------------------------------------------------------------------

TEST(InferenceDimsTest, FirstInvalidMemberAllZero)
{
    // InferenceDims{} (aggregate init with no args) → all zero.
    // Every referenced member is invalid; returns the first one (batch).
    InferenceDims const d{};
    auto const refs = allReferenced();
    EXPECT_EQ(firstInvalidMember(d, refs), &InferenceDims::batch);
}

TEST(InferenceDimsTest, FirstInvalidMemberPartialSet)
{
    // Caller sets only batch, leaves the rest zero.
    InferenceDims d{};
    d.batch = 4;
    auto const refs = allReferenced();
    // First *invalid* is seqLen (next referenced member that's still zero).
    EXPECT_EQ(firstInvalidMember(d, refs), &InferenceDims::seqLen);
}

TEST(InferenceDimsTest, FirstInvalidMemberUnreferencedFieldZero)
{
    // packedMaskLen is 0, but the engine does not reference it (no packed mask
    // tensor in this registry) — not checked, returns nullptr.
    InferenceDims d = makeValid();
    d.packedMaskLen = 0;
    std::vector<int64_t InferenceDims::*> const refs{
        &InferenceDims::batch,
        &InferenceDims::seqLen,
        &InferenceDims::kvLen,
        &InferenceDims::selectLen,
        &InferenceDims::ropeBatch,
    };
    EXPECT_EQ(firstInvalidMember(d, refs), nullptr);
}

TEST(InferenceDimsTest, FirstInvalidMemberValidPasses)
{
    InferenceDims const d = makeValid();
    auto const refs = allReferenced();
    EXPECT_EQ(firstInvalidMember(d, refs), nullptr);
}

TEST(InferenceDimsTest, FirstInvalidMemberEmptyReferencedAlwaysPasses)
{
    // Empty referenced list → nothing to check, always nullptr.
    InferenceDims const d{};
    std::vector<int64_t InferenceDims::*> const refs{};
    EXPECT_EQ(firstInvalidMember(d, refs), nullptr);
}

TEST(InferenceDimsTest, FirstInvalidMemberNegativeValueFails)
{
    // <= 0 is invalid, not just == 0.
    InferenceDims d = makeValid();
    d.kvLen = -1;
    auto const refs = allReferenced();
    EXPECT_EQ(firstInvalidMember(d, refs), &InferenceDims::kvLen);
}

TEST(InferenceDimsTest, FirstInvalidMemberStartIndexLenZeroIsValid)
{
    // `startIndexLen` is the sentinel for "initial prefill of an empty KV
    // cache" — zero is an engine-meaningful value, NOT a recipe-bypass. The
    // validator excludes it from the `> 0` check.
    InferenceDims d = makeValid();
    d.startIndexLen = 0;
    auto const refs = allReferenced();
    EXPECT_EQ(firstInvalidMember(d, refs), nullptr);
}

TEST(InferenceDimsTest, FirstInvalidMemberStartIndexLenNegativeFails)
{
    // Negative is still invalid even for the zero-allowed member.
    InferenceDims d = makeValid();
    d.startIndexLen = -1;
    auto const refs = allReferenced();
    EXPECT_EQ(firstInvalidMember(d, refs), &InferenceDims::startIndexLen);
}

TEST(InferenceDimsTest, FirstInvalidMemberContextMaskSelectorLenZeroIsValid)
{
    // `contextMaskSelectorLen` is a shape sentinel: 0 keeps causal/default
    // attention, while batch selects DiffusionGemma non-causal denoise.
    InferenceDims d = makeValid();
    d.contextMaskSelectorLen = 0;
    auto const refs = allReferenced();
    EXPECT_EQ(firstInvalidMember(d, refs), nullptr);
}

TEST(InferenceDimsTest, FirstInvalidMemberContextMaskSelectorLenNegativeFails)
{
    InferenceDims d = makeValid();
    d.contextMaskSelectorLen = -1;
    auto const refs = allReferenced();
    EXPECT_EQ(firstInvalidMember(d, refs), &InferenceDims::contextMaskSelectorLen);
}

TEST(InferenceDimsTest, FirstInvalidMemberSpecVerifyPhaseLenZeroIsValid)
{
    InferenceDims d = makeValid();
    d.specVerifyPhaseLen = 0;
    auto const refs = allReferenced();
    EXPECT_EQ(firstInvalidMember(d, refs), nullptr);
}

TEST(InferenceDimsTest, FirstInvalidMemberSpecVerifyPhaseLenNegativeFails)
{
    InferenceDims d = makeValid();
    d.specVerifyPhaseLen = -1;
    auto const refs = allReferenced();
    EXPECT_EQ(firstInvalidMember(d, refs), &InferenceDims::specVerifyPhaseLen);
}
