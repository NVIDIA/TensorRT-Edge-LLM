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

#include "common/executionPhase.h"
#include "plugins/utils/raggedPluginMetadata.h"

#include <gtest/gtest.h>
#include <stdexcept>
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
        /*.executionPhaseLen=*/static_cast<int64_t>(ExecutionPhase::kContextPrefill),
        /*.specVerifyLayoutLen=*/0,
        /*.skipSoftmaxPrefillScaleLen=*/0,
        /*.skipSoftmaxDecodeScaleLen=*/0,
        /*.swaKVCacheModeLen=*/0,
        /*.queryOffsetLen=*/3,
        /*.contextSequenceCount=*/2,
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
        &InferenceDims::executionPhaseLen,
        &InferenceDims::specVerifyLayoutLen,
        &InferenceDims::skipSoftmaxPrefillScaleLen,
        &InferenceDims::skipSoftmaxDecodeScaleLen,
        &InferenceDims::swaKVCacheModeLen,
        &InferenceDims::queryOffsetLen,
        &InferenceDims::contextSequenceCount,
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
    EXPECT_EQ(dimName(&InferenceDims::executionPhaseLen), "execution_phase_len");
    EXPECT_EQ(dimName(&InferenceDims::specVerifyLayoutLen), "spec_verify_layout_len");
    EXPECT_EQ(dimName(&InferenceDims::skipSoftmaxPrefillScaleLen), "skip_softmax_prefill_scale_len");
    EXPECT_EQ(dimName(&InferenceDims::skipSoftmaxDecodeScaleLen), "skip_softmax_decode_scale_len");
    EXPECT_EQ(dimName(&InferenceDims::swaKVCacheModeLen), "swa_kv_cache_mode_len");
    EXPECT_EQ(dimName(&InferenceDims::queryOffsetLen), "query_offset_len");
    EXPECT_EQ(dimName(&InferenceDims::contextSequenceCount), "context_sequence_count");
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
    EXPECT_NE(s.find("execution_phase_len=1"), std::string::npos) << s;
    EXPECT_NE(s.find("skip_softmax_prefill_scale_len=0"), std::string::npos) << s;
    EXPECT_NE(s.find("skip_softmax_decode_scale_len=0"), std::string::npos) << s;
    EXPECT_NE(s.find("swa_kv_cache_mode_len=0"), std::string::npos) << s;
    EXPECT_NE(s.find("query_offset_len=3"), std::string::npos) << s;
    EXPECT_NE(s.find("context_sequence_count=2"), std::string::npos) << s;
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

TEST(InferenceDimsTest, FirstInvalidMemberExecutionPhaseLenZeroFails)
{
    InferenceDims d = makeValid();
    d.executionPhaseLen = 0;
    auto const refs = allReferenced();
    EXPECT_EQ(firstInvalidMember(d, refs), &InferenceDims::executionPhaseLen);
}

TEST(InferenceDimsTest, FirstInvalidMemberExecutionPhaseLenNegativeFails)
{
    InferenceDims d = makeValid();
    d.executionPhaseLen = -1;
    auto const refs = allReferenced();
    EXPECT_EQ(firstInvalidMember(d, refs), &InferenceDims::executionPhaseLen);
}

TEST(InferenceDimsTest, ExecutionPhaseExtentAcceptsExactlyThePublishedEnum)
{
    EXPECT_EQ(static_cast<int32_t>(ExecutionPhase::kContextPrefill), 1);
    EXPECT_EQ(static_cast<int32_t>(ExecutionPhase::kContextChunk), 2);
    EXPECT_EQ(static_cast<int32_t>(ExecutionPhase::kAutoregressiveDecode), 3);
    EXPECT_EQ(static_cast<int32_t>(ExecutionPhase::kSpecDraftProposal), 4);
    EXPECT_EQ(static_cast<int32_t>(ExecutionPhase::kSpecTargetVerify), 5);
    EXPECT_EQ(static_cast<int32_t>(ExecutionPhase::kDiffusionDenoise), 6);
    EXPECT_EQ(static_cast<int32_t>(ExecutionPhase::kDiffusionCommit), 7);
    EXPECT_EQ(static_cast<int32_t>(ExecutionPhase::kMixedPrefillDecode), 8);
    EXPECT_FALSE(isExecutionPhaseExtent(0));
    for (int64_t extent = 1; extent <= 8; ++extent)
    {
        EXPECT_TRUE(isExecutionPhaseExtent(extent));
    }
    EXPECT_FALSE(isExecutionPhaseExtent(9));
}

TEST(InferenceDimsTest, ExecutionPhaseAccessorChecksAndDecodesExtent)
{
    InferenceDims dims = makeValid();
    dims.executionPhaseLen = static_cast<int64_t>(ExecutionPhase::kSpecDraftProposal);
    EXPECT_EQ(executionPhase(dims), ExecutionPhase::kSpecDraftProposal);

    dims.executionPhaseLen = 0;
    EXPECT_THROW(executionPhase(dims), std::runtime_error);
}

TEST(RaggedPluginMetadataTest, DecodesHomogeneousContextAndDecodeCarriers)
{
    auto descriptor = [](int32_t extent) {
        nvinfer1::PluginTensorDesc desc{};
        desc.dims.nbDims = 1;
        desc.dims.d[0] = extent;
        return desc;
    };
    auto const activation = descriptor(12);
    auto const queryLengths = descriptor(3);
    auto const offsets = descriptor(4);

    auto const context = trt_edgellm::plugins::decodeRaggedPluginMetadata("test", activation, queryLengths, offsets,
        descriptor(static_cast<int32_t>(ExecutionPhase::kContextPrefill)), descriptor(3));
    EXPECT_EQ(context.numContextSequences, 3);
    EXPECT_EQ(context.contextExecutionRows, 12);
    EXPECT_EQ(context.decodeExecutionRows, 0);

    auto const decode = trt_edgellm::plugins::decodeRaggedPluginMetadata("test", descriptor(3), queryLengths, offsets,
        descriptor(static_cast<int32_t>(ExecutionPhase::kAutoregressiveDecode)), descriptor(0));
    EXPECT_EQ(decode.numDecodeSequences, 3);
    EXPECT_EQ(decode.contextExecutionRows, 0);
    EXPECT_EQ(decode.decodeExecutionRows, 3);
}

TEST(RaggedPluginMetadataTest, RejectsCarrierMismatchAndInvalidExtent)
{
    // Given homogeneous-phase carriers whose context count, phase extent, or ranks disagree with the contract,
    // When a plugin decodes them,
    // Then decoding fails instead of returning a geometry.
    auto descriptor = [](int32_t extent) {
        nvinfer1::PluginTensorDesc desc{};
        desc.dims.nbDims = 1;
        desc.dims.d[0] = extent;
        return desc;
    };
    auto const activation = descriptor(12);
    auto const queryLengths = descriptor(3);
    auto const offsets = descriptor(4);
    EXPECT_THROW(trt_edgellm::plugins::decodeRaggedPluginMetadata("test", activation, queryLengths, offsets,
                     descriptor(static_cast<int32_t>(ExecutionPhase::kContextPrefill)), descriptor(0)),
        std::runtime_error);
    EXPECT_THROW(trt_edgellm::plugins::decodeRaggedPluginMetadata(
                     "test", activation, queryLengths, offsets, descriptor(9), descriptor(0)),
        std::runtime_error);
    EXPECT_THROW(trt_edgellm::plugins::decodeRaggedPluginMetadata("test", activation, queryLengths, offsets,
                     descriptor(static_cast<int32_t>(ExecutionPhase::kAutoregressiveDecode)), descriptor(0)),
        std::runtime_error);

    EXPECT_THROW(trt_edgellm::plugins::decodeRaggedPluginMetadata("test", descriptor(10), queryLengths, offsets,
                     descriptor(static_cast<int32_t>(ExecutionPhase::kContextPrefill)), descriptor(3)),
        std::runtime_error);

    auto invalidRank = descriptor(3);
    invalidRank.dims.nbDims = 0;
    EXPECT_THROW(trt_edgellm::plugins::decodeRaggedPluginMetadata("test", activation, invalidRank, offsets,
                     descriptor(static_cast<int32_t>(ExecutionPhase::kContextPrefill)), descriptor(3)),
        std::runtime_error);
}

TEST(RaggedPluginMetadataTest, DecodesSpecVerifyLayoutFromDescriptorExtent)
{
    // Given a rank-one INT32 layout marker, when phase and extent are decoded,
    // then only inactive non-verify and chain/tree target-verify pairs are valid.
    auto descriptor = [](int32_t extent, nvinfer1::DataType type = nvinfer1::DataType::kINT32) {
        nvinfer1::PluginTensorDesc desc{};
        desc.dims.nbDims = 1;
        desc.dims.d[0] = extent;
        desc.type = type;
        desc.format = nvinfer1::TensorFormat::kLINEAR;
        return desc;
    };

    EXPECT_FALSE(trt_edgellm::plugins::decodeSpecVerifyLayout("test", descriptor(0), ExecutionPhase::kContextPrefill)
            .has_value());
    EXPECT_EQ(trt_edgellm::plugins::decodeSpecVerifyLayout("test", descriptor(1), ExecutionPhase::kSpecTargetVerify),
        SpecVerifyLayout::kChain);
    EXPECT_EQ(trt_edgellm::plugins::decodeSpecVerifyLayout("test", descriptor(2), ExecutionPhase::kSpecTargetVerify),
        SpecVerifyLayout::kTree);
    EXPECT_THROW(trt_edgellm::plugins::decodeSpecVerifyLayout("test", descriptor(0), ExecutionPhase::kSpecTargetVerify),
        std::runtime_error);
    EXPECT_THROW(trt_edgellm::plugins::decodeSpecVerifyLayout("test", descriptor(1), ExecutionPhase::kContextPrefill),
        std::runtime_error);
    EXPECT_THROW(trt_edgellm::plugins::decodeSpecVerifyLayout("test", descriptor(3), ExecutionPhase::kSpecTargetVerify),
        std::runtime_error);
    EXPECT_THROW(trt_edgellm::plugins::decodeSpecVerifyLayout(
                     "test", descriptor(1, nvinfer1::DataType::kFLOAT), ExecutionPhase::kSpecTargetVerify),
        std::runtime_error);
    auto invalidRank = descriptor(1);
    invalidRank.dims.nbDims = 0;
    EXPECT_THROW(trt_edgellm::plugins::decodeSpecVerifyLayout("test", invalidRank, ExecutionPhase::kSpecTargetVerify),
        std::runtime_error);
}

TEST(RaggedPluginMetadataTest, RejectsLayoutExtentsThatNarrowToValidValues)
{
    // Given 64-bit descriptor extents with valid low bits, when the marker is decoded,
    // then validation rejects the full extent instead of accepting a truncated layout.
    constexpr int64_t kHIGH_BITS = int64_t{1} << 32;
    for (int64_t const lowBits : {0, 1, 2})
    {
        nvinfer1::PluginTensorDesc descriptor{};
        descriptor.dims.nbDims = 1;
        descriptor.dims.d[0] = kHIGH_BITS + lowBits;
        descriptor.type = nvinfer1::DataType::kINT32;
        descriptor.format = nvinfer1::TensorFormat::kLINEAR;
        auto const phase = lowBits == 0 ? ExecutionPhase::kContextPrefill : ExecutionPhase::kSpecTargetVerify;
        SCOPED_TRACE(descriptor.dims.d[0]);
        EXPECT_THROW(trt_edgellm::plugins::decodeSpecVerifyLayout("test", descriptor, phase), std::runtime_error);
    }
}

TEST(RaggedPluginMetadataTest, DefaultLayoutSupportRejectsMixedPhase)
{
    // Given a well-formed phase 8 step (1 of 3 sequences in prefill),
    // When a plugin that keeps the default padded-only layout support decodes it,
    // Then decoding fails, so plugins without compact support keep rejecting phase 8.
    auto descriptor = [](int32_t extent) {
        nvinfer1::PluginTensorDesc desc{};
        desc.dims.nbDims = 1;
        desc.dims.d[0] = extent;
        return desc;
    };
    auto const phase8 = descriptor(static_cast<int32_t>(ExecutionPhase::kMixedPrefillDecode));
    EXPECT_THROW(trt_edgellm::plugins::decodeRaggedPluginMetadata(
                     "test", descriptor(12), descriptor(3), descriptor(4), phase8, descriptor(1)),
        std::runtime_error);
    EXPECT_THROW(trt_edgellm::plugins::decodeRaggedPluginMetadata("test", descriptor(12), descriptor(3), descriptor(4),
                     phase8, descriptor(1), trt_edgellm::plugins::RaggedLayoutSupport::kPaddedOnly),
        std::runtime_error);
}

TEST(RaggedPluginMetadataTest, CompactLayoutSupportDecodesMixedPhaseGeometry)
{
    // Given compact phase 8 steps whose token count is not a multiple of N, with and without decode sequences,
    // When a plugin that declares compact layout support decodes them,
    // Then the prefill rows are T minus one row per decode sequence and the decode rows equal Nd.
    auto descriptor = [](int32_t extent) {
        nvinfer1::PluginTensorDesc desc{};
        desc.dims.nbDims = 1;
        desc.dims.d[0] = extent;
        return desc;
    };
    auto const phase8 = descriptor(static_cast<int32_t>(ExecutionPhase::kMixedPrefillDecode));
    auto const compact = trt_edgellm::plugins::RaggedLayoutSupport::kPaddedAndCompact;

    auto const mixed = trt_edgellm::plugins::decodeRaggedPluginMetadata(
        "test", descriptor(10), descriptor(3), descriptor(4), phase8, descriptor(1), compact);
    EXPECT_EQ(mixed.phase, ExecutionPhase::kMixedPrefillDecode);
    EXPECT_EQ(mixed.numSequences, 3);
    EXPECT_EQ(mixed.numContextSequences, 1);
    EXPECT_EQ(mixed.numDecodeSequences, 2);
    EXPECT_EQ(mixed.physicalTokens, 10);
    EXPECT_EQ(mixed.contextExecutionRows, 8);
    EXPECT_EQ(mixed.decodeExecutionRows, 2);

    auto const prefillOnly = trt_edgellm::plugins::decodeRaggedPluginMetadata(
        "test", descriptor(4), descriptor(2), descriptor(3), phase8, descriptor(2), compact);
    EXPECT_EQ(prefillOnly.numContextSequences, 2);
    EXPECT_EQ(prefillOnly.numDecodeSequences, 0);
    EXPECT_EQ(prefillOnly.contextExecutionRows, 4);
    EXPECT_EQ(prefillOnly.decodeExecutionRows, 0);
}

TEST(RaggedPluginMetadataTest, CompactLayoutSupportDecodesPureDecodeStepOnlyWithOneRowPerSequence)
{
    // Given phase 8 steps whose context-count carrier reports no prefill sequence, with T_exec = N and T_exec > N,
    // When a plugin that declares compact layout support decodes them,
    // Then T_exec = N decodes as N decode rows and no prefill rows, and T_exec > N fails because rows without a
    // prefill sequence cannot be placed.
    auto descriptor = [](int32_t extent) {
        nvinfer1::PluginTensorDesc desc{};
        desc.dims.nbDims = 1;
        desc.dims.d[0] = extent;
        return desc;
    };
    auto const phase8 = descriptor(static_cast<int32_t>(ExecutionPhase::kMixedPrefillDecode));
    auto const compact = trt_edgellm::plugins::RaggedLayoutSupport::kPaddedAndCompact;
    auto const decodeOnly = trt_edgellm::plugins::decodeRaggedPluginMetadata(
        "test", descriptor(3), descriptor(3), descriptor(4), phase8, descriptor(0), compact);
    EXPECT_EQ(decodeOnly.numContextSequences, 0);
    EXPECT_EQ(decodeOnly.numDecodeSequences, 3);
    EXPECT_EQ(decodeOnly.contextExecutionRows, 0);
    EXPECT_EQ(decodeOnly.decodeExecutionRows, 3);
    EXPECT_THROW(trt_edgellm::plugins::decodeRaggedPluginMetadata(
                     "test", descriptor(4), descriptor(3), descriptor(4), phase8, descriptor(0), compact),
        std::runtime_error);
}

TEST(RaggedPluginMetadataTest, CompactLayoutSupportKeepsHomogeneousPhaseRules)
{
    // Given padded homogeneous carriers that the default decoder accepts or rejects,
    // When a plugin that declares compact layout support decodes them,
    // Then it returns the same geometry and rejects the same non-divisible prefill, so compact support only
    // changes phase 8.
    auto descriptor = [](int32_t extent) {
        nvinfer1::PluginTensorDesc desc{};
        desc.dims.nbDims = 1;
        desc.dims.d[0] = extent;
        return desc;
    };
    auto const compact = trt_edgellm::plugins::RaggedLayoutSupport::kPaddedAndCompact;
    auto const prefill = descriptor(static_cast<int32_t>(ExecutionPhase::kContextPrefill));
    auto const context = trt_edgellm::plugins::decodeRaggedPluginMetadata(
        "test", descriptor(12), descriptor(3), descriptor(4), prefill, descriptor(3), compact);
    EXPECT_EQ(context.numContextSequences, 3);
    EXPECT_EQ(context.contextExecutionRows, 12);
    EXPECT_EQ(context.decodeExecutionRows, 0);
    EXPECT_THROW(trt_edgellm::plugins::decodeRaggedPluginMetadata(
                     "test", descriptor(10), descriptor(3), descriptor(4), prefill, descriptor(3), compact),
        std::runtime_error);
}

TEST(RaggedPluginMetadataTest, ValidatesIndexedResidentStateDescriptors)
{
    auto descriptor = [](std::initializer_list<int32_t> dims, nvinfer1::DataType type) {
        nvinfer1::PluginTensorDesc desc{};
        desc.dims.nbDims = static_cast<int32_t>(dims.size());
        std::copy(dims.begin(), dims.end(), desc.dims.d);
        desc.type = type;
        desc.format = nvinfer1::TensorFormat::kLINEAR;
        return desc;
    };

    auto const stateIndices = descriptor({3}, nvinfer1::DataType::kINT32);
    auto const state = descriptor({5, 4, 128, 128}, nvinfer1::DataType::kFLOAT);
    EXPECT_EQ(trt_edgellm::plugins::validateIndexedResidentStateDescriptors(
                  "test", 3, stateIndices, state, state, nvinfer1::DataType::kFLOAT, {4, 128, 128}),
        5);

    EXPECT_THROW(trt_edgellm::plugins::validateIndexedResidentStateDescriptors(
                     "test", 2, stateIndices, state, state, nvinfer1::DataType::kFLOAT, {4, 128, 128}),
        std::runtime_error);
    EXPECT_THROW(
        trt_edgellm::plugins::validateIndexedResidentStateDescriptors("test", 3,
            descriptor({3}, nvinfer1::DataType::kINT64), state, state, nvinfer1::DataType::kFLOAT, {4, 128, 128}),
        std::runtime_error);
    EXPECT_THROW(
        trt_edgellm::plugins::validateIndexedResidentStateDescriptors("test", 3, stateIndices,
            descriptor({0, 4, 128, 128}, nvinfer1::DataType::kFLOAT), state, nvinfer1::DataType::kFLOAT, {4, 128, 128}),
        std::runtime_error);
    EXPECT_THROW(
        trt_edgellm::plugins::validateIndexedResidentStateDescriptors("test", 3, stateIndices, state,
            descriptor({5, 4, 128, 64}, nvinfer1::DataType::kFLOAT), nvinfer1::DataType::kFLOAT, {4, 128, 128}),
        std::runtime_error);
}

TEST(InferenceDimsTest, FirstInvalidMemberSwaKVCacheModeLenZeroIsValid)
{
    InferenceDims d = makeValid();
    d.swaKVCacheModeLen = 0;
    auto const refs = allReferenced();
    EXPECT_EQ(firstInvalidMember(d, refs), nullptr);
}

TEST(InferenceDimsTest, FirstInvalidMemberSwaKVCacheModeLenNegativeFails)
{
    InferenceDims d = makeValid();
    d.swaKVCacheModeLen = -1;
    auto const refs = allReferenced();
    EXPECT_EQ(firstInvalidMember(d, refs), &InferenceDims::swaKVCacheModeLen);
}
