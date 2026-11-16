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

#include "runtime/decoding/jumpForward.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/specDecodeLimits.h"
#include "common/utf8.h"
#include "kernels/speculative/speculativeKernels.h"
#include "profiling/nvtx_wrapper.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/specDecodeUtils.h"
#include "runtime/state/decodingInferenceContext.h"
#include "runtime/state/pipelineIO.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

namespace
{

//! Forced text is re-encoded every step; a token rarely spells more bytes than this, so a longer prefix than the
//! budget can take only costs host time.
constexpr size_t kMAX_BYTES_PER_TOKEN{16};

//! Filler for the unused entries of a forced token's logprob row; consumers stop at a negative id.
constexpr std::pair<int32_t, float> kLOGPROB_FILLER{-1, -std::numeric_limits<float>::infinity()};

} // namespace

char const* jumpSkipReasonName(JumpSkipReason reason) noexcept
{
    switch (reason)
    {
    case JumpSkipReason::kNone: return "none";
    case JumpSkipReason::kEmptyForcedString: return "empty";
    case JumpSkipReason::kIncompleteUtf8: return "incomplete_utf8";
    case JumpSkipReason::kRetokenizeFailed: return "retokenize_failed";
    case JumpSkipReason::kNotAtWordBoundary: return "not_at_word_boundary";
    case JumpSkipReason::kBoundaryConflict: return "boundary_conflict";
    case JumpSkipReason::kNoBudget: return "no_budget";
    case JumpSkipReason::kPrunedToken: return "pruned_token";
    case JumpSkipReason::kTrailingTokenDropped: return "trailing_dropped";
    case JumpSkipReason::kRespellRefused: return "respell_refused";
    case JumpSkipReason::kDraftLonger: return "draft_longer";
    case JumpSkipReason::kCount: break;
    }
    return "unknown";
}

std::string formatJumpForwardStats(JumpForwardSlotStats const& stats)
{
    std::string text = "jumps=" + std::to_string(stats.jumps) + " forced=" + std::to_string(stats.forcedTokens)
        + " rolledBack=" + std::to_string(stats.rolledBackTokens) + " chunkSteps=" + std::to_string(stats.chunkSteps)
        + " skips{";
    bool first = true;
    for (size_t reason = 1; reason < stats.skips.size(); ++reason)
    {
        if (stats.skips[reason] == 0)
        {
            continue;
        }
        text += (first ? "" : ",") + std::string(jumpSkipReasonName(static_cast<JumpSkipReason>(reason))) + "="
            + std::to_string(stats.skips[reason]);
        first = false;
    }
    return text + "}";
}

std::string_view completeUtf8Prefix(std::string_view text) noexcept
{
    size_t end = 0;
    while (end < text.size())
    {
        int const need = utf8::leaderByteLen(static_cast<unsigned char>(text[end]));
        if (need == 0 || end + static_cast<size_t>(need) > text.size())
        {
            break;
        }
        bool continuationsValid = true;
        for (int i = 1; i < need && continuationsValid; ++i)
        {
            continuationsValid = (static_cast<unsigned char>(text[end + static_cast<size_t>(i)]) & 0xC0U) == 0x80U;
        }
        if (!continuationsValid)
        {
            break;
        }
        end += static_cast<size_t>(need);
    }
    return text.substr(0, end);
}

JumpForwardPlanner::JumpForwardPlanner(
    tokenizer::Tokenizer const& tokenizer, GuidedDecoder const& decoder, int32_t maxJumpTokens)
    : mTokenizer(&tokenizer)
    , mDecoder(&decoder)
    , mMaxJumpTokens(maxJumpTokens)
    , mNeedsWordBoundary(tokenizer.prependsSpaceMarker())
{
}

void JumpForwardPlanner::buildWindow(
    std::vector<int32_t> const& tokens, int32_t generatedCount, std::vector<int32_t>& window) const
{
    size_t begin = tokens.empty() ? 0 : tokens.size() - 1;
    if (mNeedsWordBoundary && !tokens.empty())
    {
        size_t const generated = static_cast<size_t>(std::max(generatedCount, 0));
        size_t const lowest = tokens.size() - std::min(tokens.size(), std::min(generated, kMaxWindowTokens));
        while (begin > lowest)
        {
            std::string const piece = mTokenizer->idToPiece(tokens[begin], /*skipSpecialTokens=*/false);
            if (!piece.empty() && piece.front() == ' ')
            {
                break;
            }
            --begin;
        }
    }
    window.assign(tokens.begin() + static_cast<std::ptrdiff_t>(begin), tokens.end());
}

JumpForwardPlan JumpForwardPlanner::plan(int32_t slot, std::string const& forced,
    std::vector<int32_t> const& windowFullIds, int32_t mutableTailCount, int32_t budgetTokens) const
{
    JumpForwardPlan result;
    auto const skip = [&result](JumpSkipReason reason) {
        result.rollbackCount = 0;
        result.fullVocabIds.clear();
        result.outputSpaceIds.clear();
        result.skipReason = reason;
        return result;
    };

    if (forced.empty())
    {
        return skip(JumpSkipReason::kEmptyForcedString);
    }
    std::string_view forcedPrefix = completeUtf8Prefix(forced);
    if (forcedPrefix.empty())
    {
        return skip(JumpSkipReason::kIncompleteUtf8);
    }
    int32_t const budget = std::min(budgetTokens, mMaxJumpTokens);
    if (budget <= 0)
    {
        return skip(JumpSkipReason::kNoBudget);
    }
    size_t const maxForcedBytes = static_cast<size_t>(mMaxJumpTokens) * kMAX_BYTES_PER_TOKEN;
    if (forcedPrefix.size() > maxForcedBytes)
    {
        forcedPrefix = completeUtf8Prefix(forcedPrefix.substr(0, maxForcedBytes));
    }

    auto const windowSize = static_cast<int32_t>(windowFullIds.size());
    int32_t const readOnlyCount = windowSize - std::clamp(mutableTailCount, 0, windowSize);

    // pieceStart[i] is where window token i begins in `text`; pieceStart[windowSize] is where the
    // forced text begins.
    std::string text;
    std::vector<size_t> pieceStart;
    pieceStart.reserve(static_cast<size_t>(windowSize) + 1);
    for (int32_t const id : windowFullIds)
    {
        pieceStart.push_back(text.size());
        text += mTokenizer->idToPiece(id, /*skipSpecialTokens=*/false);
    }
    pieceStart.push_back(text.size());
    text.append(forcedPrefix.data(), forcedPrefix.size());

    if (mNeedsWordBoundary && text.front() != ' ')
    {
        return skip(JumpSkipReason::kNotAtWordBoundary);
    }

    std::vector<int32_t> const tokens = mTokenizer->encode(text);
    if (tokens.empty())
    {
        return skip(JumpSkipReason::kRetokenizeFailed);
    }

    int32_t kept = 0;
    while (kept < windowSize && kept < static_cast<int32_t>(tokens.size())
        && tokens[static_cast<size_t>(kept)] == windowFullIds[static_cast<size_t>(kept)])
    {
        ++kept;
    }
    if (kept < readOnlyCount)
    {
        return skip(JumpSkipReason::kBoundaryConflict);
    }
    if (kept >= static_cast<int32_t>(tokens.size()))
    {
        return skip(JumpSkipReason::kRetokenizeFailed);
    }
    result.rollbackCount = windowSize - kept;

    // The new tokens must spell the rolled-back window text plus the forced text, or a prefix of it
    // that still reaches past the window text. Truncation at a pruned token or at the budget can
    // leave a run that does not, and that run is a skip rather than a shorter jump.
    std::string_view const expected = std::string_view(text).substr(pieceStart[static_cast<size_t>(kept)]);
    size_t const windowTailSize = pieceStart[static_cast<size_t>(windowSize)] - pieceStart[static_cast<size_t>(kept)];
    int32_t const maxNewTokens = budget + result.rollbackCount;
    JumpSkipReason truncatedBy = JumpSkipReason::kNone;
    std::string spelled;
    for (size_t i = static_cast<size_t>(kept); i < tokens.size(); ++i)
    {
        if (static_cast<int32_t>(result.fullVocabIds.size()) >= maxNewTokens)
        {
            truncatedBy = JumpSkipReason::kNoBudget;
            break;
        }
        // A stop token is as unusable as a pruned one: forced text never ends the request.
        int32_t const outputId = mDecoder->toOutputSpace(tokens[i]);
        if (outputId < 0 || mTokenizer->isEosToken(tokens[i]))
        {
            truncatedBy = JumpSkipReason::kPrunedToken;
            break;
        }
        result.fullVocabIds.push_back(tokens[i]);
        result.outputSpaceIds.push_back(outputId);
        spelled += mTokenizer->idToPiece(tokens[i], /*skipSpecialTokens=*/false);
    }
    if (spelled.size() > expected.size() || expected.compare(0, spelled.size(), spelled) != 0)
    {
        return skip(JumpSkipReason::kRetokenizeFailed);
    }
    if (spelled.size() <= windowTailSize)
    {
        return skip(truncatedBy == JumpSkipReason::kNone ? JumpSkipReason::kRetokenizeFailed : truncatedBy);
    }

    // The text after the last token is not known yet: when another vocabulary token extends its piece, a canonical
    // tokenization may merge it with what follows. A jump that ends the grammar has nothing after it to merge with.
    if (mDecoder->isExtendable(result.outputSpaceIds.back()))
    {
        bool const completes
            = spelled.size() == expected.size() && mDecoder->completesGrammar(slot, std::string(forcedPrefix));
        if (!completes)
        {
            std::string const lastPiece
                = mTokenizer->idToPiece(result.fullVocabIds.back(), /*skipSpecialTokens=*/false);
            result.fullVocabIds.pop_back();
            result.outputSpaceIds.pop_back();
            spelled.resize(spelled.size() - lastPiece.size());
            if (spelled.size() <= windowTailSize)
            {
                return skip(JumpSkipReason::kTrailingTokenDropped);
            }
        }
    }
    return result;
}

HostDraftTree::HostDraftTree(int32_t maxBatchSize, int32_t width)
    : mMaxBatchSize(maxBatchSize)
    , mWidth(width)
{
    ELLM_CHECK(maxBatchSize > 0 && width > 0, "A host draft tree needs a positive batch size and width");
    mIds = Tensor({maxBatchSize, width}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "HostDraftTree::ids");
    mParents = Tensor({maxBatchSize, width}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "HostDraftTree::parents");
    mDepths = Tensor({maxBatchSize, width}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "HostDraftTree::depths");
    mForced = Tensor({maxBatchSize, width}, DeviceType::kCPU, trt_edgellm::DataType::kINT8, "HostDraftTree::forced");
    mMask = Tensor({maxBatchSize, width, width}, DeviceType::kCPU, trt_edgellm::DataType::kINT8, "HostDraftTree::mask");
    mValidCounts = Tensor({maxBatchSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "HostDraftTree::counts");
    size_t const nodes = static_cast<size_t>(maxBatchSize) * static_cast<size_t>(width);
    mChainOf.assign(nodes, -1);
    mForcedChild.assign(nodes, -1);
    mFirstChild.assign(nodes, -1);
    mNextSibling.assign(nodes, -1);
    std::memset(mIds.rawPointer(), 0, nodes * sizeof(int32_t));
    std::fill_n(mParents.dataPointer<int32_t>(), nodes, -1);
    std::fill_n(mDepths.dataPointer<int32_t>(), nodes, -1);
    std::memset(mForced.rawPointer(), 0, nodes);
    std::memset(mMask.rawPointer(), 0, nodes * static_cast<size_t>(width));
    std::fill_n(mValidCounts.dataPointer<int32_t>(), static_cast<size_t>(maxBatchSize), 0);
}

size_t HostDraftTree::nodeOffset(int32_t slot) const
{
    ELLM_CHECK(slot >= 0 && slot < mMaxBatchSize, "Host draft tree slot out of range");
    return static_cast<size_t>(slot) * static_cast<size_t>(mWidth);
}

void HostDraftTree::loadDraft(int32_t slot, int32_t const* tokens, int32_t const* parents, int32_t draftWidth)
{
    load(slot, tokens, parents, draftWidth);
}

void HostDraftTree::loadChain(int32_t slot, int32_t const* tokens, int32_t count)
{
    load(slot, tokens, /*parents=*/nullptr, count);
}

void HostDraftTree::load(int32_t slot, int32_t const* tokens, int32_t const* parents, int32_t draftWidth)
{
    ELLM_CHECK(draftWidth > 0 && draftWidth <= mWidth, "Draft tree does not fit the grafting width");
    size_t const offset = nodeOffset(slot);
    auto const width = static_cast<size_t>(mWidth);
    int32_t* const ids = mIds.dataPointer<int32_t>() + offset;
    int32_t* const parentIds = mParents.dataPointer<int32_t>() + offset;
    int32_t* const depths = mDepths.dataPointer<int32_t>() + offset;
    int8_t* const forced = mForced.dataPointer<int8_t>() + offset;
    int8_t* const mask = mMask.dataPointer<int8_t>() + offset * width;
    std::memset(mask, 0, width * width);
    std::memset(forced, 0, width);
    auto const begin = static_cast<std::ptrdiff_t>(offset);
    std::fill_n(mChainOf.begin() + begin, width, static_cast<int8_t>(-1));
    std::fill_n(mForcedChild.begin() + begin, width, -1);
    std::fill_n(mFirstChild.begin() + begin, width, -1);
    std::fill_n(mNextSibling.begin() + begin, width, -1);
    for (int32_t node = 0; node < draftWidth; ++node)
    {
        int32_t const parent = node == 0 ? -1 : (parents != nullptr ? parents[node] : node - 1);
        ELLM_CHECK(parent < node, "Draft tree nodes must follow their parents");
        ids[node] = tokens[node];
        parentIds[node] = parent;
        int8_t* const row = mask + static_cast<size_t>(node) * width;
        if (parent >= 0)
        {
            depths[node] = depths[parent] + 1;
            std::memcpy(row, mask + static_cast<size_t>(parent) * width, width);
        }
        else
        {
            depths[node] = 0;
        }
        row[node] = 1;
    }
    for (int32_t node = draftWidth; node < mWidth; ++node)
    {
        ids[node] = 0;
        parentIds[node] = -1;
        depths[node] = -1;
    }
    // Prepending keeps siblings in ascending index order, since nodes follow their parents.
    for (int32_t node = draftWidth - 1; node >= 1; --node)
    {
        int32_t const parent = parentIds[node];
        if (parent >= 0)
        {
            mNextSibling[offset + static_cast<size_t>(node)] = mFirstChild[offset + static_cast<size_t>(parent)];
            mFirstChild[offset + static_cast<size_t>(parent)] = node;
        }
    }
    mValidCounts.dataPointer<int32_t>()[slot] = draftWidth;
}

int32_t HostDraftTree::insertChain(int32_t slot, int32_t attach, int32_t const* tokens, int32_t count, int8_t chainId)
{
    size_t const offset = nodeOffset(slot);
    auto const width = static_cast<size_t>(mWidth);
    int32_t* const ids = mIds.dataPointer<int32_t>() + offset;
    int32_t* const parentIds = mParents.dataPointer<int32_t>() + offset;
    int32_t* const depths = mDepths.dataPointer<int32_t>() + offset;
    int8_t* const forced = mForced.dataPointer<int8_t>() + offset;
    int8_t* const mask = mMask.dataPointer<int8_t>() + offset * width;
    int32_t& nodeCount = mValidCounts.dataPointer<int32_t>()[slot];
    ELLM_CHECK(attach >= 0 && attach < nodeCount && chainId >= 0, "A chain attaches to an existing node");
    int32_t node = attach;
    int32_t inserted = 0;
    for (int32_t k = 0; k < count; ++k)
    {
        int32_t const token = tokens[k];
        int32_t child = mFirstChild[offset + static_cast<size_t>(node)];
        while (child >= 0 && ids[child] != token)
        {
            child = mNextSibling[offset + static_cast<size_t>(child)];
        }
        if (child < 0)
        {
            ELLM_CHECK(nodeCount < mWidth, "A forced chain exceeds the grafting width");
            child = nodeCount++;
            ids[child] = token;
            parentIds[child] = node;
            depths[child] = depths[node] + 1;
            std::memcpy(mask + static_cast<size_t>(child) * width, mask + static_cast<size_t>(node) * width, width);
            mask[static_cast<size_t>(child) * width + static_cast<size_t>(child)] = 1;
            mNextSibling[offset + static_cast<size_t>(child)] = mFirstChild[offset + static_cast<size_t>(node)];
            mFirstChild[offset + static_cast<size_t>(node)] = child;
            ++inserted;
        }
        forced[child] = 1;
        mChainOf[offset + static_cast<size_t>(child)] = chainId;
        mForcedChild[offset + static_cast<size_t>(node)] = child;
        node = child;
    }
    return inserted;
}

void HostDraftTree::truncate(int32_t slot, int32_t count)
{
    size_t const offset = nodeOffset(slot);
    auto const width = static_cast<size_t>(mWidth);
    int32_t& nodeCount = mValidCounts.dataPointer<int32_t>()[slot];
    ELLM_CHECK(count >= 1 && count <= nodeCount, "A draft tree is truncated to between one node and its node count");
    int32_t* const firstChild = mFirstChild.data() + offset;
    int32_t* const nextSibling = mNextSibling.data() + offset;
    int32_t* const forcedChild = mForcedChild.data() + offset;
    for (int32_t node = 0; node < count; ++node)
    {
        int32_t* link = firstChild + node;
        while (*link >= 0)
        {
            if (*link >= count)
            {
                *link = nextSibling[*link];
            }
            else
            {
                link = nextSibling + *link;
            }
        }
        if (forcedChild[node] >= count)
        {
            forcedChild[node] = -1;
        }
    }
    int32_t* const ids = mIds.dataPointer<int32_t>() + offset;
    int32_t* const parentIds = mParents.dataPointer<int32_t>() + offset;
    int32_t* const depths = mDepths.dataPointer<int32_t>() + offset;
    int8_t* const forced = mForced.dataPointer<int8_t>() + offset;
    int8_t* const mask = mMask.dataPointer<int8_t>() + offset * width;
    for (int32_t node = count; node < nodeCount; ++node)
    {
        ids[node] = 0;
        parentIds[node] = -1;
        depths[node] = -1;
        forced[node] = 0;
        std::memset(mask + static_cast<size_t>(node) * width, 0, width);
        mChainOf[offset + static_cast<size_t>(node)] = -1;
        forcedChild[node] = -1;
        firstChild[node] = -1;
        nextSibling[node] = -1;
    }
    nodeCount = count;
}

int32_t HostDraftTree::count(int32_t slot) const
{
    ELLM_CHECK(slot >= 0 && slot < mMaxBatchSize, "Host draft tree slot out of range");
    return mValidCounts.dataPointer<int32_t>()[slot];
}

int32_t const* HostDraftTree::ids(int32_t slot) const
{
    return mIds.dataPointer<int32_t>() + nodeOffset(slot);
}

int32_t const* HostDraftTree::parents(int32_t slot) const
{
    return mParents.dataPointer<int32_t>() + nodeOffset(slot);
}

int32_t const* HostDraftTree::depths(int32_t slot) const
{
    return mDepths.dataPointer<int32_t>() + nodeOffset(slot);
}

int8_t const* HostDraftTree::forced(int32_t slot) const
{
    return mForced.dataPointer<int8_t>() + nodeOffset(slot);
}

int8_t const* HostDraftTree::chainOf(int32_t slot) const
{
    return mChainOf.data() + nodeOffset(slot);
}

int8_t const* HostDraftTree::mask(int32_t slot) const
{
    return mMask.dataPointer<int8_t>() + nodeOffset(slot) * static_cast<size_t>(mWidth);
}

int32_t HostDraftTree::forcedChildOf(int32_t slot, int32_t node) const
{
    ELLM_CHECK(node >= 0 && node < mWidth, "Host draft tree node out of range");
    return mForcedChild[nodeOffset(slot) + static_cast<size_t>(node)];
}

JumpForwardCoordinator::JumpForwardCoordinator(
    tokenizer::Tokenizer const& tokenizer, GuidedDecoder& decoder, int32_t maxJumpTokens)
    : mTokenizer(&tokenizer)
    , mDecoder(&decoder)
    , mPlanner(tokenizer, decoder, maxJumpTokens)
{
}

bool JumpForwardCoordinator::slotMayJump(DecodingInferenceContext const& context, int32_t slot) const
{
    auto const row = static_cast<size_t>(slot);
    // Same gate as the mask: a finished slot, or one still inside its reasoning block, is unconstrained.
    return context.jumpForwardEnabled[row] != 0 && context.finishedStates[row] == 0
        && context.guidedReasoningEnded[row] != 0 && mDecoder->hasGrammar(slot) && !mDecoder->isTerminated(slot);
}

JumpForwardPlan JumpForwardCoordinator::plan(
    DecodingInferenceContext& context, int32_t slot, int32_t mutableTailCount, int32_t budget)
{
    if (!slotMayJump(context, slot))
    {
        return {};
    }
    auto const row = static_cast<size_t>(slot);
    JumpForwardSlotStats& stats = context.jumpForwardStats[row];
    std::string const forced = mDecoder->forcedContinuation(slot);
    if (forced.empty())
    {
        ++stats.skips[static_cast<size_t>(JumpSkipReason::kEmptyForcedString)];
        JumpForwardPlan skipped;
        skipped.skipReason = JumpSkipReason::kEmptyForcedString;
        return skipped;
    }
    mPlanner.buildWindow(context.tokenIds[row], context.currentGenerateLengths[row], mWindow);
    JumpForwardPlan plan = mPlanner.plan(slot, forced, mWindow, mutableTailCount, budget);
    if (plan.empty())
    {
        ++stats.skips[static_cast<size_t>(plan.skipReason)];
    }
    return plan;
}

JumpForwardCoordinator::Applied JumpForwardCoordinator::applyPlan(
    DecodingInferenceContext& context, int32_t slot, JumpForwardPlan const& plan, int32_t sampledOutputSpaceId)
{
    auto const row = static_cast<size_t>(slot);
    JumpForwardSlotStats& stats = context.jumpForwardStats[row];
    std::vector<int32_t>& tokens = context.tokenIds[row];
    auto const planned = static_cast<int32_t>(plan.fullVocabIds.size());
    check::check(plan.rollbackCount >= 0 && plan.rollbackCount <= 1, "A jump rolls back at most the sampled token");
    check::check(!tokens.empty() && plan.fullVocabIds.size() == plan.outputSpaceIds.size(), "Malformed jump plan");
    check::check(planned > 0, "An applied jump plan carries at least one token");
    for (int32_t k = 0; k < planned; ++k)
    {
        check::check(!mTokenizer->isEosToken(plan.fullVocabIds[static_cast<size_t>(k)]),
            "A jump plan never carries a stop token");
    }

    int32_t const sampledToken = tokens.back();
    if (plan.rollbackCount == 1)
    {
        mDecoder->rollback(slot, 1);
    }
    int32_t const accepted = mDecoder->acceptForced(slot, plan.outputSpaceIds.data(), planned);
    if (plan.rollbackCount == 1)
    {
        // The sampled token must not be overwritten by a prefix of itself.
        size_t spelled = 0;
        for (int32_t k = 0; k < accepted; ++k)
        {
            spelled += mTokenizer->idToPiece(plan.fullVocabIds[static_cast<size_t>(k)], false).size();
        }
        if (spelled <= mTokenizer->idToPiece(sampledToken, false).size())
        {
            mDecoder->rollback(slot, accepted);
            check::check(mDecoder->advance(slot, sampledOutputSpaceId), "Re-accepting a sampled token failed");
            return {};
        }
    }
    else if (accepted == 0)
    {
        return {};
    }

    int32_t const numLogprobs = context.numLogprobsPerSlot[row];
    LogprobsSlot* const logprobs = numLogprobs > 0 ? &context.stepLogprobs[row] : nullptr;
    if (plan.rollbackCount == 1)
    {
        tokens.pop_back();
        --context.currentGenerateLengths[row];
        ++stats.rolledBackTokens;
        if (logprobs != nullptr)
        {
            --logprobs->numSteps;
        }
    }
    tokens.insert(tokens.end(), plan.fullVocabIds.begin(), plan.fullVocabIds.begin() + accepted);
    if (logprobs != nullptr)
    {
        writeSyntheticLogprobRows(*logprobs, numLogprobs, plan.fullVocabIds.data(), accepted);
    }
    context.currentGenerateLengths[row] += accepted;
    ++stats.jumps;
    stats.forcedTokens += accepted - plan.rollbackCount;
    return {plan.rollbackCount, accepted};
}

int32_t JumpForwardCoordinator::planDraftTree(
    DecodingInferenceContext& context, int32_t slot, HostDraftTree& tree, TreeWalkLimits const& limits)
{
    auto const row = static_cast<size_t>(slot);
    JumpForwardSlotStats& stats = context.jumpForwardStats[row];
    std::vector<int32_t> const& tokens = context.tokenIds[row];
    int32_t const draftCount = tree.count(slot);
    int32_t const width = tree.width();
    int32_t const* const ids = tree.ids(slot);
    int32_t const* const parents = tree.parents(slot);
    int32_t const* const depths = tree.depths(slot);
    int8_t const* const forced = tree.forced(slot);
    // The window reaches back over at most kMaxWindowTokens generated tokens, so that tail of the sequence stands in
    // for the whole of it; a node's root path follows it.
    auto const tail = static_cast<int32_t>(std::min(tokens.size(), JumpForwardPlanner::kMaxWindowTokens + 1));
    int32_t const generatedTail = std::min(context.currentGenerateLengths[row], tail);
    mSkip.assign(static_cast<size_t>(draftCount), 0);
    int32_t chains = 0;
    for (int32_t node = 0; node < draftCount; ++node)
    {
        if (node > 0)
        {
            int32_t const parent = parents[node];
            int32_t const forcedSibling = parent < 0 ? -1 : tree.forcedChildOf(slot, parent);
            if (parent < 0 || mSkip[static_cast<size_t>(parent)] != 0 || (forcedSibling >= 0 && forcedSibling != node))
            {
                mSkip[static_cast<size_t>(node)] = 1;
                continue;
            }
            if (forced[node] != 0)
            {
                continue; // A chain runs through it; its continuation is that chain's.
            }
        }
        int32_t const depth = depths[node];
        mPath.resize(static_cast<size_t>(depth));
        for (int32_t ancestor = node, level = depth - 1; level >= 0; ancestor = parents[ancestor], --level)
        {
            mPath[static_cast<size_t>(level)] = ids[ancestor];
        }
        int32_t accepted = 0;
        bool alive = true;
        for (int32_t const token : mPath)
        {
            int32_t const outputId = mDecoder->toOutputSpace(token);
            if (outputId < 0 || mDecoder->acceptForced(slot, &outputId, 1) == 0)
            {
                alive = false;
                break;
            }
            ++accepted;
        }
        // Nothing may follow the stop token, so the subtree is as dead as a refused one.
        alive = alive && !mDecoder->isTerminated(slot);
        if (!alive)
        {
            mSkip[static_cast<size_t>(node)] = 1;
        }
        else if (depth <= limits.maxPathLength - 2 && tree.count(slot) < width)
        {
            std::string const forcedText = mDecoder->forcedContinuation(slot);
            if (!forcedText.empty())
            {
                mScratch.assign(tokens.end() - tail, tokens.end());
                mScratch.insert(mScratch.end(), mPath.begin(), mPath.end());
                mPlanner.buildWindow(mScratch, generatedTail + depth, mWindow);
                int32_t const budget = std::min({limits.maxJumpTokens, limits.maxPathLength - 1 - depth,
                    limits.remainingGeneration - depth, width - tree.count(slot)});
                JumpForwardPlan const plan = mPlanner.plan(slot, forcedText, mWindow, /*mutableTailCount=*/1, budget);
                if (plan.rollbackCount > 0)
                {
                    ++stats.skips[static_cast<size_t>(JumpSkipReason::kRespellRefused)];
                }
                else if (plan.empty())
                {
                    ++stats.skips[static_cast<size_t>(plan.skipReason)];
                }
                else
                {
                    tree.insertChain(slot, node, plan.fullVocabIds.data(),
                        static_cast<int32_t>(plan.fullVocabIds.size()), static_cast<int8_t>(chains));
                    ++chains;
                }
            }
        }
        mDecoder->rollback(slot, accepted);
    }
    return chains;
}

int32_t JumpForwardCoordinator::acceptedRunLength(
    int32_t slot, int32_t const* ids, int32_t first, int32_t cutFrom, int32_t last)
{
    int32_t accepted = 0;
    for (int32_t node = first; node < last; ++node)
    {
        int32_t const outputId = mDecoder->toOutputSpace(ids[node]);
        if (outputId < 0 || mDecoder->acceptForced(slot, &outputId, 1) == 0)
        {
            break;
        }
        ++accepted;
    }
    mDecoder->rollback(slot, accepted);
    return std::max(accepted - (cutFrom - first), 0);
}

int32_t JumpForwardCoordinator::planDraftChain(
    DecodingInferenceContext& context, int32_t slot, HostDraftTree& tree, TreeWalkLimits const& limits)
{
    auto const row = static_cast<size_t>(slot);
    JumpForwardSlotStats& stats = context.jumpForwardStats[row];
    std::vector<int32_t> const& tokens = context.tokenIds[row];
    int32_t const draftCount = tree.count(slot);
    int32_t const width = tree.width();
    int32_t const* const ids = tree.ids(slot);
    int8_t const* const forced = tree.forced(slot);
    auto const tail = static_cast<int32_t>(std::min(tokens.size(), JumpForwardPlanner::kMaxWindowTokens + 1));
    int32_t const generatedTail = std::min(context.currentGenerateLengths[row], tail);
    mPath.clear();
    int32_t chains = 0;
    for (int32_t depth = 0; depth < draftCount; ++depth)
    {
        if (depth > 0)
        {
            int32_t const outputId = mDecoder->toOutputSpace(ids[depth]);
            if (outputId < 0 || mDecoder->acceptForced(slot, &outputId, 1) == 0)
            {
                break;
            }
            mPath.push_back(ids[depth]);
        }
        // Nothing follows the stop token. Deeper nodes leave a chain no row or no accepted position and, as in the
        // tree walk, are not counted.
        if (mDecoder->isTerminated(slot) || depth > limits.maxPathLength - 2 || depth > width - 2)
        {
            break;
        }
        if (forced[depth] != 0)
        {
            continue; // A chain runs through it; its continuation is that chain's.
        }
        std::string const forcedText = mDecoder->forcedContinuation(slot);
        if (forcedText.empty())
        {
            continue;
        }
        mScratch.assign(tokens.end() - tail, tokens.end());
        mScratch.insert(mScratch.end(), mPath.begin(), mPath.end());
        mPlanner.buildWindow(mScratch, generatedTail + depth, mWindow);
        int32_t const budget = std::min({limits.maxJumpTokens, limits.maxPathLength - 1 - depth,
            limits.remainingGeneration - depth, width - 1 - depth});
        JumpForwardPlan const plan = mPlanner.plan(slot, forcedText, mWindow, /*mutableTailCount=*/1, budget);
        if (plan.rollbackCount > 0)
        {
            ++stats.skips[static_cast<size_t>(JumpSkipReason::kRespellRefused)];
            continue;
        }
        if (plan.empty())
        {
            ++stats.skips[static_cast<size_t>(plan.skipReason)];
            continue;
        }
        auto const planned = static_cast<int32_t>(plan.fullVocabIds.size());
        int32_t reused = 0;
        while (reused < planned && depth + 1 + reused < draftCount
            && ids[depth + 1 + reused] == plan.fullVocabIds[static_cast<size_t>(reused)])
        {
            ++reused;
        }
        // Draft nodes that already spell the run stay. The first run that needs new tokens ends the walk: inserting
        // cuts the draft after the reused nodes, which costs the base the cut nodes the grammar would still accept,
        // so the run is grafted only when it brings at least that many tokens.
        bool const inserts = reused < planned;
        if (inserts)
        {
            int32_t const cutFrom = depth + 1 + reused;
            if (planned < acceptedRunLength(slot, ids, depth + 1, cutFrom, draftCount))
            {
                ++stats.skips[static_cast<size_t>(JumpSkipReason::kDraftLonger)];
                break;
            }
            tree.truncate(slot, cutFrom);
        }
        tree.insertChain(slot, depth, plan.fullVocabIds.data(), planned, static_cast<int8_t>(chains));
        ++chains;
        if (inserts)
        {
            break;
        }
    }
    mDecoder->rollback(slot, static_cast<int32_t>(mPath.size()));
    return chains;
}

void JumpForwardCoordinator::writeSyntheticLogprobRows(
    LogprobsSlot& logprobs, int32_t topK, int32_t const* tokens, int32_t count)
{
    for (int32_t k = 0; k < count; ++k)
    {
        writeSyntheticLogprobRow(logprobs, topK, logprobs.numSteps, tokens[k]);
        ++logprobs.numSteps;
    }
}

void JumpForwardCoordinator::writeSyntheticLogprobRow(LogprobsSlot& logprobs, int32_t topK, int32_t step, int32_t token)
{
    check::check(step >= 0, "A logprob row index is not negative");
    size_t const base = static_cast<size_t>(step) * static_cast<size_t>(topK);
    check::check(base + static_cast<size_t>(topK) <= logprobs.data.size(), "logprobs slot overflow");
    logprobs.data[base] = {token, 0.0F};
    std::fill(logprobs.data.begin() + static_cast<std::ptrdiff_t>(base + 1),
        logprobs.data.begin() + static_cast<std::ptrdiff_t>(base + static_cast<size_t>(topK)), kLOGPROB_FILLER);
}

SpecJumpGrafter::SpecJumpGrafter(tokenizer::Tokenizer const& tokenizer, GuidedDecoder& guidedDecoder,
    int32_t maxBatchSize, JumpGraftGeometry geometry, Limits const& limits)
    : mTokenizer(&tokenizer)
    , mGuidedDecoder(&guidedDecoder)
    , mGeometry(geometry)
    , mLimits(limits)
    , mHostTree(maxBatchSize, limits.jumpWidth)
{
    // Chain ids stay below jumpWidth, so they fit int8 and index account()'s consumed set.
    ELLM_CHECK(limits.width > 0 && limits.width <= limits.jumpWidth && limits.jumpWidth <= kMAX_SPEC_VERIFY_TREE_TOKENS
            && limits.pathCapacity > 1 && limits.maxJumpTokens > 0,
        "Jump-forward grafting limits are out of range");
    if (geometry == JumpGraftGeometry::kTree)
    {
        mMask = Tensor({maxBatchSize, limits.jumpWidth, limits.jumpWidth}, DeviceType::kGPU,
            trt_edgellm::DataType::kINT8, "SpecJumpGrafter::mask");
    }
    mForcedFlags = Tensor({maxBatchSize, limits.jumpWidth}, DeviceType::kGPU, trt_edgellm::DataType::kINT8,
        "SpecJumpGrafter::forcedFlags");
    mValidCounts
        = Tensor({maxBatchSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "SpecJumpGrafter::validCounts");
    mHostAcceptedRows = Tensor({maxBatchSize, limits.pathCapacity}, DeviceType::kCPU, trt_edgellm::DataType::kINT32,
        "SpecJumpGrafter::hostAcceptedRows");
    auto const slots = static_cast<size_t>(maxBatchSize);
    mGraftedChains.assign(slots, 0);
    mRootChains.assign(slots, 0);
    mAddedNodes.assign(slots, 0);
}

SpecJumpGrafter::Walk SpecJumpGrafter::walk(DecodingInferenceContext& context, SpecVerifyInputs const& drafted)
{
    mWide = false;
    if (!mCoordinator.has_value())
    {
        mCoordinator.emplace(*mTokenizer, *mGuidedDecoder, mLimits.maxJumpTokens);
    }
    int32_t const activeBatchSize = context.activeBatchSize;
    bool anySlot = false;
    for (int32_t i = 0; i < activeBatchSize && !anySlot; ++i)
    {
        anySlot = mCoordinator->slotMayJump(context, i);
    }
    if (!anySlot)
    {
        return Walk::kNoSlot;
    }
    NVTX_SCOPED_RANGE(nvtx_jump_walk, "JUMP_TREE_WALK", nvtx_colors::BLUE);
    bool const tree = mGeometry == JumpGraftGeometry::kTree;
    ELLM_CHECK(drafted.requestCount == activeBatchSize && drafted.tokensPerRequest == mLimits.width
            && (drafted.layout == SpecVerifyLayout::kTree) == tree && drafted.forcedFlags == nullptr,
        "The drafted verify input does not match the grafting geometry");
    mGuidedDecoder->captureDraft(drafted, context.stream);
    mGuidedDecoder->waitForDraftTopology();
    int32_t const* const draftTokens = mGuidedDecoder->hostDraftTokens();
    int32_t const* const draftParents = mGuidedDecoder->hostDraftParentIds();
    int32_t const* const draftCounts = mGuidedDecoder->hostDraftValidCounts();
    ELLM_CHECK(draftTokens != nullptr && (!tree || draftParents != nullptr), "The draft was not captured for the walk");
    int32_t totalChains = 0;
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        auto const row = static_cast<size_t>(i);
        std::ptrdiff_t const offset = static_cast<std::ptrdiff_t>(i) * mLimits.width;
        int32_t const draftCount = draftCounts != nullptr ? draftCounts[i] : mLimits.width;
        ELLM_CHECK(draftCount > 0 && draftCount <= mLimits.width, "A slot's draft node count is out of range");
        if (tree)
        {
            mHostTree.loadDraft(i, draftTokens + offset, draftParents + offset, draftCount);
        }
        else
        {
            mHostTree.loadChain(i, draftTokens + offset, draftCount);
        }
        mGraftedChains[row] = 0;
        mRootChains[row] = 0;
        mAddedNodes[row] = 0;
        if (!mCoordinator->slotMayJump(context, i))
        {
            continue;
        }
        JumpForwardCoordinator::TreeWalkLimits const limits{mLimits.pathCapacity, mLimits.maxJumpTokens,
            context.generateBudget(i) - context.currentGenerateLengths[row]};
        int32_t const chains = tree ? mCoordinator->planDraftTree(context, i, mHostTree, limits)
                                    : mCoordinator->planDraftChain(context, i, mHostTree, limits);
        mGraftedChains[row] = chains;
        mRootChains[row] = mHostTree.forcedChildOf(i, 0) >= 0 ? 1 : 0;
        mAddedNodes[row] = mHostTree.count(i) - draftCount;
        totalChains += chains;
    }
    mBatchSize = activeBatchSize;
    mWide = totalChains > 0;
    return mWide ? Walk::kGrafted : Walk::kNoChain;
}

void SpecJumpGrafter::upload(Tensor& ids, Tensor* parents, Tensor* depths, cudaStream_t stream)
{
    bool const tree = mGeometry == JumpGraftGeometry::kTree;
    ELLM_CHECK(mWide, "Only a walk that grafted has an input to upload");
    ELLM_CHECK(tree == (parents != nullptr) && tree == (depths != nullptr),
        "A tree uploads parents and depths, a chain neither");
    int32_t const width = mHostTree.width();
    check::check(ids.reshape({mBatchSize, width}) && mForcedFlags.reshape({mBatchSize, width})
            && mValidCounts.reshape({mBatchSize}),
        "Tensor reshape failed");
    size_t const nodes = static_cast<size_t>(mBatchSize) * static_cast<size_t>(width);
    // The next walk rewrites the pinned rows only after waiting for a capture the stream orders behind these copies.
    CUDA_CHECK(cudaMemcpyAsync(
        ids.rawPointer(), mHostTree.idsTensor().rawPointer(), nodes * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    if (tree)
    {
        check::check(mMask.reshape({mBatchSize, width, width}) && parents->reshape({mBatchSize, width})
                && depths->reshape({mBatchSize, width}),
            "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mMask.rawPointer(), mHostTree.maskTensor().rawPointer(),
            nodes * static_cast<size_t>(width), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(parents->rawPointer(), mHostTree.parentsTensor().rawPointer(),
            nodes * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(depths->rawPointer(), mHostTree.depthsTensor().rawPointer(), nodes * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(
        mForcedFlags.rawPointer(), mHostTree.forcedTensor().rawPointer(), nodes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mValidCounts.rawPointer(), mHostTree.validCountsTensor().rawPointer(),
        static_cast<size_t>(mBatchSize) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
}

bool SpecJumpGrafter::wide() const noexcept
{
    return mWide;
}

Tensor& SpecJumpGrafter::mask()
{
    ELLM_CHECK(mGeometry == JumpGraftGeometry::kTree, "Only tree grafting keeps a mask");
    return mMask;
}

Tensor const& SpecJumpGrafter::mask() const
{
    ELLM_CHECK(mGeometry == JumpGraftGeometry::kTree, "Only tree grafting keeps a mask");
    return mMask;
}

Tensor const* SpecJumpGrafter::forcedFlags() const noexcept
{
    return mWide ? &mForcedFlags : nullptr;
}

Tensor const* SpecJumpGrafter::validCounts() const noexcept
{
    return mWide ? &mValidCounts : nullptr;
}

decoder_utils::ForcedPathInfo SpecJumpGrafter::forcedPath(Tensor const& deviceAcceptedRows)
{
    ELLM_CHECK(mWide, "Only a grafting round has a forced path");
    return {&deviceAcceptedRows, &mHostAcceptedRows, mHostTree.forced(0), mHostTree.width()};
}

void SpecJumpGrafter::account(DecodingInferenceContext& context, int32_t const* hostAcceptLengths,
    int32_t const* hostAcceptedIds, int32_t acceptCapacity)
{
    ELLM_CHECK(mWide && context.activeBatchSize == mBatchSize, "Only the batch of a grafting round is accounted");
    decoder_utils::ForcedPathInfo const info{nullptr, &mHostAcceptedRows, mHostTree.forced(0), mHostTree.width()};
    int32_t const* const acceptedRows = mHostAcceptedRows.dataPointer<int32_t>();
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        auto const row = static_cast<size_t>(i);
        if (context.jumpForwardEnabled[row] == 0)
        {
            continue;
        }
        JumpForwardSlotStats& stats = context.jumpForwardStats[row];
        ++stats.chunkSteps;
        if (mGraftedChains[row] == 0)
        {
            continue;
        }
        int32_t const appended = hostAcceptLengths[i];
        int8_t const* const chainOf = mHostTree.chainOf(i);
        // A slot the round ended (failSlot) collected no logprob rows this round, so none are rewritten.
        LogprobsSlot* const logprobs = context.numLogprobsPerSlot[row] > 0 && context.finishedStates[row] == 0
            ? &context.stepLogprobs[row]
            : nullptr;
        std::array<bool, kMAX_SPEC_VERIFY_TREE_TOKENS> consumed{};
        int32_t forced = 0;
        int32_t chains = 0;
        for (int32_t k = 0; k < appended; ++k)
        {
            if (!decoder_utils::acceptedPositionIsForced(info, i, k, acceptCapacity))
            {
                continue;
            }
            ++forced;
            int32_t const chain = chainOf[acceptedRows[i * acceptCapacity + k + 1]];
            if (chain >= 0 && !consumed[static_cast<size_t>(chain)])
            {
                consumed[static_cast<size_t>(chain)] = true;
                ++chains;
            }
            if (logprobs != nullptr)
            {
                JumpForwardCoordinator::writeSyntheticLogprobRow(*logprobs, context.numLogprobsPerSlot[row],
                    logprobs->numSteps - appended + k, hostAcceptedIds[i * acceptCapacity + k]);
            }
        }
        stats.forcedTokens += forced;
        stats.jumps += chains;
        LOG_DEBUG("Request %d jump: grafted %d chains (%d at root, %d inserted nodes), accepted %d forced in %d chains",
            context.batchIndexMapping[row], mGraftedChains[row], mRootChains[row], mAddedNodes[row], forced, chains);
    }
}

HostDraftTree const& SpecJumpGrafter::hostTree() const noexcept
{
    return mHostTree;
}

void JumpGraftSession::initialize(tokenizer::Tokenizer const& tokenizer, GuidedDecoder& guidedDecoder,
    int32_t maxBatchSize, JumpGraftGeometry geometry, SpecDecodeConfig const& specConfig, int32_t plainWidth)
{
    if (specConfig.maxJumpTokens <= 0)
    {
        return;
    }
    mJumpWidth = specConfig.jumpVerifyTokenCount;
    mPathCapacity = specConfig.jumpMaxPathLength;
    mGrafter.emplace(tokenizer, guidedDecoder, maxBatchSize, geometry,
        SpecJumpGrafter::Limits{plainWidth, mJumpWidth, mPathCapacity, specConfig.maxJumpTokens});
}

bool JumpGraftSession::active(DecodingInferenceContext const& context) const noexcept
{
    return context.hasJumpForward && context.hasGuidedDecoding && mGrafter.has_value();
}

JumpGraftSession::Walk JumpGraftSession::walk(DecodingInferenceContext& context, SpecVerifyInputs const& drafted)
{
    Walk const walk = active(context) ? mGrafter->walk(context, drafted) : Walk::kNoSlot;
    mWide = walk == Walk::kGrafted;
    return walk;
}

void JumpGraftSession::prepareGraftedTreeDescriptors(
    PipelineIO& io, Tensor const& kvCacheLengths, int32_t activeBatchSize, cudaStream_t stream) const
{
    int32_t const jumpWidth = mJumpWidth;
    check::check(
        io.packedAttentionMask.reshape({activeBatchSize, jumpWidth, static_cast<int64_t>(divUp(jumpWidth, 32))}),
        "Tensor reshape failed");
    check::check(io.specDecodePositionIds.reshape({activeBatchSize, jumpWidth}), "Tensor reshape failed");
    check::check(io.selectTokenIndices.reshape({activeBatchSize, jumpWidth}), "Tensor reshape failed");
    check::check(io.contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    kernel::prepareSpecFullTreeInputs(mGrafter->mask(), kvCacheLengths, io.packedAttentionMask,
        io.specDecodePositionIds, io.selectTokenIndices, io.contextLengths, stream);
}

Tensor const* JumpGraftSession::validCounts(Tensor const* plain) const noexcept
{
    return mWide ? mGrafter->validCounts() : plain;
}

Tensor const* JumpGraftSession::forcedFlags() const noexcept
{
    return mWide ? mGrafter->forcedFlags() : nullptr;
}

std::optional<decoder_utils::ForcedPathInfo> JumpGraftSession::forcedPath(Tensor const& deviceAcceptedRows)
{
    if (!mWide)
    {
        return std::nullopt;
    }
    return mGrafter->forcedPath(deviceAcceptedRows);
}

void JumpGraftSession::account(DecodingInferenceContext& context, int32_t const* hostAcceptLengths,
    int32_t const* hostAcceptedIds, int32_t acceptCapacity)
{
    if (mWide)
    {
        mGrafter->account(context, hostAcceptLengths, hostAcceptedIds, acceptCapacity);
    }
}

void JumpGraftSession::addGraphTiers(std::vector<int32_t>& verifyWidths, std::vector<int32_t>* acceptLengths) const
{
    if (!mGrafter.has_value())
    {
        return;
    }
    if (mJumpWidth > verifyWidths.front())
    {
        verifyWidths.push_back(mJumpWidth);
    }
    if (acceptLengths != nullptr && mPathCapacity > acceptLengths->back())
    {
        acceptLengths->push_back(mPathCapacity);
    }
}

} // namespace rt
} // namespace trt_edgellm
