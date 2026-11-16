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

#include "runtime/decoding/guidedDecoder.h"
#include "profiling/nvtx_wrapper.h"
#include "runtime/decoding/specDecodeUtils.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/inputLimits.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "runtime/state/decodingInferenceContext.h"
#include "sampler/sampling.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <nlohmann/json.hpp>
#include <numeric>

#include <xgrammar/xgrammar.h>

namespace trt_edgellm
{
namespace rt
{

bool hasGuidedDecoding(LLMGenerationRequest const& request) noexcept
{
    return std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& slotRequest) { return slotRequest.guidedDecoding.has_value(); });
}

namespace
{

//! The two conditions are independent: a request can finish while its grammar is mid-rule,
//! and a thinking model must be free to open `<think>` before the schema takes over.
void buildSlotSuppression(
    DecodingInferenceContext const& context, int32_t activeBatchSize, std::vector<int8_t>& suppressed)
{
    suppressed.assign(static_cast<size_t>(activeBatchSize), 0);
    for (int32_t slot = 0; slot < activeBatchSize; ++slot)
    {
        bool const finished
            = slot < static_cast<int32_t>(context.finishedStates.size()) && context.finishedStates[slot] != 0;
        bool const stillThinking = slot < static_cast<int32_t>(context.guidedReasoningEnded.size())
            && context.guidedReasoningEnded[slot] == 0;
        suppressed[static_cast<size_t>(slot)] = (finished || stillThinking) ? 1 : 0;
    }
}

//! Fail one slot but keep what it already produced: streaming has already handed those tokens
//! to the client, so the non-streaming path must agree.
void failSlot(DecodingInferenceContext& context, int32_t slot, char const* what)
{
    if (context.finishedStates[slot] != 0)
    {
        return; // First writer wins; a cancel may already have landed.
    }
    context.finishedStates[slot] = 1;
    context.slotStreams[slot].terminalReason = FinishReason::kError;
    LOG_ERROR("Request %d: %s; ending the request and keeping the %d tokens generated so far.", slot, what,
        context.currentGenerateLengths[slot]);
}

void finishCompletedJson(GuidedDecoder& decoder, DecodingInferenceContext& context, int32_t slot)
{
    if (context.finishedStates[slot] != 0 || !decoder.hasCompletedJson(slot))
    {
        return;
    }
    context.finishedStates[slot] = 1;
    context.slotStreams[slot].terminalReason = FinishReason::kEndId;
    LOG_DEBUG("Request %d finished after completing its JSON guide.", slot);
}

} // namespace

std::vector<int32_t> reasoningStartMarkers(Tokenizer const& tokenizer)
{
    return {static_cast<int32_t>(tokenizer.getTokenId("<|channel>")),
        static_cast<int32_t>(tokenizer.getTokenId("<think>"))};
}

std::vector<int32_t> reasoningEndMarkers(Tokenizer const& tokenizer)
{
    return {static_cast<int32_t>(tokenizer.getTokenId("<channel|>")),
        static_cast<int32_t>(tokenizer.getTokenId("</think>"))};
}

bool reasoningClosedInPrompt(std::vector<int32_t> const& promptTokens, std::vector<int32_t> const& startMarkers,
    std::vector<int32_t> const& endMarkers, bool thinkingEnabled) noexcept
{
    auto const present = [](std::vector<int32_t> const& markers) {
        return std::any_of(markers.begin(), markers.end(), [](int32_t id) { return id >= 0; });
    };
    if (!present(startMarkers) && !present(endMarkers))
    {
        return true;
    }

    auto const matches = [](std::vector<int32_t> const& markers, int32_t token) {
        return token >= 0 && std::find(markers.begin(), markers.end(), token) != markers.end();
    };
    for (auto it = promptTokens.rbegin(); it != promptTokens.rend(); ++it)
    {
        if (matches(endMarkers, *it))
        {
            return true;
        }
        if (matches(startMarkers, *it))
        {
            return false;
        }
    }
    // No marker at all: with thinking disabled the grammar starts immediately. With thinking
    // enabled, the model may still emit the opening marker even when the template does not.
    return !thinkingEnabled;
}

void applyGuidedDecodingMask(GuidedDecoder& decoder, DecodingInferenceContext& context, Tensor& logits,
    int32_t activeBatchSize, int32_t rowsPerSlot, cudaStream_t stream)
{
    if (!context.hasGuidedDecoding || activeBatchSize <= 0)
    {
        return;
    }

    buildSlotSuppression(context, activeBatchSize, context.guidedMaskSuppressedPerSlot);
    decoder.fillMasks(
        activeBatchSize, rowsPerSlot, context.guidedMaskSuppressedPerSlot, context.guidedUnsatisfiableSlots, stream);

    for (auto const slot : context.guidedUnsatisfiableSlots)
    {
        failSlot(context, slot, "the grammar admits no token in this engine's output vocabulary");
    }

    decoder.applyMask(logits, activeBatchSize, rowsPerSlot, stream);
}

void advanceGuidedDecoding(
    GuidedDecoder& decoder, DecodingInferenceContext& context, int32_t const* outputSpaceIds, int32_t activeBatchSize)
{
    if (!context.hasGuidedDecoding)
    {
        return;
    }

    // Deliberately not routed through advanceCommitted: this path carries *output*-space IDs
    // (captured before the reduced-vocabulary remap) while that one takes full-space IDs. The
    // reasoning flag is latched for this path by the runtime's updateThinkingDone, which is
    // correct as long as a step commits exactly one token -- which is what vanilla decode means.
    buildSlotSuppression(context, activeBatchSize, context.guidedMaskSuppressedPerSlot);
    for (int32_t slot = 0; slot < activeBatchSize; ++slot)
    {
        if (!decoder.hasGrammar(slot) || context.guidedMaskSuppressedPerSlot[static_cast<size_t>(slot)] != 0)
        {
            continue;
        }
        // EOS included: the matcher only terminates by accepting the stop token.
        if (!decoder.advance(slot, outputSpaceIds[slot]))
        {
            // The mask should have made this impossible.
            failSlot(context, slot, "the grammar rejected a token the mask should have forbidden");
        }
        else
        {
            finishCompletedJson(decoder, context, slot);
        }
    }
}

void applyGuidedDecodingMaskForDraftTree(GuidedDecoder& decoder, DecodingInferenceContext& context, Tensor& logits,
    int32_t activeBatchSize, int32_t rowsPerSlot, cudaStream_t stream)
{
    if (!context.hasGuidedDecoding || activeBatchSize <= 0)
    {
        return;
    }
    // Waiting here, rather than next to the copy, is the point: the verify forward is already
    // enqueued, so the grammar walk below runs while the GPU is busy with it. A long wait is
    // normal and healthy -- it spans the drafting forwards. What matters on a profile is whether
    // the walk that follows stays inside the verify forward's span; if it sticks out, the GPU is
    // waiting on us and the overlap is gone. Hence two ranges rather than one.
    {
        NVTX_SCOPED_RANGE(nvtxWait, "GUIDED_WAIT_DRAFT_TREE", nvtx_colors::PALE_ORANGE);
        decoder.waitForDraftTopology();
    }
    int32_t const* const draftTokens = decoder.hostDraftTokens();
    if (draftTokens == nullptr)
    {
        return;
    }

    NVTX_SCOPED_RANGE(nvtxFill,
        ("GUIDED_FILL_MASK[R" + std::to_string(context.generationRound) + "," + std::to_string(activeBatchSize) + "]")
            .c_str(),
        nvtx_colors::SKY_BLUE);

    // Only the finished gate is slot-level here. Whether the reasoning block is still open is a
    // per-path property once a branch can cross the closing marker, so the walk decides it.
    context.guidedMaskSuppressedPerSlot.assign(static_cast<size_t>(activeBatchSize), 0);
    for (int32_t slot = 0; slot < activeBatchSize; ++slot)
    {
        bool const finished
            = slot < static_cast<int32_t>(context.finishedStates.size()) && context.finishedStates[slot] != 0;
        context.guidedMaskSuppressedPerSlot[static_cast<size_t>(slot)] = finished ? 1 : 0;
    }

    decoder.fillMasksForDraftTree(activeBatchSize, rowsPerSlot, draftTokens, decoder.hostDraftParentIds(),
        decoder.hostDraftValidCounts(), decoder.hostDraftForcedFlags(), context.guidedMaskSuppressedPerSlot,
        context.guidedReasoningEnded, context.guidedUnsatisfiableSlots, stream);

    for (auto const slot : context.guidedUnsatisfiableSlots)
    {
        failSlot(context, slot, "the grammar admits no token in this engine's output vocabulary");
    }

    decoder.applyMask(logits, activeBatchSize, rowsPerSlot, stream);
}

void advanceGuidedDecodingForCommitted(GuidedDecoder& decoder, DecodingInferenceContext& context,
    int32_t const* hostAcceptedTokenIds, int32_t const* hostAcceptLengths, int32_t maxAcceptDepth,
    int32_t activeBatchSize)
{
    if (!context.hasGuidedDecoding)
    {
        return;
    }
    for (int32_t slot = 0; slot < activeBatchSize; ++slot)
    {
        if (!decoder.hasGrammar(slot) || context.finishedStates[slot] != 0)
        {
            continue;
        }
        int32_t const count = hostAcceptLengths[slot];
        // advanceCommitted latches the reasoning flag off the separator, so the grammar and the
        // runtime's thinking bookkeeping cannot drift apart.
        if (!decoder.advanceCommitted(slot, hostAcceptedTokenIds + static_cast<size_t>(slot) * maxAcceptDepth, count,
                context.guidedReasoningEnded[static_cast<size_t>(slot)]))
        {
            failSlot(context, slot, "the grammar rejected a token the mask should have forbidden");
        }
        else
        {
            finishCompletedJson(decoder, context, slot);
        }
    }
}

namespace
{

/*!
 * @brief JSON Schema keywords XGrammar accepts but does not implement correctly.
 *
 * Measured on v0.2.1 by feeding violating documents to the compiled grammar and requiring
 * both full acceptance and rule completion. multipleOf / uniqueItems / contains are ignored
 * outright; minContains / maxContains degrade to a bare element count, so a schema asking for
 * two nines accepts [1, 2]. patternProperties compiles into a grammar that rejects everything.
 *
 * The list tracks measurement, not vLLM's list: propertyNames was broken in v0.1.25 and is
 * correct in v0.2.1, so it is deliberately absent. Re-measure when the pin moves.
 */
constexpr char const* kUnsupportedSchemaKeywords[] = {
    "multipleOf",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "patternProperties",
};

//! `format` values XGrammar converts to a pattern. Anything else compiles and is then ignored.
constexpr char const* kSupportedSchemaFormats[] = {
    "date",
    "date-time",
    "duration",
    "email",
    "hostname",
    "ipv4",
    "ipv6",
    "json-pointer",
    "relative-json-pointer",
    "time",
    "uri",
    "uri-reference",
    "uri-template",
    "uuid",
};

//! Keywords whose sub-object is keyed by user-chosen names rather than by schema keywords.
//! Without this, a schema with a property literally named "contains" would be rejected.
constexpr char const* kNameKeyedKeywords[] = {
    "properties",
    "$defs",
    "definitions",
    "dependentSchemas",
    "dependencies",
};

//! Keys off the keyword alone rather than a sibling "type": {"multipleOf": 5} carries no type
//! and would slip past a type-keyed check.
bool findUnsupportedSchemaFeature(nlohmann::json const& node, std::string& feature)
{
    if (node.is_array())
    {
        for (auto const& element : node)
        {
            if (findUnsupportedSchemaFeature(element, feature))
            {
                return true;
            }
        }
        return false;
    }
    if (!node.is_object())
    {
        return false;
    }

    for (auto const* keyword : kUnsupportedSchemaKeywords)
    {
        if (node.contains(keyword))
        {
            feature = keyword;
            return true;
        }
    }
    auto const formatIt = node.find("format");
    if (formatIt != node.end() && formatIt->is_string())
    {
        auto const formatValue = formatIt->get<std::string>();
        bool const supported = std::any_of(std::begin(kSupportedSchemaFormats), std::end(kSupportedSchemaFormats),
            [&formatValue](char const* candidate) { return formatValue == candidate; });
        if (!supported)
        {
            feature = "format: \"" + formatValue + "\"";
            return true;
        }
    }

    for (auto const& item : node.items())
    {
        bool const nameKeyed = item.value().is_object()
            && std::any_of(std::begin(kNameKeyedKeywords), std::end(kNameKeyedKeywords),
                [&item](char const* candidate) { return item.key() == candidate; });
        if (nameKeyed)
        {
            // One level down the keys are names, so only their subschemas are inspected.
            for (auto const& named : item.value().items())
            {
                if (findUnsupportedSchemaFeature(named.value(), feature))
                {
                    return true;
                }
            }
            continue;
        }
        if (findUnsupportedSchemaFeature(item.value(), feature))
        {
            return true;
        }
    }
    return false;
}

/*!
 * @brief Parse a kChoice guide into its alternatives.
 *
 * @param[out] failReason Set only when the call returns false
 */
bool parseChoiceList(std::string const& guide, std::vector<std::string>& choices, std::string& failReason)
{
    nlohmann::json document;
    try
    {
        document = nlohmann::json::parse(guide);
    }
    catch (std::exception const& e)
    {
        failReason = format::fmtstr("guided_decoding.choice is not valid JSON: %s", e.what());
        return false;
    }
    bool const wellFormed = document.is_array() && !document.empty()
        && std::all_of(document.begin(), document.end(), [](nlohmann::json const& item) { return item.is_string(); });
    if (!wellFormed)
    {
        failReason = "guided_decoding.choice must be a non-empty array of strings";
        return false;
    }
    for (auto const& item : document)
    {
        choices.push_back(item.get<std::string>());
    }
    return true;
}

} // namespace

bool validateGuidedDecodingParams(GuidedDecodingParams const& params, std::string& failReason)
{
    if (params.type != GuideType::kJsonObject && params.guide.empty())
    {
        failReason = std::string("guided_decoding.") + guideTypeName(params.type) + " must not be empty";
        return false;
    }
    if (params.guide.size() > limits::security::kMaxGuidedDecodingGuideBytes)
    {
        failReason = format::fmtstr("guided_decoding.%s is too large: %zu bytes (max: %zu). Limit defined in %s.",
            guideTypeName(params.type), params.guide.size(), limits::security::kMaxGuidedDecodingGuideBytes,
            limits::kInputLimitsLocation);
        return false;
    }

    if (params.type == GuideType::kChoice)
    {
        std::vector<std::string> choices;
        return parseChoiceList(params.guide, choices, failReason);
    }

    if (params.type == GuideType::kJsonSchema || params.type == GuideType::kStructuralTag)
    {
        nlohmann::json document;
        try
        {
            document = nlohmann::json::parse(params.guide);
        }
        catch (std::exception const& e)
        {
            failReason
                = format::fmtstr("guided_decoding.%s is not valid JSON: %s", guideTypeName(params.type), e.what());
            return false;
        }

        if (params.type == GuideType::kJsonSchema)
        {
            std::string feature;
            if (findUnsupportedSchemaFeature(document, feature))
            {
                failReason = format::fmtstr(
                    "guided_decoding.json_schema uses '%s', which XGrammar accepts but does not enforce; "
                    "the generated output would not be guaranteed to match the schema",
                    feature.c_str());
                return false;
            }
        }
    }
    return true;
}

namespace
{

//! Threads XGrammar may use to compile one grammar.
constexpr int kCOMPILER_THREADS = 8;

//! Compiled-grammar LRU budget; XGrammar leaves it unlimited by default.
constexpr int64_t kCACHE_LIMIT_BYTES = 64LL * 1024 * 1024;

//! Lower the alternatives onto XGrammar's EBNF. JSON string escaping is a subset of the
//! EBNF literal escaping, so a `dump()` of each choice is already a valid literal.
std::string choiceListToEbnf(std::vector<std::string> const& choices)
{
    std::string grammar = "root ::= ";
    for (size_t i = 0; i < choices.size(); ++i)
    {
        if (i > 0)
        {
            grammar += " | ";
        }
        grammar += nlohmann::json(choices[i]).dump();
    }
    return grammar;
}

} // namespace

struct GuidedDecoder::Impl
{
    Tokenizer const* tokenizer{nullptr};
    int32_t maxBatchSize{0};
    //! Bitmask rows allocated up front = maxBatchSize * maxRowsPerSlot.
    int32_t maxRows{0};
    int32_t outputVocabSize{0};
    int32_t bitmaskSize{0}; //!< int32 words per row, i.e. xgrammar::GetBitmaskSize(outputVocabSize)

    //! Output-space index -> full tokenizer ID, [outputVocabSize]. Unset when the engine does
    //! not prune, in which case the mapping is the identity.
    Tensor outputToFullVocab;
    bool hasReducedVocab{false};

    //! Full tokenizer ID -> output-space index, or -1 for a token this engine cannot emit.
    //! Only built when the engine prunes; the speculative path needs this direction because the
    //! verify tree carries full-vocabulary IDs and the matchers live in output space.
    int32_t fullVocabSize{0};
    std::vector<int32_t> fullToOutput;

    //! Reasoning-end markers in the full vocabulary; -1 entries never match.
    std::vector<int32_t> reasoningEnd;

    //! One warning per request when a draft token has no output-space image, which means the
    //! engine's reduced vocabulary is not a superset of the draft's. Silently dropping every
    //! draft node instead would look like an unexplained acceptance-rate collapse.
    std::vector<int8_t> warnedUnmappable;

    //! Full ID -> output-space index. Identity, plus a bounds check, when the engine does not
    //! prune; the caller therefore has a single code path.
    int32_t toOutputSpace(int32_t fullId) const
    {
        if (fullId < 0 || fullId >= fullVocabSize)
        {
            return -1;
        }
        return hasReducedVocab ? fullToOutput[static_cast<size_t>(fullId)] : fullId;
    }

    bool isReasoningEnd(int32_t fullId) const
    {
        return fullId >= 0 && std::find(reasoningEnd.begin(), reasoningEnd.end(), fullId) != reasoningEnd.end();
    }

    //! True when the row's mask admits at least one token.
    bool rowHasAnyToken(int32_t rowId) const
    {
        int32_t const* const row = hostBitmask.dataPointer<int32_t>() + static_cast<size_t>(rowId) * bitmaskSize;
        int32_t acc = 0;
        for (int32_t w = 0; w < bitmaskSize; ++w)
        {
            acc |= row[w];
        }
        return acc != 0;
    }

    bool rowAllows(int32_t rowId, int32_t outputToken) const
    {
        int32_t const* const row = hostBitmask.dataPointer<int32_t>() + static_cast<size_t>(rowId) * bitmaskSize;
        return ((static_cast<uint32_t>(row[outputToken >> 5]) >> (outputToken & 31)) & 1U) != 0U;
    }

    //! Built on first use: one idToPiece call per output-vocabulary entry is too
    //! expensive to pay for runs that never use guided decoding.
    std::optional<xgrammar::TokenizerInfo> tokenizerInfo;
    std::optional<xgrammar::GrammarCompiler> compiler;
    struct SlotGrammar
    {
        xgrammar::GrammarMatcher matcher;
        bool finishOnCompletion; //!< JSON object or schema: the response ends once the value is closed
    };
    std::vector<std::optional<SlotGrammar>> slots;
    std::vector<std::optional<SlotGrammar>> compactedSlots; //!< compactSlots scratch, sized in initialize
    std::vector<int32_t> stopTokenIds;                      //!< Output space; set with tokenizerInfo
    //! One mask row for the completion check, apart from hostBitmask rows staged for upload.
    std::vector<int32_t> completionBitmask;

    Tensor deviceBitmask;      //!< [maxRows, bitmaskSize] the mask the apply kernel reads
    Tensor deviceRowNeedsMask; //!< [maxRows] per-row flag; the kernel leaves a zeroed row alone
    Tensor hostBitmask;        //!< [maxRows, bitmaskSize] where XGrammar fills; only flags are cleared per step
    Tensor hostRowNeedsMask;   //!< [maxRows] staging for the flags above

    //! Speculative decoding only, allocated when a slot owns more than one verify row. Holds this
    //! step's draft geometry: the matchers live on the host, so the walk waits for this copy, and
    //! the event is blocking-sync because that wait spans the drafting forwards.
    Tensor hostDraftTokens;              //!< [batch, rowsPerSlot] node tokens, full vocabulary
    Tensor hostDraftParentIds;           //!< Tree only: [batch, rowsPerSlot] parent index; -1 at root/padding
    Tensor hostDraftValidCounts;         //!< [batch] nodes actually built; later rows are padding
    Tensor hostDraftForcedFlags;         //!< [batch, rowsPerSlot] 1 on a grafted forced node
    bool draftIsTree{false};             //!< Whether this step's draft geometry is a tree
    bool draftHasValidCounts{false};     //!< Whether this step's draft reports a node count at all
    bool draftHasForcedFlags{false};     //!< Whether this step's draft carries grafted forced nodes
    cudaEvent_t draftCopyReady{nullptr}; //!< Signals that the copy has landed
    bool draftCopyPending{false};        //!< A copy is in flight; guards against a stale event

    std::vector<int32_t> firstChild;  //!< [rowsPerSlot] lowest-numbered child, or -1 for a leaf
    std::vector<int32_t> nextSibling; //!< [rowsPerSlot] next child of the same parent, or -1

    //! [outputVocabSize] 1 when another output token has this token's piece as a strict prefix.
    //! Built by prepareJumpForward(), or on first use.
    std::vector<int8_t> extendable;

    //! One idToPiece per output-vocabulary entry, in output-space order.
    std::vector<std::string> outputVocabPieces() const;
    void ensureCompiler();
    void buildExtendableTable(std::vector<std::string> const& pieces);
    void ensureExtendableTable();
    xgrammar::CompiledGrammar compile(GuidedDecodingParams const& params);
};

std::vector<std::string> GuidedDecoder::Impl::outputVocabPieces() const
{
    ELLM_CHECK(tokenizer != nullptr, "GuidedDecoder used before initialize()");
    // Null when the engine does not prune; output space is then the identity.
    int32_t const* const outputToFull = outputToFullVocab.dataPointer<int32_t>();
    std::vector<std::string> pieces(static_cast<size_t>(outputVocabSize));
    for (int32_t outputId = 0; outputId < outputVocabSize; ++outputId)
    {
        int32_t const fullId = hasReducedVocab ? outputToFull[outputId] : outputId;
        pieces[static_cast<size_t>(outputId)] = tokenizer->idToPiece(fullId, /*skipSpecialTokens=*/false);
    }
    return pieces;
}

void GuidedDecoder::Impl::ensureExtendableTable()
{
    if (extendable.empty())
    {
        buildExtendableTable(outputVocabPieces());
    }
}

void GuidedDecoder::Impl::buildExtendableTable(std::vector<std::string> const& pieces)
{
    std::vector<int32_t> order(pieces.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&pieces](int32_t lhs, int32_t rhs) {
        return pieces[static_cast<size_t>(lhs)] < pieces[static_cast<size_t>(rhs)];
    });

    // In sorted order every string that starts with a piece follows it directly, after any
    // duplicates of the piece itself.
    extendable.assign(pieces.size(), 0);
    for (size_t i = 0; i < order.size();)
    {
        std::string const& piece = pieces[static_cast<size_t>(order[i])];
        size_t next = i + 1;
        while (next < order.size() && pieces[static_cast<size_t>(order[next])] == piece)
        {
            ++next;
        }
        bool const extended
            = next < order.size() && pieces[static_cast<size_t>(order[next])].compare(0, piece.size(), piece) == 0;
        for (size_t j = i; j < next; ++j)
        {
            extendable[static_cast<size_t>(order[j])] = extended ? 1 : 0;
        }
        i = next;
    }
}

void GuidedDecoder::Impl::ensureCompiler()
{
    if (compiler.has_value())
    {
        return;
    }
    auto const start = std::chrono::steady_clock::now();

    std::vector<std::string> const encodedVocab = outputVocabPieces();
    int32_t const* const outputToFull = outputToFullVocab.dataPointer<int32_t>();

    // Must come from the same source the runtime uses to detect EOS; XGrammar's own
    // detection guesses by name and gets it wrong.
    stopTokenIds.clear();
    for (auto const fullEosId : tokenizer->getEosIds())
    {
        int32_t outputId = static_cast<int32_t>(fullEosId);
        if (hasReducedVocab)
        {
            int32_t const* const end = outputToFull + outputVocabSize;
            auto const* const it = std::find(outputToFull, end, static_cast<int32_t>(fullEosId));
            if (it == end)
            {
                // A stop token pruned from the reduced vocabulary can never be generated,
                // so it is safe to drop from the grammar's stop set (models may declare
                // several EOS ids, e.g. via generation_config.json). At least one EOS must
                // survive — enforced below.
                LOG_WARNING(
                    "Guided decoding: EOS token %d is not in the engine's reduced vocabulary; skipping it in the "
                    "grammar stop set.",
                    static_cast<int32_t>(fullEosId));
                continue;
            }
            outputId = static_cast<int32_t>(it - outputToFull);
        }
        ELLM_CHECK(outputId >= 0 && outputId < outputVocabSize, "EOS token falls outside the output vocabulary");
        stopTokenIds.push_back(outputId);
    }
    ELLM_CHECK(!stopTokenIds.empty(),
        "Guided decoding requires at least one EOS token inside the output vocabulary; the engine's vocab_map is "
        "inconsistent with its tokenizer");

    // RAW: our tokenizer already resolves byte-level and byte-fallback pieces to raw bytes at
    // load time and discards the display form, so any other vocab type would decode twice.
    tokenizerInfo.emplace(encodedVocab, xgrammar::VocabType::RAW, outputVocabSize, stopTokenIds,
        /*add_prefix_space=*/false);

    compiler.emplace(*tokenizerInfo, kCOMPILER_THREADS, /*cache_enabled=*/true, kCACHE_LIMIT_BYTES);

    auto const elapsedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    LOG_INFO("Guided decoding: built tokenizer info over %d output-vocab entries in %.1f ms (cache limit %lld bytes)",
        outputVocabSize, elapsedMs, static_cast<long long>(kCACHE_LIMIT_BYTES));
}

xgrammar::CompiledGrammar GuidedDecoder::Impl::compile(GuidedDecodingParams const& params)
{
    // Pinned rather than left to defaults: any_whitespace keeps readable JSON legal, strict_mode forbids
    // unmentioned properties, and max_whitespace_cnt stops greedy decoding looping on whitespace; it caps
    // each whitespace run, newline and indentation together, so it must exceed the deepest indentation.
    constexpr bool kANY_WHITESPACE = true;
    constexpr bool kSTRICT_MODE = true;
    constexpr int kMAX_WHITESPACE_COUNT = 64;

    switch (params.type)
    {
    case GuideType::kJsonObject:
        // Not CompileBuiltinJSONGrammar(): that admits any JSON value, not just an object.
        return compiler->CompileJSONSchema(
            R"({"type":"object"})", kANY_WHITESPACE, std::nullopt, std::nullopt, kSTRICT_MODE, kMAX_WHITESPACE_COUNT);
    case GuideType::kJsonSchema:
        return compiler->CompileJSONSchema(
            params.guide, kANY_WHITESPACE, std::nullopt, std::nullopt, kSTRICT_MODE, kMAX_WHITESPACE_COUNT);
    case GuideType::kRegex: return compiler->CompileRegex(params.guide);
    case GuideType::kEbnf: return compiler->CompileGrammar(params.guide, "root");
    case GuideType::kStructuralTag: return compiler->CompileStructuralTag(params.guide);
    case GuideType::kChoice:
    {
        // Validated at request admission, so the throw only guards a caller that skipped it.
        std::vector<std::string> choices;
        std::string failReason;
        if (!parseChoiceList(params.guide, choices, failReason))
        {
            throw std::runtime_error(failReason);
        }
        return compiler->CompileGrammar(choiceListToEbnf(choices), "root");
    }
    }
    throw std::runtime_error("Unhandled guided decoding guide type");
}

GuidedDecoder::GuidedDecoder()
    : mImpl(std::make_unique<Impl>())
{
}

GuidedDecoder::~GuidedDecoder()
{
    if (mImpl->draftCopyReady != nullptr)
    {
        cudaEventDestroy(mImpl->draftCopyReady);
    }
}

void GuidedDecoder::initialize(int32_t maxBatchSize, int32_t maxRowsPerSlot, int32_t outputVocabSize,
    int32_t fullVocabSize, Tokenizer const* tokenizer, Tensor const& reducedToFullVocabMap, cudaStream_t stream)
{
    ELLM_CHECK(maxBatchSize > 0, "GuidedDecoder requires a positive max batch size");
    ELLM_CHECK(maxRowsPerSlot > 0, "GuidedDecoder requires a positive max rows per slot");
    ELLM_CHECK(fullVocabSize >= outputVocabSize, "Full vocabulary is smaller than the engine output vocabulary");
    ELLM_CHECK(outputVocabSize > 0, "GuidedDecoder requires a positive output vocabulary size");
    ELLM_CHECK(tokenizer != nullptr, "GuidedDecoder requires a tokenizer");

    mImpl->tokenizer = tokenizer;
    mImpl->maxBatchSize = maxBatchSize;
    mImpl->outputVocabSize = outputVocabSize;
    mImpl->bitmaskSize = xgrammar::GetBitmaskSize(outputVocabSize);
    mImpl->completionBitmask.assign(static_cast<size_t>(mImpl->bitmaskSize), 0);
    mImpl->fullVocabSize = fullVocabSize;
    mImpl->slots.assign(static_cast<size_t>(maxBatchSize), std::nullopt);
    mImpl->compactedSlots.assign(static_cast<size_t>(maxBatchSize), std::nullopt);
    mImpl->warnedUnmappable.assign(static_cast<size_t>(maxBatchSize), 0);
    mImpl->reasoningEnd = reasoningEndMarkers(*tokenizer);

    mImpl->hasReducedVocab = false;
    if (!reducedToFullVocabMap.isEmpty())
    {
        auto const entries = static_cast<size_t>(reducedToFullVocabMap.getShape().volume());
        ELLM_CHECK(entries >= static_cast<size_t>(outputVocabSize),
            "Reduced vocabulary map is smaller than the engine output vocabulary");
        mImpl->outputToFullVocab = Tensor(
            {outputVocabSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "GuidedDecoder::outputToFullVocab");
        CUDA_CHECK(cudaMemcpyAsync(mImpl->outputToFullVocab.rawPointer(), reducedToFullVocabMap.rawPointer(),
            static_cast<size_t>(outputVocabSize) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        mImpl->hasReducedVocab = true;

        // Reverse of the map above. The verify tree carries full-vocabulary IDs and the
        // matchers live in output space, and unlike vanilla decode there is no point in the
        // speculative path where the runtime still holds the pre-remap index.
        int32_t const* const outputToFull = mImpl->outputToFullVocab.dataPointer<int32_t>();
        mImpl->fullToOutput.assign(static_cast<size_t>(fullVocabSize), -1);
        for (int32_t outputId = 0; outputId < outputVocabSize; ++outputId)
        {
            int32_t const fullId = outputToFull[outputId];
            if (fullId >= 0 && fullId < fullVocabSize)
            {
                mImpl->fullToOutput[static_cast<size_t>(fullId)] = outputId;
            }
        }
    }

    // Vanilla decode emits one logits row per slot; speculative verification emits one per
    // verify node, so the buffers are sized by the deployment's verify size.
    int32_t const maxRows = maxBatchSize * maxRowsPerSlot;
    mImpl->maxRows = maxRows;
    mImpl->deviceBitmask = Tensor(
        {maxRows, mImpl->bitmaskSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "GuidedDecoder::bitmask");
    mImpl->deviceRowNeedsMask
        = Tensor({maxRows}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "GuidedDecoder::rowNeedsMask");
    mImpl->hostBitmask = Tensor(
        {maxRows, mImpl->bitmaskSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "GuidedDecoder::hostBitmask");
    mImpl->hostRowNeedsMask
        = Tensor({maxRows}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "GuidedDecoder::hostRowNeedsMask");
    std::memset(mImpl->hostBitmask.rawPointer(), 0,
        static_cast<size_t>(maxRows) * static_cast<size_t>(mImpl->bitmaskSize) * sizeof(int32_t));
    std::memset(mImpl->hostRowNeedsMask.rawPointer(), 0, static_cast<size_t>(maxRows) * sizeof(int32_t));

    if (maxRowsPerSlot > 1)
    {
        mImpl->hostDraftTokens = Tensor({maxBatchSize, maxRowsPerSlot}, DeviceType::kCPU, trt_edgellm::DataType::kINT32,
            "GuidedDecoder::hostDraftTokens");
        mImpl->hostDraftParentIds = Tensor({maxBatchSize, maxRowsPerSlot}, DeviceType::kCPU,
            trt_edgellm::DataType::kINT32, "GuidedDecoder::hostDraftParentIds");
        mImpl->hostDraftValidCounts = Tensor(
            {maxBatchSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "GuidedDecoder::hostDraftValidCounts");
        mImpl->hostDraftForcedFlags = Tensor({maxBatchSize, maxRowsPerSlot}, DeviceType::kCPU,
            trt_edgellm::DataType::kINT8, "GuidedDecoder::hostDraftForcedFlags");
        mImpl->firstChild.resize(static_cast<size_t>(maxRowsPerSlot));
        mImpl->nextSibling.resize(static_cast<size_t>(maxRowsPerSlot));
        CUDA_CHECK(cudaEventCreateWithFlags(&mImpl->draftCopyReady, cudaEventDisableTiming | cudaEventBlockingSync));
    }

    LOG_INFO(
        "GuidedDecoder initialized: maxBatchSize=%d maxRowsPerSlot=%d outputVocabSize=%d bitmaskSize=%d "
        "reducedVocab=%s bitmaskBuffers=%.2f MB",
        maxBatchSize, maxRowsPerSlot, outputVocabSize, mImpl->bitmaskSize, mImpl->hasReducedVocab ? "yes" : "no",
        2.0 * static_cast<double>(maxRows) * mImpl->bitmaskSize * sizeof(int32_t) / (1024.0 * 1024.0));
}

bool GuidedDecoder::prepareSlot(int32_t slot, GuidedDecodingParams const& params, std::string& failReason)
{
    ELLM_CHECK(
        slot >= 0 && slot < static_cast<int32_t>(mImpl->slots.size()), "GuidedDecoder::prepareSlot slot out of range");

    // The only input-driven step in this class, so the only one that catches: a malformed
    // schema must fail this slot, not the process.
    try
    {
        mImpl->ensureCompiler();

        auto const start = std::chrono::steady_clock::now();
        int64_t const cacheBefore = mImpl->compiler->GetCacheSizeBytes();
        auto compiledGrammar = mImpl->compile(params);
        auto const elapsedMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        int64_t const cacheAfter = mImpl->compiler->GetCacheSizeBytes();

        mImpl->slots[static_cast<size_t>(slot)].emplace(Impl::SlotGrammar{xgrammar::GrammarMatcher(compiledGrammar),
            params.type == GuideType::kJsonObject || params.type == GuideType::kJsonSchema});
        mImpl->warnedUnmappable[static_cast<size_t>(slot)] = 0;

        LOG_INFO("Guided decoding: compiled %s for slot %d in %.2f ms (cache %s, now %.1f KB)",
            guideTypeName(params.type), slot, elapsedMs, cacheAfter == cacheBefore ? "hit" : "miss",
            static_cast<double>(cacheAfter) / 1024.0);
        return true;
    }
    catch (std::exception const& e)
    {
        failReason = e.what();
        mImpl->slots[static_cast<size_t>(slot)].reset();
        return false;
    }
}

void GuidedDecoder::reset()
{
    std::fill(mImpl->slots.begin(), mImpl->slots.end(), std::nullopt);
}

void GuidedDecoder::compactSlots(std::vector<int32_t> const& batchMapping)
{
    if (mImpl->slots.empty())
    {
        return;
    }
    // Missing this leaves slot i pointing at another request's matcher: the constraint
    // migrates silently, and the terminated-matcher gate starts reading the wrong object.
    std::fill(mImpl->compactedSlots.begin(), mImpl->compactedSlots.end(), std::nullopt);
    for (size_t oldSlot = 0; oldSlot < batchMapping.size() && oldSlot < mImpl->slots.size(); ++oldSlot)
    {
        int32_t const newSlot = batchMapping[oldSlot];
        if (newSlot >= 0 && newSlot < static_cast<int32_t>(mImpl->compactedSlots.size()))
        {
            mImpl->compactedSlots[static_cast<size_t>(newSlot)] = std::move(mImpl->slots[oldSlot]);
        }
    }
    mImpl->slots.swap(mImpl->compactedSlots);
    std::fill(mImpl->compactedSlots.begin(), mImpl->compactedSlots.end(), std::nullopt);
}

bool GuidedDecoder::hasAnyGrammar() const noexcept
{
    return std::any_of(mImpl->slots.begin(), mImpl->slots.end(), [](auto const& slot) { return slot.has_value(); });
}

bool GuidedDecoder::hasGrammar(int32_t slot) const noexcept
{
    return slot >= 0 && slot < static_cast<int32_t>(mImpl->slots.size())
        && mImpl->slots[static_cast<size_t>(slot)].has_value();
}

bool GuidedDecoder::isTerminated(int32_t slot) const noexcept
{
    return hasGrammar(slot) && mImpl->slots[static_cast<size_t>(slot)]->matcher.IsTerminated();
}

bool GuidedDecoder::hasCompletedJson(int32_t slot)
{
    if (!hasGrammar(slot) || !mImpl->slots[static_cast<size_t>(slot)]->finishOnCompletion)
    {
        return false;
    }
    auto& matcher = mImpl->slots[static_cast<size_t>(slot)]->matcher;
    if (matcher.IsTerminated() || !matcher.IsCompleted())
    {
        return false;
    }
    // A complete value may still grow, so finish only when nothing but a stop token fits.
    std::array<int64_t, 2> shape{1, static_cast<int64_t>(mImpl->bitmaskSize)};
    DLTensor bitmaskTensor{};
    bitmaskTensor.data = mImpl->completionBitmask.data();
    bitmaskTensor.device = DLDevice{kDLCPU, 0};
    bitmaskTensor.ndim = 2;
    bitmaskTensor.dtype = xgrammar::GetBitmaskDLType();
    bitmaskTensor.shape = shape.data();
    matcher.FillNextTokenBitmask(&bitmaskTensor, 0);
    for (int32_t const stopId : mImpl->stopTokenIds)
    {
        mImpl->completionBitmask[static_cast<size_t>(stopId >> 5)] &= ~static_cast<int32_t>(1U << (stopId & 31));
    }
    return std::all_of(
        mImpl->completionBitmask.begin(), mImpl->completionBitmask.end(), [](int32_t word) { return word == 0; });
}

bool GuidedDecoder::advance(int32_t slot, int32_t outputSpaceToken)
{
    if (!hasGrammar(slot))
    {
        return true;
    }
    auto& matcher = mImpl->slots[static_cast<size_t>(slot)]->matcher;
    if (matcher.IsTerminated())
    {
        return true; // Feeding a terminated matcher only produces warnings.
    }
    return matcher.AcceptToken(outputSpaceToken);
}

void GuidedDecoder::fillMasks(int32_t activeBatchSize, int32_t rowsPerSlot,
    std::vector<int8_t> const& maskSuppressedPerSlot, std::vector<int32_t>& unsatisfiableSlots, cudaStream_t stream)
{
    unsatisfiableSlots.clear();
    if (activeBatchSize <= 0 || rowsPerSlot <= 0)
    {
        return;
    }
    ELLM_CHECK(rowsPerSlot == 1,
        "GuidedDecoder::fillMasks fills one row per slot from an un-advanced matcher, so it only means "
        "anything for vanilla decode; speculative verification must use fillMasksForDraftTree");
    int32_t const totalRows = activeBatchSize * rowsPerSlot;
    ELLM_CHECK(totalRows <= mImpl->maxRows, "GuidedDecoder::fillMasks row count exceeds the allocated buffer");

    // XGrammar writes straight into row `index` of the host buffer.
    std::array<int64_t, 2> shape{static_cast<int64_t>(mImpl->maxRows), static_cast<int64_t>(mImpl->bitmaskSize)};
    DLTensor bitmaskTensor{};
    bitmaskTensor.data = mImpl->hostBitmask.rawPointer();
    bitmaskTensor.device = DLDevice{kDLCPU, 0};
    bitmaskTensor.ndim = 2;
    bitmaskTensor.dtype = xgrammar::GetBitmaskDLType();
    bitmaskTensor.shape = shape.data();

    int32_t* const rowNeedsMask = mImpl->hostRowNeedsMask.dataPointer<int32_t>();
    int32_t const* const hostBitmask = mImpl->hostBitmask.dataPointer<int32_t>();

    auto const leaveSlotUnconstrained = [&](int32_t slot) {
        for (int32_t row = 0; row < rowsPerSlot; ++row)
        {
            rowNeedsMask[slot * rowsPerSlot + row] = 0;
        }
    };

    bool anyRowNeedsMask = false;
    for (int32_t slot = 0; slot < activeBatchSize; ++slot)
    {
        bool const suppressed = slot < static_cast<int32_t>(maskSuppressedPerSlot.size())
            && maskSuppressedPerSlot[static_cast<size_t>(slot)] != 0;

        // Both gates matter and they are independent. A request can finish on max-length,
        // a stop string, or cancellation while its grammar is still mid-rule; when EOS
        // is ignored, the matcher terminates while the request keeps going.
        // Filling a mask after termination is a hard error in XGrammar, not a no-op.
        if (!hasGrammar(slot) || suppressed || mImpl->slots[static_cast<size_t>(slot)]->matcher.IsTerminated())
        {
            leaveSlotUnconstrained(slot);
            continue;
        }

        // One mask per row, not per slot: each row sits at its own grammar state.
        bool unsatisfiable = false;
        for (int32_t row = 0; row < rowsPerSlot && !unsatisfiable; ++row)
        {
            int32_t const rowId = slot * rowsPerSlot + row;
            mImpl->slots[static_cast<size_t>(slot)]->matcher.FillNextTokenBitmask(&bitmaskTensor, rowId);
            rowNeedsMask[rowId] = 1;

            // All-zero means the grammar admits nothing; reachable only with a pruned
            // vocabulary that cannot spell what the schema needs.
            int32_t const* const rowMask = hostBitmask + static_cast<size_t>(rowId) * mImpl->bitmaskSize;
            int32_t accumulated = 0;
            for (int32_t word = 0; word < mImpl->bitmaskSize; ++word)
            {
                accumulated |= rowMask[word];
            }
            unsatisfiable = accumulated == 0;
        }

        if (unsatisfiable)
        {
            // The caller is about to fail this slot. Applying the all-zero mask first would
            // drive the whole row to the sentinel and make sampling uniformly random, so skip
            // it and leave the logits alone.
            leaveSlotUnconstrained(slot);
            unsatisfiableSlots.push_back(slot);
            continue;
        }
        anyRowNeedsMask = true;
    }

    // Always upload the flags, so a row left set by a previous step cannot be reapplied.
    CUDA_CHECK(cudaMemcpyAsync(mImpl->deviceRowNeedsMask.rawPointer(), mImpl->hostRowNeedsMask.rawPointer(),
        static_cast<size_t>(totalRows) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    if (!anyRowNeedsMask)
    {
        return;
    }

    // Unconditional: unlike logit bias, the mask changes every step.
    CUDA_CHECK(cudaMemcpyAsync(mImpl->deviceBitmask.rawPointer(), mImpl->hostBitmask.rawPointer(),
        static_cast<size_t>(totalRows) * static_cast<size_t>(mImpl->bitmaskSize) * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
}

void GuidedDecoder::captureDraft(SpecVerifyInputs const& inputs, cudaStream_t stream)
{
    if (inputs.requestCount <= 0 || inputs.tokensPerRequest <= 0 || mImpl->hostDraftTokens.isEmpty())
    {
        return;
    }
    auto const elements = static_cast<size_t>(inputs.requestCount) * static_cast<size_t>(inputs.tokensPerRequest);
    bool const tree = inputs.layout == SpecVerifyLayout::kTree;
    ELLM_CHECK(static_cast<size_t>(inputs.tokenIds.getShape().volume()) >= elements,
        "GuidedDecoder::captureDraft was given fewer tokens than the batch needs");
    ELLM_CHECK(elements <= static_cast<size_t>(mImpl->maxRows) && inputs.requestCount <= mImpl->maxBatchSize,
        "GuidedDecoder::captureDraft exceeds allocated capacity");
    if (tree)
    {
        ELLM_CHECK(
            inputs.parentIds != nullptr && static_cast<size_t>(inputs.parentIds->getShape().volume()) >= elements,
            "GuidedDecoder::captureDraft was given incomplete tree topology");
    }
    ELLM_CHECK(inputs.executionValidTokenCounts == nullptr
            || static_cast<size_t>(inputs.executionValidTokenCounts->getShape().volume())
                >= static_cast<size_t>(inputs.requestCount),
        "GuidedDecoder::captureDraft was given fewer node counts than the batch needs");
    CUDA_CHECK(cudaMemcpyAsync(mImpl->hostDraftTokens.rawPointer(), inputs.tokenIds.rawPointer(),
        elements * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    mImpl->draftIsTree = tree;
    mImpl->draftHasValidCounts = inputs.executionValidTokenCounts != nullptr;
    mImpl->draftHasForcedFlags = inputs.forcedFlags != nullptr;
    if (tree)
    {
        CUDA_CHECK(cudaMemcpyAsync(mImpl->hostDraftParentIds.rawPointer(), inputs.parentIds->rawPointer(),
            elements * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    }
    if (mImpl->draftHasValidCounts)
    {
        CUDA_CHECK(
            cudaMemcpyAsync(mImpl->hostDraftValidCounts.rawPointer(), inputs.executionValidTokenCounts->rawPointer(),
                static_cast<size_t>(inputs.requestCount) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    }
    if (mImpl->draftHasForcedFlags)
    {
        ELLM_CHECK(static_cast<size_t>(inputs.forcedFlags->getShape().volume()) >= elements,
            "GuidedDecoder::captureDraft was given fewer forced flags than the batch needs");
        CUDA_CHECK(cudaMemcpyAsync(mImpl->hostDraftForcedFlags.rawPointer(), inputs.forcedFlags->rawPointer(),
            elements * sizeof(int8_t), cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaEventRecord(mImpl->draftCopyReady, stream));
    mImpl->draftCopyPending = true;
}

int32_t const* GuidedDecoder::hostDraftTokens() const
{
    return mImpl->hostDraftTokens.isEmpty() ? nullptr : mImpl->hostDraftTokens.dataPointer<int32_t>();
}

int32_t const* GuidedDecoder::hostDraftParentIds() const
{
    return mImpl->draftIsTree ? mImpl->hostDraftParentIds.dataPointer<int32_t>() : nullptr;
}

int32_t const* GuidedDecoder::hostDraftValidCounts() const
{
    return mImpl->draftHasValidCounts ? mImpl->hostDraftValidCounts.dataPointer<int32_t>() : nullptr;
}

int8_t const* GuidedDecoder::hostDraftForcedFlags() const
{
    return mImpl->draftHasForcedFlags ? mImpl->hostDraftForcedFlags.dataPointer<int8_t>() : nullptr;
}

void GuidedDecoder::waitForDraftTopology()
{
    if (!mImpl->draftCopyPending)
    {
        return;
    }
    CUDA_CHECK(cudaEventSynchronize(mImpl->draftCopyReady));
    mImpl->draftCopyPending = false;
}

bool GuidedDecoder::draftCopyPending() const noexcept
{
    return mImpl->draftCopyPending;
}

void GuidedDecoder::fillMasksForDraftTree(int32_t activeBatchSize, int32_t rowsPerSlot,
    int32_t const* draftTokensFullSpace, int32_t const* parentIds, int32_t const* validCounts,
    int8_t const* forcedFlags, std::vector<int8_t> const& slotSuppressed,
    std::vector<int8_t> const& reasoningEndedPerSlot, std::vector<int32_t>& unsatisfiableSlots, cudaStream_t stream)
{
    unsatisfiableSlots.clear();
    if (activeBatchSize <= 0 || rowsPerSlot <= 0)
    {
        return;
    }
    ELLM_CHECK(draftTokensFullSpace != nullptr, "GuidedDecoder::fillMasksForDraftTree needs the draft tokens");
    int32_t const totalRows = activeBatchSize * rowsPerSlot;
    ELLM_CHECK(totalRows <= mImpl->maxRows, "GuidedDecoder::fillMasksForDraftTree exceeds the allocated buffer");

    // XGrammar writes straight into row `index` of the host buffer, so the whole slot block is
    // addressed by absolute row number and no sub-view is needed.
    std::array<int64_t, 2> shape{static_cast<int64_t>(mImpl->maxRows), static_cast<int64_t>(mImpl->bitmaskSize)};
    DLTensor bitmaskTensor{};
    bitmaskTensor.data = mImpl->hostBitmask.rawPointer();
    bitmaskTensor.device = DLDevice{kDLCPU, 0};
    bitmaskTensor.ndim = 2;
    bitmaskTensor.dtype = xgrammar::GetBitmaskDLType();
    bitmaskTensor.shape = shape.data();

    int32_t* const rowNeedsMask = mImpl->hostRowNeedsMask.dataPointer<int32_t>();
    int32_t* const firstChild = mImpl->firstChild.data();
    int32_t* const nextSibling = mImpl->nextSibling.data();
    bool anyRowNeedsMask = false;

    for (int32_t slot = 0; slot < activeBatchSize; ++slot)
    {
        int32_t const base = slot * rowsPerSlot;
        // Clearing the flags is what makes an unvisited row harmless: the kernel skips it and
        // never reads the stale mask left there by an earlier step. Clearing the rows
        // themselves would cost `bitmaskSize` bytes each instead of four.
        for (int32_t row = 0; row < rowsPerSlot; ++row)
        {
            rowNeedsMask[base + row] = 0;
        }

        bool const suppressed
            = slot < static_cast<int32_t>(slotSuppressed.size()) && slotSuppressed[static_cast<size_t>(slot)] != 0;
        // Both gates are independent, and filling past termination aborts inside XGrammar.
        if (!hasGrammar(slot) || suppressed || mImpl->slots[static_cast<size_t>(slot)]->matcher.IsTerminated())
        {
            continue;
        }
        auto& matcher = mImpl->slots[static_cast<size_t>(slot)]->matcher;

        // A chain is the tree with `parent[i] == i - 1`, so one walk covers both geometries and
        // the chain path cannot drift away from the tree path.
        int32_t nodeCount = rowsPerSlot;
        if (validCounts != nullptr)
        {
            nodeCount = std::min(validCounts[slot], rowsPerSlot);
        }
        if (nodeCount <= 0)
        {
            continue;
        }

        // One descending pass suffices because the tree builder appends a node only after its
        // parent, so `parent[node] < node` holds; prepending therefore also leaves siblings in
        // ascending, i.e. score-prioritized, order.
        std::fill(firstChild, firstChild + nodeCount, -1);
        for (int32_t node = nodeCount - 1; node >= 1; --node)
        {
            int32_t const parent = parentIds != nullptr ? parentIds[base + node] : node - 1;
            if (parent < 0)
            {
                // A node whose parent missed the selection hangs off nothing, so the accept walk
                // can never reach it. Selecting by score alone can produce these.
                continue;
            }
            ELLM_CHECK(parent < node,
                "GuidedDecoder::fillMasksForDraftTree got a draft tree whose nodes do not follow their parents");
            nextSibling[node] = firstChild[parent];
            firstChild[parent] = node;
        }

        // Reasoning state is per path, not per slot: one branch can leave the thinking block
        // while its sibling is still inside it, and carrying a single flag across the walk
        // would constrain the sibling from a marker it never saw.
        bool const rootReasoningEnded = slot < static_cast<int32_t>(reasoningEndedPerSlot.size())
            && reasoningEndedPerSlot[static_cast<size_t>(slot)] != 0;

        // A forced child replaces its parent's prediction, so the parent's row needs no mask and no sibling is
        // reachable. The chain was validated when it was planned.
        auto forcedChildOf = [&](int32_t node) -> int32_t {
            if (forcedFlags == nullptr)
            {
                return -1;
            }
            for (int32_t child = firstChild[node]; child >= 0; child = nextSibling[child])
            {
                if (forcedFlags[base + child] != 0)
                {
                    return child;
                }
            }
            return -1;
        };
        int32_t const rootForcedChild = forcedChildOf(0);

        // Node 0 is the token the previous step committed; the matcher already consumed it.
        if (rootReasoningEnded && rootForcedChild < 0)
        {
            matcher.FillNextTokenBitmask(&bitmaskTensor, base);
            if (!mImpl->rowHasAnyToken(base))
            {
                // Only row 0 can make the request fail: its token is always committed. A later
                // row admitting nothing just means that draft node is unreachable.
                unsatisfiableSlots.push_back(slot);
                continue;
            }
            rowNeedsMask[base] = 1;
            anyRowNeedsMask = true;
        }

        // Entered with the matcher standing where this node's parent left it, and left with it
        // standing there again, so a rejected branch costs its siblings nothing.
        auto visit = [&](auto&& self, int32_t node, bool reasoningEnded, bool forced) -> void {
            int32_t const row = base + node;
            int32_t const fullId = draftTokensFullSpace[row];

            if (!reasoningEnded)
            {
                if (mImpl->isReasoningEnd(fullId))
                {
                    // The separator itself is not constrained output and is not fed to the
                    // matcher, but the row it owns is the first one the grammar governs: it
                    // masks the token that follows the block, with the grammar still at its
                    // start state.
                    reasoningEnded = true;
                    matcher.FillNextTokenBitmask(&bitmaskTensor, row);
                    if (mImpl->rowHasAnyToken(row))
                    {
                        rowNeedsMask[row] = 1;
                        anyRowNeedsMask = true;
                    }
                }
                for (int32_t child = firstChild[node]; child >= 0; child = nextSibling[child])
                {
                    self(self, child, reasoningEnded, false);
                }
                return;
            }

            int32_t const outputId = mImpl->toOutputSpace(fullId);
            if (outputId < 0)
            {
                // The engine cannot emit this token, so the base model can never sample it and
                // the node is dead anyway. Reachable only when the reduced vocabulary is not a
                // superset of the draft's, e.g. an EAGLE export without --d2t_path.
                if (mImpl->warnedUnmappable[static_cast<size_t>(slot)] == 0)
                {
                    mImpl->warnedUnmappable[static_cast<size_t>(slot)] = 1;
                    LOG_WARNING(
                        "Request %d: draft token %d has no image in the engine's reduced vocabulary; every draft "
                        "node below it is rejected. Check that the vocabulary reduction kept the draft's tokens.",
                        slot, fullId);
                }
                return;
            }
            // The parent's row is always filled and never negative: the walk starts at node 0 or
            // a separator, both of which fill their own row, and a parentless node is never linked.
            // A forced node's parent row is not filled, and the chain needs no gate.
            int32_t const parentRow = base + (parentIds != nullptr ? parentIds[row] : node - 1);
            if (!forced && !mImpl->rowAllows(parentRow, outputId))
            {
                return; // Grammar refuses this node: it and its subtree stay unmasked.
            }
            if (!matcher.AcceptToken(outputId))
            {
                return;
            }
            if (!matcher.IsTerminated()) // Filling a mask past termination is a hard error, not a no-op.
            {
                int32_t const forcedChild = forcedChildOf(node);
                if (forcedChild >= 0)
                {
                    self(self, forcedChild, true, true);
                }
                else
                {
                    matcher.FillNextTokenBitmask(&bitmaskTensor, row);
                    rowNeedsMask[row] = 1;
                    anyRowNeedsMask = true;
                    for (int32_t child = firstChild[node]; child >= 0; child = nextSibling[child])
                    {
                        self(self, child, true, false);
                    }
                }
            }
            matcher.Rollback(1);
        };

        if (rootForcedChild >= 0)
        {
            visit(visit, rootForcedChild, rootReasoningEnded, true);
        }
        else
        {
            for (int32_t child = firstChild[0]; child >= 0; child = nextSibling[child])
            {
                visit(visit, child, rootReasoningEnded, false);
            }
        }

        // The walk is speculative: the grammar advances only once the step's tokens are known
        // to be committed, in advanceCommitted.
    }

    CUDA_CHECK(cudaMemcpyAsync(mImpl->deviceRowNeedsMask.rawPointer(), mImpl->hostRowNeedsMask.rawPointer(),
        static_cast<size_t>(totalRows) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    if (!anyRowNeedsMask)
    {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(mImpl->deviceBitmask.rawPointer(), mImpl->hostBitmask.rawPointer(),
        static_cast<size_t>(totalRows) * static_cast<size_t>(mImpl->bitmaskSize) * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
}

bool GuidedDecoder::advanceCommitted(
    int32_t slot, int32_t const* committedFullSpace, int32_t count, int8_t& reasoningEnded)
{
    if (!hasGrammar(slot) || count <= 0)
    {
        return true;
    }
    auto& matcher = mImpl->slots[static_cast<size_t>(slot)]->matcher;
    for (int32_t i = 0; i < count; ++i)
    {
        int32_t const fullId = committedFullSpace[i];
        if (reasoningEnded == 0)
        {
            if (mImpl->isReasoningEnd(fullId))
            {
                reasoningEnded = 1; // The separator is consumed here, never fed to the matcher.
            }
            continue;
        }
        if (matcher.IsTerminated())
        {
            return true; // Everything after the stop token belongs to a finished request.
        }
        int32_t const outputId = mImpl->toOutputSpace(fullId);
        if (outputId < 0)
        {
            return false; // Committed tokens come from the engine's own vocabulary: internal bug.
        }
        if (!matcher.AcceptToken(outputId))
        {
            return false;
        }
    }
    return true;
}

std::string GuidedDecoder::forcedContinuation(int32_t slot) const
{
    if (!hasGrammar(slot))
    {
        return {};
    }
    auto& matcher = mImpl->slots[static_cast<size_t>(slot)]->matcher;
    // Asking a terminated matcher is a hard error in XGrammar; a complete root rule forces nothing.
    if (matcher.IsTerminated() || matcher.IsCompleted())
    {
        return {};
    }
    return matcher.FindJumpForwardString();
}

int32_t GuidedDecoder::acceptForced(int32_t slot, int32_t const* outputSpaceIds, int32_t count)
{
    if (!hasGrammar(slot) || count <= 0)
    {
        return 0;
    }
    auto& matcher = mImpl->slots[static_cast<size_t>(slot)]->matcher;
    int32_t accepted = 0;
    while (accepted < count && !matcher.IsTerminated() && matcher.AcceptToken(outputSpaceIds[accepted]))
    {
        ++accepted;
    }
    return accepted;
}

void GuidedDecoder::rollback(int32_t slot, int32_t count)
{
    if (!hasGrammar(slot) || count <= 0)
    {
        return;
    }
    mImpl->slots[static_cast<size_t>(slot)]->matcher.Rollback(count);
}

bool GuidedDecoder::completesGrammar(int32_t slot, std::string const& text) const
{
    if (!hasGrammar(slot))
    {
        return false;
    }
    auto& matcher = mImpl->slots[static_cast<size_t>(slot)]->matcher;
    if (matcher.IsTerminated())
    {
        return false;
    }
    if (text.empty())
    {
        return matcher.IsCompleted();
    }
    // The whole string is one rollback step, and a rejected string leaves no trace.
    if (!matcher.AcceptString(text))
    {
        return false;
    }
    bool const completed = matcher.IsCompleted();
    matcher.Rollback(1);
    return completed;
}

int32_t GuidedDecoder::toOutputSpace(int32_t fullVocabId) const
{
    return mImpl->toOutputSpace(fullVocabId);
}

void GuidedDecoder::prepareJumpForward()
{
    mImpl->ensureExtendableTable();
}

bool GuidedDecoder::isExtendable(int32_t outputSpaceId) const
{
    if (outputSpaceId < 0 || outputSpaceId >= mImpl->outputVocabSize)
    {
        return true;
    }
    mImpl->ensureExtendableTable();
    return mImpl->extendable[static_cast<size_t>(outputSpaceId)] != 0;
}

void GuidedDecoder::applyMask(Tensor& logits, int32_t activeBatchSize, int32_t rowsPerSlot, cudaStream_t stream)
{
    if (activeBatchSize <= 0 || rowsPerSlot <= 0)
    {
        return;
    }
    int32_t const totalRows = activeBatchSize * rowsPerSlot;

    // The kernel takes 2D logits, so flatten a speculative [batch, rowsPerSlot, vocab] tensor
    // and restore it afterwards, as applyLogitBiasRepeatedRows does with the same tensor.
    auto const originalShape = logits.getShape();
    if (originalShape.getNumDims() == 3)
    {
        ELLM_CHECK(originalShape[1] == rowsPerSlot, "Guided mask rowsPerSlot does not match the logits shape");
        ELLM_CHECK(logits.reshape({originalShape[0] * originalShape[1], originalShape[2]}), "Tensor reshape failed");
        applyTokenBitmask(logits, mImpl->deviceBitmask, mImpl->deviceRowNeedsMask, totalRows, stream);
        ELLM_CHECK(logits.reshape(originalShape), "Tensor reshape failed");
        return;
    }
    applyTokenBitmask(logits, mImpl->deviceBitmask, mImpl->deviceRowNeedsMask, totalRows, stream);
}

int64_t GuidedDecoder::cacheSizeBytes() const
{
    return mImpl->compiler.has_value() ? mImpl->compiler->GetCacheSizeBytes() : 0;
}

} // namespace rt
} // namespace trt_edgellm
