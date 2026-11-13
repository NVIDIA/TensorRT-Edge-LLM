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

#include "runtime/decoding/decoderUtils.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/speculative/dflashRuntimeKernels.h"
#include "kernels/speculative/speculativeKernels.h"
#include "profiling/timer.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/debug/layerDebugger.h"
#include "runtime/decoding/logitBias.h"
#include "runtime/decoding/requestStableRng.h"
#include "runtime/state/pipelineIO.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace decoder_utils
{
void zeroActiveRegion(Tensor& tensor, cudaStream_t stream)
{
    auto const bytes = static_cast<size_t>(tensor.getShape().volume()) * utils::getTypeSize(tensor.getDataType());
    CUDA_CHECK(cudaMemsetAsync(tensor.rawPointer(), 0, bytes, stream));
}

void prepareSpecRaggedBindings(DecodingRuntimeContext& runtime, LLMEngineConfig const& cfg, int32_t kvCacheIndex,
    Tensor const& attentionPositions, Tensor const& committedPastLengths, Tensor const* validCounts,
    Tensor const& selectedTokenIndices, int32_t selectedRows, std::vector<ResidentRef> const* residentRefs,
    int32_t batchSize, int32_t queryWidth, InferenceDims const& dims, cudaStream_t stream)
{
    ExecutionPhase const phase = executionPhase(dims);
    PipelineIO& io = runtime.base.pipelineIO;
    int32_t const physicalTokens = batchSize * queryWidth;
    check::check(io.positions.reshape({physicalTokens}), "Tensor reshape failed");
    check::check(io.queryStartOffsets.reshape({batchSize + 1}), "Tensor reshape failed");
    check::check(io.queryLengths.reshape({batchSize}), "Tensor reshape failed");
    check::check(io.pastLengths.reshape({batchSize}), "Tensor reshape failed");
    check::check(io.attentionSequenceLengths.reshape({batchSize}), "Tensor reshape failed");
    int32_t* treeParentIds = nullptr;
    int32_t* treeDepths = nullptr;
    if (!io.specTreeParentIds.isEmpty() && !io.specTreeDepths.isEmpty())
    {
        check::check(io.specTreeParentIds.reshape({physicalTokens}), "Tensor reshape failed");
        check::check(io.specTreeDepths.reshape({physicalTokens}), "Tensor reshape failed");
        treeParentIds = io.specTreeParentIds.dataPointer<int32_t>();
        treeDepths = io.specTreeDepths.dataPointer<int32_t>();
    }
    io.uploadStateIndices(residentRefs, batchSize, stream);
    kernel::launchPrepareSpecRaggedMetadata(attentionPositions.dataPointer<int32_t>(),
        committedPastLengths.dataPointer<int32_t>(),
        validCounts == nullptr ? nullptr : validCounts->dataPointer<int32_t>(), queryWidth,
        io.positions.dataPointer<int32_t>(), io.queryStartOffsets.dataPointer<int32_t>(),
        io.queryLengths.dataPointer<int32_t>(), io.pastLengths.dataPointer<int32_t>(),
        io.attentionSequenceLengths.dataPointer<int32_t>(), treeParentIds, treeDepths,
        phase == ExecutionPhase::kSpecDraftProposal, batchSize, stream);
    check::check(io.logitsIndices.reshape({selectedRows}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(io.logitsIndices.rawPointer(), selectedTokenIndices.rawPointer(),
        static_cast<size_t>(selectedRows) * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream));
    check::check(io.contextSequenceCountCarrier.reshape({0}), "Tensor reshape failed");
    prepareRaggedKVPageTable(io, *runtime.base.sharedResources.kvPageTables.at(kvCacheIndex), batchSize, stream);
    prepareRaggedRope(io, runtime.base.sharedResources, cfg, physicalTokens, batchSize, stream);
}

RaggedExecutionBatch buildSpecPrefillRaggedBatch(
    DecodingInferenceContext const& context, int32_t queryWidth, int32_t heldBackTokens)
{
    check::check(heldBackTokens >= 0, "Held-back prefill positions must be non-negative");
    check::check(queryWidth > 0, "Speculative prefill query width must be positive");
    int32_t const batchSize = context.activeBatchSize;
    check::check(batchSize >= 0, "Speculative prefill batch size must be non-negative");
    check::check(batchSize <= std::numeric_limits<int32_t>::max() / queryWidth,
        "Speculative prefill physical-token count exceeds int32");
    check::check(context.effectivePrefillLengths.size() >= static_cast<size_t>(batchSize),
        "Missing speculative prefill query lengths");
    check::check(context.prefillStartLengths.size() >= static_cast<size_t>(batchSize),
        "Missing speculative prefill start lengths");
    check::check(
        context.residentRefs.size() >= static_cast<size_t>(batchSize), "Missing speculative prefill resident slots");
    int32_t const physicalTokens = batchSize * queryWidth;
    RaggedExecutionBatch batch;
    batch.shape = {batchSize, 0, physicalTokens, queryWidth, batchSize, 0, batchSize};
    batch.positions.resize(physicalTokens, -1);
    batch.queryStartOffsets.resize(batchSize + 1);
    batch.queryLengths.resize(batchSize);
    batch.pastLengths.assign(context.prefillStartLengths.begin(), context.prefillStartLengths.begin() + batchSize);
    batch.attentionSequenceLengths.resize(batchSize);
    batch.stateIndices.resize(batchSize);
    batch.logitsIndices.resize(batchSize);
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int32_t const queryLength = context.effectivePrefillLengths.at(static_cast<size_t>(batchIdx)) - heldBackTokens;
        check::check(queryLength > 0 && queryLength <= queryWidth,
            "Speculative prefill query length must be in [1, query width]");
        int32_t const start = batch.pastLengths[batchIdx];
        check::check(start >= 0, "Speculative prefill start length must be non-negative");
        batch.shape.validTokens += queryLength;
        batch.shape.numContextTokens += queryLength;
        batch.queryStartOffsets[batchIdx] = batchIdx * queryWidth;
        batch.queryLengths[batchIdx] = queryLength;
        batch.attentionSequenceLengths[batchIdx] = start + queryLength;
        batch.stateIndices[batchIdx] = context.residentRefs.at(static_cast<size_t>(batchIdx)).slot;
        batch.logitsIndices[batchIdx] = static_cast<int64_t>(batchIdx) * queryWidth + queryLength - 1;
        for (int32_t tokenIdx = 0; tokenIdx < queryLength; ++tokenIdx)
        {
            int32_t const row = batchIdx * queryWidth + tokenIdx;
            batch.positions[row] = start + tokenIdx;
        }
    }
    batch.queryStartOffsets[batchSize] = physicalTokens;
    return batch;
}

ExecutionPhase contextPrefillPhase(std::vector<int32_t> const& pastLengths, int32_t activeBatchSize)
{
    check::check(activeBatchSize > 0 && pastLengths.size() >= static_cast<size_t>(activeBatchSize),
        "Context phase selection requires one past length per active sequence");
    auto const activeEnd = pastLengths.begin() + activeBatchSize;
    check::check(std::all_of(pastLengths.begin(), activeEnd, [](int32_t pastLength) { return pastLength >= 0; }),
        "Context past length must be non-negative");
    bool const hasRestoredPrefix
        = std::any_of(pastLengths.begin(), activeEnd, [](int32_t pastLength) { return pastLength > 0; });
    return hasRestoredPrefix ? ExecutionPhase::kContextChunk : ExecutionPhase::kContextPrefill;
}

void prepareSpecPrefillRaggedBindings(DecodingRuntimeContext& runtime, LLMEngineConfig const& cfg, int32_t kvCacheIndex,
    DecodingInferenceContext const& context, int32_t queryWidth, InferenceDims const& dims, cudaStream_t stream,
    int32_t heldBackTokens)
{
    ExecutionPhase const phase = executionPhase(dims);
    check::check(phase == ExecutionPhase::kContextPrefill || phase == ExecutionPhase::kContextChunk,
        "Speculative draft prefill requires a context phase");
    RaggedExecutionBatch const batch = buildSpecPrefillRaggedBatch(context, queryWidth, heldBackTokens);
    int32_t const batchSize = batch.shape.numSequences;
    int32_t const physicalTokens = batch.shape.physicalTokens;
    PipelineIO& io = runtime.base.pipelineIO;
    io.uploadRaggedMetadata(batch, stream);
    prepareRaggedKVPageTable(io, *runtime.base.sharedResources.kvPageTables.at(kvCacheIndex), batchSize, stream);
    prepareRaggedRope(io, runtime.base.sharedResources, cfg, physicalTokens, batchSize, stream);
}

void copyLeadingElements(Tensor& destination, Tensor const& source, int64_t elements, cudaStream_t stream)
{
    check::check(source.getDataType() == destination.getDataType(),
        "copyLeadingElements: " + source.getName() + " and " + destination.getName() + " differ in data type");
    size_t const bytes = static_cast<size_t>(elements) * utils::getTypeSize(source.getDataType());
    check::check(elements >= 0 && static_cast<int64_t>(bytes) <= source.getMemoryCapacity()
            && static_cast<int64_t>(bytes) <= destination.getMemoryCapacity(),
        "copyLeadingElements exceeds a tensor's storage");
    if (bytes > 0)
    {
        CUDA_CHECK(
            cudaMemcpyAsync(destination.rawPointer(), source.rawPointer(), bytes, cudaMemcpyDeviceToDevice, stream));
    }
}

void copyHiddenRowsStrided(Tensor& destination, int32_t destinationStride, Tensor const& source, int32_t sourceStride,
    int32_t batch, int64_t hiddenSize, cudaStream_t stream)
{
    check::check(source.getDataType() == destination.getDataType() && sourceStride > 0
            && destinationStride >= sourceStride && batch >= 0 && hiddenSize > 0,
        "copyHiddenRowsStrided needs one data type and a destination stride covering the source's");
    size_t const rowBytes = static_cast<size_t>(hiddenSize) * utils::getTypeSize(source.getDataType());
    size_t const sourcePitch = static_cast<size_t>(sourceStride) * rowBytes;
    size_t const destinationPitch = static_cast<size_t>(destinationStride) * rowBytes;
    check::check(static_cast<int64_t>(sourcePitch * batch) <= source.getMemoryCapacity()
            && static_cast<int64_t>(destinationPitch * batch) <= destination.getMemoryCapacity(),
        "copyHiddenRowsStrided exceeds a tensor's storage");
    if (batch > 0)
    {
        CUDA_CHECK(cudaMemcpy2DAsync(destination.rawPointer(), destinationPitch, source.rawPointer(), sourcePitch,
            sourcePitch, batch, cudaMemcpyDeviceToDevice, stream));
    }
}

void seatAcceptedSpan(DecodingInferenceContext const& context, int32_t executionRow, Tensor& acceptedTokenIds,
    Tensor& acceptLength, Tensor& hostSeatStaging, Tensor& acceptedHiddenStates, int32_t hiddenStride,
    Tensor const& prefillHiddenStates, cudaStream_t stream)
{
    check::check(context.activeBatchSize == 1 && !context.tokenIds.empty() && !context.effectivePrefillLengths.empty(),
        "seatAcceptedSpan runs inside the seated row's batch-one projection");
    int32_t const prefillLength = context.effectivePrefillLengths[0];
    int32_t const sampledToken = context.tokenIds[0].back();
    int32_t const maxAcceptDepth = static_cast<int32_t>(acceptedTokenIds.getShape()[1]);
    auto const prefillShape = prefillHiddenStates.getShape();
    check::check(prefillShape.getNumDims() == 2 && prefillShape[0] >= prefillLength && prefillLength > 0
            && prefillHiddenStates.getDataType() == acceptedHiddenStates.getDataType(),
        "seatAcceptedSpan: the prefill hidden states do not cover the seated prompt");
    int64_t const hiddenSize = prefillShape[1];
    check::check(hiddenStride > 0
            && acceptedHiddenStates.getMemoryCapacity() >= static_cast<int64_t>(executionRow + 1) * hiddenStride
                    * hiddenSize * static_cast<int64_t>(utils::getTypeSize(acceptedHiddenStates.getDataType())),
        "seatAcceptedSpan: the carried hidden states have no room for the seated row");
    check::check(hostSeatStaging.getDeviceType() == DeviceType::kCPU
            && hostSeatStaging.getMemoryCapacity() >= static_cast<int64_t>((executionRow + 1) * 2 * sizeof(int32_t)),
        "seatAcceptedSpan: the host staging has no room for the seated row");
    int32_t* const staged = hostSeatStaging.dataPointer<int32_t>() + static_cast<int64_t>(executionRow) * 2;
    staged[0] = sampledToken;
    staged[1] = 1;
    CUDA_CHECK(
        cudaMemcpyAsync(acceptedTokenIds.dataPointer<int32_t>() + static_cast<int64_t>(executionRow) * maxAcceptDepth,
            staged, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(acceptLength.dataPointer<int32_t>() + executionRow, staged + 1, sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    size_t const elementBytes = utils::getTypeSize(acceptedHiddenStates.getDataType());
    size_t const rowBytes = static_cast<size_t>(hiddenSize) * elementBytes;
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(acceptedHiddenStates.rawPointer())
            + static_cast<size_t>(executionRow) * static_cast<size_t>(hiddenStride) * rowBytes,
        static_cast<char const*>(prefillHiddenStates.rawPointer()) + static_cast<size_t>(prefillLength - 1) * rowBytes,
        rowBytes, cudaMemcpyDeviceToDevice, stream));
}

void consumedTokenTail(std::vector<int32_t> const& tokens, std::vector<int32_t> const& prompt, int32_t committedLength,
    int32_t contextLength, std::vector<int32_t>& tail)
{
    int32_t const fed = std::max(0, static_cast<int32_t>(tokens.size()) - 1);
    // An EOS can stop the append short of the commit.
    int32_t const reused = std::clamp(committedLength - fed, 0, static_cast<int32_t>(prompt.size()));
    int32_t const fromPrompt = std::clamp(contextLength - fed, 0, reused);
    int32_t const fromTokens = std::min(fed, contextLength);
    tail.assign(prompt.begin() + (reused - fromPrompt), prompt.begin() + reused);
    tail.insert(tail.end(), tokens.begin() + (fed - fromTokens), tokens.begin() + fed);
}

std::unique_ptr<EngineExecutor> loadDraftEngine(std::filesystem::path const& engineDir,
    DeploymentConfig const& deployment, std::shared_ptr<RtxRuntimeCache> runtimeCache, EngineRefitOptions refit)
{
    std::filesystem::path const draftEnginePath = engineDir / "spec_draft.engine";
    std::unique_ptr<EngineExecutor> draftExecutor;
    try
    {
        draftExecutor
            = EngineExecutor::createForDraft(draftEnginePath, deployment, std::move(runtimeCache), std::move(refit));
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize draft EngineExecutor: %s", e.what());
        throw std::runtime_error("Failed to initialize draft EngineExecutor: " + std::string(e.what()));
    }
    LOG_INFO("Draft EngineExecutor successfully loaded from %s.", draftEnginePath.c_str());
    validateAgainstEngine(*deployment.draft, *draftExecutor, "draft");
    return draftExecutor;
}

void directVocabMapToOffsets(std::vector<int32_t>& vocabMap, int32_t baseVocabSize)
{
    check::check(baseVocabSize > 0, "base vocabulary size must be positive");
    for (int32_t const fullTokenId : vocabMap)
    {
        check::check(fullTokenId >= 0 && fullTokenId < baseVocabSize,
            "draft vocab_map entries must be valid base-vocab token ids");
    }
    for (size_t draftTokenId = 0; draftTokenId < vocabMap.size(); ++draftTokenId)
    {
        vocabMap[draftTokenId] -= static_cast<int32_t>(draftTokenId);
    }
}

void appendSampledTokens(DecodingInferenceContext& context, int32_t const* sampledTokenIds, int32_t activeBatchSize)
{
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        if (context.finishedStates[i])
        {
            continue;
        }
        context.tokenIds[i].push_back(sampledTokenIds[i]);
        context.currentGenerateLengths[i] += 1;
        context.pendingTokens[i] = 1;
    }
}

void appendSampledTokensForRows(
    DecodingInferenceContext& context, int32_t const* sampledTokenIds, std::vector<int32_t> const& contextRows)
{
    context.checkRowSet(contextRows);
    size_t const activeRows = static_cast<size_t>(context.activeBatchSize);
    check::check(context.finishedStates.size() >= activeRows && context.tokenIds.size() >= activeRows
            && context.currentGenerateLengths.size() >= activeRows,
        "Per-row token state does not cover the active batch");
    for (size_t i = 0; i < contextRows.size(); ++i)
    {
        size_t const row = static_cast<size_t>(contextRows[i]);
        if (context.finishedStates[row])
        {
            continue;
        }
        context.tokenIds[row].push_back(sampledTokenIds[i]);
        context.currentGenerateLengths[row] += 1;
        context.pendingTokens[row] = 1;
    }
}

float samplingUniformForSlot(DecodingInferenceContext const& context, size_t slot) noexcept
{
    uint64_t const generated = static_cast<uint64_t>(context.currentGenerateLengths[slot]);
    if (context.requestStableSampling[slot] != 0)
    {
        // The request's own stream: the same request at the same position draws the same number
        // whatever else is in the batch, so a seeded request replays and a lossless speculative
        // decoder can verify against the draw.
        constexpr uint64_t kTargetLane = 0U;
        uint64_t const position = requestStableNextAbsolutePosition(context.rawBatchedInputIds[slot].size(), generated);
        return requestStableUniform(context.samplingSeeds[slot], position, SpecRandomPurpose::kTarget, kTargetLane);
    }
    // The shared stream of release 0.11: keyed on the default seed, the number of tokens the slot has
    // generated (0 for its first token) and its row. A request repeats exactly when it lands on the
    // same row as before; rows of one batch draw different values.
    return counter_rng::counterUniform(kDefaultSamplingSeed, generated, static_cast<uint64_t>(slot));
}

void fillSamplingRowsForRows(
    DecodingInferenceContext const& context, std::vector<int32_t> const& contextRows, SamplingRow* hostRows)
{
    context.checkRowSet(contextRows);
    size_t const activeRows = static_cast<size_t>(context.activeBatchSize);
    check::check(context.samplingParams.size() >= activeRows && context.samplingSeeds.size() >= activeRows
            && context.requestStableSampling.size() >= activeRows && context.rawBatchedInputIds.size() >= activeRows
            && context.currentGenerateLengths.size() >= activeRows,
        "Per-row sampling state does not cover the active batch");
    for (size_t i = 0; i < contextRows.size(); ++i)
    {
        size_t const row = static_cast<size_t>(contextRows[i]);
        hostRows[i] = makeSamplingRow(context.samplingParams[row], samplingUniformForSlot(context, row));
    }
}

bool enqueueSamplingForRows(DecodingRuntimeContext& runtime, DecodingInferenceContext& context, Tensor& logits,
    std::vector<int32_t> const& contextRows, cudaStream_t stream)
{
    context.checkRowSet(contextRows);
    check::check(!context.hasGuidedDecoding, "Row-set sampling does not support guided decoding");
    int32_t const rows = static_cast<int32_t>(contextRows.size());
    if (rows == 0)
    {
        return true;
    }
    auto const shape = logits.getShape();
    check::check(shape.getNumDims() == 2 && shape[0] == rows,
        "Row-set sampling expects one [vocab] logits row per selected context row");

    applyLogitBiasForRows(runtime.logitBias, logits, context, contextRows, stream);

    SamplingBuffers& sampling = runtime.sampling;
    check::check(sampling.indices.reshape({rows, 1}), "Tensor reshape failed");
    size_t const rowBytes = static_cast<size_t>(rows) * sizeof(SamplingRow);
    check::check(static_cast<size_t>(sampling.hostRows.getMemoryCapacity()) >= rowBytes
            && static_cast<size_t>(sampling.rows.getMemoryCapacity()) >= rowBytes,
        "Sampling row buffers do not cover the selected rows");

    // Each selected row samples with its own parameters and the uniform of its stream, as in sampleNextTokens.
    auto* const hostRows = static_cast<SamplingRow*>(sampling.hostRows.rawPointer());
    fillSamplingRowsForRows(context, contextRows, hostRows);
    CUDA_CHECK(cudaMemcpyAsync(
        sampling.rows.rawPointer(), sampling.hostRows.rawPointer(), rowBytes, cudaMemcpyHostToDevice, stream));
    sampleRows(logits, sampling.indices, hostRows, sampling.rows, rows, sampling.workspace, stream);

    if (runtime.deployment.base.reducedVocabSize > 0)
    {
        mapReducedVocabToFullVocab(sampling.indices, sampling.baseVocabMappingTable, stream);
    }

    if (runtime.tokenBroadcast && !runtime.tokenBroadcast(sampling.indices.rawPointer(), rows, stream))
    {
        LOG_ERROR("Failed to broadcast row-set sampled tokens for parallel rank %d.", runtime.parallelRank);
        return false;
    }

    // Extracted at the widest width any selected row asks for; each row keeps its own prefix.
    int32_t const logprobsWidth = context.maxNumLogprobsForRows(contextRows);
    if (logprobsWidth > 0)
    {
        enqueueLogprobsD2H(logits, rows, runtime, logprobsWidth, stream);
    }

    check::check(sampling.hostSelectedTokenIds.reshape({rows}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(sampling.hostSelectedTokenIds.rawPointer(), sampling.indices.rawPointer(),
        static_cast<size_t>(rows) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    return true;
}

void commitSampledRows(
    DecodingRuntimeContext& runtime, DecodingInferenceContext& context, std::vector<int32_t> const& contextRows)
{
    if (contextRows.empty())
    {
        return;
    }
    appendSampledTokensForRows(context, runtime.sampling.hostSelectedTokenIds.dataPointer<int32_t>(), contextRows);
    int32_t const logprobsWidth = context.maxNumLogprobsForRows(contextRows);
    if (logprobsWidth > 0)
    {
        collectLogprobsFromHostForRows(runtime, context, contextRows, logprobsWidth);
    }
}

bool acceptedPositionIsForced(ForcedPathInfo const& info, int32_t slot, int32_t k, int32_t maxAcceptDepth)
{
    if (k < 0 || k + 1 >= maxAcceptDepth)
    {
        return false;
    }
    int32_t const next = info.hostRows->dataPointer<int32_t>()[slot * maxAcceptDepth + k + 1];
    return next >= 0 && next < info.stride
        && info.hostForced[static_cast<size_t>(slot) * static_cast<size_t>(info.stride) + static_cast<size_t>(next)]
        != 0;
}

void appendAcceptedTokens(DecodingInferenceContext& context, Tensor& hostAcceptLengths, Tensor& hostAcceptedTokenIds,
    Tensor const& deviceAcceptLength, Tensor const& deviceAcceptedTokenIds, int32_t maxAcceptDepth,
    tokenizer::Tokenizer const& tokenizer, cudaStream_t stream, int32_t proposedDraftsPerRound,
    int32_t const* perSlotProposedDrafts, int32_t forcedAcceptanceLength, bool proposedDraftsExact,
    ForcedPathInfo const* forcedPath)
{
    int32_t const activeBatchSize = context.activeBatchSize;

    check::check(hostAcceptLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(hostAcceptedTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    int32_t* hostAcceptLengthsData = hostAcceptLengths.dataPointer<int32_t>();
    int32_t* hostAcceptedTokenIdsData = hostAcceptedTokenIds.dataPointer<int32_t>();

    CUDA_CHECK(cudaMemcpyAsync(hostAcceptLengthsData, deviceAcceptLength.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hostAcceptedTokenIdsData, deviceAcceptedTokenIds.rawPointer(),
        activeBatchSize * maxAcceptDepth * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    if (forcedPath != nullptr)
    {
        check::check(forcedPath->hostRows->reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(forcedPath->hostRows->rawPointer(), forcedPath->deviceRows->rawPointer(),
            static_cast<size_t>(activeBatchSize) * static_cast<size_t>(maxAcceptDepth) * sizeof(int32_t),
            cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // The step record a runtime session reports to its scheduler: each row ran from its committed length and
    // added the positions its verification accepted (a verified-but-rejected position adds nothing), so the
    // scheduler's ledger advances by exactly what the KV cache holds.
    context.scheduledStep.id = context.nextStepId++;
    context.scheduledStep.sequences.resize(static_cast<size_t>(activeBatchSize));
    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        size_t const row = static_cast<size_t>(batchIdx);
        context.scheduledStep.sequences[row] = ScheduledSequence{
            SequenceIdentity{context.requestIds[row], context.residentRefs[row]}, SequenceWork::kDecode,
            /*queryLength=*/0, context.committedLengths[row], HostTokenRange{}, /*selectLastTokenLogits=*/true};
    }

    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        if (context.finishedStates[batchIdx])
        {
            hostAcceptLengthsData[batchIdx] = 0;
            continue;
        }
        if (forcedAcceptanceLength > 0)
        {
            ELLM_CHECK(hostAcceptedTokenIdsData[(batchIdx + 1) * maxAcceptDepth - 1] >= 0,
                "Forced acceptance length exceeds this proposal path");
        }
        int32_t const acceptLength = hostAcceptLengthsData[batchIdx];
        ELLM_CHECK(static_cast<size_t>(batchIdx) < context.committedLengths.size(),
            "Speculative acceptance is missing the host committed-length frontier");
        context.committedLengths[batchIdx] += acceptLength;
        context.scheduledStep.sequences[static_cast<size_t>(batchIdx)].queryLength = acceptLength;
        int32_t const proposedDrafts = perSlotProposedDrafts ? perSlotProposedDrafts[batchIdx] : proposedDraftsPerRound;
        int32_t appended = 0;
        for (int32_t i = 0; i < acceptLength; i++)
        {
            int32_t const token = hostAcceptedTokenIdsData[batchIdx * maxAcceptDepth + i];
            context.tokenIds[batchIdx].push_back(token);
            context.currentGenerateLengths[batchIdx]++;
            ++appended;
            bool const shouldStop = context.shouldStopAfterAcceptedToken
                ? context.shouldStopAfterAcceptedToken(batchIdx, token)
                : token == tokenizer.getEosId();
            if (shouldStop)
            {
                break;
            }
        }
        // EOS / stop can end the slot mid-accept; write back the appended count so
        // collectSpecLogprobsFromHost skips verify rows past the end of the sequence.
        hostAcceptLengthsData[batchIdx] = appended;
        // Full accept: last token is the target-sampled bonus, not a draft, so -1.
        // EOS truncation: count drafts up to and including the EOS.
        int32_t forcedTokens = 0;
        for (int32_t k = 0; forcedPath != nullptr && k < appended; ++k)
        {
            forcedTokens += acceptedPositionIsForced(*forcedPath, batchIdx, k, maxAcceptDepth) ? 1 : 0;
        }
        int32_t const acceptedDrafts = forcedAcceptanceLength > 0
            ? std::min(appended, forcedAcceptanceLength - 1)
            : std::min(std::max(0, appended - (appended == acceptLength ? 1 : 0) - forcedTokens), proposedDrafts);
        context.acceptedDraftTokens[batchIdx] += acceptedDrafts;
        context.proposedDraftTokens[batchIdx] += proposedDrafts;
        gTimer.recordAcceptance(context.batchIndexMapping[batchIdx], proposedDrafts, acceptedDrafts, appended,
            appended - acceptedDrafts - forcedTokens, context.committedLengths[batchIdx], forcedAcceptanceLength > 0,
            proposedDraftsExact);
        LOG_DEBUG(
            "SpecRound %d batchIdx %d: generated=%d, acceptedDrafts=%d, proposedDrafts=%d, "
            "cumGenerateLength=%d",
            context.generationRound, batchIdx, appended, acceptedDrafts, proposedDrafts,
            context.currentGenerateLengths[batchIdx]);
    }
}

int32_t forcedAcceptanceLength(DecodingRuntimeContext const& runtime, DecodingInferenceContext const& context)
{
    int32_t const length = runtime.benchmark == nullptr ? -1 : runtime.benchmark->forcedAcceptanceLength;
    if (length > 0)
    {
        ELLM_CHECK(!context.hasGuidedDecoding, "Fixed benchmark acceptance does not support guided decoding");
        ELLM_CHECK(context.layerDebugger == nullptr || !context.layerDebugger->hasForcedTokens(),
            "Fixed benchmark acceptance cannot be combined with teacher forcing");
    }
    return length;
}

void clampAcceptLengthsToRemainingGeneration(
    DecodingInferenceContext const& context, Tensor& deviceAcceptLength, cudaStream_t stream)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    check::check(static_cast<int32_t>(context.currentGenerateLengths.size()) >= activeBatchSize,
        "currentGenerateLengths must cover the active batch.");

    for (int32_t slotOffset = 0; slotOffset < activeBatchSize; slotOffset += kernel::kMaxAcceptLengthBudgetsPerLaunch)
    {
        int32_t const numSlots = std::min(kernel::kMaxAcceptLengthBudgetsPerLaunch, activeBatchSize - slotOffset);
        kernel::AcceptLengthBudgets budgets{};
        for (int32_t i = 0; i < numSlots; ++i)
        {
            budgets.remaining[i]
                = std::max(0, context.generateBudget(slotOffset + i) - context.currentGenerateLengths[slotOffset + i]);
        }
        kernel::clampAcceptLengths(deviceAcceptLength, budgets, slotOffset, numSlots, stream);
    }
}

void applyForcedAcceptance(DecodingInferenceContext& context, Tensor& hostAcceptLengths, Tensor& hostAcceptedTokenIds,
    Tensor& deviceAcceptLength, Tensor& deviceAcceptedTokenIds, std::vector<int32_t>& ownTokens, int32_t maxAcceptDepth,
    cudaStream_t stream)
{
    ownTokens.clear();
    if (context.layerDebugger == nullptr || !context.layerDebugger->hasForcedTokens())
    {
        return;
    }
    int32_t const activeBatchSize = context.activeBatchSize;
    check::check(hostAcceptLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(hostAcceptedTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    int32_t* lengths = hostAcceptLengths.dataPointer<int32_t>();
    int32_t* tokens = hostAcceptedTokenIds.dataPointer<int32_t>();

    CUDA_CHECK(cudaMemcpyAsync(
        lengths, deviceAcceptLength.rawPointer(), activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(tokens, deviceAcceptedTokenIds.rawPointer(),
        static_cast<size_t>(activeBatchSize) * maxAcceptDepth * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!context.layerDebugger->applyForcedAcceptance(context.currentGenerateLengths, context.batchIndexMapping,
            lengths, tokens, ownTokens, activeBatchSize, maxAcceptDepth))
    {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(
        deviceAcceptLength.rawPointer(), lengths, activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(deviceAcceptedTokenIds.rawPointer(), tokens,
        static_cast<size_t>(activeBatchSize) * maxAcceptDepth * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void dumpSpecRound(DecodingInferenceContext& context, HybridCacheManager& cacheManager, KVPageTable const& pageTable,
    KVPageTable const* swaPageTable, Tensor const& verifyLogits, Tensor const& acceptedTokenIndices,
    Tensor const& hostAcceptLengths, std::vector<int32_t> const& ownTokens, int32_t verifyTokenCount,
    int32_t maxAcceptDepth, cudaStream_t stream)
{
    if (context.layerDebugger == nullptr)
    {
        return;
    }
    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const* lengths = hostAcceptLengths.dataPointer<int32_t>();

    // Of the verify block's rows only the last accepted one has a counterpart in a vanilla golden:
    // it is the row that predicts the token this round leaves pending, which is exactly what the
    // golden's last-token logits are.
    Tensor hostIndices({activeBatchSize, maxAcceptDepth}, DeviceType::kCPU, trt_edgellm::DataType::kINT32,
        "spec_accepted_token_indices");
    auto const* indices = hostIndices.dataPointer<int32_t>();
    CUDA_CHECK(cudaMemcpyAsync(hostIndices.rawPointer(), acceptedTokenIndices.rawPointer(),
        static_cast<size_t>(activeBatchSize) * maxAcceptDepth * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Coords const shape = verifyLogits.getShape();
    int64_t const vocab = shape[shape.getNumDims() - 1];
    trt_edgellm::DataType const dtype = verifyLogits.getDataType();
    size_t const rowBytes = static_cast<size_t>(vocab) * utils::getTypeSize(dtype);
    Tensor bonusLogits({activeBatchSize, vocab}, DeviceType::kGPU, dtype, "spec_bonus_logits");
    auto* dst = static_cast<char*>(bonusLogits.rawPointer());
    auto const* src = static_cast<char const*>(verifyLogits.rawPointer());
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        int32_t const slot = std::max(0, lengths[i] - 1);
        int32_t const row = indices[static_cast<size_t>(i) * maxAcceptDepth + slot];
        size_t const offset = (static_cast<size_t>(i) * verifyTokenCount + row) * rowBytes;
        CUDA_CHECK(cudaMemcpyAsync(dst + i * rowBytes, src + offset, rowBytes, cudaMemcpyDeviceToDevice, stream));
    }

    // The bonus token is next round's input and has no cache entry yet, so the committed cache is
    // one shorter than the token list -- matching what the vanilla path reports, where the dump
    // runs before the sampled token is pushed.
    std::vector<int32_t> validLengths(activeBatchSize);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        validLengths[i] = static_cast<int32_t>(context.tokenIds[i].size()) - 1;
    }
    context.layerDebugger->dumpRound(cacheManager, pageTable, swaPageTable, bonusLogits, validLengths,
        context.batchIndexMapping, context.residentRefs, ownTokens.empty() ? nullptr : ownTokens.data(),
        activeBatchSize, stream);
}

void enqueueLogprobsD2H(
    Tensor const& inputLogits, int32_t rows, DecodingRuntimeContext& runtime, int32_t topK, cudaStream_t stream)
{
    check::check(runtime.logprobs.deviceLogprobsValues.reshape({rows, topK}), "Tensor reshape failed");
    check::check(runtime.logprobs.deviceLogprobsIndices.reshape({rows, topK}), "Tensor reshape failed");
    check::check(runtime.logprobs.hostLogprobsValues.reshape({rows, topK}), "Tensor reshape failed");
    check::check(runtime.logprobs.hostLogprobsIndices.reshape({rows, topK}), "Tensor reshape failed");
    extractTopKLogprobs(inputLogits, runtime.logprobs.deviceLogprobsValues, runtime.logprobs.deviceLogprobsIndices,
        topK, runtime.sampling.workspace, stream);
    if (runtime.deployment.base.reducedVocabSize > 0)
    {
        mapReducedVocabToFullVocab(
            runtime.logprobs.deviceLogprobsIndices, runtime.sampling.baseVocabMappingTable, stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(runtime.logprobs.hostLogprobsValues.rawPointer(),
        runtime.logprobs.deviceLogprobsValues.rawPointer(), static_cast<size_t>(rows) * topK * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(runtime.logprobs.hostLogprobsIndices.rawPointer(),
        runtime.logprobs.deviceLogprobsIndices.rawPointer(), static_cast<size_t>(rows) * topK * sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream));
}

void collectLogprobsFromHost(
    DecodingRuntimeContext& runtime, DecodingInferenceContext& context, int32_t activeBatchSize, int32_t topK)
{
    float const* hostVals = runtime.logprobs.hostLogprobsValues.dataPointer<float>();
    int32_t const* hostIdx = runtime.logprobs.hostLogprobsIndices.dataPointer<int32_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        int32_t const width = context.numLogprobsPerSlot[i];
        if (context.finishedStates[i] || width == 0)
            continue;
        check::check(width <= topK, "A slot asks for more logprobs than the batch extracted.");
        rt::LogprobsSlot& slot = context.stepLogprobs[i];
        int32_t const base = slot.numSteps * width;
        check::check(static_cast<size_t>(base + width) <= slot.data.size(), "logprobs slot overflow");
        for (int32_t k = 0; k < width; ++k)
            slot.data[base + k] = {hostIdx[i * topK + k], hostVals[i * topK + k]};
        ++slot.numSteps;
    }
}

void collectLogprobsFromHostForRows(DecodingRuntimeContext& runtime, DecodingInferenceContext& context,
    std::vector<int32_t> const& contextRows, int32_t topK)
{
    context.checkRowSet(contextRows);
    size_t const activeRows = static_cast<size_t>(context.activeBatchSize);
    check::check(context.finishedStates.size() >= activeRows && context.stepLogprobs.size() >= activeRows
            && context.numLogprobsPerSlot.size() >= activeRows,
        "Per-row logprobs state does not cover the active batch");
    float const* hostVals = runtime.logprobs.hostLogprobsValues.dataPointer<float>();
    int32_t const* hostIdx = runtime.logprobs.hostLogprobsIndices.dataPointer<int32_t>();
    for (size_t i = 0; i < contextRows.size(); ++i)
    {
        size_t const row = static_cast<size_t>(contextRows[i]);
        int32_t const width = context.numLogprobsPerSlot[row];
        if (context.finishedStates[row] || width == 0)
        {
            continue;
        }
        check::check(width <= topK, "A slot asks for more logprobs than the step extracted.");
        rt::LogprobsSlot& slot = context.stepLogprobs[row];
        size_t const base = static_cast<size_t>(slot.numSteps) * static_cast<size_t>(width);
        check::check(base + static_cast<size_t>(width) <= slot.data.size(), "logprobs slot overflow");
        size_t const staged = i * static_cast<size_t>(topK);
        for (int32_t k = 0; k < width; ++k)
        {
            slot.data[base + k] = {hostIdx[staged + k], hostVals[staged + k]};
        }
        ++slot.numSteps;
    }
}

void collectSpecLogprobsFromHost(DecodingRuntimeContext& runtime, DecodingInferenceContext& context,
    int32_t activeBatchSize, int32_t rowsPerBatch, int32_t const* hostAcceptLens, int32_t topK)
{
    float const* hostVals = runtime.logprobs.hostLogprobsValues.dataPointer<float>();
    int32_t const* hostIdx = runtime.logprobs.hostLogprobsIndices.dataPointer<int32_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        int32_t const width = context.numLogprobsPerSlot[i];
        if (context.finishedStates[i] || width == 0)
            continue;
        check::check(width <= topK, "A slot asks for more logprobs than the batch extracted.");
        int32_t const acceptLen = hostAcceptLens[i];
        rt::LogprobsSlot& slot = context.stepLogprobs[i];
        for (int32_t j = 0; j < acceptLen; ++j)
        {
            int32_t const row = i * rowsPerBatch + j;
            int32_t const base = slot.numSteps * width;
            check::check(static_cast<size_t>(base + width) <= slot.data.size(), "logprobs slot overflow");
            for (int32_t k = 0; k < width; ++k)
                slot.data[base + k] = {hostIdx[row * topK + k], hostVals[row * topK + k]};
            ++slot.numSteps;
        }
    }
}

void sampleNextTokens(DecodingInferenceContext const& context, int32_t activeBatchSize, Tensor const& logits,
    Tensor& selectedIndices, Tensor& hostRows, Tensor& deviceRows, Tensor& workspace, cudaStream_t stream)
{
    check::check(activeBatchSize > 0 && activeBatchSize <= context.activeBatchSize,
        "sampleNextTokens: the row count must cover live slots only.");
    check::check(selectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    size_t const rowBytes = static_cast<size_t>(activeBatchSize) * sizeof(SamplingRow);
    check::check(static_cast<size_t>(hostRows.getMemoryCapacity()) >= rowBytes
            && static_cast<size_t>(deviceRows.getMemoryCapacity()) >= rowBytes,
        "Sampling row buffers do not cover the active batch");

    // One row per slot: its parameters and the uniform samplingUniformForSlot derives from host state,
    // so the rows are built here and staged through pinned memory to the kernels.
    auto* const rows = static_cast<SamplingRow*>(hostRows.rawPointer());
    for (int32_t slot = 0; slot < activeBatchSize; ++slot)
    {
        rows[slot] = makeSamplingRow(context.samplingParams[slot], samplingUniformForSlot(context, slot));
    }
    CUDA_CHECK(
        cudaMemcpyAsync(deviceRows.rawPointer(), hostRows.rawPointer(), rowBytes, cudaMemcpyHostToDevice, stream));

    sampleRows(logits, selectedIndices, rows, deviceRows, activeBatchSize, workspace, stream);
}

} // namespace decoder_utils
} // namespace rt
} // namespace trt_edgellm
