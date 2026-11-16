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

#include "runtime/decoding/vanillaDecoder.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/debug/layerDebugger.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/logitBias.h"
#include "runtime/decoding/requestStableRng.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{
constexpr int32_t kPrefillProfile{0};
constexpr int32_t kDecodeProfile{1};
} // namespace

VanillaDecoder::VanillaDecoder(DecodingRuntimeContext& runtime)
    : mRuntime(runtime)
{
}

bool VanillaDecoder::needsChunkStep(DecodingInferenceContext const& context) noexcept
{
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        if (context.pendingTokens[static_cast<size_t>(i)] > 1)
        {
            return true;
        }
    }
    return false;
}

bool VanillaDecoder::decodeStep(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kLLM_GENERATION, context.stream);
    NVTX_SCOPED_RANGE(nvtx_vanilla_decoding,
        ("VANILLA_DECODING[R" + std::to_string(context.generationRound) + "," + std::to_string(context.activeBatchSize)
            + "]")
            .c_str(),
        nvtx_colors::BLUE);

    check::check(context.raggedBatchBuilder.has_value(), "Ragged batch builder is not initialized");
    bool const chunkStep = context.hasJumpForward && needsChunkStep(context);
    bool const forwardStatus = chunkStep ? runChunkForward(context) : runDecodeForward(context);
    if (!forwardStatus)
    {
        LOG_ERROR("Failed to execute vanilla decoding step for base model.");
        return false;
    }
    return sampleAndAppend(context, chunkStep);
}

bool VanillaDecoder::runDecodeForward(DecodingInferenceContext& context)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    context.scheduledStep.id = context.nextStepId++;
    context.scheduledStep.sequences.resize(static_cast<size_t>(activeBatchSize));
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        std::vector<int32_t> const& tokens = context.tokenIds[static_cast<size_t>(i)];
        context.scheduledStep.sequences[static_cast<size_t>(i)] = ScheduledSequence{
            SequenceIdentity{context.requestIds[static_cast<size_t>(i)], context.residentRefs[static_cast<size_t>(i)]},
            SequenceWork::kDecode, 1, context.committedLengths[static_cast<size_t>(i)],
            HostTokenRange{
                tokens.data(), static_cast<int32_t>(tokens.size()), static_cast<int32_t>(tokens.size()) - 1, 1},
            true};
    }
    mRuntime.base.pipelineIO.waitForStepHostStaging();
    RaggedBatchBuilder::validateRuntimeAdapterStep(
        context.scheduledStep, context.activeBatchSize, context.requestIds, context.residentRefs);
    context.raggedBatchBuilder->buildInto(context.scheduledStep, context.raggedExecutionBatch);
    RaggedExecutionBatch const& batch = context.raggedExecutionBatch;
    StepId const raggedStepId = context.scheduledStep.id;

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), batch.hostTokenIds.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    mRuntime.base.pipelineIO.recordStepHostUploads(context.stream);

    check::check(
        mRuntime.base.pipelineIO.inputsEmbeds.reshape({activeBatchSize, 1, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.preprocess.embedding.descriptor,
        mRuntime.base.pipelineIO.inputsEmbeds, context.stream);
    prepareRaggedExecutionBindings(mRuntime.base.pipelineIO, mRuntime.base.sharedResources, mRuntime.deployment.base,
        batch, /*kvCacheIndex=*/0, context.stream);
    if (mRuntime.preprocess.gemma4Ple)
    {
        mRuntime.preprocess.gemma4Ple->embed(mRuntime.preprocess.idsInput, context.stream, batch.queryLengths,
            &mRuntime.base.pipelineIO.queryLengths, &batch.hostTokenIds);
    }
    if (mRuntime.preprocess.qwen4ExpNgram)
    {
        // One token per sequence, so nothing is padded and the history advances by exactly that
        // token; prefill already seeded it from the request.
        mRuntime.preprocess.qwen4ExpNgram->embed(batch.hostTokenIds.dataPointer<int32_t>(),
            /*hostQueryLengths=*/nullptr, batch.stateIndices.data(), batch.shape.numSequences,
            /*queryWidth=*/1, context.stream);
    }
    check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape(
                     {batch.shape.physicalTokens, mRuntime.deployment.base.hiddenSize}),
        "Ragged decode embedding reshape failed");

    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");

    mRuntime.preprocess.stepPreparer.prepare(
        InferencePhase::kDecode, activeBatchSize, mRuntime.base.cacheManager, mRuntime.base.pipelineIO, context.stream);
    mRuntime.base.pipelineIO.recordStepHostUploads(context.stream);
    if (mRuntime.preprocess.deepstack)
    {
        mRuntime.preprocess.deepstack->useZeroTarget(mRuntime.base.tensorMap);
    }

    auto const decodeDims = mRuntime.deployment.base.decodeDims(activeBatchSize);
    bool decodingStatus;
    {
        auto forward
            = gTimer.startForward("base", decodeDims.batch, decodeDims.seqLen, decodeDims.selectLen, context.stream);
        decodingStatus
            = mRuntime.base.executor.prepare(kDecodeProfile, decodeDims, mRuntime.base.tensorMap, context.stream);
        if (decodingStatus)
        {
            decodingStatus = mRuntime.base.executor.execute(context.stream);
        }
    }
    if (!decodingStatus)
    {
        return false;
    }
    context.completionSequences.resize(static_cast<size_t>(activeBatchSize));
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        context.completionSequences[static_cast<size_t>(i)]
            = CompletionSequence{context.requestIds[static_cast<size_t>(i)],
                context.residentRefs[static_cast<size_t>(i)], context.committedLengths[static_cast<size_t>(i)]};
    }
    context.raggedExecutionBatch.validateCommitSnapshot(raggedStepId, context.completionSequences);
    mRuntime.base.cacheManager.commitSequenceLength(/*increment=*/1, context.stream);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        ++context.committedLengths[static_cast<size_t>(i)];
    }
    return true;
}

bool VanillaDecoder::runChunkForward(DecodingInferenceContext& context)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    PipelineIO& io = mRuntime.base.pipelineIO;
    LLMEngineConfig const& base = mRuntime.deployment.base;

    context.scheduledStep.id = context.nextStepId++;
    context.scheduledStep.sequences.resize(static_cast<size_t>(activeBatchSize));
    int32_t maxQueryLength = 1;
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        auto const slot = static_cast<size_t>(i);
        std::vector<int32_t> const& tokens = context.tokenIds[slot];
        int32_t const queryLength = context.pendingTokens[slot];
        check::check(queryLength >= 1 && queryLength <= static_cast<int32_t>(tokens.size()),
            "A catch-up step feeds every active slot's pending tokens, at least its lookahead token");
        if (queryLength > 1)
        {
            ++context.jumpForwardStats[slot].chunkSteps;
        }
        maxQueryLength = std::max(maxQueryLength, queryLength);
        context.scheduledStep.sequences[slot]
            = ScheduledSequence{SequenceIdentity{context.requestIds[slot], context.residentRefs[slot]},
                SequenceWork::kContext, queryLength, context.committedLengths[slot],
                HostTokenRange{tokens.data(), static_cast<int32_t>(tokens.size()),
                    static_cast<int32_t>(tokens.size()) - queryLength, queryLength},
                true};
    }
    NVTX_SCOPED_RANGE(nvtx_jump_chunk,
        ("JUMP_CHUNK_STEP[" + std::to_string(activeBatchSize) + "," + std::to_string(maxQueryLength) + "]").c_str(),
        nvtx_colors::BLUE);
    io.waitForStepHostStaging();
    RaggedBatchBuilder::validateRuntimeAdapterStep(
        context.scheduledStep, context.activeBatchSize, context.requestIds, context.residentRefs);
    context.raggedBatchBuilder->buildInto(context.scheduledStep, context.raggedExecutionBatch);
    RaggedExecutionBatch const& batch = context.raggedExecutionBatch;
    StepId const raggedStepId = context.scheduledStep.id;
    int32_t const physicalTokens = batch.shape.physicalTokens;
    int32_t const queryWidth = batch.shape.queryWidth;
    ExecutionPhase const phase = decoder_utils::contextPrefillPhase(batch.pastLengths, activeBatchSize);
    check::check(phase == ExecutionPhase::kContextChunk, "A catch-up step continues sequences that have KV state");

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, queryWidth}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), batch.hostTokenIds.rawPointer(),
        static_cast<size_t>(physicalTokens) * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    check::check(io.hostContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* const hostContextLengths = io.hostContextLengths.dataPointer<int32_t>();
    std::copy(batch.queryLengths.begin(), batch.queryLengths.end(), hostContextLengths);
    io.recordStepHostUploads(context.stream);

    check::check(io.inputsEmbeds.reshape({activeBatchSize, queryWidth, base.hiddenSize}), "Tensor reshape failed");
    mRuntime.preprocess.embeddingPreprocessor.embed(
        mRuntime.preprocess.idsInput, std::nullopt, std::nullopt, io, context.stream);
    if (mRuntime.preprocess.deepstack)
    {
        // The shared zero target covers one row per sequence; a chunk binds queryWidth rows per sequence.
        mRuntime.preprocess.embeddingPreprocessor.prepareDeepstack(
            mRuntime.preprocess.idsInput, OptionalInputTensors{}, io, context.stream);
    }
    prepareRaggedExecutionBindings(io, mRuntime.base.sharedResources, base, batch, /*kvCacheIndex=*/0, context.stream);
    if (mRuntime.preprocess.gemma4Ple)
    {
        mRuntime.preprocess.gemma4Ple->embed(
            mRuntime.preprocess.idsInput, context.stream, batch.queryLengths, &io.queryLengths, &batch.hostTokenIds);
    }
    if (mRuntime.preprocess.qwen4ExpNgram)
    {
        // Per-row lengths keep the padding out of the hash history.
        mRuntime.preprocess.qwen4ExpNgram->embed(batch.hostTokenIds.dataPointer<int32_t>(), batch.queryLengths.data(),
            batch.stateIndices.data(), batch.shape.numSequences, queryWidth, context.stream);
    }
    check::check(io.inputsEmbeds.reshape({physicalTokens, base.hiddenSize}), "Ragged chunk embedding reshape failed");
    check::check(io.outputHiddenStates.isEmpty() || io.outputHiddenStates.reshape({physicalTokens, base.hiddenSize}),
        "Ragged hidden-state output reshape failed");
    if (mRuntime.preprocess.deepstack)
    {
        for (Tensor& deepstack : io.deepstackEmbeds)
        {
            check::check(deepstack.reshape({physicalTokens, base.hiddenSize}), "Ragged deepstack reshape failed");
        }
        mRuntime.preprocess.deepstack->useRealFeatures(mRuntime.base.tensorMap);
    }
    check::check(io.outputLogits.reshape({activeBatchSize, base.outputVocabSize}), "Tensor reshape failed");

    mRuntime.preprocess.stepPreparer.prepare(
        InferencePhase::kPrefill, activeBatchSize, mRuntime.base.cacheManager, io, context.stream);
    io.recordStepHostUploads(context.stream);

    auto const dims = base.prefillDims(activeBatchSize, queryWidth, phase);
    bool status;
    {
        EngineExecutor::ScopedContextChunkRouting const routing(mRuntime.base.executor);
        auto forward = gTimer.startForward("base", dims.batch, dims.seqLen, dims.selectLen, context.stream);
        status = mRuntime.base.executor.prepare(kPrefillProfile, dims, mRuntime.base.tensorMap, context.stream);
        if (status)
        {
            status = mRuntime.base.executor.execute(context.stream);
        }
    }
    if (!status)
    {
        return false;
    }
    context.completionSequences.resize(static_cast<size_t>(activeBatchSize));
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        auto const slot = static_cast<size_t>(i);
        context.completionSequences[slot]
            = CompletionSequence{context.requestIds[slot], context.residentRefs[slot], context.committedLengths[slot]};
    }
    context.raggedExecutionBatch.validateCommitSnapshot(raggedStepId, context.completionSequences);
    mRuntime.base.cacheManager.commitSequenceLength(io.contextLengths, context.stream);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        auto const slot = static_cast<size_t>(i);
        context.committedLengths[slot] += batch.queryLengths[slot];
        context.pendingTokens[slot] = 0;
    }
    return true;
}

bool VanillaDecoder::sampleAndAppend(DecodingInferenceContext& context, bool chunkStep)
{
    int32_t const activeBatchSize = context.activeBatchSize;

    // GCOVR_EXCL_START
    if (context.hasLogitBias)
    {
        applyLogitBias(mRuntime.logitBias, mRuntime.base.pipelineIO.outputLogits, context, context.stream);
    }
    // GCOVR_EXCL_STOP

    if (context.hasGuidedDecoding)
    {
        // One logits row per active slot. Sits after the forward was enqueued, so the host-side
        // mask fill overlaps it; the stream keeps the ordering.
        applyGuidedDecodingMask(mRuntime.guidedDecoder, context, mRuntime.base.pipelineIO.outputLogits, activeBatchSize,
            /*rowsPerSlot=*/1, context.stream);
    }

    decoder_utils::sampleNextTokens(context, activeBatchSize, mRuntime.base.pipelineIO.outputLogits,
        mRuntime.sampling.indices, mRuntime.sampling.hostRows, mRuntime.sampling.rows, mRuntime.sampling.workspace,
        context.stream);

    // Capture the indices while still in the output vocabulary, the space the matchers work
    // in. Rides the round's single sync below.
    if (context.hasGuidedDecoding)
    {
        check::check(mRuntime.sampling.hostOutputSpaceIds.reshape({activeBatchSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mRuntime.sampling.hostOutputSpaceIds.dataPointer<int32_t>(),
            mRuntime.sampling.indices.rawPointer(), activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost,
            context.stream));
    }

    if (mRuntime.deployment.base.reducedVocabSize > 0)
    {
        mapReducedVocabToFullVocab(mRuntime.sampling.indices, mRuntime.sampling.baseVocabMappingTable, context.stream);
    }

    if (mRuntime.tokenBroadcast
        && !mRuntime.tokenBroadcast(mRuntime.sampling.indices.rawPointer(), activeBatchSize, context.stream))
    {
        LOG_ERROR("Failed to broadcast vanilla decode sampled tokens for parallel rank %d.", mRuntime.parallelRank);
        return false;
    }

    // Enqueue logprobs extraction + D2H before the round's single synchronization so the
    // copies ride the same sync as the sampled-token D2H below. Extracted at the widest width
    // any slot asks for; each slot keeps its own.
    int32_t const logprobsWidth = context.maxNumLogprobs();
    if (logprobsWidth > 0)
    {
        decoder_utils::enqueueLogprobsD2H(
            mRuntime.base.pipelineIO.outputLogits, activeBatchSize, mRuntime, logprobsWidth, context.stream);
    }

    check::check(mRuntime.sampling.hostSelectedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostSelectedTokenIdsData = mRuntime.sampling.hostSelectedTokenIds.dataPointer<int32_t>();
    CUDA_CHECK(cudaMemcpyAsync(hostSelectedTokenIdsData, mRuntime.sampling.indices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Few-layer-validation debug: dump this decode round. The KV cache is committed (line above) so
    // tokenIds[i].size() == the committed cache length for this round. A catch-up step is not dumped: its
    // rows do not match the one-token-per-round layout the dump assumes.
    if (context.layerDebugger != nullptr && !chunkStep)
    {
        std::vector<int32_t> validLengths(activeBatchSize);
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            validLengths[i] = static_cast<int32_t>(context.tokenIds[i].size());
        }
        context.layerDebugger->dumpRound(mRuntime.base.cacheManager, *mRuntime.base.sharedResources.kvPageTables[0],
            mRuntime.base.sharedResources.getSwaKVPageTable(0), mRuntime.base.pipelineIO.outputLogits, validLengths,
            context.batchIndexMapping, context.residentRefs, hostSelectedTokenIdsData, activeBatchSize, context.stream);

        // Teacher-forcing — feed the golden's tokens instead of our own (no-op unless
        // EDGELLM_FORCE_TOKENS_FILE is set). After the dump, so the dump keeps our own sampled token.
        context.layerDebugger->applyForcedTokens(
            context.currentGenerateLengths, context.batchIndexMapping, hostSelectedTokenIdsData, activeBatchSize);
    }

    decoder_utils::appendSampledTokens(context, hostSelectedTokenIdsData, activeBatchSize);

    if (context.hasGuidedDecoding)
    {
        advanceGuidedDecoding(mRuntime.guidedDecoder, context,
            mRuntime.sampling.hostOutputSpaceIds.dataPointer<int32_t>(), activeBatchSize);
    }

    if (logprobsWidth > 0)
    {
        decoder_utils::collectLogprobsFromHost(mRuntime, context, activeBatchSize, logprobsWidth);
    }

    if (context.hasJumpForward && context.hasGuidedDecoding)
    {
        collectJumps(context);
    }

    return true;
}

void VanillaDecoder::ensureCoordinator()
{
    if (!mCoordinator.has_value())
    {
        mCoordinator.emplace(mRuntime.tokenizer, mRuntime.guidedDecoder, mRuntime.chunkMaxJumpTokens);
    }
}

void VanillaDecoder::collectJumps(DecodingInferenceContext& context)
{
    NVTX_SCOPED_RANGE(nvtx_jump_collect, "JUMP_COLLECT", nvtx_colors::BLUE);
    ensureCoordinator();
    int32_t const* const sampledOutputSpaceIds = mRuntime.sampling.hostOutputSpaceIds.dataPointer<int32_t>();
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        auto const slot = static_cast<size_t>(i);
        // The sampled token is the only one a plan may replace, so a slot mid-catch-up is not planned.
        if (context.pendingTokens[slot] != 1)
        {
            continue;
        }
        int32_t const budget = context.generateBudget(i) - context.currentGenerateLengths[slot];
        JumpForwardPlan const plan = mCoordinator->plan(context, i, /*mutableTailCount=*/1, budget);
        if (plan.empty())
        {
            continue;
        }
        JumpForwardCoordinator::Applied const applied
            = mCoordinator->applyPlan(context, i, plan, sampledOutputSpaceIds[i]);
        if (applied.rolledBack + applied.appended == 0)
        {
            continue;
        }
        context.pendingTokens[slot] = 1 - applied.rolledBack + applied.appended;
        LOG_DEBUG("Request %d jump: rolled back %d, appended %d of %d planned, pending %d",
            context.batchIndexMapping[slot], applied.rolledBack, applied.appended,
            static_cast<int32_t>(plan.fullVocabIds.size()), context.pendingTokens[slot]);
    }
}

bool VanillaDecoder::captureCudaGraphs(cudaStream_t stream)
{
    bool baseVanillaDecodingCaptureStatus{true};
    RaggedEngineContract const contract{TokenLayoutBackend::kEntryPaddedCompatibility,
        mRuntime.deployment.base.maxNumSequences, mRuntime.deployment.base.maxQueryLength,
        mRuntime.deployment.base.maxPhysicalTokens, mRuntime.deployment.base.recurrentPoolRows,
        /* mixedStepSupported = */ false};
    RaggedBatchBuilder builder(contract);
    RaggedExecutionBatch batch;
    builder.reserve(batch);
    for (int32_t batchSize = 1; batchSize <= mRuntime.maxRuntimeBatchSize; ++batchSize)
    {
        check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape({batchSize, 1, mRuntime.deployment.base.hiddenSize}),
            "Tensor reshape failed");
        check::check(
            mRuntime.base.pipelineIO.outputLogits.reshape({batchSize, mRuntime.deployment.base.outputVocabSize}),
            "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.selectTokenIndices.reshape({batchSize, 1}), "Tensor reshape failed");
        check::check(mRuntime.preprocess.idsInput.reshape({batchSize, 1}), "Tensor reshape failed");
        std::vector<int32_t> dummyTokens(static_cast<size_t>(batchSize), 0);
        std::vector<ScheduledSequence> sequences(static_cast<size_t>(batchSize));
        for (int32_t i = 0; i < batchSize; ++i)
        {
            sequences[static_cast<size_t>(i)]
                = ScheduledSequence{SequenceIdentity{static_cast<RequestId>(i) + 1, ResidentRef{i, 1}},
                    SequenceWork::kDecode, 1, 0, HostTokenRange{dummyTokens.data() + i, 1, 0, 1}, true};
        }
        ScheduledStep const step{1, std::move(sequences)};
        builder.buildInto(step, batch);
        prepareRaggedExecutionBindings(mRuntime.base.pipelineIO, mRuntime.base.sharedResources,
            mRuntime.deployment.base, batch, /*kvCacheIndex=*/0, stream);
        check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape({batchSize, mRuntime.deployment.base.hiddenSize}),
            "Ragged graph-capture embedding reshape failed");

        mRuntime.preprocess.stepPreparer.prepare(
            InferencePhase::kDecode, batchSize, mRuntime.base.cacheManager, mRuntime.base.pipelineIO, stream);
        if (mRuntime.preprocess.deepstack)
        {
            mRuntime.preprocess.deepstack->useZeroTarget(mRuntime.base.tensorMap);
        }

        auto const decodeDims = mRuntime.deployment.base.decodeDims(batchSize);
        baseVanillaDecodingCaptureStatus &= mRuntime.base.captureGraph(decodeDims, stream);
    }
    return baseVanillaDecodingCaptureStatus;
}

} // namespace rt
} // namespace trt_edgellm
