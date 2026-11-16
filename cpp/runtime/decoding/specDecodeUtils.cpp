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

#include "runtime/decoding/specDecodeUtils.h"
#include "common/checkMacros.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/speculative/dsparkKernels.h"
#include "kernels/speculative/speculativeKernels.h"
#include "kernels/speculative/speculativeSampling.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/decoding/decoderUtils.h"

#include <limits>
#include <string>

namespace trt_edgellm::rt::spec_decode
{
void accept(GreedyAcceptInputs const& inputs, Tensor& ids, Tensor& counts, Tensor* rows, cudaStream_t stream,
    int32_t forcedAcceptanceLength)
{
    auto const& verify = inputs.verify;
    auto const shape = ids.getShape();
    ELLM_CHECK(shape.getNumDims() == 2 && shape[0] == verify.requestCount && shape[1] >= 0
            && shape[1] <= std::numeric_limits<int32_t>::max(),
        "Greedy acceptance output must have shape [N,A] with an INT32 output stride");
    ELLM_CHECK(verify.layout == SpecVerifyLayout::kChain || verify.layout == SpecVerifyLayout::kTree,
        "Unknown greedy verification layout");
    if (verify.layout == SpecVerifyLayout::kTree)
    {
        ELLM_CHECK(
            verify.metadataReady && verify.parentIds != nullptr, "Tree acceptance requires ready parent metadata");
        kernel::acceptSpecGreedyTree(inputs.targetTokenIds, verify.tokenIds, *verify.parentIds,
            inputs.acceptValidTokenCounts, inputs.forcedFlags, ids, counts, rows, verify.requestCount,
            verify.tokensPerRequest, static_cast<int32_t>(shape[1]), stream, forcedAcceptanceLength);
    }
    else
    {
        kernel::acceptSpecGreedyChain(inputs.targetTokenIds, verify.tokenIds, inputs.acceptValidTokenCounts,
            inputs.forcedFlags, ids, counts, rows, verify.requestCount, verify.tokensPerRequest,
            static_cast<int32_t>(shape[1]), stream, forcedAcceptanceLength);
    }
}

void accept(DenseProbabilityAcceptInputs const& inputs, Tensor& ids, Tensor& counts, cudaStream_t stream,
    int32_t forcedAcceptanceLength)
{
    kernel::dsparkProbabilisticAccept(inputs.targetProbabilities, inputs.draftProbabilities, inputs.proposalTokenIds,
        inputs.proposalLengths, inputs.uniforms, ids, counts, inputs.batchSize, inputs.draftStride,
        inputs.verifyProposalLen, inputs.vocabSize, stream, forcedAcceptanceLength);
}

void accept(SparseProbabilityAcceptInputs const& inputs, Tensor& ids, Tensor& counts, Tensor* rows, cudaStream_t stream,
    int32_t forcedAcceptanceLength)
{
    kernel::speculativeSparseAccept(inputs.targetProbabilities, inputs.targetSupportIds, inputs.draftProbabilities,
        inputs.draftSupportIds, inputs.proposalTokenIds, inputs.proposalLengths, inputs.uniforms, ids, counts, rows,
        stream, inputs.maxAcceptLengths, forcedAcceptanceLength);
}

void accept(DenseTargetSparseDraftAcceptInputs const& inputs, Tensor& ids, Tensor& counts, Tensor* rows,
    cudaStream_t stream, int32_t forcedAcceptanceLength)
{
    kernel::speculativeDenseTargetAccept(inputs.targetProbabilities, inputs.draftProbabilities, inputs.draftSupportIds,
        inputs.proposalTokenIds, inputs.proposalLengths, inputs.uniforms, ids, counts, rows, stream,
        inputs.maxAcceptLengths, forcedAcceptanceLength);
}

void commitAcceptedTargetPath(DecodingRuntimeContext& runtime, DecodingInferenceContext const& context,
    SpecAcceptResult const& accepted, int32_t verifyTokenCount, std::optional<SpecVerifyLayout> layoutOverride)
{
    auto& cache = runtime.base.cacheManager;
    auto& io = runtime.base.pipelineIO;
    auto& recurrent = cache.getMambaCacheManager();
    auto const layout = layoutOverride.value_or(runtime.deployment.specConfig->verifyLayout);
    int32_t const hiddenDim = runtime.deployment.specConfig->baseOutputHiddenDim;
    ELLM_CHECK(io.baseHiddenStates.reshape({context.activeBatchSize, verifyTokenCount, hiddenDim}),
        "Target hidden-state commit reshape failed");
    if (layout == SpecVerifyLayout::kChain)
    {
        // Preserve the verification stride: later batches are not packed to the accepted output stride.
        cache.commitSequenceLength(accepted.emittedTokenCounts, context.stream);
        recurrent.scatterAcceptedLinearStates(accepted.emittedTokenCounts, io.stateIndices, context.stream);
        cache.scatterAcceptedAuxConvStates(accepted.emittedTokenCounts, io.stateIndices, context.stream);
        return;
    }
    ELLM_CHECK(layout == SpecVerifyLayout::kTree, "Unknown target commit layout");
    ELLM_CHECK(accepted.acceptedRowIndices != nullptr, "Indexed target commit requires predictor rows");
    auto& kv = cache.getKVCacheManager();
    auto const& pageTable = *runtime.base.sharedResources.kvPageTables[0];
    for (auto const& group : cache.getKVHeadDimGroups())
    {
        kernel::commitSpecTargetKV(*accepted.acceptedRowIndices, accepted.emittedTokenCounts, cache.getKVCacheLengths(),
            io.stateIndices, group.deviceLayerInfos, group.numLayers, group.headDim, group.maxKVHeads,
            context.activeBatchSize, runtime.deployment.base.recurrentPoolRows, accepted.outputStride,
            kv.getConfig().kvCacheType, context.stream, pageTable.kernelView().dataPointer<int32_t>(), kv.numPages(),
            pageTable.maxPagesPerSeq());
    }
    kernel::gatherSpecTargetHiddenStates(
        *accepted.acceptedRowIndices, accepted.emittedTokenCounts, io.baseHiddenStates, context.stream);
    cache.commitSequenceLength(accepted.emittedTokenCounts, context.stream);
    if (recurrent.hasIntermediateRecurrentStates() || recurrent.hasIntermediateConvStates())
    {
        recurrent.replayCommitAcceptedTreeStates(
            *accepted.acceptedRowIndices, accepted.emittedTokenCounts, io.stateIndices, context.stream);
    }
    cache.scatterAcceptedAuxConvStates(
        accepted.emittedTokenCounts, io.stateIndices, context.stream, accepted.acceptedRowIndices);
    ELLM_CHECK(io.baseHiddenStates.reshape({context.activeBatchSize, accepted.outputStride, hiddenDim}),
        "Accepted hidden-state reshape failed");
}

namespace
{
void validateIntTensor(Tensor const& tensor, int64_t elements, char const* name)
{
    ELLM_CHECK(tensor.getDeviceType() == DeviceType::kGPU, std::string(name) + " must be a GPU tensor");
    ELLM_CHECK(tensor.getDataType() == trt_edgellm::DataType::kINT32, std::string(name) + " must be INT32");
    ELLM_CHECK(tensor.getShape().volume() == elements, std::string(name) + " has an invalid element count");
    ELLM_CHECK(tensor.rawPointer() != nullptr
            && tensor.getMemoryCapacity() >= elements * static_cast<int64_t>(sizeof(int32_t)),
        std::string(name) + " has insufficient storage");
}
} // namespace

void validateSpecVerifyInputs(SpecVerifyInputs const& inputs, SpecDecodeConfig const& config)
{
    ELLM_CHECK(inputs.requestCount > 0, "SpecVerifyInputs requestCount must be positive");
    ELLM_CHECK(inputs.tokensPerRequest > 0 && inputs.tokensPerRequest <= config.maxVerifyTokens,
        "SpecVerifyInputs tokensPerRequest=" + std::to_string(inputs.tokensPerRequest)
            + " must be in [1, maxVerifyTokens=" + std::to_string(config.maxVerifyTokens) + "]");
    ELLM_CHECK(inputs.requestCount <= std::numeric_limits<int32_t>::max() / inputs.tokensPerRequest,
        "SpecVerifyInputs physical token count exceeds INT32 capacity");
    ELLM_CHECK(inputs.maxPathLength > 0 && inputs.maxPathLength <= inputs.tokensPerRequest
            && inputs.maxPathLength <= config.maxPathLength,
        "SpecVerifyInputs maxPathLength exceeds verification width or configured path capacity");
    // A chain deployment's jump-forward grafting round verifies a tree: grafted nodes branch off the draft chain.
    bool const graftedTree = inputs.layout == SpecVerifyLayout::kTree && config.verifyLayout == SpecVerifyLayout::kChain
        && inputs.tokensPerRequest == config.jumpVerifyTokenCount && config.maxJumpTokens > 0;
    ELLM_CHECK((inputs.layout == SpecVerifyLayout::kChain || inputs.layout == SpecVerifyLayout::kTree)
            && (inputs.layout == config.verifyLayout || graftedTree),
        "SpecVerifyInputs layout does not match the resolved configuration");
    ELLM_CHECK((inputs.parentIds == nullptr) == (inputs.depths == nullptr),
        "SpecVerifyInputs parent IDs and depths must be supplied together");
    if (inputs.layout == SpecVerifyLayout::kTree)
    {
        ELLM_CHECK(inputs.metadataReady, "Tree verification requires ready selected metadata");
        ELLM_CHECK(inputs.parentIds != nullptr, "Tree verification requires parent IDs and depths");
    }
    int64_t const physicalTokens = static_cast<int64_t>(inputs.requestCount) * inputs.tokensPerRequest;
    validateIntTensor(inputs.tokenIds, physicalTokens, "tokenIds");
    auto const& shape = inputs.tokenIds.getShape();
    ELLM_CHECK(shape.getNumDims() == 1
            || (shape.getNumDims() == 2 && shape[0] == inputs.requestCount && shape[1] == inputs.tokensPerRequest),
        "tokenIds must have shape [N*Q] or [N,Q]");
    if (inputs.executionValidTokenCounts != nullptr)
    {
        validateIntTensor(*inputs.executionValidTokenCounts, inputs.requestCount, "executionValidTokenCounts");
        ELLM_CHECK(inputs.executionValidTokenCounts->getShape().getNumDims() == 1,
            "executionValidTokenCounts must have shape [N]");
    }
    for (auto const* tensor : {inputs.parentIds, inputs.depths})
    {
        if (tensor != nullptr)
        {
            validateIntTensor(*tensor, physicalTokens, "verify ancestry");
            auto const& ancestryShape = tensor->getShape();
            ELLM_CHECK(ancestryShape.getNumDims() == 1
                    || (ancestryShape.getNumDims() == 2 && ancestryShape[0] == inputs.requestCount
                        && ancestryShape[1] == inputs.tokensPerRequest),
                "verify ancestry must have shape [N*Q] or [N,Q]");
        }
    }
}
void prepareTargetVerifyInputs(SpecVerifyInputs const& inputs, DecodingRuntimeContext& runtime,
    DecodingInferenceContext const& context, int32_t targetKvCacheIndex)
{
    ELLM_CHECK(runtime.deployment.specConfig.has_value(), "Target verification requires speculative configuration");
    validateSpecVerifyInputs(inputs, *runtime.deployment.specConfig);
    ELLM_CHECK(inputs.requestCount == context.activeBatchSize,
        "SpecVerifyInputs requestCount does not match the active execution batch");
    ELLM_CHECK(inputs.requestCount <= runtime.deployment.maxRuntimeBatchSize(),
        "SpecVerifyInputs requestCount exceeds the runtime batch capacity");
    auto& io = runtime.base.pipelineIO;
    auto const& cfg = runtime.deployment.base;
    int64_t const rows = static_cast<int64_t>(inputs.requestCount) * inputs.tokensPerRequest;
    auto const dims = cfg.specVerifyDims(inputs.requestCount, inputs.tokensPerRequest, inputs.layout);
    ELLM_CHECK(io.inputsEmbeds.reshape({rows, cfg.hiddenSize}), "Target embedding reshape failed");
    ELLM_CHECK(io.outputLogits.reshape({rows, cfg.outputVocabSize}), "Target logits reshape failed");
    if (!io.baseHiddenStates.isEmpty())
    {
        ELLM_CHECK(io.baseHiddenStates.reshape({rows, runtime.deployment.specConfig->baseOutputHiddenDim}),
            "Target hidden-state reshape failed");
    }
    ELLM_CHECK(io.packedAttentionMask.reshape({rows, dims.packedMaskLen}), "Target packed mask reshape failed");
    ELLM_CHECK(io.specDecodePositionIds.reshape({rows}), "Target positions reshape failed");
    ELLM_CHECK(io.selectTokenIndices.reshape({rows}), "Target selection indices reshape failed");
    ELLM_CHECK(io.contextLengths.reshape({inputs.requestCount}), "Target context lengths reshape failed");
    ELLM_CHECK(io.specVerifyLayout.reshape({dims.specVerifyLayoutLen}), "Target layout marker reshape failed");
    Tensor* parents = nullptr;
    Tensor* depths = nullptr;
    if (!io.specTreeParentIds.isEmpty() && !io.specTreeDepths.isEmpty())
    {
        ELLM_CHECK(io.specTreeParentIds.reshape({rows}) && io.specTreeDepths.reshape({rows}),
            "Target ancestry reshape failed");
        parents = &io.specTreeParentIds;
        depths = &io.specTreeDepths;
    }
    if (inputs.parentIds != nullptr)
    {
        ELLM_CHECK(parents != nullptr && inputs.parentIds->rawPointer() == parents->rawPointer()
                && inputs.depths->rawPointer() == depths->rawPointer(),
            "Selected ancestry must borrow existing PipelineIO storage");
    }
    kernel::embeddingLookup(inputs.tokenIds, runtime.preprocess.embedding.table,
        runtime.preprocess.embedding.scalesAsOptional(), runtime.preprocess.embedding.descriptor, io.inputsEmbeds,
        context.stream);
    if (runtime.preprocess.deepstack != nullptr)
    {
        runtime.preprocess.deepstack->useZeroTarget(runtime.base.tensorMap);
    }
    Tensor const& committedLengths = runtime.base.cacheManager.getKVCacheLengths();
    if (!inputs.metadataReady)
    {
        kernel::prepareSpecChainVerifyInputs(committedLengths, inputs.executionValidTokenCounts, io.packedAttentionMask,
            io.specDecodePositionIds, io.selectTokenIndices, io.contextLengths, parents, depths, inputs.requestCount,
            inputs.tokensPerRequest, context.stream);
    }
    decoder_utils::prepareSpecRaggedBindings(runtime, cfg, targetKvCacheIndex, io.specDecodePositionIds,
        committedLengths, inputs.executionValidTokenCounts, io.selectTokenIndices, static_cast<int32_t>(rows),
        &context.residentRefs, inputs.requestCount, inputs.tokensPerRequest, dims, context.stream);
    runtime.specPleStaging.embed(
        runtime, context, inputs.tokenIds, inputs.requestCount, inputs.tokensPerRequest, context.stream);
    runtime.base.cacheManager.getMambaCacheManager().reshapeIntermediateStates(
        inputs.requestCount, inputs.tokensPerRequest);
    runtime.base.cacheManager.reshapeIntermediateAuxConvStates(inputs.requestCount, inputs.tokensPerRequest);
}
} // namespace trt_edgellm::rt::spec_decode
