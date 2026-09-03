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

#include "runtime/llmRankRuntime.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/inputLimits.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/pagedKvTypes.h"
#include "common/parallelArtifactNames.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "kernels/speculative/batchEvictKernels.h"
#include "multimodal/common/multimodalRunner.h"
#include "multimodal/qwen2/qwenViTRunner.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/contextCacheRequest.h"
#include "runtime/debug/layerDebugger.h"
#include "runtime/decoding/decoderRegistry.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/contextCache/blockHash.h"
#include "runtime/state/contextCache/contextCacheCoordinator.h"
#include "sampler/sampling.h"
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace
{
//! Optimization-profile indices for the composable stack. Profile 0 is prefill, profile 1 is decode
//! (including speculative tree-verification / proposal / accept). These match the profile layout baked
//! into the engines by `llmBuilder`.
constexpr int32_t kPrefillProfile{0};
constexpr int32_t kDecodeProfile{1};

} // namespace

namespace rt
{

std::vector<int32_t> LLMRankRuntime::countPromptTokens(LLMGenerationRequest const& request) const
{
    std::vector<int32_t> counts;
    counts.reserve(request.requests.size());
    for (auto const& item : request.requests)
    {
        ELLM_CHECK(item.imageBuffers.empty() && item.audioBuffers.empty() && !item.pastTrajectory.has_value(),
            "Prompt token counting is only available for text requests");

        LLMGenerationRequest::FormattedRequest formatted;
        ELLM_CHECK(mTokenizer->applyChatTemplate(
                       item, formatted, request.applyChatTemplate, request.addGenerationPrompt, request.enableThinking),
            "Failed to apply chat template while counting prompt tokens");
        auto const tokenIds = mTokenizer->encode(formatted.formattedCompleteRequest, false);
        ELLM_CHECK(!tokenIds.empty(), "Failed to tokenize prompt while counting prompt tokens");
        counts.push_back(static_cast<int32_t>(tokenIds.size()));
    }
    return counts;
}

namespace
{
bool needsCachedBlockDraftDDTreeHybridBindings(DeploymentConfig const& deployment)
{
    return deployment.specConfig.has_value() && isCachedBlockDraftMode(deployment.specDecodeMode())
        && deployment.specConfig->draftingTopK > 1 && deployment.base.numLinearAttnLayers > 0;
}

void validateCachedBlockDraftTreeMetadataBindings(
    DeploymentConfig const& deployment, EngineExecutor const& baseExecutor)
{
    if (!deployment.specConfig.has_value() || !isCachedBlockDraftMode(deployment.specDecodeMode()))
    {
        return;
    }

    char const* modeName = deployment.specDecodeMode() == SpecDecodeMode::kJetSpec ? "JetSpec" : "DFlash";
    char const* treeBaseFlag
        = deployment.specDecodeMode() == SpecDecodeMode::kJetSpec ? "--jetspec-tree-base" : "--dflash-tree-base";
    char const* linearBaseFlag
        = deployment.specDecodeMode() == SpecDecodeMode::kJetSpec ? "--jetspec-base" : "--dflash-base";

    bool const hasTreeParentIds = baseExecutor.hasIOTensor(binding_names::kTreeParentIds);
    bool const hasTreeDepths = baseExecutor.hasIOTensor(binding_names::kTreeDepths);
    bool const hasTreeMetadata = hasTreeParentIds || hasTreeDepths;
    bool const usesDDTree = deployment.specConfig->draftingTopK > 1;
    if (hasTreeMetadata)
    {
        ELLM_CHECK(hasTreeParentIds && hasTreeDepths,
            std::string(modeName) + " tree-base engine must expose both INT32 tree metadata bindings '"
                + binding_names::kTreeParentIds + "' and '" + binding_names::kTreeDepths + "'.");
        ELLM_CHECK(baseExecutor.getBindingDataType(binding_names::kTreeParentIds) == DataType::kINT32
                && baseExecutor.getBindingDataType(binding_names::kTreeDepths) == DataType::kINT32,
            std::string(modeName) + " tree-base engine tree metadata bindings must be INT32: '"
                + binding_names::kTreeParentIds + "' and '" + binding_names::kTreeDepths + "'.");
        ELLM_CHECK(usesDDTree,
            std::string(modeName) + " base engine was exported with " + treeBaseFlag
                + ", but runtime is configured for linear mode because specDraftTopK=1. "
                  "Use --specDraftTopK > 1 for DDTree, or re-export the base model with "
                + linearBaseFlag + ".");
    }

    if (!needsCachedBlockDraftDDTreeHybridBindings(deployment))
    {
        return;
    }

    ELLM_CHECK(hasTreeParentIds && hasTreeDepths,
        std::string(modeName) + " DDTree hybrid base engine requires INT32 tree metadata bindings '"
            + binding_names::kTreeParentIds + "' and '" + binding_names::kTreeDepths
            + "'. Re-export the base model with " + treeBaseFlag + ", then rebuild spec_base.engine.");
}

void validateMtpTreeMetadataBindings(DeploymentConfig const& deployment, EngineExecutor const& baseExecutor)
{
    if (!deployment.specConfig.has_value() || deployment.specDecodeMode() != SpecDecodeMode::kMTP
        || deployment.base.numLinearAttnLayers == 0)
    {
        return;
    }

    bool const hasTreeParentIds = baseExecutor.hasIOTensor(binding_names::kTreeParentIds);
    bool const hasTreeDepths = baseExecutor.hasIOTensor(binding_names::kTreeDepths);
    bool const usesDDTree = deployment.specConfig->draftingTopK > 1;
    ELLM_CHECK(hasTreeParentIds == hasTreeDepths,
        std::string("MTP tree-base engine must expose both INT32 tree metadata bindings '")
            + binding_names::kTreeParentIds + "' and '" + binding_names::kTreeDepths + "'.");
    if (hasTreeParentIds)
    {
        ELLM_CHECK(baseExecutor.getBindingDataType(binding_names::kTreeParentIds) == DataType::kINT32
                && baseExecutor.getBindingDataType(binding_names::kTreeDepths) == DataType::kINT32,
            std::string("MTP tree-base engine tree metadata bindings must be INT32: '") + binding_names::kTreeParentIds
                + "' and '" + binding_names::kTreeDepths + "'.");
    }
    ELLM_CHECK(usesDDTree == hasTreeParentIds,
        usesDDTree ? "Hybrid MTP DDTree requires a tree-base engine. Rebuild with --tree-base before using "
                     "--specDraftTopK > 1."
                   : "Hybrid MTP base engine was built with --tree-base, but runtime is configured for linear MTP. "
                     "Use --specDraftTopK > 1, or rebuild without --tree-base.");
}

} // namespace

LLMRankRuntime::LLMRankRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap,
    std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream, ParallelMapping const& mapping,
    tokenizer::Tokenizer& tokenizer, ContextCacheConfig const& contextCacheConfig, std::string const& checkpointDir,
    std::string const& draftCheckpointDir)
{
    initializeFromEngineDir(engineDir, multimodalEngineDir, loraWeightsMap, draftingConfig, stream, mapping, tokenizer,
        contextCacheConfig, checkpointDir, draftCheckpointDir);
}

LLMRankRuntime::LLMRankRuntime(ModelArtifacts&& artifacts, std::string const& engineDir,
    std::string const& multimodalEngineDir, std::unordered_map<std::string, std::string> const& loraWeightsMap,
    std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream, ParallelMapping const& mapping,
    tokenizer::Tokenizer& tokenizer, ContextCacheConfig const& contextCacheConfig)
{
    initializeCommon(std::move(artifacts), engineDir, multimodalEngineDir, loraWeightsMap, draftingConfig, stream,
        mapping, tokenizer, contextCacheConfig);
}

LLMRankRuntime::~LLMRankRuntime()
{
    if (mContextCache != nullptr && mContextCache->shutdown() != ContextCacheCoordinatorStatus::kOk)
    {
        LOG_ERROR("Context-cache shutdown could not prove stream quiescence.");
        std::terminate();
    }
}

void LLMRankRuntime::initializeFromEngineDir(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap,
    std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream, ParallelMapping const& mapping,
    tokenizer::Tokenizer& tokenizer, ContextCacheConfig const& contextCacheConfig, std::string const& checkpointDir,
    std::string const& draftCheckpointDir)
{
    int32_t const tensorParallelSize = mapping.tensorParallelSize;
    int32_t const tensorParallelRank = mapping.tensorParallelRank;
    int32_t const worldSize = mapping.worldSize;
    int32_t const globalRank = mapping.globalRank;

    ELLM_CHECK(tensorParallelSize > 0, "tensorParallelSize must be positive");
    ELLM_CHECK(tensorParallelRank >= 0 && tensorParallelRank < tensorParallelSize,
        "tensorParallelRank must be in [0, tensorParallelSize)");

    ModelArtifacts artifacts;
    std::filesystem::path const engineRoot{engineDir};
    parallel_artifacts::RankArtifactContext const artifactContext{
        worldSize > 0 ? worldSize : tensorParallelSize, globalRank >= 0 ? globalRank : tensorParallelRank};
    std::filesystem::path const baseConfigPath = draftingConfig.has_value()
        ? engineRoot / "base_config.json"
        : engineRoot / parallel_artifacts::configFileName(artifactContext);
    artifacts.checkpointDir = checkpointDir;
    artifacts.draftCheckpointDir = draftCheckpointDir;

    // Finish checkpoint reads and weight conversion before any engine can run.
    std::optional<int32_t> const weightTpRank
        = tensorParallelSize > 1 ? std::optional<int32_t>{tensorParallelRank} : std::nullopt;
    std::optional<int32_t> const weightTpSize
        = tensorParallelSize > 1 ? std::optional<int32_t>{tensorParallelSize} : std::nullopt;
    artifacts.weights.load(engineRoot, baseConfigPath, stream, artifacts.checkpointDir, {}, weightTpRank, weightTpSize);
    if (auto embedding = artifacts.weights.takeEmbedding())
    {
        artifacts.embedding.table = std::move(*embedding);
    }
    else
    {
        artifacts.embedding = loadEmbeddingTable(engineRoot / "embedding.safetensors", stream);
    }
    artifacts.pleEmbedding = artifacts.weights.takePleEmbedding();

    // Parse engine configurations and attach user drafting. The bundle factory
    // performs cross-engine consistency and drafting-vs-capacity checks.
    std::optional<std::filesystem::path> const draftConfigPath = draftingConfig.has_value()
        ? std::optional<std::filesystem::path>{engineRoot / "draft_config.json"}
        : std::nullopt;

    artifacts.deployment
        = createDeploymentConfig(baseConfigPath, draftConfigPath, draftingConfig, globalRank, worldSize);
    if (draftingConfig.has_value() && artifacts.deployment.specDecodeMode() == SpecDecodeMode::kMTP)
    {
        ELLM_CHECK(artifacts.draftCheckpointDir.empty(),
            "Native MTP draft weights are part of --checkpointDir; do not pass --draftCheckpointDir.");
        artifacts.draftCheckpointDir = artifacts.checkpointDir;
    }

    std::filesystem::path const baseEnginePath = artifacts.deployment.base.isDiffusionBackbone
        ? engineRoot / "dllm.engine"
        : draftingConfig.has_value() ? engineRoot / "spec_base.engine"
                                     : engineRoot / parallel_artifacts::engineFileName(artifactContext);
    ELLM_CHECK(std::filesystem::exists(baseEnginePath), "Engine file not found: " + baseEnginePath.string());

    try
    {
        std::optional<int32_t> const specDecodeBaseOutputHiddenDim = artifacts.deployment.specConfig.has_value()
            ? std::optional<int32_t>{artifacts.deployment.specConfig->baseOutputHiddenDim}
            : std::nullopt;
        artifacts.baseExecutor
            = EngineExecutor::createForLLM(baseEnginePath, artifacts.deployment.base, specDecodeBaseOutputHiddenDim);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize base EngineExecutor: %s", e.what());
        throw std::runtime_error("Failed to initialize base EngineExecutor: " + std::string(e.what()));
    }
    LOG_INFO("Base EngineExecutor successfully loaded from %s.", baseEnginePath.c_str());

    artifacts.weights.validateAgainstEngine(*artifacts.baseExecutor, "base");
    validateAgainstEngine(artifacts.deployment.base, *artifacts.baseExecutor, "base");
    validateCachedBlockDraftTreeMetadataBindings(artifacts.deployment, *artifacts.baseExecutor);
    validateMtpTreeMetadataBindings(artifacts.deployment, *artifacts.baseExecutor);

    // Validate the draft engine ABI before its sidecar geometry is used to allocate
    // physical cache resources. Ownership is transferred to the selected decoder.
    if (draftingConfig.has_value())
    {
        artifacts.draftExecutor = decoder_utils::loadDraftEngine(engineRoot, artifacts.deployment);
        std::filesystem::path const draftConfig = engineRoot / "draft_config.json";
        if (artifacts.deployment.specDecodeMode() == SpecDecodeMode::kMTP)
        {
            artifacts.draftWeights.load(engineRoot, draftConfig, stream, artifacts.checkpointDir);
        }
        else
        {
            artifacts.draftWeights.load(
                engineRoot, draftConfig, stream, artifacts.draftCheckpointDir, artifacts.checkpointDir);
        }
        std::string const draftLabel
            = std::string{specDecodeModeName(artifacts.deployment.specDecodeMode())} + " draft";
        artifacts.draftWeights.validateAgainstEngine(*artifacts.draftExecutor, draftLabel);
    }

    if (artifacts.deployment.base.reducedVocabSize > 0)
    {
        std::filesystem::path const vocabMapPath = engineRoot / binding_names::kVocabMapFileName;
        std::vector<rt::Tensor> vocabMapTensors;
        ELLM_CHECK(safetensors::loadSafetensors(vocabMapPath, vocabMapTensors, stream),
            "Failed to load " + std::string(binding_names::kVocabMapFileName) + " from model directory: " + engineDir);
        check::check(vocabMapTensors.size() == 1,
            std::string(binding_names::kVocabMapFileName) + " should contain exactly one tensor");
        check::check(vocabMapTensors[0].getShape().getNumDims() == 1, "vocab_map tensor should be 1D");
        check::check(vocabMapTensors[0].getShape()[0] == artifacts.deployment.base.reducedVocabSize,
            "vocab_map tensor length should match base model reduced vocab size");
        check::check(vocabMapTensors[0].getDataType() == DataType::kINT32, "vocab_map tensor should be INT32");
        artifacts.vocabMap = std::move(vocabMapTensors[0]);
    }

    initializeCommon(std::move(artifacts), engineDir, multimodalEngineDir, loraWeightsMap, draftingConfig, stream,
        mapping, tokenizer, contextCacheConfig);
}

void LLMRankRuntime::initializeCommon(ModelArtifacts&& artifacts, std::string const& engineDir,
    std::string const& multimodalEngineDir, std::unordered_map<std::string, std::string> const& loraWeightsMap,
    std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream, ParallelMapping const& mapping,
    tokenizer::Tokenizer& tokenizer, ContextCacheConfig const& contextCacheConfig)
{
    mMapping = mapping;
    mTokenizer = &tokenizer;
    ELLM_CHECK(mMapping.tensorParallelSize > 0, "tensorParallelSize must be positive");
    ELLM_CHECK(mMapping.tensorParallelRank >= 0 && mMapping.tensorParallelRank < mMapping.tensorParallelSize,
        "tensorParallelRank must be in [0, tensorParallelSize)");

    mDeployment = std::move(artifacts.deployment);
    mBaseExecutor = std::move(artifacts.baseExecutor);
    mEmbedding = std::move(artifacts.embedding);
    mCheckpointDir = std::move(artifacts.checkpointDir);
    mDraftCheckpointDir = std::move(artifacts.draftCheckpointDir);
    ExternalWeightManager preparedWeights = std::move(artifacts.weights);
    std::unique_ptr<EngineExecutor> draftExecutor = std::move(artifacts.draftExecutor);
    ExternalWeightManager draftWeights = std::move(artifacts.draftWeights);
    auto pleEmbedding = std::move(artifacts.pleEmbedding);
    auto vocabMap = std::move(artifacts.vocabMap);

    ELLM_CHECK(mBaseExecutor != nullptr, "Model artifacts require a base engine executor.");
    std::optional<ContextCacheDeploymentProfile> contextCacheDeploymentProfile;
    if (contextCacheConfig.enabled)
    {
        contextCacheDeploymentProfile = validateContextCacheDeployment(mDeployment);
    }

    ELLM_CHECK(mDeployment.base.isDiffusionBackbone || mDeployment.base.numDeepstackFeatures <= 0
            || !multimodalEngineDir.empty(),
        "--multimodalEngineDir is required for VLM engine.");

    // -----------------------------------------------------------------------
    // 5. Set runtime batch size.
    // -----------------------------------------------------------------------
    mMaxRuntimeBatchSize = mDeployment.maxRuntimeBatchSize();
    LOG_INFO("Runtime batch size set to: %d (from engine bundle)", mMaxRuntimeBatchSize);

    // -----------------------------------------------------------------------
    // 6. SharedResources + PipelineIO. PipelineIO is held via unique_ptr so
    //    its address is stable for the TensorMap pointers below (TensorMap
    //    stores non-owning Tensor* into PipelineIO members).
    // -----------------------------------------------------------------------
    bool const hasDraft = draftingConfig.has_value();
    if (hasDraft)
    {
        mSharedResources
            = SharedResources::createForSpecDecode(mDeployment, mMaxRuntimeBatchSize, loraWeightsMap, stream);
        mPipelineIO = std::make_unique<PipelineIO>(PipelineIO::createForSpecDecode(
            mDeployment, mMaxRuntimeBatchSize, stream, mBaseExecutor->hasIOTensor(binding_names::kAcceptHiddenStates)));
    }
    else
    {
        mSharedResources = SharedResources::createForLLM(mDeployment.base, loraWeightsMap, stream);
        mPipelineIO = std::make_unique<PipelineIO>(PipelineIO::createForLLM(mDeployment.base, stream));
    }
    *mSharedResources->externalWeightManager = std::move(preparedWeights);

    // -----------------------------------------------------------------------
    // 7. Build base TensorMap (kvCacheIndex=0) and publish static external
    //    weight bindings. Speculative decoders add tree-mask / position IDs
    //    to this same map further down.
    // -----------------------------------------------------------------------
    if (mDeployment.base.isDiffusionBackbone)
    {
        buildTensorMapForDiffusionBackbone(
            mBaseTensorMap, *mPipelineIO, *mSharedResources, mDeployment.base, /*kvCacheIndex=*/0);
    }
    else
    {
        buildTensorMap(mBaseTensorMap, *mPipelineIO, *mSharedResources, mDeployment.base, /*kvCacheIndex=*/0);
    }
    mSharedResources->externalWeightManager->registerTensorMapEntries(mBaseTensorMap);

    // -----------------------------------------------------------------------
    // 8. LoRA: register engine bindings and seed the base tensor map with
    //    dummy / active adapter tensors. Only the base engine carries LoRA
    //    bindings — draft does not.
    // -----------------------------------------------------------------------
    if (mSharedResources->loraManager)
    {
        mSharedResources->loraManager->initializeEngineBindings(*mBaseExecutor);
        mSharedResources->loraManager->refreshTensorMap(mBaseTensorMap);
    }

    // -----------------------------------------------------------------------
    // 9. Preprocessors.
    // -----------------------------------------------------------------------
    mStepPreparer = std::make_unique<StepPreparer>(mDeployment.base);
    mEmbeddingPre = std::make_unique<EmbeddingPreprocessor>(mEmbedding, mDeployment.base);
    if (!mDeployment.base.isDiffusionBackbone && mDeployment.base.numDeepstackFeatures > 0)
    {
        mDeepstack = std::make_unique<DeepstackBinding>(mPipelineIO->deepstackEmbeds, mSharedResources->zeroBuffer);
    }

    // -----------------------------------------------------------------------
    // 10. Allocate runtime-local tensors (sampling workspace, host pinned scratch,
    //     batch-eviction mapping). Strategy-specific tensors are owned by strategies.
    // -----------------------------------------------------------------------
    int32_t const effectiveMaxProposalSize = hasDraft ? mDeployment.effectiveMaxDraftProposalSize() : 1;
    int32_t const effectiveDraftTopK = hasDraft ? draftingConfig->draftingTopK : 1;
    int32_t const diffusionCanvasLen
        = mDeployment.base.isDiffusionBackbone ? std::max(1, mDeployment.base.diffusionCanvasLength) : 1;
    int32_t const maxInputLength = hasDraft
        ? std::max(mDeployment.base.maxSupportedInputLength, mDeployment.draft->maxSupportedInputLength)
        : std::max(mDeployment.base.maxSupportedInputLength, diffusionCanvasLen);
    int32_t const diffusionSamplingSize = mMaxRuntimeBatchSize * diffusionCanvasLen;
    int32_t const maxSamplingSize = hasDraft ? std::max(mMaxRuntimeBatchSize * effectiveMaxProposalSize,
                                                   mMaxRuntimeBatchSize * effectiveDraftTopK * effectiveDraftTopK)
                                             : diffusionSamplingSize;

    // Reserve enough workspace for sampling, accounting for batch dimension in draft proposal stage.
    // Always include vanilla sampling workspace size because per-request disable_spec_decode
    // can fall back to topK/topP sampling even when draft is loaded.
    int32_t const vanillaSamplingRows = mDeployment.base.isDiffusionBackbone ? 0 : mMaxRuntimeBatchSize;
    // DiffusionGemma uses BlockDiffusionDecoder's custom sampler kernels on full-canvas logits and does not use the
    // generic vanilla selectAllTopK/topKtopP workspace. Keeping that workspace sized to B*C rows would reserve about
    // 1 GiB for B=4, canvas=256, vocab=262144 with no runtime consumer.
    size_t const vanillaSamplingWorkspaceSize = mDeployment.base.isDiffusionBackbone
        ? 0U
        : getTopKtopPSamplingWorkspaceSize(vanillaSamplingRows, mDeployment.base.outputVocabSize,
              SamplingParams(vanillaSamplingRows, mDeployment.base.outputVocabSize, 1.0f, 0, 0.9f));
    bool const isDSparkDraft = hasDraft && mDeployment.specDecodeMode() == SpecDecodeMode::kDSpark;
    constexpr int32_t kDSparkMaxSparseTopK = 128;
    bool const isLinearBlockDraft = hasDraft
        && (isCachedBlockDraftMode(mDeployment.specDecodeMode())
            || mDeployment.specDecodeMode() == SpecDecodeMode::kDSpark);
    int32_t const draftSamplingRows = isLinearBlockDraft ? mMaxRuntimeBatchSize * mDeployment.specConfig->verifySize
                                                         : mMaxRuntimeBatchSize * effectiveDraftTopK;
    int32_t const draftSamplingTopK
        = isLinearBlockDraft ? (isDSparkDraft ? kDSparkMaxSparseTopK : 1) : effectiveDraftTopK;
    size_t const dsparkBaseTopKWorkspaceSize = isDSparkDraft
        ? getSelectAllTopKWorkspaceSize(mMaxRuntimeBatchSize * mDeployment.specConfig->verifySize,
              mDeployment.base.outputVocabSize, kDSparkMaxSparseTopK)
        : 0;
    // DiffusionGemma logprobs require B*canvasLen rows, which is a GiB-scale log-softmax workspace for 26B.
    // Keep that allocation off the default serving path and grow it lazily only for numLogprobs requests.
    mLogprobsMaxBatchDim
        = mDeployment.base.isDiffusionBackbone ? 0 : mMaxRuntimeBatchSize * mDeployment.maxAcceptedTokensPerRound();
    size_t const logprobsWorkspaceSize = mLogprobsMaxBatchDim > 0
        ? getExtractTopKLogprobsWorkspaceSize(mLogprobsMaxBatchDim, mDeployment.base.outputVocabSize, kMaxLogprobsK)
        : 0U;
    size_t const maxSamplingWorkspaceSize = hasDraft
        ? std::max({vanillaSamplingWorkspaceSize,
              getSelectAllTopKWorkspaceSize(vanillaSamplingRows, mDeployment.base.outputVocabSize, 1),
              getSelectAllTopKWorkspaceSize(draftSamplingRows, mDeployment.draft->outputVocabSize, draftSamplingTopK),
              dsparkBaseTopKWorkspaceSize, logprobsWorkspaceSize})
        : std::max(vanillaSamplingWorkspaceSize, logprobsWorkspaceSize);
    check::check(maxSamplingWorkspaceSize <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "Sampling workspace size exceeds tensor dimension range");
    int64_t const maxSamplingWorkspaceLen = static_cast<int64_t>(std::max<size_t>(maxSamplingWorkspaceSize, 1U));

    try
    {
        mIdsInput = rt::Tensor({mMaxRuntimeBatchSize, maxInputLength}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMRankRuntime::mIdsInput");

        mSamplingWorkspace = rt::Tensor(
            {maxSamplingWorkspaceLen}, rt::DeviceType::kGPU, DataType::kINT8, "LLMRankRuntime::mSamplingWorkspace");
        mSamplingIndices
            = rt::Tensor({maxSamplingSize}, rt::DeviceType::kGPU, DataType::kINT32, "LLMRankRuntime::mSamplingIndices");
        mSamplingScores
            = rt::Tensor({maxSamplingSize}, rt::DeviceType::kGPU, DataType::kFLOAT, "LLMRankRuntime::mSamplingScores");
        allocateLogitBias(mLogitBias, mMaxRuntimeBatchSize);

        // Batch mapping tensor for batch eviction.
        mDeviceBatchMapping = rt::Tensor(
            {mMaxRuntimeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32, "LLMRankRuntime::mDeviceBatchMapping");
        if (mMapping.worldSize > 1)
        {
            mDeviceCancellationStates = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32,
                "LLMRankRuntime::mDeviceCancellationStates");
            mHostCancellationStates = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32,
                "LLMRankRuntime::mHostCancellationStates");
        }

        mHostPackedTokenIds = rt::Tensor({mMaxRuntimeBatchSize, maxInputLength}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMRankRuntime::mHostPackedTokenIds");
        mHostSelectedTokenIds = rt::Tensor(
            {maxSamplingSize}, rt::DeviceType::kCPU, DataType::kINT32, "LLMRankRuntime::mHostSelectedTokenIds");
        mHostReuseKVCacheLengths = rt::Tensor(
            {mMaxRuntimeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32, "LLMRankRuntime::mHostReuseKVCacheLengths");

        // Pre-allocate multimodal indices tensor (used for audio/vision embedding lookup).
        mMultimodalIndices = rt::Tensor({mMaxRuntimeBatchSize, maxInputLength}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMRankRuntime::mMultimodalIndices");
        mHostMultimodalIndices = rt::Tensor({mMaxRuntimeBatchSize, maxInputLength}, rt::DeviceType::kCPU,
            DataType::kINT32, "LLMRankRuntime::mHostMultimodalIndices");

        if (mLogprobsMaxBatchDim > 0)
        {
            ensureLogprobsCapacity(mLogprobsMaxBatchDim, kMaxLogprobsK);
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to allocate runtime tensors: %s", e.what());
        throw std::runtime_error("Failed to allocate runtime tensors: " + std::string(e.what()));
    }
    if (mDeployment.base.pleEnabled)
    {
        int32_t const maxPleSeqLen = std::max(maxInputLength, std::max(1, mDeployment.base.maxVerifyTreeSize));
        mGemma4Ple = std::make_unique<Gemma4EmbeddingPreprocessor>(std::filesystem::path(engineDir), mDeployment.base,
            mMaxRuntimeBatchSize, maxPleSeqLen, mBaseTensorMap, stream, std::move(pleEmbedding));
    }
    LOG_INFO("Runtime tensors successfully allocated.");

    // -----------------------------------------------------------------------
    // 11. Load optional base model reduced-vocab mapping table.
    // -----------------------------------------------------------------------
    if (vocabMap.has_value())
    {
        mBaseVocabMappingTable = std::move(*vocabMap);
        setLogitBiasVocabMap(
            mLogitBias, mBaseVocabMappingTable, mDeployment.base.vocabSize, mDeployment.base.reducedVocabSize, stream);
    }

    // -----------------------------------------------------------------------
    // 12. Decoding strategies.
    // -----------------------------------------------------------------------
    buildDecodingRuntimeContext();
    mDecoderRegistry = std::make_unique<DecoderRegistry>(*mDecodingRuntimeContext,
        DecoderRegistryInit{std::filesystem::path(engineDir), draftingConfig, std::move(draftExecutor),
            std::move(draftWeights), stream});

    // -----------------------------------------------------------------------
    // 13. Optional multimodal runners.
    // -----------------------------------------------------------------------
    if (!multimodalEngineDir.empty())
    {
        // A missing engine file means the deployment simply has no such
        // encoder. One that is present but fails to load is fatal: continuing
        // would answer image and audio prompts from the text tokens alone.
        auto loadRunner = [&](std::string const& dir, std::string const& engineFile,
                              std::string const& name) -> std::unique_ptr<MultimodalRunner> {
            if (!std::filesystem::exists(std::filesystem::path(dir) / engineFile))
            {
                LOG_DEBUG("No %s engine at %s/%s", name.c_str(), dir.c_str(), engineFile.c_str());
                return nullptr;
            }
            LOG_DEBUG("Attempting to load %s runner from %s", name.c_str(), dir.c_str());
            auto runner = MultimodalRunner::create(dir, mDeployment.base.maxSupportedBatchSize,
                mDeployment.base.maxKVCacheCapacity, stream, mCheckpointDir);
            LOG_INFO("%s runner successfully initialized", name.c_str());
            return runner;
        };

        mAudioRunner = loadRunner(multimodalEngineDir + "/audio", "audio_encoder.engine", "Audio");
        mVisionRunner = loadRunner(multimodalEngineDir + "/visual", "visual.engine", "Visual");
        if (!mVisionRunner)
        {
            mVisionRunner = loadRunner(multimodalEngineDir, "visual.engine", "Vision");
        }

        // At least one multimodal runner must be available
        ELLM_CHECK(mAudioRunner || mVisionRunner, "No valid multimodal engine found in " + multimodalEngineDir);

        // Try to load action expert from multimodalEngineDir/action
        try
        {
            std::string actionDir = multimodalEngineDir + "/action";
            LOG_INFO("Attempting to load Action runner from %s", actionDir.c_str());
            auto actionRunner = std::make_unique<Alpamayo1ActionRunner>(
                actionDir, mCheckpointDir, stream, mSharedResources->cacheManagers[0]->getKVCacheManager().getConfig());
            auto const& basePageTable = *mSharedResources->kvPageTables[0];
            auto actionKvBatchCollector = std::make_unique<ActionKvBatchCollector>(
                mMaxRuntimeBatchSize, basePageTable.maxPagesPerSeq(), basePageTable.numPages());
            mActionRunner = std::move(actionRunner);
            mActionKvBatchCollector = std::move(actionKvBatchCollector);
            LOG_INFO("Alpamayo 1 action expert loaded.");
        }
        catch (std::exception const& e)
        {
            LOG_INFO("Failed to load Action runner from %s: %s", (multimodalEngineDir + "/action").c_str(), e.what());
        }

        // Validate that the action engine's max KV cache capacity matches the LLM engine's.
        if (mActionRunner)
        {
            int32_t const actionMaxKVCacheCapacity = mActionRunner->getMaxKVCacheCapacity();
            int32_t const llmMaxKVCacheCapacity = mDeployment.base.maxKVCacheCapacity;
            ELLM_CHECK(actionMaxKVCacheCapacity == llmMaxKVCacheCapacity,
                format::fmtstr(
                    "Action engine max_kv_cache_capacity (%d) does not match LLM engine max_kv_cache_capacity (%d). "
                    "Re-export and rebuild the action engine with --max_kv_cache_capacity=%d to match the LLM engine.",
                    actionMaxKVCacheCapacity, llmMaxKVCacheCapacity, llmMaxKVCacheCapacity));
        }
    }

    if (contextCacheConfig.enabled)
    {
        ELLM_CHECK(mActionRunner == nullptr, "Context reuse cannot be enabled with action runners.");
        ELLM_CHECK(!mSharedResources->cacheManagers.empty()
                && mSharedResources->cacheManagers.size() == mSharedResources->kvPageTables.size()
                && mSharedResources->cacheManagers.size() <= 2,
            "Context reuse requires one base cache and at most one draft cache.");
        HybridCacheManager* const draftCache
            = mSharedResources->cacheManagers.size() == 2 ? mSharedResources->cacheManagers[1].get() : nullptr;
        KVPageTable* const draftPageTable
            = mSharedResources->kvPageTables.size() == 2 ? mSharedResources->kvPageTables[1].get() : nullptr;
        ContextCachePhysicalResources cacheResources{
            *mSharedResources->cacheManagers[0], *mSharedResources->kvPageTables[0], draftCache, draftPageTable};
        ELLM_CHECK(contextCacheDeploymentProfile.has_value(), "Context-cache deployment was not validated");
        mContextCache = std::make_unique<ContextCacheCoordinator>(
            contextCacheConfig, mDeployment, *contextCacheDeploymentProfile, cacheResources, stream);

        mHybridMtpContextReuseDeployment
            = contextCacheDeploymentProfile->isHybrid() && contextCacheDeploymentProfile->isSpeculative();
        if (contextCacheDeploymentProfile->isHybrid() && contextCacheDeploymentProfile->isSpeculative())
        {
            rt::Coords const bhShape = mPipelineIO->baseHiddenStates.getShape();
            mBoundaryFoldScratch = rt::Tensor({bhShape[1], bhShape[2]}, rt::DeviceType::kGPU,
                mPipelineIO->baseHiddenStates.getDataType(), "LLMRankRuntime::mBoundaryFoldScratch");
            mBoundaryFoldMaxRows = math::cast<int32_t>(bhShape[1]);
        }
    }

    // -----------------------------------------------------------------------
    // 14. Shared execution context memory for all engines (base, optional
    //     draft, and optional vision/audio). All engines execute serially so
    //     they can share a single buffer sized to the max requirement.
    // -----------------------------------------------------------------------
    int64_t const baseContextMemorySize = mBaseExecutor->getRequiredContextMemorySize();
    int64_t const strategyContextMemorySize = mDecoderRegistry ? mDecoderRegistry->getRequiredContextMemorySize() : 0;
    int64_t const visionContextMemorySize = mVisionRunner ? mVisionRunner->getRequiredContextMemorySize() : 0;
    int64_t const audioContextMemorySize = mAudioRunner ? mAudioRunner->getRequiredContextMemorySize() : 0;
    int64_t const actionContextMemorySize = mActionRunner ? mActionRunner->getRequiredContextMemorySize() : 0;
    int64_t const sharedContextMemorySize = std::max({baseContextMemorySize, strategyContextMemorySize,
        visionContextMemorySize, audioContextMemorySize, actionContextMemorySize});
    mSharedExecContextMemory = rt::Tensor({sharedContextMemorySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "LLMRankRuntime::mSharedExecContextMemory");
    mBaseExecutor->setContextMemory(mSharedExecContextMemory);
    if (mDecoderRegistry)
    {
        mDecoderRegistry->setContextMemory(mSharedExecContextMemory);
    }
    if (mVisionRunner)
    {
        mVisionRunner->setContextMemory(mSharedExecContextMemory);
    }
    if (mAudioRunner)
    {
        mAudioRunner->setContextMemory(mSharedExecContextMemory);
    }
    if (mActionRunner)
    {
        mActionRunner->setContextMemory(mSharedExecContextMemory);
    }
    LOG_INFO(
        "Setup shared execution context memory: %zu bytes (base requires: %zu, strategy requires: %zu, vision "
        "requires: "
        "%zu, audio requires: %zu, action requires: %zu)",
        static_cast<size_t>(sharedContextMemorySize), static_cast<size_t>(baseContextMemorySize),
        static_cast<size_t>(strategyContextMemorySize), static_cast<size_t>(visionContextMemorySize),
        static_cast<size_t>(audioContextMemorySize), static_cast<size_t>(actionContextMemorySize));

    // Encoder embedding cache — content-addressed GPU cache for ViT/audio encoder outputs.
    if ((mVisionRunner || mAudioRunner) && contextCacheConfig.encoderEmbeddingCacheBudgetBytes > 0)
    {
        auto const budgetBytes = contextCacheConfig.encoderEmbeddingCacheBudgetBytes;
        mEncoderEmbeddingCache = std::make_unique<EncoderEmbeddingCache>(budgetBytes);
        LOG_INFO("Encoder embedding cache initialized with %zu MiB budget.",
            static_cast<size_t>(budgetBytes / (1024 * 1024)));
    }
}

void LLMRankRuntime::ensureLogprobsCapacity(int32_t logprobsRows, int32_t topK)
{
    check::check(logprobsRows > 0, "logprobsRows must be positive when logprobs are requested.");
    check::check(topK > 0 && topK <= static_cast<int32_t>(kMaxLogprobsK), "numLogprobs is out of supported range.");

    size_t const requiredWorkspaceSize
        = getExtractTopKLogprobsWorkspaceSize(logprobsRows, mDeployment.base.outputVocabSize, topK);
    check::check(requiredWorkspaceSize <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "Logprobs workspace size exceeds tensor dimension range");
    if (mSamplingWorkspace.isEmpty()
        || static_cast<size_t>(mSamplingWorkspace.getMemoryCapacity()) < requiredWorkspaceSize)
    {
        mSamplingWorkspace = rt::Tensor({static_cast<int64_t>(requiredWorkspaceSize)}, rt::DeviceType::kGPU,
            DataType::kINT8, "LLMRankRuntime::mSamplingWorkspace");
    }

    int64_t const requiredValueBytes = static_cast<int64_t>(logprobsRows) * kMaxLogprobsK * sizeof(float);
    int64_t const requiredIndexBytes = static_cast<int64_t>(logprobsRows) * kMaxLogprobsK * sizeof(int32_t);
    if (mDeviceLogprobsValues.isEmpty() || mDeviceLogprobsValues.getMemoryCapacity() < requiredValueBytes)
    {
        mDeviceLogprobsValues = rt::Tensor({logprobsRows, kMaxLogprobsK}, rt::DeviceType::kGPU, DataType::kFLOAT,
            "LLMRankRuntime::mDeviceLogprobsValues");
        mHostLogprobsValues = rt::Tensor({logprobsRows, kMaxLogprobsK}, rt::DeviceType::kCPU, DataType::kFLOAT,
            "LLMRankRuntime::mHostLogprobsValues");
    }
    if (mDeviceLogprobsIndices.isEmpty() || mDeviceLogprobsIndices.getMemoryCapacity() < requiredIndexBytes)
    {
        mDeviceLogprobsIndices = rt::Tensor({logprobsRows, kMaxLogprobsK}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMRankRuntime::mDeviceLogprobsIndices");
        mHostLogprobsIndices = rt::Tensor({logprobsRows, kMaxLogprobsK}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMRankRuntime::mHostLogprobsIndices");
    }
    if (mDeployment.specConfig.has_value())
    {
        int64_t const requiredGatheredBytes
            = static_cast<int64_t>(logprobsRows) * mDeployment.base.outputVocabSize * sizeof(float);
        if (mGatheredLogits.isEmpty() || mGatheredLogits.getMemoryCapacity() < requiredGatheredBytes)
        {
            mGatheredLogits = rt::Tensor({logprobsRows, mDeployment.base.outputVocabSize}, rt::DeviceType::kGPU,
                DataType::kFLOAT, "LLMRankRuntime::mGatheredLogits");
        }
    }
    mLogprobsMaxBatchDim = std::max(mLogprobsMaxBatchDim, logprobsRows);
}

void LLMRankRuntime::buildDecodingRuntimeContext()
{
    ELLM_CHECK(mTokenizer != nullptr, "LLMRankRuntime requires a shared tokenizer.");
    BaseEngineResources baseResources{*mBaseExecutor, mBaseTensorMap, *mSharedResources,
        *mSharedResources->cacheManagers[0], *mPipelineIO, [this](InferenceDims const& dims, cudaStream_t stream) {
            return captureBaseGraphWithLoraFanout(dims, stream);
        }};
    PreprocessResources preprocessResources{
        *mStepPreparer, *mEmbeddingPre, mEmbedding, mIdsInput, mDeepstack.get(), mGemma4Ple.get()};
    SamplingBuffers sampling{mSamplingWorkspace, mSamplingIndices, mSamplingScores, mBaseVocabMappingTable,
        mHostPackedTokenIds, mHostSelectedTokenIds};
    LogprobsBuffers logprobs{
        mDeviceLogprobsValues, mDeviceLogprobsIndices, mHostLogprobsValues, mHostLogprobsIndices, mGatheredLogits};
    mDecodingRuntimeContext.reset(new DecodingRuntimeContext{mDeployment, mMaxRuntimeBatchSize, mCheckpointDir,
        mDraftCheckpointDir, baseResources, preprocessResources, *mTokenizer, mLogitBias, sampling, logprobs});
}

void LLMRankRuntime::setActionNoiseSeed(int32_t seed) noexcept
{
    if (mActionRunner)
    {
        mActionRunner->setNoiseSeed(seed);
    }
}

std::optional<ContextCacheMetrics> LLMRankRuntime::getContextCacheMetrics() const noexcept
{
    if (mContextCache == nullptr)
    {
        return std::nullopt;
    }
    return mContextCache->metrics();
}

void LLMRankRuntime::setVisualPrunerConfig(VisualPrunerConfig const& config)
{
    mVisualPruner.reset();
    if (!config.enabled)
    {
        return;
    }
    if (mDeployment.base.ropeConfig.type != RopeType::kMRope || mDeployment.base.imageTokenId < 0)
    {
        LOG_WARNING("Visual-token pruning requires an mRoPE VLM engine (image token id present); leaving it disabled.");
        return;
    }
    if (mDeployment.specConfig.has_value())
    {
        LOG_WARNING("Visual-token pruning is not supported together with speculative decoding; leaving it disabled.");
        return;
    }
    if (mActionRunner)
    {
        LOG_WARNING("Visual-token pruning is not supported together with an action runner; leaving it disabled.");
        return;
    }
    mVisualPruner = createVisualTokenPruner(config, mDeployment.base);
    LOG_INFO("Visual-token pruning enabled: algorithm=%s, reductionRatio=%.3f, minVisualTokens=%d",
        mVisualPruner->name(), config.reductionRatio, config.minVisualTokens);
}

bool LLMRankRuntime::broadcastInt32(void* gpuBuffer, int32_t count, cudaStream_t stream)
{
    if (!mTokenBroadcast)
    {
        return true;
    }
    return mTokenBroadcast(gpuBuffer, count, stream);
}

bool LLMRankRuntime::synchronizeCancellationStates(DecodingInferenceContext& context)
{
    if (mMapping.worldSize <= 1 || !mTokenBroadcast || mParallelRank < 0 || context.activeBatchSize <= 0)
    {
        return true;
    }

    int32_t const activeBatchSize = context.activeBatchSize;
    check::check(mDeviceCancellationStates.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mHostCancellationStates.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostStates = mHostCancellationStates.dataPointer<int32_t>();
    std::fill(hostStates, hostStates + activeBatchSize, 0);

    if (mParallelRank == 0)
    {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            hostStates[i]
                = context.finishedStates[i] && context.slotStreams[i].terminalReason == FinishReason::kCancelled ? 1
                                                                                                                 : 0;
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(mDeviceCancellationStates.rawPointer(), hostStates,
        static_cast<size_t>(activeBatchSize) * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    if (!broadcastInt32(mDeviceCancellationStates.rawPointer(), activeBatchSize, context.stream))
    {
        return false;
    }
    CUDA_CHECK(cudaMemcpyAsync(hostStates, mDeviceCancellationStates.rawPointer(),
        static_cast<size_t>(activeBatchSize) * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    // Only rank 0 owns live stream channels. Streaming requests therefore need this host-visible consensus before
    // the next decode collective; otherwise a root cancellation can let ranks enter different execution paths.
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        if (hostStates[i] != 0 && !context.finishedStates[i])
        {
            context.finishedStates[i] = 1;
            context.slotStreams[i].terminalReason = FinishReason::kCancelled;
            LOG_DEBUG("Batch %d finished after root-rank cancellation synchronization.", i);
        }
    }
    return true;
}

bool LLMRankRuntime::handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response,
    cudaStream_t stream, bool outputThinkerEmbeddings, TokenBroadcastFn tokenBroadcast, int32_t parallelRank)
{
    bool expected = false;
    if (!mHandleRequestInProgress.compare_exchange_strong(
            expected, true, std::memory_order_acquire, std::memory_order_relaxed))
    {
        LOG_ERROR("Overlapping handleRequest() calls on one runtime are not supported.");
        return false;
    }
    struct HandleRequestGuard
    {
        explicit HandleRequestGuard(std::atomic<bool>& active) noexcept
            : mActive(active)
        {
        }

        ~HandleRequestGuard() noexcept
        {
            mActive.store(false, std::memory_order_release);
        }

        std::atomic<bool>& mActive;
    };
    HandleRequestGuard const handleRequestGuard{mHandleRequestInProgress};

    mTokenBroadcast = std::move(tokenBroadcast);
    mParallelRank = parallelRank;
    if (mDecodingRuntimeContext)
    {
        mDecodingRuntimeContext->tokenBroadcast = mTokenBroadcast;
        mDecodingRuntimeContext->parallelRank = mParallelRank;
    }
    if (mParallelRank > 0 && !mTokenBroadcast)
    {
        LOG_ERROR("Parallel rank %d requires a token broadcast callback.", mParallelRank);
        return false;
    }
    // Clear per-request portal state. Buffers themselves stay allocated and are
    // reshaped/overwritten when populated below — see getBaseModelHiddenStates() contract.
    mHiddenStatesRegistry.clear();
    mLastPrefillLength = 0;
    mLastInputTokenIds.clear();

    // Clear per-request response state. On failure (early return) the four vectors
    // stay empty; on success they are repopulated together below to matched sizes.
    response.outputIds.clear();
    response.outputTexts.clear();
    response.logprobs.clear();
    response.outputTrajectories.clear();
    response.finishReasons.clear();

    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());
    std::string const& loraWeightsName = request.loraWeightsName;

    if (!validateRequestConfig(request))
    {
        return false;
    }

    if (mContextCache != nullptr && request.saveSystemPromptKVCache)
    {
        LOG_ERROR("Legacy system-prompt KV-cache capture cannot be combined with the context-cache manager.");
        return false;
    }

    if (!validateStreamingSubmission(request))
    {
        return false;
    }

    DecodingStrategy& decodingStrategy = mDecoderRegistry->select(request);
    bool const enableSpecDecode = decodingStrategy.isSpeculative();
    // DSpark implements the paper-equivalent probabilistic verifier and can keep non-greedy sampling params.
    // Other speculative decoders still run greedy-compatible verification.
    bool const hasNonGreedySampling = shouldUseNonGreedySampling(request.temperature, request.topK, request.topP);
    bool const dsparkSpecDecode = enableSpecDecode && decodingStrategy.kind() == DecodingStrategyKind::kDSpark;
    if (enableSpecDecode && hasNonGreedySampling && !dsparkSpecDecode)
    {
        LOG_WARNING("Spec-decode active: overriding sampling params to greedy (ignoring temp/topK/topP).");
    }
    if (mDeployment.specDecodeMode() == SpecDecodeMode::kEAGLE
        && decodingStrategy.kind() == DecodingStrategyKind::kVanilla && !request.disableSpecDecode
        && hasNonGreedySampling)
    {
        LOG_WARNING(
            "Decoder fallback: reason=non_greedy_eagle_unsupported, selected=vanilla, "
            "temperature=%.3f, topK=%" PRId64 ", topP=%.3f.",
            request.temperature, request.topK, request.topP);
    }

    int32_t maxGenerateLength = request.maxGenerateLength;

    if (request.formattedRequests.size() != static_cast<size_t>(activeBatchSize)
        || request.preTokenizedInputIds.size() != static_cast<size_t>(activeBatchSize))
    {
        LOG_ERROR(
            "LLMRankRuntime received an unprepared request; RuntimeCoordinator must populate formatted and "
            "pre-tokenized request state.");
        return false;
    }

    DecodingInferenceContext context;
    context.initialize(
        activeBatchSize, maxGenerateLength, std::nullopt, rt::OptionalInputTensors{}, loraWeightsName, stream);
    context.layerDebugger = LayerDebugger::fromEnv();
    bool const supportsMultimodalInput
        = (mAudioRunner != nullptr) || (mVisionRunner != nullptr) || (mActionRunner != nullptr);

    if (supportsMultimodalInput)
    {
        if (!multiModalRuntimePreprocess(request, context, stream))
        {
            return false;
        }
    }
    else
    {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            context.systemPrompts[i] = request.formattedRequests[i].formattedSystemPrompt;
            if (i < static_cast<int32_t>(request.preTokenizedInputIds.size())
                && !request.preTokenizedInputIds[i].empty())
            {
                context.rawBatchedInputIds.emplace_back(request.preTokenizedInputIds[i]);
            }
            else
            {
                context.rawBatchedInputIds.emplace_back(
                    mTokenizer->encode(request.formattedRequests[i].formattedCompleteRequest, false));
            }
            if (context.rawBatchedInputIds[i].empty())
            {
                LOG_ERROR("Failed to tokenize input text for request %d in batch", i);
                return false;
            }
        }
    }

    bool const hasActionRequest = mActionRunner != nullptr
        && std::any_of(request.requests.begin(), request.requests.end(),
            [](auto const& req) { return req.pastTrajectory.has_value(); });
    if (hasActionRequest)
    {
        ELLM_CHECK(mActionKvBatchCollector != nullptr, "Action KV batch collector was not initialized");
        if (mVisionRunner == nullptr || mVisionRunner->getModelType() != multimodal::ModelType::QWEN3_VL)
        {
            LOG_ERROR("Alpamayo1ActionRunner requires a Qwen3-VL vision runner for MRoPE deltas.");
            return false;
        }
        auto* qwenVision = static_cast<rt::QwenViTRunner*>(mVisionRunner.get());
        std::vector<int64_t> const& ropeDeltas = qwenVision->getMropeRopeDeltasPerBatch();
        if (ropeDeltas.size() != request.requests.size())
        {
            LOG_ERROR("MRoPE delta count %zu does not match request batch size %zu", ropeDeltas.size(),
                request.requests.size());
            return false;
        }
        std::vector<bool> actionSlots;
        actionSlots.reserve(request.requests.size());
        for (auto const& slot : request.requests)
        {
            actionSlots.push_back(slot.pastTrajectory.has_value());
        }
        mActionKvBatchCollector->beginRequest(actionSlots, ropeDeltas);
    }

    // Reject overlong inputs with the marker consumed by the Python server's
    // HTTP 413 mapping, before TensorRT reports a less actionable shape error.
    for (size_t i = 0; i < context.rawBatchedInputIds.size(); ++i)
    {
        int32_t const inputLen = static_cast<int32_t>(context.rawBatchedInputIds[i].size());
        if (inputLen > mDeployment.base.maxSupportedInputLength)
        {
            LOG_ERROR(
                "Input length (%d) exceeds engine max input length (%d). "
                "Rebuild the engine with a larger --maxInputLen.",
                inputLen, mDeployment.base.maxSupportedInputLength);
            throw std::runtime_error("EDGELLM_INPUT_TOO_LONG: input length " + std::to_string(inputLen)
                + " exceeds engine max_input_len " + std::to_string(mDeployment.base.maxSupportedInputLength)
                + " (rebuild engine with a larger --maxInputLen)");
        }
    }

    // Forward sampling params to context; non-DSpark speculative decoders run greedy.
    bool const forceGreedySpecDecode = enableSpecDecode && !dsparkSpecDecode;
    context.temperature = forceGreedySpecDecode ? 1.0f : request.temperature;
    context.topP = forceGreedySpecDecode ? 1.0f : request.topP;
    context.topK = forceGreedySpecDecode ? 0 : request.topK;
    context.diffusionMaxDenoisingSteps = request.diffusionMaxDenoisingSteps;
    context.outputThinkerEmbeddings = outputThinkerEmbeddings;
    context.onTokenGenerated = request.onTokenGenerated;

    prepareLogitBias(mLogitBias, request, context);

    if (request.numLogprobs > static_cast<int32_t>(kMaxLogprobsK))
    {
        LOG_WARNING("numLogprobs %d exceeds maximum %d; clamping.", request.numLogprobs, kMaxLogprobsK);
    }
    context.numLogprobs = std::min(request.numLogprobs, static_cast<int32_t>(kMaxLogprobsK));
    if (context.numLogprobs > 0)
    {
        int32_t const logprobsRows = activeBatchSize * mDeployment.maxAcceptedTokensPerRound();
        ensureLogprobsCapacity(logprobsRows, context.numLogprobs);

        // Spec-decode verify may accept more than 1 token in one step, overshooting maxGenerateLength.
        int32_t const overshoot = mDeployment.maxAcceptedTokensPerRound() - 1;
        for (auto& slot : context.stepLogprobs)
        {
            slot.data.resize(static_cast<size_t>(context.maxGenerateLength + overshoot) * context.numLogprobs);
            slot.numSteps = 0;
        }
    }

    // Forward per-slot stop strings and cache the longest length to avoid
    // recomputing it on every emitChunks iteration.
    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        context.stopStringsPerSlot[i] = request.requests[i].stopStrings;
        size_t maxLen = 0;
        for (auto const& s : request.requests[i].stopStrings)
        {
            if (s.size() > maxLen)
            {
                maxLen = s.size();
            }
        }
        context.slotStreams[i].maxStopLen = maxLen;
    }

    DecodingKvHeadroom const kvHeadroom = decodingStrategy.requiredKvHeadroom();
    ELLM_CHECK(
        kvHeadroom.baseExtraTokens > 0 && kvHeadroom.draftExtraTokens >= 0, "Decoder returned invalid KV headroom");
    ELLM_CHECK(!enableSpecDecode || kvHeadroom.draftExtraTokens == 0 || mDeployment.draft.has_value(),
        "Decoder requested draft KV headroom without a draft engine");

    // In production, the system-prompt KV cache is saved during warm-up.
    // We disable profiling here to make benchmarking closer to production inference result.
    bool profilingEnabled = getProfilingEnabled();
    if (profilingEnabled)
    {
        setProfilingEnabled(false);
    }

    // Generate system prompt KVCache for each sequence in the batch
    if (request.saveSystemPromptKVCache)
    {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            bool const saveCacheStatus = genAndSaveSystemPromptKVCache(context, i);
            if (!saveCacheStatus)
            {
                LOG_WARNING(
                    "Failed to save system prompt KVCache for request %d in batch. "
                    "Continue to handle the request without saving the system prompt KVCache.",
                    i);
            }
        }
    }

    if (profilingEnabled)
    {
        setProfilingEnabled(true);
    }

    // Collect valid media placeholder token IDs for content-addressed cache hashing.
    std::vector<int32_t> mediaTokenIds;
    if (mDeployment.base.imageTokenId >= 0)
    {
        mediaTokenIds.push_back(mDeployment.base.imageTokenId);
    }
    if (mDeployment.base.audioTokenId >= 0)
    {
        mediaTokenIds.push_back(mDeployment.base.audioTokenId);
    }

    std::optional<ContextCacheRequest> contextCacheRequest;
    if (mContextCache != nullptr)
    {
        std::optional<ContextCacheRequest> admitted = ContextCacheRequest::begin(
            *mContextCache, request, context, decodingStrategy.isSpeculative(), kvHeadroom, mediaTokenIds);
        if (!admitted.has_value())
        {
            return false;
        }
        contextCacheRequest.emplace(std::move(*admitted));
    }
    ContextCacheRequest* const managedRequest = contextCacheRequest.has_value() ? &*contextCacheRequest : nullptr;

    // Conduct the preparation work to handle a new set of sequences, including inputIds packing, input/output tensor
    // preparation, reset the KVCache state, and apply reused prefix KVCache if available.
    std::vector<int32_t> const* const contextCachePrefillStarts
        = managedRequest != nullptr ? &managedRequest->prefillStarts() : nullptr;
    if (!setUpForPrefillExecution(context, decodingStrategy, contextCachePrefillStarts))
    {
        LOG_ERROR("Prefill execution setup failed. This request cannot be handled.");
        return false;
    }
    if (managedRequest != nullptr && !managedRequest->preparePrefill())
    {
        return false;
    }

    // ── Streaming setup ──────────────────────────────────────────────────────
    // Attach first, record in slotStreams only on success — a throw from attach
    // keeps foreign channels out of the finalizer's reach. Seed sentTokenCount
    // to the prompt length so streaming emits only generated tokens.
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        if (static_cast<size_t>(i) < context.callbackEmittedTokenCounts.size())
        {
            context.callbackEmittedTokenCounts[i] = static_cast<int32_t>(context.tokenIds[i].size());
        }
        if (request.streamChannels.empty() || !request.streamChannels[i])
        {
            continue;
        }
        attachStreamChannel(request.streamChannels[i], context.batchIndexMapping[i]);
        auto& slot = context.slotStreams[i];
        slot.channel = request.streamChannels[i];
        slot.sentTokenCount = context.tokenIds[i].size();
        slot.lastEmittedTokenCount = slot.sentTokenCount;
    }
    StreamChannelFinalizer streamFinalizer(context, *mTokenizer);

    // Visual-token pruning is data-dependent and runs inside prefill after embeddings are assembled. Use the
    // unpruned resident endpoint here so prefill is always capacity-safe; this may conservatively reduce generation
    // length for a request that is subsequently pruned.
    std::vector<int32_t> residentInputLengths;
    residentInputLengths.reserve(context.rawBatchedInputIds.size());
    for (std::vector<int32_t> const& tokenIds : context.rawBatchedInputIds)
    {
        residentInputLengths.push_back(static_cast<int32_t>(tokenIds.size()));
    }
    int32_t clampedMaxGenerateLength = clampMaxGenerateLengthForKVCapacity(residentInputLengths,
        request.maxGenerateLength, mDeployment.base.maxKVCacheCapacity, kvHeadroom.baseExtraTokens);
    if (kvHeadroom.draftExtraTokens > 0)
    {
        ELLM_CHECK(mDeployment.draft.has_value(), "Draft KV headroom requires a draft engine");
        clampedMaxGenerateLength = clampMaxGenerateLengthForKVCapacity(residentInputLengths, clampedMaxGenerateLength,
            mDeployment.draft->maxKVCacheCapacity, kvHeadroom.draftExtraTokens);
    }
    if (clampedMaxGenerateLength != context.maxGenerateLength)
    {
        context.maxGenerateLength = clampedMaxGenerateLength;
        LOG_WARNING("Reduce max generation length to %d", context.maxGenerateLength);
    }
    if (context.maxGenerateLength <= 0)
    {
        LOG_ERROR("Insufficient KV cache capacity for generation for this request.");
        return false;
    }

    // Hybrid+MTP endpoint reuse recomputes the successor-dependent boundary draft slot from a saved base hidden state,
    // so a checkpoint reuses across turns regardless of the token that follows it. Enabled only when the request looks
    // up or publishes state (not a bypass request) and the deployment is MTP over a hybrid base.
    bool const requestCacheEnabled = contextCacheRequest.has_value()
        && request.contextCacheLookupPolicy != ContextCacheLookupPolicy::kBypass && !request.generateAudio
        && !context.outputThinkerEmbeddings;
    bool const hybridMtpContextReuse = shouldUseHybridMtpEndpointReuse(
        decodingStrategy.kind(), mHybridMtpContextReuseDeployment, requestCacheEnabled, requestCacheEnabled);
    if (hybridMtpContextReuse)
    {
        bool const textOnly = !context.visualEmbeddings.has_value() && !context.audioEmbeddings.has_value();
        if (context.activeBatchSize != 1 || !textOnly || request.recurrentCaptureInterval != 0
            || request.contextCacheCommitPolicy != ContextCacheCommitPolicy::kPrefillStateOnly)
        {
            LOG_ERROR(
                "Hybrid MTP context reuse requires a text-only batch of one, endpoint-only capture, and "
                "PREFILL_STATE_ONLY commit policy");
            return false;
        }
    }

    // Prefill from the base model; subsequent iterations are delegated to the selected strategy.
    bool prefillStatus;
    if (hybridMtpContextReuse)
    {
        context.hybridMtpEndpointReuse = true;
        context.contextCacheReplayTailLength = request.contextCacheReplayTailLength;
        prefillStatus = runHybridMtpPrefill(context, decodingStrategy, *contextCacheRequest);
    }
    else
    {
        prefillStatus = runBaseModelPrefill(context, managedRequest);
    }
    if (!prefillStatus)
    {
        LOG_ERROR("Failed to execute prefill step for base model.");
        return false;
    }

    if (managedRequest != nullptr && !decodingStrategy.initializeForGeneration(context))
    {
        LOG_ERROR("Failed to initialize generation state for %s decoding strategy.", decodingStrategy.name());
        return false;
    }

    std::vector<int32_t> const& commonMaterializedStateLengths = decodingStrategy.commonMaterializedStateLengths();
    if (managedRequest != nullptr && !managedRequest->completePrefill(context, commonMaterializedStateLengths))
    {
        return false;
    }

    // Streaming consumers (e.g. the Qwen3-Omni Talker) run concurrently with
    // the base model's decode loop and read the prefill-time input embeddings
    // and engine hidden_states output. Copy both into `streamingPrefill`
    // between prefill and the first decode step — the live PipelineIO buffers
    // are reshaped to `{B, 1, H}` and overwritten by every decode iteration.
    if (outputThinkerEmbeddings)
    {
        ELLM_CHECK(!mPipelineIO->outputHiddenStates.isEmpty(),
            std::string("Thinker hidden-state capture requested but the base engine exposes no "
                        "accept-layer output; re-export the thinker so it emits '")
                + binding_names::kAcceptHiddenStates + "'.");
        // -1 is the documented "use the post-norm output" sentinel and is fine,
        // but 0 is the slot the input embeddings are written to just below, so
        // it would silently clobber them.
        ELLM_CHECK(
            request.acceptHiddenLayer != 0, "acceptHiddenLayer 0 collides with the input-embeddings registry slot.");
        int32_t const prefillSequenceLength
            = *std::max_element(context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.end());
        mPipelineIO->streamingPrefill.populateFromPrefill(mPipelineIO->inputsEmbeds, mPipelineIO->outputHiddenStates,
            activeBatchSize, prefillSequenceLength, mDeployment.base.hiddenSize, mMaxRuntimeBatchSize,
            mDeployment.base.maxSupportedInputLength, stream);
        mLastPrefillLength = prefillSequenceLength;
        mLastInputTokenIds = context.rawBatchedInputIds;
        mHiddenStatesRegistry[0] = &mPipelineIO->streamingPrefill.inputEmbeds;
        mHiddenStatesRegistry[request.acceptHiddenLayer] = &mPipelineIO->streamingPrefill.engineHiddenStates;
    }

    // Lambda to check if all batches are finished
    auto checkAllFinished = [&]() {
        // Check if all batches have been evicted
        if (context.activeBatchSize == 0)
        {
            return true;
        }
        for (int32_t i = 0; i < context.activeBatchSize; ++i)
        {
            if (!context.finishedStates[i])
            {
                return false;
            }
        }
        return true;
    };

    // Used for Alpamayo 1
    int32_t trajFutureStartId = 0;
    if (mActionRunner && mActionRunner->getModelType() == action::ActionModelType::ALPAMAYO1)
    {
        trajFutureStartId = static_cast<int32_t>(mTokenizer->getTokenId("<|traj_future_start|>"));
    }

    // Once thinking is complete (or the model never entered thinking),
    // secondary EOS tokens terminate generation normally.
    std::vector<int8_t> thinkingDone(context.activeBatchSize, 0);
    int32_t const endOfChannelId = static_cast<int32_t>(mTokenizer->getTokenId("<channel|>"));
    int32_t const endOfThinkId = static_cast<int32_t>(mTokenizer->getTokenId("</think>"));
    int32_t const startOfChannelId = static_cast<int32_t>(mTokenizer->getTokenId("<|channel>"));
    int32_t const startOfThinkId = static_cast<int32_t>(mTokenizer->getTokenId("<think>"));

    auto updateThinkingDoneForToken = [&](int32_t batchIdx, int32_t tokenId) {
        if (!request.enableThinking || thinkingDone[batchIdx])
        {
            return;
        }
        if (tokenId == endOfChannelId || tokenId == endOfThinkId)
        {
            thinkingDone[batchIdx] = true;
        }
        else if (context.currentGenerateLengths[batchIdx] == 1 && tokenId != startOfChannelId
            && tokenId != startOfThinkId)
        {
            thinkingDone[batchIdx] = true;
            LOG_DEBUG("Batch %d: first token %d is not thinking-start, marking thinkingDone", batchIdx, tokenId);
        }
    };

    auto updateThinkingDone = [&]() {
        if (!request.enableThinking)
        {
            return;
        }
        for (int32_t i = 0; i < context.activeBatchSize; ++i)
        {
            if (!context.tokenIds[i].empty())
            {
                updateThinkingDoneForToken(i, context.tokenIds[i].back());
            }
        }
    };

    // Few-layer-validation / fixed-output perf: when EDGELLM_IGNORE_EOS is set,
    // suppress EOS-based termination so the run produces exactly maxGenerateLength
    // tokens. This also applies to multi-token accept paths such as DiffusionGemma
    // canvas commit; otherwise EOS inside a canvas can truncate a fixed-output block.
    bool const ignoreEos = []() {
        char const* value = std::getenv("EDGELLM_IGNORE_EOS");
        return value != nullptr && std::string(value) != "0" && std::string(value) != "false";
    }();
    if (ignoreEos)
    {
        LOG_INFO("EDGELLM_IGNORE_EOS set: ignoring EOS; running to maxGenerateLength.");
    }

    context.shouldStopAfterAcceptedToken = [&](int32_t batchIdx, int32_t tokenId) {
        updateThinkingDoneForToken(batchIdx, tokenId);
        bool isEos = !ignoreEos && mTokenizer->isEosToken(tokenId);
        if (isEos && request.enableThinking && tokenId != mTokenizer->getEosId() && !thinkingDone[batchIdx])
        {
            isEos = false;
        }
        return isEos || context.currentGenerateLengths[batchIdx] >= context.maxGenerateLength;
    };

    // Lambda to update finish states based on EOS and max_length. Latches
    // terminalReason atomically with the state flip — the !finishedStates guard
    // keeps first-writer-wins semantics relative to applyCancellationToFinishStates.
    auto updateFinishStates = [&]() {
        for (int32_t i = 0; i < context.activeBatchSize; ++i)
        {
            if (context.finishedStates[i])
            {
                continue; // Respect first-writer-wins (cancel may have fired).
            }
            auto& s = context.slotStreams[i];
            // terminalReason is set for all slots; non-streaming slots surface it via
            // BatchResult.terminalReason -> response.finishReasons.
            if (hasActionRequest
                && request.requests[static_cast<size_t>(context.batchIndexMapping[i])].pastTrajectory.has_value())
            {
                if (context.tokenIds[i].size() > 1 && trajFutureStartId >= 0
                    && context.tokenIds[i][context.tokenIds[i].size() - 2] == trajFutureStartId)
                {
                    context.finishedStates[i] = 1;
                    s.terminalReason = FinishReason::kEndId;
                    LOG_DEBUG("Batch %d finished, reason: traj_future_start", i);
                    continue;
                }
            }
            else
            {
                // Check EOS (supports multiple EOS tokens, e.g. Gemma4 [1, 106]).
                // In thinking mode, suppress secondary EOS until thinking is complete.
                // EDGELLM_IGNORE_EOS bypasses EOS entirely to force a fixed-length run.
                if (!ignoreEos && !context.tokenIds[i].empty())
                {
                    auto const lastToken = context.tokenIds[i].back();
                    bool isEos = mTokenizer->isEosToken(lastToken);
                    if (isEos && request.enableThinking && lastToken != mTokenizer->getEosId() && !thinkingDone[i])
                    {
                        isEos = false;
                    }
                    if (isEos)
                    {
                        context.finishedStates[i] = 1;
                        s.terminalReason = FinishReason::kEndId;
                        LOG_DEBUG("Batch %d finished, reason: EOS", i);
                        continue;
                    }
                }
            }
            // Check max length
            if (context.currentGenerateLengths[i] >= context.maxGenerateLength)
            {
                context.finishedStates[i] = 1;
                s.terminalReason = FinishReason::kLength;
                LOG_DEBUG(
                    "Batch %d finished, total tokens=%d, reason: max_length", i, context.currentGenerateLengths[i]);
                continue;
            }
        }

        // Stop-string override pass — runs after EOS/length so it can override
        // kEndId/kLength (user-relevant cause). Cancel/error still win because
        // decodePerSlot skipped the match when those reasons were latched.
        for (int32_t i = 0; i < context.activeBatchSize; ++i)
        {
            auto& s = context.slotStreams[i];
            if (s.stopMatchedThisIter && s.terminalReason != FinishReason::kCancelled
                && s.terminalReason != FinishReason::kError)
            {
                context.finishedStates[i] = 1;
                s.terminalReason = FinishReason::kStopWords;
                LOG_DEBUG("Batch %d finished, reason: stop_words", i);
            }
        }
    };

    auto performBatchEvictAndSnapshot = [&]() {
        if (hasActionRequest)
        {
            mActionKvBatchCollector->captureFinished(*mSharedResources->kvPageTables[0],
                mSharedResources->cacheManagers[0]->getKVCacheLengths(), context.finishedStates,
                context.batchIndexMapping, context.stream);
        }
        bool const status = performBatchEvict(context, decodingStrategy, thinkingDone, managedRequest);
        if (status && hasActionRequest)
        {
            mActionKvBatchCollector->completeCapture();
        }
        return status;
    };

    // Post-prefill per-iter pipeline:
    //   cancel → decode (emitDelta + stop match) → finalize (EOS/length/stop) → emit
    // DiffusionGemma prefill writes prompt KV only and does not produce a generated token.
    if (!mDeployment.base.isDiffusionBackbone)
    {
        applyCancellationToFinishStates(context);
        if (!request.streamChannels.empty() && !synchronizeCancellationStates(context))
        {
            LOG_ERROR("Failed to synchronize cancellation state across parallel ranks after prefill.");
            return false;
        }
        decodePerSlot(context, *mTokenizer);

        updateThinkingDone();

        updateFinishStates();
        emitChunks(context, *mTokenizer);

        // Managed page rows can be compacted without moving physical KV. Keep the unmanaged lifecycle unchanged;
        // Qwen-style MTP cannot compact its recurrent draft state after partial prefill eviction.
        bool const supportsPartialPrefillEviction
            = (managedRequest != nullptr && decodingStrategy.kind() != DecodingStrategyKind::kMTP) || hasActionRequest;
        if (context.activeBatchSize > 0 && (supportsPartialPrefillEviction || checkAllFinished()))
        {
            bool const batchEvictStatus = performBatchEvictAndSnapshot();
            if (!batchEvictStatus)
            {
                LOG_ERROR("Failed to perform batch eviction.");
                return false;
            }
        }
    }

    while (!checkAllFinished())
    {
        // Observe any consumer cancels at the top of the iteration so they land
        // first in the per-slot terminalReason latch.
        applyCancellationToFinishStates(context);
        if (!request.streamChannels.empty() && !synchronizeCancellationStates(context))
        {
            LOG_ERROR("Failed to synchronize cancellation state across parallel ranks.");
            return false;
        }

        if (managedRequest != nullptr && !managedRequest->prepareDecodeStep(context, kvHeadroom))
        {
            return false;
        }

        if (!decodingStrategy.decodeStep(context))
        {
            LOG_ERROR("Failed to decode tokens with %s decoding strategy.", decodingStrategy.name());
            return false;
        }

        // Per-iter pipeline: decode -> finalize finish state -> emit chunks.
        decodePerSlot(context, *mTokenizer);
        updateThinkingDone();
        updateFinishStates();

        std::vector<int32_t> const& commonMaterializedStateLengths = decodingStrategy.commonMaterializedStateLengths();
        if (managedRequest != nullptr && !managedRequest->completeDecodeStep(context, commonMaterializedStateLengths))
        {
            return false;
        }
        emitChunks(context, *mTokenizer);

        emitTokenCallbacks(context);
        context.generationRound += 1;

        // Perform batch eviction after all old-slot progress and terminal publication are complete.
        if (!performBatchEvictAndSnapshot())
        {
            LOG_ERROR("Failed to perform batch eviction.");
            return false;
        }
    }

    // Few-layer-validation debug: write the accumulated per-layer logits/KV dump for this request.
    if (context.layerDebugger)
    {
        context.layerDebugger->flush(stream);
    }

    if (context.activeBatchSize != 0)
    {
        LOG_ERROR("Eviction failure, there should be no active batch at the end of the inference. activeBatchSize: %d",
            context.activeBatchSize);
        return false;
    }

    if (managedRequest != nullptr && !managedRequest->finish())
    {
        return false;
    }

    // Record metrics - accumulate across all batches (active + evicted)
    int32_t totalReusedTokens = 0;
    int32_t totalComputedTokens = 0;
    int32_t totalPrunedTokens = 0;
    int32_t totalGeneratedTokens = 0;
    int32_t totalIterations = 0;

    // Accumulate from completed batches
    for (auto const& [originalIdx, batchResult] : context.completedBatches)
    {
        int32_t rawPromptLength = static_cast<int32_t>(batchResult.rawBatchedInputIds.size());
        int32_t computedLength = batchResult.effectivePrefillLength;
        totalReusedTokens += (rawPromptLength - computedLength - batchResult.prunedPrefillTokens);
        totalComputedTokens += computedLength;
        totalPrunedTokens += batchResult.prunedPrefillTokens;
        totalGeneratedTokens += batchResult.generateLength;
        totalIterations += batchResult.actualIterations;
    }

    mPrefillMetrics.recordRun(totalReusedTokens, totalComputedTokens, totalPrunedTokens);
    if (enableSpecDecode)
    {
        mSpecDecodeGenerationMetrics.recordRun(totalIterations, totalGeneratedTokens);
    }
    else
    {
        mGenerationMetrics.recordRun(totalGeneratedTokens);
    }

    // Save output ids and decoded texts to response.
    // Maintain original batch order using original batch indices.
    response.outputIds.resize(context.completedBatches.size());
    response.outputTexts.resize(context.completedBatches.size());
    response.logprobs.resize(context.completedBatches.size());
    response.outputTrajectories.resize(context.completedBatches.size());
    response.finishReasons.resize(context.completedBatches.size(), FinishReason::kNotFinished);
    response.inputTokenCounts.assign(context.completedBatches.size(), 0);

    // Add outputs from completed batches (using saved original indices)
    for (auto const& [originalIdx, batchResult] : context.completedBatches)
    {
        int32_t genLength = batchResult.generateLength;

        // Log acceptance metrics for evicted batch
        if (enableSpecDecode)
        {
            int32_t const verificationTokens = genLength > 0 ? genLength - 1 : 0;
            float const acceptanceRate = batchResult.actualIterations > 0
                ? static_cast<float>(verificationTokens) / static_cast<float>(batchResult.actualIterations)
                : 0.0f;
            LOG_DEBUG(
                "Batch (completed with SpecDecode, original idx %d) - Acceptance rate: %.3f, Generated tokens: %d, "
                "Iterations: %d",
                originalIdx, acceptanceRate, genLength, batchResult.actualIterations);
        }

        // Extract generated tokens
        int32_t const totalLength = static_cast<int32_t>(batchResult.tokenIds.size());

        check::check(totalLength >= genLength, "Total length should be greater than or equal to generated length");
        response.outputIds[originalIdx] = std::vector<int32_t>(
            batchResult.tokenIds.begin() + (totalLength - genLength), batchResult.tokenIds.end());
        response.outputTexts[originalIdx] = mTokenizer->decode(response.outputIds[originalIdx], true);
        response.logprobs[originalIdx] = batchResult.logprobs;
        response.finishReasons[originalIdx] = batchResult.terminalReason;
        // Prompt length after chat templating and media expansion (OpenAI usage).
        response.inputTokenCounts[originalIdx] = static_cast<int32_t>(batchResult.rawBatchedInputIds.size());

        // Trim this slot's own stop strings from its output text by delegating
        // to applyStopStringMatch with isFinal=true — single source of truth
        // for earliest-position-wins semantics, shared with the streaming path.
        // outputIds is intentionally left intact (full token stream).
        if (originalIdx < static_cast<int32_t>(request.requests.size())
            && !request.requests[originalIdx].stopStrings.empty())
        {
            auto const& slotStops = request.requests[originalIdx].stopStrings;
            size_t maxLen = 0;
            for (auto const& s : slotStops)
            {
                maxLen = std::max(maxLen, s.size());
            }
            auto& text = response.outputTexts[originalIdx];
            auto outcome = applyStopStringMatch(text, slotStops, maxLen, /*isFinal=*/true);
            text = std::move(outcome.emitted);
            if (outcome.stopMatched)
            {
                // emitDelta (incremental) and one-shot Tokenizer::decode can differ at BPE
                // piece boundaries — upgrade the reason if one-shot surfaced a stop the
                // streaming-path matcher missed.
                response.finishReasons[originalIdx] = FinishReason::kStopWords;
            }
        }
    }

    if (hasActionRequest)
    {
        rt::ActionKvBatchView const actionKvBatch = mActionKvBatchCollector->materialize(stream);
        rt::HybridCacheManager const& kvcache = *mSharedResources->cacheManagers[0];
        std::vector<std::vector<rt::FutureTrajectoryPoint>> trajectories
            = mActionRunner->sampleTrajectory(stream, kvcache, actionKvBatch);
        if (trajectories.size() != static_cast<size_t>(actionKvBatch.batchSize))
        {
            LOG_ERROR("Alpamayo1ActionRunner trajectory sampling failed.");
            return false;
        }
        auto const& originalRequestIndices = mActionKvBatchCollector->originalRequestIndices();
        ELLM_CHECK(response.outputTrajectories.size() == request.requests.size(),
            "Action trajectory output batch does not match the request batch");
        ELLM_CHECK(trajectories.size() == originalRequestIndices.size(),
            "Action trajectory count does not match the materialized Action batch");
        for (size_t denseIndex = 0; denseIndex < trajectories.size(); ++denseIndex)
        {
            int32_t const originalIndex = originalRequestIndices[denseIndex];
            ELLM_CHECK(originalIndex >= 0 && originalIndex < static_cast<int32_t>(response.outputTrajectories.size()),
                "Action trajectory original request index is out of range");
            ELLM_CHECK(response.outputTrajectories[static_cast<size_t>(originalIndex)].empty(),
                "Action trajectory output slot was already populated");
            response.outputTrajectories[static_cast<size_t>(originalIndex)] = std::move(trajectories[denseIndex]);
        }
    }

    return true;
}

bool LLMRankRuntime::validateRequestConfig(LLMGenerationRequest const& request)
{
    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());
    bool const hasAudio = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.audioBuffers.empty(); });
    bool const hasVision = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.imageBuffers.empty(); });
    bool const hasTrajectoryHistory = std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& req) { return req.pastTrajectory.has_value(); });

    if (activeBatchSize == 0)
    {
        LOG_ERROR("Empty request with no requests");
        return false;
    }

    if (activeBatchSize > mMaxRuntimeBatchSize)
    {
        LOG_ERROR(
            "Requested batch size %d exceeds maximum supported batch size %d", activeBatchSize, mMaxRuntimeBatchSize);
        return false;
    }
    if (request.disableSpecDecode && mDeployment.specDecodeMode() == SpecDecodeMode::kGemma4MTP)
    {
        LOG_ERROR(
            "disable_spec_decode is not supported by a Gemma4 MTP verification engine. Use the matched assistant, or "
            "build a standalone target engine for target-only inference.");
        return false;
    }
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        if (request.requests[i].messages.empty())
        {
            LOG_ERROR("Request %d in batch is empty: no messages provided", i);
            return false;
        }
        auto const& logitBias = request.requests[i].logitBias;
        if (logitBias.size() > limits::security::kMaxLogitBiasTokens)
        {
            LOG_ERROR("Request %d has too many logit_bias entries: %zu (max: %zu)", i, logitBias.size(),
                limits::security::kMaxLogitBiasTokens);
            return false;
        }
        for (auto const& [tokenId, bias] : logitBias)
        {
            if (tokenId < 0 || tokenId >= mDeployment.base.vocabSize)
            {
                LOG_ERROR("Request %d logit_bias token ID %d is outside the full vocabulary range [0, %d)", i, tokenId,
                    mDeployment.base.vocabSize);
                return false;
            }
            if (!std::isfinite(bias) || bias < limits::security::kMinLogitBias
                || bias > limits::security::kMaxLogitBias)
            {
                LOG_ERROR("Request %d logit_bias for token ID %d must be finite and in [%.1f, %.1f], got %.6f", i,
                    tokenId, limits::security::kMinLogitBias, limits::security::kMaxLogitBias, bias);
                return false;
            }
        }
    }
    if (hasAudio && !mAudioRunner)
    {
        LOG_ERROR("Request contains audio input, but this runtime does not have an audio runner.");
        return false;
    }
    if (hasVision && !mVisionRunner)
    {
        LOG_ERROR("Request contains vision input, but this runtime does not have a vision runner.");
        return false;
    }
    if (hasTrajectoryHistory && !mActionRunner)
    {
        LOG_ERROR("Request contains trajectory history input, but this runtime does not have an action runner.");
        return false;
    }
    if (mDeployment.base.useVisionBidirectionalAttention && request.saveSystemPromptKVCache)
    {
        LOG_ERROR("System-prompt KV-cache reuse is not supported with Gemma4 vision bidirectional attention.");
        return false;
    }

    return true;
}

bool LLMRankRuntime::multiModalRuntimePreprocess(
    LLMGenerationRequest const& request, DecodingInferenceContext& context, cudaStream_t stream)
{
    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());
    bool const hasAudio = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.audioBuffers.empty(); });
    bool const hasVision = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.imageBuffers.empty(); });
    bool const hasTrajectoryHistory = std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& req) { return req.pastTrajectory.has_value(); });

    // Clear request-scoped multimodal state up front so previous requests cannot leak through reused runtime members.
    context.visualEmbeddings = std::nullopt;
    context.audioEmbeddings = std::nullopt;
    context.deepstackFeatures.clear();
    // Treat multimodal indices as request-scoped state. Only request paths that explicitly rebuild
    // mMultimodalIndices for the current request should observe a non-empty tensor downstream.
    check::check(mMultimodalIndices.reshape({0}), "Tensor reshape failed");

    // Mark multimodal preprocessing and inference for NVTX profiling
    NVTX_SCOPED_RANGE(nvtx_multimodal, "MULTIMODAL_PROCESSING", nvtx_colors::ORANGE);

    std::vector<std::vector<int32_t>> batchedInputIds;
    auto appendPreparedTextInputIds = [&]() -> bool {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            if (i < static_cast<int32_t>(request.preTokenizedInputIds.size())
                && !request.preTokenizedInputIds[i].empty())
            {
                batchedInputIds.push_back(request.preTokenizedInputIds[i]);
            }
            else
            {
                batchedInputIds.push_back(
                    mTokenizer->encode(request.formattedRequests[i].formattedCompleteRequest, false));
            }
            if (batchedInputIds.back().empty())
            {
                LOG_ERROR("Failed to tokenize input text for request %d in batch", i);
                return false;
            }
        }
        return true;
    };

    // MRope cos/sin output cache is supplied only for MRope-based runners (QwenViT, Qwen3OmniAudio).
    // Runners with standard RoPE (InternViT, Phi4MMViT) ignore it; see MultimodalRunner::preprocess.
    rt::OptionalOutputTensor mropeCosSinOut = (mDeployment.base.ropeConfig.type == RopeType::kMRope)
        ? rt::OptionalOutputTensor{std::ref(mPipelineIO->mropeCosSin)}
        : std::nullopt;

    // Process audio inputs (if present)
    if (hasAudio && mAudioRunner)
    {
        bool audioCacheHit = false;
        std::vector<Hash128> audioHashes;

        if (mEncoderEmbeddingCache)
        {
            bool hashable = true;
            for (auto const& req : request.requests)
            {
                for (auto const& audio : req.audioBuffers)
                {
                    if (audio.pcm && !audio.pcm->samples.empty())
                    {
                        auto const* rawPtr = reinterpret_cast<char const*>(audio.pcm->samples.data());
                        size_t const rawBytes = audio.pcm->samples.size() * sizeof(float);
                        audioHashes.push_back(hashOpaqueIdentity(std::string_view(rawPtr, rawBytes)));
                    }
                    else
                    {
                        hashable = false;
                    }
                }
            }
            if (!hashable)
            {
                audioHashes.clear();
            }
        }

        bool allHit = mEncoderEmbeddingCache && !audioHashes.empty();
        std::vector<std::reference_wrapper<rt::Tensor const>> cachedAudio;
        if (allHit)
        {
            for (auto const& h : audioHashes)
            {
                auto r = mEncoderEmbeddingCache->lookup(h);
                if (!r)
                {
                    allHit = false;
                    break;
                }
                cachedAudio.push_back(std::cref(r->get()));
            }
        }

        if (allHit)
        {
            LOG_INFO("Encoder embedding cache HIT for all %zu audio clips — skipping audio encoder execution",
                audioHashes.size());
            if (!mAudioRunner->preprocess(request, batchedInputIds, mTokenizer, mropeCosSinOut, stream, false, true))
            {
                LOG_ERROR("Audio text preprocessing failed on cache hit.");
                return false;
            }
            rt::Tensor& output = mAudioRunner->getOutputEmbedding();
            int64_t byteOffset = 0;
            for (auto const& entry : cachedAudio)
            {
                int64_t const bytes = entry.get().getMemoryCapacity();
                CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(output.rawPointer()) + byteOffset,
                    entry.get().rawPointer(), bytes, cudaMemcpyDeviceToDevice, stream));
                byteOffset += bytes;
            }
            audioCacheHit = true;
        }

        if (!audioCacheHit)
        {
            if (!mAudioRunner->preprocess(request, batchedInputIds, mTokenizer, mropeCosSinOut, stream))
            {
                LOG_ERROR("Audio preprocessing failed. This request cannot be handled.");
                return false;
            }

            if (!mAudioRunner->infer(stream))
            {
                LOG_ERROR("Audio inference failed. This request cannot be handled.");
                return false;
            }

            if (mEncoderEmbeddingCache && !audioHashes.empty())
            {
                rt::Tensor const& output = mAudioRunner->getOutputEmbedding();
                auto const& tokenLengths = mAudioRunner->getLastMediaTokenLengths();
                int64_t const hiddenSize = output.getShape()[1];
                size_t const typeSize = rt::utils::getTypeSize(output.getDataType());
                int64_t byteOffset = 0;
                for (size_t i = 0; i < audioHashes.size(); ++i)
                {
                    int64_t const numTok = tokenLengths[i];
                    mEncoderEmbeddingCache->storeSlice(audioHashes[i],
                        static_cast<char const*>(output.rawPointer()) + byteOffset, numTok, hiddenSize,
                        output.getDataType(), stream);
                    byteOffset += numTok * hiddenSize * static_cast<int64_t>(typeSize);
                }
            }
        }
    }

    // Process vision inputs (if present)
    if (hasVision && mVisionRunner)
    {
        bool visionCacheHit = false;
        std::vector<Hash128> imageHashes;

        if (mEncoderEmbeddingCache)
        {
            bool hashable = true;
            for (auto const& req : request.requests)
            {
                for (auto const& img : req.imageBuffers)
                {
                    if (img.buffer && !img.buffer->isEmpty())
                    {
                        auto const* rawPtr = reinterpret_cast<char const*>(img.data());
                        size_t const rawBytes = static_cast<size_t>(img.bytesPerFrame()) * img.frames;
                        imageHashes.push_back(hashOpaqueIdentity(std::string_view(rawPtr, rawBytes)));
                    }
                    else
                    {
                        hashable = false;
                    }
                }
            }
            if (!hashable)
            {
                imageHashes.clear();
            }
        }

        bool allHit = mEncoderEmbeddingCache && !imageHashes.empty();
        std::vector<std::reference_wrapper<rt::Tensor const>> cachedVision;
        if (allHit)
        {
            for (auto const& h : imageHashes)
            {
                auto r = mEncoderEmbeddingCache->lookup(h);
                if (!r)
                {
                    allHit = false;
                    break;
                }
                cachedVision.push_back(std::cref(r->get()));
            }
        }

        if (allHit)
        {
            LOG_INFO(
                "Encoder embedding cache HIT for all %zu images — skipping ViT encoder execution", imageHashes.size());
            if (!mVisionRunner->preprocess(request, batchedInputIds, mTokenizer, mropeCosSinOut, stream, false, true))
            {
                LOG_ERROR("Vision text preprocessing failed on cache hit.");
                return false;
            }
            rt::Tensor& output = mVisionRunner->getOutputEmbedding();
            int64_t byteOffset = 0;
            for (auto const& entry : cachedVision)
            {
                int64_t const bytes = entry.get().getMemoryCapacity();
                CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(output.rawPointer()) + byteOffset,
                    entry.get().rawPointer(), bytes, cudaMemcpyDeviceToDevice, stream));
                byteOffset += bytes;
            }
            visionCacheHit = true;
        }

        if (!visionCacheHit)
        {
            if (!mVisionRunner->preprocess(request, batchedInputIds, mTokenizer, mropeCosSinOut, stream))
            {
                LOG_ERROR("Vision preprocessing failed. This request cannot be handled.");
                return false;
            }

            if (!mVisionRunner->infer(stream))
            {
                LOG_ERROR("Vision inference failed. This request cannot be handled.");
                return false;
            }

            if (mEncoderEmbeddingCache && !imageHashes.empty())
            {
                rt::Tensor const& output = mVisionRunner->getOutputEmbedding();
                auto const& tokenLengths = mVisionRunner->getLastMediaTokenLengths();
                if (tokenLengths.size() == imageHashes.size())
                {
                    int64_t const hiddenSize = output.getShape()[1];
                    size_t const typeSize = rt::utils::getTypeSize(output.getDataType());
                    int64_t byteOffset = 0;
                    for (size_t i = 0; i < imageHashes.size(); ++i)
                    {
                        int64_t const numTok = tokenLengths[i];
                        mEncoderEmbeddingCache->storeSlice(imageHashes[i],
                            static_cast<char const*>(output.rawPointer()) + byteOffset, numTok, hiddenSize,
                            output.getDataType(), stream);
                        byteOffset += numTok * hiddenSize * static_cast<int64_t>(typeSize);
                    }
                }
            }
        }
    }

    // Process action inputs (if present)
    if (hasTrajectoryHistory && mActionRunner)
    {
        LOG_INFO("Processing trajectory history inputs");
        if (!hasAudio && !hasVision && batchedInputIds.empty() && !appendPreparedTextInputIds())
        {
            return false;
        }
        if (!mActionRunner->preprocess(request, batchedInputIds, mTokenizer))
        {
            LOG_ERROR("LLMRankRuntime(): Trajectory history preprocessing failed. This request cannot be handled.");
            return false;
        }
    }

    if (!hasAudio && !hasVision)
    {
        if (batchedInputIds.empty() && !appendPreparedTextInputIds())
        {
            return false;
        }
        if (mDeployment.base.ropeConfig.type == RopeType::kMRope)
        {
            rt::Tensor& ropeCosSinCache = mPipelineIO->mropeCosSin;
            check::check(ropeCosSinCache.reshape({mDeployment.base.maxSupportedBatchSize,
                             mDeployment.base.maxKVCacheCapacity, mDeployment.base.rotaryDim}),
                "Tensor reshape failed");
            kernel::initializeTextOnlyMRopeCosSin(ropeCosSinCache.dataPointer<float>(),
                mDeployment.base.ropeConfig.rotaryTheta, mDeployment.base.rotaryDim,
                mDeployment.base.maxKVCacheCapacity, mDeployment.base.maxSupportedBatchSize, stream);
        }
    }

    // Get embeddings from independent runners — gate on request having multimodal data,
    // not just runner existence, to avoid leaking stale embeddings from previous requests.
    rt::OptionalInputTensor visionEmbeddings
        = (hasVision && mVisionRunner) ? std::optional{std::ref(mVisionRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensor audioEmbeddings
        = (hasAudio && mAudioRunner) ? std::optional{std::ref(mAudioRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensors deepstackFeatures
        = (hasVision && mVisionRunner) ? mVisionRunner->getDeepstackFeatures() : rt::OptionalInputTensors{};

    context.visualEmbeddings = visionEmbeddings;
    context.deepstackFeatures = deepstackFeatures;
    context.audioEmbeddings = audioEmbeddings;

    // Populate system prompts and raw input IDs from batchedInputIds
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        context.systemPrompts[i] = request.formattedRequests[i].formattedSystemPrompt;
        context.rawBatchedInputIds.push_back(batchedInputIds[i]);
    }

    return true;
}

bool LLMRankRuntime::runHybridMtpPrefill(
    DecodingInferenceContext& context, DecodingStrategy& strategy, ContextCacheRequest& contextCacheRequest)
{
    // Adapted from reference llmInferenceRuntime.cpp::runHybridMtpPrefill (:2515-2702). The reference reads
    // context.sequenceCacheStates + mHybridSnapshotStorage directly; on this target the cache lifecycle is owned by the
    // coordinator, so publication/restore go through the adapter and the reuse length comes from the admission plan.
    // The fold (row shift + boundary-hidden restore + boundary-token prepend) runs here in the runtime, exactly as in
    // the reference, because the coordinator is only reachable from the runtime side (not from MTPDecoder).
    ELLM_CHECK(context.activeBatchSize == 1, "Hybrid MTP endpoint prefill requires one combined cache sequence");

    int32_t const reuseLength = contextCacheRequest.reuseTokenLength(0);
    int32_t const inputLength = math::cast<int32_t>(context.rawBatchedInputIds[0].size());
    int32_t const replayTailLength = context.contextCacheReplayTailLength;
    int32_t const suffixLenOrig = context.effectivePrefillLengths[0];
    LOG_DEBUG("Hybrid+MTP prefill: inputLength=%d reuseLength=%d basePrefill=%d", inputLength, reuseLength,
        inputLength - reuseLength);

    int32_t const boundaryHiddenDim = mDeployment.specConfig->baseOutputHiddenDim;
    size_t const rowBytes
        = static_cast<size_t>(boundaryHiddenDim) * rt::utils::getTypeSize(mPipelineIO->baseHiddenStates.getDataType());

    // Run the (folded) draft prefill through the strategy's initialize-for-generation override. Resetting the guard
    // lets it run again where the reference re-invokes prepareFirstDecodeStep (mirrors ref :2560/:2626). The final
    // invocation leaves speculativeDraftPrefillComplete = true so the outer initializeForGeneration and decode round 0
    // both skip it. Driving a re-run through a guard flag is a workaround for DecodingStrategy having no explicit
    // "run the draft prefill now" entry point; see issue #655 for the intended interface.
    auto runFoldedDraftPrefill = [&]() -> bool {
        context.speculativeDraftPrefillComplete = false;
        return strategy.initializeForGeneration(context);
    };

    // Shift baseHiddenStates' first chunkLength rows down by one and restore the reused checkpoint's boundary hidden
    // into row 0, so the draft prefill sees [base_hidden[boundary], base_hidden[boundary+1 ..]]. Writing chunkLength+1
    // rows needs one spare row over the chunk itself; that invariant is local to this fold, so check it here rather
    // than relying on engine-level sizing of baseHiddenStates.
    auto foldBoundaryHiddenIntoRow0 = [&](int32_t chunkLength) -> bool {
        ELLM_CHECK(chunkLength + 1 <= mBoundaryFoldMaxRows,
            "Hybrid+MTP boundary fold needs one row beyond the prefill chunk in baseHiddenStates");
        check::check(
            mPipelineIO->baseHiddenStates.reshape({1, chunkLength + 1, boundaryHiddenDim}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mBoundaryFoldScratch.rawPointer(), mPipelineIO->baseHiddenStates.rawPointer(),
            static_cast<size_t>(chunkLength) * rowBytes, cudaMemcpyDeviceToDevice, context.stream));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(mPipelineIO->baseHiddenStates.rawPointer()) + rowBytes,
            mBoundaryFoldScratch.rawPointer(), static_cast<size_t>(chunkLength) * rowBytes, cudaMemcpyDeviceToDevice,
            context.stream));
        return contextCacheRequest.restoreHybridMtpBoundaryHidden(0, mPipelineIO->baseHiddenStates, 0);
    };

    // The generation-prompt tail is volatile when the chat template appends tokens that the next turn's render of the
    // same history does not reproduce (for example Qwen3's `<think>\n\n</think>\n\n` under enable_thinking=false).
    // Those tokens must not enter the published checkpoint, so the prefill splits into a stable predecessor chunk plus
    // a replayed tail. The tail length is measured server-side (tool_chat_template.format_with_replay_tail).
    bool const hasVolatileTail = replayTailLength > 0 && suffixLenOrig > replayTailLength;

    // Cold sequence with a volatile generation-prompt tail: publish the checkpoint at the STABLE boundary
    // predecessorLength = inputLength - replayTailLength (two-chunk prefill), not at the full inputLength (ref :2534).
    if (reuseLength == 0 && hasVolatileTail)
    {
        std::vector<int32_t> const completeSuffix = context.tokenIds[0];
        int32_t const predecessorLength = inputLength - replayTailLength;
        int32_t const predecessorChunkLength = suffixLenOrig - replayTailLength;
        auto const replayBegin = completeSuffix.end() - replayTailLength;

        // Chunk 1: base prefill of the predecessor [0, predecessorLength), no sampling.
        context.tokenIds[0].assign(completeSuffix.begin(), replayBegin);
        context.effectivePrefillLengths[0] = predecessorChunkLength;
        if (!runBaseModelPrefill(context, /*contextCacheRequest=*/nullptr, /*sampleOutput=*/false))
        {
            return false;
        }
        // Non-sampling chunk must sync before MTP repacks the shared pinned token buffer.
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        // Provide the boundary's next token so the predecessor draft prefill covers the boundary slot, then publish.
        context.tokenIds[0].push_back(*replayBegin);
        if (!runFoldedDraftPrefill())
        {
            return false;
        }
        if (!contextCacheRequest.publishHybridMtpEndpoint(
                0, predecessorLength, mPipelineIO->baseHiddenStates, context.effectivePrefillLengths[0] - 1))
        {
            return false;
        }

        // Chunk 2: replay the volatile tail [predecessorLength, inputLength) with sampling, then restore bookkeeping.
        context.tokenIds[0].assign(replayBegin, completeSuffix.end());
        context.effectivePrefillLengths[0] = replayTailLength;
        if (!runBaseModelPrefill(context, /*contextCacheRequest=*/nullptr, /*sampleOutput=*/true))
        {
            return false;
        }
        int32_t const sampledToken = context.tokenIds[0].back();
        if (!runFoldedDraftPrefill())
        {
            return false;
        }
        context.tokenIds[0] = completeSuffix;
        context.tokenIds[0].push_back(sampledToken);
        context.effectivePrefillLengths[0] = suffixLenOrig;
        return true;
    }

    // Hit sequence with a volatile tail: publish at the stable predecessorLength like the cold path, but first
    // reconstruct the reused checkpoint's boundary draft slot via the consume-side fold (ref :2581).
    if (reuseLength > 0 && hasVolatileTail)
    {
        std::vector<int32_t> const completeSuffix = context.tokenIds[0];
        int32_t const predecessorLength = inputLength - replayTailLength;
        int32_t const predecessorChunkLength = suffixLenOrig - replayTailLength;
        auto const replayBegin = completeSuffix.end() - replayTailLength;

        // Chunk 1: base prefill of the predecessor [reuseLength, predecessorLength), no sampling.
        context.tokenIds[0].assign(completeSuffix.begin(), replayBegin);
        context.effectivePrefillLengths[0] = predecessorChunkLength;
        if (!runBaseModelPrefill(context, /*contextCacheRequest=*/nullptr, /*sampleOutput=*/false))
        {
            return false;
        }
        CUDA_CHECK(cudaStreamSynchronize(context.stream));

        // Fold the reused checkpoint's boundary hidden into the predecessor draft prefill, reconstructing the reused
        // draft slot [reuseLength-1] (ref :2602-2621).
        if (!foldBoundaryHiddenIntoRow0(predecessorChunkLength))
        {
            return false;
        }

        int32_t const boundaryToken = context.rawBatchedInputIds[0][static_cast<size_t>(reuseLength - 1)];
        context.tokenIds[0].insert(context.tokenIds[0].begin(), boundaryToken);
        context.effectivePrefillLengths[0] = predecessorChunkLength + 1;
        if (!runFoldedDraftPrefill())
        {
            return false;
        }
        if (!contextCacheRequest.publishHybridMtpEndpoint(
                0, predecessorLength, mPipelineIO->baseHiddenStates, context.effectivePrefillLengths[0] - 1))
        {
            return false;
        }
        context.tokenIds[0].erase(context.tokenIds[0].begin());
        context.effectivePrefillLengths[0] = predecessorChunkLength;

        // Chunk 2: replay the volatile tail [predecessorLength, inputLength) with sampling, then restore bookkeeping.
        context.tokenIds[0].assign(replayBegin, completeSuffix.end());
        context.effectivePrefillLengths[0] = replayTailLength;
        if (!runBaseModelPrefill(context, /*contextCacheRequest=*/nullptr, /*sampleOutput=*/true))
        {
            return false;
        }
        int32_t const sampledToken = context.tokenIds[0].back();
        if (!runFoldedDraftPrefill())
        {
            return false;
        }
        context.tokenIds[0] = completeSuffix;
        context.tokenIds[0].push_back(sampledToken);
        context.effectivePrefillLengths[0] = suffixLenOrig;
        return true;
    }

    // Base prefill of the suffix (the full prompt when cold). Reuses base KV [0, reuseLength) and samples the first
    // output token, appending it to tokenIds (ref :2646).
    if (!runBaseModelPrefill(context, /*contextCacheRequest=*/nullptr, /*sampleOutput=*/true))
    {
        return false;
    }

    if (reuseLength == 0)
    {
        // Cold sequence: no reused checkpoint boundary to fold in (ref :2651).
        if (!runFoldedDraftPrefill())
        {
            return false;
        }
        if (!contextCacheRequest.publishHybridMtpEndpoint(
                0, inputLength, mPipelineIO->baseHiddenStates, context.effectivePrefillLengths[0] - 1))
        {
            return false;
        }
        return true;
    }

    // On a hit the restore path reused draft KV only up to reuseLength-1. Fold the boundary into the suffix draft
    // prefill as a *context* position: prepend the checkpoint's saved base boundary hidden (base_hidden[reuseLength-1])
    // and token[reuseLength-1] so the single draft prefill covers [reuseLength-1, inputLength) and writes a correct,
    // successor-aware boundary KV (ref :2662-2701).
    int32_t const suffixLen = context.effectivePrefillLengths[0];

    // Shift the base suffix hidden states [0, suffixLen) down one row and place the restored boundary hidden state at
    // row 0, giving the draft [base_hidden[reuseLength-1], base_hidden[reuseLength..inputLength)].
    if (!foldBoundaryHiddenIntoRow0(suffixLen))
    {
        return false;
    }

    // Prepend token[reuseLength-1] and extend the prefill length by one. runDraftModelPrefill pairs draft slot k with
    // (baseHiddenStates[k], tokenIds[k+1]); with the shifted hidden states and this prepend, draft slot reuseLength-1+k
    // consumes (base_hidden[reuseLength-1+k], token[reuseLength+k]) as required.
    int32_t const boundaryToken = context.rawBatchedInputIds[0][static_cast<size_t>(reuseLength - 1)];
    context.tokenIds[0].insert(context.tokenIds[0].begin(), boundaryToken);
    context.effectivePrefillLengths[0] = suffixLen + 1;
    if (!runFoldedDraftPrefill())
    {
        return false;
    }
    // Publish while effectivePrefillLengths still reflects the folded prefill (the boundary-hidden capture reads the
    // last shifted row), then restore the suffix-only execution bookkeeping for the decode loop.
    if (!contextCacheRequest.publishHybridMtpEndpoint(
            0, inputLength, mPipelineIO->baseHiddenStates, context.effectivePrefillLengths[0] - 1))
    {
        return false;
    }
    context.tokenIds[0].erase(context.tokenIds[0].begin());
    context.effectivePrefillLengths[0] = suffixLen;
    return true;
}

bool LLMRankRuntime::runBaseModelPrefill(
    DecodingInferenceContext& context, ContextCacheRequest* contextCacheRequest, bool sampleOutput)
{
    TIME_STAGE(metrics::StageNames::kLLM_PREFILL, context.stream);
    NVTX_SCOPED_RANGE(nvtx_base_prefill,
        ("SPEC_DECODE_BASE_PREFILL[" + std::to_string(context.activeBatchSize) + "]").c_str(), nvtx_colors::BLUE);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t inputIdsLength
        = *std::max_element(context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.end());
    int32_t const baseOutputHiddenDim
        = mDeployment.specConfig.has_value() ? mDeployment.specConfig->baseOutputHiddenDim : 0;

    // Reshape IO tensors for this step.
    check::check(mIdsInput.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    check::check(mPipelineIO->hostContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mPipelineIO->inputsEmbeds.reshape({activeBatchSize, inputIdsLength, mDeployment.base.hiddenSize}),
        "Tensor reshape failed");
    if (mDeployment.base.isDiffusionBackbone)
    {
        check::check(mPipelineIO->outputLogits.reshape({activeBatchSize, 1, mDeployment.base.outputVocabSize}),
            "Tensor reshape failed");
        if (mDeployment.base.diffusionUnifiedConditioning)
        {
            Tensor* canvasIds = mBaseTensorMap.get(binding_names::kCanvasIds);
            Tensor* prevSelfConditioningEmbeds = mBaseTensorMap.get(binding_names::kPrevSelfConditioningEmbeds);
            Tensor* nextSelfConditioningEmbeds = mBaseTensorMap.get(binding_names::kNextSelfConditioningEmbeds);
            Tensor* selfConditioningTemperature = mBaseTensorMap.get(binding_names::kSelfConditioningTemperature);
            check::check(canvasIds != nullptr && prevSelfConditioningEmbeds != nullptr
                    && nextSelfConditioningEmbeds != nullptr && selfConditioningTemperature != nullptr,
                "DiffusionGemma unified conditioning bindings are missing for prefill.");
            check::check(canvasIds->reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
            check::check(
                prevSelfConditioningEmbeds->reshape({activeBatchSize, inputIdsLength, mDeployment.base.hiddenSize}),
                "Tensor reshape failed");
            check::check(nextSelfConditioningEmbeds->reshape({activeBatchSize, 1, mDeployment.base.hiddenSize}),
                "Tensor reshape failed");
            bindDiffusionUnifiedBackboneTensors(mBaseTensorMap, *mPipelineIO, mPipelineIO->outputLogits, *canvasIds,
                *prevSelfConditioningEmbeds, *nextSelfConditioningEmbeds, *selfConditioningTemperature);
        }
        check::check(mPipelineIO->phaseIsEncoder.reshape({activeBatchSize}), "Tensor reshape failed");
        check::check(mPipelineIO->hostPhaseIsEncoder.reshape({activeBatchSize}), "Tensor reshape failed");
        check::check(mPipelineIO->contextMaskSelector.reshape({0}), "Tensor reshape failed");
        int32_t* hostPhase = mPipelineIO->hostPhaseIsEncoder.dataPointer<int32_t>();
        std::fill(hostPhase, hostPhase + activeBatchSize, 1);
        CUDA_CHECK(cudaMemcpyAsync(mPipelineIO->phaseIsEncoder.rawPointer(), hostPhase,
            activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    }
    else
    {
        check::check(mPipelineIO->outputLogits.reshape({activeBatchSize, mDeployment.base.outputVocabSize}),
            "Tensor reshape failed");
    }
    if (mDeployment.specConfig.has_value())
    {
        // SpecDecode base engines emit target features that feed the draft engine.
        check::check(mPipelineIO->baseHiddenStates.reshape({activeBatchSize, inputIdsLength, baseOutputHiddenDim}),
            "Tensor reshape failed");
    }

    // Populate host-side context lengths with effective (unpadded) prefill lengths and pack tokens.
    int32_t* hostCtxLenData = mPipelineIO->hostContextLengths.dataPointer<int32_t>();
    check::check(mHostPackedTokenIds.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    int32_t* hostPackedTokenIdsData = mHostPackedTokenIds.dataPointer<int32_t>();

    // Clear the entire pinned buffer first so trailing pad slots from prior batches don't leak into the
    // multimodal-indices walk, which scans all inputIdsLength positions per row, not just up to context_length.
    std::fill(hostPackedTokenIdsData, hostPackedTokenIdsData + activeBatchSize * inputIdsLength, 0);

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        int32_t const requestedSeqLen = context.effectivePrefillLengths[i];
        ELLM_CHECK(requestedSeqLen >= 0 && requestedSeqLen <= inputIdsLength,
            "Effective prefill length must be within the current input sequence");
        hostCtxLenData[i] = requestedSeqLen;
        std::copy(context.tokenIds[i].begin(), context.tokenIds[i].end(), hostPackedTokenIdsData + i * inputIdsLength);
    }

    CUDA_CHECK(cudaMemcpyAsync(mIdsInput.rawPointer(), hostPackedTokenIdsData,
        activeBatchSize * inputIdsLength * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    bool const baseKVAllEmpty = mSharedResources->cacheManagers[0]->getKVCacheAllEmpty();
    if (mDeployment.base.useVisionBidirectionalAttention)
    {
        // Vision-block attention supports only non-chunked prefill. Decode
        // ignores this binding and uses causal decode attention over the
        // canonical KV cache.
        if (!baseKVAllEmpty)
        {
            LOG_ERROR(
                "Gemma4 vision bidirectional attention does not yet support prefix-cache reuse or chunked prefill.");
            return false;
        }
        check::check(mPipelineIO->visionBlockIds.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
        rt::Tensor hostVisionBlockIds = generateVisionBlockIds(mHostPackedTokenIds, mDeployment.base.imageTokenId);
        // hostVisionBlockIds owns short-lived pinned storage. Keep this copy
        // synchronous so the source remains alive until H2D completion.
        CUDA_CHECK(cudaMemcpy(mPipelineIO->visionBlockIds.rawPointer(), hostVisionBlockIds.rawPointer(),
            activeBatchSize * inputIdsLength * sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    // Embedding lookup (text / vision / audio-multimodal) into mPipelineIO->inputsEmbeds;
    // deepstack slots are populated from features or zero-filled depending on the request.
    //
    // When context-cache reuse is active, tokenIds are the suffix (prefix tokens are served from KV cache).
    // The ViT output concatenates ALL batch elements' embeddings sequentially, so per-row offsets are needed
    // to index past earlier sequences' embeddings that were consumed by the cached prefix.
    //
    // Multimodal indices are computed on CPU from the host-side packed token IDs (still available in
    // mHostPackedTokenIds) and uploaded once. This avoids a single-threaded GPU kernel launch plus
    // separate H2D copies for per-batch base offsets.
    rt::OptionalInputTensor precomputedIndices = std::nullopt;
    if (context.visualEmbeddings.has_value() || context.audioEmbeddings.has_value())
    {
        int32_t const imageTokenId = mDeployment.base.imageTokenId;
        int32_t const audioTokenId = mDeployment.base.audioTokenId;

        check::check(mMultimodalIndices.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
        check::check(mHostMultimodalIndices.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
        int32_t* hostIndices = mHostMultimodalIndices.dataPointer<int32_t>();

        if (contextCacheRequest != nullptr)
        {
            int32_t cumulativeImageTokens = 0;
            int32_t cumulativeAudioTokens = 0;
            for (int32_t i = 0; i < activeBatchSize; ++i)
            {
                auto const& fullSeq = context.rawBatchedInputIds[i];
                int32_t const prefixLen = static_cast<int32_t>(fullSeq.size()) - context.effectivePrefillLengths[i];
                int32_t prefixImageCount = 0;
                int32_t prefixAudioCount = 0;
                for (int32_t p = 0; p < prefixLen; ++p)
                {
                    if (fullSeq[p] == imageTokenId)
                    {
                        ++prefixImageCount;
                    }
                    else if (fullSeq[p] == audioTokenId)
                    {
                        ++prefixAudioCount;
                    }
                }
                int32_t const rowImageBase = cumulativeImageTokens + prefixImageCount;
                int32_t const rowAudioBase = cumulativeAudioTokens + prefixAudioCount;

                // Generate indices for this row from the packed host token IDs.
                int32_t rowImageIdx = rowImageBase;
                int32_t rowAudioIdx = rowAudioBase;
                int32_t const* rowTokens = hostPackedTokenIdsData + static_cast<int64_t>(i) * inputIdsLength;
                int32_t* rowIndices = hostIndices + static_cast<int64_t>(i) * inputIdsLength;
                for (int32_t col = 0; col < inputIdsLength; ++col)
                {
                    int32_t const tok = rowTokens[col];
                    if (imageTokenId >= 0 && tok == imageTokenId)
                    {
                        rowIndices[col] = rowImageIdx++;
                    }
                    else if (audioTokenId >= 0 && tok == audioTokenId)
                    {
                        rowIndices[col] = rowAudioIdx++;
                    }
                    else
                    {
                        rowIndices[col] = 0;
                    }
                }

                for (size_t p = 0; p < fullSeq.size(); ++p)
                {
                    if (fullSeq[p] == imageTokenId)
                    {
                        ++cumulativeImageTokens;
                    }
                    else if (fullSeq[p] == audioTokenId)
                    {
                        ++cumulativeAudioTokens;
                    }
                }
            }
        }
        else
        {
            int32_t imageIndex = 0;
            int32_t audioIndex = 0;
            for (int32_t i = 0; i < activeBatchSize; ++i)
            {
                int32_t const* rowTokens = hostPackedTokenIdsData + static_cast<int64_t>(i) * inputIdsLength;
                int32_t* rowIndices = hostIndices + static_cast<int64_t>(i) * inputIdsLength;
                for (int32_t col = 0; col < inputIdsLength; ++col)
                {
                    int32_t const tok = rowTokens[col];
                    if (imageTokenId >= 0 && tok == imageTokenId)
                    {
                        rowIndices[col] = imageIndex++;
                    }
                    else if (audioTokenId >= 0 && tok == audioTokenId)
                    {
                        rowIndices[col] = audioIndex++;
                    }
                    else
                    {
                        rowIndices[col] = 0;
                    }
                }
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(mMultimodalIndices.rawPointer(), hostIndices,
            static_cast<size_t>(activeBatchSize) * inputIdsLength * sizeof(int32_t), cudaMemcpyHostToDevice,
            context.stream));
        precomputedIndices = rt::OptionalInputTensor{mMultimodalIndices};
    }
    mEmbeddingPre->embed(
        mIdsInput, context.visualEmbeddings, context.audioEmbeddings, *mPipelineIO, context.stream, precomputedIndices);
    mEmbeddingPre->prepareDeepstack(mIdsInput, context.deepstackFeatures, *mPipelineIO, context.stream);
    if (mGemma4Ple)
    {
        mGemma4Ple->embed(mIdsInput, context.stream);
    }

    // Visual-token pruning compacts the assembled embeddings before the engine runs. The coordinator currently
    // rejects multi-device pruning, and setVisualPrunerConfig excludes spec decode and action execution.
    if (mVisualPruner && activeBatchSize == 1 && baseKVAllEmpty && !mDeployment.base.isDiffusionBackbone
        && !context.outputThinkerEmbeddings && context.layerDebugger == nullptr)
    {
        int32_t const prunedLen
            = mVisualPruner->pruneForPrefill(context.tokenIds[0], *mPipelineIO, inputIdsLength, context.stream);
        if (prunedLen < inputIdsLength)
        {
            LOG_DEBUG("Visual-token pruning (%s) shortened prefill from %d to %d tokens.", mVisualPruner->name(),
                inputIdsLength, prunedLen);
            context.prunedPrefillTokens.assign(context.effectivePrefillLengths.size(), 0);
            context.prunedPrefillTokens[0] = inputIdsLength - prunedLen;
            context.effectivePrefillLengths[0] = prunedLen;
            hostCtxLenData[0] = prunedLen;
            inputIdsLength = prunedLen;
        }
    }

    // Dispatch per-step sequence prep (context lengths H2D, selectTokenIndices).
    mStepPreparer->prepare(
        InferencePhase::kPrefill, activeBatchSize, *mSharedResources->cacheManagers[0], *mPipelineIO, context.stream);
    // Bind real deepstack features for this prefill (no-op when feature absent).
    if (mDeepstack)
    {
        mDeepstack->useRealFeatures(mBaseTensorMap);
    }

    // Execute base prefill through the EngineExecutor. Empty-cache is
    // runtime-dynamic; prefillDims uses it to set InferenceDims::startIndexLen
    // (0 for the "initial prefill" sentinel, else batch).
    auto const prefillDims = mDeployment.base.prefillDims(activeBatchSize, inputIdsLength, baseKVAllEmpty);
    check::check(mBaseExecutor->prepare(kPrefillProfile, prefillDims, mBaseTensorMap, context.stream),
        "Failed to prepare base model for prefill step.");
    check::check(mBaseExecutor->execute(context.stream), "Failed to execute base model for prefill step.");
    mSharedResources->cacheManagers[0]->commitSequenceLength(mPipelineIO->contextLengths, context.stream);
    if (contextCacheRequest != nullptr && !contextCacheRequest->enqueuePrefillCaptures())
    {
        return false;
    }

    if (mDeployment.base.isDiffusionBackbone)
    {
        return true;
    }

    if (!sampleOutput)
    {
        return true;
    }

    // GCOVR_EXCL_START
    if (context.hasLogitBias)
    {
        applyLogitBias(mLogitBias, mPipelineIO->outputLogits, context, context.stream);
    }
    // GCOVR_EXCL_STOP

    // Sampling from the prefill stage logits follows the same policy as vanilla decoding.
    // DSpark keeps non-greedy params; other speculative decoders are normalized to greedy
    // before decoding.
    check::check(mSamplingIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    if (shouldUseNonGreedySampling(context.temperature, context.topK, context.topP))
    {
        SamplingParams params(activeBatchSize, mDeployment.base.outputVocabSize, context.temperature,
            static_cast<int32_t>(context.topK), context.topP);
        topKtopPSamplingFromLogits(
            mPipelineIO->outputLogits, mSamplingIndices, params, mSamplingWorkspace, context.stream);
    }
    else
    {
        constexpr int32_t kSAMPLING_TOP_K = 1;
        selectAllTopK(mPipelineIO->outputLogits, std::nullopt, mSamplingIndices, kSAMPLING_TOP_K, mSamplingWorkspace,
            context.stream);
    }

    // Apply vocabulary mapping if base model uses reduced vocabulary.
    if (mDeployment.base.reducedVocabSize > 0)
    {
        mapReducedVocabToFullVocab(mSamplingIndices, mBaseVocabMappingTable, context.stream);
    }

    if (!broadcastInt32(mSamplingIndices.rawPointer(), activeBatchSize, context.stream))
    {
        LOG_ERROR("Failed to broadcast prefill sampled tokens across parallel ranks.");
        return false;
    }

    // Enqueue logprobs extraction + D2H before the round's single synchronization so the
    // copies ride the same sync as the sampled-token D2H below.
    if (context.numLogprobs > 0)
    {
        decoder_utils::enqueueLogprobsD2H(mDecodingRuntimeContext->base.pipelineIO.outputLogits, activeBatchSize,
            *mDecodingRuntimeContext, context.numLogprobs, context.stream);
    }

    check::check(mHostSelectedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostSelectedTokenIdsData = mHostSelectedTokenIds.dataPointer<int32_t>();
    CUDA_CHECK(cudaMemcpyAsync(hostSelectedTokenIdsData, mSamplingIndices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Few-layer-validation debug: dump round 0 (prefill). At this point the KV cache is committed and
    // tokenIds[i].size() == the prefill length == the committed cache length.
    if (context.layerDebugger != nullptr)
    {
        std::vector<int32_t> validLengths(activeBatchSize);
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            validLengths[i] = static_cast<int32_t>(context.tokenIds[i].size());
        }
        context.layerDebugger->dumpRound(*mSharedResources->cacheManagers[0], mPipelineIO->outputLogits, validLengths,
            hostSelectedTokenIdsData, activeBatchSize, context.stream);

        // Teacher-forcing: feed the golden tokens instead of sampled tokens when configured.
        context.layerDebugger->applyForcedTokens(
            context.currentGenerateLengths, hostSelectedTokenIdsData, activeBatchSize);
    }

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        if (!context.finishedStates[i])
        {
            context.tokenIds[i].push_back(hostSelectedTokenIdsData[i]);
            context.currentGenerateLengths[i] += 1;
        }
    }

    if (context.numLogprobs > 0)
    {
        decoder_utils::collectLogprobsFromHost(*mDecodingRuntimeContext, context, activeBatchSize, context.numLogprobs);
    }

    emitTokenCallbacks(context);
    return true;
}

bool LLMRankRuntime::captureBaseGraphWithLoraFanout(InferenceDims const& dims, cudaStream_t stream)
{
    auto captureOnce = [&](std::string const& loraName) -> bool {
        if (mSharedResources->loraManager)
        {
            if (loraName.empty())
            {
                mSharedResources->loraManager->resetWeights();
            }
            else
            {
                mSharedResources->loraManager->switchWeights(loraName);
            }
            mSharedResources->loraManager->refreshTensorMap(mBaseTensorMap);
        }
        if (!mBaseExecutor->prepare(kDecodeProfile, dims, mBaseTensorMap, stream))
        {
            return false;
        }
        return mBaseExecutor->captureGraph(stream);
    };

    bool ok = captureOnce(mEmptyLoraWeightsName);
    if (mDeployment.base.maxSupportedLoraRank > 0 && mSharedResources->loraManager)
    {
        for (auto const& loraWeightsName : mSharedResources->loraManager->getAdapterNames())
        {
            ok &= captureOnce(loraWeightsName);
        }
    }
    return ok;
}

bool LLMRankRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
{
    try
    {
        return mDecoderRegistry ? mDecoderRegistry->captureCudaGraphs(stream) : true;
    }
    catch (std::exception const& e)
    {
        LOG_WARNING("CUDA graph capture failed with exception: %s", e.what());
        static_cast<void>(cudaGetLastError());
        return false;
    }
    catch (...)
    {
        LOG_WARNING("CUDA graph capture failed with an unknown exception.");
        static_cast<void>(cudaGetLastError());
        return false;
    }
}

void LLMRankRuntime::restoreRecurrentStates(
    int32_t batchIdx, SystemPromptKVCache const& cachedStates, cudaStream_t stream)
{
    auto& cacheMgrBase = *mSharedResources->cacheManagers[0];
    auto& mambaMgr = cacheMgrBase.getMambaCacheManager();
    auto const& mambaConfig = mambaMgr.getConfig();

    size_t const recurrentElemSize = rt::utils::getTypeSize(mambaConfig.recurrentStateType);
    size_t const convElemSize = rt::utils::getTypeSize(mambaConfig.convStateType);
    size_t const recurrentBatchBytes = static_cast<size_t>(mambaConfig.recurrentStateNumHeads
                                           * mambaConfig.recurrentStateHeadDim * mambaConfig.recurrentStateSize)
        * recurrentElemSize;
    size_t const convBatchBytes = static_cast<size_t>(mambaConfig.convDim * mambaConfig.convKernel) * convElemSize;

    for (int32_t layer = 0; layer < mambaMgr.numLayers(); ++layer)
    {
        rt::Tensor& recurrentLayer = mambaMgr.getRecurrentState(layer);
        rt::Tensor& convLayer = mambaMgr.getConvState(layer);

        auto* recurrentDst = static_cast<std::byte*>(recurrentLayer.rawPointer()) + batchIdx * recurrentBatchBytes;
        auto* convDst = static_cast<std::byte*>(convLayer.rawPointer()) + batchIdx * convBatchBytes;

        if (layer < static_cast<int32_t>(cachedStates.recurrentStateContents.size()))
        {
            CUDA_CHECK(cudaMemcpyAsync(recurrentDst, cachedStates.recurrentStateContents[layer].rawPointer(),
                recurrentBatchBytes, cudaMemcpyDeviceToDevice, stream));
        }
        else
        {
            CUDA_CHECK(cudaMemsetAsync(recurrentDst, 0, recurrentBatchBytes, stream));
        }

        if (layer < static_cast<int32_t>(cachedStates.convStateContents.size()))
        {
            CUDA_CHECK(cudaMemcpyAsync(convDst, cachedStates.convStateContents[layer].rawPointer(), convBatchBytes,
                cudaMemcpyDeviceToDevice, stream));
        }
        else
        {
            CUDA_CHECK(cudaMemsetAsync(convDst, 0, convBatchBytes, stream));
        }
    }
}

void LLMRankRuntime::zeroRecurrentStates(int32_t batchIdx, cudaStream_t stream)
{
    auto& cacheMgrBase = *mSharedResources->cacheManagers[0];
    auto& mambaMgr = cacheMgrBase.getMambaCacheManager();
    auto const& mambaConfig = mambaMgr.getConfig();

    size_t const recurrentElemSize = rt::utils::getTypeSize(mambaConfig.recurrentStateType);
    size_t const convElemSize = rt::utils::getTypeSize(mambaConfig.convStateType);
    size_t const recurrentBatchBytes = static_cast<size_t>(mambaConfig.recurrentStateNumHeads
                                           * mambaConfig.recurrentStateHeadDim * mambaConfig.recurrentStateSize)
        * recurrentElemSize;
    size_t const convBatchBytes = static_cast<size_t>(mambaConfig.convDim * mambaConfig.convKernel) * convElemSize;

    for (int32_t layer = 0; layer < mambaMgr.numLayers(); ++layer)
    {
        rt::Tensor& recurrentLayer = mambaMgr.getRecurrentState(layer);
        rt::Tensor& convLayer = mambaMgr.getConvState(layer);

        auto* recurrentDst = static_cast<std::byte*>(recurrentLayer.rawPointer()) + batchIdx * recurrentBatchBytes;
        auto* convDst = static_cast<std::byte*>(convLayer.rawPointer()) + batchIdx * convBatchBytes;
        CUDA_CHECK(cudaMemsetAsync(recurrentDst, 0, recurrentBatchBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(convDst, 0, convBatchBytes, stream));
    }
}

bool LLMRankRuntime::setUpForPrefillExecution(DecodingInferenceContext& context, DecodingStrategy& strategy,
    std::vector<int32_t> const* contextCachePrefillStarts)
{
    NVTX_SCOPED_RANGE(nvtx_setup, "SETUP_PREFILL_EXECUTION", nvtx_colors::PALE_GREEN);

    // LoRA switching goes through the LoRAManager on SharedResources.
    if (mDeployment.base.maxSupportedLoraRank > 0 && mSharedResources->loraManager)
    {
        try
        {
            if (context.loraWeightsName.empty())
            {
                mSharedResources->loraManager->resetWeights();
            }
            else
            {
                mSharedResources->loraManager->switchWeights(context.loraWeightsName);
            }
            mSharedResources->loraManager->refreshTensorMap(mBaseTensorMap);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to switch LoRA weights to %s: %s", context.loraWeightsName.c_str(), e.what());
            return false;
        }
    }

    int32_t const activeBatchSize = context.activeBatchSize;
    std::vector<std::vector<int32_t>> const& batchedInputIds = context.rawBatchedInputIds;
    bool const needsStrategyKVCache = strategy.isSpeculative();
    auto& cacheMgrBase = *mSharedResources->cacheManagers[0];

    context.tokenIds.clear();
    context.tokenIds.resize(activeBatchSize);

    if (contextCachePrefillStarts != nullptr)
    {
        ELLM_CHECK(mContextCache != nullptr,
            "Managed context-cache prefill requires an initialized context-cache coordinator");
        ELLM_CHECK(static_cast<int32_t>(contextCachePrefillStarts->size()) == activeBatchSize,
            "Managed context-cache execution recipe must describe every active sequence");
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            int32_t const prefillStart = (*contextCachePrefillStarts)[static_cast<size_t>(i)];
            ELLM_CHECK(prefillStart >= 0 && prefillStart < static_cast<int32_t>(batchedInputIds[i].size()),
                "Managed context-cache prefill boundary is outside the input sequence");
            context.tokenIds[i].assign(batchedInputIds[i].begin() + prefillStart, batchedInputIds[i].end());
            context.effectivePrefillLengths[i] = static_cast<int32_t>(context.tokenIds[i].size());
        }
    }
    else
    {
        for (auto& pageTable : mSharedResources->kvPageTables)
        {
            if (!pageTable->isIdentity())
            {
                pageTable->setIdentity();
                pageTable->upload(context.stream);
            }
        }

        // Record the length of the reused legacy system-prompt KV cache for each sequence.
        check::check(mHostReuseKVCacheLengths.reshape({activeBatchSize}), "Tensor reshape failed");
        int32_t* reuseKVCacheLengthsData = mHostReuseKVCacheLengths.dataPointer<int32_t>();
        std::fill(reuseKVCacheLengthsData, reuseKVCacheLengthsData + activeBatchSize, 0);

        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            auto const& prompt = context.systemPrompts[i];
            auto const promptKey = keySystemPromptWithLoraWeights(prompt, context.loraWeightsName);
            if (mSystemPromptKVCacheBase.count(promptKey) > 0)
            {
                auto& precachedKVCacheBase = mSystemPromptKVCacheBase[promptKey];
                auto const& kvCacheLayersBase = precachedKVCacheBase.kvCacheLayers;
                cacheMgrBase.restoreKVCache(kvCacheLayersBase, i, context.stream);

                if (needsStrategyKVCache)
                {
                    check::check(strategy.hasSystemPromptKVCache(promptKey),
                        "System prompt cache inconsistency between base and active decoding strategy");
                    strategy.restoreSystemPromptKVCache(promptKey, i, context.stream);
                }

                // Restore recurrent/conv states for hybrid models (vanilla path only — spec decode handles this in
                // decoder).
                if (mDeployment.base.numLinearAttnLayers > 0)
                {
                    restoreRecurrentStates(i, precachedKVCacheBase, context.stream);
                }

                // Cached token length comes from the tokenized prompt that was actually captured, not from
                // any KV-tensor's physical shape (see computeSystemPromptReuse) — this also covers
                // pure-recurrent models, whose kvCacheLayersBase is empty.
                auto reuse = computeSystemPromptReuse(precachedKVCacheBase, batchedInputIds[i]);
                reuseKVCacheLengthsData[i] = reuse.reuseKVCacheLength;
                context.tokenIds[i] = std::move(reuse.tokenIds);
                context.effectivePrefillLengths[i] = reuse.effectivePrefillLength;

                bool const matchIds = std::equal(precachedKVCacheBase.tokenizedPrompt.begin(),
                    precachedKVCacheBase.tokenizedPrompt.end(), batchedInputIds[i].begin());
                if (!matchIds)
                {
                    LOG_WARNING(
                        "Though system prompt strings are matched, token_ids are not perfectly aligned."
                        "This may generate incorrect result, please check your system prompt design.");
                }
            }
            else
            {
                context.tokenIds[i] = batchedInputIds[i];
                context.effectivePrefillLengths[i] = static_cast<int32_t>(batchedInputIds[i].size());
                reuseKVCacheLengthsData[i] = 0;

                if (mDeployment.base.numLinearAttnLayers > 0)
                {
                    zeroRecurrentStates(i, context.stream);
                }
            }
        }
    }

    int32_t const maxInputLength
        = *std::max_element(context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.end());
    if (maxInputLength > mDeployment.base.maxSupportedInputLength)
    {
        LOG_ERROR("The max input length (%d) exceeds the max supported input length (%d) of the LLM Engine.",
            maxInputLength, mDeployment.base.maxSupportedInputLength);
        return false;
    }

    if (contextCachePrefillStarts == nullptr)
    {
        mSharedResources->cacheManagers[0]->resetForNewSequences(mHostReuseKVCacheLengths, context.stream);
        if (needsStrategyKVCache)
        {
            strategy.resetForNewSequences(mHostReuseKVCacheLengths, context.stream);
        }
    }
    return true;
}

bool LLMRankRuntime::genAndSaveSystemPromptKVCache(DecodingInferenceContext& context, int32_t genAndSaveBatchIdx)
{
    if (mContextCache != nullptr)
    {
        LOG_ERROR("Legacy system-prompt KV-cache capture cannot be combined with the context-cache manager.");
        return false;
    }
    if (mDeployment.base.useVisionBidirectionalAttention)
    {
        LOG_ERROR("System-prompt KV-cache reuse is not supported with Gemma4 vision bidirectional attention.");
        return false;
    }

    std::string const& loraWeightsName = context.loraWeightsName;
    std::string const prompt = context.systemPrompts[genAndSaveBatchIdx];
    auto const promptKey = keySystemPromptWithLoraWeights(prompt, loraWeightsName);

    if (prompt.empty())
    {
        LOG_DEBUG("The systemPrompt is empty. Skip saving system prompt KVCache.");
        return true;
    }

    DecodingStrategy& cacheStrategy = mDecoderRegistry->cachePrimingStrategy();
    bool const hasDraft = cacheStrategy.isSpeculative();
    auto baseCacheIt = mSystemPromptKVCacheBase.find(promptKey);
    if (baseCacheIt != mSystemPromptKVCacheBase.end() && (!hasDraft || cacheStrategy.hasSystemPromptKVCache(promptKey)))
    {
        LOG_DEBUG("The system prompt KVCache already exists for the prompt: {%s}", prompt.c_str());
        return true;
    }
    if (baseCacheIt != mSystemPromptKVCacheBase.end())
    {
        mSystemPromptKVCacheBase.erase(baseCacheIt);
    }

    auto tokenizedPrompt = mTokenizer->encode(prompt, true);
    if (tokenizedPrompt.empty())
    {
        LOG_ERROR("Failed to encode system prompt for KVCache generation.");
        return false;
    }
    int32_t const promptIdsLength = static_cast<int32_t>(tokenizedPrompt.size());

    if (promptIdsLength > mDeployment.base.maxSupportedInputLength)
    {
        LOG_ERROR("System prompt length (%d) exceeds max supported input length (base=%d)", promptIdsLength,
            mDeployment.base.maxSupportedInputLength);
        return false;
    }

    if (hasDraft && promptIdsLength > mDeployment.draft->maxSupportedInputLength)
    {
        LOG_ERROR("System prompt length (%d) exceeds max supported input length (draft=%d)", promptIdsLength,
            mDeployment.draft->maxSupportedInputLength);
        return false;
    }

    // Temporary single-batch context to reuse the existing prefill functions.
    DecodingInferenceContext tempContext;
    tempContext.initialize(1, 1, context.visualEmbeddings, context.deepstackFeatures, loraWeightsName, context.stream);
    tempContext.systemPrompts[0] = prompt;
    tempContext.rawBatchedInputIds.push_back(tokenizedPrompt);
    tempContext.tokenIds[0] = tokenizedPrompt;

    if (!setUpForPrefillExecution(tempContext, cacheStrategy))
    {
        LOG_ERROR("Prefill execution setup failed for system prompt KVCache generation.");
        return false;
    }

    bool prefillStatus = runBaseModelPrefill(tempContext);
    if (!prefillStatus)
    {
        LOG_ERROR("Failed to execute base model prefill for system prompt KVCache generation.");
        return false;
    }

    // Tokens produced during system KV-cache reuse prefill do not count as generated tokens.
    tempContext.currentGenerateLengths[0] -= 1;

    if (hasDraft)
    {
        bool draftPrefillStatus = cacheStrategy.runSystemPromptPrefill(tempContext);
        if (!draftPrefillStatus)
        {
            LOG_ERROR("Failed to execute draft model prefill for system prompt KVCache generation.");
            return false;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Capture base KV cache content from the new-stack shared KV cache.
    auto& cacheMgrBase = *mSharedResources->cacheManagers[0];
    constexpr int32_t CACHE_BATCH_IDX{0};

    SystemPromptKVCache savedKVCacheBase;
    savedKVCacheBase.systemPrompt = prompt;
    savedKVCacheBase.tokenizedPrompt = tokenizedPrompt;
    savedKVCacheBase.kvCacheLayers = cacheMgrBase.captureKVCache(CACHE_BATCH_IDX, promptIdsLength, context.stream);

    // Save recurrent / conv states for hybrid layers.
    if (mDeployment.base.numLinearAttnLayers > 0)
    {
        savedKVCacheBase.recurrentStateContents = cacheMgrBase.captureRecurrentStates(CACHE_BATCH_IDX, context.stream);
        savedKVCacheBase.convStateContents = cacheMgrBase.captureConvStates(CACHE_BATCH_IDX, context.stream);
    }

    mSystemPromptKVCacheBase.insert({promptKey, std::move(savedKVCacheBase)});

    cacheStrategy.saveSystemPromptKVCache(promptKey, prompt, tokenizedPrompt, promptIdsLength, context.stream);

    CUDA_CHECK(cudaStreamSynchronize(context.stream));
    LOG_DEBUG("System prompt KVCache saved for batch %d: {%s}", genAndSaveBatchIdx, prompt.c_str());

    return true;
}

bool LLMRankRuntime::genAndSaveSystemPromptKVCache(
    std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream)
{
    if (mDeployment.base.useVisionBidirectionalAttention)
    {
        LOG_ERROR("System-prompt KV-cache reuse is not supported with Gemma4 vision bidirectional attention.");
        return false;
    }

    if (prompt.empty())
    {
        LOG_DEBUG("The systemPrompt is empty. Skip saving system prompt KVCache.");
        return true;
    }
    auto const promptKey = keySystemPromptWithLoraWeights(prompt, loraWeightsName);
    DecodingStrategy& cacheStrategy = mDecoderRegistry->cachePrimingStrategy();
    if (mSystemPromptKVCacheBase.find(promptKey) != mSystemPromptKVCacheBase.end()
        && (!cacheStrategy.isSpeculative() || cacheStrategy.hasSystemPromptKVCache(promptKey)))
    {
        LOG_DEBUG("The system prompt KVCache already exists for the prompt: {%s}", prompt.c_str());
        return true;
    }
    DecodingInferenceContext tempContext;
    tempContext.initialize(1, 1, std::nullopt, rt::OptionalInputTensors{}, loraWeightsName, stream);
    tempContext.systemPrompts[0] = prompt;
    auto tokenizedPrompt = mTokenizer->encode(prompt, true);
    if (tokenizedPrompt.empty())
    {
        LOG_ERROR("Failed to encode system prompt for KVCache generation.");
        return false;
    }
    tempContext.rawBatchedInputIds.push_back(tokenizedPrompt);
    tempContext.tokenIds[0] = tokenizedPrompt;
    return genAndSaveSystemPromptKVCache(tempContext, 0);
}

bool LLMRankRuntime::performBatchEvict(DecodingInferenceContext& context, DecodingStrategy& strategy,
    std::vector<int8_t>& thinkingDone, ContextCacheRequest* contextCacheRequest)
{
    // Check if any batch has finished
    bool hasFinishedBatch = false;
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        if (context.finishedStates[i])
        {
            hasFinishedBatch = true;
            break;
        }
    }

    if (!hasFinishedBatch)
    {
        return true;
    }

    int32_t const oldActiveBatch = context.activeBatchSize;

    // Build batch mapping
    std::vector<int32_t> batchMapping = buildBatchMapping(context.finishedStates);

    // Calculate new active batch size
    int32_t newActiveBatch = 0;
    for (auto newIdx : batchMapping)
    {
        if (newIdx >= 0)
        {
            newActiveBatch = std::max(newActiveBatch, newIdx + 1);
        }
    }

    // Log eviction details
    std::vector<int32_t> evictedIndices;
    for (int32_t i = 0; i < oldActiveBatch; ++i)
    {
        if (batchMapping[i] < 0)
        {
            evictedIndices.push_back(i);
        }
    }
    LOG_DEBUG("Batch eviction: %d active batches to %d remaining (evicted %d batch(es): indices [%s])", oldActiveBatch,
        newActiveBatch, static_cast<int32_t>(evictedIndices.size()),
        [&evictedIndices]() {
            std::string result;
            for (size_t i = 0; i < evictedIndices.size(); ++i)
            {
                if (i > 0)
                {
                    result += ", ";
                }
                result += std::to_string(evictedIndices[i]);
            }
            return result;
        }()
            .c_str());

    bool const managedContextCache = contextCacheRequest != nullptr;
    if (managedContextCache)
    {
        if (!contextCacheRequest->beginBatchCompaction(batchMapping, newActiveBatch, mDeviceBatchMapping))
        {
            return false;
        }
    }
    else
    {
        check::check(mDeviceBatchMapping.reshape({oldActiveBatch}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mDeviceBatchMapping.rawPointer(), batchMapping.data(),
            static_cast<size_t>(oldActiveBatch) * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
        ELLM_CHECK(mSharedResources->cacheManagers.size() == mSharedResources->kvPageTables.size(),
            "KV cache managers and page tables are not index-aligned");
        size_t const activeCacheCount = strategy.isSpeculative() ? mSharedResources->cacheManagers.size() : 1U;
        ELLM_CHECK(activeCacheCount > 0, "Active decoding strategy has no KV cache resource");
        for (size_t cacheIndex = 0; cacheIndex < activeCacheCount; ++cacheIndex)
        {
            auto& pageTable = *mSharedResources->kvPageTables[cacheIndex];
            auto& cacheManager = *mSharedResources->cacheManagers[cacheIndex];
            pageTable.compactRows(batchMapping, newActiveBatch);
            pageTable.upload(context.stream);
            cacheManager.compactBatchSlotState(mDeviceBatchMapping, oldActiveBatch, newActiveBatch, context.stream);
            cacheManager.setActiveBatchSize(newActiveBatch);
        }
    }

    // Compact base model's RoPE cache (stored per-batch for MRope on mPipelineIO->mropeCosSin).
    if (mDeployment.base.ropeConfig.type == RopeType::kMRope && newActiveBatch > 0)
    {
        rt::Tensor& baseRopeCache = mPipelineIO->mropeCosSin;
        if (baseRopeCache.getShape().getNumDims() == 3 && baseRopeCache.getShape()[0] == oldActiveBatch)
        {
            kernel::compactTensorBatch(
                baseRopeCache, mDeviceBatchMapping, baseRopeCache, oldActiveBatch, newActiveBatch, context.stream);
            auto const seqLen = static_cast<int32_t>(baseRopeCache.getShape()[1]);
            auto const rotaryDim = static_cast<int32_t>(baseRopeCache.getShape()[2]);
            check::check(baseRopeCache.reshape({newActiveBatch, seqLen, rotaryDim}), "Tensor reshape failed");
        }
    }

    strategy.onBatchEvict(batchMapping, oldActiveBatch, newActiveBatch, mDeviceBatchMapping, context.stream);

    // Consume the existing eviction synchronization. Managed paging moves page-table rows and slot state only;
    // physical KV pages remain in place.
    if (managedContextCache)
    {
        if (!contextCacheRequest->completeBatchCompaction())
        {
            return false;
        }
    }
    else
    {
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
    }

    // Save evicted batches' results before compacting (using original batch index)
    for (size_t i = 0; i < batchMapping.size(); ++i)
    {
        if (batchMapping[i] < 0 && context.finishedStates[i])
        {
            // This batch is evicted and finished, save its results with original index
            int32_t originalIdx = context.batchIndexMapping[i];

            // Create and populate BatchResult with all related data
            BatchResult result;
            result.tokenIds = std::move(context.tokenIds[i]);
            result.generateLength = context.currentGenerateLengths[i];
            result.actualIterations = context.generationRound;
            result.rawBatchedInputIds = std::move(context.rawBatchedInputIds[i]);
            result.effectivePrefillLength = context.effectivePrefillLengths[i];
            result.prunedPrefillTokens = i < context.prunedPrefillTokens.size() ? context.prunedPrefillTokens[i] : 0;
            result.terminalReason = context.slotStreams[i].terminalReason;
            // Convert flat LogprobsSlot -> nested vector for BatchResult (once per completed request).
            // Enrich each (token_id, logprob) with the raw token piece so consumers can render the
            // token string / bytes without needing a tokenizer (see LogprobEntry).
            rt::LogprobsSlot const& slot = context.stepLogprobs[i];
            result.logprobs.resize(slot.numSteps);
            for (int32_t step = 0; step < slot.numSteps; ++step)
            {
                auto const* begin = slot.data.data() + step * context.numLogprobs;
                auto& stepEntries = result.logprobs[step];
                stepEntries.reserve(context.numLogprobs);
                for (int32_t k = 0; k < context.numLogprobs; ++k)
                {
                    stepEntries.push_back({begin[k].first, begin[k].second, mTokenizer->idToPiece(begin[k].first)});
                }
            }

            context.completedBatches[originalIdx] = std::move(result);
        }
    }

    rt::compactVector(batchMapping, context.finishedStates);
    if (contextCacheRequest != nullptr)
    {
        rt::compactVector(batchMapping, thinkingDone);
    }
    rt::compactVector(batchMapping, context.currentGenerateLengths);
    rt::compactVector(batchMapping, context.tokenIds);
    rt::compactVector(batchMapping, context.systemPrompts);
    rt::compactVector(batchMapping, context.rawBatchedInputIds);
    rt::compactVector(batchMapping, context.effectivePrefillLengths);
    rt::compactVector(batchMapping, context.batchIndexMapping);
    rt::compactVector(batchMapping, context.callbackEmittedTokenCounts);
    rt::compactVector(batchMapping, context.slotStreams);
    rt::compactVector(batchMapping, context.stopStringsPerSlot);
    rt::compactVector(batchMapping, context.logitBiasPerSlot);
    context.hasLogitBias = std::any_of(context.logitBiasPerSlot.begin(), context.logitBiasPerSlot.end(),
        [](auto const& slotLogitBias) { return !slotLogitBias.empty(); });
    context.logitBiasGpuDirty = context.hasLogitBias;
    rt::compactVector(batchMapping, context.stepLogprobs);

    // Update active batch size
    context.activeBatchSize = newActiveBatch;

    return true;
}

} // namespace rt
} // namespace trt_edgellm
