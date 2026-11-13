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
#include "chatTemplate/chatTemplate.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/identityHash.h"
#include "common/inputLimits.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/pagedKvTypes.h"
#include "common/parallelArtifactNames.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "kernels/speculative/batchEvictKernels.h"
#include "multimodal/common/multimodalRunner.h"
#include "multimodal/qwen2/qwenViTRunner.h"
#include "multimodal/qwen3/qwen3vlViTRunner.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/config/compactCapability.h"
#include "runtime/contextCacheRequest.h"
#include "runtime/debug/layerDebugger.h"
#include "runtime/decoding/decoderRegistry.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/requestStableRng.h"
#include "runtime/deviceFailure.h"
#include "runtime/embedding/pleIoBudget.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/exec/rtxRuntimeCache.h"
#include "runtime/imageUtils.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/managedKVCacheRequest.h"
#include "runtime/runtimeSession.h"
#include "runtime/state/boundedSwaKVPageManager.h"
#include "runtime/state/contextCache/contextCacheCoordinator.h"
#include "runtime/state/kvPageTable.h"
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

//! Restores the base cache manager's batch on scope exit: a compact step materializes the step's
//! sequence count into it, while admission and eviction require it to match the context's.
struct CacheBatchRestore
{
    rt::HybridCacheManager& cacheManager;
    int32_t batch;
    ~CacheBatchRestore() noexcept
    {
        cacheManager.restoreActiveBatchSize(batch);
    }
};

} // namespace

namespace rt
{

LLMRankRuntime::LLMRankRuntime(DeploymentOptions const& options, cudaStream_t stream, ParallelMapping const& mapping,
    tokenizer::Tokenizer& tokenizer, chat_template::ChatTemplate const& chatTemplate,
    std::shared_ptr<PleIoBudget> pleIoBudget)
    : mPleIoBudget(pleIoBudget ? std::move(pleIoBudget) : std::make_shared<PleIoBudget>())
{
    initializeFromEngineDir(options, stream, mapping, tokenizer, chatTemplate);
}

LLMRankRuntime::LLMRankRuntime(ModelArtifacts&& artifacts, DeploymentOptions const& options, cudaStream_t stream,
    ParallelMapping const& mapping, tokenizer::Tokenizer& tokenizer, chat_template::ChatTemplate const& chatTemplate,
    std::shared_ptr<PleIoBudget> pleIoBudget)
    : mPleIoBudget(pleIoBudget ? std::move(pleIoBudget) : std::make_shared<PleIoBudget>())
{
    artifacts.validateDeploymentOptions(options);
    initializeCommon(std::move(artifacts), options, stream, mapping, tokenizer, chatTemplate);
}

LLMRankRuntime::~LLMRankRuntime()
{
    if (mBoundedSwaKVPageManager != nullptr && mBoundedSwaKVPageManager->shutdown() != SwaKVCacheStatus::kOk)
    {
        LOG_ERROR("SWA KV-cache shutdown could not prove stream quiescence.");
        std::terminate();
    }
    if (mContextCache != nullptr && mContextCache->shutdown() != ContextCacheCoordinatorStatus::kOk)
    {
        LOG_ERROR("Context-cache shutdown could not prove stream quiescence.");
        std::terminate();
    }
}

void LLMRankRuntime::initializeFromEngineDir(DeploymentOptions const& options, cudaStream_t stream,
    ParallelMapping const& mapping, tokenizer::Tokenizer& tokenizer, chat_template::ChatTemplate const& chatTemplate)
{
    auto const& engineDir = options.engineDir;
    auto const& draftingConfig = options.draftingConfig;
    auto const& checkpointDir = options.checkpointDir;
    auto const& draftCheckpointDir = options.draftCheckpointDir;
    auto const& runtimeCachePath = options.runtimeCachePath;
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
        artifacts.embedding
            = loadEmbeddingTable(engineRoot / "embedding.safetensors", stream, options.offloadEmbedding);
    }

    // Parse engine configurations and attach user drafting. The bundle factory
    // performs cross-engine consistency and drafting-vs-capacity checks.
    std::optional<std::filesystem::path> const draftConfigPath = draftingConfig.has_value()
        ? std::optional<std::filesystem::path>{engineRoot / "draft_config.json"}
        : std::nullopt;

    artifacts.deployment
        = createDeploymentConfig(baseConfigPath, draftConfigPath, draftingConfig, globalRank, worldSize);
    // EngineExecutor captures the active KV-pool geometry in its tensor registry.
    // Select the immutable runtime policy before constructing that registry.
    artifacts.deployment.selectSwaKVCacheMode(options.contextCacheConfig.enabled);
    artifacts.deployment.runtimeCachePath = runtimeCachePath;
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

    std::shared_ptr<RtxRuntimeCache> runtimeCache;
#if IS_TRT_RTX
    runtimeCache = RtxRuntimeCache::create(baseEnginePath, artifacts.deployment.runtimeCachePath);
#endif

    try
    {
        std::optional<int32_t> const specDecodeBaseOutputHiddenDim = artifacts.deployment.specConfig.has_value()
            ? std::optional<int32_t>{artifacts.deployment.specConfig->baseOutputHiddenDim}
            : std::nullopt;
        artifacts.baseExecutor
            = EngineExecutor::createForLLM(baseEnginePath, artifacts.deployment.base, specDecodeBaseOutputHiddenDim,
                runtimeCache, EngineRefitOptions{artifacts.weights.takeRefitWeights(), stream});
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize base EngineExecutor: %s", e.what());
        throw std::runtime_error("Failed to initialize base EngineExecutor: " + std::string(e.what()));
    }
    LOG_INFO("Base EngineExecutor successfully loaded from %s.", baseEnginePath.c_str());

    artifacts.weights.validateAgainstEngine(*artifacts.baseExecutor, "base");
    validateAgainstEngine(artifacts.deployment.base, *artifacts.baseExecutor, "base");
    validateRecurrentVerifyMetadataBindings(artifacts.deployment, *artifacts.baseExecutor);

    // Validate the draft engine ABI before its sidecar geometry is used to allocate
    // physical cache resources. Ownership is transferred to the selected decoder.
    if (draftingConfig.has_value())
    {
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
        artifacts.draftExecutor = decoder_utils::loadDraftEngine(engineRoot, artifacts.deployment, runtimeCache,
            EngineRefitOptions{artifacts.draftWeights.takeRefitWeights(), stream});
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

    initializeCommon(std::move(artifacts), options, stream, mapping, tokenizer, chatTemplate);
}

void LLMRankRuntime::initializeCommon(ModelArtifacts&& artifacts, DeploymentOptions const& options, cudaStream_t stream,
    ParallelMapping const& mapping, tokenizer::Tokenizer& tokenizer, chat_template::ChatTemplate const& chatTemplate)
{
    auto const& engineDir = options.engineDir;
    auto const& multimodalEngineDir = options.multimodalEngineDir;
    auto const& loraWeightsMap = options.loraWeightsMap;
    auto const draftingConfig = artifacts.deployment.draftingConfig();
    auto const& contextCacheConfig = options.contextCacheConfig;
    auto const& pleConfig = options.pleConfig;
    ELLM_CHECK(artifacts.baseExecutor != nullptr, "Model artifacts require a base engine executor.");
    validateRecurrentVerifyMetadataBindings(artifacts.deployment, *artifacts.baseExecutor);

    ELLM_CHECK(options.benchmark.maxDecodeRounds >= 0
            && (options.benchmark.forcedAcceptanceLength == -1 || options.benchmark.forcedAcceptanceLength > 0),
        "Invalid benchmark round or acceptance budget");
    ELLM_CHECK(!options.benchmark.enabled() || mapping.tensorParallelSize == 1,
        "Benchmark execution controls currently require single-device execution");
    ELLM_CHECK(!options.benchmark.enabled() || !contextCacheConfig.enabled,
        "Benchmark execution controls require context reuse disabled");
    ELLM_CHECK(options.benchmark.forcedAcceptanceLength < 0 || artifacts.deployment.draft.has_value(),
        "Forced acceptance requires a speculative deployment");
    mBenchmarkOptions = options.benchmark;
    mProfilingTimer->setGpuTimingEnabled(options.benchmark.gpuTiming);
    selectProfilingTimer();

    mMapping = mapping;
    mTokenizer = &tokenizer;
    mChatTemplate = &chatTemplate;
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
    auto vocabMap = std::move(artifacts.vocabMap);

    // The same engine uses bounded SWA pages whenever the active runtime mode supports them and the reduced pool
    // saves memory.
    mDeployment.selectSwaKVCacheMode(contextCacheConfig.enabled);
    if (mDeployment.base.supportsBoundedSwaKVCache())
    {
        LOG_INFO("SWA KV cache storage mode: %s (context reuse %s, bounded pages %d, full pages %d).",
            mDeployment.base.usesBoundedSwaKVCache() ? "bounded" : "full",
            contextCacheConfig.enabled ? "enabled" : "disabled", mDeployment.base.numSwaPages,
            mDeployment.base.kvPoolPages);
    }
    bool const needsBoundedSwaPageManager = mDeployment.base.usesBoundedSwaKVCache();
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
        auto const hasCompleteTreeMetadata = [](EngineExecutor const& executor, char const* role) {
            bool const hasTreeParentIds = executor.hasIOTensor(binding_names::kTreeParentIds);
            bool const hasTreeDepths = executor.hasIOTensor(binding_names::kTreeDepths);
            ELLM_CHECK(hasTreeParentIds == hasTreeDepths,
                std::string("Speculative ") + role
                    + " engine must expose both tree_parent_ids and tree_depths, or neither.");
            return hasTreeParentIds;
        };
        bool const baseHasTreeMetadata = hasCompleteTreeMetadata(*mBaseExecutor, "base");
        bool const draftHasTreeMetadata = hasCompleteTreeMetadata(*draftExecutor, "draft");
        bool const hasTreeMetadataInputs = baseHasTreeMetadata || draftHasTreeMetadata;
        mPipelineIO = std::make_unique<PipelineIO>(PipelineIO::createForSpecDecode(mDeployment, mMaxRuntimeBatchSize,
            stream, mBaseExecutor->hasIOTensor(binding_names::kAcceptHiddenStates), hasTreeMetadataInputs));
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
    int32_t const effectiveDraftTopK = hasDraft ? mDeployment.specConfig->draftTopK : 1;
    int32_t const diffusionCanvasLen
        = mDeployment.base.isDiffusionBackbone ? std::max(1, mDeployment.base.diffusionCanvasLength) : 1;
    int32_t const maxInputLength = hasDraft ? std::max({mDeployment.base.maxSupportedInputLength,
                                                  mDeployment.draft->maxSupportedInputLength, effectiveMaxProposalSize})
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
        : getRowSamplingWorkspaceSize(vanillaSamplingRows, mDeployment.base.outputVocabSize);
    bool const isDSparkDraft = hasDraft && mDeployment.specDecodeMode() == SpecDecodeMode::kDSpark;
    bool const isDFlash2Draft = hasDraft && mDeployment.draft->dflashVersion == DFlashVersion::kV2;
    constexpr int32_t kDSparkMaxSparseTopK = 128;
    bool const isLinearBlockDraft = hasDraft
        && (isCachedBlockDraftMode(mDeployment.specDecodeMode())
            || mDeployment.specDecodeMode() == SpecDecodeMode::kDSpark);
    int32_t const draftSamplingRows = isLinearBlockDraft
        ? mMaxRuntimeBatchSize * mDeployment.specConfig->verifyTokenCount
        : mMaxRuntimeBatchSize * effectiveDraftTopK;
    int32_t const draftSamplingTopK
        = isLinearBlockDraft ? (isDSparkDraft ? kDSparkMaxSparseTopK : 1) : effectiveDraftTopK;
    // A DFlash2 draft emits selector support instead of logits, so no draft tokens are selected from this workspace.
    size_t const draftTopKWorkspaceSize = isDFlash2Draft
        ? 0
        : getSelectAllTopKWorkspaceSize(draftSamplingRows, mDeployment.draft->outputVocabSize, draftSamplingTopK);
    // Only the DFlash2 chain verifies with target sparse/dense probabilities.
    bool const isDFlash2ChainDraft = isDFlash2Draft && mDeployment.specConfig->draftTopK == 1;
    size_t const dsparkBaseTopKWorkspaceSize = (isDSparkDraft || isDFlash2ChainDraft)
        ? getSelectAllTopKWorkspaceSize(mMaxRuntimeBatchSize * mDeployment.specConfig->verifyTokenCount,
              mDeployment.base.outputVocabSize, kDSparkMaxSparseTopK)
        : 0;
    size_t const dflash2TopPWorkspaceSize = isDFlash2ChainDraft
        ? getTopPProbabilitiesWorkspaceSize(
              mMaxRuntimeBatchSize * mDeployment.specConfig->verifyTokenCount, mDeployment.base.outputVocabSize)
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
              draftTopKWorkspaceSize, dsparkBaseTopKWorkspaceSize, dflash2TopPWorkspaceSize, logprobsWorkspaceSize})
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
        int64_t const samplingRowBytes = static_cast<int64_t>(mMaxRuntimeBatchSize * sizeof(SamplingRow));
        mSamplingRows
            = rt::Tensor({samplingRowBytes}, rt::DeviceType::kGPU, DataType::kINT8, "LLMRankRuntime::mSamplingRows");
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

        mHostRaggedTokenIds = rt::Tensor({mDeployment.base.maxPhysicalTokens}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMRankRuntime::mHostRaggedTokenIds");
        mHostDecoderTokenIds = rt::Tensor({mMaxRuntimeBatchSize, maxInputLength}, rt::DeviceType::kCPU,
            DataType::kINT32, "LLMRankRuntime::mHostDecoderTokenIds");
        mHostSelectedTokenIds = rt::Tensor(
            {maxSamplingSize}, rt::DeviceType::kCPU, DataType::kINT32, "LLMRankRuntime::mHostSelectedTokenIds");
        mHostOutputSpaceIds = rt::Tensor(
            {maxSamplingSize}, rt::DeviceType::kCPU, DataType::kINT32, "LLMRankRuntime::mHostOutputSpaceIds");
        mHostSamplingRows = rt::Tensor({static_cast<int64_t>(mMaxRuntimeBatchSize * sizeof(SamplingRow))},
            rt::DeviceType::kCPU, DataType::kINT8, "LLMRankRuntime::mHostSamplingRows");
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
    ELLM_CHECK(pleConfig.residency != PleResidency::kCACHED
            || (mDeployment.base.usesPlePool() && mapping.tensorParallelSize == 1
                && !mDeployment.base.isDiffusionBackbone),
        "Cached PLE currently requires a PLE-enabled engine with single-device decoding");
    if (mDeployment.base.gemmaPleEnabled)
    {
        int32_t const maxPleSeqLen = mDeployment.base.maxQueryLength;
        mGemma4Ple = std::make_unique<Gemma4EmbeddingPreprocessor>(mDeployment.base, mMaxRuntimeBatchSize, maxPleSeqLen,
            mBaseTensorMap, stream, pleConfig, mCheckpointDir, mPleIoBudget);
    }
    if (mDeployment.base.ngramEnabled())
    {
        int32_t const maxNgramSeqLen = std::max(maxInputLength, std::max(1, mDeployment.base.maxSpecVerifyTokens));
        mQwen4ExpNgram = std::make_unique<Qwen4ExpNgramPreprocessor>(mDeployment.base, mMaxRuntimeBatchSize,
            maxNgramSeqLen, mBaseTensorMap, stream, pleConfig, mCheckpointDir, mPleIoBudget);
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
    // 12. Guided decoding. Needs the tokenizer and the reduced-vocab map above.
    // -----------------------------------------------------------------------
    int32_t const guidedMaxRowsPerSlot
        = mDeployment.specConfig.has_value() ? mDeployment.specConfig->jumpVerifyTokenCount : 1;
    mGuidedDecoder.initialize(mMaxRuntimeBatchSize, guidedMaxRowsPerSlot, mDeployment.base.outputVocabSize,
        mDeployment.base.vocabSize, mTokenizer, mBaseVocabMappingTable, stream);

    // -----------------------------------------------------------------------
    // 13. Decoding strategies.
    // -----------------------------------------------------------------------
    buildDecodingRuntimeContext();
    if (mDeployment.specConfig.has_value())
    {
        mDecodingRuntimeContext->specPleStaging = SpecPleStaging(
            *mDecodingRuntimeContext, mDeployment.specConfig->maxVerifyTokens, mDeployment.specConfig->verifyLayout);
    }
    mDecoderRegistry = std::make_unique<DecoderRegistry>(*mDecodingRuntimeContext,
        DecoderRegistryInit{std::filesystem::path(engineDir), draftingConfig, std::move(draftExecutor),
            std::move(draftWeights), stream});

    // -----------------------------------------------------------------------
    // 14. Optional multimodal runners.
    // -----------------------------------------------------------------------
    mVisionRunner = std::move(artifacts.visionRunner);
    mAudioRunner = std::move(artifacts.audioRunner);
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
                mDeployment.base.maxKVCacheCapacity, stream, mCheckpointDir.string());
            LOG_INFO("%s runner successfully initialized", name.c_str());
            return runner;
        };

        if (!mAudioRunner)
        {
            mAudioRunner = loadRunner(multimodalEngineDir + "/audio", "audio_encoder.engine", "Audio");
        }
        if (!mVisionRunner)
        {
            mVisionRunner = loadRunner(multimodalEngineDir + "/visual", "visual.engine", "Visual");
        }
        if (!mVisionRunner)
        {
            mVisionRunner = loadRunner(multimodalEngineDir, "visual.engine", "Vision");
        }

        // At least one multimodal runner must be available
        ELLM_CHECK(mAudioRunner || mVisionRunner, "No valid multimodal engine found in " + multimodalEngineDir);

        std::string const actionDir = multimodalEngineDir + "/action";
        if (std::filesystem::exists(std::filesystem::path(actionDir) / "action.engine"))
        {
            ELLM_CHECK(mapping.worldSize == mapping.tensorParallelSize,
                "Alpamayo action multi-device inference currently supports tensor parallelism only");
            ELLM_CHECK(!mDeployment.base.ropeConfig.yarn.has_value(),
                "Alpamayo action runner does not support YaRN RoPE scaling");
            mActionRunner = std::make_unique<Alpamayo1ActionRunner>(actionDir, mCheckpointDir.string(), stream,
                mSharedResources->cacheManagers[0]->getKVCacheManager().getConfig(), mapping.tensorParallelSize,
                mapping.tensorParallelRank, mDeployment.base.kvCacheScales);
            LOG_INFO("Alpamayo 1 action expert loaded.");
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

    RopeConfig const& baseRope = mDeployment.base.ropeConfig;
    if (baseRope.type == RopeType::kMRope && baseRope.yarn.has_value())
    {
        // Media rows go into the same MRoPE table as text rows, so they use PipelineIO's YaRN frequencies.
        for (MultimodalRunner* runner : {mVisionRunner.get(), mAudioRunner.get()})
        {
            if (runner != nullptr)
            {
                runner->setMropeYarnFrequencies(mPipelineIO->mropeInvFreq, baseRope.yarn->mscale);
            }
        }
    }

    initializeVisionDataParallel();

    if (contextCacheConfig.enabled)
    {
        ELLM_CHECK(mActionRunner == nullptr || mapping.tensorParallelSize == 1,
            "Context reuse cannot be enabled with tensor-parallel action runners.");
        ELLM_CHECK(!mSharedResources->cacheManagers.empty()
                && mSharedResources->cacheManagers.size() == mSharedResources->kvPageTables.size()
                && mSharedResources->cacheManagers.size() <= 2,
            "Context reuse requires one base cache and at most one draft cache.");
        HybridCacheManager* const draftCache
            = mSharedResources->cacheManagers.size() == 2 ? mSharedResources->cacheManagers[1].get() : nullptr;
        KVPageTable* const draftPageTable
            = mSharedResources->kvPageTables.size() == 2 ? mSharedResources->kvPageTables[1].get() : nullptr;
        KVPageTable* const swaPageTable = needsBoundedSwaPageManager ? mSharedResources->getSwaKVPageTable(0) : nullptr;
        ContextCachePhysicalResources cacheResources{*mSharedResources->cacheManagers[0],
            *mSharedResources->kvPageTables[0], draftCache, draftPageTable, swaPageTable};
        ELLM_CHECK(contextCacheDeploymentProfile.has_value(), "Context-cache deployment was not validated");
        mContextCache = std::make_unique<ContextCacheCoordinator>(
            contextCacheConfig, mDeployment, *contextCacheDeploymentProfile, cacheResources, stream);

        mHybridSpecContextReuseDeployment
            = contextCacheDeploymentProfile->isHybrid() && contextCacheDeploymentProfile->isSpeculative();
        if (mHybridSpecContextReuseDeployment)
        {
            mSpecFutureDependencyTokens = contextCacheDeploymentProfile->specReuseContract->futureDependencyTokens;
            if (mSpecFutureDependencyTokens > 0)
            {
                rt::Coords const bhShape = mPipelineIO->baseHiddenStates.getShape();
                mBoundaryFoldScratch = rt::Tensor({bhShape[1], bhShape[2]}, rt::DeviceType::kGPU,
                    mPipelineIO->baseHiddenStates.getDataType(), "LLMRankRuntime::mBoundaryFoldScratch");
                mBoundaryFoldMaxRows = math::cast<int32_t>(bhShape[1]);
            }
        }
    }
    if (mActionRunner)
    {
        auto const& basePageTable = *mSharedResources->kvPageTables[0];
        mActionKvBatchCollector
            = std::make_unique<ActionKvBatchCollector>(mMaxRuntimeBatchSize, basePageTable.maxPagesPerSeq(),
                basePageTable.numPages(), mContextCache ? &mContextCache->manager() : nullptr);
    }
    if (needsBoundedSwaPageManager)
    {
        ELLM_CHECK(mSharedResources->cacheManagers.size() == 1 && mSharedResources->kvPageTables.size() == 1,
            "Bounded SWA supports one vanilla base cache");
        KVPageTable* const swaPageTable = mSharedResources->getSwaKVPageTable(0);
        ELLM_CHECK(swaPageTable != nullptr, "Bounded SWA requires an independent sparse page table");
        mBoundedSwaKVPageManager = std::make_unique<BoundedSwaKVPageManager>(*mSharedResources->cacheManagers[0],
            *mSharedResources->kvPageTables[0], *swaPageTable, stream, BoundedSwaKVPageManager::StreamSynchronizer{},
            contextCacheConfig.enabled);
    }

    // -----------------------------------------------------------------------
    // 15. Shared execution context memory for all engines (base, optional
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
    mSharedExecContextMemory = rt::Tensor({sharedContextMemorySize}, rt::DeviceType::kGPU,
        trt_edgellm::DataType::kUINT8, "LLMRankRuntime::mSharedExecContextMemory");
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

    // Row-digest scratch: only a deployment that can compute image content digests needs it.
    if (mVisionRunner && (mContextCache || mEncoderEmbeddingCache))
    {
        mImageDigestRows = rt::Tensor(
            {limits::multimodal::kMaxImageDigestRows, static_cast<int64_t>(sizeof(kernel::ImageRowDigest))},
            rt::DeviceType::kCPU, DataType::kUINT8, "LLMRankRuntime::mImageDigestRows");
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
    PreprocessResources preprocessResources{*mStepPreparer, *mEmbeddingPre, mEmbedding, mIdsInput, mDeepstack.get(),
        mGemma4Ple.get(), mQwen4ExpNgram.get()};
    SamplingBuffers sampling{mSamplingWorkspace, mSamplingIndices, mSamplingScores, mBaseVocabMappingTable,
        mHostDecoderTokenIds, mHostSelectedTokenIds, mHostOutputSpaceIds, mSamplingRows, mHostSamplingRows};
    LogprobsBuffers logprobs{
        mDeviceLogprobsValues, mDeviceLogprobsIndices, mHostLogprobsValues, mHostLogprobsIndices, mGatheredLogits};
    mDecodingRuntimeContext.reset(
        new DecodingRuntimeContext{mDeployment, mMaxRuntimeBatchSize, mCheckpointDir, mDraftCheckpointDir,
            baseResources, preprocessResources, *mTokenizer, mLogitBias, mGuidedDecoder, sampling, logprobs});
    mDecodingRuntimeContext->benchmark = &mBenchmarkOptions;

    // Zero disables jump-forward on the decoders that read the bound (jumpForwardRequested). A catch-up row carries
    // the sampled token plus the jump: it must fit one context-chunk row and, for bounded SWA's one spare page, one KV
    // page.
    mDecodingRuntimeContext->chunkMaxJumpTokens
        = std::max(0, std::min(mDeployment.base.maxQueryLength - 1, kTOKENS_PER_PAGE - 1));
    mDecodingRuntimeContext->maxJumpTokens
        = mDeployment.specDecodeMode() == SpecDecodeMode::kNONE ? 0 : mDeployment.specConfig->maxJumpTokens;
}

bool LLMRankRuntime::jumpForwardRequested(GuidedDecodingParams const& params, DecodingStrategy const& strategy,
    bool nonGreedySampling, bool inFlightBatching) const noexcept
{
    if (!params.jumpForward)
    {
        return false;
    }
    struct Refusal
    {
        JumpForwardRefusal reason;
        char const* unsupported;
    };
    std::optional<Refusal> refusal;
    if (mDeployment.specDecodeMode() == SpecDecodeMode::kNONE || strategy.kind() == DecodingStrategyKind::kVanilla)
    {
        // The vanilla decoder catches up on a context chunk, on a speculative deployment's fallback as well.
        if (mDeployment.base.isDiffusionBackbone)
        {
            refusal = Refusal{JumpForwardRefusal::kDiffusionBackbone, "diffusion backbones"};
        }
        else if (mDeployment.base.useVisionBidirectionalAttention)
        {
            // The catch-up step is a text context chunk, which vision-block attention does not accept mid-sequence.
            refusal = Refusal{JumpForwardRefusal::kVisionBlockAttention, "vision-block attention"};
        }
        else if (mDecodingRuntimeContext == nullptr || mDecodingRuntimeContext->chunkMaxJumpTokens <= 0)
        {
            refusal
                = Refusal{JumpForwardRefusal::kSingleTokenChunk, "engines whose context chunk holds a single token"};
        }
    }
    else if (inFlightBatching)
    {
        refusal = Refusal{JumpForwardRefusal::kSpeculativeInFlightBatching,
            "speculative decoders under in-flight batching (the blocking request path serves it)"};
    }
    else if (nonGreedySampling && strategy.capabilities().supportsLosslessSampling)
    {
        refusal = Refusal{JumpForwardRefusal::kNonGreedySampling, "non-greedy sampling under speculative decoding"};
    }
    else if (mDecodingRuntimeContext == nullptr || mDecodingRuntimeContext->maxJumpTokens <= 0)
    {
        refusal = Refusal{JumpForwardRefusal::kNoVerifyHeadroom,
            "speculative engines without verify headroom (tree drafting needs max_spec_verify_tokens above the verify "
            "token count, chain drafting at least the chain width)"};
    }
    if (!refusal.has_value())
    {
        return true;
    }
    auto const logged = static_cast<size_t>(refusal->reason);
    if (!mJumpForwardRefusalsLogged.test(logged))
    {
        mJumpForwardRefusalsLogged.set(logged);
        LOG_WARNING("Jump-forward is not supported on %s; guided requests run without it.", refusal->unsupported);
    }
    return false;
}

void LLMRankRuntime::prepareJumpForwardExecution(DecodingStrategy const& strategy, cudaStream_t stream)
{
    mGuidedDecoder.prepareJumpForward();
    // Only the vanilla decoder catches up on a context chunk.
    if (mJumpForwardContextPrepared || strategy.kind() != DecodingStrategyKind::kVanilla)
    {
        return;
    }
    mJumpForwardContextPrepared = true;
    try
    {
        if (!mBaseExecutor->dedicateContextChunkContext(kPrefillProfile, stream))
        {
            LOG_WARNING("Jump-forward catch-up steps keep the shared execution context: no dedicated context");
        }
    }
    catch (std::exception const& error)
    {
        LOG_WARNING("Jump-forward catch-up steps keep the shared execution context: %s", error.what());
    }
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
    if (mDeployment.base.useVisionBidirectionalAttention)
    {
        LOG_WARNING("Visual-token pruning is not supported with vision-block attention; leaving it disabled.");
        return;
    }
    if (mDeployment.base.ropeConfig.type != RopeType::kMRope || mDeployment.base.imageTokenId < 0)
    {
        LOG_WARNING("Visual-token pruning requires an mRoPE VLM engine (image token id present); leaving it disabled.");
        return;
    }
    if (mActionRunner)
    {
        LOG_WARNING("Visual-token pruning is not supported together with an action runner; leaving it disabled.");
        return;
    }
    mVisualPruner = createVisualTokenPruner(config, mDeployment.base);
    if (mDeployment.specConfig.has_value())
    {
        // Spec-decode prefill compacts the auxiliary inputs (compactAuxiliaryInputs); size its
        // buffers now so no allocation ever happens on the prefill path.
        mVisualPruner->preallocateAuxiliaryBuffers();
    }
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

LLMRankRuntime::GenerationSession::GenerationSession(LLMRankRuntime& runtime, DecodingInferenceContext& context,
    DecodingStrategy& strategy, ManagedKVCacheRequest* managedRequest, DecodingKvHeadroom const& kvHeadroom,
    DecodingKvHeadroom const& jumpForwardKvHeadroom, cudaStream_t stream, bool synchronizeCancellation,
    std::vector<bool> actionSlots)
    : mRuntime(runtime)
    , mContext(context)
    , mStrategy(&strategy)
    , mManagedRequest(managedRequest)
    , mKvHeadroom(kvHeadroom)
    , mJumpForwardKvHeadroom(jumpForwardKvHeadroom)
    , mStream(stream)
    , mSynchronizeCancellation(synchronizeCancellation)
    , mActionSlots(std::move(actionSlots))
    // The same lookups the grammar gate uses (reasoningEndMarkers), so both agree on which token
    // ends the block.
    , mEndOfChannelId(reasoningEndMarkers(*runtime.mTokenizer)[0])
    , mEndOfThinkId(reasoningEndMarkers(*runtime.mTokenizer)[1])
    , mStartOfChannelId(reasoningStartMarkers(*runtime.mTokenizer)[0])
    , mStartOfThinkId(reasoningStartMarkers(*runtime.mTokenizer)[1])
    , mHasActionRequest(runtime.mActionRunner != nullptr
          && std::any_of(mActionSlots.begin(), mActionSlots.end(), [](bool action) { return action; }))
{
    // Alpamayo 1 marks the start of the trajectory block with a dedicated token.
    if (mRuntime.mActionRunner && mRuntime.mActionRunner->getModelType() == action::ActionModelType::ALPAMAYO1)
    {
        mTrajFutureStartId = static_cast<int32_t>(mRuntime.mTokenizer->getTokenId("<|traj_future_start|>"));
    }

    // Few-layer-validation / fixed-output perf: when EDGELLM_IGNORE_EOS is set, suppress
    // EOS-based termination so every run produces exactly maxGenerateLength tokens. This also
    // applies to multi-token accept paths such as DiffusionGemma canvas commit; otherwise EOS
    // inside a canvas can truncate a fixed-output block. A single request asks for the same
    // through its own ignoreEos, which lands on its slot (context.ignoreEos).
    char const* const ignoreEosValue = std::getenv("EDGELLM_IGNORE_EOS");
    mIgnoreEosForAllSlots
        = ignoreEosValue != nullptr && std::string(ignoreEosValue) != "0" && std::string(ignoreEosValue) != "false";
    if (mIgnoreEosForAllSlots)
    {
        LOG_INFO("EDGELLM_IGNORE_EOS set: ignoring EOS; running to maxGenerateLength.");
    }

    // Held by the context for the duration of decoding. It closes over this object, so the
    // destructor takes it back out again.
    mContext.shouldStopAfterAcceptedToken = [this](int32_t batchIdx, int32_t tokenId) {
        updateThinkingDoneForToken(batchIdx, tokenId);
        bool isEos = !ignoresEos(batchIdx) && mRuntime.mTokenizer->isEosToken(tokenId);
        if (isEos && mContext.enableThinking && tokenId != mRuntime.mTokenizer->getEosId()
            && !mContext.thinkingDone[batchIdx])
        {
            isEos = false;
        }
        return isEos || mContext.currentGenerateLengths[batchIdx] >= mContext.generateBudget(batchIdx);
    };
}

bool LLMRankRuntime::GenerationSession::ignoresEos(int32_t row) const noexcept
{
    return mIgnoreEosForAllSlots || mContext.ignoreEos[static_cast<size_t>(row)] != 0;
}

LLMRankRuntime::GenerationSession::~GenerationSession()
{
    mContext.shouldStopAfterAcceptedToken = {};
}

bool LLMRankRuntime::GenerationSession::isActionSlot(int32_t batchIdx) const noexcept
{
    if (!mHasActionRequest)
    {
        return false;
    }
    int32_t const originalIndex = mContext.batchIndexMapping[static_cast<size_t>(batchIdx)];
    return originalIndex >= 0 && static_cast<size_t>(originalIndex) < mActionSlots.size()
        && mActionSlots[static_cast<size_t>(originalIndex)];
}

LLMRankRuntime::SteppedGeneration::~SteppedGeneration() noexcept
{
    // Drain failures while request buffers and cache ownership are still alive. Successful requests already drained.
    if (!completed)
    {
        mFailure.drainFailedRequest(mStream);
    }
    mHostTokenStorage = std::move(context.raggedExecutionBatch.hostTokenIds);
    if (actionKvCollector != nullptr)
    {
        try
        {
            actionKvCollector->releaseMaterialized(context.stream);
        }
        catch (...)
        {
            LOG_ERROR("Action KV cleanup could not prove stream quiescence.");
            std::terminate();
        }
    }
}

bool LLMRankRuntime::GenerationSession::finished() const
{
    // Check if all batches have been evicted
    if (mContext.activeBatchSize == 0)
    {
        return true;
    }
    for (int32_t i = 0; i < mContext.activeBatchSize; ++i)
    {
        if (!mContext.finishedStates[i])
        {
            return false;
        }
    }
    return true;
}

void LLMRankRuntime::GenerationSession::updateThinkingDoneForToken(int32_t batchIdx, int32_t tokenId)
{
    if (!mContext.enableThinking || mContext.thinkingDone[batchIdx])
    {
        return;
    }
    if (tokenId == mEndOfChannelId || tokenId == mEndOfThinkId)
    {
        mContext.thinkingDone[batchIdx] = true;
    }
    else if (mContext.currentGenerateLengths[batchIdx] == 1 && tokenId != mStartOfChannelId
        && tokenId != mStartOfThinkId)
    {
        mContext.thinkingDone[batchIdx] = true;
        LOG_DEBUG("Batch %d: first token %d is not thinking-start, marking thinkingDone", batchIdx, tokenId);
    }
}

void LLMRankRuntime::GenerationSession::updateThinkingDoneForSlot(int32_t batchIdx)
{
    if ((!mContext.enableThinking && !mContext.hasGuidedDecoding) || mContext.tokenIds[batchIdx].empty())
    {
        return;
    }
    int32_t const tokenId = mContext.tokenIds[batchIdx].back();
    updateThinkingDoneForToken(batchIdx, tokenId);

    // Deliberately outside updateThinkingDoneForToken: that one returns early once
    // `thinkingDone` latches, and the end marker would then never be observed.
    if (mContext.hasGuidedDecoding && (tokenId == mEndOfChannelId || tokenId == mEndOfThinkId))
    {
        mContext.guidedReasoningEnded[batchIdx] = 1;
    }
}

void LLMRankRuntime::GenerationSession::updateThinkingDone()
{
    for (int32_t i = 0; i < mContext.activeBatchSize; ++i)
    {
        updateThinkingDoneForSlot(i);
    }
}

void LLMRankRuntime::GenerationSession::updateFinishState(int32_t batchIdx)
{
    int32_t const i = batchIdx;
    auto& s = mContext.slotStreams[i];
    s.profiler.recordGeneration(mContext.currentGenerateLengths[i]);
    // terminalReason is set for all slots; non-streaming slots surface it via
    // BatchResult.terminalReason -> response.finishReasons.
    if (!mContext.finishedStates[i]) // Respect first-writer-wins (cancel may have fired).
    {
        if (isActionSlot(i))
        {
            if (mContext.tokenIds[i].size() > 1 && mTrajFutureStartId >= 0
                && mContext.tokenIds[i][mContext.tokenIds[i].size() - 2] == mTrajFutureStartId)
            {
                mContext.finishedStates[i] = 1;
                s.terminalReason = FinishReason::kEndId;
                LOG_DEBUG("Batch %d finished, reason: traj_future_start", i);
            }
        }
        else if (!ignoresEos(i) && !mContext.tokenIds[i].empty())
        {
            // Check EOS (supports multiple EOS tokens, e.g. Gemma4 [1, 106]).
            // In thinking mode, suppress secondary EOS until thinking is complete.
            // ignoresEos() bypasses EOS entirely to force a fixed-length run.
            auto const lastToken = mContext.tokenIds[i].back();
            bool isEos = mRuntime.mTokenizer->isEosToken(lastToken);
            if (isEos && mContext.enableThinking && lastToken != mRuntime.mTokenizer->getEosId()
                && !mContext.thinkingDone[i])
            {
                isEos = false;
            }
            if (isEos)
            {
                mContext.finishedStates[i] = 1;
                s.terminalReason = FinishReason::kEndId;
                LOG_DEBUG("Batch %d finished, reason: EOS", i);
            }
        }
        // Check max length
        if (!mContext.finishedStates[i] && mContext.currentGenerateLengths[i] >= mContext.generateBudget(i))
        {
            mContext.finishedStates[i] = 1;
            s.terminalReason = FinishReason::kLength;
            LOG_DEBUG("Batch %d finished, total tokens=%d, reason: max_length", i, mContext.currentGenerateLengths[i]);
        }
    }
}

void LLMRankRuntime::GenerationSession::updateFinishStates()
{
    for (int32_t i = 0; i < mContext.activeBatchSize; ++i)
    {
        updateFinishState(i);
    }
}

void LLMRankRuntime::GenerationSession::applyStopMatch(int32_t batchIdx)
{
    auto& s = mContext.slotStreams[batchIdx];
    // After EOS/length so a stop string overrides kEndId/kLength (the user-relevant cause).
    // Cancel/error still win because decodeSlot skipped the match when those reasons were latched.
    if (s.stopMatchedThisIter && s.terminalReason != FinishReason::kCancelled
        && s.terminalReason != FinishReason::kError)
    {
        mContext.finishedStates[batchIdx] = 1;
        s.terminalReason = FinishReason::kStopWords;
        LOG_DEBUG("Batch %d finished, reason: stop_words", batchIdx);
    }
}

void LLMRankRuntime::GenerationSession::applyStopMatches()
{
    for (int32_t i = 0; i < mContext.activeBatchSize; ++i)
    {
        applyStopMatch(i);
    }
}

bool LLMRankRuntime::GenerationSession::performBatchEvictAndSnapshot()
{
    if (mHasActionRequest)
    {
        mRuntime.mActionKvBatchCollector->captureFinished(*mRuntime.mSharedResources->kvPageTables[0],
            mRuntime.mSharedResources->cacheManagers[0]->getKVCacheLengths(), mContext.finishedStates,
            mContext.batchIndexMapping, mContext.residentRefs, mContext.stream);
    }
    bool const status = mRuntime.performBatchEvict(mContext, *mStrategy, mManagedRequest);
    if (status && mHasActionRequest)
    {
        mRuntime.mActionKvBatchCollector->completeCapture();
    }
    return status;
}

void LLMRankRuntime::GenerationSession::rebindStrategy(DecodingStrategy& strategy)
{
    ELLM_CHECK(mContext.activeBatchSize == 0, "rebindStrategy: a strategy changes only between fills of the batch.");
    mStrategy = &strategy;
}

bool LLMRankRuntime::GenerationSession::primeFromPrefill()
{
    mRuntime.selectProfilingTimer();

    // DiffusionGemma prefill writes prompt KV only and does not produce a generated token.
    if (!mRuntime.mDeployment.base.isDiffusionBackbone)
    {
        applyCancellationToFinishStates(mContext);
        if (mSynchronizeCancellation && !mRuntime.synchronizeCancellationStates(mContext))
        {
            LOG_ERROR("Failed to synchronize cancellation state across parallel ranks after prefill.");
            return false;
        }
        updateThinkingDone();
        updateFinishStates();
        decodePerSlot(mContext, *mRuntime.mTokenizer);
        applyStopMatches();
        emitChunks(mContext, *mRuntime.mTokenizer);

        // Managed page rows can be compacted without moving physical KV. Keep the unmanaged lifecycle unchanged;
        // Qwen-style MTP cannot compact its recurrent draft state after partial prefill eviction.
        bool const supportsPartialPrefillEviction
            = (mManagedRequest != nullptr && mStrategy->kind() != DecodingStrategyKind::kMTP) || mHasActionRequest;
        if (mContext.activeBatchSize > 0 && (supportsPartialPrefillEviction || finished()))
        {
            bool const batchEvictStatus = performBatchEvictAndSnapshot();
            if (!batchEvictStatus)
            {
                LOG_ERROR("Failed to perform batch eviction.");
                return false;
            }
        }
    }
    return true;
}

bool LLMRankRuntime::GenerationSession::advance()
{
    mRuntime.selectProfilingTimer();
    gTimer.setRound(mContext.generationRound);
    TIME_STAGE("decode_round", mContext.stream);
    // Observe any consumer cancels at the top of the iteration so they land
    // first in the per-slot terminalReason latch.
    applyCancellationToFinishStates(mContext);
    if (mSynchronizeCancellation && !mRuntime.synchronizeCancellationStates(mContext))
    {
        LOG_ERROR("Failed to synchronize cancellation state across parallel ranks.");
        return false;
    }

    if (mManagedRequest != nullptr
        && !mManagedRequest->prepareDecodeStep(mContext, mKvHeadroom, mJumpForwardKvHeadroom))
    {
        return false;
    }

    if (!mStrategy->decodeStep(mContext))
    {
        LOG_ERROR("Failed to decode tokens with %s decoding strategy.", mStrategy->name());
        return false;
    }

    updateThinkingDone();
    // Terminal tokens must bypass the stream interval and flush pending UTF-8 bytes.
    updateFinishStates();
    decodePerSlot(mContext, *mRuntime.mTokenizer);
    applyStopMatches();
    if (mRuntime.mBenchmarkOptions.maxDecodeRounds > 0
        && mContext.generationRound + 1 >= mRuntime.mBenchmarkOptions.maxDecodeRounds)
    {
        for (int32_t i = 0; i < mContext.activeBatchSize; ++i)
        {
            if (!mContext.finishedStates[i])
            {
                mContext.finishedStates[i] = 1;
                mContext.slotStreams[i].terminalReason = FinishReason::kLength;
            }
        }
    }

    std::vector<int32_t> const& commonMaterializedStateLengths = mStrategy->commonMaterializedStateLengths();
    if (mManagedRequest != nullptr && !mManagedRequest->completeDecodeStep(mContext, commonMaterializedStateLengths))
    {
        return false;
    }
    emitChunks(mContext, *mRuntime.mTokenizer);

    emitTokenCallbacks(mContext);
    mContext.generationRound += 1;
    for (int32_t i = 0; i < mContext.activeBatchSize; ++i)
    {
        ++mContext.decodeRounds[static_cast<size_t>(i)];
    }

    // Perform batch eviction after all old-slot progress and terminal publication are complete.
    if (!performBatchEvictAndSnapshot())
    {
        LOG_ERROR("Failed to perform batch eviction.");
        return false;
    }
    return true;
}

AdmitDecision LLMRankRuntime::GenerationSession::reserveAdmission(AdmissionIntent& intent)
{
    SlotSeed& seed = intent.seed;
    // A speculative strategy's draft writes ahead of the committed length in its own KV cache too.
    std::vector<int32_t> const promptLength{static_cast<int32_t>(seed.promptTokenIds.size())};
    ELLM_CHECK(seed.maxGenerateLength > 0
            && clampMaxGenerateLengthForKVCapacity(promptLength, seed.maxGenerateLength,
                   mRuntime.mDeployment.base.maxKVCacheCapacity, mKvHeadroom.baseExtraTokens)
                == seed.maxGenerateLength
            && (mKvHeadroom.draftExtraTokens == 0
                || clampMaxGenerateLengthForKVCapacity(promptLength, seed.maxGenerateLength,
                       mRuntime.mDeployment.draft->maxKVCacheCapacity, mKvHeadroom.draftExtraTokens)
                    == seed.maxGenerateLength),
        "reserveAdmission: the sequence's generate budget must already fit the KV capacity behind its prompt.");

    // The batch both engines can run bounds the residents, whatever the resident pool holds.
    if (mContext.activeBatchSize >= mRuntime.maxBatchSize())
    {
        return AdmitDecision::kNoCapacity;
    }
    intent.resident = mContext.residentSlots.acquire();
    if (!intent.resident.has_value())
    {
        return AdmitDecision::kNoCapacity;
    }

    if (mManagedRequest != nullptr)
    {
        int32_t prefillStart = 0;
        ContextCacheRequest::AdmitSequenceStatus status;
        try
        {
            // Under a context cache the sequence leases its pages -- possibly sharing a published
            // prefix -- before it owns a slot, so a refused lease leaves the batch untouched. Media
            // travels into the admission so its per-position hash keys the pages: a media prefix
            // must not be reachable from a text lookup over the same token ids, and every sequence
            // keys media runs by the same deployment-derived placeholder list.
            status = mManagedRequest->admitSequence(seed.promptTokenIds, mContext.loraWeightsName, mKvHeadroom,
                prefillStart, *intent.resident, mContext.stream, intent.lookupPolicy, intent.commitPolicy,
                mRuntime.mediaTokenIds(), seed.imageContentDigests, seed.audioBuffers);
        }
        catch (...)
        {
            ELLM_CHECK(mContext.residentSlots.release(*intent.resident), "Reserved resident identity became stale");
            intent.resident.reset();
            throw;
        }
        switch (status)
        {
        case ContextCacheRequest::AdmitSequenceStatus::kAdmitted: seed.prefillStart = prefillStart; break;
        case ContextCacheRequest::AdmitSequenceStatus::kNoCapacity:
            ELLM_CHECK(mContext.residentSlots.release(*intent.resident), "Reserved resident identity became stale");
            intent.resident.reset();
            return AdmitDecision::kNoCapacity;
        case ContextCacheRequest::AdmitSequenceStatus::kOverlongPrefill:
            ELLM_CHECK(mContext.residentSlots.release(*intent.resident), "Reserved resident identity became stale");
            intent.resident.reset();
            seed.prefillStart = prefillStart;
            return AdmitDecision::kOverlongPrefill;
        case ContextCacheRequest::AdmitSequenceStatus::kFailed:
            ELLM_CHECK(mContext.residentSlots.release(*intent.resident), "Reserved resident identity became stale");
            intent.resident.reset();
            return AdmitDecision::kFailed;
        }
    }

    // The sequence's row in the batch of every cache manager the strategy drives (base, and draft for a speculative
    // strategy), with the KV length of its reused prefix, is the last reservation: from here each manager's batch
    // follows the context's, so a seat retired before its prefill is a row the eviction's compaction drops like any
    // other. A throw here is a device failure: the rows, the lease and the identity are given back and the caller
    // reports the session lost.
    size_t const cacheCount = mRuntime.activeCacheManagerCount(*mStrategy);
    for (size_t cacheIndex = 0; cacheIndex < cacheCount; ++cacheIndex)
    {
        ELLM_CHECK(
            mRuntime.mSharedResources->cacheManagers[cacheIndex]->getActiveBatchSize() == mContext.activeBatchSize,
            "reserveAdmission: a cache manager's batch disagrees with the context's.");
    }
    try
    {
        for (size_t cacheIndex = 0; cacheIndex < cacheCount; ++cacheIndex)
        {
            HybridCacheManager& cacheManager = *mRuntime.mSharedResources->cacheManagers[cacheIndex];
            cacheManager.appendSequence(seed.prefillStart, mContext.stream);
            if (mManagedRequest == nullptr)
            {
                // Without a context cache nothing restores this slot's recurrent and convolution
                // state before the sequence's first forward, and the kernels read whatever the slot
                // holds as the state before its first token. The eviction clears a retired slot, but
                // a session closed with residents (a failed one) never evicts them, so the admission
                // clears the slot it takes: a sequence starts from empty state whatever the slot's
                // previous life left there, as the blocking path's cold prefill does
                // (setUpForPrefillExecution). Under a context cache the coordinator owns this state
                // (it restores a snapshot or clears at eviction).
                cacheManager.clearResidentSlot(intent.resident->slot, mContext.stream);
            }
        }
    }
    catch (...)
    {
        releaseReservation(intent);
        throw;
    }
    return AdmitDecision::kAdmitted;
}

void LLMRankRuntime::GenerationSession::releaseReservation(AdmissionIntent& intent)
{
    ELLM_CHECK(intent.resident.has_value(), "releaseReservation requires a reserved resident identity.");
    // The manager rows, where they were taken, are the batch's last; dropping them is a size change.
    size_t const cacheCount = mRuntime.activeCacheManagerCount(*mStrategy);
    for (size_t cacheIndex = 0; cacheIndex < cacheCount; ++cacheIndex)
    {
        mRuntime.mSharedResources->cacheManagers[cacheIndex]->setActiveBatchSize(mContext.activeBatchSize);
    }
    bool const cacheRetracted = mManagedRequest == nullptr || mManagedRequest->retractSequenceAdmission();
    bool const residentReleased = mContext.residentSlots.release(*intent.resident);
    intent.resident.reset();
    if (!cacheRetracted || !residentReleased)
    {
        LOG_ERROR("Failed to fully release an admission reservation (cache=%d, resident=%d).",
            static_cast<int32_t>(cacheRetracted), static_cast<int32_t>(residentReleased));
    }
}

void LLMRankRuntime::GenerationSession::stageReservedMedia(AdmissionIntent const& intent)
{
    ELLM_CHECK(intent.resident.has_value(), "stageReservedMedia requires a reserved resident identity.");
    CompactMediaReservation* const reservation = mRuntime.mCompactMedia.get();
    ELLM_CHECK(reservation != nullptr || !intent.deferredMedia.has_value(),
        "stageReservedMedia: a deferred media intent needs the compact media reservation.");
    if (reservation == nullptr)
    {
        return;
    }
    int32_t const slot = intent.resident->slot;
    reservation->release(slot);
    if (intent.deferredMedia.has_value())
    {
        mRuntime.encodeDeferredAdmissionMedia(*intent.deferredMedia, intent.seed.imageContentDigests, slot,
            intent.seed.promptTokenIds, intent.seed.prefillStart, mStream);
    }
}

LLMRankRuntime::GenerationSession::PendingSeat LLMRankRuntime::GenerationSession::seatSlot(AdmissionIntent intent)
{
    ELLM_CHECK(intent.resident.has_value(), "seatSlot requires a reserved resident identity.");
    SlotSeed& seed = intent.seed;
    PendingSeat pending;
    pending.rawInputLength = static_cast<int32_t>(seed.promptTokenIds.size());
    pending.prefillStart = seed.prefillStart;
    // The seed is moved into the slot below; the seating still needs the media references.
    pending.mediaPayload.visualEmbeddings = seed.visualEmbeddings;
    pending.mediaPayload.audioEmbeddings = seed.audioEmbeddings;
    pending.mediaPayload.deepstackFeatures = seed.deepstackFeatures;
    pending.hasMedia = seed.visualEmbeddings.has_value() || seed.audioEmbeddings.has_value();
    pending.lookupPolicy = intent.lookupPolicy;
    pending.replayTailLength = intent.replayTailLength;
    int32_t row = -1;
    try
    {
        // The logprobs buffers grow here, after every check and reservation has passed, so a
        // refused admission never pays for a reallocation; a failure is a seat failure like any other.
        if (seed.numLogprobs > 0)
        {
            mRuntime.ensureLogprobsCapacity(
                mRuntime.mMaxRuntimeBatchSize * mRuntime.mDeployment.maxAcceptedTokensPerRound(), seed.numLogprobs);
        }
        row = mContext.appendSlot(std::move(seed), SequenceIdentity{intent.requestId, *intent.resident});
    }
    catch (...)
    {
        // The reservations exist but no slot ever did: give them back so the coordinator's
        // sequences and the cache manager's batch stay aligned with the context, then let the
        // scheduler's catch turn the throw into a rejection.
        releaseReservation(intent);
        throw;
    }

    mStrategy->appendRow(row);
    pending.row = row;
    return pending;
}

void LLMRankRuntime::GenerationSession::abortSeatedAdmission(int32_t row)
{
    ELLM_CHECK(mManagedRequest == nullptr || mManagedRequest->abortSequenceAdmission(row),
        "Failed to safely abort the seated sequence-admission transaction");
}

void LLMRankRuntime::GenerationSession::retireSeatUnprefilled(int32_t row)
{
    ELLM_CHECK(row >= 0 && row < mContext.activeBatchSize && mContext.finishedStates[static_cast<size_t>(row)] != 0,
        "retireSeatUnprefilled: the seat must be a live row already marked finished.");
    abortSeatedAdmission(row);
    emitChunk(mContext, row, *mRuntime.mTokenizer);
}

void LLMRankRuntime::GenerationSession::buildCompactSchedule(
    std::vector<CompactRow> const& rows, std::vector<int32_t>& mropeInitSlots, std::vector<int32_t>& decodeRows)
{
    mContext.scheduledStep.id = mContext.nextStepId++;
    mContext.scheduledStep.sequences.clear();
    for (CompactRow const& entry : rows)
    {
        size_t const row = static_cast<size_t>(entry.row);
        std::vector<int32_t> const& tokens = mContext.tokenIds[row];
        ELLM_CHECK(mContext.pendingTokens[row] <= 1,
            "A compact step feeds one lookahead token per decode row; a pending jump-forward run is caught up first.");
        int32_t const pastLength = mContext.committedLengths[row];
        bool const context = entry.work == SequenceWork::kContext;
        // The working history starts at the reused prefix, so a chunk's tokens sit at its
        // committed length minus that prefix; a decode row feeds its newest token.
        int32_t const begin
            = context ? pastLength - mContext.prefillStartLengths[row] : static_cast<int32_t>(tokens.size()) - 1;
        bool const sample = !context
            || pastLength + entry.queryLength == static_cast<int32_t>(mContext.rawBatchedInputIds[row].size());
        mContext.scheduledStep.sequences.push_back(
            ScheduledSequence{SequenceIdentity{mContext.requestIds[row], mContext.residentRefs[row]}, entry.work,
                entry.queryLength, pastLength,
                HostTokenRange{tokens.data(), static_cast<int32_t>(tokens.size()), begin, entry.queryLength}, sample});
        if (context && entry.firstChunk)
        {
            auto& slot = mContext.slotStreams[row];
            slot.profiler.beginPrefill(slot.channel && slot.channel->getCollectMetrics());
            // A media sequence's MRope row was written for its whole prompt at admission.
            int32_t const residentSlot = mContext.residentRefs[row].slot;
            if (mRuntime.mCompactMedia == nullptr || !mRuntime.mCompactMedia->holds(residentSlot))
            {
                mropeInitSlots.push_back(residentSlot);
            }
        }
        if (!context)
        {
            decodeRows.push_back(entry.row);
        }
    }
    mRuntime.mPipelineIO->waitForStepHostStaging();
    mContext.compactBatchBuilder->buildInto(mContext.scheduledStep, mContext.raggedExecutionBatch);
}

bool LLMRankRuntime::GenerationSession::settlePromptAdmissions(
    std::vector<CompactRow> const& rows, bool& publishedPrefix)
{
    if (mManagedRequest == nullptr)
    {
        return true;
    }
    for (CompactRow const& entry : rows)
    {
        if (entry.work != SequenceWork::kContext)
        {
            continue;
        }
        size_t const row = static_cast<size_t>(entry.row);
        int32_t const promptLength = static_cast<int32_t>(mContext.rawBatchedInputIds[row].size());
        if (!mManagedRequest->completeSequenceAdmissionDeviceWork(entry.row))
        {
            return false;
        }
        if (mContext.committedLengths[row] < promptLength)
        {
            if (!mManagedRequest->advanceSequenceAdmission(entry.row, mContext.committedLengths[row]))
            {
                return false;
            }
            continue;
        }
        // Cancellation wins over publication: a sequence cancelled while its final chunk ran
        // leaves through this step's eviction without publishing its prompt.
        applyCancellationToFinishState(mContext, entry.row);
        bool const resolved = mContext.finishedStates[row] != 0
            ? mManagedRequest->abortSequenceAdmission(entry.row)
            : mManagedRequest->finalizeSequenceAdmission(entry.row, mContext.tokenIds[row].back(), promptLength);
        if (!resolved)
        {
            return false;
        }
        publishedPrefix = publishedPrefix || mContext.finishedStates[row] == 0;
    }
    return true;
}

void LLMRankRuntime::GenerationSession::applyBenchmarkRoundLimit(int32_t row)
{
    if (mRuntime.mBenchmarkOptions.maxDecodeRounds > 0
        && mContext.generationRound + 1 >= mRuntime.mBenchmarkOptions.maxDecodeRounds
        && !mContext.finishedStates[static_cast<size_t>(row)])
    {
        mContext.finishedStates[static_cast<size_t>(row)] = 1;
        mContext.slotStreams[static_cast<size_t>(row)].terminalReason = FinishReason::kLength;
    }
}

bool LLMRankRuntime::GenerationSession::finishSampledRows(std::vector<CompactRow> const& rows)
{
    bool hasDecodeRow = false;
    for (int32_t const executionRow : mContext.raggedExecutionBatch.logitsToSequence)
    {
        CompactRow const& entry = rows[static_cast<size_t>(executionRow)];
        int32_t const row = entry.row;
        if (entry.work == SequenceWork::kContext)
        {
            mContext.slotStreams[static_cast<size_t>(row)].profiler.finishPrefill(
                mContext.currentGenerateLengths[static_cast<size_t>(row)]);
        }
        else
        {
            hasDecodeRow = true;
        }
        applyCancellationToFinishState(mContext, row);
        updateThinkingDoneForSlot(row);
        updateFinishState(row);
        decodeSlot(mContext, row, *mRuntime.mTokenizer);
        applyStopMatch(row);
        if (entry.work == SequenceWork::kDecode)
        {
            applyBenchmarkRoundLimit(row);
        }
    }
    return hasDecodeRow;
}

bool LLMRankRuntime::GenerationSession::executeCompactStep(
    std::vector<CompactRow> const& rows, bool& publishedPrefix, bool& compactRefused)
{
    mRuntime.selectProfilingTimer();
    TIME_STAGE("compact_step", mContext.stream);
    ELLM_CHECK(!rows.empty(), "executeCompactStep: a step names at least one row.");
    ELLM_CHECK(mContext.compactBatchBuilder.has_value(), "executeCompactStep: the compact builder is not configured.");
    publishedPrefix = false;
    compactRefused = false;

    PipelineIO& io = *mRuntime.mPipelineIO;
    std::vector<int32_t>& mropeInitSlots = mContext.compactMropeInitSlots;
    std::vector<int32_t>& decodeRows = mContext.compactDecodeRows;
    mropeInitSlots.clear();
    decodeRows.clear();
    RaggedExecutionBatch const& batch = mContext.raggedExecutionBatch;
    auto const isFinalChunk = [this](CompactRow const& entry) {
        size_t const row = static_cast<size_t>(entry.row);
        return entry.work == SequenceWork::kContext
            && mContext.committedLengths[row] + entry.queryLength
            == static_cast<int32_t>(mContext.rawBatchedInputIds[row].size());
    };

    {
        NVTX_SCOPED_RANGE(nvtx_compact_build, "COMPACT_BUILD", nvtx_colors::BLUE);
        buildCompactSchedule(rows, mropeInitSlots, decodeRows);
        // Every compact step pairs a decode-subset prepare with its completion: only the decode rows
        // grow pages, and every other row -- a prompt chunk, a resident the plan leaves out -- holds.
        // Each prompt chunk opens its admission chunk; the first one restores the sequence's snapshot.
        if (mManagedRequest != nullptr)
        {
            if (!mManagedRequest->prepareDecodeStep(mContext, mKvHeadroom, decodeRows))
            {
                return false;
            }
            for (CompactRow const& entry : rows)
            {
                if (entry.work == SequenceWork::kContext && !mManagedRequest->prepareSequenceAdmission(entry.row))
                {
                    return false;
                }
            }
        }
    }
    StepId const stepId = mContext.scheduledStep.id;

    HybridCacheManager& cacheManager = *mRuntime.mSharedResources->cacheManagers[0];
    {
        CacheBatchRestore const restore{cacheManager, mContext.activeBatchSize};
        CompactForwardStatus forward = CompactForwardStatus::kOk;
        {
            NVTX_SCOPED_RANGE(nvtx_compact_enqueue, "COMPACT_ENQUEUE", nvtx_colors::BLUE);
            forward = mRuntime.runCompactForward(mContext, mropeInitSlots);
        }
        if (forward == CompactForwardStatus::kEnqueueRefused)
        {
            // The layers before the refusing one already advanced recurrent and convolution state in place, so the
            // step cannot be replayed; only the steps that follow can avoid phase 8.
            CompactCapability refused;
            refused.reason
                = "a plugin refused compact execution at runtime (see the plugin's ERROR log, which "
                  "identifies the layer by kind and shape)";
            LOG_ERROR(
                "Compact execution disabled for this runtime: %s. The step's residents fail; later sessions "
                "serve single-cohort steps.",
                refused.reason.c_str());
            mRuntime.mCompactCapability = std::move(refused);
            compactRefused = true;
            return false;
        }
        if (forward != CompactForwardStatus::kOk)
        {
            LOG_ERROR("Failed to execute the base model for a compact step.");
            return false;
        }
        if (mManagedRequest != nullptr)
        {
            for (CompactRow const& entry : rows)
            {
                if (isFinalChunk(entry) && !mManagedRequest->enqueueSequenceAdmissionCapture(entry.row))
                {
                    return false;
                }
            }
        }
        mContext.completionSequences.clear();
        for (CompactRow const& entry : rows)
        {
            size_t const row = static_cast<size_t>(entry.row);
            mContext.completionSequences.push_back(CompletionSequence{
                mContext.requestIds[row], mContext.residentRefs[row], mContext.committedLengths[row]});
        }
        mContext.raggedExecutionBatch.validateCommitSnapshot(stepId, mContext.completionSequences);
        cacheManager.commitSequenceLength(io.contextLengths, mContext.stream);
    }
    for (CompactRow const& entry : rows)
    {
        size_t const row = static_cast<size_t>(entry.row);
        mContext.committedLengths[row] += entry.queryLength;
        if (entry.work == SequenceWork::kContext)
        {
            mContext.effectivePrefillLengths[row] += entry.queryLength;
        }
    }

    // Logits row r belongs to execution row logitsToSequence[r]; the placeholder rows after them have no
    // owner and are never sampled, so a step of non-final chunks only advances its sequences.
    std::vector<int32_t>& sampledRows = mContext.compactSampledRows;
    sampledRows.clear();
    for (int32_t const executionRow : batch.logitsToSequence)
    {
        sampledRows.push_back(rows[static_cast<size_t>(executionRow)].row);
    }
    {
        NVTX_SCOPED_RANGE(nvtx_compact_sample, "COMPACT_SAMPLE", nvtx_colors::BLUE);
        check::check(sampledRows.empty()
                || io.outputLogits.reshape(
                    {static_cast<int64_t>(sampledRows.size()), mRuntime.mDeployment.base.outputVocabSize}),
            "Compact sampled-logits reshape failed");
        if (!sampledRows.empty()
            && !decoder_utils::enqueueSamplingForRows(
                *mRuntime.mDecodingRuntimeContext, mContext, io.outputLogits, sampledRows, mContext.stream))
        {
            return false;
        }
        CUDA_CHECK(cudaStreamSynchronize(mContext.stream));
    }

    NVTX_SCOPED_RANGE(nvtx_compact_commit, "COMPACT_COMMIT", nvtx_colors::BLUE);
    decoder_utils::commitSampledRows(*mRuntime.mDecodingRuntimeContext, mContext, sampledRows);
    if (!settlePromptAdmissions(rows, publishedPrefix))
    {
        return false;
    }
    bool const hasDecodeRow = finishSampledRows(rows);
    // After the finish decisions, so a decode row that finished here publishes its history, and
    // before the chunks go out, so a consumer that sees its sequence finish finds that history
    // published (the order advance() keeps).
    if (mManagedRequest != nullptr
        && !mManagedRequest->completeDecodeStep(mContext, mStrategy->commonMaterializedStateLengths()))
    {
        return false;
    }
    for (int32_t const executionRow : batch.logitsToSequence)
    {
        int32_t const row = rows[static_cast<size_t>(executionRow)].row;
        emitChunk(mContext, row, *mRuntime.mTokenizer);
        emitTokenCallbackForSlot(mContext, row);
    }
    if (hasDecodeRow)
    {
        mContext.generationRound += 1;
    }
    for (CompactRow const& compactRow : rows)
    {
        if (compactRow.work == SequenceWork::kDecode)
        {
            ++mContext.decodeRounds[static_cast<size_t>(compactRow.row)];
        }
    }

    if (!performBatchEvictAndSnapshot())
    {
        LOG_ERROR("Failed to perform batch eviction.");
        return false;
    }
    return true;
}

AdmitDecision LLMRankRuntime::GenerationSession::prefillSeated(PendingSeat pending)
{
    mRuntime.selectProfilingTimer();
    int32_t const row = pending.row;
    ELLM_CHECK(row >= 0 && row < mContext.activeBatchSize, "prefillSeated: the pending seat is not a live row.");

    bool admissionClosed = false;
    auto failAdmission = [&]() -> AdmitDecision {
        admissionClosed = true;
        mContext.finishedStates[static_cast<size_t>(row)] = 1;
        mContext.slotStreams[static_cast<size_t>(row)].terminalReason = FinishReason::kError;
        abortSeatedAdmission(row);
        return AdmitDecision::kFailed;
    };

    try
    {
        if (mManagedRequest != nullptr && !mManagedRequest->prepareSequenceAdmission(row))
        {
            return failAdmission();
        }
        if (!mRuntime.prefillSlotInPlace(mContext, *mStrategy, row, pending.prefillStart,
                pending.hasMedia ? &pending.mediaPayload : nullptr, mManagedRequest, pending.lookupPolicy,
                pending.replayTailLength))
        {
            return failAdmission();
        }
        // A speculative draft holds the prompt but the position the next decode step takes (seatRow).
        std::vector<int32_t> const& commonLengths = mStrategy->commonMaterializedStateLengths();
        std::optional<int32_t> const commonLength = static_cast<size_t>(row) < commonLengths.size()
            ? std::optional<int32_t>{commonLengths[static_cast<size_t>(row)]}
            : std::nullopt;
        if (mManagedRequest != nullptr
            && !mManagedRequest->finalizeSequenceAdmission(
                row, mContext.tokenIds[static_cast<size_t>(row)].back(), pending.rawInputLength, commonLength))
        {
            return failAdmission();
        }
        admissionClosed = true;
    }
    catch (...)
    {
        std::exception_ptr const error = std::current_exception();
        if (!admissionClosed)
        {
            try
            {
                static_cast<void>(failAdmission());
            }
            catch (std::exception const& abortError)
            {
                LOG_ERROR("Failed to abort a throwing seated admission: %s", abortError.what());
            }
            catch (...)
            {
                LOG_ERROR("Failed to abort a throwing seated admission with an unknown error.");
            }
        }
        std::rethrow_exception(error);
    }

    // The seated prefill's lookahead token takes the per-token pipeline a decode step runs over
    // the batch, for this one row and in the same order: cancellation, the thinking latch and
    // the reasoning gate, the finish decision, detokenization and stop matching, the stop-string
    // override, the chunk to the row's channel and the token callback. One path for every
    // produced token: a sequence whose first token is already terminal (EOS, or a budget of one)
    // is finished -- and its streaming consumer told so -- right here, instead of being evicted
    // with a channel nobody ever finishes; and a sequence that goes on has its first token on the
    // wire before its first decode step, as a request on the blocking path does after
    // primeFromPrefill.
    applyCancellationToFinishState(mContext, row);
    updateThinkingDoneForSlot(row);
    updateFinishState(row);
    decodeSlot(mContext, row, *mRuntime.mTokenizer);
    applyStopMatch(row);
    emitChunk(mContext, row, *mRuntime.mTokenizer);
    emitTokenCallbackForSlot(mContext, row);
    return AdmitDecision::kAdmitted;
}

LLMRankRuntime::GenerationSession::AdmissionIntent LLMRankRuntime::GenerationSession::buildAdmissionIntent(
    LLMGenerationRequest const& request, int32_t originalIndex, RequestId requestId, bool deferMediaEncoders,
    DecodingStrategy const& strategy)
{
    mRuntime.selectProfilingTimer();
    ELLM_CHECK(request.requests.size() == 1, "admission takes one sequence at a time.");
    auto const& item = request.requests.front();
    ELLM_CHECK(!item.pastTrajectory.has_value(),
        "admission cannot take an action request; trajectory execution runs on the blocking path.");

    // The request's cache policies ride into its lease as the sequence's own. The lookup policy
    // is decided below, once the media preprocess has computed the image identities: an
    // image without one would key on the media token alone, every image's placeholder, so such a
    // request bypasses the lookup the way the blocking path does (ContextCacheRequest::lookupPolicyFor).
    auto const lookupPolicy = [&](SlotSeed const& seed) {
        return ContextCacheRequest::lookupPolicyFor(
            request, /*outputThinkerEmbeddings=*/false, mRuntime.mediaTokenIds(), {seed.imageContentDigests});
    };

    SlotSeed seed;
    seed.originalIndex = originalIndex;
    seed.stopStrings = item.stopStrings;
    // The same full-to-reduced vocabulary translation the blocking path applies in prepareLogitBias:
    // the sampler indexes the output vocabulary, and a raw full-vocabulary id would bias the wrong
    // token on a reduced-vocabulary engine.
    seed.logitBias = normalizeLogitBias(mRuntime.mLogitBias, item.logitBias, originalIndex);
    seed.skipSpecialTokens = request.skipSpecialTokens;
    // The sequence's own decoding settings; the budget is settled by the caller once the prompt
    // length is known, and the logprobs buffers are sized at the seat (seatSlot).
    seed.sampling = rowSamplingParamsFor(request);
    seed.samplingSeed = resolveSamplingSeed(request, item);
    seed.requestStableSampling
        = usesRequestStableSampling(request, item, strategy.capabilities().supportsLosslessSampling);
    seed.numLogprobs = mRuntime.clampLogprobsWidth(request.numLogprobs);
    seed.ignoreEos = request.ignoreEos;
    seed.jumpForward = item.guidedDecoding.has_value()
        && mRuntime.jumpForwardRequested(*item.guidedDecoding, strategy,
            shouldUseNonGreedySampling(request.temperature, request.topK, request.topP), /*inFlightBatching=*/true);
    if (seed.jumpForward)
    {
        mRuntime.prepareJumpForwardExecution(strategy, mStream);
    }
    if (!request.streamChannels.empty())
    {
        seed.channel = request.streamChannels.front();
    }

    std::optional<LLMGenerationRequest> deferredMedia;
    bool const hasMedia = !item.imageBuffers.empty() || !item.audioBuffers.empty();
    if (hasMedia)
    {
        ELLM_CHECK((item.imageBuffers.empty() || mRuntime.mVisionRunner != nullptr)
                && (item.audioBuffers.empty() || mRuntime.mAudioRunner != nullptr),
            "admission: the request carries media this deployment has no encoder for.");
        // Visual-token pruning gates on an empty base KV, so a media sequence seated beside other
        // residents would silently skip it and generate a different answer than the same request
        // run by itself. Under a pruner, media runs alone: the seed says so, and the session admits
        // it only into an empty session (RuntimeSession::requiresExclusiveAdmission is how the
        // scheduler learns this before it decides to admit). Media whose token layout only its encoder
        // determines cannot be deferred, so in a compact session it runs alone, single-cohort, the same way.
        seed.requiresExclusiveSession
            = mRuntime.mVisualPruner != nullptr || (deferMediaEncoders && mRuntime.mediaLayoutNeedsEncoder(request));
        seed.audioBuffers = item.audioBuffers;

        // A single-sequence view of the request for the request-level preprocess. The chat
        // formatting happens here, same as the text path below, so the media-token expansion
        // inside the preprocess sees the exact prompt the blocking path would have seen.
        LLMGenerationRequest single;
        single.requests = {item};
        single.applyChatTemplate = request.applyChatTemplate;
        single.addGenerationPrompt = request.addGenerationPrompt;
        single.enableThinking = request.enableThinking;
        LLMGenerationRequest::FormattedRequest formatted;
        ELLM_CHECK(mRuntime.mChatTemplate->apply(item, formatted, chat_template::ChatTemplate::optionsFrom(request)),
            "admission: chat template failed.");
        single.formattedRequests = {formatted};
        seed.systemPrompt = formatted.formattedSystemPrompt;

        if (deferMediaEncoders && mRuntime.mCompactMedia != nullptr && !seed.requiresExclusiveSession)
        {
            ELLM_CHECK(mRuntime.preprocessDeferredAdmissionMedia(single, seed, mStream),
                "admission: multimodal preprocessing failed.");
            deferredMedia = std::move(single);
        }
        else
        {
            ELLM_CHECK(mRuntime.preprocessAdmissionMedia(single, seed, mStream),
                "admission: multimodal preprocessing failed.");
        }
    }
    // The same tokenization a blocking request goes through, so an admitted prompt is not a
    // second dialect of the same input.
    else if (!request.preTokenizedInputIds.empty() && !request.preTokenizedInputIds.front().empty())
    {
        seed.promptTokenIds = request.preTokenizedInputIds.front();
    }
    else
    {
        LLMGenerationRequest::FormattedRequest formatted;
        ELLM_CHECK(mRuntime.mChatTemplate->apply(item, formatted, chat_template::ChatTemplate::optionsFrom(request)),
            "admission: chat template failed.");
        seed.promptTokenIds = mRuntime.mTokenizer->encode(formatted.formattedCompleteRequest, false);
        seed.systemPrompt = formatted.formattedSystemPrompt;
    }
    ELLM_CHECK(!seed.promptTokenIds.empty(), "admission: the prompt tokenized to nothing.");
    // Under a context cache the lease keys every media run by its content, so a prompt whose
    // placeholders name more media than the request supplies cannot be leased. Checked here, as a
    // refusal that modifies nothing, rather than inside the reservation, where the same check is
    // a session loss that takes every resident down with this one request.
    if (mManagedRequest != nullptr)
    {
        MediaPlaceholderCounts const placeholders
            = countMediaPlaceholders(seed.promptTokenIds, mRuntime.mediaTokenIds());
        if (placeholders.images > seed.imageContentDigests.size() || placeholders.audios > seed.audioBuffers.size())
        {
            throw std::invalid_argument("admission: the prompt carries " + std::to_string(placeholders.images)
                + " image and " + std::to_string(placeholders.audios) + " audio placeholder run(s), but the request "
                + "supplies " + std::to_string(seed.imageContentDigests.size()) + " image(s) and "
                + std::to_string(seed.audioBuffers.size()) + " audio clip(s).");
        }
    }

    // The request's policy with everything that forces bypass applied, narrowed by the sequence's
    // own when it has one -- the two levels the blocking path's ContextCacheRequest::begin resolves.
    ContextCacheLookupPolicy const policy
        = ContextCacheRequest::sequenceLookupPolicy(lookupPolicy(seed), item.contextCacheLookupPolicy);
    // The tail a Hybrid+spec checkpoint stops before, derived here when the request asks for it, as the request
    // coordinator derives it for a blocking request. A media prompt stops at its last media run instead.
    int32_t replayTailLength = std::max(0, request.contextCacheReplayTailLength);
    if (request.contextCacheReplayTailLength < 0 && mManagedRequest != nullptr
        && mRuntime.mHybridSpecContextReuseDeployment && !hasMedia && policy != ContextCacheLookupPolicy::kBypass)
    {
        replayTailLength = deriveContextCacheReplayTailLength(
            *mRuntime.mChatTemplate, *mRuntime.mTokenizer, request, seed.promptTokenIds);
    }
    return AdmissionIntent{std::move(seed), requestId, std::nullopt, policy, request.contextCacheCommitPolicy,
        replayTailLength, std::move(deferredMedia)};
}

LLMGenerationResponse LLMRankRuntime::GenerationSession::materializeResult(
    BatchResult const& result, std::vector<std::string> const& stopStrings) const
{
    LLMGenerationResponse response;
    auto const totalLength = static_cast<int32_t>(result.tokenIds.size());
    int32_t const generateLength = std::min(result.generateLength, totalLength);

    response.outputIds.emplace_back(result.tokenIds.end() - generateLength, result.tokenIds.end());
    response.outputTexts.push_back(mRuntime.mTokenizer->decode(response.outputIds.front(), result.skipSpecialTokens));
    response.logprobs.push_back(result.logprobs);
    response.finishReasons.push_back(result.terminalReason);
    response.inputTokenCounts.push_back(static_cast<int32_t>(result.rawBatchedInputIds.size()));
    response.reusedInputTokenCounts.push_back(reusedPromptTokens(static_cast<int32_t>(result.rawBatchedInputIds.size()),
        result.effectivePrefillLength, result.prunedPrefillTokens));
    if (mStrategy->isSpeculative())
    {
        response.specVerifyCounts.push_back(result.actualIterations);
        response.specAcceptanceLengths.push_back(result.actualIterations > 0
                ? static_cast<float>(result.generateLength) / static_cast<float>(result.actualIterations)
                : 0.0F);
    }

    // The same final stop-string trim the blocking path's assembly applies: one-shot decode can
    // surface a stop that the incremental streaming matcher straddled across piece boundaries.
    if (!stopStrings.empty())
    {
        size_t maxLen = 0;
        for (auto const& stop : stopStrings)
        {
            maxLen = std::max(maxLen, stop.size());
        }
        auto outcome = applyStopStringMatch(response.outputTexts.front(), stopStrings, maxLen, /*isFinal=*/true);
        response.outputTexts.front() = std::move(outcome.emitted);
        if (outcome.stopMatched)
        {
            response.finishReasons.front() = FinishReason::kStopWords;
        }
    }
    return response;
}

std::unordered_map<int32_t, BatchResult> LLMRankRuntime::GenerationSession::takeCompletedAtOrAbove(int32_t firstIndex)
{
    std::unordered_map<int32_t, BatchResult> taken;
    for (auto it = mContext.completedBatches.begin(); it != mContext.completedBatches.end();)
    {
        if (it->first >= firstIndex)
        {
            taken.emplace(it->first, std::move(it->second));
            it = mContext.completedBatches.erase(it);
        }
        else
        {
            ++it;
        }
    }
    return taken;
}

bool LLMRankRuntime::preprocessAdmissionMedia(LLMGenerationRequest const& request, SlotSeed& seed, cudaStream_t stream)
{
    // The full request-level preprocess, pointed at throwaway state: multiModalRuntimePreprocess
    // writes only request-scoped context fields (system prompts, raw ids, embedding references),
    // so a batch-1 temporary context absorbs them all without touching the resident batch. The
    // encoders themselves write their own output buffers, which is exactly what the seed hands on
    // to the seated prefill -- nothing else runs them before it consumes the references.
    OptionalOutputTensor const mropeOverride = admissionMropeStage();

    DecodingInferenceContext tempContext;
    tempContext.initialize(1, 1, std::nullopt, {}, std::string{}, stream);
    if (!multiModalRuntimePreprocess(request, tempContext, stream, mropeOverride))
    {
        return false;
    }
    ELLM_CHECK(tempContext.rawBatchedInputIds.size() == 1 && !tempContext.rawBatchedInputIds.front().empty(),
        "Admission media preprocess produced no tokens");

    seed.promptTokenIds = std::move(tempContext.rawBatchedInputIds.front());
    seed.systemPrompt = std::move(tempContext.systemPrompts.front());
    seed.visualEmbeddings = tempContext.visualEmbeddings;
    seed.audioEmbeddings = tempContext.audioEmbeddings;
    seed.deepstackFeatures = tempContext.deepstackFeatures;
    // The sequence keys its media positions on these identities; an image without one makes the
    // sequence bypass the cache lookup (ContextCacheRequest::lookupPolicyFor), as it would have at
    // request start.
    if (!tempContext.imageContentDigests.empty())
    {
        seed.imageContentDigests = std::move(tempContext.imageContentDigests.front());
    }
    return true;
}

OptionalOutputTensor LLMRankRuntime::admissionMropeStage()
{
    if (mDeployment.base.ropeConfig.type != RopeType::kMRope)
    {
        return std::nullopt;
    }
    if (mAdmissionMropeStage.isEmpty())
    {
        mAdmissionMropeStage = rt::Tensor({1, mDeployment.base.maxKVCacheCapacity, mDeployment.base.rotaryDim},
            rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "LLMRankRuntime::admissionMropeStage");
    }
    return OptionalOutputTensor{std::ref(mAdmissionMropeStage)};
}

bool LLMRankRuntime::mediaLayoutNeedsEncoder(LLMGenerationRequest const& request) const noexcept
{
    return mVisionRunner != nullptr && mVisionRunner->tokenLayoutNeedsEncoder(request);
}

bool LLMRankRuntime::preprocessDeferredAdmissionMedia(
    LLMGenerationRequest const& request, SlotSeed& seed, cudaStream_t stream)
{
    ELLM_CHECK(request.requests.size() == 1 && request.formattedRequests.size() == 1,
        "A deferred media admission preprocesses one formatted sequence.");
    auto const& item = request.requests.front();
    bool const hasVision = !item.imageBuffers.empty();
    bool const hasAudio = !item.audioBuffers.empty();
    NVTX_SCOPED_RANGE(nvtx_multimodal, "MULTIMODAL_ADMISSION_PREPROCESS", nvtx_colors::ORANGE);
    // Several admissions run back to back at one step boundary. The encoders rewrite pinned host staging buffers
    // that uploads of the previous admission's encoder run, still queued on the stream, read.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // The digests key the encoder embedding cache after the reservation and the context-cache lease in it.
    if (hasVision && (mEncoderEmbeddingCache != nullptr || mContextCache != nullptr))
    {
        std::vector<std::vector<Hash128>> digests = buildImageContentDigests(request, stream);
        if (!digests.empty())
        {
            seed.imageContentDigests = std::move(digests.front());
        }
    }

    OptionalOutputTensor const mropeOut = admissionMropeStage();
    std::vector<std::vector<int32_t>> batchedInputIds;
    if (hasAudio && !mAudioRunner->preprocess(request, batchedInputIds, mTokenizer, mropeOut, stream))
    {
        LOG_ERROR("Audio preprocessing failed. This request cannot be handled.");
        return false;
    }
    if (hasVision
        && !mVisionRunner->preprocess(
            request, batchedInputIds, mTokenizer, mropeOut, stream, /*imageOnly=*/false, /*skipEncoderWork=*/true))
    {
        LOG_ERROR("Vision text preprocessing failed. This request cannot be handled.");
        return false;
    }
    ELLM_CHECK(batchedInputIds.size() == 1 && !batchedInputIds.front().empty(),
        "Admission media preprocess produced no tokens");
    seed.promptTokenIds = std::move(batchedInputIds.front());
    return true;
}

void LLMRankRuntime::encodeDeferredAdmissionMedia(LLMGenerationRequest const& request,
    std::vector<Hash128> const& imageDigests, int32_t slot, std::vector<int32_t> const& promptTokenIds,
    int32_t prefillStart, cudaStream_t stream)
{
    ELLM_CHECK(mCompactMedia != nullptr, "A deferred media admission needs the compact media reservation.");
    auto const& item = request.requests.front();
    bool const hasVision = !item.imageBuffers.empty();
    bool const hasAudio = !item.audioBuffers.empty();
    NVTX_SCOPED_RANGE(nvtx_multimodal, "MULTIMODAL_ADMISSION_ENCODE", nvtx_colors::ORANGE);
    CompactMediaReservation::Sources sources;

    if (hasAudio)
    {
        std::vector<Hash128> audioHashes;
        if (mEncoderEmbeddingCache != nullptr)
        {
            for (auto const& audio : item.audioBuffers)
            {
                if (!audio.pcm || audio.pcm->numSamples() <= 0)
                {
                    LOG_WARNING(
                        "A clip of this request carries no PCM samples to identify it by; the encoder embedding cache "
                        "skips the request.");
                    audioHashes.clear();
                    break;
                }
                rt::Tensor const& samples = *audio.pcm->samples;
                auto const* rawPtr = reinterpret_cast<char const*>(samples.dataPointer<float>());
                size_t const rawBytes = static_cast<size_t>(audio.pcm->numSamples()) * sizeof(float);
                audioHashes.push_back(hashOpaqueIdentity(std::string_view(rawPtr, rawBytes), stream, false));
            }
        }
        std::vector<std::reference_wrapper<rt::Tensor const>> cachedAudio;
        for (Hash128 const& hash : audioHashes)
        {
            auto const entry = mEncoderEmbeddingCache->lookup(hash);
            if (!entry)
            {
                cachedAudio.clear();
                break;
            }
            cachedAudio.push_back(std::cref(entry->get()));
        }
        rt::Tensor& output = mAudioRunner->getOutputEmbedding();
        if (!cachedAudio.empty())
        {
            LOG_INFO("Encoder embedding cache HIT for all %zu audio clips — skipping audio encoder execution",
                cachedAudio.size());
            int64_t byteOffset = 0;
            for (auto const& entry : cachedAudio)
            {
                int64_t const bytes = entry.get().getMemoryCapacity();
                CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(output.rawPointer()) + byteOffset,
                    entry.get().rawPointer(), bytes, cudaMemcpyDeviceToDevice, stream));
                byteOffset += bytes;
            }
        }
        else
        {
            ELLM_CHECK(mAudioRunner->infer(stream), "Audio inference failed. This request cannot be handled.");
            auto const& tokenLengths = mAudioRunner->getLastMediaTokenLengths();
            if (!audioHashes.empty() && tokenLengths.size() == audioHashes.size())
            {
                int64_t const hiddenSize = output.getShape()[1];
                size_t const typeSize = rt::utils::getTypeSize(output.getDataType());
                int64_t byteOffset = 0;
                for (size_t index = 0; index < audioHashes.size(); ++index)
                {
                    int64_t const numTokens = tokenLengths[index];
                    mEncoderEmbeddingCache->storeSlice(audioHashes[index],
                        static_cast<char const*>(output.rawPointer()) + byteOffset, numTokens, hiddenSize,
                        output.getDataType(), stream);
                    byteOffset += numTokens * hiddenSize * static_cast<int64_t>(typeSize);
                }
            }
        }
        sources.audio = &output;
    }

    if (hasVision)
    {
        rt::Tensor& output = mVisionRunner->getOutputEmbedding();
        auto const features = mVisionRunner->getDeepstackFeatures();
        bool restored = mEncoderEmbeddingCache != nullptr && !imageDigests.empty()
            && std::all_of(imageDigests.begin(), imageDigests.end(),
                [this](Hash128 const& digest) { return mEncoderEmbeddingCache->lookupEntry(digest).has_value(); });
        if (restored)
        {
            restored = mEncoderEmbeddingCache->tryRestore(
                imageDigests, mVisionRunner->getLastMediaTokenLengths(), output, features, stream);
        }
        if (restored)
        {
            LOG_INFO(
                "Encoder embedding cache HIT for all %zu images — skipping ViT encoder execution", imageDigests.size());
        }
        else
        {
            // The intent stage prepared the text and MRope; only the pixel work and the encoder remain.
            std::vector<std::vector<int32_t>> unusedInputIds;
            ELLM_CHECK(mVisionRunner->preprocess(
                           request, unusedInputIds, mTokenizer, std::nullopt, stream, /*imageOnly=*/true),
                "Vision preprocessing failed. This request cannot be handled.");
            ELLM_CHECK(mVisionRunner->infer(stream), "Vision inference failed. This request cannot be handled.");
            auto const& tokenLengths = mVisionRunner->getLastMediaTokenLengths();
            if (mEncoderEmbeddingCache != nullptr && !imageDigests.empty()
                && tokenLengths.size() == imageDigests.size())
            {
                int64_t const hiddenSize = output.getShape()[1];
                size_t const typeSize = rt::utils::getTypeSize(output.getDataType());
                int64_t tokenOffset = 0;
                for (size_t index = 0; index < imageDigests.size(); ++index)
                {
                    int64_t const numTokens = tokenLengths[index];
                    mEncoderEmbeddingCache->storeSlice(imageDigests[index],
                        static_cast<char const*>(output.rawPointer())
                            + tokenOffset * hiddenSize * static_cast<int64_t>(typeSize),
                        numTokens, hiddenSize, output.getDataType(), stream, features, tokenOffset);
                    tokenOffset += numTokens;
                }
            }
        }
        sources.image = &output;
        sources.deepstack.assign(features.begin(), features.end());
    }

    LLMEngineConfig const& base = mDeployment.base;
    mCompactMedia->stage(slot, promptTokenIds, prefillStart, base.imageTokenId, base.audioTokenId, sources, stream);
    if (base.ropeConfig.type == RopeType::kMRope)
    {
        // Written once for the whole prompt and its generation; no later step rewrites the row.
        ELLM_CHECK(!mAdmissionMropeStage.isEmpty(), "The admission MRope stage is missing.");
        copyMropeRowToResidentSlot(mAdmissionMropeStage.rawPointer(), slot, stream);
    }
}

void LLMRankRuntime::copyMropeRowToResidentSlot(void const* source, int32_t slot, cudaStream_t stream)
{
    LLMEngineConfig const& base = mDeployment.base;
    ELLM_CHECK(slot >= 0 && slot < base.recurrentPoolRows, "MRoPE resident row is out of range.");
    check::check(mPipelineIO->mropeCosSin.reshape({base.recurrentPoolRows, base.maxKVCacheCapacity, base.rotaryDim}),
        "Resident MRoPE cache reshape failed");
    size_t const rowBytes
        = static_cast<size_t>(base.maxKVCacheCapacity) * static_cast<size_t>(base.rotaryDim) * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(
        static_cast<char*>(mPipelineIO->mropeCosSin.rawPointer()) + static_cast<size_t>(slot) * rowBytes, source,
        rowBytes, cudaMemcpyDeviceToDevice, stream));
}

void LLMRankRuntime::prepareCompactMediaReservation()
{
    if (mVisionRunner == nullptr && mAudioRunner == nullptr)
    {
        return;
    }
    LLMEngineConfig const& base = mDeployment.base;
    if (mCompactMedia != nullptr)
    {
        for (int32_t slot = 0; slot < mCompactMedia->shape().slots; ++slot)
        {
            mCompactMedia->release(slot);
        }
        return;
    }
    // A slot never embeds more media rows than one context step computes (promptLength - prefillStart <= S_in),
    // nor more image rows than the vision encoder produces for one request.
    CompactMediaReservation::Shape shape;
    shape.slots = base.recurrentPoolRows;
    shape.hiddenSize = base.hiddenSize;
    if (mVisionRunner != nullptr)
    {
        int64_t const rowBytes = static_cast<int64_t>(base.hiddenSize) * static_cast<int64_t>(sizeof(half));
        int64_t const encoderRows = mVisionRunner->getOutputEmbedding().getMemoryCapacity() / rowBytes;
        shape.imageRowsPerSlot
            = static_cast<int32_t>(std::min<int64_t>(base.maxSupportedInputLength, std::max<int64_t>(encoderRows, 1)));
        shape.deepstackLayers = mDeepstack != nullptr ? base.numDeepstackFeatures : 0;
    }
    if (mAudioRunner != nullptr)
    {
        // Audio runners may reallocate their output per request, so only the input length bounds the rows.
        shape.audioRowsPerSlot = base.maxSupportedInputLength;
    }
    std::string const layout
        = format::fmtstr("%d slots x (%d image rows x (1 + %d deepstack layers) + %d audio rows) x %d hidden FP16",
            shape.slots, shape.imageRowsPerSlot, shape.deepstackLayers, shape.audioRowsPerSlot, shape.hiddenSize);
    int64_t const rows = static_cast<int64_t>(shape.slots)
        * (static_cast<int64_t>(shape.imageRowsPerSlot) * (1 + shape.deepstackLayers) + shape.audioRowsPerSlot);
    double const mebibytes
        = static_cast<double>(rows * shape.hiddenSize * static_cast<int64_t>(sizeof(half))) / (1024.0 * 1024.0);
    try
    {
        mCompactMedia = std::make_unique<CompactMediaReservation>(shape);
    }
    catch (std::exception const& error)
    {
        throw std::runtime_error(
            format::fmtstr("the compact multimodal reservation (%s = %.2f MiB) could not be "
                           "allocated: %s",
                layout.c_str(), mebibytes, error.what()));
    }
    LOG_INFO("Compact multimodal reservation: %s = %.2f MiB.", layout.c_str(), mebibytes);
}

bool LLMRankRuntime::prepareCompactExecution(cudaStream_t stream)
{
    std::string failure;
    std::exception_ptr thrown;
    try
    {
        prepareCompactMediaReservation();
        // Compact steps with prompt rows run on profile 0 between profile-1 decode steps; a context of their own
        // spares each such step TensorRT's profile-switch cost. Without one the original context serves them.
        constexpr int64_t kWithContextSequences = 1;
        mBaseExecutor->dedicateCompactContext(compactStepProfileIndex(kWithContextSequences), stream);
        return true;
    }
    catch (std::exception const& error)
    {
        failure = error.what();
        thrown = std::current_exception();
    }
    // A failed allocation leaves the device usable; a stream in error does not.
    if (streamInError(stream))
    {
        std::rethrow_exception(thrown);
    }
    // The failed call's error must not be reported by the next kernel launch's error check.
    static_cast<void>(cudaGetLastError());
    mCompactMedia.reset();
    CompactCapability unavailable;
    unavailable.reason = "the resources of compact execution could not be allocated: " + failure;
    LOG_WARNING("Compact execution unavailable: %s; serving single-cohort steps.", unavailable.reason.c_str());
    mCompactCapability = std::move(unavailable);
    return false;
}

bool LLMRankRuntime::uploadCompactMediaIndices(RaggedExecutionBatch const& batch, cudaStream_t stream)
{
    int32_t const tokens = batch.shape.physicalTokens;
    int64_t const indexBytes = static_cast<int64_t>(tokens) * static_cast<int64_t>(sizeof(int32_t));
    ELLM_CHECK(mHostMultimodalIndices.getMemoryCapacity() >= indexBytes
            && mMultimodalIndices.getMemoryCapacity() >= indexBytes,
        "Compact media indices exceed the multimodal index buffers.");
    check::check(mHostMultimodalIndices.reshape({1, tokens}), "Tensor reshape failed");
    check::check(mMultimodalIndices.reshape({1, tokens}), "Tensor reshape failed");
    int32_t* const hostIndices = mHostMultimodalIndices.dataPointer<int32_t>();
    bool anyMedia = false;
    for (int32_t sequence = 0; sequence < batch.shape.numSequences; ++sequence)
    {
        size_t const index = static_cast<size_t>(sequence);
        int32_t* const rows = hostIndices + batch.queryStartOffsets[index];
        int32_t const length = batch.queryLengths[index];
        if (batch.sequenceWorks[index] == SequenceWork::kContext)
        {
            anyMedia = mCompactMedia->mapSlice(batch.stateIndices[index], batch.pastLengths[index], length, rows)
                || anyMedia;
        }
        else
        {
            std::fill(rows, rows + length, -1);
        }
    }
    if (!anyMedia)
    {
        return false;
    }
    CUDA_CHECK(cudaMemcpyAsync(mMultimodalIndices.rawPointer(), hostIndices,
        static_cast<size_t>(tokens) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    mPipelineIO->recordStepHostUploads(stream);
    return true;
}

bool LLMRankRuntime::supportsSeatedAdmission() const noexcept
{
    if (mDeployment.base.isDiffusionBackbone)
    {
        return false;
    }
    // Bounded SWA keeps per-slot window state (and its own sparse page tables) that the seating
    // swaps do not cover, and its request backend has no live-slot admission.
    if (mDeployment.base.usesBoundedSwaKVCache())
    {
        return false;
    }
    if (mContextCache != nullptr && !mContextCache->supportsLiveSequenceAdmission())
    {
        return false;
    }
    return true;
}

size_t LLMRankRuntime::activeCacheManagerCount(DecodingStrategy const& strategy) const noexcept
{
    return strategy.isSpeculative() ? mSharedResources->cacheManagers.size() : 1U;
}

int32_t LLMRankRuntime::maxBatchSize() const noexcept
{
    // The base and draft engines' common ceiling (DeploymentConfig::maxRuntimeBatchSize): a speculative batch runs
    // both, and every runtime buffer is sized to it.
    return mMaxRuntimeBatchSize;
}

bool LLMRankRuntime::prefillSlotInPlace(DecodingInferenceContext& context, DecodingStrategy& strategy, int32_t row,
    int32_t prefillStart, SlotSeed const* mediaPayload, ManagedKVCacheRequest* managedKVCacheRequest,
    ContextCacheLookupPolicy lookupPolicy, int32_t replayTailLength)
{
    ELLM_CHECK(row >= 0 && row < context.activeBatchSize, "prefillSlotInPlace: row must be a live execution row.");
    ELLM_CHECK(mContextCache != nullptr || prefillStart == 0,
        "prefillSlotInPlace: a reused prefix requires the context cache that leased its pages.");
    ELLM_CHECK(!mDeployment.base.isDiffusionBackbone,
        "prefillSlotInPlace: the diffusion prefill contract differs and has not been carried over.");
    // The GuidedDecoder's matchers are numbered by execution row and the row swap below does not
    // move them, so a constrained sequence runs only as the batch's sole row -- where the swap is
    // the identity and its matcher stays on row zero.
    ELLM_CHECK(!context.hasGuidedDecoding || context.activeBatchSize == 1,
        "prefillSlotInPlace: a grammar-constrained batch cannot take an admission beside its resident.");

    int32_t const fullBatch = context.activeBatchSize;
    ELLM_CHECK(context.committedLengths[static_cast<size_t>(row)] == prefillStart,
        "The admitted sequence's committed length does not match its reserved prefix.");
    // Every cache manager the strategy drives swaps the sequence's row to row zero: a draft cache's lengths persist
    // across steps, so every other row's must be kept.
    auto const& cacheManagers = mSharedResources->cacheManagers;
    size_t const cacheCount = activeCacheManagerCount(strategy);
    for (size_t cacheIndex = 0; cacheIndex < cacheCount; ++cacheIndex)
    {
        ELLM_CHECK(cacheManagers[cacheIndex]->getActiveBatchSize() == fullBatch,
            "prefillSlotInPlace: a cache manager's batch disagrees with the context's.");
    }

    OptionalInputTensor savedVisual = context.visualEmbeddings;
    OptionalInputTensor savedAudio = context.audioEmbeddings;
    OptionalInputTensors savedDeepstack = context.deepstackFeatures;
    OptionalInputTensor projectedVisual = std::nullopt;
    OptionalInputTensor projectedAudio = std::nullopt;
    OptionalInputTensors projectedDeepstack;
    if (mediaPayload != nullptr)
    {
        projectedVisual = mediaPayload->visualEmbeddings;
        projectedAudio = mediaPayload->audioEmbeddings;
        projectedDeepstack = mediaPayload->deepstackFeatures;
    }

    bool const copiesAdmissionMrope = mDeployment.base.ropeConfig.type == RopeType::kMRope && mediaPayload != nullptr;
    if (copiesAdmissionMrope)
    {
        ELLM_CHECK(!mAdmissionMropeStage.isEmpty(), "Admission MRope staging is missing.");
        auto const shape = mAdmissionMropeStage.getShape();
        ELLM_CHECK(shape.getNumDims() == 3 && shape[0] == 1 && shape[1] == mDeployment.base.maxKVCacheCapacity
                && shape[2] == mDeployment.base.rotaryDim,
            "Admission MRope staging does not match the active-row layout.");
        check::check(mPipelineIO->mropeActiveCosSin.reshape(shape), "Active MRope scratch reshape failed.");
    }

    context.swapExecutionRows(0, row);
    size_t swappedManagers = 0;
    struct ProjectionGuard
    {
        DecodingInferenceContext& context;
        std::vector<std::unique_ptr<HybridCacheManager>> const& cacheManagers;
        size_t cacheCount;
        size_t const& swappedManagers;
        int32_t row;
        int32_t fullBatch;
        OptionalInputTensor savedVisual;
        OptionalInputTensor savedAudio;
        OptionalInputTensors savedDeepstack;
        ~ProjectionGuard() noexcept
        {
            context.activeBatchSize = fullBatch;
            for (size_t cacheIndex = 0; cacheIndex < cacheCount; ++cacheIndex)
            {
                cacheManagers[cacheIndex]->restoreActiveBatchSize(fullBatch);
                if (cacheIndex < swappedManagers)
                {
                    // Self-inverse, and the batch is whole again. A device that cannot take the swap back has
                    // lost every row's execution length, which nothing past this point could detect.
                    try
                    {
                        cacheManagers[cacheIndex]->swapExecutionRows(0, row, context.stream);
                    }
                    catch (...)
                    {
                        std::terminate();
                    }
                }
            }
            context.restoreExecutionRows(0, row);
            context.visualEmbeddings = std::move(savedVisual);
            context.audioEmbeddings = std::move(savedAudio);
            context.deepstackFeatures = std::move(savedDeepstack);
        }
    } projection{context, cacheManagers, cacheCount, swappedManagers, row, fullBatch, std::move(savedVisual),
        std::move(savedAudio), std::move(savedDeepstack)};
    for (size_t cacheIndex = 0; cacheIndex < cacheCount; ++cacheIndex)
    {
        cacheManagers[cacheIndex]->swapExecutionRows(0, row, context.stream);
        ++swappedManagers;
    }
    // The base prefill narrows the base manager to its one row itself (materializeExecutionLengths); a draft
    // manager is narrowed here, for the strategy's draft prefill.
    for (size_t cacheIndex = 1; cacheIndex < cacheCount; ++cacheIndex)
    {
        cacheManagers[cacheIndex]->setActiveBatchSize(1);
    }

    context.visualEmbeddings = std::move(projectedVisual);
    context.audioEmbeddings = std::move(projectedAudio);
    context.deepstackFeatures = std::move(projectedDeepstack);
    context.activeBatchSize = 1;
    context.effectivePrefillLengths[0] = static_cast<int32_t>(context.tokenIds[0].size());

    if (copiesAdmissionMrope)
    {
        CUDA_CHECK(cudaMemcpyAsync(mPipelineIO->mropeActiveCosSin.rawPointer(), mAdmissionMropeStage.rawPointer(),
            mAdmissionMropeStage.getMemoryCapacity(), cudaMemcpyDeviceToDevice, context.stream));
    }

    // A Hybrid+spec sequence that looks up or publishes its prompt runs the blocking path's endpoint-reuse prefill as
    // a batch of one: the boundary fold on a hit, the volatile-tail split, and the checkpoint publication once the
    // draft has seated the prompt.
    bool const hybridSpecEndpointReuse = managedKVCacheRequest != nullptr && managedKVCacheRequest->hasContextReuse()
        && shouldUseHybridSpecEndpointReuse(strategy.kind(), mHybridSpecContextReuseDeployment,
            lookupPolicy != ContextCacheLookupPolicy::kBypass, lookupPolicy != ContextCacheLookupPolicy::kBypass);
    if (!hybridSpecEndpointReuse)
    {
        if (!runBaseModelPrefill(context, managedKVCacheRequest, /*sampleOutput=*/true, /*admissionSequenceIndex=*/row))
        {
            return false;
        }
        return strategy.seatRow(context, row);
    }
    struct EndpointReuseScope
    {
        DecodingInferenceContext& context;
        bool savedEndpointReuse;
        int32_t savedReplayTailLength;
        ~EndpointReuseScope()
        {
            context.hybridSpecEndpointReuse = savedEndpointReuse;
            context.contextCacheReplayTailLength = savedReplayTailLength;
        }
    };
    bool prefilled = false;
    int32_t const pastLength = context.committedLengths[0];
    {
        EndpointReuseScope const scope{context, context.hybridSpecEndpointReuse, context.contextCacheReplayTailLength};
        context.hybridSpecEndpointReuse = true;
        // A media prompt publishes at the end of its last media run, as on the blocking path.
        context.contextCacheReplayTailLength = mediaPayload != nullptr
                && (mediaPayload->visualEmbeddings.has_value() || mediaPayload->audioEmbeddings.has_value())
            ? mediaReplayTailLength(context.rawBatchedInputIds[0])
            : replayTailLength;
        prefilled = runHybridSpecPrefill(context, strategy, *managedKVCacheRequest, row);
    }
    if (prefilled)
    {
        // A volatile tail splits the prompt into two base forwards; the step record still describes the one context
        // step the caller scheduled, from the sequence's reused prefix over the whole tail.
        ELLM_CHECK(context.scheduledStep.sequences.size() == 1, "A seated prefill records one sequence.");
        ScheduledSequence& record = context.scheduledStep.sequences.front();
        std::vector<int32_t> const& tokens = context.tokenIds[0];
        record.pastLength = pastLength;
        record.queryLength = context.committedLengths[0] - pastLength;
        record.queryTokens = HostTokenRange{tokens.data(), static_cast<int32_t>(tokens.size()), 0, record.queryLength};
    }
    // The prefill synchronized after its last base forward, which covers the checkpoint restore the admission's
    // chunk opened with.
    return prefilled && managedKVCacheRequest->completeSequenceAdmissionDeviceWork(row);
}

std::string LLMRankRuntime::compactPreconditionFailure(DecodingInferenceContext const& context) const
{
    LLMEngineConfig const& base = mDeployment.base;
    if (mDeployment.draft.has_value() || base.isDiffusionBackbone)
    {
        return "speculative and diffusion deployments run single-cohort steps";
    }
#if IS_TRT_RTX
    return "TensorRT-RTX engines may run on a GPU other than the one they were built for";
#endif
    if (base.maxSupportedBatchSize < 2)
    {
        return "compact steps pair prompt rows with decode rows, which needs maxBatchSize >= 2; the engine's "
               "maxBatchSize is "
            + std::to_string(base.maxSupportedBatchSize);
    }
    if (base.useVisionBidirectionalAttention)
    {
        return "image tokens attend bidirectionally within an image (vision-block attention), which compact steps "
               "cannot split across chunks";
    }
    // Compact steps embed media rows from the slot reservations, which hold FP16 decoder-width rows.
    auto const decoderRows = [&base](MultimodalRunner& runner) {
        Tensor const& output = runner.getOutputEmbedding();
        return output.getDataType() == trt_edgellm::DataType::kHALF && output.getShape().getNumDims() == 2
            && output.getShape()[1] == base.hiddenSize;
    };
    if (mVisionRunner != nullptr && !decoderRows(*mVisionRunner))
    {
        return "the vision encoder's output rows are not FP16 rows of the decoder's hidden size";
    }
    if (mAudioRunner != nullptr && !decoderRows(*mAudioRunner))
    {
        return "the audio encoder's output rows are not FP16 rows of the decoder's hidden size";
    }
    // Admission staging copies one deepstack feature per decoder deepstack input; a mispaired bundle would refuse
    // every image request at staging.
    if (mVisionRunner != nullptr)
    {
        size_t const encoderLayers = mVisionRunner->getDeepstackFeatures().size();
        if (encoderLayers != static_cast<size_t>(base.numDeepstackFeatures))
        {
            return "the vision encoder produces " + std::to_string(encoderLayers)
                + " deepstack feature layers but the decoder engine takes " + std::to_string(base.numDeepstackFeatures);
        }
    }
    if (mQwen4ExpNgram != nullptr)
    {
        return "compact steps do not drive the n-gram embedding preprocessor of this model, whose row map assumes "
               "entry-padded token rows";
    }
    if (mGemma4Ple != nullptr)
    {
        // One compact step gathers the PLE rows of up to T_max = B x S_in tokens, which may span several prompts.
        int64_t const stepRows = std::min(base.plePool.rowCount, static_cast<int64_t>(base.maxPhysicalTokens));
        int64_t const capacity = mGemma4Ple->distinctRowCapacity();
        if (capacity < stepRows)
        {
            return "the per-layer embedding cache maps " + std::to_string(capacity)
                + " distinct rows per step (its cache slots less the zero slot), fewer than the "
                + std::to_string(stepRows)
                + " one compact step can need (the smaller of the PLE table's rows and the engine's B x S_in tokens); "
                  "compact steps need at least "
                + std::to_string(stepRows + 1) + " PLE cache slots or resident PLE storage";
        }
    }
    if (base.usesBoundedSwaKVCache())
    {
        return "compact steps do not support bounded sliding-window KV storage";
    }
    if (context.layerDebugger != nullptr)
    {
        return "the layer debugger dumps single-cohort steps only";
    }
    // Steps with prompt rows run on the prefill profile and decode-only subsets on the decode profile
    // (compactStepProfileIndex).
    for (auto const& [profile, name] : {std::pair{kPrefillProfile, "prefill"}, std::pair{kDecodeProfile, "decode"}})
    {
        nvinfer1::Dims maxPhase{};
        try
        {
            maxPhase = mBaseExecutor->getProfileShape(
                binding_names::kExecutionPhaseMarker, profile, nvinfer1::OptProfileSelector::kMAX);
        }
        catch (std::exception const&)
        {
            maxPhase.nbDims = 0;
        }
        if (maxPhase.nbDims != 1 || maxPhase.d[0] < static_cast<int64_t>(ExecutionPhase::kMixedPrefillDecode))
        {
            return std::string("the engine's ") + name + " profile does not admit compact execution";
        }
    }
    return {};
}

LLMRankRuntime::CompactCapability LLMRankRuntime::resolveCompactCapability(
    DecodingInferenceContext const& context) const
{
    CompactCapability capability;
    capability.reason = compactPreconditionFailure(context);
    if (!capability.reason.empty())
    {
        LOG_INFO("Compact execution unavailable: %s.", capability.reason.c_str());
        return capability;
    }
    int32_t const sm = getSMVersion();
    capability.reason = compactExecutionGap(mDeployment.base, sm);
    // The plugins size the compact workspace at build time; only the builder's report says that they all reserved it.
    if (capability.reason.empty() && !mDeployment.base.compactExecution)
    {
        capability.reason
            = "the engine was not built for compact execution (builder_config.compact_execution in its "
              "config.json is not true); the builder writes that field when every compact-capable layer reserved "
              "its compact workspace at build, and its log names the layers that did not";
    }
    capability.compact = capability.reason.empty();
    if (capability.compact)
    {
        LOG_INFO("Compact execution supported on SM%d.", sm);
    }
    else
    {
        LOG_WARNING("Compact execution unavailable: %s; serving single-cohort steps.", capability.reason.c_str());
    }
    return capability;
}

LLMRankRuntime::CompactForwardStatus LLMRankRuntime::runCompactForward(
    DecodingInferenceContext& context, std::vector<int32_t> const& mropeInitSlots)
{
    LLMEngineConfig const& base = mDeployment.base;
    RaggedExecutionBatch const& batch = context.raggedExecutionBatch;
    RaggedStepShape const& shape = batch.shape;
    int32_t const tokens = shape.physicalTokens;
    int32_t const sequences = shape.numSequences;

    check::check(mIdsInput.reshape({1, tokens}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mIdsInput.rawPointer(), batch.hostTokenIds.rawPointer(),
        static_cast<size_t>(tokens) * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    int32_t* const hostQueryLengths = mPipelineIO->hostContextLengths.dataPointer<int32_t>();
    check::check(mPipelineIO->hostContextLengths.reshape({sequences}), "Tensor reshape failed");
    std::copy(batch.queryLengths.begin(), batch.queryLengths.end(), hostQueryLengths);
    mPipelineIO->recordStepHostUploads(context.stream);

    check::check(mPipelineIO->inputsEmbeds.reshape({1, tokens, base.hiddenSize}), "Tensor reshape failed");
    bool const mediaStep = mCompactMedia != nullptr && uploadCompactMediaIndices(batch, context.stream);
    if (mediaStep)
    {
        Tensor const& image = mCompactMedia->imageRows();
        Tensor const& audio = mCompactMedia->audioRows();
        mEmbeddingPre->embed(mIdsInput, image.isEmpty() ? OptionalInputTensor{} : OptionalInputTensor{std::cref(image)},
            audio.isEmpty() ? OptionalInputTensor{} : OptionalInputTensor{std::cref(audio)}, *mPipelineIO,
            context.stream, OptionalInputTensor{std::cref(mMultimodalIndices)});
    }
    else
    {
        mEmbeddingPre->embed(mIdsInput, std::nullopt, std::nullopt, *mPipelineIO, context.stream);
    }
    if (mDeepstack != nullptr)
    {
        // The token-major deepstack inputs carry the media rows' features and zeros elsewhere (decode and text
        // rows); the shared zero target covers only decode-step shapes.
        if (mediaStep)
        {
            mEmbeddingPre->prepareDeepstack(mIdsInput, mCompactMedia->deepstackRows(), *mPipelineIO, context.stream);
        }
        else
        {
            mEmbeddingPre->prepareDeepstack(mIdsInput, OptionalInputTensors{}, *mPipelineIO, context.stream);
        }
    }
    if (mGemma4Ple != nullptr)
    {
        mGemma4Ple->embedPacked(mIdsInput, context.stream, &batch.hostTokenIds);
    }

    if (base.ropeConfig.type == RopeType::kMRope && !mropeInitSlots.empty())
    {
        // Text-only rows are identical, so one row is built and copied into each new resident row.
        prepareTextOnlyMRope(*mPipelineIO, base, 1, context.stream);
        for (int32_t const slot : mropeInitSlots)
        {
            copyMropeRowToResidentSlot(mPipelineIO->mropeActiveCosSin.rawPointer(), slot, context.stream);
        }
    }
    prepareRaggedExecutionBindings(*mPipelineIO, *mSharedResources, base, batch, /*kvCacheIndex=*/0, context.stream);

    check::check(mPipelineIO->inputsEmbeds.reshape({tokens, base.hiddenSize}), "Compact embedding reshape failed");
    check::check(
        mPipelineIO->outputHiddenStates.isEmpty() || mPipelineIO->outputHiddenStates.reshape({tokens, base.hiddenSize}),
        "Compact hidden-state output reshape failed");
    check::check(
        mPipelineIO->outputLogits.reshape({shape.numLogits, base.outputVocabSize}), "Compact logits reshape failed");
    if (mDeepstack != nullptr)
    {
        for (Tensor& deepstack : mPipelineIO->deepstackEmbeds)
        {
            check::check(deepstack.reshape({tokens, base.hiddenSize}), "Compact deepstack reshape failed");
        }
        mDeepstack->useRealFeatures(mBaseTensorMap);
    }
    mStepPreparer->prepare(
        InferencePhase::kPrefill, sequences, *mSharedResources->cacheManagers[0], *mPipelineIO, context.stream);
    mPipelineIO->recordStepHostUploads(context.stream);

    int32_t const profile = compactStepProfileIndex(shape.numContextSequences);
    InferenceDims const dims = base.compactDims(sequences, tokens, shape.numContextSequences, shape.numLogits);
    timer::TimerSession const forward
        = gTimer.startForward("base", dims.batch, dims.seqLen, dims.selectLen, context.stream);
    if (!mBaseExecutor->prepare(profile, dims, mBaseTensorMap, context.stream))
    {
        return CompactForwardStatus::kPrepareFailed;
    }
    if (mBaseExecutor->execute(context.stream))
    {
        return CompactForwardStatus::kOk;
    }
    // A plugin refusing phase 8 returns before it launches any work.
    return enqueueFailedOnDevice(context.stream) ? CompactForwardStatus::kDeviceFailed
                                                 : CompactForwardStatus::kEnqueueRefused;
}

bool LLMRankRuntime::runGenerationLoop(DecodingInferenceContext& context, DecodingStrategy& decodingStrategy,
    ManagedKVCacheRequest* managedRequest, LLMGenerationRequest const& request, DecodingKvHeadroom const& kvHeadroom,
    DecodingKvHeadroom const& jumpForwardKvHeadroom, cudaStream_t stream)
{
    // The consensus must run whenever a channel can be cancelled: the request streams.
    bool const synchronizeCancellation = !request.streamChannels.empty();
    std::vector<bool> actionSlots;
    if (mActionRunner != nullptr)
    {
        actionSlots.reserve(request.requests.size());
        for (auto const& item : request.requests)
        {
            actionSlots.push_back(item.pastTrajectory.has_value());
        }
    }
    GenerationSession session(*this, context, decodingStrategy, managedRequest, kvHeadroom, jumpForwardKvHeadroom,
        stream, synchronizeCancellation, std::move(actionSlots));
    if (!session.primeFromPrefill())
    {
        return false;
    }
    while (!session.finished())
    {
        if (!session.advance())
        {
            return false;
        }
    }
    return true;
}

bool LLMRankRuntime::handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response,
    cudaStream_t stream, bool outputThinkerEmbeddings, TokenBroadcastFn tokenBroadcast, VisionGatherFn visionGather,
    int32_t parallelRank)
{
    std::unique_ptr<SteppedGeneration> generation = beginGeneration(request, response, stream, outputThinkerEmbeddings,
        std::move(tokenBroadcast), std::move(visionGather), parallelRank);
    if (generation == nullptr)
    {
        return false;
    }
    if (!runGenerationLoop(generation->context, *generation->strategy, generation->managedRequest(), request,
            generation->kvHeadroom, generation->jumpForwardKvHeadroom, stream))
    {
        return false;
    }
    bool const succeeded = finishGeneration(*generation, request, response, stream);
    generation->completed = succeeded;
    return succeeded;
}

std::unique_ptr<LLMRankRuntime::SteppedGeneration> LLMRankRuntime::beginGeneration(LLMGenerationRequest const& request,
    LLMGenerationResponse& response, cudaStream_t stream, bool outputThinkerEmbeddings, TokenBroadcastFn tokenBroadcast,
    VisionGatherFn visionGather, int32_t parallelRank)
{
    bool expected = false;
    if (!mHandleRequestInProgress.compare_exchange_strong(
            expected, true, std::memory_order_acquire, std::memory_order_relaxed))
    {
        LOG_ERROR("Overlapping requests on one runtime are not supported.");
        return nullptr;
    }
    auto generation
        = std::make_unique<SteppedGeneration>(mHandleRequestInProgress, mHostRaggedTokenIds, mDeviceFailure, stream);
    if (!unavailableReason().empty())
    {
        return nullptr;
    }
    selectProfilingTimer();
    gTimer.beginRequest();

    mTokenBroadcast = std::move(tokenBroadcast);
    mVisionGather = std::move(visionGather);
    mParallelRank = parallelRank;
    if (mDecodingRuntimeContext)
    {
        mDecodingRuntimeContext->tokenBroadcast = mTokenBroadcast;
        mDecodingRuntimeContext->parallelRank = mParallelRank;
    }
    if (mParallelRank > 0 && !mTokenBroadcast)
    {
        LOG_ERROR("Parallel rank %d requires a token broadcast callback.", mParallelRank);
        return nullptr;
    }
    // Clear per-request portal state. Buffers themselves stay allocated and are
    // reshaped/overwritten when populated below — see getBaseModelHiddenStates() contract.
    mHiddenStatesRegistry.clear();
    mLastPrefillLength = 0;
    mLastInputTokenIds.clear();

    // Clear per-request response state. On failure (early return) the vectors
    // stay empty; on success they are repopulated together below to matched sizes.
    response.outputIds.clear();
    response.outputTexts.clear();
    response.logprobs.clear();
    response.outputTrajectories.clear();
    response.finishReasons.clear();
    response.inputTokenCounts.clear();
    response.reusedInputTokenCounts.clear();
    response.specVerifyCounts.clear();
    response.decodeRoundCounts.clear();
    response.specAcceptanceLengths.clear();

    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());
    std::string const& loraWeightsName = request.loraWeightsName;

    if (!validateRequestConfig(request))
    {
        return nullptr;
    }

    if (getProfilingEnabled())
    {
        gTimer.reserve(256, activeBatchSize);
    }

    if (!validateStreamingSubmission(request))
    {
        return nullptr;
    }

    generation->strategy = &mDecoderRegistry->select(request);
    DecodingStrategy& decodingStrategy = *generation->strategy;
    bool const enableSpecDecode = decodingStrategy.isSpeculative();
    ELLM_CHECK(mBenchmarkOptions.forcedAcceptanceLength < 0 || enableSpecDecode,
        "Forced acceptance cannot run a vanilla decoder or a speculative fallback");
    ELLM_CHECK(mBenchmarkOptions.forcedAcceptanceLength < 0 || !hasGuidedDecoding(request),
        "Forced acceptance cannot be combined with guided decoding");
    generation->enableSpecDecode = enableSpecDecode;
    // DSpark implements the paper-equivalent probabilistic verifier and can keep non-greedy sampling params.
    // Other speculative decoders still run greedy-compatible verification.
    bool const hasNonGreedySampling = shouldUseNonGreedySampling(request.temperature, request.topK, request.topP);
    DecodingStrategyCapabilities const decodingCapabilities = decodingStrategy.capabilities();
    bool const losslessSamplingSpecDecode = enableSpecDecode && decodingCapabilities.supportsLosslessSampling;
    if (enableSpecDecode && hasNonGreedySampling && !losslessSamplingSpecDecode)
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
        return nullptr;
    }

    DecodingInferenceContext& context = generation->context;
    context.initialize(activeBatchSize, maxGenerateLength, std::nullopt, rt::OptionalInputTensors{}, loraWeightsName,
        stream, mDeployment.base.recurrentPoolRows);
    context.initializeRaggedScratch(RaggedEngineContract{TokenLayoutBackend::kEntryPaddedCompatibility,
        mDeployment.base.maxNumSequences, mDeployment.base.maxQueryLength, mDeployment.base.maxPhysicalTokens,
        mDeployment.base.recurrentPoolRows, /* mixedStepSupported = */ false});
    context.layerDebugger = LayerDebugger::fromEnv();
    ELLM_CHECK(mBenchmarkOptions.forcedAcceptanceLength < 0 || context.layerDebugger == nullptr
            || !context.layerDebugger->hasForcedTokens(),
        "Forced acceptance cannot be combined with teacher forcing");
    if (context.layerDebugger != nullptr)
    {
        context.layerDebugger->setNgramPreprocessor(mQwen4ExpNgram.get());
    }
    bool const supportsMultimodalInput
        = (mAudioRunner != nullptr) || (mVisionRunner != nullptr) || (mActionRunner != nullptr);

    if (supportsMultimodalInput)
    {
        if (!multiModalRuntimePreprocess(request, context, stream))
        {
            return nullptr;
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
                return nullptr;
            }
        }
    }

    bool const hasActionRequest = mActionRunner != nullptr
        && std::any_of(request.requests.begin(), request.requests.end(),
            [](auto const& req) { return req.pastTrajectory.has_value(); });
    generation->hasActionRequest = hasActionRequest;
    if (hasActionRequest)
    {
        ELLM_CHECK(mActionKvBatchCollector != nullptr, "Action KV batch collector was not initialized");
        if (mVisionRunner == nullptr || mVisionRunner->getModelType() != multimodal::ModelType::QWEN3_VL)
        {
            LOG_ERROR("Alpamayo1ActionRunner requires a Qwen3-VL vision runner for MRoPE deltas.");
            return nullptr;
        }
        auto* qwenVision = static_cast<rt::QwenViTRunner*>(mVisionRunner.get());
        std::vector<int64_t> const& ropeDeltas = qwenVision->getMropeRopeDeltasPerBatch();
        if (ropeDeltas.size() != request.requests.size())
        {
            LOG_ERROR("MRoPE delta count %zu does not match request batch size %zu", ropeDeltas.size(),
                request.requests.size());
            return nullptr;
        }
        std::vector<bool> actionSlots;
        actionSlots.reserve(request.requests.size());
        for (auto const& slot : request.requests)
        {
            actionSlots.push_back(slot.pastTrajectory.has_value());
        }
        mActionKvBatchCollector->beginRequest(actionSlots, ropeDeltas);
        generation->actionKvCollector = mActionKvBatchCollector.get();
    }

    // Only the cache-miss suffix reaches a prefill call, so maxInputLen is checked in
    // setUpForPrefillExecution; a whole prompt is bounded by KV capacity.
    for (size_t i = 0; i < context.rawBatchedInputIds.size(); ++i)
    {
        int32_t const inputLen = static_cast<int32_t>(context.rawBatchedInputIds[i].size());
        if (inputLen > mDeployment.base.maxKVCacheCapacity)
        {
            LOG_ERROR(
                "Input length (%d) exceeds engine KV cache capacity (%d). "
                "Rebuild the engine with a larger --maxKVCacheCapacity.",
                inputLen, mDeployment.base.maxKVCacheCapacity);
            throw std::runtime_error("EDGELLM_INPUT_TOO_LONG: input length " + std::to_string(inputLen)
                + " exceeds engine max_kv_cache_capacity " + std::to_string(mDeployment.base.maxKVCacheCapacity)
                + " (rebuild engine with a larger --maxKVCacheCapacity)");
        }
    }

    // Lossless speculative decoders keep non-greedy parameters. Requests outside a decoder's exact support
    // are routed to the default decoder before this point.
    bool const forceGreedySpecDecode = enableSpecDecode && !losslessSamplingSpecDecode;
    // A batch built at request start gives every slot the request's scalars.
    RowSamplingParams const sampling = forceGreedySpecDecode ? RowSamplingParams{} : rowSamplingParamsFor(request);
    context.samplingParams.assign(activeBatchSize, sampling);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        context.samplingSeeds[i] = resolveSamplingSeed(request, request.requests[i]);
        context.requestStableSampling[i]
            = static_cast<int8_t>(usesRequestStableSampling(request, request.requests[i], losslessSamplingSpecDecode));
    }
    context.ignoreEos.assign(activeBatchSize, static_cast<int8_t>(request.ignoreEos));
    context.proposalSampling = request.proposalSampling;
    context.diffusionMaxDenoisingSteps = request.diffusionMaxDenoisingSteps;
    context.outputThinkerEmbeddings = outputThinkerEmbeddings;
    context.onTokenGenerated = request.onTokenGenerated;
    context.enableThinking = request.enableThinking;

    // Reasoning markers, looked up once: -1 when the tokenizer has no such token. Shared with
    // the grammar gate, which must agree on which token ends the block.
    std::vector<int32_t> const reasoningStartIds = reasoningStartMarkers(*mTokenizer);
    std::vector<int32_t> const reasoningEndIds = reasoningEndMarkers(*mTokenizer);

    prepareLogitBias(mLogitBias, request, context);

    // Compile each slot's grammar before any GPU work, so a failure marks just that slot.
    context.hasGuidedDecoding = false;
    if (hasGuidedDecoding(request))
    {
        // Explicit prompt markers are authoritative. If none is present, thinking mode decides
        // whether the model may still open a reasoning block before the grammar applies.
        context.guidedReasoningEnded.assign(static_cast<size_t>(activeBatchSize), 0);
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            context.guidedReasoningEnded[i] = static_cast<int8_t>(reasoningClosedInPrompt(
                context.rawBatchedInputIds[i], reasoningStartIds, reasoningEndIds, request.enableThinking));
        }

        mGuidedDecoder.reset();
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            auto const& slotGuide = request.requests[i].guidedDecoding;
            if (!slotGuide.has_value())
            {
                continue;
            }
            std::string failReason;
            if (mGuidedDecoder.prepareSlot(i, *slotGuide, failReason))
            {
                context.hasGuidedDecoding = true;
                if (jumpForwardRequested(
                        *slotGuide, decodingStrategy, hasNonGreedySampling, /*inFlightBatching=*/false))
                {
                    context.jumpForwardEnabled[i] = 1;
                    context.hasJumpForward = true;
                }
                continue;
            }
            LOG_ERROR("Request %d: failed to compile the %s guide: %s", i, guideTypeName(slotGuide->type),
                failReason.c_str());
            context.finishedStates[i] = 1;
            context.slotStreams[i].terminalReason = FinishReason::kError;
        }
        if (context.hasJumpForward)
        {
            prepareJumpForwardExecution(decodingStrategy, stream);
        }
    }

    int32_t const numLogprobs = clampLogprobsWidth(request.numLogprobs);
    context.numLogprobsPerSlot.assign(activeBatchSize, numLogprobs);
    // Spec-decode verify may accept more than 1 token in one step, overshooting the budget.
    context.logprobsStepOvershoot = mDeployment.maxAcceptedTokensPerRound() - 1;
    if (numLogprobs > 0)
    {
        ensureLogprobsCapacity(activeBatchSize * mDeployment.maxAcceptedTokensPerRound(), numLogprobs);
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
        context.slotStreams[i].skipSpecialTokens = request.skipSpecialTokens;
    }

    DecodingKvHeadroomPolicy const headroomPolicy = decodingStrategy.kvHeadroom();
    generation->kvHeadroom = headroomPolicy.plain;
    generation->jumpForwardKvHeadroom = context.hasJumpForward ? headroomPolicy.jumpForward : headroomPolicy.plain;
    DecodingKvHeadroom const& kvHeadroom = generation->kvHeadroom;
    DecodingKvHeadroom const& jumpForwardKvHeadroom = generation->jumpForwardKvHeadroom;
    DecodingTokenStateContract const tokenStateContract = decodingStrategy.tokenStateContract();
    ContextCacheCommitPolicy const contextCacheCommitPolicy = request.contextCacheCommitPolicy;
    ELLM_CHECK(
        kvHeadroom.baseExtraTokens > 0 && kvHeadroom.draftExtraTokens >= 0, "Decoder returned invalid KV headroom");
    ELLM_CHECK(!enableSpecDecode || kvHeadroom.draftExtraTokens == 0 || mDeployment.draft.has_value(),
        "Decoder requested draft KV headroom without a draft engine");
    ELLM_CHECK(jumpForwardKvHeadroom.baseExtraTokens >= kvHeadroom.baseExtraTokens
            && jumpForwardKvHeadroom.draftExtraTokens >= kvHeadroom.draftExtraTokens
            && (jumpForwardKvHeadroom.draftExtraTokens == 0) == (kvHeadroom.draftExtraTokens == 0),
        "Decoder returned invalid jump-forward KV headroom");

    std::optional<ManagedKVCacheRequest>& managedKVCacheRequest = generation->managedKVCacheRequest;
    // The deployment's media placeholder list keys media runs in the cache and locates the media tail below.
    std::vector<int32_t> const mediaTokens = mediaTokenIds();
    if (mContextCache != nullptr || mBoundedSwaKVPageManager != nullptr)
    {
        std::optional<ManagedKVCacheRequest> admitted = ManagedKVCacheRequest::begin(mContextCache.get(),
            mBoundedSwaKVPageManager.get(), request, context, decodingStrategy.isSpeculative(), kvHeadroom,
            tokenStateContract, contextCacheCommitPolicy, mediaTokens);
        if (!admitted.has_value())
        {
            return nullptr;
        }
        managedKVCacheRequest.emplace(std::move(*admitted));
    }
    if (mVisualPruner && managedKVCacheRequest.has_value() && managedKVCacheRequest->hasContextReuse())
    {
        LOG_ERROR("Visual-token pruning is not supported with managed context reuse.");
        return nullptr;
    }
    ManagedKVCacheRequest* const managedRequest = managedKVCacheRequest.has_value() ? &*managedKVCacheRequest : nullptr;

    // Conduct the preparation work to handle a new set of sequences, including inputIds packing, input/output tensor
    // preparation, reset the KVCache state, and apply reused prefix KVCache if available.
    std::vector<int32_t> const* const contextCachePrefillStarts
        = managedRequest != nullptr ? managedRequest->prefillStarts() : nullptr;
    if (!setUpForPrefillExecution(context, decodingStrategy, contextCachePrefillStarts))
    {
        LOG_ERROR("Prefill execution setup failed. This request cannot be handled.");
        return nullptr;
    }
    if (managedRequest != nullptr && !managedRequest->preparePrefill())
    {
        return nullptr;
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
    generation->streamFinalizer.emplace(context, *mTokenizer);

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
    if (mBenchmarkOptions.maxDecodeRounds > 0)
    {
        int64_t const perRound = mBenchmarkOptions.forcedAcceptanceLength > 0
            ? static_cast<int64_t>(mBenchmarkOptions.forcedAcceptanceLength)
            : mDeployment.maxAcceptedTokensPerRound();
        int64_t const requiredOutput = (mDeployment.base.isDiffusionBackbone ? 0 : 1)
            + static_cast<int64_t>(mBenchmarkOptions.maxDecodeRounds) * perRound;
        ELLM_CHECK(clampedMaxGenerateLength >= requiredOutput,
            "Fixed-round benchmark requires output and KV headroom for every complete decode round");
    }
    if (clampedMaxGenerateLength != request.maxGenerateLength)
    {
        LOG_WARNING("Reduce max generation length to %d", clampedMaxGenerateLength);
    }
    if (clampedMaxGenerateLength <= 0)
    {
        LOG_ERROR("Insufficient KV cache capacity for generation for this request.");
        return nullptr;
    }
    if (getProfilingEnabled())
    {
        int64_t const draftSteps = mDeployment.specConfig.has_value() ? mDeployment.specConfig->draftSteps : 0;
        int64_t samplesPerRound
            = std::max<int64_t>(activeBatchSize, mDeployment.specConfig.has_value() ? 32 + 4 * draftSteps : 8);
        if (mDeployment.base.isDiffusionBackbone)
        {
            int64_t const maxDenoisingSteps = request.diffusionMaxDenoisingSteps > 0
                ? std::min(request.diffusionMaxDenoisingSteps, mDeployment.base.diffusionMaxDenoisingSteps)
                : mDeployment.base.diffusionMaxDenoisingSteps;
            samplesPerRound = std::max(samplesPerRound, 8 + 2 * maxDenoisingSteps);
        }
        int64_t const rounds = mBenchmarkOptions.maxDecodeRounds > 0
            ? std::min<int64_t>(mBenchmarkOptions.maxDecodeRounds, clampedMaxGenerateLength)
            : clampedMaxGenerateLength;
        ELLM_CHECK(static_cast<uint64_t>(rounds)
                <= (std::numeric_limits<size_t>::max() - 256) / static_cast<uint64_t>(samplesPerRound),
            "Profiling request exceeds timing storage capacity");
        gTimer.reserve(256 + static_cast<size_t>(rounds) * static_cast<size_t>(samplesPerRound), activeBatchSize);
    }
    context.maxGenerateLengths.assign(activeBatchSize, clampedMaxGenerateLength);
    // A slot that runs with jump-forward keeps its wider headroom behind its own prompt; the other slots keep the
    // budget computed above.
    for (int32_t i = 0; context.hasJumpForward && i < activeBatchSize; ++i)
    {
        if (context.jumpForwardEnabled[i] == 0)
        {
            continue;
        }
        std::vector<int32_t> const promptLength{residentInputLengths[i]};
        int32_t budget = clampMaxGenerateLengthForKVCapacity(promptLength, context.maxGenerateLengths[i],
            mDeployment.base.maxKVCacheCapacity, jumpForwardKvHeadroom.baseExtraTokens);
        if (jumpForwardKvHeadroom.draftExtraTokens > 0)
        {
            budget = clampMaxGenerateLengthForKVCapacity(
                promptLength, budget, mDeployment.draft->maxKVCacheCapacity, jumpForwardKvHeadroom.draftExtraTokens);
        }
        if (budget <= 0)
        {
            LOG_ERROR("Insufficient KV cache capacity for jump-forward generation for request %d.", i);
            return nullptr;
        }
        if (budget < context.maxGenerateLengths[i])
        {
            context.maxGenerateLengths[i] = budget;
            LOG_WARNING("Reduce max generation length to %d for request %d (jump-forward KV headroom)", budget, i);
        }
    }
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        context.allocateLogprobsAccumulator(i);
    }

    // Hybrid+spec endpoint reuse recomputes the successor-dependent boundary draft slot from a saved base hidden state,
    // so a checkpoint reuses across turns regardless of the token that follows it. Enabled only when the request looks
    // up or publishes state (not a bypass request) and the deployment uses speculative decoding over a hybrid base.
    bool const requestCacheEnabled = managedRequest != nullptr && managedRequest->hasContextReuse()
        && request.contextCacheLookupPolicy != ContextCacheLookupPolicy::kBypass && !request.generateAudio
        && !context.outputThinkerEmbeddings;
    bool const hybridSpecContextReuse = shouldUseHybridSpecEndpointReuse(
        decodingStrategy.kind(), mHybridSpecContextReuseDeployment, requestCacheEnabled, requestCacheEnabled);
    bool const textOnly = !context.visualEmbeddings.has_value() && !context.audioEmbeddings.has_value();
    bool const hybridSpecTextReuse = hybridSpecContextReuse && textOnly;
    bool const hybridSpecMediaReuse = hybridSpecContextReuse && !textOnly;
    if (hybridSpecTextReuse || hybridSpecMediaReuse)
    {
        if (request.recurrentCaptureInterval != 0
            || request.contextCacheCommitPolicy != ContextCacheCommitPolicy::kPrefillStateOnly)
        {
            LOG_ERROR("Hybrid spec context reuse requires endpoint-only capture and PREFILL_STATE_ONLY commit policy");
            return nullptr;
        }
        if (hybridSpecMediaReuse && context.activeBatchSize != 1)
        {
            LOG_ERROR("Hybrid spec media context reuse requires batch size 1 (per-slot media tail not implemented)");
            return nullptr;
        }
    }

    // Prefill from the base model; subsequent iterations are delegated to the selected strategy.
    bool prefillStatus;
    if (hybridSpecTextReuse)
    {
        context.hybridSpecEndpointReuse = true;
        context.contextCacheReplayTailLength = request.contextCacheReplayTailLength;
        prefillStatus = runHybridSpecPrefill(context, decodingStrategy, *managedRequest);
    }
    else if (hybridSpecMediaReuse)
    {
        context.hybridSpecEndpointReuse = true;
        int32_t const mediaTailLength = mediaReplayTailLength(context.rawBatchedInputIds[0]);
        context.contextCacheReplayTailLength = mediaTailLength;
        LOG_DEBUG("hybridSpecMediaReuse: mediaTailLength=%d inputLen=%zu imageTokenId=%d mediaTokenIds.size=%zu",
            mediaTailLength, context.rawBatchedInputIds[0].size(), mDeployment.base.imageTokenId, mediaTokens.size());
        prefillStatus = runHybridSpecPrefill(context, decodingStrategy, *managedRequest);
    }
    else
    {
        prefillStatus = runBaseModelPrefill(context, managedRequest);
    }
    if (!prefillStatus)
    {
        LOG_ERROR("Failed to execute prefill step for base model.");
        return nullptr;
    }

    if (managedRequest != nullptr && !decodingStrategy.initializeForGeneration(context))
    {
        LOG_ERROR("Failed to initialize generation state for %s decoding strategy.", decodingStrategy.name());
        return nullptr;
    }

    std::vector<int32_t> const& commonMaterializedStateLengths = decodingStrategy.commonMaterializedStateLengths();
    if (managedRequest != nullptr && !managedRequest->completePrefill(context, commonMaterializedStateLengths))
    {
        return nullptr;
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

    return generation;
}

std::unique_ptr<RuntimeSession> LLMRankRuntime::openSession(cudaStream_t stream, SessionOptions options)
{
    bool expected = false;
    if (!mHandleRequestInProgress.compare_exchange_strong(
            expected, true, std::memory_order_acquire, std::memory_order_relaxed))
    {
        LOG_ERROR("Cannot open a runtime session while a request is being served on this runtime.");
        return nullptr;
    }
    // The session's guard releases the latch on every exit from here on, including a failed open.
    std::unique_ptr<RankRuntimeSession> session(new RankRuntimeSession(*this, stream, options.allowCompactExecution));
    if (!unavailableReason().empty())
    {
        return nullptr;
    }
    if (!supportsSeatedAdmission())
    {
        LOG_ERROR(
            "This deployment cannot seat sequences into a running batch, so it cannot host a runtime session; serve "
            "it through the blocking handleRequest path.");
        return nullptr;
    }
    if (!session->open())
    {
        return nullptr;
    }
    return session;
}

bool LLMRankRuntime::finishGeneration(SteppedGeneration& generation, LLMGenerationRequest const& request,
    LLMGenerationResponse& response, cudaStream_t stream)
{
    DecodingInferenceContext& context = generation.context;
    ManagedKVCacheRequest* const managedRequest = generation.managedRequest();
    bool const enableSpecDecode = generation.enableSpecDecode;
    bool const hasActionRequest = generation.hasActionRequest;

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
    int64_t totalAcceptedDraftTokens = 0;
    int64_t totalProposedDraftTokens = 0;

    // Accumulate from completed batches
    for (auto const& [originalIdx, batchResult] : context.completedBatches)
    {
        int32_t rawPromptLength = static_cast<int32_t>(batchResult.rawBatchedInputIds.size());
        int32_t computedLength = batchResult.effectivePrefillLength;
        totalReusedTokens += reusedPromptTokens(rawPromptLength, computedLength, batchResult.prunedPrefillTokens);
        totalComputedTokens += computedLength;
        totalPrunedTokens += batchResult.prunedPrefillTokens;
        totalGeneratedTokens += batchResult.generateLength;
        totalIterations += batchResult.actualIterations;
        totalAcceptedDraftTokens += batchResult.acceptedDraftTokens;
        totalProposedDraftTokens += batchResult.proposedDraftTokens;
    }

    mPrefillMetrics.recordRun(totalReusedTokens, totalComputedTokens, totalPrunedTokens);
    if (enableSpecDecode)
    {
        mSpecDecodeGenerationMetrics.recordRun(
            totalIterations, totalGeneratedTokens, totalAcceptedDraftTokens, totalProposedDraftTokens);
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
    response.reusedInputTokenCounts.assign(context.completedBatches.size(), 0);
    response.specVerifyCounts.assign(context.completedBatches.size(), 0);
    response.decodeRoundCounts.assign(context.completedBatches.size(), 0);
    response.specAcceptanceLengths.assign(context.completedBatches.size(), 0.0F);

    // Add outputs from completed batches (using saved original indices)
    for (auto const& [originalIdx, batchResult] : context.completedBatches)
    {
        response.decodeRoundCounts[originalIdx] = batchResult.actualIterations;
        int32_t genLength = batchResult.generateLength;

        // Log acceptance metrics for evicted batch
        if (enableSpecDecode)
        {
            float const acceptLength = batchResult.actualIterations > 0
                ? static_cast<float>(genLength) / static_cast<float>(batchResult.actualIterations)
                : 0.0f;
            float const acceptanceRate = batchResult.proposedDraftTokens > 0
                ? static_cast<float>(batchResult.acceptedDraftTokens)
                    / static_cast<float>(batchResult.proposedDraftTokens)
                : 0.0f;
            LOG_DEBUG(
                "Batch (completed with SpecDecode, original idx %d) - Accept length: %.3f, Acceptance rate: "
                "%.3f, Generated tokens: %d, Iterations: %d",
                originalIdx, acceptLength, acceptanceRate, genLength, batchResult.actualIterations);
        }

        // Extract generated tokens
        int32_t const totalLength = static_cast<int32_t>(batchResult.tokenIds.size());

        check::check(totalLength >= genLength, "Total length should be greater than or equal to generated length");
        response.outputIds[originalIdx] = std::vector<int32_t>(
            batchResult.tokenIds.begin() + (totalLength - genLength), batchResult.tokenIds.end());
        response.outputTexts[originalIdx]
            = mTokenizer->decode(response.outputIds[originalIdx], request.skipSpecialTokens);
        response.logprobs[originalIdx] = batchResult.logprobs;
        response.finishReasons[originalIdx] = batchResult.terminalReason;
        // Prompt length after chat templating and media expansion (OpenAI usage).
        response.inputTokenCounts[originalIdx] = static_cast<int32_t>(batchResult.rawBatchedInputIds.size());
        response.reusedInputTokenCounts[originalIdx]
            = reusedPromptTokens(static_cast<int32_t>(batchResult.rawBatchedInputIds.size()),
                batchResult.effectivePrefillLength, batchResult.prunedPrefillTokens);
        if (enableSpecDecode)
        {
            response.specVerifyCounts[originalIdx] = batchResult.actualIterations;
            response.specAcceptanceLengths[originalIdx] = batchResult.actualIterations > 0
                ? static_cast<float>(batchResult.generateLength) / static_cast<float>(batchResult.actualIterations)
                : 0.0F;
        }

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
        rt::HybridCacheManager const& kvcache = *mSharedResources->cacheManagers[0];
        rt::ActionKvBatchView const actionKvBatch = mActionKvBatchCollector->materialize(stream);
        std::vector<std::vector<rt::FutureTrajectoryPoint>> trajectories;
        try
        {
            trajectories = mActionRunner->sampleTrajectory(stream, kvcache, actionKvBatch);
        }
        catch (...)
        {
            mActionKvBatchCollector->releaseMaterialized(stream);
            throw;
        }
        mActionKvBatchCollector->releaseMaterialized(stream);
        if (!mActionRunner->isRoot())
        {
            ELLM_CHECK(trajectories.empty(), "Non-root action rank unexpectedly produced trajectories");
            return true;
        }
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

uint64_t LLMRankRuntime::resolveSamplingSeed(
    LLMGenerationRequest const& request, LLMGenerationRequest::Request const& item) noexcept
{
    if (item.samplingSeed.has_value())
    {
        return *item.samplingSeed;
    }
    if (request.samplingSeed.has_value())
    {
        return *request.samplingSeed;
    }
    return kDefaultSamplingSeed;
}

bool LLMRankRuntime::usesRequestStableSampling(LLMGenerationRequest const& request,
    LLMGenerationRequest::Request const& item, bool losslessSamplingSpecDecode) noexcept
{
    return item.samplingSeed.has_value() || request.samplingSeed.has_value() || losslessSamplingSpecDecode;
}

int32_t LLMRankRuntime::clampLogprobsWidth(int32_t requested) noexcept
{
    if (requested > static_cast<int32_t>(kMaxLogprobsK))
    {
        LOG_WARNING("numLogprobs %d exceeds maximum %d; clamping.", requested, kMaxLogprobsK);
    }
    // A negative width means disabled, like zero.
    return std::max(0, std::min(requested, static_cast<int32_t>(kMaxLogprobsK)));
}

int64_t LLMRankRuntime::clampSamplingTopK(int64_t requested) noexcept
{
    if (requested > kMaxSamplingTopK)
    {
        LOG_WARNING("topK %" PRId64 " exceeds the sampler's maximum %d; clamping.", requested, kMaxSamplingTopK);
        return kMaxSamplingTopK;
    }
    return requested;
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
    bool const gemma4MtpTree = mDeployment.specDecodeMode() == SpecDecodeMode::kGemma4MTP
        && mDeployment.specConfig.has_value() && mDeployment.specConfig->draftTopK > 1;
    // The sampler's domain: rejected here rather than thrown from inside a step, where it would fail
    // every resident. Top-k has no upper bound here; rowSamplingParamsFor clamps it.
    if (!std::isfinite(request.temperature) || request.temperature < 0.0F)
    {
        LOG_ERROR("temperature must be finite and non-negative; got %f.", static_cast<double>(request.temperature));
        return false;
    }
    if (!std::isfinite(request.topP) || request.topP <= 0.0F || request.topP > 1.0F)
    {
        LOG_ERROR("topP must be in (0, 1]; got %f.", static_cast<double>(request.topP));
        return false;
    }
    if (gemma4MtpTree && shouldUseNonGreedySampling(request.temperature, request.topK, request.topP))
    {
        LOG_ERROR(
            "Gemma4 MTP tree decoding supports greedy requests only. Build a standalone target engine for "
            "non-greedy sampling.");
        return false;
    }
    if (hasGuidedDecoding(request))
    {
        // Block diffusion denoises a whole canvas per step rather than appending one token, so
        // it carries neither of the mask hooks. Rejecting beats accepting and ignoring the guide.
        if (mDeployment.base.isDiffusionBackbone)
        {
            LOG_ERROR("guided_decoding is not supported by a block-diffusion engine yet.");
            return false;
        }
    }
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        bool const hasPreTokenizedInput
            = i < static_cast<int32_t>(request.preTokenizedInputIds.size()) && !request.preTokenizedInputIds[i].empty();
        if (request.requests[i].messages.empty() && !hasPreTokenizedInput)
        {
            LOG_ERROR("Request %d in batch is empty: no messages or pre-tokenized input provided", i);
            return false;
        }
        if (request.requests[i].guidedDecoding.has_value())
        {
            // Batch-level, unlike a compile failure: an unsupported guide is a request
            // configuration error, in the same class as an out-of-range logit_bias.
            std::string failReason;
            if (!validateGuidedDecodingParams(*request.requests[i].guidedDecoding, failReason))
            {
                LOG_ERROR("Request %d has an unsupported guided_decoding guide: %s", i, failReason.c_str());
                return false;
            }
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

    return true;
}

std::vector<std::vector<Hash128>> LLMRankRuntime::buildImageContentDigests(
    LLMGenerationRequest const& request, cudaStream_t stream)
{
    // Every image in the request gets its own range of mImageDigestRows, so all of their kernels can be
    // launched back-to-back and drained with one sync instead of one sync per image.
    std::vector<std::vector<rt::imageUtils::ImageDigestLaunch>> launches(request.requests.size());
    int64_t rowOffset = 0;
    for (size_t seqIdx = 0; seqIdx < request.requests.size(); ++seqIdx)
    {
        auto const& images = request.requests[seqIdx].imageBuffers;
        launches[seqIdx].reserve(images.size());
        for (size_t imageIdx = 0; imageIdx < images.size(); ++imageIdx)
        {
            try
            {
                launches[seqIdx].push_back(
                    rt::imageUtils::launchImageContentDigest(images[imageIdx], mImageDigestRows, rowOffset, stream));
                rowOffset += launches[seqIdx].back().rowCount;
            }
            catch (std::exception const& error)
            {
                // launchImageContentDigest also throws when this request's images together outgrow
                // limits::multimodal::kMaxImageDigestRows, the capacity of mImageDigestRows.
                LOG_WARNING(
                    "Image %zu of request %zu has no content identity (%s); the encoder embedding cache "
                    "and the context cache both skip this request.",
                    imageIdx, seqIdx, error.what());
                return {};
            }
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<std::vector<Hash128>> digests(request.requests.size());
    for (size_t seqIdx = 0; seqIdx < request.requests.size(); ++seqIdx)
    {
        digests[seqIdx].reserve(launches[seqIdx].size());
        for (auto const& launch : launches[seqIdx])
        {
            digests[seqIdx].push_back(rt::imageUtils::finishImageContentDigest(launch, mImageDigestRows));
        }
    }
    return digests;
}

void LLMRankRuntime::initializeVisionDataParallel()
{
    constexpr int32_t kTpSize{2};
    if (mMapping.tensorParallelSize != kTpSize || !mVisionRunner
        || mVisionRunner->getModelType() != multimodal::ModelType::QWEN3_VL)
    {
        return;
    }

    auto* qwenVision = static_cast<rt::Qwen3VLViTRunner*>(mVisionRunner.get());
    qwenVision->initializeDataParallel();
}

bool LLMRankRuntime::runJointVisionDataParallel(LLMGenerationRequest const& request, VisionDpPlan const& plan,
    DecodingInferenceContext& context, std::vector<std::vector<int32_t>>& batchedInputIds, cudaStream_t stream,
    OptionalOutputTensor mropeCosSinOut)
{
    constexpr int32_t kTpSize{2};
    int32_t const rank = mMapping.tensorParallelRank;
    ELLM_CHECK(rank >= 0 && rank < kTpSize, "Joint-image vision data parallelism requires a valid TP2 rank.");

    auto const& images = request.requests.front().imageBuffers;
    size_t const localBegin = rank == 0 ? 0 : plan.splitIndex;
    size_t const localEnd = rank == 0 ? plan.splitIndex : images.size();
    auto* qwenVision = static_cast<rt::Qwen3VLViTRunner*>(mVisionRunner.get());
    int64_t const outputTokenOffset = rank == 0 ? 0 : plan.tokenCounts[0];
    if (!qwenVision->preprocessDataParallel(request, rank, plan, batchedInputIds, mTokenizer, mropeCosSinOut, stream)
        || !qwenVision->infer(stream, outputTokenOffset))
    {
        LOG_ERROR("Joint-image vision data-parallel execution failed on rank %d.", rank);
        return false;
    }

    auto const [rank0Tokens, rank1Tokens] = plan.tokenCounts;
    auto const& outputs = qwenVision->getOutputTensors();
    {
        TIME_STAGE(metrics::StageNames::kVISION_GATHER, stream);
        for (auto const& output : outputs)
        {
            if (!gatherVisionTensor(output.get(), plan.tokenCounts, rank, mVisionGather, stream))
            {
                return false;
            }
        }
    }

    context.deepstackFeatures.reserve(outputs.size() - 1);
    for (size_t index = 1; index < outputs.size(); ++index)
    {
        context.deepstackFeatures.emplace_back(std::cref(outputs[index].get()));
    }
    LOG_INFO(
        "Joint-image vision DP: rank %d encoded images [%zu, %zu), split %zu+%zu, "
        "gathered token shards %ld+%ld",
        rank, localBegin, localEnd, plan.splitIndex, images.size() - plan.splitIndex, rank0Tokens, rank1Tokens);
    return true;
}

bool LLMRankRuntime::multiModalRuntimePreprocess(LLMGenerationRequest const& request, DecodingInferenceContext& context,
    cudaStream_t stream, OptionalOutputTensor mropeCosSinOverride)
{
    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());
    bool const hasAudio = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.audioBuffers.empty(); });
    bool const hasVision = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.imageBuffers.empty(); });
    bool const hasTrajectoryHistory = std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& req) { return req.pastTrajectory.has_value(); });
    VisionDpPlan const* const visionDpPlan = hasVision && !hasAudio && !hasTrajectoryHistory
            && mMapping.tensorParallelSize == 2 && mVisionGather && mVisionRunner
            && mVisionRunner->getModelType() == multimodal::ModelType::QWEN3_VL
        ? static_cast<rt::QwenViTRunner*>(mVisionRunner.get())->planDataParallel(request)
        : nullptr;

    // Clear request-scoped multimodal state up front so previous requests cannot leak through reused runtime members.
    context.visualEmbeddings = std::nullopt;
    context.audioEmbeddings = std::nullopt;
    context.deepstackFeatures.clear();
    // Treat multimodal indices as request-scoped state. Only request paths that explicitly rebuild
    // mMultimodalIndices for the current request should observe a non-empty tensor downstream.
    check::check(mMultimodalIndices.reshape({0}), "Tensor reshape failed");

    // Ahead of the encoders, which is where the embedding cache reads them to decide whether the ViT
    // runs at all. The context cache reads the same table later, at admission.
    if (hasVision && (mEncoderEmbeddingCache != nullptr || mContextCache != nullptr))
    {
        context.imageContentDigests = buildImageContentDigests(request, stream);
    }

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
        ? (mropeCosSinOverride.has_value() ? mropeCosSinOverride
                                           : rt::OptionalOutputTensor{std::ref(mPipelineIO->mropeActiveCosSin)})
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
                    if (audio.pcm && audio.pcm->numSamples() > 0)
                    {
                        rt::Tensor const& samples = *audio.pcm->samples;
                        auto const* rawPtr = reinterpret_cast<char const*>(samples.dataPointer<float>());
                        size_t const rawBytes = static_cast<size_t>(audio.pcm->numSamples()) * sizeof(float);
                        audioHashes.push_back(hashOpaqueIdentity(std::string_view(rawPtr, rawBytes), stream, false));
                    }
                    else
                    {
                        hashable = false;
                    }
                }
            }
            if (!hashable)
            {
                LOG_WARNING(
                    "A clip of this request carries no PCM samples to identify it by; the encoder "
                    "embedding cache skips the request.");
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
                if (tokenLengths.size() == audioHashes.size())
                {
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
        context.audioEmbeddings = std::cref(mAudioRunner->getOutputEmbedding());
    }

    // Joint-image DP bypasses the encoder cache so every rank executes the same visual collectives.
    if (visionDpPlan != nullptr)
    {
        if (!runJointVisionDataParallel(request, *visionDpPlan, context, batchedInputIds, stream, mropeCosSinOut))
        {
            return false;
        }
    }
    // Process vision inputs (if present)
    else if (hasVision && mVisionRunner)
    {
        bool visionCacheHit = false;
        bool visionTextPreprocessed = false;
        std::vector<Hash128> imageHashes;

        if (mEncoderEmbeddingCache)
        {
            // Flattened in the order the digests were built, which is the order the encoder lays its
            // output embeddings out in; an empty table keeps the request out of the lookup and store below.
            for (auto const& sequenceDigests : context.imageContentDigests)
            {
                imageHashes.insert(imageHashes.end(), sequenceDigests.begin(), sequenceDigests.end());
            }
        }

        // A hit lays the prompt out without the encoder, which a request whose layout the encoder determines
        // cannot take.
        bool allHit
            = mEncoderEmbeddingCache && !imageHashes.empty() && !mVisionRunner->tokenLayoutNeedsEncoder(request);
        if (allHit)
        {
            for (auto const& h : imageHashes)
            {
                auto r = mEncoderEmbeddingCache->lookupEntry(h);
                if (!r)
                {
                    allHit = false;
                    break;
                }
            }
        }

        if (allHit)
        {
            if (!mVisionRunner->preprocess(request, batchedInputIds, mTokenizer, mropeCosSinOut, stream, false, true))
            {
                LOG_ERROR("Vision text preprocessing failed on cache hit.");
                return false;
            }
            visionTextPreprocessed = true;
            visionCacheHit = mEncoderEmbeddingCache->tryRestore(imageHashes, mVisionRunner->getLastMediaTokenLengths(),
                mVisionRunner->getOutputEmbedding(), mVisionRunner->getDeepstackFeatures(), stream);
            if (visionCacheHit)
            {
                LOG_INFO("Encoder embedding cache HIT for all %zu images — skipping ViT encoder execution",
                    imageHashes.size());
            }
        }

        if (!visionCacheHit)
        {
            // On a layout mismatch, text and MRoPE are already prepared for this request. Only redo pixel work.
            if (!mVisionRunner->preprocess(
                    request, batchedInputIds, mTokenizer, mropeCosSinOut, stream, visionTextPreprocessed))
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
                auto const features = mVisionRunner->getDeepstackFeatures();
                auto const& tokenLengths = mVisionRunner->getLastMediaTokenLengths();
                if (tokenLengths.size() == imageHashes.size())
                {
                    int64_t const hiddenSize = output.getShape()[1];
                    size_t const typeSize = rt::utils::getTypeSize(output.getDataType());
                    int64_t tokenOffset = 0;
                    for (size_t i = 0; i < imageHashes.size(); ++i)
                    {
                        int64_t const numTok = tokenLengths[i];
                        mEncoderEmbeddingCache->storeSlice(imageHashes[i],
                            static_cast<char const*>(output.rawPointer())
                                + tokenOffset * hiddenSize * static_cast<int64_t>(typeSize),
                            numTok, hiddenSize, output.getDataType(), stream, features, tokenOffset);
                        tokenOffset += numTok;
                    }
                }
            }
        }
        auto const features = mVisionRunner->getDeepstackFeatures();
        context.deepstackFeatures.assign(features.begin(), features.end());
    }
    if (hasVision && mVisionRunner)
    {
        context.visualEmbeddings = std::cref(mVisionRunner->getOutputEmbedding());
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
    }

    // Populate system prompts and raw input IDs from batchedInputIds
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        context.systemPrompts[i] = request.formattedRequests[i].formattedSystemPrompt;
        context.rawBatchedInputIds.push_back(batchedInputIds[i]);
    }

    return true;
}

int32_t LLMRankRuntime::mediaReplayTailLength(std::vector<int32_t> const& inputIds) const
{
    std::vector<int32_t> const mediaTokens = mediaTokenIds();
    int32_t tailLength = 0;
    if (!mediaTokens.empty())
    {
        for (auto it = inputIds.rbegin(); it != inputIds.rend(); ++it)
        {
            if (std::find(mediaTokens.begin(), mediaTokens.end(), *it) != mediaTokens.end())
            {
                break;
            }
            ++tailLength;
        }
    }
    return tailLength;
}

bool LLMRankRuntime::runHybridSpecPrefill(DecodingInferenceContext& context, DecodingStrategy& strategy,
    ManagedKVCacheRequest& managedKVCacheRequest, int32_t seatedRow)
{
    // Adapted from reference llmInferenceRuntime.cpp::runHybridSpecPrefill (:2515-2702). The reference reads
    // context.sequenceCacheStates + mHybridSnapshotStorage directly; on this target the cache lifecycle is owned by the
    // coordinator, so publication/restore go through the adapter and the reuse length comes from the admission plan.
    // The fold (row shift + boundary-hidden restore + boundary-token prepend) runs here in the runtime, exactly as in
    // the reference, because the coordinator is only reachable from the runtime side (not from MTPDecoder).
    //
    // Batch >= 1 is supported for the non-volatile-tail path: each slot independently resolves its reuse plan (cold vs
    // hot) and the base + draft prefills are batched across all slots. Volatile-tail two-chunk prefill remains batch-1.
    int32_t const batchSize = context.activeBatchSize;
    int32_t const replayTailLength = context.contextCacheReplayTailLength;

    // Volatile-tail two-chunk prefill interleaves per-slot token manipulation with base/draft prefill calls. Batching
    // those branches requires dynamic per-slot scheduling; restrict to batch == 1 until that is needed.
    ELLM_CHECK(batchSize >= 1, "Hybrid spec endpoint prefill requires at least one sequence");
    // A sequence seated into a running batch runs as row zero of a batch-one projection (prefillSlotInPlace); the
    // cache knows it by its row in the full batch.
    bool const seated = seatedRow >= 0;
    ELLM_CHECK(!seated || batchSize == 1, "A seated Hybrid+spec prefill runs as a batch of one");
    auto const sequenceIndex = [&](int32_t slot) { return seated ? seatedRow : slot; };
    if (batchSize > 1 && replayTailLength > 0)
    {
        LOG_ERROR("Hybrid spec prefill with volatile generation-prompt tail requires batch size 1");
        return false;
    }

    // Per-slot reuse plan.
    std::vector<int32_t> reuseLengths(static_cast<size_t>(batchSize));
    std::vector<int32_t> inputLengths(static_cast<size_t>(batchSize));
    std::vector<int32_t> suffixLensOrig(static_cast<size_t>(batchSize));
    for (int32_t s = 0; s < batchSize; ++s)
    {
        reuseLengths[s] = managedKVCacheRequest.reuseTokenLength(sequenceIndex(s));
        inputLengths[s] = math::cast<int32_t>(context.rawBatchedInputIds[s].size());
        suffixLensOrig[s] = context.effectivePrefillLengths[s];
        LOG_DEBUG("Hybrid+spec prefill slot %d: inputLength=%d reuseLength=%d basePrefill=%d", s, inputLengths[s],
            reuseLengths[s], inputLengths[s] - reuseLengths[s]);
    }

    int32_t const boundaryHiddenDim = mDeployment.specConfig->baseOutputHiddenDim;
    size_t const rowBytes
        = static_cast<size_t>(boundaryHiddenDim) * rt::utils::getTypeSize(mPipelineIO->baseHiddenStates.getDataType());

    // When the boundary fold re-strides slot data to a wider pitch, this holds the new per-slot stride so that
    // foldBoundaryHiddenForSlot addresses slots correctly in the 2D layout. Zero means no re-stride was performed.
    int32_t foldSlotStride = 0;

    // Run the (folded) draft prefill of a chunk that ends before the prompt does (the stable predecessor of a
    // volatile tail). A blocking request goes through the strategy's initialize-for-generation override: resetting
    // the guard lets it run again where the reference re-invokes prepareFirstDecodeStep (mirrors ref :2560/:2626).
    // Driving a re-run through a guard flag is a workaround for DecodingStrategy having no explicit "run the draft
    // prefill now" entry point. A seated sequence has one: DecodingStrategy::prefillSeatedChunk.
    auto runChunkDraftPrefill = [&]() -> bool {
        if (seated)
        {
            return strategy.prefillSeatedChunk(context);
        }
        context.speculativeDraftPrefillComplete = false;
        return strategy.initializeForGeneration(context);
    };
    // The draft prefill of the prompt's last chunk. The blocking invocation leaves speculativeDraftPrefillComplete =
    // true so the outer initializeForGeneration and decode round 0 both skip it; a seated sequence is seated by it.
    auto runFinalDraftPrefill
        = [&]() -> bool { return seated ? strategy.seatRow(context, seatedRow) : runChunkDraftPrefill(); };
    // A hit restores the draft's paged state only up to reuseLength - futureDependencyTokens: the fold rewrites the
    // boundary positions, so the draft cache length of each folded slot starts there.
    auto rewindDraftToBoundary = [&](std::vector<int32_t> const& foldedSlots) {
        if (mSpecFutureDependencyTokens == 0)
        {
            return;
        }
        check::check(mPipelineIO->hostContextLengths.reshape({batchSize}), "Tensor reshape failed");
        check::check(mPipelineIO->contextLengths.reshape({batchSize}), "Tensor reshape failed");
        int32_t* rewind = mPipelineIO->hostContextLengths.dataPointer<int32_t>();
        std::fill_n(rewind, batchSize, 0);
        for (int32_t const slot : foldedSlots)
        {
            rewind[slot] = -mSpecFutureDependencyTokens;
        }
        CUDA_CHECK(cudaMemcpyAsync(mPipelineIO->contextLengths.rawPointer(), rewind,
            static_cast<size_t>(batchSize) * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
        mPipelineIO->recordStepHostUploads(context.stream);
        mSharedResources->cacheManagers[1]->commitSequenceLength(mPipelineIO->contextLengths, context.stream);
    };

    // Per-slot boundary fold: shift baseHiddenStates rows within one batch slot by +1 and restore the reused
    // checkpoint's boundary hidden into the slot's first row. After runBaseModelPrefill the tensor is 2D
    // [batchSize * maxSeqLen, hiddenDim]; each slot occupies maxSeqLen contiguous rows starting at
    // (slot * maxSeqLen). For a 3D tensor [batch, seq, hidden] the stride is shape[1] instead.
    auto foldBoundaryHiddenForSlot = [&](int32_t slot, int32_t chunkLength) -> bool {
        ELLM_CHECK(chunkLength + 1 <= mBoundaryFoldMaxRows,
            "Hybrid+spec boundary fold needs one row beyond the prefill chunk in baseHiddenStates");
        auto const& shape = mPipelineIO->baseHiddenStates.getShape();
        int32_t const seqDim = (shape.getNumDims() == 3) ? shape[1]
            : (foldSlotStride > 0)                       ? foldSlotStride
                                                         : *std::max_element(context.effectivePrefillLengths.begin(),
                                                               context.effectivePrefillLengths.begin() + batchSize);
        int32_t const slotStartRow = slot * seqDim;
        size_t const slotOffset = static_cast<size_t>(slotStartRow) * rowBytes;
        auto* slotBase = static_cast<std::byte*>(mPipelineIO->baseHiddenStates.rawPointer()) + slotOffset;
        CUDA_CHECK(cudaMemcpyAsync(mBoundaryFoldScratch.rawPointer(), slotBase,
            static_cast<size_t>(chunkLength) * rowBytes, cudaMemcpyDeviceToDevice, context.stream));
        CUDA_CHECK(cudaMemcpyAsync(slotBase + rowBytes, mBoundaryFoldScratch.rawPointer(),
            static_cast<size_t>(chunkLength) * rowBytes, cudaMemcpyDeviceToDevice, context.stream));
        return managedKVCacheRequest.restoreHybridSpecBoundaryHidden(
            sequenceIndex(slot), mPipelineIO->baseHiddenStates, slotStartRow);
    };

    // --- Volatile-tail branches (batch size 1, guarded above) ---

    // The generation-prompt tail is volatile when the chat template appends tokens that the next turn's render of the
    // same history does not reproduce (for example Qwen3's `<think>\n\n</think>\n\n` under enable_thinking=false).
    // Those tokens must not enter the published checkpoint, so the prefill splits into a stable predecessor chunk plus
    // a replayed tail. The shared native chat-template renderer computes the tail before dispatching to ranks.
    bool const hasVolatileTail = batchSize == 1 && replayTailLength > 0 && suffixLensOrig[0] > replayTailLength;

    // Cold sequence with a volatile generation-prompt tail: publish the checkpoint at the STABLE boundary
    // predecessorLength = inputLength - replayTailLength (two-chunk prefill), not at the full inputLength (ref :2534).
    if (reuseLengths[0] == 0 && hasVolatileTail)
    {
        std::vector<int32_t> const completeSuffix = context.tokenIds[0];
        int32_t const predecessorLength = inputLengths[0] - replayTailLength;
        int32_t const predecessorChunkLength = suffixLensOrig[0] - replayTailLength;
        auto const replayBegin = completeSuffix.end() - replayTailLength;

        // Chunk 1: base prefill of the predecessor [0, predecessorLength), no sampling.
        context.tokenIds[0].assign(completeSuffix.begin(), replayBegin);
        context.effectivePrefillLengths[0] = predecessorChunkLength;
        if (!runBaseModelPrefill(context, /*managedKVCacheRequest=*/nullptr, /*sampleOutput=*/false))
        {
            return false;
        }
        // Non-sampling chunk must sync before MTP repacks the shared pinned token buffer.
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        // Provide the boundary's next token so the predecessor draft prefill covers the boundary slot, then publish.
        context.tokenIds[0].push_back(*replayBegin);
        if (!runChunkDraftPrefill())
        {
            return false;
        }
        if (!managedKVCacheRequest.publishHybridSpecEndpoint(sequenceIndex(0), predecessorLength,
                mPipelineIO->baseHiddenStates, context.effectivePrefillLengths[0] - 1))
        {
            return false;
        }

        // Chunk 2: replay the volatile tail [predecessorLength, inputLength) with sampling, then restore bookkeeping.
        context.tokenIds[0].assign(replayBegin, completeSuffix.end());
        context.effectivePrefillLengths[0] = replayTailLength;
        if (!runBaseModelPrefill(context, /*managedKVCacheRequest=*/nullptr, /*sampleOutput=*/true))
        {
            return false;
        }
        int32_t const sampledToken = context.tokenIds[0].back();
        if (!runFinalDraftPrefill())
        {
            return false;
        }
        context.tokenIds[0] = completeSuffix;
        context.tokenIds[0].push_back(sampledToken);
        context.pendingTokens[0] = 1;
        context.effectivePrefillLengths[0] = suffixLensOrig[0];
        return true;
    }

    // Hit sequence with a volatile tail: publish at the stable predecessorLength like the cold path, but first
    // reconstruct the reused checkpoint's boundary draft slot via the consume-side fold (ref :2581) when
    // futureDependencyTokens > 0. When 0, the draft prefill runs directly without a fold.
    if (reuseLengths[0] > 0 && hasVolatileTail)
    {
        std::vector<int32_t> const completeSuffix = context.tokenIds[0];
        int32_t const predecessorLength = inputLengths[0] - replayTailLength;
        int32_t const predecessorChunkLength = suffixLensOrig[0] - replayTailLength;
        auto const replayBegin = completeSuffix.end() - replayTailLength;
        bool const needsFold = mSpecFutureDependencyTokens > 0;

        // Chunk 1: base prefill of the predecessor [reuseLength, predecessorLength), no sampling.
        context.tokenIds[0].assign(completeSuffix.begin(), replayBegin);
        context.effectivePrefillLengths[0] = predecessorChunkLength;
        if (!runBaseModelPrefill(context, /*managedKVCacheRequest=*/nullptr, /*sampleOutput=*/false))
        {
            return false;
        }
        CUDA_CHECK(cudaStreamSynchronize(context.stream));

        if (needsFold)
        {
            int32_t const foldedLen = predecessorChunkLength + 1;
            auto const& shape = mPipelineIO->baseHiddenStates.getShape();
            if (shape.getNumDims() == 3 && shape[1] < foldedLen)
            {
                ELLM_CHECK(mPipelineIO->baseHiddenStates.reshape({shape[0], foldedLen, shape[2]}),
                    "Failed to reshape baseHiddenStates for volatile-tail boundary fold");
            }
            else if (shape.getNumDims() == 2 && shape[0] < foldedLen)
            {
                ELLM_CHECK(mPipelineIO->baseHiddenStates.reshape({foldedLen, shape[1]}),
                    "Failed to reshape baseHiddenStates for volatile-tail boundary fold");
            }
            if (!foldBoundaryHiddenForSlot(0, predecessorChunkLength))
            {
                return false;
            }
            int32_t const boundaryToken = context.rawBatchedInputIds[0][static_cast<size_t>(reuseLengths[0] - 1)];
            context.tokenIds[0].insert(context.tokenIds[0].begin(), boundaryToken);
            context.effectivePrefillLengths[0] = foldedLen;
            rewindDraftToBoundary({0});
        }
        context.prefillStartLengths[0] = reuseLengths[0] - 1;
        if (!runChunkDraftPrefill())
        {
            return false;
        }
        if (!managedKVCacheRequest.publishHybridSpecEndpoint(sequenceIndex(0), predecessorLength,
                mPipelineIO->baseHiddenStates, context.effectivePrefillLengths[0] - 1))
        {
            return false;
        }
        if (needsFold)
        {
            auto const& shape = mPipelineIO->baseHiddenStates.getShape();
            if (shape.getNumDims() == 3 && shape[1] != predecessorChunkLength)
            {
                ELLM_CHECK(mPipelineIO->baseHiddenStates.reshape({shape[0], predecessorChunkLength, shape[2]}),
                    "Failed to restore baseHiddenStates shape after volatile-tail boundary fold");
            }
            else if (shape.getNumDims() == 2 && shape[0] != predecessorChunkLength)
            {
                ELLM_CHECK(mPipelineIO->baseHiddenStates.reshape({predecessorChunkLength, shape[1]}),
                    "Failed to restore baseHiddenStates shape after volatile-tail boundary fold");
            }
            context.tokenIds[0].erase(context.tokenIds[0].begin());
            context.effectivePrefillLengths[0] = predecessorChunkLength;
        }

        // Chunk 2: replay the volatile tail [predecessorLength, inputLength) with sampling, then restore bookkeeping.
        context.tokenIds[0].assign(replayBegin, completeSuffix.end());
        context.effectivePrefillLengths[0] = replayTailLength;
        if (!runBaseModelPrefill(context, /*managedKVCacheRequest=*/nullptr, /*sampleOutput=*/true))
        {
            return false;
        }
        int32_t const sampledToken = context.tokenIds[0].back();
        if (!runFinalDraftPrefill())
        {
            return false;
        }
        context.tokenIds[0] = completeSuffix;
        context.tokenIds[0].push_back(sampledToken);
        context.pendingTokens[0] = 1;
        context.effectivePrefillLengths[0] = suffixLensOrig[0];
        return true;
    }

    // --- Non-volatile-tail path (batch >= 1): mixed cold/hot slots ---

    // Base prefill of the suffix (the full prompt when cold, reuse-truncated suffix when hot). Batched across all
    // slots; the engine handles variable effective lengths via padding.
    if (!runBaseModelPrefill(context, /*managedKVCacheRequest=*/nullptr, /*sampleOutput=*/true))
    {
        return false;
    }

    // On a hit the restore path reused draft KV only up to reuseLength - futureDependencyTokens. When
    // futureDependencyTokens > 0 (MTP, EAGLE), fold the boundary hidden into the suffix draft prefill as a *context*
    // position so the single draft prefill covers the successor-dependent slot. When futureDependencyTokens == 0
    // (DFlash, JetSpec, DSpark), no fold is needed — the draft prefill runs directly on the suffix.
    bool const anySlotHot
        = std::any_of(reuseLengths.begin(), reuseLengths.begin() + batchSize, [](int32_t r) { return r > 0; });
    bool const needsBoundaryFold = mSpecFutureDependencyTokens > 0 && anySlotHot;

    // The fold shifts each hot slot's rows by +1 and inserts the boundary hidden at row 0. Compute the post-fold
    // max length first and widen the tensor so that every slot's stride includes the extra row. Without this, a hot
    // slot whose suffix equals the pre-fold maximum would write one row past the slot boundary.
    if (needsBoundaryFold)
    {
        int32_t foldedMaxLen = 0;
        for (int32_t s = 0; s < batchSize; ++s)
        {
            int32_t const len = context.effectivePrefillLengths[s] + (reuseLengths[s] > 0 ? 1 : 0);
            foldedMaxLen = std::max(foldedMaxLen, len);
        }
        auto const& shape = mPipelineIO->baseHiddenStates.getShape();
        int32_t const oldSlotStride = (shape.getNumDims() == 3)
            ? shape[1]
            : *std::max_element(
                  context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.begin() + batchSize);

        if (shape.getNumDims() == 3 && shape[1] < foldedMaxLen)
        {
            ELLM_CHECK(mPipelineIO->baseHiddenStates.reshape({shape[0], foldedMaxLen, shape[2]}),
                "Failed to reshape baseHiddenStates for boundary fold");
        }
        else if (shape.getNumDims() == 2 && shape[0] < batchSize * foldedMaxLen)
        {
            ELLM_CHECK(mPipelineIO->baseHiddenStates.reshape({batchSize * foldedMaxLen, shape[1]}),
                "Failed to reshape baseHiddenStates for boundary fold");
        }

        // The reshape above only widens the tensor metadata — the underlying buffer is unchanged, so slot data is
        // still packed at the old stride. When the stride increases and there are multiple batches, re-stride the
        // slot data back-to-front so that each slot occupies foldedMaxLen contiguous rows. Without this, the +1
        // row shift in foldBoundaryHiddenForSlot for the first hot slot would overwrite the next slot's data.
        if (foldedMaxLen > oldSlotStride && batchSize > 1)
        {
            auto* base = static_cast<std::byte*>(mPipelineIO->baseHiddenStates.rawPointer());
            for (int32_t s = batchSize - 1; s >= 1; --s)
            {
                size_t const srcOffset = static_cast<size_t>(s) * oldSlotStride * rowBytes;
                size_t const dstOffset = static_cast<size_t>(s) * foldedMaxLen * rowBytes;
                int32_t const validRows = context.effectivePrefillLengths[s];
                if (validRows > 0)
                {
                    // Source and destination may overlap (stride difference is typically 1 row), so bounce through
                    // the scratch buffer which is already sized to hold mBoundaryFoldMaxRows rows.
                    CUDA_CHECK(cudaMemcpyAsync(mBoundaryFoldScratch.rawPointer(), base + srcOffset,
                        static_cast<size_t>(validRows) * rowBytes, cudaMemcpyDeviceToDevice, context.stream));
                    CUDA_CHECK(cudaMemcpyAsync(base + dstOffset, mBoundaryFoldScratch.rawPointer(),
                        static_cast<size_t>(validRows) * rowBytes, cudaMemcpyDeviceToDevice, context.stream));
                }
            }
        }
        foldSlotStride = foldedMaxLen;
    }

    // Fold boundary hidden and prepend boundary token for each hot slot that needs it.
    std::vector<int32_t> foldedSlots;
    for (int32_t s = 0; s < batchSize; ++s)
    {
        if (reuseLengths[s] == 0 || !needsBoundaryFold)
        {
            continue;
        }
        int32_t const suffixLen = context.effectivePrefillLengths[s];
        if (!foldBoundaryHiddenForSlot(s, suffixLen))
        {
            return false;
        }
        int32_t const boundaryToken = context.rawBatchedInputIds[s][static_cast<size_t>(reuseLengths[s] - 1)];
        context.tokenIds[s].insert(context.tokenIds[s].begin(), boundaryToken);
        context.effectivePrefillLengths[s] = suffixLen + 1;
        foldedSlots.push_back(s);
    }
    for (int32_t s = 0; s < batchSize; ++s)
    {
        if (reuseLengths[s] > 0)
        {
            context.prefillStartLengths[s] = reuseLengths[s] - 1;
        }
    }
    if (!foldedSlots.empty())
    {
        rewindDraftToBoundary(foldedSlots);
    }

    if (!runFinalDraftPrefill())
    {
        return false;
    }

    // Publish per-slot endpoints. baseHiddenStates is 2D ragged [activeBatchSize * maxSeqLen, hiddenDim], so the
    // physical row for slot s is (s * maxSeqLen + effectivePrefillLengths[s] - 1).
    int32_t const maxSeqLen = *std::max_element(
        context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.begin() + batchSize);
    for (int32_t s = 0; s < batchSize; ++s)
    {
        int32_t const physicalRow = s * maxSeqLen + context.effectivePrefillLengths[s] - 1;
        if (!managedKVCacheRequest.publishHybridSpecEndpoint(
                sequenceIndex(s), inputLengths[s], mPipelineIO->baseHiddenStates, physicalRow))
        {
            return false;
        }
    }

    // Restore fold bookkeeping for hot slots and undo the baseHiddenStates reshape.
    if (needsBoundaryFold)
    {
        int32_t const restoredMaxLen = *std::max_element(suffixLensOrig.begin(), suffixLensOrig.end());
        auto const& shape = mPipelineIO->baseHiddenStates.getShape();
        if (shape.getNumDims() == 3 && shape[1] != restoredMaxLen)
        {
            ELLM_CHECK(mPipelineIO->baseHiddenStates.reshape({shape[0], restoredMaxLen, shape[2]}),
                "Failed to restore baseHiddenStates shape after boundary fold");
        }
        for (int32_t s = 0; s < batchSize; ++s)
        {
            if (reuseLengths[s] == 0)
            {
                continue;
            }
            context.tokenIds[s].erase(context.tokenIds[s].begin());
            context.effectivePrefillLengths[s] = suffixLensOrig[s];
        }
    }
    return true;
}

bool LLMRankRuntime::runBaseModelPrefill(DecodingInferenceContext& context,
    ManagedKVCacheRequest* managedKVCacheRequest, bool sampleOutput, int32_t admissionSequenceIndex)
{
    TIME_STAGE(metrics::StageNames::kLLM_PREFILL, context.stream);
    NVTX_SCOPED_RANGE(nvtx_base_prefill,
        ("SPEC_DECODE_BASE_PREFILL[" + std::to_string(context.activeBatchSize) + "]").c_str(), nvtx_colors::BLUE);

    int32_t const activeBatchSize = context.activeBatchSize;
    // Over the active batch, not the whole vector: a seated single-slot prefill narrows the batch
    // while the per-slot vectors keep their full length, and a resident's longer prompt must not
    // widen the pass.
    int32_t inputIdsLength = *std::max_element(
        context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.begin() + activeBatchSize);
    int32_t const baseOutputHiddenDim
        = mDeployment.specConfig.has_value() ? mDeployment.specConfig->baseOutputHiddenDim : 0;
    StepId raggedStepId{0};
    context.scheduledStep.id = context.nextStepId++;
    context.scheduledStep.sequences.resize(static_cast<size_t>(activeBatchSize));
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        auto& slot = context.slotStreams[i];
        slot.profiler.beginPrefill(slot.channel && slot.channel->getCollectMetrics());
        std::vector<int32_t> const& tokens = context.tokenIds[static_cast<size_t>(i)];
        context.scheduledStep.sequences[static_cast<size_t>(i)] = ScheduledSequence{
            SequenceIdentity{context.requestIds[static_cast<size_t>(i)], context.residentRefs[static_cast<size_t>(i)]},
            SequenceWork::kContext, context.effectivePrefillLengths[static_cast<size_t>(i)],
            context.committedLengths[static_cast<size_t>(i)],
            HostTokenRange{tokens.data(), static_cast<int32_t>(tokens.size()), 0,
                context.effectivePrefillLengths[static_cast<size_t>(i)]},
            true};
    }
    check::check(context.raggedBatchBuilder.has_value(), "Ragged batch builder is not initialized");
    mPipelineIO->waitForStepHostStaging();
    RaggedBatchBuilder::validateRuntimeAdapterStep(
        context.scheduledStep, context.activeBatchSize, context.requestIds, context.residentRefs);
    context.raggedBatchBuilder->buildInto(context.scheduledStep, context.raggedExecutionBatch);
    RaggedExecutionBatch& batch = context.raggedExecutionBatch;
    ELLM_CHECK(batch.pastLengths.size() == static_cast<size_t>(activeBatchSize)
            && context.prefillStartLengths.size() >= batch.pastLengths.size(),
        "Ragged prefill lengths do not match the active execution rows.");
    std::copy(batch.pastLengths.begin(), batch.pastLengths.end(), context.prefillStartLengths.begin());
    raggedStepId = context.scheduledStep.id;

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
        check::check(mPipelineIO->phaseIsEncoder.reshape({1}), "Tensor reshape failed");
        check::check(mPipelineIO->hostPhaseIsEncoder.reshape({1}), "Tensor reshape failed");
        check::check(mPipelineIO->contextMaskSelector.reshape({0}), "Tensor reshape failed");
        int32_t* hostPhase = mPipelineIO->hostPhaseIsEncoder.dataPointer<int32_t>();
        hostPhase[0] = 1;
        CUDA_CHECK(cudaMemcpyAsync(mPipelineIO->phaseIsEncoder.rawPointer(), hostPhase, sizeof(int32_t),
            cudaMemcpyHostToDevice, context.stream));
        mPipelineIO->recordStepHostUploads(context.stream);
    }
    else
    {
        check::check(mPipelineIO->outputLogits.reshape({activeBatchSize, mDeployment.base.outputVocabSize}),
            "Tensor reshape failed");
    }
    // Populate host-side context lengths and upload the builder-owned pinned token snapshot.
    int32_t* hostCtxLenData = mPipelineIO->hostContextLengths.dataPointer<int32_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        int32_t const requestedSeqLen = context.effectivePrefillLengths[i];
        ELLM_CHECK(requestedSeqLen >= 0 && requestedSeqLen <= inputIdsLength,
            "Effective prefill length must be within the current input sequence");
        hostCtxLenData[i] = requestedSeqLen;
    }

    CUDA_CHECK(cudaMemcpyAsync(mIdsInput.rawPointer(), batch.hostTokenIds.rawPointer(),
        activeBatchSize * inputIdsLength * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    mPipelineIO->recordStepHostUploads(context.stream);

    bool const executionKVAllEmpty
        = std::all_of(batch.pastLengths.begin(), batch.pastLengths.end(), [](int32_t length) { return length == 0; });
    if (mDeployment.base.useVisionBidirectionalAttention)
    {
        // Vision-block attention accepts only request-local initial prefill;
        // other resident requests may already own KV pages.
        if (std::any_of(context.committedLengths.begin(), context.committedLengths.begin() + activeBatchSize,
                [](int32_t length) { return length != 0; }))
        {
            LOG_ERROR(
                "Gemma4 vision bidirectional attention does not support request-local prefix reuse or chunked "
                "prefill.");
            return false;
        }
        check::check(mPipelineIO->visionBlockIds.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
        kernel::generateVisionBlockIds(
            mIdsInput, mPipelineIO->visionBlockIds, mDeployment.base.imageTokenId, context.stream);
    }

    // Embedding lookup (text / vision / audio-multimodal) into mPipelineIO->inputsEmbeds;
    // deepstack slots are populated from features or zero-filled depending on the request.
    //
    // When context-cache reuse is active, tokenIds are the suffix (prefix tokens are served from KV cache).
    // The ViT output concatenates ALL batch elements' embeddings sequentially, so per-row offsets are needed
    // to index past earlier sequences' embeddings that were consumed by the cached prefix.
    //
    // Multimodal indices are computed from the builder-owned pinned token snapshot and uploaded once.
    rt::OptionalInputTensor precomputedIndices = std::nullopt;
    if (context.visualEmbeddings.has_value() || context.audioEmbeddings.has_value())
    {
        int32_t const imageTokenId = mDeployment.base.imageTokenId;
        int32_t const audioTokenId = mDeployment.base.audioTokenId;
        int32_t const* packedTokenIds = batch.hostTokenIds.dataPointer<int32_t>();

        check::check(mMultimodalIndices.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
        check::check(mHostMultimodalIndices.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
        int32_t* hostIndices = mHostMultimodalIndices.dataPointer<int32_t>();

        if (managedKVCacheRequest != nullptr && managedKVCacheRequest->hasContextReuse())
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
                int32_t const* rowTokens = packedTokenIds + static_cast<int64_t>(i) * inputIdsLength;
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
                int32_t const* rowTokens = packedTokenIds + static_cast<int64_t>(i) * inputIdsLength;
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
        mPipelineIO->recordStepHostUploads(context.stream);
        precomputedIndices = rt::OptionalInputTensor{mMultimodalIndices};
    }
    mEmbeddingPre->embed(
        mIdsInput, context.visualEmbeddings, context.audioEmbeddings, *mPipelineIO, context.stream, precomputedIndices);

    mEmbeddingPre->prepareDeepstack(mIdsInput, context.deepstackFeatures, *mPipelineIO, context.stream);
    if (mQwen4ExpNgram)
    {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            // The hash reaches back past this call's first token, into a reused prefix or an earlier chunk.
            // Both are still in the request; the batch's past length says where this call starts.
            auto const& requestTokens = context.rawBatchedInputIds[i];
            mQwen4ExpNgram->seedHistory(batch.stateIndices[static_cast<size_t>(i)], requestTokens.data(),
                batch.pastLengths[static_cast<size_t>(i)]);
        }
    }
    // Visual-token pruning compacts the assembled embeddings before the engine runs, pruning each
    // request in the batch independently. Bounded SWA context records use the raw token position as
    // their logical endpoint, so keep that endpoint aligned with the executed prefill until pruned
    // media spans can be represented in the cached SWA state.
    bool const boundedSwaContextReuse = managedKVCacheRequest != nullptr && managedKVCacheRequest->hasContextReuse()
        && managedKVCacheRequest->hasBoundedSwa();
    // The coordinator currently rejects multi-device pruning, and setVisualPrunerConfig excludes action execution.
    if (mVisualPruner && !boundedSwaContextReuse && executionKVAllEmpty && !mDeployment.base.isDiffusionBackbone
        && !context.outputThinkerEmbeddings && context.layerDebugger == nullptr)
    {
        // The pruner reports one count per row it executed, and it sizes its output to that. The
        // context's per-slot vector is wider than that under a seated prefill: the batch-one
        // projection executes one row of a batch that keeps every other slot's entry, and the
        // projection's restore swaps the executed row's entry back into its slot. So the pruner
        // writes its own vector, and only the executed rows are copied into the context's, which
        // grows to the batch on the first prune and is never shrunk by one.
        std::vector<int32_t> prunedTokensPerRow;
        int32_t const newMaxLen = mVisualPruner->pruneBatchForPrefill(context.tokenIds, *mPipelineIO,
            context.effectivePrefillLengths, inputIdsLength, prunedTokensPerRow, context.stream);
        ELLM_CHECK(static_cast<int32_t>(prunedTokensPerRow.size()) == activeBatchSize,
            "Visual-token pruning reported a different number of rows than the prefill executes.");
        if (context.prunedPrefillTokens.size() < static_cast<size_t>(activeBatchSize))
        {
            context.prunedPrefillTokens.resize(static_cast<size_t>(activeBatchSize), 0);
        }
        std::copy(prunedTokensPerRow.begin(), prunedTokensPerRow.end(), context.prunedPrefillTokens.begin());
        bool const anyPruned
            = std::any_of(prunedTokensPerRow.begin(), prunedTokensPerRow.end(), [](int32_t n) { return n > 0; });
        if (anyPruned)
        {
            LOG_DEBUG("Visual-token pruning (%s) shortened prefill from %d to %d tokens.", mVisualPruner->name(),
                inputIdsLength, newMaxLen);
            for (int32_t i = 0; i < activeBatchSize; ++i)
            {
                hostCtxLenData[i] = context.effectivePrefillLengths[i];
            }
            // The finalizer reuses the pinned token source after its H2D snapshot, so wait only for that upload.
            mPipelineIO->waitForStepHostStaging();
            if (mDeployment.specConfig.has_value())
            {
                // Spec-decode draft prefill re-embeds the prompt from host token ids and the raw
                // visual feature rows; compact them to the same keep lists so the draft sees
                // exactly the pruned sequence the base KV cache holds.
                mVisualPruner->compactAuxiliaryInputs(context.tokenIds, activeBatchSize, context.prunedPrefillTokens,
                    context.visualEmbeddings, context.stream);
            }
            context.raggedBatchBuilder->finalizeSubsetSelection(context.scheduledStep,
                mVisualPruner->keepStartOffsets(), mVisualPruner->concatenatedKeepIndices(), batch);
            inputIdsLength = batch.shape.queryWidth;
            // The step record describes what runs, not what was planned: a scheduler's ledger
            // advances by the query length reported back from it. Rewritten only now, after the
            // finalizer checked the step against the provisional batch it was built from.
            for (int32_t i = 0; i < activeBatchSize; ++i)
            {
                context.scheduledStep.sequences[static_cast<size_t>(i)].queryLength
                    = context.effectivePrefillLengths[i];
            }
        }
    }

    // Bounded SWA admission reserves pages by batch size, but its sparse mappings must use the final executed lengths.
    // Visual pruning can shorten those lengths after embeddings are assembled, so bind the window only now.
    if (managedKVCacheRequest != nullptr && !managedKVCacheRequest->prepareSwaPrefill(batch.queryLengths))
    {
        return false;
    }

    mPipelineIO->uploadRaggedMetadata(batch, context.stream);
    if (mGemma4Ple)
    {
        mGemma4Ple->embed(
            mIdsInput, context.stream, batch.queryLengths, &mPipelineIO->queryLengths, &batch.hostTokenIds);
    }
    if (mQwen4ExpNgram)
    {
        mQwen4ExpNgram->embed(batch.hostTokenIds.dataPointer<int32_t>(), hostCtxLenData, batch.stateIndices.data(),
            batch.shape.numSequences, batch.shape.queryWidth, context.stream);
    }
    mSharedResources->cacheManagers[0]->materializeExecutionLengths(mPipelineIO->pastLengths, context.stream);
    if (mDeployment.base.ropeConfig.type == RopeType::kMRope)
    {
        if (!context.visualEmbeddings.has_value() && !context.audioEmbeddings.has_value())
        {
            prepareTextOnlyMRope(*mPipelineIO, mDeployment.base, batch.shape.numSequences, context.stream);
        }
        scatterActiveMRopeToResident(*mPipelineIO, batch, mDeployment.base, context.stream);
    }
    prepareRaggedKVPageTable(
        *mPipelineIO, *mSharedResources->kvPageTables[0], batch.shape.numSequences, context.stream);
    if (mDeployment.base.usesBoundedSwaKVCache())
    {
        KVPageTable* const swaPageTable = mSharedResources->getSwaKVPageTable(0);
        check::check(swaPageTable != nullptr, "Bounded SWA page table is missing.");
        prepareRaggedSwaKVPageTable(*mPipelineIO, *swaPageTable, batch.shape.numSequences, context.stream);
    }
    prepareRaggedRope(*mPipelineIO, *mSharedResources, mDeployment.base, batch.shape.physicalTokens,
        batch.shape.numSequences, context.stream);
    check::check(mPipelineIO->inputsEmbeds.reshape({batch.shape.physicalTokens, mDeployment.base.hiddenSize}),
        "Ragged input embedding reshape failed");
    if (mDeployment.specConfig.has_value())
    {
        check::check(mPipelineIO->baseHiddenStates.reshape({batch.shape.physicalTokens, baseOutputHiddenDim}),
            "Ragged base hidden-state output reshape failed");
    }
    check::check(mPipelineIO->outputHiddenStates.isEmpty()
            || mPipelineIO->outputHiddenStates.reshape({batch.shape.physicalTokens, mDeployment.base.hiddenSize}),
        "Ragged hidden-state output reshape failed");
    for (Tensor& deepstack : mPipelineIO->deepstackEmbeds)
    {
        check::check(deepstack.reshape({batch.shape.physicalTokens, mDeployment.base.hiddenSize}),
            "Ragged deepstack reshape failed");
    }
    if (!mPipelineIO->visionBlockIds.isEmpty())
    {
        check::check(
            mPipelineIO->visionBlockIds.reshape({batch.shape.physicalTokens}), "Ragged vision-block ID reshape failed");
    }

    // Dispatch per-step sequence prep (context lengths H2D, selectTokenIndices).
    mStepPreparer->prepare(
        InferencePhase::kPrefill, activeBatchSize, *mSharedResources->cacheManagers[0], *mPipelineIO, context.stream);
    mPipelineIO->recordStepHostUploads(context.stream);
    // Bind real deepstack features for this prefill (no-op when feature absent).
    if (mDeepstack)
    {
        mDeepstack->useRealFeatures(mBaseTensorMap);
    }

    ExecutionPhase const prefillPhase = decoder_utils::contextPrefillPhase(batch.pastLengths, batch.shape.numSequences);
    auto const prefillDims = mDeployment.base.prefillDims(activeBatchSize, inputIdsLength,
        mDeployment.base.isDiffusionBackbone ? ExecutionPhase::kDiffusionCommit : prefillPhase);
    {
        auto forward
            = gTimer.startForward("base", prefillDims.batch, prefillDims.seqLen, prefillDims.selectLen, context.stream);
        check::check(mBaseExecutor->prepare(kPrefillProfile, prefillDims, mBaseTensorMap, context.stream),
            "Failed to prepare base model for prefill step.");
        check::check(mBaseExecutor->execute(context.stream), "Failed to execute base model for prefill step.");
    }
    context.completionSequences.resize(static_cast<size_t>(activeBatchSize));
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        context.completionSequences[static_cast<size_t>(i)]
            = CompletionSequence{context.requestIds[static_cast<size_t>(i)],
                context.residentRefs[static_cast<size_t>(i)], context.committedLengths[static_cast<size_t>(i)]};
    }
    context.raggedExecutionBatch.validateCommitSnapshot(raggedStepId, context.completionSequences);
    mSharedResources->cacheManagers[0]->commitSequenceLength(mPipelineIO->contextLengths, context.stream);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        context.committedLengths[static_cast<size_t>(i)] += context.effectivePrefillLengths[static_cast<size_t>(i)];
    }
    bool const captureReady = managedKVCacheRequest == nullptr
        || (admissionSequenceIndex >= 0 ? managedKVCacheRequest->enqueueSequenceAdmissionCapture(admissionSequenceIndex)
                                        : managedKVCacheRequest->enqueuePrefillCaptures());
    if (!captureReady)
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

    if (context.hasGuidedDecoding)
    {
        // Prefill emits one logits row per request. The matcher never consumes prompt tokens,
        // so prompt chunking is irrelevant to it.
        applyGuidedDecodingMask(
            mGuidedDecoder, context, mPipelineIO->outputLogits, activeBatchSize, /*rowsPerSlot=*/1, context.stream);
    }

    // Sampling from the prefill stage logits follows the same policy as vanilla decoding.
    // DFlash2 and DSpark keep non-greedy params; other speculative decoders are
    // normalized to greedy before decoding.
    decoder_utils::sampleNextTokens(context, activeBatchSize, mPipelineIO->outputLogits, mSamplingIndices,
        mHostSamplingRows, mSamplingRows, mSamplingWorkspace, context.stream);

    // Matchers live in the output vocabulary, but the remap below rewrites mSamplingIndices
    // into full token IDs, so capture them first. Rides the round's single sync.
    if (context.hasGuidedDecoding)
    {
        check::check(mHostOutputSpaceIds.reshape({activeBatchSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mHostOutputSpaceIds.dataPointer<int32_t>(), mSamplingIndices.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
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
    // copies ride the same sync as the sampled-token D2H below. Extracted at the widest width
    // any slot asks for; each slot keeps its own.
    int32_t const logprobsWidth = context.maxNumLogprobs();
    if (logprobsWidth > 0)
    {
        decoder_utils::enqueueLogprobsD2H(mDecodingRuntimeContext->base.pipelineIO.outputLogits, activeBatchSize,
            *mDecodingRuntimeContext, logprobsWidth, context.stream);
    }

    check::check(mHostSelectedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostSelectedTokenIdsData = mHostSelectedTokenIds.dataPointer<int32_t>();
    CUDA_CHECK(cudaMemcpyAsync(hostSelectedTokenIdsData, mSamplingIndices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));
    if (managedKVCacheRequest != nullptr && admissionSequenceIndex >= 0
        && !managedKVCacheRequest->completeSequenceAdmissionDeviceWork(admissionSequenceIndex))
    {
        return false;
    }

    // Few-layer-validation debug: dump round 0 (prefill). At this point the KV cache is committed and
    // tokenIds[i].size() == the prefill length == the committed cache length.
    if (context.layerDebugger != nullptr)
    {
        std::vector<int32_t> validLengths(activeBatchSize);
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            validLengths[i] = static_cast<int32_t>(context.tokenIds[i].size());
        }
        // Context reuse executes only the suffix after the cached prefix, so tokenIds counts
        // fewer tokens than the cache holds; record the difference once for every later round.
        // Keyed by original request row -- here they still coincide, since prefill runs before
        // any sequence can finish and the batch can be compacted.
        std::vector<int32_t> reusedPrefix(activeBatchSize);
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            reusedPrefix[i] = static_cast<int32_t>(context.rawBatchedInputIds[i].size()) - validLengths[i];
        }
        context.layerDebugger->setReusedPrefixLengths(std::move(reusedPrefix));

        context.layerDebugger->dumpRound(*mSharedResources->cacheManagers[0], *mSharedResources->kvPageTables[0],
            mSharedResources->getSwaKVPageTable(0), mPipelineIO->outputLogits, validLengths, context.batchIndexMapping,
            context.residentRefs, hostSelectedTokenIdsData, activeBatchSize, context.stream);

        // Teacher-forcing: feed the golden tokens instead of sampled tokens when configured.
        context.layerDebugger->applyForcedTokens(
            context.currentGenerateLengths, context.batchIndexMapping, hostSelectedTokenIdsData, activeBatchSize);
    }

    decoder_utils::appendSampledTokens(context, hostSelectedTokenIdsData, activeBatchSize);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        context.slotStreams[i].profiler.finishPrefill(context.currentGenerateLengths[i]);
    }

    if (context.hasGuidedDecoding)
    {
        // Runs before updateThinkingDone(), which is what keeps `</think>` itself out of the
        // grammar: on the step that emits it the slot still counts as thinking.
        advanceGuidedDecoding(mGuidedDecoder, context, mHostOutputSpaceIds.dataPointer<int32_t>(), activeBatchSize);
    }

    if (logprobsWidth > 0)
    {
        decoder_utils::collectLogprobsFromHost(*mDecodingRuntimeContext, context, activeBatchSize, logprobsWidth);
    }

    emitTokenCallbacks(context);
    return true;
}

bool LLMRankRuntime::captureBaseGraphWithLoraFanout(InferenceDims const& dims, cudaStream_t stream)
{
    TIME_STAGE("graph_capture", stream);
    if (mGemma4Ple)
    {
        mGemma4Ple->reshapeOutputsTokenMajor(dims.seqLen);
    }
    if (mQwen4ExpNgram)
    {
        mQwen4ExpNgram->reshapeOutput(dims.seqLen);
    }
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
    if (!unavailableReason().empty())
    {
        return false;
    }
    selectProfilingTimer();
    try
    {
        TIME_STAGE("graph_capture", stream);
        if (!mDecoderRegistry || mDecoderRegistry->captureCudaGraphs(stream))
        {
            return true;
        }
    }
    catch (std::exception const& e)
    {
        LOG_WARNING("CUDA graph capture failed with exception: %s", e.what());
        static_cast<void>(cudaGetLastError());
    }
    catch (...)
    {
        LOG_WARNING("CUDA graph capture failed with an unknown exception.");
        static_cast<void>(cudaGetLastError());
    }
    mDeviceFailure.drainFailedRequest(stream);
    return false;
}

std::vector<int32_t> LLMRankRuntime::mediaTokenIds() const
{
    std::vector<int32_t> ids;
    if (mDeployment.base.imageTokenId >= 0)
    {
        ids.push_back(mDeployment.base.imageTokenId);
    }
    if (mDeployment.base.audioTokenId >= 0)
    {
        ids.push_back(mDeployment.base.audioTokenId);
    }
    return ids;
}

bool LLMRankRuntime::selectLoraWeights(std::string const& loraWeightsName)
{
    // LoRA switching goes through the LoRAManager on SharedResources.
    if (mDeployment.base.maxSupportedLoraRank <= 0 || !mSharedResources->loraManager)
    {
        return true;
    }
    try
    {
        if (loraWeightsName.empty())
        {
            mSharedResources->loraManager->resetWeights();
        }
        else
        {
            mSharedResources->loraManager->switchWeights(loraWeightsName);
        }
        mSharedResources->loraManager->refreshTensorMap(mBaseTensorMap);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to switch LoRA weights to %s: %s", loraWeightsName.c_str(), e.what());
        return false;
    }
    return true;
}

bool LLMRankRuntime::setUpForPrefillExecution(DecodingInferenceContext& context, DecodingStrategy& strategy,
    std::vector<int32_t> const* contextCachePrefillStarts)
{
    NVTX_SCOPED_RANGE(nvtx_setup, "SETUP_PREFILL_EXECUTION", nvtx_colors::PALE_GREEN);

    if (!selectLoraWeights(context.loraWeightsName))
    {
        return false;
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
            context.committedLengths[i] = prefillStart;
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

        check::check(mHostReuseKVCacheLengths.reshape({activeBatchSize}), "Tensor reshape failed");
        int32_t* reuseKVCacheLengthsData = mHostReuseKVCacheLengths.dataPointer<int32_t>();
        std::fill(reuseKVCacheLengthsData, reuseKVCacheLengthsData + activeBatchSize, 0);

        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            ResidentRef const resident = context.residentRefs.at(static_cast<size_t>(i));
            ELLM_CHECK(context.residentSlots.contains(resident), "Prefill setup received a stale resident slot");
            int32_t const residentSlot = resident.slot;

            context.tokenIds[i] = batchedInputIds[i];
            context.effectivePrefillLengths[i] = static_cast<int32_t>(batchedInputIds[i].size());
            context.committedLengths[i] = 0;
            reuseKVCacheLengthsData[i] = 0;

            if (mDeployment.base.numLinearAttnLayers > 0)
            {
                cacheMgrBase.getMambaCacheManager().clearSlot(residentSlot, context.stream);
            }
            if (cacheMgrBase.numAuxConvStates() > 0)
            {
                cacheMgrBase.clearAuxConvStates(residentSlot, context.stream);
            }
            if (mQwen4ExpNgram)
            {
                mQwen4ExpNgram->resetHistory(residentSlot);
            }
        }
    }

    strategy.beginBatch();

    int32_t const maxInputLength
        = *std::max_element(context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.end());
    if (maxInputLength > mDeployment.base.maxSupportedInputLength)
    {
        LOG_ERROR(
            "Prefill length after context reuse (%d) exceeds engine max input length (%d). "
            "Rebuild the engine with a larger --max-input-len.",
            maxInputLength, mDeployment.base.maxSupportedInputLength);
        throw std::runtime_error("EDGELLM_INPUT_TOO_LONG: prefill length " + std::to_string(maxInputLength)
            + " exceeds engine max_input_len " + std::to_string(mDeployment.base.maxSupportedInputLength)
            + " (rebuild engine with a larger --max-input-len)");
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

bool LLMRankRuntime::performBatchEvict(
    DecodingInferenceContext& context, DecodingStrategy& strategy, ManagedKVCacheRequest* managedKVCacheRequest)
{
    // Cheap (a dozen size comparisons) and worth it at every eviction: growth and compaction must
    // touch the same set of per-slot vectors, and one missed by either shows up as another slot's
    // state after the move. This turns that into an immediate failure at the boundary.
    ELLM_CHECK(context.perSlotSizesConsistent(), "Per-slot vectors disagree on the batch size before eviction.");

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

    bool const managedKVCache = managedKVCacheRequest != nullptr;
    std::vector<ResidentRef> retiredResidents;
    retiredResidents.reserve(evictedIndices.size());
    for (int32_t const index : evictedIndices)
    {
        retiredResidents.push_back(context.residentRefs[static_cast<size_t>(index)]);
    }
    if (managedKVCache)
    {
        if (!managedKVCacheRequest->beginBatchCompaction(batchMapping, newActiveBatch, mDeviceBatchMapping))
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
        size_t const activeCacheCount = activeCacheManagerCount(strategy);
        ELLM_CHECK(activeCacheCount > 0, "Active decoding strategy has no KV cache resource");
        for (size_t cacheIndex = 0; cacheIndex < activeCacheCount; ++cacheIndex)
        {
            auto& cacheManager = *mSharedResources->cacheManagers[cacheIndex];
            cacheManager.compactKVCacheLengths(mDeviceBatchMapping, oldActiveBatch, newActiveBatch, context.stream);
            for (ResidentRef const resident : retiredResidents)
            {
                cacheManager.clearResidentSlot(resident.slot, context.stream);
            }
            cacheManager.setActiveBatchSize(newActiveBatch);
        }
    }

    if (mDeployment.base.ropeConfig.type == RopeType::kMRope && !mPipelineIO->mropeCosSin.isEmpty())
    {
        rt::Tensor& residentRope = mPipelineIO->mropeCosSin;
        ELLM_CHECK(residentRope.getShape().getNumDims() == 3, "Resident MRoPE cache must be rank 3");
        size_t const slotBytes = static_cast<size_t>(residentRope.getShape()[1]) * residentRope.getShape()[2]
            * utils::getTypeSize(residentRope.getDataType());
        for (ResidentRef const resident : retiredResidents)
        {
            ELLM_CHECK(resident.slot >= 0 && resident.slot < residentRope.getShape()[0],
                "Retired MRoPE resident slot is out of range");
            auto* const slot = static_cast<char*>(residentRope.rawPointer()) + resident.slot * slotBytes;
            CUDA_CHECK(cudaMemsetAsync(slot, 0, slotBytes, context.stream));
        }
    }

    strategy.onBatchEvict(batchMapping, oldActiveBatch, newActiveBatch, mDeviceBatchMapping, context.stream);

    // Consume the existing eviction synchronization before releasing retired resident identities.
    if (managedKVCache)
    {
        if (!managedKVCacheRequest->completeBatchCompaction(batchMapping))
        {
            return false;
        }
    }
    else
    {
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
    }
    for (ResidentRef const resident : retiredResidents)
    {
        ELLM_CHECK(context.residentSlots.release(resident), "Retired resident identity is stale");
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
            result.actualIterations = context.decodeRounds[i];
            result.rawBatchedInputIds = std::move(context.rawBatchedInputIds[i]);
            result.effectivePrefillLength = context.effectivePrefillLengths[i];
            result.prunedPrefillTokens = i < context.prunedPrefillTokens.size() ? context.prunedPrefillTokens[i] : 0;
            result.acceptedDraftTokens = context.acceptedDraftTokens[i];
            result.proposedDraftTokens = context.proposedDraftTokens[i];
            result.terminalReason = context.slotStreams[i].terminalReason;
            result.skipSpecialTokens = context.slotStreams[i].skipSpecialTokens;
            // Convert flat LogprobsSlot -> nested vector for BatchResult (once per completed request).
            // Enrich each (token_id, logprob) with the raw token piece so consumers can render the
            // token string / bytes without needing a tokenizer (see LogprobEntry).
            rt::LogprobsSlot const& slot = context.stepLogprobs[i];
            int32_t const width = context.numLogprobsPerSlot[i];
            result.logprobs.resize(slot.numSteps);
            for (int32_t step = 0; step < slot.numSteps; ++step)
            {
                // A forced token's synthetic row holds one entry; the padding behind it is not an entry.
                result.logprobs[step] = logprobEntriesForStep(slot, step, width, *mTokenizer);
            }
            if (context.jumpForwardEnabled[i] != 0)
            {
                LOG_INFO("Request %d jump-forward: %s", originalIdx,
                    formatJumpForwardStats(context.jumpForwardStats[i]).c_str());
            }

            context.completedBatches[originalIdx] = std::move(result);
        }
    }

    rt::compactVector(batchMapping, context.finishedStates);
    rt::compactVector(batchMapping, context.thinkingDone);
    rt::compactVector(batchMapping, context.guidedReasoningEnded);
    // Same slot numbering as the vectors above, so it must be reindexed at the same moment;
    // skipping it migrates a constraint onto a different request.
    mGuidedDecoder.compactSlots(batchMapping);
    context.hasGuidedDecoding = mGuidedDecoder.hasAnyGrammar();
    rt::compactVector(batchMapping, context.jumpForwardEnabled);
    rt::compactVector(batchMapping, context.pendingTokens);
    rt::compactVector(batchMapping, context.jumpForwardStats);
    context.hasJumpForward = std::any_of(context.jumpForwardEnabled.begin(), context.jumpForwardEnabled.end(),
        [](int8_t enabled) { return enabled != 0; });
    rt::compactVector(batchMapping, context.currentGenerateLengths);
    rt::compactVector(batchMapping, context.samplingParams);
    rt::compactVector(batchMapping, context.samplingSeeds);
    rt::compactVector(batchMapping, context.requestStableSampling);
    rt::compactVector(batchMapping, context.maxGenerateLengths);
    rt::compactVector(batchMapping, context.numLogprobsPerSlot);
    rt::compactVector(batchMapping, context.ignoreEos);
    rt::compactVector(batchMapping, context.decodeRounds);
    rt::compactVector(batchMapping, context.tokenIds);
    rt::compactVector(batchMapping, context.systemPrompts);
    rt::compactVector(batchMapping, context.rawBatchedInputIds);
    rt::compactVector(batchMapping, context.effectivePrefillLengths);
    rt::compactVector(batchMapping, context.prefillStartLengths);
    rt::compactVector(batchMapping, context.acceptedDraftTokens);
    rt::compactVector(batchMapping, context.proposedDraftTokens);
    // Conditional vector: sized only while visual-token pruning is active, exactly as appendSlot
    // and swapExecutionRows treat it.
    if (!context.prunedPrefillTokens.empty())
    {
        rt::compactVector(batchMapping, context.prunedPrefillTokens);
    }
    rt::compactVector(batchMapping, context.committedLengths);
    rt::compactVector(batchMapping, context.requestIds);
    rt::compactVector(batchMapping, context.residentRefs);
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
