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

#include "llmInferenceRuntime.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/inputLimits.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/pagedKvTypes.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "kernels/speculative/batchEvictKernels.h"
#include "multimodal/multimodalRunner.h"
#include "multimodal/qwenViTRunner.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/contextCacheRequest.h"
#include "runtime/debug/layerDebugger.h"
#include "runtime/decoding/decoderRegistry.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/llmRuntimeUtils.h"
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
namespace
{
bool needsDFlashDDTreeHybridBindings(DeploymentConfig const& deployment)
{
    return deployment.specConfig.has_value() && deployment.specDecodeMode() == SpecDecodeMode::kDFlash
        && deployment.specConfig->draftingTopK > 1 && deployment.base.numLinearAttnLayers > 0;
}

void validateDFlashTreeMetadataBindings(DeploymentConfig const& deployment, EngineExecutor const& baseExecutor)
{
    if (!deployment.specConfig.has_value() || deployment.specDecodeMode() != SpecDecodeMode::kDFlash)
    {
        return;
    }

    bool const hasTreeParentIds = baseExecutor.hasIOTensor(binding_names::kTreeParentIds);
    bool const hasTreeDepths = baseExecutor.hasIOTensor(binding_names::kTreeDepths);
    bool const hasTreeMetadata = hasTreeParentIds || hasTreeDepths;
    bool const usesDDTree = deployment.specConfig->draftingTopK > 1;
    if (hasTreeMetadata)
    {
        ELLM_CHECK(hasTreeParentIds && hasTreeDepths,
            std::string("DFlash tree-base engine must expose both INT32 tree metadata bindings '")
                + binding_names::kTreeParentIds + "' and '" + binding_names::kTreeDepths + "'.");
        ELLM_CHECK(baseExecutor.getBindingDataType(binding_names::kTreeParentIds) == DataType::kINT32
                && baseExecutor.getBindingDataType(binding_names::kTreeDepths) == DataType::kINT32,
            std::string("DFlash tree-base engine tree metadata bindings must be INT32: '")
                + binding_names::kTreeParentIds + "' and '" + binding_names::kTreeDepths + "'.");
        ELLM_CHECK(usesDDTree,
            std::string("DFlash base engine was exported with --dflash-tree-base, but runtime is configured for "
                        "linear DFlash because specDraftTopK=1. Use --specDraftTopK > 1 for DDTree, or re-export "
                        "the base model with --dflash-base for linear DFlash."));
    }

    if (!needsDFlashDDTreeHybridBindings(deployment))
    {
        return;
    }

    ELLM_CHECK(hasTreeParentIds && hasTreeDepths,
        std::string("DFlash DDTree hybrid base engine requires INT32 tree metadata bindings '")
            + binding_names::kTreeParentIds + "' and '" + binding_names::kTreeDepths
            + "'. Re-export the base model with --dflash-tree-base, then rebuild spec_base.engine.");
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

LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, SpecDecodeDraftingConfig const& draftingConfig,
    cudaStream_t stream, ContextCacheConfig const& contextCacheConfig, std::string const& checkpointDir,
    std::string const& draftCheckpointDir)
{
    initializeCommon(engineDir, multimodalEngineDir, loraWeightsMap, draftingConfig, stream, contextCacheConfig,
        checkpointDir, draftCheckpointDir);
}

LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream,
    ContextCacheConfig const& contextCacheConfig, std::string const& checkpointDir)
{
    initializeCommon(
        engineDir, multimodalEngineDir, loraWeightsMap, std::nullopt, stream, contextCacheConfig, checkpointDir, "");
}

LLMInferenceRuntime::~LLMInferenceRuntime() noexcept
{
    if (mContextCache != nullptr && mContextCache->shutdown() != ContextCacheCoordinatorStatus::kOk)
    {
        LOG_ERROR("Context-cache shutdown could not prove stream quiescence.");
        std::terminate();
    }
}

void LLMInferenceRuntime::initializeCommon(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap,
    std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream,
    ContextCacheConfig const& contextCacheConfig, std::string const& checkpointDir,
    std::string const& draftCheckpointDir)
{
    std::filesystem::path const engineDirPath{engineDir};
    std::filesystem::path const baseConfigPath
        = draftingConfig.has_value() ? engineDirPath / "base_config.json" : engineDirPath / "config.json";
    mCheckpointDir = checkpointDir;
    mDraftCheckpointDir = draftCheckpointDir;

    // Finish checkpoint reads and weight conversion before any engine can run.
    ExternalWeightManager preparedWeights;
    preparedWeights.load(engineDirPath, baseConfigPath, stream, mCheckpointDir);
    if (auto embedding = preparedWeights.takeEmbedding())
    {
        mEmbedding.table = std::move(*embedding);
    }
    else
    {
        mEmbedding = loadEmbeddingTable(engineDirPath / "embedding.safetensors", stream);
    }
    auto pleEmbedding = preparedWeights.takePleEmbedding();

    // -----------------------------------------------------------------------
    // 3. Parse engine configurations and attach user drafting (bundle factory
    //    performs cross-engine consistency and drafting-vs-capacity checks).
    // -----------------------------------------------------------------------
    std::optional<std::filesystem::path> const draftConfigPath = draftingConfig.has_value()
        ? std::optional<std::filesystem::path>{engineDirPath / "draft_config.json"}
        : std::nullopt;

    mDeployment = createDeploymentConfig(baseConfigPath, draftConfigPath, draftingConfig);
    if (draftingConfig.has_value() && mDeployment.specDecodeMode() == SpecDecodeMode::kMTP)
    {
        ELLM_CHECK(mDraftCheckpointDir.empty(),
            "Native MTP draft weights are part of --checkpointDir; do not pass --draftCheckpointDir.");
        mDraftCheckpointDir = mCheckpointDir;
    }
    std::optional<ContextCacheDeploymentKind> contextCacheDeploymentKind;
    if (contextCacheConfig.enabled)
    {
        contextCacheDeploymentKind = validateContextCacheDeployment(mDeployment);
    }

    std::filesystem::path const baseEnginePath = draftingConfig.has_value()
        ? engineDirPath / "spec_base.engine"
        : (mDeployment.base.isDiffusionBackbone ? engineDirPath / "dllm.engine" : engineDirPath / "llm.engine");

    ELLM_CHECK(mDeployment.base.isDiffusionBackbone || mDeployment.base.numDeepstackFeatures <= 0
            || !multimodalEngineDir.empty(),
        "--multimodalEngineDir is required for VLM engine.");

    // -----------------------------------------------------------------------
    // 3. Construct Runners (registries built internally from the parsed configs).
    // -----------------------------------------------------------------------
    try
    {
        std::optional<int32_t> const specDecodeBaseOutputHiddenDim = mDeployment.specConfig.has_value()
            ? std::optional<int32_t>{mDeployment.specConfig->baseOutputHiddenDim}
            : std::nullopt;
        mBaseExecutor = EngineExecutor::createForLLM(baseEnginePath, mDeployment.base, specDecodeBaseOutputHiddenDim);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize base EngineExecutor: %s", e.what());
        throw std::runtime_error("Failed to initialize base EngineExecutor: " + std::string(e.what()));
    }
    LOG_INFO("Base EngineExecutor successfully loaded from %s.", baseEnginePath.c_str());

    // -----------------------------------------------------------------------
    // 4. Validate engine binding dtypes against the parsed configs.
    // -----------------------------------------------------------------------
    validateAgainstEngine(mDeployment.base, *mBaseExecutor, "base");
    validateDFlashTreeMetadataBindings(mDeployment, *mBaseExecutor);
    validateMtpTreeMetadataBindings(mDeployment, *mBaseExecutor);

    // Validate the draft engine ABI before its sidecar geometry is used to allocate
    // physical cache resources. Ownership is transferred to the selected decoder.
    std::unique_ptr<EngineExecutor> draftExecutor;
    if (draftingConfig.has_value())
    {
        draftExecutor = decoder_utils::loadDraftEngine(engineDirPath, mDeployment);
    }
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
        mPipelineIO
            = std::make_unique<PipelineIO>(PipelineIO::createForSpecDecode(mDeployment, mMaxRuntimeBatchSize, stream));
    }
    else
    {
        mSharedResources = SharedResources::createForLLM(mDeployment.base, loraWeightsMap, stream);
        mPipelineIO = std::make_unique<PipelineIO>(PipelineIO::createForLLM(mDeployment.base, stream));
    }
    *mSharedResources->externalWeightManager = std::move(preparedWeights);
    mSharedResources->externalWeightManager->validateAgainstEngine(*mBaseExecutor, "base");

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
        && (mDeployment.specDecodeMode() == SpecDecodeMode::kDFlash
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
            "LLMInferenceRuntime::mIdsInput");

        mSamplingWorkspace = rt::Tensor({maxSamplingWorkspaceLen}, rt::DeviceType::kGPU, DataType::kINT8,
            "LLMInferenceRuntime::mSamplingWorkspace");
        mSamplingIndices = rt::Tensor(
            {maxSamplingSize}, rt::DeviceType::kGPU, DataType::kINT32, "LLMInferenceRuntime::mSamplingIndices");
        mSamplingScores = rt::Tensor(
            {maxSamplingSize}, rt::DeviceType::kGPU, DataType::kFLOAT, "LLMInferenceRuntime::mSamplingScores");
        allocateLogitBias(mLogitBias, mMaxRuntimeBatchSize);

        // Batch mapping tensor for batch eviction.
        mDeviceBatchMapping = rt::Tensor(
            {mMaxRuntimeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32, "LLMInferenceRuntime::mDeviceBatchMapping");

        mHostPackedTokenIds = rt::Tensor({mMaxRuntimeBatchSize, maxInputLength}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMInferenceRuntime::mHostPackedTokenIds");
        mHostSelectedTokenIds = rt::Tensor(
            {maxSamplingSize}, rt::DeviceType::kCPU, DataType::kINT32, "LLMInferenceRuntime::mHostSelectedTokenIds");
        mHostReuseKVCacheLengths = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMInferenceRuntime::mHostReuseKVCacheLengths");

        // Pre-allocate multimodal indices tensor (used for audio/vision embedding lookup).
        mMultimodalIndices = rt::Tensor({mMaxRuntimeBatchSize, maxInputLength}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMInferenceRuntime::mMultimodalIndices");

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
    if (mDeployment.base.reducedVocabSize > 0)
    {
        LOG_INFO("Loading vocabulary mapping table for base model reduced vocab size: %d -> %d",
            mDeployment.base.reducedVocabSize, mDeployment.base.vocabSize);
        std::filesystem::path const vocabMapPath = std::filesystem::path(engineDir) / binding_names::kVocabMapFileName;

        std::vector<rt::Tensor> vocabMapTensors;
        ELLM_CHECK(safetensors::loadSafetensors(vocabMapPath, vocabMapTensors, stream),
            "Failed to load " + std::string(binding_names::kVocabMapFileName) + " from model directory: " + engineDir);

        check::check(vocabMapTensors.size() == 1,
            std::string(binding_names::kVocabMapFileName) + " should contain exactly one tensor");
        check::check(vocabMapTensors[0].getShape().getNumDims() == 1, "vocab_map tensor should be 1D");
        check::check(vocabMapTensors[0].getShape()[0] == mDeployment.base.reducedVocabSize,
            "vocab_map tensor length should match base model reduced vocab size");
        check::check(vocabMapTensors[0].getDataType() == DataType::kINT32, "vocab_map tensor should be INT32");
        mBaseVocabMappingTable = std::move(vocabMapTensors[0]);
        setLogitBiasVocabMap(
            mLogitBias, mBaseVocabMappingTable, mDeployment.base.vocabSize, mDeployment.base.reducedVocabSize, stream);
        LOG_INFO("Base model vocabulary mapping table successfully loaded.");
    }

    // -----------------------------------------------------------------------
    // 12. Tokenizer.
    // -----------------------------------------------------------------------
    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    LOG_INFO("Start loading tokenizer from model directory: %s", engineDir.c_str());
    ELLM_CHECK(mTokenizer->loadFromHF(engineDir), "Failed to load tokenizer from model directory: " + engineDir);
    LOG_INFO("Tokenizer successfully loaded from model directory: %s", engineDir.c_str());

    // Set additional EOS token IDs from parsed config (e.g. Gemma4 has eos_token_id: [1, 106])
    if (!mDeployment.base.eosTokenIds.empty())
    {
        std::vector<tokenizer::Rank> additionalEos(
            mDeployment.base.eosTokenIds.begin(), mDeployment.base.eosTokenIds.end());
        mTokenizer->setAdditionalEosIds(additionalEos);
        LOG_INFO("Loaded %zu EOS token IDs from config", additionalEos.size());
    }

    // -----------------------------------------------------------------------
    // 13. Decoding strategies.
    // -----------------------------------------------------------------------
    buildDecodingRuntimeContext();
    mDecoderRegistry = std::make_unique<DecoderRegistry>(*mDecodingRuntimeContext,
        DecoderRegistryInit{std::filesystem::path(engineDir), draftingConfig, std::move(draftExecutor), stream});

    // -----------------------------------------------------------------------
    // 14. Optional multimodal runners.
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
                mDeployment.base.maxKVCacheCapacity, stream, checkpointDir);
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
            mActionRunner = std::make_unique<Alpamayo1ActionRunner>(actionDir, checkpointDir, stream,
                mSharedResources->cacheManagers[0]->getKVCacheManager().getConfig(),
                mSharedResources->kvPageTables[0]->isIdentity());
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
        ELLM_CHECK(contextCacheDeploymentKind.has_value(), "Context-cache deployment was not validated");
        mContextCache = std::make_unique<ContextCacheCoordinator>(
            contextCacheConfig, mDeployment, *contextCacheDeploymentKind, cacheResources, stream);
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
    mSharedExecContextMemory = rt::Tensor({sharedContextMemorySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "LLMInferenceRuntime::mSharedExecContextMemory");
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
}

void LLMInferenceRuntime::ensureLogprobsCapacity(int32_t logprobsRows, int32_t topK)
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
            DataType::kINT8, "LLMInferenceRuntime::mSamplingWorkspace");
    }

    int64_t const requiredValueBytes = static_cast<int64_t>(logprobsRows) * kMaxLogprobsK * sizeof(float);
    int64_t const requiredIndexBytes = static_cast<int64_t>(logprobsRows) * kMaxLogprobsK * sizeof(int32_t);
    if (mDeviceLogprobsValues.isEmpty() || mDeviceLogprobsValues.getMemoryCapacity() < requiredValueBytes)
    {
        mDeviceLogprobsValues = rt::Tensor({logprobsRows, kMaxLogprobsK}, rt::DeviceType::kGPU, DataType::kFLOAT,
            "LLMInferenceRuntime::mDeviceLogprobsValues");
        mHostLogprobsValues = rt::Tensor({logprobsRows, kMaxLogprobsK}, rt::DeviceType::kCPU, DataType::kFLOAT,
            "LLMInferenceRuntime::mHostLogprobsValues");
    }
    if (mDeviceLogprobsIndices.isEmpty() || mDeviceLogprobsIndices.getMemoryCapacity() < requiredIndexBytes)
    {
        mDeviceLogprobsIndices = rt::Tensor({logprobsRows, kMaxLogprobsK}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMInferenceRuntime::mDeviceLogprobsIndices");
        mHostLogprobsIndices = rt::Tensor({logprobsRows, kMaxLogprobsK}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMInferenceRuntime::mHostLogprobsIndices");
    }
    if (mDeployment.specConfig.has_value())
    {
        int64_t const requiredGatheredBytes
            = static_cast<int64_t>(logprobsRows) * mDeployment.base.outputVocabSize * sizeof(float);
        if (mGatheredLogits.isEmpty() || mGatheredLogits.getMemoryCapacity() < requiredGatheredBytes)
        {
            mGatheredLogits = rt::Tensor({logprobsRows, mDeployment.base.outputVocabSize}, rt::DeviceType::kGPU,
                DataType::kFLOAT, "LLMInferenceRuntime::mGatheredLogits");
        }
    }
    mLogprobsMaxBatchDim = std::max(mLogprobsMaxBatchDim, logprobsRows);
}

void LLMInferenceRuntime::buildDecodingRuntimeContext()
{
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

void LLMInferenceRuntime::setActionNoiseSeed(int32_t seed) noexcept
{
    if (mActionRunner)
    {
        mActionRunner->setNoiseSeed(seed);
    }
}

std::optional<ContextCacheMetrics> LLMInferenceRuntime::getContextCacheMetrics() const noexcept
{
    if (mContextCache == nullptr)
    {
        return std::nullopt;
    }
    return mContextCache->metrics();
}

bool LLMInferenceRuntime::handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response,
    cudaStream_t stream, bool outputThinkerEmbeddings)
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

    // Clear per-request portal state. Buffers themselves stay allocated and are
    // reshaped/overwritten when populated below — see getBaseModelHiddenStates() contract.
    mHiddenStatesRegistry.clear();
    mLastPrefillLength = 0;
    mLastInputTokenIds.clear();

    // Clear per-request response state. On failure (early return) the four vectors
    // stay empty; on success they are repopulated together below to matched sizes.
    response.outputIds.clear();
    response.outputTexts.clear();
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
    if (shouldRejectLogitBiasWithSpecDecode(request, enableSpecDecode))
    {
        LOG_ERROR(
            "logit_bias is not supported while speculative decoding is enabled; set disable_spec_decode=true or use "
            "a vanilla decoding strategy.");
        return false;
    }

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

    // Apply chat template for all requests (common for both multimodal and non-multimodal)
    request.formattedRequests.resize(activeBatchSize);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // Apply chat template to populate both formatted system prompt and full formatted prompt
        mTokenizer->applyChatTemplate(request.requests[i], request.formattedRequests[i], request.applyChatTemplate,
            request.addGenerationPrompt, request.enableThinking);
    }

    DecodingInferenceContext context;
    context.initialize(
        activeBatchSize, maxGenerateLength, std::nullopt, rt::OptionalInputTensors{}, loraWeightsName, stream);

    // Few-layer-validation debug: per-layer logits/KV dump + optional teacher-forcing (both no-ops
    // unless the env vars are set). Owned by the context via RAII so it shares the request's lifetime
    // exactly; prefill and the vanilla decode loop dump rounds through context.layerDebugger.
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
            context.rawBatchedInputIds.emplace_back(
                mTokenizer->encode(request.formattedRequests[i].formattedCompleteRequest, false));
            if (context.rawBatchedInputIds[i].empty())
            {
                LOG_ERROR("Failed to tokenize input text for request %d in batch", i);
                return false;
            }
        }
    }

    // Guard: reject inputs longer than the engine's built max input length with a clear,
    // distinguishable error instead of failing opaquely downstream (TRT profile/shape error).
    // The Python server maps the EDGELLM_INPUT_TOO_LONG marker to HTTP 413.
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

    int32_t const kvCacheCapacity = enableSpecDecode
        ? std::min(mDeployment.base.maxKVCacheCapacity, mDeployment.draft->maxKVCacheCapacity)
        : mDeployment.base.maxKVCacheCapacity;
    int32_t kvcReserve = 0;
    if (enableSpecDecode)
    {
        // Preserve the historical reserve for unmanaged speculative strategies. Managed EAGLE preflights exact
        // base-verification and draft-proposal working sets, so its admission clamp must use the same geometry.
        constexpr int32_t kLEGACY_SPEC_KV_CACHE_RESERVE{100};
        kvcReserve = kLEGACY_SPEC_KV_CACHE_RESERVE;
        if (mContextCache != nullptr && decodingStrategy.kind() == DecodingStrategyKind::kEAGLE)
        {
            ELLM_CHECK(mDeployment.specConfig.has_value(), "EAGLE decoding requires speculative configuration");
            int64_t const draftWorkingTokens = static_cast<int64_t>(mDeployment.specConfig->draftingStep)
                * static_cast<int64_t>(mDeployment.specConfig->draftingTopK);
            int64_t const reserve = std::max<int64_t>(mDeployment.specConfig->verifySize, draftWorkingTokens);
            ELLM_CHECK(reserve > 0 && reserve <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                "EAGLE KV-cache working-set reserve exceeds int32");
            kvcReserve = static_cast<int32_t>(reserve);
        }
        else if (decodingStrategy.kind() == DecodingStrategyKind::kDSpark)
        {
            ELLM_CHECK(mDeployment.specConfig.has_value(), "DSpark decoding requires speculative configuration");
            kvcReserve = mDeployment.specConfig->verifySize;
        }
    }

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
        std::optional<ContextCacheRequest> admitted
            = ContextCacheRequest::begin(*mContextCache, request, context, decodingStrategy.kind(), mediaTokenIds);
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

    std::vector<int32_t> contextCacheResidentInputLengths;
    std::vector<int32_t> const* capacityInputLengths = &context.effectivePrefillLengths;
    if (managedRequest != nullptr)
    {
        contextCacheResidentInputLengths.reserve(context.rawBatchedInputIds.size());
        for (std::vector<int32_t> const& tokenIds : context.rawBatchedInputIds)
        {
            contextCacheResidentInputLengths.push_back(static_cast<int32_t>(tokenIds.size()));
        }
        capacityInputLengths = &contextCacheResidentInputLengths;
    }
    int32_t const clampedMaxGenerateLength = clampMaxGenerateLengthForKVCapacity(
        *capacityInputLengths, request.maxGenerateLength, kvCacheCapacity, kvcReserve);
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

    // Prefill from the base model; subsequent iterations are delegated to the selected strategy.
    bool const prefillStatus = runBaseModelPrefill(context, managedRequest);
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
        int32_t const prefillSequenceLength
            = *std::max_element(context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.end());
        mPipelineIO->streamingPrefill.populateFromPrefill(mPipelineIO->inputsEmbeds, mPipelineIO->outputHiddenStates,
            activeBatchSize, prefillSequenceLength, mDeployment.base.hiddenSize, mDeployment.base.outputHiddenSize,
            mMaxRuntimeBatchSize, mDeployment.base.maxSupportedInputLength, stream);
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

    // Per-slot tracking: once thinking is complete (end marker emitted or model
    // never entered thinking), secondary EOS tokens terminate generation normally.
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
            if (context.tokenIds[i].empty())
            {
                continue;
            }
            updateThinkingDoneForToken(i, context.tokenIds[i].back());
        }
    };

    // Few-layer-validation / fixed-output perf: when EDGELLM_IGNORE_EOS is set,
    // suppress EOS-based termination so the run produces exactly maxGenerateLength
    // tokens. This also applies to multi-token accept paths such as DiffusionGemma
    // canvas commit; otherwise EOS inside a canvas can truncate a fixed-output block.
    bool const ignoreEos = []() {
        char const* v = std::getenv("EDGELLM_IGNORE_EOS");
        return v != nullptr && std::string(v) != "0" && std::string(v) != "false";
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
            if (mActionRunner && mActionRunner->getModelType() == action::ActionModelType::ALPAMAYO1)
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
                    auto lastToken = context.tokenIds[i].back();
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

    // Post-prefill per-iter pipeline:
    //   cancel -> decode (emitDelta + stop match) -> finalize (EOS/length/stop) -> emit.
    // DiffusionGemma prefill writes prompt KV only and does not produce a generated token.
    if (!mDeployment.base.isDiffusionBackbone)
    {
        applyCancellationToFinishStates(context);
        decodePerSlot(context, *mTokenizer);

        updateThinkingDone();

        updateFinishStates();
        emitChunks(context, *mTokenizer);

        // Managed vanilla and EAGLE requests may remove individual slots that finished on the prefill token before
        // decoding starts. Unmanaged requests and other speculative strategies retain the legacy all-finished-only
        // path; their first-round state is outside this integration's partial-compaction contract.
        bool const supportsPartialPrefillEviction = managedRequest != nullptr
            && (decodingStrategy.kind() == DecodingStrategyKind::kVanilla
                || decodingStrategy.kind() == DecodingStrategyKind::kEAGLE);
        if (context.activeBatchSize > 0 && (supportsPartialPrefillEviction || checkAllFinished()))
        {
            bool const batchEvictStatus = performBatchEvict(context, decodingStrategy, thinkingDone, managedRequest);
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

        if (managedRequest != nullptr && !managedRequest->prepareDecodeStep(context))
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

        // Update thinking-done state: check if the last generated token is an
        // end-of-thinking marker (<channel|> for Gemma4, </think> for Qwen3/Nemotron).
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
        bool const batchEvictStatus = performBatchEvict(context, decodingStrategy, thinkingDone, managedRequest);
        if (!batchEvictStatus)
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
    int32_t totalGeneratedTokens = 0;
    int32_t totalIterations = 0;

    // Accumulate from completed batches
    for (auto const& [originalIdx, batchResult] : context.completedBatches)
    {
        int32_t rawPromptLength = static_cast<int32_t>(batchResult.rawBatchedInputIds.size());
        int32_t computedLength = batchResult.effectivePrefillLength;
        totalReusedTokens += (rawPromptLength - computedLength);
        totalComputedTokens += computedLength;
        totalGeneratedTokens += batchResult.generateLength;
        totalIterations += batchResult.actualIterations;
    }

    mPrefillMetrics.recordRun(totalReusedTokens, totalComputedTokens);
    if (enableSpecDecode)
    {
        mSpecDecodeGenerationMetrics.recordRun(totalIterations, totalGeneratedTokens);
    }
    else
    {
        mGenerationMetrics.recordRun(totalGeneratedTokens);
    }

    // Save output ids, decoded texts, and logprobs to response.
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
        response.finishReasons[originalIdx] = batchResult.terminalReason;
        response.logprobs[originalIdx] = batchResult.logprobs;
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

    bool const hasTrajectoryHistory = std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& req) { return req.pastTrajectory.has_value(); });
    // If action engine is loaded, run one batched trajectory sample and fill output for all batch items.
    if (hasTrajectoryHistory && mActionRunner && mActionRunner->getModelType() == action::ActionModelType::ALPAMAYO1)
    {
        if (!mVisionRunner)
        {
            LOG_ERROR("Alpamayo1ActionRunner requires a vision runner (e.g. QwenViTRunner) for MRoPE rope deltas.");
            return false;
        }

        multimodal::ModelType const visionType = mVisionRunner->getModelType();
        bool const isQwen3ViT = visionType == multimodal::ModelType::QWEN3_VL;
        if (!isQwen3ViT)
        {
            LOG_ERROR(
                "Alpamayo1ActionRunner requires a Qwen3-VL vision runner but a different vision runner is loaded.");
            return false;
        }
        // The Qwen3-VL runner is a Qwen3VLViTRunner (derives from QwenViTRunner); upcast to read the base rope deltas.
        auto* qwenVision = static_cast<rt::QwenViTRunner*>(mVisionRunner.get());
        std::vector<int64_t> const& ropeDeltas = qwenVision->getMropeRopeDeltasPerBatch();
        rt::HybridCacheManager& kvcache = *mSharedResources->cacheManagers[0];
        std::vector<std::vector<rt::FutureTrajectoryPoint>> trajectories
            = mActionRunner->sampleTrajectory(stream, activeBatchSize, kvcache, ropeDeltas);
        if (trajectories.size() != static_cast<size_t>(activeBatchSize))
        {
            LOG_ERROR("Alpamayo1ActionRunner trajectory sampling failed.");
            return false;
        }
        for (size_t i = 0; i < trajectories.size() && i < static_cast<size_t>(activeBatchSize); ++i)
        {
            if (!trajectories[i].empty())
            {
                response.outputTrajectories[i] = std::move(trajectories[i]);
            }
        }
    }

    return true;
}

bool LLMInferenceRuntime::validateRequestConfig(LLMGenerationRequest const& request)
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

bool LLMInferenceRuntime::multiModalRuntimePreprocess(
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

    // MRope cos/sin output cache is supplied only for MRope-based runners (QwenViT, Qwen3OmniAudio).
    // Runners with standard RoPE (InternViT, Phi4MMViT) ignore it; see MultimodalRunner::preprocess.
    rt::OptionalOutputTensor mropeCosSinOut = (mDeployment.base.ropeConfig.type == RopeType::kMRope)
        ? rt::OptionalOutputTensor{std::ref(mPipelineIO->mropeCosSin)}
        : std::nullopt;

    // Process audio inputs (if present)
    if (hasAudio && mAudioRunner)
    {
        LOG_INFO("Processing audio inputs");
        if (!mAudioRunner->preprocess(request, batchedInputIds, mTokenizer.get(), mropeCosSinOut, stream))
        {
            LOG_ERROR("Audio preprocessing failed. This request cannot be handled.");
            return false;
        }

        if (!mAudioRunner->infer(stream))
        {
            LOG_ERROR("Audio inference failed. This request cannot be handled.");
            return false;
        }
    }

    // Process vision inputs (if present)
    if (hasVision && mVisionRunner)
    {
        LOG_INFO("Processing vision inputs");
        if (!mVisionRunner->preprocess(request, batchedInputIds, mTokenizer.get(), mropeCosSinOut, stream))
        {
            LOG_ERROR("Vision preprocessing failed. This request cannot be handled.");
            return false;
        }

        if (!mVisionRunner->infer(stream))
        {
            LOG_ERROR("Vision inference failed. This request cannot be handled.");
            return false;
        }
    }

    // Process action inputs (if present)
    if (hasTrajectoryHistory && mActionRunner)
    {
        LOG_INFO("Processing trajectory history inputs");
        if (!mActionRunner->preprocess(request, batchedInputIds, mTokenizer.get()))
        {
            LOG_ERROR(
                "LLMInferenceRuntime(): Trajectory history preprocessing failed. This request cannot be handled.");
            return false;
        }
    }

    if (!hasAudio && !hasVision)
    {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            batchedInputIds.push_back(mTokenizer->encode(request.formattedRequests[i].formattedCompleteRequest, false));
            if (batchedInputIds.back().empty())
            {
                LOG_ERROR("Failed to tokenize input text for request %d in batch", i);
                return false;
            }
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

bool LLMInferenceRuntime::runBaseModelPrefill(
    DecodingInferenceContext& context, ContextCacheRequest* contextCacheRequest)
{
    TIME_STAGE(metrics::StageNames::kLLM_PREFILL, context.stream);
    NVTX_SCOPED_RANGE(nvtx_base_prefill,
        ("SPEC_DECODE_BASE_PREFILL[" + std::to_string(context.activeBatchSize) + "]").c_str(), nvtx_colors::BLUE);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const inputIdsLength
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
    mEmbeddingPre->embed(mIdsInput, context.visualEmbeddings, context.audioEmbeddings, *mPipelineIO, context.stream);
    mEmbeddingPre->prepareDeepstack(mIdsInput, context.deepstackFeatures, *mPipelineIO, context.stream);
    if (mGemma4Ple)
    {
        mGemma4Ple->embed(mIdsInput, context.stream);
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

    applyLogitBias(mLogitBias, mPipelineIO->outputLogits, context, context.stream);

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

        // Teacher-forcing — feed the golden's tokens instead of our own sampled ones (no-op unless
        // EDGELLM_FORCE_TOKENS_FILE is set). Applied after the dump so the dump still records what we
        // *would* have sampled; the pushed token below is the forced one.
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

bool LLMInferenceRuntime::captureBaseGraphWithLoraFanout(InferenceDims const& dims, cudaStream_t stream)
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

bool LLMInferenceRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
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

void LLMInferenceRuntime::restoreRecurrentStates(
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

void LLMInferenceRuntime::zeroRecurrentStates(int32_t batchIdx, cudaStream_t stream)
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

bool LLMInferenceRuntime::setUpForPrefillExecution(DecodingInferenceContext& context, DecodingStrategy& strategy,
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
        ELLM_CHECK(
            mContextCache != nullptr && (!needsStrategyKVCache || strategy.kind() == DecodingStrategyKind::kEAGLE),
            "Managed context-cache prefill supports only vanilla or EAGLE decoding strategies");
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

bool LLMInferenceRuntime::genAndSaveSystemPromptKVCache(DecodingInferenceContext& context, int32_t genAndSaveBatchIdx)
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

bool LLMInferenceRuntime::genAndSaveSystemPromptKVCache(
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

bool LLMInferenceRuntime::performBatchEvict(DecodingInferenceContext& context, DecodingStrategy& strategy,
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
        // The legacy identity path owns its mapping upload and physical KV-row copy.
        check::check(mDeviceBatchMapping.reshape({oldActiveBatch}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mDeviceBatchMapping.rawPointer(), batchMapping.data(),
            static_cast<size_t>(oldActiveBatch) * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
        mSharedResources->cacheManagers[0]->compactBatch(
            mDeviceBatchMapping, oldActiveBatch, newActiveBatch, context.stream);
        mSharedResources->cacheManagers[0]->setActiveBatchSize(newActiveBatch);
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

    BatchCompactionMode const compactionMode
        = managedContextCache ? BatchCompactionMode::kManagedPageRows : BatchCompactionMode::kLegacyPhysicalKv;
    strategy.onBatchEvict(
        batchMapping, oldActiveBatch, newActiveBatch, mDeviceBatchMapping, context.stream, compactionMode);

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
    if (managedContextCache)
    {
        // Scope this MR's additional host-state compaction to coordinator-managed requests so disabling context reuse
        // preserves the legacy path.
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
