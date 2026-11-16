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

#include "runtime/modelArtifacts.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "runtime/config/deploymentOptions.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/exec/rtxRuntimeCache.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{

//! Load the reduced-vocab mapping table and check it describes the vocab the base engine was built for.
Tensor loadVocabMap(std::filesystem::path const& engineDir, LLMEngineConfig const& baseConfig, cudaStream_t stream)
{
    LOG_INFO("Loading vocabulary mapping table for base model reduced vocab size: %d -> %d",
        baseConfig.reducedVocabSize, baseConfig.vocabSize);
    std::filesystem::path const vocabMapPath = engineDir / binding_names::kVocabMapFileName;

    std::vector<Tensor> vocabMapTensors;
    ELLM_CHECK(safetensors::loadSafetensors(vocabMapPath, vocabMapTensors, stream),
        "Failed to load " + std::string(binding_names::kVocabMapFileName)
            + " from model directory: " + engineDir.string());

    check::check(vocabMapTensors.size() == 1,
        std::string(binding_names::kVocabMapFileName) + " should contain exactly one tensor");
    check::check(vocabMapTensors[0].getShape().getNumDims() == 1, "vocab_map tensor should be 1D");
    check::check(vocabMapTensors[0].getShape()[0] == baseConfig.reducedVocabSize,
        "vocab_map tensor length should match base model reduced vocab size");
    check::check(vocabMapTensors[0].getDataType() == trt_edgellm::DataType::kINT32, "vocab_map tensor should be INT32");
    LOG_INFO("Base model vocabulary mapping table successfully loaded.");
    return std::move(vocabMapTensors[0]);
}

std::unique_ptr<tokenizer::Tokenizer> loadTokenizer(
    std::filesystem::path const& engineDir, LLMEngineConfig const& baseConfig)
{
    auto tok = std::make_unique<tokenizer::Tokenizer>();
    LOG_INFO("Start loading tokenizer from model directory: %s", engineDir.c_str());
    ELLM_CHECK(
        tok->loadFromHF(engineDir.string()), "Failed to load tokenizer from model directory: " + engineDir.string());
    LOG_INFO("Tokenizer successfully loaded from model directory: %s", engineDir.c_str());

    // Config-declared extras beyond the tokenizer files, e.g. Gemma4's eos_token_id: [1, 106].
    if (!baseConfig.eosTokenIds.empty())
    {
        std::vector<tokenizer::Rank> additionalEos(baseConfig.eosTokenIds.begin(), baseConfig.eosTokenIds.end());
        tok->setAdditionalEosIds(additionalEos);
        LOG_INFO("Loaded %zu EOS token IDs from config", additionalEos.size());
    }
    return tok;
}

} // namespace

void validateRecurrentVerifyMetadataBindings(DeploymentConfig const& deployment, EngineExecutor const& baseExecutor)
{
    if (!deployment.specConfig.has_value() || deployment.base.numLinearAttnLayers == 0)
    {
        return;
    }
    ELLM_CHECK(
        baseExecutor.hasIOTensor(binding_names::kTreeParentIds) && baseExecutor.hasIOTensor(binding_names::kTreeDepths),
        "Recurrent verify engine must expose both INT32 tree metadata bindings; re-export and rebuild the artifact");
    ELLM_CHECK(baseExecutor.getBindingDataType(binding_names::kTreeParentIds) == trt_edgellm::DataType::kINT32
            && baseExecutor.getBindingDataType(binding_names::kTreeDepths) == trt_edgellm::DataType::kINT32,
        "Recurrent verify engine tree metadata bindings must be INT32");
    ELLM_CHECK(baseExecutor.hasIOTensor(binding_names::kSpecVerifyLayout),
        "Recurrent verify engine requires spec_verify_layout; re-export and rebuild the artifact");
    ELLM_CHECK(baseExecutor.getBindingDataType(binding_names::kSpecVerifyLayout) == trt_edgellm::DataType::kINT32,
        "Recurrent verify engine spec_verify_layout must be INT32");
}

void ModelArtifacts::validateDeploymentOptions(DeploymentOptions const& options) const
{
    ELLM_CHECK(!options.offloadEmbedding || embedding.table.getDeviceType() == DeviceType::kCPU,
        "--offloadEmbedding requires a host-backed embedding artifact; checkpoint-only embeddings are not offloaded.");
    auto const normalizePath = [](std::filesystem::path const& path) {
        auto const normalized = std::filesystem::absolute(path).lexically_normal();
        return normalized.filename().empty() ? normalized.parent_path() : normalized;
    };
    auto const checkPath = [&normalizePath](std::filesystem::path const& requested, std::filesystem::path const& loaded,
                               char const* name) {
        ELLM_CHECK(requested.empty() || (!loaded.empty() && normalizePath(requested) == normalizePath(loaded)),
            std::string("Injected ") + name + " conflicts with the source already loaded in ModelArtifacts.");
    };
    checkPath(options.checkpointDir, checkpointDir, "checkpointDir");
    checkPath(options.draftCheckpointDir, draftCheckpointDir, "draftCheckpointDir");
    if (options.runtimeCachePath)
    {
        ELLM_CHECK(deployment.runtimeCachePath.has_value(),
            "Injected runtimeCachePath cannot replace the cache of an already-loaded executor.");
        checkPath(*options.runtimeCachePath, *deployment.runtimeCachePath, "runtimeCachePath");
    }
    if (!options.draftingConfig)
    {
        return;
    }
    ELLM_CHECK(deployment.specConfig.has_value(), "Injected draftingConfig requires speculative ModelArtifacts.");
    ELLM_CHECK(deployment.draft.has_value(), "Injected draftingConfig requires a draft engine configuration.");
    auto const requested = resolveSpecDecodeDraftingConfig(deployment.base, *deployment.draft, *options.draftingConfig);
    auto const resolved = *deployment.draftingConfig();
    ELLM_CHECK(std::tie(requested.draftTopK, requested.draftingStep, requested.proposalTokens, requested.verifyTokens,
                   requested.dsparkSchedulerMode, requested.dsparkConfidenceThreshold, requested.dsparkMinProposalLen,
                   requested.dsparkMaxProposalLen, requested.dsparkW2Backend)
            == std::tie(resolved.draftTopK, resolved.draftingStep, resolved.proposalTokens, resolved.verifyTokens,
                resolved.dsparkSchedulerMode, resolved.dsparkConfidenceThreshold, resolved.dsparkMinProposalLen,
                resolved.dsparkMaxProposalLen, resolved.dsparkW2Backend),
        "Injected draftingConfig conflicts with the resolved ModelArtifacts deployment.");
}

ModelArtifacts ModelArtifacts::loadFromEngineDir(std::filesystem::path const& engineDir,
    std::optional<SpecDecodeDraftingConfig> const& draftingConfig, std::filesystem::path const& checkpointDir,
    std::filesystem::path const& draftCheckpointDir, bool contextReuseEnabled, cudaStream_t stream,
    std::optional<std::filesystem::path> const& runtimeCachePath, bool offloadEmbedding,
    std::function<void()> const& checkCancelled)
{
    ModelArtifacts artifacts;
    artifacts.checkpointDir = checkpointDir;
    artifacts.draftCheckpointDir = draftCheckpointDir;

    std::filesystem::path const baseConfigPath
        = draftingConfig.has_value() ? engineDir / "base_config.json" : engineDir / "config.json";

    // Finish checkpoint reads and weight conversion before any engine can run.
    artifacts.weights.load(
        engineDir, baseConfigPath, stream, artifacts.checkpointDir, {}, std::nullopt, std::nullopt, checkCancelled);
    if (auto embedding = artifacts.weights.takeEmbedding())
    {
        artifacts.embedding.table = std::move(*embedding);
    }
    else
    {
        artifacts.embedding = loadEmbeddingTable(engineDir / "embedding.safetensors", stream, offloadEmbedding);
    }

    std::optional<std::filesystem::path> const draftConfigPath = draftingConfig.has_value()
        ? std::optional<std::filesystem::path>{engineDir / "draft_config.json"}
        : std::nullopt;

    artifacts.deployment = createDeploymentConfig(baseConfigPath, draftConfigPath, draftingConfig);
    // EngineExecutor captures the active KV-pool geometry in its tensor registry.
    // Select the immutable runtime policy before constructing that registry.
    artifacts.deployment.selectSwaKVCacheMode(contextReuseEnabled);
    artifacts.deployment.runtimeCachePath = runtimeCachePath;
    if (draftingConfig.has_value() && artifacts.deployment.specDecodeMode() == SpecDecodeMode::kMTP)
    {
        ELLM_CHECK(artifacts.draftCheckpointDir.empty(),
            "Native MTP draft weights are part of --checkpointDir; do not pass --draftCheckpointDir.");
        artifacts.draftCheckpointDir = artifacts.checkpointDir;
    }

    std::filesystem::path const baseEnginePath = draftingConfig.has_value()
        ? engineDir / "spec_base.engine"
        : (artifacts.deployment.base.isDiffusionBackbone ? engineDir / "dllm.engine" : engineDir / "llm.engine");

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

    // Validate the draft engine ABI before its sidecar geometry is used to allocate physical cache resources.
    if (draftingConfig.has_value())
    {
        std::filesystem::path const draftConfig = engineDir / "draft_config.json";
        if (artifacts.deployment.specDecodeMode() == SpecDecodeMode::kMTP)
        {
            // Native MTP draft weights ship inside the base checkpoint, so there is no separate target to fall back to.
            artifacts.draftWeights.load(engineDir, draftConfig, stream, artifacts.checkpointDir, {}, std::nullopt,
                std::nullopt, checkCancelled);
        }
        else
        {
            artifacts.draftWeights.load(engineDir, draftConfig, stream, artifacts.draftCheckpointDir,
                artifacts.checkpointDir, std::nullopt, std::nullopt, checkCancelled);
        }
        artifacts.draftExecutor = decoder_utils::loadDraftEngine(engineDir, artifacts.deployment, runtimeCache,
            EngineRefitOptions{artifacts.draftWeights.takeRefitWeights(), stream});
        // Name the mode in the label. Loading moved here from the decoders, and each of them used to pass its own
        // ("dflash_draft", "dspark_draft", ...); one call site would otherwise report every mismatch as plain "draft"
        // and leave the reader to work out which speculative mode produced it.
        std::string const draftLabel
            = std::string{specDecodeModeName(artifacts.deployment.specDecodeMode())} + " draft";
        artifacts.draftWeights.validateAgainstEngine(*artifacts.draftExecutor, draftLabel);
    }

    if (artifacts.deployment.base.reducedVocabSize > 0)
    {
        artifacts.vocabMap = loadVocabMap(engineDir, artifacts.deployment.base, stream);
    }
    artifacts.tokenizer = loadTokenizer(engineDir, artifacts.deployment.base);

    return artifacts;
}

} // namespace rt
} // namespace trt_edgellm
