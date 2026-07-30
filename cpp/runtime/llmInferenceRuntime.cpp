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
#include "kernels/speculative/eagleUtilKernels.h"
#include "multimodal/multimodalRunner.h"
#include "multimodal/qwenViTRunner.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/debug/layerDebugger.h"
#include "runtime/decoding/decoderRegistry.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/llmRuntimeUtils.h"
#include "sampler/sampling.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
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

//! Ensures request-scoped cache leases cannot release physical pages while this request still has queued GPU work.
class ContextCacheStreamGuard
{
public:
    explicit ContextCacheStreamGuard(rt::DecodingInferenceContext& context)
        : mContext(context)
    {
    }

    ~ContextCacheStreamGuard() noexcept
    {
        if (mContext.contextReuseEnabled)
        {
            (void) cudaStreamSynchronize(mContext.stream);
        }
    }

private:
    rt::DecodingInferenceContext& mContext;
};

std::vector<int32_t> buildResidentTokenHistory(
    rt::DecodingInferenceContext const& context, int32_t slot, int32_t residentStateLength)
{
    std::vector<int32_t> const& rawInput = context.rawBatchedInputIds[slot];
    ELLM_CHECK(residentStateLength >= 0, "Context cache resident length must be non-negative");
    if (residentStateLength <= math::cast<int32_t>(rawInput.size()))
    {
        return std::vector<int32_t>(rawInput.begin(), rawInput.begin() + residentStateLength);
    }

    std::vector<int32_t> logicalTokens = rawInput;
    int32_t const generatedResidentTokens = residentStateLength - math::cast<int32_t>(rawInput.size());
    ELLM_CHECK(generatedResidentTokens >= 0 && generatedResidentTokens <= context.currentGenerateLengths[slot],
        "Context cache resident length is inconsistent with generated tokens");
    if (generatedResidentTokens > 0)
    {
        std::vector<int32_t> const& executionTokens = context.tokenIds[slot];
        ELLM_CHECK(context.currentGenerateLengths[slot] >= 0
                && static_cast<size_t>(context.currentGenerateLengths[slot]) <= executionTokens.size(),
            "Context cache generated-token history exceeds the execution-token history");
        size_t const generatedBegin
            = executionTokens.size() - static_cast<size_t>(context.currentGenerateLengths[slot]);
        logicalTokens.insert(logicalTokens.end(), executionTokens.begin() + static_cast<std::ptrdiff_t>(generatedBegin),
            executionTokens.begin() + static_cast<std::ptrdiff_t>(generatedBegin + generatedResidentTokens));
    }
    ELLM_CHECK(logicalTokens.size() == static_cast<size_t>(residentStateLength),
        "Context cache logical history does not match the resident boundary");
    return logicalTokens;
}

rt::Hash128 hashImageContent(rt::imageUtils::ImageData const& image)
{
    ELLM_CHECK(image.buffer != nullptr && image.buffer->getDeviceType() == rt::DeviceType::kCPU
            && image.buffer->getDataType() == nvinfer1::DataType::kUINT8,
        "Context reuse requires host UINT8 image input");
    size_t const bytes = math::cast<size_t>(image.buffer->getShape().volume());
    rt::Hash128 const pixels
        = rt::hashOpaqueIdentity(std::string_view{reinterpret_cast<char const*>(image.buffer->rawPointer()), bytes});
    uint64_t fpsBits{};
    static_assert(sizeof(fpsBits) == sizeof(image.fps));
    std::memcpy(&fpsBits, &image.fps, sizeof(fpsBits));
    std::ostringstream identity;
    identity << "image-v1:" << pixels.hi << ':' << pixels.lo << ':' << image.frames << ':' << image.height << ':'
             << image.width << ':' << image.channels << ':' << fpsBits;
    return rt::hashOpaqueIdentity(identity.str());
}

rt::Hash128 hashAudioContent(rt::audioUtils::AudioData const& audio)
{
    ELLM_CHECK(audio.pcm != nullptr, "Context reuse requires decoded PCM audio input");
    size_t const bytes = audio.pcm->samples.size() * sizeof(float);
    rt::Hash128 const samples
        = rt::hashOpaqueIdentity(std::string_view{reinterpret_cast<char const*>(audio.pcm->samples.data()), bytes});
    std::ostringstream identity;
    identity << "audio-pcm-v1:" << samples.hi << ':' << samples.lo << ':' << audio.pcm->samples.size() << ':'
             << audio.pcm->sampleRate << ':' << audio.pcm->numChannels;
    return rt::hashOpaqueIdentity(identity.str());
}

rt::Hash128 mediaPlacementDigest(rt::MediaSpanDescriptor const& span)
{
    std::ostringstream identity;
    identity << "media-placement-v2:" << static_cast<int32_t>(span.artifactKey.modality) << ':' << span.tokenLength
             << ':' << span.itemOrder;
    for (rt::MediaTokenSegment const& segment : span.tokenSegments)
    {
        identity << ':' << segment.tokenStart << '+' << segment.tokenLength << '@' << segment.artifactRowOffset;
    }
    return rt::hashOpaqueIdentity(identity.str());
}

int32_t mediaRowsBeforeToken(rt::MediaSpanDescriptor const& span, int32_t tokenBoundary)
{
    int32_t rows{};
    for (rt::MediaTokenSegment const& segment : span.tokenSegments)
    {
        if (tokenBoundary <= segment.tokenStart)
        {
            break;
        }
        rows += std::min(tokenBoundary - segment.tokenStart, segment.tokenLength);
    }
    ELLM_CHECK(rows >= 0 && rows <= span.tokenLength, "Context cache media boundary is invalid");
    return rows;
}

std::vector<rt::BlockKeyExtras> buildBlockExtras(rt::BlockKeyExtras const& common,
    std::vector<rt::MediaSpanDescriptor> const& mediaSpans, size_t tokenCount, int32_t pageSize)
{
    size_t const blockCount = (tokenCount + static_cast<size_t>(pageSize) - 1U) / static_cast<size_t>(pageSize);
    std::vector<rt::BlockKeyExtras> extras(blockCount, common);
    for (rt::MediaSpanDescriptor const& span : mediaSpans)
    {
        ELLM_CHECK(span.tokenLength > 0 && !span.tokenSegments.empty(), "Context cache media span is invalid");
        rt::Hash128 const placement = mediaPlacementDigest(span);
        for (rt::MediaTokenSegment const& segment : span.tokenSegments)
        {
            ELLM_CHECK(segment.tokenStart >= 0 && segment.tokenLength > 0 && segment.artifactRowOffset >= 0
                    && segment.artifactRowOffset + segment.tokenLength <= span.tokenLength,
                "Context cache media token segment is invalid");
            if (static_cast<size_t>(segment.tokenStart) >= tokenCount)
            {
                continue;
            }
            int32_t const firstBlock = segment.tokenStart / pageSize;
            size_t const segmentEnd
                = static_cast<size_t>(segment.tokenStart) + static_cast<size_t>(segment.tokenLength);
            int32_t const finalToken = math::cast<int32_t>(std::min(segmentEnd, tokenCount) - 1U);
            int32_t const lastBlock = finalToken / pageSize;
            for (int32_t block = firstBlock; block <= lastBlock; ++block)
            {
                if (block < 0 || static_cast<size_t>(block) >= blockCount)
                {
                    continue;
                }
                extras[static_cast<size_t>(block)].media.push_back(
                    rt::MediaSpanKey{span.artifactKey.contentDigest, placement, segment.tokenStart - block * pageSize,
                        span.itemOrder, static_cast<uint8_t>(span.artifactKey.modality)});
            }
        }
    }
    return extras;
}

bool populateMediaSpans(rt::LLMGenerationRequest const& request, rt::LLMEngineConfig const& config,
    std::vector<std::vector<int32_t>> const& batchedInputIds, std::vector<int64_t> const& visionRowCounts,
    std::vector<int64_t> const& audioRowCounts, rt::DecodingInferenceContext& context)
{
    struct TokenRun
    {
        rt::MediaModality modality;
        int32_t start{};
        int32_t length{};
    };

    rt::Hash128 const isolation = request.contextCacheIsolationKey.empty()
        ? rt::Hash128{}
        : rt::hashOpaqueIdentity(request.contextCacheIsolationKey);
    size_t const totalVisionArtifacts = std::accumulate(request.requests.begin(), request.requests.end(), size_t{0},
        [](size_t count, auto const& slot) { return count + slot.imageBuffers.size(); });
    size_t const totalAudioArtifacts = std::accumulate(request.requests.begin(), request.requests.end(), size_t{0},
        [](size_t count, auto const& slot) { return count + slot.audioBuffers.size(); });
    if ((!visionRowCounts.empty() && visionRowCounts.size() != totalVisionArtifacts)
        || (!audioRowCounts.empty() && audioRowCounts.size() != totalAudioArtifacts))
    {
        LOG_WARNING("Context cache media inspection returned an invalid artifact row-count layout.");
        return false;
    }

    auto partitionRuns
        = [](std::vector<TokenRun> const& runs, size_t artifactCount, std::vector<int64_t> const& rowCounts,
              size_t& globalArtifactIndex,
              char const* modalityName) -> std::optional<std::vector<std::vector<rt::MediaTokenSegment>>> {
        if (rowCounts.empty() && runs.size() != artifactCount)
        {
            LOG_WARNING("Context cache media inspection could not map %s placeholders: runs=%zu, artifacts=%zu.",
                modalityName, runs.size(), artifactCount);
            return std::nullopt;
        }
        std::vector<std::vector<rt::MediaTokenSegment>> result(artifactCount);
        size_t runIndex{};
        int32_t consumedInRun{};
        for (size_t artifact = 0; artifact < artifactCount; ++artifact)
        {
            int64_t const expectedRows = rowCounts.empty() ? runs[runIndex].length : rowCounts[globalArtifactIndex];
            if (expectedRows <= 0 || expectedRows > std::numeric_limits<int32_t>::max())
            {
                LOG_WARNING("Context cache %s artifact has an invalid encoder row count.", modalityName);
                return std::nullopt;
            }
            int32_t remaining = static_cast<int32_t>(expectedRows);
            int32_t artifactRowOffset{};
            while (remaining > 0)
            {
                if (runIndex >= runs.size())
                {
                    LOG_WARNING("Context cache %s artifact rows exceed token placeholders.", modalityName);
                    return std::nullopt;
                }
                TokenRun const& run = runs[runIndex];
                int32_t const available = run.length - consumedInRun;
                int32_t const take = std::min(remaining, available);
                result[artifact].push_back(rt::MediaTokenSegment{run.start + consumedInRun, take, artifactRowOffset});
                remaining -= take;
                artifactRowOffset += take;
                consumedInRun += take;
                if (consumedInRun == run.length)
                {
                    ++runIndex;
                    consumedInRun = 0;
                }
            }
            ++globalArtifactIndex;
        }
        if (runIndex != runs.size() || consumedInRun != 0)
        {
            LOG_WARNING("Context cache %s token placeholders exceed artifact rows.", modalityName);
            return std::nullopt;
        }
        return result;
    };

    int32_t visionRows{};
    int32_t audioRows{};
    size_t visionArtifactIndex{};
    size_t audioArtifactIndex{};
    auto classifyMediaToken = [&](int32_t token) -> std::optional<rt::MediaModality> {
        if (config.audioTokenId >= 0 && token == config.audioTokenId)
        {
            return rt::MediaModality::kAudio;
        }
        if ((config.imageTokenId >= 0 && token == config.imageTokenId) || token >= config.vocabSize)
        {
            return rt::MediaModality::kVision;
        }
        return std::nullopt;
    };
    for (size_t slot = 0; slot < request.requests.size(); ++slot)
    {
        std::vector<TokenRun> runs;
        std::vector<int32_t> const& tokens = batchedInputIds[slot];
        for (size_t index = 0; index < tokens.size();)
        {
            std::optional<rt::MediaModality> const modality = classifyMediaToken(tokens[index]);
            if (!modality.has_value())
            {
                ++index;
                continue;
            }
            size_t end = index + 1;
            while (end < tokens.size() && classifyMediaToken(tokens[end]) == modality)
            {
                ++end;
            }
            runs.push_back(TokenRun{*modality, math::cast<int32_t>(index), math::cast<int32_t>(end - index)});
            index = end;
        }

        std::vector<TokenRun> visionRuns;
        std::vector<TokenRun> audioRuns;
        for (TokenRun const& run : runs)
        {
            (run.modality == rt::MediaModality::kVision ? visionRuns : audioRuns).push_back(run);
        }
        std::optional<std::vector<std::vector<rt::MediaTokenSegment>>> visionSegments = partitionRuns(
            visionRuns, request.requests[slot].imageBuffers.size(), visionRowCounts, visionArtifactIndex, "vision");
        std::optional<std::vector<std::vector<rt::MediaTokenSegment>>> audioSegments = partitionRuns(
            audioRuns, request.requests[slot].audioBuffers.size(), audioRowCounts, audioArtifactIndex, "audio");
        if (!visionSegments.has_value() || !audioSegments.has_value())
        {
            return false;
        }

        for (size_t artifact = 0; artifact < request.requests[slot].imageBuffers.size(); ++artifact)
        {
            rt::MediaSpanDescriptor span;
            span.tokenSegments = std::move((*visionSegments)[artifact]);
            span.tokenLength = std::accumulate(span.tokenSegments.begin(), span.tokenSegments.end(), int32_t{0},
                [](int32_t rows, rt::MediaTokenSegment const& segment) { return rows + segment.tokenLength; });
            span.embeddingOffset = visionRows;
            span.itemOrder = math::cast<int32_t>(artifact);
            span.artifactKey = rt::MediaArtifactKey{
                hashImageContent(request.requests[slot].imageBuffers[artifact]), isolation, rt::MediaModality::kVision};
            for (rt::MediaTokenSegment const& segment : span.tokenSegments)
            {
                int32_t const firstToken = batchedInputIds[slot][static_cast<size_t>(segment.tokenStart)];
                if (firstToken >= config.vocabSize
                    && firstToken - config.vocabSize != visionRows + segment.artifactRowOffset)
                {
                    LOG_WARNING("Context cache vision token IDs do not match packed encoder row order.");
                    return false;
                }
            }
            visionRows += span.tokenLength;
            context.mediaSpans[slot].push_back(std::move(span));
        }
        for (size_t artifact = 0; artifact < request.requests[slot].audioBuffers.size(); ++artifact)
        {
            rt::MediaSpanDescriptor span;
            span.tokenSegments = std::move((*audioSegments)[artifact]);
            span.tokenLength = std::accumulate(span.tokenSegments.begin(), span.tokenSegments.end(), int32_t{0},
                [](int32_t rows, rt::MediaTokenSegment const& segment) { return rows + segment.tokenLength; });
            span.embeddingOffset = audioRows;
            span.itemOrder = math::cast<int32_t>(artifact);
            span.artifactKey = rt::MediaArtifactKey{
                hashAudioContent(request.requests[slot].audioBuffers[artifact]), isolation, rt::MediaModality::kAudio};
            audioRows += span.tokenLength;
            context.mediaSpans[slot].push_back(std::move(span));
        }
        std::sort(
            context.mediaSpans[slot].begin(), context.mediaSpans[slot].end(), [](auto const& lhs, auto const& rhs) {
                return lhs.tokenSegments.front().tokenStart < rhs.tokenSegments.front().tokenStart;
            });
    }
    return true;
}

//! Fires `context.onTokenGenerated` once per active slot using the most recent
//! token in `tokenIds`. Called at the end of prefill (one token sampled per
//! slot) and after every decode iteration so streaming consumers see every
//! emitted token in order.
inline void emitTokenCallbacks(rt::DecodingInferenceContext& context)
{
    if (!context.onTokenGenerated.has_value())
    {
        return;
    }
    auto const& callback = context.onTokenGenerated.value();
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        auto const& slotTokens = context.tokenIds[i];
        if (slotTokens.empty())
        {
            continue;
        }
        bool const isFinished = context.finishedStates[i] != 0;
        callback(rt::TokenCallbackInfo{slotTokens.back(), i, context.generationRound, isFinished});
    }
}

} // namespace

namespace rt
{
namespace
{
void validateTreeMetadataBindings(DeploymentConfig const& deployment, EngineExecutor const& baseExecutor)
{
    if (!deployment.specConfig.has_value())
    {
        return;
    }

    SpecDecodeMode const mode = deployment.specDecodeMode();
    bool const isDFlash = mode == SpecDecodeMode::kDFlash;
    bool const isMTP = mode == SpecDecodeMode::kMTP;
    if (!isDFlash && !isMTP)
    {
        return;
    }

    std::string const modeName = isDFlash ? "DFlash" : "MTP";
    std::string const treeExportFlag = isDFlash ? "--dflash-tree-base" : "--mtp";
    bool const hasTreeParentIds = baseExecutor.hasIOTensor(binding_names::kTreeParentIds);
    bool const hasTreeDepths = baseExecutor.hasIOTensor(binding_names::kTreeDepths);
    bool const hasTreeMetadata = hasTreeParentIds || hasTreeDepths;
    bool const usesTree = isMTP || (isDFlash && deployment.specConfig->draftingTopK > 1);
    if (hasTreeMetadata)
    {
        ELLM_CHECK(hasTreeParentIds && hasTreeDepths,
            modeName + " tree-base engine must expose both INT32 tree metadata bindings '"
                + binding_names::kTreeParentIds + "' and '" + binding_names::kTreeDepths + "'.");
        ELLM_CHECK(baseExecutor.getBindingDataType(binding_names::kTreeParentIds) == DataType::kINT32
                && baseExecutor.getBindingDataType(binding_names::kTreeDepths) == DataType::kINT32,
            modeName + " tree-base engine tree metadata bindings must be INT32: '" + binding_names::kTreeParentIds
                + "' and '" + binding_names::kTreeDepths + "'.");
        if (isDFlash)
        {
            ELLM_CHECK(usesTree,
                "DFlash base engine was exported with --dflash-tree-base, but runtime is configured for linear "
                "drafting because specDraftTopK=1. Use --specDraftTopK > 1 for DFlash tree drafting, or re-export "
                "the base model with --dflash-base.");
        }
    }

    if (!usesTree || deployment.base.numLinearAttnLayers == 0)
    {
        return;
    }

    ELLM_CHECK(hasTreeParentIds && hasTreeDepths,
        modeName + " hybrid tree base engine requires INT32 tree metadata bindings '" + binding_names::kTreeParentIds
            + "' and '" + binding_names::kTreeDepths + "'. Re-export the base model with " + treeExportFlag
            + ", then rebuild spec_base.engine.");
}
} // namespace

LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, SpecDecodeDraftingConfig const& draftingConfig,
    cudaStream_t stream, ContextCacheConfig const& contextCacheConfig)
{
    initializeCommon(engineDir, multimodalEngineDir, loraWeightsMap, draftingConfig, stream, contextCacheConfig);
}

LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream,
    ContextCacheConfig const& contextCacheConfig)
{
    initializeCommon(engineDir, multimodalEngineDir, loraWeightsMap, std::nullopt, stream, contextCacheConfig);
}

void LLMInferenceRuntime::initializeCommon(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap,
    std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream,
    ContextCacheConfig const& contextCacheConfig)
{
    ELLM_CHECK(contextCacheConfig.maxRecords >= 0 && contextCacheConfig.recurrentSnapshotPoolBytes >= 0
            && contextCacheConfig.partialKvSnapshotPoolBytes >= 0 && contextCacheConfig.mediaArtifactPoolBytes >= 0
            && contextCacheConfig.maxMediaArtifacts >= 0,
        "Context cache record, snapshot, and media budgets must be non-negative");
    mContextCacheConfig = contextCacheConfig;
    // -----------------------------------------------------------------------
    // 1. Load shared embedding table (shared between base and draft models).
    // -----------------------------------------------------------------------
    std::filesystem::path const embeddingPath = std::filesystem::path(engineDir) / "embedding.safetensors";
    mEmbedding = loadEmbeddingTable(embeddingPath, stream);

    // -----------------------------------------------------------------------
    // 2. Parse engine configurations and attach user drafting (bundle factory
    //    performs cross-engine consistency and drafting-vs-capacity checks).
    // -----------------------------------------------------------------------
    std::filesystem::path const engineDirPath{engineDir};
    std::filesystem::path const baseEnginePath
        = draftingConfig.has_value() ? engineDirPath / "spec_base.engine" : engineDirPath / "llm.engine";
    std::filesystem::path const baseConfigPath
        = draftingConfig.has_value() ? engineDirPath / "base_config.json" : engineDirPath / "config.json";
    std::optional<std::filesystem::path> const draftConfigPath = draftingConfig.has_value()
        ? std::optional<std::filesystem::path>{engineDirPath / "draft_config.json"}
        : std::nullopt;

    mDeployment = createDeploymentConfig(baseConfigPath, draftConfigPath, draftingConfig);
    std::string domainIdentity = baseConfigPath.string();
    if (draftConfigPath.has_value())
    {
        domainIdentity.push_back('\0');
        domainIdentity += draftConfigPath->string();
    }
    if (!multimodalEngineDir.empty())
    {
        domainIdentity.push_back('\0');
        domainIdentity += std::filesystem::path(multimodalEngineDir).lexically_normal().string();
    }
    mContextCacheDomain = hashOpaqueIdentity(domainIdentity);
    SpecDecodeMode const specMode = mDeployment.specDecodeMode();
    if (specMode == SpecDecodeMode::kEAGLE || specMode == SpecDecodeMode::kMTP)
    {
        ELLM_CHECK(draftingConfig.has_value(), "Speculative deployment is missing its drafting configuration");
        std::string draftIdentity = domainIdentity;
        draftIdentity.push_back('\0');
        draftIdentity += specMode == SpecDecodeMode::kEAGLE ? "eagle" : "mtp";
        draftIdentity.push_back('\0');
        draftIdentity += std::to_string(draftingConfig->draftingTopK);
        draftIdentity.push_back(':');
        draftIdentity += std::to_string(draftingConfig->draftingStep);
        draftIdentity.push_back(':');
        draftIdentity += std::to_string(draftingConfig->verifySize);
        mDraftEngineSignature = hashOpaqueIdentity(draftIdentity);
    }

    if (mContextCacheConfig.enabled)
    {
        SpecDecodeMode const mode = mDeployment.specDecodeMode();
        ELLM_CHECK(mode == SpecDecodeMode::kNONE || mode == SpecDecodeMode::kEAGLE || mode == SpecDecodeMode::kMTP,
            "Production context reuse currently supports vanilla, EAGLE, and MTP deployments only");
        ELLM_CHECK(mode != SpecDecodeMode::kEAGLE || mDeployment.base.numLinearAttnLayers == 0,
            "Production context reuse does not support hybrid EAGLE deployments");
        ELLM_CHECK(mode != SpecDecodeMode::kMTP || mDeployment.base.numLinearAttnLayers > 0,
            "Production MTP context reuse is limited to hybrid base models");
        ELLM_CHECK(mDeployment.base.slidingWindowSize.has_value()
                && (!mDeployment.draft.has_value() || mDeployment.draft->slidingWindowSize.has_value()),
            "Production context reuse requires engine configs with explicit sliding_window metadata; re-export and "
            "rebuild this engine with the current toolchain");
        bool const baseUsesSlidingWindow = *mDeployment.base.slidingWindowSize > 0 || mDeployment.base.useDualRope;
        bool const draftUsesSlidingWindow = mDeployment.draft.has_value()
            && (*mDeployment.draft->slidingWindowSize > 0 || mDeployment.draft->useDualRope);
        ELLM_CHECK(!baseUsesSlidingWindow && !draftUsesSlidingWindow,
            "Production context reuse does not yet support sliding-window attention; disable context reuse or use "
            "a full-attention engine until backend capability validation is available");
    }

    ELLM_CHECK(mDeployment.base.numDeepstackFeatures <= 0 || !multimodalEngineDir.empty(),
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
    validateTreeMetadataBindings(mDeployment, *mBaseExecutor);

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
    if (mContextCacheConfig.enabled)
    {
        ResourceDemand capacities{};
        bool const baseHasAttention = !mDeployment.base.kvLayerConfigs.empty();
        capacities.baseKvPages = baseHasAttention ? mDeployment.base.kvPoolPages : 0;
        if (mDeployment.specDecodeMode() == SpecDecodeMode::kEAGLE
            || mDeployment.specDecodeMode() == SpecDecodeMode::kMTP)
        {
            capacities.draftKvPages = mDeployment.draft->kvPoolPages;
        }
        if (mDeployment.base.numLinearAttnLayers > 0)
        {
            HybridCacheManager& cacheManager = *mSharedResources->cacheManagers[0];
            HybridCacheManager::Config const& cacheConfig = cacheManager.getConfig();
            HybridCacheManager* const draftCacheManager = mDeployment.specDecodeMode() == SpecDecodeMode::kMTP
                ? mSharedResources->cacheManagers[1].get()
                : nullptr;
            HybridCacheManager::Config const* const draftCacheConfig
                = draftCacheManager == nullptr ? nullptr : &draftCacheManager->getConfig();
            size_t const recurrentBytesPerSlot = HybridSnapshotStorage::recurrentBytesPerSlot(cacheConfig);
            size_t const partialKvBytesPerSlot
                = HybridSnapshotStorage::partialKvBytesPerSlot(cacheConfig, draftCacheConfig);
            ELLM_CHECK(recurrentBytesPerSlot > 0, "Hybrid recurrent snapshot schema has zero bytes per slot");

            int64_t const recurrentSlotCount
                = mContextCacheConfig.recurrentSnapshotPoolBytes / static_cast<int64_t>(recurrentBytesPerSlot);
            int64_t const partialKvSlotCount = partialKvBytesPerSlot == 0
                ? 0
                : mContextCacheConfig.partialKvSnapshotPoolBytes / static_cast<int64_t>(partialKvBytesPerSlot);
            ELLM_CHECK(recurrentSlotCount <= std::numeric_limits<int32_t>::max()
                    && partialKvSlotCount <= std::numeric_limits<int32_t>::max(),
                "Hybrid snapshot byte budget produces too many slots");
            if (mDeployment.specDecodeMode() == SpecDecodeMode::kMTP)
            {
                // The scoped MLPerf path is serialized and retains only the immediately preceding endpoint. One
                // recurrent slot forces the old conversation's canonical lineage to retire before a new endpoint is
                // published.
                ELLM_CHECK(recurrentSlotCount == 1 && partialKvSlotCount > 0,
                    "Hybrid MTP context reuse with concurrency one requires exactly one recurrent snapshot slot and "
                    "at least one bundled partial-KV slot; realized recurrent="
                        + std::to_string(recurrentSlotCount) + " (" + std::to_string(recurrentBytesPerSlot)
                        + " bytes/slot), partialKV=" + std::to_string(partialKvSlotCount) + " ("
                        + std::to_string(partialKvBytesPerSlot) + " bytes/slot)");
            }
            capacities.recurrentSnapshotSlots = static_cast<int32_t>(recurrentSlotCount);
            capacities.partialKvSnapshotSlots = static_cast<int32_t>(partialKvSlotCount);
            // Hybrid+MTP saves one base-hidden vector per checkpoint so the successor-dependent boundary draft slot can
            // be recomputed for any following token at restore. The draft reads baseHiddenStates directly, so match its
            // width and dtype.
            int32_t const boundaryHiddenDim
                = draftCacheManager != nullptr ? mDeployment.specConfig->baseOutputHiddenDim : 0;
            nvinfer1::DataType const boundaryHiddenType = draftCacheManager != nullptr
                ? mPipelineIO->baseHiddenStates.getDataType()
                : nvinfer1::DataType::kHALF;
            mHybridSnapshotStorage
                = std::make_unique<HybridSnapshotStorage>(cacheManager, capacities.recurrentSnapshotSlots,
                    capacities.partialKvSnapshotSlots, draftCacheManager, boundaryHiddenDim, boundaryHiddenType);
            if (draftCacheManager != nullptr)
            {
                // Scratch to shift base hidden states by one row when folding the checkpoint boundary into the draft
                // prefill; sized to match baseHiddenStates' [maxSeq, hidden].
                rt::Coords const& bhShape = mPipelineIO->baseHiddenStates.getShape();
                mBoundaryFoldScratch = rt::Tensor({bhShape[1], bhShape[2]}, rt::DeviceType::kGPU, boundaryHiddenType,
                    "LLMInferenceRuntime::mBoundaryFoldScratch");
            }
            mRecurrentStateSchema = HybridSnapshotStorage::schemaId(cacheConfig, draftCacheConfig);
            LOG_INFO(
                "Context cache hybrid snapshots: recurrent=%d slots (%zu bytes/slot), partialKV=%d slots "
                "(%zu bytes/slot)",
                capacities.recurrentSnapshotSlots, recurrentBytesPerSlot, capacities.partialKvSnapshotSlots,
                partialKvBytesPerSlot);
        }
        mContextCache = std::make_unique<ContextCacheRuntimeAdapter>(
            kTOKENS_PER_PAGE, capacities, mContextCacheConfig.maxRecords);
        if (mContextCacheConfig.mediaArtifactPoolBytes > 0 && mContextCacheConfig.maxMediaArtifacts > 0)
        {
            mMediaArtifactCache = std::make_unique<MediaArtifactCache>(
                math::cast<size_t>(mContextCacheConfig.mediaArtifactPoolBytes), mContextCacheConfig.maxMediaArtifacts);
        }
    }
    // Externalized model weights: the SharedResources factory only allocates an
    // empty manager. Load external weights and validate against engine inputs.
    // This handles the base engine; the spec-decode draft engine loads its own
    // external weights from draft_config.json inside the EAGLE/MTP decoder.
    mSharedResources->externalWeightManager->load(std::filesystem::path(engineDir), baseConfigPath, stream);
    mSharedResources->externalWeightManager->validateAgainstEngine(*mBaseExecutor, "base");

    // -----------------------------------------------------------------------
    // 7. Build base TensorMap (kvCacheIndex=0) and publish static external
    //    weight bindings. Speculative decoders add tree-mask / position IDs
    //    to this same map further down.
    // -----------------------------------------------------------------------
    buildTensorMap(mBaseTensorMap, *mPipelineIO, *mSharedResources, mDeployment.base, /*kvCacheIndex=*/0);
    mSharedResources->externalWeightManager->registerTensorMapEntries(mBaseTensorMap);

    auto const baseLogitsType = mBaseExecutor->getBindingDataType(binding_names::kLogits);
    mConvertBaseLogits = baseLogitsType == DataType::kHALF;
    ELLM_CHECK(baseLogitsType == DataType::kFLOAT || mConvertBaseLogits, "Base engine logits must be FLOAT or HALF.");
    if (mConvertBaseLogits)
    {
        bool const supportedHalfLogits = hasDraft && mDeployment.specDecodeMode() == SpecDecodeMode::kMTP;
        ELLM_CHECK(supportedHalfLogits, "HALF base logits are supported only by MTP.");
        mBaseEngineOutputLogits = rt::Tensor(mPipelineIO->outputLogits.getShape(), rt::DeviceType::kGPU,
            DataType::kHALF, "LLMInferenceRuntime::mBaseEngineOutputLogits");
        mBaseTensorMap.set(binding_names::kLogits, mBaseEngineOutputLogits);
        LOG_INFO("Base engine emits HALF logits; enabled runtime FP32 publication for MTP.");
    }

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
    if (mDeployment.base.numDeepstackFeatures > 0)
    {
        mDeepstack = std::make_unique<DeepstackBinding>(mPipelineIO->deepstackEmbeds, mSharedResources->zeroBuffer);
    }

    // -----------------------------------------------------------------------
    // 10. Allocate runtime-local tensors (sampling workspace, host pinned scratch,
    //     batch-eviction mapping). Strategy-specific tensors are owned by strategies.
    // -----------------------------------------------------------------------
    int32_t const effectiveMaxProposalSize = hasDraft ? mDeployment.effectiveMaxDraftProposalSize() : 1;
    int32_t const effectiveDraftTopK = hasDraft ? draftingConfig->draftingTopK : 1;
    int32_t const maxInputLength = hasDraft
        ? std::max(mDeployment.base.maxSupportedInputLength, mDeployment.draft->maxSupportedInputLength)
        : mDeployment.base.maxSupportedInputLength;
    int32_t const maxSamplingSize = hasDraft ? std::max(mMaxRuntimeBatchSize * effectiveMaxProposalSize,
                                                   mMaxRuntimeBatchSize * effectiveDraftTopK * effectiveDraftTopK)
                                             : mMaxRuntimeBatchSize;

    // Reserve enough workspace for sampling, accounting for batch dimension in draft proposal stage.
    // Always include vanilla sampling workspace size because per-request disable_spec_decode
    // can fall back to topK/topP sampling even when draft is loaded.
    int32_t const vanillaSamplingWorkspaceSize
        = static_cast<int32_t>(getTopKtopPSamplingWorkspaceSize(mMaxRuntimeBatchSize, mDeployment.base.outputVocabSize,
            SamplingParams(mMaxRuntimeBatchSize, mDeployment.base.outputVocabSize, 1.0f, 0, 0.9f)));
    int32_t const draftSamplingRows = hasDraft && mDeployment.specDecodeMode() == SpecDecodeMode::kDFlash
        ? mMaxRuntimeBatchSize * mDeployment.specConfig->verifySize
        : mMaxRuntimeBatchSize * effectiveDraftTopK;
    int32_t const draftSamplingTopK
        = hasDraft && mDeployment.specDecodeMode() == SpecDecodeMode::kDFlash ? 1 : effectiveDraftTopK;
    mLogprobsMaxBatchDim = mMaxRuntimeBatchSize * mDeployment.maxAcceptedTokensPerRound();
    int32_t const logprobsWorkspaceSize = static_cast<int32_t>(
        getExtractTopKLogprobsWorkspaceSize(mLogprobsMaxBatchDim, mDeployment.base.outputVocabSize, kMaxLogprobsK));
    int32_t const maxSamplingWorkspaceSize = hasDraft
        ? std::max({vanillaSamplingWorkspaceSize,
              static_cast<int32_t>(
                  getSelectAllTopKWorkspaceSize(mMaxRuntimeBatchSize, mDeployment.base.outputVocabSize, 1)),
              static_cast<int32_t>(getSelectAllTopKWorkspaceSize(
                  draftSamplingRows, mDeployment.draft->outputVocabSize, draftSamplingTopK)),
              logprobsWorkspaceSize})
        : std::max(vanillaSamplingWorkspaceSize, logprobsWorkspaceSize);

    try
    {
        mIdsInput = rt::Tensor({mMaxRuntimeBatchSize, maxInputLength}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMInferenceRuntime::mIdsInput");

        mSamplingWorkspace = rt::Tensor({maxSamplingWorkspaceSize}, rt::DeviceType::kGPU, DataType::kINT8,
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
        mHostSelectedTokenIds = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMInferenceRuntime::mHostSelectedTokenIds");
        mHostReuseKVCacheLengths = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMInferenceRuntime::mHostReuseKVCacheLengths");
        mHostDraftReuseKVCacheLengths = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMInferenceRuntime::mHostDraftReuseKVCacheLengths");

        // Pre-allocate multimodal indices tensor (used for audio/vision embedding lookup).
        mMultimodalIndices = rt::Tensor({mMaxRuntimeBatchSize, maxInputLength}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMInferenceRuntime::mMultimodalIndices");

        allocateLogprobsTensors();
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
            mMaxRuntimeBatchSize, maxPleSeqLen, mBaseTensorMap, stream);
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
    mDecoderRegistry = std::make_unique<DecoderRegistry>(
        *mDecodingRuntimeContext, DecoderRegistryConfig{std::filesystem::path(engineDir), draftingConfig, stream});

    // -----------------------------------------------------------------------
    // 14. Optional multimodal runners.
    // -----------------------------------------------------------------------
    if (!multimodalEngineDir.empty())
    {
        auto tryLoadRunner = [&](std::string const& dir, std::string const& name) -> std::unique_ptr<MultimodalRunner> {
            try
            {
                LOG_DEBUG("Attempting to load %s runner from %s", name.c_str(), dir.c_str());
                auto runner = MultimodalRunner::create(
                    dir, mDeployment.base.maxSupportedBatchSize, mDeployment.base.maxKVCacheCapacity, stream);
                LOG_INFO("%s runner successfully initialized", name.c_str());
                return runner;
            }
            catch (std::exception const& e)
            {
                LOG_DEBUG("Failed to load %s runner from %s: %s", name.c_str(), dir.c_str(), e.what());
                return nullptr;
            }
        };

        mAudioRunner = tryLoadRunner(multimodalEngineDir + "/audio", "Audio");
        mVisionRunner = tryLoadRunner(multimodalEngineDir + "/visual", "Visual");
        if (!mVisionRunner)
        {
            mVisionRunner = tryLoadRunner(multimodalEngineDir, "Vision");
        }

        // At least one multimodal runner must be available
        ELLM_CHECK(mAudioRunner || mVisionRunner, "No valid multimodal engine found in " + multimodalEngineDir);

        // Try to load action expert from multimodalEngineDir/action
        try
        {
            std::string actionDir = multimodalEngineDir + "/action";
            LOG_INFO("Attempting to load Action runner from %s", actionDir.c_str());
            mActionRunner = std::make_unique<Alpamayo1ActionRunner>(actionDir, stream,
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
            ELLM_CHECK(!mContextCacheConfig.enabled,
                "Production context reuse is incompatible with the identity-only action runner");
            int32_t const actionMaxKVCacheCapacity = mActionRunner->getMaxKVCacheCapacity();
            int32_t const llmMaxKVCacheCapacity = mDeployment.base.maxKVCacheCapacity;
            ELLM_CHECK(actionMaxKVCacheCapacity == llmMaxKVCacheCapacity,
                format::fmtstr(
                    "Action engine max_kv_cache_capacity (%d) does not match LLM engine max_kv_cache_capacity (%d). "
                    "Re-export and rebuild the action engine with --max_kv_cache_capacity=%d to match the LLM engine.",
                    actionMaxKVCacheCapacity, llmMaxKVCacheCapacity, llmMaxKVCacheCapacity));
        }
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

void LLMInferenceRuntime::allocateLogprobsTensors()
{
    int32_t const logprobsRows = mMaxRuntimeBatchSize * mDeployment.maxAcceptedTokensPerRound();
    mDeviceLogprobsValues = rt::Tensor({logprobsRows, kMaxLogprobsK}, rt::DeviceType::kGPU, DataType::kFLOAT,
        "LLMInferenceRuntime::mDeviceLogprobsValues");
    mDeviceLogprobsIndices = rt::Tensor({logprobsRows, kMaxLogprobsK}, rt::DeviceType::kGPU, DataType::kINT32,
        "LLMInferenceRuntime::mDeviceLogprobsIndices");
    mHostLogprobsValues = rt::Tensor({logprobsRows, kMaxLogprobsK}, rt::DeviceType::kCPU, DataType::kFLOAT,
        "LLMInferenceRuntime::mHostLogprobsValues");
    mHostLogprobsIndices = rt::Tensor({logprobsRows, kMaxLogprobsK}, rt::DeviceType::kCPU, DataType::kINT32,
        "LLMInferenceRuntime::mHostLogprobsIndices");
    if (mDeployment.specConfig.has_value())
    {
        mGatheredLogits = rt::Tensor({logprobsRows, mDeployment.base.outputVocabSize}, rt::DeviceType::kGPU,
            DataType::kFLOAT, "LLMInferenceRuntime::mGatheredLogits");
    }
}

void LLMInferenceRuntime::buildDecodingRuntimeContext()
{
    BaseEngineResources baseResources{*mBaseExecutor, mBaseTensorMap, *mSharedResources,
        *mSharedResources->cacheManagers[0], *mPipelineIO,
        [this](InferenceDims const& dims, cudaStream_t stream) { return captureBaseGraphWithLoraFanout(dims, stream); },
        [this](cudaStream_t stream) { publishBaseLogits(stream); }};
    PreprocessResources preprocessResources{
        *mStepPreparer, *mEmbeddingPre, mEmbedding, mIdsInput, mDeepstack.get(), mGemma4Ple.get()};
    SamplingBuffers sampling{mSamplingWorkspace, mSamplingIndices, mSamplingScores, mBaseVocabMappingTable,
        mHostPackedTokenIds, mHostSelectedTokenIds};
    LogprobsBuffers logprobs{
        mDeviceLogprobsValues, mDeviceLogprobsIndices, mHostLogprobsValues, mHostLogprobsIndices, mGatheredLogits};
    mDecodingRuntimeContext.reset(new DecodingRuntimeContext{mDeployment, mMaxRuntimeBatchSize, baseResources,
        preprocessResources, *mTokenizer, mLogitBias, sampling, logprobs});
}

void LLMInferenceRuntime::publishBaseLogits(cudaStream_t stream)
{
    if (!mConvertBaseLogits)
    {
        return;
    }

    ELLM_CHECK(mBaseEngineOutputLogits.reshape(mPipelineIO->outputLogits.getShape()),
        "Failed to reshape HALF base-engine logits for FP32 publication.");
    kernel::convertLogitsToFloat(mBaseEngineOutputLogits, mPipelineIO->outputLogits, stream);
}

void LLMInferenceRuntime::setActionNoiseSeed(int32_t seed) noexcept
{
    if (mActionRunner)
    {
        mActionRunner->setNoiseSeed(seed);
    }
}

bool LLMInferenceRuntime::handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response,
    cudaStream_t stream, bool outputThinkerEmbeddings)
{
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

    // Resolve request-level fallback before validation so the constraints describe the decoder that will execute.
    DecodingStrategy& decodingStrategy = mDecoderRegistry->select(request);
    if (!validateRequestConfig(request, decodingStrategy.kind()))
    {
        return false;
    }

    bool const enableSpecDecode = decodingStrategy.isSpeculative();
    if (shouldRejectLogitBiasWithSpecDecode(request, enableSpecDecode))
    {
        LOG_ERROR(
            "logit_bias is not supported while speculative decoding is enabled; set disable_spec_decode=true or use "
            "a vanilla engine.");
        return false;
    }

    if (!validateStreamingSubmission(request))
    {
        return false;
    }

    // Current speculative decoders only support greedy-compatible sampling.
    // Warn here; active spec-decode requests are normalized when context sampling params are populated below.
    bool const hasNonGreedySampling = shouldUseNonGreedySampling(request.temperature, request.topK, request.topP);
    if (enableSpecDecode && hasNonGreedySampling)
    {
        LOG_WARNING("Spec-decode active: overriding sampling params to greedy (ignoring temp/topK/topP).");
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
    ContextCacheStreamGuard contextCacheStreamGuard(context);

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

    if (mContextCache)
    {
        context.contextReuseEnabled = true;
        bool const baseSequenceReuseSupported = context.mediaIdentityComplete
            && (mDeployment.base.numLinearAttnLayers == 0 || mHybridSnapshotStorage != nullptr);
        context.contextCacheLookupEnabled = context.contextReuseEnabled && baseSequenceReuseSupported
            && !outputThinkerEmbeddings && !request.disableContextReuse;
        context.contextCachePublicationEnabled
            = context.contextReuseEnabled && baseSequenceReuseSupported && !request.disableContextReuse;
        context.mediaArtifactReuseEnabled = context.contextReuseEnabled && context.mediaIdentityComplete
            && mMediaArtifactCache != nullptr && !request.disableContextReuse;
        context.contextCacheIsolationKey = request.contextCacheIsolationKey;
        context.contextCacheCommitPolicy = request.contextCacheCommitPolicy;
        context.recurrentCaptureInterval = request.recurrentCaptureInterval;
        context.contextCacheReplayTailLength = request.contextCacheReplayTailLength;
        if (!context.contextCacheLookupEnabled && !request.disableContextReuse)
        {
            LOG_DEBUG("Context cache lookup bypassed for a request path whose exact state identity is not wired yet.");
        }
        if (request.recurrentCaptureInterval > 0
            && (!context.contextCachePublicationEnabled || outputThinkerEmbeddings || context.hasVisionInput
                || context.hasAudioInput))
        {
            LOG_ERROR("Periodic recurrent capture requires a reusable text-only hybrid prefill path.");
            return false;
        }
    }

    // Forward sampling params to context; selected spec-decode requests run greedy.
    context.temperature = enableSpecDecode ? 1.0f : request.temperature;
    context.topP = enableSpecDecode ? 1.0f : request.topP;
    context.topK = enableSpecDecode ? 0 : request.topK;
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

    // The spec-decode path needs extra KV reserve for draft tokens during verification.
    constexpr int32_t kDRAFT_KVCACHE_RESERVE_LENGTH{100};
    int32_t const kvCacheCapacity = enableSpecDecode
        ? std::min(mDeployment.base.maxKVCacheCapacity, mDeployment.draft->maxKVCacheCapacity)
        : mDeployment.base.maxKVCacheCapacity;
    int32_t const kvcReserve = enableSpecDecode ? kDRAFT_KVCACHE_RESERVE_LENGTH : 0;

    // In production, the system-prompt KV cache is saved during warm-up.
    // We disable profiling here to make benchmarking closer to production inference result.
    bool profilingEnabled = getProfilingEnabled();
    if (profilingEnabled)
    {
        setProfilingEnabled(false);
    }

    // Generate system prompt KVCache for each sequence in the batch
    if (request.saveSystemPromptKVCache && !mContextCache)
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

    // Conduct the preparation work to handle a new set of sequences, including inputIds packing, input/output tensor
    // preparation, reset the KVCache state, and apply reused prefix KVCache if available.
    if (!setUpForPrefillExecution(context, decodingStrategy))
    {
        LOG_ERROR("Prefill execution setup failed. This request cannot be handled.");
        return false;
    }

    if (!resolveMultimodalArtifacts(request, context, stream))
    {
        LOG_ERROR("Failed to resolve multimodal encoder artifacts.");
        return false;
    }

    // ── Streaming setup ──────────────────────────────────────────────────────
    // Attach first, record in slotStreams only on success — a throw from attach
    // keeps foreign channels out of the finalizer's reach. Seed sentTokenCount
    // to the prompt length so streaming emits only generated tokens.
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
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

    std::vector<int32_t> kvResidentPrefillLengths = context.effectivePrefillLengths;
    if (context.contextReuseEnabled)
    {
        std::transform(context.rawBatchedInputIds.begin(), context.rawBatchedInputIds.end(),
            kvResidentPrefillLengths.begin(),
            [](std::vector<int32_t> const& inputIds) { return math::cast<int32_t>(inputIds.size()); });
    }
    int32_t const clampedMaxGenerateLength = clampMaxGenerateLengthForKVCapacity(
        kvResidentPrefillLengths, request.maxGenerateLength, kvCacheCapacity, kvcReserve);
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

    bool const hybridMtpContextReuse
        = shouldUseHybridMtpEndpointReuse(decodingStrategy.kind(), mDeployment.base.numLinearAttnLayers > 0,
            context.contextCacheLookupEnabled, context.contextCachePublicationEnabled);

    // Prefill from the base model; subsequent iterations are delegated to the selected strategy. Hybrid MTP recomputes
    // the successor-dependent boundary draft slot from a saved base hidden state, so a checkpoint reuses across turns
    // regardless of the token that follows it.
    bool const prefillStatus = hybridMtpContextReuse ? runHybridMtpPrefill(context, decodingStrategy)
                                                     : runBaseModelPrefillWithPeriodicHybridCaptures(context);
    if (!prefillStatus)
    {
        LOG_ERROR("Failed to execute prefill step for base model.");
        return false;
    }

    if (context.contextCachePublicationEnabled && !hybridMtpContextReuse)
    {
        for (int32_t i = 0; i < context.activeBatchSize; ++i)
        {
            // EAGLE has not run draft prefill yet, so this first commit intentionally publishes base-only state.
            // The first decode step upgrades the same prefill boundary after paired draft state becomes resident.
            publishContextCacheBoundary(
                context, i, PublicationPoint::kPrefillEnd, math::cast<int32_t>(context.rawBatchedInputIds[i].size()));
        }
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
        for (int32_t i = 0; i < context.activeBatchSize; ++i)
        {
            if (context.tokenIds[i].empty())
            {
                continue;
            }
            updateThinkingDoneForToken(i, context.tokenIds[i].back());
        }
    };

    context.shouldStopAfterAcceptedToken = [&](int32_t batchIdx, int32_t tokenId) {
        // Per-token thinking-done check (inline version of updateThinkingDone for a single batch entry).
        if (request.enableThinking && !thinkingDone[batchIdx])
        {
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
        }
        bool isEos = mTokenizer->isEosToken(tokenId);
        if (isEos && request.enableThinking && tokenId != mTokenizer->getEosId() && !thinkingDone[batchIdx])
        {
            isEos = false;
        }
        return isEos || context.currentGenerateLengths[batchIdx] >= context.maxGenerateLength;
    };

    // Few-layer-validation: when EDGELLM_IGNORE_EOS is set, suppress EOS-based
    // termination so the run produces exactly maxGenerateLength tokens, matching
    // the PyTorch golden, which forces a fixed number of decode rounds ignoring
    // EOS. Off by default; only for the numeric-validation run. (Greedy sampling
    // itself is requested separately via the input JSON's top_k=1.)
    bool const ignoreEos = []() {
        char const* v = std::getenv("EDGELLM_IGNORE_EOS");
        return v != nullptr && std::string(v) != "0" && std::string(v) != "false";
    }();
    if (ignoreEos)
    {
        LOG_INFO("EDGELLM_IGNORE_EOS set: ignoring EOS; running to maxGenerateLength.");
    }

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
            // BatchResult.terminalReason → response.finishReasons.
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
    //   cancel → decode (emitDelta + stop match) → finalize (EOS/length/stop) → emit
    applyCancellationToFinishStates(context);
    decodePerSlot(context, *mTokenizer);

    updateThinkingDone();

    updateFinishStates();
    emitChunks(context, *mTokenizer);

    // If everything finished during prefill, evict once so activeBatchSize reaches 0
    if (checkAllFinished() && context.activeBatchSize > 0)
    {
        bool const batchEvictStatus = performBatchEvict(context, decodingStrategy);
        if (!batchEvictStatus)
        {
            LOG_ERROR("Failed to perform batch eviction.");
            return false;
        }
    }

    while (!checkAllFinished())
    {
        // Observe any consumer cancels at the top of the iteration so they land
        // first in the per-slot terminalReason latch.
        applyCancellationToFinishStates(context);

        if (!prepareContextCacheForDecode(context))
        {
            LOG_ERROR("Failed to grow context cache for decode.");
            return false;
        }

        if (!decodingStrategy.decodeStep(context))
        {
            LOG_ERROR("Failed to decode tokens with %s decoding strategy.", decodingStrategy.name());
            return false;
        }

        if (context.contextReuseEnabled && context.generationRound == 0
            && decodingStrategy.kind() == DecodingStrategyKind::kEAGLE)
        {
            // The first EAGLE step has now completed draft prefill. Publish the logical prompt boundary again so a
            // base-only prefill record can gain its coherent draft path, including for kPrefillStateOnly requests.
            for (int32_t i = 0; i < context.activeBatchSize; ++i)
            {
                publishContextCacheBoundary(context, i, PublicationPoint::kPrefillEnd,
                    math::cast<int32_t>(context.rawBatchedInputIds[i].size()));
            }
        }

        // Per-iter pipeline: decode → finalize finish state → emit chunks.
        decodePerSlot(context, *mTokenizer);

        // Update thinking-done state: check if the last generated token is an
        // end-of-thinking marker (<channel|> for Gemma4, </think> for Qwen3/Nemotron).
        updateThinkingDone();

        updateFinishStates();
        emitChunks(context, *mTokenizer);

        emitTokenCallbacks(context);
        context.generationRound += 1;

        // Perform batch eviction if needed (after verification, before updating finish states)
        bool const batchEvictStatus = performBatchEvict(context, decodingStrategy);
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

    // Record metrics - accumulate across all batches (active + evicted)
    int32_t totalReusedTokens = 0;
    int32_t totalComputedTokens = 0;
    int32_t totalGeneratedTokens = 0;
    int32_t totalAcceptedTokens = 0;
    int32_t totalIterations = 0;

    // Accumulate from completed batches
    for (auto const& [originalIdx, batchResult] : context.completedBatches)
    {
        int32_t rawPromptLength = static_cast<int32_t>(batchResult.rawBatchedInputIds.size());
        int32_t computedLength = batchResult.effectivePrefillLength;
        totalReusedTokens += (rawPromptLength - computedLength);
        totalComputedTokens += computedLength;
        totalGeneratedTokens += batchResult.generateLength;
        totalAcceptedTokens += std::max(batchResult.generateLength - 1, 0);
        totalIterations += batchResult.actualIterations;
    }

    mPrefillMetrics.recordRun(totalReusedTokens, totalComputedTokens);
    if (enableSpecDecode)
    {
        mSpecDecodeGenerationMetrics.recordRun(totalIterations, totalGeneratedTokens, totalAcceptedTokens);
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

bool LLMInferenceRuntime::validateRequestConfig(
    LLMGenerationRequest const& request, DecodingStrategyKind selectedStrategy)
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
    if (request.recurrentCaptureInterval < 0)
    {
        LOG_ERROR("recurrent_capture_interval must be non-negative");
        return false;
    }
    if (request.contextCacheReplayTailLength < 0)
    {
        LOG_ERROR("context_cache_replay_tail_length must be non-negative");
        return false;
    }
    bool const requestCacheEnabled = mContextCacheConfig.enabled && !request.disableContextReuse;
    bool const hybridMtpContextReuse = shouldUseHybridMtpEndpointReuse(
        selectedStrategy, mDeployment.base.numLinearAttnLayers > 0, requestCacheEnabled, requestCacheEnabled);
    if (hybridMtpContextReuse
        && (activeBatchSize != 1 || hasAudio || hasVision || request.recurrentCaptureInterval != 0
            || request.contextCacheCommitPolicy != CommitPolicy::kPrefillStateOnly))
    {
        LOG_ERROR(
            "Hybrid MTP context reuse requires a text-only batch of one, endpoint-only capture, and "
            "PREFILL_STATE_ONLY commit policy");
        return false;
    }
    if (request.recurrentCaptureInterval > 0
        && (!mContextCacheConfig.enabled || mDeployment.base.numLinearAttnLayers == 0
            || request.recurrentCaptureInterval % kTOKENS_PER_PAGE != 0))
    {
        LOG_ERROR("recurrent_capture_interval requires production hybrid context reuse and must be a multiple of %d",
            kTOKENS_PER_PAGE);
        return false;
    }
    if (request.recurrentCaptureInterval > 0
        && (activeBatchSize != 1 || hasAudio || hasVision || mHybridSnapshotStorage == nullptr
            || mHybridSnapshotStorage->recurrentSlotCount() == 0))
    {
        LOG_ERROR(
            "recurrent_capture_interval currently requires a text-only batch of one and retained recurrent slots");
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
    context.hasAudioInput = hasAudio;
    context.hasVisionInput = hasVision;

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

    if (mContextCache != nullptr && (hasAudio || hasVision))
    {
        std::vector<int64_t> const visionRows
            = mVisionRunner ? mVisionRunner->preparedArtifactRowCounts() : std::vector<int64_t>{};
        std::vector<int64_t> const audioRows
            = mAudioRunner ? mAudioRunner->preparedArtifactRowCounts() : std::vector<int64_t>{};
        context.mediaIdentityComplete
            = populateMediaSpans(request, mDeployment.base, batchedInputIds, visionRows, audioRows, context);
    }

    // Populate system prompts and raw input IDs from batchedInputIds
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        context.systemPrompts[i] = request.formattedRequests[i].formattedSystemPrompt;
        context.rawBatchedInputIds.push_back(batchedInputIds[i]);
    }

    return true;
}

bool LLMInferenceRuntime::resolveMultimodalArtifacts(
    LLMGenerationRequest const& request, DecodingInferenceContext& context, cudaStream_t stream)
{
    if (!context.hasVisionInput && !context.hasAudioInput)
    {
        return true;
    }

    auto reuseLength = [&](int32_t slot) {
        if (!context.contextReuseEnabled || !context.sequenceCacheStates[slot].has_value())
        {
            return 0;
        }
        return context.sequenceCacheStates[slot]->lease.reuseTokenLength();
    };
    auto firstArtifactRow = [&](int32_t slot, MediaModality modality) {
        int32_t const reusedTokens = reuseLength(slot);
        for (MediaSpanDescriptor const& span : context.mediaSpans[slot])
        {
            if (span.artifactKey.modality != modality)
            {
                continue;
            }
            int32_t const consumedRows = mediaRowsBeforeToken(span, reusedTokens);
            if (consumedRows < span.tokenLength)
            {
                return span.embeddingOffset + consumedRows;
            }
        }
        return 0;
    };
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        context.visionEmbeddingOffsets[slot] = firstArtifactRow(slot, MediaModality::kVision);
        context.audioEmbeddingOffsets[slot] = firstArtifactRow(slot, MediaModality::kAudio);
    }

    struct ArtifactSpanRef
    {
        MediaSpanDescriptor const* span{};
        int32_t slot{};
        size_t originalItemIndex{};
    };
    struct ArtifactSource
    {
        ArtifactSpanRef ref;
        Tensor const* embedding{};
        std::vector<Tensor const*> deepstack;
        std::optional<MediaArtifactLease> lease;
    };

    auto resolve = [&](MediaModality modality, MultimodalRunner& runner, Tensor& assembly,
                       std::vector<Tensor>& deepstackAssembly, OptionalInputTensor& output,
                       OptionalInputTensors* deepstackOutput) -> bool {
        std::vector<ArtifactSpanRef> allSpans;
        std::vector<ArtifactSpanRef> neededSpans;
        int32_t totalRows{};
        size_t originalItemIndex{};
        for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
        {
            int32_t const reusedTokens = reuseLength(slot);
            for (MediaSpanDescriptor const& span : context.mediaSpans[slot])
            {
                if (span.artifactKey.modality != modality)
                {
                    continue;
                }
                ArtifactSpanRef const ref{&span, slot, originalItemIndex++};
                allSpans.push_back(ref);
                totalRows = std::max(totalRows, span.embeddingOffset + span.tokenLength);
                if (mediaRowsBeforeToken(span, reusedTokens) < span.tokenLength)
                {
                    neededSpans.push_back(ref);
                }
            }
        }

        if (!context.mediaIdentityComplete || allSpans.empty())
        {
            if (!runner.infer(stream))
            {
                return false;
            }
            output = std::cref(runner.getOutputEmbedding());
            if (deepstackOutput != nullptr)
            {
                *deepstackOutput = runner.getDeepstackFeatures();
            }
            return true;
        }
        if (neededSpans.empty())
        {
            runner.discardPreparedInput();
            output = std::nullopt;
            if (deepstackOutput != nullptr)
            {
                deepstackOutput->clear();
            }
            LOG_INFO("Context reuse skipped the %s encoder: every media span is resident in sequence state.",
                modality == MediaModality::kVision ? "vision" : "audio");
            return true;
        }

        size_t const expectedDeepstack
            = modality == MediaModality::kVision ? static_cast<size_t>(mDeployment.base.numDeepstackFeatures) : 0U;
        std::vector<ArtifactSource> sources;
        std::vector<ArtifactSpanRef> misses;
        sources.reserve(neededSpans.size());
        misses.reserve(neededSpans.size());
        for (ArtifactSpanRef const& ref : neededSpans)
        {
            std::optional<MediaArtifactLease> lease = context.mediaArtifactReuseEnabled
                ? mMediaArtifactCache->acquire(ref.span->artifactKey, stream)
                : std::nullopt;
            if (!lease.has_value())
            {
                misses.push_back(ref);
                continue;
            }
            std::vector<Tensor const*> features = lease->deepstackFeatures();
            Coords const& shape = lease->embedding().getShape();
            bool schemaMatches = shape.getNumDims() == 2 && shape[0] == ref.span->tokenLength
                && shape[1] == mDeployment.base.hiddenSize && features.size() == expectedDeepstack;
            schemaMatches = schemaMatches && std::all_of(features.begin(), features.end(), [&](Tensor const* feature) {
                return feature != nullptr && feature->getShape().getNumDims() == 2
                    && feature->getShape()[0] == ref.span->tokenLength;
            });
            if (!schemaMatches)
            {
                misses.push_back(ref);
                continue;
            }
            ArtifactSource source;
            source.ref = ref;
            source.lease.emplace(std::move(*lease));
            source.embedding = &source.lease->embedding();
            source.deepstack = std::move(features);
            sources.push_back(std::move(source));
        }

        bool const selectiveExecution = !sources.empty() || neededSpans.size() != allSpans.size();
        if (!misses.empty() && selectiveExecution)
        {
            LLMGenerationRequest filteredRequest{};
            filteredRequest.requests.resize(request.requests.size());
            std::vector<size_t> selectedOriginalIndices;
            selectedOriginalIndices.reserve(misses.size());
            for (ArtifactSpanRef const& miss : misses)
            {
                ELLM_CHECK(miss.slot >= 0 && static_cast<size_t>(miss.slot) < request.requests.size(),
                    "Selected media artifact slot is out of range");
                size_t const itemOrder = static_cast<size_t>(miss.span->itemOrder);
                if (modality == MediaModality::kVision)
                {
                    ELLM_CHECK(itemOrder < request.requests[miss.slot].imageBuffers.size(),
                        "Selected vision artifact index is out of range");
                    filteredRequest.requests[miss.slot].imageBuffers.push_back(
                        request.requests[miss.slot].imageBuffers[itemOrder]);
                }
                else
                {
                    ELLM_CHECK(itemOrder < request.requests[miss.slot].audioBuffers.size(),
                        "Selected audio artifact index is out of range");
                    filteredRequest.requests[miss.slot].audioBuffers.push_back(
                        request.requests[miss.slot].audioBuffers[itemOrder]);
                }
                selectedOriginalIndices.push_back(miss.originalItemIndex);
            }
            if (!runner.prepareArtifactSubset(filteredRequest, selectedOriginalIndices, stream))
            {
                return false;
            }
        }

        Tensor* encoderOutput{};
        OptionalInputTensors encoderDeepstack;
        if (!misses.empty())
        {
            if (!runner.infer(stream))
            {
                return false;
            }
            encoderOutput = &runner.getOutputEmbedding();
            encoderDeepstack = deepstackOutput != nullptr ? runner.getDeepstackFeatures() : OptionalInputTensors{};
            int32_t encodedRows{};
            for (ArtifactSpanRef const& miss : misses)
            {
                encodedRows += miss.span->tokenLength;
            }
            int32_t const expectedRows = selectiveExecution ? encodedRows : totalRows;
            ELLM_CHECK(encoderOutput->getShape().getNumDims() == 2 && encoderOutput->getShape()[0] == expectedRows
                    && encoderOutput->getShape()[1] == mDeployment.base.hiddenSize,
                "Media encoder output does not match the selected artifact layout");
            ELLM_CHECK(encoderDeepstack.size() == expectedDeepstack,
                "Media encoder deepstack output count does not match the deployment");
        }

        auto sliceRows = [](Tensor const& tensor, int32_t offset, int32_t length, std::string const& name) {
            ELLM_CHECK(tensor.getShape().getNumDims() == 2 && offset >= 0 && length > 0
                    && tensor.getShape()[0] >= offset + length,
                "Media artifact slice exceeds encoder output");
            int64_t const hidden = tensor.getShape()[1];
            size_t const rowBytes = static_cast<size_t>(hidden) * utils::getTypeSize(tensor.getDataType());
            auto* data = static_cast<std::byte*>(const_cast<void*>(tensor.rawPointer()))
                + static_cast<size_t>(offset) * rowBytes;
            return Tensor(data, {length, hidden}, tensor.getDeviceType(), tensor.getDataType(), name);
        };

        std::vector<Tensor> missEmbeddingSlices;
        std::vector<std::vector<Tensor>> missDeepstackSlices;
        missEmbeddingSlices.reserve(misses.size());
        missDeepstackSlices.reserve(misses.size());
        int32_t packedMissOffset{};
        for (ArtifactSpanRef const& miss : misses)
        {
            int32_t const sourceOffset = selectiveExecution ? packedMissOffset : miss.span->embeddingOffset;
            missEmbeddingSlices.push_back(
                sliceRows(*encoderOutput, sourceOffset, miss.span->tokenLength, "MediaArtifactCache::source"));
            missDeepstackSlices.emplace_back();
            std::vector<Tensor>& featureSlices = missDeepstackSlices.back();
            featureSlices.reserve(encoderDeepstack.size());
            std::vector<Tensor const*> featurePointers;
            featurePointers.reserve(encoderDeepstack.size());
            for (auto const& feature : encoderDeepstack)
            {
                featureSlices.push_back(sliceRows(
                    feature.get(), sourceOffset, miss.span->tokenLength, "MediaArtifactCache::deepstackSource"));
                featurePointers.push_back(&featureSlices.back());
            }

            ArtifactSource source;
            source.ref = miss;
            source.embedding = &missEmbeddingSlices.back();
            source.deepstack = featurePointers;
            sources.push_back(std::move(source));
            if (context.mediaArtifactReuseEnabled)
            {
                std::optional<MediaArtifactLease> inserted = mMediaArtifactCache->insert(
                    miss.span->artifactKey, missEmbeddingSlices.back(), featurePointers, stream);
                if (inserted.has_value())
                {
                    context.mediaArtifactLeases.push_back(std::move(*inserted));
                }
            }
            packedMissOffset += miss.span->tokenLength;
        }

        if (!selectiveExecution)
        {
            output = std::cref(*encoderOutput);
            if (deepstackOutput != nullptr)
            {
                *deepstackOutput = encoderDeepstack;
            }
            return true;
        }

        auto assemble = [&](Tensor& destination, auto select, std::string const& name) -> Tensor const& {
            ELLM_CHECK(!sources.empty(), "Media artifact assembly has no sources");
            Tensor const& exemplar = select(sources.front());
            Coords const& shape = exemplar.getShape();
            ELLM_CHECK(shape.getNumDims() == 2, "Media artifact must be a row-major matrix");
            int64_t const hidden = shape[1];
            size_t const rowBytes = static_cast<size_t>(hidden) * utils::getTypeSize(exemplar.getDataType());
            size_t const requiredBytes = static_cast<size_t>(totalRows) * rowBytes;
            if (destination.isEmpty() || static_cast<size_t>(destination.getMemoryCapacity()) < requiredBytes
                || destination.getDataType() != exemplar.getDataType())
            {
                destination = Tensor({totalRows, hidden}, DeviceType::kGPU, exemplar.getDataType(), name);
            }
            check::check(destination.reshape({totalRows, hidden}), "Media artifact assembly reshape failed");
            CUDA_CHECK(cudaMemsetAsync(destination.rawPointer(), 0, requiredBytes, stream));
            auto* destinationBytes = static_cast<std::byte*>(destination.rawPointer());
            for (ArtifactSource const& source : sources)
            {
                Tensor const& sourceTensor = select(source);
                ELLM_CHECK(sourceTensor.getShape().getNumDims() == 2
                        && sourceTensor.getShape()[0] == source.ref.span->tokenLength
                        && sourceTensor.getShape()[1] == hidden && sourceTensor.getDataType() == exemplar.getDataType(),
                    "Media artifact schema mismatch");
                CUDA_CHECK(
                    cudaMemcpyAsync(destinationBytes + static_cast<size_t>(source.ref.span->embeddingOffset) * rowBytes,
                        sourceTensor.rawPointer(), static_cast<size_t>(source.ref.span->tokenLength) * rowBytes,
                        cudaMemcpyDeviceToDevice, stream));
            }
            return destination;
        };

        output = std::cref(assemble(
            assembly, [](ArtifactSource const& source) -> Tensor const& { return *source.embedding; },
            modality == MediaModality::kVision ? "LLMInferenceRuntime::visionArtifactAssembly"
                                               : "LLMInferenceRuntime::audioArtifactAssembly"));
        if (deepstackOutput != nullptr)
        {
            deepstackAssembly.resize(expectedDeepstack);
            deepstackOutput->clear();
            for (size_t level = 0; level < expectedDeepstack; ++level)
            {
                deepstackOutput->push_back(std::cref(assemble(
                    deepstackAssembly[level],
                    [level](ArtifactSource const& source) -> Tensor const& { return *source.deepstack[level]; },
                    "LLMInferenceRuntime::visionDeepstackArtifactAssembly")));
            }
        }
        size_t artifactHits{};
        for (ArtifactSource& source : sources)
        {
            if (source.lease.has_value())
            {
                context.mediaArtifactLeases.push_back(std::move(*source.lease));
                ++artifactHits;
            }
        }
        if (misses.empty())
        {
            runner.discardPreparedInput();
        }
        LOG_INFO("Resolved %s artifacts with %zu hit(s) and %zu encoder miss(es).",
            modality == MediaModality::kVision ? "vision" : "audio", artifactHits, misses.size());
        return true;
    };

    if (context.hasAudioInput)
    {
        ELLM_CHECK(mAudioRunner != nullptr, "Audio input has no audio runner");
        if (!resolve(MediaModality::kAudio, *mAudioRunner, mAudioArtifactAssembly, mVisionDeepstackArtifactAssembly,
                context.audioEmbeddings, nullptr))
        {
            LOG_ERROR("Audio inference failed. This request cannot be handled.");
            return false;
        }
    }
    if (context.hasVisionInput)
    {
        ELLM_CHECK(mVisionRunner != nullptr, "Vision input has no vision runner");
        if (!resolve(MediaModality::kVision, *mVisionRunner, mVisionArtifactAssembly, mVisionDeepstackArtifactAssembly,
                context.visualEmbeddings, &context.deepstackFeatures))
        {
            LOG_ERROR("Vision inference failed. This request cannot be handled.");
            return false;
        }
    }
    return true;
}

bool LLMInferenceRuntime::runBaseModelPrefill(DecodingInferenceContext& context, bool sampleOutput)
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
    check::check(mPipelineIO->outputLogits.reshape({activeBatchSize, mDeployment.base.outputVocabSize}),
        "Tensor reshape failed");
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
        hostCtxLenData[i] = context.effectivePrefillLengths[i];
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
    std::vector<int32_t> const emptyMediaOffsets;
    bool const useMediaOffsets = context.contextReuseEnabled && context.mediaIdentityComplete
        && (context.hasVisionInput || context.hasAudioInput);
    mEmbeddingPre->embed(mIdsInput, context.visualEmbeddings, context.audioEmbeddings, *mPipelineIO, context.stream,
        useMediaOffsets ? context.visionEmbeddingOffsets : emptyMediaOffsets,
        useMediaOffsets ? context.audioEmbeddingOffsets : emptyMediaOffsets);
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
    check::check(
        mDecodingRuntimeContext->base.execute(context.stream), "Failed to execute base model for prefill step.");
    mSharedResources->cacheManagers[0]->commitSequenceLength(mPipelineIO->contextLengths, context.stream);

    applyLogitBias(mLogitBias, mPipelineIO->outputLogits, context, context.stream);

    if (!sampleOutput)
    {
        return true;
    }

    // Sampling from the prefill stage logits follows the same policy as vanilla decoding.
    // Speculative decoders reach this code with greedy-compatible context params because
    // handleRequest normalizes active spec-decode requests before decoding.
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

bool LLMInferenceRuntime::runBaseModelPrefillWithPeriodicHybridCaptures(DecodingInferenceContext& context)
{
    int32_t const interval = context.recurrentCaptureInterval;
    if (interval == 0)
    {
        return runBaseModelPrefill(context);
    }
    if (context.activeBatchSize != 1)
    {
        LOG_ERROR("Periodic recurrent capture currently requires batch size 1.");
        return false;
    }
    ELLM_CHECK(context.sequenceCacheStates[0].has_value() && context.sequenceCacheStates[0]->hybridCheckpoint,
        "Periodic recurrent capture requires a hybrid context-cache lease");

    std::vector<int32_t> const completeSuffix = context.tokenIds[0];
    int32_t const originalComputedLength = context.effectivePrefillLengths[0];
    int32_t processedLength = context.sequenceCacheStates[0]->lease.reuseTokenLength();
    int32_t const inputLength = math::cast<int32_t>(context.rawBatchedInputIds[0].size());
    ELLM_CHECK(processedLength + originalComputedLength == inputLength,
        "Hybrid periodic capture suffix does not cover the logical input");

    size_t suffixOffset{};
    int32_t nextBoundary = ((processedLength / interval) + 1) * interval;
    while (nextBoundary < inputLength)
    {
        int32_t const chunkLength = nextBoundary - processedLength;
        ELLM_CHECK(chunkLength > 0 && suffixOffset + static_cast<size_t>(chunkLength) <= completeSuffix.size(),
            "Hybrid periodic capture produced an invalid prefill chunk");
        context.tokenIds[0].assign(completeSuffix.begin() + static_cast<std::ptrdiff_t>(suffixOffset),
            completeSuffix.begin() + static_cast<std::ptrdiff_t>(suffixOffset + static_cast<size_t>(chunkLength)));
        context.effectivePrefillLengths[0] = chunkLength;
        if (!runBaseModelPrefill(context, /*sampleOutput=*/false))
        {
            return false;
        }
        processedLength = nextBoundary;
        suffixOffset += static_cast<size_t>(chunkLength);
        publishContextCacheBoundary(context, 0, PublicationPoint::kPrefillEnd, processedLength);
        nextBoundary += interval;
    }

    context.tokenIds[0].assign(
        completeSuffix.begin() + static_cast<std::ptrdiff_t>(suffixOffset), completeSuffix.end());
    context.effectivePrefillLengths[0] = inputLength - processedLength;
    ELLM_CHECK(
        context.effectivePrefillLengths[0] > 0, "Hybrid periodic capture must leave a non-empty final prefill chunk");
    if (!runBaseModelPrefill(context))
    {
        return false;
    }
    int32_t const sampledToken = context.tokenIds[0].back();
    context.tokenIds[0] = completeSuffix;
    context.tokenIds[0].push_back(sampledToken);
    context.effectivePrefillLengths[0] = originalComputedLength;
    return true;
}

bool LLMInferenceRuntime::runHybridMtpPrefill(DecodingInferenceContext& context, DecodingStrategy& strategy)
{
    ELLM_CHECK(context.activeBatchSize == 1 && context.sequenceCacheStates[0].has_value()
            && context.sequenceCacheStates[0]->hybridMtp,
        "Hybrid MTP endpoint prefill requires one combined cache sequence");

    SequenceCacheState& state = *context.sequenceCacheStates[0];
    int32_t const reuseLength = state.lease.reuseTokenLength();
    int32_t const inputLength = math::cast<int32_t>(context.rawBatchedInputIds[0].size());
    int32_t const replayTailLength = context.contextCacheReplayTailLength;
    int32_t const suffixLenOrig = context.effectivePrefillLengths[0];
    LOG_DEBUG("Hybrid+MTP prefill: inputLength=%d reuseLength=%d basePrefill=%d", inputLength, reuseLength,
        inputLength - reuseLength);

    // Cold sequence with a volatile generation-prompt tail: publish the checkpoint at the STABLE boundary
    // predecessorLength = inputLength - replayTailLength (two-chunk prefill), not at the full inputLength. The server
    // flags the replay tail (e.g. the reasoning-off <think></think> block) via contextCacheReplayTailLength because a
    // later turn holds real content there; baking it into the exact-prefix key makes every future lookup miss.
    // Mirrors baseline runHybridMtpPrefillWithStableEndpoint, but publishes with the successor-drop signature.
    if (reuseLength == 0 && state.publicationEnabled && replayTailLength > 0 && suffixLenOrig > replayTailLength)
    {
        std::vector<int32_t> const completeSuffix = context.tokenIds[0];
        int32_t const predecessorLength = inputLength - replayTailLength;
        int32_t const predecessorChunkLength = suffixLenOrig - replayTailLength;
        auto const replayBegin = completeSuffix.end() - replayTailLength;

        // Chunk 1: base prefill of the predecessor [0, predecessorLength), no sampling. Leaves the recurrent state at
        // the stable boundary so captureRecurrent snapshots it there.
        context.tokenIds[0].assign(completeSuffix.begin(), replayBegin);
        context.effectivePrefillLengths[0] = predecessorChunkLength;
        if (!runBaseModelPrefill(context, /*sampleOutput=*/false))
        {
            return false;
        }
        // Non-sampling chunk must sync before MTP repacks the shared pinned token buffer.
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        // Provide the boundary's next token so the predecessor draft prefill covers the boundary slot, then publish.
        context.tokenIds[0].push_back(*replayBegin);
        if (!strategy.prepareFirstDecodeStep(context))
        {
            return false;
        }
        publishHybridMtpContextCacheBoundary(context, 0, PublicationPoint::kPrefillEnd, predecessorLength);

        // Chunk 2: replay the volatile tail [predecessorLength, inputLength) with sampling, then restore bookkeeping.
        context.speculativeDraftPrefillComplete = false;
        context.tokenIds[0].assign(replayBegin, completeSuffix.end());
        context.effectivePrefillLengths[0] = replayTailLength;
        if (!runBaseModelPrefill(context))
        {
            return false;
        }
        int32_t const sampledToken = context.tokenIds[0].back();
        if (!strategy.prepareFirstDecodeStep(context))
        {
            return false;
        }
        context.tokenIds[0] = completeSuffix;
        context.tokenIds[0].push_back(sampledToken);
        context.effectivePrefillLengths[0] = suffixLenOrig;
        return true;
    }

    // Hit sequence with a volatile tail: publish at the stable predecessorLength like the cold path, but the reused
    // checkpoint's boundary draft slot must first be reconstructed (the consume-side fold) into the predecessor chunk's
    // draft prefill. Two-chunk so the recurrent snapshot + boundary hidden land on the stable boundary.
    if (reuseLength > 0 && state.publicationEnabled && replayTailLength > 0 && suffixLenOrig > replayTailLength)
    {
        ELLM_CHECK(state.lease.recurrentSnapshotSlot().has_value(),
            "Hybrid MTP cache hit is missing its recurrent snapshot slot");
        std::vector<int32_t> const completeSuffix = context.tokenIds[0];
        int32_t const predecessorLength = inputLength - replayTailLength;
        int32_t const predecessorChunkLength = suffixLenOrig - replayTailLength;
        auto const replayBegin = completeSuffix.end() - replayTailLength;
        int32_t const boundaryHiddenDim = mDeployment.specConfig->baseOutputHiddenDim;
        size_t const rowBytes
            = static_cast<size_t>(boundaryHiddenDim) * utils::getTypeSize(mPipelineIO->baseHiddenStates.getDataType());

        // Chunk 1: base prefill of the predecessor [reuseLength, predecessorLength), no sampling.
        context.tokenIds[0].assign(completeSuffix.begin(), replayBegin);
        context.effectivePrefillLengths[0] = predecessorChunkLength;
        if (!runBaseModelPrefill(context, /*sampleOutput=*/false))
        {
            return false;
        }
        CUDA_CHECK(cudaStreamSynchronize(context.stream));

        // Fold the reused checkpoint's boundary hidden into the predecessor draft prefill (shift down one row, restore
        // boundary hidden at row 0), reconstructing the reused draft slot [reuseLength-1].
        check::check(mPipelineIO->baseHiddenStates.reshape({1, predecessorChunkLength + 1, boundaryHiddenDim}),
            "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mBoundaryFoldScratch.rawPointer(), mPipelineIO->baseHiddenStates.rawPointer(),
            static_cast<size_t>(predecessorChunkLength) * rowBytes, cudaMemcpyDeviceToDevice, context.stream));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(mPipelineIO->baseHiddenStates.rawPointer()) + rowBytes,
            mBoundaryFoldScratch.rawPointer(), static_cast<size_t>(predecessorChunkLength) * rowBytes,
            cudaMemcpyDeviceToDevice, context.stream));
        mHybridSnapshotStorage->restoreBoundaryHidden(
            *state.lease.recurrentSnapshotSlot(), mPipelineIO->baseHiddenStates, 0, 0, context.stream);

        int32_t const boundaryToken = context.rawBatchedInputIds[0][static_cast<size_t>(reuseLength - 1)];
        context.tokenIds[0].insert(context.tokenIds[0].begin(), boundaryToken);
        context.effectivePrefillLengths[0] = predecessorChunkLength + 1;
        if (!strategy.prepareFirstDecodeStep(context))
        {
            return false;
        }
        publishHybridMtpContextCacheBoundary(context, 0, PublicationPoint::kPrefillEnd, predecessorLength);
        context.tokenIds[0].erase(context.tokenIds[0].begin());
        context.effectivePrefillLengths[0] = predecessorChunkLength;

        // Chunk 2: replay the volatile tail [predecessorLength, inputLength) with sampling, then restore bookkeeping.
        context.speculativeDraftPrefillComplete = false;
        context.tokenIds[0].assign(replayBegin, completeSuffix.end());
        context.effectivePrefillLengths[0] = replayTailLength;
        if (!runBaseModelPrefill(context))
        {
            return false;
        }
        int32_t const sampledToken = context.tokenIds[0].back();
        if (!strategy.prepareFirstDecodeStep(context))
        {
            return false;
        }
        context.tokenIds[0] = completeSuffix;
        context.tokenIds[0].push_back(sampledToken);
        context.effectivePrefillLengths[0] = suffixLenOrig;
        return true;
    }

    // Base prefill of the suffix (the full prompt when cold). Reuses base KV [0, reuseLength) and samples the first
    // output token, appending it to tokenIds.
    if (!runBaseModelPrefill(context))
    {
        return false;
    }

    if (reuseLength == 0)
    {
        // Cold sequence: no reused checkpoint boundary to fold in.
        if (!strategy.prepareFirstDecodeStep(context))
        {
            return false;
        }
        publishHybridMtpContextCacheBoundary(context, 0, PublicationPoint::kPrefillEnd, inputLength);
        return true;
    }

    // On a hit the restore path reused draft KV only up to reuseLength-1, because that final reused draft slot depends
    // on the token that followed the checkpoint. The MTP draft prefill does not write its last (query) position's KV,
    // so that boundary slot cannot be recomputed by a standalone one-token pass (it would write nothing). Instead fold
    // the boundary into the suffix draft prefill as a *context* position: prepend the checkpoint's saved base boundary
    // hidden state (base_hidden[reuseLength-1]) and token[reuseLength-1] so the single draft prefill covers
    // [reuseLength-1, inputLength) and writes a correct, successor-aware boundary KV.
    ELLM_CHECK(
        state.lease.recurrentSnapshotSlot().has_value(), "Hybrid MTP cache hit is missing its recurrent snapshot slot");
    int32_t const boundaryHiddenDim = mDeployment.specConfig->baseOutputHiddenDim;
    int32_t const suffixLen = context.effectivePrefillLengths[0];
    size_t const rowBytes
        = static_cast<size_t>(boundaryHiddenDim) * utils::getTypeSize(mPipelineIO->baseHiddenStates.getDataType());

    // Shift the base suffix hidden states [0, suffixLen) down one row and place the restored boundary hidden state at
    // row 0, giving the draft [base_hidden[reuseLength-1], base_hidden[reuseLength..inputLength)].
    check::check(mPipelineIO->baseHiddenStates.reshape({1, suffixLen + 1, boundaryHiddenDim}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mBoundaryFoldScratch.rawPointer(), mPipelineIO->baseHiddenStates.rawPointer(),
        static_cast<size_t>(suffixLen) * rowBytes, cudaMemcpyDeviceToDevice, context.stream));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(mPipelineIO->baseHiddenStates.rawPointer()) + rowBytes,
        mBoundaryFoldScratch.rawPointer(), static_cast<size_t>(suffixLen) * rowBytes, cudaMemcpyDeviceToDevice,
        context.stream));
    mHybridSnapshotStorage->restoreBoundaryHidden(
        *state.lease.recurrentSnapshotSlot(), mPipelineIO->baseHiddenStates, 0, 0, context.stream);

    // Prepend token[reuseLength-1] to the execution tokens and extend the prefill length by one. runDraftModelPrefill
    // pairs draft slot k with (baseHiddenStates[k], tokenIds[k+1]); with the shifted hidden states and this prepend,
    // draft slot reuseLength-1+k consumes (base_hidden[reuseLength-1+k], token[reuseLength+k]) as required.
    int32_t const boundaryToken = context.rawBatchedInputIds[0][static_cast<size_t>(reuseLength - 1)];
    context.tokenIds[0].insert(context.tokenIds[0].begin(), boundaryToken);
    context.effectivePrefillLengths[0] = suffixLen + 1;
    if (!strategy.prepareFirstDecodeStep(context))
    {
        return false;
    }
    // Publish while effectivePrefillLengths still reflects the folded prefill (the boundary-hidden capture reads the
    // last shifted row), then restore the suffix-only execution bookkeeping for the decode loop.
    publishHybridMtpContextCacheBoundary(context, 0, PublicationPoint::kPrefillEnd, inputLength);
    context.tokenIds[0].erase(context.tokenIds[0].begin());
    context.effectivePrefillLengths[0] = suffixLen;
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

bool LLMInferenceRuntime::setUpForPrefillExecution(DecodingInferenceContext& context, DecodingStrategy& strategy)
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

    // Record the length of the reused KVCache for each sequence.
    check::check(mHostReuseKVCacheLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* reuseKVCacheLengthsData = mHostReuseKVCacheLengths.dataPointer<int32_t>();
    std::fill(reuseKVCacheLengthsData, reuseKVCacheLengthsData + activeBatchSize, 0);
    // Draft (speculative) KV reuse can differ from base reuse: Hybrid+MTP reuses one fewer draft token so the
    // successor-dependent boundary slot is recomputed. Defaults to the base reuse length for all other paths.
    check::check(mHostDraftReuseKVCacheLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* draftReuseKVCacheLengthsData = mHostDraftReuseKVCacheLengths.dataPointer<int32_t>();
    std::fill(draftReuseKVCacheLengthsData, draftReuseKVCacheLengthsData + activeBatchSize, 0);

    context.tokenIds.clear();
    context.tokenIds.resize(activeBatchSize);

    if (context.contextReuseEnabled)
    {
        BlockKeyExtras commonExtras;
        if (!context.loraWeightsName.empty())
        {
            ELLM_CHECK(mSharedResources->loraManager != nullptr,
                "Context cache request selected a LoRA adapter without a LoRA manager");
            commonExtras.adapter = AdapterKey{hashOpaqueIdentity(context.loraWeightsName),
                mSharedResources->loraManager->getAdapterGeneration(context.loraWeightsName)};
        }
        if (!context.contextCacheIsolationKey.empty())
        {
            commonExtras.isolationDigest = hashOpaqueIdentity(context.contextCacheIsolationKey);
        }

        LookupPolicy const lookupPolicy
            = context.contextCacheLookupEnabled ? LookupPolicy::kUseCache : LookupPolicy::kBypass;
        KVPageTable& basePageTable = *mSharedResources->kvPageTables[0];
        bool const specEagle = strategy.kind() == DecodingStrategyKind::kEAGLE;
        bool const hybridMtp
            = strategy.kind() == DecodingStrategyKind::kMTP && mDeployment.base.numLinearAttnLayers > 0;
        bool const hybridCheckpoint = !specEagle && mDeployment.base.numLinearAttnLayers > 0;
        KVPageTable* draftPageTable = specEagle || hybridMtp ? mSharedResources->kvPageTables[1].get() : nullptr;
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            std::vector<int32_t> const& inputIds = batchedInputIds[i];
            RuntimeCacheAcquireResult acquired;
            size_t const fullBlockCount = inputIds.size() / static_cast<size_t>(kTOKENS_PER_PAGE);
            std::vector<BlockKeyExtras> const fullBlockExtras = buildBlockExtras(commonExtras, context.mediaSpans[i],
                fullBlockCount * static_cast<size_t>(kTOKENS_PER_PAGE), kTOKENS_PER_PAGE);
            std::vector<BlockHash> const fullBlockHashes
                = hashFullBlocks(inputIds.data(), inputIds.size(), kTOKENS_PER_PAGE, fullBlockExtras);
            if (specEagle)
            {
                ELLM_CHECK(mDraftEngineSignature.has_value() && draftPageTable != nullptr,
                    "EAGLE context reuse has no draft identity or page table");
                acquired = mContextCache->acquireSpec(SpecDecodeMode::kEAGLE, mContextCacheDomain,
                    *mDraftEngineSignature, fullBlockHashes, math::cast<int32_t>(inputIds.size()), lookupPolicy);
            }
            else if (hybridMtp)
            {
                ELLM_CHECK(mDraftEngineSignature.has_value() && mRecurrentStateSchema.has_value()
                        && mHybridSnapshotStorage != nullptr && draftPageTable != nullptr,
                    "Hybrid MTP context reuse has no draft identity, snapshot schema, or page table");
                std::vector<int32_t> const candidateLengths
                    = mContextCache->hybridMtpCandidateLengths(mContextCacheDomain, *mRecurrentStateSchema,
                        *mDraftEngineSignature, math::cast<int32_t>(inputIds.size()));
                std::vector<HybridMtpCheckpointCandidate> candidates;
                candidates.reserve(candidateLengths.size());
                for (int32_t const length : candidateLengths)
                {
                    ELLM_CHECK(length >= 0 && static_cast<size_t>(length) < inputIds.size(),
                        "Hybrid MTP candidate is not an interior endpoint");
                    std::vector<BlockKeyExtras> const candidateExtras = buildBlockExtras(
                        commonExtras, context.mediaSpans[i], static_cast<size_t>(length), kTOKENS_PER_PAGE);
                    candidates.push_back(HybridMtpCheckpointCandidate{length,
                        hashExactPrefix(
                            inputIds.data(), static_cast<size_t>(length), kTOKENS_PER_PAGE, candidateExtras)});
                }
                acquired = mContextCache->acquireHybridMtp(mContextCacheDomain, candidates, fullBlockHashes,
                    math::cast<int32_t>(inputIds.size()), *mRecurrentStateSchema, *mDraftEngineSignature, lookupPolicy);
            }
            else if (hybridCheckpoint)
            {
                ELLM_CHECK(mRecurrentStateSchema.has_value() && mHybridSnapshotStorage != nullptr,
                    "Hybrid context reuse has no recurrent snapshot schema");
                std::vector<int32_t> const candidateLengths = mContextCache->hybridCandidateLengths(
                    mContextCacheDomain, *mRecurrentStateSchema, math::cast<int32_t>(inputIds.size()));
                std::vector<HybridCheckpointCandidate> candidates;
                candidates.reserve(candidateLengths.size());
                for (int32_t const length : candidateLengths)
                {
                    std::vector<BlockKeyExtras> const candidateExtras = buildBlockExtras(
                        commonExtras, context.mediaSpans[i], static_cast<size_t>(length), kTOKENS_PER_PAGE);
                    candidates.push_back(HybridCheckpointCandidate{length,
                        hashExactPrefix(
                            inputIds.data(), static_cast<size_t>(length), kTOKENS_PER_PAGE, candidateExtras)});
                }
                acquired = mContextCache->acquireHybrid(mContextCacheDomain, candidates, fullBlockHashes,
                    math::cast<int32_t>(inputIds.size()), *mRecurrentStateSchema,
                    !mDeployment.base.kvLayerConfigs.empty(), lookupPolicy);
            }
            else
            {
                acquired = mContextCache->acquireVanilla(
                    mContextCacheDomain, fullBlockHashes, math::cast<int32_t>(inputIds.size()), lookupPolicy);
            }
            if (acquired.forcedCold)
            {
                LOG_DEBUG("Context cache slot %d retried as cold after cached-prefix capacity pressure (status=%d).", i,
                    static_cast<int32_t>(acquired.status));
            }
            if (acquired.status != AcquireStatus::kAcquired || !acquired.lease.has_value())
            {
                LOG_ERROR("Context cache admission failed for slot %d (status=%d).", i,
                    static_cast<int32_t>(acquired.status));
                return false;
            }

            SequenceCacheState state;
            state.lease = std::move(*acquired.lease);
            state.domain = mContextCacheDomain;
            state.blockExtras = commonExtras;
            state.commitPolicy = context.contextCacheCommitPolicy;
            state.publicationEnabled = context.contextCachePublicationEnabled;
            state.hybridCheckpoint = hybridCheckpoint;
            state.hybridMtp = hybridMtp;
            state.specEagle = specEagle;
            state.mediaSpans = context.mediaSpans[i];
            int32_t const reuseLength = state.lease.reuseTokenLength();
            // Hybrid+MTP recomputes the successor-dependent boundary draft slot (reuseLength - 1) from the checkpoint's
            // saved base hidden state, so the reused draft KV stops one token short of the base reuse. EAGLE and cold
            // sequences keep the draft and base reuse lengths equal.
            int32_t const draftReuseLength = (hybridMtp && reuseLength > 0) ? reuseLength - 1 : reuseLength;
            if (specEagle || hybridMtp)
            {
                state.draftResidentStateLength = draftReuseLength;
            }
            ELLM_CHECK(reuseLength >= 0 && static_cast<size_t>(reuseLength) < inputIds.size(),
                "Context cache plan did not leave a non-empty prefill suffix");
            reuseKVCacheLengthsData[i] = reuseLength;
            draftReuseKVCacheLengthsData[i] = draftReuseLength;
            context.tokenIds[i].assign(inputIds.begin() + reuseLength, inputIds.end());
            context.effectivePrefillLengths[i] = math::cast<int32_t>(context.tokenIds[i].size());
            context.sequenceCacheStates[i].emplace(std::move(state));
            mContextCache->bindBaseRow(basePageTable, i, context.sequenceCacheStates[i]->lease);
            if (specEagle || hybridMtp)
            {
                mContextCache->bindDraftRow(*draftPageTable, i, context.sequenceCacheStates[i]->lease);
            }
            if (hybridCheckpoint)
            {
                CacheRequestLease const& lease = context.sequenceCacheStates[i]->lease;
                if (reuseLength > 0)
                {
                    ELLM_CHECK(lease.recurrentSnapshotSlot().has_value(),
                        "Hybrid context cache hit is missing its recurrent snapshot");
                    mHybridSnapshotStorage->restoreRecurrent(*lease.recurrentSnapshotSlot(), i, context.stream);
                    // Hybrid+MTP always keeps the boundary token (reuseLength - 1) in a private partial page so its
                    // draft KV can be rewritten on restore; reserve one fewer full block so page-aligned checkpoints
                    // still land the boundary in the partial page. Plain hybrid uses the natural page split.
                    size_t const partialPageIndex = hybridMtp
                        ? static_cast<size_t>((reuseLength - 1) / kTOKENS_PER_PAGE)
                        : static_cast<size_t>(reuseLength / kTOKENS_PER_PAGE);
                    int32_t const partialTokenCount
                        = reuseLength - static_cast<int32_t>(partialPageIndex) * kTOKENS_PER_PAGE;
                    if (partialTokenCount > 0)
                    {
                        ELLM_CHECK(
                            lease.partialKvSnapshotSlot().has_value() && partialPageIndex < lease.basePages().size(),
                            "Hybrid context cache hit is missing its private partial page");
                        if (hybridMtp)
                        {
                            ELLM_CHECK(partialPageIndex < lease.draftPages().size(),
                                "Hybrid MTP context cache hit is missing its private draft partial page");
                            mHybridSnapshotStorage->restorePartialKv(*lease.partialKvSnapshotSlot(),
                                lease.basePages()[partialPageIndex], lease.draftPages()[partialPageIndex],
                                partialTokenCount, context.stream);
                        }
                        else
                        {
                            mHybridSnapshotStorage->restorePartialKv(*lease.partialKvSnapshotSlot(),
                                lease.basePages()[partialPageIndex], partialTokenCount, context.stream);
                        }
                    }
                }
                else
                {
                    zeroRecurrentStates(i, context.stream);
                }
            }
        }

        int32_t const maxInputLength
            = *std::max_element(context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.end());
        if (maxInputLength > mDeployment.base.maxSupportedInputLength)
        {
            LOG_ERROR("The context-cache prefill suffix length (%d) exceeds the engine maximum (%d).", maxInputLength,
                mDeployment.base.maxSupportedInputLength);
            return false;
        }

        basePageTable.upload(context.stream);
        if (draftPageTable != nullptr)
        {
            draftPageTable->upload(context.stream);
        }
        cacheMgrBase.resetForNewSequences(mHostReuseKVCacheLengths, context.stream);
        if (needsStrategyKVCache)
        {
            // The draft engine reuses one fewer token than the base at a Hybrid+MTP checkpoint boundary; every other
            // path leaves draft and base reuse equal.
            strategy.resetForNewSequences(mHostDraftReuseKVCacheLengths, context.stream);
        }
        return true;
    }

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

    int32_t const maxInputLength
        = *std::max_element(context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.end());
    if (maxInputLength > mDeployment.base.maxSupportedInputLength)
    {
        LOG_ERROR("The max input length (%d) exceeds the max supported input length (%d) of the LLM Engine.",
            maxInputLength, mDeployment.base.maxSupportedInputLength);
        return false;
    }

    mSharedResources->cacheManagers[0]->resetForNewSequences(mHostReuseKVCacheLengths, context.stream);
    if (needsStrategyKVCache)
    {
        strategy.resetForNewSequences(mHostReuseKVCacheLengths, context.stream);
    }
    return true;
}

bool LLMInferenceRuntime::prepareContextCacheForDecode(DecodingInferenceContext& context)
{
    if (!context.contextReuseEnabled)
    {
        return true;
    }

    ELLM_CHECK(mContextCache != nullptr, "Context-cache-managed request has no runtime adapter");
    KVPageTable& basePageTable = *mSharedResources->kvPageTables[0];
    bool basePageTableDirty = false;
    bool draftPageTableDirty = false;
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        ELLM_CHECK(
            context.sequenceCacheStates[i].has_value(), "Context-cache-managed sequence is missing its active lease");
        SequenceCacheState& state = *context.sequenceCacheStates[i];
        int64_t const residentBaseLength = static_cast<int64_t>(context.rawBatchedInputIds[i].size())
            + static_cast<int64_t>(context.currentGenerateLengths[i]) - 1;
        ELLM_CHECK(residentBaseLength >= 0, "Context cache base resident length became negative");
        if (state.hybridMtp)
        {
            ELLM_CHECK(mDeployment.specConfig.has_value() && mSharedResources->kvPageTables.size() > 1,
                "Hybrid MTP context cache growth has no speculative deployment resources");
            int64_t const baseWorkingLength = residentBaseLength + mDeployment.specConfig->verifySize;
            int64_t const draftWorkingLength = residentBaseLength + mDeployment.specConfig->draftingStep;
            int32_t const requiredBasePages
                = math::cast<int32_t>((baseWorkingLength + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE);
            int32_t const requiredDraftPages
                = math::cast<int32_t>((draftWorkingLength + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE);
            size_t const previousBasePages = state.lease.basePages().size();
            size_t const previousDraftPages = state.lease.draftPages().size();
            KVPageTable& draftPageTable = *mSharedResources->kvPageTables[1];
            if (!mContextCache->growHybridMtpToPageCounts(
                    state.lease, requiredBasePages, requiredDraftPages, basePageTable, draftPageTable, i))
            {
                return false;
            }
            basePageTableDirty |= state.lease.basePages().size() != previousBasePages;
            draftPageTableDirty |= state.lease.draftPages().size() != previousDraftPages;
            continue;
        }
        if (state.specEagle)
        {
            ELLM_CHECK(mDeployment.specConfig.has_value() && mSharedResources->kvPageTables.size() > 1,
                "EAGLE context cache growth has no speculative deployment resources");
            int64_t const baseWorkingLength = residentBaseLength + mDeployment.specConfig->verifySize;
            int64_t const draftProposalSize = static_cast<int64_t>(mDeployment.specConfig->draftingStep)
                * static_cast<int64_t>(mDeployment.specConfig->draftingTopK);
            int64_t const draftWorkingLength = residentBaseLength + draftProposalSize;
            int32_t const requiredBasePages
                = math::cast<int32_t>((baseWorkingLength + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE);
            int32_t const requiredDraftPages
                = math::cast<int32_t>((draftWorkingLength + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE);
            size_t const previousBasePages = state.lease.basePages().size();
            size_t const previousDraftPages = state.lease.draftPages().size();
            KVPageTable& draftPageTable = *mSharedResources->kvPageTables[1];
            if (!mContextCache->growSpecToPageCounts(
                    state.lease, requiredBasePages, requiredDraftPages, basePageTable, draftPageTable, i))
            {
                return false;
            }
            basePageTableDirty |= state.lease.basePages().size() != previousBasePages;
            draftPageTableDirty |= state.lease.draftPages().size() != previousDraftPages;
            continue;
        }
        if (state.hybridCheckpoint && mDeployment.base.kvLayerConfigs.empty())
        {
            continue;
        }
        int32_t const requiredPages
            = math::cast<int32_t>((residentBaseLength + 1 + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE);
        size_t const previousPages = state.lease.basePages().size();
        if (!mContextCache->growBaseToPageCount(state.lease, requiredPages, basePageTable, i))
        {
            return false;
        }
        basePageTableDirty |= state.lease.basePages().size() != previousPages;
    }
    if (basePageTableDirty)
    {
        basePageTable.upload(context.stream);
    }
    if (draftPageTableDirty)
    {
        mSharedResources->kvPageTables[1]->upload(context.stream);
    }
    return true;
}

void LLMInferenceRuntime::publishContextCacheBoundary(
    DecodingInferenceContext& context, int32_t slot, PublicationPoint point, int32_t residentStateLength)
{
    ELLM_CHECK(context.contextReuseEnabled && mContextCache != nullptr,
        "Context cache publication requires an active runtime adapter");
    ELLM_CHECK(slot >= 0 && slot < context.activeBatchSize && context.sequenceCacheStates[slot].has_value(),
        "Context cache publication slot is invalid");

    SequenceCacheState& state = *context.sequenceCacheStates[slot];
    if (!state.publicationEnabled)
    {
        return;
    }
    if (state.hybridMtp)
    {
        ELLM_CHECK(point == PublicationPoint::kDecodeEnd && state.commitPolicy == CommitPolicy::kPrefillStateOnly,
            "Hybrid MTP prefill publication requires an explicit stable successor token");
        return;
    }
    if (state.hybridCheckpoint)
    {
        publishHybridContextCacheBoundary(context, slot, point, residentStateLength);
        return;
    }
    if (residentStateLength < kTOKENS_PER_PAGE)
    {
        return;
    }

    std::vector<int32_t> const logicalTokens = buildResidentTokenHistory(context, slot, residentStateLength);

    size_t const fullBlockCount = logicalTokens.size() / static_cast<size_t>(kTOKENS_PER_PAGE);
    std::vector<BlockKeyExtras> const extrasPerBlock = buildBlockExtras(
        state.blockExtras, state.mediaSpans, fullBlockCount * static_cast<size_t>(kTOKENS_PER_PAGE), kTOKENS_PER_PAGE);
    std::vector<BlockHash> const fullBlockHashes
        = hashFullBlocks(logicalTokens.data(), logicalTokens.size(), kTOKENS_PER_PAGE, extrasPerBlock);
    if (fullBlockHashes.empty())
    {
        return;
    }

    std::optional<int32_t> draftResidentStateLength;
    if (state.specEagle)
    {
        ELLM_CHECK(
            state.draftResidentStateLength.has_value(), "EAGLE context cache state has no draft resident boundary");
        draftResidentStateLength = state.draftPublicationEnabled ? *state.draftResidentStateLength : 0;
        ELLM_CHECK(*draftResidentStateLength <= residentStateLength,
            "EAGLE draft resident boundary exceeds the base resident boundary");
    }
    PublishResult const published = mContextCache->publish(state.lease,
        PublishRequest{fullBlockHashes, residentStateLength, point, state.commitPolicy, draftResidentStateLength});
    if (published.lineageComplete)
    {
        return;
    }

    if (state.specEagle)
    {
        // The draft pages were conditioned before this physical-base canonicalization. Keep serving this request,
        // but never splice those pages into a record on a later boundary.
        state.draftPublicationEnabled = false;
    }

    bool const exactPublishedBoundary = residentStateLength == published.publishedBaseFullBlockCount * kTOKENS_PER_PAGE;
    if (!exactPublishedBoundary)
    {
        state.publicationEnabled = false;
        return;
    }

    mContextCache->rebindBasePrefix(state.lease, published.canonicalBasePages);
    KVPageTable& basePageTable = *mSharedResources->kvPageTables[0];
    mContextCache->bindBaseRow(basePageTable, slot, state.lease);
    basePageTable.upload(context.stream);
}

void LLMInferenceRuntime::publishHybridMtpContextCacheBoundary(
    DecodingInferenceContext& context, int32_t slot, PublicationPoint point, int32_t residentStateLength)
{
    SequenceCacheState& state = *context.sequenceCacheStates[slot];
    if (!state.publicationEnabled || residentStateLength <= 0
        || (point == PublicationPoint::kDecodeEnd && state.commitPolicy == CommitPolicy::kPrefillStateOnly))
    {
        return;
    }
    ELLM_CHECK(point == PublicationPoint::kPrefillEnd && mHybridSnapshotStorage != nullptr
            && mRecurrentStateSchema.has_value() && mDraftEngineSignature.has_value(),
        "Hybrid MTP context cache publication has incomplete endpoint resources");
    ELLM_CHECK(state.draftResidentStateLength.has_value() && *state.draftResidentStateLength == residentStateLength,
        "Hybrid MTP endpoint publication requires equal base and draft resident lengths");

    std::vector<int32_t> const logicalTokens = buildResidentTokenHistory(context, slot, residentStateLength);
    // The successor-dependent boundary token (residentStateLength - 1) is always kept in a private partial page so the
    // consumer can rewrite its draft KV without touching a shared reused page: reserve one fewer full block and let the
    // last 1..kTOKENS_PER_PAGE tokens live in the partial snapshot. For page-aligned lengths this makes the final full
    // page a kTOKENS_PER_PAGE-token partial page.
    size_t const fullBlockCount = static_cast<size_t>((residentStateLength - 1) / kTOKENS_PER_PAGE);
    std::vector<BlockKeyExtras> const fullBlockExtras = buildBlockExtras(
        state.blockExtras, state.mediaSpans, fullBlockCount * static_cast<size_t>(kTOKENS_PER_PAGE), kTOKENS_PER_PAGE);
    std::vector<BlockHash> const fullBlockHashes = hashFullBlocks(logicalTokens.data(),
        fullBlockCount * static_cast<size_t>(kTOKENS_PER_PAGE), kTOKENS_PER_PAGE, fullBlockExtras);
    std::vector<BlockKeyExtras> const exactExtras
        = buildBlockExtras(state.blockExtras, state.mediaSpans, logicalTokens.size(), kTOKENS_PER_PAGE);
    BlockHash const exactDigest
        = hashExactPrefix(logicalTokens.data(), logicalTokens.size(), kTOKENS_PER_PAGE, exactExtras);

    int32_t const partialTokenCount = residentStateLength - static_cast<int32_t>(fullBlockCount) * kTOKENS_PER_PAGE;
    bool const needsPartialSnapshot = true; // partialTokenCount is always in [1, kTOKENS_PER_PAGE]
    mContextCache->releaseRestoredHybridSnapshots(state.lease);
    std::optional<HybridSnapshotReservation> const snapshots
        = mContextCache->reserveHybridSnapshots(state.lease, needsPartialSnapshot);
    if (!snapshots.has_value())
    {
        LOG_DEBUG(
            "Skip Hybrid MTP endpoint publication at length %d: snapshot retention pressure.", residentStateLength);
        return;
    }

    mHybridSnapshotStorage->captureRecurrent(snapshots->recurrentSnapshotSlot, slot, context.stream);
    if (needsPartialSnapshot)
    {
        ELLM_CHECK(fullBlockCount < state.lease.basePages().size() && fullBlockCount < state.lease.draftPages().size(),
            "Hybrid MTP endpoint has no live paired partial pages");
        mHybridSnapshotStorage->capturePartialKv(*snapshots->partialKvSnapshotSlot,
            state.lease.basePages()[fullBlockCount], state.lease.draftPages()[fullBlockCount], partialTokenCount,
            context.stream);
    }
    // Save the base hidden state at the checkpoint boundary (the last prefill row, i.e. the select-token hidden). A
    // future consumer recomputes the successor-dependent boundary draft slot from it instead of matching the successor
    // in the lookup key.
    int32_t const boundaryHiddenRow = context.effectivePrefillLengths[slot] - 1;
    ELLM_CHECK(boundaryHiddenRow >= 0, "Hybrid MTP publication has an empty prefill chunk for its boundary hidden");
    mHybridSnapshotStorage->captureBoundaryHidden(
        snapshots->recurrentSnapshotSlot, mPipelineIO->baseHiddenStates, slot, boundaryHiddenRow, context.stream);
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    HybridMtpPublishRequest const request{fullBlockHashes,
        HybridMtpCheckpointKey{
            state.domain, exactDigest, residentStateLength, *mRecurrentStateSchema, *mDraftEngineSignature},
        point, state.commitPolicy, *snapshots};
    PublishResult const published = mContextCache->publishHybridMtp(state.lease, request);
    if (!published.lineageComplete)
    {
        // Recurrent and draft snapshots were produced from this request's private base path. Rebinding only the base
        // pages would splice independently produced state into one checkpoint, so retain the private request state and
        // skip this and later publication attempts.
        state.publicationEnabled = false;
        mContextCache->retireHybridSnapshotReservation(state.lease, *snapshots);
        return;
    }
    mContextCache->retireHybridSnapshotReservation(state.lease, *snapshots);
}

void LLMInferenceRuntime::publishHybridContextCacheBoundary(
    DecodingInferenceContext& context, int32_t slot, PublicationPoint point, int32_t residentStateLength)
{
    SequenceCacheState& state = *context.sequenceCacheStates[slot];
    if (!state.publicationEnabled || residentStateLength <= 0
        || (point == PublicationPoint::kDecodeEnd && state.commitPolicy == CommitPolicy::kPrefillStateOnly))
    {
        return;
    }
    ELLM_CHECK(mHybridSnapshotStorage != nullptr && mRecurrentStateSchema.has_value(),
        "Hybrid context cache publication has no snapshot storage");

    std::vector<int32_t> const logicalTokens = buildResidentTokenHistory(context, slot, residentStateLength);
    size_t const fullBlockCount = logicalTokens.size() / static_cast<size_t>(kTOKENS_PER_PAGE);
    std::vector<BlockKeyExtras> const fullBlockExtras = buildBlockExtras(
        state.blockExtras, state.mediaSpans, fullBlockCount * static_cast<size_t>(kTOKENS_PER_PAGE), kTOKENS_PER_PAGE);
    std::vector<BlockHash> const fullBlockHashes
        = hashFullBlocks(logicalTokens.data(), logicalTokens.size(), kTOKENS_PER_PAGE, fullBlockExtras);
    std::vector<BlockKeyExtras> const exactExtras
        = buildBlockExtras(state.blockExtras, state.mediaSpans, logicalTokens.size(), kTOKENS_PER_PAGE);
    BlockHash const exactDigest
        = hashExactPrefix(logicalTokens.data(), logicalTokens.size(), kTOKENS_PER_PAGE, exactExtras);

    bool const hasAttention = !mDeployment.base.kvLayerConfigs.empty();
    int32_t const partialTokenCount = residentStateLength % kTOKENS_PER_PAGE;
    bool const needsPartialSnapshot = hasAttention && partialTokenCount > 0;
    mContextCache->releaseRestoredHybridSnapshots(state.lease);
    std::optional<HybridSnapshotReservation> const snapshots
        = mContextCache->reserveHybridSnapshots(state.lease, needsPartialSnapshot);
    if (!snapshots.has_value())
    {
        LOG_DEBUG(
            "Skip hybrid context cache publication at length %d: snapshot retention pressure.", residentStateLength);
        return;
    }

    mHybridSnapshotStorage->captureRecurrent(snapshots->recurrentSnapshotSlot, slot, context.stream);
    if (needsPartialSnapshot)
    {
        ELLM_CHECK(fullBlockCount < state.lease.basePages().size(),
            "Hybrid context cache publication has no live partial page");
        mHybridSnapshotStorage->capturePartialKv(*snapshots->partialKvSnapshotSlot,
            state.lease.basePages()[fullBlockCount], partialTokenCount, context.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    HybridPublishRequest const request{fullBlockHashes,
        HybridCheckpointKey{state.domain, exactDigest, residentStateLength, *mRecurrentStateSchema}, point,
        state.commitPolicy, *snapshots};
    PublishResult const published = mContextCache->publishHybrid(state.lease, request);
    if (!published.lineageComplete)
    {
        // The recurrent snapshot depends on the request's private base path and cannot be attached to a canonical
        // prefix from another producer.
        state.publicationEnabled = false;
        mContextCache->retireHybridSnapshotReservation(state.lease, *snapshots);
        return;
    }
    mContextCache->retireHybridSnapshotReservation(state.lease, *snapshots);
}

bool LLMInferenceRuntime::genAndSaveSystemPromptKVCache(DecodingInferenceContext& context, int32_t genAndSaveBatchIdx)
{
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
    if (mContextCache != nullptr)
    {
        LOG_ERROR("Legacy system prompt KV cache cannot be used while production context cache is enabled.");
        return false;
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

bool LLMInferenceRuntime::performBatchEvict(DecodingInferenceContext& context, DecodingStrategy& strategy)
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

    if (context.contextReuseEnabled)
    {
        // Vanilla prefill/decode currently synchronizes for sampled tokens, but keep the ownership boundary explicit:
        // no lease is published or released until every preceding page write is terminal.
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        for (int32_t i = 0; i < oldActiveBatch; ++i)
        {
            if (!context.finishedStates[i])
            {
                continue;
            }
            int32_t const residentStateLength
                = math::cast<int32_t>(context.rawBatchedInputIds[i].size()) + context.currentGenerateLengths[i] - 1;
            publishContextCacheBoundary(context, i, PublicationPoint::kDecodeEnd, residentStateLength);
            context.sequenceCacheStates[i].reset();
        }
    }

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

    // Upload batch mapping to GPU
    check::check(mDeviceBatchMapping.reshape({oldActiveBatch}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBatchMapping.rawPointer(), batchMapping.data(), oldActiveBatch * sizeof(int32_t),
        cudaMemcpyHostToDevice, context.stream));

    // Production context reuse keeps KV in global physical pages: compact only slot state and move page-table rows.
    // The legacy path still compacts identity-mapped KV rows themselves.
    if (context.contextReuseEnabled)
    {
        mSharedResources->cacheManagers[0]->compactBatchSlotState(
            mDeviceBatchMapping, oldActiveBatch, newActiveBatch, context.stream);
        KVPageTable& basePageTable = *mSharedResources->kvPageTables[0];
        basePageTable.compactRows(batchMapping, newActiveBatch);
        basePageTable.upload(context.stream);
    }
    else
    {
        mSharedResources->cacheManagers[0]->compactBatch(
            mDeviceBatchMapping, oldActiveBatch, newActiveBatch, context.stream);
    }
    mSharedResources->cacheManagers[0]->setActiveBatchSize(newActiveBatch);

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

    // Compact CPU context
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

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
            // Convert flat LogprobsSlot → nested vector for BatchResult (once per completed request).
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
    rt::compactVector(batchMapping, context.currentGenerateLengths);
    rt::compactVector(batchMapping, context.tokenIds);
    rt::compactVector(batchMapping, context.systemPrompts);
    rt::compactVector(batchMapping, context.rawBatchedInputIds);
    rt::compactVector(batchMapping, context.effectivePrefillLengths);
    rt::compactVector(batchMapping, context.sequenceCacheStates);
    rt::compactVector(batchMapping, context.batchIndexMapping);
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
