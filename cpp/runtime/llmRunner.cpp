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

#include "runtime/llmRunner.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/utf8.h"
#include "runtime/audioUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/multiStage/stageRegistry.h"
#include "runtime/streaming.h"
#include "scheduler/batchCompatibility.h"

#include <algorithm>
#include <filesystem>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{

//! A conversation, as opposed to a bare UTF-8 prompt. One input carries both, and PortValue has
//! no room for a flag, so they are told apart by shape: an envelope is a JSON object with a
//! `messages` array.
bool isChatEnvelope(nlohmann::json const& parsed)
{
    return parsed.is_object() && parsed.contains("messages") && parsed.at("messages").is_array();
}

//! One turn. `content` is a bare string or an array of typed items; media items carry only
//! their position, since the bytes travel on their own ports.
Message parseTurn(nlohmann::json const& turn, size_t index)
{
    auto const where = "messages[" + std::to_string(index) + "]";
    ELLM_CHECK(turn.is_object() && turn.contains("role"), where + " needs an object with a 'role'");
    Message message;
    message.role = turn.at("role").get<std::string>();
    if (!turn.contains("content"))
    {
        return message;
    }
    auto const& content = turn.at("content");
    if (content.is_string())
    {
        message.contents.push_back({"text", content.get<std::string>()});
        return message;
    }
    ELLM_CHECK(content.is_array(), where + ": 'content' must be a string or an array of items");
    message.contentIsArray = true;
    for (auto const& item : content)
    {
        ELLM_CHECK(item.is_object() && item.contains("type"), where + ": every content item needs a 'type'");
        auto const type = item.at("type").get<std::string>();
        message.contents.push_back({type, item.value("text", std::string{})});
    }
    return message;
}

//! @p settings with the one per-request field the batching rule reads, the request's own seed.
LLMGenerationRequest batchProbe(LLMGenerationRequest const& settings, std::optional<uint64_t> requestSeed)
{
    auto probe = settings;
    probe.requests.resize(1);
    probe.requests.front().samplingSeed = requestSeed;
    return probe;
}

/*!
 * @brief The first setting @p candidate cannot share a batch with @p founder on, or empty.
 *
 * The scheduler's batching rule, plus what it exempts because it applies them per request while
 * handleRequest applies them to the whole batch: the generate budget and the chat-template flags.
 */
std::string batchDifference(LLMGenerationRequest const& founder, LLMGenerationRequest const& candidate)
{
    auto difference = scheduler::BatchCompatibility::firstDifference(founder, candidate);
    if (difference.empty() && candidate.maxGenerateLength != founder.maxGenerateLength)
    {
        difference = "maxGenerateLength";
    }
    if (difference.empty() && candidate.applyChatTemplate != founder.applyChatTemplate)
    {
        difference = "applyChatTemplate";
    }
    if (difference.empty() && candidate.addGenerationPrompt != founder.addGenerationPrompt)
    {
        difference = "addGenerationPrompt";
    }
    if (difference.empty() && candidate.reasoningEffort != founder.reasoningEffort)
    {
        difference = "reasoningEffort";
    }
    return difference;
}

} // namespace

namespace
{

constexpr char const* kRunnerName = "llm.backbone";
constexpr char const* kTextPort = "text";
constexpr char const* kEmbedsPort = "prefill_embeds";
constexpr char const* kHiddenPort = "accept_hidden";
constexpr char const* kPromptIdsPort = "prompt_ids";
constexpr char const* kPromptInput = "prompt";
constexpr char const* kAudioInput = "audio";
constexpr char const* kImageInput = "image";
constexpr char const* kVisionInput = "vision";
constexpr char const* kSamplingInput = "sampling";
constexpr char const* kRequestInput = "request";
// Keys of the sampling input, spelled as in llm_inference request files.
constexpr char const* kTemperatureKey = "temperature";
constexpr char const* kTopPKey = "top_p";
constexpr char const* kTopKKey = "top_k";
constexpr char const* kMaxGenerateLengthKey = "max_generate_length";
//! What every audio encoder in this repo expects; the encoders reject any other rate, so callers resample first.
constexpr int32_t kAudioSampleRate = 16000;
//! Deliberately not the sequence budget: an unset limit should bound a runaway, not license it.
constexpr int64_t kDefaultMaxGenerateLength = 256;

std::string resolveSubdir(std::string const& root, nlohmann::json const& config, char const* key)
{
    if (!config.contains(key))
    {
        return {};
    }
    auto const value = config.at(key).get<std::string>();
    return value.empty() ? std::string{} : (std::filesystem::path(root) / value).string();
}

nlohmann::json const& objectAt(nlohmann::json const& config, char const* key, std::string const& stageId)
{
    auto const& value = config.at(key);
    ELLM_CHECK(value.is_object(), "stage '" + stageId + "': '" + key + "' must be an object");
    return value;
}

void checkKeys(nlohmann::json const& object, std::vector<char const*> const& allowed, std::string const& where)
{
    for (auto const& item : object.items())
    {
        bool const known
            = std::any_of(allowed.begin(), allowed.end(), [&item](char const* key) { return item.key() == key; });
        ELLM_CHECK(known, where + ": unknown key '" + item.key() + "'");
    }
}

DSparkSchedulerMode parseDSparkScheduler(std::string const& mode)
{
    if (mode == "off")
    {
        return DSparkSchedulerMode::kOff;
    }
    if (mode == "threshold")
    {
        return DSparkSchedulerMode::kThreshold;
    }
    ELLM_CHECK(mode == "sps", "dspark_scheduler '" + mode + "' is not off, threshold or sps");
    return DSparkSchedulerMode::kSPS;
}

//! The runtime options a deployment states in the stage config, as llm_inference takes them on its
//! command line.
DeploymentOptions deploymentFrom(
    nlohmann::json const& config, std::string const& artifactDir, std::string const& stageId)
{
    std::string const where = "stage '" + stageId + "'";
    checkKeys(config,
        {"multimodal_artifact", "checkpoint", "draft_checkpoint", "accept_hidden_layer", "max_generate_length",
            "temperature", "top_p", "top_k", "spec_decode", "lora_adapters", "context_cache", "visual_pruning", "ple",
            "runtime_cache", "offload_embedding"},
        where);

    DeploymentOptions options;
    options.engineDir = artifactDir;
    options.multimodalEngineDir = resolveSubdir(artifactDir, config, "multimodal_artifact");
    options.checkpointDir = resolveSubdir(artifactDir, config, "checkpoint");
    options.draftCheckpointDir = resolveSubdir(artifactDir, config, "draft_checkpoint");
    options.offloadEmbedding = config.value("offload_embedding", false);
    if (config.contains("runtime_cache"))
    {
        options.runtimeCachePath = resolveSubdir(artifactDir, config, "runtime_cache");
    }
    if (config.contains("lora_adapters"))
    {
        for (auto const& [name, path] : objectAt(config, "lora_adapters", stageId).items())
        {
            options.loraWeightsMap[name] = (std::filesystem::path(artifactDir) / path.get<std::string>()).string();
        }
    }
    if (config.contains("spec_decode"))
    {
        auto const& spec = objectAt(config, "spec_decode", stageId);
        checkKeys(spec,
            {"draft_top_k", "draft_step", "proposal_tokens", "verify_tokens", "dspark_w2_backend", "dspark_scheduler",
                "dspark_confidence_threshold", "dspark_min_proposal_len", "dspark_max_proposal_len"},
            where + " spec_decode");
        // Absent keys stay unset, and the runtime resolves them from the engines as it does for llm_inference.
        SpecDecodeDraftingConfig drafting;
        auto const optional = [&spec](char const* key, std::optional<int32_t>& field) {
            if (spec.contains(key))
            {
                field = spec.at(key).get<int32_t>();
            }
        };
        optional("draft_top_k", drafting.draftTopK);
        optional("draft_step", drafting.draftingStep);
        optional("proposal_tokens", drafting.proposalTokens);
        optional("verify_tokens", drafting.verifyTokens);
        drafting.dsparkW2Backend = parseDSparkW2Backend(spec.value("dspark_w2_backend", std::string{"auto"}));
        drafting.dsparkSchedulerMode = parseDSparkScheduler(spec.value("dspark_scheduler", std::string{"off"}));
        drafting.dsparkConfidenceThreshold = spec.value("dspark_confidence_threshold", 0.0F);
        drafting.dsparkMinProposalLen = spec.value("dspark_min_proposal_len", drafting.dsparkMinProposalLen);
        drafting.dsparkMaxProposalLen = spec.value("dspark_max_proposal_len", drafting.dsparkMaxProposalLen);
        options.draftingConfig = drafting;
    }
    if (config.contains("context_cache"))
    {
        auto const& cache = objectAt(config, "context_cache", stageId);
        checkKeys(cache, {"enable_reuse", "max_records", "max_snapshots", "encoder_cache_budget_bytes"},
            where + " context_cache");
        auto& target = options.contextCacheConfig;
        target.enabled = cache.value("enable_reuse", target.enabled);
        target.maxRecords = cache.value("max_records", target.maxRecords);
        target.maxSnapshots = cache.value("max_snapshots", target.maxSnapshots);
        target.encoderEmbeddingCacheBudgetBytes
            = cache.value("encoder_cache_budget_bytes", target.encoderEmbeddingCacheBudgetBytes);
    }
    if (config.contains("ple"))
    {
        auto const& ple = objectAt(config, "ple", stageId);
        checkKeys(ple, {"file", "residency", "cache_slots"}, where + " ple");
        if (ple.contains("file"))
        {
            options.pleConfig.filePath = resolveSubdir(artifactDir, ple, "file");
        }
        auto const residency = ple.value("residency", std::string{"resident"});
        ELLM_CHECK(residency == "resident" || residency == "cached",
            where + ": ple residency '" + residency + "' is not resident or cached");
        options.pleConfig.residency = residency == "resident" ? PleResidency::kRESIDENT : PleResidency::kCACHED;
        options.pleConfig.cacheSlots = ple.value("cache_slots", options.pleConfig.cacheSlots);
    }
    return options;
}

VisualPrunerConfig visualPrunerFrom(nlohmann::json const& config, std::string const& stageId)
{
    VisualPrunerConfig pruner;
    if (!config.contains("visual_pruning"))
    {
        return pruner;
    }
    auto const& source = objectAt(config, "visual_pruning", stageId);
    checkKeys(source, {"algorithm", "reduction_ratio", "min_visual_tokens", "pivot_image_tokens", "pivot_text_tokens"},
        "stage '" + stageId + "' visual_pruning");
    pruner.enabled = true;
    pruner.algorithm = source.value("algorithm", pruner.algorithm);
    pruner.reductionRatio = source.value("reduction_ratio", pruner.reductionRatio);
    pruner.minVisualTokens = source.value("min_visual_tokens", pruner.minVisualTokens);
    pruner.pivotImageTokens = source.value("pivot_image_tokens", pruner.pivotImageTokens);
    pruner.pivotTextTokens = source.value("pivot_text_tokens", pruner.pivotTextTokens);
    return pruner;
}

} // namespace

LlmRunner::LlmRunner(multiStage::StageInitContext const& init)
    : mArtifactDir(init.artifactDir)
    , mStream(init.stream)
    , mMaxBatch(init.maxBatch)
    , mMaxSeq(init.maxSeq)
{
    ELLM_CHECK(init.config != nullptr, "llm.backbone stage built without a StageConfig");
    mStageId = init.config->id;
    auto const& config = init.config->config;
    mDeployment = deploymentFrom(config, mArtifactDir, mStageId);
    mVisualPruner = visualPrunerFrom(config, mStageId);
    mAcceptHiddenLayer = config.value("accept_hidden_layer", -1);
    mMaxGenerateLength = config.value("max_generate_length", kDefaultMaxGenerateLength);
    // Unset keys take llm_inference's request-file defaults. Greedy here would be a quieter but
    // different model.
    mTemperature = config.value("temperature", 1.0F);
    mTopP = config.value("top_p", 0.8F);
    mTopK = config.value("top_k", static_cast<int64_t>(50));

    // Emitting a hidden port is the deployment's statement that this engine has that
    // output. A tap on anything else is rejected before a byte of memory is claimed.
    for (auto const& port : init.config->emits)
    {
        if (port.kind == multiStage::PortKind::kHidden)
        {
            mHiddenPorts.push_back(port.id);
        }
    }
    // A speculative runtime allocates no hidden-state output, so llm_inference cannot pair
    // --specDecode with audio output either.
    ELLM_CHECK(!mDeployment.draftingConfig.has_value() || mHiddenPorts.empty(),
        "stage '" + mStageId + "': a speculative-decoding build cannot emit hidden states; speech needs a "
                               "non-speculative thinker build");
}

LlmRunner::LlmRunner(multiStage::StageInitContext const& init, RuntimeFactory makeRuntime)
    : LlmRunner(init)
{
    mMakeRuntime = std::move(makeRuntime);
}

LlmRunner::~LlmRunner() = default;

LlmRunner::Row& LlmRunner::rowFor(multiStage::StageExecContext const& ctx)
{
    ELLM_CHECK(ctx.slot != nullptr, "llm.backbone stage reached without a request slot");
    auto const slot = static_cast<size_t>(ctx.slot->batchSlot);
    ELLM_CHECK(ctx.slot->batchSlot >= 0 && slot < mRows.size(),
        "stage '" + mStageId + "': batch slot " + std::to_string(ctx.slot->batchSlot) + " is outside max_batch "
            + std::to_string(mMaxBatch));
    return mRows[slot];
}

LlmRunner::Row const& LlmRunner::rowFor(multiStage::RequestSlot const& slot) const
{
    auto const index = static_cast<size_t>(slot.batchSlot);
    ELLM_CHECK(slot.batchSlot >= 0 && index < mRows.size(),
        "stage '" + mStageId + "': batch slot " + std::to_string(slot.batchSlot) + " is outside max_batch "
            + std::to_string(mMaxBatch));
    return mRows[index];
}

void LlmRunner::preallocate()
{
    ELLM_CHECK(mMaxBatch >= 1, "llm.backbone stage needs a positive max_batch");
    ELLM_CHECK(mMaxSeq > 0, "llm.backbone stage needs a positive max_seq");
    ELLM_CHECK(mMaxGenerateLength > 0, "llm.backbone stage 'max_generate_length' must be positive");

    mRuntime = mMakeRuntime != nullptr ? mMakeRuntime(mDeployment, mStream)
                                       : std::make_unique<LLMInferenceRuntime>(mDeployment, mStream);
    ELLM_CHECK(mRuntime != nullptr, "stage '" + mStageId + "': no runtime was built");
    if (mVisualPruner.enabled)
    {
        mRuntime->setVisualPrunerConfig(mVisualPruner);
    }

    // Sized from the embedding table: that is the only public statement this engine
    // makes about its hidden size.
    auto const& embeddingTable = mRuntime->getEmbeddingTable();
    auto const tableShape = embeddingTable.getShape();
    ELLM_CHECK(tableShape.getNumDims() == 2, "llm.backbone embedding table must be [vocab, hidden]");

    mHiddenSize = static_cast<int32_t>(tableShape[1]);

    mRows.resize(static_cast<size_t>(mMaxBatch));
    for (auto& row : mRows)
    {
        row.outputIds.reserve(static_cast<size_t>(mMaxGenerateLength));
        row.inputIds.reserve(static_cast<size_t>(mMaxSeq));
    }
    mRequest.requests.reserve(static_cast<size_t>(mMaxBatch));
    mRequest.preTokenizedInputIds.reserve(static_cast<size_t>(mMaxBatch));
    mRequest.streamChannels.reserve(static_cast<size_t>(mMaxBatch));
    mPacked.reserve(static_cast<size_t>(mMaxBatch));

    // Decode is launch-bound at edge batch sizes, so without a graph every step pays the
    // per-layer launch overhead. Failure is not fatal -- the engine runs without one.
    if (!mRuntime->captureDecodingCUDAGraph(mStream))
    {
        LOG_WARNING("Stage '%s' could not capture a decoding CUDA graph; falling back to plain engine execution.",
            mStageId.c_str());
    }
}

bool LlmRunner::admit(multiStage::StageExecContext const& ctx, std::string& reason)
{
    auto& row = rowFor(ctx);
    row.ready = false;

    // Publication follows from an edge existing, not from a request flag. Layer 0 is the
    // input-embedding slot, so a hidden tap on it is refused here rather than mid-decode.
    row.hiddenConsumed = false;
    for (auto const& edge : ctx.slot->route.edges())
    {
        if (edge.producer < 0 || std::find(mHiddenPorts.begin(), mHiddenPorts.end(), edge.port) == mHiddenPorts.end())
        {
            continue;
        }
        if (mAcceptHiddenLayer == 0)
        {
            reason = "stage '" + mStageId + "': input '" + edge.inputName + "' consumes port '" + edge.port
                + "' but 'accept_hidden_layer' is 0, the input-embedding slot";
            return false;
        }
        row.hiddenConsumed = true;
    }

    // The settings a cohort shares are fixed by its first member; one asking for others waits for
    // a cohort of its own rather than failing this one mid-step.
    row.settings = batchSettings(ctx, row.requestSeed);
    if (row.settings.maxGenerateLength <= 0)
    {
        reason = "stage '" + mStageId + "': 'max_generate_length' must be positive";
        return false;
    }
    if (ctx.slot->batchSlot > 0)
    {
        auto const& founder = mRows.front();
        auto const difference = batchDifference(
            batchProbe(founder.settings, founder.requestSeed), batchProbe(row.settings, row.requestSeed));
        if (!difference.empty())
        {
            reason = "stage '" + mStageId + "': " + difference + " differs from the cohort's";
            return false;
        }
    }

    row.ready = true;
    return true;
}

LLMGenerationRequest LlmRunner::batchSettings(
    multiStage::StageExecContext const& ctx, std::optional<uint64_t>& requestSeed) const
{
    LLMGenerationRequest settings;
    requestSeed.reset();
    auto const* parsed = ctx.in != nullptr ? ctx.in->find(kRequestInput) : nullptr;
    if (parsed != nullptr && parsed->data != nullptr)
    {
        ELLM_CHECK(parsed->kind == multiStage::PortKind::kRequest
                && parsed->elementBytes == static_cast<int32_t>(sizeof(LLMGenerationRequest)),
            "stage '" + mStageId + "': input '" + kRequestInput + "' must carry one LLMGenerationRequest");
        auto const& source = *static_cast<LLMGenerationRequest const*>(parsed->data);
        ELLM_CHECK(source.requests.size() == 1,
            "stage '" + mStageId + "': a parsed request must hold exactly one request, not "
                + std::to_string(source.requests.size()));
        settings = source;
        settings.requests.clear();
        settings.formattedRequests.clear();
        settings.preTokenizedInputIds.clear();
        settings.streamChannels.clear();
        requestSeed = source.requests.front().samplingSeed;
    }
    else
    {
        auto const* prompt = ctx.in != nullptr ? ctx.in->find(kPromptInput) : nullptr;
        if (prompt != nullptr && prompt->data != nullptr && prompt->elementBytes == 1 && !prompt->shape.empty())
        {
            auto const envelope = nlohmann::json::parse(
                std::string(static_cast<char const*>(prompt->data), static_cast<size_t>(prompt->shape.back())), nullptr,
                /*allow_exceptions=*/false);
            if (isChatEnvelope(envelope))
            {
                settings.applyChatTemplate = envelope.value("apply_chat_template", true);
                settings.addGenerationPrompt = envelope.value("add_generation_prompt", true);
                settings.enableThinking = envelope.value("enable_thinking", false);
            }
        }
        auto const sampling = multiStage::jsonInput(ctx.in, kSamplingInput);
        settings.temperature = sampling.value(kTemperatureKey, mTemperature);
        settings.topP = sampling.value(kTopPKey, mTopP);
        settings.topK = sampling.value(kTopKKey, mTopK);
        settings.maxGenerateLength = sampling.value(kMaxGenerateLengthKey, mMaxGenerateLength);
    }
    // The layer belongs to this deployment's engine; the runtime keys its hidden-state
    // registry on it, which is where publishPrefill() reads the window back.
    settings.acceptHiddenLayer = mAcceptHiddenLayer;
    return settings;
}

void LlmRunner::prepare(multiStage::StageExecContext& ctx)
{
    auto& row = rowFor(ctx);
    ELLM_CHECK(row.ready, "llm.backbone prepare() ran without a successful admit()");
    row.generated = false;
    row.outputIds.clear();
    row.outputText.clear();
    row.finishReason = "not-finished";
    row.promptIds.clear();
    row.promptTokens = 0;
    row.specVerifyCount = 0;
    row.specAcceptanceLength = 0.0F;
    row.logprobs.clear();
    row.formatted = {};

    // admit() already took the settings; this takes the conversation and its media.
    auto const* prepared = ctx.in != nullptr ? ctx.in->find(kRequestInput) : nullptr;
    if (prepared != nullptr && prepared->data != nullptr)
    {
        adoptRequest(ctx, row, *prepared);
    }
    else
    {
        assembleRequest(ctx, row);
    }

    // The caller's channel when it attached one, so streamed text reaches it while the
    // decode loop runs. Otherwise a private channel: it is still the only public entry
    // into a loop already running, which is how cancellation lands.
    row.cancelChannel = ctx.textSink != nullptr ? ctx.textSink : StreamChannel::create();

    // The decode loop polls every stream channel's cancel flag once per iteration, so
    // cancelling the channel is the whole mechanism.
    ctx.slot->abandon.onRequest([channel = row.cancelChannel]() { channel->cancel(); });
}

void LlmRunner::adoptRequest(multiStage::StageExecContext const& ctx, Row& row, multiStage::PortValue const& value)
{
    // The parsed request carries its own media; a second copy on the media inputs would leave
    // two clips for one placeholder.
    for (char const* name : {kAudioInput, kImageInput})
    {
        ELLM_CHECK(ctx.in->find(name) == nullptr,
            "stage '" + mStageId + "': input '" + name + "' cannot be bound beside a parsed request");
    }
    auto const& source = *static_cast<LLMGenerationRequest const*>(value.data);
    row.request = source.requests.front();
    row.inputIds.clear();
    if (!source.preTokenizedInputIds.empty())
    {
        row.inputIds = source.preTokenizedInputIds.front();
    }
}

void LlmRunner::assembleRequest(multiStage::StageExecContext& ctx, Row& row)
{
    auto const* prompt = ctx.in != nullptr ? ctx.in->find(kPromptInput) : nullptr;
    ELLM_CHECK(prompt != nullptr && prompt->data != nullptr,
        "stage '" + mStageId + "': input '" + kPromptInput + "' is not bound");

    auto& request = row.request;
    request = LLMGenerationRequest::Request{};

    // Media and text share one turn: the template renders a clip the user attached to a
    // question differently from a clip they sent on its own.
    Message message;
    message.role = "user";
    bindMedia(ctx, request, message);
    bindPixels(ctx, request, message);

    if (prompt->elementBytes == static_cast<int32_t>(sizeof(int32_t)))
    {
        // Already tokenized: the caller owns the template and the vocabulary.
        ELLM_CHECK(message.contents.empty(),
            "stage '" + mStageId + "': media cannot be attached to a pre-tokenized prompt; the template that would "
                                   "place it has already been applied");
        auto const* tokens = static_cast<int32_t const*>(prompt->data);
        auto const count = prompt->shape.empty() ? 0 : prompt->shape.back();
        ELLM_CHECK(count > 0 && count <= mMaxSeq,
            "llm.backbone prompt of " + std::to_string(count) + " tokens does not fit max_seq "
                + std::to_string(mMaxSeq));
        row.inputIds.assign(tokens, tokens + count);
    }
    else
    {
        // UTF-8. The chat template and the tokenizer belong to this engine, and the talker
        // downstream scans the templated stream for its assistant segment, so applying the
        // template here rather than in the caller is what keeps that segment present.
        ELLM_CHECK(prompt->elementBytes == 1 && !prompt->shape.empty(),
            "stage '" + mStageId + "': input '" + kPromptInput + "' is neither int32 token ids nor UTF-8 bytes");
        row.inputIds.clear();
        std::string const text(static_cast<char const*>(prompt->data), static_cast<size_t>(prompt->shape.back()));
        auto const parsed = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
        if (isChatEnvelope(parsed))
        {
            auto const& turns = parsed.at("messages");
            ELLM_CHECK(!turns.empty(), "stage '" + mStageId + "': 'messages' is empty");
            for (size_t turn = 0; turn < turns.size(); ++turn)
            {
                request.messages.push_back(parseTurn(turns.at(turn), turn));
            }
            // A conversation that names its media owns the placeholders, which the encoders fill
            // in order: the count must match what arrived, and no markers are synthesised on top.
            int64_t declared = 0;
            for (auto const& turn : request.messages)
            {
                declared += std::count_if(turn.contents.begin(), turn.contents.end(),
                    [](Message::MessageContent const& item) { return item.type != "text"; });
            }
            auto const arrived = static_cast<int64_t>(request.audioBuffers.size() + request.imageBuffers.size());
            ELLM_CHECK(declared == 0 || declared == arrived,
                "stage '" + mStageId + "': the conversation declares " + std::to_string(declared)
                    + " media item(s) but " + std::to_string(arrived) + " arrived on the media ports");
            if (declared == 0 && !message.contents.empty())
            {
                auto& target = request.messages.back();
                target.contents.insert(target.contents.begin(), message.contents.begin(), message.contents.end());
            }
        }
        else
        {
            message.contents.push_back({"text", text});
            request.messages.push_back(std::move(message));
        }
    }
}

multiStage::StageStatus LlmRunner::step(multiStage::StageExecContext& ctx)
{
    std::vector<multiStage::StageExecContext*> members{&ctx};
    std::vector<multiStage::StageStatus> statuses(1, multiStage::StageStatus::kFailed);
    stepCohort(members, statuses);
    return statuses.front();
}

void LlmRunner::stepCohort(
    std::vector<multiStage::StageExecContext*> const& members, std::vector<multiStage::StageStatus>& statuses)
{
    mPacked.clear();
    mRequest.requests.clear();
    mRequest.preTokenizedInputIds.clear();
    mRequest.streamChannels.clear();

    for (size_t index = 0; index < members.size(); ++index)
    {
        auto& member = *members[index];
        auto const& row = rowFor(member);
        if (member.slot->abandon.requested() || row.generated)
        {
            statuses[index] = multiStage::StageStatus::kDone;
            continue;
        }
        mPacked.push_back(index);
    }
    if (mPacked.empty())
    {
        return;
    }

    // A cohort steps on one lane, so one stream covers it; the alternative would be a
    // batched engine call recorded on a stream only some of its members waited on.
    auto const stream = members[mPacked.front()]->stream;
    for (auto const index : mPacked)
    {
        ELLM_CHECK(members[index]->stream == stream, "llm.backbone cohort spans more than one stream");
    }

    // admit() refused a member whose settings differ from the cohort's; this guards the step.
    auto const& founder = rowFor(*members[mPacked.front()]);
    auto const founderProbe = batchProbe(founder.settings, founder.requestSeed);
    for (auto const index : mPacked)
    {
        auto const& row = rowFor(*members[index]);
        auto const difference = batchDifference(founderProbe, batchProbe(row.settings, row.requestSeed));
        ELLM_CHECK(difference.empty(),
            "stage '" + mStageId + "': requests in one cohort must share their batch-level settings; request "
                + std::to_string(members[index]->slot->requestId) + " differs from "
                + std::to_string(members[mPacked.front()]->slot->requestId) + " in " + difference);
    }
    // Assigning keeps the per-call vectors' capacity, so the step stays free of allocation there.
    mRequest = founder.settings;

    for (auto const index : mPacked)
    {
        auto& row = rowFor(*members[index]);
        mRequest.requests.push_back(std::move(row.request));
        mRequest.preTokenizedInputIds.push_back(std::move(row.inputIds));
        mRequest.streamChannels.push_back(row.cancelChannel);
    }

    // One call covers the whole decode loop; the runtime exposes no per-token entry point.
    // Capturing the prefill window also keeps context-cache reuse and visual pruning off,
    // both of which would shift the rows a consumer reads, so it is on only when this
    // cohort's route has a consumer. The members share a route, so they agree.
    bool const captureHidden = std::any_of(
        mPacked.begin(), mPacked.end(), [&](size_t index) { return rowFor(*members[index]).hiddenConsumed; });
    bool const ok = mRuntime->handleRequest(mRequest, mResponse, stream, /*outputThinkerEmbeddings=*/captureHidden);

    // Hand the moved-out storage back before the early return, so each row keeps its capacity.
    for (size_t row = 0; row < mPacked.size(); ++row)
    {
        auto& target = rowFor(*members[mPacked[row]]);
        target.request = std::move(mRequest.requests[row]);
        target.inputIds = std::move(mRequest.preTokenizedInputIds[row]);
        target.generated = true;
    }

    if (!ok)
    {
        for (auto const index : mPacked)
        {
            LOG_ERROR("Backbone request %llu failed in the decode loop",
                static_cast<unsigned long long>(members[index]->slot->requestId));
            statuses[index] = multiStage::StageStatus::kFailed;
        }
        return;
    }

    // Templated prompt ids, media placeholders included. The runtime fills these only when it
    // captures the prefill window.
    auto const& promptIds = mRuntime->getBaseModelInputTokenIds();
    for (size_t row = 0; row < mPacked.size(); ++row)
    {
        auto& member = *members[mPacked[row]];
        auto& target = rowFor(member);
        if (row < mResponse.outputIds.size())
        {
            auto const& ids = mResponse.outputIds[row];
            target.outputIds.assign(ids.begin(), ids.end());
        }
        if (row < mResponse.outputTexts.size())
        {
            target.outputText.assign(mResponse.outputTexts[row]);
        }
        if (row < promptIds.size())
        {
            target.promptIds = promptIds[row];
        }
        if (row < mResponse.finishReasons.size())
        {
            // finishReasonName() spellings, which llm_inference reports too, so the field diffs directly.
            target.finishReason = finishReasonName(mResponse.finishReasons[row]);
        }
        if (row < mResponse.inputTokenCounts.size())
        {
            target.promptTokens = mResponse.inputTokenCounts[row];
        }
        target.specVerifyCount = row < mResponse.specVerifyCounts.size() ? mResponse.specVerifyCounts[row] : 0;
        target.specAcceptanceLength
            = row < mResponse.specAcceptanceLengths.size() ? mResponse.specAcceptanceLengths[row] : 0.0F;
        target.logprobs.clear();
        if (row < mResponse.logprobs.size())
        {
            target.logprobs.swap(mResponse.logprobs[row]);
        }
        target.formatted = row < mRequest.formattedRequests.size() ? mRequest.formattedRequests[row]
                                                                   : LLMGenerationRequest::FormattedRequest{};
        bindText(member);
        bindPromptIds(member);
        publishPrefill(member, static_cast<int32_t>(row));
        statuses[mPacked[row]] = multiStage::StageStatus::kDone;
    }
}

void LlmRunner::bindMedia(multiStage::StageExecContext& ctx, LLMGenerationRequest::Request& request, Message& turn)
{
    auto const* audio = ctx.in != nullptr ? ctx.in->find(kAudioInput) : nullptr;
    if (audio == nullptr || audio->data == nullptr || audio->shape.empty())
    {
        return;
    }
    ELLM_CHECK(
        audio->kind == multiStage::PortKind::kAudio && audio->elementBytes == static_cast<int32_t>(sizeof(float)),
        "stage '" + mStageId + "': input '" + kAudioInput + "' must be mono FP32 PCM");

    auto const numSamples = audio->shape.back();
    ELLM_CHECK(numSamples > 0, "stage '" + mStageId + "': input '" + kAudioInput + "' carries no samples");

    // Borrowed: the payload outlives the request. Page-locked samples let the encoder upload
    // them asynchronously.
    auto pcm = std::make_shared<Tensor>(
        audio->data, Coords{numSamples}, DeviceType::kCPU, trt_edgellm::DataType::kFLOAT, mStageId + ".audio_pcm");

    request.audioBuffers.push_back(audioUtils::wrapPcm(std::move(pcm), kAudioSampleRate));

    // The placeholder the template scatters the encoder output into. It leads the turn, which
    // is the order the omni templates expect a clip and the question about it in.
    turn.contents.push_back({"audio", ""});
}

void LlmRunner::bindPixels(multiStage::StageExecContext& ctx, LLMGenerationRequest::Request& request, Message& turn)
{
    auto const* pixels = ctx.in != nullptr ? ctx.in->find(kImageInput) : nullptr;
    if (pixels == nullptr || pixels->data == nullptr)
    {
        return;
    }
    ELLM_CHECK(pixels->kind == multiStage::PortKind::kImage && pixels->elementBytes == 1 && pixels->shape.size() == 4
            && pixels->shape[3] == 3,
        "stage '" + mStageId + "': input '" + kImageInput + "' must be packed RGB8 {frames, height, width, 3}");

    auto const frames = pixels->shape[0];
    auto const height = pixels->shape[1];
    auto const width = pixels->shape[2];
    ELLM_CHECK(frames > 0 && height > 0 && width > 0,
        "stage '" + mStageId + "': input '" + kImageInput + "' has an empty extent");

    // Borrowed: preprocessing reads the pixels in place, and wrapImageBuffer() refuses a host
    // buffer that is not page-locked.
    auto buffer = std::make_shared<rt::Tensor>(pixels->data, Coords{frames, height, width, 3}, rt::DeviceType::kCPU,
        trt_edgellm::DataType::kUINT8, mStageId + ".pixels");
    imageUtils::ImageLayout layout;
    layout.pitchBytes[0] = width * 3;
    layout.frameStrideBytes[0] = height * width * 3;

    auto const vision = multiStage::jsonInput(ctx.in, kVisionInput);
    auto image = imageUtils::wrapImageBuffer(std::move(buffer), layout, width, height, frames);
    image.doResize = vision.value("do_resize", true);
    image.isVideo = vision.value("is_video", frames > 1);
    image.fps = vision.value("fps", 1.0);
    if (auto const stamps = vision.find("timestamps"); stamps != vision.end() && stamps->is_array())
    {
        image.timestamps = stamps->get<std::vector<double>>();
    }
    bool const isVideo = image.isVideo;
    request.imageBuffers.push_back(std::move(image));
    turn.contents.push_back({isVideo ? "video" : "image", ""});
}

void LlmRunner::bindText(multiStage::StageExecContext& ctx)
{
    auto& row = rowFor(ctx);
    // handleRequest() runs the whole decode loop, so the tail is never partial: by the
    // time anything is bound here the stream has ended.
    multiStage::PortValue text{row.outputIds.data(), {static_cast<int64_t>(row.outputIds.size())},
        multiStage::PortKind::kText, static_cast<int32_t>(sizeof(int32_t))};
    text.complete = true;
    ctx.out->bind(kTextPort, text);
}

void LlmRunner::bindPromptIds(multiStage::StageExecContext& ctx)
{
    auto& row = rowFor(ctx);
    if (row.promptIds.empty())
    {
        return;
    }
    multiStage::PortValue prompt{row.promptIds.data(), {static_cast<int64_t>(row.promptIds.size())},
        multiStage::PortKind::kText, static_cast<int32_t>(sizeof(int32_t))};
    prompt.complete = true;
    ctx.out->bind(kPromptIdsPort, prompt);
}

void LlmRunner::publishPrefill(multiStage::StageExecContext& ctx, int32_t decodeRow)
{
    // Nothing on this route reads the window, so there is nothing to publish -- and the
    // engine is not obliged to have emitted it.
    auto const& row = rowFor(ctx);
    if (!row.hiddenConsumed)
    {
        return;
    }

    auto const* embeds = mRuntime->getBaseModelHiddenStates(0);
    auto const* hidden = mRuntime->getBaseModelHiddenStates(mAcceptHiddenLayer);
    int32_t const paddedLength = mRuntime->getBaseModelPrefillLength();
    ELLM_CHECK(embeds != nullptr && hidden != nullptr && paddedLength > 0,
        "stage '" + mStageId
            + "' was asked for hidden states but published none; the engine must emit the accept-hidden output");

    auto const hiddenShape = hidden->getShape();
    auto const hiddenSize = hiddenShape[hiddenShape.getNumDims() - 1];

    ELLM_CHECK(hiddenSize == mHiddenSize,
        "stage '" + mStageId + "': engine hidden width " + std::to_string(hiddenSize) + " does not match the "
            + std::to_string(mHiddenSize) + " the embedding table declares");

    // The retained window is {batch, maxPrefillLen, hidden}, padded to the longest row: this
    // member's rows start at its decode row, and only the first promptIds.size() are real.
    auto const rowLength = static_cast<int64_t>(row.promptIds.size());
    ELLM_CHECK(rowLength > 0 && rowLength <= paddedLength,
        "stage '" + mStageId + "': prompt of " + std::to_string(rowLength)
            + " tokens does not fit the prefill window of " + std::to_string(paddedLength));
    auto const rowOffset
        = static_cast<size_t>(decodeRow) * static_cast<size_t>(paddedLength) * static_cast<size_t>(hiddenSize);

    // LLMRankRuntime already retains this view as it stood at the end of prefill
    // (StreamingPrefillBuffers), so decode does not race these pointers. They stay valid
    // until the next handleRequest() on that runtime.
    auto const hiddenBytes = rt::utils::getTypeSize(hidden->getDataType());
    ctx.out->bind(kHiddenPort,
        multiStage::PortValue{static_cast<char*>(const_cast<void*>(hidden->rawPointer())) + rowOffset * hiddenBytes,
            {rowLength, hiddenSize}, multiStage::PortKind::kHidden, static_cast<int32_t>(hiddenBytes)});

    auto const embedBytes = rt::utils::getTypeSize(embeds->getDataType());
    ctx.out->bind(kEmbedsPort,
        multiStage::PortValue{static_cast<char*>(const_cast<void*>(embeds->rawPointer())) + rowOffset * embedBytes,
            {rowLength, hiddenSize}, multiStage::PortKind::kEmbeds, static_cast<int32_t>(embedBytes)});
}

void LlmRunner::finish(multiStage::StageExecContext& ctx, multiStage::FinishReason reason)
{
    // handleRequest() terminates the channel it attached. A request that never got there, or
    // left before attaching it, has nobody else to wake the caller, and cancel() is the only
    // public way to terminate a StreamChannel.
    if (ctx.textSink != nullptr && !ctx.textSink->isFinished())
    {
        ctx.textSink->cancel();
    }
    // The text port is bound complete on every path, failure included, so a downstream
    // consumer sees an end rather than waiting on silence.
    if (ctx.out != nullptr)
    {
        bindText(ctx);
        bindPromptIds(ctx);
    }
    auto& row = rowFor(ctx);
    if (reason != multiStage::FinishReason::kCompleted)
    {
        LOG_INFO("Backbone request %llu ended: %s after %zu tokens",
            static_cast<unsigned long long>(ctx.slot->requestId), multiStage::toString(reason), row.outputIds.size());
    }
    row.cancelChannel.reset();
}

void LlmRunner::reset()
{
    // Cleared in place: the vectors keep the capacity preallocate() reserved.
    for (auto& row : mRows)
    {
        row.request.messages.clear();
        row.request.audioBuffers.clear();
        row.request.imageBuffers.clear();
        row.inputIds.clear();
        row.cancelChannel.reset();
        row.outputIds.clear();
        row.outputText.clear();
        row.promptIds.clear();
        row.promptTokens = 0;
        row.specVerifyCount = 0;
        row.specAcceptanceLength = 0.0F;
        row.logprobs.clear();
        row.formatted = {};
        row.finishReason = "not-finished";
        row.ready = false;
        row.hiddenConsumed = false;
        row.generated = false;
    }
    mPacked.clear();
    mRequest.requests.clear();
    mRequest.preTokenizedInputIds.clear();
    mRequest.streamChannels.clear();
    mResponse.outputIds.clear();
    mResponse.outputTexts.clear();
    mResponse.finishReasons.clear();
    mResponse.inputTokenCounts.clear();
}

nlohmann::json LlmRunner::diagnostics(multiStage::RequestSlot const& slot) const
{
    auto const& row = rowFor(slot);
    std::string pending;
    std::string outputText = utf8::sanitizeUtf8Streaming(row.outputText, pending);
    outputText += utf8::sanitizeUtf8Flush(pending);
    nlohmann::json report{{"finish_reason", row.finishReason}, {"output_text", std::move(outputText)},
        {"generated_tokens", static_cast<int64_t>(row.outputIds.size())}, {"prompt_tokens", row.promptTokens},
        {"spec_verify_count", row.specVerifyCount}, {"spec_acceptance_length", row.specAcceptanceLength},
        {"formatted_system_prompt", row.formatted.formattedSystemPrompt},
        {"formatted_complete_request", row.formatted.formattedCompleteRequest}};
    if (!row.logprobs.empty())
    {
        // llm_inference's layout: per step, {token_id, token, bytes, logprob}. `token` is sanitized
        // so the JSON stays valid; `bytes` keeps the raw piece.
        nlohmann::json steps = nlohmann::json::array();
        for (auto const& entries : row.logprobs)
        {
            nlohmann::json step = nlohmann::json::array();
            for (auto const& entry : entries)
            {
                std::string pending;
                std::string token = utf8::sanitizeUtf8Streaming(entry.piece, pending);
                token += utf8::sanitizeUtf8Flush(pending);
                nlohmann::json bytes = nlohmann::json::array();
                for (unsigned char const byte : entry.piece)
                {
                    bytes.push_back(static_cast<int32_t>(byte));
                }
                step.push_back({{"token_id", entry.tokenId}, {"token", std::move(token)}, {"bytes", std::move(bytes)},
                    {"logprob", entry.logprob}});
            }
            steps.push_back(std::move(step));
        }
        report["logprobs"] = std::move(steps);
    }
    return report;
}

std::unique_ptr<multiStage::IStage> createLlmRunner(multiStage::StageInitContext const& init)
{
    return std::make_unique<LlmRunner>(init);
}

multiStage::RunnerDecl llmRunnerDeclaration()
{
    multiStage::RunnerDecl decl;
    decl.name = kRunnerName;
    decl.kind = multiStage::StageKind::kAutoregressive;
    decl.ports = {{kTextPort, multiStage::PortKind::kText}, {kPromptIdsPort, multiStage::PortKind::kText},
        {kEmbedsPort, multiStage::PortKind::kEmbeds}, {kHiddenPort, multiStage::PortKind::kHidden}};
    decl.inputs = {kPromptInput, kAudioInput, kImageInput, kVisionInput, kSamplingInput, kRequestInput};
    decl.batches = true;
    decl.samplingKeys = {kTemperatureKey, kTopPKey, kTopKKey, kMaxGenerateLengthKey};
    return decl;
}

} // namespace rt
} // namespace trt_edgellm
