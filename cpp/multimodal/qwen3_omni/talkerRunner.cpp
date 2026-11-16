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

#include "multimodal/qwen3_omni/talkerRunner.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/inputLimits.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/talkerMLPKernels/talkerMLPKernels.h"
#include "multimodal/qwen3_omni/talkerRunnerKeys.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/multiStage/stageRegistry.h"
#include "runtime/state/pipelineIO.h"
#include "sampler/sampling.h"

#if defined(CUTE_DSL_GEMM_ENABLED)
#include "kernels/talkerMLPKernels/cuteDslGemmRunner.h"
#endif

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <nlohmann/json.hpp>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace qwen3_omni
{

using namespace talker;

namespace
{

using Json = nlohmann::json;

//! Codec frame rate and PCM geometry, used only to report the real-time factor.
constexpr int32_t kAudioSampleRate = 24000;
constexpr int32_t kAudioSamplesPerFrame = 1920;

//! Code predictor sampling defaults. The two layouts inherit different HF defaults:
//! the segments layout hardcodes code_predictor.generate(top_k=50, top_p=0.8) with the
//! GenerationConfig temperature of 1.0; the chunked layout routes the subtalker kwargs
//! (temperature 0.9, top_k 50, top_p 1.0) instead.
constexpr float kCpTemperatureSegments = 1.0F;
constexpr float kCpTopPSegments = 0.8F;
constexpr float kCpTemperatureChunked = 0.9F;
constexpr float kCpTopPChunked = 1.0F;
constexpr int32_t kCpTopK = 50;

//! HF `talker_suppressed_tokens` covers the last K ids of the talker vocabulary. Without
//! it the talker can sample reserved ids that have no codec embedding, which folds into
//! mispronunciations and inserted syllables.
constexpr int32_t kSuppressTailSegments = 1024;
constexpr int32_t kSuppressTailChunked = 3072;

std::string resolveSubdir(std::string const& root, nlohmann::json const& config, char const* key)
{
    if (!config.contains(key))
    {
        return {};
    }
    auto const value = config.at(key).get<std::string>();
    return value.empty() ? std::string{} : (std::filesystem::path(root) / value).string();
}

//! Non-owning [1, rows, hidden] FP16 view over a bound device pointer.
rt::Tensor viewOf(multiStage::PortValue const& value, char const* name)
{
    ELLM_CHECK(value.data != nullptr && value.shape.size() == 2,
        std::string("talker input '") + name + "' must be a [rows, hidden] device binding");
    ELLM_CHECK(value.elementBytes == 2, std::string("talker input '") + name + "' must be 16-bit");
    return rt::Tensor(value.data, rt::Coords{1, value.shape[0], value.shape[1]}, rt::DeviceType::kGPU,
        trt_edgellm::DataType::kHALF, name);
}

//! @param render Turns UTF-8 into ids when this deployment can; empty result means it cannot.
void appendIds(multiStage::PortValue const* value, char const* name, std::vector<int32_t>& out,
    std::function<std::vector<int32_t>(std::string const&)> const& render = {})
{
    if (value == nullptr || value->data == nullptr || value->shape.empty())
    {
        return;
    }
    if (value->elementBytes == 1 && render)
    {
        auto ids = render(std::string(static_cast<char const*>(value->data), static_cast<size_t>(value->shape.back())));
        ELLM_CHECK(!ids.empty(), std::string("talker input '") + name + "' is text this deployment cannot render");
        out.insert(out.end(), ids.begin(), ids.end());
        return;
    }
    // Without a tokenizer, text it cannot read is a wiring error rather than something to
    // guess at: reading UTF-8 as ids would also run four times past the end.
    ELLM_CHECK(value->elementBytes == static_cast<int32_t>(sizeof(int32_t)),
        std::string("talker input '") + name + "' must be int32 token ids, not " + std::to_string(value->elementBytes)
            + "-byte elements; this deployment has no tokenizer to render text with");
    auto const* ids = static_cast<int32_t const*>(value->data);
    out.insert(out.end(), ids, ids + value->shape.back());
}

//! Pick the four tensors of a 2-layer MLP out of one safetensors file.
bool extractMLPWeights(std::vector<rt::Tensor>& tensors, rt::Tensor& fc1Weight, rt::Tensor& fc1Bias,
    rt::Tensor& fc2Weight, rt::Tensor& fc2Bias, std::string const& projectionName)
{
    constexpr int32_t kExpectedTensorCount = 4;
    ELLM_CHECK(
        tensors.size() == kExpectedTensorCount, projectionName + ".safetensors should contain exactly 4 tensors");

    bool foundFC1Weight = false, foundFC1Bias = false, foundFC2Weight = false, foundFC2Bias = false;
    for (auto& tensor : tensors)
    {
        std::string const& name = tensor.getName();
        if (name.find("fc1.weight") != std::string::npos)
        {
            fc1Weight = std::move(tensor);
            foundFC1Weight = true;
        }
        else if (name.find("fc1.bias") != std::string::npos)
        {
            fc1Bias = std::move(tensor);
            foundFC1Bias = true;
        }
        else if (name.find("fc2.weight") != std::string::npos)
        {
            fc2Weight = std::move(tensor);
            foundFC2Weight = true;
        }
        else if (name.find("fc2.bias") != std::string::npos)
        {
            fc2Bias = std::move(tensor);
            foundFC2Bias = true;
        }
    }

    if (!foundFC1Weight || !foundFC1Bias || !foundFC2Weight || !foundFC2Bias)
    {
        LOG_ERROR("Failed to find all required tensors in %s.safetensors", projectionName.c_str());
        return false;
    }
    return true;
}

} // namespace

std::string TalkerRunner::lowered(std::string value)
{
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

TalkerRunner::TalkerRunner(multiStage::StageInitContext const& init)
    : mArtifactDir(init.artifactDir)
    , mStream(init.stream)
    , mMaxBatch(init.maxBatch)
    , mMaxSeq(init.maxSeq)
{
    ELLM_CHECK(init.config != nullptr, "talker stage built without a StageConfig");
    mStageId = init.config->id;
    auto const& config = init.config->config;

    ELLM_CHECK(config.contains("prefill_builder"),
        "talker stage config must name 'prefill_builder'; the prefill layouts are not interchangeable and there "
        "is no safe default");
    auto const builder = config.at("prefill_builder").get<std::string>();
    if (builder == "segments")
    {
        mPrefillBuilder = PrefillBuilder::kSegments;
    }
    else if (builder == "chunked")
    {
        mPrefillBuilder = PrefillBuilder::kChunked;
    }
    else if (builder == "tts")
    {
        mPrefillBuilder = PrefillBuilder::kTts;
    }
    else
    {
        ELLM_CHECK(
            false, "talker stage 'prefill_builder' is '" + builder + "'; expected 'segments', 'chunked' or 'tts'");
    }
    // Read off the wiring rather than a config flag, which could disagree with it: a talker
    // with nothing feeding its hidden port has no backbone, so it must render text itself.
    mSpeechOnly = init.config->findInput(kHiddenInput) == nullptr;

    mCodePredictorDir = resolveSubdir(mArtifactDir, config, kCodePredictorArtifactKey);
    ELLM_CHECK(!mCodePredictorDir.empty(),
        std::string("talker stage config must name '") + kCodePredictorArtifactKey
            + "'; the code predictor engine is loaded from that directory");
    mCodePredictorCheckpoint = resolveSubdir(mCodePredictorDir, config, kCodePredictorCheckpointKey);
    // 0 keeps the autoregressive RVQ loop; 2..8 asks the predictor for a verify window.
    mCpSpecVerifySize = config.value(kCpSpecVerifySizeKey, 0);
    mCloneEncoderDir = resolveSubdir(mArtifactDir, config, kCloneEncoderArtifactKey);

    mBackboneDir = resolveSubdir(mArtifactDir, config, "tokenizer_dir");
    if (mBackboneDir.empty())
    {
        mBackboneDir = std::filesystem::path(mArtifactDir).parent_path().string();
    }
    mCheckpointDir = resolveSubdir(mArtifactDir, config, "checkpoint");

    mDefaultSpeaker = config.value("speaker", std::string{});
    mAssistantInstruct = config.value("assistant_instruct", std::string{});
    mLanguage = config.value("language", std::string{});
    mTalkerLanguage = config.value("talker_language", std::string{});
    mReportCodes = config.value("report_codes", false);
    mSamplingConfig.maxAudioLength = config.value("max_audio_length", 4096);
    mSamplingConfig.temperature = config.value("temperature", 0.0F);
    mSamplingConfig.topK = config.value("top_k", 0);
    mSamplingConfig.topP = config.value("top_p", 0.0F);
    mSamplingConfig.repetitionPenalty = config.value("repetition_penalty", 1.05F);
    mSamplingConfig.cpTemperature = config.value("cp_temperature", 0.0F);
    mSamplingConfig.cpTopK = config.value("cp_top_k", 0);
    mSamplingConfig.cpTopP = config.value("cp_top_p", 0.0F);
    applySampling(nlohmann::json::object());
}

void TalkerRunner::applySampling(nlohmann::json const& overrides)
{
    // A zero means "use the architecture default", which is how the checkpoint's own values stay reachable without
    // spelling them out.
    mTemperature = overrides.value(kTalkerTemperatureKey, mSamplingConfig.temperature);
    mTopK = overrides.value(kTalkerTopKKey, mSamplingConfig.topK);
    mTopP = overrides.value(kTalkerTopPKey, mSamplingConfig.topP);
    mRepetitionPenalty = overrides.value(kRepetitionPenaltyKey, mSamplingConfig.repetitionPenalty);
    mCpTemperature = overrides.value(kCpTemperatureKey, mSamplingConfig.cpTemperature);
    mCpTopK = overrides.value(kCpTopKKey, mSamplingConfig.cpTopK);
    mCpTopP = overrides.value(kCpTopPKey, mSamplingConfig.cpTopP);

    // A request may ask past the configured default; the KV room left after prefill caps it.
    mMaxAudioLength = std::max(1, overrides.value(kMaxAudioLengthKey, mSamplingConfig.maxAudioLength));
}
void TalkerRunner::applyStyle(nlohmann::json const& style, multiStage::PortValue const* speakerCodes)
{
    mRequestInstruct = style.value("assistant_instruct", std::string{});
    mRequestLanguage = style.value("language", std::string{});
    mRequestTalkerLanguage = style.value("talker_language", std::string{});
    auto const readIds = [&style](char const* key, std::vector<int32_t>& out) {
        out.clear();
        if (auto const ids = style.find(key); ids != style.end() && ids->is_array())
        {
            out.reserve(ids->size());
            for (auto const& id : *ids)
            {
                out.push_back(id.get<int32_t>());
            }
        }
    };
    readIds("system_instruct_ids", mSystemInstructIds);
    // The ids form lets a caller own the chat template, which matters because the row plan
    // slices the reference transcript at fixed offsets into its assistant wrapper.
    readIds("instruct_ids", mInstructIds);
    readIds("ref_text_ids", mRefTextIds);
    // The plain-text form of the same three fields, wrapped here so the fixed offsets above
    // still land where the row plan expects them.
    auto const readText = [this, &style](char const* key, char const* role, std::vector<int32_t>& out) {
        auto const text = style.value(key, std::string{});
        if (!out.empty() || text.empty() || mTokenizer == nullptr)
        {
            return;
        }
        auto const wrapped = role == nullptr ? text : "<|im_start|>" + std::string(role) + "\n" + text + "<|im_end|>\n";
        for (auto const rank : mTokenizer->encode(wrapped))
        {
            out.push_back(static_cast<int32_t>(rank));
        }
    };
    readText("system_instruct", nullptr, mSystemInstructIds);
    readText("instruct", "user", mInstructIds);
    readText("ref_text", "assistant", mRefTextIds);

    mPromptSpeakerCodes.clear();
    if (speakerCodes == nullptr || speakerCodes->data == nullptr)
    {
        return;
    }
    ELLM_CHECK(speakerCodes->shape.size() == 2 && speakerCodes->elementBytes == static_cast<int32_t>(sizeof(int32_t)),
        "stage '" + mStageId + "': input '" + kSpeakerCodesInput
            + "' must be an int32 [frames, codes_per_frame] reference-voice binding");
    auto const frames = speakerCodes->shape[0];
    auto const perFrame = speakerCodes->shape[1];
    auto const* values = static_cast<int32_t const*>(speakerCodes->data);
    mPromptSpeakerCodes.resize(static_cast<size_t>(frames));
    for (int64_t f = 0; f < frames; ++f)
    {
        auto const* row = values + f * perFrame;
        mPromptSpeakerCodes[static_cast<size_t>(f)].assign(row, row + perFrame);
    }
}

TalkerRunner::~TalkerRunner()
{
#if defined(CUTE_DSL_GEMM_ENABLED)
    CuteDslGemmRunner::unloadKernelModule();
#endif
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

std::vector<int32_t> TalkerRunner::renderConversation(std::string const& envelope) const
{
    if (mTokenizer == nullptr || mChatTemplate == nullptr)
    {
        return {};
    }
    auto parsed = nlohmann::json::parse(envelope, nullptr, /*allow_exceptions=*/false);
    if (!parsed.is_object() || !parsed.contains("messages") || !parsed.at("messages").is_array())
    {
        // Bare text is what to say, so it is the assistant's turn.
        nlohmann::json const turn{{"role", "assistant"}, {"content", envelope}};
        parsed = nlohmann::json{{"messages", nlohmann::json::array({turn})}};
    }

    // The chunked plan feeds user content through hidden_projection, so with no backbone
    // upstream those rows have nothing behind them and the turn cannot be spoken.
    bool const dropsUserTurns = mPrefillBuilder == PrefillBuilder::kChunked;

    namespace security = limits::security;
    ELLM_CHECK(parsed.at("messages").size() <= security::kMaxMessagesPerRequest,
        "stage '" + mStageId + "': more than " + std::to_string(security::kMaxMessagesPerRequest) + " messages");
    rt::LLMGenerationRequest::Request turnSet;
    for (auto const& turn : parsed.at("messages"))
    {
        ELLM_CHECK(turn.is_object() && turn.contains("role"), "stage '" + mStageId + "': a turn needs a 'role'");
        rt::Message message;
        message.role = turn.at("role").get<std::string>();
        if (dropsUserTurns && message.role == "user")
        {
            LOG_WARNING(
                "stage '%s': dropping a user turn; speaking it would need thinker hidden states", mStageId.c_str());
            continue;
        }
        ELLM_CHECK(turn.contains("content"), "stage '" + mStageId + "': a turn needs 'content'");
        auto const& content = turn.at("content");
        ELLM_CHECK(content.is_string() || content.is_array(),
            "stage '" + mStageId + "': a turn's content must be a string or an array");
        // Speech has no use for media, so an array is text items read as one string.
        std::string text = content.is_string() ? content.get<std::string>() : std::string{};
        if (content.is_array())
        {
            for (auto const& item : content)
            {
                ELLM_CHECK(item.is_object() && item.value("type", std::string{}) == "text",
                    "stage '" + mStageId + "': only text content can be spoken");
                text += item.value("text", std::string{});
            }
        }
        ELLM_CHECK(text.size() <= security::kMaxMessageContentSizeBytes,
            "stage '" + mStageId + "': a turn holds " + std::to_string(text.size()) + " bytes, more than "
                + std::to_string(security::kMaxMessageContentSizeBytes));
        message.contents.push_back({"text", std::move(text)});
        turnSet.messages.push_back(std::move(message));
    }

    chat_template::ChatTemplate::Options options;
    if (dropsUserTurns)
    {
        // It speaks the last assistant turn already in the stream, so an empty one appended
        // after it would be what it reads.
        options.addGenerationPrompt = false;
    }
    else
    {
        options.applyTemplate = parsed.value("apply_chat_template", true);
        // On by default: the tts layout strips a fixed suffix that ends in the generation
        // prompt, and with it off would strip text instead.
        options.addGenerationPrompt = parsed.value("add_generation_prompt", true);
        options.enableThinking = parsed.value("enable_thinking", false);
    }
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ELLM_CHECK(mChatTemplate->apply(turnSet, formatted, options), "stage '" + mStageId + "': chat template failed");

    auto const ranks = mTokenizer->encode(formatted.formattedCompleteRequest);
    std::vector<int32_t> ids;
    ids.reserve(ranks.size());
    for (auto const rank : ranks)
    {
        ids.push_back(static_cast<int32_t>(rank));
    }
    return ids;
}

TalkerRunner::Row& TalkerRunner::rowFor(multiStage::StageExecContext const& ctx)
{
    ELLM_CHECK(ctx.slot != nullptr, "talker stage reached without a request slot");
    auto const slot = static_cast<size_t>(ctx.slot->batchSlot);
    ELLM_CHECK(ctx.slot->batchSlot >= 0 && slot < mRows.size(),
        "stage '" + mStageId + "': batch slot " + std::to_string(ctx.slot->batchSlot) + " is outside max_batch "
            + std::to_string(mMaxBatch));
    return mRows[slot];
}

TalkerRunner::Row const& TalkerRunner::rowFor(multiStage::RequestSlot const& slot) const
{
    auto const index = static_cast<size_t>(slot.batchSlot);
    ELLM_CHECK(slot.batchSlot >= 0 && index < mRows.size(),
        "stage '" + mStageId + "': batch slot " + std::to_string(slot.batchSlot) + " is outside max_batch "
            + std::to_string(mMaxBatch));
    return mRows[index];
}

rt::Tensor TalkerRunner::sliceOf(rt::Tensor const& buffer, int32_t slot, rt::Coords const& shape)
{
    int64_t elements = 1;
    for (int32_t i = 0; i < shape.getNumDims(); ++i)
    {
        elements *= shape[i];
    }
    auto const bytes = static_cast<size_t>(elements) * rt::utils::getTypeSize(buffer.getDataType());
    auto* const base = static_cast<char*>(const_cast<void*>(buffer.rawPointer())) + static_cast<size_t>(slot) * bytes;
    return rt::Tensor(base, shape, buffer.getDeviceType(), buffer.getDataType());
}

void TalkerRunner::preallocate()
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::preallocate", nvtx_colors::YELLOW);

    ELLM_CHECK(mMaxBatch >= 1, "talker stage needs a positive max_batch");
    ELLM_CHECK(mMaxSeq > 0, "talker stage needs a positive max_seq");
    ELLM_CHECK(mMaxAudioLength > 0, "talker stage 'max_audio_length' must be positive");

    auto const loadTokenizer = [](std::string const& dir) -> std::unique_ptr<tokenizer::Tokenizer> {
        auto tokenizer = std::make_unique<tokenizer::Tokenizer>();
        if (!std::filesystem::exists(std::filesystem::path(dir) / "tokenizer.json") || !tokenizer->loadFromHF(dir))
        {
            return nullptr;
        }
        return tokenizer;
    };
    if (mSpeechOnly)
    {
        for (auto const& dir : {mBackboneDir, mArtifactDir})
        {
            auto tokenizer = loadTokenizer(dir);
            if (tokenizer == nullptr)
            {
                continue;
            }
            auto const special = [&tokenizer](tokenizer::Rank id) {
                return id >= 0 ? tokenizer->idToPiece(id, /*skipSpecialTokens=*/false) : std::string{};
            };
            auto chatTemplate = std::make_unique<chat_template::ChatTemplate>();
            if (chatTemplate->load(dir, special(tokenizer->getBosId()), special(tokenizer->getEosId())))
            {
                mTokenizer = std::move(tokenizer);
                mChatTemplate = std::move(chatTemplate);
                LOG_INFO("Talker can render conversations: tokenizer and chat template loaded from %s", dir.c_str());
                break;
            }
        }
    }

    // Every layout writes tts_pad/bos/eos rows; all but tts also find chat-template tokens in the stream, and the
    // segments layout its media placeholders. The rendering tokenizer answers if one was loaded; otherwise one is
    // loaded for the lookup only, since setting mTokenizer would also start rendering text inputs.
    {
        std::unique_ptr<tokenizer::Tokenizer> lookup;
        tokenizer::Tokenizer const* source = mTokenizer.get();
        for (auto const& dir : {mBackboneDir, mArtifactDir})
        {
            if (source != nullptr)
            {
                break;
            }
            lookup = loadTokenizer(dir);
            source = lookup.get();
        }
        ELLM_CHECK(source != nullptr,
            "stage '" + mStageId + "': the talker needs tokenizer.json in '" + mBackboneDir + "' or '" + mArtifactDir
                + "' to look up its special tokens");
        if (mSpeechOnly && mTokenizer == nullptr)
        {
            LOG_WARNING(
                "stage '%s': no chat template beside a tokenizer in '%s' or '%s'; only pre-tokenized requests "
                "can be served",
                mStageId.c_str(), mBackboneDir.c_str(), mArtifactDir.c_str());
        }
        mTokenIds = resolveTalkerTokenIds(*source, /*withChatTemplate=*/mPrefillBuilder != PrefillBuilder::kTts,
            /*withMediaPads=*/mPrefillBuilder == PrefillBuilder::kSegments);
        LOG_INFO(
            "Talker token ids: tts pad/bos/eos %d/%d/%d, im_start %d, im_end %d, system %d, user %d, assistant %d, "
            "think %d/%d, newlines %d/%d, audio/image/video pads %d/%d/%d",
            mTokenIds.ttsPad, mTokenIds.ttsBos, mTokenIds.ttsEos, mTokenIds.imStart, mTokenIds.imEnd,
            mTokenIds.systemRole, mTokenIds.userRole, mTokenIds.assistantRole, mTokenIds.thinkOpen,
            mTokenIds.thinkClose, mTokenIds.newline, mTokenIds.doubleNewline, mTokenIds.audioPad, mTokenIds.imagePad,
            mTokenIds.videoPad);
    }

    ELLM_CHECK(validateAndFillConfig(mArtifactDir), "failed to read the talker config");
    ELLM_CHECK(initializeTalkerEngine(mArtifactDir, mCheckpointDir), "failed to load the talker engine");

#if defined(CUTE_DSL_GEMM_ENABLED)
    ELLM_CHECK(CuteDslGemmRunner::loadKernelModule(), "failed to load the CuTe DSL GEMM kernel module");
#endif

    ELLM_CHECK(allocateBuffers(), "failed to claim the talker workspace");
    ELLM_CHECK(loadTalkerWeights(mArtifactDir, mStream), "failed to load the talker weights");
    loadTextEmbeddingTable(mArtifactDir, mStream);
    initializeTTSEmbeddings(mStream);

    // Row-assembly staging and the reference-voice workspace, claimed once at the widest shape
    // the engine admits.
    {
        int64_t const maxRows = mTalkerLLMConfig.maxSupportedInputLength;
        int64_t const stagingBytes = maxRows * static_cast<int64_t>(sizeof(kernel::PrefillRowDesc));
        mPrefillRows.reserve(static_cast<size_t>(maxRows));
        mPrefillDescsHost = rt::Tensor(
            {mMaxBatch, stagingBytes}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT8, "prefillDescsHost");
        mPrefillDescsDevice
            = rt::Tensor({stagingBytes}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8, "prefillDescsDevice");

        constexpr int64_t kMaxRefFrames = 512; // about 40 s of reference audio at 12.5 Hz
        int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
        mVoiceCloneXVector
            = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "cloneXVector");
        mIclFrameSumBuffer = rt::Tensor(
            {kMaxRefFrames, hiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "iclFrameSums");

        if (mSpeakerCodecEmbeddings.rawPointer() != nullptr)
        {
            auto const& speakerShape = mSpeakerCodecEmbeddings.getShape();
            ELLM_CHECK(speakerShape.getNumDims() == 3,
                "speaker_codec_embeddings must be [maxSpeakers, numCodeGroups, speakerEmbedLen]");
            mHostSpeakerRow = rt::Tensor({speakerShape[1] * speakerShape[2]}, rt::DeviceType::kCPU,
                trt_edgellm::DataType::kINT64, "hostSpeakerRow");
        }
    }

    // Reference encoders are optional: only Base checkpoints ship them.
    if (!mCloneEncoderDir.empty()
        && std::filesystem::exists(std::filesystem::path(mCloneEncoderDir) / "speaker_encoder.engine"))
    {
        mCloneEncoders = std::make_unique<rt::CloneEncoderRunner>(mCloneEncoderDir, mStream);
        LOG_INFO("Loaded the reference voice encoders from %s", mCloneEncoderDir.c_str());
    }

    // Every frame takes a KV row, so the cache capacity bounds any utterance.
    mRows.resize(static_cast<size_t>(mMaxBatch));
    for (auto& row : mRows)
    {
        row.flatCodes.assign(
            static_cast<size_t>(mTalkerLLMConfig.maxKVCacheCapacity) * static_cast<size_t>(mNumCodesPerFrame), 0);
        row.seenTokens.assign(static_cast<size_t>(mTalkerConfig.talkerVocabSize), 0);
    }
    mLive.reserve(static_cast<size_t>(mMaxBatch));
    mPrefillMembers.reserve(static_cast<size_t>(mMaxBatch));
    mFrameCodes.assign(static_cast<size_t>(mMaxBatch), std::vector<int32_t>(static_cast<size_t>(mNumCodesPerFrame), 0));
    mCodeZero = rt::Tensor({mMaxBatch}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "mCodeZero");
    mCohortSlots.reserve(static_cast<size_t>(mMaxBatch));
    mQueryLengths.assign(static_cast<size_t>(mMaxBatch), 1);
    mPastLengths.assign(static_cast<size_t>(mMaxBatch), 0);
    mTalkerStepQueryLengths.assign(static_cast<size_t>(mMaxBatch), 1);
    mTalkerStepPastLengths.assign(static_cast<size_t>(mMaxBatch), 0);
    mTextTokenIds.reserve(static_cast<size_t>(mMaxSeq));
    mChunkStream.remainingTextTokens.reserve(static_cast<size_t>(mMaxSeq));

    // Ordering: the code predictor borrows the embedding tables and the codec-hidden buffer,
    // and its graph capture runs after the talker's own, so it goes last.
    setUpCodePredictor(mStream);
}

void TalkerRunner::setUpCodePredictor(cudaStream_t stream)
{
    mCodePredictor.preallocate(mCodePredictorDir, mCodePredictorCheckpoint, mMaxBatch, mCpSpecVerifySize, stream);

    // The ICL sum reads the talker table first, then one code predictor table per group, so
    // the pointer array can only be built once the predictor has claimed its own.
    {
        int64_t const tablePtrBytes = static_cast<int64_t>(mTalkerConfig.numCodeGroups) * sizeof(void*);
        mIclTablePtrsHost
            = rt::Tensor({tablePtrBytes}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT8, "iclTablePtrsHost");
        auto** tablePtrs = static_cast<void const**>(mIclTablePtrsHost.rawPointer());
        tablePtrs[0] = mTalkerEmbeddingTable.rawPointer();
        for (int32_t g = 1; g < mTalkerConfig.numCodeGroups; ++g)
        {
            tablePtrs[g] = mCodePredictor.codecEmbedTable(g - 1).rawPointer();
        }
        mIclTablePtrsGpu
            = rt::Tensor({tablePtrBytes}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8, "iclTablePtrs");
        CUDA_CHECK(cudaMemcpyAsync(mIclTablePtrsGpu.rawPointer(), tablePtrs, static_cast<size_t>(tablePtrBytes),
            cudaMemcpyHostToDevice, stream));
    }

    // One arena for both engines when the talker's is already large enough: they share this
    // stage's stream, so the two contexts never execute concurrently.
    if (static_cast<size_t>(mExecContextMemory.getMemoryCapacity()) >= mCodePredictor.requiredContextMemory())
    {
        mCodePredictor.setContextMemory(mExecContextMemory);
    }

    CodePredictor::Binding binding;
    binding.talkerHiddenSize = mTalkerConfig.talkerHiddenSize;
    binding.numCodesPerFrame = mNumCodesPerFrame;
    binding.maxBatchSize = mMaxBatch;
    // Which table code_0 lives in follows from the prefill layout: the chunked layout keeps
    // text and codec embeddings in separate tables.
    binding.codeZeroEmbedTable = usesChunkedText() ? &mTalkerCodecEmbedTable : &mTalkerEmbeddingTable;
    binding.codecHiddens = &mCodecHiddensBuffer;
    mCodePredictor.bind(binding);

    ELLM_CHECK(
        !usesChunkedText() || buildCodecEmbedPointerTable(stream), "could not build the codec embedding pointer table");
    captureDecodeGraphs(stream);
}

bool TalkerRunner::validateAndFillConfig(std::string const& engineDir)
{
    std::filesystem::path const configPath = std::filesystem::path(engineDir) / "config.json";
    LOG_INFO("Loading Talker config from: %s", configPath.string().c_str());

    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.string().c_str());
        return false;
    }

    Json configJson;
    try
    {
        configJson = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& error)
    {
        LOG_ERROR("Failed to parse config: %s", error.what());
        return false;
    }

    mTalkerConfig.thinkerHiddenSize = configJson.value("thinker_hidden_size", 2048);
    mTalkerConfig.talkerHiddenSize = configJson["hidden_size"].get<int32_t>();
    mTalkerConfig.talkerVocabSize = configJson["vocab_size"].get<int32_t>();

    mTalkerConfig.numCodeGroups = configJson["num_code_groups"].get<int32_t>();
    ELLM_CHECK(mTalkerConfig.numCodeGroups >= 2 && mTalkerConfig.numCodeGroups <= kMaxCodeGroups,
        "num_code_groups must be in [2, " + std::to_string(kMaxCodeGroups)
            + "], got: " + std::to_string(mTalkerConfig.numCodeGroups));
    mNumRvqLayers = mTalkerConfig.numCodeGroups - 1;
    mNumCodesPerFrame = mTalkerConfig.numCodeGroups;

    mTalkerConfig.maxSeqLen = configJson.value("max_position_embeddings", 8192);

    constexpr int32_t kMaxReasonableVocabSize = 200000;
    constexpr int32_t kMaxReasonableHiddenSize = 16384;
    constexpr int32_t kMaxReasonableSeqLen = 131072;

    ELLM_CHECK(mTalkerConfig.talkerVocabSize > 0 && mTalkerConfig.talkerVocabSize < kMaxReasonableVocabSize,
        "Invalid talker vocab size: " + std::to_string(mTalkerConfig.talkerVocabSize));
    ELLM_CHECK(mTalkerConfig.thinkerHiddenSize > 0 && mTalkerConfig.thinkerHiddenSize < kMaxReasonableHiddenSize,
        "Invalid backbone hidden size: " + std::to_string(mTalkerConfig.thinkerHiddenSize));
    ELLM_CHECK(mTalkerConfig.talkerHiddenSize > 0 && mTalkerConfig.talkerHiddenSize < kMaxReasonableHiddenSize,
        "Invalid talker hidden size: " + std::to_string(mTalkerConfig.talkerHiddenSize));
    ELLM_CHECK(mTalkerConfig.maxSeqLen > 0 && mTalkerConfig.maxSeqLen < kMaxReasonableSeqLen,
        "Invalid max sequence length: " + std::to_string(mTalkerConfig.maxSeqLen));

    mTalkerConfig.codecNothinkId = configJson["codec_nothink_id"].get<int32_t>();
    mTalkerConfig.codecThinkId = configJson.value("codec_think_id", -1);
    mTalkerConfig.codecThinkBosId = configJson["codec_think_bos_id"].get<int32_t>();
    mTalkerConfig.codecThinkEosId = configJson["codec_think_eos_id"].get<int32_t>();
    mTalkerConfig.codecPadId = configJson["codec_pad_id"].get<int32_t>();
    mTalkerConfig.codecBosId = configJson["codec_bos_id"].get<int32_t>();
    // Checkpoints spell this either way; codec_eos_token_id wins when both are present.
    mTalkerConfig.codecEosId = configJson.contains("codec_eos_token_id")
        ? configJson["codec_eos_token_id"].get<int32_t>()
        : configJson["codec_eos_id"].get<int32_t>();

    if (configJson.contains("talker_assistant_prompt_id_mapping")
        && configJson["talker_assistant_prompt_id_mapping"].is_object())
    {
        for (auto const& [name, ids] : configJson["talker_assistant_prompt_id_mapping"].items())
        {
            if (!ids.is_array())
            {
                continue;
            }
            std::vector<int32_t> tokens;
            tokens.reserve(ids.size());
            for (auto const& tid : ids)
            {
                tokens.push_back(tid.get<int32_t>());
            }
            mAssistantPromptIds[name] = std::move(tokens);
        }
        LOG_INFO("Loaded %zu talker_assistant_prompt_id_mapping entries", mAssistantPromptIds.size());
    }
    if (configJson.contains("talker_language_id") && configJson["talker_language_id"].is_object())
    {
        for (auto const& [lang, id] : configJson["talker_language_id"].items())
        {
            if (id.is_number_integer())
            {
                mLanguageIds[lang] = id.get<int32_t>();
            }
        }
        LOG_INFO("Loaded %zu talker_language_id entries", mLanguageIds.size());
    }

    // The tts layout conditions on its own table, and reads a speaker's dialect as the
    // language when the request names none.
    mTalkerConfig.ttsModelType = configJson.value("tts_model_type", std::string{});
    if (configJson.contains("codec_language_id") && configJson["codec_language_id"].is_object())
    {
        for (auto const& [lang, id] : configJson["codec_language_id"].items())
        {
            if (id.is_number_integer())
            {
                mTalkerConfig.codecLanguageIds[lowered(lang)] = id.get<int32_t>();
            }
        }
        LOG_INFO("Loaded %zu codec_language_id entries", mTalkerConfig.codecLanguageIds.size());
    }
    if (configJson.contains("spk_is_dialect") && configJson["spk_is_dialect"].is_object())
    {
        for (auto const& [speaker, dialect] : configJson["spk_is_dialect"].items())
        {
            if (dialect.is_string())
            {
                mTalkerConfig.speakerDialects[lowered(speaker)] = lowered(dialect.get<std::string>());
            }
        }
    }

    mTalkerConfig.defaultSpeakerId = configJson.value("default_speaker_id", 2301);

    mTalkerConfig.maxThinkerToTalkerMmTokens
        = configJson.value("max_thinker_to_talker_mm_tokens", mTalkerConfig.maxThinkerToTalkerMmTokens);

    int32_t const suppressTail = usesChunkedText() ? kSuppressTailChunked : kSuppressTailSegments;
    mTalkerConfig.talkerSuppressStart = std::max(0, mTalkerConfig.talkerVocabSize - suppressTail);

    if (configJson.contains("speaker_id") && configJson["speaker_id"].is_object())
    {
        for (auto const& [speakerName, speakerId] : configJson["speaker_id"].items())
        {
            mSpeakerIdMap[speakerName] = speakerId.get<int32_t>();
        }
        LOG_INFO("Loaded %zu speaker IDs from config", mSpeakerIdMap.size());

        // speaker_system_prompt_id is keyed by name in the config but looked up by id at
        // prefill time, so it is re-indexed here.
        if (configJson.contains("speaker_system_prompt_id") && configJson["speaker_system_prompt_id"].is_object())
        {
            for (auto const& [speakerName, ids] : configJson["speaker_system_prompt_id"].items())
            {
                auto const it = mSpeakerIdMap.find(speakerName);
                if (it == mSpeakerIdMap.end())
                {
                    continue;
                }
                std::vector<int32_t> tokens;
                tokens.reserve(ids.size());
                for (auto const& tid : ids)
                {
                    tokens.push_back(tid.get<int32_t>());
                }
                mSpeakerSystemPromptIds[it->second] = std::move(tokens);
            }
            LOG_INFO("Loaded %zu speaker_system_prompt_id entries", mSpeakerSystemPromptIds.size());
        }

        std::filesystem::path const voiceMapPath = std::filesystem::path(engineDir) / "voice_map.json";
        if (std::filesystem::exists(voiceMapPath))
        {
            std::ifstream voiceMapStream(voiceMapPath);
            nlohmann::json voiceMap;
            try
            {
                voiceMapStream >> voiceMap;
                for (auto const& [friendlyName, internalName] : voiceMap.items())
                {
                    if (!internalName.is_string())
                    {
                        continue;
                    }
                    mVoiceAliasMap[lowered(friendlyName)] = internalName.get<std::string>();
                }
                LOG_INFO("Loaded %zu voice_map.json aliases", mVoiceAliasMap.size());
            }
            catch (std::exception const& error)
            {
                LOG_WARNING("Failed to parse %s: %s", voiceMapPath.string().c_str(), error.what());
            }
        }
    }

    LOG_INFO("Talker config: vocabSize=%d, hiddenSize=%d, backboneHiddenSize=%d, codesPerFrame=%d, defaultSpeaker=%d",
        mTalkerConfig.talkerVocabSize, mTalkerConfig.talkerHiddenSize, mTalkerConfig.thinkerHiddenSize,
        mNumCodesPerFrame, mTalkerConfig.defaultSpeakerId);
    return true;
}

bool TalkerRunner::initializeTalkerEngine(std::string const& engineDir, std::string const& checkpointDir)
{
    std::filesystem::path const enginePath = std::filesystem::path(engineDir) / "llm.engine";
    std::filesystem::path const configPath = std::filesystem::path(engineDir) / "config.json";
    LOG_INFO("Loading Talker LLM engine from: %s", enginePath.string().c_str());

    try
    {
        mTalkerLLMConfig = rt::parseEngineConfig(configPath);
        auto talkerWeights = std::make_unique<rt::ExternalWeightManager>();
        talkerWeights->load(engineDir, configPath, mStream, checkpointDir);
        mTalkerExec = rt::EngineExecutor::createForLLM(enginePath, mTalkerLLMConfig, std::nullopt, nullptr,
            rt::EngineRefitOptions{talkerWeights->takeRefitWeights(), mStream});
        rt::validateAgainstEngine(mTalkerLLMConfig, *mTalkerExec, "omni_talker");

        // Talker exports differ in what hidden_states carries: some emit the whole sequence,
        // others gather the last token inside the graph and emit a single row. The prefill
        // buffer must follow whichever the engine declares, or the last-row read lands on
        // memory the engine never wrote.
        {
            auto const hiddenDims = mTalkerExec->getEngine().getTensorShape(binding_names::kOutputHiddenStates);
            mTalkerHiddenIsGathered = (hiddenDims.nbDims == 3 && hiddenDims.d[1] == 1);
            LOG_INFO("Talker hidden_states is %s", mTalkerHiddenIsGathered ? "gathered (one row)" : "full-sequence");
        }
        std::unordered_map<std::string, std::string> emptyLoraMap;
        mTalkerSharedRes = rt::SharedResources::createForLLM(mTalkerLLMConfig, emptyLoraMap, mStream);
        mTalkerPipelineIO = std::make_unique<rt::PipelineIO>(rt::PipelineIO::createForLLM(mTalkerLLMConfig, mStream));
        mTalkerStepPreparer = std::make_unique<rt::StepPreparer>(mTalkerLLMConfig);
        rt::buildTensorMap(
            mTalkerTensorMap, *mTalkerPipelineIO, *mTalkerSharedRes, mTalkerLLMConfig, /*kvCacheIndex=*/0);
        mTalkerSharedRes->externalWeightManager = std::move(talkerWeights);
        mTalkerSharedRes->externalWeightManager->validateAgainstEngine(*mTalkerExec, "talker");
        mTalkerSharedRes->externalWeightManager->registerTensorMapEntries(mTalkerTensorMap);

        int64_t const contextBytes = mTalkerExec->getRequiredContextMemorySize();
        mExecContextMemory = rt::Tensor(
            {contextBytes}, rt::DeviceType::kGPU, trt_edgellm::DataType::kUINT8, "TalkerRunner::mExecContextMemory");
        if (!mTalkerExec->setContextMemory(mExecContextMemory))
        {
            LOG_ERROR("Failed to set context memory for the Talker engine");
            return false;
        }

        ELLM_CHECK(mTalkerLLMConfig.maxSupportedBatchSize >= mMaxBatch,
            "talker engine supports batch " + std::to_string(mTalkerLLMConfig.maxSupportedBatchSize) + ", stage asks "
                + std::to_string(mMaxBatch));

        LOG_INFO("Talker LLM engine loaded: vocabSize=%d, hiddenSize=%d", mTalkerLLMConfig.vocabSize,
            mTalkerLLMConfig.hiddenSize);
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Failed to load Talker LLM engine: %s", error.what());
        return false;
    }
    return true;
}

bool TalkerRunner::allocateBuffers()
{
    int64_t const maxSeqLen = mTalkerConfig.maxSeqLen;
    // A prefill never exceeds one engine input; only the text buffers need maxSeqLen.
    int64_t const maxPrefillLen = mTalkerLLMConfig.maxSupportedInputLength;
    int64_t const thinkerHiddenSize = mTalkerConfig.thinkerHiddenSize;
    int64_t const talkerHiddenSize = mTalkerConfig.talkerHiddenSize;
    // Buffers that outlive one prepare() carry a row per cohort member; the assembly scratch
    // below does not, because prepare() stages one member at a time.
    int64_t const maxBatch = mMaxBatch;

    try
    {
        mThinkerEmbedBuffer = rt::Tensor(
            {maxSeqLen, thinkerHiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "mThinkerEmbedBuffer");
        mGpuTokenIdsBuffer
            = rt::Tensor({1, maxSeqLen}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "mGpuTokenIdsBuffer");
        mHostTokenIds
            = rt::Tensor({maxBatch, maxSeqLen}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "mHostTokenIds");
        mMLPWorkspace = rt::Tensor(
            {maxSeqLen, thinkerHiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "mMLPWorkspace");
        mProjectedBuffer = rt::Tensor(
            {maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "mProjectedBuffer");
        // Assembly scratch: every prefill builder writes it from row 0, and prepare() then copies
        // the finished block into this member's own row below. The chunked re-prefill also appends
        // to it and runs the engine on it directly, which admit() keeps to one member.
        mTalkerInputEmbeds = rt::Tensor({maxPrefillLen + kPreambleScratchRows, talkerHiddenSize}, rt::DeviceType::kGPU,
            trt_edgellm::DataType::kHALF, "mTalkerInputEmbeds");
        mTalkerPrefillRows = rt::Tensor({maxBatch * maxPrefillLen, talkerHiddenSize}, rt::DeviceType::kGPU,
            trt_edgellm::DataType::kHALF, "mTalkerPrefillRows");
        mGatherIndicesBuffer
            = rt::Tensor({maxSeqLen}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "mGatherIndicesBuffer");
        mHostGatherIndices = rt::Tensor(
            {maxBatch, maxSeqLen}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "mHostGatherIndices");
        // Only the segments layout adds text per frame, and prepare() caps that text at max_seq.
        // One extra row: the trailing text always ends with a tts_eos row.
        if (mPrefillBuilder == PrefillBuilder::kSegments)
        {
            int64_t const maxTrailingRows = static_cast<int64_t>(mMaxSeq) + 1;
            mTrailingTextHidden = rt::Tensor({maxTrailingRows, talkerHiddenSize}, rt::DeviceType::kGPU,
                trt_edgellm::DataType::kHALF, "mTrailingTextHidden");
            mTrailingRows = rt::Tensor({maxBatch * maxTrailingRows, talkerHiddenSize}, rt::DeviceType::kGPU,
                trt_edgellm::DataType::kHALF, "mTrailingRows");
        }

        mTalkerLogits = rt::Tensor({maxBatch, mTalkerConfig.talkerVocabSize}, rt::DeviceType::kGPU,
            trt_edgellm::DataType::kFLOAT, "mTalkerLogits");
        mTalkerSelectedIndices
            = rt::Tensor({maxBatch, 1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "mTalkerSelectedIndices");
        mHostSelectedTokenIds
            = rt::Tensor({maxBatch}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "mHostSelectedTokenIds");
        mSeenCodecTokensBuf = rt::Tensor({maxBatch, mTalkerLLMConfig.maxKVCacheCapacity}, rt::DeviceType::kGPU,
            trt_edgellm::DataType::kINT32, "mSeenCodecTokensBuf");
        mHostReuseKVCacheLengths
            = rt::Tensor({maxBatch}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "mHostReuseKVCacheLengths");

        // Sized for the widest sampler shape a request can ask for: the top-k path grows with topK,
        // which takeSampling() caps at the vocabulary, and the top-p path with the vocabulary.
        auto const batch = static_cast<int32_t>(maxBatch);
        int32_t const vocab = mTalkerConfig.talkerVocabSize;
        SamplingParams const topPParams(batch, vocab, /*temperature=*/1.0F, /*topK=*/0, /*topP=*/0.9F);
        SamplingParams const topKParams(batch, vocab, /*temperature=*/1.0F, /*topK=*/vocab, /*topP=*/1.0F);
        int64_t const samplingWorkspaceSize
            = static_cast<int64_t>(std::max(trt_edgellm::getTopKtopPSamplingWorkspaceSize(batch, vocab, topPParams),
                trt_edgellm::getTopKtopPSamplingWorkspaceSize(batch, vocab, topKParams)));
        mSamplingWorkspace = rt::Tensor(
            {samplingWorkspaceSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8, "mSamplingWorkspace");

        mTalkerHiddenStatesBuffer = rt::Tensor({maxBatch, talkerHiddenSeqDim(maxPrefillLen), talkerHiddenSize},
            rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "mTalkerHiddenStatesBuffer");
        mTalkerLastHidden = rt::Tensor(
            {maxBatch, talkerHiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "mTalkerLastHidden");
        mResidualEmbedBuffer = rt::Tensor({maxBatch, 1, talkerHiddenSize}, rt::DeviceType::kGPU,
            trt_edgellm::DataType::kHALF, "mResidualEmbedBuffer");
        mCodecHiddensBuffer = rt::Tensor({maxBatch, mNumCodesPerFrame, talkerHiddenSize}, rt::DeviceType::kGPU,
            trt_edgellm::DataType::kHALF, "mCodecHiddensBuffer");

        mZeroResidualAddend
            = rt::Tensor({talkerHiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "mZeroResidualAddend");
        mNegInfConst = rt::Tensor({1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "mNegInfConst");

        int64_t const numCodeGroups = mTalkerConfig.numCodeGroups;
        mCodecEmbPtrTable = rt::Tensor({numCodeGroups, static_cast<int64_t>(sizeof(__half const*))},
            rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8, "mCodecEmbPtrTable");
        mCodecEmbVocabSizes
            = rt::Tensor({numCodeGroups}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "mCodecEmbVocabSizes");
        mCodecRowCodes
            = rt::Tensor({numCodeGroups}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT64, "mCodecRowCodes");
        // Only the chunked layout sums codec rows, and each lands in a prefill capped at the
        // engine's input length.
        if (usesChunkedText())
        {
            mHostCodecRowCodes = rt::Tensor({mTalkerLLMConfig.maxSupportedInputLength, numCodeGroups},
                rt::DeviceType::kCPU, trt_edgellm::DataType::kINT64, "mHostCodecRowCodes");
        }

        LOG_INFO("Talker buffers allocated (maxBatch=%ld, maxSeqLen=%ld, maxPrefillLen=%ld, talkerH=%ld)", maxBatch,
            maxSeqLen, maxPrefillLen, talkerHiddenSize);
        return true;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Failed to allocate Talker buffers: %s", error.what());
        return false;
    }
}

bool TalkerRunner::loadTalkerWeights(std::string const& weightsDir, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::loadTalkerWeights", nvtx_colors::YELLOW);

    std::filesystem::path const textProjPath = std::filesystem::path(weightsDir) / "text_projection.safetensors";
    std::filesystem::path const hiddenProjPath = std::filesystem::path(weightsDir) / "hidden_projection.safetensors";

    if (usesChunkedText())
    {
        // Text rows come straight out of the talker's own embed_tokens, already at talker
        // hidden width; only the multimodal rows are projected, through a single Linear.
        std::vector<rt::Tensor> hiddenTensors;
        if (!safetensors::loadSafetensors(hiddenProjPath, hiddenTensors, stream))
        {
            LOG_ERROR("Failed to load hidden_projection from: %s", hiddenProjPath.string().c_str());
            return false;
        }
        // hidden_projection is one Linear: the 2-D tensor is the weight, the 1-D one the bias.
        for (auto& tensor : hiddenTensors)
        {
            int32_t const ndims = tensor.getShape().getNumDims();
            if (ndims == 2)
            {
                mHiddenProjLinearWeight = std::move(tensor);
            }
            else if (ndims == 1)
            {
                mHiddenProjLinearBias = std::move(tensor);
            }
        }
        ELLM_CHECK(mHiddenProjLinearWeight.rawPointer() != nullptr && mHiddenProjLinearBias.rawPointer() != nullptr,
            "hidden_projection.safetensors must contain a 2D weight and a 1D bias");

        std::filesystem::path const codecEmbedPath = std::filesystem::path(weightsDir) / "codec_embedding.safetensors";
        std::vector<rt::Tensor> codecEmbedTensors;
        if (!safetensors::loadSafetensors(codecEmbedPath, codecEmbedTensors, stream) || codecEmbedTensors.empty())
        {
            LOG_ERROR("The chunked prefill layout requires codec_embedding.safetensors at %s",
                codecEmbedPath.string().c_str());
            return false;
        }
        mTalkerCodecEmbedTable = std::move(codecEmbedTensors[0]);

        std::filesystem::path const speakerPath
            = std::filesystem::path(weightsDir) / "speaker_codec_embeddings.safetensors";
        std::vector<rt::Tensor> speakerTensors;
        if (!safetensors::loadSafetensors(speakerPath, speakerTensors, stream) || speakerTensors.empty())
        {
            LOG_ERROR("The chunked prefill layout requires speaker_codec_embeddings.safetensors at %s",
                speakerPath.string().c_str());
            return false;
        }
        mSpeakerCodecEmbeddings = std::move(speakerTensors[0]);
        mTextEmbedFromBackbone = false;
    }
    else
    {
        std::vector<rt::Tensor> textTensors;
        if (!safetensors::loadSafetensors(textProjPath, textTensors, stream))
        {
            LOG_ERROR("Failed to load text_projection from: %s", textProjPath.string().c_str());
            return false;
        }
        if (!extractMLPWeights(
                textTensors, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias, "text_projection"))
        {
            return false;
        }

        // A hidden projection is what makes the multimodal rows reachable, and it only
        // ships with checkpoints whose text embeddings live with the backbone.
        mTextEmbedFromBackbone = std::filesystem::exists(hiddenProjPath);
        if (mTextEmbedFromBackbone)
        {
            std::vector<rt::Tensor> hiddenTensors;
            if (!safetensors::loadSafetensors(hiddenProjPath, hiddenTensors, stream))
            {
                LOG_ERROR("Failed to load hidden_projection from: %s", hiddenProjPath.string().c_str());
                return false;
            }
            if (!extractMLPWeights(hiddenTensors, mHiddenFC1Weight, mHiddenFC1Bias, mHiddenFC2Weight, mHiddenFC2Bias,
                    "hidden_projection"))
            {
                return false;
            }
        }
        else
        {
            LOG_INFO("hidden_projection.safetensors not found at %s (multimodal token projection unavailable)",
                hiddenProjPath.string().c_str());
        }
    }

    std::filesystem::path const talkerEmbedPath = std::filesystem::path(weightsDir) / "embedding.safetensors";
    std::vector<rt::Tensor> talkerEmbedTensors;
    if (!safetensors::loadSafetensors(talkerEmbedPath, talkerEmbedTensors, stream))
    {
        LOG_ERROR("Failed to load Talker embedding from: %s", talkerEmbedPath.string().c_str());
        return false;
    }
    ELLM_CHECK(talkerEmbedTensors.size() == 1, "Talker embedding.safetensors should contain exactly one tensor");
    ELLM_CHECK(talkerEmbedTensors[0].getShape().getNumDims() == 2,
        "Talker embedding tensor should be 2D [vocabSize, hiddenSize]");
    mTalkerEmbeddingTable = std::move(talkerEmbedTensors[0]);
    LOG_INFO("Talker embedding table loaded: [%lld, %lld]", mTalkerEmbeddingTable.getShape()[0],
        mTalkerEmbeddingTable.getShape()[1]);
    return true;
}

void TalkerRunner::loadTextEmbeddingTable(std::string const& weightsDir, cudaStream_t stream)
{
    std::filesystem::path const path = mTextEmbedFromBackbone
        ? std::filesystem::path(mBackboneDir) / "embedding.safetensors"
        : std::filesystem::path(weightsDir) / "text_embedding.safetensors";
    LOG_INFO("Loading text embedding from: %s", path.string().c_str());

    std::vector<rt::Tensor> tensors;
    bool const loaded = safetensors::loadSafetensors(path, tensors, stream);
    ELLM_CHECK(loaded, "Failed to load text embedding from: " + path.string());
    ELLM_CHECK(!tensors.empty(), "text embedding file is empty");
    ELLM_CHECK(tensors[0].getShape().getNumDims() == 2, "text embedding tensor should be 2D [vocabSize, hiddenSize]");
    mTextEmbeddingTable = std::move(tensors[0]);
}

void TalkerRunner::initializeTTSEmbeddings(cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::initializeTTSEmbeddings", nvtx_colors::YELLOW);

    auto const shape = mTextEmbeddingTable.getShape();
    ELLM_CHECK(
        shape.getNumDims() == 2, "Text embedding table must be 2D, got " + std::to_string(shape.getNumDims()) + "D");

    int64_t const vocabSize = shape[0];
    int64_t const thinkerHiddenSize = shape[1];

    auto const inVocab = [vocabSize](int32_t id) { return id >= 0 && id < vocabSize; };
    ELLM_CHECK(inVocab(mTokenIds.ttsPad) && inVocab(mTokenIds.ttsBos) && inVocab(mTokenIds.ttsEos),
        "TTS token IDs out of vocab range: pad=" + std::to_string(mTokenIds.ttsPad)
            + ", bos=" + std::to_string(mTokenIds.ttsBos) + ", eos=" + std::to_string(mTokenIds.ttsEos)
            + ", vocabSize=" + std::to_string(vocabSize));

    constexpr int32_t kNumTtsTokens = 3;
    std::vector<int32_t> const hostTtsIds = {mTokenIds.ttsPad, mTokenIds.ttsBos, mTokenIds.ttsEos};

    rt::Tensor ttsIds({1, kNumTtsTokens}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor ttsRaw({1, kNumTtsTokens, thinkerHiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    rt::Tensor ttsProjected(
        {kNumTtsTokens, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    rt::Tensor workspace({kNumTtsTokens, thinkerHiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);

    CUDA_CHECK(cudaMemcpyAsync(
        ttsIds.rawPointer(), hostTtsIds.data(), kNumTtsTokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    kernel::embeddingLookup(ttsIds, mTextEmbeddingTable, std::nullopt, ttsRaw, stream);
    ELLM_CHECK(ttsRaw.reshape({kNumTtsTokens, thinkerHiddenSize}), "Tensor reshape failed");
    if (usesChunkedText())
    {
        // The table is already at talker hidden width here, so there is nothing to project.
        CUDA_CHECK(cudaMemcpyAsync(ttsProjected.rawPointer(), ttsRaw.rawPointer(),
            static_cast<size_t>(kNumTtsTokens) * thinkerHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        kernel::invokeTalkerMLP(
            ttsRaw, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias, ttsProjected, workspace, stream);
    }

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    mTtsPadEmbed = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    mTtsBosEmbed = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    mTtsEosEmbed = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);

    __half* const projectedPtr = static_cast<__half*>(ttsProjected.rawPointer());
    size_t const embedSize = hiddenSize * sizeof(__half);
    CUDA_CHECK(cudaMemcpyAsync(
        mTtsPadEmbed.rawPointer(), projectedPtr + 0 * hiddenSize, embedSize, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mTtsBosEmbed.rawPointer(), projectedPtr + 1 * hiddenSize, embedSize, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mTtsEosEmbed.rawPointer(), projectedPtr + 2 * hiddenSize, embedSize, cudaMemcpyDeviceToDevice, stream));

    CUDA_CHECK(cudaMemsetAsync(mZeroResidualAddend.rawPointer(), 0, embedSize, stream));

    float const negativeInfinity = -std::numeric_limits<float>::infinity();
    CUDA_CHECK(
        cudaMemcpyAsync(mNegInfConst.rawPointer(), &negativeInfinity, sizeof(float), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool TalkerRunner::buildCodecEmbedPointerTable(cudaStream_t stream)
{
    int32_t const numCodeGroups = mTalkerConfig.numCodeGroups;
    if (mCodePredictor.numRvqLayers() < numCodeGroups - 1)
    {
        LOG_ERROR("Residual codec embed tables (%d) < numCodeGroups-1 (%d)", mCodePredictor.numRvqLayers(),
            numCodeGroups - 1);
        return false;
    }
    std::vector<__half const*> hostEmbPtrs(numCodeGroups);
    std::vector<int32_t> hostVocabSizes(numCodeGroups);
    hostEmbPtrs[0] = static_cast<__half const*>(mTalkerCodecEmbedTable.rawPointer());
    hostVocabSizes[0] = static_cast<int32_t>(mTalkerCodecEmbedTable.getShape()[0]);
    for (int32_t g = 1; g < numCodeGroups; ++g)
    {
        rt::Tensor const& table = mCodePredictor.codecEmbedTable(g - 1);
        hostEmbPtrs[g] = static_cast<__half const*>(table.rawPointer());
        hostVocabSizes[g] = static_cast<int32_t>(table.getShape()[0]);
    }
    CUDA_CHECK(cudaMemcpyAsync(mCodecEmbPtrTable.rawPointer(), hostEmbPtrs.data(),
        numCodeGroups * sizeof(__half const*), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mCodecEmbVocabSizes.rawPointer(), hostVocabSizes.data(), numCodeGroups * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    return true;
}

void TalkerRunner::captureDecodeGraphs(cudaStream_t stream)
{
    auto& cacheManager = *mTalkerSharedRes->cacheManagers[0];

    // Capture from a mid-sequence KV state so the graph reflects realistic plugin shapes
    // and KV-length math.
    constexpr int32_t kSimulateCacheLength{128};

    // One graph per cohort width. Without the wider ones a batched utterance replays every
    // layer launch from the host on every frame, which costs more than the batch saves.
    bool captured = true;
    for (int32_t bs = 1; bs <= mMaxBatch; ++bs)
    {
        std::vector<int32_t> reuseLens(static_cast<size_t>(bs), kSimulateCacheLength);
        rt::Tensor simulatedReuse(
            reuseLens.data(), rt::Coords{bs}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32);
        cacheManager.resetForNewSequences(simulatedReuse, stream);

        ELLM_CHECK(mResidualEmbedBuffer.reshape({bs, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        ELLM_CHECK(mTalkerHiddenStatesBuffer.reshape({bs, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        ELLM_CHECK(mTalkerLogits.reshape({bs, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");

        mTalkerTensorMap.set(binding_names::kInputsEmbeds, mResidualEmbedBuffer);
        mTalkerTensorMap.set(binding_names::kLogits, mTalkerLogits);
        mTalkerTensorMap.set(binding_names::kOutputHiddenStates, mTalkerHiddenStatesBuffer);

        // The capture has to see the same ragged bindings a real decode writes, or the graph
        // records whatever the previous step left in them.
        std::vector<int32_t> const captureQuery(static_cast<size_t>(bs), 1);
        std::vector<int32_t> const capturePast(static_cast<size_t>(bs), kSimulateCacheLength);
        prepareRaggedStep(mTalkerRagged, *mTalkerPipelineIO, *mTalkerSharedRes, mTalkerLLMConfig, captureQuery,
            capturePast, bs, /*queryWidth=*/1, rt::ExecutionPhase::kAutoregressiveDecode, stream);

        auto const decodeDims = mTalkerLLMConfig.decodeDims(bs);
        // Every width is attempted even after one fails: a missing graph only costs that
        // width the enqueue fallback, so losing one must not cost the others their capture.
        captured = mTalkerExec->prepare(/*decodeProfile=*/1, decodeDims, mTalkerTensorMap, stream) && captured;
        captured = mTalkerExec->captureGraph(stream) && captured;
    }

    // The simulated length leaves both the per-sequence lengths and the engine-written KV
    // contents mid-sequence; every real prefill expects them empty.
    std::vector<int32_t> zeroLens(static_cast<size_t>(mMaxBatch), 0);
    rt::Tensor zeroReuse(zeroLens.data(), rt::Coords{mMaxBatch}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32);
    cacheManager.resetForNewSequences(zeroReuse, stream);

    // Not short-circuited: the code predictor's graphs are worth having even when the
    // talker's capture failed, and they fall back independently.
    bool const predictorCaptured = mCodePredictor.captureDecodeGraphs(stream);
    captured = captured && predictorCaptured;
    if (!captured)
    {
        LOG_WARNING("Failed to capture some decoding CUDA graphs; falling back to plain engine execution.");
    }
}

// ---------------------------------------------------------------------------
// Speaker
// ---------------------------------------------------------------------------

int32_t TalkerRunner::lookupSpeakerId(std::string const& speakerName) const
{
    auto const it = mSpeakerIdMap.find(speakerName);
    if (it != mSpeakerIdMap.end())
    {
        return it->second;
    }

    auto const key = lowered(speakerName);
    if (auto const alias = mVoiceAliasMap.find(key); alias != mVoiceAliasMap.end())
    {
        if (auto const mapped = mSpeakerIdMap.find(alias->second); mapped != mSpeakerIdMap.end())
        {
            return mapped->second;
        }
    }
    if (auto const direct = mSpeakerIdMap.find(key); direct != mSpeakerIdMap.end())
    {
        return direct->second;
    }
    // A speaker id as digits, which must still be one the table holds.
    int32_t id = -1;
    auto const [end, error] = std::from_chars(speakerName.data(), speakerName.data() + speakerName.size(), id);
    if (error == std::errc{} && end == speakerName.data() + speakerName.size() && id >= 0)
    {
        for (auto const& [name, known] : mSpeakerIdMap)
        {
            if (known == id)
            {
                return id;
            }
        }
    }
    return -1;
}

int32_t TalkerRunner::getSpeakerIdByName(std::string const& speakerName) const
{
    int32_t const speakerId = lookupSpeakerId(speakerName);
    if (speakerId >= 0)
    {
        return speakerId;
    }
    LOG_WARNING(
        "Speaker '%s' not found, using default speaker ID %d", speakerName.c_str(), mTalkerConfig.defaultSpeakerId);
    return mTalkerConfig.defaultSpeakerId;
}

//! Finds the last `<|im_start|>assistant`, steps over the role and newline, then over the empty
//! think block the C++ chat template renders where HF's renders none: its tokens are not speakable text.
int32_t TalkerRunner::assistantContentStart() const
{
    auto const count = static_cast<int32_t>(mTextTokenIds.size());
    int32_t start = -1;
    for (int32_t i = 0; i + 1 < count; ++i)
    {
        if (mTextTokenIds[static_cast<size_t>(i)] != mTokenIds.imStart
            || mTextTokenIds[static_cast<size_t>(i + 1)] != mTokenIds.assistantRole)
        {
            continue;
        }
        int32_t pos = i + 2;
        if (pos < count && mTextTokenIds[static_cast<size_t>(pos)] == mTokenIds.newline)
        {
            ++pos;
        }
        start = pos;
    }
    ELLM_CHECK(start >= 0 && start < count,
        "stage '" + mStageId + "': no assistant turn in the token stream; a talker without a backbone "
        "needs the text rendered through the chat template");

    if (mTextTokenIds[static_cast<size_t>(start)] == mTokenIds.thinkOpen)
    {
        for (int32_t j = start + 1; j < count; ++j)
        {
            if (mTextTokenIds[static_cast<size_t>(j)] != mTokenIds.thinkClose)
            {
                continue;
            }
            start = j + 1;
            if (start < count
                && (mTextTokenIds[static_cast<size_t>(start)] == mTokenIds.newline
                    || mTextTokenIds[static_cast<size_t>(start)] == mTokenIds.doubleNewline))
            {
                ++start;
            }
            break;
        }
    }
    return start;
}

bool TalkerRunner::speakerResolves(std::string const& speaker) const
{
    return lookupSpeakerId(speaker) >= 0;
}

std::string TalkerRunner::requestedSpeaker(multiStage::StageExecContext const& ctx) const
{
    auto const* voice = ctx.in != nullptr ? ctx.in->find(kVoiceInput) : nullptr;
    if (voice != nullptr && voice->data != nullptr && voice->elementBytes == 1 && !voice->shape.empty())
    {
        auto const* text = static_cast<char const*>(voice->data);
        std::string name(text, static_cast<size_t>(voice->shape.back()));
        auto const nul = name.find('\0');
        if (nul != std::string::npos)
        {
            name.resize(nul);
        }
        return name;
    }
    return mDefaultSpeaker;
}

// ---------------------------------------------------------------------------
// Admission and per-request setup
// ---------------------------------------------------------------------------

bool TalkerRunner::admit(multiStage::StageExecContext const& ctx, std::string& reason)
{
    auto& row = rowFor(ctx);
    row.ready = false;

    if (!mDefaultSpeaker.empty() && !speakerResolves(mDefaultSpeaker))
    {
        reason = "stage '" + mStageId + "': configured speaker '" + mDefaultSpeaker
            + "' is in neither this checkpoint's speaker table nor its alias map";
        return false;
    }
    if (auto const speaker = requestedSpeaker(ctx); !speaker.empty() && !speakerResolves(speaker))
    {
        reason = "stage '" + mStageId + "': requested speaker '" + speaker + "' does not resolve in this checkpoint";
        return false;
    }
    // A cohort decodes under one sampling setup, so a request asking for another waits for a
    // cohort of its own.
    takeSampling(row, multiStage::jsonInput(ctx.in, kSamplingInput));
    if (ctx.slot->batchSlot > 0 && !sharesSampling(mRows.front(), row))
    {
        reason = "stage '" + mStageId + "': talker or code-predictor sampling differs from the cohort's";
        return false;
    }

    // The chunked layout feeds text by repeated KV-reset re-prefills at a per-member frame
    // boundary, which a shared cache cannot serve for several members at once.
    if (usesChunkedText() && ctx.slot->batchSlot > 0)
    {
        reason
            = "stage '" + mStageId + "': chunked text feeding drives one request at a time; this topology cannot batch";
        return false;
    }

    row.ready = true;
    return true;
}

void TalkerRunner::prepare(multiStage::StageExecContext& ctx)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::prepare", nvtx_colors::PURPLE);
    auto& row = rowFor(ctx);
    ELLM_CHECK(row.ready, "talker prepare() ran without a successful admit()");
    ELLM_CHECK(ctx.in != nullptr, "talker prepare() without input bindings");

    mPrepareSlot = ctx.slot->batchSlot;
    row.prepared = false;
    row.finished = false;
    row.hitEos = false;
    row.frames = 0;
    row.pastLen = 0;
    row.numSeenTokens = 0;
    row.codecToken = -1;
    std::fill(row.seenTokens.begin(), row.seenTokens.end(), 0);
    resetChunkStream();
    takeSampling(row, multiStage::jsonInput(ctx.in, kSamplingInput));
    applyStyle(multiStage::jsonInput(ctx.in, kStyleInput), ctx.in->find(kSpeakerCodesInput));
    // Only some layouts have rows for these; dropping them silently would pass for a voice or
    // accent regression.
    if (!mPromptSpeakerCodes.empty() && mPrefillBuilder != PrefillBuilder::kChunked)
    {
        LOG_WARNING(
            "stage '%s': speaker_codes ignored; only the chunked prefill layout clones from codes", mStageId.c_str());
    }
    if (mPrefillBuilder == PrefillBuilder::kSegments)
    {
        for (auto const* language : {&mRequestLanguage, &mRequestTalkerLanguage})
        {
            if (!language->empty() && lowered(*language) != "auto")
            {
                LOG_WARNING("stage '%s': language '%s' ignored; the segments prefill layout is language-free",
                    mStageId.c_str(), language->c_str());
            }
        }
    }

    // The reference encoders want 24 kHz, not the 16 kHz the backbone's audio port carries.
    mRefAudioView = rt::Tensor{};
    if (auto const* refAudio = ctx.in->find(kRefAudioInput); refAudio != nullptr && refAudio->data != nullptr)
    {
        ELLM_CHECK(refAudio->kind == multiStage::PortKind::kAudio
                && refAudio->elementBytes == static_cast<int32_t>(sizeof(float)) && !refAudio->shape.empty(),
            "stage '" + mStageId + "': input '" + kRefAudioInput + "' must be mono FP32 PCM at 24 kHz");
        mRefAudioView = rt::Tensor(refAudio->data, rt::Coords{refAudio->shape.back()}, rt::DeviceType::kCPU,
            trt_edgellm::DataType::kFLOAT, kRefAudioInput);
    }

    // Absent when no backbone feeds this stage: a text-to-speech deployment hands the talker
    // tokens only, and every layout then looks each row up in the text embedding table.
    auto const* const hiddenBinding = ctx.in->find(kHiddenInput);
    mHiddenView = hiddenBinding != nullptr ? viewOf(*hiddenBinding, kHiddenInput) : rt::Tensor{};
    auto const* const embedsBinding = ctx.in->find(kEmbedsInput);
    mEmbedsView = embedsBinding != nullptr ? viewOf(*embedsBinding, kEmbedsInput) : rt::Tensor{};

    mTextTokenIds.clear();
    auto const render = [this](std::string const& text) { return renderConversation(text); };
    appendIds(ctx.in->find(kPromptInput), kPromptInput, mTextTokenIds, render);
    auto const promptTokenCount = mTextTokenIds.size();
    appendIds(ctx.in->find(kTextInput), kTextInput, mTextTokenIds);
    ELLM_CHECK(!mTextTokenIds.empty(), "talker prepare() got an empty token sequence");

    // Where the assistant's own content begins, the split the segments and chunked builders
    // assemble around: the tapped hidden window's width with a backbone, else recovered by
    // assistantContentStart(). The tts layout strips a fixed wrapper instead.
    int32_t const prefillLength = hiddenBinding != nullptr
        ? static_cast<int32_t>(mHiddenView.getShape()[1])
        : (mPrefillBuilder == PrefillBuilder::kTts ? 0 : assistantContentStart());

    // A prompt that is not exactly as long as the hidden window shifts every assistant row
    // by the difference, so the two must agree before anything is assembled.
    ELLM_CHECK(hiddenBinding == nullptr || promptTokenCount == static_cast<size_t>(prefillLength),
        "stage '" + mStageId + "': prompt is " + std::to_string(promptTokenCount) + " tokens but the hidden window is "
            + std::to_string(prefillLength) + " rows; the assistant split would land mid-prompt");
    ELLM_CHECK(mTextTokenIds.size() <= static_cast<size_t>(mMaxSeq),
        "talker text sequence of " + std::to_string(mTextTokenIds.size()) + " tokens exceeds max_seq "
            + std::to_string(mMaxSeq));

    // admit() already refused an unknown voice from the request; one an upstream stage
    // published arrives only now.
    auto const speaker = requestedSpeaker(ctx);
    ELLM_CHECK(speaker.empty() || speakerResolves(speaker),
        "stage '" + mStageId + "': requested speaker '" + speaker + "' does not resolve in this checkpoint");
    int32_t const speakerId = speaker.empty() ? mTalkerConfig.defaultSpeakerId : getSpeakerIdByName(speaker);
    mRequestSpeakerName = speaker;

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;

    int32_t trailingCount = 0;
    int64_t outSeqLen = 0;
    auto const* const hiddenForBuilder = hiddenBinding != nullptr ? &mHiddenView : nullptr;
    bool built = false;
    switch (mPrefillBuilder)
    {
    case PrefillBuilder::kChunked:
        built = buildChunkedPrefill(hiddenForBuilder, prefillLength, speakerId, outSeqLen, ctx.stream);
        break;
    case PrefillBuilder::kTts: built = buildTtsPrefill(speakerId, outSeqLen, ctx.stream); break;
    case PrefillBuilder::kSegments:
        built = buildSegmentsPrefill(embedsBinding != nullptr ? &mEmbedsView : nullptr, hiddenForBuilder, prefillLength,
            speakerId, mTrailingTextHidden, trailingCount, outSeqLen, ctx.stream);
        break;
    }
    ELLM_CHECK(built, "stage '" + mStageId + "': talker prefill assembly failed");

    int64_t const hiddenRow = hiddenSize * static_cast<int64_t>(sizeof(__half));
    auto const slot = ctx.slot->batchSlot;

    // Only the segments layout adds text per frame. The tts layout consumes its eos row inside
    // the prefill, so an empty view keeps its addend on tts_pad; the chunked one adds zero.
    row.trailingView = rt::Tensor{};
    if (mPrefillBuilder == PrefillBuilder::kSegments)
    {
        finalizeTrailing(mTrailingTextHidden, trailingCount, ctx.stream);
        ++trailingCount;
        // Out of the shared assembly scratch and into this member's own rows, before the next
        // member's prepare() assembles over it.
        int64_t const trailingRows = mTrailingTextHidden.getShape()[0];
        auto* const trailingDst = static_cast<char*>(mTrailingRows.rawPointer())
            + static_cast<size_t>(slot) * static_cast<size_t>(trailingRows) * static_cast<size_t>(hiddenRow);
        CUDA_CHECK(cudaMemcpyAsync(trailingDst, mTrailingTextHidden.rawPointer(),
            static_cast<size_t>(trailingCount) * static_cast<size_t>(hiddenRow), cudaMemcpyDeviceToDevice, ctx.stream));
        row.trailingView = rt::Tensor(trailingDst, rt::Coords{static_cast<int64_t>(trailingCount), hiddenSize},
            rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    }

    int64_t const maxPrefillLen = mTalkerLLMConfig.maxSupportedInputLength;
    ELLM_CHECK(outSeqLen <= maxPrefillLen,
        "stage '" + mStageId + "': assembled prefill of " + std::to_string(outSeqLen)
            + " rows exceeds the talker engine's max input length " + std::to_string(maxPrefillLen));
    // Refused here, for this member alone: the first frame's decode would fail the whole cohort.
    ELLM_CHECK(outSeqLen < mTalkerLLMConfig.maxKVCacheCapacity,
        "stage '" + mStageId + "': assembled prefill of " + std::to_string(outSeqLen)
            + " rows leaves the talker's KV cache of " + std::to_string(mTalkerLLMConfig.maxKVCacheCapacity)
            + " no room to decode a frame");
    auto* const embedsDst = static_cast<char*>(mTalkerPrefillRows.rawPointer())
        + static_cast<size_t>(slot) * static_cast<size_t>(maxPrefillLen) * static_cast<size_t>(hiddenRow);
    CUDA_CHECK(cudaMemcpyAsync(embedsDst, mTalkerInputEmbeds.rawPointer(),
        static_cast<size_t>(outSeqLen) * static_cast<size_t>(hiddenRow), cudaMemcpyDeviceToDevice, ctx.stream));

    row.prefillSeqLen = outSeqLen;
    setFrameBudget(row, outSeqLen, /*framesSoFar=*/0);
    LOG_INFO("Talker prefill staged: %zu text tokens (prefillLen=%d), %zu queued for re-prefill, maxFrames=%d",
        mTextTokenIds.size(), prefillLength, mChunkStream.remainingTextTokens.size(), row.maxFrames);
    row.prepared = true;
}

void TalkerRunner::takeSampling(Row& row, nlohmann::json const& overrides)
{
    applySampling(overrides);
    row.temperature = mTemperature;
    // A top-k past the vocabulary keeps every token; the sampling workspaces are sized for no more.
    row.topK = std::min(mTopK, mTalkerConfig.talkerVocabSize);
    row.topP = mTopP;
    row.repetitionPenalty = mRepetitionPenalty;
    row.cpTemperature = mCpTemperature;
    row.cpTopK = std::min(mCpTopK, mCodePredictor.codebookSize());
    row.cpTopP = mCpTopP;
}

bool TalkerRunner::sharesSampling(Row const& founder, Row const& row)
{
    return row.temperature == founder.temperature && row.topK == founder.topK && row.topP == founder.topP
        && row.cpTemperature == founder.cpTemperature && row.cpTopK == founder.cpTopK && row.cpTopP == founder.cpTopP;
}

void TalkerRunner::checkCohortSampling(
    std::vector<multiStage::StageExecContext*> const& members, std::vector<size_t> const& live) const
{
    auto const& founder = rowFor(*members[live.front()]->slot);
    for (auto const index : live)
    {
        auto const& row = rowFor(*members[index]->slot);
        ELLM_CHECK(sharesSampling(founder, row),
            "stage '" + mStageId + "': requests in one cohort must share talker and code-predictor sampling; request "
                + std::to_string(members[index]->slot->requestId) + " differs from "
                + std::to_string(members[live.front()]->slot->requestId));
    }
}

// ---------------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------------

multiStage::StageStatus TalkerRunner::step(multiStage::StageExecContext& ctx)
{
    std::vector<multiStage::StageExecContext*> members{&ctx};
    std::vector<multiStage::StageStatus> statuses(1, multiStage::StageStatus::kFailed);
    stepCohort(members, statuses);
    return statuses.front();
}

void TalkerRunner::stepCohort(
    std::vector<multiStage::StageExecContext*> const& members, std::vector<multiStage::StageStatus>& statuses)
{
    auto& live = mLive;
    live.clear();
    for (size_t index = 0; index < members.size(); ++index)
    {
        auto& member = *members[index];
        auto& row = rowFor(member);
        if (member.slot->abandon.requested())
        {
            row.finished = true;
        }
        if (row.finished)
        {
            statuses[index] = multiStage::StageStatus::kDone;
            continue;
        }
        ELLM_CHECK(row.prepared, "talker step() ran without a successful prepare()");
        live.push_back(index);
    }
    // After the prefill every unfinished member is handed over each round, so one missing here
    // was cancelled or failed: its row stops producing rather than decoding on to EOS.
    if (mCohortPrefilled)
    {
        for (auto const slot : mCohortSlots)
        {
            bool handed = false;
            for (auto const* member : members)
            {
                handed = handed || member->slot->batchSlot == slot;
            }
            if (!handed)
            {
                mRows[static_cast<size_t>(slot)].finished = true;
            }
        }
    }
    if (live.empty())
    {
        return;
    }

    auto const stream = members[live.front()]->stream;
    for (auto const index : live)
    {
        ELLM_CHECK(members[index]->stream == stream, "talker cohort spans more than one stream");
    }
    checkCohortSampling(members, live);

    // The prefill is one batched call over every member staged by prepare(), so it cannot run
    // until the executor has prepared them all -- which is what the first step is. A member
    // cancelled before it has no frame, so it stays out of the cohort.
    if (!mCohortPrefilled)
    {
        mPrefillMembers.clear();
        for (auto const index : live)
        {
            mPrefillMembers.push_back(members[index]);
        }
        prefillCohort(mPrefillMembers);
        mCohortPrefilled = true;
    }

    if (!decodeCohortFrame(stream))
    {
        for (auto const index : live)
        {
            LOG_ERROR("Talker request %llu failed at codec frame %d",
                static_cast<unsigned long long>(members[index]->slot->requestId), rowFor(*members[index]).frames);
            statuses[index] = multiStage::StageStatus::kFailed;
        }
        return;
    }

    bool anyFinished = false;
    for (auto const index : live)
    {
        auto& member = *members[index];
        auto const& row = rowFor(member);
        bindCodes(member, row.finished);
        statuses[index] = row.finished ? multiStage::StageStatus::kDone : multiStage::StageStatus::kMore;
        anyFinished = anyFinished || row.finished;
    }
    if (anyFinished)
    {
        mCodePredictor.logAndResetSpecStats();
    }
}

void TalkerRunner::prefillCohort(std::vector<multiStage::StageExecContext*> const& members)
{
    auto const stream = members.front()->stream;
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;

    mCohortSlots.clear();
    mCohortStep = 0;
    int64_t paddedSeqLen = 0;
    for (auto const* member : members)
    {
        auto const& row = rowFor(*member);
        mCohortSlots.push_back(member->slot->batchSlot);
        paddedSeqLen = std::max(paddedSeqLen, row.prefillSeqLen);
    }
    auto const batchSize = static_cast<int32_t>(mCohortSlots.size());

    // The engine reads one padded rectangle; queryLengths is what tells it how much of each
    // row is real, and logitsIndices then picks each row's own last position.
    mQueryLengths.resize(static_cast<size_t>(batchSize));
    for (int32_t b = 0; b < batchSize; ++b)
    {
        mQueryLengths[static_cast<size_t>(b)]
            = static_cast<int32_t>(rowFor(*members[static_cast<size_t>(b)]).prefillSeqLen);
    }

    // The rows are laid out at a stride of the engine's max input length; the engine wants them at paddedSeqLen, so
    // the cohort is compacted in place front to back: slots ascend with b, so no row is overwritten
    // before it has been moved. A row whose source and destination overlap goes through the
    // assembly scratch, which every prepare() is done with by now.
    auto const stride = static_cast<size_t>(mTalkerLLMConfig.maxSupportedInputLength) * static_cast<size_t>(hiddenSize);
    auto const rowElements = static_cast<size_t>(paddedSeqLen) * static_cast<size_t>(hiddenSize);
    auto const rowBytes = rowElements * sizeof(__half);
    auto* const base = static_cast<__half*>(mTalkerPrefillRows.rawPointer());
    for (int32_t b = 0; b < batchSize; ++b)
    {
        auto const slot = static_cast<size_t>(mCohortSlots[static_cast<size_t>(b)]);
        auto* const dst = base + static_cast<size_t>(b) * rowElements;
        auto* const src = base + slot * stride;
        if (dst == src)
        {
            continue;
        }
        if (src < dst + rowElements)
        {
            CUDA_CHECK(
                cudaMemcpyAsync(mTalkerInputEmbeds.rawPointer(), src, rowBytes, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(
                cudaMemcpyAsync(dst, mTalkerInputEmbeds.rawPointer(), rowBytes, cudaMemcpyDeviceToDevice, stream));
        }
        else
        {
            CUDA_CHECK(cudaMemcpyAsync(dst, src, rowBytes, cudaMemcpyDeviceToDevice, stream));
        }
    }

    rt::Tensor inputEmbeds(mTalkerPrefillRows.rawPointer(), rt::Coords{batchSize, paddedSeqLen, hiddenSize},
        rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    ELLM_CHECK(mTalkerHiddenStatesBuffer.reshape({batchSize, talkerHiddenSeqDim(paddedSeqLen), hiddenSize}),
        "Tensor reshape failed");

    {
        TIME_STAGE(metrics::StageNames::kTALKER_PREFILL, stream);
        ELLM_CHECK(
            executeTalkerPrefillStep(inputEmbeds, mQueryLengths, mTalkerLogits, mTalkerHiddenStatesBuffer, stream),
            "stage '" + mStageId + "': talker prefill failed");
    }

    ELLM_CHECK(mTalkerLogits.reshape({batchSize, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");
    ELLM_CHECK(mTalkerSelectedIndices.reshape({batchSize, 1}), "Tensor reshape failed");
    for (int32_t b = 0; b < batchSize; ++b)
    {
        auto const& row = rowFor(*members[static_cast<size_t>(b)]);
        auto logitRow = sliceOf(mTalkerLogits, b, rt::Coords{1, mTalkerConfig.talkerVocabSize});
        kernel::invokeTalkerLogitAdjust(sliceOf(mSeenCodecTokensBuf, mCohortSlots[static_cast<size_t>(b)],
                                            rt::Coords{1, mTalkerLLMConfig.maxKVCacheCapacity}),
            logitRow, mTalkerConfig.talkerSuppressStart, mTalkerConfig.talkerVocabSize, mTalkerConfig.codecEosId, 0,
            row.repetitionPenalty, stream);

        // Text is still queued, so this call is not the last one and owes at least
        // framesPerCall frames. The logit-adjust kernel spares codec_eos by design, so without
        // this the very first sample of the utterance could end it before a chunk was spoken.
        if (mChunkStream.active)
        {
            suppressTalkerEosLogit(mTalkerLogits, b, stream);
        }
    }

    // Random offset 0 here; each decode step then advances it by one (mCohortStep).
    auto const& founder = rowFor(*members.front());
    float const temperature = (founder.temperature > 0) ? founder.temperature : kTalkerTemperature;
    int32_t const topK = (founder.topK > 0) ? founder.topK : kTalkerTopK;
    float const topP = (founder.topP > 0) ? founder.topP : kTalkerTopP;
    SamplingParams const talkerSampling(batchSize, mTalkerConfig.talkerVocabSize, temperature, topK, topP);
    trt_edgellm::topKtopPSamplingFromLogits(mTalkerLogits, mTalkerSelectedIndices, talkerSampling, mSamplingWorkspace,
        stream, kTalkerSamplingSeed, /*randomOffset=*/0);
    CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(),
        static_cast<size_t>(batchSize) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t b = 0; b < batchSize; ++b)
    {
        auto& row = rowFor(*members[static_cast<size_t>(b)]);
        row.codecToken = mHostSelectedTokenIds.dataPointer<int32_t>()[b];
        row.pastLen = static_cast<int32_t>(row.prefillSeqLen);
        trackSeenToken(row, mCohortSlots[static_cast<size_t>(b)], row.codecToken,
            mTalkerSelectedIndices.dataPointer<int32_t>() + b, stream);
    }

    if (getProfilingEnabled())
    {
        auto const generationTiming = gTimer.getTimingData(metrics::StageNames::kTALKER_GENERATION);
        float const mark = generationTiming ? generationTiming->getTotalGpuTimeMs() : 0.0F;
        for (auto const* member : members)
        {
            rowFor(*member).generationMsAtStart = mark;
        }
    }
}

bool TalkerRunner::decodeCohortFrame(cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::decodeCohortFrame", nvtx_colors::PURPLE);
    TIME_STAGE(metrics::StageNames::kTALKER_GENERATION, stream);

    int32_t const codecEosId = mTalkerConfig.codecEosId;
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;

    // Only the tail may leave. An engine row is also its KV state index and the cache manager
    // commits lengths over the first activeBatchSize resident slots, so dropping a row from
    // the middle would move every row after it onto a neighbour's cache. Dropping a finished
    // suffix needs no remapping: rows 0..width-1 keep their state indices.
    auto width = static_cast<int32_t>(mCohortSlots.size());
    while (width > 1 && mRows[static_cast<size_t>(mCohortSlots[static_cast<size_t>(width - 1)])].finished)
    {
        --width;
    }
    auto const batchSize = width;

    // Rows are one token wide from the second frame on, but the first one reads the prefill
    // window, so the extent has to survive the narrowing.
    auto const hiddenShape = mTalkerHiddenStatesBuffer.getShape();
    ELLM_CHECK(mTalkerHiddenStatesBuffer.reshape({batchSize, hiddenShape[1], hiddenSize}), "Tensor reshape failed");

    auto const& founder = mRows[static_cast<size_t>(mCohortSlots.front())];
    float const temperature = (founder.temperature > 0) ? founder.temperature : kTalkerTemperature;
    int32_t const topK = (founder.topK > 0) ? founder.topK : kTalkerTopK;
    float const topP = (founder.topP > 0) ? founder.topP : kTalkerTopP;
    SamplingParams const talkerSampling(batchSize, mTalkerConfig.talkerVocabSize, temperature, topK, topP);

    float const cpTemperature = (founder.cpTemperature > 0)
        ? founder.cpTemperature
        : (usesChunkedText() ? kCpTemperatureChunked : kCpTemperatureSegments);
    int32_t const cpTopK = (founder.cpTopK > 0) ? founder.cpTopK : kCpTopK;
    float const cpTopP = (founder.cpTopP > 0) ? founder.cpTopP : (usesChunkedText() ? kCpTopPChunked : kCpTopPSegments);
    SamplingParams const cpSampling(batchSize, mCodePredictor.codebookSize(), cpTemperature, cpTopK, cpTopP);

    // The previous frame ended in a stream sync, so its upload of these slots has completed.
    auto* const codeZero = mCodeZero.dataPointer<int32_t>();
    for (int32_t b = 0; b < batchSize; ++b)
    {
        codeZero[b] = mRows[static_cast<size_t>(mCohortSlots[static_cast<size_t>(b)])].codecToken;
    }

    rt::Tensor lastHidden(mTalkerLastHidden.rawPointer(), rt::Coords{batchSize, hiddenSize}, rt::DeviceType::kGPU,
        trt_edgellm::DataType::kHALF);
    if (!extractTalkerLastHidden(mTalkerHiddenStatesBuffer, mQueryLengths, lastHidden, stream))
    {
        LOG_ERROR("extractTalkerLastHidden failed");
        return false;
    }

    ELLM_CHECK(mCodecHiddensBuffer.reshape({batchSize, mNumCodesPerFrame, hiddenSize}), "Tensor reshape failed");

    mCodePredictor.resetKvCache(batchSize, stream);

    CodePredictor::FramePlan plan;
    plan.activeBatchSize = batchSize;
    plan.frameIdx = mCohortStep;
    plan.codeZero = &mCodeZero;
    plan.talkerLastHidden = &lastHidden;
    plan.sampling = &cpSampling;
    if (!mCodePredictor.runFrame(plan, mFrameCodes, stream))
    {
        LOG_ERROR("Code predictor failed");
        return false;
    }

    ELLM_CHECK(mResidualEmbedBuffer.reshape({batchSize, 1, hiddenSize}), "Tensor reshape failed");
    for (int32_t b = 0; b < batchSize; ++b)
    {
        auto& row = mRows[static_cast<size_t>(mCohortSlots[static_cast<size_t>(b)])];
        auto const& codes = mFrameCodes[static_cast<size_t>(b)];
        ELLM_CHECK(static_cast<int32_t>(codes.size()) == mNumCodesPerFrame,
            "code predictor returned " + std::to_string(codes.size()) + " codes, expected "
                + std::to_string(mNumCodesPerFrame));
        if (!row.finished)
        {
            std::copy(codes.begin(), codes.end(),
                row.flatCodes.begin() + static_cast<size_t>(row.frames) * static_cast<size_t>(mNumCodesPerFrame));
            ++row.frames;
        }

        auto residualRow = sliceOf(mResidualEmbedBuffer, b, rt::Coords{1, 1, hiddenSize});
        auto codecHiddensRow = sliceOf(mCodecHiddensBuffer, b, rt::Coords{1, mNumCodesPerFrame, hiddenSize});
        if (!computeResidualConnection(row, codes, row.frames - 1, residualRow, codecHiddensRow, stream))
        {
            LOG_ERROR("Residual connection failed at frame %d", row.frames - 1);
            return false;
        }
    }

    ELLM_CHECK(mTalkerHiddenStatesBuffer.reshape({batchSize, 1, hiddenSize}), "Tensor reshape failed");
    mPastLengths.resize(static_cast<size_t>(batchSize));
    for (int32_t b = 0; b < batchSize; ++b)
    {
        auto const& row = mRows[static_cast<size_t>(mCohortSlots[static_cast<size_t>(b)])];
        // A finished row stays until the rows after it finish, decoding a token nothing reads. It rewrites its
        // last position rather than appending, so it cannot outgrow its KV cache and fail the cohort.
        mPastLengths[static_cast<size_t>(b)] = row.finished ? row.pastLen - 1 : row.pastLen;
    }
    if (!executeTalkerDecodingStep(
            mResidualEmbedBuffer, mPastLengths, mTalkerLogits, mTalkerHiddenStatesBuffer, stream))
    {
        LOG_ERROR("Talker decoding step failed");
        return false;
    }

    ELLM_CHECK(mTalkerLogits.reshape({batchSize, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");
    ELLM_CHECK(mTalkerSelectedIndices.reshape({batchSize, 1}), "Tensor reshape failed");
    for (int32_t b = 0; b < batchSize; ++b)
    {
        auto const slot = mCohortSlots[static_cast<size_t>(b)];
        auto const& row = mRows[static_cast<size_t>(slot)];
        auto logitRow = sliceOf(mTalkerLogits, b, rt::Coords{1, mTalkerConfig.talkerVocabSize});
        kernel::invokeTalkerLogitAdjust(
            sliceOf(mSeenCodecTokensBuf, slot, rt::Coords{1, mTalkerLLMConfig.maxKVCacheCapacity}), logitRow,
            mTalkerConfig.talkerSuppressStart, mTalkerConfig.talkerVocabSize, codecEosId, row.numSeenTokens,
            row.repetitionPenalty, stream);

        // Hold EOS off through the chunk-boundary frame so the re-prefill always runs before
        // the finished check below. The last chunk is exempt.
        if (mChunkStream.active && (row.frames - mChunkStream.firstFrameOfCallIdx) <= mChunkStream.framesPerCall)
        {
            suppressTalkerEosLogit(mTalkerLogits, b, stream);
        }
    }

    // Rows are told apart by their row index, so one offset covers the cohort -- but it has to be the
    // step, not a member's frame count, which stops advancing once it finishes.
    ++mCohortStep;
    trt_edgellm::topKtopPSamplingFromLogits(mTalkerLogits, mTalkerSelectedIndices, talkerSampling, mSamplingWorkspace,
        stream, kTalkerSamplingSeed, static_cast<uint64_t>(mCohortStep));
    CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(),
        static_cast<size_t>(batchSize) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t b = 0; b < batchSize; ++b)
    {
        auto const slot = mCohortSlots[static_cast<size_t>(b)];
        auto& row = mRows[static_cast<size_t>(slot)];
        if (row.finished)
        {
            continue;
        }
        ++row.pastLen;
        row.codecToken = mHostSelectedTokenIds.dataPointer<int32_t>()[b];
        trackSeenToken(
            row, slot, row.codecToken, static_cast<int32_t const*>(mTalkerSelectedIndices.rawPointer()) + b, stream);

        if (row.codecToken == codecEosId || row.frames >= row.maxFrames)
        {
            row.hitEos = (row.codecToken == codecEosId);
            row.finished = true;
        }
    }

    // Chunked feeding is admitted for one member only, so the re-prefill below cannot disturb
    // a neighbour's cache.
    auto& first = mRows[static_cast<size_t>(mCohortSlots.front())];
    if (mChunkStream.active && !first.finished)
    {
        Reprefill const reprefilled = maybeReprefillChunk(stream);
        if (reprefilled == Reprefill::kFailed)
        {
            LOG_ERROR("Chunked re-prefill failed after frame %d", first.frames - 1);
            return false;
        }
        if (reprefilled == Reprefill::kInputFull)
        {
            // As in llm_inference, the utterance keeps the frames so far and ends as a length stop.
            LOG_WARNING(
                "The talker engine takes %ld input rows, too few to feed the rest of the reply; the "
                "utterance ends at frame %d",
                static_cast<int64_t>(mTalkerLLMConfig.maxSupportedInputLength), first.frames);
            first.maxFrames = first.frames;
            first.finished = true;
        }
        if (reprefilled == Reprefill::kDone)
        {
            // The freshly sampled first frame of the new call replaces the unconstrained one.
            first.hitEos = (first.codecToken == codecEosId);
            first.finished = first.hitEos;
            first.prefillSeqLen = mChunkStream.cumulativeSeqLen;
        }
    }
    return true;
}

void TalkerRunner::setFrameBudget(Row& row, int64_t assembledRows, int32_t framesSoFar)
{
    // The KV cache holds the rows this prefill assembled plus one per frame decoded since,
    // so the budget has to follow the re-prefills. Counting input tokens instead undercounts
    // the speaker and codec rows and lets a long utterance run past the cache.
    int64_t const room = static_cast<int64_t>(mTalkerLLMConfig.maxKVCacheCapacity) - assembledRows;
    int64_t const capacity = static_cast<int64_t>(row.flatCodes.size()) / mNumCodesPerFrame;
    row.maxFrames = static_cast<int32_t>(
        std::clamp<int64_t>(framesSoFar + room, 1, std::min<int64_t>(mMaxAudioLength, capacity)));
}

void TalkerRunner::bindCodes(multiStage::StageExecContext& ctx, bool complete)
{
    auto& row = rowFor(ctx);
    multiStage::PortValue codes{row.flatCodes.data(), {row.frames, mNumCodesPerFrame}, multiStage::PortKind::kCodec,
        static_cast<int32_t>(sizeof(int32_t))};
    // The vocoder cannot tell a pause in generation from the end of the utterance, and
    // guessing either way costs it: latching done truncates, waiting forever hangs.
    codes.complete = complete;
    ctx.out->bind(kCodesPort, codes);
}

void TalkerRunner::finish(multiStage::StageExecContext& ctx, multiStage::FinishReason reason)
{
    // Bind the codec port on every path so the vocoder downstream sees an end, whether this
    // request completed, failed or was cancelled.
    if (ctx.out != nullptr)
    {
        bindCodes(ctx, /*complete=*/true);
    }

    auto& row = rowFor(ctx);
    if (getProfilingEnabled())
    {
        auto const generationTiming = gTimer.getTimingData(metrics::StageNames::kTALKER_GENERATION);
        float const generationTotalMs = generationTiming ? generationTiming->getTotalGpuTimeMs() : 0.0F;
        // A total below the mark means the timer was reset since, so all of it is this member's.
        float const generationMs = generationTotalMs >= row.generationMsAtStart
            ? generationTotalMs - row.generationMsAtStart
            : generationTotalMs;
        row.audioDurationSeconds
            = static_cast<float>(row.frames * kAudioSamplesPerFrame) / static_cast<float>(kAudioSampleRate);
        row.realTimeFactor = (generationMs > 0.0F) ? (row.audioDurationSeconds / (generationMs / 1000.0F)) : 0.0F;
    }

    LOG_INFO("Talker ended: %d frames, %s, %d text tokens still queued", row.frames,
        row.hitEos ? "eos" : (row.prepared && row.frames >= row.maxFrames ? "max_frames" : "stopped"),
        static_cast<int32_t>(mChunkStream.remainingTextTokens.size()) - mChunkStream.cursorToken);
    if (reason != multiStage::FinishReason::kCompleted)
    {
        LOG_INFO("Talker request %llu ended: %s after %d frames", static_cast<unsigned long long>(ctx.slot->requestId),
            multiStage::toString(reason), row.frames);
    }
}

nlohmann::json TalkerRunner::diagnostics(rt::multiStage::RequestSlot const& slot) const
{
    auto const& row = rowFor(slot);
    // Same spellings as llm_inference's exit_reason; "stopped" means the request was cancelled.
    char const* exitReason = "stopped";
    if (row.hitEos)
    {
        exitReason = "eos";
    }
    // A talker that never finished prepare() decoded nothing and reached no limit.
    else if (row.prepared && row.frames >= row.maxFrames)
    {
        exitReason = "max_length";
    }

    nlohmann::json report{{"exit_reason", exitReason}, {"audio_frames", row.frames},
        {"total_rvq_codes", static_cast<int64_t>(row.frames) * mNumCodesPerFrame},
        {"prefill_seq_len", static_cast<int32_t>(row.prefillSeqLen)}};
    if (getProfilingEnabled())
    {
        report["real_time_factor"] = row.realTimeFactor;
        report["audio_duration_seconds"] = row.audioDurationSeconds;
    }
    if (mReportCodes)
    {
        nlohmann::json codes = nlohmann::json::array();
        for (int32_t frame = 0; frame < row.frames; ++frame)
        {
            auto const first = row.flatCodes.begin() + static_cast<std::ptrdiff_t>(frame) * mNumCodesPerFrame;
            codes.push_back(std::vector<int32_t>(first, first + mNumCodesPerFrame));
        }
        report["rvq_codes"] = std::move(codes);
    }
    return report;
}

void TalkerRunner::reset()
{
    // The two vectors claimed in preallocate() are carried over, so no request reallocates them.
    for (auto& row : mRows)
    {
        auto codes = std::move(row.flatCodes);
        std::fill(codes.begin(), codes.end(), 0);
        auto seen = std::move(row.seenTokens);
        std::fill(seen.begin(), seen.end(), 0);
        row = Row{};
        row.flatCodes = std::move(codes);
        row.seenTokens = std::move(seen);
    }
    mCohortSlots.clear();
    mCohortPrefilled = false;
    mCohortStep = 0;
    mTextTokenIds.clear();
    resetChunkStream();
    mEmbedsView = rt::Tensor{};
    mHiddenView = rt::Tensor{};
}

// ---------------------------------------------------------------------------
// Engine steps
// ---------------------------------------------------------------------------

bool TalkerRunner::executeTalkerPrefillStep(rt::Tensor const& inputEmbeds, std::vector<int32_t> const& seqLens,
    rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::prefillStep", nvtx_colors::PURPLE);

    auto const inputShape = inputEmbeds.getShape();
    if (inputShape.getNumDims() != 3)
    {
        LOG_ERROR("talker prefill input must be 3D [batch, seqLen, hidden], got %dD", inputShape.getNumDims());
        return false;
    }
    int64_t const batchSize = inputEmbeds.getShape()[0];
    int64_t const seqLen = inputEmbeds.getShape()[1];
    if (batchSize > mMaxBatch)
    {
        LOG_ERROR("talker prefill batch %ld exceeds max batch %d", batchSize, mMaxBatch);
        return false;
    }

    auto& cacheManager = *mTalkerSharedRes->cacheManagers[0];
    ELLM_CHECK(mHostReuseKVCacheLengths.reshape({batchSize}), "Tensor reshape failed");
    std::fill_n(mHostReuseKVCacheLengths.dataPointer<int32_t>(), batchSize, 0);
    cacheManager.resetForNewSequences(mHostReuseKVCacheLengths, stream);

    // Mamba SSM buffers are engine inputs and are not covered by resetForNewSequences.
    cacheManager.getMambaCacheManager().clearStates(stream);

    ELLM_CHECK(static_cast<int64_t>(seqLens.size()) >= batchSize, "talker prefill is missing per-row lengths");
    ELLM_CHECK(mTalkerPipelineIO->hostContextLengths.reshape({batchSize}), "Tensor reshape failed");
    int32_t* const hostContextLength = mTalkerPipelineIO->hostContextLengths.dataPointer<int32_t>();
    for (int64_t i = 0; i < batchSize; ++i)
    {
        hostContextLength[i] = seqLens[static_cast<size_t>(i)];
    }

    // Graph capture may have left the logits at the decode shape.
    ELLM_CHECK(outputLogits.reshape({batchSize, outputLogits.getShape()[outputLogits.getShape().getNumDims() - 1]}),
        "Tensor reshape failed");

    // KV cache and rope cache were registered statically by buildTensorMap; only the
    // bindings whose address or shape moves per step are rewired here.
    mTalkerTensorMap.set(binding_names::kInputsEmbeds, const_cast<rt::Tensor&>(inputEmbeds));
    mTalkerTensorMap.set(binding_names::kLogits, outputLogits);
    mTalkerTensorMap.set(binding_names::kOutputHiddenStates, outputHiddenStates);

    mTalkerStepPreparer->prepare(
        rt::InferencePhase::kPrefill, static_cast<int32_t>(batchSize), cacheManager, *mTalkerPipelineIO, stream);

    // The cache was reset above: a re-prefill runs the whole assembled sequence again.
    auto const phase = rt::ExecutionPhase::kContextPrefill;
    std::fill_n(mTalkerStepPastLengths.begin(), batchSize, 0);
    prepareRaggedStep(mTalkerRagged, *mTalkerPipelineIO, *mTalkerSharedRes, mTalkerLLMConfig, seqLens,
        mTalkerStepPastLengths, static_cast<int32_t>(batchSize), static_cast<int32_t>(seqLen), phase, stream);
    auto const prefillDims = mTalkerLLMConfig.prefillDims(batchSize, seqLen, phase);
    if (!mTalkerExec->prepare(/*prefillProfile=*/0, prefillDims, mTalkerTensorMap, stream))
    {
        LOG_ERROR("Talker prefill prepare failed");
        return false;
    }
    if (!mTalkerExec->execute(stream))
    {
        LOG_ERROR("Talker prefill execute failed");
        return false;
    }
    cacheManager.commitSequenceLength(mTalkerPipelineIO->contextLengths, stream);
    return true;
}

bool TalkerRunner::executeTalkerDecodingStep(rt::Tensor const& inputEmbeds, std::vector<int32_t> const& pastLengths,
    rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::decodeStep", nvtx_colors::PURPLE);

    auto const inputShape = inputEmbeds.getShape();
    ELLM_CHECK(inputShape.getNumDims() == 3, "talker decode input must be 3D [batch, 1, hidden]");
    int64_t const batchSize = inputShape[0];
    ELLM_CHECK(batchSize <= mMaxBatch, "talker decode batch size exceeds max batch");

    auto& cacheManager = *mTalkerSharedRes->cacheManagers[0];
    mTalkerTensorMap.set(binding_names::kInputsEmbeds, const_cast<rt::Tensor&>(inputEmbeds));
    mTalkerTensorMap.set(binding_names::kLogits, outputLogits);
    mTalkerTensorMap.set(binding_names::kOutputHiddenStates, outputHiddenStates);

    // No StepPreparer here: its decode branch only writes select_token_indices and
    // context_lengths, which this engine does not bind, and a decode commits by increment.
    ELLM_CHECK(static_cast<int64_t>(pastLengths.size()) >= batchSize, "talker decode is missing per-row past lengths");
    std::fill_n(mTalkerStepQueryLengths.begin(), batchSize, 1);
    prepareRaggedStep(mTalkerRagged, *mTalkerPipelineIO, *mTalkerSharedRes, mTalkerLLMConfig, mTalkerStepQueryLengths,
        pastLengths, static_cast<int32_t>(batchSize), /*queryWidth=*/1, rt::ExecutionPhase::kAutoregressiveDecode,
        stream);

    auto const decodeDims = mTalkerLLMConfig.decodeDims(batchSize);
    if (!mTalkerExec->prepare(/*decodeProfile=*/1, decodeDims, mTalkerTensorMap, stream))
    {
        LOG_ERROR("Talker decode prepare failed");
        return false;
    }
    if (!mTalkerExec->execute(stream))
    {
        LOG_ERROR("Talker decode execute failed");
        return false;
    }
    // Vanilla decode advances the KV cache by exactly one token per call.
    cacheManager.commitSequenceLength(/*increment=*/1, stream);
    return true;
}

bool TalkerRunner::extractTalkerLastHidden(rt::Tensor const& talkerHiddenStates, std::vector<int32_t> const& rowLengths,
    rt::Tensor& outputLastHidden, cudaStream_t stream)
{
    auto const& shape = talkerHiddenStates.getShape();
    if (shape.getNumDims() != 3)
    {
        LOG_ERROR("extractTalkerLastHidden expects [batch, seqLen, hidden], got %dD", shape.getNumDims());
        return false;
    }

    int64_t const batchSize = shape[0];
    int64_t const seqLen = shape[1];
    int64_t const hiddenSize = shape[2];

    // Owning tensors get reshaped; a non-owning view must already have the right shape.
    if (!outputLastHidden.reshape({batchSize, hiddenSize}))
    {
        ELLM_CHECK((outputLastHidden.getShape() == rt::Coords{batchSize, hiddenSize}),
            "extractTalkerLastHidden: non-owning output has wrong shape");
    }

    size_t const rowBytes = static_cast<size_t>(hiddenSize) * sizeof(__half);
    auto const* const base = static_cast<char const*>(talkerHiddenStates.rawPointer());
    auto* const out = static_cast<char*>(outputLastHidden.rawPointer());

    // A prefill window is padded to its longest row, so a shorter row ends before the padded
    // extent does; its last padded slot holds whatever the padding computed.
    bool ragged = false;
    if (seqLen > 1)
    {
        ELLM_CHECK(static_cast<int64_t>(rowLengths.size()) >= batchSize,
            "extractTalkerLastHidden: a padded window needs every row's length");
        for (int64_t b = 0; b < batchSize; ++b)
        {
            auto const length = rowLengths[static_cast<size_t>(b)];
            ELLM_CHECK(length >= 1 && length <= seqLen, "extractTalkerLastHidden: row length outside the window");
            ragged = ragged || length != seqLen;
        }
    }
    if (ragged)
    {
        for (int64_t b = 0; b < batchSize; ++b)
        {
            auto const last = b * seqLen + rowLengths[static_cast<size_t>(b)] - 1;
            CUDA_CHECK(cudaMemcpyAsync(out + static_cast<size_t>(b) * rowBytes,
                base + static_cast<size_t>(last) * rowBytes, rowBytes, cudaMemcpyDeviceToDevice, stream));
        }
        return true;
    }
    CUDA_CHECK(cudaMemcpy2DAsync(out, rowBytes, base + static_cast<size_t>(seqLen - 1) * rowBytes,
        static_cast<size_t>(seqLen) * rowBytes, rowBytes, static_cast<size_t>(batchSize), cudaMemcpyDeviceToDevice,
        stream));
    return true;
}

bool TalkerRunner::computeResidualConnection(Row const& row, std::vector<int32_t> const& codes, int32_t generationStep,
    rt::Tensor& outputResidual, rt::Tensor const& codecHiddens, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::residual", nvtx_colors::BLUE);

    ELLM_CHECK(static_cast<int32_t>(codes.size()) == mNumCodesPerFrame,
        "Expected " + std::to_string(mNumCodesPerFrame) + " codes, got " + std::to_string(codes.size()));

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    if (!outputResidual.reshape({1, 1, hiddenSize}))
    {
        ELLM_CHECK(
            (outputResidual.getShape() == rt::Coords{1, 1, hiddenSize}), "Non-owning residual tensor has wrong shape");
    }

    // The segments layout adds its next trailing text row, then tts_pad once the text runs out;
    // the tts layout has no trailing rows, so it adds tts_pad from the first frame.
    __half const* addend = usesChunkedText() ? static_cast<__half const*>(mZeroResidualAddend.rawPointer())
                                             : mTtsPadEmbed.dataPointer<__half>();
    if (!usesChunkedText() && row.trailingView.rawPointer() != nullptr)
    {
        int64_t const trailingLen = row.trailingView.getShape()[0];
        if (generationStep < trailingLen)
        {
            addend = static_cast<__half const*>(row.trailingView.rawPointer()) + generationStep * hiddenSize;
        }
    }

    rt::Tensor const& position0Table = usesChunkedText() ? mTalkerCodecEmbedTable : mTalkerEmbeddingTable;
    kernel::invokeResidualConnection(codecHiddens, position0Table, mCodePredictor.codecEmbedTable(mNumRvqLayers - 1),
        codes[0], codes[mNumRvqLayers], addend, outputResidual, stream);
    return true;
}

void TalkerRunner::suppressTalkerEosLogit(rt::Tensor& logits, int64_t row, cudaStream_t stream)
{
    // D2D copy from the persistent -INF constant; an async copy must not read a stack value.
    float* const dst
        = static_cast<float*>(logits.rawPointer()) + row * mTalkerConfig.talkerVocabSize + mTalkerConfig.codecEosId;
    CUDA_CHECK(cudaMemcpyAsync(dst, mNegInfConst.rawPointer(), sizeof(float), cudaMemcpyDeviceToDevice, stream));
}

void TalkerRunner::trackSeenToken(Row& row, int32_t slot, int32_t token, int32_t const* tokenDev, cudaStream_t stream)
{
    // A freshly sampled token enters the repetition-penalty window once; the host flags gate
    // the append so it stays coherent with the GPU buffer.
    ELLM_CHECK(token >= 0 && static_cast<size_t>(token) < row.seenTokens.size(),
        "sampled codec token " + std::to_string(token) + " is outside the talker vocabulary");
    if (row.seenTokens[static_cast<size_t>(token)] != 0)
    {
        return;
    }
    row.seenTokens[static_cast<size_t>(token)] = 1;
    int32_t* const dst = mSeenCodecTokensBuf.dataPointer<int32_t>()
        + static_cast<int64_t>(slot) * mTalkerLLMConfig.maxKVCacheCapacity + row.numSeenTokens;
    CUDA_CHECK(cudaMemcpyAsync(dst, tokenDev, sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
    ++row.numSeenTokens;
}

std::unique_ptr<multiStage::IStage> createTalkerRunner(multiStage::StageInitContext const& init)
{
    return std::make_unique<TalkerRunner>(init);
}

multiStage::RunnerDecl talkerRunnerDeclaration()
{
    multiStage::RunnerDecl decl;
    decl.name = kRunnerName;
    decl.kind = multiStage::StageKind::kAutoregressive;
    decl.ports = {{kCodesPort, multiStage::PortKind::kCodec}};
    decl.inputs = {kHiddenInput, kEmbedsInput, kTextInput, kPromptInput, kVoiceInput, kSamplingInput,
        kSpeakerCodesInput, kStyleInput, kRefAudioInput};
    decl.batches = true;
    decl.requiredConfigKeys = {"prefill_builder", kCodePredictorArtifactKey};
    decl.samplingKeys = {kTalkerTemperatureKey, kTalkerTopKKey, kTalkerTopPKey, kRepetitionPenaltyKey,
        kCpTemperatureKey, kCpTopKKey, kCpTopPKey, kMaxAudioLengthKey};
    decl.styleFields = {"talker_language", "language", "assistant_instruct", "instruct", "system_instruct", "ref_text"};
    // qwen3_tts_inference request files spell the code predictor's keys subtalker_*, and the voice speaker.
    decl.requestAliases = {{"subtalker_temperature", kCpTemperatureKey}, {"subtalker_top_k", kCpTopKKey},
        {"subtalker_top_p", kCpTopPKey}, {"speaker", kVoiceInput}, {"prompt_speaker_codes", kSpeakerCodesInput}};
    return decl;
}

} // namespace qwen3_omni
} // namespace rt
} // namespace trt_edgellm
