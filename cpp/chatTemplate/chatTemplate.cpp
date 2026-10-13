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

#include "chatTemplate.h"

#include "injaRenderer.h"

#include "common/inputLimits.h"
#include "common/logger.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace trt_edgellm::chat_template
{
namespace
{
using Json = nlohmann::json;

enum class RawProcessor
{
    kNone,
    kPhi4MM,
    kInternVL,
    kNemotronOmni,
};

struct MediaCounters
{
    size_t images{0};
    size_t audios{0};
};

std::string trimString(std::string value)
{
    auto const isSpace = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(), std::find_if_not(value.begin(), value.end(), isSpace));
    value.erase(std::find_if_not(value.rbegin(), value.rend(), isSpace).base(), value.end());
    return value;
}

Json parseJsonOrString(std::string const& value)
{
    auto parsed = nlohmann::ordered_json::parse(value, nullptr, false);
    return parsed.is_discarded() ? Json(value) : preserveJsonOrder(parsed);
}

//! Qwen3-Omni Instruct checkpoints are trained never to emit reasoning tokens. Binding
//! ``enable_thinking`` makes their provider template prepend an inert
//! ``<think>\n\n</think>\n\n``, which also shifts the Talker prefill slices off the first
//! spoken token. Leaving the flag undefined selects the no-injection branch, as the
//! provider's own ``apply_chat_template`` does when the caller passes no kwarg.
bool omitsEnableThinking(std::filesystem::path const& modelDir)
{
    auto const configPath = modelDir / "config.json";
    std::ifstream stream(configPath);
    if (!stream)
    {
        return false;
    }
    try
    {
        auto const config = Json::parse(stream);
        auto const model = config.value("model", config.value("model_type", std::string{}));
        return model.rfind("qwen3_omni", 0) == 0;
    }
    catch (std::exception const& error)
    {
        LOG_WARNING("Ignoring unparsable %s: %s", configPath.c_str(), error.what());
        return false;
    }
}

std::string readTextFile(std::filesystem::path const& path)
{
    if (!std::filesystem::is_regular_file(path))
    {
        throw std::runtime_error("file does not exist: " + path.string());
    }
    if (std::filesystem::file_size(path) > limits::tokenizer::kChatTemplateFileSizeBytes)
    {
        throw std::runtime_error("chat template exceeds size limit: " + path.string());
    }
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to open " + path.string());
    }
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string textContent(rt::Message const& message)
{
    std::string result;
    for (auto const& item : message.contents)
    {
        if (item.type == "text")
        {
            result += item.content;
        }
    }
    return result;
}

std::string systemContent(rt::Message const& message)
{
    std::string result;
    bool first = true;
    for (auto const& item : message.contents)
    {
        if (item.type != "text")
        {
            continue;
        }
        if (message.contentIsArray && !first)
        {
            result += '\n';
        }
        first = false;
        result += item.content;
    }
    return result;
}

std::vector<rt::Message> normalizeDeveloperMessages(std::vector<rt::Message> messages)
{
    for (auto& message : messages)
    {
        if (message.role == "developer")
        {
            message.role = "system";
        }
    }

    std::vector<std::string> systemContents;
    bool needsConsolidation = false;
    for (size_t index = 0; index < messages.size(); ++index)
    {
        auto const& message = messages[index];
        if (message.role != "system")
        {
            continue;
        }

        needsConsolidation = needsConsolidation || index > 0 || !systemContents.empty();
        auto content = systemContent(message);
        if (!content.empty())
        {
            systemContents.push_back(std::move(content));
        }
    }
    if (!needsConsolidation)
    {
        return messages;
    }

    std::string mergedContent;
    for (size_t index = 0; index < systemContents.size(); ++index)
    {
        if (index > 0)
        {
            mergedContent += "\n\n";
        }
        mergedContent += systemContents[index];
    }
    rt::Message systemMessage;
    systemMessage.role = "system";
    systemMessage.contents.push_back({"text", std::move(mergedContent)});

    std::vector<rt::Message> result;
    result.reserve(messages.size());
    result.push_back(std::move(systemMessage));
    for (auto& message : messages)
    {
        if (message.role != "system")
        {
            result.push_back(std::move(message));
        }
    }
    return result;
}

std::string trajectoryPlaceholder(std::optional<size_t> const& trajectoryPointCount)
{
    if (!trajectoryPointCount)
    {
        throw std::runtime_error("trajectory content requires pastTrajectory data");
    }
    std::string result = rt::kTrajHistoryStartStr;
    for (size_t index = 0; index < 3 * *trajectoryPointCount; ++index)
    {
        result += rt::kTrajHistoryPadStr;
    }
    return result + rt::kTrajHistoryEndStr;
}

std::string phi4MMContent(rt::Message const& message, MediaCounters& counters)
{
    std::string result;
    for (auto const& item : message.contents)
    {
        if (item.type == "text")
        {
            result += item.content;
        }
        else if (item.type == "image")
        {
            result += "<|image_" + std::to_string(++counters.images) + "|>";
        }
        else if (item.type == "audio")
        {
            result += "<|audio_" + std::to_string(++counters.audios) + "|>";
        }
        else
        {
            throw std::runtime_error("Phi4MMProcessor does not accept " + item.type + " message content");
        }
    }
    return result;
}

std::string internVLContent(rt::Message const& message)
{
    std::string result;
    for (auto const& item : message.contents)
    {
        if (item.type == "text")
        {
            result += item.content;
        }
        else if (item.type == "image")
        {
            result += "<IMG_CONTEXT>\n";
        }
        else if (item.type == "video")
        {
            result += "<video>\n";
        }
        else
        {
            throw std::runtime_error("InternVL processor does not accept " + item.type + " message content");
        }
    }
    return result;
}

std::string nemotronOmniContent(rt::Message const& message)
{
    // Inline each media block as its sentinel token (<image>/<audio>/<video>)
    // for the runtime media runner to expand; text blocks pass through.
    std::string result;
    for (auto const& item : message.contents)
    {
        if (item.type == "text")
        {
            result += item.content;
        }
        else if (item.type == "image" || item.type == "audio" || item.type == "video")
        {
            result += "<" + item.type + ">";
        }
        else
        {
            throw std::runtime_error("NemotronOmni processor does not accept " + item.type + " message content");
        }
    }
    return result;
}

void replaceNumberedAliases(std::string& value, std::string_view prefix, std::string_view replacement)
{
    size_t offset = 0;
    while ((offset = value.find(prefix, offset)) != std::string::npos)
    {
        size_t cursor = offset + prefix.size();
        size_t const digitBegin = cursor;
        while (cursor < value.size() && std::isdigit(static_cast<unsigned char>(value[cursor])) != 0)
        {
            ++cursor;
        }
        if (cursor == digitBegin || value.compare(cursor, 2, "|>") != 0)
        {
            offset = cursor;
            continue;
        }
        value.replace(offset, cursor + 2 - offset, replacement);
        offset += replacement.size();
    }
}

std::string applyRawProcessor(std::string value, RawProcessor processor)
{
    if (processor == RawProcessor::kPhi4MM)
    {
        replaceNumberedAliases(value, "<|image_", "<|endoftext10|>");
        replaceNumberedAliases(value, "<|audio_", "<|endoftext11|>");
    }
    return value;
}

Json contentToJson(rt::Message const& message, std::optional<size_t> const& trajectoryPointCount,
    RawProcessor processor, bool expectsContentBlocks, MediaCounters& counters)
{
    if (message.contentIsNull)
    {
        if (!message.contents.empty())
        {
            throw std::runtime_error("null message content cannot contain content blocks");
        }
        return nullptr;
    }
    if (processor == RawProcessor::kPhi4MM)
    {
        return phi4MMContent(message, counters);
    }
    if (processor == RawProcessor::kInternVL)
    {
        return internVLContent(message);
    }
    if (processor == RawProcessor::kNemotronOmni)
    {
        return nemotronOmniContent(message);
    }

    if (!expectsContentBlocks || message.role == "tool")
    {
        // A string-content template renders message.content as a single string.
        // Media requires a rendering contract the model owns: either a provider
        // template that iterates content blocks (expectsContentBlocks), or a
        // raw processor that emits the model's media sentinels. Without one, a
        // media block has no valid placeholder in the string and is rejected.
        std::string content;
        bool first = true;
        for (auto const& item : message.contents)
        {
            if (item.type != "text")
            {
                throw std::runtime_error("provider string-content template requires a model-specific media processor");
            }
            if (!first)
            {
                content += '\n';
            }
            first = false;
            content += item.content;
        }
        return content;
    }

    Json content = Json::array();
    for (auto const& item : message.contents)
    {
        Json block = makeInjaObject({{"type", item.type}});
        if (item.type == "text")
        {
            setInjaObjectField(block, "text", item.content);
        }
        else if (item.type == "trajectory")
        {
            setInjaObjectField(block, "placeholder", trajectoryPlaceholder(trajectoryPointCount));
        }
        else
        {
            setInjaObjectField(block, item.type, item.content);
        }
        content.push_back(std::move(block));
    }
    return content;
}

Json messageToJson(rt::Message const& message, std::optional<size_t> const& trajectoryPointCount,
    RawProcessor processor, bool expectsContentBlocks, MediaCounters& counters)
{
    Json value = makeInjaObject({{"role", message.role}});
    if (message.hasContent)
    {
        setInjaObjectField(
            value, "content", contentToJson(message, trajectoryPointCount, processor, expectsContentBlocks, counters));
    }
    if (message.hasReasoningContent)
    {
        setInjaObjectField(value, "reasoning_content", message.reasoningContent);
        setInjaObjectField(value, "reasoning", message.reasoningContent);
    }
    if (!message.toolCallId.empty())
    {
        setInjaObjectField(value, "tool_call_id", message.toolCallId);
    }
    if (!message.name.empty())
    {
        setInjaObjectField(value, "name", message.name);
    }
    if (message.hasToolCalls || !message.toolCalls.empty())
    {
        Json toolCalls = Json::array();
        for (auto const& call : message.toolCalls)
        {
            Json function = makeInjaObject({{"name", call.name},
                {"arguments", call.argumentsIsString ? Json(call.arguments) : parseJsonOrString(call.arguments)}});
            toolCalls.push_back(
                makeInjaObject({{"id", call.id}, {"type", call.type}, {"function", std::move(function)}}));
        }
        setInjaObjectField(value, "tool_calls", std::move(toolCalls));
    }
    return value;
}

Json toolToJson(rt::ToolDefinition const& tool)
{
    Json function = makeInjaObject({{"name", tool.name}});
    if (tool.hasDescription)
    {
        setInjaObjectField(function, "description", tool.description);
    }
    if (tool.hasParameters)
    {
        setInjaObjectField(function, "parameters", parseJsonOrString(tool.parameters));
    }
    if (tool.hasStrict)
    {
        setInjaObjectField(function, "strict", tool.strict);
    }
    return makeInjaObject({{"type", "function"}, {"function", std::move(function)}});
}

Json toolChoiceToJson(rt::ToolChoice const& toolChoice)
{
    switch (toolChoice.mode)
    {
    case rt::ToolChoice::Mode::kNone: return "none";
    case rt::ToolChoice::Mode::kRequired: return "required";
    case rt::ToolChoice::Mode::kFunction:
        return makeInjaObject(
            {{"type", "function"}, {"function", makeInjaObject({{"name", toolChoice.functionName}})}});
    case rt::ToolChoice::Mode::kAuto: return "auto";
    }
    return "auto";
}

std::string renderRaw(rt::LLMGenerationRequest::Request const& request)
{
    std::string result;
    for (auto const& message : request.messages)
    {
        for (auto const& item : message.contents)
        {
            if (item.type == "text")
            {
                result += item.content;
            }
            else if (item.type == "trajectory")
            {
                auto const count
                    = request.pastTrajectory ? std::optional<size_t>(request.pastTrajectory->size()) : std::nullopt;
                result += trajectoryPlaceholder(count);
            }
            else if (!item.content.empty())
            {
                result += item.content;
            }
            else
            {
                throw std::runtime_error("raw prompt mode requires preformatted content for " + item.type + " inputs");
            }
        }
    }
    return result;
}

std::string renderManual(std::string const& family, std::vector<rt::Message> const& messages,
    std::optional<size_t> const& trajectoryPointCount, bool addGenerationPrompt, bool systemPromptOnly)
{
    std::string result;
    if (family == "alpamayo")
    {
        for (auto const& message : messages)
        {
            result += "<|im_start|>" + message.role + "\n";
            for (auto const& item : message.contents)
            {
                if (item.type == "text")
                    result += item.content;
                else if (item.type == "image")
                    result += "<|vision_start|><|image_pad|><|vision_end|>";
                else if (item.type == "video")
                    result += "<|vision_start|><|video_pad|><|vision_end|>";
                else if (item.type == "trajectory")
                    result += trajectoryPlaceholder(trajectoryPointCount);
                else
                    throw std::runtime_error("alpamayo does not accept " + item.type + " message content");
            }
            result += "<|im_end|>\n";
        }
    }
    else if (family == "qwen3_asr")
    {
        std::string systemPrompt;
        std::string audioPrompt;
        for (auto const& message : messages)
        {
            for (auto const& item : message.contents)
            {
                if (message.role == "system" && item.type == "text")
                    systemPrompt += item.content;
                if (!systemPromptOnly && item.type == "audio")
                    audioPrompt += "<|audio_start|><|audio_pad|><|audio_end|>";
            }
        }
        result = "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n<|im_start|>user\n";
        if (systemPromptOnly)
        {
            return result;
        }
        result += audioPrompt + "<|im_end|>\n";
    }
    else if (family == "qwen3_tts")
    {
        for (auto const& message : messages)
        {
            result += "<|im_start|>" + message.role + "\n";
            for (auto const& item : message.contents)
            {
                if (item.type != "text")
                    throw std::runtime_error("qwen3_tts does not accept " + item.type + " message content");
                result += item.content;
            }
            result += "<|im_end|>\n";
        }
    }
    else
    {
        throw std::runtime_error("unknown native chat renderer: " + family);
    }

    if (addGenerationPrompt)
    {
        result += "<|im_start|>assistant\n";
    }
    return result;
}

} // namespace

class ChatTemplate::Impl
{
public:
    bool load(std::filesystem::path const& modelDir, std::string bosToken, std::string eosToken)
    {
        auto const providerPath = modelDir / "chat_template.jinja";
        auto const additionalPath = modelDir / "additional_chat_templates";
        auto const manualPath = modelDir / "chat_template.model";
        auto const processorPath = modelDir / "chat_template.processor";
        mMode = Mode::kUninitialized;
        mProviderTemplate.reset();
        mManualFamily.clear();
        mRawProcessor = RawProcessor::kNone;
        mOmitEnableThinking = omitsEnableThinking(modelDir);
        mBosToken = std::move(bosToken);
        mEosToken = std::move(eosToken);

        try
        {
            if (std::filesystem::exists(additionalPath))
            {
                throw std::runtime_error(
                    "named provider chat templates are unsupported; the model must provide chat_template.jinja");
            }
            if (std::filesystem::exists(providerPath))
            {
                mProviderTemplate = std::make_unique<InjaRenderer>();
                mProviderTemplate->load(providerPath);
            }

            bool const hasProvider = mProviderTemplate != nullptr;
            bool const hasManual = std::filesystem::exists(manualPath);
            if (hasProvider && hasManual)
            {
                throw std::runtime_error("both provider template and native renderer marker exist");
            }
            if (std::filesystem::exists(processorPath) && !hasProvider)
            {
                throw std::runtime_error("raw processor marker requires a provider template");
            }
            if (hasProvider)
            {
                if (std::filesystem::exists(processorPath))
                {
                    auto const processor = trimString(readTextFile(processorPath));
                    if (processor == "phi4mm")
                    {
                        mRawProcessor = RawProcessor::kPhi4MM;
                    }
                    else if (processor == "internvl")
                    {
                        mRawProcessor = RawProcessor::kInternVL;
                    }
                    else if (processor == "nemotron_omni")
                    {
                        mRawProcessor = RawProcessor::kNemotronOmni;
                    }
                    else
                    {
                        throw std::runtime_error("unsupported raw processor: " + processor);
                    }
                }
                mMode = Mode::kJinja;
                LOG_INFO("Loaded provider Jinja chat template through Pantor Inja from %s", providerPath.c_str());
                return true;
            }
            if (hasManual)
            {
                mManualFamily = trimString(readTextFile(manualPath));
                static std::unordered_set<std::string> const supported{"alpamayo", "qwen3_asr", "qwen3_tts"};
                if (supported.count(mManualFamily) == 0)
                {
                    throw std::runtime_error("unknown native chat renderer: " + mManualFamily);
                }
                mMode = Mode::kManual;
                LOG_INFO("Loaded native chat renderer '%s' from %s", mManualFamily.c_str(), manualPath.c_str());
                return true;
            }
            LOG_ERROR("No provider Jinja or model-specific native chat renderer exists in %s", modelDir.c_str());
        }
        catch (std::exception const& error)
        {
            LOG_ERROR("Failed to load chat template from %s: %s", modelDir.c_str(), error.what());
        }
        return false;
    }

    bool apply(rt::LLMGenerationRequest::Request const& request,
        rt::LLMGenerationRequest::FormattedRequest& formattedRequest, ChatTemplate::Options const& options) const
    {
        if (request.messages.empty())
        {
            LOG_ERROR("Request must contain at least one message");
            return false;
        }

        try
        {
            if (!options.applyTemplate)
            {
                formattedRequest.formattedCompleteRequest = renderRaw(request);
                formattedRequest.formattedSystemPrompt
                    = request.messages.front().role == "system" ? textContent(request.messages.front()) : "";
                return true;
            }
            if (mMode == Mode::kUninitialized)
            {
                throw std::runtime_error("chat template is not loaded");
            }

            auto const* messages = &request.messages;
            std::vector<rt::Message> normalizedMessages;
            auto const trajectoryPointCount
                = request.pastTrajectory ? std::optional<size_t>(request.pastTrajectory->size()) : std::nullopt;
            InjaRenderer const* provider = mMode == Mode::kJinja ? mProviderTemplate.get() : nullptr;
            if (provider != nullptr && !provider->supportsDeveloperRole()
                && std::any_of(request.messages.begin(), request.messages.end(),
                    [](rt::Message const& message) { return message.role == "developer"; }))
            {
                normalizedMessages = normalizeDeveloperMessages(request.messages);
                messages = &normalizedMessages;
            }

            auto render = [&](std::vector<rt::Message> const& messages, bool addGenerationPrompt,
                              bool systemPromptOnly = false) {
                if (mMode == Mode::kManual)
                {
                    return renderManual(
                        mManualFamily, messages, trajectoryPointCount, addGenerationPrompt, systemPromptOnly);
                }

                Json inputs = makeInjaObject({{"messages", Json::array()}});
                MediaCounters counters;
                bool const expectsContentBlocks = provider->expectsContentBlocks();
                for (auto const& message : messages)
                {
                    inputs["messages"].push_back(
                        messageToJson(message, trajectoryPointCount, mRawProcessor, expectsContentBlocks, counters));
                }
                setInjaObjectField(inputs, "tools", options.tools.empty() ? Json(nullptr) : Json::array());
                for (auto const& tool : options.tools)
                {
                    inputs["tools"].push_back(toolToJson(tool));
                }
                setInjaObjectField(inputs, "add_generation_prompt", addGenerationPrompt);
                setInjaObjectField(inputs, "bos_token", mBosToken);
                setInjaObjectField(inputs, "eos_token", mEosToken);
                setInjaObjectField(inputs, "strftime_now", true);
                if (!mOmitEnableThinking)
                {
                    setInjaObjectField(inputs, "enable_thinking", options.enableThinking);
                }
                if (!options.reasoningEffort.empty())
                {
                    setInjaObjectField(inputs, "reasoning_effort", options.reasoningEffort);
                }
                setInjaObjectField(inputs, "parallel_tool_calls", options.parallelToolCalls);
                setInjaObjectField(inputs, "tool_choice", toolChoiceToJson(options.toolChoice));
                setInjaObjectField(inputs, "continue_final_message", false);
                return applyRawProcessor(provider->render(inputs), mRawProcessor);
            };

            formattedRequest.formattedCompleteRequest = render(*messages, options.addGenerationPrompt);
            formattedRequest.formattedSystemPrompt.clear();
            if (messages->front().role == "system")
            {
                if (mMode == Mode::kManual && mManualFamily == "qwen3_asr")
                {
                    formattedRequest.formattedSystemPrompt = render(*messages, false, true);
                }
                else
                {
                    try
                    {
                        auto const prefix = render({messages->front()}, false);
                        if (formattedRequest.formattedCompleteRequest.rfind(prefix, 0) == 0)
                        {
                            formattedRequest.formattedSystemPrompt = prefix;
                        }
                    }
                    catch (std::exception const&)
                    {
                        // Some provider templates require a user turn. In that case,
                        // skip system-only KV reuse instead of inventing a prefix.
                    }
                }
            }
            return true;
        }
        catch (std::exception const& error)
        {
            LOG_ERROR("Failed to apply chat template: %s", error.what());
            return false;
        }
    }

    bool isLoaded() const noexcept
    {
        return mMode != Mode::kUninitialized;
    }

private:
    enum class Mode
    {
        kUninitialized,
        kJinja,
        kManual,
    };

    std::unique_ptr<InjaRenderer> mProviderTemplate;
    std::string mManualFamily;
    std::string mBosToken;
    std::string mEosToken;
    RawProcessor mRawProcessor{RawProcessor::kNone};
    bool mOmitEnableThinking{false};
    Mode mMode{Mode::kUninitialized};
};

ChatTemplate::ChatTemplate()
    : mImpl(std::make_unique<Impl>())
{
}

ChatTemplate::Options ChatTemplate::optionsFrom(rt::LLMGenerationRequest const& request)
{
    Options options;
    options.applyTemplate = request.applyChatTemplate;
    options.addGenerationPrompt = request.addGenerationPrompt;
    options.enableThinking = request.enableThinking;
    options.reasoningEffort = request.reasoningEffort;
    options.tools = request.tools;
    options.toolChoice = request.toolChoice;
    options.parallelToolCalls = request.parallelToolCalls;
    return options;
}

ChatTemplate::~ChatTemplate() = default;
ChatTemplate::ChatTemplate(ChatTemplate&&) noexcept = default;
ChatTemplate& ChatTemplate::operator=(ChatTemplate&&) noexcept = default;

bool ChatTemplate::load(std::filesystem::path const& modelDir, std::string bosToken, std::string eosToken)
{
    return mImpl->load(modelDir, std::move(bosToken), std::move(eosToken));
}

bool ChatTemplate::apply(rt::LLMGenerationRequest::Request const& request,
    rt::LLMGenerationRequest::FormattedRequest& formattedRequest, Options const& options) const
{
    return mImpl->apply(request, formattedRequest, options);
}

bool ChatTemplate::isLoaded() const noexcept
{
    return mImpl->isLoaded();
}

} // namespace trt_edgellm::chat_template
