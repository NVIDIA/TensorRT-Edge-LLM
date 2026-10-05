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

#include "requestFileParser.h"

#include "common/checkMacros.h"
#include "common/inputLimits.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "runtime/audioLoader.h"
#include "runtime/audioUtils.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/imageUtils.h"
#include "sampler/sampling.h" // kMaxLogprobsK

#include <cmath>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>

namespace trt_edgellm
{
namespace exampleUtils
{

using Json = nlohmann::json;

namespace
{

std::unordered_map<int32_t, float> parseLogitBias(Json const& logitBiasJson, std::string const& fieldName)
{
    check::check(logitBiasJson.is_object(), fieldName + " must be an object mapping token IDs to bias values");
    check::check(logitBiasJson.size() <= limits::security::kMaxLogitBiasTokens,
        format::fmtstr("%s has %zu entries, max is %zu (matching vLLM)", fieldName.c_str(), logitBiasJson.size(),
            limits::security::kMaxLogitBiasTokens));

    std::unordered_map<int32_t, float> logitBias;
    logitBias.reserve(logitBiasJson.size());
    for (auto const& [tokenIdStr, biasJson] : logitBiasJson.items())
    {
        size_t parsedChars = 0;
        long long tokenIdLong = 0;
        try
        {
            tokenIdLong = std::stoll(tokenIdStr, &parsedChars);
        }
        catch (std::exception const&)
        {
            throw std::runtime_error(fieldName + " token ID '" + tokenIdStr + "' is not an integer");
        }
        check::check(parsedChars == tokenIdStr.size(), fieldName + " token ID '" + tokenIdStr + "' is not an integer");
        check::check(
            tokenIdLong >= std::numeric_limits<int32_t>::min() && tokenIdLong <= std::numeric_limits<int32_t>::max(),
            fieldName + " token ID '" + tokenIdStr + "' is outside int32 range");
        check::check(biasJson.is_number(), fieldName + " value for token ID '" + tokenIdStr + "' must be numeric");
        float const bias = biasJson.get<float>();
        check::check(
            std::isfinite(bias) && bias >= limits::security::kMinLogitBias && bias <= limits::security::kMaxLogitBias,
            format::fmtstr("%s value for token ID '%s' must be finite and in [%.1f, %.1f]", fieldName.c_str(),
                tokenIdStr.c_str(), limits::security::kMinLogitBias, limits::security::kMaxLogitBias));
        logitBias[static_cast<int32_t>(tokenIdLong)] = bias;
    }
    return logitBias;
}

/*!
 * @brief Parse the low-level `guided_decoding` field into runtime parameters.
 *
 * Flat, mutually exclusive fields rather than a {type, guide} pair, because that reads
 * far better in a hand-written JSON file. `json_schema` and `structural_tag` accept an
 * inline object, and `choice` an inline array -- requiring an escaped JSON string would
 * make them nearly unusable -- and the value is re-serialized so that different
 * formatting of the same schema collapse onto one compiled-grammar cache entry.
 */
std::optional<rt::GuidedDecodingParams> parseGuidedDecoding(Json const& guidedJson, std::string const& fieldName)
{
    check::check(guidedJson.is_object(), fieldName + " must be an object");

    struct FieldSpec
    {
        char const* key;
        rt::GuideType type;
    };
    // Order matches the documented modes.
    constexpr FieldSpec kFIELDS[] = {
        {"json_object", rt::GuideType::kJsonObject},
        {"json_schema", rt::GuideType::kJsonSchema},
        {"regex", rt::GuideType::kRegex},
        {"ebnf", rt::GuideType::kEbnf},
        {"structural_tag", rt::GuideType::kStructuralTag},
        {"choice", rt::GuideType::kChoice},
    };

    std::optional<rt::GuidedDecodingParams> parsed;
    std::string firstKey;
    for (auto const& [key, type] : kFIELDS)
    {
        if (!guidedJson.contains(key) || guidedJson[key].is_null())
        {
            continue;
        }
        auto const& value = guidedJson[key];

        rt::GuidedDecodingParams candidate;
        candidate.type = type;
        if (type == rt::GuideType::kJsonObject)
        {
            check::check(value.is_boolean(), fieldName + ".json_object must be a boolean");
            if (!value.get<bool>())
            {
                continue; // `false` means "not requested", not "constrain to nothing".
            }
        }
        else if (type == rt::GuideType::kChoice)
        {
            check::check(value.is_array(), fieldName + ".choice must be an array of strings");
            candidate.guide = value.dump();
        }
        else if (value.is_string())
        {
            candidate.guide = value.get<std::string>();
            // An empty string means "unset". Treating it as a grammar would compile to a
            // language accepting nothing and surface as an unsatisfiable mask.
            if (candidate.guide.empty())
            {
                continue;
            }
        }
        else if (type == rt::GuideType::kJsonSchema || type == rt::GuideType::kStructuralTag)
        {
            check::check(value.is_object(), fieldName + "." + key + " must be an object or a JSON string");
            candidate.guide = value.dump();
        }
        else
        {
            throw std::runtime_error(fieldName + "." + key + " must be a string");
        }

        check::check(!parsed.has_value(),
            format::fmtstr("%s sets both '%s' and '%s'; exactly one guided decoding mode may be used",
                fieldName.c_str(), firstKey.c_str(), key));
        firstKey = key;
        parsed = std::move(candidate);
    }

    if (parsed.has_value())
    {
        // Same validation the runtime applies, run here so the CLI reports the real
        // reason instead of a generic request rejection.
        std::string failReason;
        // Validate first and test afterwards: the two arguments of check::check are
        // unsequenced, so building the message inline can read failReason before it is set.
        bool const valid = rt::validateGuidedDecodingParams(*parsed, failReason);
        check::check(valid, fieldName + ": " + failReason);
    }
    return parsed;
}

rt::ContextCacheLookupPolicy parseContextCacheLookupPolicy(Json const& input)
{
    std::string const value = input.value("context_cache_lookup_policy", "use_cache");
    if (value == "use_cache")
    {
        return rt::ContextCacheLookupPolicy::kUseCache;
    }
    if (value == "bypass")
    {
        return rt::ContextCacheLookupPolicy::kBypass;
    }
    throw std::runtime_error("context_cache_lookup_policy must be either 'use_cache' or 'bypass', got '" + value + "'");
}

rt::ContextCacheCommitPolicy parseContextCacheCommitPolicy(Json const& input)
{
    std::string const value = input.value("context_cache_commit_policy", "including_generated_tokens");
    if (value == "including_generated_tokens")
    {
        return rt::ContextCacheCommitPolicy::kIncludingGeneratedTokens;
    }
    if (value == "prefill_state_only")
    {
        return rt::ContextCacheCommitPolicy::kPrefillStateOnly;
    }
    throw std::runtime_error(
        "context_cache_commit_policy must be either 'including_generated_tokens' or "
        "'prefill_state_only', got '"
        + value + "'");
}

} // namespace

std::pair<std::unordered_map<std::string, std::string>, std::vector<rt::LLMGenerationRequest>> parseRequestFile(
    std::filesystem::path const& inputFilePath, int32_t batchSizeOverride, int64_t maxGenerateLengthOverride,
    int32_t numLogprobsOverride)
{
    std::vector<rt::LLMGenerationRequest> batchedRequests;

    Json inputData;
    std::ifstream inputFileStream(inputFilePath);
    check::check(inputFileStream.is_open(), "Failed to open input file: " + inputFilePath.string());
    try
    {
        inputData = Json::parse(inputFileStream);
        inputFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        throw std::runtime_error(
            format::fmtstr("Failed to parse input file %s with error: %s", inputFilePath.string().c_str(), e.what()));
    }

    int batchSize = (batchSizeOverride != -1) ? batchSizeOverride : inputData.value("batch_size", 1);
    check::check(batchSize > 0, format::fmtstr("Invalid batch_size value: %d (must be positive)", batchSize));
    check::check(batchSize <= limits::security::kReasonableMaxBatchSize,
        format::fmtstr("Input rejected: batch_size %d exceeds limit %d. Limit defined in %s.", batchSize,
            limits::security::kReasonableMaxBatchSize, limits::kInputLimitsLocation));

    float temperature = inputData.value("temperature", 1.0f);
    float topP = inputData.value("top_p", 0.8f);
    int64_t topK = inputData.value("top_k", 50);
    std::optional<uint64_t> defaultSamplingSeed;
    if (inputData.contains("sampling_seed") && !inputData["sampling_seed"].is_null())
    {
        defaultSamplingSeed = inputData["sampling_seed"].get<uint64_t>();
    }
    std::string const proposalSamplingValue = inputData.value("spec_proposal_sampling", std::string{"auto"});
    rt::SpecProposalSampling proposalSampling = rt::SpecProposalSampling::kAuto;
    if (proposalSamplingValue == "greedy")
    {
        proposalSampling = rt::SpecProposalSampling::kGreedy;
    }
    else if (proposalSamplingValue == "probabilistic")
    {
        proposalSampling = rt::SpecProposalSampling::kProbabilistic;
    }
    else
    {
        check::check(proposalSamplingValue == "auto", "spec_proposal_sampling must be auto, greedy, or probabilistic");
    }
    int64_t maxGenerateLength
        = (maxGenerateLengthOverride != -1) ? maxGenerateLengthOverride : inputData.value("max_generate_length", 256);
    check::check(maxGenerateLength > 0,
        format::fmtstr(
            "Invalid max_generate_length value: %lld (must be positive)", static_cast<long long>(maxGenerateLength)));

    bool applyChatTemplate = inputData.value("apply_chat_template", true);
    bool addGenerationPrompt = inputData.value("add_generation_prompt", true);
    bool enableThinking = inputData.value("enable_thinking", false);
    int32_t diffusionMaxDenoisingSteps = 0;
    if (inputData.contains("diffusion_config") && !inputData["diffusion_config"].is_null())
    {
        check::check(inputData["diffusion_config"].is_object(), "diffusion_config must be an object");
        diffusionMaxDenoisingSteps = inputData["diffusion_config"].value("max_denoising_steps", 0);
    }
    diffusionMaxDenoisingSteps = inputData.value("diffusion_max_denoising_steps", diffusionMaxDenoisingSteps);
    check::check(diffusionMaxDenoisingSteps >= 0,
        format::fmtstr(
            "Invalid diffusion max_denoising_steps value: %d (must be non-negative)", diffusionMaxDenoisingSteps));
    rt::ContextCacheLookupPolicy const contextCacheLookupPolicy = parseContextCacheLookupPolicy(inputData);
    rt::ContextCacheCommitPolicy const contextCacheCommitPolicy = parseContextCacheCommitPolicy(inputData);
    std::unordered_map<int32_t, float> defaultLogitBias;
    if (inputData.contains("logit_bias") && !inputData["logit_bias"].is_null())
    {
        defaultLogitBias = parseLogitBias(inputData["logit_bias"], "logit_bias");
    }
    // A top-level guided_decoding applies to every request unless a request overrides it.
    std::optional<rt::GuidedDecodingParams> defaultGuidedDecoding;
    if (inputData.contains("guided_decoding") && !inputData["guided_decoding"].is_null())
    {
        defaultGuidedDecoding = parseGuidedDecoding(inputData["guided_decoding"], "guided_decoding");
    }
    // Top-level num_logprobs is the default for every request; a request may raise it.
    // A CLI override (>= 0) takes precedence over both file levels, mirroring
    // batchSizeOverride / maxGenerateLengthOverride.
    int32_t const defaultNumLogprobs
        = (numLogprobsOverride >= 0) ? numLogprobsOverride : inputData.value("num_logprobs", 0);
    check::check(defaultNumLogprobs >= 0 && defaultNumLogprobs <= kMaxLogprobsK,
        format::fmtstr("Invalid num_logprobs value: %d (must be in [0, %d])", defaultNumLogprobs, kMaxLogprobsK));

    std::unordered_map<std::string, std::string> loraWeightsMap;
    if (inputData.contains("available_lora_weights") && inputData["available_lora_weights"].is_object())
    {
        auto const& availableLoraWeights = inputData["available_lora_weights"];
        for (auto const& [loraName, loraPath] : availableLoraWeights.items())
        {
            check::check(loraPath.is_string(), "LoRA weight path for '" + loraName + "' must be a string");
            check::check(loraWeightsMap.find(loraName) == loraWeightsMap.end(),
                "Lora weights with name " + loraName + " already exists");
            loraWeightsMap[loraName] = loraPath.get<std::string>();
            LOG_INFO("Registered LoRA weights '%s' -> '%s'", loraName.c_str(), loraWeightsMap[loraName].c_str());
        }
    }

    if (!(inputData.contains("requests") && inputData["requests"].is_array()))
    {
        throw std::runtime_error("'requests' array not found in input file");
    }

    auto& requestsArray = inputData["requests"];
    size_t const numRequests = requestsArray.size();

    for (size_t startIdx = 0; startIdx < numRequests; startIdx += batchSize)
    {
        rt::LLMGenerationRequest batchRequest;
        batchRequest.temperature = temperature;
        batchRequest.topP = topP;
        batchRequest.topK = topK;
        batchRequest.maxGenerateLength = maxGenerateLength;
        batchRequest.samplingSeed = defaultSamplingSeed;
        batchRequest.proposalSampling = proposalSampling;
        batchRequest.diffusionMaxDenoisingSteps = diffusionMaxDenoisingSteps;
        batchRequest.applyChatTemplate = applyChatTemplate;
        batchRequest.addGenerationPrompt = addGenerationPrompt;
        batchRequest.enableThinking = enableThinking;
        batchRequest.numLogprobs = defaultNumLogprobs;
        batchRequest.contextCacheLookupPolicy = contextCacheLookupPolicy;
        batchRequest.contextCacheCommitPolicy = contextCacheCommitPolicy;

        std::string batchLoraWeightsName;
        bool firstInBatch = true;

        size_t const endIdx = std::min(startIdx + static_cast<size_t>(batchSize), numRequests);
        for (size_t requestIdx = startIdx; requestIdx < endIdx; ++requestIdx)
        {
            auto const& requestItem = requestsArray[requestIdx];
            check::check(requestItem.is_object(), "Each request must be an object with 'messages' key");

            bool saveSystemPromptKVCache = requestItem.value("save_system_prompt_kv_cache", false);
            if (saveSystemPromptKVCache)
            {
                batchRequest.saveSystemPromptKVCache = true;
            }
            bool disableSpecDecode = requestItem.value("disable_spec_decode", false);
            if (disableSpecDecode)
            {
                batchRequest.disableSpecDecode = true;
            }
            // num_logprobs: request value overrides the top-level default (unless a CLI
            // override is active); applied batch-uniformly (like disable_spec_decode) —
            // the batch uses the max.
            int32_t const requestNumLogprobs = (numLogprobsOverride >= 0)
                ? numLogprobsOverride
                : requestItem.value("num_logprobs", defaultNumLogprobs);
            check::check(requestNumLogprobs >= 0 && requestNumLogprobs <= kMaxLogprobsK,
                format::fmtstr(
                    "Invalid num_logprobs value: %d (must be in [0, %d])", requestNumLogprobs, kMaxLogprobsK));
            batchRequest.numLogprobs = std::max(batchRequest.numLogprobs, requestNumLogprobs);

            std::string requestLoraName;
            if (requestItem.contains("lora_name") && !requestItem["lora_name"].is_null())
            {
                requestLoraName = requestItem["lora_name"].get<std::string>();
                check::check(requestLoraName.empty() || loraWeightsMap.find(requestLoraName) != loraWeightsMap.end(),
                    "LoRA name '" + requestLoraName + "' not found in available_lora_weights");
            }

            if (firstInBatch)
            {
                batchLoraWeightsName = requestLoraName;
                firstInBatch = false;
            }
            else
            {
                check::check(requestLoraName == batchLoraWeightsName,
                    "Different LoRA weights within the same batch are not supported");
            }

            // Per-request messages
            check::check(requestItem.contains("messages") && requestItem["messages"].is_array(),
                "Each request object must contain a 'messages' array");

            auto const& messagesArray = requestItem["messages"];

            check::check(messagesArray.size() <= limits::security::kMaxMessagesPerRequest,
                format::fmtstr("Input rejected: too many messages in request %zu: %zu (max: %zu). Limit defined in %s.",
                    requestIdx, messagesArray.size(), limits::security::kMaxMessagesPerRequest,
                    limits::kInputLimitsLocation));

            std::vector<rt::Message> chatMessages;
            std::vector<rt::imageUtils::ImageData> imageBuffers;
            std::vector<rt::audioUtils::AudioData> audioBuffers;

            for (auto const& messageJson : messagesArray)
            {
                check::check(messageJson.contains("role") && messageJson.contains("content"),
                    "Each message must have 'role' and 'content' fields");

                rt::Message chatMsg;
                chatMsg.role = messageJson["role"].get<std::string>();

                auto const& contentJson = messageJson["content"];

                if (contentJson.is_string())
                {
                    std::string const& contentStr = contentJson.get<std::string>();
                    check::check(contentStr.size() <= limits::security::kMaxMessageContentSizeBytes,
                        format::fmtstr(
                            "Input rejected: message content too large in request %zu: %zu bytes (max: %zu). "
                            "Limit defined in %s.",
                            requestIdx, contentStr.size(), limits::security::kMaxMessageContentSizeBytes,
                            limits::kInputLimitsLocation));

                    rt::Message::MessageContent msgContent;
                    msgContent.type = "text";
                    msgContent.content = contentStr;
                    chatMsg.contents.push_back(msgContent);
                }
                else if (contentJson.is_array())
                {
                    check::check(contentJson.size() <= limits::security::kMaxContentItemsPerMessage,
                        format::fmtstr("Input rejected: too many content items in message %zu: %zu (max: %zu). "
                                       "Limit defined in %s.",
                            requestIdx, contentJson.size(), limits::security::kMaxContentItemsPerMessage,
                            limits::kInputLimitsLocation));

                    for (auto const& contentItemJson : contentJson)
                    {
                        check::check(contentItemJson.contains("type"), "Each content item must have a 'type' field");
                        rt::Message::MessageContent msgContent;
                        msgContent.type = contentItemJson["type"].get<std::string>();

                        if (msgContent.type == "text")
                        {
                            std::string const& textContent = contentItemJson["text"].get<std::string>();
                            check::check(textContent.size() <= limits::security::kMaxMessageContentSizeBytes,
                                format::fmtstr("Input rejected: message content too large in request %zu: %zu bytes "
                                               "(max: %zu). Limit defined in %s.",
                                    requestIdx, textContent.size(), limits::security::kMaxMessageContentSizeBytes,
                                    limits::kInputLimitsLocation));
                            msgContent.content = textContent;
                        }
                        else if (msgContent.type == "image")
                        {
                            msgContent.content = contentItemJson["image"].get<std::string>();
                            auto image = rt::imageUtils::loadRgbImageFromFile(msgContent.content);
                            if (image.buffer != nullptr)
                            {
                                image.doResize = contentItemJson.value("do_resize", true);
                                imageBuffers.push_back(std::move(image));
                            }
                        }
                        else if (msgContent.type == "video")
                        {
                            // Schema: {"type": "video", "frames": [path1, path2, ...], "fps": 1.0}
                            // fps is optional (default 1.0); frames are assumed pre-sampled in temporal order.
                            check::check(contentItemJson.contains("frames") && contentItemJson["frames"].is_array(),
                                "Video content must have a 'frames' array of frame file paths");
                            std::vector<std::string> framePaths;
                            for (auto const& f : contentItemJson["frames"])
                            {
                                framePaths.push_back(f.get<std::string>());
                            }
                            double const fps = contentItemJson.value("fps", 1.0);
                            msgContent.content = "video[" + std::to_string(framePaths.size()) + " frames]";
                            auto video = rt::imageUtils::loadRgbVideoFromFrames(framePaths, fps);
                            if (video.buffer != nullptr)
                            {
                                video.doResize = contentItemJson.value("do_resize", true);
                                imageBuffers.push_back(std::move(video));
                            }
                        }
                        else if (msgContent.type == "audio")
                        {
                            msgContent.content = contentItemJson["audio"].get<std::string>();
                            std::string const audioPath = msgContent.content;
                            size_t const dotPos = audioPath.find_last_of('.');
                            std::string const extension = (dotPos != std::string::npos) ? audioPath.substr(dotPos) : "";
                            bool const isRawAudio
                                = (extension == ".wav" || extension == ".mp3" || extension == ".flac");
                            bool const isMelSpectrogram = (extension == ".safetensors");

                            if (isRawAudio)
                            {
                                // Raw audio path: miniaudio decode → PCM. The runtime audio
                                // runner extracts mel internally per its audio/config.json,
                                // mirroring how the visual runner owns image preprocessing.
                                constexpr int32_t kTargetSampleRate = 16000;
                                rt::audioUtils::AudioData audio;
                                if (!rt::audioUtils::loadAudioDataFromFile(audioPath, kTargetSampleRate, audio))
                                {
                                    throw std::runtime_error(format::fmtstr(
                                        "Failed to decode audio file: %s (unsupported container or corrupt bytes)",
                                        audioPath.c_str()));
                                }
                                int64_t const numSamples = audio.pcm->numSamples();
                                audioBuffers.push_back(std::move(audio));
                                LOG_INFO("Decoded audio (PCM): %s (%ld samples @ %d Hz)", audioPath.c_str(),
                                    static_cast<long>(numSamples), kTargetSampleRate);
                            }
                            else if (isMelSpectrogram)
                            {
                                // Pre-computed mel-spectrogram (Gemma4 audio encoder input).
                                rt::audioUtils::AudioData audio;
                                audio.melSpectrogramPath = audioPath;
                                audio.melSpectrogramFormat = "safetensors";
                                audioBuffers.push_back(std::move(audio));
                                LOG_INFO("Mel-spectrogram input: %s", audioPath.c_str());
                            }
                            else
                            {
                                LOG_WARNING(
                                    "Unsupported audio format: %s (CLI accepts .wav / .mp3 / .flac / .safetensors)",
                                    audioPath.c_str());
                            }
                        }
                        else
                        {
                            throw std::runtime_error(
                                format::fmtstr("Content type must be 'text', 'image', 'video', 'audio', but got: %s",
                                    msgContent.type.c_str()));
                        }
                        chatMsg.contents.push_back(msgContent);
                    }
                }
                else
                {
                    throw std::runtime_error("Message content must be a string or an array");
                }

                chatMessages.push_back(chatMsg);
            }

            rt::LLMGenerationRequest::Request request;
            request.samplingSeed = requestItem.contains("sampling_seed") && !requestItem["sampling_seed"].is_null()
                ? std::optional<uint64_t>{requestItem["sampling_seed"].get<uint64_t>()}
                : defaultSamplingSeed;
            if (requestItem.contains("context_cache_lookup_policy")
                && !requestItem["context_cache_lookup_policy"].is_null())
            {
                request.contextCacheLookupPolicy = parseContextCacheLookupPolicy(requestItem);
            }
            request.messages = std::move(chatMessages);
            request.imageBuffers = std::move(imageBuffers);
            request.audioBuffers = std::move(audioBuffers);
            request.logitBias = defaultLogitBias;
            if (requestItem.contains("logit_bias") && !requestItem["logit_bias"].is_null())
            {
                request.logitBias
                    = parseLogitBias(requestItem["logit_bias"], format::fmtstr("requests[%zu].logit_bias", requestIdx));
            }

            // Optional per-request grammar constraint ("guided_decoding": {...}).
            request.guidedDecoding = defaultGuidedDecoding;
            if (requestItem.contains("guided_decoding") && !requestItem["guided_decoding"].is_null())
            {
                request.guidedDecoding = parseGuidedDecoding(
                    requestItem["guided_decoding"], format::fmtstr("requests[%zu].guided_decoding", requestIdx));
            }

            // Optional per-request stop strings ("stop": string | string[]).
            if (requestItem.contains("stop") && !requestItem["stop"].is_null())
            {
                auto const& stopField = requestItem["stop"];
                if (stopField.is_string())
                {
                    request.stopStrings.push_back(stopField.get<std::string>());
                }
                else if (stopField.is_array())
                {
                    for (auto const& s : stopField)
                    {
                        check::check(s.is_string(), "Each entry in 'stop' must be a string");
                        request.stopStrings.push_back(s.get<std::string>());
                    }
                }
                else
                {
                    throw std::runtime_error("'stop' must be a string or array of strings");
                }
            }

            batchRequest.requests.push_back(std::move(request));
        }

        if (!batchLoraWeightsName.empty())
        {
            batchRequest.loraWeightsName = batchLoraWeightsName;
        }

        batchedRequests.push_back(std::move(batchRequest));
    }

    return {std::move(loraWeightsMap), std::move(batchedRequests)};
}

} // namespace exampleUtils
} // namespace trt_edgellm
