/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#pragma once

#include "runtime/llmRuntimeUtils.h"
#include "runtime/imageUtils.h"
#include "tokenizer/tokenizer.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <random>
#include <string>

namespace trt_edgellm
{
namespace server
{

using Json = nlohmann::json;

// Generate unique completion ID (chatcmpl-xxxxxxxx)
inline std::string generateCompletionId()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static const char* hexChars = "0123456789abcdef";

    std::string id = "chatcmpl-";
    for (int i = 0; i < 8; ++i)
    {
        id += hexChars[dis(gen)];
    }
    return id;
}

// Get current Unix timestamp
inline int64_t getCurrentTimestamp()
{
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// Parse OpenAI request JSON to LLMGenerationRequest
// Returns true on success, false on failure with errorMessage set
inline bool parseOpenAIRequest(const Json& requestJson,
                               rt::LLMGenerationRequest& request,
                               std::string& errorMessage)
{
    // Check required field: messages
    if (!requestJson.contains("messages") || !requestJson["messages"].is_array())
    {
        errorMessage = "Missing required field: messages";
        return false;
    }

    const auto& messagesArray = requestJson["messages"];

    // Parse messages into internal format
    std::vector<rt::Message> chatMessages;
    std::vector<rt::imageUtils::ImageData> imageBuffers;

    for (const auto& messageJson : messagesArray)
    {
        if (!messageJson.contains("role"))
        {
            errorMessage = "Each message must have a 'role' field";
            return false;
        }

        rt::Message chatMsg;
        chatMsg.role = messageJson["role"].get<std::string>();

        if (!messageJson.contains("content"))
        {
            errorMessage = "Each message must have a 'content' field";
            return false;
        }

        const auto& contentJson = messageJson["content"];

        // Support both string and array content formats
        if (contentJson.is_string())
        {
            // Simple string format
            rt::Message::MessageContent msgContent;
            msgContent.type = "text";
            msgContent.content = contentJson.get<std::string>();
            chatMsg.contents.push_back(msgContent);
        }
        else if (contentJson.is_array())
        {
            // Multimodal array format
            for (const auto& contentItem : contentJson)
            {
                if (!contentItem.contains("type"))
                {
                    errorMessage = "Each content item must have a 'type' field";
                    return false;
                }

                rt::Message::MessageContent msgContent;
                msgContent.type = contentItem["type"].get<std::string>();

                if (msgContent.type == "text")
                {
                    if (!contentItem.contains("text"))
                    {
                        errorMessage = "Text content item must have 'text' field";
                        return false;
                    }
                    msgContent.content = contentItem["text"].get<std::string>();
                }
                else if (msgContent.type == "image_url")
                {
                    if (!contentItem.contains("image_url") || !contentItem["image_url"].is_object())
                    {
                        errorMessage = "image_url content must have 'image_url' object";
                        return false;
                    }
                    const auto& imageUrl = contentItem["image_url"];
                    if (!imageUrl.contains("url"))
                    {
                        errorMessage = "image_url must have 'url' field";
                        return false;
                    }

                    std::string url = imageUrl["url"].get<std::string>();

                    // Extract file path from URL (support file:// scheme or direct path)
                    std::string imagePath;
                    if (url.substr(0, 7) == "file://")
                    {
                        imagePath = url.substr(7);
                    }
                    else
                    {
                        // Assume it's a direct file path
                        imagePath = url;
                    }

                    msgContent.content = imagePath;

                    // Load image
                    auto image = rt::imageUtils::loadImageFromFile(imagePath);
                    if (image.buffer != nullptr)
                    {
                        imageBuffers.push_back(std::move(image));
                    }
                }
                else
                {
                    errorMessage = "Unsupported content type: " + msgContent.type;
                    return false;
                }

                chatMsg.contents.push_back(msgContent);
            }
        }
        else
        {
            errorMessage = "Message content must be a string or array";
            return false;
        }

        chatMessages.push_back(chatMsg);
    }

    // Create single request (no batch support)
    rt::LLMGenerationRequest::Request req;
    req.messages = std::move(chatMessages);
    req.imageBuffers = std::move(imageBuffers);
    request.requests.push_back(std::move(req));

    // Set generation parameters with defaults
    request.temperature = requestJson.value("temperature", 1.0f);
    request.topP = requestJson.value("top_p", 0.8f);
    request.topK = 50;  // OpenAI doesn't have top_k, use internal default
    request.maxGenerateLength = requestJson.value("max_tokens", 256);

    // Apply chat template by default
    request.applyChatTemplate = true;
    request.addGenerationPrompt = true;

    return true;
}

// Format LLMGenerationResponse to OpenAI response JSON
inline Json formatOpenAIResponse(const rt::LLMGenerationResponse& response,
                                 const std::string& modelId,
                                 const rt::LLMGenerationRequest& request,
                                 tokenizer::Tokenizer* tokenizer)
{
    Json responseJson;

    responseJson["id"] = generateCompletionId();
    responseJson["object"] = "chat.completion";
    responseJson["created"] = getCurrentTimestamp();
    responseJson["model"] = modelId;

    // Single choice (no streaming, single response)
    Json choice;
    choice["index"] = 0;

    Json message;
    message["role"] = "assistant";
    message["content"] = response.outputTexts.empty() ? "" : response.outputTexts[0];
    choice["message"] = message;
    choice["finish_reason"] = "stop";

    responseJson["choices"] = Json::array({choice});

    // Token usage (estimated if tokenizer available)
    Json usage;
    if (tokenizer != nullptr && !request.requests.empty())
    {
        // Estimate prompt tokens from input text
        std::string inputText;
        for (const auto& msg : request.requests[0].messages)
        {
            for (const auto& content : msg.contents)
            {
                if (content.type == "text")
                {
                    inputText += content.content + " ";
                }
            }
        }
        auto inputTokens = tokenizer->encode(inputText);
        usage["prompt_tokens"] = static_cast<int64_t>(inputTokens.size());

        // Estimate completion tokens from output
        auto outputTokens = tokenizer->encode(response.outputTexts.empty() ? "" : response.outputTexts[0]);
        usage["completion_tokens"] = static_cast<int64_t>(outputTokens.size());
        usage["total_tokens"] = usage["prompt_tokens"].get<int64_t>() + usage["completion_tokens"].get<int64_t>();
    }
    else
    {
        // Without tokenizer, return null
        usage = nullptr;
    }
    responseJson["usage"] = usage;

    return responseJson;
}

// Format error response in OpenAI API style
inline Json formatErrorResponse(const std::string& message, const std::string& type)
{
    Json errorJson;
    errorJson["error"]["message"] = message;
    errorJson["error"]["type"] = type;
    return errorJson;
}

} // namespace server
} // namespace trt_edgellm
