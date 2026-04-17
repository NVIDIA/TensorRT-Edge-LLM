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

#include "common/checkMacros.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/imageUtils.h"
#include "tokenizer/tokenizer.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <random>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace server
{

using Json = nlohmann::json;

// Base64 decode function
inline std::vector<unsigned char> base64Decode(const std::string& encoded)
{
    static const int decodeTable[256] = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1,
        -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1,
        -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    };

    std::vector<unsigned char> decoded;
    decoded.reserve(encoded.size() * 3 / 4);

    int val = 0;
    int valb = -8;

    for (unsigned char c : encoded)
    {
        if (decodeTable[c] == -1)
        {
            break;  // Stop at padding or invalid character
        }
        val = (val << 6) + decodeTable[c];
        valb += 6;
        if (valb >= 0)
        {
            decoded.push_back(static_cast<unsigned char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }

    return decoded;
}

// Parse data URL: data:image/{format};base64,{data}
// Returns decoded image data, or empty vector if not a data URL
inline std::vector<unsigned char> parseDataUrl(const std::string& url, std::string& format)
{
    const std::string dataPrefix = "data:image/";
    if (url.substr(0, dataPrefix.size()) != dataPrefix)
    {
        return {};  // Not a data URL
    }

    // Find format (e.g., jpeg, png, webp)
    size_t formatEnd = url.find(';', dataPrefix.size());
    if (formatEnd == std::string::npos)
    {
        return {};
    }

    format = url.substr(dataPrefix.size(), formatEnd - dataPrefix.size());

    // Check for base64 encoding
    size_t base64Start = url.find("base64,", formatEnd);
    if (base64Start == std::string::npos)
    {
        return {};
    }

    size_t dataStart = base64Start + 7;  // Length of "base64,"
    std::string base64Data = url.substr(dataStart);

    // Remove any whitespace or URL encoding artifacts
    base64Data.erase(base64Data.find_last_not_of(" \t\n\r\f\v") + 1);

    return base64Decode(base64Data);
}

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

                    // Try to parse as data URL (base64 encoded image)
                    std::string imageFormat;
                    auto decodedData = parseDataUrl(url, imageFormat);

                    if (!decodedData.empty())
                    {
                        // Base64 data URL - load from decoded memory
                        LOG_INFO("Loading base64 image (format: %s, size: %zu bytes)",
                            imageFormat.c_str(), decodedData.size());
                        msgContent.content = "base64_image:" + imageFormat;

                        auto image = rt::imageUtils::loadImageFromMemory(
                            decodedData.data(), decodedData.size());
                        if (image.buffer != nullptr)
                        {
                            LOG_INFO("Base64 image loaded successfully (width: %lld, height: %lld)",
                                image.width, image.height);
                            imageBuffers.push_back(std::move(image));
                        }
                        else
                        {
                            LOG_WARNING("Failed to load base64 image");
                        }
                    }
                    else
                    {
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

                        // Load image from file
                        auto image = rt::imageUtils::loadImageFromFile(imagePath);
                        if (image.buffer != nullptr)
                        {
                            imageBuffers.push_back(std::move(image));
                        }
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
        // Without tokenizer, estimate based on character count (~4 chars per token)
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
        std::string outputText = response.outputTexts.empty() ? "" : response.outputTexts[0];

        // Rough estimate: ~4 characters per token
        int64_t promptTokens = static_cast<int64_t>(inputText.size() / 4) + 1;
        int64_t completionTokens = static_cast<int64_t>(outputText.size() / 4) + 1;

        usage["prompt_tokens"] = promptTokens;
        usage["completion_tokens"] = completionTokens;
        usage["total_tokens"] = promptTokens + completionTokens;
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
