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

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm::safety
{

struct Message
{
    struct MessageContent
    {
        std::string type; //!< "text" or "image"; image bytes are supplied through imageBuffers.
        std::string content;
    };
    std::string role;
    std::vector<MessageContent> contents;
};

//! Borrowed, tightly packed RGB UINT8 host pixels; storage outlives handleRequest().
struct ImageData
{
    uint8_t const* data{nullptr};
    std::size_t sizeInBytes{0};
    int32_t width{0};
    int32_t height{0};
};

//! Caller prepares input containers before handleRequest(); the runtime only reads them.
//! The initial InternVL recipe admits one request with one image and uses greedy decoding.
struct LLMGenerationRequest
{
    struct Request
    {
        std::vector<Message> messages;
        std::vector<ImageData> imageBuffers;
    };
    std::vector<Request> requests;
    int64_t maxGenerateLength{0};
};

//! Caller sizes the outer vectors to one and reserves token/text capacity before handleRequest().
//! Operational processing writes within that capacity; outputs are valid only on success.
struct LLMGenerationResponse
{
    std::vector<std::vector<int32_t>> outputIds;
    std::vector<std::string> outputTexts;
};

} // namespace trt_edgellm::safety
