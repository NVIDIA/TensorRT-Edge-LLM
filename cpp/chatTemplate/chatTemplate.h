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

#include "runtime/llmRuntimeUtils.h"

#include <filesystem>
#include <memory>
#include <string>

namespace trt_edgellm::chat_template
{

class ChatTemplate
{
public:
    struct Options
    {
        bool applyTemplate{true};
        bool addGenerationPrompt{true};
        bool enableThinking{false};
        std::string reasoningEffort;
        std::vector<rt::ToolDefinition> tools;
        rt::ToolChoice toolChoice;
        bool parallelToolCalls{true};
    };

    ChatTemplate();
    ~ChatTemplate();
    ChatTemplate(ChatTemplate&&) noexcept;
    ChatTemplate& operator=(ChatTemplate&&) noexcept;
    ChatTemplate(ChatTemplate const&) = delete;
    ChatTemplate& operator=(ChatTemplate const&) = delete;

    /*! Translate the runtime request controls into renderer options. */
    static Options optionsFrom(rt::LLMGenerationRequest const& request);

    /*! Load the provider Jinja templates or an explicit native-renderer marker. */
    bool load(std::filesystem::path const& modelDir, std::string bosToken = {}, std::string eosToken = {});

    /*! Render one structured conversation. */
    bool apply(rt::LLMGenerationRequest::Request const& request,
        rt::LLMGenerationRequest::FormattedRequest& formattedRequest, Options const& options) const;

    bool isLoaded() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace trt_edgellm::chat_template
