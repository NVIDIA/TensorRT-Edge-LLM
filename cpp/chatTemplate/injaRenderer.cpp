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

#include "injaRenderer.h"

#include "injaCallbacks.h"

#include "common/inputLimits.h"

#include <algorithm>
#include <cctype>
#include <inja/inja.hpp>
#include <stdexcept>
#include <string_view>

namespace trt_edgellm::chat_template
{
namespace
{
void validateTemplateFile(std::filesystem::path const& path)
{
    if (!std::filesystem::is_regular_file(path))
    {
        throw std::runtime_error("file does not exist: " + path.string());
    }
    if (std::filesystem::file_size(path) > limits::tokenizer::kChatTemplateFileSizeBytes)
    {
        throw std::runtime_error("chat template exceeds size limit: " + path.string());
    }
}

bool detectContentBlocks(std::string_view source)
{
    size_t offset = 0;
    while ((offset = source.find("{%", offset)) != std::string_view::npos)
    {
        auto const end = source.find("%}", offset + 2);
        if (end == std::string_view::npos)
        {
            return false;
        }

        auto statement = source.substr(offset + 2, end - offset - 2);
        auto const isDecoration
            = [](char ch) { return std::isspace(static_cast<unsigned char>(ch)) != 0 || ch == '-' || ch == '+'; };
        while (!statement.empty() && isDecoration(statement.front()))
        {
            statement.remove_prefix(1);
        }
        while (!statement.empty() && isDecoration(statement.back()))
        {
            statement.remove_suffix(1);
        }
        offset = end + 2;

        if (statement.size() < 4 || statement.substr(0, 4) != "for ")
        {
            continue;
        }
        auto const in = statement.find(" in ", 4);
        if (in == std::string_view::npos)
        {
            continue;
        }

        auto iterableView = statement.substr(in + 4);
        auto const filter = iterableView.find(" if ");
        if (filter != std::string_view::npos)
        {
            iterableView = iterableView.substr(0, filter);
        }
        std::string iterable(iterableView);
        iterable.erase(
            std::remove_if(iterable.begin(), iterable.end(), [](unsigned char ch) { return std::isspace(ch) != 0; }),
            iterable.end());
        if (iterable.find(".content") != std::string::npos || iterable.find("['content']") != std::string::npos
            || iterable.find("[\"content\"]") != std::string::npos || iterable == "content"
            || iterable.rfind("content|", 0) == 0)
        {
            return true;
        }
    }
    return false;
}
} // namespace

class InjaRenderer::Impl
{
public:
    void load(std::filesystem::path const& templatePath)
    {
        mLoaded = false;
        validateTemplateFile(templatePath);
        mEnvironment = std::make_unique<inja::Environment>(templatePath.parent_path());
        mEnvironment->set_html_autoescape(false);
        mEnvironment->set_trim_blocks(true);
        mEnvironment->set_lstrip_blocks(true);
        mEnvironment->set_line_statement({});
        mEnvironment->set_search_included_templates_in_files(false);
        registerInjaCallbacks(*mEnvironment);
        mTemplate = mEnvironment->parse_template(templatePath.filename());
        mExpectsContentBlocks = detectContentBlocks(mTemplate.content);
        mSupportsDeveloperRole = mTemplate.content.find("\"developer\"") != std::string::npos
            || mTemplate.content.find("'developer'") != std::string::npos;
        mLoaded = true;
    }

    std::string render(Json const& data) const
    {
        if (!mLoaded)
        {
            throw std::runtime_error("Inja chat template is not loaded");
        }
        return mEnvironment->render(mTemplate, data);
    }

    bool expectsContentBlocks() const noexcept
    {
        return mExpectsContentBlocks;
    }

    bool supportsDeveloperRole() const noexcept
    {
        return mSupportsDeveloperRole;
    }

private:
    std::unique_ptr<inja::Environment> mEnvironment;
    inja::Template mTemplate;
    bool mLoaded{false};
    bool mExpectsContentBlocks{false};
    bool mSupportsDeveloperRole{false};
};

InjaRenderer::InjaRenderer()
    : mImpl(std::make_unique<Impl>())
{
}

InjaRenderer::~InjaRenderer() = default;
InjaRenderer::InjaRenderer(InjaRenderer&&) noexcept = default;
InjaRenderer& InjaRenderer::operator=(InjaRenderer&&) noexcept = default;

void InjaRenderer::load(std::filesystem::path const& templatePath)
{
    mImpl->load(templatePath);
}

std::string InjaRenderer::render(Json const& data) const
{
    return mImpl->render(data);
}

bool InjaRenderer::expectsContentBlocks() const noexcept
{
    return mImpl->expectsContentBlocks();
}

bool InjaRenderer::supportsDeveloperRole() const noexcept
{
    return mImpl->supportsDeveloperRole();
}

} // namespace trt_edgellm::chat_template
