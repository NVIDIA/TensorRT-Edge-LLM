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

#include "runtime/multiStage/stage.h"

#include "common/checkMacros.h"

#include <string>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace multiStage
{

char const* toString(StageStatus v)
{
    switch (v)
    {
    case StageStatus::kMore: return "more";
    case StageStatus::kDone: return "done";
    case StageStatus::kBlocked: return "blocked";
    case StageStatus::kFailed: return "failed";
    }
    return "unknown";
}

char const* toString(FinishReason v)
{
    switch (v)
    {
    case FinishReason::kCompleted: return "completed";
    case FinishReason::kCancelled: return "cancelled";
    case FinishReason::kFailed: return "failed";
    }
    return "unknown";
}

void PortBindings::bind(std::string const& name, PortValue value)
{
    mValues.insert_or_assign(name, std::move(value));
}

PortValue const* PortBindings::find(std::string const& name) const
{
    auto const it = mValues.find(name);
    return it == mValues.end() ? nullptr : &it->second;
}

PortValue const& PortBindings::at(std::string const& name) const
{
    auto const it = mValues.find(name);
    ELLM_CHECK(it != mValues.end(), "input '" + name + "' is not bound; use find() if it may be skipped");
    return it->second;
}

void PortBindings::clear()
{
    mValues.clear();
}

nlohmann::json jsonInput(PortBindings const* in, std::string const& name)
{
    auto const* value = in != nullptr ? in->find(name) : nullptr;
    if (value == nullptr || value->data == nullptr || value->shape.empty())
    {
        return nlohmann::json::object();
    }
    ELLM_CHECK(value->elementBytes == 1, "input '" + name + "' must be UTF-8 bytes");
    std::string const text(static_cast<char const*>(value->data), static_cast<size_t>(value->shape.back()));
    auto parsed = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
    ELLM_CHECK(!parsed.is_discarded() && parsed.is_object(), "input '" + name + "' is not a JSON object");
    return parsed;
}

std::vector<std::string> PortBindings::names() const
{
    std::vector<std::string> out;
    out.reserve(mValues.size());
    for (auto const& entry : mValues)
    {
        out.push_back(entry.first);
    }
    return out;
}

} // namespace multiStage
} // namespace rt
} // namespace trt_edgellm
