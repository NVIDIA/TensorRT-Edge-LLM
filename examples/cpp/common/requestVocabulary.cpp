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

#include "requestVocabulary.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace trt_edgellm
{
namespace exampleUtils
{

namespace
{

template <typename T>
void appendMissing(std::vector<T>& out, std::vector<T> const& values)
{
    for (auto const& value : values)
    {
        if (std::find(out.begin(), out.end(), value) == out.end())
        {
            out.push_back(value);
        }
    }
}

bool contains(std::vector<std::string> const& values, std::string const& value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

//! The sampling key @p spelling stands for, or nullptr when it is no alias of one.
std::string const* samplingAliasTarget(RequestVocabulary const& vocabulary, std::string const& spelling)
{
    for (auto const& [alias, name] : vocabulary.aliases)
    {
        if (alias == spelling && contains(vocabulary.samplingKeys, name))
        {
            return &name;
        }
    }
    return nullptr;
}

//! A sampling layer given as an object or as JSON text. Throws, naming @p origin, when it is neither.
nlohmann::json samplingLayer(nlohmann::json const& raw, char const* origin)
{
    auto layer
        = raw.is_string() ? nlohmann::json::parse(raw.get<std::string>(), nullptr, /*allow_exceptions=*/false) : raw;
    if (!layer.is_object())
    {
        auto const text = raw.is_string() ? raw.get<std::string>() : raw.dump();
        throw std::runtime_error(std::string{origin} + " '" + text + "' is not a JSON object");
    }
    return layer;
}

} // namespace

RequestVocabulary vocabularyOf(rt::multiStage::StageRegistry const& runners)
{
    RequestVocabulary vocabulary;
    for (auto const& name : runners.names())
    {
        auto const& decl = runners.find(name)->decl;
        appendMissing(vocabulary.samplingKeys, decl.samplingKeys);
        appendMissing(vocabulary.styleFields, decl.styleFields);
        appendMissing(vocabulary.aliases, decl.requestAliases);
    }
    return vocabulary;
}

nlohmann::json foldSampling(RequestVocabulary const& vocabulary, std::string const& cliSampling,
    nlohmann::json const& document, nlohmann::json const& entry)
{
    nlohmann::json merged = nlohmann::json::object();
    auto const absorb = [&merged, &vocabulary](nlohmann::json const& layer) {
        for (auto const& [key, value] : layer.items())
        {
            auto const* target = samplingAliasTarget(vocabulary, key);
            if (target == nullptr)
            {
                merged[key] = value;
            }
            else if (layer.contains(*target))
            {
                std::cerr << "sampling key '" << key << "' ignored; '" << *target << "' is also set\n";
            }
            else
            {
                std::cerr << "sampling key '" << key << "' read as '" << *target << "'\n";
                merged[*target] = value;
            }
        }
    };

    if (!cliSampling.empty())
    {
        absorb(samplingLayer(cliSampling, "--sampling"));
    }
    nlohmann::json documentLayer = nlohmann::json::object();
    for (auto const& key : vocabulary.samplingKeys)
    {
        if (document.contains(key))
        {
            documentLayer[key] = document.at(key);
        }
    }
    for (auto const& [alias, name] : vocabulary.aliases)
    {
        if (contains(vocabulary.samplingKeys, name) && document.contains(alias))
        {
            documentLayer[alias] = document.at(alias);
        }
    }
    absorb(documentLayer);
    if (entry.contains("sampling"))
    {
        absorb(samplingLayer(entry.at("sampling"), "request sampling"));
    }

    for (auto it = merged.begin(); it != merged.end();)
    {
        if (!contains(vocabulary.samplingKeys, it.key()))
        {
            std::cerr << "sampling key '" << it.key() << "' is read by no stage; ignored\n";
            it = merged.erase(it);
            continue;
        }
        ++it;
    }
    return merged;
}

nlohmann::json const* findRequestField(RequestVocabulary const& vocabulary, std::string const& name,
    nlohmann::json const& document, nlohmann::json const& entry)
{
    for (auto const* level : {&entry, &document})
    {
        if (!level->is_object())
        {
            continue;
        }
        if (auto const it = level->find(name); it != level->end())
        {
            return &*it;
        }
        for (auto const& [alias, target] : vocabulary.aliases)
        {
            if (auto const it = level->find(alias); target == name && it != level->end())
            {
                return &*it;
            }
        }
    }
    return nullptr;
}

nlohmann::json foldStyle(
    RequestVocabulary const& vocabulary, nlohmann::json const& document, nlohmann::json const& entry)
{
    nlohmann::json style = nlohmann::json::object();
    if (entry.contains("style"))
    {
        auto const& raw = entry.at("style");
        style = raw.is_string() ? nlohmann::json::parse(raw.get<std::string>(), nullptr, false) : raw;
        if (!style.is_object())
        {
            style = nlohmann::json::object();
        }
    }
    for (auto const& field : vocabulary.styleFields)
    {
        if (style.contains(field))
        {
            continue;
        }
        // A style field may be text, such as a language, or a number, such as a video's height.
        auto const* value
            = entry.contains(field) ? &entry.at(field) : (document.contains(field) ? &document.at(field) : nullptr);
        if (value != nullptr && !value->is_null()
            && !(value->is_string() && value->get_ref<std::string const&>().empty()))
        {
            style[field] = *value;
        }
    }
    return style;
}

} // namespace exampleUtils
} // namespace trt_edgellm
