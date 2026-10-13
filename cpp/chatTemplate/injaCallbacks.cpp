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

#include "injaCallbacks.h"

#include "injaJson.h"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trt_edgellm::chat_template
{
namespace
{
using Arguments = inja::Arguments;
using Json = InjaJson;

std::string pythonString(Json const& value)
{
    return injaPythonString(value);
}

std::string strip(std::string value, bool left, bool right, std::string const& characters = {})
{
    auto const keep = [&](unsigned char ch) {
        return characters.empty() ? std::isspace(ch) == 0 : characters.find(static_cast<char>(ch)) == std::string::npos;
    };
    if (left)
    {
        value.erase(value.begin(), std::find_if(value.begin(), value.end(), keep));
    }
    if (right)
    {
        value.erase(std::find_if(value.rbegin(), value.rend(), keep).base(), value.end());
    }
    return value;
}

Json splitString(std::string const& value, std::string const& delimiter, int64_t maxSplits = -1)
{
    if (delimiter.empty())
    {
        throw std::runtime_error("split delimiter cannot be empty");
    }
    Json result = Json::array();
    size_t begin = 0;
    int64_t splits = 0;
    while (maxSplits < 0 || splits < maxSplits)
    {
        auto const end = value.find(delimiter, begin);
        if (end == std::string::npos)
        {
            break;
        }
        result.push_back(value.substr(begin, end - begin));
        begin = end + delimiter.size();
        ++splits;
    }
    result.push_back(value.substr(begin));
    return result;
}

Json const* findAttribute(Json const& value, std::string const& path)
{
    Json const* current = &value;
    size_t begin = 0;
    while (begin <= path.size())
    {
        auto const end = path.find('.', begin);
        auto const name = path.substr(begin, end - begin);
        if (!current->is_object() || !current->contains(name))
        {
            return nullptr;
        }
        current = &current->at(name);
        if (end == std::string::npos)
        {
            break;
        }
        begin = end + 1;
    }
    return current;
}

bool isTruthy(Json const* value)
{
    if (value == nullptr || value->is_null())
    {
        return false;
    }
    if (value->is_boolean())
    {
        return value->get<bool>();
    }
    if (value->is_number())
    {
        return *value != Json(0);
    }
    if (value->is_object())
    {
        return injaVisibleSize(*value) != 0;
    }
    if (value->is_string() || value->is_array())
    {
        return !value->empty();
    }
    return true;
}

bool attributeMatches(Json const* value, std::string const& test, Json const* expected)
{
    if (test.empty())
    {
        return isTruthy(value);
    }
    if (test == "defined")
    {
        return value != nullptr;
    }
    if (test == "undefined")
    {
        return value == nullptr;
    }
    if (value == nullptr)
    {
        return false;
    }
    if (test == "equalto" || test == "eq")
    {
        if (expected == nullptr)
        {
            throw std::runtime_error(test + " requires a comparison value");
        }
        return *value == *expected;
    }
    if (test == "none")
    {
        return value->is_null();
    }
    if (test == "string")
    {
        return value->is_string();
    }
    if (test == "mapping")
    {
        return value->is_object();
    }
    if (test == "sequence" || test == "iterable")
    {
        return value->is_array() || value->is_object() || value->is_string();
    }
    if (test == "boolean")
    {
        return value->is_boolean();
    }
    if (test == "number")
    {
        return value->is_number();
    }
    throw std::runtime_error("unsupported attribute test: " + test);
}

Json filterByAttribute(Arguments& arguments, bool reject)
{
    if (arguments.size() < 2 || arguments.size() > 4 || !arguments[1]->is_string())
    {
        throw std::runtime_error("expected a sequence, attribute, optional test, and optional value");
    }
    if (arguments[0]->is_null())
    {
        return Json::array();
    }
    if (!arguments[0]->is_array())
    {
        throw std::runtime_error("expected a sequence, attribute, optional test, and optional value");
    }
    auto const attribute = arguments[1]->get<std::string>();
    auto const test = arguments.size() >= 3 ? arguments[2]->get<std::string>() : "";
    auto const expected = arguments.size() == 4 ? arguments[3] : nullptr;
    Json result = Json::array();
    for (auto const& item : *arguments[0])
    {
        auto const matches = attributeMatches(findAttribute(item, attribute), test, expected);
        if (matches != reject)
        {
            result.push_back(item);
        }
    }
    return result;
}

Json makeNamespace(Arguments& arguments)
{
    if (arguments.size() % 2 != 0)
    {
        throw std::runtime_error("namespace expects name/value pairs");
    }
    Json result = makeInjaObject();
    for (size_t index = 0; index < arguments.size(); index += 2)
    {
        if (!arguments[index]->is_string())
        {
            throw std::runtime_error("namespace field names must be strings");
        }
        setInjaObjectField(result, arguments[index]->get<std::string>(), *arguments[index + 1]);
    }
    return result;
}

std::string jsonString(Json const& value, std::string_view itemSeparator = ", ", std::string_view keySeparator = ": ",
    bool ensureAscii = false)
{
    if (value.is_array())
    {
        std::string result{"["};
        for (size_t index = 0; index < value.size(); ++index)
        {
            if (index != 0)
            {
                result += itemSeparator;
            }
            result += jsonString(value[index], itemSeparator, keySeparator, ensureAscii);
        }
        return result + "]";
    }
    if (value.is_object())
    {
        std::string result{"{"};
        bool first = true;
        for (auto const& key : injaObjectKeys(value))
        {
            if (!first)
            {
                result += itemSeparator;
            }
            result += Json(key).dump(-1, ' ', ensureAscii) + std::string(keySeparator)
                + jsonString(value.at(key), itemSeparator, keySeparator, ensureAscii);
            first = false;
        }
        return result + "}";
    }
    return value.dump(-1, ' ', ensureAscii);
}

nlohmann::ordered_json orderedJson(Json const& value)
{
    if (value.is_object())
    {
        nlohmann::ordered_json result = nlohmann::ordered_json::object();
        for (auto const& key : injaObjectKeys(value))
        {
            result[key] = orderedJson(value.at(key));
        }
        return result;
    }
    if (value.is_array())
    {
        nlohmann::ordered_json result = nlohmann::ordered_json::array();
        for (auto const& item : value)
        {
            result.push_back(orderedJson(item));
        }
        return result;
    }
    return nlohmann::ordered_json::parse(value.dump());
}

inja::CallbackFunction checked(std::string name, inja::CallbackFunction callback)
{
    return [name = std::move(name), callback = std::move(callback)](Arguments& arguments) {
        try
        {
            return callback(arguments);
        }
        catch (std::exception const& error)
        {
            throw std::runtime_error(name + ": " + error.what());
        }
    };
}

void add(inja::Environment& environment, std::string const& name, int count, inja::CallbackFunction callback)
{
    environment.add_callback(name, count, checked(name, std::move(callback)));
}

void addVariadic(inja::Environment& environment, std::string const& name, inja::CallbackFunction callback)
{
    environment.add_callback(name, checked(name, std::move(callback)));
}
} // namespace

void registerInjaCallbacks(inja::Environment& environment)
{
    add(environment, "tojson", 1, [](Arguments& arguments) { return jsonString(*arguments[0]); });
    add(environment, "tojson", 2, [](Arguments& arguments) {
        if (arguments[1]->is_boolean())
        {
            return jsonString(*arguments[0], ", ", ": ", arguments[1]->get<bool>());
        }
        if (arguments[1]->is_number_integer())
        {
            return orderedJson(*arguments[0]).dump(arguments[1]->get<int>(), ' ', false);
        }
        if (arguments[1]->is_array() && arguments[1]->size() == 2 && (*arguments[1])[0].is_string()
            && (*arguments[1])[1].is_string())
        {
            return jsonString(
                *arguments[0], (*arguments[1])[0].get<std::string>(), (*arguments[1])[1].get<std::string>());
        }
        throw std::runtime_error("unsupported keyword argument");
    });
    add(environment, "safe", 1, [](Arguments& arguments) { return *arguments[0]; });
    add(environment, "string", 1, [](Arguments& arguments) { return pythonString(*arguments[0]); });
    add(environment, "trim", 1, [](Arguments& arguments) { return strip(pythonString(*arguments[0]), true, true); });
    add(environment, "lstrip", 1, [](Arguments& arguments) { return strip(pythonString(*arguments[0]), true, false); });
    add(environment, "rstrip", 1, [](Arguments& arguments) { return strip(pythonString(*arguments[0]), false, true); });
    add(environment, "strip", 1, [](Arguments& arguments) { return strip(pythonString(*arguments[0]), true, true); });
    add(environment, "strip", 2, [](Arguments& arguments) {
        return strip(pythonString(*arguments[0]), true, true, arguments[1]->get<std::string>());
    });
    add(environment, "lstrip", 2, [](Arguments& arguments) {
        return strip(pythonString(*arguments[0]), true, false, arguments[1]->get<std::string>());
    });
    add(environment, "rstrip", 2, [](Arguments& arguments) {
        return strip(pythonString(*arguments[0]), false, true, arguments[1]->get<std::string>());
    });
    add(environment, "startswith", 2, [](Arguments& arguments) {
        auto const value = pythonString(*arguments[0]);
        auto const prefix = arguments[1]->get<std::string>();
        return value.rfind(prefix, 0) == 0;
    });
    add(environment, "endswith", 2, [](Arguments& arguments) {
        auto const value = pythonString(*arguments[0]);
        auto const suffix = arguments[1]->get<std::string>();
        return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    });
    add(environment, "split", 2, [](Arguments& arguments) {
        return splitString(pythonString(*arguments[0]), arguments[1]->get<std::string>());
    });
    add(environment, "split", 3, [](Arguments& arguments) {
        return splitString(pythonString(*arguments[0]), arguments[1]->get<std::string>(), arguments[2]->get<int64_t>());
    });
    add(environment, "get", 2, [](Arguments& arguments) {
        if (!arguments[0]->is_object())
        {
            return Json(nullptr);
        }
        auto const value = arguments[0]->find(arguments[1]->get<std::string>());
        return value == arguments[0]->end() ? Json(nullptr) : *value;
    });
    add(environment, "get", 3, [](Arguments& arguments) {
        if (!arguments[0]->is_object())
        {
            return *arguments[2];
        }
        auto const value = arguments[0]->find(arguments[1]->get<std::string>());
        return value == arguments[0]->end() ? *arguments[2] : *value;
    });
    add(environment, "items", 1, [](Arguments& arguments) {
        Json result = Json::array();
        for (auto const& key : injaObjectKeys(*arguments[0]))
        {
            result.push_back(Json::array({key, arguments[0]->at(key)}));
        }
        return result;
    });
    add(environment, "list", 1, [](Arguments& arguments) {
        if (arguments[0]->is_array())
        {
            return *arguments[0];
        }
        Json result = Json::array();
        if (arguments[0]->is_object())
        {
            for (auto const& key : injaObjectKeys(*arguments[0]))
            {
                result.push_back(key);
            }
        }
        return result;
    });
    add(environment, "fromjson", 1, [](Arguments& arguments) {
        return preserveJsonOrder(nlohmann::ordered_json::parse(arguments[0]->get<std::string>()));
    });
    add(environment, "reject", 3, [](Arguments& arguments) {
        if (!arguments[0]->is_array() || arguments[1]->get<std::string>() != "equalto")
        {
            throw std::runtime_error("reject currently supports equalto on arrays");
        }
        Json result = Json::array();
        for (auto const& item : *arguments[0])
        {
            if (item != *arguments[2])
            {
                result.push_back(item);
            }
        }
        return result;
    });
    add(environment, "dictsort", 1, [](Arguments& arguments) {
        if (!arguments[0]->is_object())
        {
            throw std::runtime_error("dictsort expects an object");
        }
        auto keys = injaObjectKeys(*arguments[0]);
        std::sort(keys.begin(), keys.end());
        Json result = Json::array();
        for (auto const& key : keys)
        {
            result.push_back(Json::array({key, arguments[0]->at(key)}));
        }
        return result;
    });
    add(environment, "sort", 2, [](Arguments& arguments) {
        if (!arguments[0]->is_array() || !arguments[1]->is_string())
        {
            throw std::runtime_error("sort(attribute=...) expects an array and attribute name");
        }
        auto result = *arguments[0];
        auto const attribute = arguments[1]->get<std::string>();
        std::stable_sort(result.begin(), result.end(), [&](Json const& lhs, Json const& rhs) {
            if (!lhs.is_object() || !rhs.is_object() || !lhs.contains(attribute) || !rhs.contains(attribute))
            {
                throw std::runtime_error("sort attribute is missing");
            }
            return lhs.at(attribute) < rhs.at(attribute);
        });
        return result;
    });
    add(environment, "map", 2, [](Arguments& arguments) {
        if (!arguments[0]->is_array() || !arguments[1]->is_string())
        {
            throw std::runtime_error("map expects a sequence and filter name");
        }
        auto const filter = arguments[1]->get<std::string>();
        Json result = Json::array();
        for (auto const& item : *arguments[0])
        {
            auto value = pythonString(item);
            if (filter == "upper")
            {
                std::transform(value.begin(), value.end(), value.begin(),
                    [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
            }
            else if (filter == "lower")
            {
                std::transform(value.begin(), value.end(), value.begin(),
                    [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            }
            else if (item.is_object())
            {
                auto const attribute = findAttribute(item, filter);
                if (attribute == nullptr)
                {
                    throw std::runtime_error("map attribute is missing: " + filter);
                }
                result.push_back(*attribute);
                continue;
            }
            else
            {
                throw std::runtime_error("unsupported map filter: " + filter);
            }
            result.push_back(std::move(value));
        }
        return result;
    });
    addVariadic(environment, "selectattr", [](Arguments& arguments) { return filterByAttribute(arguments, false); });
    addVariadic(environment, "rejectattr", [](Arguments& arguments) { return filterByAttribute(arguments, true); });
    add(environment, "unique", 1, [](Arguments& arguments) {
        if (!arguments[0]->is_array())
        {
            throw std::runtime_error("unique expects a sequence");
        }
        Json result = Json::array();
        for (auto const& item : *arguments[0])
        {
            if (std::find(result.begin(), result.end(), item) == result.end())
            {
                result.push_back(item);
            }
        }
        return result;
    });
    addVariadic(environment, "range", [](Arguments& arguments) {
        if (arguments.size() < 2 || arguments.size() > 3)
        {
            throw std::runtime_error("range expects two or three integer arguments");
        }
        auto start = arguments[0]->get<int64_t>();
        auto const stop = arguments[1]->get<int64_t>();
        auto const step = arguments.size() == 3 ? arguments[2]->get<int64_t>() : int64_t{1};
        if (step == 0)
        {
            throw std::runtime_error("range step cannot be zero");
        }
        Json result = Json::array();
        for (auto value = start; step > 0 ? value < stop : value > stop; value += step)
        {
            result.push_back(value);
        }
        return result;
    });
    add(environment, "namespace", 0, makeNamespace);
    addVariadic(environment, "namespace", makeNamespace);
    add(environment, "raise_exception", 1,
        [](Arguments& arguments) -> Json { throw std::runtime_error(pythonString(*arguments[0])); });
    add(environment, "strftime_now", 1, [](Arguments& arguments) {
        auto const format = arguments[0]->get<std::string>();
        auto const now = std::time(nullptr);
        std::tm local{};
        localtime_r(&now, &local);
        std::vector<char> output(256);
        while (std::strftime(output.data(), output.size(), format.c_str(), &local) == 0)
        {
            if (output.size() >= 16384)
            {
                throw std::runtime_error("strftime output exceeds limit");
            }
            output.resize(output.size() * 2);
        }
        return std::string(output.data());
    });
}

} // namespace trt_edgellm::chat_template
