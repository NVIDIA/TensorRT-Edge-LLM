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

#include "injaJson.h"

#include <inja/inja.hpp>

namespace trt_edgellm::chat_template
{
namespace
{
std::string const kObjectOrderKey{inja::Renderer::object_order_key};
}

std::vector<std::string> injaObjectKeys(InjaJson const& value)
{
    return value.is_object() ? inja::Renderer::object_keys(value) : std::vector<std::string>{};
}

size_t injaVisibleSize(InjaJson const& value)
{
    if (value.is_object())
    {
        return injaObjectKeys(value).size();
    }
    if (value.is_string())
    {
        return value.get_ref<std::string const&>().size();
    }
    return value.is_array() ? value.size() : 0;
}

std::string injaPythonRepr(InjaJson const& value)
{
    return inja::Renderer::python_repr(value);
}

std::string injaPythonString(InjaJson const& value)
{
    return value.is_string() ? value.get<std::string>() : injaPythonRepr(value);
}

InjaJson makeInjaObject(std::initializer_list<std::pair<std::string, InjaJson>> fields)
{
    InjaJson result = InjaJson::object();
    result[kObjectOrderKey] = InjaJson::array();
    for (auto const& [key, value] : fields)
    {
        setInjaObjectField(result, key, value);
    }
    return result;
}

void setInjaObjectField(InjaJson& object, std::string key, InjaJson value)
{
    if (!object.is_object())
    {
        object = InjaJson::object();
    }
    if (!object.contains(kObjectOrderKey) || !object.at(kObjectOrderKey).is_array())
    {
        InjaJson order = InjaJson::array();
        for (auto const& [existing, item] : object.items())
        {
            (void) item;
            if (existing != kObjectOrderKey)
            {
                order.push_back(existing);
            }
        }
        object[kObjectOrderKey] = std::move(order);
    }
    if (key != kObjectOrderKey && !object.contains(key))
    {
        object[kObjectOrderKey].push_back(key);
    }
    if (key != kObjectOrderKey)
    {
        object[std::move(key)] = std::move(value);
    }
}

InjaJson preserveJsonOrder(nlohmann::ordered_json const& value)
{
    if (value.is_object())
    {
        InjaJson result = makeInjaObject();
        for (auto const& [key, item] : value.items())
        {
            setInjaObjectField(result, key, preserveJsonOrder(item));
        }
        return result;
    }
    if (value.is_array())
    {
        InjaJson result = InjaJson::array();
        for (auto const& item : value)
        {
            result.push_back(preserveJsonOrder(item));
        }
        return result;
    }
    return InjaJson::parse(value.dump());
}

} // namespace trt_edgellm::chat_template
