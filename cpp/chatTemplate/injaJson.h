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

#include <initializer_list>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace trt_edgellm::chat_template
{

//! JSON representation used while rendering a provider chat template.
//!
//! Pantor Inja retains pointers to object values while adding render-local
//! variables. Its data model therefore requires the stable references provided
//! by map-backed JSON; vector-backed ordered_json is not compatible.
using InjaJson = nlohmann::json;

//! Construct an Inja-compatible object from the supplied fields.
InjaJson makeInjaObject(std::initializer_list<std::pair<std::string, InjaJson>> fields = {});

//! Insert or replace an object field without invalidating unrelated values.
void setInjaObjectField(InjaJson& object, std::string key, InjaJson value);

//! Convert ordered checkpoint metadata while retaining its observable key order.
InjaJson preserveJsonOrder(nlohmann::ordered_json const& value);

//! Return object keys in their provider-visible order.
std::vector<std::string> injaObjectKeys(InjaJson const& value);

//! Return the size visible to Python-compatible Jinja helpers.
size_t injaVisibleSize(InjaJson const& value);

//! Format a JSON value using Python's repr conventions used by provider Jinja.
std::string injaPythonRepr(InjaJson const& value);

//! Format a JSON value using Python's string conventions used by provider Jinja.
std::string injaPythonString(InjaJson const& value);

} // namespace trt_edgellm::chat_template
