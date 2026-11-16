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

#include <cstdint>
#include <string>

namespace trt_edgellm
{
namespace rt
{
enum class PleResidency
{
    kRESIDENT,
    kCACHED
};

//! Runtime storage choice; both policies consume the same pool/map engine.
struct PleRuntimeConfig
{
    PleResidency residency{PleResidency::kRESIDENT};
    int32_t cacheSlots{0};  //!< Includes the zero slot; required for cached residency.
    std::string filePath{}; //!< Explicit PLE data file; empty selects the checkpoint source.
};
} // namespace rt
} // namespace trt_edgellm
