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

#include "safety_inference/runtime/safetyStatus.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace trt_edgellm::safety
{

//! Stage identifiers refer to immutable model-definition storage.
using LMStageId = std::string_view;

struct LMStageContext;

//! Runner invokes preparation after selecting its initialized context and before graph execution.
using LMStagePrepareCallback = SafetyStatus (*)(LMStageContext const& context, void* userData) noexcept;

//! Borrowed Runner output; the prepared context outlives request processing.
struct LMStageOutputs
{
    LMStageContext const* context{nullptr};
};

struct TokenIdView
{
    int32_t const* data{nullptr};
    std::size_t size{0};
};

//! Storage is prepared during Initialization and outlives the request strategy.
struct TokenIdBuffer
{
    int32_t* data{nullptr};
    std::size_t capacity{0};
};

enum class FinishReason
{
    kNotFinished,
    kEndId,
    kLength,
    kError,
};

} // namespace trt_edgellm::safety
