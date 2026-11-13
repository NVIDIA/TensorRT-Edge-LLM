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

#include <array>
#include <cstdint>
#include <string_view>

namespace trt_edgellm::safety
{

enum class SafetyTensorDataType
{
    kFLOAT,
    kHALF,
    kINT64,
    kINT32,
    kINT8,
    kUINT8,
    kBOOL
};

enum class SafetyTensorMemoryPlacement
{
    kGPU,
    kCPU,
    kCPU_PINNED
};

enum class SafetyTensorIOMode
{
    kNONE,
    kINPUT,
    kOUTPUT
};

//! Borrowed fixed-profile IO view. Metadata is immutable after Initialization; data remains writable.
//! Storage and names are borrowed until Deinitialization begins. Outputs require successful execution.
struct SafetyTensorView
{
    std::string_view name;
    void* data{nullptr};
    SafetyTensorDataType dataType{SafetyTensorDataType::kFLOAT};
    SafetyTensorMemoryPlacement memPlacement{SafetyTensorMemoryPlacement::kGPU};
    SafetyTensorIOMode ioMode{SafetyTensorIOMode::kNONE};
    int32_t logicalRank{-1};
    std::array<int64_t, 8> logicalShape{};
    int32_t physicalRank{-1};
    std::array<int64_t, 9> physicalShape{};
    int32_t strideRank{-1};
    std::array<int64_t, 9> strides{}; //!< Strides count scalar components, not bytes.
    int32_t strideOrderRank{-1};
    std::array<int64_t, 9> strideOrder{};
    uint64_t bytesPerComponent{0};
    uint64_t componentsPerVector{1};
    int64_t vectorizedDim{-1};
    uint64_t profileBytes{0};
    uint64_t capacityBytes{0};
};

} // namespace trt_edgellm::safety
