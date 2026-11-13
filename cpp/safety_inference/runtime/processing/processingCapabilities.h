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

#include "safety_inference/runtime/model_inference/modelInferenceTypes.h"
#include "safety_inference/runtime/model_inference/multimodal/vlmRequestContext.h"
#include "safety_inference/runtime/resources/runtimeResource.h"

#include <array>
#include <filesystem>

namespace trt_edgellm::safety::llm
{
struct ModelLocalData;
}

namespace trt_edgellm::safety
{

enum class CapabilityPreparationState
{
    kEMPTY,
    kDECLARED,
    kREADY,
    kCLOSED,
};

struct TextView
{
    char const* data{nullptr};
    std::size_t size{0};
};

struct MutableTextBuffer
{
    char* data{nullptr};
    std::size_t capacity{0};
};

struct ProcessingLimits
{
    std::size_t maxPromptBytes{0};
    std::size_t maxPromptTokens{0};
    std::size_t maxResponseBytes{0};
    std::size_t maxGeneratedTokens{0};
    std::size_t maxContextTokens{0};
    int32_t vocabularySize{0};
    int32_t hiddenSize{0};
    float ropeTheta{1000000.0F};
    float ropeScale{1.0F};
};

struct ImageProcessingLimits
{
    int32_t maxImageWidth{0};
    int32_t maxImageHeight{0};
    int32_t minBlocks{1};
    int32_t maxMainBlocks{0};
    int32_t visualProfileBlocks{0};
    int32_t tokensPerTile{0};
    std::size_t maxImageTokens{0};
    bool addThumbnail{true};
    float normalizationScale{1.0F / 255.0F};
    std::array<float, 3> normalizationMean{};
    std::array<float, 3> normalizationDeviation{};
};

struct TokenizerArtifactSpec
{
    std::filesystem::path path{};
    int32_t imagePlaceholderId{-1};
    int32_t imageStartId{-1};
    int32_t imageContextId{-1};
    int32_t imageEndId{-1};
    int32_t padTokenId{-1};
    std::array<int32_t, 8> stopTokenIds{};
    std::size_t stopTokenCount{0};
};

struct EmbeddingArtifactSpec
{
    std::filesystem::path path{};
};

struct ProcessingConfiguration
{
    TokenizerArtifactSpec tokenizer{};
    EmbeddingArtifactSpec embedding{};
    ProcessingLimits limits{};
    ImageProcessingLimits image{};
};

struct ImageProcessingPlan
{
    SafetyStatus status{SafetyStatus::ok()};
    std::size_t imageTokenCount{0};
};

struct ProcessingRequest
{
    TokenIdView prompt{};
    vlm::ImageInputView image{};
    std::size_t imageTokenOffset{0};
    std::size_t imageTokenCount{0};
    int64_t maxGeneratedTokens{0};
    TokenIdView stopTokenIds{};
};

} // namespace trt_edgellm::safety
