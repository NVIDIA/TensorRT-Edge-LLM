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

#include "safety_inference/runtime/resources/safetyModelLocalData.h"

#include <array>

namespace trt_edgellm::safety::llm
{

//! Artifact reader supplies these immutable LLM dimensions before resource declaration.
struct ModelResourceSpec
{
    ResourceOwnerId owner{0};
    int32_t vocabSize{0};
    int32_t hiddenSize{0};
    int32_t layerCount{0};
    int32_t kvHeadCount{0};
    int32_t headDim{0};
    int32_t padTokenId{0};
};

//! Engine-query metadata for one selected stage/profile; no allocated resources.
struct StageResourceSpec
{
    LMStageId stage{};
    int32_t profileId{-1};
    std::vector<RuntimeTensorDesc> bindings{};
};

enum class ModelMemory : RuntimeResourceKey
{
    kTOKEN_IDS,
    kEMBEDDING_TOKEN_IDS,
    kCONTEXT_LENGTH,
    kKV_START,
    kLAST_TOKEN,
    kLOGITS,
    kEMBEDDING_TABLE,
    kROPE_CACHE,
    kDEVICE_TOKEN_IDS,
    kCOUNT,
};

struct StageResources
{
    LMStageId stage{};
    int32_t profileId{-1};
    std::vector<RuntimeTensorView const*> bindings{};
    cudaStream_t stream{nullptr};
};

//! Runtime-owned metadata over manager-owned storage. Processing initializes embedding/RoPE payloads
//! and synchronizes Initialization GPU work before SafetyRuntime publishes readiness.
struct ModelLocalData final : SafetyModelLocalData
{
    ModelResourceSpec spec{};
    StageResources prefill{};
    StageResources decode{};
    std::array<RuntimeMemoryView, static_cast<std::size_t>(ModelMemory::kCOUNT)> storage{};
    int32_t maxPrefillTokens{0};
    int32_t maxContextLength{0};
};

//! Batched FP16 embeddings and contiguous FP16 KV, shared by the initial InternVL LLM stages.
//! The factory owns query metadata; borrowed tensor views remain owned by the Runner adapter.
class ModelLocalDataFactory final : public SafetyModelLocalDataFactory
{
public:
    ModelLocalDataFactory(ModelResourceSpec spec, StageResourceSpec prefill, StageResourceSpec decode);
    SafetyStatus declareResources(ResourceDeclaration& output) const noexcept override;
    SafetyStatus create(
        PreparedResources const& resources, std::unique_ptr<SafetyModelLocalData>& output) const noexcept override;

private:
    struct Capacities
    {
        std::size_t tokens{0};
        std::size_t logitsBytes{0};
        std::size_t ropeBytes{0};
        int32_t contextLength{0};
    };
    SafetyStatus validate(Capacities& output) const;
    ModelResourceSpec mSpec;
    StageResourceSpec mPrefill;
    StageResourceSpec mDecode;
};

} // namespace trt_edgellm::safety::llm
