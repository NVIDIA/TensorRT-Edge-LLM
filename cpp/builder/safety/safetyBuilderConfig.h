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
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace trt_edgellm::safety::builder
{

using Json = nlohmann::json;

struct SafetyLLMStaticShapeProfile
{
    int64_t batchSize{1};
    int64_t inputSeqLen{128};
    int64_t kvCacheCapacity{512};
    int64_t imageTokens{0};

    Json toJson() const
    {
        return {{"batch_size", batchSize}, {"input_seq_len", inputSeqLen}, {"kv_cache_capacity", kvCacheCapacity},
            {"image_tokens", imageTokens}};
    }
};

struct SafetyLLMBuilderConfig
{
    std::vector<SafetyLLMStaticShapeProfile> profiles;
    std::string remoteAutoTuningConfig;
    std::string safetyPluginLibrary;
    bool isVlm{false};
    bool profilingDetailed{false};

    Json toJson() const
    {
        Json profilesJson = Json::array();
        for (auto const& profile : profiles)
        {
            profilesJson.push_back(profile.toJson());
        }
        return {{"engine_capability", "safety"}, {"is_safety", true}, {"is_vlm", isVlm},
            {"remote_auto_tuning", !remoteAutoTuningConfig.empty()}, {"profiles", std::move(profilesJson)}};
    }
};

struct SafetyVisualStaticShapeProfile
{
    int64_t imageTokens{256};

    Json toJson() const
    {
        return {{"image_tokens", imageTokens}};
    }
};

struct SafetyVisualBuilderConfig
{
    std::vector<SafetyVisualStaticShapeProfile> profiles;
    std::string remoteAutoTuningConfig;
    std::string safetyPluginLibrary;
    int64_t maxImageTokensPerImage{512};
    bool profilingDetailed{false};

    Json toJson() const
    {
        Json profilesJson = Json::array();
        for (auto const& profile : profiles)
        {
            profilesJson.push_back(profile.toJson());
        }
        return {{"engine_capability", "safety"}, {"is_safety", true},
            {"max_image_tokens_per_image", maxImageTokensPerImage},
            {"remote_auto_tuning", !remoteAutoTuningConfig.empty()}, {"profiles", std::move(profilesJson)}};
    }
};

} // namespace trt_edgellm::safety::builder
