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
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace trt_edgellm
{
namespace gr00t
{

//! \brief Parse one JSON file of the engine bundle, naming it in the error.
inline nlohmann::json loadJsonFile(std::filesystem::path const& path, char const* what)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error(std::string("Failed to open GR00T ") + what + ": " + path.string());
    }
    nlohmann::json document;
    file >> document;
    return document;
}

//! Stamped on every component config.json and policy.json of one build. A bundle whose
//! files disagree was assembled from two builds or two build routes and is refused.
constexpr char const* kExportIdKey = "export_id";
constexpr char const* kFrontendKey = "frontend";

struct BundleIdentity
{
    std::string exportId;
    std::string frontend;
};

//! \brief The identity a bundle file carries; a file without one predates the check.
inline BundleIdentity requireIdentity(nlohmann::json const& document, std::string const& what)
{
    BundleIdentity const found{
        document.value(kExportIdKey, std::string{}), document.value(kFrontendKey, std::string{})};
    if (found.exportId.empty() || found.frontend.empty())
    {
        throw std::runtime_error("GR00T " + what + " carries no bundle identity (" + kExportIdKey + ", " + kFrontendKey
            + "); rebuild the bundle so its components can be checked against each other");
    }
    return found;
}

inline void requireSameIdentity(
    BundleIdentity const& expected, BundleIdentity const& found, std::string const& what, std::string const& reference)
{
    if (found.exportId != expected.exportId || found.frontend != expected.frontend)
    {
        throw std::runtime_error("GR00T " + what + " is from build " + found.exportId + " (" + found.frontend + ") but "
            + reference + " is from " + expected.exportId + " (" + expected.frontend
            + "); rebuild the whole bundle from one checkpoint and one route");
    }
}

namespace binding_names
{

// vl_backbone
constexpr char const* kInputsEmbeds = "inputs_embeds";
constexpr char const* kRopeCosSin = "rope_rotary_cos_sin";
constexpr char const* kAttentionPosId = "attention_pos_id";

// action
constexpr char const* kNoiseTrajectory = "noise_trajectory";
constexpr char const* kState = "state";
constexpr char const* kAdanormModulation = "adanorm_modulation";
constexpr char const* kActionTimeBias = "action_time_bias";
constexpr char const* kTextMask = "text_attention_mask";
constexpr char const* kImageMask = "image_attention_mask";
constexpr char const* kVelocity = "velocity";

inline std::string deepstackEmbeds(int32_t index)
{
    return "deepstack_embeds_" + std::to_string(index);
}

//! Block numbers are two digits in the exported names.
inline std::string twoDigit(int32_t block)
{
    std::string const digits = std::to_string(block);
    return digits.size() < 2 ? "0" + digits : digits;
}

inline std::string crossKey(int32_t block)
{
    return "cross_k_layer" + twoDigit(block);
}

inline std::string crossValue(int32_t block)
{
    return "cross_v_layer" + twoDigit(block);
}

} // namespace binding_names
} // namespace gr00t
} // namespace trt_edgellm
