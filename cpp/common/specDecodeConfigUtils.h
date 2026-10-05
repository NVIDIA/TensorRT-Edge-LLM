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
#include <nlohmann/json.hpp>

namespace trt_edgellm
{

//! Return whether an engine config carries any legacy speculative-decoding marker.
inline bool configRevealsSpecDecode(nlohmann::json const& config)
{
    constexpr std::array<char const*, 16> kTOP_LEVEL_FLAGS{"eagle_base", "is_eagle3_draft", "mtp_base", "is_mtp_draft",
        "mtp_tree_base", "dflash_base", "dflash_tree_base", "is_dflash_draft", "jetspec_base", "jetspec_tree_base",
        "is_jetspec_draft", "dspark_base", "is_dspark_draft", "gemma4_mtp_base", "gemma4_mtp_draft",
        "shares_target_kv"};
    for (char const* flag : kTOP_LEVEL_FLAGS)
    {
        if (config.value(flag, false))
        {
            return true;
        }
    }

    auto const builderConfig = config.find("builder_config");
    return builderConfig != config.end() && builderConfig->is_object()
        && (builderConfig->value("spec_base", false) || builderConfig->value("spec_draft", false));
}

} // namespace trt_edgellm
