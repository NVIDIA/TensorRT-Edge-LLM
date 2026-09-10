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

#include "runtime/config/deploymentConfig.h"
#include "runtime/state/contextCache/specReuseContract.h"

#include <cstdint>
#include <optional>

namespace trt_edgellm::rt
{

//! Base-model state topology for a cache-enabled deployment.
enum class ContextCacheModelStateKind : uint8_t
{
    kAttentionOnly,
    kPureRecurrent,
    kHybrid,
};

//! Orthogonal deployment profile consumed by the context-cache coordinator.
struct ContextCacheDeploymentProfile
{
    ContextCacheModelStateKind baseStateKind{};
    std::optional<SpecReuseContract> specReuseContract;

    bool hasAttention() const noexcept
    {
        return baseStateKind != ContextCacheModelStateKind::kPureRecurrent;
    }

    bool isHybrid() const noexcept
    {
        return baseStateKind == ContextCacheModelStateKind::kHybrid;
    }

    bool isPureRecurrent() const noexcept
    {
        return baseStateKind == ContextCacheModelStateKind::kPureRecurrent;
    }

    bool usesCheckpointReuse() const noexcept
    {
        return isHybrid() || isPureRecurrent();
    }

    bool isSpeculative() const noexcept
    {
        return specReuseContract.has_value();
    }

    bool ownsPagedSpecState() const noexcept
    {
        return isSpeculative() && specReuseContract->ownsPagedSpecState;
    }
};

//! Validate the logical deployment contract and return its context-cache profile.
//!
//! This is an early configuration gate. Request-level policy such as greedy-only EAGLE, media bypass, and
//! full-hidden-output bypass is selected before admission by the runtime. Physical binding compatibility is validated
//! separately after the engines are loaded.
//! @throws std::runtime_error when a requested cache-enabled deployment is outside the supported matrix; callers must
//! fail initialization rather than silently changing the configured execution mode
ContextCacheDeploymentProfile validateContextCacheDeployment(DeploymentConfig const& deployment);

} // namespace trt_edgellm::rt
