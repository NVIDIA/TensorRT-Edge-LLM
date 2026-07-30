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

#include "common/checkMacros.h"

#include <cmath>
#include <cstdint>
#include <optional>

namespace trt_edgellm
{

//! Validate an absolute QK^T multiplier.
inline void validateAttentionScale(float attentionScale)
{
    ELLM_CHECK(std::isfinite(attentionScale) && attentionScale > 0.0F,
        "Attention scale must be finite and greater than zero.");
}

//! Resolve an optional absolute QK^T multiplier, preserving the default.
inline float resolveAttentionScale(std::optional<float> attentionScale, int32_t headSize)
{
    if (!attentionScale.has_value())
    {
        ELLM_CHECK(headSize > 0, "Attention head size must be greater than zero when attention scale is absent.");
        attentionScale = 1.0F / std::sqrt(static_cast<float>(headSize));
    }

    validateAttentionScale(*attentionScale);
    return *attentionScale;
}

} // namespace trt_edgellm
