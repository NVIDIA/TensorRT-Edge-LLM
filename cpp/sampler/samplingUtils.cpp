/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "sampler/sampling.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace trt_edgellm
{

bool shouldUseNonGreedySampling(float temperature, int64_t topK, float topP) noexcept
{
    // SamplingParams normalizes near-zero temperatures to greedy top-1 sampling.
    if (temperature >= 0.0F && temperature < 1e-3F)
    {
        return false;
    }

    // topK == 1 forces greedy regardless of other params (only one candidate token)
    if (topK == 1)
    {
        return false;
    }
    return (topK > 1) || (topP < 1.0f - 1e-6f) || (temperature > 1e-3f && std::fabs(temperature - 1.0f) > 1e-3f);
}

bool shouldUseNonGreedySampling(RowSamplingParams const& params) noexcept
{
    return shouldUseNonGreedySampling(params.temperature, params.topK, params.topP);
}

bool operator==(RowSamplingParams const& lhs, RowSamplingParams const& rhs) noexcept
{
    return lhs.temperature == rhs.temperature && lhs.topK == rhs.topK && lhs.topP == rhs.topP;
}

bool operator!=(RowSamplingParams const& lhs, RowSamplingParams const& rhs) noexcept
{
    return !(lhs == rhs);
}

RowSamplingParams makeRowSamplingParams(float temperature, int64_t topK, float topP)
{
    if (topK > std::numeric_limits<int32_t>::max())
    {
        throw std::invalid_argument("top-k does not fit the sampler's int32 top-k");
    }
    return RowSamplingParams{temperature, topK < 0 ? int32_t{0} : static_cast<int32_t>(topK), topP};
}

} // namespace trt_edgellm
