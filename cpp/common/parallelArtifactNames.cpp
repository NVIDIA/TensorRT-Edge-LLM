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

#include "common/parallelArtifactNames.h"
#include "common/checkMacros.h"

namespace trt_edgellm
{
namespace parallel_artifacts
{

namespace
{

bool isParallel(RankArtifactContext const& context) noexcept
{
    return context.worldSize > 1;
}

void validateContext(RankArtifactContext const& context)
{
    ELLM_CHECK(context.worldSize > 0, "Artifact world size must be positive.");
    ELLM_CHECK(context.globalRank >= 0 && context.globalRank < context.worldSize,
        "Artifact global rank must be within [0, worldSize).");
}

std::string worldSuffix(RankArtifactContext const& context)
{
    validateContext(context);
    if (!isParallel(context))
    {
        return "";
    }
    return "_world" + std::to_string(context.worldSize);
}

std::string worldRankSuffix(RankArtifactContext const& context)
{
    validateContext(context);
    if (!isParallel(context))
    {
        return "";
    }
    return "_world" + std::to_string(context.worldSize) + "_rank" + std::to_string(context.globalRank);
}

} // namespace

std::string onnxFileName(RankArtifactContext const& context)
{
    return "model" + worldRankSuffix(context) + ".onnx";
}

std::string configFileName(RankArtifactContext const& context)
{
    return "config" + worldSuffix(context) + ".json";
}

std::string engineFileName(RankArtifactContext const& context)
{
    return "llm" + worldRankSuffix(context) + ".engine";
}

} // namespace parallel_artifacts
} // namespace trt_edgellm
