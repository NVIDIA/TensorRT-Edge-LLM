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

#include "multimodal/cosmos3/cosmos3EdgeViTRunner.h"
#include "common/checkMacros.h"
#include <cmath>

namespace trt_edgellm
{
namespace rt
{

bool Cosmos3EdgeViTRunner::validateExtraConfig(nlohmann::json const& jsonConfig)
{
    auto const& visionConfig = jsonConfig["vision_config"];
    // SigLIP2 stores its learned pos-emb size as num_patches (256 -> 16x16 grid); the runtime resizes it to
    // each image grid through the inherited Qwen3-VL fast-pos-emb bilinear idx/weight inputs.
    auto const numPatches = visionConfig["num_patches"].get<int64_t>();
    mNumGridPerSide = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(numPatches))));
    if (mNumGridPerSide <= 0 || mNumGridPerSide * mNumGridPerSide != numPatches)
    {
        LOG_ERROR("vision_config.num_patches = %ld is not a positive square", numPatches);
        return false;
    }
    mNumDeepstackFeatures = 0;       // SigLIP2 has no deepstack taps
    mConfig.mropeInterleaved = true; // Cosmos3-Edge uses interleaved mrope_section [24, 20, 20] (architectural)
    return true;
}

bool Cosmos3EdgeViTRunner::validateAndFillConfig(std::string const& engineDir)
{
    if (!Qwen3VLViTRunner::validateAndFillConfig(engineDir))
    {
        return false;
    }
    // SigLIP2 has no temporal patching: its preprocessor config has no temporal_patch_size and the base
    // defaults it to 1. Anything else means a foreign preprocessor config was exported with this engine.
    if (mConfig.temporalPatchSize != 1)
    {
        LOG_ERROR("Cosmos3-Edge expects temporal_patch_size == 1, got %ld", mConfig.temporalPatchSize);
        return false;
    }
    // Linear patch embed over one [3, patchSize, patchSize] patch — the engine's packed-patch input width.
    int64_t const expectedInputDim = 3 * mConfig.patchSize * mConfig.patchSize;
    if (mConfig.inputDim != expectedInputDim)
    {
        LOG_ERROR(
            "Visual engine input dim %ld does not match the SigLIP2 Linear patch-embed width %ld "
            "(3 * patchSize^2, patchSize = %ld)",
            mConfig.inputDim, expectedInputDim, mConfig.patchSize);
        return false;
    }
    return true;
}

} // namespace rt
} // namespace trt_edgellm
