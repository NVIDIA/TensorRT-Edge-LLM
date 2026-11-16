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

#include "compactPhaseSupport.h"

#include "common/logger.h"
#include "kernels/gdnKernels/cuteDslGDNRunner.h"

#include <exception>

namespace trt_edgellm
{

CompactPhaseSupport gatedDeltaNetCompactSupport([[maybe_unused]] int32_t kDim, [[maybe_unused]] int32_t vDim,
    [[maybe_unused]] int32_t smVersion, [[maybe_unused]] bool specVerifyState,
    [[maybe_unused]] bool diffusionState) noexcept
{
#ifdef CUTE_DSL_GDN_ENABLED
    if (specVerifyState || diffusionState)
    {
        return {false, "spec-verify and diffusion layers have no compact execution path"};
    }
    try
    {
        if (!CuteDslGDNRunner::canImplementPackedVarlen(kDim, vDim, smVersion))
        {
            return {false, "no packed-varlen GDN prefill kernel for this SM, SM count and K/V head dims in this build"};
        }
        if (!cuteDslArtifactCarriesSm(smVersion))
        {
            return {false, "the linked CuTe DSL artifact has no kernels for this SM"};
        }
        return {true, nullptr};
    }
    catch (std::exception const& e)
    {
        LOG_WARNING("gated_delta_net: compact capability query failed (%s); compact execution disabled", e.what());
        return {false, "the packed-varlen GDN capability query failed"};
    }
#else
    return {false, "this build has no CuTe DSL GDN kernels"};
#endif
}

} // namespace trt_edgellm
