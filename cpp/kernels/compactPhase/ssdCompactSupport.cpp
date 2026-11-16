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
#include "kernels/mamba/cuteDslSSDRunner.h"

#include <exception>

namespace trt_edgellm
{

CompactPhaseSupport mamba2SsdCompactSupport([[maybe_unused]] int32_t headDim, [[maybe_unused]] int32_t stateDim,
    [[maybe_unused]] int32_t smVersion, bool specVerifyState) noexcept
{
    if (specVerifyState)
    {
        return {false, "spec-verify layers have no compact execution path"};
    }
#ifdef CUTE_DSL_SSD_ENABLED
    try
    {
        if (CuteDslSSDRunner::canImplementPacked(headDim, stateDim, smVersion)
            || CuteDslSSDRunner::canImplementPackedStatePool(headDim, stateDim, smVersion))
        {
            return {true, nullptr};
        }
        return {false, "no packed SSD prefill kernel for this SM, head dim and state size in this build"};
    }
    catch (std::exception const& e)
    {
        LOG_WARNING("update_ssm_state: packed SSD capability query failed (%s); compact execution disabled", e.what());
        return {false, "the packed SSD capability query failed"};
    }
#else
    return {false, "this build has no CuTe DSL SSD kernels"};
#endif
}

} // namespace trt_edgellm
