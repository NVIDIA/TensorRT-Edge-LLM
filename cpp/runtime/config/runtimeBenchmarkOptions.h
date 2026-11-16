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

namespace trt_edgellm
{
namespace rt
{

//! Optional experiment controls. Defaults preserve normal request semantics.
struct RuntimeBenchmarkOptions
{
    int32_t maxDecodeRounds{0};         //!< Zero leaves the request token/EOS budget in control.
    int32_t forcedAcceptanceLength{-1}; //!< Total accepted tokens including the bonus; -1 uses natural acceptance.
    bool gpuTiming{true}; //!< CUDA-event stage timing while profiling; acceptance records do not depend on it.

    bool enabled() const noexcept
    {
        return maxDecodeRounds > 0 || forcedAcceptanceLength > 0;
    }
};

} // namespace rt
} // namespace trt_edgellm
