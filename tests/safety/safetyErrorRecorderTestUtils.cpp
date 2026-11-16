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

#include "safety_inference/runtime/safetyErrorState.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace trt_edgellm::safety::test
{

bool sameSafetyStatus(SafetyStatus const& lhs, SafetyStatus const& rhs) noexcept;

//! Verify fixed recorder bounds, invalid-index behavior, overflow behavior, and clear isolation from the fatal
//! latch.
//! @param state Shared state that owns the recorder and terminal latch.
//! @param recorder Recorder under test.
//! @param firstFatal Expected immutable terminal status.
//! @return True when every recorder contract holds.
bool verifyBoundedRecorderContracts(
    SafetyErrorState& state, nvinfer2::safe::ISafeRecorder& recorder, SafetyStatus const& firstFatal) noexcept
{
    recorder.clear();
    if (recorder.getNbErrors() != 0 || recorder.hasOverflowed() || std::strcmp(recorder.getErrorDesc(0), "") != 0
        || !sameSafetyStatus(state.status(), firstFatal))
    {
        return false;
    }

    constexpr size_t kEXTRA_DESCRIPTION_BYTES{23U};
    std::array<char, kDLI_DESCRIPTION_CAPACITY + kEXTRA_DESCRIPTION_BYTES> longDescription{};
    std::fill(longDescription.begin(), longDescription.end() - 1, 'x');
    for (int32_t index = 0; index <= kMAX_DLI_ERRORS; ++index)
    {
        if (!recorder.reportError(nvinfer1::ErrorCode::kINTERNAL_ERROR, longDescription.data()))
        {
            return false;
        }
    }
    auto const* retainedDescription = recorder.getErrorDesc(0);
    size_t const expectedLength{kDLI_DESCRIPTION_CAPACITY - 1U};
    if (recorder.getNbErrors() != kMAX_DLI_ERRORS || !recorder.hasOverflowed()
        || recorder.getErrorCode(0) != nvinfer1::ErrorCode::kINTERNAL_ERROR || retainedDescription == nullptr
        || std::strlen(retainedDescription) != expectedLength || retainedDescription[expectedLength] != '\0')
    {
        return false;
    }

    recorder.clear();
    return recorder.getNbErrors() == 0 && !recorder.hasOverflowed() && sameSafetyStatus(state.status(), firstFatal);
}

} // namespace trt_edgellm::safety::test
