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

#include "safety_inference/runtime/safetyStatus.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace trt_edgellm::safety::test
{

//! Compare every field of two statuses, including their owned descriptions.
//! @param lhs First status.
//! @param rhs Second status.
//! @return True only when all fields are equal.
bool sameSafetyStatus(SafetyStatus const& lhs, SafetyStatus const& rhs) noexcept
{
    return lhs.code() == rhs.code() && lhs.fatalStatus() == rhs.fatalStatus()
        && std::strcmp(lhs.message(), rhs.message()) == 0;
}

//! Verify null and oversized producer descriptions remain bounded and owned.
//! @return True when all bounded-copy contracts hold.
bool verifySafetyStatusBounds() noexcept
{
    auto const nullStatus = SafetyStatus::error(SafetyStatusCode::kInvalidArgument, nullptr);
    if (nullStatus.message()[0] != '\0')
    {
        return false;
    }

    constexpr size_t kEXTRA_DESCRIPTION_BYTES{17U};
    std::array<char, kSAFETY_STATUS_DESCRIPTION_CAPACITY + kEXTRA_DESCRIPTION_BYTES> longDescription{};
    std::fill(longDescription.begin(), longDescription.end() - 1, 'y');
    auto const bounded = SafetyStatus::error(SafetyStatusCode::kInvalidArgument, longDescription.data());
    size_t const expectedLength{kSAFETY_STATUS_DESCRIPTION_CAPACITY - 1U};
    if (std::strlen(bounded.message()) != expectedLength || bounded.message()[expectedLength] != '\0')
    {
        return false;
    }
    return std::all_of(bounded.message(), bounded.message() + expectedLength, [](char value) { return value == 'y'; });
}

} // namespace trt_edgellm::safety::test
