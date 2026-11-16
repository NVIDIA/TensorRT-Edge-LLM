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

#include "common/checkMacros.h"
#include "runtime/deviceFailure.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace trt_edgellm;

// Given the exceptions a failing CUDA_CHECK and CUDA_DRIVER_CHECK throw, and those a failing ELLM_CHECK and
//       check::check throw
// When each is classified
// Then the first two are CUDA call failures and the other two are not
TEST(DeviceFailureTest, CudaCheckFailuresAreCudaCallFailures)
{
    // 1 for a CUDA call failure, 0 for another runtime_error, -1 when nothing was thrown.
    auto const classify = [](auto const& throwing) {
        try
        {
            throwing();
        }
        catch (std::runtime_error const& error)
        {
            return rt::isCudaCallFailure(error) ? 1 : 0;
        }
        return -1;
    };
    EXPECT_EQ(classify([] { CUDA_CHECK(cudaErrorInvalidValue); }), 1);
    EXPECT_EQ(classify([] { CUDA_DRIVER_CHECK(CUDA_ERROR_INVALID_VALUE); }), 1);
    EXPECT_EQ(classify([] { ELLM_CHECK(false, "CUDA runtime error in a message"); }), 0);
    EXPECT_EQ(classify([] { check::check(false, "a host-side failure"); }), 0);
}
