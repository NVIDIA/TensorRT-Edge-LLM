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

#include "common/cudaUtils.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace trt_edgellm;

//! getDeviceMultiProcessorCount() must match a direct attribute query of the
//! current device and always be a usable persistent-grid size (>= 1).
//! The second call exercises the warm process cache path and must return
//! the same value.
TEST(CudaUtilsTest, MultiProcessorCountMatchesDeviceAttribute)
{
    int device = -1;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    int expected = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&expected, cudaDevAttrMultiProcessorCount, device), cudaSuccess);

    int32_t const cold = getDeviceMultiProcessorCount();
    EXPECT_EQ(cold, expected);
    EXPECT_GE(cold, 1);
    EXPECT_EQ(getDeviceMultiProcessorCount(), expected); // cached path
}
