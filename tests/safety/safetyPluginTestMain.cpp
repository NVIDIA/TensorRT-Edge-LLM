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

#include "safety_inference/kernels/safetyAttentionModules.h"
#include "safety_inference/plugins/safetyAttentionPlugin.h"

#include <cstring>
#include <gtest/gtest.h>

namespace
{
class GpuEnvironment : public testing::Environment
{
public:
    void SetUp() override
    {
        auto const* tests = testing::UnitTest::GetInstance();
        bool gpuTests = false;
        for (int i = 0; i < tests->total_test_suite_count(); ++i)
        {
            auto const* suite = tests->GetTestSuite(i);
            if (std::strncmp(suite->name(), "SafetyGpu", 9) == 0 && suite->should_run())
            {
                gpuTests = true;
            }
        }
        if (!gpuTests)
        {
            return;
        }
        ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
        CUdevice device{};
        ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
#if CUDA_VERSION >= 13000
        ASSERT_EQ(cuCtxCreate(&mContext, nullptr, 0, device), CUDA_SUCCESS);
#else
        ASSERT_EQ(cuCtxCreate(&mContext, 0, device), CUDA_SUCCESS);
#endif
        ASSERT_EQ(mModules.load(), CUDA_SUCCESS);
        mReady = true;
    }

    void TearDown() override
    {
        if (mReady)
        {
            EXPECT_EQ(mModules.release(), CUDA_SUCCESS);
            EXPECT_EQ(cuCtxDestroy(mContext), CUDA_SUCCESS);
        }
    }

private:
    trt_edgellm::safety::SafetyAttentionModules mModules;
    trt_edgellm::safety::SafetyAttentionModuleScope mScope{mModules};
    CUcontext mContext{};
    bool mReady{};
};
} // namespace

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new GpuEnvironment);
    return RUN_ALL_TESTS();
}
