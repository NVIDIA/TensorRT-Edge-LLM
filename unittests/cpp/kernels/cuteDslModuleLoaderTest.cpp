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

#include "kernels/cuteDslModuleLoader.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>
#include <vector>

namespace trt_edgellm
{
namespace
{

struct FakeModule
{
    void* module{};
};

int gModuleToken{};
std::atomic<int> gLoadCalls{};
std::atomic<int> gUnloadCalls{};
std::atomic<int> gActiveLoads{};
std::atomic<int> gMaxActiveLoads{};

void resetCounters()
{
    gLoadCalls = 0;
    gUnloadCalls = 0;
    gActiveLoads = 0;
    gMaxActiveLoads = 0;
    detail::clearCuteDslCudaError();
}

void successfulLoader(FakeModule* module)
{
    ++gLoadCalls;
    module->module = &gModuleToken;
}

void successfulUnloader(FakeModule*)
{
    ++gUnloadCalls;
}

void nullThenSuccessfulLoader(FakeModule* module)
{
    if (++gLoadCalls > 1)
    {
        module->module = &gModuleToken;
    }
}

void throwingThenSuccessfulLoader(FakeModule* module)
{
    if (++gLoadCalls == 1)
    {
        throw std::runtime_error("injected loader exception");
    }
    module->module = &gModuleToken;
}

void cudaErrorThenSuccessfulLoader(FakeModule* module)
{
    module->module = &gModuleToken;
    if (++gLoadCalls == 1)
    {
        detail::recordCuteDslCudaError(cudaErrorInvalidValue);
    }
}

void alwaysFailingLoader(FakeModule* module)
{
    ++gLoadCalls;
    module->module = &gModuleToken;
    detail::recordCuteDslCudaError(cudaErrorInvalidValue);
}

void failingUnloader(FakeModule*)
{
    ++gUnloadCalls;
    detail::recordCuteDslCudaError(cudaErrorUnknown);
}

void updateMaxActiveLoads(int active)
{
    int observed = gMaxActiveLoads.load();
    while (observed < active && !gMaxActiveLoads.compare_exchange_weak(observed, active))
    {
    }
}

void slowLoaderA(FakeModule* module)
{
    ++gLoadCalls;
    int const active = ++gActiveLoads;
    updateMaxActiveLoads(active);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    --gActiveLoads;
    module->module = &gModuleToken;
}

void slowLoaderB(FakeModule* module)
{
    ++gLoadCalls;
    int const active = ++gActiveLoads;
    updateMaxActiveLoads(active);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    --gActiveLoads;
    module->module = &gModuleToken;
}

class CuteDslModuleLoaderTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        resetCounters();
        cudaError_t const error = cudaStreamCreate(&mStream);
        if (error != cudaSuccess)
        {
            GTEST_SKIP() << "CUDA stream unavailable: " << cudaGetErrorString(error);
        }
    }

    void TearDown() override
    {
        if (mStream != nullptr)
        {
            EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
        }
    }

    cudaStream_t mStream{};
};

TEST(CuteDslModuleLoaderErrorRecorderTest, KeepsOnlyFirstErrorAndConsumeClearsIt)
{
    detail::clearCuteDslCudaError();
    detail::recordCuteDslCudaError(cudaErrorInvalidValue);
    detail::recordCuteDslCudaError(cudaErrorUnknown);
    EXPECT_EQ(detail::takeCuteDslCudaError(), cudaErrorInvalidValue);
    EXPECT_EQ(detail::takeCuteDslCudaError(), cudaSuccess);
}

TEST_F(CuteDslModuleLoaderTest, LoadsExactlyOnceAndKeepsModuleResident)
{
    {
        detail::LazyKernelModule<FakeModule> state;
        EXPECT_TRUE((detail::ensureModuleLoaded<successfulLoader, successfulUnloader>(state, "successful", mStream)));
        EXPECT_TRUE((detail::ensureModuleLoaded<successfulLoader, successfulUnloader>(state, "successful", mStream)));
        EXPECT_EQ(state.module.module, &gModuleToken);
        EXPECT_EQ(gLoadCalls, 1);
    }
    EXPECT_EQ(gUnloadCalls, 0);
}

TEST_F(CuteDslModuleLoaderTest, ConcurrentCallersShareOneInitialization)
{
    detail::LazyKernelModule<FakeModule> state;
    std::atomic<int> successes{};
    std::vector<std::thread> threads;
    for (int32_t index = 0; index < 16; ++index)
    {
        threads.emplace_back([&] {
            if (detail::ensureModuleLoaded<slowLoaderA, successfulUnloader>(state, "concurrent", mStream))
            {
                ++successes;
            }
        });
    }
    for (auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(successes, 16);
    EXPECT_EQ(gLoadCalls, 1);
}

TEST_F(CuteDslModuleLoaderTest, SerializesSlowPathAcrossDistinctVariants)
{
    detail::LazyKernelModule<FakeModule> first;
    detail::LazyKernelModule<FakeModule> second;
    bool firstLoaded{};
    bool secondLoaded{};
    std::thread firstThread([&] {
        firstLoaded = detail::ensureModuleLoaded<slowLoaderA, successfulUnloader>(first, "variant_a", mStream);
    });
    std::thread secondThread([&] {
        secondLoaded = detail::ensureModuleLoaded<slowLoaderB, successfulUnloader>(second, "variant_b", mStream);
    });
    firstThread.join();
    secondThread.join();

    EXPECT_TRUE(firstLoaded);
    EXPECT_TRUE(secondLoaded);
    EXPECT_EQ(gLoadCalls, 2);
    EXPECT_EQ(gMaxActiveLoads, 1);
}

TEST_F(CuteDslModuleLoaderTest, NullHandleFailureCanRetry)
{
    detail::LazyKernelModule<FakeModule> state;
    EXPECT_FALSE(
        (detail::ensureModuleLoaded<nullThenSuccessfulLoader, successfulUnloader>(state, "null_then_retry", mStream)));
    EXPECT_TRUE(
        (detail::ensureModuleLoaded<nullThenSuccessfulLoader, successfulUnloader>(state, "null_then_retry", mStream)));
    EXPECT_EQ(gLoadCalls, 2);
    EXPECT_EQ(gUnloadCalls, 0);
}

TEST_F(CuteDslModuleLoaderTest, LoaderExceptionCanRetry)
{
    detail::LazyKernelModule<FakeModule> state;
    EXPECT_FALSE((detail::ensureModuleLoaded<throwingThenSuccessfulLoader, successfulUnloader>(
        state, "throw_then_retry", mStream)));
    EXPECT_TRUE((detail::ensureModuleLoaded<throwingThenSuccessfulLoader, successfulUnloader>(
        state, "throw_then_retry", mStream)));
    EXPECT_EQ(gLoadCalls, 2);
}

TEST_F(CuteDslModuleLoaderTest, PartialModuleIsCleanedBeforeRetry)
{
    detail::LazyKernelModule<FakeModule> state;
    EXPECT_FALSE((detail::ensureModuleLoaded<cudaErrorThenSuccessfulLoader, successfulUnloader>(
        state, "partial_then_retry", mStream)));
    EXPECT_EQ(state.module.module, nullptr);
    EXPECT_TRUE((detail::ensureModuleLoaded<cudaErrorThenSuccessfulLoader, successfulUnloader>(
        state, "partial_then_retry", mStream)));
    EXPECT_EQ(gLoadCalls, 2);
    EXPECT_EQ(gUnloadCalls, 1);
}

TEST_F(CuteDslModuleLoaderTest, CleanupFailurePoisonsState)
{
    detail::LazyKernelModule<FakeModule> state;
    EXPECT_FALSE((detail::ensureModuleLoaded<alwaysFailingLoader, failingUnloader>(state, "poisoned", mStream)));
    EXPECT_FALSE((detail::ensureModuleLoaded<alwaysFailingLoader, failingUnloader>(state, "poisoned", mStream)));
    EXPECT_EQ(gLoadCalls, 1);
    EXPECT_EQ(gUnloadCalls, 1);
    EXPECT_EQ(state.status.load(), detail::LazyKernelModuleStatus::kPoisoned);
}

TEST_F(CuteDslModuleLoaderTest, RejectsFirstLoadDuringStreamCapture)
{
    detail::LazyKernelModule<FakeModule> state;
    ASSERT_EQ(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    EXPECT_FALSE(
        (detail::ensureModuleLoaded<successfulLoader, successfulUnloader>(state, "captured_first_use", mStream)));
    EXPECT_EQ(gLoadCalls, 0);

    cudaGraph_t graph{};
    ASSERT_EQ(cudaStreamEndCapture(mStream, &graph), cudaSuccess);
    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);

    EXPECT_TRUE(
        (detail::ensureModuleLoaded<successfulLoader, successfulUnloader>(state, "captured_first_use", mStream)));
    EXPECT_EQ(gLoadCalls, 1);
}

#if defined(ENABLE_CUTEDSL_MODULE_TEST_HOOK)
TEST_F(CuteDslModuleLoaderTest, TestHookMatchesExactVariantAndFailureCanRetry)
{
    struct EnvironmentGuard
    {
        ~EnvironmentGuard()
        {
            unsetenv("TRT_EDGELLM_TEST_FAIL_CUTEDSL_MODULE");
        }
    } const guard;

    ASSERT_EQ(setenv("TRT_EDGELLM_TEST_FAIL_CUTEDSL_MODULE", "target_variant", 1), 0);
    detail::LazyKernelModule<FakeModule> target;
    detail::LazyKernelModule<FakeModule> other;

    EXPECT_FALSE((detail::ensureModuleLoaded<successfulLoader, successfulUnloader>(target, "target_variant", mStream)));
    EXPECT_TRUE((detail::ensureModuleLoaded<successfulLoader, successfulUnloader>(other, "other_variant", mStream)));
    EXPECT_EQ(gLoadCalls, 1);

    ASSERT_EQ(unsetenv("TRT_EDGELLM_TEST_FAIL_CUTEDSL_MODULE"), 0);
    EXPECT_TRUE((detail::ensureModuleLoaded<successfulLoader, successfulUnloader>(target, "target_variant", mStream)));
    EXPECT_EQ(gLoadCalls, 2);
}
#endif

} // namespace
} // namespace trt_edgellm
