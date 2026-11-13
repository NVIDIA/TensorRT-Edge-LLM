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
#include <atomic>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace
{
using namespace trt_edgellm::safety;

struct ReentryContext
{
    SafetyErrorRecorder* recorder{};
    int32_t observedCount{-1};
    bool callbackCompleted{};
};

//! Query and clear the recorder from its forwarding callback.
//! @param opaque Borrowed ReentryContext pointer.
void reenterRecorder(void* opaque, nvinfer1::ErrorCode, nvinfer1::IErrorRecorder::ErrorDesc) noexcept
{
    auto& context = *static_cast<ReentryContext*>(opaque);
    context.observedCount = context.recorder->getNbErrors();
    context.recorder->clear();
    context.callbackCompleted = true;
}

TEST(SafetyErrorRecorder, SerializesConcurrentDiagnosticsAndForwardsOutsideLock)
{
    // Given reporters, concurrent clear/value queries, and reentry, when errors arrive, then state stays bounded.
    // Mutation mapping: C1-C3.
    ReentryContext reentryContext;
    SafetyErrorRecorder reentryRecorder(&reenterRecorder, &reentryContext);
    reentryContext.recorder = &reentryRecorder;
    EXPECT_TRUE(reentryRecorder.reportError(nvinfer1::ErrorCode::kINTERNAL_ERROR, "reentrant"));
    EXPECT_TRUE(reentryContext.callbackCompleted);
    EXPECT_EQ(reentryContext.observedCount, 1);
    EXPECT_EQ(reentryRecorder.getNbErrors(), 0);

    SafetyErrorRecorder stressRecorder(nullptr, nullptr);
    constexpr int32_t kSTRESS_ROUNDS{16};
    constexpr size_t kDESCRIPTION_BYTES{32U};
    for (int32_t round = 0; round < kSTRESS_ROUNDS; ++round)
    {
        stressRecorder.clear();
        std::array<std::array<char, kDESCRIPTION_BYTES>, kMAX_DLI_ERRORS> descriptions{};
        for (int32_t index = 0; index < kMAX_DLI_ERRORS; ++index)
        {
            std::snprintf(descriptions[index].data(), descriptions[index].size(), "round-%d-error-%d", round, index);
        }

        std::atomic<bool> start{false};
        std::atomic<int32_t> completed{0};
        std::vector<std::thread> reporters;
        reporters.reserve(kMAX_DLI_ERRORS);
        for (int32_t index = 0; index < kMAX_DLI_ERRORS; ++index)
        {
            reporters.emplace_back([&, index]() {
                while (!start.load(std::memory_order_acquire))
                {
                    std::this_thread::yield();
                }
                EXPECT_TRUE(
                    stressRecorder.reportError(nvinfer1::ErrorCode::kINTERNAL_ERROR, descriptions[index].data()));
                completed.fetch_add(1, std::memory_order_release);
            });
        }
        start.store(true, std::memory_order_release);
        while (completed.load(std::memory_order_acquire) != kMAX_DLI_ERRORS)
        {
            int32_t const observed = stressRecorder.getNbErrors();
            EXPECT_GE(observed, 0);
            EXPECT_LE(observed, kMAX_DLI_ERRORS);
            EXPECT_FALSE(stressRecorder.hasOverflowed());
        }
        for (auto& reporter : reporters)
        {
            reporter.join();
        }

        ASSERT_EQ(stressRecorder.getNbErrors(), kMAX_DLI_ERRORS);
        EXPECT_FALSE(stressRecorder.hasOverflowed());
        std::array<bool, kMAX_DLI_ERRORS> seen{};
        for (int32_t retained = 0; retained < kMAX_DLI_ERRORS; ++retained)
        {
            EXPECT_EQ(stressRecorder.getErrorCode(retained), nvinfer1::ErrorCode::kINTERNAL_ERROR);
            auto const* description = stressRecorder.getErrorDesc(retained);
            ASSERT_NE(description, nullptr);
            EXPECT_NE(std::memchr(description, '\0', kDLI_DESCRIPTION_CAPACITY), nullptr);
            auto const found = std::find_if(descriptions.begin(), descriptions.end(),
                [description](auto const& expected) { return std::strcmp(description, expected.data()) == 0; });
            ASSERT_NE(found, descriptions.end());
            size_t const index = static_cast<size_t>(std::distance(descriptions.begin(), found));
            EXPECT_FALSE(seen[index]);
            seen[index] = true;
        }
        EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }));
    }

    SafetyErrorRecorder clearStressRecorder(nullptr, nullptr);
    constexpr int32_t kCONCURRENT_OPERATIONS{1000};
    constexpr int32_t kNB_STRESS_WORKERS{2};
    std::atomic<bool> clearStressStart{false};
    std::atomic<int32_t> clearStressCompleted{0};
    std::atomic<bool> snapshotsValid{true};
    std::thread reporter([&]() {
        while (!clearStressStart.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
        for (int32_t operation = 0; operation < kCONCURRENT_OPERATIONS; ++operation)
        {
            clearStressRecorder.reportError(nvinfer1::ErrorCode::kINTERNAL_ERROR, "clear-stress");
        }
        clearStressCompleted.fetch_add(1, std::memory_order_release);
    });
    std::thread clearer([&]() {
        while (!clearStressStart.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
        for (int32_t operation = 0; operation < kCONCURRENT_OPERATIONS; ++operation)
        {
            clearStressRecorder.clear();
        }
        clearStressCompleted.fetch_add(1, std::memory_order_release);
    });
    clearStressStart.store(true, std::memory_order_release);
    while (clearStressCompleted.load(std::memory_order_acquire) != kNB_STRESS_WORKERS)
    {
        int32_t const count = clearStressRecorder.getNbErrors();
        auto const code = clearStressRecorder.getErrorCode(0);
        static_cast<void>(clearStressRecorder.hasOverflowed());
        bool const countValid = count >= 0 && count <= kMAX_DLI_ERRORS;
        bool const codeValid
            = code == nvinfer1::ErrorCode::kINTERNAL_ERROR || code == nvinfer1::ErrorCode::kUNSPECIFIED_ERROR;
        if (!countValid || !codeValid)
        {
            snapshotsValid.store(false, std::memory_order_release);
        }
    }
    reporter.join();
    clearer.join();
    EXPECT_TRUE(snapshotsValid.load(std::memory_order_acquire));
    clearStressRecorder.clear();
    EXPECT_EQ(clearStressRecorder.getNbErrors(), 0);
    EXPECT_FALSE(clearStressRecorder.hasOverflowed());
}

TEST(SafetyErrorRecorder, CopiedDescriptionSurvivesConcurrentReplacement)
{
    // Given a retained description, when another thread clears and replaces it, then the copy remains stable.
    // Mutation mapping: C4.
    SafetyErrorRecorder recorder(nullptr, nullptr);
    ASSERT_TRUE(recorder.reportError(nvinfer1::ErrorCode::kINTERNAL_ERROR, "before-clear"));
    std::atomic<bool> snapshotReady{false};
    std::atomic<bool> replacementReady{false};
    DliDescription snapshot{};
    bool copySucceeded{};
    bool snapshotStable{};
    std::thread reader([&]() {
        copySucceeded = recorder.copyErrorDescription(0, snapshot);
        snapshotReady.store(true, std::memory_order_release);
        while (!replacementReady.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
        snapshotStable = std::strcmp(snapshot.data(), "before-clear") == 0;
    });
    std::thread writer([&]() {
        while (!snapshotReady.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
        recorder.clear();
        EXPECT_TRUE(recorder.reportError(nvinfer1::ErrorCode::kINTERNAL_ERROR, "after-clear"));
        replacementReady.store(true, std::memory_order_release);
    });
    reader.join();
    writer.join();
    EXPECT_TRUE(copySucceeded);
    EXPECT_TRUE(snapshotStable);
    EXPECT_STREQ(recorder.getErrorDesc(0), "after-clear");
}

TEST(SafetyErrorRecorder, RawDescriptionSlotsRemainDistinct)
{
    // Given two recorder instances and multiple slots, when raw descriptions are queried, then no getter aliases them.
    // Mutation mapping: C5.
    SafetyErrorRecorder first(nullptr, nullptr);
    SafetyErrorRecorder second(nullptr, nullptr);
    ASSERT_TRUE(first.reportError(nvinfer1::ErrorCode::kINTERNAL_ERROR, "first-zero"));
    ASSERT_TRUE(first.reportError(nvinfer1::ErrorCode::kFAILED_EXECUTION, "first-one"));
    ASSERT_TRUE(second.reportError(nvinfer1::ErrorCode::kINVALID_STATE, "second-zero"));

    auto const* firstZero = first.getErrorDesc(0);
    auto const* firstOne = first.getErrorDesc(1);
    auto const* secondZero = second.getErrorDesc(0);
    EXPECT_NE(firstZero, firstOne);
    EXPECT_NE(firstZero, secondZero);
    EXPECT_NE(firstOne, secondZero);
    EXPECT_STREQ(first.getErrorDesc(-1), "");
    EXPECT_STREQ(firstZero, "first-zero");
    EXPECT_STREQ(firstOne, "first-one");
    EXPECT_STREQ(secondZero, "second-zero");
}

TEST(SafetyErrorRecorder, ReferenceUnderflowLatchesFatalStatus)
{
    // Given a recorder with no borrowers, when a release underflows, then it reports a fatal without wrapping.
    // Mutation mapping: C6.
    SafetyErrorState state;
    auto& recorder = static_cast<SafetyErrorRecorder&>(state.recorder());
    EXPECT_EQ(recorder.decRefCount(), 0);
    EXPECT_EQ(recorder.referenceCount(), 0);
    ASSERT_EQ(recorder.getNbErrors(), 1);
    EXPECT_EQ(recorder.getErrorCode(0), nvinfer1::ErrorCode::kINTERNAL_ERROR);
    EXPECT_STREQ(recorder.getErrorDesc(0), "Safety recorder reference count underflow");
    auto const fatal = state.status();
    EXPECT_TRUE(fatal.fatalStatus());
    EXPECT_EQ(fatal.code(), SafetyStatusCode::kFailedExecution);
    EXPECT_STREQ(fatal.message(), "Safety recorder reference count underflow");
}

} // namespace
