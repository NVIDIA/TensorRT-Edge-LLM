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

#include "runtime/llmRankRuntime.h"

#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>

using namespace trt_edgellm::rt;

namespace
{

constexpr int32_t kMAX_TOKENS = 64;
constexpr int32_t kREQUESTS = 3;
RaggedEngineContract const kCONTRACT{TokenLayoutBackend::kEntryPaddedCompatibility, 4, 16, kMAX_TOKENS, 4, false};

} // namespace

TEST(SteppedGenerationStorageTests, ReusesOwningStorageAcrossRequests)
{
    // Given runtime-owned pinned storage, when consecutive requests reserve and reshape their token snapshots,
    // then each uses the same allocation and returns its ownership and full capacity before the next request.
    Tensor storage({kMAX_TOKENS}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
    void* const pointer = storage.rawPointer();
    std::atomic<bool> active{false};
    for (int32_t request = 0; request < kREQUESTS; ++request)
    {
        active.store(true);
        {
            LLMRankRuntime::SteppedGeneration generation(active, storage);
            EXPECT_TRUE(storage.isEmpty());
            generation.context.initializeRaggedScratch(kCONTRACT);
            Tensor& snapshot = generation.context.raggedExecutionBatch.hostTokenIds;
            ASSERT_EQ(snapshot.rawPointer(), pointer);
            EXPECT_TRUE(snapshot.getOwnMemory());
            EXPECT_TRUE(active.load());
            ASSERT_TRUE(snapshot.reshape({request + 1}));
            snapshot.dataPointer<int32_t>()[request] = request;
        }
        EXPECT_FALSE(active.load());
        ASSERT_EQ(storage.rawPointer(), pointer);
        EXPECT_TRUE(storage.getOwnMemory());
        EXPECT_EQ(storage.getMemoryCapacity(), kMAX_TOKENS * sizeof(int32_t));
        EXPECT_EQ(storage.getShape().volume(), request + 1);
        EXPECT_EQ(storage.dataPointer<int32_t>()[request], request);
    }
}

TEST(SteppedGenerationStorageTests, ReturnsStorageOnException)
{
    // Given a request borrowing pinned storage, when request processing throws,
    // then unwinding returns ownership and releases the latch so the next request can reuse the buffer.
    Tensor storage({kMAX_TOKENS}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
    void* const pointer = storage.rawPointer();
    std::atomic<bool> active{true};
    EXPECT_THROW(
        {
            LLMRankRuntime::SteppedGeneration generation(active, storage);
            generation.context.initializeRaggedScratch(kCONTRACT);
            throw std::runtime_error("request failed");
        },
        std::runtime_error);
    EXPECT_FALSE(active.load());
    ASSERT_EQ(storage.rawPointer(), pointer);
    EXPECT_TRUE(storage.getOwnMemory());
    active.store(true);
    {
        LLMRankRuntime::SteppedGeneration generation(active, storage);
        generation.context.initializeRaggedScratch(kCONTRACT);
        EXPECT_EQ(generation.context.raggedExecutionBatch.hostTokenIds.rawPointer(), pointer);
    }
    EXPECT_FALSE(active.load());
}

TEST(SteppedGenerationStorageTests, ReturnsStorageBeforeScratchInitialization)
{
    // Given a request rejected before scratch initialization, when its generation state is destroyed,
    // then the borrowed allocation is returned without requiring a ragged batch builder.
    Tensor storage({kMAX_TOKENS}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
    void* const pointer = storage.rawPointer();
    std::atomic<bool> active{true};
    {
        LLMRankRuntime::SteppedGeneration generation(active, storage);
        EXPECT_FALSE(generation.context.raggedBatchBuilder.has_value());
        EXPECT_TRUE(storage.isEmpty());
    }
    EXPECT_FALSE(active.load());
    EXPECT_EQ(storage.rawPointer(), pointer);
    EXPECT_TRUE(storage.getOwnMemory());
}
