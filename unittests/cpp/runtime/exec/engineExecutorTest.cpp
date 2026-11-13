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

#include "runtime/exec/engineExecutor.h"
#include <NvInferRuntime.h>
#include <gtest/gtest.h>

using namespace trt_edgellm::rt;

// --------------------------------------------------------------------------
// BindingSnapshot tests — exercise operator== without needing a TRT engine.
// These are restored from the pre-refactor runnerTest.cpp coverage after
// Runner -> EngineExecutor rename.
// --------------------------------------------------------------------------

TEST(EngineExecutorTest, BindingSnapshotEmptyEqual)
{
    EngineExecutor::BindingSnapshot s1;
    EngineExecutor::BindingSnapshot s2;
    EXPECT_TRUE(s1 == s2);
}

TEST(EngineExecutorTest, BindingSnapshotEquality)
{
    EngineExecutor::BindingSnapshot s1;
    EngineExecutor::BindingSnapshot s2;

    nvinfer1::Dims dims2d;
    dims2d.nbDims = 2;
    dims2d.d[0] = 4;
    dims2d.d[1] = 128;

    s1.bindings = {{0x1000, dims2d}, {0x2000, dims2d}};
    s2.bindings = {{0x1000, dims2d}, {0x2000, dims2d}};

    EXPECT_TRUE(s1 == s2);
}

TEST(EngineExecutorTest, BindingSnapshotDifferentAddresses)
{
    EngineExecutor::BindingSnapshot s1;
    EngineExecutor::BindingSnapshot s2;

    nvinfer1::Dims dims2d;
    dims2d.nbDims = 2;
    dims2d.d[0] = 4;
    dims2d.d[1] = 128;

    s1.bindings = {{0x1000, dims2d}, {0x2000, dims2d}};
    s2.bindings = {{0x1000, dims2d}, {0x3000, dims2d}};

    EXPECT_FALSE(s1 == s2);
}

TEST(EngineExecutorTest, BindingSnapshotDifferentNbDims)
{
    EngineExecutor::BindingSnapshot s1;
    EngineExecutor::BindingSnapshot s2;

    nvinfer1::Dims d1;
    d1.nbDims = 2;
    d1.d[0] = 4;
    d1.d[1] = 128;

    nvinfer1::Dims d2;
    d2.nbDims = 3;
    d2.d[0] = 4;
    d2.d[1] = 128;
    d2.d[2] = 1;

    s1.bindings = {{0x1000, d1}};
    s2.bindings = {{0x1000, d2}};

    EXPECT_FALSE(s1 == s2);
}

TEST(EngineExecutorTest, BindingSnapshotDifferentSizes)
{
    EngineExecutor::BindingSnapshot s1;
    EngineExecutor::BindingSnapshot s2;

    nvinfer1::Dims dims;
    dims.nbDims = 1;
    dims.d[0] = 8;

    s1.bindings = {{0x1000, dims}};
    s2.bindings = {{0x1000, dims}, {0x2000, dims}};

    EXPECT_FALSE(s1 == s2);
}

TEST(EngineExecutorTest, BindingSnapshotDifferentShapeValues)
{
    EngineExecutor::BindingSnapshot s1;
    EngineExecutor::BindingSnapshot s2;

    nvinfer1::Dims d1;
    d1.nbDims = 2;
    d1.d[0] = 4;
    d1.d[1] = 128;

    nvinfer1::Dims d2;
    d2.nbDims = 2;
    d2.d[0] = 4;
    d2.d[1] = 1;

    s1.bindings = {{0x1000, d1}};
    s2.bindings = {{0x1000, d2}};

    EXPECT_FALSE(s1 == s2);
}

TEST(EngineExecutorTest, GraphKeyIncludesEngineProfileShapesAndAddresses)
{
    nvinfer1::Dims dims;
    dims.nbDims = 1;
    dims.d[0] = 4;
    EngineExecutor::BindingSnapshot snapshot;
    snapshot.bindings = {{0x1000, dims}};

    size_t const base = computeExecutionGraphKey(0xA0, 0, snapshot);
    EXPECT_NE(base, computeExecutionGraphKey(0xB0, 0, snapshot));
    EXPECT_NE(base, computeExecutionGraphKey(0xA0, 1, snapshot));

    snapshot.bindings[0].first = 0x2000;
    EXPECT_NE(base, computeExecutionGraphKey(0xA0, 0, snapshot));
    snapshot.bindings[0].first = 0x1000;
    snapshot.bindings[0].second.d[0] = 8;
    EXPECT_NE(base, computeExecutionGraphKey(0xA0, 0, snapshot));
}

TEST(EngineExecutorTest, RaggedDimensionRelations)
{
    InferenceDims dims{/*batch=*/3, /*seqLen=*/15, /*kvLen=*/128, /*selectLen=*/3, /*attnMaskSeqLen=*/15,
        /*ropeBatch=*/1, /*packedMaskLen=*/1, /*contextMaskSelectorLen=*/0, /*startIndexLen=*/0,
        /*executionPhaseLen=*/static_cast<int64_t>(ExecutionPhase::kContextPrefill), /*specVerifyLayoutLen=*/0,
        /*skipSoftmaxPrefillScaleLen=*/0, /*skipSoftmaxDecodeScaleLen=*/0,
        /*swaKVCacheModeLen=*/0, /*queryOffsetLen=*/4, /*contextSequenceCount=*/3};
    EXPECT_TRUE(validateRaggedInferenceDims(dims, /*profileIndex=*/0));
    EXPECT_TRUE(validateRaggedInferenceDims(dims, /*profileIndex=*/1));

    dims.selectLen = 15;
    EXPECT_TRUE(validateRaggedInferenceDims(dims, /*profileIndex=*/0));
    dims.selectLen = 16;
    EXPECT_FALSE(validateRaggedInferenceDims(dims, /*profileIndex=*/0));
    EXPECT_TRUE(validateRaggedInferenceDims(dims, /*profileIndex=*/0, /*allowSelectBeyondPhysicalTokens=*/true));
    dims.selectLen = 3;

    dims.attnMaskSeqLen = 1;
    EXPECT_FALSE(validateRaggedInferenceDims(dims, /*profileIndex=*/0));
    dims.attnMaskSeqLen = 15;

    dims.seqLen = 14;
    EXPECT_FALSE(validateRaggedInferenceDims(dims, /*profileIndex=*/0));
    dims.seqLen = 15;
    dims.queryOffsetLen = 3;
    EXPECT_FALSE(validateRaggedInferenceDims(dims, /*profileIndex=*/0));
    dims.queryOffsetLen = 4;
    dims.executionPhaseLen = static_cast<int64_t>(ExecutionPhase::kAutoregressiveDecode);
    dims.contextSequenceCount = 0;
    EXPECT_FALSE(validateRaggedInferenceDims(dims, /*profileIndex=*/1));
    dims.executionPhaseLen = static_cast<int64_t>(ExecutionPhase::kSpecTargetVerify);
    dims.specVerifyLayoutLen = 1;
    dims.selectLen = 15;
    EXPECT_TRUE(validateRaggedInferenceDims(dims, /*profileIndex=*/1));
    dims.executionPhaseLen = static_cast<int64_t>(ExecutionPhase::kDiffusionDenoise);
    dims.specVerifyLayoutLen = 0;
    EXPECT_TRUE(validateRaggedInferenceDims(dims, /*profileIndex=*/0));
    dims.executionPhaseLen = static_cast<int64_t>(ExecutionPhase::kAutoregressiveDecode);
    dims.selectLen = 3;
    dims.seqLen = 3;
    dims.attnMaskSeqLen = 3;
    EXPECT_TRUE(validateRaggedInferenceDims(dims, /*profileIndex=*/1));

    dims.contextSequenceCount = 1;
    EXPECT_FALSE(validateRaggedInferenceDims(dims, /*profileIndex=*/1));
    dims.executionPhaseLen = static_cast<int64_t>(ExecutionPhase::kMixedPrefillDecode);
    EXPECT_FALSE(validateRaggedInferenceDims(dims, /*profileIndex=*/1));
}

// Given an executor whose profile 0 has a dedicated compact context
// When a step's profile and execution phase are routed
// Then only a phase-8 step on profile 0 runs on the dedicated context; phase-1/2 prefills on profile 0, phase-3 or
//      phase-8 steps on profile 1, and every step of an executor without a dedicated context keep the original one
TEST(EngineExecutorTest, OnlyCompactStepsOfTheDedicatedProfileLeaveTheOriginalContext)
{
    auto const dims = [](ExecutionPhase phase) {
        InferenceDims result{};
        result.executionPhaseLen = static_cast<int64_t>(phase);
        return result;
    };
    InferenceDims const compact = dims(ExecutionPhase::kMixedPrefillDecode);
    EXPECT_TRUE(runsOnCompactContext(/*profileIndex=*/0, compact, /*dedicatedProfileIndex=*/0));
    EXPECT_FALSE(runsOnCompactContext(0, dims(ExecutionPhase::kContextPrefill), 0));
    EXPECT_FALSE(runsOnCompactContext(0, dims(ExecutionPhase::kContextChunk), 0));
    EXPECT_FALSE(runsOnCompactContext(1, dims(ExecutionPhase::kAutoregressiveDecode), 0));
    EXPECT_FALSE(runsOnCompactContext(1, compact, 0));
    EXPECT_FALSE(runsOnCompactContext(0, compact, -1));
}

// Given an executor whose profile 0 has a dedicated context and was asked to route context chunks to it
// When a step's profile and execution phase are routed
// Then only a phase-2 context-chunk step on profile 0 is routed there; a phase-1 prefill, a decode step, a chunk on
//      another profile, and every step of an executor without the routing or the context keep the original one
TEST(EngineExecutorTest, OnlyContextChunksOfTheDedicatedProfileAreRoutedWhenAsked)
{
    auto const dims = [](ExecutionPhase phase) {
        InferenceDims result{};
        result.executionPhaseLen = static_cast<int64_t>(phase);
        return result;
    };
    InferenceDims const chunk = dims(ExecutionPhase::kContextChunk);
    EXPECT_TRUE(runsOnContextChunkContext(/*profileIndex=*/0, chunk, /*dedicatedProfileIndex=*/0, true));
    EXPECT_FALSE(runsOnContextChunkContext(0, dims(ExecutionPhase::kContextPrefill), 0, true));
    EXPECT_FALSE(runsOnContextChunkContext(0, dims(ExecutionPhase::kMixedPrefillDecode), 0, true));
    EXPECT_FALSE(runsOnContextChunkContext(1, dims(ExecutionPhase::kAutoregressiveDecode), 0, true));
    EXPECT_FALSE(runsOnContextChunkContext(1, chunk, 0, true));
    EXPECT_FALSE(runsOnContextChunkContext(0, chunk, 0, false));
    EXPECT_FALSE(runsOnContextChunkContext(0, chunk, -1, true));
}
