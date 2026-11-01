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

// This file is compiled only on TRT-RTX builds (guarded in CMake). Path resolution touches no TRT APIs;
// a full serialize/deserialize round-trip needs a TRT-RTX engine and GPU and belongs in an integration test.
#include "runtime/exec/rtxRuntimeCache.h"

#include <filesystem>
#include <gtest/gtest.h>

using trt_edgellm::rt::RtxRuntimeCache;

TEST(RtxRuntimeCacheTest, DefaultPathSharedForBaseAndDraft)
{
    auto const base = RtxRuntimeCache::resolveCachePath("/engines/spec/spec_base.engine", std::nullopt);
    auto const draft = RtxRuntimeCache::resolveCachePath("/engines/spec/spec_draft.engine", std::nullopt);
    EXPECT_EQ(base, draft);
    EXPECT_EQ(base, std::filesystem::path("/engines/spec/runtime.rtxcache"));
}

TEST(RtxRuntimeCacheTest, ExplicitPathSharedAcrossEngineDirectories)
{
    std::filesystem::path const cachePath{"/shared/all-models.rtxcache"};
    auto const modelA = RtxRuntimeCache::resolveCachePath("/models/a/llm.engine", cachePath);
    auto const modelB = RtxRuntimeCache::resolveCachePath("/models/b/llm.engine", cachePath);
    EXPECT_EQ(modelA, cachePath);
    EXPECT_EQ(modelB, cachePath);
}

TEST(RtxRuntimeCacheTest, DefaultPathSeparatedAcrossEngineDirectories)
{
    auto const modelA = RtxRuntimeCache::resolveCachePath("/models/a/llm.engine", std::nullopt);
    auto const modelB = RtxRuntimeCache::resolveCachePath("/models/b/llm.engine", std::nullopt);
    EXPECT_NE(modelA, modelB);
}
