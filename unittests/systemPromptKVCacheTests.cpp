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

#include "runtime/state/systemPromptKVCache.h"
#include <gtest/gtest.h>

using namespace trt_edgellm;

namespace
{

// Build a SystemPromptKVCache with a tokenized prompt of the given length. `numKvLayers` controls
// how many (empty, metadata-only) kvCacheLayers entries are present — 0 models a pure-recurrent
// model with no attention layers, matching HybridCacheManager::captureKVCache's return for that case.
SystemPromptKVCache makeCache(size_t tokenizedPromptLength, int32_t numKvLayers)
{
    SystemPromptKVCache cache;
    cache.tokenizedPrompt.assign(tokenizedPromptLength, tokenizer::Rank{0});
    cache.kvCacheLayers.resize(numKvLayers);
    return cache;
}

std::vector<int32_t> makeInputIds(size_t length)
{
    std::vector<int32_t> ids(length);
    for (size_t i = 0; i < length; ++i)
    {
        ids[i] = static_cast<int32_t>(i);
    }
    return ids;
}

} // namespace

// Regression for review finding #1 (merge blocker): the cached token length must come from the
// tokenized prompt, not from a KV-tensor's physical shape. This case is H < S in the old (buggy)
// terms — the cached length (H, here driven by tokenizedPrompt) is smaller than the input length (S).
TEST(SystemPromptKVCacheTests, ReuseLengthSmallerThanInputLength)
{
    auto const cache = makeCache(/*tokenizedPromptLength=*/8, /*numKvLayers=*/2);
    auto const inputIds = makeInputIds(20);

    auto const result = computeSystemPromptReuse(cache, inputIds);

    // Reuses cachedLength - 1 tokens; the rest come from the fresh input.
    EXPECT_EQ(result.reuseKVCacheLength, 7);
    EXPECT_EQ(result.effectivePrefillLength, 20 - 7);
    ASSERT_EQ(result.tokenIds.size(), static_cast<size_t>(20 - 7));
    EXPECT_EQ(result.tokenIds.front(), 7);
    EXPECT_EQ(result.tokenIds.back(), 19);
}

// Cached length close to the input length (still valid: cachedLength < inputIds.size()).
TEST(SystemPromptKVCacheTests, ReuseLengthOneLessThanInputLength)
{
    auto const cache = makeCache(/*tokenizedPromptLength=*/19, /*numKvLayers=*/3);
    auto const inputIds = makeInputIds(20);

    auto const result = computeSystemPromptReuse(cache, inputIds);

    EXPECT_EQ(result.reuseKVCacheLength, 18);
    EXPECT_EQ(result.effectivePrefillLength, 2);
    ASSERT_EQ(result.tokenIds.size(), 2u);
    EXPECT_EQ(result.tokenIds.front(), 18);
    EXPECT_EQ(result.tokenIds.back(), 19);
}

// Pure-recurrent models have no attention layers, so kvCacheLayers is empty. The cached length must
// still be derivable purely from the tokenized prompt (review finding #1's "no first KV layer from
// which to infer a length" case) — deliberately supported here, not rejected.
TEST(SystemPromptKVCacheTests, PureRecurrentModelWithNoKvLayersIsSupported)
{
    auto const cache = makeCache(/*tokenizedPromptLength=*/5, /*numKvLayers=*/0);
    ASSERT_TRUE(cache.kvCacheLayers.empty());
    auto const inputIds = makeInputIds(10);

    auto const result = computeSystemPromptReuse(cache, inputIds);

    EXPECT_EQ(result.reuseKVCacheLength, 4);
    EXPECT_EQ(result.effectivePrefillLength, 6);
}

// Cached length must not reach or exceed the input length — this is rejected (falls back to full
// prefill via the caller not calling into system-prompt reuse for such prompts up front, but here we
// verify the guard directly).
TEST(SystemPromptKVCacheTests, RejectsCachedLengthGreaterOrEqualToInputLength)
{
    auto const inputIds = makeInputIds(8);

    // cachedLength == inputLength
    EXPECT_THROW(computeSystemPromptReuse(makeCache(8, 1), inputIds), std::exception);
    // cachedLength > inputLength
    EXPECT_THROW(computeSystemPromptReuse(makeCache(9, 1), inputIds), std::exception);
}

TEST(SystemPromptKVCacheTests, RejectsEmptyTokenizedPrompt)
{
    auto const inputIds = makeInputIds(8);
    EXPECT_THROW(computeSystemPromptReuse(makeCache(0, 1), inputIds), std::exception);
}
