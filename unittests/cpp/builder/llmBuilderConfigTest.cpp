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

#include "builder/llmBuilder.h"
#include "common/pagedKvTypes.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

using namespace trt_edgellm;
using namespace trt_edgellm::builder;

namespace
{

LLMBuilderConfig makeConfig()
{
    LLMBuilderConfig config;
    config.maxBatchSize = 2;
    config.maxKVCacheCapacity = 256;
    return config;
}

} // namespace

TEST(LLMBuilderConfigTest, DefaultPoolPagesResolveToMinimumActivePages)
{
    LLMBuilderConfig const config = makeConfig();

    EXPECT_EQ(config.resolvedKVPoolPages(), 4);
    EXPECT_EQ(config.toJson().at("max_kv_pool_pages"), 4);
}

TEST(LLMBuilderConfigTest, ExplicitPoolPagesAreSerialized)
{
    LLMBuilderConfig config = makeConfig();
    config.maxKVPoolPages = 9;

    EXPECT_EQ(config.resolvedKVPoolPages(), 9);
    EXPECT_EQ(config.toJson().at("max_kv_pool_pages"), 9);
}

TEST(LLMBuilderConfigTest, PoolPagesBelowMinimumActivePagesAreRejected)
{
    LLMBuilderConfig config = makeConfig();
    config.maxKVPoolPages = 3;

    EXPECT_THROW(config.resolvedKVPoolPages(), std::runtime_error);
    EXPECT_THROW(config.toJson(), std::runtime_error);
}

TEST(LLMBuilderConfigTest, PoolPagesRoundTripThroughJson)
{
    LLMBuilderConfig config = makeConfig();
    config.maxKVPoolPages = 9;

    LLMBuilderConfig const parsed = LLMBuilderConfig::fromJson(config.toJson());

    EXPECT_EQ(parsed.maxKVPoolPages, 9);
    EXPECT_EQ(parsed.resolvedKVPoolPages(), 9);
}

TEST(LLMBuilderConfigTest, RaggedBackendIsTheOnlySerializedRaggedContract)
{
    LLMBuilderConfig config = makeConfig();
    config.maxInputLen = 128;
    config.maxBatchSize = 2;

    Json const json = config.toJson();
    EXPECT_EQ(json.at("ragged_backend"), "entry_padded_compatibility");
    for (char const* field : {"max_num_sequences", "max_query_length", "max_physical_tokens", "recurrent_pool_rows",
             "mixed_step_supported", "indexed_recurrent_state_supported", "indexed_conv_state_supported",
             "token_padding_supported", "multimodal_input_supported", "token_aligned_rope_supported",
             "token_aligned_deepstack_supported", "token_aligned_vision_mask_supported"})
    {
        EXPECT_FALSE(json.contains(field)) << field;
    }

    LLMBuilderConfig const parsed = LLMBuilderConfig::fromJson(json);
    EXPECT_EQ(parsed.raggedBackend, config.raggedBackend);
}

TEST(LLMBuilderConfigTest, RaggedPhysicalCapacityRejectsCheckedProductOverflow)
{
    LLMBuilderConfig config;
    config.maxBatchSize = std::numeric_limits<int32_t>::max();
    config.maxInputLen = 2;
    config.maxKVCacheCapacity = 2;

    EXPECT_THROW(config.resolvedMaxPhysicalTokens(), std::runtime_error);
}

TEST(LLMBuilderConfigTest, DraftRoleResolvesTreeSizedRaggedCapacity)
{
    LLMBuilderConfig config = makeConfig();
    config.maxInputLen = 8;
    config.maxDraftTreeSize = 16;
    config.specDraft = true;

    EXPECT_EQ(config.resolvedMaxQueryLength(), 16);
    EXPECT_EQ(config.resolvedMaxPhysicalTokens(), 32);
    EXPECT_EQ(config.raggedMultiTokenGenerationProfileRange(config.resolvedRoleQueryLength()).max.physicalTokens, 32);
}

TEST(LLMBuilderConfigTest, BaseRoleResolvesVerifyTreeSizedRaggedCapacity)
{
    LLMBuilderConfig config = makeConfig();
    config.maxInputLen = 8;
    config.maxVerifyTreeSize = 24;
    config.specBase = true;

    EXPECT_EQ(config.resolvedMaxQueryLength(), 24);
    EXPECT_EQ(config.resolvedMaxPhysicalTokens(), 48);
}

TEST(LLMBuilderConfigTest, RoleProfileDoesNotExpandToLargerPrefillWidth)
{
    LLMBuilderConfig config = makeConfig();
    config.maxInputLen = 128;
    config.maxDraftTreeSize = 16;
    config.specDraft = true;

    EXPECT_EQ(config.resolvedRoleQueryLength(), 16);
    EXPECT_EQ(config.resolvedMaxQueryLength(), 128);
    EXPECT_EQ(config.raggedMultiTokenGenerationProfileRange(config.resolvedRoleQueryLength()).max.physicalTokens, 32);
    EXPECT_EQ(config.resolvedMaxPhysicalTokens(), 256);
}

TEST(LLMBuilderConfigTest, ModelRoleOverrideResolvesCanvasSizedRaggedCapacity)
{
    LLMBuilderConfig config = makeConfig();
    config.maxInputLen = 8;
    config.maxQueryLength = 32;

    EXPECT_EQ(config.resolvedMaxQueryLength(), 32);
    EXPECT_EQ(config.resolvedMaxPhysicalTokens(), 64);
    EXPECT_EQ(config.raggedMultiTokenGenerationProfileRange(config.resolvedRoleQueryLength()).max.physicalTokens, 64);
}

TEST(LLMBuilderConfigTest, RoleAwareRaggedCapacityRejectsCheckedProductOverflow)
{
    LLMBuilderConfig config = makeConfig();
    config.maxBatchSize = std::numeric_limits<int32_t>::max();
    config.maxInputLen = 1;
    config.maxQueryLength = 2;

    EXPECT_THROW(config.resolvedMaxPhysicalTokens(), std::runtime_error);
}

TEST(LLMBuilderConfigTest, RaggedPrefillAndDecodeProfileRangesCoverEveryBindingClass)
{
    LLMBuilderConfig config = makeConfig();
    config.maxInputLen = 128;

    RaggedProfileRange const prefill = config.raggedPrefillProfileRange();
    EXPECT_EQ(prefill.min.numSequences, 1);
    EXPECT_EQ(prefill.min.physicalTokens, 1);
    EXPECT_EQ(prefill.min.queryOffsets, 2);
    EXPECT_EQ(prefill.min.logitsRows, 1);
    EXPECT_EQ(prefill.opt.numSequences, 2);
    EXPECT_EQ(prefill.opt.physicalTokens, 128);
    EXPECT_EQ(prefill.opt.queryOffsets, 3);
    EXPECT_EQ(prefill.opt.logitsRows, 2);
    EXPECT_EQ(prefill.max.numSequences, 2);
    EXPECT_EQ(prefill.max.physicalTokens, 256);
    EXPECT_EQ(prefill.max.queryOffsets, 3);
    EXPECT_EQ(prefill.max.logitsRows, 2);

    RaggedProfileRange const decode = config.raggedDecodeProfileRange();
    EXPECT_EQ(decode.min.physicalTokens, 1);
    EXPECT_EQ(decode.opt.physicalTokens, 2);
    EXPECT_EQ(decode.max.physicalTokens, 2);
    EXPECT_EQ(decode.max.numSequences, 2);
    EXPECT_EQ(decode.max.queryOffsets, 3);
    EXPECT_EQ(decode.max.logitsRows, 2);
}

TEST(LLMBuilderConfigTest, RaggedMultiTokenGenerationProfileCoversWholeBatchTree)
{
    LLMBuilderConfig config = makeConfig();

    RaggedProfileRange const generation = config.raggedMultiTokenGenerationProfileRange(60);
    EXPECT_EQ(generation.min.physicalTokens, 1);
    EXPECT_EQ(generation.opt.physicalTokens, 120);
    EXPECT_EQ(generation.max.physicalTokens, 120);
    EXPECT_EQ(generation.max.numSequences, 2);
    EXPECT_EQ(generation.max.queryOffsets, 3);
    EXPECT_EQ(generation.max.logitsRows, 120);
}

TEST(LLMBuilderConfigTest, RaggedMultiTokenGenerationProfileRejectsInvalidCapacity)
{
    LLMBuilderConfig config = makeConfig();

    EXPECT_THROW(config.raggedMultiTokenGenerationProfileRange(0), std::runtime_error);
    config.maxBatchSize = std::numeric_limits<int32_t>::max();
    EXPECT_THROW(config.raggedMultiTokenGenerationProfileRange(2), std::runtime_error);
}

TEST(LLMBuilderConfigTest, PoolPagesRejectDerivedVIdOverflow)
{
    LLMBuilderConfig config = makeConfig();
    config.maxKVPoolPages = rt::kMAX_KV_POOL_PAGES + 1;

    EXPECT_THROW(config.resolvedKVPoolPages(), std::runtime_error);
}

TEST(LLMBuilderConfigTest, KVCapacityRejectsPageAlignmentOverflow)
{
    LLMBuilderConfig config;
    config.maxBatchSize = 1;
    config.maxKVCacheCapacity = static_cast<int64_t>(rt::kMAX_KV_CACHE_CAPACITY) + 1;

    EXPECT_THROW(config.resolvedKVPoolPages(), std::runtime_error);
}

TEST(LLMBuilderConfigTest, MinimumActivePagesRejectNarrowingOverflow)
{
    LLMBuilderConfig config;
    config.maxBatchSize = std::numeric_limits<int32_t>::max();
    config.maxKVCacheCapacity = rt::kMAX_KV_CACHE_CAPACITY;

    EXPECT_THROW(config.resolvedKVPoolPages(), std::runtime_error);
}

TEST(LLMBuilderConfigTest, ZeroSwaPageBudgetAutoSizesAndPersists)
{
    builder::LLMBuilderConfig config;
    config.maxBatchSize = 3;
    int32_t const expectedPages
        = static_cast<int32_t>(rt::computeMinimumSwaPoolPages(config.maxBatchSize, /*slidingWindowCapacity=*/257));

    EXPECT_EQ(config.resolveNumSwaPages(/*slidingWindowCapacity=*/257), expectedPages);
    EXPECT_EQ(config.numSwaPages, expectedPages);

    Json const json = config.toJson();
    EXPECT_EQ(json.at("num_swa_pages").get<int64_t>(), expectedPages);

    builder::LLMBuilderConfig const restored = builder::LLMBuilderConfig::fromJson(json);
    EXPECT_EQ(restored.numSwaPages, config.numSwaPages);
}

TEST(LLMBuilderConfigTest, ExplicitSwaPageBudgetStillValidatesActivePrivateFloor)
{
    builder::LLMBuilderConfig config;
    config.maxBatchSize = 2;
    int32_t const minimumPages
        = static_cast<int32_t>(rt::computeMinimumSwaPoolPages(/*maxBatchSize=*/2, /*slidingWindowCapacity=*/129));

    config.numSwaPages = minimumPages;
    EXPECT_EQ(config.resolveNumSwaPages(/*slidingWindowCapacity=*/129), minimumPages);

    config.numSwaPages = minimumPages - 1;
    EXPECT_THROW(config.resolveNumSwaPages(/*slidingWindowCapacity=*/129), std::invalid_argument);
}

TEST(LLMBuilderConfigTest, SwaCapableProfileOptimizesForBoundedAndCoversFullPages)
{
    LLMBuilderConfig config;
    config.maxBatchSize = 1;
    config.maxKVCacheCapacity = 8192;
    constexpr int32_t kWINDOW_SIZE = 129;
    int32_t const boundedPages = config.resolveNumSwaPages(kWINDOW_SIZE);
    int64_t const fullPages = config.resolvedKVPoolPages();
    ASSERT_LT(boundedPages, fullPages);

    EXPECT_EQ(
        config.resolveKVPoolPageProfile(kWINDOW_SIZE), (std::array<int64_t, 3>{boundedPages, boundedPages, fullPages}));
    EXPECT_EQ(config.resolveKVPoolPageProfile(/*kvCacheCapacity=*/0),
        (std::array<int64_t, 3>{fullPages, fullPages, fullPages}));
}

TEST(LLMBuilderConfigTest, SwaCapableProfileOptimizesForFullWhenBoundedDoesNotSaveMemory)
{
    LLMBuilderConfig config;
    config.maxBatchSize = 1;
    config.maxKVCacheCapacity = 256;
    constexpr int32_t kWINDOW_SIZE = 129;
    int32_t const boundedPages = config.resolveNumSwaPages(kWINDOW_SIZE);
    int64_t const fullPages = config.resolvedKVPoolPages();
    ASSERT_GT(boundedPages, fullPages);

    EXPECT_EQ(
        config.resolveKVPoolPageProfile(kWINDOW_SIZE), (std::array<int64_t, 3>{fullPages, fullPages, boundedPages}));

    config.maxKVCacheCapacity = static_cast<int64_t>(boundedPages) * rt::kTOKENS_PER_PAGE;
    EXPECT_EQ(config.resolvedKVPoolPages(), boundedPages);
    EXPECT_EQ(config.resolveKVPoolPageProfile(kWINDOW_SIZE),
        (std::array<int64_t, 3>{boundedPages, boundedPages, boundedPages}));
}

TEST(LLMBuilderConfigTest, UnifiedDFlashTypeDetectsVersionTwoDraft)
{
    Json const config = {
        {"spec_decode_type", "dflash"},
        {"engine_role", "draft"},
        {"dflash_config", {{"version", 2}}},
    };

    EXPECT_TRUE(isDFlashV2DraftConfig(config));
}

TEST(LLMBuilderConfigTest, UnifiedDFlashTypeDoesNotPromoteVersionOneOrBase)
{
    Json const versionOne = {
        {"spec_decode_type", "dflash"},
        {"engine_role", "draft"},
        {"dflash_config", {{"version", 1}}},
    };
    Json const base = {
        {"spec_decode_type", "dflash"},
        {"engine_role", "base"},
        {"dflash_config", {{"version", 2}}},
    };

    EXPECT_FALSE(isDFlashV2DraftConfig(versionOne));
    EXPECT_FALSE(isDFlashV2DraftConfig(base));
}
