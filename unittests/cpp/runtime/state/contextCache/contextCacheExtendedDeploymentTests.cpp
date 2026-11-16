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

// KV cache reuse trigger/non-trigger decision tests.
// Verifies, through the ContextCacheManager, that reuse is triggered (reuseTokenLength > 0) or
// suppressed (reuseTokenLength == 0) under specific conditions for each deployment mode:
// vanilla, speculative (DFlash/JetSpec/EAGLE/Gemma4 MTP), FP8 KV, and hybrid MTP.
// All tests are host-only using the manager directly — no GPU required.

#include "runtime/state/contextCache/contextCacheDeployment.h"
#include "runtime/state/contextCache/contextCacheManager.h"
#include "runtime/state/contextCache/reusePlan.h"
#include "runtime/state/contextCache/specReuseContract.h"

#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"
#include "runtime/config/llmEngineConfig.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

using namespace trt_edgellm::rt;

namespace
{

constexpr int32_t kPAGE_SIZE = 4;
constexpr BlockHash kHASH_A{0x1111111111111111ULL, 0xAAAAAAAAAAAAAAAAULL};
constexpr BlockHash kHASH_B{0x2222222222222222ULL, 0xBBBBBBBBBBBBBBBBULL};
constexpr BlockHash kHASH_C{0x3333333333333333ULL, 0xCCCCCCCCCCCCCCCCULL};
constexpr BlockHash kHASH_D{0x4444444444444444ULL, 0xDDDDDDDDDDDDDDDDULL};

SpecReuseContract dflashContract()
{
    return SpecReuseContract{/*ownsPagedSpecState=*/true, /*futureDependencyTokens=*/0};
}

SpecReuseContract eagleContract()
{
    return SpecReuseContract{/*ownsPagedSpecState=*/true, /*futureDependencyTokens=*/1};
}

SpecReuseContract gemma4MtpContract()
{
    return SpecReuseContract{/*ownsPagedSpecState=*/false, /*futureDependencyTokens=*/0};
}

SpecReuseContract hybridMtpContract()
{
    return SpecReuseContract{/*ownsPagedSpecState=*/true, /*futureDependencyTokens=*/1};
}

CacheRequestLease takeLease(AcquireResult& result)
{
    if (!result.lease.has_value())
    {
        throw std::runtime_error("Test context cache acquisition failed");
    }
    return std::move(*result.lease);
}

void publishVanilla(ContextCacheManager& manager, std::vector<BlockHash> const& hashes)
{
    int32_t const tokenCount = static_cast<int32_t>(hashes.size()) * kPAGE_SIZE;
    AcquireResult cold = manager.acquireVanilla(hashes, tokenCount);
    CacheRequestLease lease = takeLease(cold);
    manager.publish(lease, PublishRequest{hashes, tokenCount});
    lease.release();
}

void publishSpec(ContextCacheManager& manager, std::vector<BlockHash> const& hashes)
{
    int32_t const tokenCount = static_cast<int32_t>(hashes.size()) * kPAGE_SIZE;
    AcquireResult cold = manager.acquireSpec(hashes, tokenCount);
    CacheRequestLease lease = takeLease(cold);
    manager.publish(lease, PublishRequest{hashes, tokenCount});
    lease.release();
}

LLMEngineConfig makeAttentionConfig(int32_t attentionLayers = 4)
{
    LLMEngineConfig config;
    config.hiddenSize = 1024;
    config.numDecoderLayers = attentionLayers;
    config.numAttentionLayers = attentionLayers;
    config.numKVHeads = 8;
    config.headDim = 128;
    config.rotaryDim = 128;
    config.maxSupportedBatchSize = 2;
    config.maxKVCacheCapacity = 512;
    int64_t const minimumActivePages
        = computeMinimumKvPoolPages(config.maxSupportedBatchSize, config.maxKVCacheCapacity);
    ELLM_CHECK(minimumActivePages <= kMAX_KV_POOL_PAGES, "Test KV pool page count must fit int32.");
    config.kvPoolPages = static_cast<int32_t>(minimumActivePages);
    config.kvCacheDtype = trt_edgellm::DataType::kHALF;
    config.layerTypes.assign(attentionLayers, HybridCacheManager::LayerType::kAttention);
    config.kvLayerConfigs.assign(attentionLayers, KVLayerConfig{config.numKVHeads, config.headDim});
    return config;
}

LLMEngineConfig makeHybridConfig(int32_t attentionLayers = 2, int32_t recurrentLayers = 2)
{
    LLMEngineConfig config = makeAttentionConfig(attentionLayers);
    config.numDecoderLayers = attentionLayers + recurrentLayers;
    config.numLinearAttnLayers = recurrentLayers;
    config.layerTypes.insert(config.layerTypes.end(), recurrentLayers, HybridCacheManager::LayerType::kMamba);
    config.recurrentStateNumHeads = 16;
    config.recurrentStateHeadDim = 32;
    config.recurrentStateSize = 64;
    config.convDim = 256;
    config.convKernel = 4;
    config.recurrentStateDtype = trt_edgellm::DataType::kHALF;
    config.convStateDtype = trt_edgellm::DataType::kHALF;
    return config;
}

DeploymentConfig makeDFlashDeployment()
{
    DeploymentConfig deployment;
    deployment.base = makeAttentionConfig();
    deployment.base.specDecodeType = SpecDecodeMode::kDFlash;
    deployment.base.isSpecDecodeBase = true;
    deployment.draft = makeAttentionConfig(/*attentionLayers=*/2);
    deployment.draft->specDecodeType = SpecDecodeMode::kDFlash;
    deployment.draft->baseModelHiddenSize = deployment.base.hiddenSize;
    SpecDecodeConfig specConfig;
    specConfig.draftTopK = 1;
    specConfig.draftSteps = 1;
    specConfig.verifyTokenCount = 8;
    specConfig.draftBlockSize = 8;
    deployment.specConfig = specConfig;
    return deployment;
}

DeploymentConfig makeJetSpecDeployment()
{
    DeploymentConfig deployment = makeDFlashDeployment();
    deployment.base.specDecodeType = SpecDecodeMode::kJetSpec;
    deployment.draft->specDecodeType = SpecDecodeMode::kJetSpec;
    return deployment;
}

DeploymentConfig makeHybridMtpDeployment()
{
    DeploymentConfig deployment;
    deployment.base = makeHybridConfig();
    deployment.base.specDecodeType = SpecDecodeMode::kMTP;
    deployment.base.isSpecDecodeBase = true;
    deployment.draft = makeAttentionConfig(/*attentionLayers=*/2);
    deployment.draft->specDecodeType = SpecDecodeMode::kMTP;
    deployment.draft->hasOwnKVCache = true;
    deployment.draft->sharesTargetKV = false;
    deployment.draft->baseModelHiddenSize = deployment.base.hiddenSize;
    SpecDecodeConfig specConfig;
    specConfig.draftTopK = 2;
    specConfig.draftSteps = 3;
    specConfig.verifyTokenCount = 5;
    deployment.specConfig = specConfig;
    return deployment;
}

} // namespace

// =============================================================================
// Deployment admission validation
// =============================================================================

TEST(ContextCacheExtendedDeploymentTests, DFlashDeploymentAdmission)
{
    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(makeDFlashDeployment());
    EXPECT_EQ(profile.baseStateKind, ContextCacheModelStateKind::kAttentionOnly);
    ASSERT_TRUE(profile.specReuseContract.has_value());
    EXPECT_TRUE(profile.specReuseContract->ownsPagedSpecState);
    EXPECT_EQ(profile.specReuseContract->futureDependencyTokens, 0);
}

TEST(ContextCacheExtendedDeploymentTests, JetSpecDeploymentAdmission)
{
    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(makeJetSpecDeployment());
    EXPECT_EQ(profile.baseStateKind, ContextCacheModelStateKind::kAttentionOnly);
    ASSERT_TRUE(profile.specReuseContract.has_value());
    EXPECT_TRUE(profile.specReuseContract->ownsPagedSpecState);
    EXPECT_EQ(profile.specReuseContract->futureDependencyTokens, 0);
}

TEST(ContextCacheExtendedDeploymentTests, HybridMtpDeploymentAdmission)
{
    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(makeHybridMtpDeployment());
    EXPECT_EQ(profile.baseStateKind, ContextCacheModelStateKind::kHybrid);
    ASSERT_TRUE(profile.specReuseContract.has_value());
    EXPECT_TRUE(profile.specReuseContract->ownsPagedSpecState);
    EXPECT_EQ(profile.specReuseContract->futureDependencyTokens, 1);
}

TEST(ContextCacheExtendedDeploymentTests, Fp8KvDFlashDeploymentAdmission)
{
    DeploymentConfig deployment = makeDFlashDeployment();
    deployment.base.kvCacheDtype = trt_edgellm::DataType::kFP8;
    deployment.draft->kvCacheDtype = trt_edgellm::DataType::kFP8;
    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(deployment);
    ASSERT_TRUE(profile.specReuseContract.has_value());
    EXPECT_EQ(profile.specReuseContract->futureDependencyTokens, 0);
}

TEST(ContextCacheExtendedDeploymentTests, Fp8KvJetSpecDeploymentAdmission)
{
    DeploymentConfig deployment = makeJetSpecDeployment();
    deployment.base.kvCacheDtype = trt_edgellm::DataType::kFP8;
    deployment.draft->kvCacheDtype = trt_edgellm::DataType::kFP8;
    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(deployment);
    ASSERT_TRUE(profile.specReuseContract.has_value());
    EXPECT_EQ(profile.specReuseContract->futureDependencyTokens, 0);
}

TEST(ContextCacheExtendedDeploymentTests, BlockDraftModesHaveZeroReplayPages)
{
    EXPECT_EQ(replayPages(dflashContract(), kPAGE_SIZE), 0);
    EXPECT_EQ(replayPages(dflashContract(), 128), 0);
    EXPECT_EQ(replayPages(eagleContract(), kPAGE_SIZE), 1);
    EXPECT_EQ(replayPages(eagleContract(), 128), 1);
    EXPECT_EQ(replayPages(hybridMtpContract(), kPAGE_SIZE), 1);
    EXPECT_EQ(replayPages(gemma4MtpContract(), kPAGE_SIZE), 0);
}

// =============================================================================
// Vanilla: reuse triggered vs not triggered
// =============================================================================

TEST(ContextCacheExtendedDeploymentTests, VanillaColdMissReportsNoReuse)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 0, 0, 0}, 4);

    AcquireResult result = manager.acquireVanilla({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 0);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kNoReusablePrefix);
    takeLease(result).release();
}

TEST(ContextCacheExtendedDeploymentTests, VanillaHitTriggersReuse)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 0, 0, 0}, 4);
    publishVanilla(manager, {kHASH_A, kHASH_B});

    AcquireResult result = manager.acquireVanilla({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 2 * kPAGE_SIZE);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kStandard);
    EXPECT_FALSE(result.plan.basePageBindings.empty());
    takeLease(result).release();
}

TEST(ContextCacheExtendedDeploymentTests, VanillaDisjointPrefixDoesNotTriggerReuse)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 0, 0, 0}, 4);
    publishVanilla(manager, {kHASH_A, kHASH_B});

    AcquireResult result = manager.acquireVanilla({kHASH_C, kHASH_D}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 0);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kNoReusablePrefix);
    takeLease(result).release();
}

TEST(ContextCacheExtendedDeploymentTests, VanillaPartialPrefixMatchReusesOnlyMatchedBlocks)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 0, 0, 0}, 4);
    publishVanilla(manager, {kHASH_A, kHASH_B});

    // Request [A, C, D]: only the leading A matches.
    AcquireResult result = manager.acquireVanilla({kHASH_A, kHASH_C, kHASH_D}, 3 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, kPAGE_SIZE);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kStandard);
    takeLease(result).release();
}

TEST(ContextCacheExtendedDeploymentTests, VanillaExactBlockAlignedInputRewinds)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 0, 0, 0}, 4);
    publishVanilla(manager, {kHASH_A, kHASH_B});

    // Exact block-aligned input: the last block is rewound so the token boundary can be recomputed.
    AcquireResult result = manager.acquireVanilla({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kFullInputRewind);
    EXPECT_EQ(result.plan.reuseTokenLength, kPAGE_SIZE);
    takeLease(result).release();
}

TEST(ContextCacheExtendedDeploymentTests, VanillaBypassPolicySuppressesReuse)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 0, 0, 0}, 4);
    publishVanilla(manager, {kHASH_A, kHASH_B});

    AcquireResult result
        = manager.acquireVanilla({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1, ContextCacheLookupPolicy::kBypass);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 0);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kNoReusablePrefix);
    takeLease(result).release();
}

// =============================================================================
// DFlash speculative: reuse triggered vs not triggered
// =============================================================================

TEST(ContextCacheExtendedDeploymentTests, DFlashColdMissReportsNoReuse)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, 0, 0}, 4, dflashContract());

    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 0);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kNoReusablePrefix);
    takeLease(result).release();
}

TEST(ContextCacheExtendedDeploymentTests, DFlashHitTriggersFullPrefixReuse)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, 0, 0}, 4, dflashContract());
    publishSpec(manager, {kHASH_A, kHASH_B});

    // futureDependencyTokens=0: the full matched prefix is reusable without replay.
    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 2 * kPAGE_SIZE);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kStandard);
    CacheRequestLease lease = takeLease(result);
    EXPECT_EQ(lease.basePages().size(), 3U);
    EXPECT_EQ(lease.draftPages().size(), 3U);
    lease.release();
}

TEST(ContextCacheExtendedDeploymentTests, DFlashDisjointPrefixDoesNotTriggerReuse)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, 0, 0}, 4, dflashContract());
    publishSpec(manager, {kHASH_A, kHASH_B});

    AcquireResult result = manager.acquireSpec({kHASH_C, kHASH_D}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 0);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kNoReusablePrefix);
    takeLease(result).release();
}

// =============================================================================
// JetSpec speculative: same contract as DFlash (futureDependencyTokens=0)
// =============================================================================

TEST(ContextCacheExtendedDeploymentTests, JetSpecColdMissReportsNoReuse)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, 0, 0}, 4, dflashContract());

    AcquireResult result = manager.acquireSpec({kHASH_A}, kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 0);
    takeLease(result).release();
}

TEST(ContextCacheExtendedDeploymentTests, JetSpecHitTriggersFullPrefixReuse)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, 0, 0}, 4, dflashContract());
    publishSpec(manager, {kHASH_A});

    AcquireResult result = manager.acquireSpec({kHASH_A}, kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, kPAGE_SIZE);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kStandard);
    takeLease(result).release();
}

// =============================================================================
// EAGLE speculative: futureDependencyTokens=1 trims reuse by one replay page
// =============================================================================

TEST(ContextCacheExtendedDeploymentTests, EagleColdMissReportsNoReuse)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, 0, 0}, 4, eagleContract());

    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 0);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kNoReusablePrefix);
    takeLease(result).release();
}

TEST(ContextCacheExtendedDeploymentTests, EagleHitTrimsReuseByOneReplayPage)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, 0, 0}, 4, eagleContract());
    publishSpec(manager, {kHASH_A, kHASH_B});

    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    // EAGLE replays the last page: reuseTokenLength = (matchedBlocks - 1) * pageSize.
    EXPECT_EQ(result.plan.reuseTokenLength, (2 - 1) * kPAGE_SIZE);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kStandard);
    takeLease(result).release();
}

TEST(ContextCacheExtendedDeploymentTests, EagleSingleBlockHitCannotReuse)
{
    // With one matched block, EAGLE's replay consumes the entire match: reuseTokenLength = 0.
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, 0, 0}, 4, eagleContract());
    publishSpec(manager, {kHASH_A});

    AcquireResult result = manager.acquireSpec({kHASH_A}, kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 0);
    takeLease(result).release();
}

// =============================================================================
// Gemma4 MTP: shared-target-KV (ownsPagedSpecState=false, futureDependencyTokens=0)
// =============================================================================

TEST(ContextCacheExtendedDeploymentTests, Gemma4MTPDeploymentAdmission)
{
    DeploymentConfig deployment;
    deployment.base = makeAttentionConfig();
    deployment.base.specDecodeType = SpecDecodeMode::kGemma4MTP;
    deployment.base.isSpecDecodeBase = true;
    deployment.draft = makeAttentionConfig();
    deployment.draft->specDecodeType = SpecDecodeMode::kGemma4MTP;
    deployment.draft->hasOwnKVCache = false;
    deployment.draft->sharesTargetKV = true;
    SpecDecodeConfig specConfig;
    specConfig.draftTopK = 1;
    specConfig.draftSteps = 3;
    specConfig.verifyTokenCount = 5;
    deployment.specConfig = specConfig;

    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(deployment);
    ASSERT_TRUE(profile.specReuseContract.has_value());
    EXPECT_FALSE(profile.specReuseContract->ownsPagedSpecState);
    EXPECT_EQ(profile.specReuseContract->futureDependencyTokens, 0);
}

TEST(ContextCacheExtendedDeploymentTests, Gemma4MTPColdMissReportsNoReuse)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 0, 0, 0}, 4, gemma4MtpContract());

    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 0);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kNoReusablePrefix);
    takeLease(result).release();
}

TEST(ContextCacheExtendedDeploymentTests, Gemma4MTPHitTriggersReuseWithNoDraftPages)
{
    // ownsPagedSpecState=false: only base pages are allocated; draft pages remain empty.
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 0, 0, 0}, 4, gemma4MtpContract());
    publishSpec(manager, {kHASH_A, kHASH_B});

    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 2 * kPAGE_SIZE);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kStandard);
    CacheRequestLease lease = takeLease(result);
    EXPECT_EQ(lease.basePages().size(), 3U);
    EXPECT_TRUE(lease.draftPages().empty());
    lease.release();
}

TEST(ContextCacheExtendedDeploymentTests, Gemma4MTPDisjointPrefixDoesNotReuse)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 0, 0, 0}, 4, gemma4MtpContract());
    publishSpec(manager, {kHASH_A, kHASH_B});

    AcquireResult result = manager.acquireSpec({kHASH_C, kHASH_D}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 0);
    takeLease(result).release();
}

// =============================================================================
// DFlash speculative grow + reuse lifecycle
// =============================================================================

TEST(ContextCacheExtendedDeploymentTests, DFlashGrowAndPublishTriggersReuseOnLongerPrefix)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{12, 12, 0, 0}, 4, dflashContract());

    AcquireResult cold = manager.acquireSpec({kHASH_A}, kPAGE_SIZE);
    CacheRequestLease lease = takeLease(cold);
    EXPECT_TRUE(manager.growSpecPages(lease, 2, 2));
    EXPECT_EQ(lease.basePages().size(), 3U);
    EXPECT_EQ(lease.draftPages().size(), 3U);
    manager.publish(lease, PublishRequest{{kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE});
    lease.release();

    AcquireResult hit = manager.acquireSpec({kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE + 1);
    ASSERT_EQ(hit.status, AcquireStatus::kAcquired);
    EXPECT_EQ(hit.plan.reuseTokenLength, 3 * kPAGE_SIZE);
    EXPECT_EQ(hit.plan.kind, ReusePlanKind::kStandard);
    takeLease(hit).release();
}

// =============================================================================
// Hybrid MTP: checkpoint-based reuse trigger/non-trigger
// =============================================================================

TEST(ContextCacheExtendedDeploymentTests, HybridMtpColdMissWithNoCandidatesDoesNotReuse)
{
    int32_t const snapshotSlots = 4;
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, snapshotSlots, snapshotSlots}, 4, hybridMtpContract());

    std::vector<HybridCheckpointCandidate> const noCandidates;
    AcquireResult result = manager.acquireHybridMtp(noCandidates, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 0);
    takeLease(result).release();
}

TEST(ContextCacheExtendedDeploymentTests, HybridMtpPublishedCheckpointTriggersReuse)
{
    int32_t const snapshotSlots = 4;
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, snapshotSlots, snapshotSlots}, 4, hybridMtpContract());

    // Publish a hybrid checkpoint.
    std::vector<HybridCheckpointCandidate> const noCandidates;
    AcquireResult cold = manager.acquireHybridMtp(noCandidates, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE);
    CacheRequestLease lease = takeLease(cold);
    std::optional<HybridSnapshotReservation> reservation = manager.reserveHybridSnapshots(lease, true);
    ASSERT_TRUE(reservation.has_value());
    HybridPublishRequest hybridPub;
    hybridPub.fullBlockHashes = {kHASH_A};
    hybridPub.checkpoint = HybridCheckpointKey{kHASH_B, 2 * kPAGE_SIZE};
    hybridPub.snapshots = *reservation;
    PublishResult pubResult = manager.publishHybridMtp(lease, hybridPub);
    EXPECT_EQ(pubResult.status, PublishStatus::kPublished);
    manager.retireHybridSnapshotReservation(lease, *reservation);
    lease.release();

    // Re-acquire with the published checkpoint as a candidate.
    std::vector<HybridCheckpointCandidate> const candidates{HybridCheckpointCandidate{2 * kPAGE_SIZE, kHASH_B}};
    AcquireResult hit = manager.acquireHybridMtp(candidates, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(hit.status, AcquireStatus::kAcquired);
    EXPECT_GT(hit.plan.reuseTokenLength, 0) << "published hybrid checkpoint must trigger reuse";
    takeLease(hit).release();
}

TEST(ContextCacheExtendedDeploymentTests, HybridMtpStaleCheckpointDigestDoesNotTriggerReuse)
{
    int32_t const snapshotSlots = 4;
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, snapshotSlots, snapshotSlots}, 4, hybridMtpContract());

    // Cold acquire — nothing published.
    constexpr BlockHash kSTALE_DIGEST{0x9999999999999999ULL, 0xFFFFFFFFFFFFFFFFULL};
    std::vector<HybridCheckpointCandidate> const staleCandidates{
        HybridCheckpointCandidate{2 * kPAGE_SIZE, kSTALE_DIGEST}};
    AcquireResult result = manager.acquireHybridMtp(staleCandidates, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 0) << "stale digest with no matching record must not reuse";
    takeLease(result).release();
}

// =============================================================================
// Hybrid DFlash: checkpoint-based reuse with futureDependencyTokens=0
// =============================================================================

// Given a hybrid+DFlash deployment with futureDependencyTokens == 0,
// When deployment validation is performed,
// Then validation succeeds because zero-dependency modes need no boundary-hidden snapshot.
TEST(ContextCacheExtendedDeploymentTests, HybridDFlashDeploymentAdmission)
{
    DeploymentConfig deployment;
    deployment.base = makeHybridConfig();
    deployment.base.specDecodeType = SpecDecodeMode::kDFlash;
    deployment.base.isSpecDecodeBase = true;
    deployment.draft = makeHybridConfig(/*attentionLayers=*/2, /*recurrentLayers=*/2);
    deployment.draft->specDecodeType = SpecDecodeMode::kDFlash;
    deployment.draft->hasOwnKVCache = true;
    deployment.draft->sharesTargetKV = false;
    deployment.draft->baseModelHiddenSize = deployment.base.hiddenSize;
    SpecDecodeConfig specConfig;
    specConfig.draftTopK = 1;
    specConfig.draftSteps = 1;
    specConfig.verifyTokenCount = 8;
    specConfig.draftBlockSize = 8;
    deployment.specConfig = specConfig;

    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(deployment);
    EXPECT_EQ(profile.baseStateKind, ContextCacheModelStateKind::kHybrid);
    ASSERT_TRUE(profile.specReuseContract.has_value());
    EXPECT_TRUE(profile.specReuseContract->ownsPagedSpecState);
    EXPECT_EQ(profile.specReuseContract->futureDependencyTokens, 0);
}

TEST(ContextCacheExtendedDeploymentTests, HybridDFlashColdMissDoesNotReuse)
{
    int32_t const snapshotSlots = 4;
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, snapshotSlots, snapshotSlots}, 4, dflashContract());

    std::vector<HybridCheckpointCandidate> const noCandidates;
    AcquireResult result = manager.acquireHybridMtp(noCandidates, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 0);
    takeLease(result).release();
}

TEST(ContextCacheExtendedDeploymentTests, HybridDFlashPublishedCheckpointTriggersReuse)
{
    int32_t const snapshotSlots = 4;
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, snapshotSlots, snapshotSlots}, 4, dflashContract());

    std::vector<HybridCheckpointCandidate> const noCandidates;
    AcquireResult cold = manager.acquireHybridMtp(noCandidates, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE);
    CacheRequestLease lease = takeLease(cold);
    std::optional<HybridSnapshotReservation> reservation = manager.reserveHybridSnapshots(lease, true);
    ASSERT_TRUE(reservation.has_value());
    HybridPublishRequest hybridPub;
    hybridPub.fullBlockHashes = {kHASH_A};
    hybridPub.checkpoint = HybridCheckpointKey{kHASH_B, 2 * kPAGE_SIZE};
    hybridPub.snapshots = *reservation;
    manager.publishHybridMtp(lease, hybridPub);
    manager.retireHybridSnapshotReservation(lease, *reservation);
    lease.release();

    std::vector<HybridCheckpointCandidate> const candidates{HybridCheckpointCandidate{2 * kPAGE_SIZE, kHASH_B}};
    AcquireResult hit = manager.acquireHybridMtp(candidates, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(hit.status, AcquireStatus::kAcquired);
    EXPECT_GT(hit.plan.reuseTokenLength, 0) << "published hybrid DFlash checkpoint must trigger reuse";
    CacheRequestLease hitLease = takeLease(hit);
    EXPECT_EQ(hitLease.basePages().size(), 3U);
    EXPECT_EQ(hitLease.draftPages().size(), 3U);
    hitLease.release();
}

// =============================================================================
// Cross-mode: spec acquire ignores base-only published records
// =============================================================================

TEST(ContextCacheExtendedDeploymentTests, SpecAcquireIgnoresVanillaOnlyPublishedRecord)
{
    // A vanilla-published record (base pages only, no spec state) must not be treated as a hit
    // by a speculative acquire — the draft-model state is missing.
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{12, 12, 0, 0}, 4, dflashContract());
    publishVanilla(manager, {kHASH_A, kHASH_B});

    AcquireResult specResult = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(specResult.status, AcquireStatus::kAcquired);
    EXPECT_EQ(specResult.plan.reuseTokenLength, 0) << "spec acquire must not reuse base-only records";
    takeLease(specResult).release();
}

// =============================================================================
// Gemma4 MTP grow: only base pages grow (ownsPagedSpecState=false)
// =============================================================================

TEST(ContextCacheExtendedDeploymentTests, Gemma4MTPGrowBaseOnlyAndReuse)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{12, 0, 0, 0}, 4, gemma4MtpContract());

    AcquireResult cold = manager.acquireSpec({kHASH_A}, kPAGE_SIZE);
    CacheRequestLease lease = takeLease(cold);
    EXPECT_TRUE(manager.growSpecPages(lease, 2, 0));
    EXPECT_EQ(lease.basePages().size(), 3U);
    EXPECT_TRUE(lease.draftPages().empty());
    manager.publish(lease, PublishRequest{{kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE});
    lease.release();

    AcquireResult hit = manager.acquireSpec({kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE + 1);
    ASSERT_EQ(hit.status, AcquireStatus::kAcquired);
    EXPECT_EQ(hit.plan.reuseTokenLength, 3 * kPAGE_SIZE);
    CacheRequestLease hitLease = takeLease(hit);
    EXPECT_EQ(hitLease.basePages().size(), 4U);
    EXPECT_TRUE(hitLease.draftPages().empty());
    hitLease.release();
}

// =============================================================================
// Spec bypass policy suppresses reuse
// =============================================================================

TEST(ContextCacheExtendedDeploymentTests, SpecBypassPolicySuppressesReuse)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, 0, 0}, 4, dflashContract());
    publishSpec(manager, {kHASH_A, kHASH_B});

    AcquireResult result
        = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1, ContextCacheLookupPolicy::kBypass);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 0) << "kBypass must suppress spec reuse even when prefix exists";
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kNoReusablePrefix);
    takeLease(result).release();
}

// =============================================================================
// DFlash partial prefix match: only leading matched blocks are reused
// =============================================================================

TEST(ContextCacheExtendedDeploymentTests, DFlashPartialPrefixMatchReusesOnlyMatchedBlocks)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, 0, 0}, 4, dflashContract());
    publishSpec(manager, {kHASH_A, kHASH_B});

    // Request [A, C]: only leading A matches the published [A, B].
    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_C}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, kPAGE_SIZE);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kStandard);
    takeLease(result).release();
}

// =============================================================================
// EAGLE exact block-aligned input triggers full input rewind
// =============================================================================

TEST(ContextCacheExtendedDeploymentTests, EagleExactBlockAlignedInputRewinds)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, 0, 0}, 4, eagleContract());
    publishSpec(manager, {kHASH_A, kHASH_B});

    // Exact block-aligned: 2 blocks * pageSize tokens. Triggers full input rewind.
    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kFullInputRewind);
    takeLease(result).release();
}

// =============================================================================
// Cross-mode: vanilla acquire reuses base pages from spec-published record
// =============================================================================

TEST(ContextCacheExtendedDeploymentTests, VanillaAcquireReusesSpecPublishedPrefix)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{12, 12, 0, 0}, 4, dflashContract());
    publishSpec(manager, {kHASH_A, kHASH_B});

    // Vanilla acquire should see the base pages published by the spec path.
    AcquireResult result = manager.acquireVanilla({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 2 * kPAGE_SIZE)
        << "vanilla acquire must reuse base pages from spec-published records";
    EXPECT_EQ(result.plan.kind, ReusePlanKind::kStandard);
    takeLease(result).release();
}
