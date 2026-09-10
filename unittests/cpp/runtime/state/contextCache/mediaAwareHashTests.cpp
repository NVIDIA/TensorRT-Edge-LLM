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

#include "runtime/state/contextCache/blockHash.h"
#include "runtime/state/contextCache/contextCacheCoordinator.h"
#include "runtime/state/contextCache/contextCacheDeployment.h"

#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/kvPageTable.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

constexpr int32_t kPAGE_SIZE = kTOKENS_PER_PAGE; // 128

Hash128 makeMediaHash(uint64_t seed)
{
    return Hash128{seed, seed ^ 0xABCDABCDABCDABCDULL};
}

std::vector<int32_t> makeTokens(int32_t count, int32_t startId = 1)
{
    std::vector<int32_t> tokens(static_cast<size_t>(count));
    std::iota(tokens.begin(), tokens.end(), startId);
    return tokens;
}

} // namespace

// --- Media-aware block hashing tests (unit-level, no GPU) ---

TEST(MediaAwareHashTests, TextOnlyEmptyMediaHash)
{
    auto const tokens = makeTokens(kPAGE_SIZE);
    // hashBlock with nullptr perPositionMediaHash should equal hashBlock without it.
    BlockHash const withNull = hashBlock(kCHAIN_ROOT, tokens.data(), tokens.size(), {}, nullptr);
    BlockHash const withoutMedia = hashBlock(kCHAIN_ROOT, tokens.data(), tokens.size());
    EXPECT_EQ(withNull, withoutMedia);

    // Also test hashFullBlocks: empty perPositionMediaHash vector produces same result.
    auto const fullWithout = hashFullBlocks(tokens.data(), tokens.size(), kPAGE_SIZE);
    auto const fullWithNull = hashFullBlocks(tokens.data(), tokens.size(), kPAGE_SIZE, {}, nullptr);
    EXPECT_EQ(fullWithout, fullWithNull);
}

TEST(MediaAwareHashTests, SameImageSamePositionSameHash)
{
    auto const tokens = makeTokens(kPAGE_SIZE);
    Hash128 const imageA = makeMediaHash(0x1111);

    std::vector<Hash128> mediaHash(kPAGE_SIZE, Hash128{});
    mediaHash[10] = imageA;

    BlockHash const first = hashBlock(kCHAIN_ROOT, tokens.data(), tokens.size(), {}, mediaHash.data());
    BlockHash const second = hashBlock(kCHAIN_ROOT, tokens.data(), tokens.size(), {}, mediaHash.data());
    EXPECT_EQ(first, second);
}

TEST(MediaAwareHashTests, DifferentImageSamePositionDifferentHash)
{
    auto const tokens = makeTokens(kPAGE_SIZE);
    Hash128 const imageA = makeMediaHash(0x1111);
    Hash128 const imageB = makeMediaHash(0x2222);

    std::vector<Hash128> mediaHashA(kPAGE_SIZE, Hash128{});
    mediaHashA[10] = imageA;
    std::vector<Hash128> mediaHashB(kPAGE_SIZE, Hash128{});
    mediaHashB[10] = imageB;

    BlockHash const hashA = hashBlock(kCHAIN_ROOT, tokens.data(), tokens.size(), {}, mediaHashA.data());
    BlockHash const hashB = hashBlock(kCHAIN_ROOT, tokens.data(), tokens.size(), {}, mediaHashB.data());
    EXPECT_NE(hashA, hashB);
}

TEST(MediaAwareHashTests, SameImageDifferentPositionDifferentHash)
{
    auto const tokens = makeTokens(kPAGE_SIZE);
    Hash128 const imageA = makeMediaHash(0x3333);

    std::vector<Hash128> mediaHashPos5(kPAGE_SIZE, Hash128{});
    mediaHashPos5[5] = imageA;
    std::vector<Hash128> mediaHashPos50(kPAGE_SIZE, Hash128{});
    mediaHashPos50[50] = imageA;

    BlockHash const hash5 = hashBlock(kCHAIN_ROOT, tokens.data(), tokens.size(), {}, mediaHashPos5.data());
    BlockHash const hash50 = hashBlock(kCHAIN_ROOT, tokens.data(), tokens.size(), {}, mediaHashPos50.data());
    EXPECT_NE(hash5, hash50);
}

TEST(MediaAwareHashTests, MultiImageOrderMatters)
{
    auto const tokens = makeTokens(kPAGE_SIZE);
    Hash128 const imageA = makeMediaHash(0x4444);
    Hash128 const imageB = makeMediaHash(0x5555);

    std::vector<Hash128> mediaAB(kPAGE_SIZE, Hash128{});
    mediaAB[10] = imageA;
    mediaAB[20] = imageB;

    std::vector<Hash128> mediaBA(kPAGE_SIZE, Hash128{});
    mediaBA[10] = imageB;
    mediaBA[20] = imageA;

    BlockHash const hashAB = hashBlock(kCHAIN_ROOT, tokens.data(), tokens.size(), {}, mediaAB.data());
    BlockHash const hashBA = hashBlock(kCHAIN_ROOT, tokens.data(), tokens.size(), {}, mediaBA.data());
    EXPECT_NE(hashAB, hashBA);
}

// --- Coordinator-level media-aware tests (requires GPU) ---

namespace
{

constexpr int32_t kMAX_BATCH{2};
constexpr int32_t kMAX_SEQUENCE_LENGTH{512};

LLMEngineConfig makeEngineConfig()
{
    LLMEngineConfig config;
    config.modelType = "media-hash-test";
    config.hiddenSize = 64;
    config.numDecoderLayers = 1;
    config.numAttentionLayers = 1;
    config.numKVHeads = 1;
    config.headDim = 8;
    config.rotaryDim = 8;
    config.maxSupportedBatchSize = kMAX_BATCH;
    config.maxSupportedInputLength = kMAX_SEQUENCE_LENGTH;
    config.maxKVCacheCapacity = kMAX_SEQUENCE_LENGTH;
    int64_t const minimumActivePages = computeMinimumKvPoolPages(kMAX_BATCH, kMAX_SEQUENCE_LENGTH);
    ELLM_CHECK(minimumActivePages <= kMAX_KV_POOL_PAGES, "Test KV pool page count must fit int32.");
    config.kvPoolPages = static_cast<int32_t>(minimumActivePages);
    config.kvCacheDtype = nvinfer1::DataType::kHALF;
    config.layerTypes = {HybridCacheManager::LayerType::kAttention};
    config.kvLayerConfigs = {KVLayerConfig{config.numKVHeads, config.headDim}};
    return config;
}

HybridCacheManager::Config makeCacheConfig(LLMEngineConfig const& engine)
{
    KVCacheManager::Config kvConfig{
        /*.numAttentionLayers=*/engine.numAttentionLayers,
        /*.maxBatchSize=*/engine.maxSupportedBatchSize,
        /*.maxSequenceLength=*/engine.maxKVCacheCapacity,
        /*.layerConfigs=*/engine.kvLayerConfigs,
        /*.kvCacheType=*/engine.kvCacheDtype,
        /*.numPages=*/engine.kvPoolPages,
    };
    MambaCacheManager::Config mambaConfig{
        /*.numRecurrentLayers=*/0,
        /*.maxBatchSize=*/engine.maxSupportedBatchSize,
    };
    return HybridCacheManager::Config{
        /*.layerTypes=*/engine.layerTypes,
        /*.kvConfig=*/std::move(kvConfig),
        /*.mambaConfig=*/std::move(mambaConfig),
        /*.maxBatchSize=*/engine.maxSupportedBatchSize,
    };
}

class MediaAwareCoordinatorTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        mEngine = makeEngineConfig();
        mDeployment = DeploymentConfig{mEngine, std::nullopt, std::nullopt};
        mCache = std::make_unique<HybridCacheManager>(makeCacheConfig(mEngine), mStream);
        KVCacheManager const& kv = mCache->getKVCacheManager();
        mPageTable = std::make_unique<KVPageTable>(kMAX_BATCH, pagesPerSlot(kv.maxCapPadded()), kv.numPages());
        mPageTable->setIdentity();
        mPageTable->upload(mStream);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        createCoordinator();
    }

    void TearDown() override
    {
        if (mCoordinator)
        {
            EXPECT_EQ(mCoordinator->shutdown(), ContextCacheCoordinatorStatus::kOk);
            mCoordinator.reset();
        }
        mPageTable.reset();
        mCache.reset();
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    void createCoordinator()
    {
        ContextCachePhysicalResources resources{*mCache, *mPageTable, nullptr, nullptr};
        mCoordinator
            = std::make_unique<ContextCacheCoordinator>(ContextCacheConfig{/*.enabled=*/true, /*.maxRecords=*/16},
                mDeployment, validateContextCacheDeployment(mDeployment), resources, mStream);
    }

    ContextCacheCoordinator::BeginRequestResult beginWithMedia(
        std::vector<int32_t> const& tokens, std::vector<Hash128> const& perPositionMediaHash)
    {
        ContextCacheBatchAdmission admission;
        admission.lookupPolicy = ContextCacheLookupPolicy::kUseCache;
        admission.commitPolicy = ContextCacheCommitPolicy::kIncludingGeneratedTokens;
        ContextCacheSequenceAdmission seq;
        seq.tokenIds = tokens;
        seq.perPositionMediaHash = perPositionMediaHash;
        admission.sequences.push_back(std::move(seq));
        return mCoordinator->beginRequest(admission, DecodingKvHeadroom{1, 0}, mStream);
    }

    void finalizePrefillAndFinish(
        ContextCacheCoordinator::AdmissionResult& admission, std::vector<int32_t> const& tokens)
    {
        ASSERT_EQ(mCoordinator->preparePrefill(admission.request), ContextCacheCoordinatorStatus::kOk);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        int32_t const inputLength = static_cast<int32_t>(tokens.size());
        std::vector<int32_t> lookahead(1, 9001);
        std::vector<ContextCacheSequenceAdvance> progress;
        progress.push_back(ContextCacheSequenceAdvance{lookahead.data(), 1, inputLength});
        ASSERT_EQ(
            mCoordinator->finalizePrefillPublication(admission.request, progress), ContextCacheCoordinatorStatus::kOk);
        ASSERT_EQ(mCoordinator->finish(admission.request), ContextCacheCoordinatorStatus::kOk);
    }

    cudaStream_t mStream{};
    LLMEngineConfig mEngine;
    DeploymentConfig mDeployment;
    std::unique_ptr<HybridCacheManager> mCache;
    std::unique_ptr<KVPageTable> mPageTable;
    std::unique_ptr<ContextCacheCoordinator> mCoordinator;
};

} // namespace

TEST_F(MediaAwareCoordinatorTests, CoordinatorMediaAwareSequenceMetric)
{
    int32_t const tokenCount = kPAGE_SIZE * 2;
    auto const tokens = makeTokens(tokenCount);
    Hash128 const imageA = makeMediaHash(0xAAAA);

    // Request with media hash.
    std::vector<Hash128> mediaHash(tokenCount, Hash128{});
    mediaHash[5] = imageA;

    auto result = beginWithMedia(tokens, mediaHash);
    ASSERT_EQ(result.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(result.admission.has_value());

    auto const& metrics = mCoordinator->metrics();
    EXPECT_GE(metrics.mediaAwareSequences, 1U);

    ASSERT_EQ(mCoordinator->finish(result.admission->request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(MediaAwareCoordinatorTests, CoordinatorSameMediaHitDifferentMediaMiss)
{
    int32_t const tokenCount = kPAGE_SIZE * 2;
    auto const tokens = makeTokens(tokenCount);
    Hash128 const imageA = makeMediaHash(0xBBBB);
    Hash128 const imageC = makeMediaHash(0xCCCC);

    std::vector<Hash128> mediaHashA(tokenCount, Hash128{});
    mediaHashA[5] = imageA;

    // First request: establishes cache entry.
    auto result1 = beginWithMedia(tokens, mediaHashA);
    ASSERT_EQ(result1.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(result1.admission.has_value());
    finalizePrefillAndFinish(*result1.admission, tokens);

    // Same tokens, same media hash → cache hit (prefillStart > 0).
    auto result2 = beginWithMedia(tokens, mediaHashA);
    ASSERT_EQ(result2.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(result2.admission.has_value());
    EXPECT_GT(result2.admission->prefillStarts[0], 0);
    ASSERT_EQ(mCoordinator->finish(result2.admission->request), ContextCacheCoordinatorStatus::kOk);

    // Same tokens, different media hash → cache miss (prefillStart == 0).
    std::vector<Hash128> mediaHashC(tokenCount, Hash128{});
    mediaHashC[5] = imageC;

    auto result3 = beginWithMedia(tokens, mediaHashC);
    ASSERT_EQ(result3.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(result3.admission.has_value());
    EXPECT_EQ(result3.admission->prefillStarts[0], 0);
    ASSERT_EQ(mCoordinator->finish(result3.admission->request), ContextCacheCoordinatorStatus::kOk);
}
