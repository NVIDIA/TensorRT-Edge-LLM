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

#include "runtime/state/contextCache/contextCacheDeployment.h"

#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"

#include <gtest/gtest.h>

#include <array>

using namespace trt_edgellm::rt;

namespace
{

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
    config.kvCacheDtype = nvinfer1::DataType::kHALF;
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
    config.recurrentStateDtype = nvinfer1::DataType::kHALF;
    config.convStateDtype = nvinfer1::DataType::kHALF;
    return config;
}

DeploymentConfig makeEagleDeployment()
{
    DeploymentConfig deployment;
    deployment.base = makeAttentionConfig();
    deployment.base.specDecodeType = SpecDecodeMode::kEAGLE;
    deployment.base.isSpecDecodeBase = true;
    deployment.base.specTargetLayerIds = {0, 2, 3};
    deployment.draft = makeAttentionConfig(/*attentionLayers=*/2);
    deployment.draft->specDecodeType = SpecDecodeMode::kEAGLE;
    deployment.draft->baseModelHiddenSize = deployment.base.hiddenSize * 3;
    SpecDecodeConfig specConfig;
    specConfig.draftingTopK = 1;
    specConfig.draftingStep = 2;
    specConfig.verifySize = 4;
    deployment.specConfig = specConfig;
    return deployment;
}

DeploymentConfig makeGemma4MTPDeployment()
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
    specConfig.draftingTopK = 1;
    specConfig.draftingStep = 4;
    specConfig.verifySize = 5;
    deployment.specConfig = specConfig;
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
    specConfig.draftingTopK = 2;
    specConfig.draftingStep = 3;
    specConfig.verifySize = 5;
    deployment.specConfig = specConfig;
    return deployment;
}

} // namespace

TEST(ContextCacheDeploymentTests, ClassifiesSupportedVanillaHybridAndPureRecurrentDeployments)
{
    DeploymentConfig vanilla{makeAttentionConfig(), std::nullopt, std::nullopt};
    ContextCacheDeploymentProfile const vanillaProfile = validateContextCacheDeployment(vanilla);
    EXPECT_EQ(vanillaProfile.baseStateKind, ContextCacheModelStateKind::kAttentionOnly);
    EXPECT_FALSE(vanillaProfile.specReuseContract.has_value());

    DeploymentConfig hybrid{makeHybridConfig(), std::nullopt, std::nullopt};
    ContextCacheDeploymentProfile const hybridProfile = validateContextCacheDeployment(hybrid);
    EXPECT_EQ(hybridProfile.baseStateKind, ContextCacheModelStateKind::kHybrid);
    EXPECT_FALSE(hybridProfile.specReuseContract.has_value());

    LLMEngineConfig recurrent = makeHybridConfig(/*attentionLayers=*/0, /*recurrentLayers=*/3);
    DeploymentConfig pureRecurrent{recurrent, std::nullopt, std::nullopt};
    ContextCacheDeploymentProfile const recurrentProfile = validateContextCacheDeployment(pureRecurrent);
    EXPECT_EQ(recurrentProfile.baseStateKind, ContextCacheModelStateKind::kPureRecurrent);
    EXPECT_FALSE(recurrentProfile.specReuseContract.has_value());
}

TEST(ContextCacheDeploymentTests, RejectsBlockDiffusion)
{
    DeploymentConfig deployment{makeAttentionConfig(), std::nullopt, std::nullopt};
    deployment.base.isDiffusionBackbone = true;

    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, SupportsFp8AndRejectsUnsupportedDtypeAndVisionAttention)
{
    DeploymentConfig deployment{makeAttentionConfig(), std::nullopt, std::nullopt};

    deployment.base.kvCacheDtype = nvinfer1::DataType::kFP8;
    EXPECT_NO_THROW(validateContextCacheDeployment(deployment));

    deployment.base = makeAttentionConfig();
    deployment.base.kvCacheDtype = nvinfer1::DataType::kBF16;
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);

    deployment.base = makeAttentionConfig();
    deployment.base.useVisionBidirectionalAttention = true;
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, AllowsNonStatefulDecoderLayers)
{
    DeploymentConfig deployment{makeHybridConfig(), std::nullopt, std::nullopt};
    deployment.base.numDecoderLayers += 2;
    EXPECT_NO_THROW(validateContextCacheDeployment(deployment));
}

TEST(ContextCacheDeploymentTests, RejectsMalformedDeploymentTuples)
{
    DeploymentConfig malformedVanilla{makeAttentionConfig(), std::nullopt, SpecDecodeConfig{}};
    EXPECT_THROW(validateContextCacheDeployment(malformedVanilla), std::runtime_error);

    DeploymentConfig malformedEagle = makeEagleDeployment();
    malformedEagle.draft.reset();
    EXPECT_THROW(validateContextCacheDeployment(malformedEagle), std::runtime_error);

    malformedEagle = makeEagleDeployment();
    malformedEagle.base.isSpecDecodeBase = false;
    EXPECT_THROW(validateContextCacheDeployment(malformedEagle), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, RejectsUnsafeKvDonorGraphsAndLayouts)
{
    DeploymentConfig deployment{makeAttentionConfig(/*attentionLayers=*/3), std::nullopt, std::nullopt};
    deployment.base.kvSharingDonors = {-1, 0, 1};
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);

    deployment.base.kvSharingDonors = {-1, 0, -1};
    deployment.base.kvLayerConfigs[1].headDim += 16;
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, AdmitsPageTableAwareSpecModesAndRejectsHybridEagle)
{
    std::array<SpecDecodeMode, 3> const supportedModes{
        SpecDecodeMode::kDFlash,
        SpecDecodeMode::kJetSpec,
        SpecDecodeMode::kDSpark,
    };
    for (SpecDecodeMode const mode : supportedModes)
    {
        DeploymentConfig deployment = makeEagleDeployment();
        deployment.base.specDecodeType = mode;
        deployment.draft->specDecodeType = mode;
        if (mode == SpecDecodeMode::kDSpark)
        {
            deployment.draft->specDraftBlockSize = 4;
        }
        else
        {
            deployment.specConfig->dflashBlockSize = 4;
        }
        ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(deployment);
        ASSERT_TRUE(profile.specReuseContract.has_value());
    }

    DeploymentConfig hybridEagle = makeEagleDeployment();
    hybridEagle.base = makeHybridConfig();
    hybridEagle.base.specDecodeType = SpecDecodeMode::kEAGLE;
    hybridEagle.base.isSpecDecodeBase = true;
    EXPECT_THROW(validateContextCacheDeployment(hybridEagle), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, AdmitsHybridMtpAndRejectsAttentionOnlyOrSharedKvMtp)
{
    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(makeHybridMtpDeployment());
    EXPECT_EQ(profile.baseStateKind, ContextCacheModelStateKind::kHybrid);
    ASSERT_TRUE(profile.specReuseContract.has_value());
    EXPECT_TRUE(profile.specReuseContract->ownsPagedSpecState);
    EXPECT_EQ(profile.specReuseContract->futureDependencyTokens, 1);

    DeploymentConfig attentionOnly = makeHybridMtpDeployment();
    attentionOnly.base = makeAttentionConfig();
    attentionOnly.base.specDecodeType = SpecDecodeMode::kMTP;
    attentionOnly.base.isSpecDecodeBase = true;
    EXPECT_THROW(validateContextCacheDeployment(attentionOnly), std::runtime_error);

    DeploymentConfig sharedKv = makeHybridMtpDeployment();
    sharedKv.draft->hasOwnKVCache = false;
    sharedKv.draft->sharesTargetKV = true;
    EXPECT_THROW(validateContextCacheDeployment(sharedKv), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, ResolvesPerMethodReuseContract)
{
    DeploymentConfig eagle = makeEagleDeployment();
    SpecReuseContract const eagleContract = *validateContextCacheDeployment(eagle).specReuseContract;
    EXPECT_TRUE(eagleContract.ownsPagedSpecState);
    EXPECT_EQ(eagleContract.futureDependencyTokens, 1);

    for (SpecDecodeMode const mode : {SpecDecodeMode::kDFlash, SpecDecodeMode::kJetSpec})
    {
        DeploymentConfig blockDraft = makeEagleDeployment();
        blockDraft.base.specDecodeType = mode;
        blockDraft.draft->specDecodeType = mode;
        blockDraft.specConfig->dflashBlockSize = 7;
        SpecReuseContract const contract = *validateContextCacheDeployment(blockDraft).specReuseContract;
        EXPECT_TRUE(contract.ownsPagedSpecState);
        EXPECT_EQ(contract.futureDependencyTokens, 0);
    }

    DeploymentConfig dspark = makeEagleDeployment();
    dspark.base.specDecodeType = SpecDecodeMode::kDSpark;
    dspark.draft->specDecodeType = SpecDecodeMode::kDSpark;
    dspark.draft->specDraftBlockSize = 9;
    SpecReuseContract const dsparkContract = *validateContextCacheDeployment(dspark).specReuseContract;
    EXPECT_TRUE(dsparkContract.ownsPagedSpecState);
    EXPECT_EQ(dsparkContract.futureDependencyTokens, 0);

    SpecReuseContract const gemmaContract
        = *validateContextCacheDeployment(makeGemma4MTPDeployment()).specReuseContract;
    EXPECT_FALSE(gemmaContract.ownsPagedSpecState);
    EXPECT_EQ(gemmaContract.futureDependencyTokens, 0);
}

TEST(ContextCacheDeploymentTests, ClassifiesSupportedEagle)
{
    DeploymentConfig deployment = makeEagleDeployment();
    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(deployment);
    ASSERT_TRUE(profile.specReuseContract.has_value());
    EXPECT_EQ(profile.baseStateKind, ContextCacheModelStateKind::kAttentionOnly);
    EXPECT_TRUE(profile.specReuseContract->ownsPagedSpecState);
    EXPECT_EQ(profile.specReuseContract->futureDependencyTokens, 1);
}

TEST(ContextCacheDeploymentTests, AdmitsEagleTreeConfiguration)
{
    DeploymentConfig deployment = makeEagleDeployment();
    deployment.specConfig->draftingTopK = 2;

    EXPECT_NO_THROW(validateContextCacheDeployment(deployment));
}

TEST(ContextCacheDeploymentTests, AdmitsAttentionOnlyGemma4MTPWithoutPagedSpecState)
{
    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(makeGemma4MTPDeployment());

    EXPECT_EQ(profile.baseStateKind, ContextCacheModelStateKind::kAttentionOnly);
    ASSERT_TRUE(profile.specReuseContract.has_value());
    EXPECT_FALSE(profile.specReuseContract->ownsPagedSpecState);
}
