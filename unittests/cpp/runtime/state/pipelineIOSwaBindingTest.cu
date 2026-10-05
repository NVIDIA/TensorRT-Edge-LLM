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

#include "common/bindingNames.h"
#include "common/pagedKvTypes.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

__global__ void delayMetadataUploads(unsigned long long cycles)
{
    unsigned long long const start = clock64();
    while (clock64() - start < cycles)
    {
    }
}

LLMEngineConfig makeBindingConfig(bool useSwa)
{
    LLMEngineConfig cfg;
    cfg.hiddenSize = 16;
    cfg.outputVocabSize = 32;
    cfg.numAttentionLayers = 3;
    cfg.numDecoderLayers = 3;
    cfg.numKVHeads = 1;
    cfg.headDim = 8;
    cfg.rotaryDim = 8;
    cfg.maxSupportedBatchSize = 2;
    cfg.maxSupportedInputLength = 4;
    cfg.maxKVCacheCapacity = 8192;
    cfg.kvPoolPages
        = static_cast<int32_t>(computeMinimumKvPoolPages(cfg.maxSupportedBatchSize, cfg.maxKVCacheCapacity));
    cfg.numSwaPages = 64;
    cfg.kvCacheDtype = nvinfer1::DataType::kHALF;
    cfg.ropeConfig.type = RopeType::kNoRope;
    cfg.layerTypes.assign(3, HybridCacheManager::LayerType::kAttention);

    int32_t const window = useSwa ? 129 : 0;
    cfg.kvLayerConfigs = {
        KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8},
        KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8, /*kvCacheCapacity=*/window},
        KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8, /*kvCacheCapacity=*/window},
    };
    cfg.kvSharingDonors = {-1, -1, 1};
    return cfg;
}

std::unique_ptr<SharedResources> makeResources(LLMEngineConfig const& cfg)
{
    std::unordered_map<std::string, std::string> const noLoraWeights;
    return SharedResources::createForLLM(cfg, noLoraWeights, nullptr);
}

} // namespace

TEST(PipelineIOSwaBindingTest, ReusablePinnedMetadataIsNotOverwrittenWhileUploadIsPending)
{
    LLMEngineConfig cfg;
    cfg.hiddenSize = 16;
    cfg.outputVocabSize = 32;
    cfg.maxSupportedBatchSize = 2;
    cfg.maxSupportedInputLength = 4;
    cfg.maxKVCacheCapacity = 8;
    cfg.maxPhysicalTokens = 8;
    cfg.maxNumSequences = 2;
    cfg.recurrentPoolRows = 2;
    cfg.rotaryDim = 8;
    cfg.ropeConfig.type = RopeType::kNoRope;

    auto makeBatch = [](int32_t positionBase, int32_t stateBase) {
        RaggedExecutionBatch batch;
        batch.shape = {/*numSequences=*/2, /*validTokens=*/4, /*physicalTokens=*/4, /*queryWidth=*/2,
            /*numContextSequences=*/2, /*numContextTokens=*/4, /*numLogits=*/2};
        batch.positions = {positionBase, positionBase + 1, positionBase + 2, positionBase + 3};
        batch.queryStartOffsets = {0, 2, 4};
        batch.queryLengths = {2, 2};
        batch.pastLengths = {0, 0};
        batch.attentionSequenceLengths = {2, 2};
        batch.stateIndices = {stateBase, stateBase + 1};
        batch.logitsIndices = {1, 3};
        return batch;
    };

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    {
        PipelineIO io = PipelineIO::createForLLM(cfg, stream);
        RaggedExecutionBatch const first = makeBatch(10, 0);
        RaggedExecutionBatch const second = makeBatch(20, 0);
        Tensor firstSnapshot({4}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "firstMetadataSnapshot");
        Tensor secondSnapshot({4}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "secondMetadataSnapshot");

        constexpr unsigned long long kDELAY_CYCLES{50000000ULL};
        delayMetadataUploads<<<1, 1, 0, stream>>>(kDELAY_CYCLES);
        io.uploadRaggedMetadata(first, stream);
        ASSERT_EQ(cudaMemcpyAsync(firstSnapshot.rawPointer(), io.positions.rawPointer(), 4 * sizeof(int32_t),
                      cudaMemcpyDeviceToDevice, stream),
            cudaSuccess);
        io.uploadRaggedMetadata(second, stream);
        ASSERT_EQ(cudaMemcpyAsync(secondSnapshot.rawPointer(), io.positions.rawPointer(), 4 * sizeof(int32_t),
                      cudaMemcpyDeviceToDevice, stream),
            cudaSuccess);

        std::array<int32_t, 4> firstHost{};
        std::array<int32_t, 4> secondHost{};
        ASSERT_EQ(cudaMemcpyAsync(
                      firstHost.data(), firstSnapshot.rawPointer(), sizeof(firstHost), cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(secondHost.data(), secondSnapshot.rawPointer(), sizeof(secondHost),
                      cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(firstHost, (std::array<int32_t, 4>{10, 11, 12, 13}));
        EXPECT_EQ(secondHost, (std::array<int32_t, 4>{20, 21, 22, 23}));
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(PipelineIOSwaBindingTest, StateIndexOnlyUploadSharesRaggedMetadataStagingFence)
{
    LLMEngineConfig cfg;
    cfg.hiddenSize = 16;
    cfg.outputVocabSize = 32;
    cfg.maxSupportedBatchSize = 2;
    cfg.maxSupportedInputLength = 4;
    cfg.maxKVCacheCapacity = 8;
    cfg.maxPhysicalTokens = 8;
    cfg.maxNumSequences = 2;
    cfg.recurrentPoolRows = 4;
    cfg.rotaryDim = 8;
    cfg.ropeConfig.type = RopeType::kNoRope;

    auto makeBatch = [](std::array<int32_t, 2> const& stateIndices) {
        RaggedExecutionBatch batch;
        batch.shape = {/*numSequences=*/2, /*validTokens=*/4, /*physicalTokens=*/4, /*queryWidth=*/2,
            /*numContextSequences=*/2, /*numContextTokens=*/4, /*numLogits=*/2};
        batch.positions = {0, 1, 0, 1};
        batch.queryStartOffsets = {0, 2, 4};
        batch.queryLengths = {2, 2};
        batch.pastLengths = {0, 0};
        batch.attentionSequenceLengths = {2, 2};
        batch.stateIndices.assign(stateIndices.begin(), stateIndices.end());
        batch.logitsIndices = {1, 3};
        return batch;
    };

    RaggedExecutionBatch const first = makeBatch({3, 1});
    std::vector<ResidentRef> const second{{1, 10}, {3, 11}};
    RaggedExecutionBatch const third = makeBatch({0, 2});

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    {
        PipelineIO io = PipelineIO::createForLLM(cfg, stream);
        Tensor firstSnapshot({2}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "firstStateIndexSnapshot");
        Tensor secondSnapshot({2}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "secondStateIndexSnapshot");
        Tensor thirdSnapshot({2}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "thirdStateIndexSnapshot");
        Tensor identitySnapshot({2}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "identityStateIndexSnapshot");

        constexpr unsigned long long kDELAY_CYCLES{50000000ULL};
        delayMetadataUploads<<<1, 1, 0, stream>>>(kDELAY_CYCLES);
        io.uploadRaggedMetadata(first, stream);
        ASSERT_EQ(cudaMemcpyAsync(firstSnapshot.rawPointer(), io.stateIndices.rawPointer(), 2 * sizeof(int32_t),
                      cudaMemcpyDeviceToDevice, stream),
            cudaSuccess);
        delayMetadataUploads<<<1, 1, 0, stream>>>(kDELAY_CYCLES);
        io.uploadStateIndices(&second, 2, stream);
        ASSERT_EQ(cudaMemcpyAsync(secondSnapshot.rawPointer(), io.stateIndices.rawPointer(), 2 * sizeof(int32_t),
                      cudaMemcpyDeviceToDevice, stream),
            cudaSuccess);
        io.uploadRaggedMetadata(third, stream);
        ASSERT_EQ(cudaMemcpyAsync(thirdSnapshot.rawPointer(), io.stateIndices.rawPointer(), 2 * sizeof(int32_t),
                      cudaMemcpyDeviceToDevice, stream),
            cudaSuccess);
        io.uploadStateIndices(nullptr, 2, stream);
        ASSERT_EQ(cudaMemcpyAsync(identitySnapshot.rawPointer(), io.stateIndices.rawPointer(), 2 * sizeof(int32_t),
                      cudaMemcpyDeviceToDevice, stream),
            cudaSuccess);

        std::array<int32_t, 2> firstHost{};
        std::array<int32_t, 2> secondHost{};
        std::array<int32_t, 2> thirdHost{};
        std::array<int32_t, 2> identityHost{};
        ASSERT_EQ(cudaMemcpyAsync(
                      firstHost.data(), firstSnapshot.rawPointer(), sizeof(firstHost), cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(secondHost.data(), secondSnapshot.rawPointer(), sizeof(secondHost),
                      cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      thirdHost.data(), thirdSnapshot.rawPointer(), sizeof(thirdHost), cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(identityHost.data(), identitySnapshot.rawPointer(), sizeof(identityHost),
                      cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(firstHost, (std::array<int32_t, 2>{3, 1}));
        EXPECT_EQ(secondHost, (std::array<int32_t, 2>{1, 3}));
        EXPECT_EQ(thirdHost, (std::array<int32_t, 2>{0, 2}));
        EXPECT_EQ(identityHost, (std::array<int32_t, 2>{0, 1}));
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(PipelineIOSwaBindingTest, StepFenceProtectsReusablePinnedTokenSnapshots)
{
    LLMEngineConfig cfg;
    cfg.hiddenSize = 16;
    cfg.outputVocabSize = 32;
    cfg.maxSupportedBatchSize = 2;
    cfg.maxSupportedInputLength = 2;
    cfg.maxKVCacheCapacity = 8;
    cfg.maxPhysicalTokens = 4;
    cfg.maxNumSequences = 2;
    cfg.recurrentPoolRows = 2;
    cfg.rotaryDim = 8;
    cfg.ropeConfig.type = RopeType::kNoRope;

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    {
        PipelineIO io = PipelineIO::createForLLM(cfg, stream);
        Tensor hostTokens({2}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "hostTokens");
        Tensor deviceTokens({2}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "deviceTokens");
        auto* source = hostTokens.dataPointer<int32_t>();
        constexpr size_t kBytes{2 * sizeof(int32_t)};

        source[0] = 11;
        source[1] = 22;
        delayMetadataUploads<<<1, 1, 0, stream>>>(50000000ULL);
        ASSERT_EQ(
            cudaMemcpyAsync(deviceTokens.rawPointer(), hostTokens.rawPointer(), kBytes, cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        io.recordStepHostUploads(stream);
        io.waitForStepHostStaging();
        source[0] = 33;
        source[1] = 44;

        std::array<int32_t, 2> first{};
        ASSERT_EQ(cudaMemcpyAsync(first.data(), deviceTokens.rawPointer(), kBytes, cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(first, (std::array<int32_t, 2>{11, 22}));

        ASSERT_EQ(
            cudaMemcpyAsync(deviceTokens.rawPointer(), hostTokens.rawPointer(), kBytes, cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        io.recordStepHostUploads(stream);
        io.waitForStepHostStaging();
        std::array<int32_t, 2> second{};
        ASSERT_EQ(cudaMemcpyAsync(second.data(), deviceTokens.rawPointer(), kBytes, cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(second, (std::array<int32_t, 2>{33, 44}));
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(PipelineIOSwaBindingTest, PrefillToDenoiseStepStagingIsNotOverwrittenWhileUploadIsPending)
{
    LLMEngineConfig cfg;
    cfg.hiddenSize = 16;
    cfg.outputVocabSize = 32;
    cfg.maxSupportedBatchSize = 2;
    cfg.maxSupportedInputLength = 4;
    cfg.maxKVCacheCapacity = 8;
    cfg.maxPhysicalTokens = 8;
    cfg.maxNumSequences = 2;
    cfg.recurrentPoolRows = 2;
    cfg.rotaryDim = 8;
    cfg.ropeConfig.type = RopeType::kNoRope;

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    {
        PipelineIO io = PipelineIO::createForLLM(cfg, stream);
        Tensor hostTemperature({1}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "hostTemperature");
        Tensor deviceTemperature({1}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "deviceTemperature");
        Tensor firstContext({2}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "firstContext");
        Tensor secondContext({2}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "secondContext");
        Tensor firstPhase({1}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "firstPhase");
        Tensor secondPhase({1}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "secondPhase");
        Tensor firstTemperature({1}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "firstTemperature");
        Tensor secondTemperature({1}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "secondTemperature");

        auto upload = [&](std::array<int32_t, 2> const& lengths, int32_t phase, float temperature,
                          Tensor& contextSnapshot, Tensor& phaseSnapshot, Tensor& temperatureSnapshot) {
            std::copy(lengths.begin(), lengths.end(), io.hostContextLengths.dataPointer<int32_t>());
            io.hostPhaseIsEncoder.dataPointer<int32_t>()[0] = phase;
            hostTemperature.dataPointer<float>()[0] = temperature;
            ASSERT_EQ(cudaMemcpyAsync(io.contextLengths.rawPointer(), io.hostContextLengths.rawPointer(),
                          sizeof(lengths), cudaMemcpyHostToDevice, stream),
                cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(io.phaseIsEncoder.rawPointer(), io.hostPhaseIsEncoder.rawPointer(),
                          sizeof(int32_t), cudaMemcpyHostToDevice, stream),
                cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(deviceTemperature.rawPointer(), hostTemperature.rawPointer(), sizeof(float),
                          cudaMemcpyHostToDevice, stream),
                cudaSuccess);
            io.recordStepHostUploads(stream);
            ASSERT_EQ(cudaMemcpyAsync(contextSnapshot.rawPointer(), io.contextLengths.rawPointer(), sizeof(lengths),
                          cudaMemcpyDeviceToDevice, stream),
                cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(phaseSnapshot.rawPointer(), io.phaseIsEncoder.rawPointer(), sizeof(int32_t),
                          cudaMemcpyDeviceToDevice, stream),
                cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(temperatureSnapshot.rawPointer(), deviceTemperature.rawPointer(), sizeof(float),
                          cudaMemcpyDeviceToDevice, stream),
                cudaSuccess);
        };

        constexpr unsigned long long kDELAY_CYCLES{50000000ULL};
        delayMetadataUploads<<<1, 1, 0, stream>>>(kDELAY_CYCLES);
        upload({8, 5}, 1, 0.75F, firstContext, firstPhase, firstTemperature);
        io.waitForStepHostStaging();
        upload({2, 1}, 0, 0.25F, secondContext, secondPhase, secondTemperature);

        std::array<int32_t, 2> firstContextHost{};
        std::array<int32_t, 2> secondContextHost{};
        int32_t firstPhaseHost{};
        int32_t secondPhaseHost{};
        float firstTemperatureHost{};
        float secondTemperatureHost{};
        ASSERT_EQ(cudaMemcpyAsync(firstContextHost.data(), firstContext.rawPointer(), sizeof(firstContextHost),
                      cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(secondContextHost.data(), secondContext.rawPointer(), sizeof(secondContextHost),
                      cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpyAsync(&firstPhaseHost, firstPhase.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      &secondPhaseHost, secondPhase.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(&firstTemperatureHost, firstTemperature.rawPointer(), sizeof(float),
                      cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(&secondTemperatureHost, secondTemperature.rawPointer(), sizeof(float),
                      cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(firstContextHost, (std::array<int32_t, 2>{8, 5}));
        EXPECT_EQ(secondContextHost, (std::array<int32_t, 2>{2, 1}));
        EXPECT_EQ(firstPhaseHost, 1);
        EXPECT_EQ(secondPhaseHost, 0);
        EXPECT_FLOAT_EQ(firstTemperatureHost, 0.75F);
        EXPECT_FLOAT_EQ(secondTemperatureHost, 0.25F);
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(PipelineIOSwaBindingTest, SharedSwaConsumerUsesDonorPoolAndIndependentTable)
{
    LLMEngineConfig const cfg = makeBindingConfig(/*useSwa=*/true);
    auto resources = makeResources(cfg);
    PipelineIO io;
    TensorMap map;
    buildTensorMap(map, io, *resources, cfg, /*kvCacheIndex=*/0);

    KVCacheManager& kv = resources->cacheManagers[0]->getKVCacheManager();
    EXPECT_EQ(map.get(binding_names::formatKVCacheName(1, /*isPast=*/true)), &kv.getCombinedKVCache(1));
    EXPECT_EQ(map.get(binding_names::formatKVCacheName(2, /*isPast=*/true)), &kv.getCombinedKVCache(1));
    EXPECT_EQ(map.get(binding_names::formatKVCacheName(2, /*isPast=*/false)), &kv.getCombinedKVCache(1));

    ASSERT_TRUE(map.contains(binding_names::kKVPageTable));
    ASSERT_TRUE(map.contains(binding_names::kSwaKVPageTable));
    ASSERT_TRUE(map.contains(binding_names::kSwaKVCacheMode));
    ASSERT_NE(resources->getSwaKVPageTable(0), nullptr);
    EXPECT_EQ(map.get(binding_names::kKVPageTable), &io.raggedKVPageTable);
    EXPECT_EQ(map.get(binding_names::kSwaKVPageTable), &io.raggedSwaKVPageTable);
    EXPECT_EQ(map.get(binding_names::kSwaKVCacheMode), &resources->swaKVCacheMode);
    EXPECT_EQ(cfg.prefillDims(/*batch=*/1, /*seqLen=*/1, ExecutionPhase::kContextPrefill).swaKVCacheModeLen, 1);

    int32_t const logicalPages = computeMaxPagesPerSeq(cfg.maxKVCacheCapacity);
    EXPECT_EQ(resources->kvPageTables[0]->maxPagesPerSeq(), logicalPages);
    EXPECT_EQ(resources->getSwaKVPageTable(0)->maxPagesPerSeq(), logicalPages);
    EXPECT_NE(kv.numPages(), kv.numPages(1));

    // The two V namespaces are derived from their own physical page counts.
    int32_t const* fullRow = resources->kvPageTables[0]->hostRow(0);
    EXPECT_EQ(fullRow[logicalPages], kv.numPages());
    resources->getSwaKVPageTable(0)->setEntry(/*slot=*/0, /*logicalPage=*/0, /*kPageId=*/0);
    int32_t const* swaRow = resources->getSwaKVPageTable(0)->hostRow(0);
    EXPECT_EQ(swaRow[logicalPages], kv.numPages(1));
}

TEST(PipelineIOSwaBindingTest, DefaultOffKeepsFullTableAndExistingDonorRouting)
{
    LLMEngineConfig const cfg = makeBindingConfig(/*useSwa=*/false);
    auto resources = makeResources(cfg);
    PipelineIO io;
    TensorMap map;
    buildTensorMap(map, io, *resources, cfg, /*kvCacheIndex=*/0);

    KVCacheManager& kv = resources->cacheManagers[0]->getKVCacheManager();
    ASSERT_EQ(resources->swaKVPageTables.size(), 1u);
    EXPECT_EQ(resources->getSwaKVPageTable(0), nullptr);
    EXPECT_FALSE(map.contains(binding_names::kSwaKVPageTable));
    EXPECT_FALSE(map.contains(binding_names::kSwaKVCacheMode));
    EXPECT_TRUE(map.contains(binding_names::kKVPageTable));
    EXPECT_EQ(map.get(binding_names::formatKVCacheName(2, /*isPast=*/true)), &kv.getCombinedKVCache(1));
    EXPECT_EQ(kv.numPages(0), kv.numPages(1));
}

TEST(PipelineIOSwaBindingTest, FullModeAliasesOrdinaryTableAndUsesFullPhysicalPools)
{
    LLMEngineConfig cfg = makeBindingConfig(/*useSwa=*/true);
    int32_t const capabilityMarker = cfg.kvLayerConfigs[1].kvCacheCapacity;
    cfg.setSwaKVCacheMode(SwaKVCacheMode::kFull);
    auto resources = makeResources(cfg);
    PipelineIO io;
    TensorMap map;
    buildTensorMap(map, io, *resources, cfg, /*kvCacheIndex=*/0);

    KVCacheManager& kv = resources->cacheManagers[0]->getKVCacheManager();
    EXPECT_FALSE(kv.hasReducedKVCache());
    EXPECT_EQ(kv.numPages(0), cfg.kvPoolPages);
    EXPECT_EQ(kv.numPages(1), cfg.kvPoolPages);
    EXPECT_EQ(resources->getSwaKVPageTable(0), nullptr);
    ASSERT_TRUE(map.contains(binding_names::kSwaKVPageTable));
    ASSERT_TRUE(map.contains(binding_names::kSwaKVCacheMode));
    EXPECT_EQ(map.get(binding_names::kSwaKVPageTable), &io.raggedKVPageTable);
    EXPECT_EQ(map.get(binding_names::kSwaKVCacheMode), &resources->swaKVCacheMode);
    EXPECT_EQ(cfg.prefillDims(/*batch=*/1, /*seqLen=*/1, ExecutionPhase::kContextPrefill).swaKVCacheModeLen, 0);
    EXPECT_EQ(cfg.kvLayerConfigs[1].kvCacheCapacity, capabilityMarker);
}

TEST(PipelineIOSwaBindingTest, SpecDecodeAllocatesRopeBuffersForHeterogeneousBaseAndDraft)
{
    DeploymentConfig deployment;
    deployment.base.hiddenSize = 64;
    deployment.base.outputVocabSize = 16;
    deployment.base.maxSupportedInputLength = 4;
    deployment.base.maxKVCacheCapacity = 8;
    deployment.base.maxPhysicalTokens = 8;
    deployment.base.maxNumSequences = 2;
    deployment.base.rotaryDim = 128;
    deployment.base.useDualRope = true;
    deployment.base.slidingRotaryDim = 64;
    deployment.base.fullRotaryDim = 128;
    deployment.base.specDecodeType = SpecDecodeMode::kEAGLE;

    LLMEngineConfig draft = deployment.base;
    draft.vocabSize = 16;
    draft.maxPhysicalTokens = 10;
    draft.useDualRope = false;
    draft.rotaryDim = 128;
    deployment.draft = draft;

    SpecDecodeConfig spec;
    spec.baseOutputHiddenDim = 64;
    spec.draftHiddenSize = 64;
    spec.maxVerifySize = 2;
    spec.maxDraftProposalSize = 2;
    spec.draftingTopK = 1;
    spec.draftingStep = 1;
    spec.verifySize = 2;
    deployment.specConfig = spec;

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    {
        PipelineIO io = PipelineIO::createForSpecDecode(deployment, /*maxRuntimeBatchSize=*/1, stream,
            /*hasAcceptHiddenOutput=*/false,
            /*hasTreeMetadataInputs=*/false);
        EXPECT_EQ(io.raggedRopeCosSin.getShape(), Coords({10, 128}));
        EXPECT_EQ(io.raggedRopeCosSinSliding.getShape(), Coords({8, 64}));
        EXPECT_EQ(io.raggedRopeCosSinFull.getShape(), Coords({8, 128}));

        std::array<int32_t, 2> const positions{0, 1};
        std::array<int32_t, 2> const queryStartOffsets{0, 2};
        std::array<int32_t, 1> const queryLengths{2};
        ASSERT_EQ(cudaMemcpyAsync(
                      io.positions.rawPointer(), positions.data(), sizeof(positions), cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(io.queryStartOffsets.rawPointer(), queryStartOffsets.data(),
                      sizeof(queryStartOffsets), cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(io.queryLengths.rawPointer(), queryLengths.data(), sizeof(queryLengths),
                      cudaMemcpyHostToDevice, stream),
            cudaSuccess);

        SharedResources resources;
        EXPECT_NO_THROW(prepareRaggedRope(io, resources, *deployment.draft, /*physicalTokens=*/2,
            /*numSequences=*/1, stream));
        EXPECT_EQ(io.raggedRopeCosSin.getShape(), Coords({2, 128}));
        EXPECT_NO_THROW(
            prepareRaggedRope(io, resources, deployment.base, /*physicalTokens=*/2, /*numSequences=*/1, stream));
        EXPECT_EQ(io.raggedRopeCosSinSliding.getShape(), Coords({2, 64}));
        EXPECT_EQ(io.raggedRopeCosSinFull.getShape(), Coords({2, 128}));
        EXPECT_NO_THROW(prepareRaggedRope(io, resources, *deployment.draft, /*physicalTokens=*/2,
            /*numSequences=*/1, stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        DeploymentConfig reverse = deployment;
        reverse.base.useDualRope = false;
        reverse.base.rotaryDim = 128;
        reverse.draft->useDualRope = true;
        reverse.draft->slidingRotaryDim = 64;
        reverse.draft->fullRotaryDim = 128;
        PipelineIO reverseIO = PipelineIO::createForSpecDecode(reverse, /*maxRuntimeBatchSize=*/1, stream,
            /*hasAcceptHiddenOutput=*/false,
            /*hasTreeMetadataInputs=*/false);
        EXPECT_EQ(reverseIO.raggedRopeCosSin.getShape(), Coords({8, 128}));
        EXPECT_EQ(reverseIO.raggedRopeCosSinSliding.getShape(), Coords({10, 64}));
        EXPECT_EQ(reverseIO.raggedRopeCosSinFull.getShape(), Coords({10, 128}));
        ASSERT_EQ(cudaMemcpyAsync(reverseIO.positions.rawPointer(), positions.data(), sizeof(positions),
                      cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(reverseIO.queryStartOffsets.rawPointer(), queryStartOffsets.data(),
                      sizeof(queryStartOffsets), cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(reverseIO.queryLengths.rawPointer(), queryLengths.data(), sizeof(queryLengths),
                      cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        SharedResources reverseResources;
        EXPECT_NO_THROW(prepareRaggedRope(reverseIO, reverseResources, reverse.base, /*physicalTokens=*/2,
            /*numSequences=*/1, stream));
        EXPECT_NO_THROW(prepareRaggedRope(reverseIO, reverseResources, *reverse.draft, /*physicalTokens=*/2,
            /*numSequences=*/1, stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(PipelineIOSwaBindingTest, SpecDecodeMRopeUsesResidentPoolAndPreservesInactiveSlots)
{
    constexpr int32_t kRuntimeBatch = 2;
    constexpr int32_t kResidentRows = 6;
    DeploymentConfig deployment;
    deployment.base.hiddenSize = 64;
    deployment.base.outputVocabSize = 16;
    deployment.base.maxSupportedInputLength = 4;
    deployment.base.maxKVCacheCapacity = 8;
    deployment.base.maxPhysicalTokens = 8;
    deployment.base.maxNumSequences = kResidentRows;
    deployment.base.recurrentPoolRows = kResidentRows;
    deployment.base.rotaryDim = 64;
    deployment.base.ropeConfig.type = RopeType::kMRope;
    deployment.base.specDecodeType = SpecDecodeMode::kMTP;

    LLMEngineConfig draft = deployment.base;
    draft.vocabSize = 16;
    deployment.draft = draft;
    SpecDecodeConfig spec;
    spec.baseOutputHiddenDim = 64;
    spec.draftHiddenSize = 64;
    spec.maxVerifySize = 2;
    spec.maxDraftProposalSize = 2;
    spec.draftingTopK = 1;
    spec.draftingStep = 1;
    spec.verifySize = 2;
    deployment.specConfig = spec;

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    {
        PipelineIO io = PipelineIO::createForSpecDecode(
            deployment, kRuntimeBatch, stream, /*hasAcceptHiddenOutput=*/false, /*hasTreeMetadataInputs=*/false);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(io.executionPhaseMarker.getMemoryCapacity(), 8 * static_cast<int64_t>(sizeof(int32_t)));
        EXPECT_EQ(
            io.contextSequenceCountCarrier.getMemoryCapacity(), kResidentRows * static_cast<int64_t>(sizeof(int32_t)));
        EXPECT_EQ(io.mropeCosSin.getShape(), Coords({kResidentRows, 8, 64}));
        EXPECT_EQ(io.mropeActiveCosSin.getShape(), Coords({kRuntimeBatch, 8, 64}));

        std::vector<float> activeRope(static_cast<size_t>(kRuntimeBatch * 8 * 64));
        for (int32_t activeRow = 0; activeRow < kRuntimeBatch; ++activeRow)
        {
            for (int32_t position = 0; position < 8; ++position)
            {
                for (int32_t dim = 0; dim < 64; ++dim)
                {
                    activeRope[static_cast<size_t>((activeRow * 8 + position) * 64 + dim)]
                        = static_cast<float>((5 - activeRow * 4) * 100 + position * 10 + dim);
                }
            }
        }
        std::array<int32_t, 2> const positions{1, 2};
        std::array<int32_t, 3> const queryStartOffsets{0, 1, 2};
        std::array<int32_t, 2> const queryLengths{1, 1};
        std::array<int32_t, 2> const stateIndices{5, 1};
        ASSERT_EQ(cudaMemcpyAsync(io.mropeActiveCosSin.rawPointer(), activeRope.data(),
                      activeRope.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      io.positions.rawPointer(), positions.data(), sizeof(positions), cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(io.queryStartOffsets.rawPointer(), queryStartOffsets.data(),
                      sizeof(queryStartOffsets), cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(io.queryLengths.rawPointer(), queryLengths.data(), sizeof(queryLengths),
                      cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(io.stateIndices.rawPointer(), stateIndices.data(), sizeof(stateIndices),
                      cudaMemcpyHostToDevice, stream),
            cudaSuccess);

        RaggedExecutionBatch batch;
        batch.shape.numSequences = kRuntimeBatch;
        batch.stateIndices.assign(stateIndices.begin(), stateIndices.end());
        scatterActiveMRopeToResident(io, batch, deployment.base, stream);
        EXPECT_EQ(io.mropeCosSin.getShape(), Coords({kResidentRows, 8, 64}));

        SharedResources resources;
        prepareRaggedRope(io, resources, deployment.base, /*physicalTokens=*/2, /*numSequences=*/2, stream);
        std::array<float, 128> gathered{};
        ASSERT_EQ(cudaMemcpyAsync(gathered.data(), io.raggedRopeCosSin.rawPointer(), sizeof(gathered),
                      cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        for (int32_t dim = 0; dim < 64; ++dim)
        {
            EXPECT_EQ(gathered[static_cast<size_t>(dim)], static_cast<float>(510 + dim));
            EXPECT_EQ(gathered[static_cast<size_t>(64 + dim)], static_cast<float>(120 + dim));
        }

        std::vector<float> residentSentinel(static_cast<size_t>(8 * 64), 777.0F);
        auto* residentBase = static_cast<float*>(io.mropeCosSin.rawPointer());
        ASSERT_EQ(cudaMemcpyAsync(residentBase + 5 * 8 * 64, residentSentinel.data(),
                      residentSentinel.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        kernel::initializeTextOnlyMRopeCosSin(io.mropeActiveCosSin.dataPointer<float>(),
            deployment.base.ropeConfig.rotaryTheta, deployment.base.rotaryDim, deployment.base.maxKVCacheCapacity,
            kRuntimeBatch, stream);
        batch.stateIndices = {1, 2};
        ASSERT_EQ(cudaMemcpyAsync(io.stateIndices.rawPointer(), batch.stateIndices.data(),
                      batch.stateIndices.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        scatterActiveMRopeToResident(io, batch, deployment.base, stream);

        std::vector<float> preserved(residentSentinel.size());
        std::vector<float> activeText(residentSentinel.size());
        std::vector<float> residentText(residentSentinel.size());
        ASSERT_EQ(cudaMemcpyAsync(preserved.data(), residentBase + 5 * 8 * 64, preserved.size() * sizeof(float),
                      cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(activeText.data(), io.mropeActiveCosSin.rawPointer(),
                      activeText.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(residentText.data(), residentBase + 1 * 8 * 64, residentText.size() * sizeof(float),
                      cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(preserved, residentSentinel);
        EXPECT_EQ(residentText, activeText);
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(PipelineIOSwaBindingTest, VanillaTextMRopeNarrowsActiveScratchBeforeResidentScatter)
{
    LLMEngineConfig cfg;
    cfg.hiddenSize = 64;
    cfg.outputVocabSize = 16;
    cfg.maxSupportedBatchSize = 8;
    cfg.maxSupportedInputLength = 4;
    cfg.maxKVCacheCapacity = 8;
    cfg.maxPhysicalTokens = 8;
    cfg.maxNumSequences = 6;
    cfg.recurrentPoolRows = 6;
    cfg.rotaryDim = 64;
    cfg.ropeConfig.type = RopeType::kMRope;

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    {
        PipelineIO io = PipelineIO::createForLLM(cfg, stream);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(io.mropeActiveCosSin.getShape(), Coords({8, 8, 64}));

        prepareTextOnlyMRope(io, cfg, /*activeRows=*/1, stream);
        EXPECT_EQ(io.mropeActiveCosSin.getShape(), Coords({1, 8, 64}));
        RaggedExecutionBatch batch;
        batch.shape.numSequences = 1;
        batch.stateIndices = {5};
        EXPECT_NO_THROW(scatterActiveMRopeToResident(io, batch, cfg, stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(PipelineIOSwaBindingTest, RaggedExecutionBindingsGatherResidentMRopeRows)
{
    LLMEngineConfig cfg;
    cfg.hiddenSize = 64;
    cfg.outputVocabSize = 32;
    cfg.numAttentionLayers = 1;
    cfg.numDecoderLayers = 1;
    cfg.numKVHeads = 1;
    cfg.headDim = 64;
    cfg.rotaryDim = 64;
    cfg.maxSupportedBatchSize = 2;
    cfg.maxSupportedInputLength = 4;
    cfg.maxKVCacheCapacity = 8;
    cfg.maxPhysicalTokens = 8;
    cfg.maxNumSequences = 2;
    cfg.recurrentPoolRows = 2;
    cfg.kvPoolPages = static_cast<int32_t>(computeMinimumKvPoolPages(cfg.recurrentPoolRows, cfg.maxKVCacheCapacity));
    cfg.kvCacheDtype = nvinfer1::DataType::kHALF;
    cfg.ropeConfig.type = RopeType::kMRope;
    cfg.layerTypes.assign(1, HybridCacheManager::LayerType::kAttention);
    cfg.kvLayerConfigs = {KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/64}};
    cfg.kvSharingDonors = {-1};

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    {
        auto resources = makeResources(cfg);
        PipelineIO io = PipelineIO::createForLLM(cfg, stream);
        Tensor residentRope({cfg.recurrentPoolRows, cfg.maxKVCacheCapacity, cfg.rotaryDim}, DeviceType::kCPU,
            nvinfer1::DataType::kFLOAT, "residentRopeStaging");
        float* const residentRopeData = residentRope.dataPointer<float>();
        for (int32_t row = 0; row < cfg.recurrentPoolRows; ++row)
        {
            for (int32_t position = 0; position < cfg.maxKVCacheCapacity; ++position)
            {
                for (int32_t dim = 0; dim < cfg.rotaryDim; ++dim)
                {
                    residentRopeData[(row * cfg.maxKVCacheCapacity + position) * cfg.rotaryDim + dim]
                        = static_cast<float>(row * 100 + position * 10 + dim);
                }
            }
        }
        ASSERT_EQ(cudaMemcpyAsync(io.mropeCosSin.rawPointer(), residentRope.rawPointer(),
                      residentRope.getMemoryCapacity(), cudaMemcpyHostToDevice, stream),
            cudaSuccess);

        RaggedExecutionBatch batch;
        batch.shape = {/*numSequences=*/2, /*validTokens=*/2, /*physicalTokens=*/2, /*queryWidth=*/1,
            /*numContextSequences=*/0, /*numContextTokens=*/0, /*numLogits=*/2};
        batch.positions = {3, 5};
        batch.queryStartOffsets = {0, 1, 2};
        batch.queryLengths = {1, 1};
        batch.pastLengths = {3, 5};
        batch.attentionSequenceLengths = {4, 6};
        batch.stateIndices = {1, 0};
        batch.logitsIndices = {0, 1};

        prepareRaggedExecutionBindings(io, *resources, cfg, batch, /*kvCacheIndex=*/0, stream);

        Tensor gatheredRope({2, cfg.rotaryDim}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "gatheredRopeStaging");
        ASSERT_EQ(cudaMemcpyAsync(gatheredRope.rawPointer(), io.raggedRopeCosSin.rawPointer(),
                      gatheredRope.getMemoryCapacity(), cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        float const* const gatheredRopeData = gatheredRope.dataPointer<float>();
        for (int32_t dim = 0; dim < cfg.rotaryDim; ++dim)
        {
            EXPECT_EQ(gatheredRopeData[dim], static_cast<float>(130 + dim));
            EXPECT_EQ(gatheredRopeData[cfg.rotaryDim + dim], static_cast<float>(50 + dim));
        }
        EXPECT_EQ(io.raggedKVPageTable.getShape(), Coords({2, 2, computeMaxPagesPerSeq(cfg.maxKVCacheCapacity)}));
        EXPECT_THROW(
            prepareRaggedExecutionBindings(io, *resources, cfg, batch, /*kvCacheIndex=*/1, stream), std::runtime_error);
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(PipelineIOSwaBindingTest, RaggedExecutionBindingsPrepareBoundedSwaAndDualRope)
{
    LLMEngineConfig cfg = makeBindingConfig(/*useSwa=*/true);
    cfg.maxPhysicalTokens = 8;
    cfg.maxNumSequences = 2;
    cfg.recurrentPoolRows = 2;
    cfg.useDualRope = true;
    cfg.slidingRotaryDim = 4;
    cfg.fullRotaryDim = 8;

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    {
        auto resources = makeResources(cfg);
        PipelineIO io = PipelineIO::createForLLM(cfg, stream);
        KVPageTable* const swaPageTable = resources->getSwaKVPageTable(0);
        ASSERT_NE(swaPageTable, nullptr);
        swaPageTable->setEntry(/*slot=*/0, /*logicalPage=*/0, /*kPageId=*/0);
        swaPageTable->setEntry(/*slot=*/1, /*logicalPage=*/0, /*kPageId=*/1);
        swaPageTable->uploadDirty(stream);

        RaggedExecutionBatch batch;
        batch.shape = {/*numSequences=*/2, /*validTokens=*/3, /*physicalTokens=*/4, /*queryWidth=*/2,
            /*numContextSequences=*/2, /*numContextTokens=*/3, /*numLogits=*/2};
        batch.positions = {4, 5, 7, -1};
        batch.queryStartOffsets = {0, 2, 4};
        batch.queryLengths = {2, 1};
        batch.pastLengths = {4, 7};
        batch.attentionSequenceLengths = {6, 8};
        batch.stateIndices = {1, 0};
        batch.logitsIndices = {1, 2};

        prepareRaggedExecutionBindings(io, *resources, cfg, batch, /*kvCacheIndex=*/0, stream);
        Tensor gatheredSwa({2}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "gatheredSwaStaging");
        int32_t const maxPagesPerSeq = computeMaxPagesPerSeq(cfg.maxKVCacheCapacity);
        ASSERT_EQ(cudaMemcpyAsync(gatheredSwa.rawPointer(), io.raggedSwaKVPageTable.rawPointer(), sizeof(int32_t),
                      cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(gatheredSwa.dataPointer<int32_t>() + 1,
                      static_cast<int32_t*>(io.raggedSwaKVPageTable.rawPointer()) + 2 * maxPagesPerSeq, sizeof(int32_t),
                      cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(gatheredSwa.dataPointer<int32_t>()[0], 1);
        EXPECT_EQ(gatheredSwa.dataPointer<int32_t>()[1], 0);
        EXPECT_EQ(io.raggedSwaKVPageTable.getShape(), Coords({2, 2, maxPagesPerSeq}));
        EXPECT_EQ(io.raggedRopeCosSinSliding.getShape(), Coords({4, cfg.slidingRotaryDim}));
        EXPECT_EQ(io.raggedRopeCosSinFull.getShape(), Coords({4, cfg.fullRotaryDim}));
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
