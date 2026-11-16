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

#include "runtime/decoding/specDecodeUtils.h"
#include "common/cudaUtils.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/decoding/decodingStrategy.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/logitBias.h"
#include "substituteEngine.h"
#include "testUtils.h"
#include <gtest/gtest.h>

using namespace trt_edgellm::rt;

TEST(SpecTargetCommitTest, PreservesChainPrefixAndCompactsTreeAcrossBatch)
{
    // Given two verification batches with different accepted lengths, when committing through the shared helper,
    // then chains need no row indices and retain their stride, while trees commit the indexed path and lengths.
    using namespace trt_edgellm;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    HybridCacheManager::Config cacheConfig{};
    cacheConfig.maxBatchSize = 2;
    cacheConfig.layerTypes = {HybridCacheManager::LayerType::kAttention};
    cacheConfig.kvConfig = {1, 2, 128, {{1, 64}}, trt_edgellm::DataType::kHALF};
    HybridCacheManager cache(cacheConfig, stream);
    cache.setActiveBatchSize(2);
    SharedResources shared;
    shared.kvPageTables.push_back(std::make_unique<KVPageTable>(2, 1, cache.getKVCacheManager().numPages()));
    shared.kvPageTables[0]->setIdentity();
    shared.kvPageTables[0]->upload(stream);
    DeploymentConfig deployment{};
    deployment.base.maxSupportedBatchSize = 2;
    deployment.base.maxPhysicalTokens = 8;
    deployment.base.recurrentPoolRows = 2;
    deployment.specConfig.emplace();
    deployment.specConfig->baseOutputHiddenDim = 8;
    PipelineIO io;
    io.baseHiddenStates = Tensor({2, 4, 8}, DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    io.stateIndices = Tensor({2}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    copyHostToDevice<int32_t>(io.stateIndices, {0, 1});
    Tensor ids({2, 3}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor counts({2}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor rows({2, 3}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    copyHostToDevice<int32_t>(counts, {2, 3});
    copyHostToDevice<int32_t>(rows, {0, 2, -1, 0, 1, 3});
    substitute_engine::MockEngineExecutor executor;
    TensorMap tensorMap;
    EmbeddingData embedding;
    EmbeddingPreprocessor embeddingPreprocessor(embedding, deployment.base);
    StepPreparer stepPreparer(deployment.base);
    tokenizer::Tokenizer tokenizer;
    GuidedDecoder guided;
    LogitBias bias;
    Tensor unused;
    std::filesystem::path checkpoint;
    DecodingRuntimeContext runtime{deployment, 2, checkpoint, checkpoint, {executor, tensorMap, shared, cache, io, {}},
        {stepPreparer, embeddingPreprocessor, embedding, unused, nullptr, nullptr, nullptr}, tokenizer, bias, guided,
        {unused, unused, unused, unused, unused, unused, unused, unused, unused},
        {unused, unused, unused, unused, unused}};
    DecodingInferenceContext context;
    context.activeBatchSize = 2;
    context.stream = stream;
    for (bool tree : {false, true})
    {
        SCOPED_TRACE(tree);
        deployment.specConfig->verifyLayout = tree ? SpecVerifyLayout::kTree : SpecVerifyLayout::kChain;
        check::check(io.baseHiddenStates.reshape({2, 4, 8}), "Test hidden-state reshape failed");
        std::vector<half> hidden(64);
        for (int32_t row = 0; row < 8; ++row)
        {
            std::fill_n(hidden.begin() + row * 8, 8, __float2half(static_cast<float>(row + 1)));
        }
        copyHostToDevice<half>(io.baseHiddenStates, hidden);
        copyHostToDevice<int32_t>(cache.getKVCacheLengths(), {0, 0});
        auto [keys, values] = cache.getSeparateKVCache(0);
        std::vector<half> kv(keys.getShape().volume());
        for (int32_t row = 0; row < 256; ++row)
        {
            std::fill_n(kv.begin() + row * 64, 64, __float2half(static_cast<float>(row)));
        }
        copyHostToDevice<half>(keys, kv);
        copyHostToDevice<half>(values, kv);
        spec_decode::commitAcceptedTargetPath(
            runtime, context, SpecAcceptResult{ids, counts, tree ? &rows : nullptr, 3}, 4);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto actual = copyDeviceToHost<int32_t>(cache.getKVCacheLengths());
        int32_t const stride = io.baseHiddenStates.getShape()[1];
        actual.push_back(stride);
        auto const committedHidden = copyDeviceToHost<half>(io.baseHiddenStates);
        for (int32_t offset : {0, 1, stride, stride + 1, stride + 2})
        {
            actual.push_back(static_cast<int32_t>(__half2float(committedHidden[offset * 8])));
        }
        for (auto const* tensor : {&keys, &values})
        {
            auto const committed = copyDeviceToHost<half>(*tensor);
            for (int32_t offset : {0, 1, 128, 129, 130})
            {
                actual.push_back(static_cast<int32_t>(__half2float(committed[offset * 64])));
            }
        }
        std::vector<int32_t> const expected = tree
            ? std::vector<int32_t>{2, 3, 3, 1, 3, 5, 6, 8, 0, 2, 128, 129, 131, 0, 2, 128, 129, 131}
            : std::vector<int32_t>{2, 3, 4, 1, 2, 5, 6, 7, 0, 1, 128, 129, 130, 0, 1, 128, 129, 130};
        EXPECT_EQ(actual, expected);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SpecVerifyInputsTest, AcceptsRootInclusiveChainView)
{
    // Given a root-inclusive [N,Q] chain view within capacity, when validating
    // its descriptor, then no tree metadata is required.
    Tensor tokens({2, 4}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    SpecDecodeConfig config{};
    config.maxVerifyTokens = 8;
    config.maxPathLength = 4;
    SpecVerifyInputs inputs{tokens, nullptr, nullptr, nullptr, 2, 4, 4, SpecVerifyLayout::kChain};
    EXPECT_NO_THROW(spec_decode::validateSpecVerifyInputs(inputs, config));
}

TEST(SpecVerifyInputsTest, RejectsTreeWithoutReadyMetadata)
{
    // Given tree candidates and ancestry, when metadata is not ready,
    // then validation rejects the descriptor before target execution.
    Tensor tokens({8}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor parents({8}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor depths({8}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    SpecDecodeConfig config{};
    config.maxVerifyTokens = 8;
    config.maxPathLength = 4;
    config.verifyLayout = SpecVerifyLayout::kTree;
    SpecVerifyInputs inputs{tokens, nullptr, &parents, &depths, 2, 4, 4, SpecVerifyLayout::kTree};
    EXPECT_THROW(spec_decode::validateSpecVerifyInputs(inputs, config), std::runtime_error);
}

TEST(SpecVerifyInputsTest, RejectsWidthBeyondEngineCapacity)
{
    // Given a target with four verify rows per request, when a descriptor
    // requests eight, then validation fails instead of truncating it.
    Tensor tokens({8}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    SpecDecodeConfig config{};
    config.maxVerifyTokens = 4;
    config.maxPathLength = 4;
    SpecVerifyInputs inputs{tokens, nullptr, nullptr, nullptr, 1, 8, 4, SpecVerifyLayout::kChain};
    EXPECT_THROW(spec_decode::validateSpecVerifyInputs(inputs, config), std::runtime_error);
}

class SpecProbabilityAcceptTest : public ::testing::TestWithParam<int>
{
};

TEST_P(SpecProbabilityAcceptTest, PreservesProposalBonusAndEmbeddedBudget)
{
    // Given deterministic proposal and bonus distributions, when accepting through
    // each typed route, then tokens, counts and predictor rows retain its budget semantics.
    Tensor target({1, 2, 3}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT);
    Tensor draft({1, 1, 3}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT);
    Tensor targetIds({1, 2, 3}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor draftIds({1, 1, 3}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor proposals({1, 1}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor lengths({1}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor uniforms({1, 3}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT);
    Tensor ids({1, 2}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor counts({1}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor rows({1, 2}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor budget({1}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    copyHostToDevice<float>(target, {0, 1, 0, 0, 0, 1});
    copyHostToDevice<float>(draft, {0, 1, 0});
    copyHostToDevice<int32_t>(targetIds, {0, 1, 2, 0, 1, 2});
    copyHostToDevice<int32_t>(draftIds, {0, 1, 2});
    copyHostToDevice<int32_t>(proposals, {1});
    copyHostToDevice<int32_t>(lengths, {1});
    copyHostToDevice<float>(uniforms, {0.5F, 0.5F, 0.5F});
    copyHostToDevice<int32_t>(ids, {-99, -99});
    copyHostToDevice<int32_t>(counts, {-99});
    copyHostToDevice<int32_t>(rows, {-99, -99});
    copyHostToDevice<int32_t>(budget, {1});
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    int const route = GetParam() % 3;
    bool const limited = GetParam() >= 3;
    if (route == 0)
    {
        spec_decode::accept(
            DenseProbabilityAcceptInputs{target, draft, proposals, lengths, uniforms, 1, 1, 1, 3}, ids, counts, stream);
    }
    else if (route == 1)
    {
        spec_decode::accept(SparseProbabilityAcceptInputs{target, targetIds, draft, draftIds, proposals, lengths,
                                uniforms, limited ? &budget : nullptr},
            ids, counts, &rows, stream);
    }
    else
    {
        spec_decode::accept(DenseTargetSparseDraftAcceptInputs{target, draft, draftIds, proposals, lengths, uniforms,
                                limited ? &budget : nullptr},
            ids, counts, &rows, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto actual = copyDeviceToHost<int32_t>(counts);
    auto const tokens = copyDeviceToHost<int32_t>(ids);
    int const count = limited ? 1 : 2;
    actual.insert(actual.end(), tokens.begin(), tokens.begin() + count);
    std::vector<int32_t> expected{count, 1};
    if (!limited)
        expected.push_back(2);
    if (route != 0)
    {
        auto const indices = copyDeviceToHost<int32_t>(rows);
        actual.insert(actual.end(), indices.begin(), indices.begin() + count);
        expected.push_back(0);
        if (!limited)
            expected.push_back(1);
    }
    EXPECT_EQ(actual, expected);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

INSTANTIATE_TEST_SUITE_P(TypedRoutes, SpecProbabilityAcceptTest, ::testing::Values(0, 1, 2, 4, 5));

TEST(SpecGreedyAcceptTest, DispatchesBorrowedChainAndTreeWithIndependentOutputStride)
{
    // Given identical borrowed candidates with a non-chain parent map, when accepting naturally or at a forced
    // length, then tokens and predictor rows follow the selected layout without changing storage.
    Tensor tokens({4}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor target({4}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor parents({4}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor depths({4}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor ids({1, 3}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor rows({1, 3}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor counts({1}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    copyHostToDevice<int32_t>(tokens, {0, 7, 8, 7});
    copyHostToDevice<int32_t>(target, {7, 7, 9, 10});
    copyHostToDevice<int32_t>(parents, {-1, 0, 0, 1});
    copyHostToDevice<int32_t>(depths, {0, 1, 1, 2});
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    for (bool tree : {false, true})
    {
        SpecVerifyInputs verify{tokens, nullptr, &parents, &depths, 1, 4, 3,
            tree ? SpecVerifyLayout::kTree : SpecVerifyLayout::kChain, tree};
        spec_decode::accept(GreedyAcceptInputs{verify, target, nullptr}, ids, counts, &rows, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto actual = copyDeviceToHost<int32_t>(counts);
        auto const tokenValues = copyDeviceToHost<int32_t>(ids);
        auto const rowValues = copyDeviceToHost<int32_t>(rows);
        actual.insert(actual.end(), tokenValues.begin(), tokenValues.end());
        actual.insert(actual.end(), rowValues.begin(), rowValues.end());
        auto const expected
            = tree ? std::vector<int32_t>{3, 7, 7, 10, 0, 1, 3} : std::vector<int32_t>{2, 7, 7, 0, 0, 1, -1};
        EXPECT_EQ(actual, expected) << "tree=" << tree;

        spec_decode::accept(GreedyAcceptInputs{verify, target, nullptr}, ids, counts, &rows, stream, 3);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        actual = copyDeviceToHost<int32_t>(counts);
        auto const forcedTokens = copyDeviceToHost<int32_t>(ids);
        auto const forcedRows = copyDeviceToHost<int32_t>(rows);
        actual.insert(actual.end(), forcedTokens.begin(), forcedTokens.end());
        actual.insert(actual.end(), forcedRows.begin(), forcedRows.end());
        auto const forcedExpected
            = tree ? std::vector<int32_t>{3, 7, 7, 10, 0, 1, 3} : std::vector<int32_t>{3, 7, 8, 9, 0, 1, 2};
        EXPECT_EQ(actual, forcedExpected) << "forced tree=" << tree;
    }

    // Given no valid proposal row and a one-token output, when forcing one token, then the target bonus survives.
    Tensor noValid({1}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor oneToken({1, 1}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor oneRow({1, 1}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    copyHostToDevice<int32_t>(noValid, {0});
    SpecVerifyInputs oneWide{tokens, &noValid, &parents, &depths, 1, 4, 1, SpecVerifyLayout::kChain, false};
    spec_decode::accept(GreedyAcceptInputs{oneWide, target, &noValid}, oneToken, counts, &oneRow, stream, 1);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(copyDeviceToHost<int32_t>(counts), (std::vector<int32_t>{1}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(oneToken), (std::vector<int32_t>{7}));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
