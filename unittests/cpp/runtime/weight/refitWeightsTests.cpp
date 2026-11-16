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

#include "runtime/state/externalWeightManager.h"
#include "runtime/weight/refitWeights.h"
#include "scratchDir.h"
#include "testUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>
#include <type_traits>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;
using ::testing::HasSubstr;

namespace
{

//! Construct the minimal refit contract, without requiring a checkpoint.
Json weightlessConfig()
{
    return {{"weightless_engine", true},
        {"checkpoint_weight_bindings",
            Json::array({{{"engine_name", "linear.weights"}, {"destination", "refit"}, {"dtype", "F16"}}})}};
}

//! Capture a contract error; other exception types propagate and fail the test.
template <typename Fn>
std::string errorFrom(Fn&& fn)
{
    try
    {
        fn();
    }
    catch (std::runtime_error const& error)
    {
        return error.what();
    }
    return {};
}

TEST(WeightlessConfigTest, LegacyBindingsRemainInputs)
{
    // Given legacy input bindings, when validating, then no refit is requested.
    EXPECT_FALSE(validateWeightlessConfig(Json::object()));
    Json const config{{"checkpoint_weight_bindings", Json::array({{{"engine_name", "linear.weight"}}})}};
    EXPECT_FALSE(validateWeightlessConfig(config));
    EXPECT_FALSE(validateWeightlessConfig(Json{{"weightless_engine", false}}));
}

TEST(WeightlessConfigTest, MarkerAndRefitBindingsMustAppearTogether)
{
    // Given an incomplete contract, when validating, then the missing half is diagnosed.
    Json config = weightlessConfig();
    config.erase("weightless_engine");
    EXPECT_THAT(errorFrom([&] { validateWeightlessConfig(config); }), HasSubstr("present together"));
    config["weightless_engine"] = false;
    EXPECT_THAT(errorFrom([&] { validateWeightlessConfig(config); }), HasSubstr("present together"));
    config = weightlessConfig();
    config["checkpoint_weight_bindings"] = Json::array();
    EXPECT_THAT(errorFrom([&] { validateWeightlessConfig(config); }), HasSubstr("present together"));
}

TEST(WeightlessConfigTest, RejectsMalformedMetadata)
{
    // Given incompatible mode metadata, when validating, then it is rejected before loading.
    for (Json const& mode : std::vector<Json>{nullptr, 0, 1, "true", {{"version", 1}}})
    {
        SCOPED_TRACE(mode.dump());
        Json config = weightlessConfig();
        config["weightless_engine"] = mode;
        EXPECT_THAT(errorFrom([&] { validateWeightlessConfig(config); }), HasSubstr("must be a boolean"));
    }
}

TEST(WeightlessConfigTest, RejectsUnknownDestinations)
{
    // Given an unknown binding destination, when validating, then it cannot silently become an input.
    Json config = weightlessConfig();
    config["checkpoint_weight_bindings"][0]["destination"] = "refitt";
    EXPECT_THAT(
        errorFrom([&] { validateWeightlessConfig(config); }), HasSubstr("Unknown checkpoint weight destination"));
}

TEST(WeightlessConfigTest, RejectsUnsupportedRefitRoles)
{
    // Embeddings and plugin resources have lifetimes outside TensorRT refit.
    for (Json const& patch : std::vector<Json>{{{"role", "embedding"}}, {{"engine_name", "__embedding__"}},
             {{"plugin_resource_id", 0}}, {{"plugin_resource_kind", "weight"}}})
    {
        SCOPED_TRACE(patch.dump());
        Json config = weightlessConfig();
        config["checkpoint_weight_bindings"][0].update(patch);
        EXPECT_THAT(errorFrom([&] { validateWeightlessConfig(config); }), HasSubstr("cannot be refit weights"));
    }
}

TEST(WeightlessConfigTest, EnforcesBuildOptIn)
{
    // Given a valid weightless contract, when validating, then only an opted-in runtime accepts it.
#if defined(ENABLE_EXPERIMENTAL_WEIGHTLESS)
    EXPECT_TRUE(validateWeightlessConfig(weightlessConfig()));
#else
    EXPECT_THAT(errorFrom([] { validateWeightlessConfig(weightlessConfig()); }),
        HasSubstr("ENABLE_EXPERIMENTAL_WEIGHTLESS=ON"));
#endif
}

TEST(RefitWeightOwnershipTest, RequiresLoadBeforeTransfer)
{
    // Given a fresh manager, when transferring refit ownership, then an uninitialized load cannot escape.
    ExternalWeightManager manager;
    EXPECT_THAT(errorFrom([&] { manager.takeRefitWeights(); }), HasSubstr("before load"));
    static_assert(!std::is_copy_constructible_v<RefitWeightSet>);
    static_assert(std::is_move_constructible_v<RefitWeightSet>);
}

#if defined(ENABLE_EXPERIMENTAL_WEIGHTLESS)

class RefitCheckpointTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
        mDir = makeScratchDir("refitWeightsTests");
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
        std::filesystem::remove_all(mDir);
    }

    //! Write one aligned BF16 tensor reused by a refit matrix and a persistent embedding.
    Json writeCheckpoint()
    {
        std::vector<__nv_bfloat16> payload;
        for (float value : mValues)
        {
            payload.push_back(__float2bfloat16(value));
        }
        size_t const bytes = payload.size() * sizeof(payload[0]);
        Json const shape = Json::array({2, 2});
        Json const header{{"weight", {{"dtype", "BF16"}, {"shape", shape}, {"data_offsets", {0, bytes}}}}};
        std::string text = header.dump();
        text.resize((text.size() + 7) / 8 * 8, ' ');
        uint64_t const length = text.size();
        std::ofstream shard(mDir / "model.safetensors", std::ios::binary);
        shard.write(reinterpret_cast<char const*>(&length), sizeof(length));
        shard.write(text.data(), text.size());
        shard.write(reinterpret_cast<char const*>(payload.data()), bytes);
        shard.close();

        Json config = weightlessConfig();
        Json& refit = config["checkpoint_weight_bindings"][0];
        refit.update({{"shape", shape}, {"checkpoint_keys", {"weight"}}, {"source_layout", "checkpoint"}});
        Json embedding = refit;
        embedding.update({{"engine_name", "__embedding__"}, {"role", "embedding"}, {"destination", "input"}});
        config["checkpoint_weight_bindings"].push_back(embedding);
        Json const identity{{"dtype", "BF16"}, {"shape", shape}, {"bytes", bytes}, {"samples", Json::array()}};
        config["checkpoint_identity"]
            = {{"version", 1}, {"sources", {{"component", {{"tensors", {{"weight", identity}}}}}}}};
        return config;
    }

    //! Load through the production checkpoint reader, transforms and arena allocator.
    void load(ExternalWeightManager& manager, Json const& config)
    {
        std::filesystem::path const path = mDir / "config.json";
        std::ofstream(path) << config.dump();
        manager.load(mDir, path, mStream, mDir);
    }

    //! Read final FP16 values after startup has completed.
    std::vector<float> readValues(Tensor const& tensor)
    {
        std::vector<float> result;
        for (__half value : copyDeviceToHost<__half>(tensor))
        {
            result.push_back(__half2float(value));
        }
        return result;
    }

    cudaStream_t mStream{};
    std::filesystem::path mDir;
    std::vector<float> const mValues{1.0F, -2.0F, 0.5F, 4.0F};
};

TEST_F(RefitCheckpointTest, MovesSeparateArenaAndKeepsEmbeddingAlive)
{
    // Given tied checkpoint data, when moving and releasing refit ownership, then the embedding remains valid.
    ExternalWeightManager original;
    load(original, writeCheckpoint());
    ExternalWeightManager moved(std::move(original));
    ExternalWeightManager manager;
    manager = std::move(moved);
    EXPECT_EQ(manager.size(), 0U);
    auto embedding = manager.takeEmbedding();
    ASSERT_TRUE(embedding.has_value());
    auto refit = manager.takeRefitWeights();
    ASSERT_NE(refit, nullptr);
    ASSERT_EQ(refit->weights.size(), 1U);
    EXPECT_EQ(refit->weights[0].getName(), "linear.weights");
    EXPECT_EQ(refit->weights[0].rawPointer(), refit->storage.rawPointer());
    EXPECT_NE(refit->weights[0].rawPointer(), embedding->rawPointer());
    EXPECT_EQ(readValues(refit->weights[0]), mValues);
    EXPECT_THAT(errorFrom([&] { manager.takeRefitWeights(); }), HasSubstr("more than once"));
    EXPECT_THAT(errorFrom([&] { original.takeRefitWeights(); }), HasSubstr("before load"));
    EXPECT_THAT(errorFrom([&] { moved.takeRefitWeights(); }), HasSubstr("before load"));
    refit.reset();
    EXPECT_EQ(readValues(*embedding), mValues);
}

TEST_F(RefitCheckpointTest, RejectsCrossLifetimeAlias)
{
    // Given an embedding alias into the temporary arena, when loading, then no dangling persistent view is published.
    Json config = writeCheckpoint();
    config["checkpoint_weight_bindings"][1]["storage_alias_of"] = "linear.weights";
    ExternalWeightManager manager;
    EXPECT_THAT(errorFrom([&] { load(manager, config); }), HasSubstr("alias"));
    EXPECT_THAT(errorFrom([&] { manager.takeRefitWeights(); }), HasSubstr("before load"));
}

#endif

} // namespace
