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

// Gemma4 MTP speculative decoding assembled around two substitute engines.

#include "substituteEngine.h"

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{
// --------------------------------------------------------------------------
// Gemma4 MTP speculative decoding.
//
// The assistant shares the target's KV cache instead of owning one, which is
// why createDeploymentConfig checks far more here than for the other decoders:
// the two engines' paged-KV geometry has to be identical, since the assistant's
// profiles fix the pool shape and the runtime binds the target's pool to them.
// --------------------------------------------------------------------------

Json makeGemma4MtpBaseConfig()
{
    Json config = makeSpecBaseConfig("gemma4_mtp", kVerifySize);
    config["model"] = "gemma4_text";
    return config;
}

Json makeGemma4MtpDraftConfig(Json const& baseConfig)
{
    Json config = makeSpecDraftConfig("gemma4_mtp", kVerifySize);
    config["model"] = "gemma4_assistant";

    // The assistant reads the base hidden state unchanged and decodes the base vocabulary.
    config["base_model_hidden_size"] = baseConfig["hidden_size"];
    config["vocab_size"] = baseConfig["vocab_size"];
    // Not tuning knobs: createDeploymentConfig rejects an assistant that does not declare all four.
    config["shares_target_kv"] = true;
    config["has_own_kv_cache"] = false;
    config["returns_feedback_hidden"] = true;
    config["constant_draft_positions"] = true;
    // One entry per assistant attention layer, naming the target layer whose KV it reads.
    config["kv_sharing_map"] = Json::array({Json::object({{"assistant_layer", 0}, {"target_attention_layer", 0}})});
    return config;
}

class Gemma4MtpAssemblyTest : public SpecAssemblyTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        auto const base = makeGemma4MtpBaseConfig();
        return writeSpecModelDir("gemma4MtpAssemblyTests", base, makeGemma4MtpDraftConfig(base));
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeMtpDrafting();
    }
};

// Given an assistant declaring that it shares the target's KV cache and owns none
// When the pair is assembled
// Then the deployment carries that sharing forward instead of allocating the assistant a pool
TEST_F(Gemma4MtpAssemblyTest, AssemblesAnAssistantThatSharesTheTargetKvCache)
{
    auto artifacts = makeArtifacts();

    // The property the whole deployment is built around, and the reason its config validation is the strictest of
    // the speculative modes.
    ASSERT_TRUE(artifacts.deployment.draft.has_value());
    EXPECT_TRUE(artifacts.deployment.draft->sharesTargetKV);
    EXPECT_FALSE(artifacts.deployment.draft->hasOwnKVCache);

    auto runtime = makeRuntime(std::move(artifacts));

    ASSERT_FALSE(std::filesystem::exists(mModelDir / "spec_base.engine"));
    EXPECT_TRUE(runtime.hasDraftModel());
    EXPECT_STREQ(runtime.getSpeculativeDecodingStrategyName(), "gemma4_mtp");
}

// Given an assistant whose KV pool geometry disagrees with the target it shares
// When the pair is assembled
// Then assembly refuses it, naming the mismatch
TEST_F(Gemma4MtpAssemblyTest, RejectsAnAssistantWhoseKvPoolGeometryDiffersFromTheTarget)
{
    // The assistant's engine profiles fix the pool page count and page-table width, and the runtime binds the
    // target's pool to those bindings. There is no min() escape hatch as there is for batch size, so a mismatch
    // cannot be reconciled at run time -- it has to be refused here rather than deep inside a TensorRT shape check.
    auto const base = makeGemma4MtpBaseConfig();
    Json draft = makeGemma4MtpDraftConfig(base);
    // Kept internally consistent -- capacity and pool pages moved together -- so this reaches the cross-engine
    // check rather than tripping the per-engine sizing rule first.
    int64_t const draftCapacity = 2 * base["builder_config"]["max_kv_cache_capacity"].get<int64_t>();
    draft["builder_config"]["max_kv_cache_capacity"] = draftCapacity;
    draft["builder_config"]["max_kv_pool_pages"] = rt::computeMinimumKvPoolPages(kMaxBatchSize, draftCapacity);
    std::ofstream(mModelDir / "draft_config.json") << draft.dump(2);

    std::string message;
    try
    {
        auto artifacts = makeArtifacts();
        FAIL() << "a mismatched assistant KV pool was accepted";
    }
    catch (std::exception const& e)
    {
        message = e.what();
    }
    EXPECT_THAT(message, ::testing::HasSubstr("shared-KV pool geometry mismatch"));
}

} // namespace
