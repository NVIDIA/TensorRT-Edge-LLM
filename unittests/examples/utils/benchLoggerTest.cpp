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

#include "benchLogger.h"
#include "benchRunner.h"
#include "runtimeArgs.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>

using namespace trt_edgellm;
using namespace trt_edgellm::exampleUtils;

namespace
{
TEST(BenchLoggerTest, CountsActualBatchOutput)
{
    // Given uneven outputs in a batch, when reporting one request, then throughput counts every emitted token.
    rt::LLMGenerationResponse response;
    response.outputIds = {{1, 2, 3}, {4}};
    auto const sample = makeBenchSample(0, 0, 2.0, response, {});
    EXPECT_EQ(sample.at("output_tokens"), 4);
    EXPECT_EQ(sample.at("throughput_tps"), 2000.0);
}

TEST(BenchLoggerTest, KeepsRequestAndInvocationPopulationsSeparate)
{
    // Given two requests with unequal invocation counts, when summarizing, then request statistics stay unweighted.
    rt::LLMGenerationResponse response;
    timer::TimingSample invocation;
    invocation.stageId = "forward/base";
    invocation.forward = true;
    invocation.forwardIndex = 0;
    invocation.gpuTimeMs = 2.0F;
    auto const samples = nlohmann::json::array({makeBenchSample(0, 0, 10, response, {invocation}),
        makeBenchSample(1, 0, 20, response, {invocation, invocation, invocation})});
    auto const summary = summarizeBenchSamples(samples);
    EXPECT_EQ(summary.at("request_wall_ms").at("count"), 2);
    EXPECT_EQ(summary.at("request_wall_ms").at("mean"), 15.0);
    EXPECT_EQ(summary.at("invocation_gpu_ms").at("forward/base").at("count"), 4);
}

TEST(BenchLoggerTest, ExcludesCaptureFromSteadyStatisticsButRetainsItsCost)
{
    // Given a graph capture request, when reporting, then its real wall time remains in a separate population.
    rt::LLMGenerationResponse response;
    timer::TimingSample invocation;
    invocation.stageId = "graph_capture";
    auto const samples = nlohmann::json::array(
        {makeBenchSample(0, 0, 10, response, {}), makeBenchSample(1, 0, 100, response, {invocation})});
    auto const summary = summarizeBenchSamples(samples);
    EXPECT_EQ(summary.at("request_wall_ms").at("mean"), 10.0);
    EXPECT_EQ(summary.at("non_steady_request_wall_ms").at("mean"), 100.0);
}

TEST(BenchLoggerTest, SerializesSemanticForwardIdentityWithoutEngineDetails)
{
    // Given a semantic draft forward, when reporting it, then role, parent, ordinal and shape survive without
    // claiming engine identity or CUDA graph execution details.
    timer::TimingSample invocation;
    invocation.stageId = "forward/draft";
    invocation.parentStage = "proposal";
    invocation.requestId = 42;
    invocation.round = 2;
    invocation.forward = true;
    invocation.forwardIndex = 3;
    invocation.batchSize = 2;
    invocation.queryTokens = 8;
    invocation.selectedTokens = 2;
    invocation.pastLengths = {17, 19};
    invocation.gpuTimeMs = 2.5F;
    auto const sample = makeBenchSample(3, 1, 10.0, rt::LLMGenerationResponse{}, {invocation});
    nlohmann::json const expected{{"stage", "forward/draft"}, {"parent_stage", "proposal"}, {"request_id", 42},
        {"round", 2}, {"forward", true}, {"forward_index", 3}, {"batch_size", 2}, {"query_tokens", 8},
        {"selected_tokens", 2}, {"past_lengths", {17, 19}}, {"gpu_ms", 2.5}};
    EXPECT_EQ(sample.at("invocations").at(0), expected);
}

TEST(BenchLoggerTest, WritesSemanticForwardColumnsToCsv)
{
    // Given a forward with a parent stage and a per-sequence KV shape, when exporting CSV, then its semantic
    // identity and shape are preserved with correct field quoting.
    auto const directory = std::filesystem::temp_directory_path()
        / ("runtime_bench_forward_csv_test_"
            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    timer::TimingSample invocation;
    invocation.stageId = "forward/draft";
    invocation.parentStage = "proposal";
    invocation.requestId = 42;
    invocation.round = 2;
    invocation.forward = true;
    invocation.forwardIndex = 3;
    invocation.batchSize = 2;
    invocation.queryTokens = 8;
    invocation.selectedTokens = 2;
    invocation.pastLengths = {17, 19};
    invocation.gpuTimeMs = 2.5F;
    auto const sample = makeBenchSample(3, 1, 10.0, rt::LLMGenerationResponse{}, {invocation});
    writeBenchReport(
        directory, {{"schema_version", 2}, {"status", "complete"}, {"samples", nlohmann::json::array({sample})}});
    std::ifstream file(directory / "invocations.csv");
    std::ostringstream contents;
    contents << file.rdbuf();
    file.close();
    std::filesystem::remove_all(directory);
    EXPECT_EQ(contents.str(),
        "iteration,workload,request_id,round,stage,parent_stage,forward,forward_index,batch_size,"
        "query_tokens,selected_tokens,past_lengths,gpu_ms\n"
        "3,1,42,2,\"forward/draft\",\"proposal\",true,3,2,8,2,\"[17,19]\",2.5\n");
}

TEST(BenchLoggerTest, RefusesToOverwriteExistingReports)
{
    // Given an existing result directory, when writing a report, then existing data cannot be overwritten.
    auto const directory = std::filesystem::temp_directory_path()
        / ("runtime_bench_existing_report_test_"
            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    EXPECT_THROW(writeBenchReport(directory, {{"samples", nlohmann::json::array()}}), std::runtime_error);
    std::filesystem::remove_all(directory);
}

TEST(BenchWorkloadTest, FixedRoundBudgetUsesResolvedAcceptanceCapacity)
{
    // Given normalized block verification, when budgeting a fixed round, then reserve its whole accepted block
    // plus the prefill token; an explicit AL reserves exactly that accepted length including its bonus.
    for (auto const mode : {rt::SpecDecodeMode::kDFlash, rt::SpecDecodeMode::kJetSpec})
    {
        rt::DeploymentConfig deployment;
        deployment.base.specDecodeType = mode;
        deployment.draft.emplace();
        deployment.specConfig.emplace();
        deployment.specConfig->verifyTokenCount = 8;
        deployment.specConfig->draftBlockSize = 8;
        deployment.specConfig->maxPathLength = 8;
        EXPECT_EQ(fixedRoundOutputBudget(deployment, -1), 9);
        EXPECT_EQ(fixedRoundOutputBudget(deployment, 3), 4);
    }
}

TEST(BenchWorkloadTest, FixedRoundBudgetUsesDSparkTreePathLength)
{
    // Given a DSpark tree whose verify capacity exceeds one accepted path, when budgeting a fixed round, then reserve
    // the block path plus bonus and prefill rather than every verify node.
    rt::DeploymentConfig deployment;
    deployment.base.specDecodeType = rt::SpecDecodeMode::kDSpark;
    deployment.draft.emplace();
    deployment.draft->specDraftBlockSize = 7;
    deployment.specConfig.emplace();
    deployment.specConfig->draftTopK = 4;
    deployment.specConfig->verifyTokenCount = 64;
    deployment.specConfig->maxPathLength = 8;
    EXPECT_EQ(fixedRoundOutputBudget(deployment, -1), 9);
}

TEST(BenchWorkloadTest, FixedRoundBudgetPreservesVanillaAndDiffusionPrefillSemantics)
{
    // Given vanilla or block diffusion execution, when budgeting a fixed round, then only vanilla adds a prefill token.
    rt::DeploymentConfig deployment;
    EXPECT_EQ(fixedRoundOutputBudget(deployment, -1), 2);
    deployment.base.isDiffusionBackbone = true;
    deployment.base.diffusionCanvasLength = 32;
    EXPECT_EQ(fixedRoundOutputBudget(deployment, -1), 32);
}

TEST(BenchWorkloadTest, SyntheticInputsHaveExactLengthsAndRepeatableSeed)
{
    // Given a seed and token shape, when constructing synthetic input twice, then all sequences are exact and
    // reproducible.
    auto const first = makeSyntheticBenchRequest(2, 7, 11, 42);
    auto const second = makeSyntheticBenchRequest(2, 7, 11, 42);
    ASSERT_EQ(first.preTokenizedInputIds.size(), 2);
    EXPECT_EQ(first.preTokenizedInputIds[0].size(), 7);
    EXPECT_EQ(first.preTokenizedInputIds[1].size(), 7);
    EXPECT_EQ(first.preTokenizedInputIds, second.preTokenizedInputIds);
    EXPECT_NE(first.preTokenizedInputIds, makeSyntheticBenchRequest(2, 7, 11, 43).preTokenizedInputIds);
}

TEST(BenchWorkloadTest, RejectsStopStringsForFixedRound)
{
    // Given a stop string that may interrupt a round, when preparing fixed_round, then the conflicting policy is
    // rejected.
    auto request = makeSyntheticBenchRequest(1, 7, 11, 42);
    request.requests[0].stopStrings = {"stop"};
    EXPECT_THROW(prepareBenchRequest(request, 16, true, true, 42), std::invalid_argument);
}

TEST(BenchWorkloadTest, AppliesBudgetEosSeedAndPrivateCachePolicy)
{
    // Given a fresh request, when preparing a benchmark repetition, then its budget, EOS, seed and cache policy are
    // explicit.
    auto request = makeSyntheticBenchRequest(1, 7, 11, 42);
    prepareBenchRequest(request, 16, true, false, 99);
    EXPECT_EQ(request.maxGenerateLength, 16);
    EXPECT_TRUE(request.ignoreEos);
    EXPECT_EQ(request.requests[0].samplingSeed, 99);
    EXPECT_EQ(request.contextCacheLookupPolicy, rt::ContextCacheLookupPolicy::kBypass);
}

TEST(RuntimeArgsTest, SharedParserBuildsDeploymentWithExplicitSpecOverrides)
{
    // Given shared loading options, when parsing and constructing a deployment, then explicit draft values reach
    // runtime.
    RuntimeArgs args;
    parseRuntimeOption(RuntimeOptionId::ENGINE_DIR, "engine", args);
    parseRuntimeOption(RuntimeOptionId::CHECKPOINT_DIR, "weights", args);
    parseRuntimeOption(RuntimeOptionId::SPEC_DECODE, nullptr, args);
    parseRuntimeOption(RuntimeOptionId::SPEC_DRAFT_TOP_K, "2", args);
    parseRuntimeOption(RuntimeOptionId::SPEC_DRAFT_STEP, "3", args);
    parseRuntimeOption(RuntimeOptionId::SPEC_PROPOSAL_TOKENS, "7", args);
    parseRuntimeOption(RuntimeOptionId::SPEC_VERIFY_TOKENS, "8", args);
    auto const deployment = makeDeploymentOptions(args);
    EXPECT_EQ(deployment.engineDir, "engine");
    EXPECT_EQ(deployment.checkpointDir, "weights");
    ASSERT_TRUE(deployment.draftingConfig.has_value());
    EXPECT_EQ(deployment.draftingConfig->draftTopK, 2);
    EXPECT_EQ(deployment.draftingConfig->draftingStep, 3);
    EXPECT_EQ(deployment.draftingConfig->proposalTokens, 7);
    EXPECT_EQ(deployment.draftingConfig->verifyTokens, 8);
    EXPECT_TRUE(args.specDecodeArgs.draftStepSet);
    EXPECT_TRUE(args.specDecodeArgs.proposalTokensSet);
    EXPECT_TRUE(args.specDecodeArgs.verifyTokensSet);
}

TEST(RuntimeArgsTest, MixedEngineDirectorySelectsConfigByDecodingMode)
{
    // Given vanilla and speculative configs in the same directory, when selecting deployment defaults, then
    // speculative mode reads its base config and vanilla mode retains its own config path.
    auto const directory = std::filesystem::temp_directory_path()
        / ("runtime_args_mixed_config_test_"
            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    nlohmann::json const base{{"num_hidden_layers", 12}, {"num_key_value_heads", 4}, {"head_dim", 64},
        {"hidden_size", 768}, {"vocab_size", 32000}, {"kv_cache_dtype", "fp16"}, {"spec_decode_type", "dflash"},
        {"engine_role", "base"},
        {"dflash_config", {{"block_size", 8}, {"mask_token_id", 31999}, {"target_layer_ids", {1, 8}}}},
        {"builder_config",
            {{"max_batch_size", 1}, {"max_input_len", 128}, {"max_kv_cache_capacity", 256}, {"max_kv_pool_pages", 2},
                {"max_lora_rank", 0}, {"spec_base", true}, {"max_spec_verify_tokens", 16},
                {"ragged_backend", "entry_padded_compatibility"}}}};
    auto vanilla = base;
    vanilla["spec_decode_type"] = "none";
    vanilla["engine_role"] = "llm";
    vanilla["builder_config"]["spec_base"] = false;
    vanilla["builder_config"].erase("max_spec_verify_tokens");
    std::ofstream(directory / "base_config.json") << base;
    std::ofstream(directory / "config.json") << vanilla;
    RuntimeArgs args;
    args.engineDir = directory.string();
    EXPECT_EQ(getBaseConfigPath(args), directory / "config.json");
    args.specDecodeArgs.enabled = true;
    bool const resolved = applyEngineSpecDecodeDefaults(args);
    std::filesystem::remove_all(directory);
    ASSERT_TRUE(resolved);
    EXPECT_FALSE(args.specDecodeArgs.proposalTokensSet);
    EXPECT_FALSE(args.specDecodeArgs.verifyTokensSet);
}

//! Linear DFlash leaves omitted verification width to the unified native resolver.
TEST(RuntimeArgsTest, LeavesLinearVerifyWidthForUnifiedResolver)
{
    // Given a DFlash engine with explicit P=3, when Q is omitted, then the CLI preserves that distinction.
    auto const directory = std::filesystem::temp_directory_path()
        / ("runtime_args_dflash_explicit_proposal_test_"
            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    nlohmann::json const base{{"num_hidden_layers", 12}, {"num_key_value_heads", 4}, {"head_dim", 64},
        {"hidden_size", 768}, {"vocab_size", 32000}, {"kv_cache_dtype", "fp16"}, {"spec_decode_type", "dflash"},
        {"engine_role", "base"},
        {"dflash_config", {{"block_size", 8}, {"mask_token_id", 31999}, {"target_layer_ids", {1, 8}}}},
        {"builder_config",
            {{"max_batch_size", 1}, {"max_input_len", 128}, {"max_kv_cache_capacity", 256}, {"max_kv_pool_pages", 2},
                {"max_lora_rank", 0}, {"spec_base", true}, {"max_spec_verify_tokens", 8},
                {"ragged_backend", "entry_padded_compatibility"}}}};
    std::ofstream(directory / "base_config.json") << base;

    RuntimeArgs args;
    args.engineDir = directory.string();
    args.specDecodeArgs.enabled = true;
    args.specDecodeArgs.proposalTokens = 3;
    args.specDecodeArgs.proposalTokensSet = true;
    bool const resolved = applyEngineSpecDecodeDefaults(args);
    std::filesystem::remove_all(directory);

    ASSERT_TRUE(resolved);
    EXPECT_TRUE(args.specDecodeArgs.proposalTokensSet);
    EXPECT_FALSE(args.specDecodeArgs.verifyTokensSet);
}

TEST(RuntimeArgsTest, LeavesOmittedDSparkCapacityForUnifiedResolver)
{
    // Given a false-anchor DSpark engine, when CLI capacities are omitted, then the unified resolver owns P/Q defaults.
    auto const directory = std::filesystem::temp_directory_path()
        / ("runtime_args_dspark_defaults_test_"
            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    nlohmann::json base{{"num_hidden_layers", 12}, {"num_key_value_heads", 4}, {"head_dim", 64}, {"hidden_size", 768},
        {"vocab_size", 32000}, {"kv_cache_dtype", "fp16"}, {"spec_decode_type", "dspark"}, {"engine_role", "base"},
        {"dspark_config",
            {{"block_size", 7}, {"mask_token_id", 31999}, {"sample_from_anchor", false}, {"target_layer_ids", {1, 8}}}},
        {"builder_config",
            {{"max_batch_size", 1}, {"max_input_len", 128}, {"max_kv_cache_capacity", 256}, {"max_kv_pool_pages", 2},
                {"max_lora_rank", 0}, {"spec_base", true}, {"max_spec_verify_tokens", 8},
                {"ragged_backend", "entry_padded_compatibility"}}}};
    auto draft = base;
    draft["num_hidden_layers"] = 2;
    draft["engine_role"] = "draft";
    draft["base_model_hidden_size"] = 768;
    draft["builder_config"]["spec_base"] = false;
    draft["builder_config"]["spec_draft"] = true;
    draft["builder_config"]["max_spec_draft_tokens"] = 8;
    draft["builder_config"].erase("max_spec_verify_tokens");
    std::ofstream(directory / "base_config.json") << base;
    std::ofstream(directory / "draft_config.json") << draft;

    RuntimeArgs args;
    args.engineDir = directory.string();
    args.specDecodeArgs.enabled = true;
    bool const resolved = applyEngineSpecDecodeDefaults(args);
    std::filesystem::remove_all(directory);

    ASSERT_TRUE(resolved);
    EXPECT_FALSE(args.specDecodeArgs.proposalTokensSet);
    EXPECT_FALSE(args.specDecodeArgs.verifyTokensSet);
}

TEST(RuntimeArgsTest, AppliesDefaultVerifyTokensForIterativeTree)
{
    // Given an iterative tree engine and omitted runtime overrides, when resolving example defaults, then the
    // root-inclusive verification budget is forwarded to the native deployment configuration.
    auto const directory = std::filesystem::temp_directory_path()
        / ("runtime_args_iterative_tree_defaults_test_"
            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    nlohmann::json const base{{"num_hidden_layers", 12}, {"num_key_value_heads", 4}, {"head_dim", 64},
        {"hidden_size", 768}, {"vocab_size", 32000}, {"kv_cache_dtype", "fp16"}, {"spec_decode_type", "eagle3"},
        {"engine_role", "base"}, {"eagle_hidden_state_layers", {0, 5, 11}},
        {"builder_config",
            {{"max_batch_size", 1}, {"max_input_len", 128}, {"max_kv_cache_capacity", 256}, {"max_kv_pool_pages", 2},
                {"max_lora_rank", 0}, {"spec_base", true}, {"max_spec_verify_tokens", 60},
                {"ragged_backend", "entry_padded_compatibility"}}}};
    std::ofstream(directory / "base_config.json") << base;

    RuntimeArgs args;
    args.engineDir = directory.string();
    args.specDecodeArgs.enabled = true;
    bool const resolved = applyEngineSpecDecodeDefaults(args);
    auto const deployment = makeDeploymentOptions(args);
    std::filesystem::remove_all(directory);

    ASSERT_TRUE(resolved);
    ASSERT_TRUE(deployment.draftingConfig.has_value());
    EXPECT_EQ(deployment.draftingConfig->verifyTokens, 60);
}

TEST(RuntimeArgsTest, RejectsIndependentDraftWithoutSpecDecode)
{
    // Given draft weights without speculative decoding, when constructing deployment, then both entry points reject it.
    RuntimeArgs args;
    args.engineDir = "engine";
    args.draftCheckpointDir = "draft";
    EXPECT_THROW(makeDeploymentOptions(args), std::invalid_argument);
}
TEST(BenchLoggerTest, FingerprintsDistinguishDifferentOutputsOfEqualLength)
{
    // Given outputs with equal token counts, when content differs, then token and text fingerprints distinguish them.
    rt::LLMGenerationResponse first;
    first.outputIds = {{1, 2}};
    first.outputTexts = {"ab"};
    rt::LLMGenerationResponse second;
    second.outputIds = {{1, 3}};
    second.outputTexts = {"ac"};
    auto const a = makeBenchSample(0, 0, 1.0, first, {});
    auto const b = makeBenchSample(0, 0, 1.0, second, {});
    EXPECT_NE(a.at("output_token_fingerprints"), b.at("output_token_fingerprints"));
    EXPECT_NE(a.at("output_text_fingerprints"), b.at("output_text_fingerprints"));
}

TEST(BenchLoggerTest, FingerprintEncodingIsPortableAndPreservesExactBytes)
{
    // Given signed token IDs and binary UTF-8 text, when fingerprinting, then the documented encoding has fixed
    // results.
    EXPECT_EQ(fingerprintTokenIds({1, -1, 256}), "ec97e1fe0754e1ab");
    EXPECT_EQ(fingerprintBytes(std::string_view{"a\0\xc3\xa9", 4}), "a9fea4820c2f5004");
}

TEST(BenchLoggerTest, FingerprintsTheWholeBinaryInputFile)
{
    // Given an input file with an embedded zero byte, when fingerprinting, then every byte contributes to its identity.
    auto const path = std::filesystem::temp_directory_path()
        / ("runtime_bench_fingerprint_test_"
            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    {
        std::ofstream file(path, std::ios::binary);
        file.write("a\0\xc3\xa9", 4);
    }
    EXPECT_EQ(fingerprintFile(path), "a9fea4820c2f5004");
    std::filesystem::remove(path);
}

TEST(BenchLoggerTest, ReportAssemblyFailureCannotLeaveCompleteStatus)
{
    // Given a report with incomplete CSV fields, when writing fails, then its owned JSON report records failure.
    auto const directory = std::filesystem::temp_directory_path()
        / ("runtime_bench_failed_report_test_"
            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    nlohmann::json const report{{"status", "complete"}, {"samples", nlohmann::json::array({{{"iteration", 0}}})}};
    try
    {
        writeBenchReport(directory, report);
    }
    catch (std::exception const&)
    {
    }
    std::ifstream file(directory / "report.json");
    EXPECT_EQ(nlohmann::json::parse(file).at("status"), "failed");
    file.close();
    std::filesystem::remove_all(directory);
}

} // namespace
