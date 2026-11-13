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
#include "benchSourceRevision.h"
#include "requestFileParser.h"
#include "runtimeArgs.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <tensorrt_edgellm/common/bindingNames.h>
#include <tensorrt_edgellm/common/checkMacros.h>
#include <tensorrt_edgellm/common/cudaUtils.h>
#include <tensorrt_edgellm/common/fileUtils.h>
#include <tensorrt_edgellm/common/logger.h>
#include <tensorrt_edgellm/common/tensor.h>
#include <tensorrt_edgellm/common/trtUtils.h>
#include <tensorrt_edgellm/common/version.h>
#include <tensorrt_edgellm/kernels/speculative/dflashRuntimeKernels.h>
#include <tensorrt_edgellm/kernels/speculative/speculativeKernels.h>
#include <tensorrt_edgellm/multimodal/common/multimodalRunner.h>
#include <tensorrt_edgellm/profiling/nvtx_wrapper.h>
#include <tensorrt_edgellm/runtime/config/deploymentConfig.h>
#include <tensorrt_edgellm/runtime/config/inferenceDims.h>
#include <tensorrt_edgellm/runtime/config/inferencePhase.h>
#include <tensorrt_edgellm/runtime/config/llmEngineConfig.h>
#include <tensorrt_edgellm/runtime/exec/engineExecutor.h>
#include <tensorrt_edgellm/runtime/exec/raggedBatchBuilder.h>
#include <tensorrt_edgellm/runtime/exec/tensorMap.h>
#include <tensorrt_edgellm/runtime/features/deepstackBinding.h>
#include <tensorrt_edgellm/runtime/llmInferenceRuntime.h>
#include <tensorrt_edgellm/runtime/preprocess/stepPreparer.h>
#include <tensorrt_edgellm/runtime/state/pipelineIO.h>
#include <tensorrt_edgellm/runtime/state/sharedResources.h>

using namespace trt_edgellm;
using Json = nlohmann::json;

namespace
{
struct BenchArgs : exampleUtils::RuntimeArgs
{
    BenchArgs()
    {
        contextCacheConfig.encoderEmbeddingCacheBudgetBytes = 0;
    }

    std::string mode{"fixed_round"};
    std::string inputFile;
    std::string outputDir;
    int32_t batchSize{-1};
    int32_t inputLen{2048};
    int32_t outputLen{128};
    int32_t iterations{10};
    int32_t warmup{3};
    int32_t forcedAcceptanceLength{-1};
    std::optional<uint64_t> seed;
    bool ignoreEos{true};
    bool gpuTiming{true};
    bool help{false};
    bool debug{false};
    bool inputLenSet{false};
};

void usage(char const* executable)
{
    std::cout
        << "Usage: " << executable
        << " --engine-dir DIR --checkpoint-dir DIR --output-dir NEW_DIR [options]\n"
           "  --mode fixed_round|generate   One complete decode round or continuous generation\n"
           "  --input-len N --batch-size N    Exact synthetic token lengths (defaults: 2048, 1)\n"
           "  --input-file FILE             llm_inference request JSON, including media and LoRA\n"
           "  --osl N                      Output budget for generate (default: 128)\n"
           "  --warmup N --iterations N    Whole-request repetitions per workload (defaults: 3, 10)\n"
           "  --force-al N                  Force acceptance length N >= 1, including the target bonus\n"
           "                               N=1 accepts zero drafts; omitted uses natural acceptance\n"
           "  --seed N                     Override input and sampling seed\n"
           "  --respect-eos                 Allow EOS termination (default: ignore EOS)\n"
           "  --no-gpu-timing                Wall time, NVTX and acceptance without CUDA-event timing\n"
           "  --debug                      Enable verbose runtime logging\n"
           "  --draft-checkpoint-dir DIR --multimodal-engine-dir DIR\n"
           "  --spec-decode --spec-draft-top-k N --spec-draft-step N --spec-proposal-tokens N --spec-verify-tokens N\n"
           "  Other deployment options have the same names/defaults as llm_inference.\n"
           "  Context and encoder reuse are disabled; --encoder-cache-budget-bytes must be 0.\n"
           "  Use nsys for kernel/layer analysis.\n"
           "  Legacy --profile/--extract-layer-info and component modes have been removed.\n";
}

template <typename T>
T integer(char const* value, T minimum)
{
    std::string_view const input{value};
    T result{};
    auto const [end, error] = std::from_chars(input.data(), input.data() + input.size(), result);
    if (error != std::errc{} || end != input.data() + input.size() || result < minimum)
    {
        throw std::invalid_argument("Invalid integer argument: " + std::string(input));
    }
    return result;
}

BenchArgs parseArgs(int argc, char** argv)
{
    enum : int
    {
        kHELP = 2000,
        kDEBUG,
        kMODE,
        kINPUT_LEN,
        kBATCH,
        kOSL,
        kWARMUP,
        kITERATIONS,
        kFORCE_AL,
        kSEED,
        kINPUT_FILE,
        kOUTPUT_DIR,
        kRESPECT_EOS,
        kNO_GPU_TIMING,
        kLEGACY
    };
    auto options = exampleUtils::runtimeOptions();
    std::vector<option> const benchOptions{{"help", no_argument, nullptr, kHELP},
        {"debug", no_argument, nullptr, kDEBUG}, {"mode", required_argument, nullptr, kMODE},
        {"input-len", required_argument, nullptr, kINPUT_LEN}, {"inputLen", required_argument, nullptr, kINPUT_LEN},
        {"batch-size", required_argument, nullptr, kBATCH}, {"batchSize", required_argument, nullptr, kBATCH},
        {"osl", required_argument, nullptr, kOSL}, {"warmup", required_argument, nullptr, kWARMUP},
        {"iterations", required_argument, nullptr, kITERATIONS}, {"force-al", required_argument, nullptr, kFORCE_AL},
        {"forceAL", required_argument, nullptr, kFORCE_AL}, {"seed", required_argument, nullptr, kSEED},
        {"input-file", required_argument, nullptr, kINPUT_FILE}, {"inputFile", required_argument, nullptr, kINPUT_FILE},
        {"output-dir", required_argument, nullptr, kOUTPUT_DIR}, {"outputDir", required_argument, nullptr, kOUTPUT_DIR},
        {"respect-eos", no_argument, nullptr, kRESPECT_EOS}, {"respectEos", no_argument, nullptr, kRESPECT_EOS},
        {"no-gpu-timing", no_argument, nullptr, kNO_GPU_TIMING}, {"noGpuTiming", no_argument, nullptr, kNO_GPU_TIMING},
        {"profile", no_argument, nullptr, kLEGACY}, {"extract-layer-info", optional_argument, nullptr, kLEGACY},
        {"extractLayerInfo", optional_argument, nullptr, kLEGACY}, {"past-kv-len", required_argument, nullptr, kLEGACY},
        {"pastKVLen", required_argument, nullptr, kLEGACY}, {nullptr, 0, nullptr, 0}};
    options.insert(options.end(), benchOptions.begin(), benchOptions.end());
    BenchArgs args;
    int id;
    while ((id = getopt_long(argc, argv, "", options.data(), nullptr)) != -1)
    {
        if (auto const parsed = exampleUtils::parseRuntimeOption(id, optarg, args))
        {
            if (!*parsed)
            {
                throw std::invalid_argument("Invalid deployment option");
            }
            continue;
        }
        switch (id)
        {
        case kHELP: args.help = true; break;
        case kDEBUG: args.debug = true; break;
        case kMODE: args.mode = optarg; break;
        case kINPUT_LEN:
            args.inputLen = integer<int32_t>(optarg, 1);
            args.inputLenSet = true;
            break;
        case kBATCH: args.batchSize = integer<int32_t>(optarg, 1); break;
        case kOSL: args.outputLen = integer<int32_t>(optarg, 1); break;
        case kWARMUP: args.warmup = integer<int32_t>(optarg, 0); break;
        case kITERATIONS: args.iterations = integer<int32_t>(optarg, 1); break;
        case kFORCE_AL: args.forcedAcceptanceLength = integer<int32_t>(optarg, 1); break;
        case kSEED: args.seed = integer<uint64_t>(optarg, 0); break;
        case kINPUT_FILE: args.inputFile = optarg; break;
        case kOUTPUT_DIR: args.outputDir = optarg; break;
        case kRESPECT_EOS: args.ignoreEos = false; break;
        case kNO_GPU_TIMING: args.gpuTiming = false; break;
        case kLEGACY:
            throw std::invalid_argument(
                "Legacy component/layer profiling options were removed; use fixed_round "
                "with --input-len to establish context, and nsys for kernel analysis");
        default: throw std::invalid_argument("Unknown benchmark option; use --help");
        }
    }
    if (args.help)
    {
        return args;
    }
    if (optind != argc || args.engineDir.empty() || args.outputDir.empty())
    {
        throw std::invalid_argument("--engine-dir and --output-dir are required; positional arguments are unsupported");
    }
    if (args.mode != "fixed_round" && args.mode != "generate")
    {
        throw std::invalid_argument("Supported modes: fixed_round, generate. Legacy component modes were removed");
    }
    if (args.contextCacheConfig.enabled)
    {
        throw std::invalid_argument("Benchmark repetitions require context reuse disabled");
    }
    if (args.contextCacheConfig.encoderEmbeddingCacheBudgetBytes != 0)
    {
        throw std::invalid_argument(
            "--encoder-cache-budget-bytes must be 0: encoder reuse conflicts with the no-reuse benchmark workload");
    }
    if (args.inputLenSet && !args.inputFile.empty())
    {
        throw std::invalid_argument("--input-len and --input-file are mutually exclusive");
    }
    if (std::filesystem::exists(args.outputDir))
    {
        throw std::invalid_argument("--output-dir must be a new directory; existing results are never overwritten");
    }
    if (!exampleUtils::applyEngineSpecDecodeDefaults(args))
    {
        throw std::invalid_argument("Could not resolve engine drafting defaults");
    }
    return args;
}

Json loadConfig(std::filesystem::path const& path)
{
    std::ifstream file(path);
    return Json::parse(file);
}

Json deploymentMetadata(
    BenchArgs const& args, std::optional<rt::SpecDecodeDraftingConfig> const& normalized = std::nullopt)
{
    auto const& spec = args.specDecodeArgs;
    auto const drafting = normalized.value_or(rt::SpecDecodeDraftingConfig{});
    auto const optionalInteger
        = [](std::optional<int32_t> value) { return value.has_value() ? Json(*value) : Json(nullptr); };
    return {{"engine_dir", std::filesystem::absolute(args.engineDir).string()}, {"checkpoint_dir", args.checkpointDir},
        {"draft_checkpoint_dir", args.draftCheckpointDir}, {"multimodal_engine_dir", args.multimodalEngineDir},
        {"rtx_runtime_cache_path", args.rtxRuntimeCachePath}, {"spec_decode", spec.enabled},
        {"draft_top_k", optionalInteger(normalized ? drafting.draftTopK : std::optional<int32_t>{spec.draftTopK})},
        {"draft_step", optionalInteger(normalized ? drafting.draftingStep : std::optional<int32_t>{spec.draftStep})},
        {"proposal_tokens",
            optionalInteger(normalized
                    ? drafting.proposalTokens
                    : (spec.proposalTokensSet ? std::optional<int32_t>{spec.proposalTokens} : std::nullopt))},
        {"verify_tokens",
            optionalInteger(normalized
                    ? drafting.verifyTokens
                    : (spec.verifyTokensSet ? std::optional<int32_t>{spec.verifyTokens} : std::nullopt))},
        {"dspark_w2_backend", static_cast<int32_t>((normalized ? drafting.dsparkW2Backend : spec.dsparkW2Backend))},
        {"dspark_scheduler",
            static_cast<int32_t>((normalized ? drafting.dsparkSchedulerMode : spec.dsparkSchedulerMode))},
        {"dspark_confidence_threshold",
            (normalized ? drafting.dsparkConfidenceThreshold : spec.dsparkConfidenceThreshold)},
        {"dspark_min_proposal_len", (normalized ? drafting.dsparkMinProposalLen : spec.dsparkMinProposalLen)},
        {"dspark_max_proposal_len", (normalized ? drafting.dsparkMaxProposalLen : spec.dsparkMaxProposalLen)},
        {"ple_residency", static_cast<int32_t>(args.pleConfig.residency)},
        {"ple_cache_slots", args.pleConfig.cacheSlots}, {"ple_file", args.pleConfig.filePath}, {"context_reuse", false},
        {"encoder_cache_budget_bytes", args.contextCacheConfig.encoderEmbeddingCacheBudgetBytes},
        {"visual_prune", args.visualPrunerConfig.enabled},
        {"visual_prune_algorithm", args.visualPrunerConfig.algorithm},
        {"visual_prune_reduction_ratio", args.visualPrunerConfig.reductionRatio},
        {"visual_prune_image_pivots", args.visualPrunerConfig.pivotImageTokens},
        {"visual_prune_text_pivots", args.visualPrunerConfig.pivotTextTokens}};
}
} // namespace

int main(int argc, char** argv)
{
    std::unique_ptr<void, DlDeleter> pluginHandles;
    cudaStream_t stream{};
    std::unique_ptr<rt::LLMInferenceRuntime> runtime;
    std::string reportDirectory;
    Json samples = Json::array();
    Json report;
    try
    {
        auto args = parseArgs(argc, argv);
        if (args.help)
        {
            usage(argv[0]);
            return EXIT_SUCCESS;
        }
        gLogger.setLevel(args.debug ? trt_edgellm::logger::Severity::kVERBOSE : trt_edgellm::logger::Severity::kINFO);
        reportDirectory = args.outputDir;
        report = {{"schema_version", 2}, {"mode", args.mode}, {"deployment", deploymentMetadata(args)}};
        auto const configPath = exampleUtils::getBaseConfigPath(args);
        auto options = exampleUtils::makeDeploymentOptions(args);
        auto const deployment = rt::createDeploymentConfig(configPath,
            args.specDecodeArgs.enabled
                ? std::make_optional(std::filesystem::path(args.engineDir) / "draft_config.json")
                : std::nullopt,
            options.draftingConfig);
        options.draftingConfig = deployment.draftingConfig();
        auto const& config = deployment.base;
        bool const fixedRound = args.mode == "fixed_round";
        int64_t const outputBudget = fixedRound
            ? exampleUtils::fixedRoundOutputBudget(deployment, args.forcedAcceptanceLength)
            : args.outputLen;
        report["deployment"] = deploymentMetadata(args, options.draftingConfig);
        std::unordered_map<std::string, std::string> loraWeights;
        std::vector<rt::LLMGenerationRequest> workloads;
        Json inputFileFingerprint = nullptr;
        if (!args.inputFile.empty())
        {
            inputFileFingerprint = exampleUtils::fingerprintFile(args.inputFile);
            auto parsed = exampleUtils::parseRequestFile(args.inputFile, args.batchSize, outputBudget);
            loraWeights = std::move(parsed.first);
            workloads = std::move(parsed.second);
        }
        else
        {
            workloads.push_back(exampleUtils::makeSyntheticBenchRequest(args.batchSize > 0 ? args.batchSize : 1,
                args.inputLen, config.vocabSize, args.seed.value_or(rt::kDefaultSamplingSeed)));
        }
        if (workloads.empty())
        {
            throw std::invalid_argument("No benchmark requests");
        }
        // Always seed: a seeded sequence draws from its own request-stable stream, which the report
        // states and the benchmark replays.
        std::optional<uint64_t> const samplingSeed = args.seed.value_or(rt::kDefaultSamplingSeed);
        for (auto& request : workloads)
        {
            exampleUtils::prepareBenchRequest(request, outputBudget, args.ignoreEos, fixedRound, samplingSeed);
        }
        options.loraWeightsMap = std::move(loraWeights);
        options.benchmark
            = rt::RuntimeBenchmarkOptions{fixedRound ? 1 : 0, args.forcedAcceptanceLength, args.gpuTiming};
        pluginHandles = loadEdgellmPluginLib();
        if (!pluginHandles)
        {
            throw std::runtime_error("Failed to load EdgeLLM plugin library");
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        runtime = std::make_unique<rt::LLMInferenceRuntime>(std::move(options), stream);
        if (args.visualPrunerConfig.enabled)
        {
            runtime->setVisualPrunerConfig(args.visualPrunerConfig);
        }
        setProfilingEnabled(false);
        bool const captured = runtime->captureDecodingCUDAGraph(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        Json workloadMetadata = Json::array();
        for (size_t workload = 0; workload < workloads.size(); ++workload)
        {
            auto const& request = workloads[workload];
            Json seeds = Json::array();
            Json inputFingerprints = Json::array();
            for (auto const& ids : request.preTokenizedInputIds)
            {
                inputFingerprints.push_back(exampleUtils::fingerprintTokenIds(ids));
            }
            for (auto const& sequence : request.requests)
            {
                seeds.push_back(
                    sequence.samplingSeed.value_or(request.samplingSeed.value_or(rt::kDefaultSamplingSeed)));
            }
            workloadMetadata.push_back({{"batch_size", request.requests.size()}, {"sampling_seeds", seeds},
                {"temperature", request.temperature}, {"top_p", request.topP}, {"top_k", request.topK},
                {"output_budget", request.maxGenerateLength},
                {"diffusion_max_denoising_steps_override", request.diffusionMaxDenoisingSteps},
                {"lora_name", request.loraWeightsName}, {"synthetic", args.inputFile.empty()},
                {"pretokenized_input_fingerprints", inputFingerprints}});
            for (int32_t iteration = -args.warmup; iteration < args.iterations; ++iteration)
            {
                setProfilingEnabled(true);
                gTimer.reset();
                rt::LLMGenerationResponse response;
                bool success;
                double wallMs;
                {
                    std::string const rangeName
                        = "bench/workload=" + std::to_string(workload) + "/iteration=" + std::to_string(iteration);
                    NVTX_SCOPED_RANGE(benchmarkRequest, rangeName.c_str());
                    auto const start = std::chrono::steady_clock::now();
                    success = runtime->handleRequest(request, response, stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    wallMs
                        = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                }
                if (!success)
                {
                    samples.push_back({{"iteration", iteration}, {"workload", workload}, {"wall_ms", wallMs},
                        {"status", "failed"}, {"invocations", Json::array()}});
                    throw std::runtime_error("handleRequest failed for workload " + std::to_string(workload)
                        + " iteration " + std::to_string(iteration));
                }
                if (iteration >= 0)
                {
                    auto sample = exampleUtils::makeBenchSample(iteration, static_cast<int32_t>(workload), wallMs,
                        response, gTimer.getSamples(), gTimer.getAcceptanceSamples());
                    sample["stop_policy"] = fixedRound ? "round_budget" : "token_budget_or_eos";
                    samples.push_back(std::move(sample));
                    if (fixedRound
                        && (response.decodeRoundCounts.size() != request.requests.size()
                            || !std::all_of(response.decodeRoundCounts.begin(), response.decodeRoundCounts.end(),
                                [](int32_t rounds) { return rounds == 1; })))
                    {
                        throw std::runtime_error("fixed_round request did not execute exactly one decode round");
                    }
                }
            }
        }
        setProfilingEnabled(false);
        int device{};
        cudaDeviceProp properties{};
        int driverVersion{};
        int runtimeVersion{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
        report = {{"schema_version", 2}, {"runtime_version", version::kRUNTIME_VERSION}, {"mode", args.mode},
            {"deployment", deploymentMetadata(args, deployment.draftingConfig())},
            {"engine_config", loadConfig(configPath)}, {"gpu", properties.name}, {"cuda_driver_version", driverVersion},
            {"cuda_runtime_version", runtimeVersion}, {"tensorrt_version", NV_TENSORRT_VERSION},
            {"warmup", args.warmup}, {"iterations", args.iterations}, {"ignore_eos", args.ignoreEos},
            {"forced_acceptance_length", args.forcedAcceptanceLength}, {"gpu_timing", args.gpuTiming},
            {"initial_graph_capture_succeeded", captured}, {"workloads", workloadMetadata}, {"samples", samples},
            {"statistics", exampleUtils::summarizeBenchSamples(samples)},
            {"statistics_definition", "population stddev; upper median; sorted[min(floor(N*p),N-1)] percentile"}};
        report["fingerprints"]
            = {{"algorithm", "FNV-1a-64, noncryptographic"}, {"representation", "16 lowercase hexadecimal digits"},
                {"token_encoding", "per-sequence int32 token IDs, little-endian bytes, no separators"},
                {"text_encoding", "per-sequence output text bytes, without normalization"},
                {"input_file_scope", "request JSON bytes only; referenced media content is not included"},
                {"input_file", inputFileFingerprint}};
        report["status"] = "complete";
#if defined(EDGELLM_BENCH_SOURCE_REVISION)
        report["source_revision"] = EDGELLM_BENCH_SOURCE_REVISION;
        report["source_dirty_at_build"] = static_cast<bool>(EDGELLM_BENCH_SOURCE_DIRTY);
#endif
        auto const draftConfigPath = std::filesystem::path(args.engineDir) / "draft_config.json";
        if (std::filesystem::is_regular_file(draftConfigPath))
        {
            report["draft_engine_config"] = loadConfig(draftConfigPath);
        }
        exampleUtils::writeBenchReport(args.outputDir, report);
        std::cout << report.at("statistics").dump(2) << '\n';
        std::cout << "Benchmark report: " << (std::filesystem::path(args.outputDir) / "report.json") << '\n';
        // Drain the WHOLE device before tearing anything down: the timed loop synchronized only
        // `stream`, so GPU work still in flight on the runtime's own streams would otherwise
        // surface as a sticky error inside the destructors below, after the report was written.
        // The elapsed time is logged so a slow drain (a driver stall) is visible evidence rather
        // than a silent gap before the error.
        {
            auto const drainStart = std::chrono::steady_clock::now();
            cudaError_t const drainStatus = cudaDeviceSynchronize();
            double const drainMs
                = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - drainStart).count();
            LOG_INFO("Device drain before teardown: %.1f ms (%s)", drainMs, cudaGetErrorString(drainStatus));
            CUDA_CHECK(drainStatus);
        }
        runtime.reset();
        auto const streamStatus = cudaStreamDestroy(stream);
        stream = nullptr;
        CUDA_CHECK(streamStatus);
        return EXIT_SUCCESS;
    }
    catch (std::exception const& error)
    {
        setProfilingEnabled(false);
        if (stream != nullptr)
        {
            static_cast<void>(cudaStreamSynchronize(stream));
        }
        runtime.reset();
        if (stream != nullptr)
        {
            static_cast<void>(cudaStreamDestroy(stream));
            stream = nullptr;
        }
        if (!reportDirectory.empty())
        {
            report["status"] = "failed";
            report["error"] = error.what();
            report["samples"] = samples;
            try
            {
                if (std::filesystem::create_directories(reportDirectory))
                {
                    std::ofstream file(std::filesystem::path(reportDirectory) / "report.json");
                    file << report.dump(2) << '\n';
                    file.flush();
                }
            }
            catch (std::exception const& reportingError)
            {
                LOG_ERROR("Could not preserve failed benchmark report: %s", reportingError.what());
            }
        }
        LOG_ERROR("llm_bench: %s", error.what());
        return EXIT_FAILURE;
    }
}
