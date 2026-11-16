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
#include "profileFormatter.h"
#include <array>
#include <tensorrt_edgellm/common/logger.h>
#include <tensorrt_edgellm/common/trtUtils.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace trt_edgellm
{
namespace exampleUtils
{
namespace
{
constexpr uint64_t kFnvOffsetBasis{14695981039346656037ULL};
constexpr uint64_t kFnvPrime{1099511628211ULL};

void hashByte(uint64_t& hash, uint8_t byte)
{
    hash = (hash ^ byte) * kFnvPrime;
}

std::string hashString(uint64_t hash)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
}

nlohmann::json statistics(std::vector<float> const& values)
{
    auto const stats = StatisticalAnalysis::calculate(values);
    return {{"count", stats.count}, {"mean", stats.mean}, {"stddev", stats.stddev}, {"min", stats.min},
        {"max", stats.max}, {"median", stats.median}, {"p95", stats.p95}, {"p99", stats.p99}};
}

std::string csvString(std::string const& value)
{
    std::string result{"\""};
    for (char c : value)
    {
        if (c == '"')
        {
            result += '"';
        }
        result += c;
    }
    return result + '"';
}
} // namespace

std::string fingerprintBytes(std::string_view bytes)
{
    uint64_t hash = kFnvOffsetBasis;
    for (unsigned char byte : bytes)
    {
        hashByte(hash, byte);
    }
    return hashString(hash);
}

std::string fingerprintTokenIds(std::vector<int32_t> const& tokens)
{
    uint64_t hash = kFnvOffsetBasis;
    for (int32_t token : tokens)
    {
        uint32_t const word = static_cast<uint32_t>(token);
        for (int32_t shift = 0; shift < 32; shift += 8)
        {
            hashByte(hash, static_cast<uint8_t>(word >> shift));
        }
    }
    return hashString(hash);
}

std::string fingerprintFile(std::filesystem::path const& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Cannot fingerprint input file: " + path.string());
    }
    uint64_t hash = kFnvOffsetBasis;
    std::array<char, 65536> buffer;
    while (file)
    {
        file.read(buffer.data(), buffer.size());
        for (std::streamsize i = 0; i < file.gcount(); ++i)
        {
            hashByte(hash, static_cast<uint8_t>(buffer[i]));
        }
    }
    if (!file.eof())
    {
        throw std::runtime_error("Cannot read input file fingerprint: " + path.string());
    }
    return hashString(hash);
}

nlohmann::json makeBenchSample(int32_t iteration, int32_t workload, double wallTimeMs,
    rt::LLMGenerationResponse const& response, std::vector<timer::TimingSample> const& timings,
    std::vector<timer::AcceptanceSample> const& acceptance)
{
    int64_t outputTokens = 0;
    std::vector<size_t> outputLengths;
    std::vector<std::string> tokenFingerprints;
    std::vector<std::string> textFingerprints;
    for (auto const& ids : response.outputIds)
    {
        outputTokens += ids.size();
        outputLengths.push_back(ids.size());
        tokenFingerprints.push_back(fingerprintTokenIds(ids));
    }
    for (auto const& text : response.outputTexts)
    {
        textFingerprints.push_back(fingerprintBytes(text));
    }
    nlohmann::json invocations = nlohmann::json::array();
    bool nonSteady = false;
    for (auto const& sample : timings)
    {
        nonSteady = nonSteady || sample.stageId == "graph_capture";
        invocations.push_back(timingSampleToJson(sample));
    }
    nlohmann::json acceptanceJson = nlohmann::json::array();
    for (auto const& sample : acceptance)
    {
        acceptanceJson.push_back(acceptanceSampleToJson(sample));
    }
    std::vector<int32_t> finishReasons;
    for (auto reason : response.finishReasons)
    {
        finishReasons.push_back(static_cast<int32_t>(reason));
    }
    return {{"iteration", iteration}, {"workload", workload}, {"wall_ms", wallTimeMs},
        {"input_token_counts", response.inputTokenCounts}, {"output_token_counts", outputLengths},
        {"output_token_fingerprints", tokenFingerprints}, {"output_text_fingerprints", textFingerprints},
        {"output_tokens", outputTokens}, {"throughput_tps", wallTimeMs > 0 ? outputTokens * 1000.0 / wallTimeMs : 0},
        {"finish_reasons", finishReasons}, {"spec_verify_counts", response.specVerifyCounts},
        {"decode_round_counts", response.decodeRoundCounts}, {"non_steady", nonSteady}, {"invocations", invocations},
        {"acceptance", acceptanceJson}};
}

nlohmann::json summarizeBenchSamples(nlohmann::json const& samples)
{
    std::vector<float> wallTimes;
    std::vector<float> nonSteadyWallTimes;
    std::map<std::string, std::vector<float>> invocationTimes;
    std::map<std::string, std::vector<float>> shapeTimes;
    std::map<std::string, std::vector<float>> workloadTimes;
    for (auto const& sample : samples)
    {
        if (sample.value("non_steady", false))
        {
            nonSteadyWallTimes.push_back(sample.at("wall_ms").get<float>());
            continue;
        }
        wallTimes.push_back(sample.at("wall_ms").get<float>());
        workloadTimes[sample.at("workload").dump()].push_back(sample.at("wall_ms").get<float>());
        for (auto const& invocation : sample.at("invocations"))
        {
            float const milliseconds = invocation.at("gpu_ms").get<float>();
            invocationTimes[invocation.at("stage").get<std::string>()].push_back(milliseconds);
            auto shape = invocation;
            shape.erase("request_id");
            shape.erase("gpu_ms");
            shape["workload"] = sample.at("workload");
            shapeTimes[shape.dump()].push_back(milliseconds);
        }
    }
    nlohmann::json stages = nlohmann::json::object();
    nlohmann::json workloads = nlohmann::json::object();
    nlohmann::json shapes = nlohmann::json::array();
    for (auto const& [name, times] : invocationTimes)
    {
        stages[name] = statistics(times);
    }
    for (auto const& [name, times] : workloadTimes)
    {
        workloads[name] = statistics(times);
    }
    for (auto const& [identity, times] : shapeTimes)
    {
        shapes.push_back({{"identity", nlohmann::json::parse(identity)}, {"gpu_ms", statistics(times)}});
    }
    return {{"request_wall_ms", statistics(wallTimes)}, {"non_steady_request_wall_ms", statistics(nonSteadyWallTimes)},
        {"workload_request_wall_ms", workloads}, {"invocation_gpu_ms", stages}, {"invocation_shapes", shapes}};
}

void writeBenchReport(std::filesystem::path const& directory, nlohmann::json const& report)
{
    if (!std::filesystem::create_directories(directory))
    {
        throw std::runtime_error("Output directory already exists: " + directory.string());
    }
    try
    {
        std::ofstream jsonFile(directory / "report.json");
        std::ofstream requests(directory / "requests.csv");
        std::ofstream invocations(directory / "invocations.csv");
        jsonFile << report.dump(2) << '\n';
        requests << "iteration,workload,wall_ms,output_tokens,throughput_tps\n";
        invocations << "iteration,workload,request_id,round,stage,parent_stage,forward,forward_index,batch_size,"
                       "query_tokens,selected_tokens,past_lengths,gpu_ms\n";
        requests << std::setprecision(10);
        invocations << std::setprecision(10);
        for (auto const& sample : report.at("samples"))
        {
            requests << sample.at("iteration") << ',' << sample.at("workload") << ',' << sample.at("wall_ms") << ','
                     << sample.at("output_tokens") << ',' << sample.at("throughput_tps") << '\n';
            for (auto const& call : sample.at("invocations"))
            {
                invocations << sample.at("iteration") << ',' << sample.at("workload") << ',' << call.at("request_id")
                            << ',' << call.at("round") << ',' << csvString(call.at("stage")) << ','
                            << csvString(call.at("parent_stage")) << ',' << call.at("forward") << ','
                            << call.at("forward_index") << ',' << call.at("batch_size") << ','
                            << call.at("query_tokens") << ',' << call.at("selected_tokens") << ','
                            << csvString(call.at("past_lengths").dump()) << ',' << call.at("gpu_ms") << '\n';
            }
        }
        jsonFile.flush();
        requests.flush();
        invocations.flush();
        if (!jsonFile || !requests || !invocations)
        {
            throw std::runtime_error("Failed to write benchmark reports");
        }
    }
    catch (std::exception const& error)
    {
        auto failedReport = report;
        failedReport["status"] = "failed";
        failedReport["error"] = error.what();
        std::ofstream failureFile(directory / "report.json");
        failureFile << failedReport.dump(2) << '\n';
        failureFile.flush();
        throw;
    }
}
} // namespace exampleUtils
} // namespace trt_edgellm
