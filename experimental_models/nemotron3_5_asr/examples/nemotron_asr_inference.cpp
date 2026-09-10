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

/*
 * Nemotron-3.5-ASR offline transcription example.
 *
 * Runs the standalone RNN-T pipeline: audio file → mel → FastConformer
 * encoder engine → greedy transducer loop over the RNN-T step engine →
 * transcript.
 *
 * Engine directory layout:
 *   Direct builder: audio/audio_encoder.engine + rnnt/rnnt_step.engine
 *   ONNX builder:   audio_encoder.engine + rnnt_step.engine at the root
 *   Both layouts:   config.json and tokenizer.json at the root
 */

#include "common/checkMacros.h"
#include "common/logger.h"
#include "runtime/audioLoader.h"
#include "runtime/nemotronAsrRuntime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

struct Args
{
    bool help{false};
    std::string engineDir;
    std::string tokenizerDir;
    std::string audioFile;
    int32_t promptId{-1};  //!< -1 = config default (auto language detection)
    bool benchmark{false}; //!< Measure per-phase latency instead of a single run.
    int32_t iters{20};     //!< Timed iterations in benchmark mode.
    int32_t warmup{3};     //!< Warmup iterations (graph capture / engine init).
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --help                  Display this help message\n"
              << "  --engineDir=<path>      Direct component bundle or flat ONNX-built bundle.\n"
              << "  --tokenizerDir=<path>   Directory with tokenizer.json. Defaults to --engineDir.\n"
              << "  --audioFile=<path>      Audio file to transcribe (wav/mp3/flac). Required.\n"
              << "  --promptId=<int>        Language prompt index. Default: config default_prompt_id\n"
              << "                          (automatic language detection; the model emits an\n"
              << "                          <xx-XX> language tag).\n"
              << "  --benchmark             Measure per-phase latency (mel/encoder/decode) and\n"
              << "                          real-time factor over repeated runs on the same clip.\n"
              << "  --iters=<int>           Timed iterations in --benchmark mode (default 20).\n"
              << "  --warmup=<int>          Warmup iterations before timing (default 3).\n"
              << std::endl;
}

enum OptionId : int
{
    HELP = 900,
    ENGINE_DIR,
    TOKENIZER_DIR,
    AUDIO_FILE,
    PROMPT_ID,
    BENCHMARK,
    ITERS,
    WARMUP
};

int32_t parseInt32(char const* text, char const* name)
{
    size_t parsed = 0;
    long long const value = std::stoll(text, &parsed);
    if (text[parsed] != '\0' || value < std::numeric_limits<int32_t>::min()
        || value > std::numeric_limits<int32_t>::max())
    {
        throw std::invalid_argument(std::string(name) + " must be a signed 32-bit integer");
    }
    return static_cast<int32_t>(value);
}

bool parseArgs(Args& args, int argc, char* argv[])
{
    static struct option options[]
        = {{"help", no_argument, 0, OptionId::HELP}, {"engineDir", required_argument, 0, OptionId::ENGINE_DIR},
            {"tokenizerDir", required_argument, 0, OptionId::TOKENIZER_DIR},
            {"audioFile", required_argument, 0, OptionId::AUDIO_FILE},
            {"promptId", required_argument, 0, OptionId::PROMPT_ID}, {"benchmark", no_argument, 0, OptionId::BENCHMARK},
            {"iters", required_argument, 0, OptionId::ITERS}, {"warmup", required_argument, 0, OptionId::WARMUP},
            {0, 0, 0, 0}};

    try
    {
        int opt;
        while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
        {
            switch (opt)
            {
            case OptionId::HELP: args.help = true; return true;
            case OptionId::ENGINE_DIR: args.engineDir = optarg; break;
            case OptionId::TOKENIZER_DIR: args.tokenizerDir = optarg; break;
            case OptionId::AUDIO_FILE: args.audioFile = optarg; break;
            case OptionId::PROMPT_ID: args.promptId = parseInt32(optarg, "--promptId"); break;
            case OptionId::BENCHMARK: args.benchmark = true; break;
            case OptionId::ITERS: args.iters = parseInt32(optarg, "--iters"); break;
            case OptionId::WARMUP: args.warmup = parseInt32(optarg, "--warmup"); break;
            default: return false;
            }
        }
    }
    catch (std::exception const& error)
    {
        std::cerr << "Invalid numeric option: " << error.what() << std::endl;
        return false;
    }
    if (args.engineDir.empty() || args.audioFile.empty())
    {
        std::cerr << "--engineDir and --audioFile are required." << std::endl;
        return false;
    }
    if (args.tokenizerDir.empty())
    {
        args.tokenizerDir = args.engineDir;
    }
    if (args.promptId < -1 || args.iters < 1 || args.warmup < 0)
    {
        std::cerr << "--promptId must be >= -1, --iters must be >= 1, and --warmup must be >= 0." << std::endl;
        return false;
    }
    return true;
}

//! Summary statistics over a set of per-iteration latency samples (ms).
struct Stats
{
    double mean{0.0};
    double median{0.0};
    double min{0.0};
    double p90{0.0};
};

Stats computeStats(std::vector<double> v)
{
    Stats s{};
    if (v.empty())
    {
        return s;
    }
    std::sort(v.begin(), v.end());
    s.min = v.front();
    s.median = v[v.size() / 2];
    size_t p90Idx = static_cast<size_t>(std::ceil(0.9 * static_cast<double>(v.size())));
    p90Idx = p90Idx == 0 ? 0 : std::min(p90Idx - 1, v.size() - 1);
    s.p90 = v[p90Idx];
    double sum = 0.0;
    for (double x : v)
    {
        sum += x;
    }
    s.mean = sum / static_cast<double>(v.size());
    return s;
}

void printStatsRow(char const* label, Stats const& s)
{
    std::cout << "  " << std::left << std::setw(12) << label << std::right << std::fixed << std::setprecision(3)
              << std::setw(10) << s.mean << std::setw(10) << s.median << std::setw(10) << s.min << std::setw(10)
              << s.p90 << "\n";
}

//! Repeatedly transcribe one clip, reporting the mel/encoder/decode phase
//! breakdown and real-time factor. Phases mirror an LLM's prefill/decode
//! split: the encoder forward is prefill-like, the RNN-T step loop decode-like.
int runBenchmark(NemotronAsrRuntime& runtime, audio::AudioPCM const& pcm, int32_t promptId, double audioSeconds,
    int32_t iters, int32_t warmup, cudaStream_t stream)
{
    if (iters < 1)
    {
        std::cerr << "--iters must be >= 1" << std::endl;
        return EXIT_FAILURE;
    }
    for (int32_t i = 0; i < warmup; ++i)
    {
        runtime.transcribe(pcm, promptId, stream);
    }

    std::vector<double> mel, enc, dec, tot;
    mel.reserve(iters);
    enc.reserve(iters);
    dec.reserve(iters);
    tot.reserve(iters);
    NemotronAsrRuntime::Result last;
    for (int32_t i = 0; i < iters; ++i)
    {
        NemotronAsrRuntime::Timings tm;
        auto const t0 = std::chrono::steady_clock::now();
        last = runtime.transcribe(pcm, promptId, stream, &tm);
        auto const t1 = std::chrono::steady_clock::now();
        mel.push_back(tm.melMs);
        enc.push_back(tm.encoderMs);
        dec.push_back(tm.decodeMs);
        tot.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    Stats const totStats = computeStats(tot);
    Stats const decStats = computeStats(dec);
    double const perStepMs = last.numDecodeSteps > 0 ? decStats.mean / static_cast<double>(last.numDecodeSteps) : 0.0;
    double const rtf = audioSeconds > 0.0 ? (totStats.mean / 1000.0) / audioSeconds : 0.0;

    std::cout << "\nBenchmark: " << iters << " iters (" << warmup << " warmup), batch 1\n"
              << "Audio:      " << audioSeconds << " s -> " << last.numMelFrames << " mel frames -> "
              << last.numEncoderFrames << " encoder frames, " << last.numDecodeSteps << " decode steps, "
              << last.tokens.size() << " tokens\n\n"
              << "  " << std::left << std::setw(12) << "Phase (ms)" << std::right << std::setw(10) << "mean"
              << std::setw(10) << "median" << std::setw(10) << "min" << std::setw(10) << "p90" << "\n";
    printStatsRow("mel(cpu)", computeStats(mel));
    printStatsRow("encoder", computeStats(enc));
    printStatsRow("decode", decStats);
    printStatsRow("end2end", totStats);
    std::cout << std::fixed << std::setprecision(3) << "\nDecode per-step: " << perStepMs << " ms/step\n"
              << "RTF: " << rtf << "  (" << std::setprecision(1) << (rtf > 0.0 ? 1.0 / rtf : 0.0)
              << "x faster than real-time)" << std::endl;
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char* argv[])
{
    Args args;
    if (!parseArgs(args, argc, argv) || args.help)
    {
        printUsage(argv[0]);
        return args.help ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    try
    {
        NemotronAsrRuntime runtime(args.engineDir, args.tokenizerDir, stream);

        audio::AudioPCM pcm;
        check::check(audio::loadAudioFile(args.audioFile, /*targetSampleRate=*/16000, pcm),
            "Failed to load audio file: " + args.audioFile);
        double const audioSeconds = static_cast<double>(pcm.samples.size()) / pcm.sampleRate;

        int32_t const promptId = args.promptId >= 0 ? args.promptId : runtime.defaultPromptId();

        if (args.benchmark)
        {
            int const rc = runBenchmark(runtime, pcm, promptId, audioSeconds, args.iters, args.warmup, stream);
            cudaStreamDestroy(stream);
            return rc;
        }

        auto const result = runtime.transcribe(pcm, promptId, stream);

        std::cout << "Audio:      " << args.audioFile << " (" << audioSeconds << " s)" << std::endl;
        std::cout << "Transcript: " << result.text << std::endl;
        std::cout << "Stats:      " << result.numMelFrames << " mel frames -> " << result.numEncoderFrames
                  << " encoder frames, " << result.numDecodeSteps << " decode steps, " << result.tokens.size()
                  << " tokens" << std::endl;
    }
    catch (std::exception const& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    cudaStreamDestroy(stream);
    return EXIT_SUCCESS;
}
