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
 * Audio model builder for Qwen3-Omni
 * Builds TensorRT engines for audio encoder (speech input) and Code2Wav vocoder (speech output)
 *
 * Build type is auto-detected from config.json:
 *   - audio_config present -> builds audio_encoder.engine
 *   - code2wav_config present -> builds code2wav.engine
 */

#include "builder/audioBuilder.h"
#include "common/logger.h"

#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>

using namespace trt_edgellm;

struct AudioBuildArgs
{
    std::string onnxDir;
    std::string engineDir;
    bool help{false};
    bool debug{false};

    // Optimization profile config (applies to both build types)
    // Audio encoder profile
    int64_t minTimeSteps{100};
    int64_t maxTimeSteps{6000};

    // Code2Wav profile
    int64_t minCodeLen{1};
    int64_t optCodeLen{300};
    int64_t maxCodeLen{2000};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " [--help] <--onnx-dir str> <--engine-dir str> [--debug]" << std::endl;
    std::cerr << "       Audio encoder profile: [--min-time-steps int] [--max-time-steps int]" << std::endl;
    std::cerr << "       Code2Wav profile: [--min-code-len int] [--opt-code-len int] [--max-code-len int]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "General Options:" << std::endl;
    std::cerr << "  --help            Display this help message" << std::endl;
    std::cerr << "  --onnx-dir        Directory containing ONNX model (model.onnx) and config.json. Required."
              << std::endl;
    std::cerr << "  --engine-dir      Output directory for the engine. Required." << std::endl;
    std::cerr << "  --debug           Use debug mode with verbose output" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Build Type Auto-Detection (from config.json):" << std::endl;
    std::cerr << "  - If 'audio_config'/'sound_config'/'encoder_config' exists: builds audio_encoder.engine"
              << std::endl;
    std::cerr << "  - If 'code2wav_config' exists: builds code2wav.engine" << std::endl;
    std::cerr << "  - If 'rnnt_decoder_config' exists: builds rnnt_step.engine (Nemotron-3.5-ASR)" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Audio Encoder Profile Options (when audio_config detected):" << std::endl;
    std::cerr << "  --min-time-steps  Minimum audio time steps. Default = 100 (~0.64s audio)" << std::endl;
    std::cerr << "  --max-time-steps  Maximum audio time steps. Default = 6000 (~38.4s audio)" << std::endl;
    std::cerr << "  Time steps formula: duration_seconds = (time_steps * 160) / 16000" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Code2Wav Profile Options (when code2wav_config detected):" << std::endl;
    std::cerr << "  --min-code-len    Minimum code sequence length (frames). Default = 1" << std::endl;
    std::cerr << "  --opt-code-len    Optimal code sequence length (frames). Default = 300" << std::endl;
    std::cerr << "  --max-code-len    Maximum code sequence length (frames). Default = 2000" << std::endl;
    std::cerr << "  Duration formula: duration_seconds = code_len * 0.08 (at 24kHz, 1920 upsample)" << std::endl;
    std::cerr << "  Example: 300 frames = 24s, 1000 frames = 80s, 2000 frames = 160s" << std::endl;
    std::cerr << "Legacy camelCase option spellings remain accepted.\n";
}

bool parseAudioBuildArgs(AudioBuildArgs& args, int argc, char* argv[])
{
    // Option IDs for long options without short equivalents
    enum OptionId
    {
        OPT_MIN_CODE_LEN,
        OPT_OPT_CODE_LEN,
        OPT_MAX_CODE_LEN
    };

    static struct option longOptions[]
        = {{"help", no_argument, nullptr, 'h'}, {"onnx-dir", required_argument, nullptr, 'o'},
            {"onnxDir", required_argument, nullptr, 'o'}, {"engine-dir", required_argument, nullptr, 'e'},
            {"engineDir", required_argument, nullptr, 'e'}, {"debug", no_argument, nullptr, 'd'},
            // Audio encoder profile options
            {"min-time-steps", required_argument, nullptr, 'm'}, {"minTimeSteps", required_argument, nullptr, 'm'},
            {"max-time-steps", required_argument, nullptr, 'M'}, {"maxTimeSteps", required_argument, nullptr, 'M'},
            // Code2Wav profile options
            {"min-code-len", required_argument, nullptr, OPT_MIN_CODE_LEN},
            {"minCodeLen", required_argument, nullptr, OPT_MIN_CODE_LEN},
            {"opt-code-len", required_argument, nullptr, OPT_OPT_CODE_LEN},
            {"optCodeLen", required_argument, nullptr, OPT_OPT_CODE_LEN},
            {"max-code-len", required_argument, nullptr, OPT_MAX_CODE_LEN},
            {"maxCodeLen", required_argument, nullptr, OPT_MAX_CODE_LEN}, {nullptr, 0, nullptr, 0}};

    int optionIndex = 0;
    int opt;
    while ((opt = getopt_long(argc, argv, "ho:e:dm:M:", longOptions, &optionIndex)) != -1)
    {
        switch (opt)
        {
        case 'h': args.help = true; return true;
        case 'o': args.onnxDir = optarg; break;
        case 'e': args.engineDir = optarg; break;
        case 'd': args.debug = true; break;
        // Audio encoder profile options
        case 'm': args.minTimeSteps = std::stoll(optarg); break;
        case 'M': args.maxTimeSteps = std::stoll(optarg); break;
        // Code2Wav profile options
        case OPT_MIN_CODE_LEN: args.minCodeLen = std::stoll(optarg); break;
        case OPT_OPT_CODE_LEN: args.optCodeLen = std::stoll(optarg); break;
        case OPT_MAX_CODE_LEN: args.maxCodeLen = std::stoll(optarg); break;
        default:
            std::cerr << "Error: Invalid argument" << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }

    if (args.onnxDir.empty() || args.engineDir.empty())
    {
        std::cerr << "Error: --onnx-dir and --engine-dir are required" << std::endl;
        printUsage(argv[0]);
        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    AudioBuildArgs args;
    if ((argc < 2) || (!parseAudioBuildArgs(args, argc, argv)))
    {
        LOG_ERROR("Unable to parse builder args.");
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    // Validate input directory - config.json is required for auto-detection
    std::string configPath = args.onnxDir + "/config.json";
    std::ifstream configFile(configPath);
    if (!configFile.good())
    {
        LOG_ERROR("config.json not found in onnx directory: %s", args.onnxDir.c_str());
        LOG_ERROR("config.json is required for auto-detecting build type (audio_config or code2wav_config)");
        return EXIT_FAILURE;
    }
    configFile.close();

    // Create AudioBuilderConfig with all profile parameters
    // Build type will be auto-detected from config.json by AudioBuilder
    builder::AudioBuilderConfig config;
    config.minTimeSteps = args.minTimeSteps;
    config.maxTimeSteps = args.maxTimeSteps;
    config.minCodeLen = args.minCodeLen;
    config.optCodeLen = args.optCodeLen;
    config.maxCodeLen = args.maxCodeLen;

    LOG_INFO("Building audio model from: %s", args.onnxDir.c_str());
    LOG_INFO("Output directory: %s", args.engineDir.c_str());
    LOG_INFO("Build type will be auto-detected from config.json");

    // Create and run the builder
    // AudioBuilder auto-detects whether to build audio_encoder or code2wav from config.json
    builder::AudioBuilder audioBuilder(args.onnxDir, args.engineDir, config);
    if (!audioBuilder.build())
    {
        LOG_ERROR("Failed to build audio engine.");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
