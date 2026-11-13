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

//! The service's main: read the command line, init and warm up the runtime, serve until Ctrl-C.

#include "ipcServer.h"

#include "common/trtUtils.h"

#include <getopt.h>

#include <csignal>
#include <iostream>
#include <memory>
#include <string>

namespace
{

using namespace edgellm::ipc;

constexpr char const* kUsage
    = "usage: media_ipc_server --llm-engine-dir DIR [options]\n"
      "  --llm-engine-dir DIR         LLM engine: llm.engine, config.json and the tokenizer\n"
      "  --multimodal-engine-dir DIR  audio/ and visual/ engines; without it, text only\n"
      "Listens on /tmp/edgellm-ipc.sock. EDGELLM_PLUGIN_PATH overrides the libNvInfer_edgellm_plugin.so location.\n";

option const kOptions[] = {
    {"llm-engine-dir", required_argument, nullptr, 'e'},
    {"multimodal-engine-dir", required_argument, nullptr, 'm'},
    {"help", no_argument, nullptr, 'h'},
    {nullptr, 0, nullptr, 0},
};

Server* gServer = nullptr;

void onSignal(int)
{
    if (gServer != nullptr)
    {
        gServer->requestStop();
    }
}

} // namespace

int main(int argc, char** argv)
{
    std::string llmEngineDir;
    std::string multimodalEngineDir;
    for (int option = 0; (option = getopt_long(argc, argv, "h", kOptions, nullptr)) != -1;)
    {
        switch (option)
        {
        case 'e': llmEngineDir = optarg; break;
        case 'm': multimodalEngineDir = optarg; break;
        case 'h': std::cout << kUsage; return 0;
        default: std::cerr << kUsage; return 1;
        }
    }
    if (llmEngineDir.empty() || optind != argc)
    {
        std::cerr << kUsage;
        return 1;
    }
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    // ----------------------step 1 init TensorRT Edge-LLM: CUDA stream, plugin, LLM / multimodal engines----------
    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    std::unique_ptr<CUstream_st, decltype(&cudaStreamDestroy)> const streamOwner(stream, cudaStreamDestroy);
    // The plugin library must outlive the engines, so it is declared before the runtime.
    auto const pluginLib = trt_edgellm::loadEdgellmPluginLib();
    if (!pluginLib)
    {
        std::cerr << "cannot load the TensorRT Edge-LLM plugin library; check EDGELLM_PLUGIN_PATH\n";
        return 1;
    }
    // Encoder embedding cache off: every frame runs the vision and audio encoders, as real camera frames would.
    rt::ContextCacheConfig contextCacheConfig;
    contextCacheConfig.encoderEmbeddingCacheBudgetBytes = 0;
    rt::LLMInferenceRuntime runtime(llmEngineDir, multimodalEngineDir, {}, stream, contextCacheConfig);
    runtime.captureDecodingCUDAGraph(stream);
    std::cout << "llm engine " << llmEngineDir << ", multimodal "
              << (multimodalEngineDir.empty() ? "none" : multimodalEngineDir) << '\n';

    // ----------------------step 2 warm up the runtime with one text-only handleRequest---------------------
    SamplingParams warmupParams;
    warmupParams.maxGenerateLength = 1;
    rt::LLMGenerationResponse warmupResponse;
    if (!runtime.handleRequest(
            buildLLMGenerationRequest(warmupParams, nullptr, nullptr, {}, "Hi."), warmupResponse, stream))
    {
        std::cerr << "the warmup request failed; the engine will not serve either\n";
        return 1;
    }

    // ----------------------step 3 listen for clients, open NvSci for client frames, start the worker----------
    Server server(runtime, stream);
    gServer = &server;
    std::cout << "listening on " << kSocketPath << ", Ctrl-C to stop" << std::endl;

    // ----------------------step 4 start the server and run until Ctrl-C---------------------
    server.run();

    gServer = nullptr;
    return 0;
}
