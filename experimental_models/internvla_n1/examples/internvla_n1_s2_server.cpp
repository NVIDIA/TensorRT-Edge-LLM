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

//! Long-lived System-2 generation server, one JSON request per stdin line.
//!
//! Closed-loop navigation asks System 2 for a decision at every step, and a
//! benchmark run is tens of thousands of steps. Spawning llm_inference per step
//! would reload a 14 GB engine each time, so the engine is loaded once here and
//! the caller streams requests over a pipe.
//!
//! Protocol, one JSON object per line in each direction:
//!   in : {"messages":[{"role":"user","contents":[{"type":"image","content":"/a.png"},
//!                                                {"type":"text","content":"..."}]}],
//!         "max_new_tokens":64}
//!   out: {"text":"..."}   or   {"error":"..."}
//! A blank line or EOF on stdin shuts the server down.

#include "common/trtUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;
using json = nlohmann::json;

namespace
{
std::string argOf(int argc, char** argv, char const* flag, std::string const& fallback = "")
{
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (std::strcmp(argv[i], flag) == 0)
        {
            return argv[i + 1];
        }
    }
    return fallback;
}
} // namespace

int main(int argc, char** argv)
{
    std::string const llmDir = argOf(argc, argv, "--engineDir");
    std::string const visDir = argOf(argc, argv, "--multimodalEngineDir");
    if (llmDir.empty())
    {
        std::fprintf(stderr, "usage: %s --engineDir DIR [--multimodalEngineDir DIR]\n", argv[0]);
        return 2;
    }

    auto const pluginHandles = loadEdgellmPluginLib();
    cudaStream_t stream{};
    cudaStreamCreate(&stream);
    std::unordered_map<std::string, std::string> const noLora;
    rt::LLMInferenceRuntime runtime(llmDir, visDir, noLora, stream);

    // The caller blocks on this line before sending work.
    std::printf("{\"ready\":true}\n");
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
        {
            break;
        }
        json out;
        try
        {
            auto const in = json::parse(line);
            rt::LLMGenerationRequest request;
            request.requests.resize(1);
            // "raw_text" carries a prompt the caller has already templated. Re-running
            // the engine's own chat template over it would double the control tokens
            // and change the prompt the model sees, so the template is turned off and
            // the string is passed through untouched.
            if (in.contains("raw_text"))
            {
                rt::Message msg;
                msg.role = "user";
                msg.contents.push_back({"text", in.at("raw_text").get<std::string>()});
                request.requests[0].messages.push_back(std::move(msg));
                request.applyChatTemplate = false;
            }
            for (auto const& m : in.value("messages", json::array()))
            {
                rt::Message msg;
                msg.role = m.value("role", "user");
                for (auto const& c : m.at("contents"))
                {
                    msg.contents.push_back({c.at("type").get<std::string>(), c.at("content").get<std::string>()});
                }
                request.requests[0].messages.push_back(std::move(msg));
            }
            for (auto const& p : in.value("images", std::vector<std::string>{}))
            {
                auto image = rt::imageUtils::loadImageFromFile(p);
                if (image.buffer != nullptr)
                {
                    request.requests[0].imageBuffers.push_back(std::move(image));
                }
            }
            // None of these have default initialisers in the struct; leaving topK
            // uninitialised makes the sampler size its workspace from stack garbage.
            request.temperature = 1.0F;
            request.topP = 1.0F;
            request.topK = 1;
            request.maxGenerateLength = in.value("max_new_tokens", 64);

            rt::LLMGenerationResponse response;
            if (!runtime.handleRequest(request, response, stream, /*outputThinkerEmbeddings=*/false))
            {
                out["error"] = "handleRequest failed";
            }
            else
            {
                out["text"] = response.outputTexts.empty() ? std::string{} : response.outputTexts[0];
            }
        }
        catch (std::exception const& e)
        {
            out["error"] = e.what();
        }
        std::printf("%s\n", out.dump().c_str());
        std::fflush(stdout);
    }
    cudaStreamDestroy(stream);
    return 0;
}
