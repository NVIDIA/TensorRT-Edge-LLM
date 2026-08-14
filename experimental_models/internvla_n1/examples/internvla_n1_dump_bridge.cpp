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

//! Dump the System-2 bridge tensor for one prompt, so the same prompt can be compared across
//! precisions. Exists because the engine's KV cache is paged: driving it from Python means
//! reimplementing the page table, while the runtime already does that correctly.

#include "runtime/llmInferenceRuntime.h"

#include "common/tensor.h"
#include "common/trtUtils.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{
//! Any index other than 0 works. Zero is where the runtime parks the post-multimodal input
//! embeddings, and the engine hidden states would silently overwrite them.
constexpr int32_t kBridgeLayer = 1;

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
    std::string const engineDir = argOf(argc, argv, "--engineDir");
    std::string const outPath = argOf(argc, argv, "--output", "bridge.bin");
    std::string const prompt = argOf(argc, argv, "--prompt",
        "You are an autonomous navigation assistant. Your task is to go to the kitchen. "
        "Where should you go next to stay on track?");
    if (engineDir.empty())
    {
        std::fprintf(stderr, "usage: %s --engineDir DIR [--output bridge.bin] [--prompt TEXT]\n", argv[0]);
        return 2;
    }

    // The engine embeds the edgellm plugins; without loading the library first, deserialization
    // fails with a plugin-creator-not-found assertion rather than anything that names the cause.
    auto const pluginHandles = loadEdgellmPluginLib();

    cudaStream_t stream{};
    cudaStreamCreate(&stream);
    std::unordered_map<std::string, std::string> const noLora;
    rt::LLMInferenceRuntime runtime(engineDir, "", noLora, stream);

    rt::LLMGenerationRequest request;
    request.requests.resize(1);
    rt::Message message;
    message.role = "user";
    message.contents.push_back({"text", prompt});
    request.requests[0].messages.push_back(message);
    // Distinct from the input-embeds slot; see kBridgeLayer.
    request.acceptHiddenLayer = kBridgeLayer;

    rt::LLMGenerationResponse response;
    // The last argument is what registers the hidden states at all. Left at its default the
    // registry stays empty and the lookup below returns nullptr.
    bool const ok = runtime.handleRequest(request, response, stream, /*outputThinkerEmbeddings=*/true);
    if (!ok)
    {
        std::fprintf(stderr, "handleRequest failed\n");
        return 1;
    }

    rt::Tensor const* hidden = runtime.getBaseModelHiddenStates(kBridgeLayer);
    if (hidden == nullptr || hidden->isEmpty())
    {
        std::fprintf(stderr, "no hidden states registered for layer %d\n", kBridgeLayer);
        return 1;
    }
    int32_t const prefillLen = runtime.getBaseModelPrefillLength();
    auto const dims = hidden->getShape().getTRTDims();
    std::printf("hidden states [");
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        std::printf("%ld%s", static_cast<long>(dims.d[i]), i + 1 < dims.nbDims ? ", " : "");
    }
    std::printf("], prefill length %d\n", prefillLen);

    std::vector<float> host(static_cast<size_t>(hidden->getShape().volume()));
    cudaMemcpy(host.data(), hidden->rawPointer(), host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::ofstream sink(outPath, std::ios::binary);
    sink.write(reinterpret_cast<char const*>(host.data()), static_cast<std::streamsize>(host.size() * sizeof(float)));
    std::printf("wrote %s (%zu floats)\n", outPath.c_str(), host.size());
    cudaStreamDestroy(stream);
    return 0;
}
