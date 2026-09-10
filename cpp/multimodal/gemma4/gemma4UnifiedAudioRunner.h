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

#pragma once

#include "multimodal/common/multimodalRunner.h"
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Runtime configuration for the encoder-free Gemma4 Unified audio embedder.
struct Gemma4UnifiedAudioConfig
{
    int64_t minFrames{0};
    int64_t maxFrames{0};
    int64_t samplesPerFrame{640};
    int64_t outputHiddenSize{0};
    int32_t sampleRate{16000};
    int32_t audioTokenId{0};
    int32_t beginAudioTokenId{0};
    int32_t endAudioTokenId{0};
};

//! Frame raw mono PCM into 640-sample tokens and run the lightweight Unified audio projection.
class Gemma4UnifiedAudioRunner : public MultimodalRunner
{
public:
    Gemma4UnifiedAudioRunner(std::string const& engineDir, cudaStream_t stream);
    ~Gemma4UnifiedAudioRunner() noexcept = default;

    bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer, rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream,
        bool imageOnly = false, bool skipEncoderWork = false) override;
    bool infer(cudaStream_t stream) noexcept override;
    bool validateAndFillConfig(std::string const& engineDir) override;
    bool allocateBuffer(cudaStream_t stream) override;

private:
    void frameAudio(rt::LLMGenerationRequest const& request, std::vector<int64_t>& audioTokenLengths,
        std::vector<int64_t>& audiosPerRequest, cudaStream_t stream);
    void frameAudioTokenLengthsOnly(rt::LLMGenerationRequest const& request, std::vector<int64_t>& audioTokenLengths,
        std::vector<int64_t>& audiosPerRequest);
    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        std::vector<int64_t> const& audioTokenLengths, std::vector<int64_t> const& audiosPerRequest,
        tokenizer::Tokenizer const* tokenizer);

    Gemma4UnifiedAudioConfig mConfig{};
    rt::Tensor mInputFeatures{};
    rt::Tensor mInputFeaturesHost{};
};

} // namespace rt
} // namespace trt_edgellm
