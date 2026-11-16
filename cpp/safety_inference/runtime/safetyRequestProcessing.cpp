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

#include "safetyRequestProcessing.h"

#include "model_inference/modelInferenceCatalog.h"
#include "processing/models/internvl3/internvl3RequestProcessing.h"
#include "safety_inference/embedding/models/internvl3/internvl3EmbeddingCapability.h"
#include "safety_inference/image_preprocess/models/internvl3/internvl3MultimodalPreprocessorCapability.h"
#include "safety_inference/sampler/greedySamplerCapability.h"
#include "safety_inference/tokenizer/models/internvl3/internvl3TokenizerCapability.h"

namespace trt_edgellm::safety
{
namespace
{
std::unique_ptr<TokenizerCapability> createTokenizer(std::string_view id)
{
    return id == "internvl3_tokenizer" ? std::make_unique<InternVL3TokenizerCapability>() : nullptr;
}

std::unique_ptr<MultimodalPreprocessorCapability> createPreprocessor(std::string_view id)
{
    return id == "internvl3_multimodal_preprocessor" ? std::make_unique<InternVL3MultimodalPreprocessorCapability>()
                                                     : nullptr;
}

std::unique_ptr<EmbeddingCapability> createEmbedding(std::string_view id)
{
    return id == "fp16_vlm_embedding" ? std::make_unique<InternVL3EmbeddingCapability>() : nullptr;
}

std::unique_ptr<SamplerCapability> createSampler(std::string_view id)
{
    return id == "greedy_sampler" ? std::make_unique<GreedySamplerCapability>() : nullptr;
}
} // namespace

std::unique_ptr<SafetyRequestProcessing> createSafetyRequestProcessing(ModelInferenceDefinition const& definition)
{
    if (definition.family != SafetyModelFamily::kInternVL3_1B_Vlm)
    {
        return nullptr;
    }
    auto tokenizer = createTokenizer(definition.tokenizer);
    auto preprocessor = createPreprocessor(definition.multimodalPreprocessor);
    auto embedding = createEmbedding(definition.embedding);
    auto sampler = createSampler(definition.sampler);
    if (tokenizer == nullptr || preprocessor == nullptr || embedding == nullptr || sampler == nullptr)
    {
        return nullptr;
    }
    return std::make_unique<InternVL3RequestProcessing>(
        std::move(tokenizer), std::move(preprocessor), std::move(embedding), std::move(sampler));
}

} // namespace trt_edgellm::safety
