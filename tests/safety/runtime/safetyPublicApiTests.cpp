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

#include "safetyRuntimeTestDoubles.h"
#include "safety_inference/runtime/safetyRequestProcessing.h"
#include "safety_inference/runtime/safetyRuntime.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>

namespace trt_edgellm::safety::runtime_test
{
namespace
{
class ApiProcessing final : public SafetyRequestProcessing,
                            public TokenizerCapability,
                            public MultimodalPreprocessorCapability,
                            public EmbeddingCapability,
                            public SamplerCapability
{
public:
    ~ApiProcessing() override
    {
        gBackend.operations.push_back({"processingDestroyed"});
    }
    SafetyStatus configure(
        SafetyRuntimeArtifactBundle const&, nlohmann::json const&, nlohmann::json const&) noexcept override
    {
        mConfiguration.tokenizer.padTokenId = 0;
        mConfiguration.limits.vocabularySize = 16;
        return SafetyStatus::ok();
    }
    ProcessingConfiguration const& configuration() const noexcept override
    {
        return mConfiguration;
    }
    ProcessingCapabilitySet capabilities() noexcept override
    {
        return {*this, *this, *this, *this};
    }
    SafetyStatus declareResources(
        TokenizerArtifactSpec const&, ProcessingLimits const&, ResourceDeclaration&) noexcept override
    {
        return SafetyStatus::ok();
    }
    SafetyStatus declareResources(ImageProcessingLimits const&, ResourceDeclaration&) noexcept override
    {
        return SafetyStatus::ok();
    }
    SafetyStatus declareResources(
        EmbeddingArtifactSpec const&, ProcessingLimits const&, ResourceDeclaration&) noexcept override
    {
        return SafetyStatus::ok();
    }
    SafetyStatus declareResources(int32_t, ResourceDeclaration&) noexcept override
    {
        return SafetyStatus::ok();
    }
    SafetyStatus initialize(TokenizerArtifactSpec const&, PreparedResources const&) noexcept override
    {
        return SafetyStatus::ok();
    }
    SafetyStatus initialize(PreparedResources const&, LMStageContext const&) noexcept override
    {
        return SafetyStatus::ok();
    }
    SafetyStatus initialize(EmbeddingArtifactSpec const&, PreparedResources const&, LMStageContext const&,
        LMStageContext const&, LMStageContext const&) noexcept override
    {
        return SafetyStatus::ok();
    }
    SafetyStatus initialize(
        PreparedResources const&, llm::ModelLocalData&, LMStageContext const&, LMStageContext const&) noexcept override
    {
        return SafetyStatus::ok();
    }
    SafetyStatus prepareRequest(LLMGenerationRequest const& request, ProcessingRequest& prepared) noexcept override
    {
        // A fixed tokenizer/image-processing result; model arithmetic is outside this host contract.
        prepared = {
            {mPrompt.data(), mPrompt.size()}, {&mImage, sizeof(mImage)}, 1, 1, request.maxGenerateLength, {&mEos, 1}};
        return SafetyStatus::ok();
    }
    SafetyStatus validateResponse(
        LLMGenerationRequest const&, LLMGenerationResponse const& response) const noexcept override
    {
        return response.outputTexts.size() == 1 && response.outputTexts[0].capacity() >= 2
            ? SafetyStatus::ok()
            : SafetyStatus::error(SafetyStatusCode::kInvalidArgument);
    }
    SafetyStatus formatResponse(ModelInferenceResult const&, LLMGenerationResponse& response) noexcept override
    {
        response.outputTexts[0] = "ok";
        return SafetyStatus::ok();
    }
    TokenizationResult encodeMessages(LLMGenerationRequest::Request const&) noexcept override
    {
        return {SafetyStatus::error(SafetyStatusCode::kInvalidState), {}};
    }
    TokenizationResult encodePrompt(TextView) noexcept override
    {
        return {SafetyStatus::error(SafetyStatusCode::kInvalidState), {}};
    }
    ExpandedTokenResult expandImagePlaceholder(TokenIdView, std::size_t) noexcept override
    {
        return {SafetyStatus::error(SafetyStatusCode::kInvalidState), {}, 0};
    }
    SafetyStatus decodeResponse(TokenIdView, MutableTextBuffer, std::size_t&) noexcept override
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState);
    }
    TokenIdView stopTokens() const noexcept override
    {
        return {&mEos, 1};
    }
    ImageProcessingPlan planImage(vlm::ImageInputView) noexcept override
    {
        return {SafetyStatus::ok(), 1};
    }
    SafetyStatus prepareVisualInput(LMStageContext const& context, vlm::ImageInputView) noexcept override
    {
        context.profiles[0].resources->bindings[0].getFloat()[0] = 1;
        return SafetyStatus::ok();
    }
    SafetyStatus overlayVisualEmbeddings(
        LMStageContext const&, LMStageOutputs const&, std::size_t, std::size_t) noexcept override
    {
        return SafetyStatus::ok();
    }
    SafetyStatus embedPrompt(LMStageContext const& context, TokenIdView prompt) noexcept override
    {
        return fillLength(context, static_cast<int32_t>(prompt.size));
    }
    SafetyStatus embedDecodeToken(LMStageContext const& context, int32_t, int32_t length) noexcept override
    {
        return fillLength(context, length);
    }
    SafetyStatus select(LMStageOutputs const& output, int32_t& token) noexcept override
    {
        SafetyTensorView const* logits{nullptr};
        auto const status = output.context->getTensorView("logits", logits);
        if (status.okStatus())
        {
            token = static_cast<int32_t>(*static_cast<float*>(logits->data));
        }
        return status;
    }
    SafetyStatus initializeModelData(llm::ModelLocalData&) noexcept override
    {
        return SafetyStatus::ok();
    }
    SafetyStatus prepareWarmup(LMStageContext const& context, int64_t) noexcept override
    {
        return context.plan.id == "visual_encode" ? prepareVisualInput(context, {}) : fillLength(context, 1);
    }
    SafetyStatus deinitialize() noexcept override
    {
        return SafetyStatus::ok();
    }

private:
    SafetyStatus fillLength(LMStageContext const& context, int32_t length) noexcept
    {
        SafetyTensorView const* view{nullptr};
        auto const status = context.getTensorView("context_lengths", view);
        if (status.okStatus())
        {
            *static_cast<int32_t*>(view->data) = length;
        }
        return status;
    }
    std::array<int32_t, 2> mPrompt{3, 4};
    int32_t mEos{3};
    float mImage{1};
    ProcessingConfiguration mConfiguration{};
};

void describeLLM()
{
    using Type = nvinfer2::safe::DataType;
    gBackend.llmProfiles.resize(2);
    for (int32_t profile = 0; profile < 2; ++profile)
    {
        auto add = [&](char const* name, Type type, std::initializer_list<int64_t> shape, bool output = false) {
            nvinfer2::safe::TensorDescriptor descriptor{};
            descriptor.tensorName = name;
            descriptor.dataType = type;
            descriptor.ioMode = output ? nvinfer2::safe::TensorIOMode::kOUTPUT : nvinfer2::safe::TensorIOMode::kINPUT;
            descriptor.memPlacement = nvinfer2::safe::MemoryPlacement::kGPU;
            descriptor.bytesPerComponent = type == Type::kHALF ? 2 : type == Type::kINT64 ? 8 : 4;
            descriptor.shape.nbDims = descriptor.stride.nbDims = static_cast<int32_t>(shape.size());
            std::copy(shape.begin(), shape.end(), descriptor.shape.d);
            int64_t elements{1};
            for (int32_t dim = descriptor.shape.nbDims - 1; dim >= 0; --dim)
            {
                descriptor.stride.d[dim] = elements;
                elements *= descriptor.shape.d[dim];
            }
            descriptor.sizeInBytes = elements * descriptor.bytesPerComponent;
            gBackend.llmProfiles[profile].push_back(descriptor);
        };
        add("inputs_embeds", Type::kHALF, {1, profile == 0 ? 4 : 1, 4});
        add("rope_rotary_cos_sin", Type::kFLOAT, {1, 8, 2});
        add("logits", Type::kFLOAT, {1, 1, 16}, true);
        add("last_token_ids", Type::kINT64, {1, 1});
        add("context_lengths", Type::kINT32, {1});
        add("kvcache_start_index", Type::kINT32, {1});
        add("past_key_values_0", Type::kHALF, {1, 2, 1, 8, 2});
        add("present_key_values_0", Type::kHALF, {1, 2, 1, 8, 2}, true);
    }
}

class SafetyPublicApi : public testing::Test
{
protected:
    void SetUp() override
    {
        gBackend.reset();
        describeLLM();
        char pattern[]{"/tmp/edgellm-api-XXXXXX"};
        auto* directory = mkdtemp(pattern);
        if (directory == nullptr)
        {
            throw std::runtime_error("Cannot create artifact fixture");
        }
        mDirectory = directory;
        mArtifacts = {mDirectory / "llm", mDirectory / "visual"};
        std::filesystem::create_directory(mArtifacts.engineDir);
        std::filesystem::create_directory(mArtifacts.multimodalEngineDir);
        nlohmann::json llm{{"model", "qwen2"}, {"vocab_size", 16}, {"hidden_size", 4}, {"num_hidden_layers", 1},
            {"num_key_value_heads", 1}, {"num_attention_heads", 2},
            {"builder_config",
                {{"engine_capability", "safety"}, {"is_safety", true}, {"is_vlm", true},
                    {"profiles",
                        {{{"batch_size", 1}, {"input_seq_len", 4}, {"kv_cache_capacity", 8}},
                            {{"batch_size", 1}, {"input_seq_len", 1}, {"kv_cache_capacity", 8}}}}}}};
        nlohmann::json visual{{"model_type", "internvl"},
            {"builder_config",
                {{"engine_capability", "safety"}, {"is_safety", true}, {"profiles", {{{"image_tokens", 1}}}}}}};
        std::ofstream(mArtifacts.engineDir / "config.json") << llm;
        std::ofstream(mArtifacts.multimodalEngineDir / "config.json") << visual;
        for (auto const& path :
            {mArtifacts.engineDir / "llm_safety.engine", mArtifacts.multimodalEngineDir / "visual_safety.engine"})
        {
            std::ofstream(path) << "fake-engine";
            std::ofstream(path.string() + ".so") << "fake-companion";
        }
        mRuntime = std::make_unique<SafetyRuntime>();
        mRequest.requests.resize(1);
        mRequest.requests[0].messages = {{"user", {{"image", ""}, {"text", "Describe the image."}}}};
        mRequest.requests[0].imageBuffers = {{mPixels.data(), mPixels.size(), 1, 1}};
        mRequest.maxGenerateLength = 3;
        mResponse.outputIds.resize(1);
        mResponse.outputIds[0].reserve(8);
        mResponse.outputTexts.resize(1);
        mResponse.outputTexts[0].reserve(32);
    }
    void TearDown() override
    {
        // Fake SDK containment releases retained graphs while the runtime and recorders are still alive.
        gBackend.reset();
        mRuntime.reset();
        std::filesystem::remove_all(mDirectory);
    }
    std::unique_ptr<SafetyRuntime> mRuntime;
    std::filesystem::path mDirectory;
    SafetyRuntimeArtifactBundle mArtifacts;
    std::array<uint8_t, 3> mPixels{1, 2, 3};
    LLMGenerationRequest mRequest{};
    LLMGenerationResponse mResponse{};
};

// Given builder artifacts and prepared output storage, when using only the public lifecycle/request APIs,
// then initialization warms all stages, repeated requests reuse storage, and normal teardown releases graphs first.
TEST_F(SafetyPublicApi, InitializesRunsAndReleasesOwnedResources)
{
    auto const initialized = mRuntime->initialize(mArtifacts, {});
    ASSERT_TRUE(initialized.okStatus()) << initialized.message();
    EXPECT_EQ(gBackend.count("execute"), 3);
    auto const allocations = gBackend.allocationCalls;
    auto* output = mResponse.outputIds[0].data();
    for (int request = 0; request < 2; ++request)
    {
        ASSERT_TRUE(mRuntime->handleRequest(mRequest, mResponse)) << mRuntime->requestStatus().message();
        EXPECT_EQ(mResponse.outputIds[0], (std::vector<int32_t>{2, 2, 3}));
        EXPECT_EQ(mResponse.outputTexts[0], "ok");
        EXPECT_EQ(mResponse.outputIds[0].data(), output);
    }
    EXPECT_EQ(gBackend.allocationCalls, allocations);
    ASSERT_TRUE(mRuntime->deinitialize().okStatus());
    EXPECT_EQ(mRuntime->lifecycleState(), LifecycleState::kCLOSED);
    EXPECT_TRUE(gBackend.allocations.empty());
    auto const destroyed = std::find_if(gBackend.operations.begin(), gBackend.operations.end(),
        [](auto const& op) { return op.name == "processingDestroyed"; });
    EXPECT_EQ(
        std::count_if(gBackend.operations.begin(), destroyed, [](auto const& op) { return op.name == "destroy"; }), 2);
}

// Given an initialized public runtime, when input is invalid it is rejected before execution;
// when execution fails fatally, later requests and SDK cleanup stop while the runtime stays retained.
TEST_F(SafetyPublicApi, RejectsInvalidInputAndContainsFatalFailure)
{
    auto const initialized = mRuntime->initialize(mArtifacts, {});
    ASSERT_TRUE(initialized.okStatus()) << initialized.message();
    auto const beforeInvalid = gBackend.operations.size();
    mRequest.maxGenerateLength = 0;
    EXPECT_FALSE(mRuntime->handleRequest(mRequest, mResponse));
    EXPECT_EQ(mRuntime->requestStatus().code(), SafetyStatusCode::kInvalidArgument);
    EXPECT_EQ(gBackend.operations.size(), beforeInvalid);
    mRequest.maxGenerateLength = 3;
    std::array<LLMGenerationRequest, 3> invalidRequests{mRequest, mRequest, mRequest};
    invalidRequests[0].requests[0].imageBuffers[0].sizeInBytes = 2;
    invalidRequests[1].requests[0].messages[0].contents[0].type = "audio";
    invalidRequests[2].requests[0].imageBuffers.push_back(mRequest.requests[0].imageBuffers[0]);
    for (auto const& invalid : invalidRequests)
    {
        EXPECT_FALSE(mRuntime->handleRequest(invalid, mResponse));
        EXPECT_EQ(mRuntime->requestStatus().code(), SafetyStatusCode::kInvalidArgument);
        EXPECT_EQ(gBackend.operations.size(), beforeInvalid);
    }
    gBackend.executionResult = ErrorCode::kFAILED_EXECUTION;
    EXPECT_FALSE(mRuntime->handleRequest(mRequest, mResponse));
    EXPECT_TRUE(mRuntime->requestStatus().fatalStatus());
    EXPECT_TRUE(mResponse.outputIds[0].empty());
    auto const afterFatal = gBackend.operations.size();
    EXPECT_FALSE(mRuntime->handleRequest(mRequest, mResponse));
    EXPECT_FALSE(mRuntime->deinitialize().okStatus());
    EXPECT_EQ(gBackend.operations.size(), afterFatal);
    EXPECT_EQ(mRuntime->lifecycleState(), LifecycleState::kFAILED);
}
} // namespace
} // namespace trt_edgellm::safety::runtime_test

namespace trt_edgellm::safety
{
// The host test links this factory instead of the production capability implementation.
std::unique_ptr<SafetyRequestProcessing> createSafetyRequestProcessing(ModelInferenceDefinition const& definition)
{
    if (definition.id != ModelInferenceId::kInternVL3_1B_Vlm_Vanilla || definition.tokenizer != "internvl3_tokenizer"
        || definition.multimodalPreprocessor != "internvl3_multimodal_preprocessor"
        || definition.embedding != "fp16_vlm_embedding" || definition.sampler != "greedy_sampler")
    {
        return nullptr;
    }
    return std::make_unique<runtime_test::ApiProcessing>();
}
} // namespace trt_edgellm::safety
