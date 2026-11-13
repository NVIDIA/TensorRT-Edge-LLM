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

#include "safetyRuntime.h"
#include "safetyRequestProcessing.h"

#include <algorithm>
#include <fstream>
#include <limits>

#include <new>
#include <stdexcept>

namespace trt_edgellm
{
namespace safety
{
namespace
{
nlohmann::json readConfig(std::filesystem::path const& directory)
{
    std::ifstream input(directory / "config.json");
    if (!input)
    {
        throw std::invalid_argument("Missing Safety engine config.json");
    }
    return nlohmann::json::parse(input);
}

int32_t positive(nlohmann::json const& config, char const* key)
{
    auto const& value = config.at(key);
    if (!value.is_number_integer())
    {
        throw std::invalid_argument("Safety model dimensions must be integers");
    }
    auto const size = value.get<int64_t>();
    if (size <= 0 || size > std::numeric_limits<int32_t>::max())
    {
        throw std::invalid_argument("Safety model dimension is out of range");
    }
    return static_cast<int32_t>(size);
}

void readEngine(std::filesystem::path const& path, std::vector<char>& bytes)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    auto const size = input.tellg();
    if (!input || size <= 0 || static_cast<uint64_t>(size) > bytes.max_size())
    {
        throw std::invalid_argument("Missing or empty Safety engine");
    }
    bytes.resize(static_cast<std::size_t>(size));
    input.seekg(0);
    if (!input.read(bytes.data(), size))
    {
        throw std::invalid_argument("Cannot read complete Safety engine");
    }
}

void validateBuilder(nlohmann::json const& config, std::size_t profiles)
{
    auto const& builder = config.at("builder_config");
    if (builder.at("engine_capability") != "safety" || builder.at("is_safety") != true
        || !builder.at("profiles").is_array() || builder.at("profiles").size() != profiles)
    {
        throw std::invalid_argument("Engine configuration does not match the Safety inference recipe");
    }
}
constexpr ResourceOwnerId kOUTPUT_OWNER{2};
constexpr RuntimeResourceKey kOUTPUT_TOKENS{0};

SafetyStatus describeStage(LMStageContext const& context, nlohmann::json const& profile, llm::StageResourceSpec& spec)
{
    spec.stage = context.plan.id;
    spec.profileId = static_cast<int32_t>(context.plan.profiles.front());
    for (auto const& tensor : context.profiles.front().tensors)
    {
        RuntimeTensorDesc descriptor{};
        auto const status = describeRuntimeTensor(tensor, spec.profileId, descriptor);
        if (!status.okStatus())
        {
            return status;
        }
        auto const& shape = descriptor.profiles.front().shape;
        if ((descriptor.name == "inputs_embeds"
                && (shape.nbDims != 3 || shape.d[1] != profile.at("input_seq_len").get<int64_t>()))
            || (descriptor.name.find("past_key_values_") == 0
                && (shape.nbDims != 5 || shape.d[3] != profile.at("kv_cache_capacity").get<int64_t>())))
        {
            return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Engine IO disagrees with builder profiles");
        }
        spec.bindings.push_back(std::move(descriptor));
    }
    return SafetyStatus::ok();
}
} // namespace

SafetyRuntime::SafetyRuntime() noexcept
    : mErrorState()
    , mDeclaredResources(mLifecycleState)
    , mResources(mLifecycleState, mErrorState)
    , mRunner(mLifecycleState, mErrorState, mResources)
{
}

SafetyRuntime::SafetyRuntime(nvinfer2::safe::ISafeRecorder& recorder) noexcept
    : mErrorState(recorder)
    , mDeclaredResources(mLifecycleState)
    , mResources(mLifecycleState, mErrorState)
    , mRunner(mLifecycleState, mErrorState, mResources)
{
}

SafetyRuntime::~SafetyRuntime() noexcept = default;

SafetyStatus SafetyRuntime::initialize(
    SafetyRuntimeArtifactBundle const& artifacts, SafetyRuntimeOptions const& options) noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mLifecycleState != LifecycleState::kINITIALIZATION || !mArtifactPaths.engineDir.empty() || !contexts().empty()
        || mProcessing != nullptr)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "Runtime initialization has already started");
    }
    try
    {
        auto status = loadArtifacts(artifacts, options);
        if (status.okStatus())
        {
            status = prepareModel();
        }
        return recordResult(status);
    }
    catch (std::bad_alloc const&)
    {
        return recordResult(
            SafetyStatus::error(SafetyStatusCode::kFailedAllocation, "Runtime setup allocation failed"));
    }
    catch (std::exception const& error)
    {
        return recordResult(SafetyStatus::error(SafetyStatusCode::kInvalidArgument, error.what()));
    }
    catch (...)
    {
        return recordResult(
            SafetyStatus::fatal(SafetyStatusCode::kFailedExecution, "Unexpected runtime setup failure"));
    }
}

bool SafetyRuntime::handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response) noexcept
{
    if (!observeErrors())
    {
        mRequestStatus = mErrorState.status();
    }
    else if (mLifecycleState != LifecycleState::kOPERATIONAL || mInference == nullptr || mProcessing == nullptr
        || mCapabilityCoordinator == nullptr)
    {
        mRequestStatus = SafetyStatus::error(SafetyStatusCode::kInvalidState, "Request requires initialized runtime");
    }
    else
    {
        mRequestStatus = recordResult(runRequest(request, response));
    }
    return mRequestStatus.okStatus();
}

SafetyStatus SafetyRuntime::requestStatus() const noexcept
{
    return mRequestStatus;
}

SafetyStatus SafetyRuntime::initialize(SafetyRuntimeInitSpec const& spec) noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mLifecycleState != LifecycleState::kINITIALIZATION)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Runtime initialization is closed");
    }
    return recordResult(mRunner.initialize(spec));
}

std::vector<LMStageContext> const& SafetyRuntime::contexts() const noexcept
{
    return mRunner.contexts();
}

SafetyStatus SafetyRuntime::completeContext(
    char const* stage, ResourceSlotId slot, std::vector<SafetyModelLocalResourceView> const& views) noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mLifecycleState != LifecycleState::kINITIALIZATION)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Context preparation requires Initialization");
    }
    return recordResult(mRunner.completeContext(stage, slot, views));
}

SafetyStatus SafetyRuntime::prepare(LMStageWarmupCallback warmup, void* userData) noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mLifecycleState != LifecycleState::kINITIALIZATION)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Warmup requires Initialization");
    }
    auto const resources = recordResult(mDeclaredResources.validatePrepared());
    if (!resources.okStatus())
    {
        return resources;
    }
    auto const status = recordResult(mRunner.prepare(warmup, userData));
    if (status.okStatus())
    {
        mLifecycleState = LifecycleState::kOPERATIONAL;
    }
    return status;
}

SafetyStatus SafetyRuntime::runLMStage(LMStageId stage, ResourceSlotId slot, LMStageOutputs& outputs,
    LMStagePrepareCallback prepare, void* userData) noexcept
{
    outputs.context = nullptr;
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mLifecycleState != LifecycleState::kOPERATIONAL)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Stage execution requires Operational");
    }
    return recordResult(mRunner.runLMStage(stage, slot, outputs, prepare, userData));
}

SafetyStatus SafetyRuntime::deinitializeGraphs() noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mLifecycleState == LifecycleState::kCLOSED)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Runtime execution is already closed");
    }
    auto const status = recordResult(mRunner.deinitializeGraphs());
    if (status.okStatus())
    {
        mLifecycleState = LifecycleState::kCLOSED;
    }
    return status;
}

SafetyStatus SafetyRuntime::deinitializeResources() noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mLifecycleState != LifecycleState::kCLOSED || mDeclaredResources.hasResources())
    {
        return mErrorState.reject(
            SafetyStatusCode::kInvalidState, "Release graphs and declared resources before TRT IO");
    }
    return recordResult(mRunner.deinitializeResources());
}

SafetyStatus SafetyRuntime::deinitialize() noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mDeclaredResources.hasResources() && mRunnerSpec.executors.empty())
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Declared resources require phased teardown");
    }
    if (mLifecycleState != LifecycleState::kCLOSED)
    {
        auto const status = deinitializeGraphs();
        if (!status.okStatus())
        {
            return status;
        }
    }
    // Graphs no longer borrow processing/model-local storage. Destroy borrowers before their backing memory.
    mInference.reset();
    if (mCapabilityCoordinator != nullptr)
    {
        auto const status = recordResult(mCapabilityCoordinator->deinitialize());
        if (!status.okStatus())
        {
            return status;
        }
    }
    mCapabilityCoordinator.reset();
    mProcessing.reset();
    mObjects.clear();
    mModel.reset();
    mOutput = {};
    if (!mRunnerSpec.executors.empty())
    {
        auto const status = recordResult(mDeclaredResources.deinitialize());
        if (!status.okStatus())
        {
            return status;
        }
    }
    mRunnerSpec = {};
    mEngines = {};
    mCompanions = {};
    mLLMConfig = {};
    mVisualConfig = {};
    mModelSpec = {};
    mArtifactPaths = {};
    mDefinition = nullptr;
    return deinitializeResources();
}

SafetyStatus SafetyRuntime::loadArtifacts(
    SafetyRuntimeArtifactBundle const& bundle, SafetyRuntimeOptions const& options)
{
    auto const* definition = findModelInferenceDefinition(options.model);
    if (definition == nullptr || bundle.engineDir.empty() || bundle.multimodalEngineDir.empty())
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid Safety model or artifact paths");
    }
    mProcessing = createSafetyRequestProcessing(*definition);
    if (mProcessing == nullptr)
    {
        return SafetyStatus::error(
            SafetyStatusCode::kInvalidState, "Processing capabilities are unavailable for this model recipe");
    }
    mDefinition = definition;
    mArtifactPaths = bundle;
    mLLMConfig = readConfig(mArtifactPaths.engineDir);
    mVisualConfig = readConfig(mArtifactPaths.multimodalEngineDir);
    validateBuilder(mLLMConfig, 2);
    validateBuilder(mVisualConfig, 1);
    if (mLLMConfig.at("model") != "qwen2" || mVisualConfig.at("model_type") != "internvl"
        || mLLMConfig.at("builder_config").at("is_vlm") != true)
    {
        throw std::invalid_argument("Expected InternVL Safety artifacts");
    }
    mModelSpec.owner = 1;
    mModelSpec.vocabSize = positive(mLLMConfig, "vocab_size");
    mModelSpec.hiddenSize = positive(mLLMConfig, "hidden_size");
    mModelSpec.layerCount = positive(mLLMConfig, "num_hidden_layers");
    mModelSpec.kvHeadCount = positive(mLLMConfig, "num_key_value_heads");
    auto const attentionHeads = positive(mLLMConfig, "num_attention_heads");
    if (mModelSpec.hiddenSize % attentionHeads != 0)
    {
        throw std::invalid_argument("Invalid attention head dimensions");
    }
    mModelSpec.headDim
        = mLLMConfig.contains("head_dim") ? positive(mLLMConfig, "head_dim") : mModelSpec.hiddenSize / attentionHeads;
    auto const& profiles = mLLMConfig.at("builder_config").at("profiles");
    for (auto const& profile : profiles)
    {
        if (positive(profile, "batch_size") != 1
            || positive(profile, "input_seq_len") > positive(profile, "kv_cache_capacity"))
        {
            throw std::invalid_argument("Expected single-request fixed Safety profiles");
        }
    }
    if (positive(profiles[1], "input_seq_len") != 1)
    {
        throw std::invalid_argument("Decode profile must contain one token");
    }
    for (std::size_t index = 0; index < definition->executorCount; ++index)
    {
        auto const visual = definition->executors[index].engineRole == EngineRole::kVisual;
        auto const enginePath
            = std::filesystem::absolute((visual ? mArtifactPaths.multimodalEngineDir : mArtifactPaths.engineDir)
                / (visual ? "visual_safety.engine" : "llm_safety.engine"));
        mCompanions[index] = enginePath.string() + ".so";
        if (!std::filesystem::is_regular_file(mCompanions[index]))
        {
            throw std::invalid_argument("Missing Safety engine companion library");
        }
        readEngine(enginePath, mEngines[index]);
        mRunnerSpec.executors.push_back(
            {mEngines[index].data(), static_cast<int64_t>(mEngines[index].size()), mCompanions[index].c_str(),
                visual ? std::vector<int64_t>{0} : std::vector<int64_t>{0, 1}, options.enableCudaGraphCapture});
        mRunnerSpec.executorMainStreamIds.push_back(0);
    }
    for (std::size_t index = 0; index < definition->stageCount; ++index)
    {
        auto const& stage = definition->stages[index];
        mRunnerSpec.stages.push_back({std::string(stage.id), stage.resourceSlot,
            static_cast<std::size_t>(stage.executorId), {stage.profileId}, {{"llm", sizeof(llm::ModelLocalData)}}});
    }
    for (int32_t layer = 0; layer < mModelSpec.layerCount; ++layer)
    {
        mRunnerSpec.ioAliases.push_back(
            {1, {"past_key_values_" + std::to_string(layer), "present_key_values_" + std::to_string(layer)}});
    }
    return SafetyStatus::ok();
}

SafetyStatus SafetyRuntime::prepareModel()
{
    auto& processing = *mProcessing;
    auto status = initialize(mRunnerSpec);
    if (!status.okStatus())
    {
        return status;
    }
    status = mResolver.prepare(contexts());
    if (!status.okStatus())
    {
        return status;
    }
    llm::StageResourceSpec prefill{};
    llm::StageResourceSpec decode{};
    auto const& profiles = mLLMConfig.at("builder_config").at("profiles");
    for (auto const& context : contexts())
    {
        if (context.plan.id == "prefill" || context.plan.id == "decode")
        {
            bool const isPrefill = context.plan.id == "prefill";
            status = describeStage(context, profiles[isPrefill ? 0 : 1], isPrefill ? prefill : decode);
            if (!status.okStatus())
            {
                return status;
            }
        }
    }
    status = processing.configure(mArtifactPaths, mLLMConfig, mVisualConfig);
    if (!status.okStatus())
    {
        return status;
    }
    mCapabilityCoordinator = std::make_unique<ProcessingCapabilityCoordinator>(
        mLifecycleState, processing.configuration(), processing.capabilities());
    status = mCapabilityCoordinator->configureStages(*mDefinition, contexts());
    if (!status.okStatus())
    {
        return status;
    }
    auto& capabilities = *mCapabilityCoordinator;
    auto modelSpec = mModelSpec;
    modelSpec.padTokenId = processing.configuration().tokenizer.padTokenId;
    llm::ModelLocalDataFactory factory{modelSpec, std::move(prefill), std::move(decode)};
    ResourceDeclaration modelDeclaration{};
    status = factory.declareResources(modelDeclaration);
    if (!status.okStatus())
    {
        return status;
    }
    ResourceDeclaration capabilityDeclaration{};
    status = capabilities.declareResources(capabilityDeclaration);
    if (!status.okStatus())
    {
        return status;
    }
    // The runtime reserves owners 1 and 2 for model-local data and generation output.
    if (capabilityDeclaration.owner <= kOUTPUT_OWNER)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Processing resource owner is reserved");
    }
    auto const capacity = profiles[0].at("kv_cache_capacity").get<std::size_t>();
    if (capacity > std::numeric_limits<std::size_t>::max() / sizeof(int32_t))
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Generation capacity is out of range");
    }
    ResourceDeclaration outputDeclaration{};
    outputDeclaration.owner = kOUTPUT_OWNER;
    outputDeclaration.memory.push_back({kOUTPUT_TOKENS, capacity * sizeof(int32_t), RuntimeMemoryKind::kPAGEABLE});
    auto& manager = declaredResources();
    status = manager.prepareResources({modelDeclaration, capabilityDeclaration, outputDeclaration});
    if (!status.okStatus())
    {
        return status;
    }
    status = manager.resolveBorrowedResources(modelDeclaration.owner, mResolver);
    if (!status.okStatus())
    {
        return status;
    }
    PreparedResources modelResources{};
    status = manager.resourcesFor(modelDeclaration.owner, modelResources);
    if (!status.okStatus())
    {
        return status;
    }
    status = factory.create(modelResources, mModel);
    if (!status.okStatus())
    {
        return status;
    }
    // This factory creates exactly llm::ModelLocalData; no type-erased external object enters here.
    auto& data = static_cast<llm::ModelLocalData&>(*mModel);
    mObjects.push_back({0, "llm", &data});
    if (!capabilityDeclaration.borrowed.empty())
    {
        status = manager.resolveBorrowedResources(capabilityDeclaration.owner, mResolver);
        if (!status.okStatus())
        {
            return status;
        }
    }
    PreparedResources capabilityResources{};
    status = manager.resourcesFor(capabilityDeclaration.owner, capabilityResources);
    if (!status.okStatus())
    {
        return status;
    }
    status = capabilities.initializeResources(capabilityResources, data);
    if (!status.okStatus())
    {
        return status;
    }
    status = capabilities.initializeModelData(data);
    if (!status.okStatus())
    {
        return status;
    }
    RuntimeMemoryView output{};
    status = manager.memory(kOUTPUT_OWNER, kOUTPUT_TOKENS, output);
    if (!status.okStatus())
    {
        return status;
    }
    mOutput = {static_cast<int32_t*>(output.data), output.sizeInBytes / sizeof(int32_t)};
    for (auto const& context : contexts())
    {
        status = completeContext(context.plan.id.c_str(), context.plan.slot, {{"llm", &data, sizeof(data)}});
        if (!status.okStatus())
        {
            return status;
        }
    }
    status = mDefinition == nullptr
        ? SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Prepared model recipe is unavailable")
        : createModelInference(*mDefinition, data, contexts(), capabilities, capabilities, mInference);
    if (!status.okStatus())
    {
        return status;
    }
    auto warmup = [](LMStageContext const& context, int64_t profile, void* userData) {
        return static_cast<ProcessingCapabilityCoordinator*>(userData)->prepareWarmup(context, profile);
    };
    return prepare(warmup, &capabilities);
}

SafetyStatus SafetyRuntime::runRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response) noexcept
{
    auto& processing = *mProcessing;
    if (mInference == nullptr || request.requests.size() != 1 || request.maxGenerateLength <= 0
        || static_cast<uint64_t>(request.maxGenerateLength) > mOutput.capacity || response.outputIds.size() != 1
        || response.outputIds[0].capacity() < static_cast<uint64_t>(request.maxGenerateLength))
    {
        return SafetyStatus::error(
            SafetyStatusCode::kInvalidArgument, "Invalid request or insufficient output capacity");
    }
    auto const& input = request.requests.front();
    if (input.messages.empty() || input.imageBuffers.size() != 1)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Expected messages and one InternVL image");
    }
    auto const& image = input.imageBuffers.front();
    constexpr std::size_t kRGB_CHANNELS{3};
    if (image.data == nullptr || image.width <= 0 || image.height <= 0
        || static_cast<std::size_t>(image.width) > std::numeric_limits<std::size_t>::max() / kRGB_CHANNELS
        || static_cast<std::size_t>(image.height)
            > image.sizeInBytes / (static_cast<std::size_t>(image.width) * kRGB_CHANNELS))
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid RGB image storage");
    }
    for (auto const& message : input.messages)
    {
        for (auto const& content : message.contents)
        {
            if (content.type != "text" && content.type != "image")
            {
                return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Unsupported message content type");
            }
        }
    }
    auto status = processing.validateResponse(request, response);
    if (!status.okStatus())
    {
        return status;
    }
    ProcessingRequest prepared{};
    status = processing.prepareRequest(request, prepared);
    if (!status.okStatus())
    {
        return status;
    }
    if (prepared.maxGeneratedTokens != request.maxGenerateLength)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Prepared generation limit differs from request");
    }
    ModelInferenceResult result{};
    response.outputIds[0].clear();
    status = mInference->run(*this, prepared, mOutput, result);
    if (status.okStatus())
    {
        status = processing.formatResponse(result, response);
    }
    if (status.okStatus())
    {
        auto& ids = response.outputIds[0];
        ids.resize(result.generatedTokens.size);
        std::copy_n(result.generatedTokens.data, result.generatedTokens.size, ids.data());
    }
    return status;
}

bool SafetyRuntime::fatal() const noexcept
{
    return mLifecycleState == LifecycleState::kFAILED || mErrorState.fatal() || mRunner.fatal()
        || mDeclaredResources.status().fatalStatus();
}

nvinfer2::safe::ISafeRecorder& SafetyRuntime::errorRecorder() noexcept
{
    return mErrorState.recorder();
}

LifecycleState SafetyRuntime::lifecycleState() const noexcept
{
    return fatal() ? LifecycleState::kFAILED : mLifecycleState;
}

bool SafetyRuntime::observeErrors() noexcept
{
    auto const declared = mDeclaredResources.status();
    if (declared.fatalStatus())
    {
        mErrorState.setFatal(declared);
    }
    if (fatal())
    {
        mErrorState.setFatal(SafetyStatus::fatal(SafetyStatusCode::kFailedExecution, "Safety runtime is terminal"));
        mLifecycleState = LifecycleState::kFAILED;
    }
    return mErrorState.observe();
}

SafetyStatus SafetyRuntime::recordResult(SafetyStatus status) noexcept
{
    // Retain the first fatal error and mark the shared lifecycle kFAILED. Nonfatal results preserve the lifecycle
    // unless a terminal failure is observed in a dependent component or recorder.
    if (status.fatalStatus())
    {
        mErrorState.setFatal(status);
        mLifecycleState = LifecycleState::kFAILED;
    }
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    return status;
}

} // namespace safety
} // namespace trt_edgellm
