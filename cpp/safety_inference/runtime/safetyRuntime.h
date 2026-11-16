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

#include "model_inference/llm/llmModelLocalData.h"
#include "model_inference/modelInference.h"
#include "model_inference/modelInferenceCatalog.h"
#include "resources/declaredResourceManager.h"
#include "resources/trtResourceManager.h"
#include "safetyGenerationTypes.h"
#include "safety_inference/trt_runner/runnerResourceResolver.h"
#include "safety_inference/trt_runner/trtSafetyRunner.h"

#include <array>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace trt_edgellm
{
namespace safety
{
//! Directories produced by the Safety LLM and visual builders.
struct SafetyRuntimeArtifactBundle
{
    std::filesystem::path engineDir;
    std::filesystem::path multimodalEngineDir;
};

struct SafetyRuntimeOptions
{
    ModelInferenceId model{ModelInferenceId::kInternVL3_1B_Vlm_Vanilla};
    bool enableCudaGraphCapture{false};
};

class SafetyRequestProcessing;
class ProcessingCapabilityCoordinator;

//! Internal execution owner for one caller-owned current CUDA context and externally serialized requests.
//! Healthy teardown releases every graph borrower before destruction. Terminal failure forbids SDK cleanup, so the
//! application must retain the Runtime and graph handles until containment termination.
class SafetyRuntime : public ModelInferenceRunner
{
public:
    SafetyRuntime() noexcept;
    explicit SafetyRuntime(nvinfer2::safe::ISafeRecorder& recorder) noexcept;
    ~SafetyRuntime() noexcept;
    SafetyRuntime(SafetyRuntime const&) = delete;
    SafetyRuntime& operator=(SafetyRuntime const&) = delete;

    LifecycleState lifecycleState() const noexcept;
    //! Initialization resource preparation shares this runtime's lifecycle.
    DeclaredResourceManager& declaredResources() noexcept
    {
        return mDeclaredResources;
    }

    //! Initialization only: validate fixed assignments and prepare owned TRT IO and streams.
    SafetyStatus initialize(SafetyRuntimeInitSpec const& spec) noexcept;

    //! Load builder artifacts and prepare the complete fixed model flow, including mandatory warmup.
    SafetyStatus initialize(SafetyRuntimeArtifactBundle const& artifacts, SafetyRuntimeOptions const& options) noexcept;

    //! One externally serialized request; input and caller-preallocated response storage outlive this call.
    //! A fatal failure requires retaining the runtime and its recorders for application containment.
    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response) noexcept;

    //! Last public request result; lifecycleState() separately describes admission/readiness.
    SafetyStatus requestStatus() const noexcept;

    //! Immutable context metadata for caller-owned model-local resource preparation.
    std::vector<LMStageContext> const& contexts() const noexcept;

    //! Initialization only: complete a declared context with stable, non-owning resource views.
    SafetyStatus completeContext(
        char const* stage, ResourceSlotId slot, std::vector<SafetyModelLocalResourceView> const& views) noexcept;

    //! Initialization only: warm every admitted stage/profile and seal Operational resources.
    SafetyStatus prepare(LMStageWarmupCallback warmup, void* userData = nullptr) noexcept;

    //! Operational catalog dispatch prepares the selected fixed-profile context before execution.
    SafetyStatus runLMStage(LMStageId stage, ResourceSlotId slot, LMStageOutputs& outputs,
        LMStagePrepareCallback prepare = nullptr, void* userData = nullptr) noexcept override;

    //! Healthy Deinit Preparation: destroy graphs/modules, retaining borrowed IO/streams for caller cleanup.
    SafetyStatus deinitializeGraphs() noexcept;

    //! Release IO/streams after graph, model-local object and declared-resource teardown.
    SafetyStatus deinitializeResources() noexcept;

    //! Combined teardown when no caller-owned resources require cleanup between the two phases.
    SafetyStatus deinitialize() noexcept;

    //! Terminal errors prohibit further CUDA/DLI work, including cleanup.
    bool fatal() const noexcept;

    //! Runtime recorder shared by every graph and execution component.
    nvinfer2::safe::ISafeRecorder& errorRecorder() noexcept;

private:
    SafetyStatus loadArtifacts(SafetyRuntimeArtifactBundle const& bundle, SafetyRuntimeOptions const& options);
    SafetyStatus prepareModel();
    SafetyStatus runRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response) noexcept;
    bool observeErrors() noexcept;
    SafetyStatus recordResult(SafetyStatus status) noexcept;
    LifecycleState mLifecycleState{LifecycleState::kINITIALIZATION};
    SafetyErrorState mErrorState;
    DeclaredResourceManager mDeclaredResources;
    TRTResourceManager mResources;
    TRTSafetyRunner mRunner;
    ModelInferenceDefinition const* mDefinition{nullptr};
    std::unique_ptr<SafetyRequestProcessing> mProcessing;
    std::unique_ptr<ProcessingCapabilityCoordinator> mCapabilityCoordinator;
    // Graph initialization borrows these engine bytes and companion paths until graph teardown.
    SafetyRuntimeArtifactBundle mArtifactPaths;
    std::array<std::vector<char>, 2> mEngines;
    std::array<std::string, 2> mCompanions;
    nlohmann::json mLLMConfig;
    nlohmann::json mVisualConfig;
    llm::ModelResourceSpec mModelSpec;
    SafetyRuntimeInitSpec mRunnerSpec;
    std::vector<ModelLocalDataBinding> mObjects;
    RunnerResourceResolver mResolver{mObjects};
    std::unique_ptr<SafetyModelLocalData> mModel;
    std::unique_ptr<ModelInference> mInference;
    TokenIdBuffer mOutput{};
    SafetyStatus mRequestStatus{SafetyStatus::ok()};
};

} // namespace safety
} // namespace trt_edgellm
