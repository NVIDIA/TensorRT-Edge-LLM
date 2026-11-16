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

#include "trtSafetyRunner.h"

#include <algorithm>
#include <exception>

namespace trt_edgellm
{
namespace safety
{
namespace
{
bool validSpec(SafetyRuntimeInitSpec const& spec) noexcept
{
    if (spec.executors.empty() || spec.stages.empty())
    {
        return false;
    }
    if (!spec.executorMainStreamIds.empty())
    {
        if (spec.executorMainStreamIds.size() != spec.executors.size())
        {
            return false;
        }
        auto const largest = *std::max_element(spec.executorMainStreamIds.begin(), spec.executorMainStreamIds.end());
        if (largest >= spec.executors.size())
        {
            return false;
        }
        for (size_t id = 0; id <= largest; ++id)
        {
            if (std::find(spec.executorMainStreamIds.begin(), spec.executorMainStreamIds.end(), id)
                == spec.executorMainStreamIds.end())
            {
                return false;
            }
        }
    }
    for (auto const& group : spec.ioAliases)
    {
        if (group.executorIndex >= spec.executors.size() || group.tensors.size() < 2)
        {
            return false;
        }
        for (size_t member = 0; member < group.tensors.size(); ++member)
        {
            auto const& name = group.tensors[member];
            if (name.empty()
                || std::find(group.tensors.begin(), group.tensors.begin() + member, name)
                    != group.tensors.begin() + member)
            {
                return false;
            }
        }
    }
    for (auto const& config : spec.executors)
    {
        if (config.engine == nullptr || config.engineBytes <= 0 || config.companionSoPath == nullptr
            || config.companionSoPath[0] != '/' || config.declaredProfiles.empty())
        {
            return false;
        }
        for (size_t index = 0; index < config.declaredProfiles.size(); ++index)
        {
            auto const profile = config.declaredProfiles[index];
            if (profile < 0
                || std::find(config.declaredProfiles.begin(), config.declaredProfiles.begin() + index, profile)
                    != config.declaredProfiles.begin() + index)
            {
                return false;
            }
        }
    }
    for (size_t index = 0; index < spec.stages.size(); ++index)
    {
        auto const& plan = spec.stages[index];
        if (plan.id.empty() || plan.executorIndex >= spec.executors.size() || plan.profiles.empty())
        {
            return false;
        }
        for (size_t previous = 0; previous < index; ++previous)
        {
            if (plan.id == spec.stages[previous].id && plan.slot == spec.stages[previous].slot)
            {
                return false;
            }
        }
        auto const& admitted = spec.executors[plan.executorIndex].declaredProfiles;
        for (size_t profileIndex = 0; profileIndex < plan.profiles.size(); ++profileIndex)
        {
            auto const profile = plan.profiles[profileIndex];
            if (std::find(admitted.begin(), admitted.end(), profile) == admitted.end()
                || std::find(plan.profiles.begin(), plan.profiles.begin() + profileIndex, profile)
                    != plan.profiles.begin() + profileIndex)
            {
                return false;
            }
        }
        for (size_t resourceIndex = 0; resourceIndex < plan.modelLocalResources.size(); ++resourceIndex)
        {
            auto const& resource = plan.modelLocalResources[resourceIndex];
            if (resource.name.empty())
            {
                return false;
            }
            for (size_t previous = 0; previous < resourceIndex; ++previous)
            {
                if (resource.name == plan.modelLocalResources[previous].name)
                {
                    return false;
                }
            }
        }
    }
    for (size_t index = 0; index < spec.executors.size(); ++index)
    {
        for (auto const profile : spec.executors[index].declaredProfiles)
        {
            bool const assigned
                = std::any_of(spec.stages.begin(), spec.stages.end(), [index, profile](auto const& plan) {
                      return plan.executorIndex == index
                          && std::find(plan.profiles.begin(), plan.profiles.end(), profile) != plan.profiles.end();
                  });
            if (!assigned)
            {
                return false;
            }
        }
    }
    return true;
}
} // namespace

TRTSafetyRunner::TRTSafetyRunner(
    LifecycleState& lifecycleState, SafetyErrorState& errorState, TRTResourceManager& resources) noexcept
    : mErrorState(errorState)
    , mResources(resources)
    , mLifecycleState(lifecycleState)
{
}

bool TRTSafetyRunner::fatal() const noexcept
{
    return mLifecycleState == LifecycleState::kFAILED || mErrorState.fatal() || mResources.fatal()
        || std::any_of(mExecutors.begin(), mExecutors.end(), [](auto const& executor) { return executor->fatal(); });
}

bool TRTSafetyRunner::observeErrors() noexcept
{
    if (fatal())
    {
        mErrorState.setFatal(
            SafetyStatus::fatal(SafetyStatusCode::kFailedExecution, "Safety dependency reported a failure"));
    }
    return mErrorState.observe();
}

SafetyStatus TRTSafetyRunner::fail(nvinfer2::safe::ErrorCode code, char const* message) noexcept
{
    return mErrorState.fail(code, message);
}

SafetyStatus TRTSafetyRunner::initialize(SafetyRuntimeInitSpec const& spec) noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mLifecycleState != LifecycleState::kINITIALIZATION || !mExecutors.empty())
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Runner initialization requires an empty runtime");
    }
    if (!validSpec(spec))
    {
        return mErrorState.reject(
            SafetyStatusCode::kInvalidArgument, "Invalid executor, stage, profile or resource declaration");
    }
    std::vector<LMStageContext> contexts;
    try
    {
        mExecutors.reserve(spec.executors.size());
        mExecutorViews.reserve(spec.executors.size());
        contexts.reserve(spec.stages.size());
        for (size_t index = 0; index < spec.executors.size(); ++index)
        {
            mExecutors.push_back(std::make_unique<TRTSafetyGraphExecutor>(mErrorState));
            mExecutorViews.push_back(mExecutors.back().get());
        }
        for (auto const& plan : spec.stages)
        {
            contexts.push_back({plan, nullptr, {}, {}, plan.modelLocalResources.empty()});
            contexts.back().profiles.reserve(plan.profiles.size());
        }
    }
    catch (std::exception const& error)
    {
        return mErrorState.fail(
            nvinfer2::safe::ErrorCode::kFAILED_ALLOCATION, "Safety runner metadata allocation failed", error.what());
    }
    catch (...)
    {
        return fail(nvinfer2::safe::ErrorCode::kFAILED_ALLOCATION, "Safety runner metadata allocation failed");
    }
    // Safe CUDA requires all application modules to load before runtime initialization on the current context.
    for (auto& executor : mExecutors)
    {
        auto const executorStatus = executor->loadModules();
        if (!executorStatus.okStatus())
        {
            return mErrorState.propagateFailure(executorStatus, nvinfer2::safe::ErrorCode::kFAILED_INITIALIZATION);
        }
    }
    // Safe CUDA device queries cannot lazily initialize cudart for a driver-created context.
    size_t stackLimit{};
    if (!mErrorState.updateStatus(fromCudaError(cudaDeviceGetLimit(&stackLimit, cudaLimitStackSize),
                                      "Safety CUDA runtime initialization failed"),
            nvinfer2::safe::ErrorCode::kFAILED_INITIALIZATION))
    {
        return mErrorState.status();
    }
    for (size_t index = 0; index < mExecutors.size(); ++index)
    {
        auto const status = mExecutors[index]->initialize(spec.executors[index]);
        if (!status.okStatus())
        {
            return mErrorState.propagateFailure(status, nvinfer2::safe::ErrorCode::kFAILED_INITIALIZATION);
        }
    }
    auto const resourceStatus = mResources.prepareResources(mExecutorViews, spec.executorMainStreamIds, spec.ioAliases);
    if (!resourceStatus.okStatus())
    {
        if (resourceStatus.fatalStatus())
        {
            mErrorState.setFatal(resourceStatus);
        }
        return resourceStatus;
    }
    try
    {
        for (auto& context : contexts)
        {
            auto const index = context.plan.executorIndex;
            context.mainStream = mResources.mainStream(index);
            auto const& metadata = mExecutors[index]->profiles();
            auto const& resources = mResources.profiles(index);
            for (auto const profile : context.plan.profiles)
            {
                auto const found = std::find_if(metadata.begin(), metadata.end(),
                    [profile](auto const& candidate) { return candidate.profileIndex == profile; });
                size_t const profileIndex = static_cast<size_t>(found - metadata.begin());
                context.profiles.push_back({&metadata[profileIndex], &resources[profileIndex], {}});
                auto const status = detail::initializeTensorViews(context.profiles.back());
                if (!status.okStatus())
                {
                    return fail(status.code() == SafetyStatusCode::kFailedAllocation
                            ? nvinfer2::safe::ErrorCode::kFAILED_ALLOCATION
                            : nvinfer2::safe::ErrorCode::kINVALID_CONFIG,
                        status.message());
                }
            }
        }
    }
    catch (std::exception const& error)
    {
        return mErrorState.fail(
            nvinfer2::safe::ErrorCode::kFAILED_ALLOCATION, "Safety tensor view publication failed", error.what());
    }
    catch (...)
    {
        return fail(nvinfer2::safe::ErrorCode::kFAILED_ALLOCATION, "Safety tensor view publication failed");
    }
    mContexts = std::move(contexts);
    return SafetyStatus::ok();
}

LMStageContext* TRTSafetyRunner::findContext(char const* stage, ResourceSlotId slot) noexcept
{
    return stage == nullptr ? nullptr : findContext(LMStageId{stage}, slot);
}

LMStageContext* TRTSafetyRunner::findContext(LMStageId stage, ResourceSlotId slot) noexcept
{
    for (auto& context : mContexts)
    {
        if (context.plan.id == stage && context.plan.slot == slot)
        {
            return &context;
        }
    }
    return nullptr;
}

SafetyStatus TRTSafetyRunner::completeContext(
    char const* stage, ResourceSlotId slot, std::vector<SafetyModelLocalResourceView> const& views) noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mLifecycleState != LifecycleState::kINITIALIZATION || mContexts.empty())
    {
        return mErrorState.reject(
            SafetyStatusCode::kInvalidState, "Model-local resources require initialized contexts");
    }
    auto* context = findContext(stage, slot);
    if (context == nullptr || views.size() != context->plan.modelLocalResources.size())
    {
        return mErrorState.reject(
            SafetyStatusCode::kInvalidArgument, "Unknown stage/slot or inconsistent model-local resource count");
    }
    if (context->complete)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Model-local context is already complete");
    }
    for (auto const& requirement : context->plan.modelLocalResources)
    {
        size_t matches{0};
        for (auto const& view : views)
        {
            if (view.name == requirement.name)
            {
                if (view.data == nullptr || view.sizeInBytes < requirement.minimumBytes)
                {
                    return mErrorState.reject(
                        SafetyStatusCode::kInvalidArgument, "Model-local resource pointer or capacity is invalid");
                }
                ++matches;
            }
        }
        if (matches != 1)
        {
            return mErrorState.reject(
                SafetyStatusCode::kInvalidArgument, "Model-local resource name must occur exactly once");
        }
    }
    try
    {
        context->modelLocalResources = views;
    }
    catch (std::exception const& error)
    {
        return mErrorState.fail(
            nvinfer2::safe::ErrorCode::kFAILED_ALLOCATION, "Safety model-local view allocation failed", error.what());
    }
    catch (...)
    {
        return fail(nvinfer2::safe::ErrorCode::kFAILED_ALLOCATION, "Safety model-local view allocation failed");
    }
    context->complete = true;
    return SafetyStatus::ok();
}

SafetyStatus TRTSafetyRunner::prepare(LMStageWarmupCallback warmup, void* userData) noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mLifecycleState != LifecycleState::kINITIALIZATION || mContexts.empty()
        || std::any_of(mContexts.begin(), mContexts.end(), [](auto const& context) { return !context.complete; }))
    {
        return mErrorState.reject(
            SafetyStatusCode::kInvalidState, "Every stage context must be complete before readiness preparation");
    }
    if (warmup == nullptr)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidArgument, "Warmup requires an input preparation callback");
    }
    try
    {
        for (auto const& context : mContexts)
        {
            for (auto const profile : context.plan.profiles)
            {
                auto const status = warmup(context, profile, userData);
                if (!status.okStatus())
                {
                    return mErrorState.propagateFailure(status, nvinfer2::safe::ErrorCode::kFAILED_INITIALIZATION);
                }
                if (!observeErrors())
                {
                    return mErrorState.status();
                }
                auto const executorStatus = mExecutors[context.plan.executorIndex]->warmup(profile);
                if (!executorStatus.okStatus())
                {
                    return mErrorState.propagateFailure(
                        executorStatus, nvinfer2::safe::ErrorCode::kFAILED_INITIALIZATION);
                }
            }
        }
    }
    catch (std::exception const& error)
    {
        return mErrorState.fail(nvinfer2::safe::ErrorCode::kFAILED_INITIALIZATION,
            "Safety warmup callback threw an exception", error.what());
    }
    catch (...)
    {
        return fail(nvinfer2::safe::ErrorCode::kFAILED_INITIALIZATION, "Safety warmup callback threw an exception");
    }
    for (auto& executor : mExecutors)
    {
        auto const executorStatus = executor->seal();
        if (!executorStatus.okStatus())
        {
            return mErrorState.propagateFailure(executorStatus, nvinfer2::safe::ErrorCode::kFAILED_INITIALIZATION);
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus TRTSafetyRunner::runLMStage(LMStageId stage, ResourceSlotId slot, LMStageOutputs& outputs,
    LMStagePrepareCallback prepare, void* userData) noexcept
{
    outputs.context = nullptr;
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mLifecycleState != LifecycleState::kOPERATIONAL || mContexts.empty())
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Stage execution requires an operational runtime");
    }
    auto const* context = findContext(stage, slot);
    if (context == nullptr || context->plan.profiles.size() != 1)
    {
        return mErrorState.reject(
            SafetyStatusCode::kInvalidArgument, "Stage/slot requires one initialized fixed profile");
    }
    if (prepare != nullptr)
    {
        auto const status = prepare(*context, userData);
        if (status.fatalStatus())
        {
            return mErrorState.propagateFailure(status, nvinfer2::safe::ErrorCode::kFAILED_EXECUTION);
        }
        if (!observeErrors())
        {
            return mErrorState.status();
        }
        if (!status.okStatus())
        {
            return status;
        }
    }
    auto const status = mExecutors[context->plan.executorIndex]->execute(context->plan.profiles.front());
    if (!status.okStatus())
    {
        return mErrorState.propagateFailure(status, nvinfer2::safe::ErrorCode::kFAILED_EXECUTION);
    }
    outputs.context = context;
    return SafetyStatus::ok();
}

SafetyStatus TRTSafetyRunner::deinitializeGraphs() noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mExecutors.empty() || mLifecycleState == LifecycleState::kCLOSED)
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Runner has no active graphs to deinitialize");
    }
    for (auto& executor : mExecutors)
    {
        auto const executorStatus = executor->deinitialize();
        if (!executorStatus.okStatus())
        {
            return mErrorState.propagateFailure(executorStatus, nvinfer2::safe::ErrorCode::kINVALID_STATE);
        }
    }
    return SafetyStatus::ok();
}

SafetyStatus TRTSafetyRunner::deinitializeResources() noexcept
{
    if (!observeErrors())
    {
        return mErrorState.status();
    }
    if (mLifecycleState != LifecycleState::kCLOSED || mExecutors.empty())
    {
        return mErrorState.reject(SafetyStatusCode::kInvalidState, "Destroy graphs before releasing Runner resources");
    }
    auto const status = mResources.deinitialize(mExecutorViews);
    if (!status.okStatus())
    {
        return mErrorState.propagateFailure(status, nvinfer2::safe::ErrorCode::kINVALID_STATE);
    }
    mContexts.clear();
    mExecutorViews.clear();
    mExecutors.clear();
    return SafetyStatus::ok();
}

std::vector<LMStageContext> const& TRTSafetyRunner::contexts() const noexcept
{
    return mContexts;
}

} // namespace safety
} // namespace trt_edgellm
