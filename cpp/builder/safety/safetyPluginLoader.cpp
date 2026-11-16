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
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "builder/safety/safetyPluginLoader.h"

#include "common/logger.h"
#include "safety_inference/plugins/safetyCreatorRegistration.h"

#include <NvInferSafePlugin.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <type_traits>
#include <vector>

namespace trt_edgellm::safety::builder
{
namespace
{

class SafetyPluginRecorder : public nvinfer2::safe::ISafeRecorder
{
public:
    SafetyPluginRecorder()
        : ISafeRecorder(nvinfer2::safe::Severity::kINFO)
    {
    }

    int32_t getNbErrors() const noexcept override
    {
        return std::min(mNbErrors.load(), kMaxErrors);
    }

    nvinfer1::ErrorCode getErrorCode(int32_t errorIdx) const noexcept override
    {
        return validIndex(errorIdx) ? mRecordedErrors[errorIdx].code : nvinfer1::ErrorCode::kUNSPECIFIED_ERROR;
    }

    nvinfer1::IErrorRecorder::ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept override
    {
        return validIndex(errorIdx) ? mRecordedErrors[errorIdx].description.data() : "";
    }

    bool hasOverflowed() const noexcept override
    {
        return mOverflowed.load();
    }

    void clear() noexcept override
    {
        mNbErrors = 0;
        mOverflowed = false;
    }

    bool reportError(nvinfer1::ErrorCode code, nvinfer1::IErrorRecorder::ErrorDesc description) noexcept override
    {
        int32_t const index = mNbErrors.fetch_add(1);
        if (index >= kMaxErrors)
        {
            mOverflowed = true;
            return code == nvinfer1::ErrorCode::kINTERNAL_ERROR;
        }
        mRecordedErrors[index].code = code;
        if (description == nullptr)
        {
            mRecordedErrors[index].description[0] = '\0';
        }
        else
        {
            std::strncpy(
                mRecordedErrors[index].description.data(), description, nvinfer1::IErrorRecorder::kMAX_DESC_LENGTH);
            mRecordedErrors[index].description[nvinfer1::IErrorRecorder::kMAX_DESC_LENGTH] = '\0';
        }
        return code == nvinfer1::ErrorCode::kINTERNAL_ERROR;
    }

    nvinfer2::safe::RefCount incRefCount() noexcept override
    {
        return ++mRefCount;
    }

    nvinfer2::safe::RefCount decRefCount() noexcept override
    {
        return --mRefCount;
    }

    bool reportWarn(nvinfer2::safe::ErrorDesc description) noexcept override
    {
        LOG_WARNING("TensorRT Safety: %s", description != nullptr ? description : "");
        return true;
    }

    bool reportDebug(nvinfer2::safe::ErrorDesc description) noexcept override
    {
        LOG_DEBUG("TensorRT Safety: %s", description != nullptr ? description : "");
        return true;
    }

    bool reportInfo(nvinfer2::safe::ErrorDesc description) noexcept override
    {
        LOG_INFO("TensorRT Safety: %s", description != nullptr ? description : "");
        return true;
    }

    bool reportVerbose(nvinfer2::safe::ErrorDesc description) noexcept override
    {
        LOG_DEBUG("TensorRT Safety: %s", description != nullptr ? description : "");
        return true;
    }

private:
    static constexpr int32_t kMaxErrors{16};
    static constexpr size_t kDescriptionSize{nvinfer1::IErrorRecorder::kMAX_DESC_LENGTH + 1};

    struct StoredError
    {
        nvinfer1::ErrorCode code{nvinfer1::ErrorCode::kSUCCESS};
        std::array<char, kDescriptionSize> description{};
    };

    bool validIndex(int32_t index) const noexcept
    {
        return index >= 0 && index < std::min(mNbErrors.load(), kMaxErrors);
    }

    std::array<StoredError, kMaxErrors> mRecordedErrors{};
    std::atomic<int32_t> mNbErrors{0};
    std::atomic<bool> mOverflowed{false};
    std::atomic<nvinfer2::safe::RefCount> mRefCount{0};
};

SafetyPluginRecorder& processLifetimeRecorder() noexcept
{
    // ISafePluginRegistry is a process-wide singleton and retains its recorder.
    // Deliberately keep this recorder alive through shared-library teardown.
    static auto* recorder = new SafetyPluginRecorder();
    return *recorder;
}

std::mutex& registryMutex() noexcept
{
    // Registration mutates a process-wide registry. Serialize registration
    // lifetimes so one build cannot remove creators used by another build.
    static auto* mutex = new std::mutex();
    return *mutex;
}

// CUDA Safety and native headers expose different public context-creation signatures.
template <typename CreateContext>
CUresult createBuildContext(CreateContext create, CUcontext& context, CUdevice device)
{
    if constexpr (std::is_invocable_v<CreateContext, CUcontext*, unsigned int, CUdevice>)
    {
        return create(&context, 0U, device);
    }
    else
    {
        return create(&context, nullptr, 0U, device);
    }
}

class SafetyPluginRegistrationImpl final : public SafetyPluginRegistration
{
public:
    SafetyPluginRegistrationImpl()
        : mRegistryLock(registryMutex())
    {
    }

    ~SafetyPluginRegistrationImpl() noexcept override
    {
        finishBuild();
        bool canUnloadLibrary{true};
        if (mRegistry != nullptr)
        {
            for (auto iterator = mCreators.rbegin(); iterator != mCreators.rend(); ++iterator)
            {
                auto const status = mRegistry->deregisterCreator(**iterator);
                if (status != nvinfer1::ErrorCode::kSUCCESS)
                {
                    LOG_WARNING("Failed to deregister a Safety plugin creator (error=%d).", static_cast<int>(status));
                    canUnloadLibrary = false;
                }
            }
            auto const status = mRegistry->setSafeRecorder(processLifetimeRecorder());
            if (status != nvinfer1::ErrorCode::kSUCCESS)
            {
                LOG_WARNING(
                    "Failed to restore the process-lifetime Safety recorder (error=%d).", static_cast<int>(status));
            }
        }
        if (mLibrary != nullptr)
        {
            if (!canUnloadLibrary)
            {
                // Keep the DSO mapped if the registry may still contain one
                // of its creator pointers.
                LOG_WARNING("Retaining the Safety plugin library because creator deregistration failed.");
            }
            else if (dlclose(mLibrary) != 0)
            {
                char const* const error = dlerror();
                LOG_WARNING("Failed to unload the Safety plugin library: %s",
                    error != nullptr ? error : "unknown dynamic-loader error");
            }
        }
    }

    bool initialize(std::string const& configuredPath, std::initializer_list<SafetyPluginDescriptor> requiredPlugins)
    {
        if (requiredPlugins.size() == 0 || requiredPlugins.size() > kMAX_SAFETY_CREATORS)
        {
            LOG_ERROR("Between 1 and 100 required Safety plugins must be specified.");
            return false;
        }

        mRegistry = nvinfer2::safe::getSafePluginRegistry(processLifetimeRecorder());
        if (mRegistry == nullptr)
        {
            LOG_ERROR("TensorRT did not provide a Safety plugin registry.");
            return false;
        }
        nvinfer1::IPluginCreatorInterface* const* registered{nullptr};
        int32_t registeredCount{};
        auto const enumerationStatus
            = getValidatedSafetyCreators(*mRegistry, processLifetimeRecorder(), registered, registeredCount);
        if (enumerationStatus != nvinfer1::ErrorCode::kSUCCESS)
        {
            LOG_ERROR("Cannot enumerate valid Safety plugin creators (error=%d).", static_cast<int>(enumerationStatus));
            return false;
        }

        size_t existingCount{};
        for (auto const& descriptor : requiredPlugins)
        {
            if (descriptor.pluginNamespace == nullptr || descriptor.pluginName == nullptr
                || descriptor.pluginVersion == nullptr)
            {
                LOG_ERROR("Safety plugin descriptors must contain a namespace, name, and version.");
                return false;
            }
            for (int32_t i = 0; i < registeredCount; ++i)
            {
                auto* creator = static_cast<nvinfer2::safe::ISafePluginCreatorV3One*>(registered[i]);
                if (std::strcmp(creator->getPluginName(), descriptor.pluginName) != 0
                    || std::strcmp(creator->getPluginVersion(), descriptor.pluginVersion) != 0
                    || std::strcmp(creator->getPluginNamespace(), descriptor.pluginNamespace) != 0)
                {
                    continue;
                }
                Dl_info owner{};
                if (dladdr(creator, &owner) == 0 || owner.dli_fname == nullptr)
                {
                    LOG_ERROR("Cannot resolve the DSO owning an existing Safety plugin creator.");
                    return false;
                }
                if (!mCreatorLibrary.empty() && mCreatorLibrary != owner.dli_fname)
                {
                    LOG_ERROR("Required Safety plugin creators belong to different libraries.");
                    return false;
                }
                mCreatorLibrary = owner.dli_fname;
                ++existingCount;
                break;
            }
        }
        if (existingCount != 0 && existingCount != requiredPlugins.size())
        {
            LOG_ERROR("Cannot combine existing and newly loaded Safety plugin creators.");
            return false;
        }
        std::string path = mCreatorLibrary.empty() ? configuredPath : mCreatorLibrary;
        if (path.empty())
        {
            char const* const environmentPath = std::getenv("EDGELLM_SAFETY_PLUGIN_PATH");
            path = environmentPath != nullptr ? environmentPath : "build/libNvInfer_edgellm_plugin_safe.so";
        }

        mLibrary = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (mLibrary == nullptr)
        {
            LOG_ERROR("Cannot load Safety plugin library '%s': %s", path.c_str(), dlerror());
            return false;
        }

        using GetCreator = nvinfer1::IPluginCreatorInterface* (*) (char const*, char const*);
        dlerror();
        auto const getCreator = reinterpret_cast<GetCreator>(dlsym(mLibrary, "getSafetyPluginCreator"));
        char const* const symbolError = dlerror();
        if (symbolError != nullptr || getCreator == nullptr)
        {
            LOG_ERROR("Safety plugin library '%s' does not export getSafetyPluginCreator: %s", path.c_str(),
                symbolError != nullptr ? symbolError : "symbol unavailable");
            return false;
        }

        std::vector<nvinfer2::safe::ISafePluginCreatorV3One*> selected;
        selected.reserve(requiredPlugins.size());
        for (auto const& descriptor : requiredPlugins)
        {
            auto* creator = validateSafetyCreator(
                getCreator(descriptor.pluginNamespace, descriptor.pluginName), processLifetimeRecorder());
            if (creator == nullptr || std::strcmp(creator->getPluginName(), descriptor.pluginName) != 0
                || std::strcmp(creator->getPluginVersion(), descriptor.pluginVersion) != 0
                || std::strcmp(creator->getPluginNamespace(), descriptor.pluginNamespace) != 0)
            {
                LOG_ERROR("Safety plugin library '%s' does not provide required creator %s::%s version %s.",
                    path.c_str(), descriptor.pluginNamespace, descriptor.pluginName, descriptor.pluginVersion);
                return false;
            }
            selected.push_back(creator);
        }
        mCreators.resize(selected.size());
        int32_t newlyRegisteredCount{};
        auto const status = registerSafetyCreators(*mRegistry, processLifetimeRecorder(), selected.data(),
            static_cast<int32_t>(selected.size()), mCreators.data(), newlyRegisteredCount);
        mCreators.resize(newlyRegisteredCount);
        if (status != nvinfer1::ErrorCode::kSUCCESS)
        {
            LOG_ERROR("Failed to register required Safety plugin creators (error=%d).", static_cast<int>(status));
            return false;
        }
        return true;
    }

    bool initializeAttentionModules(bool remoteAutoTuning) noexcept override
    {
        if (mPhase != Phase::kIDLE || mLibrary == nullptr)
        {
            LOG_ERROR("Safety attention build initialization requires a loaded DSO and cannot be retried.");
            return false;
        }
        mPhase = Phase::kINITIALIZING;
        using CreateSession = int32_t (*)(void**, bool) noexcept;
        auto const create = reinterpret_cast<CreateSession>(dlsym(mLibrary, "createSafetyAttentionBuildSession"));
        mDestroySession = reinterpret_cast<DestroySession>(dlsym(mLibrary, "destroySafetyAttentionBuildSession"));
        if (create == nullptr || mDestroySession == nullptr)
        {
            LOG_ERROR("Safety plugin library lacks attention build-session support; rebuild it with the host builder.");
            return false;
        }
        if (!remoteAutoTuning)
        {
            CUdevice device{};
            auto const check = [](CUresult status, char const* operation) noexcept {
                if (status != CUDA_SUCCESS)
                {
                    LOG_ERROR("Safety attention build CUDA initialization failed in %s (status=%d).", operation,
                        static_cast<int>(status));
                    return false;
                }
                return true;
            };
            bool const initialized = check(cuInit(0), "cuInit")
                && check(cuCtxGetCurrent(&mPreviousContext), "cuCtxGetCurrent")
                && check(cuDeviceGet(&device, 0), "cuDeviceGet")
                && check(createBuildContext(&cuCtxCreate, mContext, device), "cuCtxCreate");
            if (!initialized)
            {
                return false;
            }
        }
        // Safe CUDA requires module loading immediately after context creation, before cudart setup.
        int32_t const sessionStatus = create(&mSession, !remoteAutoTuning);
        if (sessionStatus != static_cast<int32_t>(CUDA_SUCCESS))
        {
            LOG_ERROR("Cannot initialize Safety attention build modules (status=%d).", sessionStatus);
            return false;
        }
        if (!remoteAutoTuning)
        {
            size_t stackLimit{};
            auto const runtimeStatus = cudaDeviceGetLimit(&stackLimit, cudaLimitStackSize);
            if (runtimeStatus != cudaSuccess)
            {
                mCudaError = static_cast<int32_t>(runtimeStatus);
                LOG_ERROR("Cannot initialize cudart for Safety attention build (status=%d).",
                    static_cast<int>(runtimeStatus));
                return false;
            }
        }
        mPhase = Phase::kREADY;
        return true;
    }

    bool completeGpuWork() noexcept override
    {
        if (mPhase == Phase::kCOMPLETED || mPhase == Phase::kFINISHED || mPhase == Phase::kCLEANUP_FAILED)
        {
            return mCudaError == 0;
        }
        bool const runtimeInitialized = mPhase == Phase::kREADY;
        mPhase = Phase::kCOMPLETED;
        if (mContext != nullptr)
        {
            CUresult status = cuCtxSetCurrent(mContext);
            if (status == CUDA_SUCCESS && runtimeInitialized && mCudaError == 0)
            {
                status = cuCtxSynchronize();
            }
            if (status != CUDA_SUCCESS)
            {
                if (mCudaError == 0)
                {
                    mCudaError = static_cast<int32_t>(status);
                }
                LOG_ERROR("Safety attention build CUDA completion failed (status=%d).", static_cast<int>(status));
            }
        }
        return mCudaError == 0;
    }

    bool finishBuild() noexcept override
    {
        if (mPhase == Phase::kFINISHED || mPhase == Phase::kCLEANUP_FAILED)
        {
            return mPhase == Phase::kFINISHED;
        }
        bool success = completeGpuWork();
        if (mSession != nullptr)
        {
            int32_t const sessionStatus = mDestroySession(mSession, mCudaError == 0);
            mSession = nullptr;
            if (sessionStatus != static_cast<int32_t>(CUDA_SUCCESS))
            {
                LOG_ERROR("Cannot release Safety attention build modules (status=%d).", sessionStatus);
                success = false;
            }
        }
        if (mContext != nullptr)
        {
            CUresult const status = cuCtxDestroy(mContext);
            mContext = nullptr;
            CUresult const restoreStatus = cuCtxSetCurrent(mPreviousContext);
            if (status != CUDA_SUCCESS || restoreStatus != CUDA_SUCCESS)
            {
                LOG_ERROR("Safety attention build CUDA context teardown failed (destroy=%d, restore=%d).",
                    static_cast<int>(status), static_cast<int>(restoreStatus));
                success = false;
            }
        }
        mPhase = success ? Phase::kFINISHED : Phase::kCLEANUP_FAILED;
        return success;
    }

private:
    using DestroySession = int32_t (*)(void*, bool) noexcept;
    void* mSession{nullptr};
    DestroySession mDestroySession{nullptr};
    CUcontext mContext{nullptr};
    CUcontext mPreviousContext{nullptr};
    enum class Phase
    {
        kIDLE,
        kINITIALIZING,
        kREADY,
        kCOMPLETED,
        kFINISHED,
        kCLEANUP_FAILED
    };
    Phase mPhase{Phase::kIDLE};
    // Zero means GPU work is healthy; preserve the first CUDA failure through cleanup.
    int32_t mCudaError{};
    std::string mCreatorLibrary;
    std::unique_lock<std::mutex> mRegistryLock;
    nvinfer2::safe::ISafePluginRegistry* mRegistry{nullptr};
    std::vector<nvinfer1::IPluginCreatorInterface*> mCreators;
    void* mLibrary{nullptr};
};

} // namespace

std::unique_ptr<SafetyPluginRegistration> loadSafetyPlugins(
    std::string const& configuredPath, std::initializer_list<SafetyPluginDescriptor> requiredPlugins)
{
    auto registration = std::make_unique<SafetyPluginRegistrationImpl>();
    if (!registration->initialize(configuredPath, requiredPlugins))
    {
        return nullptr;
    }
    return registration;
}

} // namespace trt_edgellm::safety::builder
