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

#include "cuteDslModuleLoader.h"

#include "common/logger.h"

#include <unordered_map>

#if defined(ENABLE_CUTEDSL_MODULE_TEST_HOOK)
#include <cstdlib>
#include <cstring>
#endif

namespace trt_edgellm::detail
{

namespace
{

thread_local cudaError_t gCuteDslCudaError{cudaSuccess};

char const* nonNullName(char const* moduleName) noexcept
{
    return moduleName != nullptr ? moduleName : "<unnamed>";
}

} // namespace

void recordCuteDslCudaError(cudaError_t error) noexcept
{
    if (error != cudaSuccess && gCuteDslCudaError == cudaSuccess)
    {
        gCuteDslCudaError = error;
    }
}

void clearCuteDslCudaError() noexcept
{
    gCuteDslCudaError = cudaSuccess;
}

cudaError_t takeCuteDslCudaError() noexcept
{
    cudaError_t const error = gCuteDslCudaError;
    gCuteDslCudaError = cudaSuccess;
    return error;
}

namespace module_loader
{

std::mutex& getGlobalMutex()
{
    return *getGlobalMutexOwner();
}

std::shared_ptr<std::mutex> getGlobalMutexOwner()
{
    static auto const mutex = std::make_shared<std::mutex>();
    return mutex;
}

std::shared_ptr<void> acquireSharedModule(
    SharedModuleFactory factory, char const* moduleName, cudaStream_t stream) noexcept
{
    try
    {
        static std::mutex mutex;
        static std::unordered_map<SharedModuleFactory, std::shared_ptr<SharedModuleEntry>> cache;
        std::shared_ptr<SharedModuleEntry> entry;
        {
            std::lock_guard<std::mutex> const lock(mutex);
            auto& cachedEntry = cache[factory];
            if (!cachedEntry)
            {
                cachedEntry = std::make_shared<SharedModuleEntry>();
            }
            entry = cachedEntry;
        }
        std::unique_lock<std::mutex> entryLock(entry->mutex);
        auto module = entry->module.lock();
        // An expired weak pointer can precede completion of its deleter. Finish
        // unloading before reinitializing generated wrappers' static kernel handles.
        while (!module && entry->resident)
        {
            entry->released.wait(entryLock);
            module = entry->module.lock();
        }
        if (entry->poisoned)
        {
            logFailure(moduleName, "module cleanup failed; cache entry is poisoned");
            return {};
        }
        if (!module)
        {
            module = factory(entry, nonNullName(moduleName), stream);
            entry->module = module;
            entry->resident = static_cast<bool>(module);
        }
        return module;
    }
    catch (std::exception const& error)
    {
        logFailure(moduleName, error.what());
    }
    catch (...)
    {
        logFailure(moduleName, "unexpected shared-module acquisition failure");
    }
    return {};
}

void logLoaded(char const* moduleName) noexcept
{
    try
    {
        LOG_DEBUG("CuTe DSL kernel module loaded: %s", nonNullName(moduleName));
    }
    catch (...)
    {
    }
}

void logUnloaded(char const* moduleName) noexcept
{
    try
    {
        LOG_DEBUG("CuTe DSL kernel module unloaded: %s", nonNullName(moduleName));
    }
    catch (...)
    {
    }
}

void logFailure(char const* moduleName, char const* reason) noexcept
{
    try
    {
        LOG_ERROR("CuTe DSL kernel module '%s' failed: %s", nonNullName(moduleName),
            reason != nullptr ? reason : "unknown failure");
    }
    catch (...)
    {
    }
}

void logCudaFailure(char const* moduleName, char const* operation, cudaError_t error) noexcept
{
    try
    {
        LOG_ERROR("CuTe DSL kernel module '%s' failed to %s: %s (%s)", nonNullName(moduleName),
            operation != nullptr ? operation : "complete a CUDA operation", cudaGetErrorName(error),
            cudaGetErrorString(error));
    }
    catch (...)
    {
    }
}

#if defined(ENABLE_CUTEDSL_MODULE_TEST_HOOK)
bool shouldInjectFailure(char const* moduleName) noexcept
{
    char const* const requestedModule = std::getenv("TRT_EDGELLM_TEST_FAIL_CUTEDSL_MODULE");
    return requestedModule != nullptr && moduleName != nullptr && std::strcmp(requestedModule, moduleName) == 0;
}
#endif

} // namespace module_loader

} // namespace trt_edgellm::detail
