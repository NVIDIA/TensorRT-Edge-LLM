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

#include "common/checkMacros.h"
#include "pluginJitCubin.h"
#include "pluginJitKey.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace trt_edgellm
{

//! Base of every driver-loaded plugin JIT module: owns the CUmodule and remembers the context it
//! was loaded in and the kernel it came from. Plugins derive from it and add their CUfunction
//! handles, which PluginJitModuleRegistry::load fills through the bindFunctions callback.
template <typename Key>
struct PluginJitLoadedModule
{
    PluginJitLoadedModule() = default;
    PluginJitLoadedModule(PluginJitLoadedModule const&) = delete;
    PluginJitLoadedModule& operator=(PluginJitLoadedModule const&) = delete;

    ~PluginJitLoadedModule()
    {
        if (module != nullptr)
        {
            (void) cuModuleUnload(module);
        }
    }

    CUcontext context{};
    CUmodule module{};
    Key key{};
    PluginJitDigest digest{};
};

//! Process-wide registry of driver-loaded plugin JIT modules, keyed by (CUDA context, key, digest).
//! Entries are weak: a module stays loaded while at least one runner holds it and is loaded again on
//! the next request after the last holder is gone. `LoadedModule` derives from
//! PluginJitLoadedModule<Key>.
template <typename Key, typename LoadedModule>
class PluginJitModuleRegistry
{
public:
    //! \param name Error-message prefix, e.g. "NVFP4-A16 Blackwell GEMV JIT".
    explicit PluginJitModuleRegistry(char const* const name) noexcept
        : mName(name)
    {
    }

    //! Load `kernel` into the current CUDA context, or share the module already loaded there for the
    //! same key and digest. `bindFunctions(LoadedModule&)` resolves the entry points from
    //! `loaded.module`; if it throws, the module is unloaded and nothing is registered.
    //!
    //! Establishes the primary context when none is current, so this is not capture-safe: call it
    //! from configurePlugin / onShapeChange / deserialization, never from enqueue.
    //! \throws std::invalid_argument when the cubin is not a native ELF image or its digest does not
    //!         match, std::runtime_error when no CUDA context can be made current
    template <typename BindFunctions>
    std::shared_ptr<LoadedModule> load(PluginJitKernel<Key> const& kernel, BindFunctions&& bindFunctions)
    {
        if (!isNativeCubin(kernel.cubin.data(), kernel.cubin.size()))
        {
            throw std::invalid_argument(std::string(mName) + " cubin is not a native ELF image");
        }
        if (computePluginJitDigest(kernel.key, kernel.cubin.data(), kernel.cubin.size()) != kernel.digest)
        {
            throw std::invalid_argument(std::string(mName) + " cubin digest mismatch");
        }

        CUDA_CHECK(cudaFree(nullptr));
        CUcontext context{};
        CUDA_DRIVER_CHECK(cuCtxGetCurrent(&context));
        if (context == nullptr)
        {
            throw std::runtime_error(std::string(mName) + " module load requires a current CUDA context");
        }

        RegistryKey const registryKey{context, kernel.key, kernel.digest};
        std::lock_guard<std::mutex> const lock(mMutex);
        auto const existing = mModules.find(registryKey);
        if (existing != mModules.end())
        {
            if (auto loaded = existing->second.lock())
            {
                return loaded;
            }
            mModules.erase(existing);
        }

        auto loaded = std::make_shared<LoadedModule>();
        loaded->context = context;
        loaded->key = kernel.key;
        loaded->digest = kernel.digest;
        CUDA_DRIVER_CHECK(cuModuleLoadData(&loaded->module, kernel.cubin.data()));
        // A throwing bindFunctions unwinds through `loaded`, whose destructor unloads the module.
        bindFunctions(*loaded);
        mModules.emplace(registryKey, loaded);
        return loaded;
    }

private:
    struct RegistryKey
    {
        CUcontext context{};
        Key key{};
        PluginJitDigest digest{};

        bool operator==(RegistryKey const& other) const noexcept
        {
            return context == other.context && key == other.key && digest == other.digest;
        }
    };

    struct RegistryKeyHasher
    {
        size_t operator()(RegistryKey const& registryKey) const noexcept
        {
            size_t hash = mixPluginJitHash(static_cast<size_t>(kPLUGIN_JIT_FNV_OFFSET_BASIS),
                static_cast<size_t>(reinterpret_cast<uintptr_t>(registryKey.context)));
            hash = hashPluginJitWords(registryKey.key.keyWords(), hash);
            hash = mixPluginJitHash(hash, static_cast<size_t>(registryKey.digest.lo));
            return mixPluginJitHash(hash, static_cast<size_t>(registryKey.digest.hi));
        }
    };

    char const* mName;
    std::mutex mMutex;
    std::unordered_map<RegistryKey, std::weak_ptr<LoadedModule>, RegistryKeyHasher> mModules;
};

} // namespace trt_edgellm
