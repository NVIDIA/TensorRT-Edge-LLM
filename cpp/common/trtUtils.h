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

#pragma once

#include "logger.h"
#include "stringUtils.h"
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cstdlib>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

//! TRT-RTX reports NV_TENSORRT_MAJOR == 1; gate TRT-RTX-specific code paths on this. Usable in
//! both `#if` preprocessor conditionals and normal expressions.
#define IS_TRT_RTX (NV_TENSORRT_MAJOR == 1)

namespace trt_edgellm
{

#if defined(_WIN32)
inline constexpr char kDEFAULT_EDGELLM_PLUGIN_PATH[] = "NvInfer_edgellm_plugin.dll";

//! @brief Open a dynamic library and return its native handle.
inline void* openDynamicLibrary(char const* path) noexcept
{
    return static_cast<void*>(LoadLibraryA(path));
}

//! @brief Close a dynamic library handle.
inline void closeDynamicLibrary(void* handle) noexcept
{
    FreeLibrary(static_cast<HMODULE>(handle));
}

//! @brief Log the platform error from a failed dynamic library load.
inline void logDynamicLibraryOpenError(char const* path) noexcept
{
    LOG_ERROR("Cannot open plugin library %s (Windows error %lu)", path, static_cast<unsigned long>(GetLastError()));
}

//! @brief Resolve a symbol from a loaded dynamic library.
inline void* getDynamicLibrarySymbol(void* handle, char const* name) noexcept
{
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
}

//! @brief Return storage for the non-owning Edge-LLM plugin module handle.
inline void*& loadedEdgellmPluginHandle() noexcept
{
    static void* sHandle = nullptr;
    return sHandle;
}

//! @brief Resolve a symbol from the loaded TensorRT Edge-LLM plugin library.
inline void* getLoadedEdgellmPluginSymbol(char const* name) noexcept
{
    void* const handle = loadedEdgellmPluginHandle();
    return handle == nullptr ? nullptr : getDynamicLibrarySymbol(handle, name);
}
#else
inline constexpr char kDEFAULT_EDGELLM_PLUGIN_PATH[] = "build/libNvInfer_edgellm_plugin.so";

//! @brief Open a dynamic library and return its native handle.
inline void* openDynamicLibrary(char const* path) noexcept
{
    // RTLD_GLOBAL exposes plugin registration entry points to runtime resource
    // registration through dlsym(RTLD_DEFAULT, ...), avoiding build-time plugin
    // linkage and duplicate CUDA device code. RTLD_NODELETE keeps TensorRT plugin
    // code mapped through engine teardown.
    return dlopen(path, RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE);
}

//! @brief Close a dynamic library handle.
inline void closeDynamicLibrary(void* handle) noexcept
{
    dlclose(handle);
}

//! @brief Log the platform error from a failed dynamic library load.
inline void logDynamicLibraryOpenError(char const* path) noexcept
{
    LOG_ERROR("Cannot open plugin library %s: %s", path, dlerror());
}

//! @brief Resolve a symbol from a loaded dynamic library.
inline void* getDynamicLibrarySymbol(void* handle, char const* name) noexcept
{
    return dlsym(handle, name);
}

//! @brief Resolve a symbol from the loaded TensorRT Edge-LLM plugin library.
inline void* getLoadedEdgellmPluginSymbol(char const* name) noexcept
{
    return dlsym(RTLD_DEFAULT, name);
}
#endif

/*!
 * @brief Custom deleter for dynamic library handles.
 */
struct DlDeleter
{
    //! @brief Delete operator for library handles.
    void operator()(void* handle) const noexcept
    {
        if (handle)
        {
            closeDynamicLibrary(handle);
        }
    }
};

/*!
 * @brief Load TensorRT Edge-LLM plugin library.
 *
 * Loads the plugin library from EDGELLM_PLUGIN_PATH. If unset, uses the
 * platform-native plugin filename from the current working directory.
 *
 * @return Unique pointer to library handle, or nullptr on failure.
 */
inline std::unique_ptr<void, DlDeleter> loadEdgellmPluginLib(void) noexcept
{
    char const* pluginPath = std::getenv("EDGELLM_PLUGIN_PATH");

    if (pluginPath != nullptr)
    {
        LOG_INFO("EDGELLM_PLUGIN_PATH: %s", pluginPath);
    }
    else
    {
        pluginPath = kDEFAULT_EDGELLM_PLUGIN_PATH;
        LOG_INFO("EDGELLM_PLUGIN_PATH is not set. Defaulting to %s", pluginPath);
    }

    auto handle = std::unique_ptr<void, DlDeleter>(openDynamicLibrary(pluginPath));
    if (!handle)
    {
        logDynamicLibraryOpenError(pluginPath);
        return std::unique_ptr<void, DlDeleter>(nullptr);
    }
#if defined(_WIN32)
    loadedEdgellmPluginHandle() = handle.get();
#endif

    using InitPluginsFn = bool (*)(void*, char const*);
    auto initPlugins = reinterpret_cast<InitPluginsFn>(getDynamicLibrarySymbol(handle.get(), "initEdgellmPlugins"));
    if (initPlugins)
    {
        initPlugins(static_cast<nvinfer1::ILogger*>(&gLogger), "");
    }

    return handle;
}
//! Capture a TensorRT CUDA graph from an execution context and stream.
//! @return Pair of graph and graph exec on success, std::nullopt on failure
std::optional<std::pair<cudaGraph_t, cudaGraphExec_t>> captureTRTCudaGraph(
    nvinfer1::IExecutionContext* context, cudaStream_t stream);

//! RAII owner for the non-blocking auxiliary streams passed to
//! IExecutionContext::setAuxStreams(). Declare it before the context member so the
//! context is destroyed first (the streams must outlive it).
class AuxStreamSet
{
public:
    AuxStreamSet() = default;
    ~AuxStreamSet() noexcept
    {
        destroy();
    }
    AuxStreamSet(AuxStreamSet const&) = delete;
    AuxStreamSet& operator=(AuxStreamSet const&) = delete;
    AuxStreamSet(AuxStreamSet&& other) noexcept
        : mStreams(std::move(other.mStreams))
    {
        other.mStreams.clear();
    }
    AuxStreamSet& operator=(AuxStreamSet&& other) noexcept
    {
        if (this != &other)
        {
            destroy();
            mStreams = std::move(other.mStreams);
            other.mStreams.clear();
        }
        return *this;
    }

    void add(cudaStream_t stream)
    {
        mStreams.push_back(stream);
    }

    //! Number of streams currently held.
    size_t size() const noexcept
    {
        return mStreams.size();
    }

    //! Pointer to the contiguous stream storage, for passing to setAuxStreams().
    //! Only valid until the next add(); read it after all add() calls.
    cudaStream_t* data() noexcept
    {
        return mStreams.data();
    }

private:
    void destroy() noexcept
    {
        for (cudaStream_t stream : mStreams)
        {
            cudaStreamDestroy(stream);
        }
        mStreams.clear();
    }

    std::vector<cudaStream_t> mStreams;
};

//! Create non-blocking auxiliary streams for the context, register them via
//! IExecutionContext::setAuxStreams(), and append them to @p out (which owns them).
//! No-op when the engine reports zero aux streams.
void setNonBlockingAuxStreams(
    nvinfer1::IExecutionContext* context, nvinfer1::ICudaEngine const* engine, AuxStreamSet& out);

//! Convert TensorRT dimensions to a string representation.
std::string dimsToString(nvinfer1::Dims const& dims) noexcept;

//! Return true when a TensorRT dimensions object contains runtime dimensions.
bool hasDynamicDims(nvinfer1::Dims const& dims) noexcept;

//! Return true when two TensorRT dimensions objects have the same rank and extent.
bool dimsEqual(nvinfer1::Dims const& lhs, nvinfer1::Dims const& rhs) noexcept;

//! Return true when @p tensorName exists in the engine I/O list and is an input.
bool isEngineInput(nvinfer1::ICudaEngine const& engine, std::string const& tensorName) noexcept;

//! Print the engine information for a specific profile index.
std::string printEngineInfo(nvinfer1::ICudaEngine const* engine, int32_t profileIndex) noexcept;

//! Deserialize a TensorRT engine plan from disk without mapping the full plan into process memory.
std::unique_ptr<nvinfer1::ICudaEngine> deserializeCudaEngineFromFile(
    nvinfer1::IRuntime& runtime, std::filesystem::path const& enginePath);

//! Short, human-readable name for a TensorRT data type (e.g. "FLOAT16").
//! Used in logs and error messages; not intended for serialization.
constexpr char const* getDataTypeString(nvinfer1::DataType const dataType) noexcept
{
    switch (dataType)
    {
    case nvinfer1::DataType::kINT64: return "INT64";
    case nvinfer1::DataType::kINT32: return "INT32";
    case nvinfer1::DataType::kFLOAT: return "FLOAT32";
    case nvinfer1::DataType::kHALF: return "FLOAT16";
    case nvinfer1::DataType::kBF16: return "BFLOAT16";
    case nvinfer1::DataType::kFP8: return "FLOAT8_E4M3";
    case nvinfer1::DataType::kINT8: return "INT8";
    case nvinfer1::DataType::kUINT8: return "UINT8";
    case nvinfer1::DataType::kBOOL: return "BOOL";
#if IS_TRT_RTX || NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 12)
    case nvinfer1::DataType::kE8M0: return "FLOAT8_E8M0";
#endif
#if IS_TRT_RTX || NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 8)
    case nvinfer1::DataType::kFP4: return "FLOAT4";
#endif
    case nvinfer1::DataType::kINT4: return "INT4";
    }
    return "UNKNOWN";
}
//! Whether the engine exposes an output binding with the given name.
//! Use this instead of getTensorIOMode(name) when probing for an optional binding
//! — getTensorIOMode logs a spurious TensorRT ERROR for unknown names.
//! @pre engine and tensorName are non-null.
bool engineHasOutputTensor(nvinfer1::ICudaEngine const* engine, char const* tensorName) noexcept;

} // namespace trt_edgellm
