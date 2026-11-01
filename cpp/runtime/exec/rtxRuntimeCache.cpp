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

#include "runtime/exec/rtxRuntimeCache.h"

#include "common/logger.h"

#include <NvInferRuntime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string_view>
#include <system_error>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{
std::optional<nvinfer1::DynamicShapesKernelSpecializationStrategy> parseSpecializeStrategy(char const* value)
{
    std::string_view const sv{value};
    if (sv == "lazy")
    {
        return nvinfer1::DynamicShapesKernelSpecializationStrategy::kLAZY;
    }
    if (sv == "eager")
    {
        return nvinfer1::DynamicShapesKernelSpecializationStrategy::kEAGER;
    }
    if (sv == "none")
    {
        return nvinfer1::DynamicShapesKernelSpecializationStrategy::kNONE;
    }
    return std::nullopt;
}
} // namespace

std::filesystem::path RtxRuntimeCache::resolveCachePath(
    std::filesystem::path const& enginePath, std::optional<std::filesystem::path> const& cachePath)
{
    if (cachePath.has_value() && !cachePath->empty())
    {
        return *cachePath;
    }
    return enginePath.parent_path() / "runtime.rtxcache";
}

std::shared_ptr<RtxRuntimeCache> RtxRuntimeCache::create(
    std::filesystem::path const& enginePath, std::optional<std::filesystem::path> const& cachePath)
{
    char const* const cacheEnv = std::getenv("EDGELLM_RTX_RUNTIME_CACHE");
    char const* const strategyEnv = std::getenv("EDGELLM_RTX_SPECIALIZE");

    std::optional<std::filesystem::path> requestedPath = cachePath;
    if ((!requestedPath.has_value() || requestedPath->empty()) && cacheEnv != nullptr && *cacheEnv != '\0')
    {
        requestedPath = std::filesystem::path(cacheEnv);
    }
    return std::shared_ptr<RtxRuntimeCache>(
        new RtxRuntimeCache(resolveCachePath(enginePath, requestedPath), strategyEnv));
}

RtxRuntimeCache::RtxRuntimeCache(std::filesystem::path const& cachePath, char const* strategyEnv)
    : mCachePath(cachePath.string())
{
    auto strategy = nvinfer1::DynamicShapesKernelSpecializationStrategy::kLAZY;
    if (strategyEnv != nullptr)
    {
        if (auto const parsed = parseSpecializeStrategy(strategyEnv); parsed.has_value())
        {
            strategy = *parsed;
        }
        else
        {
            LOG_WARNING("ignoring unrecognized EDGELLM_RTX_SPECIALIZE='%s' (want lazy|eager|none)", strategyEnv);
        }
    }
    mSpecializationStrategy = static_cast<int32_t>(strategy);

    std::filesystem::path const parent = cachePath.parent_path();
    if (!parent.empty())
    {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec)
        {
            LOG_WARNING(
                "failed to create runtime cache directory %s: %s", parent.string().c_str(), ec.message().c_str());
        }
    }
}

std::unique_ptr<nvinfer1::IRuntimeConfig> RtxRuntimeCache::createConfig(nvinfer1::ICudaEngine& engine)
{
    auto config = std::unique_ptr<nvinfer1::IRuntimeConfig>(engine.createRuntimeConfig());
    if (config == nullptr)
    {
        LOG_WARNING("createRuntimeConfig failed; falling back to default execution context");
        return nullptr;
    }
    config->setExecutionContextAllocationStrategy(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);
    config->setDynamicShapesKernelSpecializationStrategy(
        static_cast<nvinfer1::DynamicShapesKernelSpecializationStrategy>(mSpecializationStrategy));

    if (mCache == nullptr)
    {
        mCache = std::unique_ptr<nvinfer1::IRuntimeCache>(config->createRuntimeCache());
        if (mCache == nullptr)
        {
            LOG_WARNING("createRuntimeCache failed; running without a persistent runtime cache");
            return config;
        }

        std::ifstream in(mCachePath, std::ios::binary);
        if (in)
        {
            std::vector<char> const blob{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
            if (!blob.empty() && mCache->deserialize(blob.data(), blob.size()))
            {
                LOG_INFO("loaded TRT-RTX runtime cache (%zu bytes) from %s", blob.size(), mCachePath.c_str());
            }
            else if (!blob.empty())
            {
                LOG_WARNING("TRT-RTX runtime cache at %s rejected (version/GPU/driver mismatch); rebuilding",
                    mCachePath.c_str());
            }
        }
    }
    if (!config->setRuntimeCache(*mCache))
    {
        LOG_WARNING("setRuntimeCache failed; running this engine without a persistent runtime cache");
    }
    return config;
}

RtxRuntimeCache::~RtxRuntimeCache()
{
    persist();
}

void RtxRuntimeCache::persist() const noexcept
{
    if (mCache == nullptr || mCachePath.empty())
    {
        return;
    }
    std::unique_ptr<nvinfer1::IHostMemory> const blob(mCache->serialize());
    if (blob == nullptr || blob->size() == 0)
    {
        return;
    }

    // Write to a per-process temporary file and rename it over the target so a crash or a second
    // runtime sharing the path can never observe a partial or interleaved blob. Replacement is
    // last-writer-wins, which is safe for a cache: a stale winner only costs a rebuild.
    std::string const tmpPath = mCachePath + ".tmp."
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "."
        + std::to_string(reinterpret_cast<std::uintptr_t>(this));

    {
        std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
        if (!out)
        {
            LOG_WARNING("failed to open temporary runtime cache file %s", tmpPath.c_str());
            return;
        }
        out.write(static_cast<char const*>(blob->data()), static_cast<std::streamsize>(blob->size()));
        out.flush();
        if (!out)
        {
            std::error_code rmEc;
            std::filesystem::remove(tmpPath, rmEc);
            LOG_WARNING("failed to write TRT-RTX runtime cache to %s", tmpPath.c_str());
            return;
        }
    }

    std::error_code ec;
    std::filesystem::rename(tmpPath, mCachePath, ec);
    if (ec)
    {
        std::error_code rmEc;
        std::filesystem::remove(tmpPath, rmEc);
        LOG_WARNING("failed to replace TRT-RTX runtime cache %s: %s", mCachePath.c_str(), ec.message().c_str());
        return;
    }
    LOG_INFO("persisted TRT-RTX runtime cache (%zu bytes) to %s", blob->size(), mCachePath.c_str());
}

} // namespace rt
} // namespace trt_edgellm
