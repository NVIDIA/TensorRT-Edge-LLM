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

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace nvinfer1
{
class ICudaEngine;
class IRuntimeCache;
class IRuntimeConfig;
} // namespace nvinfer1

namespace trt_edgellm
{
namespace rt
{

//! @brief Manages one TRT-RTX runtime cache shared by a deployment's engines.
//!
//! Builds a separate @c IRuntimeConfig for each execution context and attaches the same @c IRuntimeCache
//! to every config. The cache is serialized back on destruction, so an @c RtxRuntimeCache must outlive
//! every execution context created from its configs.
class RtxRuntimeCache
{
public:
    //! @brief Create a deployment-scoped cache manager.
    //! @param enginePath on-disk engine path used to derive the default cache path.
    //! @param cachePath exact cache-file path. Overrides EDGELLM_RTX_RUNTIME_CACHE; when neither is specified,
    //!        defaults to @c <engine-directory>/runtime.rtxcache.
    static std::shared_ptr<RtxRuntimeCache> create(
        std::filesystem::path const& enginePath, std::optional<std::filesystem::path> const& cachePath);

    //! Serializes the cache to disk if one is loaded.
    ~RtxRuntimeCache();

    RtxRuntimeCache(RtxRuntimeCache const&) = delete;
    RtxRuntimeCache& operator=(RtxRuntimeCache const&) = delete;

    //! Build an engine-specific runtime config backed by the shared runtime cache.
    std::unique_ptr<nvinfer1::IRuntimeConfig> createConfig(nvinfer1::ICudaEngine& engine);

    //! Resolve an explicit cache-file path or derive one from the engine directory.
    static std::filesystem::path resolveCachePath(
        std::filesystem::path const& enginePath, std::optional<std::filesystem::path> const& cachePath);

private:
    RtxRuntimeCache(std::filesystem::path const& cachePath, char const* strategyEnv);

    void persist() const noexcept;

    std::unique_ptr<nvinfer1::IRuntimeCache> mCache;
    std::string mCachePath;
    int32_t mSpecializationStrategy{};
};

} // namespace rt
} // namespace trt_edgellm
