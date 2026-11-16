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

#include <NvInfer.h>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <string_view>

namespace trt_edgellm::safety::builder
{
//! Write metadata and sidecars only inside the supplied staging directory; false aborts publication.
using SafetyMetadataWriter = std::function<bool(std::filesystem::path const&)>;

//! Save a nonempty engine/companion pair with staged metadata.
//! Stages writes before replacing existing files and rolls back publication failures.
//! The output directory must not be modified concurrently or consumed during publication.
//! Declared optional metadata filenames absent from staging are removed transactionally; other files are untouched.
bool saveSafetyEngineArtifacts(std::filesystem::path const& enginePath, std::string_view engine,
    std::string_view companion, SafetyMetadataWriter const& writeMetadata = {},
    std::initializer_list<std::string_view> optionalMetadata = {}) noexcept;

//! Build Safety artifacts with TensorRT 11.3 or later.
bool buildAndSerializeSafetyEngine(nvinfer1::IBuilder& builder, nvinfer1::INetworkDefinition& network,
    nvinfer1::IBuilderConfig& config, std::filesystem::path const& enginePath,
    SafetyMetadataWriter const& writeMetadata, std::initializer_list<std::string_view> optionalMetadata = {}) noexcept;
} // namespace trt_edgellm::safety::builder
