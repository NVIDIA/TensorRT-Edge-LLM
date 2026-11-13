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

#include <NvInferSafePlugin.h>

#include <cstring>

namespace trt_edgellm::safety
{

// Bound traversal of the SDK-owned creator list and fixed registration working arrays.
inline constexpr int32_t kMAX_SAFETY_CREATORS{100};

//! Validate the interface before accessing Safety creator methods.
inline nvinfer2::safe::ISafePluginCreatorV3One* validateSafetyCreator(
    nvinfer1::IPluginCreatorInterface* creator, nvinfer2::safe::ISafeRecorder& recorder) noexcept
{
    using namespace nvinfer1;
    using namespace nvinfer2::safe;
    if (creator == nullptr)
    {
        recorder.reportError(ErrorCode::kINVALID_CONFIG, "Null creator entry in Safety plugin registry");
        return nullptr;
    }
    auto const info = creator->getInterfaceInfo();
    if (info.kind == nullptr || std::strcmp(info.kind, "PLUGIN SAFE CREATOR_V3ONE") != 0 || info.major != 1)
    {
        recorder.reportError(ErrorCode::kINVALID_CONFIG,
            "Unsupported Safety plugin creator interface: expected PLUGIN SAFE CREATOR_V3ONE major 1");
        return nullptr;
    }
    auto* safeCreator = static_cast<ISafePluginCreatorV3One*>(creator);
    if (safeCreator->getPluginName() == nullptr || safeCreator->getPluginVersion() == nullptr
        || safeCreator->getPluginNamespace() == nullptr)
    {
        recorder.reportError(ErrorCode::kINVALID_CONFIG, "Safety plugin creator has a null identity attribute");
        return nullptr;
    }
    return safeCreator;
}

//! Enumerate without treating a failed creator lookup as evidence of absence.
inline nvinfer1::ErrorCode getValidatedSafetyCreators(nvinfer2::safe::ISafePluginRegistry& registry,
    nvinfer2::safe::ISafeRecorder& recorder, nvinfer1::IPluginCreatorInterface* const*& creators,
    int32_t& count) noexcept
{
    using nvinfer1::ErrorCode;
    auto const status = registry.getAllCreators(creators, count);
    if (status != ErrorCode::kSUCCESS)
    {
        return status;
    }
    if (count < 0 || count > kMAX_SAFETY_CREATORS || (count != 0 && creators == nullptr))
    {
        recorder.reportError(ErrorCode::kINVALID_CONFIG,
            "Invalid Safety plugin creator list: expected count 0..100 and non-null storage when nonempty");
        return ErrorCode::kINVALID_CONFIG;
    }
    for (int32_t i = 0; i < count; ++i)
    {
        if (validateSafetyCreator(creators[i], recorder) == nullptr)
        {
            return ErrorCode::kINVALID_CONFIG;
        }
    }
    return ErrorCode::kSUCCESS;
}

//! Externally serialized initialization. The caller retains creators, their DSO, and the recorder.
//! newlyRegistered has room for selectedCount entries, including partial success on failure.
inline nvinfer1::ErrorCode registerSafetyCreators(nvinfer2::safe::ISafePluginRegistry& registry,
    nvinfer2::safe::ISafeRecorder& recorder, nvinfer2::safe::ISafePluginCreatorV3One* const* selected,
    int32_t selectedCount, nvinfer1::IPluginCreatorInterface** newlyRegistered, int32_t& newlyRegisteredCount) noexcept
{
    using namespace nvinfer1;
    using namespace nvinfer2::safe;
    newlyRegisteredCount = 0;
    if (selected == nullptr || newlyRegistered == nullptr || selectedCount <= 0 || selectedCount > kMAX_SAFETY_CREATORS)
    {
        recorder.reportError(ErrorCode::kINVALID_CONFIG, "Invalid required Safety plugin creator list");
        return ErrorCode::kINVALID_CONFIG;
    }
    for (int32_t role = 0; role < selectedCount; ++role)
    {
        if (validateSafetyCreator(selected[role], recorder) == nullptr)
        {
            return ErrorCode::kINVALID_CONFIG;
        }
    }
    IPluginCreatorInterface* const* creators{nullptr};
    int32_t count{};
    auto const status = getValidatedSafetyCreators(registry, recorder, creators, count);
    if (status != ErrorCode::kSUCCESS)
    {
        return status;
    }
    bool present[kMAX_SAFETY_CREATORS]{};
    for (int32_t i = 0; i < count; ++i)
    {
        auto* creator = static_cast<ISafePluginCreatorV3One*>(creators[i]);
        for (int32_t role = 0; role < selectedCount; ++role)
        {
            auto* expected = selected[role];
            if (std::strcmp(creator->getPluginName(), expected->getPluginName()) == 0
                && std::strcmp(creator->getPluginVersion(), expected->getPluginVersion()) == 0
                && std::strcmp(creator->getPluginNamespace(), expected->getPluginNamespace()) == 0)
            {
                if (creator != expected)
                {
                    recorder.reportError(ErrorCode::kINVALID_CONFIG,
                        "Safety plugin identity is already registered by a different creator instance");
                    return ErrorCode::kINVALID_CONFIG;
                }
                if (creator->getSafeRecorder() != &recorder)
                {
                    recorder.reportError(
                        ErrorCode::kINVALID_CONFIG, "Safety plugin creator is bound to a different registry recorder");
                    return ErrorCode::kINVALID_CONFIG;
                }
                present[role] = true;
            }
        }
    }
    for (int32_t role = 0; role < selectedCount; ++role)
    {
        if (!present[role])
        {
            selected[role]->setSafeRecorder(recorder);
            auto const registrationStatus
                = registry.registerCreator(*selected[role], selected[role]->getPluginNamespace(), recorder);
            if (registrationStatus != ErrorCode::kSUCCESS)
            {
                return registrationStatus;
            }
            newlyRegistered[newlyRegisteredCount++] = selected[role];
        }
    }
    return ErrorCode::kSUCCESS;
}

} // namespace trt_edgellm::safety
