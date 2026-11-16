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

#include "safetyGraphInitialization.h"

namespace trt_edgellm
{
namespace safety
{
namespace
{

void forwardRegistrationFailure(nvinfer2::safe::ISafeRecorder& registryRecorder, int32_t previousErrorCount,
    nvinfer1::ErrorCode fallbackCode, char const* fallbackDescription,
    nvinfer2::safe::ISafeRecorder& graphRecorder) noexcept
{
    auto const currentErrorCount = registryRecorder.getNbErrors();
    if (currentErrorCount > previousErrorCount)
    {
        auto const errorIndex = currentErrorCount - 1;
        graphRecorder.reportError(registryRecorder.getErrorCode(errorIndex), registryRecorder.getErrorDesc(errorIndex));
        return;
    }
    graphRecorder.reportError(fallbackCode, fallbackDescription);
}

} // namespace

nvinfer1::ErrorCode createSafetyGraph(SafetyAttentionModules& modules, nvinfer2::safe::ITRTGraph*& graph,
    void const* engine, int64_t engineBytes, nvinfer2::safe::ISafeRecorder& graphRecorder,
    char const* companionSoPath) noexcept
{
    using nvinfer1::ErrorCode;
    if (graph != nullptr || engine == nullptr || engineBytes <= 0)
    {
        graphRecorder.reportError(ErrorCode::kINVALID_ARGUMENT, "Invalid Safety graph initialization arguments");
        return ErrorCode::kINVALID_ARGUMENT;
    }
    if (companionSoPath == nullptr || companionSoPath[0] != '/')
    {
        graphRecorder.reportError(
            ErrorCode::kINVALID_ARGUMENT, "Safety graph requires an absolute companion library path");
        return ErrorCode::kINVALID_ARGUMENT;
    }
    auto& registryRecorder = processRegistrationRecorder();
    auto const previousErrorCount = registryRecorder.getNbErrors();
    auto* registry = nvinfer2::safe::getSafePluginRegistry(registryRecorder);
    if (registry == nullptr)
    {
        forwardRegistrationFailure(registryRecorder, previousErrorCount, ErrorCode::kUNSPECIFIED_ERROR,
            "TensorRT Safety plugin registry is unavailable", graphRecorder);
        return ErrorCode::kUNSPECIFIED_ERROR;
    }
    auto const status = registerSafetyPlugins(*registry);
    if (status != ErrorCode::kSUCCESS)
    {
        forwardRegistrationFailure(
            registryRecorder, previousErrorCount, status, "Safety plugin registration failed", graphRecorder);
        return status;
    }
    SafetyAttentionModuleScope moduleScope(modules);
    return nvinfer2::safe::createTRTGraph(graph, engine, engineBytes, companionSoPath, graphRecorder, true, nullptr);
}
} // namespace safety
} // namespace trt_edgellm
