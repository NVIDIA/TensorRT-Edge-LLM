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

#include "safetyStatusUtils.h"

#include <cstdio>

namespace trt_edgellm::safety
{
namespace
{

void printStatus(char const* source, int32_t sourceCode, SafetyStatus const& status) noexcept
{
    auto const* message = status.message();
    (void) std::fprintf(stderr, "%s error %d -> SafetyStatusCode %d%s%s\n", source, sourceCode,
        static_cast<int32_t>(status.code()), message[0] == '\0' ? "" : ": ", message);
}

} // namespace

SafetyStatus fromTrtError(nvinfer2::safe::ErrorCode code, std::string_view errMsg, bool printError) noexcept
{
    using ErrorCode = nvinfer2::safe::ErrorCode;
    if (code == ErrorCode::kSUCCESS)
    {
        return SafetyStatus::ok();
    }

    auto mappedCode = SafetyStatusCode::kFailedExecution;
    switch (code)
    {
    case ErrorCode::kINVALID_ARGUMENT:
    case ErrorCode::kINVALID_CONFIG: mappedCode = SafetyStatusCode::kInvalidArgument; break;
    case ErrorCode::kINVALID_STATE:
    case ErrorCode::kUNSUPPORTED_STATE: mappedCode = SafetyStatusCode::kInvalidState; break;
    case ErrorCode::kFAILED_ALLOCATION: mappedCode = SafetyStatusCode::kFailedAllocation; break;
    case ErrorCode::kSUCCESS:
    case ErrorCode::kUNSPECIFIED_ERROR:
    case ErrorCode::kINTERNAL_ERROR:
    case ErrorCode::kFAILED_INITIALIZATION:
    case ErrorCode::kFAILED_EXECUTION:
    case ErrorCode::kFAILED_COMPUTATION:
    default: break;
    }

    // DLI requires all reported errors, including invalid arguments, to be nonrecoverable.
    auto const status = SafetyStatus::fatal(mappedCode, errMsg);
    if (printError)
    {
        printStatus("TRT", static_cast<int32_t>(code), status);
    }
    return status;
}

SafetyStatus fromCudaError(cudaError_t code, std::string_view errMsg, bool printError) noexcept
{
    if (code == cudaSuccess)
    {
        return SafetyStatus::ok();
    }

    auto const status = SafetyStatus::fatal(SafetyStatusCode::kCudaError, errMsg);
    if (printError)
    {
        printStatus("CUDA", static_cast<int32_t>(code), status);
    }
    return status;
}

SafetyStatus fromCudaDriverError(CUresult code, std::string_view errMsg, bool printError) noexcept
{
    if (code == CUDA_SUCCESS)
    {
        return SafetyStatus::ok();
    }
    auto const status = SafetyStatus::fatal(SafetyStatusCode::kCudaError, errMsg);
    if (printError)
    {
        printStatus("CUDA driver", static_cast<int32_t>(code), status);
    }
    return status;
}

} // namespace trt_edgellm::safety
