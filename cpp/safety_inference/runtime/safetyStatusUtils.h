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

#include "safetyStatus.h"

#include <NvInferSafeRuntime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace trt_edgellm::safety
{

//! Converts a TRT Safety result; every reported error is fatal.
//! errMsg is copied into bounded status storage. printError enables optional stderr diagnostics.
SafetyStatus fromTrtError(
    nvinfer2::safe::ErrorCode code, std::string_view errMsg = {}, bool printError = false) noexcept;

//! Converts a CUDA runtime result; every reported error is fatal.
//! errMsg may be empty. Error conversion and printing make no CUDA API calls.
SafetyStatus fromCudaError(cudaError_t code, std::string_view errMsg = {}, bool printError = false) noexcept;

//! Converts a CUDA driver result without calling the driver during error containment.
SafetyStatus fromCudaDriverError(CUresult code, std::string_view errMsg = {}, bool printError = false) noexcept;

} // namespace trt_edgellm::safety
