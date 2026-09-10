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

#include <NvInferRuntime.h>

#include <cstdint>

namespace trt_edgellm
{

//! Compile and load an XQA JIT kernel variant for direct runner unit tests.
bool loadXQAJitKernelForTest(int32_t smVersion, nvinfer1::DataType dataType, nvinfer1::DataType kvDataType,
    int32_t headSize, int32_t numQHeads, int32_t numKVHeads, bool slidingWindow, bool specDecode,
    int32_t tokensPerPage = 0);

} // namespace trt_edgellm
