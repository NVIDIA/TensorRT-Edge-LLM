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

#include "builder/safety/safetyBuilderConfig.h"

#include <NvInfer.h>
#include <string_view>

namespace trt_edgellm::safety::builder
{
//! Resolve a batched, contiguous-KV input shape and check its exported rank and fixed dimensions.
//! Throws std::invalid_argument for unsupported inputs or incompatible model/profile dimensions.
nvinfer1::Dims getSafetyLLMInputShape(std::string_view name, nvinfer1::Dims const& exportedShape,
    SafetyLLMStaticShapeProfile const& profile, int64_t hiddenSize, int64_t numKVHeads, int64_t headSize,
    int64_t rotaryDim);
} // namespace trt_edgellm::safety::builder
