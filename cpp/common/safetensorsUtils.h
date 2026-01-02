/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensor.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace safetensors
{

/*!
 * @brief Save tensors to a safetensors file
 *
 * Each tensor in the vector must have a unique name associated with it.
 *
 * @param filePath Path to output safetensors file
 * @param tensors Vector of tensors to save
 * @param stream CUDA stream for async operations
 * @return True on success, false on failure
 */
bool saveSafetensors(std::filesystem::path const& filePath, std::vector<Tensor> const& tensors, cudaStream_t stream);

/*!
 * @brief Load tensors from a safetensors file
 *
 * Tensors are loaded into the provided vector. Each tensor owns its memory.
 *
 * @param filePath Path to input safetensors file
 * @param tensors Output vector to store loaded tensors
 * @param stream CUDA stream for async operations
 * @return True on success, false on failure
 */
bool loadSafetensors(std::filesystem::path const& filePath, std::vector<Tensor>& tensors, cudaStream_t stream);

} // namespace safetensors
} // namespace rt
} // namespace trt_edgellm
