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

#include "tensor.h"
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace safetensors
{

//! Metadata for one tensor stored in a safetensors payload.
struct TensorMetadata
{
    std::string name;
    trt_edgellm::DataType dataType{};
    Coords shape;
    size_t offset{0}; //!< Byte offset relative to FileMetadata::dataOffset.
    size_t bytes{0};
};

//! Parsed safetensors header. Parsing touches only the fixed prefix and JSON
//! header; tensor payload pages remain demand-paged.
struct FileMetadata
{
    size_t dataOffset{0}; //!< Byte offset of the tensor payload in the file.
    std::vector<TensorMetadata> tensors;
    std::unordered_map<std::string, std::string> metadata;
};

//! Convert a safetensors dtype name to the corresponding TensorRT type.
trt_edgellm::DataType dataTypeFromString(std::string_view dtype);

//! Return the safetensors dtype name for a supported TensorRT type.
std::string dataTypeToString(trt_edgellm::DataType dtype);

//! Parse and validate a safetensors header from an existing file mapping.
FileMetadata parseMetadata(void const* data, size_t bytes, std::string_view source);

/*!
 * @brief Save tensors to a safetensors file
 *
 * Each tensor in the vector must have a unique name associated with it.
 *
 * @param filePath Path to output safetensors file
 * @param tensors Vector of tensors to save
 * @param stream CUDA stream for async operations
 * @param metadata Optional safetensors string metadata
 * @return True on success, false on failure
 * @throws std::runtime_error If CUDA operations fail or data type is unsupported
 */
bool saveSafetensors(std::filesystem::path const& filePath, std::vector<Tensor> const& tensors, cudaStream_t stream,
    std::unordered_map<std::string, std::string> const& metadata = {});

/*!
 * @brief Load tensors from a safetensors file
 *
 * Tensors are loaded into the provided vector. Each tensor owns its memory.
 *
 * @param filePath Path to input safetensors file
 * @param tensors Output vector to store loaded tensors
 * @param stream CUDA stream for async operations
 * @return True on success, false on failure
 * @throws std::runtime_error If CUDA operations fail, data type is unsupported, or tensor allocation fails
 */
bool loadSafetensors(std::filesystem::path const& filePath, std::vector<Tensor>& tensors, cudaStream_t stream,
    DeviceType device = DeviceType::kGPU);

} // namespace safetensors
} // namespace rt
} // namespace trt_edgellm
