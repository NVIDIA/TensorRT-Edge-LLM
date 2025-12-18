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

#include "safetensorsUtils.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/mmapReader.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>

namespace trt_edgellm
{
namespace rt
{
namespace safetensors
{

namespace
{

// Helper function to convert TensorRT data type to safetensors dtype string
std::string dataTypeToString(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT: return "F32";
    case nvinfer1::DataType::kHALF: return "F16";
    case nvinfer1::DataType::kBF16: return "BF16";
    case nvinfer1::DataType::kINT8: return "I8";
    case nvinfer1::DataType::kUINT8: return "U8";
    case nvinfer1::DataType::kINT32: return "I32";
    case nvinfer1::DataType::kINT64: return "I64";
    case nvinfer1::DataType::kFP8: return "F8_E4M3";
    default: throw std::runtime_error("Unsupported data type for safetensors serialization");
    }
}

// Helper function to convert safetensors dtype string to TensorRT data type
nvinfer1::DataType stringToDataType(std::string const& dtype)
{
    if (dtype == "F32")
        return nvinfer1::DataType::kFLOAT;
    if (dtype == "F16")
        return nvinfer1::DataType::kHALF;
    if (dtype == "BF16")
        return nvinfer1::DataType::kBF16;
    if (dtype == "I8")
        return nvinfer1::DataType::kINT8;
    if (dtype == "U8")
        return nvinfer1::DataType::kUINT8;
    if (dtype == "I32")
        return nvinfer1::DataType::kINT32;
    if (dtype == "I64")
        return nvinfer1::DataType::kINT64;
    if (dtype == "F8_E4M3")
        return nvinfer1::DataType::kFP8;

    throw std::runtime_error("Unsupported data type: " + dtype);
}
} // namespace

bool saveSafetensors(std::filesystem::path const& filePath, std::vector<Tensor> const& tensors, cudaStream_t stream)
{
    if (tensors.empty())
    {
        LOG_ERROR("Cannot serialize empty tensor vector");
        return false;
    }

    // Create the JSON header
    nlohmann::json header;

    // Add metadata
    header["__metadata__"]
        = nlohmann::json::object({{"format", "PT"}, {"version", "1.0"}, {"provider", "tensorrt-edge-llm"}});

    // Calculate data offsets and build header
    size_t currentOffset = 0;
    std::set<std::string> tensorNames;
    for (auto const& tensor : tensors)
    {
        Coords shape = tensor.getShape();
        nvinfer1::DataType dataType = tensor.getDataType();
        std::string name = tensor.getName();

        if (name.empty())
        {
            LOG_ERROR("Tensor name is empty. Please set a name for each tensor for safetensors serialization.");
            return false;
        }
        if (tensorNames.find(name) != tensorNames.end())
        {
            LOG_ERROR(
                "Tensor name %s already exists. Please use a unique name for each tensor for safetensors "
                "serialization.",
                name.c_str());
            return false;
        }
        tensorNames.insert(name);

        // Convert shape to vector<size_t>
        std::vector<size_t> shapeVec;
        for (int32_t i = 0; i < shape.getNumDims(); ++i)
        {
            shapeVec.push_back(static_cast<size_t>(shape[i]));
        }

        // Calculate tensor size using existing getTypeSize function
        size_t tensorSize = tensor.getShape().volume() * rt::utils::getTypeSize(dataType);

        // Add tensor info to header
        header[name] = nlohmann::json::object({{"dtype", dataTypeToString(dataType)}, {"shape", shapeVec},
            {"data_offsets", nlohmann::json::array({currentOffset, currentOffset + tensorSize})}});

        currentOffset += tensorSize;
    }

    // Serialize header to string
    std::string headerStr = header.dump();

    // Calculate header size
    uint64_t headerSize = headerStr.size();

    // Open file for writing
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open())
    {
        LOG_ERROR("Failed to open file for writing: %s", filePath.string().c_str());
        return false;
    }

    // Write header size (8 bytes, little-endian)
    file.write(reinterpret_cast<char const*>(&headerSize), sizeof(headerSize));

    // Write header JSON
    file.write(headerStr.c_str(), headerSize);

    // Write tensor data
    for (auto const& tensor : tensors)
    {
        void const* data = tensor.rawPointer();
        size_t dataSize = tensor.getShape().volume() * rt::utils::getTypeSize(tensor.getDataType());

        // If tensor is on GPU, we need to copy to CPU first
        if (tensor.getDeviceType() == DeviceType::kGPU)
        {
            std::vector<uint8_t> cpuData(dataSize);
            CUDA_CHECK(cudaMemcpyAsync(cpuData.data(), data, dataSize, cudaMemcpyDeviceToHost, stream));
            file.write(reinterpret_cast<char const*>(cpuData.data()), dataSize);
        }
        else
        {
            // Tensor is already on CPU
            file.write(reinterpret_cast<char const*>(data), dataSize);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    file.close();
    return true;
}

bool loadSafetensors(std::filesystem::path const& filePath, std::vector<Tensor>& tensors, cudaStream_t stream)
{
    tensors.clear();

    // Read the file into memory
    std::unique_ptr<file_io::MmapReader> mmapReader;
    try
    {
        mmapReader = std::make_unique<file_io::MmapReader>(filePath.string());
    }
    catch (std::runtime_error const& e)
    {
        LOG_ERROR("Failed to open file: %s", filePath.string().c_str());
        return false;
    }

    // Read the header size (8 bytes)
    uint64_t headerSize = *reinterpret_cast<uint64_t const*>(mmapReader->getByteData());

    // Read the metadata JSON
    std::string metadataStr(reinterpret_cast<char const*>(mmapReader->getByteData() + sizeof(headerSize)), headerSize);

    // Parse the metadata
    nlohmann::json header;
    try
    {
        header = nlohmann::json::parse(metadataStr);
    }
    catch (nlohmann::json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse JSON metadata: %s", e.what());
        return false;
    }

    // Validate tensor entries and create tensors
    size_t tensorDataStart = sizeof(headerSize) + headerSize;

    for (auto const& [key, value] : header.items())
    {
        if (key == "__metadata__")
        {
            LOG_DEBUG("Loading SafeTensor Header, Metadata: %s", value.dump().c_str());
            continue;
        }

        // Validate tensor entry
        if (!value.is_object() || !value.contains("dtype") || !value["dtype"].is_string() || !value.contains("shape")
            || !value["shape"].is_array() || !value.contains("data_offsets") || !value["data_offsets"].is_array()
            || value["data_offsets"].size() != 2)
        {
            LOG_ERROR("Malformed tensor entry of SafeTensor object: %s : %s", key.c_str(), value.dump().c_str());
            return false;
        }

        // Extract tensor information
        std::string dtype = value["dtype"].get<std::string>();
        std::vector<size_t> shapeVec = value["shape"].get<std::vector<size_t>>();
        size_t dataStart = value["data_offsets"][0].get<size_t>();
        size_t dataEnd = value["data_offsets"][1].get<size_t>();
        size_t dataSize = dataEnd - dataStart;

        // Convert shape to Coords
        std::vector<int64_t> shapeInt64;
        for (size_t dim : shapeVec)
        {
            shapeInt64.push_back(static_cast<int64_t>(dim));
        }
        Coords shape(shapeInt64);

        // Convert dtype to TensorRT data type
        nvinfer1::DataType dataType;
        try
        {
            dataType = stringToDataType(dtype);
        }
        catch (std::runtime_error const& e)
        {
            LOG_ERROR("Unsupported data type: %s", dtype.c_str());
            return false;
        }

        // Create tensor with owned memory
        Tensor tensor(shape, DeviceType::kGPU, dataType, key);

        // Copy data from file to GPU
        int8_t const* tensorData = mmapReader->getByteData() + tensorDataStart + dataStart;
        CUDA_CHECK(cudaMemcpyAsync(tensor.rawPointer(), tensorData, dataSize, cudaMemcpyHostToDevice, stream));

        tensors.push_back(std::move(tensor));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    return true;
}

} // namespace safetensors
} // namespace rt
} // namespace trt_edgellm
