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

#include "safetensorsUtils.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/mmapReader.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>

namespace trt_edgellm
{
namespace rt
{
namespace safetensors
{

//! @brief Helper function to convert a data type to safetensors dtype string
//! @param dataType Data type
//! @return Safetensors dtype string
//! @throws std::runtime_error If data type is unsupported
std::string dataTypeToString(trt_edgellm::DataType dataType)
{
    switch (dataType)
    {
    case trt_edgellm::DataType::kFLOAT: return "F32";
    case trt_edgellm::DataType::kHALF: return "F16";
    case trt_edgellm::DataType::kBF16: return "BF16";
    case trt_edgellm::DataType::kINT8: return "I8";
    case trt_edgellm::DataType::kUINT8: return "U8";
    case trt_edgellm::DataType::kBOOL: return "BOOL";
    case trt_edgellm::DataType::kINT32: return "I32";
    case trt_edgellm::DataType::kINT64: return "I64";
    case trt_edgellm::DataType::kFP8: return "F8_E4M3";
    default: throw std::runtime_error("Unsupported data type for safetensors serialization");
    }
}

namespace
{

size_t nonnegativeSize(nlohmann::json const& value, std::string const& label)
{
    uint64_t result{0};
    if (value.is_number_unsigned())
    {
        result = value.get<uint64_t>();
    }
    else
    {
        ELLM_CHECK(value.is_number_integer(), label + " must be an integer");
        int64_t const signedResult = value.get<int64_t>();
        ELLM_CHECK(signedResult >= 0, label + " must be nonnegative");
        result = static_cast<uint64_t>(signedResult);
    }
    ELLM_CHECK(result <= std::numeric_limits<size_t>::max(), label + " exceeds the host size limit");
    return static_cast<size_t>(result);
}
} // namespace

trt_edgellm::DataType dataTypeFromString(std::string_view dtype)
{
    if (dtype == "F32")
        return trt_edgellm::DataType::kFLOAT;
    if (dtype == "F16")
        return trt_edgellm::DataType::kHALF;
    if (dtype == "BF16")
        return trt_edgellm::DataType::kBF16;
    if (dtype == "I8")
        return trt_edgellm::DataType::kINT8;
    if (dtype == "U8")
        return trt_edgellm::DataType::kUINT8;
    if (dtype == "BOOL")
        return trt_edgellm::DataType::kBOOL;
    if (dtype == "I32")
        return trt_edgellm::DataType::kINT32;
    if (dtype == "I64")
        return trt_edgellm::DataType::kINT64;
    if (dtype == "F8_E4M3")
        return trt_edgellm::DataType::kFP8;

    throw std::runtime_error("Unsupported safetensors data type: " + std::string(dtype));
}

FileMetadata parseMetadata(void const* data, size_t bytes, std::string_view source)
{
    std::string const sourceName = source.empty() ? std::string{"safetensors mapping"} : std::string{source};
    ELLM_CHECK(data != nullptr, "Null " + sourceName);
    ELLM_CHECK(bytes >= sizeof(uint64_t), "Truncated safetensors file: " + sourceName);

    auto const* byteData = static_cast<int8_t const*>(data);
    uint64_t headerBytesRaw{0};
    std::memcpy(&headerBytesRaw, byteData, sizeof(headerBytesRaw));
    ELLM_CHECK(headerBytesRaw <= bytes - sizeof(headerBytesRaw), "Invalid safetensors header size: " + sourceName);
    size_t const headerBytes = static_cast<size_t>(headerBytesRaw);

    FileMetadata result;
    result.dataOffset = sizeof(headerBytesRaw) + headerBytes;
    std::string const headerText(reinterpret_cast<char const*>(byteData + sizeof(headerBytesRaw)), headerBytes);
    nlohmann::json const header = nlohmann::json::parse(headerText);
    result.tensors.reserve(header.size());
    for (auto const& [name, value] : header.items())
    {
        if (name == "__metadata__")
        {
            ELLM_CHECK(value.is_object(), "Malformed __metadata__ in " + sourceName);
            for (auto const& [metadataName, metadataValue] : value.items())
            {
                ELLM_CHECK(metadataValue.is_string(),
                    "Safetensors metadata value must be a string for " + metadataName + " in " + sourceName);
                result.metadata.emplace(metadataName, metadataValue.get<std::string>());
            }
            continue;
        }
        ELLM_CHECK(value.is_object() && value.contains("dtype") && value["dtype"].is_string() && value.contains("shape")
                && value["shape"].is_array() && value.contains("data_offsets") && value["data_offsets"].is_array()
                && value["data_offsets"].size() == 2,
            "Malformed tensor entry " + name + " in " + sourceName);

        std::vector<int64_t> dimensions;
        dimensions.reserve(value["shape"].size());
        for (auto const& dimension : value["shape"])
        {
            uint64_t dimensionValue{0};
            if (dimension.is_number_unsigned())
            {
                dimensionValue = dimension.get<uint64_t>();
            }
            else
            {
                ELLM_CHECK(dimension.is_number_integer(),
                    "Tensor dimension must be an integer for " + name + " in " + sourceName);
                int64_t const signedDimension = dimension.get<int64_t>();
                ELLM_CHECK(
                    signedDimension >= 0, "Tensor dimension must be nonnegative for " + name + " in " + sourceName);
                dimensionValue = static_cast<uint64_t>(signedDimension);
            }
            ELLM_CHECK(dimensionValue <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "Tensor dimension exceeds INT64 for " + name + " in " + sourceName);
            dimensions.push_back(static_cast<int64_t>(dimensionValue));
        }

        size_t const begin = nonnegativeSize(value["data_offsets"][0], "Tensor data offset");
        size_t const end = nonnegativeSize(value["data_offsets"][1], "Tensor data offset");
        ELLM_CHECK(begin <= end && end <= bytes - result.dataOffset,
            "Invalid tensor offsets for " + name + " in " + sourceName);
        result.tensors.push_back(TensorMetadata{
            name,
            dataTypeFromString(value["dtype"].get_ref<std::string const&>()),
            Coords(dimensions),
            begin,
            end - begin,
        });
    }
    return result;
}

bool saveSafetensors(std::filesystem::path const& filePath, std::vector<Tensor> const& tensors, cudaStream_t stream,
    std::unordered_map<std::string, std::string> const& metadata)
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
    for (auto const& [name, value] : metadata)
    {
        header["__metadata__"][name] = value;
    }

    // Calculate data offsets and build header
    size_t currentOffset = 0;
    std::set<std::string> tensorNames;
    for (auto const& tensor : tensors)
    {
        Coords shape = tensor.getShape();
        trt_edgellm::DataType dataType = tensor.getDataType();
        std::string const& name = tensor.getName();

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

bool loadSafetensors(
    std::filesystem::path const& filePath, std::vector<Tensor>& tensors, cudaStream_t stream, DeviceType device)
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

    FileMetadata metadata;
    try
    {
        metadata = parseMetadata(mmapReader->getData(), mmapReader->getSize(), filePath.string());
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to parse safetensors metadata: %s", e.what());
        return false;
    }
    for (TensorMetadata const& entry : metadata.tensors)
    {
        // Create tensor with owned memory
        Tensor tensor(entry.shape, device, entry.dataType, entry.name);
        ELLM_CHECK(static_cast<size_t>(tensor.getMemoryCapacity()) == entry.bytes,
            "Safetensors byte count does not match tensor metadata: " + entry.name);

        // Copy data from file to GPU
        int8_t const* tensorData = mmapReader->getByteData() + metadata.dataOffset + entry.offset;
        CUDA_CHECK(cudaMemcpyAsync(tensor.rawPointer(), tensorData, entry.bytes,
            device == DeviceType::kGPU ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost, stream));

        tensors.push_back(std::move(tensor));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    return true;
}

} // namespace safetensors
} // namespace rt
} // namespace trt_edgellm
