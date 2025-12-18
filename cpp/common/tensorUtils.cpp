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

#include "tensor.h"

#include "checkMacros.h"
#if CUDA_VERSION >= 11080
#include <cuda_fp8.h>
#endif
#include <iomanip>
#include <memory>
#include <sstream>
#include <type_traits>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace rt
{
namespace utils
{

constexpr char const* getDataTypeString(DataType const dataType)
{
    switch (dataType)
    {
    case DataType::kINT64: return "INT64";
    case DataType::kINT32: return "INT32";
    case DataType::kFLOAT: return "FLOAT32";
    case DataType::kHALF: return "FLOAT16";
    case DataType::kBF16: return "BFLOAT16";
    case DataType::kFP8: return "FLOAT8_E4M3";
    case DataType::kINT8: return "INT8";
    case DataType::kUINT8: return "UINT8";
    }

    return "UNKNOWN";
}

constexpr char const* getDeviceTypeString(DeviceType deviceType)
{
    switch (deviceType)
    {
    case DeviceType::kCPU: return "CPU";
    case DeviceType::kGPU: return "GPU";
    }

    return "UNKNOWN";
}

// Helper function to acquire the string representation of a numeric value.
// With Float type, we always upcast the value to float32 for printing and emit
// the float value with 4 decimal places.
template <typename T>
std::string formatElement(T value)
{
    std::stringstream ss;
    if constexpr (std::is_integral_v<T>)
    {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            // stringstream treat int8_t and uint8_t as char, upcast to int32_t.
            ss << static_cast<int32_t>(value);
        }
        else
        {
            ss << value;
        }
    }
    else
    {
        // Convert floating point numbers to float at first.
        float fval{0.0F};
        if constexpr (std::is_same_v<T, float>)
        {
            fval = value;
        }
        else if constexpr (std::is_same_v<T, half>)
        {
            fval = __half2float(value);
        }
        else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        {
            fval = __bfloat162float(value);
        }
#if CUDA_VERSION >= 11080
        else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
        {
            // Use __nv_fp8_e4m3's "explicit operator float()" to convert to float.
            fval = static_cast<float>(value);
        }
#endif
        else
        {
            throw std::runtime_error("Unsupported data type for formatElement. Please extend the usage.");
        }
        ss << std::fixed << std::setprecision(4) << fval;
    }
    return ss.str();
}

template <typename T>
size_t getMaxFormatDataWidth(Coords const& shape, T const* data)
{
    size_t maxWidth = 0;
    for (int64_t i = 0; i < shape.volume(); ++i)
    {
        maxWidth = std::max(maxWidth, formatElement(data[i]).size());
    }
    return maxWidth;
}

template <typename T>
void buildStringRecursive(std::stringstream& ss, T const* data, Coords const& shape,
    std::array<int64_t, kMAX_DIMS> const& strides, size_t& offset, size_t dimension, size_t maxWidth,
    std::string const& indent)
{
    // Apply pytorch-style printing logic, for large tensors, we print the head and tail of the tensor
    // with intermediate value skipped. For "scalar dimension", we use threshold of 32 and print 10 elements at the
    // edge. For "vector dimension", we use threshold of 5 and print 2 elements at the edge. We hardcode the printing
    // threshold here which may need to be adjusted variables if needed.
    constexpr int32_t VECTOR_PRINT_THRESHOLD = 5;
    constexpr int32_t SCALAR_PRINT_THRESHOLD = 32;
    constexpr int32_t VECTOR_EDGE_ITEMS = 2;
    constexpr int32_t SCALAR_EDGE_ITEMS = 10;

    // With empty shape, print an empty array.
    if (shape.getNumDims() == 0)
    {
        ss << "[]";
        return;
    }

    check::check(dimension < shape.getNumDims() && dimension >= 0, "Dimension shall in range of the shape.");
    ss << "[";
    int64_t dimSize = shape[dimension];

    // Base case: Last dimension, print the numbers
    if (dimension == shape.getNumDims() - 1)
    {
        if (dimSize > SCALAR_PRINT_THRESHOLD)
        {
            for (int32_t i = 0; i < SCALAR_EDGE_ITEMS; ++i)
            {
                ss << std::setw(maxWidth) << formatElement(data[offset++]) << (i < SCALAR_EDGE_ITEMS - 1 ? ", " : "");
            }
            ss << ", ............, ";
            offset += dimSize - (2 * SCALAR_EDGE_ITEMS);
            for (int32_t i = 0; i < SCALAR_EDGE_ITEMS; ++i)
            {
                ss << std::setw(maxWidth) << formatElement(data[offset++]) << (i < SCALAR_EDGE_ITEMS - 1 ? ", " : "");
            }
        }
        else
        {
            for (int32_t i = 0; i < dimSize; ++i)
            {
                ss << std::setw(maxWidth) << formatElement(data[offset++]) << (i < dimSize - 1 ? ", " : "");
            }
        }
    }
    // Recursive step: Not the last dimension
    else
    {
        if (dimSize > VECTOR_PRINT_THRESHOLD)
        {
            for (int32_t i = 0; i < VECTOR_EDGE_ITEMS; ++i)
            {
                if (i > 0)
                {
                    ss << ",\n" << indent << " ";
                }
                buildStringRecursive(ss, data, shape, strides, offset, dimension + 1, maxWidth, indent + " ");
            }
            ss << ",\n" << indent << " ............\n" << indent << " ";
            offset += (dimSize - 2 * VECTOR_EDGE_ITEMS) * strides[dimension];
            for (int32_t i = 0; i < VECTOR_EDGE_ITEMS; ++i)
            {
                if (i > 0)
                {
                    ss << ",\n" << indent << " ";
                }
                buildStringRecursive(ss, data, shape, strides, offset, dimension + 1, maxWidth, indent + " ");
            }
        }
        else
        {
            for (size_t i = 0; i < dimSize; ++i)
            {
                if (i > 0)
                {
                    ss << ",\n" << indent << " ";
                }
                buildStringRecursive(ss, data, shape, strides, offset, dimension + 1, maxWidth, indent + " ");
            }
        }
    }
    ss << "]";
}

std::string formatString(Tensor const& tensor)
{
    void const* dataPtr{nullptr};
    auto const& shape = tensor.getShape();
    auto const& strides = utils::computeStrides(shape);
    auto const dataType = tensor.getDataType();

    // We may need a CPU tensor to host the device data for printing.
    std::unique_ptr<Tensor> cpuTensor{nullptr};
    if (tensor.getDeviceType() == DeviceType::kCPU)
    {
        dataPtr = tensor.rawPointer();
    }
    else
    {
        // Allocate a typed CPU tensor to hold the data on host for printing.
        // Please note "this" could be reshaped so the actutal memory capacity may differ from the cpuTensor.
        // Default stream is used because this is a debug only function
        cpuTensor = std::make_unique<Tensor>(shape, DeviceType::kCPU, tensor.getDataType(), tensor.getName());
        CUDA_CHECK(cudaMemcpy(
            cpuTensor->rawPointer(), tensor.rawPointer(), cpuTensor->getMemoryCapacity(), cudaMemcpyDeviceToHost));
        dataPtr = cpuTensor->rawPointer();
    }

    size_t offset = 0;
    size_t const startDim{0};
    std::string const startIndent{"       "};
    std::stringstream ss;
    ss << "\nTensor(";

    switch (dataType)
    {
    case DataType::kFLOAT:
    {
        size_t maxWidth = getMaxFormatDataWidth(shape, static_cast<float const*>(dataPtr));
        buildStringRecursive(
            ss, static_cast<float const*>(dataPtr), shape, strides, offset, startDim, maxWidth, startIndent);
        break;
    }
    case DataType::kHALF:
    {
        size_t maxWidth = getMaxFormatDataWidth(shape, static_cast<half const*>(dataPtr));
        buildStringRecursive(
            ss, static_cast<half const*>(dataPtr), shape, strides, offset, startDim, maxWidth, startIndent);
        break;
    }
    case DataType::kBF16:
    {
        size_t maxWidth = getMaxFormatDataWidth(shape, static_cast<__nv_bfloat16 const*>(dataPtr));
        buildStringRecursive(
            ss, static_cast<__nv_bfloat16 const*>(dataPtr), shape, strides, offset, startDim, maxWidth, startIndent);
        break;
    }
    case DataType::kINT64:
    {
        size_t maxWidth = getMaxFormatDataWidth(shape, static_cast<int64_t const*>(dataPtr));
        buildStringRecursive(
            ss, static_cast<int64_t const*>(dataPtr), shape, strides, offset, startDim, maxWidth, startIndent);
        break;
    }
    case DataType::kINT32:
    {
        size_t maxWidth = getMaxFormatDataWidth(shape, static_cast<int32_t const*>(dataPtr));
        buildStringRecursive(
            ss, static_cast<int32_t const*>(dataPtr), shape, strides, offset, startDim, maxWidth, startIndent);
        break;
    }
    case DataType::kINT8:
    {
        size_t maxWidth = getMaxFormatDataWidth(shape, static_cast<int8_t const*>(dataPtr));
        buildStringRecursive(
            ss, static_cast<int8_t const*>(dataPtr), shape, strides, offset, startDim, maxWidth, startIndent);
        break;
    }
    case DataType::kFP8:
    {
#if CUDA_VERSION < 11080
        throw std::runtime_error("FP8 data type is not supported with CUDA version < 11.8");
#else
        size_t maxWidth = getMaxFormatDataWidth(shape, static_cast<__nv_fp8_e4m3 const*>(dataPtr));
        buildStringRecursive(
            ss, static_cast<__nv_fp8_e4m3 const*>(dataPtr), shape, strides, offset, startDim, maxWidth, startIndent);
        break;
#endif
    }
    case DataType::kUINT8:
    {
        size_t maxWidth = getMaxFormatDataWidth(shape, static_cast<uint8_t const*>(dataPtr));
        buildStringRecursive(
            ss, static_cast<uint8_t const*>(dataPtr), shape, strides, offset, startDim, maxWidth, startIndent);
        break;
    }
    default:
    {
        throw std::runtime_error("Unsupported data type for formatString. Please extend the usage.");
    }
    }
    ss << ", shape=" << shape.formatString() << ", device=" << utils::getDeviceTypeString(tensor.getDeviceType())
       << ", dtype=" << utils::getDataTypeString(tensor.getDataType()) << ")";
    return ss.str();
}

double toKB(size_t bytes)
{
    return static_cast<double>(bytes) / 1024.0;
}

double toMB(size_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

double toGB(size_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

} // namespace utils
} // namespace rt
} // namespace trt_edgellm