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

#include "common/tensor.h"

#include <NvInferRuntime.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

namespace trt_edgellm
{
namespace plugins
{
//! Device memory alignment requirement (128 bytes)
constexpr int32_t kDEVICE_ALIGNMENT{128};

//! @brief Align device pointer to device alignment.
//! @param ptr Device pointer that might not be aligned to device alignment.
//! @return Aligned device pointer
void* alignDevicePtr(void* ptr);

/*!
 * @brief Convert C++ type to TensorRT PluginFieldType
 * @tparam T C++ type
 * @return Corresponding PluginFieldType
 */
template <typename T>
nvinfer1::PluginFieldType toFieldType();

//! @cond INTERNAL
#define SPECIALIZE_TO_FIELD_TYPE(T, type)                                                                              \
    template <>                                                                                                        \
    inline nvinfer1::PluginFieldType toFieldType<T>()                                                                  \
    {                                                                                                                  \
        return nvinfer1::PluginFieldType::type;                                                                        \
    }
SPECIALIZE_TO_FIELD_TYPE(float, kFLOAT32)
SPECIALIZE_TO_FIELD_TYPE(int32_t, kINT32)
#undef SPECIALIZE_TO_FIELD_TYPE
//! @endcond

/*!
 * @brief Parse scalar field from plugin field collection
 *
 * Extracts a scalar value from TensorRT plugin field collection.
 *
 * @tparam T Field data type
 * @param fieldName Name of field to parse
 * @param fc Plugin field collection
 * @return Optional containing value if found, nullopt otherwise
 */
template <typename T>
inline std::optional<T> parsePluginScalarField(std::string const& fieldName, nvinfer1::PluginFieldCollection const* fc)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        nvinfer1::PluginField const& pluginField = fc->fields[i];
        if (fieldName.compare(pluginField.name) == 0)
        {
            assert(toFieldType<T>() == pluginField.type && "Mismatch datatype of plugin field");
            assert(pluginField.length == 1 && pluginField.data != nullptr && "Invalid plugin field");
            return std::optional{*static_cast<T const*>(pluginField.data)};
        }
    }

    return std::nullopt;
}

/*!
 * @brief Generic serializer template for plugin data
 *
 * Provides serialization/deserialization for plugin state.
 * Specialized for arithmetic and enum types.
 *
 * @tparam T Type to serialize
 * @tparam Enable SFINAE enabler
 */
template <typename T, class Enable = void>
struct Serializer
{
};

//! @brief Serializer specialization for arithmetic and enum types
template <typename T>
struct Serializer<T, typename std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>>>
{
    //! @brief Serialize value to buffer
    //! @param buffer Output buffer pointer (advanced after write)
    //! @param value Value to serialize
    static void serialize(void** buffer, T const& value)
    {
        ::memcpy(*buffer, &value, sizeof(T));
        reinterpret_cast<char*&>(*buffer) += sizeof(T);
    }

    //! @brief Deserialize value from buffer
    //! @param buffer Input buffer pointer (advanced after read)
    //! @param buffer_size Buffer size (decremented after read)
    //! @param value Output value
    static void deserialize(void const** buffer, size_t* buffer_size, T* value)
    {
        assert(*buffer_size >= sizeof(T));
        ::memcpy(value, *buffer, sizeof(T));
        reinterpret_cast<char const*&>(*buffer) += sizeof(T);
        *buffer_size -= sizeof(T);
    }
};

/*!
 * @brief Serialize a value to buffer
 * @tparam T Type to serialize
 * @param buffer Output buffer
 * @param value Value to serialize
 */
template <typename T>
inline void serializeValue(void** buffer, T const& value)
{
    return Serializer<T>::serialize(buffer, value);
}

/*!
 * @brief Deserialize a value from buffer
 * @tparam T Type to deserialize
 * @param buffer Input buffer
 * @param buffer_size Buffer size
 * @param value Output value
 */
template <typename T>
inline void deserializeValue(void const** buffer, size_t* buffer_size, T* value)
{
    return Serializer<T>::deserialize(buffer, buffer_size, value);
}

//! @brief Accumulate workspace size for a given shape and data type. Device alignment will be applied automatically.
//! @param currentSize Current workspace size
//! @param shape Tensor shape
//! @param dataType Tensor data type
//! @return Accumulated workspace size that aligned to device alignment.
size_t accumulateWorkspaceSize(size_t currentSize, rt::Coords const& shape, nvinfer1::DataType dataType);

//! @brief Given a contiguous workspace, assign (non-owned) tensor with specified shape and data type from the start of
//! workspace. After assignment, the workspace pointer will shift the size of tensor and align to device alignment.
//! @param workspace The contiguous workspace pointer
//! @param shape Requested Tensor shape
//! @param dataType Requested Tensor data type
//! @return Assigned tensor
rt::Tensor assignTensorFromWorkspace(void*& workspace, rt::Coords const& shape, nvinfer1::DataType dataType);

} // namespace plugins
} // namespace trt_edgellm