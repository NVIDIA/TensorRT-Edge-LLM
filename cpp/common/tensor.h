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

#include "common/checkMacros.h"
#include "cudaMacros.h"
#include <NvInferRuntime.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <functional>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Device type enumeration
enum class DeviceType
{
    kCPU = 0, //!< CPU device
    kGPU = 1, //!< GPU device
};

//! Maximum number of dimensions supported for tensors
constexpr int32_t kMAX_DIMS = 8;

/*!
 * @brief Extended arithmetic type trait
 *
 * Extends std::is_arithmetic to support fp16 and bfloat16 types for convenience.
 *
 * @tparam T Type to check
 */
template <typename T>
struct is_arithmetic_ext : std::is_arithmetic<T>
{
};

//! @brief Specialization for half precision floating point
template <>
struct is_arithmetic_ext<half> : std::true_type
{
};

//! @brief Specialization for bfloat16 precision floating point
template <>
struct is_arithmetic_ext<__nv_bfloat16> : std::true_type
{
};

#if SUPPORTS_FP8
//! @brief Specialization for fp8 precision floating point
template <>
struct is_arithmetic_ext<__nv_fp8_e4m3> : std::true_type
{
};
#endif

//! Array of dimensions that used to store the shape of a tensor.
//! Support up to 8 dimensions with all dimensions are non-negative.
//! Default constructor create empty coords with 0 dimensions and 0 volumes.
class Coords
{
public:
    //! @brief Default constructor
    Coords() = default;

    //! @brief Copy constructor
    //! @param coords Coordinates to copy
    Coords(Coords const& coords) noexcept
        : mDims(coords.mDims)
        , mNumDims(coords.mNumDims)
    {
    }

    //! @brief Move constructor
    Coords(Coords&& coords) noexcept = default;

    //! @brief Copy assignment operator
    //! @param coords Coordinates to copy
    //! @return Reference to this
    Coords& operator=(Coords const& coords) noexcept = default;

    //! @brief Move assignment operator
    //! @param coords Coordinates to move
    //! @return Reference to this
    Coords& operator=(Coords&& coords) noexcept = default;

    //! @brief Construct from TensorRT Dims
    //! @param dims TensorRT dimensions
    Coords(nvinfer1::Dims const& dims) noexcept
        : mNumDims(dims.nbDims)
    {
        std::copy(dims.d, dims.d + mNumDims, mDims.begin());
    }

    //! @brief Equality comparison operator
    //! @param other Coords object to compare with
    //! @return True if both Coords have the same number of dimensions
    //!         and all corresponding dimension values are equal, false otherwise
    bool operator==(Coords const& other) const noexcept;

    //! @brief Inequality comparison operator
    //! @param other Coords object to compare with
    //! @return True if the Coords are not equal (i.e., operator== returns false), false otherwise
    bool operator!=(Coords const& other) const noexcept;

    /*!
     * @brief Construct from iterator range
     * @tparam IT Iterator type
     * @param begin Beginning of range
     * @param end End of range
     * @throws std::runtime_error if number of dimensions exceeds kMAX_DIMS
     */
    template <typename IT>
    Coords(IT begin, IT end)
        : mNumDims(std::distance(begin, end))
    {
        ELLM_CHECK(mNumDims <= kMAX_DIMS, "Coords: number of dimensions out of range");
        std::copy(begin, end, mDims.begin());
    }

    //! @brief Construct from initializer list
    //! @param init Initializer list of dimensions
    //! @throws std::runtime_error if number of dimensions exceeds kMAX_DIMS
    Coords(std::initializer_list<int64_t> init)
        : Coords(init.begin(), init.end())
    {
    }

    //! @brief Construct from vector
    //! @param vec Vector of dimensions
    //! @throws std::runtime_error if number of dimensions exceeds kMAX_DIMS
    Coords(std::vector<int64_t> const& vec)
        : Coords(vec.begin(), vec.end())
    {
    }

    //! @brief Get number of dimensions
    //! @return Number of dimensions
    int32_t getNumDims() const noexcept
    {
        return mNumDims;
    }

    /*!
     * @brief Array subscript operator (mutable)
     * @param idx Index to access
     * @return Reference to dimension at index
     * @throws std::out_of_range if index is out of bounds
     */
    int64_t& operator[](int32_t idx)
    {
        if (idx < 0 || idx >= mNumDims)
        {
            throw std::out_of_range("Coords: index out of range");
        }
        return mDims[idx];
    }

    /*!
     * @brief Array subscript operator (const)
     * @param idx Index to access
     * @return Dimension value at index
     * @throws std::out_of_range if index is out of bounds
     */
    int64_t operator[](int32_t idx) const
    {
        if (idx < 0 || idx >= mNumDims)
        {
            throw std::out_of_range("Coords: index out of range");
        }
        return mDims[idx];
    }

    //! @brief Calculate total volume (product of all dimensions)
    //! @return Volume of the coordinates
    int64_t volume() const noexcept;

    //! @brief Convert to TensorRT Dims
    //! @return TensorRT dimensions
    nvinfer1::Dims getTRTDims() const noexcept;

    //! @brief Format coordinates as string
    //! @return String representation of coordinates
    //! @throws std::runtime_error If coordinates are invalid or cannot be formatted
    std::string formatString() const;

private:
    std::array<int64_t, kMAX_DIMS> mDims{};
    int32_t mNumDims{0};
};

//! Tensor class that wrap linear layout tensor.
//! The underlying memory can either be owned by the tensor object or be reused from another allocation.
//! The Tensor Object support reshapes when memory is owned by the object and has sufficient capacity.
//! The default constructor creates an empty tensor object with zero volume with no underlying memory.
class Tensor
{
public:
    //! @brief Default constructor
    Tensor() noexcept = default;

    /*!
     * @brief Disable copy constructor to enforce explicit memory ownership transfer
     *
     * Non-owned tensor object can be constructed explicitly to workaround
     * the limitation of deleted copy constructor.
     */
    Tensor(Tensor const& other) = delete;

    /*!
     * @brief Disable copy assignment operator to enforce explicit memory ownership transfer
     * @return Reference to this
     */
    Tensor& operator=(Tensor const& other) = delete;

    //! @brief Move constructor allows transfer of memory ownership
    //! @param other Tensor to move from
    Tensor(Tensor&& other) noexcept;

    //! @brief Move assignment operator allows transfer of memory ownership
    //! @param other Tensor to move from
    //! @return Reference to this
    Tensor& operator=(Tensor&& other) noexcept;

    //! @brief Destructor
    ~Tensor() noexcept;

    /*!
     * @brief Constructor that allocates memory on the specified device
     *
     * The memory is owned by the tensor object and will be freed when the
     * tensor object is destroyed.
     *
     * @param extent The shape of the tensor (must have non-zero volume)
     * @param deviceType The device type to allocate memory on
     * @param dataType The data type of the tensor (sub-types like kInt4 or kE2M1 are not supported)
     * @param name Optional name for the tensor
     * @throws std::runtime_error If volume is zero, data type is unsupported, or CUDA allocation fails
     */
    Tensor(Coords const& extent, DeviceType deviceType, nvinfer1::DataType dataType, std::string const& name = "");

    /*!
     * @brief Constructor that reuses external memory
     *
     * Memory is not owned by the tensor object. The caller must ensure the
     * lifecycle of the memory.
     *
     * @param data Pointer to existing memory
     * @param extent The shape of the tensor
     * @param deviceType The device type of the memory
     * @param dataType The data type of the tensor
     * @param name Optional name for the tensor
     */
    Tensor(void* data, Coords const& extent, DeviceType deviceType, nvinfer1::DataType dataType,
        std::string const& name = "");

    //! @brief Get the shape of the tensor
    //! @return Coordinates representing the tensor shape
    Coords getShape() const noexcept;

    //! @brief Get the device type
    //! @return Device type (CPU or GPU)
    DeviceType getDeviceType() const noexcept;

    //! @brief Get the data type
    //! @return TensorRT data type
    nvinfer1::DataType getDataType() const noexcept;

    //! @brief Get TensorRT dimensions
    //! @return TensorRT Dims object
    nvinfer1::Dims getTRTDims() const noexcept;

    //! @brief Check if tensor owns its memory
    //! @return True if memory is owned, false otherwise
    bool getOwnMemory() const noexcept;

    //! @brief Check if tensor is empty
    //! @return True if tensor is empty, false otherwise
    bool isEmpty() const noexcept;

    //! @brief Get the name of the tensor
    //! @return Tensor name
    std::string const& getName() const noexcept;

    /*!
     * @brief Get memory capacity of the underlying buffer
     *
     * Returns the memory capacity when the instance was constructed.
     * The value can differ from getShape().volume() * sizeof(dataType) when
     * the tensor is reshaped.
     *
     * @return Memory capacity in bytes
     */
    int64_t getMemoryCapacity() const noexcept;

    /*!
     * @brief Bytes covered by the current shape (not the allocated capacity).
     *
     * Prefer this over getMemoryCapacity() when zeroing or copying only the live
     * view after reshape — capacity can be hundreds of MiB for spec-decode
     * hidden-state scratch allocated at max input length.
     */
    int64_t getActiveBytes() const noexcept;

    //! @brief Get stride of the tensor at given dimension
    //! @param idx Dimension index
    //! @return Stride value
    //! @throws std::out_of_range If idx is out of bounds
    [[nodiscard]] int64_t getStride(int32_t idx) const;

    //! @brief Get raw data pointer (const)
    //! @return Const pointer to tensor data
    void const* rawPointer() const noexcept;

    //! @brief Get raw data pointer (mutable)
    //! @return Pointer to tensor data
    void* rawPointer() noexcept;

    /*!
     * @brief Get typed data pointer (mutable)
     *
     * Mismatching data type will lead to undefined behavior.
     *
     * @tparam T Data type to cast to
     * @return Typed pointer to tensor data
     */
    template <typename T>
    T* dataPointer() noexcept
    {
        if constexpr (!is_arithmetic_ext<T>::value)
        {
            static_assert(is_arithmetic_ext<T>::value, "Only arithmetic types are supported");
        }
        return reinterpret_cast<T*>(data);
    }

    /*!
     * @brief Get typed data pointer (const)
     *
     * Mismatching data type will lead to undefined behavior.
     *
     * @tparam T Data type to cast to
     * @return Const typed pointer to tensor data
     */
    template <typename T>
    T const* dataPointer() const noexcept
    {
        if constexpr (!is_arithmetic_ext<T>::value)
        {
            static_assert(is_arithmetic_ext<T>::value, "Only arithmetic types are supported");
        }
        return reinterpret_cast<T const*>(data);
    }

    /*!
     * @brief Reshape the tensor
     *
     * Explicitly disallows reshape when the memory is not owned by the tensor
     * object to avoid misuse. Reshape will not happen when memory capacity
     * is insufficient.
     *
     * @param extent New shape
     * @return True if reshape succeeded, false otherwise
     */
    [[nodiscard]] bool reshape(Coords extent) noexcept;

private:
    std::string mName{};
    Coords mShape{};
    std::array<int64_t, kMAX_DIMS> mStrides{};
    DeviceType mDeviceType{};
    nvinfer1::DataType mDataType{};
    void* data{};
    bool ownMemory{};

    // Determined once the tensor is constructed.
    int64_t memoryCapacity{};

    //! @brief Release the owned memory and set tensor to empty state
    void releaseResource();
};

namespace utils
{
//! @brief Get size in bytes of a TensorRT data type
//! @param dataType TensorRT data type (nvinfer1::DataType)
//! @return Size in bytes
//! @throws std::runtime_error If data type is unsupported (e.g., sub-byte types)
size_t getTypeSize(nvinfer1::DataType dataType);

//! @brief Compute strides for given shape
//! @param shape Tensor shape
//! @return Array of strides
//! @throws std::out_of_range If shape index access is out of bounds
std::array<int64_t, kMAX_DIMS> computeStrides(Coords const& shape);

//! @brief Format tensor as string for debugging
//! @param tensor Tensor to format
//! @return String representation of tensor
//! @throws std::runtime_error If data type is unsupported or CUDA memory copy fails
std::string formatString(Tensor const& tensor);

//! @brief Convert bytes to kilobytes
//! @param bytes Size in bytes
//! @return Size in kilobytes
double toKB(size_t bytes) noexcept;

//! @brief Convert bytes to megabytes
//! @param bytes Size in bytes
//! @return Size in megabytes
double toMB(size_t bytes) noexcept;

//! @brief Convert bytes to gigabytes
//! @param bytes Size in bytes
//! @return Size in gigabytes
double toGB(size_t bytes) noexcept;

//! @brief Get maximum value from an INT32 tensor on CPU
//! @param tensor CPU tensor containing INT32 data
//! @return Maximum value in the tensor (0 if tensor is empty)
//! @throws std::runtime_error If tensor is not on CPU or data type is not INT32
int32_t getMaxInt32Value(Tensor const& tensor);
} // namespace utils

//! @brief Optional input tensor type wrapper
using OptionalInputTensor = std::optional<std::reference_wrapper<rt::Tensor const>>;

//! @brief Optional output tensor type wrapper
using OptionalOutputTensor = std::optional<std::reference_wrapper<rt::Tensor>>;

//! @brief Optional input tensors type wrapper (e.g. deepstack features for Qwen3-VL)
using OptionalInputTensors = std::vector<std::reference_wrapper<rt::Tensor const>>;

} // namespace rt
} // namespace trt_edgellm
