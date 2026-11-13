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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "testUtils.h"

using namespace trt_edgellm;

// Given extents built from an initializer list and from a vector
// When they are read and their elements are assigned
// Then the rank, every extent and the volume follow the values given
TEST(TensorTest, Coords)
{
    // Verify initializ(ed list constructor.
    rt::Coords const extent({2, 3, 4});
    ASSERT_EQ(extent.volume(), 24);
    ASSERT_EQ(extent.getNumDims(), 3);
    ASSERT_EQ(extent[0], 2);
    ASSERT_EQ(extent[1], 3);
    ASSERT_EQ(extent[2], 4);

    // Verify vector constructor.
    rt::Coords extent2{std::vector<int64_t>{5, 6, 7, 8}};
    ASSERT_EQ(extent2.volume(), 1680);
    ASSERT_EQ(extent2.getNumDims(), 4);

    // Verify [] operator.
    extent2[2] = 5;
    extent2[3] = 6;
    ASSERT_EQ(extent2.volume(), 900);
}

TEST(TensorTest, CoordsNegative)
{
    // Verify out of bounds access.
    rt::Coords extent({2, 3, 4});
    ASSERT_THROW(extent[3], std::out_of_range);
    ASSERT_THROW(extent[-1], std::out_of_range);
    ASSERT_THROW(extent[100], std::out_of_range);

    // Verify out of bounds constructor.
    std::vector<int64_t> vec(10, 5);
    ASSERT_THROW(rt::Coords extent2(vec), std::runtime_error);
}

// Given a host tensor that allocates its own memory
// When it is created and reshaped
// Then it reports its shape, capacity and row-major strides, and refuses a reshape beyond its capacity
TEST(TensorTest, HostTensorOwnMemory)
{
    rt::Tensor tensor({2, 3, 4}, rt::DeviceType::kCPU, trt_edgellm::DataType::kFLOAT);
    ASSERT_TRUE(tensor.getOwnMemory());
    ASSERT_EQ(tensor.getDeviceType(), rt::DeviceType::kCPU);
    ASSERT_EQ(tensor.getDataType(), trt_edgellm::DataType::kFLOAT);

    // Memory capacity is initial volume * sizeof(float)
    ASSERT_EQ(tensor.getMemoryCapacity(), 96);

    rt::Coords const shape = tensor.getShape();
    ASSERT_EQ(shape.getNumDims(), 3);
    ASSERT_EQ(shape[0], 2);
    ASSERT_EQ(shape[1], 3);
    ASSERT_EQ(shape[2], 4);

    ASSERT_EQ(tensor.getStride(0), 12);
    ASSERT_EQ(tensor.getStride(1), 4);
    ASSERT_EQ(tensor.getStride(2), 1);

    ASSERT_NE(tensor.rawPointer(), nullptr);
    ASSERT_NE(tensor.dataPointer<float>(), nullptr);

    // Verify Identical reshape
    ASSERT_TRUE(tensor.reshape({4, 3, 2}));
    ASSERT_EQ(tensor.getMemoryCapacity(), 96);
    ASSERT_EQ(tensor.getStride(0), 6);
    ASSERT_EQ(tensor.getStride(1), 2);
    ASSERT_EQ(tensor.getStride(2), 1);
    rt::Coords tensorExtent = tensor.getShape();
    ASSERT_EQ(tensorExtent[0], 4);
    ASSERT_EQ(tensorExtent[1], 3);
    ASSERT_EQ(tensorExtent[2], 2);

    // Verify non-identical reshape
    ASSERT_TRUE(tensor.reshape({4, 3}));
    // Capacity keeps unchanged
    ASSERT_EQ(tensor.getMemoryCapacity(), 96);
    ASSERT_EQ(tensor.getStride(0), 3);
    ASSERT_EQ(tensor.getStride(1), 1);
    tensorExtent = tensor.getShape();
    ASSERT_EQ(tensorExtent[0], 4);
    ASSERT_EQ(tensorExtent[1], 3);

    // Verify out of range stride access
    ASSERT_THROW(static_cast<void>(tensor.getStride(3)), std::out_of_range);
    ASSERT_THROW(static_cast<void>(tensor.getStride(-1)), std::out_of_range);

    // Expect reshape to fail when reshape to shapes out of capacity
    ASSERT_FALSE(tensor.reshape({4, 4, 4}));
}

// Given a device tensor that allocates its own memory
// When it is created and then moved
// Then it reports its shape, capacity and strides, and the move hands the memory over and empties the source
TEST(TensorTest, DeviceTensorOwnMemory)
{
    rt::Tensor tensor({8, 16, 32}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    ASSERT_TRUE(tensor.getOwnMemory());
    ASSERT_EQ(tensor.getDeviceType(), rt::DeviceType::kGPU);
    ASSERT_EQ(tensor.getDataType(), trt_edgellm::DataType::kHALF);

    // Memory capacity is initial volume * sizeof(half)
    ASSERT_EQ(tensor.getMemoryCapacity(), 8192);

    rt::Coords const shape = tensor.getShape();
    ASSERT_EQ(shape.getNumDims(), 3);
    ASSERT_EQ(shape[0], 8);
    ASSERT_EQ(shape[1], 16);
    ASSERT_EQ(shape[2], 32);

    ASSERT_EQ(tensor.getStride(0), 512);
    ASSERT_EQ(tensor.getStride(1), 32);
    ASSERT_EQ(tensor.getStride(2), 1);

    half* data = tensor.dataPointer<half>();

    // Verify CUDA calls can succeed on the memory pointer
    CUDA_CHECK(cudaMemset(data, 0, tensor.getMemoryCapacity()));

    // Move assignment of this tensor object.
    rt::Tensor tensor2 = std::move(tensor);
    ASSERT_EQ(tensor2.getDeviceType(), rt::DeviceType::kGPU);
    ASSERT_EQ(tensor2.getDataType(), trt_edgellm::DataType::kHALF);
    ASSERT_EQ(tensor2.getMemoryCapacity(), 8192);
    ASSERT_EQ(tensor2.getOwnMemory(), true);
    ASSERT_EQ(tensor2.getShape().volume(), 4096);

    // New tensor own the memory while original tensor is empty.
    ASSERT_NE(tensor2.dataPointer<half>(), nullptr);
    ASSERT_EQ(tensor.dataPointer<half>(), nullptr);
    ASSERT_EQ(tensor.getOwnMemory(), false);
    ASSERT_EQ(tensor.getShape().volume(), 0);
}

// Given a device tensor that wraps memory it does not own
// When it is created, reshaped and moved
// Then it reports its shape, capacity and strides, refuses a reshape even within capacity, and never owns the memory
TEST(TensorTest, DeviceTensorNonOwnMemory)
{
    void* devicePtr{nullptr};
    CUDA_CHECK(cudaMallocAsync(&devicePtr, 512, 0));

    rt::Tensor tensor(devicePtr, {4, 8, 16}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFP8);
    ASSERT_FALSE(tensor.getOwnMemory());
    ASSERT_EQ(tensor.getDeviceType(), rt::DeviceType::kGPU);
    ASSERT_EQ(tensor.getDataType(), trt_edgellm::DataType::kFP8);

    // Memory capacity is initial volume * sizeof(fp8)
    ASSERT_EQ(tensor.getMemoryCapacity(), 512);

    rt::Coords const shape = tensor.getShape();
    ASSERT_EQ(shape.getNumDims(), 3);
    ASSERT_EQ(shape[0], 4);
    ASSERT_EQ(shape[1], 8);
    ASSERT_EQ(shape[2], 16);

    ASSERT_EQ(tensor.getStride(0), 128);
    ASSERT_EQ(tensor.getStride(1), 16);
    ASSERT_EQ(tensor.getStride(2), 1);

    // Reshape on a non-own memory tensor should fail even if the new shape is within capacity.
    ASSERT_FALSE(tensor.reshape({4, 4, 16}));

    // Move construction of a new tensor object.
    rt::Tensor tensor2(std::move(tensor));

    // New tensor does not own the memory. Old tensor is empty.
    ASSERT_EQ(tensor.getOwnMemory(), false);
    ASSERT_EQ(tensor2.getOwnMemory(), false);

    ASSERT_NE(tensor2.rawPointer(), nullptr);
    ASSERT_EQ(tensor.rawPointer(), nullptr);

    // Clean up the manually allocated memory
    CUDA_CHECK(cudaFreeAsync(devicePtr, 0));
}

TEST(TensorTest, TensorNameFunctionality)
{
    // Test tensor with name
    rt::Tensor namedTensor({2, 3}, rt::DeviceType::kCPU, trt_edgellm::DataType::kFLOAT, "test_tensor");
    ASSERT_EQ(namedTensor.getName(), "test_tensor");

    // Test tensor with empty name (default)
    rt::Tensor unnamedTensor({2, 3}, rt::DeviceType::kCPU, trt_edgellm::DataType::kFLOAT);
    ASSERT_EQ(unnamedTensor.getName(), "");

    // Test tensor with explicit empty name
    rt::Tensor explicitEmptyTensor({2, 3}, rt::DeviceType::kCPU, trt_edgellm::DataType::kFLOAT, "");
    ASSERT_EQ(explicitEmptyTensor.getName(), "");

    // Test tensor with long name
    std::string longName = "very_long_tensor_name_with_many_characters_for_testing_purposes";
    rt::Tensor longNamedTensor({1, 1}, rt::DeviceType::kCPU, trt_edgellm::DataType::kFLOAT, longName);
    ASSERT_EQ(longNamedTensor.getName(), longName);

    // Test non-own memory tensor with name
    void* devicePtr{nullptr};
    CUDA_CHECK(cudaMallocAsync(&devicePtr, 24, 0));
    rt::Tensor nonOwnNamedTensor(
        devicePtr, {2, 3}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "non_own_tensor");
    ASSERT_EQ(nonOwnNamedTensor.getName(), "non_own_tensor");

    // Test move constructor preserves name
    rt::Tensor movedTensor = std::move(namedTensor);
    ASSERT_EQ(movedTensor.getName(), "test_tensor");
    ASSERT_EQ(namedTensor.getName(), ""); // Original tensor should have empty name after move

    // Test move assignment preserves name
    rt::Tensor assignedTensor({1, 1}, rt::DeviceType::kCPU, trt_edgellm::DataType::kFLOAT, "original_name");
    assignedTensor = std::move(movedTensor);
    ASSERT_EQ(assignedTensor.getName(), "test_tensor");
    ASSERT_EQ(movedTensor.getName(), ""); // Moved tensor should have empty name after assignment

    // Clean up the manually allocated memory
    CUDA_CHECK(cudaFreeAsync(devicePtr, 0));
}

TEST(TensorTest, TensorFormatString)
{
    // Set to verbose to print the tensor format string
    trt_edgellm::logger::Severity logLevel = gLogger.getLevel();
    gLogger.setLevel(trt_edgellm::logger::Severity::kVERBOSE);

    // Use defer to restore the log level to handle case where
    // failure happens in the test.
    Defer defer([logLevel]() { gLogger.setLevel(logLevel); });

    // Print some small tensor for sanity check
    std::vector<float> hostData(24);
    std::iota(hostData.begin(), hostData.end(), 1.0f);
    rt::Tensor tensor(hostData.data(), {2, 3, 4}, rt::DeviceType::kCPU, trt_edgellm::DataType::kFLOAT);
    LOG_DEBUG("Tensor format with CPU tensor: %s", rt::utils::formatString(tensor).c_str());

    // Print GPU tensor
    std::vector<half> hostData2(24);
    for (size_t i = 0; i < hostData2.size(); ++i)
    {
        hostData2[i] = __float2half(25.0f - hostData[i]);
    }
    rt::Tensor tensor2({2, 3, 4}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    CUDA_CHECK(cudaMemcpy(tensor2.rawPointer(), hostData2.data(), tensor2.getMemoryCapacity(), cudaMemcpyHostToDevice));
    LOG_DEBUG("Tensor format with GPU tensor: %s", rt::utils::formatString(tensor2).c_str());

    // Try case where one dimension is large
    std::vector<int32_t> hostData3(64);
    std::iota(hostData3.begin(), hostData3.end(), 1);
    rt::Tensor tensor3({64}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32);
    memcpy(tensor3.rawPointer(), hostData3.data(), tensor3.getMemoryCapacity());
    LOG_DEBUG("Tensor format with large dimension: %s", rt::utils::formatString(tensor3).c_str());

    // Try case where src tensor has been reshaped.
    rt::Tensor tensor4({8, 8, 8}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT8);
    std::vector<int8_t> hostData4(512);
    for (size_t i = 0; i < hostData4.size(); ++i)
    {
        hostData4[i] = static_cast<int8_t>(i) % 8;
    }
    memcpy(tensor4.rawPointer(), hostData4.data(), tensor4.getMemoryCapacity());
    ASSERT_TRUE(tensor4.reshape({3, 3, 3}));
    LOG_DEBUG("Tensor format with reshaped tensor: %s", rt::utils::formatString(tensor4).c_str());
}
