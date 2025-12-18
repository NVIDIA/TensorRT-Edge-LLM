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

#include "common/checkMacros.h"
#include "common/safetensorsUtils.h"
#include "testUtils.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace trt_edgellm;

TEST(SafetensorsUtilsTest, FloatTensors)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::string testFile = "test_float_tensors.safetensors";
    if (std::filesystem::exists(testFile))
    {
        std::filesystem::remove(testFile);
    }

    // CPU tensor: 1D
    rt::Tensor cpuTensor({5}, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "cpu_float");
    // GPU tensor: 2D
    rt::Tensor gpuTensor({2, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "gpu_float");

    // Test data
    std::vector<float> cpuData = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
    std::vector<float> gpuData = {10.1f, 10.2f, 10.3f, 10.4f, 10.5f, 10.6f};

    // Copy data
    std::memcpy(cpuTensor.rawPointer(), cpuData.data(), 5 * sizeof(float));
    CUDA_CHECK(
        cudaMemcpyAsync(gpuTensor.rawPointer(), gpuData.data(), 6 * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Serialize
    std::vector<rt::Tensor> tensors;
    tensors.push_back(std::move(cpuTensor));
    tensors.push_back(std::move(gpuTensor));
    EXPECT_TRUE(rt::safetensors::saveSafetensors(testFile, tensors, stream));

    // Load and verify
    std::vector<rt::Tensor> loadedTensors;
    EXPECT_TRUE(rt::safetensors::loadSafetensors(testFile, loadedTensors, stream));
    EXPECT_EQ(loadedTensors.size(), 2);

    // Find tensors
    rt::Tensor* loadedCpu = nullptr;
    rt::Tensor* loadedGpu = nullptr;
    for (auto& tensor : loadedTensors)
    {
        if (tensor.getName() == "cpu_float")
            loadedCpu = &tensor;
        else if (tensor.getName() == "gpu_float")
            loadedGpu = &tensor;
    }

    // Verify CPU tensor
    EXPECT_NE(loadedCpu, nullptr);
    EXPECT_EQ(loadedCpu->getShape().getNumDims(), 1);
    EXPECT_EQ(loadedCpu->getShape()[0], 5);
    EXPECT_EQ(loadedCpu->getDataType(), nvinfer1::DataType::kFLOAT);

    std::vector<float> loadedCpuData(5);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedCpuData.data(), loadedCpu->rawPointer(), 5 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 5; ++i)
    {
        EXPECT_TRUE(isclose(loadedCpuData[i], cpuData[i], 1e-5, 1e-5));
    }

    // Verify GPU tensor
    EXPECT_NE(loadedGpu, nullptr);
    EXPECT_EQ(loadedGpu->getShape().getNumDims(), 2);
    EXPECT_EQ(loadedGpu->getShape()[0], 2);
    EXPECT_EQ(loadedGpu->getShape()[1], 3);
    EXPECT_EQ(loadedGpu->getDataType(), nvinfer1::DataType::kFLOAT);

    std::vector<float> loadedGpuData(6);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedGpuData.data(), loadedGpu->rawPointer(), 6 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 6; ++i)
    {
        EXPECT_TRUE(isclose(loadedGpuData[i], gpuData[i], 1e-5, 1e-5));
    }

    std::filesystem::remove(testFile);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SafetensorsUtilsTest, HalfTensors)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::string testFile = "test_half_tensors.safetensors";
    if (std::filesystem::exists(testFile))
    {
        std::filesystem::remove(testFile);
    }

    // CPU tensor: 3D
    rt::Tensor cpuTensor({2, 2, 2}, rt::DeviceType::kCPU, nvinfer1::DataType::kHALF, "cpu_half");
    // GPU tensor: 4D
    rt::Tensor gpuTensor({1, 2, 3, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "gpu_half");

    // Test data
    std::vector<half> cpuData = {__float2half(1.1f), __float2half(2.2f), __float2half(3.3f), __float2half(4.4f),
        __float2half(5.5f), __float2half(6.6f), __float2half(7.7f), __float2half(8.8f)};
    std::vector<half> gpuData = {__float2half(-2.0f), __float2half(-1.25f), __float2half(0.0f), __float2half(1.1f),
        __float2half(2.0f), __float2half(3.0f), __float2half(4.0f), __float2half(5.0f), __float2half(6.0f),
        __float2half(7.0f), __float2half(8.0f), __float2half(9.0f)};

    // Copy data
    std::memcpy(cpuTensor.rawPointer(), cpuData.data(), 8 * sizeof(half));
    CUDA_CHECK(
        cudaMemcpyAsync(gpuTensor.rawPointer(), gpuData.data(), 12 * sizeof(half), cudaMemcpyHostToDevice, stream));

    // Serialize
    std::vector<rt::Tensor> tensors;
    tensors.push_back(std::move(cpuTensor));
    tensors.push_back(std::move(gpuTensor));
    EXPECT_TRUE(rt::safetensors::saveSafetensors(testFile, tensors, stream));

    // Load and verify
    std::vector<rt::Tensor> loadedTensors;
    EXPECT_TRUE(rt::safetensors::loadSafetensors(testFile, loadedTensors, stream));
    EXPECT_EQ(loadedTensors.size(), 2);

    // Find tensors
    rt::Tensor* loadedCpu = nullptr;
    rt::Tensor* loadedGpu = nullptr;
    for (auto& tensor : loadedTensors)
    {
        if (tensor.getName() == "cpu_half")
            loadedCpu = &tensor;
        else if (tensor.getName() == "gpu_half")
            loadedGpu = &tensor;
    }

    // Verify CPU tensor
    EXPECT_NE(loadedCpu, nullptr);
    EXPECT_EQ(loadedCpu->getShape().getNumDims(), 3);
    EXPECT_EQ(loadedCpu->getShape()[0], 2);
    EXPECT_EQ(loadedCpu->getShape()[1], 2);
    EXPECT_EQ(loadedCpu->getShape()[2], 2);
    EXPECT_EQ(loadedCpu->getDataType(), nvinfer1::DataType::kHALF);

    std::vector<half> loadedCpuData(8);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedCpuData.data(), loadedCpu->rawPointer(), 8 * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_TRUE(isclose(loadedCpuData[i], cpuData[i], 1e-2, 1e-2));
    }

    // Verify GPU tensor
    EXPECT_NE(loadedGpu, nullptr);
    EXPECT_EQ(loadedGpu->getShape().getNumDims(), 4);
    EXPECT_EQ(loadedGpu->getShape()[0], 1);
    EXPECT_EQ(loadedGpu->getShape()[1], 2);
    EXPECT_EQ(loadedGpu->getShape()[2], 3);
    EXPECT_EQ(loadedGpu->getShape()[3], 2);
    EXPECT_EQ(loadedGpu->getDataType(), nvinfer1::DataType::kHALF);

    std::vector<half> loadedGpuData(12);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedGpuData.data(), loadedGpu->rawPointer(), 12 * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 12; ++i)
    {
        EXPECT_TRUE(isclose(loadedGpuData[i], gpuData[i], 1e-2, 1e-2));
    }

    std::filesystem::remove(testFile);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SafetensorsUtilsTest, Bf16Tensors)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::string testFile = "test_bf16_tensors.safetensors";
    if (std::filesystem::exists(testFile))
    {
        std::filesystem::remove(testFile);
    }

    // CPU tensor: 2D
    rt::Tensor cpuTensor({3, 2}, rt::DeviceType::kCPU, nvinfer1::DataType::kBF16, "cpu_bf16");
    // GPU tensor: 5D
    rt::Tensor gpuTensor({1, 1, 2, 2, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kBF16, "gpu_bf16");

    // Test data
    std::vector<__nv_bfloat16> cpuData = {__float2bfloat16(-3.8f), __float2bfloat16(0.0f), __float2bfloat16(2.7f),
        __float2bfloat16(1.5f), __float2bfloat16(-0.5f), __float2bfloat16(3.2f)};
    std::vector<__nv_bfloat16> gpuData
        = {__float2bfloat16(1.0f), __float2bfloat16(2.0f), __float2bfloat16(3.0f), __float2bfloat16(4.0f),
            __float2bfloat16(5.0f), __float2bfloat16(6.0f), __float2bfloat16(7.0f), __float2bfloat16(8.0f)};

    // Copy data
    std::memcpy(cpuTensor.rawPointer(), cpuData.data(), 6 * sizeof(__nv_bfloat16));
    CUDA_CHECK(cudaMemcpyAsync(
        gpuTensor.rawPointer(), gpuData.data(), 8 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));

    // Serialize
    std::vector<rt::Tensor> tensors;
    tensors.push_back(std::move(cpuTensor));
    tensors.push_back(std::move(gpuTensor));
    EXPECT_TRUE(rt::safetensors::saveSafetensors(testFile, tensors, stream));

    // Load and verify
    std::vector<rt::Tensor> loadedTensors;
    EXPECT_TRUE(rt::safetensors::loadSafetensors(testFile, loadedTensors, stream));
    EXPECT_EQ(loadedTensors.size(), 2);

    // Find tensors
    rt::Tensor* loadedCpu = nullptr;
    rt::Tensor* loadedGpu = nullptr;
    for (auto& tensor : loadedTensors)
    {
        if (tensor.getName() == "cpu_bf16")
            loadedCpu = &tensor;
        else if (tensor.getName() == "gpu_bf16")
            loadedGpu = &tensor;
    }

    // Verify CPU tensor
    EXPECT_NE(loadedCpu, nullptr);
    EXPECT_EQ(loadedCpu->getShape().getNumDims(), 2);
    EXPECT_EQ(loadedCpu->getShape()[0], 3);
    EXPECT_EQ(loadedCpu->getShape()[1], 2);
    EXPECT_EQ(loadedCpu->getDataType(), nvinfer1::DataType::kBF16);

    std::vector<__nv_bfloat16> loadedCpuData(6);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedCpuData.data(), loadedCpu->rawPointer(), 6 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 6; ++i)
    {
        EXPECT_TRUE(isclose(loadedCpuData[i], cpuData[i], 1e-2, 1e-2));
    }

    // Verify GPU tensor
    EXPECT_NE(loadedGpu, nullptr);
    EXPECT_EQ(loadedGpu->getShape().getNumDims(), 5);
    EXPECT_EQ(loadedGpu->getShape()[0], 1);
    EXPECT_EQ(loadedGpu->getShape()[1], 1);
    EXPECT_EQ(loadedGpu->getShape()[2], 2);
    EXPECT_EQ(loadedGpu->getShape()[3], 2);
    EXPECT_EQ(loadedGpu->getShape()[4], 2);
    EXPECT_EQ(loadedGpu->getDataType(), nvinfer1::DataType::kBF16);

    std::vector<__nv_bfloat16> loadedGpuData(8);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedGpuData.data(), loadedGpu->rawPointer(), 8 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_TRUE(isclose(loadedGpuData[i], gpuData[i], 1e-2, 1e-2));
    }

    std::filesystem::remove(testFile);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SafetensorsUtilsTest, Int32Tensors)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::string testFile = "test_int32_tensors.safetensors";
    if (std::filesystem::exists(testFile))
    {
        std::filesystem::remove(testFile);
    }

    // CPU tensor: 4D
    rt::Tensor cpuTensor({1, 2, 2, 2}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "cpu_int32");
    // GPU tensor: 1D
    rt::Tensor gpuTensor({8}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "gpu_int32");

    // Test data
    std::vector<int32_t> cpuData = {100, 200, 300, 400, 500, 600, 700, 800};
    std::vector<int32_t> gpuData = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000};

    // Copy data
    std::memcpy(cpuTensor.rawPointer(), cpuData.data(), 8 * sizeof(int32_t));
    CUDA_CHECK(cudaMemcpyAsync(gpuTensor.rawPointer(), gpuData.data(), 8 * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Serialize
    std::vector<rt::Tensor> tensors;
    tensors.push_back(std::move(cpuTensor));
    tensors.push_back(std::move(gpuTensor));
    EXPECT_TRUE(rt::safetensors::saveSafetensors(testFile, tensors, stream));

    // Load and verify
    std::vector<rt::Tensor> loadedTensors;
    EXPECT_TRUE(rt::safetensors::loadSafetensors(testFile, loadedTensors, stream));
    EXPECT_EQ(loadedTensors.size(), 2);

    // Find tensors
    rt::Tensor* loadedCpu = nullptr;
    rt::Tensor* loadedGpu = nullptr;
    for (auto& tensor : loadedTensors)
    {
        if (tensor.getName() == "cpu_int32")
            loadedCpu = &tensor;
        else if (tensor.getName() == "gpu_int32")
            loadedGpu = &tensor;
    }

    // Verify CPU tensor
    EXPECT_NE(loadedCpu, nullptr);
    EXPECT_EQ(loadedCpu->getShape().getNumDims(), 4);
    EXPECT_EQ(loadedCpu->getShape()[0], 1);
    EXPECT_EQ(loadedCpu->getShape()[1], 2);
    EXPECT_EQ(loadedCpu->getShape()[2], 2);
    EXPECT_EQ(loadedCpu->getShape()[3], 2);
    EXPECT_EQ(loadedCpu->getDataType(), nvinfer1::DataType::kINT32);

    std::vector<int32_t> loadedCpuData(8);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedCpuData.data(), loadedCpu->rawPointer(), 8 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_EQ(loadedCpuData[i], cpuData[i]);
    }

    // Verify GPU tensor
    EXPECT_NE(loadedGpu, nullptr);
    EXPECT_EQ(loadedGpu->getShape().getNumDims(), 1);
    EXPECT_EQ(loadedGpu->getShape()[0], 8);
    EXPECT_EQ(loadedGpu->getDataType(), nvinfer1::DataType::kINT32);

    std::vector<int32_t> loadedGpuData(8);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedGpuData.data(), loadedGpu->rawPointer(), 8 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_EQ(loadedGpuData[i], gpuData[i]);
    }

    std::filesystem::remove(testFile);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SafetensorsUtilsTest, NonOwnedMemoryTensors)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::string testFile = "test_non_owned_tensors.safetensors";
    if (std::filesystem::exists(testFile))
    {
        std::filesystem::remove(testFile);
    }

    // Allocate shared memory buffer using cudaMallocAsync for better memory management
    void* sharedMemoryBuffer = nullptr;
    CUDA_CHECK(cudaMallocAsync(&sharedMemoryBuffer, 200, stream));

    // Non-owned tensor 1: FLOAT at offset 0
    float* floatPtr = static_cast<float*>(sharedMemoryBuffer);
    rt::Tensor nonOwnedFloatTensor(
        floatPtr, {2, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "non_owned_float");

    // Non-owned tensor 2: HALF at offset 16 bytes (4 floats)
    half* halfPtr = static_cast<half*>(sharedMemoryBuffer) + 8; // 4 floats = 16 bytes
    rt::Tensor nonOwnedHalfTensor(halfPtr, {3, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "non_owned_half");

    // Test data
    std::vector<float> floatData = {10.1f, 10.2f, 10.3f, 10.4f};
    std::vector<half> halfData = {__float2half(1.0f), __float2half(2.0f), __float2half(3.0f), __float2half(4.0f),
        __float2half(5.0f), __float2half(6.0f), __float2half(7.0f), __float2half(8.0f), __float2half(9.0f)};

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpyAsync(
        nonOwnedFloatTensor.rawPointer(), floatData.data(), 4 * sizeof(float), cudaMemcpyHostToDevice, 0));
    CUDA_CHECK(
        cudaMemcpyAsync(nonOwnedHalfTensor.rawPointer(), halfData.data(), 9 * sizeof(half), cudaMemcpyHostToDevice, 0));

    // Synchronize to ensure data is copied before proceeding
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Serialize
    std::vector<rt::Tensor> tensors;
    tensors.push_back(std::move(nonOwnedFloatTensor));
    tensors.push_back(std::move(nonOwnedHalfTensor));
    EXPECT_TRUE(rt::safetensors::saveSafetensors(testFile, tensors, stream));

    // Load and verify
    std::vector<rt::Tensor> loadedTensors;
    EXPECT_TRUE(rt::safetensors::loadSafetensors(testFile, loadedTensors, stream));
    EXPECT_EQ(loadedTensors.size(), 2);

    // Find tensors
    rt::Tensor* loadedFloat = nullptr;
    rt::Tensor* loadedHalf = nullptr;
    for (auto& tensor : loadedTensors)
    {
        if (tensor.getName() == "non_owned_float")
            loadedFloat = &tensor;
        else if (tensor.getName() == "non_owned_half")
            loadedHalf = &tensor;
    }

    // Verify FLOAT tensor
    EXPECT_NE(loadedFloat, nullptr);
    EXPECT_EQ(loadedFloat->getShape().getNumDims(), 2);
    EXPECT_EQ(loadedFloat->getShape()[0], 2);
    EXPECT_EQ(loadedFloat->getShape()[1], 2);
    EXPECT_EQ(loadedFloat->getDataType(), nvinfer1::DataType::kFLOAT);

    std::vector<float> loadedFloatData(4);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedFloatData.data(), loadedFloat->rawPointer(), 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_TRUE(isclose(loadedFloatData[i], floatData[i], 1e-5, 1e-5));
    }

    // Verify HALF tensor
    EXPECT_NE(loadedHalf, nullptr);
    EXPECT_EQ(loadedHalf->getShape().getNumDims(), 2);
    EXPECT_EQ(loadedHalf->getShape()[0], 3);
    EXPECT_EQ(loadedHalf->getShape()[1], 3);
    EXPECT_EQ(loadedHalf->getDataType(), nvinfer1::DataType::kHALF);

    std::vector<half> loadedHalfData(9);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedHalfData.data(), loadedHalf->rawPointer(), 9 * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 9; ++i)
    {
        EXPECT_TRUE(isclose(loadedHalfData[i], halfData[i], 1e-3, 1e-3));
    }

    // Clean up
    CUDA_CHECK(cudaFreeAsync(sharedMemoryBuffer, stream));
    std::filesystem::remove(testFile);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SafetensorsUtilsTest, MultipleTensorsWithSharedMemory)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::string testFile = "test_shared_memory_tensors.safetensors";
    if (std::filesystem::exists(testFile))
    {
        std::filesystem::remove(testFile);
    }

    // Allocate large shared memory buffer using cudaMallocAsync for better memory management
    void* sharedMemoryBuffer = nullptr;
    CUDA_CHECK(cudaMallocAsync(&sharedMemoryBuffer, 500, stream));

    // Multiple non-owned tensors sharing the same memory buffer
    float* floatPtr = static_cast<float*>(sharedMemoryBuffer);
    rt::Tensor tensor1(floatPtr, {2, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "shared_float");

    half* halfPtr = static_cast<half*>(sharedMemoryBuffer) + 8; // 4 floats = 16 bytes
    rt::Tensor tensor2(halfPtr, {3, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "shared_half");

    __nv_bfloat16* bf16Ptr = static_cast<__nv_bfloat16*>(sharedMemoryBuffer) + 17; // 34 bytes offset
    rt::Tensor tensor3(bf16Ptr, {2, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kBF16, "shared_bf16");

    int32_t* int32Ptr = static_cast<int32_t*>(sharedMemoryBuffer) + 25; // 50 bytes offset
    rt::Tensor tensor4(int32Ptr, {2, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "shared_int32");

    // Test data
    std::vector<float> floatData = {1.1f, 2.2f, 3.3f, 4.4f};
    std::vector<half> halfData = {__float2half(5.0f), __float2half(6.0f), __float2half(7.0f), __float2half(8.0f),
        __float2half(9.0f), __float2half(10.0f), __float2half(11.0f), __float2half(12.0f), __float2half(13.0f)};
    std::vector<__nv_bfloat16> bf16Data
        = {__float2bfloat16(14.0f), __float2bfloat16(15.0f), __float2bfloat16(16.0f), __float2bfloat16(17.0f),
            __float2bfloat16(18.0f), __float2bfloat16(19.0f), __float2bfloat16(20.0f), __float2bfloat16(21.0f)};
    std::vector<int32_t> int32Data = {100, 200, 300, 400};

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpyAsync(tensor1.rawPointer(), floatData.data(), 4 * sizeof(float), cudaMemcpyHostToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(tensor2.rawPointer(), halfData.data(), 9 * sizeof(half), cudaMemcpyHostToDevice, 0));
    CUDA_CHECK(
        cudaMemcpyAsync(tensor3.rawPointer(), bf16Data.data(), 8 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(tensor4.rawPointer(), int32Data.data(), 4 * sizeof(int32_t), cudaMemcpyHostToDevice, 0));

    // Synchronize to ensure data is copied before proceeding
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Serialize
    std::vector<rt::Tensor> tensors;
    tensors.push_back(std::move(tensor1));
    tensors.push_back(std::move(tensor2));
    tensors.push_back(std::move(tensor3));
    tensors.push_back(std::move(tensor4));
    EXPECT_TRUE(rt::safetensors::saveSafetensors(testFile, tensors, stream));

    // Load and verify
    std::vector<rt::Tensor> loadedTensors;
    EXPECT_TRUE(rt::safetensors::loadSafetensors(testFile, loadedTensors, stream));
    EXPECT_EQ(loadedTensors.size(), 4);

    // Find tensors
    rt::Tensor* loadedFloat = nullptr;
    rt::Tensor* loadedHalf = nullptr;
    rt::Tensor* loadedBf16 = nullptr;
    rt::Tensor* loadedInt32 = nullptr;
    for (auto& tensor : loadedTensors)
    {
        if (tensor.getName() == "shared_float")
            loadedFloat = &tensor;
        else if (tensor.getName() == "shared_half")
            loadedHalf = &tensor;
        else if (tensor.getName() == "shared_bf16")
            loadedBf16 = &tensor;
        else if (tensor.getName() == "shared_int32")
            loadedInt32 = &tensor;
    }

    // Verify all tensors
    EXPECT_NE(loadedFloat, nullptr);
    EXPECT_NE(loadedHalf, nullptr);
    EXPECT_NE(loadedBf16, nullptr);
    EXPECT_NE(loadedInt32, nullptr);

    // Verify FLOAT tensor
    std::vector<float> loadedFloatData(4);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedFloatData.data(), loadedFloat->rawPointer(), 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_TRUE(isclose(loadedFloatData[i], floatData[i], 1e-5, 1e-5));
    }

    // Verify HALF tensor
    std::vector<half> loadedHalfData(9);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedHalfData.data(), loadedHalf->rawPointer(), 9 * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 9; ++i)
    {
        EXPECT_TRUE(isclose(loadedHalfData[i], halfData[i], 1e-3, 1e-3));
    }

    // Verify BF16 tensor
    std::vector<__nv_bfloat16> loadedBf16Data(8);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedBf16Data.data(), loadedBf16->rawPointer(), 8 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_TRUE(isclose(loadedBf16Data[i], bf16Data[i], 1e-3, 1e-3));
    }

    // Verify INT32 tensor
    std::vector<int32_t> loadedInt32Data(4);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedInt32Data.data(), loadedInt32->rawPointer(), 4 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(loadedInt32Data[i], int32Data[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFreeAsync(sharedMemoryBuffer, stream));
    std::filesystem::remove(testFile);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SafetensorsUtilsTest, SafetensorsLoadingFromFile)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::string testFile = std::string(PROJECT_ROOT_DIR) + "/unittests/resources/test_safetensors.safetensors";

    // Load the file using the unified interface
    std::vector<trt_edgellm::rt::Tensor> tensors;
    EXPECT_TRUE(trt_edgellm::rt::safetensors::loadSafetensors(testFile, tensors, stream));

    // Verify we got the expected number of tensors
    EXPECT_EQ(tensors.size(), 2);

    // Find tensor0 and tensor1 by name
    rt::Tensor* tensor0 = nullptr;
    rt::Tensor* tensor1 = nullptr;

    for (auto& tensor : tensors)
    {
        if (tensor.getName() == "tensor0")
        {
            tensor0 = &tensor;
        }
        else if (tensor.getName() == "tensor1")
        {
            tensor1 = &tensor;
        }
    }

    // Verify tensor0
    EXPECT_NE(tensor0, nullptr);
    EXPECT_EQ(tensor0->getShape().getNumDims(), 2);
    EXPECT_EQ(tensor0->getShape()[0], 1);
    EXPECT_EQ(tensor0->getShape()[1], 5);
    EXPECT_EQ(tensor0->getDataType(), nvinfer1::DataType::kHALF);

    // Copy tensor0 data from GPU to CPU using direct CUDA operations
    std::vector<half> tensor0Cpu(5);
    CUDA_CHECK(
        cudaMemcpyAsync(tensor0Cpu.data(), tensor0->rawPointer(), 5 * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify tensor0 values using direct half comparison
    EXPECT_TRUE(isclose(tensor0Cpu[0], __float2half(-2.0f), 1e-2, 1e-2));
    EXPECT_TRUE(isclose(tensor0Cpu[1], __float2half(-1.25f), 1e-2, 1e-2));
    EXPECT_TRUE(isclose(tensor0Cpu[2], __float2half(0.0f), 1e-2, 1e-2));
    EXPECT_TRUE(isclose(tensor0Cpu[3], __float2half(1.1f), 1e-2, 1e-2));
    EXPECT_TRUE(isclose(tensor0Cpu[4], __float2half(2.0f), 1e-2, 1e-2));

    // Verify tensor1
    EXPECT_NE(tensor1, nullptr);
    EXPECT_EQ(tensor1->getShape().getNumDims(), 2);
    EXPECT_EQ(tensor1->getShape()[0], 1);
    EXPECT_EQ(tensor1->getShape()[1], 3);
    EXPECT_EQ(tensor1->getDataType(), nvinfer1::DataType::kBF16);

    // Copy tensor1 data from GPU to CPU using direct CUDA operations
    std::vector<__nv_bfloat16> tensor1Cpu(3);
    CUDA_CHECK(cudaMemcpyAsync(
        tensor1Cpu.data(), tensor1->rawPointer(), 3 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify tensor1 values using direct bfloat16 comparison
    EXPECT_TRUE(isclose(tensor1Cpu[0], __float2bfloat16(-3.8f), 1e-2, 1e-2));
    EXPECT_TRUE(isclose(tensor1Cpu[1], __float2bfloat16(0.0f), 1e-2, 1e-2));
    EXPECT_TRUE(isclose(tensor1Cpu[2], __float2bfloat16(2.7f), 1e-2, 1e-2));

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SafetensorsUtilsTest, SafetensorsErrorHandling)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Test 1: Empty tensor vector
    std::string testFile = "test_empty_vector.safetensors";
    std::vector<rt::Tensor> emptyTensors;
    EXPECT_FALSE(rt::safetensors::saveSafetensors(testFile, emptyTensors, stream));
    EXPECT_FALSE(std::filesystem::exists(testFile));

    // Test 2: Empty tensor name
    rt::Tensor tensorWithEmptyName({2, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "");
    std::vector<rt::Tensor> tensorsWithEmptyName;
    tensorsWithEmptyName.push_back(std::move(tensorWithEmptyName));
    EXPECT_FALSE(rt::safetensors::saveSafetensors(testFile, tensorsWithEmptyName, stream));
    EXPECT_FALSE(std::filesystem::exists(testFile));

    // Test 3: Duplicate tensor names
    rt::Tensor tensor1({2, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "duplicate_name");
    rt::Tensor tensor2({4, 5}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "duplicate_name");
    std::vector<rt::Tensor> tensorsWithDuplicateNames;
    tensorsWithDuplicateNames.push_back(std::move(tensor1));
    tensorsWithDuplicateNames.push_back(std::move(tensor2));
    EXPECT_FALSE(rt::safetensors::saveSafetensors(testFile, tensorsWithDuplicateNames, stream));
    EXPECT_FALSE(std::filesystem::exists(testFile));

    // Test 4: Load non-existent file
    std::vector<rt::Tensor> loadedTensors;
    EXPECT_FALSE(rt::safetensors::loadSafetensors("non_existent_file.safetensors", loadedTensors, stream));
    EXPECT_EQ(loadedTensors.size(), 0);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SafetensorsUtilsTest, Int8Tensors)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::string testFile = "test_int8_tensors.safetensors";
    if (std::filesystem::exists(testFile))
    {
        std::filesystem::remove(testFile);
    }

    // CPU tensor: 2D
    rt::Tensor cpuTensor({3, 4}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT8, "cpu_int8");
    // GPU tensor: 3D
    rt::Tensor gpuTensor({2, 2, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "gpu_int8");

    // Test data
    std::vector<int8_t> cpuData = {-128, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4};
    std::vector<int8_t> gpuData = {8, 16, 32, 64, 127, -127, -100, -50, 50, 100, 0, 42};

    // Copy data
    std::memcpy(cpuTensor.rawPointer(), cpuData.data(), 12 * sizeof(int8_t));
    CUDA_CHECK(
        cudaMemcpyAsync(gpuTensor.rawPointer(), gpuData.data(), 12 * sizeof(int8_t), cudaMemcpyHostToDevice, stream));

    // Serialize
    std::vector<rt::Tensor> tensors;
    tensors.push_back(std::move(cpuTensor));
    tensors.push_back(std::move(gpuTensor));
    EXPECT_TRUE(rt::safetensors::saveSafetensors(testFile, tensors, stream));

    // Load and verify
    std::vector<rt::Tensor> loadedTensors;
    EXPECT_TRUE(rt::safetensors::loadSafetensors(testFile, loadedTensors, stream));
    EXPECT_EQ(loadedTensors.size(), 2);

    // Find tensors
    rt::Tensor* loadedCpu = nullptr;
    rt::Tensor* loadedGpu = nullptr;
    for (auto& tensor : loadedTensors)
    {
        if (tensor.getName() == "cpu_int8")
            loadedCpu = &tensor;
        else if (tensor.getName() == "gpu_int8")
            loadedGpu = &tensor;
    }

    // Verify CPU tensor
    EXPECT_NE(loadedCpu, nullptr);
    EXPECT_EQ(loadedCpu->getShape().getNumDims(), 2);
    EXPECT_EQ(loadedCpu->getShape()[0], 3);
    EXPECT_EQ(loadedCpu->getShape()[1], 4);
    EXPECT_EQ(loadedCpu->getDataType(), nvinfer1::DataType::kINT8);

    std::vector<int8_t> loadedCpuData(12);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedCpuData.data(), loadedCpu->rawPointer(), 12 * sizeof(int8_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 12; ++i)
    {
        EXPECT_EQ(loadedCpuData[i], cpuData[i]);
    }

    // Verify GPU tensor
    EXPECT_NE(loadedGpu, nullptr);
    EXPECT_EQ(loadedGpu->getShape().getNumDims(), 3);
    EXPECT_EQ(loadedGpu->getShape()[0], 2);
    EXPECT_EQ(loadedGpu->getShape()[1], 2);
    EXPECT_EQ(loadedGpu->getShape()[2], 3);
    EXPECT_EQ(loadedGpu->getDataType(), nvinfer1::DataType::kINT8);

    std::vector<int8_t> loadedGpuData(12);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedGpuData.data(), loadedGpu->rawPointer(), 12 * sizeof(int8_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 12; ++i)
    {
        EXPECT_EQ(loadedGpuData[i], gpuData[i]);
    }

    std::filesystem::remove(testFile);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SafetensorsUtilsTest, Uint8Tensors)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::string testFile = "test_uint8_tensors.safetensors";
    if (std::filesystem::exists(testFile))
    {
        std::filesystem::remove(testFile);
    }

    // CPU tensor: 1D
    rt::Tensor cpuTensor({8}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "cpu_uint8");
    // GPU tensor: 4D
    rt::Tensor gpuTensor({1, 2, 2, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "gpu_uint8");

    // Test data
    std::vector<uint8_t> cpuData = {0, 1, 2, 4, 8, 16, 32, 64};
    std::vector<uint8_t> gpuData = {128, 255, 100, 200, 50, 150, 75, 225};

    // Copy data
    std::memcpy(cpuTensor.rawPointer(), cpuData.data(), 8 * sizeof(uint8_t));
    CUDA_CHECK(
        cudaMemcpyAsync(gpuTensor.rawPointer(), gpuData.data(), 8 * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));

    // Serialize
    std::vector<rt::Tensor> tensors;
    tensors.push_back(std::move(cpuTensor));
    tensors.push_back(std::move(gpuTensor));
    EXPECT_TRUE(rt::safetensors::saveSafetensors(testFile, tensors, stream));

    // Load and verify
    std::vector<rt::Tensor> loadedTensors;
    EXPECT_TRUE(rt::safetensors::loadSafetensors(testFile, loadedTensors, stream));
    EXPECT_EQ(loadedTensors.size(), 2);

    // Find tensors
    rt::Tensor* loadedCpu = nullptr;
    rt::Tensor* loadedGpu = nullptr;
    for (auto& tensor : loadedTensors)
    {
        if (tensor.getName() == "cpu_uint8")
            loadedCpu = &tensor;
        else if (tensor.getName() == "gpu_uint8")
            loadedGpu = &tensor;
    }

    // Verify CPU tensor
    EXPECT_NE(loadedCpu, nullptr);
    EXPECT_EQ(loadedCpu->getShape().getNumDims(), 1);
    EXPECT_EQ(loadedCpu->getShape()[0], 8);
    EXPECT_EQ(loadedCpu->getDataType(), nvinfer1::DataType::kUINT8);

    std::vector<uint8_t> loadedCpuData(8);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedCpuData.data(), loadedCpu->rawPointer(), 8 * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_EQ(loadedCpuData[i], cpuData[i]);
    }

    // Verify GPU tensor
    EXPECT_NE(loadedGpu, nullptr);
    EXPECT_EQ(loadedGpu->getShape().getNumDims(), 4);
    EXPECT_EQ(loadedGpu->getShape()[0], 1);
    EXPECT_EQ(loadedGpu->getShape()[1], 2);
    EXPECT_EQ(loadedGpu->getShape()[2], 2);
    EXPECT_EQ(loadedGpu->getShape()[3], 2);
    EXPECT_EQ(loadedGpu->getDataType(), nvinfer1::DataType::kUINT8);

    std::vector<uint8_t> loadedGpuData(8);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedGpuData.data(), loadedGpu->rawPointer(), 8 * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_EQ(loadedGpuData[i], gpuData[i]);
    }

    std::filesystem::remove(testFile);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SafetensorsUtilsTest, Int64Tensors)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::string testFile = "test_int64_tensors.safetensors";
    if (std::filesystem::exists(testFile))
    {
        std::filesystem::remove(testFile);
    }

    // CPU tensor: 2D
    rt::Tensor cpuTensor({2, 3}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64, "cpu_int64");
    // GPU tensor: 1D
    rt::Tensor gpuTensor({6}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "gpu_int64");

    // Test data
    std::vector<int64_t> cpuData
        = {-9223372036854775807LL, -1000000000000LL, -1000000LL, 0LL, 1000000LL, 1000000000000LL};
    std::vector<int64_t> gpuData
        = {9223372036854775807LL, 1234567890123456789LL, -987654321098765432LL, 42LL, -42LL, 0LL};

    // Copy data
    std::memcpy(cpuTensor.rawPointer(), cpuData.data(), 6 * sizeof(int64_t));
    CUDA_CHECK(
        cudaMemcpyAsync(gpuTensor.rawPointer(), gpuData.data(), 6 * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    // Serialize
    std::vector<rt::Tensor> tensors;
    tensors.push_back(std::move(cpuTensor));
    tensors.push_back(std::move(gpuTensor));
    EXPECT_TRUE(rt::safetensors::saveSafetensors(testFile, tensors, stream));

    // Load and verify
    std::vector<rt::Tensor> loadedTensors;
    EXPECT_TRUE(rt::safetensors::loadSafetensors(testFile, loadedTensors, stream));
    EXPECT_EQ(loadedTensors.size(), 2);

    // Find tensors
    rt::Tensor* loadedCpu = nullptr;
    rt::Tensor* loadedGpu = nullptr;
    for (auto& tensor : loadedTensors)
    {
        if (tensor.getName() == "cpu_int64")
            loadedCpu = &tensor;
        else if (tensor.getName() == "gpu_int64")
            loadedGpu = &tensor;
    }

    // Verify CPU tensor
    EXPECT_NE(loadedCpu, nullptr);
    EXPECT_EQ(loadedCpu->getShape().getNumDims(), 2);
    EXPECT_EQ(loadedCpu->getShape()[0], 2);
    EXPECT_EQ(loadedCpu->getShape()[1], 3);
    EXPECT_EQ(loadedCpu->getDataType(), nvinfer1::DataType::kINT64);

    std::vector<int64_t> loadedCpuData(6);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedCpuData.data(), loadedCpu->rawPointer(), 6 * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 6; ++i)
    {
        EXPECT_EQ(loadedCpuData[i], cpuData[i]);
    }

    // Verify GPU tensor
    EXPECT_NE(loadedGpu, nullptr);
    EXPECT_EQ(loadedGpu->getShape().getNumDims(), 1);
    EXPECT_EQ(loadedGpu->getShape()[0], 6);
    EXPECT_EQ(loadedGpu->getDataType(), nvinfer1::DataType::kINT64);

    std::vector<int64_t> loadedGpuData(6);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedGpuData.data(), loadedGpu->rawPointer(), 6 * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 6; ++i)
    {
        EXPECT_EQ(loadedGpuData[i], gpuData[i]);
    }

    std::filesystem::remove(testFile);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SafetensorsUtilsTest, Fp8Tensors)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::string testFile = "test_fp8_tensors.safetensors";
    if (std::filesystem::exists(testFile))
    {
        std::filesystem::remove(testFile);
    }

    // CPU tensor: 3D
    rt::Tensor cpuTensor({2, 2, 2}, rt::DeviceType::kCPU, nvinfer1::DataType::kFP8, "cpu_fp8");
    // GPU tensor: 2D
    rt::Tensor gpuTensor({3, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kFP8, "gpu_fp8");

    // Test data - FP8 E4M3 format
    // Note: FP8 values are represented as uint8_t in memory
    std::vector<uint8_t> cpuData = {0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40};
    std::vector<uint8_t> gpuData = {0x80, 0xFF, 0x7F, 0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01};

    // Copy data
    std::memcpy(cpuTensor.rawPointer(), cpuData.data(), 8 * sizeof(uint8_t));
    CUDA_CHECK(
        cudaMemcpyAsync(gpuTensor.rawPointer(), gpuData.data(), 9 * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));

    // Serialize
    std::vector<rt::Tensor> tensors;
    tensors.push_back(std::move(cpuTensor));
    tensors.push_back(std::move(gpuTensor));
    EXPECT_TRUE(rt::safetensors::saveSafetensors(testFile, tensors, stream));

    // Load and verify
    std::vector<rt::Tensor> loadedTensors;
    EXPECT_TRUE(rt::safetensors::loadSafetensors(testFile, loadedTensors, stream));
    EXPECT_EQ(loadedTensors.size(), 2);

    // Find tensors
    rt::Tensor* loadedCpu = nullptr;
    rt::Tensor* loadedGpu = nullptr;
    for (auto& tensor : loadedTensors)
    {
        if (tensor.getName() == "cpu_fp8")
            loadedCpu = &tensor;
        else if (tensor.getName() == "gpu_fp8")
            loadedGpu = &tensor;
    }

    // Verify CPU tensor
    EXPECT_NE(loadedCpu, nullptr);
    EXPECT_EQ(loadedCpu->getShape().getNumDims(), 3);
    EXPECT_EQ(loadedCpu->getShape()[0], 2);
    EXPECT_EQ(loadedCpu->getShape()[1], 2);
    EXPECT_EQ(loadedCpu->getShape()[2], 2);
    EXPECT_EQ(loadedCpu->getDataType(), nvinfer1::DataType::kFP8);

    std::vector<uint8_t> loadedCpuData(8);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedCpuData.data(), loadedCpu->rawPointer(), 8 * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_EQ(loadedCpuData[i], cpuData[i]);
    }

    // Verify GPU tensor
    EXPECT_NE(loadedGpu, nullptr);
    EXPECT_EQ(loadedGpu->getShape().getNumDims(), 2);
    EXPECT_EQ(loadedGpu->getShape()[0], 3);
    EXPECT_EQ(loadedGpu->getShape()[1], 3);
    EXPECT_EQ(loadedGpu->getDataType(), nvinfer1::DataType::kFP8);

    std::vector<uint8_t> loadedGpuData(9);
    CUDA_CHECK(cudaMemcpyAsync(
        loadedGpuData.data(), loadedGpu->rawPointer(), 9 * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 9; ++i)
    {
        EXPECT_EQ(loadedGpuData[i], gpuData[i]);
    }

    std::filesystem::remove(testFile);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(SafetensorsUtilsTest, AllDataTypesMixed)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::string testFile = "test_all_datatypes_mixed.safetensors";
    if (std::filesystem::exists(testFile))
    {
        std::filesystem::remove(testFile);
    }

    // Create tensors with all supported data types
    std::vector<rt::Tensor> tensors;

    // FLOAT tensor
    rt::Tensor floatTensor({2, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "float_tensor");
    std::vector<float> floatData = {1.1f, 2.2f, 3.3f, 4.4f};
    CUDA_CHECK(
        cudaMemcpyAsync(floatTensor.rawPointer(), floatData.data(), 4 * sizeof(float), cudaMemcpyHostToDevice, stream));
    tensors.push_back(std::move(floatTensor));

    // HALF tensor
    rt::Tensor halfTensor({2, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "half_tensor");
    std::vector<half> halfData = {__float2half(5.0f), __float2half(6.0f), __float2half(7.0f), __float2half(8.0f)};
    CUDA_CHECK(
        cudaMemcpyAsync(halfTensor.rawPointer(), halfData.data(), 4 * sizeof(half), cudaMemcpyHostToDevice, stream));
    tensors.push_back(std::move(halfTensor));

    // BF16 tensor
    rt::Tensor bf16Tensor({2, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kBF16, "bf16_tensor");
    std::vector<__nv_bfloat16> bf16Data
        = {__float2bfloat16(9.0f), __float2bfloat16(10.0f), __float2bfloat16(11.0f), __float2bfloat16(12.0f)};
    CUDA_CHECK(cudaMemcpyAsync(
        bf16Tensor.rawPointer(), bf16Data.data(), 4 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));
    tensors.push_back(std::move(bf16Tensor));

    // INT8 tensor
    rt::Tensor int8Tensor({2, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "int8_tensor");
    std::vector<int8_t> int8Data = {13, 14, 15, 16};
    CUDA_CHECK(
        cudaMemcpyAsync(int8Tensor.rawPointer(), int8Data.data(), 4 * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
    tensors.push_back(std::move(int8Tensor));

    // UINT8 tensor
    rt::Tensor uint8Tensor({2, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "uint8_tensor");
    std::vector<uint8_t> uint8Data = {17, 18, 19, 20};
    CUDA_CHECK(cudaMemcpyAsync(
        uint8Tensor.rawPointer(), uint8Data.data(), 4 * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    tensors.push_back(std::move(uint8Tensor));

    // INT32 tensor
    rt::Tensor int32Tensor({2, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "int32_tensor");
    std::vector<int32_t> int32Data = {21, 22, 23, 24};
    CUDA_CHECK(cudaMemcpyAsync(
        int32Tensor.rawPointer(), int32Data.data(), 4 * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    tensors.push_back(std::move(int32Tensor));

    // INT64 tensor
    rt::Tensor int64Tensor({2, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "int64_tensor");
    std::vector<int64_t> int64Data = {25LL, 26LL, 27LL, 28LL};
    CUDA_CHECK(cudaMemcpyAsync(
        int64Tensor.rawPointer(), int64Data.data(), 4 * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    tensors.push_back(std::move(int64Tensor));

    // FP8 tensor
    rt::Tensor fp8Tensor({2, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kFP8, "fp8_tensor");
    std::vector<uint8_t> fp8Data = {29, 30, 31, 32};
    CUDA_CHECK(
        cudaMemcpyAsync(fp8Tensor.rawPointer(), fp8Data.data(), 4 * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    tensors.push_back(std::move(fp8Tensor));

    // Serialize all tensors
    EXPECT_TRUE(rt::safetensors::saveSafetensors(testFile, tensors, stream));

    // Load and verify
    std::vector<rt::Tensor> loadedTensors;
    EXPECT_TRUE(rt::safetensors::loadSafetensors(testFile, loadedTensors, stream));
    EXPECT_EQ(loadedTensors.size(), 8);

    // Verify each tensor
    std::map<std::string, rt::Tensor*> tensorMap;
    for (auto& tensor : loadedTensors)
    {
        tensorMap[tensor.getName()] = &tensor;
    }

    // Verify FLOAT tensor
    EXPECT_NE(tensorMap["float_tensor"], nullptr);
    EXPECT_EQ(tensorMap["float_tensor"]->getDataType(), nvinfer1::DataType::kFLOAT);
    std::vector<float> loadedFloatData(4);
    CUDA_CHECK(cudaMemcpyAsync(loadedFloatData.data(), tensorMap["float_tensor"]->rawPointer(), 4 * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_TRUE(isclose(loadedFloatData[i], floatData[i], 1e-5, 1e-5));
    }

    // Verify HALF tensor
    EXPECT_NE(tensorMap["half_tensor"], nullptr);
    EXPECT_EQ(tensorMap["half_tensor"]->getDataType(), nvinfer1::DataType::kHALF);
    std::vector<half> loadedHalfData(4);
    CUDA_CHECK(cudaMemcpyAsync(loadedHalfData.data(), tensorMap["half_tensor"]->rawPointer(), 4 * sizeof(half),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_TRUE(isclose(loadedHalfData[i], halfData[i], 1e-3, 1e-3));
    }

    // Verify BF16 tensor
    EXPECT_NE(tensorMap["bf16_tensor"], nullptr);
    EXPECT_EQ(tensorMap["bf16_tensor"]->getDataType(), nvinfer1::DataType::kBF16);
    std::vector<__nv_bfloat16> loadedBf16Data(4);
    CUDA_CHECK(cudaMemcpyAsync(loadedBf16Data.data(), tensorMap["bf16_tensor"]->rawPointer(), 4 * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_TRUE(isclose(loadedBf16Data[i], bf16Data[i], 1e-3, 1e-3));
    }

    // Verify INT8 tensor
    EXPECT_NE(tensorMap["int8_tensor"], nullptr);
    EXPECT_EQ(tensorMap["int8_tensor"]->getDataType(), nvinfer1::DataType::kINT8);
    std::vector<int8_t> loadedInt8Data(4);
    CUDA_CHECK(cudaMemcpyAsync(loadedInt8Data.data(), tensorMap["int8_tensor"]->rawPointer(), 4 * sizeof(int8_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(loadedInt8Data[i], int8Data[i]);
    }

    // Verify UINT8 tensor
    EXPECT_NE(tensorMap["uint8_tensor"], nullptr);
    EXPECT_EQ(tensorMap["uint8_tensor"]->getDataType(), nvinfer1::DataType::kUINT8);
    std::vector<uint8_t> loadedUint8Data(4);
    CUDA_CHECK(cudaMemcpyAsync(loadedUint8Data.data(), tensorMap["uint8_tensor"]->rawPointer(), 4 * sizeof(uint8_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(loadedUint8Data[i], uint8Data[i]);
    }

    // Verify INT32 tensor
    EXPECT_NE(tensorMap["int32_tensor"], nullptr);
    EXPECT_EQ(tensorMap["int32_tensor"]->getDataType(), nvinfer1::DataType::kINT32);
    std::vector<int32_t> loadedInt32Data(4);
    CUDA_CHECK(cudaMemcpyAsync(loadedInt32Data.data(), tensorMap["int32_tensor"]->rawPointer(), 4 * sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(loadedInt32Data[i], int32Data[i]);
    }

    // Verify INT64 tensor
    EXPECT_NE(tensorMap["int64_tensor"], nullptr);
    EXPECT_EQ(tensorMap["int64_tensor"]->getDataType(), nvinfer1::DataType::kINT64);
    std::vector<int64_t> loadedInt64Data(4);
    CUDA_CHECK(cudaMemcpyAsync(loadedInt64Data.data(), tensorMap["int64_tensor"]->rawPointer(), 4 * sizeof(int64_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(loadedInt64Data[i], int64Data[i]);
    }

    // Verify FP8 tensor
    EXPECT_NE(tensorMap["fp8_tensor"], nullptr);
    EXPECT_EQ(tensorMap["fp8_tensor"]->getDataType(), nvinfer1::DataType::kFP8);
    std::vector<uint8_t> loadedFp8Data(4);
    CUDA_CHECK(cudaMemcpyAsync(loadedFp8Data.data(), tensorMap["fp8_tensor"]->rawPointer(), 4 * sizeof(uint8_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(loadedFp8Data[i], fp8Data[i]);
    }

    std::filesystem::remove(testFile);
    CUDA_CHECK(cudaStreamDestroy(stream));
}
