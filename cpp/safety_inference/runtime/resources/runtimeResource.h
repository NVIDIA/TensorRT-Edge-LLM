/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "containerStorage.h"
#include "safety_inference/runtime/model_inference/modelInferenceTypes.h"

#include <NvInferSafeRuntime.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace trt_edgellm::safety
{

//! One declaration owner is a capability or a model-local-data factory; zero is invalid.
using ResourceOwnerId = std::uint32_t;
using RuntimeResourceKey = std::uint32_t;

enum class RuntimeMemoryKind
{
    kPAGEABLE,
    kPINNED,
    kDEVICE,
};

struct RuntimeMemoryRequest
{
    RuntimeResourceKey key{0};
    std::size_t sizeInBytes{0};
    RuntimeMemoryKind kind{RuntimeMemoryKind::kPAGEABLE};
};

struct RuntimeMemoryView
{
    void* data{nullptr};
    std::size_t sizeInBytes{0};
    RuntimeMemoryKind kind{RuntimeMemoryKind::kPAGEABLE};
};

struct RuntimeTensorProfile
{
    int32_t profileId{-1};
    nvinfer1::Dims shape{};
    std::size_t sizeInBytes{0};
};

//! Immutable linear-binding metadata obtained from engine queries during Initialization.
struct RuntimeTensorDesc
{
    std::string name{};
    nvinfer1::DataType dataType{nvinfer1::DataType::kFLOAT};
    nvinfer1::TensorIOMode ioMode{nvinfer1::TensorIOMode::kNONE};
    std::vector<RuntimeTensorProfile> profiles{};
};

//! Both the descriptor and payload outlive this borrowed view.
struct RuntimeTensorView
{
    RuntimeTensorDesc const* desc{nullptr};
    RuntimeMemoryView memory{};
};

//! Checks static profile shapes, byte sizes and duplicate profile IDs without allocation.
SafetyStatus validateTensorDesc(RuntimeTensorDesc const& desc) noexcept;

enum class RuntimeBorrowedResourceKind
{
    kIO_BINDING,
    kMODEL_LOCAL_DATA,
    kCUDA_STREAM,
};

//! Stage identifiers use immutable model-definition storage; binding names are owned by the declaration.
//! Stream requests borrow the stage's explicit main stream.
struct RuntimeBorrowedResourceRequest
{
    RuntimeResourceKey key{0};
    RuntimeBorrowedResourceKind kind{RuntimeBorrowedResourceKind::kIO_BINDING};
    LMStageId stage{};
    std::size_t resourceSlot{0};
    std::string binding{};
};

struct ResourceDeclaration
{
    ResourceOwnerId owner{0};
    std::vector<RuntimeMemoryRequest> memory{};
    std::vector<RuntimeResourceKey> streams{};
    std::optional<SafetyContainerStorageRequest> containerStorage{};
    std::vector<RuntimeBorrowedResourceRequest> borrowed{};
    std::vector<RuntimeResourceKey> events{};
};

class SafetyModelLocalData;

struct RuntimeBorrowedResource
{
    RuntimeTensorView const* tensor{nullptr};
    cudaStream_t stream{nullptr};
    SafetyModelLocalData* modelLocalData{nullptr};
};

//! Initialization adapter over prepared Runner resources and runtime-owned model-local objects.
//! Returned resources outlive their borrowers; resolving a request acquires no ownership.
class BorrowedResourceResolver
{
public:
    virtual ~BorrowedResourceResolver() = default;
    virtual SafetyStatus resolve(
        RuntimeBorrowedResourceRequest const& request, RuntimeBorrowedResource& output) const noexcept = 0;
};

class DeclaredResourceManager;

//! Owner-scoped lookup with immutable metadata and mutable payloads. The manager outlives this view.
class PreparedResources
{
public:
    PreparedResources() = default;
    PreparedResources(DeclaredResourceManager const& manager, ResourceOwnerId owner) noexcept
        : mManager(&manager)
        , mOwner(owner)
    {
    }
    //! Factory/capability setup must use its own owner during Initialization.
    SafetyStatus validateInitialization(ResourceOwnerId owner) const noexcept;
    SafetyStatus memory(RuntimeResourceKey key, RuntimeMemoryView& output) const noexcept;
    SafetyStatus tensor(RuntimeResourceKey key, RuntimeTensorView const*& output) const noexcept;
    SafetyStatus stream(RuntimeResourceKey key, cudaStream_t& output) const noexcept;
    SafetyStatus event(RuntimeResourceKey key, cudaEvent_t& output) const noexcept;
    SafetyStatus modelLocalData(RuntimeResourceKey key, SafetyModelLocalData*& output) const noexcept;
    SafetyStatus containerStorage(SafetyContainerStorage*& output) const noexcept;

private:
    DeclaredResourceManager const* mManager{nullptr};
    ResourceOwnerId mOwner{0};
};

} // namespace trt_edgellm::safety
