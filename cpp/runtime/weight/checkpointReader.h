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

#include "common/mmapReader.h"
#include "common/tensor.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Read individual tensors from Hugging Face safetensors or indexed PyTorch
//! ZIP checkpoint ranges.
//!
//! Shards are memory-mapped on first use. The reader is only used while model
//! weights are prepared during runtime initialization and is destroyed before
//! the first inference enqueue.
class CheckpointReader
{
public:
    struct TensorLocation
    {
        std::string name;
        std::filesystem::path file;
        std::string dtype;
        Coords shape;
        size_t offset{0};
        size_t bytes{0};
    };

    //! A checkpoint tensor has separate host and CUDA aliases into one mapped
    //! file range. This is intentionally not rt::Tensor: Tensor has one mutable
    //! address and one device type, so wrapping this range would discard either
    //! the const host view used for metadata checks or the CUDA alias used by
    //! transform kernels.
    struct View
    {
        trt_edgellm::DataType dtype{};
        Coords shape;
        uint8_t const* data{nullptr};
        uint8_t const* deviceData{nullptr};
        size_t bytes{0};
    };

    explicit CheckpointReader(std::filesystem::path const& directory);
    CheckpointReader(std::filesystem::path const& directory, std::vector<TensorLocation> const& explicitTensorLocations,
        std::filesystem::path const& singleFile = {});
    ~CheckpointReader() noexcept;

    //! CUDA-map the page ranges containing the requested tensors. Ranges are
    //! merged per shard and must be registered before find() returns a device
    //! pointer.
    void registerTensors(std::vector<std::string> const& names);

    //! CUDA-map one byte range inside a checkpoint tensor.
    //!
    //! No other ranges may be registered while the returned view is used.
    //! unregisterTensors() releases the mapping after its consumer stream has
    //! completed.
    View registerTensorRange(std::string const& name, size_t offset, size_t bytes);

    //! Release every CUDA registration created by registerTensors().
    //!
    //! Runtime initialization calls this after each completed output binding so
    //! source checkpoint pages never accumulate beside the final weight arena.
    void unregisterTensors() noexcept;

    //! Evict no-longer-needed source pages from this process's file mappings.
    void discardTensors(std::vector<std::string> const& names) noexcept;
    void discardTensorRange(std::string const& name, size_t offset, size_t bytes) noexcept;

    size_t peakRegisteredBytes() const noexcept
    {
        return mPeakRegisteredBytes;
    }

    size_t registeredBytes() const noexcept
    {
        return mRegisteredBytes;
    }

    std::chrono::nanoseconds registrationTime() const noexcept
    {
        return mRegistrationTime;
    }

    //! Find a tensor without requiring CUDA page registration. The returned
    //! view has a host pointer and a null device pointer.
    bool findHost(std::string const& name, View& view) const;

    //! Locate the payload for bounded file reads without registering it with CUDA.
    TensorLocation locate(std::string const& name) const;

    bool find(std::string const& name, View& view) const;

private:
    struct TensorInfo
    {
        trt_edgellm::DataType dtype{};
        Coords shape;
        size_t offset{0};
        size_t bytes{0};
    };

    struct Shard
    {
        struct Registration
        {
            size_t offset{0};
            size_t bytes{0};
            uint8_t const* deviceData{nullptr};
        };

        std::unique_ptr<file_io::MmapReader> file;
        size_t dataOffset{0};
        std::unordered_map<std::string, TensorInfo> tensors;
        std::vector<Registration> registrations;
    };

    Shard& openShard(std::filesystem::path const& path) const;
    TensorInfo const& tensorInfo(std::string const& name, Shard const& shard) const;

    std::unordered_map<std::string, std::filesystem::path> mTensorShards;
    std::unordered_map<std::string, TensorInfo> mExplicitTensors;
    std::unordered_set<std::string> mRawShards;
    mutable std::unordered_map<std::string, Shard> mOpenShards;
    size_t mRegisteredBytes{0};
    size_t mPeakRegisteredBytes{0};
    std::chrono::nanoseconds mRegistrationTime{0};
};

} // namespace rt
} // namespace trt_edgellm
