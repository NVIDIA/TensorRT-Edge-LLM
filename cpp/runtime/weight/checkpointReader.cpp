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

#include "runtime/weight/checkpointReader.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

namespace trt_edgellm
{
namespace rt
{
namespace
{

using Json = nlohmann::json;

bool isSafetensorsIndex(std::filesystem::path const& path)
{
    std::string const name = path.filename().string();
    return name.size() >= 23 && name.compare(name.size() - 23, 23, ".safetensors.index.json") == 0;
}

int discardMappedPages(void* address, size_t length)
{
#if defined(__QNXNTO__)
    return posix_madvise(address, length, POSIX_MADV_DONTNEED);
#else
    return madvise(address, length, MADV_DONTNEED);
#endif
}

} // namespace

CheckpointReader::CheckpointReader(std::filesystem::path const& directory)
    : CheckpointReader(directory, {})
{
}

CheckpointReader::CheckpointReader(
    std::filesystem::path const& directory, std::vector<TensorLocation> const& explicitTensorLocations)
{
    ELLM_CHECK(std::filesystem::is_directory(directory), "Checkpoint directory does not exist: " + directory.string());

    std::vector<std::filesystem::path> indices;
    std::vector<std::filesystem::path> shards;
    for (auto const& entry : std::filesystem::directory_iterator(directory))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        if (isSafetensorsIndex(entry.path()))
        {
            indices.push_back(entry.path());
        }
        else if (entry.path().extension() == ".safetensors")
        {
            shards.push_back(entry.path());
        }
    }
    std::sort(indices.begin(), indices.end());
    std::sort(shards.begin(), shards.end());

    auto const preferred = directory / "model.safetensors.index.json";
    auto const indexIt = std::find(indices.begin(), indices.end(), preferred);
    if (indexIt != indices.end())
    {
        std::iter_swap(indices.begin(), indexIt);
    }

    if (!indices.empty())
    {
        std::ifstream file(indices.front());
        ELLM_CHECK(file.is_open(), "Failed to open checkpoint index: " + indices.front().string());
        Json const index = Json::parse(file);
        ELLM_CHECK(index.contains("weight_map") && index["weight_map"].is_object(),
            "Checkpoint index has no weight_map: " + indices.front().string());
        for (auto const& [name, shard] : index["weight_map"].items())
        {
            ELLM_CHECK(shard.is_string(), "Checkpoint weight_map entries must name a shard");
            mTensorShards.emplace(name, directory / shard.get<std::string>());
        }
    }
    else
    {
        for (auto const& path : shards)
        {
            Shard const& shard = openShard(path);
            for (auto const& [name, _] : shard.tensors)
            {
                mTensorShards.emplace(name, path);
            }
        }
    }

    for (TensorLocation const& location : explicitTensorLocations)
    {
        ELLM_CHECK(!location.name.empty(), "Explicit checkpoint tensor name is empty");
        ELLM_CHECK(!location.file.empty() && !location.file.is_absolute(),
            "Explicit checkpoint tensor file must be relative: " + location.file.string());
        for (auto const& part : location.file)
        {
            ELLM_CHECK(part != "..", "Explicit checkpoint tensor file escapes its checkpoint directory");
        }
        ELLM_CHECK(
            location.shape.volume() > 0 && location.bytes > 0, "Explicit checkpoint tensor is empty: " + location.name);

        std::filesystem::path const path = directory / location.file;
        ELLM_CHECK(std::filesystem::is_regular_file(path), "Checkpoint shard does not exist: " + path.string());
        size_t const fileBytes = static_cast<size_t>(std::filesystem::file_size(path));
        ELLM_CHECK(location.offset <= fileBytes && location.bytes <= fileBytes - location.offset,
            "Explicit checkpoint tensor range exceeds " + path.string() + ": " + location.name);

        auto const [shardIt, inserted] = mTensorShards.emplace(location.name, path);
        ELLM_CHECK(inserted || shardIt->second == path, "Conflicting checkpoint tensor location: " + location.name);
        TensorInfo const info{
            safetensors::dataTypeFromString(location.dtype), location.shape, location.offset, location.bytes};
        auto const [tensorIt, tensorInserted] = mExplicitTensors.emplace(location.name, info);
        ELLM_CHECK(tensorInserted
                || (tensorIt->second.dtype == info.dtype && tensorIt->second.shape == info.shape
                    && tensorIt->second.offset == info.offset && tensorIt->second.bytes == info.bytes),
            "Conflicting explicit checkpoint tensor location: " + location.name);
        mRawShards.insert(path.string());
    }

    ELLM_CHECK(!mTensorShards.empty(), "No safetensors or indexed PyTorch checkpoint found in " + directory.string());
    LOG_DEBUG("Indexed %zu checkpoint tensors from %s", mTensorShards.size(), directory.string().c_str());
}

CheckpointReader::~CheckpointReader() noexcept
{
    unregisterTensors();
}

void CheckpointReader::unregisterTensors() noexcept
{
    size_t registeredBytes = 0;
    for (auto& [_, shard] : mOpenShards)
    {
        std::vector<Shard::Registration> failed;
        failed.reserve(shard.registrations.size());
        for (Shard::Registration const& registration : shard.registrations)
        {
            auto* hostData = const_cast<int8_t*>(shard.file->getByteData()) + registration.offset;
            cudaError_t const status = cudaHostUnregister(hostData);
            if (status != cudaSuccess)
            {
                LOG_WARNING("Failed to unregister checkpoint shard mapping: %s", cudaGetErrorString(status));
                failed.push_back(registration);
                registeredBytes += registration.bytes;
            }
        }
        shard.registrations = std::move(failed);
    }
    mRegisteredBytes = registeredBytes;
}

void CheckpointReader::discardTensors(std::vector<std::string> const& names) noexcept
{
    for (std::string const& name : names)
    {
        try
        {
            auto const location = mTensorShards.find(name);
            if (location == mTensorShards.end())
            {
                continue;
            }
            auto const shardIt = mOpenShards.find(location->second.string());
            if (shardIt == mOpenShards.end())
            {
                continue;
            }
            TensorInfo const& info = tensorInfo(name, shardIt->second);
            discardTensorRange(name, 0, info.bytes);
        }
        catch (std::exception const& error)
        {
            LOG_WARNING("Failed to release checkpoint pages for %s: %s", name.c_str(), error.what());
        }
        catch (...)
        {
            LOG_WARNING("Failed to release checkpoint pages for %s", name.c_str());
        }
    }
}

void CheckpointReader::discardTensorRange(std::string const& name, size_t offset, size_t bytes) noexcept
{
    try
    {
        auto const location = mTensorShards.find(name);
        if (location == mTensorShards.end() || bytes == 0)
        {
            return;
        }
        auto const shardIt = mOpenShards.find(location->second.string());
        if (shardIt == mOpenShards.end())
        {
            return;
        }
        Shard& shard = shardIt->second;
        TensorInfo const& info = tensorInfo(name, shard);
        if (offset > info.bytes || bytes > info.bytes - offset)
        {
            LOG_WARNING("Checkpoint discard range exceeds tensor %s", name.c_str());
            return;
        }
        long const pageSizeResult = sysconf(_SC_PAGESIZE);
        if (pageSizeResult <= 0)
        {
            LOG_WARNING("Failed to query system page size while releasing checkpoint pages");
            return;
        }
        size_t const pageSize = static_cast<size_t>(pageSizeResult);
        size_t const rangeBegin = shard.dataOffset + info.offset + offset;
        size_t const begin = rangeBegin - rangeBegin % pageSize;
        size_t const rangeEnd = rangeBegin + bytes;
        size_t end = rangeEnd;
        size_t const remainder = rangeEnd % pageSize;
        if (remainder != 0)
        {
            end += std::min(pageSize - remainder, shard.file->getSize() - rangeEnd);
        }
        if (begin < end && discardMappedPages(const_cast<int8_t*>(shard.file->getByteData()) + begin, end - begin) != 0)
        {
            LOG_WARNING("Failed to release checkpoint pages for %s", name.c_str());
        }
    }
    catch (std::exception const& error)
    {
        LOG_WARNING("Failed to release checkpoint pages for %s: %s", name.c_str(), error.what());
    }
    catch (...)
    {
        LOG_WARNING("Failed to release checkpoint pages for %s", name.c_str());
    }
}

CheckpointReader::Shard& CheckpointReader::openShard(std::filesystem::path const& path) const
{
    std::string const key = path.string();
    auto const found = mOpenShards.find(key);
    if (found != mOpenShards.end())
    {
        return found->second;
    }

    Shard shard;
    shard.file = std::make_unique<file_io::MmapReader>(path, file_io::MmapReader::Mode::kCopyOnWrite);
    if (mRawShards.count(key) != 0)
    {
        return mOpenShards.emplace(key, std::move(shard)).first->second;
    }

    safetensors::FileMetadata const metadata
        = safetensors::parseMetadata(shard.file->getData(), shard.file->getSize(), key);
    shard.dataOffset = metadata.dataOffset;
    for (safetensors::TensorMetadata const& tensor : metadata.tensors)
    {
        shard.tensors.emplace(tensor.name, TensorInfo{tensor.dataType, tensor.shape, tensor.offset, tensor.bytes});
    }

    return mOpenShards.emplace(key, std::move(shard)).first->second;
}

CheckpointReader::TensorInfo const& CheckpointReader::tensorInfo(std::string const& name, Shard const& shard) const
{
    auto const explicitTensor = mExplicitTensors.find(name);
    if (explicitTensor != mExplicitTensors.end())
    {
        return explicitTensor->second;
    }
    auto const tensor = shard.tensors.find(name);
    ELLM_CHECK(tensor != shard.tensors.end(), "Checkpoint index points to a shard without tensor " + name);
    return tensor->second;
}

void CheckpointReader::registerTensors(std::vector<std::string> const& names)
{
    auto const registrationStart = std::chrono::steady_clock::now();
    long const pageSizeResult = sysconf(_SC_PAGESIZE);
    ELLM_CHECK(pageSizeResult > 0, "Failed to query system page size");
    size_t const pageSize = static_cast<size_t>(pageSizeResult);
    ELLM_CHECK((pageSize & (pageSize - 1)) == 0, "System page size is not a power of two");

    using Range = std::pair<size_t, size_t>;
    std::unordered_map<std::string, std::vector<Range>> rangesByShard;
    for (std::string const& name : names)
    {
        auto const location = mTensorShards.find(name);
        if (location == mTensorShards.end())
        {
            continue;
        }
        Shard& shard = openShard(location->second);
        TensorInfo const& info = tensorInfo(name, shard);
        if (info.bytes == 0)
        {
            continue;
        }
        ELLM_CHECK(info.offset <= std::numeric_limits<size_t>::max() - shard.dataOffset,
            "Checkpoint tensor offset overflow: " + name);
        size_t const tensorBegin = shard.dataOffset + info.offset;
        ELLM_CHECK(info.bytes <= shard.file->getSize() - tensorBegin, "Checkpoint tensor range overflow: " + name);
        size_t const tensorEnd = tensorBegin + info.bytes;
        size_t const begin = tensorBegin & ~(pageSize - 1);
        size_t end = tensorEnd;
        size_t const remainder = tensorEnd & (pageSize - 1);
        if (remainder != 0)
        {
            end += std::min(pageSize - remainder, shard.file->getSize() - tensorEnd);
        }
        rangesByShard[location->second.string()].emplace_back(begin, end);
    }

    for (auto& [path, ranges] : rangesByShard)
    {
        Shard& shard = openShard(path);
        ELLM_CHECK(shard.registrations.empty(), "Checkpoint shard ranges were already registered: " + path);
        std::sort(ranges.begin(), ranges.end());
        std::vector<Range> merged;
        for (Range const& range : ranges)
        {
            if (merged.empty() || range.first > merged.back().second)
            {
                merged.push_back(range);
            }
            else
            {
                merged.back().second = std::max(merged.back().second, range.second);
            }
        }

        shard.registrations.reserve(merged.size());
        for (Range const& range : merged)
        {
            ELLM_CHECK(range.first < range.second, "Empty checkpoint registration range");
            auto* hostData = const_cast<int8_t*>(shard.file->getByteData()) + range.first;
            size_t const bytes = range.second - range.first;
            CUDA_CHECK(cudaHostRegister(hostData, bytes, cudaHostRegisterMapped));
            void* deviceData{nullptr};
            cudaError_t const mappingStatus = cudaHostGetDevicePointer(&deviceData, hostData, 0);
            if (mappingStatus != cudaSuccess)
            {
                static_cast<void>(cudaHostUnregister(hostData));
                CUDA_CHECK(mappingStatus);
            }
            shard.registrations.push_back(
                Shard::Registration{range.first, bytes, static_cast<uint8_t const*>(deviceData)});
            mRegisteredBytes += bytes;
            mPeakRegisteredBytes = std::max(mPeakRegisteredBytes, mRegisteredBytes);
        }
    }
    mRegistrationTime += std::chrono::steady_clock::now() - registrationStart;
}

CheckpointReader::View CheckpointReader::registerTensorRange(std::string const& name, size_t offset, size_t bytes)
{
    ELLM_CHECK(mRegisteredBytes == 0, "Checkpoint source ranges are already CUDA-mapped");
    auto const location = mTensorShards.find(name);
    ELLM_CHECK(location != mTensorShards.end(), "Checkpoint tensor not found: " + name);
    Shard& shard = openShard(location->second);
    TensorInfo const& info = tensorInfo(name, shard);
    ELLM_CHECK(offset <= info.bytes && bytes > 0 && bytes <= info.bytes - offset,
        "Checkpoint registration range exceeds tensor " + name);

    long const pageSizeResult = sysconf(_SC_PAGESIZE);
    ELLM_CHECK(pageSizeResult > 0, "Failed to query system page size");
    size_t const pageSize = static_cast<size_t>(pageSizeResult);
    ELLM_CHECK((pageSize & (pageSize - 1)) == 0, "System page size is not a power of two");
    size_t const tensorBegin = shard.dataOffset + info.offset;
    size_t const rangeBegin = tensorBegin + offset;
    size_t const begin = rangeBegin & ~(pageSize - 1);
    size_t const rangeEnd = rangeBegin + bytes;
    size_t const end = std::min((rangeEnd + pageSize - 1) & ~(pageSize - 1), shard.file->getSize());
    ELLM_CHECK(begin < end, "Empty checkpoint registration range: " + name);

    auto* hostData = const_cast<int8_t*>(shard.file->getByteData()) + begin;
    size_t const registeredBytes = end - begin;
    auto const registrationStart = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaHostRegister(hostData, registeredBytes, cudaHostRegisterMapped));
    void* deviceData{nullptr};
    cudaError_t const mappingStatus = cudaHostGetDevicePointer(&deviceData, hostData, 0);
    if (mappingStatus != cudaSuccess)
    {
        static_cast<void>(cudaHostUnregister(hostData));
        CUDA_CHECK(mappingStatus);
    }
    shard.registrations.push_back(Shard::Registration{begin, registeredBytes, static_cast<uint8_t const*>(deviceData)});
    mRegisteredBytes = registeredBytes;
    mPeakRegisteredBytes = std::max(mPeakRegisteredBytes, mRegisteredBytes);
    mRegistrationTime += std::chrono::steady_clock::now() - registrationStart;
    size_t const viewOffset = rangeBegin - begin;
    return View{info.dtype, info.shape, reinterpret_cast<uint8_t const*>(hostData) + viewOffset,
        static_cast<uint8_t const*>(deviceData) + viewOffset, bytes};
}

bool CheckpointReader::findHost(std::string const& name, View& view) const
{
    auto const location = mTensorShards.find(name);
    if (location == mTensorShards.end())
    {
        return false;
    }
    Shard const& shard = openShard(location->second);
    TensorInfo const& info = tensorInfo(name, shard);
    size_t const offset = shard.dataOffset + info.offset;
    view = View{info.dtype, info.shape, reinterpret_cast<uint8_t const*>(shard.file->getByteData()) + offset, nullptr,
        info.bytes};
    return true;
}

bool CheckpointReader::find(std::string const& name, View& view) const
{
    if (!findHost(name, view))
    {
        return false;
    }
    if (view.bytes > 0)
    {
        auto const location = mTensorShards.find(name);
        ELLM_CHECK(location != mTensorShards.end(), "Checkpoint tensor disappeared while resolving: " + name);
        Shard const& shard = openShard(location->second);
        TensorInfo const& info = tensorInfo(name, shard);
        size_t const offset = shard.dataOffset + info.offset;
        auto const registration = std::find_if(shard.registrations.begin(), shard.registrations.end(),
            [offset, &info](Shard::Registration const& candidate) {
                return candidate.offset <= offset && offset + info.bytes <= candidate.offset + candidate.bytes;
            });
        ELLM_CHECK(registration != shard.registrations.end(), "Checkpoint tensor was not CUDA-mapped: " + name);
        view.deviceData = registration->deviceData + (offset - registration->offset);
    }
    return true;
}

} // namespace rt
} // namespace trt_edgellm
