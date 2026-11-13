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

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Immutable embedding row layout. Opening validates metadata, not payload digests.
class PleTable
{
public:
    enum class DType
    {
        kFP8,
        kFP16,
        kBF16,
        kFP32
    };

    struct Sample
    {
        int64_t offset{};
        std::string data;
    };

    //! One checkpoint tensor holding a contiguous row range. Sample offsets are relative to it.
    struct ShardSource
    {
        std::string tensorKey;
        int64_t rows{};
        int64_t bytes{};
        std::vector<Sample> samples;
        std::filesystem::path file;
        int64_t offset{};
    };

    //! Original checkpoint tensor contract. An explicit location describes indexed PyTorch payloads.
    //! A table split across tensors sets ``shards`` instead of ``tensorKey``, in row order.
    struct Source
    {
        std::string tensorKey;
        DType dtype{DType::kFP16};
        int64_t bytes{};
        double scale{1.0};
        std::vector<Sample> samples;
        std::filesystem::path file;
        int64_t offset{};
        std::vector<ShardSource> shards;
    };

    struct Shard
    {
        std::filesystem::path path;
        int64_t rowBegin{};
        int64_t rowCount{};
        int64_t byteOffset{};
        int64_t byteLength{};
    };

    struct RowLocation
    {
        size_t shardIndex{};
        int64_t byteOffset{};
    };

    //! Expected identity and scale come from the engine's bound artifact metadata.
    static PleTable open(std::filesystem::path const& manifest, std::string const& expectedTableId,
        double expectedScale, std::filesystem::path const& dataFile = {});

    //! Resolve the source tensors from a checkpoint or explicit file and validate their metadata and
    //! bounded samples. ``poolDtype``/``poolScale`` describe the rows the pool will hold, which differ
    //! from the source when the fill converts; the defaults are the cast-to-F16 contract.
    static PleTable fromSource(Source const& source, int64_t rowCount, int64_t rowWidth,
        std::filesystem::path const& checkpointDir, std::filesystem::path const& dataFile = {},
        DType poolDtype = DType::kFP16, double poolScale = 1.0);

    //! Translate a valid global row into an exact file range without reading payload bytes.
    RowLocation locate(int64_t row) const;
    int64_t rowCount() const noexcept
    {
        return mRowCount;
    }
    int64_t rowWidth() const noexcept
    {
        return mRowWidth;
    }
    int64_t rowBytes() const noexcept
    {
        return mRowBytes;
    }
    DType dtype() const noexcept
    {
        return mDtype;
    }
    DType sourceDtype() const noexcept
    {
        return mSourceDtype;
    }
    int64_t sourceRowBytes() const noexcept
    {
        return mSourceRowBytes;
    }
    double sourceScale() const noexcept
    {
        return mSourceScale;
    }
    double scale() const noexcept
    {
        return mScale;
    }
    std::string const& tableId() const noexcept
    {
        return mTableId;
    }
    std::vector<Shard> const& shards() const noexcept
    {
        return mShards;
    }

private:
    PleTable() = default;
    int64_t mRowCount{};
    int64_t mRowWidth{};
    int64_t mRowBytes{};
    DType mDtype{};
    DType mSourceDtype{};
    int64_t mSourceRowBytes{};
    double mSourceScale{1.0};
    double mScale{};
    std::string mTableId;
    std::vector<Shard> mShards;
};

} // namespace rt
} // namespace trt_edgellm
