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

#include "pleTable.h"
#include "common/safetensorsUtils.h"
#include "runtime/weight/checkpointReader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{
namespace
{
void require(bool condition, char const* message)
{
    if (!condition)
    {
        throw std::invalid_argument(message);
    }
}

int64_t integer(nlohmann::json const& object, char const* key)
{
    auto const& value = object.at(key);
    require(value.is_number_integer(), "PLE descriptor field must be an integer");
    if (value.is_number_unsigned())
    {
        require(value.get<uint64_t>() <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "PLE descriptor integer overflow");
    }
    return value.get<int64_t>();
}

bool isDigest(std::string const& digest)
{
    return digest.size() == 64 && std::all_of(digest.begin(), digest.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

char const* sourceTypeName(PleTable::DType type)
{
    switch (type)
    {
    case PleTable::DType::kFP8: return "F8_E4M3";
    case PleTable::DType::kFP16: return "F16";
    case PleTable::DType::kBF16: return "BF16";
    case PleTable::DType::kFP32: return "F32";
    default: throw std::invalid_argument("Unsupported PLE checkpoint dtype");
    }
}

int64_t elementBytesOf(PleTable::DType type)
{
    switch (type)
    {
    case PleTable::DType::kFP8: return 1;
    case PleTable::DType::kFP32: return 4;
    default: return 2;
    }
}

uint8_t hexDigit(char c)
{
    if (c >= '0' && c <= '9')
    {
        return static_cast<uint8_t>(c - '0');
    }
    require(c >= 'a' && c <= 'f', "Malformed PLE identity sample");
    return static_cast<uint8_t>(c - 'a' + 10);
}
} // namespace

PleTable PleTable::open(std::filesystem::path const& manifest, std::string const& expectedTableId, double expectedScale,
    std::filesystem::path const& dataFile)
{
    constexpr uintmax_t kMAX_MANIFEST_BYTES{1U << 20};
    auto const manifestBytes = std::filesystem::file_size(manifest);
    require(manifestBytes > 0 && manifestBytes <= kMAX_MANIFEST_BYTES, "Invalid PLE descriptor length");
    std::ifstream input(manifest, std::ios::binary);
    std::string bytes(static_cast<size_t>(manifestBytes), '\0');
    require(static_cast<bool>(input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))),
        "Cannot read PLE descriptor");
    auto const descriptor = nlohmann::json::parse(bytes);
    require(integer(descriptor, "schema_version") == 1, "Unsupported PLE descriptor version");
    PleTable table;
    auto const dtype = descriptor.at("dtype").get<std::string>();
    require(dtype == "F8_E4M3" || dtype == "F16" || dtype == "BF16", "Unsupported PLE dtype");
    table.mDtype = dtype == "F8_E4M3" ? DType::kFP8 : (dtype == "F16" ? DType::kFP16 : DType::kBF16);
    int64_t const elementBytes = table.mDtype == DType::kFP8 ? 1 : 2;
    table.mRowWidth = integer(descriptor, "row_width");
    require(table.mRowWidth > 0 && table.mRowWidth <= std::numeric_limits<int64_t>::max() / elementBytes,
        "Invalid PLE row width");
    table.mRowBytes = table.mRowWidth * elementBytes;
    table.mSourceRowBytes = table.mRowBytes;
    table.mSourceDtype = table.mDtype;
    table.mTableId = descriptor.at("table_id").get<std::string>();
    require(
        table.mTableId.rfind("sha256:", 0) == 0 && isDigest(table.mTableId.substr(7)), "Invalid PLE table identity");
    require(table.mTableId == expectedTableId, "PLE table identity does not match engine");
    table.mScale = descriptor.at("scale").get<double>();
    require(std::isfinite(table.mScale) && table.mScale > 0 && table.mScale == expectedScale,
        "PLE scale does not match engine");
    require(static_cast<double>(static_cast<float>(table.mScale)) == table.mScale,
        "PLE scale must be exactly representable as float32");
    require(table.mDtype == DType::kFP8 || table.mScale == 1.0, "Native PLE storage requires unit scale");
    table.mRowCount = integer(descriptor, "row_count");
    require(table.mRowCount > 0 && table.mRowCount <= std::numeric_limits<int64_t>::max() / table.mRowBytes,
        "Invalid PLE row count");
    auto const& shards = descriptor.at("shards");
    require(shards.is_array() && !shards.empty(), "PLE table has no shards");
    require(dataFile.empty() || shards.size() == 1, "Explicit PLE file requires a single-shard artifact");
    int64_t nextRow{0};
    for (auto const& entry : shards)
    {
        Shard shard;
        shard.rowBegin = integer(entry, "row_begin");
        shard.rowCount = integer(entry, "row_count");
        shard.byteOffset = integer(entry, "byte_offset");
        shard.byteLength = integer(entry, "byte_length");
        require(shard.rowBegin == nextRow && shard.rowCount > 0 && shard.rowCount <= table.mRowCount - nextRow,
            "PLE shard rows have a gap, overlap, or invalid range");
        require(shard.byteLength == shard.rowCount * table.mRowBytes && shard.byteOffset >= 0,
            "Invalid PLE shard byte range");
        require(isDigest(entry.at("sha256").get<std::string>()), "Invalid PLE shard digest");
        std::filesystem::path const relative = entry.at("file").get<std::string>();
        require(!relative.empty() && !relative.has_root_path(), "PLE shard path must be relative");
        for (auto const& component : relative)
        {
            require(component != "..", "PLE shard path escapes artifact directory");
        }
        shard.path = dataFile.empty() ? manifest.parent_path() / relative : std::filesystem::absolute(dataFile);
        require(std::filesystem::is_regular_file(shard.path), "Missing PLE shard file");
        auto const fileBytes = std::filesystem::file_size(shard.path);
        require(static_cast<uintmax_t>(shard.byteOffset) <= fileBytes
                && static_cast<uintmax_t>(shard.byteLength) <= fileBytes - static_cast<uintmax_t>(shard.byteOffset),
            "PLE shard file is truncated");
        require(shard.byteOffset <= std::numeric_limits<int64_t>::max() - shard.byteLength,
            "PLE shard file offset overflow");
        nextRow += shard.rowCount;
        table.mShards.push_back(std::move(shard));
    }
    require(nextRow == table.mRowCount, "PLE shard rows do not cover table");
    return table;
}

PleTable PleTable::fromSource(Source const& source, int64_t rowCount, int64_t rowWidth,
    std::filesystem::path const& checkpointDir, std::filesystem::path const& dataFile, DType poolDtype,
    double poolScale)
{
    // A single-tensor source is one shard, so the sharded path is the only path below.
    std::vector<ShardSource> shards;
    if (source.shards.empty())
    {
        require(!source.tensorKey.empty(), "PLE source tensor key is missing");
        shards.push_back({source.tensorKey, rowCount, source.bytes, source.samples, source.file, source.offset});
    }
    else
    {
        require(source.tensorKey.empty(), "PLE source carries both a tensor key and shards");
        shards = source.shards;
    }

    // Source geometry and scales, plus the directory the shard files resolve against.
    char const* const dtype = sourceTypeName(source.dtype);
    int64_t const elementBytes = elementBytesOf(source.dtype);
    require(rowCount > 0 && rowWidth > 0 && rowWidth <= std::numeric_limits<int64_t>::max() / elementBytes,
        "Invalid PLE source geometry");
    int64_t const rowBytes = rowWidth * elementBytes;
    require(rowCount <= std::numeric_limits<int64_t>::max() / rowBytes && source.bytes == rowCount * rowBytes,
        "PLE source byte count does not match pool geometry");
    require(std::isfinite(source.scale) && source.scale > 0
            && static_cast<double>(static_cast<float>(source.scale)) == source.scale,
        "Invalid PLE source conversion scale");
    require(
        std::isfinite(poolScale) && poolScale > 0 && static_cast<double>(static_cast<float>(poolScale)) == poolScale,
        "Invalid PLE pool scale");
    int64_t const poolRowBytes = rowWidth * elementBytesOf(poolDtype);
    require(!dataFile.empty() || !checkpointDir.empty(), "PLE requires --checkpointDir or an explicit --pleFile");
    auto const overrideFile = dataFile.empty() ? std::filesystem::path{} : std::filesystem::absolute(dataFile);
    auto const directory = overrideFile.empty() ? std::filesystem::absolute(checkpointDir) : overrideFile.parent_path();

    // Only a shard carrying its own file needs an explicit location; the rest the reader finds by
    // key in the checkpoint index.
    std::vector<CheckpointReader::TensorLocation> locations;
    for (auto const& shard : shards)
    {
        require(!shard.tensorKey.empty(), "PLE shard tensor key is missing");
        require(shard.rows > 0 && shard.rows <= std::numeric_limits<int64_t>::max() / rowBytes
                && shard.bytes == shard.rows * rowBytes,
            "PLE shard byte count does not match its row count");
        if (!shard.file.empty())
        {
            require(shard.offset >= 0, "Negative PLE checkpoint offset");
            locations.push_back({shard.tensorKey, overrideFile.empty() ? shard.file : overrideFile.filename(), dtype,
                Coords{shard.rows, rowWidth}, static_cast<size_t>(shard.offset), static_cast<size_t>(shard.bytes)});
        }
    }

    // Check every shard against the metadata the engine was built with, then lay them out in row
    // order. Identity samples are spot checks, not a digest of the payload.
    CheckpointReader reader(directory, locations, overrideFile);
    PleTable table;
    int64_t rowBegin{};
    int64_t totalBytes{};
    for (auto const& shard : shards)
    {
        auto const location = reader.locate(shard.tensorKey);
        require(location.dtype == dtype && location.shape == Coords{shard.rows, rowWidth}
                && location.bytes == static_cast<size_t>(shard.bytes),
            "PLE source metadata does not match the engine");
        CheckpointReader::View view;
        require(reader.findHost(shard.tensorKey, view), "PLE source tensor is missing");
        require(!shard.samples.empty() && shard.samples.size() <= 64,
            "PLE source identity samples are missing or excessive");
        for (auto const& sample : shard.samples)
        {
            require(!sample.data.empty() && sample.data.size() <= 8192 && sample.data.size() % 2 == 0,
                "Invalid PLE source identity sample length");
            int64_t const bytes = static_cast<int64_t>(sample.data.size() / 2);
            require(sample.offset >= 0 && sample.offset <= shard.bytes && bytes <= shard.bytes - sample.offset,
                "PLE identity sample exceeds the tensor");
            for (int64_t index = 0; index < bytes; ++index)
            {
                auto const expected = static_cast<uint8_t>(
                    hexDigit(sample.data[index * 2]) * 16 + hexDigit(sample.data[index * 2 + 1]));
                require(view.data[sample.offset + index] == expected,
                    "PLE source sampled identity does not match the engine");
            }
        }
        require(location.offset <= static_cast<size_t>(std::numeric_limits<int64_t>::max() - shard.bytes),
            "PLE source file offset overflow");
        require(rowBegin <= std::numeric_limits<int64_t>::max() - shard.rows
                && totalBytes <= std::numeric_limits<int64_t>::max() - shard.bytes,
            "PLE shard row or byte total overflow");
        table.mShards.push_back(
            {location.file, rowBegin, shard.rows, static_cast<int64_t>(location.offset), shard.bytes});
        rowBegin += shard.rows;
        totalBytes += shard.bytes;
    }
    require(
        rowBegin == rowCount && totalBytes == source.bytes, "PLE shard rows or bytes do not sum to the pool geometry");

    // The pool may hold a different dtype and scale than the checkpoint; the fill converts.
    table.mRowCount = rowCount;
    table.mRowWidth = rowWidth;
    table.mRowBytes = poolRowBytes;
    table.mDtype = poolDtype;
    table.mScale = poolScale;
    table.mSourceRowBytes = rowBytes;
    table.mSourceDtype = source.dtype;
    table.mSourceScale = source.scale;
    return table;
}

PleTable::RowLocation PleTable::locate(int64_t row) const
{
    if (row < 0 || row >= mRowCount)
    {
        throw std::out_of_range("PLE row is outside table");
    }
    auto const found = std::upper_bound(
        mShards.begin(), mShards.end(), row, [](int64_t value, Shard const& shard) { return value < shard.rowBegin; });
    auto const shard = std::prev(found);
    return {
        static_cast<size_t>(shard - mShards.begin()), shard->byteOffset + (row - shard->rowBegin) * mSourceRowBytes};
}

} // namespace rt
} // namespace trt_edgellm
