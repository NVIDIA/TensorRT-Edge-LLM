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

#include "runtime/embedding/pleTable.h"
#include "scratchDir.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{
namespace
{
class PleTableTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mDirectory = makeScratchDir("ple-table");
        mIdentity = "sha256:" + std::string(64, 'a');
        mManifest = {{"schema_version", 1}, {"table_id", mIdentity}, {"row_count", 5}, {"row_width", 160},
            {"dtype", "F8_E4M3"}, {"scale", 0.25},
            {"shards",
                {{{"file", "a.bin"}, {"row_begin", 0}, {"row_count", 2}, {"byte_offset", 13}, {"byte_length", 320},
                     {"sha256", std::string(64, 'b')}},
                    {{"file", "b.bin"}, {"row_begin", 2}, {"row_count", 3}, {"byte_offset", 29}, {"byte_length", 480},
                        {"sha256", std::string(64, 'c')}}}}};
        for (auto const& entry : mManifest["shards"])
        {
            auto const path = mDirectory / entry["file"].get<std::string>();
            std::ofstream output(path, std::ios::binary);
            output << std::string(entry["byte_offset"].get<size_t>() + entry["byte_length"].get<size_t>(), 'x');
            ASSERT_TRUE(output.good());
        }
    }

    void TearDown() override
    {
        if (!mDirectory.empty())
        {
            std::filesystem::remove_all(mDirectory);
        }
    }

    PleTable open(double scale = 0.25)
    {
        auto const manifest = mDirectory / "manifest.json";
        {
            std::ofstream output(manifest);
            output << mManifest;
        }
        return PleTable::open(manifest, mIdentity, scale);
    }

    PleTable::Source writeCheckpointSource(PleTable::DType dtype = PleTable::DType::kBF16)
    {
        std::string const dtypeName = dtype == PleTable::DType::kFP32 ? "F32"
            : dtype == PleTable::DType::kBF16                         ? "BF16"
                                                                      : "F16";
        int64_t const bytes = 3 * 2 * (dtype == PleTable::DType::kFP32 ? 4 : 2);
        nlohmann::json const header{
            {"ple.weight", {{"dtype", dtypeName}, {"shape", {3, 2}}, {"data_offsets", {0, bytes}}}}};
        std::string encoded = header.dump();
        encoded.append((8 - encoded.size() % 8) % 8, ' ');
        uint64_t const headerBytes = encoded.size();
        mSourceOffset = sizeof(headerBytes) + headerBytes;
        std::ofstream file(mDirectory / "model.safetensors", std::ios::binary);
        file.write(reinterpret_cast<char const*>(&headerBytes), sizeof(headerBytes));
        file << encoded;
        for (int64_t i = 0; i < bytes; ++i)
        {
            file.put(static_cast<char>(i));
        }
        if (!file.good())
        {
            throw std::runtime_error("Cannot write checkpoint fixture");
        }
        return PleTable::Source{"ple.weight", dtype, bytes, 1.5, {{0, "00010203"}}};
    }

    //! Write ``shards`` FP8 tensors of ``rowsPerShard`` rows into one safetensors file and describe
    //! them as an ordered sharded source. Byte ``j`` of the payload holds ``j % 251`` so every row
    //! has content that identifies its global position.
    PleTable::Source writeShardedSource(int32_t shards, int64_t rowsPerShard, int64_t rowWidth)
    {
        int64_t const shardBytes = rowsPerShard * rowWidth;
        nlohmann::json header = nlohmann::json::object();
        for (int32_t i = 0; i < shards; ++i)
        {
            header["ngram.shard_" + std::to_string(i) + ".weight"] = {{"dtype", "F8_E4M3"},
                {"shape", {rowsPerShard, rowWidth}}, {"data_offsets", {i * shardBytes, (i + 1) * shardBytes}}};
        }
        std::string encoded = header.dump();
        encoded.append((8 - encoded.size() % 8) % 8, ' ');
        uint64_t const headerBytes = encoded.size();
        std::ofstream file(mDirectory / "model.safetensors", std::ios::binary);
        file.write(reinterpret_cast<char const*>(&headerBytes), sizeof(headerBytes));
        file << encoded;
        for (int64_t j = 0; j < shardBytes * shards; ++j)
        {
            file.put(static_cast<char>(j % 251));
        }
        if (!file.good())
        {
            throw std::runtime_error("Cannot write sharded checkpoint fixture");
        }
        PleTable::Source source;
        source.dtype = PleTable::DType::kFP8;
        source.bytes = shardBytes * shards;
        source.scale = 1.0;
        for (int32_t i = 0; i < shards; ++i)
        {
            char buffer[9]{};
            for (int32_t b = 0; b < 4; ++b)
            {
                std::snprintf(buffer + b * 2, 3, "%02x", static_cast<unsigned>((i * shardBytes + b) % 251));
            }
            source.shards.push_back({"ngram.shard_" + std::to_string(i) + ".weight", rowsPerShard, shardBytes,
                {{0, std::string(buffer)}}, {}, 0});
        }
        return source;
    }

    int64_t mSourceOffset{};
    std::filesystem::path mDirectory;
    std::string mIdentity;
    nlohmann::json mManifest;
};

TEST_F(PleTableTest, ResolvesShardBoundariesAndOffsets)
{
    auto const table = open();
    EXPECT_EQ(table.rowCount(), 5);
    EXPECT_EQ(table.tableId(), mIdentity);
    EXPECT_DOUBLE_EQ(table.scale(), 0.25);
    EXPECT_EQ(table.locate(0).byteOffset, 13);
    EXPECT_EQ(table.locate(1).byteOffset, 173);
    EXPECT_EQ(table.locate(2).shardIndex, 1);
    EXPECT_EQ(table.locate(2).byteOffset, 29);
    EXPECT_EQ(table.locate(4).byteOffset, 349);
    EXPECT_THROW(table.locate(-1), std::out_of_range);
    EXPECT_THROW(table.locate(5), std::out_of_range);
}

TEST_F(PleTableTest, RejectsIdentityScaleAndSchemaMismatch)
{
    auto const valid = mManifest;
    for (auto const& replacement : std::vector<nlohmann::json>{{{"table_id", "sha256:" + std::string(64, 'd')}},
             {{"scale", 1.0}}, {{"scale", -0.25}}, {{"schema_version", 2}}, {{"row_width", 0}}, {{"row_width", -1}},
             {{"dtype", "F32"}}, {{"row_count", std::numeric_limits<uint64_t>::max()}}})
    {
        mManifest = valid;
        mManifest.update(replacement);
        EXPECT_THROW(open(), std::invalid_argument) << replacement;
    }
}

TEST_F(PleTableTest, RejectsGapsOverlapsLengthsAndMissingFiles)
{
    auto const valid = mManifest;
    for (auto const& replacement : std::vector<nlohmann::json>{{{"row_begin", 1}}, {{"row_begin", 3}},
             {{"byte_length", 479}}, {{"byte_offset", -1}}, {{"row_count", 4}}, {{"file", "missing.bin"}},
             {{"file", "../outside.bin"}}, {{"sha256", "bad"}}})
    {
        mManifest = valid;
        mManifest["shards"][1].update(replacement);
        EXPECT_THROW(open(), std::invalid_argument) << replacement;
    }
    mManifest = valid;
    mManifest["shards"].erase(1);
    EXPECT_THROW(open(), std::invalid_argument);
    mManifest = valid;
    std::filesystem::resize_file(mDirectory / "b.bin", 508);
    EXPECT_THROW(open(), std::invalid_argument);
}

TEST_F(PleTableTest, ResolvesNativeRowsAndRejectsByteSizeOverflow)
{
    auto const original = mManifest;
    for (auto const& dtype : {"F16", "BF16"})
    {
        mManifest = original;
        mManifest["dtype"] = dtype;
        mManifest["scale"] = 1.0;
        mManifest["row_width"] = 7;
        mManifest["shards"][0]["byte_length"] = 28;
        mManifest["shards"][1]["byte_length"] = 42;
        auto const table = open(1.0);
        EXPECT_EQ(table.rowWidth(), 7);
        EXPECT_EQ(table.rowBytes(), 14);
        EXPECT_EQ(table.dtype(), std::string(dtype) == "F16" ? PleTable::DType::kFP16 : PleTable::DType::kBF16);
        EXPECT_EQ(table.locate(1).byteOffset, 27);
        EXPECT_EQ(table.locate(4).byteOffset, 57);
        mManifest["scale"] = 0.25;
        EXPECT_THROW(open(), std::invalid_argument);
        mManifest["scale"] = 1.0;
        mManifest["row_width"] = std::numeric_limits<int64_t>::max();
        EXPECT_THROW(open(1.0), std::invalid_argument);
        mManifest["row_width"] = std::numeric_limits<int64_t>::max() / 2;
        EXPECT_THROW(open(1.0), std::invalid_argument);
    }
}

TEST_F(PleTableTest, OpensSparseFiftyGiBTableWithoutReadingPayload)
{
    constexpr int64_t kROW_WIDTH{160};
    int64_t const rows = (50LL << 30) / kROW_WIDTH;
    int64_t const bytes = rows * kROW_WIDTH;
    mManifest["row_count"] = rows;
    mManifest["shards"].erase(1);
    auto& shard = mManifest["shards"][0];
    shard["row_count"] = rows;
    shard["byte_length"] = bytes;
    std::filesystem::resize_file(mDirectory / "a.bin", bytes + 13);
    auto const table = open();
    EXPECT_EQ(table.shards().size(), 1);
    EXPECT_EQ(table.locate(rows - 1).byteOffset, bytes - kROW_WIDTH + 13);
}

TEST_F(PleTableTest, ResolvesCheckpointTensorPayloadWithoutMaterializingRows)
{
    for (auto const dtype : {PleTable::DType::kFP16, PleTable::DType::kBF16, PleTable::DType::kFP32})
    {
        auto const source = writeCheckpointSource(dtype);
        auto const table = PleTable::fromSource(source, 3, 2, mDirectory);
        EXPECT_EQ(table.dtype(), PleTable::DType::kFP16);
        EXPECT_EQ(table.sourceDtype(), dtype);
        EXPECT_EQ(table.rowBytes(), 4);
        EXPECT_EQ(table.sourceRowBytes(), dtype == PleTable::DType::kFP32 ? 8 : 4);
        EXPECT_DOUBLE_EQ(table.sourceScale(), 1.5);
        EXPECT_DOUBLE_EQ(table.scale(), 1.0);
        ASSERT_EQ(table.shards().size(), 1U);
        EXPECT_EQ(table.shards()[0].path, mDirectory / "model.safetensors");
        EXPECT_EQ(table.locate(0).byteOffset, mSourceOffset);
        EXPECT_EQ(table.locate(2).byteOffset, mSourceOffset + 2 * table.sourceRowBytes());
    }
}

TEST_F(PleTableTest, ExplicitCheckpointFileOverridesDirectoryAndUnrelatedIndex)
{
    auto const source = writeCheckpointSource();
    {
        std::ofstream index(mDirectory / "model.safetensors.index.json");
        index << R"({"weight_map":{"ple.weight":"missing.safetensors"}})";
    }
    EXPECT_THROW(PleTable::fromSource(source, 3, 2, mDirectory), std::exception);
    auto const explicitFile = mDirectory / "model.safetensors";
    auto const table = PleTable::fromSource(source, 3, 2, mDirectory / "unavailable", explicitFile);
    EXPECT_EQ(table.shards()[0].path, explicitFile);
    EXPECT_EQ(table.locate(1).byteOffset, mSourceOffset + 4);
    EXPECT_NO_THROW(PleTable::fromSource(source, 3, 2, mDirectory, explicitFile));
    EXPECT_THROW(PleTable::fromSource(source, 3, 2, mDirectory, mDirectory / "missing.safetensors"), std::exception);
}

TEST_F(PleTableTest, IndexedCheckpointUsesAbsolutePayloadOffsetAndExplicitFile)
{
    PleTable::Source source{"ple.weight", PleTable::DType::kFP16, 12, 2.0, {{0, "00010203"}}, "bundle.bin", 37};
    {
        std::ofstream file(mDirectory / "bundle.bin", std::ios::binary);
        file << std::string(37, 'x');
        for (int i = 0; i < 12; ++i)
        {
            file.put(static_cast<char>(i));
        }
    }
    auto const table = PleTable::fromSource(source, 3, 2, mDirectory);
    EXPECT_EQ(table.locate(0).byteOffset, 37);
    EXPECT_EQ(table.locate(2).byteOffset, 45);
    auto const relocated = mDirectory / "relocated.bin";
    std::filesystem::rename(mDirectory / "bundle.bin", relocated);
    auto const overridden = PleTable::fromSource(source, 3, 2, mDirectory / "unavailable", relocated);
    EXPECT_EQ(overridden.shards()[0].path, relocated);
    EXPECT_EQ(overridden.locate(2).byteOffset, 45);
    source.offset = 38;
    EXPECT_THROW(PleTable::fromSource(source, 3, 2, {}, relocated), std::exception);
}

TEST_F(PleTableTest, RejectsCheckpointMetadataAndSampleMismatch)
{
    auto const valid = writeCheckpointSource();
    auto invalid = valid;
    invalid.dtype = PleTable::DType::kFP16;
    EXPECT_THROW(PleTable::fromSource(invalid, 3, 2, mDirectory), std::exception);
    invalid.dtype = PleTable::DType::kFP8;
    EXPECT_THROW(PleTable::fromSource(invalid, 3, 2, mDirectory), std::exception);
    EXPECT_THROW(PleTable::fromSource(valid, 2, 3, mDirectory), std::exception);
    EXPECT_THROW(PleTable::fromSource(valid, 4, 2, mDirectory), std::exception);
    invalid = valid;
    invalid.tensorKey = "missing.weight";
    EXPECT_THROW(PleTable::fromSource(invalid, 3, 2, mDirectory), std::exception);
    invalid = valid;
    invalid.samples.clear();
    EXPECT_THROW(PleTable::fromSource(invalid, 3, 2, mDirectory), std::exception);
    for (auto const& sample : std::vector<PleTable::Sample>{{0, "ff"}, {-1, "00"}, {12, "00"}, {0, "z0"}, {0, "0"}})
    {
        invalid = valid;
        invalid.samples = {sample};
        EXPECT_THROW(PleTable::fromSource(invalid, 3, 2, mDirectory), std::exception);
    }
    invalid = valid;
    invalid.scale = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(PleTable::fromSource(invalid, 3, 2, mDirectory), std::exception);
    std::filesystem::remove(mDirectory / "model.safetensors");
    EXPECT_THROW(PleTable::fromSource(valid, 3, 2, mDirectory), std::exception);
}

TEST_F(PleTableTest, ShardedSourceConcatenatesRowsInArrayOrder)
{
    // Given a table split across three checkpoint tensors listed in row order,
    // When the source is resolved,
    // Then every global row must land in the shard that covers it, at that shard's own byte offset
    // -- the invariant the n-gram row IDs rely on (shard = row / rowsPerShard).
    constexpr int64_t kROWS_PER_SHARD{4};
    constexpr int64_t kROW_WIDTH{160};
    auto const source = writeShardedSource(3, kROWS_PER_SHARD, kROW_WIDTH);
    auto const table = PleTable::fromSource(source, 12, kROW_WIDTH, mDirectory, {}, PleTable::DType::kFP8, 0.5);

    EXPECT_EQ(table.rowCount(), 12);
    EXPECT_EQ(table.rowBytes(), kROW_WIDTH);
    EXPECT_DOUBLE_EQ(table.scale(), 0.5);
    ASSERT_EQ(table.shards().size(), 3U);
    int64_t const base = table.shards().front().byteOffset;
    for (int64_t row = 0; row < 12; ++row)
    {
        auto const located = table.locate(row);
        EXPECT_EQ(located.shardIndex, static_cast<size_t>(row / kROWS_PER_SHARD)) << "row " << row;
        EXPECT_EQ(located.byteOffset, base + row * kROW_WIDTH) << "row " << row;
    }
    EXPECT_THROW(table.locate(12), std::out_of_range);
}

TEST_F(PleTableTest, ShardedSourceRejectsRowsThatDoNotCoverThePool)
{
    // Given a sharded source whose byte total matches the pool but whose shards cover fewer rows
    // than the pool claims -- what a dropped shard looks like, and the shard suffix sorts
    // lexicographically so one is easy to lose,
    // When the source is resolved,
    // Then it must be rejected rather than silently addressing a shorter table.
    auto source = writeShardedSource(3, 4, 160);
    source.shards.pop_back();
    // Byte total still describes all 12 rows, so only the row-coverage check can catch this.
    EXPECT_EQ(source.bytes, 12 * 160);
    EXPECT_THROW(PleTable::fromSource(source, 12, 160, mDirectory, {}, PleTable::DType::kFP8, 0.5), std::exception);

    // The mirrored case: every shard present, but the pool claims more rows than they cover.
    auto extra = writeShardedSource(3, 4, 160);
    extra.bytes = 16 * 160;
    EXPECT_THROW(PleTable::fromSource(extra, 16, 160, mDirectory, {}, PleTable::DType::kFP8, 0.5), std::exception);
}

TEST_F(PleTableTest, ShardedSourceRejectsATensorKeyAlongsideShards)
{
    // Given a source carrying both the single-tensor key and a shard list,
    // When it is resolved,
    // Then it must be rejected: the two describe different row spaces and silently preferring one
    // would make the table's extent depend on resolution order.
    auto source = writeShardedSource(2, 4, 160);
    source.tensorKey = "ngram.shard_0.weight";
    EXPECT_THROW(PleTable::fromSource(source, 8, 160, mDirectory, {}, PleTable::DType::kFP8, 0.5), std::exception);
}

} // namespace
} // namespace rt
} // namespace trt_edgellm
