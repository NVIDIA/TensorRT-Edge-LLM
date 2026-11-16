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

#include "common/checkMacros.h"
#include "runtime/embedding/pleFileUtils.h"
#include "runtime/embedding/pleRowStore.h"
#include "scratchDir.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{
uint8_t payload(int64_t row, int64_t column)
{
    return static_cast<uint8_t>(1 + (row * 17 + column) % 112);
}

PleIoConfig io(size_t rows, size_t sourceRowBytes = 160, size_t readBytes = 4096)
{
    PleIoConfig result;
    result.fillBytes = rows * sourceRowBytes;
    result.preloadBytes = rows * sourceRowBytes;
    result.readBytes = readBytes;
    return result;
}

std::unique_ptr<PleIoBudget::Reservation> reserveCapacity(
    std::shared_ptr<PleIoBudget> const& budget, size_t remainingBytes, size_t pageBytes)
{
    // Models capacity held by another store without allocating its buffers.
    size_t const rows = (PleIoBudget::kLIMIT_BYTES - remainingBytes) / 8;
    return budget->reserve(io(rows, 4, pageBytes), 4, rows, pageBytes, false);
}

class PleRowStoreTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mDirectory = makeScratchDir("ple-row-store");
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
        nlohmann::json shards = nlohmann::json::array();
        for (int shard = 0; shard < 2; ++shard)
        {
            writeShard(shard);
            shards.push_back({{"file", std::to_string(shard) + ".bin"}, {"row_begin", shard * 32}, {"row_count", 32},
                {"byte_offset", prefix(shard)}, {"byte_length", 32 * 160}, {"sha256", std::string(64, 'b')}});
        }
        nlohmann::json const manifest{{"schema_version", 1}, {"table_id", identity()}, {"row_count", 64},
            {"row_width", 160}, {"dtype", "F8_E4M3"}, {"scale", 0.25}, {"shards", shards}};
        std::ofstream output(mDirectory / "manifest.json");
        output << manifest;
        ASSERT_TRUE(output.good());
    }
    void TearDown() override
    {
        if (mStream != nullptr)
        {
            EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
            EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
        }
        std::filesystem::remove_all(mDirectory);
    }
    static int prefix(int shard)
    {
        return shard == 0 ? 13 : 29;
    }
    static std::string identity()
    {
        return "sha256:" + std::string(64, 'a');
    }
    PleTable table()
    {
        return PleTable::open(mDirectory / "manifest.json", identity(), 0.25);
    }
    void writeShard(int shard)
    {
        std::ofstream output(mDirectory / (std::to_string(shard) + ".bin"), std::ios::binary | std::ios::trunc);
        output << std::string(prefix(shard), 'x');
        for (int64_t row = shard * 32; row < (shard + 1) * 32; ++row)
        {
            std::array<uint8_t, 160> bytes{};
            for (int64_t column = 0; column < 160; ++column)
            {
                bytes[column] = payload(row, column);
            }
            output.write(reinterpret_cast<char const*>(bytes.data()), bytes.size());
        }
        ASSERT_TRUE(output.good());
    }
    PleTable customTable(std::string const& dtype, int64_t width, int64_t rows = 8)
    {
        int64_t const rowBytes = width * (dtype == "F8_E4M3" ? 1 : 2);
        {
            std::ofstream output(mDirectory / "custom.bin", std::ios::binary);
            output << std::string(13, 'x');
            for (int64_t row = 0; row < rows; ++row)
            {
                for (int64_t column = 0; column < rowBytes; ++column)
                {
                    auto const value = payload(row, column);
                    output.write(reinterpret_cast<char const*>(&value), 1);
                }
            }
        }
        nlohmann::json const manifest{{"schema_version", 1}, {"table_id", identity()}, {"row_count", rows},
            {"row_width", width}, {"dtype", dtype}, {"scale", 1.0},
            {"shards",
                {{{"file", "custom.bin"}, {"row_begin", 0}, {"row_count", rows}, {"byte_offset", 13},
                    {"byte_length", rows * rowBytes}, {"sha256", std::string(64, 'b')}}}}};
        {
            std::ofstream output(mDirectory / "custom.json");
            output << manifest;
        }
        return PleTable::open(mDirectory / "custom.json", identity(), 1.0);
    }
    void verifyBytes(PleRowStore& store, std::vector<PleRowStore::RequestLease> const& leases,
        std::vector<std::vector<int64_t>> const& requests, int64_t rowBytes)
    {
        Tensor host({rowBytes}, DeviceType::kCPU, trt_edgellm::DataType::kINT8);
        ASSERT_EQ(leases.size(), requests.size());
        for (size_t request = 0; request < requests.size(); ++request)
        {
            auto const& mapping = leases[request].mapping();
            ASSERT_EQ(mapping.size(), requests[request].size());
            for (size_t index = 0; index < mapping.size(); ++index)
            {
                auto const* source = static_cast<char const*>(store.pool().rawPointer())
                    + static_cast<int64_t>(mapping[index]) * rowBytes;
                CUDA_CHECK(cudaMemcpyAsync(host.rawPointer(), source, rowBytes, cudaMemcpyDeviceToHost, mStream));
                CUDA_CHECK(cudaStreamSynchronize(mStream));
                for (int64_t column = 0; column < rowBytes; ++column)
                {
                    auto const expected
                        = requests[request][index] == -1 ? 0 : payload(requests[request][index], column);
                    ASSERT_EQ(static_cast<uint8_t const*>(host.rawPointer())[column], expected)
                        << "request=" << request << " token-row=" << index << " column=" << column;
                }
            }
        }
    }
    void verify(PleRowStore& store, std::vector<PleRowStore::RequestLease> const& leases,
        std::vector<std::vector<int64_t>> const& requests)
    {
        verifyBytes(store, leases, requests, 160);
    }
    static float sourceValue(PleTable::DType dtype, int64_t row, int64_t column)
    {
        constexpr std::array<float, 8> values{1.0006F, -2.001F, 0.3333F, 4.125F, -0.06251F, 7.99F, 0.001F, -1.5F};
        float const value = values[static_cast<size_t>(row * 3 + column) % values.size()];
        if (dtype == PleTable::DType::kFP16)
        {
            return __half2float(__float2half_rn(value));
        }
        if (dtype == PleTable::DType::kBF16)
        {
            return __bfloat162float(__float2bfloat16_rn(value));
        }
        return value;
    }
    PleTable sourceTable(PleTable::DType dtype, int64_t rows = 8, int64_t width = 7)
    {
        int64_t const elementBytes = dtype == PleTable::DType::kFP32 ? 4 : 2;
        std::vector<char> bytes(static_cast<size_t>(rows * width * elementBytes));
        for (int64_t row = 0; row < rows; ++row)
        {
            for (int64_t column = 0; column < width; ++column)
            {
                char* destination = bytes.data() + (row * width + column) * elementBytes;
                float const value = sourceValue(dtype, row, column);
                if (dtype == PleTable::DType::kFP16)
                {
                    __half const encoded = __float2half_rn(value);
                    std::memcpy(destination, &encoded, sizeof(encoded));
                }
                else if (dtype == PleTable::DType::kBF16)
                {
                    __nv_bfloat16 const encoded = __float2bfloat16_rn(value);
                    std::memcpy(destination, &encoded, sizeof(encoded));
                }
                else
                {
                    std::memcpy(destination, &value, sizeof(value));
                }
            }
        }
        {
            std::ofstream file(mDirectory / "source.bin", std::ios::binary);
            file << std::string(13, 'x');
            file.write(bytes.data(), bytes.size());
            if (!file.good())
            {
                throw std::runtime_error("Cannot write PLE source fixture");
            }
        }
        std::string sample;
        char const* digits = "0123456789abcdef";
        for (size_t i = 0; i < 4; ++i)
        {
            auto const value = static_cast<uint8_t>(bytes[i]);
            sample.push_back(digits[value >> 4]);
            sample.push_back(digits[value & 15]);
        }
        PleTable::Source const source{
            "ple.weight", dtype, static_cast<int64_t>(bytes.size()), 1.5, {{0, sample}}, "source.bin", 13};
        return PleTable::fromSource(source, rows, width, mDirectory);
    }
    void verifyConverted(PleRowStore& store, std::vector<PleRowStore::RequestLease> const& leases,
        std::vector<std::vector<int64_t>> const& requests, PleTable::DType dtype, int64_t width = 7)
    {
        ASSERT_EQ(store.pool().getDataType(), trt_edgellm::DataType::kHALF);
        Tensor host({width}, DeviceType::kCPU, trt_edgellm::DataType::kHALF);
        ASSERT_EQ(leases.size(), requests.size());
        for (size_t request = 0; request < requests.size(); ++request)
        {
            auto const& mapping = leases[request].mapping();
            ASSERT_EQ(mapping.size(), requests[request].size());
            for (size_t index = 0; index < mapping.size(); ++index)
            {
                auto const* source = static_cast<char const*>(store.pool().rawPointer()) + mapping[index] * width * 2;
                CUDA_CHECK(cudaMemcpyAsync(host.rawPointer(), source, width * 2, cudaMemcpyDeviceToHost, mStream));
                CUDA_CHECK(cudaStreamSynchronize(mStream));
                for (int64_t column = 0; column < width; ++column)
                {
                    float const value = requests[request][index] == -1
                        ? 0.0F
                        : __half2float(__float2half_rn(sourceValue(dtype, requests[request][index], column)));
                    float const expected = __half2float(__float2half_rn(value * 1.5F));
                    EXPECT_EQ(__half2float(host.dataPointer<__half>()[column]), expected)
                        << "row=" << requests[request][index] << " column=" << column;
                }
            }
        }
    }
    std::filesystem::path mDirectory;
    cudaStream_t mStream{};
};

TEST_F(PleRowStoreTest, SamePageUsesOneReadWithinOneFillAndWarmHits)
{
    PleRowStore store(table(), 9, mStream, io(6));
    std::vector<std::vector<int64_t>> const requests{{0, 1, 2, -1, 0}, {2, 3, 4, 5, 5}};
    auto leases = store.prepare(requests);
    ASSERT_NO_FATAL_FAILURE(verify(store, leases, requests));
    EXPECT_EQ(store.stats().uniqueRows, 6);
    EXPECT_EQ(store.stats().misses, 6);
    EXPECT_EQ(store.stats().readCalls, 1);
    EXPECT_EQ(store.stats().readBytes, 4096);
    EXPECT_EQ(store.stats().usefulBytes, 6 * 160);
    EXPECT_EQ(store.stats().fillWaves, 1);
    EXPECT_EQ(store.pool().getMemoryCapacity(), 9 * 160);
    leases.clear();
    auto warm = store.prepare(requests);
    ASSERT_NO_FATAL_FAILURE(verify(store, warm, requests));
    EXPECT_EQ(store.stats().misses, 0);
    EXPECT_EQ(store.stats().readCalls, 0);
    EXPECT_EQ(store.stats().readBytes, 0);
    EXPECT_EQ(store.stats().fillWaves, 0);
}

TEST_F(PleRowStoreTest, SharedPageRowsRemainCorrectAcrossSeparateFillWaves)
{
    PleRowStore store(table(), 7, mStream, io(2));
    std::vector<std::vector<int64_t>> const requests{{5, 0, 4, 1, 3, 2, 0, -1}};
    auto leases = store.prepare(requests);
    ASSERT_NO_FATAL_FAILURE(verify(store, leases, requests));
    EXPECT_EQ(store.stats().fillWaves, 3);
    EXPECT_EQ(store.stats().readCalls, 3);
    EXPECT_EQ(store.stats().readBytes, 3 * 4096);
    EXPECT_EQ(store.stats().usefulBytes, 6 * 160);
}

TEST_F(PleRowStoreTest, DefaultFillByteBudgetAccommodatesMoreThan256SmallRows)
{
    constexpr int64_t kROWS = 512;
    auto descriptor = customTable("F8_E4M3", 160, kROWS);
    PleRowStore store(descriptor, kROWS + 1, mStream);
    std::vector<std::vector<int64_t>> requests(1);
    requests[0].resize(kROWS);
    std::iota(requests[0].begin(), requests[0].end(), 0);
    auto leases = store.prepare(requests);
    ASSERT_NO_FATAL_FAILURE(verify(store, leases, requests));
    EXPECT_EQ(store.stats().misses, kROWS);
    EXPECT_EQ(store.stats().fillWaves, 1);
    EXPECT_EQ(store.stats().readCalls, 1);
    EXPECT_EQ(store.stats().readBytes, 13 + kROWS * 160);
    EXPECT_EQ(store.stats().usefulBytes, kROWS * 160);
}

TEST_F(PleRowStoreTest, OversizedReaderBuffersFitBudgetAndPreserveRows)
{
    auto budget = std::make_shared<PleIoBudget>();
    {
        PleRowStore store(
            table(), 7, mStream, io(6, 160, PleIoBudget::kLIMIT_BYTES), PleRowStore::Residency::kCACHED, budget);
        EXPECT_GT(budget->usedBytes(), 0);
        EXPECT_LE(budget->usedBytes(), PleIoBudget::kLIMIT_BYTES);
        std::vector<std::vector<int64_t>> const requests{{25, 31, 0}, {32, 57, 63, -1, 25}};
        auto leases = store.prepare(requests);
        ASSERT_NO_FATAL_FAILURE(verify(store, leases, requests));
        EXPECT_EQ(store.stats().misses, 6);
    }
    EXPECT_EQ(budget->usedBytes(), 0);
}

TEST_F(PleRowStoreTest, CachedStoresShareBudgetAndReturnCapacityOnDestruction)
{
    auto budget = std::make_shared<PleIoBudget>();
    auto const config = io(6, 160, 8U << 20);
    {
        PleRowStore first(table(), 7, mStream, config, PleRowStore::Residency::kCACHED, budget);
        size_t const firstBytes = budget->usedBytes();
        {
            PleRowStore second(table(), 7, mStream, config, PleRowStore::Residency::kCACHED, budget);
            EXPECT_GT(budget->usedBytes(), firstBytes);
            EXPECT_LE(budget->usedBytes(), PleIoBudget::kLIMIT_BYTES);
            EXPECT_LT(budget->usedBytes() - firstBytes, firstBytes);
            auto firstLeases = first.prepare({{0, 31, 32}});
            auto secondLeases = second.prepare({{63, 57, 25, -1}});
            ASSERT_NO_FATAL_FAILURE(verify(first, firstLeases, {{0, 31, 32}}));
            ASSERT_NO_FATAL_FAILURE(verify(second, secondLeases, {{63, 57, 25, -1}}));
        }
        EXPECT_EQ(budget->usedBytes(), firstBytes);
    }
    EXPECT_EQ(budget->usedBytes(), 0);
    PleRowStore replacement(table(), 7, mStream, config, PleRowStore::Residency::kCACHED, budget);
    auto leases = replacement.prepare({{63, 0}});
    ASSERT_NO_FATAL_FAILURE(verify(replacement, leases, {{63, 0}}));
}

TEST_F(PleRowStoreTest, ResidentPreloadReturnsSharedCapacityWhileCachedStoreRemainsLive)
{
    auto budget = std::make_shared<PleIoBudget>();
    PleRowStore cached(table(), 7, mStream, io(6), PleRowStore::Residency::kCACHED, budget);
    size_t const cachedBytes = budget->usedBytes();
    PleRowStore resident(table(), 65, mStream, io(7), PleRowStore::Residency::kRESIDENT, budget);
    EXPECT_EQ(budget->usedBytes(), cachedBytes);
    EXPECT_GT(resident.preloadStats().fillWaves, 1);
    auto cachedLeases = cached.prepare({{63, 0, 32}});
    auto residentLeases = resident.prepare({{63, 0, 32}});
    ASSERT_NO_FATAL_FAILURE(verify(cached, cachedLeases, {{63, 0, 32}}));
    ASSERT_NO_FATAL_FAILURE(verify(resident, residentLeases, {{63, 0, 32}}));
}

TEST_F(PleRowStoreTest, SourceOpenFailureReturnsSharedBudget)
{
    auto budget = std::make_shared<PleIoBudget>();
    PleRowStore cached(table(), 7, mStream, io(6), PleRowStore::Residency::kCACHED, budget);
    size_t const cachedBytes = budget->usedBytes();
    auto descriptor = table();
    ASSERT_TRUE(std::filesystem::remove(mDirectory / "1.bin"));
    EXPECT_THROW(
        PleRowStore(descriptor, 7, mStream, io(6), PleRowStore::Residency::kCACHED, budget), std::system_error);
    EXPECT_EQ(budget->usedBytes(), cachedBytes);
    auto leases = cached.prepare({{63, 0}});
    ASSERT_NO_FATAL_FAILURE(verify(cached, leases, {{63, 0}}));
}

TEST_F(PleRowStoreTest, ReservedCapacityReducesFillBatchWithoutChangingRows)
{
    size_t const pageBytes = pleSystemPageSize();
    std::vector<std::vector<int64_t>> const requests{{0, 1, 31, 32, 57, 63, -1}};
    PleRowStore baseline(table(), 7, mStream, io(6, 160, pageBytes));
    auto baselineLeases = baseline.prepare(requests);
    ASSERT_NO_FATAL_FAILURE(verify(baseline, baselineLeases, requests));
    ASSERT_EQ(baseline.stats().fillWaves, 1);

    auto budget = std::make_shared<PleIoBudget>();
    auto pressure = reserveCapacity(budget, 4 * pageBytes + 2 * 164, pageBytes);
    size_t const reservedBytes = budget->usedBytes();
    {
        PleRowStore constrained(table(), 7, mStream, io(6, 160, pageBytes), PleRowStore::Residency::kCACHED, budget);
        auto leases = constrained.prepare(requests);
        ASSERT_NO_FATAL_FAILURE(verify(constrained, leases, requests));
        EXPECT_EQ(constrained.stats().fillWaves, 3);
        EXPECT_EQ(constrained.stats().uniqueRows, baseline.stats().uniqueRows);
        EXPECT_EQ(constrained.pool().getShape(), baseline.pool().getShape());
        EXPECT_LE(budget->usedBytes(), PleIoBudget::kLIMIT_BYTES);
    }
    EXPECT_EQ(budget->usedBytes(), reservedBytes);
}

TEST_F(PleRowStoreTest, InsufficientRemainingBudgetRejectsBeforeSourceOpen)
{
    size_t const pageBytes = pleSystemPageSize();
    auto budget = std::make_shared<PleIoBudget>();
    auto pressure = reserveCapacity(budget, pageBytes + 160, pageBytes);
    size_t const reservedBytes = budget->usedBytes();
    auto descriptor = table();
    ASSERT_TRUE(std::filesystem::remove(mDirectory / "1.bin"));
    EXPECT_THROW(PleRowStore(descriptor, 7, mStream, io(6, 160, pageBytes), PleRowStore::Residency::kCACHED, budget),
        std::length_error);
    EXPECT_EQ(budget->usedBytes(), reservedBytes);
    pressure.reset();
    EXPECT_EQ(budget->usedBytes(), 0);
}

TEST_F(PleRowStoreTest, ReaderCountsProduceIdenticalRowsAcrossShards)
{
    std::vector<std::vector<int64_t>> const requests{{25, 31, 0}, {32, 57, 63, -1, 25}};
    for (size_t const readers : {1, 4})
    {
        SCOPED_TRACE(readers);
        auto config = io(6);
        config.readers = readers;
        PleRowStore store(table(), 7, mStream, config);
        auto leases = store.prepare(requests);
        ASSERT_NO_FATAL_FAILURE(verify(store, leases, requests));
        EXPECT_EQ(store.stats().fillWaves, 1);
    }
}

TEST_F(PleRowStoreTest, ReaderFailuresRollBackAllFillsAndPreserveWarmRows)
{
    for (size_t const readers : {1, 4})
    {
        for (size_t const stagingRows : {1, 2})
        {
            SCOPED_TRACE(::testing::Message() << "readers=" << readers << " staging_rows=" << stagingRows);
            auto config = io(stagingRows);
            config.readers = readers;
            PleRowStore store(table(), 4, mStream, config);
            auto initial = store.prepare({{0}});
            initial.clear();
            std::filesystem::resize_file(mDirectory / "1.bin", 100);
            EXPECT_THROW(store.prepare({{0, 2, 40}}), std::runtime_error);
            EXPECT_EQ(store.stats().fillWaves, stagingRows == 1 ? 1U : 0U);
            auto preserved = store.prepare({{0}});
            EXPECT_EQ(store.stats().misses, 0);
            EXPECT_EQ(store.stats().readCalls, 0);
            ASSERT_NO_FATAL_FAILURE(verify(store, preserved, {{0}}));
            preserved.clear();
            writeShard(1);
            auto retry = store.prepare({{2, 40}});
            EXPECT_EQ(store.stats().misses, 2);
            ASSERT_NO_FATAL_FAILURE(verify(store, retry, {{2, 40}}));
        }
    }
}

TEST_F(PleRowStoreTest, HandlesCrossPageRowsAndBothShardTails)
{
    PleRowStore store(table(), 5, mStream, io(2, 160, 4096));
    std::vector<std::vector<int64_t>> const requests{{25, 31, 32, 63}};
    auto leases = store.prepare(requests);
    ASSERT_NO_FATAL_FAILURE(verify(store, leases, requests));
    EXPECT_EQ(store.stats().misses, 4);
    EXPECT_EQ(store.stats().readCalls, 4);
    EXPECT_EQ(store.stats().readBytes, 13 + 32 * 160 + 29 + 32 * 160);
    EXPECT_EQ(store.stats().fillWaves, 2);
}

TEST_F(PleRowStoreTest, CoalescesAdjacentNeededPagesWithinTheWindowBudget)
{
    PleRowStore store(table(), 5, mStream, io(4, 160, 8192));
    auto leases = store.prepare({{25, 31}});
    ASSERT_NO_FATAL_FAILURE(verify(store, leases, {{25, 31}}));
    EXPECT_EQ(store.stats().readCalls, 1);
    EXPECT_EQ(store.stats().readBytes, 13 + 32 * 160);
}

TEST_F(PleRowStoreTest, EvictionReloadsExactRowsAndProtectsHits)
{
    PleRowStore store(table(), 3, mStream, io(2, 160, 4096));
    auto first = store.prepare({{0, 1}});
    ASSERT_NO_FATAL_FAILURE(verify(store, first, {{0, 1}}));
    first.clear();
    auto second = store.prepare({{2, 3}});
    ASSERT_NO_FATAL_FAILURE(verify(store, second, {{2, 3}}));
    second.clear();
    auto next = store.prepare({{0}, {2}});
    ASSERT_NO_FATAL_FAILURE(verify(store, next, {{0}, {2}}));
    EXPECT_EQ(store.stats().misses, 1);
}

TEST_F(PleRowStoreTest, ResidentPreloadsAndDoesNotReadFilesDuringPrepare)
{
    auto config = io(1);
    config.preloadBytes = 7 * 160;
    PleRowStore store(table(), 65, mStream, config, PleRowStore::Residency::kRESIDENT);
    EXPECT_EQ(store.preloadStats().fillWaves, 10);
    EXPECT_EQ(store.preloadStats().readCalls, 10);
    EXPECT_EQ(store.preloadStats().readBytes, 64 * 160);
    EXPECT_EQ(store.preloadStats().usefulBytes, 64 * 160);
    EXPECT_EQ(store.stats().readCalls, 0);
    EXPECT_EQ(store.stats().readBytes, 0);
    EXPECT_EQ(store.stats().usefulBytes, 0);
    EXPECT_EQ(store.stats().fillWaves, 0);
    std::filesystem::resize_file(mDirectory / "0.bin", 0);
    std::filesystem::resize_file(mDirectory / "1.bin", 0);
    std::vector<std::vector<int64_t>> const requests{{0, 25, 63, -1, 25}, {32, 31}};
    auto leases = store.prepare(requests);
    ASSERT_NO_FATAL_FAILURE(verify(store, leases, requests));
    EXPECT_EQ(leases[0].mapping(), (std::vector<int32_t>{1, 26, 64, 0, 26}));
    auto concurrent = store.prepare({{62, 0}});
    ASSERT_NO_FATAL_FAILURE(verify(store, concurrent, {{62, 0}}));
    EXPECT_EQ(store.stats().misses, 0);
    EXPECT_EQ(store.stats().readCalls, 0);
    EXPECT_EQ(store.stats().fillWaves, 0);
    auto moved = std::move(leases[0]);
    EXPECT_THROW(leases[0].mapping(), std::logic_error);
    EXPECT_EQ(moved.mapping()[0], 1);
    moved.release();
    EXPECT_THROW(moved.mapping(), std::logic_error);
    EXPECT_THROW(store.prepare({{64}}), std::out_of_range);
    EXPECT_THROW(store.prepare({{-2}}), std::out_of_range);
}

TEST_F(PleRowStoreTest, NativeAndOddWidthRowsMatchAcrossResidencyAndEviction)
{
    for (auto const& dtype : {"F8_E4M3", "F16", "BF16"})
    {
        auto descriptor = customTable(dtype, 7);
        auto const rowBytes = descriptor.rowBytes();
        PleRowStore cached(descriptor, 3, mStream, io(1, descriptor.sourceRowBytes(), 4096));
        PleRowStore resident(
            descriptor, 9, mStream, io(3, descriptor.sourceRowBytes(), 4096), PleRowStore::Residency::kRESIDENT);
        EXPECT_EQ(resident.pool().getMemoryCapacity(), 9 * rowBytes);
        for (auto const& requests : std::vector<std::vector<std::vector<int64_t>>>{{{0, 1, -1, 0}}, {{7, 6}}, {{1, 0}}})
        {
            auto cacheLeases = cached.prepare(requests);
            auto residentLeases = resident.prepare(requests);
            ASSERT_NO_FATAL_FAILURE(verifyBytes(cached, cacheLeases, requests, rowBytes));
            ASSERT_NO_FATAL_FAILURE(verifyBytes(resident, residentLeases, requests, rowBytes));
            EXPECT_EQ(cached.stats().misses, 2);
            EXPECT_EQ(resident.stats().readCalls, 0);
        }
    }
}

TEST_F(PleRowStoreTest, RowsLargerThanReadWindowCrossManyPages)
{
    auto descriptor = customTable("BF16", 40001);
    auto const rowBytes = descriptor.rowBytes();
    PleRowStore cached(descriptor, 3, mStream, io(2, descriptor.sourceRowBytes(), 4096));
    PleRowStore resident(
        descriptor, 9, mStream, io(2, descriptor.sourceRowBytes(), 65536), PleRowStore::Residency::kRESIDENT);
    std::vector<std::vector<int64_t>> const requests{{7, 0, -1, 7}};
    auto cacheLeases = cached.prepare(requests);
    auto residentLeases = resident.prepare(requests);
    ASSERT_NO_FATAL_FAILURE(verifyBytes(cached, cacheLeases, requests, rowBytes));
    ASSERT_NO_FATAL_FAILURE(verifyBytes(resident, residentLeases, requests, rowBytes));
    EXPECT_GT(cached.stats().readCalls, 32);
    EXPECT_EQ(cached.stats().usefulBytes, 2 * rowBytes);
}

TEST_F(PleRowStoreTest, RejectsResidentCapacityMismatch)
{
    EXPECT_THROW(
        PleRowStore(table(), 64, mStream, io(2, 160, 4096), PleRowStore::Residency::kRESIDENT), std::invalid_argument);
    EXPECT_THROW(
        PleRowStore(table(), 66, mStream, io(2, 160, 4096), PleRowStore::Residency::kRESIDENT), std::invalid_argument);
}

TEST_F(PleRowStoreTest, RejectsDefaultStreamAndInvalidRequestsBeforeReads)
{
    EXPECT_THROW(PleRowStore(table(), 3, nullptr), std::invalid_argument);
    EXPECT_THROW(PleRowStore(table(), 3, mStream, io(0)), std::invalid_argument);
    EXPECT_THROW(PleRowStore(table(), 3, mStream, io(2, 160, 1024)), std::invalid_argument);
    PleRowStore store(table(), 3, mStream, io(2, 160, 4096));
    EXPECT_THROW(store.prepare({{0, 64}}), std::out_of_range);
    EXPECT_THROW(store.prepare({{0, 1, 2}}), std::length_error);
    EXPECT_EQ(store.stats().readCalls, 0);
    auto padding = store.prepare({{-1, -1}});
    ASSERT_NO_FATAL_FAILURE(verify(store, padding, {{-1, -1}}));
    EXPECT_EQ(store.stats().misses, 0);
    EXPECT_EQ(store.stats().fillWaves, 0);
}

TEST_F(PleRowStoreTest, RejectsUndersizedPayloadForEachResidency)
{
    auto config = io(2);
    config.fillBytes = 159;
    EXPECT_THROW(PleRowStore(table(), 3, mStream, config), std::invalid_argument);
    config = io(2);
    config.preloadBytes = 159;
    EXPECT_THROW(PleRowStore(table(), 65, mStream, config, PleRowStore::Residency::kRESIDENT), std::invalid_argument);
}

TEST_F(PleRowStoreTest, FillBudgetCountsSourceBytesBeforeF16Conversion)
{
    auto const descriptor = sourceTable(PleTable::DType::kFP32);
    auto config = io(2, descriptor.sourceRowBytes());
    // A partial extra source row must not increase the number of rows in a fill.
    config.fillBytes += descriptor.sourceRowBytes() - 1;
    PleRowStore store(descriptor, 5, mStream, config);
    std::vector<std::vector<int64_t>> const requests{{0, 1, 2, 3}};
    auto leases = store.prepare(requests);
    ASSERT_NO_FATAL_FAILURE(verifyConverted(store, leases, requests, PleTable::DType::kFP32));
    EXPECT_EQ(store.stats().fillWaves, 2);
    EXPECT_EQ(store.stats().usefulBytes, 4 * descriptor.sourceRowBytes());
}

TEST_F(PleRowStoreTest, NativeCheckpointRowsCastBeforeScalingInBothPolicies)
{
    float const nonHalfValue = sourceValue(PleTable::DType::kFP32, 0, 0);
    ASSERT_NE(__half2float(__float2half_rn(nonHalfValue * 1.5F)),
        __half2float(__float2half_rn(__half2float(__float2half_rn(nonHalfValue)) * 1.5F)));
    for (auto const dtype : {PleTable::DType::kFP16, PleTable::DType::kBF16, PleTable::DType::kFP32})
    {
        auto const descriptor = sourceTable(dtype);
        PleRowStore cached(descriptor, 3, mStream, io(1, descriptor.sourceRowBytes(), 4096));
        PleRowStore resident(
            descriptor, 9, mStream, io(2, descriptor.sourceRowBytes(), 4096), PleRowStore::Residency::kRESIDENT);
        EXPECT_EQ(cached.pool().getMemoryCapacity(), 3 * 7 * 2);
        EXPECT_EQ(resident.pool().getMemoryCapacity(), 9 * 7 * 2);
        for (auto const& requests : std::vector<std::vector<std::vector<int64_t>>>{{{0, 1, -1, 0}}, {{7, 6}}, {{0, 1}}})
        {
            auto cachedLeases = cached.prepare(requests);
            auto residentLeases = resident.prepare(requests);
            ASSERT_NO_FATAL_FAILURE(verifyConverted(cached, cachedLeases, requests, dtype));
            ASSERT_NO_FATAL_FAILURE(verifyConverted(resident, residentLeases, requests, dtype));
            EXPECT_EQ(cached.stats().usefulBytes, 2 * descriptor.sourceRowBytes());
            EXPECT_EQ(resident.stats().readBytes, 0);
        }
    }
}

TEST_F(PleRowStoreTest, CheckpointMissesReadOnlyBoundedSourcePages)
{
    auto const descriptor = sourceTable(PleTable::DType::kFP32, 4096, 3);
    PleRowStore cached(descriptor, 3, mStream, io(2, descriptor.sourceRowBytes(), 4096));
    auto leases = cached.prepare({{0, 1, -1, 0}});
    ASSERT_NO_FATAL_FAILURE(verifyConverted(cached, leases, {{0, 1, -1, 0}}, PleTable::DType::kFP32, 3));
    EXPECT_EQ(cached.stats().readCalls, 1);
    EXPECT_EQ(cached.stats().readBytes, 4096);
    EXPECT_EQ(cached.stats().usefulBytes, 2 * 3 * 4);
    EXPECT_EQ(cached.stats().fillWaves, 1);
    EXPECT_LT(cached.stats().readBytes, std::filesystem::file_size(mDirectory / "source.bin"));
    leases.clear();
    auto warm = cached.prepare({{1, 0}});
    ASSERT_NO_FATAL_FAILURE(verifyConverted(cached, warm, {{1, 0}}, PleTable::DType::kFP32, 3));
    EXPECT_EQ(cached.stats().readBytes, 0);
    EXPECT_EQ(cached.stats().fillWaves, 0);
}

TEST_F(PleRowStoreTest, ResidentSourceCanBeUnlinkedWhileCachedMissRequiresReadableBytes)
{
    PleRowStore resident(sourceTable(PleTable::DType::kBF16), 9, mStream, io(2, 14), PleRowStore::Residency::kRESIDENT);
    ASSERT_TRUE(std::filesystem::remove(mDirectory / "source.bin"));
    auto residentLeases = resident.prepare({{7, 0, -1}});
    ASSERT_NO_FATAL_FAILURE(verifyConverted(resident, residentLeases, {{7, 0, -1}}, PleTable::DType::kBF16));
    EXPECT_EQ(resident.stats().readBytes, 0);

    PleRowStore cached(sourceTable(PleTable::DType::kBF16), 3, mStream, io(1, 14));
    auto initial = cached.prepare({{0}});
    ASSERT_NO_FATAL_FAILURE(verifyConverted(cached, initial, {{0}}, PleTable::DType::kBF16));
    initial.clear();
    std::filesystem::resize_file(mDirectory / "source.bin", 13);
    EXPECT_THROW(cached.prepare({{7}}), std::runtime_error);
    auto preserved = cached.prepare({{0}});
    ASSERT_NO_FATAL_FAILURE(verifyConverted(cached, preserved, {{0}}, PleTable::DType::kBF16));
    EXPECT_EQ(cached.stats().readCalls, 0);
}

} // namespace
} // namespace rt
} // namespace trt_edgellm
