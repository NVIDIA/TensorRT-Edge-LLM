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

#include "runtime/embedding/pleIoBudget.h"

#include "common/logger.h"
#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{

constexpr size_t kPAGE_BYTES = 4096;
constexpr size_t kROW_BYTES = 160;
constexpr size_t kMIB = 1U << 20;

size_t stagingBytes(size_t rows, size_t rowBytes)
{
    return (rows * rowBytes + 3) / 4 * 4 + rows * 4;
}

std::unique_ptr<PleIoBudget::Reservation> occupy(std::shared_ptr<PleIoBudget> const& budget, size_t bytes)
{
    PleIoConfig io;
    io.preloadBytes = bytes / 2;
    return budget->reserve(io, 4, bytes / 8, kPAGE_BYTES, false);
}

class PleIoBudgetTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mLevel = gLogger.getLevel();
        gLogger.setLevel(trt_edgellm::logger::Severity::kWARNING);
    }
    void TearDown() override
    {
        gLogger.setLevel(mLevel);
    }
    trt_edgellm::logger::Severity mLevel{};
};

TEST_F(PleIoBudgetTest, DefaultsPreserveRequestedBatches)
{
    auto budget = std::make_shared<PleIoBudget>();
    PleIoConfig const io;
    auto cached = budget->reserve(io, kROW_BYTES, 1000000, kPAGE_BYTES, true);
    EXPECT_FALSE(cached->adjusted());
    EXPECT_EQ(cached->rows(), io.fillBytes / kROW_BYTES);
    EXPECT_EQ(cached->readers(), io.readers);
    EXPECT_EQ(cached->readBytes(), io.readBytes);
    EXPECT_EQ(cached->hostBytes(), stagingBytes(cached->rows(), kROW_BYTES) + io.readers * io.readBytes);
    auto resident = budget->reserve(io, kROW_BYTES, 1000000, kPAGE_BYTES, false);
    EXPECT_FALSE(resident->adjusted());
    EXPECT_EQ(resident->rows(), io.preloadBytes / kROW_BYTES);
    EXPECT_EQ(resident->readers(), 0U);
    EXPECT_EQ(resident->readBytes(), 0U);
    EXPECT_EQ(resident->hostBytes(), stagingBytes(resident->rows(), kROW_BYTES));
    EXPECT_EQ(budget->usedBytes(), cached->hostBytes() + resident->hostBytes());
}

TEST_F(PleIoBudgetTest, SmallTableAndUnalignedRowsAccountOnlyActualStorage)
{
    auto budget = std::make_shared<PleIoBudget>();
    auto reservation = budget->reserve({}, 161, 3, kPAGE_BYTES, false);
    EXPECT_EQ(reservation->rows(), 3U);
    EXPECT_EQ(reservation->hostBytes(), 496U);
    EXPECT_FALSE(reservation->adjusted());
}

TEST_F(PleIoBudgetTest, OversizedUploadShrinksRowsBeforeReaders)
{
    auto budget = std::make_shared<PleIoBudget>();
    PleIoConfig io;
    io.fillBytes = 128 * kMIB;
    auto reservation = budget->reserve(io, kROW_BYTES, 1000000, kPAGE_BYTES, true);
    EXPECT_TRUE(reservation->adjusted());
    EXPECT_LT(reservation->rows(), io.fillBytes / kROW_BYTES);
    EXPECT_EQ(reservation->readers(), io.readers);
    EXPECT_EQ(reservation->readBytes(), io.readBytes);
    EXPECT_LE(reservation->hostBytes(), PleIoBudget::kLIMIT_BYTES);
    EXPECT_GT(stagingBytes(reservation->rows() + 1, kROW_BYTES) + io.readers * io.readBytes, PleIoBudget::kLIMIT_BYTES);
}

TEST_F(PleIoBudgetTest, OversizedReadBuffersKeepConcurrencyWhenPagesFit)
{
    auto budget = std::make_shared<PleIoBudget>();
    PleIoConfig io;
    io.readBytes = 128 * kMIB;
    auto reservation = budget->reserve(io, kROW_BYTES, 1000000, kPAGE_BYTES, true);
    EXPECT_TRUE(reservation->adjusted());
    EXPECT_EQ(reservation->readers(), io.readers);
    EXPECT_LT(reservation->readBytes(), io.readBytes);
    EXPECT_EQ(reservation->readBytes() % kPAGE_BYTES, 0U);
    EXPECT_GE(reservation->rows(), 1U);
    EXPECT_LE(reservation->hostBytes(), PleIoBudget::kLIMIT_BYTES);
}

TEST_F(PleIoBudgetTest, SharedAllocationsFirstReduceUploadRows)
{
    auto budget = std::make_shared<PleIoBudget>();
    auto first = occupy(budget, PleIoBudget::kLIMIT_BYTES - 6 * kMIB);
    auto second = budget->reserve({}, kROW_BYTES, 1000000, kPAGE_BYTES, true);
    EXPECT_TRUE(second->adjusted());
    EXPECT_EQ(second->readers(), 4U);
    EXPECT_EQ(second->readBytes(), kMIB);
    EXPECT_LE(stagingBytes(second->rows(), kROW_BYTES), 2 * kMIB);
    EXPECT_EQ(budget->usedBytes(), first->hostBytes() + second->hostBytes());
    EXPECT_LE(budget->usedBytes(), PleIoBudget::kLIMIT_BYTES);
}

TEST_F(PleIoBudgetTest, SharedAllocationsReduceConcurrencyWhenNecessary)
{
    auto budget = std::make_shared<PleIoBudget>();
    auto first = occupy(budget, PleIoBudget::kLIMIT_BYTES - 2 * kPAGE_BYTES);
    auto second = budget->reserve({}, kROW_BYTES, 1000000, kPAGE_BYTES, true);
    EXPECT_TRUE(second->adjusted());
    EXPECT_EQ(second->readers(), 1U);
    EXPECT_EQ(second->readBytes(), kPAGE_BYTES);
    EXPECT_EQ(second->rows(), 24U);
    EXPECT_LE(budget->usedBytes(), PleIoBudget::kLIMIT_BYTES);
}

TEST_F(PleIoBudgetTest, UnalignedMinimumFitsExactly)
{
    auto budget = std::make_shared<PleIoBudget>();
    size_t const minimum = stagingBytes(1, 161) + kPAGE_BYTES;
    auto first = occupy(budget, PleIoBudget::kLIMIT_BYTES - minimum);
    PleIoConfig io;
    io.fillBytes = 161;
    io.readBytes = kPAGE_BYTES;
    io.readers = 1;
    auto second = budget->reserve(io, 161, 1, kPAGE_BYTES, true);
    EXPECT_FALSE(second->adjusted());
    EXPECT_EQ(second->rows(), 1U);
    EXPECT_EQ(second->hostBytes(), minimum);
    EXPECT_EQ(budget->usedBytes(), PleIoBudget::kLIMIT_BYTES);
}

TEST_F(PleIoBudgetTest, ConcurrentReservationsShareOneLimit)
{
    auto budget = std::make_shared<PleIoBudget>();
    std::vector<std::unique_ptr<PleIoBudget::Reservation>> reservations(8);
    std::vector<std::exception_ptr> unexpected(8);
    std::atomic<size_t> rejected{0};
    std::vector<std::thread> threads;
    for (size_t index = 0; index < reservations.size(); ++index)
    {
        threads.emplace_back([&, index] {
            try
            {
                reservations[index] = occupy(budget, 16 * kMIB);
            }
            catch (std::length_error const&)
            {
                ++rejected;
            }
            catch (...)
            {
                unexpected[index] = std::current_exception();
            }
        });
    }
    for (auto& thread : threads)
    {
        thread.join();
    }
    for (auto const& error : unexpected)
    {
        EXPECT_FALSE(error);
    }
    EXPECT_EQ(rejected.load(), 4U);
    EXPECT_EQ(budget->usedBytes(), PleIoBudget::kLIMIT_BYTES);
    reservations.clear();
    EXPECT_EQ(budget->usedBytes(), 0U);
}

TEST_F(PleIoBudgetTest, ImpossibleMinimumFailsWithoutChangingAccounting)
{
    auto budget = std::make_shared<PleIoBudget>();
    auto first = occupy(budget, PleIoBudget::kLIMIT_BYTES);
    testing::internal::CaptureStderr();
    EXPECT_THROW(budget->reserve({}, kROW_BYTES, 1, kPAGE_BYTES, true), std::length_error);
    auto const warning = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(warning.empty());
    EXPECT_EQ(budget->usedBytes(), first->hostBytes());
}

TEST_F(PleIoBudgetTest, SingleRowExceedingLimitReportsRequiredMinimum)
{
    auto budget = std::make_shared<PleIoBudget>();
    PleIoConfig io;
    io.preloadBytes = PleIoBudget::kLIMIT_BYTES;
    try
    {
        budget->reserve(io, PleIoBudget::kLIMIT_BYTES, 1, kPAGE_BYTES, false);
        FAIL() << "An oversized row must fail before allocation";
    }
    catch (std::length_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("requires 67108868 bytes"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("available=67108864"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("limit=67108864"), std::string::npos);
    }
    EXPECT_EQ(budget->usedBytes(), 0U);
}

TEST_F(PleIoBudgetTest, InvalidSettingsAreNotSilentlyClamped)
{
    auto budget = std::make_shared<PleIoBudget>();
    testing::internal::CaptureStderr();
    EXPECT_THROW(budget->reserve({}, 0, 1, kPAGE_BYTES, true), std::invalid_argument);
    EXPECT_THROW(budget->reserve({}, kROW_BYTES, 0, kPAGE_BYTES, true), std::invalid_argument);
    EXPECT_THROW(budget->reserve({}, kROW_BYTES, 1, 0, true), std::invalid_argument);
    PleIoConfig io;
    io.readers = 33;
    EXPECT_THROW(budget->reserve(io, kROW_BYTES, 1, kPAGE_BYTES, true), std::invalid_argument);
    io.readers = 0;
    EXPECT_THROW(budget->reserve(io, kROW_BYTES, 1, kPAGE_BYTES, false), std::invalid_argument);
    io = {};
    io.fillBytes = kROW_BYTES - 1;
    EXPECT_THROW(budget->reserve(io, kROW_BYTES, 1, kPAGE_BYTES, true), std::invalid_argument);
    io = {};
    io.readBytes = kPAGE_BYTES - 1;
    EXPECT_THROW(budget->reserve(io, kROW_BYTES, 1, kPAGE_BYTES, true), std::invalid_argument);
    io.readBytes = kPAGE_BYTES + 1;
    EXPECT_THROW(budget->reserve(io, kROW_BYTES, 1, kPAGE_BYTES, true), std::invalid_argument);
    auto const warning = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(warning.empty());
    EXPECT_EQ(budget->usedBytes(), 0U);
}

TEST_F(PleIoBudgetTest, ArithmeticOverflowIsRejectedBeforeClamping)
{
    auto budget = std::make_shared<PleIoBudget>();
    size_t const tensorLimit = static_cast<size_t>(std::numeric_limits<int64_t>::max());
    PleIoConfig io;
    testing::internal::CaptureStderr();
    io.preloadBytes = tensorLimit;
    EXPECT_THROW(budget->reserve(io, tensorLimit, 1, kPAGE_BYTES, false), std::overflow_error);
    io.preloadBytes = tensorLimit - 3;
    EXPECT_THROW(budget->reserve(io, tensorLimit - 3, 1, kPAGE_BYTES, false), std::overflow_error);
    io.preloadBytes = tensorLimit;
    EXPECT_THROW(budget->reserve(io, 1, tensorLimit, kPAGE_BYTES, false), std::overflow_error);
    io = {};
    io.readBytes = tensorLimit / kPAGE_BYTES * kPAGE_BYTES;
    EXPECT_THROW(budget->reserve(io, kROW_BYTES, 1, kPAGE_BYTES, true), std::overflow_error);
    auto const warning = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(warning.empty());
    EXPECT_EQ(budget->usedBytes(), 0U);
}

TEST_F(PleIoBudgetTest, DestructionReturnsBudgetForReuseAndKeepsOwnerAlive)
{
    auto budget = std::make_shared<PleIoBudget>();
    auto full = occupy(budget, PleIoBudget::kLIMIT_BYTES);
    full.reset();
    EXPECT_EQ(budget->usedBytes(), 0U);
    auto next = budget->reserve({}, kROW_BYTES, 1000000, kPAGE_BYTES, true);
    EXPECT_FALSE(next->adjusted());
    std::weak_ptr<PleIoBudget> owner = budget;
    budget.reset();
    EXPECT_FALSE(owner.expired());
    next.reset();
    EXPECT_TRUE(owner.expired());
}

TEST_F(PleIoBudgetTest, AdjustmentWarnsOnceAndIncludesEffectiveConfiguration)
{
    auto budget = std::make_shared<PleIoBudget>();
    PleIoConfig io;
    io.fillBytes = 128 * kMIB;
    testing::internal::CaptureStderr();
    auto reservation = budget->reserve(io, kROW_BYTES, 1000000, kPAGE_BYTES, true);
    budget->usedBytes();
    reservation.reset();
    auto const warning = testing::internal::GetCapturedStderr();
    std::string const marker = "PLE I/O budget adjusted";
    auto const position = warning.find(marker);
    ASSERT_NE(position, std::string::npos);
    EXPECT_EQ(warning.find(marker, position + marker.size()), std::string::npos);
    EXPECT_NE(warning.find("readers 4 -> 4"), std::string::npos);
    EXPECT_NE(warning.find("available=67108864, limit=67108864"), std::string::npos);
}

} // namespace
} // namespace rt
} // namespace trt_edgellm
