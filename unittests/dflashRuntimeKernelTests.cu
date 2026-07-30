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

// checkDFlashPageTableIdentity: DFlash's target-KV update stays contiguous (identity-only, per
// DFlash opts out of KV-cache reuse). This guard makes that assumption explicit
// and checkable: it must accept a slot whose K row is the static identity range and reject any
// deviation (a remapped/scrambled page, or a short row with an unallocated tail).

#include "kernels/speculative/dflashRuntimeKernels.h"
#include "runtime/state/kvPageTable.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;

TEST(DFlashRuntimeKernels, CheckPageTableIdentityAcceptsIdentityRows)
{
    cudaStream_t stream{nullptr};
    int32_t const maxBatch = 3;
    int32_t const maxPagesPerSeq = 4;
    rt::KVPageTable pageTable(maxBatch, maxPagesPerSeq, maxBatch * maxPagesPerSeq);
    pageTable.setIdentity();
    pageTable.upload(stream);

    for (int32_t b = 0; b < maxBatch; ++b)
    {
        EXPECT_NO_THROW(kernel::checkDFlashPageTableIdentity(pageTable.hostRow(b), b, maxPagesPerSeq));
    }
}

TEST(DFlashRuntimeKernels, CheckPageTableIdentityRejectsScrambledRow)
{
    int32_t const maxPagesPerSeq = 4;
    // Slot 1's identity range would be [4, 8); scramble one entry.
    std::vector<int32_t> const scrambledRow = {4, 5, 9, 7};
    EXPECT_THROW(
        kernel::checkDFlashPageTableIdentity(scrambledRow.data(), /*slot=*/1, maxPagesPerSeq), std::runtime_error);
}

TEST(DFlashRuntimeKernels, CheckPageTableIdentityRejectsUnallocatedTail)
{
    int32_t const maxPagesPerSeq = 4;
    // Slot 0's identity range is [0, 4); a short row (unallocated tail) is not identity either.
    std::vector<int32_t> const shortRow = {0, 1, -1, -1};
    EXPECT_THROW(kernel::checkDFlashPageTableIdentity(shortRow.data(), /*slot=*/0, maxPagesPerSeq), std::runtime_error);
}

// checkDFlashRopeCapacity: the KV pool's per-slot capacity (`cap`) is PADDED up to a multiple of
// kTOKENS_PER_PAGE, while the RoPE cache is sized to the real (unpadded) configured max sequence
// length. Whenever that configured length is not itself page-aligned, cosSinSeqLen < cap is the
// NORMAL case, not an error — this is the exact class of bug (padded vs. unpadded capacity) that
// previously caused a spurious rejection on every enqueue() call for a non-128-aligned config.

TEST(DFlashRuntimeKernels, CheckRopeCapacityAcceptsNonPageAlignedCapacity)
{
    // maxKVCacheCapacity=4000 (not a multiple of 128) -> capPadded=4096. This must NOT throw.
    EXPECT_NO_THROW(kernel::checkDFlashRopeCapacity(/*cosSinSeqLen=*/4000, /*kvCapacity=*/4096));
}

TEST(DFlashRuntimeKernels, CheckRopeCapacityAcceptsExactlyPageAlignedCapacity)
{
    EXPECT_NO_THROW(kernel::checkDFlashRopeCapacity(/*cosSinSeqLen=*/4096, /*kvCapacity=*/4096));
}

TEST(DFlashRuntimeKernels, CheckRopeCapacityRejectsSeqLenExceedingCap)
{
    // A rope cache sized past the KV pool's own padded capacity indicates a genuine mismatch
    // (e.g. bound to the wrong tensor / wrong model's config).
    EXPECT_THROW(kernel::checkDFlashRopeCapacity(/*cosSinSeqLen=*/5000, /*kvCapacity=*/4096), std::runtime_error);
}
