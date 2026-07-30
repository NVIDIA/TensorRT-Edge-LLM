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

#include <cstdint>

namespace trt_edgellm::rt
{

//! Page size P for the paged-KV layout (tokens per page).
constexpr int32_t kTOKENS_PER_PAGE{128};

//! Sentinel value for an unallocated page-table entry.
constexpr int32_t kUNUSED_PAGE_ENTRY{-1};

//! Number of pages a single slot's padded token capacity spans.
inline int32_t pagesPerSlot(int32_t maxCapPadded)
{
    return maxCapPadded / kTOKENS_PER_PAGE;
}

//! Maximum number of pages one sequence's KV capacity spans (the page table's last dim).
inline int32_t computeMaxPagesPerSeq(int32_t maxKVCacheCapacity)
{
    return (maxKVCacheCapacity + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE;
}

//! The active-capacity floor: the minimum page count of the paged-KV pool, i.e.
//! `maxBatchSize * ceil(maxKVCacheCapacity / kTOKENS_PER_PAGE)`. Identity-mapped active slots
//! occupy these first pages (see `KVPageTable`). The builder may serialize a larger exact pool
//! count for retained cache entries; its optimization profile, the runtime registry's binding
//! shape, and `KVCacheManager`'s allocation must all use that same count.
inline int32_t computeKvPoolFloorPages(int32_t maxBatchSize, int32_t maxKVCacheCapacity)
{
    return maxBatchSize * computeMaxPagesPerSeq(maxKVCacheCapacity);
}

} // namespace trt_edgellm::rt
