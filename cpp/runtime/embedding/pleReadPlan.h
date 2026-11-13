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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Requested source bytes and their destination in a caller-owned packed buffer.
struct PleReadSpan
{
    size_t shard;
    int64_t offset;
    size_t bytes;
    size_t destination;
};

//! Byte offsets relative to one read buffer and the caller's packed destination buffer.
struct PleReadCopy
{
    size_t source;
    size_t destination;
    size_t bytes;
};

//! One physical read and the useful byte ranges to extract from it.
struct PleReadRange
{
    size_t shard;
    int64_t offset;
    size_t bytes;
    std::vector<PleReadCopy> copies;
};

//! Plan non-overlapping reads sorted by shard and offset, merging only aligned overlap or adjacency.
//! Reads are alignment-sized except at shard ends, and no read exceeds maxReadBytes.
//! Empty input returns no reads. Spans must be nonempty and fit their shard; destination regions must be distinct.
//! alignment must be positive and maxReadBytes must be a positive multiple of it, both fitting INT64.
std::vector<PleReadRange> planPleReads(std::vector<PleReadSpan> const& spans, std::vector<int64_t> const& shardEnds,
    size_t alignment, size_t maxReadBytes);

} // namespace rt
} // namespace trt_edgellm
