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

#include "runtime/embedding/pleReadPlan.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace trt_edgellm
{
namespace rt
{

std::vector<PleReadRange> planPleReads(
    std::vector<PleReadSpan> const& spans, std::vector<int64_t> const& shardEnds, size_t alignment, size_t maxReadBytes)
{
    auto const maxOffset = static_cast<size_t>(std::numeric_limits<int64_t>::max());
    if (alignment == 0 || alignment > maxOffset || maxReadBytes < alignment || maxReadBytes > maxOffset
        || maxReadBytes % alignment != 0)
    {
        throw std::invalid_argument("PLE read budget must be an INT64-sized positive multiple of alignment");
    }
    if (std::any_of(shardEnds.begin(), shardEnds.end(), [](int64_t end) { return end < 0; }))
    {
        throw std::invalid_argument("PLE shard ends must be nonnegative");
    }
    std::vector<PleReadRange> merged;
    merged.reserve(spans.size());
    auto const unit = static_cast<int64_t>(alignment);
    for (auto const& span : spans)
    {
        if (span.shard >= shardEnds.size() || span.offset < 0 || span.offset > shardEnds[span.shard] || span.bytes == 0
            || span.bytes > static_cast<size_t>(shardEnds[span.shard] - span.offset))
        {
            throw std::out_of_range("PLE read span must be nonempty and fit its shard");
        }
        if (span.bytes > std::numeric_limits<size_t>::max() - span.destination)
        {
            throw std::overflow_error("PLE packed destination range overflows");
        }
        int64_t const start = span.offset - span.offset % unit;
        int64_t const end = span.offset + static_cast<int64_t>(span.bytes);
        int64_t const padding = end % unit == 0 ? 0 : unit - end % unit;
        int64_t const alignedEnd = end + std::min(padding, shardEnds[span.shard] - end);
        merged.push_back({span.shard, start, static_cast<size_t>(alignedEnd - start), {}});
    }
    std::sort(merged.begin(), merged.end(), [](PleReadRange const& left, PleReadRange const& right) {
        return std::tie(left.shard, left.offset) < std::tie(right.shard, right.offset);
    });
    size_t count = 0;
    for (size_t index = 0; index < merged.size(); ++index)
    {
        auto const& next = merged[index];
        if (count > 0 && merged[count - 1].shard == next.shard
            && next.offset <= merged[count - 1].offset + static_cast<int64_t>(merged[count - 1].bytes))
        {
            auto& previous = merged[count - 1];
            previous.bytes = std::max(previous.bytes, static_cast<size_t>(next.offset - previous.offset) + next.bytes);
        }
        else
        {
            merged[count++] = next;
        }
    }
    merged.resize(count);

    std::vector<PleReadRange> ranges;
    size_t rangeCount = 0;
    for (auto const& interval : merged)
    {
        size_t const splits = 1 + (interval.bytes - 1) / maxReadBytes;
        if (splits > ranges.max_size() - rangeCount)
        {
            throw std::length_error("PLE read plan exceeds addressable metadata size");
        }
        rangeCount += splits;
    }
    ranges.reserve(rangeCount);
    for (auto const& interval : merged)
    {
        size_t consumed = 0;
        while (consumed < interval.bytes)
        {
            size_t const bytes = std::min(maxReadBytes, interval.bytes - consumed);
            ranges.push_back({interval.shard, interval.offset + static_cast<int64_t>(consumed), bytes, {}});
            consumed += bytes;
        }
    }
    for (auto const& span : spans)
    {
        auto range = std::lower_bound(
            ranges.begin(), ranges.end(), span, [](PleReadRange const& read, PleReadSpan const& requested) {
                return read.shard < requested.shard
                    || (read.shard == requested.shard
                        && read.offset + static_cast<int64_t>(read.bytes) <= requested.offset);
            });
        int64_t const end = span.offset + static_cast<int64_t>(span.bytes);
        for (; range != ranges.end() && range->shard == span.shard && range->offset < end; ++range)
        {
            int64_t const start = std::max(span.offset, range->offset);
            int64_t const copyEnd = std::min(end, range->offset + static_cast<int64_t>(range->bytes));
            range->copies.push_back({static_cast<size_t>(start - range->offset),
                span.destination + static_cast<size_t>(start - span.offset), static_cast<size_t>(copyEnd - start)});
        }
    }
    return ranges;
}

} // namespace rt
} // namespace trt_edgellm
