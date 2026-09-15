/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "runtime/state/sequencePolicy.h"
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>

namespace trt_edgellm
{
namespace rt
{
struct SequenceUpdate
{
    SequenceSample sample;
    std::string text;
};
struct SequenceRead
{
    std::optional<SequenceUpdate> update;
    bool closed{};
};
//! Single-consumer bounded stream; only readers allocate per update. Terminal notification cannot be blocked by
//! fullness.
class SequenceChannel
{
public:
    SequenceChannel(size_t records, size_t bytes);
    bool push(SequenceSample const& sample, std::string_view text);
    SequenceRead read(std::chrono::milliseconds timeout);
    void close();

private:
    struct Entry
    {
        SequenceSample sample;
        size_t bytes{};
    };
    std::vector<Entry> mRecords;
    std::vector<char> mBytes;
    std::mutex mMutex;
    std::condition_variable mWake;
    size_t mHead{}, mCount{}, mByteHead{}, mByteCount{};
    bool mClosed{};
};
} // namespace rt
} // namespace trt_edgellm
