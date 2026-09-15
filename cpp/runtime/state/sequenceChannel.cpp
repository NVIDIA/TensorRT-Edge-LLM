/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime/state/sequenceChannel.h"
#include <stdexcept>
namespace trt_edgellm
{
namespace rt
{
SequenceChannel::SequenceChannel(size_t records, size_t bytes)
    : mRecords(records)
    , mBytes(bytes)
{
    if (!records || !bytes)
    {
        throw std::invalid_argument("Empty stream capacity");
    }
}
bool SequenceChannel::push(SequenceSample const& sample, std::string_view text)
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (mClosed || mCount == mRecords.size() || text.size() > mBytes.size() - mByteCount)
    {
        return false;
    }
    auto& entry = mRecords[(mHead + mCount) % mRecords.size()];
    entry.sample = sample;
    entry.bytes = text.size();
    for (size_t i = 0; i < text.size(); ++i)
    {
        mBytes[(mByteHead + mByteCount + i) % mBytes.size()] = text[i];
    }
    ++mCount;
    mByteCount += text.size();
    mWake.notify_one();
    return true;
}
SequenceRead SequenceChannel::read(std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lock(mMutex);
    mWake.wait_for(lock, timeout, [&] { return mClosed || mCount; });
    if (!mCount)
    {
        return {std::nullopt, mClosed};
    }
    auto const& entry = mRecords[mHead];
    SequenceUpdate update;
    update.sample = entry.sample;
    update.text.resize(entry.bytes);
    for (size_t i = 0; i < entry.bytes; ++i)
    {
        update.text[i] = mBytes[(mByteHead + i) % mBytes.size()];
    }
    mByteHead = (mByteHead + entry.bytes) % mBytes.size();
    mByteCount -= entry.bytes;
    mHead = (mHead + 1) % mRecords.size();
    --mCount;
    return {std::move(update), mClosed && mCount == 0};
}
void SequenceChannel::close()
{
    std::lock_guard<std::mutex> lock(mMutex);
    mClosed = true;
    mWake.notify_all();
}
} // namespace rt
} // namespace trt_edgellm
