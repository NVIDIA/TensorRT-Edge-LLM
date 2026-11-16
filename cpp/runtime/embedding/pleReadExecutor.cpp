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

#include "pleReadExecutor.h"
#include "common/tensor.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace trt_edgellm
{
namespace rt
{

struct PleReadExecutor::State
{
    struct Reader
    {
        explicit Reader(size_t bytes)
            : buffer({static_cast<int64_t>(bytes)}, DeviceType::kCPU, trt_edgellm::DataType::kINT8)
        {
        }
        Tensor buffer;
        std::thread thread;
        size_t calls{};
        size_t bytes{};
        std::exception_ptr error;
    };

    State(std::vector<PleFileHandle> const& files_, size_t readerCount, size_t maxReadBytes)
        : files(files_)
    {
        constexpr size_t kMAX_READERS{32};
        if (readerCount == 0 || readerCount > kMAX_READERS || maxReadBytes == 0
            || maxReadBytes > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
        {
            throw std::invalid_argument("PLE readers require 1..32 workers and a positive bounded read size");
        }
        readers.reserve(readerCount);
        for (size_t index = 0; index < readerCount; ++index)
        {
            readers.emplace_back(maxReadBytes);
        }
        try
        {
            for (size_t index = 0; index < readerCount; ++index)
            {
                readers[index].thread = std::thread([this, index] { run(readers[index]); });
            }
        }
        catch (...)
        {
            stop();
            throw;
        }
    }

    ~State() noexcept
    {
        stop();
    }

    void stop() noexcept
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        for (auto& reader : readers)
        {
            if (reader.thread.joinable())
            {
                reader.thread.join();
            }
        }
    }

    void run(Reader& reader)
    {
        size_t observedGeneration{};
        std::unique_lock<std::mutex> lock(mutex);
        for (;;)
        {
            ready.wait(lock, [&] { return stopping || generation != observedGeneration; });
            if (stopping)
            {
                return;
            }
            observedGeneration = generation;
            lock.unlock();
            try
            {
                for (size_t index = next.fetch_add(1, std::memory_order_relaxed); index < ranges->size();
                    index = next.fetch_add(1, std::memory_order_relaxed))
                {
                    auto const& range = (*ranges)[index];
                    char* buffer = static_cast<char*>(reader.buffer.rawPointer());
                    size_t copied{};
                    while (copied < range.bytes)
                    {
                        ++reader.calls;
                        size_t const count = readPleFileAt(files[range.shard], buffer + copied, range.bytes - copied,
                            range.offset + static_cast<int64_t>(copied), "Reading PLE rows");
                        if (count == 0)
                        {
                            throw std::runtime_error("PLE shard ended before the requested read range");
                        }
                        copied += count;
                        reader.bytes += count;
                    }
                    for (auto const& copy : range.copies)
                    {
                        std::memcpy(packed + copy.destination, buffer + copy.source, copy.bytes);
                    }
                }
            }
            catch (...)
            {
                reader.error = std::current_exception();
            }
            lock.lock();
            if (--remaining == 0)
            {
                finished.notify_one();
            }
        }
    }

    std::vector<PleFileHandle> const files;
    std::vector<Reader> readers;
    std::mutex mutex;
    std::condition_variable ready;
    std::condition_variable finished;
    std::atomic<size_t> next{};
    std::vector<PleReadRange> const* ranges{};
    char* packed{};
    size_t generation{};
    size_t remaining{};
    bool stopping{};
};

PleReadExecutor::PleReadExecutor(std::vector<PleFileHandle> const& files, size_t readerCount, size_t maxReadBytes)
    : mState(std::make_unique<State>(files, readerCount, maxReadBytes))
{
}

PleReadExecutor::~PleReadExecutor() noexcept = default;

void PleReadExecutor::read(std::vector<PleReadRange> const& ranges, char* packed, size_t& readCalls, size_t& readBytes)
{
    if (ranges.empty())
    {
        return;
    }
    auto& state = *mState;
    std::unique_lock<std::mutex> lock(state.mutex);
    state.ranges = &ranges;
    state.packed = packed;
    state.next.store(0, std::memory_order_relaxed);
    state.remaining = state.readers.size();
    for (auto& reader : state.readers)
    {
        reader.calls = 0;
        reader.bytes = 0;
        reader.error = nullptr;
    }
    ++state.generation;
    state.ready.notify_all();
    state.finished.wait(lock, [&] { return state.remaining == 0; });
    std::exception_ptr error;
    for (auto const& reader : state.readers)
    {
        readCalls += reader.calls;
        readBytes += reader.bytes;
        if (!error && reader.error)
        {
            error = reader.error;
        }
    }
    state.ranges = nullptr;
    state.packed = nullptr;
    if (error)
    {
        std::rethrow_exception(error);
    }
}

} // namespace rt
} // namespace trt_edgellm
