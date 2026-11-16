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

#include <atomic>
#include <future>
#include <memory>
#include <thread>

class MemoryBackend;

//! Memory monitor for examples.
class MemoryMonitor
{
public:
    MemoryMonitor();
    ~MemoryMonitor();

    void start();
    void stop();

    //! Get the peak value reported by the selected GPU memory backend.
    size_t getPeakGpuMemory() const;

    //! Get peak CPU-visible process memory from the RSS high-water mark.
    size_t getPeakCpuMemory() const;

    //! Get the selected GPU memory metric identifier.
    char const* getGpuMemoryMetric() const;

private:
    void monitor();

    std::atomic_bool mActive{false};
    std::atomic_bool mSampleFailed{false};
    std::future<void> mTask;
    std::atomic<size_t> mPeakGpuMemory{0};
    std::unique_ptr<MemoryBackend> mBackend;
};
