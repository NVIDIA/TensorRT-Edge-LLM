/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <thread>

//! Memory monitor for examples
//! Automatically detects iGPU vs dGPU on start() and adjusts monitoring accordingly:
//! - iGPU: Monitors unified memory using CPU memory (RSS)
//! - dGPU: Monitors both GPU memory and CPU memory
class MemoryMonitor
{
public:
    MemoryMonitor()
        : mActive(false)
        , mPeakGpuMemory(0)
        , mBaselineGpuFreeMemory(0)
        , mIsIGPU(false)
    {
    }

    void start();
    void stop();

    //! Get peak GPU memory in bytes (returns 0 for iGPU)
    size_t getPeakGpuMemory() const;

    //! Get peak CPU memory (RSS) in bytes
    size_t getPeakCpuMemory() const;

    //! Get peak unified memory in bytes (for iGPU systems)
    size_t getPeakUnifiedMemory() const;

    //! Check if device is integrated GPU
    bool isIntegratedGPU() const
    {
        return mIsIGPU;
    }

private:
    void monitor();

    std::atomic_bool mActive{false};
    std::future<void> mTask;
    std::atomic<size_t> mPeakGpuMemory{0};
    size_t mBaselineGpuFreeMemory{0};
    bool mIsIGPU{false};
};
