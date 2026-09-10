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

#include "memoryMonitor.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <optional>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace trt_edgellm;

class MemoryBackend
{
public:
    virtual ~MemoryBackend() = default;

    virtual std::optional<size_t> sample(std::string& error) = 0;
    virtual char const* metric() const = 0;
};

namespace
{
class NvmlBackend final : public MemoryBackend
{
public:
    using Return = int;
    using Device = void*;
    struct ProcessInfo
    {
        unsigned int pid;
        unsigned long long usedGpuMemory;
    };

    using Init = Return (*)();
    using Shutdown = Return (*)();
    using ErrorString = char const* (*) (Return);
    using GetHandleByPci = Return (*)(char const*, Device*);
    using GetProcesses = Return (*)(Device, unsigned int*, ProcessInfo*);

    static std::unique_ptr<NvmlBackend> create(std::string const& pciBusId)
    {
        auto backend = std::unique_ptr<NvmlBackend>(new NvmlBackend());
        if (!backend->initialize(pciBusId))
        {
            return nullptr;
        }
        std::string error;
        if (!backend->sample(error))
        {
            return nullptr;
        }
        return backend;
    }

    ~NvmlBackend() override
    {
        if (mInitialized)
        {
            mShutdown();
        }
        if (mLibrary)
        {
            dlclose(mLibrary);
        }
    }

    std::optional<size_t> sample(std::string& error) override
    {
        constexpr Return kInsufficientSize = 7;
        constexpr Return kNotSupported = 3;
        constexpr size_t kProcessCountSlack = 8;
        constexpr int kMaxQueryAttempts = 3;
        unsigned int count{};
        Return result = mGetProcesses(mDevice, &count, nullptr);
        if (result != 0 && result != kInsufficientSize)
        {
            error = result == kNotSupported ? "NVML process accounting is unsupported" : getError(result);
            return std::nullopt;
        }
        if (count == 0)
        {
            return 0;
        }

        mProcesses.resize(std::max(mProcesses.size(), static_cast<size_t>(count) + kProcessCountSlack));
        for (int attempt = 0; attempt < kMaxQueryAttempts; ++attempt)
        {
            count = static_cast<unsigned int>(mProcesses.size());
            result = mGetProcesses(mDevice, &count, mProcesses.data());
            if (result == kInsufficientSize)
            {
                mProcesses.resize(
                    std::max(mProcesses.size() + kProcessCountSlack, static_cast<size_t>(count) + kProcessCountSlack));
                continue;
            }
            if (result != 0)
            {
                error = getError(result);
                return std::nullopt;
            }
            for (unsigned int index = 0; index < count; ++index)
            {
                if (mProcesses[index].pid == static_cast<unsigned int>(getpid()))
                {
                    if (mProcesses[index].usedGpuMemory == ULLONG_MAX)
                    {
                        error = "NVML returned N/A for current-PID memory";
                        return std::nullopt;
                    }
                    return static_cast<size_t>(mProcesses[index].usedGpuMemory);
                }
            }
            return 0;
        }
        error = "NVML process list changed during repeated queries";
        return std::nullopt;
    }

    char const* metric() const override
    {
        return "nvml_process_allocation";
    }

private:
    bool initialize(std::string const& pciBusId)
    {
        mLibrary = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!mLibrary)
        {
            return false;
        }
        mInit = reinterpret_cast<Init>(dlsym(mLibrary, "nvmlInit_v2"));
        mShutdown = reinterpret_cast<Shutdown>(dlsym(mLibrary, "nvmlShutdown"));
        mErrorString = reinterpret_cast<ErrorString>(dlsym(mLibrary, "nvmlErrorString"));
        mGetHandleByPci = reinterpret_cast<GetHandleByPci>(dlsym(mLibrary, "nvmlDeviceGetHandleByPciBusId_v2"));
        mGetProcesses = reinterpret_cast<GetProcesses>(dlsym(mLibrary, "nvmlDeviceGetComputeRunningProcesses"));
        if (!mInit || !mShutdown || !mGetHandleByPci || !mGetProcesses || pciBusId.empty())
        {
            return false;
        }

        Return result = mInit();
        if (result != 0)
        {
            return false;
        }
        mInitialized = true;
        return mGetHandleByPci(pciBusId.c_str(), &mDevice) == 0;
    }

    std::string getError(Return result) const
    {
        return mErrorString ? mErrorString(result) : "NVML error " + std::to_string(result);
    }

    void* mLibrary{};
    bool mInitialized{};
    Init mInit{};
    Shutdown mShutdown{};
    ErrorString mErrorString{};
    GetHandleByPci mGetHandleByPci{};
    GetProcesses mGetProcesses{};
    Device mDevice{};
    std::vector<ProcessInfo> mProcesses;
};

class CudaMemoryBackend final : public MemoryBackend
{
public:
    static std::unique_ptr<CudaMemoryBackend> create()
    {
        size_t freeMemory{};
        size_t totalMemory{};
        CUDA_CHECK(cudaMemGetInfo(&freeMemory, &totalMemory));
        return std::unique_ptr<CudaMemoryBackend>(new CudaMemoryBackend(freeMemory));
    }

    std::optional<size_t> sample(std::string& error) override
    {
        size_t freeMemory{};
        size_t totalMemory{};
        cudaError_t const result = cudaMemGetInfo(&freeMemory, &totalMemory);
        if (result != cudaSuccess)
        {
            error = std::string(cudaGetErrorName(result)) + ": " + cudaGetErrorString(result);
            return std::nullopt;
        }
        return freeMemory < mBaselineFreeMemory ? mBaselineFreeMemory - freeMemory : 0;
    }

    char const* metric() const override
    {
        return "cuda_free_memory_delta";
    }

private:
    explicit CudaMemoryBackend(size_t baselineFreeMemory)
        : mBaselineFreeMemory(baselineFreeMemory)
    {
    }

    size_t mBaselineFreeMemory{};
};

std::unique_ptr<MemoryBackend> selectMemoryBackend()
{
    CUDA_CHECK(cudaFree(nullptr));
    int device{-1};
    CUDA_CHECK(cudaGetDevice(&device));
    char pciBusId[64]{};
    cudaError_t const pciResult = cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device);
    auto nvml = NvmlBackend::create(pciResult == cudaSuccess ? pciBusId : "");
    if (nvml)
    {
        return nvml;
    }
    return CudaMemoryBackend::create();
}
} // namespace

MemoryMonitor::MemoryMonitor() = default;

MemoryMonitor::~MemoryMonitor()
{
    stop();
}

void MemoryMonitor::start()
{
    stop();
    mPeakGpuMemory = 0;
    mSampleFailed = false;
    mBackend = selectMemoryBackend();
    std::string error;
    if (auto initialValue = mBackend->sample(error))
    {
        mPeakGpuMemory = *initialValue;
    }
    else
    {
        mSampleFailed = true;
        LOG_WARNING("Initial memory monitor sample failed for %s: %s", mBackend->metric(), error.c_str());
    }
    LOG_INFO("Memory Monitor Started - metric: %s", mBackend->metric());
    mActive = true;
    mTask = std::async(std::launch::async, [this]() { monitor(); });
}

void MemoryMonitor::stop()
{
    if (mTask.valid())
    {
        mActive = false;
        mTask.get();
    }
}

size_t MemoryMonitor::getPeakGpuMemory() const
{
    return mSampleFailed ? 0 : mPeakGpuMemory.load();
}

size_t MemoryMonitor::getPeakCpuMemory() const
{
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        return static_cast<size_t>(usage.ru_maxrss) * 1024;
    }
    return 0;
}

char const* MemoryMonitor::getGpuMemoryMetric() const
{
    return mBackend && !mSampleFailed ? mBackend->metric() : "unavailable";
}

void MemoryMonitor::monitor()
{
    bool sampleFailureLogged{};
    while (mActive.load())
    {
        std::string error;
        auto value = mBackend->sample(error);
        if (value)
        {
            size_t peak = mPeakGpuMemory.load();
            while (peak < *value && !mPeakGpuMemory.compare_exchange_weak(peak, *value))
            {
            }
            mSampleFailed = false;
            sampleFailureLogged = false;
        }
        else
        {
            mSampleFailed = true;
            if (!sampleFailureLogged)
            {
                LOG_WARNING("Memory monitor sample failed for %s: %s", mBackend->metric(), error.c_str());
                sampleFailureLogged = true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}
