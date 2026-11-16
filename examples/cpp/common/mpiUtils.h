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

#include "memoryMonitor.h"
#include "profileFormatter.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <tensorrt_edgellm/common/checkMacros.h>
#include <tensorrt_edgellm/common/logger.h>
#include <tensorrt_edgellm/common/stringUtils.h>
#include <tensorrt_edgellm/runtime/multiDevice/ncclCollectiveBackend.h>
#include <tensorrt_edgellm/runtime/multiDevice/parallelConfig.h>
#include <utility>
#include <vector>

#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
#include <mpi.h>
#endif

namespace trt_edgellm
{
namespace example_utils
{

struct MpiProcessInfo
{
    int32_t worldSize{1};
    int32_t processRank{0};
    int32_t processDevice{0};
};

#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
[[noreturn]] inline void abortMpiWorld(int32_t processRank, int32_t worldSize, char const* phase, char const* detail)
{
    LOG_ERROR("[Rank %d/%d] Fatal %s failure: %s. Aborting the MPI world.", processRank, worldSize, phase, detail);
    int32_t const abortResult = MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    if (abortResult != MPI_SUCCESS)
    {
        LOG_ERROR("[Rank %d/%d] MPI_Abort failed with error %d.", processRank, worldSize, abortResult);
    }
    std::abort();
}
#endif

inline bool initializeMpiProcess(int& argc, char**& argv, MpiProcessInfo& info)
{
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    {
        LOG_ERROR("Failed to initialize MPI.");
        return false;
    }
    MPI_Comm_size(MPI_COMM_WORLD, &info.worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &info.processRank);

    MPI_Comm localComm = MPI_COMM_NULL;
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, info.processRank, MPI_INFO_NULL, &localComm)
        != MPI_SUCCESS)
    {
        abortMpiWorld(info.processRank, info.worldSize, "MPI device discovery", "MPI_Comm_split_type failed");
    }
    if (MPI_Comm_rank(localComm, &info.processDevice) != MPI_SUCCESS || MPI_Comm_free(&localComm) != MPI_SUCCESS)
    {
        abortMpiWorld(info.processRank, info.worldSize, "MPI device discovery",
            "failed to resolve or release the node-local rank");
    }

    try
    {
        int32_t deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (info.processDevice >= deviceCount)
        {
            throw std::runtime_error(trt_edgellm::format::fmtstr(
                "Node-local rank %d requires CUDA device %d, but this host exposes only %d device(s).",
                info.processDevice, info.processDevice, deviceCount));
        }
        CUDA_CHECK(cudaSetDevice(info.processDevice));
    }
    catch (std::exception const& e)
    {
        abortMpiWorld(info.processRank, info.worldSize, "CUDA device selection", e.what());
    }
#else
    (void) argc;
    (void) argv;
    (void) info;
#endif
    return true;
}

inline bool allProcessesSucceeded(bool localOk)
{
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
    int32_t const localStatus = localOk ? 1 : 0;
    int32_t globalStatus = 0;
    MPI_Allreduce(&localStatus, &globalStatus, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return globalStatus != 0;
#else
    return localOk;
#endif
}

inline std::vector<RankMemorySummary> collectRankMemorySummaries(MemoryMonitor const& memoryMonitor,
    trt_edgellm::rt::ParallelLaunchMode launchMode, int32_t tpSize, int32_t processRank)
{
#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
    if (launchMode != trt_edgellm::rt::ParallelLaunchMode::kMpi || tpSize <= 1)
    {
        return {};
    }

    struct RankMemorySample
    {
        int32_t rank{0};
        int32_t device{0};
        size_t peakGpuMemoryBytes{0};
        size_t peakCpuMemoryBytes{0};
    };

    int32_t device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    RankMemorySample localSample{
        processRank, device, memoryMonitor.getPeakGpuMemory(), memoryMonitor.getPeakCpuMemory()};

    std::vector<RankMemorySample> rankSamples;
    if (processRank == 0)
    {
        rankSamples.resize(static_cast<size_t>(tpSize));
    }
    void* receiveBuffer = processRank == 0 ? static_cast<void*>(rankSamples.data()) : nullptr;
    MPI_Gather(&localSample, static_cast<int32_t>(sizeof(RankMemorySample)), MPI_BYTE, receiveBuffer,
        static_cast<int32_t>(sizeof(RankMemorySample)), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (processRank != 0)
    {
        return {};
    }

    std::vector<RankMemorySummary> rankMemory;
    rankMemory.reserve(rankSamples.size());
    for (RankMemorySample const& sample : rankSamples)
    {
        rankMemory.push_back(
            RankMemorySummary{sample.rank, sample.device, sample.peakGpuMemoryBytes, sample.peakCpuMemoryBytes});
    }
    return rankMemory;
#else
    (void) memoryMonitor;
    (void) launchMode;
    (void) tpSize;
    (void) processRank;
    return {};
#endif
}

#if defined(EDGELLM_ENABLE_MULTI_DEVICE_MPI)
inline std::vector<trt_edgellm::rt::ParallelBackendHandles> createMpiBackendHandles(int32_t tpSize, int32_t tpRank)
{
    std::vector<trt_edgellm::rt::ParallelBackendHandles> backendGroups;
    if (tpSize <= 1)
    {
        return backendGroups;
    }

    try
    {
        trt_edgellm::rt::NcclUniqueId uniqueId{};
        if (tpRank == 0)
        {
            trt_edgellm::rt::NcclCollectiveBackend::getUniqueId(uniqueId);
        }
        if (MPI_Bcast(
                &uniqueId, static_cast<int32_t>(sizeof(trt_edgellm::rt::NcclUniqueId)), MPI_BYTE, 0, MPI_COMM_WORLD)
            != MPI_SUCCESS)
        {
            abortMpiWorld(tpRank, tpSize, "NCCL bootstrap", "MPI failed to broadcast the NCCL unique ID");
        }

        void* ncclComm = nullptr;
        trt_edgellm::rt::NcclCollectiveBackend::initRank(&ncclComm, tpSize, uniqueId, tpRank);

        std::vector<void*> tensorHandles(static_cast<size_t>(tpSize), nullptr);
        tensorHandles[static_cast<size_t>(tpRank)] = ncclComm;
        backendGroups.push_back(trt_edgellm::rt::ParallelBackendHandles{
            trt_edgellm::rt::ParallelType::kTensor, std::move(tensorHandles), true});
    }
    catch (std::exception const& e)
    {
        abortMpiWorld(tpRank, tpSize, "NCCL bootstrap", e.what());
    }
    catch (...)
    {
        abortMpiWorld(tpRank, tpSize, "NCCL bootstrap", "unknown exception");
    }
    return backendGroups;
}
#endif

} // namespace example_utils
} // namespace trt_edgellm
