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

#include "runtime/multiDevice/ncclCollectiveBackend.h"

#include "common/logger.h"
#include "common/stringUtils.h"
#include "runtime/multiDevice/collectiveGroup.h"

#include <cstdlib>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif
#include <mutex>
#include <stdexcept>

// NCCL API types are loaded dynamically to avoid a hard link dependency.
typedef struct ncclComm* ncclComm_t;
typedef int ncclResult_t;

namespace trt_edgellm
{
namespace rt
{

namespace
{

inline constexpr int32_t kSumOp = 0;

using NcclGetUniqueIdFn = ncclResult_t (*)(NcclUniqueId*);
using NcclCommInitRankFn = ncclResult_t (*)(ncclComm_t*, int, NcclUniqueId, int);
using NcclCommInitAllFn = ncclResult_t (*)(ncclComm_t*, int, int const*);
using NcclCommDestroyFn = ncclResult_t (*)(ncclComm_t);
using NcclCommAbortFn = ncclResult_t (*)(ncclComm_t);
using NcclAllReduceFn = ncclResult_t (*)(void const*, void*, size_t, int, int, ncclComm_t, cudaStream_t);
using NcclAllGatherFn = ncclResult_t (*)(void const*, void*, size_t, int, ncclComm_t, cudaStream_t);
using NcclBroadcastFn = ncclResult_t (*)(void const*, void*, size_t, int, int, ncclComm_t, cudaStream_t);
using NcclGetErrorStringFn = char const* (*) (ncclResult_t);

void* gNcclLibHandle = nullptr;
NcclGetUniqueIdFn gNcclGetUniqueId = nullptr;
NcclCommInitRankFn gNcclCommInitRank = nullptr;
NcclCommInitAllFn gNcclCommInitAll = nullptr;
NcclCommDestroyFn gNcclCommDestroy = nullptr;
NcclCommAbortFn gNcclCommAbort = nullptr;
NcclAllReduceFn gNcclAllReduce = nullptr;
NcclAllGatherFn gNcclAllGather = nullptr;
NcclBroadcastFn gNcclBroadcast = nullptr;
NcclGetErrorStringFn gNcclGetErrorString = nullptr;
std::mutex gNcclLoadMutex;

#if !defined(_WIN32)
// Caller must hold gNcclLoadMutex.
void resetNcclSymbols() noexcept
{
    gNcclGetUniqueId = nullptr;
    gNcclCommInitRank = nullptr;
    gNcclCommInitAll = nullptr;
    gNcclCommDestroy = nullptr;
    gNcclCommAbort = nullptr;
    gNcclAllReduce = nullptr;
    gNcclAllGather = nullptr;
    gNcclBroadcast = nullptr;
    gNcclGetErrorString = nullptr;
}

void* loadSymbol(char const* name)
{
    void* symbol = dlsym(gNcclLibHandle, name);
    if (symbol == nullptr)
    {
        LOG_ERROR("Failed to load NCCL symbol '%s': %s", name, dlerror());
    }
    return symbol;
}
#endif

bool loadNcclLibrary() noexcept
{
#if defined(_WIN32)
    // NVIDIA NCCL does not provide a Windows backend. Multi-device execution is
    // unavailable, but the single-device runtime still links this interface.
    return false;
#else
    std::lock_guard<std::mutex> lock(gNcclLoadMutex);

    if (gNcclLibHandle != nullptr)
    {
        return true;
    }

    char const* const ncclPaths[] = {"libnccl.so.2", "libnccl.so", nullptr};

    char const* envPath = std::getenv("EDGELLM_NCCL_SO_PATH");
    if (envPath != nullptr && envPath[0] != '\0')
    {
        gNcclLibHandle = dlopen(envPath, RTLD_NOW | RTLD_LOCAL);
        if (gNcclLibHandle != nullptr)
        {
            LOG_INFO("Loaded NCCL library from EDGELLM_NCCL_SO_PATH: %s", envPath);
        }
        else
        {
            LOG_WARNING("Failed to load NCCL from EDGELLM_NCCL_SO_PATH=%s: %s", envPath, dlerror());
        }
    }

    if (gNcclLibHandle == nullptr)
    {
        for (int32_t index = 0; ncclPaths[index] != nullptr; ++index)
        {
            gNcclLibHandle = dlopen(ncclPaths[index], RTLD_NOW | RTLD_LOCAL);
            if (gNcclLibHandle != nullptr)
            {
                LOG_INFO("Loaded NCCL library: %s", ncclPaths[index]);
                break;
            }
        }
    }

    if (gNcclLibHandle == nullptr)
    {
        LOG_ERROR("Failed to load NCCL library. Error: %s", dlerror());
        LOG_ERROR("Set EDGELLM_NCCL_SO_PATH to the path of libnccl.so.2 or build NCCL from source.");
        return false;
    }

    gNcclGetUniqueId = reinterpret_cast<NcclGetUniqueIdFn>(loadSymbol("ncclGetUniqueId"));
    gNcclCommInitRank = reinterpret_cast<NcclCommInitRankFn>(loadSymbol("ncclCommInitRank"));
    gNcclCommInitAll = reinterpret_cast<NcclCommInitAllFn>(loadSymbol("ncclCommInitAll"));
    gNcclCommDestroy = reinterpret_cast<NcclCommDestroyFn>(loadSymbol("ncclCommDestroy"));
    gNcclCommAbort = reinterpret_cast<NcclCommAbortFn>(loadSymbol("ncclCommAbort"));
    gNcclAllReduce = reinterpret_cast<NcclAllReduceFn>(loadSymbol("ncclAllReduce"));
    gNcclAllGather = reinterpret_cast<NcclAllGatherFn>(loadSymbol("ncclAllGather"));
    gNcclBroadcast = reinterpret_cast<NcclBroadcastFn>(loadSymbol("ncclBroadcast"));
    gNcclGetErrorString = reinterpret_cast<NcclGetErrorStringFn>(loadSymbol("ncclGetErrorString"));

    if (gNcclGetUniqueId == nullptr || gNcclCommInitRank == nullptr || gNcclCommDestroy == nullptr
        || gNcclCommAbort == nullptr || gNcclAllReduce == nullptr || gNcclAllGather == nullptr
        || gNcclBroadcast == nullptr || gNcclCommInitAll == nullptr)
    {
        LOG_ERROR("Failed to load one or more required NCCL symbols");
        dlclose(gNcclLibHandle);
        gNcclLibHandle = nullptr;
        resetNcclSymbols();
        return false;
    }

    return true;
#endif
}

void checkNcclResult(int32_t result, char const* operation)
{
    if (result != NcclCollectiveBackend::kSuccess)
    {
        if (result == NcclCollectiveBackend::kUnavailable)
        {
            throw std::runtime_error(format::fmtstr(
                "NCCL %s is unavailable because the communicator or function pointer is null.", operation));
        }
        char const* errorString = gNcclGetErrorString ? gNcclGetErrorString(result) : "unknown NCCL error";
        throw std::runtime_error(
            format::fmtstr("NCCL %s failed with error code %d: %s", operation, result, errorString));
    }
}

} // namespace

namespace NcclCollectiveBackend
{

void load()
{
    if (!loadNcclLibrary())
    {
        throw std::runtime_error("Failed to load NCCL library for parallel communication");
    }
}

bool isLoaded() noexcept
{
    std::lock_guard<std::mutex> lock(gNcclLoadMutex);
    return gNcclLibHandle != nullptr;
}

void getUniqueId(NcclUniqueId& uniqueId)
{
    load();
    checkNcclResult(gNcclGetUniqueId(&uniqueId), "ncclGetUniqueId");
}

void initRank(void** comm, int32_t size, NcclUniqueId const& uniqueId, int32_t rank)
{
    load();
    ncclComm_t ncclComm = nullptr;
    checkNcclResult(gNcclCommInitRank(&ncclComm, size, uniqueId, rank), "ncclCommInitRank");
    *comm = ncclComm;
}

std::vector<void*> initAll(std::vector<int32_t> const& devices)
{
    load();

    int32_t const size = static_cast<int32_t>(devices.size());
    std::vector<int> deviceList(devices.begin(), devices.end());
    std::vector<ncclComm_t> comms(size);
    checkNcclResult(gNcclCommInitAll(comms.data(), size, deviceList.data()), "ncclCommInitAll");

    std::vector<void*> commHandles(size);
    for (int32_t rank = 0; rank < size; ++rank)
    {
        commHandles[rank] = comms[rank];
    }
    return commHandles;
}

void destroyComm(void* comm) noexcept
{
    if (comm != nullptr && gNcclCommDestroy != nullptr)
    {
        gNcclCommDestroy(static_cast<ncclComm_t>(comm));
    }
}

void abortComm(void* comm) noexcept
{
    if (comm == nullptr || gNcclCommAbort == nullptr)
    {
        return;
    }

    int32_t const result = gNcclCommAbort(static_cast<ncclComm_t>(comm));
    if (result != kSuccess)
    {
        char const* errorString = gNcclGetErrorString ? gNcclGetErrorString(result) : "unknown NCCL error";
        LOG_WARNING("NCCL communicator abort failed with error code %d: %s", result, errorString);
    }
}

void abortComms(std::vector<void*> const& comms) noexcept
{
    for (void* comm : comms)
    {
        abortComm(comm);
    }
}

void destroyComms(std::vector<void*> const& comms) noexcept
{
    for (void* comm : comms)
    {
        destroyComm(comm);
    }
}

void allReduce(void const* sendBuffer, void* recvBuffer, size_t count, CollectiveDataType dataType, void* comm,
    cudaStream_t stream)
{
    load();
    checkNcclResult(allReduceRaw(sendBuffer, recvBuffer, count, dataType, comm, stream), "ncclAllReduce");
}

void allGather(void const* sendBuffer, void* recvBuffer, size_t sendCount, CollectiveDataType dataType, void* comm,
    cudaStream_t stream)
{
    load();
    checkNcclResult(allGatherRaw(sendBuffer, recvBuffer, sendCount, dataType, comm, stream), "ncclAllGather");
}

void broadcast(void const* sendBuffer, void* recvBuffer, size_t count, CollectiveDataType dataType, int32_t rootRank,
    void* comm, cudaStream_t stream)
{
    load();
    checkNcclResult(broadcastRaw(sendBuffer, recvBuffer, count, dataType, rootRank, comm, stream), "ncclBroadcast");
}

int32_t allReduceRaw(void const* sendBuffer, void* recvBuffer, size_t count, CollectiveDataType dataType, void* comm,
    cudaStream_t stream) noexcept
{
    if (gNcclAllReduce == nullptr || comm == nullptr)
    {
        return kUnavailable;
    }
    return gNcclAllReduce(
        sendBuffer, recvBuffer, count, static_cast<int32_t>(dataType), kSumOp, static_cast<ncclComm_t>(comm), stream);
}

int32_t allGatherRaw(void const* sendBuffer, void* recvBuffer, size_t sendCount, CollectiveDataType dataType,
    void* comm, cudaStream_t stream) noexcept
{
    if (gNcclAllGather == nullptr || comm == nullptr)
    {
        return kUnavailable;
    }
    return gNcclAllGather(
        sendBuffer, recvBuffer, sendCount, static_cast<int32_t>(dataType), static_cast<ncclComm_t>(comm), stream);
}

int32_t broadcastRaw(void const* sendBuffer, void* recvBuffer, size_t count, CollectiveDataType dataType,
    int32_t rootRank, void* comm, cudaStream_t stream) noexcept
{
    if (gNcclBroadcast == nullptr || comm == nullptr)
    {
        return kUnavailable;
    }
    return gNcclBroadcast(
        sendBuffer, recvBuffer, count, static_cast<int32_t>(dataType), rootRank, static_cast<ncclComm_t>(comm), stream);
}

void* allReduceFunction() noexcept
{
    if (!loadNcclLibrary())
    {
        return nullptr;
    }
    return reinterpret_cast<void*>(gNcclAllReduce);
}

} // namespace NcclCollectiveBackend

} // namespace rt
} // namespace trt_edgellm
