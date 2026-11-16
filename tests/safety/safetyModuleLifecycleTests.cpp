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

#include "safety_inference/kernels/safetyAttentionModules.h"

#include <cstdio>
#include <cstring>

namespace
{
struct DriverState
{
    int maps{};
    int failMap{};
    int loads{};
    int lookups{};
    int attributes{};
    int launches{};
    int unloads{};
    int preparedModules{};
    int failLoad{};
    int failLookup{};
    int failAttribute{};
    int failUnload{};
    bool failLaunch{};
    bool failDeviceQuery{};
    bool badOrder{};
};

DriverState sDriver{};
trt_edgellm::safety::SafetyAttentionModules sModules;

bool expect(bool condition, char const* message)
{
    if (!condition)
    {
        std::fprintf(stderr, "%s\n", message);
    }
    return condition;
}

bool load()
{
    using namespace trt_edgellm::safety;
    return expect(sModules.load() == CUDA_SUCCESS, "Loading failed")
        && expect(sDriver.loads >= 2, "Expected multiple pinned variants")
        && expect(sDriver.lookups == 0 && sDriver.attributes == 0, "Loading performed preparation")
        && expect(!sModules.ready(), "Loading made modules ready prematurely");
}

bool prepare()
{
    using namespace trt_edgellm::safety;
    sDriver.preparedModules = sDriver.loads;
    return expect(sModules.prepare() == CUDA_SUCCESS, "Preparation failed")
        && expect(sModules.ready(), "Preparation did not publish readiness")
        && expect(sDriver.lookups == sDriver.loads && sDriver.attributes == sDriver.loads,
            "Preparation did not resolve and configure every variant")
        && expect(!sDriver.badOrder, "Driver calls violated lifecycle order");
}

CUresult launch(bool visual)
{
    using namespace trt_edgellm::safety;
    AttentionLaunch shape{};
    shape.headSize = 64;
    shape.qHeads = 2;
    shape.kvHeads = 1;
    shape.tokens = 1;
    shape.capacity = 1;
    shape.sequences = 1;
    shape.ropeTokens = 1;
    shape.scale = 0.125F;
    alignas(16) half data[16]{};
    int32_t metadata[4]{};
    auto stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(1));
    fmha::PreparedBindings bindings{};
    return sModules.launch(
        bindings, shape, visual, data, data, data, metadata, metadata, metadata, data, metadata, stream);
}

bool bindingLifecycle()
{
    using namespace trt_edgellm::safety;
    AttentionLaunch shape{5, 2, 1, 64, 32, 32, 64, 1, 0.125F};
    alignas(16) half data[16]{};
    fmha::PreparedBindings bindings{};
    auto prepareBindings = [&](fmha::PreparedBindings& state, AttentionLaunch const& dimensions, half* output) {
        return sModules.prepareBindings(state, dimensions, false, data, data, data, output);
    };
    // Given fixed bindings, when Initialization is sealed, then reuse performs no tensor-map encoding.
    if (!expect(prepareBindings(bindings, shape, data) == CUDA_SUCCESS, "Initial bindings failed")
        || !expect(sDriver.maps == 4, "Expected four tensor maps")
        || !expect(sModules.seal() == CUDA_SUCCESS, "Initialization did not seal")
        || !expect(prepareBindings(bindings, shape, data) == CUDA_SUCCESS, "Prepared bindings were rejected"))
    {
        return false;
    }
    int32_t metadata[2]{};
    auto stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(1));
    if (!expect(sModules.launch(
                    bindings, shape, false, data, data, data, metadata, metadata, nullptr, data, metadata, stream)
                == CUDA_SUCCESS,
            "Operational launch failed")
        || !expect(sDriver.launches == 1 && sDriver.maps == 4, "Operational launch performed preparation"))
    {
        return false;
    }
    // Given Operational state, when bindings or shape change, then rejection leaves the prepared state intact.
    fmha::PreparedBindings unprepared{};
    auto changed = shape;
    ++changed.tokens;
    if (!expect(prepareBindings(bindings, shape, data + 8) != CUDA_SUCCESS, "Changed address was admitted")
        || !expect(prepareBindings(bindings, changed, data) != CUDA_SUCCESS, "Changed shape was admitted")
        || !expect(prepareBindings(unprepared, shape, data) != CUDA_SUCCESS, "Unprepared Operational bindings admitted")
        || !expect(sDriver.maps == 4, "Operational attempted tensor-map encoding")
        || !expect(prepareBindings(bindings, shape, data) == CUDA_SUCCESS, "Rejection mutated prepared bindings")
        || !expect(sModules.seal() != CUDA_SUCCESS, "Repeated seal succeeded"))
    {
        return false;
    }
    return true;
}

bool terminal()
{
    using namespace trt_edgellm::safety;
    int const loads = sDriver.loads;
    int const lookups = sDriver.lookups;
    int const attributes = sDriver.attributes;
    int const unloads = sDriver.unloads;
    int const launches = sDriver.launches;
    return expect(!sModules.ready(), "Failure left modules ready")
        && expect(sModules.load() != CUDA_SUCCESS, "Terminal state allowed loading")
        && expect(sModules.prepare() != CUDA_SUCCESS, "Terminal state allowed preparation")
        && expect(launch(false) != CUDA_SUCCESS, "Terminal state allowed launch")
        && expect(sModules.release() != CUDA_SUCCESS, "Terminal state allowed automatic cleanup")
        && expect(sDriver.loads == loads && sDriver.lookups == lookups && sDriver.attributes == attributes
                && sDriver.unloads == unloads && sDriver.launches == launches,
            "Rejected lifecycle calls still reached the driver");
}

bool run(char const* mode)
{
    using namespace trt_edgellm::safety;
    if (!expect(!sModules.ready(), "Process did not start empty"))
    {
        return false;
    }
    if (std::strcmp(mode, "owner_isolation") == 0)
    {
        // Given independent context owners, when one fails, then the other can initialize, launch and release.
        SafetyAttentionModules failed;
        if (!expect(failed.prepare() != CUDA_SUCCESS, "Unloaded owner prepared") || !load() || !prepare()
            || !expect(launch(false) == CUDA_SUCCESS, "Failure escaped its owner")
            || !expect(!failed.ready(), "Other owner preparation revived a failed owner"))
        {
            return false;
        }
        int const lookups = sDriver.lookups;
        int const attributes = sDriver.attributes;
        // Given shared initialized modules, when another plugin prepares, then no driver setup is repeated.
        return expect(sModules.prepare() == CUDA_SUCCESS, "Shared preparation failed")
            && expect(
                sDriver.lookups == lookups && sDriver.attributes == attributes, "Preparation repeated driver work")
            && expect(sModules.release() == CUDA_SUCCESS, "Independent teardown failed")
            && expect(failed.load() != CUDA_SUCCESS, "Teardown reset another owner");
    }
    if (std::strcmp(mode, "wrong_order") == 0)
    {
        return expect(sModules.prepare() != CUDA_SUCCESS, "Preparation before load succeeded")
            && expect(sDriver.loads == 0 && sDriver.lookups == 0, "Wrong-order call reached driver") && terminal();
    }
    if (std::strcmp(mode, "load_failure") == 0)
    {
        sDriver.failLoad = 2;
        return expect(sModules.load() == CUDA_ERROR_INVALID_IMAGE, "Load failure was not propagated")
            && expect(sDriver.loads == 2 && sDriver.lookups == 0 && sDriver.unloads == 0,
                "Failed loading continued or performed rollback")
            && terminal();
    }
    if (!load())
    {
        return false;
    }
    if (std::strcmp(mode, "loaded_teardown") == 0)
    {
        // Given loaded modules unused by the builder, when deinitializing, then unload without function preparation.
        int const count = sDriver.loads;
        return expect(sModules.release() == CUDA_SUCCESS, "Unused loaded modules could not be released")
            && expect(sDriver.unloads == count && sDriver.lookups == 0 && sDriver.attributes == 0,
                "Unused teardown skipped modules or performed preparation")
            && terminal();
    }
    if (std::strcmp(mode, "repeated_load") == 0)
    {
        int const count = sDriver.loads;
        return expect(sModules.load() != CUDA_SUCCESS, "Repeated load succeeded")
            && expect(sDriver.loads == count, "Repeated load reached driver") && terminal();
    }
    if (std::strcmp(mode, "prepare_failure") == 0 || std::strcmp(mode, "attribute_failure") == 0)
    {
        bool const lookupFailure = std::strcmp(mode, "prepare_failure") == 0;
        sDriver.failLookup = lookupFailure ? 2 : 0;
        sDriver.failAttribute = lookupFailure ? 0 : 2;
        sDriver.preparedModules = sDriver.loads;
        return expect(sModules.prepare() == CUDA_ERROR_INVALID_VALUE, "Preparation failure was not propagated")
            && expect(sDriver.lookups == 2 && sDriver.attributes == (lookupFailure ? 1 : 2) && sDriver.unloads == 0
                    && !sDriver.badOrder,
                "Failed preparation continued or performed rollback")
            && terminal();
    }
    if (std::strcmp(mode, "device_query_failure") == 0)
    {
        // Given a failed Initialization query, when preparing modules, then the failure is terminal.
        sDriver.failDeviceQuery = true;
        sDriver.preparedModules = sDriver.loads;
        return expect(sModules.prepare() != CUDA_SUCCESS, "Device query failure was ignored") && terminal();
    }
    if (!prepare())
    {
        return false;
    }
    if (std::strcmp(mode, "bindings") == 0)
    {
        return bindingLifecycle() && terminal();
    }
    if (std::strcmp(mode, "map_failure") == 0)
    {
        // Given a failing encoder, when preparing bindings, then no later map or launch is attempted.
        sDriver.failMap = 2;
        return expect(launch(false) == CUDA_ERROR_INVALID_VALUE, "Map error was not propagated")
            && expect(sDriver.maps == 2 && sDriver.launches == 0 && sDriver.unloads == 0,
                "Map failure continued execution or cleanup")
            && terminal();
    }
    if (std::strcmp(mode, "launch_failure") == 0)
    {
        sDriver.failLaunch = true;
        return expect(launch(false) == CUDA_ERROR_LAUNCH_FAILED, "Launch failure was not propagated")
            && expect(sDriver.launches == 1 && sDriver.unloads == 0, "Launch failure performed cleanup") && terminal();
    }
    if (std::strcmp(mode, "teardown_failure") == 0)
    {
        sDriver.failUnload = 2;
        return expect(sModules.release() == CUDA_ERROR_INVALID_HANDLE, "Teardown failure was not propagated")
            && expect(sDriver.unloads == 2, "Teardown continued after failure") && terminal();
    }
    if (!expect(std::strcmp(mode, "success") == 0, "Unknown lifecycle test mode"))
    {
        return false;
    }
    int const count = sDriver.loads;
    if (!expect(launch(false) == CUDA_SUCCESS && launch(true) == CUDA_SUCCESS, "Prepared launch failed")
        || !expect(sDriver.launches == 2 && sDriver.loads == count && sDriver.lookups == count
                && sDriver.attributes == count && sDriver.unloads == 0 && !sDriver.badOrder,
            "Operational launch performed lifecycle work"))
    {
        return false;
    }
    return expect(sModules.release() == CUDA_SUCCESS, "Healthy teardown failed")
        && expect(sDriver.unloads == count, "Teardown did not unload every module") && terminal();
}
} // namespace

extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* device)
{
    *device = 0;
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetAttribute(int* value, cudaDeviceAttr, int)
{
    *value = 20;
    return sDriver.failDeviceQuery ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" CUresult CUDAAPI cuTensorMapEncodeTiled(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
    cuuint64_t const*, cuuint64_t const*, cuuint32_t const*, cuuint32_t const*, CUtensorMapInterleave,
    CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill)
{
    ++sDriver.maps;
    return sDriver.maps == sDriver.failMap ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuModuleLoadData(CUmodule* module, void const*)
{
    ++sDriver.loads;
    sDriver.badOrder = sDriver.badOrder || sDriver.lookups != 0 || sDriver.attributes != 0;
    if (sDriver.loads == sDriver.failLoad)
    {
        return CUDA_ERROR_INVALID_IMAGE;
    }
    *module = reinterpret_cast<CUmodule>(static_cast<uintptr_t>(sDriver.loads));
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuModuleGetFunction(CUfunction* function, CUmodule module, char const*)
{
    ++sDriver.lookups;
    sDriver.badOrder = sDriver.badOrder || sDriver.preparedModules != sDriver.loads;
    if (sDriver.lookups == sDriver.failLookup)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *function = reinterpret_cast<CUfunction>(module);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuFuncSetAttribute(CUfunction, CUfunction_attribute, int)
{
    ++sDriver.attributes;
    return sDriver.attributes == sDriver.failAttribute ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuModuleUnload(CUmodule)
{
    ++sDriver.unloads;
    return sDriver.unloads == sDriver.failUnload ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, CUstream, void**, void**)
{
    ++sDriver.launches;
    sDriver.badOrder = sDriver.badOrder || sDriver.loads != sDriver.preparedModules || sDriver.lookups != sDriver.loads
        || sDriver.attributes != sDriver.loads || sDriver.unloads != 0;
    return sDriver.failLaunch ? CUDA_ERROR_LAUNCH_FAILED : CUDA_SUCCESS;
}

int main(int argc, char** argv)
{
    return argc == 2 && run(argv[1]) ? 0 : 1;
}
