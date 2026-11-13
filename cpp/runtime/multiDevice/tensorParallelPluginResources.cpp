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

#include "runtime/multiDevice/tensorParallelPluginResources.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "runtime/multiDevice/backends/nccl/tensorParallelNcclResources.h"

#include <algorithm>
#include <exception>
#include <new>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

TensorParallelPluginResources::TensorParallelPluginResources(TensorParallelPluginResourcesConfig const& config)
    : mTpSize(config.tpSize)
    , mLocalRanks(config.localRanks)
    , mLocalDevices(config.localDevices)
{
    if (config.tpSize <= 1)
    {
        return;
    }

    if (mLocalRanks.empty())
    {
        mLocalRanks.reserve(config.tpSize);
        for (int32_t rank = 0; rank < config.tpSize; ++rank)
        {
            mLocalRanks.push_back(rank);
        }
    }
    if (mLocalDevices.empty())
    {
        mLocalDevices = mLocalRanks;
    }
    ELLM_CHECK(mLocalRanks.size() == mLocalDevices.size(),
        format::fmtstr(
            "TP plugin resources require matching local rank/device counts: localRanks=%zu, localDevices=%zu",
            mLocalRanks.size(), mLocalDevices.size()));
    for (int32_t const rank : mLocalRanks)
    {
        ELLM_CHECK(rank >= 0 && rank < config.tpSize,
            format::fmtstr("TP plugin resource local rank is out of range: rank=%d, tpSize=%d", rank, config.tpSize));
    }

    int32_t const deviceCount = detectCudaDeviceCount();
    for (int32_t const device : mLocalDevices)
    {
        ELLM_CHECK(device >= 0 && device < deviceCount,
            format::fmtstr("Requested CUDA device %d but only %d devices are available", device, deviceCount));
    }

    auto ncclResources = createTensorParallelNcclResources(
        config.tpSize, mLocalRanks, mLocalDevices, config.ncclComms, config.ownsNcclComms);
    mRuntimeCollectives = ncclResources->runtimeCollectives();
    ELLM_CHECK(mRuntimeCollectives != nullptr, "NCCL path did not provide runtime collective resources.");
    mAllReducePaths.push_back(std::move(ncclResources));

}

TensorParallelPluginResources::~TensorParallelPluginResources() noexcept = default;

RuntimeCollectiveResources const* TensorParallelPluginResources::runtimeCollectives() const noexcept
{
    return mRuntimeCollectives;
}

bool TensorParallelPluginResources::hasAllReducePath(AllReducePathType type) const noexcept
{
    return std::any_of(mAllReducePaths.begin(), mAllReducePaths.end(),
        [type](auto const& path) { return path != nullptr && path->type() == type && path->registered(); });
}

bool TensorParallelPluginResources::abortOwnedRuntimeCollectives() noexcept
{
    return mRuntimeCollectives != nullptr && mRuntimeCollectives->abortOwnedCommunicators();
}

} // namespace rt
} // namespace trt_edgellm
