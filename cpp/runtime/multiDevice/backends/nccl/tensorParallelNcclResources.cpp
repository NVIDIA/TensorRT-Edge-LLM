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

#include "runtime/multiDevice/backends/nccl/tensorParallelNcclResources.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "runtime/multiDevice/multiDevicePluginCommRegistry.h"
#include "runtime/multiDevice/ncclCollectiveBackend.h"
#include "runtime/multiDevice/parallelConfig.h"

#include <atomic>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

namespace
{

class TensorParallelNcclResources final : public PluginAllReducePathResources, public RuntimeCollectiveResources
{
public:
    TensorParallelNcclResources(int32_t tpSize, std::vector<int32_t> localRanks, std::vector<int32_t> localDevices,
        std::vector<void*> ncclComms, bool ownsNcclComms)
        : mTpSize(tpSize)
        , mLocalRanks(std::move(localRanks))
        , mLocalDevices(std::move(localDevices))
        , mNcclComms(std::move(ncclComms))
        , mOwnsNcclComms(ownsNcclComms)
    {
        try
        {
            initializeCommunicators();
            ELLM_CHECK(registerWithPlugin(), "Failed to register NCCL resources with the AllReduce plugin.");
        }
        catch (...)
        {
            unregisterFromPlugin();
            static_cast<void>(abortOwnedCommunicators());
            destroyOwnedCommunicators();
            throw;
        }
    }

    ~TensorParallelNcclResources() noexcept override
    {
        unregisterFromPlugin();
        destroyOwnedCommunicators();
    }

    AllReducePathType type() const noexcept override
    {
        return AllReducePathType::kNccl;
    }

    bool registered() const noexcept override
    {
        return !mLocalRanks.empty() && mRegisteredCount == mLocalRanks.size();
    }

    RuntimeCollectiveResources* runtimeCollectives() noexcept override
    {
        return this;
    }

    void* communicatorForRank(int32_t rank) const noexcept override
    {
        if (rank < 0 || rank >= static_cast<int32_t>(mNcclComms.size()))
        {
            return nullptr;
        }
        return mNcclComms[static_cast<size_t>(rank)];
    }

    bool abortOwnedCommunicators() noexcept override
    {
        if (!mOwnsNcclComms)
        {
            LOG_WARNING("Skipping NCCL abort because the TP communicators are externally owned.");
            return false;
        }

        bool expected = false;
        if (mCommunicatorsAborted.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        {
            LOG_ERROR("Aborting runtime-owned NCCL communicators after a fatal parallel-rank failure.");
            NcclCollectiveBackend::abortComms(mNcclComms);
        }
        return true;
    }

private:
    bool registerWithPlugin() noexcept
    {
        if (registered())
        {
            return true;
        }

        while (mRegisteredCount < mLocalRanks.size())
        {
            size_t const index = mRegisteredCount;
            int32_t const rank = mLocalRanks[index];
            int32_t const device = mLocalDevices[index];
            void* ncclComm = communicatorForRank(rank);
            if (ncclComm == nullptr || !registerNcclCommForAllReducePlugin(device, ncclComm))
            {
                unregisterFromPlugin();
                return false;
            }
            ++mRegisteredCount;
            LOG_INFO("[TP rank %d/%d] Registered NCCL communicator for device %d.", rank, mTpSize, device);
        }
        return registered();
    }

    bool unregisterFromPlugin() noexcept
    {
        bool success = true;
        while (mRegisteredCount > 0)
        {
            size_t const index = mRegisteredCount - 1;
            int32_t const rank = mLocalRanks[index];
            int32_t const device = mLocalDevices[index];
            void* ncclComm = communicatorForRank(rank);
            if (!unregisterNcclCommForAllReducePlugin(device, ncclComm))
            {
                success = false;
            }
            --mRegisteredCount;
        }
        return success;
    }

    void initializeCommunicators()
    {
        if (mNcclComms.empty())
        {
            ELLM_CHECK(isFullLocalParallelGroup(mTpSize, mLocalRanks),
                format::fmtstr("ncclCommInitAll requires all TP ranks to be local and ordered: localRanks=%zu, "
                               "tpSize=%d",
                    mLocalRanks.size(), mTpSize));

            LOG_INFO("Initializing %d TP ranks in single process with ncclCommInitAll", mTpSize);
            mNcclComms = NcclCollectiveBackend::initAll(mLocalDevices);
            // The caller's ownership flag applies only to supplied communicators.
            // Communicators created here are always owned by this resource object.
            mOwnsNcclComms = true;
            LOG_INFO("ncclCommInitAll succeeded for %d devices", mTpSize);
        }
        else
        {
            ELLM_CHECK(static_cast<int32_t>(mNcclComms.size()) == mTpSize,
                format::fmtstr("External NCCL communicator list must be indexed by TP rank: ncclComms=%zu, tpSize=%d",
                    mNcclComms.size(), mTpSize));
        }

        for (size_t index = 0; index < mLocalRanks.size(); ++index)
        {
            int32_t const rank = mLocalRanks[index];
            void* ncclComm = communicatorForRank(rank);
            ELLM_CHECK(ncclComm != nullptr,
                format::fmtstr("NCCL communicator for TP rank %d is not initialized: tpSize=%d", rank, mTpSize));
        }
    }

    void destroyOwnedCommunicators() noexcept
    {
        if (!mOwnsNcclComms)
        {
            return;
        }
        if (!mCommunicatorsAborted.load(std::memory_order_acquire))
        {
            NcclCollectiveBackend::destroyComms(mNcclComms);
        }
        mNcclComms.clear();
    }

    int32_t mTpSize{1};
    std::vector<int32_t> mLocalRanks{};
    std::vector<int32_t> mLocalDevices{};
    std::vector<void*> mNcclComms{};
    bool mOwnsNcclComms{true};
    std::atomic<bool> mCommunicatorsAborted{false};
    size_t mRegisteredCount{0};
};

} // namespace

std::unique_ptr<PluginAllReducePathResources> createTensorParallelNcclResources(int32_t tpSize,
    std::vector<int32_t> localRanks, std::vector<int32_t> localDevices, std::vector<void*> ncclComms,
    bool ownsNcclComms)
{
    return std::make_unique<TensorParallelNcclResources>(
        tpSize, std::move(localRanks), std::move(localDevices), std::move(ncclComms), ownsNcclComms);
}

} // namespace rt
} // namespace trt_edgellm
