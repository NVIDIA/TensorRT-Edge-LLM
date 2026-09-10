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

#include "runtime/multiDevice/multiDevicePluginResources.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

/*!
 * @brief Configuration for single-process TP plugin communication resources.
 */
struct TensorParallelPluginResourcesConfig
{
    int32_t tpSize{1};               //!< Tensor parallel world size.
    std::vector<int32_t> localRanks; //!< TP ranks owned by this process.
    std::vector<int32_t> localDevices;
    std::vector<void*> ncclComms;
    bool ownsNcclComms{true}; //!< Transfer ownership of non-empty externally supplied NCCL communicators.
};

/*!
 * @brief Own and register communication resources for single-process TP launch.
 *
 * Threaded TP launch initializes all NCCL communicators in one process through
 * ncclCommInitAll(), then registers the per-device communicator handles with
 * the TensorRT plugin registry before rank-local engines are deserialized.
 */
class TensorParallelPluginResources : public MultiDevicePluginResources
{
public:
    explicit TensorParallelPluginResources(TensorParallelPluginResourcesConfig const& config);
    ~TensorParallelPluginResources() noexcept override;

    TensorParallelPluginResources(TensorParallelPluginResources const&) = delete;
    TensorParallelPluginResources& operator=(TensorParallelPluginResources const&) = delete;
    TensorParallelPluginResources(TensorParallelPluginResources&&) = delete;
    TensorParallelPluginResources& operator=(TensorParallelPluginResources&&) = delete;

    ParallelType type() const noexcept override
    {
        return ParallelType::kTensor;
    }

    int32_t size() const noexcept override
    {
        return mTpSize;
    }

    RuntimeCollectiveResources const* runtimeCollectives() const noexcept override;
    bool hasAllReducePath(AllReducePathType type) const noexcept override;
    bool abortOwnedRuntimeCollectives() noexcept override;

private:
    int32_t mTpSize{1};
    std::vector<int32_t> mLocalRanks{};
    std::vector<int32_t> mLocalDevices{};
    //! Path owners are immutable after construction. The runtime-collective observer aliases the NCCL path owner and
    //! remains valid only for the lifetime of this TensorParallelPluginResources object. Externally supplied NCCL
    //! communicators remain owned by the caller and must outlive this object.
    std::vector<std::unique_ptr<PluginAllReducePathResources>> mAllReducePaths{};
    RuntimeCollectiveResources* mRuntimeCollectives{nullptr};
};

} // namespace rt
} // namespace trt_edgellm
