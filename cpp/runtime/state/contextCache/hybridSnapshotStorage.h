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

#include "common/tensor.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/contextCache/blockHash.h"
#include "runtime/state/contextCache/contextCacheTypes.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Preallocated device storage for exact hybrid checkpoints.
//!
//! A recurrent slot is the common leading index across every recurrent and convolution layer. A partial-KV slot is
//! likewise the leading index across every attention layer's K/V page snapshot. ResourcePools owns the slot IDs and
//! refcounts; this class owns only the corresponding immutable device slabs and performs stream-ordered D2D copies.
class HybridSnapshotStorage
{
public:
    //! boundaryHiddenDim > 0 (Hybrid+MTP only) additionally allocates one base-hidden vector per recurrent slot. The
    //! successor-dependent boundary MTP draft slot is recomputed at restore from this saved hidden state instead of
    //! being matched by the lookup key, which lets a checkpoint be reused regardless of the token that follows it.
    HybridSnapshotStorage(HybridCacheManager& cacheManager, int32_t recurrentSlotCount, int32_t partialKvSlotCount,
        HybridCacheManager* draftCacheManager = nullptr, int32_t boundaryHiddenDim = 0,
        nvinfer1::DataType boundaryHiddenType = nvinfer1::DataType::kHALF);
    HybridSnapshotStorage(HybridSnapshotStorage const&) = delete;
    HybridSnapshotStorage& operator=(HybridSnapshotStorage const&) = delete;

    static size_t recurrentBytesPerSlot(HybridCacheManager::Config const& config);
    static size_t partialKvBytesPerSlot(
        HybridCacheManager::Config const& config, HybridCacheManager::Config const* draftConfig = nullptr);
    static RecurrentStateSchemaId schemaId(
        HybridCacheManager::Config const& config, HybridCacheManager::Config const* draftConfig = nullptr);

    void captureRecurrent(int32_t snapshotSlot, int32_t batchSlot, cudaStream_t stream);
    void restoreRecurrent(int32_t snapshotSlot, int32_t batchSlot, cudaStream_t stream);
    void capturePartialKv(int32_t snapshotSlot, PageId sourcePage, int32_t validTokenCount, cudaStream_t stream);
    void restorePartialKv(int32_t snapshotSlot, PageId destinationPage, int32_t validTokenCount, cudaStream_t stream);
    void capturePartialKv(int32_t snapshotSlot, PageId sourceBasePage, PageId sourceDraftPage, int32_t validTokenCount,
        cudaStream_t stream);
    void restorePartialKv(int32_t snapshotSlot, PageId destinationBasePage, PageId destinationDraftPage,
        int32_t validTokenCount, cudaStream_t stream);

    //! Save one base-hidden row (the checkpoint's successor-dependent boundary hidden state) from a live [batch, seq,
    //! hidden] tensor into the boundary-hidden slab at snapshotSlot. Requires boundaryHiddenDim > 0.
    void captureBoundaryHidden(int32_t snapshotSlot, Tensor const& sourceHiddenStates, int32_t batchSlot,
        int32_t position, cudaStream_t stream);
    //! Restore the saved boundary hidden row into destinationHiddenStates[batchSlot, position, :].
    void restoreBoundaryHidden(int32_t snapshotSlot, Tensor& destinationHiddenStates, int32_t batchSlot,
        int32_t position, cudaStream_t stream);

    int32_t recurrentSlotCount() const noexcept;
    int32_t partialKvSlotCount() const noexcept;
    int32_t boundaryHiddenDim() const noexcept;

private:
    void validateRecurrentSlots(int32_t snapshotSlot, int32_t batchSlot) const;
    void validatePartialSlots(
        HybridCacheManager& cacheManager, int32_t snapshotSlot, PageId page, int32_t validTokenCount) const;
    void capturePartialKv(HybridCacheManager& cacheManager, std::vector<Tensor>& snapshots, int32_t snapshotSlot,
        PageId sourcePage, int32_t validTokenCount, cudaStream_t stream);
    void restorePartialKv(HybridCacheManager& cacheManager, std::vector<Tensor>& snapshots, int32_t snapshotSlot,
        PageId destinationPage, int32_t validTokenCount, cudaStream_t stream);

    HybridCacheManager& mCacheManager;
    HybridCacheManager* mDraftCacheManager{};
    int32_t mRecurrentSlotCount{};
    int32_t mPartialKvSlotCount{};
    int32_t mBoundaryHiddenDim{};
    std::vector<Tensor> mRecurrentSnapshots;
    std::vector<Tensor> mConvSnapshots;
    std::vector<Tensor> mPartialKvSnapshots;
    std::vector<Tensor> mDraftPartialKvSnapshots;
    //! One [recurrentSlotCount, boundaryHiddenDim] slab holding each checkpoint's saved boundary hidden state.
    Tensor mBoundaryHiddenSnapshot;
};

} // namespace rt
} // namespace trt_edgellm
