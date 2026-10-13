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

#include "common/tensor.h"

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! One non-owning host-row update consumed synchronously by KVPageTable::setRows().
//! `kPageIds` may be null only when `count` is zero.
struct KVPageTableRowUpdate
{
    int32_t slot{};
    int32_t const* kPageIds{};
    int32_t count{};
};

//! One logical KV page table per model (shared by the K and V halves; V page id
//! == K page id + numPages) plus its
//! derived K/V kernel view.
//!
//! Host state is the source of truth: `mHost` stores, per slot, the K page
//! ids (`-1` for unused); the V id of a live entry is always `k + numPages`
//! and `-1` stays `-1`. `kernelView()` is the single device tensor consumed
//! by the paged kernels, laid out as `[maxBatch, 2, maxPagesPerSeq]` with the
//! K half first and the derived V half second.
class KVPageTable
{
public:
    //! Row validation policy.
    enum class Mode : uint8_t
    {
        //! Live mappings form one dense prefix followed by sentinels.
        kDense,
        //! Live mappings may occupy disjoint logical ranges, as an SWA window advances.
        kSparseWindow,
    };

    //! @brief Construct a page table for up to `maxBatch` slots of `maxPagesPerSeq`
    //!        logical pages each, backed by a pool of `numPages` physical pages.
    //! @throws std::runtime_error if any argument is not positive
    KVPageTable(int32_t maxBatch, int32_t maxPagesPerSeq, int32_t numPages);
    ~KVPageTable() noexcept;
    KVPageTable(KVPageTable const&) = delete;
    KVPageTable& operator=(KVPageTable const&) = delete;

    //! @brief Construct a page table with an explicit row validation mode.
    //! @throws std::runtime_error if any argument is not positive or `mode` is unsupported
    KVPageTable(int32_t maxBatch, int32_t maxPagesPerSeq, int32_t numPages, Mode mode);

    //! @brief Assign every slot its static identity range: slot `b` gets
    //!        K pages `[b*maxPagesPerSeq, (b+1)*maxPagesPerSeq)` (V = K + numPages).
    void setIdentity();

    //! @brief Set slot `slot`'s live K page ids from `kPageIds[0..count)`; V ids are
    //!        derived as `k + numPages`. Entries `[count, maxPagesPerSeq)` are cleared.
    //! @throws std::runtime_error if the row description or a page id is invalid
    void setRow(int32_t slot, int32_t const* kPageIds, int32_t count);

    //! @brief Prevalidate all row updates, then apply them as one host-side commit.
    //!        A slot may appear at most once. An empty row clears that slot.
    //! @throws std::runtime_error without changing any row if any update is invalid
    void setRows(std::vector<KVPageTableRowUpdate> const& updates);

    //! @brief Set one logical mapping and its derived V mapping.
    //! @throws std::runtime_error if `slot`, `logicalPage`, or `kPageId` is out of range
    void setEntry(int32_t slot, int32_t logicalPage, int32_t kPageId);

    //! @brief Clear one logical K/V mapping to the unused-page sentinel.
    //! @throws std::runtime_error if `slot` or `logicalPage` is out of range
    void clearEntry(int32_t slot, int32_t logicalPage);

    //! @brief Validate the host table: every K id is the sentinel or in `[0, numPages)`,
    //!        and every V id is derived from its K id. Dense rows reject live mappings after
    //!        a sentinel; sparse-window rows instead reject duplicate live K ids within a slot.
    //! @param error Set to a description of the first violation found
    //! @return true if the table is valid
    bool checkInvariants(std::string& error) const;

    //! @brief Validate the table and enqueue copies for coalesced dirty-row ranges.
    //!        The first call uploads the complete table. A later call with no dirty
    //!        rows performs no CUDA operation and does not wait for a prior upload.
    //! @return true if at least one device copy was enqueued; false for a no-op
    //! @throws std::runtime_error if `checkInvariants` fails or the copy fails
    bool upload(cudaStream_t stream);

    //! @brief Upload only K/V entries changed since the previous upload. Adjacent dirty entries are
    //!        coalesced into one H2D copy. Sparse-window mutations maintain their invariants incrementally,
    //!        so this operation scales with the number of changed entries rather than the logical table width.
    //!        On the first dirty upload, the device table is initialized to sentinels before applying the changes.
    //! @throws std::runtime_error if a dense table is invalid or a copy fails
    void uploadDirty(cudaStream_t stream);

    //! @brief Number of int32 K/V entries copied by the most recent successful upload operation.
    size_t lastUploadEntryCount() const
    {
        return mLastUploadEntryCount;
    }

    //! @brief Number of H2D ranges submitted by the most recent successful upload operation.
    size_t lastUploadRangeCount() const
    {
        return mLastUploadRangeCount;
    }

    //! Gather resident rows into an active-step table without changing resident ownership or moving KV payload.
    //! `destination` must have fixed storage for at least `[numRows, 2, maxPagesPerSeq]` INT32 values.
    void gatherRows(
        rt::Tensor& destination, rt::Tensor const& residentSlots, int32_t numRows, cudaStream_t stream) const;

    //! @brief The device tensor consumed by the paged kernels: int32 `[maxBatch, 2, maxPagesPerSeq]`.
    rt::Tensor const& kernelView() const;

    //! @brief Mutable overload for binding into a `TensorMap` (which stores non-owning `Tensor*`).
    rt::Tensor& kernelView();

    //! @brief Host K row for slot `slot` (`maxPagesPerSeq` entries); used by
    //!        writeKV/gather helpers that need the logical page ids on host.
    int32_t const* hostRow(int32_t slot) const;

    //! @brief Row stride of `kernelView()` (the `maxPagesPerSeq` this table was constructed with).
    int32_t maxPagesPerSeq() const
    {
        return mMaxPagesPerSeq;
    }

    int32_t maxBatch() const
    {
        return mMaxBatch;
    }

    int32_t numPages() const
    {
        return mNumPages;
    }

    //! @brief True only if the table's current contents were last set by `setIdentity()` (and never
    //!        touched by a row or entry mutator since). Conservative: any mutator call clears this even
    //!        if the supplied ids happen to describe an identity mapping, so callers that gate
    //!        identity-only consumers fail closed rather than risk a false positive.
    bool isIdentity() const
    {
        return mIsIdentity;
    }

private:
    void applyRows(KVPageTableRowUpdate const* updates, size_t count);
    void setHostValue(size_t index, int32_t value);

    int32_t mMaxBatch{};
    int32_t mMaxPagesPerSeq{};
    int32_t mNumPages{};
    Mode mMode{Mode::kDense};
    bool mIsIdentity{false};
    std::vector<int32_t> mHost;      //!< [maxBatch, 2, maxPagesPerSeq], row-major.
    std::vector<uint8_t> mDirtyRows; //!< One bit-like byte per logical slot.
    std::vector<int32_t> mUploadedHost;
    //! Ordered changed entries permit O(changes) upload and adjacent-range coalescing.
    std::set<size_t> mDirtyIndices;
    //! Per-slot reverse mappings make sparse duplicate checks O(1) on entry rotation.
    std::vector<std::unordered_set<int32_t>> mSparseActivePages;
    //! Per-slot live logical mappings keep sparse row compaction proportional to retained pages.
    std::vector<std::unordered_map<int32_t, int32_t>> mSparseLogicalPages;
    bool mHasUploaded{false};
    size_t mLastUploadEntryCount{};
    size_t mLastUploadRangeCount{};
    mutable rt::Tensor mDevice;
    //! Immutable pinned snapshot used by an in-flight H2D upload. Reuse waits for mUploadComplete.
    rt::Tensor mUploadStaging;
    cudaEvent_t mUploadComplete{};
    bool mUploadPending{false};
};

} // namespace rt
} // namespace trt_edgellm
