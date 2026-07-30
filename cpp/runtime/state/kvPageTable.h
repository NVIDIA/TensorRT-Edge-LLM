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

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

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
    //! @brief Construct a page table for up to `maxBatch` slots of `maxPagesPerSeq`
    //!        logical pages each, backed by a pool of `numPages` physical pages.
    //! @throws std::runtime_error if any argument is not positive
    KVPageTable(int32_t maxBatch, int32_t maxPagesPerSeq, int32_t numPages);
    ~KVPageTable() noexcept;
    KVPageTable(KVPageTable const&) = delete;
    KVPageTable& operator=(KVPageTable const&) = delete;

    //! @brief Assign every slot its static identity range: slot `b` gets
    //!        K pages `[b*maxPagesPerSeq, (b+1)*maxPagesPerSeq)` (V = K + numPages).
    void setIdentity();

    //! @brief Set slot `slot`'s K page ids from `kPageIds[0..count)`; V ids are
    //!        derived as `k + numPages` (`-1` stays `-1`). Entries `[count, maxPagesPerSeq)`
    //!        are set to the sentinel in both halves.
    //! @throws std::runtime_error if `slot` or `count` is out of range
    void setRow(int32_t slot, int32_t const* kPageIds, int32_t count);

    //! @brief Compact host rows using an old-slot to new-slot mapping without copying any KV page data.
    //!        Every destination in `[0, newBatch)` must appear exactly once; `-1` retires an old row.
    void compactRows(std::vector<int32_t> const& oldToNew, int32_t newBatch);

    //! @brief Validate the host K table: every id is either the sentinel `-1`
    //!        or in `[0, numPages)`, and no live id follows a sentinel within a row.
    //! @param error Set to a description of the first violation found
    //! @return true if the table is valid
    bool checkInvariants(std::string& error) const;

    //! @brief Validate the table (throws on failure) and copy it to the device tensor.
    //! @throws std::runtime_error if `checkInvariants` fails or the copy fails
    void upload(cudaStream_t stream);

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

    //! @brief True only if the table's current contents were last set by `setIdentity()` (and never
    //!        touched by `setRow()` since). Conservative: any `setRow()` call clears this even if the
    //!        supplied ids happen to describe an identity mapping, so callers that gate identity-only
    //!        consumers (e.g. the Alpamayo action runner) fail closed rather than
    //!        risk a false positive.
    bool isIdentity() const
    {
        return mIsIdentity;
    }

private:
    int32_t* mutableHostRow(int32_t slot);

    int32_t mMaxBatch{};
    int32_t mMaxPagesPerSeq{};
    int32_t mNumPages{};
    bool mIsIdentity{false};
    std::vector<int32_t> mHost;        //!< [maxBatch, 2, maxPagesPerSeq], row-major.
    std::vector<int32_t> mHostScratch; //!< Preallocated row-compaction scratch.
    rt::Tensor mDevice;
    //! Immutable pinned snapshot used by an in-flight H2D upload. Reuse waits for mUploadComplete.
    rt::Tensor mUploadStaging;
    cudaEvent_t mUploadComplete{};
    bool mUploadPending{false};
};

} // namespace rt
} // namespace trt_edgellm
