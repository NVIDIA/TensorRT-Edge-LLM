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

// Addressing and moving entries in the pools a decoder keeps per sequence: the paged KV cache, and the
// unpaged recurrent and convolution state.
//
// Split from the engine because reading an entry back is a separate concern from running a forward: a test
// asserting that a token's KV survived a cache hit uses these directly.

#include "fakeState.h"

#include "common/tensor.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace substitute_engine
{

using namespace trt_edgellm;

struct FakeKVGeometry
{
    int64_t pages{};
    int64_t tokensPerPage{};
    int64_t entryWidth{}; //!< heads * channels: the run of elements one token occupies.
    int64_t pagesPerSequence{};

    static FakeKVGeometry from(rt::Tensor const& pool, size_t pageTableSize, int64_t batch)
    {
        auto const shape = pool.getShape();
        if (shape.getNumDims() != 5)
        {
            throw std::runtime_error("FakeKVGeometry: expected a [2, pages, tokensPerPage, heads, channels] KV pool");
        }
        FakeKVGeometry geometry;
        geometry.pages = shape[1];
        geometry.tokensPerPage = shape[2];
        geometry.entryWidth = shape[3] * shape[4];
        geometry.pagesPerSequence = batch > 0 ? static_cast<int64_t>(pageTableSize) / (2 * batch) : 0;
        return geometry;
    }

    //! Element offset of the run belonging to `physicalPage`'s `tokenInPage`-th token.
    int64_t entryOffset(int32_t physicalPage, int64_t tokenInPage) const noexcept
    {
        return (static_cast<int64_t>(physicalPage) * tokensPerPage + tokenInPage) * entryWidth;
    }

    //! Index into a `[batch, 2, pagesPerSequence]` page table.
    size_t pageTableIndex(int64_t slot, bool isValue, int64_t logicalPage) const noexcept
    {
        return static_cast<size_t>((slot * 2 + (isValue ? 1 : 0)) * pagesPerSequence + logicalPage);
    }
};

//! Where this forward's rows go in each sequence's cache.
//!
//! Deliberately not derived from `positions`. That binding carries the *rotary* position of a row, and for the
//! ordinary prefill and decode paths it happens to equal the cache index, which is why reading it as one works
//! everywhere until it does not: a speculative verification forward gives every candidate row the same rotary
//! position while they occupy consecutive cache slots. The runtime states the cache layout in `past_lengths` and
//! `query_lengths` instead -- row `r` of sequence `s` lands at `past_lengths[s] + r`, and
//! `attention_sequence_lengths[s]` is the sum the two imply.
//!
//! `query_lengths` also supersedes the -1 padding marker in `positions`: a row past its sequence's query length
//! is entry padding, whatever `positions` happens to hold there.

class FakeRecurrentWriter
{
public:
    //! Elements per sequence row: everything after the leading batch dimension.
    static int64_t rowWidth(rt::Tensor const& pool)
    {
        auto const shape = pool.getShape();
        int64_t const rows = shape.getNumDims() > 0 ? shape[0] : 0;
        return rows > 0 ? shape.volume() / rows : 0;
    }

    //! Fill the row `state_indices` points at with `value`.
    static void write(rt::Tensor& pool, int32_t stateIndex, double value, cudaStream_t stream)
    {
        writeValue(pool, static_cast<int64_t>(stateIndex) * rowWidth(pool), rowWidth(pool), value, stream);
    }

    //! Read back the row `state_indices` currently points at, to compare against the value its occupant should
    //! have stored.
    static double read(rt::Tensor const& pool, int32_t stateIndex, cudaStream_t stream)
    {
        int64_t const offset = static_cast<int64_t>(stateIndex) * rowWidth(pool);
        switch (pool.getDataType())
        {
        case trt_edgellm::DataType::kHALF: return __half2float(readElement<half>(pool, offset, stream));
        case trt_edgellm::DataType::kBF16: return __bfloat162float(readElement<__nv_bfloat16>(pool, offset, stream));
        case trt_edgellm::DataType::kFLOAT: return readElement<float>(pool, offset, stream);
        default: throw std::runtime_error("FakeRecurrentWriter::read: unsupported recurrent state dtype");
        }
    }

private:
    static void writeValue(rt::Tensor& pool, int64_t offset, int64_t width, double value, cudaStream_t stream)
    {
        switch (pool.getDataType())
        {
        case trt_edgellm::DataType::kHALF:
            fill<half>(pool, offset, width, __float2half(static_cast<float>(value)), stream);
            break;
        case trt_edgellm::DataType::kBF16:
            fill<__nv_bfloat16>(pool, offset, width, __float2bfloat16(static_cast<float>(value)), stream);
            break;
        case trt_edgellm::DataType::kFLOAT: fill<float>(pool, offset, width, static_cast<float>(value), stream); break;
        default:
            // Throws rather than falling through: an unhandled dtype would otherwise have this writer reinterpret
            // the pool's bytes as whatever type it guessed.
            throw std::runtime_error("FakeRecurrentWriter: unsupported recurrent state dtype");
        }
    }

    template <typename T>
    static void fill(rt::Tensor& pool, int64_t offset, int64_t width, T value, cudaStream_t stream)
    {
        std::vector<T> host(static_cast<size_t>(width), value);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<T*>(pool.rawPointer()) + offset, host.data(), host.size() * sizeof(T),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    template <typename T>
    static T readElement(rt::Tensor const& pool, int64_t offset, cudaStream_t stream)
    {
        T value{};
        CUDA_CHECK(cudaMemcpyAsync(
            &value, static_cast<T const*>(pool.rawPointer()) + offset, sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return value;
    }
};

//! Writes KV entries at the addresses the runtime published, dtype-aware.
//!
//! Only the tokens of the current forward are touched. Filling the whole pool would be simpler, but it would also
//! erase the entries belonging to other sequences and to pages a later request is meant to reuse -- which is
//! precisely what the tests this substitute exists for need to observe.
class FakeKVWriter
{
public:
    //! Store `value` in the entry `slot` names, at the address `pageTable` maps it to.
    //!
    //! Only entries the caller names are touched. Filling the whole pool would be simpler, but it would also erase
    //! the entries belonging to other sequences and to pages a later request is meant to reuse -- which is
    //! precisely what the tests this substitute exists for need to observe.
    static void write(rt::Tensor& pool, FakeKVGeometry const& geometry, std::vector<int32_t> const& pageTable,
        int64_t batchSlot, FakeStateSlot const& slot, double value, cudaStream_t stream)
    {
        int64_t const logicalPage = slot.position / geometry.tokensPerPage;
        if (logicalPage >= geometry.pagesPerSequence)
        {
            return; // Beyond what this forward mapped; the runtime is not asking for this token to be stored.
        }
        auto const index = geometry.pageTableIndex(batchSlot, slot.kind == FakeStateKind::kValue, logicalPage);
        if (index >= pageTable.size())
        {
            return;
        }
        int64_t const offset = geometry.entryOffset(pageTable[index], slot.position % geometry.tokensPerPage);
        switch (pool.getDataType())
        {
        case trt_edgellm::DataType::kHALF:
            writeRun<half>(pool, offset, geometry.entryWidth, __float2half(static_cast<float>(value)), stream);
            break;
        case trt_edgellm::DataType::kBF16:
            writeRun<__nv_bfloat16>(
                pool, offset, geometry.entryWidth, __float2bfloat16(static_cast<float>(value)), stream);
            break;
        case trt_edgellm::DataType::kFLOAT:
            writeRun<float>(pool, offset, geometry.entryWidth, static_cast<float>(value), stream);
            break;
        default:
            // Throws rather than falling through: an unhandled dtype would otherwise have this writer reinterpret
            // the pool's bytes as whatever type it guessed.
            throw std::runtime_error("FakeKVWriter: unsupported KV dtype");
        }
    }

    //! One physical page's worth of entries, one value per token, in a single copy.
    //!
    //! The engine re-reads a sequence's whole history on every forward, so this has to cost a copy per page rather
    //! than a copy per token; at the pool sizes these tests configure that is the difference between milliseconds
    //! and seconds across the suite.
    static std::vector<double> readPage(
        rt::Tensor const& pool, FakeKVGeometry const& geometry, int32_t physicalPage, cudaStream_t stream)
    {
        int64_t const offset = geometry.entryOffset(physicalPage, 0);
        int64_t const count = geometry.tokensPerPage * geometry.entryWidth;
        switch (pool.getDataType())
        {
        case trt_edgellm::DataType::kHALF:
            return strideEntries<half>(pool, offset, count, geometry, stream, __half2float);
        case trt_edgellm::DataType::kBF16:
            return strideEntries<__nv_bfloat16>(pool, offset, count, geometry, stream, __bfloat162float);
        case trt_edgellm::DataType::kFLOAT:
            return strideEntries<float>(pool, offset, count, geometry, stream, [](float v) { return v; });
        default: throw std::runtime_error("FakeKVWriter::readPage: unsupported KV dtype");
        }
    }

    //! Read back what is stored for `slot`, resolving its address through the same page table the writer used.
    //! Tests use this to ask "is this token's entry still the one written for it", independent of placement.
    static double read(rt::Tensor const& pool, FakeKVGeometry const& geometry, std::vector<int32_t> const& pageTable,
        int64_t batchSlot, FakeStateSlot const& slot, cudaStream_t stream)
    {
        int64_t const logicalPage = slot.position / geometry.tokensPerPage;
        auto const index = geometry.pageTableIndex(batchSlot, slot.kind == FakeStateKind::kValue, logicalPage);
        if (index >= pageTable.size())
        {
            throw std::runtime_error("FakeKVWriter::read: position " + std::to_string(slot.position)
                + " falls outside the page table this forward published");
        }
        int64_t const offset = geometry.entryOffset(pageTable[index], slot.position % geometry.tokensPerPage);
        switch (pool.getDataType())
        {
        case trt_edgellm::DataType::kHALF: return __half2float(readAs<half>(pool, offset, stream));
        case trt_edgellm::DataType::kBF16: return __bfloat162float(readAs<__nv_bfloat16>(pool, offset, stream));
        case trt_edgellm::DataType::kFLOAT: return readAs<float>(pool, offset, stream);
        default: throw std::runtime_error("FakeKVWriter::read: unsupported KV dtype");
        }
    }

private:
    //! Copy a run covering one page and keep the first element of each token's entry: every channel of an entry
    //! holds the same number, so one per token is the whole of the information stored there.
    template <typename T, typename Convert>
    static std::vector<double> strideEntries(rt::Tensor const& pool, int64_t offset, int64_t count,
        FakeKVGeometry const& geometry, cudaStream_t stream, Convert convert)
    {
        std::vector<T> raw(static_cast<size_t>(count));
        CUDA_CHECK(cudaMemcpyAsync(raw.data(), static_cast<T const*>(pool.rawPointer()) + offset,
            raw.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<double> entries(static_cast<size_t>(geometry.tokensPerPage));
        for (int64_t token = 0; token < geometry.tokensPerPage; ++token)
        {
            entries[static_cast<size_t>(token)]
                = static_cast<double>(convert(raw[static_cast<size_t>(token * geometry.entryWidth)]));
        }
        return entries;
    }

    template <typename T>
    static void writeRun(rt::Tensor& pool, int64_t offset, int64_t width, T value, cudaStream_t stream)
    {
        thread_local std::vector<T> host;
        host.assign(static_cast<size_t>(width), value);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<T*>(pool.rawPointer()) + offset, host.data(), host.size() * sizeof(T),
            cudaMemcpyHostToDevice, stream));
    }

    template <typename T>
    static T readAs(rt::Tensor const& pool, int64_t offset, cudaStream_t stream)
    {
        T value{};
        CUDA_CHECK(cudaMemcpyAsync(
            &value, static_cast<T const*>(pool.rawPointer()) + offset, sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return value;
    }
};

//! State-based EngineExecutor substitute. Construction fixes the binding-name surface (from `FakeLayerComposition`);
//! `prepare()` records the TensorMap handed in, and `execute()` writes every KV binding through
//! `FakeContextWriter` plus the logits binding through the `tokensPerRow` pattern `writeLogits()` already defines
//! in substituteEngine.h, so existing assertions on sampled tokens keep working unchanged.

} // namespace substitute_engine
