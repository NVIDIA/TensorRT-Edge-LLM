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

#include "qwen4ExpNgramPreprocessor.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "runtime/embedding/pleIoBudget.h"
#include "runtime/weight/checkpointReader.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace trt_edgellm
{
namespace rt
{

struct Qwen4ExpNgramPreprocessor::PooledResources
{
    PooledResources(PleTable table, int32_t slots, cudaStream_t stream, int64_t maxTokens, int32_t heads,
        PleResidency mode, std::shared_ptr<PleIoBudget> pleIoBudget)
        : store(std::move(table), slots, stream, PleIoConfig{},
              mode == PleResidency::kRESIDENT ? PleRowStore::Residency::kRESIDENT : PleRowStore::Residency::kCACHED,
              std::move(pleIoBudget))
        , rowMap({maxTokens, heads}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "ple_row_map")
        , hostMap({maxTokens, heads}, DeviceType::kCPU, trt_edgellm::DataType::kINT32)
    {
    }
    PleRowStore store;
    Tensor rowMap;
    Tensor hostMap;
    std::vector<PleRowStore::RequestLease> leases;
};

Qwen4ExpNgramPreprocessor::Qwen4ExpNgramPreprocessor(LLMEngineConfig const& config, int32_t maxBatchSize,
    int32_t maxSeqLen, TensorMap& tensorMap, cudaStream_t stream, PleRuntimeConfig const& pleConfig,
    std::filesystem::path const& checkpointDir, std::shared_ptr<PleIoBudget> pleIoBudget)
    : mConfig(config)
    , mStream(stream)
{
    ELLM_CHECK(mConfig.ngramEnabled(), "Qwen4ExpNgramPreprocessor constructed for a model without an injection");
    ELLM_CHECK(maxBatchSize > 0, "Qwen4ExpNgramPreprocessor requires positive max batch size");
    ELLM_CHECK(maxSeqLen > 0, "Qwen4ExpNgramPreprocessor requires positive max sequence length");
    ELLM_CHECK(stream != nullptr, "Pooled PLE requires an explicit stream");

    // Open the table the engine gathers from, after checking its geometry against the model.
    int32_t const ngramHeads = mConfig.ngramHeads();
    auto const& poolConfig = mConfig.plePool;
    ELLM_CHECK(poolConfig.heads == ngramHeads,
        "PLE pool head count " + std::to_string(poolConfig.heads) + " does not match the model's "
            + std::to_string(ngramHeads));
    ELLM_CHECK(
        poolConfig.rowWidth * poolConfig.heads == mConfig.ngramEmbedDim, "PLE pool rows do not fill ngram_embed_dim");
    ELLM_CHECK(!poolConfig.source.shards.empty() || !poolConfig.source.tensorKey.empty(),
        "PLE pool requires checkpoint source metadata; re-export and rebuild the engine.");
    auto const sourceDir = checkpointDir.empty() ? std::filesystem::path(poolConfig.checkpointDir) : checkpointDir;
    auto table = PleTable::fromSource(poolConfig.source, poolConfig.rowCount, poolConfig.rowWidth, sourceDir,
        pleConfig.filePath, poolConfig.dtype, poolConfig.scale);
    ELLM_CHECK(table.rowCount() < std::numeric_limits<int32_t>::max(), "PLE table exceeds INT32 slot capacity");

    // Read the hash parameters from the checkpoint the rows came from, rather than carrying copies
    // in the engine config, so they cannot drift out of step with the rows they index.
    ELLM_CHECK(!poolConfig.layerMultipliersKey.empty(),
        "PLE pool is missing its hash constant keys; re-export and rebuild the engine.");
    {
        CheckpointReader constants(sourceDir); // Scoped: the views alias its mapping
        auto const readInt64 = [&constants](std::string const& key, int32_t expected) {
            CheckpointReader::View view;
            ELLM_CHECK(constants.findHost(key, view), "PLE hash constant is missing: " + key);
            ELLM_CHECK(view.dtype == trt_edgellm::DataType::kINT64 && view.shape.getNumDims() == 1
                    && view.shape[0] == expected,
                "PLE hash constant must be a 1-D INT64 tensor of " + std::to_string(expected) + " values: " + key);
            std::vector<int64_t> values(static_cast<size_t>(expected));
            std::memcpy(values.data(), view.data, values.size() * sizeof(int64_t));
            return values;
        };
        auto const multipliers = readInt64(poolConfig.layerMultipliersKey, mConfig.ngramSize);
        mLayerMultipliers.assign(multipliers.begin(), multipliers.end());
        mHeadVocabSizes = readInt64(poolConfig.headVocabSizesKey, ngramHeads);
        mHeadOffsets = readInt64(poolConfig.headOffsetsKey, ngramHeads);
    }
    for (int32_t head = 0; head < ngramHeads; ++head)
    {
        ELLM_CHECK(mHeadOffsets[head] >= 0 && mHeadVocabSizes[head] > 0
                && mHeadOffsets[head] + mHeadVocabSizes[head] <= table.rowCount(),
            "n-gram head " + std::to_string(head) + " addresses past the end of the table");
    }

    // Flatten everything the row-ID computation needs into one struct.
    mParams.ngramSize = mConfig.ngramSize;
    mParams.headsPerNgram = mConfig.ngramHeadsPerOrder;
    mParams.ngramHeads = ngramHeads;
    mParams.eosTokenId = mConfig.eosTokenIds.empty() ? 0 : mConfig.eosTokenIds.back();
    mParams.layerMultipliers = mLayerMultipliers.data();
    mParams.headVocabSizes = mHeadVocabSizes.data();
    mParams.headOffsets = mHeadOffsets.data();

    // Host scratch, sized once for the worst-case batch.
    mStatePoolRows = maxBatchSize;
    int32_t const contextLength = mConfig.ngramContextLength();
    mTokenHistory.assign(static_cast<size_t>(maxBatchSize) * contextLength, mParams.eosTokenId);
    mActiveHistory.resize(static_cast<size_t>(maxBatchSize) * contextLength);
    int64_t const maxTokens = static_cast<int64_t>(maxBatchSize) * maxSeqLen;
    mRowIds.resize(static_cast<size_t>(maxTokens) * ngramHeads);
    mRequests.resize(static_cast<size_t>(maxBatchSize));

    // Allocate the pool and publish it to the engine's bindings.
    int32_t const slots = pleConfig.residency == PleResidency::kRESIDENT ? static_cast<int32_t>(table.rowCount() + 1)
                                                                         : pleConfig.cacheSlots;
    ELLM_CHECK(slots >= 2 && slots <= table.rowCount() + 1, "PLE cache slots must be between 2 and row_count + 1");
    mPooled = std::make_unique<PooledResources>(
        std::move(table), slots, stream, maxTokens, ngramHeads, pleConfig.residency, std::move(pleIoBudget));
    try
    {
        tensorMap.set("ple_cache_pool", mPooled->store.pool());
        tensorMap.set("ple_row_map", mPooled->rowMap);
        reshapeOutput(maxTokens);
        CUDA_CHECK(cudaMemsetAsync(mPooled->rowMap.rawPointer(), 0, mPooled->rowMap.getMemoryCapacity(), mStream));
    }
    catch (...)
    {
        if (!drainPooled())
        {
            (void) mPooled.release();
        }
        throw;
    }

    LOG_INFO("Initialized Qwen4-Exp n-gram preprocessor: rows=%ld width=%ld heads=%d slots=%d residency=%s eos=%d",
        static_cast<long>(poolConfig.rowCount), static_cast<long>(poolConfig.rowWidth), ngramHeads, slots,
        pleConfig.residency == PleResidency::kRESIDENT ? "resident" : "cached", mParams.eosTokenId);
}

Qwen4ExpNgramPreprocessor::~Qwen4ExpNgramPreprocessor() noexcept
{
    if (mPooled && !drainPooled())
    {
        // The store's leases pin pool rows the device may still be reading; leaking the whole
        // resource set is the only safe outcome when the stream cannot be drained.
        (void) mPooled.release();
    }
}

bool Qwen4ExpNgramPreprocessor::drainPooled() noexcept
{
    if (mStream == nullptr)
    {
        return true;
    }
    return cudaStreamSynchronize(mStream) == cudaSuccess;
}

void Qwen4ExpNgramPreprocessor::releaseLeases()
{
    if (!mPooled->leases.empty())
    {
        // The pins keep this batch's rows in place while the engine reads them, so they can only
        // be dropped once that work has completed.
        ELLM_CHECK(drainPooled(), "n-gram PLE consumers did not complete before lease release");
        mPooled->leases.clear();
    }
}

Tensor const& Qwen4ExpNgramPreprocessor::rowMap() const
{
    ELLM_CHECK(mPooled != nullptr, "n-gram preprocessor has no row map");
    return mPooled->rowMap;
}

PleRowStore::Stats const& Qwen4ExpNgramPreprocessor::stats() const
{
    ELLM_CHECK(mPooled != nullptr, "n-gram preprocessor has no row store");
    return mPooled->store.stats();
}

int32_t* Qwen4ExpNgramPreprocessor::historyForSlot(int32_t residentSlot)
{
    int32_t const contextLength = mConfig.ngramContextLength();
    ELLM_CHECK(residentSlot >= 0 && static_cast<size_t>(residentSlot + 1) * contextLength <= mTokenHistory.size(),
        "n-gram history slot out of range: " + std::to_string(residentSlot));
    return mTokenHistory.data() + static_cast<size_t>(residentSlot) * contextLength;
}

void Qwen4ExpNgramPreprocessor::advanceHistory(int32_t residentSlot, int32_t const* tokens, int32_t count)
{
    int32_t const contextLength = mConfig.ngramContextLength();
    int32_t* history = historyForSlot(residentSlot);
    int32_t const taken = std::min(count, contextLength);
    // A step shorter than the context keeps part of the old history, so the two are joined rather
    // than the new tokens simply replacing it.
    std::copy(history + taken, history + contextLength, history);
    std::copy(tokens + count - taken, tokens + count, history + contextLength - taken);
}

void Qwen4ExpNgramPreprocessor::seedHistory(int32_t residentSlot, int32_t const* consumedTokens, int32_t consumedCount)
{
    ELLM_CHECK(consumedCount >= 0, "n-gram history cannot be seeded from a negative token count");
    int32_t const contextLength = mConfig.ngramContextLength();
    int32_t* history = historyForSlot(residentSlot);
    int32_t const carried = std::min(consumedCount, contextLength);
    // A short history pads on the left with EOS rather than with zeros: EOS is the segment marker,
    // so the padding is what makes the first tokens read as the start of a document.
    std::fill(history, history + contextLength - carried, mParams.eosTokenId);
    if (carried > 0)
    {
        std::copy(consumedTokens + consumedCount - carried, consumedTokens + consumedCount,
            history + contextLength - carried);
    }
}

void Qwen4ExpNgramPreprocessor::embed(int32_t const* hostTokenIds, int32_t const* hostQueryLengths,
    int32_t const* residentSlots, int32_t numSequences, int32_t queryWidth, cudaStream_t stream,
    int32_t const* hostTreeParents)
{
    int64_t const physicalTokens = static_cast<int64_t>(numSequences) * queryWidth;
    ELLM_CHECK(numSequences > 0 && queryWidth > 0, "n-gram step must hold at least one token");
    ELLM_CHECK(numSequences <= mStatePoolRows, "n-gram sequence count exceeds the state pool");
    ELLM_CHECK(stream == mStream, "n-gram PLE must use its bound consumer stream");
    releaseLeases();
    reshapeOutput(physicalTokens);

    int32_t const ngramHeads = mParams.ngramHeads;
    int32_t const contextLength = mConfig.ngramContextLength();
    auto const slotOf
        = [residentSlots](int32_t sequence) { return residentSlots ? residentSlots[sequence] : sequence; };

    // The hash kernel indexes the history by execution row, the store by resident slot, so the
    // step's histories are gathered before it runs and folded back per sequence after.
    for (int32_t sequence = 0; sequence < numSequences; ++sequence)
    {
        std::copy_n(historyForSlot(slotOf(sequence)), contextLength,
            mActiveHistory.begin() + static_cast<ptrdiff_t>(sequence) * contextLength);
    }

    // Row IDs for the whole padded batch in one call. Padded positions hash their padding and
    // would map to some row; they are masked to -1 below so the store never admits them.
    if (hostTreeParents == nullptr)
    {
        computeNgramRowIds(
            hostTokenIds, mActiveHistory.data(), mRowIds.data(), nullptr, numSequences, queryWidth, mParams);
    }
    else
    {
        // Each node becomes a one-token sequence whose history is its nearest ancestors, topped up
        // from the committed history when the path is shorter than the context.
        mNodeHistory.resize(static_cast<size_t>(physicalTokens) * contextLength);
        for (int32_t sequence = 0; sequence < numSequences; ++sequence)
        {
            int32_t const* const committed = mActiveHistory.data() + static_cast<ptrdiff_t>(sequence) * contextLength;
            int32_t const* const tokens = hostTokenIds + static_cast<ptrdiff_t>(sequence) * queryWidth;
            int32_t const* const parents = hostTreeParents + static_cast<ptrdiff_t>(sequence) * queryWidth;
            for (int32_t node = 0; node < queryWidth; ++node)
            {
                int32_t* const window
                    = mNodeHistory.data() + (static_cast<ptrdiff_t>(sequence) * queryWidth + node) * contextLength;
                int32_t slot = contextLength - 1;
                int32_t child = node;
                for (int32_t ancestor = parents[node]; slot >= 0 && ancestor >= 0; --slot)
                {
                    ELLM_CHECK(ancestor < child, "n-gram tree parents must precede their children");
                    window[slot] = tokens[ancestor];
                    child = ancestor;
                    ancestor = parents[ancestor];
                }
                std::copy_n(committed + (contextLength - 1 - slot), slot + 1, window);
            }
        }
        computeNgramRowIds(hostTokenIds, mNodeHistory.data(), mRowIds.data(), nullptr,
            static_cast<int32_t>(physicalTokens), 1, mParams);
    }

    mRequests.resize(static_cast<size_t>(numSequences));
    for (int32_t sequence = 0; sequence < numSequences; ++sequence)
    {
        int32_t const valid
            = hostQueryLengths == nullptr ? queryWidth : std::min(hostQueryLengths[sequence], queryWidth);
        auto& request = mRequests[static_cast<size_t>(sequence)];
        request.clear();
        request.reserve(static_cast<size_t>(queryWidth) * ngramHeads);
        for (int32_t position = 0; position < queryWidth; ++position)
        {
            size_t const token = static_cast<size_t>(sequence) * queryWidth + position;
            for (int32_t head = 0; head < ngramHeads; ++head)
            {
                // -1 is the store's padding row; it maps to slot 0 without occupying the cache.
                request.push_back(position < valid ? mRowIds[token * ngramHeads + head] : -1);
            }
        }
        // The history advances over this sequence's real tokens only. The kernel's own
        // nextTokenHistory would take the tail of the padded row instead, which for a short
        // sequence in a mixed batch is padding rather than the tokens preceding the next chunk.
        if (valid > 0 && hostTreeParents == nullptr)
        {
            advanceHistory(slotOf(sequence), hostTokenIds + static_cast<size_t>(sequence) * queryWidth, valid);
        }
    }

    // Leases pin this step's rows until the next prepare; the device reads them in between.
    mPooled->leases = mPooled->store.prepare(mRequests);
    auto* map = mPooled->hostMap.dataPointer<int32_t>();
    for (size_t sequence = 0; sequence < mPooled->leases.size(); ++sequence)
    {
        auto const& mapping = mPooled->leases[sequence].mapping();
        std::copy(mapping.begin(), mapping.end(), map + sequence * static_cast<size_t>(queryWidth) * ngramHeads);
    }
    CUDA_CHECK(cudaMemcpyAsync(mPooled->rowMap.rawPointer(), mPooled->hostMap.rawPointer(),
        static_cast<size_t>(physicalTokens) * ngramHeads * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    auto const& counters = mPooled->store.stats();
    LOG_DEBUG("n-gram rows: requested=%ld unique=%zu misses=%zu readBytes=%zu usefulBytes=%zu waves=%zu",
        static_cast<long>(physicalTokens) * ngramHeads, counters.uniqueRows, counters.misses, counters.readBytes,
        counters.usefulBytes, counters.fillWaves);
}

void Qwen4ExpNgramPreprocessor::resetHistory(int32_t residentSlot)
{
    seedHistory(residentSlot, /*consumedTokens=*/nullptr, /*consumedCount=*/0);
}

void Qwen4ExpNgramPreprocessor::reshapeOutput(int64_t physicalTokens)
{
    ELLM_CHECK(physicalTokens > 0 && physicalTokens <= mPooled->hostMap.getShape()[0],
        "n-gram row map shape exceeds allocated capacity");
    ELLM_CHECK(mPooled->rowMap.reshape({physicalTokens, mConfig.ngramHeads()}), "Cannot reshape the n-gram row map");
}

} // namespace rt
} // namespace trt_edgellm
