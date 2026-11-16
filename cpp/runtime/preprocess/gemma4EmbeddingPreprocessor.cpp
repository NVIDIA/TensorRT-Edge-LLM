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

#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "kernels/embeddingKernels/pleGather.h"
#include "runtime/embedding/pleRowStore.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{

struct Gemma4EmbeddingPreprocessor::PooledResources
{
    PooledResources(PleTable table, int32_t slots, cudaStream_t stream, int32_t maxBatch, int32_t maxSeq,
        PleResidency mode, std::shared_ptr<PleIoBudget> pleIoBudget)
        : store(std::move(table), slots, stream, PleIoConfig{},
              mode == PleResidency::kRESIDENT ? PleRowStore::Residency::kRESIDENT : PleRowStore::Residency::kCACHED,
              std::move(pleIoBudget))
        , rowMap({static_cast<int64_t>(maxBatch) * maxSeq, 1}, DeviceType::kGPU, trt_edgellm::DataType::kINT32,
              "ple_row_map")
        , residency(mode)
    {
        if (mode == PleResidency::kCACHED)
        {
            hostMap = Tensor({maxBatch, maxSeq}, DeviceType::kCPU, trt_edgellm::DataType::kINT32);
        }
    }
    PleRowStore store;
    Tensor rowMap;
    Tensor hostMap;
    PleResidency residency;
    std::vector<PleRowStore::RequestLease> leases;
};

Gemma4EmbeddingPreprocessor::Gemma4EmbeddingPreprocessor(LLMEngineConfig const& config, int32_t maxBatchSize,
    int32_t maxSeqLen, TensorMap& tensorMap, cudaStream_t stream, PleRuntimeConfig const& pleConfig,
    std::filesystem::path const& checkpointDir, std::shared_ptr<PleIoBudget> pleIoBudget)
    : mConfig(config)
    , mStream(stream)
    , mMaxBatchSize(maxBatchSize)
    , mMaxSeqLen(maxSeqLen)
{
    ELLM_CHECK(mConfig.gemmaPleEnabled, "Gemma4EmbeddingPreprocessor constructed while PLE is disabled");
    ELLM_CHECK(maxBatchSize > 0, "Gemma4EmbeddingPreprocessor requires positive max batch size");
    ELLM_CHECK(maxSeqLen > 0, "Gemma4EmbeddingPreprocessor requires positive max sequence length");

    ELLM_CHECK(stream != nullptr, "Pooled PLE requires an explicit stream");
    auto const& poolConfig = mConfig.plePool;
    ELLM_CHECK(!poolConfig.source.tensorKey.empty(),
        "PLE pool requires checkpoint source metadata; re-export and rebuild the engine.");
    auto table = PleTable::fromSource(poolConfig.source, poolConfig.rowCount, poolConfig.rowWidth,
        checkpointDir.empty() ? std::filesystem::path(poolConfig.checkpointDir) : checkpointDir, pleConfig.filePath,
        poolConfig.dtype, poolConfig.scale);
    ELLM_CHECK(poolConfig.rowWidth == static_cast<int64_t>(mConfig.gemmaPleLayerCount) * mConfig.gemmaPleHiddenSize,
        "PLE row width must equal gemma_ple_layer_count * gemma_ple_hidden_size");
    ELLM_CHECK(table.rowCount() < std::numeric_limits<int32_t>::max(), "PLE table exceeds INT32 slot capacity");
    if (mConfig.imageTokenId >= 0 || mConfig.audioTokenId >= 0)
    {
        ELLM_CHECK(mConfig.padTokenId >= 0 && mConfig.padTokenId < table.rowCount(),
            "Gemma4 multimodal PLE pad_token_id is outside the PLE table vocabulary");
    }
    int32_t const slots = pleConfig.residency == PleResidency::kRESIDENT ? static_cast<int32_t>(table.rowCount() + 1)
                                                                         : pleConfig.cacheSlots;
    ELLM_CHECK(slots >= 2 && slots <= table.rowCount() + 1, "PLE cache slots must be between 2 and row_count + 1");
    mDistinctRowCapacity = slots - 1;
    mPooled = std::make_unique<PooledResources>(
        std::move(table), slots, stream, maxBatchSize, maxSeqLen, pleConfig.residency, std::move(pleIoBudget));
    try
    {
        tensorMap.set("ple_cache_pool", mPooled->store.pool());
        tensorMap.set("ple_row_map", mPooled->rowMap);
        reshapeOutputsTokenMajor(static_cast<int64_t>(maxBatchSize) * maxSeqLen);
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
}

Gemma4EmbeddingPreprocessor::~Gemma4EmbeddingPreprocessor() noexcept
{
    if (mPooled && !drainPooled())
    {
        LOG_ERROR("Gemma PLE quarantines resources because stream completion could not be established.");
        (void) mPooled.release();
    }
}

bool Gemma4EmbeddingPreprocessor::drainPooled() noexcept
{
    auto const status = cudaStreamSynchronize(mStream);
    if (status != cudaSuccess)
    {
        mPoisoned = true;
        LOG_ERROR("Gemma PLE stream drain failed: %s", cudaGetErrorString(status));
    }
    return status == cudaSuccess;
}

void Gemma4EmbeddingPreprocessor::releasePooledLeases()
{
    ELLM_CHECK(!mPoisoned, "Gemma pooled PLE is poisoned after a CUDA completion failure");
    if (!mPooled->leases.empty())
    {
        ELLM_CHECK(drainPooled(), "Gemma PLE consumers did not complete before lease release");
        mPooled->leases.clear();
    }
}

void Gemma4EmbeddingPreprocessor::reshapeOutputsTokenMajor(int64_t physicalTokens)
{
    ELLM_CHECK(physicalTokens > 0 && physicalTokens <= static_cast<int64_t>(mMaxBatchSize) * mMaxSeqLen,
        "Gemma PLE shape exceeds allocated map capacity");
    ELLM_CHECK(mPooled->rowMap.reshape({physicalTokens, 1}), "Cannot reshape Gemma PLE row map");
}

bool Gemma4EmbeddingPreprocessor::needsHostTokens() const noexcept
{
    return mPooled != nullptr && mPooled->residency == PleResidency::kCACHED;
}

void Gemma4EmbeddingPreprocessor::embed(Tensor const& tokenIds, cudaStream_t stream,
    std::vector<int32_t> const& queryLengths, Tensor const* deviceQueryLengths, Tensor const* hostTokenIds)
{
    auto tokenShape = tokenIds.getShape();
    ELLM_CHECK(tokenShape.getNumDims() == 1 || tokenShape.getNumDims() == 2,
        "Gemma4 PLE token IDs must be [physical_tokens] or [batch, seq_len]");
    if (tokenShape.getNumDims() == 1)
    {
        ELLM_CHECK(deviceQueryLengths != nullptr && deviceQueryLengths->getShape().getNumDims() == 1,
            "Flat PLE token IDs require per-request device query lengths");
        int64_t const batchSize = deviceQueryLengths->getShape()[0];
        ELLM_CHECK(batchSize > 0 && tokenShape[0] % batchSize == 0,
            "Flat PLE token IDs must contain equal-width physical request rows");
        tokenShape = Coords{batchSize, tokenShape[0] / batchSize};
    }
    ELLM_CHECK(stream == mStream, "Gemma pooled PLE must use its bound consumer stream");
    ELLM_CHECK(tokenIds.getDataType() == trt_edgellm::DataType::kINT32 && tokenIds.getDeviceType() == DeviceType::kGPU,
        "PLE token IDs must be GPU INT32");
    ELLM_CHECK(tokenShape[0] > 0 && tokenShape[0] <= mMaxBatchSize && tokenShape[1] > 0 && tokenShape[1] <= mMaxSeqLen,
        "Gemma PLE token shape exceeds allocated capacity");
    ELLM_CHECK(queryLengths.empty() || deviceQueryLengths != nullptr,
        "PLE host query lengths require matching device query lengths");
    ELLM_CHECK(mPooled->residency == PleResidency::kRESIDENT || queryLengths.empty() == (deviceQueryLengths == nullptr),
        "Cached PLE padding requires matching host and device query lengths");
    if (mPooled->residency == PleResidency::kCACHED)
    {
        ELLM_CHECK(hostTokenIds != nullptr && hostTokenIds->getDeviceType() == DeviceType::kCPU
                && hostTokenIds->getDataType() == trt_edgellm::DataType::kINT32
                && hostTokenIds->getShape().volume() == tokenShape.volume(),
            "Cached PLE requires a CPU INT32 snapshot of the physical token IDs");
    }
    if (deviceQueryLengths != nullptr)
    {
        ELLM_CHECK(deviceQueryLengths->getDeviceType() == DeviceType::kGPU
                && deviceQueryLengths->getDataType() == trt_edgellm::DataType::kINT32
                && deviceQueryLengths->getShape() == Coords{tokenShape[0]},
            "PLE device query lengths must be GPU INT32 [batch]");
    }
    if (!queryLengths.empty())
    {
        ELLM_CHECK(queryLengths.size() == static_cast<size_t>(tokenShape[0]), "PLE query length count mismatch");
        for (int32_t const length : queryLengths)
        {
            ELLM_CHECK(length >= 0 && length <= tokenShape[1], "PLE query length exceeds token row");
        }
    }
    releasePooledLeases();
    reshapeOutputsTokenMajor(tokenShape[0] * tokenShape[1]);
    embedPooled(tokenIds, tokenShape, queryLengths, deviceQueryLengths, hostTokenIds);
}

void Gemma4EmbeddingPreprocessor::embedPacked(Tensor const& tokenIds, cudaStream_t stream, Tensor const* hostTokenIds)
{
    ELLM_CHECK(stream == mStream, "Gemma pooled PLE must use its bound consumer stream");
    ELLM_CHECK(tokenIds.getDataType() == trt_edgellm::DataType::kINT32 && tokenIds.getDeviceType() == DeviceType::kGPU,
        "PLE token IDs must be GPU INT32");
    int64_t const physicalTokens = tokenIds.getShape().volume();
    ELLM_CHECK(physicalTokens > 0 && physicalTokens <= static_cast<int64_t>(mMaxBatchSize) * mMaxSeqLen,
        "Packed Gemma PLE token count exceeds allocated capacity");
    if (mPooled->residency == PleResidency::kCACHED)
    {
        ELLM_CHECK(hostTokenIds != nullptr && hostTokenIds->getDeviceType() == DeviceType::kCPU
                && hostTokenIds->getDataType() == trt_edgellm::DataType::kINT32
                && hostTokenIds->getShape().volume() == physicalTokens,
            "Cached PLE requires a CPU INT32 snapshot of the physical token IDs");
    }
    releasePooledLeases();
    reshapeOutputsTokenMajor(physicalTokens);
    embedPooled(tokenIds, Coords{1, physicalTokens}, {}, nullptr, hostTokenIds);
}

void Gemma4EmbeddingPreprocessor::embedPooled(Tensor const& tokenIds, Coords const& shape,
    std::vector<int32_t> const& queryLengths, Tensor const* deviceQueryLengths, Tensor const* hostTokenIds)
{
    auto& pooled = *mPooled;
    int64_t const count = shape[0] * shape[1];
    auto const vocabulary = mConfig.plePool.rowCount;
    if (pooled.residency == PleResidency::kRESIDENT)
    {
        CUDA_CHECK(
            kernel::launchGemmaPleResidentMap(tokenIds.dataPointer<int32_t>(), pooled.rowMap.dataPointer<int32_t>(),
                count, vocabulary, mConfig.padTokenId, mConfig.imageTokenId, mConfig.audioTokenId, mStream,
                deviceQueryLengths ? deviceQueryLengths->dataPointer<int32_t>() : nullptr, shape[1]));
        return;
    }
    auto const* ids = hostTokenIds->dataPointer<int32_t>();
    std::vector<std::vector<int64_t>> requests(static_cast<size_t>(shape[0]));
    for (int64_t batch = 0; batch < shape[0]; ++batch)
    {
        auto& request = requests[batch];
        request.reserve(static_cast<size_t>(shape[1]));
        for (int64_t token = 0; token < shape[1]; ++token)
        {
            int32_t const id = ids[batch * shape[1] + token];
            bool const padding = !queryLengths.empty() && token >= queryLengths[batch];
            bool const media = (mConfig.imageTokenId >= 0 && id == mConfig.imageTokenId)
                || (mConfig.audioTokenId >= 0 && id == mConfig.audioTokenId);
            int32_t const lookupId = media ? mConfig.padTokenId : id;
            request.push_back(padding || lookupId < 0 || lookupId >= vocabulary ? -1 : lookupId);
        }
    }
    pooled.leases = pooled.store.prepare(requests);
    auto* map = pooled.hostMap.dataPointer<int32_t>();
    for (size_t batch = 0; batch < pooled.leases.size(); ++batch)
    {
        auto const& mapping = pooled.leases[batch].mapping();
        std::copy(mapping.begin(), mapping.end(), map + batch * shape[1]);
    }
    CUDA_CHECK(cudaMemcpyAsync(pooled.rowMap.rawPointer(), pooled.hostMap.rawPointer(), count * sizeof(int32_t),
        cudaMemcpyHostToDevice, mStream));
}

} // namespace rt
} // namespace trt_edgellm
