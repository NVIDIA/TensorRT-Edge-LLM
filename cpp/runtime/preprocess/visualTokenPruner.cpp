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

#include "runtime/preprocess/visualTokenPruner.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "kernels/dart/dartGatherKernels.h"
#include "runtime/preprocess/dartPruner.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{

// ---------------------------------------------------------------------------
// VisualTokenPruner base: guards + partitioning (pruneForPrefill) and the
// shared subset-selection compaction (compactToKeepList).
// ---------------------------------------------------------------------------

VisualTokenPruner::VisualTokenPruner(VisualPrunerConfig const& config, LLMEngineConfig const& engineConfig)
    : mConfig(config)
    , mImageTokenId(engineConfig.imageTokenId)
    , mHiddenSize(engineConfig.hiddenSize)
    , mRotaryDim(engineConfig.rotaryDim)
    , mMaxKVCacheCapacity(engineConfig.maxKVCacheCapacity)
    , mMaxBatchSize(std::max(1, engineConfig.maxSupportedBatchSize))
    , mMaxInputLength(engineConfig.maxSupportedInputLength)
{
    check::check(mConfig.reductionRatio > 0.0F && mConfig.reductionRatio < 1.0F, "reductionRatio must be in (0, 1)");
    check::check(mImageTokenId >= 0, "visual-token pruning requires a VLM engine with an image token id");

    int32_t const maxInputLen = engineConfig.maxSupportedInputLength;
    mKeepIdxDevice = Tensor({mMaxBatchSize, mMaxKVCacheCapacity}, DeviceType::kGPU, nvinfer1::DataType::kINT32,
        "VisualTokenPruner::keepIdxDevice");
    mKeepIdxHost = Tensor({mMaxBatchSize, mMaxKVCacheCapacity}, DeviceType::kCPU, nvinfer1::DataType::kINT32,
        "VisualTokenPruner::keepIdxHost");
    int64_t const embedPlaneBytes = static_cast<int64_t>(maxInputLen) * mHiddenSize * sizeof(half);
    int64_t const ropePlaneBytes = static_cast<int64_t>(mMaxKVCacheCapacity) * mRotaryDim * sizeof(float);
    mGatherScratch = Tensor({std::max(embedPlaneBytes, ropePlaneBytes)}, DeviceType::kGPU, nvinfer1::DataType::kINT8,
        "VisualTokenPruner::gatherScratch");

    // Batched-flow vectors are fully preallocated so the prefill hot path never touches the heap.
    mSlotKeepLists.resize(mMaxBatchSize);
    for (auto& keepList : mSlotKeepLists)
    {
        keepList.reserve(maxInputLen);
    }
    mOldLens.reserve(mMaxBatchSize);
    mNewLens.reserve(mMaxBatchSize);
    mKeepStartOffsets.reserve(static_cast<size_t>(mMaxBatchSize) + 1);
    mConcatenatedKeepIndices.reserve(static_cast<size_t>(mMaxBatchSize) * maxInputLen);
    mKeepIndicesHost.reserve(maxInputLen);
    mImagePositions.reserve(maxInputLen);
    mTextPositions.reserve(maxInputLen);
}

void VisualTokenPruner::preallocateAuxiliaryBuffers()
{
    // Upper bound on raw feature rows: every token of every slot is visual.
    int64_t const maxRows = static_cast<int64_t>(mMaxBatchSize) * mMaxInputLength;
    mFeatureIdxPinned
        = Tensor({maxRows}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "VisualTokenPruner::featureIdxPinned");
    mFeatureIdxDevice
        = Tensor({maxRows}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "VisualTokenPruner::featureIdxDevice");
    mCompactVisualFeatures = Tensor(
        {maxRows, mHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "VisualTokenPruner::compactFeatures");
}

int32_t VisualTokenPruner::pruneForPrefill(
    std::vector<int32_t> const& hostTokenIds, PipelineIO& io, int32_t origLen, cudaStream_t stream)
{
    check::check(static_cast<int32_t>(hostTokenIds.size()) == origLen, "token ids length mismatch");
    check::check(io.inputsEmbeds.getShape()[0] == 1, "pruneForPrefill expects batch size 1");
    check::check(io.inputsEmbeds.getShape()[1] == origLen, "inputsEmbeds length mismatch");

    mSlotKeepLists[0].clear();
    mDeferCompaction = true;
    mCurrentSlot = 0;
    Tensor const embedsView(
        io.inputsEmbeds.rawPointer(), {origLen, mHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    int32_t prunedLen{origLen};
    try
    {
        prunedLen = selectForSlot(hostTokenIds, embedsView, io, origLen, stream);
    }
    catch (...)
    {
        mDeferCompaction = false;
        throw;
    }
    mDeferCompaction = false;
    check::check(prunedLen == origLen || static_cast<int32_t>(mSlotKeepLists[0].size()) == prunedLen,
        std::string(name()) + " must publish subset provenance through compactToKeepList");
    mOldLens.assign(1, origLen);
    mNewLens.assign(1, prunedLen);
    publishKeepMap(mNewLens, 1);
    if (prunedLen < origLen)
    {
        executeBatchCompaction(io, mOldLens, mNewLens, origLen, prunedLen, stream);
    }
    return prunedLen;
}

int32_t VisualTokenPruner::pruneBatchForPrefill(std::vector<std::vector<int32_t>> const& hostTokenIds, PipelineIO& io,
    std::vector<int32_t>& effectiveLens, int32_t maxLen, std::vector<int32_t>& prunedTokensOut, cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(io.inputsEmbeds.getShape()[0]);
    check::check(batch >= 1 && batch <= mMaxBatchSize, "batch size out of range for visual-token pruning");
    check::check(
        static_cast<int32_t>(hostTokenIds.size()) >= batch && static_cast<int32_t>(effectiveLens.size()) >= batch,
        "token ids / effective lengths must cover the batch");
    check::check(io.inputsEmbeds.getShape()[1] == maxLen, "inputsEmbeds length mismatch");

    prunedTokensOut.assign(batch, 0);
    if (batch == 1)
    {
        int32_t const prunedLen = pruneForPrefill(hostTokenIds[0], io, maxLen, stream);
        prunedTokensOut[0] = maxLen - prunedLen;
        effectiveLens[0] = prunedLen;
        return prunedLen;
    }

    // A slot whose token ids don't cover exactly this prefill's effective length (chunked
    // continuation) can't be partitioned by modality — skip the whole batch rather than fail.
    for (int32_t i = 0; i < batch; ++i)
    {
        if (static_cast<int32_t>(hostTokenIds[i].size()) != effectiveLens[i] || effectiveLens[i] > maxLen)
        {
            LOG_WARNING("Visual-token pruning skipped: slot %d token ids do not cover the full prefill.", i);
            return maxLen;
        }
    }

    // Per-slot selection with deferred compaction: prune() sees each slot's contiguous
    // [len, hidden] plane with slot-local positions, and compactToKeepList() records the keep
    // list instead of gathering. The repack to the new row pitch happens once, below.
    for (int32_t i = 0; i < batch; ++i)
    {
        mSlotKeepLists[i].clear(); // capacity reserved in the constructor; no reallocation
    }
    mDeferCompaction = true;
    mNewLens.assign(effectiveLens.begin(), effectiveLens.begin() + batch);
    std::vector<int32_t>& newLens = mNewLens;
    try
    {
        for (int32_t i = 0; i < batch; ++i)
        {
            mCurrentSlot = i;
            int32_t const len = effectiveLens[i];
            void* slotPtr
                = static_cast<half*>(io.inputsEmbeds.rawPointer()) + static_cast<int64_t>(i) * maxLen * mHiddenSize;
            Tensor const slotView(slotPtr, {len, mHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
            int32_t const slotPrunedLen = selectForSlot(hostTokenIds[i], slotView, io, len, stream);
            check::check(slotPrunedLen == len || static_cast<int32_t>(mSlotKeepLists[i].size()) == slotPrunedLen,
                std::string(name()) + " must publish subset provenance through compactToKeepList");
            newLens[i] = slotPrunedLen;
        }
    }
    catch (...)
    {
        mDeferCompaction = false;
        throw;
    }
    mDeferCompaction = false;

    int32_t newMaxLen = 0;
    bool anyPruned = false;
    for (int32_t i = 0; i < batch; ++i)
    {
        newMaxLen = std::max(newMaxLen, newLens[i]);
        anyPruned = anyPruned || newLens[i] < effectiveLens[i];
    }
    publishKeepMap(newLens, batch);
    if (!anyPruned)
    {
        return maxLen;
    }

    mOldLens.assign(effectiveLens.begin(), effectiveLens.begin() + batch);
    executeBatchCompaction(io, mOldLens, newLens, maxLen, newMaxLen, stream);
    for (int32_t i = 0; i < batch; ++i)
    {
        prunedTokensOut[i] = effectiveLens[i] - newLens[i];
        effectiveLens[i] = newLens[i];
    }
    return newMaxLen;
}

void VisualTokenPruner::publishKeepMap(std::vector<int32_t> const& lengths, int32_t batch)
{
    mKeepStartOffsets.clear();
    mConcatenatedKeepIndices.clear();
    mKeepStartOffsets.push_back(0);
    for (int32_t slot = 0; slot < batch; ++slot)
    {
        std::vector<int32_t>& keep = mSlotKeepLists[slot];
        if (keep.empty())
        {
            keep.resize(static_cast<size_t>(lengths[slot]));
            std::iota(keep.begin(), keep.end(), 0);
        }
        mConcatenatedKeepIndices.insert(mConcatenatedKeepIndices.end(), keep.begin(), keep.end());
        mKeepStartOffsets.push_back(static_cast<int32_t>(mConcatenatedKeepIndices.size()));
    }
}

int32_t VisualTokenPruner::selectForSlot(std::vector<int32_t> const& hostTokenIds, Tensor const& embedsView,
    PipelineIO& io, int32_t origLen, cudaStream_t stream)
{
    // Partition positions by modality.
    mImagePositions.clear();
    mTextPositions.clear();
    mImagePositions.reserve(origLen);
    mTextPositions.reserve(origLen);
    for (int32_t i = 0; i < origLen; ++i)
    {
        (hostTokenIds[i] == mImageTokenId ? mImagePositions : mTextPositions).push_back(i);
    }
    int32_t const numVisual = static_cast<int32_t>(mImagePositions.size());
    if (numVisual == 0 || numVisual < mConfig.minVisualTokens)
    {
        return origLen;
    }

    // Split the visual positions into contiguous spans (one per image / video-frame block) and
    // assign each its proportional retention quota, floored at one token. Pruning per span
    // instead of over one global pool guarantees no image is starved by the others.
    mImageSpans.clear();
    int32_t targetImageTokens = 0;
    for (size_t i = 0; i < mImagePositions.size();)
    {
        size_t j = i + 1;
        while (j < mImagePositions.size() && mImagePositions[j] == mImagePositions[j - 1] + 1)
        {
            ++j;
        }
        int32_t const spanLen = static_cast<int32_t>(j - i);
        int32_t target = static_cast<int32_t>(std::ceil(static_cast<double>(spanLen) * (1.0 - mConfig.reductionRatio)));
        target = std::max(1, std::min(spanLen, target));
        mImageSpans.push_back({mImagePositions[i], mImagePositions[j - 1] + 1, target});
        targetImageTokens += target;
        i = j;
    }
    if (targetImageTokens >= numVisual)
    {
        return origLen;
    }

    PruneRequest req;
    req.embeds = &embedsView;
    req.imagePositions = &mImagePositions;
    req.textPositions = &mTextPositions;
    req.imageSpans = &mImageSpans;
    req.targetImageTokens = targetImageTokens;
    req.origLen = origLen;

    int32_t const prunedLen = prune(req, io, stream);
    check::check(prunedLen > 0 && prunedLen <= origLen, "pruner returned an invalid pruned length");
    return prunedLen;
}

int32_t VisualTokenPruner::compactToKeepList(
    PipelineIO& io, std::vector<int32_t> const& retainedImageIndices, PruneRequest const& req, cudaStream_t stream)
{
    int32_t const origLen = req.origLen;
    check::check(
        !retainedImageIndices.empty() && static_cast<int32_t>(retainedImageIndices.size()) <= req.targetImageTokens,
        "pruner returned an invalid retained set");

    // Validate the retained set before it drives device gathers: every index must be one of
    // the request's visual positions (in particular in [0, origLen)) and appear exactly once.
    // Custom pruners are external code — an unchecked bad index would cause out-of-bounds
    // reads in the gather kernels or a corrupted keep list.
    std::vector<char> isRetainable(origLen, 0);
    for (int32_t pos : *req.imagePositions)
    {
        isRetainable[pos] = 1;
    }
    for (int32_t idx : retainedImageIndices)
    {
        check::check(idx >= 0 && idx < origLen && isRetainable[idx] == 1,
            "pruner returned an index that is out of range, duplicated, or not a visual token: " + std::to_string(idx));
        isRetainable[idx] = 0;
    }

    // Final keep list: all text tokens + retained image tokens, original order.
    std::vector<int32_t> const& textPositions = *req.textPositions;
    mKeepIndicesHost.clear();
    mKeepIndicesHost.reserve(textPositions.size() + retainedImageIndices.size());
    mKeepIndicesHost.insert(mKeepIndicesHost.end(), textPositions.begin(), textPositions.end());
    mKeepIndicesHost.insert(mKeepIndicesHost.end(), retainedImageIndices.begin(), retainedImageIndices.end());
    std::sort(mKeepIndicesHost.begin(), mKeepIndicesHost.end());
    int32_t const prunedLen = static_cast<int32_t>(mKeepIndicesHost.size());
    if (prunedLen >= origLen)
    {
        return origLen;
    }
    // Record the slot's keep list — the batched flow consumes it in executeBatchCompaction(),
    // and compactAuxiliaryInputs() (spec-decode draft re-embedding) reads it in both flows.
    // assign() reuses the constructor-reserved capacity, keeping this allocation-free.
    mSlotKeepLists[mCurrentSlot].assign(mKeepIndicesHost.begin(), mKeepIndicesHost.end());
    if (mDeferCompaction)
    {
        // Batched flow: executeBatchCompaction() performs the gathers for all slots at once
        // after every slot has been selected.
        return prunedLen;
    }
    int32_t const numPruned = origLen - prunedLen;

    // Rope-extended index list: rows [0, prunedLen) gather the kept positions; rows
    // [prunedLen, cap - numPruned) shift the original continuation rows [origLen, cap) down so
    // decode reads the positions right after the unpruned sequence (matching the HF reference,
    // where generation continues at maxPos + 1).
    int32_t const ropeRows = mMaxKVCacheCapacity - numPruned;
    int32_t* keepHost = mKeepIdxHost.dataPointer<int32_t>();
    std::copy(mKeepIndicesHost.begin(), mKeepIndicesHost.end(), keepHost);
    for (int32_t i = prunedLen; i < ropeRows; ++i)
    {
        keepHost[i] = origLen + (i - prunedLen);
    }
    CUDA_CHECK(cudaMemcpyAsync(mKeepIdxDevice.rawPointer(), keepHost, static_cast<size_t>(ropeRows) * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    int32_t const* keepIdxDevice = mKeepIdxDevice.dataPointer<int32_t>();

    // Compact embeddings, deepstack planes, and rope rows (gather out of place into scratch,
    // then copy back — the destination overlaps the source).
    int64_t const embedRowBytes = static_cast<int64_t>(mHiddenSize) * sizeof(half);
    kernel::gatherRows(
        mGatherScratch.rawPointer(), io.inputsEmbeds.rawPointer(), keepIdxDevice, prunedLen, embedRowBytes, stream);
    CUDA_CHECK(cudaMemcpyAsync(io.inputsEmbeds.rawPointer(), mGatherScratch.rawPointer(),
        static_cast<size_t>(prunedLen) * embedRowBytes, cudaMemcpyDeviceToDevice, stream));
    check::check(io.inputsEmbeds.reshape({1, prunedLen, mHiddenSize}), "Tensor reshape failed");

    for (Tensor& deepstack : io.deepstackEmbeds)
    {
        check::check(deepstack.getShape()[1] == origLen, "deepstackEmbeds length mismatch");
        kernel::gatherRows(
            mGatherScratch.rawPointer(), deepstack.rawPointer(), keepIdxDevice, prunedLen, embedRowBytes, stream);
        CUDA_CHECK(cudaMemcpyAsync(deepstack.rawPointer(), mGatherScratch.rawPointer(),
            static_cast<size_t>(prunedLen) * embedRowBytes, cudaMemcpyDeviceToDevice, stream));
        check::check(deepstack.reshape({1, prunedLen, mHiddenSize}), "Tensor reshape failed");
    }

    if (!io.mropeActiveCosSin.isEmpty())
    {
        check::check(io.mropeActiveCosSin.getShape()[1] == mMaxKVCacheCapacity, "mropeActiveCosSin capacity mismatch");
        int64_t const ropeRowBytes = static_cast<int64_t>(mRotaryDim) * sizeof(float);
        kernel::gatherRows(mGatherScratch.rawPointer(), io.mropeActiveCosSin.rawPointer(), keepIdxDevice, ropeRows,
            ropeRowBytes, stream);
        CUDA_CHECK(cudaMemcpyAsync(io.mropeActiveCosSin.rawPointer(), mGatherScratch.rawPointer(),
            static_cast<size_t>(ropeRows) * ropeRowBytes, cudaMemcpyDeviceToDevice, stream));
        // Rows [ropeRows, cap) are stale but unreachable: generation length was clamped against
        // the unpruned sequence, so the last used slot is < prunedLen + maxGenerate <= cap - numPruned.
    }

    return prunedLen;
}

void VisualTokenPruner::executeBatchCompaction(PipelineIO& io, std::vector<int32_t> const& oldLens,
    std::vector<int32_t> const& newLens, int32_t oldMaxLen, int32_t newMaxLen, cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(oldLens.size());
    int64_t const embedRowBytes = static_cast<int64_t>(mHiddenSize) * sizeof(half);
    int64_t const ropeRowBytes = static_cast<int64_t>(mRotaryDim) * sizeof(float);
    bool const hasRope = !io.mropeActiveCosSin.isEmpty();
    if (hasRope)
    {
        check::check(io.mropeActiveCosSin.getShape()[1] == mMaxKVCacheCapacity, "mropeActiveCosSin capacity mismatch");
    }
    for (Tensor const& deepstack : io.deepstackEmbeds)
    {
        check::check(
            deepstack.getShape()[0] == batch && deepstack.getShape()[1] == oldMaxLen, "deepstackEmbeds shape mismatch");
    }

    // Slots are repacked in ascending order so writes at the new (smaller) row pitch never
    // touch a not-yet-consumed slot's source rows: slot i's destination ends at
    // (i + 1) * newMaxLen <= (i + 1) * oldMaxLen, the start of slot i + 1's source plane.
    for (int32_t i = 0; i < batch; ++i)
    {
        int32_t const oldLen = oldLens[i];
        int32_t const newLen = newLens[i];
        int32_t const numPruned = oldLen - newLen;
        // Skip slots with nothing to do: empty, or unpruned and already at their final offset
        // (slot 0 always is; every unpruned slot is when the row pitch does not change).
        if (newLen == 0 || (numPruned == 0 && (i == 0 || newMaxLen == oldMaxLen)))
        {
            continue;
        }

        // Per-slot pinned/device index regions: reusing one region across slots would let the
        // CPU overwrite an earlier slot's list before its async H2D copy has executed.
        int32_t* keepHost = mKeepIdxHost.dataPointer<int32_t>() + static_cast<int64_t>(i) * mMaxKVCacheCapacity;
        int32_t* keepDevice = mKeepIdxDevice.dataPointer<int32_t>() + static_cast<int64_t>(i) * mMaxKVCacheCapacity;
        int32_t uploadRows = newLen;
        if (numPruned > 0)
        {
            std::vector<int32_t> const& keepList = mSlotKeepLists[i];
            std::copy(keepList.begin(), keepList.end(), keepHost);
            // Rope-extended rows: shift the continuation positions [oldLen, cap) down so decode
            // reads the positions right after the unpruned sequence (see compactToKeepList).
            int32_t const ropeRows = mMaxKVCacheCapacity - numPruned;
            for (int32_t r = newLen; r < ropeRows; ++r)
            {
                keepHost[r] = oldLen + (r - newLen);
            }
            uploadRows = ropeRows;
        }
        else
        {
            std::iota(keepHost, keepHost + newLen, 0); // identity: the slot only moves pitch
        }
        CUDA_CHECK(cudaMemcpyAsync(
            keepDevice, keepHost, static_cast<size_t>(uploadRows) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        auto const repackEmbedPlane = [&](Tensor& tensor) {
            auto* base = static_cast<uint8_t*>(tensor.rawPointer());
            uint8_t const* src = base + static_cast<int64_t>(i) * oldMaxLen * embedRowBytes;
            uint8_t* dst = base + static_cast<int64_t>(i) * newMaxLen * embedRowBytes;
            kernel::gatherRows(mGatherScratch.rawPointer(), src, keepDevice, newLen, embedRowBytes, stream);
            CUDA_CHECK(cudaMemcpyAsync(dst, mGatherScratch.rawPointer(), static_cast<size_t>(newLen) * embedRowBytes,
                cudaMemcpyDeviceToDevice, stream));
        };
        repackEmbedPlane(io.inputsEmbeds);
        for (Tensor& deepstack : io.deepstackEmbeds)
        {
            repackEmbedPlane(deepstack);
        }

        if (numPruned > 0 && hasRope)
        {
            // Rope pitch (KV-cache capacity) is unchanged; the gather is in-plane.
            auto* plane = static_cast<uint8_t*>(io.mropeActiveCosSin.rawPointer())
                + static_cast<int64_t>(i) * mMaxKVCacheCapacity * ropeRowBytes;
            int32_t const ropeRows = mMaxKVCacheCapacity - numPruned;
            kernel::gatherRows(mGatherScratch.rawPointer(), plane, keepDevice, ropeRows, ropeRowBytes, stream);
            CUDA_CHECK(cudaMemcpyAsync(plane, mGatherScratch.rawPointer(), static_cast<size_t>(ropeRows) * ropeRowBytes,
                cudaMemcpyDeviceToDevice, stream));
        }
    }

    check::check(io.inputsEmbeds.reshape({batch, newMaxLen, mHiddenSize}), "Tensor reshape failed");
    for (Tensor& deepstack : io.deepstackEmbeds)
    {
        check::check(deepstack.reshape({batch, newMaxLen, mHiddenSize}), "Tensor reshape failed");
    }
}

void VisualTokenPruner::compactAuxiliaryInputs(std::vector<std::vector<int32_t>>& hostTokenIds, int32_t batch,
    std::vector<int32_t> const& prunedTokens, OptionalInputTensor& visualFeatures, cudaStream_t stream)
{
    check::check(batch >= 1 && batch <= mMaxBatchSize && static_cast<int32_t>(hostTokenIds.size()) >= batch
            && static_cast<int32_t>(prunedTokens.size()) >= batch,
        "compactAuxiliaryInputs batch mismatch");
    if (std::none_of(prunedTokens.begin(), prunedTokens.begin() + batch, [](int32_t n) { return n > 0; }))
    {
        return;
    }

    // Pass 1: per slot, compact the host token ids in place and collect the kept feature
    // ordinals. Feature rows are indexed by the running image-token count over the packed
    // batch grid, so ordinals are offset by the preceding slots' original visual counts.
    // Ordinal 0 == the batch's first image token only when embedding runs with zero
    // multimodal base offsets; the fresh-KV-cache gate on pruning guarantees that (prefix
    // reuse would start the count at a nonzero base offset).
    mFeatureKeepHost.clear();
    int32_t slotBase = 0;
    for (int32_t i = 0; i < batch; ++i)
    {
        std::vector<int32_t>& ids = hostTokenIds[i];
        std::vector<int32_t> const& keep = mSlotKeepLists[i];
        int32_t const len = static_cast<int32_t>(ids.size());
        if (prunedTokens[i] == 0)
        {
            // Unpruned slot: all of its feature rows stay, in order.
            for (int32_t pos = 0; pos < len; ++pos)
            {
                if (ids[pos] == mImageTokenId)
                {
                    mFeatureKeepHost.push_back(slotBase);
                    ++slotBase;
                }
            }
            continue;
        }
        // A pruned slot without a recorded keep list means the algorithm rewrote the buffers
        // directly (merge-style) — its token-id/feature mapping is unknowable here.
        check::check(static_cast<int32_t>(keep.size()) == len - prunedTokens[i],
            std::string(name()) + " pruned a request without recording a keep list; "
                                  "auxiliary-input compaction (spec decode) is unavailable");
        size_t k = 0;
        int32_t ordinal = 0;
        for (int32_t pos = 0; pos < len; ++pos)
        {
            bool const isVisual = ids[pos] == mImageTokenId;
            if (k < keep.size() && pos == keep[k])
            {
                if (isVisual)
                {
                    mFeatureKeepHost.push_back(slotBase + ordinal);
                }
                ids[k] = ids[pos]; // keep is ascending, so k <= pos: safe in place
                ++k;
            }
            ordinal += isVisual;
        }
        ids.resize(keep.size());
        slotBase += ordinal;
    }

    int64_t const totalOrig = slotBase;
    int64_t const totalKept = static_cast<int64_t>(mFeatureKeepHost.size());
    if (totalKept == totalOrig || !visualFeatures.has_value())
    {
        return;
    }

    // Buffers grow lazily to the request's raw feature-row count when preallocateAuxiliaryBuffers()
    // was not called (once per request, not per step).
    if (mFeatureIdxPinned.isEmpty() || mFeatureIdxPinned.getShape().volume() < totalOrig)
    {
        mFeatureIdxPinned
            = Tensor({totalOrig}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "VisualTokenPruner::featureIdxPinned");
        mFeatureIdxDevice
            = Tensor({totalOrig}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "VisualTokenPruner::featureIdxDevice");
    }
    std::copy(mFeatureKeepHost.begin(), mFeatureKeepHost.end(), mFeatureIdxPinned.dataPointer<int32_t>());
    CUDA_CHECK(cudaMemcpyAsync(mFeatureIdxDevice.rawPointer(), mFeatureIdxPinned.rawPointer(),
        static_cast<size_t>(totalKept) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    auto const compactFeaturePlane = [&](Tensor const& src, Tensor& owned) {
        auto const& shape = src.getShape();
        // The ordinal math above requires exactly one feature row per image token in the
        // request (Qwen-VL packing) — anything else would desync rows from placeholders.
        check::check(shape.getNumDims() == 2 && shape[0] == totalOrig,
            "feature rows do not match the request's visual token count");
        int64_t const rowBytes = shape[1] * static_cast<int64_t>(utils::getTypeSize(src.getDataType()));
        if (owned.isEmpty() || owned.getDataType() != src.getDataType() || owned.getShape()[1] != shape[1]
            || static_cast<int64_t>(owned.getMemoryCapacity()) < totalOrig * rowBytes)
        {
            owned = Tensor(
                {totalOrig, shape[1]}, DeviceType::kGPU, src.getDataType(), "VisualTokenPruner::compactFeatures");
        }
        kernel::gatherRows(owned.rawPointer(), src.rawPointer(), mFeatureIdxDevice.dataPointer<int32_t>(), totalKept,
            rowBytes, stream);
        check::check(owned.reshape({totalKept, shape[1]}), "Tensor reshape failed");
    };

    // Deepstack feature rows are deliberately NOT compacted: no draft strategy consumes them
    // (draft prefill embeds without deepstack and every decoder zero-fills its deepstack
    // target), and the base's deepstack planes were already compacted in PipelineIO. A future
    // draft that re-assembles deepstack from raw features would need the same gather here.
    compactFeaturePlane(visualFeatures->get(), mCompactVisualFeatures);
    visualFeatures = std::cref(mCompactVisualFeatures);
}

// ---------------------------------------------------------------------------
// Built-in pruners.
// ---------------------------------------------------------------------------

namespace
{

std::map<std::string, VisualPrunerFactory>& prunerRegistry()
{
    static std::map<std::string, VisualPrunerFactory> registry = {
        {"dart",
            [](VisualPrunerConfig const& cfg, LLMEngineConfig const& engineCfg) {
                return std::unique_ptr<VisualTokenPruner>(std::make_unique<DartPruner>(cfg, engineCfg));
            }},
    };
    return registry;
}

std::mutex& prunerRegistryMutex()
{
    static std::mutex m;
    return m;
}

} // namespace

// ---------------------------------------------------------------------------
// Registry.
// ---------------------------------------------------------------------------

void registerVisualPruner(std::string const& name, VisualPrunerFactory factory)
{
    std::lock_guard<std::mutex> lock(prunerRegistryMutex());
    prunerRegistry()[name] = std::move(factory);
}

std::unique_ptr<VisualTokenPruner> createVisualTokenPruner(
    VisualPrunerConfig const& config, LLMEngineConfig const& engineConfig)
{
    VisualPrunerFactory factory;
    {
        std::lock_guard<std::mutex> lock(prunerRegistryMutex());
        auto const it = prunerRegistry().find(config.algorithm);
        if (it == prunerRegistry().end())
        {
            std::ostringstream known;
            for (auto const& [name, unused] : prunerRegistry())
            {
                known << (known.tellp() > 0 ? ", " : "") << name;
            }
            throw std::runtime_error(
                "Unknown visual-token prune algorithm '" + config.algorithm + "'. Registered: " + known.str());
        }
        factory = it->second;
    }
    return factory(config, engineConfig);
}

std::vector<std::string> registeredVisualPrunerNames()
{
    std::lock_guard<std::mutex> lock(prunerRegistryMutex());
    std::vector<std::string> names;
    names.reserve(prunerRegistry().size());
    for (auto const& [name, unused] : prunerRegistry())
    {
        names.push_back(name);
    }
    return names;
}

} // namespace rt
} // namespace trt_edgellm
