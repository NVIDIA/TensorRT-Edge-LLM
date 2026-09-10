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
#include "kernels/dart/dartGatherKernels.h"
#include "runtime/preprocess/dartPruner.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
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
{
    check::check(mConfig.reductionRatio > 0.0F && mConfig.reductionRatio < 1.0F, "reductionRatio must be in (0, 1)");
    check::check(mImageTokenId >= 0, "visual-token pruning requires a VLM engine with an image token id");

    int32_t const maxInputLen = engineConfig.maxSupportedInputLength;
    mKeepIdxDevice = Tensor(
        {mMaxKVCacheCapacity}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "VisualTokenPruner::keepIdxDevice");
    mKeepIdxHost
        = Tensor({mMaxKVCacheCapacity}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "VisualTokenPruner::keepIdxHost");
    int64_t const embedPlaneBytes = static_cast<int64_t>(maxInputLen) * mHiddenSize * sizeof(half);
    int64_t const ropePlaneBytes = static_cast<int64_t>(mMaxKVCacheCapacity) * mRotaryDim * sizeof(float);
    mGatherScratch = Tensor({std::max(embedPlaneBytes, ropePlaneBytes)}, DeviceType::kGPU, nvinfer1::DataType::kINT8,
        "VisualTokenPruner::gatherScratch");
}

int32_t VisualTokenPruner::pruneForPrefill(
    std::vector<int32_t> const& hostTokenIds, PipelineIO& io, int32_t origLen, cudaStream_t stream)
{
    check::check(static_cast<int32_t>(hostTokenIds.size()) == origLen, "token ids length mismatch");
    check::check(io.inputsEmbeds.getShape()[0] == 1, "visual-token pruning supports batch size 1 only");
    check::check(io.inputsEmbeds.getShape()[1] == origLen, "inputsEmbeds length mismatch");

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

    Tensor const embedsView(
        io.inputsEmbeds.rawPointer(), {origLen, mHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
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

    if (!io.mropeCosSin.isEmpty())
    {
        check::check(io.mropeCosSin.getShape()[1] == mMaxKVCacheCapacity, "mropeCosSin capacity mismatch");
        int64_t const ropeRowBytes = static_cast<int64_t>(mRotaryDim) * sizeof(float);
        kernel::gatherRows(
            mGatherScratch.rawPointer(), io.mropeCosSin.rawPointer(), keepIdxDevice, ropeRows, ropeRowBytes, stream);
        CUDA_CHECK(cudaMemcpyAsync(io.mropeCosSin.rawPointer(), mGatherScratch.rawPointer(),
            static_cast<size_t>(ropeRows) * ropeRowBytes, cudaMemcpyDeviceToDevice, stream));
        // Rows [ropeRows, cap) are stale but unreachable: generation length was clamped against
        // the unpruned sequence, so the last used slot is < prunedLen + maxGenerate <= cap - numPruned.
    }

    return prunedLen;
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
