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

#include "runtime/debug/layerDebugger.h"
#include "common/pagedKvTypes.h"
#include "runtime/state/kvPageTable.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{

namespace
{
constexpr char const* kLayersEnv = "EDGELLM_DUMP_LOGITS_KVCACHE_LAYERS";
constexpr char const* kDirEnv = "EDGELLM_DUMP_LOGITS_KVCACHE_DIR";

//! Trim ASCII whitespace from both ends.
std::string trim(std::string const& s)
{
    size_t const b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos)
    {
        return "";
    }
    size_t const e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

//! Parse the layer count "k" (a single positive integer) into the leading-layer set {0..k-1}.
//! The only producer (few-layer-validation.sh) dumps the first N decoder layers; selecting
//! arbitrary / non-leading layers is intentionally unsupported -- dump the prefix and pick the
//! layers of interest at analysis time (disk is cheap; a simpler parser is worth more).
std::set<int32_t> parseLeadingLayers(std::string const& spec)
{
    int32_t const k = std::stoi(trim(spec));
    if (k < 1)
    {
        throw std::runtime_error(std::string(kLayersEnv) + ": expected a positive layer count, got '" + spec + "'.");
    }
    std::set<int32_t> out;
    for (int32_t i = 0; i < k; ++i)
    {
        out.insert(i);
    }
    return out;
}

//! Gather persistent rows into execution order for a fixed-size recurrent or convolution state.
Tensor copyResidentRows(Tensor const& src, std::vector<ResidentRef> const& residents, int32_t activeBatchSize,
    std::string const& name, cudaStream_t stream)
{
    Coords const s = src.getShape();
    ELLM_CHECK(static_cast<int32_t>(residents.size()) >= activeBatchSize,
        "LayerDebugger: resident mapping is smaller than the active batch");
    std::vector<int64_t> dims;
    dims.reserve(s.getNumDims());
    dims.push_back(activeBatchSize);
    int64_t inner = 1;
    for (int32_t d = 1; d < s.getNumDims(); ++d)
    {
        dims.push_back(s[d]);
        inner *= s[d];
    }
    nvinfer1::DataType const dtype = src.getDataType();
    Tensor host(Coords(dims), DeviceType::kCPU, dtype, name);
    size_t const rowBytes = static_cast<size_t>(inner) * utils::getTypeSize(dtype);
    for (int32_t row = 0; row < activeBatchSize; ++row)
    {
        int32_t const slot = residents[static_cast<size_t>(row)].slot;
        ELLM_CHECK(slot >= 0 && slot < s[0], "LayerDebugger: resident state slot is out of range");
        auto const* source = static_cast<char const*>(src.rawPointer()) + static_cast<size_t>(slot) * rowBytes;
        auto* destination = static_cast<char*>(host.rawPointer()) + static_cast<size_t>(row) * rowBytes;
        CUDA_CHECK(cudaMemcpyAsync(destination, source, rowBytes, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host;
}

//! Gather an attention layer's resident rows into a host tensor shaped
//! [activeBatch, 2, kvHeads, capPadded, headDim].
//!
//! Full and reduced SWA layers use distinct physical pools and page tables. Both are gathered page
//! by page and transposed [seq, head, dim] -> [head, seq, dim].
Tensor copyKVCacheAsHND(HybridCacheManager& cacheManager, KVPageTable const& fullPageTable,
    KVPageTable const* swaPageTable, int32_t layer, std::vector<ResidentRef> const& residents, int32_t activeBatchSize,
    std::string const& name, cudaStream_t stream)
{
    KVLayerStorageMetadata const storage = cacheManager.getKVLayerStorageMetadata(layer);
    KVPageTable const* pageTable = &fullPageTable;
    if (storage.kind == KVCacheStorageKind::kReducedSwa)
    {
        ELLM_CHECK(swaPageTable != nullptr, "LayerDebugger: reduced SWA layer has no sparse page table");
        pageTable = swaPageTable;
    }
    ELLM_CHECK(pageTable->maxBatch() == cacheManager.getKVCacheManager().getConfig().maxBatchSize,
        "LayerDebugger: page-table resident capacity does not match the KV pool");
    ELLM_CHECK(pageTable->maxPagesPerSeq() == storage.logicalPagesPerSequence,
        "LayerDebugger: page-table logical width does not match the KV pool contract");
    ELLM_CHECK(pageTable->numPages() == storage.physicalPages,
        "LayerDebugger: page-table physical namespace does not match the KV pool");

    Tensor const& pool = cacheManager.getCombinedKVCache(layer);
    Coords const poolShape = pool.getShape();
    ELLM_CHECK(poolShape.getNumDims() == 5 && poolShape[0] == 2 && poolShape[1] == storage.physicalPages
            && poolShape[2] == kTOKENS_PER_PAGE && poolShape[3] == storage.numKVHeads
            && poolShape[4] == storage.headDim,
        "LayerDebugger: KV pool shape does not match its storage metadata");
    int64_t const cap = static_cast<int64_t>(storage.logicalPagesPerSequence) * kTOKENS_PER_PAGE;
    int64_t const heads = storage.numKVHeads;
    int64_t const dim = storage.headDim;
    nvinfer1::DataType const dtype = pool.getDataType();
    size_t const elemSize = utils::getTypeSize(dtype);
    size_t const slotBytes = static_cast<size_t>(cap) * heads * dim * elemSize;
    size_t const rowBytes = static_cast<size_t>(dim) * elemSize;
    size_t const pageBytes = static_cast<size_t>(kTOKENS_PER_PAGE) * heads * dim * elemSize;

    int32_t const maxPagesPerSeq = pageTable->maxPagesPerSeq();
    size_t const poolBytes = static_cast<size_t>(storage.physicalPages) * pageBytes;
    ELLM_CHECK(static_cast<int32_t>(residents.size()) >= activeBatchSize,
        "LayerDebugger: resident mapping is smaller than the active batch");
    Tensor staging(
        {static_cast<int64_t>(2 * poolBytes)}, DeviceType::kCPU, nvinfer1::DataType::kINT8, name + ".staging");
    std::byte* const kStage = static_cast<std::byte*>(staging.rawPointer());
    std::byte* const vStage = kStage + poolBytes;
    auto const* poolBase = static_cast<std::byte const*>(pool.rawPointer());
    CUDA_CHECK(cudaMemcpyAsync(kStage, poolBase, poolBytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(vStage, poolBase + poolBytes, poolBytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Tensor host({activeBatchSize, 2, heads, cap, dim}, DeviceType::kCPU, dtype, name);
    std::byte* const out = static_cast<std::byte*>(host.rawPointer());
    std::memset(out, 0, static_cast<size_t>(activeBatchSize) * 2 * slotBytes);
    for (int64_t b = 0; b < activeBatchSize; ++b)
    {
        int32_t const slot = residents[static_cast<size_t>(b)].slot;
        ELLM_CHECK(slot >= 0 && slot < pageTable->maxBatch(), "LayerDebugger: resident KV slot is out of range");
        int32_t const* const pages = pageTable->hostRow(slot);
        for (int32_t lp = 0; lp < maxPagesPerSeq; ++lp)
        {
            int32_t const page = pages[lp];
            if (page == kUNUSED_PAGE_ENTRY)
            {
                continue; // unused logical page; leave it zeroed
            }
            ELLM_CHECK(page >= 0 && page < storage.physicalPages, "LayerDebugger: KV page exceeds the physical pool");
            ELLM_CHECK(static_cast<int64_t>(lp + 1) * kTOKENS_PER_PAGE <= cap,
                "LayerDebugger: logical KV page exceeds the dump shape");
            for (int64_t kv = 0; kv < 2; ++kv)
            {
                std::byte const* const src = (kv == 0 ? kStage : vStage) + static_cast<size_t>(page) * pageBytes;
                std::byte* const slotDst = out + (b * 2 + kv) * slotBytes;
                for (int64_t t = 0; t < kTOKENS_PER_PAGE; ++t)
                {
                    int64_t const token = static_cast<int64_t>(lp) * kTOKENS_PER_PAGE + t;
                    for (int64_t h = 0; h < heads; ++h)
                    {
                        std::memcpy(slotDst + (h * cap + token) * rowBytes, src + (t * heads + h) * rowBytes, rowBytes);
                    }
                }
            }
        }
    }
    return host;
}
} // namespace

LayerDebugger::LayerDebugger(std::set<int32_t> layers, std::string dir, std::vector<std::vector<int32_t>> forcedTokens)
    : mLayers(std::move(layers))
    , mDir(std::move(dir))
    , mForcedTokens(std::move(forcedTokens))
{
    static std::atomic<int32_t> sRequestCounter{0};
    mRequestIdx = sRequestCounter++;
}

std::unique_ptr<LayerDebugger> LayerDebugger::fromEnv()
{
    char const* layersEnv = std::getenv(kLayersEnv);
    char const* dirEnv = std::getenv(kDirEnv);
    bool const hasLayers = layersEnv != nullptr && *layersEnv != '\0';
    bool const hasDir = dirEnv != nullptr && *dirEnv != '\0';

    if (!hasLayers && !hasDir)
    {
        return nullptr;
    }
    if (hasLayers != hasDir)
    {
        throw std::runtime_error(std::string(kLayersEnv) + " and " + kDirEnv
            + " must be set together (set both to enable layer dumping, or neither to disable).");
    }

    std::set<int32_t> layers = parseLeadingLayers(layersEnv);
    std::filesystem::create_directories(dirEnv);
    std::vector<std::vector<int32_t>> forcedTokens = readForcedTokensFromEnv();
    LOG_INFO("LayerDebugger enabled: %zu layer(s) -> %s", layers.size(), dirEnv);
    return std::unique_ptr<LayerDebugger>(new LayerDebugger(std::move(layers), dirEnv, std::move(forcedTokens)));
}

void LayerDebugger::dumpRound(HybridCacheManager& cacheManager, KVPageTable const& pageTable,
    KVPageTable const* swaPageTable, Tensor const& logits, std::vector<int32_t> const& validLengths,
    std::vector<int32_t> const& originalIndices, std::vector<ResidentRef> const& residentRefs,
    int32_t const* generatedTokenIds, int32_t activeBatchSize, cudaStream_t stream)
{
    // The KV cache / logits are produced asynchronously on this stream; synchronise
    // so the device-side data is final before we copy it out.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int32_t const r = mRound;
    std::string const prefix = "round_" + std::to_string(r) + ".";

    // ---- logits: [activeBatch, vocab] (last-token only) ----
    {
        Coords const shape = logits.getShape();
        int64_t const vocab = shape[shape.getNumDims() - 1];
        nvinfer1::DataType const dtype = logits.getDataType();
        Tensor logitsHost({activeBatchSize, vocab}, DeviceType::kCPU, dtype, prefix + "logits");
        size_t const bytes = static_cast<size_t>(activeBatchSize) * vocab * utils::getTypeSize(dtype);
        CUDA_CHECK(cudaMemcpy(logitsHost.rawPointer(), logits.rawPointer(), bytes, cudaMemcpyDeviceToHost));
        mTensors.push_back(std::move(logitsHost));
    }

    // ---- per-layer state, by layer type ----
    // Resident rows are gathered into execution order. Sequence-length truncation is deferred to
    // the comparison tool, which uses the dumped context_lengths.
    //   Attention layers   -> full KV cache [activeBatch, 2, kvHeads, capPadded, headDim].
    //   Mamba / Gated-DeltaNet -> fixed-size recurrent + conv state (no sequence dim).
    int32_t const numLayers = cacheManager.numLayers();
    for (int32_t layer : mLayers)
    {
        if (layer < 0 || layer >= numLayers)
        {
            LOG_WARNING("LayerDebugger: layer %d out of range (model has %d layers); skipping.", layer, numLayers);
            continue;
        }
        std::string const lp = prefix + "layer_" + std::to_string(layer) + ".";
        if (cacheManager.getLayerType(layer) != HybridCacheManager::LayerType::kAttention)
        {
            // Mamba / Gated DeltaNet: fixed-size recurrent state [B, heads, headDim, stateSize] and
            // conv state [B, convDim, convKernel].
            mTensors.push_back(copyResidentRows(
                cacheManager.getRecurrentState(layer), residentRefs, activeBatchSize, lp + "recurrent_state", stream));
            mTensors.push_back(copyResidentRows(
                cacheManager.getConvState(layer), residentRefs, activeBatchSize, lp + "conv_state", stream));
            continue;
        }
        // Attention: KV cache dumped as [activeBatch, 2, kvHeads, capPadded, headDim] (transposed
        // out of the NHD pool storage; capPadded >= maxSeqLen, the comparison tool slices).
        mTensors.push_back(copyKVCacheAsHND(
            cacheManager, pageTable, swaPageTable, layer, residentRefs, activeBatchSize, lp + "kv", stream));
    }

    // ---- per-sequence valid lengths ----
    {
        Tensor ctxLenHost({static_cast<int64_t>(activeBatchSize)}, DeviceType::kCPU, nvinfer1::DataType::kINT32,
            prefix + "context_lengths");
        int32_t* p = ctxLenHost.dataPointer<int32_t>();
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            int32_t const row = originalRow(originalIndices, i);
            p[i] = validLengths.at(i) + (row < static_cast<int32_t>(mReusedPrefix.size()) ? mReusedPrefix[row] : 0);
        }
        mTensors.push_back(std::move(ctxLenHost));
    }

    // ---- per-round generated token ids (for cross-side greedy divergence detection) ----
    if (generatedTokenIds != nullptr)
    {
        Tensor genTokHost({static_cast<int64_t>(activeBatchSize)}, DeviceType::kCPU, nvinfer1::DataType::kINT32,
            prefix + "generated_token_ids");
        std::copy(generatedTokenIds, generatedTokenIds + activeBatchSize, genTokHost.dataPointer<int32_t>());
        mTensors.push_back(std::move(genTokHost));
    }

    ++mRound;
}

void LayerDebugger::flush(cudaStream_t stream)
{
    if (mTensors.empty())
    {
        LOG_WARNING("LayerDebugger: no rounds were dumped; nothing to write.");
        return;
    }
    std::string const stem = mRequestIdx == 0 ? "edgellm_dump" : "edgellm_dump_" + std::to_string(mRequestIdx);
    std::filesystem::path const path = std::filesystem::path(mDir) / (stem + ".safetensors");
    if (safetensors::saveSafetensors(path, mTensors, stream))
    {
        LOG_INFO("LayerDebugger: wrote %zu tensors across %d round(s) to %s", mTensors.size(), mRound,
            path.string().c_str());
    }
    else
    {
        LOG_ERROR("LayerDebugger: failed to write %s", path.string().c_str());
    }
}

std::vector<std::vector<int32_t>> LayerDebugger::readForcedTokensFromEnv()
{
    char const* path = std::getenv("EDGELLM_FORCE_TOKENS_FILE");
    if (path == nullptr || *path == '\0')
    {
        return {};
    }
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error(std::string("EDGELLM_FORCE_TOKENS_FILE could not be opened: ") + path);
    }
    std::vector<std::vector<int32_t>> forced;
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        std::vector<int32_t> row;
        int32_t tok;
        while (ss >> tok)
        {
            row.push_back(tok);
        }
        forced.push_back(std::move(row));
    }
    LOG_WARNING(
        "Teacher forcing ON: the decode loop will OVERRIDE its own sampled tokens with the "
        "forced tokens for %zu sequence(s) from %s (numeric-validation debug; not for production).",
        forced.size(), path);
    return forced;
}

int32_t LayerDebugger::originalRow(std::vector<int32_t> const& originalIndices, int32_t slot)
{
    // The runtime compacts its per-slot vectors when a sequence finishes, so active slot `slot`
    // stops being request row `slot` from that point on. Everything this class keys by sequence
    // is keyed by the original row instead. An empty mapping means the caller has none to give,
    // in which case the two still coincide.
    return slot < static_cast<int32_t>(originalIndices.size()) ? originalIndices[slot] : slot;
}

int32_t LayerDebugger::forcedRowBase(int32_t activeBatchSize)
{
    // Rows are handed out across the whole run in request order: one request per shared-prefix
    // prompt is how the context-reuse validation drives the runtime, and its golden batches the
    // same prompts into consecutive rows.
    static std::atomic<int32_t> sSequenceCounter{0};
    if (mForcedRowBase < 0)
    {
        mForcedRowBase = sSequenceCounter.fetch_add(activeBatchSize);
    }
    return mForcedRowBase;
}

void LayerDebugger::applyForcedTokens(std::vector<int32_t> const& genLengths,
    std::vector<int32_t> const& originalIndices, int32_t* tokenIds, int32_t activeBatchSize)
{
    if (mForcedTokens.empty())
    {
        return;
    }
    int32_t const rowBase = forcedRowBase(activeBatchSize);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // The token produced this step is generated-token index genLengths[i] (0 at prefill).
        int32_t const idx = genLengths.at(i);
        int32_t const row = rowBase + originalRow(originalIndices, i);
        if (row < static_cast<int32_t>(mForcedTokens.size()) && idx >= 0
            && idx < static_cast<int32_t>(mForcedTokens[row].size()))
        {
            tokenIds[i] = mForcedTokens.at(row).at(idx);
        }
    }
}

void LayerDebugger::setReusedPrefixLengths(std::vector<int32_t> lengths)
{
    mReusedPrefix = std::move(lengths);
}

bool LayerDebugger::applyForcedAcceptance(std::vector<int32_t> const& genLengths,
    std::vector<int32_t> const& originalIndices, int32_t* acceptLengths, int32_t* acceptedTokenIds,
    std::vector<int32_t>& ownTokens, int32_t activeBatchSize, int32_t maxAcceptDepth)
{
    ownTokens.assign(activeBatchSize, -1);
    if (mForcedTokens.empty())
    {
        return false;
    }
    bool trimmed = false;
    int32_t const rowBase = forcedRowBase(activeBatchSize);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        int32_t const row = rowBase + originalRow(originalIndices, i);
        if (row >= static_cast<int32_t>(mForcedTokens.size()))
        {
            continue;
        }
        std::vector<int32_t> const& forced = mForcedTokens[row];
        int32_t const genLen = genLengths.at(i);
        int32_t const accepted = acceptLengths[i];
        if (accepted > 0)
        {
            ownTokens[i] = acceptedTokenIds[i * maxAcceptDepth + accepted - 1];
        }
        for (int32_t j = 0; j < accepted; ++j)
        {
            int32_t const idx = genLen + j;
            if (idx < 0 || idx >= static_cast<int32_t>(forced.size()))
            {
                break; // past the end of the golden's sequence -- leave the rest as sampled.
            }
            int32_t& token = acceptedTokenIds[i * maxAcceptDepth + j];
            if (token == forced[idx])
            {
                continue;
            }
            ownTokens[i] = token;
            token = forced[idx];
            acceptLengths[i] = j + 1;
            trimmed = true;
            break;
        }
    }
    return trimmed;
}

} // namespace rt
} // namespace trt_edgellm
