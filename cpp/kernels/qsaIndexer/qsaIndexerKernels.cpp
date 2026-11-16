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

// Host launchers for the QSA indexer kernels. The device code lives in
// kernelSrcs/qsaIndexer/qsaIndexerJitKernels.cu, is NVRTC-compiled on first use through
// qsaIndexerJitCompiler.cpp, and is driver-loaded once per (CUDA context, SM, dtype); grid,
// block, and shared-memory geometry here must match the kernel comments over there.

#include "qsaIndexerKernels.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "kernels/PluginJitKernels/pluginJitModuleRegistry.h"
#include "qsaIndexerJitCompiler.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

constexpr int32_t kWarpSize = 32;

constexpr int32_t ceilDiv(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

//! Host-side validation shared by every launcher that dereferences the pool state.
void checkPoolState(QsaIndexerPoolState const& poolState, char const* who)
{
    ELLM_CHECK(
        poolState.poolPtr != nullptr && poolState.pageTable != nullptr, std::string(who) + ": null pool state pointer");
    ELLM_CHECK(poolState.maxPagesPerSeq > 0 && poolState.numPages > 0 && poolState.numKVHeads > 0,
        std::string(who) + ": pool state dimensions must be positive");
    ELLM_CHECK(poolState.headSize > 0 && poolState.poolHeadDim >= poolState.headSize + kQSA_INDEXER_HEAD_DIM,
        std::string(who) + ": pool tail columns [headSize, headSize + 128) must follow a positive headSize and fit "
            + "poolHeadDim");
    ELLM_CHECK(poolState.headSize % 2 == 0 && (poolState.numKVHeads * poolState.poolHeadDim) % 2 == 0,
        std::string(who) + ": pool tails must start on an FP32 boundary (block rope row)");
}

template <typename T>
constexpr QsaIndexerJitDataType getQsaIndexerJitDataType();

template <>
constexpr QsaIndexerJitDataType getQsaIndexerJitDataType<half>()
{
    return QsaIndexerJitDataType::kHALF;
}

template <>
constexpr QsaIndexerJitDataType getQsaIndexerJitDataType<__nv_bfloat16>()
{
    return QsaIndexerJitDataType::kBF16;
}

//! One driver-loaded QSA indexer module with its resolved entry points.
struct QsaIndexerLoadedModule : PluginJitLoadedModule<QsaIndexerJitKey>
{
    CUfunction qPrep{};
    CUfunction kCompress{};
    CUfunction scores{};
    CUfunction idsFill{};
    CUfunction expand{};
    CUfunction rawKTailWritePrefill{};
    CUfunction preDecode{};
    CUfunction seal{};
    CUfunction scoresDecode{};
    CUfunction topKExpandDecode{};
    CUfunction kCompressAppend{};
    CUfunction scoresPaged{};
    CUfunction treePaths{};
    CUfunction treeKCompress{};
    CUfunction treeScores{};
    CUfunction treeMapIndices{};
};

void bindFunctions(QsaIndexerLoadedModule& loaded)
{
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.qPrep, loaded.module, "qsa_indexer_q_prep"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.kCompress, loaded.module, "qsa_indexer_k_compress"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.scores, loaded.module, "qsa_indexer_scores"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.idsFill, loaded.module, "qsa_indexer_ids_fill"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.expand, loaded.module, "qsa_indexer_expand"));
    CUDA_DRIVER_CHECK(
        cuModuleGetFunction(&loaded.rawKTailWritePrefill, loaded.module, "qsa_indexer_raw_k_tail_write_prefill"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.preDecode, loaded.module, "qsa_indexer_pre_decode"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.seal, loaded.module, "qsa_indexer_seal"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.scoresDecode, loaded.module, "qsa_indexer_scores_decode"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.topKExpandDecode, loaded.module, "qsa_indexer_topk_expand_decode"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.kCompressAppend, loaded.module, "qsa_indexer_k_compress_append"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.scoresPaged, loaded.module, "qsa_indexer_scores_paged"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.treePaths, loaded.module, "qsa_indexer_tree_paths"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.treeKCompress, loaded.module, "qsa_indexer_tree_k_compress"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.treeScores, loaded.module, "qsa_indexer_tree_scores"));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.treeMapIndices, loaded.module, "qsa_indexer_tree_map_indices"));
}

PluginJitModuleRegistry<QsaIndexerJitKey, QsaIndexerLoadedModule>& getModuleRegistry()
{
    static PluginJitModuleRegistry<QsaIndexerJitKey, QsaIndexerLoadedModule> registry{"QSA indexer JIT"};
    return registry;
}

//! Process-lifetime owner of the loaded QSA indexer modules. The launchers have no owner object,
//! so every module that comes out of the shared registry is pinned here; the entry-point handles
//! it owns are what every launcher below hands to cuLaunchKernel.
class QsaIndexerModules
{
public:
    std::shared_ptr<QsaIndexerLoadedModule> getOrLoad(QsaIndexerJitDataType const dataType)
    {
        QsaIndexerJitKey key;
        key.sm = getSMVersion();
        key.dataType = dataType;

        // Fast path: an already-loaded module must be reachable without any CUDA
        // runtime call — TRT re-invokes onShapeChange (and thus the warmup) during
        // CUDA-graph capture, where even cudaFree(nullptr) invalidates the capture.
        // cuCtxGetCurrent is a pure query and capture-safe.
        CUcontext context{};
        CUDA_DRIVER_CHECK(cuCtxGetCurrent(&context));
        if (context != nullptr)
        {
            std::lock_guard<std::mutex> const lock(mMutex);
            if (auto module = findLocked(context, key))
            {
                return module;
            }
        }

        // Slow path (first load): compile, then load through the shared registry, which
        // establishes the context. Illegal during graph capture by design — callers must warm
        // up (ensureQsaIndexerKernelsLoaded / a non-captured enqueue) before capturing.
        std::shared_ptr<QsaIndexerLoadedModule> module
            = getModuleRegistry().load(compileQsaIndexerJitKernel(key), bindFunctions);
        std::lock_guard<std::mutex> const lock(mMutex);
        if (auto existing = findLocked(module->context, key))
        {
            return existing;
        }
        mModules.push_back(module);
        return module;
    }

private:
    std::shared_ptr<QsaIndexerLoadedModule> findLocked(CUcontext const context, QsaIndexerJitKey const& key) const
    {
        for (std::shared_ptr<QsaIndexerLoadedModule> const& module : mModules)
        {
            if (module->context == context && module->key == key)
            {
                return module;
            }
        }
        return nullptr;
    }

    std::mutex mMutex;
    std::vector<std::shared_ptr<QsaIndexerLoadedModule>> mModules;
};

QsaIndexerModules& getModules()
{
    static QsaIndexerModules modules;
    return modules;
}

void launchKernel(CUfunction function, dim3 const& grid, dim3 const& block, cudaStream_t stream, void** args)
{
    CUDA_DRIVER_CHECK(
        cuLaunchKernel(function, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0U, stream, args, nullptr));
}

} // namespace

template <typename T>
void ensureQsaIndexerKernelsLoaded()
{
    getModules().getOrLoad(getQsaIndexerJitDataType<T>());
    // The dtype-independent kernels (ids fill, expand) always launch from the kHALF module.
    getModules().getOrLoad(QsaIndexerJitDataType::kHALF);
}

template <typename T>
void launchQsaIndexQPrep(T* qNormed, T const* indexQk, float const* cosSin, int32_t const* contextLengths, T const* wQ,
    float rmsEps, int32_t batchSize, int32_t seqLen, cudaStream_t stream)
{
    ELLM_CHECK(
        qNormed != nullptr && indexQk != nullptr && cosSin != nullptr && contextLengths != nullptr && wQ != nullptr,
        "launchQsaIndexQPrep: null pointer argument");
    ELLM_CHECK(batchSize > 0 && seqLen > 0, "launchQsaIndexQPrep: batchSize and seqLen must be positive");

    auto const module = getModules().getOrLoad(getQsaIndexerJitDataType<T>());
    dim3 const grid(static_cast<uint32_t>(batchSize) * static_cast<uint32_t>(seqLen));
    dim3 const block(kQSA_INDEXER_NUM_HEADS * kWarpSize);
    void* args[] = {&qNormed, &indexQk, &cosSin, &contextLengths, &wQ, &rmsEps, &seqLen};
    launchKernel(module->qPrep, grid, block, stream, args);
}

template <typename T>
void launchQsaIndexKCompress(T* kbar, T const* indexQk, float const* cosSin, int32_t const* contextLengths, T const* wK,
    float rmsEps, int32_t batchSize, int32_t seqLen, int32_t numBlocks, cudaStream_t stream)
{
    ELLM_CHECK(kbar != nullptr && indexQk != nullptr && cosSin != nullptr && contextLengths != nullptr && wK != nullptr,
        "launchQsaIndexKCompress: null pointer argument");
    ELLM_CHECK(batchSize > 0 && seqLen > 0 && numBlocks > 0,
        "launchQsaIndexKCompress: batchSize, seqLen and numBlocks must be positive");

    auto const module = getModules().getOrLoad(getQsaIndexerJitDataType<T>());
    dim3 const grid(static_cast<uint32_t>(batchSize) * static_cast<uint32_t>(numBlocks));
    dim3 const block(kWarpSize);
    void* args[] = {&kbar, &indexQk, &cosSin, &contextLengths, &wK, &rmsEps, &seqLen, &numBlocks};
    launchKernel(module->kCompress, grid, block, stream, args);
}

template <typename T>
void launchQsaIndexScores(float* logits, int32_t* visibleCounts, T const* qNormed, T const* kbar,
    int32_t const* contextLengths, int32_t batchSize, int32_t seqLen, int32_t numBlocks, int32_t rowStart,
    int32_t numRows, cudaStream_t stream)
{
    ELLM_CHECK(logits != nullptr && visibleCounts != nullptr && qNormed != nullptr && kbar != nullptr
            && contextLengths != nullptr,
        "launchQsaIndexScores: null pointer argument");
    ELLM_CHECK(batchSize > 0 && seqLen > 0 && numBlocks > 0, "launchQsaIndexScores: invalid problem dimensions");
    ELLM_CHECK(rowStart >= 0 && numRows > 0
            && static_cast<int64_t>(rowStart) + numRows <= static_cast<int64_t>(batchSize) * seqLen,
        "launchQsaIndexScores: row chunk out of range");

    auto const module = getModules().getOrLoad(getQsaIndexerJitDataType<T>());
    dim3 const grid(
        static_cast<uint32_t>(numRows), static_cast<uint32_t>(ceilDiv(numBlocks, kQSA_INDEXER_SCORES_COLUMNS_PER_CTA)));
    dim3 const block(kQSA_INDEXER_SCORES_THREADS);
    void* args[] = {&logits, &visibleCounts, &qNormed, &kbar, &contextLengths, &seqLen, &numBlocks, &rowStart};
    launchKernel(module->scores, grid, block, stream, args);
}

void launchQsaIndexIdsFill(int32_t* ids, int32_t numRows, int32_t numBlocks, cudaStream_t stream)
{
    ELLM_CHECK(ids != nullptr, "launchQsaIndexIdsFill: null pointer argument");
    ELLM_CHECK(numRows > 0 && numBlocks > 0, "launchQsaIndexIdsFill: invalid dimensions");

    auto const module = getModules().getOrLoad(QsaIndexerJitDataType::kHALF);
    int64_t numItems = static_cast<int64_t>(numRows) * numBlocks;
    constexpr int32_t kBlockSize = 256;
    int64_t const numBlocksLaunch = std::min<int64_t>((numItems + kBlockSize - 1) / kBlockSize, 65535);
    dim3 const grid(static_cast<uint32_t>(numBlocksLaunch));
    dim3 const block(kBlockSize);
    void* args[] = {&ids, &numItems, &numBlocks};
    launchKernel(module->idsFill, grid, block, stream, args);
}

void launchQsaIndexExpand(int32_t* outIdx, int32_t const* sortedIds, int32_t const* contextLengths,
    int32_t const* pastLengths, int32_t batchSize, int32_t seqLen, int32_t numBlocks, int32_t rowStart, int32_t numRows,
    cudaStream_t stream)
{
    ELLM_CHECK(outIdx != nullptr && sortedIds != nullptr && contextLengths != nullptr,
        "launchQsaIndexExpand: null pointer argument");
    ELLM_CHECK(batchSize > 0 && seqLen > 0 && numBlocks > 0, "launchQsaIndexExpand: invalid problem dimensions");
    ELLM_CHECK(rowStart >= 0 && numRows > 0
            && static_cast<int64_t>(rowStart) + numRows <= static_cast<int64_t>(batchSize) * seqLen,
        "launchQsaIndexExpand: row chunk out of range");

    auto const module = getModules().getOrLoad(QsaIndexerJitDataType::kHALF);
    dim3 const grid(static_cast<uint32_t>(numRows));
    dim3 const block(256);
    void* args[] = {&outIdx, &sortedIds, &contextLengths, &seqLen, &numBlocks, &rowStart, &pastLengths};
    launchKernel(module->expand, grid, block, stream, args);
}

template <typename T>
void launchQsaRawKTailWritePrefill(T const* indexQk, float const* cosSin, int32_t const* contextLengths,
    int32_t const* pastLengths, QsaIndexerPoolState const& poolState, int32_t batchSize, int32_t seqLen,
    cudaStream_t stream)
{
    ELLM_CHECK(indexQk != nullptr && cosSin != nullptr && contextLengths != nullptr,
        "launchQsaRawKTailWritePrefill: null pointer argument");
    ELLM_CHECK(batchSize > 0 && seqLen > 0, "launchQsaRawKTailWritePrefill: batchSize and seqLen must be positive");
    checkPoolState(poolState, "launchQsaRawKTailWritePrefill");

    auto const module = getModules().getOrLoad(getQsaIndexerJitDataType<T>());
    dim3 const grid(static_cast<uint32_t>(batchSize) * static_cast<uint32_t>(seqLen));
    dim3 const block(kWarpSize);
    QsaIndexerPoolState pool = poolState; // addressable copy; fields feed the args pack in order
    void* args[] = {&indexQk, &cosSin, &contextLengths, &pool.poolPtr, &pool.pageTable, &pool.maxPagesPerSeq,
        &pool.numPages, &pool.numKVHeads, &pool.poolHeadDim, &pool.headSize, &seqLen, &pastLengths};
    launchKernel(module->rawKTailWritePrefill, grid, block, stream, args);
}

template <typename T>
void launchQsaIndexerPreDecode(T* qNormed, T const* indexQk, float const* cosSin, int32_t const* contextLengths,
    T const* wQ, T const* wK, float rmsEps, QsaIndexerPoolState const& poolState, int32_t batchSize,
    cudaStream_t stream)
{
    ELLM_CHECK(qNormed != nullptr && indexQk != nullptr && cosSin != nullptr && contextLengths != nullptr
            && wQ != nullptr && wK != nullptr,
        "launchQsaIndexerPreDecode: null pointer argument");
    ELLM_CHECK(batchSize > 0, "launchQsaIndexerPreDecode: batchSize must be positive");
    checkPoolState(poolState, "launchQsaIndexerPreDecode");

    auto const module = getModules().getOrLoad(getQsaIndexerJitDataType<T>());
    dim3 const grid(static_cast<uint32_t>(batchSize));
    dim3 const block(kQSA_INDEXER_NUM_HEADS * kWarpSize);
    QsaIndexerPoolState pool = poolState; // addressable copy; fields feed the args pack in order
    void* args[] = {&qNormed, &indexQk, &cosSin, &contextLengths, &wQ, &wK, &rmsEps, &pool.poolPtr, &pool.pageTable,
        &pool.maxPagesPerSeq, &pool.numPages, &pool.numKVHeads, &pool.poolHeadDim, &pool.headSize};
    launchKernel(module->preDecode, grid, block, stream, args);
}

template <typename T>
void launchQsaIndexerSeal(int32_t const* lengths, T const* wK, float rmsEps, QsaIndexerPoolState const& poolState,
    int32_t batchSize, cudaStream_t stream)
{
    ELLM_CHECK(lengths != nullptr && wK != nullptr, "launchQsaIndexerSeal: null pointer argument");
    ELLM_CHECK(batchSize > 0, "launchQsaIndexerSeal: batchSize must be positive");
    checkPoolState(poolState, "launchQsaIndexerSeal");

    auto const module = getModules().getOrLoad(getQsaIndexerJitDataType<T>());
    dim3 const grid(static_cast<uint32_t>(kQSA_SEAL_BLOCKS), static_cast<uint32_t>(batchSize));
    QsaIndexerPoolState pool = poolState; // addressable copy; fields feed the args pack in order
    void* args[] = {&lengths, &wK, &rmsEps, &pool.poolPtr, &pool.pageTable, &pool.maxPagesPerSeq, &pool.numPages,
        &pool.numKVHeads, &pool.poolHeadDim, &pool.headSize};
    launchKernel(module->seal, grid, dim3(kWarpSize), stream, args);
}

template <typename T>
void launchQsaIndexScoresDecode(float* logits, T const* qNormed, int32_t const* contextLengths,
    QsaIndexerPoolState const& poolState, int32_t batchSize, int32_t maxBlocks, cudaStream_t stream)
{
    ELLM_CHECK(logits != nullptr && qNormed != nullptr && contextLengths != nullptr,
        "launchQsaIndexScoresDecode: null pointer argument");
    ELLM_CHECK(batchSize > 0 && maxBlocks > 0, "launchQsaIndexScoresDecode: invalid problem dimensions");
    checkPoolState(poolState, "launchQsaIndexScoresDecode");

    auto const module = getModules().getOrLoad(getQsaIndexerJitDataType<T>());
    dim3 const grid(static_cast<uint32_t>(batchSize),
        static_cast<uint32_t>(ceilDiv(maxBlocks, kQSA_INDEXER_SCORES_COLUMNS_PER_CTA)));
    dim3 const block(kQSA_INDEXER_SCORES_THREADS);
    QsaIndexerPoolState pool = poolState; // addressable copy; fields feed the args pack in order
    void* args[] = {&logits, &qNormed, &contextLengths, &pool.poolPtr, &pool.pageTable, &pool.maxPagesPerSeq,
        &pool.numPages, &pool.numKVHeads, &pool.poolHeadDim, &pool.headSize, &maxBlocks};
    launchKernel(module->scoresDecode, grid, block, stream, args);
}

void launchQsaTopKExpandDecode(int32_t* outIdx, float const* logits, int32_t const* contextLengths,
    int32_t* splitCounters, int32_t numKVHeads, int32_t batchSize, int32_t maxBlocks, cudaStream_t stream)
{
    ELLM_CHECK(outIdx != nullptr && logits != nullptr && contextLengths != nullptr,
        "launchQsaTopKExpandDecode: null pointer argument");
    ELLM_CHECK(batchSize > 0 && maxBlocks > 0, "launchQsaTopKExpandDecode: invalid problem dimensions");
    ELLM_CHECK(splitCounters == nullptr || (numKVHeads > 0 && numKVHeads <= kQSA_INDEXER_TOPK_THREADS),
        std::string("launchQsaTopKExpandDecode: numKVHeads must be in [1, ") + std::to_string(kQSA_INDEXER_TOPK_THREADS)
            + "] when splitCounters is set");

    auto const module = getModules().getOrLoad(QsaIndexerJitDataType::kHALF);
    dim3 const grid(static_cast<uint32_t>(batchSize));
    dim3 const block(kQSA_INDEXER_TOPK_THREADS);
    void* args[] = {&outIdx, &logits, &contextLengths, &splitCounters, &numKVHeads, &maxBlocks};
    launchKernel(module->topKExpandDecode, grid, block, stream, args);
}

template <typename T>
void launchQsaIndexKCompressAppend(T const* indexQk, float const* cosSin, int32_t const* queryLengths,
    int32_t const* pastLengths, T const* wK, float rmsEps, QsaIndexerPoolState const& poolState, int32_t batchSize,
    int32_t seqLen, cudaStream_t stream)
{
    ELLM_CHECK(indexQk != nullptr && cosSin != nullptr && queryLengths != nullptr && wK != nullptr,
        "launchQsaIndexKCompressAppend: null pointer argument");
    ELLM_CHECK(batchSize > 0 && seqLen > 0, "launchQsaIndexKCompressAppend: invalid dimensions");
    checkPoolState(poolState, "launchQsaIndexKCompressAppend");

    auto const module = getModules().getOrLoad(getQsaIndexerJitDataType<T>());
    dim3 const grid(static_cast<uint32_t>(ceilDiv(seqLen, kQSA_COMPRESS_RATIO)), static_cast<uint32_t>(batchSize));
    dim3 const block(kWarpSize);
    QsaIndexerPoolState pool = poolState;
    void* args[] = {&indexQk, &cosSin, &queryLengths, &pastLengths, &wK, &rmsEps, &seqLen, &pool.poolPtr,
        &pool.pageTable, &pool.maxPagesPerSeq, &pool.numPages, &pool.numKVHeads, &pool.poolHeadDim, &pool.headSize};
    launchKernel(module->kCompressAppend, grid, block, stream, args);
}

template <typename T>
void launchQsaIndexScoresPaged(float* logits, int32_t* visibleCounts, T const* qNormed, int32_t const* queryLengths,
    int32_t const* pastLengths, QsaIndexerPoolState const& poolState, int32_t batchSize, int32_t seqLen,
    int32_t maxBlocks, int32_t rowStart, int32_t numRows, cudaStream_t stream)
{
    ELLM_CHECK(logits != nullptr && visibleCounts != nullptr && qNormed != nullptr && queryLengths != nullptr,
        "launchQsaIndexScoresPaged: null pointer argument");
    ELLM_CHECK(batchSize > 0 && seqLen > 0 && maxBlocks > 0, "launchQsaIndexScoresPaged: invalid dimensions");
    ELLM_CHECK(rowStart >= 0 && numRows > 0
            && static_cast<int64_t>(rowStart) + numRows <= static_cast<int64_t>(batchSize) * seqLen,
        "launchQsaIndexScoresPaged: row chunk out of range");
    checkPoolState(poolState, "launchQsaIndexScoresPaged");

    auto const module = getModules().getOrLoad(getQsaIndexerJitDataType<T>());
    dim3 const grid(
        static_cast<uint32_t>(numRows), static_cast<uint32_t>(ceilDiv(maxBlocks, kQSA_INDEXER_SCORES_COLUMNS_PER_CTA)));
    dim3 const block(kQSA_INDEXER_SCORES_THREADS);
    QsaIndexerPoolState pool = poolState;
    void* args[] = {&logits, &visibleCounts, &qNormed, &queryLengths, &pastLengths, &seqLen, &maxBlocks, &rowStart,
        &pool.poolPtr, &pool.pageTable, &pool.maxPagesPerSeq, &pool.numPages, &pool.numKVHeads, &pool.poolHeadDim,
        &pool.headSize};
    launchKernel(module->scoresPaged, grid, block, stream, args);
}

template <typename T>
void launchQsaIndexerVerify(int32_t* outIdx, T const* indexQk, float const* cosSin, int32_t const* parents,
    int32_t const* depths, int32_t const* queryLengths, int32_t const* pastLengths, T const* wQ, T const* wK,
    float rmsEps, QsaIndexerPoolState const& poolState, QsaIndexerVerifyScratch const& scratch, int32_t batchSize,
    int32_t seqLen, int32_t maxBlocks, cudaStream_t stream)
{
    checkPoolState(poolState, "launchQsaIndexerVerify");
    auto const module = getModules().getOrLoad(getQsaIndexerJitDataType<T>());
    QsaIndexerPoolState pool = poolState;
    QsaIndexerVerifyScratch work = scratch;
    int32_t const rows = batchSize * seqLen;
    dim3 const rowGrid(static_cast<uint32_t>(rows));
    void* pathArgs[] = {&work.paths, &work.logicalLengths, &parents, &depths, &queryLengths, &pastLengths,
        &pool.pageTable, &seqLen, &maxBlocks, &pool.maxPagesPerSeq, &pool.numPages};
    launchKernel(module->treePaths, rowGrid, dim3(kWarpSize), stream, pathArgs);
    launchQsaIndexQPrep(
        static_cast<T*>(work.qNormed), indexQk, cosSin, queryLengths, wQ, rmsEps, batchSize, seqLen, stream);
    void* compressArgs[] = {&work.newKbar, &indexQk, &cosSin, &work.paths, &work.logicalLengths, &pastLengths, &wK,
        &rmsEps, &seqLen, &pool.poolPtr, &pool.pageTable, &pool.maxPagesPerSeq, &pool.numPages, &pool.numKVHeads,
        &pool.poolHeadDim, &pool.headSize};
    launchKernel(module->treeKCompress, rowGrid, dim3(kWarpSize), stream, compressArgs);
    void* scoresArgs[] = {&work.logits, &work.qNormed, &work.newKbar, &work.paths, &work.logicalLengths, &pastLengths,
        &seqLen, &maxBlocks, &pool.poolPtr, &pool.pageTable, &pool.maxPagesPerSeq, &pool.numPages, &pool.numKVHeads,
        &pool.poolHeadDim, &pool.headSize};
    dim3 const scoreGrid(
        static_cast<uint32_t>(rows), static_cast<uint32_t>(ceilDiv(maxBlocks, kQSA_INDEXER_SCORES_COLUMNS_PER_CTA)));
    launchKernel(module->treeScores, scoreGrid, dim3(kQSA_INDEXER_SCORES_THREADS), stream, scoresArgs);
    launchQsaTopKExpandDecode(
        outIdx, work.logits, work.logicalLengths, nullptr, pool.numKVHeads, rows, maxBlocks, stream);
    void* mapArgs[] = {&outIdx, &work.paths, &work.logicalLengths, &pastLengths, &seqLen};
    launchKernel(module->treeMapIndices, rowGrid, dim3(256), stream, mapArgs);
}

// Explicit instantiations for the supported activation types.
template void ensureQsaIndexerKernelsLoaded<half>();
template void ensureQsaIndexerKernelsLoaded<__nv_bfloat16>();

template void launchQsaIndexQPrep<half>(
    half*, half const*, float const*, int32_t const*, half const*, float, int32_t, int32_t, cudaStream_t);
template void launchQsaIndexQPrep<__nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16 const*, float const*, int32_t const*,
    __nv_bfloat16 const*, float, int32_t, int32_t, cudaStream_t);

template void launchQsaIndexKCompress<half>(
    half*, half const*, float const*, int32_t const*, half const*, float, int32_t, int32_t, int32_t, cudaStream_t);
template void launchQsaIndexKCompress<__nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16 const*, float const*, int32_t const*,
    __nv_bfloat16 const*, float, int32_t, int32_t, int32_t, cudaStream_t);

template void launchQsaIndexScores<half>(float*, int32_t*, half const*, half const*, int32_t const*, int32_t, int32_t,
    int32_t, int32_t, int32_t, cudaStream_t);
template void launchQsaIndexScores<__nv_bfloat16>(float*, int32_t*, __nv_bfloat16 const*, __nv_bfloat16 const*,
    int32_t const*, int32_t, int32_t, int32_t, int32_t, int32_t, cudaStream_t);

template void launchQsaRawKTailWritePrefill<half>(half const*, float const*, int32_t const*, int32_t const*,
    QsaIndexerPoolState const&, int32_t, int32_t, cudaStream_t);
template void launchQsaRawKTailWritePrefill<__nv_bfloat16>(__nv_bfloat16 const*, float const*, int32_t const*,
    int32_t const*, QsaIndexerPoolState const&, int32_t, int32_t, cudaStream_t);

template void launchQsaIndexerPreDecode<half>(half*, half const*, float const*, int32_t const*, half const*,
    half const*, float, QsaIndexerPoolState const&, int32_t, cudaStream_t);
template void launchQsaIndexerPreDecode<__nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16 const*, float const*,
    int32_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, float, QsaIndexerPoolState const&, int32_t,
    cudaStream_t);

template void launchQsaIndexerSeal<half>(
    int32_t const*, half const*, float, QsaIndexerPoolState const&, int32_t, cudaStream_t);
template void launchQsaIndexerSeal<__nv_bfloat16>(
    int32_t const*, __nv_bfloat16 const*, float, QsaIndexerPoolState const&, int32_t, cudaStream_t);

template void launchQsaIndexScoresDecode<half>(
    float*, half const*, int32_t const*, QsaIndexerPoolState const&, int32_t, int32_t, cudaStream_t);
template void launchQsaIndexScoresDecode<__nv_bfloat16>(
    float*, __nv_bfloat16 const*, int32_t const*, QsaIndexerPoolState const&, int32_t, int32_t, cudaStream_t);

template void launchQsaIndexKCompressAppend<half>(half const*, float const*, int32_t const*, int32_t const*,
    half const*, float, QsaIndexerPoolState const&, int32_t, int32_t, cudaStream_t);
template void launchQsaIndexKCompressAppend<__nv_bfloat16>(__nv_bfloat16 const*, float const*, int32_t const*,
    int32_t const*, __nv_bfloat16 const*, float, QsaIndexerPoolState const&, int32_t, int32_t, cudaStream_t);

template void launchQsaIndexScoresPaged<half>(float*, int32_t*, half const*, int32_t const*, int32_t const*,
    QsaIndexerPoolState const&, int32_t, int32_t, int32_t, int32_t, int32_t, cudaStream_t);
template void launchQsaIndexScoresPaged<__nv_bfloat16>(float*, int32_t*, __nv_bfloat16 const*, int32_t const*,
    int32_t const*, QsaIndexerPoolState const&, int32_t, int32_t, int32_t, int32_t, int32_t, cudaStream_t);

template void launchQsaIndexerVerify<half>(int32_t*, half const*, float const*, int32_t const*, int32_t const*,
    int32_t const*, int32_t const*, half const*, half const*, float, QsaIndexerPoolState const&,
    QsaIndexerVerifyScratch const&, int32_t, int32_t, int32_t, cudaStream_t);
template void launchQsaIndexerVerify<__nv_bfloat16>(int32_t*, __nv_bfloat16 const*, float const*, int32_t const*,
    int32_t const*, int32_t const*, int32_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, float,
    QsaIndexerPoolState const&, QsaIndexerVerifyScratch const&, int32_t, int32_t, int32_t, cudaStream_t);

} // namespace kernel
} // namespace trt_edgellm
