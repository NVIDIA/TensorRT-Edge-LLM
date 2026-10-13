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
#include "qsaIndexerJitCompiler.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>

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

//! One driver-loaded QSA indexer module with its ten resolved entry points.
struct QsaIndexerLoadedModule
{
    ~QsaIndexerLoadedModule()
    {
        if (module != nullptr)
        {
            (void) cuModuleUnload(module);
        }
    }

    CUcontext context{};
    CUmodule module{};
    CUfunction qPrep{};
    CUfunction kCompress{};
    CUfunction kCompressPaged{};
    CUfunction scores{};
    CUfunction idsFill{};
    CUfunction expand{};
    CUfunction rawKTailWritePrefill{};
    CUfunction preDecode{};
    CUfunction scoresDecode{};
    CUfunction topKExpandDecode{};
};

struct RegistryKey
{
    CUcontext context{};
    QsaIndexerJitKey jitKey{};

    bool operator==(RegistryKey const& other) const noexcept
    {
        return context == other.context && jitKey == other.jitKey;
    }
};

struct RegistryKeyHasher
{
    size_t operator()(RegistryKey const& key) const noexcept
    {
        auto mix = [](size_t hash, size_t value) noexcept {
            constexpr size_t kPRIME{0x100000001B3ULL};
            return (hash ^ value) * kPRIME;
        };

        using DataTypeValue = std::underlying_type_t<QsaIndexerJitDataType>;
        size_t hash{0xCBF29CE484222325ULL};
        hash = mix(hash, reinterpret_cast<uintptr_t>(key.context));
        hash = mix(hash, static_cast<size_t>(key.jitKey.sm));
        return mix(hash, static_cast<size_t>(static_cast<DataTypeValue>(key.jitKey.dataType)));
    }
};

//! Process-lifetime registry of loaded QSA indexer modules, keyed by (CUDA context, JIT
//! key). Modules stay loaded until process exit; the entry-point handles they own are
//! what every launcher below hands to cuLaunchKernel.
class QsaIndexerModuleRegistry
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
            std::lock_guard<std::mutex> lock(mMutex);
            auto const existing = mModules.find(RegistryKey{context, key});
            if (existing != mModules.end())
            {
                return existing->second;
            }
        }

        // Slow path (first load): establish the context and compile. Illegal during
        // graph capture by design — callers must warm up (ensureQsaIndexerKernelsLoaded
        // / a non-captured enqueue) before capturing.
        CUDA_CHECK(cudaFree(nullptr));
        CUDA_DRIVER_CHECK(cuCtxGetCurrent(&context));
        ELLM_CHECK(context != nullptr, "QSA indexer JIT module load requires a current CUDA context");

        RegistryKey const registryKey{context, key};
        std::lock_guard<std::mutex> lock(mMutex);
        auto const existing = mModules.find(registryKey);
        if (existing != mModules.end())
        {
            return existing->second;
        }

        QsaIndexerJitKernel const kernel = compileQsaIndexerJitKernel(key);
        auto loaded = std::make_shared<QsaIndexerLoadedModule>();
        loaded->context = context;
        CUDA_DRIVER_CHECK(cuModuleLoadData(&loaded->module, kernel.cubin.data()));
        CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded->qPrep, loaded->module, "qsa_indexer_q_prep"));
        CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded->kCompress, loaded->module, "qsa_indexer_k_compress"));
        CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded->kCompressPaged, loaded->module, "qsa_indexer_k_compress_paged"));
        CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded->scores, loaded->module, "qsa_indexer_scores"));
        CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded->idsFill, loaded->module, "qsa_indexer_ids_fill"));
        CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded->expand, loaded->module, "qsa_indexer_expand"));
        CUDA_DRIVER_CHECK(
            cuModuleGetFunction(&loaded->rawKTailWritePrefill, loaded->module, "qsa_indexer_raw_k_tail_write_prefill"));
        CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded->preDecode, loaded->module, "qsa_indexer_pre_decode"));
        CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded->scoresDecode, loaded->module, "qsa_indexer_scores_decode"));
        CUDA_DRIVER_CHECK(
            cuModuleGetFunction(&loaded->topKExpandDecode, loaded->module, "qsa_indexer_topk_expand_decode"));
        mModules.emplace(registryKey, loaded);
        return loaded;
    }

private:
    std::mutex mMutex;
    std::unordered_map<RegistryKey, std::shared_ptr<QsaIndexerLoadedModule>, RegistryKeyHasher> mModules;
};

QsaIndexerModuleRegistry& getModuleRegistry()
{
    static QsaIndexerModuleRegistry registry;
    return registry;
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
    getModuleRegistry().getOrLoad(getQsaIndexerJitDataType<T>());
    // The dtype-independent kernels (ids fill, expand) always launch from the kHALF module.
    getModuleRegistry().getOrLoad(QsaIndexerJitDataType::kHALF);
}

template <typename T>
void launchQsaIndexQPrep(T* qNormed, T const* indexQk, float const* cosSin, int32_t const* contextLengths, T const* wQ,
    float rmsEps, int32_t batchSize, int32_t seqLen, cudaStream_t stream)
{
    ELLM_CHECK(
        qNormed != nullptr && indexQk != nullptr && cosSin != nullptr && contextLengths != nullptr && wQ != nullptr,
        "launchQsaIndexQPrep: null pointer argument");
    ELLM_CHECK(batchSize > 0 && seqLen > 0, "launchQsaIndexQPrep: batchSize and seqLen must be positive");

    auto const module = getModuleRegistry().getOrLoad(getQsaIndexerJitDataType<T>());
    dim3 const grid(static_cast<uint32_t>(batchSize) * static_cast<uint32_t>(seqLen));
    dim3 const block(kQSA_INDEXER_NUM_HEADS * kWarpSize);
    void* args[] = {&qNormed, &indexQk, &cosSin, &contextLengths, &wQ, &rmsEps, &seqLen};
    launchKernel(module->qPrep, grid, block, stream, args);
}

template <typename T>
void launchQsaIndexKCompress(T* kbar, T const* indexQk, float const* cosSin, int32_t const* contextLengths, T const* wK,
    float rmsEps, int32_t batchSize, int32_t seqLen, int32_t numBlocks, int32_t blockBegin, int32_t blockEnd,
    int32_t pastLen, cudaStream_t stream)
{
    ELLM_CHECK(kbar != nullptr && indexQk != nullptr && cosSin != nullptr && contextLengths != nullptr && wK != nullptr,
        "launchQsaIndexKCompress: null pointer argument");
    ELLM_CHECK(batchSize > 0 && seqLen > 0, "launchQsaIndexKCompress: batchSize and seqLen must be positive");
    ELLM_CHECK(0 <= blockBegin && blockBegin < blockEnd && blockEnd <= numBlocks,
        "launchQsaIndexKCompress: require 0 <= blockBegin < blockEnd <= numBlocks");
    ELLM_CHECK(pastLen >= 0, "launchQsaIndexKCompress: pastLen must be non-negative");

    auto const module = getModuleRegistry().getOrLoad(getQsaIndexerJitDataType<T>());
    int32_t numGroups = blockEnd - blockBegin;
    dim3 const grid(static_cast<uint32_t>(batchSize) * static_cast<uint32_t>(numGroups));
    dim3 const block(kWarpSize);
    void* args[] = {&kbar, &indexQk, &cosSin, &contextLengths, &wK, &rmsEps, &seqLen, &numBlocks, &numGroups,
        &blockBegin, &pastLen};
    launchKernel(module->kCompress, grid, block, stream, args);
}

template <typename T>
void launchQsaIndexScores(float* logits, T const* qNormed, T const* kbar, int32_t const* contextLengths,
    int32_t batchSize, int32_t seqLen, int32_t numBlocks, int32_t rowStart, int32_t numRows, cudaStream_t stream)
{
    ELLM_CHECK(logits != nullptr && qNormed != nullptr && kbar != nullptr && contextLengths != nullptr,
        "launchQsaIndexScores: null pointer argument");
    ELLM_CHECK(batchSize > 0 && seqLen > 0 && numBlocks > 0, "launchQsaIndexScores: invalid problem dimensions");
    ELLM_CHECK(rowStart >= 0 && numRows > 0
            && static_cast<int64_t>(rowStart) + numRows <= static_cast<int64_t>(batchSize) * seqLen,
        "launchQsaIndexScores: row chunk out of range");

    auto const module = getModuleRegistry().getOrLoad(getQsaIndexerJitDataType<T>());
    dim3 const grid(
        static_cast<uint32_t>(numRows), static_cast<uint32_t>(ceilDiv(numBlocks, kQSA_INDEXER_SCORES_COLUMNS_PER_CTA)));
    dim3 const block(kQSA_INDEXER_SCORES_THREADS);
    void* args[] = {&logits, &qNormed, &kbar, &contextLengths, &seqLen, &numBlocks, &rowStart};
    launchKernel(module->scores, grid, block, stream, args);
}

void launchQsaIndexIdsFill(int32_t* ids, int32_t numRows, int32_t numBlocks, cudaStream_t stream)
{
    ELLM_CHECK(ids != nullptr, "launchQsaIndexIdsFill: null pointer argument");
    ELLM_CHECK(numRows > 0 && numBlocks > 0, "launchQsaIndexIdsFill: invalid dimensions");

    auto const module = getModuleRegistry().getOrLoad(QsaIndexerJitDataType::kHALF);
    int64_t numItems = static_cast<int64_t>(numRows) * numBlocks;
    constexpr int32_t kBlockSize = 256;
    int64_t const numBlocksLaunch = std::min<int64_t>((numItems + kBlockSize - 1) / kBlockSize, 65535);
    dim3 const grid(static_cast<uint32_t>(numBlocksLaunch));
    dim3 const block(kBlockSize);
    void* args[] = {&ids, &numItems, &numBlocks};
    launchKernel(module->idsFill, grid, block, stream, args);
}

void launchQsaIndexExpand(int32_t* outIdx, int32_t const* sortedIds, int32_t const* contextLengths, int32_t batchSize,
    int32_t seqLen, int32_t numBlocks, int32_t rowStart, int32_t numRows, cudaStream_t stream)
{
    ELLM_CHECK(outIdx != nullptr && sortedIds != nullptr && contextLengths != nullptr,
        "launchQsaIndexExpand: null pointer argument");
    ELLM_CHECK(batchSize > 0 && seqLen > 0 && numBlocks > 0, "launchQsaIndexExpand: invalid problem dimensions");
    ELLM_CHECK(rowStart >= 0 && numRows > 0
            && static_cast<int64_t>(rowStart) + numRows <= static_cast<int64_t>(batchSize) * seqLen,
        "launchQsaIndexExpand: row chunk out of range");

    auto const module = getModuleRegistry().getOrLoad(QsaIndexerJitDataType::kHALF);
    dim3 const grid(static_cast<uint32_t>(numRows));
    dim3 const block(256);
    void* args[] = {&outIdx, &sortedIds, &contextLengths, &seqLen, &numBlocks, &rowStart};
    launchKernel(module->expand, grid, block, stream, args);
}

template <typename T>
void launchQsaIndexKCompressPaged(T* kbar, T const* indexQk, float const* cosSin, int32_t const* contextLengths,
    T const* wK, float rmsEps, QsaIndexerPoolState const& poolState, int32_t batchSize, int32_t seqLen,
    int32_t numBlocks, int32_t blockBegin, int32_t blockEnd, int32_t pastLen, cudaStream_t stream)
{
    ELLM_CHECK(kbar != nullptr && indexQk != nullptr && cosSin != nullptr && contextLengths != nullptr && wK != nullptr,
        "launchQsaIndexKCompressPaged: null pointer argument");
    ELLM_CHECK(batchSize > 0 && seqLen > 0, "launchQsaIndexKCompressPaged: batchSize and seqLen must be positive");
    ELLM_CHECK(0 <= blockBegin && blockBegin < blockEnd && blockEnd <= numBlocks,
        "launchQsaIndexKCompressPaged: require 0 <= blockBegin < blockEnd <= numBlocks");
    ELLM_CHECK(pastLen >= 0, "launchQsaIndexKCompressPaged: pastLen must be non-negative");
    checkPoolState(poolState, "launchQsaIndexKCompressPaged");

    auto const module = getModuleRegistry().getOrLoad(getQsaIndexerJitDataType<T>());
    int32_t numGroups = blockEnd - blockBegin;
    dim3 const grid(static_cast<uint32_t>(batchSize) * static_cast<uint32_t>(numGroups));
    dim3 const block(kWarpSize);
    QsaIndexerPoolState pool = poolState; // addressable copy; fields feed the args pack in order
    void* args[] = {&kbar, &indexQk, &cosSin, &contextLengths, &wK, &rmsEps, &seqLen, &numBlocks, &numGroups,
        &blockBegin, &pastLen, &pool.poolPtr, &pool.pageTable, &pool.maxPagesPerSeq, &pool.numPages, &pool.numKVHeads,
        &pool.poolHeadDim, &pool.headSize};
    launchKernel(module->kCompressPaged, grid, block, stream, args);
}

template <typename T>
void launchQsaRawKTailWritePrefill(T const* indexQk, int32_t const* contextLengths,
    QsaIndexerPoolState const& poolState, int32_t batchSize, int32_t seqLen, cudaStream_t stream)
{
    ELLM_CHECK(indexQk != nullptr && contextLengths != nullptr, "launchQsaRawKTailWritePrefill: null pointer argument");
    ELLM_CHECK(batchSize > 0 && seqLen > 0, "launchQsaRawKTailWritePrefill: batchSize and seqLen must be positive");
    checkPoolState(poolState, "launchQsaRawKTailWritePrefill");

    auto const module = getModuleRegistry().getOrLoad(getQsaIndexerJitDataType<T>());
    dim3 const grid(static_cast<uint32_t>(batchSize), static_cast<uint32_t>(kQSA_COMPRESS_RATIO - 1));
    dim3 const block(kWarpSize);
    QsaIndexerPoolState pool = poolState; // addressable copy; fields feed the args pack in order
    void* args[] = {&indexQk, &contextLengths, &pool.poolPtr, &pool.pageTable, &pool.maxPagesPerSeq, &pool.numPages,
        &pool.numKVHeads, &pool.poolHeadDim, &pool.headSize, &seqLen};
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

    auto const module = getModuleRegistry().getOrLoad(getQsaIndexerJitDataType<T>());
    dim3 const grid(static_cast<uint32_t>(batchSize));
    dim3 const block(kQSA_INDEXER_NUM_HEADS * kWarpSize);
    QsaIndexerPoolState pool = poolState; // addressable copy; fields feed the args pack in order
    void* args[] = {&qNormed, &indexQk, &cosSin, &contextLengths, &wQ, &wK, &rmsEps, &pool.poolPtr, &pool.pageTable,
        &pool.maxPagesPerSeq, &pool.numPages, &pool.numKVHeads, &pool.poolHeadDim, &pool.headSize};
    launchKernel(module->preDecode, grid, block, stream, args);
}

template <typename T>
void launchQsaIndexScoresDecode(float* logits, T const* qNormed, int32_t const* contextLengths,
    QsaIndexerPoolState const& poolState, int32_t batchSize, int32_t maxBlocks, cudaStream_t stream)
{
    ELLM_CHECK(logits != nullptr && qNormed != nullptr && contextLengths != nullptr,
        "launchQsaIndexScoresDecode: null pointer argument");
    ELLM_CHECK(batchSize > 0 && maxBlocks > 0, "launchQsaIndexScoresDecode: invalid problem dimensions");
    checkPoolState(poolState, "launchQsaIndexScoresDecode");

    auto const module = getModuleRegistry().getOrLoad(getQsaIndexerJitDataType<T>());
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

    auto const module = getModuleRegistry().getOrLoad(QsaIndexerJitDataType::kHALF);
    dim3 const grid(static_cast<uint32_t>(batchSize));
    dim3 const block(kQSA_INDEXER_TOPK_THREADS);
    void* args[] = {&outIdx, &logits, &contextLengths, &splitCounters, &numKVHeads, &maxBlocks};
    launchKernel(module->topKExpandDecode, grid, block, stream, args);
}

// Explicit instantiations for the supported activation types.
template void ensureQsaIndexerKernelsLoaded<half>();
template void ensureQsaIndexerKernelsLoaded<__nv_bfloat16>();

template void launchQsaIndexQPrep<half>(
    half*, half const*, float const*, int32_t const*, half const*, float, int32_t, int32_t, cudaStream_t);
template void launchQsaIndexQPrep<__nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16 const*, float const*, int32_t const*,
    __nv_bfloat16 const*, float, int32_t, int32_t, cudaStream_t);

template void launchQsaIndexKCompress<half>(half*, half const*, float const*, int32_t const*, half const*, float,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, cudaStream_t);
template void launchQsaIndexKCompress<__nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16 const*, float const*, int32_t const*,
    __nv_bfloat16 const*, float, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, cudaStream_t);

template void launchQsaIndexScores<half>(
    float*, half const*, half const*, int32_t const*, int32_t, int32_t, int32_t, int32_t, int32_t, cudaStream_t);
template void launchQsaIndexScores<__nv_bfloat16>(float*, __nv_bfloat16 const*, __nv_bfloat16 const*, int32_t const*,
    int32_t, int32_t, int32_t, int32_t, int32_t, cudaStream_t);

template void launchQsaIndexKCompressPaged<half>(half*, half const*, float const*, int32_t const*, half const*, float,
    QsaIndexerPoolState const&, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, cudaStream_t);
template void launchQsaIndexKCompressPaged<__nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16 const*, float const*,
    int32_t const*, __nv_bfloat16 const*, float, QsaIndexerPoolState const&, int32_t, int32_t, int32_t, int32_t,
    int32_t, int32_t, cudaStream_t);

template void launchQsaRawKTailWritePrefill<half>(
    half const*, int32_t const*, QsaIndexerPoolState const&, int32_t, int32_t, cudaStream_t);
template void launchQsaRawKTailWritePrefill<__nv_bfloat16>(
    __nv_bfloat16 const*, int32_t const*, QsaIndexerPoolState const&, int32_t, int32_t, cudaStream_t);

template void launchQsaIndexerPreDecode<half>(half*, half const*, float const*, int32_t const*, half const*,
    half const*, float, QsaIndexerPoolState const&, int32_t, cudaStream_t);
template void launchQsaIndexerPreDecode<__nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16 const*, float const*,
    int32_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, float, QsaIndexerPoolState const&, int32_t,
    cudaStream_t);

template void launchQsaIndexScoresDecode<half>(
    float*, half const*, int32_t const*, QsaIndexerPoolState const&, int32_t, int32_t, cudaStream_t);
template void launchQsaIndexScoresDecode<__nv_bfloat16>(
    float*, __nv_bfloat16 const*, int32_t const*, QsaIndexerPoolState const&, int32_t, int32_t, cudaStream_t);

} // namespace kernel
} // namespace trt_edgellm
