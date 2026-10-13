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

#include "cuteDslQsaSparseRunner.h"

#if defined(CUTE_DSL_QSA_ENABLED)

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/contextAttentionKernels/cuteDslTensorDescriptors.h"

namespace trt_edgellm
{

detail::LazyKernelModule<qsa_sparse_d256_fp16_Kernel_Module_t> CuteDslQsaSparsePrefillRunner::sSparseD256Fp16{};
detail::LazyKernelModule<qsa_sparse_d256_bf16_Kernel_Module_t> CuteDslQsaSparsePrefillRunner::sSparseD256Bf16{};

namespace
{

using cutedsl::makeCuSeqLenTensor;
using cutedsl::makePackedTensor;
using cutedsl::makeStridedTensor;
using cutedsl::WrapperArgT;
using cutedsl::WrapperArity;

//! The QSA family bakes head_dim = 256 and the (Br, Bc, threads) tuning from build_cutedsl.py (the decode
//! variants also bake kMaxSplits).
constexpr int32_t kQsaHeadDim{256};
//! The MMA M tile bounds the GQA group size (Qwen3.8-Flash-Next: 24 / 2 = 12).
constexpr int32_t kQsaMaxGroupSize{16};

bool isQsaSm(int32_t smVersion)
{
    return smVersion == 100 || smVersion == 101 || smVersion == 110;
}

//! Populate a [B, S, H] descriptor over a contiguous [B, S, H, D] buffer. The QSA kernels bake
//! the head dim in, so D is not one of the extents and the strides have to be spelled out.
template <class TensorT>
TensorT makeQsaBshTensor(void const* data, int32_t batchSize, int32_t seqLen, int32_t numHeads, int32_t headDim)
{
    return makeStridedTensor<TensorT>(data, {batchSize, seqLen, numHeads},
        {static_cast<int64_t>(seqLen) * numHeads * headDim, static_cast<int64_t>(numHeads) * headDim});
}

//! Launch one QSA sparse prefill variant over dense padded BSND Q/K/V/O plus the index lists.
template <auto cuteDslKernelWrapper, auto moduleLoader, auto moduleUnloader>
int32_t callQsaSparsePrefill(detail::LazyKernelModule<WrapperArgT<0, decltype(cuteDslKernelWrapper)>>& state,
    char const* moduleName, QsaSparsePrefillParams const& params)
{
    static_assert(WrapperArity<decltype(cuteDslKernelWrapper)>::value == 10,
        "callQsaSparsePrefill: not a QSA sparse prefill wrapper (module, q_tensor, k_tensor, v_tensor, o_tensor, "
        "indices, context_lengths, attention_scale, sm_count, stream).");

    if (!detail::ensureModuleLoaded<moduleLoader, moduleUnloader>(state, moduleName, params.stream))
    {
        return -1;
    }
    auto& module = state.module;

    auto qTensor = makeQsaBshTensor<WrapperArgT<1, decltype(cuteDslKernelWrapper)>>(
        params.qPtr, params.batchSize, params.seqLen, params.numQHeads, params.headDim);
    auto kTensor = makeQsaBshTensor<WrapperArgT<2, decltype(cuteDslKernelWrapper)>>(
        params.kPtr, params.batchSize, params.seqLen, params.numKVHeads, params.headDim);
    auto vTensor = makeQsaBshTensor<WrapperArgT<3, decltype(cuteDslKernelWrapper)>>(
        params.vPtr, params.batchSize, params.seqLen, params.numKVHeads, params.headDim);
    auto oTensor = makeQsaBshTensor<WrapperArgT<4, decltype(cuteDslKernelWrapper)>>(
        params.oPtr, params.batchSize, params.seqLen, params.numQHeads, params.headDim);
    auto indicesTensor = makePackedTensor<WrapperArgT<5, decltype(cuteDslKernelWrapper)>>(
        params.indices, {params.batchSize, params.seqLen, params.topK});
    auto contextLengths
        = makeCuSeqLenTensor<WrapperArgT<6, decltype(cuteDslKernelWrapper)>>(params.contextLengths, params.batchSize);

    return cuteDslKernelWrapper(&module, &qTensor, &kTensor, &vTensor, &oTensor, &indicesTensor, &contextLengths,
        params.attentionScale, getDeviceMultiProcessorCount(), params.stream);
}

bool validateQsaParams(QsaSparsePrefillParams const& params)
{
    if (params.headDim != kQsaHeadDim)
    {
        LOG_ERROR("QSA sparse prefill: unsupported head_dim=%d (expected %d)", params.headDim, kQsaHeadDim);
        return false;
    }
    if (params.numKVHeads <= 0 || params.numQHeads % params.numKVHeads != 0
        || params.numQHeads / params.numKVHeads > kQsaMaxGroupSize)
    {
        LOG_ERROR("QSA sparse prefill: unsupported GQA shape H_q=%d, H_kv=%d (group must divide and be <= %d)",
            params.numQHeads, params.numKVHeads, kQsaMaxGroupSize);
        return false;
    }
    if (params.batchSize <= 0 || params.seqLen <= 0 || params.topK <= 0)
    {
        LOG_ERROR(
            "QSA sparse prefill: degenerate shape B=%d, S=%d, topk=%d", params.batchSize, params.seqLen, params.topK);
        return false;
    }
    if (params.qPtr == nullptr || params.kPtr == nullptr || params.vPtr == nullptr || params.oPtr == nullptr
        || params.indices == nullptr || params.contextLengths == nullptr)
    {
        LOG_ERROR("QSA sparse prefill: null tensor pointer");
        return false;
    }
    return true;
}

} // namespace

bool CuteDslQsaSparsePrefillRunner::canImplement(
    int32_t numQHeads, int32_t numKVHeads, int32_t headDim, int32_t smVersion, nvinfer1::DataType dataType)
{
    if (!isQsaSm(smVersion))
    {
        return false;
    }
    if (dataType != nvinfer1::DataType::kHALF && dataType != nvinfer1::DataType::kBF16)
    {
        return false;
    }
    if (headDim != kQsaHeadDim)
    {
        return false;
    }
    if (numQHeads <= 0 || numKVHeads <= 0 || numQHeads % numKVHeads != 0 || numQHeads / numKVHeads > kQsaMaxGroupSize)
    {
        return false;
    }
    return true;
}

bool CuteDslQsaSparsePrefillRunner::preflight(nvinfer1::DataType dataType, cudaStream_t stream)
{
    if (dataType == nvinfer1::DataType::kHALF)
    {
        return detail::ensureModuleLoaded<qsa_sparse_d256_fp16_Kernel_Module_Load,
            qsa_sparse_d256_fp16_Kernel_Module_Unload>(sSparseD256Fp16, "qsa_sparse_d256_fp16", stream);
    }
    if (dataType == nvinfer1::DataType::kBF16)
    {
        return detail::ensureModuleLoaded<qsa_sparse_d256_bf16_Kernel_Module_Load,
            qsa_sparse_d256_bf16_Kernel_Module_Unload>(sSparseD256Bf16, "qsa_sparse_d256_bf16", stream);
    }
    LOG_ERROR("QSA sparse prefill: unsupported data type");
    return false;
}

bool CuteDslQsaSparsePrefillRunner::run(nvinfer1::DataType dataType, QsaSparsePrefillParams const& params)
{
    if (!validateQsaParams(params))
    {
        return false;
    }
    int32_t status = -1;
    if (dataType == nvinfer1::DataType::kHALF)
    {
        status = callQsaSparsePrefill<cute_dsl_qsa_sparse_d256_fp16_wrapper, qsa_sparse_d256_fp16_Kernel_Module_Load,
            qsa_sparse_d256_fp16_Kernel_Module_Unload>(sSparseD256Fp16, "qsa_sparse_d256_fp16", params);
    }
    else if (dataType == nvinfer1::DataType::kBF16)
    {
        status = callQsaSparsePrefill<cute_dsl_qsa_sparse_d256_bf16_wrapper, qsa_sparse_d256_bf16_Kernel_Module_Load,
            qsa_sparse_d256_bf16_Kernel_Module_Unload>(sSparseD256Bf16, "qsa_sparse_d256_bf16", params);
    }
    else
    {
        LOG_ERROR("QSA sparse prefill: unsupported data type");
        return false;
    }
    if (status != 0)
    {
        LOG_ERROR("QSA sparse prefill kernel launch failed with status %d", status);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Decode (single-launch split-K)
// ---------------------------------------------------------------------------

detail::LazyKernelModule<qsa_sparse_decode_d256_fp16_Kernel_Module_t> CuteDslQsaSparseDecodeRunner::sDecodeD256Fp16{};
detail::LazyKernelModule<qsa_sparse_decode_d256_bf16_Kernel_Module_t> CuteDslQsaSparseDecodeRunner::sDecodeD256Bf16{};

namespace
{

//! Populate a rank-1 descriptor of any element type (the exporter emits no
//! dynamic_strides member for single-dynamic-extent tensors).
template <class TensorT>
constexpr TensorT makeRank1Tensor(void const* data, int32_t length)
{
    TensorT tensor{};
    tensor.data = const_cast<void*>(data);
    tensor.dynamic_shapes[0] = length;
    return tensor;
}

//! fp32 partial workspaces are exported with either one dynamic extent (slots)
//! or all three, depending on how the exporter folds the static (rows, cols)
//! modes — adapt to whichever descriptor the generated header declares.
template <class TensorT>
constexpr TensorT makePartialTensor(void const* data, int32_t slots, int32_t rows, int32_t cols)
{
    if constexpr (cutedsl::kShapeRank<TensorT> == 1)
    {
        return makeRank1Tensor<TensorT>(data, slots);
    }
    else
    {
        return makePackedTensor<TensorT>(data, {slots, rows, cols});
    }
}

//! Launch the QSA split-K decode wrapper: (module, q, kv_pool, page_table, indices,
//! context_lengths, o, partial_o, partial_stats, counters, attention_scale, stream).
template <auto cuteDslKernelWrapper, auto moduleLoader, auto moduleUnloader>
int32_t callQsaSparseDecode(detail::LazyKernelModule<WrapperArgT<0, decltype(cuteDslKernelWrapper)>>& state,
    char const* moduleName, QsaSparseDecodeParams const& params)
{
    static_assert(WrapperArity<decltype(cuteDslKernelWrapper)>::value == 12,
        "callQsaSparseDecode: not a QSA sparse decode wrapper (module, q_tensor, kv_pool, page_table, indices, "
        "context_lengths, o_tensor, partial_o, partial_stats, counters, attention_scale, stream).");

    if (!detail::ensureModuleLoaded<moduleLoader, moduleUnloader>(state, moduleName, params.stream))
    {
        return -1;
    }
    auto& module = state.module;

    auto qTensor = makeQsaBshTensor<WrapperArgT<1, decltype(cuteDslKernelWrapper)>>(
        params.qPtr, params.batchSize, /*seqLen=*/1, params.numQHeads, params.headDim);
    // Pool descriptor carries all four extents (poolHeadDim is runtime-dynamic).
    auto poolTensor = makeStridedTensor<WrapperArgT<2, decltype(cuteDslKernelWrapper)>>(params.kvPoolPtr,
        {params.numFlatPages, 128, params.numKVHeads, params.poolHeadDim},
        {static_cast<int64_t>(128) * params.numKVHeads * params.poolHeadDim,
            static_cast<int64_t>(params.numKVHeads) * params.poolHeadDim, static_cast<int64_t>(params.poolHeadDim)});
    auto pageTableTensor = makePackedTensor<WrapperArgT<3, decltype(cuteDslKernelWrapper)>>(
        params.pageTable, {params.batchSize, 2, params.maxPagesPerSeq});
    auto indicesTensor = makePackedTensor<WrapperArgT<4, decltype(cuteDslKernelWrapper)>>(
        params.indices, {params.batchSize, 1, params.topK});
    auto contextLengths
        = makeCuSeqLenTensor<WrapperArgT<5, decltype(cuteDslKernelWrapper)>>(params.contextLengths, params.batchSize);
    auto oTensor = makeQsaBshTensor<WrapperArgT<6, decltype(cuteDslKernelWrapper)>>(
        params.oPtr, params.batchSize, /*seqLen=*/1, params.numQHeads, params.headDim);
    int32_t const partialSlots = params.batchSize * params.numKVHeads * CuteDslQsaSparseDecodeRunner::kMaxSplits;
    auto partialOTensor = makePartialTensor<WrapperArgT<7, decltype(cuteDslKernelWrapper)>>(
        params.partialO, partialSlots, CuteDslQsaSparseDecodeRunner::kPartialRows, params.headDim);
    auto partialStatsTensor = makePartialTensor<WrapperArgT<8, decltype(cuteDslKernelWrapper)>>(
        params.partialStats, partialSlots, 2, CuteDslQsaSparseDecodeRunner::kPartialRows);
    auto countersTensor = makeRank1Tensor<WrapperArgT<9, decltype(cuteDslKernelWrapper)>>(
        params.splitCounters, params.batchSize * params.numKVHeads);

    return cuteDslKernelWrapper(&module, &qTensor, &poolTensor, &pageTableTensor, &indicesTensor, &contextLengths,
        &oTensor, &partialOTensor, &partialStatsTensor, &countersTensor, params.attentionScale, params.stream);
}

} // namespace

bool CuteDslQsaSparseDecodeRunner::canImplement(int32_t numQHeads, int32_t numKVHeads, int32_t headDim,
    int32_t poolHeadDim, int32_t smVersion, nvinfer1::DataType dataType)
{
    if (!CuteDslQsaSparsePrefillRunner::canImplement(numQHeads, numKVHeads, headDim, smVersion, dataType))
    {
        return false;
    }
    if (poolHeadDim < headDim || poolHeadDim % 8 != 0)
    {
        return false;
    }
    return true;
}

bool CuteDslQsaSparseDecodeRunner::preflight(nvinfer1::DataType dataType, cudaStream_t stream)
{
    if (dataType == nvinfer1::DataType::kHALF)
    {
        return detail::ensureModuleLoaded<qsa_sparse_decode_d256_fp16_Kernel_Module_Load,
            qsa_sparse_decode_d256_fp16_Kernel_Module_Unload>(sDecodeD256Fp16, "qsa_sparse_decode_d256_fp16", stream);
    }
    if (dataType == nvinfer1::DataType::kBF16)
    {
        return detail::ensureModuleLoaded<qsa_sparse_decode_d256_bf16_Kernel_Module_Load,
            qsa_sparse_decode_d256_bf16_Kernel_Module_Unload>(sDecodeD256Bf16, "qsa_sparse_decode_d256_bf16", stream);
    }
    LOG_ERROR("QSA sparse decode: unsupported data type");
    return false;
}

bool CuteDslQsaSparseDecodeRunner::run(nvinfer1::DataType dataType, QsaSparseDecodeParams const& params)
{
    if (params.headDim != kQsaHeadDim || params.poolHeadDim < params.headDim || params.batchSize <= 0
        || params.topK <= 0 || params.qPtr == nullptr || params.kvPoolPtr == nullptr || params.pageTable == nullptr
        || params.indices == nullptr || params.contextLengths == nullptr || params.oPtr == nullptr
        || params.partialO == nullptr || params.partialStats == nullptr || params.splitCounters == nullptr)
    {
        LOG_ERROR("QSA sparse decode: invalid parameters");
        return false;
    }
    int32_t status = -1;
    if (dataType == nvinfer1::DataType::kHALF)
    {
        status = callQsaSparseDecode<cute_dsl_qsa_sparse_decode_d256_fp16_wrapper,
            qsa_sparse_decode_d256_fp16_Kernel_Module_Load, qsa_sparse_decode_d256_fp16_Kernel_Module_Unload>(
            sDecodeD256Fp16, "qsa_sparse_decode_d256_fp16", params);
    }
    else if (dataType == nvinfer1::DataType::kBF16)
    {
        status = callQsaSparseDecode<cute_dsl_qsa_sparse_decode_d256_bf16_wrapper,
            qsa_sparse_decode_d256_bf16_Kernel_Module_Load, qsa_sparse_decode_d256_bf16_Kernel_Module_Unload>(
            sDecodeD256Bf16, "qsa_sparse_decode_d256_bf16", params);
    }
    else
    {
        LOG_ERROR("QSA sparse decode: unsupported data type");
        return false;
    }
    if (status != 0)
    {
        LOG_ERROR("QSA sparse decode kernel launch failed with status %d", status);
        return false;
    }
    return true;
}

} // namespace trt_edgellm

#else

// Keep symbols available for unconditional callers; false reports that CuTe DSL kernels are unavailable.
namespace trt_edgellm
{

bool CuteDslQsaSparsePrefillRunner::canImplement(int32_t, int32_t, int32_t, int32_t, nvinfer1::DataType)
{
    return false;
}

bool CuteDslQsaSparsePrefillRunner::preflight(nvinfer1::DataType, cudaStream_t)
{
    return false;
}

bool CuteDslQsaSparsePrefillRunner::run(nvinfer1::DataType, QsaSparsePrefillParams const&)
{
    return false;
}

bool CuteDslQsaSparseDecodeRunner::canImplement(int32_t, int32_t, int32_t, int32_t, int32_t, nvinfer1::DataType)
{
    return false;
}

bool CuteDslQsaSparseDecodeRunner::preflight(nvinfer1::DataType, cudaStream_t)
{
    return false;
}

bool CuteDslQsaSparseDecodeRunner::run(nvinfer1::DataType, QsaSparseDecodeParams const&)
{
    return false;
}

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_QSA_ENABLED)
