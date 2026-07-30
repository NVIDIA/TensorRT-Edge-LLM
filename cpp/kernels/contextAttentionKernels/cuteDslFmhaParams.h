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

#include <cstdint>
#include <cuda_runtime.h>

//! Argument bundles for the CuTe DSL FMHA launchers.
//!
//! Each public run* entry point fills one of these once from its own arguments and members, and the
//! launcher then spreads it across the generated descriptors. One struct per wrapper *shape* rather
//! than a single superset: a superset would let a field that matters on one path (kvCacheCapacity on
//! the dense path, tokensPerPage on the paged one) sit silently at zero on another.
//!
//! These carry no generated types, so this header is safe to include from translation units that use the required
//! FMHA-v2 runner, the optional CUTE_DSL_FMHA_BLACKWELL_ENABLED runner, or both. The descriptor-filling machinery
//! itself lives in cuteDslTensorDescriptors.h, which stays free of any FMHA concept.
namespace trt_edgellm
{

//! Everything the dense LLM descriptors need, gathered once per CuteDslFMHARunner::run() call.
struct LlmFmhaParams
{
    void const* qPtr{};
    void const* kvPtr{};
    void* oPtr{};
    int32_t const* cuKVSeqLens{};
    int32_t batchSize{};
    int32_t seqLenQ{};
    int32_t numQHeads{};
    int32_t numKVHeads{};
    int32_t headDim{};
    int32_t kvCacheCapacity{};
    int32_t windowSizeLeft{};
    float attentionScale{};
    float scaleQ{};
    float scaleK{};
    float scaleV{};
    float invScaleO{};
    //! Skip-softmax (BLASST) threshold as log2(lambda); 0.0 = disabled. Only the
    //! *_skipsoftmax launchers forward it — the other wrappers have no such argument.
    float skipSoftmaxThresholdLog2{};
    cudaStream_t stream{};
};

//! Everything the paged LLM descriptors need, gathered once per CuteDslFMHARunner::runPaged() call.
struct LlmFmhaPagedParams
{
    void const* qPtr{};
    void const* pagedKVPoolPtr{};
    int32_t const* kvCachePageList{};
    void* oPtr{};
    int32_t const* cuQSeqLens{};
    int32_t const* cuKVSeqLens{};
    int32_t const* bidirectionalBlockBegin{};
    int32_t const* bidirectionalBlockEnd{};
    int32_t batchSize{};
    int32_t seqLenQ{};
    int32_t numQHeads{};
    int32_t numKVHeads{};
    int32_t headDim{};
    int32_t numPages{};
    int32_t maxPagesPerSeq{};
    int32_t tokensPerPage{};
    int32_t windowSizeLeft{};
    float attentionScale{};
    float scaleQ{};
    float scaleK{};
    float scaleV{};
    float invScaleO{};
    //! Skip-softmax (BLASST) threshold as log2(lambda); 0.0 = disabled. Only the
    //! *_skipsoftmax_paged launchers forward it — the other wrappers have no such argument.
    float skipSoftmaxThresholdLog2{};
    cudaStream_t stream{};
};

//! Everything the ViT descriptors need, gathered once per CuteDslFMHARunner ViT run() call.
//! The ViT AOT variants are plain MHA, hence a single head count.
struct VitFmhaParams
{
    void const* qPtr{};
    void const* kPtr{};
    void const* vPtr{};
    void* oPtr{};
    int32_t const* cuSeqLens{};
    int32_t totalSeqLen{};
    int32_t numHeads{};
    int32_t headDim{};
    int32_t maxSeqLen{};
    int32_t batchSize{};
    float scaleSoftmaxLog2{};
    float attentionScale{};
    float scaleOutput{};
    cudaStream_t stream{};
};

//! Everything the FMHA-v2 dense LLM descriptors need, gathered once per CuteDslFMHAV2Runner::run() call.
struct FmhaV2LlmParams
{
    void const* qPtr{};
    void const* kPtr{};
    void const* vPtr{};
    void* oPtr{};
    int32_t const* cuKVSeqLens{};
    int32_t batchSize{};
    int32_t seqLenQ{};
    int32_t kvSeqLen{};
    int32_t numQHeads{};
    int32_t numKVHeads{};
    int32_t headDim{};
    int32_t windowSizeLeft{};
    float attentionScale{};
    float scaleQ{};
    float scaleK{};
    float scaleV{};
    float invScaleO{};
    cudaStream_t stream{};
};

//! Everything the FMHA-v2 ViT descriptors need, gathered once per CuteDslFMHAV2Runner ViT run()
//! call. Unlike VitFmhaParams these variants are GQA, so Q and KV head counts differ.
struct FmhaV2VitParams
{
    void const* qPtr{};
    void const* kPtr{};
    void const* vPtr{};
    void* oPtr{};
    int32_t const* cuSeqLens{};
    int32_t totalSeqLen{};
    int32_t numQHeads{};
    int32_t numKVHeads{};
    int32_t headDim{};
    int32_t maxSeqLen{};
    int32_t batchSize{};
    float scaleSoftmaxLog2{};
    float attentionScale{};
    float scaleOutput{};
    cudaStream_t stream{};
};

} // namespace trt_edgellm
