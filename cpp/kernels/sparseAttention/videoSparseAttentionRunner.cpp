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

#include "videoSparseAttentionRunner.h"

#include "common/logger.h"

#if defined(CUTE_DSL_VIDEO_SPARSE_ATTENTION_ENABLED)
#include "kernels/cuteDslModuleLoader.h"

#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#if __has_include("video_sparse_attention_sm100.h")
#define VIDEO_SPARSE_ATTENTION_SM100 1
#include "video_sparse_attention_sm100.h"
#endif
#if __has_include("video_sparse_attention_sm120.h")
#define VIDEO_SPARSE_ATTENTION_SM120 1
#include "video_sparse_attention_sm120.h"
#endif
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif // defined(CUTE_DSL_VIDEO_SPARSE_ATTENTION_ENABLED)

#include <cstdint>

namespace trt_edgellm
{

namespace
{

constexpr size_t kWORKSPACE_ALIGNMENT{256};

size_t alignUp(size_t value) noexcept
{
    return (value + kWORKSPACE_ALIGNMENT - 1) / kWORKSPACE_ALIGNMENT * kWORKSPACE_ALIGNMENT;
}

int32_t numBlocks(int32_t sequenceLength) noexcept
{
    return (sequenceLength + VideoSparseAttentionRunner::kBLOCK_SIZE - 1) / VideoSparseAttentionRunner::kBLOCK_SIZE;
}

//! Byte offsets of KC, VC, the KC mean and variance, and the thresholds.
struct WorkspaceLayout
{
    size_t vc{};
    size_t kcMean{};
    size_t kcVarDiag{};
    size_t threshold{};
    size_t total{};

    explicit WorkspaceLayout(int32_t sequenceLength) noexcept
    {
        size_t const blocks = numBlocks(sequenceLength);
        size_t const rowElements
            = static_cast<size_t>(VideoSparseAttentionRunner::kNUM_HEADS) * VideoSparseAttentionRunner::kHEAD_SIZE;
        size_t const summaryBytes = alignUp(blocks * rowElements * sizeof(uint16_t));
        size_t const statisticBytes = alignUp(rowElements * sizeof(float));
        vc = summaryBytes;
        kcMean = vc + summaryBytes;
        kcVarDiag = kcMean + statisticBytes;
        threshold = kcVarDiag + statisticBytes;
        total = threshold + alignUp(blocks * VideoSparseAttentionRunner::kNUM_HEADS * sizeof(float));
    }
};

#if defined(VIDEO_SPARSE_ATTENTION_SM100)
detail::LazyKernelModule<video_sparse_attention_sm100_Kernel_Module_t> sSm100Module{};
#endif
#if defined(VIDEO_SPARSE_ATTENTION_SM120)
detail::LazyKernelModule<video_sparse_attention_sm120_Kernel_Module_t> sSm120Module{};
#endif

#if defined(CUTE_DSL_VIDEO_SPARSE_ATTENTION_ENABLED)
template <typename Module, typename Wrapper, typename... Tensors>
int32_t launch(Module& module, Wrapper wrapper, int32_t prefixLength, cudaStream_t stream, Tensors... tensors)
{
    return wrapper(&module, &tensors..., prefixLength, stream);
}

//! The artifact tensors are ``[1, rows, ...]``; the dynamic stride is that of the unit batch.
template <typename Tensor>
Tensor bindRows(void const* data, int32_t rows, int64_t batchStride)
{
    Tensor tensor{};
    tensor.data = const_cast<void*>(data);
    tensor.dynamic_shapes[0] = rows;
    tensor.dynamic_strides[0] = batchStride;
    return tensor;
}
#endif

} // namespace

bool VideoSparseAttentionRunner::canImplement(int32_t numHeads, int32_t headSize, int32_t smVersion) noexcept
{
    if (numHeads != kNUM_HEADS || headSize != kHEAD_SIZE)
    {
        return false;
    }
    switch (smVersion)
    {
#if defined(VIDEO_SPARSE_ATTENTION_SM100)
    case 100:
    case 101:
    case 103:
    case 110: return true;
#endif
#if defined(VIDEO_SPARSE_ATTENTION_SM120)
    case 120:
    case 121: return true;
#endif
    default: return false;
    }
}

size_t VideoSparseAttentionRunner::getWorkspaceSize(int32_t sequenceLength) noexcept
{
    // TensorRT does not guarantee the kernels' alignment for plugin workspace.
    return WorkspaceLayout(sequenceLength).total + kWORKSPACE_ALIGNMENT - 1;
}

bool VideoSparseAttentionRunner::run(void const* q, void const* k, void const* v, void* output, int32_t sequenceLength,
    int32_t prefixLength, void* workspace, int32_t smVersion, cudaStream_t stream) noexcept
{
#if defined(CUTE_DSL_VIDEO_SPARSE_ATTENTION_ENABLED)
    if (prefixLength <= 0 || prefixLength >= sequenceLength)
    {
        LOG_ERROR("Video sparse attention needs a prefix inside the sequence: prefix %d, sequence %d", prefixLength,
            sequenceLength);
        return false;
    }
    WorkspaceLayout const layout(sequenceLength);
    auto* const base = reinterpret_cast<uint8_t*>(alignUp(reinterpret_cast<uintptr_t>(workspace)));
    void* const kc = base;
    void* const vc = base + layout.vc;
    auto* const threshold = reinterpret_cast<float*>(base + layout.threshold);
    cudaError_t const status
        = kernel::launchVideoSparseAttentionPreprocess(q, k, v, kc, vc, reinterpret_cast<float*>(base + layout.kcMean),
            reinterpret_cast<float*>(base + layout.kcVarDiag), threshold, sequenceLength, kNUM_HEADS, stream);
    if (status != cudaSuccess)
    {
        LOG_ERROR("Video sparse attention preprocessing failed: %s", cudaGetErrorString(status));
        return false;
    }

    int32_t const blocks = numBlocks(sequenceLength);
    int64_t const rowElements = static_cast<int64_t>(kNUM_HEADS) * kHEAD_SIZE;
    // A build for several GPUs links both artifacts; the SM 12x family has its own kernel.
    bool const sm120Family = smVersion / 10 == 12;
    int32_t result{-1};
#if defined(VIDEO_SPARSE_ATTENTION_SM100)
    if (!sm120Family
        && detail::ensureModuleLoaded<video_sparse_attention_sm100_Kernel_Module_Load,
            video_sparse_attention_sm100_Kernel_Module_Unload>(sSm100Module, "video_sparse_attention_sm100", stream))
    {
        result = launch(sSm100Module.module, cute_dsl_video_sparse_attention_sm100_wrapper, prefixLength, stream,
            bindRows<video_sparse_attention_sm100_Tensor_q_t>(q, sequenceLength, rowElements * sequenceLength),
            bindRows<video_sparse_attention_sm100_Tensor_k_t>(k, sequenceLength, rowElements * sequenceLength),
            bindRows<video_sparse_attention_sm100_Tensor_v_t>(v, sequenceLength, rowElements * sequenceLength),
            bindRows<video_sparse_attention_sm100_Tensor_output_t>(
                output, sequenceLength, rowElements * sequenceLength),
            bindRows<video_sparse_attention_sm100_Tensor_kc_t>(kc, blocks, rowElements * blocks),
            bindRows<video_sparse_attention_sm100_Tensor_vc_t>(vc, blocks, rowElements * blocks),
            bindRows<video_sparse_attention_sm100_Tensor_threshold_t>(threshold, blocks, int64_t{kNUM_HEADS} * blocks));
    }
#endif
#if defined(VIDEO_SPARSE_ATTENTION_SM120)
    if (sm120Family
        && detail::ensureModuleLoaded<video_sparse_attention_sm120_Kernel_Module_Load,
            video_sparse_attention_sm120_Kernel_Module_Unload>(sSm120Module, "video_sparse_attention_sm120", stream))
    {
        result = launch(sSm120Module.module, cute_dsl_video_sparse_attention_sm120_wrapper, prefixLength, stream,
            bindRows<video_sparse_attention_sm120_Tensor_q_t>(q, sequenceLength, rowElements * sequenceLength),
            bindRows<video_sparse_attention_sm120_Tensor_k_t>(k, sequenceLength, rowElements * sequenceLength),
            bindRows<video_sparse_attention_sm120_Tensor_v_t>(v, sequenceLength, rowElements * sequenceLength),
            bindRows<video_sparse_attention_sm120_Tensor_output_t>(
                output, sequenceLength, rowElements * sequenceLength),
            bindRows<video_sparse_attention_sm120_Tensor_kc_t>(kc, blocks, rowElements * blocks),
            bindRows<video_sparse_attention_sm120_Tensor_vc_t>(vc, blocks, rowElements * blocks),
            bindRows<video_sparse_attention_sm120_Tensor_threshold_t>(threshold, blocks, int64_t{kNUM_HEADS} * blocks));
    }
#endif
    static_cast<void>(sm120Family);
    if (result != 0)
    {
        LOG_ERROR("Video sparse attention kernel failed with status %d", result);
        return false;
    }
    return true;
#else
    static_cast<void>(q);
    static_cast<void>(k);
    static_cast<void>(v);
    static_cast<void>(output);
    static_cast<void>(sequenceLength);
    static_cast<void>(prefixLength);
    static_cast<void>(workspace);
    static_cast<void>(smVersion);
    static_cast<void>(stream);
    return false;
#endif
}

} // namespace trt_edgellm
