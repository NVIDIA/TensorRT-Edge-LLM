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

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{

//! \brief Noncausal block-sparse attention over one packed ``[prefix | video]`` BF16 sequence.
//!
//! Q, K, V, and the output are ``[sequenceLength, kNUM_HEADS, kHEAD_SIZE]``. Query rows of the
//! dense prefix (text and audio) attend exactly to every key. Each ``kBLOCK_SIZE``-row query block
//! after it attends exactly to the prefix blocks, its neighboring blocks, and the key blocks whose
//! centroid score exceeds its routing threshold; every other key block contributes through its mean
//! key and summed value. The AOT kernel fixes the head geometry.
class VideoSparseAttentionRunner
{
public:
    static constexpr int32_t kNUM_HEADS{56};
    static constexpr int32_t kHEAD_SIZE{128};
    static constexpr int32_t kBLOCK_SIZE{64};

    VideoSparseAttentionRunner() = delete;

    //! \brief Whether the linked CuTe DSL artifact has a kernel for this configuration.
    static bool canImplement(int32_t numHeads, int32_t headSize, int32_t smVersion) noexcept;

    //! \brief Workspace bytes for the block summaries and routing thresholds, with alignment slack.
    static size_t getWorkspaceSize(int32_t sequenceLength) noexcept;

    //! \param prefixLength Dense prefix rows; ``0 < prefixLength < sequenceLength``.
    //! \param workspace At least getWorkspaceSize(sequenceLength) bytes at any alignment.
    //! \param smVersion The device's SM version, which canImplement() accepted.
    static bool run(void const* q, void const* k, void const* v, void* output, int32_t sequenceLength,
        int32_t prefixLength, void* workspace, int32_t smVersion, cudaStream_t stream) noexcept;
};

namespace kernel
{

//! \brief Build per-block K/V summaries and per-block routing thresholds.
//!
//! KC and VC are BF16 ``[numBlocks, numHeads, kHEAD_SIZE]`` block means of K and block sums of V;
//! the KC statistics are FP32 ``[numHeads, kHEAD_SIZE]`` and the threshold is FP32
//! ``[numBlocks, numHeads]``. KC is rounded to BF16 before its statistics are evaluated, as in
//! the reference router.
cudaError_t launchVideoSparseAttentionPreprocess(void const* q, void const* k, void const* v, void* kc, void* vc,
    float* kcMean, float* kcVarDiag, float* threshold, int32_t sequenceLength, int32_t numHeads,
    cudaStream_t stream) noexcept;

} // namespace kernel

} // namespace trt_edgellm
