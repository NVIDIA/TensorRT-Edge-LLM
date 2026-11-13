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

#include "common/dataType.h"

#include <cstdint>
#include <string>

//! Whether a phase-aware layer runs the compact phase-8 step (prefill rows and decode rows in one launch).
//!
//! The plugins call these when they reserve the phase-8 workspace and when they accept or refuse a phase-8 enqueue;
//! the runtime calls them to decide, before any step runs, whether a deployment can serve compact steps. They are
//! compiled into both, under the same CUDA version and CuTe DSL kernel-group macros, so the two agree in a build.
//! Runtime conditions (EDGELLM_DISABLE_CTM_DECODE, the device's SM count) are read from the calling process.
//! The GDN and SSD predicates live in translation units of their own: the CuTe DSL runner headers of different kernel
//! groups declare conflicting cudaLibraryUnload shims on CUDA < 12.8.
namespace trt_edgellm
{

struct CompactPhaseSupport
{
    bool supported{false};
    //! What is missing, as a static string; nullptr when supported.
    char const* reason{nullptr};
};

//! SMs whose attention decodes on the CTM kernels only. Holds for an SM before and after thorRenumberedSm().
bool isCtmOnlySM(int32_t smVersion) noexcept;

//! Workaround for the CUDA 12/13 Thor SM renumbering: Thor (SM110) selects the SM101 attention kernels, which run on
//! it. Idempotent.
int32_t thorRenumberedSm(int32_t smVersion) noexcept;

//! Whether EDGELLM_DISABLE_CTM_DECODE turns CTM decode off on SM @p smVersion; CTM-only SMs ignore it. The variable is
//! read once per process.
bool ctmDecodeDisabledByEnv(int32_t smVersion) noexcept;

//! Whether the linked CuTe DSL artifact was built for SM @p smVersion (its metadata.json gpu_arch list). The
//! kernel-group macros are defined per artifact, not per SM, so a predicate whose kernels come from the artifact checks
//! this too.
bool cuteDslArtifactCarriesSm(int32_t smVersion) noexcept;

//! Whether attention on SM @p smVersion at tensor-parallel size @p tpSize decodes on the CTM kernels only.
bool decodesOnCtmOnly(int32_t smVersion, int32_t tpSize) noexcept;

//! The static configuration of one AttentionPlugin layer that decides its kernels.
struct AttentionLayerTraits
{
    int32_t numQHeads{0};  //!< Per rank
    int32_t numKVHeads{0}; //!< Per rank
    int32_t headSize{0};
    //! The device SM, before or after thorRenumberedSm().
    int32_t smVersion{0};
    int32_t tpSize{1};
    DataType dataType{DataType::kHALF};
    bool fp8KVCache{false};
    bool slidingWindow{false};
    bool supportsBoundedKVCache{false};
    bool kvShared{false};
    //! The packed QKV input carries Q only (a KV-shared layer without the donor's current K/V).
    bool qOnlyInput{false};
    bool treeAttention{false};
    bool visionBlockAttention{false};
    bool attentionSink{false};
    bool contextMaskSelector{false};
};

enum class CompactPrefillKernel : int32_t
{
    kNONE,
    kCUTE_DSL_FMHA_BLACKWELL,
    kCUTE_DSL_FMHA_V2,
};

//! Whether vanilla decode of @p layer can run on the CTM kernels.
bool canUseCtmDecode(AttentionLayerTraits const& layer) noexcept;

//! Whether an XQA decode kernel exists for @p layer.
bool canUseXqaDecode(AttentionLayerTraits const& layer) noexcept;

//! The ragged paged FMHA kernel the compact prefill rows of @p layer run on; kNONE when none covers it.
CompactPrefillKernel selectCompactPrefillKernel(AttentionLayerTraits const& layer) noexcept;

//! AttentionPlugin: phase 8 needs a layer kind the compact path covers, a decode kernel (CTM, or XQA off the CTM-only
//! SMs) and a ragged paged prefill kernel.
CompactPhaseSupport attentionCompactSupport(AttentionLayerTraits const& layer) noexcept;

//! The heads, head dim and KV-cache traits of @p layer, as the runtime's compact capability reason and the plugin's
//! phase-8 refusal both name it.
std::string attentionLayerShape(AttentionLayerTraits const& layer);

//! GatedDeltaNetPlugin: phase 8 needs no spec-verify or diffusion state and the packed-varlen GDN kernel.
CompactPhaseSupport gatedDeltaNetCompactSupport(
    int32_t kDim, int32_t vDim, int32_t smVersion, bool specVerifyState, bool diffusionState) noexcept;

//! MambaPlugin: phase 8 needs no spec-verify state and a packed SSD kernel (the Blackwell datacenter one, or the
//! SM120/121 state-pool one).
CompactPhaseSupport mamba2SsdCompactSupport(
    int32_t headDim, int32_t stateDim, int32_t smVersion, bool specVerifyState) noexcept;

//! CausalConv1dPlugin: phase 8 needs a depthwise layer (@p groups 0 or @p dim) without spec-verify state and the
//! packed-varlen conv kernel.
CompactPhaseSupport causalConv1dCompactSupport(
    int32_t smVersion, int32_t dim, int32_t width, int32_t dilation, int32_t groups, bool specVerifyState) noexcept;

} // namespace trt_edgellm
