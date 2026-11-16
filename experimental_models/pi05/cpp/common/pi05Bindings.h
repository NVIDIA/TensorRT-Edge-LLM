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

#include "common/bindingNames.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace trt_edgellm
{
namespace pi05
{
//! Export contract this runtime accepts; must track ``CONTRACT_VERSION`` in
//! ``tensorrt_edgellm/models/pi05/export.py``.
inline constexpr int32_t kContractVersion = 1;

//! Identifies one export run. The root contract and every component config.json
//! carry it, and the builder refuses a set that disagrees.
inline constexpr char const* kExportIdKey = "export_id";

namespace binding_names
{

//! Visual component.
inline constexpr char const* kPixelValues = "pixel_values";
inline constexpr char const* kImageFeatures = "image_features";

//! Prefix component.
inline constexpr char const* kInputsEmbeds = "inputs_embeds";

//! Shared by the prefix and action components.
inline constexpr char const* kRopeCosSin = "rope_rotary_cos_sin";
inline constexpr char const* kAttentionPosId = "attention_pos_id";

//! Action component.
inline constexpr char const* kNoiseTrajectory = "noise_trajectory";
inline constexpr char const* kTimestep = "timestep";
inline constexpr char const* kActionPred = "action_pred";

//! Token-major AttentionPlugin metadata the action graph binds; the shared
//! contract owns the names, so they are pulled in rather than restated.
using trt_edgellm::binding_names::kAttentionSequenceLengths;
using trt_edgellm::binding_names::kContextSequenceCountCarrier;
using trt_edgellm::binding_names::kExecutionPhaseMarker;
using trt_edgellm::binding_names::kQueryLengths;
using trt_edgellm::binding_names::kQueryStartOffsets;

//! Paged-pool inputs; see cpp/common/bindingNames.h for the shared
//! AttentionPlugin's contract these follow.
inline constexpr char const* kKVCacheStartIndex = "kvcache_start_index";
inline constexpr char const* kKVPageTable = "kv_page_table";
inline constexpr char const* kAttentionMask = "attention_mask";

//! Cond component output, and the action input that replaces kTimestep when the
//! AdaRMS modulation is hoisted out of the per-step graph.
inline constexpr char const* kAdarmsModulation = "adarms_modulation";

//! Per-layer K/V produced by the prefix graph.
inline std::string formatPrefixKName(int32_t layerIdx)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "k_layer%02d", layerIdx);
    return buf;
}

inline std::string formatPrefixVName(int32_t layerIdx)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "v_layer%02d", layerIdx);
    return buf;
}

//! One packed cache per layer, which the prefix graph's K and V outputs are
//! bound into halves of and which the action graph appends its own tokens to.
inline std::string formatActionKVCacheName(int32_t layerIdx)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "kv_cache_layer%02d", layerIdx);
    return buf;
}

//! The same cache as an action-graph output; bound to the input's address so the
//! plugin's append stays in place while TensorRT still sees the write.
inline std::string formatActionPresentKVCacheName(int32_t layerIdx)
{
    char buf[40];
    std::snprintf(buf, sizeof(buf), "present_kv_cache_layer%02d", layerIdx);
    return buf;
}

} // namespace binding_names
} // namespace pi05
} // namespace trt_edgellm
