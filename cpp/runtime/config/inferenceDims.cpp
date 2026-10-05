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

#include "runtime/config/inferenceDims.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <sstream>

namespace trt_edgellm
{
namespace rt
{

std::string_view dimName(int64_t InferenceDims::* member)
{
    for (auto const& [ptr, name] : detail::kDimNames)
    {
        if (ptr == member)
        {
            return name;
        }
    }
    return {};
}

std::string toString(InferenceDims const& dims)
{
    std::ostringstream ss;
    ss << "{batch=" << dims.batch << ", seq_len=" << dims.seqLen << ", kv_len=" << dims.kvLen
       << ", select_len=" << dims.selectLen << ", attn_seq_len=" << dims.attnMaskSeqLen
       << ", rope_batch=" << dims.ropeBatch << ", packed_mask_len=" << dims.packedMaskLen
       << ", context_mask_selector_len=" << dims.contextMaskSelectorLen << ", start_index_len=" << dims.startIndexLen
       << ", execution_phase_len=" << dims.executionPhaseLen << ", skip_softmax_scale_len=" << dims.skipSoftmaxScaleLen
       << ", swa_kv_cache_mode_len=" << dims.swaKVCacheModeLen << ", query_offset_len=" << dims.queryOffsetLen
       << ", context_sequence_count=" << dims.contextSequenceCount << "}";
    return ss.str();
}

ExecutionPhase executionPhase(InferenceDims const& dims)
{
    ELLM_CHECK(isExecutionPhaseExtent(dims.executionPhaseLen), "Invalid execution phase extent");
    return static_cast<ExecutionPhase>(dims.executionPhaseLen);
}

int64_t InferenceDims::* firstInvalidMember(
    InferenceDims const& dims, std::vector<int64_t InferenceDims::*> const& referenced)
{
    for (auto member : referenced)
    {
        bool const zeroAllowed
            = std::find(detail::kZeroAllowedMembers.begin(), detail::kZeroAllowedMembers.end(), member)
            != detail::kZeroAllowedMembers.end();
        bool const invalid = zeroAllowed ? (dims.*member < 0) : (dims.*member <= 0);
        if (invalid)
        {
            return member;
        }
    }
    return nullptr;
}

} // namespace rt
} // namespace trt_edgellm
