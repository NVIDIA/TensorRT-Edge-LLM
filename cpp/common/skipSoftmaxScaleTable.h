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

namespace trt_edgellm
{
namespace rt
{

//! Element-count bound of the `skip_softmax_prefill_scale` and `skip_softmax_decode_scale` table bindings:
//! 128 flattened (seqlen_bin, S) pairs each. The exporters mirror it as SKIP_SOFTMAX_SCALE_TABLE_MAX_LEN.
constexpr int32_t kSKIP_SOFTMAX_SCALE_TABLE_MAX_LEN{256};

} // namespace rt
} // namespace trt_edgellm
